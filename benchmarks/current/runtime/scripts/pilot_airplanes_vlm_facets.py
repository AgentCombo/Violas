#!/usr/bin/env python3
"""Compare direct visual facet assignment to the frozen Caltech prototype labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

import diversity_preprocess as preprocess


ENTITY = "airplanes"
SOURCE = Path("diversity/v2_caltech99_kv_schemas")
ROOT = Path("dataset/caltech-101/101_ObjectCategories")
CACHE = Path("cache/caltech_101_ObjectCategories_ts0.1_rs42_a0.500_folder.pkl")
OUT = Path("artifacts/airplanes_qwen_facet_pilot")
MODEL = "qwen3.5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def prepare():
    schema = preprocess.load_schemas(SOURCE / "schemas.jsonl")[ENTITY]
    old = {row["object_id"]: row for row in preprocess.read_jsonl(SOURCE / "assignments.jsonl")
           if row["entity"] == ENTITY}
    vectors = preprocess.load_cache_vectors(CACHE, ROOT.resolve())[ENTITY]
    ids = sorted((old.keys() & vectors.keys()) - set(schema["sample_relative_paths"]))
    matrix = np.stack([np.asarray(vectors[oid], dtype=np.float32).reshape(-1) for oid in ids])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine = matrix @ matrix.T
    mask = np.triu(np.ones_like(cosine, dtype=bool), k=1) & (cosine < 0.9999)
    samples, pairs = [], []
    used = set()

    def add(index, reason):
        oid = ids[index]
        tag = f"P{len(samples) + 1:02d}"
        image = OUT / "images" / f"{tag}.jpg"
        with Image.open(ROOT / oid) as source:
            thumb = ImageOps.exif_transpose(source).convert("RGB")
            thumb.thumbnail((768, 768), Image.Resampling.LANCZOS)
            thumb.save(image, quality=93)
        samples.append({"id": tag, "object_id": oid, "image": image.relative_to(OUT).as_posix(),
                        "selection": reason, "prototype": old[oid]["attributes"]})
        used.add(index)
        return tag

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    facets = [facet["id"] for facet in schema["facets"]]
    for facet in facets:
        labels = np.array([old[oid]["attributes"][facet]["value"] for oid in ids])
        uncertain = np.array([old[oid]["attributes"][facet]["uncertain"] for oid in ids])
        same = labels[:, None] == labels[None, :]
        for kind, condition in (("different prototype values", ~same),
                                ("same prototype value; at least one uncertain", same & (uncertain[:, None] | uncertain[None, :]))):
            free = np.array([i not in used for i in range(len(ids))])
            candidates = mask & condition & free[:, None] & free[None, :]
            if not candidates.any():
                raise ValueError(f"no usable pair for {facet}: {kind}")
            i, j = np.unravel_index(np.argmax(np.where(candidates, cosine, -np.inf)), cosine.shape)
            reason = f"{facet}: {kind}"
            pairs.append({"samples": [add(int(i), reason), add(int(j), reason)], "facet": facet,
                          "kind": kind, "clip_cosine": float(cosine[i, j])})

    for facet in (*facets, "all"):
        keys = facets if facet == "all" else (facet,)
        index = min((i for i in range(len(ids)) if i not in used),
                    key=lambda i: min(old[ids[i]]["attributes"][key]["margin"] for key in keys))
        add(index, f"low prototype margin: {facet}")

    rng = np.random.default_rng(42)
    for value in ("propeller", "jet"):
        controls = [i for i, oid in enumerate(ids) if i not in used
                    and old[oid]["attributes"]["propulsion_type"]["value"] == value
                    and all(not item["uncertain"] for item in old[oid]["attributes"].values())]
        for i in rng.choice(controls, size=2, replace=False):
            add(int(i), "high-margin control")
    preprocess.make_contact_sheet([OUT / row["image"] for row in samples], [row["id"] for row in samples],
                                  OUT / "contact_sheet.jpg", tile_size=235, columns=4)
    write(OUT / "sample.json", {"schema": schema, "source_images": len(old), "pairs": pairs, "samples": samples})
    print(f"PREPARED {len(samples)} images; {len(pairs)} close CLIP pairs", flush=True)


def assignment_prompt(schema):
    facets = [{"id": facet["id"], "description": facet["description"],
               "values": {value["id"]: value["description"] for value in facet["values"]}}
              for facet in schema["facets"]]
    return ("Assign visual facet values to the main aircraft in this photograph using this fixed schema. "
            "Use visible evidence, such as propeller blades, jet nacelles, wheels touching the ground, "
            "visible separation from the ground, a military roundel or airline livery. Landing gear down or a sky "
            "background alone is insufficient for flight state. Use unknown when the evidence is ambiguous or occluded. "
            "Return only a JSON object with exactly the three facet IDs. For each ID, return an object with exactly "
            "value (one schema value or unknown) and evidence (one short English sentence describing a visible cue "
            "or why it is unclear). No other fields.\nSchema:\n" + json.dumps(facets, ensure_ascii=False))


def validate(value, schema):
    allowed = {facet["id"]: {option["id"] for option in facet["values"]} | {"unknown"}
               for facet in schema["facets"]}
    if not isinstance(value, dict) or set(value) != set(allowed):
        raise ValueError("unexpected facet keys")
    for key, labels in allowed.items():
        entry = value[key]
        if (not isinstance(entry, dict) or set(entry) != {"value", "evidence"}
                or not isinstance(entry["value"], str) or entry["value"] not in labels
                or not isinstance(entry["evidence"], str) or not entry["evidence"].strip()):
            raise ValueError(f"invalid assignment for {key}")
    return value


def classify(bundle, jobs, short_response=False):
    key = preprocess.resolve_api_key("VIOLAS_VLM_API_KEY")
    if not key:
        raise RuntimeError("CSTCloud API key is unavailable")
    request_options = {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1024} if short_response else {}
    client = preprocess.VisionApiClient(key, "https://uni-api.cstcloud.cn/v1", MODEL,
                                        timeout=55 if short_response else 120, retries=0 if short_response else 1)
    prompt = assignment_prompt(bundle["schema"])
    path = OUT / "vlm_assignments.json"
    assigned = read(path) if path.exists() else {}

    def one(sample):
        start = time.perf_counter()
        image = OUT / sample["image"]
        attempts = 1 if short_response else 2
        for attempt in range(attempts):
            try:
                value, meta = client.discover(ENTITY, [image], prompt, request_options=request_options)
                labels = validate(value, bundle["schema"])
            except (TimeoutError, ValueError):
                if attempt + 1 < attempts:
                    continue
                raise
            return {"object_id": sample["object_id"], "attributes": labels, "usage": meta["response"].get("usage"),
                    "seconds": round(time.perf_counter() - start, 3), "request_options": request_options}

    failures = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(one, sample): sample["id"] for sample in bundle["samples"] if sample["id"] not in assigned}
        for future in as_completed(futures):
            tag = futures[future]
            try:
                assigned[tag] = future.result()
            except Exception as exc:
                failures[tag] = str(exc).replace(key, "[redacted]")
                print(f"FAILED {tag}: {failures[tag]}", flush=True)
                continue
            write(path, assigned)
            print(f"ASSIGNED {tag} {len(assigned)}/{len(bundle['samples'])} {assigned[tag]['seconds']}s", flush=True)
    write(OUT / "errors.json", failures)
    return not failures


def render(bundle):
    path = OUT / "vlm_assignments.json"
    assigned = read(path) if path.exists() else {}
    ref_path = OUT / "visual_reference.json"
    reference = read(ref_path) if ref_path.exists() else {}
    stats = {facet["id"]: Counter() for facet in bundle["schema"]["facets"]}
    cards = []
    for sample in bundle["samples"]:
        tag = sample["id"]
        old = sample["prototype"]
        new = assigned.get(tag, {}).get("attributes", {})
        tr = []
        for facet, count in stats.items():
            before = old[facet]
            after = new.get(facet, {})
            result = after.get("value", "pending")
            expected = reference.get(tag, {}).get(facet, "")
            if after:
                count["completed"] += 1
                count["prototype_uncertain"] += before["uncertain"]
                count["qwen_unknown"] += result == "unknown"
                count["different_label"] += result not in ("unknown", before["value"])
                if expected:
                    count["visual_reference"] += 1
                    count["prototype_match"] += before["value"] == expected
                    count["qwen_match"] += result == expected
            flag = " class='changed'" if result not in ("pending", before["value"]) else ""
            tr.append(f"<tr{flag}><td>{escape(facet)}</td><td>{escape(before['value'])} · Δ {before['margin']:.3f}</td>"
                      f"<td>{escape(result)}</td><td>{escape(after.get('evidence', ''))}</td><td>{escape(expected)}</td></tr>")
        retry = " · 补跑参数 enable_thinking=false，max_tokens=1024" if assigned.get(tag, {}).get("request_options") else ""
        cards.append(f"<article id='{tag}'><img src='{sample['image']}' alt='{tag}'><div><h2>{tag} · "
                     f"{escape(Path(sample['object_id']).name)}</h2><p>{escape(sample['selection'])}{retry}</p>"
                     "<table><tr><th>Facet</th><th>CLIP 原型 · margin</th><th>Qwen</th><th>Qwen 证据</th><th>目视参考</th></tr>"
                     + "".join(tr) + "</table></div></article>")
    pair_text = "".join(f"<li><a href='#{p['samples'][0]}'>{p['samples'][0]}</a> / "
                        f"<a href='#{p['samples'][1]}'>{p['samples'][1]}</a> · {escape(p['facet'])} · "
                        f"CLIP cosine {p['clip_cosine']:.3f}</li>" for p in bundle["pairs"])
    summary = "".join(f"<tr><td>{escape(f)}</td><td>{s['completed']}</td><td>{s['prototype_uncertain']}</td>"
                      f"<td>{s['different_label']}</td><td>{s['qwen_unknown']}</td><td>{s['visual_reference']}</td>"
                      f"<td>{s['prototype_match']}</td><td>{s['qwen_match']}</td></tr>" for f, s in stats.items())
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>飞机 facet 逐图标注试验</title><style>body{{max-width:1220px;margin:28px auto;padding:0 16px;background:#f5f7fb;color:#182435;font:15px system-ui,sans-serif}}
article{{display:flex;gap:18px;margin:12px 0;padding:14px;background:white;border:1px solid #dce4ed;border-radius:10px}}article img{{width:240px;max-height:210px;object-fit:contain}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #e4eaf1}}.changed{{background:#fff0d4}}h2{{margin-top:0}}
@media(max-width:800px){{article{{display:block}}table{{font-size:12px}}}}</style>
<h1>飞机 facet 逐图标注试验</h1><p>冻结当前报告的 schema，Qwen 仅看到单张图片和定义。定向样本用于观察边界差异。目视参考在读取 Qwen 答案之前记录，只覆盖能清楚判断的格子。</p>
<p><a href='contact_sheet.jpg'>样本总图</a> · <a href='sample.json'>采样记录</a> · <a href='vlm_assignments.json'>Qwen 输出</a></p>
<table><tr><th>Facet</th><th>完成</th><th>原型低 margin</th><th>Qwen 改类</th><th>Qwen unknown</th><th>目视可判</th><th>原型与目视一致</th><th>Qwen 与目视一致</th></tr>{summary}</table>
<h2>CLIP 近邻图对</h2><ul>{pair_text}</ul>{''.join(cards)}</html>"""
    (OUT / "audit.html").write_text(html, encoding="utf-8")
    write(OUT / "summary.json", stats)
    print(f"RENDERED {len(assigned)}/{len(bundle['samples'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "classify", "render"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--short-response", action="store_true", help="Bound generation when resuming a slow request")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        bundle = read(OUT / "sample.json")
        complete = args.stage != "classify" or classify(bundle, args.jobs, args.short_response)
        render(bundle)
        if not complete:
            raise SystemExit("some assignments failed; classify resumes the missing samples")


if __name__ == "__main__":
    main()
