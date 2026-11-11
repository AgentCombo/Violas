#!/usr/bin/env python3
"""Run the frozen Caltech report when the Qwen-50 label checkpoint is complete."""
from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
from pathlib import Path
from statistics import fmean
import subprocess
import sys
import time

import run_fancy_experiments as fancy
import run_soft_entity_bench as soft
from rescore_vdb_coverage import score


OUT = Path("artifacts/caltech_qwen50")
DATASET = "caltech101"
OLD = Path("artifacts/six_dataset_soft_entity_online_cosine/caltech101/results.json")
OLD_CACHE = Path("cache/caltech101_kv_train_v3_hdmg.pkl")
SOURCE = Path("cache/caltech_101_ObjectCategories_ts0.1_rs42_a0.500_folder.pkl")
LABELS = {"vg_hdmg": "VG+HDMG", "vg_rep": "VG−HDMG", "milvus_ann": "Milvus ANN", "milvus_key": "Milvus key",
          "qdrant_ann": "Qdrant ANN", "qdrant_key": "Qdrant key", "chroma_ann": "Chroma ANN", "chroma_key": "Chroma key"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(script, *args):
    subprocess.run([sys.executable, "-u", f"scripts/{script}", *map(str, args)], check=True)


def wait_for_labels():
    completion = OUT / "complete.json"
    while not completion.exists():
        pid = int((OUT / "run.pid").read_text())
        proc = Path("/proc") / str(pid)
        if not proc.exists() or b"label_caltech_qwen50.py" not in (proc / "cmdline").read_bytes():
            if completion.exists():
                break
            raise RuntimeError("label job stopped before completion; inspect run.log and resume labeling")
        time.sleep(20)
    return read(completion)


def prepare_labels():
    merged = OUT / "assignments_hybrid.jsonl"
    target = OUT / "assignments_for_index.jsonl"
    with merged.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            if row["source"] == "qwen3.5_visual_no_thinking":
                row["attributes"] = {key: value for key, value in row["attributes"].items()
                                     if value["value"] != "unknown"}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def build_vdb(cache, tag):
    objects = soft.bench.retrieval.make_object_map(soft.bench.load_pickle(cache, []))
    fancy.build_scalar_milvus(Path(f"cache/five_system/{tag}_milvus_array.db"), f"{tag}_array", objects, 128)
    soft.backends.build_chroma(Path(f"cache/five_system/{tag}_chroma"), f"{tag}_chroma", objects)


def old_and_new(old, new, categories):
    compare = []
    methods = [f"exp{exp}_{method}" for exp in (1, 2) for method in LABELS]
    for a, b in zip(old["rows"], new["rows"], strict=True):
        plan = a["resolved_plan"]
        if any(a[key] != b[key] for key in ("id", "query_object", "resolved_plan")):
            raise ValueError(f"query drift {a['id']}")
        compare.append({"id": a["id"], "query_type": a["query_type"],
                        "qwen_entity": a["entity_hint"] in categories,
                        "has_diversity": bool(plan["category_preferences"]),
                        "eligible_before": a["eligible_count"], "eligible_after": b["eligible_count"],
                        "methods": {key: {"before": score(a["methods"][key], plan),
                                          "after": score(b["methods"][key], plan)} for key in methods}})
    return compare


def render(old, new, rows):
    def average(values):
        valid = [value for value in values if value is not None]
        return f"{fmean(valid):.4f}" if valid else "—"

    no_match = [row["id"] for row in rows if row["eligible_after"] == 0]
    blocks = [f"<p>新标签下全库无合格对象的查询共 {len(no_match)} 条。"
              + (f"查询 ID 为 {escape(', '.join(no_match))}。" if no_match else "") + "</p>"]
    for exp in (1, 2):
        blocks.append(f"<h2>实验 {exp}</h2>")
        for label, subset in (("全部固定查询", rows),
                              ("首选类别已标注", [row for row in rows if row["qwen_entity"]]),
                              ("有多样性要求", [row for row in rows if row["has_diversity"]])):
            cells = []
            for method, name in LABELS.items():
                key = f"exp{exp}_{method}"
                values = [row["methods"][key] for row in subset]
                paired = [row for row in values if all(row[side]["objective_ratio_old"] is not None
                                                       for side in ("before", "after"))]
                pairs = [" → ".join(average([row[side]["objective_ratio_old"] for row in paired])
                                    for side in ("before", "after"))]
                pairs.extend(" → ".join(average([row[side][metric] for row in values]) for side in ("before", "after"))
                             for metric in ("objective", "relevance", "coverage"))
                latency = " → ".join(f"{report['aggregate']['overall'][key]['latency']['p50_ms']:.3f}"
                                     for report in (old, new)) if len(subset) == len(rows) else "—"
                cells.append(f"<tr><th scope='row'>{escape(name)}</th>" + "".join(f"<td>{item}</td>" for item in (*pairs, latency)) + "</tr>")
            blocks.append(f"<h3>{label} · {len(subset)} 条</h3><table><thead><tr><th>方法</th><th>原 objective ratio</th>"
                          "<th>覆盖加权分</th><th>相关性</th><th>Facet 覆盖率</th><th>p50（ms）</th></tr></thead><tbody>"
                          + "".join(cells) + "</tbody></table>")
    return """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Caltech · Qwen 50 类重标注对照</title>
<style>body{max-width:1150px;margin:32px auto;padding:0 16px;background:#f4f7fc;color:#1c2b40;font:15px system-ui,sans-serif}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dde7f4;text-align:right}th:first-child{text-align:left}thead{background:#e6eefb}h2{margin-top:35px}</style>
<h1>Caltech · Qwen 50 类重标注对照</h1>
<p>每格为原标签 → Qwen50 混合标签。两边使用相同的 gallery 对象、HDMG 拓扑、查询、intent、检索和 SELECT 配置。VDB 候选上限为 80。</p>
<p>原 objective ratio 各自使用对应标签下按原公式求解的全库 Oracle，表中只平均前后都有定义的查询。覆盖加权分为绝对分，有 diversity 时权重为 70% 相关性与 30% facet 覆盖率。两批时延分别测量。</p>
<p><a href='benchmark/caltech101/results.json'>检索原始结果</a> · <a href='score_comparison.json'>逐条分数对照</a> · <a href='manifest.json'>标注批次</a></p>
""" + "\n".join(blocks) + "</html>"


def main():
    print("WAITING FOR LABELS", flush=True)
    complete = wait_for_labels()
    manifest = read(OUT / "manifest.json")
    if complete["selected_images"] != manifest["images"] or complete["hybrid_images"] != 7807:
        raise ValueError("incomplete hybrid assignments")
    labels = prepare_labels()
    digest = hashlib.sha256(labels.read_bytes()).hexdigest()[:10]
    tag = f"caltech101_qwen50_{digest}"
    cache = Path(f"cache/{tag}_hdmg.pkl")
    directory = Path(f"cache/{tag}_directory.pkl")
    print(f"BUILDING {tag}", flush=True)
    run("build_multidataset_kv_cache.py", "--dataset", DATASET, "--source-cache", SOURCE,
        "--schemas", OUT / "schemas.jsonl", "--assignments", labels, "--output-cache", cache,
        "--directory-cache", directory, "--embedding-k", 16, "--semantic-intra-k", 8)
    original = soft.bench.load_pickle(OLD_CACHE, [])
    updated = soft.bench.load_pickle(cache, [])
    orig_objects = soft.bench.retrieval.make_object_map(original)
    objects = soft.bench.retrieval.make_object_map(updated)
    if orig_objects.keys() != objects.keys() or len(objects) != 6984:
        raise ValueError("hybrid gallery differs from original object universe")
    for attr in ("_hdmg_embedding_edges", "_hdmg_semantic_edges", "_hdmg_key_entries"):
        if getattr(original["vectormap"], attr) != getattr(updated["vectormap"], attr):
            raise ValueError(f"unexpected topology drift: {attr}")
    # Closing a Milvus client can leave the embedded server holding the file lock
    # until process exit. Finish index creation before launching the reader.
    run("run_caltech_qwen50_bench.py", "--build-vdb", cache, tag)
    run("run_soft_entity_bench.py", "--dataset", DATASET, "--source-cache", SOURCE,
        "--hdmg-cache", cache, "--directory-cache", directory, "--candidate-k", 80,
        "--index-tag", tag, "--frozen-report", OLD, "--output-root", OUT / "benchmark")
    old, new = read(OLD), read(OUT / "benchmark" / DATASET / "results.json")
    rows = old_and_new(old, new, set(manifest["categories"]))
    (OUT / "score_comparison.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "benchmark_summary.html").write_text(render(old, new, rows), encoding="utf-8")
    print(f"BENCHMARK_COMPLETE {len(rows)} queries {tag}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-vdb", nargs=2, metavar=("CACHE", "TAG"))
    args = parser.parse_args()
    if args.build_vdb:
        cache, tag = args.build_vdb
        build_vdb(Path(cache), tag)
    else:
        main()
