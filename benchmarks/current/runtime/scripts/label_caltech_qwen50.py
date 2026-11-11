#!/usr/bin/env python3
"""Label 50 frozen Caltech categories with Qwen and write a hybrid assignment file."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import random
import shutil
from tempfile import NamedTemporaryFile
import time

from PIL import Image, ImageOps

from pilot_airplanes_vlm_facets import MODEL, ROOT, SOURCE, preprocess, validate, write


OUT = Path("artifacts/caltech_qwen50")
OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1024,
           "response_format": {"type": "json_object"}}


def assignment_prompt(schema):
    facets = [{"id": facet["id"], "description": facet["description"],
               "values": {value["id"]: value["description"] for value in facet["values"]}}
              for facet in schema["facets"]]
    template = {facet["id"]: {"value": "unknown", "evidence": "Describe the visible cue."} for facet in facets}
    return (f"Assign the fixed visual facet values for the main {schema['entity']} in the photograph. "
            "Use visible evidence for each attribute, and use unknown when you cannot identify a reliable cue. "
            "Return only a JSON object with exactly the facet IDs listed below. For each facet return exactly "
            "value (one listed value or unknown) and evidence (one short English sentence about the visible cue "
            "or what is unclear). No other fields.\nUse exactly this JSON structure and fill each value: "
            + json.dumps(template, ensure_ascii=False) + "\nAllowed labels and definitions:\n"
            + json.dumps(facets, ensure_ascii=False))


@contextmanager
def input_image(path):
    with Image.open(path) as source, NamedTemporaryFile(suffix=".jpg") as tmp:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        image.save(tmp.name, quality=93)
        yield Path(tmp.name)


def load_results(path, rows):
    if not path.exists():
        return {}
    # An interrupted process can leave only the last line incomplete.
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
        path.write_bytes(data)
    results = {}
    for line in data.splitlines():
        result = json.loads(line)
        oid = result["object_id"]
        if oid not in rows or result["entity"] != rows[oid]["entity"]:
            raise ValueError(f"result does not match the selected objects: {oid}")
        results[oid] = result
    return results


def assign(row, prompt, schema, client):
    oid = row["object_id"]
    start = time.perf_counter()
    with input_image(ROOT / oid) as path:
        for attempt in range(2):
            try:
                answer, meta = client.discover(row["entity"], [path], prompt, request_options=OPTIONS)
                attrs = validate(answer, schema)
                break
            except (TimeoutError, RuntimeError, ValueError):
                if attempt:
                    raise
                time.sleep(2)
    return {"object_id": oid, "entity": row["entity"], "attributes": attrs,
            "schema_version": schema["schema_version"],
            "seconds": round(time.perf_counter() - start, 3), "usage": meta["response"].get("usage")}


def run_batch(rows, schemas, jobs, done, total, out):
    prompts = {entity: assignment_prompt(schema) for entity, schema in schemas.items()}
    key = preprocess.resolve_api_key("VIOLAS_VLM_API_KEY")
    if not key:
        raise RuntimeError("CSTCloud API key is unavailable")
    client = preprocess.VisionApiClient(key, "https://uni-api.cstcloud.cn/v1", MODEL, timeout=60, retries=0)
    failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(assign, row, prompts[row["entity"]], schemas[row["entity"]], client): row
                   for row in rows}
        with (out / "qwen_assignments.jsonl").open("a", encoding="utf-8") as output, \
                (out / "errors.jsonl").open("a", encoding="utf-8") as errors:
            for future in as_completed(futures):
                row = futures[future]
                oid = row["object_id"]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    errors.write(json.dumps({"object_id": oid, "error": str(exc).replace(key, "[redacted]")}) + "\n")
                    errors.flush()
                    continue
                done[oid] = result
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                if len(done) <= jobs or len(done) % 20 == 0 or len(done) == total:
                    print(f"LABELED {len(done)}/{total} {oid} {result['seconds']}s", flush=True)
    return failed


def merge(all_rows, results, output):
    pending = output.with_suffix(".pending.jsonl")
    with pending.open("w", encoding="utf-8") as stream:
        for row in all_rows:
            result = results.get(row["object_id"])
            if result:
                attributes = result["attributes"]
                row = {**row, "attributes": {facet: {**label, "uncertain": label["value"] == "unknown"}
                                             for facet, label in attributes.items()},
                       "tokens": [f"{facet}={label['value']}" for facet, label in attributes.items()
                                  if label["value"] != "unknown"],
                       "uncertain": any(label["value"] == "unknown" for label in attributes.values()),
                       "schema_version": result.get("schema_version", row["schema_version"]),
                       "source": "qwen3.5_visual_no_thinking"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending.replace(output)


def label_rows(selected, schemas, jobs, out):
    done = load_results(out / "qwen_assignments.jsonl", selected)
    pending = [row for oid, row in sorted(selected.items()) if oid not in done]
    print(f"START categories={len({row['entity'] for row in selected.values()})} images={len(selected)} resumed={len(done)} workers={jobs}", flush=True)
    if pending and len(done) < jobs:
        probe = pending[:jobs]
        if run_batch(probe, schemas, jobs, done, len(selected), out) >= len(probe) / 2:
            raise RuntimeError("too many failures in the initial API checks; inspect errors.jsonl")
    for attempt in range(4):
        pending = [row for row in pending if row["object_id"] not in done]
        if not pending:
            break
        if attempt:
            time.sleep(15 * attempt)
        failures = 0
        for start in range(0, len(pending), 64):
            batch = pending[start:start + 64]
            missed = run_batch(batch, schemas, jobs if attempt == 0 else min(jobs, 4), done, len(selected), out)
            failures += missed
            if missed >= len(batch) / 2:
                time.sleep(30)
                retry = [row for row in batch if row["object_id"] not in done]
                remaining = run_batch(retry, schemas, min(jobs, 2), done, len(selected), out)
                if remaining >= len(batch) / 2:
                    raise RuntimeError("persistent API failures; rerun to resume after service recovers")
        print(f"PASS {attempt + 1} completed={len(done)} failures={failures}", flush=True)
    if len(done) != len(selected):
        raise RuntimeError(f"{len(selected) - len(done)} objects still missing; rerun to resume")
    return done


def main(jobs):
    all_rows = preprocess.read_jsonl(SOURCE / "assignments.jsonl")
    schemas = preprocess.load_schemas(SOURCE / "schemas.jsonl")
    categories = sorted(["airplanes", *random.Random(42).sample(sorted(set(schemas) - {"airplanes"}), 49)])
    selected = {row["object_id"]: row for row in all_rows if row["entity"] in categories}
    if len(schemas) != 99 or len({r["entity"] for r in selected.values()}) != 50:
        raise ValueError("expected 50 selected entities from the frozen 99-category source")
    manifest = {"model": MODEL, "request_options": OPTIONS, "seed": 42, "categories": categories,
                "images": len(selected), "splits": dict(Counter(row["split"] for row in selected.values())),
                "source": str(SOURCE), "full_gallery_label_count": len(all_rows)}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("existing batch differs from current configuration")
    write(path, manifest)
    shutil.copyfile(SOURCE / "schemas.jsonl", OUT / "schemas.jsonl")
    done = label_rows(selected, schemas, jobs, OUT)
    merge(all_rows, done, OUT / "assignments_hybrid.jsonl")
    write(OUT / "complete.json", {"selected_images": len(done), "hybrid_images": len(all_rows),
                                  "assignment_file": "assignments_hybrid.jsonl"})
    print(f"COMPLETE {len(done)} Qwen images; hybrid corpus {len(all_rows)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.jobs)
