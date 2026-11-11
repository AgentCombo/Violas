#!/usr/bin/env python3
"""Direct Qwen labeling for the fixed 25-category Caltech MVP."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

from label_caltech_qwen50 import MODEL, OPTIONS, ROOT, SOURCE, label_rows, merge, preprocess, write


OUT = Path("artifacts/caltech_local_mvp")
CATEGORIES = tuple("airplanes watch accordion camera crocodile_head cup helicopter binocular inline_skate metronome "
                   "wild_cat saxophone scissors wrench water_lilly panda platypus octopus stapler anchor mandolin "
                   "cannon pagoda stop_sign mayfly".split())


def main(schema_file, jobs):
    all_schemas = preprocess.load_schemas(schema_file)
    schemas = {entity: all_schemas[entity] for entity in CATEGORIES}
    rows = [row for row in preprocess.read_jsonl(SOURCE / "assignments.jsonl") if row["entity"] in schemas]
    selected = {row["object_id"]: row for row in rows}
    splits = dict(Counter(row["split"] for row in rows))
    if len(schemas) != 25 or len(rows) != len(selected) or splits != {"train": 1840, "test": 216}:
        raise ValueError("category selection or frozen split differs from the approved MVP")
    if any(not (ROOT / oid).is_file() for oid in selected):
        raise ValueError("one or more selected images are missing")
    manifest = {"model": MODEL, "request_options": OPTIONS, "categories": list(CATEGORIES), "images": len(rows),
                "splits": splits, "source_schema": str(schema_file),
                "schema_sha256": hashlib.sha256(schema_file.read_bytes()).hexdigest(),
                "image_source": str(ROOT), "workers": jobs}
    path = OUT / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("existing batch differs from current configuration")
    write(path, manifest)
    (OUT / "schemas.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in schemas.values()),
                                       encoding="utf-8")
    done = label_rows(selected, schemas, jobs, OUT)
    merge(rows, done, OUT / "assignments.jsonl")
    write(OUT / "complete.json", {"images": len(done), "splits": splits, "assignment_file": "assignments.jsonl"})
    print(f"COMPLETE {len(done)} direct Qwen assignments", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemas", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.schemas, args.jobs)
