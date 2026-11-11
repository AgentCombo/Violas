#!/usr/bin/env python3
"""Run the two 100-case canonical baselines for any prepared facet dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "20news-bench",
    PROJECT_ROOT / "ohsumed-bench",
    PROJECT_ROOT / "yahoo-answer-bench",
    PROJECT_ROOT / "CUB-bench",
    PROJECT_ROOT / "flower-bench",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech100_canonical_bench as canonical
import run_caltech_kv_retrieval as retrieval
import run_fancy_experiments as fancy
import run_guideline_pipeline as core
import run_milvus_capability_experiment as ablation
import run_qdrant_chroma_backends as extra_backends


TYPE_ORDER = canonical.TYPE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modality", choices=("image", "text"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--hdmg-cache", type=Path, required=True)
    parser.add_argument("--directory-cache", type=Path, required=True)
    parser.add_argument("--entity-milvus-uri", type=Path, required=True)
    parser.add_argument("--scalar-milvus-uri", type=Path, required=True)
    parser.add_argument("--entity-collection", required=True)
    parser.add_argument("--scalar-collection", required=True)
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--qdrant-collection", required=True)
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--chroma-collection", required=True)
    parser.add_argument("--canonical-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module-path", action="append", default=[])
    parser.add_argument("--milvus-nlist", type=int, default=128)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument("--milvus-candidate-k", type=int, default=80)
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--score-budget", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pickle(path: Path, module_paths: list[str]) -> Any:
    for value in module_paths:
        if value and value not in sys.path:
            sys.path.insert(0, value)
    with path.open("rb") as handle:
        return pickle.load(handle)


def choose_entities(
    candidates: dict[str, dict[str, list[dict[str, Any]]]], seed: int
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    entities = sorted(candidates)
    order = [entities[int(index)] for index in rng.permutation(len(entities))]
    used: set[str] = set()
    result: dict[str, list[str]] = {}
    for query_type in TYPE_ORDER:
        feasible = [entity for entity in order if candidates[entity][query_type]]
        if not feasible:
            raise RuntimeError(f"no entity supports query type {query_type}")
        unused = [entity for entity in feasible if entity not in used]
        selected = unused[:20]
        cursor = 0
        while len(selected) < 20:
            selected.append(feasible[cursor % len(feasible)])
            cursor += 1
        result[query_type] = selected
        used.update(selected)
    return result


def generate_workload(objects: dict[str, dict[str, Any]], seed: int, dataset: str) -> dict[str, Any]:
    grouped = canonical.entity_objects(objects)
    candidates = canonical.enumerate_candidates(objects, grouped)
    assigned = choose_entities(candidates, seed)
    records = []
    index = 1
    occurrence: Counter[tuple[str, str]] = Counter()
    for query_type in TYPE_ORDER:
        for entity in assigned[query_type]:
            local_index = occurrence[(query_type, entity)]
            occurrence[(query_type, entity)] += 1
            candidate = canonical.choose_candidate(
                candidates[entity][query_type],
                local_index,
                query_type in {"H1", "H2", "H+D"},
            )
            intent = canonical.build_intent(entity, query_type, candidate, index - 1)
            records.append(
                {
                    "id": f"{dataset}-cq{index:03d}",
                    "query_type": query_type,
                    "entity_hint": entity,
                    "canonical_intent": intent,
                    "feasibility": {
                        key: value
                        for key, value in candidate.items()
                        if key
                        in {
                            "eligible_count",
                            "eligible_ratio",
                            "selectivity_bucket",
                            "target_values",
                            "target_counts",
                        }
                    },
                }
            )
            index += 1
    return {
        "format": "multidataset-canonical-intent-workload-v1",
        "dataset": dataset,
        "seed": seed,
        "top_k": 3,
        "query_count": len(records),
        "unique_entity_count": len({row["entity_hint"] for row in records}),
        "query_type_counts": dict(Counter(row["query_type"] for row in records)),
        "records": records,
    }


def object_id(entity: str, raw_path: str) -> str:
    name = PurePosixPath(str(raw_path).replace("\\", "/")).name
    return f"{entity}/{name}"


def choose_queries(
    records: list[dict[str, Any]], source: dict[str, Any], gallery_ids: set[str]
) -> dict[str, dict[str, Any]]:
    cursor: Counter[str] = Counter()
    chosen: dict[str, dict[str, Any]] = {}
    for record in records:
        entity = record["entity_hint"]
        split = source["test_data"][entity]
        selected = None
        for offset in range(len(split["vectors"])):
            index = (cursor[entity] + offset) % len(split["vectors"])
            description = split["descriptions"][index]
            oid = object_id(entity, description["img_path"])
            if oid in gallery_ids:
                continue
            vector = np.asarray(split["vectors"][index], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                continue
            selected = {
                "test_index": index,
                "object_id": oid,
                "img_path": str(description["img_path"]),
                "normalized": vector / norm,
            }
            cursor[entity] = index + 1
            break
        if selected is None:
            raise RuntimeError(f"no test query for {record['id']} / {entity}")
        chosen[record["id"]] = selected
    return chosen


def latency(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": round(float(statistics.fmean(values)), 4),
        "p50_ms": round(float(np.percentile(array, 50)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "min_ms": round(float(array.min()), 4),
        "max_ms": round(float(array.max()), 4),
    }


def timed(fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(warmup):
        fn()
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return result, samples


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if "trace" not in stats:
        return stats
    return {key: value for key, value in stats.items() if key != "trace"}


def method_result(
    ids: list[str],
    scored: dict[str, float],
    stats: dict[str, Any],
    samples: list[float],
    oracle_ids: list[str],
    oracle_value: float,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    eligible_count: int,
) -> dict[str, Any]:
    metrics = core.method_metrics(
        ids, scored, oracle_ids, oracle_value, plan, objects, eligible_count
    )
    metrics["target_coverage_rate"] = canonical.target_coverage(ids, plan, objects)
    return {
        "results": core.result_rows(ids, scored, objects),
        "metrics": metrics,
        "latency": latency(samples),
        "stats": compact_stats(stats),
    }


def entity_only_result(
    ids: list[str],
    scored: dict[str, float],
    stats: dict[str, Any],
    samples: list[float],
    oracle_ids: list[str],
    oracle_value: float,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    eligible_count: int,
) -> dict[str, Any]:
    valid_ids = [
        oid for oid in ids if core.satisfies(objects[oid]["description"], plan)
    ]
    metrics = core.method_metrics(
        valid_ids,
        scored,
        oracle_ids,
        oracle_value,
        plan,
        objects,
        eligible_count,
    )
    metrics["raw_hard_condition_validity"] = (
        round(len(valid_ids) / len(ids), 6) if ids else None
    )
    metrics["target_coverage_rate"] = canonical.target_coverage(
        valid_ids, plan, objects
    )
    return {
        "results": core.result_rows(ids, scored, objects),
        "evaluator_valid_results": core.result_rows(valid_ids, scored, objects),
        "metrics": metrics,
        "latency": latency(samples),
        "stats": compact_stats(stats),
    }


def mean_metric(rows: list[dict[str, Any]], method: str, name: str) -> float | None:
    values = [
        row["methods"][method]["metrics"].get(name)
        for row in rows
        if row["methods"][method]["metrics"].get(name) is not None
    ]
    return round(float(np.mean(values)), 6) if values else None


def aggregate(
    rows: list[dict[str, Any]], method: str, samples: list[float]
) -> dict[str, Any]:
    validity = (
        "raw_hard_condition_validity"
        if method.endswith("_entity_only")
        else "hard_condition_validity"
    )
    return {
        "query_count": len(rows),
        "mean_objective_ratio": mean_metric(rows, method, "objective_ratio"),
        "mean_oracle_object_recall": mean_metric(rows, method, "oracle_object_recall"),
        "mean_hard_condition_validity": mean_metric(rows, method, validity),
        "mean_fill_rate": mean_metric(rows, method, "fill_rate"),
        "mean_target_coverage_rate": mean_metric(rows, method, "target_coverage_rate"),
        "latency": latency(samples),
    }


def preview(path: Path, limit: int = 600) -> str | None:
    try:
        return " ".join(path.read_text(encoding="utf-8", errors="replace").split())[:limit]
    except OSError:
        return None


def text_path(root: Path, entity: str, raw: str) -> Path:
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    direct = root / candidate
    return direct if direct.exists() else root / entity / candidate.name


def add_previews(
    row: dict[str, Any], objects: dict[str, dict[str, Any]], root: Path, modality: str
) -> None:
    if modality != "text":
        return
    row["query_object"]["preview"] = preview(
        text_path(root, row["entity_hint"], row["query_object"]["img_path"])
    )
    for method in row["methods"].values():
        for result in method.get("results", []):
            description = objects[result["object_id"]]["description"]
            result["preview"] = preview(
                text_path(root, row["entity_hint"], description["img_path"])
            )
    for result in row["oracle"]["results"]:
        description = objects[result["object_id"]]["description"]
        result["preview"] = preview(
            text_path(root, row["entity_hint"], description["img_path"])
        )


def run(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    from pymilvus import MilvusClient

    source = load_pickle(args.source_cache, args.module_path)
    cache = load_pickle(args.hdmg_cache, args.module_path)
    directory = load_pickle(args.directory_cache, args.module_path)
    objects = retrieval.make_object_map(cache)
    queries = choose_queries(workload["records"], source, set(objects))
    entity_manifest = ablation.build_entity_only_collection(
        args.entity_milvus_uri,
        args.entity_collection,
        objects,
        args.milvus_nlist,
    )
    scalar_manifest = fancy.build_scalar_milvus(
        args.scalar_milvus_uri,
        args.scalar_collection,
        objects,
        args.milvus_nlist,
    )
    qdrant_manifest = extra_backends.build_qdrant(
        args.qdrant_path, args.qdrant_collection, objects
    )
    chroma_manifest = extra_backends.build_chroma(
        args.chroma_path, args.chroma_collection, objects
    )
    entity_client = MilvusClient(str(args.entity_milvus_uri.resolve()))
    scalar_client = MilvusClient(str(args.scalar_milvus_uri.resolve()))
    entity_client.load_collection(args.entity_collection)
    scalar_client.load_collection(args.scalar_collection)
    qdrant_client = extra_backends.open_qdrant(args.qdrant_path)
    chroma_client, chroma_collection = extra_backends.open_chroma(
        args.chroma_path, args.chroma_collection
    )
    all_samples: dict[str, list[float]] = defaultdict(list)
    type_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows = []
    for index, record in enumerate(workload["records"], 1):
        plan = core.materialize_plan_preferences(
            core.resolve_plan(record["canonical_intent"]), directory
        )
        query_row = queries[record["id"]]
        query = query_row["normalized"]
        hdmg_fn = lambda: core.interleaved_search(
            query,
            plan,
            cache,
            directory,
            objects,
            args.nav_budget,
            args.score_budget,
            args.beam_width,
            args.batch_size,
            "paper",
        )
        flat_fn = lambda: core.group_scan_search(
            query, plan, cache["entity_objects"], objects
        )
        milvus_entity_fn = lambda: ablation.entity_only_search(
            entity_client,
            args.entity_collection,
            query,
            plan,
            objects,
            args.milvus_nprobe,
        )
        milvus_facet_fn = lambda: fancy.scalar_milvus_search(
            scalar_client,
            args.scalar_collection,
            query,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )
        qdrant_entity_fn = lambda: extra_backends.qdrant_search(
            qdrant_client,
            args.qdrant_collection,
            query,
            plan,
            objects,
            args.milvus_candidate_k,
            False,
        )
        qdrant_facet_fn = lambda: extra_backends.qdrant_search(
            qdrant_client,
            args.qdrant_collection,
            query,
            plan,
            objects,
            args.milvus_candidate_k,
            True,
        )
        chroma_entity_fn = lambda: extra_backends.chroma_search(
            chroma_collection,
            query,
            plan,
            objects,
            args.milvus_candidate_k,
            False,
        )
        chroma_facet_fn = lambda: extra_backends.chroma_search(
            chroma_collection,
            query,
            plan,
            objects,
            args.milvus_candidate_k,
            True,
        )
        functions = {
            "vg_hdmg": hdmg_fn,
            "vg_flat": flat_fn,
            "milvus_entity_only": milvus_entity_fn,
            "milvus_facet": milvus_facet_fn,
            "qdrant_entity_only": qdrant_entity_fn,
            "qdrant_facet": qdrant_facet_fn,
            "chroma_entity_only": chroma_entity_fn,
            "chroma_facet": chroma_facet_fn,
        }
        outputs: dict[str, tuple[list[str], dict[str, float], dict[str, Any]]] = {}
        samples: dict[str, list[float]] = {}
        for method, fn in functions.items():
            outputs[method], samples[method] = timed(
                fn, args.warmup, args.repeats
            )
        oracle_ids, oracle_scored = core.exhaustive_oracle(query, plan, objects)
        eligible_count = len(oracle_scored)
        oracle_value = core.objective(
            oracle_ids,
            oracle_scored,
            objects,
            plan["facet_dimension"],
            plan["lambda"],
            plan["category_preferences"],
        )
        for method, values in samples.items():
            all_samples[method].extend(values)
            type_samples[record["query_type"]][method].extend(values)
        methods = {}
        for method in ("vg_hdmg", "vg_flat", "milvus_facet", "qdrant_facet", "chroma_facet"):
            ids, scored, stats = outputs[method]
            methods[method] = method_result(
                ids,
                scored,
                stats,
                samples[method],
                oracle_ids,
                oracle_value,
                plan,
                objects,
                eligible_count,
            )
        for method in ("milvus_entity_only", "qdrant_entity_only", "chroma_entity_only"):
            ids, scored, stats = outputs[method]
            methods[method] = entity_only_result(
                ids,
                scored,
                stats,
                samples[method],
                oracle_ids,
                oracle_value,
                plan,
                objects,
                eligible_count,
            )
        row = {
            **record,
            "resolved_plan": plan,
            "eligible_count": eligible_count,
            "query_object": {
                key: value for key, value in query_row.items() if key != "normalized"
            },
            "oracle": {
                "results": core.result_rows(oracle_ids, oracle_scored, objects),
                "objective": round(oracle_value, 6),
            },
            "methods": methods,
        }
        add_previews(row, objects, args.dataset_root, args.modality)
        rows.append(row)
        print(
            f"DONE {index:03d}/100 {record['id']} {record['query_type']} "
            f"entity={record['entity_hint']} eligible={eligible_count} "
            f"obj HDMG={methods['vg_hdmg']['metrics']['objective_ratio']:.3f} "
            f"flat={methods['vg_flat']['metrics']['objective_ratio']:.3f} "
            f"M={methods['milvus_facet']['metrics']['objective_ratio']:.3f} "
            f"Q={methods['qdrant_facet']['metrics']['objective_ratio']:.3f} "
            f"C={methods['chroma_facet']['metrics']['objective_ratio']:.3f}",
            flush=True,
        )
    entity_client.close()
    scalar_client.close()
    qdrant_client.close()
    method_names = (
        "vg_hdmg",
        "vg_flat",
        "milvus_entity_only",
        "milvus_facet",
        "qdrant_entity_only",
        "qdrant_facet",
        "chroma_entity_only",
        "chroma_facet",
    )
    overall = {
        method: aggregate(rows, method, all_samples[method])
        for method in method_names
    }
    by_type = {}
    for query_type in TYPE_ORDER:
        subset = [row for row in rows if row["query_type"] == query_type]
        by_type[query_type] = {
            method: aggregate(
                subset, method, type_samples[query_type][method]
            )
            for method in method_names
        }
    return {
        "format": "multidataset-canonical100-five-system-report-v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": args.dataset,
        "modality": args.modality,
        "query_contract": {
            "query_count": 100,
            "unique_entity_count": workload["unique_entity_count"],
            "query_types": {query_type: 20 for query_type in TYPE_ORDER},
            "query_input": "held-out test embedding + canonical JSON intent",
            "gallery_object_count": len(objects),
            "natural_language_or_llm_used_online": False,
            "top_k": 3,
        },
        "config": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "violas": {
                "nav_budget": args.nav_budget,
                "score_budget": args.score_budget,
                "beam_width": args.beam_width,
                "batch_size": args.batch_size,
            },
            "vg_flat": {
                "execution": "entity_key_group_scan",
                "score_budget": None,
            },
            "milvus_nlist": args.milvus_nlist,
            "milvus_nprobe": args.milvus_nprobe,
            "milvus_candidate_k": args.milvus_candidate_k,
            "entity_manifest": entity_manifest,
            "scalar_manifest": scalar_manifest,
            "qdrant_manifest": qdrant_manifest,
            "chroma_manifest": chroma_manifest,
            "measurement": "single execution per query; no warmup or unmeasured pre-call",
        },
        "aggregate": {"overall": overall, "by_query_type": by_type},
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    core.self_test_select()
    cache = load_pickle(args.hdmg_cache, args.module_path)
    objects = retrieval.make_object_map(cache)
    if args.canonical_output.exists():
        workload = json.loads(args.canonical_output.read_text(encoding="utf-8"))
        if len(workload.get("records", [])) != 100:
            raise RuntimeError(
                f"existing workload is not 100 queries: {args.canonical_output}"
            )
        print(f"REUSE {args.canonical_output} queries=100", flush=True)
    else:
        workload = generate_workload(objects, args.seed, args.dataset)
        args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
        args.canonical_output.write_text(
            json.dumps(workload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"WROTE {args.canonical_output} queries=100 entities={workload['unique_entity_count']}",
            flush=True,
        )
    report = run(args, workload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
