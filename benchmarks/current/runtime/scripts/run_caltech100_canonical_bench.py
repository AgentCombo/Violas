#!/usr/bin/env python3
"""Generate 100 legal Caltech-101 canonical intents and run two baselines."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech_kv_retrieval as retrieval
import run_fancy_experiments as fancy
import run_guideline_pipeline as core
import run_image_intent_pipeline as image_pipeline
import run_milvus_capability_experiment as ablation


TYPE_ORDER = ("E", "H1", "H2", "D", "H+D")
LEVELS = ("light", "balanced", "strong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("cache/caltech_101_ObjectCategories_ts0.1_rs42_a0.500_folder.pkl"),
    )
    parser.add_argument(
        "--hdmg-cache", type=Path, default=Path("cache/caltech101_kv_train_v3_hdmg.pkl")
    )
    parser.add_argument(
        "--directory-cache",
        type=Path,
        default=Path("cache/caltech101_semantic_directory_v1.pkl"),
    )
    parser.add_argument(
        "--entity-milvus-uri",
        type=Path,
        default=Path("cache/caltech101_entity_only_milvus_ivf.db"),
    )
    parser.add_argument("--entity-collection", default="caltech101_entity_only_train")
    parser.add_argument(
        "--scalar-milvus-uri",
        type=Path,
        default=Path("cache/caltech101_kv_milvus_scalar_ivf.db"),
    )
    parser.add_argument("--scalar-collection", default="caltech101_kv_scalar_train")
    parser.add_argument("--milvus-nlist", type=int, default=128)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument("--milvus-candidate-k", type=int, default=80)
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--score-budget", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=Path("queries/caltech101_canonical_100.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/caltech100_canonical_baselines"),
    )
    return parser.parse_args()


def selectivity_bucket(ratio: float) -> str:
    if ratio < 0.10:
        return "rare"
    if ratio < 0.30:
        return "medium"
    return "common"


def entity_objects(
    objects: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for oid, obj in objects.items():
        grouped[obj["description"]["entity"]].append(oid)
    return {entity: sorted(ids) for entity, ids in grouped.items()}


def values_for(
    oid: str, objects: dict[str, dict[str, Any]]
) -> dict[str, str]:
    return objects[oid]["description"].get("facet_values", {})


def enumerate_candidates(
    objects: dict[str, dict[str, Any]], grouped: dict[str, list[str]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entity, ids in grouped.items():
        total = len(ids)
        facet_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for oid in ids:
            for key, value in values_for(oid, objects).items():
                facet_counts[key][value] += 1
        candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in TYPE_ORDER}
        candidates["E"] = [{"eligible_count": total, "eligible_ratio": 1.0}]

        for key, counts in sorted(facet_counts.items()):
            for value, count in sorted(counts.items()):
                if count >= 3:
                    ratio = count / total
                    candidates["H1"].append(
                        {
                            "hard_filters": [{"key": key, "op": "eq", "value": value}],
                            "eligible_count": count,
                            "eligible_ratio": ratio,
                            "selectivity_bucket": selectivity_bucket(ratio),
                        }
                    )

        facet_keys = sorted(facet_counts)
        for key1, key2 in itertools.combinations(facet_keys, 2):
            combos: Counter[tuple[str, str]] = Counter()
            for oid in ids:
                values = values_for(oid, objects)
                if key1 in values and key2 in values:
                    combos[(values[key1], values[key2])] += 1
            for (value1, value2), count in sorted(combos.items()):
                if count >= 3:
                    ratio = count / total
                    candidates["H2"].append(
                        {
                            "hard_filters": [
                                {"key": key1, "op": "eq", "value": value1},
                                {"key": key2, "op": "eq", "value": value2},
                            ],
                            "eligible_count": count,
                            "eligible_ratio": ratio,
                            "selectivity_bucket": selectivity_bucket(ratio),
                        }
                    )

        for dimension, counts in sorted(facet_counts.items()):
            supported = [value for value, count in sorted(counts.items()) if count >= 1]
            if len(supported) >= 2:
                target_values = sorted(
                    supported, key=lambda value: (-counts[value], value)
                )[:2]
                candidates["D"].append(
                    {
                        "facet_dimension": dimension,
                        "target_values": target_values,
                        "target_counts": {value: counts[value] for value in target_values},
                        "eligible_count": total,
                        "eligible_ratio": 1.0,
                    }
                )

        for hard_key, counts in sorted(facet_counts.items()):
            for hard_value, hard_count in sorted(counts.items()):
                if hard_count < 3:
                    continue
                conditioned = [
                    oid
                    for oid in ids
                    if values_for(oid, objects).get(hard_key) == hard_value
                ]
                for dimension in facet_keys:
                    if dimension == hard_key:
                        continue
                    diversity_counts: Counter[str] = Counter(
                        values_for(oid, objects).get(dimension)
                        for oid in conditioned
                        if values_for(oid, objects).get(dimension) is not None
                    )
                    supported = [
                        value for value, count in sorted(diversity_counts.items()) if count >= 1
                    ]
                    if len(supported) < 2:
                        continue
                    target_values = sorted(
                        supported, key=lambda value: (-diversity_counts[value], value)
                    )[:2]
                    ratio = hard_count / total
                    candidates["H+D"].append(
                        {
                            "hard_filters": [
                                {"key": hard_key, "op": "eq", "value": hard_value}
                            ],
                            "facet_dimension": dimension,
                            "target_values": target_values,
                            "target_counts": {
                                value: diversity_counts[value] for value in target_values
                            },
                            "eligible_count": hard_count,
                            "eligible_ratio": ratio,
                            "selectivity_bucket": selectivity_bucket(ratio),
                        }
                    )
        for rows in candidates.values():
            rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
        result[entity] = candidates
    return result


def assign_entities(
    candidates: dict[str, dict[str, list[dict[str, Any]]]], seed: int
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    entities = sorted(candidates)
    rank = {entity: int(value) for entity, value in zip(entities, rng.permutation(len(entities)))}
    complex_types = ("H+D", "H2", "D", "H1")
    slots = [query_type for query_type in complex_types for _ in range(20)]
    slots.sort(
        key=lambda query_type: sum(
            bool(candidates[entity][query_type]) for entity in entities
        )
    )
    slot_to_entity: dict[int, str] = {}
    entity_to_slot: dict[str, int] = {}

    def augment(slot_index: int, seen: set[str]) -> bool:
        query_type = slots[slot_index]
        eligible = [entity for entity in entities if candidates[entity][query_type]]
        eligible.sort(
            key=lambda entity: (
                sum(bool(candidates[entity][kind]) for kind in complex_types),
                rank[entity],
                entity,
            )
        )
        for entity in eligible:
            if entity in seen:
                continue
            seen.add(entity)
            previous = entity_to_slot.get(entity)
            if previous is None or augment(previous, seen):
                entity_to_slot[entity] = slot_index
                slot_to_entity[slot_index] = entity
                return True
        return False

    for slot_index in range(len(slots)):
        if not augment(slot_index, set()):
            raise RuntimeError(f"cannot assign 20 unique entities for {slots[slot_index]}")

    assigned: dict[str, list[str]] = {query_type: [] for query_type in TYPE_ORDER}
    for slot_index, entity in slot_to_entity.items():
        assigned[slots[slot_index]].append(entity)
    used = set(entity_to_slot)
    remaining = [entity for entity in entities if entity not in used]
    if len(remaining) != 19:
        raise AssertionError(f"expected 19 unassigned entities, got {len(remaining)}")
    assigned["E"].extend(remaining)
    duplicate = min(
        entities,
        key=lambda entity: (0 if entity == "airplanes" else 1, rank[entity], entity),
    )
    assigned["E"].append(duplicate)
    for query_type in assigned:
        assigned[query_type].sort(key=lambda entity: (rank[entity], entity))
        if len(assigned[query_type]) != 20:
            raise AssertionError((query_type, len(assigned[query_type])))
    return assigned


def choose_candidate(
    rows: list[dict[str, Any]], index: int, use_selectivity: bool
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no feasible candidate")
    if use_selectivity:
        desired = ("common", "medium", "rare")[index % 3]
        matching = [row for row in rows if row.get("selectivity_bucket") == desired]
        if matching:
            rows = matching
    return dict(rows[index % len(rows)])


def build_intent(
    entity: str,
    query_type: str,
    candidate: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    if query_type in {"D", "H+D"}:
        level = LEVELS[index % len(LEVELS)]
        preferences = [
            {"value": value, "quota": 1, "weight_level": level}
            for value in candidate["target_values"]
        ]
        dimension = candidate["facet_dimension"]
    else:
        level = "off"
        preferences = []
        dimension = None
    return {
        "entity_scope": entity,
        "hard_filters": candidate.get("hard_filters", []),
        "facet_dimension": dimension,
        "diversity_level": level,
        "category_preferences": preferences,
        "top_k": 3,
        "embedding_text_en": entity.replace("_", " "),
        "unsupported": [],
    }


def generate_workload(
    objects: dict[str, dict[str, Any]], seed: int
) -> dict[str, Any]:
    grouped = entity_objects(objects)
    candidates = enumerate_candidates(objects, grouped)
    assigned = assign_entities(candidates, seed)
    records: list[dict[str, Any]] = []
    global_index = 1
    for query_type in TYPE_ORDER:
        for type_index, entity in enumerate(assigned[query_type]):
            candidate = choose_candidate(
                candidates[entity][query_type],
                type_index,
                query_type in {"H1", "H2", "H+D"},
            )
            intent = build_intent(entity, query_type, candidate, type_index)
            records.append(
                {
                    "id": f"cq{global_index:03d}",
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
            global_index += 1
    counts = Counter(record["query_type"] for record in records)
    unique_entities = {record["entity_hint"] for record in records}
    if counts != Counter({query_type: 20 for query_type in TYPE_ORDER}):
        raise AssertionError(counts)
    if len(unique_entities) != 99:
        raise AssertionError(f"expected 99 unique entities, got {len(unique_entities)}")
    return {
        "format": "caltech101-canonical-intent-workload-v1",
        "seed": seed,
        "top_k": 3,
        "query_count": len(records),
        "unique_entity_count": len(unique_entities),
        "query_type_counts": dict(counts),
        "records": records,
    }


def latency_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "mean_ms": round(float(statistics.fmean(values)), 4),
        "p50_ms": round(float(np.percentile(array, 50)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "min_ms": round(float(array.min()), 4),
        "max_ms": round(float(array.max()), 4),
    }


def timed(fn: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return values


def target_coverage(
    ids: list[str], plan: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> float | None:
    targets = set(plan["category_preferences"])
    if not targets or plan["facet_dimension"] is None:
        return None
    covered = {
        core.label_of(objects, oid, plan["facet_dimension"])
        for oid in ids
    }
    return round(len(targets & covered) / len(targets), 6)


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
    metrics["target_coverage_rate"] = target_coverage(ids, plan, objects)
    return {
        "results": core.result_rows(ids, scored, objects),
        "metrics": metrics,
        "latency": latency_summary(samples),
        "stats": stats,
    }


def mean_metric(
    rows: list[dict[str, Any]], method: str, metric: str
) -> float | None:
    values = [
        row["methods"][method]["metrics"].get(metric)
        for row in rows
        if row["methods"][method]["metrics"].get(metric) is not None
    ]
    return round(float(np.mean(values)), 6) if values else None


def aggregate(
    rows: list[dict[str, Any]], method: str, samples: list[float]
) -> dict[str, Any]:
    validity_metric = (
        "raw_hard_condition_validity"
        if method == "milvus_entity_only"
        else "hard_condition_validity"
    )
    return {
        "query_count": len(rows),
        "mean_objective_ratio": mean_metric(rows, method, "objective_ratio"),
        "mean_oracle_object_recall": mean_metric(rows, method, "oracle_object_recall"),
        "mean_hard_condition_validity": mean_metric(rows, method, validity_metric),
        "mean_fill_rate": mean_metric(rows, method, "fill_rate"),
        "mean_target_coverage_rate": mean_metric(rows, method, "target_coverage_rate"),
        "latency": latency_summary(samples),
    }


def run_benchmark(
    args: argparse.Namespace,
    workload: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
    source: dict[str, Any],
    objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from pymilvus import MilvusClient

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
    records_for_images = [
        {"id": row["id"], "intent": {"entity_scope": row["entity_hint"]}}
        for row in workload["records"]
    ]
    query_images = image_pipeline.choose_test_queries(
        records_for_images, source, set(objects)
    )
    if set(row["object_id"] for row in query_images.values()) & set(objects):
        raise AssertionError("query/gallery leakage detected")
    entity_client = MilvusClient(str(args.entity_milvus_uri.resolve()))
    entity_client.load_collection(args.entity_collection)
    scalar_client = MilvusClient(str(args.scalar_milvus_uri.resolve()))
    scalar_client.load_collection(args.scalar_collection)
    all_samples: dict[str, list[float]] = defaultdict(list)
    type_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows = []
    for index, record in enumerate(workload["records"], 1):
        plan = core.materialize_plan_preferences(
            core.resolve_plan(record["canonical_intent"]), directory
        )
        query_image = query_images[record["id"]]
        query = query_image["normalized"]
        eligible_count = len(core.eligible_ids(objects, plan))
        if eligible_count != record["feasibility"]["eligible_count"]:
            raise AssertionError((record["id"], eligible_count, record["feasibility"]))
        oracle_ids, oracle_scored = core.exhaustive_oracle(query, plan, objects)
        oracle_value = core.objective(
            oracle_ids,
            oracle_scored,
            objects,
            plan["facet_dimension"],
            plan["lambda"],
            plan["category_preferences"],
        )
        v_fn = lambda: core.interleaved_search(
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
        e_fn = lambda: ablation.entity_only_search(
            entity_client,
            args.entity_collection,
            query,
            plan,
            objects,
            args.milvus_nprobe,
        )
        s_fn = lambda: fancy.scalar_milvus_search(
            scalar_client,
            args.scalar_collection,
            query,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )
        v_ids, v_scored, v_stats = v_fn()
        e_ids, e_scored, e_stats = e_fn()
        s_ids, s_scored, s_stats = s_fn()
        e_valid = [
            oid
            for oid in e_ids
            if core.satisfies(objects[oid]["description"], plan)
        ]
        v_samples = timed(v_fn, args.warmup, args.repeats)
        e_samples = timed(e_fn, args.warmup, args.repeats)
        s_samples = timed(s_fn, args.warmup, args.repeats)
        for name, samples in (
            ("violas", v_samples),
            ("milvus_entity_only", e_samples),
            ("milvus_scalar", s_samples),
        ):
            all_samples[name].extend(samples)
            type_samples[record["query_type"]][name].extend(samples)
        e_metrics = core.method_metrics(
            e_valid,
            e_scored,
            oracle_ids,
            oracle_value,
            plan,
            objects,
            eligible_count,
        )
        e_metrics["raw_hard_condition_validity"] = (
            round(len(e_valid) / len(e_ids), 6) if e_ids else None
        )
        e_metrics["target_coverage_rate"] = target_coverage(e_valid, plan, objects)
        e_metrics["raw_result_count"] = len(e_ids)
        e_metrics["evaluator_valid_count"] = len(e_valid)
        row = {
            **record,
            "query_image": {
                key: value for key, value in query_image.items() if key != "normalized"
            },
            "resolved_plan": plan,
            "eligible_count": eligible_count,
            "oracle": {
                "results": core.result_rows(oracle_ids, oracle_scored, objects),
                "objective": round(oracle_value, 6),
            },
            "methods": {
                "violas": method_result(
                    v_ids,
                    v_scored,
                    v_stats,
                    v_samples,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
                "milvus_entity_only": {
                    "results": core.result_rows(e_ids, e_scored, objects),
                    "evaluator_valid_results": core.result_rows(e_valid, e_scored, objects),
                    "metrics": e_metrics,
                    "latency": latency_summary(e_samples),
                    "stats": e_stats,
                },
                "milvus_scalar": method_result(
                    s_ids,
                    s_scored,
                    s_stats,
                    s_samples,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
            },
        }
        rows.append(row)
        print(
            f"DONE {index:03d}/100 {record['id']} type={record['query_type']} "
            f"entity={record['entity_hint']} eligible={eligible_count} "
            f"obj V={row['methods']['violas']['metrics']['objective_ratio']:.3f} "
            f"E={e_metrics['objective_ratio']:.3f} "
            f"S={row['methods']['milvus_scalar']['metrics']['objective_ratio']:.3f}",
            flush=True,
        )
    entity_client.close()
    scalar_client.close()
    by_type: dict[str, Any] = {}
    for query_type in TYPE_ORDER:
        subset = [row for row in rows if row["query_type"] == query_type]
        by_type[query_type] = {
            name: aggregate(subset, name, type_samples[query_type][name])
            for name in ("violas", "milvus_entity_only", "milvus_scalar")
        }
    overall = {
        name: aggregate(rows, name, all_samples[name])
        for name in ("violas", "milvus_entity_only", "milvus_scalar")
    }
    return {
        "format": "caltech101-canonical100-baseline-report-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_contract": {
            "query_count": 100,
            "unique_entity_count": 99,
            "query_types": {query_type: 20 for query_type in TYPE_ORDER},
            "query_input": "frozen Caltech-101 test image embedding + canonical JSON intent",
            "gallery": "Caltech-101 train split, 6,984 objects",
            "natural_language_or_llm_used": False,
            "top_k": 3,
            "query_gallery_overlap": 0,
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
            "milvus_nlist": args.milvus_nlist,
            "milvus_nprobe": args.milvus_nprobe,
            "milvus_candidate_k": args.milvus_candidate_k,
            "entity_manifest": entity_manifest,
            "scalar_manifest": scalar_manifest,
        },
        "aggregate": {"overall": overall, "by_query_type": by_type},
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    core.self_test_select()
    with args.hdmg_cache.open("rb") as handle:
        cache = pickle.load(handle)
    with args.directory_cache.open("rb") as handle:
        directory = pickle.load(handle)
    with args.source_cache.open("rb") as handle:
        source = pickle.load(handle)
    objects = retrieval.make_object_map(cache)
    workload = generate_workload(objects, args.seed)
    args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
    args.canonical_output.write_text(
        json.dumps(workload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"WROTE workload: {args.canonical_output} "
        f"queries={workload['query_count']} entities={workload['unique_entity_count']}",
        flush=True,
    )
    report = run_benchmark(args, workload, cache, directory, source, objects)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
