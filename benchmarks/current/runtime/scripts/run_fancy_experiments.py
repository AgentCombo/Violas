#!/usr/bin/env python3
"""Run intent-trajectory and query-complexity experiments on Caltech-101."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech_kv_retrieval as retrieval
import run_guideline_pipeline as core
import run_image_intent_pipeline as image_pipeline


TRAJECTORY_QUERIES = [
    {
        "id": "traj0",
        "entity_hint": "airplanes",
        "natural_query": "只看飞机，返回与查询图片外观最相似的三张结果，不需要多样化。",
    },
    {
        "id": "traj1",
        "entity_hint": "airplanes",
        "natural_query": "只看飞机，以视觉相似为主，但轻微偏好推进类型更丰富一些。",
    },
    {
        "id": "traj2",
        "entity_hint": "airplanes",
        "natural_query": "只看飞机，均衡兼顾视觉相似度和推进类型多样性，尽量覆盖螺旋桨飞机与喷气式飞机。",
    },
    {
        "id": "traj3",
        "entity_hint": "airplanes",
        "natural_query": "只看飞机，强烈优先覆盖不同推进类型，至少包含螺旋桨飞机和喷气式飞机，视觉相似度其次。",
    },
]


COMPLEXITY_QUERIES = [
    {
        "id": "cx0",
        "level": "entity only",
        "entity_hint": "airplanes",
        "natural_query": "只看飞机，返回与查询图片外观最相似的三张结果，不需要多样化。",
    },
    {
        "id": "cx1",
        "level": "+ 1 hard facet",
        "entity_hint": "airplanes",
        "natural_query": "只返回喷气式飞机，按与查询图片的视觉相似度返回三张。",
    },
    {
        "id": "cx2",
        "level": "+ 2 hard facets",
        "entity_hint": "airplanes",
        "natural_query": "只返回正在飞行中的喷气式飞机，按与查询图片的视觉相似度返回三张。",
    },
    {
        "id": "cx3",
        "level": "+ 3 hard facets",
        "entity_hint": "airplanes",
        "natural_query": "只返回正在飞行中的军用喷气式飞机，按与查询图片的视觉相似度返回三张。",
    },
    {
        "id": "cx4",
        "level": "+ 2 hard facets + diversity",
        "entity_hint": "airplanes",
        "natural_query": "只返回正在飞行中的喷气式飞机，在保持视觉相关的同时，让军用和民用类型都尽量得到覆盖。",
    },
]


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
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--force-intents", action="store_true")
    parser.add_argument(
        "--trajectory-intents",
        type=Path,
        default=Path("queries/caltech101_intent_trajectory_qwen.json"),
    )
    parser.add_argument(
        "--complexity-intents",
        type=Path,
        default=Path("queries/caltech101_complexity_scaling_qwen.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/caltech_fancy_experiments"),
    )
    return parser.parse_args()


def compile_intents(
    rows: list[dict[str, Any]],
    output: Path,
    cache: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    if output.exists() and not force:
        print(f"LOAD intents: {output}", flush=True)
        return json.loads(output.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    attempts_by_id: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(core.parse_one, row, cache["schemas"][row["entity_hint"]], 3): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            intent, attempts = future.result()
            records[row["id"]] = {
                **row,
                "intent": intent,
                "semantic_text_en": intent["embedding_text_en"],
            }
            attempts_by_id[row["id"]] = attempts
            print(f"INTENT {row['id']}: {intent}", flush=True)
    document = {
        "format": "caltech101-fancy-intent-qwen-v1",
        "model": core.MODEL,
        "temperature": 0.0,
        "seed": 42,
        "lambda_mapping": core.LAMBDA_BY_LEVEL,
        "records": [records[row["id"]] for row in rows],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_path = output.with_name(output.stem + ".raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for attempt in attempts_by_id[row["id"]]:
                handle.write(json.dumps({"id": row["id"], **attempt}, ensure_ascii=False) + "\n")
    return document


def manifest_path(uri: Path) -> Path:
    return uri.with_name(uri.name + ".manifest.json")


def build_scalar_milvus(
    uri: Path,
    collection: str,
    objects: dict[str, dict[str, Any]],
    nlist: int,
) -> dict[str, Any]:
    """Build Milvus with logical ARRAY facet tokens for the baseline."""
    from pymilvus import DataType, MilvusClient

    mp = manifest_path(uri)
    if uri.exists() and mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if (
            manifest.get("collection") == collection
            and manifest.get("object_count") == len(objects)
            and manifest.get("nlist") == nlist
            and manifest.get("format") == "caltech101-kv-array-milvus-v2-no-compaction"
        ):
            print(f"LOAD array-facet Milvus: {uri}", flush=True)
            return manifest
    if uri.exists():
        resolved = uri.resolve()
        cache_root = (PROJECT_ROOT / "cache").resolve()
        if cache_root not in resolved.parents:
            raise RuntimeError(f"refusing to replace path outside cache: {resolved}")
        shutil.rmtree(resolved) if resolved.is_dir() else resolved.unlink()
    mp.unlink(missing_ok=True)
    uri.parent.mkdir(parents=True, exist_ok=True)
    print(f"BUILD array-facet Milvus: {uri}", flush=True)
    client = MilvusClient(str(uri.resolve()))
    dimension = len(next(iter(objects.values()))["normalized"])
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="object_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="entity", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="facet_tokens",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=16,
        max_length=160,
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": nlist},
    )
    client.create_collection(collection_name=collection, schema=schema, index_params=indexes)
    scalar = client.prepare_index_params()
    scalar.add_index(field_name="entity", index_type="INVERTED", index_name="entity_inverted")
    client.create_index(collection_name=collection, index_params=scalar)
    rows: list[dict[str, Any]] = []
    inserted = 0
    for numeric_id, oid in enumerate(sorted(objects)):
        obj = objects[oid]
        tokens = [
            f"{key}={value}"
            for key, value in sorted(obj["description"].get("facet_values", {}).items())
        ]
        row = {
            "id": numeric_id,
            "object_id": oid,
            "entity": obj["description"]["entity"],
            "facet_tokens": tokens,
            "vector": obj["normalized"].astype(np.float32).tolist(),
        }
        rows.append(row)
        if len(rows) >= 500:
            client.insert(collection, rows)
            inserted += len(rows)
            rows = []
            print(f"  inserted={inserted}", flush=True)
    if rows:
        client.insert(collection, rows)
        inserted += len(rows)
    client.load_collection(collection)
    client.close()
    manifest = {
        "format": "caltech101-kv-array-milvus-v2-no-compaction",
        "collection": collection,
        "object_count": inserted,
        "dimension": dimension,
        "nlist": nlist,
        "vector_index": "IVF_FLAT/COSINE",
        "scalar_indexes": ["entity"],
        "facet_filter_storage": "facet_tokens ARRAY",
        "facet_filter_operator": "ARRAY_CONTAINS_ALL",
    }
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def scalar_milvus_search(
    client: Any,
    collection: str,
    query: np.ndarray,
    plan: dict[str, Any],
    cache: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    candidate_k: int,
    nprobe: int,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    legacy = {"entity": plan["entity_scope"], "must": plan["hard_filters"]}
    expression = retrieval.milvus_filter(legacy, cache["schemas"])
    response = client.search(
        collection_name=collection,
        data=[np.asarray(query, dtype=np.float32).tolist()],
        anns_field="vector",
        filter=expression,
        limit=candidate_k,
        output_fields=["object_id", "facet_tokens"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
    )
    hits = response[0] if response else []
    ids: list[str] = []
    for hit in hits:
        entity = hit.get("entity", {})
        oid = entity.get("object_id") or hit.get("object_id")
        if oid in objects and oid not in ids and core.satisfies(objects[oid]["description"], plan):
            ids.append(oid)
    scored = {oid: core.score_object(query, objects[oid]["normalized"]) for oid in ids}
    selected = core.select_exact(
        scored,
        objects,
        plan["facet_dimension"],
        plan["top_k"],
        plan["lambda"],
        plan["category_preferences"],
    )
    return selected, scored, {
        "filter": expression,
        "ann_hit_count": len(ids),
        "candidate_k": candidate_k,
        "nprobe": nprobe,
        "facet_filter_storage": "facet_tokens ARRAY",
        "facet_filter_operator": "ARRAY_CONTAINS_ALL",
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "mean_ms": round(float(statistics.fmean(values)), 4),
        "p50_ms": round(percentile(values, 50), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
    }


def timed(fn: Callable[[], Any], warmup: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(warmup):
        fn()
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = fn()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return result, values


def common_query_image(
    source: dict[str, Any], objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dummy = [{"id": "common", "intent": {"entity_scope": "airplanes"}}]
    return image_pipeline.choose_test_queries(dummy, source, set(objects))["common"]


def prepare_plan(record: dict[str, Any], directory: dict[str, Any]) -> dict[str, Any]:
    return core.materialize_plan_preferences(core.resolve_plan(record["intent"]), directory)


def evaluate(
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
    return {
        "results": core.result_rows(ids, scored, objects),
        "metrics": core.method_metrics(
            ids, scored, oracle_ids, oracle_value, plan, objects, eligible_count
        ),
        "latency": latency(samples),
        "stats": stats,
    }


def run_trajectory(
    args: argparse.Namespace,
    intents: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    query_image: dict[str, Any],
) -> dict[str, Any]:
    query = query_image["normalized"]
    rows = []
    for record in intents["records"]:
        plan = prepare_plan(record, directory)
        eligible_count = len(core.eligible_ids(objects, plan))
        oracle_ids, oracle_scored = core.exhaustive_oracle(query, plan, objects)
        oracle_value = core.objective(
            oracle_ids,
            oracle_scored,
            objects,
            plan["facet_dimension"],
            plan["lambda"],
            plan["category_preferences"],
        )
        fn = lambda: core.interleaved_search(
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
        (ids, scored, stats), samples = timed(fn, args.warmup, args.repeats)
        rows.append(
            {
                **record,
                "resolved_plan": plan,
                "eligible_count": eligible_count,
                "oracle": {
                    "results": core.result_rows(oracle_ids, oracle_scored, objects),
                    "objective": round(oracle_value, 6),
                },
                "violas": evaluate(
                    ids,
                    scored,
                    stats,
                    samples,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
            }
        )
        print(
            f"TRAJECTORY {record['id']}: level={plan['diversity_level']} "
            f"lambda={plan['lambda']:.2f} coverage={rows[-1]['violas']['metrics']['known_category_coverage']}",
            flush=True,
        )
    return {
        "format": "violas-intent-trajectory-v1",
        "query_image": {k: v for k, v in query_image.items() if k != "normalized"},
        "shared_query_image": True,
        "model": intents["model"],
        "temperature": intents["temperature"],
        "seed": intents["seed"],
        "rows": rows,
    }


def run_complexity(
    args: argparse.Namespace,
    intents: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    query_image: dict[str, Any],
    scalar_manifest: dict[str, Any],
) -> dict[str, Any]:
    from pymilvus import MilvusClient

    client = MilvusClient(str(args.scalar_milvus_uri.resolve()))
    client.load_collection(args.scalar_collection)
    query = query_image["normalized"]
    rows = []
    for record in intents["records"]:
        plan = prepare_plan(record, directory)
        eligible_count = len(core.eligible_ids(objects, plan))
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
        m_fn = lambda: scalar_milvus_search(
            client,
            args.scalar_collection,
            query,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )
        (v_ids, v_scored, v_stats), v_samples = timed(
            v_fn, args.warmup, args.repeats
        )
        (m_ids, m_scored, m_stats), m_samples = timed(
            m_fn, args.warmup, args.repeats
        )
        rows.append(
            {
                **record,
                "resolved_plan": plan,
                "eligible_count": eligible_count,
                "oracle": {
                    "results": core.result_rows(oracle_ids, oracle_scored, objects),
                    "objective": round(oracle_value, 6),
                },
                "violas": evaluate(
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
                "milvus_scalar": evaluate(
                    m_ids,
                    m_scored,
                    m_stats,
                    m_samples,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
            }
        )
        print(
            f"COMPLEXITY {record['id']}: hard={len(plan['hard_filters'])} "
            f"d={plan['facet_dimension']} eligible={eligible_count} "
            f"p50 V={rows[-1]['violas']['latency']['p50_ms']:.2f} "
            f"M={rows[-1]['milvus_scalar']['latency']['p50_ms']:.2f}",
            flush=True,
        )
    client.close()
    return {
        "format": "violas-query-complexity-scaling-v1",
        "query_image": {k: v for k, v in query_image.items() if k != "normalized"},
        "shared_query_image": True,
        "model": intents["model"],
        "temperature": intents["temperature"],
        "seed": intents["seed"],
        "scalar_milvus_manifest": scalar_manifest,
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
    trajectory_intents = compile_intents(
        TRAJECTORY_QUERIES, args.trajectory_intents, cache, args.force_intents
    )
    complexity_intents = compile_intents(
        COMPLEXITY_QUERIES, args.complexity_intents, cache, args.force_intents
    )
    scalar_manifest = build_scalar_milvus(
        args.scalar_milvus_uri,
        args.scalar_collection,
        objects,
        args.milvus_nlist,
    )
    query_image = common_query_image(source, objects)
    trajectory = run_trajectory(
        args, trajectory_intents, cache, directory, objects, query_image
    )
    complexity = run_complexity(
        args,
        complexity_intents,
        cache,
        directory,
        objects,
        query_image,
        scalar_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "intent_trajectory_results.json"
    complexity_path = args.output_dir / "query_complexity_results.json"
    trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    complexity_path.write_text(json.dumps(complexity, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {trajectory_path}", flush=True)
    print(f"WROTE {complexity_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
