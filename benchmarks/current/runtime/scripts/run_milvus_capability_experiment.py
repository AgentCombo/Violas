#!/usr/bin/env python3
"""Compare Violas with two intentionally different Milvus capabilities.

Experiment A stores only entity and vector in Milvus, applies one entity
filter, and returns the raw ANN top-k.  Facet labels are evaluator-only.

Experiment B uses the existing entity+facet Milvus collection, applies hard
facet predicates, retrieves an ANN candidate pool, and runs the exact same
soft-quota SELECT objective used by the full-corpus Oracle and Violas.

The natural-language intent and frozen test-image embedding are prepared once
and excluded from retrieval latency for every backend.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech_kv_retrieval as retrieval
import run_guideline_pipeline as core
import run_image_intent_pipeline as image_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intents",
        type=Path,
        default=Path("queries/caltech101_paper_soft_quota_10_qwen.json"),
    )
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("cache/caltech_101_ObjectCategories_ts0.1_rs42_a0.500_folder.pkl"),
    )
    parser.add_argument(
        "--hdmg-cache",
        type=Path,
        default=Path("cache/caltech101_kv_train_v3_hdmg.pkl"),
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
        "--facet-milvus-uri",
        type=Path,
        default=Path("cache/caltech101_kv_milvus_ivf.db"),
    )
    parser.add_argument("--facet-collection", default="caltech101_kv_train")
    parser.add_argument("--milvus-nlist", type=int, default=128)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument("--milvus-candidate-k", type=int, default=80)
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--score-budget", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/caltech10_milvus_capability_ablation"),
    )
    parser.add_argument(
        "--local-dataset-root",
        default=r"dataset/caltech-101/101_ObjectCategories",
    )
    return parser.parse_args()


def entity_manifest_path(uri: Path) -> Path:
    return uri.with_name(uri.name + ".manifest.json")


def build_entity_only_collection(
    uri: Path,
    collection: str,
    objects: dict[str, dict[str, Any]],
    nlist: int,
) -> dict[str, Any]:
    """Build a physical Milvus baseline with no facet fields at all."""
    from pymilvus import DataType, MilvusClient

    manifest_path = entity_manifest_path(uri)
    if uri.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "format": "caltech101-entity-only-milvus-v2-no-compaction",
            "collection": collection,
            "object_count": len(objects),
            "nlist": nlist,
            "fields": ["id", "object_id", "entity", "vector"],
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            print(f"LOAD entity-only Milvus cache: {uri}", flush=True)
            return manifest

    if uri.exists():
        resolved = uri.resolve()
        cache_root = (PROJECT_ROOT / "cache").resolve()
        if cache_root not in resolved.parents:
            raise RuntimeError(f"refusing to replace path outside cache: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    manifest_path.unlink(missing_ok=True)
    uri.parent.mkdir(parents=True, exist_ok=True)

    print(f"BUILD entity-only Milvus IVF_FLAT: {uri}", flush=True)
    client = MilvusClient(str(uri.resolve()))
    dimension = len(next(iter(objects.values()))["normalized"])
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="object_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="entity", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": nlist},
    )
    client.create_collection(
        collection_name=collection, schema=schema, index_params=index_params
    )
    scalar = client.prepare_index_params()
    scalar.add_index(
        field_name="entity", index_type="INVERTED", index_name="entity_inverted"
    )
    client.create_index(collection_name=collection, index_params=scalar)

    rows: list[dict[str, Any]] = []
    inserted = 0
    for numeric_id, oid in enumerate(sorted(objects)):
        obj = objects[oid]
        rows.append(
            {
                "id": numeric_id,
                "object_id": oid,
                "entity": obj["description"]["entity"],
                "vector": obj["normalized"].astype(np.float32).tolist(),
            }
        )
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
        "format": "caltech101-entity-only-milvus-v2-no-compaction",
        "collection": collection,
        "object_count": inserted,
        "dimension": dimension,
        "fields": ["id", "object_id", "entity", "vector"],
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "nlist": nlist,
        "scalar_indexes": ["entity"],
        "facet_fields_present": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DONE entity-only Milvus: objects={inserted}", flush=True)
    return manifest


def entity_only_search(
    client: Any,
    collection: str,
    query: np.ndarray,
    plan: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    nprobe: int,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    """Entity filter + ANN top-k; deliberately no facet filter or SELECT."""
    legacy = {"entity": plan["entity_scope"], "must": []}
    expression = retrieval.milvus_filter(legacy, None)
    response = client.search(
        collection_name=collection,
        data=[np.asarray(query, dtype=np.float32).tolist()],
        anns_field="vector",
        filter=expression,
        limit=plan["top_k"],
        output_fields=["object_id"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": nprobe}},
    )
    hits = response[0] if response else []
    ids: list[str] = []
    for hit in hits:
        entity = hit.get("entity", {})
        oid = entity.get("object_id") or hit.get("object_id")
        if oid in objects and oid not in ids:
            ids.append(oid)
    scored = {
        oid: core.score_object(query, objects[oid]["normalized"])
        for oid in ids
    }
    return ids[: plan["top_k"]], scored, {
        "filter": expression,
        "ann_hit_count": len(ids),
        "nprobe": nprobe,
        "post_ann_select": False,
        "facet_fields_available": False,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": len(values),
        "mean_ms": round(float(statistics.fmean(values)), 4),
        "p50_ms": round(percentile(values, 50), 4),
        "p95_ms": round(percentile(values, 95), 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
    }


def timed_samples(
    fn: Callable[[], Any], warmup: int, repeats: int
) -> tuple[Any, list[float]]:
    for _ in range(warmup):
        fn()
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    values = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = fn()
        values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return result, values


def mean_metric(
    rows: list[dict[str, Any]], method: str, metric: str, query_ids: set[str] | None = None
) -> float | None:
    values = [
        row["methods"][method]["metrics"][metric]
        for row in rows
        if (query_ids is None or row["id"] in query_ids)
        and row["methods"][method]["metrics"].get(metric) is not None
    ]
    return round(float(np.mean(values)), 6) if values else None


def aggregate_method(
    rows: list[dict[str, Any]],
    method: str,
    all_latency: list[float],
    query_ids: set[str] | None = None,
) -> dict[str, Any]:
    validity_metric = (
        "raw_hard_condition_validity"
        if method == "milvus_entity_only"
        else "hard_condition_validity"
    )
    return {
        "query_count": sum(1 for row in rows if query_ids is None or row["id"] in query_ids),
        "mean_objective_ratio": mean_metric(rows, method, "objective_ratio", query_ids),
        "mean_oracle_object_recall": mean_metric(rows, method, "oracle_object_recall", query_ids),
        "mean_hard_condition_validity": mean_metric(rows, method, validity_metric, query_ids),
        "mean_fill_rate": mean_metric(rows, method, "fill_rate", query_ids),
        "latency": latency_summary(all_latency),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from pymilvus import MilvusClient

    core.self_test_select()
    with args.hdmg_cache.open("rb") as handle:
        cache = pickle.load(handle)
    with args.directory_cache.open("rb") as handle:
        directory = pickle.load(handle)
    with args.source_cache.open("rb") as handle:
        source = pickle.load(handle)
    query_doc = json.loads(args.intents.read_text(encoding="utf-8"))
    objects = retrieval.make_object_map(cache)
    query_images = image_pipeline.choose_test_queries(
        query_doc["records"], source, set(objects)
    )
    if set(row["object_id"] for row in query_images.values()) & set(objects):
        raise AssertionError("query/gallery leakage detected")

    entity_manifest = build_entity_only_collection(
        args.entity_milvus_uri,
        args.entity_collection,
        objects,
        args.milvus_nlist,
    )
    entity_client = MilvusClient(str(args.entity_milvus_uri.resolve()))
    entity_client.load_collection(args.entity_collection)
    facet_client = MilvusClient(str(args.facet_milvus_uri.resolve()))
    facet_client.load_collection(args.facet_collection)

    rows: list[dict[str, Any]] = []
    latency_all: dict[str, list[float]] = {
        "violas": [],
        "milvus_entity_only": [],
        "milvus_entity_facet_select": [],
    }
    for record in query_doc["records"]:
        plan = core.materialize_plan_preferences(
            core.resolve_plan(record["intent"]), directory
        )
        query_image = query_images[record["id"]]
        query = query_image["normalized"]
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

        violas_fn = lambda: core.interleaved_search(
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
        entity_fn = lambda: entity_only_search(
            entity_client,
            args.entity_collection,
            query,
            plan,
            objects,
            args.milvus_nprobe,
        )
        facet_fn = lambda: core.milvus_search(
            facet_client,
            args.facet_collection,
            query,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )

        (violas_ids, violas_scored, violas_stats), violas_samples = timed_samples(
            violas_fn, args.warmup, args.repeats
        )
        (entity_ids, entity_scored, entity_stats), entity_samples = timed_samples(
            entity_fn, args.warmup, args.repeats
        )
        (facet_ids, facet_scored, facet_stats), facet_samples = timed_samples(
            facet_fn, args.warmup, args.repeats
        )
        sample_map = {
            "violas": violas_samples,
            "milvus_entity_only": entity_samples,
            "milvus_entity_facet_select": facet_samples,
        }
        query_latency = {}
        for name, samples in sample_map.items():
            latency_all[name].extend(samples)
            query_latency[name] = latency_summary(samples)

        # Entity-only does not execute facet predicates.  The evaluator drops
        # invalid objects without replacement, making the quality penalty and
        # lost fill explicit while keeping retrieval truly facet-blind.
        entity_valid_ids = [
            oid
            for oid in entity_ids
            if core.satisfies(objects[oid]["description"], plan)
        ]
        entity_metrics = core.method_metrics(
            entity_valid_ids,
            entity_scored,
            oracle_ids,
            oracle_value,
            plan,
            objects,
            eligible_count,
        )
        entity_metrics["raw_hard_condition_validity"] = round(
            len(entity_valid_ids) / len(entity_ids), 6
        ) if entity_ids else None
        entity_metrics["raw_result_count"] = len(entity_ids)
        entity_metrics["evaluator_valid_count"] = len(entity_valid_ids)

        methods = {
            "oracle": {
                "results": core.result_rows(oracle_ids, oracle_scored, objects),
                "objective": round(oracle_value, 6),
            },
            "violas": {
                "results": core.result_rows(violas_ids, violas_scored, objects),
                "metrics": core.method_metrics(
                    violas_ids,
                    violas_scored,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
                "stats": violas_stats,
            },
            "milvus_entity_only": {
                "results": core.result_rows(entity_ids, entity_scored, objects),
                "evaluator_valid_results": core.result_rows(
                    entity_valid_ids, entity_scored, objects
                ),
                "metrics": entity_metrics,
                "stats": entity_stats,
            },
            "milvus_entity_facet_select": {
                "results": core.result_rows(facet_ids, facet_scored, objects),
                "metrics": core.method_metrics(
                    facet_ids,
                    facet_scored,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
                "stats": facet_stats,
            },
        }

        for name in sample_map:
            methods[name]["latency"] = query_latency[name]

        rows.append(
            {
                **record,
                "query_image": {
                    key: value
                    for key, value in query_image.items()
                    if key != "normalized"
                },
                "resolved_plan": plan,
                "has_hard_facet_predicate": bool(plan["hard_filters"]),
                "eligible_count": eligible_count,
                "methods": methods,
            }
        )
        print(
            f"DONE {record['id']}: hard={bool(plan['hard_filters'])} "
            f"objective V={methods['violas']['metrics']['objective_ratio']:.3f} "
            f"E={methods['milvus_entity_only']['metrics']['objective_ratio']:.3f} "
            f"F={methods['milvus_entity_facet_select']['metrics']['objective_ratio']:.3f}; "
            f"p50ms V={query_latency['violas']['p50_ms']:.2f} "
            f"E={query_latency['milvus_entity_only']['p50_ms']:.2f} "
            f"F={query_latency['milvus_entity_facet_select']['p50_ms']:.2f}",
            flush=True,
        )

    supported_ids = {
        row["id"] for row in rows if not row["has_hard_facet_predicate"]
    }
    group1 = {
        "definition": "Milvus physical schema has entity+vector only; entity filter + raw ANN top-k; no facet filter and no SELECT.",
        "all_10_queries": {
            "violas": aggregate_method(rows, "violas", latency_all["violas"]),
            "milvus_entity_only": aggregate_method(
                rows, "milvus_entity_only", latency_all["milvus_entity_only"]
            ),
            "entity_only_quality_note": "For hard-facet queries, invalid raw results are dropped by the evaluator without replacement; no facet-aware retrieval or reranking is added.",
        },
        "common_capability_subset_no_hard_facet": {
            "query_ids": sorted(supported_ids),
            "violas": aggregate_method(
                rows, "violas", latency_all["violas"], supported_ids
            ),
            "milvus_entity_only": aggregate_method(
                rows,
                "milvus_entity_only",
                latency_all["milvus_entity_only"],
                supported_ids,
            ),
            "latency_note": "Latency distributions contain all 10 queries; quality is restricted to the listed common-capability queries.",
        },
    }
    group2 = {
        "definition": "Milvus entity+facet hard filter + ANN candidate pool + the same exact soft-quota SELECT as Oracle/Violas.",
        "all_10_queries": {
            "violas": aggregate_method(rows, "violas", latency_all["violas"]),
            "milvus_entity_facet_select": aggregate_method(
                rows,
                "milvus_entity_facet_select",
                latency_all["milvus_entity_facet_select"],
            ),
        },
    }
    entity_client.close()
    facet_client.close()
    return {
        "format": "violas-milvus-capability-ablation-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_contract": {
            "query_count": len(rows),
            "relevance_input": "cached CLIP ViT-B/32 embedding of frozen Caltech-101 test image",
            "intent_input": "cached Qwen3.5 strict JSON intent",
            "gallery": "Caltech-101 train split, 6,984 objects",
            "query_gallery_overlap": 0,
            "excluded_from_retrieval_latency": [
                "Qwen3.5 intent generation",
                "CLIP query-image encoding",
                "one-time index loading/build",
            ],
            "included_in_retrieval_latency": "filter, ANN/HDMG navigation, scoring, and SELECT when the method defines SELECT",
        },
        "config": {
            "execution_profile": "paper",
            "top_k": 3,
            "warmup_per_query_method": args.warmup,
            "timed_repeats_per_query_method": args.repeats,
            "milvus_nlist": args.milvus_nlist,
            "milvus_nprobe": args.milvus_nprobe,
            "milvus_facet_candidate_k": args.milvus_candidate_k,
            "violas": {
                "nav_budget": args.nav_budget,
                "score_budget": args.score_budget,
                "beam_width": args.beam_width,
                "batch_size": args.batch_size,
            },
            "entity_only_manifest": entity_manifest,
        },
        "experiment_1_entity_only": group1,
        "experiment_2_entity_facet_same_select": group2,
        "queries": rows,
    }


def image_uri(local_root: str, object_id: str) -> str:
    from artifact_paths import image_uri as resolve_image_uri
    return resolve_image_uri(local_root, object_id)


def f(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def metric_table(title: str, block: dict[str, Any]) -> str:
    methods = [(key, value) for key, value in block.items() if isinstance(value, dict) and "query_count" in value]
    rows = []
    for name, data in methods:
        latency = data["latency"]
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(name)}</b></td>"
            f"<td>{data['query_count']}</td>"
            f"<td>{f(data['mean_objective_ratio'], 4)}</td>"
            f"<td>{f(data['mean_oracle_object_recall'], 4)}</td>"
            f"<td>{f(data['mean_hard_condition_validity'], 4)}</td>"
            f"<td>{f(data['mean_fill_rate'], 4)}</td>"
            f"<td>{f(latency['p50_ms'], 2)}</td>"
            f"<td>{f(latency['p95_ms'], 2)}</td>"
            "</tr>"
        )
    return f"""
    <section class="panel"><h2>{html.escape(title)}</h2>
      <table><thead><tr><th>method</th><th>queries</th><th>objective / Oracle</th><th>Oracle object recall</th><th>hard validity</th><th>fill rate</th><th>latency p50 ms</th><th>latency p95 ms</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
    </section>"""


def render_html(report: dict[str, Any], local_root: str) -> str:
    exp1 = report["experiment_1_entity_only"]
    exp2 = report["experiment_2_entity_facet_same_select"]
    query_rows = []
    for row in report["queries"]:
        methods = row["methods"]
        eo = methods["milvus_entity_only"]
        query_rows.append(
            f"""<tr>
            <td><img class="thumb" src="{image_uri(local_root, row['query_image']['object_id'])}"><b>{row['id']}</b><br><small>{html.escape(row['natural_query'])}</small></td>
            <td>{html.escape(row['resolved_plan']['entity_scope'])}<br><small>hard={html.escape(json.dumps(row['resolved_plan']['hard_filters'], ensure_ascii=False))}<br>facet={html.escape(str(row['resolved_plan']['facet_dimension']))}</small></td>
            <td>{f(methods['violas']['metrics']['objective_ratio'])} / {f(methods['violas']['metrics']['oracle_object_recall'])}<br><small>{f(methods['violas']['latency']['p50_ms'],2)} ms</small></td>
            <td>{f(eo['metrics']['objective_ratio'])} / {f(eo['metrics']['oracle_object_recall'])}<br><small>{f(eo['metrics']['raw_hard_condition_validity'])} valid · {f(eo['latency']['p50_ms'],2)} ms</small></td>
            <td>{f(methods['milvus_entity_facet_select']['metrics']['objective_ratio'])} / {f(methods['milvus_entity_facet_select']['metrics']['oracle_object_recall'])}<br><small>{f(methods['milvus_entity_facet_select']['latency']['p50_ms'],2)} ms</small></td>
            </tr>"""
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Violas vs Milvus capability ablation</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#172033;--muted:#657186;--line:#dce3ef;--blue:#315fe8;--green:#16794b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Microsoft YaHei",sans-serif}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}h2{{font-size:19px;margin:0 0 12px}}.lede{{color:var(--muted);max-width:1000px;line-height:1.65}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 5px 20px #20304a0a}}.tag{{display:inline-block;background:#eaf0ff;color:var(--blue);border-radius:999px;padding:5px 9px;margin:3px;font-size:12px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}.thumb{{width:74px;height:55px;object-fit:contain;background:#edf0f5;border-radius:6px;float:left;margin-right:8px}}small{{color:var(--muted);line-height:1.5}}code{{background:#edf1f7;padding:2px 5px;border-radius:5px}}.flow{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.flow>div{{border:1px solid var(--line);border-radius:10px;padding:13px}}.flow b{{color:var(--blue)}}@media(max-width:850px){{.flow{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}
</style></head><body><main>
<h1>Violas vs Milvus：两种能力边界实验</h1>
<p class="lede">两组实验共享完全相同的 10 个自然语言 intent、测试图 embedding、训练集 gallery 与 Top-3 目标。Qwen 和 CLIP 在计时前完成；延迟只计算检索后端。每个 query/method 预热 {report['config']['warmup_per_query_method']} 次、计时 {report['config']['timed_repeats_per_query_method']} 次。</p>
<div><span class="tag">objective ratio = 方法集合目标 / 全遍历 Oracle 目标</span><span class="tag">object recall = 与 Oracle Top-3 重合数 / 3</span><span class="tag">latency = filter + search + method-defined SELECT</span></div>
<section class="panel"><h2>能力定义</h2><div class="flow"><div><b>实验 1 · Entity-only Milvus</b><p>物理 schema 只有 <code>entity + vector</code>。执行 entity filter 后直接返回 ANN Top-3，不读取 facet、不运行 same SELECT。facet 仅由离线 evaluator 用于核验质量；不满足 hard facet 的结果直接判无效且不补位。</p></div><div><b>实验 2 · Entity+Facet+same SELECT</b><p>Milvus 使用 entity 与 hard facet filter，取 80 个 ANN candidates，再运行与 Oracle/Violas 相同的 exact soft-quota SELECT。与当前完整 bench 一致，主要观察相同语义能力下的检索延迟。</p></div></div></section>
{metric_table('实验 1 · 全部 10 条（hard-facet 不支持造成的损失也计入）', exp1['all_10_queries'])}
{metric_table('实验 1 补充 · 共同能力子集（6 条无 hard-facet query）', exp1['common_capability_subset_no_hard_facet'])}
{metric_table('实验 2 · 全部 10 条（Entity+Facet+same SELECT）', exp2['all_10_queries'])}
<section class="panel"><h2>逐 query 结果</h2><table><thead><tr><th>query</th><th>plan</th><th>Violas<br>objective / recall / p50</th><th>Milvus Entity-only<br>objective / recall / validity / p50</th><th>Milvus Entity+Facet+SELECT<br>objective / recall / p50</th></tr></thead><tbody>{''.join(query_rows)}</tbody></table></section>
<section class="panel"><h2>复现实验配置</h2><pre>{html.escape(json.dumps({'query_contract': report['query_contract'], 'config': report['config']}, ensure_ascii=False, indent=2))}</pre></section>
</main></body></html>"""


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    html_path = args.output_dir / "violas_vs_milvus_capability_ablation.html"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_html(report, args.local_dataset_root), encoding="utf-8")
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
