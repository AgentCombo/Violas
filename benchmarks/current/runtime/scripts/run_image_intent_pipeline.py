#!/usr/bin/env python3
"""Run ten image+language intent queries through the guideline pipeline.

The query image comes from the frozen Caltech-101 test split and supplies the
CLIP image embedding.  Natural language is parsed separately and supplies the
entity scope, hard predicates, one diversity dimension, lambda tier, and k.
The indexed gallery remains the 6,984-object training split.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_caltech_kv_retrieval as retrieval
import run_guideline_pipeline as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intents",
        type=Path,
        default=Path("queries/caltech101_guideline_10_qwen.json"),
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
        "--milvus-uri",
        type=Path,
        default=Path("cache/caltech101_kv_milvus_ivf.db"),
    )
    parser.add_argument("--collection", default="caltech101_kv_train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/caltech10_image_intent_paper_pipeline"),
    )
    parser.add_argument("--nav-budget", type=int, default=12)
    parser.add_argument("--score-budget", type=int, default=80)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--execution-profile",
        choices=("paper", "subset-v2"),
        default="paper",
    )
    parser.add_argument("--milvus-candidate-k", type=int, default=80)
    parser.add_argument("--milvus-nprobe", type=int, default=16)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="experiment-level override for the intent's requested top_k",
    )
    parser.add_argument(
        "--local-dataset-root",
        default=r"dataset/caltech-101/101_ObjectCategories",
    )
    return parser.parse_args()


def relative_object_id(img_path: str) -> str:
    normalized = img_path.replace("\\", "/")
    marker = "101_ObjectCategories/"
    if marker not in normalized:
        raise ValueError(f"unexpected Caltech path: {img_path}")
    return normalized.split(marker, 1)[1]


def choose_test_queries(
    records: list[dict[str, Any]],
    source: dict[str, Any],
    gallery_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Choose deterministic, non-overlapping test images per entity occurrence."""
    used_index: dict[str, int] = defaultdict(int)
    chosen: dict[str, dict[str, Any]] = {}
    for record in records:
        entity = record["intent"]["entity_scope"]
        split = source["test_data"][entity]
        start = used_index[entity]
        selected = None
        for offset in range(len(split["vectors"])):
            index = (start + offset) % len(split["vectors"])
            description = split["descriptions"][index]
            object_id = relative_object_id(description["img_path"])
            if object_id in gallery_ids:
                continue
            vector = np.asarray(split["vectors"][index], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                continue
            selected = {
                "test_index": index,
                "object_id": object_id,
                "img_path": description["img_path"],
                "source_idx": description.get("idx"),
                "normalized": vector / norm,
            }
            used_index[entity] = index + 1
            break
        if selected is None:
            raise RuntimeError(f"no usable test query image for {record['id']} / {entity}")
        chosen[record["id"]] = selected
    return chosen


def run_benchmark(
    args: argparse.Namespace,
    query_doc: dict[str, Any],
    cache: dict[str, Any],
    directory: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    from pymilvus import MilvusClient

    objects = retrieval.make_object_map(cache)
    query_images = choose_test_queries(query_doc["records"], source, set(objects))
    if set(row["object_id"] for row in query_images.values()) & set(objects):
        raise AssertionError("query/gallery leakage detected")
    client = MilvusClient(str(args.milvus_uri.resolve()))
    client.load_collection(args.collection)
    rows: list[dict[str, Any]] = []
    for record in query_doc["records"]:
        plan = core.materialize_plan_preferences(
            core.resolve_plan(record["intent"]), directory
        )
        if args.top_k is not None:
            if args.top_k < 0:
                raise ValueError("--top-k must be non-negative")
            plan["top_k"] = args.top_k
        query_image = query_images[record["id"]]
        query_vector = query_image["normalized"]
        eligible_count = len(core.eligible_ids(objects, plan))

        started = time.perf_counter()
        oracle_ids, oracle_scored = core.exhaustive_oracle(query_vector, plan, objects)
        oracle_ms = (time.perf_counter() - started) * 1000
        oracle_value = core.objective(
            oracle_ids,
            oracle_scored,
            objects,
            plan["facet_dimension"],
            plan["lambda"],
            plan["category_preferences"],
        )

        started = time.perf_counter()
        violas_ids, violas_scored, violas_stats = core.interleaved_search(
            query_vector,
            plan,
            cache,
            directory,
            objects,
            args.nav_budget,
            args.score_budget,
            args.beam_width,
            args.batch_size,
            args.execution_profile,
        )
        violas_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        milvus_ids, milvus_scored, milvus_stats = core.milvus_search(
            client,
            args.collection,
            query_vector,
            plan,
            cache,
            objects,
            args.milvus_candidate_k,
            args.milvus_nprobe,
        )
        milvus_ms = (time.perf_counter() - started) * 1000

        methods: dict[str, Any] = {}
        for name, ids, scored, elapsed, stats in (
            ("oracle", oracle_ids, oracle_scored, oracle_ms, {"scored_objects": len(oracle_scored), "guarantee": "full-corpus exact"}),
            ("violas", violas_ids, violas_scored, violas_ms, violas_stats),
            ("milvus", milvus_ids, milvus_scored, milvus_ms, milvus_stats),
        ):
            methods[name] = {
                "latency_ms": round(elapsed, 3),
                "results": core.result_rows(ids, scored, objects),
                "metrics": core.method_metrics(
                    ids,
                    scored,
                    oracle_ids,
                    oracle_value,
                    plan,
                    objects,
                    eligible_count,
                ),
                "stats": stats,
            }
        rows.append(
            {
                **record,
                "query_mode": "test_image_plus_language_intent",
                "query_image": {
                    key: value
                    for key, value in query_image.items()
                    if key != "normalized"
                },
                "resolved_plan": plan,
                "eligible_count": eligible_count,
                "methods": methods,
            }
        )
        print(
            f"DONE {record['id']}: image={query_image['object_id']} "
            f"d={plan['facet_dimension']} lambda={plan['lambda']:.2f} "
            f"Violas={methods['violas']['metrics']['objective_ratio']:.3f} "
            f"Milvus={methods['milvus']['metrics']['objective_ratio']:.3f}",
            flush=True,
        )

    aggregate: dict[str, Any] = {}
    for name in ("violas", "milvus"):
        aggregate[name] = {
            "mean_objective_ratio": round(
                float(np.mean([row["methods"][name]["metrics"]["objective_ratio"] for row in rows])),
                6,
            ),
            "mean_oracle_object_recall": round(
                float(np.mean([row["methods"][name]["metrics"]["oracle_object_recall"] for row in rows])),
                6,
            ),
            "mean_latency_ms": round(
                float(np.mean([row["methods"][name]["latency_ms"] for row in rows])),
                3,
            ),
        }
    return {
        "format": "violas-image-intent-paper-pipeline-report-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "query_contract": {
            "relevance_input": "CLIP ViT-B/32 image embedding from frozen test split",
            "intent_input": "natural language compiled by Qwen3.5",
            "gallery": "Caltech-101 training split; 6984 objects",
            "leakage_check": "all query image IDs absent from gallery",
            "text_embedding_used_for_similarity": False,
        },
        "config": {
            "lambda_mapping": core.LAMBDA_BY_LEVEL,
            "nav_budget": args.nav_budget,
            "score_budget": args.score_budget,
            "beam_width": args.beam_width,
            "batch_size": args.batch_size,
            "execution_profile": args.execution_profile,
            "semantic_directory": directory["format"],
            "member_priority": "cell-level center-radius upper bound plus remaining quota reward",
            "semantic_entry_policy": "eligible member count, then cell ID",
            "member_order": "canonical object ID within each cell-category subset",
            "milvus_candidate_k": args.milvus_candidate_k,
            "milvus_nprobe": args.milvus_nprobe,
            "top_k_override": args.top_k,
        },
        "aggregate": aggregate,
        "queries": rows,
    }


def image_uri(local_root: str, object_id: str) -> str:
    from artifact_paths import image_uri as resolve_image_uri
    return resolve_image_uri(local_root, object_id)


def metric_badges(method: dict[str, Any]) -> str:
    metrics = method["metrics"]
    return (
        f'<span>objective <b>{metrics["objective_ratio"]:.3f}</b></span>'
        f'<span>similarity <b>{metrics["mean_embedding_similarity"]:.3f}</b></span>'
        f'<span>coverage <b>{metrics["known_category_coverage"]}</b></span>'
        f'<span>oracle recall <b>{metrics["oracle_object_recall"]:.3f}</b></span>'
    )


def result_cards(
    method: dict[str, Any],
    local_root: str,
    active_dimension: str | None,
) -> str:
    cards = []
    for rank, result in enumerate(method["results"], 1):
        chips = []
        for key, value in sorted(result["facet_values"].items()):
            active = " active" if key == active_dimension else ""
            chips.append(
                f'<span class="chip{active}">{html.escape(key)}={html.escape(str(value))}</span>'
            )
        cards.append(
            f"""
            <article class="result">
              <div class="rank">{rank}</div>
              <img src="{image_uri(local_root, result['object_id'])}" loading="lazy">
              <div><b>{html.escape(result['object_id'])}</b>
              <div class="score">image-image similarity {result['score']:.4f}</div>
              <div class="chips">{''.join(chips)}</div></div>
            </article>"""
        )
    return "".join(cards)


def render_html(report: dict[str, Any], local_root: str) -> str:
    sections = []
    for row in report["queries"]:
        plan = row["resolved_plan"]
        methods = row["methods"]
        method_columns = []
        for key, title in (
            ("oracle", "Full-corpus Oracle"),
            ("violas", "Interleaved Violas"),
            ("milvus", "Milvus ANN + same SELECT"),
        ):
            method = methods[key]
            method_columns.append(
                f"""
                <div class="method">
                  <h3>{title}</h3>
                  <div class="badges">{metric_badges(method)}</div>
                  {result_cards(method, local_root, plan['facet_dimension'])}
                  <details><summary>execution details</summary><pre>{html.escape(json.dumps(method['stats'], ensure_ascii=False, indent=2))}</pre></details>
                </div>"""
            )
        input_contract = {
            "query_image_embedding": row["query_image"]["object_id"],
            "language_intent": row["intent"],
            "resolved_plan": plan,
        }
        sections.append(
            f"""
            <section class="query">
              <div class="query-head">
                <div class="query-image"><div class="eyebrow">TEST QUERY IMAGE</div><img src="{image_uri(local_root, row['query_image']['object_id'])}"><b>{html.escape(row['query_image']['object_id'])}</b></div>
                <div class="query-text"><div class="eyebrow">NATURAL-LANGUAGE INTENT</div><h2>{row['id']} · {html.escape(row['natural_query'])}</h2>
                  <div class="plan"><b>scope</b> {html.escape(plan['entity_scope'])} · <b>must</b> {html.escape(json.dumps(plan['hard_filters'], ensure_ascii=False))} · <b>facet</b> {html.escape(str(plan['facet_dimension']))} · <b>λ</b> {plan['lambda']:.2f} · <b>k</b> {plan['top_k']}</div>
                  <div class="contract"><span>image → relevance embedding</span><span>language → intent only</span><span>gallery overlap = 0</span></div>
                  <details><summary>完整输入合同</summary><pre>{html.escape(json.dumps(input_contract, ensure_ascii=False, indent=2))}</pre></details>
                </div>
              </div>
              <div class="methods">{''.join(method_columns)}</div>
            </section>"""
        )
    agg = report["aggregate"]
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Violas image + intent queries</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#66738a;--line:#dce3ef;--blue:#315fe8;--soft:#edf2ff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Microsoft YaHei",sans-serif}}main{{max-width:1540px;margin:auto;padding:26px}}h1{{margin:0 0 7px}}.subtitle{{color:var(--muted);margin-bottom:18px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}}.summary div{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px}}.summary b{{display:block;font-size:22px;color:var(--blue)}}.query{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:17px;margin:18px 0;box-shadow:0 5px 18px #20304a0a}}.query-head{{display:grid;grid-template-columns:210px 1fr;gap:16px;margin-bottom:14px}}.query-image{{background:#f2f4f8;border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:6px;font-size:11px;overflow-wrap:anywhere}}.query-image img{{width:100%;height:145px;object-fit:contain;background:#e8ecf2;border-radius:7px}}.eyebrow{{font-size:10px;letter-spacing:.08em;color:var(--muted);font-weight:700}}.query-text h2{{font-size:18px;margin:5px 0 10px}}.plan{{background:var(--soft);padding:9px;border-radius:7px;font-size:12px}}.contract{{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}}.contract span,.badges span{{font-size:11px;background:#f0f3f8;border-radius:999px;padding:4px 7px}}.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.method{{border:1px solid var(--line);border-radius:10px;padding:10px;min-width:0}}.method h3{{font-size:15px;margin:0 0 7px}}.badges{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px}}.result{{display:grid;grid-template-columns:22px 100px minmax(0,1fr);gap:8px;align-items:center;padding:8px 0;border-top:1px solid var(--line);font-size:12px}}.result img{{width:100px;height:76px;object-fit:contain;background:#eef1f5;border-radius:7px}}.rank{{color:var(--muted);font-weight:700}}.score{{color:var(--muted);margin:3px 0}}.chips{{display:flex;flex-wrap:wrap;gap:3px}}.chip{{font-size:9px;padding:2px 5px;border-radius:999px;background:#eef1f5;color:#4b5565}}.chip.active{{background:#dce7ff;color:var(--blue);font-weight:700}}details{{margin-top:8px}}summary{{cursor:pointer;color:var(--blue);font-size:12px}}pre{{background:#111827;color:#dbeafe;padding:10px;border-radius:7px;white-space:pre-wrap;overflow:auto;font-size:11px}}@media(max-width:1050px){{.methods,.summary{{grid-template-columns:1fr}}}}@media(max-width:680px){{.query-head{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Image + Natural-Language Intent · 10-query Pipeline</h1>
<div class="subtitle">Paper profile：测试图提供 CLIP image query embedding；自然语言产生 entity、predicate、facet 与 soft quota；HDMG 导航、成员评分和集合选择交错执行。所有返回图片来自独立 train/gallery split。</div>
<div class="summary"><div><b>10</b>test-image queries</div><div><b>0</b>query/gallery overlap</div><div><b>{agg['violas']['mean_objective_ratio']:.3f}</b>Violas mean objective ratio</div><div><b>{agg['milvus']['mean_objective_ratio']:.3f}</b>Milvus mean objective ratio</div></div>
<details open><summary>查询与评测合同</summary><pre>{html.escape(json.dumps({'query_contract': report['query_contract'], 'config': report['config']}, ensure_ascii=False, indent=2))}</pre></details>
{''.join(sections)}
</main></body></html>"""


def main() -> int:
    args = parse_args()
    core.self_test_select()
    with args.hdmg_cache.open("rb") as handle:
        cache = pickle.load(handle)
    with args.directory_cache.open("rb") as handle:
        directory = pickle.load(handle)
    with args.source_cache.open("rb") as handle:
        source = pickle.load(handle)
    query_doc = json.loads(args.intents.read_text(encoding="utf-8"))
    report = run_benchmark(args, query_doc, cache, directory, source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    html_name = (
        f"image_intent_10queries_top{args.top_k}.html"
        if args.top_k is not None
        else "image_intent_10.html"
    )
    html_path = args.output_dir / html_name
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report, args.local_dataset_root), encoding="utf-8")
    print(f"WROTE {json_path}", flush=True)
    print(f"WROTE {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
