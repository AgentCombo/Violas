#!/usr/bin/env python3
"""Compare exact prefiltering with embedding top-10 followed by filtering."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
from pymilvus import MilvusClient
from qdrant_client import models

import render_soft_entity_report as html
import run_soft_entity_bench as soft
from label_caltech_local_mvp import OUT as MVP, write
from run_caltech_local_mvp import IMAGE_ROOT
from run_caltech_qwen50_bench import read


OUT = MVP / "filter_order"
DBS = ("milvus", "qdrant", "chroma")
LABELS = {"vg_hdmg": "VG + HDMG", "vg_rep": "VG − HDMG",
          "milvus": "Milvus", "qdrant": "Qdrant", "chroma": "Chroma"}
MODES = {"pre": "前过滤", "post": "后过滤"}


def fetch(db, client, config, corpus, query, plan, pre):
    tag = config.index_tag
    entity = plan["required_entity"]
    tokens = [f"{r['key']}={r['value']}" for r in plan["hard_filters"]]
    if db == "milvus":
        collection = f"{tag}_array"
        fields = ["object_id", "vector"]
        if pre:
            clauses = [f"entity == {json.dumps(entity)}"]
            if tokens:
                clauses.append(f"ARRAY_CONTAINS_ALL(facet_tokens, {json.dumps(tokens)})")
            rows = client.query(collection, filter=" and ".join(clauses),
                                output_fields=fields, limit=len(corpus.ids))
        else:
            response = client.search(collection, data=[query.tolist()], anns_field="vector",
                                     limit=config.candidate_k, output_fields=fields,
                                     search_params={"metric_type": "COSINE", "params": {"nprobe": config.nprobe}})
            rows = [hit["entity"] for hit in response[0]]
        return [(row["object_id"], row["vector"]) for row in rows]
    if db == "qdrant":
        collection = f"soft_{tag}"
        if pre:
            conditions = [models.FieldCondition(key="entity", match=models.MatchValue(value=entity))]
            conditions.extend(models.FieldCondition(key="facet_tokens", match=models.MatchValue(value=token))
                              for token in tokens)
            query_filter = models.Filter(must=conditions)
            points, offset = [], None
            while True:
                page, offset = client.scroll(collection, scroll_filter=query_filter, offset=offset,
                                             limit=len(corpus.ids), with_payload=["object_id"], with_vectors=True)
                points.extend(page)
                if offset is None:
                    break
        else:
            response = client.query_points(collection, query=query.tolist(), limit=config.candidate_k,
                                           search_params=models.SearchParams(hnsw_ef=config.ef),
                                           with_payload=["object_id"], with_vectors=True)
            points = response.points
        return [(point.payload["object_id"], point.vector) for point in points]
    if pre:
        clauses = [{"entity": {"$eq": entity}}]
        clauses.extend({f"facet__{r['key']}": {"$eq": r["value"]}} for r in plan["hard_filters"])
        where = {"$and": clauses} if len(clauses) > 1 else clauses[0]
        response = client.get(where=where, include=["embeddings"])
        if not response["ids"]:
            return []
        return list(zip(response["ids"], response["embeddings"], strict=True))
    response = client.query(query_embeddings=[query.tolist()], n_results=config.candidate_k,
                            include=["embeddings"])
    return list(zip(response["ids"][0], response["embeddings"][0], strict=True))


def select_fetched(fetched, corpus, query, plan, pre):
    indices = [corpus.locator[oid] for oid, _ in fetched]
    if pre:
        valid = list(range(len(indices)))
    else:
        valid = [pos for pos, index in enumerate(indices) if corpus.eligible(index, plan)]
    scores = {}
    if valid:
        vectors = np.asarray([fetched[pos][1] for pos in valid], dtype=np.float32)
        similarities = np.einsum("ij,j->i", vectors, query)
        keys = corpus.entity_ids[[indices[pos] for pos in valid]]
        values = corpus.mixed_values(similarities, keys, plan, corpus.semantic_scores(plan))
        scores = {fetched[pos][0]: float(value) for pos, value in zip(valid, values, strict=True)}
    return soft.select(scores, corpus, plan), scores, {
        "candidate_count": len(indices), "unique_member_scores": len(scores),
    }


def search_cases(config, corpus, source, rows, clients):
    bindings = soft.bench.choose_queries(rows, source, set(corpus.ids))
    methods = [(name, "vg") for name in ("vg_hdmg", "vg_rep")] + [
        (db, mode) for mode in MODES for db in DBS
    ]
    results = []
    for number, row in enumerate(rows):
        bound = bindings[row["id"]]
        if bound["object_id"] != row["query_object"]["object_id"]:
            raise ValueError(f"query binding changed for {row['id']}")
        query = bound["normalized"]
        plan = {**row["resolved_plan"], "required_entity": row["resolved_plan"]["entity_preference"],
                "entity_mode": "hard", "search_scope": "required_entity"}
        raw = np.asarray(source["test_data"][row["entity_hint"]]["vectors"][bound["test_index"]],
                         dtype=np.float32).reshape(-1)
        measured = {}
        shift = number % len(methods)
        for name, mode in methods[shift:] + methods[:shift]:
            start = perf_counter_ns()
            if mode == "vg":
                if name == "vg_hdmg":
                    ids, scores, stats = soft.group_search(corpus, query, plan, config, graph=True)
                else:
                    ids, scores, stats = soft.representative_scan(
                        corpus, raw, plan, config.group_budget, online_cosine=True)
                fetched = None
            else:
                fetched = fetch(name, clients[name], config, corpus, query, plan, pre=mode == "pre")
                fetched_at = perf_counter_ns()
                ids, scores, stats = select_fetched(fetched, corpus, query, plan, pre=mode == "pre")
            stop = perf_counter_ns()
            if fetched is not None:
                stats.update({"candidate_ids": [oid for oid, _ in fetched],
                              "fetch_ms": (fetched_at - start) / 1e6, "select_ms": (stop - fetched_at) / 1e6})
            measured[(name, mode)] = (ids, scores, stats, (stop - start) / 1e6)

        # Independently enumerate the full gallery under the same two hard predicates.
        eligible = [index for index in range(len(corpus.ids)) if corpus.eligible(index, plan)]
        oracle_scores = corpus.scores(eligible, query, plan, corpus.semantic_scores(plan))
        oracle_ids = soft.select(oracle_scores, corpus, plan)
        oracle_value = soft.core.objective(oracle_ids, oracle_scores, corpus.objects, plan["facet_dimension"],
                                           plan["lambda"], plan["category_preferences"])
        entries = {}
        for (name, mode), (ids, scores, stats, ms) in measured.items():
            entry = soft.result(ids, scores, stats, ms, oracle_ids, oracle_value, plan, corpus, len(eligible))
            if entry["metrics"]["hard_condition_validity"] != 1:
                raise ValueError(f"invalid result for {row['id']} / {name} / {mode}")
            if mode == "pre" and (len(stats["candidate_ids"]) != len(eligible)
                                  or scores.keys() != oracle_scores.keys() or set(ids) != set(oracle_ids)
                                  or not math.isclose(entry["metrics"]["objective_ratio"], 1.0, abs_tol=2e-6)):
                raise ValueError(f"prefilter failed exact parity for {row['id']} / {name}")
            if mode == "post" and stats["candidate_count"] != config.candidate_k:
                raise ValueError(f"postfilter must recall {config.candidate_k} candidates for {row['id']} / {name}")
            entries[name if mode == "vg" else f"{mode}_{name}"] = entry
        results.append({**row, "resolved_plan": plan, "eligible_count": len(eligible), "methods": entries,
                        "oracle": {"results": soft.core.result_rows(oracle_ids, oracle_scores, corpus.objects),
                                   "objective": oracle_value}})
    return results


def summarize(rows):
    summary = {}
    for name in rows[0]["methods"]:
        entries = [row["methods"][name] for row in rows]
        summary[name] = {
            "objective_ratio": soft.bench.mean_metric(rows, name, "objective_ratio"),
            "object_recall": soft.bench.mean_metric(rows, name, "oracle_object_recall"),
            "latency": soft.bench.latency([e["latency"]["mean_ms"] for e in entries]),
            "mean_scored": fmean(e["stats"]["unique_member_scores"] for e in entries),
            "underfilled_queries": sum(len(e["results"]) < min(3, row["eligible_count"])
                                       for e, row in zip(entries, rows, strict=True)),
            "empty_queries": sum(not e["results"] for e in entries),
        }
    return summary


def render(report, mode):
    names = {name: name if name.startswith("vg_") else f"{mode}_{name}" for name in LABELS}
    protocol = ("先按 entity 和全部硬 facet 枚举所有合格对象，对它们打分并执行完整 SELECT。"
                if mode == "pre" else
                "数据库只按全库 embedding 距离取前 10，再在候选内检查 entity 和硬 facet，并执行完整 SELECT。")
    other = "post" if mode == "pre" else "pre"
    sections = [f"<header><h1>Caltech-101 · {MODES[mode]}</h1><p>{protocol}</p>"
                "<p>25 条测试图 + intent，1,840 张训练图。同一个 entity + 硬 facet 穷举 Oracle。"
                "多样性是 SELECT 的软覆盖奖励。两个 VG 版本使用原生检索，适用同一硬约束。</p>"
                "<p>计时含数据库返回 ID 和向量、应用侧打分和 SELECT。两组同轮交替测量，每条每方法一次，无主动预热。</p>"
                f"<p><a href='{other}filter.html'>查看{MODES[other]}</a> · <a href='results.json'>原始结果</a></p></header>"]
    table = []
    for name, key in names.items():
        metrics = report["aggregate"][key]
        latency = metrics["latency"]
        table.append(f"<tr><td>{LABELS[name]}</td><td>{html.fmt(metrics['objective_ratio'])}</td>"
                     f"<td>{html.fmt(metrics['object_recall'])}</td><td>{latency['mean_ms']:.3f}</td>"
                     f"<td>{latency['p50_ms']:.3f}</td><td>{latency['p95_ms']:.3f}</td>"
                     f"<td>{metrics['mean_scored']:.1f}</td><td>{metrics['underfilled_queries']}</td>"
                     f"<td>{metrics['empty_queries']}</td></tr>")
    sections.append("<section><h2>25 条查询汇总</h2><div class='table'><table><thead><tr><th>方法</th>"
                    "<th>Objective ratio</th><th>Object recall</th><th>Mean ms</th><th>p50 ms</th><th>p95 ms</th>"
                    "<th>平均打分对象</th><th>未凑满的查询</th><th>零结果</th></tr></thead><tbody>"
                    + "".join(table) + "</tbody></table></div></section>")
    for row in report["rows"]:
        cards = []
        for name, key in {**names, "oracle": "oracle"}.items():
            entry = row["oracle"] if name == "oracle" else row["methods"][key]
            if name == "oracle":
                note = f"全库合格 {row['eligible_count']} 个"
            else:
                stats = entry["stats"]
                note = (f"Objective ratio {html.fmt(entry['metrics']['objective_ratio'])} · "
                        f"打分 {stats['unique_member_scores']} 个 · {entry['latency']['mean_ms']:.3f} ms")
                if "candidate_count" in stats:
                    note += f" · 召回 {stats['candidate_count']} 个"
            photos = "".join(html.result_card(item, rank, report, IMAGE_ROOT, "相关性")
                             for rank, item in enumerate(entry["results"], 1))
            cards.append(f"<section><h3>{LABELS.get(name, '全库 Oracle')}</h3><p>{note}</p>"
                         f"{photos or '<p>零结果</p>'}</section>")
        intent = html.esc(json.dumps(row["resolved_plan"], indent=2, ensure_ascii=False))
        sections.append(f"<details><summary>{html.query_media(row, report, IMAGE_ROOT)}"
                        f"<b>{html.esc(row['id'])} · {html.esc(row['entity_hint'])}</b></summary>"
                        f"<p>{html.esc(row['english'])}</p><pre>{intent}</pre>"
                        f"<div class='methods'>{''.join(cards)}</div></details>")
    (OUT / f"{mode}filter.html").write_text(html.page(MODES[mode], "".join(sections)), encoding="utf-8")


def main():
    previous = read(MVP / "benchmark/caltech101/results.json")
    workload = read(MVP / "frozen_queries.json")
    config = SimpleNamespace(**previous["config"])
    if len(workload["rows"]) != 25 or config.candidate_k != 10:
        raise ValueError("expected 25 frozen queries and a postfilter candidate cap of 10")
    source, cache, directory = [soft.bench.load_pickle(Path(path), config.module_path)
                                for path in (config.source_cache, config.hdmg_cache, config.directory_cache)]
    corpus = soft.Corpus(cache, directory)
    if len(corpus.ids) != 1840:
        raise ValueError("expected the original 25-category gallery")
    qdrant, qinfo = soft.setup_qdrant(config, corpus)
    tag = config.index_tag
    milvus = MilvusClient(str(Path(f"cache/five_system/{tag}_milvus_array.db").resolve()))
    try:
        milvus.load_collection(f"{tag}_array")
        _, chroma = soft.backends.open_chroma(Path(f"cache/five_system/{tag}_chroma"), f"{tag}_chroma")
        clients = {"milvus": milvus, "qdrant": qdrant, "chroma": chroma}
        rows = search_cases(config, corpus, source, workload["rows"], clients)
    finally:
        milvus.close()
        qdrant.close()
    report = {"dataset": config.dataset, "modality": workload["modality"],
              "config": {"candidate_k": config.candidate_k, "gallery_objects": len(corpus.ids),
                         "index_tag": tag, "nprobe": config.nprobe, "ef": config.ef, "qdrant": qinfo,
                         "measurement": "rotating order, once per query/method, no explicit warmup",
                         "oracle_scope": "exact entity and hard facet predicates",
                         "result_fields": ["object_id", "vector"]},
              "aggregate": summarize(rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "results.json", report)
    for mode in MODES:
        render(report, mode)
    print("COMPLETE prefilter parity verified and postfilter evaluated for all 25 queries", flush=True)


if __name__ == "__main__":
    main()
