#!/usr/bin/env python3
"""Run the 25-query Caltech local-facet MVP after the label job releases its lock."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from html import escape
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.optimize import linear_sum_assignment

from label_caltech_local_mvp import CATEGORIES, OUT, preprocess, write
import render_soft_entity_report as html
from rescore_vdb_coverage import score
from run_caltech_qwen50_bench import LABELS, SOURCE, read, run, soft


DRAFTS = OUT / "query_drafts.json"
SCHEMAS = OUT / "source_schemas.jsonl"
TYPE_COUNTS = {kind: 5 for kind in ("E", "H1", "H2", "D", "H+D")}
IMAGE_ROOT = r"dataset/caltech-101/101_ObjectCategories"


def load_drafts():
    contract = read(DRAFTS)
    if contract["vdb_candidate_k"] != 10 or contract["top_k"] != 3:
        raise ValueError("this workload requires a VDB candidate cap of 10 and top-3 results")
    drafts = contract["cases"]
    schemas = preprocess.load_schemas(SCHEMAS)
    if len(drafts) != 25 or set(d["entity"] for d in drafts) != set(CATEGORIES):
        raise ValueError("expected one drafted query per approved category")
    if Counter(d["query_type"] for d in drafts) != TYPE_COUNTS or len({d["id"] for d in drafts}) != 25:
        raise ValueError("expected five queries per intent type and unique query IDs")
    for draft in drafts:
        allowed = {facet["id"]: {value["id"] for value in facet["values"]}
                   for facet in schemas[draft["entity"]]["facets"]}
        hard, dimension, targets = draft["hard_filters"], draft["facet_dimension"], draft["target_values"]
        count = {"E": 0, "H1": 1, "H2": 2, "D": 0, "H+D": 1}[draft["query_type"]]
        if len(hard) != count or (dimension is not None) != (draft["query_type"] in {"D", "H+D"}):
            raise ValueError(f"query type does not match the constraints: {draft['id']}")
        if any(rule["op"] != "eq" or rule["value"] not in allowed.get(rule["key"], set()) for rule in hard):
            raise ValueError(f"unknown hard value: {draft['id']}")
        if dimension and (len(set(targets)) != 2 or not set(targets) <= allowed.get(dimension, set())):
            raise ValueError(f"invalid diversity targets: {draft['id']}")
    return contract


def feasibility(option, entity, grouped, objects):
    hard = option.get("hard_filters", [])
    dimension = option.get("facet_dimension")
    facets = [objects[oid]["description"]["facet_values"] for oid in grouped[entity]]
    matching = [values for values in facets if all(values.get(r["key"]) == r["value"] for r in hard)]
    target_counts = {value: sum(values.get(dimension) == value for values in matching)
                     for value in option.get("target_values", [])}
    return {"eligible_in_preferred_entity": len(matching), "target_counts": target_counts,
            "feasible": len(matching) >= 3 and all(target_counts.values())}


def freeze_workload(contract, objects, source, directory):
    drafts = contract["cases"]
    canonical = soft.bench.canonical
    grouped = canonical.entity_objects(objects)
    candidates = canonical.enumerate_candidates(objects, grouped)
    draft_checks = [feasibility(draft, draft["entity"], grouped, objects) for draft in drafts]
    costs = np.full((25, 25), np.inf)
    for i, draft in enumerate(drafts):
        for j, entity in enumerate(CATEGORIES):
            if entity == draft["entity"] and draft_checks[i]["feasible"]:
                costs[i, j] = 0
            elif candidates[entity][draft["query_type"]]:
                costs[i, j] = 1 if entity == draft["entity"] else 5
    slots, entities = linear_sum_assignment(costs)
    records = []
    for i, j in zip(slots.tolist(), entities.tolist(), strict=True):
        draft, entity = drafts[i], CATEGORIES[j]
        kind = draft["query_type"]
        if costs[i, j] == 0:
            option, english = draft, draft["english"]
        else:
            option = canonical.choose_candidate(candidates[entity][kind], i, kind in {"H1", "H2", "H+D"})
            english = f"Prefer images of {entity.replace('_', ' ')} similar to the attached image."
            for rule in option.get("hard_filters", []):
                english += f" Require {rule['key'].replace('_', ' ')} to be {rule['value'].replace('_', ' ')}."
            if option.get("facet_dimension"):
                targets = " and ".join(value.replace("_", " ") for value in option["target_values"])
                english += f" Try to cover both {targets} for {option['facet_dimension'].replace('_', ' ')}."
        checked = feasibility(option, entity, grouped, objects)
        if not checked["feasible"]:
            raise ValueError(f"chosen query is infeasible: {draft['id']}")
        intent = {"entity_scope": entity, "hard_filters": option.get("hard_filters", []),
                  "facet_dimension": option.get("facet_dimension"),
                  "diversity_level": "balanced" if option.get("facet_dimension") else "off",
                  "category_preferences": [{"value": v, "quota": 1, "weight_level": "balanced"}
                                           for v in option.get("target_values", [])],
                  "top_k": contract["top_k"], "embedding_text_en": entity.replace("_", " "), "unsupported": [],
                  "semantic_weight": contract["semantic_weight"], "embedding_weight": contract["embedding_weight"]}
        records.append({"id": draft["id"], "query_type": kind, "entity_hint": entity, "english": english,
                        "canonical_intent": intent, "feasibility": checked,
                        "draft_adjustment": {0: "none", 1: "facet", 5: "entity"}[costs[i, j]]})
    queries = soft.bench.choose_queries(records, source, set(objects))
    if len({q["object_id"] for q in queries.values()}) != 25:
        raise ValueError("expected 25 distinct held-out images")
    for i, row in enumerate(records):
        row["query_object"] = {k: v for k, v in queries[row["id"]].items() if k != "normalized"}
        row["resolved_plan"] = soft.make_plan(row, directory, i)
    workload = OUT / "frozen_queries.json"
    write(workload, {"dataset": "caltech101", "modality": "image", "rows": records})
    return workload


def render(report, workload, experiments=(1, 2)):
    entity_only = experiments == (3,)
    report_link = "benchmark_exp3/caltech101/results.json" if entity_only else "benchmark/caltech101/results.json"
    summary_file = "soft_entity_exp3_summary.html" if entity_only else "soft_entity_summary.html"
    sections = ["<header><h1>Caltech 局部视觉 Facet · 25 类 MVP</h1>"
                "<p>25 类，1,840 张训练图建库。25 条固定测试图 + 英文 intent。每条 VDB 最多召回 10 个候选，最终返回 top-3。</p>"
                "<p>原 objective ratio 按同一批 Qwen 标签求全库 Oracle。覆盖加权分使用 70% 相关性和 30% facet 覆盖率，无 diversity 时使用相关性。</p>"
                f"<p><a href='frozen_queries.json'>英文查询和执行计划</a> · <a href='{report_link}'>原始结果</a></p></header>"]
    if entity_only:
        sections.append("<p>完整 SELECT 在召回的 10 个候选内执行，合格对象少于 3 个时按实际数量返回。"
                        "两个 VG 版本使用原检索路径，在实验 3 同轮计时。<a href='soft_entity_summary.html'>实验 1 和 2</a></p>")
    labels = html.ENTITY_LABELS if entity_only else LABELS
    for exp in experiments:
        sections.append(html.summary_section(report, exp))
        lines = []
        for method, label in labels.items():
            name = f"exp{exp}_{method}"
            pairs = [(row, score(row["methods"][name], row["resolved_plan"])) for row in report["rows"]]
            with_diversity = [entry for _, entry in pairs if entry["coverage"] is not None]
            mean = report["aggregate"]["overall"][name]["latency"]["mean_ms"]
            lines.append(f"<tr><td>{escape(label)}</td><td>{fmean(entry['objective'] for _, entry in pairs):.4f}</td>"
                         f"<td>{fmean(entry['objective'] for entry in with_diversity):.4f}</td>"
                         f"<td>{fmean(entry['coverage'] for entry in with_diversity):.1%}</td><td>{mean:.3f}</td></tr>")
        sections.append(f"<section><h3>实验 {exp} · 覆盖加权分</h3><table><thead><tr><th>方法</th><th>全部 25 条</th>"
                        "<th>10 条多样性查询</th><th>Facet 覆盖率</th><th>Mean ms</th></tr></thead><tbody>"
                        + "".join(lines) + "</tbody></table></section>")
        if entity_only:
            hard_rows = [row for row in report["rows"] if row["resolved_plan"]["hard_filters"]]
            lines = []
            for method, label in labels.items():
                answers = [row["methods"][f"exp3_{method}"] for row in hard_rows]
                fills = [len(entry["results"]) for entry in answers]
                lines.append(f"<tr><td>{escape(label)}</td><td>{fmean(fills):.2f}</td>"
                             f"<td>{sum(n < 3 for n in fills)}/{len(fills)}</td><td>{fills.count(0)}</td></tr>")
            sections.append("<section><h3>有硬 facet 的查询 · 返回数量</h3><table><thead><tr><th>方法</th>"
                            "<th>平均返回数量</th><th>不足 3 张</th><th>零结果</th></tr></thead><tbody>"
                            + "".join(lines) + "</tbody></table></section>")
        (OUT / f"caltech101_exp{exp}.html").write_text(html.case_page(report, IMAGE_ROOT, exp), encoding="utf-8")
    sections.append("<section><h2>25 条英文查询</h2><ol>" + "".join(
        f"<li>{escape(row['id'])} · {escape(row['query_type'])} · {escape(row['english'])}</li>"
        for row in read(workload)["rows"]
    ) + "</ol></section>")
    (OUT / summary_file).write_text(html.page("Caltech 局部视觉 Facet MVP", "".join(sections)), encoding="utf-8")


def run_experiment3():
    baseline = read(OUT / "benchmark/caltech101/results.json")
    config = baseline["config"]
    if len(baseline["rows"]) != 25 or config["candidate_k"] != 10:
        raise ValueError("expected the original 25-query, 10-candidate run")
    output = OUT / "benchmark_exp3"
    workload = OUT / "frozen_queries.json"
    options = ("dataset", "source_cache", "hdmg_cache", "directory_cache", "candidate_k",
               "key_budget", "group_budget", "representative_kernel", "nav_budget", "nprobe", "ef",
               "qdrant_url", "index_tag")
    arguments = [arg for key in options for arg in ("--" + key.replace("_", "-"), config[key])]
    run("run_soft_entity_bench.py", "--experiment3-only", "--frozen-report", workload,
        "--output-root", output, *arguments)
    report = read(output / "caltech101/results.json")
    if report["config"]["workload_sha256"] != config["workload_sha256"]:
        raise ValueError("experiment 3 query workload changed")
    for old, new in zip(baseline["rows"], report["rows"], strict=True):
        if any(old[key] != new[key] for key in ("id", "query_object", "resolved_plan", "oracle")):
            raise ValueError(f"experiment 3 reference changed for {old['id']}")
        plan = new["resolved_plan"]
        for method in ("milvus_entity", "qdrant_entity", "chroma_entity"):
            entry = new["methods"][f"exp3_{method}"]
            candidates = entry["stats"]["candidate_ids"]
            if len(candidates) > config["candidate_k"] or any(
                oid.split("/", 1)[0] != plan["entity_preference"] for oid in candidates
            ):
                raise ValueError(f"invalid entity recall for {new['id']} / {method}")
            if len(entry["results"]) > plan["top_k"] or entry["metrics"]["hard_condition_validity"] != 1 or any(
                item["object_id"] not in candidates for item in entry["results"]
            ):
                raise ValueError(f"invalid selection for {new['id']} / {method}")
    render(report, workload, experiments=(3,))
    print("EXPERIMENT_3_COMPLETE 25 queries; preferred-entity top-10; shared SELECT", flush=True)


def main(contract):
    print("WAITING FOR LABEL JOB LOCK", flush=True)
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        complete = read(OUT / "complete.json")
        if complete["images"] != 2056 or complete["splits"] != {"train": 1840, "test": 216}:
            raise ValueError("label checkpoint is incomplete")
    labels = OUT / "assignments_for_index.jsonl"
    rows = preprocess.read_jsonl(OUT / "assignments.jsonl")
    if len(rows) != 2056 or any(row["source"] != "qwen3.5_visual_no_thinking" for row in rows):
        raise ValueError("expected a complete Qwen-only assignment file")
    with labels.open("w", encoding="utf-8") as output:
        for row in rows:
            row["attributes"] = {key: value for key, value in row["attributes"].items() if value["value"] != "unknown"}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256((OUT / "schemas.jsonl").read_bytes() + labels.read_bytes()).hexdigest()[:10]
    tag = f"caltech_local25_{digest}"
    cache, directory = Path(f"cache/{tag}_hdmg.pkl"), Path(f"cache/{tag}_directory.pkl")
    run("build_multidataset_kv_cache.py", "--dataset", "caltech101", "--source-cache", SOURCE,
        "--schemas", OUT / "schemas.jsonl", "--assignments", labels, "--output-cache", cache,
        "--directory-cache", directory, "--embedding-k", 16, "--semantic-intra-k", 8)
    objects = soft.bench.retrieval.make_object_map(soft.bench.load_pickle(cache, []))
    if len(objects) != 1840 or {obj["description"]["entity"] for obj in objects.values()} != set(CATEGORIES):
        raise ValueError("gallery must contain only the 25 approved categories and 1840 train images")
    workload = freeze_workload(contract, objects, soft.bench.load_pickle(SOURCE, []), soft.bench.load_pickle(directory, []))
    run("run_caltech_qwen50_bench.py", "--build-vdb", cache, tag)
    run("run_soft_entity_bench.py", "--dataset", "caltech101", "--source-cache", SOURCE,
        "--hdmg-cache", cache, "--directory-cache", directory, "--candidate-k", contract["vdb_candidate_k"],
        "--index-tag", tag, "--frozen-report", workload, "--output-root", OUT / "benchmark")
    report = read(OUT / "benchmark/caltech101/results.json")
    if len(report["rows"]) != 25 or report["config"]["candidate_k"] != 10:
        raise ValueError("unexpected query count or VDB candidate budget")
    render(report, workload)
    print("BENCHMARK_COMPLETE 25 queries; 10 candidates; two experiments", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-drafts", action="store_true")
    mode.add_argument("--experiment3", action="store_true", help="Reuse the frozen workload and indexes")
    args = parser.parse_args()
    if args.experiment3:
        run_experiment3()
    else:
        contract = load_drafts()
        if args.validate_drafts:
            print("VALIDATED 25 query drafts against the new schemas")
        else:
            main(contract)
