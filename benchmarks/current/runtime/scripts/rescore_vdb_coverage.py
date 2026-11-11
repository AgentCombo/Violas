#!/usr/bin/env python3
"""Rescore frozen top-80 and top-10 answers with normalized facet coverage."""
from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from statistics import fmean


DATASETS = ("caltech101", "cub200", "flowers102", "20news", "ohsumed", "yahoo")
TEXT_REPORTS = {"20news": "soft_entity_text_verified", "ohsumed": "soft_entity_text_sparse",
                "yahoo": "soft_entity_text_verified"}
METHODS = {"vg_hdmg": "VG+HDMG", "vg_rep": "VG−HDMG", "milvus_ann": "Milvus ANN",
           "milvus_key": "Milvus key", "qdrant_ann": "Qdrant ANN", "qdrant_key": "Qdrant key",
           "chroma_ann": "Chroma ANN", "chroma_key": "Chroma key"}
DIVERSITY_WEIGHT = 0.30
ROOT = Path("artifacts")
OUTPUT = ROOT / "six_dataset_coverage_rescore"


def score(entry, plan):
    valid = [item for item in entry["results"] if all(
        item["facet_values"].get(rule["key"]) == rule["value"] for rule in plan["hard_filters"]
    )]
    dimension = plan["facet_dimension"]
    counts = Counter(item["facet_values"].get(dimension) for item in valid)
    targets = plan["category_preferences"]
    covered = {value: min(counts[value], pref["quota"]) for value, pref in targets.items()}
    old_reward = sum(targets[value]["lambda"] * count for value, count in covered.items())
    raw_relevance = entry["metrics"]["objective"] - old_reward
    if abs(raw_relevance - sum(item["score"] for item in valid)) > 2e-6:
        raise ValueError("stored objective differs from relevance plus facet reward")
    requested = sum(pref["quota"] for pref in targets.values())
    coverage = sum(covered.values()) / requested if requested else None
    relevance = raw_relevance / plan["top_k"]
    objective = relevance if coverage is None else (1 - DIVERSITY_WEIGHT) * relevance + DIVERSITY_WEIGHT * coverage
    return {"objective": objective, "relevance": relevance, "coverage": coverage,
            "objective_ratio_old": entry["metrics"]["objective_ratio"], "covered": covered}


def read_report(batch, dataset):
    return json.loads((ROOT / batch / dataset / "results.json").read_text(encoding="utf-8"))


def rescore():
    rows = []
    for dataset in DATASETS:
        old = read_report(TEXT_REPORTS.get(dataset, "six_dataset_soft_entity_online_cosine"), dataset)
        new = read_report("six_dataset_vdb_top10", dataset)
        if len(old["rows"]) != 100 or len(new["rows"]) != 100:
            raise ValueError(f"expected 100 queries per batch for {dataset}")
        for a, b in zip(old["rows"], new["rows"], strict=True):
            if any(a[key] != b[key] for key in ("id", "query_object", "resolved_plan", "oracle")):
                raise ValueError(f"mismatched original queries for {dataset}/{a['id']}")
            plan = a["resolved_plan"]
            for exp in (1, 2):
                for method in METHODS:
                    name = f"exp{exp}_{method}"
                    is_vg = method.startswith("vg_")
                    if is_vg and a["methods"][name] != b["methods"][name]:
                        raise ValueError(f"VG outputs changed for {dataset}/{a['id']}")
                    rows.append({"dataset": dataset, "id": a["id"], "experiment": exp, "method": method,
                                 "query_type": a["query_type"], "facet_dimension": plan["facet_dimension"],
                                 "targets": plan["category_preferences"],
                                 "at80": score(a["methods"][name], plan),
                                 "at10": None if is_vg else score(b["methods"][name], plan)})
    return rows


def mean_scores(entries):
    entries = [entry for entry in entries if entry is not None]
    if not entries:
        return None
    result = {}
    for key in ("objective", "relevance", "coverage", "objective_ratio_old"):
        values = [entry[key] for entry in entries if entry[key] is not None]
        result[key] = fmean(values) if values else None
    return result


def table(rows):
    def cells(result):
        return "".join(f"<td>{result[key]:.4f}</td>" if result and result[key] is not None else "<td>—</td>"
                       for key in ("relevance", "coverage", "objective"))

    body = []
    for method, label in METHODS.items():
        records = [row for row in rows if row["method"] == method]
        a, b = (mean_scores([row[budget] for row in records]) for budget in ("at80", "at10"))
        body.append(f"<tr><th scope='row'>{escape(label)}</th>{cells(a)}{cells(b)}</tr>")
    return ("<table><thead><tr><th>方法</th><th>80 相关性</th><th>80 覆盖率</th><th>80 新分数</th>"
            "<th>10 相关性</th><th>10 覆盖率</th><th>10 新分数</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table>")


def main():
    rows = rescore()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    sections = []
    for exp in (1, 2):
        sections.append(f"<h2>实验 {exp} · 六数据集汇总</h2>")
        for subset in ("all", "diversity", "no_diversity"):
            records = [row for row in rows if row["experiment"] == exp and (
                subset == "all" or bool(row["targets"]) == (subset == "diversity")
            )]
            label = {"all": "全部查询", "diversity": "有多样性要求", "no_diversity": "无多样性要求"}[subset]
            sections.append(f"<h3>{label}</h3>" + table(records))
            for dataset in ("all", *DATASETS):
                for method in METHODS:
                    selected = [row for row in records if row["method"] == method and (
                        dataset == "all" or row["dataset"] == dataset
                    )]
                    summary.append({"experiment": exp, "subset": subset, "dataset": dataset, "method": method,
                                    "count": len(selected), **{k: mean_scores([row[k] for row in selected])
                                                               for k in ("at80", "at10")}})
    for dataset in DATASETS:
        sections.append(f"<h2>{escape(dataset)} · 有多样性要求</h2>")
        for exp in (1, 2):
            sections.append(f"<h3>实验 {exp}</h3>" + table([
                row for row in rows if row["dataset"] == dataset and row["experiment"] == exp and row["targets"]
            ]))
    payload = {"diversity_weight": DIVERSITY_WEIGHT, "evaluation": "frozen answers; absolute score", "summary": summary,
               "rows": rows}
    (OUTPUT / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    html = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Violas facet 覆盖重评分</title>
<style>body{max-width:1100px;margin:32px auto;padding:0 18px;font:15px/1.6 system-ui,sans-serif;color:#18243a;background:#f6f8fc}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:white}th,td{padding:8px;text-align:right;border-bottom:1px solid #dde5f0}
th:first-child{text-align:left}thead{background:#e8eff9}h2{margin-top:44px}a{color:#194b9c}</style>
<h1>Violas facet 覆盖重评分</h1>
<p>沿用原报告和候选上限 10 复测的返回结果。相关性是有效对象混合分之和除以请求的 k。覆盖率是已满足的目标配额除以总目标配额。</p>
<p>有多样性要求时，新分数等于 0.70 × 相关性 + 0.30 × 覆盖率。其余查询直接用相关性。表中分数是 0–1 的绝对分数。VG 每组只有一套结果，列在左侧。</p>
<p><a href='scores.json'>逐条分数和汇总 JSON</a></p>
""" + "\n".join(sections) + "</html>"
    (OUTPUT / "summary.html").write_text(html, encoding="utf-8")
    print(f"COMPLETE {len(rows)} rows {OUTPUT}")


if __name__ == "__main__":
    main()
