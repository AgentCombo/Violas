#!/usr/bin/env python3
"""Render the six-dataset, five-system Experiment 1/2 comparison."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote


METHODS = {
    "vg_hdmg": ("VG + HDMG", "#315be8"),
    "vg_flat": ("VG − HDMG", "#805ad5"),
    "milvus": ("Milvus", "#e9842b"),
    "qdrant": ("Qdrant", "#16805a"),
    "chroma": ("Chroma", "#c33f70"),
}
GROUP_SCAN_NOTE = (
    "VG−HDMG 按 entity key 定位 VectorGroup，顺扫组内对象并检查硬 facet，"
    "对全部合格对象计算相似度，再用共享 SELECT 返回 Top-3。"
    "VG+HDMG 使用对象评分预算，VG−HDMG 的评分数随组内合格对象数量变化。"
)
EXPERIMENT_METHODS = {
    "exp1": {
        "vg_hdmg": "vg_hdmg",
        "vg_flat": "vg_flat",
        "milvus": "milvus_entity_only",
        "qdrant": "qdrant_entity_only",
        "chroma": "chroma_entity_only",
    },
    "exp2": {
        "vg_hdmg": "vg_hdmg",
        "vg_flat": "vg_flat",
        "milvus": "milvus_facet",
        "qdrant": "qdrant_facet",
        "chroma": "chroma_facet",
    },
}


def esc(value: Any) -> str:
    return html.escape(str(value))


def metric(method: dict[str, Any], name: str) -> float | None:
    value = method["metrics"].get(name)
    return None if value is None else float(value)


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def image_uri(root: str, object_id: str) -> str:
    from artifact_paths import image_uri as resolve_image_uri
    return resolve_image_uri(root, object_id)


def result_card(
    result: dict[str, Any], rank: int, report: dict[str, Any], root: str,
    score_label: str = "similarity",
) -> str:
    if report["modality"] == "image":
        media = f'<img src="{image_uri(root,result["object_id"])}" loading="lazy">'
    else:
        media = f'<p class="snippet">{esc(result.get("preview") or "Preview unavailable")}</p>'
    facets = "".join(
        f"<i>{esc(key)}={esc(value)}</i>"
        for key, value in sorted(result.get("facet_values", {}).items())
    )
    return f'<article>{media}<div><b>#{rank} {esc(result["object_id"])}</b><span>{esc(score_label)} {float(result["score"]):.4f}</span><small>{facets}</small></div></article>'


def query_media(row: dict[str, Any], report: dict[str, Any], root: str) -> str:
    query = row["query_object"]
    if report["modality"] == "image":
        return f'<img class="query" src="{image_uri(root,query["object_id"])}" loading="lazy">'
    return f'<p class="query snippet">{esc(query.get("preview") or "Preview unavailable")}</p>'


def case_html(
    row: dict[str, Any], report: dict[str, Any], root: str, experiment: str
) -> str:
    columns = []
    for display_key, internal_key in EXPERIMENT_METHODS[experiment].items():
        method = row["methods"][internal_key]
        label, color = METHODS[display_key]
        validity = method["metrics"].get(
            "raw_hard_condition_validity",
            method["metrics"].get("hard_condition_validity"),
        )
        cards = "".join(
            result_card(result, rank, report, root)
            for rank, result in enumerate(method["results"], 1)
        )
        stats = method.get("stats", {})
        scored = stats.get("unique_member_scores")
        work_badges = f"<b>scored {scored}</b>" if scored is not None else ""
        if "group_size" in stats:
            work_badges += f'<b>group size {stats["group_size"]}</b>'
        columns.append(
            f'<section class="method" style="--accent:{color}"><h3>{esc(label)}</h3>'
            f'<div class="badges"><b>obj {fmt(metric(method,"objective_ratio"),3)}</b>'
            f'<b>recall {fmt(metric(method,"oracle_object_recall"),3)}</b>'
            f'<b>valid {fmt(None if validity is None else float(validity),3)}</b>'
            f'<b>{method["latency"]["p50_ms"]:.2f} ms</b>{work_badges}</div>{cards}</section>'
        )
    intent = esc(json.dumps(row["canonical_intent"], ensure_ascii=False, indent=2))
    return (
        f'<details class="case"><summary>{query_media(row,report,root)}<span><b>{esc(row["id"])} · '
        f'{esc(row["query_type"])} · {esc(row["entity_hint"])}</b><small>eligible '
        f'{row["eligible_count"]}</small></span></summary><pre>{intent}</pre>'
        f'<div class="method-grid">{"".join(columns)}</div></details>'
    )


def bars(
    report: dict[str, Any], experiment: str, metric_name: str, latency: bool = False
) -> str:
    overall = report["aggregate"]["overall"]
    values = []
    for display_key, internal_key in EXPERIMENT_METHODS[experiment].items():
        if latency:
            value = float(overall[internal_key]["latency"]["p50_ms"])
        else:
            value = float(overall[internal_key][metric_name] or 0.0)
        values.append((display_key, value))
    maximum = max(value for _, value in values) or 1.0
    rows = []
    for display_key, value in values:
        label, color = METHODS[display_key]
        width = (
            math.log1p(value) / math.log1p(maximum) * 100
            if latency
            else min(100.0, value * 100)
        )
        rows.append(
            f'<div class="bar"><span>{esc(label)}</span><i><b style="width:{width:.2f}%;background:{color}"></b></i>'
            f'<strong>{value:.3f}{" ms" if latency else ""}</strong></div>'
        )
    return "".join(rows)


def dataset_page(
    report: dict[str, Any], root: str, experiment: str
) -> str:
    exp_name = "Experiment 1 · entity-only DB baselines" if experiment == "exp1" else "Experiment 2 · entity+facet DB baselines + same SELECT"
    cases = "".join(
        case_html(row, report, root, experiment) for row in report["rows"]
    )
    flat_note = (
        GROUP_SCAN_NOTE
        if report["config"].get("vg_flat", {}).get("execution") == "entity_key_group_scan"
        else "VG−HDMG 使用对应结果文件中的历史消融配置。"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(report['dataset'])} · {esc(exp_name)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font-family:Inter,Microsoft YaHei,system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}.hero,.panel,.case{{background:white;border:1px solid #dce5f0;border-radius:15px;margin:14px 0}}.hero{{padding:24px;background:linear-gradient(120deg,#172044,#315be8);color:white}}.hero p{{color:#dbe5ff}}.panels{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.panel{{padding:16px}}.bar{{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center;margin:9px 0;font-size:11px}}.bar i{{height:10px;background:#edf1f6}}.bar i b{{height:100%;display:block}}.bar strong{{text-align:right}}.case>summary{{display:grid;grid-template-columns:110px 1fr;gap:12px;align-items:center;padding:10px;cursor:pointer}}.query{{width:110px;height:78px;object-fit:contain;background:#edf1f6}}.snippet{{overflow:hidden;background:#edf3ff;padding:7px;font-size:8px;line-height:1.4}}pre{{margin:0 10px 10px;background:#111827;color:#dbeafe;padding:10px;border-radius:8px;max-height:220px;overflow:auto}}.method-grid{{display:grid;grid-template-columns:repeat(5,minmax(245px,1fr));gap:9px;padding:10px;overflow:auto}}.method{{border-top:4px solid var(--accent);background:#fafcff;border-radius:8px;padding:8px}}.method h3{{margin:0 0 6px}}.badges{{display:flex;flex-wrap:wrap;gap:4px}}.badges b{{font-size:9px;background:white;border:1px solid #dce5f0;padding:3px;border-radius:999px}}article{{display:grid;grid-template-columns:90px 1fr;gap:6px;border-top:1px solid #dce5f0;padding:7px 0;font-size:9px}}article img{{width:90px;height:68px;object-fit:contain;background:#edf1f6}}article span,article small,article i{{display:block}}article i{{font-style:normal;font-size:7px;color:#617087}}@media(max-width:900px){{.panels{{grid-template-columns:1fr}}}}</style></head><body><main><section class="hero"><h1>{esc(report['dataset'])} · {esc(exp_name)}</h1><p>100个相同的held-out object + canonical intent；Top-3；无warmup、每条query单次计时。{flat_note}</p></section><div class="panels"><section class="panel"><h2>Mean objective ratio</h2>{bars(report,experiment,'mean_objective_ratio')}</section><section class="panel"><h2>Retrieval p50 latency（log scale）</h2>{bars(report,experiment,'',True)}</section></div><h2>All 100 cases</h2>{cases}</main></body></html>"""


def summary_page(reports: list[dict[str, Any]]) -> str:
    rows = []
    chart_sections = []
    for report in reports:
        overall = report["aggregate"]["overall"]
        slug = report["dataset"].lower().replace(" ", "_")
        for experiment in ("exp1", "exp2"):
            hdmg_latency = overall["vg_hdmg"]["latency"]["p50_ms"]
            for display_key, internal_key in EXPERIMENT_METHODS[experiment].items():
                method = overall[internal_key]
                latency_value = method["latency"]["p50_ms"]
                rows.append(
                    f'<tr><td>{esc(report["dataset"])}</td><td><a href="{slug}_{experiment}.html">{experiment.upper()}</a></td><td>{esc(METHODS[display_key][0])}</td>'
                    f'<td>{fmt(method["mean_objective_ratio"])}</td><td>{fmt(method["mean_oracle_object_recall"])}</td>'
                    f'<td>{fmt(method["mean_hard_condition_validity"])}</td><td>{latency_value:.3f}</td>'
                    f'<td>{latency_value/max(hdmg_latency,1e-12):.2f}×</td></tr>'
                )
        chart_sections.append(
            f'<section class="dataset"><h2>{esc(report["dataset"])}</h2><div><h3>Exp.1 objective</h3>{bars(report,"exp1","mean_objective_ratio")}</div>'
            f'<div><h3>Exp.1 latency</h3>{bars(report,"exp1","",True)}</div><div><h3>Exp.2 objective</h3>{bars(report,"exp2","mean_objective_ratio")}</div>'
            f'<div><h3>Exp.2 latency</h3>{bars(report,"exp2","",True)}</div></section>'
        )
    flat_note = (
        GROUP_SCAN_NOTE
        if all(
            report["config"].get("vg_flat", {}).get("execution") == "entity_key_group_scan"
            for report in reports
        )
        else "各数据集的 VG−HDMG 配置见对应详情页。"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Six datasets · five systems</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fb;color:#172033;font-family:Inter,Microsoft YaHei,system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:26px}}header,.dataset,.table{{background:white;border:1px solid #dce5f0;border-radius:15px;padding:18px;margin:15px 0}}header{{background:linear-gradient(120deg,#172044,#315be8);color:white}}header p{{color:#dbe5ff;line-height:1.6}}.dataset{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}.dataset h2{{grid-column:1/-1}}.bar{{display:grid;grid-template-columns:90px 1fr 75px;gap:6px;align-items:center;margin:7px 0;font-size:9px}}.bar i{{height:8px;background:#edf1f6}}.bar i b{{height:100%;display:block}}.bar strong{{text-align:right}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:11px}}th,td{{padding:8px;border-bottom:1px solid #e3e9f1;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}a{{color:#315be8}}@media(max-width:1000px){{.dataset{{grid-template-columns:1fr 1fr}}}}@media(max-width:650px){{.dataset{{grid-template-columns:1fr}}}}</style></head><body><main><header><h1>Six Datasets · Five-System Canonical Benchmark</h1><p>VG+HDMG、VG−HDMG、Milvus、Qdrant、Chroma；Experiment 1仅给外部数据库entity能力，Experiment 2提供entity+facet并共享最终SELECT。所有方法不做warmup，每条query只计时一次。{flat_note}表中的“vs HDMG”大于1表示HDMG更快。</p></header>{''.join(chart_sections)}<section class="table"><table><thead><tr><th>Dataset</th><th>Experiment</th><th>System</th><th>Objective</th><th>Object recall</th><th>Hard validity</th><th>p50 ms</th><th>vs HDMG</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) != len(args.dataset_root):
        raise SystemExit("--result and --dataset-root counts must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for result_path, root in zip(args.result, args.dataset_root):
        report = json.loads(result_path.read_text(encoding="utf-8"))
        reports.append(report)
        slug = report["dataset"].lower().replace(" ", "_")
        for experiment in ("exp1", "exp2"):
            (args.output_dir / f"{slug}_{experiment}.html").write_text(
                dataset_page(report, root, experiment), encoding="utf-8"
            )
    (args.output_dir / "six_dataset_five_system_summary.html").write_text(
        summary_page(reports), encoding="utf-8"
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
