#!/usr/bin/env python3
"""Render the soft entity benchmark as a summary and per-query reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from render_five_system_canonical import esc, fmt, query_media, result_card


LABELS = {
    "vg_hdmg": "VG + HDMG", "vg_rep": "VG − HDMG",
    "milvus_ann": "Milvus · ANN", "milvus_key": "Milvus · key",
    "qdrant_ann": "Qdrant · ANN", "qdrant_key": "Qdrant · key",
    "chroma_ann": "Chroma · ANN", "chroma_key": "Chroma · key",
}
ENTITY_LABELS = {
    "vg_hdmg": LABELS["vg_hdmg"], "vg_rep": LABELS["vg_rep"],
    "milvus_entity": "Milvus · entity top-10",
    "qdrant_entity": "Qdrant · entity top-10",
    "chroma_entity": "Chroma · entity top-10",
}
EXPERIMENTS = {
    1: "数据库重排使用 embedding 和软 entity，两个 VG 还使用 facet",
    2: "八种方法均使用 facet，并共享最终 SELECT",
    3: "数据库按首选 entity 过滤，再按 embedding 召回前 10。完整 SELECT 检查硬 facet 并计算覆盖奖励，最多返回 3 项",
}
STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#18263c;font:15px/1.5 system-ui,sans-serif}
main{margin:auto;max-width:1520px;padding:24px}header,section,details{background:white;border:1px solid #dae2ee;border-radius:12px;padding:16px;margin:14px 0}
header{background:#173456;color:white}header a{color:#c3ebff}h1,h2,h3{line-height:1.2}
a{color:#195cac}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{text-align:right;padding:8px;border-bottom:1px solid #e3e9f1;white-space:nowrap}
td:first-child,th:first-child{text-align:left}.table{overflow:auto}.plots,.methods{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.methods{grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}.methods section{margin:0}
.bar{display:grid;grid-template-columns:120px 1fr 78px;gap:6px;align-items:center;font-size:13px;margin:7px 0}
.bar i{display:block;height:9px;background:#e9eef5}.bar i b{display:block;height:100%;background:#4887bd}
.bar.vg i b{background:#188376}.bar strong{text-align:right}
summary{cursor:pointer;display:flex;align-items:center;gap:12px}.query{width:100px;height:72px;object-fit:contain;margin:0}
article{display:flex;gap:8px;border-top:1px solid #e3e9f1;padding:6px 0;overflow-wrap:anywhere}
article img{width:80px;height:65px;object-fit:contain}article span,article small,article i{display:block}
article i{font-style:normal;color:#536582}.snippet{max-width:130px;max-height:94px;overflow:auto;font-size:10px}
pre{overflow:auto;background:#f4f7fb;padding:12px}h3{margin-top:0}
@media(max-width:750px){.plots{grid-template-columns:1fr}}
"""


def page(title, body):
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(title)}</title><style>{STYLE}</style><main>{body}</main></html>'
    )


def representative_note(report):
    kernel = report["config"].get("representative_kernel")
    if kernel == "cosine":
        return "VG−HDMG 逐向量计算余弦，对象和 key 的范数计入查询耗时。"
    if kernel == "scalar":
        return "VG−HDMG 使用逐向量 np.dot 和 Python 组选择。"
    return "VG−HDMG 使用 NumPy 批量打分。"


def tuning_note(report):
    config = report["config"].get("hdmg_tuned")
    if config is None:
        return ""
    verified = report["config"].get("hdmg_verified")
    if verified is not None:
        kernel = (
            f'核验使用精确 CSR 稀疏乘法，非零元素占 {verified["embedding_density"]:.2%}。'
            if verified.get("verification_kernel") == "csr" else ""
        )
        return (
            f'VG+HDMG 先用 {verified["seed_nav_budget"]} 步导航和 {verified["seed_group_budget"]} 个组生成初始答案，'
            '再用组分数上界筛出需要批量核验的组。'
            f'{kernel}'
            f'硬 facet 交集不超过 {config["exact_limit"]} 个对象时直接评分。'
            f'上一版 VG+HDMG 同轮 p50 为 {report["config"]["hdmg_before_paired_latency"]["p50_ms"]:.3f} ms。'
            '其他方法沿用原报告。'
        )
    return (
        f'VG+HDMG 使用开发集选出的 {config["nav_budget"]} 步导航、'
        f'{config["group_budget"]} 个候选组和每组 {config["pivots"]} 个成员代表。'
        f'硬 facet 交集不超过 {config["exact_limit"]} 个对象时直接对交集评分。'
        f'旧 VG+HDMG 同轮复测 p50 为 {report["config"]["hdmg_before_paired_latency"]["p50_ms"]:.3f} ms。'
        '其他方法沿用原报告的结果和时延。'
    )


def summary_section(report, experiment):
    metrics = report["aggregate"]["overall"]
    labels = ENTITY_LABELS if experiment == 3 else LABELS
    methods = {method: metrics[f"exp{experiment}_{method}"] for method in labels}
    hdmg_ms = methods["vg_hdmg"]["latency"]["p50_ms"]
    max_ms = max(value["latency"]["p50_ms"] for value in methods.values())
    rows, objective_bars, latency_bars = [], [], []
    for method, value in methods.items():
        ms = value["latency"]["p50_ms"]
        objective = value["mean_objective_ratio"]
        rows.append(
            f'<tr><td>{esc(labels[method])}</td><td>{fmt(objective)}</td>'
            f'<td>{fmt(value["mean_oracle_object_recall"])}</td>'
            f'<td>{fmt(value["mean_hard_condition_validity"])}</td>'
            f'<td>{fmt(value["mean_cross_entity_rate"])}</td>'
            f'<td>{ms:.3f}</td><td>{value["latency"]["p95_ms"]:.3f}</td>'
            f'<td>{ms / hdmg_ms:.2f}×</td></tr>'
        )
        start = f'<div class="bar {"vg" if method.startswith("vg_") else ""}"><span>{esc(labels[method])}</span>'
        objective_bars.append(f'{start}<i><b style="width:{(objective or 0) * 100:.3f}%"></b></i><strong>{fmt(objective)}</strong></div>')
        latency_bars.append(f'{start}<i><b style="width:{100 * math.log1p(ms) / math.log1p(max_ms):.3f}%"></b></i><strong>{ms:.3f} ms</strong></div>')
    dataset = report["dataset"]
    oracle_cross = sum(row["oracle"]["cross_entity_rate"] for row in report["rows"]) / len(report["rows"])
    return (
        f'<section><h2>{esc(dataset)} · 实验 {experiment}</h2><p>{EXPERIMENTS[experiment]}。'
        f'{representative_note(report)}{tuning_note(report)}共 {len(report["rows"])} 条查询。'
        f'Oracle 跨 entity 结果比例 {oracle_cross:.1%}。'
        f' <a href="{esc(dataset)}_exp{experiment}.html">查看逐条结果</a></p>'
        f'<div class="plots"><div><h3>Objective ratio</h3>{"".join(objective_bars)}</div>'
        f'<div><h3>p50 latency（log1p 刻度）</h3>{"".join(latency_bars)}</div></div>'
        '<div class="table"><table><thead><tr><th>方法</th><th>Objective ratio</th>'
        '<th>Object recall</th><th>Hard validity</th><th>跨 entity 比例</th>'
        '<th>p50 ms</th><th>p95 ms</th><th>时延 ÷ HDMG 时延</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def case_page(report, root, experiment):
    blocks = []
    labels = ENTITY_LABELS if experiment == 3 else LABELS
    for row in report["rows"]:
        cards = []
        for method, label in {**labels, "oracle": "全库 Oracle"}.items():
            result = row["oracle"] if method == "oracle" else row["methods"][f"exp{experiment}_{method}"]
            if method == "oracle":
                details = f'Objective {result["objective"]:.4f}'
            else:
                metrics = result["metrics"]
                stats = result["stats"]
                details = (
                    f'Objective ratio {fmt(metrics["objective_ratio"])} · '
                    f'Recall {fmt(metrics["oracle_object_recall"])} · '
                    f'Validity {fmt(metrics["hard_condition_validity"])} · '
                    f'{result["latency"]["p50_ms"]:.3f} ms'
                )
                details += f' · 评分 {stats["unique_member_scores"]} 个对象'
                if "candidate_ids" in stats:
                    details += f' · 召回 {stats["candidate_count"]} 个，返回 {len(result["results"])} 个'
                if "center_scores" in stats:
                    details += f' · 检查 {stats["center_scores"]} 个中心，展开 {stats["searched_groups"]} 个候选组'
                if stats.get("route") == "exact_facet_posting":
                    details += " · 直接评分硬 facet 交集"
                if stats.get("route") == "verified_graph":
                    details += f' · 核验阶段评分 {stats["verification_scores"]} 个对象'
                    if stats.get("dense_pass"):
                        details += (
                            " · 全库稀疏矩阵打分"
                            if stats.get("verification_kernel") == "csr"
                            else " · 连续全库矩阵打分"
                        )
                if stats.get("routed_keys"):
                    details += ' · key 路由 ' + "、".join(stats["routed_keys"])
            results = "".join(
                result_card(item, rank, report, root, "混合相关性")
                for rank, item in enumerate(result["results"], 1)
            )
            cards.append(f'<section><h3>{esc(label)}</h3><p>{esc(details)}</p>{results}</section>')
        plan = esc(json.dumps(row["resolved_plan"], ensure_ascii=False, indent=2))
        english = f'<p>{esc(row["english"])}</p>' if row.get("english") else ""
        blocks.append(
            f'<details><summary>{query_media(row, report, root)}'
            f'<b>{esc(row["id"])} · {esc(row["query_type"])} · {esc(row["entity_hint"])}</b>'
            f'<span>合格对象 {row["eligible_count"]} · 覆盖 {row["eligible_entity_count"]} 个 entity</span>'
            f'</summary>{english}<pre>{plan}</pre><div class="methods">{"".join(cards)}</div></details>'
        )
    title = f'{report["dataset"]} · 实验 {experiment}'
    summary_file = "soft_entity_exp3_summary.html" if experiment == 3 else "soft_entity_summary.html"
    return page(title, f'<header><h1>{esc(title)}</h1><p>{EXPERIMENTS[experiment]}。{representative_note(report)}{tuning_note(report)}'
                f'<a href="{summary_file}">返回汇总</a></p></header>' + "".join(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) != len(args.dataset_root):
        parser.error("each --result needs one --dataset-root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for path, root in zip(args.result, args.dataset_root):
        report = json.loads(path.read_text(encoding="utf-8"))
        for experiment in EXPERIMENTS:
            sections.append(summary_section(report, experiment))
            (args.output_dir / f'{report["dataset"]}_exp{experiment}.html').write_text(
                case_page(report, root, experiment), encoding="utf-8"
            )
    intro = (
        '<header><h1>软 entity 检索 · 八种方法</h1>'
        '<p>Entity 是偏好，硬 facet 决定答案资格。ANN 路径在全库向量索引上召回，'
        'key 路径先选多个相关 key。两个 VG 分别以图游走或全局 representative 排序发现候选组。'
        '所有结果使用同一个全库 Oracle 评价。</p>'
        '<p>时延含在线召回和重排，采用单次计时和轮换执行顺序。'
        '实验使用 Milvus Lite、Qdrant Server 和嵌入式 Chroma。</p></header>'
    )
    (args.output_dir / "soft_entity_summary.html").write_text(
        page("软 entity 检索 · 八种方法", intro + "".join(sections)), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
