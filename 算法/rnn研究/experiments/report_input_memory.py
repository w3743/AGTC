"""Build an offline Chinese report for input-memory RTRL experiments.

Usage:
  python report_input_memory.py RUN_DIRECTORY [--catalog HISTORY_CATALOG.json]

Requires a completed run directory containing aggregate.json, runs_summary.json,
audit.json and per-run events.jsonl. Writes report.html, CSV exports and PNG/SVG
figures beside the run. Does not modify experiment outputs.
"""
from __future__ import annotations

import argparse
import base64
import csv
from html import escape
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, r"C:\pioneer\rnn-report-deps")
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT_PATH)
FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [FONT]
plt.rcParams["axes.unicode_minus"] = False

KINDS = ("baseline_rtrl20", "fixed_input_memory20", "learned_input_memory20")
KIND_NAMES = {"baseline_rtrl20": "基线 RTRL", "fixed_input_memory20": "固定输入记忆", "learned_input_memory20": "可学习输入记忆"}
COLORS = {"baseline_rtrl20": "#344e9a", "fixed_input_memory20": "#df9141", "learned_input_memory20": "#bd3e42"}
TASK_NAMES = {
    "binary_alternating": "二进制交替 01", "binary_triples": "二进制三连 000111", "binary_period16": "二进制周期16",
    "decimal_cycle": "十进制循环", "decimal_triples": "十进制三连", "decimal_zigzag": "十进制锯齿",
    "decimal_fibonacci_mod10": "十进制 Fibonacci mod 10", "decimal_switching": "十进制切换", "decimal_iid_random": "十进制 IID 随机",
}
TASKS_BINARY = ("binary_alternating", "binary_triples", "binary_period16")
TASKS_DECIMAL = tuple(k for k in TASK_NAMES if k.startswith("decimal_"))
SEEDS = (2, 20261027, 20261028)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_audit(audit: dict) -> None:
    """Require an explicit passing result; don't infer pass from file presence."""
    if audit.get("passed") is not True:
        raise ValueError("audit.json 的 passed 字段不是 true，拒绝生成最终报告")


def read_events(run: Path, summaries: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    result = {}
    for s in summaries:
        key = (s["kind"], s["task"], int(s["seed"]))
        path = run / s["run_directory"] / "events.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(rows) != 600 or [int(x.get("step", -1)) for x in rows] != list(range(1, 601)):
            raise ValueError(f"事件缺失或 step 不连续：{path}")
        result[key] = rows
    expected = {(k, t, seed) for k in KINDS for t in TASK_NAMES for seed in SEEDS}
    if set(result) != expected:
        raise ValueError(f"runs_summary/events 组合不完整：期望 {len(expected)}，实际 {len(result)}")
    return result


def rolling(values: list[float], window: int = 50) -> list[float]:
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i-window]
        out.append(total / min(i + 1, window))
    return out


def save_curves(run: Path, events: dict) -> list[tuple[str, Path, Path]]:
    figures = []
    for key, tasks, title in (("binary", TASKS_BINARY, "二进制任务：50步滚动准确率"), ("decimal", TASKS_DECIMAL, "十进制任务：50步滚动准确率")):
        cols = 3 if key == "binary" else 3
        rows = 1 if key == "binary" else 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.3 if rows == 1 else 8), sharey=True)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, task in zip(axes, tasks):
            for kind in KINDS:
                curves = [rolling([float(bool(e.get("hit"))) for e in events[(kind, task, seed)]]) for seed in SEEDS]
                mean = [statistics.mean(c[i] for c in curves) for i in range(600)]
                ax.plot(range(1, 601), mean, color=COLORS[kind], label=KIND_NAMES[kind], linewidth=1.7)
            ax.set_title(TASK_NAMES[task]); ax.set_xlabel("预测步"); ax.set_ylim(0, 1.02); ax.grid(alpha=.22)
        for ax in axes[len(tasks):]:
            ax.set_visible(False)
        axes[0].set_ylabel("命中率（50步窗口，3个种子均值）")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(title); fig.tight_layout()
        png, svg = run / f"input_memory_{key}_learning_curves.png", run / f"input_memory_{key}_learning_curves.svg"
        fig.savefig(png, dpi=160); fig.savefig(svg); plt.close(fig)
        figures.append((title, png, svg))
    return figures


def table_html(headers: list[str], rows: list[list[object]], table_id: str) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape('' if v is None else str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f'<input class="filter" data-table="{table_id}" placeholder="筛选此表…"><div class="scroll"><table id="{table_id}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def metric(value):
    return "" if value is None else f"{float(value):.4f}"


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)


def history_records(catalog: dict) -> tuple[list[dict], list[dict]]:
    """Return normalized rows and comparison groups; decimal families stay distinct."""
    normalized = []
    for raw in catalog.get("records", []):
        stage = raw.get("stage", "unknown")
        task = str(raw.get("task", "unknown"))
        if stage in {"decimal_guess", "linear_readout"}:
            family = "decimal"
        elif stage == "retention_no_decay_legacy":
            family = "legacy"
        else:
            family = "binary" if task in {"alternating", "triples", "period16"} else "other"
        row = {**raw, "task_family": family}
        if row.get("parameters") is None and row.get("protocol_ref"):
            protocol_path = Path(row["protocol_ref"])
            if protocol_path.is_file():
                protocol = load(protocol_path)
                row["parameters"] = protocol.get("parameter_count") or row.get("parameters")
                row["width"] = protocol.get("neuron_count") or protocol.get("neurons")
        normalized.append(row)

    # Per-seed summaries are preferred. An aggregate-only row is kept only if its
    # stage/model/task has no per-seed rows; it is never added as another seed.
    groups: dict[tuple, list[dict]] = {}
    for row in normalized:
        key = (row["stage"], row.get("model"), row.get("task"))
        groups.setdefault(key, []).append(row)
    selected = []
    for key, values in groups.items():
        seeded = [x for x in values if x.get("seed") is not None]
        selected.extend(seeded if seeded else values)
    return normalized, selected


def history_architecture_rows(selected: list[dict], family: str) -> list[list[object]]:
    """Summarize seed runs at stage/model/task granularity for primary task families."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in selected:
        if r.get("task_family") != family:
            continue
        if family == "binary" and r.get("task") not in {"alternating", "triples", "period16"}:
            continue
        if family == "decimal" and r.get("stage") not in {"decimal_guess", "linear_readout"}:
            continue
        key = (str(r.get("stage")), str(r.get("model")), str(r.get("task")))
        groups.setdefault(key, []).append(r)
    rows = []
    for (stage, model, task), vals in sorted(groups.items()):
        seeded = [v for v in vals if v.get("seed") is not None]
        metrics = [v.get("metrics", {}) for v in (seeded or vals)]
        def avg(k):
            xs = [float(m[k]) for m in metrics if m.get(k) is not None]
            return statistics.mean(xs) if xs else None
        architectures = sorted({f"{v.get('parameters')}参数" + (f"/{v.get('width')}神经元" if v.get("width") else "") for v in (seeded or vals) if v.get("parameters") is not None})
        rows.append([stage, model, ", ".join(architectures), task, len(seeded), metric(avg("accuracy")), metric(avg("first100_accuracy")), metric(avg("last100_accuracy")), metric(avg("mean_nll")), "来源行含复用标注" if any(v.get("duplicate_source_of") for v in vals) else "按该阶段记录汇总"])
    return rows


def raw_catalog_rows(normalized: list[dict]) -> tuple[list[str], list[list[object]]]:
    headers = ["stage", "run_id", "model", "task", "task_family", "seed", "parameters", "updates", "accuracy", "first100_accuracy", "last100_accuracy", "last_window_accuracy", "last_window_length", "mean_nll", "count", "hits", "origin", "source_origin", "duplicate_source_of", "comparison_scope", "summary_ref", "aggregate_ref", "protocol_ref"]
    rows = []
    for r in normalized:
        m = r.get("metrics", {})
        window_length = m.get("last_window_count") or (100 if m.get("last100_accuracy") is not None else None)
        rows.append([r.get("stage"), r.get("run_id"), r.get("model"), r.get("task"), r.get("task_family"), r.get("seed"), r.get("parameters"), r.get("updates"), m.get("accuracy"), m.get("first100_accuracy"), m.get("last100_accuracy"), m.get("last_window_accuracy"), window_length, m.get("mean_nll"), m.get("count"), m.get("hits"), r.get("origin"), r.get("source_origin"), r.get("duplicate_source_of"), r.get("comparison_scope"), r.get("summary_ref"), r.get("aggregate_ref"), r.get("protocol_ref")])
    return headers, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_directory", type=Path)
    ap.add_argument("--catalog", type=Path)
    args = ap.parse_args()
    run = args.run_directory.resolve()
    catalog_path = args.catalog or run / "history_catalog_normalized.json"
    if not catalog_path.is_file() and args.catalog is None:
        catalog_path = run / "history_catalog.json"
    if not catalog_path.is_file() and args.catalog is None:
        catalog_path = Path(r"C:\pioneer\rnn-research-runs\input_memory_20261031\history_catalog_normalized.json")
    aggregate = load(run / "aggregate.json")
    summaries = load(run / "runs_summary.json")
    audit = load(run / "audit.json")
    validate_audit(audit)
    if not isinstance(aggregate, list) or len(aggregate) != 27:
        raise ValueError("aggregate.json 必须包含27个 kind×task 聚合行")
    if len(summaries) != 81:
        raise ValueError("runs_summary.json 必须包含81个 kind×task×seed run")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"找不到历史 catalog：{catalog_path}")
    catalog = load(catalog_path)
    original_count = int(catalog.get("original_record_count", catalog.get("record_count", -1)))
    if original_count != 463:
        raise ValueError(f"历史 catalog 原始记录数应为463，实际{original_count}，请核对 catalog 版本")
    if len(catalog.get("records", [])) != int(catalog.get("record_count", -1)):
        raise ValueError("catalog record_count 与 records 实际条数不一致")
    events = read_events(run, summaries)
    figures = save_curves(run, events)
    profile = load(run / "compute_profile.json")

    baseline_accuracy = {r.get("task"): float(r.get("accuracy")) for r in aggregate if r.get("kind") == "baseline_rtrl20" and r.get("accuracy") is not None}
    new_headers = ["任务", "任务族", "模型", "runs", "updates/run", "accuracy", "相对baseline差(pp)", "first100", "last100", "mean NLL", "参数", "core参数"]
    new_rows = []
    for r in aggregate:
        task = str(r["task"])
        kind = str(r["kind"])
        cfg = profile.get(kind, {})
        core = {"baseline_rtrl20": 641, "fixed_input_memory20": 1241, "learned_input_memory20": 1271}.get(kind)
        delta = 100.0 * (float(r["accuracy"]) - baseline_accuracy[task]) if r.get("accuracy") is not None and task in baseline_accuracy else None
        new_rows.append([TASK_NAMES.get(task, task), "binary" if task.startswith("binary_") else "decimal", KIND_NAMES.get(kind, kind), r.get("runs", 9), 599, metric(r.get("accuracy")), f"{delta:+.2f}" if delta is not None else None, metric(r.get("first100_accuracy")), metric(r.get("last100_accuracy")), metric(r.get("mean_nll")), cfg.get("parameters"), core])
    write_csv(run / "input_memory_new_results.csv", new_headers, new_rows)

    norm, hist_selected = history_records(catalog)
    hist_headers = ["阶段", "模型", "任务", "任务族", "seed", "accuracy", "末段准确率", "末段窗口长度", "NLL", "count", "updates", "origin/source", "独立性备注", "catalog路径"]
    hist_rows = []
    for r in hist_selected:
        m = r.get("metrics", {})
        last_metric = m.get("last100_accuracy", m.get("last_window_accuracy"))
        window_length = m.get("last_window_count") or (100 if m.get("last100_accuracy") is not None else None)
        hist_rows.append([r.get("stage"), r.get("model"), r.get("task"), r.get("task_family"), r.get("seed"), metric(m.get("accuracy")), metric(last_metric), window_length, metric(m.get("mean_nll")), m.get("count"), r.get("updates"), r.get("origin") or r.get("source_origin"), "复用/来源关联，非独立证据" if r.get("duplicate_source_of") else ("仅有aggregate" if r.get("seed") is None else "seed run"), r.get("summary_ref") or r.get("aggregate_ref")])
    write_csv(run / "input_memory_history.csv", hist_headers, hist_rows)
    raw_headers, raw_rows = raw_catalog_rows(norm)
    write_csv(run / "input_memory_history_all_records.csv", raw_headers, raw_rows)
    original_headers, original_rows = raw_catalog_rows(catalog.get("original_records", norm))
    write_csv(run / "input_memory_history_catalog_records.csv", original_headers, original_rows)

    arch_headers = ["阶段", "模型/架构", "参数量字段", "任务", "种子run数", "平均accuracy", "平均first100", "平均last100", "平均mean NLL", "来源说明"]
    binary_arch_rows = history_architecture_rows(hist_selected, "binary")
    decimal_arch_rows = history_architecture_rows(hist_selected, "decimal")
    write_csv(run / "input_memory_history_binary_architectures.csv", arch_headers, binary_arch_rows)
    write_csv(run / "input_memory_history_decimal_architectures.csv", arch_headers, decimal_arch_rows)

    figures_html = []
    for title, png, svg in figures:
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        figures_html.append(f'<figure><figcaption>{escape(title)}</figcaption><img alt="{escape(title)}" src="data:image/png;base64,{b64}"><p><a href="{escape(svg.name)}">SVG</a> · <a href="{escape(png.name)}">PNG</a></p></figure>')

    new_html = table_html(new_headers, new_rows, "new-results")
    hist_html = table_html(hist_headers, hist_rows, "history")
    binary_arch_html = table_html(arch_headers, binary_arch_rows, "binary-history-architectures")
    decimal_arch_html = table_html(arch_headers, decimal_arch_rows, "decimal-history-architectures")
    raw_headers, raw_rows = raw_catalog_rows(norm)
    raw_hist_html = table_html(raw_headers, raw_rows, "history-all-records")
    original_hist_count = len(original_rows)
    profile_headers = ["模型", "参数", "稳态 proxy", "全程 proxy", "相对 baseline 全程预算", "eligibility 元素", "float32 eligibility 字节"]
    profile_rows = []
    for kind in KINDS:
        p = profile.get(kind, {})
        base_budget = profile.get("baseline_rtrl20", {}).get("full_run_proxy")
        ratio = float(p["full_run_proxy"]) / float(base_budget) if p.get("full_run_proxy") is not None and base_budget else None
        profile_rows.append([KIND_NAMES[kind], p.get("parameters"), p.get("steady_proxy"), p.get("full_run_proxy"), f"{ratio:.3f}×" if ratio is not None else None, p.get("eligibility_elements"), p.get("eligibility_bytes_float32")])
    profile_html = table_html(profile_headers, profile_rows, "compute-profile")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>输入记忆 RTRL 报告</title>
<style>body{{font:15px/1.55 "Microsoft YaHei",sans-serif;color:#202a36;max-width:1500px;margin:24px auto;padding:0 20px}}h1,h2{{color:#233b64}}.note{{background:#f0f5fc;padding:12px 16px;border-left:4px solid #5275ad}}figure{{margin:26px 0}}figure img{{max-width:100%;height:auto;border:1px solid #ddd}}figcaption{{font-weight:bold}}.scroll{{overflow:auto;max-height:620px;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:6px 8px;border-bottom:1px solid #e3e7ed;text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#edf2f8}}.filter{{margin:8px 0;padding:8px;width:min(440px,90%)}}small{{color:#596574}}</style>
<h1>输入记忆 RTRL：新实验与历史对照</h1><p>运行目录：<code>{escape(str(run))}</code>　历史目录：<code>{escape(str(catalog_path))}</code></p>
<div class="note"><b>协议与解释</b><br>新实验为3种模型 × 9项任务 × 3个种子，共81次 run；每次600步、599次优化更新。三个 binary 任务和六个 decimal 任务分族绘图、分族比较。外部输入与20维上一时刻隐藏状态共30个源，各源一个指数记忆摘要；这不是每个时间步单独存一份记忆，也不保存无限长数字历史。隐藏状态 h 与快速权重 F 持续更新，RTRL 资格迹增加 E<sub>m</sub>。固定/学习记忆率的权重衰减为0，其余沿用协议；不加入 activity a。表中按任务列出相对 baseline 的准确率百分点差。新旧结果任务、模型与算力/更新数不完全相同，历史差异不能单独归因于输入记忆；不会用全局总体优势概括结果。IID 随机任务的命中率接近十类机会水平，不能解释成随机序列预测突破。</div>
<p>模型参数数：baseline 851（core 641），fixed_input_memory20 1451（core 1241），learned_input_memory20 1481（core 1271）。完成审计：<b>audit.json 已读取且 passed=true</b>。catalog 原始记录数为463；它们是阶段结果行，含种子级和聚合级记录，不能解释为463次独立训练。</p>
<h2>新实验聚合结果</h2><p><a href="input_memory_new_results.csv">下载新结果 CSV</a></p>{new_html}
<p>可学习记忆相对 baseline 在 decimal zigzag（+10.44 pp）和 decimal switching（+5.22 pp）提升；IID random 的 +1.39 pp 仍接近十类机会水平，不能算随机序列预测能力。五项低于 baseline，binary alternating 持平。结果不支持总体优势结论。</p>
<h2>计算与资格迹规模</h2><p>计算 proxy 是协议定义的算术代理，不等同硬件 FLOPs；eligibility 元素数与 float32 理论字节单独列出。</p>{profile_html}
<h2>学习曲线</h2>{''.join(figures_html)}
<h2>历史主要架构对照：二进制任务</h2><p>以下只汇总历史 binary alternating、triples、period16，按阶段/模型/任务跨 seed 求均值。包含权重衰减两种模式、shared/per_neuron 和19神经元线性读出等历史模型。不同阶段可能有不同训练设置，行间差异不能单独归因于模型架构。period16 历史参考中，MLP 为97.11%、TCN为94.89%、GRU为88.11%。binary_readout 的63/72次旧架构运行是同种子与输入下的从头重训；它与 representative_compute 的部分事件轨迹完全重现，表示确定性复现，不是来源文件复用，也没有增加新的种子证据。</p><p><a href="input_memory_history_binary_architectures.csv">下载二进制架构汇总 CSV</a></p>{binary_arch_html}
<h2>历史主要架构对照：十进制任务</h2><p>以下分别呈现 decimal_guess 与 linear_readout 的十进制历史任务，不与二进制任务混排；19神经元线性读出模型按独立模型名及参数量保留。Fibonacci mod 10 的历史 MLP 为89.06%，19神经元 custom_linear20 为66.39%。历史 run 多数是600位序列、1,797次更新；本轮为600步、599次更新，模型参数与预算也不同，仅作为任务背景参照。</p><p><a href="input_memory_history_decimal_architectures.csv">下载十进制架构汇总 CSV</a></p>{decimal_arch_html}
<h2>历史逐种子对照</h2><p>保留同一来源在不同阶段的重复呈现，并将已标记的来源复用注明为非独立证据。若同一阶段/模型/任务同时有 seed run 与 aggregate-only 行，表中优先使用 seed run，避免把聚合数重复计为一个 run。decimal_guess 与 linear_readout 的任务归入十进制族；其他阶段的 alternating/triples/period16 归入二进制族；legacy 原任务名原样保留。表格共{len(hist_rows)}条已去除同组 aggregate-only 重复的逐种子/聚合fallback记录。</p><p><a href="input_memory_history.csv">下载历史汇总对照 CSV</a></p>{hist_html}
<details><summary>规范化历史记录（{len(raw_rows)}条）</summary><p><a href="input_memory_history_all_records.csv">下载规范化完整历史 CSV</a></p>{raw_hist_html}</details><p>冻结原始 catalog 共有{original_count}条条目（包含错误命名或重复 aggregate 行）；完整原始快照另见<a href="input_memory_history_catalog_records.csv">冻结 catalog 原始记录 CSV（{original_hist_count}条）</a>。规范化依据见 <a href="history_catalog_normalized.json">normalized catalog</a>。</p>
<script>document.querySelectorAll('.filter').forEach(i=>i.addEventListener('input',()=>{{const q=i.value.toLowerCase();document.querySelectorAll('#'+i.dataset.table+' tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}}))</script></html>'''
    (run / "report.html").write_text(report, encoding="utf-8")
    print(run / "report.html")


if __name__ == "__main__":
    main()
