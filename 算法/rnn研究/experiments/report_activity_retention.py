"""Build an offline Chinese report for activity-retention RTRL experiments.

Usage: python report_activity_retention.py RUN_DIRECTORY
Writes report.html and PNG/SVG figures into RUN_DIRECTORY. Charts are embedded
in the HTML as data URIs, so the report remains self-contained offline.
"""
from __future__ import annotations

import argparse
import base64
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
CHINESE_FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [CHINESE_FONT]
plt.rcParams["axes.unicode_minus"] = False

KINDS = ("baseline_rtrl20", "fixed_activity20", "learned_activity20")
TASKS = {"alternating": "交替 01", "triples": "三连 000111", "period16": "周期16"}
NAMES = {"baseline_rtrl20": "基线 RTRL", "fixed_activity20": "固定活动保留", "learned_activity20": "可学习活动保留"}
COLORS = {"baseline_rtrl20": "#344e9a", "fixed_activity20": "#df9141", "learned_activity20": "#bd3e42"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for_events(root: Path, summaries: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    result = {}
    for summary in summaries:
        path = root / summary["run_directory"] / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(events) != 600 or [int(x["step"]) for x in events] != list(range(1, 601)):
            raise ValueError(f"事件缺失或 step 不连续：{path}")
        result[(summary["kind"], summary["task"], int(summary["seed"]))] = events
    return result


def rolling_hit(events: list[dict], window: int = 50):
    hits = [int(bool(row.get("hit", False))) for row in events]
    return [sum(hits[max(0, i-window+1):i+1]) / len(hits[max(0, i-window+1):i+1]) for i in range(len(hits))]


def save_curve(root: Path, events: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        for kind in KINDS:
            curves = [rolling_hit(events[(kind, task, int(seed))]) for seed in (2, 20261027, 20261028)]
            mean = [statistics.mean(x[i] for x in curves) for i in range(600)]
            ax.plot(range(1, 601), mean, color=COLORS[kind], label=NAMES[kind], linewidth=1.8)
        ax.set_title(TASKS[task]); ax.set_xlabel("预测步"); ax.set_ylim(0, 1.02); ax.grid(alpha=.2)
    axes[0].set_ylabel("命中率（50步窗口，种子均值）")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    png, svg = root / "activity_learning_curves.png", root / "activity_learning_curves.svg"
    fig.savefig(png, dpi=160); fig.savefig(svg); plt.close(fig)
    return png, svg


def save_retention(root: Path, events: dict):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for row, task in enumerate(TASKS):
        for col, (field, label) in enumerate((("a_after", "活动保留 a"), ("r_after", "痕迹保留 r"))):
            ax = axes[row, col]
            for kind in KINDS:
                vals = []
                for seed in (2, 20261027, 20261028):
                    event = events[(kind, task, seed)][-1]
                    val = event.get(field)
                    if val is not None:
                        vals.extend(val if isinstance(val, list) else [val])
                    elif kind == "baseline_rtrl20" and field == "a_after":
                        # The baseline has no activity-mixing path; zero is an equivalent display value.
                        vals.extend([0.0] * 20)
                if vals:
                    ax.hist(vals, bins=20, range=(0, 1), alpha=.48, label=NAMES[kind], color=COLORS[kind])
            ax.set_title(f"{TASKS[task]} · {label}"); ax.set_xlabel("系数值"); ax.set_ylabel("神经元数（合并种子）")
            ax.legend(fontsize=8)
    fig.tight_layout()
    png, svg = root / "activity_retention_distribution.png", root / "activity_retention_distribution.svg"
    fig.savefig(png, dpi=160); fig.savefig(svg); plt.close(fig)
    return png, svg


def image_data(path: Path):
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def validate(root: Path):
    aggregate = load(root / "aggregate.json")
    summaries = load(root / "runs_summary.json")
    audit = load(root / "audit.json") if (root / "audit.json").is_file() else {"passed": False, "status": "missing"}
    completed = load(root / "completed.json") if (root / "completed.json").is_file() else {}
    if len(aggregate) != 9 or len(summaries) != 27:
        raise ValueError("需要 9 条 kind×task 聚合和 27 条逐 seed 运行摘要")
    for kind in KINDS:
        for task in TASKS:
            ag = [r for r in aggregate if r.get("kind") == kind and r.get("task") == task]
            sr = [r for r in summaries if r.get("kind") == kind and r.get("task") == task]
            if len(ag) != 1 or len(sr) != 3 or {int(x["seed"]) for x in sr} != {2, 20261027, 20261028}:
                raise ValueError(f"缺少记录：{kind}/{task}")
            if any(int(x.get("count", 0)) != 600 for x in sr):
                raise ValueError(f"运行长度不是600：{kind}/{task}")
    return aggregate, summaries, audit, completed


def pct(value):
    return "—" if value is None else f"{100*float(value):.2f}%"


def comparison_summary(aggregate):
    out = []
    rows_by_kind = {kind: [r for r in aggregate if r.get("kind") == kind] for kind in KINDS}
    baseline_acc = sum(int(r["hits"]) for r in rows_by_kind["baseline_rtrl20"]) / sum(int(r["count"]) for r in rows_by_kind["baseline_rtrl20"])
    verdicts = []
    for kind in ("fixed_activity20", "learned_activity20"):
        rs = rows_by_kind[kind]
        pooled_acc = sum(int(r["hits"]) for r in rs) / sum(int(r["count"]) for r in rs)
        lower_tasks = sum(float(r["accuracy"]) < float(next(b for b in rows_by_kind["baseline_rtrl20"] if b["task"] == r["task"])["accuracy"]) for r in rs)
        verdicts.append(f"{escape(NAMES[kind])}在3项任务中有{lower_tasks}/3项准确率低于基线；合并准确率 {pct(pooled_acc)}，基线为 {pct(baseline_acc)}（差 {100*(pooled_acc-baseline_acc):+.2f} 个百分点）")
    for task, task_name in TASKS.items():
        rows = {r["kind"]: r for r in aggregate if r.get("task") == task}
        base = rows["baseline_rtrl20"]
        for kind, reference_kind in (("fixed_activity20", "baseline_rtrl20"),
                                     ("learned_activity20", "baseline_rtrl20"),
                                     ("learned_activity20", "fixed_activity20")):
            row = rows[kind]
            reference = rows[reference_kind]
            acc_delta = 100 * (float(row["accuracy"]) - float(reference["accuracy"]))
            nll_delta = float(row["mean_nll"]) - float(reference["mean_nll"])
            acc_text = (f"准确率高 {acc_delta:.2f} 个百分点" if acc_delta > 1e-9 else
                        f"准确率低 {abs(acc_delta):.2f} 个百分点" if acc_delta < -1e-9 else "准确率相同")
            nll_text = (f"NLL 高 {nll_delta:.4f}" if nll_delta > 1e-9 else
                        f"NLL 低 {abs(nll_delta):.4f}" if nll_delta < -1e-9 else "NLL 相同")
            out.append(f"<li><b>{escape(task_name)}：{escape(NAMES[kind])}</b> 相对{escape(NAMES[reference_kind])} {acc_text}，{nll_text}。</li>")
    return "<p><b>总体判断：</b>" + "；".join(verdicts) + "。本轮活动保留改动没有改善对基线的总体准确率；即使 learned 在个别任务胜过 fixed，也不能据此掩盖相对基线的退步。</p><ul>" + "".join(out) + "</ul>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="含 aggregate.json、runs_summary.json 的正式运行根目录")
    args = parser.parse_args(); root = args.root.resolve()
    aggregate, summaries, audit, completed = validate(root)
    events = rows_for_events(root, summaries)
    curve_png, curve_svg = save_curve(root, events)
    dist_png, dist_svg = save_retention(root, events)
    cells = []
    for task, task_name in TASKS.items():
        for kind in KINDS:
            row = next(r for r in aggregate if r["kind"] == kind and r["task"] == task)
            cells.append(f"<tr><td>{escape(task_name)}</td><td>{escape(NAMES[kind])}</td><td>{pct(row.get('accuracy'))}</td><td>{pct(row.get('first100_accuracy'))}</td><td>{pct(row.get('last100_accuracy'))}</td><td>{float(row.get('mean_nll', float('nan'))):.4f}</td><td>{pct(row.get('seed_accuracy_min'))}–{pct(row.get('seed_accuracy_max'))}</td></tr>")
    profile = load(root / "compute_profile.json") if (root / "compute_profile.json").is_file() else {}
    budget_rows = []
    for kind in KINDS:
        p = profile.get(kind, {})
        budget_rows.append(f"<tr><td>{escape(NAMES[kind])}</td><td>{p.get('parameters','—')}</td><td>{p.get('steady_proxy','—')}</td><td>{p.get('full_run_proxy','—')}</td><td>{p.get('eligibility_elements','—')}</td><td>{p.get('eligibility_bytes_float32','—')}</td><td>{p.get('serial_single_thread_mean_ms','—')}</td><td>{p.get('serial_single_thread_median_ms','—')}</td></tr>")
    audit_passed = audit.get("passed") is True
    audit_label = "独立审计通过" if audit_passed else "独立审计尚未通过或尚未完成"
    audit_color = "#16794b" if audit_passed else "#9a5b00"
    comparisons = comparison_summary(aggregate)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>活动保留系数实验报告</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;color:#202b38;line-height:1.65}}h1,h2{{color:#233b59}}section{{margin:30px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d8dee8;padding:8px;text-align:right}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}th{{background:#edf2f8}}.status{{padding:12px 16px;background:#f6f5ef;border-left:5px solid {audit_color};color:{audit_color};font-weight:700}}img{{max-width:100%;height:auto}}.note{{color:#596675}}</style>
<body><h1>活动保留系数：同更新预算下的机制对照</h1><p>运行根目录：<code>{escape(str(root))}</code></p><p class="status">审计状态：{audit_label}。{('audit.json 的 passed 字段为 true。' if audit_passed else '本报告不会把未通过或缺失的审计描述为通过。')}</p>
<section><h2>实验问题与协议</h2><p>本轮重新运行 baseline_rtrl20、fixed_activity20、learned_activity20 三种方案，覆盖交替、三连、周期16三个任务及种子 2、20261027、20261028。每次运行600步，共27次运行、16,200次预测；首位没有历史输入可供训练，因此每次运行599次参数更新。</p><p>基线沿用原式 h=candidate。两个活动保留方案使用 h=a·h<sub>prev</sub>+(1−a)·candidate；固定对照的20个 a 恒为0.5，可学习对照以20个独立logit学习 a=sigmoid(logit)。a 是状态更新中的常量/可学习系数，不是依赖输入的门控网络。快速连接痕迹 F 仍沿用原 r 痕迹递推式，只把原隐藏状态替换为新 h。其余优化设置保持原协议：核心中的 r 与 a 不使用 weight decay，其余参数采用 lr=0.015、Adam weight decay=1e-4、全局梯度裁剪1，输入软阈值系数0.02。</p><p>固定/可学习方案分别为851/871个训练参数；RTRL核心灵敏度分别为641/661维。固定方案不训练 a，但保留比例固定。每个方案的比较对象都是相同步数与更新次数，算术代理及串行单线程耗时另行列出；它们不表示同等计算量。</p></section>
<section><h2>关键结果摘要</h2><p>下列差值由 aggregate.json 中每个任务的准确率和平均NLL现场计算，相对同任务基线；准确率差以百分点表示。</p>{comparisons}<p class="note">这些是三种子描述性汇总，没有显著性推断。</p></section>
<section><h2>预测结果</h2><div style="overflow:auto"><table><thead><tr><th>任务</th><th>方案</th><th>累计准确率</th><th>首100步</th><th>末100步</th><th>平均NLL</th><th>种子准确率范围</th></tr></thead><tbody>{''.join(cells)}</tbody></table></div><img alt="50步窗口学习曲线" src="{image_data(curve_png)}"><p class="note">曲线为各种子50步滚动命中率再取均值；图表同时保存为 PNG 和 SVG。种子仅3个，范围是描述性统计。</p></section>
<section><h2>最终 a / r 系数分布</h2><img alt="活动保留与痕迹保留系数分布" src="{image_data(dist_png)}"><p class="note">分布取各运行最终事件中的 a_after 和 r_after，每个神经元每个种子只计一次。基线无活动混合通路，图中用 a=0 作等效表示。a 或 r 上升只表示状态/痕迹更新时更偏向保留前态，不能直接推出有用记忆增加或任务表现改善。</p></section>
<section><h2>计算与灵敏度预算</h2><div style="overflow:auto"><table><thead><tr><th>方案</th><th>参数量</th><th>单步 proxy</th><th>完整运行 proxy</th><th>eligibility 元素</th><th>float32 字节</th><th>单线程均值 ms</th><th>单线程中位数 ms</th></tr></thead><tbody>{''.join(budget_rows)}</tbody></table></div><p class="note">proxy 是约定下的计算代理，不等同硬件 FLOPs；字节数只估算 eligibility float32 存储。时间仅适用于给定设备与测量方法。</p></section>
<section><h2>审计与数据文件</h2><p>完成标记：<code>{escape(json.dumps(completed,ensure_ascii=False))}</code></p><details><summary>查看 audit.json</summary><pre>{escape(json.dumps(audit,ensure_ascii=False,indent=2))}</pre></details><p><a href="aggregate.json">aggregate.json</a> · <a href="runs_summary.json">runs_summary.json</a> · <a href="compute_profile.json">compute_profile.json</a> · <a href="audit.json">audit.json</a> · 图表：activity_learning_curves.png/.svg、activity_retention_distribution.png/.svg</p></section></body></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")
    print(root / "report.html")


if __name__ == "__main__":
    main()
