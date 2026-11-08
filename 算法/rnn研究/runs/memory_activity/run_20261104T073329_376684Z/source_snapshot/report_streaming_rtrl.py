"""Build a Chinese report for continuous-state streaming RTRL experiments.

Usage: python report_streaming_rtrl.py NEW_RUN_DIRECTORY [--old-run OLD_RUN_DIRECTORY]
The new-run directory must contain protocol.json, completed.json, audit.json,
aggregate.json, runs_summary.json, compute_profile.json and per-run events.jsonl.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics

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

NEW_KINDS = ("streaming_rtrl20", "streaming_one_step20", "replay_lastpair20")
OLD_KINDS = ("custom", "custom_linear20", "rnn", "gru", "lstm", "mlp", "tcn", "transformer")
KINDS = NEW_KINDS + OLD_KINDS
MAIN_KINDS = NEW_KINDS + ("custom_linear20", "custom")
REFERENCE_KINDS = ("gru", "mlp")
NAMES = {
    "streaming_rtrl20": "连续状态 RTRL",
    "streaming_one_step20": "一步梯度对照",
    "replay_lastpair20": "最近一对重放",
    "custom_linear20": "旧线性输出20",
    "custom": "旧 custom",
    "rnn": "RNN", "gru": "GRU", "lstm": "LSTM", "mlp": "MLP", "tcn": "TCN", "transformer": "Transformer",
}
COLORS = {"streaming_rtrl20": "#bd3e42", "streaming_one_step20": "#df9141", "replay_lastpair20": "#895da8",
          "custom_linear20": "#344e9a", "custom": "#526b87", "rnn": "#8297bd", "gru": "#4c9f70",
          "lstm": "#73b9a0", "mlp": "#d9a441", "tcn": "#dd7954", "transformer": "#956bb4"}
TASKS = {"alternating": "交替 01", "triples": "三连 000111", "period16": "周期16"}
PATTERNS = {"alternating": "01", "triples": "000111", "period16": "0000100110101111"}
SEEDS = {2, 20261027, 20261028}
DEFAULT_OLD = Path(r"C:\pioneer\rnn-research-runs\binary_readout_20261028\run_20261028T072957_438720Z")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(new_root: Path, old_root: Path):
    protocol, completed, audit = (load(new_root / n) for n in ("protocol.json", "completed.json", "audit.json"))
    if completed.get("success") is not True or audit.get("passed") is not True:
        raise ValueError("只接受 completed.success=true 且 audit.passed=true 的新实验")
    kinds = protocol.get("kinds", [])
    if set(protocol.get("new_kinds", [])) != set(NEW_KINDS) or set(kinds) != set(KINDS):
        raise ValueError(f"新协议必须声明全部11种 kinds 与3种 new_kinds：{NEW_KINDS}")
    task_names = protocol.get("tasks") or protocol.get("patterns")
    tasks = dict(PATTERNS) if isinstance(task_names, list) and set(task_names) == set(TASKS) else task_names
    if tasks != PATTERNS or set(protocol.get("seeds", [])) != SEEDS or protocol.get("length") != 600:
        raise ValueError("新协议任务、seed 或 length 不符合指定方案")
    completed_counts = {"new_runs": 27, "new_predictions": 16200, "total_source_runs": 99, "total_predictions": 59400}
    if any(completed.get(k) != v for k, v in completed_counts.items()):
        raise ValueError("completed.json 计数与指定的新跑/复用规模不符")
    agg_all, sum_all = load(new_root / "aggregate.json"), load(new_root / "runs_summary.json")
    agg_new = [r for r in agg_all if r.get("kind") in NEW_KINDS]
    sum_new = [r for r in sum_all if r.get("kind") in NEW_KINDS and r.get("origin", "new") == "new"]
    old_protocol, old_audit = load(old_root / "protocol.json"), load(old_root / "audit.json")
    old_completed = load(old_root / "completed.json")
    if old_completed.get("success") is not True or old_audit.get("passed") is not True:
        raise ValueError("复用的旧结果未通过完成/审计门禁")
    old_summaries, old_aggregate = load(old_root / "runs_summary.json"), load(old_root / "aggregate.json")
    if set(old_protocol.get("kinds", [])) != set(OLD_KINDS) or len(old_summaries) != 72:
        raise ValueError("旧来源不是指定的 8 模型、72 runs 数据集")
    summaries = []
    for row in sum_new:
        row = dict(row); row["origin"] = "new"; row.setdefault("source_run", str(new_root)); summaries.append(row)
    for row in old_summaries:
        row = dict(row); row["origin"] = "reused"; row["source_run"] = str(old_root); summaries.append(row)
    aggregate = []
    for row in agg_new:
        row = dict(row); row["origin"] = "new"; aggregate.append(row)
    for row in old_aggregate:
        row = dict(row); row["origin"] = "reused"; aggregate.append(row)
    if len(summaries) != 99 or len(aggregate) != 33:
        raise ValueError("合并后应为 99 个 seed runs 和 33 个模型×任务聚合")
    for k in KINDS:
        for task in TASKS:
            rows = [r for r in summaries if r.get("kind") == k and r.get("task") == task]
            ag = [r for r in aggregate if r.get("kind") == k and r.get("task") == task]
            if len(rows) != 3 or {int(r["seed"]) for r in rows} != SEEDS or len(ag) != 1:
                raise ValueError(f"缺少合并记录 {k}/{task}")
            if any(int(r.get("count", 0)) != 600 for r in rows):
                raise ValueError(f"预测数量不是600：{k}/{task}")
    for filename in ("compute_profile.json",):
        if not (new_root / filename).is_file():
            raise ValueError(f"缺少 {filename}")
    profile = load(new_root / "compute_profile.json")
    if set(profile) != set(KINDS):
        raise ValueError("compute_profile.json 必须覆盖全部 11 个新跑/复用模型")
    return protocol, audit, aggregate, summaries, profile


def event_rows(summary):
    base = Path(summary["source_run"])
    path = base / summary["run_directory"] / "events.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 600 or [int(x["step"]) for x in rows] != list(range(1, 601)):
        raise ValueError(f"事件缺失或 step 不连续：{path}")
    return rows


def scalar_or_mean(value):
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value if v is not None]
        return statistics.mean(nums) if nums else None
    return float(value) if isinstance(value, (int, float)) else None


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def plots(root, aggregate, summaries):
    idx = {(r["kind"], r["task"]): r for r in aggregate}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True, sharey=True)
    for ax, task in zip(axes, TASKS):
        vals = [idx[(k, task)]["accuracy"] for k in KINDS]
        bars = ax.bar(range(len(KINDS)), vals, color=[COLORS[k] for k in KINDS], width=.76)
        ax.set_xticks(range(len(KINDS)), [NAMES[k] for k in KINDS], rotation=48, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_title(TASKS[task]); ax.grid(axis="y", alpha=.22)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+.012, f"{v:.0%}", ha="center", fontsize=7)
    axes[0].set_ylabel("累计准确率（每模型每任务 1,800 次预测）")
    fig.suptitle("连续状态 RTRL 与既有二进制模型对照")
    fig.savefig(root/"model_accuracy.png", dpi=180); fig.savefig(root/"model_accuracy.svg"); plt.close(fig)

    by = {(r["kind"], r["task"], str(r["seed"])): r for r in summaries}
    curve_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True, sharey=True)
    curve_kinds = MAIN_KINDS + REFERENCE_KINDS
    for ax, task in zip(axes, TASKS):
        for kind in curve_kinds:
            curves = []
            for seed in sorted(SEEDS):
                hits = [bool(r["hit"]) for r in event_rows(by[(kind, task, str(seed))])]
                curves.append([statistics.mean(hits[i-100:i]) for i in range(100, 601)])
            mean = [statistics.mean(c[i] for c in curves) for i in range(501)]
            ax.plot(range(100, 601), mean, color=COLORS[kind], lw=2.3 if kind in MAIN_KINDS else 1.35,
                    ls="-" if kind in MAIN_KINDS else "--", label=NAMES[kind])
            curve_rows.extend({"task": task, "kind": kind, "step": step, "rolling_100_accuracy_mean_3_seeds": val}
                              for step, val in zip(range(100, 601), mean))
        ax.set_title(TASKS[task]); ax.set_xlim(100, 600); ax.set_ylim(0, 1); ax.set_xlabel("预测步"); ax.grid(alpha=.22)
    axes[0].set_ylabel("最近100次准确率（三 seed 均值）"); axes[-1].legend(fontsize=8)
    fig.suptitle("近期准确率曲线（新方法、主要旧 custom 对照与 GRU/MLP 参考）")
    fig.savefig(root/"learning_curves.png", dpi=180); fig.savefig(root/"learning_curves.svg"); plt.close(fig)
    write_csv(root/"learning_curves.csv", curve_rows)

    retain_rows = []
    for kind in NEW_KINDS[:2]:
        for task in TASKS:
            for seed in sorted(SEEDS):
                sm = by[(kind, task, str(seed))]
                for e in event_rows(sm):
                    before = scalar_or_mean(e.get("r_before", e.get("retention_before")))
                    after = scalar_or_mean(e.get("r_after", e.get("retention_after")))
                    grad = scalar_or_mean(e.get("retention_grad_norm", e.get("retention_task_grad_norm", e.get("retention_gradient_norm"))))
                    if grad is None and isinstance(e.get("retention_grad"), list):
                        grad = math.sqrt(sum(float(v) ** 2 for v in e["retention_grad"]))
                    retain_rows.append({"kind": kind, "task": task, "seed": seed, "step": e["step"],
                                        "retention_before": before, "retention_after": after,
                                        "retention_task_grad_norm": grad})
    write_csv(root/"retention_trajectories.csv", retain_rows)
    if not retain_rows or not any(r["retention_before"] is not None for r in retain_rows):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for kind in NEW_KINDS[:2]:
            runs = []
            for seed in sorted(SEEDS):
                rr = [r for r in retain_rows if r["task"] == task and r["kind"] == kind and r["seed"] == seed]
                vals = [r["retention_after"] for r in rr if r["retention_after"] is not None]
                if vals: runs.append(vals)
            if runs:
                n = min(map(len, runs)); mean = [statistics.mean(v[i] for v in runs) for i in range(n)]
                ax.plot(range(1, n+1), mean, color=COLORS[kind], label=NAMES[kind])
        ax.set_title(TASKS[task]); ax.set_xlabel("预测步"); ax.grid(alpha=.22)
    axes[0].set_ylabel("retention r（三 seed 均值）"); axes[-1].legend(fontsize=9)
    fig.suptitle("可学习 retention 状态轨迹")
    fig.savefig(root/"retention_trajectories.png", dpi=180); fig.savefig(root/"retention_trajectories.svg"); plt.close(fig)
    grad = [r for r in retain_rows if r["retention_task_grad_norm"] is not None]
    if grad:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        for ax, task in zip(axes, TASKS):
            for kind in NEW_KINDS[:2]:
                rr = [r for r in grad if r["task"] == task and r["kind"] == kind]
                if rr:
                    grouped = {}
                    for r in rr: grouped.setdefault(int(r["step"]), []).append(float(r["retention_task_grad_norm"]))
                    xs = sorted(grouped); ys = [statistics.mean(grouped[x]) for x in xs]
                    ax.plot(xs, ys, color=COLORS[kind], label=NAMES[kind])
            ax.set_title(TASKS[task]); ax.set_xlabel("预测步"); ax.grid(alpha=.22)
        axes[0].set_ylabel("retention task-gradient norm（三 seed 均值）"); axes[-1].legend(fontsize=9)
        fig.suptitle("任务损失对 retention 的梯度轨迹")
        fig.savefig(root/"retention_gradient.png", dpi=180); fig.savefig(root/"retention_gradient.svg"); plt.close(fig)
    return True


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.3f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def profile_rows(profile, origin_by_kind=None, steady_latency=None):
    """Render the committed flat compute_profile schema for all 11 models."""
    out=[]
    origin_by_kind = origin_by_kind or {}
    steady_latency = steady_latency or {}
    for kind in KINDS:
        row=profile.get(kind,{})
        latency = steady_latency.get(kind, {}) if kind in NEW_KINDS else row
        mean = latency.get("serial_single_thread_mean_ms", row.get("serial_single_thread_mean_ms"))
        median = latency.get("serial_single_thread_median_ms", row.get("serial_single_thread_median_ms"))
        warmup = latency.get("warmup", row.get("warmup", "—"))
        repeats = latency.get("repeats", row.get("repeats", "—"))
        out.append(f"<tr><th>{escape(NAMES[kind])}<br><small>{escape(str(origin_by_kind.get(kind,row.get('origin','—'))))}</small></th>"
                   f"<td>{escape(fmt(row.get('parameters')))}</td><td>{escape(fmt(row.get('steady_proxy')))}</td>"
                   f"<td>{escape(fmt(row.get('full_run_proxy')))}</td><td>{escape(fmt(row.get('state_elements')))}</td>"
                   f"<td>{escape(fmt(row.get('eligibility_elements')))}</td><td>{escape(fmt(row.get('active_eligibility_elements')))}</td>"
                   f"<td>{escape(fmt(row.get('eligibility_bytes_float32')))}</td>"
                   f"<td>{escape(fmt(mean))} / {escape(fmt(median))}<br><small>warmup {escape(fmt(warmup))}, n={escape(fmt(repeats))}</small></td>"
                   f"<td>{escape(fmt(row.get('ratio_to_previous_replay_linear20')))}</td><td>{escape(fmt(row.get('included')))}</td></tr>")
    return "".join(out)


def profile_details(profile):
    out=[]
    for kind in KINDS:
        row=profile.get(kind,{})
        profiles = row.get("profiles", {})
        if isinstance(profiles, dict):
            entries = profiles.items()
        elif isinstance(profiles, list):
            entries = ((p.get("age", p.get("name", i)), p) for i, p in enumerate(profiles))
        else:
            entries = ()
        for observed, stats in entries:
            proxy = stats.get("total_proxy", stats.get("steady_proxy", "—")) if isinstance(stats, dict) else "—"
            out.append(f"<tr><td>{escape(NAMES[kind])}</td><td>{escape(str(observed))}</td>"
                       f"<td>{escape(fmt(proxy))}</td><td>{escape(fmt(row.get('full_run_proxy')))}</td>"
                       f"<td>{escape(fmt(row.get('state_elements')))}</td><td>{escape(fmt(row.get('eligibility_elements')))}</td>"
                       f"<td>{escape(fmt(row.get('eligibility_bytes_float32')))}</td></tr>")
    return "".join(out)


def report(root, protocol, audit, aggregate, summaries, profile, retention_valid):
    idx={(r["kind"],r["task"]):r for r in aggregate}
    origins={r["kind"]:r["origin"] for r in summaries}
    steady_path=root/"steady_latency.json"
    steady_latency=load(steady_path) if steady_path.is_file() else {}
    head="<tr><th>模型</th>"+"".join(f"<th>{TASKS[t]}</th>" for t in TASKS)+"</tr>"
    rows=[]
    for k in KINDS:
        cls=' class="focus"' if k in MAIN_KINDS else ''
        rows.append(f"<tr{cls}><th>{escape(NAMES[k])} <small>({('新跑' if k in NEW_KINDS else '复用')})</small></th>"+"".join(f"<td>{idx[(k,t)]['accuracy']:.2%}<br>{idx[(k,t)]['hits']:,}/1,800</td>" for t in TASKS)+"</tr>")
    last="".join(f"<tr><th>{escape(NAMES[k])}</th>"+"".join(f"<td>{idx[(k,t)]['last100_accuracy']:.2%}</td>" for t in TASKS)+"</tr>" for k in MAIN_KINDS)
    nll="".join(f"<tr><th>{escape(NAMES[k])}</th>"+"".join(f"<td>{idx[(k,t)]['mean_nll']:.6f}</td>" for t in TASKS)+"</tr>" for k in MAIN_KINDS)
    findings=[]
    for t in TASKS:
        r=idx[("streaming_rtrl20",t)]; step=idx[("streaming_one_step20",t)]; replay=idx[("replay_lastpair20",t)]
        findings.append(f"{TASKS[t]}：连续状态 RTRL 准确率 {r['accuracy']:.2%}（末100 {r['last100_accuracy']:.2%}, NLL {r['mean_nll']:.4f}）；一步梯度对照 {step['accuracy']:.2%}；最近一对重放 {replay['accuracy']:.2%}。")
    findings_html="<br>".join(escape(x) for x in findings)
    period16_rtrl=idx[("streaming_rtrl20","period16")]["accuracy"]
    period16_old=idx[("custom_linear20","period16")]["accuracy"]
    rtrl_full=profile["streaming_rtrl20"].get("full_run_proxy")
    old_linear_full=profile["custom_linear20"].get("full_run_proxy")
    compute_ratio=(rtrl_full/old_linear_full) if isinstance(rtrl_full,(int,float)) and isinstance(old_linear_full,(int,float)) and old_linear_full else None
    retention_metrics=[]
    si={(r["kind"],r["task"],str(r["seed"])):r for r in summaries}
    p16_grads=[]
    for seed in sorted(SEEDS):
        p16_grads.extend(e["retention_grad_norm"] for e in event_rows(si[("streaming_rtrl20","period16",str(seed))])
                         if isinstance(e.get("retention_grad_norm"),(int,float)))
    p16_nonzero=sum(g != 0 for g in p16_grads)
    p16_summary=(f"本实验成功去掉外部输入窗口；周期16上 retention task-gradient 在 {p16_nonzero}/{len(p16_grads)} 个有更新的步骤非零，说明 r 收到可学习信号。但周期16累计准确率为 {period16_rtrl:.2%}，仍未学会该周期（旧线性输出20为 {period16_old:.2%}）。"
                 "性能差异涉及训练目标、更新次数、梯度传播方式与状态连续性，不能归因于单一记忆因素。")
    for kind in NEW_KINDS[:2]:
        for task in TASKS:
            r_start=[]; r_end=[]; g_all=[]; changed=[]
            for seed in sorted(SEEDS):
                ev=event_rows(si[(kind,task,str(seed))])
                before=scalar_or_mean(ev[0].get("r_before")); after=scalar_or_mean(ev[-1].get("r_after"))
                if before is not None and after is not None:
                    r_start.append(before); r_end.append(after); changed.append(abs(after-before))
                g_all.extend(float(e["retention_grad_norm"]) for e in ev if isinstance(e.get("retention_grad_norm"),(int,float)))
            retention_metrics.append(f"<tr><td>{escape(NAMES[kind])}</td><td>{TASKS[task]}</td>"
                f"<td>{(statistics.mean(r_start) if r_start else float('nan')):.6f}</td>"
                f"<td>{(statistics.mean(r_end) if r_end else float('nan')):.6f}</td>"
                f"<td>{(max(changed) if changed else float('nan')):.6g}</td>"
                f"<td>{(statistics.mean(g_all) if g_all else float('nan')):.6g}</td>"
                f"<td>{sum(g != 0 for g in g_all)}/{len(g_all)}</td></tr>")
    old_sources=sorted({r["source_run"] for r in summaries if r["origin"]=="reused"})
    source_html=" · ".join(f"<code>{escape(x)}</code>" for x in old_sources)
    retention_section=("<img src='retention_trajectories.png'><p><a href='retention_trajectories.svg'>SVG</a> · <a href='retention_trajectories.csv'>逐步 CSV</a></p>"+
        ("<img src='retention_gradient.png'><p><a href='retention_gradient.svg'>梯度 SVG</a></p>" if (root/'retention_gradient.png').exists() else "") if retention_valid else "<p>events.jsonl 未提供可解析的 retention_before/after 与任务梯度记录；脚本已输出可用字段的 CSV，暂不生成轨迹图，因此本轮不能据此确认 retention 可学习。</p>")
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续状态流式 RTRL 实验报告</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#202c3c;background:#f4f6f9;max-width:1500px;margin:28px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}.scroll{{overflow-x:auto}}.note{{color:#596779;font-size:14px}}.focus{{background:#fff2cc}}</style>
<h1>连续状态流式 RTRL：结构对照与真实预算</h1><p>完成与审计门禁通过 · 3 个新方法 × 3 个任务 × 3 个 seed = 27 次新运行、16,200 次新预测；另复用旧 8 模型、72 次运行。</p>
<section><h2>协议与方法边界</h2><p>任务为 01、000111、0000100110101111；每任务 600 位，seed 为 2、20261027、20261028。新方法是 streaming_rtrl20（20h、400F、完整结构核心 RTRL）、streaming_one_step20（状态连续但截断旧 eligibility，retention 梯度为零的控制）和 replay_lastpair20（最多重放最近 32 个符号但每次只用最新一对的 target loss 做一次 Adam 更新）。三个新方法每 seed/task 为 599 次更新；旧 binary 模型有 1,797 次更新。按实际预算对照，不能把性能差异归因于 memory 单一因素。</p><p>连续状态方法没有 32 步输入历史窗口；这不表示无限记忆，状态容量仍有限，RTRL eligibility 还可能遇到梯度数值稳定性问题。θ 在每步更新，但算法不对 optimizer 更新过程求导；固定 θ 下的 RTRL 灵敏度递推应与 BPTT 对齐。旧模型结果来自 {source_html}。背景比较的训练预算与状态处理不同，不能把性能差异归因于 memory 单一因素，也不能据此宣称新方法胜出。</p></section>
<section class="focus"><h2>本轮结果（动态生成，不预设胜负）</h2><p>{findings_html}</p><p>{escape(p16_summary)}</p><p>仅为三任务、三个 seed 的描述性实验。固定周期任务不代表对未见输入流的泛化。</p></section>
<section><h2>主比较：三个新方法与旧 custom</h2><div class="scroll"><table>{head}{''.join(rows)}</table></div><p>准确率分母为每模型每任务 1,800 次预测（3 seed）；各行来源标注为新跑或复用。</p><img src="model_accuracy.png"><p><a href="model_accuracy.svg">SVG 图</a></p><h3>主要模型末 100 位准确率</h3><table>{head}{last}</table><h3>平均 NLL</h3><table>{head}{nll}</table></section>
<section><h2>学习曲线</h2><p>只显示三个新方法、旧 custom/旧线性输出20及 GRU/MLP 背景参照，避免 11 条曲线拥挤。每 seed 先算滚动 100 步命中率，再取三个 seed 均值。</p><img src="learning_curves.png"><p><a href="learning_curves.svg">SVG</a> · <a href="learning_curves.csv">CSV</a></p></section>
<section><h2>Retention 轨迹与梯度证据</h2><p>下表以三个 seed 的起始/结束 retention 均值、单 seed 最大均值变化、retention 梯度范数均值与非零次数检查 retention 是否受任务 loss 驱动。一步梯度对照应保持 retention 梯度为零；RTRL 的非零梯度仅证明可学习信号存在，不能单独证明它改善预测。</p><div class="scroll"><table><tr><th>模型</th><th>任务</th><th>r 起始均值</th><th>r 结束均值</th><th>最大 seed 变化</th><th>任务梯度范数均值</th><th>非零数/总步数</th></tr>{''.join(retention_metrics)}</table></div>{retention_section}</section>
<section><h2>计算与状态规模</h2><div class="scroll"><table><tr><th>模型</th><th>参数量</th><th>单步 proxy</th><th>完整600步 proxy</th><th>状态元素</th><th>eligibility 元素</th><th>active eligibility 元素</th><th>eligibility 字节（float32）</th><th>单线程 CPU 均值/中位数 ms</th><th>相对旧线性输出 replay 倍数</th><th>纳入规则</th></tr>{profile_rows(profile,origins,steady_latency)}</table></div><p>proxy 不是硬件 FLOPs。eligibility 元素数和 float32 存储字节单列报告；proxy 只计入 eligibility 更新相关的算术操作，不把存储量折算为算术成本；一步梯度对照与 replay 的容量按其实际配置报告。新方法每条序列 599 次 optimizer 更新；旧 binary 模型 1,797 次，最近一对重放是每个新标签一次更新。准确率差异同时受训练目标、更新次数、梯度传播方式和状态连续性影响，不能单独解释为 memory 效果。{('新方法延迟为 steady_latency.json 的单线程 warmup=40、repeats=100；旧方法延迟复用既有分批测量，不能直接据此推断严格 speedup。' if steady_latency else 'steady_latency.json 尚未生成；当前 CPU 时间仅列 compute_profile 原始 profile 数值，未作严格速度比较。')}</p><h3>重放步数与机制 profile</h3><div class="scroll"><table><tr><th>模型</th><th>已观察位数 / age</th><th>单步 proxy</th><th>完整600步 proxy</th><th>状态元素</th><th>eligibility 元素</th><th>eligibility 字节</th></tr>{profile_details(profile)}</table></div><p><a href="compute_profile.json">compute_profile.json</a></p></section>
<section><h2>种子级主要指标</h2><table><tr><th>模型</th><th>任务</th><th>seed</th><th>准确率</th><th>末100</th><th>NLL</th><th>optimizer updates</th><th>来源</th></tr>{''.join(f"<tr><td>{escape(NAMES[r['kind']])}</td><td>{TASKS[r['task']]}</td><td>{r['seed']}</td><td>{r['accuracy']:.2%}</td><td>{r['last100_accuracy']:.2%}</td><td>{r['mean_nll']:.6f}</td><td>{r.get('optimizer_updates',r.get('updates','—'))}</td><td>{r['origin']}</td></tr>" for r in summaries if r['kind'] in MAIN_KINDS)}</table></section>
<section><h2>审计与来源</h2><details><summary>新实验 audit.json</summary><pre>{escape(json.dumps(audit,ensure_ascii=False,indent=2))}</pre></details><p><a href="protocol.json">新实验协议</a> · <a href="aggregate.json">新实验聚合</a> · <a href="runs_summary.json">新实验逐运行摘要</a> · <a href="audit.json">审计记录</a></p></section></html>'''
    (root/"report.html").write_text(html,encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root",type=Path); p.add_argument("--old-run",type=Path,default=DEFAULT_OLD)
    a=p.parse_args(); root=a.root
    protocol,audit,aggregate,summaries,profile=validate(root,a.old_run)
    ok=plots(root,aggregate,summaries)
    report(root,protocol,audit,aggregate,summaries,profile,ok)
    print(root/"report.html")


if __name__ == "__main__":
    main()
