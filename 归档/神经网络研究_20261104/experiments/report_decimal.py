"""Build a Chinese report for the audited next-digit benchmark (0–9).

Run with matplotlib installed in the report environment:
    python report_decimal.py RUN_DIRECTORY
The script reads a completed, independently audited run and writes report.html,
two plots (PNG/SVG), and learning_curves.csv beside the run data.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
CHINESE_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc").get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [CHINESE_FONT]
plt.rcParams["axes.unicode_minus"] = False

KINDS = ("custom", "rnn", "gru", "lstm", "mlp", "tcn", "transformer")
KIND_NAMES = {"custom": "PlasticRNN", "rnn": "RNN", "gru": "GRU", "lstm": "LSTM",
              "mlp": "Window MLP", "tcn": "Causal TCN", "transformer": "Causal Transformer"}
TASK_NAMES = {"cycle": "循环 0123456789", "triples": "每位数字连续三次", "zigzag": "往返锯齿",
              "fibonacci_mod10": "斐波那契模 10", "switching": "切换规则",
              "iid_random": "独立随机数字"}
COLORS = {"custom": "#344e9a", "rnn": "#6c8ebf", "gru": "#4c9f70", "lstm": "#73b9a0",
          "mlp": "#d9a441", "tcn": "#dd7954", "transformer": "#956bb4"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(root: Path):
    completed, audit = load(root / "completed.json"), load(root / "audit.json")
    if completed.get("success") is not True:
        raise ValueError("completed.json must report success=true")
    if audit.get("passed") is not True:
        raise ValueError("audit.json must report passed=true")
    aggregate, summaries = load(root / "aggregate.json"), load(root / "runs_summary.json")
    protocol = load(root / "protocol.json")
    matched, full, latency = (load(root / name) for name in
                              ("matched_budgets.json", "full_run_budgets.json", "cpu_latency.json"))
    sequences = load(root / "input_sequences.json")
    phases = load(root / "switching_phases.json")
    baselines = load(root / "baseline_scores.json")
    tasks = list(protocol.get("tasks", {}))
    seeds = protocol.get("seeds", [])
    if len(tasks) != 6 or len(seeds) != 3 or len(aggregate) != 42 or len(summaries) != 126:
        raise ValueError("Expected six tasks, three seeds, 42 aggregates and 126 run summaries")
    for mapping in (matched, full, latency):
        if set(mapping) != set(KINDS):
            raise ValueError("Budget and CPU latency files must cover all seven model kinds")
    if set(sequences) != set(tasks) or any(len(sequences[t]) != 600 for t in tasks):
        raise ValueError("input_sequences.json must contain 600 digits for each protocol task")
    if any(any(int(d) not in range(10) for d in sequences[t]) for t in tasks):
        raise ValueError("All input sequences must contain decimal digits 0..9")
    for kind in KINDS:
        for task in tasks:
            ag = [r for r in aggregate if r["kind"] == kind and r["task"] == task]
            sr = [r for r in summaries if r["kind"] == kind and r["task"] == task]
            if len(ag) != 1 or ag[0].get("count") != 1800:
                raise ValueError(f"Expected one 1,800-prediction aggregate for {kind}/{task}")
            if len(sr) != 3 or {r["seed"] for r in sr} != set(seeds):
                raise ValueError(f"Expected three paired seeds for {kind}/{task}")
    if any(r.get("count") != 600 for r in summaries):
        raise ValueError("Every run summary must report 600 predictions")
    switch = [r for r in phases if r.get("kind") in KINDS]
    if len(switch) != 21 or {r.get("phase") for r in switch} != {1, 2, 3}:
        raise ValueError("switching_phases.json must contain three phase summaries per model")
    if len(baselines) != 12 or {r.get("task") for r in baselines} != set(tasks) or \
            {r.get("kind") for r in baselines} != {"copy_last", "window_majority"}:
        raise ValueError("baseline_scores.json must contain copy_last and window_majority for all six tasks")
    return aggregate, summaries, protocol, audit, matched, full, latency, sequences, phases, baselines, tasks, seeds


def read_events(root: Path, summary, sequence):
    path = root / summary["run_directory"] / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600:
        raise ValueError(f"{path}: expected 600 events, found {len(rows)}")
    if any(int(row["step"]) != i for i, row in enumerate(rows, 1)):
        raise ValueError(f"{path}: event steps are not consecutive")
    hits = [bool(row["hit"]) for row in rows]
    # Require decimal labels whenever the event schema exposes them. The audit
    # record is responsible for checking sequence alignment and target timing.
    for i, row in enumerate(rows):
        target_key = next((k for k in ("target", "target_digit", "y") if k in row), None)
        if target_key is not None and int(row[target_key]) not in range(10):
            raise ValueError(f"{path}: target is outside decimal digits 0..9")
    return hits


def make_accuracy_plot(root, aggregate, tasks):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharey=True)
    for ax, task in zip(axes.flat, tasks):
        rows = {r["kind"]: r for r in aggregate if r["task"] == task}
        values = [rows[k]["accuracy"] for k in KINDS]
        bars = ax.bar(range(7), values, color=[COLORS[k] for k in KINDS], width=.76)
        ax.set_xticks(range(7), [KIND_NAMES[k] for k in KINDS], rotation=38, ha="right", fontsize=8)
        ax.set_title(TASK_NAMES.get(task, task))
        ax.set_ylim(0, 1.12); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.grid(axis="y", alpha=.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + .015, f"{value:.1%}", ha="center", fontsize=7)
    axes[0, 0].set_ylabel("准确率（每项任务 1,800 次预测）")
    axes[1, 0].set_ylabel("准确率（每项任务 1,800 次预测）")
    fig.suptitle("六项下一位数字任务：七种模型准确率")
    fig.savefig(root / "model_accuracy.png", dpi=180); fig.savefig(root / "model_accuracy.svg"); plt.close(fig)


def make_learning_plot(root, summaries, tasks, seeds, sequences):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharey=True)
    csv_rows = []
    for ax, task in zip(axes.flat, tasks):
        for kind in KINDS:
            curves = []
            for seed in seeds:
                summary = next(s for s in summaries if s["kind"] == kind and s["task"] == task and s["seed"] == seed)
                hits = read_events(root, summary, sequences[task])
                curves.append([statistics.mean(hits[i-100:i]) for i in range(100, 601)])
            mean_curve = [statistics.mean(c[i] for c in curves) for i in range(501)]
            steps = range(100, 601)
            ax.plot(steps, mean_curve, color=COLORS[kind], lw=1.6, label=KIND_NAMES[kind])
            csv_rows.extend({"task": task, "kind": kind, "step": step,
                             "rolling_100_accuracy_mean_3_seeds": val} for step, val in zip(steps, mean_curve))
        if task == "switching":
            ax.axvline(200, color="#333", ls="--", lw=1); ax.axvline(400, color="#333", ls="--", lw=1)
            ax.text(205, .96, "规则切换", fontsize=8, va="top"); ax.text(405, .96, "规则切换", fontsize=8, va="top")
        if task == "iid_random":
            ax.axhline(.1, color="#555", ls=":", lw=1.4, label="随机猜测期望 10%")
        ax.set_title(TASK_NAMES.get(task, task)); ax.set_xlabel("预测步")
        ax.set_xlim(100, 600); ax.set_ylim(0, 1); ax.grid(alpha=.22)
    axes[0, 0].set_ylabel("最近 100 次预测准确率"); axes[1, 0].set_ylabel("最近 100 次预测准确率")
    axes[-1, -1].legend(fontsize=7, loc="best")
    fig.suptitle("滚动 100 次准确率（三种子逐点平均）")
    fig.savefig(root / "learning_curves.png", dpi=180); fig.savefig(root / "learning_curves.svg"); plt.close(fig)
    with (root / "learning_curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)


def make_html(root, aggregate, summaries, protocol, audit, matched, full, latency, phases, baselines, tasks, seeds):
    head = "<tr><th>任务</th>" + "".join(f"<th>{escape(KIND_NAMES[k])}</th>" for k in KINDS) + "</tr>"
    acc_rows, last_rows = [], []
    for task in tasks:
        vals = {r["kind"]: r for r in aggregate if r["task"] == task}
        prefix = f"<tr><th>{escape(TASK_NAMES.get(task, task))}<br><code>{escape(task)}</code></th>"
        acc_rows.append(prefix + "".join(f"<td>{vals[k]['accuracy']:.2%}<br>{vals[k]['hits']:,}/1,800</td>" for k in KINDS) + "</tr>")
        last_rows.append(prefix + "".join(f"<td>{vals[k]['last100_accuracy']:.2%}</td>" for k in KINDS) + "</tr>")
    budget_rows = []
    for kind in KINDS:
        config = matched[kind]["config"]; arch = config.get("architecture", {})
        budget_rows.append(f"<tr><th>{escape(KIND_NAMES[kind])}</th><td>{config.get('width','—')}</td><td>{config.get('parameters','—'):,}</td>"
                           f"<td>{escape(str(arch))}</td><td>{matched[kind]['total_proxy']:,.0f} ({matched[kind].get('ratio_to_custom',1):.1%})</td>"
                           f"<td>{full[kind]['total_proxy']:,.0f} ({full[kind].get('ratio_to_custom',1):.1%})</td>"
                           f"<td>{latency[kind]['serial_single_thread_mean_ms']:.3f} / {latency[kind]['serial_single_thread_median_ms']:.3f}</td></tr>")
    phase_rows = []
    for kind in KINDS:
        by = {r["phase"]: r for r in phases if r["kind"] == kind}
        phase_rows.append(f"<tr><th>{escape(KIND_NAMES[kind])}</th>" + "".join(
            f"<td>{by[p]['accuracy']:.2%}<br>前30：{by[p]['first30_accuracy']:.2%}<br>后50：{by[p]['last50_accuracy']:.2%}<br>NLL {by[p]['mean_nll']:.4f}</td>" for p in (1,2,3)) + "</tr>")
    baseline_by = {(r["task"], r["kind"]): r for r in baselines}
    baseline_rows = []
    for task in tasks:
        copy = baseline_by[(task, "copy_last")]
        majority = baseline_by[(task, "window_majority")]
        baseline_rows.append(f"<tr><th>{escape(TASK_NAMES.get(task, task))}</th>"
                             f"<td>{copy['accuracy']:.2%}<br>末100：{copy['last100_accuracy']:.2%}</td>"
                             f"<td>{majority['accuracy']:.2%}<br>末100：{majority['last100_accuracy']:.2%}</td></tr>")
    nonlinear = protocol.get("nonlinear_unit_cost", 1)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>0–9 下一位预测基准报告</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#202c3c;background:#f4f6f9;max-width:1500px;margin:30px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}code{{background:#eef2f7;padding:2px 4px}}.scroll{{overflow-x:auto}}.note{{color:#596779;font-size:14px}}.new{{background:#fff2cc}}</style>
<h1>0–9 下一位数字预测：六任务计算预算匹配基准</h1><p>审计通过 · 完成运行 · 7 种模型 · 6 项任务 · 3 个配对种子 · 126 次运行 · 75,600 次预测</p>
<section><h2>任务准确率</h2><p>准确率按三个种子合计 1,800 次预测计算；下表单列每个运行末 100 位的合并准确率。</p><div class="scroll"><table>{head}{''.join(acc_rows)}</table><table><tr><th>任务</th>{''.join(f'<th>{escape(KIND_NAMES[k])}</th>' for k in KINDS)}</tr>{''.join(last_rows)}</table></div><img src="model_accuracy.png" alt="六任务七模型准确率分组柱状图"><p><a href="model_accuracy.svg">SVG 矢量图</a></p></section>
<section><h2>滚动学习曲线与随机对照</h2><p>滚动窗口为最近 100 次预测，三种子逐点求均值。切换任务在第 200、400 位处标示规则边界。独立随机任务的 10% 线表示均匀随机猜测下的期望值；它不是每段数据必须达到的精确阈值。</p><img src="learning_curves.png" alt="六任务最近100次预测准确率曲线"><p><a href="learning_curves.svg">SVG 矢量图</a> · <a href="learning_curves.csv">逐步曲线 CSV</a></p><p>下表列出复制上一位数字和最近 32 位众数两个简单参考；它们按任务计算，不含模型种子，不属于同算力神经网络排名。</p><div class="scroll"><table><tr><th>任务</th><th>复制上一位（准确率 / 末100）</th><th>最近32位众数（准确率 / 末100）</th></tr>{''.join(baseline_rows)}</table></div></section>
<section class="new"><h2>切换规则阶段指标</h2><p>阶段 1：递增循环；阶段 2：递减循环；阶段 3：步长 3 循环（0369258147）。每阶段 200 个预测，指标跨三个种子汇总为 600 个预测。前 30 和后 50 指标分别突出切换后的初期适应与阶段末表现。201 和 401 位边界不额外重置模型权重、优化器或历史窗口；每次预测仍按既定流程从零开始重放最多 32 位历史。</p><div class="scroll"><table><tr><th>模型</th><th>阶段 1：递增</th><th>阶段 2：递减</th><th>阶段 3：步长 3</th></tr>{''.join(phase_rows)}</table></div></section>
<section><h2>实验协议与解释范围</h2><p>六项任务：cycle 为 0123456789 循环；triples 为每位数字连续三次（000111…999）；zigzag 为 012345678987654321 往返；fibonacci_mod10 从 1、2 起按前两项之和模 10；switching 为 200 位递增、200 位递减、200 位步长 3 循环；iid_random 为随机种子 891732849 生成的独立随机数字。所有模型在同一任务上使用相同输入序列，模型训练种子为配对种子。</p><p>斐波那契模 10 任务只测试从初值 1、2 开始的一条固定递推序列。即使命中率较高，也可能来自记住这条序列；不能据此断言模型学会了可泛化到新初值的模 10 加法规则。</p><p>学习率固定且未调参。模型尺寸依据已验证的同尺寸预算 proxy 选择，容差 ±5%。预算计数包含前向、反向及 Adam 更新；逐元素非线性按每元素 {nonlinear} 个单位计入，softmax、LayerNorm、梯度范数与 loss 按文档化算术计入；内存搬运不计。这些值是统一规则下的算术 proxy，不是硬件 FLOPs。CPU 延迟沿用上一轮同实现的单线程微基准，本轮未重测，仅作参考。</p><p>PlasticRNN 使用 input proximity 0.02，标准基线不带该项，因此这是完整方案对照而非仅改变架构的单变量消融。随机猜测 10% 是期望值而非精确门槛。每个模型种子共享相同的 600 位随机序列，所以跨模型/种子的 1,800 次预测并非 1,800 个独立随机样本，不能据此作独立样本推断。每次预测都会从零重放最多 32 位历史，因此结果不能证明模型具备超出该上下文窗口的长程记忆能力。</p></section>
<section><h2>模型尺寸与预算</h2><div class="scroll"><table><tr><th>模型</th><th>宽度</th><th>参数</th><th>架构配置</th><th>匹配预算 proxy（相对 custom）</th><th>完整 600 步 proxy（相对 custom）</th><th>CPU ms 均值/中位数</th></tr>{''.join(budget_rows)}</table></div></section>
<section><h2>逐任务种子范围与 NLL</h2><table><tr><th>任务</th><th>模型</th><th>准确率最小/最大</th><th>平均 NLL</th></tr>{''.join(f"<tr><td>{escape(TASK_NAMES.get(r['task'],r['task']))}</td><td>{escape(KIND_NAMES[r['kind']])}</td><td>{r['seed_accuracy_min']:.2%} / {r['seed_accuracy_max']:.2%}</td><td>{r['mean_nll']:.6f}</td></tr>" for r in aggregate)}</table><p class="note">三个种子的范围为描述性结果，不是总体不确定性的精确估计。</p><details><summary>独立审计记录</summary><pre>{escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details></section>
<p><a href="aggregate.json">汇总</a> · <a href="runs_summary.json">运行摘要</a> · <a href="protocol.json">协议</a> · <a href="input_sequences.json">输入序列</a> · <a href="baseline_scores.json">简单参考分数</a> · <a href="switching_phases.json">切换阶段指标</a> · <a href="audit.json">审计</a></p></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed and independently audited run directory")
    root = parser.parse_args().root
    aggregate, summaries, protocol, audit, matched, full, latency, sequences, phases, baselines, tasks, seeds = validate_inputs(root)
    make_accuracy_plot(root, aggregate, tasks)
    make_learning_plot(root, summaries, tasks, seeds, sequences)
    make_html(root, aggregate, summaries, protocol, audit, matched, full, latency, phases, baselines, tasks, seeds)
    print(root / "report.html")


if __name__ == "__main__":
    main()
