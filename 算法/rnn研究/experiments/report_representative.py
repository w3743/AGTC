"""Create an offline Chinese report for completed representative-model runs.

Run with matplotlib available, for example:
    PYTHONPATH=C:\\pioneer\\rnn-report-deps python report_representative.py RUN_DIRECTORY
This reads an audited run and writes HTML, CSV, PNG and SVG report artifacts.
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
TASK_NAMES = {"alternating": "Alternating", "triples": "Triples", "period16": "Period 16"}
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
    aggregate = load(root / "aggregate.json")
    summaries = load(root / "runs_summary.json")
    protocol = load(root / "protocol.json")
    matched = load(root / "matched_budgets.json")
    full = load(root / "full_run_budgets.json")
    latency = load(root / "cpu_latency.json")
    if len(aggregate) != 21 or len(summaries) != 63:
        raise ValueError("Expected 21 task aggregates and 63 run summaries")
    tasks = list(protocol.get("patterns", {}))
    seeds = protocol.get("seeds", [])
    if len(tasks) != 3 or len(seeds) != 3:
        raise ValueError("protocol must contain three tasks and three seeds")
    if set(matched) != set(KINDS) or set(full) != set(KINDS) or set(latency) != set(KINDS):
        raise ValueError("Budget and CPU latency files must cover all seven model kinds")
    if {row["kind"] for row in aggregate} != set(KINDS):
        raise ValueError("aggregate.json is missing one or more model kinds")
    for kind in KINDS:
        for task in tasks:
            if len([r for r in aggregate if r["kind"] == kind and r["task"] == task]) != 1:
                raise ValueError(f"Expected one aggregate for {kind}/{task}")
            selected = [r for r in summaries if r["kind"] == kind and r["task"] == task]
            if len(selected) != 3 or {r["seed"] for r in selected} != set(seeds):
                raise ValueError(f"Expected three paired seeds for {kind}/{task}")
    for row in summaries:
        if row.get("count") != 600:
            raise ValueError(f"Unexpected count in {row.get('run_directory')}")
    return aggregate, summaries, protocol, audit, matched, full, latency, tasks, seeds


def read_events(root: Path, summary):
    path = root / summary["run_directory"] / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600:
        raise ValueError(f"{path}: expected 600 events, found {len(rows)}")
    hits = [bool(row["hit"]) for row in rows]
    if any(int(row["step"]) != i for i, row in enumerate(rows, 1)):
        raise ValueError(f"{path}: event steps are not consecutive")
    return hits


def make_accuracy_plot(root: Path, aggregate, tasks):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    for ax, task in zip(axes, tasks):
        rows = {r["kind"]: r for r in aggregate if r["task"] == task}
        values = [rows[k]["accuracy"] for k in KINDS]
        bars = ax.bar(range(len(KINDS)), values, color=[COLORS[k] for k in KINDS], width=0.76)
        ax.set_xticks(range(len(KINDS)), [KIND_NAMES[k] for k in KINDS], rotation=34, ha="right")
        ax.set_title(TASK_NAMES.get(task, task))
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.1%}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Accuracy (all 1,800 predictions)")
    fig.suptitle("Prediction accuracy by task and model")
    fig.savefig(root / "model_accuracy.png", dpi=180)
    fig.savefig(root / "model_accuracy.svg")
    plt.close(fig)


def make_learning_plot(root: Path, summaries, tasks, seeds):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    csv_rows = []
    for ax, task in zip(axes, tasks):
        for kind in KINDS:
            seed_curves = []
            for seed in seeds:
                summary = next(s for s in summaries if s["kind"] == kind and s["task"] == task
                               and s["seed"] == seed)
                hits = read_events(root, summary)
                curve = [statistics.mean(hits[i - 100:i]) for i in range(100, len(hits) + 1)]
                seed_curves.append(curve)
            mean_curve = [statistics.mean(curve[i] for curve in seed_curves)
                          for i in range(len(seed_curves[0]))]
            steps = range(100, 601)
            ax.plot(steps, mean_curve, color=COLORS[kind], lw=1.8, label=KIND_NAMES[kind])
            csv_rows.extend({"task": task, "kind": kind, "step": step,
                             "rolling_100_accuracy_mean_3_seeds": value}
                            for step, value in zip(steps, mean_curve))
        ax.set_title(TASK_NAMES.get(task, task))
        ax.set_xlabel("Prediction step")
        ax.set_xlim(100, 600)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Rolling accuracy over the latest 100 predictions")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Learning curves: trailing 100-step accuracy, averaged over three seeds")
    fig.savefig(root / "learning_curves.png", dpi=180)
    fig.savefig(root / "learning_curves.svg")
    plt.close(fig)
    with (root / "learning_curves.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def _fmt(value, digits=3):
    return f"{value:,.{digits}f}"


def make_html(root: Path, aggregate, summaries, protocol, audit, matched, full, latency, tasks, seeds):
    accuracy_rows = []
    for task in tasks:
        values = {r["kind"]: r for r in aggregate if r["task"] == task}
        accuracy_rows.append(
            f"<tr><th>{escape(TASK_NAMES.get(task, task))}<br><code>{escape(task)}</code></th>" +
            "".join(f"<td>{values[k]['accuracy']:.2%}<br><small>末 100 位 {values[k]['last100_accuracy']:.2%}</small>"
                    f"<br>{values[k]['hits']:,}/1,800</td>" for k in KINDS) + "</tr>")
    budget_rows = []
    for kind in KINDS:
        config = matched[kind]["config"]
        architecture = config.get("architecture", {})
        total_proxy = matched[kind]["total_proxy"]
        full_row = full[kind]
        ms_mean = latency[kind]["serial_single_thread_mean_ms"]
        ms_median = latency[kind]["serial_single_thread_median_ms"]
        budget_rows.append(
            f"<tr><th>{escape(KIND_NAMES[kind])} <code>{kind}</code></th>"
            f"<td>{escape(str(config.get('width', '—')))}</td>"
            f"<td>{config.get('parameters', '—'):,}</td>"
            f"<td>{escape(str(architecture))}</td>"
            f"<td>{total_proxy:,.0f}<br>{matched[kind].get('ratio_to_custom', 1.0):.1%}</td>"
            f"<td>{full_row['total_proxy']:,.0f}<br>{full_row.get('ratio_to_custom', 1.0):.1%}</td>"
            f"<td>{_fmt(full_row['prediction_proxy'], 0)} / {_fmt(full_row['training_proxy'], 0)}</td>"
            f"<td>{ms_mean:.3f} / {ms_median:.3f}</td></tr>")
    seed_rows = ""
    for task in tasks:
        for kind in KINDS:
            row = next(r for r in aggregate if r["kind"] == kind and r["task"] == task)
            seed_rows += (f"<tr><td>{escape(TASK_NAMES.get(task, task))}</td><td>{escape(KIND_NAMES[kind])}</td>"
                          f"<td>{row['seed_accuracy_min']:.2%}</td><td>{row['seed_accuracy_max']:.2%}</td>"
                          f"<td>{row['mean_nll']:.6f}</td></tr>")
    nonlinear_cost = protocol.get("nonlinear_unit_cost", 1)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>代表序列模型基准报告</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#202c3c;background:#f4f6f9;max-width:1500px;margin:30px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}code{{background:#eef2f7;padding:2px 4px}}a{{color:#255fa5}}.scroll{{overflow-x:auto}}.note{{color:#596779;font-size:14px}}</style>
<h1>代表性序列模型：计算量匹配对照</h1>
<p>审计状态：passed · 完成状态：success · 7 种模型 · 3 个任务 · 3 个配对种子 · 63 次运行 · 37,800 次预测</p>
<section><h2>实验摘要</h2><p>各模型在每个任务上报告三个种子合计 1,800 次预测的准确率，并同时列出最后 100 位的准确率。学习曲线采用截至该步为止最近 100 次预测的命中率，并对三个种子逐点求均值；从第 100 步开始绘制。</p><p>模型尺寸通过匹配稳态窗口下的预测加三次更新计算预算选定。表中预算是统一计数器的算术标量操作 proxy，包括前向、反向和 Adam 更新：tanh、sigmoid 等基本逐元素非线性按每元素 {nonlinear_cost} 个单位计；softmax、LayerNorm、梯度范数和 loss 则依据文档化的标量算术公式计入多步操作。内存访问和搬运不计入。预算匹配容差为 ±5%。这些数值是计数规则下的估算 proxy，不是硬件 FLOPs，也不等同于硬件精确 FLOPs。</p><p>CPU 延迟由单线程 CPU 上独立串行 microbenchmark 测量预测加训练调用，包含 5 次预热和 20 次计时；它不是 63 次正式运行的并发 elapsed time。</p><p>这是完整方案对照：PlasticRNN 使用 input proximity 0.02，其他标准基线没有该特定 proximity 项；各模型共享 Adam 学习率 0.015、weight decay 1e-4 和梯度裁剪设置，PlasticRNN 的 retention 参数免除 weight decay。因此结果比较的是当前完整模型设计及固定训练方案，不是只改变架构的单因子消融。</p></section>
<section><h2>任务准确率</h2><div class="scroll"><table><tr><th>任务</th>{''.join(f'<th>{escape(KIND_NAMES[k])}</th>' for k in KINDS)}</tr>{''.join(accuracy_rows)}</table></div><p><a href="model_accuracy.svg">SVG 矢量图</a></p><img src="model_accuracy.png" alt="三项任务中七种模型的准确率分组柱状图"></section>
<section><h2>最近 100 位命中率曲线</h2><img src="learning_curves.png" alt="各任务中七种模型的滚动准确率曲线"><p>每条线是该模型在三个种子上最近 100 位准确率的逐点平均。<a href="learning_curves.svg">SVG 矢量图</a> · <a href="learning_curves.csv">逐步曲线 CSV</a></p></section>
<section><h2>参数量、尺寸与计算预算</h2><div class="scroll"><table><tr><th>模型</th><th>宽度</th><th>参数量</th><th>架构配置</th><th>稳态单步预算<br>总量 / 对 PlasticRNN 比</th><th>完整 600 步预算<br>总量 / 对 PlasticRNN 比</th><th>完整预算<br>预测 / 训练</th><th>CPU 毫秒<br>均值 / 中位数</th></tr>{''.join(budget_rows)}</table></div><p>“稳态单步预算”是固定 32 位历史下的一次预测加三次训练更新；“完整 600 步预算”按启动阶段不同历史长度求和。参数量和宽度由预算搜索选择，因此不要求各模型参数相同。</p></section>
<section><h2>逐任务种子范围与 NLL</h2><table><tr><th>任务</th><th>模型</th><th>种子准确率最小</th><th>种子准确率最大</th><th>平均 NLL</th></tr>{seed_rows}</table><p class="note">三种子统计是描述性结果；本报告不作显著性检验。</p></section>
<section><h2>解释范围与模型背景</h2><p>基准使用固定学习率 0.015，未进行超参数调优。只有三个种子，准确率区间用于展示种子间变化，不构成总体不确定性的精确估计。这里的 MLP、TCN、Transformer 与循环网络都是为本次任务设置的微型结构；结果不能代表更大模型或所有任务上的能力。简单循环网络在每步重放的 32 位窗口内运行，不应据此推断它们具备长程记忆能力。</p><p>模型结构的参考资料：<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html">PyTorch RNN</a> · <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html">PyTorch GRU</a> · <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html">PyTorch LSTM</a> · <a href="https://arxiv.org/abs/1803.01271">TCN: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling</a> · <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></p><details><summary>独立审计记录</summary><pre>{escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details></section>
<p><a href="aggregate.json">汇总 JSON</a> · <a href="runs_summary.json">运行摘要</a> · <a href="matched_budgets.json">匹配预算</a> · <a href="full_run_budgets.json">完整运行预算</a> · <a href="cpu_latency.json">CPU 延迟</a> · <a href="protocol.json">协议</a> · <a href="audit.json">独立审计</a> · <a href="completed.json">完成标记</a></p></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed, independently audited run directory")
    root = parser.parse_args().root
    aggregate, summaries, protocol, audit, matched, full, latency, tasks, seeds = validate_inputs(root)
    make_accuracy_plot(root, aggregate, tasks)
    make_learning_plot(root, summaries, tasks, seeds)
    make_html(root, aggregate, summaries, protocol, audit, matched, full, latency, tasks, seeds)
    print(root / "report.html")


if __name__ == "__main__":
    main()
