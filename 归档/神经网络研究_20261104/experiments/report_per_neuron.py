"""Build an offline Chinese report and scientific plots for per-neuron retention runs.

Run with a Python environment that has matplotlib available, for example:
    PYTHONPATH=C:\\pioneer\\rnn-report-deps python report_per_neuron.py RUN_DIRECTORY
This script only reads completed run data and writes report artifacts into RUN_DIRECTORY.
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

# Register a known Windows Chinese font explicitly so headless rendering does
# not fall back to a font with missing CJK glyphs.
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
CHINESE_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc").get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [CHINESE_FONT]
plt.rcParams["axes.unicode_minus"] = False


TASK_NAMES = {"alternating": "交替", "triples": "三连", "period16": "16 位循环"}
MODE_NAMES = {"shared": "共享 r（623 参数）", "per_neuron": "逐接收神经元 r（642 参数）"}
MODE_COLORS = {"shared": "#3266b5", "per_neuron": "#d76a22"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def vector(value, expected_length: int, label: str):
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{label}: expected {expected_length} retention values")
    return [float(x) for x in value]


def validate_inputs(root: Path):
    completed = load(root / "completed.json")
    audit = load(root / "audit.json")
    if completed.get("success") is not True:
        raise ValueError("completed.json does not report success=true")
    if audit.get("passed") is not True:
        raise ValueError("audit.json does not report passed=true")
    aggregate = load(root / "aggregate.json")
    summaries = load(root / "runs_summary.json")
    protocol = load(root / "protocol.json")
    if len(summaries) != 18 or len(aggregate) != 3:
        raise ValueError("Expected 18 run summaries and 3 task aggregates")
    expected_seeds = protocol.get("seeds", [])
    if len(expected_seeds) != 3:
        raise ValueError("protocol.json must list the three paired seeds")
    tasks = [item["task"] for item in aggregate]
    if len(set(tasks)) != 3:
        raise ValueError("aggregate.json must contain three distinct tasks")
    for task in tasks:
        for seed in expected_seeds:
            for mode, n in (("shared", 1), ("per_neuron", 20)):
                matches = [s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == mode]
                if len(matches) != 1:
                    raise ValueError(f"Expected one summary for {task}/{seed}/{mode}")
                vector(matches[0]["final_retention"], n, f"{task}/{seed}/{mode}")
    return aggregate, summaries, protocol, audit


def read_events(root: Path, summary):
    path = root / summary["run_directory"] / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600:
        raise ValueError(f"{path}: expected 600 events, found {len(rows)}")
    n = 1 if summary["mode"] == "shared" else 20
    start = vector(rows[0]["retention_before"], n, f"{path}: initial retention")
    trace = [start]
    for index, event in enumerate(rows, 1):
        trace.append(vector(event["retention_after"], n, f"{path}: event {index}"))
    return trace


def make_trajectory_plot(root: Path, aggregate, summaries, seeds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharey=True)
    csv_rows = []
    for ax, item in zip(axes, aggregate):
        task = item["task"]
        for mode in ("shared", "per_neuron"):
            seed_traces = []
            for seed in seeds:
                summary = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == mode)
                raw = read_events(root, summary)
                # Shared has one value; per-neuron curves show each seed's mean across its 20 receivers.
                trace = [statistics.mean(step) for step in raw]
                seed_traces.append(trace)
                ax.plot(range(601), trace, color=MODE_COLORS[mode], alpha=0.28, lw=1.0)
                csv_rows.extend({"task": task, "mode": mode, "seed": seed, "step": step,
                                 "retention_mean_across_receivers": value}
                                for step, value in enumerate(trace))
            mean_trace = [statistics.mean(trace[i] for trace in seed_traces) for i in range(601)]
            ax.plot(range(601), mean_trace, color=MODE_COLORS[mode], lw=2.4,
                    label=f"{MODE_NAMES[mode]}：3 个种子均值")
        ax.axhline(0.5, color="#858b94", ls="--", lw=0.9)
        ax.set_title(TASK_NAMES.get(task, task) + " repeated")
        ax.set_xlabel("输入位数 / 训练步")
        ax.grid(alpha=0.2)
        ax.set_ylim(0.35, 1.0)
    axes[0].set_ylabel("保留比例 r（逐接收神经元后取均值）")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("共享与逐接收神经元保留参数的轨迹\n细线：各随机种子的接收行均值；粗线：3 个种子的均值", fontsize=12)
    fig.savefig(root / "retention_trajectories.png", dpi=170)
    fig.savefig(root / "retention_trajectories.svg")
    plt.close(fig)
    with (root / "retention_trajectories.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def make_final_plot(root: Path, aggregate, summaries, seeds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), constrained_layout=True, sharey=True)
    offsets = {seed: (idx - (len(seeds) - 1) / 2) * 0.13 for idx, seed in enumerate(seeds)}
    markers = ["o", "s", "^"]
    for ax, item in zip(axes, aggregate):
        task = item["task"]
        for si, seed in enumerate(seeds):
            shared = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == "shared")
            independent = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == "per_neuron")
            value = vector(shared["final_retention"], 1, "shared final_retention")[0]
            x = [i + 1 + offsets[seed] for i in range(20)]
            y = vector(independent["final_retention"], 20, "per_neuron final_retention")
            ax.scatter(x, y, s=23, marker=markers[si % len(markers)], color=MODE_COLORS["per_neuron"],
                       alpha=0.72, label=f"逐神经元，seed {seed}" if ax is axes[0] else None)
            ax.hlines(value, 0.55, 20.45, color=f"C{si}", linestyle="--", linewidth=1.35,
                      label=f"共享 r，seed {seed}" if ax is axes[0] else None)
        ax.set_title(TASK_NAMES.get(task, task))
        ax.set_xlabel("接收神经元索引（1–20）")
        ax.set_xticks(range(1, 21, 2))
        ax.grid(alpha=0.2)
        ax.set_xlim(0.5, 20.5)
    axes[0].set_ylabel("最终保留比例 r")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("逐神经元最终 r 与同种子共享 r 对照\n散点为各接收行；虚线为该种子的共享参数", fontsize=12)
    fig.savefig(root / "per_neuron_final.png", dpi=170)
    fig.savefig(root / "per_neuron_final.svg")
    plt.close(fig)


def make_html(root: Path, aggregate, summaries, protocol, audit):
    seeds = protocol["seeds"]
    task_rows = ""
    for item in aggregate:
        task = item["task"]
        shared, per = item["shared"], item["per_neuron"]
        task_rows += (f"<tr><td>{escape(TASK_NAMES.get(task, task))} <code>{escape(task)}</code></td>"
                      f"<td>{shared['accuracy']:.2%}<br>{shared['hits']}/1800</td>"
                      f"<td>{per['accuracy']:.2%}<br>{per['hits']}/1800</td>"
                      f"<td>{item['accuracy_delta_pp']:+.4f}</td>"
                      f"<td>{shared['last100_accuracy']:.2%} / {per['last100_accuracy']:.2%}</td>"
                      f"<td>{shared['mean_nll']:.6f} / {per['mean_nll']:.6f}</td>"
                      f"<td>{shared['final_retention_mean']:.6f} / {per['final_retention_mean']:.6f}<br>"
                      f"神经元内标准差的三种子均值：{shared['final_retention_std']:.6f} / "
                      f"{per['final_retention_std']:.6f}</td>"
                      f"<td>{per['retention_min']:.6f} – {per['retention_max']:.6f}</td></tr>")
    seed_tables = ""
    for item in aggregate:
        task = item["task"]
        seed_tables += f"<h3>{escape(TASK_NAMES.get(task, task))} <code>{escape(task)}</code></h3>"
        seed_tables += "<table><tr><th>种子</th><th>模式</th><th>最终 r（接收神经元索引 1–20）</th></tr>"
        for seed in seeds:
            for mode in ("shared", "per_neuron"):
                summary = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == mode)
                vals = vector(summary["final_retention"], 1 if mode == "shared" else 20, "final_retention")
                if mode == "shared":
                    display = f"共享值：{vals[0]:.7f}"
                else:
                    display = "<br>".join(f"神经元 {i}: {v:.7f}" for i, v in enumerate(vals, 1))
                seed_tables += f"<tr><td>{seed}</td><td>{MODE_NAMES[mode]}</td><td>{display}</td></tr>"
        seed_tables += "</table>"
    mode_counts = protocol.get("modes", {"shared": 623, "per_neuron": 642})
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>逐神经元遗忘参数对照</title>
<style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#202c3c;background:#f5f7fa;max-width:1320px;margin:32px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 24px}}td,th{{padding:10px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}code{{background:#eef2f7;padding:2px 4px}}a{{color:#255fa5}}.scroll{{overflow-x:auto}}</style>
<h1>共享与逐接收神经元遗忘参数对照</h1>
<p>任务 {len(aggregate)} 个 · 配对种子 {escape(str(seeds))} · 18 次运行 · 10,800 次预测 · 两组遗忘参数的 weight decay 均为 0</p>
<section><h2>实验协议</h2><p>共享模式含 {mode_counts['shared']} 个参数；逐接收神经元模式含 {mode_counts['per_neuron']} 个参数。改动是把一个共享遗忘 logit 拆为 20 个接收行各自学习的 logit。两组均沿用 400 份连接短期痕迹，输入序列每个任务重复 600 位。</p>
<p>每步先预测并计分，再训练；每次按最新 32 位历史从零重放，每个输入 3 次 Adam 更新，学习率 0.015，梯度裁剪 1.0。初始保留比例为 0.8。任务指标按 3 个种子合并为 1,800 次预测。种子均值为描述性统计，未作显著性检验。</p></section>
<section><h2>任务汇总</h2><div class="scroll"><table><tr><th>任务</th><th>共享准确率</th><th>逐神经元准确率</th><th>准确率差<br>百分点</th><th>末 100 步准确率<br>共享 / 逐神经元</th><th>平均 NLL<br>共享 / 逐神经元</th><th>最终 r 均值<br>共享 / 逐神经元<br>；神经元内标准差的三种子均值</th><th>逐神经元 r 范围</th></tr>{task_rows}</table><p>表中标准差项是每个种子内部 20 个接收神经元 r 的标准差，再对 3 个种子取均值；它不是将 60 个 r 合并后的 pooled 标准差。共享 r 每种子只有一个值，故其神经元内标准差为 0。</p></div></section>
<section><h2>训练中的保留比例</h2><img src="retention_trajectories.png" alt="三个任务的逐步保留比例轨迹"><p>逐神经元曲线每步先对 20 个接收行取均值。细线为单个种子，粗线为三个种子的均值。<a href="retention_trajectories.svg">SVG 矢量图</a> · <a href="retention_trajectories.csv">轨迹 CSV</a></p></section>
<section><h2>最终逐神经元 r</h2><img src="per_neuron_final.png" alt="三个任务的逐神经元最终保留比例"><p>散点对应接收神经元索引；虚线对应同种子的共享 r。跨种子的神经元索引不作语义对齐。<a href="per_neuron_final.svg">SVG 矢量图</a></p><p><a href="final_retention.csv">下载 runner 生成的 final_retention.csv</a></p></section>
<section><h2>各任务、种子和神经元的最终值</h2>{seed_tables}</section>
<section><h2>解释边界</h2><p>r 是短期痕迹更新中的保留比例，不代表实际可记忆步数。32 位重放窗口没有改变；每次重放从零开始，实验没有增加主动清空痕迹操作。逐神经元参数的索引对应接收行，但不同随机种子之间不建立神经元语义对应关系。</p><details><summary>独立审计结果</summary><pre>{escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details></section>
<p><a href="runs_summary.csv">运行汇总 CSV</a> · <a href="aggregate.json">汇总 JSON</a> · <a href="protocol.json">协议</a> · <a href="audit.json">独立审计</a> · <a href="completed.json">完成标记</a></p></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="completed per-neuron run directory")
    args = parser.parse_args()
    root = args.root
    aggregate, summaries, protocol, audit = validate_inputs(root)
    make_trajectory_plot(root, aggregate, summaries, protocol["seeds"])
    make_final_plot(root, aggregate, summaries, protocol["seeds"])
    make_html(root, aggregate, summaries, protocol, audit)
    print(root / "report.html")


if __name__ == "__main__":
    main()
