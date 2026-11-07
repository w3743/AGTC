"""Render an offline HTML report and scientific plots from completed raw runs."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    assert load(root / "completed.json")["success"]
    aggregate = load(root / "aggregate.json")
    summaries = load(root / "runs_summary.json")
    protocol = load(root / "protocol.json")
    preflight = load(root / "preflight.json")
    audit = load(root / "audit.json")
    colors = {"with_decay": "#3266b5", "without_decay": "#d76a22"}
    labels = {"with_decay": "Retention WD = 0.0001", "without_decay": "Retention WD = 0"}
    names = {"alternating": "交替", "triples": "三连", "period16": "16 位循环"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True, sharey=True)
    trajectory_rows = []
    for ax, item in zip(axes, aggregate):
        task = item["task"]
        for mode in colors:
            paths = [s for s in summaries if s["task"] == task and s["mode"] == mode]
            traces = []
            for summary in paths:
                events = [json.loads(line) for line in (root / summary["run_directory"] / "events.jsonl").read_text().splitlines()]
                trace = [events[0]["retention_before"]] + [e["retention_after"] for e in events]
                traces.append(trace)
                ax.plot(range(601), trace, color=colors[mode], alpha=0.25, lw=0.9)
                for step, retention in enumerate(trace):
                    trajectory_rows.append({"task": task, "mode": mode, "seed": summary["seed"], "step": step, "retention": retention})
            mean = [statistics.mean(t[i] for t in traces) for i in range(601)]
            ax.plot(range(601), mean, color=colors[mode], lw=2.2, label=labels[mode])
        ax.axhline(0.5, color="#8c929a", ls="--", lw=1)
        ax.set_title(item["pattern"] + " repeated")
        ax.set_xlabel("Observed bits / training presses")
        ax.grid(alpha=0.18)
        ax.set_ylim(0.35, 1.0)
    axes[0].set_ylabel("Trace retention r (higher = slower decay)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Effect of removing weight decay from the shared retention logit\nThin lines: paired seeds; thick lines: mean of 3 seeds", fontsize=12)
    fig.savefig(root / "retention_trajectories.png", dpi=160)
    fig.savefig(root / "retention_trajectories.svg")
    plt.close(fig)
    with (root / "retention_trajectories.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    table = ""
    for item in aggregate:
        a, b = item["with_decay"], item["without_decay"]
        table += (f'<tr><td>{names[item["task"]]} <code>{item["pattern"]}</code></td>'
                  f'<td>{a["accuracy"]:.2%}<br>{a["hits"]}/1800</td>'
                  f'<td>{b["accuracy"]:.2%}<br>{b["hits"]}/1800</td>'
                  f'<td>{item["accuracy_delta_pp"]:+.4f}</td>'
                  f'<td>{a["last100_accuracy"]:.2%} / {b["last100_accuracy"]:.2%}</td>'
                  f'<td>{a["final_retention"]:.6f} / {b["final_retention"]:.6f}</td>'
                  f'<td>{a["mean_nll"]:.6f} / {b["mean_nll"]:.6f}</td></tr>')
    paired = ""
    for task in protocol["patterns"]:
        for seed in protocol["seeds"]:
            a = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == "with_decay")
            b = next(s for s in summaries if s["task"] == task and s["seed"] == seed and s["mode"] == "without_decay")
            paired += (f'<tr><td>{names[task]}</td><td>{seed}</td><td>{a["hits"]} / {b["hits"]}</td>'
                       f'<td>{a["final_retention"]:.6f} / {b["final_retention"]:.6f}</td>'
                       f'<td>{a["final_eta"]:.6f} / {b["final_eta"]:.6f}</td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>遗忘参数权重衰减对照 · 2026-10-28</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#202c3c;background:#f6f8fb;max-width:1280px;margin:40px auto;padding:0 24px}}section{{background:white;padding:24px;border:1px solid #dce2eb;border-radius:12px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #dde3eb;text-align:left}}th{{background:#edf2f8}}img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}code{{background:#eef2f7;padding:2px 4px}}a{{color:#255fa5}}</style>
<h1>只去掉遗忘参数的权重衰减</h1><p>2026-10-28 · 分支 <code>codex/retention-decay-ablation</code> · 18 次运行 · 10,800 次预测</p>
<section><h2>实验协议</h2><p>两组都使用共享的遗忘参数，623 个参数、20 个神经元、10 类输出。唯一变量：<code>decay_logit</code> 的 Adam weight_decay 为 0.0001 或 0。其他参数仍为 0.0001；32 位窗口从零重放，每键 3 次 Adam 更新，学习率 0.015，梯度裁剪 1，输入软阈值收缩 0.02。初始保留比例 0.8。</p>
<p>配对种子：{escape(str(protocol['seeds']))}。每种模式重复输入 600 位。每步先预测和计分，再输入答案训练；包含冷启动均匀分布，平分时取最小类别。因此第 1 次预测为 0。每次运行 1,797 次更新。三种子均值是描述性结果，未作显著性检验。</p></section>
<section><h2>命中率与最终保留比例</h2><p>“旧 / 新”分别表示保留 / 去掉遗忘参数权重衰减。保留比例越高，旧痕迹衰减越慢；它不是实际可记忆步数。</p><table><tr><th>模式</th><th>旧命中率</th><th>新命中率</th><th>变化（百分点）</th><th>末 100 命中率<br>旧 / 新</th><th>最终 r 均值<br>旧 / 新</th><th>平均 NLL<br>旧 / 新</th></tr>{table}</table></section>
<section><h2>保留比例随训练变化</h2><img src="retention_trajectories.png" alt="三个任务中保留比例随600次输入变化的配对曲线"><p>细线为各随机种子，粗线为均值；虚线表示 0.5。<a href="retention_trajectories.svg">矢量图</a> · <a href="retention_trajectories.csv">完整轨迹 CSV</a></p></section>
<section><h2>每个种子的配对结果</h2><table><tr><th>任务</th><th>种子</th><th>命中数 / 600<br>旧 / 新</th><th>最终 r<br>旧 / 新</th><th>最终痕迹作用系数 eta<br>旧 / 新</th></tr>{paired}</table></section>
<section><h2>核验与解释边界</h2><p>旧设置在 120 步预测、训练和参数比较中与原始单组优化器逐位一致；该源文件 SHA-256 与历史记录相同。零任务梯度、全新优化器的 10 次更新中，旧 r 从 0.8 变为 {preflight['zero_task_gradient']['with_decay']['after_10_zero_gradient_updates']:.6f}，新 r 保持 0.8。</p>
<p>去掉正则后仍可能因任务梯度改变保留比例；结果也仍受任务、32 位重放和其他参数学习的影响。该实验未改变共享参数为逐神经元参数，未验证长期记忆，未更改运行中的网页服务。</p><details><summary>独立审计结果</summary><pre>{escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details></section>
<p><a href="runs_summary.csv">各次运行 CSV</a> · <a href="aggregate.json">汇总 JSON</a> · <a href="protocol.json">协议</a> · <a href="preflight.json">预检</a> · <a href="audit.json">独立审计</a></p></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")
    print(root / "report.html")


if __name__ == "__main__":
    main()
