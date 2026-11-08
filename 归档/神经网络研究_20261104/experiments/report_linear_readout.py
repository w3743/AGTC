"""Generate an offline Chinese report for the linear-output-layer ablation.

Usage: python report_linear_readout.py RUN_DIRECTORY
Reads the new run plus explicitly referenced reused runs, and writes HTML,
PNG/SVG plots, per-step curves, and switching recovery tables into RUN_DIRECTORY.
"""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
import re
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

OLD_KINDS = ("custom", "rnn", "gru", "lstm", "mlp", "tcn", "transformer")
OLD_NAMES = {"custom": "PlasticRNN（旧）", "rnn": "RNN（旧）", "gru": "GRU（旧）",
             "lstm": "LSTM（旧）", "mlp": "Window MLP（旧）", "tcn": "Causal TCN（旧）",
             "transformer": "Causal Transformer（旧）"}
TASK_NAMES = {"cycle": "循环 0123456789", "triples": "三连数字块", "zigzag": "往返锯齿",
              "fibonacci_mod10": "斐波那契模 10", "switching": "三阶段规则切换",
              "iid_random": "固定独立随机数字"}
COLORS = {"custom": "#344e9a", "rnn": "#8297bd", "gru": "#4c9f70", "lstm": "#73b9a0",
          "mlp": "#d9a441", "tcn": "#dd7954", "transformer": "#956bb4",
          "custom_linear20": "#bd3e42", "custom_linear19": "#ed8b32", "custom_linear18": "#7a4eab"}
LINEAR_PALETTE = ("#bd3e42", "#ed8b32", "#7a4eab", "#167d9a", "#a84c8b", "#638b2e")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def kind_name(kind: str) -> str:
    if kind in OLD_NAMES:
        return OLD_NAMES[kind]
    match = re.fullmatch(r"custom_linear(\d+)", kind)
    if match:
        return f"线性输出 {match.group(1)}"
    return kind.replace("_", " ")


def kind_color(kind: str) -> str:
    if kind in COLORS:
        return COLORS[kind]
    if kind.startswith("custom_linear"):
        width = int(kind.removeprefix("custom_linear"))
        return LINEAR_PALETTE[width % len(LINEAR_PALETTE)]
    return "#888888"


def validate(root: Path):
    protocol, completed, audit = (load(root / n) for n in ("protocol.json", "completed.json", "audit.json"))
    if completed.get("success") is not True or audit.get("passed") is not True:
        raise ValueError("报告只接受 completed.success=true 且 audit.passed=true 的运行")
    kinds = protocol.get("kinds") or protocol.get("variants")
    if isinstance(kinds, dict):
        kinds = list(kinds)
    tasks = list(protocol.get("tasks", {})); seeds = protocol.get("seeds", [])
    if not kinds or not tasks or not seeds:
        raise ValueError("protocol.json 必须包含 kinds/variants、tasks 与 seeds")
    aggregate = load(root / "aggregate.json"); summaries = load(root / "runs_summary.json")
    if not isinstance(aggregate, list) or not isinstance(summaries, list):
        raise ValueError("aggregate.json 与 runs_summary.json 应为数组")
    for kind in kinds:
        for task in tasks:
            rows = [r for r in aggregate if r.get("kind") == kind and r.get("task") == task]
            sr = [r for r in summaries if r.get("kind") == kind and r.get("task") == task]
            if len(rows) != 1 or len(sr) != len(seeds):
                raise ValueError(f"缺少聚合或逐种子摘要：{kind}/{task}")
    return protocol, audit, aggregate, summaries, kinds, tasks, seeds


def event_rows(summary):
    source = summary.get("source_run")
    if not source:
        raise ValueError(f"摘要缺少 source_run：{summary.get('kind')}/{summary.get('task')}")
    path = Path(source) / summary["run_directory"] / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(summary.get("count", len(rows))):
        raise ValueError(f"事件数不符：{path}")
    if rows and [int(x["step"]) for x in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"事件 step 不连续：{path}")
    return rows


def rolling(hits, window=100):
    return [(i + 1, statistics.mean(hits[max(0, i + 1 - window):i + 1])) for i in range(len(hits))]


def recovery(hits, start, end, threshold=.90, window=25):
    """First post-boundary 25-step window >= threshold, followed by another such window.

    start/end bound the current stage. Confirmation must finish before end, so a
    later stage cannot validate an earlier stage's recovery. The threshold is
    retrospectively confirmed by a second adjacent qualifying window.
    """
    for left in range(start - 1, end - 2 * window + 1):
        first = statistics.mean(hits[left:left + window])
        second = statistics.mean(hits[left + window:left + 2 * window])
        if first >= threshold and second >= threshold:
            return {"recovery_step": left + window, "confirmation_step": left + 2 * window,
                    "latency": left + window - (start - 1),
                    "first_window_accuracy": first, "confirmation_accuracy": second}
    return {"recovery_step": None, "confirmation_step": None, "latency": None, "first_window_accuracy": None,
            "confirmation_accuracy": None}


def make_curves(root, summaries, kinds, tasks, seeds):
    indexed = {(r["kind"], r["task"], str(r["seed"])): r for r in summaries}
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharey=True)
    for ax, task in zip(axes.flat, tasks):
        for kind in kinds:
            curves = []
            for seed in seeds:
                s = indexed[(kind, task, str(seed))]
                hits = [bool(r["hit"]) for r in event_rows(s)]
                curves.append(dict(rolling(hits, 100)))
            steps = range(100, 601)
            mean_curve = [statistics.mean(c.get(i, float("nan")) for c in curves) for i in steps]
            style = "-" if kind.startswith("custom_linear") else "--"
            ax.plot(steps, mean_curve, color=kind_color(kind), lw=2.3 if kind.startswith("custom_linear") else 1.2,
                    ls=style, label=kind_name(kind))
            for step, val in zip(steps, mean_curve):
                rows.append({"task": task, "kind": kind, "step": step,
                             "rolling_100_accuracy_mean": val})
        if task == "switching":
            ax.axvline(200, color="#444", ls=":", lw=1); ax.axvline(400, color="#444", ls=":", lw=1)
            ax.axvline(400, color="#bd3e42", ls="-.", lw=1.4)
        if task == "iid_random":
            ax.axhline(.1, color="#555", ls=":", lw=1, label="随机猜测期望 10%")
        ax.set_title(TASK_NAMES.get(task, task)); ax.set_xlabel("预测步")
        ax.set_xlim(100, 600); ax.set_ylim(0, 1); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("最近 100 次准确率（三种子均值）")
    axes[1, 0].set_ylabel("最近 100 次准确率（三种子均值）")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("六任务学习曲线：粗实线为新线性输出层模型，虚线为复用旧结果")
    fig.savefig(root / "learning_curves.png", dpi=180); fig.savefig(root / "learning_curves.svg"); plt.close(fig)
    with (root / "learning_curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def make_switching(root, summaries, kinds, seeds):
    selected = [k for k in kinds if k.startswith("custom_linear") or k in ("custom", "gru", "lstm", "mlp")]
    rows = []
    for kind in selected:
        for seed in seeds:
            s = next(x for x in summaries if x["kind"] == kind and x["task"] == "switching" and str(x["seed"]) == str(seed))
            hits = [bool(r["hit"]) for r in event_rows(s)]
            for boundary, first, end in ((200, 201, 400), (400, 401, 600)):
                rec = recovery(hits, first, end)
                rows.append({"kind": kind, "seed": seed, "boundary": boundary,
                             "first30_accuracy": statistics.mean(hits[first - 1:first + 29]),
                             "first50_accuracy": statistics.mean(hits[first - 1:first + 49]),
                             "first25_accuracy": statistics.mean(hits[first - 1:first + 24]),
                             "recovery_step": rec["recovery_step"], "recovery_latency": rec["latency"],
                             "confirmation_step": rec["confirmation_step"],
                             "recovery_window_accuracy": rec["first_window_accuracy"],
                             "confirmation_window_accuracy": rec["confirmation_accuracy"]})
    with (root / "switching_recovery.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (root / "switching_recovery.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    curves = []
    by_kind = {}
    for kind in selected:
        seed_hits = []
        for seed in seeds:
            s = next(x for x in summaries if x["kind"] == kind and x["task"] == "switching" and str(x["seed"]) == str(seed))
            seed_hits.append([bool(r["hit"]) for r in event_rows(s)])
        by_kind[kind] = seed_hits
        for start in range(401, 601, 25):
            end = min(600, start + 24)
            vals = [statistics.mean(h[start - 1:end]) for h in seed_hits]
            curves.append({"kind": kind, "start_step": start, "end_step": end,
                           "accuracy_mean_3_seeds": statistics.mean(vals),
                           "seed_accuracies": ";".join(f"{v:.6f}" for v in vals)})
    with (root / "switching_25step.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for kind in selected:
        points = [r for r in curves if r["kind"] == kind]
        ax.plot([r["start_step"] for r in points], [r["accuracy_mean_3_seeds"] for r in points],
                marker="o", color=kind_color(kind), lw=2.2 if kind.startswith("custom_linear") else 1.5,
                ls="-" if kind.startswith("custom_linear") else "--", label=kind_name(kind))
    ax.axvline(401, color="#333", ls=":"); ax.set_xticks(range(401, 601, 25)); ax.set_ylim(0, 1)
    ax.set_xlabel("第 401–600 步（每点为连续 25 次预测）"); ax.set_ylabel("准确率（三种子均值）")
    ax.set_title("第二次规则切换后的分段表现"); ax.grid(alpha=.22); ax.legend(ncol=2, fontsize=9)
    fig.savefig(root / "switching_recovery.png", dpi=180); fig.savefig(root / "switching_recovery.svg"); plt.close(fig)
    return selected, rows, curves


def table(headers, rows):
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" + "".join(rows) + "</table></div>"


def make_html(root, protocol, audit, aggregate, summaries, kinds, all_kinds, tasks, seeds, recovery_kinds, recoveries, switching_curves):
    agg = {(r["kind"], r["task"]): r for r in aggregate}
    acc_rows, last_rows, nll_rows = [], [], []
    for task in tasks:
        acc_rows.append("<tr><th>" + escape(TASK_NAMES.get(task, task)) + "</th>" + "".join(
            f"<td>{agg[(k,task)]['accuracy']:.2%}<br>{agg[(k,task)]['hits']:,}/{agg[(k,task)]['count']:,}</td>" for k in kinds) + "</tr>")
        last_rows.append("<tr><th>" + escape(TASK_NAMES.get(task, task)) + "</th>" + "".join(
            f"<td>{agg[(k,task)]['last100_accuracy']:.2%}</td>" for k in kinds) + "</tr>")
        nll_rows.append("<tr><th>" + escape(TASK_NAMES.get(task, task)) + "</th>" + "".join(
            f"<td>{agg[(k,task)]['mean_nll']:.4f}</td>" for k in kinds) + "</tr>")
    findings = []
    if "custom" in kinds and "custom_linear20" in kinds:
        def pair(task, metric):
            a, b = agg[("custom_linear20", task)][metric], agg[("custom", task)][metric]
            return a, b, a - b
        if "fibonacci_mod10" in tasks:
            a, b, d = pair("fibonacci_mod10", "accuracy")
            al, bl, dl = pair("fibonacci_mod10", "last100_accuracy")
            findings.append(f"<li><b>斐波那契模 10：</b>线性输出 20 累计准确率 {a:.2%}，旧 custom {b:.2%}（差 {d:+.2%}）；末 100 位 {al:.2%} 对 {bl:.2%}（差 {dl:+.2%}）。本任务只有固定序列，不能据此推断对新初值泛化。</li>")
        if "switching" in tasks:
            a, b, d = pair("switching", "accuracy")
            al, bl, dl = pair("switching", "last100_accuracy")
            phase3 = [r for r in recoveries if r["kind"] in ("custom", "custom_linear20") and r["boundary"] == 400]
            rec_by_kind = {k: [r for r in phase3 if r["kind"] == k] for k in ("custom", "custom_linear20")}
            first30 = {k: statistics.mean(r["first30_accuracy"] for r in rec_by_kind[k]) for k in rec_by_kind if rec_by_kind[k]}
            recovery_text = ""
            if "custom_linear20" in first30 and "custom" in first30:
                recovery_text = f"；切换后首 30 步分别为 {first30['custom_linear20']:.2%}、{first30['custom']:.2%}"
            findings.append(f"<li><b>三阶段切换：</b>线性输出 20 累计准确率 {a:.2%}，旧 custom {b:.2%}（差 {d:+.2%}）；末 100 位 {al:.2%} 对 {bl:.2%}（差 {dl:+.2%}）{recovery_text}。恢复结论应结合阶段内两窗确认指标。</li>")
        if "zigzag" in tasks:
            a, b, d = pair("zigzag", "accuracy")
            findings.append(f"<li><b>往返锯齿：</b>线性输出 20 累计准确率 {a:.2%}，旧 custom {b:.2%}（差 {d:+.2%}），该任务表现下降。</li>")
        if "iid_random" in tasks:
            a, b, d = pair("iid_random", "mean_nll")
            findings.append(f"<li><b>固定随机任务：</b>线性输出 20 平均 NLL {a:.4f}，旧 custom {b:.4f}（差 {d:+.4f}）。随机任务的概率质量变差；这反映错误预测上概率分布更差、存在对错误预测过度自信的风险，不能宣称随机序列可预测。</li>")
    khead = "<tr><th>任务</th>" + "".join(f"<th>{escape(kind_name(k))}</th>" for k in kinds) + "</tr>"
    recovery_rows = []
    for r in recoveries:
        recovery_windows = ("NA" if r["recovery_window_accuracy"] is None else
                           f"{r['recovery_window_accuracy']:.1%} / {r['confirmation_window_accuracy']:.1%}")
        recovery_rows.append(f"<tr><td>{escape(kind_name(r['kind']))}</td><td>{r['seed']}</td><td>{r['boundary']}（第{r['boundary']+1}步生效）</td>"
          f"<td>{r['first30_accuracy']:.1%}</td><td>{r['first50_accuracy']:.1%}</td><td>{r['first25_accuracy']:.1%}</td>"
          f"<td>{'NA' if r['recovery_latency'] is None else str(r['recovery_latency'])+' 步'}</td>"
          f"<td>{recovery_windows}</td></tr>")
    seed_diff_rows = []
    seed_index = {(r['kind'], r['task'], str(r['seed'])): r for r in summaries}
    custom_kind = next((k for k in kinds if k == "custom"), None)
    linear_kinds = [k for k in kinds if k.startswith("custom_linear")]
    if custom_kind:
        for task in tasks:
            for seed in seeds:
                base = seed_index[(custom_kind, task, str(seed))]
                for lk in linear_kinds:
                    cand = seed_index[(lk, task, str(seed))]
                    seed_diff_rows.append(f"<tr><td>{escape(TASK_NAMES.get(task,task))}</td><td>{seed}</td>"
                        f"<td>{escape(kind_name(lk))}</td><td>{cand['accuracy']:.2%} − {base['accuracy']:.2%} = {(cand['accuracy']-base['accuracy']):+.2%}</td>"
                        f"<td>{cand['mean_nll']:.4f} − {base['mean_nll']:.4f} = {(cand['mean_nll']-base['mean_nll']):+.4f}</td></tr>")
    budget_rows = []
    budget_appendix_rows = []
    budget_file = root / "full_run_budgets.json"
    cpu_file = root / "cpu_latency.json"
    cpu = load(cpu_file) if cpu_file.exists() else {}
    matched_file = root / "matched_budgets.json"
    matched = load(matched_file) if matched_file.exists() else {}
    if budget_file.exists():
        budgets = load(budget_file)
        for k in all_kinds:
            item = budgets.get(k, {})
            if item.get("origin") == "reused" and k in matched:
                config = matched[k].get("config", {})
            else:
                config = item.get("config", matched.get(k, {}).get("config", {}))
            proxy = item.get('total_proxy')
            proxy_text = f"{proxy:,.0f}" if isinstance(proxy, (int, float)) else "—"
            ratio = item.get('ratio_to_original_custom', item.get('ratio_to_custom', '—'))
            ratio_text = f"{ratio:.1%}" if isinstance(ratio, (int, float)) else str(ratio)
            latency = cpu.get(k, {})
            latency_text = (f"{latency['serial_single_thread_mean_ms']:.3f} / {latency['serial_single_thread_median_ms']:.3f}"
                            if "serial_single_thread_mean_ms" in latency else "—")
            row = f"<tr><td>{escape(kind_name(k))}</td><td>{config.get('parameters','—')}</td><td>{config.get('width','—')}</td><td>{proxy_text}</td>"
            row += f"<td>{ratio_text}</td><td>{latency_text}</td><td>{escape(str(item.get('origin', latency.get('origin','—'))))}</td></tr>"
            (budget_rows if k != "custom_linear19" else budget_appendix_rows).append(row)
    comparison = []
    oldroot = protocol.get("reference_run", "旧报告已审计运行")
    for k in kinds:
        sources = {r.get("origin", "unknown") for r in summaries if r["kind"] == k}
        comparison.append(f"<li><b>{escape(kind_name(k))}</b>：{'；'.join(sorted(sources))} 数据；摘要覆盖 {sum(1 for r in summaries if r['kind']==k)} 个任务种子。</li>")
    mechanism = root / "mechanism_validation.json"
    mechanism_text = escape(json.dumps(load(mechanism), ensure_ascii=False, indent=2)) if mechanism.exists() else "（未提供机制验证文件）"
    phase_rows = []
    for kind in recovery_kinds:
        selected_recs = [r for r in recoveries if r['kind'] == kind and r['boundary'] == 400]
        f30 = statistics.mean(r['first30_accuracy'] for r in selected_recs)
        f50 = statistics.mean(r['first50_accuracy'] for r in selected_recs)
        phase_rows.append(f"<tr><td>{escape(kind_name(kind))}</td><td>{f30:.1%}</td><td>{f50:.1%}</td></tr>")
    segment_rows = []
    for curve in switching_curves:
        segment_rows.append(f"<tr><td>{escape(kind_name(curve['kind']))}</td><td>{curve['start_step']}–{curve['end_step']}</td><td>{curve['accuracy_mean_3_seeds']:.1%}</td><td>{escape(curve['seed_accuracies'])}</td></tr>")
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>线性输出层消融报告</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#202c3c;background:#f4f6f9;max-width:1600px;margin:28px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}.scroll{{overflow-x:auto}}.note{{color:#596779;font-size:14px}}.new{{background:#fff5dc}}</style>
<h1>独立线性输出层消融：六任务数字预测</h1><p>审计通过 · {len(kinds)} 种模型 · {len(tasks)} 项任务 · {len(seeds)} 个配对种子 · 聚合与逐 seed 指标</p>
<section class="new"><h2>数据来源与比较范围</h2><p>旧标准模型结果复用自 <code>{escape(str(oldroot))}</code>，并由协议与复用清单核对；本实验固定主体、初始化和训练流程，将输出层参数化方式设为线性作为单变量结构对照，<b>custom_linear20</b> 是主要因果对照。若包含较小宽度线性变体（如 custom_linear18/19），其同时改变主体规模或核心配置，只作为额外预算对照，不解释为单因素效应。来源清单：</p><ul>{''.join(comparison)}</ul><p>切换对照重点展示旧 custom、线性输出变体及 GRU/LSTM/MLP。</p></section>
<section class="new"><h2>主要结果解读</h2><p>以下差值均由当前聚合表动态计算，属于描述性观察。</p><ul>{''.join(findings)}</ul><p>结果呈任务依赖性：线性输出 20 在部分任务有提升，也有任务下降。准确率与 NLL 衡量不同方面，不将这些结果概括为全面改进。</p></section>
<section><h2>累计准确率与每次运行末 100 位</h2><p>累计准确率由聚合预测数加权；末 100 位采用三个种子合并的末段命中率。</p><h3>全部 600 次累计准确率</h3>{table(['任务']+[escape(kind_name(k)) for k in kinds],acc_rows)}<h3>每次运行最后 100 次准确率</h3>{table(['任务']+[escape(kind_name(k)) for k in kinds],last_rows)}<img src="model_accuracy.png" alt="模型准确率对比"></section>
<section><h2>滚动学习曲线</h2><p>最近 100 次命中率按 seed 分别计算，再逐步平均；新模型用较粗实线，复用结果用虚线。图附 PNG 与 SVG，逐步数值见 CSV。</p><img src="learning_curves.png" alt="学习曲线"><p><a href="learning_curves.svg">SVG</a> · <a href="learning_curves.csv">CSV</a></p></section>
<section class="new"><h2>切换后适应与恢复速度</h2><p>对第 201、401 步切换后的逐 seed 命中序列计算。阶段 2 的候选恢复窗口及其确认窗必须均结束于第 400 步以前；阶段 3 同理截止第 600 步。首 30/50 位为跨 seed 均值。恢复定义：切换后首次出现连续两个、各含 25 次预测且准确率均不低于 90% 的相邻窗口；需等第二窗结束后回溯确认，故分别记录首窗结束步和确认窗结束步，恢复延迟按首窗结束计算。未在本阶段内满足时记为 NA。该指标为描述性门槛，不是显著性检验。</p>{table(['模型','seed','边界','首30准确率','首50准确率','首25准确率','首窗结束/恢复延迟','恢复/确认窗'],recovery_rows)}<h3>401–600 步每 25 位表现（三 seed 均值）</h3><img src="switching_recovery.png" alt="第二次切换后恢复重点图">{table(['模型','步段','三 seed 均值','seed 准确率'],segment_rows)}<p><a href="switching_recovery.svg">重点图 SVG</a> · <a href="switching_25step.csv">逐段 CSV</a> · <a href="switching_recovery.csv">逐 seed 恢复 CSV</a> · <a href="switching_recovery.json">JSON</a></p><h3>第二切换初期聚合表现</h3>{table(['模型','前30（三 seed 均值）','前50（三 seed 均值）'],phase_rows)}</section>
<section><h2>NLL</h2><p>交叉熵按预测数汇总的平均负对数似然；越低表示预测分布越贴近真实目标。</p>{table(['任务']+[escape(kind_name(k)) for k in kinds],nll_rows)}</section>
<section><h2>逐 seed 配对差异</h2><p>列出各新线性输出层变体相对同 seed 旧 custom 的准确率差（百分点）与 NLL 差；负 NLL 差代表 NLL 降低。该配对仅用于描述性对照。</p>{table(['任务','seed','新模型','准确率差','NLL 差'],seed_diff_rows)}</section>
<section><h2>模型参数、计算预算与 CPU 延迟</h2><p>完整 CPU 延迟数据提供实测均值与中位数。预算 proxy 是实现定义下的算术代理值，不等于硬件 FLOPs。主对照 custom_linear20 的完整预算为旧 custom 的 110.83%（高 10.83%），因此它是输出层参数化的结构对照，不是同预算对照；未重跑旧基线。</p>{table(['模型','参数量','宽度','完整预算 proxy','相对旧 custom 比例','CPU 均值/中位数 ms','来源'],budget_rows)}<details><summary>预算辅助变体 custom_linear19（不纳入主比较）</summary><p>19 仅提供预算辅助信息，不进入准确率主表、学习曲线或恢复结论。</p>{table(['模型','参数量','宽度','完整预算 proxy','相对旧 custom 比例','CPU 均值/中位数 ms','来源'],budget_appendix_rows)}</details><p>完整预算数据：<a href="full_run_budgets.json">full_run_budgets.json</a> · <a href="matched_budgets.json">matched_budgets.json</a> · CPU 记录：<a href="cpu_latency.json">cpu_latency.json</a></p><p class="note">CPU 测量在不同运行时段完成，机器当前负载不同；均值与中位数仅作描述，不能将其变化归因为模型结构。</p><h3>解释限制</h3><ul><li>每次预测最多重放 32 位历史，结果不能证明超出该上下文长度的记忆能力。</li><li>斐波那契模 10 只测一条固定初值序列，不构成对新初值泛化或加法规则学习的证据。</li><li>学习率固定且未做超参数搜索，结果不代表调参后的上限。</li><li>模型种子共享同一条 600 位随机序列，跨模型数据并非独立抽样。</li></ul><details><summary>机制验证</summary><pre>{mechanism_text}</pre></details><details><summary>审计记录</summary><pre>{escape(json.dumps(audit,ensure_ascii=False,indent=2))}</pre></details></section>
<p>协议：<a href="protocol.json">protocol.json</a> · 复用清单：<a href="reuse_manifest.json">reuse_manifest.json</a> · 汇总：<a href="aggregate.json">aggregate.json</a> · 逐运行摘要：<a href="runs_summary.json">runs_summary.json</a></p></html>'''
    (root / "report.html").write_text(html, encoding="utf-8")


def make_accuracy_plot(root, aggregate, kinds, tasks):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharey=True)
    indexed = {(r['kind'], r['task']): r for r in aggregate}
    for ax, task in zip(axes.flat, tasks):
        vals = [indexed[(k, task)]['accuracy'] for k in kinds]
        bars = ax.bar(range(len(kinds)), vals, color=[kind_color(k) for k in kinds])
        ax.set_xticks(range(len(kinds)), [kind_name(k) for k in kinds], rotation=42, ha='right', fontsize=8)
        ax.set_title(TASK_NAMES.get(task, task)); ax.set_ylim(0, 1.12); ax.grid(axis='y', alpha=.2)
        for b,v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+.01, f'{v:.1%}', ha='center', fontsize=7)
    axes[0,0].set_ylabel('累计准确率'); axes[1,0].set_ylabel('累计准确率')
    fig.suptitle('六任务模型累计准确率')
    fig.savefig(root/'model_accuracy.png', dpi=180); fig.savefig(root/'model_accuracy.svg'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    root = p.parse_args().root
    protocol, audit, aggregate, summaries, kinds, tasks, seeds = validate(root)
    main_kinds = [k for k in kinds if k != "custom_linear19"]
    make_accuracy_plot(root, aggregate, main_kinds, tasks)
    make_curves(root, summaries, main_kinds, tasks, seeds)
    recovery_kinds, recoveries, switching_curves = make_switching(root, summaries, main_kinds, seeds)
    make_html(root, protocol, audit, aggregate, summaries, main_kinds, kinds, tasks, seeds, recovery_kinds, recoveries, switching_curves)
    print(root / 'report.html')


if __name__ == '__main__':
    main()
