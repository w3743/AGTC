"""Generate a Chinese comparison report for the fresh binary-readout benchmark.

Usage: python report_binary_readout.py RUN_DIRECTORY
Only completed, independently audited runs are accepted. The script writes
report.html, PNG/SVG figures, and rolling-curve CSV beside the run data.
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

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT_PATH)
CHINESE_FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [CHINESE_FONT]
plt.rcParams["axes.unicode_minus"] = False

KINDS = ("custom", "custom_linear20", "rnn", "gru", "lstm", "mlp", "tcn", "transformer")
NAMES = {"custom": "原输出20", "custom_linear20": "新线性输出20", "rnn": "RNN", "gru": "GRU",
         "lstm": "LSTM", "mlp": "MLP", "tcn": "TCN", "transformer": "Transformer"}
COLORS = {"custom": "#344e9a", "custom_linear20": "#bd3e42", "rnn": "#8297bd", "gru": "#4c9f70",
          "lstm": "#73b9a0", "mlp": "#d9a441", "tcn": "#dd7954", "transformer": "#956bb4"}
TASK_NAMES = {"alternating": "交替 01", "triples": "三连 000111", "period16": "周期16"}
EXPECTED_PATTERNS = {"alternating": "01", "triples": "000111", "period16": "0000100110101111"}
EXPECTED_SEEDS = {2, 20261027, 20261028}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path):
    protocol, completed, audit = (load(root / name) for name in ("protocol.json", "completed.json", "audit.json"))
    if completed.get("success") is not True or audit.get("passed") is not True:
        raise ValueError("只接受 completed.success=true 且 audit.passed=true 的运行")
    kinds = protocol.get("kinds") or protocol.get("variants")
    if isinstance(kinds, dict):
        kinds = list(kinds)
    patterns = protocol.get("patterns") or protocol.get("tasks")
    if isinstance(patterns, list):
        patterns = {task: EXPECTED_PATTERNS.get(task) for task in patterns}
    seeds = protocol.get("seeds", [])
    if tuple(kinds or ()) != KINDS and set(kinds or ()) != set(KINDS):
        raise ValueError("protocol 必须覆盖八种指定模型")
    if patterns != EXPECTED_PATTERNS or set(seeds) != EXPECTED_SEEDS:
        raise ValueError("protocol 任务 pattern 或三个种子与指定方案不符")
    if protocol.get("classes") != 10 or protocol.get("input_count") != 10 or protocol.get("all_models_freshly_rerun") is not True:
        raise ValueError("协议必须确认 10 类/10 输入接口且所有模型均本轮重跑")
    aggregate, summaries = load(root / "aggregate.json"), load(root / "runs_summary.json")
    if len(aggregate) != 24 or len(summaries) != 72:
        raise ValueError("需要 24 条任务×模型聚合及 72 条逐 seed 摘要")
    for kind in KINDS:
        for task in EXPECTED_PATTERNS:
            ag = [r for r in aggregate if r.get("kind") == kind and r.get("task") == task]
            rows = [r for r in summaries if r.get("kind") == kind and r.get("task") == task]
            if len(ag) != 1 or len(rows) != 3 or {int(r["seed"]) for r in rows} != EXPECTED_SEEDS:
                raise ValueError(f"缺少任务/模型/seed 记录：{kind}/{task}")
            if any(r.get("count") != 600 for r in rows):
                raise ValueError(f"运行预测数不是 600：{kind}/{task}")
            if any(r.get("origin") != "new" or r.get("source_run") != str(root) for r in rows):
                raise ValueError(f"本轮要求全部 fresh，发现非 new 来源：{kind}/{task}")
            if any(r.get("pattern") != EXPECTED_PATTERNS[task] for r in rows):
                raise ValueError(f"逐运行摘要 pattern 与协议不符：{kind}/{task}")
        ag_rows = [r for r in aggregate if r.get("kind") == kind]
        if len(ag_rows) != 3 or any(r.get("origin") != "new" for r in ag_rows):
            raise ValueError(f"聚合来源缺失或非本轮 fresh：{kind}")
    for filename in ("matched_budgets.json", "full_run_budgets.json", "cpu_latency.json", "mechanism_validation.json", "task_validation.json", "input_sequences.json"):
        if not (root / filename).is_file():
            raise ValueError(f"缺少协议产物 {filename}")
    sequences = load(root / "input_sequences.json")
    if set(sequences) != set(EXPECTED_PATTERNS) or any(
        sequences[t] != [int(EXPECTED_PATTERNS[t][i % len(EXPECTED_PATTERNS[t])]) for i in range(600)]
        for t in EXPECTED_PATTERNS
    ):
        raise ValueError("input_sequences.json 与指定的 600 位原 pattern 不符")
    if load(root / "task_validation.json").get("passed") is not True:
        raise ValueError("task_validation.json 未通过")
    for filename in ("matched_budgets.json", "full_run_budgets.json", "cpu_latency.json"):
        if set(load(root / filename)) != set(KINDS):
            raise ValueError(f"{filename} 未覆盖八种模型")
    return protocol, audit, aggregate, summaries


def events(summary):
    source = Path(summary["source_run"])
    rows = [json.loads(line) for line in (source / summary["run_directory"] / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 600 or [int(r["step"]) for r in rows] != list(range(1, 601)):
        raise ValueError(f"事件缺失或 step 不连续：{summary['run_directory']}")
    return rows


def plots(root, aggregate, summaries):
    tasks = list(EXPECTED_PATTERNS)
    index = {(r["kind"], r["task"]): r for r in aggregate}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True, sharey=True)
    for ax, task in zip(axes, tasks):
        vals = [index[(k, task)]["accuracy"] for k in KINDS]
        bars = ax.bar(range(8), vals, color=[COLORS[k] for k in KINDS], width=.76)
        ax.set_xticks(range(8), [NAMES[k] for k in KINDS], rotation=38, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12); ax.set_title(TASK_NAMES[task]); ax.grid(axis="y", alpha=.22)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, value+.015, f"{value:.1%}", ha="center", fontsize=8)
    axes[0].set_ylabel("累计准确率（每模型每任务 1,800 次预测）")
    fig.suptitle("二进制序列任务：累计准确率")
    fig.savefig(root/"model_accuracy.png", dpi=180); fig.savefig(root/"model_accuracy.svg"); plt.close(fig)

    by = {(r["kind"], r["task"], str(r["seed"])): r for r in summaries}
    output = []
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True, sharey=True)
    for ax, task in zip(axes, tasks):
        for kind in KINDS:
            curves = []
            for seed in sorted(EXPECTED_SEEDS):
                rows = events(by[(kind, task, str(seed))])
                hits = [bool(r["hit"]) for r in rows]
                curves.append([statistics.mean(hits[i-100:i]) for i in range(100, 601)])
            mean = [statistics.mean(c[i] for c in curves) for i in range(501)]
            ax.plot(range(100, 601), mean, color=COLORS[kind], lw=2.6 if kind in ("custom", "custom_linear20") else 1.35,
                    ls="-" if kind in ("custom", "custom_linear20") else "--", label=NAMES[kind])
            output.extend({"task": task, "kind": kind, "step": step, "rolling_100_accuracy_mean_3_seeds": v}
                          for step, v in zip(range(100, 601), mean))
        if task == "period16":
            ax.axhline(.5, color="#333", ls=":", lw=1.4, label="独立均匀猜 0/1 参考期望 50%")
        ax.set_title(TASK_NAMES[task]); ax.set_xlabel("预测步"); ax.set_xlim(100, 600); ax.set_ylim(0, 1); ax.grid(alpha=.22)
    axes[0].set_ylabel("最近 100 次预测准确率（三 seed 均值）")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("最近 100 次准确率曲线；粗实线突出旧 custom 与新线性输出20")
    fig.savefig(root/"learning_curves.png", dpi=180); fig.savefig(root/"learning_curves.svg"); plt.close(fig)
    with (root/"learning_curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)


def html_report(root, protocol, audit, aggregate, summaries):
    agg = {(r["kind"], r["task"]): r for r in aggregate}
    cell = lambda k, t, metric: f"{agg[(k,t)][metric]:.2%}"
    head = "<tr><th>任务</th>" + "".join(f"<th>{escape(NAMES[k])}</th>" for k in KINDS) + "</tr>"
    table_rows = "".join(f"<tr><th>{TASK_NAMES[t]}<br><code>{t}：{EXPECTED_PATTERNS[t]}</code></th>" + "".join(
        f"<td>{cell(k,t,'accuracy')}<br>{agg[(k,t)]['hits']:,}/1,800</td>" for k in KINDS) + "</tr>" for t in EXPECTED_PATTERNS)
    last_rows = "".join(f"<tr><th>{TASK_NAMES[t]}</th>" + "".join(f"<td>{cell(k,t,'last100_accuracy')}</td>" for k in KINDS)+"</tr>" for t in EXPECTED_PATTERNS)
    nll_rows = "".join(f"<tr><th>{TASK_NAMES[t]}</th>" + "".join(f"<td>{agg[(k,t)]['mean_nll']:.6f}</td>" for k in KINDS)+"</tr>" for t in EXPECTED_PATTERNS)
    paired=[]
    si={(r['kind'],r['task'],str(r['seed'])):r for r in summaries}
    for t in EXPECTED_PATTERNS:
        for seed in sorted(EXPECTED_SEEDS):
            old,new=si[('custom',t,str(seed))],si[('custom_linear20',t,str(seed))]
            paired.append(f"<tr><td>{TASK_NAMES[t]}</td><td>{seed}</td><td>{new['accuracy']-old['accuracy']:+.2%}</td><td>{new['mean_nll']-old['mean_nll']:+.6f}</td></tr>")
    seed_detail=[]
    for t in EXPECTED_PATTERNS:
        for seed in sorted(EXPECTED_SEEDS):
            old,new=si[('custom',t,str(seed))],si[('custom_linear20',t,str(seed))]
            seed_detail.append(f"<tr><td>{TASK_NAMES[t]}</td><td>{seed}</td><td>{old['accuracy']:.2%}</td><td>{old['last100_accuracy']:.2%}</td><td>{old['mean_nll']:.6f}</td><td>{new['accuracy']:.2%}</td><td>{new['last100_accuracy']:.2%}</td><td>{new['mean_nll']:.6f}</td></tr>")
    finding_parts=[]
    for t in EXPECTED_PATTERNS:
        old,new=agg[('custom',t)],agg[('custom_linear20',t)]
        da=new['accuracy']-old['accuracy']; dl=new['last100_accuracy']-old['last100_accuracy']; dn=new['mean_nll']-old['mean_nll']
        finding_parts.append(f"{TASK_NAMES[t]}：新线性输出20准确率 {new['accuracy']:.2%}，旧 custom {old['accuracy']:.2%}，差 {da:+.2%}；末100差 {dl:+.2%}；NLL {new['mean_nll']:.4f} 对 {old['mean_nll']:.4f}（差 {dn:+.4f}）。")
    p16_best=max(agg[(k,'period16')]['accuracy'] for k in KINDS)
    p16_tied=[NAMES[k] for k in KINDS if agg[(k,'period16')]['accuracy']==p16_best]
    finding_parts.append(f"period16 的模型最高准确率为 {'、'.join(p16_tied)}（{p16_best:.2%}）。")
    findings_html="<br>".join(escape(s) for s in finding_parts)
    budget_rows=[]
    matched=load(root/'matched_budgets.json'); full=load(root/'full_run_budgets.json'); latency=load(root/'cpu_latency.json')
    widths={'custom':20,'custom_linear20':20,'rnn':31,'gru':17,'lstm':15,'mlp':7,'tcn':11,'transformer':9}
    for k in KINDS:
        b=full.get(k,{}); c=b.get('config',matched.get(k,{}).get('config',{})); lat=latency.get(k,{})
        proxy=b.get('total_proxy', matched.get(k,{}).get('total_proxy','—'))
        params=c.get('parameters','—')
        ms=(f"{lat['serial_single_thread_mean_ms']:.3f} / {lat['serial_single_thread_median_ms']:.3f}" if 'serial_single_thread_mean_ms' in lat else '—')
        budget_rows.append(f"<tr><th>{escape(NAMES[k])}</th><td>{widths[k]}</td><td>{params}</td><td>{proxy if isinstance(proxy,str) else f'{proxy:,.0f}'}</td><td>{ms}</td></tr>")
    mech=load(root/'mechanism_validation.json')
    sources=sorted({str(r.get('source_run')) for r in summaries})
    source_links=" · ".join(f"<code>{escape(s)}</code>" for s in sources)
    hashes=protocol.get("source_sha256", {})
    hash_links=" · ".join(f"<a href=\"source_snapshot/{escape(name)}\"><code>{escape(name)}</code></a> SHA-256 <code>{escape(str(value))}</code>"
                          for name,value in hashes.items()) or "协议未提供 source_sha256"
    diagnosis_html = ""
    if (root/'binary_diagnosis.json').exists():
        diagnosis = load(root/'binary_diagnosis.json')['models']['new_custom_linear20']
        pooled = diagnosis['pooled_seed_means']
        count = sum(r['window']['n'] for r in diagnosis['runs'])
        outside = sum(r['argmax_outside_binary_count'] for r in diagnosis['runs'])
        diagnosis_html = f'''<section><h2>16位任务的只读错误诊断</h2><p>检查第441–600步，三个种子共{count}次预测，其中输出2–9类的次数为{outside}。这些类别的平均总概率为{pooled['mean_probability_mass_classes_2_9']:.6%}，退化发生在0/1内部选择。预测与前31位标签多数类的符合率为{pooled['binary_prediction_matches_preceding31_majority_rate']:.2%}；对这条平衡16周期，前31位多数类恰好总与当前目标相反。这是只读相关性观察，可作为“依赖近期边际频率”的待检验解释，尚未通过额外消融建立因果关系。</p><p><a href="binary_diagnosis.json">逐种子与16相位诊断JSON</a> · <a href="analysis_source/analyze_binary_period16.py">只读分析源码</a></p></section>'''
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>二进制读出模型对照</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#202c3c;background:#f4f6f9;max-width:1600px;margin:28px auto;padding:0 22px}}section{{background:#fff;padding:22px 26px;border:1px solid #dce2eb;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:left;vertical-align:top}}th{{background:#edf2f8}}img{{width:100%}}.scroll{{overflow-x:auto}}.note{{color:#596779;font-size:14px}}.focus{{background:#fff2cc}}</style>
<h1>二进制序列预测：线性输出读出对照</h1><p>审计通过 · 全部本轮 fresh · 8 种模型 · 3 个 pattern · 3 个配对 seed · 72 次运行 · 43,200 次预测</p>
<section><h2>协议与解释边界</h2><p>三个固定 pattern：01、000111、0000100110101111；每项 600 位，模型种子为 2、20261027、20261028。输入标签仅为 0/1，但模型保留 10 输入、10 输出接口，概率与 loss 仍按 10 类计算。冷启动时十类概率均匀，argmax 按最低索引固定预测 0；该确定性预测会实际计入得分。图中 50% 线只在 period16 面板标作独立均匀猜 0/1 的参考期望，与模型的冷启动分布不同。每次预测从零开始重放最多 32 位历史，训练和预测都受此上下文限制；任务是固定周期序列，结论不代表任意二进制流泛化。</p><p>所有模型和任务本轮重新运行，来源：{source_links}。学习率固定、未做超参数搜索；三个 seed 仅提供描述性配对结果，不是总体不确定性的精确估计。</p><p>source snapshot 与 SHA-256：{hash_links}</p></section>
<section class="focus"><h2>主要发现（按实际聚合结果生成）</h2><p>{findings_html}</p><p>period16 pattern 的长度为 16；task_validation.json 记录了其循环四元组和三元组歧义检查。结果只说明这些固定 pattern 上的表现，不能单独证明对新序列泛化或形成通用二元规则。</p></section>
<section class="focus"><h2>主要结构与预算说明</h2><p>新 custom_linear20 保留 20 主体宽度；10 类线性输出使参数量为 851，相比旧 custom 的 642 多 209。完整预算比旧 custom 高 10.83%，因此这是结构对照，<b>不是同预算比较</b>。其余标准模型采用旧 matched widths：RNN 31、GRU 17、LSTM 15、MLP 7、TCN 11、Transformer 9。输入为二元而接口仍为 10 类，成本、冷启动输出及分类目标均受此接口约束。</p></section>
<section><h2>累计准确率（三任务×八模型）</h2><p>累计值按每任务每模型三个 seed 共 1,800 次预测汇总。</p><div class="scroll"><table>{head}{table_rows}</table></div><img src="model_accuracy.png" alt="三项二进制任务准确率"><p><a href="model_accuracy.svg">SVG 图</a></p><h3>各运行末 100 位准确率</h3><table>{head}{last_rows}</table><h3>平均 NLL</h3><table>{head}{nll_rows}</table></section>
<section><h2>学习曲线</h2><p>曲线为逐 seed 先计算最近 100 位命中率，再逐步求三 seed 均值。旧 custom 与新线性输出20使用较粗实线突出，其他标准对照为虚线。</p><img src="learning_curves.png" alt="滚动准确率"><p><a href="learning_curves.svg">SVG</a> · <a href="learning_curves.csv">逐步 CSV</a></p></section>
<section><h2>逐 seed 原始指标与配对差异</h2><p>原始表分别给出旧 custom 与新线性输出20的准确率、末100位准确率及平均 NLL；后表为同 seed 配对差。三个 seed 仅作描述性比较。</p><div class="scroll"><table><tr><th>任务</th><th>seed</th><th>旧准确率</th><th>旧末100</th><th>旧NLL</th><th>新准确率</th><th>新末100</th><th>新NLL</th></tr>{''.join(seed_detail)}</table></div><table><tr><th>任务</th><th>seed</th><th>准确率差</th><th>NLL 差</th></tr>{''.join(paired)}</table></section>
<section><h2>预算与 CPU 延迟</h2><table><tr><th>模型</th><th>宽度</th><th>参数量</th><th>完整运行预算 proxy</th><th>CPU ms 均值/中位数</th></tr>{''.join(budget_rows)}</table><p>预算 proxy 是统一计数规则下的算术估算，不等于硬件 FLOPs。CPU 延迟为独立单线程测量数据，以本轮 cpu_latency.json 的实测项为准。</p><p>预算文件：<a href="matched_budgets.json">matched_budgets.json</a> · <a href="full_run_budgets.json">full_run_budgets.json</a> · <a href="cpu_latency.json">cpu_latency.json</a></p><h3>机制验证</h3><details><summary>打开 mechanism_validation.json</summary><pre>{escape(json.dumps(mech,ensure_ascii=False,indent=2))}</pre></details><h3>限制</h3><ul><li>仅有三个周期任务，不能推断对未见 pattern 或非周期输入的泛化。</li><li>网络保留十类概率与 loss；冷启动为十类均匀分布，但 argmax 固定为 0 并实际计分。50% 线仅表示独立均匀猜 0/1 的参考期望，与冷启动分布不同。</li><li>每次预测从零开始重放最多 32 位历史，因此结果不能表明模型具备更长上下文记忆。</li><li>固定训练配置且未调参，标准模型尺寸沿用既定 matched widths。</li></ul><details><summary>审计记录</summary><pre>{escape(json.dumps(audit,ensure_ascii=False,indent=2))}</pre></details></section>
{diagnosis_html}<p><a href="protocol.json">协议</a> · <a href="aggregate.json">聚合</a> · <a href="runs_summary.json">逐运行摘要</a> · <a href="mechanism_validation.json">机制验证</a> · <a href="audit.json">审计</a> · <a href="completed.json">完成标记</a></p></html>'''
    (root/'report.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    root=parser.parse_args().root
    protocol,audit,aggregate,summaries=validate(root)
    plots(root,aggregate,summaries)
    html_report(root,protocol,audit,aggregate,summaries)
    print(root/'report.html')


if __name__ == '__main__':
    main()
