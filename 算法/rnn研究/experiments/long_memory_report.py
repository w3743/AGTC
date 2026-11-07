"""Audit benchmark outputs and build a standalone HTML report with figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


TASK_NAMES = {'recall': '延迟复述', 'binding': '按键查值', 'arithmetic': '模 4 加减'}
MODEL_NAMES = {
    'Full-memory RTRL': '全输入源 RTRL', 'RNN': '普通循环网络（RNN）',
    'GRU': '门控循环网络（GRU）', 'LSTM': '长短期记忆网络（LSTM）',
    'MLP-32': '窗口感知机（MLP-32）', 'TCN-512': '时序卷积网络（TCN-512）',
    'Transformer': '注意力网络（Transformer）',
    'Sparse-1024 BPTT': '1024 神经元稀疏版（BPTT）',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def winner(probabilities):
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def target(task, digits):
    if task == 'recall':
        return digits[1]
    if task == 'binding':
        return digits[digits[:6].index(digits[-1]) + 1]
    a, b = digits[1], digits[3]
    return (a + b if digits[-1] == 0 else a - b) % 4


def collect(root, protocol):
    per_seed, examples, shared, files = [], {}, {}, {}
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() == digest, name
    for model in protocol['models']:
        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                path = root / f'{model}_{task}_{seed}.json'
                run = read(path)
                assert (run['model'], run['task'], run['seed']) == (model, task, seed)
                assert len(run['train_loss']) == (len(protocol['gaps'])-1)*protocol['train_per_gap']
                if model == 'Sparse-1024 BPTT':
                    assert run['optimizer_updates'] == protocol['sparse_updates']
                    assert run['batch_size'] == protocol['sparse_batch'] and run['edges'] == protocol['sparse_edges']
                    assert run.get('learning_rate', .015) == protocol.get('sparse_learning_rate', .015)
                groups = defaultdict(list)
                for row in run['test']:
                    groups[row['gap']].append(row)
                assert set(groups) == set(protocol['gaps'])
                for gap, records in groups.items():
                    count = protocol['test_per_gap']
                    assert [r['index'] for r in records] == list(range(count))
                    hits, paired_hits, nll, cue_effect = [], [], [], []
                    for r in records:
                        x, xp, y, yp = (r[k] for k in ('digits', 'paired_digits', 'target', 'paired_target'))
                        assert sum(a != b for a, b in zip(x, xp, strict=True)) == 1
                        assert y == target(task, x) and yp == target(task, xp) and yp == (y+1)%4
                        key = (task, gap, r['index'])
                        assert shared.setdefault(key, (x, xp, y, yp)) == (x, xp, y, yp)
                        p, pp = r['probabilities'], r['paired_probabilities']
                        for probs in (p, pp):
                            assert len(probs) == 10 and all(0 <= v <= 1 for v in probs) and abs(sum(probs)-1) < 1e-6
                        hit = winner(p) == y
                        hits.append(hit)
                        paired_hits.append(hit and winner(pp) == yp)
                        nll.append(-math.log(max(p[y], 1e-30)))
                        cue_effect.append((p[y]-pp[y]+pp[yp]-p[yp])/2)
                    assert all(sum(r['target'] == label for r in records) == count//4 for label in range(4))
                    per_seed.append(dict(model=model, task=task, seed=seed, gap=gap, n=count,
                                         accuracy=mean(hits), paired_accuracy=mean(paired_hits), nll=mean(nll),
                                         cue_effect=mean(cue_effect), params=run['params'], train_seconds=run['train_seconds']))
                    if seed == protocol['seeds'][0]:
                        examples.setdefault(task, {}).setdefault(gap, {})[model] = records[0]
                for file in (path, path.with_suffix('.pt')):
                    files[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    return per_seed, examples, files


def update_comparison(reference, protocol, index, hashes):
    old = read(reference/'protocol.json')
    for key in ('models', 'tasks', 'gaps', 'seeds', 'train_per_gap', 'test_per_gap', 'sparse_edges', 'sparse_width'):
        if old[key] != protocol[key]:
            raise ValueError(f'Comparison protocol differs for {key}')
    for name in ('sparse_input_memory.py', 'long_memory_eval.py'):
        if old['source_sha256'][name] != protocol['source_sha256'][name]:
            raise ValueError(f'Comparison source differs: {name}')
    previous, _, previous_hashes = collect(reference, old)
    sparse = 'Sparse-1024 BPTT'
    if any(hashes[name] != digest for name, digest in previous_hashes.items() if not name.startswith(sparse)):
        raise ValueError('The two runs use different reference predictions or weights')
    groups = defaultdict(list)
    for r in previous:
        if r['model'] == sparse:
            groups[r['task'], r['gap']].append(r)
    gaps = protocol['gaps']
    headers = ''.join(f'<th scope="col" class="number">{gap} 步</th>' for gap in gaps)
    tables = []
    for metric, label in [('accuracy', '单题准确率'), ('paired_accuracy', '成对正确率')]:
        rows = []
        for task in protocol['tasks']:
            cells = []
            for gap in gaps:
                before = mean(r[metric] for r in groups[task, gap])
                after = index[sparse, task, gap][metric]
                delta = (after-before)*100
                css = 'correct' if delta > 0 else 'incorrect' if delta < 0 else ''
                cells.append(f'<td class="number"><strong class="{css}">{delta:+.1f}</strong>'
                             f'<small>{before:.1%} → {after:.1%}</small></td>')
            rows.append(f'<tr><th scope="row">{TASK_NAMES[task]}</th>{"".join(cells)}</tr>')
        tables.append(f'<h3>{label}</h3><div class="table-wrap"><table><thead><tr><th scope="col">任务</th>'
                      f'{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    lr_before, lr_after = (p.get('sparse_learning_rate', .015) for p in (old, protocol))
    if lr_before != lr_after:
        title = '稀疏原型学习率对比'
        description = (f'相同训练题、顺序和种子，每批 {protocol["sparse_batch"]} 题、{protocol["sparse_updates"]} 次更新。'
                       f'学习率 {lr_before:g} → {lr_after:g}。')
    else:
        title = '稀疏原型更新次数对比'
        description = (f'相同训练题、顺序和种子，每批 {old["sparse_batch"]} 题、{old["sparse_updates"]} 次更新 → '
                       f'每批 {protocol["sparse_batch"]} 题、{protocol["sparse_updates"]} 次更新。')
    return (f'<h2>{title}</h2><p>{description}大字为百分点变化，小字为改动前后的实测值。</p>'
            + ''.join(tables))


def diagnostic_panel(root, tasks, gaps, chinese_font):
    diagnostics = read(root/'diagnostics.json')
    records = diagnostics['records']
    rates = list(dict.fromkeys(r['lr'] for r in records))
    colors = ['#566279', '#007766']
    fig, axes = plt.subplots(2, len(tasks), figsize=(13, 6.8), sharey='row', layout='constrained')
    table_rows = []
    for col, task in enumerate(tasks):
        summaries = []
        for lr, color in zip(rates, colors, strict=True):
            series = [[r for r in records if (r['lr'], r['task'], r['gap']) == (lr, task, gap)] for gap in gaps]
            losses = list(zip(*(r['loss_blocks'] for r in series[0]), strict=True))
            x = [diagnostics['loss_block']*(i+1) for i in range(len(losses))]
            for ax, axis, ys in ((axes[0, col], x, losses),
                                 (axes[1, col], gaps, [[100*r['saturation'] for r in group] for group in series])):
                ax.plot(axis, list(map(mean, ys)), 'o-', color=color, label=f'{lr:g}')
                ax.fill_between(axis, list(map(min, ys)), list(map(max, ys)), color=color, alpha=.12)
            summaries.append((mean(losses[-1]), mean(r['saturation'] for r in series[0]),
                              mean(r['saturation'] for r in series[-1])))
        cells = ''.join(f'<td class="metric">{summaries[0][i]:{fmt}} → {summaries[1][i]:{fmt}}</td>'
                        for i, fmt in enumerate(('.2f', '.1%', '.1%')))
        table_rows.append(f'<tr><th scope="row">{TASK_NAMES[task]}</th>{cells}</tr>')
        axes[0, col].set_title(TASK_NAMES[task] if chinese_font else task)
        axes[0, col].set_xlabel('训练题数（每 80 题平均）' if chinese_font else 'Training examples (80-example blocks)')
        ax = axes[1, col]
        ax.set(xscale='log', xticks=gaps, xticklabels=[str(g) for g in gaps], ylim=(-3, 103),
               xlabel='干扰步数' if chinese_font else 'Distractor steps')
        ax.minorticks_off()
    axes[0, 0].set_ylabel('训练交叉熵' if chinese_font else 'Training cross-entropy')
    axes[1, 0].set_ylabel('查询处饱和比例（%）' if chinese_font else 'Saturation at query (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='学习率' if chinese_font else 'Learning rate', ncol=2, loc='outside lower center')
    fig.savefig(root/'training_diagnostics.svg')
    fig.savefig(root/'training_diagnostics.png', dpi=160)
    plt.close(fig)
    return (f'<h2>训练损失和神经元饱和</h2><p>从 {rates[0]:g} 到 {rates[1]:g}，表中为三种子的均值。'
            f'每档取固定的前 {diagnostics["probe_pairs"]} 对测试输入，统计查询处 |h| &gt; {diagnostics["threshold"]:g} 的比例。'
            '曲线阴影为三个种子的最小值至最大值。</p>'
            '<div class="table-wrap"><table><thead><tr><th>任务</th><th class="metric">末 80 题交叉熵</th>'
            '<th class="metric">8 步饱和比例</th><th class="metric">256 步饱和比例</th></tr></thead><tbody>'
            + ''.join(table_rows) + '</tbody></table></div><img src="training_diagnostics.svg" alt="训练损失和查询处饱和比例曲线">'
            '<p><a href="diagnostics-zh.csv">逐种子诊断表</a></p>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--compare', type=Path, help='Compare with an earlier sparse training configuration')
    args = parser.parse_args()
    root = args.root
    protocol = read(root/'protocol.json')
    rows, examples, hashes = collect(root, protocol)
    models, tasks, gaps = (protocol[k] for k in ('models', 'tasks', 'gaps'))
    sparse = 'Sparse-1024 BPTT'
    if sparse in models:
        models = [sparse, *[name for name in models if name != sparse]]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['model'], r['task'], r['gap']].append(r)
    summary = []
    for (model, task, gap), group in grouped.items():
        summary.append(dict(model=model, task=task, gap=gap, n=sum(r['n'] for r in group),
                            **{key: mean(r[key] for r in group) for key in ('accuracy', 'paired_accuracy', 'nll', 'cue_effect')},
                            accuracy_min=min(r['accuracy'] for r in group), accuracy_max=max(r['accuracy'] for r in group),
                            params=group[0]['params'], train_seconds=mean(r['train_seconds'] for r in group)))
    index = {(r['model'], r['task'], r['gap']): r for r in summary}
    comparison = update_comparison(args.compare, protocol, index, hashes) if args.compare else ''
    for name, data in [('per_seed.csv', rows), ('scores.csv', summary)]:
        with (root/name).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (root/'scores-zh.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '任务', '干扰步数', '原题次数', '单题准确率', '成对正确率',
                         '负对数似然', '线索效应', '最低种子准确率', '最高种子准确率', '参数量', '平均训练秒数'])
        for r in summary:
            writer.writerow([MODEL_NAMES[r['model']], TASK_NAMES[r['task']], r['gap'], r['n'],
                             f'{r["accuracy"]:.2%}', f'{r["paired_accuracy"]:.2%}', f'{r["nll"]:.4f}',
                             f'{r["cue_effect"]:+.4f}', f'{r["accuracy_min"]:.2%}', f'{r["accuracy_max"]:.2%}',
                             r['params'], f'{r["train_seconds"]:.2f}'])
    (root/'audit.json').write_text(json.dumps(dict(passed=True, runs=len(hashes)//2, probabilities=2*sum(r['n'] for r in rows),
                                                  score_rows=len(summary), sha256=hashes), indent=2), encoding='utf-8')

    plt.style.use('seaborn-v0_8-whitegrid')
    chinese_font = next((f for f in font_manager.fontManager.ttflist
                         if f.name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC')), None)
    if chinese_font:
        plt.rcParams['font.family'] = chinese_font.name
    diagnostic = diagnostic_panel(root, tasks, gaps, chinese_font) if (root/'diagnostics.json').exists() else ''
    colors = ['#29275a', '#cb5e25', '#25709c', '#35855f', '#94919d', '#ac3875', '#ab850b', '#1b817a']
    labels = TASK_NAMES if chinese_font else {
        'recall': 'Delayed recall', 'binding': 'Key-value retrieval', 'arithmetic': 'Delayed mod-4 arithmetic'}
    legend_names = {'Full-memory RTRL': MODEL_NAMES['Full-memory RTRL'], sparse: '稀疏 1024'} if chinese_font else {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey='row', layout='constrained')
    for col, task in enumerate(tasks):
        for mi, model in enumerate(models):
            data = [index[model, task, g] for g in gaps]
            for row, metric in enumerate(('accuracy', 'paired_accuracy')):
                ax = axes[row, col]
                y = [100*d[metric] for d in data]
                ax.plot(gaps, y, 'o-', color=colors[mi], lw=2.8 if mi == 0 else 1.4, ms=4,
                        label=legend_names.get(model, model))
                if row == 0 and mi == 0:
                    ax.fill_between(gaps, [100*d['accuracy_min'] for d in data],
                                    [100*d['accuracy_max'] for d in data], color=colors[mi], alpha=.16)
        for row, ax in enumerate(axes[:, col]):
            ax.axvline(gaps[-1], color='#777777', ls=':', lw=1)
            ax.axhline(25 if row == 0 else 6.25, color='#555555', ls='--', lw=.8)
            ax.set(xscale='log', ylim=(-3, 103), xticks=gaps, xticklabels=[str(g) for g in gaps])
            ax.minorticks_off()
        axes[0, col].set_title(labels[task])
        axes[1, col].set_xlabel('干扰步数（256 步未参与训练）' if chinese_font else 'Distractor steps (256 unseen in training)')
    axes[0, 0].set_ylabel('单题准确率（%）' if chinese_font else 'Accuracy (%)')
    axes[1, 0].set_ylabel('成对正确率（%）' if chinese_font else 'Both cue variants correct (%)')
    fig.suptitle('长程记忆与处理评测 · 三个种子 · 冻结测试' if chinese_font else
                 'Long-memory benchmark | 3 seeds, frozen held-out evaluation', fontsize=16)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=len(models), frameon=False)
    for ext in ('png', 'svg'):
        fig.savefig(root/f'long_memory.{ext}', dpi=175)
    plt.close(fig)

    last_gap = gaps[-1]
    overview_rows = []
    for model in models:
        row = [index[model, task, last_gap] for task in tasks]
        cells = ''.join(f'<td class="metric"><strong>{r["accuracy"]:.1%}</strong>'
                        f'<small>成对 {r["paired_accuracy"]:.1%}</small></td>' for r in row)
        row_style = ' class="current"' if model == models[0] else ''
        overview_rows.append(f'<tr{row_style}><th scope="row">{MODEL_NAMES[model]}</th>'
                             f'{cells}<td class="metric">{row[0]["params"]:,}</td></tr>')
    overview = '\n'.join(overview_rows)
    matrix = {task: {model: [index[model, task, gap] for gap in gaps] for model in models} for task in tasks}
    description = (f'{len(tasks)} 项任务，{len(gaps)} 档干扰长度，{len(models)} 种模型。'
                   f'每个任务每个种子训练 {protocol["train_per_gap"]*(len(gaps)-1)} 题，'
                   f'每档测试 {protocol["test_per_gap"]} 道原题及同量配对题。')
    if sparse in models:
        description += (f'稀疏原型有 {protocol["sparse_width"]:,} 个神经元和 {protocol["sparse_edges"]:,} 条循环边，'
                        f'以每批 {protocol["sparse_batch"]} 题做 {protocol["sparse_updates"]} 次完整序列更新。'
                        '20 神经元版及六个基线沿用此前同题成绩，每题一次更新。')
    resources = ''
    if (root/'resources.json').exists():
        profiles = read(root/'resources.json')
        lookup = {(r['model'], r['mode']): r for r in profiles}
        assert {(name, mode) for name in models for mode in ('inference', 'training')} <= set(lookup)
        resource_rows = []
        for name in models:
            cells = ''.join(f'<td class="metric">{lookup[name, mode]["median_ms_per_token"]:.4f}</td>'
                            f'<td class="metric">{lookup[name, mode]["peak_mib"]:.1f}</td>'
                            for mode in ('inference', 'training'))
            resource_rows.append(f'<tr class="{"current" if name == models[0] else ""}">'
                                 f'<th scope="row">{MODEL_NAMES[name]}</th>{cells}</tr>')
        resources = ('<h2>本机耗时与峰值内存</h2><p>单线程 CPU，每个模型和模式在独立进程测量。推理用 256 步干扰复述题，'
                     '训练探针用 128 步干扰复述题，含梯度与一次更新。每步耗时为整题耗时除以输入位数。'
                     '内存为进程峰值常驻内存，包含 Python、PyTorch 和权重，训练列对应单题探针。'
                     '20 神经元版推理使用与准确率测试相同的冻结前向路径。</p>'
                     '<div class="table-wrap"><table><thead><tr><th>模型</th><th class="metric">推理毫秒/步</th>'
                     '<th class="metric">推理峰值（MiB）</th><th class="metric">训练毫秒/步</th>'
                     '<th class="metric">训练峰值（MiB）</th></tr></thead><tbody>'
                     + ''.join(resource_rows) + '</tbody></table></div><p><a href="resources.json">资源原始记录</a></p>')
        if (sparse, 'training-batch8') in lookup:
            batch_probe = lookup[sparse, 'training-batch8']
            resources += (f'<p>稀疏原型以实际训练批量 {batch_probe["batch_size"]} 题另测，'
                          f'每输入位耗时 {batch_probe["median_ms_per_token"]:.4f} 毫秒，'
                          f'进程峰值 {batch_probe["peak_mib"]:.1f} MiB。</p>')
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="only light"><title>长程记忆评测 · 中文对比版</title>
<style>
:root{color-scheme:only light;background:#fff;color:#1d2431}
body{margin:0}
#memory-report{box-sizing:border-box;max-width:1200px;margin:0 auto;padding:28px 18px;font:16px/1.6 system-ui}
#memory-report :is(h1,h2,p,label,td){color:#1d2431}
#memory-report :is(h1,h2){line-height:1.2}
#memory-report a{color:#154d9e}#memory-report :focus-visible{outline:2px solid #154d9e;outline-offset:2px}
#memory-report img{width:100%;background:#fff}#memory-report table{border-collapse:collapse;width:100%;background:#fff}
#memory-report .table-wrap{overflow-x:auto;border:1px solid #cbd5e1;border-radius:10px;margin:12px 0 24px}
#memory-report table{min-width:650px;font-variant-numeric:tabular-nums}
#memory-report :is(th,td){padding:10px 14px;text-align:left;border-bottom:1px solid #d7e0ec}
#memory-report :is(thead th,pre,select,option){background:#edf3fa;color:#1d2431}
#memory-report th[scope=row]{white-space:nowrap;font-weight:600;color:#1d2431}
#memory-report tbody tr:nth-child(even){background:#f7f9fc}
#memory-report tr.current{background:#eaf2ff;font-weight:650}
#memory-report tr.current th[scope=row]{border-left:4px solid #235aa6}
#memory-report :is(.metric,.number){text-align:right}
#memory-report small{display:block;color:#415775;white-space:nowrap;font-size:.8rem;font-weight:500}
#memory-report .unseen{background:#fff3da;border-left:2px solid #e4c787}
#memory-report .bar{display:block;min-width:55px;height:5px;background:#dee7f2;border-radius:4px;overflow:hidden;margin:4px 0 0}
#memory-report .bar i{display:block;height:100%;background:#3066af;border-radius:4px}
#memory-report .correct{color:#12654c}#memory-report .incorrect{color:#9c2e26}
#memory-report select{margin:0 10px 0 4px;padding:3px;border:1px solid #909bad;font:inherit}
#memory-report pre{white-space:pre-wrap;padding:10px}#memory-report mark{background:#ffe19c;color:#2c2005}
</style></head><body>
<main id="memory-report" style="background-color:#fff;color:#1d2431;color-scheme:only light">
<h1>长程记忆与处理评测</h1><p>'''+description+'''</p>
<h2>256 步干扰 · 三项任务一览</h2>
<p>256 步未参与训练。大字为单题准确率，小字为同一题改动早期线索前后都答对的比例。蓝色行是当前模型。</p>
<div class="table-wrap"><table aria-label="256 步干扰成绩">
<thead><tr><th scope="col">模型</th><th scope="col" class="metric">延迟复述</th><th scope="col" class="metric">按键查值</th><th scope="col" class="metric">模 4 加减</th><th scope="col" class="metric">参数量</th></tr></thead>
<tbody>'''+overview+'''</tbody></table></div>
'''+comparison+diagnostic+'''
<h2>各长度成绩对比</h2>
<label>任务<select id="matrix-task"><option value="recall">延迟复述</option><option value="binding">按键查值</option><option value="arithmetic">模 4 加减</option></select></label>
<label>指标<select id="metric"><option value="accuracy">单题准确率</option><option value="paired_accuracy">成对正确率</option><option value="nll">负对数似然</option><option value="cue_effect">线索效应</option></select></label>
<p id="matrix-note" aria-live="polite"></p><div class="table-wrap" id="matrix"></div>
<h2>成绩随干扰步数变化</h2>
<p>上排是原样本准确率，阴影是当前模型三个种子的最小至最大值。下排要求原样本和修改线索后的样本都答对。虚线是四类均匀独立猜测的期望，256 步未参与训练。</p>
<img src="long_memory.svg" alt="各模型准确率和成对准确率随干扰步数的变化">
'''+resources+'''
<h2>同一干扰串，改动一处线索</h2><p>每个任务和长度固定展示第一个测试对，使用种子 2。黄色为唯一改动，表中概率是对应正确答案的概率。</p>
<label>任务<select id="task"><option value="recall">延迟复述</option><option value="binding">按键查值</option><option value="arithmetic">模 4 加减</option></select></label>
<label>干扰步数<select id="gap"><option>8</option><option>32</option><option>64</option><option>128</option><option>256</option></select></label>
<div id="pair"></div><h2>下载成绩</h2><p><a href="scores-zh.csv">中文成绩表</a> · <a href="scores.csv">聚合原始 CSV</a> · <a href="per_seed.csv">逐种子 CSV</a> · <a href="audit.json">完整性核验</a></p>
</main>
<script>
const samples = '''+json.dumps(examples)+''';
const scores = '''+json.dumps(matrix)+''';
const names = '''+json.dumps(MODEL_NAMES)+''';
const currentModel = '''+json.dumps(models[0])+''';
const byId = id => document.getElementById(id);
const matrixTask = byId('matrix-task'), metric = byId('metric');
const notes = {
 accuracy: '单题准确率越高越好。四类均匀猜测的期望为 25%。',
 paired_accuracy: '修改线索前后都答对才计分。两次独立均匀猜测的期望为 6.25%。',
 nll: '负对数似然按正确答案的预测概率计算，越低越好。',
 cue_effect: '只修改早期线索后，正确答案概率沿正确方向移动的平均量，越高越好。'
};
function drawMatrix(){
 const entries = Object.entries(scores[matrixTask.value]), key = metric.value;
 const isRate = key === 'accuracy' || key === 'paired_accuracy';
 const unseen = i => i === entries[0][1].length - 1;
 const headers = entries[0][1].map((r,i)=>`<th scope="col" class="number ${unseen(i)?'unseen':''}">${r.gap} 步${unseen(i)?'<small>训练未见</small>':''}</th>`).join('');
 const rows = entries.map(([model,series])=>{
   const cells = series.map((r,i)=>{
     const value = r[key];
     const formatted = isRate ? `${(value*100).toFixed(1)}%` : value.toFixed(3);
     const bar = isRate ? `<span aria-hidden="true" class="bar"><i style="width:${value*100}%"></i></span>` : '';
     return `<td class="number ${unseen(i)?'unseen':''}">${formatted}${bar}</td>`;
   }).join('');
   return `<tr class="${model===currentModel?'current':''}"><th scope="row">${names[model]}</th>${cells}</tr>`;
 }).join('');
 byId('matrix-note').textContent = notes[key];
 byId('matrix').innerHTML=`<table aria-label="各干扰步数成绩"><thead><tr><th scope="col">模型</th>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
}
matrixTask.onchange=metric.onchange=drawMatrix;drawMatrix();
const task = document.getElementById('task'), gap = document.getElementById('gap');
const argmax = p => p.indexOf(Math.max(...p));
const cell = (p,y) => {
 const predicted = argmax(p), correct = predicted === y;
 return `<td class="metric"><strong>预测 ${predicted}</strong> · <span class="${correct?'correct':'incorrect'}">${correct?'答对':'答错'}</span><small>正确答案概率 ${(p[y]*100).toFixed(1)}%</small></td>`;
};
function draw(){
 const entries = Object.entries(samples[task.value][gap.value]), e = entries[0][1];
 const line = x => x.map((v,i)=>e.digits[i]!==e.paired_digits[i]?`<mark>${v}</mark>`:v).join(' ');
 const rows = entries.map(([model,r])=>`<tr class="${model===currentModel?'current':''}"><th scope="row">${names[model]}</th>${cell(r.probabilities,r.target)}${cell(r.paired_probabilities,r.paired_target)}</tr>`).join('');
 byId('pair').innerHTML=`<pre>原输入 ${line(e.digits)}  → 答案 ${e.target}\n改线索 ${line(e.paired_digits)}  → 答案 ${e.paired_target}</pre><div class="table-wrap"><table><thead><tr><th scope="col">模型</th><th scope="col" class="metric">原题 · 答案 ${e.target}</th><th scope="col" class="metric">改线索后 · 答案 ${e.paired_target}</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
task.onchange=gap.onchange=draw;draw();
</script></body></html>'''
    for filename in ('report.html', 'report-readable.html', 'report-cn.html'):
        (root/filename).write_text(html, encoding='utf-8')
    print(f'Verified {len(hashes)//2} runs and wrote {len(summary)} score rows to {root}')


if __name__ == '__main__':
    main()
