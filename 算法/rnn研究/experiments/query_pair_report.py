"""Recompute query-switch results and write the Chinese comparison report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from long_memory_report import MODEL_NAMES, TASK_NAMES
from matplotlib import font_manager
import matplotlib.pyplot as plt
from query_pair_eval import check_pair, expected, training_batches
from standard_memory_eval import digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root, p):
    for parent, key in ((root, 'data_sha256'), (root/'source', 'source_sha256')):
        for name, checksum in p[key].items():
            assert digest(parent/name) == checksum, name
    data = {task:read(root/f'{task}-data.json') for task in p['tasks']}
    for task, dataset in data.items():
        forbidden = {tuple(x) for x in dataset['forbidden_prefixes']}
        observed = set()
        for split in ('validation', 'test'):
            for gap in p['gaps']:
                for row in dataset[split][str(gap)]:
                    check_pair(task, gap, row)
                    if task != 'arithmetic' or gap != 0:
                        key = tuple(row[0][:-1])
                        assert key not in observed and key in forbidden
                        observed.add(key)
        for phase, seed in itertools.product(p['train_gaps'], [p['tuning_seed'], *p['seeds']]):
            rows = [r for batch in training_batches(task, phase, seed, forbidden) for r in batch]
            assert len(rows) == p['budget']
            assert all(tuple(x[:-1]) not in forbidden and expected(task, x) == y for x, y in rows)
    selected = {(r['phase'], r['model'], r['task']):r['lr'] for r in read(root/'selection.json')}
    by_key = defaultdict(list)
    seed_rows, examples, files = [], {}, {}
    trials = evaluations = decisions = 0
    for phase, model, task in itertools.product(p['train_gaps'], p['models'], p['tasks']):
        lr = selected[phase, model, task]
        prefix = f'{phase}_{model}_{task}'
        ranks = []
        for rate in p['rates']:
            path = root/'tuning'/f'{prefix}_{p["tuning_seed"]}_{rate:g}.json'
            r = read(path)
            scores = r['scores']
            assert r['lr'] == rate and r['seed'] == p['tuning_seed']
            assert (r['phase'], r['model'], r['task']) == (phase, model, task)
            assert [s['gap'] for s in scores] == p['train_gaps'][phase]
            assert len(r['loss'])*p['batch'] == p['budget'] and all(map(math.isfinite,r['loss']))
            ranks.append((sum(s['paired_accuracy'] for s in scores), -sum(s['nll'] for s in scores), -rate))
            files[str(path.relative_to(root))] = digest(path)
            trials += 1
        assert -max(ranks)[2] == lr
        for seed in p['seeds']:
            path = root/'evaluation'/f'{prefix}_{seed}_{lr:g}.json'
            r = read(path)
            assert (r['phase'], r['model'], r['task'], r['seed'], r['lr']) == (phase, model, task, seed, lr)
            assert len(r['loss'])*p['batch'] == p['budget'] and all(map(math.isfinite,r['loss']))
            assert [s['gap'] for s in r['scores']] == ([0] if phase == 'zero' else p['gaps'])
            for row in r['scores']:
                gap = row['gap']
                pairs = data[task]['test'][str(gap)]
                hits = []
                by_query = defaultdict(set)
                for i, ((x, xp, y, yp), probs) in enumerate(zip(pairs, row['probabilities'], strict=True)):
                    a, b = probs
                    assert all(len(v)==10 and all(0<=w<=1 for w in v) and abs(sum(v)-1)<1e-6 for v in probs)
                    pa, pb = (max(range(10), key=v.__getitem__) for v in probs)
                    hits.append((pa == y, pb == yp))
                    by_query[x[-1]].add(pa)
                    by_query[xp[-1]].add(pb)
                    decisions += 2
                    if phase == 'delay' and gap == 256 and seed == p['seeds'][0] and i == 0:
                        examples.setdefault(task, {})[model] = dict(x=x, xp=xp, y=y, yp=yp, pa=pa, pb=pb,
                                                                   p=a[y], pp=b[yp])
                paired = mean(a and b for a,b in hits)
                assert math.isclose(mean(a for a,b in hits), row['accuracy'], abs_tol=1e-7)
                assert math.isclose(paired, row['paired_accuracy'], abs_tol=1e-7)
                seed_score = dict(phase=phase, model=model, task=task, gap=gap, seed=seed,
                                  single=mean(x for pair in hits for x in pair), paired=paired,
                                  fixed_by_query=all(len(values) == 1 for values in by_query.values()),
                                  pairs=len(pairs), lr=lr, parameters=r['parameters'])
                by_key[phase, task, model, gap].append(seed_score)
                seed_rows.append(seed_score)
            for file in (path, path.with_suffix('.pt')):
                files[str(file.relative_to(root))] = digest(file)
            evaluations += 1
    summaries = []
    for (phase, task, model, gap), group in by_key.items():
        values = [r['paired'] for r in group]
        summaries.append(dict(phase=phase, task=task, model=model, gap=gap, paired=mean(values),
                              single=mean(r['single'] for r in group), minimum=min(values), maximum=max(values),
                              fixed_query_seeds=sum(r['fixed_by_query'] for r in group),
                              pairs=sum(r['pairs'] for r in group), lr=group[0]['lr'], parameters=group[0]['parameters']))
    (root/'audit.json').write_text(json.dumps(dict(tuning_runs=trials, evaluation_runs=evaluations,
        test_decisions=decisions, aggregates=len(summaries), sha256=files), indent=2), encoding='utf-8')
    return summaries, seed_rows, examples


def make_plot(root, p, index, ceiling):
    plt.style.use('seaborn-v0_8-whitegrid')
    font = next((f.name for f in font_manager.fontManager.ttflist if f.name in
                 ('Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC')), None)
    if font:
        plt.rcParams['font.family'] = font
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharey=True, layout='constrained')
    colors = ('#28659c', '#a84973', '#087763')
    short = ('GRU','TCN-512','稀疏 1024' if font else 'Sparse 1024')
    x = list(range(len(p['gaps'])))
    for task, ax in zip(p['tasks'], axes, strict=True):
        for model, label, color in zip(p['models'], short, colors, strict=True):
            scores = [index['delay', task, model, gap] for gap in p['gaps']]
            ax.plot(x, [100*r['paired'] for r in scores], 'o-', color=color, label=label)
            ax.fill_between(x, [100*r['minimum'] for r in scores], [100*r['maximum'] for r in scores], color=color, alpha=.12)
        ax.axvline(x[-1], color='#686868', ls=':', lw=1)
        ax.axhline(6.25, color='#666', ls='--', lw=.8)
        ax.plot(x, [100*ceiling[task][gap] for gap in p['gaps']], color='#626262', ls='-.', lw=1.2,
                label='只读查询的上限' if font else 'Query-only ceiling')
        ax.set(ylim=(-2,103), xticks=x, xticklabels=p['gaps'], title=TASK_NAMES[task] if font else task,
               xlabel='干扰步数（256 未参与训练）' if font else 'Distractors (256 unseen)')
    axes[0].set_ylabel('查询切换成对正确率（%）' if font else 'Query-switch paired accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    for ext in ('png', 'svg'):
        fig.savefig(root/f'query_pairs.{ext}', dpi=170)
    plt.close(fig)


def query_only_ceiling(pairs):
    """Best fixed mapping from the last token to an answer on this test set."""
    queries = sorted({seq[-1] for x, xp, _, _ in pairs for seq in (x, xp)})
    return max(mean(mapping[x[-1]] == y and mapping[xp[-1]] == yp for x, xp, y, yp in pairs)
               for answers in itertools.product(range(4), repeat=len(queries))
               for mapping in [dict(zip(queries, answers, strict=True))])


def report(root, p, summaries, seed_rows, examples):
    index = {(r['phase'], r['task'], r['model'], r['gap']):r for r in summaries}
    ceiling = {task:{int(gap):query_only_ceiling(pairs) for gap,pairs in
                     read(root/f'{task}-data.json')['test'].items()} for task in p['tasks']}
    make_plot(root, p, index, ceiling)
    for filename, rows in [('scores-zh.csv', summaries), ('per-seed-zh.csv', seed_rows)]:
        with (root/filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            extra = ['种子', '每个查询是否固定预测'] if 'seed' in rows[0] else ['配对最低种子', '配对最高种子', '每个查询固定预测的种子数']
            writer.writerow(['阶段','任务','模型','干扰步数','单题准确率','成对正确率','题对数','学习率','参数量',*extra])
            for r in rows:
                tail = [r['seed'], '是' if r['fixed_by_query'] else '否'] if 'seed' in r else [
                    f'{r["minimum"]:.2%}', f'{r["maximum"]:.2%}', r['fixed_query_seeds']]
                writer.writerow(['0 步规则诊断' if r['phase']=='zero' else '长程训练', TASK_NAMES[r['task']],
                    MODEL_NAMES[r['model']], r['gap'], f'{r["single"]:.2%}', f'{r["paired"]:.2%}',
                    r['pairs'], r['lr'], r['parameters'], *tail])
    overview = []
    for model in p['models']:
        cells = []
        for task, phase, gap in [(task, phase, gap) for task in p['tasks'] for phase,gap in (('zero',0),('delay',256))]:
            r = index[phase, task, model, gap]
            cells.append(f'<td><b>{r["paired"]:.1%}</b><small>单题 {r["single"]:.1%}</small></td>')
        overview.append(f'<tr><th scope="row">{MODEL_NAMES[model]}</th>{"".join(cells)}</tr>')
    ceiling_cells = []
    for task in p['tasks']:
        ceiling_cells.extend(f'<td>{ceiling[task][gap]:.1%}</td>' for gap in (0,256))
    overview.append('<tr><th scope="row">只读查询的成绩上限</th>'+''.join(ceiling_cells)+'</tr>')
    fixed = '、'.join(f'{MODEL_NAMES[m]} {index["delay", "arithmetic", m, 256]["fixed_query_seeds"]}/{len(p["seeds"])}'
                     for m in p['models'])
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="only light"><title>查询切换配对评测</title><style>
:root{color-scheme:only light;background:#fff;color:#18263b}body{margin:0;font:16px/1.6 system-ui}
main{max-width:1170px;margin:auto;padding:25px;box-sizing:border-box}h1,h2,p,label{color:#18263b}
a{color:#164f9b}select{font:inherit;margin:4px 12px 4px 4px;padding:4px;background:#edf4fb;color:#18263b}
.wrap{overflow:auto;border:1px solid #cbd5e1;border-radius:8px;margin:12px 0 22px}
table{min-width:650px;width:100%;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;border-bottom:1px solid #d7e0ec;color:#18263b;text-align:right}thead th{background:#eaf1fa}
th:first-child{text-align:left;white-space:nowrap}tbody tr:nth-child(even){background:#f6f9fc}.unseen{background:#fff2d6}
small{display:block;font-size:.8rem;color:#415775}img{width:100%;background:white}
pre{white-space:pre-wrap;background:#edf4fb;padding:12px;color:#18263b}mark{background:#ffe094;color:#2c220e}
</style></head><body><main style="background:#fff;color:#18263b;color-scheme:only light">
<h1>查询切换配对评测</h1>
<p>同一对题只改最后一位查询。查值使用三个不同值，只换查询键；模 4 加减固定操作数并使用奇数 B，只换运算指令。忽略查询、两边给同一答案的策略无法通过任一配对。</p>
<p>三个模型每次均训练 2,000 道题、每批 16 题，共 125 次更新。从同一组三个学习率候选中用验证集选参，再用三个种子训练。</p>
<h2>0 步规则诊断与 256 步外推</h2><p>大字为成对正确率，小字为两边答案合并的单题准确率。0 步诊断和长程阶段分别从头训练。</p>
<div class="wrap"><table><thead><tr><th rowspan="2">模型</th><th colspan="2">按键查值</th><th colspan="2">模 4 加减</th></tr>
<tr><th>0 步诊断</th><th>外推 256 步</th><th>0 步诊断</th><th>外推 256 步</th></tr></thead><tbody>'''+''.join(overview)+'''</tbody></table></div>
<p>查值的 0 步诊断使用未见键值配置；模 4 运算的 0 步诊断穷举 8 个奇数 B 的操作数组合，题目形式与训练重合，测的是有限规则表拟合。</p>
<p>最后一行穷举只看末位查询的固定输出映射，用测试答案计算其可达到的最高成对分数，作为排除捷径的诊断参照。该行不是训练模型或独立测试成绩。</p>
<p>256 步模 4 测试中，对每个运算指令始终预测同一个答案的种子数为 '''+fixed+'''。统计基于测试集中每种指令的所有最高概率预测。</p>
<h2>各档干扰对比</h2><label>任务<select id="task"><option value="binding">按键查值</option><option value="arithmetic">模 4 加减</option></select></label>
<label>指标<select id="metric"><option value="paired">成对正确率</option><option value="single">单题准确率</option></select></label>
<p>长程训练覆盖 0–128 步，256 步只用于最终测试。成对分数下方为三个种子的最低值和最高值。图中灰色点划线是只读查询的上限。</p><div class="wrap" id="matrix"></div>
<img src="query_pairs.svg" alt="各档干扰下的成对正确率和三种子范围">
<h2>同一段输入，改最后一位查询</h2><p>显示所选任务的第一对 256 步题，使用种子 2。表格概率对应正确答案。</p><div id="example"></div>
<p><a href="scores-zh.csv">中文聚合成绩</a> · <a href="per-seed-zh.csv">逐种子成绩</a> · <a href="audit.json">核验记录</a></p></main>
<script>
const rows='''+json.dumps(summaries)+''', examples='''+json.dumps(examples)+''', names='''+json.dumps(MODEL_NAMES,ensure_ascii=False)+''';
const el=id=>document.getElementById(id), pct=v=>(100*v).toFixed(1)+'%';
function draw(){
 const task=el('task').value, metric=el('metric').value, rs=rows.filter(r=>r.phase==='delay'&&r.task===task);
 const groups=Object.groupBy(rs,r=>r.model);
 const table=Object.entries(groups).map(([m,group])=>`<tr><th scope="row">${names[m]}<small>${group[0].parameters.toLocaleString()} 参数 · 学习率 ${group[0].lr}</small></th>${group.map(r=>`<td class="${r.gap===256?'unseen':''}"><b>${pct(r[metric])}</b>${metric==='paired'?`<small>${pct(r.minimum)}–${pct(r.maximum)}</small>`:''}</td>`).join('')}</tr>`).join('');
 el('matrix').innerHTML=`<table><thead><tr><th>模型</th>${[0,8,32,64,128,256].map(g=>`<th>${g} 步${g===256?'<small>训练未见</small>':''}</th>`).join('')}</tr></thead><tbody>${table}</tbody></table>`;
 const entries=Object.entries(examples[task]), e=entries[0][1], line=x=>`${x.slice(0,-1).join(' ')} <mark>${x.at(-1)}</mark>`;
 const sample=entries.map(([m,r])=>`<tr><th scope="row">${names[m]}</th><td>预测 ${r.pa} · ${pct(r.p)}</td><td>预测 ${r.pb} · ${pct(r.pp)}</td></tr>`).join('');
 el('example').innerHTML=`<pre>原查询 ${line(e.x)} → 答案 ${e.y}\n改查询 ${line(e.xp)} → 答案 ${e.yp}</pre><div class="wrap"><table><thead><tr><th>模型</th><th>原查询</th><th>改查询后</th></tr></thead><tbody>${sample}</tbody></table></div>`;
}
el('task').onchange=el('metric').onchange=draw;draw();
</script></body></html>'''
    (root/'report-cn.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    protocol = read(args.root/'protocol.json')
    summaries, seeds, examples = audit(args.root, protocol)
    report(args.root, protocol, summaries, seeds, examples)
    print(f'Verified {len(seeds)} seed-level rows and {len(summaries)} aggregates.')


if __name__ == '__main__':
    main()
