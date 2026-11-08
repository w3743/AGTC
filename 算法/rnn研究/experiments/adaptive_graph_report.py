"""Audit and visualize learned historical routing."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from long_memory_report import TASK_NAMES
from matplotlib import font_manager
import matplotlib.pyplot as plt
from adaptive_graph_eval import candidates, source_edits
from query_pair_eval import check_pair, expected
from standard_memory_eval import digest


NAMES = {'adaptive':'自适应稀疏连边', 'fixed':'固定位置连边', 'dense':'全连接注意力'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def winner(p):
    assert len(p) == 10 and all(0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-6
    return max(range(10), key=p.__getitem__)


def audit(root, protocol):
    for folder, key in ((root, 'data_sha256'), (root/'source', 'source_sha256')):
        for name, checksum in protocol[key].items():
            assert digest(folder/name) == checksum
    groups, files = defaultdict(list), {}
    test = {t:read(root/f'{t}-data.json')['test'] for t in protocol['tasks']}
    edge_examples = {}
    predictions = 0
    for mode, task, seed in itertools.product(protocol['modes'], protocol['tasks'], protocol['seeds']):
        file = root/f'{mode}_{task}_{seed}.json'
        run = read(file)
        assert (run['mode'],run['task'],run['seed']) == (mode,task,seed)
        assert len(run['losses'])*protocol['batch'] == protocol['train_questions'] and all(map(math.isfinite,run['losses']))
        assert [(x['shift'],x['gap']) for x in run['scores']] == list(itertools.product(protocol['test_shifts'],protocol['gaps']))
        for s in run['scores']:
            shift, gap = s['shift'], s['gap']
            rows = []
            for idx, (x, xp, y, yp) in enumerate(test[task][str(gap)]):
                check_pair(task,gap,(x,xp,y,yp))
                row = s['rows'][idx]
                assert (row['y'],row['yp']) == (y,yp)
                a,b = winner(row['probabilities']) == y, winner(row['swapped_probabilities']) == yp
                edit_hits = []
                for (pos,value), edit in zip(source_edits(task,x),row['source_edits'],strict=True):
                    changed = x.copy()
                    changed[pos] = value
                    assert edit['y'] == expected(task,changed) and edit['y'] != y
                    edit_hits.append(winner(edit['probabilities']) == edit['y'])
                predictions += 2+len(edit_hits)
                result = dict(single=(a+b)/2, paired=a and b, source_pair=a and all(edit_hits), joint=a and b and all(edit_hits))
                if mode != 'dense':
                    relevant = [i+shift for i in candidates(task,x)]
                    swapped = [i+shift for i in candidates(task,xp)]
                    assert row['relevant'] == relevant and row['swapped_relevant'] == swapped
                    u,v = [],[]
                    for edge_key, dest in (('edges',u), ('swapped_edges',v)):
                        assert len(row[edge_key]) == 2 and all(len(hop)==4 for hop in row[edge_key])
                        for hop in row[edge_key]:
                            for head in hop:
                                assert len(head)==4 and len(set(head))==4 and all(0<=i<len(x)+shift-1 for i in head)
                                dest.append(set(head))
                    result.update(relevant_edges=(sum(len(e.intersection(relevant)) for e in u)+
                                                   sum(len(e.intersection(swapped)) for e in v))/64,
                                  rewire=mean(1-len(e1 & e2)/4 for e1,e2 in zip(u,v,strict=True)))
                    if mode == 'fixed':
                        assert result['rewire'] == 0
                if mode == 'adaptive':
                    result['frozen'] = a and winner(row['frozen_edge_probabilities']) == yp
                    predictions += 1
                    if seed == protocol['seeds'][0] and shift == protocol['test_shifts'][-1] and gap == 256 and idx == 0:
                        edge_examples[task] = dict(x=x, xp=xp, shift=shift, result=row)
                rows.append(result)
            assert len(rows) == len(s['rows'])
            groups[mode,task,shift,gap].append(dict(seed=seed, parameters=run['parameters'], n=len(rows),
                                                   **{k:mean(r[k] for r in rows) for k in rows[0]}))
        for p in (file,file.with_suffix('.pt')):
            files[p.name] = digest(p)
    summary, seeds = [], []
    for (mode,task,shift,gap), group in groups.items():
        key = dict(mode=mode,task=task,shift=shift,gap=gap)
        seeds.extend({**key,**r} for r in group)
        metrics = {k:mean(r[k] for r in group) for k in group[0] if k not in ('seed','n','parameters')}
        summary.append(dict(**key,**metrics,n=sum(r['n'] for r in group),parameters=group[0]['parameters'],
                            minimum=min(r['paired'] for r in group),maximum=max(r['paired'] for r in group)))
    (root/'audit.json').write_text(json.dumps(dict(runs=len(files)//2,predictions=predictions,
                        aggregate_rows=len(summary),sha256=files),indent=2),encoding='utf-8')
    return summary,seeds,edge_examples


def save_csv(path, rows):
    fields = {'mode':'连边方式','task':'任务','shift':'额外前置干扰','gap':'线索后干扰','seed':'种子',
              'parameters':'参数量','n':'题对数','single':'单题正确率','paired':'换查询两边都正确',
              'source_pair':'改来源前后都正确','joint':'换查询与改来源全部正确','frozen':'复用旧边后的换查询正确率',
              'relevant_edges':'指向任务相关值节点的边占比','rewire':'换查询时被替换的边占比',
              'minimum':'成对最低种子','maximum':'成对最高种子'}
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.writer(f)
        w.writerow(fields.values())
        for r in rows:
            values = {**r,'mode':NAMES[r['mode']],'task':TASK_NAMES[r['task']]}
            for k in fields.keys()-{'mode','task','shift','gap','seed','parameters','n'}:
                if k in values:
                    values[k] = f'{values[k]:.2%}'
            w.writerow([values.get(k,'') for k in fields])


def report(root, protocol, summary, seeds, examples):
    save_csv(root/'scores-zh.csv',summary)
    save_csv(root/'per-seed-zh.csv',seeds)
    (root/'edge-examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    index = {(r['mode'],r['task'],r['shift'],r['gap']):r for r in summary}
    plt.style.use('seaborn-v0_8-whitegrid')
    font = next((f.name for f in font_manager.fontManager.ttflist if f.name in ('Microsoft YaHei','Noto Sans CJK SC','PingFang SC')),None)
    if font:
        plt.rcParams['font.family'] = font
    fig, axes = plt.subplots(2,2,figsize=(12,7),sharey=True,layout='constrained')
    for ri, task in enumerate(protocol['tasks']):
        for ci, shift in enumerate(protocol['test_shifts']):
            ax = axes[ri,ci]
            for mode, color in zip(protocol['modes'],('#087661','#a65e21','#386499'),strict=True):
                data = [index[mode,task,shift,g] for g in protocol['gaps']]
                x = range(len(data))
                ax.plot(x,[100*r['paired'] for r in data], marker='o', color=color,
                        ls='--' if mode == 'adaptive' else '-',
                        markerfacecolor='white' if mode == 'adaptive' else color,
                        zorder=4 if mode == 'adaptive' else 3, label=NAMES[mode] if font else mode)
                ax.fill_between(x,[100*r['minimum'] for r in data],[100*r['maximum'] for r in data],color=color,alpha=.10)
            ax.axvline(len(protocol['gaps'])-1,ls=':',color='#777')
            ax.set(xticks=range(len(protocol['gaps'])),xticklabels=protocol['gaps'],ylim=(-2,103),
                   title=f'{TASK_NAMES[task]} · 前置 {shift} 步' if font else f'{task}, prefix shift {shift}',
                   xlabel='线索后干扰步数' if font else 'Distractor steps')
        axes[ri,0].set_ylabel('换查询成对正确率（%）' if font else 'Query-pair accuracy (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    for ext in ('svg','png'):
        fig.savefig(root/f'adaptive_graph.{ext}',dpi=170)
    plt.close(fig)

    columns = [(t,s) for t in protocol['tasks'] for s in protocol['test_shifts']]
    last=protocol['gaps'][-1]
    overview=[]
    for mode in protocol['modes']:
        cells=[]
        for task, shift in columns:
            r=index[mode,task,shift,last]
            cells.append(f'<td><b>{r["paired"]:.1%}</b><small>来源配对 {r["source_pair"]:.1%} · 联合 {r["joint"]:.1%}</small></td>')
        overview.append(f'<tr><th scope="row">{NAMES[mode]}</th>{"".join(cells)}</tr>')
    intervention=[]
    for task,shift in columns:
        r=index['adaptive',task,shift,last]
        intervention.append(f'<tr><th scope="row">{TASK_NAMES[task]} · 前置 {shift} 步</th><td>{r["paired"]:.1%}</td>'
                            f'<td>{r["frozen"]:.1%}</td><td>{r["rewire"]:.1%}</td><td>{r["relevant_edges"]:.1%}</td></tr>')
    sample = examples['binding']
    x, xp, shift, route = (sample[k] for k in ('x','xp','shift','result'))
    counts = [list(itertools.chain.from_iterable(itertools.chain.from_iterable(route[key])))
              for key in ('edges','swapped_edges')]
    example_rows = ''.join(f'<tr><th scope="row">键 {x[i]} → 值 {x[i+1]}</th><td>{shift+i+1}</td>'
                           f'<td>{counts[0].count(shift+i+1)}</td><td>{counts[1].count(shift+i+1)}</td></tr>'
                           for i in (0,2,4))
    example = (f'<h2>一对真实选边</h2><p>种子 {protocol["seeds"][0]}，前置 {shift} 步、线索后 256 步。'
               f'查询从键 {x[-1]} 改为键 {xp[-1]}，正确值从 {route["y"]} 改为 {route["yp"]}。'
               '每次查询共选 32 条边，表中按值节点汇总被各跳各头选中的次数，位置从 0 开始。</p>'
               '<div class="wrap"><table><thead><tr><th>写入内容</th><th>值节点位置</th>'
               f'<th>原查询的边数</th><th>换查询后的边数</th></tr></thead><tbody>{example_rows}</tbody></table></div>')
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="only light"><title>自适应记忆拓扑实验</title><style>
:root{background:white;color:#18263b;color-scheme:only light}body{margin:0;font:16px/1.6 system-ui}
main{max-width:1170px;box-sizing:border-box;margin:auto;padding:24px}h1,h2,p,label{color:#18263b}
.wrap{overflow:auto;border:1px solid #cbd5e1;border-radius:8px;margin:10px 0 24px}
table{width:100%;min-width:640px;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;color:#18263b;border-bottom:1px solid #d7e0ec}thead th{background:#eaf1fa}
th:first-child{text-align:left;white-space:nowrap}tr:nth-child(even){background:#f6f9fc}
small{display:block;color:#415775;font-size:.8rem}img{width:100%;background:white}a{color:#15519c}
select{font:inherit;color:#18263b;background:#edf4fb;padding:4px;margin:4px 12px 4px 4px}.unseen{background:#fff1d5}
</style></head><body><main style="background:#fff;color:#18263b;color-scheme:only light">
<h1>自适应记忆拓扑实验</h1><p>模型把历史数字编码为节点，每次查询选择实际读取的连边。三种方式共用同一节点编码、消息更新和读出网络，仅改变选边规则。每组 8,000 道训练题、500 次更新，三个种子。训练前置噪声为 0–32 步，测试额外包含前置 64 步。</p>
<h2>256 步外推</h2><p>大字为换查询前后均正确。来源配对要求原题与更改每个相关值后的题都正确。联合要求两类变体全部正确。</p>
<div class="wrap"><table><thead><tr><th>连边方式</th>'''+''.join(f'<th>{TASK_NAMES[t]}<small>前置 {s} 步</small></th>' for t,s in columns)+'''</tr></thead><tbody>'''+''.join(overview)+'''</tbody></table></div>
<h2>连边干预</h2><p>对同一组已训练权重，换查询时强制读取原查询选中的边；仍重新计算新查询下的权重。相关边比例只统计实际连向正确值节点的边。换边比例按每跳每头分别计算。</p>
<div class="wrap"><table><thead><tr><th>任务</th><th>正常换查询</th><th>强制复用原边</th><th>被替换的边</th><th>相关值节点边占比</th></tr></thead><tbody>'''+''.join(intervention)+'''</tbody></table></div>
'''+example+'''
<h2>各档干扰成绩</h2><label>任务<select id="task"><option value="binding">按键查值</option><option value="arithmetic">模 4 加减</option></select></label>
<label>前置噪声<select id="shift"><option>0</option><option>64</option></select></label>
<label>指标<select id="metric"><option value="paired">换查询成对</option><option value="source_pair">改来源成对</option><option value="joint">全部变体正确</option><option value="single">单题</option></select></label>
<div class="wrap" id="matrix"></div><img src="adaptive_graph.svg" alt="成对成绩与三种子范围曲线">
<p><a href="scores-zh.csv">中文聚合成绩</a> · <a href="per-seed-zh.csv">逐种子成绩</a> · <a href="edge-examples.json">实际连边样例</a> · <a href="audit.json">核验记录</a></p>
</main><script>
const data='''+json.dumps(summary)+''', names='''+json.dumps(NAMES,ensure_ascii=False)+''', el=id=>document.getElementById(id);
function draw(){ const task=el('task').value, shift=+el('shift').value, metric=el('metric').value;
 const rows=Object.entries(names).map(([mode,label])=>{const group=data.filter(r=>r.mode===mode&&r.task===task&&r.shift===shift);
  return `<tr><th scope="row">${label}</th>${group.map(r=>`<td class="${r.gap===256?'unseen':''}">${(r[metric]*100).toFixed(1)}%</td>`).join('')}</tr>`}).join('');
 el('matrix').innerHTML=`<table><thead><tr><th>连边方式</th>${[0,8,32,64,128,256].map(g=>`<th>${g} 步${g===256?'<small>训练未见</small>':''}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;}
for(const id of ['task','shift','metric'])el(id).onchange=draw;draw();
</script></body></html>'''
    (root/'report-cn.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    protocol=read(args.root/'protocol.json')
    summary,seeds,examples=audit(args.root,protocol)
    report(args.root,protocol,summary,seeds,examples)
    print(f'Verified {len(summary)} aggregates and {len(seeds)} seed-level rows.')


if __name__=='__main__':
    main()
