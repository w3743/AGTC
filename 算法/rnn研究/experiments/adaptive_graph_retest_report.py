"""Verify fresh topology retest predictions and render Chinese graphic tables."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
from statistics import mean

from adaptive_graph_report import NAMES as GRAPH_NAMES, read, winner
from adaptive_graph_eval import source_edits
from adaptive_graph_retest import excluded_prefixes
from long_memory_report import TASK_NAMES
from matplotlib import font_manager
import matplotlib.pyplot as plt
from query_pair_eval import SPARSE, expected
from standard_memory_eval import digest


NAMES = {SPARSE:'旧稀疏 1024', 'GRU':'GRU', 'TCN-512':'TCN-512', **GRAPH_NAMES}
LAYOUT_NAMES = {'compact':'原顺序 · 紧凑线索', 'scattered':'换顺序 · 拆散线索'}


def canonical(task, layout, gap, sample):
    spans = sample['spans']
    order = [i for i,_,_ in spans]
    assert sorted(order) == list(range(len(spans)))
    assert (order == list(range(len(spans)))) == (layout == 'compact')
    assert all(v == k+1 for _,k,v in spans)
    assert all(a[2] < b[1] for a,b in zip(spans,spans[1:]))
    end = spans[-1][2]+1
    assert end-2*len(spans) == sample['extra']
    result = []
    for x,y in sample['variants']:
        assert len(x) == end+gap+2 and x[-2]==9
        prefix = list(itertools.chain.from_iterable((x[k],x[v]) for _,k,v in sorted(spans)))
        seq = prefix+x[end:]
        assert expected(task,seq)==y
        result.append((seq,y))
    x,y = result[0]
    xp,yp = result[1]
    assert x[:-1]==xp[:-1] and x[-1]!=xp[-1] and y!=yp
    for (pos,val),(edited,ans) in zip(source_edits(task,x),result[2:],strict=True):
        assert edited[pos]==val and sum(a!=b for a,b in zip(x,edited,strict=True))==1 and ans!=y
    return result


def audit(root,p):
    for folder,key in ((root,'data_sha256'),(root/'source','source_sha256')):
        for filename,checksum in p[key].items():
            assert digest(folder/filename)==checksum
    ref, graph = Path(p['reference']),Path(p['graph'])
    manifests = {folder:read(folder/'audit.json')['sha256'] for folder in (ref,graph)}
    for folder,names in ((ref,('sparse_input_memory.py','standard_memory_eval.py')),
                         (graph,('adaptive_memory_graph.py',))):
        historical = read(folder/'protocol.json')['source_sha256']
        assert all(p['source_sha256'][name] == historical[name] for name in names)
    data = {t:read(root/f'{t}-fresh.json') for t in p['tasks']}
    forbidden = {t:excluded_prefixes(t,ref,graph) for t in p['tasks']}
    for task,gap in itertools.product(p['tasks'],p['gaps']):
        a,b = (data[task][layout][str(gap)] for layout in p['layouts'])
        assert len(a)==len(b)==p['pairs']
        canonical_cases = [canonical(task,'compact',gap,s) for s in a]
        assert canonical_cases == [canonical(task,'scattered',gap,s) for s in b]
        assert all(tuple(seq[:-1]) not in forbidden[task] for case in canonical_cases for seq,_ in case)
    groups, files, examples = defaultdict(list), {}, defaultdict(dict)
    predictions = 0
    for model,task,seed in itertools.product(p['models'],p['tasks'],p['seeds']):
        path = root/f'{model}_{task}_{seed}.json'
        run = read(path)
        assert (run['model'],run['task'],run['seed']) == (model,task,seed)
        weight=Path(run['checkpoint'])
        parent=graph if model in GRAPH_NAMES else ref
        assert digest(weight)==run['checkpoint_sha256']==manifests[parent][str(weight.relative_to(parent))]
        assert [(r['layout'],r['gap']) for r in run['groups']]==list(itertools.product(p['layouts'],p['gaps']))
        for group in run['groups']:
            layout,gap=group['layout'],group['gap']
            stats=[]
            for i,(sample,row) in enumerate(zip(data[task][layout][str(gap)],group['rows'],strict=True)):
                targets=[y for _,y in sample['variants']]
                outputs=[winner(v) for v in row['probabilities']]
                hits=[a==b for a,b in zip(outputs,targets,strict=True)]
                predictions+=len(hits)
                stat=dict(single=mean(hits[:2]),paired=all(hits[:2]),source=hits[0] and all(hits[2:]),joint=all(hits))
                if model=='adaptive':
                    stat['frozen']=hits[0] and winner(row['frozen'])==targets[1]
                    rewires=[]
                    for base,swapped in row['first_edge_sets']:
                        for a,b in zip(base,swapped,strict=True):
                            assert len(set(a))==len(set(b))==4 and all(0<=x<len(sample['variants'][0][0])-1 for x in a+b)
                            rewires.append(1-len(set(a)&set(b))/4)
                    assert len(rewires)==8
                    stat['rewire']=mean(rewires)
                    predictions+=1
                if gap==1024 and layout=='scattered' and i==0 and seed==p['seeds'][0]:
                    examples[task][model]=dict(outputs=outputs,targets=targets,hits=hits)
                stats.append(stat)
            groups[model,task,layout,gap].append(dict(seed=seed,**{k:mean(r[k] for r in stats) for k in stats[0]}))
        files[path.name]=digest(path)
    out,per_seed=[],[]
    for (model,task,layout,gap),rows in groups.items():
        key=dict(model=model,task=task,layout=layout,gap=gap)
        per_seed.extend({**key,**r} for r in rows)
        out.append(dict(**key,**{k:mean(r[k] for r in rows) for k in rows[0] if k!='seed'},
                        min_joint=min(r['joint'] for r in rows),max_joint=max(r['joint'] for r in rows)))
    (root/'audit.json').write_text(json.dumps(dict(runs=len(files),test_predictions=predictions,aggregates=len(out),
        sha256=files),indent=2),encoding='utf-8')
    return out,per_seed,examples,data


def save_csv(path,rows):
    fields={'model':'模型','task':'任务','layout':'线索布局','gap':'线索后干扰','seed':'种子',
            'single':'单题准确率','paired':'换查询配对','source':'改来源配对','joint':'全部变体正确',
            'frozen':'复用原连边的换查询配对','rewire':'换查询时换边比例','min_joint':'联合最低种子','max_joint':'联合最高种子'}
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(fields.values())
        for row in rows:
            r={**row,'model':NAMES[row['model']],'task':TASK_NAMES[row['task']],'layout':LAYOUT_NAMES[row['layout']]}
            for key in fields.keys()-{'model','task','layout','gap','seed'}:
                if key in r:
                    r[key]=f'{r[key]:.2%}'
            w.writerow([r.get(k,'') for k in fields])


def heatmap(root,p,index,delta):
    font=next((f.name for f in font_manager.fontManager.ttflist if f.name in ('Microsoft YaHei','Noto Sans CJK SC','PingFang SC')),None)
    if font:
        plt.rcParams['font.family']=font
    labels=[m for m in p['models'] if not delta or m!='adaptive']
    fig,axes=plt.subplots(2,2,figsize=(14,8.3),layout='constrained')
    for row,task in enumerate(p['tasks']):
        for col,layout in enumerate(p['layouts']):
            ax=axes[row,col]
            values=[[100*(index[m,task,layout,g]['joint'] if not delta else
                           index['adaptive',task,layout,g]['joint']-index[m,task,layout,g]['joint'])
                     for g in p['gaps']] for m in labels]
            im=ax.imshow(values,cmap='RdBu' if delta else 'Blues',vmin=-100 if delta else 0,vmax=100,aspect='auto')
            for i,rr in enumerate(values):
                for j,val in enumerate(rr):
                    ax.text(j,i,f'{val:+.1f}' if delta else f'{val:.1f}%',ha='center',va='center',
                            color='white' if abs(val)>60 else '#1c293d',fontsize=10)
            ax.set(xticks=range(len(p['gaps'])),xticklabels=p['gaps'],yticks=range(len(labels)),
                   yticklabels=[NAMES[m] if font else m for m in labels],
                   title=f'{TASK_NAMES[task]} · {LAYOUT_NAMES[layout]}' if font else f'{task} · {layout}',
                   xlabel='线索后干扰步数' if font else 'Distractor steps')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.68,pad=.02)
    title = ('自适应版减去各对照 · 联合正确率百分点' if delta else '换查询和逐个改来源都正确 · 联合正确率') if font else ('Adaptive minus comparison, pp' if delta else 'All variants correct')
    fig.suptitle(title,fontsize=16)
    name='delta_heatmap' if delta else 'joint_heatmap'
    for ext in ('png','svg'):
        fig.savefig(root/f'{name}.{ext}',dpi=160)
    plt.close(fig)


def report(root,p,rows,seeds,examples,data):
    index={(r['model'],r['task'],r['layout'],r['gap']):r for r in rows}
    save_csv(root/'scores-zh.csv',rows)
    save_csv(root/'per-seed-zh.csv',seeds)
    for delta in (False,True):
        heatmap(root,p,index,delta)
    top=[]
    for model in p['models']:
        cells=[]
        for task,gap in itertools.product(p['tasks'],(256,1024)):
            r=index[model,task,'scattered',gap]
            cells.append(f'<td><b>{r["joint"]:.1%}</b><small>换查询 {r["paired"]:.1%}</small></td>')
        top.append(f'<tr><th scope="row">{NAMES[model]}</th>{"".join(cells)}</tr>')
    interventions = []
    for task,layout in itertools.product(p['tasks'],p['layouts']):
        r=index['adaptive',task,layout,1024]
        interventions.append(f'<tr><th scope="row">{TASK_NAMES[task]} · {LAYOUT_NAMES[layout]}</th>'
                             f'<td>{r["paired"]:.1%}</td><td>{r["frozen"]:.1%}</td><td>{r["rewire"]:.1%}</td></tr>')
    layout_rows = []
    for model in p['models']:
        cells = []
        for task in p['tasks']:
            compact, scattered = [index[model,task,layout,1024]['joint'] for layout in p['layouts']]
            cells.append(f'<td>{compact:.1%} → {scattered:.1%}<small>{100*(scattered-compact):+.1f} 个百分点</small></td>')
        layout_rows.append(f'<tr><th scope="row">{NAMES[model]}</th>{"".join(cells)}</tr>')
    evidence = {}
    for task in p['tasks']:
        sample = data[task]['scattered']['1024'][0]
        x, xp = [seq for seq,_ in sample['variants'][:2]]
        labels = ([f'查询键 {x[-1]}', f'查询键 {xp[-1]}'] if task == 'binding' else
                  ['加法' if seq[-1] == 0 else '减法' for seq in (x,xp)])
        for seq,_ in sample['variants'][2:]:
            index_changed = next(i for i,(a,b) in enumerate(zip(x,seq,strict=True)) if a != b)
            labels.append(f'改位置 {index_changed} 的值 · {x[index_changed]} → {seq[index_changed]}')
        evidence[task] = dict(spans=sample['spans'],digits=x,models=examples[task],labels=labels)
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="only light"><title>自适应记忆拓扑复测</title><style>
:root{color-scheme:only light;background:#fff;color:#18263b}body{margin:0;font:16px/1.58 system-ui}
main{max-width:1210px;margin:auto;padding:26px;box-sizing:border-box}h1,h2,p,label{color:#18263b}a{color:#15529c}
.wrap{overflow:auto;border:1px solid #cbd5e1;border-radius:9px;margin:10px 0 26px}
table{width:100%;min-width:680px;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;color:#18263b;border-bottom:1px solid #d7e0ec}thead th{background:#edf3fb}
th:first-child{text-align:left;white-space:nowrap}tr:nth-child(even){background:#f7f9fc}.emphasis{background:#e7f5f0}
small{display:block;font-size:.78rem;color:#405678}select{font:inherit;color:#18263b;background:#edf4fb;padding:4px;margin:4px 12px 4px 4px}
img{width:100%;background:#fff}.clue{display:inline-block;background:#e7f4ed;padding:4px 9px;margin:4px;border-radius:5px}.hit{color:#176046}.miss{color:#a42c23}
</style></head><body><main style="color:#18263b;background:#fff;color-scheme:only light">
<h1>记忆拓扑复测</h1><p>六个模型使用各自已训练的权重，同一套新题逐一作答，三个种子。测试把线索后的干扰扩展到 1,024 步，并将完整线索块换序、拆散。相邻的键和值或操作数标记仍保留。</p>
<h2>最难条件一览</h2><p>以下均为拆散线索。大字要求原题、换查询和每个改来源题全部正确，小字只要求原题和换查询都正确。</p>
<div class="wrap"><table><thead><tr><th>模型</th><th>查值 · 256 步</th><th>查值 · 1,024 步</th><th>模 4 · 256 步</th><th>模 4 · 1,024 步</th></tr></thead><tbody>'''+''.join(top)+'''</tbody></table></div>
<h2>只改变线索布局 · 1,024 步</h2><p>同一批线索和干扰尾部，比较原顺序紧凑布局与换序拆散布局。换序与插入线索间噪声同时发生。</p>
<div class="wrap"><table><thead><tr><th>模型</th><th>查值 · 紧凑 → 拆散</th><th>模 4 · 紧凑 → 拆散</th></tr></thead><tbody>'''+''.join(layout_rows)+'''</tbody></table></div>
<h2>图形化成绩表</h2><p>每格是三个种子的联合正确率均值，颜色越深表示越高。</p><img src="joint_heatmap.svg" alt="六种模型、两种线索布局、六档干扰长度的联合正确率热力表">
<h2>进步有多大</h2><p>每格是自适应版减去该行模型的百分点。旧稀疏版、GRU 和 TCN-512 来自 2,000 题训练；三种事件图来自 8,000 题训练。前后三模型对比反映现有系统差距，三种事件图之间才是同训练量的连边对照。</p><img src="delta_heatmap.svg" alt="自适应模型减去其他模型的正确率差值热力表">
<h2>连边干预 · 1,024 步</h2><p>仍用新查询计算边权，只强制复用原查询的边。该表使用换查询成对正确率。</p>
<div class="wrap"><table><thead><tr><th>条件</th><th>正常换边</th><th>强制复用旧边</th><th>实际换边比例</th></tr></thead><tbody>'''+''.join(interventions)+'''</tbody></table></div>
<h2>按指标切换</h2><label>任务<select id="task"><option value="binding">按键查值</option><option value="arithmetic">模 4 加减</option></select></label>
<label>线索布局<select id="layout"><option value="scattered">换序拆散</option><option value="compact">原顺序紧凑</option></select></label>
<label>指标<select id="metric"><option value="joint">全部变体正确</option><option value="paired">换查询配对</option><option value="source">改来源配对</option><option value="single">单题准确率</option></select></label>
<div class="wrap" id="matrix"></div>
<h2>同一题，六种模型怎么答</h2><p>固定展示所选任务的一组 1,024 步拆散线索，使用种子 2。位置从 0 开始，长干扰省略。</p><div id="example"></div>
<p><a href="scores-zh.csv">中文汇总成绩</a> · <a href="per-seed-zh.csv">逐种子成绩</a> · <a href="audit.json">核验记录</a></p>
</main><script>
const rows='''+json.dumps(rows)+''', names='''+json.dumps(NAMES,ensure_ascii=False)+''', samples='''+json.dumps(evidence)+''';
const el=id=>document.getElementById(id), pct=v=>(100*v).toFixed(1)+'%';
function draw(){
 const task=el('task').value, layout=el('layout').value, metric=el('metric').value;
 const ms=Object.entries(names).map(([key,label])=>{const r=rows.filter(x=>x.model===key&&x.task===task&&x.layout===layout);
  return `<tr class="${key==='adaptive'?'emphasis':''}"><th scope="row">${label}</th>${r.map(s=>`<td><b>${pct(s[metric])}</b>${metric==='joint'?`<small>${pct(s.min_joint)}–${pct(s.max_joint)}</small>`:''}</td>`).join('')}</tr>`;}).join('');
 el('matrix').innerHTML=`<table><thead><tr><th>模型</th>${[8,32,128,256,512,1024].map(g=>`<th>${g} 步</th>`).join('')}</tr></thead><tbody>${ms}</tbody></table>`;
 const ex=samples[task], blocks=ex.spans.map(([_,k,v])=>`<span class="clue">位置 ${k}–${v} · ${ex.digits[k]} ${ex.digits[v]}</span>`).join('');
 const e=Object.values(ex.models)[0];
 const er=Object.entries(names).map(([m,label])=>`<tr><th scope="row">${label}</th>${ex.models[m].outputs.map((v,i)=>`<td class="${ex.models[m].hits[i]?'hit':'miss'}">${v} · ${ex.models[m].hits[i]?'正确':'错误'}</td>`).join('')}</tr>`).join('');
 el('example').innerHTML=`<p>${blocks}</p><div class="wrap"><table><thead><tr><th>模型</th>${e.targets.map((y,i)=>`<th>${ex.labels[i]}<small>答案 ${y}</small></th>`).join('')}</tr></thead><tbody>${er}</tbody></table></div>`;
}
for(const id of ['task','layout','metric'])el(id).onchange=draw;draw();
</script></body></html>'''
    (root/'report-cn.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    p=read(args.root/'protocol.json')
    rows,seeds,examples,data=audit(args.root,p)
    report(args.root,p,rows,seeds,examples,data)
    print(f'Verified {len(rows)} aggregate rows and {len(seeds)} seed-level rows.')


if __name__=='__main__':
    main()
