"""Chinese offline dashboard for development and sealed frontier results."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

from frontier_eval import read
from frontier_models import STANDARDS


NAMES={'adaptive':'自适应槽图','GRU':'GRU','LSTM':'LSTM','TCN':'全感受野 TCN','Transformer':'全序列 Transformer',
       'event_dense':'全历史事件图','slot_dense':'同槽全连接读取','slot_fixed':'同槽固定读取'}
TASKS={'lookup':'直接查值','latest':'查最新值','pointer':'多跳查值','arithmetic':'模 4 运算'}


def pct(value): return f'{value:.2%}'


def heatmap(root,rows,models):
    gaps=sorted({r['gap'] for r in rows})
    buckets=[(task,gap) for task,gap in itertools.product(TASKS,gaps)]
    aggregate={(n,t,g):mean(r['joint'] for r in rows if (r['name'],r['task'],r['gap'])==(n,t,g))
               for n in models for t,g in buckets}
    fig,axs=plt.subplots(2,1,figsize=(12,8.2),layout='constrained')
    metrics=[[100*aggregate[n,t,g] for t,g in buckets] for n in models]
    values=[[100*(aggregate['adaptive',t,g]-max(aggregate[n,t,g] for n in STANDARDS)) for t,g in buckets]]
    for ax,matrix,labels,delta in zip(axs,[metrics,values],[list(map(NAMES.get,models)),['自适应减最强标准']],(False,True)):
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu' if delta else 'Blues',vmin=-100 if delta else 0,vmax=100)
        ax.set(xticks=range(len(buckets)),xticklabels=[f'{TASKS[t]}\n{g} 步' for t,g in buckets],
               yticks=range(len(labels)),yticklabels=labels,title='差值（百分点）' if delta else '联合正确率（%）')
        for i,row in enumerate(matrix):
            for j,val in enumerate(row): ax.text(j,i,f'{val:+.1f}' if delta else f'{val:.1f}',ha='center',va='center',color='white' if abs(val)>60 else '#213247')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(root/'score-heatmap.svg');fig.savefig(root/'score-heatmap.png',dpi=150);plt.close(fig)


def training_plot(root,p):
    if not (root/'milestones.json').exists(): return ''
    rows=read(root/'milestones.json'); budgets=sorted({r['questions'] for r in rows})
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    for name in p['models']:
        series=[mean(r['joint'] for r in rows if (r['name'],r['questions'])==(name,q)) for q in budgets]
        ax.plot(budgets,[100*v for v in series],'o-',label=NAMES[name])
    ax.set(xlabel='累计训练题数',ylabel='联合正确率（%）',xticks=budgets)
    ax.legend(ncol=4);fig.savefig(root/'training-budget.svg');plt.close(fig)
    return '<h2>训练量曲线</h2><p>每格固定取两道验证题，曲线用于定位开始学习的阶段。</p><img src="training-budget.svg" alt="训练题数与联合正确率">'


def resource_plot(root,assessment):
    resources=assessment['resources']
    if not resources: return ''
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    gap=resources[0]['gap']
    for r in resources:
        x=r['latency_ms'];y=100*mean(c['joint'] for c in assessment['cells'] if c['gap']==gap and c['name']==r['name'])
        ax.scatter(x,y,s=40+8*r['peak_increment_mib'],alpha=.70,label=NAMES[r['name']])
        ax.annotate(NAMES[r['name']],(x,y),xytext=(4,5),textcoords='offset points',fontsize=9)
    ax.set(xlabel='整题耗时（毫秒）',ylabel='联合正确率（%）',xscale='log',title=f'{gap} 步 · 点大小代表增量内存')
    fig.savefig(root/'resource-frontier.svg');plt.close(fig)
    cells=''.join(f'<tr><th scope="row">{NAMES[r["name"]]}</th><td>{r["latency_ms"]:.3f}</td>'
                  f'<td>{r["ms_per_token"]:.5f}</td><td>{r["peak_increment_mib"]:.2f}</td>'
                  f'<td>{r["active_read_edges"] if r["active_read_edges"] is not None else "—"}</td></tr>' for r in resources)
    return '<h2>精度与资源</h2><p>单线程 CPU、批量 1。增量内存以装载权重与题目后的常驻内存为基准。</p><img src="resource-frontier.svg" alt="精度和耗时散点图"><div class="wrap"><table><thead><tr><th>模型</th><th>毫秒／题</th><th>毫秒／位</th><th>增量 MiB</th><th>活跃读边</th></tr></thead><tbody>'+cells+'</tbody></table></div>'


def trace_plot(root,p,a):
    if a['split']=='sealed':
        path=root/'predictions-sealed'/f'adaptive_{p["seeds"][0]}.json'
    elif (root/'predictions-development').exists():
        path=root/'predictions-development'/f'adaptive_{p["seeds"][0]}.json'
    else:
        path=root/'runs'/f'adaptive_{p["seeds"][0]}_{a["selection"]["adaptive"]:g}'/'development.json'
    row=next(r for r in read(path) if "'arithmetic'" in r['id'] and "'scattered'" in r['id'])
    fig,axs=plt.subplots(2,1,figsize=(10,4.2),sharex=True,layout='constrained')
    for branch,ax in enumerate(axs):
        matrix=[[int(slot in head) for slot in range(32)] for hop in row['edges'] for head in hop[branch]]
        ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        ax.set(title=('原查询' if not branch else '换查询'),ylabel='跳 × 头',yticks=(0,3,7,11,15))
    axs[-1].set_xlabel('记忆槽编号');fig.savefig(root/'actual-edges.svg');plt.close(fig)
    return '<h2>同题真实连边</h2><p>深色代表该次读取选中了对应槽位。该图显示实际连边，答题正确性以成绩表为准。</p><img src="actual-edges.svg" alt="原查询和改查询的实际槽位连边">'


def diagnostic_panels(root,models):
    sections=[]
    old=root.parent/'diagnosis.json'
    if old.exists():
        data=read(old); cells=[]
        for reorder,spread,title in ((False,False,'原布局'),(True,False,'只换序'),(False,True,'只拉开'),(True,True,'换序加拉开')):
            rows=[r for r in data if (r['task'],r['gap'],r['reorder'],r['spread'])==('arithmetic',1024,reorder,spread)]
            values=[mean(r[k] for r in rows) for k in ('joint','coverage','covered_but_wrong','rewire')]
            cells.append(f'<tr><th scope="row">{title}</th>'+''.join(f'<td>{pct(v)}</td>' for v in values)+'</tr>')
        sections.append('<h2>旧事件图四布局诊断</h2><p>1,024 步模 4。覆盖指实际读边命中值节点或携带该值的相邻节点，覆盖本身无法证明正确解码。</p><div class="wrap"><table><thead><tr><th>布局</th><th>联合正确</th><th>操作数覆盖</th><th>覆盖但答错</th><th>换边比例</th></tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>')
    write_screen=root.parent/'write-retention-screen'/'screen.json'
    if write_screen.exists():
        screen=read(write_screen)
        cells=''.join(f'<tr><th scope="row">{r["seed"]}</th><td>{pct(r["original_joint"])}</td><td>{pct(r["retention_joint"])}</td></tr>' for r in screen['rows'])
        sections.append('<h2>保守写入门筛选</h2><p>相同种子、学习率与 2,000 道训练题。唯一变化是将写入门的初始偏置设为 −3。</p><div class="wrap"><table><thead><tr><th>种子</th><th>原写入门</th><th>保守写入门</th></tr></thead><tbody>'+cells+'</tbody></table></div>')
    rule_screen=root.parent/'short-rule-screen'/'screen.json'
    if rule_screen.exists():
        screen=read(rule_screen); cells=[]
        for name in models:
            vals=[]
            for gap in (0,16,384,768):
                records=[r for job in screen['rows'] if job['name']==name for r in job['rows'] if r['gap']==gap]
                vals.append(f'<td>{pct(mean(r["original"]["joint"] for r in records))} → '
                            f'{pct(mean(r["rule_curriculum"]["joint"] for r in records))}</td>')
            cells.append(f'<tr><th scope="row">{NAMES[name]}</th>'+''.join(vals)+'</tr>')
        sections.append('<h2>零干扰查值专项训练</h2><p>每格为 8,000 题混合训练 → 8,000 题零干扰查值专项训练的联合正确率。两个配置在同一批独立诊断题上测量。</p><div class="wrap"><table><thead><tr><th>模型</th><th>0 步</th><th>16 步</th><th>384 步</th><th>768 步</th></tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>')
    return ''.join(sections)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    p=read(root/'protocol.json')
    path=root/'assessment-sealed.json'
    if not path.exists(): path=root/'assessment-development.json'
    a=read(path); g=a['gates']
    rows=a['cells']
    font=next((f.name for f in font_manager.fontManager.ttflist if f.name=='Microsoft YaHei'),None)
    if font: plt.rcParams['font.family']=font
    heatmap(root,rows,p['models'])
    with (root/'scores-zh.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['模型','任务','布局','干扰步数','难度','单题正确率','联合正确率','种子2','种子20261027','种子20261028','题组数'])
        for r in rows: writer.writerow([NAMES[r['name']],TASKS[r['task']],'拆散' if r['layout']=='scattered' else '紧凑',
            r['gap'],r['difficulty'],pct(r['single']),pct(r['joint']),*map(pct,r['seed_values']),r['cases']])
    cards=[('复杂结构',all(r['passed'] for r in g['complex']), ' · '.join(f'{TASKS[r["task"]]} {100*r["delta"]:+.2f} 点' for r in g['complex'])),
           ('长程保持',all(r['passed'] for r in g['retention']), ' · '.join(f'{r["gap"]} 步 {pct(r["adaptive"])}' for r in g['retention'])),
           ('资源效率',None if g['resource'] is None else g['resource']['passed'],
            '待测' if g['resource'] is None else f'对照 {NAMES[g["resource"]["baseline"]]}'),
           ('拓扑贡献',g['topology']['passed'], f'冻结连边下降 {100*g["topology"]["delta"]:.2f} 点')]
    card=''.join(f'<article><h2>{title}</h2><b>{"通过" if ok else "待测" if ok is None else "未达标"}</b><p>{text}</p></article>' for title,ok,text in cards)
    complex_rows=''.join(f'<tr><th scope="row">{TASKS[r["task"]]}</th><td>{pct(r["adaptive"])}</td>'
                         f'<td>{NAMES[r["baseline"]]} · {pct(r["baseline_score"])}</td><td>{100*r["delta"]:+.2f}</td>'
                         f'<td>{100*r["ci"][0]:+.2f} 至 {100*r["ci"][1]:+.2f}</td></tr>' for r in g['complex'])
    shortcuts=''.join(f'<tr><th>{TASKS[task]}</th><td>{pct(scores["query_only"])}</td>'
                      f'<td>{pct(scores["tail_only"])}</td><td>{pct(scores["ignore_instruction"])}</td></tr>' for task,scores in a['shortcuts'].items())
    single_rows=''.join(f'<tr><th scope="row">{NAMES[name]}</th>'+''.join(
        f'<td>{pct(mean(r["single"] for r in rows if r["name"]==name and r["task"]==task))}</td>' for task in TASKS)
        +'</tr>' for name in p['models'])
    status=('封存测试' if a['split']=='sealed' else '开发集筛选 · 封存集未打开')
    if not a['qualification_ready']: status+=' · 尚未达到全量资格评估'
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="color-scheme" content="only light"><meta name="viewport" content="width=device-width,initial-scale=1"><title>记忆拓扑超越实验</title>
<style>:root{color-scheme:only light;background:#fff;color:#18263b}body{margin:0;font:16px/1.55 system-ui}main{max-width:1200px;margin:auto;padding:24px;box-sizing:border-box;color:#18263b;background:white}h1,h2,p{color:#18263b}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.cards article{padding:12px;background:#edf4fb;border-radius:8px}.cards h2{font-size:18px;margin:0}.wrap{overflow:auto;margin:12px 0 24px}table{border-collapse:collapse;width:100%;min-width:590px;background:white}th,td{padding:8px 10px;color:#18263b;border-bottom:1px solid #d5dfec;text-align:right}th:first-child{text-align:left}thead th{background:#edf3fa}img{width:100%;background:white}a{color:#15519c}</style></head><body><main style="background:#fff;color:#18263b;color-scheme:only light">
<h1>自适应记忆拓扑 · 多维度对照</h1><p>'''+status+f'。每模型三个种子，每次 {p["steps"]*16:,} 道训练题，评测每格 {a["cases_per_cell"]} 组。'+'''</p><div class="cards">'''+card+'''</div>
<h2>与最强标准模型对照</h2><div class="wrap"><table><thead><tr><th>任务</th><th>自适应版</th><th>最强标准基线</th><th>差值（点）</th><th>题组配对 95% 区间</th></tr></thead><tbody>'''+complex_rows+'''</tbody></table></div>
<h2>成绩热力表</h2><img src="score-heatmap.svg" alt="按任务和长度比较八种模型">
<h2>单题诊断</h2><p>单题允许不同变体分别计分，用于检查联合分数为零时是否已经学到部分规则。</p>
<div class="wrap"><table><thead><tr><th>模型</th>'''+''.join(f'<th>{v}</th>' for v in TASKS.values())+'''</tr></thead><tbody>'''+single_rows+'''</tbody></table></div>
<h2>捷径对照</h2><div class="wrap"><table><thead><tr><th>任务</th><th>只读查询</th><th>只读末尾</th><th>忽略指令</th></tr></thead><tbody>'''+shortcuts+'''</tbody></table></div>
'''+training_plot(root,p)+resource_plot(root,a)+trace_plot(root,p,a)+diagnostic_panels(root,p['models'])+'''
<p><a href="scores-zh.csv">中文成绩表</a> · <a href="'''+path.name+'''">核验及达标线</a></p></main></body></html>'''
    (root/'report-cn.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    main()
