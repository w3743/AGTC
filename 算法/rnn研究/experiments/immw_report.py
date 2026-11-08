"""Audit IMMW supervision and plot the independent difficulty axes."""

import argparse
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import mean

from frontier_eval import SEEDS, read, write
from immw import CAPACITIES, GAPS, NOVEL_MASKS, RATES, SKILLS, TRAIN_MASKS, cases, suffix_ceiling
from shared_address_report import CSS, heat, mean_range, table


NAMES = dict(checkpoint='旧检查点',operation='累积操作',live='最近状态',rule='规则',interruption='切换事件',
             irrelevant='无关任务')


def reference(tokens):
    """Independent replay using separate stacks of each task's observed events."""
    events = {}
    for token in tokens[:-5]:
        if 1<=token<=8:
            task = token
            events.setdefault(task,[])
        elif 10<=token<=23:
            events[task].append(token)
    values = events[task]
    old = next(t-10 for t in values if 10<=t<=13)
    operation = sum(t-14 for t in values if 14<=t<=17)%4
    live = next(t-18 for t in reversed(values) if 18<=t<=21)
    rule = next(t-22 for t in reversed(values) if 22<=t<=23)
    flags = [t==29 for t in tokens[-5:-1]]
    result = sum(v for v,f in zip((old,operation,live),flags[:3],strict=True) if f)%4
    return (1-result)%4 if rule and flags[3] else result


def run_audit(count):
    results = []
    example = None
    assert not set(TRAIN_MASKS)&set(NOVEL_MASKS)
    assert all({(mask>>i)&1 for mask in TRAIN_MASKS}=={0,1} for i in range(4))
    for novel,gap,rate,capacity in product((False,True),(*GAPS,12000),RATES,CAPACITIES):
        group = []
        for f,identity,world,q in cases('holdout',(gap,),(rate,),(capacity,),count,novel):
            phase = world.history[world.gap_start:world.gap_end]
            assert len(phase)==f.gap
            switches = sum(1<=t<=8 for t in phase)
            assert switches==world.interruptions==round(f.gap*f.rate/256)
            assert len({t for t in world.history[:world.gap_start] if 1<=t<=8})==capacity
            assert sum(14<=t<=17 for t in phase)==max(1,gap//96)
            assert sum(22<=t<=23 for t in phase)==gap//512
            base = q['variants'][0]
            assert q['ages'][0]>=gap and q['ages'][2]<2*capacity+6
            for token_input,target in zip(q['variants'],q['targets'],strict=True):
                assert reference(token_input)==target
            for variant,pos in zip(q['variants'][1:],q['edits'],strict=True):
                assert len(base)==len(variant) and base[pos]!=variant[pos]
                assert base[:pos]==variant[:pos] and base[pos+1:]==variant[pos+1:]
            assert all(t!=q['targets'][0] for t in q['targets'][1:-1]) and q['targets'][-1]==q['targets'][0]
            group.append(q)
            if example is None and (novel,gap,rate,capacity)==(True,2048,16,4):
                example = dict(identity=identity,starts=world.gap_start,ends=world.gap_end,
                               target=world.target,interruptions=world.interruptions,
                               history=world.history,sources=world.sources,question=q)
        results.append(dict(novel=novel,gap=gap,rate=rate,capacity=capacity,groups=len(group),
                            hashes=[sha256(bytes(q['variants'][0])).hexdigest() for q in group],
                            suffix32=mean(suffix_ceiling(q,32) for q in group),
                            suffix256=mean(suffix_ceiling(q,256) for q in group),
                            query_only=mean(suffix_ceiling(q,5) for q in group),
                            switch_pairs=sum('interruption' in q['edit_kinds'] for q in group),
                            old_age=[min(q['ages'][0] for q in group),max(q['ages'][0] for q in group)],
                            recent_age=[min(q['ages'][2] for q in group),max(q['ages'][2] for q in group)]))
    return dict(cells=results,example=example)


def timeline(example):
    seq = example['history']
    begin,end = example['starts'],example['ends']
    x = lambda pos:70+790*pos/len(seq)
    svg = ['<svg viewBox="0 0 910 215" role="img" aria-label="一题的任务中断与恢复时序">']
    for start,stop,title,color in ((0,begin,'初始化','#ebf1f9'),(begin,end,'记忆间隔与中断','#eef8f5'),
                                   (end,len(seq),'恢复','#ebf1f9')):
        svg.append(f'<rect x="{x(start)}" y="38" width="{max(1,x(stop)-x(start))}" height="108" fill="{color}"/>')
        if stop-start>40:
            svg.append(f'<text class="axis" x="{x((start+stop)/2)}" y="64" text-anchor="middle">{title}</text>')
    for pos,token in enumerate(seq):
        if 1<=token<=8:
            color = '#00634f' if token==example['target']+1 else '#72859c'
            svg.append(f'<path d="M{x(pos)} 95V143" stroke="{color}"/>')
        elif 14<=token<=17:
            svg.append(f'<circle cx="{x(pos)}" cy="82" r="2" fill="#7540a9"/>')
    for i,pos in enumerate(example['sources']):
        svg.append(f'<circle cx="{x(pos)}" cy="{153+i*11}" r="4" fill="#a34700"/>')
    svg.append('<text class="axis" x="70" y="205">旧信息写入</text>'
               '<text class="axis" x="860" y="205" text-anchor="end">恢复目标任务并提问</text>')
    return ''.join(svg)+'</svg>'


def model_table(root):
    path = root/'development.json'
    if not path.exists():
        return ''
    runs = read(path)
    if len(runs)!=6:
        return ''
    for seed in SEEDS:
        assert len({r['train_sha256'] for r in runs if r['seed']==seed})==1
    rows = []
    for novel,gap in product((False,True),(16,32,64,128,256)):
        values = [[c['joint'] for r in runs if r['mode']==m for c in r['history'][-1]['scores']
                   if (c['novel'],c['gap'])==(novel,gap)] for m in ('dense','GRU')]
        rows.append([('未见组合' if novel else '已见组合',''),(gap,''),*[mean_range(v) for v in values]])
    return ('<h2>512 神经元全连接 RNN 与参数量近似的 GRU</h2>'
            '<p>开发筛选，训练只用 16、32、64 步，均为全序列反传。正式能力留出榜暂未开测。</p>'
            +table(('组合','记忆间隔','全连接 RNN','GRU'),rows))


def render(root,summary):
    example = summary['example']
    rows = []
    for gap,rate in product((*GAPS,12000),RATES):
        cells = [r for r in summary['cells'] if r['novel'] and r['gap']==gap and r['rate']==rate]
        cells.sort(key=lambda r:r['capacity'])
        rows.append([(gap,''),(rate,''),*[heat(r['suffix32']) for r in cells],
                     (sum(r['switch_pairs'] for r in cells),'')])
    q = example['question']
    edits = [[(NAMES[label],''),(pos+1,''),(q['targets'][i+1],'')]
             for i,(label,pos) in enumerate(zip(q['edit_kinds'],q['edits'],strict=True))]
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>IMMW 任务诊断</title><style>{CSS}</style>
    <main><p class="eyebrow">自演化稀疏 RNN · IMMW</p><h1>中断之后，还记得哪一件事？</h1>
    <p>首版是低维事件流。模型接收当前发生的事件和最终问题，任务状态需要从历史重建。
    三个变量独立变化：旧信息保留多久、期间切换多频繁、同时维护多少个任务状态。</p>
    <p>每个并存任务有旧检查点、累计操作、最近状态和规则。任务切换决定接下来的操作归属。
    训练只使用单项和两项组合，开发与留出另含三项和四项组合。</p>
    <h2>最近 32 步能解释多少个未见组合题组？</h2>
    <p>只要同一题组的两个输入在最近 32 步完全相同、答案却不同，任何只读该窗口的确定性模型都无法两题全对。
    表中是逐题组上界。切换反事实固定所有操作，只改变一个切换事件。</p>
    {table(('记忆间隔','每 256 步切换次数','并存 2 项','并存 4 项','并存 8 项','切换配对组数'),rows)}
    <h2>一题的中断与恢复</h2><p>2,048 步，四个并存任务。绿线为切回目标任务，灰线为切到其他任务，紫点为操作事件。
    橙点为四种证据最后一次写入的位置。原答案为 {q['targets'][0]}。</p>{timeline(example)}
    {table(('单处改动','事件位置','改后答案'),edits)}
    {model_table(root)}
    <p class="muted">这里的操作与位置由符号事件表示。交互式导航、可复用模块干预和跨公开基准复验属于后续阶段。</p>
    </main></html>'''
    (root/'report-cn.html').write_text(doc,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--count',type=int,default=24)
    args = parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    path = args.root/'task-audit.json'
    if path.exists():
        summary = read(path)
    else:
        summary = run_audit(args.count)
        write(path,summary)
    render(args.root,summary)
    print(len(summary['cells']),sum(c['groups'] for c in summary['cells']),
          sum(c['switch_pairs'] for c in summary['cells']))


if __name__=='__main__':
    main()
