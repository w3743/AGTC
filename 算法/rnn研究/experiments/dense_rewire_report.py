"""Chinese offline report for topology evolution and causal edge cuts."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import torch

from dense_rewire import MODES, QUESTIONS, paired_questions, scores
from frontier_eval import SEEDS, read
from shared_address_report import CSS, mean_range, table


LABELS = dict(dense='始终全连接',random='随机断边重连',adaptive='学习断边重连')
COLORS = dict(dense='#2b60a3',random='#994700',adaptive='#00634f')
TASKS = dict(A='只问甲',B='只问乙',both='判断甲乙是否相同')


def metrics(row,source):
    original = row['test']
    result = row['interventions'][source]
    other = 'B' if source=='A' else 'A'
    drop = [original[q]['joint']-result['scores'][q]['joint'] for q in QUESTIONS]
    selectivity = drop[QUESTIONS.index(source)]-drop[QUESTIONS.index(other)]
    control = mean(sample['scores'][other]['joint']-sample['scores'][source]['joint']
                   + original[source]['joint']-original[other]['joint'] for sample in result['random'])
    return [sum(map(len,result['edges'].values())),*drop,selectivity,control]


def audit(root):
    protocol = read(root/'protocol.json')
    runs = read(root/'results.json')
    assert {(r['mode'],r['seed']) for r in runs}==set(product(MODES,SEEDS)) and len(runs)==9
    for seed in SEEDS:
        same = [r for r in runs if r['seed']==seed]
        for field in ('init_sha256','train_sha256'):
            assert len({r[field] for r in same})==1
        sparse = [r for r in same if r['mode']!='dense']
        assert [p['active'] for p in sparse[0]['timeline']]==[p['active'] for p in sparse[1]['timeline']]
    test = paired_questions(1,protocol['groups'])
    topology = {}
    for r in runs:
        name,seed = r['mode'],r['seed']
        folder = root/f'{name}-{seed}'
        raw = read(folder/'predictions.json')
        assert raw['targets']=={name:y.tolist() for name,(_,y) in test.items()}
        predictions = {q:torch.tensor(p) for q,p in raw['baseline'].items()}
        assert scores(predictions,test)==r['test']
        for q in ('A','B'):
            actual = {k:torch.tensor(p) for k,p in raw['cuts'][q].items()}
            assert scores(actual,test)==r['interventions'][q]['scores']
        active = [set(range(n)) for n in r['total']]
        changes = read(folder/'rewiring.json')
        if name=='dense':
            assert not changes
        snapshots = {}
        dropped,regrown = 0,0
        change_by_step = {event['step']:event['layers'] for event in changes}
        for step in range(protocol['steps']+1):
            for layer,change in enumerate(change_by_step.get(step,[])):
                gone,born = set(change['dropped']),set(change['regrown'])
                assert gone<=active[layer] and not born & active[layer]
                active[layer].difference_update(gone)
                active[layer].update(born)
                assert len(active[layer])==change['active']
                dropped += len(gone)
                regrown += len(born)
            if step not in {c['step'] for c in r['timeline']}:
                continue
            weights = torch.load(folder/f'weights-{step}.pt',weights_only=True)
            masks = [weights[f'layers.{i}.mask'] for i in range(len(active))]
            for mask,indices in zip(masks,active,strict=True):
                assert set(torch.where(mask.flatten())[0].tolist())==indices
            if seed==2:
                snapshots[step] = masks
        for q in ('A','B'):
            ids = r['interventions'][q]['edges']
            assert set(ids)=={'1','2'}
            assert all(set(v)<=active[int(i)] for i,v in ids.items())
        r['dropped'] = dropped
        r['regrown'] = regrown
        r['events'] = [dict(step=event['step'],drop=sum(len(x['dropped']) for x in event['layers']),
                            grow=sum(len(x['regrown']) for x in event['layers'])) for event in changes]
        if seed==2:
            topology[name] = snapshots
    return protocol,runs,topology


def plot(runs,protocol,density=False):
    steps = [c['step'] for c in runs[0]['timeline']]
    width,height,left,right,top,bottom = 940,290,66,902,24,244
    x = lambda step:left+(right-left)*step/protocol['steps']
    y = lambda value:bottom-(bottom-top)*value
    title = '活跃连接比例' if density else '双证据开发集联合正确率'
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">']
    for fraction in (0,.25,.5,.75,1):
        yy = y(fraction)
        svg.append(f'<path d="M{left} {yy}H{right}" stroke="#ced8e3"/>'
                   f'<text class="axis" x="55" y="{yy+4}" text-anchor="end">{fraction:.0%}</text>')
    for step in steps:
        px = x(step)
        svg.append(f'<text class="axis" x="{px}" y="271" text-anchor="middle">{step:,}</text>')
    for step in (protocol['warmup'],protocol['last_rewire']):
        px = x(step)
        svg.append(f'<path d="M{px} {top}V{bottom}" stroke="#66778a" stroke-dasharray="4 4"/>')
    for mode in (('dense','adaptive') if density else MODES):
        chosen = [r for r in runs if r['mode']==mode]
        series = []
        for i,step in enumerate(steps):
            value = mean(sum(r['timeline'][i]['active'])/sum(r['total']) if density
                         else r['timeline'][i]['development']['both']['joint'] for r in chosen)
            series.append((x(step),y(value)))
        points = ' '.join(f'{px},{py}' for px,py in series)
        color = COLORS[mode]
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for px,py in series:
            svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
    return ''.join(svg)+'</svg>'


def mask_panels(topology,protocol):
    stages = (0,protocol['target_step'],protocol['steps'])
    svg = ['<svg viewBox="0 0 790 550" role="img" aria-label="两组真实连接掩码的演化快照">']
    for col,step in enumerate(stages):
        svg.append(f'<text class="axis" x="{217+235*col}" y="24" text-anchor="middle">更新 {step:,} 次</text>')
    for row,mode in enumerate(('adaptive','random')):
        svg.append(f'<text class="axis" x="12" y="{54+252*row}">{LABELS[mode]}</text>')
        for col,step in enumerate(stages):
            mask = topology[mode][step][1]
            origin_x,origin_y = 120+235*col,48+252*row
            groups = 24
            block = mask.reshape(groups,4,groups,4).float().mean((1,3))
            for i in range(groups):
                for j in range(groups):
                    density = float(block[i,j])
                    shade = tuple(round(a+(b-a)*density) for a,b in zip((241,246,252),(0,83,65),strict=True))
                    color = '#%02x%02x%02x'%shade
                    svg.append(f'<rect x="{origin_x+8*j}" y="{origin_y+8*i}" width="8" height="8" fill="{color}"/>')
    return ''.join(svg)+'</svg>'


def cut_graph(model_row):
    choices = {s:{int(i):set(v) for i,v in model_row['interventions'][s]['edges'].items()}
               for s in ('A','B')}
    cols = (95,450,805)
    colors = {'A':'#00634f','B':'#7540a9','both':'#9a4600'}
    y = lambda node:53+node*5
    svg = ['<svg viewBox="0 0 900 552" role="img" aria-label="甲乙断边干预用到的真实隐藏连接">']
    for i in (1,2):
        a,b = choices['A'][i],choices['B'][i]
        for index in sorted(a|b):
            dst,src = divmod(index,96)
            group = 'both' if index in a and index in b else 'A' if index in a else 'B'
            svg.append(f'<path d="M{cols[i-1]} {y(src)} L{cols[i]} {y(dst)}" stroke="{colors[group]}" stroke-width=".95" opacity=".52"/>')
    for i,x in enumerate(cols,1):
        svg.append(f'<text class="axis" x="{x}" y="27" text-anchor="middle">隐藏层 {i}</text>')
        for neuron in range(96):
            svg.append(f'<circle cx="{x}" cy="{y(neuron)}" r="1.8" fill="#253c56"/>')
    return ''.join(svg)+'</svg>'


def turnover_chart(row):
    events = row['events']
    svg = ['<svg viewBox="0 0 900 230" role="img" aria-label="实际断边和重连次数">']
    maximum = max(event['drop'] for event in events)
    for value in (0,maximum//2,maximum):
        y = 175-140*value/maximum
        svg.append(f'<path d="M65 {y}H865" stroke="#ced8e3"/>'
                   f'<text class="axis" x="54" y="{y+4}" text-anchor="end">{value:,}</text>')
    for i,event in enumerate(events):
        x = 67+i*31
        for dx,key,color in ((0,'drop','#995100'),(11,'grow','#00634f')):
            height = 140*event[key]/maximum
            svg.append(f'<rect x="{x+dx}" y="{175-height}" width="10" height="{height}" fill="{color}"/>')
        if i%4==0 or i==len(events)-1:
            svg.append(f'<text class="axis" x="{x+10}" y="199" text-anchor="middle">{event["step"]:,}</text>')
    return ('<div class="legend"><span style="color:#995100">● 断边</span>'
            '<span style="color:#00634f">● 重新启用</span></div>'+''.join(svg)+'</svg>')


def controls_table(root,runs,protocol):
    path = root/'controls/results.json'
    if not path.exists():
        return ''
    controls = read(path)
    names = dict(prune_only='只按权重剪枝',reset_adaptive='固定学习重连掩码，重新初始化',
                 reset_random='固定随机掩码，重新初始化')
    assert {(r['mode'],r['seed']) for r in controls}==set(product(names,SEEDS)) and len(controls)==9
    test = {i:paired_questions(i,protocol['groups']) for i in (1,2)}
    for row in controls:
        mode,seed = row['mode'],row['seed']
        source = next(r for r in runs if r['seed']==seed and r['mode']=='adaptive')
        assert row['active']==source['timeline'][-1]['active']
        assert row['train_sha256']==source['train_sha256']
        folder = root/f'controls/{mode}-{seed}'
        raw = read(folder/'predictions.json')
        data = test[1 if mode=='prune_only' else 2]
        assert raw['targets']=={k:y.tolist() for k,(_,y) in data.items()}
        assert scores({q:torch.tensor(p) for q,p in raw['baseline'].items()},data)==row['test']
        if mode.startswith('reset_'):
            other = mode.removeprefix('reset_')
            original = torch.load(root/f'{other}-{seed}/weights-{protocol["steps"]}.pt',weights_only=True)
            reset = torch.load(folder/'weights.pt',weights_only=True)
            assert all(torch.equal(v,reset[k]) for k,v in original.items() if k.endswith('.mask'))
    rows = [[(label,''),('主留出集' if mode=='prune_only' else '另一批独立布局',''),
             *[mean_range([r['test'][q]['joint'] for r in controls if r['mode']==mode]) for q in QUESTIONS]]
            for mode,label in names.items()]
    return f'''<h2>重连与结构各自贡献多少？</h2>
    <p>只剪枝组从相同稠密权重起步，按同一密度曲线下降并继续训练。后两组只复用最后的连接掩码，
    给两组相同的新随机权重和训练预算，训练时固定连接，再测另一批训练未见的布局。</p>
    {table(('补充对照','测试集','只问甲','只问乙','双证据'),rows)}'''


def pp(values):
    return f'{100*mean(values):+.1f}'


def render(root,previous):
    protocol,runs,topology = audit(root)
    by_mode = {m:[r for r in runs if r['mode']==m] for m in MODES}
    main_rows = []
    cuts = []
    for mode,group in by_mode.items():
        main_rows.append([(LABELS[mode],''),(sum(group[0]['timeline'][-1]['active']),''),
                         *[mean_range([r['test'][q]['joint'] for r in group]) for q in QUESTIONS]])
        for source in ('A','B'):
            vals = list(zip(*(metrics(r,source) for r in group),strict=True))
            cuts.append([(f'{LABELS[mode]} · 切{"甲" if source=="A" else "乙"}',''),
                         (f'{min(vals[0])}–{max(vals[0])}',''),
                         *[(pp(v),'') for v in vals[1:]]])
    seed_rows = [[(LABELS[r['mode']],''),(r['seed'],''),*[mean_range([r['test'][q]['joint']]) for q in QUESTIONS]]
                 for r in runs]
    prior_html = ''
    if previous:
        old = read(previous/'results.json')
        old_protocol = read(previous/'protocol.json')
        old_rows = [[(LABELS[m],''),mean_range([r['test']['both']['joint'] for r in old if r['mode']==m]),
                    mean_range([r['test']['both']['joint'] for r in by_mode[m]])] for m in MODES]
        prior_html = (f'<h2>训练成熟后再断边</h2><p>首轮在 {old_protocol["warmup"]} 次更新后开始稀疏化。'
                      f'本轮在 {protocol["warmup"]} 次更新后开始，三组同步增加全连接训练，'
                      '随后沿用相同的断边、重连和恢复长度。</p>'
                      +table(('模型','首轮双证据','本轮双证据'),old_rows))
    selected = next(r for r in by_mode['adaptive'] if r['seed']==2)
    paired = lambda m:mean(r['test']['both']['joint'] for r in by_mode[m])
    cards = ''.join(f'<div class="card">{title}<strong>{value}</strong>{detail}</div>' for title,value,detail in (
        ('学习重连 · 双证据',f'{paired("adaptive"):.1%}','三个种子的留出集联合正确率'),
        ('比同稀疏度随机组',f'{100*(paired("adaptive")-paired("random")):+.1f} 点','联合正确率差，单位为百分点'),
        ('种子 2 实际重连',f'{selected["regrown"]:,} 次','全程启用原先不活跃连接的次数'),
    ))
    legend = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in MODES)
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>从全连接学出连接结构</title>
    <style>{CSS}</style><main><p class="eyebrow">神经网络研究 · 连接演化</p><h1>全连接网络能学出怎样的分工？</h1>
    <div class="cards">{cards}</div>
    <p>所有组都从相同的普通全连接 ReLU 网络起步。输入是各自打乱的甲、乙记录和一个最终问题。网络可被问甲的值、乙的值，或两者是否相同。模型接收整个平铺输入，没有注意力头、记忆槽或任务路由模块。</p>
    <p>每组 {protocol["steps"]*protocol["batch"]:,} 条训练序列，{protocol["steps"]:,} 次更新。三个种子使用同一留出集，每类 {protocol["groups"]} 组。联合正确要求原题、改甲、改乙、改无关记录四题都答对。甲乙两组记录在输入中各有固定位置，组内顺序打乱，留出集的顺序组合独立。</p>
    <h2>留出成绩</h2><p>第一行为三种子均值，第二行为最小至最大。活跃连接数不含偏置。</p>
    {table(('训练方式','最终活跃连接','只问甲','只问乙','双证据'),main_rows)}
    <details><summary>逐种子成绩</summary>{table(('模型','种子','只问甲','只问乙','双证据'),seed_rows)}</details>
    {prior_html}
    {controls_table(root,runs,protocol)}
    <h2>连接怎样变化</h2><p>第一图是双证据开发集曲线，第二图是活跃连接比例。横轴为更新次数。两条竖虚线标记稀疏化开始和最后一次重连。随机组与学习组共享同一密度曲线。</p>
    <div class="legend">{legend}</div>{plot(runs,protocol)}{plot(runs,protocol,True)}
    <h3>每次实际断边和恢复 · 学习重连组 · 种子 2</h3>{turnover_chart(selected)}
    <h3>第一、二隐藏层之间的真实掩码</h3><p>种子 2。每个色块汇总相邻 4 × 4 个可能连接，颜色越深表示保留比例越高。各快照从记录的断边重连事件逐步重放核对。</p>
    {mask_panels(topology,protocol)}
    <h2>切隐藏边会伤到哪份证据？</h2><p>用开发集找甲或乙敏感的隐藏层间连接，冻结切法后在留出集测试。以下为联合正确率降幅，单位为百分点。选择性等于目标降幅减去另一来源降幅。随机断边匹配被切的层和条数，重复 12 次。</p>
    {table(('模型与干预','切断边数','甲降幅','乙降幅','双证据降幅','选择性','随机切选择性'),cuts)}
    <h3>实际切断的隐藏边 · 学习重连组 · 种子 2</h3><p>每列 96 个普通神经元。绿色为甲敏感切法，紫色为乙敏感切法，棕色为两者共有。显示的是开发集选中的干预边。</p>
    {cut_graph(selected)}
    <p class="muted">本实验允许层间连接断开和恢复。隐藏层的层数、顺序和候选连接由实验固定。实现以掩码模拟连接开关，仍用稠密矩阵算子运行。</p>
    </main></html>'''
    (root/'report-cn.html').write_text(doc,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--previous',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    render(args.output,args.previous)
