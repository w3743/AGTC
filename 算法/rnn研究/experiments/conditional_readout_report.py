"""Render the arithmetic composition experiment as an offline Chinese dashboard."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

from frontier_data import oracle
from frontier_eval import SEEDS, digest, read
from shared_address_eval import aggregate
from shared_address_report import CSS, heat, mean_range, table


LABELS = dict(conditioned='指令门控', history_gate='证据门控', shared='原读出', original='原共享地址图')
COLORS = dict(conditioned='#00634f', history_gate='#7041ab', shared='#2b60a3', original='#994700')
LAYOUTS = dict(compact='紧凑', scattered='换序与拆散')
PRIMARY = ('scattered',0,2)


def key(row):
    return row['layout'],row['gap'],row['difficulty']


def guess(probs):
    return max(range(4),key=probs.__getitem__)


def interventions(cases,predictions):
    cells = {}
    for case,p in zip(cases,predictions,strict=True):
        base,swapped,*_ = p['probabilities']
        original,updated = p['edges'][0]
        sources = {i+2 for i,t in enumerate(case['variants'][0][:-2]) if t==5}
        selected = [{i for head in edges for i in head} for edges in (original,updated)]
        assert len(sources)==case['difficulty']
        y = case['targets'][1]
        metrics = [
            all(sources<=s for s in selected),
            guess(base)!=guess(swapped),
            guess(swapped)==y,
            guess(p['readout_only'])==y,
            guess(p['old_readout'])==y,
        ]
        if 'evidence_only' in p:
            for source in ('relations','graph'):
                metrics.append(all(guess(prob)==label for prob,label in
                                   zip(p['evidence_only'][source],case['targets'],strict=True)))
        cells.setdefault(key(case),[]).append(metrics)
    return {k:[mean(v) for v in zip(*rows,strict=True)] for k,rows in cells.items()}


def audit(root):
    protocol = read(root/'protocol.json')
    development = read(root/'development.json')
    expected = set(product(protocol['modes'],SEEDS))
    assert {(r['mode'],r['seed']) for r in development}==expected and len(development)==len(expected)
    for seed in SEEDS:
        assert len({r['train_sha256'] for r in development if r['seed']==seed})==1
    for r in development:
        folder = root/f'{r["mode"]}-{r["seed"]}'
        for point in r['checkpoints']:
            assert digest(folder/f'weights-{point["questions"]//protocol["batch"]}.pt')==point['sha256']
    if not (root/'results.json').exists():
        return protocol,development,{},None,False
    runs = read(root/'results.json')
    assert {(r['mode'],r['seed']) for r in runs}==expected and len(runs)==len(expected)
    cases = read(root/'holdout-cases.json')
    prefixes = [tuple(r['variants'][0][:64]) for r in read(root/'development-cases.json')+cases]
    assert len(prefixes)==len(set(prefixes))
    for case in cases:
        base,changed,*edits = case['variants']
        assert base[:-1]==changed[:-1] and base[-1]!=changed[-1]
        assert case['targets']==[oracle(v) for v in case['variants']]
        assert all(sum(a!=b for a,b in zip(base,v,strict=True))==1 for v in edits)
    routes = {}
    example = None
    for r in runs:
        folder = root/f'{r["mode"]}-{r["seed"]}'
        p = read(folder/'predictions.json')
        assert aggregate(cases,p)==r['holdout']
        if r['mode']=='conditioned':
            for k,values in interventions(cases,p).items():
                routes.setdefault(k,[]).append(values)
            if r['seed']==2:
                i = next(i for i,c in enumerate(cases) if key(c)==PRIMARY)
                example = cases[i],p[i]
    return protocol,runs,routes,example,True


def training_chart(runs,modes):
    svg = ['<svg viewBox="0 0 790 265" role="img" aria-label="零步拆散两操作数联合正确率训练曲线">']
    points = [point['questions'] for point in runs[0]['checkpoints']]
    left,right,top,bottom = 65,750,24,210
    for value in (0,.25,.5,.75,1):
        y = bottom-value*(bottom-top)
        svg.append(f'<path d="M{left} {y}H{right}" stroke="#cbd6e0"/>'
                   f'<text class="axis" x="53" y="{y+4}" text-anchor="end">{value:.0%}</text>')
    for q in points:
        x = left+(right-left)*q/points[-1]
        svg.append(f'<text class="axis" x="{x}" y="241" text-anchor="middle">{q:,}</text>')
    for mode in modes:
        color = COLORS[mode]
        coords = []
        for i,q in enumerate(points):
            v = mean(c['joint'] for r in runs if r['mode']==mode
                     for c in r['checkpoints'][i]['development'] if key(c)==PRIMARY)
            coords.append((left+(right-left)*q/points[-1],bottom-v*(bottom-top)))
        coords_text = ' '.join(f'{x},{y}' for x,y in coords)
        svg.append(f'<polyline points="{coords_text}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x,y in coords:
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    legend = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in modes)
    return f'<div class="legend">{legend}</div>' + ''.join(svg) + '</svg>'


def config_label(protocol):
    arities = protocol.get('train_arities',(2,3))
    parts = ['混合两、三操作数训练' if 3 in arities else '只训练两操作数']
    if protocol.get('neutral_routing'):
        parts.append('固定取证查询')
    if protocol.get('binding'):
        parts.append('局部相邻符号相乘')
    if protocol.get('relation_bank'):
        parts.append('相邻关系表')
    if protocol.get('ranking'):
        parts.append('配对答案排序')
    elif protocol.get('contrastive'):
        parts.append('固定证据的指令差异损失')
    else:
        parts.append('交叉熵')
    return '、'.join(parts)


def trial_table(root, previous):
    rows = []
    for i,folder in enumerate([*previous,root],1):
        protocol = read(folder/'protocol.json')
        runs = read(folder/'development.json')
        values = []
        for mode in ('conditioned','history_gate','shared','original'):
            scores = [c['joint'] for r in runs if r['mode']==mode
                      for c in r['checkpoints'][-1]['development'] if key(c)==PRIMARY]
            values.append(mean_range(scores) if scores else ('—',''))
        rows.append([(f'{i}. {config_label(protocol)}',''),*values])
    return '<h2>迭代记录 · 零步拆散两操作数开发集</h2>' + table(
        ('配置','指令门控','证据门控','原读出','原共享地址图'),rows)


def render(root, previous):
    protocol,runs,routes,example,heldout = audit(root)
    section = 'holdout' if heldout else 'development'
    modes = protocol['modes']

    def cells(mode,condition):
        return [c['joint'] for r in runs if r['mode']==mode
                for c in (r['holdout'] if heldout else r['checkpoints'][-1]['development']) if key(c)==condition]

    conditions = list(product((2,3),LAYOUTS,protocol['holdout_gaps']))
    label = lambda n,l,g: f'{n} 个操作数 · {LAYOUTS[l]} · {g} 步'
    summary = []
    for n,l,g in conditions:
        values = [cells(m,(l,g,n)) for m in modes]
        difference = mean(values[0])-max(mean(v) for v in values[1:])
        summary.append([(label(n,l,g),'')]+[mean_range(v) for v in values]+[(f'{100*difference:+.1f}','')])
    primary = cells('conditioned',PRIMARY)
    cards = [
        ('零步拆散 · 两操作数',f'{min(primary):.1%}','三个种子的最低联合正确率，门槛 85%'),
        ('768 步拆散 · 两操作数',f'{mean(cells("conditioned",("scattered",768,2))):.1%}','三个种子的平均联合正确率'),
        ('零步拆散 · 三操作数',f'{mean(cells("conditioned",("scattered",0,3))):.1%}','三个种子的平均联合正确率'),
    ]
    cards_html = ''.join(f'<div class="card">{t}<strong>{v}</strong>{d}</div>' for t,v,d in cards)
    audit_html = ''
    if heldout:
        audit_rows = [[(label(n,l,g),'')]+[mean_range(values) for values in list(zip(*routes[l,g,n],strict=True))[:5]]
                      for n,l,g in conditions]
        case,p = example
        intervention_headers = ('条件','两题均覆盖全部操作数','换指令改变预测','正常换指令正确',
                                '仅换读出指令正确','读出沿用旧指令正确')
        samples = [(name,guess(prob)) for name,prob in (
            ('原题',p['probabilities'][0]),('换指令',p['probabilities'][1]),
            ('仅换读出指令',p['readout_only']),('读出沿用旧指令',p['old_readout']))]
        audit_html = f'''<h2>固定读取证据，只换读出指令</h2>
        <p>仅换读出指令固定原题的读取向量。沿用旧指令固定换指令题的读取向量，再把读出层指令换回去。
        所有“正确”列均用换指令题的真实标签评分。覆盖只检查活跃边的位置。</p>
        {table(intervention_headers,audit_rows)}
        <p>种子 2 的首个零步拆散题组，原答案 {case["targets"][0]}，换指令后答案 {case["targets"][1]}。</p>
        {table(('同题干预','预测'),[[(name,''),(answer,'')] for name,answer in samples])}'''
        if protocol.get('relation_bank'):
            rows = []
            for n,l,g in conditions:
                ablations = list(zip(*routes[l,g,n],strict=True))[5:]
                assert len(ablations)==2
                rows.append([(label(n,l,g),''),mean_range(cells('conditioned',(l,g,n))),
                             *[mean_range(v) for v in ablations]])
            audit_html += f'''<h2>收益来自哪份证据？</h2>
            <p>固定同一组权重，分别将关系表或稀疏图的读取向量置零。以下按完整题组评分。置零改变了训练时的输入分布，因此同时给出从头训练的对照。</p>
            {table(('条件','两条路径','仅关系表','仅稀疏图'),rows)}'''
    data_name = '独立留出集' if section=='holdout' else '开发集'
    budget = protocol['steps']*protocol['batch']
    seed_rows = []
    for mode in modes:
        for r in runs:
            if r['mode']!=mode:
                continue
            scores = r['holdout'] if heldout else r['checkpoints'][-1]['development']
            by_key = {key(c):c['joint'] for c in scores}
            seed_rows.append([(LABELS[mode],''),(r['seed'],''),(f'{r["parameters"]:,}',''),
                              *[heat(by_key['scattered',g,2]) for g in (0,384,768)]])
    model_details = ''
    if protocol.get('relation_bank'):
        model_details = (
            '<p>相邻关系表按前一符号归并后继符号的分布，用可学习的投影读取。'
            '它给三个同构版本提供共同证据，同时保留稀疏图的读取作为残差。原共享地址图按原路径取证。</p>'
        )
        if heldout:
            _,_,_,_,_,relations_only,graph_only = [mean(v) for v in zip(*routes[PRIMARY],strict=True)]
            model_details += (f'<p><b>零步拆散的证据消融</b>　只保留关系表为 {relations_only:.1%}，'
                              f'只保留稀疏图为 {graph_only:.1%}。两者均使用同一组训练后的权重。</p>')
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>指令条件化组合评测</title>
    <style>{CSS}</style><main><p class="eyebrow">神经网络研究 · 组合诊断</p><h1>读到了操作数，能按指令算吗？</h1>
    <p>{data_name}。同题固定输入记录，分别切换加减指令和逐个替换操作数。所有变体答对才计为联合正确。</p>
    <div class="cards">{cards_html}</div>
    <p>各组使用相同训练批次。指令门控让查询直接调节读取向量，证据门控让读取向量自行调节。
    原读出组用常规拼接读出。每个种子训练 {budget:,} 题、{protocol["steps"]:,} 次更新，
    批量 {protocol["batch"]}，AdamW，学习率 {protocol["learning_rate"]}。</p>
    <p>本轮配置为{config_label(protocol)}。{'三操作数只用于检验跨操作数外推。' if protocol.get('train_arities')==[2] else ''}</p>
    {model_details}
    <h2>按操作数、布局和延迟拆分</h2>
    <p>每格第一行为三种子均值，第二行为最小至最大。差值单位为百分点。</p>
    {table(('条件',*(LABELS[m] for m in modes),'比最佳对照'),summary)}
    <h2>拆散两操作数 · 逐种子成绩</h2>
    {table(('模型','种子','可训练参数','0 步','384 步','768 步'),seed_rows)}
    <h2>训练题量曲线</h2><p>只画零步拆散两操作数开发题。横轴为累计训练题数。</p>
    {training_chart(runs,modes)}{trial_table(root,previous)}{audit_html}
    <p class="muted">本轮保留完整历史，结论适用于给定题库的组合与长度外推。固定容量和资源效率仍待验证。</p>
    </main></html>'''
    (root/'report-cn.html').write_text(doc,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--previous',nargs='+',type=Path,default=[])
    args = parser.parse_args()
    render(args.output,args.previous)
