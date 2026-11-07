"""Audit the paired-cue topology experiment and render its Chinese report."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import torch

from evolving_rnn_eval import MODES, WIDTH, cells
from evolving_rnn_two_cues import DEV_GAPS, TEST_GAPS, TRAIN_GAPS, data_sets
from frontier_eval import SEEDS, digest, read
from shared_address_report import CSS, mean_range, table


RECIPES = dict(answers='只用查询答案', **{'joint-state':'训练时辅助读出两条线索'},compute='重复查询三步')
LABELS = dict(dense='始终全连接',fixed='同密度固定随机',decay='只衰减',evolve='衰减与再生',
              **{'evolve-frozen':'同一演化权重，冻结拓扑'})
COLORS = dict(dense='#2b60a3',fixed='#7041ab',decay='#994700',evolve='#00634f')


def audit_development(root,protocol):
    rows = read(root/'development.json')
    assert {(r['recipe'],r['seed']) for r in rows}==set(product(protocol['recipes'],SEEDS))
    for seed in SEEDS:
        assert len({r['train_sha256'] for r in rows if r['seed']==seed})==1
    return rows


def audit_test(root):
    rows = read(root/'results.json')
    assert {(r['mode'],r['seed']) for r in rows}==set(product(MODES,SEEDS))
    for seed in SEEDS:
        same = [r for r in rows if r['seed']==seed]
        assert len({r['train_sha256'] for r in same})==1
        active = {r['mode']:r['active'] for r in same}
        assert active['fixed']==active['evolve']
    labels = {gap:y for gap,_,y in data_sets('test',TEST_GAPS,128)}
    for r in rows:
        folder = root/f'{r["recipe"]}-{r["mode"]}-{r["seed"]}'
        checkpoint = folder/'weights.pt'
        assert digest(checkpoint)==r['weights_sha256']
        assert int(torch.load(checkpoint,weights_only=True)['active'].sum())==r['active']
        raw = read(folder/'test.json')
        for a,b in zip(raw,r['test'],strict=True):
            y = labels[b['gap']]
            assert a['gap']==b['gap'] and a['targets']==y.tolist()
            hits = torch.tensor(a['probabilities']).argmax(-1)==y
            assert b['joint']==float(hits.flatten(1).all(1).float().mean())
            assert b['per_cue']==hits.all(1).float().mean(0).tolist()
        topology = read(folder/'topology.json')
        current = WIDTH**2
        for event in topology:
            current += event['woke']-event['slept']
            assert current==event['active']
        if r['mode'] in ('decay','evolve'):
            assert current==r['active']
        r['topology'] = topology
    return rows


def density_plot(rows,protocol):
    steps = protocol['steps']
    x = lambda step:65+780*step/steps
    y = lambda fraction:210-182*fraction
    svg = ['<svg viewBox="0 0 900 255" role="img" aria-label="活跃循环边比例随训练变化">']
    for f in (0,.25,.5,.75,1):
        yy = y(f)
        svg.append(f'<path d="M65 {yy}H845" stroke="#d1dbe5"/>'
                   f'<text class="axis" x="53" y="{yy+4}" text-anchor="end">{f:.0%}</text>')
    for step in (0,protocol['warmup'],steps//2,steps):
        svg.append(f'<text class="axis" x="{x(step)}" y="240" text-anchor="middle">{step:,}</text>')
    for mode in MODES:
        selected = [r for r in rows if r['mode']==mode]
        if mode in ('dense','fixed'):
            pairs = [(s,mean(r['active'] for r in selected)) for s in (0,steps)]
        else:
            pairs = [(0,WIDTH**2),(protocol['warmup'],WIDTH**2)]+[
                (group[0]['step'],mean(c['active'] for c in group))
                for group in zip(*(r['topology'] for r in selected),strict=True)]
        points = ' '.join(f'{x(t)},{y(n/WIDTH**2)}' for t,n in pairs)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[mode]}" stroke-width="3"/>')
    return ''.join(svg)+'</svg>'


def adaptation(root):
    path = root/'adaptation.json'
    if not path.exists():
        return ''
    data = read(path)
    if len(data)!=15:
        return ''
    rows = []
    for seed in SEEDS:
        selected = [r for r in data if r['seed']==seed]
        assert len({r['train_sha256'] for r in selected})==1
        evolve,frozen = (next(r for r in selected if r['mode']==m) for m in ('evolve','evolve-frozen'))
        assert evolve['source_sha256']==frozen['source_sha256']
        assert evolve['points'][0]==frozen['points'][0]
    for mode in (*MODES,'evolve-frozen'):
        selected = [r for r in data if r['mode']==mode]
        for title,gaps in (('训练范围',TRAIN_GAPS),('64–512 步',(64,128,256,512))):
            scores = [[mean(c['joint'] for c in r['points'][i]['scores'] if c['gap'] in gaps)
                       for r in selected] for i in range(5)]
            rows.append([(LABELS[mode],''),(title,''),*[mean_range(v) for v in scores]])
    return ('<h2>更换查询规则</h2><p>交换两个查询符号指向的线索。两条线索和干扰序列保持原格式。'
            '冻结组仍更新权重，生存强度和活跃掩码保持固定。</p>'
            +table(('模型','延迟','起点','40 次','100 次','200 次','360 次'),rows))


def cuts(root):
    path = root/'cuts.json'
    if not path.exists():
        return ''
    data = read(path)
    if len(data)!=6:
        return ''
    rows = []
    for mode,cue,i in product(('dense','evolve'),range(2),range(4)):
        subset = [r for r in data if r['mode']==mode]
        base = [r['base'][i] for r in subset]
        treated = [r['effects'][cue] for r in subset]
        drop = [[b['per_cue'][j]-t['target'][i]['per_cue'][j] for b,t in zip(base,treated,strict=True)]
                for j in range(2)]
        random = [b['per_cue'][cue]-mean(c[i]['per_cue'][cue] for c in t['random'])
                  for b,t in zip(base,treated,strict=True)]
        rows.append([(LABELS[mode],''),(f'线索 {cue+1}',''),(base[0]['gap'],''),
                     *[(f'{100*mean(v):+.1f}','') for v in (*drop,random)]])
    return ('<h2>线索选择性断边</h2><p>在 32 步开发题用两条线索的损失敏感度之差选约 1% 的活跃边。'
            '随机对照匹配每个神经元失去的入边数。正值表示成绩下降，单位为百分点。</p>'
            +table(('模型','定向切除','延迟','线索 1 降幅','线索 2 降幅','随机切除的目标降幅'),rows))


def resources(root):
    paths = [root/f'profile-{m}.json' for m in MODES]
    if not all(p.exists() for p in paths):
        return ''
    rows = []
    for r in map(read,paths):
        csr = r['csr']
        rows.append([(LABELS[r['mode']],''),(f'{r["masked_ms"]:.2f}',''),
                     (f'{r["masked_peak_increment_mib"]:.2f}',''),
                     (f'{csr["ms"]:.2f}' if 'ms' in csr else '当前后端不可用',''),
                     ('一致' if csr.get('matches') else '未确认','')])
    return ('<h2>实际推理成本</h2><p>种子 2，512 步干扰，单线程 CPU，批量 1。CSR 时间复用打包好的循环矩阵。</p>'
            +table(('模型','普通整题毫秒','首次峰值增量 MiB','CSR 整题毫秒','CSR 输出检查'),rows))


def four_records(root,include_dev=True):
    if root is None or not (root/'probes.json').exists():
        return ''
    data = read(root/'probes.json')
    if len(data)!=3:
        return ''
    rows = []
    for step in (1600,2400,data[0]['steps']):
        values = []
        for gap in (*TRAIN_GAPS,64):
            values.append([c['joint'] for r in data for p in r['history'] if p['step']==step
                           for c in p['scores'] if c['task']=='lookup' and c['gap']==gap])
        rows.append([(step,'')]+[mean_range(v) for v in values])
    text = ('<h2>四记录原协议的容量课程</h2><p>训练从两条记录加到三条，再到四条，然后增加延迟。'
            '键、值继续分开输入。以下开发筛选与双线索协议分别计分。</p>'
            +table(('训练更新','0 步','4 步','16 步','32 步','64 步'),rows)) if include_dev else ''
    if not (root/'results.json').exists():
        return text
    results = read(root/'results.json')
    assert {(r['mode'],r['seed']) for r in results}==set(product(MODES,SEEDS)) and len(results)==12
    for seed in SEEDS:
        same = [r for r in results if r['seed']==seed]
        assert len({r['train_sha256'] for r in same})==1
        active = {r['mode']:r['active'] for r in same}
        assert active['fixed']==active['evolve']
    expected = {(t,g):y for t,g,_,y in cells('capacity-holdout',TEST_GAPS)}
    for r in results:
        folder = root/f'{r["probe"]}-{r["mode"]}-{r["seed"]}'
        weights = folder/'weights.pt'
        assert digest(weights)==r['weights_sha256']
        assert int(torch.load(weights,weights_only=True)['active'].sum())==r['active']
        for raw,reported in zip(read(folder/'holdout.json'),r['holdout'],strict=True):
            label = expected[reported['task'],reported['gap']]
            assert raw['targets']==label.tolist()
            hits = torch.tensor(raw['probabilities']).argmax(-1)==label
            assert reported['joint']==float(hits.all(1).float().mean())
        topology = read(folder/'topology.json')
        count = WIDTH**2
        for event in topology:
            count += event['woke']-event['slept']
            assert count==event['active']
        if r['mode'] in ('evolve','decay'):
            assert count==r['active']
        r['topology'] = topology
    protocol = read(root/'protocol.json')
    selection = read(root/'selection.json')
    protocol['warmup'] = selection['topology_warmup']
    rate = selection.get('survival',{}).get('rate',.18)
    title = {'recall':'单线索回忆','lookup':'四记录查值'}
    lines = []
    for task,mode in product(('recall','lookup'),MODES):
        group = [r for r in results if r['mode']==mode]
        values = [[c['joint'] for r in group for c in r['holdout'] if (c['task'],c['gap'])==(task,g)]
                  for g in TEST_GAPS]
        lines.append([(title[task],''),(LABELS[mode],'')]+[mean_range(v) for v in values])
    density = density_plot(results,protocol)
    labels = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in MODES)
    text += f'''<h3>四记录协议 · 生存更新系数 {rate:.2f} · 同预算冻结留出</h3>
    <p>按种子匹配最终边数的固定随机模型从头训练。四种结构看到相同训练题和相同更新次数。
    表内分别列示训练范围与 64–512 步外推，最终预测使用普通读出头。</p>
    {table(('任务','模型',*[f'{g} 步' for g in TEST_GAPS]),lines)}
    <h3>四记录协议 · 活跃边比例</h3><div class="legend">{labels}</div>{density}'''
    retrain_path = root/'retrained-masks.json'
    if retrain_path.exists():
        retrained = read(retrain_path)
        if len(retrained)==3:
            for r in retrained:
                original = next(x for x in results if x['mode']=='evolve' and x['seed']==r['seed'])
                assert r['mask_source_sha256']==original['weights_sha256']
                assert r['train_sha256']==original['train_sha256'] and r['active']==original['active']
            masking = []
            for name,group in (('演化过程中的原权重',[r for r in results if r['mode']=='evolve']),
                               ('演化掩码冻结后从头训练',retrained),
                               ('同密度随机掩码从头训练',[r for r in results if r['mode']=='fixed'])):
                values = [[c['joint'] for r in group for c in r['holdout'] if c['task']=='lookup' and c['gap']==gap]
                          for gap in TEST_GAPS]
                masking.append([(name,'')]+[mean_range(v) for v in values])
            text += ('<h3>把最终连接图和训练历史分开</h3><p>两个固定掩码组从相同随机初值重新训练。'
                     '演化掩码组继承边的位置，权重与优化器状态重新初始化。</p>'
                     +table(('模式',*[f'{g} 步' for g in TEST_GAPS]),masking))
    diagnostic_path = root/'binding-diagnosis-topology.json'
    if diagnostic_path.exists():
        diagnostics = read(diagnostic_path)
        lines = []
        for mode in MODES:
            group = [r for r in diagnostics if r['mode']==mode]
            for gap in (0,32):
                selected = [c for r in group for c in r['by_gap'] if c['gap']==gap]
                if selected:
                    lines.append([(LABELS[mode],''),(gap,''),
                                  *[mean_range([c[k] for c in selected]) for k in
                                    ('auxiliary_single','all_queries','prediction_changes')]])
        text += ('<h3>四记录协议 · 查询前保留与选择</h3><p>辅助读出从查询前状态解码键值。'
                 '最终结果仍由普通读出头评分。</p>'
                 +table(('模型','干扰步数','辅助单键正确','最终四问全对','答案不同时预测改变'),lines))
    adaptation_path = root/'adaptation.json'
    if adaptation_path.exists():
        adapts = read(adaptation_path)
        if len(adapts)==15:
            rows = []
            for seed in SEEDS:
                same = [r for r in adapts if r['seed']==seed]
                assert len({r['train_sha256'] for r in same})==1
                first = [r for r in same if r['mode'] in ('evolve','evolve_frozen')]
                assert first[0]['source_sha256']==first[1]['source_sha256']
                assert first[0]['history'][0]==first[1]['history'][0]
            for mode in (*MODES,'evolve-frozen'):
                logged = mode.replace('-','_')
                selected = [r for r in adapts if r['mode']==logged]
                scores = [[mean(c['joint'] for c in r['history'][i]['lookup'] if c['gap'] in TRAIN_GAPS)
                           for r in selected] for i in range(5)]
                rows.append([(LABELS[mode],'')]+[mean_range(s) for s in scores])
            text += ('<h3>四记录协议 · 规则变化后</h3><p>查询改为读取下一个键，重新训练 480 次。'
                     '冻结对照继续更新权重，保持原生存强度和掩码。</p>'
                     +table(('模型','起点','80 次','160 次','320 次','480 次'),rows))
    return text+resources(root)


def render(root,capacity,slow):
    protocol = read(root/'protocol.json')
    development = audit_development(root,protocol)
    dev = [[(RECIPES[p],'')]+[mean_range([c['joint'] for r in development if r['recipe']==p
                           for c in r['history'][-1]['scores'] if c['gap']==gap]) for gap in DEV_GAPS]
           for p in protocol['recipes']]
    tested = ''
    if (root/'results.json').exists():
        rows = audit_test(root)
        result_rows = [[(LABELS[mode],'')]+[mean_range([c['joint'] for r in rows if r['mode']==mode
                                      for c in r['test'] if c['gap']==gap]) for gap in TEST_GAPS] for mode in MODES]
        counts = [[(LABELS[r['mode']],''),(r['seed'],''),(f'{r["active"]:,}',''),
                   (f'{sum(e["woke"] for e in r["topology"]):,}','')] for r in rows]
        legend = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in MODES)
        tested = f'''<h2>双线索留出成绩</h2><p>每个条件 128 组，横跨训练延迟与 64–512 步外推。</p>
        {table(('模型',*[f'{g} 步' for g in TEST_GAPS]),result_rows)}
        <h2>循环边演化</h2><div class="legend">{legend}</div>{density_plot(rows,protocol)}
        {table(('模型','种子','最终活跃边','累计再生次数'),counts)}
        {adaptation(root)}{cuts(root)}{resources(root)}'''
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>双线索与容量课程</title>
    <style>{CSS}</style><main><p class="eyebrow">自演化稀疏 RNN · 512 神经元</p>
    <h1>能否同时保存和读取两条独立线索？</h1>
    <p>两条记录的顺序随机，各自独立取值。每组要求原始历史、改第一条来源、改第二条来源各回答两个查询，共六题全对。
    辅助读出只用于训练。留出评测的每次查询仅使用最终预测头。</p>
    <h2>同训练数据的三种开发训练方式</h2>
    {table(('训练方式',*[f'{g} 步' for g in DEV_GAPS]),dev)}
    {tested}{four_records(capacity)}{four_records(slow,False)}
    </main></html>'''
    (root/'report-cn.html').write_text(doc,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--capacity',type=Path)
    parser.add_argument('--slow',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    render(args.root,args.capacity,args.slow)
