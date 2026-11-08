"""Audit the first 512-neuron topology rounds and render a Chinese report."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import torch

from evolving_rnn_eval import MODES, TASKS, TEST_GAPS, TRAIN_GAPS, cells
from frontier_eval import SEEDS, digest, read
from shared_address_report import CSS, heat, mean_range, table


LABELS = dict(dense='始终全连接',fixed='同密度固定随机',decay='只衰减',evolve='衰减与再生',
              evolve_frozen='同一演化权重，冻结拓扑',evolve_guarded='允许演化，校准失败时回滚')
COLORS = dict(dense='#2b60a3',fixed='#7041ab',decay='#994700',evolve='#00634f')
TASK_NAMES = dict(recall='延迟回忆',lookup='关联查值')


def audit(root):
    protocol = read(root/'protocol.json')
    runs = read(root/'results.json')
    assert {(r['mode'],r['seed']) for r in runs}==set(product(MODES,SEEDS)) and len(runs)==12
    expected = list(cells('holdout',TEST_GAPS))
    for seed in SEEDS:
        same = [r for r in runs if r['seed']==seed]
        assert len({r['train_sha256'] for r in same})==1
        active = {r['mode']:r['active'] for r in same}
        assert active['fixed']==active['evolve']
    for r in runs:
        folder = root/f'{r["mode"]}-{r["seed"]}'
        assert digest(folder/'weights.pt')==r['weights_sha256']
        assert int(torch.load(folder/'weights.pt',weights_only=True)['active'].sum())==r['active']
        for observed,raw,(task,gap,_,labels) in zip(r['holdout'],read(folder/'holdout.json'),expected,strict=True):
            assert (raw['task'],raw['gap'])==(task,gap)
            assert raw['targets']==labels.tolist()
            hits = torch.tensor(raw['probabilities']).argmax(-1)==labels
            assert raw['joint']==observed['joint']==float(hits.all(1).float().mean())
            assert raw['single']==observed['single']==float(hits[:,0].float().mean())
        topology = read(folder/'topology.json')
        count = protocol['width']**2
        for event in topology:
            count += event['woke']-event['slept']
            assert count==event['active']
        if r['mode'] in ('decay','evolve'):
            assert count==r['active']
        r['topology'] = topology
    return protocol,runs


def topology_plot(runs,protocol):
    x = lambda step:60+790*step/protocol['steps']
    y = lambda count:235-210*count/protocol['width']**2
    svg = ['<svg viewBox="0 0 900 283" role="img" aria-label="活跃循环边随训练更新变化">']
    total = protocol['width']**2
    for f in (0,.25,.50,.75,1):
        yy = y(f*total)
        svg.append(f'<path d="M60 {yy}H850" stroke="#d1dbe5"/>'
                   f'<text class="axis" x="52" y="{yy+4}" text-anchor="end">{f:.0%}</text>')
    for step in (0,protocol['warmup'],400,600,800,protocol['steps']):
        svg.append(f'<text class="axis" x="{x(step)}" y="264" text-anchor="middle">{step:,}</text>')
    for mode in MODES:
        selected = [r for r in runs if r['mode']==mode]
        if mode in ('dense','fixed'):
            pairs = [(0,mean(r['active'] for r in selected)),(protocol['steps'],mean(r['active'] for r in selected))]
        else:
            pairs = [(0,total),(protocol['warmup'],total)]+[
                (event[0]['step'],mean(e['active'] for e in event))
                for event in zip(*(r['topology'] for r in selected),strict=True)]
        coords = ' '.join(f'{x(a)},{y(b)}' for a,b in pairs)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[mode]}" stroke-width="3"/>')
    return ''.join(svg)+'</svg>'


def diagnostics(root):
    sections = []
    if (root/'transfer/results.json').exists():
        results = read(root/'transfer/results.json')
        assert len(results)==15
        for seed in SEEDS:
            assert len({r['adapt_sha256'] for r in results if r['seed']==seed})==1
        rows = []
        for mode in (*MODES,'evolve_frozen'):
            selected = [r for r in results if r['mode']==mode]
            values = [[mean(c['joint'] for c in r['points'][i]['new_rule'] if c['gap'] in TRAIN_GAPS)
                       for r in selected] for i in range(4)]
            rows.append([(LABELS[mode],'')]+[mean_range(v) for v in values])
        sections.append('<h2>查值规则改变后的恢复</h2><p>同一查询改为读取相邻键。权重继续训练，'
                        '只有冻结拓扑组阻止原演化网络改边。表中是新规则、训练延迟范围内的联合正确率。'
                        '原规则的基线尚未学稳，因此不能把这次低分解释为遗忘。</p>'
                        +table(('模型','改变前','80 次更新','160 次更新','320 次更新'),rows))
    if (root/'long-transfer/results.json').exists():
        results = read(root/'long-transfer/results.json')
        assert len(results)==15
        guarded_path = root/'long-transfer-guarded/results.json'
        if guarded_path.exists():
            guarded = read(guarded_path)
            assert len(guarded)==3
            results += guarded
            attempts = [event for r in guarded for event in r['structural']]
            sections.append(f'<h2>结构更新校准</h2><p>每次先后用独立校准题比较更新前后的短、长回忆。'
                            f'{len(attempts)} 次提议中回滚 {sum(not e["accepted"] for e in attempts)} 次。'
                            '回滚恢复循环权重、活跃边、生存变量和对应的优化器状态。'
                            '校准增加前向开销，训练题目与更新次数相同。</p>')
        for seed in SEEDS:
            assert len({r['adapt_sha256'] for r in results if r['seed']==seed})==1
        rows = []
        for mode in (*MODES,'evolve_frozen',*(('evolve_guarded',) if guarded_path.exists() else ())):
            runs = [r for r in results if r['mode']==mode]
            for title,gaps in (('加入训练的长延迟',(64,128)),('继续外推',(256,512))):
                values = [[mean(c['joint'] for c in r['points'][i]['new_rule'] if c['gap'] in gaps)
                           for r in runs] for i in range(4)]
                old = [mean(c['joint'] for c in r['points'][-1]['old_rule'] if c['task']=='recall') for r in runs]
                rows.append([(LABELS[mode],''),(title,'')]+[mean_range(v) for v in values]+[mean_range(old)])
        sections.append('<h2>突然加入长记忆需求</h2><p>从冻结检查点继续训练，三分之二的更新使用 64、128 步回忆，'
                        '三分之一继续使用原任务。固定拓扑的演化对照保留原生存强度和活跃掩码，权重继续更新。'
                        '最后一列是原短延迟回忆的保留情况。</p>'
                        +table(('模型','测试条件','起点','80 次','160 次','320 次','旧回忆'),rows))
    if (root/'cuts.json').exists():
        cuts = read(root/'cuts.json')
        assert len(cuts)==12
        rows = []
        for mode in MODES:
            selected = [r for r in cuts if r['mode']==mode]
            for i,gap in enumerate((4,32,128,256)):
                base = [r['base'][i]['joint'] for r in selected]
                targeted = [r['base'][i]['joint']-r['targeted'][i]['joint'] for r in selected]
                random = [r['base'][i]['joint']-mean(c[i]['joint'] for c in r['random']) for r in selected]
                rows.append([(LABELS[mode],''),(gap,''),mean_range(base),(f'{100*mean(targeted):+.1f}',''),
                             (f'{100*mean(random):+.1f}','')])
        sections.append('<h2>隐藏循环边干预</h2><p>用开发题筛选对 32 步比 4 步更敏感的边，固定切掉约 1%。'
                        '随机对照匹配各神经元失去的入边数。降幅用百分点表示。</p>'
                        +table(('模型','延迟步数','原联合正确率','定向切边降幅','随机切边降幅'),rows))
    paths = [root/f'profile-{m}.json' for m in MODES]
    if all(p.exists() for p in paths):
        rows = []
        measured = [read(p) for p in paths]
        tokens = measured[0]['tokens']
        assert all(r['tokens']==tokens for r in measured)
        for r in measured:
            csr = r['csr']
            display = f'{csr["ms"]:.2f}' if 'ms' in csr else '当前后端不支持'
            rows.append([(LABELS[r['mode']],''),(f'{r["masked_ms"]:.2f}',''),
                         (f'{r["masked_peak_increment_mib"]:.2f}',''),(display,''),
                         ('是' if csr.get('matches') else '未确认','')])
        sections.append(f'<h2>单线程 CPU 推理</h2><p>种子 2，批量 1，{tokens} 符号。普通实现使用带掩码的稠密循环核。'
                        'CSR 是复用已打包矩阵的独立推理测量，采用相同权重。</p>'
                        +table(('模型','普通整题毫秒','首次峰值增量 MiB','CSR 整题毫秒','CSR 与普通输出一致'),rows))
    return ''.join(sections)


def age_comparison(reference,runs):
    _,older = audit(reference)
    rows = []
    for seed in SEEDS:
        common = next(r for r in older if (r['mode'],r['seed'])==('evolve',seed))
        independent = next(r for r in runs if (r['mode'],r['seed'])==('evolve',seed))
        assert common['train_sha256']==independent['train_sha256']
        for title,gaps in (('训练长度范围',TRAIN_GAPS),('长度外推',(64,128,256,512))):
            values = [mean(c['joint'] for c in r['holdout'] if c['task']=='recall' and c['gap'] in gaps)
                      for r in (common,independent)]
            rows.append([(seed,''),(title,''),*[heat(v) for v in values],(f'{100*(values[1]-values[0]):+.1f}','')])
    return ('<h2>新生边独立计步</h2><p>只改变恢复边的 AdamW 偏差修正步数。相同种子的训练题目与顺序散列一致。'
            '下表是冻结权重后的延迟回忆联合正确率。正差表示独立计步更高。</p>'
            +table(('种子','范围','共享步数','新边独立步数','差值（百分点）'),rows))


def render(root,rounds,reference=None):
    protocol,runs = audit(root)
    selected = {m:[r for r in runs if r['mode']==m] for m in MODES}
    rows = []
    for task,gap in product(TASKS,TEST_GAPS):
        values = [[c['joint'] for r in selected[m] for c in r['holdout']
                   if (c['task'],c['gap'])==(task,gap)] for m in MODES]
        rows.append([(TASK_NAMES[task],''),(gap,''),('训练长度范围' if gap in TRAIN_GAPS else '外推',''),
                     *[mean_range(v) for v in values]])
    round_rows = []
    for directory in rounds:
        p = read(directory/'protocol.json')
        config = p['survival']
        group = [r for r in read(directory/'development.json') if r['mode']=='evolve']
        assert {r['seed'] for r in group}==set(SEEDS)
        births = [sum(e['woke'] for e in read(directory/f'evolve-{r["seed"]}'/'topology.json')) for r in group]
        title = f'再生收益 {config["dormant_credit"]}'
        if config.get('traces'):
            title += ' · 多时间尺度'
        if config.get('dynamics'):
            title += ' · 扰动响应'
        if config.get('independent_age'):
            title += ' · 新生边独立计步'
        round_rows.append([(title,''),(f'{min(births):,}–{max(births):,}',''),
                           mean_range([r['active']/p['width']**2 for r in group]),
                           *[mean_range([mean(c['joint'] for c in r['history'][-1]['scores'] if c['task']==t)
                                         for r in group]) for t in TASKS]])
    legend = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in MODES)
    data = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>全连接 RNN 拓扑演化</title>
    <style>{CSS}</style><main><p class="eyebrow">神经网络研究 · 512 神经元</p>
    <h1>循环边会自行休眠和再生吗？</h1>
    <p>全连接起步。任务收益、维持成本和神经元局部竞争驱动生存变量。休眠边由潜在梯度触发再生。
    最终连边数量由动力学决定。对照共享题目、全序列反传预算和三个种子。</p>
    <p><b>当前瓶颈</b>　四组均未稳定学会关联查值。成功的回忆条件和实际再生可独立审计。
    多时间尺度和扰动响应的初轮开发筛选未带来稳定收益。</p>
    <h2>活跃循环边比例</h2><p>横轴为更新次数。同密度固定对照在演化结束后按同种子的活跃边数另行匹配。</p>
    <div class="legend">{legend}</div>{topology_plot(runs,protocol)}
    <h2>配对留出成绩</h2><p>每格显示三种子均值和范围。回忆要求原题和改线索都答对。查值还要求改来源题答对。
    留出使用独立随机种子，零步等有限题型仍可能重复相同序列。</p>
    {table(('任务','干扰步数','长度范围',*(LABELS[m] for m in MODES)),rows)}
    <h2>单因素开发筛选</h2>
    {table(('版本','累计再生次数范围','最终活跃比例','回忆联合','查值联合'),round_rows)}
    {age_comparison(reference,runs) if reference else ''}
    {diagnostics(root)}
    </main></html>'''
    (root/'report-cn.html').write_text(data,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--rounds',nargs='+',type=Path,required=True)
    parser.add_argument('--age-reference',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    render(args.root,args.rounds,args.age_reference)
