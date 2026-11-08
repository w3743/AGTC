"""Verify held-out predictions and render the conductance experiment in Chinese."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

import torch

from evolving_rnn_conductance import SNAPSHOTS
from evolving_rnn_eval import TEST_GAPS, TRAIN_GAPS, WIDTH, cells
from evolving_rnn_lookup import capacity_batch
from evolving_rnn_sleep_guard import TrainingCalibration
from frontier_eval import SEEDS, digest, read
from shared_address_report import CSS, mean_range, table


LABEL = dict(dense='始终全连接', gradual='逐渐减弱＋再生', binary='全强度＋再生', fixed='同密度固定随机',
             paced='分批断边＋再生', frozen='冻结拓扑',
             **{'binary-decay':'全强度＋只断边','binary-frozen':'全强度＋冻结拓扑',
                'paced-decay':'分批断边，无再生','paced-frozen':'分批版冻结拓扑',
                'guarded':'校准筛边＋再生','guarded-decay':'校准筛边，无再生',
                'guarded-frozen':'校准版冻结拓扑','prefix':'校准断边数量＋再生',
                'prefix-decay':'校准断边数量，无再生','prefix-frozen':'校准数量版冻结拓扑',
                'prefix-no-wake':'校准数量版仅继续断边','dense-cpu':'全连接，等计算',
                'fixed-cpu':'固定随机，等计算','previous':'完整批次校准',
                'dense-indexed-cpu':'全连接，匹配索引版计算','fixed-indexed-cpu':'固定随机，匹配索引版计算',
                'prefix-dynamics':'扰动响应评分','prefix-traces':'时间资格迹评分',
                'prefix-no-dynamics':'移除扰动响应评分'})
COLOR = dict(dense='#265791', gradual='#974800', binary='#00634f', fixed='#6a3d99',paced='#974800',
             guarded='#974800',prefix='#50399b',**{'binary-decay':'#a33368','paced-decay':'#515479',
                                                    'guarded-decay':'#515479','prefix-decay':'#515479'})


def group(rows,mode):
    return [r for r in rows if r['mode']==mode]


def joint(scores,task,gaps):
    return mean(c['joint'] for c in scores if c['task']==task and c['gap'] in gaps)


def text_table(headers,rows):
    return table(headers,[[(v,'') for v in row] for row in rows])


def audit_holdout(root,row,expected,checkpoint=None):
    folder = root/f'{row["mode"]}-{row["seed"]}'
    path = folder/'weights.pt' if checkpoint is None else checkpoint
    assert digest(path)==row['weights_sha256']
    assert int(torch.load(path,weights_only=True)['active'].sum())==row['active']
    raw_predictions = read(folder/'holdout.json')
    assert {(c['task'],c['gap']) for c in raw_predictions}==set(expected)
    for raw,summary in zip(raw_predictions,row['holdout'],strict=True):
        assert (raw['task'],raw['gap'])==(summary['task'],summary['gap'])
        y = expected[raw['task'],raw['gap']]
        assert y.tolist()==raw['targets']
        hit = torch.tensor(raw['probabilities']).argmax(-1)==y
        assert float(hit.all(1).float().mean())==summary['joint']
        assert float(hit[:,0].float().mean())==summary['single']


def audit(root,rows,arms,selected,partition):
    assert {(r['mode'],r['seed']) for r in rows}==set(product((*arms,'fixed'),SEEDS))
    expected = {(task,gap):y for task,gap,_,y in cells(partition,TEST_GAPS)}
    for seed in SEEDS:
        matched = {r['mode']:r for r in rows if r['seed']==seed}
        assert len({r['train_sha256'] for r in matched.values()})==1
        assert len({matched[m]['source_sha256'] for m in arms})==1
        assert all(matched[m]['history'][0]==matched['dense']['history'][0] for m in arms)
        assert matched['fixed']['active']==matched[selected]['active']
    for r in rows:
        folder = root/f'{r["mode"]}-{r["seed"]}'
        audit_holdout(root,r,expected)
        r['events'] = [] if r['mode'] in ('fixed','dense') else read(folder/'topology.json')
        active = WIDTH**2
        for e in r['events']:
            active += e['woke']-e['slept']
            assert active==e['active']
        if r['events']:
            assert active==r['active']
            if r['mode'].startswith('paced'):
                window = 0
                for e in r['events']:
                    if 'eligible' in e:
                        assert window==0
                        window = e['eligible']
                    else:
                        assert e['woke']==0
                    window -= e['slept']
                    assert window>=0
                assert window==0


def curve(rows,arms,metric,description):
    x = lambda t:65+720*t/SNAPSHOTS[-1]
    y = lambda p:208-178*p
    svg = [f'<svg viewBox="0 0 850 248" role="img" aria-label="{description}">']
    for p in (0,.25,.5,.75,1):
        svg.append(f'<path d="M65 {y(p)}H785" stroke="#d1dbe5"/><text class="axis" x="56" y="{y(p)+4}" text-anchor="end">{p:.0%}</text>')
    for i,t in enumerate(SNAPSHOTS):
        svg.append(f'<text class="axis" x="{x(t)}" y="230" text-anchor="middle">{t}</text>')
        if i:
            svg.append(f'<path d="M{x(t)} 30V208" stroke="#e5edf4"/>')
    for mode in arms:
        coords = [(x(t),y(mean(metric(r['history'][i]) for r in group(rows,mode))))
                  for i,t in enumerate(SNAPSHOTS)]
        points = ' '.join(f'{a},{b}' for a,b in coords)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{COLOR[mode]}" stroke-width="3"/>')
        svg.extend(f'<circle cx="{a}" cy="{b}" r="3" fill="{COLOR[mode]}"/>' for a,b in coords)
    return ''.join(svg)+'</svg>'


def changes(rows,arms):
    events = []
    for mode in arms[1:]:
        for r in group(rows,mode):
            current = {e['step']:e for e in r['events']}
            for h in r['history'][1:]:
                if 'before_structure' not in h:
                    continue
                delta = joint(h['scores'],'lookup',TRAIN_GAPS)-joint(h['before_structure'],'lookup',TRAIN_GAPS)
                e = current[h['step']]
                events.append((mode,r['seed'],h['step'],e['slept'],e['woke'],delta))
    return text_table(('机制','种子','继续训练更新数','断开边','再生边','当次查值变化'),
                      [(LABEL[m],s,t,f'{d:,}',f'{w:,}',f'{100*p:+.2f} 个百分点')
                       for m,s,t,d,w,p in sorted(events,key=lambda row:row[-1])[:9]])


def topology_totals(rows,arms):
    return text_table(('方案','种子','累计断开','累计再生','单步断开峰值','最终活跃边'),[
        (LABEL[m],r['seed'],f'{sum(e["slept"] for e in r["events"]):,}',
         f'{sum(e["woke"] for e in r["events"]):,}',f'{max(e["slept"] for e in r["events"]):,}',f'{r["active"]:,}')
        for m in arms[1:] for r in group(rows,m)])


def guard_audit(rows,settings,reuse=False):
    totals, paired = [],[]
    task_names = ('回忆','四记录查值')
    modes = ('guarded','prefix','prefix-decay') if 'prefix_divisors' in settings else ('guarded','guarded-decay')
    for mode in modes:
        for r in group(rows,mode):
            candidates = [e for e in r['events'] if e['eligible']]
            assert all('guard' in e and 0<=e['slept']<=e['eligible'] for e in candidates)
            passes = [e for e in candidates if e['slept']==e['eligible']]
            changed = [e for e in candidates if 0<e['slept']<e['eligible']]
            comparisons = [e['guard']['equal_count'] for e in changed if 'equal_count' in e['guard']]
            eligible = sum(e['eligible'] for e in candidates)
            slept = sum(e['slept'] for e in candidates)
            calls = sum(1+e['guard']['checks'] for e in candidates)
            default_sequences = settings['samples_per_gap']*len(TRAIN_GAPS)*5
            sequences = sum((1+e['guard']['checks'])*e['guard'].get('sequences_per_check',default_sequences)
                            for e in candidates)
            totals.append((LABEL[mode],r['seed'],len(candidates),len(passes),len(changed),
                           f'{eligible-slept:,}',f'{calls:,}',f'{sequences:,}'))
            for e in candidates:
                before,after = (e['guard'][s]['loss'] for s in ('before','after'))
                assert all(a<=b+d+1e-10 for b,a,d in zip(before,after,settings['tolerance'],strict=True))
                if mode.startswith('prefix'):
                    assert e['guard']['checks']<=1+len(settings['prefix_divisors'])
                    valid = {0,e['eligible'],*(e['eligible']//d for d in settings['prefix_divisors'])}
                    assert e['slept'] in valid
            for i,task in enumerate(task_names):
                if comparisons:
                    effect = mean(c['guarded']['joint'][i]-c['low_survival']['joint'][i] for c in comparisons)
                    loss = mean(c['guarded']['loss'][i]-c['low_survival']['loss'][i] for c in comparisons)
                    paired.append((LABEL[mode],r['seed'],task,len(comparisons),
                                   sum(not c['same_edges'] for c in comparisons),f'{100*effect:+.2f}',f'{loss:+.4f}'))
    prefix_note = ('数量校准版仅检查全批及其二分之一、四分之一、八分之一和十六分之一。'
                   '每个候选掩码始终取生存值最低的连续前缀。') if 'prefix_divisors' in settings else ''
    calibration_source = ('<p>数量校准使用最近的训练批次，逐组筛边使用独立校准题。' if reuse
                          else '<p>每次从新的随机流抽校准题。')
    return ('<h2>校准成本和断边选择</h2>' + calibration_source +
            '先测试所有候选。逐组筛边在整批超出容许损失时，按生存值顺序测试八组。'
            +prefix_note+
            '比较的是相对同一个未断边起点的累计损失。暂缓的边保留原生存值，后续可再次竞争。'
            '校准前向不参与梯度更新。回忆题组含两个输入，查值题组含三个输入。额外序列数包括重复测同题，不含事后审计。重复候选可以多次暂缓。</p>'
            +text_table(('方案','种子','有候选更新','全批通过','部分通过','暂缓次数累计','额外校准轮数','额外前向序列数'),totals)
            +'<p>每 100 次更新审计部分通过的事件，用另一批配对题比较已批准的边与同数量、生存值最低的候选边。'
             '正的准确率差及负的交叉熵差有利于筛选版。这一审计只改变当前检查点的边掩码。</p>'
            +text_table(('方案','种子','任务','可审计事件','实际换了选边的事件','联合正确率差（百分点）','交叉熵差'),paired))


def blind_control(root,guarded,partition):
    rows = read(root/'blind-results.json')
    assert {r['seed'] for r in rows}==set(SEEDS)
    expected = {(task,gap):y for task,gap,_,y in cells(partition,TEST_GAPS)}
    comparisons = []
    for row in rows:
        audit_holdout(root,row,expected)
        seed = row['seed']
        reference = next(r for r in guarded if r['seed']==seed)
        assert all(row[key]==reference[key] for key in ('source_sha256','train_sha256'))
        control = read(root/f'blind-{seed}'/'topology.json')
        for e,r in zip(control,reference['events'],strict=True):
            assert e['step']==r['step'] and e['requested_sleep']==r['slept']
            assert e['slept']==min(e['eligible'],e['requested_sleep'])
        for model,events,label in ((reference,reference['events'],'校准筛边'),(row,control,'按生存值断边')):
            slept = sum(e['slept'] for e in events)
            woke = sum(e['woke'] for e in events)
            assert WIDTH**2+woke-slept==model['active']
            comparisons.append((seed,label,f'{slept:,}',f'{woke:,}',f'{model["active"]:,}',
                                *(f'{joint(model["holdout"],task,gaps):.2%}' for task,gaps in
                                  (('recall',TRAIN_GAPS),('lookup',TRAIN_GAPS),('lookup',(512,))))))
    return ('<h2>只复用断边数量，重新训练</h2><p>从同一全连接检查点开始。对照逐次请求与校准组相同的休眠数量，只按生存值挑选候选。'
            '候选不足时按实际数量执行。两组分别按潜在梯度再生，最终边数因此可以不同。该对照用于区分节制断边数量与校准选边的作用。</p>'
            +text_table(('种子','方案','实际断边','实际再生','最终活跃边','训练内回忆','训练内查值','512 步查值'),comparisons))


def compute_comparison(root,rows,protocol,indexed=False):
    tag = 'indexed-cpu' if indexed else 'cpu'
    modes = tuple(f'{mode}-{tag}' for mode in ('dense','fixed'))
    name = 'compute-indexed-results.json' if indexed else 'compute-results.json'
    controls = read(root/name)
    assert {(r['mode'],r['seed']) for r in controls}==set(product(modes,SEEDS))
    expected = {(task,gap):y for task,gap,_,y in cells(protocol['holdout_partition'],TEST_GAPS)}
    regular = {(r['mode'],r['seed']):r for r in rows}
    budget = ({r['seed']:r for r in read(root/'indexed-results.json')} if indexed
              else {r['seed']:r for r in group(rows,'prefix')})
    limits = []
    for r in controls:
        audit_holdout(root,r,expected)
        source = regular[r['mode'].removesuffix('-'+tag),r['seed']]
        reference = budget[r['seed']]
        assert r['source_sha256']==source['weights_sha256'] and r['active']==source['active']
        assert r['train_sha256']==source['train_sha256']==reference['train_sha256']
        assert r['base_cpu_seconds']==source['training_cpu_seconds']
        assert r['target_cpu_seconds']==reference['training_cpu_seconds']
        actual = r['base_cpu_seconds']+r['extra_cpu_seconds']
        assert actual+1e-8>=r['target_cpu_seconds']
        limits.append((r['seed'],LABEL[source['mode']],f'{r["base_cpu_seconds"]:.2f}',
                       f'{r["target_cpu_seconds"]:.2f}',f'{actual:.2f}',r['extra_steps'],
                       f'{joint(regular["prefix",r["seed"]]["holdout"],"lookup",TRAIN_GAPS):.2%}',
                       f'{joint(r["holdout"],"lookup",TRAIN_GAPS):.2%}'))
    for seed in (() if indexed else SEEDS):
        replay = TrainingCalibration(protocol.get('calibration_groups_per_cell'))
        rng = torch.Generator().manual_seed(245287+seed)
        for step in range(1,protocol['pretrain_steps']+protocol['continuation_steps']+1):
            lookup,x,y,_ = capacity_batch(seed,step,rng)
            if step>protocol['pretrain_steps']:
                replay.add(lookup,x,y)
        for mode in ('prefix','prefix-decay'):
            saved = torch.load(root/f'{mode}-{seed}'/'calibration.pt',weights_only=True)
            assert set(saved)==set(replay.batches)
            for key,pair in saved.items():
                assert all(torch.equal(a,b) for a,b in zip(pair,replay.batches[key],strict=True))
            for e in regular[mode,seed]['events']:
                if 'guard' in e:
                    expected_count = sum(y.numel() for task in replay.data(e['step']) for _,y in task)
                    assert e['guard']['sequences_per_check']==expected_count
    metrics = (('recall',TRAIN_GAPS),('lookup',TRAIN_GAPS),('recall',(512,)),('lookup',(512,)))
    selected = group(rows,'prefix')
    comparisons = [(LABEL['prefix'],selected),*((LABEL[m],group(controls,m)) for m in modes)]
    title = '按索引优化后的 CPU 预算重配对照' if indexed else '复用训练题并匹配实际训练计算'
    return (f'<h2>{title}</h2><p>数量校准保留每项任务和每档延迟最近的训练批次。'
            '全连接和固定随机从各自同更新数的检查点继续，恢复 AdamW 的完整状态，在同一批已见训练题上补做权重更新。'
            '预算是分叉后的单线程进程 CPU 时间，计入反传、选边和校准，排除出题、验证及读写。最后一次更新可能略超预算。</p>'
            +text_table(('种子','对照','原训练秒','演化目标秒','对照实际秒','额外更新','演化查值','对照查值'),limits)
            +table(('模型','训练内回忆','训练内查值','512 步回忆','512 步查值'),[
                [(name,''),*[mean_range([joint(r['holdout'],t,g) for r in group_rows]) for t,g in metrics]]
                for name,group_rows in comparisons]))


def paired_interval(paired):
    rng = torch.Generator().manual_seed(918121)
    indices = torch.randint(paired.shape[1],(len(paired),2048,paired.shape[1]),generator=rng)
    draws = paired.gather(1,indices.flatten(1)).reshape(indices.shape).mean(dim=(0,2))
    low,high = torch.quantile(draws,torch.tensor([.025,.975])).tolist()
    return 100*float(paired.mean()),100*low,100*high


def paired_lookup(root,selected,has_blind,has_compute=False,has_previous=False,has_indexed_compute=False,
                  credit_modes=(),previous_label=None,task='lookup'):
    modes = ('dense','binary','fixed','blind') if has_blind else ('dense','binary','fixed')
    if selected=='prefix':
        modes += ('guarded','prefix-decay')
    if has_compute:
        modes += ('dense-cpu','fixed-cpu')
    if has_previous:
        modes += ('previous',)
    if has_indexed_compute:
        modes += ('dense-indexed-cpu','fixed-indexed-cpu')
    modes += tuple(credit_modes)
    values = {}
    for mode in (selected,*modes):
        for seed in SEEDS:
            raw = read(root/f'{mode}-{seed}'/'holdout.json')
            for cell in raw:
                if cell['task']==task:
                    predictions = torch.tensor(cell['probabilities']).argmax(-1)
                    values[mode,seed,cell['gap']] = (predictions==torch.tensor(cell['targets'])).all(1).float()
    comparisons = []
    for title,gaps in (('训练延迟',TRAIN_GAPS),('512 步',(512,))):
        for mode in modes:
            paired = torch.stack([torch.stack([values[selected,s,g]-values[mode,s,g] for s in SEEDS]).mean(0)
                                  for g in gaps])
            delta,low,high = paired_interval(paired)
            label = previous_label if mode=='previous' and previous_label else LABEL.get(mode,'按生存值断边')
            comparisons.append((title,label,f'{delta:+.2f}',f'{low:+.2f} 至 {high:+.2f}'))
    title = '查值' if task=='lookup' else '回忆'
    return (f'<h3>{title}成绩的配对差</h3><p>在每个延迟内重采样相同留出题组，三个已训练种子共享抽取索引。'
            '区间衡量这些种子下的题目抽样误差。</p>'
            +text_table(('延迟',LABEL[selected]+'的对照','均值差（百分点）','95% 区间（百分点）'),comparisons))


def credit_ablation(root,primary,partition):
    protocol = read(root/'credit-protocol.json')
    rows = read(root/'credit-results.json')
    variants = protocol['credit_variants']
    assert {(r['mode'],r['seed']) for r in rows}==set(product(variants,SEEDS))
    expected = {(task,gap):y for task,gap,_,y in cells(partition,TEST_GAPS)}
    for row in rows:
        audit_holdout(root,row,expected)
        base = next(r for r in primary if r['seed']==row['seed'])
        assert all(row[k]==base[k] for k in ('source_sha256','train_sha256'))
        assert row['history'][0]==base['history'][0]
        events = read(root/f'{row["mode"]}-{row["seed"]}'/'topology.json')
        assert WIDTH**2+sum(e['woke']-e['slept'] for e in events)==row['active']
    labels = {'prefix':'原始 BPTT 评分','prefix-traces':'时间常数 2、16、64 的资格迹',
              'prefix-dynamics':'15% 扰动响应评分','prefix-no-dynamics':'移除扰动响应评分'}
    if 'prefix-no-dynamics' in variants:
        labels.update(prefix='15% 扰动响应评分',**{'prefix-traces':'保留扰动并加入资格迹'})
    table_rows = []
    for mode in ('prefix',*variants):
        model = primary if mode=='prefix' else group(rows,mode)
        metrics = (('recall',TRAIN_GAPS),('lookup',TRAIN_GAPS),('recall',(512,)),('lookup',(512,)))
        table_rows.append([(labels[mode],''),mean_range([1-r['active']/WIDTH**2 for r in model]),
                           *[mean_range([joint(r['holdout'],t,g) for r in model]) for t,g in metrics]])
    factors = ('各对照相对主版本只改变一个因素。移除扰动版保留原反传评分，加入资格迹版保留 15% 扰动并加权不同时距的梯度。'
               if 'prefix-no-dynamics' in variants else '资格迹和扰动响应分别启用，分别改变时间梯度或活跃边收益。')
    return ('<h2>单独改变连接的收益评分</h2><p>所有候选从同一检查点继续，校准、休眠规则和权重更新次数一致。'
            +factors+'</p>'
            +table(('方案','减少的边','训练内回忆','训练内查值','512 步回忆','512 步查值'),table_rows))


def calibration_size(root,primary,partition,dynamics=False,consolidation=None):
    reference = read(root/'previous-results.json')
    assert {r['seed'] for r in reference}==set(SEEDS)
    expected = {(task,gap):y for task,gap,_,y in cells(partition,TEST_GAPS)}
    output = []
    for seed in SEEDS:
        light = next(r for r in primary if r['seed']==seed)
        full = next(r for r in reference if r['seed']==seed)
        assert all(light[k]==full[k] for k in ('source_sha256','train_sha256'))
        checkpoint = Path(full['checkpoint_root'])/f'prefix-{seed}'/'weights.pt'
        audit_holdout(root,full,expected,checkpoint)
        labels = ('提前固定拓扑','持续演化') if consolidation else ('加入扰动评分','原始评分') if dynamics else ('40 条','192 条')
        for label,row,folder in ((labels[0],light,root/f'prefix-{seed}'),(labels[1],full,checkpoint.parent)):
            events = read(folder/'topology.json')
            forwards = sum((1+e['guard']['checks'])*e['guard']['sequences_per_check']
                           for e in events if 'guard' in e)
            output.append((seed,label,f'{forwards:,}',f'{row["training_cpu_seconds"]:.2f}',
                           f'{1-row["active"]/WIDTH**2:.1%}',f'{joint(row["holdout"],"lookup",TRAIN_GAPS):.2%}',
                           f'{joint(row["holdout"],"lookup",(512,)):.2%}'))
    intro = ('<h2>扰动评分的独立复验</h2><p>同一初始检查点、训练流和轻量校准，只有活跃边收益评分不同。'
             if dynamics else '<h2>校准样本量的直接对照</h2><p>同一初始检查点和训练流，分别用完整批次与轻量题组校准。')
    if consolidation:
        intro = (f'<h2>拓扑固定时点的对照</h2><p>两个版本在第 {consolidation} 次更新前的结构事件逐次相同。'
                 '提前固定组继续学习权重，另一个版本保持原来的拓扑更新时点。')
    return (intro+
            '两轮检查点统一重测这次的留出题。CPU 时间来自各自训练时的进程计数。</p>'
            +text_table(('种子','方案','校准序列累计','分叉后 CPU 秒','活跃边减少','训练内查值','512 步查值'),output))


def indexed_optimizer(root,primary):
    rows = read(root/'indexed-results.json')
    assert {r['seed'] for r in rows}==set(SEEDS)
    results = []
    for row in rows:
        seed = row['seed']
        base = next(r for r in primary if r['seed']==seed)
        assert all(base[k]==row[k] for k in ('train_sha256','source_sha256','history','active'))
        old,new = root/f'prefix-{seed}',root/f'prefix-indexed-{seed}'
        assert read(old/'topology.json')==read(new/'topology.json')
        for name in ('weights.pt','auxiliary.pt'):
            previous = torch.load(old/name,weights_only=True)
            updated = torch.load(new/name,weights_only=True)
            assert set(previous)==set(updated)
            assert all(torch.equal(previous[key],updated[key]) for key in previous)
        a,b = base['training_cpu_seconds'],row['training_cpu_seconds']
        results.append((seed,f'{a:.2f}',f'{b:.2f}',f'{1-b/a:+.1%}','逐元素相同'))
    return ('<h2>新生边优化器的等价复跑</h2><p>新生边使用一次索引定位，原来对每个张量分别扫描布尔掩码。'
            '三个种子均复跑完整训练，检查每次结构事件和最终张量。两轮的进程 CPU 时间来自各自并行运行，只作实测参考。</p>'
            +text_table(('种子','原版 CPU 秒','索引版 CPU 秒','用时减少','最终张量'),results))


def adaptation(root,modes,selected_mode):
    rows = read(root/'adaptation.json')
    frozen_mode = selected_mode+'-frozen'
    extensions = (frozen_mode,'prefix-no-wake') if selected_mode=='prefix' else (frozen_mode,)
    for seed in SEEDS:
        selected = {r['mode']:r for r in rows if r['seed']==seed}
        assert set(selected)==set((*modes,*extensions))
        assert len({r['train_sha256'] for r in selected.values()})==1
        live,frozen = selected[selected_mode],selected[frozen_mode]
        assert live['source_sha256']==frozen['source_sha256'] and live['points'][0]==frozen['points'][0]
        assert not frozen['updates']
        if selected_mode=='prefix':
            no_wake = selected['prefix-no-wake']
            assert live['source_sha256']==no_wake['source_sha256'] and live['points'][0]==no_wake['points'][0]
            assert all(not e['woke'] for e in no_wake['updates'])
    return table(('模型','开始','80 次','160 次','320 次','480 次','原任务回忆'),[
        [(LABEL[m],''),*[mean_range([joint(r['points'][i]['lookup'],'lookup',TRAIN_GAPS) for r in group(rows,m)])
                         for i in range(5)],mean_range([joint(r['points'][-1]['recall'],'recall',TRAIN_GAPS)
                                                        for r in group(rows,m)])]
        for m in (*modes,*extensions)])


def paired_adaptation(root,protocol,compute=False):
    modes = ('prefix','prefix-no-wake','prefix-frozen')
    keys = set(product(modes,SEEDS))
    tag = 'adapt-cpu-final' if compute else 'adapt-final'
    paths = {(mode,seed):root/f'{tag}-{mode}-{seed}.json' for mode,seed in keys}
    if not all(path.is_file() for path in paths.values()):
        return ''
    original = {(r['mode'],r['seed']):r for r in read(root/'adaptation.json') if (r['mode'],r['seed']) in keys}
    assert set(original)==keys
    rows = original
    audit = root/'adaptation-audit.json'
    if not compute and audit.is_file():
        repeated = {(r['mode'],r['seed']):r for r in read(audit)}
        if set(repeated)!=keys:
            return ''
        assert all(repeated[k]=={key:value for key,value in rows[k].items()
                                 if key not in ('cpu_seconds','state_sha256')} for k in keys)
    detail = ''
    if compute:
        rows = {(r['mode'],r['seed']):r for r in read(root/'adapt-compute-results.json')}
        assert set(rows)==keys
        budgets = {seed:max(original[mode,seed]['cpu_seconds'] for mode in modes) for seed in SEEDS}
        details = []
        for mode,seed in sorted(keys):
            a,b = rows[mode,seed],original[mode,seed]
            assert all(a[k]==b[k] for k in ('train_sha256','state_sha256'))
            assert a['target_cpu_seconds']==budgets[seed] and a['base_cpu_seconds']==b['cpu_seconds']
            used = a['base_cpu_seconds']+a['extra_cpu_seconds']
            assert used+1e-8>=budgets[seed]
            assert a['active']==b['points'][-1]['active']+sum(e['woke']-e['slept'] for e in a['updates'])
            if not a['extra_steps']:
                assert all(a[k]==b['points'][-1][k] for k in ('lookup','recall','active'))
            if mode=='prefix-frozen':
                assert not a['updates']
            elif mode=='prefix-no-wake':
                assert all(e['woke']==0 for e in a['updates'])
            details.append((seed,LABEL[mode],f'{used:.2f}',a['extra_steps'],
                            f'{joint(a["lookup"],"lookup",TRAIN_GAPS):.2%}',
                            f'{joint(a["lookup"],"lookup",(64,)):.2%}'))
        detail = ('<p>每个种子以三种策略完成 480 次更新时用时最大的进程 CPU 秒为目标。'
                  '其余策略恢复该时点的权重和优化器状态，在同一批已见训练题上继续更新。</p>'
                  +text_table(('种子','策略','总 CPU 秒','追加更新','训练内查值','64 步查值'),details))
    partition = protocol.get('adaptation_partition','prefix-adaptation')
    labels = {gap:y for task,gap,_,y in cells(partition,(*TRAIN_GAPS,64),n=96,rule=1) if task=='lookup'}
    hit = {}
    for (mode,seed),path in paths.items():
        summary = rows[mode,seed]['lookup'] if compute else rows[mode,seed]['points'][-1]['lookup']
        for c,b in zip(read(path),summary,strict=True):
            assert c['gap']==b['gap'] and c['targets']==labels[c['gap']].tolist()
            correct = torch.tensor(c['probabilities']).argmax(-1)==labels[c['gap']]
            assert float(correct[:,0].float().mean())==b['single']
            hit[mode,seed,c['gap']] = correct.all(1).float()
            assert float(hit[mode,seed,c['gap']].mean())==b['joint']
    out = []
    for label,gaps in (('训练延迟',TRAIN_GAPS),('64 步',(64,))):
        for mode in modes[1:]:
            seed_rows = torch.stack([torch.stack([hit['prefix',seed,g]-hit[mode,seed,g] for g in gaps])
                                     for seed in SEEDS])
            delta,low,high = paired_interval(seed_rows.mean(0))
            seeds = [f'{100*float(row.mean()):+.2f}' for row in seed_rows]
            out.append((label,LABEL[mode],f'{delta:+.2f}',f'{low:+.2f} 至 {high:+.2f}',*seeds))
    title = '等 CPU 预算的规则恢复' if compute else '同一检查点的规则恢复配对差'
    return (f'<h3>{title}</h3>'+detail+'<p>用逐题概率重算联合正确率。'
            '继续再生与两个对照共享题目和初始权重。区间只反映三个训练种子下的题组抽样误差。</p>'
            +text_table(('延迟','继续再生的对照','均值差（百分点）','95% 区间（百分点）',*SEEDS),out))


def cuts(root,selected):
    rows = read(root/'cuts.json')
    return table(('模型','任务','完整','定向断边','匹配随机断边'),[
        [(LABEL[m],''),(title,''),*[mean_range(values) for values in (
            [joint(r['base'],task,(4,16,32)) for r in group(rows,m)],
            [joint(r['targeted'],task,(4,16,32)) for r in group(rows,m)],
            [mean(joint(control,task,(4,16,32)) for control in r['random']) for r in group(rows,m)])]]
        for m,(task,title) in product(('dense','fixed',selected),(('lookup','查值'),('recall','回忆')))])


def time_cuts(root,selected):
    rows = read(root/'time-cuts.json')
    models = ('dense','fixed',selected)
    decay = root/'time-cuts-decay.json'
    if decay.is_file():
        candidates = read(decay)
        if len(candidates)==len(SEEDS):
            rows += candidates
            models += ('prefix-decay',)
    assert {(r['mode'],r['seed']) for r in rows}==set(product(models,SEEDS))
    for row in rows:
        masks = [torch.bincount(torch.tensor(indices)//WIDTH,minlength=WIDTH) for indices in row['indices']]
        assert all(torch.equal(m,masks[0]) for m in masks)
        assert int(masks[0].sum())==row['count']
    out = []
    for mode,(label,gaps) in product(models,(('短延迟',(4,32)),('长延迟',(256,512)))):
        data = group(rows,mode)
        base = [joint(r['base'],'recall',gaps) for r in data]
        long = [joint(r['long'],'recall',gaps) for r in data]
        short = [joint(r['short'],'recall',gaps) for r in data]
        random = [mean(joint(c,'recall',gaps) for c in r['random']) for r in data]
        difference = 100*mean(s-l for s,l in zip(short,long,strict=True))
        out.append([(LABEL[mode],''),(label,''),*[mean_range(v) for v in (base,long,short,random)],
                    (f'{difference:+.2f}','')])
    return ('<h2>短延迟和长延迟是否依赖不同连接</h2><p>在独立的 4 步和 128 步回忆题上比较归一化梯度敏感度。'
            '按两者的差值选出偏长或偏短的边，每个神经元断开相同数量的活跃入边，约为活跃入边的 1%。'
            '两种定向掩码互斥，随机掩码重复五次。最后一列为断长边比断短边多损失的百分点。'
            '短延迟取 4、32 步，长延迟取 256、512 步。</p>'
            +table(('模型','留出条件','完整','断偏长边','断偏短边','随机断边','断偏长边多掉分'),out))


def modules(root,selected):
    rows = read(root/'modules.json')
    assert {(r['mode'],r['seed']) for r in rows}==set(product(('dense','fixed',selected),SEEDS))
    data = []
    for row in rows:
        assert sorted(i for g in row['groups'] for i in g)==list(range(WIDTH))
        d = row['selection']
        focus = 'lookup' if d['lookup']>=d['recall'] else 'recall'
        other = 'recall' if focus=='lookup' else 'lookup'
        targeted = row['targeted'][focus]-row['targeted'][other]
        control = mean(v[focus]-v[other] for v in row['random'])
        data.append((LABEL[row['mode']],row['seed'],len(row['groups']),len(row['groups'][row['selected']]),
                     f'{row["modularity"]:.3f}',f'{mean(row["null_modularity"]):.3f}',
                     '查值' if focus=='lookup' else '回忆',f'{100*targeted:+.1f}',f'{100*control:+.1f}',
                     f'{row["query_contrast"]:.4f}',f'{mean(row["random_query_contrast"]):.4f}'))
    return ('<h2>网络社群与选择性失活</h2><p>按循环权重绝对值自动发现加权社群。零对照随机打乱各神经元的入边来源，'
            '保留该神经元的入边权重分布。用 4、16 步开发题选择两任务受损差异最大的社群，在独立留出上与八组同大小的随机神经元对比。'
            '失活差为目标任务掉分减另一任务掉分。响应离散度衡量同一历史改查询时，社群平均状态变化之间的标准差。'
            '查询响应可由输入变化产生，条件路由需要进一步的因果证据。</p>'
            +text_table(('模型','种子','社群数','失活规模','模块度','打乱权重模块度','目标任务','社群失活差（百分点）',
                         '随机失活差（百分点）','社群响应离散度','随机分组离散度'),data))


def resources(root,modes):
    rows = [read(root/f'profile-{m}.json') for m in modes]
    return text_table(('模型','输入步','掩码推理壁钟每题','每步','掩码推理 CPU 每题','峰值增量',
                       '压缩矩阵壁钟每题','压缩矩阵 CPU 每题','一次性打包'),[
        (LABEL[r['mode']],r['tokens'],f'{r["masked_ms"]:.2f} 毫秒',f'{1000*r["masked_ms"]/r["tokens"]:.1f} 微秒',
         f'{r["masked_cpu_ms"]:.2f} 毫秒' if 'masked_cpu_ms' in r else '—',
         f'{r["masked_peak_increment_mib"]:.2f} MiB',
         f'{r["csr"]["ms"]:.2f} 毫秒' if r['csr'].get('matches') else '无通过校验的数据',
         f'{r["csr"]["cpu_ms"]:.2f} 毫秒' if r['csr'].get('matches') and 'cpu_ms' in r['csr'] else '—',
         f'{r["csr"]["build_ms"]:.2f} 毫秒' if r['csr'].get('matches') else '—') for r in rows])


def replay_table(root):
    rows = read(root/'replay.json')
    assert {r['seed'] for r in rows}==set(SEEDS)
    result = []
    deltas = {19:[],39:[]}
    for r in rows:
        immediate,paced,frozen = r['arms']
        expected = next(e for e in read(root/f'binary-{r["seed"]}'/'topology.json') if e['step']==r['step'])
        assert immediate['mask_sha256']==paced['mask_sha256']
        assert immediate['slept']==paced['slept']==expected['slept']
        assert immediate['woke']==paced['woke']==expected['woke']
        assert immediate['points'][0]['active']==expected['active']
        assert immediate['points'][3]['active']==paced['points'][3]['active']
        assert frozen['slept']==frozen['woke']==0
        for before,after in zip(immediate['points'],paced['points'],strict=True):
            offset = before['offset']
            assert offset==after['offset']
            if offset in deltas:
                deltas[offset].append(joint(after['scores'],'lookup',TRAIN_GAPS)
                                      -joint(before['scores'],'lookup',TRAIN_GAPS))
        for arm in r['arms']:
            before = joint(r['before'],'lookup',TRAIN_GAPS)
            values = [joint(p['scores'],'lookup',TRAIN_GAPS) for p in arm['points']]
            result.append((r['seed'],r['step'],LABEL[arm['mode']],f'{arm["slept"]:,}',f'{before:.2%}',
                           *(f'{v:.2%}' for v in values)))
    return ('<h2>同一批连接在 20 步内分散断开</h2><p>从三个种子各自第一次至少断开 5,000 条边的检查点分叉。权重、优化器、再生边和后续输入完全匹配。'
            '到第 19 次后续更新，两种时序的活跃边掩码逐位一致。冻结组在此窗口内也训练权重。表格为训练延迟范围内的开发题联合正确率。</p>'
            f'<p>分批相对集中在相同掩码时平均变化 {100*mean(deltas[19]):+.2f} 个百分点。'
            f'再训练 20 次后的变化为 {100*mean(deltas[39]):+.2f} 个百分点。</p>'
            +text_table(('种子','触发更新','方案','该批断开数','更新前','当步','第 4 次','第 9 次','第 19 次','第 39 次'),result))


def report(root):
    rows = read(root/'results.json')
    protocol = read(root/'protocol.json')
    arms = tuple(protocol['arms'])
    modes = (*arms,'fixed')
    pacing = 'sleep_window' in protocol
    guard = protocol.get('sleep_guard')
    selected = protocol.get('selected','paced' if pacing else 'binary')
    dynamics = bool(protocol.get('credit_variants',{}).get(selected,{}).get('dynamics'))
    consolidation = protocol.get('consolidate_after')
    partition = protocol.get('holdout_partition','conductance-holdout')
    audit(root,rows,arms,selected,partition)
    legend = '　'.join(f'<b style="color:{COLOR[m]}">● {LABEL[m]}</b>' for m in arms)
    reductions = [1-r['active']/WIDTH**2 for r in group(rows,selected)]
    lookup_delta = mean(joint(r['holdout'],'lookup',TRAIN_GAPS) for r in group(rows,selected))-mean(
        joint(r['holdout'],'lookup',TRAIN_GAPS) for r in group(rows,'dense'))
    recall_delta = mean(joint(r['holdout'],'recall',TRAIN_GAPS) for r in group(rows,selected))-mean(
        joint(r['holdout'],'recall',TRAIN_GAPS) for r in group(rows,'dense'))
    title = '集中断边与分批断边' if pacing else '把循环边的减弱与断开分开测试'
    schedule = ('<p>每 20 次权重更新评估一次生存值。集中组立即断开所有候选，分批组把同批候选依生存值排序，'
                '均摊到本次及后续 19 次更新。等待的边保持全强度，休眠和再生门槛不变。再生边即时接通。'
                '两组最后一次结构评分在第 1,180 次更新，以完整执行最后一批。</p>') if pacing else ''
    if guard:
        title = '让校准决定每次断多少条循环边' if selected=='prefix' else '用配对题损失筛选待断循环边'
        recall,lookup = guard['tolerance']
        schedule = (f'<p>每 20 次权重更新评估生存值。校准组额外抽取每任务、每档 {guard["samples_per_gap"]} 个题组。'
                    f'回忆和查值允许的每答案交叉熵增量分别为 {recall} 和 {lookup}。再生规则照常运行。</p>')
    if protocol.get('reuse_calibration'):
        title = '复用训练批次的拓扑演化'
        schedule = ('<p>数量校准组从已经参与权重训练的批次中复用题目。逐组筛边对照仍抽取独立的校准题。'
                    '切换规则后，数量校准组使用最近的新规则查值批次，回忆校准继续使用原训练中已见的批次。</p>')
        if protocol.get('calibration_groups_per_cell'):
            schedule += '<p>每次校准从每项任务和每档延迟轮取一个完整题组，共 40 条输入序列。</p>'
        if dynamics:
            title = '轻量校准下的扰动响应评分'
            schedule += '<p>数量校准的两个主实验在活跃边收益中混入 15% 的扰动响应分数。</p>'
        if consolidation:
            title = '提前固定拓扑，让权重继续学习'
            schedule += f'<p>原任务第 {consolidation} 次更新后，数量校准组固定循环掩码并继续优化权重。</p>'
    status = lambda ok: '达标' if ok else '未达标'
    summary = table(('模型','训练内查值','训练内回忆','最长档查值','活跃边范围','减少比例'),[
        [(LABEL[m],''),*[mean_range([joint(r['holdout'],task,gaps) for r in group(rows,m)])
                         for task,gaps in (('lookup',TRAIN_GAPS),('recall',TRAIN_GAPS),('lookup',(512,)))],
         (f'{min(r["active"] for r in group(rows,m)):,}–{max(r["active"] for r in group(rows,m)):,}',''),
         mean_range([1-r['active']/WIDTH**2 for r in group(rows,m)])] for m in modes])
    by_gap = ''.join(f'<h3>{title}</h3>'+table(('模型',*TEST_GAPS),[
        [(LABEL[m],''),*[mean_range([joint(r['holdout'],task,(g,)) for r in group(rows,m)]) for g in TEST_GAPS]]
        for m in modes]) for task,title in (('lookup','四记录查值'),('recall','延迟回忆')))
    has_blind = guard and (root/'blind-results.json').is_file()
    has_credit = (selected=='prefix' and (root/'credit-results.json').is_file()
                  and len(read(root/'credit-results.json'))==2*len(SEEDS))
    credit_modes = tuple(read(root/'credit-protocol.json')['credit_variants']) if has_credit else ()
    has_previous = (root/'previous-results.json').is_file() and len(read(root/'previous-results.json'))==len(SEEDS)
    has_indexed = (root/'indexed-results.json').is_file() and len(read(root/'indexed-results.json'))==len(SEEDS)
    has_indexed_compute = (root/'compute-indexed-results.json').is_file() and len(read(root/'compute-indexed-results.json'))==2*len(SEEDS)
    has_modules = (root/'modules.json').is_file() and len(read(root/'modules.json'))==3*len(SEEDS)
    previous_label = '持续演化的扰动评分' if consolidation else '轻量校准的原始评分' if dynamics else None
    compute_card = ''
    if protocol.get('reuse_calibration'):
        name = 'compute-indexed-results.json' if has_indexed_compute else 'compute-results.json'
        mode = 'dense-indexed-cpu' if has_indexed_compute else 'dense-cpu'
        equal_cpu = group(read(root/name),mode)
        delta = mean(joint(r['holdout'],'lookup',TRAIN_GAPS) for r in group(rows,selected))-mean(
            joint(r['holdout'],'lookup',TRAIN_GAPS) for r in equal_cpu)
        compute_card = (f'<div class="card"><strong>{100*delta:+.2f} 个百分点</strong>'
                        f'等计算查值均值相对全连接，{status(delta>=-.01)}</div>')
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style><main>
<h1>{title}</h1>
<div class="cards"><div class="card"><strong>{min(reductions):.1%}–{max(reductions):.1%}</strong>活跃边减少，25% 门槛{status(min(reductions)>=.25)}</div>
<div class="card"><strong>{lookup_delta*100:+.2f} 个百分点</strong>训练内查值相对全连接，{status(lookup_delta>=-.01)}</div>
<div class="card"><strong>{recall_delta*100:+.2f} 个百分点</strong>训练内回忆相对全连接，{status(recall_delta>=-.01)}</div>{compute_card}</div>
<p>512 个神经元。三个种子分别从同一个已完成 4,400 次更新的全连接检查点分叉，再训练 1,200 次。分叉时各组都重置 AdamW。固定随机组从头训练，边数与同种子的{LABEL[selected]}组最终边数相同。固定组也在第 4,400 次更新后重置优化器。课程查值训练沿用中间状态的辅助读出，评测只使用末尾查询输出。</p>
{schedule}
{summary}<p>每个成绩显示三个种子的均值及最小到最大值。联合正确率要求同组反事实均回答正确。四记录查值和延迟回忆分别统计。</p>
{replay_table(root) if pacing else ''}
{guard_audit(rows,guard,protocol.get('reuse_calibration',False)) if guard else ''}
{compute_comparison(root,rows,protocol) if protocol.get('reuse_calibration') else ''}
{calibration_size(root,group(rows,selected),partition,dynamics,consolidation) if has_previous else ''}
{indexed_optimizer(root,group(rows,selected)) if has_indexed else ''}
{compute_comparison(root,rows,protocol,indexed=True) if has_indexed_compute else ''}
{blind_control(root,group(rows,'guarded'),partition) if has_blind else ''}
<h2>训练中的变化</h2><p>{legend}</p>
<h3>训练范围查值联合正确率</h3>{curve(rows,arms,lambda h:joint(h['scores'],'lookup',TRAIN_GAPS),'查值正确率随训练变化')}
<h3>活跃循环边比例</h3>{curve(rows,arms,lambda h:h['active']/WIDTH**2,'活跃边比例随训练变化')}
<h3>断开和再生规模</h3>{topology_totals(rows,arms)}
<h3>单次结构更新前后，查值降幅最大的记录</h3><p>同一批开发题在权重更新后测两次，中间仅执行一次结构更新。每隔 200 次权重更新采样，表中仅展示已采样事件。</p>{changes(rows,arms)}
<h2>独立留出题的各档成绩</h2>{by_gap}
{paired_lookup(root,selected,has_blind,protocol.get('reuse_calibration',False),has_previous,has_indexed_compute,credit_modes,previous_label) if guard else ''}
{paired_lookup(root,selected,False,has_compute=True,task='recall') if consolidation else ''}
{credit_ablation(root,group(rows,selected),partition) if has_credit else ''}
<h2>换查询规则后能否恢复</h2><p>查询改为读取下一个键。训练 480 次，冻结对照从同一检查点开始，仍然学习权重。表内为训练延迟范围内的联合正确率。</p>{adaptation(root,modes,selected)}
{paired_adaptation(root,protocol) if selected=='prefix' else ''}
{paired_adaptation(root,protocol,compute=True) if (root/'adapt-compute-results.json').is_file() else ''}
<h2>断边干预</h2><p>用 16 步开发题按查值相对回忆的梯度敏感性选 1% 活跃边。随机组每个神经元断开同样多的入边，抽五组。结果为 4、16、32 步留出题均值。不同任务的起点需同时考虑。</p>{cuts(root,selected)}
{time_cuts(root,selected) if selected=='prefix' else ''}
{modules(root,selected) if has_modules else ''}
<h2>实际推理开销</h2><p>种子 2，单线程 CPU，批量 1，各模型单独进程，同一 512 步延迟查值题。壁钟时间取中位数，CPU 时间取批内摊销均值。峰值增量以加载权重后的进程驻留内存为基线。压缩矩阵耗时复用一次打包结果。</p>{resources(root,modes)}
<h2>核对</h2><p>同一种子各组训练流散列一致。每个检查点按保存的散列校验，活跃边计数由断开和再生逐次核对，留出标签从生成器重建，联合与原题正确率从保存概率重算。</p></main></html>'''
    (root/'report-cn.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
