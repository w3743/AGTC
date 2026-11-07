"""Chinese report for the lookup curriculum and new-memory acquisition tests."""

import argparse
from itertools import product
from pathlib import Path
from statistics import mean

from evolving_rnn_eval import TRAIN_GAPS
from frontier_eval import SEEDS, read
from shared_address_report import CSS, heat, mean_range, table


LABELS = {
    'queries':'同历史四问',
    'prefix-readout':'查询前辅助读出',
    'ranked':'正确答案排序约束',
    'query-ticks':'增加三步查询计算',
    'write-readout':'每次写入后辅助读出',
    'packed-records':'成对输入与写后读出',
    'retention-readout':'原输入加干扰期读出',
    'packed-retention':'成对输入加干扰期读出',
    'identity-retention':'原输入加干扰期读出，近恒等初值',
    'frozen':'冻结拓扑',
    'evolve':'继续演化',
}
COLORS = dict(evolve='#00634f',frozen='#7041ab')


def lookup_scores(root_list):
    score_rows,diagnostic_rows = [],[]
    all_runs = []
    for root in root_list:
        protocol = read(root/'protocol.json')
        runs = read(root/'probes.json')
        assert {(r['probe'],r['seed']) for r in runs}==set(product(protocol['probes'],SEEDS))
        all_runs.extend(runs)
        for probe in protocol['probes']:
            selected = [r for r in runs if r['probe']==probe]
            rows = [r['history'][-1]['scores'] for r in selected]
            values = [[next(c['joint'] for c in series if c['task']=='lookup' and c['gap']==gap)
                       for series in rows] for gap in (*TRAIN_GAPS,64)]
            score_rows.append([(LABELS[probe],'')]+[mean_range(v) for v in values])
        path = root/'binding-diagnosis.json'
        if path.exists():
            measured = read(path)
            for probe in protocol['probes']:
                group = [r for r in measured if r['probe']==probe]
                for gap in (0,32):
                    rows = [c for r in group for c in r['by_gap'] if c['gap']==gap]
                    aux = [c['auxiliary_single'] for c in rows if c['auxiliary_single'] is not None]
                    diagnostic_rows.append([(LABELS[probe],''),(gap,''),mean_range([c['single'] for c in rows]),
                                            mean_range([c['all_queries'] for c in rows]),
                                            mean_range([c['prediction_changes'] for c in rows]),
                                            mean_range(aux) if aux else ('—','')])
    for seed in SEEDS:
        assert len({r['train_sha256'] for r in all_runs if r['seed']==seed})==1
    return (table(('训练方式','0 步','4 步','16 步','32 步','64 步'),score_rows),
            table(('训练方式','干扰步数','单查询正确','同历史四问全对','不同答案时预测改变','辅助解码正确'),
                  diagnostic_rows) if diagnostic_rows else '')


def acquisition_plot(runs,old=False):
    axis = 'old' if old else 'novel'
    title = '旧线索保持' if old else '新线索长延迟学习'
    x = lambda step:60+780*step/480
    y = lambda score:215-185*score
    svg = [f'<svg viewBox="0 0 900 260" role="img" aria-label="{title}">']
    for v in (0,.25,.5,.75,1):
        yy = y(v)
        svg.append(f'<path d="M60 {yy}H840" stroke="#d1dbe5"/>'
                   f'<text class="axis" x="50" y="{yy+4}" text-anchor="end">{v:.0%}</text>')
    for step in (0,80,160,320,480):
        svg.append(f'<text class="axis" x="{x(step)}" y="241" text-anchor="middle">{step}</text>')
    for mode in ('evolve','frozen'):
        subset = [r for r in runs if r['mode']==mode]
        points = [(x(p[0]['step']),y(mean(c['joint'] for point in p for c in point[axis])))
                  for p in zip(*(r['points'] for r in subset),strict=True)]
        coords = ' '.join(f'{px},{py}' for px,py in points)
        color = COLORS[mode]
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for px,py in points:
            svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
    return ''.join(svg)+'</svg>'


def acquisition(root,staged=False):
    path = root/('adaptation-staged.json' if staged else 'adaptation.json')
    if not path.exists():
        return ''
    runs = read(path)
    if len(runs)!=6:
        return ''
    assert {(r['mode'],r['seed']) for r in runs}==set(product(('evolve','frozen'),SEEDS))
    for seed in SEEDS:
        same = [r for r in runs if r['seed']==seed]
        for key in ('train_sha256','source_sha256'):
            assert len({r[key] for r in same})==1
        assert same[0]['points'][0]==same[1]['points'][0]
    rows = []
    for mode in ('evolve','frozen'):
        selected = [r for r in runs if r['mode']==mode]
        conditions = [('old','旧线索 0–32 步'),('novel','新线索 64–128 步'),('extra','新线索 256–512 步')]
        if staged:
            conditions.insert(1,('novel_short','新线索 0–32 步'))
        for kind,title in conditions:
            values = []
            for i in range(5):
                channel = kind if kind in ('old','novel_short') else 'novel'
                values.append([mean(c['joint'] for c in r['points'][i][channel]
                                    if channel!='novel' or (c['gap']<=128)==(kind=='novel')) for r in selected])
            rows.append([(LABELS[mode],''),(title,'')]+[mean_range(v) for v in values])
    counts = []
    for r in runs:
        wake = sum(e['woke'] for e in r['topology'])
        sleep = sum(e['slept'] for e in r['topology'])
        counts.append([(LABELS[r['mode']],''),(r['seed'],''),(f'{sleep:,}',''),(f'{wake:,}',''),
                       (f'{r["points"][-1]["active"]:,}','')])
    legend = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in ('evolve','frozen'))
    title = '从短逐步加长' if staged else '直接加入长延迟'
    return f'''<h2>新增记忆需求 · {title}</h2>
    <p>预训练只要求回忆第一条线索，第二条线索一直出现，但与答案无关。适应阶段才开始问第二条线索。
    每个种子的两组用相同初始权重和训练样本。冻结组仍更新权重，生存变量和掩码保持固定。
    每题须同时答对原题、改变相关线索和改变无关线索三个版本。</p>
    {table(('结构','测试条件','起点','80 次','160 次','320 次','480 次'),rows)}
    <div class="legend">{legend}</div>
    <h3>新增线索 · 64 至 512 步联合正确率</h3>{acquisition_plot(runs)}
    <h3>旧线索 · 0 至 32 步联合正确率</h3>{acquisition_plot(runs,True)}
    {table(('结构','种子','累计休眠','累计再生','末尾活跃边'),counts)}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--rounds',nargs='+',type=Path,required=True)
    parser.add_argument('--acquisition',type=Path,required=True)
    args = parser.parse_args()
    scores,diag = lookup_scores(args.rounds)
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>关联查值与新记忆需求</title>
    <style>{CSS}</style><main><p class="eyebrow">自演化稀疏 RNN · 512 神经元</p>
    <h1>多条事实的绑定与新记忆学习</h1>
    <h2>关联查值 · 开发筛选</h2>
    <p>每段历史的四个键都参与训练，并加入只改一处来源的配对历史。训练标签来自实际查询答案。
    配对评测要求原题、换查询、改来源三题全对。辅助头仅用于训练或诊断。
    加三步的版本在读出前重复同一个查询符号。成对输入版本把已分隔的键和值合为一个输入符号。</p>
    {scores}
    <h2>编码和选择的分解检查</h2><p>独立诊断题中，在两个查询正确答案不同的情况下，统计预测是否改变。
    辅助解码检查查询前状态还能恢复多少条记录。</p>{diag}
    {acquisition(args.acquisition)}
    {acquisition(args.acquisition,True)}
    </main></html>'''
    args.output.write_text(doc,encoding='utf-8')


if __name__=='__main__':
    main()
