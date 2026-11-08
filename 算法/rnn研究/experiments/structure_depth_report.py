"""Audit and render the learned routing depth probe as offline Chinese HTML."""

import argparse
from collections import Counter
from hashlib import blake2b, sha256
from itertools import groupby, product
from math import cos, pi, sin
from pathlib import Path
from statistics import mean

from frontier_eval import SEEDS, read
from shared_address_report import CSS, heat, mean_range, table
from structure_depth import BATCH, MODES, PROGRAM, VALUES, resolve


def key(row):
    return row['width'],row['depth'],row['gap']


def validate_cases(cases):
    for r in cases:
        n = r.get('program_size',PROGRAM)
        original,swapped,edited = r['variants']
        q, = [i for i,(a,b) in enumerate(zip(original,swapped,strict=True)) if a!=b]
        e, = [i for i,(a,b) in enumerate(zip(original,edited,strict=True)) if a!=b]
        assert q==len(original)-n+r['branch'] and e<len(original)-n-2
        for seq,target,path in zip(r['variants'],r['targets'],r['paths'],strict=True):
            assert resolve(seq,n)==(target,path)
            assert len(path)==r['depth']+1


def validate_predictions(cases, predictions, cells, trace):
    for (condition,pairs),cell in zip(groupby(zip(cases,predictions,strict=True),lambda p:key(p[0])),cells,strict=True):
        rows = list(pairs)
        assert condition==key(cell) and all(r['id']==p['id'] for r,p in rows)
        assert mean(p['prediction']==r['targets'] for r,p in rows)==cell['joint']
        if trace:
            assert mean(p['hard']==r['targets'] for r,p in rows)==cell['hard_joint']
            assert mean(all(a[:r['depth']+1]==b for a,b in zip(p['trace'],r['paths'],strict=True))
                        for r,p in rows)==cell['trace_exact']
            assert mean(p['hard'][0]==r['targets'][0] and p['hard'][2]==r['targets'][2]
                        and p['frozen']==r['targets'][1] for r,p in rows)==cell['frozen_joint']


def audit(root):
    protocol = read(root/'protocol.json')
    runs = read(root/'results.json')
    cyclic = read(root/'cycles/results.json')
    cases = read(root/'cases.json')
    cycles = read(root/'cycles/cases.json')
    assert {(r['mode'],r['seed']) for r in runs}==set(product(MODES,SEEDS)) and len(runs)==9
    assert {r['seed'] for r in cyclic}==set(SEEDS) and len(cyclic)==3
    validate_cases(cases+cycles)
    prefixes = {tuple(r['variants'][0][:64]) for r in cases+cycles}
    assert len(prefixes)==len(cases)+len(cycles)
    for seed,step,i in product(SEEDS,range(protocol['steps']),range(BATCH//2)):
        identity = f'prefix/structure-depth/v1/train/{seed}/{step}/{i}'
        nonce = blake2b(identity.encode(),digest_size=8).digest()
        prefix = tuple(1+((byte>>bit)&1) for byte in nonce for bit in range(8))
        assert prefix not in prefixes
    for seed in SEEDS:
        assert len({r['training_sha256'] for r in runs if r['seed']==seed})==1
    for r in runs:
        folder = root/f'{r["mode"]}-{r["seed"]}'
        preds = read(folder/'predictions.json')
        validate_predictions(cases,preds,r['cells'],r['mode']!='GRU')
    for r in cyclic:
        checkpoint = root/f'adaptive-{r["seed"]}/weights.pt'
        assert sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
        preds = read(root/f'cycles/predictions-{r["seed"]}.json')
        validate_predictions(cycles,preds,r['cells'],True)
    deep = next(i for i,r in enumerate(cases) if key(r)==(4,8,1024))
    loop = next(i for i,r in enumerate(cycles) if r['depth']==64)
    return (protocol,runs,cyclic,cycles,
            (cases[deep],read(root/'adaptive-2/predictions.json')[deep]),
            (cycles[loop],read(root/'cycles/predictions-2.json')[loop]))


def coordinates(edges, cyclic):
    sources = sorted({src for src,_,_ in edges})
    vertices = sorted(set(sources)|{dst for _,_,dst in edges})
    if cyclic:
        coords = {n:(405+310*cos(i*2*pi/len(sources)),260+204*sin(i*2*pi/len(sources)))
                  for i,n in enumerate(sources)}
        coords.update({n:(805,150+70*i) for i,n in enumerate(v for v in vertices if v not in sources)})
        return coords
    first = {src:dst for src,kind,dst in edges if kind==0}
    groups = {}
    for vertex in vertices:
        rank,current = 0,vertex
        while current in first:
            rank,current = rank+1,first[current]
        groups.setdefault(rank,[]).append(vertex)
    top = max(groups)
    return {n:(48+804*(top-level)/top,75+122*i)
            for level,group in groups.items() for i,n in enumerate(group)}


def edge_svg(case, prediction, variant):
    edges = [tuple(edge) for edge in case['records']]
    cycle = 'program_size' in case
    path = prediction['trace'][variant][:case['depth']+1]
    seq = case['variants'][variant]
    suffix = seq[-case.get('program_size',PROGRAM):]
    selected = Counter((a,token-50,b) for a,token,b in zip(path[:-1],suffix[:case['depth']],path[1:],strict=True))
    if variant==2:
        step = case['branch']
        truth = case['paths'][variant]
        edges = [(s,k,truth[step+1] if (s,k)==(truth[step],suffix[step]-50) else d) for s,k,d in edges]
    coords = coordinates(edges,cycle)
    arrow = f'arrow-{"cycle" if cycle else "layer"}-{variant}'
    svg = ['<svg viewBox="0 0 900 520" role="img" aria-label="原始边和模型逐步选出的边">'
           f'<defs><marker id="{arrow}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
           '<path d="M0 0 L7 3.5 L0 7 Z" fill="context-stroke"/></marker></defs>']
    for src,kind,dst in sorted(edges,key=lambda e:e in selected):
        x1,y1 = coords[src]
        x2,y2 = coords[dst]
        active = selected[src,kind,dst]
        color = ('#00634f','#7041ab')[kind] if active else '#b8c5d4'
        thickness = 2.5+min(active,4)*.4 if active else 1
        if src==dst:
            curve = f'M{x1-8:.1f} {y1-8:.1f} C{x1-36:.1f} {y1-42:.1f} {x1+36:.1f} {y1-42:.1f} {x1+8:.1f} {y1-8:.1f}'
        else:
            dx,dy = x2-x1,y2-y1
            distance = (dx*dx+dy*dy)**.5
            curve = (f'M{x1+13*dx/distance:.1f} {y1+13*dy/distance:.1f} '
                     f'Q{(x1+x2)/2-.10*dy:.1f} {(y1+y2)/2+.10*dx:.1f} '
                     f'{x2-15*dx/distance:.1f} {y2-15*dy/distance:.1f}')
        svg.append(f'<path d="{curve}" fill="none" stroke="{color}" stroke-width="{thickness}" marker-end="url(#{arrow})"/>')
    for node,(x,y) in coords.items():
        fill = '#d8f3e9' if node in path else '#fff'
        svg.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}" stroke="#426075"/>'
                   f'<text class="axis" x="{x}" y="{y+4}" text-anchor="middle">{node}</text>')
    return ''.join(svg)+'</svg>'


def example_html(title, case, prediction):
    anchor = 'cycle-example' if 'program_size' in case else 'layer-example'
    parts = [f'<h2 id="{anchor}">{title}</h2><p>种子 2 的固定首题。灰线是题目给定的其他边，绿线和紫线是模型实际选中的两类边。圆圈内为符号编号。</p>']
    for variant,name in enumerate(('原题','改一条查询指令','改一条边的目的地')):
        path = prediction['trace'][variant][:case['depth']+1]
        repeats = len(path)-len(set(path))
        chips = ''.join(f'<span class="chip"><small>{i}</small> {n}</span>' for i,n in enumerate(path))
        panel = f'{name} · 预测 {prediction["hard"][variant]} · 正确答案 {case["targets"][variant]}'
        parts.append(f'<details{" open" if variant==0 else ""}><summary>{panel}</summary>'
                     f'{edge_svg(case,prediction,variant)}<p>逐步访问路径，重复访问 {repeats} 次。</p>'
                     f'<div class="chips">{chips}</div></details>')
    return ''.join(parts)


def render(root):
    protocol,runs,cyclic,cycles,layer_example,cycle_example = audit(root)
    def values(data,mode,condition,metric):
        return [c[metric] for r in data if r['mode']==mode for c in r['cells'] if key(c)==condition]

    layer_rows = []
    for width,depth in product((2,4),(1,2,4,6,8)):
        condition = width,depth,1024
        layer_rows.append([(f'每层 {width} 个节点 · {depth} 跳',''),(f'{width*depth} 个 / {2*width*depth} 条',''),
                           *[mean_range(values(runs,m,condition,'joint')) for m in MODES],
                           mean_range(values(runs,'adaptive',condition,'trace_exact'))])
    cycle_rows = []
    for depth in (8,16,32,64):
        condition = 32,depth,1024
        revisits = [len(p)-len(set(p)) for r in cycles if r['depth']==depth for p in r['paths']]
        cycle_rows.append([(f'{depth} 跳',''),(f'{min(revisits)}–{max(revisits)} 次',''),
                           *[mean_range(values(cyclic,'adaptive',condition,m)) for m in
                             ('joint','hard_joint','trace_exact','frozen_joint')]])
    cards = ''.join(f'<div class="card">{label}<strong>{value}</strong>{detail}</div>' for label,value,detail in (
        ('层状分支图','8 跳','每层 4 个节点，32 个内部节点、64 条边'),
        ('带环图','64 跳','冻结权重后外推，包含重复访问'),
        ('最深档逐步路径',f'{min(values(cyclic,"adaptive",(32,64,1024),"trace_exact")):.0%}','三个种子中最低的整条路径正确率'),
    ))
    data = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>可学习路由的结构深度</title>
    <style>{CSS}
    details{{border:1px solid #c8d2dc;border-radius:8px;padding:10px;margin:8px 0}}.chips{{display:flex;flex-wrap:wrap;gap:5px}}
    .chip{{padding:3px 7px;border:1px solid #a6c9bd;border-radius:5px;background:#eef8f5}}.chip small{{color:#42566f}}
    </style><main><p class="eyebrow">神经网络研究 · 拓扑压力测试</p><h1>可学习路由能走多深？</h1>
    <div class="cards">{cards}</div>
    <p>输入把边记录打乱并拆散。模型从局部三符号窗口为候选边打分，通过共享权重重复传播。题目给出起点和每一步的指令。训练提供答案。地址按符号身份共享，每次传播沿一条关系边前进的规则是架构先验。</p>
    <p>训练只用 1–4 跳，末尾干扰最长 64 步，三个种子各 {protocol["steps"]*protocol["batch"]:,} 条训练序列。层状图每个条件 {protocol["groups"]} 题组，同时答对原题、换单条指令题和改单条边题才记为联合正确。每格是三种子均值和范围。</p>
    <h2>层状图 · 1,024 步末尾干扰</h2>
    {table(('深度与宽度','内部节点 / 边','可学习路由','共用路由消融','GRU','全路径对齐'),layer_rows)}
    <p>可学习路由与共用路由消融使用相同参数量和训练配置。消融用同一套边分数处理两类指令。GRU 使用相同题目与更新次数，参数量和学习率不同。层状图在 0 和 256 步的成绩也保存在原始结果中。</p>
    <h2>带环图 · 1,024 步末尾干扰</h2>
    <p>同一批权重直接运行更多步。每个图有 32 个内部节点、64 条出边，允许重复访问。固定旧路由只对换指令题保留原先的逐步路由，读出仍接收修改后的指令。</p>
    {table(('深度','路径中重复访问','联合正确','每步只留一条边','全路径对齐','固定旧路由'),cycle_rows)}
    {example_html('同题演示 · 层状分支图',*layer_example)}
    {example_html('同题演示 · 64 跳带环图',*cycle_example)}
    <h2>解释范围</h2><p>边记录在输入里明示。训练学到了怎样把指令和局部边类型对应起来，图的实例随输入构建。这轮验证的是分支、合流、回访和路径深度。隐藏关系发现、多路径同时推理、在线改写内部结构和超过 64 跳的行为仍待验证。</p>
    </main></html>'''
    (root/'report-cn.html').write_text(data,encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    render(parser.parse_args().output)
