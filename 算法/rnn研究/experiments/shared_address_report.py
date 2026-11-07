"""Render the shared-address experiment as a self-contained Chinese report."""

import argparse
from pathlib import Path
from statistics import mean

from frontier_data import oracle
from frontier_eval import digest, read
from shared_address_eval import MODELS, SEEDS, aggregate


LABELS = dict(shared='共享地址图', permuted='打乱初始地址图', GRU='GRU', TCN='全感受野 TCN')
COLORS = dict(shared='#00634f', permuted='#7041ab', GRU='#994700', TCN='#2b60a3')
LAYOUTS = dict(compact='紧凑', scattered='换序与拆散')
CSS = """
html{color-scheme:only light;background:#fff;color:#19273b}
body{margin:0;font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif;background:#fff;color:#19273b}
main{max-width:1140px;margin:auto;padding:32px 24px 64px;background:#fff;color:#19273b}
h1,h2,h3{color:#15253c;line-height:1.3}h1{font-size:2.2rem;margin:.2em 0 .4em}
h2{margin:2em 0 .5em;border-top:1px solid #c8d2dc;padding-top:1em}
.eyebrow{color:#00634f;font-weight:bold;letter-spacing:.1em}.muted{color:#42566f}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{border:1px solid #b8cbd1;border-radius:10px;padding:16px;background:#eef8f5}
.card strong{font-size:1.7rem;color:#005642;display:block;font-variant-numeric:tabular-nums}
.scroll{overflow:auto;border:1px solid #c8d2dc;border-radius:8px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:10px 13px;text-align:right;border-bottom:1px solid #d0d8e3;color:#15253c;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#ebf1f9}td{background:#fff}
tbody tr:last-child td{border-bottom:0}td small{display:block;color:inherit;opacity:.9}
svg{display:block;width:100%;height:auto;background:#fff;color:#19273b}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-weight:600}.axis{font:14px system-ui;fill:#324863}
summary{cursor:pointer;color:#174b85;font-weight:bold}
"""


def table(headers, rows):
    head = ''.join(f'<th scope="col">{h}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td{s}>{v}</td>' for v, s in row) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def heat(value, detail=''):
    bg, fg = ('#0d5345', '#fff') if value >= .8 else ('#a9dfcf', '#132e27') if value >= .5 else (
        '#daf1eb', '#143b30') if value >= .2 else ('#f2f5fa', '#22364e')
    return f'{value:.1%}{detail}', f' style="background:{bg};color:{fg}"'


def mean_range(values):
    return heat(mean(values), f'<small>{min(values):.1%}–{max(values):.1%}</small>')


def training_curve(runs):
    budgets = [c['questions'] for c in runs[0]['checkpoints']]
    left, right, top, bottom = 64, 765, 22, 226
    x = lambda v: left + (right-left)*v/budgets[-1]
    y = lambda v: bottom - (bottom-top)*v
    svg = ['<svg viewBox="0 0 810 275" role="img" aria-label="零步开发集联合正确率随训练题数变化">']
    for rate in (0,.25,.5,.75,1):
        svg.append(f'<path d="M{left} {y(rate)}H{right}" stroke="#d1dbe5"/>'
                   f'<text class="axis" x="52" y="{y(rate)+4}" text-anchor="end">{rate:.0%}</text>')
    for budget in budgets:
        svg.append(f'<text class="axis" x="{x(budget)}" y="253" text-anchor="middle">{budget:,}</text>')
    for name in MODELS:
        values = [mean(c['joint'] for r in runs if r['model']==name
                       for c in r['checkpoints'][i]['development']) for i in range(len(budgets))]
        points = [(x(b), y(v)) for b,v in zip(budgets,values,strict=True)]
        color = COLORS[name]
        coords = ' '.join(f'{px},{py}' for px,py in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for px,py in points:
            svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
    return ''.join(svg) + '</svg>'


def edge_example(case, prediction):
    original, swapped = prediction['edges'][0]
    positions = sorted({p for graph in (original,swapped) for head in graph for p in head})
    colors = ('#00634f','#7041ab','#994700','#2b60a3')
    left = 54
    width = max(760, 76*len(positions))
    xs = {p: left + i*(width-2*left)/max(1,len(positions)-1) for i,p in enumerate(positions)}
    seq = case['variants'][0]
    svg = [f'<svg viewBox="0 0 {width} 260" role="img" aria-label="同题原查询与换键后的实际连边">']
    for n,(graph,py) in enumerate(((original,32),(swapped,228))):
        q = case['variants'][n][-1]
        svg.append(f'<text class="axis" x="{width/2}" y="{py}" text-anchor="middle">查询 {q}</text>')
        for head,selected in enumerate(graph):
            for pos in selected:
                py_start = py + (10 if n==0 else -21)
                py_end = 108 if n==0 else 154
                svg.append(f'<path d="M{width/2+(head-1.5)*24} {py_start} L{xs[pos]+(head-1.5)*4} {py_end}" '
                           f'fill="none" stroke="{colors[head]}" stroke-width="{3 if head==0 else 1.4}" opacity=".72"/>')
    for pos,px in xs.items():
        context = ' '.join(map(str,seq[max(0,pos-2):pos+1]))
        svg.append(f'<text class="axis" x="{px}" y="128" text-anchor="middle">位置 {pos+1}</text>'
                   f'<text class="axis" x="{px}" y="147" text-anchor="middle">{context}</text>')
    probs = prediction['probabilities']
    guess = lambda p: max(range(4), key=p.__getitem__)
    return ''.join(svg) + '</svg>' + (
        f'<p>原查询答 <b>{guess(probs[0])}</b>，正确答案 {case["targets"][0]}。'
        f'换键后答 <b>{guess(probs[1])}</b>，正确答案 {case["targets"][1]}。'
        f'换键后强制沿用原边，答 <b>{guess(prediction["frozen"])}</b>。</p>'
    )


def role_audit(cases, predictions):
    """Count position coverage from logged edges, using operand roles for audit only."""
    buckets = {}
    for case, prediction in zip(cases, predictions, strict=True):
        seq = case['variants'][0]
        sources = {i+2 for i, token in enumerate(seq[:-2]) if token==5}
        assert len(sources) == case['difficulty']
        covered = []
        for heads in prediction['edges'][0]:
            active = {i for head in heads for i in head}
            covered.append(sources <= active)
        first, swapped, *_ = prediction['probabilities']
        pick = lambda p: max(range(4), key=p.__getitem__)
        condition = case['layout'], case['gap'], case['difficulty']
        buckets.setdefault(condition, []).append((all(covered), pick(first) == pick(swapped)))
    return buckets


def audited_results(root):
    protocol = read(root/'protocol.json')
    runs = read(root/'results.json')
    expected = {(task,name,seed) for task in ('lookup','arithmetic') for name in MODELS for seed in SEEDS}
    assert {(r['task'],r['model'],r['seed']) for r in runs} == expected and len(runs) == len(expected)
    samples = {task: read(root/f'{task}-holdout.json') for task in protocol['tasks']}
    hashes = {}
    for cases in samples.values():
        for r in cases:
            base, swapped, *edited = r['variants']
            assert base[:-1] == swapped[:-1] and base[-1] != swapped[-1]
            assert all(sum(a!=b for a,b in zip(base,x,strict=True)) == 1 for x in edited)
            assert r['targets'] == [oracle(x) for x in r['variants']]
    prefixes = [bytes(r['variants'][0][:64]) for cases in samples.values() for r in cases]
    assert len(prefixes) == len(set(prefixes))
    example = None
    coverage = {}
    address_hits = []
    for r in runs:
        key = r['task'],r['seed']
        assert hashes.setdefault(key,r['train_sha256']) == r['train_sha256']
        folder = root / f'{r["task"]}-{r["model"]}-{r["seed"]}'
        for c in r['checkpoints']:
            assert digest(folder/f'weights-{c["questions"]//protocol["batch"]}.pt') == c['sha256']
        predictions = read(folder/'predictions.json')
        assert aggregate(samples[r['task']], predictions) == r['holdout']
        if (r['task'],r['model']) == ('lookup','shared'):
            for case,prediction in zip(samples['lookup'], predictions, strict=True):
                swapped = case['variants'][1]
                destination = swapped.index(swapped[-1])+1
                before,after = prediction['edges'][0]
                address_hits.append((destination in after[0], destination in before[0]))
        if (r['task'],r['model']) == ('arithmetic','shared'):
            for key, values in role_audit(samples['arithmetic'], predictions).items():
                coverage.setdefault(key, []).append((mean(hit for hit,_ in values), mean(same for _,same in values)))
        if (r['task'], r['model'], r['seed']) == ('lookup','shared',2):
            index = next(i for i,case in enumerate(samples['lookup']) if (case['layout'],case['gap']) == ('scattered',0))
            example = edge_example(samples['lookup'][index], predictions[index])
    return protocol, runs, example, coverage, [mean(v) for v in zip(*address_hits, strict=True)]


def render(root):
    protocol, runs, example, coverage, (new_source, old_source) = audited_results(root)

    def cells(task, model, layout, gap, difficulty, key='joint'):
        return [c[key] for r in runs if (r['task'],r['model'])==(task,model)
                for c in r['holdout'] if (c['layout'],c['gap'],c['difficulty'])==(layout,gap,difficulty)]

    lookup = [(layout,gap,1) for layout in LAYOUTS for gap in protocol['holdout_gaps']]
    arithmetic = [(layout,gap,n) for n in (2,3) for layout in LAYOUTS for gap in protocol['holdout_gaps']]
    zero = min(v for layout in LAYOUTS for v in cells('lookup','shared',layout,0,1))
    long = min(v for layout in LAYOUTS for gap in (384,768) for v in cells('lookup','shared',layout,gap,1))
    drop = mean(c['joint']-c['frozen'] for r in runs if (r['task'],r['model'])==('lookup','shared') for c in r['holdout'])
    composition = mean(v for l,g,n in arithmetic for v in cells('arithmetic','shared',l,g,n))
    cards = f'''<div class="cards">
    <div class="card">零步直接查值 · {'达标' if zero>=.85 else '未达标'}<strong>{zero:.1%}</strong>三种子、两种布局最低值，门槛 85%</div>
    <div class="card">384／768 步外推<strong>{long:.1%}</strong>直接查值联合正确率最低值</div>
    <div class="card">换键后复用旧边<strong>下降 {100*drop:.1f} 点</strong>直接查值联合正确率平均降幅，单位为百分点</div>
    <div class="card">模 4 角色组合<strong>{composition:.1%}</strong>联合正确率，所有测试条件平均</div></div>'''
    score_tables = []
    for task,title,conditions in (('lookup','直接查值',lookup),('arithmetic','模 4 角色组合',arithmetic)):
        rows = []
        for layout,gap,n in conditions:
            results = [cells(task,name,layout,gap,n) for name in MODELS]
            delta = mean(results[0]) - max(mean(v) for v in results[2:])
            label = f'{LAYOUTS[layout]} · {gap} 步' + (f' · {n} 个操作数' if task=='arithmetic' else '')
            rows.append([(label,'')]+[mean_range(v) for v in results]+[(f'{100*delta:+.1f}','')])
        score_tables.append(f'<h2>{title}</h2>'+table(
            ('测试条件',*(LABELS[m] for m in MODELS),'较最强标准<br>百分点'), rows))
    roles = []
    for layout,gap,n in arithmetic:
        variants = list(zip(*cells('arithmetic','shared',layout,gap,n,'variant_accuracy'), strict=True))
        covered, same = zip(*coverage[layout,gap,n], strict=True)
        roles.append([(f'{LAYOUTS[layout]} · {gap} 步 · {n} 个操作数',''),
                      mean_range(covered),mean_range(same),mean_range(variants[0]),mean_range(variants[1]),
                      *[mean_range(v) for v in variants[2:]], *([('—','')] if n==2 else [])])
    intervention = []
    for layout,gap,n in lookup:
        fresh = cells('lookup','shared',layout,gap,n)
        forced = cells('lookup','shared',layout,gap,n,'frozen')
        intervention.append([(f'{LAYOUTS[layout]} · {gap} 步',''),mean_range(fresh),mean_range(forced),
                             (f'{100*(mean(fresh)-mean(forced)):+.1f}','')])
    seeds = []
    for r in runs:
        if r['task'] != 'lookup':
            continue
        values = [c['joint'] for layout,gap,n in lookup for c in r['holdout']
                  if (c['layout'],c['gap'],c['difficulty']) == (layout,gap,n)]
        seeds.append([(LABELS[r['model']],''),
                      (str(r['seed']),''),(f'{r["parameters"]:,}',''),*[heat(v) for v in values]])
    names = ''.join(f'<span style="color:{COLORS[m]}">● {LABELS[m]}</span>' for m in MODELS)
    curves = ''.join(f'<h3>{title}</h3>'+training_curve([r for r in runs if r['task']==task])
                     for task,title in (('lookup','直接查值'),('arithmetic','模 4 运算')))
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="color-scheme" content="only light">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>共享地址记忆图 · 阶段验证</title>
    <style>{CSS}</style><main><p class="eyebrow">神经网络研究 · 路由诊断</p><h1>共享地址能否学会按键查值？</h1>
    <p>每个题组要答对原题、换查询和所有单处来源改动，才计为联合正确。每个条件有 128 个新题组，三个种子用同一套测试题。</p>{cards}
    <p class="muted">模型保留完整历史。第一头的地址分数比较查询符号与每个历史位置的前一符号。对齐组从相同地址出发，打乱组从固定随机置换出发。两组的参数都可训练。训练只提供输入序列与答案。</p>
    <p class="muted">每任务各训练 {protocol['steps']*protocol['batch']:,} 题，每批 {protocol['batch']} 题，AdamW，固定学习率 {protocol['learning_rate']}。训练最长尾部干扰 256 步。表内第一行为三种子均值，第二行为最小至最大。最强标准按各格 GRU、TCN 三种子均值取高者。</p>
    {''.join(score_tables)}
    <h2>模 4 失分拆解 · 共享地址图</h2>
    <p>覆盖要求原题和换指令题均选中每个操作数所在节点。覆盖只检查连边位置，信息是否被正确解码还取决于边权和输出层。每组换指令后正确答案都会变，预测未变说明模型至少错了一题。后五列分别是单个变体的准确率。</p>
    {table(('测试条件','两题均覆盖全部来源','换指令预测未变','原题','换指令','改第一操作数','改第二操作数','改第三操作数'),roles)}
    <h2>联合正确率随训练量变化</h2><p>零步开发集，各布局和三个种子取平均。横轴为累计训练题数。</p>
    <div class="legend">{names}</div>{curves}
    <h2>复用旧边会发生什么</h2><p>只对换查询变体强制使用原查询选出的边，仍用新查询重算边权；原题和来源改动变体按正常路径评分。</p>
    <p>换键后地址头选中正确值位置的比例，正常选边为 <b>{new_source:.1%}</b>，复用旧边为 <b>{old_source:.1%}</b>。正确位置由审计程序根据题目复算。</p>
    {table(('测试条件','正常选边','复用原边','下降百分点'),intervention)}
    <h3>同一题真实连边</h3><p>固定选种子 2、零步拆散的第一个题组。绿色粗线是地址头，其余颜色代表三个内容头。中间第二行是各位置及其前两位的原始符号。</p>{example}
    <h2>直接查值 · 逐种子成绩</h2>
    {table(('模型','种子','参数量',*(f'{LAYOUTS[l]}<br>{g} 步' for l,g,_ in lookup)),seeds)}
    <h2>适用边界</h2><p>本轮用全历史定位路由瓶颈。它随输入增长保留更多节点，因此此处的准确率不能用于判定 32 槽容量或 2,048 步资源门槛。模 4 使用独立训练。多跳和重复写入留在固定槽图的独立评测。</p>
    <p class="muted">发布核查覆盖留出集标签、单处改动、前缀隔离、同题训练散列、检查点散列，以及由原始概率重算的成绩。</p></main></html>'''
    (root/'report-cn.html').write_text(doc,encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output)
