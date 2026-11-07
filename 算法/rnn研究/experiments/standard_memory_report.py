"""Audit the standard-model benchmark and build a Chinese offline report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from long_memory_report import MODEL_NAMES, TASK_NAMES, target as expected_target
from standard_memory_eval import training_batches
from matplotlib import font_manager
import matplotlib.pyplot as plt


TIER_NAMES = {'small': '约 1,500 参数', 'matched': '约 10 万参数'}
REGIME_NAMES = {'extrapolate': '训练至 128 步，外推到 256 步', 'seen256': '训练覆盖 256 步'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def collect(root, protocol):
    for folder, group in ((root, 'data_sha256'), (root/'source', 'source_sha256')):
        for name, digest in protocol[group].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    selected = {tuple(r['job']):r['lr'] for r in read(root/'selection.json')}
    tasks, models, tiers, regimes, seeds, gaps, budgets = (
        protocol[key] for key in ('tasks', 'models', 'tiers', 'regimes', 'seeds', 'gaps', 'budgets'))
    data = {}
    for task in tasks:
        splits = read(root/f'{task}-data.json')
        disjoint = set()
        forbidden = {tuple(x) for x in splits['forbidden']}
        for split in ('validation', 'test'):
            for gap in gaps:
                pairs = splits[split][str(gap)]
                assert len(pairs) == 64 and all(sum(r[2] == label for r in pairs) == 16 for label in range(4))
                for x, xp, y, yp in pairs:
                    assert sum(a != b for a, b in zip(x, xp, strict=True)) == 1
                    assert expected_target(task, x) == y and expected_target(task, xp) == yp
                    keys = {tuple(x), tuple(xp)}
                    assert keys <= forbidden
                    if gap:
                        assert not keys & disjoint
                        disjoint.update(keys)
        for regime, seed in itertools.product(regimes, [protocol['tuning_seed'], *seeds]):
            batches = list(training_batches(task, regime, seed, forbidden))
            assert sum(map(len, batches)) == budgets[-1]
            prefix_length = {'recall':3, 'binding':8, 'arithmetic':6}[task]
            for batch in batches:
                for x, y in batch:
                    assert expected_target(task, x) == y
                    assert len(x) == prefix_length or tuple(x) not in forbidden
        data[task] = splits['test']
    groups = defaultdict(list)
    files = {}
    trials = 0
    for tier, model, task, regime in itertools.product(tiers, models, tasks, regimes):
        job = (tier, model, task, regime)
        prefix = '_'.join(job)
        trial_ranks = []
        for lr in protocol['rates']:
            path = root/'tuning'/f'{prefix}_{protocol["tuning_seed"]}_{lr:g}.json'
            run = read(path)
            score = run['scores'][str(budgets[-1])]
            assert [r['gap'] for r in score] == [g for g in regimes[regime] if g > 0]
            assert all('probabilities' not in r for r in score)
            trial_ranks.append((sum(r['paired_accuracy'] for r in score), -sum(r['nll'] for r in score), -lr))
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            trials += 1
        assert -max(trial_ranks)[2] == selected[job]
        for seed in seeds:
            path = root/'evaluation'/f'{prefix}_{seed}_{selected[job]:g}.json'
            run = read(path)
            assert tuple(run[k] for k in ('tier', 'model', 'task', 'regime')) == job
            assert run['seed'] == seed and run['lr'] == selected[job] and len(run['loss'])*protocol['batch'] == budgets[-1]
            for budget in budgets:
                score = run['scores'][str(budget)]
                assert [r['gap'] for r in score] == gaps
                for row in score:
                    gap = row['gap']
                    expected = data[task][str(gap)]
                    hits, paired = [], []
                    for probs, (_, _, target, paired_target) in zip(row['probabilities'], expected, strict=True):
                        for p in probs:
                            assert len(p) == 10 and all(0 <= v <= 1 for v in p) and abs(sum(p)-1) < 1e-6
                        a, b = [max(range(10), key=p.__getitem__) for p in probs]
                        hits.append(a == target)
                        paired.append(a == target and b == paired_target)
                    assert row['n'] == len(expected) and mean(hits) == row['accuracy'] and mean(paired) == row['paired_accuracy']
                    groups[tier, regime, task, budget, model, gap].append((row, run['parameters'], run['lr']))
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for (tier, regime, task, budget, model, gap), runs in groups.items():
        accuracy = [r['accuracy'] for r, _, _ in runs]
        paired = [r['paired_accuracy'] for r, _, _ in runs]
        rows.append(dict(tier=tier, regime=regime, task=task, budget=budget, model=model, gap=gap,
                         accuracy=mean(accuracy), paired=mean(paired),
                         accuracy_min=min(accuracy), accuracy_max=max(accuracy), paired_min=min(paired), paired_max=max(paired),
                         parameters=runs[0][1], lr=runs[0][2], n=sum(r['n'] for r, _, _ in runs)))
    audit = dict(tuning_runs=trials, evaluation_runs=len(files)-trials, aggregate_rows=len(rows),
                 test_predictions=2*sum(r['n'] for r in rows), sha256=files)
    (root/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    protocol = read(root/'protocol.json')
    rows = collect(root, protocol)
    plt.style.use('seaborn-v0_8-whitegrid')
    chinese_font = next((f for f in font_manager.fontManager.ttflist
                         if f.name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC')), None)
    if chinese_font:
        plt.rcParams['font.family'] = chinese_font.name
    tier = protocol['tiers'][-1]
    budget = protocol['budgets'][-1]
    index = {(r['task'], r['gap'], r['model'], r['regime']):r['paired'] for r in rows
             if r['tier'] == tier and r['budget'] == budget}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharey=True, layout='constrained')
    for row, gap in enumerate((0, 256)):
        for col, task in enumerate(protocol['tasks']):
            ax = axes[row, col]
            x = list(range(len(protocol['models'])))
            for offset, color, regime, label in (
                (-.2, '#587296', 'extrapolate', '训练至 128 步' if chinese_font else 'Train to 128'),
                (.2, '#008673', 'seen256', '训练覆盖 256 步' if chinese_font else 'Train on 256'),
            ):
                ax.bar([i+offset for i in x], [100*index[task,gap,m,regime] for m in protocol['models']],
                       .37, color=color, label=label)
            ax.axhline(6.25, color='#596171', ls='--', lw=1)
            ax.set(ylim=(0, 103), xticks=x, xticklabels=protocol['models'])
            ax.tick_params(axis='x', labelrotation=30)
            title = f'{TASK_NAMES[task]} · {gap} 步' if chinese_font else f'{task} · gap {gap}'
            ax.set_title(title)
        axes[row, 0].set_ylabel('成对正确率（%）' if chinese_font else 'Paired accuracy (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle(f'{TIER_NAMES[tier]} · {budget:,} 题 · 短题诊断与 256 步能力' if chinese_font else
                 f'{tier} · {budget:,} training items · short diagnostic vs 256 steps', fontsize=16)
    for ext in ('png', 'svg'):
        fig.savefig(root/f'standard_memory.{ext}', dpi=160)
    plt.close(fig)
    with (root/'scores-zh.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['参数档', '训练长度', '任务', '训练题数', '模型', '干扰步数', '单题准确率', '成对正确率',
                         '单题最低种子', '单题最高种子', '成对最低种子', '成对最高种子', '参数量', '选定学习率', '原题预测数'])
        for r in rows:
            writer.writerow([TIER_NAMES[r['tier']], REGIME_NAMES[r['regime']], TASK_NAMES[r['task']], r['budget'],
                             MODEL_NAMES[r['model']], r['gap'], *[f'{r[k]:.2%}' for k in (
                                 'accuracy','paired','accuracy_min','accuracy_max','paired_min','paired_max')],
                             r['parameters'], r['lr'], r['n']])
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="only light">
<title>标准模型长程记忆公平对照</title><style>
:root{color-scheme:only light;color:#18263b;background:white}body{margin:0;font:16px/1.55 system-ui}
main{max-width:1250px;margin:auto;padding:24px;box-sizing:border-box}h1,h2,p,label{color:#18263b}
a{color:#15529c}select{font:inherit;color:#18263b;background:#eef4fb;margin:4px 12px 4px 4px;padding:4px}
img{width:100%;background:white}
.wrap{overflow-x:auto;border:1px solid #cbd5e1;border-radius:8px;margin:10px 0 26px}
table{width:100%;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums;min-width:730px}
th,td{padding:8px 12px;border-bottom:1px solid #d7e0ec;text-align:right;color:#18263b}thead th{background:#eaf1fa}
th:first-child{text-align:left;white-space:nowrap}tr:nth-child(even){background:#f7f9fd}.unseen{background:#fff3d7}
small{display:block;color:#405678;font-size:.78rem}.bar{height:5px;background:#dbe6f2;border-radius:4px;margin:4px 0}
.bar i{display:block;height:100%;background:#2866a8;border-radius:4px}.strong{font-weight:700;color:#155a43}
</style></head><body><main style="background:#fff;color:#18263b;color-scheme:only light">
<h1>标准模型长程记忆公平对照</h1>
<p>各模型、任务、参数档和训练长度条件都使用相同的学习率候选 0.001、0.003、0.01。只用独立验证题选学习率，再用三个种子重新训练。每批 16 题，记录同一条训练轨迹在 480 和 2,000 题时的成绩。</p>
<h2>256 步成绩一览</h2><p>比较训练中见过 256 步和只见过至 128 步。每格上方为单题准确率，下方为同一题修改早期线索前后都答对的比例。</p>
<label>参数档<select id="overview-tier"></select></label><label>训练题数<select id="overview-budget"></select></label>
<div class="wrap" id="overview"></div>
<h2>各长度成绩对比</h2>
<label>参数档<select id="tier"></select></label><label>训练长度<select id="regime"></select></label>
<label>任务<select id="task"></select></label><label>训练题数<select id="budget"></select></label>
<label>指标<select id="metric"><option value="paired">成对正确率</option><option value="accuracy">单题准确率</option></select></label>
<p>每格给出三种子均值与最低至最高值。0 步检查任务规则，因输入空间有限，可与训练题重复。其余长度的测试题及配对与训练分离。均匀四类猜测的单题期望为 25%，独立配对期望为 6.25%。</p>
<div class="wrap" id="matrix"></div>
<h2>短题诊断与长题能力</h2><p>下图用较大的参数档展示成对正确率。虚线是独立四类猜测的 6.25%。</p>
<img src="standard_memory.svg" alt="各标准模型在 0 步和 256 步的成对正确率">
<p><a href="scores-zh.csv">中文成绩表</a> · <a href="selection.json">验证集学习率选择</a> · <a href="audit.json">核验记录</a></p>
</main><script>
const data = '''+json.dumps(rows)+''', protocol = '''+json.dumps(protocol)+''';
const modelNames = '''+json.dumps(MODEL_NAMES,ensure_ascii=False)+''', taskNames = '''+json.dumps(TASK_NAMES,ensure_ascii=False)+''';
const tierNames = '''+json.dumps(TIER_NAMES,ensure_ascii=False)+''', regimeNames = '''+json.dumps(REGIME_NAMES,ensure_ascii=False)+''';
const el = id => document.getElementById(id), pct = v => (v*100).toFixed(1)+'%';
function fill(id, options, labels){ el(id).innerHTML=options.map(k=>`<option value="${k}">${labels[k]||k}</option>`).join(''); }
for(const id of ['tier','overview-tier']) fill(id,protocol.tiers,tierNames);
for(const id of ['budget','overview-budget']) fill(id,protocol.budgets,{});
fill('task',protocol.tasks,taskNames);fill('regime',Object.keys(protocol.regimes),regimeNames);
el('overview-budget').value=el('budget').value=String(protocol.budgets.at(-1));
if(protocol.tiers.includes('matched')) el('overview-tier').value=el('tier').value='matched';
function overview(){
 const tier=el('overview-tier').value, budget=+el('overview-budget').value;
 const chosen=data.filter(r=>r.tier===tier&&r.budget===budget&&r.gap===256);
 const cells=(m,t)=>Object.keys(protocol.regimes).map(k=>{
   const r=chosen.find(r=>r.model===m&&r.task===t&&r.regime===k);
   return `<td><strong>${pct(r.accuracy)}</strong><small>成对 ${pct(r.paired)}</small></td>`;
 }).join('');
 const rows=protocol.models.map(m=>`<tr><th scope="row">${modelNames[m]}</th>${protocol.tasks.map(t=>cells(m,t)).join('')}</tr>`).join('');
 const headings=protocol.tasks.map(t=>`<th colspan="2">${taskNames[t]}</th>`).join('');
 el('overview').innerHTML=`<table><thead><tr><th rowspan="2">模型</th>${headings}</tr><tr>${protocol.tasks.map(()=>'<th>外推 256</th><th>见过 256</th>').join('')}</tr></thead><tbody>${rows}</tbody></table>`;
}
function matrix(){
 const tier=el('tier').value, regime=el('regime').value, task=el('task').value, budget=+el('budget').value, metric=el('metric').value;
 const chosen=data.filter(r=>r.tier===tier&&r.regime===regime&&r.task===task&&r.budget===budget);
 const rows=protocol.models.map(m=>{
   const group=chosen.filter(r=>r.model===m);
   const cells=group.map(r=>`<td class="${r.gap===256&&regime==='extrapolate'?'unseen':''}"><strong>${pct(r[metric])}</strong><small>${pct(r[metric+'_min'])}–${pct(r[metric+'_max'])}</small><div class="bar" aria-hidden="true"><i style="width:${r[metric]*100}%"></i></div></td>`).join('');
   return `<tr><th scope="row">${modelNames[m]}<small>${group[0].parameters.toLocaleString()} 参数 · 学习率 ${group[0].lr}</small></th>${cells}</tr>`;
 }).join('');
 el('matrix').innerHTML=`<table><thead><tr><th>模型</th>${protocol.gaps.map(g=>`<th>${g} 步</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
}
for(const id of ['overview-tier','overview-budget']) el(id).onchange=overview;
for(const id of ['tier','regime','task','budget','metric']) el(id).onchange=matrix;
overview();matrix();
</script></body></html>'''
    (root/'report-cn.html').write_text(html, encoding='utf-8')
    print(f'Verified and exported {len(rows)} aggregate rows.', flush=True)


if __name__ == '__main__':
    main()
