"""Only retrain the changed readout; reuse verified decimal benchmark controls."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time
import torch
import representative_benchmark as base
import decimal_benchmark as decimal
from decimal_tasks import TASKS, SEEDS, make_sequence
from linear_readout import LinearReadoutRNN, validate_mechanism

PREVIOUS = Path('C:/pioneer/rnn-research-runs/decimal_guess_20261028/run_20261028T034514_978315Z')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class LinearOnline(base.OnlineModel):
    def __init__(self, kind, width, seed):
        self.kind, self.width = 'custom', width
        self.network = LinearReadoutRNN(seed=seed, width=width)
        self.optimizer = self.network.optimizer
        self.updates = 0
        self.config = {'kind': kind, 'width': width,
                       'parameters': sum(p.numel() for p in self.network.parameters()),
                       'architecture': {'neurons': width, 'connection_traces': width*width,
                           'retention_parameters': width, 'readout': 'full linear 10 x width + bias',
                           'input_sparsity': .02, 'retention_weight_decay': 0.0}}

def profile(width, length=32, history=None):
    history = history or make_sequence('cycle')[:32]
    model = LinearOnline(f'custom_linear{width}', width, 73)
    model.train(history)
    prediction = base.counted(lambda: model.predict(history[:length]))
    training = base.counted(lambda: model.train(history[:length]))
    return {'config': model.config, 'prediction': prediction, 'training_3_updates': training,
            'total_proxy': prediction['total_proxy'] + training['total_proxy']}

def full_budget(width):
    lengths = {n: profile(width, n) for n in range(33)}
    prediction = sum(lengths[min(i-1,32)]['prediction']['total_proxy'] for i in range(1,601))
    training = sum(lengths[min(i,32)]['training_3_updates']['total_proxy'] for i in range(1,601))
    return {'prediction_proxy': prediction, 'training_proxy': training, 'total_proxy': prediction+training,
            'per_length': {n:{'prediction':r['prediction']['total_proxy'],
                             'training':r['training_3_updates']['total_proxy']} for n,r in lengths.items()}}

def latency(width):
    model = LinearOnline(f'custom_linear{width}', width, 73)
    history = make_sequence('cycle')[:32]
    for _ in range(5):
        model.predict(history); model.train(history)
    timings=[]
    for _ in range(20):
        start=time.perf_counter(); model.predict(history); model.train(history)
        timings.append((time.perf_counter()-start)*1000)
    return {'serial_single_thread_mean_ms':statistics.mean(timings),
            'serial_single_thread_median_ms':statistics.median(timings),
            'warmup':5,'repeats':20,'origin':'new'}

def run_one(root, kind, width, task, sequence, seed):
    # Reuse the exact old scoring/training/checkpoint loop, only replace its factory
    # in this isolated worker. Standard models are never instantiated here.
    decimal.base.OnlineModel = LinearOnline
    result = decimal.run_one(root,kind,width,task,sequence,seed)
    result.update(source_run=root, origin='new')
    base.common.write_json(Path(root)/result['run_directory']/'summary.json',result)
    return result

def aggregate(summaries, kinds):
    rows=[]
    for kind in kinds:
        for task in TASKS:
            selected=[s for s in summaries if s['kind']==kind and s['task']==task]
            assert len(selected)==3
            rows.append({'kind':kind,'task':task,'count':1800,'hits':sum(s['hits'] for s in selected),
                **{m:statistics.mean(s[m] for s in selected) for m in
                   ('accuracy','first100_accuracy','last100_accuracy','mean_nll')},
                'seed_accuracy_min':min(s['accuracy'] for s in selected),
                'seed_accuracy_max':max(s['accuracy'] for s in selected),
                'origin':selected[0]['origin']})
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=PREVIOUS)
    parser.add_argument('--output-parent',type=Path,default=Path('C:/pioneer/rnn-research-runs/linear_readout_20261028'))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    previous=args.reference
    oldprotocol=load(previous/'protocol.json')
    assert load(previous/'completed.json')['success'] and load(previous/'audit.json')['passed']
    assert oldprotocol['seeds']==SEEDS and oldprotocol['context']==32 and oldprotocol['updates_per_press']==3
    root=args.output_parent/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    (root/'source_snapshot').mkdir(parents=True)
    print(f'OUTPUT_ROOT={root}',flush=True)
    checks=load(previous/'checksums.json')
    for name,sha in checks.items():
        assert base.common.digest(previous/name)==sha, name
    for name,sha in oldprotocol['source_sha256'].items():
        source=base.common.MODEL_ROOT/name if name in ('model.py','requirements.txt') else Path(__file__).with_name(name)
        assert base.common.digest(source)==sha, f'Changed old source {name}'
        shutil.copyfile(source,root/'source_snapshot'/name)
    for name in ('linear_readout.py','test_linear_readout.py','linear_readout_experiment.py'):
        shutil.copyfile(Path(__file__).with_name(name),root/'source_snapshot'/name)
    sequences=load(previous/'input_sequences.json')
    assert sequences=={task:make_sequence(task) for task in TASKS}
    for name in ('input_sequences.json','baseline_scores.json','task_validation.json'):
        shutil.copyfile(previous/name,root/name)
    base.common.write_json(root/'reuse_manifest.json',{'reference_run':str(previous),'verified_raw_files':len(checks),
        'sha256':checks,'protocol_sha256':base.common.digest(previous/'protocol.json'),
        'source_sha256':oldprotocol['source_sha256'],'reused_runs':126,'rerun_standard_models':False})
    base.common.write_json(root/'mechanism_validation.json',validate_mechanism(previous))
    matched=load(previous/'matched_budgets.json'); full=load(previous/'full_run_budgets.json')
    timings=load(previous/'cpu_latency.json')
    for kind in base.KINDS:
        matched[kind]['origin']='reused'; full[kind]['origin']='reused'; timings[kind]['origin']='reused'
        full[kind]['ratio_to_original_custom']=full[kind]['ratio_to_custom']
    target=full['custom']['total_proxy']
    widths=[20]; candidates={20:full_budget(20)}
    if abs(candidates[20]['total_proxy']/target-1)>.05:
        # Select using arithmetic alone, before any task accuracy is observed.
        for width in range(16,20):
            candidates[width]=full_budget(width)
        chosen=min(candidates,key=lambda w:abs(candidates[w]['total_proxy']/target-1))
        assert chosen!=20 and abs(candidates[chosen]['total_proxy']/target-1)<=.05
        widths.append(chosen)
    for width in widths:
        kind=f'custom_linear{width}'
        matched[kind]=profile(width); matched[kind]['origin']='new'
        matched[kind]['ratio_to_custom']=matched[kind]['total_proxy']/matched['custom']['total_proxy']
        full[kind]=candidates[width]; full[kind]['origin']='new'
        full[kind]['ratio_to_original_custom']=full[kind]['total_proxy']/target
        full[kind]['ratio_to_custom']=full[kind]['ratio_to_original_custom']
        timings[kind]=latency(width)
    kinds=base.KINDS+[f'custom_linear{w}' for w in widths]
    protocol={**oldprotocol,'created_at_utc':datetime.now(timezone.utc).isoformat(),'reference_run':str(previous),
        'kinds':kinds,'variants':[{'kind':f'custom_linear{w}','width':w,
            'role':'fixed_core_causal' if w==20 else 'compute_matched'} for w in widths],
        'only_changed_models_retrained':True,'new_runs':len(widths)*18,
        'readout_initialization':'W[:, :10] = old output_scale * I; other columns and bias zero; remove scale_raw',
        'cold_start_budget_note':'Same previous initialized-Adam per-length proxy protocol; actual raw checkpoints start with fresh Adam.',
        'cpu_latency_remeasured':True,'source_sha256':{p.name:base.common.digest(p) for p in (root/'source_snapshot').iterdir()}}
    base.common.write_json(root/'budget_search.json',candidates)
    base.common.write_json(root/'matched_budgets.json',matched)
    base.common.write_json(root/'full_run_budgets.json',full)
    base.common.write_json(root/'cpu_latency.json',timings)
    base.common.write_json(root/'protocol.json',protocol)
    records=[]
    for width in widths:
        for task,seq in sequences.items():
            p=profile(width,history=seq[:32])
            assert p['total_proxy']==matched[f'custom_linear{width}']['total_proxy']
            records.append({'kind':f'custom_linear{width}','task':task,'total_proxy':p['total_proxy']})
    base.common.write_json(root/'budget_validation.json',{'passed':True,'profiles':records,
        'new_full_run_ratios':{f'custom_linear{w}':full[f'custom_linear{w}']['ratio_to_original_custom'] for w in widths}})
    print('BUDGET='+json.dumps(load(root/'budget_validation.json')),flush=True)
    summaries=[{**s,'source_run':str(previous),'origin':'reused'} for s in load(previous/'runs_summary.json')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,str(root),f'custom_linear{w}',w,task,seq,seed)
              for w in widths for task,seq in sequences.items() for seed in SEEDS]
        for job in as_completed(jobs):
            summary=job.result(); summaries.append(summary)
            print('RUN='+json.dumps(summary),flush=True)
    summaries.sort(key=lambda s:(s['kind'],s['task'],s['seed']))
    base.common.write_json(root/'runs_summary.json',summaries)
    base.common.write_json(root/'aggregate.json',aggregate(summaries,kinds))
    with (root/'runs_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[k for k in summaries[0] if k!='phase_metrics'])
        writer.writeheader(); writer.writerows({k:v for k,v in s.items() if k!='phase_metrics'} for s in summaries)
    base.common.write_json(root/'checksums.json',{str(p.relative_to(root)):base.common.digest(p)
        for p in root.rglob('*') if p.is_file()})
    base.common.write_json(root/'completed.json',{'success':True,'new_runs':len(widths)*18,
        'new_predictions':len(widths)*10800,'reused_runs':126,'reused_predictions':75600})
    print(f'COMPLETE={root}',flush=True)

if __name__=='__main__':
    main()
