"""Rerun all eight models on the three original binary patterns."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import torch
import representative_benchmark as base
import decimal_benchmark as decimal
from linear_readout import LinearReadoutRNN

KINDS = ['custom', 'custom_linear20', 'rnn', 'gru', 'lstm', 'mlp', 'tcn', 'transformer']
WIDTHS = {'custom':20, 'custom_linear20':20, 'rnn':31, 'gru':17, 'lstm':15,
          'mlp':7, 'tcn':11, 'transformer':9}
PATTERNS, SEEDS = base.common.PATTERNS, base.common.SEEDS
PREVIOUS = Path('C:/pioneer/rnn-research-runs/representative_compute_20261028/run_20261028T032905_545175Z')
ORIGINAL_FACTORY = base.OnlineModel

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class OnlineModel(ORIGINAL_FACTORY):
    def __init__(self, kind, width, seed):
        if kind != 'custom_linear20':
            super().__init__(kind, width, seed)
            return
        assert width == 20
        self.kind, self.width = 'custom', width
        self.network = LinearReadoutRNN(seed=seed, width=width)
        self.optimizer = self.network.optimizer
        self.updates = 0
        self.config = {'kind':kind, 'width':width,
            'parameters':sum(p.numel() for p in self.network.parameters()),
            'architecture':{'neurons':20,'connection_traces':400,'retention_parameters':20,
                'readout':'10 x 20 linear + 10 bias; initialized as legacy scale*[I|0]',
                'input_sparsity':.02,'retention_weight_decay':0.0}}

def sequences():
    return {task:[int(pattern[i % len(pattern)]) for i in range(600)] for task,pattern in PATTERNS.items()}

def validate_tasks(streams):
    for task,pattern in PATTERNS.items():
        assert len(streams[task])==600 and set(streams[task])=={0,1}
        assert streams[task]==[int(pattern[i%len(pattern)]) for i in range(600)]
    period=[int(c) for c in PATTERNS['period16']]
    cyclic=period+period[:4]
    fourgrams={tuple(cyclic[i:i+4]) for i in range(16)}
    successors={}
    for i in range(16):
        successors.setdefault(tuple(cyclic[i:i+3]),set()).add(cyclic[i+3])
    assert len(fourgrams)==16 and any(len(s)>1 for s in successors.values())
    return {'passed':True,'patterns':PATTERNS,'length':600,
        'histograms':{t:dict(Counter(s)) for t,s in streams.items()},
        'period16_cyclic_fourgrams':len(fourgrams),'period16_has_ambiguous_threegram':True,
        'input_symbols':[0,1],'input_ports':10,'output_classes':10,
        'note':'Original binary task streams; original 10-class network interface retained.'}

def validate_initialization():
    checked=[]
    for seed in SEEDS:
        old, new = OnlineModel('custom',20,seed), OnlineModel('custom_linear20',20,seed)
        for name,value in old.network.state_dict().items():
            if name!='scale_raw':
                assert torch.equal(value,new.network.state_dict()[name]),name
        assert not hasattr(new.network,'scale_raw')
        assert new.network.decay_logit.shape==(20,1)
        assert new.config['parameters']==851 and old.config['parameters']==642
        assert new.optimizer.state_dict()['state']==old.optimizer.state_dict()['state']=={}
        for history in ([0],[0,1,0,1],sequences()['period16'][:32]):
            a,sa=old.network.replay(history); b,sb=new.network.replay(history)
            assert torch.equal(a,b) and torch.equal(sa.activity,sb.activity) and torch.equal(sa.fast_edges,sb.fast_edges)
        checked.append(seed)
    return {'passed':True,'seeds_checked':checked,'core_initialization_bitwise_equal':True,
        'initial_logits_and_states_bitwise_equal':True,'fresh_optimizer_state':True,
        'old_parameters':642,'new_parameters':851,'retention_shape':[20,1],
        'changed_factor':'Replace scalar positive output scale with full linear output W,b; remove scale_raw.'}

def profile(kind,width,length=32,history=None):
    warm=history if history is not None else sequences()['period16'][:32]
    model=OnlineModel(kind,width,73)
    model.train(warm)
    prediction=base.counted(lambda:model.predict(warm[:length]))
    training=base.counted(lambda:model.train(warm[:length]))
    return {'config':model.config,'prediction':prediction,'training_3_updates':training,
            'total_proxy':prediction['total_proxy']+training['total_proxy'],'origin':'new'}

def budget(kind,width):
    lengths={n:profile(kind,width,n) for n in range(33)}
    prediction=sum(lengths[min(i-1,32)]['prediction']['total_proxy'] for i in range(1,601))
    training=sum(lengths[min(i,32)]['training_3_updates']['total_proxy'] for i in range(1,601))
    return lengths[32],{'prediction_proxy':prediction,'training_proxy':training,'total_proxy':prediction+training,
        'per_length':{n:{'prediction':p['prediction']['total_proxy'],
                          'training':p['training_3_updates']['total_proxy']} for n,p in lengths.items()},'origin':'new'}

def latency(kind,width):
    m=OnlineModel(kind,width,73); history=sequences()['period16'][:32]
    for _ in range(5): m.predict(history); m.train(history)
    timings=[]
    for _ in range(20):
        start=time.perf_counter(); m.predict(history); m.train(history)
        timings.append((time.perf_counter()-start)*1000)
    return {'serial_single_thread_mean_ms':statistics.mean(timings),
        'serial_single_thread_median_ms':statistics.median(timings),'warmup':5,'repeats':20,'origin':'new'}

def run_one(root,kind,width,task,sequence,seed):
    # The original scoring/training loop is unchanged; only its factory is
    # replaced inside this isolated worker to include the new readout model.
    decimal.base.OnlineModel=OnlineModel
    result=decimal.run_one(root,kind,width,task,sequence,seed)
    result.update(source_run=root,origin='new',pattern=PATTERNS[task])
    base.common.write_json(Path(root)/result['run_directory']/'summary.json',result)
    return result

def aggregate(summaries):
    rows=[]
    for kind in KINDS:
        for task in PATTERNS:
            selected=[s for s in summaries if s['kind']==kind and s['task']==task]
            assert len(selected)==3
            rows.append({'kind':kind,'task':task,'pattern':PATTERNS[task],'count':1800,
                'hits':sum(s['hits'] for s in selected),
                **{m:statistics.mean(s[m] for s in selected) for m in
                   ('accuracy','first100_accuracy','last100_accuracy','mean_nll')},
                'seed_accuracy_min':min(s['accuracy'] for s in selected),
                'seed_accuracy_max':max(s['accuracy'] for s in selected),'origin':'new'})
    return rows

def reproduce(root,summaries,previous):
    count=0; checkpoints=0
    for summary in summaries:
        if summary['kind']=='custom_linear20': continue
        directory=summary['run_directory']
        before=[json.loads(s) for s in (previous/directory/'events.jsonl').read_text().splitlines()]
        after=[json.loads(s) for s in (root/directory/'events.jsonl').read_text().splitlines()]
        assert len(before)==len(after)==600
        for a,b in zip(before,after):
            assert a['probabilities']==b['probabilities'] and a['training_loss']==b['training_loss']
            count+=1
        old=torch.load(previous/directory/'checkpoint_600.pt',weights_only=False,map_location='cpu')
        new=torch.load(root/directory/'checkpoint_600.pt',weights_only=False,map_location='cpu')
        for name,value in old['model'].items(): assert torch.equal(value,new['model'][name])
        checkpoints+=1
    assert count==37800 and checkpoints==63
    return {'passed':True,'reference_run':str(previous),'results_reused':False,
        'fresh_predictions_bitwise_reproduced':count,'final_model_checkpoints_bitwise_reproduced':checkpoints}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent',type=Path,default=Path('C:/pioneer/rnn-research-runs/binary_readout_20261028'))
    parser.add_argument('--reference',type=Path,default=PREVIOUS)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root=args.output_parent/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    (root/'source_snapshot').mkdir(parents=True)
    print(f'OUTPUT_ROOT={root}',flush=True)
    names=['representative_benchmark.py','representative_recurrent.py','representative_sequence.py',
        'compute_budget.py','retention_weight_decay.py','per_neuron_retention.py',
        'decimal_benchmark.py','decimal_tasks.py','linear_readout.py','binary_readout_benchmark.py']
    for source in [base.common.MODEL_ROOT/'model.py',base.common.MODEL_ROOT/'requirements.txt',
                   *[Path(__file__).with_name(name) for name in names]]:
        shutil.copyfile(source,root/'source_snapshot'/source.name)
    previous_protocol=load(args.reference/'protocol.json')
    for name,sha in previous_protocol['source_sha256'].items():
        assert base.common.digest(root/'source_snapshot'/name)==sha,f'Historical source changed {name}'
    streams=sequences()
    base.common.write_json(root/'input_sequences.json',streams)
    base.common.write_json(root/'task_validation.json',validate_tasks(streams))
    base.common.write_json(root/'mechanism_validation.json',validate_initialization())
    profiles={};full={};timings={}
    for kind in KINDS:
        profiles[kind],full[kind]=budget(kind,WIDTHS[kind])
        timings[kind]=latency(kind,WIDTHS[kind])
        print('PROFILE='+json.dumps({'kind':kind,'width':WIDTHS[kind],
            'parameters':profiles[kind]['config']['parameters'],'full_proxy':full[kind]['total_proxy']}),flush=True)
    target=full['custom']['total_proxy']
    for kind in KINDS:
        full[kind]['ratio_to_custom']=full[kind]['total_proxy']/target
        profiles[kind]['ratio_to_custom']=profiles[kind]['total_proxy']/profiles['custom']['total_proxy']
        if kind!='custom_linear20': assert abs(full[kind]['ratio_to_custom']-1)<=.05
    reference_full=load(args.reference/'full_run_budgets.json')
    for kind in base.KINDS: assert full[kind]['total_proxy']==reference_full[kind]['total_proxy']
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'kinds':KINDS,'tasks':PATTERNS,
        'patterns':PATTERNS,'seeds':SEEDS,'length':600,'classes':10,'input_count':10,'task_symbols':[0,1],
        'binary_task_not_two_class_rearchitecture':True,'all_models_freshly_rerun':True,'results_reused':False,
        'context':32,'updates_per_press':3,'learning_rate':.015,'weight_decay':1e-4,
        'custom_retention_weight_decay':0.0,'custom_input_prox':.02,'baseline_input_prox':0.0,
        'clip_norm':1.0,'training_order':'predict(history), freeze and score, append target, train',
        'cold_start':'uniform 10 classes, argmax lowest-index 0, included in scores',
        'binary_random_guess_expected_accuracy':.5,'uniform_10_class_nll':2.302585092994046,
        'compute_metric':'auditable scalar-op proxy including prediction + 3 updates; not hardware exact FLOPs',
        'budget_policy':'Fixed previously chosen widths; standards within 5% of old custom20; linear20 separately disclosed extra compute.',
        'fixed_core_width':20,'new_readout_extra_compute':full['custom_linear20']['ratio_to_custom']-1,
        'cpu_latency_remeasured':True,'device':'cpu','dtype':'float32','torch_threads':1,
        'workers':args.workers,'torch':str(torch.__version__),'python':sys.version,
        'reference_run':str(args.reference),'reference_purpose':'post-run exact reproducibility check only; no result reuse',
        'source_sha256':{p.name:base.common.digest(p) for p in (root/'source_snapshot').iterdir()}}
    base.common.write_json(root/'protocol.json',protocol)
    base.common.write_json(root/'matched_budgets.json',profiles)
    base.common.write_json(root/'full_run_budgets.json',full)
    base.common.write_json(root/'cpu_latency.json',timings)
    records=[]
    for kind in KINDS:
        for task,seq in streams.items():
            p=profile(kind,WIDTHS[kind],history=seq[:32])
            assert p['total_proxy']==profiles[kind]['total_proxy']
            records.append({'kind':kind,'task':task,'total_proxy':p['total_proxy']})
    base.common.write_json(root/'budget_validation.json',{'passed':True,'profiles':records,
        'linear20_ratio_to_custom':full['custom_linear20']['ratio_to_custom']})
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,str(root),kind,WIDTHS[kind],task,seq,seed)
            for kind in KINDS for task,seq in streams.items() for seed in SEEDS]
        for job in as_completed(jobs):
            summary=job.result();summaries.append(summary)
            print('RUN='+json.dumps(summary),flush=True)
    assert len(summaries)==72 and sum(s['count'] for s in summaries)==43200
    summaries.sort(key=lambda s:(s['kind'],s['task'],s['seed']))
    base.common.write_json(root/'runs_summary.json',summaries)
    base.common.write_json(root/'aggregate.json',aggregate(summaries))
    with (root/'runs_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[k for k in summaries[0] if k!='phase_metrics'])
        writer.writeheader();writer.writerows({k:v for k,v in s.items() if k!='phase_metrics'} for s in summaries)
    base.common.write_json(root/'historical_reproduction.json',reproduce(root,summaries,args.reference))
    base.common.write_json(root/'checksums.json',{str(p.relative_to(root)):base.common.digest(p)
        for p in root.rglob('*') if p.is_file()})
    base.common.write_json(root/'completed.json',{'success':True,'runs':72,'predictions':43200,'results_reused':False})
    print(f'COMPLETE={root}',flush=True)

if __name__=='__main__': main()
