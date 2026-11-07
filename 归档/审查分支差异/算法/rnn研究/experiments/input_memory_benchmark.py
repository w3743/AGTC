"""Paired online RTRL experiment for memory on all external and recurrent inputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import torch
from streaming_rtrl import StreamingRTRL
from streaming_rtrl_benchmark import digest, write_json, streams
from streaming_rtrl_compute import counted, process
from decimal_tasks import SEEDS, TASKS as DECIMAL_TASKS, make_sequence

KINDS = ['baseline_rtrl20', 'fixed_input_memory20', 'learned_input_memory20']
TASKS = ['binary_alternating','binary_triples','binary_period16','decimal_cycle','decimal_triples','decimal_zigzag','decimal_fibonacci_mod10','decimal_switching','decimal_iid_random']
PREVIOUS = Path('C:/pioneer/rnn-research-runs/streaming_rtrl_20261028/run_20261028T142841_111140Z')
OUTPUT_PARENT = Path('C:/pioneer/rnn-research-runs/input_memory_20261031')

def sequences():
    return {**{'binary_'+k:v for k,v in streams().items()}, **{'decimal_'+k:make_sequence(k) for k in DECIMAL_TASKS}}

def build_model(kind, seed):
    if kind == 'baseline_rtrl20':
        return StreamingRTRL(seed=seed, credit='rtrl')
    from input_memory_rtrl import InputMemoryRTRL
    return InputMemoryRTRL(seed=seed, mode={'fixed_input_memory20':'fixed', 'learned_input_memory20':'learned'}[kind])

def retention(model):
    return (model.input_retention.detach().reshape(-1).tolist()
            if hasattr(model, 'input_retention') else [])

def checkpoint(model, path, kind):
    payload = model.state_payload()
    payload['config'] = {'kind':kind, 'width':20, 'parameters':sum(p.numel() for p in model.network.parameters()),
                         'core_size':model.core_size, 'input_memory_mode':getattr(model, 'mode', 'none')}
    torch.save(payload, path)

def run_one(root_string, kind, task, sequence, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    directory = root / f'{kind}_{task}_seed{seed}'
    directory.mkdir()
    model = build_model(kind, seed)
    checkpoint(model, directory/'checkpoint_000.pt', kind)
    rows=[]
    start=time.perf_counter()
    with (directory/'events.jsonl').open('w', encoding='utf-8') as handle:
        for i, target in enumerate(sequence):
            probs=model.predict().detach().tolist()
            assert len(probs)==10 and all(math.isfinite(v) and 0<=v<=1 for v in probs)
            assert abs(sum(probs)-1)<1e-6
            prediction=max(range(10), key=lambda j:(probs[j], -j))
            a_before=retention(model)
            r_before=model.network.trace_decay.detach().reshape(-1).tolist()
            stats=model.learn_target(target) if i else None
            a_after=retention(model)
            model.advance(target)
            assert not hasattr(model, 'history')
            tensors=(model.h,model.fast_edges,model.eh,model.ef,*model.network.parameters()) + ((model.memory,model.em) if hasattr(model,'memory') else ())
            assert all(torch.isfinite(t).all() for t in tensors), (kind,task,seed,i)
            row={'step':i+1,'target':target,'prediction':prediction,'probabilities':probs,
                 'hit':prediction==target,'nll':-math.log(max(probs[target],1e-300)),
                 'loss':stats['loss'] if stats else None,
                 'grad_norm':stats['grad_norm'] if stats else None,
                 'retention_grad_norm':stats['retention_grad_norm'] if stats else None,
                 'retention_grad':stats['retention_grad'] if stats else [],
                 'input_retention_grad_norm':stats.get('input_retention_grad_norm',0.0) if stats else None,
                 'input_retention_grad':stats.get('input_retention_grad',[0.0]*30) if stats else [],
                 'input_retention_before':a_before,'input_retention_after':a_after,'r_before':r_before,
                 'r_after':model.network.trace_decay.detach().reshape(-1).tolist(),
                 'memory_norm':float(model.memory.norm()) if hasattr(model,'memory') else 0.0,'memory_grad_norm':stats.get('memory_grad_norm',0.0) if stats else None,'em_norm':float(model.em.norm()) if hasattr(model,'em') else 0.0,'eta':float(model.network.eta.detach()),'eh_norm':float(model.eh.norm()),'ef_norm':float(model.ef.norm()),
                 'steps':model.steps,'updates_after':model.updates,
                 'elig_elements':model.eh.numel()+model.ef.numel()+(model.em.numel() if hasattr(model,'em') else 0)}
            if stats:
                assert abs(row['loss']-row['nll'])<1e-5
            handle.write(json.dumps(row,allow_nan=False)+'\n')
            rows.append(row)
            if (i+1)%100==0:
                handle.flush()
                checkpoint(model,directory/f'checkpoint_{i+1:03d}.pt',kind)
    assert model.steps==len(sequence) and model.updates==len(sequence)-1
    summary={'kind':kind,'task':task,'seed':seed,'width':20,'count':len(rows),
             'hits':sum(r['hit'] for r in rows),'accuracy':statistics.mean(r['hit'] for r in rows),
             'first100_accuracy':statistics.mean(r['hit'] for r in rows[:100]),
             'last100_accuracy':statistics.mean(r['hit'] for r in rows[-100:]),
             'mean_nll':statistics.mean(r['nll'] for r in rows),'optimizer_updates':model.updates,
             'parameters':sum(p.numel() for p in model.network.parameters()),'core_size':model.core_size,
             'final_input_retention':retention(model),
             'final_trace_retention':model.network.trace_decay.detach().reshape(-1).tolist(),
             'input_retention_gradient_nonzero_updates':sum((r['input_retention_grad_norm'] or 0)>0 for r in rows),
             'run_directory':directory.name,'concurrent_elapsed_seconds':time.perf_counter()-start}
    write_json(directory/'summary.json',summary)
    return summary

def measure_models():
    seq=sequences()['binary_period16']
    result={}
    for kind in KINDS:
        profiles={}
        for age in [0,1,2,32]:
            model=build_model(kind,73)
            for digit in seq[:age]: process(model,digit)
            profiles[age]=counted(lambda:process(model,seq[age]))
        assert profiles[2]['total_proxy']==profiles[32]['total_proxy']
        model=build_model(kind,73)
        for digit in seq[:40]: process(model,digit)
        times=[]
        for digit in seq[40:140]:
            start=time.perf_counter(); process(model,digit)
            times.append((time.perf_counter()-start)*1000)
        eligible=model.eh.numel()+model.ef.numel()+(model.em.numel() if hasattr(model,'em') else 0)
        result[kind]={'parameters':sum(p.numel() for p in model.network.parameters()),
            'steady_proxy':profiles[32]['total_proxy'],
            'full_run_proxy':profiles[0]['total_proxy']+profiles[1]['total_proxy']+598*profiles[32]['total_proxy'],
            'serial_single_thread_mean_ms':statistics.mean(times),
            'serial_single_thread_median_ms':statistics.median(times),'warmup':40,'repeats':100,
            'eligibility_elements':eligible,'eligibility_bytes_float32':eligible*4,
            'profiles':profiles,'included':'predict, one Adam/clip/prox learning update, advance including RTRL; no event/checkpoint I/O',
            'metric':'documented scalar arithmetic proxy, memory/index/copy excluded; not hardware FLOPs'}
        print('PROFILE='+json.dumps({k:v for k,v in result[kind].items() if k!='profiles'}),flush=True)
    for data in result.values(): data['ratio_to_baseline']=data['full_run_proxy']/result[KINDS[0]]['full_run_proxy']
    return result

def snapshot(root):
    exp=Path(__file__).parent
    names=['input_memory_rtrl.py','input_memory_benchmark.py','test_input_memory_rtrl.py',
           'streaming_rtrl.py','streaming_rtrl_benchmark.py','streaming_rtrl_compute.py',
           'linear_readout.py','retention_weight_decay.py','representative_benchmark.py',
           'per_neuron_retention.py','representative_recurrent.py','representative_sequence.py',
           'compute_budget.py','decimal_benchmark.py','decimal_tasks.py']
    for name in names: shutil.copyfile(exp/name,root/'source_snapshot'/name)
    for name in ['model.py','requirements.txt']: shutil.copyfile(exp.parent/name,root/'source_snapshot'/name)
    return {p.name:digest(p) for p in (root/'source_snapshot').iterdir()}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output-parent',type=Path,default=OUTPUT_PARENT)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import unittest, test_input_memory_rtrl
    tests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_input_memory_rtrl))
    assert tests.wasSuccessful()
    root=args.output_parent/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    (root/'source_snapshot').mkdir(parents=True)
    print('OUTPUT_ROOT='+str(root),flush=True)
    hashes=snapshot(root)
    shutil.copyfile(args.output_parent/'history_catalog.json',root/'history_catalog.json')
    old_protocol=json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
    for name in ['streaming_rtrl.py','linear_readout.py','model.py','compute_budget.py']:
        assert hashes[name]==old_protocol['source_sha256'][name], name
    write_json(root/'mechanism_validation.json',{'passed':True,'tests_run':tests.testsRun})
    input_sequences=sequences()
    write_json(root/'input_sequences.json',input_sequences)
    write_json(root/'protocol.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'kinds':KINDS,'tasks':TASKS,'seeds':SEEDS,'length':600,'runs':81,'predictions':48600,
        'neurons':20,'classes':10,'input_per_step':1,'persistent_state':True,'history_buffer':False,'numeric_state_elements_by_kind':{'baseline_rtrl20':420,'fixed_input_memory20':450,'learned_input_memory20':450},
        'training_order':'predict before target; learn once if prior input exists; advance observed digit once',
        'optimizer':'Adam lr=.015, ordinary WD=1e-4, trace/input retention WD=0, global gradient clip=1, input prox=.02',
        'input_retention_initial':.8,'input_memory_sources':'10 external onehot channels + 20 previous neuron outputs; one state per source shared across receivers','input_memory_formula':'h=tanh(Win*x+(R+eta*F_previous)*h_previous+V*m_previous+b); m_new=rho*m_previous+(1-rho)*concat(x,h_previous)','memory_readout_initial':'V=0; original common parameters match baseline' ,
        'trace_formula':'F=r*F_previous+(1-r)*outer(h_new,h_previous)',
        'credit':'RTRL exact for fixed parameters; conventional carried sensitivity during online Adam; no derivative through Adam history',
        'comparison':'paired same neurons/seed/updates and common initial parameters; arithmetic budgets differ',
        'baseline_reference':str(PREVIOUS),'baseline_retrained':True,'checkpoint_interval':100,
        'source_sha256':hashes,'python':sys.version,'torch':str(torch.__version__),
        'git_branch_at_start':subprocess.check_output(['git','branch','--show-current'],text=True).strip()})
    write_json(root/'compute_profile.json',measure_models())
    summaries=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,str(root),kind,task,input_sequences[task],seed)
                 for kind in KINDS for task in TASKS for seed in SEEDS]
        for future in as_completed(futures):
            row=future.result();summaries.append(row)
            print('RUN='+json.dumps(row),flush=True)
    summaries.sort(key=lambda s:(s['kind'],s['task'],s['seed']))
    write_json(root/'runs_summary.json',summaries)
    aggregate=[]
    for kind in KINDS:
        for task in TASKS:
            rows=[r for r in summaries if r['kind']==kind and r['task']==task]
            assert len(rows)==3
            aggregate.append({'kind':kind,'task':task,'count':1800,'hits':sum(r['hits'] for r in rows),
                **{key:statistics.mean(r[key] for r in rows) for key in ['accuracy','first100_accuracy','last100_accuracy','mean_nll']},
                'seed_accuracy_min':min(r['accuracy'] for r in rows),'seed_accuracy_max':max(r['accuracy'] for r in rows)})
    write_json(root/'aggregate.json',aggregate)
    fields=['kind','task','seed','parameters','count','hits','accuracy','first100_accuracy','last100_accuracy','mean_nll','optimizer_updates','input_retention_gradient_nonzero_updates','run_directory']
    with (root/'runs_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)
    write_json(root/'completed.json',{'success':True,'new_runs':81,'new_predictions':48600,'all_three_variants_retrained':True})
    write_json(root/'checksums.json',{str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()})
    print('COMPLETE='+str(root),flush=True)

if __name__=='__main__': main()
