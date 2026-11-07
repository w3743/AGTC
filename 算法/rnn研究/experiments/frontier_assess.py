"""Audit predictions, assess the four gates, and protect sealed test access."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

import numpy as np
import torch

from frontier_data import VALUES, batch_episodes, oracle, split_episodes
from frontier_eval import digest, evaluate, read, score, setup, write
from frontier_models import STANDARDS, build


def paths(root,protocol,selection):
    return {(n,s):root/'runs'/f'{n}_{s}_{selection[n]:g}'/f'{protocol["steps"]*16}.pt'
            for n,s in itertools.product(protocol['models'],protocol['seeds'])}


def evaluate_saved(name,checkpoint,rows,sealed):
    setup(2)
    model=build(name)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    return evaluate(model,rows,intervene=name=='adaptive',batch=2 if sealed else 8)


def predicate(row,task):
    if task=='arithmetic':
        return row['task']=='arithmetic' and row['layout']=='scattered'
    return row['task']=='pointer' and row['difficulty']>=2


def metrics(rows,predictions,names,seeds):
    hits={}; frozen=[]; cells=[]
    for name in names:
        per_seed=[]; singles=[]
        for seed in seeds:
            result=[]; alt=[]; single=[]
            for e,p in zip(rows,predictions[name,seed],strict=True):
                assert e['id']==p['id']
                for probs in [*p['probabilities'],*([p['frozen']] if 'frozen' in p else [])]:
                    assert len(probs)==4 and all(0<=v<=1 for v in probs) and abs(sum(probs)-1)<1e-6
                a,b=score(p,e['targets']); result.append(a); alt.append(b)
                single.append(mean(max(range(4),key=probs.__getitem__)==y
                                   for probs,y in zip(p['probabilities'],e['targets'],strict=True)))
                if name=='adaptive':
                    assert len(p['edges'])==4
                    assert all(len(head)==4 and len(set(head))==4 and all(0<=i<32 for i in head)
                               for hop in p['edges'] for branch in hop for head in branch)
            per_seed.append(result); singles.append(single)
            if name=='adaptive': frozen.append(alt)
        hits[name]=np.asarray(per_seed,dtype=float)
        for cell,group in itertools.groupby(enumerate(rows),key=lambda t:(t[1]['task'],t[1]['layout'],t[1]['gap'],t[1]['difficulty'])):
            ix=[i for i,_ in group]
            values=hits[name][:,ix].mean(axis=1)
            cells.append(dict(name=name,task=cell[0],layout=cell[1],gap=cell[2],difficulty=cell[3],
                              joint=float(values.mean()),single=float(np.asarray(singles)[:,ix].mean()),
                              seed_values=values.tolist(),cases=len(ix)))
    return hits,np.asarray(frozen,dtype=float),cells


def paired_ci(rows,hits,mask):
    # Preserve task/length composition and three paired training seeds in each draw.
    rng=np.random.default_rng(32191)
    boot=np.zeros((len(STANDARDS)+1,4096))
    arrays=np.stack([hits[n].mean(axis=0) for n in ('adaptive',*STANDARDS)])
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if mask[i]: groups[r['task'],r['layout'],r['gap'],r['difficulty']].append(i)
    for ix in groups.values():
        take=rng.choice(ix,size=(4096,len(ix)),replace=True)
        boot+=arrays[:,take].sum(axis=-1)
    delta=(boot[0]-boot[1:].max(axis=0))/sum(map(len,groups.values()))
    return [float(x) for x in np.quantile(delta,[.025,.975])]


def gates(rows,hits,frozen,profiles):
    complex_scores=[]
    for task in ('arithmetic','pointer'):
        mask=np.array([predicate(r,task) for r in rows]); rates={n:float(hits[n][:,mask].mean()) for n in ('adaptive',*STANDARDS)}
        leader=max(STANDARDS,key=rates.__getitem__); delta=rates['adaptive']-rates[leader]
        ci=paired_ci(rows,hits,mask)
        complex_scores.append(dict(task=task,adaptive=rates['adaptive'],baseline=leader,baseline_score=rates[leader],
                                   delta=delta,ci=ci,passed=delta>=.05 and ci[0]>0))
    retention=[]
    gaps=sorted({r['gap'] for r in rows})[-2:]
    for gap in gaps:
        mask=[r['task']=='lookup' and r['gap']==gap for r in rows]
        ours=float(hits['adaptive'][:,mask].mean())
        baseline=max(float(hits[n][:,mask].mean()) for n in STANDARDS)
        retention.append(dict(gap=gap,adaptive=ours,baseline=baseline,passed=ours>=.98 and ours>=baseline-.01))
    mask=[predicate(r,'arithmetic') or predicate(r,'pointer') for r in rows]
    impact=float(hits['adaptive'][:,mask].mean()-frozen[:,mask].mean())
    resource=None
    if profiles:
        index={r['name']:r for r in profiles}; maxgap=max(gaps)
        m=[r['gap']==maxgap for r in rows]
        acc={n:float(hits[n][:,m].mean()) for n in ('adaptive',*STANDARDS)}
        eligible=[n for n in STANDARDS if abs(acc[n]-acc['adaptive'])<=.02]
        baseline=(min(eligible,key=lambda n:(index[n]['peak_increment_mib'],index[n]['latency_ms']))
                  if eligible else max(STANDARDS,key=acc.__getitem__))
        a,b=index['adaptive'],index[baseline]
        resource=dict(baseline=baseline,accuracy_delta=acc['adaptive']-acc[baseline],
                      memory_ratio=a['peak_increment_mib']/b['peak_increment_mib'] if b['peak_increment_mib'] else None,
                      latency_ratio=a['latency_ms']/b['latency_ms'])
        resource['passed']=bool(resource['memory_ratio'] is not None and resource['memory_ratio']<=.75 and resource['latency_ratio']<=.85)
    return dict(complex=complex_scores,retention=retention,resource=resource,topology=dict(delta=impact,passed=impact>=.05),
                passed=all(r['passed'] for r in complex_scores+retention) and bool(resource and resource['passed']) and impact>=.05)


def profiles_for(p,checkpoints,gap):
    script=Path(__file__).with_name('frontier_profile.py')
    return [json.loads(subprocess.check_output([sys.executable,str(script),n,str(checkpoints[n,p['seeds'][0]]),str(gap)],text=True))
            for n in p['models']]


def shortcut_scores(rows,round_id):
    lookup,tail=defaultdict(Counter),defaultdict(Counter)
    for r in split_episodes('validation',round_id,16):
        for x,y in zip(r['variants'],r['targets'],strict=True):
            lookup[r['task'],x[-1]][y]+=1;tail[r['task'],tuple(x[-8:])][y]+=1
    scores=defaultdict(lambda:defaultdict(list))
    for r in rows:
        predictions=defaultdict(list)
        for x,y in zip(r['variants'],r['targets'],strict=True):
            q=lookup[r['task'],x[-1]].most_common(1)
            end=tail[r['task'],tuple(x[-8:])].most_common(1)
            guessed=q[0][0] if q else 0
            predictions['query_only'].append(guessed==y)
            predictions['tail_only'].append((end[0][0] if end else guessed)==y)
            ignored=oracle(x[:-1]+[50 if x[-1] in (50,51) else 52]) if r['task']=='arithmetic' else next((v-42 for v in reversed(x[:-2]) if v in VALUES),0)
            predictions['ignore_instruction'].append(ignored==y)
        for name,h in predictions.items(): scores[r['task']][name].append(all(h))
    return {task:{n:mean(h) for n,h in models.items()} for task,models in scores.items()}


def audit_inputs(rows,p,split):
    # The leading 64 distractors encode unique, split-independent case fingerprints.
    known=set()
    for seed in [91273,*p['seeds']]:
        for step in range(p['tune_steps'] if seed==91273 else p['steps']):
            for x,y in batch_episodes(seed,step):
                assert oracle(x)==y
                prefix=bytes(x[:64]); assert prefix not in known;known.add(prefix)
    for r in itertools.chain(split_episodes('validation',p['round'],8),rows):
        x=r['variants'][0]
        prefix=bytes(x[:64]);assert prefix not in known;known.add(prefix)
        assert all(bytes(v[:64])==prefix and oracle(v)==y for v,y in zip(r['variants'],r['targets'],strict=True))
        assert x[:-1]==r['variants'][1][:-1] and x[-1]!=r['variants'][1][-1]
        for v in r['variants'][2:]: assert sum(a!=b for a,b in zip(x,v,strict=True))==1 and x[-2:]==v[-2:]
    if split=='sealed':
        assert all(bytes(r['variants'][0][:64]) not in known for r in split_episodes('development',p['round']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--full-development',action='store_true')
    parser.add_argument('--seal',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    root=args.root.resolve(); p=read(root/'protocol.json'); selection=read(root/'selection.json')
    for name,checksum in p['source_sha256'].items():
        assert digest(root/'source'/name)==checksum
        if name!='frontier_eval.py': assert digest(Path(__file__).with_name(name))==checksum
    for name in p['models']:
        ranks=[]
        for rate in p['rates']:
            trial=read(root/'tune'/f'{name}_91273_{rate:g}'/'result.json')
            assert trial['seed']==91273 and trial['steps']==p['tune_steps'] and trial['lr']==rate
            ranks.append((trial['joint'],-trial['nll'],-rate))
        assert -max(ranks)[2]==selection[name]
    ckpts=paths(root,p,selection)
    for path in ckpts.values(): assert path.is_file(),path
    if args.seal:
        dev=read(root/'assessment-development.json')
        if not dev['gates']['passed'] or not dev['qualification_ready']:
            raise RuntimeError('Sealed set stays closed until all full development gates pass')
        locked={str(path.relative_to(root)):digest(path) for path in ckpts.values()}
        if dev['checkpoint_sha256']!=locked:
            raise RuntimeError('Development-qualified weights changed')
        split='sealed'; per_cell=128
        rows=list(split_episodes(split,p['round'],per_cell))
        lock=dict(round=p['round'],checkpoint_sha256=locked,protocol_sha256=digest(root/'protocol.json'),
                  test_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
        if (root/'sealed-lock.json').exists() and read(root/'sealed-lock.json')!=lock:
            raise RuntimeError('Sealed protocol changed')
        write(root/'sealed-lock.json',lock)
    else:
        split='development';per_cell=128 if args.full_development else p['dev_cases']
        rows=list(split_episodes(split,p['round'],per_cell))
    fresh=args.seal or args.full_development
    if fresh:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs={k:pool.submit(evaluate_saved,k[0],path,rows,args.seal) for k,path in ckpts.items()}
            predictions={k:f.result() for k,f in jobs.items()}
        dst=root/f'predictions-{split}'
        dst.mkdir(exist_ok=True)
        for (n,s),pred in predictions.items(): write(dst/f'{n}_{s}.json',pred)
    else:
        predictions={k:read(path.parent/'development.json') for k,path in ckpts.items()}
    audit_inputs(rows,p,split)
    hits,frozen,cells=metrics(rows,predictions,p['models'],p['seeds'])
    resources=profiles_for(p,ckpts,max(r['gap'] for r in rows)) if args.profile or args.seal else []
    res=gates(rows,hits,frozen,resources)
    output=dict(split=split,round=p['round'],cases_per_cell=per_cell,cases=len(rows),gates=res,cells=cells,
                shortcuts=shortcut_scores(rows,p['round']),resources=resources,selection=selection,
                qualification_ready=per_cell==128 and p['steps']==1000,
                checkpoint_sha256={str(path.relative_to(root)):digest(path) for path in ckpts.values()})
    write(root/f'assessment-{split}.json',output)
    print(split,json.dumps(res,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
