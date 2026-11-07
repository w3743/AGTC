"""Matched-budget experiment with development and sealed evaluation stages."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import itertools
import json
from pathlib import Path
import shutil
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F

from frontier_data import batch_episodes, split_episodes
from frontier_models import NAMES, build


SEEDS=(2,20261027,20261028)
TUNE_SEED=91273
RATES=(.001,.003,.01)
CHECKPOINTS=(125,500,1000)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def setup(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, rows, intervene=False, batch=16):
    model.eval()
    scored=[]
    for _, group in itertools.groupby(rows,key=lambda r:(r['task'],r['layout'],r['gap'],r['difficulty'])):
        group=list(group)
        for start in range(0,len(group),batch):
            episodes=group[start:start+batch]
            size=len(episodes[0]['variants'])
            x=torch.tensor([x for r in episodes for x in r['variants']])
            if intervene:
                logits,edges=model(x,return_edges=True)
                forced=model(x[1::size],forced=[e[::size] for e in edges]).softmax(-1).tolist()
            else:
                logits=model(x)
            probs=logits.softmax(-1).reshape(len(episodes),size,4).tolist()
            for i,(episode,p) in enumerate(zip(episodes,probs,strict=True)):
                row=dict(id=episode['id'],probabilities=p)
                if intervene:
                    row.update(frozen=forced[i],edges=[e[i*size:i*size+2].tolist() for e in edges])
                scored.append(row)
    return scored


def score(row,targets):
    hits=[max(range(4),key=p.__getitem__)==y for p,y in zip(row['probabilities'],targets,strict=True)]
    fixed=hits[0] and all(hits[2:]) and max(range(4),key=row['frozen'].__getitem__)==targets[1] if 'frozen' in row else None
    return all(hits),fixed


def fit(root,name,lr,seed,steps,round_id,tune,dev_cases,write_bias=None):
    setup(seed)
    model=build(name)
    if write_bias is not None:
        with torch.no_grad(): model.gate.bias.fill_(write_bias)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    directory=root/('tune' if tune else 'runs')/f'{name}_{seed}_{lr:g}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        return name,seed,lr,read(directory/'result.json')
    losses=[]
    started=perf_counter()
    for step in range(steps):
        data=list(batch_episodes(seed,step))
        x,y=zip(*data)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(torch.tensor(x)),torch.tensor(y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        losses.append(loss.item())
        if not tune and ((step+1) in CHECKPOINTS or step+1==steps):
            torch.save(model.state_dict(),directory/f'{16*(step+1)}.pt')
    split='validation' if tune else 'development'
    cases=8 if tune else dev_cases
    rows=list(split_episodes(split,round_id,cases))
    predictions=evaluate(model,rows,intervene=not tune and name=='adaptive')
    joint=mean(score(r,d['targets'])[0] for r,d in zip(predictions,rows,strict=True))
    nll=mean(F.nll_loss(torch.tensor(r['probabilities']).clamp_min(1e-30).log(),torch.tensor(d['targets'])).item()
             for r,d in zip(predictions,rows,strict=True))
    if not tune:
        write(directory/'development.json',predictions)
    result=dict(name=name,seed=seed,lr=lr,steps=steps,joint=joint,nll=nll,losses=losses,
                params=sum(p.numel() for p in model.parameters()),seconds=perf_counter()-started)
    write(directory/'result.json',result)
    return name,seed,lr,result


def train(args):
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    protocol=dict(steps=args.steps,tune_steps=args.tune_steps,dev_cases=args.dev_cases,round=args.round,
                  models=NAMES,seeds=SEEDS,rates=RATES,torch=str(torch.__version__))
    here=Path(__file__).resolve().parent
    paths=[here/n for n in ('frontier_eval.py','frontier_data.py','frontier_models.py','slot_memory_graph.py','adaptive_memory_graph.py')]
    protocol['source_sha256']={p.name:digest(p) for p in paths}
    protocol=json.loads(json.dumps(protocol))
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=protocol:
            raise ValueError('Resume requires identical parameters and model/data sources')
    else:
        (root/'source').mkdir(exist_ok=True)
        for path in paths:
            shutil.copyfile(path,root/'source'/path.name)
        write(root/'protocol.json',protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tune=[pool.submit(fit,root,n,lr,TUNE_SEED,args.tune_steps,args.round,True,args.dev_cases)
              for n,lr in itertools.product(NAMES,RATES)]
        best={}
        for job in as_completed(tune):
            n,_,lr,r=job.result()
            rank=(r['joint'],-r['nll'],-lr)
            if n not in best or rank>best[n][0]:
                best[n]=rank,lr
            print('TUNE',n,lr,round(r['joint'],4),flush=True)
        selection={n:best[n][1] for n in NAMES}
        write(root/'selection.json',selection)
        jobs=[pool.submit(fit,root,n,selection[n],s,args.steps,args.round,False,args.dev_cases)
              for n,s in itertools.product(NAMES,SEEDS)]
        for job in as_completed(jobs):
            n,s,lr,r=job.result()
            print('DEV',n,s,lr,round(r['joint'],4),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--tune-steps',type=int,default=125)
    p.add_argument('--dev-cases',type=int,default=128)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--round',type=int,default=1)
    args=p.parse_args()
    if not (1<=args.steps<=1000 and 1<=args.tune_steps<=1000 and 1<=args.dev_cases<=128 and args.workers>0 and args.round>0):
        p.error('Invalid experiment budget')
    train(args)


if __name__=='__main__':
    main()
