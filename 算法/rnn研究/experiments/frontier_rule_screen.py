"""Equal-budget short-rule curriculum screen across all eight models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from frontier_data import episode
from frontier_eval import digest, evaluate, read, score, setup, write
from frontier_models import build


def run(root,reference,name,seed,lr):
    setup(seed)
    model=build(name)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    for step in range(500):
        examples=[episode('lookup','compact',0,1,f'rule-train/{seed}/{step}/{i}') for i in range(16)]
        data=[(r['variants'][i%3],r['targets'][i%3]) for i,r in enumerate(examples)]
        x,y=zip(*data)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(torch.tensor(x)),torch.tensor(y))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
    checkpoint=root/f'{name}_{seed}.pt'
    torch.save(model.state_dict(),checkpoint)
    old=build(name)
    path=reference/'runs'/f'{name}_{seed}_{lr:g}'/'8000.pt'
    old.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    rows=[]
    for gap in (0,16,384,768):
        data=[episode('lookup','compact',gap,1,f'rule-dev/{i}') for i in range(32)]
        def stats(m):
            pred=evaluate(m,data,batch=8)
            joint=mean(score(r,d['targets'])[0] for r,d in zip(pred,data,strict=True))
            single=mean(max(range(4),key=p.__getitem__)==y for r,d in zip(pred,data,strict=True)
                        for p,y in zip(r['probabilities'],d['targets'],strict=True))
            return dict(joint=joint,single=single)
        rows.append(dict(gap=gap,original=stats(old),rule_curriculum=stats(model)))
    result=dict(name=name,seed=seed,lr=lr,rows=rows,source_checkpoint_sha256=digest(path),curriculum_checkpoint_sha256=digest(checkpoint))
    write(root/f'{name}_{seed}.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=read(args.reference/'protocol.json');selection=read(args.reference/'selection.json')
    with ProcessPoolExecutor(max_workers=10) as pool:
        jobs=[pool.submit(run,args.output,args.reference,n,s,selection[n]) for n,s in itertools.product(p['models'],p['seeds'])]
        rows=[]
        for f in as_completed(jobs):
            r=f.result(); rows.append(r)
            print(r['name'],r['seed'],[(x['gap'],x['rule_curriculum']['joint']) for x in r['rows']],flush=True)
    write(args.output/'screen.json',dict(questions=8000,steps=500,source_sha256=digest(Path(__file__)),rows=rows))


if __name__=='__main__': main()
