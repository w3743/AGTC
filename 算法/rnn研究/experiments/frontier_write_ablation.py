"""Paired 2,000-question screen of conservative slot overwrite initialization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import torch

from frontier_data import split_episodes
from frontier_eval import digest, evaluate, fit, read, score, setup, write
from frontier_models import build


def baseline(path,seed,round_id,cases):
    setup(seed)
    model=build('adaptive')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    rows=list(split_episodes('development',round_id,cases))
    return [score(a,b['targets'])[0] for a,b in zip(evaluate(model,rows),rows,strict=True)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=read(args.reference/'protocol.json'); lr=read(args.reference/'selection.json')['adaptive']
    cases=16; steps=125; bias=-3.0
    with ProcessPoolExecutor(max_workers=6) as pool:
        old={};new={};checksums={}
        for seed in protocol['seeds']:
            path=args.reference/'runs'/f'adaptive_{seed}_{lr:g}'/'2000.pt'
            checksums[str(path)]=digest(path)
            old[seed]=pool.submit(baseline,path,seed,protocol['round'],cases)
            new[seed]=pool.submit(fit,args.output,'adaptive',lr,seed,steps,protocol['round'],False,cases,bias)
        rows=[]
        for seed in protocol['seeds']:
            new_result=new[seed].result()[3]
            rows.append(dict(seed=seed,original_joint=mean(old[seed].result()),retention_joint=new_result['joint']))
            print(rows[-1],flush=True)
    write(args.output/'screen.json',dict(questions=steps*16,cases_per_cell=cases,lr=lr,write_bias=bias,
        rows=rows,reference_checkpoint_sha256=checksums,source_sha256={p.name:digest(p) for p in (
            Path(__file__),Path(__file__).with_name('frontier_eval.py'),Path(__file__).with_name('slot_memory_graph.py'))}))


if __name__=='__main__': main()
