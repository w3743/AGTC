"""Trace source edits through the slot state and score saved training milestones."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
from statistics import mean

import torch

from frontier_assess import paths
from frontier_data import TASKS, episode, split_episodes
from frontier_eval import evaluate, read, score, setup, write
from frontier_models import build
from slot_memory_graph import CHUNK


@torch.no_grad()
def inspect_slots(checkpoint,seed):
    setup(seed)
    model=build('adaptive').eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    rows=[]
    for task,gap in itertools.product(TASKS,(0,16,128,384,768)):
        records=[]
        for i in range(12):
            difficulty=2 if task in ('pointer','arithmetic') else 1
            row=episode(task,'scattered',gap,difficulty,f'probe/{seed}/{task}/{i}')
            x=torch.tensor(row['variants'])
            logits=model(x)
            hit=(logits.argmax(-1)==torch.tensor(row['targets'])).tolist()
            result=dict(single=mean(hit),joint=all(hit))
            clue_end=x.shape[1]-2-gap
            cut=((clue_end+CHUNK-1)//CHUNK)*CHUNK
            if cut<=x.shape[1]-2:
                early=model.push(model.initial(len(x),x.device),x[:,:cut])
                late=model.push(early,x[:,cut:-2])
                state=late.slots if not late.pending.shape[1] else model.write_block(late.slots,late.previous,late.pending)[0]
                result.update(early=float((early.slots[0]-early.slots[2]).square().mean().sqrt()),
                              late=float((state[0]-state[2]).square().mean().sqrt()))
            records.append(result)
        rows.append(dict(seed=seed,task=task,gap=gap,**{k:mean(r[k] for r in records) for k in records[0]}))
    return rows


def milestone(name,path,round_id,seed):
    setup(seed)
    m=build(name)
    m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    data=list(split_episodes('validation',round_id,2))
    hits=evaluate(m,data)
    return dict(name=name,seed=seed,questions=int(path.stem),joint=mean(score(a,b['targets'])[0] for a,b in zip(hits,data,strict=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--slots-only',action='store_true')
    args=p.parse_args(); protocol=read(args.root/'protocol.json')
    ckpts=paths(args.root,protocol,read(args.root/'selection.json'))
    rows=[]
    for seed in protocol['seeds']:
        rows.extend(inspect_slots(ckpts['adaptive',seed],seed))
    write(args.root/'slot-probe.json',rows)
    if not args.slots_only:
        with ProcessPoolExecutor(max_workers=8) as pool:
            jobs=[pool.submit(milestone,n,f,protocol['round'],s) for (n,s),path in ckpts.items()
                  for f in sorted(path.parent.glob('*.pt'))]
            write(args.root/'milestones.json',[f.result() for f in jobs])
    print('Probe and available training milestones saved.',flush=True)


if __name__=='__main__':
    main()
