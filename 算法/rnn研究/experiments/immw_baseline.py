"""Matched-budget dense RNN and GRU reference training for symbolic IMMW."""

import argparse
from hashlib import sha256
from itertools import product
from pathlib import Path
import random
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from evolving_rnn import EvolvingRNN
from evolving_rnn_eval import launch
from frontier_eval import SEEDS, setup, write
from immw import VOCAB, TRAIN_MASKS, Factors, ask, cases, create_world


MODES = ('dense','GRU')
BATCH = 16
DEV_GAPS = (16,32,64,128,256)
TRAIN_GAPS = (16,32,64)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB,24)
        self.core = nn.GRU(24,290,batch_first=True)
        self.head = nn.Linear(290,4)

    def forward(self,x):
        states,_ = self.core(self.embedding(x))
        return self.head(states[:,-1])


def train_batch(seed,step):
    rng = random.Random(seed*39041+step)
    gap = TRAIN_GAPS[(step+seed)%len(TRAIN_GAPS)] if step>240 else 16
    capacity = (2,4)[(step//3+seed)%2] if step>240 else 2
    rate = rng.choice((4,16,64))
    factors = Factors(gap,rate,capacity)
    items = []
    for i in range(BATCH//2):
        identity = f'immw/v1/train-baseline/{seed}/{step}/{i}'
        world = create_world(factors,identity)
        masks = (1,2,4,8) if step<=120 else TRAIN_MASKS
        q = ask(world,masks[(i+step)%len(masks)])
        variant = 1+rng.randrange(len(q['variants'])-1)
        items.extend((q['variants'][i],q['targets'][i]) for i in (0,variant))
    return tuple(torch.tensor(values) for values in zip(*items,strict=True))


@torch.inference_mode()
def evaluate(model):
    model.eval()
    rows = []
    for novel,gap in product((False,True),DEV_GAPS):
        episodes = list(cases('development',gaps=(gap,),rates=(16,),capacities=(2,),count=64,novel=novel))
        inputs = torch.tensor([v for _,_,_,q in episodes for v in q['variants']])
        p = torch.cat([model(x).argmax(-1) for x in inputs.split(24)]).tolist()
        offset = 0
        joint = []
        first = []
        for _,_,_,q in episodes:
            count = len(q['targets'])
            joint.append(tuple(p[offset:offset+count])==q['targets'])
            first.append(p[offset]==q['targets'][0])
            offset += count
        rows.append(dict(novel=novel,gap=gap,joint=mean(joint),single=mean(first)))
    return rows


def train(root,mode,seed,steps):
    setup(seed)
    model = EvolvingRNN(512,'dense',vocab=VOCAB) if mode=='dense' else GRU()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    digest = sha256()
    folder = root/f'{mode}-{seed}'
    folder.mkdir()
    history = []
    for step in range(1,steps+1):
        model.train()
        x,y = train_batch(seed,step)
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x),y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step in (240,600) or step==steps:
            scores = evaluate(model)
            history.append(dict(step=step,scores=scores,loss=loss.item()))
            write(folder/'development.json',history)
            print(mode,seed,step,[round(c['joint'],3) for c in scores],flush=True)
    torch.save(model.state_dict(),folder/'weights.pt')
    return dict(mode=mode,seed=seed,steps=steps,train_sha256=digest.hexdigest(),history=history,
                parameters=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    jobs = [(train,(args.output,m,s,args.steps)) for m,s in product(MODES,SEEDS)]
    write(args.output/'protocol.json',dict(modes=MODES,seeds=SEEDS,steps=args.steps,batch=BATCH,
          train_gaps=TRAIN_GAPS,development_gaps=DEV_GAPS))
    launch(args.output,jobs,args.workers,'development.json')


if __name__=='__main__':
    main()
