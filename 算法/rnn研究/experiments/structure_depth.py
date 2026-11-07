"""Counterfactual depth and branching probe for learned symbol-graph routing."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import blake2b, sha256
from itertools import product
from pathlib import Path
import random
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from frontier_eval import SEEDS, setup, write
from frontier_models import build


WIDTHS = (2, 4)
DEPTHS = (1, 2, 4, 6, 8)
GAPS = (0, 256, 1024)
STEPS = 625
BATCH = 16
MODES = ('adaptive', 'fixed', 'GRU')
KEYS = tuple(range(10, 42))
VALUES = tuple(range(42, 46))
MARKERS = (4, 7)
COMMANDS = (50, 51)
PROGRAM = 8


def resolve(sequence, program_size=PROGRAM):
    edges = {}
    for i,token in enumerate(sequence[:-program_size-2]):
        if token in MARKERS:
            edges[sequence[i+1], MARKERS.index(token)] = sequence[i+2]
    state = sequence[-program_size-1]
    path = [state]
    for token in sequence[-program_size:]:
        if token:
            state = edges[state, COMMANDS.index(token)]
            path.append(state)
    return state-VALUES[0], path


def episode(width, depth, gap, identity):
    rng = random.Random(int.from_bytes(blake2b(identity.encode(), digest_size=16).digest(), 'little'))
    keys = rng.sample(KEYS, width*depth)
    layers = [keys[i:i+width] for i in range(0,len(keys),width)] + [list(VALUES)]
    nonce = blake2b(('prefix/'+identity).encode(),digest_size=8).digest()
    prefix = [1+((byte>>bit)&1) for byte in nonce for bit in range(8)]
    for _ in range(256):
        graph = {(src,j):dst for first,second in zip(layers[:-1],layers[1:],strict=True)
                 for src in first for j,dst in enumerate(rng.sample(second,2))}
        start = rng.choice(layers[0])
        route = [rng.randrange(2) for _ in range(depth)]
        states = [start]
        for step in route:
            states.append(graph[states[-1],step])
        branches = list(range(depth))
        rng.shuffle(branches)
        for branch in branches:
            alternative = route.copy()
            alternative[branch] ^= 1
            state = start
            for step in alternative:
                state = graph[state,step]
            if state != states[-1]:
                break
        else:
            continue
        break
    else:
        raise RuntimeError('Could not find a discriminating route')

    # Swap one edge on the original path, using the alternative branch destination.
    touched = states[branch],route[branch]
    replacement = graph[states[branch],alternative[branch]]
    records = [(src,j,dst) for (src,j),dst in graph.items()]
    rng.shuffle(records)
    spacers = [rng.choices((1,2),k=3) for _ in records]
    tail = rng.choices((1,2),k=gap)

    def encode(steps, edit):
        sequence = prefix.copy()
        for (src,j,dst),spacer in zip(records,spacers,strict=True):
            sequence.extend(spacer)
            sequence.extend((MARKERS[j],src,replacement if edit and (src,j)==touched else dst))
        return sequence + tail + [6,start] + [COMMANDS[j] for j in steps] + [0]*(PROGRAM-depth)

    variants = [encode(route,False), encode(alternative,False), encode(route,True)]
    labels,paths = zip(*(resolve(v) for v in variants),strict=True)
    assert labels[0]!=labels[1] and labels[1]==labels[2]
    return dict(id=identity, width=width, depth=depth, gap=gap, variants=variants,
                targets=labels, paths=paths, records=records, branch=branch)


class SymbolWalk(nn.Module):
    """Learn the relation types. Symbol identity provides a common address space."""

    def __init__(self, adaptive):
        super().__init__()
        self.adaptive = adaptive
        self.marker = nn.Embedding(64,24)
        self.instruction = nn.Embedding(64,24)
        self.answer = nn.Embedding(64,4)
        self.program = nn.Sequential(nn.Linear(24,32),nn.GELU(),nn.Linear(32,4))

    def matrices(self, history, commands):
        kind = self.marker(history[:,:-2])
        q = self.instruction(commands)
        if not self.adaptive:
            q = q.mean(dim=1,keepdim=True).expand_as(q)
        weights = ((q @ kind.transpose(1,2))/24**.5).clamp(-12,12).exp()
        indices = history[:,1:-1]*64 + history[:,2:]
        rows = weights.new_zeros((len(history),len(COMMANDS),64*64))
        rows.scatter_add_(2,indices[:,None].expand_as(weights),weights)
        rows = rows.unflatten(2,(64,64))
        return rows/rows.sum(-1,keepdim=True).clamp_min(1e-8)

    def forward(self, x, trace=False, hard=False, forced=None, program_size=PROGRAM):
        history, program = x[:,:-program_size-2], x[:,-program_size:]
        tokens = torch.tensor(COMMANDS,device=x.device).expand(len(x),-1)
        graph = self.matrices(history,tokens)
        if hard:
            maximum,edges = graph.max(-1,keepdim=True)
            graph = torch.zeros_like(graph).scatter_(-1,edges,(maximum>0).to(graph.dtype))
        slots = (program==COMMANDS[1]).long() if forced is None else (forced==COMMANDS[1]).long()
        batch = torch.arange(len(x),device=x.device)
        mass = F.one_hot(x[:,-program_size-1],64).to(graph.dtype)
        visited = [mass.argmax(-1)] if trace else None
        for step in range(program_size):
            new_mass = (mass.unsqueeze(1) @ graph[batch,slots[:,step]]).squeeze(1)
            mass = torch.where(program[:,step:step+1]!=0,new_mass,mass)
            if trace:
                visited.append(mass.argmax(-1))
        # Empty rows can lose probability mass. Normalize at the final node.
        # Both variants also receive the same direct program summary.
        mass = mass/mass.sum(-1,keepdim=True).clamp_min(1e-8)
        instruction = self.instruction(program).mean(1)
        logits = mass @ self.answer.weight + self.program(instruction)
        return (logits,torch.stack(visited,dim=1)) if trace else logits


def train_examples(seed,step):
    depth = 1+(step%2) if step<100 else 1+(step%4)
    width = WIDTHS[(step//4)%2]
    gap = (0,16,64)[(step//8)%3]
    for i in range(BATCH//2):
        row = episode(width,depth,gap,f'structure-depth/v1/train/{seed}/{step}/{i}')
        for j in (0,1+(i+step)%2):
            yield row['variants'][j],row['targets'][j]


def heldout(count):
    return [episode(w,d,g,f'structure-depth/v1/test/{w}/{d}/{g}/{i}')
            for w,d,g in product(WIDTHS,DEPTHS,GAPS) for i in range(count)]


@torch.no_grad()
def evaluate(model, mode, rows, count):
    model.eval()
    predictions = []
    cells = []
    for start in range(0,len(rows),count):
        group = rows[start:start+count]
        hits,hard_hits,trace_hits,frozen_hits = [],[],[],[]
        for offset in range(0,len(group),16):
            chunk = group[offset:offset+16]
            x = torch.tensor([v for r in chunk for v in r['variants']])
            program_size = chunk[0].get('program_size',PROGRAM)
            logits = model(x) if mode=='GRU' else model(x,program_size=program_size)
            guess = logits.argmax(-1).reshape(len(chunk),3).tolist()
            if mode!='GRU':
                hard,trace = model(x,trace=True,hard=True,program_size=program_size)
                h = hard.argmax(-1).reshape(len(chunk),3).tolist()
                trace = trace.reshape(len(chunk),3,program_size+1).tolist()
                frozen = model(x[1::3],hard=True,forced=x[::3,-program_size:],
                               program_size=program_size).argmax(-1).tolist()
            for i,(r,p) in enumerate(zip(chunk,guess,strict=True)):
                hits.append(p==list(r['targets']))
                record = dict(id=r['id'],prediction=p)
                if mode!='GRU':
                    hard_hits.append(h[i]==list(r['targets']))
                    trace_hits.append(all(v[:r['depth']+1]==list(path)
                                          for v,path in zip(trace[i],r['paths'],strict=True)))
                    frozen_hits.append(h[i][0]==r['targets'][0] and h[i][2]==r['targets'][2]
                                       and frozen[i]==r['targets'][1])
                    record.update(hard=h[i],trace=trace[i],frozen=frozen[i])
                predictions.append(record)
        w,d,g = (group[0][k] for k in ('width','depth','gap'))
        cell = dict(width=w,depth=d,gap=g,joint=mean(hits))
        if mode!='GRU':
            cell.update(hard_joint=mean(hard_hits),trace_exact=mean(trace_hits),frozen_joint=mean(frozen_hits))
        cells.append(cell)
    return cells,predictions


def run(root, mode, seed, steps, count):
    setup(seed)
    model = build('GRU') if mode=='GRU' else SymbolWalk(mode=='adaptive')
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01 if mode!='GRU' else .003)
    digest = sha256()
    losses = []
    for step in range(steps):
        x,y = (torch.tensor(z) for z in zip(*train_examples(seed,step),strict=True))
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x),y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step+1)%125==0:
            print(mode,seed,step+1,round(mean(losses),3),flush=True)
            losses.clear()
    run_dir = root/f'{mode}-{seed}'
    run_dir.mkdir()
    torch.save(model.state_dict(),run_dir/'weights.pt')
    cells,predictions = evaluate(model,mode,heldout(count),count)
    write(run_dir/'predictions.json',predictions)
    return dict(mode=mode,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                training_sha256=digest.hexdigest(),cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--count',type=int,default=128)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    write(args.output/'protocol.json',dict(steps=args.steps,batch=BATCH,groups=args.count,
          train_depths=[1,2,3,4],train_gaps=[0,16,64],seeds=SEEDS,modes=args.modes))
    write(args.output/'cases.json',heldout(args.count))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run,args.output,m,s,args.steps,args.count) for m,s in product(args.modes,SEEDS)]
        for job in as_completed(jobs):
            results.append(job.result())
            write(args.output/'results.json',sorted(results,key=lambda r:(r['mode'],r['seed'])))


if __name__=='__main__':
    main()
