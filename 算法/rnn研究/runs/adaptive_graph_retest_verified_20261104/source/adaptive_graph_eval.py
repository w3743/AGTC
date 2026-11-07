"""Matched-data experiment for learned query-to-history topology."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from adaptive_memory_graph import AdaptiveMemoryGraph
from query_pair_eval import GAPS, SEEDS, TASKS, expected, training_batches
from standard_memory_eval import digest, write


MODES = ('adaptive', 'fixed', 'dense')
EPOCHS = 4
LR = .003
SHIFT = 64


def source_edits(task, x):
    if task == 'binding':
        idx = x[:6].index(x[-1])+1
        value = next(v for v in range(4) if v not in x[1:6:2])
        return [(idx, value)]
    return [(1, (x[1]+1)%4), (3, 4-x[3])]


def candidates(task, x):
    if task == 'binding':
        return [x[:6].index(x[-1])+1]
    return [1, 3]


@torch.no_grad()
def evaluate(model, task, data):
    model.eval()
    scores = []
    for shift, gap in itertools.product((0, SHIFT), GAPS):
        rng = random.Random(256134+gap)
        rows = []
        pairs = data['test'][str(gap)]
        for start in range(0, len(pairs), 16):
            batch = pairs[start:start+16]
            prefixes = [rng.choices((4,5), k=shift) for _ in batch]
            x0 = torch.tensor([prefix+x for prefix,(x,_,_,_) in zip(prefixes,batch,strict=True)])
            x1 = torch.tensor([prefix+xp for prefix,(_,xp,_,_) in zip(prefixes,batch,strict=True)])
            logits0, edges0 = model(x0, return_edges=True)
            logits1, edges1 = model(x1, return_edges=True)
            edited = [[] for _ in batch]
            for i, (x,_,_,_) in enumerate(batch):
                for index, value in source_edits(task, x):
                    clone = x.copy()
                    clone[index] = value
                    edited[i].append((prefixes[i]+clone, expected(task,clone)))
            edit_logits = model(torch.tensor([x for edits in edited for x,_ in edits])).reshape(len(batch),-1,10)
            forced = model(x1, forced=edges0) if model.mode == 'adaptive' else None
            for i,(x,xp,y,yp) in enumerate(batch):
                pos0 = [p+shift for p in candidates(task,x)]
                pos1 = [p+shift for p in candidates(task,xp)]
                a = logits0[i].softmax(-1).tolist()
                b = logits1[i].softmax(-1).tolist()
                item = dict(y=y, yp=yp, probabilities=a, swapped_probabilities=b,
                            source_edits=[dict(y=target, probabilities=probs.softmax(-1).tolist())
                                          for (_,target),probs in zip(edited[i],edit_logits[i],strict=True)])
                if model.mode != 'dense':
                    item.update(edges=[e[i].tolist() for e in edges0], swapped_edges=[e[i].tolist() for e in edges1],
                                relevant=pos0, swapped_relevant=pos1)
                if forced is not None:
                    item['frozen_edge_probabilities'] = forced[i].softmax(-1).tolist()
                rows.append(item)
        scores.append(dict(shift=shift, gap=gap, rows=rows))
    return scores


def run_one(root, mode, task, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    data = json.loads((root/f'{task}-data.json').read_text(encoding='utf-8'))
    forbidden = {tuple(x) for x in data['forbidden_prefixes']}
    model = AdaptiveMemoryGraph(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    augmentation = random.Random(172903+seed)
    losses = []
    start = perf_counter()
    for epoch in range(EPOCHS):
        for batch in training_batches(task, 'delay', seed+epoch*100003, forbidden):
            # Choose one length per batch, but independent distractors per pair.
            shift = augmentation.randrange(33)
            rows = []
            for i in range(0, len(batch), 2):
                prefix = augmentation.choices((4,5), k=shift)
                rows.extend((prefix+x,y) for x,y in batch[i:i+2])
            x,y = zip(*rows)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(torch.tensor(x)), torch.tensor(y))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            losses.append(loss.item())
    path = root/f'{mode}_{task}_{seed}.json'
    train_seconds = perf_counter()-start
    write(path, dict(mode=mode, task=task, seed=seed, train_seconds=train_seconds, losses=losses,
                     parameters=sum(p.numel() for p in model.parameters()), scores=evaluate(model,task,data)))
    torch.save(model.state_dict(), path.with_suffix('.pt'))
    return path.name, round(train_seconds,2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    here = Path(__file__).resolve().parent
    for name in ('adaptive_graph_eval.py', 'adaptive_memory_graph.py', 'query_pair_eval.py', 'sparse_input_memory.py',
                 'standard_memory_eval.py', 'long_memory_eval.py', 'input_memory_rtrl.py', 'streaming_rtrl.py',
                 'linear_readout.py', '../model.py'):
        path = here/name
        shutil.copyfile(path, source/path.name)
    for task in TASKS:
        shutil.copyfile(args.reference/f'{task}-data.json',root/f'{task}-data.json')
    write(root/'protocol.json', dict(modes=MODES, tasks=TASKS, seeds=SEEDS, gaps=GAPS, epochs=EPOCHS, lr=LR,
            batch=16, train_questions=EPOCHS*2000, train_shifts=list(range(33)), test_shifts=[0,SHIFT],
            reference=str(args.reference.resolve()), torch=str(torch.__version__),
            source_sha256={p.name:digest(p) for p in source.iterdir()},
            data_sha256={p.name:digest(p) for p in root.glob('*-data.json')}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_one,root,*job) for job in itertools.product(MODES,TASKS,SEEDS)]
        for f in as_completed(pending):
            print(*f.result(), flush=True)


if __name__ == '__main__':
    main()
