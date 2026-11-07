"""Frozen, paired retest with fresh distractors and rearranged clue blocks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import shutil

import torch

from adaptive_graph_eval import MODES, source_edits
from adaptive_memory_graph import AdaptiveMemoryGraph
from query_pair_eval import SEEDS, SPARSE, TASKS, expected, pair, training_batches
from sparse_input_memory import SparseInputMemory
from standard_memory_eval import Standard, digest, write


BASELINES = (SPARSE, 'GRU', 'TCN-512')
MODELS = (*BASELINES, *MODES)
GAPS = (8, 32, 128, 256, 512, 1024)
LAYOUTS = ('compact', 'scattered')
PAIRS = 32


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_data(task, reference, graph):
    shared = read(reference/f'{task}-data.json')
    assert digest(reference/f'{task}-data.json') == digest(graph/f'{task}-data.json')
    held = {tuple(x) for x in shared['forbidden_prefixes']}
    training = set(held)
    # Exclude both original test/validation prefixes and all training content,
    # even if a graph model saw it with a random front prefix.
    for seed, epoch in itertools.product(SEEDS, range(read(graph/'protocol.json')['epochs'])):
        for batch in training_batches(task, 'delay', seed+epoch*100003, held):
            training.update(tuple(x[:-1]) for x,_ in batch)
    rng = random.Random(827669)
    blocks = 3 if task == 'binding' else 2
    result = {layout:{} for layout in LAYOUTS}
    for gap in GAPS:
        canonical = []
        for label in list(range(4))*(PAIRS//4):
            while True:
                x, xp, y, yp = pair(task,gap,label,rng)
                if tuple(x[:-1]) not in training:
                    break
            training.add(tuple(x[:-1]))
            raw = [(x,y), (xp,yp)]
            for idx,value in source_edits(task,x):
                changed = x.copy()
                changed[idx] = value
                raw.append((changed,expected(task,changed)))
            canonical.append(raw)
        rng.shuffle(canonical)
        for layout in LAYOUTS:
            samples = []
            for raw in canonical:
                order = list(range(blocks))
                pads = [[] for _ in order]
                if layout == 'scattered':
                    # Derange the clue order and spread the blocks across
                    # different positions while preserving each local pair.
                    while order == list(range(blocks)):
                        rng.shuffle(order)
                    extra = 128 if blocks == 3 else 96
                    front = rng.randrange(32,65)
                    cuts = [front,extra] if blocks == 2 else [front,rng.randrange(front+8,extra-7),extra]
                    pads = [rng.choices((4,5),k=b-a) for a,b in zip([0,*cuts[:-1]],cuts,strict=True)]
                variants = []
                spans = []
                cursor = 0
                for i, noise in zip(order,pads,strict=True):
                    cursor += len(noise)
                    spans.append((i,cursor,cursor+1))
                    cursor += 2
                for x,y in raw:
                    flat = []
                    for i, noise in zip(order,pads,strict=True):
                        flat.extend(noise)
                        flat.extend(x[2*i:2*i+2])
                    variants.append((flat+x[2*blocks:],y))
                samples.append(dict(variants=variants, spans=spans, extra=sum(map(len,pads))))
            result[layout][str(gap)] = samples
    return result


def load_model(name, task, seed, reference, graph):
    if name in MODES:
        path = graph/f'{name}_{task}_{seed}.pt'
        model = AdaptiveMemoryGraph(name)
    else:
        paths = list((reference/'evaluation').glob(f'delay_{name}_{task}_{seed}_*.pt'))
        if len(paths) != 1:
            raise ValueError(f'Expected one frozen checkpoint for {name}/{task}/{seed}')
        path = paths[0]
        model = SparseInputMemory() if name == SPARSE else Standard(name,'matched')
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval(), path


@torch.no_grad()
def run_one(root, name, task, seed, reference, graph):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model,path = load_model(name,task,seed,reference,graph)
    data = read(root/f'{task}-fresh.json')
    groups = []
    for layout,gap in itertools.product(LAYOUTS,GAPS):
        inputs = data[layout][str(gap)]
        rows = []
        for i in range(0,len(inputs),8):
            batch = inputs[i:i+8]
            count = len(batch[0]['variants'])
            digits = torch.tensor([x for sample in batch for x,_ in sample['variants']])
            if name == 'adaptive':
                logits,edges = model(digits,return_edges=True)
                previous = [e[::count] for e in edges]
                frozen = model(digits[1::count],forced=previous).softmax(-1).tolist()
            else:
                logits = model(digits)
            probs = logits.softmax(-1).reshape(len(batch),count,10).tolist()
            for j,row in enumerate(probs):
                result = dict(probabilities=row)
                if name == 'adaptive':
                    result['frozen'] = frozen[j]
                    result['first_edge_sets'] = [e[j*count:j*count+2].tolist() for e in edges]
                rows.append(result)
        groups.append(dict(layout=layout,gap=gap,rows=rows))
    file = root/f'{name}_{task}_{seed}.json'
    write(file,dict(model=name,task=task,seed=seed,checkpoint=str(path.resolve()),checkpoint_sha256=digest(path),groups=groups))
    return file.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    (root/'source').mkdir()
    here = Path(__file__).resolve().parent
    for name in ('adaptive_graph_retest.py','adaptive_graph_eval.py','adaptive_memory_graph.py','query_pair_eval.py',
                 'standard_memory_eval.py','long_memory_eval.py','sparse_input_memory.py','streaming_rtrl.py',
                 'input_memory_rtrl.py','linear_readout.py','../model.py'):
        path = here/name
        shutil.copyfile(path,root/'source'/path.name)
    for task in TASKS:
        write(root/f'{task}-fresh.json',test_data(task,args.reference,args.graph))
    write(root/'protocol.json',dict(models=MODELS,tasks=TASKS,layouts=LAYOUTS,gaps=GAPS,seeds=SEEDS,
          pairs=PAIRS,reference=str(args.reference.resolve()),graph=str(args.graph.resolve()),
          source_sha256={p.name:digest(p) for p in (root/'source').iterdir()},
          data_sha256={p.name:digest(p) for p in root.glob('*-fresh.json')}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one,root,*j,args.reference.resolve(),args.graph.resolve())
                for j in itertools.product(MODELS,TASKS,SEEDS)]
        for f in as_completed(jobs):
            print(f.result(),flush=True)


if __name__ == '__main__':
    main()
