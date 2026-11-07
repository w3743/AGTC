"""Evaluate response to a changed query with all preceding input held fixed."""
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

from sparse_input_memory import SparseInputMemory
from standard_memory_eval import BATCH, RATES, SEEDS, TUNING_SEED, Standard, digest, evaluate, write


SPARSE = 'Sparse-1024 BPTT'
MODELS = ('GRU', 'TCN-512', SPARSE)
TASKS = ('binding', 'arithmetic')
GAPS = (0, 8, 32, 64, 128, 256)
TRAIN = {'zero': (0,), 'delay': GAPS[:-1]}
BUDGET = 2000


def pair(task, gap, y, rng):
    distractors = rng.choices((4, 5), k=gap)
    if task == 'binding':
        keys = rng.sample((6, 7, 8), 3)
        other = rng.sample([v for v in range(4) if v != y], 2)
        slot = rng.randrange(3)
        values = other.copy()
        values.insert(slot, y)
        switched = rng.choice([i for i in range(3) if i != slot])
        base = list(itertools.chain.from_iterable(zip(keys, values))) + distractors + [9]
        return base+[keys[slot]], base+[keys[switched]], y, values[switched]
    rule, b = rng.randrange(2), rng.choice((1, 3))
    a = (y - b if rule == 0 else y + b) % 4
    base = [6, a, 7, b] + distractors + [9]
    return base+[rule], base+[1-rule], y, (y+2)%4


def expected(task, x):
    if task == 'binding':
        return x[x[:6].index(x[-1])+1]
    return (x[1] + (1 if x[-1] == 0 else -1)*x[3]) % 4


def check_pair(task, gap, row):
    x, xp, y, yp = row
    assert x[:-1] == xp[:-1] and x[-1] != xp[-1] and y != yp
    assert len(x) == gap+(8 if task == 'binding' else 6) and x[-2] == 9
    assert y == expected(task, x) and yp == expected(task, xp)
    if task == 'binding':
        assert set(x[:6:2]) == {6, 7, 8} and len(set(x[1:6:2])) == 3
    else:
        assert x[3] % 2 == 1 and {x[-1], xp[-1]} == {0, 1}


def held_out(task):
    used = set()
    out = {}
    for split, seed in (('validation', 658773), ('test', 488561)):
        rng = random.Random(seed)
        out[split] = {}
        for gap in GAPS:
            if task == 'arithmetic' and gap == 0:
                # Exhaustive finite truth table, reported as a rule-fit diagnostic.
                rows = [([6,a,7,b,9,0], [6,a,7,b,9,1], (a+b)%4, (a-b)%4)
                        for a, b in itertools.product(range(4), (1, 3))]
            else:
                rows = []
                for label in list(range(4))*(6 if gap == 0 else 16):
                    while True:
                        row = pair(task, gap, label, rng)
                        key = tuple(row[0][:-1])
                        if key not in used:
                            break
                    used.add(key)
                    rows.append(row)
                rng.shuffle(rows)
            for row in rows:
                check_pair(task, gap, row)
            out[split][str(gap)] = rows
    out['forbidden_prefixes'] = sorted(used)
    return out


def training_batches(task, phase, seed, forbidden):
    gaps = TRAIN[phase]
    per_gap = BUDGET//len(gaps)
    buckets = {}
    for gap in gaps:
        rng = random.Random(95087 + 4471*seed + gap)
        labels = list(range(4))*(per_gap//8)
        rng.shuffle(labels)
        rows = []
        for label in labels:
            while True:
                x, xp, y, yp = pair(task, gap, label, rng)
                if tuple(x[:-1]) not in forbidden:
                    break
            rows.extend(((x,y), (xp,yp)))
        buckets[gap] = [rows[i:i+BATCH] for i in range(0, len(rows), BATCH)]
    rng = random.Random(90807 + seed)
    for i in range(per_gap//BATCH):
        for gap in rng.sample(gaps, len(gaps)):
            yield buckets[gap][i]


def run_job(root, phase, name, task, lr, seed, tuning):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    data = json.loads((root/f'{task}-data.json').read_text(encoding='utf-8'))
    forbidden = {tuple(x) for x in data['forbidden_prefixes']}
    model = SparseInputMemory() if name == SPARSE else Standard(name, 'matched')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    curve = []
    start = perf_counter()
    for batch in training_batches(task, phase, seed, forbidden):
        x, y = zip(*batch)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(torch.tensor(x)), torch.tensor(y))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        curve.append(loss.item())
    assert len(curve)*BATCH == BUDGET
    gaps = TRAIN[phase] if tuning or phase == 'zero' else GAPS
    rows = evaluate(model, data['validation' if tuning else 'test'], gaps, not tuning)
    folder = 'tuning' if tuning else 'evaluation'
    path = root/folder/f'{phase}_{name}_{task}_{seed}_{lr:g}.json'
    write(path, dict(phase=phase, model=name, task=task, lr=lr, seed=seed, scores=rows, loss=curve,
                     parameters=sum(p.numel() for p in model.parameters()), seconds=perf_counter()-start))
    if not tuning:
        torch.save(model.state_dict(), path.with_suffix('.pt'))
    return name, task, lr, path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'tuning', 'evaluation'):
        (root/name).mkdir()
    here = Path(__file__).resolve().parent
    for name in ('query_pair_eval.py', 'sparse_input_memory.py', 'standard_memory_eval.py', 'long_memory_eval.py',
                 'input_memory_rtrl.py', 'streaming_rtrl.py', 'linear_readout.py', '../model.py'):
        path = here/name
        shutil.copyfile(path, root/'source'/path.name)
    for task in TASKS:
        write(root/f'{task}-data.json', held_out(task))
    write(root/'protocol.json', dict(models=MODELS, tasks=TASKS, gaps=GAPS, train_gaps=TRAIN, seeds=SEEDS,
                                    tuning_seed=TUNING_SEED, rates=RATES, batch=BATCH, budget=BUDGET,
                                    torch=str(torch.__version__), source_sha256={p.name:digest(p) for p in (root/'source').iterdir()},
                                    data_sha256={p.name:digest(p) for p in root.glob('*-data.json')}))
    selections = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for phase in TRAIN:
            jobs = list(itertools.product(MODELS, TASKS))
            chosen = {}
            pending = [pool.submit(run_job, root, phase, model, task, lr, TUNING_SEED, True)
                       for model, task in jobs for lr in RATES]
            for i, future in enumerate(as_completed(pending), 1):
                model, task, lr, filename = future.result()
                rows = json.loads((root/'tuning'/filename).read_text(encoding='utf-8'))['scores']
                rank = (sum(r['paired_accuracy'] for r in rows), -sum(r['nll'] for r in rows), -lr)
                if (model, task) not in chosen or rank > chosen[model, task][0]:
                    chosen[model, task] = rank, lr
                if i%3 == 0:
                    print(phase, 'TUNING', i, '/', len(pending), flush=True)
            selections.extend(dict(phase=phase, model=m, task=t, lr=chosen[m,t][1]) for m,t in jobs)
            write(root/'selection.json', selections)
            pending = [pool.submit(run_job, root, phase, m, t, chosen[m,t][1], s, False)
                       for m,t in jobs for s in SEEDS]
            for i, future in enumerate(as_completed(pending), 1):
                future.result()
                if i%3 == 0:
                    print(phase, 'EVALUATION', i, '/', len(pending), flush=True)


if __name__ == '__main__':
    main()
