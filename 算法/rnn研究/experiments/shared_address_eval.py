"""Run the shared-address diagnosis independently of the sealed frontier suite."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F

from frontier_data import episode
from frontier_eval import CHECKPOINTS, SEEDS, digest, evaluate, score, setup, write
from frontier_models import build
from shared_address_graph import SharedAddressGraph


MODELS = ('shared', 'permuted', 'GRU', 'TCN')
LAYOUTS = ('compact', 'scattered')
TRAIN_GAPS = (0, 16, 64, 256)
TEST_GAPS = (0, 384, 768)
RATE = .003
BATCH = 16


def cases(task, gaps, partition, count):
    for layout, gap, difficulty in itertools.product(LAYOUTS, gaps, levels(task)):
        for i in range(count):
            identity = f'shared-address/v1/{partition}/{task}/{layout}/{gap}/{difficulty}/{i}'
            yield episode(task, layout, gap, difficulty, identity)


def levels(task):
    return (1,) if task == 'lookup' else (2, 3)


def training_batch(task, seed, step):
    cells = list(itertools.product(LAYOUTS, TRAIN_GAPS, levels(task)))
    layout, gap, difficulty = cells[(step - 1) % len(cells)]
    rng = random.Random(seed * 4099 + step)
    for i in range(BATCH):
        identity = f'shared-address/v1/train/{task}/{seed}/{step}/{i}'
        row = episode(task, layout, gap, difficulty, identity)
        variant = rng.randrange(len(row['variants']))
        yield row, variant


def prefix(row):
    return bytes(row['variants'][0][:64])


def aggregate(rows, predictions):
    result = []
    pairs = zip(rows, predictions, strict=True)
    for cell, group in itertools.groupby(pairs, key=lambda pair: tuple(pair[0][k] for k in ('layout','gap','difficulty'))):
        samples = list(group)
        assert all(r['id'] == p['id'] for r, p in samples)
        scores = [score(p, r['targets']) for r, p in samples]
        variants = [
            [max(range(4), key=probs.__getitem__) == label for probs, label in zip(p['probabilities'], r['targets'], strict=True)]
            for r, p in samples
        ]
        frozen = [f for _, f in scores if f is not None]
        result.append(dict(
            layout=cell[0], gap=cell[1], difficulty=cell[2], groups=len(samples),
            joint=mean(j for j, _ in scores), frozen=mean(frozen) if frozen else None,
            variant_accuracy=[mean(col) for col in zip(*variants, strict=True)],
        ))
    return result


def run_one(root, task, name, seed, steps):
    start = perf_counter()
    setup(seed)
    graph = name in ('shared', 'permuted')
    model = SharedAddressGraph(aligned=name == 'shared') if graph else build(name)
    params = sum(p.numel() for p in model.parameters())
    assert 90_000 <= params <= 110_000, (name, params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE)
    dev = list(cases(task, (0,), 'development', 64))
    evaluation = list(cases(task, TEST_GAPS, 'holdout', 128))
    isolated = {prefix(row) for row in dev + evaluation}
    assert len(isolated) == len(dev) + len(evaluation)
    run = root / f'{task}-{name}-{seed}'
    run.mkdir()
    loss_sum = 0.0
    loss_steps = 0
    train_hash = hashlib.sha256()
    checkpoints = []
    for step in range(1, steps + 1):
        batch = list(training_batch(task, seed, step))
        for row, _ in batch:
            assert prefix(row) not in isolated
        x = torch.tensor([r['variants'][i] for r, i in batch])
        y = torch.tensor([r['targets'][i] for r, i in batch])
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        loss_steps += 1
        if step in CHECKPOINTS or step == steps:
            weights = run / f'weights-{step}.pt'
            torch.save(model.state_dict(), weights)
            development = aggregate(dev, evaluate(model, dev, intervene=graph))
            point = dict(questions=step*BATCH, loss=loss_sum/loss_steps, development=development, sha256=digest(weights))
            checkpoints.append(point)
            write(run / 'development.json', checkpoints)
            loss_sum, loss_steps = 0.0, 0
            print(task, name, seed, step, round(point['loss'], 3),
                  [round(c['joint'], 3) for c in development], flush=True)
    predictions = evaluate(model, evaluation, intervene=graph)
    write(run / 'predictions.json', predictions)
    result = dict(
        task=task, model=name, seed=seed, steps=steps, parameters=params,
        seconds=perf_counter()-start, train_sha256=train_hash.hexdigest(),
        checkpoints=checkpoints, holdout=aggregate(evaluation, predictions),
    )
    write(run / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=1000, choices=range(1, 1001), metavar='1..1000')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--tasks', nargs='+', default=['lookup','arithmetic'], choices=('lookup','arithmetic'))
    parser.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS, choices=SEEDS)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    write(args.output / 'protocol.json', dict(
        tasks=args.tasks, models=args.models, seeds=args.seeds, steps=args.steps, batch=BATCH,
        learning_rate=RATE, optimizer='AdamW', train_gaps=TRAIN_GAPS, holdout_gaps=TEST_GAPS,
        groups_per_holdout_cell=128, torch=torch.__version__,
    ))
    for task in args.tasks:
        write(args.output / f'{task}-holdout.json', list(cases(task, TEST_GAPS, 'holdout', 128)))
    jobs = list(itertools.product(args.tasks, args.models, args.seeds))
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(run_one, args.output, *job, args.steps) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            write(args.output / 'results.json', sorted(results, key=lambda r: (r['task'], r['model'], r['seed'])))


if __name__ == '__main__':
    main()
