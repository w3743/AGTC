"""Paired, held-out long-memory evaluation for the current digit model."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from input_memory_rtrl import InputMemoryRTRL


TASKS = ('recall', 'binding', 'arithmetic')
GAPS = (8, 32, 64, 128, 256)
MODELS = ('Full-memory RTRL', 'RNN', 'GRU', 'LSTM', 'MLP-32', 'TCN-512', 'Transformer')
SEEDS = (2, 20261027, 20261028)
TRAIN_PER_GAP = 120
TEST_PER_GAP = 64


def episode(task, gap, target, rng):
    filler = rng.choices((4, 5), k=gap)
    if task == 'recall':
        prefix, query, changed = [6, target], [9], 1
    elif task == 'binding':
        keys = rng.sample([6, 7, 8], 3)
        key = rng.choice(keys)
        prefix = [item for k in keys for item in (k, target if k == key else rng.randrange(4))]
        query, changed = [9, key], prefix.index(key) + 1
    else:
        rule, b = rng.randrange(2), rng.randrange(4)
        a = (target - b if rule == 0 else target + b) % 4
        prefix, query, changed = [6, a, 7, b], [9, rule], 1
    original = prefix + filler + query
    paired = original.copy()
    paired[changed] = (paired[changed] + 1) % 4
    return original, paired, target, (target + 1) % 4


def oracle(task, sequence):
    if task == 'recall':
        return sequence[1]
    if task == 'binding':
        return sequence[sequence[:6].index(sequence[-1]) + 1]
    a, b = sequence[1], sequence[3]
    return (a + b if sequence[-1] == 0 else a - b) % 4


def datasets(task, seed):
    rng = random.Random(519077)
    test = {}
    forbidden = set()
    for gap in GAPS:
        test[gap] = []
        for label in range(4):
            for _ in range(TEST_PER_GAP // 4):
                while True:
                    row = episode(task, gap, label, rng)
                    pair = {tuple(row[0]), tuple(row[1])}
                    if not (pair & forbidden):
                        break
                forbidden.update(pair)
                test[gap].append(row)
        rng.shuffle(test[gap])
    rng = random.Random(seed + 77431)
    schedule = [(gap, i % 4) for gap in GAPS[:-1] for i in range(TRAIN_PER_GAP)]
    rng.shuffle(schedule)
    train = []
    for gap, label in schedule:
        while True:
            x, _, y, _ = episode(task, gap, label, rng)
            if tuple(x) not in forbidden:
                break
        train.append((x, y))
    assert all(oracle(task, x) == y for x, y in train)
    assert all(oracle(task, x) == y and oracle(task, xp) == yp
               for rows in test.values() for x, xp, y, yp in rows)
    return train, test


class Baseline(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name in ('RNN', 'GRU', 'LSTM'):
            cell, width = {'RNN': (nn.RNN, 29), 'GRU': (nn.GRU, 16), 'LSTM': (nn.LSTM, 13)}[name]
            self.core = cell(10, width, batch_first=True)
        elif name == 'MLP-32':
            width = 4
            self.core = nn.Linear(320, width)
        elif name == 'TCN-512':
            width = 8
            self.embed = nn.Linear(10, width)
            self.convs = nn.ModuleList(nn.Conv1d(width, width, 2, dilation=2**i) for i in range(9))
        else:
            width = 12
            self.embed = nn.Linear(10, width)
            self.core = nn.TransformerEncoderLayer(width, 2, 24, dropout=0, batch_first=True)
            positions = torch.arange(264)[:, None] / (10000 ** (torch.arange(0, width, 2) / width))
            self.register_buffer('positions', torch.stack((positions.sin(), positions.cos()), dim=-1).flatten(1))
        self.head = nn.Linear(width, 10)

    def forward(self, digits):
        x = F.one_hot(digits, 10).float()
        if self.name in ('RNN', 'GRU', 'LSTM'):
            h = self.core(x)[0][:, -1]
        elif self.name == 'MLP-32':
            x = x[:, -32:]
            h = self.core(F.pad(x, (0, 0, 32-x.shape[1], 0)).flatten(1)).tanh()
        elif self.name == 'TCN-512':
            x = self.embed(x).transpose(1, 2)
            for conv in self.convs:
                x = x + conv(F.pad(x, (conv.dilation[0], 0))).relu()
            h = x[:, :, -1]
        else:
            # All tokens in the episode are known before the query is scored.
            h = self.core(self.embed(x) + self.positions[:x.shape[1]])[:, -1]
        return self.head(h)


@torch.no_grad()
def frozen_rtrl(model, digits):
    """Batch the unchanged forward recurrence, omitting training sensitivities."""
    n = model.network
    batch = digits.shape[0]
    h = torch.zeros(batch, 20)
    m = torch.zeros(batch, 30)
    f = torch.zeros(batch, 20, 20)
    rho, r, eta = model.input_retention, n.trace_decay, n.eta
    for x in F.one_hot(digits, 10).float().transpose(0, 1):
        h0 = h
        h = torch.tanh(x @ n.input_weights.T + ((n.recurrent_weights + eta*f) @ h0[..., None]).squeeze(-1)
                       + m @ n.memory_weights.T + n.bias)
        f = r*f + (1-r)*(h[..., None]*h0[:, None, :])
        m = rho*m + (1-rho)*torch.cat((x, h0), dim=-1)
    return h @ n.W_out.T + n.b_out


def run_one(name, task, seed, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    train, test = datasets(task, seed)
    online = name == MODELS[0]
    model = InputMemoryRTRL(seed=seed) if online else Baseline(name)
    network = model.network if online else model
    if not online:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015, weight_decay=1e-4)
    start = perf_counter()
    train_curve = []
    for i, (x, y) in enumerate(train, 1):
        if online:
            model.reset_state()
            for digit in x:
                model.advance(digit)
            result = model.learn_target(y)
            loss = result['loss']
        else:
            optimizer.zero_grad(set_to_none=True)
            objective = F.cross_entropy(model(torch.tensor([x])), torch.tensor([y]))
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            loss = objective.item()
        train_curve.append(loss)
    train_seconds = perf_counter()-start
    network.eval()
    records = []
    with torch.no_grad():
        for gap, rows in test.items():
            inputs = torch.tensor([s for x, xp, _, _ in rows for s in (x, xp)])
            logits = frozen_rtrl(model, inputs) if online else model(inputs)
            assert torch.isfinite(logits).all()
            probs = logits.softmax(dim=-1).tolist()
            if online:
                # Check batched inference against the production advance/predict path.
                for j in (0, 1):
                    model.reset_state()
                    for digit in inputs[j].tolist():
                        model.advance(digit)
                    torch.testing.assert_close(model.predict(), torch.tensor(probs[j]), atol=2e-5, rtol=2e-5)
            for i, (x, xp, y, yp) in enumerate(rows):
                p, pp = probs[2*i:2*i+2]
                records.append(dict(gap=gap, index=i, digits=x, paired_digits=xp, target=y,
                                    paired_target=yp, probabilities=p, paired_probabilities=pp))
    result = dict(model=name, task=task, seed=seed, params=sum(p.numel() for p in network.parameters()),
                  train_seconds=train_seconds, train_loss=train_curve, test=records)
    filename = f'{name}_{task}_{seed}.json'
    (output / filename).write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    torch.save(network.state_dict(), output / filename.replace('.json', '.pt'))
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    here = Path(__file__).resolve().parent
    for file in [Path(__file__), *(here / n for n in ('input_memory_rtrl.py', 'streaming_rtrl.py', 'linear_readout.py')),
                 here.parent / 'model.py', here.parent / 'requirements.txt']:
        shutil.copyfile(file, source / file.name)
    protocol = dict(tasks=TASKS, gaps=GAPS, models=MODELS, seeds=SEEDS, train_per_gap=TRAIN_PER_GAP,
                    test_per_gap=TEST_PER_GAP, torch=str(torch.__version__), workers=args.workers,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    jobs = list(itertools.product(MODELS, TASKS, SEEDS))
    # Queue the expensive online jobs first, so baselines fill workers as they free up.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *job, output) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print(f'COMPLETE {len(jobs)} {output}', flush=True)


if __name__ == '__main__':
    main()
