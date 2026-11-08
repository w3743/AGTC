"""Equal-budget tuning of standard sequence models on the three memory tasks."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from long_memory_eval import MODELS, SEEDS, TASKS, episode, oracle, datasets


MODELS = MODELS[1:]
GAPS = (0, 8, 32, 64, 128, 256)
REGIMES = {'extrapolate': (0, 8, 32, 64, 128), 'seen256': (0, 8, 32, 64, 256)}
TIERS = {
    'small': dict(zip(MODELS, (29, 16, 13, 4, 8, 8), strict=True)),
    'matched': dict(zip(MODELS, (304, 174, 151, 300, 74, 64), strict=True)),
}
RATES = (.001, .003, .01)
TUNING_SEED = 91273
BATCH = 16
BUDGETS = (480, 2000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def make_data(task):
    _, prior = datasets(task, SEEDS[0])
    used = {tuple(seq) for rows in prior.values() for x, xp, _, _ in rows for seq in (x, xp)}
    out = {}
    for split, split_seed in (('validation', 86763), ('test', 718761)):
        rng = random.Random(split_seed)
        buckets = {}
        for gap in GAPS:
            rows = []
            for label in range(4):
                for _ in range(16):
                    while True:
                        x, xp, y, yp = episode(task, gap, label, rng)
                        keys = {tuple(x), tuple(xp)}
                        if gap == 0 or not keys & used:
                            break
                    assert oracle(task, x) == y and oracle(task, xp) == yp
                    used.update(keys)
                    rows.append((x, xp, y, yp))
            rng.shuffle(rows)
            buckets[str(gap)] = rows
        out[split] = buckets
    return out, used


def training_batches(task, regime, seed, forbidden):
    gaps = REGIMES[regime]
    buckets = {}
    for gap in gaps:
        rng = random.Random(99277 + 8737*seed + gap)
        labels = list(range(4))*100
        rng.shuffle(labels)
        rows = []
        for label in labels:
            while True:
                x, _, y, _ = episode(task, gap, label, rng)
                if gap == 0 or tuple(x) not in forbidden:
                    break
            assert oracle(task, x) == y
            rows.append((x, y))
        buckets[gap] = [rows[i:i+BATCH] for i in range(0, len(rows), BATCH)]
    rng = random.Random(27819+seed)
    for i in range(25):
        for gap in rng.sample(gaps, len(gaps)):
            yield buckets[gap][i]


class Standard(nn.Module):
    def __init__(self, name, tier):
        super().__init__()
        self.name = name
        width = TIERS[tier][name]
        if name in ('RNN', 'GRU', 'LSTM'):
            self.core = getattr(nn, name)(10, width, batch_first=True)
        elif name == 'MLP-32':
            self.core = nn.Linear(320, width)
        elif name == 'TCN-512':
            self.embed = nn.Linear(10, width)
            self.convs = nn.ModuleList(nn.Conv1d(width, width, 2, dilation=2**i) for i in range(9))
        else:
            self.embed = nn.Linear(10, width)
            heads, layers = (2, 2) if tier == 'small' else (4, 3)
            self.core = nn.ModuleList(nn.TransformerEncoderLayer(width, heads, 2*width, dropout=0,
                                                                  batch_first=True) for _ in range(layers))
            angles = torch.arange(264)[:, None] / (10000 ** (torch.arange(0, width, 2)/width))
            self.register_buffer('positions', torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1))
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
            x = self.embed(x) + self.positions[:x.shape[1]]
            for layer in self.core:
                x = layer(x)
            h = x[:, -1]
        return self.head(h)


@torch.no_grad()
def evaluate(model, buckets, gaps, keep_predictions):
    model.eval()
    rows = []
    for gap in gaps:
        pairs = buckets[str(gap)]
        x = torch.tensor([seq for a, b, _, _ in pairs for seq in (a, b)])
        y = torch.tensor([label for _, _, a, b in pairs for label in (a, b)])
        logits = torch.cat([model(batch) for batch in x.split(BATCH)])
        hits = (logits.argmax(dim=-1) == y).reshape(-1, 2)
        loss = F.cross_entropy(logits, y, reduction='none').reshape(-1, 2)
        row = dict(gap=gap, n=len(pairs), accuracy=hits[:, 0].float().mean().item(),
                   paired_accuracy=hits.all(dim=1).float().mean().item(), nll=loss[:, 0].mean().item())
        if keep_predictions:
            row['probabilities'] = logits.softmax(dim=-1).reshape(-1, 2, 10).tolist()
        rows.append(row)
    return rows


def run_job(output, job, lr, seed, tuning):
    tier, name, task, regime = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    data = json.loads((output/f'{task}-data.json').read_text(encoding='utf-8'))
    forbidden = {tuple(seq) for seq in data['forbidden']}
    model = Standard(name, tier)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    curve, results = [], {}
    start = perf_counter()
    for i, batch in enumerate(training_batches(task, regime, seed, forbidden), 1):
        x, y = zip(*batch)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(torch.tensor(x)), torch.tensor(y))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        curve.append(loss.item())
        count = i*BATCH
        if count in BUDGETS and (not tuning or count == BUDGETS[-1]):
            split = 'validation' if tuning else 'test'
            gaps = [g for g in REGIMES[regime] if g > 0] if tuning else GAPS
            results[str(count)] = evaluate(model, data[split], gaps, not tuning)
    result = dict(tier=tier, model=name, task=task, regime=regime, lr=lr, seed=seed,
                  parameters=sum(p.numel() for p in model.parameters()), seconds=perf_counter()-start,
                  loss=curve, scores=results)
    folder = 'tuning' if tuning else 'evaluation'
    path = output/folder/f'{tier}_{name}_{task}_{regime}_{seed}_{lr:g}.json'
    write(path, result)
    return job, lr, seed, path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tiers', nargs='+', choices=TIERS, default=list(TIERS))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or len(set(args.tiers)) != len(args.tiers):
        parser.error('Use positive workers and unique tiers')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('tuning', 'evaluation', 'source'):
        (root/name).mkdir()
    for name in ('standard_memory_eval.py', 'long_memory_eval.py', 'input_memory_rtrl.py',
                 'streaming_rtrl.py', 'linear_readout.py', '../model.py'):
        path = Path(__file__).parent/name
        shutil.copyfile(path, root/'source'/path.name)
    for task in TASKS:
        data, forbidden = make_data(task)
        data['forbidden'] = sorted(forbidden)
        write(root/f'{task}-data.json', data)
    write(root/'protocol.json', dict(models=MODELS, tasks=TASKS, regimes=REGIMES, tiers=args.tiers,
                                    seeds=SEEDS, gaps=GAPS, rates=RATES, tuning_seed=TUNING_SEED,
                                    budgets=BUDGETS, batch=BATCH, torch=str(torch.__version__),
                                    data_sha256={p.name:digest(p) for p in root.glob('*-data.json')},
                                    source_sha256={p.name:digest(p) for p in (root/'source').iterdir()}))
    jobs = list(itertools.product(args.tiers, MODELS, TASKS, REGIMES))
    choices = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_job, root, job, lr, TUNING_SEED, True) for job in jobs for lr in RATES]
        for i, f in enumerate(as_completed(pending), 1):
            job, lr, _, filename = f.result()
            run = json.loads((root/'tuning'/filename).read_text(encoding='utf-8'))
            rows = run['scores'][str(BUDGETS[-1])]
            rank = (sum(r['paired_accuracy'] for r in rows), -sum(r['nll'] for r in rows), -lr)
            if job not in choices or rank > choices[job][0]:
                choices[job] = rank, lr
            if i % 12 == 0 or i == len(pending):
                print(f'TUNING {i}/{len(pending)}', flush=True)
        write(root/'selection.json', [{'job':job,'lr':choices[job][1]} for job in jobs])
        pending = [pool.submit(run_job, root, job, choices[job][1], seed, False) for job in jobs for seed in SEEDS]
        for i, f in enumerate(as_completed(pending), 1):
            f.result()
            if i % 12 == 0 or i == len(pending):
                print(f'EVALUATION {i}/{len(pending)}', flush=True)


if __name__ == '__main__':
    main()
