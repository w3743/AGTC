"""Measure the query-state saturation and training loss of two sparse runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F

from sparse_input_memory import SparseInputMemory


NAME = 'Sparse-1024 BPTT'
PROBE_PAIRS = 16
LOSS_BLOCK = 80
THRESHOLD = .95


@torch.no_grad()
def saturation(model, pairs):
    digits = torch.tensor([row[key] for row in pairs for key in ('digits', 'paired_digits')])
    state = model.initial_state(len(digits))
    rho, retention = model.memory_logit.sigmoid(), model.trace_logit.sigmoid()
    eta = .5*model.eta_raw.tanh()
    for x in F.one_hot(digits, 10).to(model.win.dtype).transpose(0, 1):
        state = model.step(x, state, rho, retention, eta)
    return (state[0].abs() > THRESHOLD).float().mean().item()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    old, new = (read(root/'protocol.json') for root in (args.reference, args.candidate))
    for key in ('tasks', 'gaps', 'seeds', 'sparse_batch', 'sparse_updates', 'sparse_order', 'sparse_edges'):
        if old[key] != new[key]:
            raise ValueError(f'Incompatible protocols: {key}')
    for name in ('sparse_input_memory.py', 'long_memory_eval.py'):
        if old['source_sha256'][name] != new['source_sha256'][name]:
            raise ValueError(f'Incompatible source: {name}')
    torch.set_num_threads(1)
    rows = []
    for root, protocol in ((args.reference, old), (args.candidate, new)):
        lr = protocol.get('sparse_learning_rate', .015)
        for task in protocol['tasks']:
            for seed in protocol['seeds']:
                file = root / f'{NAME}_{task}_{seed}.json'
                run = read(file)
                assert run['optimizer_updates'] == 480 and run['batch_size'] == 1
                model = SparseInputMemory().eval()
                model.load_state_dict(torch.load(file.with_suffix('.pt'), map_location='cpu', weights_only=True))
                losses = run['train_loss']
                blocks = [mean(losses[i:i+LOSS_BLOCK]) for i in range(0, len(losses), LOSS_BLOCK)]
                for gap in protocol['gaps']:
                    pairs = [r for r in run['test'] if r['gap'] == gap][:PROBE_PAIRS]
                    assert len(pairs) == PROBE_PAIRS
                    rows.append(dict(lr=lr, task=task, seed=seed, gap=gap, loss_first80=blocks[0],
                                     loss_last80=blocks[-1], saturation=saturation(model, pairs), loss_blocks=blocks))
    result = dict(probe_pairs=PROBE_PAIRS, threshold=THRESHOLD, loss_block=LOSS_BLOCK, records=rows)
    (args.candidate/'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (args.candidate/'diagnostics-zh.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['学习率', '任务', '种子', '干扰步数', '首80题平均损失', '末80题平均损失', '查询处饱和比例'])
        labels = {'recall':'延迟复述', 'binding':'按键查值', 'arithmetic':'模 4 加减'}
        for r in rows:
            writer.writerow([r['lr'], labels[r['task']], r['seed'], r['gap'], f'{r["loss_first80"]:.4f}',
                             f'{r["loss_last80"]:.4f}', f'{r["saturation"]:.2%}'])
    print(f'Measured {len(rows)} seed/task/length groups.', flush=True)


if __name__ == '__main__':
    main()
