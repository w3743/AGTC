"""Paired exploratory comparison of external-only versus all-source memory."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys
import unittest

import torch
from decimal_tasks import SEEDS
from input_memory_benchmark import build_model, digest, run_one, sequences, write_json
from streaming_rtrl_compute import counted, process
import test_input_memory_rtrl

KINDS = ('baseline_rtrl20', 'external_input_memory20', 'learned_input_memory20')
TASKS = ('binary_triples', 'binary_period16', 'decimal_triples', 'decimal_zigzag', 'decimal_switching')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_input_memory_rtrl))
    if not result.wasSuccessful():
        raise RuntimeError('mechanism tests failed')
    root = args.output_parent / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    for path in list(here.glob('*.py')) + [here.parent / 'model.py', here.parent / 'requirements.txt']:
        shutil.copy2(path, snapshot / path.name)
    seqs = sequences()
    write_json(root / 'input_sequences.json', {task: seqs[task] for task in TASKS})
    write_json(root / 'protocol.json', {
        'kinds': KINDS, 'tasks': TASKS, 'seeds': SEEDS, 'length': 600,
        'training': 'predict, learn once except first step, then advance; 599 updates',
        'comparison': 'paired seeds and inputs, all models freshly trained; compute and parameter counts differ',
        'python': sys.version, 'torch': str(torch.__version__), 'tests_run': result.testsRun,
        'source_sha256': {p.name: digest(p) for p in snapshot.iterdir()},
    })
    profiles = {}
    for kind in KINDS:
        model = build_model(kind, 73)
        for digit in seqs['binary_period16'][:32]:
            process(model, digit)
        profiles[kind] = {'parameters': sum(p.numel() for p in model.network.parameters()),
                          'eligibility_elements': sum(t.numel() for t in model.sensitivities) if hasattr(model, 'sensitivities') else model.eh.numel() + model.ef.numel(),
                          'steady_step_proxy': counted(lambda: process(model, 0))['total_proxy']}
    write_json(root / 'compute_profile.json', profiles)
    print('OUTPUT_ROOT=' + str(root), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, str(root), kind, task, seqs[task], seed)
                   for kind in KINDS for task in TASKS for seed in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('kind', 'task', 'seed', 'accuracy')}), flush=True)
    rows.sort(key=lambda r: (r['kind'], r['task'], r['seed']))
    write_json(root / 'runs_summary.json', rows)
    aggregate = []
    for task in TASKS:
        for kind in KINDS:
            group = [r for r in rows if r['kind'] == kind and r['task'] == task]
            if len(group) != len(SEEDS) or any(r['count'] != 600 or r['optimizer_updates'] != 599 for r in group):
                raise RuntimeError('incomplete run')
            aggregate.append({'task': task, 'kind': kind, 'count': sum(r['count'] for r in group),
                              **{key: statistics.mean(r[key] for r in group)
                                 for key in ('accuracy', 'last100_accuracy', 'mean_nll')}})
    write_json(root / 'aggregate.json', aggregate)
    write_json(root / 'completed.json', {'success': True, 'runs': len(rows), 'predictions': sum(r['count'] for r in rows)})
    write_json(root / 'checksums.json', {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()})
    print('COMPLETE=' + str(root), flush=True)


if __name__ == '__main__':
    main()
