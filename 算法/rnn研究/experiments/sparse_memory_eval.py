"""Train and benchmark the 1,024-neuron sparse model on the existing protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch
from torch.nn import functional as F

from long_memory_eval import Baseline, InputMemoryRTRL, datasets, frozen_rtrl
from sparse_input_memory import SparseInputMemory


NAME = 'Sparse-1024 BPTT'
BATCH = 8


def train_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def settings(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def run_one(task, seed, output, batch_size, learning_rate):
    settings(seed)
    train, test = datasets(task, seed)
    model = SparseInputMemory()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    buckets = defaultdict(list)
    for x, y in train:
        buckets[len(x)].append((x, y))
    batches = [bucket[i:i+BATCH] for bucket in buckets.values() for i in range(0, len(bucket), BATCH)]
    random.Random(seed).shuffle(batches)
    if batch_size == 1:
        # Preserve the previous eight-item batch order and its within-batch order.
        batches = [[example] for batch in batches for example in batch]
    start = perf_counter()
    losses = []
    for batch in batches:
        x, y = zip(*batch)
        loss = train_step(model, optimizer, torch.tensor(x), torch.tensor(y))
        losses.extend([loss]*len(batch))
    seconds = perf_counter()-start
    model.eval()
    records = []
    with torch.no_grad():
        for gap, rows in test.items():
            digits = torch.tensor([s for x, xp, _, _ in rows for s in (x, xp)])
            probs = torch.cat([model(batch).softmax(dim=-1) for batch in digits.split(BATCH)]).tolist()
            for i, (x, xp, y, yp) in enumerate(rows):
                p, pp = probs[2*i:2*i+2]
                records.append(dict(gap=gap, index=i, digits=x, paired_digits=xp, target=y,
                                    paired_target=yp, probabilities=p, paired_probabilities=pp))
    prefix = output/f'{NAME}_{task}_{seed}'
    torch.save(model.state_dict(), prefix.with_suffix('.pt'))
    result = dict(model=NAME, task=task, seed=seed, params=sum(p.numel() for p in model.parameters()),
                  edges=model.edge_count, optimizer_updates=len(batches), batch_size=batch_size,
                  learning_rate=learning_rate,
                  train_seconds=seconds, train_loss=losses, test=records)
    prefix.with_suffix('.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    return task, seed, round(seconds, 2)


def peak_mib():
    if sys.platform == 'win32':
        size_t = ctypes.c_size_t

        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (name, size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                                            'PagefileUsage', 'PeakPagefileUsage')]

        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return counters.PeakWorkingSetSize / 2**20
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if sys.platform == 'darwin' else 1024)


def profile(name, mode, output):
    settings(2)
    if mode not in ('inference', 'training', 'training-batch8') or (mode == 'training-batch8' and name != NAME):
        raise ValueError('Unsupported profiling mode for this model')
    is_rtrl = name == 'Full-memory RTRL'
    model = InputMemoryRTRL(seed=2) if is_rtrl else SparseInputMemory() if name == NAME else Baseline(name)
    network = model.network if is_rtrl else model
    run = json.loads((output/f'{name}_recall_2.json').read_text(encoding='utf-8'))
    network.load_state_dict(torch.load(output/f'{name}_recall_2.pt', map_location='cpu', weights_only=True))
    # Identical max training length for the update probe, and unseen test length for inference.
    gap = 256 if mode == 'inference' else 128
    count = BATCH if mode == 'training-batch8' else 1
    selected = [r for r in run['test'] if r['gap'] == gap][:count]
    x = torch.tensor([row['digits'] for row in selected])
    y = torch.tensor([row['target'] for row in selected])
    if mode == 'inference':
        network.eval()

        @torch.no_grad()
        def operation():
            return frozen_rtrl(model, x) if is_rtrl else network(x)
    elif is_rtrl:
        digits = x[0].tolist()

        def operation():
            model.reset_state()
            for digit in digits:
                model.advance(digit)
            model.learn_target(int(y.item()))
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=run.get('learning_rate', .015), weight_decay=1e-4)

        def operation():
            train_step(network, optimizer, x, y)

    operation()
    samples = []
    for _ in range(9 if mode == 'inference' else 5):
        start = perf_counter()
        operation()
        samples.append(1000*(perf_counter()-start)/x.numel())
    print(json.dumps(dict(model=name, mode=mode, median_ms_per_token=median(samples),
                          samples_ms_per_token=samples, sequence_length=x.shape[1],
                          batch_size=count, peak_mib=peak_mib())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, choices=(1, BATCH), default=1)
    parser.add_argument('--learning-rate', type=float, default=.015)
    parser.add_argument('--skip-profile', action='store_true', help='Evaluate scores without rerunning resource probes')
    parser.add_argument('--profile', nargs=2, metavar=('MODEL', 'MODE'))
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive')
    output = args.output.resolve()
    if args.profile:
        profile(*args.profile, output)
        return
    if not args.profile_only:
        if not args.reference or args.workers < 1:
            parser.error('Training requires --reference and positive --workers')
        protocol = json.loads((args.reference/'protocol.json').read_text(encoding='utf-8'))
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(args.reference/'source', output/'source')
        for path in [Path(__file__), Path(__file__).with_name('sparse_input_memory.py')]:
            shutil.copyfile(path, output/'source'/path.name)
        for model, task, seed in itertools.product(protocol['models'], protocol['tasks'], protocol['seeds']):
            for suffix in ('json', 'pt'):
                path = args.reference/f'{model}_{task}_{seed}.{suffix}'
                shutil.copyfile(path, output/path.name)
        protocol['models'].append(NAME)
        protocol.update(sparse_edges=32768, sparse_width=1024, sparse_batch=args.batch_size,
                        sparse_updates=protocol['train_per_gap']*(len(protocol['gaps'])-1)//args.batch_size,
                        sparse_learning_rate=args.learning_rate,
                        sparse_order='previous batch-8 order, flattened for batch size 1',
                        reference=str(args.reference.resolve()),
                        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'source').iterdir()})
        (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, task, seed, output, args.batch_size, args.learning_rate)
                       for task, seed in itertools.product(protocol['tasks'], protocol['seeds'])]
            for future in as_completed(futures):
                print('TRAINED', *future.result(), flush=True)
    if args.skip_profile:
        return
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    profiles = []
    jobs = list(itertools.product(protocol['models'], ('inference', 'training')))
    for name, mode in [*jobs, (NAME, 'training-batch8')]:
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output), '--profile', name, mode]
        result = json.loads(subprocess.check_output(command, text=True))
        profiles.append(result)
        print('PROFILE', name, mode, round(result['median_ms_per_token'], 4), round(result['peak_mib'], 1), flush=True)
    (output/'resources.json').write_text(json.dumps(profiles, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
