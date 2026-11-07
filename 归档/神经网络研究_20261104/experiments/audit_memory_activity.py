"""Verify a completed memory/activity experiment without executing its model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    protocol, sequences = read('protocol.json'), read('input_sequences.json')
    rows, aggregate, files = read('runs_summary.json'), read('aggregate.json'), read('checksums.json')
    expected_keys = {(kind, task, seed) for kind in protocol['kinds']
                     for task in protocol['tasks'] for seed in protocol['seeds']}
    if len(rows) != len(expected_keys) or {(r['kind'], r['task'], r['seed']) for r in rows} != expected_keys:
        raise ValueError('missing, duplicate, or unexpected runs')
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'file checksum mismatch: {name}')
    count = 0
    for row in rows:
        events = [json.loads(line) for line in (root / row['run_directory'] / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(events) != protocol['length'] or [e['target'] for e in events] != sequences[row['task']]:
            raise ValueError(f"wrong sequence: {row['run_directory']}")
        for index, e in enumerate(events):
            probs = e['probabilities']
            if len(probs) != 10 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or abs(sum(probs)-1) >= 1e-6:
                raise ValueError('invalid probabilities')
            prediction = max(range(10), key=lambda j: (probs[j], -j))
            if (e['step'] != index + 1 or e['prediction'] != prediction or
                e['hit'] != (prediction == e['target']) or
                abs(e['nll'] + math.log(max(probs[e['target']], 1e-300))) > 1e-9 or
                e['updates_after'] != index):
                raise ValueError(f"invalid event: {row['run_directory']} step {index+1}")
        if (row['hits'] != sum(e['hit'] for e in events) or row['optimizer_updates'] != len(events)-1 or
            abs(row['mean_nll'] - sum(e['nll'] for e in events)/len(events)) > 1e-10 or
            abs(row['accuracy'] - sum(e['hit'] for e in events)/len(events)) > 1e-10 or
            abs(row['last100_accuracy'] - sum(e['hit'] for e in events[-100:])/100) > 1e-10):
            raise ValueError(f"summary mismatch: {row['run_directory']}")
        count += len(events)
    if len(aggregate) != len(protocol['tasks']) * len(protocol['kinds']):
        raise ValueError('incorrect aggregate count')
    for item in aggregate:
        group = [r for r in rows if (r['kind'], r['task']) == (item['kind'], item['task'])]
        if len(group) != len(protocol['seeds']) or any(abs(item[key] - sum(r[key] for r in group)/len(group)) > 1e-12
                for key in ('accuracy', 'last100_accuracy', 'mean_nll')):
            raise ValueError('aggregate mismatch')
    print(json.dumps({'passed': True, 'files': len(files), 'runs': len(rows), 'events': count,
                      'aggregates': len(aggregate), 'scope': 'checksums, event scoring, sequences and summaries; no checkpoint replay'}))


if __name__ == '__main__':
    main()
