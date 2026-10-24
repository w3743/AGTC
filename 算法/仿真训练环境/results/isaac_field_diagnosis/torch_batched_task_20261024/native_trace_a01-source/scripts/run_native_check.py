"""Run a bounded native check, preserving its log and actual process exit code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not re.fullmatch(r'[a-z0-9_-]+', args.tag) or not command or args.timeout < 1:
        parser.error('provide a simple tag, positive timeout, and command')
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f'{args.tag}.log'
    receipt_path = output_dir / f'{args.tag}-process.json'
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError('native evidence is immutable: use a new tag')
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('isaac_field', 'scripts') for path in sorted((root / folder).rglob('*.py'))
    }
    snapshot = output_dir / f'{args.tag}-source'
    snapshot.mkdir(exist_ok=False)
    for relative, digest in source_hashes.items():
        source, target = root / relative, snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'native source copy hash mismatch: {relative}')
    environment = os.environ.copy()
    environment.update(OMNI_KIT_ACCEPT_EULA='YES', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    environment['PYTHONPATH'] = str(root) + os.pathsep + environment.get('PYTHONPATH', '')
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    timed_out = False
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
            else:
                process.kill()
            returncode = process.wait(timeout=30)
    receipt = dict(command=command, started_utc=started,
        elapsed_seconds=round(time.monotonic() - clock, 3), returncode=returncode,
        timed_out=timed_out, stdout_stderr_log=str(log_path), source_sha256=source_hashes,
        source_snapshot=str(snapshot),
        log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_sha256'}), flush=True)
    return 124 if timed_out else (0 if returncode == 0 else 1)


if __name__ == '__main__':
    sys.exit(main())
