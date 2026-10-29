"""Supervised parameter switch; independent of the service being restarted."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from uuid import uuid4

from calibration import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def prepare_transaction(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    store = root / 'calibration'
    store.mkdir(exist_ok=True)
    lock = store / 'apply.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('已有标定应用任务，请等待完成')
    os.close(fd)
    try:
        identifier = uuid4().hex
        history = store / 'history'
        transactions = store / 'transactions'
        history.mkdir(exist_ok=True)
        transactions.mkdir(exist_ok=True)
        current = root / 'app' / 'calibration.npz'
        backup = history / (identifier + '.npz')
        staged = transactions / (identifier + '.npz')
        shutil.copy2(str(current), str(backup))
        shutil.copy2(str(candidate), str(staged))
        atomic_json(history / (identifier + '.json'), dict(id=identifier, calibration_id=digest(backup), created_at=time.time()))
        tx = transactions / (identifier + '.json')
        atomic_json(tx, dict(id=identifier, root=str(root), candidate=str(staged), backup=str(backup),
                             expected=digest(staged), previous=digest(backup), state='pending', created_at=time.time()))
        atomic_json(store / 'latest-transaction.json', {'id': identifier})
        return tx
    except Exception:
        lock.unlink()
        raise


def run_transaction(transaction, restart, health, timeout=30):
    tx = Path(transaction)
    data = json.loads(tx.read_text(encoding='utf-8'))
    root = Path(data['root'])
    current = root / 'app' / 'calibration.npz'
    temp = current.with_name('calibration.pending.npz')
    def record(state, error=None):
        data.update(state=state, updated_at=time.time(), error=error)
        atomic_json(tx, data)
    try:
        record('applying')
        shutil.copy2(data['candidate'], str(temp))
        os.replace(str(temp), str(current))
        if not restart():
            raise RuntimeError('服务重启失败')
        deadline = time.monotonic() + timeout
        while True:
            if health(data['expected']):
                record('applied')
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('新标定服务健康检查失败')
            time.sleep(.5)
    except Exception as exc:
        try:
            shutil.copy2(data['backup'], str(temp))
            os.replace(str(temp), str(current))
            restarted = restart()
            deadline = time.monotonic() + timeout
            healthy = False
            while restarted:
                if health(data['previous']):
                    healthy = True
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(.5)
            record('rolled_back' if healthy else 'rollback_restart_failed', str(exc))
        except Exception as rollback_exc:
            record('rollback_restart_failed', str(exc) + '; rollback: ' + str(rollback_exc))
    finally:
        lock = root / 'calibration' / 'apply.lock'
        if lock.exists() and data['state'] in ('applied', 'rolled_back'):
            lock.unlink()


def launch_transaction(transaction):
    tx = Path(transaction).resolve()
    data = json.loads(tx.read_text(encoding='utf-8'))
    root = Path(data['root'])
    args = ['systemd-run', '--quiet', '--unit=rk-calibration-' + data['id'],
            '--property=Environment=PYTHONPATH=' + str(root / 'runtime' / 'vendor'),
            '/usr/bin/python3', '-B', str(root / 'app' / 'calibration_apply.py'), '--transaction', str(tx)]
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=10)
    except Exception as exc:
        data.update(state='launch_failed', error=str(exc))
        atomic_json(tx, data)
        (root / 'calibration' / 'apply.lock').unlink()
        raise RuntimeError('无法启动标定应用监督任务，当前参数未更改') from exc
    return data['id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transaction', required=True)
    args = parser.parse_args()
    # Allow the HTTP response to reach the page before stopping its service.
    time.sleep(1)
    def restart():
        return subprocess.run(['systemctl', 'restart', 'rk-lightstereo.service'], timeout=35).returncode == 0
    def health(expected):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8088/status', timeout=2) as response:
                state = json.load(response)
            ages = state.get('camera_age_ms') or []
            return (state.get('calibration_id') == expected and len(ages) == 2
                    and all(v is not None and v < 1500 for v in ages))
        except Exception:
            return False
    run_transaction(args.transaction, restart, health)


if __name__ == '__main__':
    main()
