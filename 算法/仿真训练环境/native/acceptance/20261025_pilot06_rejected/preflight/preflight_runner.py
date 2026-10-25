import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path('/mnt/c/pioneer/agtc-native-acceptance-backup/pilot06_rejected_audit_20261025_175637/preflight')
run = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/native_v1_1_pilot06_20261025_170308')
script = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/trial_checks.py')
protocol = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/PROTOCOL_NATIVE_V1.1.sha256')
out = root / 'rejected_output'
argv = [sys.executable, '-B', str(script), 'preflight-checkpoint', '--checkpoint', str(run / 'latest.pt'), '--out', str(out), '--protocol-sha', str(protocol)]
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['CUDA_VISIBLE_DEVICES'] = ''
env['OMP_NUM_THREADS'] = '1'
env['MKL_NUM_THREADS'] = '1'
start = time.time()
result = subprocess.run(argv, cwd='/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train', env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
(root / 'preflight.stdout.txt').write_bytes(result.stdout)
(root / 'preflight.stderr.txt').write_bytes(result.stderr)
(root / 'preflight.exit.txt').write_text(f'{result.returncode}\n', encoding='ascii')
record = {'argv': argv, 'exit_code': result.returncode, 'elapsed_s': round(time.time() - start, 6), 'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES'], 'source_run_unchanged_by_command': True}
(root / 'preflight_result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, sort_keys=True))