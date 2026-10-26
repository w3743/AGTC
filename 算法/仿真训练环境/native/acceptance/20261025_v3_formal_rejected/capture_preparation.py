import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

base = Path('/mnt/c/pioneer/agtc-native-acceptance-backup/fixed_physics_v3_prep_20261025_184052')
native_tools = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools')
sys.path.insert(0, str(native_tools))
import benchmark_training as bt

identity = bt.source_identity()
git_root = next((p for p in (bt.PROJECT, *bt.PROJECT.parents) if (p / '.git').exists()), None)
if git_root is None:
    raise RuntimeError('Git root not found for source snapshot')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def now_pair():
    utc = datetime.now(timezone.utc).replace(microsecond=0)
    beijing = utc.astimezone(timezone(timedelta(hours=8)))
    return utc.isoformat().replace('+00:00','Z'), beijing.isoformat()

copied = []
snapshot_root = base / 'source_snapshot'
for row in identity['raw_source_bytes']:
    rel = Path(row['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise RuntimeError(f'unsafe source path in identity manifest: {row["path"]}')
    src = git_root / rel
    raw = src.read_bytes()
    dst = snapshot_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    copied.append({'path': row['path'], 'bytes': len(raw), 'source_sha256': row['sha256'],
                   'snapshot_sha256': sha(dst.read_bytes()), 'byte_identical': row['sha256'] == sha(dst.read_bytes())})

train = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train')
control_pairs = [
    (train / 'run_native_v1_trial.sh', 'native/train/run_native_v1_trial.sh'),
    (train / 'run_native_v1_trial.ps1', 'native/train/run_native_v1_trial.ps1'),
    (train / 'PROTOCOL_NATIVE_V1.1.md', 'native/train/PROTOCOL_NATIVE_V1.1.md'),
    (train / 'PROTOCOL_NATIVE_V1.1.sha256', 'native/train/PROTOCOL_NATIVE_V1.1.sha256'),
    (train / 'PROTOCOL_NATIVE_V1.md', 'native/train/PROTOCOL_NATIVE_V1.md'),
    (train / 'PROTOCOL_NATIVE_V1.sha256', 'native/train/PROTOCOL_NATIVE_V1.sha256'),
]
controls = []
control_root = base / 'control_files'
for src, relpath in control_pairs:
    raw = src.read_bytes()
    dst = control_root / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw)
    controls.append({'path': relpath, 'bytes': len(raw), 'source_sha256': sha(raw),
                     'snapshot_sha256': sha(dst.read_bytes()), 'byte_identical': raw == dst.read_bytes()})

versions = bt.versions()
for name in ('gymnasium', 'scipy', 'psutil', 'packaging', 'tensorboard'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

utc, beijing = now_pair()
tag = 'fixed_physics_v3_20261025_184052'
runner = '/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/run_native_v1_trial.sh'
trainer = '/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/train_wsl.py'
out_template = f'runs/native_v1_1_{tag}_<wrapper-generated-%Y%m%d_%H%M%S>'
wrapper_argv = ['/usr/bin/env', 'SIM_DT=0.001', 'DECIMATION=50', 'PYTHON=/opt/venv/bin/python', '/bin/bash', runner, tag]
child_argv_template = ['/opt/venv/bin/python', '-u', trainer, '--policy', 'lstm', '--lstm-hidden', '512', '--num-envs', '64', '--num-steps', '32', '--iterations', '300', '--seed', '0', '--threads', '8', '--device', 'cuda', '--physics', 'newton_xpbd', '--task-backend', 'fast_stop_v2', '--lidar', 'warp', '--feedback', 'vqf', '--depth', 'analytic', '--detections', 'analytic', '--detections-warp', '--sim-dt', '0.001', '--decimation', '50', '--save-interval', '20', '--nonfinite-limit', '0', '--out', out_template]
launcher_argv_template = ['wsl.exe', '-d', 'Ubuntu-24.04-Arm64', '--', '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', '/opt/venv/bin/python', '-B', str(base / 'launch.py'), '--execute', '--root-release-notified-at', '<required ISO timestamp>']
all_copied = all(x['byte_identical'] for x in copied + controls)
manifest = {
  'schema': 'fixed_physics_v3_prelaunch_v1', 'captured_at_utc': utc, 'captured_at_beijing': beijing,
  'launch_state': 'prepared_not_started_waiting_for_root_gpu_release',
  'launch_ready_source_gate': identity['git']['relevant_sources_dirty'] is False and all_copied,
  'launch_start_pid_windows': None, 'launch_start_pid_wsl_controller': None, 'launch_start_pid_wsl_wrapper': None,
  'gpu_queried_during_preparation': False, 'training_started_during_preparation': False,
  'tag': tag, 'seed': 0, 'resume': False, 'resume_checkpoint': None,
  'repo_root': str(git_root), 'train_directory': str(train), 'runtime': bt.runtime(),
  'python_executable': sys.executable, 'python_version': platform.python_version(),
  'dependency_versions': versions, 'source_identity': identity,
  'source_snapshot': {'directory': str(snapshot_root), 'files': copied, 'file_count': len(copied), 'all_byte_identical': all(x['byte_identical'] for x in copied)},
  'control_file_snapshot': {'directory': str(control_root), 'files': controls, 'all_byte_identical': all(x['byte_identical'] for x in controls)},
  'wrapper_argv_template': wrapper_argv, 'child_train_argv_template': child_argv_template,
  'launcher_argv_template': launcher_argv_template,
  'fixed_recipe': {'iterations': 300, 'seed': 0, 'threads': 8, 'num_envs': 64, 'num_steps': 32,
    'policy': 'lstm', 'lstm_hidden': 512, 'device': 'cuda', 'physics': 'newton_xpbd', 'task_backend': 'fast_stop_v2',
    'lidar': 'warp', 'feedback': 'vqf', 'depth': 'analytic', 'detections': 'analytic', 'detections_warp': True,
    'sim_dt': 0.001, 'decimation': 50, 'save_interval': 20, 'nonfinite_limit': 0},
  'rawlog_plan': {'wrapper_stdout': 'wrapper.stdout.rawlog', 'wrapper_stderr': 'wrapper.stderr.rawlog', 'gpu_samples': 'nvidia_smi_samples.jsonl', 'monitor': 'monitor.jsonl', 'whole_wrapper_wall': 'time.perf_counter from before wrapper Popen through wrapper exit', 'trainer_wall_source': 'run/summary.json total_wall_s'},
  'monitor_interval_s': 30, 'on_nonfinite': 'allow train_wsl.py nonfinite-limit=0 fail-fast; never restart or change budget',
}
(base / 'source_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
(base / 'prepare_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'captured_at_utc': utc, 'captured_at_beijing': beijing, 'git': identity['git'], 'source_manifest_sha256': identity['manifest_sha256'], 'source_count': len(copied), 'source_copies_match': all_copied, 'control_count': len(controls), 'control_copies_match': all(x['byte_identical'] for x in controls), 'versions': versions, 'runtime': bt.runtime(), 'launch_ready_source_gate': manifest['launch_ready_source_gate'], 'tag': tag}, ensure_ascii=False, sort_keys=True))