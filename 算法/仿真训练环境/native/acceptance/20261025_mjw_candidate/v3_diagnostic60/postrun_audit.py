from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import torch

base = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/acceptance/20261025_v3_diagnostic60')
run = base / 'run'
diag = run / 'diagnostics'
manifest = json.loads((diag / 'diagnostic_manifest.json').read_text(encoding='utf-8'))
native = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native')
repo = native.parents[2]
proto = native / 'proto'
files = sorted(list((native / 'train').glob('*.py')) + list((native / 'train').glob('PROTOCOL_NATIVE_V1.1.*')) + list(proto.glob('*.py')) + list(proto.glob('*.json')))
current_rows = []
for path in files:
    raw = path.read_bytes()
    current_rows.append({'path': path.relative_to(repo).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
current_manifest_sha = hashlib.sha256(json.dumps(current_rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
before = {row['path']: row for row in manifest['source_files']}
after = {row['path']: row for row in current_rows}
changed = sorted(path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path))
wrapper_sha = hashlib.sha256((base / 'diagnostic_wrapper.py').read_bytes()).hexdigest()
contract = json.loads((run / 'observation_contract.json').read_text(encoding='utf-8'))
def git(*args):
    try:
        p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=10, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def inspect_checkpoint(path: Path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location='cpu', weights_only=True)
    finite = True
    floating_tensors = 0
    tensors = 0
    def walk(value):
        nonlocal finite, floating_tensors, tensors
        if isinstance(value, torch.Tensor):
            tensors += 1
            if value.is_floating_point() or value.is_complex():
                floating_tensors += 1
                if not bool(torch.isfinite(value).all().item()):
                    finite = False
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
    walk(payload)
    identity = payload.get('identity') if isinstance(payload, dict) else None
    if not isinstance(identity, dict):
        identity = {}
    policy = identity.get('policy') if isinstance(identity.get('policy'), dict) else {}
    return {
        'path': str(path), 'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'finite': finite, 'tensors_checked': tensors, 'floating_or_complex_tensors_checked': floating_tensors,
        'iteration': payload.get('iteration') if isinstance(payload, dict) else None,
        'policy_type': policy.get('policy_type'), 'hidden_size': policy.get('hidden_size'),
        'task_identity_sha256': payload.get('task_identity_sha256') if isinstance(payload, dict) else None,
        'optimizer_state_entries': len((payload.get('optimizer') or {}).get('state', {})) if isinstance(payload, dict) and isinstance(payload.get('optimizer'), dict) else None,
        'top_level_keys': sorted(payload) if isinstance(payload, dict) else [],
    }

checkpoints = [inspect_checkpoint(run / name) for name in ('latest.pt', 'checkpoint_60.pt')]
result = {
    'schema': 'native-v3-diagnostic-postrun-audit-v1',
    'manifest_git_commit': manifest.get('git_commit'), 'git_commit_after': git('rev-parse', 'HEAD'),
    'manifest_source_sha256': manifest.get('source_manifest_sha256'),
    'source_sha256_after': current_manifest_sha,
    'source_manifest_stable': not changed and current_manifest_sha == manifest.get('source_manifest_sha256'),
    'changed_source_paths': changed,
    'diagnostic_wrapper_sha256': wrapper_sha,
    'diagnostic_wrapper_sha256_recorded': manifest.get('diagnostic_wrapper_sha256'),
    'diagnostic_wrapper_stable': wrapper_sha == manifest.get('diagnostic_wrapper_sha256'),
    'task_identity_sha256': contract.get('task_identity_sha256'),
    'checkpoints': checkpoints,
}
out = diag / 'postrun_audit.json'
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(result, ensure_ascii=False))