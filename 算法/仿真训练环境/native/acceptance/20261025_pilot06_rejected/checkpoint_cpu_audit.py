import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def check_tree(value):
    tensors = 0
    nonfinite_tensors = 0
    nonfinite_values = 0
    def visit(item):
        nonlocal tensors, nonfinite_tensors, nonfinite_values
        if isinstance(item, torch.Tensor):
            tensors += 1
            if item.is_floating_point() or item.is_complex():
                bad = ~torch.isfinite(item)
                count = int(torch.count_nonzero(bad).item())
                if count:
                    nonfinite_tensors += 1
                    nonfinite_values += count
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif isinstance(item, float) and not math.isfinite(item):
            nonfinite_values += 1
    visit(value)
    return {"tensor_count": tensors, "nonfinite_tensor_count": nonfinite_tensors, "nonfinite_value_count": nonfinite_values, "all_finite": nonfinite_values == 0}

def optimizer_steps(optimizer):
    state = optimizer.get("state", {}) if isinstance(optimizer, dict) else {}
    steps = []
    for record in state.values():
        if not isinstance(record, dict) or "step" not in record:
            continue
        value = record["step"]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        steps.append(float(value))
    unique = sorted(set(steps))
    return {
        "parameter_state_count": len(state),
        "step_state_count": len(steps),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "unique_steps": unique,
        "all_steps_equal": bool(steps) and len(unique) == 1,
    }

paths = [Path(x) for x in sys.argv[1:]]
result = {
    "audit": "cpu_safe_checkpoint_audit",
    "torch_version": str(torch.__version__),
    "map_location": "cpu",
    "weights_only": True,
    "torch_num_threads": torch.get_num_threads(),
    "torch_num_interop_threads": torch.get_num_interop_threads(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cuda_initialized_before_load": torch.cuda.is_initialized(),
    "checkpoints": [],
}
if result["cuda_initialized_before_load"]:
    raise RuntimeError("CUDA was initialized before CPU checkpoint load")
for path in paths:
    before = {"sha256": sha256(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    after = {"sha256": sha256(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload is {type(payload).__name__}, expected dict")
    policy = check_tree(payload.get("policy"))
    optimizer = check_tree(payload.get("optimizer"))
    item = {
        "file": str(path),
        "payload_iteration": payload.get("iteration"),
        "policy_type": payload.get("policy_type"),
        "hidden_size": payload.get("hidden_size"),
        "task_backend": payload.get("task_backend"),
        "task_identity_sha256": payload.get("task_identity_sha256"),
        "torch_checkpoint_version": payload.get("torch"),
        "before": before,
        "after": after,
        "stable_during_load": before == after,
        "policy_finiteness": policy,
        "optimizer_finiteness": optimizer,
        "optimizer_steps": optimizer_steps(payload.get("optimizer")),
        "payload_keys": sorted(payload.keys()),
    }
    result["checkpoints"].append(item)
result["cuda_initialized_after_load"] = torch.cuda.is_initialized()
if result["cuda_initialized_after_load"]:
    raise RuntimeError("CUDA was initialized during CPU checkpoint load")
print(json.dumps(result, sort_keys=True, separators=(",", ":")))