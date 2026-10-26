from __future__ import annotations
import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

RUN = pathlib.Path("/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/native_v1_1_pilot06_20261025_170308")
OUT = pathlib.Path("/mnt/c/pioneer/agtc-native-acceptance-backup/pilot06_observation_20261025_171104")
REPO = pathlib.Path("/mnt/c/pioneer/先进通用履带底盘工程")
PID = 750
COMMIT = "f0e3bea7"
OUT.mkdir(parents=True, exist_ok=True)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def proc_record(pid: int):
    try:
        raw = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,ppid=,etime=,pcpu=,rss=,stat=,comm="],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not raw:
            return None
        fields = raw.split(None, 6)
        if len(fields) < 7:
            return {"pid": pid, "ps_fields": fields}
        return {
            "pid": int(fields[0]), "ppid": int(fields[1]), "elapsed": fields[2],
            "cpu_percent": fields[3], "rss_kib": fields[4],
            "state": fields[5], "comm": fields[6],
        }
    except Exception as exc:
        return {"pid": pid, "error": f"{type(exc).__name__}: {exc}"}

def safe_args(pid: int):
    path = pathlib.Path(f"/proc/{pid}/cmdline")
    try:
        tokens = [v.decode("utf-8", "replace") for v in path.read_bytes().split(b"\0") if v]
        allowed = {
            "--iterations", "--seed", "--task-backend", "--num-envs",
            "--num-steps", "--device", "--physics", "--policy",
        }
        selected = {}
        for i, token in enumerate(tokens):
            if token in allowed and i + 1 < len(tokens):
                selected[token.lstrip("-").replace("-", "_")] = tokens[i + 1]
        return {"executable_basename": pathlib.Path(tokens[0]).name if tokens else None,
                "allowlisted_arguments": selected,
                "full_command_line_recorded": False}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "full_command_line_recorded": False}

def process_chain(start: int):
    chain, seen = [], set()
    current = start
    while current > 0 and current not in seen and len(chain) < 16:
        seen.add(current)
        row = proc_record(current)
        if row is None:
            chain.append({"pid": current, "present": False})
            break
        row["present"] = True
        if current == start:
            row["allowlisted_command"] = safe_args(current)
        chain.append(row)
        parent = row.get("ppid")
        if not isinstance(parent, int) or parent <= 1:
            break
        current = parent
    return chain

def copy_snapshot(name: str):
    src = RUN / name
    if not src.is_file():
        return {"name": name, "present": False}
    before = src.stat()
    data = src.read_bytes()
    after = src.stat()
    digest = sha(data)
    dst = OUT / name
    dst.write_bytes(data)
    copied = dst.read_bytes()
    return {
        "name": name, "present": True, "bytes": len(data), "sha256": digest,
        "snapshot_path": str(dst), "snapshot_sha256": sha(copied),
        "byte_identical": digest == sha(copied),
        "source_stat_stable_during_read": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
        "source_mtime_utc": datetime.datetime.fromtimestamp(after.st_mtime, datetime.timezone.utc).isoformat(timespec="milliseconds"),
    }

def metric_rows(raw: bytes):
    rows, incomplete = [], False
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        complete = line.endswith(b"\n")
        try:
            row = json.loads(line)
            if complete:
                rows.append(row)
            else:
                incomplete = True
        except Exception:
            if index == len(lines) - 1:
                incomplete = True
    return rows, incomplete

def checkpoint_meta(path: pathlib.Path):
    before = path.stat()
    before_sha = sha(path.read_bytes())
    try:
        import torch
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("optimizer", {}).get("state", {})
        steps = []
        for value in state.values():
            step = value.get("step") if isinstance(value, dict) else None
            if hasattr(step, "item"):
                steps.append(float(step.item()))
            elif step is not None:
                steps.append(float(step))
        after = path.stat()
        after_sha = sha(path.read_bytes())
        return {
            "file": path.name, "bytes": after.st_size, "sha256": after_sha,
            "mtime_utc": datetime.datetime.fromtimestamp(after.st_mtime, datetime.timezone.utc).isoformat(timespec="milliseconds"),
            "iteration": payload.get("iteration"), "policy_type": payload.get("policy_type"),
            "hidden_size": payload.get("hidden_size"),
            "task_backend": payload.get("task_backend"),
            "identity_schema_version": payload.get("identity_schema_version"),
            "task_identity_sha256": payload.get("task_identity_sha256"),
            "observation_contract_sha256": payload.get("observation_contract_sha256"),
            "optimizer_parameter_states": len(state),
            "optimizer_step_min": min(steps) if steps else None,
            "optimizer_step_max": max(steps) if steps else None,
            "optimizer_unique_steps": sorted(set(steps)),
            "cpu_load_map_location": "cpu",
            "source_file_stable_during_read": before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns and before_sha == after_sha,
            "checkpoint_copied": False,
        }
    except Exception as exc:
        after = path.stat() if path.exists() else None
        return {
            "file": path.name, "bytes": after.st_size if after else None,
            "sha256": sha(path.read_bytes()) if path.exists() else None,
            "load_error": f"{type(exc).__name__}: {exc}", "checkpoint_copied": False,
        }

def source_blob_hash(relative: str):
    tree = subprocess.run(
        ["git", "ls-tree", COMMIT, "--", relative],
        cwd=REPO, capture_output=True, check=False,
    )
    if tree.returncode != 0:
        return {"path": relative, "git_ls_tree_exit": tree.returncode}
    lines = tree.stdout.decode("utf-8", "replace").splitlines()
    if not lines:
        return {"path": relative, "blob_found": False}
    meta, listed_path = lines[0].split("\t", 1)
    mode, kind, oid = meta.split()
    blob = subprocess.run(["git", "cat-file", "blob", oid], cwd=REPO, capture_output=True, check=False)
    current_path = REPO / relative
    current = current_path.read_bytes() if current_path.is_file() else None
    committed = blob.stdout
    return {
        "path": relative, "commit": COMMIT, "git_blob_oid": oid, "git_blob_mode": mode,
        "git_object_type": kind, "committed_blob_bytes": len(committed),
        "committed_blob_sha256": sha(committed),
        "worktree_bytes": len(current) if current is not None else None,
        "worktree_sha256": sha(current) if current is not None else None,
        "worktree_matches_committed_blob_byte_for_byte": current == committed if current is not None else False,
        "git_commands_used": ["git ls-tree", "git cat-file blob"],
        "index_or_status_read": False,
    }

now = datetime.datetime.now(datetime.timezone.utc)
beijing = now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
files = [copy_snapshot(n) for n in (
    "metrics.jsonl", "identity.json", "observation_contract.json",
    "trial_manifest.json", "summary.json", "failure.json",
)]
metrics_info = next((x for x in files if x["name"] == "metrics.jsonl" and x.get("present")), None)
if metrics_info:
    raw_metrics = (OUT / "metrics.jsonl").read_bytes()
    rows, incomplete_tail = metric_rows(raw_metrics)
else:
    rows, incomplete_tail = [], False

run_entries = []
if RUN.is_dir():
    for p in sorted(RUN.iterdir(), key=lambda x: x.name):
        if p.is_file() and (p.name == "latest.pt" or re.fullmatch(r"checkpoint_\d+\.pt", p.name) or p.name.endswith(".tmp")):
            st = p.stat()
            run_entries.append({
                "name": p.name, "bytes": st.st_size,
                "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(timespec="milliseconds"),
                "kind": "checkpoint_temp" if p.name.endswith(".tmp") else "checkpoint",
                "sha256_not_computed_for_temp": True if p.name.endswith(".tmp") else None,
            })
candidates = []
if (RUN / "latest.pt").is_file():
    candidates.append(RUN / "latest.pt")
checkpoint_paths = []
for entry in run_entries:
    match = re.fullmatch(r"checkpoint_(\d+)\.pt", entry["name"])
    if match:
        checkpoint_paths.append((int(match.group(1)), RUN / entry["name"]))
if checkpoint_paths:
    candidates.append(max(checkpoint_paths, key=lambda x: x[0])[1])
# Load at most the atomic latest file and highest-numbered immutable checkpoint, deduplicate.
unique = []
seen = set()
for p in candidates:
    if str(p) not in seen:
        unique.append(p)
        seen.add(str(p))
checkpoints = [checkpoint_meta(p) for p in unique]

recent = rows[-5:]
wall = [float(r["wall_s"]) for r in recent if isinstance(r.get("wall_s"), (int, float))]
median_wall = sorted(wall)[len(wall)//2] if wall and len(wall)%2 else ((sorted(wall)[len(wall)//2-1]+sorted(wall)[len(wall)//2])/2 if wall else None)
mean_wall = sum(wall)/len(wall) if wall else None
checkpoint_iters = [c.get("iteration") for c in checkpoints if isinstance(c.get("iteration"), int)]
current_iteration = max([int(r.get("iter", 0)) for r in rows] + checkpoint_iters, default=0)
remaining = max(0, 300-current_iteration)
estimate = remaining * median_wall if median_wall is not None else None

# Exactly one GPU telemetry invocation in this observer.
gpu_cmd = ["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
gpu = subprocess.run(gpu_cmd, capture_output=True, text=True, check=False)

pid_chain = process_chain(PID)
source_files = [
    "算法/仿真训练环境/native/train/train_wsl.py",
    "算法/仿真训练环境/native/proto/native_env.py",
    "算法/仿真训练环境/native/train/contract_identity.py",
    "算法/仿真训练环境/native/proto/newton_backend.py",
    "算法/仿真训练环境/native/proto/observation_provider.py",
    "算法/仿真训练环境/native/proto/reward_fast_stop.py",
    "算法/仿真训练环境/native/train/policy_lstm.py",
    "算法/仿真训练环境/native/train/train_m0.py",
    "算法/仿真训练环境/isaac_field/tensor_task.py",
]
source_hashes = [source_blob_hash(p) for p in source_files]

report = {
    "schema": "pilot06_midrun_observation_v1",
    "observation_type": "mid-run observation reconstructed from current read-only process/runtime files",
    "created_at_utc": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    "created_at_beijing": beijing.isoformat(timespec="milliseconds"),
    "external_output_directory": str(OUT),
    "source_run_directory": str(RUN),
    "capture_scope": {
        "run_start_beijing": "2026-10-25T17:03:08+08:00",
        "run_start_source": "parent-provided context",
        "full_source_inventory_at_launch_was_not_collected_by_this_observation": True,
        "this_is_not_a_launch_capture": True,
        "no_process_was_stopped_or_signaled": True,
        "no_checkpoint_file_was_copied": True,
    },
    "run_status": {
        "reported_target_iterations": 300,
        "metrics_complete_row_count": len(rows),
        "metrics_last_complete_row": rows[-1] if rows else None,
        "metrics_incomplete_tail_present": incomplete_tail,
        "current_iteration_estimate": current_iteration,
        "remaining_iterations_estimate": remaining,
        "recent_wall_s_rows": wall,
        "recent_wall_s_mean": mean_wall,
        "recent_wall_s_median": median_wall,
        "remaining_seconds_estimate_from_this_run_recent_median": estimate,
        "remaining_time_estimate_note": "Estimate = remaining iterations × median wall_s of this run's last up to five complete metrics rows. It excludes future checkpoint overhead and iteration-time variation.",
        "latest_checkpoint_metadata": checkpoints,
        "checkpoint_directory_entries": run_entries,
        "identity_task_backend": None,
        "contract_task_backend": None,
    },
    "run_sidecar_snapshots": files,
    "process_observation": {
        "sampled_process_pid": PID,
        "pid_present_at_sample": bool(pid_chain and pid_chain[0].get("present")),
        "sanitized_process_and_parent_chain": pid_chain,
        "only_allowlisted_arguments_recorded": True,
        "full_command_line_recorded_or_printed": False,
        "human_author_or_codex_task_attribution": "not established from this process chain; no person or task source inferred",
    },
    "gpu_telemetry": {
        "query_invocation_count": 1,
        "command_fields": gpu_cmd[1],
        "exit_code": gpu.returncode,
        "stdout": gpu.stdout.strip(),
        "stderr": gpu.stderr.strip(),
        "sample_timestamp_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "not_a_benchmark": True,
    },
    "source_blob_comparison_to_f0e3bea7": source_hashes,
    "git_access": {
        "commit_reference": COMMIT,
        "used": ["git ls-tree", "git cat-file blob"],
        "did_not_use": ["git status", "index inspection", "LFS smudge/pull"],
    },
    "cuda_or_benchmark_work": {
        "cuda_telemetry_queries": 1,
        "training_or_eval_started_by_observer": False,
        "benchmark_started_by_observer": False,
    },
}
identity_path = OUT / "identity.json"
contract_path = OUT / "observation_contract.json"
if identity_path.is_file():
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        report["run_status"]["identity_task_backend"] = identity.get("task_backend") or identity.get("env", {}).get("task_backend")
        report["run_status"]["identity_task_identity_sha256"] = identity.get("task_identity_sha256")
        report["run_status"]["identity_num_envs"] = identity.get("num_envs")
        report["run_status"]["identity_seed"] = identity.get("seed")
    except Exception as exc:
        report["run_status"]["identity_parse_error"] = f"{type(exc).__name__}: {exc}"
if contract_path.is_file():
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        report["run_status"]["contract_task_backend"] = contract.get("task_backend")
        report["run_status"]["contract_task_identity_sha256"] = contract.get("task_identity_sha256") or contract.get("observation_contract_sha256")
        report["run_status"]["capability_metrics_allowed"] = contract.get("capability_metrics_allowed")
    except Exception as exc:
        report["run_status"]["contract_parse_error"] = f"{type(exc).__name__}: {exc}"

(OUT / "host_observation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "created_at_utc": report["created_at_utc"],
    "run_dir_exists": RUN.is_dir(),
    "progress_iteration": current_iteration,
    "metrics_rows": len(rows),
    "remaining_estimate_seconds": estimate,
    "process_chain": pid_chain,
    "checkpoint_metadata": checkpoints,
    "gpu_query_exit": gpu.returncode,
    "gpu_query_output": gpu.stdout.strip(),
    "source_comparisons": source_hashes,
    "output_directory": str(OUT),
}, ensure_ascii=False, indent=2))