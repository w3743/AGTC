#!/usr/bin/env python3
"""Short native WSL throughput run, reusing train_wsl.py, with a read-only auditor."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
PROJECT = NATIVE.parent
TRAIN, PROTO = NATIVE / "train", NATIVE / "proto"
TRAIN_SCRIPT = TRAIN / "train_wsl.py"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))
from contract_identity import (  # noqa: E402
    IDENTITY_SCHEMA_VERSION,
    attach_contract_identity,
    canonical_contract_identity,
    contract_identity_sha256,
)
BANNER = "WSL_NATIVE_TRAINING_PREP"
EXPECTED = {"num_envs": 64, "num_steps": 32, "device": "cuda", "physics": "newton_xpbd",
            "lidar": "warp", "feedback": "vqf", "sim_dt_s": 0.001, "decimation": 50,
            "policy_type": "lstm", "hidden_size": 512, "task_backend": "fast_stop_v2"}
BACKENDS = {"lidar": "warp_kernel", "depth": "analytic_warp",
            "detections": "analytic_detector_warp", "feedback": "vqf_reference"}
CHANNELS = ("lidar", "depth", "detections", "state", "feedback")
TIMING_NOTE = (
    "collection_s and learning_s are host perf_counter timings, not CUDA-event timings. wall_s covers "
    "rollout + PPO update but is read before finite guards/checkpoint saves. Launcher wall includes process "
    "startup, warmup, guards and checkpoint saves. No explicit CUDA synchronize is used."
)


class AuditError(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_git_tree(output: bytes) -> dict[str, str]:
    """Parse `git ls-tree -r -z` records without asking Git to read the index."""
    entries = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = metadata.split(b" ", 2)
        if kind == b"blob":
            entries[os.fsdecode(raw_path)] = object_id.decode("ascii")
    return entries


def _git_blob_sha(raw: bytes, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise AuditError(f"unsupported Git object format: {object_format!r}")
    hasher = hashlib.new(object_format)
    hasher.update(b"blob " + str(len(raw)).encode("ascii") + b"\0")
    hasher.update(raw)
    return hasher.hexdigest()


def _relevant_source_changes(raw_sources: dict[str, bytes], tree: dict[str, str],
                             object_format: str) -> list[dict[str, str]]:
    changes = []
    for path, raw in sorted(raw_sources.items()):
        raw_blob = _git_blob_sha(raw, object_format)
        head_blob = tree.get(path)
        if head_blob is None:
            changes.append({"path": path, "state": "untracked", "raw_git_blob_sha": raw_blob})
        elif head_blob != raw_blob:
            changes.append({"path": path, "state": "modified", "head_git_blob_sha": head_blob,
                            "raw_git_blob_sha": raw_blob})
    return changes


def read_json(path: Path) -> Any:
    def reject(value: str):
        raise ValueError(f"non-finite JSON value {value}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_identity() -> dict[str, Any]:
    git_root = next((p for p in (PROJECT, *PROJECT.parents) if (p / ".git").exists()), None)
    if git_root is None:
        raise AuditError("Git root not found")
    ignored = {"runs", "results", "__pycache__", ".cache", "source_snapshot", "source_snapshots"}
    paths = {Path(__file__).resolve(), TRAIN / "PROTOCOL_NATIVE_V1.1.md"}
    for root in (TRAIN, PROTO, PROJECT / "isaac_field"):
        if root.is_dir():
            paths.update(p.resolve() for p in root.rglob("*.py") if not ignored.intersection(p.parts))
    verify = {"m1_verification.json", "m1_vqf_verification.json", "depth_verification.json",
              "m1_depth_verification.json", "detections_verification.json",
              "detections_warp_verification.json", "detections_batch_verification.json", "m2_verification.json"}
    paths.update(p.resolve() for p in PROTO.iterdir() if p.is_file() and p.name in verify)
    files, raw_sources = [], {}
    for path in sorted(paths, key=lambda p: str(p).casefold()):
        raw = path.read_bytes()
        relative = path.relative_to(git_root).as_posix()
        files.append({"path": relative, "bytes": len(raw), "sha256": digest(raw)})
        raw_sources[relative] = raw
    details = subprocess.run(["git", "-C", str(git_root), "rev-parse", "HEAD", "--abbrev-ref", "HEAD",
                              "--show-object-format=storage"],
                             capture_output=True, text=True, shell=False, check=False)
    fields = details.stdout.splitlines()
    if details.returncode or len(fields) != 3:
        raise AuditError("could not capture Git commit/branch and object format")
    commit, branch, object_format = fields
    tree = subprocess.run(["git", "-C", str(git_root), "ls-tree", "-r", "-z", commit],
                          capture_output=True, shell=False, check=False)
    if tree.returncode:
        raise AuditError("could not read the captured Git commit tree identity")
    changes = _relevant_source_changes(raw_sources, _parse_git_tree(tree.stdout), object_format)
    change_manifest = json.dumps(changes, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {"captured_at_utc": now(),
            "git": {"commit": commit, "branch": branch, "object_format": object_format,
                    "relevant_sources_dirty": bool(changes),
                    "dirty_scope": "raw_source_bytes vs HEAD only; whole workspace dirty not measured",
                    "modified_source_paths": [x["path"] for x in changes if x["state"] == "modified"],
                    "untracked_source_paths": [x["path"] for x in changes if x["state"] == "untracked"],
                    "relevant_source_changes": changes,
                    "relevant_source_changes_sha256": digest(change_manifest)},
            "raw_source_bytes": files,
            "manifest_sha256": digest(json.dumps(files, sort_keys=True, ensure_ascii=False).encode())}


def source_hashes(identity: dict[str, Any]) -> dict[str, str]:
    return {row["path"]: row["sha256"] for row in identity["raw_source_bytes"]}


def versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for name in ("torch", "numpy", "warp-lang", "newton", "mujoco", "ovphysx", "vqf"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def runtime() -> dict[str, Any]:
    return {"system": platform.system(), "machine": platform.machine(), "release": platform.release(),
            "wsl_distro_name": os.environ.get("WSL_DISTRO_NAME")}


def check_runtime() -> None:
    host = runtime()
    if host["system"] != "Linux" or str(host["machine"]).casefold() not in {"aarch64", "arm64"}:
        raise AuditError(f"expected native ARM64 Linux, got {host}")
    if "microsoft" not in host["release"].casefold() or "ubuntu-24.04" not in (host["wsl_distro_name"] or "").casefold():
        raise AuditError(f"expected Ubuntu-24.04 WSL2, got {host}")


PROCESS_LABELS = {
    "train_wsl.py": "training", "llama-server": "llama", "llama-server.exe": "llama",
    "kit": "kit", "kit.exe": "kit", "isaac-sim": "isaac", "isaac-sim.sh": "isaac",
}


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _is_python(value: str) -> bool:
    name = _basename(value)
    if name.endswith(".exe"):
        name = name[:-4]
    return name in {"python", "python3"} or name.startswith("python3.")


def process_labels(pid: int, comm: str, argv: list[str], own_pid: int | None = None) -> list[str]:
    """Label known process executables/scripts without searching arbitrary arguments."""
    if own_pid is not None and pid == own_pid:
        return []
    names = {_basename(comm)}
    if argv:
        executable = _basename(argv[0])
        names.add(executable)
        if _is_python(executable):
            script = next((arg for arg in argv[1:] if not arg.startswith("-")), None)
            if script is not None:
                names.add(_basename(script))
        if len(argv) > 1 and executable in {"sh", "bash", "dash", "zsh"}:
            names.add(_basename(argv[1]))
    found = {PROCESS_LABELS[name] for name in names if name in PROCESS_LABELS}
    return [label for label in ("training", "llama", "kit", "isaac") if label in found]


def process_scan() -> list[dict[str, Any]]:
    matches, proc = [], Path("/proc")
    if not proc.is_dir():
        return matches
    own_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            argv = [arg.decode(errors="replace") for arg in (entry / "cmdline").read_bytes().split(b"\0") if arg]
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        labels = process_labels(pid, comm, argv, own_pid)
        if labels:
            matches.append({"pid": pid, "comm": comm, "labels": labels,
                            "command_sha256": digest("\0".join(argv).encode())})
    return matches


def gpu_snapshot() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"verified": False, "error": "nvidia-smi not found", "processes": process_scan()}
    def query(*argv: str):
        return subprocess.run([exe, *argv], capture_output=True, text=True, shell=False, check=False)
    gpu = query("--query-gpu=name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits")
    apps = query("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
    active = [x.strip() for x in apps.stdout.splitlines()
              if x.strip() and "no running processes found" not in x.casefold()]
    return {"verified": gpu.returncode == apps.returncode == 0,
            "gpu": [x.strip() for x in gpu.stdout.splitlines() if x.strip()], "compute_apps": active,
            "query_errors": [gpu.stderr.strip(), apps.stderr.strip()], "processes": process_scan()}


def train_argv(out: Path, iterations: int, seed: int, threads: int) -> list[str]:
    return [sys.executable, str(TRAIN_SCRIPT), "--num-envs", "64", "--num-steps", "32",
            "--iterations", str(iterations), "--seed", str(seed), "--threads", str(threads),
            "--device", "cuda", "--physics", "newton_xpbd", "--lidar", "warp", "--feedback", "vqf",
            "--feedback-noise", "--depth", "analytic", "--depth-noise", "--detections", "analytic",
            "--detections-warp", "--detections-noise", "--task-backend", "fast_stop_v2",
            "--sim-dt", "0.001", "--decimation", "50", "--policy", "lstm", "--lstm-hidden", "512",
            "--epochs", "4", "--lr", "0.0003", "--save-interval", "10", "--nonfinite-limit", "0",
            "--out", str(out)]


def _finite(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path} is non-finite"]
    if isinstance(value, dict):
        return [e for k, v in value.items() for e in _finite(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [e for i, v in enumerate(value) for e in _finite(v, f"{path}[{i}]")]
    return []


def _metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {n}")
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {n} is not an object")
            rows.append(row)
    return rows


def _launch_record(run: Path) -> dict[str, Any] | None:
    for parent in (run, *run.parents[:5]):
        path = parent / "benchmark_result.json"
        if path.is_file():
            try:
                record = read_json(path)
                return record if isinstance(record, dict) else None
            except (OSError, ValueError, json.JSONDecodeError):
                return None
    return None


def summarize_run(path: str | Path) -> dict[str, Any]:
    """Read-only validation; missing, failed, placeholder, or non-finite runs cannot succeed."""
    run = Path(path).expanduser().resolve()
    names = ("metrics.jsonl", "summary.json", "identity.json", "observation_contract.json", "latest.pt")
    missing = [name for name in names if not (run / name).is_file()]
    if not run.is_dir() or missing:
        return {"status": "invalid", "success": False, "run_dir": str(run),
                "errors": ["run directory missing" if not run.is_dir() else "missing: " + ", ".join(missing)],
                "capability_metrics_allowed": False}
    errors = ["failure.json is present"] if (run / "failure.json").exists() else []
    try:
        summary, identity, contract = (read_json(run / name) for name in
                                        ("summary.json", "identity.json", "observation_contract.json"))
        rows = _metrics(run / "metrics.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "success": False, "run_dir": str(run),
                "errors": [f"invalid artifact: {type(exc).__name__}: {exc}"], "capability_metrics_allowed": False}
    if not all(isinstance(x, dict) for x in (summary, identity, contract)):
        return {"status": "invalid", "success": False, "run_dir": str(run),
                "errors": ["summary/identity/contract root must be a JSON object"], "capability_metrics_allowed": False}
    errors += _finite(summary, "summary") + _finite(identity, "identity") + _finite(contract, "contract")
    for i, row in enumerate(rows):
        errors += _finite(row, f"metrics[{i}]")

    iters, start, end = summary.get("iterations"), summary.get("start_iteration"), summary.get("end_iteration")
    complete = (isinstance(iters, int) and not isinstance(iters, bool) and iters > 0
                and isinstance(start, int) and isinstance(end, int) and end == start + iters
                and len(rows) == iters and [r.get("iter") for r in rows] == list(range(start + 1, end + 1)))
    if not complete:
        errors.append(f"iteration summary/metrics mismatch: iterations={iters!r}, rows={len(rows)}")
    if summary.get("banner") != BANNER or identity.get("banner") != BANNER:
        errors.append("completed native training identity missing")
    if summary.get("phase") != "wsl_training_prep" or summary.get("capability_metrics_allowed") is not False:
        errors.append("summary is not a completed prep-phase run with capability metrics disabled")
    if contract.get("capability_metrics_allowed") is not False:
        errors.append("contract must keep capability metrics disabled")

    channels = contract.get("channels", {})
    if not isinstance(channels, dict):
        channels = {}
    for name in CHANNELS:
        channel = channels.get(name, {})
        if not isinstance(channel, dict) or channel.get("mode") != "real" or not channel.get("source"):
            errors.append(f"channel {name} is missing or not real")
            continue
        if name in BACKENDS and channel.get("backend") != BACKENDS[name]:
            errors.append(f"channel {name} backend mismatch: {channel.get('backend')!r}")
        if name in {"depth", "detections", "feedback"} and not channel.get("verified_vs"):
            errors.append(f"channel {name} verification reference missing")
    policy = identity.get("policy", {})
    actual = {"num_envs": identity.get("num_envs"), "num_steps": identity.get("num_steps"),
              "device": identity.get("device"), "physics": identity.get("physics"),
              "lidar": identity.get("lidar"), "feedback": identity.get("feedback"),
              "depth": "analytic" if channels.get("depth", {}).get("backend") == BACKENDS["depth"] else None,
              "detections": "analytic" if channels.get("detections", {}).get("mode") == "real" else None,
              "sim_dt_s": contract.get("physics_dt_s"), "decimation": contract.get("decimation"),
              "policy_type": policy.get("policy_type") if isinstance(policy, dict) else None,
              "hidden_size": policy.get("hidden_size") if isinstance(policy, dict) else None,
              "task_backend": contract.get("task_backend"),
              "detections_warp": channels.get("detections", {}).get("backend") == BACKENDS["detections"]}
    for key, expected in EXPECTED.items():
        value = actual.get(key)
        valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)) if isinstance(expected, float) else value == expected
        if not valid:
            errors.append(f"target identity {key}: expected {expected!r}, got {value!r}")
    env_identity = identity.get("env", {})
    if identity.get("task_backend") != EXPECTED["task_backend"]:
        errors.append("identity task_backend differs from requested task")
    if not isinstance(env_identity, dict) or env_identity.get("task_backend") != EXPECTED["task_backend"]:
        errors.append("identity.env task_backend differs from requested task")
    elif env_identity.get("physics") != contract.get("physics"):
        errors.append("identity.env physics payload differs from observation contract")
    if contract.get("physics_experimental") is not False or contract.get("run_id") != run.name:
        errors.append("contract run/physics identity mismatch")
    if any(summary.get(k) != v for k, v in (("num_envs", 64), ("num_steps", 32),
                                            ("device", "cuda"), ("physics", "newton_xpbd"))):
        errors.append("summary configuration differs from requested target")
    contract_sha = contract.get("observation_contract_sha256")
    if not contract_sha or summary.get("observation_contract_sha256") != contract_sha or any(
        r.get("observation_contract_sha256") != contract_sha for r in rows):
        errors.append("observation contract SHA missing/inconsistent")
    try:
        canonical_sha = contract_identity_sha256(contract)
        canonical_identity = canonical_contract_identity(contract)
    except (TypeError, ValueError) as exc:
        errors.append(f"shared task/physics contract identity is invalid: {exc}")
    else:
        if contract.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
            errors.append("contract identity schema version mismatch")
        if identity.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
            errors.append("identity schema version mismatch")
        if not isinstance(env_identity, dict) or env_identity.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION:
            errors.append("identity.env schema version mismatch")
        if contract.get("task_identity") != canonical_identity:
            errors.append("stored task_identity differs from shared canonical identity")
        for name, value in (
            ("contract.task_identity_sha256", contract.get("task_identity_sha256")),
            ("contract.observation_contract_sha256", contract_sha),
            ("identity.task_identity_sha256", identity.get("task_identity_sha256")),
            ("identity.env.task_identity_sha256", env_identity.get("task_identity_sha256")
             if isinstance(env_identity, dict) else None),
        ):
            if value != canonical_sha:
                errors.append(f"{name} differs from shared canonical identity SHA")
        if identity.get("task_identity") != canonical_identity:
            errors.append("identity task_identity differs from shared canonical identity")

    totals = {"samples": 0, "collection_s": 0.0, "learning_s": 0.0, "wall_s": 0.0}
    for i, row in enumerate(rows):
        samples = row.get("samples")
        if samples != 64 * 32:
            errors.append(f"metrics[{i}] sample count is not 64x32")
            continue
        totals["samples"] += samples
        for field in ("collection_s", "learning_s", "wall_s"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                errors.append(f"metrics[{i}].{field} must be finite and positive")
            else:
                totals[field] += value
                rate_field = "samples_per_s" if field == "collection_s" else "end_to_end_samples_per_s" if field == "wall_s" else None
                if rate_field and isinstance(row.get(rate_field), (int, float)):
                    if not math.isclose(float(row[rate_field]), samples / value, rel_tol=0.025, abs_tol=0.05):
                        errors.append(f"metrics[{i}].{rate_field} denominator mismatch")
        if row.get("nonfinite_env_resets") != 0 or row.get("nonfinite_actions") != 0:
            errors.append(f"metrics[{i}] reports non-finite reset/action")
    nf = run / "nonfinite_events.jsonl"
    if nf.exists() and nf.stat().st_size:
        errors.append("nonfinite_events.jsonl is non-empty")

    checkpoint = (run / "latest.pt").read_bytes()
    ck = {"path": str(run / "latest.pt"), "bytes": len(checkpoint), "sha256": digest(checkpoint)}
    ref = summary.get("final_checkpoint")
    refs = [Path(str(ref))] if ref and Path(str(ref)).is_absolute() else [run / str(ref), Path.cwd() / str(ref)] if ref else []
    if not any(p.is_file() and p.resolve() == (run / "latest.pt").resolve() for p in refs):
        errors.append("summary final_checkpoint does not point to run/latest.pt")
    launch = _launch_record(run)
    if not launch or not launch.get("source_identity", {}).get("raw_source_bytes"):
        errors.append("launch-time raw source identity missing")
    if launch and launch.get("execution", {}).get("returncode") != 0:
        errors.append(f"training child exit code is {launch.get('execution', {}).get('returncode')!r}")
    warmup = launch.get("warmup_iterations", 0) if launch else 0
    measured = rows[warmup:] if isinstance(warmup, int) and 0 <= warmup <= len(rows) else rows
    measured_samples = sum(r.get("samples", 0) for r in measured if isinstance(r.get("samples"), int))
    def rate(field: str, selected: list[dict[str, Any]]) -> float | None:
        count = sum(r.get("samples", 0) for r in selected if isinstance(r.get("samples"), int))
        secs = sum(float(r[field]) for r in selected if isinstance(r.get(field), (int, float)))
        return count / secs if secs > 0 else None
    launch_wall = launch.get("execution", {}).get("launcher_wall_s") if launch else None
    report = {"schema_version": 1, "mode": "read_only_native_run_summary", "generated_at_utc": now(),
              "run_dir": str(run), "status": "success" if not errors else "invalid", "success": not errors,
              "errors": errors, "iterations": iters, "warmup_iterations": warmup,
              "measured_iterations": len(measured), "seed": identity.get("seed"),
              "observation_contract_sha256": contract_sha, "checkpoint": ck,
              "task_backend": contract.get("task_backend"),
              "identity_schema_version": contract.get("identity_schema_version"),
              "task_identity_sha256": contract.get("task_identity_sha256"),
              "total_samples": totals["samples"], "channels": {k: channels.get(k) for k in CHANNELS},
              "throughput": {"measured_samples": measured_samples,
                             "collection_samples_per_s": rate("collection_s", measured),
                             "ppo_update_samples_per_s": rate("learning_s", measured),
                             "iteration_window_samples_per_s": rate("wall_s", measured),
                             "iteration_window_definition": "rollout + PPO; excludes post-iteration guard/save",
                             "launcher_wall_s": launch_wall,
                             "launch_full_measured_samples_per_s": measured_samples / launch_wall if isinstance(launch_wall, (int, float)) and launch_wall > 0 else None,
                             "launch_full_all_samples_per_s": totals["samples"] / launch_wall if isinstance(launch_wall, (int, float)) and launch_wall > 0 else None,
                             "launch_full_definition": "whole child process including startup, warmup and checkpoint save",
                             "timing_note": TIMING_NOTE},
              "source_identity": launch.get("source_identity") if launch else None,
              "git_identity": launch.get("source_identity", {}).get("git") if launch else None,
              "dependency_versions": launch.get("dependency_versions") if launch else None,
              "resource_contention": launch.get("resource_contention") if launch else None,
              "capability_metrics_allowed": False, "capability_claim": False}
    return report


def _sample_gpu(path: Path, stop: threading.Event, count: list[int]) -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return
    query = ["--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"]
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        while not stop.is_set():
            try:
                r = subprocess.run([exe, *query], capture_output=True, text=True, shell=False, check=False)
                row = {"sampled_at_utc": now(), "returncode": r.returncode, "stdout_raw": r.stdout, "stderr_raw": r.stderr}
            except OSError as exc:
                row = {"sampled_at_utc": now(), "returncode": None, "stdout_raw": "",
                       "stderr_raw": f"{type(exc).__name__}: {exc}"}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            count[0] += 1
            if stop.wait(1.0):
                break


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    check_runtime()
    if not args.exclusive_window:
        raise AuditError("pass --exclusive-window to record the operator's isolation declaration")
    if min(args.iterations, args.warmup_iterations, args.threads) < 1:
        raise AuditError("iteration and thread counts must be positive")
    out = Path(args.out).expanduser().resolve() if args.out else TRAIN / "runs" / f"benchmark_{now().replace(':', '').replace('-', '')}_seed{args.seed}"
    if out.exists() or out.is_symlink():
        raise AuditError(f"refusing to overwrite output: {out}")
    sources, before = source_identity(), gpu_snapshot()
    if not before.get("verified") or before.get("compute_apps"):
        raise AuditError(f"GPU busy/unverifiable before run: {before}")
    busy = [p for p in before["processes"] if set(p["labels"]) & {"training", "llama", "kit", "isaac"}]
    if busy:
        raise AuditError(f"known training/LLM/Kit process found: {busy}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    run, result_path = out / "run", out / "benchmark_result.json"
    total_iters = args.warmup_iterations + args.iterations
    argv = train_argv(run, total_iters, args.seed, args.threads)
    log_path, sample_path = out / "train.launcher.log", out / "nvidia_smi_samples.jsonl"
    result = {"schema_version": 1, "mode": "native_wsl_training_benchmark", "status": "running",
              "started_at_utc": now(), "output_dir": str(out), "seed": args.seed,
              "warmup_iterations": args.warmup_iterations, "measured_iterations": args.iterations,
              "child_iterations": total_iters, "target_profile": EXPECTED, "runtime": runtime(),
              "dependency_versions": versions(), "source_identity": sources,
              "resource_contention": {"operator_statement": args.resource_contention_note,
                                       "gpu_process_snapshots": "nvidia-smi before and after",
                                       "host_process_snapshots": "read-only /proc scan for train/llama/Kit",
                                       "telemetry": "raw 1 Hz utilization/memory/temp/SM clock/power; power is unreliable",
                                       "this_tool_stops_processes": False},
              "gpu_before": before, "capability_metrics_allowed": False, "capability_claim": False,
              "timing_note": TIMING_NOTE,
              "execution": {"command_argv": argv, "shell": False, "returncode": None,
                            "log_path": str(log_path), "nvidia_smi_samples_path": str(sample_path),
                            "sample_interval_s": 1.0}}
    write_json(result_path, result)
    stop, count = threading.Event(), [0]
    sampler = threading.Thread(target=_sample_gpu, args=(sample_path, stop, count), daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        with log_path.open("wb") as log:
            child = subprocess.run(argv, cwd=str(PROJECT), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, shell=False, check=False)
            log.flush()
            os.fsync(log.fileno())
        rc = child.returncode
    except OSError as exc:
        rc = 127
        result["execution"]["launch_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        sampler.join(timeout=3)
    wall = time.perf_counter() - start
    after = gpu_snapshot()
    result["execution"].update(returncode=rc, launcher_wall_s=wall, telemetry_samples=count[0])
    result["gpu_after"] = after
    write_json(result_path, result)
    run_report = summarize_run(run)
    result["run_audit"] = run_report
    result["source_identity_still_matches_at_finish"] = source_hashes(sources) == source_hashes(source_identity())
    errors = []
    if rc != 0:
        errors.append(f"training child exit code {rc}")
    if not run_report["success"]:
        errors.extend(run_report["errors"])
    if run_report.get("iterations") != total_iters:
        errors.append("completed iteration count differs from warmup + measured request")
    if not result["source_identity_still_matches_at_finish"]:
        errors.append("source bytes changed during run")
    if not after.get("verified") or after.get("compute_apps") or any(
        set(p["labels"]) & {"training", "llama", "kit", "isaac"} for p in after.get("processes", [])
    ):
        errors.append("post-run GPU/process snapshot is not clear")
    if errors:
        result.update(status="failed", success=False, errors=errors, completed_at_utc=now())
    else:
        rows = _metrics(run / "metrics.jsonl")[args.warmup_iterations:]
        samples = sum(r["samples"] for r in rows)
        sums = {key: sum(r[field] for r in rows) for key, field in
                (("collection", "collection_s"), ("ppo", "learning_s"), ("iteration", "wall_s"))}
        result["throughput"] = {"measured_samples": samples,
                                "warmup_samples_excluded": args.warmup_iterations * 64 * 32,
                                "collection_samples_per_s": samples / sums["collection"],
                                "ppo_update_samples_per_s": samples / sums["ppo"],
                                "iteration_window_samples_per_s": samples / sums["iteration"],
                                "launch_full_wall_s": wall,
                                "launch_full_measured_samples_per_s": samples / wall,
                                "launch_full_all_samples_per_s": run_report["total_samples"] / wall,
                                "launch_full_scope": "same child process; warmup excluded from numerator, included in wall denominator"}
        result.update(status="success", success=True, completed_at_utc=now(),
                      summary="throughput evidence only; no capability claim")
    write_json(result_path, result)
    if not result["success"]:
        raise AuditError(f"benchmark failed; evidence retained at {result_path}: {errors}")
    return result


def summarize_benchmark(path: Path) -> dict[str, Any]:
    record = read_json(path / "benchmark_result.json")
    run = summarize_run(path / "run")
    errors = list(record.get("errors", []))
    if record.get("status") != "success" or record.get("success") is not True:
        errors.append(f"benchmark status is {record.get('status')!r}")
    if not run.get("success"):
        errors.extend(run.get("errors", []))
    if run.get("iterations") != record.get("child_iterations"):
        errors.append("run iteration count differs from the launcher request")
    if run.get("warmup_iterations") != record.get("warmup_iterations"):
        errors.append("run warmup split differs from the launcher request")
    if record.get("source_identity_still_matches_at_finish") is not True:
        errors.append("source hashes were not verified unchanged at finish")
    return {"mode": "read_only_benchmark_summary", "generated_at_utc": now(),
            "benchmark_dir": str(path), "status": "success" if not errors else "invalid",
            "success": not errors, "errors": errors, "run": run, "throughput": record.get("throughput"),
            "source_identity": record.get("source_identity"), "git_identity": record.get("source_identity", {}).get("git"),
            "dependency_versions": record.get("dependency_versions"),
            "resource_contention": record.get("resource_contention"),
            "command_and_seed": {"seed": record.get("seed"),
                                 "argv": record.get("execution", {}).get("command_argv"), "shell": False},
            "capability_metrics_allowed": False, "capability_claim": False,
            "timing_note": record.get("timing_note", TIMING_NOTE)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", metavar="RUN", help="read-only audit a run or benchmark directory")
    mode.add_argument("--dry-run", action="store_true", help="print the child command; do not start training")
    parser.add_argument("--out", default="", help="new output directory; an existing path is rejected")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5, help="measured iterations")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exclusive-window", action="store_true", help="record operator's exclusive GPU window")
    parser.add_argument("--resource-contention-note", default="No other known training/inference work; desktop GPU activity and aria2 downloads remain; this is not whole-system isolation.")
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            path = Path(args.summarize).expanduser().resolve()
            report = summarize_benchmark(path) if (path / "benchmark_result.json").is_file() else summarize_run(path)
        elif args.dry_run:
            out = Path(args.out).expanduser().resolve() if args.out else TRAIN / "runs" / "planned"
            if out.exists() or out.is_symlink():
                raise AuditError(f"refusing to overwrite output: {out}")
            report = {"mode": "dry_run", "status": "planned", "success": False, "output_dir": str(out),
                      "child_argv": train_argv(out / "run", args.warmup_iterations + args.iterations,
                                               args.seed, args.threads),
                      "warmup_iterations_in_child": args.warmup_iterations,
                      "measured_iterations_in_child": args.iterations, "shell": False,
                      "capability_metrics_allowed": False, "capability_claim": False}
        else:
            report = run_benchmark(args)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if report.get("success") is True or report.get("status") == "planned" else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "success": False,
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
