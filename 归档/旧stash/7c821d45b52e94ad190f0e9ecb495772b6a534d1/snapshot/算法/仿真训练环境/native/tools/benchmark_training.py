#!/usr/bin/env python3
"""Short, reproducible native WSL throughput run and read-only run audit."""
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
TRAIN = NATIVE / "train"
PROTO = NATIVE / "proto"
ISAAC_FIELD = PROJECT / "isaac_field"
TRAIN_SCRIPT = TRAIN / "train_wsl.py"
BANNER = "WSL_NATIVE_TRAINING_PREP"
EXPECTED = {
    "num_envs": 64, "num_steps": 32, "device": "cuda", "physics": "newton_xpbd",
    "lidar": "warp", "feedback": "vqf", "depth": "analytic", "detections": "analytic",
    "detections_warp": True, "sim_dt_s": 0.001, "decimation": 50,
    "policy_type": "lstm", "hidden_size": 512, "task_backend": "fast_stop_v2",
}
BACKENDS = {
    "lidar": "warp_kernel", "depth": "analytic_warp",
    "detections": "analytic_detector_warp", "feedback": "vqf_reference",
}
CHANNELS = ("lidar", "depth", "detections", "state", "feedback")
TIMING_NOTE = (
    "collection_s and learning_s use train_wsl.py host perf_counter timers; no CUDA events or explicit "
    "CUDA synchronization are used. wall_s covers rollout + PPO update but is read before finite guards "
    "and checkpoint writes. Launcher wall covers the entire child process, including startup, warmup, "
    "guards and checkpoint writes."
)


class AuditError(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    def reject(value: str):
        raise ValueError(f"non-finite JSON constant {value}")
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
    paths: set[Path] = {Path(__file__).resolve()}
    ignored = {"runs", "results", "__pycache__", ".cache", "source_snapshot", "source_snapshots"}
    for root in (TRAIN, PROTO, ISAAC_FIELD):
        if root.is_dir():
            paths.update(p.resolve() for p in root.rglob("*.py") if not ignored.intersection(p.parts))
    verification_names = {
        "m1_verification.json", "m1_vqf_verification.json", "depth_verification.json",
        "m1_depth_verification.json", "detections_verification.json",
        "detections_warp_verification.json", "detections_batch_verification.json", "m2_verification.json",
    }
    paths.add(TRAIN / "PROTOCOL_NATIVE_V1.1.md")
    paths.update(p.resolve() for p in PROTO.iterdir() if p.name in verification_names and p.is_file())
    files = []
    for path in sorted(paths, key=lambda p: str(p).casefold()):
        raw = path.read_bytes()
        shown = path.relative_to(git_root).as_posix() if git_root else str(path)
        files.append({"path": shown, "bytes": len(raw), "sha256": sha(raw)})
    if git_root is None:
        raise AuditError("Git root not found")
    commit = subprocess.run(["git", "-C", str(git_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, shell=False, check=False)
    status = subprocess.run(["git", "-C", str(git_root), "status", "--porcelain=v1", "-z",
                             "--untracked-files=all"], capture_output=True, shell=False, check=False)
    if commit.returncode or status.returncode:
        raise AuditError("could not capture Git commit and dirty identity")
    raw_status = status.stdout
    return {
        "captured_at_utc": now(),
        "git": {"commit": commit.stdout.strip(), "dirty": bool(raw_status),
                "dirty_entry_count": sum(bool(x) for x in raw_status.split(b"\0")),
                "status_porcelain_sha256": sha(raw_status)},
        "raw_source_bytes": files,
        "source_manifest_sha256": sha(json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")),
    }


def source_hashes(identity: dict[str, Any]) -> dict[str, str]:
    return {item["path"]: item["sha256"] for item in identity.get("raw_source_bytes", [])}


def dependencies() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "numpy", "warp-lang", "newton", "mujoco", "ovphysx", "vqf"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def runtime() -> dict[str, Any]:
    return {"system": platform.system(), "machine": platform.machine(), "release": platform.release(),
            "wsl_distro_name": os.environ.get("WSL_DISTRO_NAME")}


def runtime_errors() -> list[str]:
    host = runtime()
    errors = []
    if host["system"] != "Linux":
        errors.append("run the benchmark inside WSL2 Linux")
    if str(host["machine"]).casefold() not in {"aarch64", "arm64"}:
        errors.append(f"expected native ARM64 Python, got {host['machine']!r}")
    distro = (host["wsl_distro_name"] or "").casefold()
    if "microsoft" not in host["release"].casefold() or "ubuntu-24.04" not in distro:
        errors.append("expected Ubuntu 24.04 on WSL2")
    return errors


def selected_processes() -> list[dict[str, Any]]:
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    markers = {"training": "train_wsl.py", "llama": "llama-server", "kit": "kit", "isaac": "isaac"}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        labels = [label for label, text in markers.items() if text in (comm + " " + cmd).casefold()]
        if labels:
            matches.append({"pid": int(entry.name), "comm": comm, "labels": labels,
                            "command_sha256": sha(cmd.encode("utf-8"))})
    return matches


def gpu_snapshot() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"verified": False, "error": "nvidia-smi not found", "host_processes": selected_processes()}
    def query(*args: str):
        return subprocess.run([exe, *args], capture_output=True, text=True, shell=False, check=False)
    gpu = query("--query-gpu=name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits")
    apps = query("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
    lines = [x.strip() for x in apps.stdout.splitlines() if x.strip()
             and "no running processes found" not in x.casefold()]
    return {"verified": gpu.returncode == 0 and apps.returncode == 0,
            "gpu_rows": [x.strip() for x in gpu.stdout.splitlines() if x.strip()],
            "compute_apps": lines, "gpu_query_returncode": gpu.returncode,
            "compute_app_query_returncode": apps.returncode,
            "gpu_query_stderr": gpu.stderr.strip(), "compute_app_query_stderr": apps.stderr.strip(),
            "host_processes": selected_processes()}


def train_argv(out: Path, iterations: int, seed: int, threads: int) -> list[str]:
    return [sys.executable, str(TRAIN_SCRIPT),
            "--num-envs", "64", "--num-steps", "32", "--iterations", str(iterations),
            "--seed", str(seed), "--threads", str(threads), "--device", "cuda",
            "--physics", "newton_xpbd", "--lidar", "warp", "--feedback", "vqf", "--feedback-noise",
            "--depth", "analytic", "--depth-noise", "--detections", "analytic", "--detections-warp",
            "--detections-noise", "--task-backend", "fast_stop_v2", "--sim-dt", "0.001",
            "--decimation", "50", "--policy", "lstm", "--lstm-hidden", "512",
            "--epochs", "4", "--lr", "0.0003", "--save-interval", "10",
            "--nonfinite-limit", "0", "--out", str(out)]


def _finite(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{prefix} is not finite"]
    if isinstance(value, dict):
        return [e for k, v in value.items() for e in _finite(v, f"{prefix}.{k}")]
    if isinstance(value, list):
        return [e for i, v in enumerate(value) for e in _finite(v, f"{prefix}[{i}]")]
    return []


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {n}")
            row = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {n} is not an object")
            rows.append(row)
    return rows


def _launch_record(run: Path) -> dict[str, Any] | None:
    for parent in (run, *run.parents[:5]):
        path = parent / "benchmark_result.json"
        if path.is_file():
            try:
                record = load_json(path)
                if isinstance(record, dict):
                    return record
            except (OSError, ValueError, json.JSONDecodeError):
                return None
    return None


def summarize_run(run_path: str | Path) -> dict[str, Any]:
    """Read only; incomplete, failed, placeholder, or non-finite runs never succeed."""
    run = Path(run_path).expanduser().resolve()
    errors: list[str] = []
    required = {name: run / name for name in
                ("metrics.jsonl", "summary.json", "identity.json", "observation_contract.json", "latest.pt")}
    missing = [name for name, path in required.items() if not path.is_file()]
    if not run.is_dir() or missing:
        errors.append("run directory missing" if not run.is_dir() else "missing: " + ", ".join(missing))
        return {"status": "invalid", "success": False, "run_dir": str(run), "errors": errors,
                "capability_metrics_allowed": False}
    if (run / "failure.json").exists():
        errors.append("failure.json is present")
    try:
        summary, identity, contract = (load_json(required[name]) for name in
                                        ("summary.json", "identity.json", "observation_contract.json"))
        rows = _load_metrics(required["metrics.jsonl"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "success": False, "run_dir": str(run),
                "errors": [f"invalid run artifact: {type(exc).__name__}: {exc}"],
                "capability_metrics_allowed": False}
    if not all(isinstance(x, dict) for x in (summary, identity, contract)):
        errors.append("summary, identity, and contract roots must be JSON objects")
        summary, identity, contract = (x if isinstance(x, dict) else {} for x in (summary, identity, contract))
    errors += _finite(summary, "summary") + _finite(identity, "identity") + _finite(contract, "contract")
    for i, row in enumerate(rows):
        errors += _finite(row, f"metrics[{i}]")

    iters = summary.get("iterations")
    start, end = summary.get("start_iteration"), summary.get("end_iteration")
    complete = (isinstance(iters, int) and not isinstance(iters, bool) and iters > 0
                and isinstance(start, int) and isinstance(end, int) and end == start + iters
                and len(rows) == iters and [r.get("iter") for r in rows] == list(range(start + 1, end + 1)))
    if not complete:
        errors.append(f"iteration summary/metrics mismatch: iterations={iters!r}, rows={len(rows)}")
    if summary.get("banner") != BANNER or identity.get("banner") != BANNER:
        errors.append("completed native training banner missing")
    if summary.get("phase") != "wsl_training_prep" or summary.get("capability_metrics_allowed") is not False:
        errors.append("run is not a completed prep-phase throughput run with capability metrics disabled")
    if contract.get("capability_metrics_allowed") is not False:
        errors.append("contract does not explicitly disable capability metrics")

    channels = contract.get("channels", {})
    if not isinstance(channels, dict):
        channels = {}
    for name in CHANNELS:
        channel = channels.get(name, {})
        if not isinstance(channel, dict) or channel.get("mode") != "real" or not channel.get("source"):
            errors.append(f"channel {name} is missing or not real")
            continue
        if name in BACKENDS and channel.get("backend") != BACKENDS[name]:
            errors.append(f"channel {name} backend is {channel.get('backend')!r}")
        if name in {"depth", "detections", "feedback"} and not channel.get("verified_vs"):
            errors.append(f"channel {name} verification reference missing")

    policy = identity.get("policy", {})
    actual = {
        "num_envs": identity.get("num_envs"), "num_steps": identity.get("num_steps"),
        "device": identity.get("device"), "physics": identity.get("physics"),
        "lidar": identity.get("lidar"), "feedback": identity.get("feedback"),
        "depth": "analytic" if channels.get("depth", {}).get("backend") == BACKENDS["depth"] else None,
        "detections": "analytic" if channels.get("detections", {}).get("mode") == "real" else None,
        "detections_warp": channels.get("detections", {}).get("backend") == BACKENDS["detections"],
        "sim_dt_s": contract.get("physics_dt_s"), "decimation": contract.get("decimation"),
        "policy_type": policy.get("policy_type") if isinstance(policy, dict) else None,
        "hidden_size": policy.get("hidden_size") if isinstance(policy, dict) else None,
        "task_backend": contract.get("task_backend"),
    }
    for key, expected in EXPECTED.items():
        value = actual.get(key)
        ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
              and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-12)) if isinstance(expected, float) else value == expected
        if not ok:
            errors.append(f"identity mismatch {key}: expected {expected!r}, got {value!r}")
    if contract.get("physics_experimental") is not False:
        errors.append("contract does not identify the 1 ms/50 physics as non-experimental")
    if contract.get("run_id") != run.name:
        errors.append(f"contract run_id mismatch: {contract.get('run_id')!r}")
    if any(summary.get(k) != v for k, v in (("num_envs", 64), ("num_steps", 32),
                                            ("device", "cuda"), ("physics", "newton_xpbd"))):
        errors.append("summary configuration differs from target identity")

    contract_sha = contract.get("observation_contract_sha256")
    if not contract_sha or summary.get("observation_contract_sha256") != contract_sha or any(
        row.get("observation_contract_sha256") != contract_sha for row in rows
    ):
        errors.append("observation contract SHA missing or inconsistent across artifacts")
    rates = {"samples": 0, "collection_s": 0.0, "learning_s": 0.0, "wall_s": 0.0}
    for i, row in enumerate(rows):
        if row.get("samples") != EXPECTED["num_envs"] * EXPECTED["num_steps"]:
            errors.append(f"metrics[{i}] sample count is not 64x32")
            continue
        rates["samples"] += int(row["samples"])
        for field in ("collection_s", "learning_s", "wall_s"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                errors.append(f"metrics[{i}].{field} must be finite and positive")
            else:
                rates[field] += float(value)
        if row.get("nonfinite_env_resets") != 0 or row.get("nonfinite_actions") != 0:
            errors.append(f"metrics[{i}] records a non-finite reset/action")
    nonfinite = run / "nonfinite_events.jsonl"
    if nonfinite.exists() and nonfinite.stat().st_size:
        errors.append("nonfinite_events.jsonl is non-empty")
    checkpoint = required["latest.pt"].read_bytes()
    checkpoint_info = {"path": str(required["latest.pt"]), "bytes": len(checkpoint), "sha256": sha(checkpoint)}
    final_ref = summary.get("final_checkpoint")
    if not final_ref:
        errors.append("summary final_checkpoint missing")
    else:
        ref = Path(str(final_ref))
        candidates = [ref] if ref.is_absolute() else [run / ref, Path.cwd() / ref]
        if not any(p.is_file() and p.resolve() == required["latest.pt"].resolve() for p in candidates):
            errors.append("summary final_checkpoint does not point to this run/latest.pt")

    launch = _launch_record(run)
    if not launch or not launch.get("source_identity", {}).get("raw_source_bytes"):
        errors.append("launch-time source byte identity is missing")
    if launch and launch.get("execution", {}).get("returncode") != 0:
        errors.append(f"training child exit code is {launch.get('execution', {}).get('returncode')!r}")
    warmup = launch.get("warmup_iterations", 0) if launch else 0
    measured = rows[warmup:] if isinstance(warmup, int) and 0 <= warmup <= len(rows) else rows
    measured_count = sum(int(r.get("samples", 0)) for r in measured if isinstance(r.get("samples"), int))
    def rate(selected: list[dict[str, Any]], field: str) -> float | None:
        count = sum(int(r.get("samples", 0)) for r in selected if isinstance(r.get("samples"), int))
        elapsed = sum(float(r.get(field, 0)) for r in selected if isinstance(r.get(field), (int, float)))
        return count / elapsed if elapsed > 0 else None
    launch_wall = launch.get("execution", {}).get("launcher_wall_s") if launch else None
    throughput = {
        "measured_samples": measured_count,
        "collection_samples_per_s": rate(measured, "collection_s"),
        "ppo_update_samples_per_s": rate(measured, "learning_s"),
        "iteration_window_samples_per_s": rate(measured, "wall_s"),
        "iteration_window_definition": "collection + PPO update; excludes post-iteration guards and checkpoint save",
        "launcher_wall_s": launch_wall,
        "launch_full_measured_samples_per_s": measured_count / launch_wall if isinstance(launch_wall, (int, float)) and launch_wall > 0 else None,
        "launch_full_all_samples_per_s": rates["samples"] / launch_wall if isinstance(launch_wall, (int, float)) and launch_wall > 0 else None,
        "launch_full_definition": "entire child process; includes startup, 2 warmup iterations, guards and checkpoint save",
        "timing_note": TIMING_NOTE,
    }
    return {
        "schema_version": 1, "mode": "read_only_native_run_summary", "generated_at_utc": now(),
        "run_dir": str(run), "status": "success" if not errors else "invalid", "success": not errors,
        "errors": errors, "iterations": iters, "warmup_iterations": warmup,
        "measured_iterations": len(measured), "seed": identity.get("seed"),
        "observation_contract_sha256": contract_sha, "checkpoint": checkpoint_info,
        "total_samples": rates["samples"], "throughput": throughput,
        "source_identity": launch.get("source_identity") if launch else None,
        "git_identity": launch.get("source_identity", {}).get("git") if launch else None,
        "dependency_versions": launch.get("dependency_versions") if launch else None,
        "resource_contention": launch.get("resource_contention") if launch else None,
        "channels": {name: {"mode": channels.get(name, {}).get("mode"),
                             "backend": channels.get(name, {}).get("backend")}
                     for name in CHANNELS},
        "capability_metrics_allowed": False, "capability_claim": False,
    }


def _write_gpu_samples(path: Path, stop: threading.Event, count: list[int]) -> None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return
    query = ["--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"]
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        while not stop.is_set():
            try:
                r = subprocess.run([exe, *query], capture_output=True, text=True, shell=False, check=False)
                item = {"sampled_at_utc": now(), "returncode": r.returncode,
                        "stdout_raw": r.stdout, "stderr_raw": r.stderr}
            except OSError as exc:
                item = {"sampled_at_utc": now(), "returncode": None,
                        "stdout_raw": "", "stderr_raw": f"{type(exc).__name__}: {exc}"}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            count[0] += 1
            if stop.wait(1.0):
                break


def _resource(note: str) -> dict[str, Any]:
    return {"operator_statement": note, "gpu_process_snapshots": "nvidia-smi before and after training",
            "other_process_snapshots": "read-only /proc scan for train_wsl.py, llama-server, Isaac/Kit",
            "telemetry": "raw 1 Hz GPU utilization/memory/temperature/SM clock/power samples retained",
            "power_reading_reliable": False, "this_tool_stops_processes": False,
            "limits": "snapshots cannot rule out CPU contention or a GPU process appearing mid-run"}


def default_out(seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return TRAIN / "runs" / f"benchmark_{stamp}_seed{seed}"


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    problems = runtime_errors()
    if problems:
        raise AuditError("; ".join(problems))
    if not args.exclusive_window:
        raise AuditError("pass --exclusive-window to record the operator's exclusive GPU window")
    if not args.exclusive_window:
        raise AuditError("pass --exclusive-window to record the operator's exclusive GPU window")
    if args.warmup_iterations < 1 or args.iterations < 1 or args.threads < 1:
        raise AuditError("warmup iterations, measured iterations and threads must be positive")
    output = Path(args.out).expanduser().resolve() if args.out else default_out(args.seed).resolve()
    if output.exists() or output.is_symlink():
        raise AuditError(f"refusing to overwrite existing output: {output}")
    src = source_identity()
    before = gpu_snapshot()
    if not before.get("verified") or before.get("compute_apps"):
        raise AuditError(f"GPU not idle/verifiable before run: {before}")
    workload = before.get("host_processes", [])
    if any(set(p["labels"]) & {"training", "llama", "kit", "isaac"} for p in workload):
        raise AuditError(f"known concurrent training/LLM/Kit process found: {workload}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    run_dir = output / "run"
    report_path = output / "benchmark_result.json"
    result: dict[str, Any] = {
        "schema_version": 1, "mode": "native_wsl_training_benchmark", "status": "running",
        "started_at_utc": now(), "output_dir": str(output), "seed": args.seed,
        "warmup_iterations": args.warmup_iterations, "measured_iterations": args.iterations,
        "child_iterations": args.warmup_iterations + args.iterations, "threads": args.threads,
        "target_profile": EXPECTED, "runtime": runtime(), "dependency_versions": dependencies(),
        "source_identity": src, "resource_contention": _resource(args.resource_contention_note),
        "gpu_before": before, "capability_metrics_allowed": False, "capability_claim": False,
        "timing_note": TIMING_NOTE,
    }
    write_json(report_path, result)
    argv = train_argv(run_dir, result["child_iterations"], args.seed, args.threads)
    log_path, samples_path = output / "train.launcher.log", output / "nvidia_smi_samples.jsonl"
    result["execution"] = {"command_argv": argv, "shell": False, "log_path": str(log_path),
                           "nvidia_smi_samples_path": str(samples_path), "sample_interval_s": 1.0,
                           "status": "running"}
    write_json(report_path, result)
    sampler_stop, sample_count = threading.Event(), [0]
    sampler = threading.Thread(target=_write_gpu_samples, args=(samples_path, sampler_stop, sample_count),
                               name="gpu-sampler", daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        with log_path.open("wb") as log:
            child = subprocess.run(argv, cwd=str(PROJECT), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, shell=False, check=False)
            log.flush()
            os.fsync(log.fileno())
    finally:
        sampler_stop.set()
        sampler.join(timeout=3)
    wall = time.perf_counter() - start
    after = gpu_snapshot()
    result["execution"].update(returncode=child.returncode, launcher_wall_s=wall,
                               telemetry_samples=sample_count[0], status="finished")
    result["gpu_after"] = after
    run_report = summarize_run(run_dir)
    result["run_audit"] = run_report
    source_same = source_hashes(src) == source_hashes(source_identity())
    result["source_identity_still_matches_at_finish"] = source_same
    problems = []
    if child.returncode != 0:
        problems.append(f"train_wsl.py exit code {child.returncode}")
    if not run_report["success"]:
        problems.extend(run_report["errors"])
    if not source_same:
        problems.append("source bytes changed during run")
    if not after.get("verified") or after.get("compute_apps"):
        problems.append("GPU process state is not clear/verifiable after run")
    if any(set(p["labels"]) & {"training", "llama", "kit", "isaac"} for p in after.get("host_processes", [])):
        problems.append("training/LLM/Kit process appeared in post-run process scan")
    if run_report.get("iterations") != result["child_iterations"]:
        problems.append("completed iteration count differs from warmup + measured request")
    if problems:
        result.update(status="failed", success=False, errors=problems, completed_at_utc=now())
    else:
        measured = _load_metrics(run_dir / "metrics.jsonl")[args.warmup_iterations:]
        samples = sum(int(r["samples"]) for r in measured)
        collect = sum(float(r["collection_s"]) for r in measured)
        ppo = sum(float(r["learning_s"]) for r in measured)
        iteration = sum(float(r["wall_s"]) for r in measured)
        result["throughput"] = {
            "measured_samples": samples,
            "warmup_samples_excluded": args.warmup_iterations * EXPECTED["num_envs"] * EXPECTED["num_steps"],
            "collection_samples_per_s": samples / collect,
            "ppo_update_samples_per_s": samples / ppo,
            "iteration_window_samples_per_s": samples / iteration,
            "launch_full_wall_s": wall,
            "launch_full_measured_samples_per_s": samples / wall,
            "launch_full_all_samples_per_s": run_report["total_samples"] / wall,
            "launch_full_scope": "one process; measured numerator excludes warmup, wall denominator includes warmup",
        }
        result.update(status="success", success=True, completed_at_utc=now(),
                      summary="throughput evidence only; no capability claim")
    write_json(report_path, result)
    if not result["success"]:
        raise AuditError(f"benchmark failed; evidence retained at {report_path}: {result['errors']}")
    return result


def summarize_benchmark(path: Path) -> dict[str, Any]:
    record = load_json(path / "benchmark_result.json")
    run = summarize_run(path / "run")
    errors = list(record.get("errors", []))
    if record.get("status") != "success" or record.get("success") is not True:
        errors.append(f"benchmark status is {record.get('status')!r}")
    if not run.get("success"):
        errors.extend(run.get("errors", []))
    if record.get("source_identity_still_matches_at_finish") is not True:
        errors.append("source bytes were not verified unchanged through the run")
    return {"schema_version": 1, "mode": "read_only_benchmark_summary", "generated_at_utc": now(),
            "benchmark_dir": str(path), "status": "success" if not errors else "invalid",
            "success": not errors, "errors": errors, "run": run,
            "throughput": record.get("throughput"), "source_identity": record.get("source_identity"),
            "git_identity": record.get("source_identity", {}).get("git"),
            "dependency_versions": record.get("dependency_versions"),
            "resource_contention": record.get("resource_contention"),
            "command_and_seed": {"seed": record.get("seed"),
                                 "argv": record.get("execution", {}).get("command_argv"), "shell": False},
            "capability_metrics_allowed": False, "capability_claim": False,
            "timing_note": record.get("timing_note", TIMING_NOTE)}


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out).expanduser().resolve() if args.out else default_out(args.seed).resolve()
    if output.exists() or output.is_symlink():
        raise AuditError(f"refusing to overwrite existing output: {output}")
    return {"mode": "dry_run", "status": "planned", "success": False, "output_dir": str(output),
            "child_argv": train_argv(output / "run", args.warmup_iterations + args.iterations,
                                      args.seed, args.threads),
            "warmup_iterations_in_child": args.warmup_iterations,
            "measured_iterations_in_child": args.iterations, "shell": False,
            "capability_metrics_allowed": False, "capability_claim": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", metavar="RUN", help="read-only audit a run or benchmark output directory")
    mode.add_argument("--dry-run", action="store_true", help="print the command without launching training")
    parser.add_argument("--out", default="", help="new output directory; existing paths are rejected")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5, help="measured iterations")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--exclusive-window", action="store_true", help="record operator's exclusive GPU window")
    parser.add_argument("--resource-contention-note",
                        default="Operator declared an exclusive GPU window; process snapshots are retained.")
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            path = Path(args.summarize).expanduser().resolve()
            if (path / "benchmark_result.json").is_file():
                report = summarize_benchmark(path)
            else:
                report = summarize_run(path)
        elif args.dry_run:
            report = dry_run(args)
        else:
            problems = runtime_errors()
            if problems:
                raise AuditError("; ".join(problems))
            report = run_benchmark(args)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if report.get("success") is True or report.get("status") == "planned" else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "success": False,
                          "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
