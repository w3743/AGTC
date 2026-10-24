"""Resume the audited a04 checkpoint with GPU-batched task math and paired evaluation.

The official RSL-RL CLI remains the trainer. This small recovery entry point
reuses the existing a04/a03/a02 pipeline helpers and only owns the new
recovery handoff, source lineage, and a06 output paths.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


TRIAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = TRIAL_DIR.parents[2]
REPO_ROOT = TRAINING_ROOT.parents[1]
A04_DIR = TRIAL_DIR.parent / "physx_pointmove_flat12_mirror_20261024_a04"
A04_PIPELINE = A04_DIR / "run_pipeline.py"
A04_RUN_DIR = A04_DIR / "logs/rsl_rl" / A04_DIR.name / "training_live_a04"
A04_AUDIT_PATH = A04_DIR / "diagnosis/reboot_20261024T1700/checkpoint_audit.json"
A04_CHECKPOINT = A04_RUN_DIR / "model_48.pt"
A02_DIR = TRIAL_DIR.parent / "physx_pointmove_flat12_20261024_a02"
A02_PIPELINE = A02_DIR / "run_pipeline.py"
HANDOFF_PATH = TRIAL_DIR / "handoff.json"
SOURCE_SNAPSHOT = TRIAL_DIR / "source_snapshot"
PROTOCOL_PATH = TRIAL_DIR / "protocol.json"
MANIFEST_PATH = TRIAL_DIR / "source_manifest.json"
STATUS_PATH = TRIAL_DIR / "run_status.json"
PREFLIGHT_PATH = TRIAL_DIR / "preflight.json"
LOCK_PATH = TRIAL_DIR / ".run_pipeline.lock"
LIVE_STATE_PATH = TRIAL_DIR / "live_state.json"
PYTHON_EXE = Path(r"C:\pioneer\isaac-training\.venv\Scripts\python.exe")
EXPERIMENT_NAME = "physx_pointmove_flat12_gpu_20261024_a06"
RUN_TIMESTAMP = "training_gpu_a06"
NUM_ENVS, ROLLOUT_STEPS, TARGET_TRANSITIONS = 4, 128, 196608
ADAM_STEPS_PER_ITERATION = 20
EVAL_SEED_START, EVAL_SCENARIOS, EVAL_GOAL_ANGLE_DEG = 87010, 16, 180.0
EXPECTED_A04_PROTOCOL_SHA256 = "d024e42b3a8af4e45eb1dbb1f6147fc115300631ece1ce95bfda13dbf99d6ab7"
EXPECTED_CHECKPOINT_SHA256 = "26c55ba2c1fc9e3da09ed174ec01b20b08c78186ee390657b400dde6152a9025"
EXPECTED_BASELINE_SHA256 = "ac1684aaf6d2cfd9a5dfcf33482221dae652aff5d33a3f4a45fdfdaa9ff3d198"
EXPECTED_CHECKPOINT_ITER, EXPECTED_CHECKPOINT_TRANSITIONS = 48, 26112
EXPECTED_CHECKPOINT_ADAM_STEP, EXPECTED_OPTIMIZER_ENTRIES = 1020, 37
RECOVERY_HANDOFF_SCHEMA = "agtc-a05-recovery-handoff-v1"
PROCESS_ROLES = ("controller", "resume_training_cli", "training_native", "watcher", "viewer")
VIEWER_SCRIPT = "scripts/view_training_state_web.py"
VIEWER_HOST, VIEWER_PORT = "127.0.0.1", 8790
TASK_BACKEND = "torch_batched"
GPU_EVALUATOR_PATH = TRIAL_DIR / "evaluate_torch_batched.py"
NATIVE_EVIDENCE_DIR = TRAINING_ROOT / "results/isaac_field_diagnosis/torch_batched_task_20261024"
NATIVE_REPORT_PATH = NATIVE_EVIDENCE_DIR / "native_drive_final_a01.json"
NATIVE_PROCESS_PATH = NATIVE_EVIDENCE_DIR / "native_drive_final_a01-process.json"
NATIVE_LOG_PATH = NATIVE_EVIDENCE_DIR / "native_drive_final_a01.log"
GPU_MATH_PATH = NATIVE_EVIDENCE_DIR / "cuda_math_final.json"
DRIVE_CHECK_PATH = NATIVE_EVIDENCE_DIR / "drive_validation.json"
CPU_GATE_PATH = NATIVE_EVIDENCE_DIR / "cpu_gate_after_drive_20261024T1233Z/cpu_gate_after_drive.json"
EXPECTED_NATIVE_EVIDENCE_SHA256 = {
    NATIVE_REPORT_PATH.name: "cacbccbe1a3bb5764f01f909e47e5a8b3c7987505114fe95c5a5110105a85869",
    NATIVE_PROCESS_PATH.name: "abb6e8cfce1f29bfd81732f0031e594b20ea5331931c1b6cc434676e18b6f8c0",
    NATIVE_LOG_PATH.name: "ecacac5381415753f8fd66da12a46c84e2efb22cfe806241d82686b4c21e98a6",
    GPU_MATH_PATH.name: "91cfa46972ac17ae1a49d71f2a7087f67525971c5e62f855ba45690db44fa4bb",
    DRIVE_CHECK_PATH.name: "6f6d1f2c4864d31c42470a50146088d3406be4f05c9145ed6060cb656d793a00",
    CPU_GATE_PATH.name: "edfa5e8c8752a94d385d73fc86b73389e7df6e88fe4125964b9b380b1d5ca0b6",
}
_PROCESS_CTIME_TOLERANCE_S = 0.01

# This exact list is the only permitted production-source delta from a04.
# Parent/current hashes are pinned so prepare fails if another edit lands before
# the a06 source freeze. New per-trial evaluator code is frozen as origin=trial.
AUTHORIZED_SOURCE_DELTAS = {
    "isaac_field/drive_control.py": {
        "kind": "modified",
        "parent_sha256": "245e1afb80a229fd651b7fcb9acbf755b2188d5e5f83fe0e7aa50e98939b33eb",
        "current_sha256": "5e59b327256320d433dde85b2c3a6b5242a0fa59df144946ec6c2bdf4824d372",
        "reason": "Cache the fixed wheel-side topology and reuse validated tensor indices in the GPU task backend.",
    },
    "isaac_field/env.py": {
        "kind": "modified",
        "parent_sha256": "f7c8d32593418c690fc2ff0beb60e465f5fd8c544463d3a5b34a3432b9ed1941",
        "current_sha256": "8bef5facd7233175c39277a1aa98c4a1b74a5103387af4bcf98d06cf9886b9a5",
        "reason": "Add the explicit torch_batched task dispatch while keeping reference as the default and preserving the sensor/physics contract.",
    },
    "isaac_field/task.py": {
        "kind": "modified",
        "parent_sha256": "cea25964cee98c88b1e0886d8e51e354d1b695c95b5f85f05f3c76b8e06f0153",
        "current_sha256": "7e68b2973d0012466614c8337448913c07b938c7f065111f06a2f1c825b83b7f",
        "reason": "Connect the batched task/sensor implementation behind the explicit backend without changing the task objective or observation contract.",
    },
    "isaac_field/tensor_sensors.py": {
        "kind": "added",
        "parent_sha256": None,
        "current_sha256": "0940e8d82cd8f94b64c567c4df31fd9a5c7d9b1a33589152fdc8ebbfef978f1f",
        "reason": "Add the tested batched sensor tensor path required by torch_batched.",
    },
    "isaac_field/tensor_task.py": {
        "kind": "added",
        "parent_sha256": None,
        "current_sha256": "e86cf0e8a3bf58d022f2a4ddc8c058d8ad37a1c3b3d4d58c116cfa4104896c7d",
        "reason": "Add the tested GPU batched reward, termination, and observation path required by torch_batched.",
    },
    "scripts/view_training_state_web.py": {
        "kind": "added",
        "parent_sha256": None,
        "current_sha256": "5dcd6e8d47733bee27bdf86426c043449b2d3a9c22b365fb87e09165f522cccd",
        "reason": "Add the independently validated lightweight read-only live-state viewer; it does not load another simulator or policy.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_a04():
    spec = importlib.util.spec_from_file_location("isaac_field_a04_recovery_base", A04_PIPELINE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reusable a04 pipeline helpers from {A04_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A04 = _load_a04()
A03 = A04.A03


def _configure_reused_pipeline() -> None:
    """Redirect the inherited helper module's globals to this isolated trial."""
    a04_values = {
        "TRIAL_DIR": TRIAL_DIR,
        "TRAINING_ROOT": TRAINING_ROOT,
        "REPO_ROOT": REPO_ROOT,
        "A03_DIR": A04_DIR,
        "A03_PIPELINE": A04_PIPELINE,
        "A03_RUN_DIR": TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / RUN_TIMESTAMP,
        "A02_DIR": A02_DIR,
        "A02_PIPELINE": A02_PIPELINE,
        "HANDOFF_PATH": HANDOFF_PATH,
        "SOURCE_SNAPSHOT": SOURCE_SNAPSHOT,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATUS_PATH": STATUS_PATH,
        "PREFLIGHT_PATH": PREFLIGHT_PATH,
        "LOCK_PATH": LOCK_PATH,
        "LIVE_STATE_PATH": LIVE_STATE_PATH,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "RUN_TIMESTAMP": RUN_TIMESTAMP,
        "NUM_ENVS": NUM_ENVS,
        "ROLLOUT_STEPS": ROLLOUT_STEPS,
        "TARGET_TRANSITIONS": TARGET_TRANSITIONS,
        "ADAM_STEPS_PER_ITERATION": ADAM_STEPS_PER_ITERATION,
        "EVAL_SEED_START": EVAL_SEED_START,
        "EVAL_SCENARIOS": EVAL_SCENARIOS,
        "EVAL_GOAL_ANGLE_DEG": EVAL_GOAL_ANGLE_DEG,
        "CORE_PROCESS_ROLES": ("training_native", "resume_training_cli", "controller", "watcher"),
    }
    for name, value in a04_values.items():
        setattr(A04, name, value)

    # a04 deliberately loads a03 as a reusable helper. Repoint only its output
    # globals; its source paths remain the original a02/a03 lineage.
    a03_values = {
        "TRIAL_DIR": TRIAL_DIR,
        "TRAINING_ROOT": TRAINING_ROOT,
        "REPO_ROOT": REPO_ROOT,
        "HANDOFF_PATH": HANDOFF_PATH,
        "SOURCE_SNAPSHOT": SOURCE_SNAPSHOT,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATUS_PATH": STATUS_PATH,
        "PREFLIGHT_PATH": PREFLIGHT_PATH,
        "LOCK_PATH": LOCK_PATH,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "RUN_TIMESTAMP": RUN_TIMESTAMP,
        "TARGET_TRANSITIONS": TARGET_TRANSITIONS,
        "NUM_ENVS": NUM_ENVS,
        "ROLLOUT_STEPS": ROLLOUT_STEPS,
        "EVAL_SEED_START": EVAL_SEED_START,
        "EVAL_SCENARIOS": EVAL_SCENARIOS,
        "EVAL_GOAL_ANGLE_DEG": EVAL_GOAL_ANGLE_DEG,
        "PROCESS_ROLES": PROCESS_ROLES,
    }
    for name, value in a03_values.items():
        setattr(A03, name, value)
_configure_reused_pipeline()
utc_now = A03.utc_now
read_json = A03.read_json
write_json_atomic = A03.write_json_atomic


def resume_plan(iteration: int, transitions: int, adam_step: int, optimizer_entries: int) -> dict[str, int]:
    batch = NUM_ENVS * ROLLOUT_STEPS
    if type(iteration) is not int or iteration < 0 or type(adam_step) is not int or adam_step < 0:
        raise ValueError("checkpoint iteration and Adam step must be non-negative integers")
    if type(transitions) is not int or transitions < 0 or transitions >= TARGET_TRANSITIONS or transitions % batch:
        raise ValueError(f"checkpoint transitions must be aligned and below {TARGET_TRANSITIONS}")
    if type(optimizer_entries) is not int or optimizer_entries <= 0:
        raise ValueError("resumable checkpoint must contain optimizer parameter states")
    remaining = (TARGET_TRANSITIONS - transitions) // batch
    if remaining <= 0:
        raise ValueError("checkpoint already reached the target")
    return {
        "remaining_iterations": remaining,
        "expected_num_timesteps": TARGET_TRANSITIONS,
        "expected_final_payload_iter": iteration + remaining - 1,
        "expected_adam_step": adam_step + remaining * ADAM_STEPS_PER_ITERATION,
    }


def training_argv(checkpoint: Path, plan: dict[str, int]) -> list[str]:
    return [
        str(PYTHON_EXE), "-m", "isaac_field", "resume", "--checkpoint", str(checkpoint.resolve()),
        "--num_envs", str(NUM_ENVS), "--max_iterations", str(plan["remaining_iterations"]),
        "--device", "cuda:0", "--seed", "69041", "--logger", "tensorboard",
        "--experiment_name", EXPERIMENT_NAME, "--run_timestamp", RUN_TIMESTAMP,
        "--resume_optimizer_state", "--viz", "none", "--live-state-file", str(LIVE_STATE_PATH.resolve()),
        "env.task.size=12.0", "env.scene.env_spacing=22.0", f"env.task_backend={TASK_BACKEND}",
    ]


def evaluation_argv(checkpoint: Path, output: Path) -> list[str]:
    """Run the established full-episode RSL evaluator with the same task backend."""
    return [
        str(PYTHON_EXE), str(GPU_EVALUATOR_PATH.resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--output", str(output.resolve()),
        "--task-backend", TASK_BACKEND,
        "--seed-start", str(EVAL_SEED_START),
        "--scenario-count", str(EVAL_SCENARIOS),
        "--goal-angle-deg", str(EVAL_GOAL_ANGLE_DEG),
    ]


def live_observer_argv() -> list[str]:
    return [
        str(PYTHON_EXE), str((TRAINING_ROOT / VIEWER_SCRIPT).resolve()),
        "--state-file", str(LIVE_STATE_PATH.resolve()),
        "--output", str((TRIAL_DIR / "live_viewer").resolve()),
        "--host", VIEWER_HOST, "--port", str(VIEWER_PORT),
        "--expected-run-id", TRIAL_DIR.name,
    ]


def _path_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else TRAINING_ROOT / path


def _verify_recovery_processes(
    records: list[dict[str, Any]],
    boot_time: datetime,
    process_creation_time: Callable[[int], datetime | None],
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("handoff processes must be an array")
    by_role: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("role") in by_role:
            raise ValueError("handoff process records must be unique objects keyed by role")
        by_role[record.get("role")] = record
    missing = sorted(set(PROCESS_ROLES) - set(by_role))
    extra = sorted(set(by_role) - set(PROCESS_ROLES))
    if missing or extra:
        raise ValueError(f"handoff process roles mismatch: missing={missing}, extra={extra}")

    outcomes: list[dict[str, Any]] = []
    for role in PROCESS_ROLES:
        record = by_role[role]
        pid = record.get("pid")
        if type(pid) is not int or pid <= 0 or record.get("state") != "stopped":
            raise ValueError(f"{role} must identify a positive stopped PID")
        created = record.get("created_at_utc")
        if created is not None and parse_utc(created) >= boot_time:
            raise ValueError(f"{role} creation time is not before the recorded reboot")
        last_seen = parse_utc(record.get("last_seen_at_utc"))
        if last_seen >= boot_time:
            raise ValueError(f"{role} was observed after the recorded reboot")

        current_created = process_creation_time(pid)
        if current_created is None:
            outcome = "missing"
        else:
            current_created = current_created.astimezone(timezone.utc)
            if current_created < boot_time:
                if created is not None and abs((current_created - parse_utc(created)).total_seconds()) <= _PROCESS_CTIME_TOLERANCE_S:
                    raise RuntimeError(f"original {role} identity is still running (PID {pid})")
                raise RuntimeError(f"PID {pid} for {role} has a pre-reboot creation time; identity is ambiguous")
            if created is not None and abs((current_created - parse_utc(created)).total_seconds()) <= _PROCESS_CTIME_TOLERANCE_S:
                raise RuntimeError(f"original {role} identity is still running (PID {pid})")
            outcome = "pid_reused_after_boot"
        outcomes.append({"role": role, "pid": pid, "original_identity": "stopped", "current_pid_state": outcome})
    return outcomes


def _process_creation_time_utc(pid: int) -> datetime | None:
    """Read a PID's creation time without signaling it; None means the PID is gone."""
    if os.name != "nt":
        raise OSError("recovery process identity checks require Windows process metadata")
    import ctypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                         ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: this PID does not exist.
            return None
        raise OSError(error, f"OpenProcess({pid}) failed")
    try:
        creation, exit_time, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                        ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise OSError(ctypes.get_last_error(), f"GetProcessTimes({pid}) failed")
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        unix_seconds = ticks / 10_000_000 - 11_644_473_600
        return datetime.fromtimestamp(unix_seconds, timezone.utc)
    finally:
        kernel32.CloseHandle(handle)


def _os_boot_time_utc() -> datetime:
    command = "$d=(Get-CimInstance Win32_OperatingSystem).LastBootUpTime; $d.ToUniversalTime().ToString('o')"
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return parse_utc(result.stdout.strip())


def _verify_handoff(*, expected_sha256: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    handoff = read_json(HANDOFF_PATH)
    if handoff.get("schema") != RECOVERY_HANDOFF_SCHEMA:
        raise ValueError(f"handoff schema must be {RECOVERY_HANDOFF_SCHEMA}")
    if handoff.get("source_trial") != A04_DIR.name:
        raise ValueError("handoff must identify a04 as the source trial")
    if handoff.get("source_protocol_sha256") != sha256_file(A04_DIR / "protocol.json"):
        raise ValueError("handoff source protocol SHA differs from frozen a04 protocol")
    if handoff.get("source_protocol_sha256") != EXPECTED_A04_PROTOCOL_SHA256:
        raise ValueError("source a04 protocol is not the audited frozen protocol")
    if expected_sha256 is not None and sha256_file(HANDOFF_PATH) != expected_sha256:
        raise RuntimeError("recovery handoff changed after prepare")
    boot_time = parse_utc(handoff.get("os", {}).get("last_boot_up_time_utc"))
    observed_at = parse_utc(handoff.get("observed_at_utc"))
    current_boot = _os_boot_time_utc()
    if abs((current_boot - boot_time).total_seconds()) > 3:
        raise RuntimeError("handoff reboot timestamp differs from the current OS boot time")
    if observed_at < boot_time:
        raise RuntimeError("handoff process observation predates the recorded reboot")

    status = read_json(A04_DIR / "run_status.json")
    if (status.get("status") != "interrupted" or status.get("phase") != "system_reboot"
            or status.get("protocol_sha256") != EXPECTED_A04_PROTOCOL_SHA256):
        raise RuntimeError("source a04 must remain interrupted by the recorded system reboot")
    started = status.get("started_at_utc")
    if started and parse_utc(started) >= boot_time:
        raise RuntimeError("source a04 started after the reboot recorded in handoff")

    evidence = handoff.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("handoff must cite hashed reboot and process evidence")
    evidence_by_path: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("invalid handoff evidence row")
        path = _path_from_root(item["path"])
        digest = item.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
            raise RuntimeError(f"handoff evidence missing or changed: {path}")
        evidence_by_path[str(path.resolve()).casefold()] = digest
    for record in handoff.get("processes", []):
        evidence_path = record.get("evidence_path") if isinstance(record, dict) else None
        path = _path_from_root(evidence_path) if isinstance(evidence_path, str) else None
        if path is None or str(path.resolve()).casefold() not in evidence_by_path:
            raise ValueError("each stopped process must reference a listed hashed evidence file")

    checkpoint_record = handoff.get("checkpoint")
    expected_checkpoint = {
        "iter": EXPECTED_CHECKPOINT_ITER,
        "num_timesteps": EXPECTED_CHECKPOINT_TRANSITIONS,
        "adam_step": EXPECTED_CHECKPOINT_ADAM_STEP,
        "sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    if not isinstance(checkpoint_record, dict):
        raise ValueError("handoff must identify the audited resume checkpoint")
    if Path(checkpoint_record.get("path", "")).resolve() != A04_CHECKPOINT.resolve():
        raise ValueError("handoff checkpoint path differs from the a04 candidate")
    for key, value in expected_checkpoint.items():
        if checkpoint_record.get(key) != value:
            raise ValueError(f"handoff checkpoint {key} differs from the audited candidate")

    outcomes = _verify_recovery_processes(handoff.get("processes"), boot_time, _process_creation_time_utc)
    return handoff, outcomes


def _production_rows(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        row["path"]: row["sha256"]
        for row in manifest.get("files", [])
        if row.get("origin") == "project" and not row.get("path", "").startswith("results/")
    }


def compare_production_manifests(parent: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    old, new = _production_rows(parent), _production_rows(current)
    old_paths, new_paths = set(old), set(new)
    added, removed = sorted(new_paths - old_paths), sorted(old_paths - new_paths)
    changed = sorted(path for path in old_paths & new_paths if old[path] != new[path])
    actual_deltas = {
        **{path: "added" for path in added},
        **{path: "removed" for path in removed},
        **{path: "modified" for path in changed},
    }
    expected_paths = set(AUTHORIZED_SOURCE_DELTAS)
    if set(actual_deltas) != expected_paths:
        raise RuntimeError(
            "production source drift from a04 does not match the exact a06 allowlist: "
            f"actual={actual_deltas}, expected={sorted(expected_paths)}"
        )
    authorized_rows: list[dict[str, Any]] = []
    for path in sorted(AUTHORIZED_SOURCE_DELTAS):
        expected = AUTHORIZED_SOURCE_DELTAS[path]
        if actual_deltas[path] != expected["kind"]:
            raise RuntimeError(f"production source delta kind changed for {path}: {actual_deltas[path]}")
        parent_sha = old.get(path)
        current_sha = new.get(path)
        if parent_sha != expected["parent_sha256"] or current_sha != expected["current_sha256"]:
            raise RuntimeError(
                f"production source SHA differs from the pinned a06 allowlist for {path}: "
                f"parent={parent_sha}, current={current_sha}"
            )
        authorized_rows.append({
            "path": path,
            "kind": expected["kind"],
            "parent_sha256": parent_sha,
            "current_sha256": current_sha,
            "reason": expected["reason"],
        })
    return {
        "status": "passed",
        "shared_file_count": len(old_paths & new_paths),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "authorized_deltas": authorized_rows,
        "source_delta_allowlist_exact": True,
    }


def _verify_a04_source_snapshot() -> dict[str, Any]:
    protocol_path = A04_DIR / "protocol.json"
    manifest_path = A04_DIR / "source_manifest.json"
    if sha256_file(protocol_path) != EXPECTED_A04_PROTOCOL_SHA256:
        raise RuntimeError("a04 frozen protocol SHA changed")
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    if manifest != protocol.get("source_snapshot"):
        raise RuntimeError("a04 source_manifest.json differs from its frozen protocol")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("a04 frozen source manifest is empty")
    failures: list[str] = []
    for row in rows:
        origin = row.get("origin", "project")
        source = (A04_DIR / row["path"]) if origin == "trial" else (TRAINING_ROOT / row["path"])
        snapshot = A04_DIR / row["snapshot_path"]
        if not snapshot.is_file() or sha256_file(snapshot) != row["sha256"]:
            failures.append(f"source snapshot mismatch: {row['path']}")
        if not source.is_file():
            failures.append(f"live source missing: {row['path']}")
            continue
        live_sha = sha256_file(source)
        if live_sha != row["sha256"]:
            expected = AUTHORIZED_SOURCE_DELTAS.get(row["path"])
            if expected is None or expected["kind"] != "modified" or live_sha != expected["current_sha256"]:
                failures.append(f"unauthorized live source mismatch: {row['path']}")
    if failures:
        raise RuntimeError(f"a04 frozen sources are incomplete or changed: {failures[:5]}")
    return {"status": "passed", "protocol_sha256": EXPECTED_A04_PROTOCOL_SHA256, "files_checked": len(rows)}


def _validate_a04_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, manifest = read_json(A04_DIR / "protocol.json"), read_json(A04_DIR / "source_manifest.json")
    _verify_a04_source_snapshot()
    status = read_json(A04_DIR / "run_status.json")
    if (status.get("status") != "interrupted" or status.get("phase") != "system_reboot"
            or status.get("protocol_sha256") != EXPECTED_A04_PROTOCOL_SHA256):
        raise RuntimeError("a04 must remain an interrupted/system_reboot source run")
    if status.get("interruption", {}).get("cpu_checkpoint_audit", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("a04 run status does not cite the selected audited checkpoint")

    audit = read_json(A04_AUDIT_PATH)
    source_integrity = audit.get("source_integrity", {})
    saved = audit.get("checkpoint_cpu", {})
    checkpoint_saved = saved.get("checkpoint", {})
    optimizer_saved = saved.get("optimizer", {})
    expected_integrity = {
        "status": "passed",
        "passed": True,
        "protocol_sha256": EXPECTED_A04_PROTOCOL_SHA256,
        "expected_protocol_sha256": EXPECTED_A04_PROTOCOL_SHA256,
        "protocol_manifest_equal": True,
        "manifest_file_count": 86,
        "live_files_matching_manifest": 86,
        "snapshot_files_matching_manifest": 86,
    }
    for key, value in expected_integrity.items():
        actual = source_integrity.get(key) if key != "status" else audit.get("status")
        if actual != value:
            raise RuntimeError(f"a04 source/checkpoint audit {key} is not {value!r}")
    if saved.get("status") != "passed" or checkpoint_saved.get("sha256_before_load") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("a04 checkpoint CPU audit did not pass for the expected bytes")
    if (Path(checkpoint_saved.get("path", "")).resolve() != A04_CHECKPOINT.resolve()
            or checkpoint_saved.get("sha256_after_load") != EXPECTED_CHECKPOINT_SHA256
            or checkpoint_saved.get("iter") != EXPECTED_CHECKPOINT_ITER
            or checkpoint_saved.get("num_timesteps") != EXPECTED_CHECKPOINT_TRANSITIONS
            or checkpoint_saved.get("adam_step") != EXPECTED_CHECKPOINT_ADAM_STEP
            or optimizer_saved.get("state_entries") != EXPECTED_OPTIMIZER_ENTRIES
            or checkpoint_saved.get("cpu_finite_tree") is not True):
        raise RuntimeError("a04 checkpoint CPU audit counters or finite-state result differ")
    if audit.get("training_log_vs_checkpoint", {}).get("logs_used_as_recovery_checkpoint") is not False:
        raise RuntimeError("recovery must use the saved checkpoint, not training logs")

    baseline = read_json(A02_DIR / "baseline_validation.json")
    baseline_config = protocol.get("training", {}).get("baseline", {})
    baseline_path = Path(baseline_config.get("checkpoint", ""))
    if (baseline.get("status") != "passed" or not baseline_path.is_file()
            or baseline_config.get("checkpoint_sha256") != EXPECTED_BASELINE_SHA256
            or baseline.get("checkpoint_sha256") != EXPECTED_BASELINE_SHA256
            or sha256_file(baseline_path) != EXPECTED_BASELINE_SHA256):
        raise RuntimeError("original a02 zero-step baseline validation or checkpoint bytes changed")
    return protocol, manifest, {"baseline": baseline, "checkpoint_audit": audit, "status": status}


def _audit_resume_checkpoint(checkpoint: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint missing: {checkpoint}")
    before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns, sha256_file(checkpoint))
    if before[2] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("resume checkpoint SHA differs from the audited a04 bytes")
    helper = A04._helper()
    payload, safe = helper.cpu_load_checkpoint(checkpoint)
    after = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns, sha256_file(checkpoint))
    if before != after:
        raise RuntimeError("resume checkpoint changed while its CPU audit was running")
    if safe.get("finite_tensor_tree") is not True:
        raise ValueError("resume checkpoint contains non-finite tensors")
    if (type(payload.get("iter")) is not int or payload["iter"] != EXPECTED_CHECKPOINT_ITER
            or type(payload.get("num_timesteps")) is not int
            or payload["num_timesteps"] != EXPECTED_CHECKPOINT_TRANSITIONS):
        raise ValueError("resume checkpoint iteration or cumulative transitions differ from the handoff")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("resume checkpoint lacks its Adam optimizer state")
    adam_step = A03._adam_step(payload)
    entries = A03._optimizer_state_size(payload)
    if adam_step != EXPECTED_CHECKPOINT_ADAM_STEP or entries != EXPECTED_OPTIMIZER_ENTRIES:
        raise ValueError("resume checkpoint Adam step or optimizer parameter count differs from the audit")
    if not isinstance(payload.get("actor_state_dict"), dict) or not isinstance(payload.get("critic_state_dict"), dict):
        raise ValueError("resume checkpoint lacks actor/critic state dictionaries")
    if not payload["actor_state_dict"] or not payload["critic_state_dict"]:
        raise ValueError("resume checkpoint actor/critic state dictionaries are empty")
    requirements = protocol["expected_checkpoint_contract_requirements"]
    contract = helper._assert_checkpoint_contract(payload.get("field_policy_contract"), requirements)
    baseline_path = Path(protocol["training"]["baseline"]["checkpoint"])
    baseline_payload, _ = helper.cpu_load_checkpoint(baseline_path)
    if contract["contract"] != baseline_payload.get("field_policy_contract"):
        raise ValueError("resume checkpoint policy/physics contract differs from the a02 baseline")
    report = {
        "schema": "isaac-field-a06-source-checkpoint-cpu-audit-v1",
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": after[0],
        "checkpoint_sha256": after[2],
        "iter": payload["iter"],
        "num_timesteps": payload["num_timesteps"],
        "adam_step": adam_step,
        "optimizer_state_entries": entries,
        "optimizer_finite": safe["finite_tensor_tree"],
        "field_policy_contract_sha256": contract["contract_sha256"],
        "complete_contract_matches_a02_baseline": True,
        "zero_step_baseline_sha256": sha256_file(baseline_path),
        "map_location": "cpu",
        "training_or_simulation_started": False,
    }
    return payload, report


def _snapshot_parent_evidence(manifest: dict[str, Any]) -> None:
    rows = manifest["files"]
    sources = {
        A04_DIR / "run_pipeline.py": "source_snapshot/inherited/a04/run_pipeline.py",
        A04_DIR / "protocol.json": "source_snapshot/inherited/a04/protocol.json",
        A04_DIR / "source_manifest.json": "source_snapshot/inherited/a04/source_manifest.json",
        A04_DIR / "run_status.json": "source_snapshot/inherited/a04/run_status.json",
        A04_DIR / "baseline_validation.json": "source_snapshot/inherited/a04/baseline_validation.json",
        A04_AUDIT_PATH: "source_snapshot/inherited/a04/checkpoint_audit.json",
        A02_DIR / "baseline_validation.json": "source_snapshot/inherited/a02/baseline_validation.json",
        NATIVE_REPORT_PATH: "source_snapshot/inherited/native/native_drive_final_a01.json",
        NATIVE_PROCESS_PATH: "source_snapshot/inherited/native/native_drive_final_a01-process.json",
        NATIVE_LOG_PATH: "source_snapshot/inherited/native/native_drive_final_a01.log",
        GPU_MATH_PATH: "source_snapshot/inherited/native/cuda_math_final.json",
        DRIVE_CHECK_PATH: "source_snapshot/inherited/native/drive_validation.json",
        CPU_GATE_PATH: "source_snapshot/inherited/native/cpu_gate_after_drive.json",
    }
    for source, relative in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"required source evidence is missing: {source}")
        destination = TRIAL_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({"path": relative, "origin": "trial", "snapshot_path": relative, "sha256": sha256_file(source)})
    manifest["builder"] += "; pinned a04/a02 lineage plus the GPU-batched CPU/CUDA/native 16-step validation evidence"
    rows.sort(key=lambda row: (row["origin"], row["path"]))


def _snapshot_trial_local_sources(manifest: dict[str, Any]) -> None:
    trial_sources = (GPU_EVALUATOR_PATH, TRIAL_DIR / "verify_first_checkpoint.py")
    for source in trial_sources:
        if not source.is_file():
            raise FileNotFoundError(f"required a06 trial source is missing: {source}")
        relative = source.name
        snapshot_rel = (Path("source_snapshot") / relative).as_posix()
        shutil.copyfile(source, TRIAL_DIR / snapshot_rel)
        manifest["files"].append({
            "path": relative,
            "origin": "trial",
            "snapshot_path": snapshot_rel,
            "sha256": sha256_file(source),
        })
    manifest["files"].sort(key=lambda row: (row["origin"], row["path"]))
    manifest["builder"] += "; pinned the a06-only GPU evaluator and first-checkpoint CPU verifier"


def _verify_native_gpu_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    paths = (NATIVE_REPORT_PATH, NATIVE_PROCESS_PATH, NATIVE_LOG_PATH,
             GPU_MATH_PATH, DRIVE_CHECK_PATH, CPU_GATE_PATH)
    evidence_hashes: dict[str, str] = {}
    for path in paths:
        expected = EXPECTED_NATIVE_EVIDENCE_SHA256[path.name]
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"pinned GPU/native validation evidence is missing or changed: {path}")
        evidence_hashes[path.name] = expected

    report, process = read_json(NATIVE_REPORT_PATH), read_json(NATIVE_PROCESS_PATH)
    required_report = {
        "status": "passed",
        "training_started": False,
        "policy_loaded": False,
        "optimizer_started": False,
        "fps_benchmark_started": False,
        "num_envs": NUM_ENVS,
        "steps": 16,
        "camera_positions_checked": 2,
        "partial_reset_checked": True,
        "old_observation_immutable": True,
        "device_frames_retained": True,
    }
    for key, expected in required_report.items():
        if report.get(key) != expected:
            raise RuntimeError(f"native 16-step report {key}={report.get(key)!r}, expected {expected!r}")
    physics = report.get("physics", {})
    if (physics.get("task_backend") != TASK_BACKEND or physics.get("device") != "cuda:0"
            or physics.get("gpu_dynamics") is not True):
        raise RuntimeError("native 16-step report did not verify torch_batched on CUDA physics")
    if process.get("returncode") != 0 or process.get("timed_out") is not False:
        raise RuntimeError("native 16-step process did not exit successfully")
    if process.get("log_sha256") != evidence_hashes[NATIVE_LOG_PATH.name]:
        raise RuntimeError("native 16-step process does not cite the pinned log bytes")

    process_sources = {
        str(path).replace("\\", "/"): digest
        for path, digest in process.get("source_sha256", {}).items()
    }
    report_before = report.get("source_sha256", {})
    report_after = report.get("source_sha256_after", {})
    snapshot_root = Path(process.get("source_snapshot", ""))
    source_rows: list[dict[str, str]] = []
    frozen_production = _production_rows(manifest)
    for path, expected in sorted(AUTHORIZED_SOURCE_DELTAS.items()):
        current_sha = expected["current_sha256"]
        if frozen_production.get(path) != current_sha:
            raise RuntimeError(f"native gate source SHA differs from the a06 source manifest: {path}")
        if process_sources.get(path) != current_sha:
            raise RuntimeError(f"native 16-step source record does not match current source: {path}")
        if path in report_before and report_before[path] != current_sha:
            raise RuntimeError(f"native report pre-run source SHA mismatch: {path}")
        if path in report_after and report_after[path] != current_sha:
            raise RuntimeError(f"native report post-run source SHA mismatch: {path}")
        snapshot_path = snapshot_root.joinpath(*path.split("/"))
        if not snapshot_path.is_file() or sha256_file(snapshot_path) != current_sha:
            raise RuntimeError(f"native source snapshot does not match current source: {path}")
        source_rows.append({"path": path, "sha256": current_sha, "native_source_snapshot": str(snapshot_path)})

    return {
        "status": "passed",
        "report_path": str(NATIVE_REPORT_PATH.resolve()),
        "process_path": str(NATIVE_PROCESS_PATH.resolve()),
        "log_path": str(NATIVE_LOG_PATH.resolve()),
        "evidence_sha256": evidence_hashes,
        "returncode": process["returncode"],
        "timed_out": process["timed_out"],
        "steps": report["steps"],
        "num_envs": report["num_envs"],
        "task_backend": physics["task_backend"],
        "device": physics["device"],
        "gpu_dynamics": physics["gpu_dynamics"],
        "partial_reset_checked": report["partial_reset_checked"],
        "old_observation_immutable": report["old_observation_immutable"],
        "source_sha256": source_rows,
        "gate_evidence_hashes": evidence_hashes,
        "training_started": False,
    }


def prepare(checkpoint_argument: str | Path) -> dict[str, Any]:
    if PROTOCOL_PATH.exists() or MANIFEST_PATH.exists() or SOURCE_SNAPSHOT.exists() or LOCK_PATH.exists():
        raise FileExistsError(f"refusing to freeze over existing a06 data: {TRIAL_DIR}")
    handoff_sha = sha256_file(HANDOFF_PATH)
    handoff, process_outcomes = _verify_handoff(expected_sha256=handoff_sha)
    protocol_a04, parent_manifest, parent_evidence = _validate_a04_parent()
    viewer_source = TRAINING_ROOT / VIEWER_SCRIPT
    if not viewer_source.is_file():
        raise FileNotFoundError(f"the required lightweight web observer is not ready: {viewer_source}")
    checkpoint = A04_CHECKPOINT.resolve()
    if Path(checkpoint_argument).resolve() != checkpoint:
        raise RuntimeError(f"--checkpoint must select the audited a04 checkpoint: {checkpoint}")
    checkpoint_record = handoff["checkpoint"]
    payload, checkpoint_audit = _audit_resume_checkpoint(checkpoint, protocol_a04)
    if checkpoint_record["iter"] != payload["iter"] or checkpoint_record["num_timesteps"] != payload["num_timesteps"]:
        raise RuntimeError("handoff checkpoint counters differ from the CPU-loaded payload")
    plan = resume_plan(payload["iter"], payload["num_timesteps"], EXPECTED_CHECKPOINT_ADAM_STEP,
                       EXPECTED_OPTIMIZER_ENTRIES)

    helper = A04._helper()
    manifest = helper.build_source_manifest()
    native_gate = _verify_native_gpu_gate(manifest)
    production_check = compare_production_manifests(parent_manifest, manifest)
    _snapshot_parent_evidence(manifest)
    _snapshot_trial_local_sources(manifest)
    requirements = helper.expected_contract_requirements()
    protocol = copy.deepcopy(protocol_a04)
    protocol.update({
        "schema": "isaac-field-gpu-batched-recovery-protocol-v1",
        "status": "prepared",
        "created_at_utc": utc_now(),
        "trial_id": TRIAL_DIR.name,
        "objective": "resume the verified a04 checkpoint at 26112 cumulative transitions to 196608 with the explicit torch_batched task backend, then paired a02-baseline evaluation on the same 16 development seeds and backend",
        "git_identity": helper.repository_identity(),
        "runtime_versions": helper.installed_versions(),
        "source_snapshot": manifest,
        "source_parent": {
            "trial": A04_DIR.name,
            "protocol_sha256": sha256_file(A04_DIR / "protocol.json"),
            "source_manifest_sha256": sha256_file(A04_DIR / "source_manifest.json"),
            "run_status_sha256": sha256_file(A04_DIR / "run_status.json"),
            "checkpoint_audit_path": str(A04_AUDIT_PATH.resolve()),
            "checkpoint_audit_sha256": sha256_file(A04_AUDIT_PATH),
            "a04_source_snapshot": _verify_a04_source_snapshot(),
            "current_production_manifest_comparison": production_check,
            "native_gpu_batch_gate": native_gate,
            "baseline_validation_sha256": sha256_file(A02_DIR / "baseline_validation.json"),
        },
        "source_handoff": {"path": str(HANDOFF_PATH.resolve()), "sha256": handoff_sha,
                            "schema": RECOVERY_HANDOFF_SCHEMA, "process_outcomes": process_outcomes},
        "expected_checkpoint_contract_requirements": requirements,
        "live_observer": {
            "state_file": str(LIVE_STATE_PATH.resolve()),
            "expected_run_id": TRIAL_DIR.name,
            "reader_argv": live_observer_argv(),
            "viewer_started_by_pipeline": False,
            "simulator_or_policy_instance": False,
            "source_sha256": sha256_file(viewer_source),
        },
        "completion_watcher": {"script": "scripts/watch_training_completion.py", "interval_seconds": 30,
                               "delivery_deduplicated": True, "started_by_pipeline": False},
        "physx_scene_query_support": {"enabled": False, "reason": "headless official CLI matches the a02/a04 task configuration"},
    })
    train = protocol["training"]
    train.update({
        "max_iterations": plan["remaining_iterations"],
        "target_transitions": TARGET_TRANSITIONS,
        "remaining_iterations": plan["remaining_iterations"],
        "transition_arithmetic": f"{payload['num_timesteps']} + {plan['remaining_iterations']} * {NUM_ENVS} * {ROLLOUT_STEPS} = {TARGET_TRANSITIONS}",
        "optimizer_resume": True,
        "task_backend": TASK_BACKEND,
        "headless": {"selected": True, "visualizer": "none", "reason": "one headless training Kit; the web observer reads live_state.json without another simulator"},
    })
    train.pop("gui", None)
    train["resume"] = {
        "argv_template": training_argv(checkpoint, plan),
        "run_dir": str(TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / RUN_TIMESTAMP),
        "checkpoint_selection": "explicit a04 model_48.pt selected by the reboot CPU audit and identity-checked handoff",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_iter": payload["iter"],
        "checkpoint_num_timesteps": payload["num_timesteps"],
        "checkpoint_adam_step": EXPECTED_CHECKPOINT_ADAM_STEP,
        "checkpoint_optimizer_state_entries": EXPECTED_OPTIMIZER_ENTRIES,
        **plan,
        "expected_iteration_filename_not_hardcoded": True,
        "resume_optimizer_state": True,
    }
    train["baseline"].update({"reused_from_trial": A02_DIR.name, "argv": [], "retrained": False,
                              "checkpoint_sha256": EXPECTED_BASELINE_SHA256})
    protocol["evaluation"]["outputs"] = {"baseline": str(TRIAL_DIR / "eval_baseline"),
                                           "final": str(TRIAL_DIR / "eval_final")}
    protocol["evaluation"].update({
        "script": str(GPU_EVALUATOR_PATH.resolve()),
        "task_backend": TASK_BACKEND,
        "same_task_backend_for_baseline_and_final": True,
        "backend_override": ["--task-backend", TASK_BACKEND],
        "seed_start": EVAL_SEED_START,
        "scenario_count": EVAL_SCENARIOS,
        "scenario_seeds": list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SCENARIOS)),
    })
    protocol["export"]["output"] = str(TRIAL_DIR / "final_policy_torchscript.pt")
    protocol["output_policy"] = "a06 checkpoints, live state, and reports are rooted in this recovery trial; the byte-identical a02 baseline is reused and both paired evaluations use torch_batched."
    protocol.setdefault("subprocess", {})["cwd"] = str(TRIAL_DIR)

    write_json_atomic(MANIFEST_PATH, manifest)
    write_json_atomic(PROTOCOL_PATH, protocol)
    status = {
        "schema": "isaac-field-one-shot-run-status-v1",
        "status": "prepared",
        "phase": "prepared",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "parent_pid": os.getpid(),
        "children": [],
        "current_stage": None,
        "source_trial": A04_DIR.name,
        "resume_checkpoint": train["resume"],
        "export": {"status": "pending_optional"},
    }
    write_json_atomic(STATUS_PATH, status)
    write_json_atomic(TRIAL_DIR / "source_checkpoint_cpu.json", checkpoint_audit)
    baseline_audit = helper.audit_baseline(protocol, status)
    status = read_json(STATUS_PATH)
    status.update({"status": "prepared", "phase": "prepared", "current_stage": None,
                   "baseline_checkpoint": {"path": baseline_audit["checkpoint"],
                                           "sha256": baseline_audit["checkpoint_sha256"],
                                           "audit": "passed", "reused": True}})
    helper._stage_status(status, status)
    result = {
        "schema": "isaac-field-a06-gpu-batched-recovery-prepare-v1",
        "status": "prepared",
        "trial_dir": str(TRIAL_DIR),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_file_count": len(manifest["files"]),
        "resume_checkpoint": train["resume"],
        "remaining_iterations": plan["remaining_iterations"],
        "target_transitions": TARGET_TRANSITIONS,
        "source_snapshot": production_check,
        "task_backend": TASK_BACKEND,
        "paired_evaluation_backend": TASK_BACKEND,
        "native_processes_started": False,
        "training_started": False,
        "viewer_started": False,
        "kit_imported": False,
    }
    write_json_atomic(TRIAL_DIR / "prepare.json", result)
    return result


def _validate_frozen_parent(protocol: dict[str, Any]) -> None:
    parent = protocol.get("source_parent", {})
    for path, expected in (
        (A04_DIR / "protocol.json", parent.get("protocol_sha256")),
        (A04_DIR / "source_manifest.json", parent.get("source_manifest_sha256")),
        (A04_DIR / "run_status.json", parent.get("run_status_sha256")),
        (A04_AUDIT_PATH, parent.get("checkpoint_audit_sha256")),
        (A02_DIR / "baseline_validation.json", parent.get("baseline_validation_sha256")),
    ):
        if not expected or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen parent evidence changed or is missing: {path}")
    a04_protocol, a04_manifest, _ = _validate_a04_parent()
    if protocol.get("source_snapshot") != read_json(MANIFEST_PATH):
        raise RuntimeError("a06 source_manifest.json differs from the manifest frozen in protocol")
    helper = A04._helper()
    snapshot = helper._verify_manifest(protocol["source_snapshot"])
    if not snapshot["valid"]:
        raise RuntimeError(f"a06 source snapshot mismatch: {snapshot['failures'][:5]}")
    current_production = {"files": protocol["source_snapshot"].get("files", [])}
    current_comparison = compare_production_manifests(a04_manifest, current_production)
    if current_comparison != parent.get("current_production_manifest_comparison"):
        raise RuntimeError("a06 production-source delta audit differs from the frozen protocol")
    native_gate = _verify_native_gpu_gate(protocol["source_snapshot"])
    if native_gate != parent.get("native_gpu_batch_gate"):
        raise RuntimeError("native GPU 16-step evidence or source hashes differ from the frozen protocol")
    if protocol.get("source_parent", {}).get("protocol_sha256") != sha256_file(A04_DIR / "protocol.json"):
        raise RuntimeError("a04 parent protocol changed after prepare")
    if helper.repository_identity() != protocol.get("git_identity") or helper.installed_versions() != protocol.get("runtime_versions"):
        raise RuntimeError("repository/Isaac Lab identity or Python runtime versions changed after prepare")
    if a04_protocol.get("status") not in ("prepared", "running", "interrupted"):
        raise RuntimeError("a04 protocol no longer represents its prepared recovery lineage")


def preflight(*, write_report: bool = True) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not STATUS_PATH.is_file():
        raise FileNotFoundError("run --prepare --checkpoint <audited-a04-checkpoint> first")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    status, protocol = read_json(STATUS_PATH), read_json(PROTOCOL_PATH)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("a06 is not an unchanged prepared run")
    handoff_sha = protocol.get("source_handoff", {}).get("sha256")
    handoff, process_outcomes = _verify_handoff(expected_sha256=handoff_sha)
    if handoff_sha != sha256_file(HANDOFF_PATH):
        raise RuntimeError("recovery handoff changed after prepare")
    if protocol.get("source_handoff", {}).get("process_outcomes") != process_outcomes:
        raise RuntimeError("stopped-process identity evidence changed after prepare")
    if protocol.get("live_observer", {}).get("source_sha256") != sha256_file(TRAINING_ROOT / VIEWER_SCRIPT):
        raise RuntimeError("frozen web observer source changed after prepare")
    if (protocol.get("training", {}).get("task_backend") != TASK_BACKEND
            or protocol.get("evaluation", {}).get("task_backend") != TASK_BACKEND
            or protocol.get("evaluation", {}).get("same_task_backend_for_baseline_and_final") is not True):
        raise RuntimeError("a06 training and paired evaluation must both use torch_batched")
    _validate_frozen_parent(protocol)
    resume = protocol["training"]["resume"]
    checkpoint = Path(resume["checkpoint"])
    if checkpoint.resolve() != A04_CHECKPOINT.resolve() or sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("selected a04 checkpoint is missing or its source bytes changed")
    _, source_checkpoint_audit = _audit_resume_checkpoint(checkpoint, protocol)
    if (source_checkpoint_audit["iter"] != resume["checkpoint_iter"]
            or source_checkpoint_audit["num_timesteps"] != resume["checkpoint_num_timesteps"]
            or source_checkpoint_audit["adam_step"] != resume["checkpoint_adam_step"]
            or source_checkpoint_audit["optimizer_state_entries"] != resume["checkpoint_optimizer_state_entries"]):
        raise RuntimeError("selected a04 checkpoint counters changed after prepare")
    if (resume.get("argv_template") != training_argv(checkpoint, resume)
            or f"env.task_backend={TASK_BACKEND}" not in resume.get("argv_template", [])):
        raise RuntimeError("official RSL resume argv must explicitly select torch_batched")
    baseline = protocol["training"]["baseline"]
    baseline_path = Path(baseline["checkpoint"])
    if baseline.get("checkpoint_sha256") != EXPECTED_BASELINE_SHA256 or sha256_file(baseline_path) != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("reused a02 zero-step baseline bytes changed")
    planned = [Path(resume["run_dir"]), TRIAL_DIR / "eval_baseline", TRIAL_DIR / "eval_final", LIVE_STATE_PATH]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"a06 output paths already exist: {existing}")
    baseline_evaluation_argv = evaluation_argv(baseline_path, TRIAL_DIR / "eval_baseline")
    final_evaluation_argv_template = evaluation_argv(
        Path("<audited-final-checkpoint>"), TRIAL_DIR / "eval_final",
    )
    report = {
        "schema": "isaac-field-a06-gpu-batched-recovery-preflight-v1",
        "status": "passed",
        "checked_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "a04_parent_integrity": "passed",
        "handoff_sha256": handoff_sha,
        "old_processes_stopped": True,
        "process_outcomes": process_outcomes,
        "selected_checkpoint": resume,
        "checkpoint_cpu_audit": source_checkpoint_audit,
        "remaining_iterations": resume["remaining_iterations"],
        "target_transitions": TARGET_TRANSITIONS,
        "training_argv": resume["argv_template"],
        "task_backend": TASK_BACKEND,
        "paired_evaluation_backend": TASK_BACKEND,
        "baseline_evaluation_argv": baseline_evaluation_argv,
        "final_evaluation_argv_template": final_evaluation_argv_template,
        "visualizer": "none",
        "headless_environment": {"HEADLESS": "1", "LIVESTREAM": "0", "XR": "0"},
        "live_observer": protocol["live_observer"],
        "planned_outputs_absent": True,
        "native_processes_started": False,
        "viewer_started": False,
        "training_started": False,
    }
    if write_report:
        write_json_atomic(PREFLIGHT_PATH, report)
    return report


def _headless_child_environment(environment: dict[str, str], *, training: bool) -> dict[str, str]:
    result = dict(environment)
    result.update({"HEADLESS": "1", "LIVESTREAM": "0", "XR": "0"})
    if training:
        result["ISAAC_FIELD_LIVE_STATE_RUN_ID"] = TRIAL_DIR.name
    else:
        result.pop("ISAAC_FIELD_LIVE_STATE_RUN_ID", None)
    return result


A04.preflight = preflight


def execute_pipeline(*, export: bool = False) -> dict[str, Any]:
    # Keep a04/a03's tested lock, power request, stage bookkeeping, final CPU
    # audit, and paired evaluation. a06 injects its source/backend preflight and
    # the per-trial paired evaluator; no alternate trainer is introduced.
    _configure_reused_pipeline()
    A03._evaluation_argv = evaluation_argv
    A04.preflight = preflight
    A04._headless_child_environment = _headless_child_environment
    return A04.execute_pipeline(export=export)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="freeze a06 from the verified reboot handoff; CPU only")
    actions.add_argument("--preflight", action="store_true", help="verify frozen a06/a04 sources and checkpoint; CPU only")
    actions.add_argument("--dry-run", action="store_true", help="print the exact launch/evaluation plan; CPU only")
    parser.add_argument("--checkpoint", help="explicit audited a04 checkpoint required by --prepare")
    parser.add_argument("--export", action="store_true", help="also attempt optional TorchScript export")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if not args.checkpoint:
                parser.error("--prepare requires --checkpoint")
            result = prepare(args.checkpoint)
        elif args.preflight:
            result = preflight()
        elif args.dry_run:
            result = preflight()
            result["ordered_stages"] = [
                "resume_training_cli --viz none",
                "audit final transition and Adam counts",
                "evaluate_baseline on 16 seeds 87010-87025",
                "evaluate_final on the same 16 seeds",
                "paired development comparison",
            ]
        else:
            result = execute_pipeline(export=args.export)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"a06 recovery pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
