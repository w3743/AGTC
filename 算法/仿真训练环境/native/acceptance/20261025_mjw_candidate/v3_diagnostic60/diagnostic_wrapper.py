#!/usr/bin/env python3
"""External diagnostic trampoline; does not modify native training or physics sources."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

NATIVE = Path(__file__).resolve().parents[2]
REPO = NATIVE.parents[2]
PROTO = NATIVE / "proto"
TRAIN = NATIVE / "train" / "train_wsl.py"
if str(PROTO) not in sys.path:
    sys.path.insert(0, str(PROTO))
if str(TRAIN.parent) not in sys.path:
    sys.path.insert(0, str(TRAIN.parent))

TRAIN_ARGS = list(sys.argv[1:])
def _arg_value(name: str, default: str | None = None) -> str | None:
    for index, item in enumerate(TRAIN_ARGS):
        if item == name and index + 1 < len(TRAIN_ARGS):
            return TRAIN_ARGS[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default
SEED = int(_arg_value("--seed", "0"))
MAX_ITERATIONS = int(_arg_value("--iterations", "100"))
EXPECTED = {
    "--num-envs": "64", "--num-steps": "32", "--seed": "0",
    "--policy": "lstm", "--lstm-hidden": "512", "--physics": "newton_xpbd",
    "--task-backend": "fast_stop_v2", "--lidar": "warp", "--feedback": "vqf",
    "--depth": "analytic", "--detections": "analytic", "--sim-dt": "0.001",
    "--decimation": "50", "--nonfinite-limit": "0",
}
for option, expected_value in EXPECTED.items():
    if _arg_value(option) != expected_value:
        raise SystemExit(f"diagnostic recipe mismatch: {option} must be {expected_value}")
if SEED != 0 or not 1 <= MAX_ITERATIONS <= 60:
    raise SystemExit("diagnostic accepts seed 0 and at most 60 iterations")
if "--detections-warp" not in TRAIN_ARGS:
    raise SystemExit("diagnostic recipe requires --detections-warp")
try:
    OUT = Path(TRAIN_ARGS[TRAIN_ARGS.index("--out") + 1]).resolve()
except (ValueError, IndexError):
    raise SystemExit("diagnostic wrapper requires --out <new run directory>")
DIAG_DIR = OUT / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                               text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _versions() -> dict[str, str | None]:
    names = ("numpy", "torch", "warp-lang", "newton", "mujoco", "vqf")
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


source_files = sorted(
    list((NATIVE / "train").glob("*.py"))
    + list((NATIVE / "train").glob("PROTOCOL_NATIVE_V1.1.*"))
    + list(PROTO.glob("*.py"))
    + list(PROTO.glob("*.json"))
)
source_rows = []
for path in source_files:
    raw = path.read_bytes()
    source_rows.append({"path": path.relative_to(REPO).as_posix(), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
source_manifest = json.dumps(source_rows, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
wrapper_raw_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
wrapper_argv = list(sys.argv)
trainer_argv = [str(TRAIN), *TRAIN_ARGS]
manifest = {
    "schema": "native-v3-nonfinite-diagnostic-v1",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "run_out": str(OUT),
    "wrapper_argv": wrapper_argv,
    "trainer_argv": trainer_argv,
    "seed": SEED,
    "max_iterations": MAX_ITERATIONS,
    "capture_policy_frames": "last 2 full 64-world policy-step frames",
    "capture_point": "Newton backend step return and read(), before NativeFieldEnv auto-reset",
    "device_note": "training process requests cuda; wrapper itself does not initialize CUDA",
    "python": sys.version,
    "platform": platform.platform(),
    "versions": _versions(),
    "git_commit": _git("rev-parse", "HEAD"),
    "git_branch": _git("branch", "--show-current"),
    "git_dirty_scope": "not measured; raw listed source bytes are recorded below",
    "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
    "diagnostic_wrapper_sha256": wrapper_raw_sha256,
    "source_files": source_rows,
}
manifest_path = DIAG_DIR / "diagnostic_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")

from newton_backend import NewtonXPBDGBackend  # noqa: E402

_HISTORY: collections.deque[dict[str, Any]] = collections.deque(maxlen=2)
_ORIGINAL_STEP = NewtonXPBDGBackend.step
_ORIGINAL_READ = NewtonXPBDGBackend.read


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value).copy()
    except Exception:
        return None


def _snapshot(backend: NewtonXPBDGBackend, actions: Any = None) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for state_name in ("state_0", "state_1"):
        state = getattr(backend, state_name)
        for field in ("body_q", "body_qd", "body_f"):
            snap[f"{state_name}.{field}"] = _array(getattr(state, field, None))
    for field in ("joint_target_q", "joint_target_qd", "joint_f"):
        snap[f"control.{field}"] = _array(getattr(backend.control, field, None))
    for field in ("joint_q", "joint_qd"):
        snap[field] = _array(getattr(backend, field, None))
    for field in ("track_command", "gimbal_target", "wheel_target_qd", "last_effort", "step_count"):
        snap[field] = _array(getattr(backend, field, None))
    snap["pi.integral"] = _array(getattr(getattr(backend, "pi", None), "integral", None))
    if actions is not None:
        snap["actions"] = np.asarray(actions).copy()
    return snap


def _nonfinite_fields(snapshot: dict[str, Any]) -> dict[str, list[int]]:
    bad: dict[str, list[int]] = {}
    for key, value in snapshot.items():
        if not isinstance(value, np.ndarray) or not value.size:
            continue
        if not np.issubdtype(value.dtype, np.number):
            continue
        try:
            mask = ~np.isfinite(value)
        except TypeError:
            continue
        if not mask.any():
            continue
        if value.ndim == 0:
            rows = [0]
        else:
            rows = np.flatnonzero(mask.reshape(value.shape[0], -1).any(axis=1)).astype(int).tolist()
        bad[key] = rows
    return bad


def _contacts_snapshot(backend: NewtonXPBDGBackend) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    contacts = backend.contacts
    count = _array(getattr(contacts, "rigid_contact_count", None))
    if count is not None:
        result["contacts.rigid_contact_count"] = count
    n = int(np.sum(count)) if count is not None else 0
    names = (
        "rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0",
        "rigid_contact_point1", "rigid_contact_normal", "rigid_contact_surface_velocity",
        "rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_margin0",
        "rigid_contact_margin1", "rigid_contact_stiffness", "rigid_contact_damping",
        "rigid_contact_friction", "rigid_contact_tids",
    )
    for name in names:
        value = _array(getattr(contacts, name, None))
        if value is not None:
            if value.ndim and n and value.shape[0] >= n:
                value = value[:n].copy()
            result[f"contacts.{name}"] = value
    return result


def _model_snapshot(backend: NewtonXPBDGBackend) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in ("body_world", "joint_world", "joint_q_start", "joint_qd_start",
                 "joint_target_q_start", "shape_world"):
        value = _array(getattr(backend.model, name, None))
        if value is not None:
            result[f"model.{name}"] = value
    for name in ("body_label", "joint_label"):
        try:
            result[f"model.{name}"] = np.asarray([str(x) for x in getattr(backend.model, name)], dtype="U")
        except Exception:
            pass
    return result


def _flatten(prefix: str, value: Any, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}__{key}" if prefix else str(key), item, arrays, metadata)
    elif isinstance(value, np.ndarray):
        key = prefix.replace(".", "__")
        arrays[key] = value
        metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    elif value is None or isinstance(value, (str, int, float, bool)):
        metadata[prefix] = value
    else:
        metadata[prefix] = repr(value)


def _save_event(backend: NewtonXPBDGBackend, reason: str, bad: dict[str, list[int]],
                read_values: dict[str, Any] | None = None, exception: str | None = None) -> None:
    if getattr(backend, "_diag_event_saved", False):
        return
    backend._diag_event_saved = True
    arrays: dict[str, np.ndarray] = {}
    descriptions: dict[str, Any] = {}
    frames = list(_HISTORY)
    for index, frame in enumerate(frames):
        _flatten(f"frame{index}", frame, arrays, descriptions)
    _flatten("contacts", _contacts_snapshot(backend), arrays, descriptions)
    _flatten("model", _model_snapshot(backend), arrays, descriptions)
    if read_values is not None:
        _flatten("post_read", read_values, arrays, descriptions)
    identity = backend.identity()
    try:
        task_contract = json.loads((OUT / "observation_contract.json").read_text(encoding="utf-8"))
        task_identity_sha256 = task_contract.get("task_identity_sha256")
    except (OSError, json.JSONDecodeError):
        task_identity_sha256 = None
    event = {
        "schema": "native-v3-nonfinite-event-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": reason,
        "nonfinite_rows_by_field": bad,
        "world_ids": sorted({int(world) for key, rows in bad.items()
                              if key.startswith(("state_0.body_", "state_1.body_"))
                              for row in rows
                              for world in [int(backend.model.body_world.numpy()[row])]}),
        "backend_call_index_zero_based": getattr(backend, "_diag_policy_call_index", None),
        "history_frames": [
            {"index": i, "call_index_zero_based": frame.get("call_index_zero_based"),
             "captured_at_utc": frame.get("captured_at_utc"),
             "action_shape": list(frame["pre"].get("actions", np.zeros((0,))).shape)}
            for i, frame in enumerate(frames)
        ],
        "backend_identity": identity,
        "task_identity_sha256": task_identity_sha256,
        "diagnostic_manifest": str(manifest_path),
        "array_descriptions": descriptions,
        "exception": exception,
    }
    path = DIAG_DIR / "nonfinite_repro.npz"
    np.savez_compressed(path, **arrays)
    event["npz_path"] = str(path)
    event_path = DIAG_DIR / "nonfinite_event.json"
    event_path.write_text(json.dumps(event, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8", newline="\n")
    print("NONFINITE_DIAGNOSTIC " + json.dumps({"npz": str(path), "event": str(event_path),
          "reason": reason, "bad": bad}, ensure_ascii=False), flush=True)


def _instrumented_step(self: NewtonXPBDGBackend, actions: np.ndarray) -> None:
    call_index = int(getattr(self, "_diag_policy_call_index", 0))
    self._diag_policy_call_index = call_index + 1
    frame: dict[str, Any] = {
        "call_index_zero_based": call_index,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pre": _snapshot(self, actions),
    }
    _HISTORY.append(frame)
    self._diag_current_frame = frame
    self._diag_in_backend_step = True
    try:
        _ORIGINAL_STEP(self, actions)
    except BaseException as exc:
        self._diag_in_backend_step = False
        frame["exception"] = f"{type(exc).__name__}: {exc}"
        frame["post"] = _snapshot(self)
        bad = _nonfinite_fields(frame["post"])
        bad.update({f"internal_read.{key}": rows for key, rows
                    in frame.get("internal_read_nonfinite_fields", {}).items()})
        _save_event(self, "backend_step_exception", bad,
                    frame.get("internal_read_values"), exception=frame["exception"])
        raise
    finally:
        self._diag_in_backend_step = False
    frame["post"] = _snapshot(self)
    bad = _nonfinite_fields(frame["post"])
    bad.update({f"internal_read.{key}": rows for key, rows
                in frame.get("internal_read_nonfinite_fields", {}).items()})
    if bad:
        _save_event(self, "nonfinite_backend_snapshot_after_policy_step", bad,
                    frame.get("internal_read_values"))


def _instrumented_read(self: NewtonXPBDGBackend) -> dict[str, np.ndarray]:
    values = _ORIGINAL_READ(self)
    read_bad = _nonfinite_fields(values)
    frame = getattr(self, "_diag_current_frame", None)
    if getattr(self, "_diag_in_backend_step", False):
        if read_bad and frame is not None:
            frame["internal_read_nonfinite_fields"] = read_bad
            frame["internal_read_values"] = {name: np.asarray(value).copy()
                                             for name, value in values.items()
                                             if isinstance(value, np.ndarray)}
    elif read_bad:
        if frame is not None:
            frame["post_read"] = {name: np.asarray(value).copy() for name, value in values.items()
                                  if isinstance(value, np.ndarray)}
        bad = _nonfinite_fields(frame.get("post", {})) if frame else {}
        bad.update({f"read.{key}": rows for key, rows in read_bad.items()})
        _save_event(self, "nonfinite_backend_read_before_env_reset", bad, values)
    return values


NewtonXPBDGBackend.step = _instrumented_step
NewtonXPBDGBackend.read = _instrumented_read
print("DIAGNOSTIC_MANIFEST " + str(manifest_path), flush=True)
print("DIAGNOSTIC_TRAIN_ARGV " + json.dumps(trainer_argv, ensure_ascii=False), flush=True)
sys.argv = trainer_argv
try:
    runpy.run_path(str(TRAIN), run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 0
    (DIAG_DIR / "wrapper_exit.json").write_text(
        json.dumps({"exit_code": code, "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
                   indent=2) + "\n", encoding="utf-8", newline="\n")
    raise
