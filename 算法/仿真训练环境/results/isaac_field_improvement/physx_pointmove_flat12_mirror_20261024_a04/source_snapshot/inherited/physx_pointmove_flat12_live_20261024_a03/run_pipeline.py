"""Resume the frozen a02 run in a visible Kit session and audit the exact target."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


TRIAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = TRIAL_DIR.parents[2]
REPO_ROOT = TRAINING_ROOT.parents[1]
A02_DIR = TRIAL_DIR.parent / "physx_pointmove_flat12_20261024_a02"
A02_PIPELINE = A02_DIR / "run_pipeline.py"
HANDOFF_PATH = TRIAL_DIR / "handoff.json"
SOURCE_SNAPSHOT = TRIAL_DIR / "source_snapshot"
PROTOCOL_PATH = TRIAL_DIR / "protocol.json"
MANIFEST_PATH = TRIAL_DIR / "source_manifest.json"
STATUS_PATH = TRIAL_DIR / "run_status.json"
PREFLIGHT_PATH = TRIAL_DIR / "preflight.json"
LOCK_PATH = TRIAL_DIR / ".run_pipeline.lock"
PYTHON_EXE = Path(r"C:\pioneer\isaac-training\.venv\Scripts\python.exe")
EXPERIMENT_NAME = "physx_pointmove_flat12_live_20261024_a03"
RUN_TIMESTAMP = "training_live_a03"
NUM_ENVS = 4
ROLLOUT_STEPS = 128
TARGET_TRANSITIONS = 196608
ADAM_STEPS_PER_ITERATION = 20  # 5 epochs x 4 minibatches in the frozen RSL-RL config.
EVAL_SEED_START = 87010
EVAL_SCENARIOS = 16
EVAL_GOAL_ANGLE_DEG = 180.0
KIT_ARGS = "--/isaac/startup/create_new_stage=false"
PROCESS_ROLES = ("training_native", "resume_training_cli", "controller", "watcher", "static_gui")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _old_pipeline():
    spec = importlib.util.spec_from_file_location("a02_run_pipeline", A02_PIPELINE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen a02 helpers from {A02_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse the immutable implementation while redirecting all helper outputs to a03.
    module.TRIAL_DIR = TRIAL_DIR
    module.TRAINING_ROOT = TRAINING_ROOT
    module.REPO_ROOT = REPO_ROOT
    module.EXPERIMENT_NAME = EXPERIMENT_NAME
    module.TRAINING_TRANSITIONS = TARGET_TRANSITIONS
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.STATUS_PATH = STATUS_PATH
    module.PREFLIGHT_PATH = PREFLIGHT_PATH
    module.LOCK_PATH = LOCK_PATH
    module.EVAL_SEED_START = EVAL_SEED_START
    module.EVAL_SCENARIOS = EVAL_SCENARIOS
    module.EVAL_GOAL_ANGLE_DEG = EVAL_GOAL_ANGLE_DEG
    return module


def _require_stopped_handoff(path: Path = HANDOFF_PATH) -> dict[str, Any]:
    handoff = read_json(path)
    checkpoint = handoff.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("handoff.json must contain checkpoint metadata")
    old_processes = handoff.get("old_processes")
    if not isinstance(old_processes, dict):
        raise ValueError("handoff.json old_processes must be an object keyed by process role")
    missing = [role for role in PROCESS_ROLES if role not in old_processes]
    if missing:
        raise ValueError(f"handoff is missing stopped-process roles: {missing}")
    for role in PROCESS_ROLES:
        record = old_processes[role]
        if (not isinstance(record, dict) or record.get("state") != "stopped"
                or record.get("verified") is not True or not record.get("evidence_path")
                or type(record.get("pid")) is not int):
            raise ValueError(f"old process {role} lacks verified stopped evidence")
    old_run_status = handoff.get("old_run_status")
    if not isinstance(old_run_status, dict) or old_run_status.get("status") != "superseded":
        raise ValueError("handoff must mark a02 superseded before preparing a03")
    successor = old_run_status.get("superseded_by")
    accepted_successors = (None, TRIAL_DIR.name, str(TRIAL_DIR.resolve()))
    if successor not in accepted_successors:
        raise ValueError("a02 is superseded by a different trial")
    return handoff


def _adam_step(payload: dict[str, Any]) -> int:
    optimizer = payload.get("optimizer_state_dict")
    state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("resume checkpoint has no Adam state")
    steps: set[int] = set()
    for parameter_state in state.values():
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            raise ValueError("Adam parameter state is missing its step counter")
        step = parameter_state["step"]
        steps.add(int(step.item()) if hasattr(step, "item") else int(step))
    if len(steps) != 1:
        raise ValueError(f"Adam parameter steps diverge: {sorted(steps)}")
    return steps.pop()


def _optimizer_state_size(payload: dict[str, Any]) -> int:
    optimizer = payload.get("optimizer_state_dict")
    state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("resume checkpoint has no optimizer parameter states")
    return len(state)


def remaining_iterations(num_timesteps: int, target: int = TARGET_TRANSITIONS) -> int:
    batch = NUM_ENVS * ROLLOUT_STEPS
    if num_timesteps < 0 or num_timesteps >= target:
        raise ValueError(f"checkpoint transitions must be in [0, {target}); got {num_timesteps}")
    if num_timesteps % batch or target % batch:
        raise ValueError("checkpoint and target must align to one complete vectorized rollout")
    return (target - num_timesteps) // batch


def a02_training_run_dir() -> Path:
    return A02_DIR / "logs/rsl_rl" / A02_DIR.name / "training_a02"


def select_latest_complete_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Choose the newest stable, safely loadable checkpoint with consistent counts."""
    helper = _old_pipeline()
    checkpoints = helper._numeric_model_checkpoints(run_dir)
    failures: list[str] = []
    for filename_iter, path in reversed(checkpoints):
        try:
            first = (path.stat().st_size, path.stat().st_mtime_ns)
            time.sleep(0.2)
            second = (path.stat().st_size, path.stat().st_mtime_ns)
            if first != second or first[0] <= 0:
                raise ValueError("checkpoint file is still changing or empty")
            payload, safe = helper.cpu_load_checkpoint(path)
            iteration = payload.get("iter")
            transitions = payload.get("num_timesteps")
            if type(iteration) is not int or iteration != filename_iter:
                raise ValueError(f"filename iteration {filename_iter} != payload iter {iteration}")
            batch = NUM_ENVS * ROLLOUT_STEPS
            if type(transitions) is not int or transitions <= 0 or transitions % batch:
                raise ValueError(f"checkpoint has invalid cumulative transitions {transitions}")
            if transitions > TARGET_TRANSITIONS:
                raise ValueError(f"checkpoint exceeds target transitions: {transitions}")
            if transitions < TARGET_TRANSITIONS:
                remaining_iterations(transitions)
            if not isinstance(payload.get("actor_state_dict"), dict) or not isinstance(payload.get("critic_state_dict"), dict):
                raise ValueError("checkpoint lacks actor/critic state")
            helper._assert_checkpoint_contract(payload.get("field_policy_contract"), helper.expected_contract_requirements())
            optimizer = payload.get("optimizer_state_dict")
            if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
                raise ValueError("checkpoint lacks optimizer state")
            step = _adam_step(payload)
            return path, payload, {
                **safe,
                "adam_step": step,
                "optimizer_state_entries": _optimizer_state_size(payload),
                "filename_iteration": filename_iter,
            }
        except (OSError, RuntimeError, ValueError, EOFError) as exc:
            failures.append(f"{path.name}: {exc}")
    raise RuntimeError(f"no complete resumable checkpoint in {run_dir}: {failures[-5:]}")


def training_argv(checkpoint: Path, iterations: int) -> list[str]:
    return [
        str(PYTHON_EXE), "-m", "isaac_field", "resume",
        "--checkpoint", str(checkpoint.resolve()),
        "--num_envs", str(NUM_ENVS), "--max_iterations", str(iterations),
        "--device", "cuda:0", "--seed", "69041", "--logger", "tensorboard",
        "--experiment_name", EXPERIMENT_NAME, "--run_timestamp", RUN_TIMESTAMP,
        "--resume_optimizer_state", "--viz", "kit", f"--kit_args={KIT_ARGS}",
        "env.task.size=12.0", "env.scene.env_spacing=22.0",
    ]


def _snapshot_a02_files(manifest: dict[str, Any]) -> dict[str, Any]:
    files = {
        A02_PIPELINE: "source_snapshot/a02/run_pipeline.py",
        A02_DIR / "protocol.json": "source_snapshot/a02/protocol.json",
        A02_DIR / "source_manifest.json": "source_snapshot/a02/source_manifest.json",
        A02_DIR / "baseline_validation.json": "source_snapshot/a02/baseline_validation.json",
    }
    rows = manifest["files"]
    for source, rel_snapshot in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"required immutable a02 evidence is missing: {source}")
        destination = TRIAL_DIR / rel_snapshot
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({
            "path": source.relative_to(TRAINING_ROOT).as_posix(),
            "origin": "project",
            "snapshot_path": rel_snapshot,
            "sha256": sha256_file(source),
        })
    return manifest


def prepare(checkpoint_argument: str | Path) -> dict[str, Any]:
    if PROTOCOL_PATH.exists() or MANIFEST_PATH.exists() or (TRIAL_DIR / "source_snapshot").exists():
        raise FileExistsError(f"refusing to freeze over existing a03 data: {TRIAL_DIR}")
    handoff = _require_stopped_handoff()
    process_evidence = _verify_process_evidence(handoff)
    old_status = read_json(A02_DIR / "run_status.json")
    if old_status.get("status") != "superseded":
        raise RuntimeError("a02 run_status.json must be superseded before a03 prepare")
    superseded_by = old_status.get("superseded_by")
    if superseded_by not in (None, TRIAL_DIR.name, str(TRIAL_DIR.resolve())):
        raise RuntimeError(f"a02 status points to another successor: {superseded_by}")

    helper = _old_pipeline()
    old_protocol_path = A02_DIR / "protocol.json"
    old_protocol_sha = sha256_file(old_protocol_path)
    if old_status.get("protocol_sha256") != old_protocol_sha:
        raise RuntimeError("a02 run_status.json no longer matches its frozen protocol SHA")
    if handoff.get("source_protocol_sha256") != old_protocol_sha:
        raise RuntimeError("handoff a02 protocol SHA does not match the frozen source protocol")
    baseline_audit = read_json(A02_DIR / "baseline_validation.json")
    if baseline_audit.get("status") != "passed":
        raise RuntimeError("a02 baseline_validation.json is not passed")

    checkpoint, payload, safe = select_latest_complete_checkpoint(a02_training_run_dir())
    supplied = Path(checkpoint_argument).resolve()
    if supplied != checkpoint.resolve():
        raise RuntimeError(f"--checkpoint must select the latest complete checkpoint: {checkpoint}")
    checkpoint_meta = handoff["checkpoint"]
    if (Path(checkpoint_meta.get("path", "")).resolve() != checkpoint.resolve()
            or checkpoint_meta.get("sha256") != sha256_file(checkpoint)
            or checkpoint_meta.get("iter") != payload["iter"]
            or checkpoint_meta.get("num_timesteps") != payload["num_timesteps"]
            or checkpoint_meta.get("adam_step") != safe["adam_step"]):
        raise RuntimeError("handoff checkpoint identity/counts differ from the latest complete checkpoint")
    start_transitions = payload["num_timesteps"]
    if start_transitions != (payload["iter"] + 1) * NUM_ENVS * ROLLOUT_STEPS:
        raise ValueError("a02 source checkpoint iteration/transition counters are inconsistent")
    iterations = remaining_iterations(start_transitions)
    expected_final_adam_step = safe["adam_step"] + iterations * ADAM_STEPS_PER_ITERATION

    identity = helper.repository_identity()
    versions = helper.installed_versions()
    manifest = _snapshot_a02_files(helper.build_source_manifest())
    manifest["files"].sort(key=lambda row: (row["origin"], row["path"]))
    requirements = helper.expected_contract_requirements()
    protocol = copy.deepcopy(read_json(old_protocol_path))
    protocol.update({
        "schema": "isaac-field-live-resume-protocol-v1",
        "status": "prepared",
        "created_at_utc": utc_now(),
        "trial_id": TRIAL_DIR.name,
        "objective": "resume the latest complete a02 checkpoint to exactly 196608 cumulative transitions, then paired development evaluation",
        "git_identity": identity,
        "runtime_versions": versions,
        "source_snapshot": manifest,
        "expected_checkpoint_contract_requirements": requirements,
        "source_handoff": {"path": str(HANDOFF_PATH.resolve()), "sha256": sha256_file(HANDOFF_PATH)},
        "stopped_process_evidence": process_evidence,
    })
    protocol["training"].update({
        "max_iterations": iterations,
        "target_transitions": TARGET_TRANSITIONS,
        "remaining_iterations": iterations,
        "transition_arithmetic": f"{start_transitions} + {iterations} * {NUM_ENVS} * {ROLLOUT_STEPS} = {TARGET_TRANSITIONS}",
        "optimizer_resume": True,
        "headless": {"selected": False, "reason": "official --viz kit enables the interactive Kit visualizer"},
        "gui": {
            "visualizer": "kit",
            "camera_origin_type": "asset",
            "camera_asset_key": "robot",
            "camera_env_index": 0,
            "camera_eye": [3.0, -4.0, 2.0],
            "camera_lookat": [0.0, 0.0, 0.15],
            "streaming_view": False,
            "max_visible_envs": None,
            "all_training_environments_visible": True,
            "kit_args": KIT_ARGS,
            "startup_experience": "official Isaac Lab default",
            "livestream": False,
            "xr": False,
        },
        "baseline": {
            "reused_from_trial": A02_DIR.name,
            "argv": [],
            "run_dir": protocol["training"]["baseline"]["run_dir"],
            "checkpoint": protocol["training"]["baseline"]["checkpoint"],
            "checkpoint_sha256": baseline_audit["checkpoint_sha256"],
            "baseline_validation_sha256": sha256_file(A02_DIR / "baseline_validation.json"),
            "retrained": False,
        },
        "resume": {
            "argv_template": training_argv(checkpoint, iterations),
            "run_dir": str(TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / RUN_TIMESTAMP),
            "checkpoint_selection": "explicit handoff checkpoint, required to equal latest stable CPU-loadable numeric a02 checkpoint",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_iter": payload["iter"],
            "checkpoint_num_timesteps": start_transitions,
            "checkpoint_adam_step": safe["adam_step"],
            "checkpoint_optimizer_state_entries": safe["optimizer_state_entries"],
            "expected_num_timesteps": TARGET_TRANSITIONS,
            "expected_adam_step": expected_final_adam_step,
            "expected_final_payload_iter": payload["iter"] + iterations - 1,
            "expected_iteration_filename_not_hardcoded": True,
            "resume_optimizer_state": True,
        },
    })
    protocol.pop("headless_cli_evidence", None)
    protocol["evaluation"]["outputs"] = {
        "baseline": str(TRIAL_DIR / "eval_baseline"),
        "final": str(TRIAL_DIR / "eval_final"),
    }
    protocol["evaluation"]["seed_start"] = EVAL_SEED_START
    protocol["evaluation"]["scenario_count"] = EVAL_SCENARIOS
    protocol["evaluation"]["scenario_seeds"] = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SCENARIOS))
    protocol["export"]["output"] = str(TRIAL_DIR / "final_policy_torchscript.pt")
    protocol["output_policy"] = "New a03 outputs are rooted in this trial; the verified zero-step baseline and its original logs remain reused from a02."
    protocol["subprocess"]["cwd"] = str(TRIAL_DIR)
    protocol["subprocess"]["gui_training_environment"] = {"HEADLESS": "0", "LIVESTREAM": "0", "XR": "0"}
    protocol["subprocess"]["windows_creation_flag"] = "CREATE_NO_WINDOW for the child console; Kit opens its normal GUI window"

    write_json_atomic(MANIFEST_PATH, manifest)
    write_json_atomic(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    status = {
        "schema": "isaac-field-one-shot-run-status-v1",
        "status": "prepared",
        "phase": "prepared",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "parent_pid": os.getpid(),
        "children": [],
        "current_stage": None,
        "source_trial": A02_DIR.name,
        "resume_checkpoint": protocol["training"]["resume"],
        "export": {"status": "pending_optional"},
    }
    write_json_atomic(STATUS_PATH, status)
    # Re-run the established CPU audit against the already-existing a02 baseline.
    baseline = helper.audit_baseline(protocol, status)
    status = read_json(STATUS_PATH)
    status.update({"status": "prepared", "phase": "prepared", "current_stage": None,
                   "baseline_checkpoint": {"path": baseline["checkpoint"], "sha256": baseline["checkpoint_sha256"], "audit": "passed", "reused": True}})
    helper._stage_status(status, status)
    result = {
        "schema": "isaac-field-live-resume-prepare-v1",
        "status": "prepared",
        "trial_dir": str(TRIAL_DIR),
        "protocol_sha256": protocol_sha,
        "source_file_count": len(manifest["files"]),
        "resume_checkpoint": protocol["training"]["resume"],
        "remaining_iterations": iterations,
        "target_transitions": TARGET_TRANSITIONS,
        "native_processes_started": False,
        "kit_imported": False,
        "training_started": False,
    }
    write_json_atomic(TRIAL_DIR / "prepare.json", result)
    return result


def _verify_process_evidence(handoff: dict[str, Any]) -> list[dict[str, str]]:
    evidence_paths = set()
    for role in PROCESS_ROLES:
        path = Path(handoff["old_processes"][role]["evidence_path"])
        evidence_paths.add(path if path.is_absolute() else REPO_ROOT / path)
    for path in evidence_paths:
        if not path.is_file():
            raise FileNotFoundError(f"stopped-process evidence missing: {path}")
    running = [
        role for role in PROCESS_ROLES
        if _pid_running(handoff["old_processes"][role]["pid"])
    ]
    if running:
        raise RuntimeError(f"old processes are still alive according to the OS: {running}")
    return [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sorted(evidence_paths)]


def _pid_running(pid: int) -> bool:
    """Read a process handle/exit code without signaling or changing the process."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER means the PID no longer exists.
                return False
            raise OSError(error, f"OpenProcess({pid}) failed")
        try:
            code = ctypes.c_uint32()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), f"GetExitCodeProcess({pid}) failed")
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def apply_training_gui_environment(environment: dict[str, str], *, training: bool) -> dict[str, str]:
    result = dict(environment)
    if training:
        result.update({"HEADLESS": "0", "LIVESTREAM": "0", "XR": "0"})
    return result


def _set_execution_state(flags: int) -> bool:
    import ctypes

    request = ctypes.windll.kernel32.SetThreadExecutionState
    request.argtypes = [ctypes.c_uint32]
    request.restype = ctypes.c_uint32
    return bool(request(flags))


def preflight(*, write_report: bool = True) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not STATUS_PATH.is_file():
        raise FileNotFoundError("run --prepare --checkpoint <latest-complete-a02-checkpoint> first")
    helper = _old_pipeline()
    handoff = _require_stopped_handoff()
    process_evidence = _verify_process_evidence(handoff)
    status = read_json(STATUS_PATH)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("a03 is not an unchanged prepared run")
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("source_handoff", {}).get("sha256") != sha256_file(HANDOFF_PATH):
        raise RuntimeError("handoff changed after prepare")
    if protocol.get("stopped_process_evidence") != process_evidence:
        raise RuntimeError("stopped-process evidence changed after prepare")
    if read_json(MANIFEST_PATH) != protocol.get("source_snapshot"):
        raise RuntimeError("source_manifest.json differs from the manifest frozen into protocol.json")
    if helper.repository_identity() != protocol.get("git_identity"):
        raise RuntimeError("repository or Isaac Lab identity changed after prepare")
    if helper.installed_versions() != protocol.get("runtime_versions"):
        raise RuntimeError("runtime versions changed after prepare")
    manifest_check = helper._verify_manifest(protocol["source_snapshot"])
    if not manifest_check["valid"]:
        raise RuntimeError(f"frozen source snapshot mismatch: {manifest_check['failures'][:5]}")
    source_status = read_json(A02_DIR / "run_status.json")
    source_successor = source_status.get("superseded_by")
    if (source_status.get("status") != "superseded"
            or source_successor not in (None, TRIAL_DIR.name, str(TRIAL_DIR.resolve()))):
        raise RuntimeError("a02 is no longer marked superseded by this live trial")
    resume = protocol["training"]["resume"]
    checkpoint = Path(resume["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != resume["checkpoint_sha256"]:
        raise RuntimeError("selected source checkpoint is missing or changed")
    latest_path, latest_payload, latest_safe = select_latest_complete_checkpoint(checkpoint.parent)
    if latest_path.resolve() != checkpoint.resolve():
        raise RuntimeError("a newer complete a02 checkpoint appeared after prepare; prepare a03 again from it")
    if (latest_payload["num_timesteps"] != resume["checkpoint_num_timesteps"]
            or latest_safe["adam_step"] != resume["checkpoint_adam_step"]
            or latest_safe["optimizer_state_entries"] != resume["checkpoint_optimizer_state_entries"]):
        raise RuntimeError("source checkpoint counters changed after prepare")
    planned = [Path(resume["run_dir"]), TRIAL_DIR / "eval_baseline", TRIAL_DIR / "eval_final"]
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"a03 output paths already exist: {existing}")
    baseline_path = Path(protocol["training"]["baseline"]["checkpoint"])
    if sha256_file(baseline_path) != protocol["training"]["baseline"]["checkpoint_sha256"]:
        raise RuntimeError("reused a02 baseline checkpoint changed")
    report = {
        "schema": "isaac-field-live-resume-preflight-v1",
        "status": "passed",
        "checked_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "source_manifest": manifest_check,
        "a02_status": "superseded",
        "old_processes_stopped": True,
        "selected_checkpoint": {key: resume[key] for key in (
            "checkpoint", "checkpoint_sha256", "checkpoint_iter", "checkpoint_num_timesteps",
            "checkpoint_adam_step", "checkpoint_optimizer_state_entries", "expected_final_payload_iter",
        )},
        "remaining_iterations": resume["argv_template"][resume["argv_template"].index("--max_iterations") + 1],
        "target_transitions": TARGET_TRANSITIONS,
        "training_argv": resume["argv_template"],
        "visualizer": protocol["training"]["gui"],
        "planned_outputs_absent": True,
        "native_processes_started": False,
        "kit_imported": False,
        "training_started": False,
    }
    if write_report:
        write_json_atomic(PREFLIGHT_PATH, report)
    return report


def audit_final(protocol: dict[str, Any], status: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    helper = _old_pipeline()
    helper._assert_frozen("final_cpu_audit")
    resume = protocol["training"]["resume"]
    run_dir = Path(resume["run_dir"])
    env_yaml, agent_yaml = run_dir / "params/env.yaml", run_dir / "params/agent.yaml"
    if not env_yaml.is_file() or not agent_yaml.is_file():
        raise FileNotFoundError("official resumed CLI did not write environment/agent config")
    env_checks = {
        "task.size": helper.require_yaml_value(env_yaml, "task", "size", "12.0"),
        "task.plants": helper.require_yaml_value(env_yaml, "task", "plants", "0"),
        "task.terrain_enabled": helper.require_yaml_value(env_yaml, "task", "terrain_enabled", "false"),
        "task.rock_density": helper.require_yaml_value(env_yaml, "task", "rock_density", "0.0"),
        "task.max_time": helper.require_yaml_value(env_yaml, "task", "max_time", "60.0"),
        "task.goal_tolerance": helper.require_yaml_value(env_yaml, "task", "goal_tolerance", "0.4"),
        "task.goal_stop_speed": helper.require_yaml_value(env_yaml, "task", "goal_stop_speed", "0.05"),
        "task.goal_stop_angular_speed": helper.require_yaml_value(env_yaml, "task", "goal_stop_angular_speed", "0.1"),
        "task.goal_hold": helper.require_yaml_value(env_yaml, "task", "goal_hold", "0.5"),
        "task.input_schema": helper.require_yaml_value(env_yaml, "task", "input_schema", "sensor-absolute-goal-v1"),
        "task.goto_reward_profile": helper.require_yaml_value(env_yaml, "task", "goto_reward_profile", "fast_stop_v2"),
        "scene.env_spacing": helper.require_yaml_value(env_yaml, "scene", "env_spacing", "22.0"),
        "scene.num_envs": helper.require_yaml_value(env_yaml, "scene", "num_envs", str(NUM_ENVS)),
    }
    required_agent_lines = (
        f"seed: 69041", f"num_steps_per_env: {ROLLOUT_STEPS}",
        f"max_iterations: {protocol['training']['remaining_iterations']}", "logger: tensorboard",
    )
    agent_text = agent_yaml.read_text(encoding="utf-8")
    for line in required_agent_lines:
        if not any(row.strip() == line for row in agent_text.splitlines()):
            raise ValueError(f"resumed CLI agent.yaml lacks expected setting: {line}")

    path, payload, safe = select_latest_complete_checkpoint(run_dir)
    if payload["num_timesteps"] != TARGET_TRANSITIONS:
        raise ValueError(f"final checkpoint has {payload['num_timesteps']} transitions; expected {TARGET_TRANSITIONS}")
    if payload["iter"] != resume["expected_final_payload_iter"]:
        raise ValueError(f"final payload iteration {payload['iter']} != runner resume expectation {resume['expected_final_payload_iter']}")
    expected_adam_step = resume["expected_adam_step"]
    if safe["adam_step"] != expected_adam_step:
        raise ValueError(f"final Adam step {safe['adam_step']} != expected {expected_adam_step}")
    if safe["optimizer_state_entries"] != resume["checkpoint_optimizer_state_entries"]:
        raise ValueError("final optimizer state entry count differs from the restored checkpoint")
    baseline_path = Path(baseline["checkpoint"])
    baseline_sha = baseline["checkpoint_sha256"]
    if sha256_file(baseline_path) != baseline_sha:
        raise RuntimeError("reused a02 baseline SHA changed during continuation")
    baseline_payload, _ = helper.cpu_load_checkpoint(baseline_path)
    changes = helper._state_dict_changes(baseline_payload, payload, __import__("torch"))
    final_contract = helper._assert_checkpoint_contract(payload.get("field_policy_contract"), protocol["expected_checkpoint_contract_requirements"])
    if final_contract["contract"] != baseline["field_policy_contract"]:
        raise ValueError("final policy/physics contract differs from the reused a02 baseline")
    if sha256_file(baseline_path) != baseline_sha:
        raise RuntimeError("reused a02 baseline changed while auditing final checkpoint")
    report = {
        "schema": "isaac-field-trained-checkpoint-audit-v1",
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "max_numeric_filename_iteration": safe["filename_iteration"],
        "checkpoint_payload_iter": payload["iter"],
        "iteration_filename_not_assumed": True,
        "num_timesteps": payload["num_timesteps"],
        "expected_num_timesteps": TARGET_TRANSITIONS,
        "remaining_iterations": protocol["training"]["remaining_iterations"],
        "optimizer_step_at_resume": resume["checkpoint_adam_step"],
        "optimizer_steps_per_iteration": ADAM_STEPS_PER_ITERATION,
        "optimizer_step_expected": expected_adam_step,
        "optimizer_step_actual": safe["adam_step"],
        "optimizer_state_entries": len(payload["optimizer_state_dict"]["state"]),
        "optimizer_state_entries_at_resume": resume["checkpoint_optimizer_state_entries"],
        "optimizer_finite": True,
        "model_state_changes_from_baseline": changes,
        "safe_load": safe,
        "official_cli_config": {
            "env_yaml": str(env_yaml), "env_yaml_sha256": sha256_file(env_yaml),
            "agent_yaml": str(agent_yaml), "agent_yaml_sha256": sha256_file(agent_yaml),
            "checks": env_checks,
        },
        "field_policy_contract_sha256": final_contract["contract_sha256"],
        "complete_contract_matches_baseline": True,
        "baseline_checkpoint": str(baseline_path),
        "baseline_sha256_before": baseline_sha,
        "baseline_sha256_after": sha256_file(baseline_path),
    }
    write_json_atomic(TRIAL_DIR / "final_validation.json", report)
    status["final_checkpoint"] = {
        "path": str(path), "sha256": report["checkpoint_sha256"], "audit": "passed",
        "num_timesteps": TARGET_TRANSITIONS, "adam_step": safe["adam_step"],
    }
    status["phase"] = "final_checkpoint_validated"
    helper._stage_status(status, status)
    return report


def _evaluation_argv(checkpoint: Path, output: Path) -> list[str]:
    helper = _old_pipeline()
    return helper.evaluation_argv(checkpoint, output)


def execute_pipeline(*, export: bool = False) -> dict[str, Any]:
    preflight_report = preflight()
    if preflight_report["status"] != "passed":
        raise RuntimeError("a03 preflight did not pass")
    try:
        descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()} started={utc_now()}\n".encode("utf-8"))
        os.fsync(descriptor)
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError(f"pipeline lock exists; inspect it before starting: {LOCK_PATH}") from exc

    if not _set_execution_state(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        LOCK_PATH.unlink(missing_ok=True)
        raise OSError("Windows process sleep-prevention request failed")
    power = {"active": True, "parent_pid": os.getpid(), "started_utc": utc_now(),
             "scope": "this pipeline main thread", "system_power_plan_modified": False}
    try:
        write_json_atomic(TRIAL_DIR / "power_request.json", power)
        helper = _old_pipeline()
        status = read_json(STATUS_PATH)
        protocol = read_json(PROTOCOL_PATH)
        status.update({"status": "running", "phase": "resume_training", "started_at_utc": utc_now(), "parent_pid": os.getpid()})
        helper._stage_status(status, status)
        base_environment = helper.child_environment

        def child_environment() -> dict[str, str]:
            return apply_training_gui_environment(
                base_environment(), training=status.get("phase") in ("resume_training", "resume_training_cli")
            )

        helper.child_environment = child_environment
    except BaseException:
        power.update(active=False, released_utc=utc_now(), released=_set_execution_state(0x80000000))
        write_json_atomic(TRIAL_DIR / "power_request.json", power)
        LOCK_PATH.unlink(missing_ok=True)
        raise
    try:
        baseline = read_json(TRIAL_DIR / "baseline_validation.json")
        if baseline.get("status") != "passed":
            raise RuntimeError("reused a02 baseline CPU audit is not passed")
        if helper.run_child("resume_training_cli", protocol["training"]["resume"]["argv_template"], status) != 0:
            raise RuntimeError("official GUI resumed-training process exited nonzero")
        status["phase"] = "final_checkpoint_audit"
        helper._stage_status(status, status)
        final = audit_final(protocol, status, baseline)
        final_path = Path(final["checkpoint"])

        status["phase"] = "baseline_evaluation"
        helper._stage_status(status, status)
        baseline_path = Path(baseline["checkpoint"])
        if helper.run_child("evaluate_baseline", _evaluation_argv(baseline_path, TRIAL_DIR / "eval_baseline"), status) != 0:
            raise RuntimeError("reused baseline development evaluation exited nonzero")
        helper._assert_frozen("baseline_evaluation_report_check")
        helper._validate_evaluation_report(TRIAL_DIR / "eval_baseline/evaluation.json", baseline_path, protocol)

        status["phase"] = "final_evaluation"
        helper._stage_status(status, status)
        if helper.run_child("evaluate_final", _evaluation_argv(final_path, TRIAL_DIR / "eval_final"), status) != 0:
            raise RuntimeError("final development evaluation exited nonzero")
        helper._assert_frozen("final_evaluation_report_check")
        comparison = helper.compare_evaluations(protocol, baseline_path, final_path)

        export_code = None
        if export:
            status["phase"] = "optional_export"
            helper._stage_status(status, status)
            export_code = helper.run_child("optional_export", helper.export_argv(final_path), status, optional=True)
        status["status"] = "completed"
        status["phase"] = "completed"
        status["completed_at_utc"] = utc_now()
        status["current_stage"] = None
        status["comparison_acceptance"] = comparison["acceptance"]
        status["baseline_sha256_final"] = sha256_file(baseline_path)
        status["final_checkpoint_sha256"] = sha256_file(final_path)
        status["acceptance_threshold_met"] = comparison["acceptance"]["paired_development_gate_passed"]
        status["training_process_completed"] = True
        status["evaluations_completed"] = True
        status["optional_export_exit_code"] = export_code
        helper._stage_status(status, status)
        return status
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        latest = read_json(STATUS_PATH)
        current_stage = latest.get("current_stage")
        if latest.get("status") == "running" and isinstance(current_stage, dict) and current_stage.get("child_still_running"):
            latest["orchestrator_exception"] = repr(exc)
            latest["updated_at_utc"] = utc_now()
        else:
            latest["status"] = "failed"
            latest["failure"] = {"exception": repr(exc), "failed_at_utc": utc_now()}
            latest["current_stage"] = None
        helper._stage_status(latest, latest)
        raise
    finally:
        try:
            power.update(active=False, released_utc=utc_now(), released=_set_execution_state(0x80000000))
        except OSError as exc:
            power.update(active=False, released_utc=utc_now(), release_error=repr(exc))
        write_json_atomic(TRIAL_DIR / "power_request.json", power)
        LOCK_PATH.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="freeze a03 from a stopped-process handoff; CPU only")
    actions.add_argument("--preflight", action="store_true", help="verify frozen protocol and checkpoint; CPU only")
    actions.add_argument("--dry-run", action="store_true", help="print the launch/evaluation plan; CPU only")
    parser.add_argument("--checkpoint", help="explicit latest complete a02 checkpoint, required by --prepare")
    parser.add_argument("--export", action="store_true", help="also attempt optional TorchScript export after paired evaluation")
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
            result["ordered_stages"] = ["resume_training_cli --viz kit", "audit final transition and Adam counts", "evaluate_baseline 16 seeds", "evaluate_final same 16 seeds", "paired comparison"]
        else:
            result = execute_pipeline(export=args.export)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"a03 pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
