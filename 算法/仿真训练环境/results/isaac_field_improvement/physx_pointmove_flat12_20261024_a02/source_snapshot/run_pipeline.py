"""One-shot, auditable baseline/resume/development-evaluation pipeline.

Preparation and dry-run are CPU/stdlib-only. The default mode launches the
official Isaac Lab RSL-RL CLI, then existing project evaluation/export tools.
This file deliberately does not implement or wrap a training algorithm.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


TRIAL_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = TRIAL_DIR.parents[2]
REPO_ROOT = TRAINING_ROOT.parents[1]
PYTHON_EXE = Path(r"C:\pioneer\isaac-training\.venv\Scripts\python.exe")
EXPECTED_HEAD = "b3fd9cab574c9630706c007db7d75f006b9b2789"
EXPECTED_BRANCH = "chore/physx-pointmove-training"
EXPECTED_ISAACLAB_COMMIT = "5870804e4f6aee57bdef68d59f67d328b990d1e1"
EXPERIMENT_NAME = "physx_pointmove_flat12_20261024_a02"
TRAIN_SEED = 69041
NUM_ENVS = 4
ROLLOUT_STEPS = 128
TRAINING_ITERATIONS = 384
TRAINING_TRANSITIONS = NUM_ENVS * ROLLOUT_STEPS * TRAINING_ITERATIONS
EVAL_SEED_START = 87010
EVAL_SCENARIOS = 16
EVAL_GOAL_ANGLE_DEG = 180.0
TASK_NAME = "Isaac-Field-FastStop-Rsl-Direct"
PROTOCOL_PATH = TRIAL_DIR / "protocol.json"
STATUS_PATH = TRIAL_DIR / "run_status.json"
PREPARE_PATH = TRIAL_DIR / "prepare.json"
PREFLIGHT_PATH = TRIAL_DIR / "preflight.json"
LOCK_PATH = TRIAL_DIR / ".run_pipeline.lock"

EXPECTED_PACKAGES = {
    "isaacsim": "6.2.0.0",
    "isaaclab": "28.0.0",
    "isaaclab-physx": "8.0.0",
    "isaaclab-rl": "1.5.1",
    "rsl-rl-lib": "5.5.1",
    "torch": "2.11.0+cu130",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def repository_identity() -> dict[str, str]:
    branch = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    upstream = subprocess.run(
        ["git", "-C", str(TRAINING_ROOT / "external/IsaacLab"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    return {"branch": branch, "head": head, "isaaclab_commit": upstream}


def installed_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {"python": sys.version.split()[0], "packages": {}}
    packages = values["packages"]
    assert isinstance(packages, dict)
    for name in EXPECTED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return values


def expected_contract_requirements() -> dict[str, Any]:
    return {
        "backend": "rsl_rl",
        "simulator": "isaac_sim",
        "physics_backend": "physx",
        "learning_backend": "rsl_rl",
        "input_schema": "sensor-absolute-goal-v1",
        "encoder_schema": "depth64-lidar-single1d-absolute-goal-v4",
        "lstm": {"type": "lstm", "hidden_dim": 512, "num_layers": 1},
        "num_envs": NUM_ENVS,
        "num_steps_per_env": ROLLOUT_STEPS,
        "task_config": {
            "size": 12.0,
            "plants": 0,
            "terrain_enabled": False,
            "rock_density": 0.0,
            "max_time": 60.0,
            "dt": 0.05,
            "goal_tolerance": 0.4,
            "goal_stop_speed": 0.05,
            "goal_stop_angular_speed": 0.1,
            "goal_hold": 0.5,
            "goal_progress_budget": 5.0,
            "goto_reward_profile": "fast_stop_v2",
            "input_schema": "sensor-absolute-goal-v1",
            "fast_stop_failure_remaining_time_cost": True,
            "damage_metric": "stem-contact-v2",
        },
        "absolute_goal_coordinates": {
            "source": "task.goal",
            "coordinate_frame": "field_world_xy",
            "origin_xy_m": [0.0, 0.0],
            "unit": "m",
            "field_size_m": 12.0,
            "normalization": "world_xy / field_size_m * 2 - 1",
            "state_clip": [-1.0, 1.0],
        },
        "front_guide_coupling": "same_side_track_speed",
        "body_stability_reward_weights": {"tilt_weight": 0.0, "angular_rate_weight": 0.0},
    }


def _copy_source_row(relative: str, origin: str) -> dict[str, str]:
    if origin == "trial":
        source = TRIAL_DIR / relative
    else:
        source = TRAINING_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"missing frozen production source: {source}")
    snapshot_rel = Path("source_snapshot") / ("run_pipeline.py" if origin == "trial" else relative)
    destination = TRIAL_DIR / snapshot_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "path": relative,
        "origin": origin,
        "snapshot_path": snapshot_rel.as_posix(),
        "sha256": sha256_file(source),
    }


def build_source_manifest() -> dict[str, Any]:
    """Use the established fresh-trial manifest and expand it to all project scripts."""
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
    helper = importlib.import_module("scripts.train_isaac_foundation_fresh")
    inherited = helper._source_manifest(TRIAL_DIR, copy_files=True)
    rows: dict[str, dict[str, str]] = {
        row["path"]: {**row, "origin": "project"} for row in inherited["files"]
    }

    extra: set[str] = {"pyproject.toml"}
    scripts_root = TRAINING_ROOT / "scripts"
    for suffix in ("*.py", "*.ps1", "*.cmd", "*.sh"):
        extra.update(path.relative_to(TRAINING_ROOT).as_posix() for path in scripts_root.rglob(suffix))
    for relative in sorted(extra - set(rows)):
        rows[relative] = _copy_source_row(relative, "project")
    own = _copy_source_row("run_pipeline.py", "trial")
    rows["__trial__/run_pipeline.py"] = own
    ordered = [rows[key] for key in sorted(rows)]
    return {
        "manifest_schema": "isaac-field-source-snapshot-v1",
        "builder": "scripts.train_isaac_foundation_fresh._source_manifest(copy_files=True) plus all scripts Python/launch files and this run_pipeline.py",
        "files": ordered,
    }


def train_argv(stage: str, checkpoint: Path | None = None) -> list[str]:
    command = [str(PYTHON_EXE), "-m", "isaac_field"]
    if stage == "baseline":
        command.append("train")
        command.extend([
            "--num_envs", str(NUM_ENVS),
            "--max_iterations", "0",
            "--device", "cuda:0",
            "--seed", str(TRAIN_SEED),
            "--logger", "tensorboard",
            "--experiment_name", EXPERIMENT_NAME,
            "--run_timestamp", "baseline_a02",
            "env.task.size=12.0",
            "env.scene.env_spacing=22.0",
        ])
    elif stage == "training":
        if checkpoint is None:
            raise ValueError("resume argv requires the baseline checkpoint")
        command.append("resume")
        command.extend([
            "--checkpoint", str(checkpoint.resolve()),
            "--num_envs", str(NUM_ENVS),
            "--max_iterations", str(TRAINING_ITERATIONS),
            "--device", "cuda:0",
            "--seed", str(TRAIN_SEED),
            "--logger", "tensorboard",
            "--experiment_name", EXPERIMENT_NAME,
            "--run_timestamp", "training_a02",
            "--resume_optimizer_state",
            "env.task.size=12.0",
            "env.scene.env_spacing=22.0",
        ])
    else:
        raise ValueError(stage)
    # The current KitLauncher rejects the removed --headless flag. The project
    # wrapper fixes headless=True internally in _prepare_official_rsl_cli_arguments.
    return command


def evaluation_argv(checkpoint: Path, output: Path) -> list[str]:
    return [
        str(PYTHON_EXE), str(TRAINING_ROOT / "scripts/evaluate_rsl_curriculum.py"),
        "--checkpoint", str(checkpoint.resolve()),
        "--output", str(output.resolve()),
        "--seed-start", str(EVAL_SEED_START),
        "--scenario-count", str(EVAL_SCENARIOS),
        "--goal-angle-deg", str(EVAL_GOAL_ANGLE_DEG),
    ]


def export_argv(checkpoint: Path) -> list[str]:
    return [
        str(PYTHON_EXE), str(TRAINING_ROOT / "scripts/export_rsl_policy.py"),
        "--checkpoint", str(checkpoint.resolve()),
        "--output", str((TRIAL_DIR / "final_policy_torchscript.pt").resolve()),
    ]


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME", None)
    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible.startswith("-"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({
        "PYTHONPATH": str(TRAINING_ROOT),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ISAAC_FIELD_RUNTIME_DIR": r"C:\pioneer\isaac-training",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def process_environment_contract() -> dict[str, Any]:
    return {
        "set": {
            "PYTHONPATH": str(TRAINING_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAAC_FIELD_RUNTIME_DIR": r"C:\pioneer\isaac-training",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        },
        "removed_for_native_children": ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"],
        "conditional_removal_for_native_children": {
            "CUDA_VISIBLE_DEVICES": "remove only when set to a negative visibility value"
        },
        "cpu_checkpoint_audit": {
            "map_location": "cpu",
            "weights_only": True,
            "CUDA_VISIBLE_DEVICES": "",
            "ISAAC_FIELD_DISABLE_NATIVE_RUNTIME": "1",
        },
    }


def protocol_document(source_manifest: dict[str, Any], versions: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    baseline_dir = TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / "baseline_a02"
    training_dir = TRIAL_DIR / "logs/rsl_rl" / EXPERIMENT_NAME / "training_a02"
    baseline_checkpoint = baseline_dir / "model_0.pt"
    return {
        "schema": "isaac-field-one-shot-pointmove-protocol-v1",
        "status": "prepared",
        "created_at_utc": utc_now(),
        "trial_id": TRIAL_DIR.name,
        "objective": "one official RSL-RL zero-step baseline, 196608-transition optimizer-resuming continuation, paired full-episode development evaluation",
        "git_identity": identity,
        "runtime_versions": versions,
        "expected_runtime_versions": EXPECTED_PACKAGES,
        "backend": {"simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl"},
        "task": {
            "registered_task": TASK_NAME,
            "reward_profile": "fast_stop_v2",
            "input_schema": "sensor-absolute-goal-v1",
            "encoder_schema": "depth64-lidar-single1d-absolute-goal-v4",
            "field_size_m": 12.0,
            "terrain_enabled": False,
            "plants": 0,
            "rock_density": 0.0,
            "episode_horizon_s": 60.0,
            "full_map_global_start_and_goal": True,
            "absolute_world_goal": True,
            "strict_stop": {
                "position_tolerance_m": 0.4,
                "horizontal_xy_speed_m_s": 0.05,
                "full_body_angular_speed_rad_s": 0.1,
                "continuous_hold_s": 0.5,
            },
            "front_guide_coupling": "same_side_track_speed",
            "policy": {"recurrent_type": "lstm", "hidden_size": 512, "layers": 1},
        },
        "training": {
            "launcher": "python -m isaac_field (official Isaac Lab run_train_cli, rsl_rl)",
            "seed": TRAIN_SEED,
            "num_envs": NUM_ENVS,
            "num_steps_per_env": ROLLOUT_STEPS,
            "max_iterations": TRAINING_ITERATIONS,
            "target_transitions": TRAINING_TRANSITIONS,
            "transition_arithmetic": f"{TRAINING_ITERATIONS} * {ROLLOUT_STEPS} * {NUM_ENVS}",
            "optimizer_resume": True,
            "legacy_newton_weights_loaded": False,
            "weight_transfer": False,
            "headless": {
                "cli_flag_included": False,
                "reason": "current Isaac Lab KitLauncher parser rejects removed --headless; project wrapper sets headless=True internally",
                "static_evidence": [
                    "isaac_field/runtime.py:_prepare_official_rsl_cli_arguments calls _launcher_options(device='cuda:0', headless=True, ...)",
                    "external/IsaacLab/source/isaaclab_physx/test/app/test_kit_launcher.py rejects deprecated --headless",
                ],
            },
            "baseline": {
                "argv": train_argv("baseline"),
                "run_dir": str(baseline_dir),
                "checkpoint": str(baseline_checkpoint),
                "expected_num_timesteps": 0,
                "expected_iter": 0,
                "expected_adam_state_empty": True,
            },
            "resume": {
                "argv_template": train_argv("training", baseline_checkpoint),
                "run_dir": str(training_dir),
                "checkpoint_selection": "maximum numeric suffix among model_*.pt",
                "expected_num_timesteps": TRAINING_TRANSITIONS,
                "expected_iteration_filename_not_hardcoded": True,
            },
        },
        "headless_cli_evidence": _check_cli_headless_evidence(),
        "evaluation": {
            "script": str(TRAINING_ROOT / "scripts/evaluate_rsl_curriculum.py"),
            "seed_start": EVAL_SEED_START,
            "scenario_count": EVAL_SCENARIOS,
            "scenario_seeds": list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SCENARIOS)),
            "goal_distance_m": None,
            "goal_sampling": "global full field",
            "goal_angle_deg": EVAL_GOAL_ANGLE_DEG,
            "max_steps_cap": None,
            "episode_horizon_s": 60.0,
            "outputs": {"baseline": str(TRIAL_DIR / "eval_baseline"), "final": str(TRIAL_DIR / "eval_final")},
            "paired_requirements": ["same report task contract", "same initial conditions for each development seed", "full episode evaluation"],
            "assessment_scope": "development only; no 830xx data and no held-out claim",
            "strictpass_threshold": {"scenarios": 16, "clean_successes_min": 13, "damage_total_max": 0, "collision_total_max": 0, "unsafe_episodes_max": 0},
            "nonpassing_threshold_is_not_process_failure": True,
        },
        "comparison": {
            "metrics": ["successful goals", "strict clean successes", "success time", "damage episodes/total", "collision episodes/total", "unsafe episodes"],
            "safety_gate": "clean successes must increase for improvement; damage and collisions must not worsen",
            "threshold_gate": "final clean successes >= 13/16, unsafe episodes = 0, damage=0, collisions=0",
            "gate_failure_behavior": "record the measured result and complete normally if all processes/contract checks succeeded",
        },
        "export": {
            "script": str(TRAINING_ROOT / "scripts/export_rsl_policy.py"),
            "output": str(TRIAL_DIR / "final_policy_torchscript.pt"),
            "optional": True,
            "failure_effect": "record separately; completed training remains completed",
        },
        "subprocess": {
            "python": str(PYTHON_EXE),
            "cwd": str(TRIAL_DIR),
            "environment": process_environment_contract(),
            "windows_creation_flag": "CREATE_NO_WINDOW",
            "stdout_stderr": "one independent pair of files per native child stage",
            "exit_policy": "record actual process return code; stop after nonzero; no timeout kill",
        },
        "output_policy": "all pipeline-owned files and official CLI logs are rooted under this new trial directory",
        "expected_checkpoint_contract_requirements": expected_contract_requirements(),
        "source_snapshot": source_manifest,
    }


def prepare() -> dict[str, Any]:
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists() or (TRIAL_DIR / "source_snapshot").exists() or STATUS_PATH.exists():
        raise FileExistsError(f"refusing to re-prepare an existing trial: {TRIAL_DIR}")
    identity = repository_identity()
    if identity["head"] != EXPECTED_HEAD or identity["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"repository identity changed: {identity}")
    if identity["isaaclab_commit"] != EXPECTED_ISAACLAB_COMMIT:
        raise RuntimeError(f"Isaac Lab submodule commit changed: {identity['isaaclab_commit']}")
    versions = installed_versions()
    package_values = versions["packages"]
    if not isinstance(package_values, dict):
        raise TypeError("package version lookup returned an invalid value")
    mismatches = {
        name: {"expected": expected, "actual": package_values.get(name)}
        for name, expected in EXPECTED_PACKAGES.items()
        if package_values.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"installed runtime versions differ from the frozen environment: {mismatches}")
    if not PYTHON_EXE.is_file():
        raise FileNotFoundError(f"training interpreter is missing: {PYTHON_EXE}")

    manifest = build_source_manifest()
    protocol = protocol_document(manifest, versions, identity)
    write_json_atomic(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    write_json_atomic(TRIAL_DIR / "source_manifest.json", manifest)
    write_json_atomic(STATUS_PATH, {
        "schema": "isaac-field-one-shot-run-status-v1",
        "status": "prepared",
        "phase": "prepared",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "parent_pid": os.getpid(),
        "children": [],
        "current_stage": None,
        "export": {"status": "pending_optional"},
    })
    result = {
        "schema": "isaac-field-one-shot-prepare-v1",
        "status": "prepared",
        "created_at_utc": utc_now(),
        "trial_dir": str(TRIAL_DIR),
        "protocol": "protocol.json",
        "protocol_sha256": protocol_sha,
        "source_manifest": "source_manifest.json",
        "source_file_count": len(manifest["files"]),
        "python": sys.executable,
        "runtime_versions": versions,
        "native_processes_started": False,
        "kit_imported": False,
        "training_started": False,
    }
    write_json_atomic(PREPARE_PATH, result)
    return result


def _verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("protocol source manifest is empty")
    checked = 0
    seen: set[str] = set()
    failures: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            failures.append({"path": "<invalid>", "reason": "row is not an object"})
            continue
        path = row.get("path")
        origin = row.get("origin", "project")
        snapshot_rel = row.get("snapshot_path")
        expected_sha = row.get("sha256")
        if not all(isinstance(value, str) and value for value in (path, snapshot_rel, expected_sha)):
            failures.append({"path": str(path), "reason": "missing manifest fields"})
            continue
        unique_key = f"{origin}:{path}"
        if unique_key in seen:
            failures.append({"path": path, "reason": "duplicate source entry"})
            continue
        seen.add(unique_key)
        source = TRIAL_DIR / path if origin == "trial" else TRAINING_ROOT / path
        snapshot = TRIAL_DIR / snapshot_rel
        try:
            if sha256_file(source) != expected_sha:
                failures.append({"path": path, "reason": "live source SHA-256 changed"})
            if sha256_file(snapshot) != expected_sha:
                failures.append({"path": path, "reason": "source snapshot SHA-256 mismatch"})
            checked += 1
        except OSError as exc:
            failures.append({"path": path, "reason": repr(exc)})
    return {"valid": not failures, "files_checked": checked, "failures": failures}


def _check_cli_headless_evidence() -> dict[str, Any]:
    runtime_path = TRAINING_ROOT / "isaac_field/runtime.py"
    launcher_path = TRAINING_ROOT / "external/IsaacLab/source/isaaclab_physx/isaaclab_physx/app/kit_launcher.py"
    test_path = TRAINING_ROOT / "external/IsaacLab/source/isaaclab_physx/test/app/test_kit_launcher.py"
    runtime_source = runtime_path.read_text(encoding="utf-8")
    launcher_source = launcher_path.read_text(encoding="utf-8")
    tests_source = test_path.read_text(encoding="utf-8")
    return {
        "wrapper_forces_headless": "headless=True" in runtime_source and "_launcher_options(" in runtime_source,
        "launcher_parser_has_removed_headless_flag": '"--headless"' in tests_source and "deprecated_arg" in tests_source,
        "launcher_resolves_headless": "launcher_args.get(\"headless\", False)" in launcher_source,
        "cli_flag_included": False,
        "result": "wrapper supplies headless=True; do not pass the removed --headless CLI flag",
        "evidence_sha256": {
            "runtime.py": sha256_file(runtime_path),
            "kit_launcher.py": sha256_file(launcher_path),
            "test_kit_launcher.py": sha256_file(test_path),
        },
    }


def preflight(*, write_report: bool = True) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not STATUS_PATH.is_file():
        raise FileNotFoundError("run --prepare before --preflight or execution")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    protocol = read_json(PROTOCOL_PATH)
    status = read_json(STATUS_PATH)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("trial is not an unchanged prepared run")
    identity = repository_identity()
    if identity != protocol.get("git_identity"):
        raise RuntimeError(f"Git/upstream identity changed after prepare: {identity}")
    if identity["head"] != EXPECTED_HEAD or identity["branch"] != EXPECTED_BRANCH:
        raise RuntimeError(f"repository no longer matches prepared branch/HEAD: {identity}")
    versions = installed_versions()
    if versions != protocol.get("runtime_versions"):
        raise RuntimeError("Python or installed package versions changed after prepare")
    package_values = versions["packages"]
    assert isinstance(package_values, dict)
    wrong_versions = {
        name: {"expected": version, "actual": package_values.get(name)}
        for name, version in EXPECTED_PACKAGES.items() if package_values.get(name) != version
    }
    if wrong_versions:
        raise RuntimeError(f"runtime dependency mismatch: {wrong_versions}")
    manifest_result = _verify_manifest(protocol["source_snapshot"])
    if not manifest_result["valid"]:
        raise RuntimeError(f"frozen source snapshot is invalid: {manifest_result['failures'][:5]}")
    headless = _check_cli_headless_evidence()
    if not all(headless[key] for key in ("wrapper_forces_headless", "launcher_parser_has_removed_headless_flag", "launcher_resolves_headless")):
        raise RuntimeError(f"headless launch behavior is no longer supported by the frozen evidence: {headless}")
    if headless["evidence_sha256"] != protocol.get("headless_cli_evidence", {}).get("evidence_sha256"):
        raise RuntimeError("static Kit headless CLI evidence changed after protocol freeze")

    baseline_path = Path(protocol["training"]["baseline"]["checkpoint"])
    baseline_run_dir = Path(protocol["training"]["baseline"]["run_dir"])
    training_run_dir = Path(protocol["training"]["resume"]["run_dir"])
    planned_outputs = [
        baseline_run_dir,
        training_run_dir,
        TRIAL_DIR / "eval_baseline",
        TRIAL_DIR / "eval_final",
        TRIAL_DIR / "final_policy_torchscript.pt",
        TRIAL_DIR / "final_policy_torchscript.json",
    ]
    existing_outputs = [str(path) for path in planned_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"planned output paths already exist; refusing overwrite: {existing_outputs}")
    if baseline_path.exists():
        raise FileExistsError(f"baseline checkpoint already exists: {baseline_path}")
    report = {
        "schema": "isaac-field-one-shot-preflight-v1",
        "status": "passed",
        "checked_at_utc": utc_now(),
        "protocol_sha256": protocol_sha,
        "git_identity": identity,
        "runtime_versions": versions,
        "source_manifest": manifest_result,
        "headless_cli_check": headless,
        "all_planned_outputs_absent": True,
        "planned_outputs": [str(path) for path in planned_outputs],
        "native_processes_started": False,
        "kit_imported": False,
        "training_started": False,
        "train_and_resume_commands": {
            "baseline": protocol["training"]["baseline"]["argv"],
            "resume_template": protocol["training"]["resume"]["argv_template"],
        },
        "eval_commands": {
            "baseline": evaluation_argv(baseline_path, TRIAL_DIR / "eval_baseline"),
            "final": evaluation_argv(TRIAL_DIR / "<selected-final-checkpoint>.pt", TRIAL_DIR / "eval_final"),
        },
    }
    if write_report:
        write_json_atomic(PREFLIGHT_PATH, report)
    return report


def _stage_status(status: dict[str, Any], stage: dict[str, Any]) -> None:
    status["updated_at_utc"] = utc_now()
    if stage is not status:
        status["current_stage"] = stage
    write_json_atomic(STATUS_PATH, status)


def _assert_frozen(stage_name: str) -> tuple[dict[str, Any], str]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    status = read_json(STATUS_PATH)
    if status.get("protocol_sha256") != protocol_sha:
        raise RuntimeError(f"protocol SHA changed before stage {stage_name}")
    protocol = read_json(PROTOCOL_PATH)
    source = _verify_manifest(protocol["source_snapshot"])
    if not source["valid"]:
        raise RuntimeError(f"frozen sources changed before stage {stage_name}: {source['failures'][:5]}")
    return protocol, protocol_sha


def _heartbeat_wait(process: subprocess.Popen, status: dict[str, Any], stage: dict[str, Any]) -> int:
    while True:
        try:
            return_code = process.wait(timeout=15)
            return int(return_code)
        except subprocess.TimeoutExpired:
            stage["last_heartbeat_at_utc"] = utc_now()
            stage["child_still_running"] = process.poll() is None
            _stage_status(status, stage)
        except KeyboardInterrupt:
            return_code = process.poll()
            stage["orchestrator_interrupted_at_utc"] = utc_now()
            stage["child_still_running"] = return_code is None
            if return_code is not None:
                stage["actual_exit_code"] = int(return_code)
                stage["status"] = "child_exited_during_interrupt"
            _stage_status(status, stage)
            # Do not signal or terminate a native child. The durable status
            # deliberately remains running if its actual exit is not known.
            raise


def run_child(
    stage_name: str,
    argv: list[str],
    status: dict[str, Any],
    *,
    optional: bool = False,
) -> int:
    _assert_frozen(stage_name)
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", stage_name)
    stdout_path = TRIAL_DIR / f"{safe_name}.stdout.log"
    stderr_path = TRIAL_DIR / f"{safe_name}.stderr.log"
    started = utc_now()
    record: dict[str, Any] = {
        "name": stage_name,
        "status": "starting",
        "started_at_utc": started,
        "parent_pid": os.getpid(),
        "argv": argv,
        "cwd": str(TRIAL_DIR),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "optional": optional,
    }
    status.setdefault("children", []).append(record)
    status["status"] = "running"
    status["phase"] = stage_name
    _stage_status(status, record)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            process = subprocess.Popen(
                argv,
                cwd=str(TRIAL_DIR),
                env=child_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
                creationflags=creationflags,
            )
            record["child_pid"] = process.pid
            record["status"] = "running"
            record["started_at_utc"] = started
            _stage_status(status, record)
            return_code = _heartbeat_wait(process, status, record)
    except KeyboardInterrupt:
        status["status"] = "running"
        status["phase"] = stage_name
        status["updated_at_utc"] = utc_now()
        _stage_status(status, record)
        raise
    except BaseException as exc:
        record["status"] = "launcher_error"
        record["exception"] = repr(exc)
        record["completed_at_utc"] = utc_now()
        if optional:
            status["export"] = {"status": "failed", "exception": repr(exc), "stage": stage_name}
            status["current_stage"] = None
            _stage_status(status, status)
            return -1
        status["status"] = "failed"
        status["phase"] = stage_name
        status["failure"] = {"stage": stage_name, "exception": repr(exc)}
        status["current_stage"] = record
        _stage_status(status, record)
        raise
    record["child_still_running"] = False
    record["actual_exit_code"] = return_code
    record["completed_at_utc"] = utc_now()
    record["status"] = "completed" if return_code == 0 else "failed"
    status["current_stage"] = None
    if optional:
        status["export"] = {"status": "completed" if return_code == 0 else "failed", "actual_exit_code": return_code}
    _stage_status(status, status)
    if return_code != 0 and not optional:
        status["status"] = "failed"
        status["phase"] = stage_name
        status["failure"] = {"stage": stage_name, "actual_exit_code": return_code}
        _stage_status(status, status)
    return return_code


def _yaml_scalar(path: Path, section: str, key: str) -> str | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_section = False
    section_indent = -1
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if not in_section:
            if indent == 0 and stripped == f"{section}:":
                in_section = True
                section_indent = indent
            continue
        if indent <= section_indent:
            break
        if indent == section_indent + 2 and stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip()
    return None


def require_yaml_value(path: Path, section: str, key: str, expected: str) -> str:
    actual = _yaml_scalar(path, section, key)
    if actual is None or actual != expected:
        raise ValueError(f"{path} expected {section}.{key}={expected}, got {actual}")
    return actual


def _assert_subset(actual: Any, expected: Any, path: str = "contract") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path} must be an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path}.{key} is missing")
            _assert_subset(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if actual != expected:
            raise ValueError(f"{path} differs: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, float) and isinstance(actual, (float, int)) and not isinstance(actual, bool):
        if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{path} differs: expected {expected!r}, got {actual!r}")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} differs: expected {expected!r}, got {actual!r}")


def _assert_checkpoint_contract(contract: Any, expected: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("checkpoint is missing its complete field_policy_contract object")
    requirements = expected_contract_requirements()
    _assert_subset(contract, {key: value for key, value in requirements.items() if key in contract})
    # Check both duplicated architecture descriptions and the exact absolute-goal map semantics.
    if contract.get("input_schema") != requirements["input_schema"]:
        raise ValueError("checkpoint input schema is not the absolute-goal contract")
    encoder = contract.get("encoder_config")
    if not isinstance(encoder, dict) or encoder.get("schema") != requirements["encoder_schema"]:
        raise ValueError("checkpoint encoder schema is not the 512 absolute-goal encoder")
    obs_cfg = contract.get("observation_config")
    if not isinstance(obs_cfg, dict):
        raise ValueError("checkpoint lacks full observation_config")
    _assert_subset(obs_cfg, {"absolute_goal_coordinates": requirements["absolute_goal_coordinates"]}, "observation_config")
    for component in ("actor_config", "critic_config"):
        config = contract.get(component)
        if not isinstance(config, dict):
            raise ValueError(f"checkpoint lacks {component}")
        _assert_subset(config, {
            "rnn_type": "lstm",
            "rnn_hidden_dim": 512,
            "rnn_num_layers": 1,
            "input_schema": "sensor-absolute-goal-v1",
            "encoder_schema": "depth64-lidar-single1d-absolute-goal-v4",
        }, component)
    if contract["lstm"].get("hidden_dim") != 512:
        raise ValueError("checkpoint LSTM hidden_dim differs from 512")

    task = contract.get("task_config")
    if not isinstance(task, dict):
        raise ValueError("checkpoint does not contain the complete task_config")
    _assert_subset(task, requirements["task_config"], "task_config")
    guide_spec = contract.get("front_guide_spec")
    if not isinstance(guide_spec, dict) or guide_spec.get("rotation_coupling") != "same_side_track_speed":
        raise ValueError("checkpoint front-guide coupling differs from the trial")
    body = contract.get("body_stability_config")
    if not isinstance(body, dict):
        raise ValueError("checkpoint lacks body_stability_config")
    _assert_subset(body, requirements["body_stability_reward_weights"], "body_stability_config")

    drive = contract.get("drive_config")
    damage = contract.get("damage_config")
    if not isinstance(drive, dict) or not isinstance(damage, dict):
        raise ValueError("checkpoint lacks the complete drive/damage physics identity")
    if drive.get("interface") != "independent_left_right_track_speed_coupled_pi_v7":
        raise ValueError("checkpoint drive interface differs from the fixed PhysX drive model")
    if drive.get("actuator_model") != "explicit_IdealPD_velocity_servo":
        raise ValueError("checkpoint actuator model differs from the fixed PhysX drive model")
    if drive.get("actuator_effort_limit_nm") != 2.0 or drive.get("joint_effort_limit_nm") != 2.0:
        raise ValueError("checkpoint wheel effort clamp differs from 2 Nm")
    coupling = drive.get("same_side_wheel_coupling")
    if not isinstance(coupling, dict) or coupling.get("enabled") is not True or coupling.get("relations_per_world") != 14:
        raise ValueError("checkpoint does not record the 14 same-side PhysX mimic constraints")
    physics = drive.get("physics_contact")
    if not isinstance(physics, dict):
        raise ValueError("checkpoint lacks the full PhysX contact configuration")
    _assert_subset(physics, {
        "simulator": "isaac_sim",
        "physics_backend": "physx",
        "learning_backend": "rsl_rl",
        "backend": "PhysxCfg",
        "solver": "TGS",
        "physics_dt_s": 0.001,
        "decimation": 50,
        "policy_dt_s": 0.05,
        "calibrated_hardware": False,
    }, "drive_config.physics_contact")
    materials = physics.get("usd_authored_materials")
    if not isinstance(materials, dict) or not materials:
        raise ValueError("checkpoint lacks authored PhysX material identity")
    for material_path, material in materials.items():
        if not isinstance(material, dict) or not math.isclose(float(material.get("static_friction", -1)), 0.8, abs_tol=1e-6):
            raise ValueError(f"unexpected static friction in {material_path}")
        if not math.isclose(float(material.get("dynamic_friction", -1)), 0.8, abs_tol=1e-6):
            raise ValueError(f"unexpected dynamic friction in {material_path}")
    _assert_subset(damage, {"metric": "stem-contact-v2", "penalty_per_plant": 20.0}, "damage_config")
    # Save and compare the whole actual contract, not this hand-maintained subset.
    contract_sha = canonical_sha256(contract)
    return {"contract": copy.deepcopy(contract), "contract_sha256": contract_sha}


def _tensor_finite(value: Any, torch_module: Any, path: str, bad: list[str]) -> None:
    if torch_module.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch_module.isfinite(value).all().item()):
                bad.append(path)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            bad.append(path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _tensor_finite(item, torch_module, f"{path}.{key}", bad)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _tensor_finite(item, torch_module, f"{path}[{index}]", bad)


def _state_dict_changes(before: dict[str, Any], after: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    changed = 0
    compared = 0
    changed_by_group: dict[str, int] = {}
    for group in ("actor_state_dict", "critic_state_dict"):
        old = before.get(group)
        new = after.get(group)
        if not isinstance(old, dict) or not isinstance(new, dict) or old.keys() != new.keys():
            raise ValueError(f"checkpoint {group} keys differ from the zero-step baseline")
        count = 0
        for key in old:
            old_value, new_value = old[key], new[key]
            if torch_module.is_tensor(old_value) and torch_module.is_tensor(new_value):
                compared += 1
                if old_value.shape != new_value.shape or old_value.dtype != new_value.dtype:
                    raise ValueError(f"checkpoint tensor layout changed: {group}.{key}")
                if not torch_module.equal(old_value, new_value):
                    changed += 1
                    count += 1
            elif old_value != new_value:
                changed += 1
                count += 1
        changed_by_group[group] = count
    if changed == 0:
        raise ValueError("final checkpoint has no actor/critic state tensor changes from the actual baseline")
    return {"changed_state_entries": changed, "changed_by_group": changed_by_group, "tensor_entries_compared": compared}


def cpu_load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    old_disable = os.environ.get("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("safe checkpoint payload is not a plain dictionary")
        nonfinite: list[str] = []
        _tensor_finite(payload, torch, "checkpoint", nonfinite)
        if nonfinite:
            raise ValueError(f"non-finite model/optimizer tensors: {nonfinite[:8]}")
        return payload, {"device": "cpu", "map_location": "cpu", "weights_only": True, "finite_tensor_tree": True}
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        if old_disable is None:
            os.environ.pop("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME", None)
        else:
            os.environ["ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"] = old_disable


def audit_baseline(protocol: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    _assert_frozen("baseline_cpu_audit")
    path = Path(protocol["training"]["baseline"]["checkpoint"])
    payload, safe = cpu_load_checkpoint(path)
    if payload.get("num_timesteps") != 0 or payload.get("iter") != 0:
        raise ValueError(f"zero-step checkpoint count/iteration mismatch: {payload.get('num_timesteps')}, {payload.get('iter')}")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or optimizer.get("state") not in ({}, None):
        raise ValueError("zero-step baseline must contain the official optimizer state with no Adam updates")
    params = path.parent / "params"
    env_yaml = params / "env.yaml"
    agent_yaml = params / "agent.yaml"
    if not env_yaml.is_file() or not agent_yaml.is_file():
        raise FileNotFoundError("official CLI did not write both environment and agent config YAML files")
    config_checks = {
        "task.size": require_yaml_value(env_yaml, "task", "size", "12.0"),
        "task.plants": require_yaml_value(env_yaml, "task", "plants", "0"),
        "task.terrain_enabled": require_yaml_value(env_yaml, "task", "terrain_enabled", "false"),
        "task.rock_density": require_yaml_value(env_yaml, "task", "rock_density", "0.0"),
        "task.max_time": require_yaml_value(env_yaml, "task", "max_time", "60.0"),
        "task.goal_tolerance": require_yaml_value(env_yaml, "task", "goal_tolerance", "0.4"),
        "task.goal_stop_speed": require_yaml_value(env_yaml, "task", "goal_stop_speed", "0.05"),
        "task.goal_stop_angular_speed": require_yaml_value(env_yaml, "task", "goal_stop_angular_speed", "0.1"),
        "task.goal_hold": require_yaml_value(env_yaml, "task", "goal_hold", "0.5"),
        "task.input_schema": require_yaml_value(env_yaml, "task", "input_schema", "sensor-absolute-goal-v1"),
        "task.goto_reward_profile": require_yaml_value(env_yaml, "task", "goto_reward_profile", "fast_stop_v2"),
        "scene.env_spacing": require_yaml_value(env_yaml, "scene", "env_spacing", "22.0"),
        "scene.num_envs": require_yaml_value(env_yaml, "scene", "num_envs", "4"),
    }
    agent_text = agent_yaml.read_text(encoding="utf-8")
    for line, expected in (("seed: 69041", "seed"), ("num_steps_per_env: 128", "num_steps_per_env"),
                           ("max_iterations: 0", "max_iterations"), ("logger: tensorboard", "logger")):
        if not re.search(rf"(?m)^{re.escape(line)}\s*$", agent_text):
            raise ValueError(f"official CLI agent.yaml lacks expected {expected}: {line}")
    requirements = protocol["expected_checkpoint_contract_requirements"]
    actual_contract = _assert_checkpoint_contract(payload.get("field_policy_contract"), requirements)
    if not isinstance(payload.get("actor_state_dict"), dict) or not isinstance(payload.get("critic_state_dict"), dict):
        raise ValueError("baseline missing official actor/critic state dictionaries")
    if not payload["actor_state_dict"] or not payload["critic_state_dict"]:
        raise ValueError("baseline actor/critic state dictionaries are empty")
    report = {
        "schema": "isaac-field-zero-step-baseline-audit-v1",
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "iteration": payload["iter"],
        "num_timesteps": payload["num_timesteps"],
        "optimizer_state_empty": True,
        "state_dict_tensor_counts": {
            "actor": sum(1 for value in payload["actor_state_dict"].values() if hasattr(value, "shape")),
            "critic": sum(1 for value in payload["critic_state_dict"].values() if hasattr(value, "shape")),
        },
        "safe_load": safe,
        "official_cli_config": {
            "env_yaml": str(env_yaml),
            "env_yaml_sha256": sha256_file(env_yaml),
            "agent_yaml": str(agent_yaml),
            "agent_yaml_sha256": sha256_file(agent_yaml),
            "checks": config_checks,
        },
        "field_policy_contract_sha256": actual_contract["contract_sha256"],
        "field_policy_contract": actual_contract["contract"],
        "runtime_source_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "gradient_update_claimed": False,
    }
    write_json_atomic(TRIAL_DIR / "baseline_validation.json", report)
    status["baseline_checkpoint"] = {"path": str(path), "sha256": report["checkpoint_sha256"], "audit": "passed"}
    status["phase"] = "baseline_validated"
    _stage_status(status, status)
    return report


def _numeric_model_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^model_(\d+)\.pt$")
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"no official model_<iteration>.pt checkpoints in {run_dir}")
    return sorted(found, key=lambda item: item[0])


def audit_final(protocol: dict[str, Any], status: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    _assert_frozen("final_cpu_audit")
    run_dir = Path(protocol["training"]["resume"]["run_dir"])
    env_yaml = run_dir / "params/env.yaml"
    agent_yaml = run_dir / "params/agent.yaml"
    if not env_yaml.is_file() or not agent_yaml.is_file():
        raise FileNotFoundError("official resumed CLI did not write both environment and agent config YAML files")
    actual_env_checks = {
        "task.size": require_yaml_value(env_yaml, "task", "size", "12.0"),
        "task.plants": require_yaml_value(env_yaml, "task", "plants", "0"),
        "task.terrain_enabled": require_yaml_value(env_yaml, "task", "terrain_enabled", "false"),
        "task.rock_density": require_yaml_value(env_yaml, "task", "rock_density", "0.0"),
        "task.max_time": require_yaml_value(env_yaml, "task", "max_time", "60.0"),
        "task.goal_tolerance": require_yaml_value(env_yaml, "task", "goal_tolerance", "0.4"),
        "task.goal_stop_speed": require_yaml_value(env_yaml, "task", "goal_stop_speed", "0.05"),
        "task.goal_stop_angular_speed": require_yaml_value(env_yaml, "task", "goal_stop_angular_speed", "0.1"),
        "task.goal_hold": require_yaml_value(env_yaml, "task", "goal_hold", "0.5"),
        "task.input_schema": require_yaml_value(env_yaml, "task", "input_schema", "sensor-absolute-goal-v1"),
        "task.goto_reward_profile": require_yaml_value(env_yaml, "task", "goto_reward_profile", "fast_stop_v2"),
        "scene.env_spacing": require_yaml_value(env_yaml, "scene", "env_spacing", "22.0"),
        "scene.num_envs": require_yaml_value(env_yaml, "scene", "num_envs", "4"),
    }
    agent_text = agent_yaml.read_text(encoding="utf-8")
    for line, expected in (("seed: 69041", "seed"), ("num_steps_per_env: 128", "num_steps_per_env"),
                           ("max_iterations: 384", "max_iterations"), ("logger: tensorboard", "logger")):
        if not re.search(rf"(?m)^{re.escape(line)}\s*$", agent_text):
            raise ValueError(f"resumed CLI agent.yaml lacks expected {expected}: {line}")
    iterations = _numeric_model_checkpoints(run_dir)
    file_iteration, path = iterations[-1]
    baseline_path = Path(baseline["checkpoint"])
    baseline_sha = baseline["checkpoint_sha256"]
    if sha256_file(baseline_path) != baseline_sha:
        raise RuntimeError("baseline checkpoint SHA changed during continuation")
    payload, safe = cpu_load_checkpoint(path)
    if payload.get("num_timesteps") != TRAINING_TRANSITIONS:
        raise ValueError(f"final checkpoint has {payload.get('num_timesteps')} transitions; expected {TRAINING_TRANSITIONS}")
    if type(payload.get("iter")) is not int or payload["iter"] < 1:
        raise ValueError("final checkpoint has no positive integer RSL iteration")
    if ("actor_state_dict" not in payload or "critic_state_dict" not in payload
            or not isinstance(payload.get("optimizer_state_dict"), dict)):
        raise ValueError("final checkpoint is missing actor/critic/Adam state")
    optimizer_state = payload["optimizer_state_dict"].get("state")
    if not isinstance(optimizer_state, dict) or not optimizer_state:
        raise ValueError("final checkpoint contains no restored-and-updated Adam state")
    baseline_payload, _ = cpu_load_checkpoint(baseline_path)
    state_changes = _state_dict_changes(baseline_payload, payload, __import__("torch"))
    baseline_contract = baseline.get("field_policy_contract")
    final_contract = _assert_checkpoint_contract(payload.get("field_policy_contract"), protocol["expected_checkpoint_contract_requirements"])
    if final_contract["contract"] != baseline_contract:
        raise ValueError("final checkpoint's complete field policy/physics contract differs from the validated baseline")
    baseline_after_sha = sha256_file(baseline_path)
    if baseline_after_sha != baseline_sha:
        raise RuntimeError("baseline SHA changed while auditing the final checkpoint")
    report = {
        "schema": "isaac-field-trained-checkpoint-audit-v1",
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "max_numeric_filename_iteration": file_iteration,
        "checkpoint_payload_iter": payload["iter"],
        "iteration_filename_not_assumed": True,
        "num_timesteps": payload["num_timesteps"],
        "expected_num_timesteps": TRAINING_TRANSITIONS,
        "official_cli_config": {
            "env_yaml": str(env_yaml),
            "env_yaml_sha256": sha256_file(env_yaml),
            "agent_yaml": str(agent_yaml),
            "agent_yaml_sha256": sha256_file(agent_yaml),
            "checks": actual_env_checks,
        },
        "optimizer_state_entries": len(optimizer_state),
        "optimizer_finite": True,
        "model_state_changes_from_baseline": state_changes,
        "safe_load": safe,
        "field_policy_contract_sha256": final_contract["contract_sha256"],
        "complete_contract_matches_baseline": True,
        "baseline_checkpoint": str(baseline_path),
        "baseline_sha256_before": baseline_sha,
        "baseline_sha256_after": baseline_after_sha,
    }
    write_json_atomic(TRIAL_DIR / "final_validation.json", report)
    status["final_checkpoint"] = {"path": str(path), "sha256": report["checkpoint_sha256"], "audit": "passed", "num_timesteps": TRAINING_TRANSITIONS}
    status["phase"] = "final_checkpoint_validated"
    _stage_status(status, status)
    return report


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != EVAL_SCENARIOS:
        raise ValueError("evaluation report must contain exactly 16 complete scenario rows")
    successful = [row for row in rows if row.get("success") is True and row.get("reason") == "goal_reached"]
    clean = [row for row in rows if row.get("clean_success") is True]
    damages = [row for row in rows if int(row.get("damage", -1)) > 0]
    collisions = [row for row in rows if int(row.get("collisions", -1)) > 0]
    unsafe = [row for row in rows if row.get("unsafe") is True]
    times = [float(row["episode_time_s"]) for row in rows]
    success_times = [float(row["episode_time_s"]) for row in successful]
    if any(not math.isfinite(value) or value < 0 for value in times + success_times):
        raise ValueError("evaluation report contains invalid episode times")
    damage_total = sum(int(row["damage"]) for row in rows)
    collision_total = sum(int(row["collisions"]) for row in rows)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("evaluation report lacks summary")
    return {
        "scenarios": len(rows),
        "successful_goals": len(successful),
        "success_rate": len(successful) / len(rows),
        "strict_clean_successes": len(clean),
        "strict_clean_success_rate": len(clean) / len(rows),
        "mean_episode_time_s": sum(times) / len(times),
        "mean_success_time_s": None if not success_times else sum(success_times) / len(success_times),
        "median_success_time_s": None if not success_times else statistics.median(success_times),
        "damage_episodes": len(damages),
        "damage_total": damage_total,
        "collision_episodes": len(collisions),
        "collision_total": collision_total,
        "unsafe_episodes": len(unsafe),
        "strictpass": bool(summary.get("strictpass", False)),
        "assessment": report.get("assessment"),
    }


def _validate_evaluation_report(path: Path, checkpoint: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation process exited zero but did not write {path}")
    report = read_json(path)
    if report.get("schema") != "isaac-field-rsl-curriculum-evaluation-v1" or report.get("backend") != "rsl_rl":
        raise ValueError("evaluation report schema/backend mismatch")
    if report.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("evaluation report checkpoint SHA does not match the selected policy")
    expected_task = protocol["expected_checkpoint_contract_requirements"]["task_config"]
    _assert_subset(report.get("task_config"), expected_task, "evaluation.task_config")
    sampling = report.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("evaluation report lacks sampling contract")
    expected_seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + EVAL_SCENARIOS))
    if sampling.get("scenario_seeds") != expected_seeds or sampling.get("goal_distance_m") is not None:
        raise ValueError("evaluation did not use the frozen global development seed plan")
    _assert_subset(sampling, {
        "goal_sampling": "global",
        "goal_angle_deg": EVAL_GOAL_ANGLE_DEG,
        "episode_duration_s": 60.0,
        "max_steps_cap_applied": False,
        "record_trajectories": True,
        "record_initial_conditions": True,
        "device": "cpu",
    }, "evaluation.sampling")
    if report.get("assessment_scope") not in (None, "development"):
        raise ValueError("evaluation claim scope is not development data")
    if report.get("claim_limit") is None:
        raise ValueError("evaluation report lacks its development-only claim limit")
    return report


def compare_evaluations(protocol: dict[str, Any], baseline_path: Path, final_path: Path) -> dict[str, Any]:
    baseline_report = _validate_evaluation_report(TRIAL_DIR / "eval_baseline/evaluation.json", baseline_path, protocol)
    final_report = _validate_evaluation_report(TRIAL_DIR / "eval_final/evaluation.json", final_path, protocol)
    if baseline_report.get("task_config") != final_report.get("task_config"):
        raise ValueError("baseline/final evaluation task contracts differ")
    if baseline_report.get("front_guide_spec") != final_report.get("front_guide_spec"):
        raise ValueError("baseline/final front-guide identities differ")
    if baseline_report.get("body_stability_config") != final_report.get("body_stability_config"):
        raise ValueError("baseline/final body stability identities differ")
    if baseline_report.get("sampling") != final_report.get("sampling"):
        raise ValueError("baseline/final sampling configurations differ")
    if baseline_report.get("initial_conditions") != final_report.get("initial_conditions"):
        raise ValueError("baseline/final per-seed initial conditions are not paired")
    baseline_metrics = summarize_report(baseline_report)
    final_metrics = summarize_report(final_report)
    clean_increased = final_metrics["strict_clean_successes"] > baseline_metrics["strict_clean_successes"]
    safety_not_worse = (
        final_metrics["damage_total"] <= baseline_metrics["damage_total"]
        and final_metrics["collision_total"] <= baseline_metrics["collision_total"]
    )
    strictpass_threshold = (
        final_metrics["strict_clean_successes"] >= 13
        and final_metrics["damage_total"] == 0
        and final_metrics["collision_total"] == 0
    )
    no_unsafe = final_metrics["unsafe_episodes"] == 0
    acceptance = {
        "final_strictpass_at_least_13_of_16": strictpass_threshold,
        "final_unsafe_episodes_zero": no_unsafe,
        "clean_successes_increased_from_baseline": clean_increased,
        "damage_and_collision_totals_not_worse": safety_not_worse,
        "paired_development_gate_passed": bool(strictpass_threshold and no_unsafe and clean_increased and safety_not_worse),
        "process_success_independent_of_acceptance": True,
    }
    result = {
        "schema": "isaac-field-paired-development-comparison-v1",
        "status": "completed",
        "created_at_utc": utc_now(),
        "data_scope": "development",
        "held_out_claim": False,
        "reserved_830xx_seeds_used": False,
        "same_config_and_initial_conditions": True,
        "baseline": {"checkpoint": str(baseline_path), "checkpoint_sha256": sha256_file(baseline_path), "metrics": baseline_metrics},
        "final": {"checkpoint": str(final_path), "checkpoint_sha256": sha256_file(final_path), "metrics": final_metrics},
        "delta": {
            "successful_goals": final_metrics["successful_goals"] - baseline_metrics["successful_goals"],
            "strict_clean_successes": final_metrics["strict_clean_successes"] - baseline_metrics["strict_clean_successes"],
            "mean_success_time_s": None if final_metrics["mean_success_time_s"] is None or baseline_metrics["mean_success_time_s"] is None else final_metrics["mean_success_time_s"] - baseline_metrics["mean_success_time_s"],
            "damage_total": final_metrics["damage_total"] - baseline_metrics["damage_total"],
            "collision_total": final_metrics["collision_total"] - baseline_metrics["collision_total"],
            "unsafe_episodes": final_metrics["unsafe_episodes"] - baseline_metrics["unsafe_episodes"],
        },
        "acceptance": acceptance,
        "claim_limit": "Paired full-episode development evaluation only. This does not establish held-out generalization, whole-field navigation, plant avoidance, or real-vehicle performance.",
    }
    write_json_atomic(TRIAL_DIR / "comparison.json", result)
    return result


def _load_status() -> dict[str, Any]:
    status = read_json(STATUS_PATH)
    if status.get("status") != "prepared":
        raise RuntimeError(f"pipeline requires prepared status; found {status.get('status')!r}")
    return status


def execute_pipeline() -> dict[str, Any]:
    report = preflight(write_report=True)
    if not report["status"] == "passed":
        raise RuntimeError("preflight did not pass")
    lock_descriptor: int | None = None
    try:
        lock_descriptor = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_descriptor, f"pid={os.getpid()} started={utc_now()}\n".encode("utf-8"))
        os.fsync(lock_descriptor)
    except FileExistsError as exc:
        raise RuntimeError(f"pipeline lock exists; inspect its PID/status before another run: {LOCK_PATH}") from exc
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)

    status = _load_status()
    status.update({"status": "running", "phase": "baseline_launch", "started_at_utc": utc_now(), "parent_pid": os.getpid()})
    _stage_status(status, status)
    try:
        protocol, _ = _assert_frozen("baseline_launch")
        baseline_argv = protocol["training"]["baseline"]["argv"]
        if run_child("baseline_cli", baseline_argv, status) != 0:
            raise RuntimeError("official zero-step baseline CLI exited nonzero")
        status["phase"] = "baseline_audit"
        _stage_status(status, status)
        baseline = audit_baseline(protocol, status)

        status["phase"] = "resume_training"
        _stage_status(status, status)
        baseline_path = Path(baseline["checkpoint"])
        resume_argv = train_argv("training", baseline_path)
        if run_child("resume_training_cli", resume_argv, status) != 0:
            raise RuntimeError("official resumed RSL-RL training exited nonzero")

        status["phase"] = "final_checkpoint_audit"
        _stage_status(status, status)
        final = audit_final(protocol, status, baseline)
        final_path = Path(final["checkpoint"])

        status["phase"] = "baseline_evaluation"
        _stage_status(status, status)
        if run_child("evaluate_baseline", evaluation_argv(baseline_path, TRIAL_DIR / "eval_baseline"), status) != 0:
            raise RuntimeError("baseline development evaluation exited nonzero")
        _assert_frozen("baseline_evaluation_report_check")
        _validate_evaluation_report(TRIAL_DIR / "eval_baseline/evaluation.json", baseline_path, protocol)

        status["phase"] = "final_evaluation"
        _stage_status(status, status)
        if run_child("evaluate_final", evaluation_argv(final_path, TRIAL_DIR / "eval_final"), status) != 0:
            raise RuntimeError("final development evaluation exited nonzero")
        _assert_frozen("final_evaluation_report_check")
        comparison = compare_evaluations(protocol, baseline_path, final_path)

        if sha256_file(baseline_path) != baseline["checkpoint_sha256"]:
            raise RuntimeError("zero-step baseline SHA changed before optional export")
        status["phase"] = "optional_export"
        _stage_status(status, status)
        export_exit = run_child("optional_export", export_argv(final_path), status, optional=True)
        status["optional_export_exit_code"] = export_exit
        status["comparison_acceptance"] = comparison["acceptance"]
        status["status"] = "completed"
        status["phase"] = "completed"
        status["completed_at_utc"] = utc_now()
        status["current_stage"] = None
        status["baseline_sha256_final"] = sha256_file(baseline_path)
        status["final_checkpoint_sha256"] = sha256_file(final_path)
        status["acceptance_threshold_met"] = comparison["acceptance"]["paired_development_gate_passed"]
        status["training_process_completed"] = True
        status["evaluations_completed"] = True
        status["export_is_optional"] = True
        _stage_status(status, status)
        return status
    except KeyboardInterrupt:
        # Preserve the last durable stage and child PID. Never signal the child.
        raise
    except BaseException as exc:
        latest = read_json(STATUS_PATH)
        current_stage = latest.get("current_stage")
        if (latest.get("status") == "running" and isinstance(current_stage, dict)
                and current_stage.get("child_still_running")):
            latest["orchestrator_exception"] = repr(exc)
            latest["updated_at_utc"] = utc_now()
        else:
            latest["status"] = "failed"
            latest["phase"] = latest.get("phase") or "unknown"
            latest["failure"] = {"exception": repr(exc), "failed_at_utc": utc_now()}
            latest["current_stage"] = None
        write_json_atomic(STATUS_PATH, latest)
        raise
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def dry_run() -> dict[str, Any]:
    report = preflight(write_report=True)
    protocol = read_json(PROTOCOL_PATH)
    plan = {
        "schema": "isaac-field-one-shot-dry-run-v1",
        "status": "passed",
        "dry_run_at_utc": utc_now(),
        "protocol_sha256": report["protocol_sha256"],
        "ordered_stages": [
            {"stage": "baseline_cli", "argv": protocol["training"]["baseline"]["argv"]},
            {"stage": "baseline_cpu_audit", "command": "in-process torch.load(weights_only=True, map_location='cpu')"},
            {"stage": "resume_training_cli", "argv": protocol["training"]["resume"]["argv_template"]},
            {"stage": "final_cpu_audit", "command": "select maximum numeric model_*.pt, validate 196608 transitions and Adam/actor/critic"},
            {"stage": "evaluate_baseline", "argv": evaluation_argv(Path(protocol["training"]["baseline"]["checkpoint"]), TRIAL_DIR / "eval_baseline")},
            {"stage": "evaluate_final", "argv": evaluation_argv(TRIAL_DIR / "<selected-final-checkpoint>.pt", TRIAL_DIR / "eval_final")},
            {"stage": "compare", "command": "paired development comparison; threshold failure is data, not process failure"},
            {"stage": "optional_export", "argv": export_argv(TRIAL_DIR / "<selected-final-checkpoint>.pt")},
        ],
        "native_processes_started": False,
        "kit_imported": False,
        "training_started": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="freeze protocol and source snapshot; starts no simulator")
    actions.add_argument("--preflight", action="store_true", help="verify the frozen trial and planned output paths")
    actions.add_argument("--dry-run", action="store_true", help="print ordered commands; launches no child process")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.preflight:
            print(json.dumps(preflight(write_report=True), ensure_ascii=False, indent=2))
        elif args.dry_run:
            dry_run()
        else:
            import ctypes
            request = ctypes.windll.kernel32.SetThreadExecutionState
            request.argtypes = [ctypes.c_uint32]
            request.restype = ctypes.c_uint32
            if not request(0x80000001):
                raise OSError("Windows process sleep-prevention request failed")
            power = {"active": True, "parent_pid": os.getpid(), "started_utc": utc_now(),
                     "scope": "this pipeline main thread", "system_power_plan_modified": False}
            write_json_atomic(TRIAL_DIR / "power_request.json", power)
            try:
                result = execute_pipeline()
                print(json.dumps(result, ensure_ascii=False, indent=2))
            finally:
                power.update(active=False, released_utc=utc_now(), released=bool(request(0x80000000)))
                write_json_atomic(TRIAL_DIR / "power_request.json", power)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
