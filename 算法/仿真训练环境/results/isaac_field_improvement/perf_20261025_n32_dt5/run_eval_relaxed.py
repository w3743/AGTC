"""Evaluate a perf-trial checkpoint on its own 12 m task with a relaxed evaluator.

The shared strict evaluator hardcodes the frozen 100 m full-map task profile, so
it rejects any checkpoint whose task_config.size differs (this perf trial uses the
a07 12 m task). This launcher reuses the same evaluator and the same trajectory
metrics, but rebuilds the task from the checkpoint's own recorded task_config and
only keeps the semantic checks that are still meaningful for a 12 m flat field.
It must not be used to claim comparability with the frozen 100 m protocol.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

ADAPTER = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\physx_pointmove_flat12_gpu_20261024_a07\evaluate_torch_batched.py"
CHECKPOINT = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\perf_20261025_n32_dt5\logs\rsl_rl\perf_20261025_n32_dt5\train\model_95.pt"
OUTPUT = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\perf_20261025_n32_dt5\eval_final"

from scripts import evaluate_fast_stop_mastery as mastery  # noqa: E402
from scripts import view_fast_stop_goal as goal_viewer  # noqa: E402
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA  # noqa: E402
from isaac_field.task import FAST_STOP_REWARD_PROFILES  # noqa: E402


def _relaxed_validate(contract):
    """Keep fast-stop semantics, accept the checkpoint's own field size."""
    profile = (contract.get("task_config") or {}).get("goto_reward_profile")
    if profile not in {"fast_stop_v1", "fast_stop_v2"} or profile not in FAST_STOP_REWARD_PROFILES:
        raise ValueError(f"unsupported fast-stop reward profile: {profile!r}")
    if contract.get("input_schema") != ABSOLUTE_GOAL_INPUT_SCHEMA:
        raise ValueError("checkpoint input_schema must be the absolute-goal schema")
    task, front_guides, body_stability = goal_viewer._rsl_environment_configs(contract)
    if float(task.max_time) != 60.0:
        raise ValueError(f"checkpoint max_time must be 60 s, got {task.max_time!r}")
    if float(task.size) <= 0.0:
        raise ValueError(f"checkpoint field size must be positive, got {task.size!r}")
    if int(task.plants) != 0 or bool(task.terrain_enabled) or float(task.rock_density) != 0.0:
        raise ValueError("perf trial expects a flat field without plants, terrain or rocks")
    return task, front_guides, body_stability


mastery._validate_rsl_task_contract = _relaxed_validate


def _relaxed_verify_environment(engine, contract):
    """Strict on observation/action/damage, tolerant only on physics-contact timing.

    The perf policy was trained with 5 ms physics substeps; this evaluation runs
    the default 1 ms protocol, so drive_config.physics_contact is expected to
    differ. Everything that defines the policy interface must still match.
    """
    import json as _json

    for name in ("observation_config", "action_config", "damage_config"):
        expected = contract.get(name)
        actual = getattr(engine, name, None)
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ValueError(f"missing {name} metadata for verification")
        if goal_viewer._stable_hash(goal_viewer._safe_json(actual)) != goal_viewer._stable_hash(
            goal_viewer._safe_json(expected)
        ):
            raise ValueError(f"live {name} differs from the checkpoint contract")
    expected_drive = dict(contract.get("drive_config") or {})
    actual_drive = dict(getattr(engine, "drive_config", None) or {})
    expected_contact = expected_drive.pop("physics_contact", None)
    actual_contact = actual_drive.pop("physics_contact", None)
    if goal_viewer._stable_hash(goal_viewer._safe_json(actual_drive)) != goal_viewer._stable_hash(
        goal_viewer._safe_json(expected_drive)
    ):
        raise ValueError("live drive_config differs from the checkpoint contract outside physics_contact")
    print(
        "PHYSICS_CONTACT_DIFF "
        + _json.dumps({"expected": expected_contact, "actual": actual_contact}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


goal_viewer._verify_rsl_environment_contract = _relaxed_verify_environment

sys.argv = [
    ADAPTER,
    "--checkpoint", CHECKPOINT,
    "--output", OUTPUT,
    "--task-backend", "torch_batched",
    "--seed-start", "87010",
    "--scenario-count", "16",
    "--goal-angle-deg", "180",
]
runpy.run_path(ADAPTER, run_name="__main__")
