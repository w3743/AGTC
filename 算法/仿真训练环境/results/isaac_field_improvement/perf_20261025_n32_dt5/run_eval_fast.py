"""Fast policy evaluation for the 5 ms perf trial.

Reuses the a07 evaluation adapter, but:
- runs the evaluation env with the same 5 ms / 10-substep physics as training,
  so the single-env episode is not dominated by 50 GPU readbacks per policy step;
- relaxes only the frozen 100 m task-size check (this trial uses the 12 m task);
- tolerates the drive_config.physics_contact difference that is expected when
  the evaluation physics differs from training (it runs at 5 ms here, matching);
- keeps strict checks on observation/action/damage contracts.
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[3]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

os.environ.setdefault("ISAAC_FIELD_PERF_PHYSICS", "1")

import isaac_field.env as env_module  # noqa: E402

_OrigFieldEnvCfg = env_module.FieldEnvCfg


class _PerfFieldEnvCfg(_OrigFieldEnvCfg):
    """Evaluation cfg with the same physics substep as the perf training run."""

    def __post_init__(self):
        post_init = getattr(super(), "__post_init__", None)
        if post_init is not None:
            post_init()
        self.sim.dt = 0.005
        self.decimation = 10


env_module.FieldEnvCfg = _PerfFieldEnvCfg

from scripts import evaluate_fast_stop_mastery as mastery  # noqa: E402
from scripts import view_fast_stop_goal as goal_viewer  # noqa: E402
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA  # noqa: E402
from isaac_field.task import FAST_STOP_REWARD_PROFILES  # noqa: E402

ADAPTER = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\physx_pointmove_flat12_gpu_20261024_a07\evaluate_torch_batched.py"
RUN_DIR = r"C:\pioneer\先进通用履带底盘工程\算法\仿真训练环境\results\isaac_field_improvement\perf_20261025_n32_dt5"
CHECKPOINT = os.environ.get(
    "PERF_EVAL_CHECKPOINT",
    RUN_DIR + r"\logs\rsl_rl\perf_20261025_n32_dt5\train\model_95.pt",
)
OUTPUT = os.environ.get(
    "PERF_EVAL_OUTPUT",
    RUN_DIR + r"\eval_final_fast",
)
SCENARIO_COUNT = int(os.environ.get("PERF_EVAL_SCENARIOS", "2"))


def _relaxed_validate(contract):
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


def _relaxed_verify_environment(engine, contract):
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
    expected_drive.pop("physics_contact", None)
    actual_drive.pop("physics_contact", None)
    if goal_viewer._stable_hash(goal_viewer._safe_json(actual_drive)) != goal_viewer._stable_hash(
        goal_viewer._safe_json(expected_drive)
    ):
        raise ValueError("live drive_config differs from the checkpoint contract outside physics_contact")


mastery._validate_rsl_task_contract = _relaxed_validate
goal_viewer._verify_rsl_environment_contract = _relaxed_verify_environment

sys.argv = [
    ADAPTER,
    "--checkpoint", CHECKPOINT,
    "--output", OUTPUT,
    "--task-backend", "torch_batched",
    "--seed-start", "87010",
    "--scenario-count", str(SCENARIO_COUNT),
    "--goal-angle-deg", "180",
]
runpy.run_path(ADAPTER, run_name="__main__")
