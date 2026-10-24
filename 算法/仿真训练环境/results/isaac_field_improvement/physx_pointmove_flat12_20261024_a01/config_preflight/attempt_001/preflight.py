import json
import math
import os
from pathlib import Path

from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
import isaac_field.tasks
from isaaclab.app import validate_launch
from isaaclab.utils.dict import class_to_dict
from isaaclab_tasks.utils.hydra import resolve_task_config

TASK_ID = "Isaac-Field-FastStop-Rsl-Direct"
OVERRIDES = ["env.task.size=12.0", "env.scene.env_spacing=20.0"]
OUT_DIR = Path(__file__).resolve().parent

def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        try:
            return plain(value.tolist())
        except Exception:
            pass
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return plain(value.value)
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    return repr(value)

env_cfg, runner_cfg = resolve_task_config(
    TASK_ID,
    "rsl_rl_cfg_entry_point",
    overrides=OVERRIDES,
)

# validate_launch scans renderer, cameras, and physics config only; it does not launch Kit.
scan, kit_sources = validate_launch(env_cfg, {"livestream": 0})

task = env_cfg.task
checks = {
    "task_id": TASK_ID,
    "task_size_12m": math.isclose(float(task.size), 12.0, rel_tol=0.0, abs_tol=1e-9),
    "plants_zero": int(task.plants) == 0,
    "terrain_flat": task.terrain_enabled is False,
    "rock_density_zero": math.isclose(float(task.rock_density), 0.0, rel_tol=0.0, abs_tol=1e-12),
    "horizon_60s": math.isclose(float(task.max_time), 60.0, rel_tol=0.0, abs_tol=1e-9),
    "strict_fast_stop_v2": task.goto_reward_profile == "fast_stop_v2",
    "goal_tolerance_0p4m": math.isclose(float(task.goal_tolerance), 0.4, rel_tol=0.0, abs_tol=1e-9),
    "xy_speed_strictly_below_0p05": math.isclose(float(task.goal_stop_speed), 0.05, rel_tol=0.0, abs_tol=1e-9),
    "full_angular_speed_strictly_below_0p1": math.isclose(float(task.goal_stop_angular_speed), 0.1, rel_tol=0.0, abs_tol=1e-9),
    "hold_0p5s": math.isclose(float(task.goal_hold), 0.5, rel_tol=0.0, abs_tol=1e-9),
    "absolute_goal_input": task.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA,
    "failure_remaining_time_cost": task.fast_stop_failure_remaining_time_cost is True,
    "global_random_goal_sampling": env_cfg.goal_distance is None and math.isclose(float(env_cfg.goal_angle), math.pi, rel_tol=0.0, abs_tol=1e-9),
    "scene_spacing_20m": math.isclose(float(env_cfg.scene.env_spacing), 20.0, rel_tol=0.0, abs_tol=1e-9),
    "runner_rollout_128": int(runner_cfg.num_steps_per_env) == 128,
    "runner_iterations_384": int(runner_cfg.max_iterations) == 384,
    "actor_lstm_512": int(runner_cfg.actor.rnn_hidden_dim) == 512 and int(runner_cfg.actor.rnn_num_layers) == 1,
    "critic_lstm_512": int(runner_cfg.critic.rnn_hidden_dim) == 512 and int(runner_cfg.critic.rnn_num_layers) == 1,
    "camera_detected_by_launch_scan": bool(scan.has_kit_camera),
    "kit_required_by_launch_scan": bool(scan.needs_kit),
    "no_ovrtx_conflict": not bool(scan.has_ovrtx),
}
failed = [name for name, passed in checks.items() if not passed]
scan_summary = {
    "passed": True,
    "kit_sources": list(kit_sources),
    "has_kit_camera": bool(scan.has_kit_camera),
    "has_kit_physics": bool(scan.has_kit_physics),
    "has_ovrtx": bool(scan.has_ovrtx),
    "needs_kit": bool(scan.needs_kit),
    "launcher_types": list(scan.launcher_types),
    "runtime_started": False,
    "method": "isaaclab.app.validate_launch config scan",
}
resolved = {
    "task_id": TASK_ID,
    "hydra_overrides": OVERRIDES,
    "env_cfg": plain(class_to_dict(env_cfg)),
    "runner_cfg": plain(class_to_dict(runner_cfg)),
    "launch_scan": scan_summary,
    "checks": checks,
}
(OUT_DIR / "resolved_cfg.json").write_text(
    json.dumps(resolved, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
result = {
    "status": "passed" if not failed else "failed",
    "task_id": TASK_ID,
    "hydra_overrides": OVERRIDES,
    "checks_passed": sum(bool(value) for value in checks.values()),
    "checks_total": len(checks),
    "failed_checks": failed,
    "launch_scan": scan_summary,
    "kit_started": False,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "native_runtime_disabled": os.environ.get("ISAAC_FIELD_DISABLE_NATIVE_RUNTIME"),
    "resolved_cfg": str(OUT_DIR / "resolved_cfg.json"),
}
(OUT_DIR / "summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
if failed:
    raise SystemExit(2)
