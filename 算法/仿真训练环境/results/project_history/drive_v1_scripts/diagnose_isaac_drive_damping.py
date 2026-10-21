"""Sweep public Isaac Lab wheel-joint damping without changing the model or env code.

Runs a single Isaac/ Newton context and changes only the wheel joints' live
public damping property between groups.  The baseline (.2) is archived in
``results/isaac_field_diagnosis/skid_response/wheel_response.json``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/skid_response/drive_damping_sweep.json"
STEPS_PER_CASE = 30  # 1.5 s at the task's 0.05 s control period.
GAINS = (2.0, 20.0)
CASES = (
    ("straight_v0.12", (0.12, 0.0, 0.0, 0.0)),
    ("spin_turn_0.20", (0.0, 0.2, 0.0, 0.0)),
    ("spin_turn_0.70", (0.0, 0.7, 0.0, 0.0)),
    ("spin_turn_0.05", (0.0, 0.05, 0.0, 0.0)),
)


def _numpy(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _values(rover, field: str, joint_ids: list[int]) -> list[float] | None:
    try:
        value = _numpy(getattr(rover.data, field))
    except (AttributeError, RuntimeError, TypeError):
        return None
    if value.ndim == 2:
        value = value[0]
    if value.ndim != 1 or max(joint_ids, default=-1) >= len(value):
        return None
    return value[joint_ids].astype(np.float64).tolist()


def _wrapped_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _summary(rows: list[dict[str, Any]], dt: float) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    tail = rows[-10:]
    left_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("left_")]
    right_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("right_")]
    actual = np.asarray([row["actual_wheel_velocity_radps"] for row in tail], dtype=np.float64)
    target = np.asarray([row["target_wheel_velocity_radps"] for row in tail], dtype=np.float64)
    body = np.asarray([row["actual_v_w"] for row in tail], dtype=np.float64)
    command = np.asarray([row["command_v_w"] for row in tail], dtype=np.float64)
    torques = [row["applied_wheel_torque_nm"] for row in tail]
    torque_mean_abs = None
    if all(value is not None for value in torques):
        torque_mean_abs = float(np.mean(np.abs(np.asarray(torques, dtype=np.float64))))
    return {
        "steps": len(rows),
        "duration_s": float(len(rows) * dt),
        "steady_window_steps": len(tail),
        "root_yaw_delta_rad": float(_wrapped_delta(first["root_yaw_rad"], last["root_yaw_rad"])),
        "command_v_w_final": last["command_v_w"],
        "actual_v_w_final": last["actual_v_w"],
        "steady_mean_command_v_w": np.mean(command, axis=0).astype(float).tolist(),
        "steady_mean_actual_v_w": np.mean(body, axis=0).astype(float).tolist(),
        "steady_mean_wheel_target_radps_left": float(np.mean(target[:, left_ids])),
        "steady_mean_wheel_target_radps_right": float(np.mean(target[:, right_ids])),
        "steady_mean_wheel_actual_radps_left": float(np.mean(actual[:, left_ids])),
        "steady_mean_wheel_actual_radps_right": float(np.mean(actual[:, right_ids])),
        "steady_mean_abs_wheel_velocity_error_radps": float(np.mean(np.abs(actual - target))),
        "steady_mean_abs_applied_torque_nm": torque_mean_abs,
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=0,
        layout="rows",
        terrain_enabled=False,
        validate_scene=False,
        max_time=5.0,
    )
    env = make_env(
        cfg=cfg,
        mode="goto",
        num_envs=1,
        seed=80_003,
        auto_reset=False,
        goal_distance=(2.0, 2.0),
        goal_angle=0.0,
        distance_cost=0.0,
    )
    adapter = SingleFieldAdapter(env)
    wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
    wheel_ids = [env.rover.joint_names.index(name) for name in wheel_names]
    left_ids = [i for i, wheel in enumerate(ROBOT_SPEC.wheel_specs) if wheel.side == "left"]
    right_ids = [i for i, wheel in enumerate(ROBOT_SPEC.wheel_specs) if wheel.side == "right"]
    result: dict[str, Any] = {
        "schema": "isaac-field-drive-damping-sweep-v1",
        "setup": {
            "backend": "Isaac Lab Newton MJWarp",
            "one_context": True,
            "plants": 0,
            "flat_terrain": True,
            "scene_seed": 80_003,
            "start_xy_yaw": [50.0, 50.0, 0.0],
            "goal_xy": [52.0, 50.0],
            "dt_s": float(cfg.dt),
            "steps_per_case": STEPS_PER_CASE,
            "duration_s": float(STEPS_PER_CASE * cfg.dt),
            "wheel_count": len(wheel_names),
            "wheel_radius_m": float(ROBOT_SPEC.wheel_radius_m),
            "track_separation_m": float(ROBOT_SPEC.track_separation_m),
            "unchanged_effort_limit_nm": 2.0,
            "gain_write_api": "Articulation.write_joint_damping_to_sim_index",
            "cases": {name: list(action) for name, action in CASES},
        },
        "gains": {},
        "status": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for gain in GAINS:
            # Public Isaac Lab articulation API; no config monkeypatch or backend access.
            env.rover.write_joint_damping_to_sim_index(damping=float(gain), joint_ids=wheel_ids)
            readback = _values(env.rover, "joint_damping", wheel_ids)
            effort_limits = _values(env.rover, "joint_effort_limits", wheel_ids)
            if readback is None or not np.allclose(readback, gain, rtol=0.0, atol=1e-5):
                raise RuntimeError(f"wheel damping write/readback failed for {gain}: {readback}")
            if effort_limits is None or not np.allclose(effort_limits, 2.0, rtol=0.0, atol=1e-5):
                raise RuntimeError(f"wheel effort limits changed unexpectedly: {effort_limits}")

            group: dict[str, Any] = {
                "damping_readback_rad_s": readback,
                "effort_limit_readback_nm": effort_limits,
                "cases": {},
            }
            for case_name, action_values in CASES:
                adapter.reset(
                    seed=80_003,
                    options={
                        "mode": "goto",
                        "start": [50.0, 50.0, 0.0],
                        "goal": [52.0, 50.0],
                        "random_start": False,
                    },
                )
                action = np.asarray(action_values, dtype=np.float32)
                rows: list[dict[str, Any]] = []
                for step in range(1, STEPS_PER_CASE + 1):
                    adapter.step(action)
                    command = env.command[0].detach().cpu().numpy().astype(np.float64)
                    actual_body = np.asarray(adapter.robot[3:5], dtype=np.float64).copy()
                    target = [
                        float(
                            (command[0] + command[1] * (-1.0 if wheel.side == "left" else 1.0)
                             * ROBOT_SPEC.track_separation_m / 2.0)
                            / ROBOT_SPEC.wheel_radius_m
                        )
                        for wheel in ROBOT_SPEC.wheel_specs
                    ]
                    actual_all = _numpy(env.rover.data.joint_vel)
                    if actual_all.ndim == 2:
                        actual_all = actual_all[0]
                    rows.append({
                        "step": step,
                        "time_s": float(adapter.time),
                        "command_v_w": command.tolist(),
                        "actual_v_w": actual_body.tolist(),
                        "root_yaw_rad": float(adapter.robot[2]),
                        "actual_wheel_velocity_radps": actual_all[wheel_ids].astype(float).tolist(),
                        "target_wheel_velocity_radps": target,
                        "applied_wheel_torque_nm": _values(env.rover, "applied_torque", wheel_ids),
                        "computed_wheel_torque_nm": _values(env.rover, "computed_torque", wheel_ids),
                        "wheel_names": wheel_names,
                        "left_wheel_indices": left_ids,
                        "right_wheel_indices": right_ids,
                    })
                    if adapter.done:
                        break
                item = {
                    "action": list(action_values),
                    "summary": _summary(rows, float(cfg.dt)),
                    "steps": rows,
                }
                group["cases"][case_name] = item
                result["gains"][str(gain)] = group
                result["status"] = f"running:damping_{gain}:{case_name}"
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"damping": gain, "case": case_name, "summary": item["summary"]}), flush=True)
        result["status"] = "complete"
    finally:
        adapter.close()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
