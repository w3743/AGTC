"""Measure wheel-target and chassis-yaw response for constant skid-steer inputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/skid_response"
STEPS_PER_CASE = 40  # 2 s at the task's 0.05 s control period.


def _numpy(value: Any):
    if value is None:
        return None
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _wheel_channel(rover, field: str, joint_ids: list[int]):
    try:
        value = _numpy(getattr(rover.data, field))
    except (AttributeError, RuntimeError, TypeError):
        return None
    if value is None:
        return None
    if value.ndim == 2:
        value = value[0]
    if value.ndim != 1 or max(joint_ids, default=-1) >= len(value):
        return None
    return value[joint_ids].astype(np.float64).tolist()


def _case_summary(records: list[dict[str, Any]], dt: float) -> dict[str, Any]:
    first, last = records[0], records[-1]
    yaw_delta = (last["root_yaw_rad"] - first["root_yaw_rad"] + math.pi) % (2 * math.pi) - math.pi
    wheel_actual = np.asarray([row["actual_wheel_velocity_radps"] for row in records], dtype=np.float64)
    wheel_target = np.asarray([row["target_wheel_velocity_radps"] for row in records], dtype=np.float64)
    body_commands = np.asarray([row["command_v_w"] for row in records], dtype=np.float64)
    actual_body = np.asarray([row["actual_v_w"] for row in records], dtype=np.float64)
    left_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("left_")]
    right_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("right_")]
    return {
        "steps": len(records),
        "duration_s": float(len(records) * dt),
        "root_yaw_start_rad": float(first["root_yaw_rad"]),
        "root_yaw_end_rad": float(last["root_yaw_rad"]),
        "root_yaw_delta_rad": float(yaw_delta),
        "root_yaw_rate_peak_abs_radps": float(max(abs(row["actual_v_w"][1]) for row in records)),
        "body_command_v_w_final": last["command_v_w"],
        "body_actual_v_w_final": last["actual_v_w"],
        "mean_wheel_target_radps_left": float(np.mean(wheel_target[:, left_ids])),
        "mean_wheel_target_radps_right": float(np.mean(wheel_target[:, right_ids])),
        "mean_wheel_actual_radps_left": float(np.mean(wheel_actual[:, left_ids])),
        "mean_wheel_actual_radps_right": float(np.mean(wheel_actual[:, right_ids])),
        "mean_abs_wheel_velocity_error_radps": float(np.mean(np.abs(wheel_actual - wheel_target))),
        "mean_abs_forward_command_error_mps": float(np.mean(np.abs(actual_body[:, 0] - body_commands[:, 0]))),
        "mean_abs_yaw_command_error_radps": float(np.mean(np.abs(actual_body[:, 1] - body_commands[:, 1]))),
    }


def run(*, device: str = "cuda:0", output: Path = OUTPUT) -> dict[str, Any]:
    import torch
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
    test_cases = [
        {"name": f"spin_v0_turn_{turn:+.2f}", "action": [0.0, turn, 0.0, 0.0]}
        for turn in (-0.7, -0.2, -0.05, 0.05, 0.2, 0.7)
    ] + [
        {"name": f"drive_v0.12_turn_{turn:+.2f}", "action": [0.12, turn, 0.0, 0.0]}
        for turn in (-0.2, 0.2)
    ]
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
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "isaac-field-skid-response-v1",
        "setup": {
            "backend": "Isaac Lab Newton MJWarp",
            "device": str(device),
            "plants": 0,
            "flat_terrain": True,
            "validate_scene": False,
            "scene_seed": 80_003,
            "reset_seed_per_case": 80_003,
            "start_xy_yaw": [50.0, 50.0, 0.0],
            "goal_xy": [52.0, 50.0],
            "dt_s": float(cfg.dt),
            "steps_per_case": STEPS_PER_CASE,
            "requested_duration_s": float(STEPS_PER_CASE * cfg.dt),
            "wheel_count": len(ROBOT_SPEC.wheel_specs),
            "wheel_radius_m": float(ROBOT_SPEC.wheel_radius_m),
            "track_separation_m": float(ROBOT_SPEC.track_separation_m),
            "action_definition": "[forward_norm, turn_norm, gimbal_yaw_norm, gimbal_pitch_norm]",
        },
        "cases": {},
        "status": "running",
    }
    wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
    joint_ids = [env.rover.joint_names.index(name) for name in wheel_names]
    report_path = output / "wheel_response.json"
    try:
        for case in test_cases:
            observation, reset_info = adapter.reset(
                seed=80_003,
                options={
                    "mode": "goto",
                    "start": [50.0, 50.0, 0.0],
                    "goal": [52.0, 50.0],
                    "random_start": False,
                },
            )
            rows: list[dict[str, Any]] = []
            action = np.asarray(case["action"], dtype=np.float32)
            for step in range(1, STEPS_PER_CASE + 1):
                adapter.step(action)
                command = env.command[0].detach().cpu().numpy().astype(np.float64)
                actual_body = np.asarray(adapter.robot[3:5], dtype=np.float64).copy()
                wheel_target = []
                for wheel in ROBOT_SPEC.wheel_specs:
                    side = -1.0 if wheel.side == "left" else 1.0
                    wheel_target.append(
                        float((command[0] + command[1] * side * ROBOT_SPEC.track_separation_m / 2.0)
                              / ROBOT_SPEC.wheel_radius_m)
                    )
                wheel_actual_all = _numpy(env.rover.data.joint_vel)
                if wheel_actual_all.ndim == 2:
                    wheel_actual_all = wheel_actual_all[0]
                wheel_actual = wheel_actual_all[joint_ids].astype(np.float64).tolist()
                applied_torque = _wheel_channel(env.rover, "applied_torque", joint_ids)
                computed_torque = _wheel_channel(env.rover, "computed_torque", joint_ids)
                rows.append({
                    "step": step,
                    "time_s": float(adapter.time),
                    "action_4": action.astype(float).tolist(),
                    "command_v_w": command.tolist(),
                    "root_yaw_rad": float(adapter.robot[2]),
                    "actual_v_w": actual_body.tolist(),
                    "actual_wheel_velocity_radps": wheel_actual,
                    "target_wheel_velocity_radps": wheel_target,
                    "wheel_names": wheel_names,
                    "applied_wheel_torque_nm": applied_torque,
                    "computed_wheel_torque_nm": computed_torque,
                    "distance_to_goal_m": float(np.linalg.norm(adapter.goal - adapter.robot[:2])),
                })
                if adapter.done:
                    break
            item = {
                "action": case["action"],
                "reset_info": {key: value for key, value in (reset_info or {}).items()
                               if isinstance(value, (str, int, float, bool, type(None)))},
                "summary": _case_summary(rows, float(cfg.dt)),
                "steps": rows,
            }
            result["cases"][case["name"]] = item
            result["status"] = f"running:{case['name']}"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"case": case["name"], "summary": item["summary"]}, ensure_ascii=False), flush=True)
        result["status"] = "complete"
    finally:
        adapter.close()
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(device=args.device, output=args.output)


if __name__ == "__main__":
    main()
