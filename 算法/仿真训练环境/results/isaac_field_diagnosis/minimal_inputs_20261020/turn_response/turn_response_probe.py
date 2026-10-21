#!/usr/bin/env python3
"""Read-only physical response probe for differential track commands."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import runpy

import numpy as np
import torch


def finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite physical readout: {result!r}")
    return result


def main() -> int:
    if os.environ.get("RUN_ISAAC_PHYSICS") != "1":
        raise SystemExit("Refusing Newton probe without RUN_ISAAC_PHYSICS=1")
    out_dir = Path(__file__).resolve().parent
    samples_path = out_dir / "turn_response_samples.jsonl"
    summary_path = out_dir / "turn_response_summary.json"
    if samples_path.exists() or summary_path.exists():
        raise SystemExit(f"Refusing to overwrite outputs in {out_dir}")

    helpers = runpy.run_path("tests/test_env_integration.py")
    make_env, step, to_numpy = helpers["_make_env"], helpers["_step"], helpers["_to_numpy"]
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import euler_xyz_from_quat

    source_paths = (
        "isaac_field/env.py",
        "isaac_field/task.py",
        "isaac_field/action_contract.py",
        "isaac_field/assets.py",
        "tests/test_env_integration.py",
    )
    source_sha256 = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in source_paths
    }
    cases = [
        {"name": f"diff_left_plus_{amp:.2f}", "action": [amp, -amp, 0.0, 0.0]}
        for amp in (0.05, 0.10, 0.20, 0.40, 0.80)
    ]
    cases.extend(
        [
            {"name": "diff_left_minus_0.20", "action": [-0.20, 0.20, 0.0, 0.0]},
            {"name": "diff_left_minus_0.80", "action": [-0.80, 0.80, 0.0, 0.0]},
            {"name": "straight_forward_0.20", "action": [0.20, 0.20, 0.0, 0.0]},
        ]
    )

    env = make_env(plants=0, max_time=20.0, terrain_enabled=False, seed=90231)
    summaries = []
    sample_count = 0
    try:
        dt = finite(env.step_dt)
        max_speed = finite(env.field_cfg.max_speed)
        acceleration = finite(env.field_cfg.acceleration)
        separation = finite(ROBOT_SPEC.track_separation_m)
        wheel_radius = finite(ROBOT_SPEC.wheel_radius_m)
        wheel_ids = env._wheel_ids
        side = np.asarray(to_numpy(env._wheel_track_ids), dtype=np.int64).reshape(-1)
        if len(side) != len(np.asarray(to_numpy(wheel_ids)).reshape(-1)):
            raise RuntimeError("wheel side map does not match wheel joint indices")
        left, right = side == 0, side == 1
        if int(left.sum()) != 7 or int(right.sum()) != 7:
            raise RuntimeError(f"expected 7 joints per side, got {left.sum()} and {right.sum()}")

        def read_state(previous_yaw, yaw_unwrapped):
            pose_t = env.rover.data.root_link_pose_w.torch
            pose = np.asarray(to_numpy(pose_t)[0], dtype=np.float64)
            angle_t = torch.stack(euler_xyz_from_quat(pose_t[:, 3:]), dim=-1)
            roll, pitch, yaw = np.asarray(to_numpy(angle_t)[0], dtype=np.float64)
            if previous_yaw is None:
                yaw_unwrapped = float(yaw)
            else:
                yaw_unwrapped += (float(yaw) - previous_yaw + math.pi) % (2 * math.pi) - math.pi
            origin = np.asarray(to_numpy(env.scene.env_origins)[0], dtype=np.float64)
            root_v = np.asarray(to_numpy(env.rover.data.root_com_vel_w.torch)[0], dtype=np.float64)
            robot = np.asarray(env.tasks[0].robot, dtype=np.float64)
            target = np.asarray(to_numpy(env.wheel_target_velocity)[0], dtype=np.float64)
            actual = np.asarray(to_numpy(env.rover.data.joint_vel.torch[:, wheel_ids])[0], dtype=np.float64)
            effort = np.asarray(
                to_numpy(env.rover.actuators.applied_effort.torch[:, wheel_ids])[0],
                dtype=np.float64,
            )
            track_command = np.asarray(to_numpy(env.track_command)[0], dtype=np.float64)
            state = {
                "task_time_s": finite(env.tasks[0].time),
                "position_xy_m": [finite(v) for v in robot[:2]],
                "root_z_m": finite(pose[2] - origin[2]),
                "roll_pitch_yaw_rad": [finite(roll), finite(pitch), finite(yaw)],
                "yaw_unwrapped_rad": finite(yaw_unwrapped),
                "world_linear_velocity_mps": [finite(v) for v in root_v[:3]],
                "world_angular_velocity_rad_s": [finite(v) for v in root_v[3:6]],
                "task_forward_speed_mps": finite(robot[3]),
                "task_yaw_rate_rad_s": finite(robot[4]),
                "track_command_mps": [finite(v) for v in track_command],
                "wheel_target_rad_s": [finite(v) for v in target],
                "wheel_actual_qd_rad_s": [finite(v) for v in actual],
                "wheel_applied_effort_nm": [finite(v) for v in effort],
                "left_target_mean_rad_s": finite(np.mean(target[left])),
                "right_target_mean_rad_s": finite(np.mean(target[right])),
                "left_actual_qd_mean_rad_s": finite(np.mean(actual[left])),
                "right_actual_qd_mean_rad_s": finite(np.mean(actual[right])),
                "left_applied_effort_mean_nm": finite(np.mean(effort[left])),
                "right_applied_effort_mean_nm": finite(np.mean(effort[right])),
            }
            return state, float(yaw), yaw_unwrapped

        with samples_path.open("w", encoding="utf-8", newline="\n") as samples:
            def write_row(row):
                nonlocal sample_count
                samples.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                sample_count += 1

            for case in cases:
                action = np.asarray(case["action"], dtype=np.float32)
                env.reset(
                    seed=90231,
                    options={
                        "mode": "search",
                        "start": [50.0, 50.0, 0.0],
                        "goal": [90.0, 50.0],
                    },
                )
                previous_yaw = None
                yaw_unwrapped = 0.0
                for index in range(1, round(1.0 / dt) + 1):
                    step(env, np.zeros(4, dtype=np.float32))
                    state, previous_yaw, yaw_unwrapped = read_state(previous_yaw, yaw_unwrapped)
                    write_row({
                        "case": case["name"],
                        "phase": "settle",
                        "step": index,
                        "policy_normalized_action": [0.0, 0.0, 0.0, 0.0],
                        "raw_action_track_speed_mps": [0.0, 0.0],
                        "nominal_yaw_rate_raw_action_rad_s": 0.0,
                        "nominal_yaw_rate_slewed_command_rad_s": finite(
                            (state["track_command_mps"][1] - state["track_command_mps"][0]) / separation
                        ),
                        **state,
                    })
                start = state
                drive = []
                raw_track = action[:2].astype(np.float64) * max_speed
                raw_nominal_yaw = (raw_track[1] - raw_track[0]) / separation
                for index in range(1, round(8.0 / dt) + 1):
                    step(env, action)
                    state, previous_yaw, yaw_unwrapped = read_state(previous_yaw, yaw_unwrapped)
                    drive.append(state)
                    write_row({
                        "case": case["name"],
                        "phase": "drive",
                        "step": index,
                        "policy_normalized_action": [finite(v) for v in action],
                        "raw_action_track_speed_mps": [finite(v) for v in raw_track],
                        "nominal_yaw_rate_raw_action_rad_s": finite(raw_nominal_yaw),
                        "nominal_yaw_rate_slewed_command_rad_s": finite(
                            (state["track_command_mps"][1] - state["track_command_mps"][0]) / separation
                        ),
                        **state,
                    })

                yaw_series = np.asarray([row["yaw_unwrapped_rad"] for row in drive], dtype=np.float64)
                yaw_start = float(start["yaw_unwrapped_rad"])
                yaw_steps = np.diff(np.concatenate(([yaw_start], yaw_series)))
                first_second, last_second = drive[:round(1.0 / dt)], drive[-round(1.0 / dt):]

                def mean_field(rows, field):
                    return finite(np.mean([row[field] for row in rows]))

                summary = {
                    "case": case["name"],
                    "policy_normalized_action": [finite(v) for v in action],
                    "duration_s": finite(len(drive) * dt),
                    "raw_track_command_mps": [finite(v) for v in raw_track],
                    "nominal_yaw_rate_raw_action_rad_s": finite(raw_nominal_yaw),
                    "end_slewed_track_command_mps": drive[-1]["track_command_mps"],
                    "nominal_yaw_rate_end_slewed_command_rad_s": finite(
                        (drive[-1]["track_command_mps"][1] - drive[-1]["track_command_mps"][0]) / separation
                    ),
                    "start_xy_m": start["position_xy_m"],
                    "end_xy_m": drive[-1]["position_xy_m"],
                    "net_xy_displacement_m": finite(np.linalg.norm(
                        np.asarray(drive[-1]["position_xy_m"]) - np.asarray(start["position_xy_m"])
                    )),
                    "start_yaw_unwrapped_rad": yaw_start,
                    "end_yaw_unwrapped_rad": drive[-1]["yaw_unwrapped_rad"],
                    "net_yaw_delta_unwrapped_rad": finite(drive[-1]["yaw_unwrapped_rad"] - yaw_start),
                    "cumulative_abs_yaw_path_rad": finite(np.sum(np.abs(yaw_steps))),
                    "mean_signed_actual_yaw_rate_first_1s_rad_s": mean_field(first_second, "task_yaw_rate_rad_s"),
                    "mean_signed_actual_yaw_rate_last_1s_rad_s": mean_field(last_second, "task_yaw_rate_rad_s"),
                    "mean_abs_actual_yaw_rate_last_1s_rad_s": finite(np.mean(
                        [abs(row["task_yaw_rate_rad_s"]) for row in last_second]
                    )),
                    "last_1s_track_side_means": {
                        field: mean_field(last_second, field)
                        for field in (
                            "left_target_mean_rad_s",
                            "right_target_mean_rad_s",
                            "left_actual_qd_mean_rad_s",
                            "right_actual_qd_mean_rad_s",
                            "left_applied_effort_mean_nm",
                            "right_applied_effort_mean_nm",
                        )
                    },
                    "last_1s_mean_abs_qd_tracking_error_rad_s": {
                        "left": finite(np.mean([
                            abs(row["left_target_mean_rad_s"] - row["left_actual_qd_mean_rad_s"])
                            for row in last_second
                        ])),
                        "right": finite(np.mean([
                            abs(row["right_target_mean_rad_s"] - row["right_actual_qd_mean_rad_s"])
                            for row in last_second
                        ])),
                    },
                    "root_z_range_m": [
                        finite(min(row["root_z_m"] for row in drive)),
                        finite(max(row["root_z_m"] for row in drive)),
                    ],
                    "max_abs_roll_pitch_rad": {
                        "roll": finite(max(abs(row["roll_pitch_yaw_rad"][0]) for row in drive)),
                        "pitch": finite(max(abs(row["roll_pitch_yaw_rad"][1]) for row in drive)),
                    },
                }
                summaries.append(summary)
                print("TURN RESPONSE CASE " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()

    summary_data = {
        "schema": "isaac-field-native-turn-response-v1",
        "purpose": "read-only Newton physical response diagnostic; not training, evaluation, or FPS",
        "environment": {
            "plants": 0,
            "terrain_enabled": False,
            "same_start_xy_yaw": [50.0, 50.0, 0.0],
            "goal_xy": [90.0, 50.0],
            "max_time_s": 20.0,
            "settle_s": 1.0,
            "drive_s": 8.0,
            "step_dt_s": dt,
            "max_track_speed_mps": max_speed,
            "track_acceleration_mps2": acceleration,
            "track_separation_m": separation,
            "wheel_radius_m": wheel_radius,
            "helper": "tests/test_env_integration.py::_make_env/_step",
        },
        "case_count": len(summaries),
        "sample_count": sample_count,
        "source_sha256": source_sha256,
        "cases": summaries,
        "interpretation_guard": [
            "No pass/fail threshold is imposed on low-amplitude turns.",
            "Compare sign symmetry, larger amplitudes, direct wheel tracking, body yaw, straight control, and roll/pitch before calling a response a defect.",
            "Applied effort is the simulator readout, not measured hardware motor torque.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary_data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "TURN RESPONSE SUMMARY "
        + json.dumps({
            "summary_path": str(summary_path),
            "samples_path": str(samples_path),
            "case_count": len(summaries),
            "sample_count": sample_count,
        }, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
