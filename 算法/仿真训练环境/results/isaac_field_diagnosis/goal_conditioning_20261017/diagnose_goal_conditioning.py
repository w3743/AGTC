#!/usr/bin/env python3
"""Capture native reset observations across controlled goto goal bearings.

This is a read-only diagnostic: it does not step the policy or physics after
reset, and does not load a checkpoint.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from isaac_field.env import SingleFieldAdapter, make_env
from isaac_field.task import Config


OUT = Path(__file__).resolve().parent
ANGLES_DEG = np.asarray([-90, -60, -30, 0, 30, 60, 90], dtype=np.float64)
RESET_SEED = 78021
START = np.asarray([50.0, 50.0, 0.0], dtype=np.float64)
GOAL_DISTANCE = 2.0


def main() -> None:
    cfg = Config(
        plants=0,
        terrain_enabled=False,
        max_time=20.0,
        damage_penalty=20.0,
        goal_progress_budget=5.0,
    )
    engine = make_env(
        cfg,
        mode="goto",
        num_envs=1,
        seed=RESET_SEED,
        auto_reset=False,
        goal_distance=(GOAL_DISTANCE, GOAL_DISTANCE),
        goal_angle=math.pi,
        distance_cost=0.0,
    )
    adapter = SingleFieldAdapter(engine)
    try:
        branch_names = ("lidar", "depth", "detections", "state")
        observations: dict[str, list[np.ndarray]] = {name: [] for name in branch_names}
        goals_xy, robot_state, pose_noise_xy, estimated_xy = [], [], [], []
        state_goal_xy, expected_goal_xy, state_goal_error = [], [], []
        infos, gimbals, modes = [], [], []
        for angle_deg in ANGLES_DEG:
            angle = math.radians(float(angle_deg))
            goal = START[:2] + GOAL_DISTANCE * np.asarray([math.cos(angle), math.sin(angle)])
            observation, info = adapter.reset(
                seed=RESET_SEED,
                options={
                    "mode": "goto",
                    "start": START.tolist(),
                    "goal": goal.tolist(),
                    "random_start": False,
                },
            )
            task = adapter.task
            actual_robot = np.asarray(task.robot, dtype=np.float64).copy()
            actual_goal = np.asarray(task.goal, dtype=np.float64).copy()
            noise = np.asarray(task._pose_noise, dtype=np.float64).copy()
            estimate = actual_robot[:2] + noise
            state = np.asarray(observation["state"], dtype=np.float32).copy()
            delta = actual_goal - estimate
            theta = float(actual_robot[2])
            expected = np.asarray([
                math.cos(theta) * delta[0] + math.sin(theta) * delta[1],
                -math.sin(theta) * delta[0] + math.cos(theta) * delta[1],
            ], dtype=np.float64) / cfg.size
            for name in branch_names:
                observations[name].append(np.asarray(observation[name]).copy())
            goals_xy.append(actual_goal)
            robot_state.append(actual_robot)
            pose_noise_xy.append(noise)
            estimated_xy.append(estimate)
            state_goal_xy.append(state[2:4].astype(np.float64))
            expected_goal_xy.append(expected)
            state_goal_error.append(state[2:4].astype(np.float64) - expected)
            infos.append(info)
            gimbals.append(np.asarray(task.gimbal, dtype=np.float64).copy())
            modes.append(str(task.mode))

        stacked = {name: np.stack(rows, axis=0) for name, rows in observations.items()}
        np.savez_compressed(
            OUT / "goal_observations.npz",
            angles_deg=ANGLES_DEG,
            goals_xy=np.asarray(goals_xy, dtype=np.float64),
            start_pose_xy_yaw=np.asarray(robot_state, dtype=np.float64)[:, :3],
            robot_state_xy_yaw_v_w=np.asarray(robot_state, dtype=np.float64),
            pose_noise_xy=np.asarray(pose_noise_xy, dtype=np.float64),
            estimated_xy=np.asarray(estimated_xy, dtype=np.float64),
            state_goal_xy=np.asarray(state_goal_xy, dtype=np.float64),
            expected_state_goal_xy=np.asarray(expected_goal_xy, dtype=np.float64),
            state_goal_error=np.asarray(state_goal_error, dtype=np.float64),
            **{f"obs_{name}": data for name, data in stacked.items()},
        )

        baseline = {name: data[0] for name, data in stacked.items()}
        branch_diffs = {}
        for name, data in stacked.items():
            if name == "state":
                mask = np.ones(data.shape[1], dtype=bool)
                mask[2:4] = False
                differences = np.abs(data[:, mask] - baseline[name][mask])
            else:
                differences = np.abs(data - baseline[name])
            branch_diffs[name] = {
                "shape": list(data.shape[1:]),
                "dtype": str(data.dtype),
                "max_abs_diff_from_angle_-90": float(differences.max(initial=0.0)),
                "exactly_equal_to_angle_-90_under_comparison": bool(
                    differences.max(initial=0.0) == 0.0),
                "comparison": "all elements" if name != "state" else "excluding state[2:4] goal encoding",
            }

        error = np.asarray(state_goal_error, dtype=np.float64)
        xy_state = stacked["state"][:, 4:6].astype(np.float64)
        estimated_from_state = (xy_state + 1.0) * (cfg.size / 2.0)
        actual_robot_array = np.asarray(robot_state, dtype=np.float64)
        estimated_xy_array = np.asarray(estimated_xy, dtype=np.float64)
        yaw = actual_robot_array[:, 2]
        signed_angle = np.deg2rad(ANGLES_DEG)
        expected_signs = {
            "state_forward_matches_projected_goal": bool(np.allclose(
                stacked["state"][:, 2], np.asarray(expected_goal_xy)[:, 0], atol=1e-6)),
            "state_left_matches_projected_goal": bool(np.allclose(
                stacked["state"][:, 3], np.asarray(expected_goal_xy)[:, 1], atol=1e-6)),
            "positive_bearing_gives_positive_left_component": bool(np.all(
                stacked["state"][[4, 5, 6], 3] > 0)),
            "negative_bearing_gives_negative_left_component": bool(np.all(
                stacked["state"][[0, 1, 2], 3] < 0)),
            "zero_bearing_left_component_matches_uwb_noise": bool(np.isclose(
                stacked["state"][3, 3], -np.asarray(pose_noise_xy)[3, 1] / cfg.size, atol=1e-6)),
            "yaw_is_fixed_at_requested_zero": bool(np.allclose(yaw, 0.0, atol=1e-6)),
            "actual_goals_match_requested_geometry": bool(np.allclose(
                np.asarray(goals_xy), START[None, :2] + GOAL_DISTANCE * np.stack(
                    (np.cos(signed_angle), np.sin(signed_angle)), axis=1), atol=1e-6)),
        }
        pose_noise_recon = estimated_from_state - actual_robot_array[:, :2]
        checks = {
            **expected_signs,
            "state_goal_encoding_max_abs_error": float(np.max(np.abs(error))),
            "state_xy_matches_pose_plus_recorded_noise_max_abs_error": float(np.max(
                np.abs(estimated_from_state - estimated_xy_array))),
            "state_goal_encoding_matches_actual_goal_and_noisy_pose": bool(np.max(np.abs(error)) < 1e-6),
            "state_xy_noise_encoding_matches_task_pose_noise": bool(np.max(np.abs(
                pose_noise_recon - np.asarray(pose_noise_xy))) < 1e-5),
            "fixed_start_matches_requested_xy_yaw": bool(np.allclose(
                actual_robot_array[:, :3], np.broadcast_to(START, (len(ANGLES_DEG), 3)), atol=1e-6)),
            "gimbal_identical_across_angles": bool(np.allclose(
                np.asarray(gimbals), np.broadcast_to(gimbals[0], np.asarray(gimbals).shape), atol=1e-7)),
            "mode_is_goto_for_all_angles": bool(all(mode == "goto" for mode in modes)),
            "reset_time_is_zero_for_all_angles": bool(all(abs(float(info["time"])) < 1e-12 for info in infos)),
        }
        manifest = {
            "schema": "isaac-field-goal-conditioning-reset-observations-v1",
            "status": "captured",
            "native_capture_only": True,
            "model_inference": False,
            "physics_steps_after_reset": 0,
            "environment": {
                "mode": "goto",
                "num_envs": 1,
                "environment_seed": RESET_SEED,
                "episode_reset_seed": RESET_SEED,
                "options_start_xy_yaw": START.tolist(),
                "random_start": False,
                "goal_distance_m": GOAL_DISTANCE,
                "goal_angles_deg": ANGLES_DEG.tolist(),
                "plant_count": int(len(adapter.task.scene.xy)),
                "terrain_enabled": bool(cfg.terrain_enabled),
                "max_time_s": cfg.max_time,
                "damage_penalty": cfg.damage_penalty,
                "goal_progress_budget": cfg.goal_progress_budget,
                "distance_cost": 0.0,
            },
            "observation_file": "goal_observations.npz",
            "observation_keys": {f"obs_{name}": branch_diffs[name] for name in branch_diffs},
            "state_goal_encoding": {
                "indices": [2, 3],
                "semantics": ["forward projected goal delta / map size", "left projected goal delta / map size"],
                "formula_uses": "actual task.goal and robot pose plus task._pose_noise (UWB position estimate)",
                "state_xy_indices": [4, 5],
                "state_xy_semantics": "estimated pose mapped from [0,size] to [-1,1]",
                "max_abs_encoding_error": float(np.max(np.abs(error))),
            },
            "per_angle": [
                {
                    "angle_deg": float(ANGLES_DEG[i]),
                    "goal_xy": np.asarray(goals_xy[i]).tolist(),
                    "robot_state_xy_yaw_v_w": np.asarray(robot_state[i]).tolist(),
                    "gimbal_yaw_pitch": np.asarray(gimbals[i]).tolist(),
                    "pose_noise_xy": np.asarray(pose_noise_xy[i]).tolist(),
                    "estimated_xy": np.asarray(estimated_xy[i]).tolist(),
                    "state_goal_xy": np.asarray(state_goal_xy[i]).tolist(),
                    "expected_state_goal_xy": np.asarray(expected_goal_xy[i]).tolist(),
                    "state_goal_error": np.asarray(state_goal_error[i]).tolist(),
                    "mode": modes[i],
                    "sensor_sequences": infos[i].get("sensor_sequences", {}),
                    "time_s": float(infos[i].get("time", math.nan)),
                }
                for i in range(len(ANGLES_DEG))
            ],
            "non_goal_observation_invariance": branch_diffs,
            "checks": checks,
            "interpretation": (
                "Branch invariance is reported as captured; only state[2:4] is treated as goal-conditioned. "
                "Any sensor/render differences are retained and not normalized away."
            ),
        }
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": manifest["status"],
            "observation_file": str(OUT / "goal_observations.npz"),
            "manifest_file": str(OUT / "manifest.json"),
            "branch_diffs": branch_diffs,
            "checks": checks,
            "sensor_sequences": [row["sensor_sequences"] for row in manifest["per_angle"]],
        }, indent=2))
        if any(not bool(value) for name, value in checks.items() if name != "state_goal_encoding_max_abs_error"
               and name != "state_xy_matches_pose_plus_recorded_noise_max_abs_error"):
            raise RuntimeError("one or more goal-conditioning validation checks failed; see manifest.json")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
