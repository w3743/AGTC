"""Check whether reset plant poses persist through native zero-action steps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    import torch
    import warp as wp

    from isaac_field.env import make_env
    from isaac_field.task import Config

    cfg = Config(size=100.0, plants=16, terrain_enabled=False, validate_scene=False,
                 max_time=20.0, lidar_rays=400, depth_height=160, depth_width=88)
    env = make_env(cfg, mode="goto", num_envs=1, seed=80003, auto_reset=False,
                   goal_distance=(1.0, 3.0), goal_angle=np.deg2rad(60.0))
    output = Path("results/isaac_field_diagnosis/plant_pose_consistency/goto_viewer_reset_seeds_401_402_80003.json")
    try:
        model = env.sim.physics_manager.get_model()
        plant_ids = _numpy(env._plant_body_ids[0]).astype(int)
        body_q_default = wp.to_torch(model.body_q).detach().cpu().numpy()
        body_labels = list(model.body_label) if model.body_label is not None else []
        result = {
            "env_seed": 80003,
            "mode_config": "goto",
            "goal_distance_m": [1.0, 3.0],
            "goal_angle_deg": 60.0,
            "plant_count": 16,
            "reset_seeds": [401, 402, 80003],
            "zero_action_sample_steps": [0, 1, 2, 10],
            "plant_body_ids": plant_ids.tolist(),
            "plant_body_labels": [body_labels[i] for i in plant_ids] if body_labels else [],
            "model_body_q_default": body_q_default[plant_ids].tolist(),
            "resets": [],
        }
        action = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        for seed in result["reset_seeds"]:
            env.reset(seed=seed)
            task = env.tasks[0]
            origin = _numpy(env.scene.env_origins[0])
            expected = np.column_stack((task.scene.xy, task.scene.root_z)) + origin
            record = {"seed": seed, "task_mode": task.mode, "robot_initial": task.robot.tolist(),
                      "goal": task.goal.tolist(), "scene_xy": task.scene.xy.tolist(), "root_z": task.scene.root_z.tolist(),
                      "samples": []}
            previous_step = 0
            for step in result["zero_action_sample_steps"]:
                for _ in range(step - previous_step):
                    env.step(action)
                previous_step = step
                actual = _numpy(env.plant_positions())[0]
                error = actual - expected
                record["samples"].append({
                    "steps": step,
                    "actual_plant_xyz": actual.tolist(),
                    "max_xyz_error_m": float(np.linalg.norm(error, axis=1).max()),
                    "max_xy_error_m": float(np.linalg.norm(error[:, :2], axis=1).max()),
                    "max_abs_z_error_m": float(np.abs(error[:, 2]).max()),
                })
            result["resets"].append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output),
            "model_body_q_default_xyz": result["model_body_q_default"],
            "resets": [{"seed": item["seed"], "samples": [
                {key: sample[key] for key in ("steps", "max_xyz_error_m", "max_xy_error_m", "max_abs_z_error_m")}
                for sample in item["samples"]]} for item in result["resets"]],
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
