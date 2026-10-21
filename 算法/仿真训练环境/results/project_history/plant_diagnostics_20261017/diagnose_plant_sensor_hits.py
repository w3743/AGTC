"""Record native camera instance labels and range-sensor hits for one reset."""

import json
from pathlib import Path

import numpy as np


def main():
    import torch
    import warp as wp

    from isaaclab.scene_data import SceneDataFormat
    from isaac_field.env import make_env
    from isaac_field.task import Config

    cfg = Config(size=100.0, plants=16, terrain_enabled=False, validate_scene=False,
                 max_time=20.0, lidar_rays=400, depth_height=160, depth_width=88)
    env = make_env(cfg, mode="goto", num_envs=1, seed=80003, auto_reset=False,
                   goal_distance=(1.0, 3.0), goal_angle=np.deg2rad(60.0))
    try:
        model = env.sim.physics_manager.get_model()
        plant_ids = env._plant_body_ids[0].detach().cpu().numpy().astype(int)
        model_default_q = wp.to_torch(model.body_q).detach().cpu().numpy()[plant_ids]
        provider = env.sim._scene_data_provider
        mapping = provider.create_mapping(list(model.body_label))
        env.reset(seed=80003)
        task = env.tasks[0]
        masks, metadata, rangefinder = env._sensor_cache["camera"]
        expected_world = np.column_stack((task.scene.xy, task.scene.root_z)) + env.scene.env_origins[0].detach().cpu().numpy()
        pose_samples = []
        action = torch.zeros((1, 4), dtype=torch.float32, device=env.device)
        previous_step = 0
        for step in (0, 1, 2, 10):
            for _ in range(step - previous_step):
                env.step(action)
            previous_step = step
            provider_poses = SceneDataFormat.Transform()
            provider_ok = provider.get_transforms(provider_poses, mapping=mapping, count=model.body_count)
            provider_q = wp.to_torch(provider_poses.transforms).detach().cpu().numpy()[plant_ids]
            actual_q = wp.to_torch(env.sim.physics_manager.get_state_0().body_q).detach().cpu().numpy()[plant_ids]
            pose_samples.append({
                "steps": step, "provider_ok": bool(provider_ok),
                "provider_max_error_m": float(np.linalg.norm(provider_q[:, :3] - expected_world, axis=1).max()),
                "actual_state_max_error_m": float(np.linalg.norm(actual_q[:, :3] - expected_world, axis=1).max()),
                "provider_vs_actual_max_error_m": float(np.linalg.norm(provider_q[:, :3] - actual_q[:, :3], axis=1).max()),
            })
        image = masks[0]
        ids, counts = np.unique(image, return_counts=True)
        labels = metadata.get("idToLabels", {})
        instances = [{"instance_id": int(instance), "pixel_count": int(count),
                      "label": labels.get(str(int(instance)), labels.get(int(instance), ""))}
                     for instance, count in zip(ids, counts)]
        lidar = task.frames["lidar"]
        depth = task.frames["depth"]
        camera_origin = np.asarray(task.frames["camera"]["origin"], dtype=float)
        forward, right, up = (np.asarray(axis, dtype=float) for axis in task.view_basis)
        plant_xyz = np.column_stack((task.scene.xy, task.scene.root_z))
        rel = plant_xyz - camera_origin
        result = {
            "seed": 80003,
            "config": {"mode": "goto", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0,
                       "plants": 16, "terrain_enabled": False, "reset_options": None},
            "task_mode": task.mode,
            "scene_xy": task.scene.xy.tolist(),
            "robot": {"x": float(task.robot[0]), "y": float(task.robot[1]), "yaw": float(task.robot[2])},
            "gimbal_yaw_pitch": task.gimbal.tolist(),
            "camera_origin_local": camera_origin.tolist(),
            "camera_basis_forward_right_up": [forward.tolist(), right.tolist(), up.tolist()],
            "plants_in_detector_camera_coordinates": [
                {"plant": i, "forward_m": float(np.dot(delta, forward)),
                 "right_m": float(np.dot(delta, right)), "up_m": float(np.dot(delta, up)),
                 "horizontal_angle_deg": float(np.degrees(np.arctan2(np.dot(delta, right), np.dot(delta, forward))))}
                for i, delta in enumerate(rel)
            ],
            "official_scene_data_provider": {
                "viewer_path_samples": pose_samples,
                "model_body_q_default": model_default_q.tolist(),
            },
            "camera_instance_segmentation": {
                "shape": list(image.shape),
                "labels": instances,
                "corn_instances": [entry for entry in instances if "/Corn_" in str(entry["label"])],
                "detections_after_label_matching": task.frames["camera"]["detections"],
                "idToLabels": labels,
            },
            "lidar_actual_hits": {
                "valid_count": int(lidar["valid"].sum()),
                "ray_ranges_m": lidar["ranges"].tolist(),
            },
            "rangefinder_camera_center_range_m": float(rangefinder[0, 0]),
            "depth_actual_hits": {
                "valid_pixels": int(depth["valid"].sum()),
                "min_valid_depth_m": float(depth["values"][depth["valid"]].min())
                if depth["valid"].any() else None,
            },
        }
        output = Path("results/isaac_field_diagnosis/plant_pose_consistency/goto_viewer_seed80003_sensor_hits.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({
            "output": str(output),
            "camera_corn_instances": result["camera_instance_segmentation"]["corn_instances"],
            "camera_detections": result["camera_instance_segmentation"]["detections_after_label_matching"],
            "robot": result["robot"],
            "provider": result["official_scene_data_provider"],
            "lidar_valid_count": result["lidar_actual_hits"]["valid_count"],
            "rangefinder_m": result["rangefinder_camera_center_range_m"],
            "depth_valid_pixels": result["depth_actual_hits"]["valid_pixels"],
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
