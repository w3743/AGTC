"""Probe imported rover wheel-link geometry and stationary ground clearances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/skid_response/geometry_probe.json"
STATIC_STEPS = 20


def _np(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _rotation_from_quat(quaternions: np.ndarray, torch, matrix_from_quat) -> np.ndarray:
    q = torch.as_tensor(quaternions, dtype=torch.float32)
    return matrix_from_quat(q).detach().cpu().numpy()


def _wp_array_numpy(value, wp) -> np.ndarray:
    return wp.to_torch(value).detach().cpu().numpy()


def _shape_field(model, *candidates: str):
    for name in candidates:
        try:
            return name, getattr(model, name)
        except AttributeError:
            continue
    return None, None


def _quat_apply_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[3] * t + np.cross(xyz, t)


def _snapshot(env, adapter, ROBOT_SPEC, torch, wp, newton, matrix_from_quat, euler_xyz_from_quat) -> dict[str, Any]:
    rover = env.rover
    names = list(rover.body_names)
    poses = _np(rover.data.body_link_pose_w)[0].astype(np.float64)
    root = _np(rover.data.root_link_pose_w)[0].astype(np.float64)
    root_rotation = _rotation_from_quat(root[3:7][None, :], torch, matrix_from_quat)[0]
    wheel_rows = []
    body_indices = []
    for wheel in ROBOT_SPEC.wheel_specs:
        body_name = f"{wheel.name}_wheel_link"
        if body_name not in names:
            raise RuntimeError(f"imported rover is missing wheel body {body_name!r}")
        body_index = names.index(body_name)
        body_indices.append(body_index)
        pose = poses[body_index]
        rotation = _rotation_from_quat(pose[3:7][None, :], torch, matrix_from_quat)[0]
        relative_position = root_rotation.T @ (pose[:3] - root[:3])
        world_y_axis = rotation[:, 1]
        expected = np.asarray([wheel.x_m, wheel.y_m, wheel.z_m], dtype=np.float64)
        axis_dot_y = float(np.dot(world_y_axis, [0.0, 1.0, 0.0]))
        wheel_rows.append({
            "wheel": wheel.name,
            "side": wheel.side,
            "body_name": body_name,
            "body_index": body_index,
            "actual_world_xyz_m": pose[:3].tolist(),
            "actual_relative_base_xyz_m": relative_position.tolist(),
            "expected_relative_base_xyz_m": expected.tolist(),
            "relative_position_error_xyz_m": (relative_position - expected).tolist(),
            "body_quaternion_xyzw": pose[3:7].tolist(),
            "wheel_local_y_axis_world": world_y_axis.tolist(),
            "wheel_axis_dot_world_y": axis_dot_y,
            "wheel_axis_abs_angle_to_world_y_deg": float(np.degrees(np.arccos(np.clip(abs(axis_dot_y), -1.0, 1.0)))),
            "estimated_bottom_clearance_m": float(pose[2] - ROBOT_SPEC.wheel_radius_m),
        })
    if len(set(body_indices)) != len(body_indices):
        raise RuntimeError("multiple wheel specs resolved to the same imported rigid body")

    root_quat = torch.as_tensor(root[3:7][None, :], dtype=torch.float32, device=env.device)
    roll_t, pitch_t, yaw_t = euler_xyz_from_quat(root_quat)
    roll, pitch, yaw = [float(value[0].detach().cpu()) for value in (roll_t, pitch_t, yaw_t)]
    left_y = [row["actual_relative_base_xyz_m"][1] for row in wheel_rows if row["side"] == "left"]
    right_y = [row["actual_relative_base_xyz_m"][1] for row in wheel_rows if row["side"] == "right"]
    relative = np.asarray([row["actual_relative_base_xyz_m"] for row in wheel_rows])
    pairwise = np.linalg.norm(relative[:, None, :] - relative[None, :, :], axis=-1)
    pairwise[pairwise == 0] = np.inf
    data_public_contact_fields = [
        name for name in dir(rover.data)
        if not name.startswith("_") and any(token in name.lower() for token in ("contact", "normal_force"))
    ]
    scene_sensor_names = []
    try:
        scene_sensor_names = sorted(str(key) for key in env.scene.sensors.keys())
    except (AttributeError, TypeError):
        pass
    contact_sensors = [name for name in scene_sensor_names if "contact" in name.lower()]

    # Read the public Newton shape arrays only. In Newton's model, each shape
    # transform is local to its owning body and uses Warp's xyzw quaternion.
    model = env.sim.physics_manager.get_model()
    model_body_labels = list(model.body_label)
    shape_labels = list(model.shape_label)
    shape_body = _wp_array_numpy(model.shape_body, wp).astype(np.int64)
    shape_transform = _wp_array_numpy(model.shape_transform, wp).astype(np.float64)
    shape_type_field, shape_type_data = _shape_field(model, "shape_geo_type", "shape_type")
    shape_scale_field, shape_scale_data = _shape_field(model, "shape_geo_scale", "shape_scale")
    shape_types = _wp_array_numpy(shape_type_data, wp) if shape_type_data is not None else None
    shape_scales = _wp_array_numpy(shape_scale_data, wp) if shape_scale_data is not None else None
    body_label_matches = {}
    model_wheel_shapes = []
    for wheel in ROBOT_SPEC.wheel_specs:
        body_name = f"{wheel.name}_wheel_link"
        matches = [i for i, label in enumerate(model_body_labels) if str(label).endswith("/" + body_name)]
        body_label_matches[body_name] = [int(i) for i in matches]
        if len(matches) != 1:
            model_wheel_shapes.append({
                "wheel": wheel.name,
                "body_name": body_name,
                "matching_newton_body_ids": [int(i) for i in matches],
                "shapes": [],
            })
            continue
        body_id = matches[0]
        shape_ids = np.flatnonzero(shape_body == body_id)
        shape_rows = []
        for shape_id in shape_ids:
            xform = shape_transform[shape_id]
            local_quaternion_xyzw = xform[3:7]
            # Newton cylinder primitives use local Z. This rotation shows
            # whether USD's authored wheel-Y axis survived shape import.
            cylinder_axis_body = _quat_apply_xyzw(local_quaternion_xyzw, np.array([0.0, 0.0, 1.0]))
            cylinder_axis_world = rotation @ cylinder_axis_body
            geo_id = int(shape_types[shape_id]) if shape_types is not None else None
            try:
                geo_name = newton.GeoType(geo_id).name if geo_id is not None else None
            except (AttributeError, ValueError):
                geo_name = None
            shape_rows.append({
                "shape_index": int(shape_id),
                "shape_label": str(shape_labels[shape_id]),
                "shape_body_id": int(shape_body[shape_id]),
                "shape_type_field": shape_type_field,
                "shape_type_id": geo_id,
                "shape_type_name": geo_name,
                "shape_scale_field": shape_scale_field,
                "shape_scale": shape_scales[shape_id].tolist() if shape_scales is not None else None,
                "local_shape_translation_xyz": xform[:3].tolist(),
                "local_shape_quaternion_xyzw": local_quaternion_xyzw.tolist(),
                "canonical_cylinder_z_axis_in_body": cylinder_axis_body.tolist(),
                "canonical_cylinder_z_axis_in_world": cylinder_axis_world.tolist(),
                "canonical_cylinder_axis_abs_dot_body_y": float(abs(np.dot(cylinder_axis_body, [0.0, 1.0, 0.0]))),
                "canonical_cylinder_axis_abs_dot_world_y": float(abs(np.dot(cylinder_axis_world, [0.0, 1.0, 0.0]))),
            })
        model_wheel_shapes.append({
            "wheel": wheel.name,
            "body_name": body_name,
            "newton_body_id": int(body_id),
            "newton_body_label": str(model_body_labels[body_id]),
            "shapes": shape_rows,
        })

    return {
        "root_world_xyz_m": root[:3].tolist(),
        "root_quaternion_xyzw": root[3:7].tolist(),
        "root_roll_pitch_yaw_rad": [float(roll), float(pitch), float(yaw)],
        "task_pose_xy_yaw_v_w": np.asarray(adapter.robot, dtype=np.float64).tolist(),
        "body_names": names,
        "wheel_body_count": len(wheel_rows),
        "all_expected_wheel_bodies_present_once": len(wheel_rows) == 14 and len(set(body_indices)) == 14,
        "minimum_wheel_center_separation_m": float(np.min(pairwise)),
        "left_wheel_mean_relative_y_m": float(np.mean(left_y)),
        "right_wheel_mean_relative_y_m": float(np.mean(right_y)),
        "left_right_y_mirror_error_m": float(abs(np.mean(left_y) + np.mean(right_y))),
        "chassis_bottom_clearance_m": float(root[2] - ROBOT_SPEC.body_height_m / 2.0),
        "minimum_wheel_bottom_clearance_m": float(min(row["estimated_bottom_clearance_m"] for row in wheel_rows)),
        "wheel_links": wheel_rows,
        "newton_shape_array_fields": {
            "body_label_count": len(model_body_labels),
            "shape_label_count": len(shape_labels),
            "shape_type_field": shape_type_field,
            "shape_scale_field": shape_scale_field,
            "shape_transform_quaternion_convention": "xyzw",
            "wheel_body_label_matches": body_label_matches,
            "wheel_collision_shapes_by_body": model_wheel_shapes,
        },
        "public_articulation_contact_fields": data_public_contact_fields,
        "scene_sensor_names": scene_sensor_names,
        "contact_sensors_present": contact_sensors,
    }


def run(*, output: Path = OUTPUT) -> dict[str, Any]:
    import torch
    import warp as wp
    import newton
    from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=False,
        validate_scene=False,
        max_time=10.0,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        adapter.reset(
            seed=80_003,
            options={
                "mode": "goto",
                "start": [50.0, 50.0, 0.0],
                "goal": [52.0, 50.0],
                "random_start": False,
            },
        )
        before = _snapshot(env, adapter, ROBOT_SPEC, torch, wp, newton, matrix_from_quat, euler_xyz_from_quat)
        for _ in range(STATIC_STEPS):
            adapter.step(np.zeros(4, dtype=np.float32))
        after = _snapshot(env, adapter, ROBOT_SPEC, torch, wp, newton, matrix_from_quat, euler_xyz_from_quat)
        result = {
            "schema": "isaac-field-rover-geometry-probe-v1",
            "setup": {
                "backend": "Isaac Lab Newton MJWarp",
                "num_envs": 1,
                "single_simulation_context": True,
                "plants": 0,
                "flat_terrain": True,
                "validate_scene": False,
                "seed": 80_003,
                "start_xy_yaw": [50.0, 50.0, 0.0],
                "goal_xy": [52.0, 50.0],
                "dt_s": float(cfg.dt),
                "static_zero_action_steps": STATIC_STEPS,
                "static_duration_s": float(STATIC_STEPS * cfg.dt),
                "robot_wheel_specs": [
                    {"name": w.name, "side": w.side, "x_m": w.x_m, "y_m": w.y_m, "z_m": w.z_m}
                    for w in ROBOT_SPEC.wheel_specs
                ],
                "wheel_radius_m": float(ROBOT_SPEC.wheel_radius_m),
                "body_height_m": float(ROBOT_SPEC.body_height_m),
            },
            "immediately_after_reset": before,
            "after_static_steps": after,
            "status": "complete",
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": result["status"],
            "before": {key: before[key] for key in ("root_world_xyz_m", "root_roll_pitch_yaw_rad", "left_wheel_mean_relative_y_m", "right_wheel_mean_relative_y_m", "minimum_wheel_center_separation_m", "minimum_wheel_bottom_clearance_m")},
            "after": {key: after[key] for key in ("root_world_xyz_m", "root_roll_pitch_yaw_rad", "left_wheel_mean_relative_y_m", "right_wheel_mean_relative_y_m", "minimum_wheel_center_separation_m", "minimum_wheel_bottom_clearance_m")},
            "wheel_count": after["wheel_body_count"],
            "contact_sensors": after["contact_sensors_present"],
            "output": str(output),
        }, ensure_ascii=False), flush=True)
        return result
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(output=args.output)


if __name__ == "__main__":
    main()
