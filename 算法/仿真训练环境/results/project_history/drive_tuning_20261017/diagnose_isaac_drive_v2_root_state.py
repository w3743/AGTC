"""Native state/contact probe for the drive-v2 speed discrepancy.

This is a read-only physics diagnostic. It records Isaac Lab/Newton's public root
pose/velocity and ContactSensorCfg measurements alongside finite-difference
position velocity and the task's reported forward speed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/drive_v2/root_state_consistency_probe.json"
SEED = 80_017
DT = 0.05
PHASES = (
    ("high_mix", (0.5, 0.2, 0.0, 0.0), 40),
    ("high_mix_stop", (0.0, 0.0, 0.0, 0.0), 80),
    ("low_mix", (0.12, 0.05, 0.0, 0.0), 40),
    ("low_mix_stop", (0.0, 0.0, 0.0, 0.0), 80),
)


def _numpy(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_numpy(value).reshape(-1)[0])


def _forward_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray((1.0 - 2.0 * (y * y + z * z),
                       2.0 * (x * y + w * z),
                       2.0 * (x * z - w * y)), dtype=np.float64)


def _sample(adapter, env, contact_sensor, wheel_ids: list[int], wheel_radius: float,
            previous_position: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat
    import torch

    pose = _numpy(env.rover.data.root_link_pose_w)[0].astype(np.float64)
    link_velocity = _numpy(env.rover.data.root_link_vel_w)[0].astype(np.float64)
    com_velocity = _numpy(env.rover.data.root_com_vel_w)[0].astype(np.float64)
    quat_t = torch.as_tensor(pose[3:], dtype=torch.float32, device=env.device).unsqueeze(0)
    rpy_t = euler_xyz_from_quat(quat_t)
    rpy = np.asarray([_scalar(value) for value in rpy_t], dtype=np.float64)
    rotation = _numpy(matrix_from_quat(quat_t))[0].astype(np.float64)
    forward = rotation[:, 0]
    joint_velocity = _numpy(env.rover.data.joint_vel)[0].astype(np.float64)
    wheel_radps = joint_velocity[wheel_ids]

    force_data = contact_sensor.data.net_forces_w
    if force_data is None:
        wheel_force = None
    else:
        wheel_force = _numpy(force_data)
        if wheel_force.ndim == 3:
            wheel_force = wheel_force[0]
        wheel_force = wheel_force.astype(np.float64)

    if previous_position is None:
        finite_difference_velocity = None
    else:
        finite_difference_velocity = (pose[:3] - previous_position) / DT

    row = {
        "time_s": float(adapter.time),
        "root_position_world_m": pose[:3].tolist(),
        "root_position_local_xy_m": np.asarray(adapter.robot[:2], dtype=np.float64).tolist(),
        "root_quaternion_world_xyzw": pose[3:].tolist(),
        "root_link_linear_velocity_world_mps": link_velocity[:3].tolist(),
        "root_link_angular_velocity_world_radps": link_velocity[3:].tolist(),
        "root_com_linear_velocity_world_mps": com_velocity[:3].tolist(),
        "root_com_angular_velocity_world_radps": com_velocity[3:].tolist(),
        "finite_difference_root_position_velocity_world_mps": (
            None if finite_difference_velocity is None else finite_difference_velocity.tolist()
        ),
        "body_roll_pitch_yaw_rad": rpy.tolist(),
        "body_roll_pitch_deg": np.rad2deg(rpy[:2]).tolist(),
        "task_reported_forward_speed_and_world_yaw_rate": np.asarray(adapter.robot[3:5], dtype=np.float64).tolist(),
        "root_com_velocity_projected_on_body_forward_mps": float(np.dot(com_velocity[:3], forward)),
        "root_link_velocity_projected_on_body_forward_mps": float(np.dot(link_velocity[:3], forward)),
        "actual_wheel_joint_velocity_radps": wheel_radps.tolist(),
        "actual_wheel_surface_speed_mps": (wheel_radps * wheel_radius).tolist(),
        "wheel_contact_net_force_world_N": None if wheel_force is None else wheel_force.tolist(),
        "wheel_contact_force_norm_N": None if wheel_force is None else np.linalg.norm(wheel_force, axis=1).tolist(),
        "wheel_contact_vertical_force_N": None if wheel_force is None else wheel_force[:, 2].tolist(),
        "task_done": bool(adapter.task.done),
        "task_reason": str(adapter.task.reason),
    }
    return row, pose[:3].copy()


def _run_phase(adapter, env, sensor, wheel_ids: list[int], wheel_radius: float,
               phase_name: str, action_values: tuple[float, ...], steps: int) -> dict[str, Any]:
    action = np.asarray(action_values, dtype=np.float32)
    previous_position = _numpy(env.rover.data.root_link_pose_w)[0, :3].astype(np.float64)
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    for step in range(steps):
        if adapter.task.done:
            break
        try:
            _, _, terminated, truncated, info = adapter.step(action)
        except RuntimeError as exc:
            if adapter.task.done:
                break
            raise
        row, previous_position = _sample(adapter, env, sensor, wheel_ids, wheel_radius, previous_position)
        row["phase_step"] = step + 1
        row["action"] = action.astype(float).tolist()
        row["terminated"] = bool(terminated)
        row["truncated"] = bool(truncated)
        row["info_reason"] = str(info.get("reason", ""))
        rows.append(row)

    if rows:
        link_velocity = np.asarray([r["root_link_linear_velocity_world_mps"] for r in rows], dtype=np.float64)
        fd_rows = [r["finite_difference_root_position_velocity_world_mps"] for r in rows]
        valid = np.asarray([value is not None for value in fd_rows], dtype=bool)
        fd = np.asarray([value for value in fd_rows if value is not None], dtype=np.float64)
        link_for_fd = link_velocity[valid]
        fd_error = fd - link_for_fd if len(fd) else np.empty((0, 3), dtype=np.float64)
        task_forward = np.asarray([r["task_reported_forward_speed_and_world_yaw_rate"][0] for r in rows])
        projected_com = np.asarray([r["root_com_velocity_projected_on_body_forward_mps"] for r in rows])
        forward_axes = np.asarray([_forward_from_xyzw(r["root_quaternion_world_xyzw"]) for r in rows])
        fd_forward = np.sum(fd * forward_axes[valid], axis=1) if len(fd) else np.empty(0, dtype=np.float64)
        first10 = min(10, len(rows))
        first10_task_forward = task_forward[:first10]
        first10_fd_forward = fd_forward[:first10]
        first10_fd_error = fd_error[:first10]
        roll_pitch = np.asarray([r["body_roll_pitch_yaw_rad"][:2] for r in rows], dtype=np.float64)
        actual_wheels = np.asarray([r["actual_wheel_surface_speed_mps"] for r in rows], dtype=np.float64)
        contact_norms = [r["wheel_contact_force_norm_N"] for r in rows]
        contact_norms_arr = (
            np.asarray([value for value in contact_norms if value is not None], dtype=np.float64)
            if any(value is not None for value in contact_norms) else np.empty((0, 0), dtype=np.float64)
        )
        max_roll_pitch = np.max(np.abs(roll_pitch), axis=0)
        summary = {
            "steps_completed": len(rows),
            "requested_steps": steps,
            "duration_s": len(rows) * DT,
            "start_root_position_world_m": rows[0]["root_position_world_m"],
            "end_root_position_world_m": rows[-1]["root_position_world_m"],
            "net_root_displacement_world_m": float(np.linalg.norm(
                np.asarray(rows[-1]["root_position_world_m"]) - np.asarray(rows[0]["root_position_world_m"])
            )),
            "maximum_abs_roll_pitch_deg": np.rad2deg(max_roll_pitch).tolist(),
            "rollover_threshold_rad": 1.2,
            "rollover_threshold_crossed": bool(np.any(max_roll_pitch > 1.2)),
            "task_done_at_end": bool(adapter.task.done),
            "task_reason_at_end": str(adapter.task.reason),
            "terminated_at_end": bool(terminated),
            "truncated_at_end": bool(truncated),
            "task_forward_vs_com_forward_projection_mean_abs_error_mps": float(np.mean(np.abs(task_forward - projected_com))),
            "first_10_steps_after_reset": {
                "steps": first10,
                "task_reported_forward_speed_mean_mps": float(np.mean(first10_task_forward)),
                "root_position_derivative_projected_on_forward_mean_mps": (
                    None if not len(first10_fd_forward) else float(np.mean(first10_fd_forward))
                ),
                "task_forward_vs_position_derivative_mean_abs_error_mps": (
                    None if not len(first10_fd_forward) else float(np.mean(np.abs(first10_task_forward - first10_fd_forward)))
                ),
                "position_derivative_vs_root_link_velocity_rms_error_mps": (
                    None if not len(first10_fd_error) else float(np.sqrt(np.mean(np.sum(first10_fd_error * first10_fd_error, axis=1))))
                ),
            },
            "task_forward_vs_position_derivative_mean_abs_error_mps": (
                None if not len(fd_forward) else float(np.mean(np.abs(task_forward[valid] - fd_forward)))
            ),
            "finite_difference_vs_root_link_velocity_rms_error_mps": (
                None if not len(fd_error) else float(np.sqrt(np.mean(np.sum(fd_error * fd_error, axis=1))))
            ),
            "finite_difference_vs_root_link_velocity_mean_abs_xyz_error_mps": (
                None if not len(fd_error) else np.mean(np.abs(fd_error), axis=0).tolist()
            ),
            "maximum_abs_task_forward_speed_mps": float(np.max(np.abs(task_forward))),
            "maximum_abs_root_link_linear_speed_mps": float(np.max(np.linalg.norm(link_velocity, axis=1))),
            "maximum_abs_wheel_surface_speed_mps": float(np.max(np.abs(actual_wheels))),
            "maximum_wheel_contact_force_norm_N": (
                None if not len(contact_norms_arr) else float(np.max(contact_norms_arr))
            ),
            "mean_total_wheel_vertical_contact_force_N": (
                None if not len(contact_norms_arr) else float(np.mean([
                    np.sum(r["wheel_contact_vertical_force_N"]) for r in rows
                    if r["wheel_contact_vertical_force_N"] is not None
                ]))
            ),
        }
    else:
        summary = {
            "steps_completed": 0,
            "requested_steps": steps,
            "duration_s": 0.0,
            "rollover_threshold_rad": 1.2,
            "rollover_threshold_crossed": False,
            "task_done_at_end": bool(adapter.task.done),
            "task_reason_at_end": str(adapter.task.reason),
            "terminated_at_end": bool(terminated),
            "truncated_at_end": bool(truncated),
        }
    return {"action": list(action_values), "summary": summary, "samples": rows}


def run(output: Path = OUTPUT) -> dict[str, Any]:
    from isaaclab.sensors import ContactSensorCfg
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import FieldEnvCfg, IsaacFieldEnv, SingleFieldAdapter
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=0,
        layout="rows",
        terrain_enabled=False,
        validate_scene=False,
        max_time=30.0,
    )
    env_cfg = FieldEnvCfg(task=cfg, mode="goto", auto_reset=False, seed=SEED)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.wheel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_wheel_link",
        update_period=0.0,
        history_length=1,
    )
    env = IsaacFieldEnv(env_cfg)
    adapter = SingleFieldAdapter(env)
    wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
    wheel_ids = [env.rover.joint_names.index(name) for name in wheel_names]
    sensor = env.scene["wheel_contacts"]
    result: dict[str, Any] = {
        "schema": "isaac-field-drive-v2-root-state-contact-probe-v1",
        "setup": {
            "one_context": True,
            "seed": SEED,
            "plants": 0,
            "flat_terrain": True,
            "episode_max_time_s": cfg.max_time,
            "control_dt_s": float(cfg.dt),
            "start_xy_yaw": [50.0, 50.0, 0.0],
            "goal_xy": [90.0, 50.0],
            "phases": [{"name": name, "action": list(action), "steps": steps, "duration_s": steps * DT}
                       for name, action, steps in PHASES],
            "contact_sensor_config": {
                "class": "isaaclab.sensors.ContactSensorCfg",
                "prim_path": env_cfg.scene.wheel_contacts.prim_path,
                "update_period_s": env_cfg.scene.wheel_contacts.update_period,
                "history_length": env_cfg.scene.wheel_contacts.history_length,
                "sensor_count": int(sensor.num_sensors),
                "sensor_names": list(sensor.sensor_names or []),
                "force_quantity": "net_forces_w in world frame, N; Newton ContactSensor total force",
            },
            "drive_config": env.drive_config,
            "wheel_names": wheel_names,
            "wheel_radius_m": float(ROBOT_SPEC.wheel_radius_m),
            "wheel_joint_ids": wheel_ids,
            "pose_quaternion_order": "xyzw",
        },
        "segments": {},
        "status": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if int(sensor.num_sensors) != 14:
            raise RuntimeError(f"expected 14 wheel body contact sensors, found {sensor.num_sensors}: {sensor.sensor_names}")
        for index, (name, action, steps) in enumerate(PHASES):
            if index in (0, 2):
                adapter.reset(
                    seed=SEED,
                    options={
                        "mode": "goto",
                        "start": [50.0, 50.0, 0.0],
                        "goal": [90.0, 50.0],
                        "random_start": False,
                    },
                )
            item = _run_phase(adapter, env, sensor, wheel_ids, float(ROBOT_SPEC.wheel_radius_m), name, action, steps)
            result["segments"][name] = item
            result["status"] = f"running:{name}"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"phase": name, "summary": item["summary"]}), flush=True)

        result["status"] = "complete"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        result["status"] = f"failed:{type(exc).__name__}:{exc}"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    run()
