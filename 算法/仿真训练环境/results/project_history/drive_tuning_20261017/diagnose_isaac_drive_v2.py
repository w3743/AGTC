"""Short response check for full-range forward/differential track-speed v2."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/drive_v2/drive_v2_functional_probe.json"
SEED = 80_003
MU = 0.8
DRIVE_GAIN = 20.0
EFFORT_LIMIT_NM = 2.0
MAX_STEERING_TRACK_SPEED = 15.0 / 3.6
TRACK_ACCELERATION = 1.5
SEGMENTS = (
    ("positive_pure_turn", (0.0, 1.0, 0.0, 0.0), 60, True),
    ("negative_pure_turn", (0.0, -1.0, 0.0, 0.0), 60, True),
    ("forward_mix", (0.5, 0.2, 0.0, 0.0), 40, True),
    ("stop_after_forward_mix", (0.0, 0.0, 0.0, 0.0), 40, False),
)


def _numpy(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def _selected(rover, field: str, joint_ids: list[int]) -> list[float] | None:
    try:
        value = _numpy(getattr(rover.data, field))
    except (AttributeError, RuntimeError, TypeError):
        return None
    if value.ndim == 2:
        value = value[0]
    if value.ndim != 1 or max(joint_ids, default=-1) >= len(value):
        return None
    return value[joint_ids].astype(np.float64).tolist()


def _yaw_delta(start: float, end: float) -> float:
    return (end - start + np.pi) % (2.0 * np.pi) - np.pi


def _run_segment(adapter, env, name: str, action_values: tuple[float, ...], steps: int,
                 *, reset: bool, previous_track: np.ndarray | None, wheel_ids: list[int],
                 wheel_names: list[str], wheel_radius: float) -> dict[str, Any]:
    if reset:
        adapter.reset(
            seed=SEED,
            options={
                "mode": "goto",
                "start": [50.0, 50.0, 0.0],
                "goal": [52.0, 50.0],
                "random_start": False,
            },
        )
        previous_track = np.zeros(2, dtype=np.float64)
    elif previous_track is None:
        raise ValueError("a continuous segment needs its previous track target")

    start_yaw = float(adapter.robot[2])
    start_xy = np.asarray(adapter.robot[:2], dtype=np.float64).copy()
    action = np.asarray(action_values, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    slew_values: list[np.ndarray] = []
    for step in range(1, steps + 1):
        adapter.step(action)
        command = env.command[0].detach().cpu().numpy().astype(np.float64)
        track_target = _numpy(env.track_command)[0].astype(np.float64)
        slew = (track_target - previous_track) / float(env.field_cfg.dt)
        slew_values.append(slew.copy())
        previous_track = track_target.copy()
        wheel_actual = _numpy(env.rover.data.joint_vel)
        if wheel_actual.ndim == 2:
            wheel_actual = wheel_actual[0]
        root_xy = np.asarray(adapter.robot[:2], dtype=np.float64).copy()
        rows.append({
            "step": step,
            "time_s": float(adapter.time),
            "action": action.astype(float).tolist(),
            "root_xy_m": root_xy.tolist(),
            "root_yaw_rad": float(adapter.robot[2]),
            "task_measured_forward_and_world_yaw_rate": np.asarray(adapter.robot[3:5], dtype=np.float64).tolist(),
            "nominal_twist_from_tracks_mps_radps": command.tolist(),
            "target_track_speed_mps_left_right": track_target.astype(float).tolist(),
            "target_track_slew_mps2_left_right": slew.astype(float).tolist(),
            "actual_wheel_angular_speed_radps": wheel_actual[wheel_ids].astype(float).tolist(),
            "actual_wheel_surface_speed_mps": (wheel_actual[wheel_ids] * wheel_radius).astype(float).tolist(),
            "implicit_effort_telemetry_nm": _selected(env.rover, "applied_torque", wheel_ids),
            "wheel_names": wheel_names,
        })
        if adapter.done:
            break

    yaw_increments = [
        _yaw_delta(start_yaw, rows[0]["root_yaw_rad"]),
        *[_yaw_delta(rows[i - 1]["root_yaw_rad"], rows[i]["root_yaw_rad"])
          for i in range(1, len(rows))],
    ]
    yaw_rates = np.asarray(yaw_increments, dtype=np.float64) / float(env.field_cfg.dt)
    tail = rows[-10:]
    actual_tracks = np.asarray([r["actual_wheel_surface_speed_mps"] for r in tail], dtype=np.float64)
    left_ids = [i for i, joint_name in enumerate(wheel_names) if joint_name.startswith("left_")]
    right_ids = [i for i, joint_name in enumerate(wheel_names) if joint_name.startswith("right_")]
    wheel_groups = np.asarray([0 if joint_name.startswith("left_") else 1 for joint_name in wheel_names])
    target_tail = np.asarray([r["target_track_speed_mps_left_right"] for r in tail], dtype=np.float64)
    expected_wheel_surface = target_tail[:, wheel_groups]
    targets = np.asarray([r["target_track_speed_mps_left_right"] for r in rows], dtype=np.float64)
    slews = np.asarray(slew_values, dtype=np.float64)
    result = {
        "action": list(action_values),
        "steps": len(rows),
        "duration_s": float(len(rows) * env.field_cfg.dt),
        "start_xy_m": start_xy.tolist(),
        "end_xy_m": rows[-1]["root_xy_m"],
        "root_xy_displacement_m": float(np.linalg.norm(np.asarray(rows[-1]["root_xy_m"]) - start_xy)),
        "root_yaw_start_rad": start_yaw,
        "root_yaw_end_rad": rows[-1]["root_yaw_rad"],
        "root_yaw_increment_sum_rad": float(np.sum(yaw_increments)),
        "root_yaw_rate_mean_radps": float(np.mean(yaw_rates)),
        "root_yaw_rate_last10_mean_radps": float(np.mean(yaw_rates[-10:])),
        "task_measured_forward_and_world_yaw_rate_final": rows[-1]["task_measured_forward_and_world_yaw_rate"],
        "steady_mean_task_measured_forward_and_world_yaw_rate": np.mean(
            [r["task_measured_forward_and_world_yaw_rate"] for r in tail], axis=0
        ).astype(float).tolist(),
        "target_track_speed_final_mps_left_right": rows[-1]["target_track_speed_mps_left_right"],
        "max_abs_target_track_speed_mps": float(np.max(np.abs(targets))),
        "max_abs_target_track_slew_mps2": float(np.max(np.abs(slews))),
        "within_track_speed_limit": bool(np.max(np.abs(targets)) <= MAX_STEERING_TRACK_SPEED + 1e-6),
        "within_track_slew_limit": bool(np.max(np.abs(slews)) <= TRACK_ACCELERATION + 1e-5),
        "steady_mean_actual_wheel_surface_speed_mps_left": float(np.mean(actual_tracks[:, left_ids])),
        "steady_mean_actual_wheel_surface_speed_mps_right": float(np.mean(actual_tracks[:, right_ids])),
        "steady_mean_abs_wheel_surface_speed_error_mps": float(
            np.mean(np.abs(actual_tracks - expected_wheel_surface))
        ),
        "rows": rows,
    }
    return result


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
        seed=SEED,
        auto_reset=False,
        goal_distance=(2.0, 2.0),
        goal_angle=0.0,
        distance_cost=0.0,
    )
    adapter = SingleFieldAdapter(env)
    wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
    wheel_ids = [env.rover.joint_names.index(name) for name in wheel_names]
    try:
        damping = _selected(env.rover, "joint_damping", wheel_ids)
        effort = _selected(env.rover, "joint_effort_limits", wheel_ids)
        drive = env.drive_config
        if drive["interface"] != "forward_and_differential_track_speed_v2":
            raise AssertionError(f"unexpected drive interface: {drive}")
        if not np.allclose(damping, DRIVE_GAIN, atol=1e-5, rtol=0.0):
            raise AssertionError(f"drive damping does not match v2 config: {damping}")
        if not np.allclose(effort, EFFORT_LIMIT_NM, atol=1e-5, rtol=0.0):
            raise AssertionError(f"drive effort limit does not match v2 config: {effort}")
        if float(env.cfg.sim.physics.default_shape_cfg.mu) != MU:
            raise AssertionError("unexpected friction config")

        result: dict[str, Any] = {
            "schema": "isaac-field-drive-v2-functional-probe-v1",
            "setup": {
                "one_context": True,
                "plants": 0,
                "flat_terrain": True,
                "seed": SEED,
                "start_xy_yaw": [50.0, 50.0, 0.0],
                "duration_by_segment_steps": {name: steps for name, _, steps, _ in SEGMENTS},
                "dt_s": float(cfg.dt),
                "drive_config": drive,
                "friction_mu": MU,
                "cone": str(env.cfg.sim.physics.solver_cfg.cone),
                "drive_damping_readback": damping,
                "wheel_effort_limit_readback_nm": effort,
                "track_speed_limit_mps": MAX_STEERING_TRACK_SPEED,
                "track_slew_limit_mps2": TRACK_ACCELERATION,
                "wheel_radius_m": float(ROBOT_SPEC.wheel_radius_m),
                "track_separation_m": float(ROBOT_SPEC.track_separation_m),
                "torque_note": "Implicit actuator effort is Isaac telemetry, not guaranteed solver-reported torque.",
            },
            "segments": {},
            "status": "running",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        previous_track = None
        for name, action, steps, reset in SEGMENTS:
            item = _run_segment(
                adapter,
                env,
                name,
                action,
                steps,
                reset=reset,
                previous_track=previous_track,
                wheel_ids=wheel_ids,
                wheel_names=wheel_names,
                wheel_radius=float(ROBOT_SPEC.wheel_radius_m),
            )
            result["segments"][name] = item
            if name == "forward_mix":
                previous_track = _numpy(env.track_command)[0].astype(np.float64)
            result["status"] = f"running:{name}"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"segment": name, "summary": {k: v for k, v in item.items() if k != "rows"}}), flush=True)

        result["status"] = "complete"
        result["assertions"] = {
            "all_target_speeds_within_limit": all(
                value["within_track_speed_limit"] for value in result["segments"].values()
            ),
            "all_target_slews_within_limit": all(
                value["within_track_slew_limit"] for value in result["segments"].values()
            ),
            "positive_turn_yaw_positive": result["segments"]["positive_pure_turn"]["root_yaw_increment_sum_rad"] > 0.0,
            "negative_turn_yaw_negative": result["segments"]["negative_pure_turn"]["root_yaw_increment_sum_rad"] < 0.0,
            "stop_target_zero": bool(np.allclose(result["segments"]["stop_after_forward_mix"]["target_track_speed_final_mps_left_right"], 0.0, atol=1e-6)),
        }
        result["status"] = "complete"
        if not result["assertions"]["all_target_speeds_within_limit"]:
            result["status"] = "failed:track_speed_limit"
        if not result["assertions"]["all_target_slews_within_limit"]:
            result["status"] = "failed:track_slew_limit"
    finally:
        adapter.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
