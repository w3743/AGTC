"""Compare only Newton contact-cone choices at fixed wheel damping/effort."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/isaac_field_diagnosis/skid_response/cone_sweep.json"
STEPS_PER_CASE = 40  # 2.0 s at the task's 0.05 s control period.
CONES = ("elliptic", "pyramidal")
CASES = (
    ("spin_turn_0.20", (0.0, 0.2, 0.0, 0.0)),
    ("spin_turn_0.70", (0.0, 0.7, 0.0, 0.0)),
    ("drive_v0.12_turn_0.20", (0.12, 0.2, 0.0, 0.0)),
)
SEED = 80_003
DRIVE_DAMPING = 20.0
EFFORT_LIMIT_NM = 2.0


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
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _summarize(rows: list[dict[str, Any]], dt: float) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    tail = rows[-10:]
    left_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("left_")]
    right_ids = [i for i, name in enumerate(first["wheel_names"]) if name.startswith("right_")]
    actual = np.asarray([r["actual_wheel_velocity_radps"] for r in tail], dtype=np.float64)
    target = np.asarray([r["target_wheel_velocity_radps"] for r in tail], dtype=np.float64)
    body = np.asarray([r["actual_v_w"] for r in tail], dtype=np.float64)
    cmd = np.asarray([r["command_v_w"] for r in tail], dtype=np.float64)
    # Joint efforts are actuator telemetry for the implicit drive, not solver-reported torque.
    torque = [r["applied_wheel_torque_nm"] for r in tail]
    torque_mean = None
    if all(x is not None for x in torque):
        torque_mean = float(np.mean(np.abs(np.asarray(torque, dtype=np.float64))))
    yaw_rates = [
        _yaw_delta(rows[i - 1]["root_yaw_rad"], rows[i]["root_yaw_rad"]) / dt
        for i in range(1, len(rows))
    ]
    return {
        "steps": len(rows),
        "duration_s": float(len(rows) * dt),
        "steady_window_steps": len(tail),
        "root_yaw_delta_rad": float(_yaw_delta(first["root_yaw_rad"], last["root_yaw_rad"])),
        "root_yaw_rate_mean_radps": float(np.mean(yaw_rates)) if yaw_rates else 0.0,
        "root_yaw_rate_last10_mean_radps": float(np.mean(yaw_rates[-10:])) if yaw_rates else 0.0,
        "command_v_w_final": last["command_v_w"],
        "actual_v_w_final": last["actual_v_w"],
        "steady_mean_command_v_w": np.mean(cmd, axis=0).astype(float).tolist(),
        "steady_mean_actual_v_w": np.mean(body, axis=0).astype(float).tolist(),
        "steady_mean_target_wheel_radps_left": float(np.mean(target[:, left_ids])),
        "steady_mean_target_wheel_radps_right": float(np.mean(target[:, right_ids])),
        "steady_mean_actual_wheel_radps_left": float(np.mean(actual[:, left_ids])),
        "steady_mean_actual_wheel_radps_right": float(np.mean(actual[:, right_ids])),
        "steady_mean_abs_wheel_error_radps": float(np.mean(np.abs(actual - target))),
        "steady_mean_abs_implicit_effort_telemetry_nm": torque_mean,
    }


def _run_cone(cone: str) -> dict[str, Any]:
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import FieldEnvCfg, IsaacFieldEnv, SingleFieldAdapter
    from isaac_field.task import Config

    default_cone = FieldEnvCfg().sim.physics.solver_cfg.cone
    env_cfg = FieldEnvCfg(
        task=Config(
            size=100.0,
            plants=0,
            layout="rows",
            terrain_enabled=False,
            validate_scene=False,
            max_time=5.0,
        ),
        mode="goto",
        auto_reset=False,
        seed=SEED,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.sim.physics.solver_cfg.cone = cone
    if FieldEnvCfg().sim.physics.solver_cfg.cone != default_cone:
        raise RuntimeError("setting the instance cone unexpectedly mutated the default config")

    env = IsaacFieldEnv(env_cfg)
    adapter = SingleFieldAdapter(env)
    wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
    wheel_ids = [env.rover.joint_names.index(name) for name in wheel_names]
    try:
        env.rover.write_joint_damping_to_sim_index(damping=DRIVE_DAMPING, joint_ids=wheel_ids)
        damping_readback = _selected(env.rover, "joint_damping", wheel_ids)
        effort_readback = _selected(env.rover, "joint_effort_limits", wheel_ids)
        if damping_readback is None or not np.allclose(damping_readback, DRIVE_DAMPING, atol=1e-5, rtol=0.0):
            raise RuntimeError(f"damping readback mismatch for {cone}: {damping_readback}")
        if effort_readback is None or not np.allclose(effort_readback, EFFORT_LIMIT_NM, atol=1e-5, rtol=0.0):
            raise RuntimeError(f"effort readback mismatch for {cone}: {effort_readback}")

        cases: dict[str, Any] = {}
        for case_name, action_values in CASES:
            adapter.reset(
                seed=SEED,
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
                actual_all = _numpy(env.rover.data.joint_vel)
                if actual_all.ndim == 2:
                    actual_all = actual_all[0]
                wheel_targets = [
                    float(
                        (command[0] + command[1] * (-1.0 if wheel.side == "left" else 1.0)
                         * ROBOT_SPEC.track_separation_m / 2.0)
                        / ROBOT_SPEC.wheel_radius_m
                    )
                    for wheel in ROBOT_SPEC.wheel_specs
                ]
                rows.append({
                    "step": step,
                    "time_s": float(adapter.time),
                    "root_yaw_rad": float(adapter.robot[2]),
                    "actual_v_w": np.asarray(adapter.robot[3:5], dtype=np.float64).tolist(),
                    "command_v_w": command.tolist(),
                    "actual_wheel_velocity_radps": actual_all[wheel_ids].astype(float).tolist(),
                    "target_wheel_velocity_radps": wheel_targets,
                    "applied_wheel_torque_nm": _selected(env.rover, "applied_torque", wheel_ids),
                    "wheel_names": wheel_names,
                })
                if adapter.done:
                    break
            cases[case_name] = {
                "action": list(action_values),
                "summary": _summarize(rows, float(env.field_cfg.dt)),
                "steps": rows,
            }
            print(json.dumps({"cone": cone, "case": case_name, "summary": cases[case_name]["summary"]}), flush=True)
        return {
            "cone": cone,
            "friction_mu": float(env.cfg.sim.physics.default_shape_cfg.mu),
            "drive_damping_readback": damping_readback,
            "effort_limit_readback_nm": effort_readback,
            "cases": cases,
        }
    finally:
        adapter.close()


def run(output: Path = OUTPUT) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "isaac-field-cone-sweep-v1",
        "setup": {
            "one_context_per_cone": True,
            "context_closed_between_cones": True,
            "plants": 0,
            "flat_terrain": True,
            "seed": SEED,
            "start_xy_yaw": [50.0, 50.0, 0.0],
            "goal_xy": [52.0, 50.0],
            "duration_s": 2.0,
            "steps_per_case": STEPS_PER_CASE,
            "drive_damping": DRIVE_DAMPING,
            "wheel_effort_limit_nm": EFFORT_LIMIT_NM,
            "cases": {name: list(action) for name, action in CASES},
            "torque_note": "Implicit actuator effort is Isaac telemetry, not guaranteed solver-reported torque.",
        },
        "cones": {},
        "status": "running",
    }
    try:
        for cone in CONES:
            result["cones"][cone] = _run_cone(cone)
            result["status"] = f"complete:{cone}"
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["status"] = "complete"
    finally:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
