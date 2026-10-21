#!/usr/bin/env python3
"""Single-variable Newton/MuJoCo contact-source A/B; no training or FPS."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
OUTPUT_ROOT = EXPERIMENT_ROOT / "contact_probe_runs"
BASE_PROBE = ROOT / "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_transfer_probe.py"
TRIAL_PROTOCOL = ROOT / "results/isaac_field_improvement/fresh_side_pi_20261020/protocol.json"
SEED = 90231
START = [50.0, 50.0, 0.0]
GOAL = [90.0, 50.0]
SETTLE_S = 1.0
DRIVE_S = 8.0
ACTION = [0.10, -0.10, 0.0, 0.0]
EXPECTED_KP = 0.04
EXPECTED_KI = 0.1
EXPECTED_EFFORT_LIMIT_NM = 2.0
INITIAL_STATE_ABS_TOLERANCE = {
    "root_pose_world_xyzw": 1e-6,
    "root_velocity_world_mps_rad_s": 1e-6,
    "joint_position_rad": 1e-6,
    "joint_velocity_rad_s": 1e-6,
    "wheel_target_velocity_rad_s": 1e-6,
    "track_command_mps": 1e-6,
    "drive_pi_integral_nm": 1e-6,
    "drive_pi_feedforward_nm": 1e-6,
}


def load_base_probe():
    spec = importlib.util.spec_from_file_location("side_pi_contact_transfer_probe", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reusable probe helpers from {BASE_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_probe()
as_numpy = BASE.as_numpy
finite_array = BASE.finite_array
sha256_file = BASE.sha256_file
quat_rotate_x = BASE.quat_rotate_x
quat_rotate_y = BASE.quat_rotate_y


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def data_row(env, name: str, *, finite: bool = True) -> np.ndarray:
    value = as_numpy(getattr(env.rover.data, name))[0]
    return finite_array(value, f"rover.data.{name}") if finite else np.asarray(value)


def initial_state_snapshot(env) -> dict[str, Any]:
    """Capture reset state before settle/physics steps for cross-process pairing."""
    root_pose = finite_array(as_numpy(env.rover.data.root_link_pose_w)[0], "initial root pose")
    root_velocity = finite_array(as_numpy(env.rover.data.root_link_vel_w)[0], "initial root velocity")
    joint_position = data_row(env, "joint_pos")
    joint_velocity = data_row(env, "joint_vel")
    wheel_target = finite_array(as_numpy(env.wheel_target_velocity)[0], "initial wheel targets")
    track_command = finite_array(as_numpy(env.track_command)[0], "initial track commands")
    pi_integral = finite_array(as_numpy(env._drive_pi.integral)[0], "initial PI integral")
    pi_feedforward = finite_array(as_numpy(env._drive_pi.feedforward)[0], "initial PI feedforward")
    origin = finite_array(as_numpy(env.scene.env_origins)[0], "initial environment origin")
    return {
        "capture_point": "immediately after env.reset(seed, fixed start/goal), before any env.step or settling",
        "root_pose_world_xyzw": root_pose.tolist(),
        "root_velocity_world_mps_rad_s": root_velocity.tolist(),
        "joint_order": list(env.rover.joint_names),
        "joint_position_rad": joint_position.tolist(),
        "joint_velocity_rad_s": joint_velocity.tolist(),
        "wheel_target_velocity_rad_s": wheel_target.tolist(),
        "track_command_mps": track_command.tolist(),
        "drive_pi_integral_nm": pi_integral.tolist(),
        "drive_pi_feedforward_nm": pi_feedforward.tolist(),
        "environment_origin_world_m": origin.tolist(),
    }


def body_and_joint_properties(env) -> dict[str, Any]:
    body_names = list(env.rover.body_names)
    joint_names = list(env.rover.joint_names)
    masses = data_row(env, "body_mass")
    inertias = data_row(env, "body_inertia")
    com_poses = data_row(env, "body_com_pose_b")
    if len(masses.reshape(-1)) != len(body_names):
        raise RuntimeError(f"body mass count does not match link names: {masses.shape} vs {len(body_names)}")
    if inertias.shape[0] != len(body_names):
        raise RuntimeError(f"body inertia count does not match link names: {inertias.shape} vs {len(body_names)}")
    if com_poses.shape[0] != len(body_names):
        raise RuntimeError(f"body COM pose count does not match link names: {com_poses.shape} vs {len(body_names)}")
    masses_flat = masses.reshape(-1)
    body_rows = []
    for index, name in enumerate(body_names):
        raw_inertia = np.asarray(inertias[index]).reshape(-1)
        inertia_matrix = raw_inertia.reshape(3, 3).tolist() if raw_inertia.size == 9 else None
        body_rows.append({
            "body_index": index,
            "link_name": name,
            "mass_kg": float(masses_flat[index]),
            "inertia_raw": raw_inertia.tolist(),
            "inertia_raw_shape": list(np.asarray(inertias[index]).shape),
            "inertia_matrix_3x3": inertia_matrix,
            "center_of_mass_pose_in_link_frame_xyzw": np.asarray(com_poses[index]).reshape(-1).tolist(),
        })

    joint_armature = data_row(env, "joint_armature")
    if joint_armature.reshape(-1).size != len(joint_names):
        raise RuntimeError(f"joint armature count does not match joint names: {joint_armature.shape} vs {len(joint_names)}")
    joint_armature_flat = joint_armature.reshape(-1)
    joint_rows = []
    for index, name in enumerate(joint_names):
        joint_rows.append({
            "joint_index": index,
            "joint_name": name,
            "armature_kg_m2": float(joint_armature_flat[index]),
            "armature_semantics": "solver joint armature only; not a full joint/link inertia measurement",
        })
    root_name = "base_link"
    root_index = body_names.index(root_name) if root_name in body_names else None
    return {
        "data_api": "Isaac Lab ArticulationData: body_mass, body_inertia, body_com_pose_b, joint_armature",
        "root_link_name": root_name,
        "root_link_properties": body_rows[root_index] if root_index is not None else None,
        "all_link_mass_inertia_com": body_rows,
        "all_joint_armature": joint_rows,
        "units_note": "body inertia is emitted exactly as exposed by ArticulationData; raw shape is recorded, and a 3x3 matrix is added only for a 9-element row",
    }


def optional_contact_rows(env, wheel_body_names: list[str]) -> dict[str, Any]:
    try:
        count, rows = BASE.wheel_pair_rows(env, wheel_body_names)
        return {"status": "available", "record_count": int(count), "wheel_rows": rows, "error": None}
    except Exception as exc:
        return {"status": "unavailable", "record_count": None, "wheel_rows": None,
                "error": f"{type(exc).__name__}: {exc}"}


def sample(env, wheel_names: list[str], sensor_ids: list[int], wheel_body_ids: list[int],
           wheel_sides: list[int], phase: str, phase_step: int, policy_step_global: int,
           decimation: int, physics_dt: float, policy_dt: float) -> dict[str, Any]:
    sensor = env.scene["wheel_contacts"]
    sensor_names = list(sensor.sensor_names or [])
    normal_rows: list[Any] | None = None
    friction_rows: list[Any] | None = None
    sensor_status = "available"
    sensor_error = None
    try:
        normal_force = as_numpy(sensor.data.net_normal_forces_w)[0]
        friction_force = as_numpy(sensor.data.net_friction_forces_w)[0]
        expected = (len(sensor_ids), 3)
        if normal_force.shape != expected or friction_force.shape != expected:
            raise RuntimeError(f"unexpected wheel contact sensor shapes normal={normal_force.shape}, friction={friction_force.shape}, expected={expected}")
        normal_force = finite_array(normal_force, "wheel normal force aggregate")
        friction_force = finite_array(friction_force, "wheel friction force aggregate")
        normal_rows, friction_rows = normal_force, friction_force
    except Exception as exc:
        sensor_status = "unavailable"
        sensor_error = f"{type(exc).__name__}: {exc}"

    body_pose = finite_array(as_numpy(env.rover.data.body_link_pose_w)[0], "body link pose")
    body_omega = finite_array(as_numpy(env.rover.data.body_link_ang_vel_w)[0], "body link angular velocity")
    joint_position = data_row(env, "joint_pos")
    joint_velocity = data_row(env, "joint_vel")
    wheel_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64).tolist()
    wheel_q = joint_position[wheel_ids]
    wheel_qd = joint_velocity[wheel_ids]
    wheel_target = finite_array(as_numpy(env.wheel_target_velocity)[0], "wheel target velocity")
    track_command = finite_array(as_numpy(env.track_command)[0], "track command")
    root_pose = finite_array(as_numpy(env.rover.data.root_link_pose_w)[0], "root link pose")
    root_velocity = finite_array(as_numpy(env.rover.data.root_link_vel_w)[0], "root link velocity")
    env_origin = finite_array(as_numpy(env.scene.env_origins)[0], "environment origin")
    root_forward = quat_rotate_x(root_pose[3:])
    root_omega = root_velocity[3:6]
    from isaac_field.assets import ROBOT_SPEC

    radius = float(ROBOT_SPEC.wheel_radius_m)
    width = float(ROBOT_SPEC.wheel_width_m)
    wheel_force_rows: list[dict[str, Any]] = []
    for index, (name, sensor_id, body_id, side) in enumerate(zip(wheel_names, sensor_ids, wheel_body_ids, wheel_sides)):
        pose = body_pose[body_id]
        axle_world = quat_rotate_y(pose[3:])
        vertical_support_extent = (
            radius * math.sqrt(max(0.0, 1.0 - float(axle_world[2]) ** 2))
            + 0.5 * width * abs(float(axle_world[2]))
        )
        local_center = pose[:3] - env_origin
        ground_z = float(env.terrain.height(local_center[:2]))
        wheel_gap = float(local_center[2] - vertical_support_extent - ground_z)
        relative_omega = body_omega[body_id] - root_omega
        if normal_rows is not None and friction_rows is not None:
            nf: list[float] | None = normal_rows[sensor_id].tolist()
            ff: list[float] | None = friction_rows[sensor_id].tolist()
            normal_up: float | None = float(normal_rows[sensor_id][2])
            friction_forward: float | None = float(np.dot(friction_rows[sensor_id], root_forward))
        else:
            nf = ff = normal_up = friction_forward = None
        wheel_force_rows.append({
            "wheel_body": name,
            "side_id": int(side),
            "sensor_row": int(sensor_id),
            "normal_force_world_aggregate_N": nf,
            "normal_force_world_up_component_N": normal_up,
            "friction_force_world_aggregate_N": ff,
            "friction_force_root_forward_projection_N": friction_forward,
            "wheel_joint_position_rad": float(wheel_q[index]),
            "wheel_joint_velocity_rad_s": float(wheel_qd[index]),
            "wheel_target_velocity_rad_s": float(wheel_target[index]),
            "wheel_center_world_m": pose[:3].tolist(),
            "wheel_orientation_world_xyzw": pose[3:].tolist(),
            "wheel_axis_world_y_unit": axle_world.tolist(),
            "wheel_body_spin_minus_root_projected_on_axle_rad_s": float(np.dot(relative_omega, axle_world)),
            "wheel_bottom_gap_to_base_terrain_m": wheel_gap,
            "gap_definition": "oriented finite-cylinder support extent minus the flat base terrain height; not solver separation/penetration",
        })

    pair_info = optional_contact_rows(env, wheel_names)
    return {
        "phase": phase,
        "phase_step": int(phase_step),
        "policy_step_global": int(policy_step_global),
        "simulation_time_s": float(policy_step_global * policy_dt),
        "contact_sensor_sample_period_s": float(policy_dt),
        "contact_sensor_sample_at_physics_iteration_end": int(policy_step_global * decimation),
        "sensor_update_timing": "one sample after env.step; final MJWarp decimation state for that policy tick",
        "normalized_action": ACTION if phase == "drive" else [0.0, 0.0, 0.0, 0.0],
        "track_command_mps": track_command.tolist(),
        "nominal_no_slip_yaw_command_rad_s": float(as_numpy(env.command)[0, 1]),
        "root_pose_world_xyzw": root_pose.tolist(),
        "root_linear_velocity_world_mps": root_velocity[:3].tolist(),
        "root_angular_velocity_world_rad_s": root_velocity[3:6].tolist(),
        "root_forward_world_unit": root_forward.tolist(),
        "measured_task_v_w": finite_array(env.tasks[0].robot[3:5], "task v/w").tolist(),
        "side_integral_state_nm": finite_array(as_numpy(env._drive_pi.integral)[0], "PI integral").tolist(),
        "wheel_force_scope": "per wheel body sum across all current contact counterparts; not a ground-only or per-pair force",
        "wheel_contact_sensor_status": sensor_status,
        "wheel_contact_sensor_error": sensor_error,
        "wheel_forces": wheel_force_rows,
        "contact_identity_api": pair_info,
        "pair_identity_semantics": "manifold identity and shape0-to-shape1 normal only; no per-record force is inferred",
    }


def compare_initial_states(current: dict[str, Any], peer_summary_path: Path) -> dict[str, Any]:
    if not peer_summary_path.is_file():
        return {"status": "peer_missing", "peer_summary": str(peer_summary_path), "matched": None}
    try:
        peer = json.loads(peer_summary_path.read_text(encoding="utf-8"))
        other = peer["initial_state_after_reset_before_settle"]
    except Exception as exc:
        return {"status": "peer_invalid", "peer_summary": str(peer_summary_path), "matched": False,
                "error": f"{type(exc).__name__}: {exc}"}
    deltas: dict[str, Any] = {}
    shape_matches = True
    all_within = True
    for field, tolerance in INITIAL_STATE_ABS_TOLERANCE.items():
        left = np.asarray(current[field], dtype=np.float64)
        right = np.asarray(other[field], dtype=np.float64)
        if left.shape != right.shape:
            shape_matches = False
            deltas[field] = {"shape_current": list(left.shape), "shape_peer": list(right.shape),
                             "max_abs_delta": None, "tolerance": tolerance, "pass": False}
            all_within = False
            continue
        delta = float(np.max(np.abs(left - right))) if left.size else 0.0
        passed = bool(np.isfinite(left).all() and np.isfinite(right).all() and delta <= tolerance)
        deltas[field] = {"shape": list(left.shape), "max_abs_delta": delta,
                         "tolerance": tolerance, "pass": passed}
        all_within = all_within and passed
    joint_order_matches = current["joint_order"] == other["joint_order"]
    return {
        "status": "compared",
        "peer_summary": str(peer_summary_path),
        "peer_contact_source": peer.get("contact_source"),
        "joint_order_matches": joint_order_matches,
        "shapes_match": shape_matches,
        "fields": deltas,
        "matched": bool(joint_order_matches and shape_matches and all_within),
        "note": "only reset state before any settling step is compared; subsequent root-z/contact differences are retained as effects of contact-source treatment",
    }


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("RUN_CONTACT_SOURCE_AB") != "1":
        raise SystemExit("Prepared only; set RUN_CONTACT_SOURCE_AB=1 after explicit parent authorization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-source", choices=("newton", "mujoco"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must be inside {OUTPUT_ROOT}") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "contact_source_ab_summary.json"
    samples_path = output_dir / "policy_step_contact_samples.jsonl"
    efforts_path = output_dir / "physics_substep_applied_effort.jsonl"
    initial_state: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "schema": "isaac-field-contact-source-ab-v1",
        "status": "running",
        "contact_source": args.contact_source,
        "purpose": "Single-variable current Newton contact pipeline versus native MuJoCo contact generation; no training/FPS/hardware claim.",
        "scope": "one env, plants=0, same generated flat Field mesh, seed/start/goal/action/controller/friction/effort cap; only use_mujoco_contacts changes",
        "seed": SEED,
        "start_xy_yaw": START,
        "goal_xy": GOAL,
        "settle_s": SETTLE_S,
        "drive_s": DRIVE_S,
        "action": ACTION,
        "output_paths": {"policy_step_samples": str(samples_path), "physics_substep_efforts": str(efforts_path)},
        "execution_guard": "Requires RUN_CONTACT_SOURCE_AB=1; output directory must be new and under contact_probe_runs/.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = None
    try:
        training_hashes, diagnostic_hashes = BASE.source_hashes()
        helper_path = Path(__file__).resolve()
        runner_path = EXPERIMENT_ROOT / "run_contact_source_ab.sh"
        summary.update({
            "training_source_snapshot_matches": True,
            "training_source_sha256": training_hashes,
            "training_protocol_sha256": diagnostic_hashes["training_protocol_sha256"],
            "training_source_file_count": len(training_hashes),
            "base_probe_sha256": sha256_file(BASE_PROBE),
            "new_helper_sha256": sha256_file(helper_path),
            "runner_sha256": sha256_file(runner_path) if runner_path.is_file() else None,
            "diagnostic_source_sha256": diagnostic_hashes,
            "training_protocol": str(TRIAL_PROTOCOL),
        })

        import torch
        import warp as wp
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_newton.physics import NewtonManager
        from isaac_field.assets import ROBOT_SPEC, ensure_robot_asset
        from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
        from isaac_field.task import Config

        cfg_task = Config(size=100.0, plants=0, layout="rows", terrain_enabled=False,
                          validate_scene=False, max_time=20.0)
        cfg = FieldEnvCfg(task=cfg_task, mode="search", auto_reset=False, seed=SEED)
        cfg.scene.num_envs = 1
        cfg.scene.wheel_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
            update_period=0.0,
            history_length=1,
            track_friction_forces=True,
        )
        expected_mujoco_contacts = args.contact_source == "mujoco"
        cfg.sim.physics.solver_cfg.use_mujoco_contacts = expected_mujoco_contacts
        env = IsaacFieldEnv(cfg)

        physics, solver, shape = cfg.sim.physics, cfg.sim.physics.solver_cfg, cfg.sim.physics.default_shape_cfg
        physics_config = {
            "physics_dt_s": float(cfg.sim.dt),
            "policy_decimation": int(cfg.decimation),
            "policy_dt_s": float(cfg.sim.dt * cfg.decimation),
            "use_cuda_graph": bool(physics.use_cuda_graph),
            "use_mujoco_contacts": bool(solver.use_mujoco_contacts),
            "solver_njmax": int(solver.njmax),
            "solver_nconmax": int(solver.nconmax),
            "solver_integrator": str(solver.integrator),
            "shape_margin_m": float(shape.margin),
            "shape_gap_m": float(shape.gap),
            "shape_mu": float(shape.mu),
            "collision_cfg_present": getattr(physics, "collision_cfg", None) is not None,
        }
        if physics_config["physics_dt_s"] != 0.005 or physics_config["policy_decimation"] != 10:
            raise RuntimeError(f"unexpected simulation timing: {physics_config}")
        if physics_config["use_mujoco_contacts"] is not expected_mujoco_contacts:
            raise RuntimeError(f"contact source preflight failed: {physics_config}")
        if physics_config["collision_cfg_present"]:
            raise RuntimeError(f"unexpected collision_cfg would confound/native path is disallowed: {physics_config}")
        if not math.isclose(physics_config["shape_mu"], 0.8, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"unexpected default friction: {physics_config}")

        drive_config = env.drive_config
        actuator = cfg.scene.robot.actuators["drive"]
        drive_checks = {
            "interface": drive_config.get("interface") == "independent_left_right_track_speed_explicit_pi_v6",
            "P_gain": math.isclose(float(drive_config.get("velocity_gain")), EXPECTED_KP, rel_tol=0.0, abs_tol=1e-12),
            "Ki_gain": math.isclose(float(drive_config.get("integral_gain_nm_per_proxy_wheel_per_rad_s_s")), EXPECTED_KI, rel_tol=0.0, abs_tol=1e-12),
            "actuator_effort_limit": math.isclose(float(actuator.actuator_effort_limit), EXPECTED_EFFORT_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12),
            "joint_effort_limit": math.isclose(float(actuator.joint_effort_limit), EXPECTED_EFFORT_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12),
        }
        if not all(drive_checks.values()):
            raise RuntimeError(f"current drive/config preflight failed: {drive_checks}, {drive_config}")

        wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_body_names = [f"{wheel.name}_wheel_link" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64).tolist()
        actual_names = [env.rover.joint_names[index] for index in wheel_ids]
        if len(wheel_ids) != 14 or actual_names != wheel_names:
            raise RuntimeError(f"expected 14 wheel joints in current order: expected={wheel_names}, got={actual_names}")
        wheel_body_ids = [env.rover.body_names.index(name) for name in wheel_body_names]
        wheel_sides = as_numpy(env._wheel_track_ids).reshape(-1).astype(int).tolist()
        sensor = env.scene["wheel_contacts"]
        sensor_names = list(sensor.sensor_names or [])
        if len(sensor_names) != 14 or set(sensor_names) != set(wheel_body_names):
            raise RuntimeError(f"expected wheel contact sensor over exactly 14 bodies, got {sensor_names}")
        sensor_ids = [sensor_names.index(name) for name in wheel_body_names]

        field_fingerprint = hashlib.sha256(json.dumps(
            {"config": asdict(cfg_task), "seed": cfg.seed, "asset_revision": 8}, sort_keys=True
        ).encode("utf-8")).hexdigest()[:16]
        terrain_path = ROOT / "results/isaac_field/assets" / field_fingerprint / "terrain.usda"
        robot_asset_path = ensure_robot_asset(ROOT / "results/isaac_field/assets")
        if not terrain_path.is_file() or not robot_asset_path.is_file():
            raise FileNotFoundError(f"same-geometry assets unavailable: terrain={terrain_path}, robot={robot_asset_path}")

        invariant_config = {
            "task_config": asdict(cfg_task),
            "seed": SEED,
            "start": START,
            "goal": GOAL,
            "plants": 0,
            "field_geometry": {"terrain_usd_sha256": sha256_file(terrain_path),
                               "robot_usd_sha256": sha256_file(robot_asset_path),
                               "robot_spec": asdict(ROBOT_SPEC)},
            "sensor": {"prim_path": "{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
                       "update_period_s": float(cfg.scene.wheel_contacts.update_period),
                       "history_length": int(cfg.scene.wheel_contacts.history_length),
                       "track_friction_forces": bool(cfg.scene.wheel_contacts.track_friction_forces)},
            "simulation": {key: value for key, value in physics_config.items() if key != "use_mujoco_contacts"},
            "drive_config": drive_config,
            "drive_actuator": {"P": float(drive_config["velocity_gain"]),
                               "Ki": float(drive_config["integral_gain_nm_per_proxy_wheel_per_rad_s_s"]),
                               "actuator_effort_limit_nm": float(actuator.actuator_effort_limit),
                               "joint_effort_limit_nm": float(actuator.joint_effort_limit)},
            "action": ACTION,
            "settle_s": SETTLE_S,
            "drive_s": DRIVE_S,
        }
        invariant_hash = json_sha256(invariant_config)
        summary.update({
            "comparison_contract_sha256_excluding_contact_source": invariant_hash,
            "comparison_contract": invariant_config,
            "physics_config": physics_config,
            "drive_config": drive_config,
            "drive_preflight": drive_checks,
            "actuator_config": {"model": type(actuator).__name__,
                                "stiffness": float(actuator.stiffness),
                                "damping": float(actuator.damping),
                                "actuator_effort_limit_nm": float(actuator.actuator_effort_limit),
                                "joint_effort_limit_nm": float(actuator.joint_effort_limit),
                                "joint_velocity_limit_rad_s": float(actuator.joint_velocity_limit)},
            "asset_hashes": {"field_fingerprint": field_fingerprint,
                             "field_terrain_usd": str(terrain_path),
                             "field_terrain_usd_sha256": sha256_file(terrain_path),
                             "robot_asset_usd": str(robot_asset_path),
                             "robot_asset_usd_sha256": sha256_file(robot_asset_path),
                             "robot_asset_spec": asdict(ROBOT_SPEC),
                             "robot_asset_spec_sha256": json_sha256(asdict(ROBOT_SPEC))},
            "wheel_order": wheel_names,
            "wheel_sides": wheel_sides,
            "sensor_names": sensor_names,
            "sensor_ids_in_wheel_order": sensor_ids,
            "sampling_plan": {"settle_policy_steps": int(round(SETTLE_S / (cfg.sim.dt * cfg.decimation))),
                              "drive_policy_steps": int(round(DRIVE_S / (cfg.sim.dt * cfg.decimation))),
                              "contact_samples_hz": 1.0 / float(cfg.sim.dt * cfg.decimation),
                              "effort_substeps_per_policy_step": int(cfg.decimation),
                              "effort_samples_hz": 1.0 / float(cfg.sim.dt),
                              "surface": "original generated flat Field mesh; no added plane or replacement collider"},
            "body_joint_mass_inertia": body_and_joint_properties(env),
            "contact_api_expectation": {
                "manager": "NewtonManager.get_contacts()",
                "native_mujoco_contact_buffer": "MJWarp manager allocates Contacts from solver max contact count and solver.update_contacts populates it for sensor reporting",
                "identity_fields": ["rigid_contact_count", "rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_normal"],
                "force_scope": "per-wheel ContactSensor aggregate normal/friction vectors; no per-pair force claim",
            },
        })
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})
        initial_state = initial_state_snapshot(env)
        summary["initial_state_after_reset_before_settle"] = initial_state
        summary["initial_state_canonical_sha256"] = json_sha256(initial_state)
        peer_name = "mujoco_v1" if args.contact_source == "newton" else "newton_v1"
        peer_summary_path = OUTPUT_ROOT / peer_name / "contact_source_ab_summary.json"
        summary["paired_initial_state_check"] = compare_initial_states(initial_state, peer_summary_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        capture_kernel = __import__("runpy").run_path(str(ROOT / "tests/newton_effort_capture.py"))["record_wheel_applied_effort"]
        wheel_ids_warp = wp.from_torch(torch.as_tensor(wheel_ids, device=env.device, dtype=torch.int32).contiguous())
        applied_effort_warp = env.rover.actuators.applied_effort.warp
        decimation = int(cfg.decimation)
        policy_dt = float(cfg.sim.dt * decimation)
        physics_dt = float(cfg.sim.dt)
        settle_steps = int(round(SETTLE_S / policy_dt))
        drive_steps = int(round(DRIVE_S / policy_dt))
        expected_effort_count = (settle_steps + drive_steps) * decimation
        effort_tape = wp.empty((expected_effort_count, len(wheel_ids)), dtype=wp.float32, device=env.device)
        effort_counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_after_native_actuator_step() -> None:
            wp.launch(capture_kernel, dim=1,
                      inputs=[applied_effort_warp, wheel_ids_warp, effort_tape, effort_counter, expected_effort_count],
                      device=env.device)

        NewtonManager.register_post_actuator_callback(capture_after_native_actuator_step)
        sample_count = 0
        callback_deltas: list[int] = []
        pair_api_unavailable_samples = 0
        sensor_api_unavailable_samples = 0
        max_effort = 0.0
        zero_action = torch.zeros((1, 4), device=env.device, dtype=torch.float32)
        drive_action = torch.as_tensor([ACTION], device=env.device, dtype=torch.float32)
        with samples_path.open("x", encoding="utf-8") as sample_stream:
            previous_callback_count = 0
            for phase, action_tensor, phase_steps in (
                ("p_only_settle", zero_action, settle_steps),
                ("drive", drive_action, drive_steps),
            ):
                for phase_step in range(1, phase_steps + 1):
                    env.step(action_tensor)
                    sample_count += 1
                    callback_count = int(as_numpy(effort_counter).reshape(-1)[0])
                    callback_delta = callback_count - previous_callback_count
                    callback_deltas.append(callback_delta)
                    if callback_delta != decimation:
                        raise RuntimeError(f"post-actuator callback delta {callback_delta}, expected {decimation}")
                    previous_callback_count = callback_count
                    if phase == "p_only_settle" and phase_step == phase_steps:
                        env._drive_pi.reset(torch.tensor([0], device=env.device, dtype=torch.long))
                        env.rover.actuators.target_command.set_effort_index(
                            value=torch.zeros_like(env._drive_pi.feedforward), joint_ids=env._wheel_ids
                        )
                    row = sample(env, wheel_names, sensor_ids, wheel_body_ids, wheel_sides, phase, phase_step,
                                 sample_count, decimation, physics_dt, policy_dt)
                    pair_api_unavailable_samples += row["contact_identity_api"]["status"] != "available"
                    sensor_api_unavailable_samples += row["wheel_contact_sensor_status"] != "available"
                    policy_efforts = as_numpy(effort_tape)[max(0, callback_count-decimation):callback_count]
                    if policy_efforts.size:
                        max_effort = max(max_effort, float(np.max(np.abs(policy_efforts))))
                    sample_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    sample_stream.flush()
        captured_count = int(as_numpy(effort_counter).reshape(-1)[0])
        if captured_count != expected_effort_count:
            raise RuntimeError(f"effort callback captured {captured_count}, expected {expected_effort_count}")
        efforts = finite_array(as_numpy(effort_tape)[:captured_count], "native applied effort tape")
        max_effort = float(np.max(np.abs(efforts)))
        cap_pass = max_effort <= EXPECTED_EFFORT_LIMIT_NM + 1e-5
        with efforts_path.open("x", encoding="utf-8") as effort_stream:
            for slot, row in enumerate(efforts):
                policy_step = slot // decimation + 1
                phase = "p_only_settle" if policy_step <= settle_steps else "drive"
                phase_step = policy_step if phase == "p_only_settle" else policy_step - settle_steps
                effort_stream.write(json.dumps({
                    "physics_iteration_global": slot + 1,
                    "simulation_time_s": float((slot + 1) * physics_dt),
                    "policy_step_global": policy_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "physics_substep_in_policy": slot % decimation + 1,
                    "applied_effort_nm_by_wheel_order": row.tolist(),
                }, ensure_ascii=False, allow_nan=False) + "\n")
        if not cap_pass:
            raise RuntimeError(f"applied effort exceeded 2 Nm cap: max={max_effort}")

        contact_pair_samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
        available_pair_rows = [row["contact_identity_api"]["wheel_rows"] or [] for row in contact_pair_samples]
        wheel_pair_counts = {name: 0 for name in wheel_body_names}
        wheel_ground_counts = {name: 0 for name in wheel_body_names}
        for sample_row in available_pair_rows:
            for pair in sample_row:
                name = pair["wheel_body"]
                if name in wheel_pair_counts:
                    wheel_pair_counts[name] += 1
                if pair["classification_from_counterpart_label"] == "ground_label_match" and name in wheel_ground_counts:
                    wheel_ground_counts[name] += 1
        summary.update({
            "status": "complete",
            "contact_sensor_samples_captured": sample_count,
            "contact_sensor_samples_expected": settle_steps + drive_steps,
            "contact_identity_api_unavailable_samples": int(pair_api_unavailable_samples),
            "wheel_contact_sensor_unavailable_samples": int(sensor_api_unavailable_samples),
            "effort_callback_deltas_per_policy_step": callback_deltas,
            "effort_callback_delta_values": sorted(set(callback_deltas)),
            "physics_substeps_expected_and_captured": [expected_effort_count, captured_count],
            "physics_effort_sample_period_s": physics_dt,
            "native_effort_max_abs_nm": max_effort,
            "native_effort_cap_nm": EXPECTED_EFFORT_LIMIT_NM,
            "native_effort_cap_pass": cap_pass,
            "all_effort_samples_finite": bool(np.isfinite(efforts).all()),
            "all_state_samples_finite": True,
            "contact_pair_records_across_time_not_unique_pairs": {
                "wheel_involving_manifold_rows": sum(wheel_pair_counts.values()),
                "wheel_ground_label_match_rows": sum(wheel_ground_counts.values()),
                "per_wheel_manifold_rows": wheel_pair_counts,
                "per_wheel_ground_label_match_rows": wheel_ground_counts,
            },
            "physical_interpretation_limit": "contact forces are per-wheel sensor aggregates in world coordinates; pair rows identify manifolds/normals only and are not independent unique pairs or per-pair forces",
            "source_hashes_unchanged_after_probe": True,
        })
        final_training_hashes, final_diagnostic_hashes = BASE.source_hashes()
        if final_training_hashes != training_hashes or final_diagnostic_hashes != diagnostic_hashes:
            raise RuntimeError("source hashes changed during the probe")
        if sha256_file(helper_path) != summary["new_helper_sha256"] or sha256_file(runner_path) != summary["runner_sha256"]:
            raise RuntimeError("new helper/runner hash changed during the probe")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "contact_source": args.contact_source,
                          "output_dir": str(output_dir), "contact_samples": sample_count,
                          "pair_api_unavailable_samples": pair_api_unavailable_samples,
                          "sensor_api_unavailable_samples": sensor_api_unavailable_samples,
                          "physics_effort_samples": captured_count, "max_abs_effort_nm": max_effort,
                          "cap_pass": cap_pass, "initial_state_sha256": summary["initial_state_canonical_sha256"],
                          "paired_initial_state_check": summary["paired_initial_state_check"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        if initial_state is not None:
            summary["initial_state_after_reset_before_settle"] = initial_state
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
