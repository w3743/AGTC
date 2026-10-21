#!/usr/bin/env python3
"""One-case, contact-instrumented side-PI diagnostic. Prepared only; no training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_probe_runs"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_numpy(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def finite_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite {name}")
    return result


def quat_rotate_y(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    return np.asarray([2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z),
                       2.0 * (y * z + x * w)], dtype=np.float64)


def quat_rotate_x(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    return np.asarray([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w),
                       2.0 * (x * z - y * w)], dtype=np.float64)


def source_hashes() -> tuple[dict[str, str], dict[str, str]]:
    """Hash training-frozen project sources and the diagnostic/API sources separately."""
    if not TRIAL_PROTOCOL.is_file():
        raise FileNotFoundError(f"fresh-side-PI training protocol is missing: {TRIAL_PROTOCOL}")
    protocol_bytes = TRIAL_PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    expected_rows = protocol.get("source_snapshot", {}).get("files", [])
    if not expected_rows:
        raise RuntimeError("training protocol has no source snapshot file list")
    training_expected = {row["path"]: row["sha256"] for row in expected_rows}
    training_actual = {
        relative: sha256_file(ROOT / relative)
        for relative in training_expected
    }
    mismatches = {
        path: {"expected": training_expected[path], "actual": training_actual[path]}
        for path in training_expected
        if training_actual[path] != training_expected[path]
    }
    if mismatches:
        raise RuntimeError(f"training source snapshot changed; refusing probe: {mismatches}")

    local_paths = [
        "isaac_field/env.py",
        "isaac_field/assets.py",
        "isaac_field/task.py",
        "isaac_field/scene_assets.py",
        "isaac_field/terrain.py",
        "tests/test_env_integration.py",
        "tests/newton_effort_capture.py",
        "scripts/isaaclab_python.sh",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor_data.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor_kernels.py",
        "external/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py",
        "external/IsaacLab/source/isaaclab_newton/test/sensors/test_contact_sensor.py",
        "external/IsaacLab/source/isaaclab_newton/test/physics/physics_test_utils.py",
        "scripts/diagnose_isaac_ground_contact.py",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_protocol.json",
        "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/physics_gap_diagnosis.md",
        "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_transfer_probe.py",
        "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/inspect_newton_contact_api.py",
    ]
    diagnostic_hashes = {path: sha256_file(ROOT / path) for path in local_paths}
    diagnostic_hashes["training_protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    diagnostic_hashes["training_source_file_count"] = str(len(training_expected))

    from newton import Contacts
    from newton.sensors import SensorContact

    diagnostic_hashes["runtime_newton_contacts_source_sha256"] = sha256_file(Path(__import__(
        "inspect").getfile(Contacts)))
    diagnostic_hashes["runtime_newton_sensor_contact_source_sha256"] = sha256_file(Path(__import__(
        "inspect").getfile(SensorContact)))
    return training_actual, diagnostic_hashes


def wheel_pair_rows(env, wheel_body_names: list[str]) -> tuple[int, list[dict[str, Any]]]:
    """Read Newton pair identities only; no per-pair force is inferred or serialized."""
    from isaaclab_newton.physics import NewtonManager

    contacts = NewtonManager.get_contacts()
    if contacts is None:
        raise RuntimeError("active Newton MJWarp path did not expose a Contacts buffer")
    count = int(as_numpy(contacts.rigid_contact_count).reshape(-1)[0])
    shape0 = as_numpy(contacts.rigid_contact_shape0).reshape(-1)
    shape1 = as_numpy(contacts.rigid_contact_shape1).reshape(-1)
    normals = as_numpy(contacts.rigid_contact_normal).reshape(-1, 3)
    labels = list(getattr(NewtonManager.get_model(), "shape_label", []) or [])
    if not labels:
        raise RuntimeError("Newton model did not expose shape_label for contact-pair identity")

    def label(shape_id: int) -> str | None:
        return labels[shape_id] if 0 <= shape_id < len(labels) else None

    def wheel_for(shape_label: str | None) -> str | None:
        if shape_label is None:
            return None
        for body_name in wheel_body_names:
            if f"/{body_name}/" in shape_label or shape_label.endswith("/" + body_name):
                return body_name
        return None

    rows: list[dict[str, Any]] = []
    for contact_index in range(count):
        id0, id1 = int(shape0[contact_index]), int(shape1[contact_index])
        label0, label1 = label(id0), label(id1)
        wheel_name = wheel_for(label0) or wheel_for(label1)
        if wheel_name is None:
            continue
        counterpart = label1 if wheel_for(label0) else label0
        lowered = (counterpart or "").lower()
        if "ground" in lowered or "/terrain/" in lowered:
            classification = "ground_label_match"
        elif "/field/" in lowered and "/robot/" not in lowered:
            classification = "field_counterpart_unresolved"
        else:
            classification = "other_or_unresolved_counterpart"
        rows.append({
            "contact_record_index": contact_index,
            "wheel_body": wheel_name,
            "shape0_id": id0,
            "shape0_label": label0,
            "shape1_id": id1,
            "shape1_label": label1,
            "classification_from_counterpart_label": classification,
            "shape0_to_shape1_normal_unitless": finite_array(
                normals[contact_index], "Newton contact normal"
            ).tolist(),
        })
    return count, rows


def one_sample(env, wheel_names: list[str], sensor_ids: list[int], wheel_body_ids: list[int],
               wheel_sides: list[int], phase: str, phase_step: int, policy_step_global: int,
               decimation: int, physics_dt: float, policy_dt: float) -> dict[str, Any]:
    sensor = env.scene["wheel_contacts"]
    normal_force = as_numpy(sensor.data.net_normal_forces_w)[0]
    friction_force = as_numpy(sensor.data.net_friction_forces_w)[0]
    if normal_force.shape != (len(sensor_ids), 3) or friction_force.shape != (len(sensor_ids), 3):
        raise RuntimeError(
            f"unexpected contact sensor output shapes normal={normal_force.shape}, friction={friction_force.shape}"
        )
    normal_force = finite_array(normal_force, "wheel normal-force aggregate")
    friction_force = finite_array(friction_force, "wheel friction-force aggregate")
    body_pose = finite_array(
        as_numpy(env.rover.data.body_link_pose_w)[0], "body_link_pose_w"
    )
    joint_qd = finite_array(
        as_numpy(env.rover.data.joint_vel)[0, env._wheel_ids.detach().cpu().numpy()], "wheel joint velocity"
    )
    wheel_target = finite_array(as_numpy(env.wheel_target_velocity)[0], "wheel target velocity")
    track_command = finite_array(as_numpy(env.track_command)[0], "track command")
    root_pose = finite_array(as_numpy(env.rover.data.root_link_pose_w)[0], "root_link_pose_w")
    root_velocity = finite_array(as_numpy(env.rover.data.root_link_vel_w)[0], "root_link_vel_w")
    env_origin = finite_array(as_numpy(env.scene.env_origins)[0], "environment origin")
    root_forward = quat_rotate_x(root_pose[3:])
    wheel_force_rows: list[dict[str, Any]] = []
    from isaac_field.assets import ROBOT_SPEC

    radius = float(ROBOT_SPEC.wheel_radius_m)
    width = float(ROBOT_SPEC.wheel_width_m)
    for name, sensor_id, body_id, side in zip(wheel_names, sensor_ids, wheel_body_ids, wheel_sides):
        pose = body_pose[body_id]
        axle_world = quat_rotate_y(pose[3:])
        vertical_support_extent = (
            radius * math.sqrt(max(0.0, 1.0 - float(axle_world[2]) ** 2))
            + 0.5 * width * abs(float(axle_world[2]))
        )
        local_center = pose[:3] - env_origin
        ground_z = float(env.terrain.height(local_center[:2]))
        wheel_gap = float(local_center[2] - vertical_support_extent - ground_z)
        nf = normal_force[sensor_id]
        ff = friction_force[sensor_id]
        wheel_force_rows.append({
            "wheel_body": name,
            "side_id": int(side),
            "sensor_row": int(sensor_id),
            "normal_force_world_aggregate_N": nf.tolist(),
            "normal_force_world_up_component_N": float(nf[2]),
            "friction_force_world_aggregate_N": ff.tolist(),
            "friction_force_root_forward_projection_N": float(np.dot(ff, root_forward)),
            "wheel_joint_velocity_rad_s": float(joint_qd[wheel_names.index(name)]),
            "wheel_target_velocity_rad_s": float(wheel_target[wheel_names.index(name)]),
            "wheel_center_world_m": pose[:3].tolist(),
            "wheel_bottom_gap_to_base_terrain_m": wheel_gap,
            "gap_definition": "oriented finite-cylinder geometric support extent minus env.terrain.height; excludes Newton margin/gap and is not solver penetration",
        })
    pair_total, pair_rows = wheel_pair_rows(env, wheel_names)
    return {
        "phase": phase,
        "phase_step": int(phase_step),
        "policy_step_global": int(policy_step_global),
        "simulation_time_s": float(policy_step_global * policy_dt),
        "contact_sensor_sample_period_s": float(policy_dt),
        "contact_sensor_sample_at_physics_iteration_end": int(policy_step_global * decimation),
        "sensor_update_timing": "one read immediately after each env.step; MJWarp decimation has completed, so this is the final-state aggregate for that policy tick",
        "normalized_action": ACTION if phase == "drive" else [0.0, 0.0, 0.0, 0.0],
        "track_command_mps": track_command.tolist(),
        "nominal_no_slip_yaw_command_rad_s": float(as_numpy(env.command)[0, 1]),
        "root_pose_world_m_xyzw": root_pose.tolist(),
        "root_linear_velocity_world_mps": root_velocity[:3].tolist(),
        "root_angular_velocity_world_rad_s": root_velocity[3:6].tolist(),
        "body_forward_world_unit": root_forward.tolist(),
        "measured_task_v_w": finite_array(env.tasks[0].robot[3:5], "task v/w").tolist(),
        "side_integral_state_nm": finite_array(as_numpy(env._drive_pi.integral)[0], "PI integral").tolist(),
        "wheel_force_scope": "per-wheel-body sum over all current contact counterparts; not a ground-only or per-contact-pair force",
        "wheel_forces": wheel_force_rows,
        "newton_rigid_contact_record_count": int(pair_total),
        "wheel_contact_pair_identities_only": pair_rows,
        "pair_identity_semantics": "each row is a Newton contact manifold record with shape IDs/labels and shape0-to-shape1 normal; no per-record force is claimed",
    }


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("RUN_CONTACT_TRANSFER_PROBE") != "1":
        raise SystemExit("Prepared only; set RUN_CONTACT_TRANSFER_PROBE=1 after explicit parent authorization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must be inside {OUT_ROOT}") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "contact_transfer_probe_summary.json"
    samples_path = output_dir / "policy_step_contact_samples.jsonl"
    efforts_path = output_dir / "physics_substep_applied_effort.jsonl"

    summary: dict[str, Any] = {
        "schema": "isaac-field-contact-transfer-probe-v1",
        "status": "running",
        "purpose": "One fixed diff_left_plus_0.10 contact-transfer diagnostic; no training, no curriculum change, no measured hardware claim.",
        "scope": "single environment; plants=0; terrain disabled to retain the same flat base surface; one second zero-action settle then eight seconds normalized action [0.10,-0.10,0,0]",
        "case": "diff_left_plus_0.10",
        "seed": SEED,
        "start_pose_xy_yaw": START,
        "goal_xy": GOAL,
        "settle_s": SETTLE_S,
        "drive_s": DRIVE_S,
        "action": ACTION,
        "execution_guard": "This helper requires RUN_CONTACT_TRANSFER_PROBE=1; output directory must be new and under contact_probe_runs/.",
        "raw_paths": {
            "policy_step_contact_samples": str(samples_path),
            "physics_substep_applied_effort": str(efforts_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = None
    try:
        frozen_training_hashes, diagnosis_hashes = source_hashes()
        summary["training_source_snapshot_matches"] = True
        summary["training_source_sha256"] = frozen_training_hashes
        summary["diagnostic_source_sha256"] = diagnosis_hashes
        summary["training_protocol"] = str(TRIAL_PROTOCOL)

        import torch
        import warp as wp
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_newton.physics import NewtonManager
        from isaac_field.assets import ROBOT_SPEC
        from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
        from isaac_field.task import Config

        cfg_task = Config(
            size=100.0,
            plants=0,
            layout="rows",
            terrain_enabled=False,
            validate_scene=False,
            max_time=20.0,
        )
        cfg = FieldEnvCfg(task=cfg_task, mode="search", auto_reset=False, seed=SEED)
        cfg.scene.num_envs = 1
        cfg.scene.wheel_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_wheel_link",
            update_period=0.0,
            history_length=1,
            track_friction_forces=True,
        )
        env = IsaacFieldEnv(cfg)
        physics = cfg.sim.physics
        solver = physics.solver_cfg
        shape = physics.default_shape_cfg
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
        }
        if physics_config["physics_dt_s"] != 0.005 or physics_config["policy_decimation"] != 10:
            raise RuntimeError(f"unexpected simulation timing: {physics_config}")
        if physics_config["use_mujoco_contacts"] is not False:
            raise RuntimeError(f"expected frozen Newton collision-pipeline path use_mujoco_contacts=False: {physics_config}")
        if not math.isclose(physics_config["shape_mu"], 0.8, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"unexpected default contact friction: {physics_config}")

        drive_config = env.drive_config
        drive_actuator_cfg = cfg.scene.robot.actuators["drive"]
        checks = {
            "interface": drive_config.get("interface") == "independent_left_right_track_speed_explicit_pi_v6",
            "P_gain": math.isclose(float(drive_config.get("velocity_gain")), EXPECTED_KP, rel_tol=0.0, abs_tol=1e-12),
            "Ki_gain": math.isclose(float(drive_config.get("integral_gain_nm_per_proxy_wheel_per_rad_s_s")), EXPECTED_KI, rel_tol=0.0, abs_tol=1e-12),
            "actuator_limit": math.isclose(float(drive_actuator_cfg.actuator_effort_limit), EXPECTED_EFFORT_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12),
            "joint_limit": math.isclose(float(drive_actuator_cfg.joint_effort_limit), EXPECTED_EFFORT_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12),
        }
        if not all(checks.values()):
            raise RuntimeError(f"drive/config preflight failed: checks={checks}, drive={drive_config}")

        wheel_joint_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_body_names = [f"{wheel.name}_wheel_link" for wheel in ROBOT_SPEC.wheel_specs]
        wheel_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64).tolist()
        actual_joint_names = [env.rover.joint_names[index] for index in wheel_ids]
        if actual_joint_names != wheel_joint_names:
            raise RuntimeError(f"wheel joint order changed: expected={wheel_joint_names}, got={actual_joint_names}")
        wheel_body_ids = [env.rover.body_names.index(name) for name in wheel_body_names]
        wheel_sides = as_numpy(env._wheel_track_ids).reshape(-1).astype(int).tolist()
        sensor = env.scene["wheel_contacts"]
        sensor_names = list(sensor.sensor_names or [])
        if len(sensor_names) != 14 or set(sensor_names) != set(wheel_body_names):
            raise RuntimeError(f"expected contact sensor to match 14 wheel bodies, got {sensor_names}")
        sensor_ids = [sensor_names.index(name) for name in wheel_body_names]

        from inspect import getfile
        from newton import Contacts
        from newton.sensors import SensorContact

        summary["sensor_and_solver_api"] = {
            "active_solver": "Isaac Lab Newton MJWarp; collision pipeline enabled (use_mujoco_contacts=False)",
            "public_contacts_api": "NewtonManager.get_contacts(): rigid_contact_count, rigid_contact_shape0/1, rigid_contact_normal",
            "pair_force_policy": "pair identities and normals only; no per-pair force is emitted",
            "rigid_contact_force_note": "installed Contacts source marks rigid_contact_force as currently unused; SensorContact consumes the extended contacts.force spatial-vector field",
            "sensor_api": "ContactSensor.data.net_normal_forces_w and net_friction_forces_w, per sensing wheel body, world-frame aggregate N across all counterparts",
            "sensor_source": str(getfile(SensorContact)),
            "contacts_source": str(getfile(Contacts)),
            "filtered_matrix_policy": "filter matrix is not enabled; official Newton tests skip the use_mujoco_contacts=False filtered matrix path due non-deterministic reporting",
            "sensor_names": sensor_names,
            "sensor_indices_for_wheel_order": sensor_ids,
            "sensor_update_period_s": float(cfg.scene.wheel_contacts.update_period),
            "sensor_track_friction_forces": bool(cfg.scene.wheel_contacts.track_friction_forces),
        }
        summary["physics_config"] = physics_config
        summary["drive_config"] = drive_config
        summary["actuator_config"] = {
            "model": type(drive_actuator_cfg).__name__,
            "stiffness": float(drive_actuator_cfg.stiffness),
            "damping": float(drive_actuator_cfg.damping),
            "actuator_effort_limit_nm": float(drive_actuator_cfg.actuator_effort_limit),
            "joint_effort_limit_nm": float(drive_actuator_cfg.joint_effort_limit),
            "joint_velocity_limit_rad_s": float(drive_actuator_cfg.joint_velocity_limit),
        }
        summary["robot_asset_spec"] = asdict(ROBOT_SPEC)
        summary["wheel_order"] = wheel_joint_names
        summary["wheel_sides"] = wheel_sides
        summary["sampling_plan"] = {
            "contact_sensor_read_count_expected": 20 + 160,
            "contact_sensor_sample_rate_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
            "contact_sample_state": "one final-state sample after each env.step; contact record identities and aggregated sensor forces refer to the final MJWarp decimation state",
            "effort_callback": "tests/newton_effort_capture.py via NewtonManager.register_post_actuator_callback, after each native actuator call",
            "physics_effort_samples_expected": (20 + 160) * int(cfg.decimation),
            "physics_effort_sample_rate_hz": 1.0 / float(cfg.sim.dt),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})
        capture_kernel = runpy.run_path(str(ROOT / "tests/newton_effort_capture.py"))["record_wheel_applied_effort"]
        wheel_ids_warp = wp.from_torch(torch.as_tensor(wheel_ids, device=env.device, dtype=torch.int32).contiguous())
        applied_effort_warp = env.rover.actuators.applied_effort.warp
        decimation = int(cfg.decimation)
        policy_dt = float(cfg.sim.dt * decimation)
        physics_dt = float(cfg.sim.dt)
        expected_effort_count = (round(SETTLE_S / policy_dt) + round(DRIVE_S / policy_dt)) * decimation
        effort_tape = wp.empty((expected_effort_count, len(wheel_ids)), dtype=wp.float32, device=env.device)
        effort_counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_after_native_actuator_step() -> None:
            wp.launch(
                capture_kernel,
                dim=1,
                inputs=[applied_effort_warp, wheel_ids_warp, effort_tape, effort_counter, expected_effort_count],
                device=env.device,
            )

        NewtonManager.register_post_actuator_callback(capture_after_native_actuator_step)
        sample_count = 0
        callback_deltas: list[int] = []
        policy_samples: list[dict[str, Any]] = []
        zero_action = torch.zeros((1, 4), device=env.device, dtype=torch.float32)
        drive_action = torch.as_tensor([ACTION], device=env.device, dtype=torch.float32)
        with samples_path.open("x", encoding="utf-8") as sample_stream:
            previous_callback_count = 0
            for phase, action_tensor, phase_steps in (
                ("p_only_settle", zero_action, round(SETTLE_S / policy_dt)),
                ("drive", drive_action, round(DRIVE_S / policy_dt)),
            ):
                for phase_step in range(1, phase_steps + 1):
                    env.step(action_tensor)
                    sample_count += 1
                    callback_count = int(as_numpy(effort_counter).reshape(-1)[0])
                    callback_delta = callback_count - previous_callback_count
                    callback_deltas.append(callback_delta)
                    if callback_delta != decimation:
                        raise RuntimeError(
                            f"post-actuator callback count for policy step {sample_count} was {callback_delta}, expected {decimation}"
                        )
                    previous_callback_count = callback_count
                    if phase == "p_only_settle" and phase_step == phase_steps:
                        env._drive_pi.reset(torch.tensor([0], device=env.device, dtype=torch.long))
                        env.rover.actuators.target_command.set_effort_index(
                            value=torch.zeros_like(env._drive_pi.feedforward), joint_ids=env._wheel_ids
                        )
                    sample = one_sample(
                        env,
                        wheel_body_names,
                        sensor_ids,
                        wheel_body_ids,
                        wheel_sides,
                        phase,
                        phase_step,
                        sample_count,
                        decimation,
                        physics_dt,
                        policy_dt,
                    )
                    policy_samples.append(sample)
                    sample_stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    sample_stream.flush()

        captured_count = int(as_numpy(effort_counter).reshape(-1)[0])
        if captured_count != expected_effort_count:
            raise RuntimeError(f"effort callback captured {captured_count}, expected {expected_effort_count}")
        efforts = finite_array(as_numpy(effort_tape)[:captured_count], "native applied effort tape")
        maximum_effort = float(np.max(np.abs(efforts)))
        cap_pass = maximum_effort <= EXPECTED_EFFORT_LIMIT_NM + 1e-5
        with efforts_path.open("x", encoding="utf-8") as effort_stream:
            for slot, row in enumerate(efforts):
                policy_step = slot // decimation + 1
                phase = "p_only_settle" if policy_step <= round(SETTLE_S / policy_dt) else "drive"
                phase_step = policy_step if phase == "p_only_settle" else policy_step - round(SETTLE_S / policy_dt)
                effort_row = {
                    "physics_iteration_global": slot + 1,
                    "simulation_time_s": float((slot + 1) * physics_dt),
                    "policy_step_global": policy_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "physics_substep_in_policy": slot % decimation + 1,
                    "applied_effort_nm_by_wheel_order": row.tolist(),
                }
                effort_stream.write(json.dumps(effort_row, ensure_ascii=False, allow_nan=False) + "\n")
        if not cap_pass:
            raise RuntimeError(f"native per-wheel applied effort cap failed: max={maximum_effort} Nm")

        pair_count_total = sum(sample["newton_rigid_contact_record_count"] for sample in policy_samples)
        wheel_pair_rows_total = sum(len(sample["wheel_contact_pair_identities_only"]) for sample in policy_samples)
        pair_rows_per_wheel: dict[str, int] = {name: 0 for name in wheel_body_names}
        ground_label_rows_per_wheel: dict[str, int] = {name: 0 for name in wheel_body_names}
        field_unresolved_rows_per_wheel: dict[str, int] = {name: 0 for name in wheel_body_names}
        for sample in policy_samples:
            for pair in sample["wheel_contact_pair_identities_only"]:
                name = pair["wheel_body"]
                pair_rows_per_wheel[name] += 1
                if pair["classification_from_counterpart_label"] == "ground_label_match":
                    ground_label_rows_per_wheel[name] += 1
                elif pair["classification_from_counterpart_label"] == "field_counterpart_unresolved":
                    field_unresolved_rows_per_wheel[name] += 1
        summary.update({
            "status": "complete",
            "contact_sensor_samples_captured": sample_count,
            "contact_sensor_sample_period_s": policy_dt,
            "effort_callback_deltas_per_policy_step": callback_deltas,
            "effort_callback_delta_values": sorted(set(callback_deltas)),
            "physics_substeps_expected_and_captured": [expected_effort_count, captured_count],
            "physics_effort_sample_period_s": physics_dt,
            "native_effort_max_abs_nm": maximum_effort,
            "native_effort_cap_nm": EXPECTED_EFFORT_LIMIT_NM,
            "native_effort_cap_pass": cap_pass,
            "all_effort_samples_finite": bool(np.isfinite(efforts).all()),
            "all_policy_sample_force_gap_and_state_values_finite": True,
            "contact_pair_counts_across_policy_samples": {
                "all_newton_rigid_contact_records": int(pair_count_total),
                "wheel_involving_contact_records": int(wheel_pair_rows_total),
                "per_wheel_records": pair_rows_per_wheel,
                "per_wheel_ground_label_matches": ground_label_rows_per_wheel,
                "per_wheel_field_unresolved_matches": field_unresolved_rows_per_wheel,
            },
            "physical_interpretation_limit": "normal/friction vectors are sensor-body aggregate in world frame; friction forward projection is not per-pair and is not asserted to be ground-only; pair identity rows have no associated force field",
            "next_step": "Interpret contact records and per-wheel aggregate vectors together; if shape labels cannot identify the field ground, retain them as unresolved rather than inventing pair semantics.",
        })
        final_source_hashes = {path: sha256_file(ROOT / path) for path in diagnosis_hashes if path in {
            "isaac_field/env.py", "isaac_field/assets.py", "isaac_field/task.py", "isaac_field/scene_assets.py",
            "isaac_field/terrain.py", "tests/test_env_integration.py", "tests/newton_effort_capture.py",
            "scripts/isaaclab_python.sh", "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
            "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py",
            "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py",
            "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor_data.py",
            "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor_kernels.py",
            "external/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py",
            "external/IsaacLab/source/isaaclab_newton/test/sensors/test_contact_sensor.py",
            "external/IsaacLab/source/isaaclab_newton/test/physics/physics_test_utils.py",
            "scripts/diagnose_isaac_ground_contact.py",
            "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/confirmation_protocol.json",
            "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/physics_gap_diagnosis.md",
            "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_transfer_probe.py",
            "results/isaac_field_diagnosis/side_pi_physics_gap_20261020/inspect_newton_contact_api.py",
        }}
        if final_source_hashes != {path: digest for path, digest in diagnosis_hashes.items() if path in final_source_hashes}:
            raise RuntimeError("one or more frozen diagnostic/project source files changed during the probe")
        summary["status"] = "complete"
        summary["source_hashes_unchanged_after_probe"] = True
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": summary["status"],
            "output_dir": str(output_dir),
            "contact_sensor_samples": sample_count,
            "physics_effort_samples": captured_count,
            "max_abs_effort_nm": maximum_effort,
            "cap_pass": cap_pass,
            "pair_count_summary": summary["contact_pair_counts_across_policy_samples"],
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
