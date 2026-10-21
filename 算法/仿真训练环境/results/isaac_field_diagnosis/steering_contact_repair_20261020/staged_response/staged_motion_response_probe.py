#!/usr/bin/env python3
"""Record an eight-command native Newton motion matrix; no training or repair verdict."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020"
OUT_ROOT = EXPERIMENT / "staged_response/runs"
REUSE_PROBE = EXPERIMENT / "contact_source_ab_probe.py"
TEST_STEP_HELPER = ROOT / "tests/test_env_integration.py"
EFFORT_KERNEL = ROOT / "tests/newton_effort_capture.py"
SEED = 90231
START = [50.0, 50.0, 0.0]
GOAL = [90.0, 50.0]
POLICY_DT = 0.05
SETTLE_STEPS = 20
DRIVE_STEPS = 40
BRAKE_STEPS = 100
BRAKE_TAIL_STEPS = 6
EFFORT_CAP_NM = 2.0
DT_DECIMATION = {0.005: 10, 0.002: 25, 0.001: 50}

CASES = (
    {"name": "forward", "action": [0.12, 0.12, 0.0, 0.0], "velocity_sign": 1, "yaw_sign": 0},
    {"name": "reverse", "action": [-0.12, -0.12, 0.0, 0.0], "velocity_sign": -1, "yaw_sign": 0},
    {"name": "pivot_left_low", "action": [-0.10, 0.10, 0.0, 0.0], "velocity_sign": 0, "yaw_sign": 1},
    {"name": "pivot_right_low", "action": [0.10, -0.10, 0.0, 0.0], "velocity_sign": 0, "yaw_sign": -1},
    {"name": "arc_left", "action": [0.12, 0.24, 0.0, 0.0], "velocity_sign": 1, "yaw_sign": 1},
    {"name": "arc_right", "action": [0.24, 0.12, 0.0, 0.0], "velocity_sign": 1, "yaw_sign": -1},
    {"name": "pivot_right_high", "action": [0.40, -0.40, 0.0, 0.0], "velocity_sign": 0, "yaw_sign": -1},
    {"name": "pivot_left_high", "action": [-0.40, 0.40, 0.0, 0.0], "velocity_sign": 0, "yaw_sign": 1},
)


def load_reuse_probe():
    spec = importlib.util.spec_from_file_location("staged_contact_source_probe", REUSE_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing probe helpers: {REUSE_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REUSE = load_reuse_probe()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if value is None:
        return "NA"
    try:
        array = REUSE.as_numpy(value)
        if array.size == 1:
            item = array.reshape(-1)[0].item()
            return item if isinstance(item, (bool, int, float, str)) else str(item)
        return array.tolist()
    except Exception:
        try:
            return str(value)
        except Exception:
            return "NA"


def runtime_snapshot(cfg, env) -> dict:
    """Keep option/material fields that this Isaac/Newton version exposes; missing fields stay NA."""
    from isaaclab_newton.physics import NewtonManager

    shape = cfg.sim.physics.default_shape_cfg
    solver_cfg = cfg.sim.physics.solver_cfg
    solver = getattr(NewtonManager, "_solver", None)
    try:
        model = NewtonManager.get_model()
    except Exception:
        model = None
    host_opt = getattr(getattr(solver, "mj_model", None), "opt", None)
    warp_opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
    body_names = list(env.rover.body_names)
    root_index = body_names.index("base_link") if "base_link" in body_names else None
    try:
        root_mass = json_safe(env.rover.data.body_mass[0, root_index]) if root_index is not None else "NA"
        root_inertia = json_safe(env.rover.data.body_inertia[0, root_index]) if root_index is not None else "NA"
    except Exception:
        root_mass = root_inertia = "NA"
    try:
        armature = json_safe(env.rover.data.joint_armature[0])
    except Exception:
        armature = "NA"
    try:
        labels = getattr(model, "shape_label", None)
        shape_labels = list(labels) if labels is not None else "NA"
    except Exception:
        shape_labels = "NA"
    shape_materials = {
        key: json_safe(getattr(model, key, None))
        for key in ("shape_material_mu", "shape_material_kf", "shape_material_ke", "shape_material_kd",
                    "shape_margin", "shape_gap")
    }
    return {
        "physics_dt_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "policy_dt_s": float(cfg.sim.dt * cfg.decimation),
        "contact_source": "Newton collision pipeline (use_mujoco_contacts=False)",
        "shape_material_config": {key: json_safe(getattr(shape, key, None))
                                   for key in ("mu", "kf", "ke", "kd", "condim", "solref", "solimp", "margin", "gap")},
        "newton_runtime_shape_labels": shape_labels,
        "newton_runtime_shape_material_arrays_in_label_order": shape_materials,
        "solver_config": {key: json_safe(getattr(solver_cfg, key, None))
                          for key in ("cone", "impratio", "condim", "solref", "solimp", "iterations", "integrator", "use_mujoco_contacts")},
        "host_mj_model_opt": {key: json_safe(getattr(host_opt, key, None))
                               for key in ("cone", "impratio")} if host_opt is not None else "NA",
        "mjwarp_model_opt": {key: json_safe(getattr(warp_opt, key, None))
                             for key in ("cone", "impratio_invsqrt")} if warp_opt is not None else "NA",
        "root_link_mass_kg": root_mass,
        "root_link_inertia_api_value": root_inertia,
        "joint_armature_api_value": armature,
        "note": "Runtime fields are captured when exposed by this installed API; NA is not inferred from config.",
    }


def validate_ground_wheel_mu(physics: dict, expected_mu: float) -> dict:
    labels = physics["newton_runtime_shape_labels"]
    materials = physics["newton_runtime_shape_material_arrays_in_label_order"]
    mus = materials.get("shape_material_mu")
    if not isinstance(labels, list) or not isinstance(mus, list) or len(labels) != len(mus):
        return {"status": "unavailable", "expected_mu": expected_mu,
                "shape_labels": labels, "shape_material_mu": mus}
    ground = [i for i, label in enumerate(labels) if str(label).endswith("/Field/Ground")]
    wheels = [i for i, label in enumerate(labels)
              if "_wheel_link/" in str(label) and str(label).endswith("/wheel_collision_visual")]
    if len(ground) != 1 or len(wheels) != 14:
        return {"status": "fail", "expected_mu": expected_mu,
                "ground_shape_count": len(ground), "wheel_shape_count": len(wheels)}
    selected = ground + wheels
    try:
        observed = [float(mus[i]) for i in selected]
    except (TypeError, ValueError):
        return {"status": "unavailable", "expected_mu": expected_mu,
                "observed_ground_and_14_wheel_mu": [mus[i] for i in selected]}
    matches = all(math.isfinite(value) and math.isclose(value, expected_mu, rel_tol=0.0, abs_tol=1e-6)
                  for value in observed)
    return {
        "status": "pass" if matches else "fail",
        "expected_mu": expected_mu,
        "ground_shape_label": labels[ground[0]],
        "wheel_shape_labels": [labels[i] for i in wheels],
        "observed_ground_and_14_wheel_mu": observed,
    }


def counter_value(counter, wp) -> int:
    return int(np.asarray(counter.numpy()).reshape(-1)[0])


def state_row(env, case_name: str, phase: str, phase_step: int, action: list[float],
              sim_time_s: float, effort_slot_start: int, effort_slot_end: int) -> dict:
    task = env.tasks[0]
    robot = np.asarray(task.robot, dtype=np.float64)
    root_pose = np.asarray(REUSE.as_numpy(env.rover.data.root_link_pose_w)[0], dtype=np.float64)
    root_velocity = np.asarray(REUSE.as_numpy(env.rover.data.root_link_vel_w)[0], dtype=np.float64)
    wheel_ids = env._wheel_ids.detach().cpu().numpy().astype(np.int64)
    wheel_qd = np.asarray(REUSE.as_numpy(env.rover.data.joint_vel)[0, wheel_ids], dtype=np.float64)
    wheel_target = np.asarray(REUSE.as_numpy(env.wheel_target_velocity)[0], dtype=np.float64)
    track_command = np.asarray(REUSE.as_numpy(env.track_command)[0], dtype=np.float64)
    pi_integral = np.asarray(REUSE.as_numpy(env._drive_pi.integral)[0], dtype=np.float64)
    pi_feedforward = np.asarray(REUSE.as_numpy(env._drive_pi.feedforward)[0], dtype=np.float64)
    body_pose = np.asarray(task.body_pose, dtype=np.float64)  # z, pitch, roll
    arrays = (robot, root_pose, root_velocity, wheel_qd, wheel_target, track_command, pi_integral,
              pi_feedforward, body_pose)
    info = task._info()
    return {
        "case": case_name,
        "phase": phase,
        "phase_step": phase_step,
        "simulation_time_s_from_case_start": sim_time_s,
        "normalized_action": action,
        "task_robot_xy_yaw_v_w": robot.tolist(),
        "root_pose_world_xyzw": root_pose.tolist(),
        "root_link_velocity_world_linear_angular": root_velocity.tolist(),
        "body_pose_z_pitch_roll": body_pose.tolist(),
        "wheel_joint_names": [env.rover.joint_names[int(i)] for i in wheel_ids],
        "wheel_joint_velocity_rad_s_policy_sample": wheel_qd.tolist(),
        "wheel_target_velocity_rad_s": wheel_target.tolist(),
        "track_command_mps": track_command.tolist(),
        "drive_pi_integral_nm_per_side": pi_integral.tolist(),
        "drive_pi_feedforward_nm_by_wheel": pi_feedforward.tolist(),
        "damage_count": int(info.get("damage", -1)),
        "collision_count": int(info.get("collisions", -1)),
        "done": bool(task.done),
        "reason": str(task.reason),
        "native_effort_slot_range_zero_based": [effort_slot_start, effort_slot_end],
        "native_effort_substeps_expected": effort_slot_end - effort_slot_start,
        "finite": bool(all(np.isfinite(value).all() for value in arrays)),
    }


def case_metrics(case: dict, rows: list[dict], initial_state: dict, initial_hash: str,
                 native_efforts: np.ndarray, effort_slots: tuple[int, int]) -> dict:
    drive = [row for row in rows if row["phase"] == "drive"]
    brake = [row for row in rows if row["phase"] == "brake"]
    settle = [row for row in rows if row["phase"] == "settle"]
    before_drive = settle[-1]["task_robot_xy_yaw_v_w"]
    drive_robot = np.asarray([before_drive, *[row["task_robot_xy_yaw_v_w"] for row in drive]], dtype=np.float64)
    xy = drive_robot[:, :2]
    yaw = np.unwrap(drive_robot[:, 2])
    start_xy = xy[0]
    drift = np.linalg.norm(xy - start_xy[None, :], axis=1)
    heading = float(drive_robot[0, 2])
    displacement = xy[-1] - start_xy
    projected = float(displacement @ np.asarray([math.cos(heading), math.sin(heading)]))
    drive_v = np.asarray([row["task_robot_xy_yaw_v_w"][3] for row in drive], dtype=np.float64)
    drive_w = np.asarray([row["task_robot_xy_yaw_v_w"][4] for row in drive], dtype=np.float64)
    tail = brake[-BRAKE_TAIL_STEPS:]
    tail_v = np.asarray([row["task_robot_xy_yaw_v_w"][3] for row in tail], dtype=np.float64)
    tail_w = np.asarray([row["task_robot_xy_yaw_v_w"][4] for row in tail], dtype=np.float64)
    brake_v = np.asarray([row["task_robot_xy_yaw_v_w"][3] for row in brake], dtype=np.float64)
    brake_w = np.asarray([row["task_robot_xy_yaw_v_w"][4] for row in brake], dtype=np.float64)
    if case["velocity_sign"]:
        rebound = float(np.maximum(0.0, -case["velocity_sign"] * brake_v).sum() * POLICY_DT)
        rebound_name = "opposite_translation_speed_integral_m"
    else:
        rebound = float(np.maximum(0.0, -case["yaw_sign"] * brake_w).sum() * POLICY_DT)
        rebound_name = "opposite_yaw_rate_integral_rad"
    effort = native_efforts[effort_slots[0]:effort_slots[1]]
    max_effort = float(np.max(np.abs(effort))) if effort.size else None
    damage_max = max((row["damage_count"] for row in rows), default=-1)
    collision_max = max((row["collision_count"] for row in rows), default=-1)
    max_abs_roll = max(abs(row["body_pose_z_pitch_roll"][2]) for row in rows)
    max_abs_pitch = max(abs(row["body_pose_z_pitch_roll"][1]) for row in rows)
    z_values = [row["body_pose_z_pitch_roll"][0] for row in rows]
    all_finite = all(row["finite"] for row in rows) and bool(effort.size and np.isfinite(effort).all())
    last_drive_v = float(drive_v[-20:].mean())
    signed_yaw_delta = float(case["yaw_sign"] * (yaw[-1] - yaw[0]))
    signed_drive_v = float(case["velocity_sign"] * last_drive_v) if case["velocity_sign"] else None
    reference_checks = {
        "finite_policy_and_effort_samples": all_finite,
        "all_physics_step_efforts_at_or_below_2Nm": bool(max_effort is not None and max_effort <= EFFORT_CAP_NM + 1e-5),
        "no_task_damage_or_collision_counters": damage_max == 0 and collision_max == 0,
        "no_episode_done_or_reason": not any(row["done"] for row in rows),
        "native_overturn_limit_abs_roll_pitch_le_1_2_rad": max_abs_roll <= 1.2 and max_abs_pitch <= 1.2,
        "brake_final_abs_forward_speed_lt_0_02m_s": bool(tail_v.size and abs(float(tail_v[-1])) < 0.02),
        "brake_final_abs_yaw_rate_lt_0_3rad_s": bool(tail_w.size and abs(float(tail_w[-1])) < 0.3),
        "brake_last_0_3s_max_abs_forward_speed_lt_0_02m_s": bool(tail_v.size and np.max(np.abs(tail_v)) < 0.02),
        "brake_last_0_3s_max_abs_yaw_rate_lt_0_3rad_s": bool(tail_w.size and np.max(np.abs(tail_w)) < 0.3),
    }
    informational_checks = {}
    if case["name"].startswith("pivot"):
        if case["name"].endswith("_low"):
            informational_checks["signed_pivot_yaw_delta_ge_0_1rad"] = signed_yaw_delta >= 0.1
        else:
            reference_checks["signed_pivot_yaw_delta_ge_0_1rad"] = signed_yaw_delta >= 0.1
        reference_checks["pivot_max_center_drift_lt_0_75m"] = float(drift.max()) < 0.75
    elif case["name"] in ("forward", "reverse"):
        reference_checks["commanded_translation_direction_last_1s"] = signed_drive_v is not None and signed_drive_v > 0.05
        signed_displacement = case["velocity_sign"] * projected
        reference_checks["commanded_translation_displacement_0_7_to_1_3m"] = 0.7 <= signed_displacement <= 1.3
    else:
        reference_checks["arc_translation_direction_last_1s"] = signed_drive_v is not None and signed_drive_v > 0.05
        reference_checks["arc_yaw_direction"] = signed_yaw_delta > 0.0
    return {
        "case": case["name"],
        "initial_state": initial_state,
        "initial_state_canonical_sha256": initial_hash,
        "same_initial_state_hash_as_first_case": None,
        "action": case["action"],
        "settle_s_drive_s_brake_s": [1.0, 2.0, 5.0],
        "brake_last_0_3s_action_all_zero": all(row["normalized_action"] == [0.0, 0.0, 0.0, 0.0] for row in tail),
        "drive_distance_projected_on_start_heading_m": projected,
        "drive_yaw_delta_unwrapped_rad": float(yaw[-1] - yaw[0]),
        "drive_signed_yaw_delta_rad": signed_yaw_delta,
        "drive_mean_v_last_1s_m_s": last_drive_v,
        "drive_mean_w_last_1s_rad_s": float(drive_w[-20:].mean()),
        "drive_max_center_drift_m": float(drift.max()),
        "drive_last_1s_signed_v": signed_drive_v,
        "brake_last_0_3s_mean_v_m_s": float(tail_v.mean()),
        "brake_last_0_3s_mean_w_rad_s": float(tail_w.mean()),
        "brake_last_0_3s_max_abs_v_m_s": float(np.max(np.abs(tail_v))),
        "brake_last_0_3s_max_abs_w_rad_s": float(np.max(np.abs(tail_w))),
        "brake_final_v_m_s": float(brake_v[-1]),
        "brake_final_w_rad_s": float(brake_w[-1]),
        rebound_name: rebound,
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
        "body_z_range_m": [float(min(z_values)), float(max(z_values))],
        "max_damage_count": damage_max,
        "max_collision_count": collision_max,
        "episode_done_reasons": sorted({row["reason"] for row in rows if row["done"]}),
        "native_effort_samples": int(len(effort)),
        "native_effort_max_abs_nm": max_effort,
        "reference_checks": reference_checks,
        "informational_checks": informational_checks,
        "interpretation": "response evidence only; these existing motion/safety checks do not establish a physics repair; require same-reset paired old/new evaluation",
    }


def positive_finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def main(argv=None) -> int:
    if os.environ.get("RUN_STAGED_RESPONSE_PROBE") != "1":
        raise SystemExit("Draft is non-running by default; set RUN_STAGED_RESPONSE_PROBE=1 only for the authorized Isaac run")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-dt", type=float, choices=tuple(DT_DECIMATION), required=True)
    parser.add_argument("--cone", choices=("pyramidal", "elliptic"), required=True)
    parser.add_argument("--impratio", type=positive_finite_float, required=True)
    parser.add_argument("--shape-mu", type=positive_finite_float, default=None,
                        help="optional Newton default shape friction; omitted means the config default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    decimation = DT_DECIMATION[args.physics_dt]
    out_dir = args.output_dir.expanduser().resolve()
    try:
        out_dir.relative_to(OUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"output must be under {OUT_ROOT}") from exc
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "staged_motion_response_summary.json"
    trajectories_path = out_dir / "policy_trajectories.jsonl"
    efforts_path = out_dir / "native_physics_efforts.jsonl"
    summary = {
        "schema": "staged-native-motion-response-v1",
        "status": "running",
        "purpose": "Eight-command native motion response and actual per-physics-step wheel effort; no training, no PI overlay, no repair verdict.",
        "seed": SEED,
        "start_xy_yaw": START,
        "goal_xy": GOAL,
        "case_count": len(CASES),
        "cases": [case["name"] for case in CASES],
        "settle_s_drive_s_brake_s": [1.0, 2.0, 5.0],
        "policy_dt_s": POLICY_DT,
        "physics_dt_s": args.physics_dt,
        "decimation": decimation,
        "solver_treatment": {"cone": args.cone, "impratio": args.impratio},
        "shape_mu_treatment": {"mode": "explicit" if args.shape_mu is not None else "cfg_default",
                               "requested_mu": args.shape_mu},
        "output_paths": {"policy_trajectories": str(trajectories_path), "native_physics_efforts": str(efforts_path)},
        "paired_run_rule": "Every case records reset state and hash before settling; candidate repair claims require matched old/new hashes and paired response comparison.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = None
    try:
        training_hashes, diagnostic_hashes = REUSE.BASE.source_hashes()
        summary["training_source_sha256"] = training_hashes
        summary["diagnostic_source_sha256"] = diagnostic_hashes
        summary["staged_probe_sha256"] = sha256_file(Path(__file__).resolve())
        summary["reuse_helper_sha256"] = sha256_file(REUSE_PROBE)
        summary["integration_helper_sha256"] = sha256_file(TEST_STEP_HELPER)
        summary["effort_kernel_sha256"] = sha256_file(EFFORT_KERNEL)

        from isaaclab_newton.physics import NewtonManager
        import torch
        import warp as wp
        from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
        from isaac_field.task import Config

        step_helper = runpy.run_path(str(TEST_STEP_HELPER))["_step"]
        effort_kernel = runpy.run_path(str(EFFORT_KERNEL))["record_wheel_applied_effort"]
        cfg_task = Config(size=100.0, plants=0, layout="rows", terrain_enabled=False,
                          validate_scene=False, max_time=15.0)
        cfg_task.dt = POLICY_DT
        cfg = FieldEnvCfg(task=cfg_task, mode="search", auto_reset=False, seed=SEED)
        cfg.scene.num_envs = 1
        cfg.sim.dt = args.physics_dt
        cfg.decimation = decimation
        if args.shape_mu is not None:
            cfg.sim.physics.default_shape_cfg.mu = args.shape_mu
        cfg.sim.physics.solver_cfg.cone = args.cone
        cfg.sim.physics.solver_cfg.impratio = args.impratio
        cfg.sim.physics.solver_cfg.use_mujoco_contacts = False
        env = IsaacFieldEnv(cfg)
        physics = runtime_snapshot(cfg, env)
        if not math.isclose(physics["policy_dt_s"], POLICY_DT, abs_tol=1e-12):
            raise RuntimeError(f"policy timestep changed: {physics}")
        expected_mu = args.shape_mu if args.shape_mu is not None else float(physics["shape_material_config"]["mu"])
        physics["shape_mu_runtime_validation"] = validate_ground_wheel_mu(physics, expected_mu)
        summary["runtime_physics"] = physics
        if physics["shape_mu_runtime_validation"]["status"] != "pass":
            raise RuntimeError(f"runtime ground+14 wheel mu validation failed: {physics['shape_mu_runtime_validation']}")
        summary["drive_config"] = env.drive_config

        first_options = {"mode": "search", "start": START, "goal": GOAL}
        env.reset(seed=SEED, options=first_options)
        wheel_ids_torch = env._wheel_ids.to(dtype=torch.int32).contiguous()
        wheel_ids_warp = wp.from_torch(wheel_ids_torch)
        applied_effort_warp = env.rover.actuators.applied_effort.warp
        total_policy_steps = len(CASES) * (SETTLE_STEPS + DRIVE_STEPS + BRAKE_STEPS)
        capacity = total_policy_steps * decimation
        effort_tape = wp.empty((capacity, 14), dtype=wp.float32, device=env.device)
        counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_effort():
            wp.launch(effort_kernel, dim=1, inputs=[applied_effort_warp, wheel_ids_warp,
                                                    effort_tape, counter, capacity], device=env.device)

        NewtonManager.register_post_actuator_callback(capture_effort)
        case_summaries = []
        pending_cases = []
        reset_hashes = []
        expected_slot = 0
        native_effort_host = np.empty((capacity, 14), dtype=np.float64)
        effort_tape_torch = wp.to_torch(effort_tape)
        with trajectories_path.open("x", encoding="utf-8") as trajectory_stream, \
                efforts_path.open("x", encoding="utf-8") as effort_stream:
            for case_index, case in enumerate(CASES):
                if case_index:
                    before_reset = counter_value(counter, wp)
                    env.reset(seed=SEED, options=first_options)
                    after_reset = counter_value(counter, wp)
                    if after_reset != before_reset:
                        raise RuntimeError(f"physics callback ran during reset: {before_reset}->{after_reset}")
                initial_state = REUSE.initial_state_snapshot(env)
                initial_hash = REUSE.json_sha256(initial_state)
                reset_hashes.append(initial_hash)
                start_slot = expected_slot
                case_rows = []
                local_elapsed = 0.0
                schedule = (("settle", [0.0, 0.0, 0.0, 0.0], SETTLE_STEPS),
                            ("drive", case["action"], DRIVE_STEPS),
                            ("brake", [0.0, 0.0, 0.0, 0.0], BRAKE_STEPS))
                for phase, action, phase_count in schedule:
                    for phase_step in range(1, phase_count + 1):
                        slot_before = counter_value(counter, wp)
                        step_helper(env, np.asarray(action, dtype=np.float32))
                        slot_after = counter_value(counter, wp)
                        if slot_after - slot_before != decimation:
                            raise RuntimeError(f"{case['name']} {phase} step {phase_step}: effort capture {slot_before}->{slot_after}, expected {decimation}")
                        sampled_effort = np.asarray(
                            effort_tape_torch[slot_before:slot_after].detach().cpu().numpy(), dtype=np.float64
                        )
                        if sampled_effort.shape != (decimation, 14) or not np.isfinite(sampled_effort).all():
                            raise RuntimeError(f"{case['name']} {phase} step {phase_step}: invalid native effort slice {sampled_effort.shape}")
                        native_effort_host[slot_before:slot_after] = sampled_effort
                        for substep, values in enumerate(sampled_effort, start=1):
                            slot = slot_before + substep - 1
                            effort_stream.write(json.dumps({
                                "case": case["name"],
                                "phase": phase,
                                "policy_step_in_phase": phase_step,
                                "physics_substep_in_policy": substep,
                                "physics_iteration_global": slot + 1,
                                "simulation_time_s_global": (slot + 1) * args.physics_dt,
                                "native_applied_effort_nm_by_wheel_order": values.tolist(),
                                "max_abs_effort_nm": float(np.max(np.abs(values))),
                                "finite": bool(np.isfinite(values).all()),
                            }, ensure_ascii=False, allow_nan=False) + "\n")
                        effort_stream.flush()
                        expected_slot += decimation
                        local_elapsed += POLICY_DT
                        row = state_row(env, case["name"], phase, phase_step, action,
                                        local_elapsed, slot_before, slot_after)
                        case_rows.append(row)
                        trajectory_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        trajectory_stream.flush()
                end_slot = expected_slot
                pending_cases.append((case, case_rows, initial_state, initial_hash, (start_slot, end_slot)))

        captured = counter_value(counter, wp)
        if captured != capacity or expected_slot != capacity:
            raise RuntimeError(f"captured {captured} substeps; expected {capacity}")
        native_efforts = native_effort_host[:captured]
        if not np.isfinite(native_efforts).all():
            raise RuntimeError("native effort tape has non-finite values")
        max_effort = float(np.max(np.abs(native_efforts)))
        for case, case_rows, initial_state, initial_hash, slots in pending_cases:
            row = case_metrics(case, case_rows, initial_state, initial_hash, native_efforts, slots)
            row["native_effort_slot_range_zero_based"] = list(slots)
            case_summaries.append(row)
        for row in case_summaries:
            row["native_effort_cap_pass"] = row["native_effort_max_abs_nm"] <= EFFORT_CAP_NM + 1e-5
            row["reference_checks"]["all_physics_step_efforts_at_or_below_2Nm"] = row["native_effort_cap_pass"]
        first_hash = reset_hashes[0] if reset_hashes else None
        for row in case_summaries:
            row["same_initial_state_hash_as_first_case"] = row["initial_state_canonical_sha256"] == first_hash
        final_training_hashes, final_diagnostic_hashes = REUSE.BASE.source_hashes()
        if final_training_hashes != training_hashes or final_diagnostic_hashes != diagnostic_hashes:
            raise RuntimeError("training or diagnostic source hashes changed during probe")
        summary.update({
            "status": "complete",
            "cases_summary": case_summaries,
            "initial_state_hashes_match_across_all_cases": bool(reset_hashes and len(set(reset_hashes)) == 1),
            "initial_state_hashes_by_case": dict(zip((case["name"] for case in CASES), reset_hashes)),
            "native_physics_effort_substeps_captured": captured,
            "native_physics_effort_substeps_expected": capacity,
            "native_effort_max_abs_nm_all_cases": max_effort,
            "native_effort_cap_nm": EFFORT_CAP_NM,
            "native_effort_cap_pass": max_effort <= EFFORT_CAP_NM + 1e-5,
            "training_and_diagnostic_hashes_unchanged_after_run": True,
            "overall_repair_verdict": "not_evaluated; response matrix only; pair each future candidate with same-reset old-physics run",
        })
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "cases": len(case_summaries), "effort_substeps": captured,
                          "max_effort_nm": max_effort,
                          "initial_hashes_match": summary["initial_state_hashes_match_across_all_cases"]},
                         ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
