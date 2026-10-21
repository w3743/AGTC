#!/usr/bin/env python3
"""Diagnostic-only runtime PI counterfactual; does not alter project sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import types

import numpy as np
import torch
import warp as wp

from runtime_pi import RuntimePIController, broadcast_side_values, side_mean


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent
P_SUMMARY = ROOT / "results/isaac_field_diagnosis/minimal_inputs_20261020/turn_response/turn_response_summary.json"
SUBSTEP_CAPTURE = ROOT / "tests/newton_effort_capture.py"
EFFORT_LIMIT_SMOKE = ROOT / "results/isaac_field_diagnosis/minimal_inputs_20261020/effort_limit_smoke/effort_limit_smoke.py"

KI_NM_PER_RAD_S_S = 0.1
INTEGRAL_LIMIT_NM = 2.0
EXPECTED_KP_NM_PER_RAD_S = 0.04
EXPECTED_WHEEL_LIMIT_NM = 2.0
SETTLE_S = 1.0
DRIVE_S = 8.0
BRAKE_S = 2.0
SEED = 90231
START = [50.0, 50.0, 0.0]
GOAL = [90.0, 50.0]


def finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value in PI counterfactual: {result!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    if os.environ.get("RUN_ISAAC_PHYSICS") != "1":
        raise SystemExit("Refusing native PI probe without RUN_ISAAC_PHYSICS=1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ki", type=float, default=KI_NM_PER_RAD_S_S)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    selected_ki = finite(args.ki)
    if selected_ki < 0.0:
        raise ValueError(f"--ki must be non-negative, got {selected_ki}")
    is_capped_p_only = selected_ki == 0.0
    controller_name = "capped_p_only_reference" if is_capped_p_only else "runtime_pi_counterfactual"
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(OUT_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"output directory must be inside {OUT_DIR}: {output_dir}") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "policy_step_samples.jsonl"
    substeps_path = output_dir / "physics_substep_effort.jsonl"
    summary_path = output_dir / "counterfactual_summary.json"

    p_summary = json.loads(P_SUMMARY.read_text(encoding="utf-8"))
    p_cases = p_summary.get("cases", [])
    if p_summary.get("case_count") != 8 or len(p_cases) != 8:
        raise ValueError(f"expected the original 8-case P summary, got {len(p_cases)}")
    same_action_cases = []
    for row in p_cases:
        action = [finite(v) for v in row["policy_normalized_action"]]
        if len(action) != 4:
            raise ValueError(f"invalid original action for {row['case']}: {action}")
        same_action_cases.append({
            "name": row["case"],
            "action": action,
            "historical_action_source": {
                "summary_case": row["case"],
                "policy_normalized_action": action,
            },
        })
    brake_cases = [
        {"name": "brake_straight_0.20_to_zero", "action": [0.2, 0.2, 0.0, 0.0]},
        {"name": "brake_diff_left_plus_0.80_to_zero", "action": [0.8, -0.8, 0.0, 0.0]},
    ]

    helpers = runpy.run_path("tests/test_env_integration.py")
    make_env, step, to_numpy = helpers["_make_env"], helpers["_step"], helpers["_to_numpy"]
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import euler_xyz_from_quat

    source_paths = (
        "isaac_field/env.py",
        "isaac_field/task.py",
        "isaac_field/assets.py",
        "isaac_field/action_contract.py",
        "scripts/train_isaac_foundation_fresh.py",
        "tests/test_env_integration.py",
        "tests/newton_effort_capture.py",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/turn_response/turn_response_probe.py",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/turn_response/turn_response_summary.json",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/effort_limit_smoke/effort_limit_smoke.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "external/IsaacLab/source/isaaclab/isaaclab/actuators/newton/kernels.py",
        "external/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py",
        "external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/runtime_pi.py",
        "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual/pi_counterfactual_probe.py",
    )
    source_sha256 = {path: sha256(ROOT / path) for path in source_paths}

    env = make_env(plants=0, max_time=20.0, terrain_enabled=False, seed=SEED)
    controller = None
    summaries = []
    sample_count = 0
    base_drive_config = None
    substep_monitor = {
        "checked_physics_substeps": 0,
        "max_abs_applied_effort_nm": 0.0,
        "by_case_phase": {},
    }
    try:
        policy_dt = finite(env.step_dt)
        physics_dt = finite(env.cfg.sim.dt)
        decimation = int(env.cfg.decimation)
        kp = finite(env.cfg.drive_velocity_gain)
        wheel_limit = finite(env.cfg.drive_effort_limit)
        actuator_limit = finite(env.cfg.scene.robot.actuators["drive"].actuator_effort_limit)
        if not math.isclose(kp, EXPECTED_KP_NM_PER_RAD_S, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"native P gain changed: expected {EXPECTED_KP_NM_PER_RAD_S}, got {kp}")
        if not math.isclose(wheel_limit, EXPECTED_WHEEL_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"native wheel effort limit changed: expected {EXPECTED_WHEEL_LIMIT_NM}, got {wheel_limit}")
        if not math.isclose(actuator_limit, EXPECTED_WHEEL_LIMIT_NM, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"native actuator effort clamp changed: expected {EXPECTED_WHEEL_LIMIT_NM}, got {actuator_limit}")
        if not math.isclose(physics_dt, 0.005, rel_tol=0.0, abs_tol=1e-12) or decimation != 10:
            raise RuntimeError(f"unexpected native timing: physics_dt={physics_dt}, decimation={decimation}")
        if not math.isclose(policy_dt, physics_dt * decimation, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"policy dt {policy_dt} does not match physics_dt * decimation")

        wheel_ids = env._wheel_ids
        wheel_count = int(wheel_ids.numel())
        if int(env.num_envs) != 1:
            raise RuntimeError(f"the isolated capture kernel is configured for one env, got {env.num_envs}")
        if wheel_count != 14:
            raise RuntimeError(f"expected 14 proxy wheel joints, got {wheel_count}")
        sides = np.asarray(to_numpy(env._wheel_track_ids), dtype=np.int64).reshape(-1)
        if sides.size != wheel_count or int((sides == 0).sum()) != 7 or int((sides == 1).sum()) != 7:
            raise RuntimeError("wheel side map must contain seven left and seven right joints")
        left, right = sides == 0, sides == 1
        base_drive_config = env.drive_config
        if base_drive_config.get("interface") != "independent_left_right_track_speed_explicit_v5":
            raise RuntimeError(f"PI counterfactual requires the capped v5 drive, got {base_drive_config!r}")
        if not math.isclose(
            finite(base_drive_config.get("actuator_effort_limit_nm")),
            EXPECTED_WHEEL_LIMIT_NM,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("live v5 drive metadata does not record the 2 Nm actuator clamp")

        settle_count = round(SETTLE_S / policy_dt)
        drive_count = round(DRIVE_S / policy_dt)
        brake_count = round(BRAKE_S / policy_dt)
        total_policy_steps = (
            len(same_action_cases) * (settle_count + drive_count)
            + len(brake_cases) * (settle_count + drive_count + brake_count)
        )
        capture_capacity = total_policy_steps * decimation
        from isaaclab_newton.physics import NewtonManager

        capture_kernel = runpy.run_path(str(SUBSTEP_CAPTURE))["record_wheel_applied_effort"]
        wheel_ids_warp = wp.from_torch(wheel_ids.to(dtype=torch.int32).contiguous())
        applied_effort_warp = env.rover.actuators.applied_effort.warp
        substep_tape = wp.empty((capture_capacity, wheel_count), dtype=wp.float32, device=env.device)
        substep_counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_applied_effort_after_each_actuator_step():
            wp.launch(
                capture_kernel,
                dim=1,
                inputs=[applied_effort_warp, wheel_ids_warp, substep_tape, substep_counter, capture_capacity],
                device=env.device,
            )

        env.reset(seed=SEED, options={"mode": "search", "start": START, "goal": GOAL})
        NewtonManager.register_post_actuator_callback(capture_applied_effort_after_each_actuator_step)
        controller = RuntimePIController(
            env.num_envs,
            2,
            device=env.device,
            kp=kp,
            ki=selected_ki,
            dt=policy_dt,
            limit=INTEGRAL_LIMIT_NM,
        )
        integral_enabled = {"value": False}
        original_pre = env._pre_physics_step
        original_apply = env._apply_action
        original_reset = env._reset_idx

        def wrapped_pre(self, actions):
            original_pre(actions)
            target = self.wheel_target_velocity
            actual = self.rover.data.joint_vel.torch[:, self._wheel_ids]
            side_ids = torch.as_tensor(sides, device=target.device, dtype=torch.long)
            target_side = side_mean(target, side_ids)
            actual_side = side_mean(actual, side_ids)
            if integral_enabled["value"] and selected_ki > 0.0:
                controller.update(target_side, actual_side)
            else:
                # Both runs use the identical native-P-only settling phase.
                controller.last_error = target_side - actual_side
                controller.last_p_estimate = controller.kp * controller.last_error
                controller.last_blocked = torch.zeros_like(controller.last_error, dtype=torch.bool)

        def wrapped_apply(self):
            original_apply()
            self.rover.actuators.target_command.set_effort_index(
                value=broadcast_side_values(
                    controller.feedforward,
                    torch.as_tensor(sides, device=controller.feedforward.device, dtype=torch.long),
                ),
                joint_ids=self._wheel_ids,
            )

        def wrapped_reset(self, env_ids):
            original_reset(env_ids)
            controller.reset(env_ids)
            self.rover.actuators.target_command.set_effort_index(
                value=torch.zeros(
                    (len(env_ids), wheel_count),
                    device=controller.feedforward.device,
                    dtype=controller.feedforward.dtype,
                ),
                joint_ids=self._wheel_ids,
                env_ids=env_ids,
            )

        env._pre_physics_step = types.MethodType(wrapped_pre, env)
        env._apply_action = types.MethodType(wrapped_apply, env)
        env._reset_idx = types.MethodType(wrapped_reset, env)

        def read_state(previous_yaw, yaw_unwrapped):
            pose_t = env.rover.data.root_link_pose_w.torch
            pose = np.asarray(to_numpy(pose_t)[0], dtype=np.float64)
            angles_t = torch.stack(euler_xyz_from_quat(pose_t[:, 3:]), dim=-1)
            roll, pitch, yaw = np.asarray(to_numpy(angles_t)[0], dtype=np.float64)
            if previous_yaw is None:
                yaw_unwrapped = float(yaw)
            else:
                yaw_unwrapped += (float(yaw) - previous_yaw + math.pi) % (2.0 * math.pi) - math.pi
            origin = np.asarray(to_numpy(env.scene.env_origins)[0], dtype=np.float64)
            root_v = np.asarray(to_numpy(env.rover.data.root_com_vel_w.torch)[0], dtype=np.float64)
            robot = np.asarray(env.tasks[0].robot, dtype=np.float64)
            target = np.asarray(to_numpy(env.wheel_target_velocity)[0], dtype=np.float64)
            actual = np.asarray(to_numpy(env.rover.data.joint_vel.torch[:, wheel_ids])[0], dtype=np.float64)
            effort = np.asarray(to_numpy(env.rover.actuators.applied_effort.torch[:, wheel_ids])[0], dtype=np.float64)
            track_command = np.asarray(to_numpy(env.track_command)[0], dtype=np.float64)
            return {
                "task_time_s": finite(env.tasks[0].time),
                "position_xy_m": [finite(v) for v in robot[:2]],
                "root_z_m": finite(pose[2] - origin[2]),
                "roll_pitch_yaw_rad": [finite(roll), finite(pitch), finite(yaw)],
                "yaw_unwrapped_rad": finite(yaw_unwrapped),
                "world_linear_velocity_mps": [finite(v) for v in root_v[:3]],
                "world_angular_velocity_rad_s": [finite(v) for v in root_v[3:6]],
                "actual_v_w": [finite(robot[3]), finite(robot[4])],
                "track_command_mps": [finite(v) for v in track_command],
                "nominal_v_w": [
                    finite(float(track_command.mean())),
                    finite(float(
                        (track_command[1] - track_command[0])
                        / ROBOT_SPEC.track_separation_m
                    )),
                ],
                "wheel_target_rad_s": [finite(v) for v in target],
                "wheel_actual_qd_rad_s": [finite(v) for v in actual],
                "wheel_applied_effort_nm": [finite(v) for v in effort],
                "left_target_mean_rad_s": finite(np.mean(target[left])),
                "right_target_mean_rad_s": finite(np.mean(target[right])),
                "left_actual_qd_mean_rad_s": finite(np.mean(actual[left])),
                "right_actual_qd_mean_rad_s": finite(np.mean(actual[right])),
                "left_applied_effort_mean_nm": finite(np.mean(effort[left])),
                "right_applied_effort_mean_nm": finite(np.mean(effort[right])),
            }, float(yaw), yaw_unwrapped

        def begin_phase():
            return int(substep_counter.numpy()[0])

        def capture_phase(case_name, phase, counter_start, policy_steps):
            counter_end = int(substep_counter.numpy()[0])
            expected = int(policy_steps * decimation)
            captured_count = counter_end - counter_start
            if captured_count != expected:
                raise AssertionError(
                    f"{case_name}/{phase}: expected {expected} Newton actuator iterations, "
                    f"captured {captured_count}"
                )
            captured = np.asarray(
                substep_tape.numpy()[counter_start:counter_end], dtype=np.float64
            )
            if captured.shape != (expected, wheel_count) or not np.isfinite(captured).all():
                raise AssertionError(
                    f"{case_name}/{phase}: invalid substep effort capture shape/data {captured.shape}"
                )
            max_abs = float(np.max(np.abs(captured)))
            if max_abs > wheel_limit + 1e-4:
                raise AssertionError(
                    f"{case_name}/{phase}: Newton applied effort {max_abs} exceeds "
                    f"configured clamp {wheel_limit} Nm"
                )
            substep_monitor["checked_physics_substeps"] += captured_count
            substep_monitor["max_abs_applied_effort_nm"] = max(
                substep_monitor["max_abs_applied_effort_nm"], max_abs
            )
            substep_monitor["by_case_phase"][f"{case_name}/{phase}"] = {
                "checks": captured_count,
                "max_abs_applied_effort_nm": max_abs,
            }
            for local_index, effort in enumerate(captured):
                substep_stream.write(json.dumps({
                    "case": case_name,
                    "phase": phase,
                    "physics_iteration_global": counter_start + local_index,
                    "policy_step": local_index // decimation + 1,
                    "physics_substep": local_index % decimation + 1,
                    "applied_effort_nm": [finite(v) for v in effort],
                }, sort_keys=True, allow_nan=False) + "\n")
            substep_stream.flush()
            return {
                "physics_iteration_start_inclusive": counter_start,
                "physics_iteration_end_exclusive": counter_end,
                "captured_physics_iterations": captured_count,
                "max_abs_applied_effort_nm": max_abs,
            }

        def one_step(case_name, phase, index, action, previous_yaw, yaw_unwrapped, samples):
            nonlocal sample_count
            action_np = np.asarray(action, dtype=np.float32)
            step(env, action_np, 1)
            state, previous_yaw, yaw_unwrapped = read_state(previous_yaw, yaw_unwrapped)
            if len(state["wheel_applied_effort_nm"]) != wheel_count:
                raise AssertionError("applied-effort sample width mismatch")
            max_effort = max(abs(value) for value in state["wheel_applied_effort_nm"])
            if max_effort > wheel_limit + 1e-5:
                raise AssertionError(f"wheel applied effort {max_effort} exceeds {wheel_limit} Nm")
            error = np.asarray(to_numpy(controller.last_error)[0], dtype=np.float64)
            p_estimate = np.asarray(to_numpy(controller.last_p_estimate)[0], dtype=np.float64)
            feedforward = np.asarray(to_numpy(controller.feedforward)[0], dtype=np.float64)
            blocked = np.asarray(to_numpy(controller.last_blocked)[0], dtype=bool)
            if is_capped_p_only and not torch.equal(
                controller.integral, torch.zeros_like(controller.integral)
            ):
                raise AssertionError("Ki=0 capped-P reference accumulated a non-zero integral")
            row = {
                "case": case_name,
                "controller": controller_name,
                "phase": phase,
                "step": int(index),
                "policy_normalized_action": [finite(v) for v in action_np],
                "raw_action_track_speed_mps": [finite(v) for v in action_np[:2] * float(env.field_cfg.max_speed)],
                "pi_ki_nm_per_proxy_wheel_per_rad_s_side_mean_s": selected_ki,
                "pi_side_mean_pre_step_velocity_error_rad_s": [finite(v) for v in error],
                "pi_side_p_estimate_nm_per_proxy_wheel": [finite(v) for v in p_estimate],
                "pi_side_integral_feedforward_nm_per_proxy_wheel": [finite(v) for v in feedforward],
                "pi_side_estimated_p_plus_i_nm_per_proxy_wheel": [
                    finite(v) for v in p_estimate + feedforward
                ],
                "pi_integral_feedforward_broadcast_nm_per_wheel": [
                    finite(v) for v in to_numpy(broadcast_side_values(
                        controller.feedforward,
                        torch.as_tensor(sides, device=controller.feedforward.device, dtype=torch.long),
                    )[0])
                ],
                "pi_antiwindup_blocked_by_side": [bool(v) for v in blocked],
                "pi_antiwindup_blocked_side_count": int(blocked.sum()),
                "pi_integral_enabled": bool(integral_enabled["value"] and selected_ki > 0.0),
                **state,
            }
            samples.append(row)
            sample_count += 1
            return previous_yaw, yaw_unwrapped

        def run_case(case_name, action, *, brake_action=None, action_source=None):
            integral_enabled["value"] = False
            env.reset(
                seed=SEED,
                options={"mode": "search", "start": START, "goal": GOAL},
            )
            if not torch.equal(controller.integral, torch.zeros_like(controller.integral)):
                raise AssertionError("PI state was not cleared for reset episode")
            previous_yaw = None
            yaw_unwrapped = 0.0
            rows = []
            zero_action = np.zeros(4, dtype=np.float32)
            settle_capture_start = begin_phase()
            for index in range(1, settle_count + 1):
                previous_yaw, yaw_unwrapped = one_step(
                    case_name, "settle", index, zero_action, previous_yaw, yaw_unwrapped, rows
                )
            settle_effort = capture_phase(
                case_name, "settle", settle_capture_start, settle_count
            )
            settle_initial_state = {
                key: rows[-1][key]
                for key in (
                    "position_xy_m", "root_z_m", "roll_pitch_yaw_rad", "actual_v_w",
                    "track_command_mps", "wheel_target_rad_s", "wheel_actual_qd_rad_s",
                )
            }
            if not torch.equal(controller.integral, torch.zeros_like(controller.integral)):
                raise AssertionError(f"{case_name}: integral changed during P-only settling")

            integral_enabled["value"] = True
            drive_capture_start = begin_phase()
            drive_start = len(rows)
            for index in range(1, drive_count + 1):
                previous_yaw, yaw_unwrapped = one_step(
                    case_name, "drive", index, action, previous_yaw, yaw_unwrapped, rows
                )
            drive_effort = capture_phase(
                case_name, "drive", drive_capture_start, drive_count
            )
            drive_rows = rows[drive_start:]
            brake_rows = []
            brake_effort = None
            if brake_action is not None:
                brake_capture_start = begin_phase()
                brake_start = len(rows)
                for index in range(1, brake_count + 1):
                    previous_yaw, yaw_unwrapped = one_step(
                        case_name, "brake", index, brake_action,
                        previous_yaw, yaw_unwrapped, rows,
                    )
                brake_effort = capture_phase(
                    case_name, "brake", brake_capture_start, brake_count
                )
                brake_rows = rows[brake_start:]

            first, last = drive_rows[0], drive_rows[-1]
            first_second = drive_rows[:round(1.0 / policy_dt)]
            last_second = drive_rows[-round(1.0 / policy_dt):]
            raw_nominal_yaw = (
                (float(action[1]) - float(action[0])) * float(env.field_cfg.max_speed)
                / float(ROBOT_SPEC.track_separation_m)
            )
            measured_yaw_last_1s = finite(np.mean(
                [row["actual_v_w"][1] for row in last_second]
            ))
            yaw_steps = np.diff(np.asarray(
                [row["yaw_unwrapped_rad"] for row in drive_rows], dtype=np.float64
            ))
            start_yaw = float(first["yaw_unwrapped_rad"])
            path_length = sum(
                math.hypot(
                    row["position_xy_m"][0] - previous["position_xy_m"][0],
                    row["position_xy_m"][1] - previous["position_xy_m"][1],
                )
                for previous, row in zip(drive_rows, drive_rows[1:])
            )
            drive_summary = {
                "duration_s": finite(len(drive_rows) * policy_dt),
                "start_xy_m": first["position_xy_m"],
                "end_xy_m": last["position_xy_m"],
                "start_actual_v_w": first["actual_v_w"],
                "end_actual_v_w": last["actual_v_w"],
                "net_xy_displacement_m": finite(np.linalg.norm(
                    np.asarray(last["position_xy_m"]) - np.asarray(first["position_xy_m"])
                )),
                "path_length_m": finite(path_length),
                "start_yaw_unwrapped_rad": start_yaw,
                "end_yaw_unwrapped_rad": last["yaw_unwrapped_rad"],
                "net_yaw_delta_unwrapped_rad": finite(last["yaw_unwrapped_rad"] - start_yaw),
                "cumulative_abs_yaw_path_rad": finite(np.sum(np.abs(yaw_steps))),
                "mean_actual_v_w": [finite(np.mean([
                    row["actual_v_w"][0] for row in drive_rows
                ])),
                                     finite(np.mean([row["actual_v_w"][1] for row in drive_rows]))],
                "mean_signed_actual_yaw_rate_first_1s_rad_s": finite(np.mean([
                    row["actual_v_w"][1] for row in first_second
                ])),
                "mean_signed_actual_yaw_rate_last_1s_rad_s": finite(np.mean(
                    [row["actual_v_w"][1] for row in last_second]
                )),
                "yaw_sign_matches_raw_command_last_1s": bool(
                    raw_nominal_yaw == 0.0
                    or measured_yaw_last_1s * raw_nominal_yaw > 0.0
                ),
                "mean_abs_actual_yaw_rate_last_1s_rad_s": finite(np.mean(
                    [abs(row["actual_v_w"][1]) for row in last_second]
                )),
                "mean_abs_command_yaw_rate_last_1s_rad_s": finite(np.mean(
                    [abs(row["nominal_v_w"][1]) for row in last_second]
                )),
                "last_1s_track_side_means": {
                    key: finite(np.mean([row[key] for row in last_second]))
                    for key in (
                        "left_target_mean_rad_s", "right_target_mean_rad_s",
                        "left_actual_qd_mean_rad_s", "right_actual_qd_mean_rad_s",
                        "left_applied_effort_mean_nm", "right_applied_effort_mean_nm",
                    )
                },
                "last_1s_mean_abs_qd_tracking_error_rad_s": {
                    side: finite(np.mean([
                        abs(row[f"{side}_target_mean_rad_s"] - row[f"{side}_actual_qd_mean_rad_s"])
                        for row in last_second
                    ]))
                    for side in ("left", "right")
                },
                "max_abs_roll_pitch_rad_all_phases": {
                    "roll": max(abs(row["roll_pitch_yaw_rad"][0]) for row in rows),
                    "pitch": max(abs(row["roll_pitch_yaw_rad"][1]) for row in rows),
                },
                "max_abs_applied_effort_nm": drive_effort["max_abs_applied_effort_nm"],
                "max_abs_side_integral_feedforward_nm_per_proxy_wheel": max(
                    max(abs(value) for value in row["pi_side_integral_feedforward_nm_per_proxy_wheel"])
                    for row in drive_rows
                ),
                "antiwindup_blocked_side_samples": sum(
                    row["pi_antiwindup_blocked_side_count"] for row in drive_rows
                ),
            }
            summary = {
                "case": case_name,
                "policy_normalized_action": [finite(v) for v in action],
                "raw_track_command_mps": [finite(v * float(env.field_cfg.max_speed)) for v in action[:2]],
                "nominal_yaw_rate_raw_action_rad_s": finite(raw_nominal_yaw),
                "historical_action_source_only": action_source,
                "settle_control_strategy": "native capped per-wheel IdealPD; per-side integral disabled and feedforward exactly zero",
                "initial_state_after_settle": settle_initial_state,
                "physics_substep_applied_effort": {
                    "settle": settle_effort,
                    "drive": drive_effort,
                    "brake": brake_effort,
                },
                "drive": drive_summary,
                "samples_including_settle": len(rows),
                "all_recorded_policy_values_finite": True,
            }
            if brake_action is not None:
                brake_first, brake_last = brake_rows[0], brake_rows[-1]
                brake_tail = brake_rows[-max(1, round(0.3 / policy_dt)):]
                tail_v = [row["actual_v_w"][0] for row in brake_tail]
                tail_w = [row["actual_v_w"][1] for row in brake_tail]
                tail_i = np.asarray(
                    [row["pi_side_integral_feedforward_nm_per_proxy_wheel"] for row in brake_tail],
                    dtype=np.float64,
                )
                slow_v = all(abs(value) < 0.2 for value in tail_v)
                slow_w = all(abs(value) < 0.3 for value in tail_w)
                summary["brake"] = {
                    "duration_s": finite(len(brake_rows) * policy_dt),
                    "start_actual_v_w": brake_first["actual_v_w"],
                    "end_actual_v_w": brake_last["actual_v_w"],
                    "start_yaw_unwrapped_rad": brake_first["yaw_unwrapped_rad"],
                    "end_yaw_unwrapped_rad": brake_last["yaw_unwrapped_rad"],
                    "net_yaw_delta_rad": finite(brake_last["yaw_unwrapped_rad"] - brake_first["yaw_unwrapped_rad"]),
                    "last_0_3s_policy_steps": len(brake_tail),
                    "last_0_3s_mean_actual_v_w": [finite(np.mean(tail_v)), finite(np.mean(tail_w))],
                    "last_0_3s_max_abs_actual_v_mps": max(abs(value) for value in tail_v),
                    "last_0_3s_max_abs_actual_yaw_rate_rad_s": max(abs(value) for value in tail_w),
                    "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel": {
                        "mean_abs_per_side": finite(np.mean(np.abs(tail_i))),
                        "max_abs": finite(np.max(np.abs(tail_i))),
                        "end_per_side": brake_last[
                            "pi_side_integral_feedforward_nm_per_proxy_wheel"
                        ],
                        "end_broadcast_per_wheel": brake_last[
                            "pi_integral_feedforward_broadcast_nm_per_wheel"
                        ],
                    },
                    "last_0_3s_low_speed_threshold_mps": 0.2,
                    "last_0_3s_low_yaw_rate_threshold_rad_s": 0.3,
                    "last_0_3s_all_samples_meet_low_speed_threshold": bool(slow_v),
                    "last_0_3s_all_samples_meet_low_yaw_rate_threshold": bool(slow_w),
                    "last_0_3s_meets_both_low_speed_thresholds": bool(slow_v and slow_w),
                    "max_abs_applied_effort_nm": brake_effort["max_abs_applied_effort_nm"],
                    "end_track_command_mps": brake_last["track_command_mps"],
                    "end_wheel_target_rad_s": brake_last["wheel_target_rad_s"],
                    "end_wheel_actual_qd_rad_s": brake_last["wheel_actual_qd_rad_s"],
                }
            summaries.append(summary)
            for row in rows:
                sample_stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            sample_stream.flush()
            print("PI COUNTERFACTUAL CASE " + json.dumps({
                "case": case_name,
                "drive": drive_summary,
                "brake": summary.get("brake"),
            }, sort_keys=True), flush=True)

        with samples_path.open("x", encoding="utf-8", newline="\n") as sample_stream, \
                substeps_path.open("x", encoding="utf-8", newline="\n") as substep_stream:
            for case in same_action_cases:
                run_case(
                    case["name"], case["action"],
                    action_source=case["historical_action_source"],
                )
            for case in brake_cases:
                run_case(
                    case["name"], case["action"],
                    brake_action=[0.0, 0.0, 0.0, 0.0],
                )
        expected_substeps = sample_count * decimation
        if substep_monitor["checked_physics_substeps"] != expected_substeps:
            raise AssertionError(
                "physics-substep applied-effort monitor coverage mismatch: "
                f"checked={substep_monitor['checked_physics_substeps']}, "
                f"expected={expected_substeps}"
            )
    finally:
        env.close()

    summary_data = {
        "schema": "isaac-field-capped-drive-counterfactual-v1",
        "run_label": output_dir.name,
        "purpose": (
            "capped P-only reference (Ki=0, zero integral/feedforward)"
            if is_capped_p_only else
            "runtime PI counterfactual; not training, not a calibrated hardware controller, "
            "and not causal training evidence"
        ),
        "base_drive_config": base_drive_config,
        "counterfactual_controller": {
            "name": controller_name,
            "controller_architecture": "side_integral_plus_native_per_wheel_P",
            "implementation": (
                "native per-wheel IdealPD only; Ki=0 keeps the two per-side integrators and broadcast feedforward exactly zero"
                if is_capped_p_only else
                "left/right mean-speed integral feedforward added through public "
                "target_command.set_effort_index after the original _apply_action; each side value "
                "is broadcast to its seven proxy wheels while native per-wheel IdealPD remains active"
            ),
            "controller_state_count": 2,
            "controller_state_order": ["left_track_mean", "right_track_mean"],
            "integral_state_basis": "mean target-minus-actual angular velocity over seven proxy wheels per side",
            "integral_feedforward_broadcast": "one side value applied equally to each of that side's seven proxy wheels",
            "kp_nm_per_rad_s": EXPECTED_KP_NM_PER_RAD_S,
            "ki_nm_per_proxy_wheel_per_rad_s_side_mean_s": selected_ki,
            "ki_units_note": "Nm per proxy wheel / (rad/s side-mean proxy wheel speed * s)",
            "ki_status": "fixed zero for capped-P reference" if is_capped_p_only else "uncalibrated assumption",
            "integration_timebase": "policy step at _pre_physics_step",
            "settle_integral_enabled": False,
            "drive_and_brake_integrator_active": selected_ki > 0.0,
            "integral_clip_nm": [-INTEGRAL_LIMIT_NM, INTEGRAL_LIMIT_NM],
            "conditional_antiwindup": "block integration only when estimated P+I is outside +/-2 Nm and error pushes farther into saturation",
            "configured_actuator_effort_limit_nm_per_wheel": actuator_limit,
            "configured_joint_effort_limit_nm_per_wheel": wheel_limit,
            "observed_applied_effort_assert_nm": wheel_limit + 1e-4,
            "actual_applied_effort_capture": "NewtonManager.register_post_actuator_callback + Warp tape after actuator step on each internal Newton physics iteration",
            "state_and_feedforward_reset": "clear selected environment rows in _reset_idx wrapper",
        },
        "environment": {
            "plants": 0,
            "terrain_enabled": False,
            "seed": SEED,
            "start_pose_xy_yaw": START,
            "goal_xy": GOAL,
            "max_time_s": 20.0,
            "settle_s": SETTLE_S,
            "drive_s": DRIVE_S,
            "brake_observation_s": BRAKE_S,
            "physics_dt_s": physics_dt,
            "policy_dt_s": policy_dt,
            "decimation": decimation,
            "max_track_speed_mps": finite(env.field_cfg.max_speed),
            "track_acceleration_mps2": finite(env.field_cfg.acceleration),
            "track_separation_m": finite(ROBOT_SPEC.track_separation_m),
            "wheel_radius_m": finite(ROBOT_SPEC.wheel_radius_m),
            "wheel_count": wheel_count,
        },
        "historical_action_source": {
            "summary_path": str(P_SUMMARY),
            "summary_sha256": sha256(P_SUMMARY),
            "matched_case_count": len(same_action_cases),
            "case_actions_read_from_existing_v4_p_summary": True,
            "historical_policy_outcomes_used_as_v5_baseline": False,
        },
        "source_sha256": source_sha256,
        "sample_outputs": {
            "policy_steps_path": str(samples_path),
            "physics_substeps_path": str(substeps_path),
        },
        "case_count": len(summaries),
        "sample_count": sample_count,
        "expected_physics_substeps": sample_count * decimation,
        "physics_substep_applied_effort_monitor": substep_monitor,
        "cases": summaries,
        "interpretation_guard": [
            "This is a runtime-only counterfactual. The base env.drive_config remains the original drive contract; PI is recorded separately.",
            "Ki=0 is the capped-P reference with an explicit assertion that integral and feedforward remain zero; nonzero Ki is an uncalibrated diagnostic assumption, not a hardware value.",
            "Every Newton actuator iteration is captured inside the physics loop and checked against the configured 2 Nm native actuator clamp plus 1e-4.",
            "The old v4 P summary supplies only the eight action vectors; it is not used as the capped-v5 performance baseline.",
            "Architecture label: side_integral_plus_native_per_wheel_P. The two-state left/right integral is a diagnostic arrangement for the vehicle's two drive motors; native P remains per proxy wheel and this is not a complete hardware controller.",
            "The eight drive cases and two brake cases are intended for exact pairing across capped-P and PI output directories.",
            "Newton applied effort is simulator output, not measured motor torque or proof of an actual hardware PI loop.",
        ],
    }
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary_data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PI COUNTERFACTUAL SUMMARY " + json.dumps({
        "summary_path": str(summary_path),
        "samples_path": str(samples_path),
        "substeps_path": str(substeps_path),
        "run_label": output_dir.name,
        "ki_nm_per_proxy_wheel_per_rad_s_side_mean_s": selected_ki,
        "case_count": len(summaries),
        "sample_count": sample_count,
        "physics_substeps_captured": substep_monitor["checked_physics_substeps"],
        "max_abs_applied_effort_nm": substep_monitor["max_abs_applied_effort_nm"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
