"""Opt-in Isaac Lab/Newton integration checks for the live field task.

These tests allocate a real SimulationContext and GPU sensors.  They are kept
out of normal unit runs and should only be enabled deliberately with
``RUN_ISAAC_PHYSICS=1`` in the Isaac Lab runtime.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ISAAC_PHYSICS") != "1",
    reason="requires an explicit RUN_ISAAC_PHYSICS=1 Isaac Lab/Newton run",
)


def test_single_channel_depth_passes_official_sb3_wrapper_without_rescaling():
    """Exercise the real adapter's image heuristic with float metric depth."""
    from isaac_field.training import _sb3_wrapper

    env = _make_env(plants=0, max_time=2.0)
    try:
        wrapped = _sb3_wrapper(env)
        observation = wrapped.reset()
        depth = observation["depth"]
        assert depth.shape == (1, 1, 160, 88)
        assert depth.dtype == np.float32
        frame = env.tasks[0].frames["depth"]
        expected = np.where(frame["valid"], frame["values"], 0.0) / env.field_cfg.depth_range
        np.testing.assert_allclose(depth[0, 0], expected, rtol=0.0, atol=1e-7)
        assert np.all((depth >= 0.0) & (depth <= 1.0))
        assert "depth" not in wrapped.observation_processors
    finally:
        env.close()


def test_native_gimbal_imu_and_drive_feedback_use_physical_signals():
    """Check measured q/qd/IMU and explicit-applied torque, never command echoes."""
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.feedback import FEEDBACK_GROUPS, FEEDBACK_VALID_SLICE

    env = _make_env(plants=0, max_time=12.)
    try:
        env.reset(seed=90231, options={"mode": "search", "start": [50., 50., 0.], "goal": [90., 50.]})
        _step(env, np.zeros(4), 30)
        task = env.tasks[0]
        observation = task._observation()
        assert observation["state"].shape == (33,)
        assert observation["feedback"].shape == (37,)
        assert np.isfinite(observation["feedback"]).all()
        acceleration = env.scene["imu"].data.lin_acc_b.torch.detach().cpu().numpy()[0]
        assert np.linalg.norm(acceleration) == pytest.approx(9.81, abs=1.)
        before = task._observation()
        again = task._observation()
        for name in before:
            np.testing.assert_array_equal(before[name], again[name])

        _step(env, np.array([.12, .12, 0., 0.]), 20)
        feedback = task._observation()["feedback"]
        sides = env._wheel_track_ids.detach().cpu().numpy()
        effort = env.rover.actuators.applied_effort.torch[:, env._wheel_ids].detach().cpu().numpy()[0]
        omega = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy()[0]
        torque_factor = ROBOT_SPEC.drive_sprocket_radius_m / (ROBOT_SPEC.wheel_radius_m
            * ROBOT_SPEC.drive_reduction_ratio * env.cfg.drive_efficiency_assumption
            * env.cfg.motor_torque_constant_nm_per_a)
        rpm_factor = (ROBOT_SPEC.wheel_radius_m / ROBOT_SPEC.drive_sprocket_radius_m
            * ROBOT_SPEC.drive_reduction_ratio * 60. / (2. * np.pi))
        for side, iq_index, rpm_index in ((0, 17, 18), (1, 19, 20)):
            # Declared simulation readout noise: 0.02 A and 1 rpm.
            assert feedback[iq_index] * 100. == pytest.approx(effort[sides == side].sum() * torque_factor, abs=.12)
            assert feedback[rpm_index] * 10000. == pytest.approx(omega[sides == side].mean() * rpm_factor, abs=6.)
            valid_index = FEEDBACK_VALID_SLICE.start + FEEDBACK_GROUPS.index(
                "motor_left" if side == 0 else "motor_right"
            )
            assert feedback[valid_index] == 1.
        _step(env, np.array([0., 0., .4, .3]), 20)
        gyro = env.scene["imu"].data.ang_vel_b.torch.detach().cpu().numpy()[0]
        measured_rates = env.rover.data.joint_vel.torch[:, env._gimbal_ids].detach().cpu().numpy()[0]
        assert np.linalg.norm(gyro) > .03
        assert np.linalg.norm(measured_rates) > .03
        feedback = task._observation()["feedback"]
        assert feedback[14] * np.pi == pytest.approx(measured_rates[0], abs=.02)
        assert feedback[16] * np.pi == pytest.approx(measured_rates[1], abs=.02)
        assert env.drive_config["actuator_model"] == "explicit_IdealPD_velocity_servo"
        assert "model_not_measured" in env.drive_config["current_feedback_source"]
        assert env.drive_config["interface"] == "independent_left_right_track_speed_coupled_pi_v7"
        assert env.drive_config["controller"] == "native_per_wheel_P_plus_two_state_side_integral_feedforward"
        assert env.drive_config["integral_gain_nm_per_proxy_wheel_per_rad_s_s"] == pytest.approx(
            env.cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s
        )
        assert env.drive_config["integral_update_period_s"] == pytest.approx(env.step_dt)
        assert env.drive_config["integral_state_count_per_env"] == 2
        assert env.drive_config["integral_state_unit"] == "Nm_per_proxy_wheel"
        assert env.drive_config["integral_error_source"] == "per_side_mean_target_minus_actual_wheel_qd_rad_s"
        assert env.drive_config["integral_broadcast"] == "same_side_integral_ff_to_each_of_7_proxy_wheels"
        assert env.drive_config["integral_limit_nm_per_proxy_wheel"] == pytest.approx(2.0)
        assert env.drive_config["integral_gain_status"].startswith("uncalibrated simulation candidate")
        assert env.drive_config["actuator_effort_limit_nm"] == pytest.approx(env.cfg.drive_effort_limit)
        assert env.drive_config["joint_effort_limit_nm"] == pytest.approx(env.cfg.drive_effort_limit)
        assert env.drive_config["gimbal_solver_joint_effort_limit_nm"] == pytest.approx(1.0)
    finally:
        env.close()


def test_native_low_speed_drive_and_braking_have_bounded_chassis_motion():
    """Catch high-gain contact transients that wheel-target tests miss."""
    env = _make_env(plants=0, max_time=15.0)
    try:
        env.reset(seed=80017, options={"start": [50., 50., 0.], "goal": [90., 50.]})
        _step(env, np.zeros(4), 20)
        start = env.tasks[0].robot.copy()
        speeds = []
        for _ in range(40):
            _step(env, np.array([.12, .12, 0., 0.]))
            speeds.append(float(env.tasks[0].robot[3]))
        driving_end = env.tasks[0].robot.copy()
        assert max(abs(value) for value in speeds) < .8
        assert .7 < np.linalg.norm(driving_end[:2] - start[:2]) < 1.3
        for _ in range(40):
            _step(env, np.zeros(4))
            speeds.append(float(env.tasks[0].robot[3]))
        stopped = env.tasks[0].robot.copy()
        assert np.linalg.norm(stopped[:2] - driving_end[:2]) < .2
        assert abs(stopped[3]) < .02
        assert min(speeds) > -.1
        assert not env.tasks[0].done
    finally:
        env.close()


def test_native_pure_differential_turn_has_measurable_yaw_without_forward_command():
    """Measure body rotation and wheel response; commanded track speed is not chassis motion."""
    env = _make_env(plants=0, max_time=15.0)
    try:
        wheel_ids, wheel_names = env.rover.find_joints(".*_wheel_joint", preserve_order=True)
        wheel_ids = np.asarray(wheel_ids, dtype=np.int64)
        left_mask = np.asarray([name.startswith("left_") for name in wheel_names])
        right_mask = np.asarray([name.startswith("right_") for name in wheel_names])
        assert len(wheel_ids) == 14 and left_mask.sum() == right_mask.sum() == 7

        for sign in (-1, 1):
            selected = None
            for magnitude in (0.2, 0.5):
                env.reset(
                    seed=90100 + (0 if sign < 0 else 10) + int(magnitude * 10),
                    options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
                )
                _step(env, np.zeros(4), 20)
                task = env.tasks[0]
                start_xy = task.robot[:2].copy()
                yaw_samples = [float(task.robot[2])]
                position_samples = [start_xy.copy()]
                omega_samples = []
                wheel_velocity_samples = []

                action = np.asarray([-sign * magnitude, sign * magnitude, 0.0, 0.0], dtype=np.float32)
                for _ in range(40):
                    _step(env, action)
                    state = task.robot.copy()
                    yaw_samples.append(float(state[2]))
                    position_samples.append(state[:2].copy())
                    omega_samples.append(float(state[4]))
                    wheel_velocity = _to_numpy(env.rover.data.joint_vel.torch)[0, wheel_ids]
                    wheel_velocity_samples.append(wheel_velocity.copy())

                yaw_unwrapped = np.unwrap(np.asarray(yaw_samples, dtype=np.float64))
                yaw_delta = float(yaw_unwrapped[-1] - yaw_unwrapped[0])
                wheel_window = np.asarray(wheel_velocity_samples[-10:])
                left_rad_s = float(wheel_window[:, left_mask].mean())
                right_rad_s = float(wheel_window[:, right_mask].mean())
                positions = np.asarray(position_samples)
                center_drift = np.linalg.norm(positions - start_xy[None, :], axis=1)
                record = {
                    "action": action.tolist(),
                    "duration_s": 2.0,
                    "yaw_delta_unwrapped_rad": yaw_delta,
                    "mean_measured_world_yaw_rate_last_10_steps_rad_s": float(np.mean(omega_samples[-10:])),
                    "left_wheel_joint_rate_last_10_mean_rad_s": left_rad_s,
                    "right_wheel_joint_rate_last_10_mean_rad_s": right_rad_s,
                    "final_center_drift_m": float(center_drift[-1]),
                    "max_center_drift_m": float(center_drift.max()),
                }
                print("NATIVE PURE-TURN PROBE " + json.dumps(
                    {"sign": sign, "magnitude": magnitude, **record}, sort_keys=True
                ), flush=True)

                counter_rotating_wheels = sign * left_rad_s < -0.05 and sign * right_rad_s > 0.05
                if sign * yaw_delta >= 0.1 and counter_rotating_wheels:
                    selected = record
                    break

            assert selected is not None, (
                f"counter-rotating track commands did not produce a clear correctly signed chassis turn "
                f"for sign={sign}; see NATIVE PURE-TURN PROBE records"
            )
            assert selected["max_center_drift_m"] < 0.75, (
                f"pivot center drift was excessive for sign={sign}: "
                f"{selected['max_center_drift_m']:.3f} m"
            )
    finally:
        env.close()


def test_native_direct_track_commands_are_independent_limited_and_accelerated():
    """Exercise the policy-to-track mapping independently of Newton integration."""
    import torch
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.action_contract import ACTION_COMPONENTS, action_config

    env = _make_env(plants=0, max_time=5.0)
    try:
        env.reset(seed=2302, options={"start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]})
        assert env.action_config == action_config()
        assert env.observation_config["schema"] == "sensor-minimal-v6"
        assert env.observation_config["state_dim"] == 33
        assert env.observation_config["action_components"] == list(ACTION_COMPONENTS)
        assert env.observation_config["previous_command_semantics"] == "normalized_direct_track_speed_policy_command"
        drive = env.drive_config
        assert drive["interface"] == "independent_left_right_track_speed_coupled_pi_v7"
        assert drive["integral_gain_nm_per_proxy_wheel_per_rad_s_s"] == pytest.approx(
            env.cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s
        )
        assert drive["integral_update_period_s"] == pytest.approx(env.step_dt)
        assert drive["integral_state_count_per_env"] == 2
        assert drive["integral_limit_nm_per_proxy_wheel"] == pytest.approx(env.cfg.drive_effort_limit)
        assert "steering_track_speed_mps" not in drive
        assert drive["velocity_gain"] == env.cfg.drive_velocity_gain
        assert drive["wheel_effort_limit_nm"] == env.cfg.drive_effort_limit

        def apply_action(values):
            action = torch.as_tensor(np.asarray(values, dtype=np.float32)[None, :], device=env.device)
            env._pre_physics_step(action)

        gimbal_start = _to_numpy(env.gimbal_target).copy()
        delta_limit = env.field_cfg.acceleration * env.step_dt
        apply_action([2.0, 0.0, 0.0, 0.0])
        first = _to_numpy(env.track_command)[0]
        assert first[0] == pytest.approx(min(env.field_cfg.max_speed, delta_limit))
        assert first[1] == pytest.approx(0.0)
        np.testing.assert_array_equal(_to_numpy(env.gimbal_target), gimbal_start)

        env.track_command.zero_()
        steps = int(np.ceil(env.field_cfg.max_speed / delta_limit)) + 2
        for _ in range(steps):
            apply_action([2.0, 0.25, 0.0, 0.0])
        np.testing.assert_allclose(
            _to_numpy(env.track_command)[0],
            [env.field_cfg.max_speed, 0.25 * env.field_cfg.max_speed],
            rtol=0.0,
            atol=1e-6,
        )
        wheel_sides = _to_numpy(env._wheel_track_ids)
        left_ids = np.flatnonzero(wheel_sides == 0)
        right_ids = np.flatnonzero(wheel_sides == 1)
        assert len(left_ids) == len(right_ids) == 7
        wheel_targets = _to_numpy(env.wheel_target_velocity)[0]
        np.testing.assert_allclose(
            wheel_targets[left_ids], env.field_cfg.max_speed / ROBOT_SPEC.wheel_radius_m,
            rtol=3 * np.finfo(np.float32).eps, atol=1e-6
        )
        np.testing.assert_allclose(
            wheel_targets[right_ids], 0.25 * env.field_cfg.max_speed / ROBOT_SPEC.wheel_radius_m,
            rtol=3 * np.finfo(np.float32).eps, atol=1e-6
        )
        np.testing.assert_array_equal(_to_numpy(env.gimbal_target), gimbal_start)

        env.track_command.zero_()
        for _ in range(steps):
            apply_action([0.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(
            _to_numpy(env.track_command)[0], [0.0, env.field_cfg.max_speed], rtol=0.0, atol=1e-6
        )
        np.testing.assert_array_equal(_to_numpy(env.gimbal_target), gimbal_start)
    finally:
        env.close()


def _make_env(
    *, plants: int, max_time: float, num_envs: int = 1, auto_reset: bool = False,
    terrain_enabled: bool = False, terrain_amplitude: float = 0.35, seed: int = 2301,
):
    # Import only inside skipped tests so ordinary CPU-only pytest collection
    # does not import Kit/Newton modules.
    from isaac_field.env import make_env
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=plants,
        terrain_enabled=terrain_enabled,
        terrain_amplitude=terrain_amplitude,
        validate_scene=False,
        max_time=max_time,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    return make_env(cfg, mode="search", num_envs=num_envs, seed=seed, auto_reset=auto_reset)


def _step(env, action: np.ndarray, count: int = 1):
    import torch

    tensor = torch.as_tensor(np.asarray(action, dtype=np.float32)[None, :], device=env.device)
    result = None
    for _ in range(count):
        result = env.step(tensor)
    return result


def _to_numpy(value):
    return value.detach().cpu().numpy()


def _sort_xy(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort((points[:, 1], points[:, 0]))]


def _print_newton_counts(env) -> None:
    manager = env.sim.physics_manager
    model = manager.get_model()
    names = ("body_count", "joint_dof_count", "joint_coord_count", "shape_count")
    counts = {name: getattr(model, name, None) for name in names}
    solver = getattr(manager, "_solver", None)
    solver_model = getattr(solver, "_mjc_model", None)
    counts["solver"] = type(solver).__name__ if solver is not None else None
    counts["mjc_nv"] = getattr(solver_model, "nv", None)
    print(f"ISAAC integration Newton model counts: {counts}", flush=True)
    return counts


def test_live_observations_wheel_drive_gimbal_joint_and_lidar_plane():
    from isaac_field.assets import GIMBAL_PITCH_JOINT, ROBOT_SPEC

    env = _make_env(plants=8, max_time=5.0)
    try:
        obs, _ = env.reset(
            seed=31,
            options={"mode": "search", "start": [25.0, 25.0, 0.0]},
        )
        counts = _print_newton_counts(env)
        if counts["joint_dof_count"] is not None:
            assert int(counts["joint_dof_count"]) == 22, (
                "plants must be world-fixed Newton bodies with no added joint DOFs; "
                f"expected 22 rover DOFs, got {counts['joint_dof_count']}"
            )
        policy_obs = obs["policy"]
        assert tuple(policy_obs["lidar"].shape) == (1, 1, 400)
        assert tuple(policy_obs["depth"].shape) == (1, 1, 160, 88)
        # The official SB3 wrapper accepts the declared normalized-image
        # envelope; sensor values themselves still use zero or Z / range.
        depth_space = env.single_observation_space["policy"]["depth"]
        assert np.all(depth_space.low == -1.0) and np.all(depth_space.high == 1.0)
        assert bool((policy_obs["depth"] >= 0.0).all())
        assert bool((policy_obs["depth"] <= 1.0).all())
        assert tuple(policy_obs["detections"].shape) == (1, 16, 8)
        assert tuple(policy_obs["state"].shape) == (1, 33)
        assert _to_numpy(policy_obs["depth"]).dtype == np.float32
        assert _to_numpy(policy_obs["state"]).dtype == np.float32
        assert all(bool(value.isfinite().all()) for value in policy_obs.values())

        depth_frame = env.tasks[0].frames["depth"]
        policy_depth = _to_numpy(policy_obs["depth"])[0, 0]
        assert np.all(policy_depth[~depth_frame["valid"]] == 0.0)
        np.testing.assert_allclose(
            policy_depth[depth_frame["valid"]],
            depth_frame["values"][depth_frame["valid"]] / env.field_cfg.depth_range,
        )

        wheel_ids, wheel_names = env.rover.find_joints(".*_wheel_joint", preserve_order=True)
        expected_names = {f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs}
        assert len(wheel_ids) == 14
        assert set(wheel_names) == expected_names
        assert set(env._wheel_names) == expected_names

        start_xy = env.tasks[0].robot[:2].copy()
        _step(env, np.asarray([0.2, 0.2, 0.0, 0.0], dtype=np.float32), count=20)
        travelled = float(np.linalg.norm(env.tasks[0].robot[:2] - start_xy))
        assert travelled > 0.1, f"forward command moved only {travelled:.4f} m"

        pitch_joint_id = env.rover.joint_names.index(GIMBAL_PITCH_JOINT)
        pitch_before = float(_to_numpy(env.rover.data.joint_pos.torch)[0, pitch_joint_id])
        lidar = env.scene["lidar"]
        # World ray directions are exposed on the Newton caster itself; its
        # data object only contains hit outputs.  A full 360-degree plane has
        # near-zero mean z even when pitched, so test its vertical spread.
        plane_before = _to_numpy(lidar.ray_directions_w.torch)[0, :, 2].copy()
        _step(env, np.asarray([0.0, 0.0, 0.0, 0.75], dtype=np.float32), count=20)
        pitch_after = float(_to_numpy(env.rover.data.joint_pos.torch)[0, pitch_joint_id])
        plane_after = _to_numpy(lidar.ray_directions_w.torch)[0, :, 2].copy()
        assert pitch_after > pitch_before + 0.04, (
            f"positive pitch command did not raise the actual joint: {pitch_before:.4f} -> {pitch_after:.4f} rad"
        )
        assert float(np.ptp(plane_after) - np.ptp(plane_before)) > 0.02, (
            "the live LiDAR plane did not follow the pitched gimbal"
        )
    finally:
        env.close()


def test_reset_randomizes_clear_spawn_and_matches_live_plant_and_camera_state():
    env = _make_env(plants=8, max_time=5.0)
    try:
        saved_layouts = []
        saved_starts = []
        for seed in (401, 402):
            obs, _ = env.reset(seed=seed, options={"mode": "search", "random_start": True})
            task = env.tasks[0]
            saved_layouts.append(task.scene.xy.copy())
            saved_starts.append(task.robot[:3].copy())
            assert task.scene.navigable(task.robot[:2]), "sampled spawn overlaps the vehicle-clearance mask"
            assert task.frames.keys() >= {"lidar", "depth", "camera"}

            origin = _to_numpy(env.scene.env_origins)[0]
            actual_plant_xyz = _to_numpy(env.plant_positions())[0]
            expected_plant_xyz = np.column_stack((task.scene.xy, task.scene.root_z)) + origin[None, :]
            assert actual_plant_xyz.shape == expected_plant_xyz.shape
            np.testing.assert_allclose(
                _sort_xy(actual_plant_xyz[:, :2]),
                _sort_xy(expected_plant_xyz[:, :2]),
                atol=0.02,
                err_msg="live corn collection poses do not match the reset task layout",
            )
            np.testing.assert_allclose(
                np.sort(actual_plant_xyz[:, 2]), np.sort(expected_plant_xyz[:, 2]), atol=0.02
            )

            # Initial frame metadata must describe the current reset pose and
            # mount, not the camera frame cached from the previous episode.
            depth_frame = task.frames["depth"]
            np.testing.assert_allclose(depth_frame["pose"][:3], task.robot[:3], atol=1e-4)
            np.testing.assert_allclose(depth_frame["pose"][3:], task.gimbal, atol=1e-4)
            camera_data = env.scene["depth"].data
            camera_pos_local = _to_numpy(camera_data.pos_w.torch)[0] - origin
            np.testing.assert_allclose(depth_frame["origin"], camera_pos_local, atol=0.01)
            actual_k = _to_numpy(camera_data.intrinsic_matrices.torch)[0]
            np.testing.assert_allclose(depth_frame["intrinsics"], actual_k, atol=1e-5)
            assert depth_frame["calibration_source"].startswith("Isaac Lab Camera.data.intrinsic_matrices")

            policy_obs = obs["policy"]
            assert all(bool(value.isfinite().all()) for value in policy_obs.values())

        assert not np.array_equal(saved_layouts[0], saved_layouts[1]), "different reset seeds reused the same plant map"
        assert not np.allclose(saved_starts[0], saved_starts[1]), "different reset seeds reused the same blank spawn"
    finally:
        env.close()


def test_live_crop_stem_returns_native_range_without_a_leaf_collider():
    """A nearby authored stem must be queryable by Newton rays without crop collisions."""
    import newton

    env = _make_env(plants=8, max_time=10.0)
    try:
        # Keep the actual randomized layout, then place the robot in a clear
        # approach lane facing the outermost stem.  The row's neighboring
        # plants lie behind that first stem along the ray.
        env.reset(seed=67031, options={"mode": "search", "random_start": True})
        scene = env.tasks[0].scene
        plant_id = int(np.argmin(scene.xy[:, 0]))
        target = scene.xy[plant_id].astype(np.float64)
        start = np.asarray([target[0] - 1.5, target[1], 0.0], dtype=np.float64)
        assert scene.pose_clear(start[:2], start[2], include_small_rocks=True), (
            f"approach pose intersects the randomized crop/rock layout: plant={plant_id}, "
            f"target={target.tolist()}, start={start.tolist()}"
        )
        env.reset(seed=67032, options={
            "mode": "search",
            "scene": scene,
            "start": start,
            "random_start": False,
        })
        task = env.tasks[0]
        np.testing.assert_allclose(task.robot[:3], start, atol=1e-4)

        rangefinder = env.scene["rangefinder"]
        range_value = float(_to_numpy(rangefinder.data.ray_distances.torch)[0].reshape(-1)[0])
        expected_range = float(target[0] - task.view_origin[0])
        lidar = env.scene["lidar"]
        lidar_ranges = _to_numpy(lidar.data.ray_distances.torch)[0].reshape(-1)
        directions = _to_numpy(lidar.ray_directions_w.torch)[0]
        forward_ray = int(np.argmax(directions[:, 0]))
        lidar_forward = float(lidar_ranges[forward_ray])

        model = env.sim.physics_manager.get_model()
        labels = list(model.shape_label)
        flags = model.shape_flags.numpy()
        leaf_rows = []
        for label, flag in zip(labels, flags, strict=True):
            if f"Corn_{plant_id}" in label and "/Leaf_" in label:
                value = int(flag)
                leaf_rows.append({
                    "label": label,
                    "visible": bool(value & int(newton.ShapeFlags.VISIBLE)),
                    "collide_shapes": bool(value & int(newton.ShapeFlags.COLLIDE_SHAPES)),
                })

        record = {
            "seed_layout": 67031,
            "plant_id": plant_id,
            "target_xy_m": target.tolist(),
            "robot_xy_yaw": task.robot[:3].tolist(),
            "sensor_origin_m": task.view_origin.tolist(),
            "expected_stem_range_m": expected_range,
            "rangefinder_m": range_value,
            "lidar_forward_ray_index": forward_ray,
            "lidar_forward_ray_world_direction": directions[forward_ray].tolist(),
            "lidar_forward_range_m": lidar_forward,
            "selected_leaf_shape_flags": leaf_rows,
        }
        print("NATIVE CROP RAYCAST " + json.dumps(record, sort_keys=True), flush=True)

        assert .12 <= range_value <= 3.0, f"native rangefinder missed the nearby stem: {record}"
        assert .12 <= lidar_forward <= 3.0, f"forward LiDAR beam missed the nearby stem: {record}"
        assert abs(range_value - expected_range) < .4, (
            f"rangefinder return does not match the selected stem distance: {record}"
        )
        assert leaf_rows, f"Newton imported no leaf shapes for selected plant {plant_id}"
        assert any(row["visible"] for row in leaf_rows), f"selected plant leaves are not VISIBLE: {record}"
        assert not any(row["collide_shapes"] for row in leaf_rows), (
            f"crop leaves unexpectedly entered the collision-shape set: {record}"
        )
    finally:
        env.close()


def test_two_env_autoreset_returns_fresh_first_observations():
    import torch

    env = _make_env(plants=0, max_time=0.1, num_envs=2, auto_reset=True)
    try:
        env.reset(seed=51)
        old_scene_ids = [task.scene_id for task in env.tasks]
        action = torch.zeros((2, 4), dtype=torch.float32, device=env.device)
        env.step(action)
        obs, _, terminated, truncated, _ = env.step(action)
        assert bool((terminated | truncated).all())
        assert [task.scene_id for task in env.tasks] == [scene_id + 1 for scene_id in old_scene_ids]
        for index, task in enumerate(env.tasks):
            assert task.time == 0.0 and not task.done
            assert task.frames.keys() >= {"lidar", "depth", "camera"}
            frame = task.frames["depth"]
            origin = _to_numpy(env.scene.env_origins)[index]
            camera_pos_local = _to_numpy(env.scene["depth"].data.pos_w.torch)[index] - origin
            np.testing.assert_allclose(frame["origin"], camera_pos_local, atol=0.01)
            np.testing.assert_allclose(frame["pose"][:3], task.robot[:3], atol=1e-4)
        assert tuple(obs["policy"]["depth"].shape) == (2, 1, 160, 88)
        assert tuple(obs["policy"]["feedback"].shape) == (2, 37)
        for index, task in enumerate(env.tasks):
            assert task._feedback_snapshot.sample_time == 0.
            np.testing.assert_allclose(obs["policy"]["feedback"][index].cpu().numpy(), task._observation()["feedback"])
    finally:
        env.close()


def test_rough_terrain_zero_action_resets_remain_stable_and_wheels_contact_ground():
    from isaac_field.assets import ROBOT_SPEC

    env = _make_env(
        plants=8,
        max_time=5.0,
        terrain_enabled=True,
        terrain_amplitude=0.35,
        seed=7,
    )
    try:
        wheel_body_ids = [
            env.rover.body_names.index(f"{wheel.name}_wheel_link")
            for wheel in ROBOT_SPEC.wheel_specs
        ]
        origin = _to_numpy(env.scene.env_origins)[0]
        for reset_seed in (701, 702, 703, 704, 705, 706):
            obs, _ = env.reset(seed=reset_seed, options={"mode": "search", "random_start": True})
            pose = _to_numpy(env.rover.data.root_link_pose_w.torch)[0]
            assert np.isfinite(pose).all(), f"non-finite spawn pose on reset seed {reset_seed}"
            task = env.tasks[0]
            assert task.scene.terrain is not None
            assert task.reason != "outside_or_overturned"
            assert np.all((task.robot[:2] >= 0.0) & (task.robot[:2] <= task.cfg.size))

            body_poses = _to_numpy(env.rover.data.body_link_pose_w.torch)[0, wheel_body_ids]
            wheel_xy = body_poses[:, :2] - origin[None, :2]
            terrain_z = np.asarray(task.scene.terrain.surface_height(wheel_xy), dtype=np.float32)
            bottom_z = body_poses[:, 2] - origin[2] - ROBOT_SPEC.wheel_radius_m
            clearance = bottom_z - terrain_z
            assert float(clearance.min()) >= -0.01, (
                f"wheel bottom penetrates terrain by {-float(clearance.min()):.4f} m "
                f"at reset seed {reset_seed}"
            )
            assert float(clearance.max()) <= 0.03, (
                f"wheel bottom is {float(clearance.max()):.4f} m above terrain "
                f"at reset seed {reset_seed}"
            )

            assert all(bool(value.isfinite().all()) for value in obs["policy"].values())
            for _ in range(20):
                _step(env, np.zeros(4, dtype=np.float32))
                pose = _to_numpy(env.rover.data.root_link_pose_w.torch)[0]
                assert np.isfinite(pose).all(), f"non-finite body pose on reset seed {reset_seed}"
                task = env.tasks[0]
                assert task.reason != "outside_or_overturned", (
                    f"vehicle fell or left the field after reset seed {reset_seed} "
                    f"at t={task.time:.2f}s"
                )
                assert np.all((task.robot[:2] >= 0.0) & (task.robot[:2] <= task.cfg.size)), (
                    f"vehicle left field bounds after reset seed {reset_seed}"
                )
                assert abs(task.body_pose[1]) < 1.2 and abs(task.body_pose[2]) < 1.2, (
                    f"vehicle overturned after reset seed {reset_seed}: "
                    f"pitch={task.body_pose[1]:.3f}, roll={task.body_pose[2]:.3f}"
                )
    finally:
        env.close()


def test_validated_goto_reset_keeps_the_exact_validated_start_goal_pair():
    """Keep the physics spawn/target equal to the exact pair checked by prepare()."""
    from isaac_field.env import make_env
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=8,
        terrain_enabled=False,
        validate_scene=True,
        max_time=5.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    env = make_env(
        cfg,
        mode="goto",
        num_envs=1,
        seed=2301,
        auto_reset=False,
        goal_distance=(1.0, 3.0),
        goal_angle=np.deg2rad(60.0),
    )
    try:
        task = env.tasks[0]
        prepare = task.prepare
        prepared_pairs = []

        def spy_prepare(seed=None, options=None):
            actual_options = dict(options or {})
            start_option = np.asarray(actual_options["start"], dtype=np.float64).copy()
            goal_option = np.asarray(actual_options["goal"], dtype=np.float64).copy()
            try:
                result = prepare(seed=seed, options=options)
            except Exception as exc:
                prepared_pairs.append({
                    "accepted": False,
                    "start_option": start_option,
                    "goal_option": goal_option,
                    "prepared_start": task.robot[:3].copy(),
                    "prepared_goal": task.goal.copy(),
                    "validation": None if task.validation is None else dict(task.validation),
                    "error": repr(exc),
                })
                raise
            prepared_pairs.append({
                "accepted": True,
                "start_option": start_option,
                "goal_option": goal_option,
                "prepared_start": task.robot[:3].copy(),
                "prepared_goal": task.goal.copy(),
                "validation": None if task.validation is None else dict(task.validation),
                "error": None,
            })
            return result

        task.prepare = spy_prepare
        for seed in (8, 81001, 81002):
            prepared_pairs.clear()
            env.reset(seed=seed)
            assert prepared_pairs, f"task.prepare was not called for reset seed {seed}"
            accepted = [record for record in prepared_pairs if record["accepted"]]
            assert accepted, (
                f"no prepare() call accepted a pair for reset seed {seed}: "
                f"{[record['error'] for record in prepared_pairs]}"
            )
            record = accepted[-1]
            validation = record["validation"]
            assert validation is not None
            assert validation["connected"] is True
            assert validation["all_free_connected"] is True
            assert validation["start_goal_connected"] is True

            # task.robot is refreshed from the live articulation after reset;
            # task.goal is the live target consumed by the environment.
            live_start = task.robot[:3].copy()
            live_goal = task.goal.copy()
            np.testing.assert_allclose(live_start[:2], record["start_option"][:2], atol=1e-3)
            yaw_error = np.arctan2(
                np.sin(live_start[2] - record["start_option"][2]),
                np.cos(live_start[2] - record["start_option"][2]),
            )
            assert abs(float(yaw_error)) < 2e-3
            np.testing.assert_allclose(live_goal, record["goal_option"], atol=1e-6)
            np.testing.assert_allclose(record["prepared_start"], record["start_option"], atol=1e-6)
            np.testing.assert_allclose(record["prepared_goal"], record["goal_option"], atol=1e-6)

            assert task.scene.pose_clear(live_start[:2], live_start[2], include_small_rocks=True)
            delta = live_goal - live_start[:2]
            distance = float(np.linalg.norm(delta))
            bearing = float(np.arctan2(
                np.sin(np.arctan2(delta[1], delta[0]) - live_start[2]),
                np.cos(np.arctan2(delta[1], delta[0]) - live_start[2]),
            ))
            assert 1.0 - 1e-2 <= distance <= 3.0 + 1e-2
            assert abs(bearing) <= np.deg2rad(60.0) + 2e-2
    finally:
        env.close()


def test_native_pi_drive_moves_and_effort_limit_caps_each_newton_physics_iteration():
    """Exercise actual PI feedback and verify the native 2 Nm cap each physics step."""
    import runpy
    from pathlib import Path

    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    root = Path(__file__).resolve().parents[1]
    capture_kernel = runpy.run_path(str(
        Path(__file__).with_name("newton_effort_capture.py")
    ))["record_wheel_applied_effort"]
    env = _make_env(plants=0, max_time=8.0, seed=90231)
    try:
        drive_config = env.drive_config
        assert drive_config["interface"] == "independent_left_right_track_speed_coupled_pi_v7"
        assert drive_config["actuator_effort_limit_nm"] == pytest.approx(2.0)
        assert drive_config["joint_effort_limit_nm"] == pytest.approx(2.0)
        drive_cfg = env.cfg.scene.robot.actuators["drive"]
        assert drive_cfg.actuator_effort_limit == pytest.approx(2.0)
        assert drive_cfg.joint_effort_limit == pytest.approx(2.0)

        env.reset(
            seed=90231,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        decimation = int(env.cfg.decimation)
        drive_policy_steps = 40
        clamp_policy_steps = 3
        capacity = decimation * (drive_policy_steps + clamp_policy_steps)
        wheel_ids_torch = env._wheel_ids.to(dtype=torch.int32).contiguous()
        wheel_ids_warp = wp.from_torch(wheel_ids_torch)
        applied_effort_warp = env.rover.actuators.applied_effort.warp
        tape = wp.empty((capacity, 14), dtype=wp.float32, device=env.device)
        counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_every_physics_iteration():
            wp.launch(
                capture_kernel,
                dim=1,
                inputs=[applied_effort_warp, wheel_ids_warp, tape, counter, capacity],
                device=env.device,
            )

        NewtonManager.register_post_actuator_callback(capture_every_physics_iteration)
        start_xy = env.tasks[0].robot[:2].copy()
        _step(env, np.array([0.12, 0.12, 0.0, 0.0], dtype=np.float32), drive_policy_steps)

        controller = env._drive_pi
        assert controller.integral.shape == (1, 2)
        assert torch.isfinite(controller.integral).all()
        assert torch.isfinite(controller.feedforward).all()
        assert bool((controller.integral.abs() > 1e-6).any()), "the real drive error must exercise the I state"
        torch.testing.assert_close(controller.feedforward[:, :7], controller.integral[:, :1].expand(-1, 7))
        torch.testing.assert_close(controller.feedforward[:, 7:], controller.integral[:, 1:2].expand(-1, 7))
        body = env.tasks[0].robot
        wheel_qd = env.rover.data.joint_vel.torch[:, env._wheel_ids]
        assert np.isfinite(body).all()
        assert np.linalg.norm(body[:2] - start_xy) > 0.2, "the PI controller should accompany real chassis motion"
        assert torch.isfinite(wheel_qd).all()
        assert bool((wheel_qd.abs() > 0.05).any())

        original_apply = env._apply_action
        wheel_ff = torch.full_like(env.wheel_target_velocity, 4.0)

        def apply_over_limit_feedforward():
            original_apply()
            env.rover.actuators.target_command.set_effort_index(
                value=wheel_ff, joint_ids=env._wheel_ids,
            )

        env._apply_action = apply_over_limit_feedforward
        _step(env, np.zeros(4, dtype=np.float32), clamp_policy_steps)

        captured_count = int(counter.numpy()[0])
        assert captured_count == capacity == (drive_policy_steps + clamp_policy_steps) * decimation
        captured = np.asarray(tape.numpy()[:captured_count], dtype=np.float64)
        assert captured.shape == (capacity, 14)
        assert np.isfinite(captured).all()
        max_abs_effort = float(np.max(np.abs(captured)))
        assert max_abs_effort <= 2.0 + 1e-5
        assert max_abs_effort >= 1.9, "test feedforward should exercise the actuator clamp"
    finally:
        env.close()


def test_native_two_environment_partial_reset_preserves_other_pi_state_and_feedforward():
    """A reset of env 1 must clear only its side integrators and effort target."""
    import torch

    env = _make_env(plants=0, max_time=8.0, num_envs=2, seed=90232)
    try:
        env.reset(
            seed=90232,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        actions = torch.tensor(
            [[0.8, 0.4, 0.0, 0.0], [-0.6, 0.2, 0.0, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )
        for _ in range(16):
            env.step(actions)

        controller = env._drive_pi
        assert controller.integral.shape == (2, 2)
        assert torch.isfinite(controller.integral).all()
        assert bool((controller.integral.abs() > 1e-6).all()), (
            "both environments should accumulate nonzero left/right integral state before reset"
        )
        ff_before = controller.feedforward.clone()
        integral_before = controller.integral.clone()
        effort_targets = env.rover.actuators.target_command.effort.torch[:, env._wheel_ids]
        effort_before = effort_targets.clone()
        assert torch.isfinite(effort_before).all()
        torch.testing.assert_close(effort_before, ff_before)

        env._reset_idx(torch.tensor([1], dtype=torch.long, device=env.device))

        torch.testing.assert_close(controller.integral[0], integral_before[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.feedforward[0], ff_before[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.integral[1], torch.zeros_like(controller.integral[1]))
        torch.testing.assert_close(controller.feedforward[1], torch.zeros_like(controller.feedforward[1]))
        torch.testing.assert_close(controller.last_error[1], torch.zeros_like(controller.last_error[1]))
        torch.testing.assert_close(controller.last_p_estimate[1], torch.zeros_like(controller.last_p_estimate[1]))
        assert not controller.last_blocked[1].any()

        effort_after = env.rover.actuators.target_command.effort.torch[:, env._wheel_ids]
        torch.testing.assert_close(effort_after[0], effort_before[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(effort_after[1], torch.zeros_like(effort_after[1]))
    finally:
        env.close()


"""Native two-world same-side mimic acceptance test for drive v7."""

def test_native_two_environment_same_side_mimic_isolated_and_partial_reset():
    """Verify final Newton/MJWarp rows, independent worlds, effort cap and reset locality."""
    import runpy
    from pathlib import Path

    import numpy as np
    import pytest
    import torch
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaac_field.env import make_env
    from isaac_field.newton_drive import same_side_wheel_coupling_metadata
    from isaac_field.task import Config

    root = Path(__file__).resolve().parents[1]
    capture_kernel = runpy.run_path(str(root / "tests" / "newton_effort_capture.py"))["record_two_env_wheel_qd"]
    cfg = Config(
        size=100.0, plants=0, terrain_enabled=False, validate_scene=False,
        max_time=8.0, lidar_rays=400, depth_height=160, depth_width=88,
    )
    env = make_env(cfg, mode="search", num_envs=2, seed=90232, auto_reset=False)
    try:
        assert env.cfg.couple_same_side_wheels is True
        assert env.cfg.sim.dt == pytest.approx(0.001, abs=1e-12)
        assert int(env.cfg.decimation) == 50
        assert env.step_dt == pytest.approx(0.05, abs=1e-12)
        drive = env.drive_config
        assert drive["interface"] == "independent_left_right_track_speed_coupled_pi_v7"
        assert drive["same_side_wheel_coupling"] == same_side_wheel_coupling_metadata(True)
        assert "front_guide" not in drive
        assert drive["physics_contact"]["physics_dt_s"] == pytest.approx(0.001)
        assert drive["physics_contact"]["decimation"] == 50
        assert drive["physics_contact"]["policy_dt_s"] == pytest.approx(0.05)
        assert drive["physics_contact"]["cone"] == "elliptic"
        assert drive["physics_contact"]["impratio"] == pytest.approx(1.0)
        actual_materials = drive["physics_contact"]["actual_wheel_ground_shape_materials"]
        for role in ("ground", "wheel"):
            assert len(actual_materials[role]["mu"]) == 1
            assert actual_materials[role]["mu"][0] == pytest.approx(0.8)
            assert len(actual_materials[role]["kf"]) == 1
            assert actual_materials[role]["kf"][0] == pytest.approx(1000.0)
        assert drive["actuator_effort_limit_nm"] == pytest.approx(2.0)
        assert drive["velocity_gain"] == pytest.approx(0.04)
        assert drive["integral_gain_nm_per_proxy_wheel_per_rad_s_s"] == pytest.approx(0.1)

        model = env.sim.physics_manager.get_model()
        assert env._same_side_mimic_model_audit["worlds_checked"] == 2
        assert env._same_side_mimic_model_audit["relations_per_world"] == 12
        assert env._same_side_mimic_model_audit["total_mimic_rows"] == 24
        assert env._same_side_mimic_model_audit["total_equality_rows"] == 24
        assert env._same_side_mimic_model_audit["cross_world_pairs"] == 0
        hook = env._same_side_mimic_hook
        solver = env.sim.physics_manager._solver
        assert len(hook.worlds) == 2 and len(hook.relations) == 24
        host, mjw = solver.mj_model, solver.mjw_model
        host_neq, mjw_neq = int(host.neq), int(mjw.neq)
        assert host_neq == mjw_neq
        host_eq = tuple(np.asarray(getattr(host, name)).reshape(-1)
                        for name in ("eq_type", "eq_obj1id", "eq_obj2id"))
        mjw_eq = tuple(np.asarray(getattr(mjw, name).numpy() if hasattr(getattr(mjw, name), "numpy")
                                  else getattr(mjw, name)).reshape(-1)
                       for name in ("eq_type", "eq_obj1id", "eq_obj2id"))
        for host_rows, mjw_rows in zip(host_eq, mjw_eq, strict=True):
            np.testing.assert_array_equal(host_rows, mjw_rows)
            assert len(host_rows) == host_neq
        mjw_data = solver.mjw_data
        assert int(mjw_data.nworld) == 2
        assert tuple(mjw_data.qpos.shape) == (2, int(host.nq))
        assert tuple(mjw_data.qvel.shape) == (2, int(host.nv))

        env.reset(
            seed=90232,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        labels = [str(value) for value in model.joint_label]
        joint_world_raw = model.joint_world
        joint_world = np.asarray(joint_world_raw.numpy() if hasattr(joint_world_raw, "numpy")
                                 else joint_world_raw).reshape(-1).astype(np.int64)
        qd_start_raw = model.joint_qd_start
        joint_qd_start = np.asarray(qd_start_raw.numpy() if hasattr(qd_start_raw, "numpy")
                                    else qd_start_raw).reshape(-1).astype(np.int64)
        wheel_names = [name for side in ("left", "right") for name in hook.names[side]]
        offsets = np.empty((2, 14), dtype=np.int32)
        for world in range(2):
            for wheel, basename in enumerate(wheel_names):
                hits = [index for index, label in enumerate(labels)
                        if joint_world[index] == world and label.rsplit("/", 1)[-1] == basename]
                assert len(hits) == 1, f"world {world}, {basename}: found {hits}"
                offsets[world, wheel] = joint_qd_start[hits[0]]
        state_qd = NewtonManager.get_state_0().joint_qd
        assert int(offsets.max()) < int(state_qd.shape[0])
        decimation = int(env.cfg.decimation)
        capacity = 60 * decimation  # one second settle + two seconds command
        offsets_warp = wp.from_torch(torch.as_tensor(offsets.reshape(-1), device=env.device).contiguous())
        qd_tape = wp.empty((capacity, 28), dtype=wp.float32, device=env.device)
        qd_counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_qd_before_solver():
            wp.launch(
                capture_kernel,
                dim=1,
                inputs=[NewtonManager.get_state_0().joint_qd, offsets_warp, qd_tape, qd_counter, capacity],
                device=env.device,
            )

        NewtonManager.register_post_actuator_callback(capture_qd_before_solver)
        actions = torch.tensor([[0.7, 0.2, 0.0, 0.0], [-0.3, 0.6, 0.0, 0.0]],
                               dtype=torch.float32, device=env.device)
        zero = torch.zeros_like(actions)
        for _ in range(20):
            env.step(zero)
        q_rows, qd_rows, command_rows = [], [], []
        for _ in range(40):
            env.step(actions)
            q = env.rover.data.joint_pos.torch[:, env._wheel_ids].detach().cpu().numpy().copy()
            qd = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy().copy()
            commands = env.track_command.detach().cpu().numpy().copy()
            assert np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(commands).all()
            q_rows.append(q)
            qd_rows.append(qd)
            command_rows.append(commands)
        assert int(qd_counter.numpy().reshape(-1)[0]) == capacity
        native_qd = wp.to_torch(qd_tape)[:capacity].detach().cpu().numpy().reshape(capacity, 2, 14)
        assert np.isfinite(native_qd).all()
        side_indices = [np.asarray([i for i, name in enumerate(wheel_names) if name.startswith(side)], dtype=np.int64)
                        for side in ("left", "right")]
        q_window = np.asarray(q_rows[-20:])
        qd_window = native_qd[-20 * decimation:]
        for world in range(2):
            for indices in side_indices:
                q_error = q_window[:, world, indices] - q_window[:, world, indices[0], None]
                qd_error = qd_window[:, world, indices] - qd_window[:, world, indices[0], None]
                assert float(np.max(np.abs(q_error))) <= 0.02
                assert float(np.percentile(np.abs(qd_error), 95.0)) <= 0.0926
                assert float(np.max(np.abs(qd_error))) <= 0.463
        final_qd = q_rows[-1]
        for indices in side_indices:
            gap = abs(float(final_qd[0, indices].mean() - final_qd[1, indices].mean()))
            assert gap > 0.1, f"different commands did not separate side qd means: {gap:.6f}"
        assert np.linalg.norm(command_rows[-1][0] - command_rows[-1][1]) > 0.02
        root_twist = env.rover.data.root_com_vel_w.torch.detach().cpu().numpy().copy()
        assert np.isfinite(root_twist).all()
        assert float(np.linalg.norm(root_twist[0] - root_twist[1])) > 0.02
        applied_effort = env.rover.actuators.applied_effort.torch[:, env._wheel_ids]
        assert torch.isfinite(applied_effort).all()
        assert float(applied_effort.abs().max()) <= float(env.cfg.drive_effort_limit) + 1e-5

        # Snapshot the non-reset world before partial reset.
        controller = env._drive_pi
        q_before = q_rows[-1][1].copy()
        qd_before = qd_rows[-1][1].copy()
        protected = {
            "integral": controller.integral[1].clone(),
            "feedforward": controller.feedforward[1].clone(),
            "last_error": controller.last_error[1].clone(),
            "last_p_estimate": controller.last_p_estimate[1].clone(),
            "last_blocked": controller.last_blocked[1].clone(),
            "track_command": env.track_command[1].clone(),
            "root_pose": env.rover.data.root_link_pose_w.torch[1].detach().clone(),
            "root_twist": env.rover.data.root_com_vel_w.torch[1].detach().clone(),
            "target_effort": env.rover.actuators.target_command.effort.torch[1, env._wheel_ids].clone(),
            "applied_effort": applied_effort[1].clone(),
        }
        env._reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
        torch.testing.assert_close(controller.integral[1], protected["integral"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.feedforward[1], protected["feedforward"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_error[1], protected["last_error"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_p_estimate[1], protected["last_p_estimate"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_blocked[1], protected["last_blocked"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.track_command[1], protected["track_command"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.data.root_link_pose_w.torch[1], protected["root_pose"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.data.root_com_vel_w.torch[1], protected["root_twist"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.actuators.target_command.effort.torch[1, env._wheel_ids],
                                   protected["target_effort"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.actuators.applied_effort.torch[1, env._wheel_ids],
                                   protected["applied_effort"], rtol=0.0, atol=0.0)
        q_after = env.rover.data.joint_pos.torch[:, env._wheel_ids].detach().cpu().numpy()
        qd_after = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy()
        np.testing.assert_allclose(q_after[1], q_before, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(qd_after[1], qd_before, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(q_after[0], 0.0, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(qd_after[0], 0.0, atol=1e-6, rtol=0.0)
        # DirectRLEnv.reset completes `_reset_idx` with a scene write, forward,
        # and `_compute_observations`. Finish that lifecycle for this deliberate
        # partial reset before exercising the regular policy step path.
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.obs_buf = env._compute_observations()
        assert {"lidar", "depth", "camera"}.issubset(env.tasks[0].frames)
        policy_obs = env.obs_buf["policy"]
        assert policy_obs and all(torch.isfinite(value).all() for value in policy_obs.values())

        step_obs, rewards, terminated, truncated, _ = env.step(actions)
        assert all(torch.isfinite(value).all() for value in step_obs["policy"].values())
        assert torch.isfinite(rewards).all()
        assert not bool(terminated.any()) and not bool(truncated.any())
        state_after_transient = NewtonManager.get_state_0()
        assert torch.isfinite(wp.to_torch(state_after_transient.joint_q)).all()
        assert torch.isfinite(wp.to_torch(state_after_transient.joint_qd)).all()
    finally:
        env.close()
