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
        assert drive["same_side_wheel_coupling"]["enabled"] is True
        assert drive["same_side_wheel_coupling"]["relations_per_world"] == 12
        assert drive["physics_contact"]["physics_dt_s"] == pytest.approx(0.001)
        assert drive["physics_contact"]["decimation"] == 50
        assert drive["physics_contact"]["policy_dt_s"] == pytest.approx(0.05)
        assert drive["physics_contact"]["cone"] == "elliptic"
        assert drive["physics_contact"]["impratio"] == pytest.approx(1.0)
        actual_materials = drive["physics_contact"]["actual_wheel_ground_shape_materials"]
        for role in ("ground", "wheel"):
            assert actual_materials[role]["mu"] == [pytest.approx(0.8)]
            assert actual_materials[role]["kf"] == [pytest.approx(1000.0)]
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
        host_eq = tuple(np.asarray(getattr(host, name)) for name in ("eq_type", "eq_obj1id", "eq_obj2id"))
        mjw_eq = tuple(np.asarray(getattr(mjw, name).numpy() if hasattr(getattr(mjw, name), "numpy")
                                  else getattr(mjw, name)) for name in ("eq_type", "eq_obj1id", "eq_obj2id"))
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
        # Reset state is equality-compatible; only finite state is required on
        # the first transient integration after reset.
        env.step(actions)
        assert torch.isfinite(env.rover.data.joint_pos.torch[:, env._wheel_ids]).all()
        assert torch.isfinite(env.rover.data.joint_vel.torch[:, env._wheel_ids]).all()
    finally:
        env.close()
