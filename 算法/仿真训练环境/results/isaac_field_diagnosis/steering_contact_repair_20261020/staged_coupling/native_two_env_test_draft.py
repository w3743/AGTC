"""Draft only: opt-in native integration test; not registered or executed here."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import warp as wp
except ImportError:  # Keep CPU-only pytest collection independent of Isaac Lab.
    wp = None

if wp is not None:
    @wp.kernel
    def _capture_two_world_wheel_qd(
        state_qd: wp.array(dtype=wp.float32),
        offsets: wp.array(dtype=wp.int32),
        tape: wp.array2d(dtype=wp.float32),
        counter: wp.array(dtype=wp.int32),
        capacity: int,
    ):
        slot = wp.atomic_add(counter, 0, 1)
        if slot < capacity:
            for world_index in range(2):
                for wheel_index in range(14):
                    tape[slot, world_index * 14 + wheel_index] = state_qd[
                        offsets[world_index * 14 + wheel_index]
                    ]


def test_native_two_env_mimic_world_isolation_and_partial_reset():
    """After production adoption, move this into tests/test_env_integration.py."""
    import torch
    assert wp is not None, "native Isaac/Newton run requires Warp"
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.env import make_env
    from isaac_field.task import Config
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonManager

    from multi_world_mimic import (
        MultiWorldSameSideMimicHook,
        audit_builder_and_solver,
    )

    names_by_side = {
        side: [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs if wheel.side == side]
        for side in ("left", "right")
    }
    hook = MultiWorldSameSideMimicHook(names_by_side, expected_world_count=2)
    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=False,
        validate_scene=False,
        max_time=8.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    # Match the v2 acceptance timing. Telemetry below is a Warp-only tape write
    # and remains compatible with CUDA graph capture.
    cfg.sim.dt = 0.002
    cfg.decimation = 25
    # The public hook must surround environment construction, before the Newton
    # cloner builds either world. It is removed immediately after construction.
    with newton_builder_world_hook(hook):
        env = make_env(cfg, mode="search", num_envs=2, seed=90232, auto_reset=False)
    try:
        model = env.sim.physics_manager.get_model()
        solver = env.sim.physics_manager._solver
        model_audit = audit_builder_and_solver(hook, model, solver)
        assert model_audit["hook_world_count"] == model_audit["newton_world_count"] == 2
        assert model_audit["mimic_constraints"] == model_audit["mujoco_equality_custom_rows"]
        assert model_audit["mimic_constraints"] == 24

        env.reset(
            seed=90232,
            options={"mode": "search", "start": [50.0, 50.0, 0.0], "goal": [90.0, 50.0]},
        )
        # Resolve the native flattened state offsets once; the registered
        # callback itself only launches a Warp kernel into a fixed-size tape.
        labels = [str(value) for value in model.joint_label]
        joint_world = np.asarray(model.joint_world.numpy() if hasattr(model.joint_world, "numpy")
                                 else model.joint_world).reshape(-1).astype(np.int64)
        joint_qd_start = np.asarray(model.joint_qd_start.numpy() if hasattr(model.joint_qd_start, "numpy")
                                    else model.joint_qd_start).reshape(-1).astype(np.int64)
        ordered_wheel_names = [f"{wheel.name}_wheel_joint" for wheel in ROBOT_SPEC.wheel_specs]
        qd_offsets = np.empty((2, len(ordered_wheel_names)), dtype=np.int64)
        for world_index in range(2):
            for wheel_index, wheel_name in enumerate(ordered_wheel_names):
                hits = [
                    index for index, label in enumerate(labels)
                    if joint_world[index] == world_index and label.rsplit("/", 1)[-1] == wheel_name
                ]
                assert len(hits) == 1, f"world {world_index} {wheel_name}: expected one joint, got {hits}"
                qd_offsets[world_index, wheel_index] = joint_qd_start[hits[0]]
        state_qd = NewtonManager.get_state_0().joint_qd
        assert int(qd_offsets.max()) < int(state_qd.shape[0])
        capacity = 60 * int(env.cfg.decimation)
        offsets_warp = wp.from_torch(
            torch.as_tensor(qd_offsets.reshape(-1), dtype=torch.int32, device=env.device).contiguous()
        )
        qd_tape = wp.empty((capacity, 28), dtype=wp.float32, device=env.device)
        qd_counter = wp.zeros(1, dtype=wp.int32, device=env.device)

        def capture_qd_before_solver():
            active_qd = NewtonManager.get_state_0().joint_qd
            wp.launch(
                _capture_two_world_wheel_qd,
                dim=1,
                device=env.device,
                inputs=[active_qd, offsets_warp, qd_tape, qd_counter, capacity],
            )

        NewtonManager.register_post_actuator_callback(capture_qd_before_solver)
        actions = torch.tensor(
            [[0.7, 0.2, 0.0, 0.0], [-0.3, 0.6, 0.0, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )
        left_ids = (env._wheel_sides < 0).detach().cpu().numpy()
        right_ids = (env._wheel_sides > 0).detach().cpu().numpy()
        side_masks = (left_ids, right_ids)
        settle = torch.zeros_like(actions)
        for _ in range(20):  # 1.0 s zero-action settle before the differential command.
            env.step(settle)
        q_rows, qd_rows = [], []
        for _ in range(40):  # 2.0 s differential command; use its final 1.0 s for mimic limits.
            env.step(actions)
            q = env.rover.data.joint_pos.torch[:, env._wheel_ids].detach().cpu().numpy().copy()
            qd = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy().copy()
            assert np.isfinite(q).all() and np.isfinite(qd).all()
            q_rows.append(q)
            qd_rows.append(qd)
            for env_index in range(2):
                for side_mask in side_masks:
                    wheel_indices = np.flatnonzero(side_mask)
                    assert len(wheel_indices) == 7
        expected_native_rows = capacity
        actual_native_rows = int(qd_counter.numpy().reshape(-1)[0])
        assert actual_native_rows == expected_native_rows, (
            f"native post-actuator samples={actual_native_rows}, expected={expected_native_rows}"
        )
        native_qd_rows = wp.to_torch(qd_tape)[:actual_native_rows].detach().cpu().numpy().reshape(
            actual_native_rows, 2, 14
        )
        assert np.isfinite(native_qd_rows).all()
        for env_index in range(2):
            for side_mask in side_masks:
                wheel_indices = np.flatnonzero(side_mask)
                q_error = np.asarray(q_rows[-20:])[:, env_index, wheel_indices] - np.asarray(
                    q_rows[-20:]
                )[:, env_index, wheel_indices[0], None]
                # 1 s settle + 2 s differential; final 1 s = final 20 policy
                # samples × decimation, matching the v2 native qd gate window.
                qd_last_second = native_qd_rows[-20 * int(env.cfg.decimation):]
                qd_error = qd_last_second[:, env_index, wheel_indices] - qd_last_second[
                    :, env_index, wheel_indices[0], None
                ]
                assert float(np.max(np.abs(q_error))) <= 0.02
                assert float(np.percentile(np.abs(qd_error), 95.0)) <= 0.0926
                assert float(np.max(np.abs(qd_error))) <= 0.463
        qd_last = qd_rows[-1]
        for side_mask in side_masks:
            wheel_indices = np.flatnonzero(side_mask)
            cross_world_speed_gap = abs(float(qd_last[0, wheel_indices].mean() - qd_last[1, wheel_indices].mean()))
            assert cross_world_speed_gap > 0.1, (
                "different env commands should produce independent side-mean wheel rates; "
                f"observed gap={cross_world_speed_gap:.6f} rad/s"
            )
        body_twist = env.rover.data.root_com_vel_w.torch.detach().cpu().numpy().copy()
        assert np.isfinite(body_twist).all()
        body_response_gap = float(np.linalg.norm(body_twist[0] - body_twist[1]))
        assert body_response_gap > 0.02, (
            "different env commands should produce distinguishable body response; "
            f"observed root-com twist gap={body_response_gap:.6f} m/s,rad/s"
        )

        q_before = q_rows[-1][1].copy()
        qd_before = qd_rows[-1][1].copy()
        controller = env._drive_pi
        pi_before = controller.integral[1].clone()
        ff_before = controller.feedforward[1].clone()
        error_before = controller.last_error[1].clone()
        p_estimate_before = controller.last_p_estimate[1].clone()
        blocked_before = controller.last_blocked[1].clone()
        track_command_before = env.track_command[1].clone()
        root_before = env.rover.data.root_link_pose_w.torch[1].detach().clone()
        root_velocity_before = env.rover.data.root_com_vel_w.torch[1].detach().clone()
        target_effort_before = env.rover.actuators.target_command.effort.torch[1, env._wheel_ids].clone()
        effort = env.rover.actuators.applied_effort.torch[:, env._wheel_ids]
        assert torch.isfinite(effort).all()
        assert float(effort.abs().max()) <= float(env.cfg.drive_effort_limit) + 1e-5
        effort_before = effort[1].clone()

        env._reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
        # Reset world 0 to the compatible zero q/qd state; world 1 is byte-for-byte untouched.
        torch.testing.assert_close(controller.integral[1], pi_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.feedforward[1], ff_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_error[1], error_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_p_estimate[1], p_estimate_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_blocked[1], blocked_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.track_command[1], track_command_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.data.root_link_pose_w.torch[1], root_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(env.rover.data.root_com_vel_w.torch[1], root_velocity_before, rtol=0.0, atol=0.0)
        q_after_reset = env.rover.data.joint_pos.torch[:, env._wheel_ids].detach().cpu().numpy()
        qd_after_reset = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy()
        np.testing.assert_allclose(q_after_reset[1], q_before, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(qd_after_reset[1], qd_before, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(q_after_reset[0], 0.0, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(qd_after_reset[0], 0.0, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            env.rover.actuators.target_command.effort.torch[1, env._wheel_ids],
            target_effort_before, rtol=0.0, atol=0.0,
        )
        torch.testing.assert_close(
            env.rover.actuators.applied_effort.torch[1, env._wheel_ids],
            effort_before, rtol=0.0, atol=0.0,
        )

        # The reset state is compatible with q_follower=q_leader and zero qd.
        # One post-reset integration only checks finite state; settled residual
        # thresholds above are intentionally not applied to this transient.
        env.step(actions)
        q = env.rover.data.joint_pos.torch[:, env._wheel_ids].detach().cpu().numpy()
        qd = env.rover.data.joint_vel.torch[:, env._wheel_ids].detach().cpu().numpy()
        assert np.isfinite(q).all() and np.isfinite(qd).all()
    finally:
        env.close()
