"""Opt-in live Newton checks for passive front-guide geometry and stability state.

This test allocates Isaac Lab/Newton resources and is deliberately gated behind
RUN_LEGACY_NEWTON_PHYSICS=1. It is not part of the normal CPU-only test run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LEGACY_NEWTON_PHYSICS") != "1",
    reason="historical Newton check; use an archived source snapshot explicitly",
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def test_native_speed_coupled_front_guides_keep_drive_isolated_and_stability_diagnostic():
    """Check live masses, joints, materials, reset clearance and per-world state."""
    import re

    import mujoco
    import torch

    from isaac_field.assets import FrontGuideSpec, ROBOT_SPEC
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.env import make_env
    from isaac_field.task import Config

    guides = FrontGuideSpec()
    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=True,
        terrain_amplitude=0.35,
        terrain_wavelength_min=4.0,
        terrain_wavelength_max=10.0,
        rock_density=0.0,
        validate_scene=False,
        max_time=8.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    env = make_env(
        cfg,
        mode="search",
        num_envs=2,
        seed=92041,
        auto_reset=False,
        front_guide_spec=guides,
        body_stability_config=BodyStabilityConfig(),
    )
    try:
        assert env.front_guide_spec == guides
        assert env.body_stability_config is not None
        assert env._wheel_ids.numel() == 14
        assert env.drive_config["proxy_wheel_count_per_side"] == 7
        assert env.drive_config["actuator_effort_limit_nm"] == pytest.approx(2.0)
        assert env.drive_config["same_side_wheel_coupling"]["relations_per_world"] == 14
        assert env.drive_config["same_side_wheel_coupling"]["wheel_relations_per_world"] == 12
        assert env.drive_config["same_side_wheel_coupling"]["guide_relations_per_world"] == 2
        assert env._same_side_mimic_model_audit["worlds_checked"] == 2
        assert env._same_side_mimic_model_audit["relations_per_world"] == 14
        assert env._same_side_mimic_model_audit["wheel_relations_per_world"] == 12
        assert env._same_side_mimic_model_audit["guide_relations_per_world"] == 2
        assert env._same_side_mimic_model_audit["cross_world_pairs"] == 0
        assert env.drive_config["front_guide"]["enabled"] is True
        assert env.drive_config["front_guide"]["model"] == (
            "same_side_track_speed_coupled_front_guides_separate_proxy_not_continuous_track"
        )
        assert env.drive_config["front_guide"]["rotation_coupling"] == "same_side_track_speed"
        assert env.drive_config["front_guide"]["drive_motor_count"] == 0
        assert env.drive_config["front_guide"]["added_drive_torque_nm"] == 0.0
        assert env.drive_config["front_guide"]["coupling"]["enabled"] is True
        assert env.drive_config["front_guide"]["coupling"]["relations_per_world"] == 2
        assert env.drive_config["front_guide"]["coupling"]["guide_actuator_count"] == 0
        assert env.drive_config["front_guide"]["coupling"]["added_pid_count"] == 0
        assert env.drive_config["front_guide"]["coupling"]["leader_joint_by_side"] == {
            "left": "left_wheel_0_wheel_joint",
            "right": "right_wheel_0_wheel_joint",
        }
        expected_speed_ratio = ROBOT_SPEC.wheel_radius_m / guides.radius_m
        assert env.drive_config["front_guide"]["coupling"]["ratio_by_side"] == {
            "left": pytest.approx(expected_speed_ratio),
            "right": pytest.approx(expected_speed_ratio),
        }

        guide_specs = guides.wheel_specs(ROBOT_SPEC)
        assert len(guide_specs) == 2
        assert all(guide_specs[i].side != guide_specs[j].side
                   for i in range(len(guide_specs)) for j in range(i + 1, len(guide_specs)))
        drive_cfg = env.cfg.scene.robot.actuators["drive"]
        assert all(
            not any(re.fullmatch(expression, guide.joint_name) for expression in drive_cfg.joint_names_expr)
            for guide in guide_specs
        ), "passive guide joints must not match the driven-wheel actuator expression"

        # Verify the imported articulation contract, not just the USD fields.
        body_mass = env.rover.data.body_mass.torch
        torch.testing.assert_close(
            body_mass.sum(dim=1),
            torch.full((2,), 30.0, dtype=body_mass.dtype, device=body_mass.device),
            rtol=0.0,
            atol=1e-4,
        )
        model = env.sim.physics_manager.get_model()
        joint_labels = [str(label) for label in model.joint_label]
        joint_world_raw = model.joint_world
        joint_world = _to_numpy(joint_world_raw).reshape(-1).astype(np.int64)
        guide_joint_ids: dict[int, list[int]] = {0: [], 1: []}
        for world in (0, 1):
            for guide in guide_specs:
                hits = [index for index, label in enumerate(joint_labels)
                        if joint_world[index] == world
                        and label.rsplit("/", 1)[-1] == guide.joint_name]
                assert len(hits) == 1, f"world {world}, {guide.joint_name}: found {hits}"
                guide_joint_ids[world].extend(hits)
        assert set(guide_joint_ids[0]).isdisjoint(set(guide_joint_ids[1]))

        solver = env.sim.physics_manager._solver
        host = solver.mj_model
        # Newton clone ids and MuJoCo template joint ids are different
        # namespaces. Use the solver's official per-world reverse maps rather
        # than guessing from USD labels (MuJoCo may normalize those names).
        newton_joint_to_host = _to_numpy(solver.mjc_jnt_to_newton_jnt).astype(np.int64)
        newton_dof_to_host = _to_numpy(solver.mjc_dof_to_newton_dof).astype(np.int64)
        host_dof_to_newton = _to_numpy(solver.mjc_jnt_to_newton_dof).astype(np.int64)
        assert newton_joint_to_host.shape == (2, int(host.njnt))
        assert newton_dof_to_host.shape == (2, int(host.nv))
        assert host_dof_to_newton.shape == (2, int(host.njnt))
        newton_qd_start = _to_numpy(model.joint_qd_start).reshape(-1).astype(np.int64)
        host_guide_joint_ids: dict[int, list[int]] = {0: [], 1: []}
        host_guide_dof_ids: dict[int, list[int]] = {0: [], 1: []}
        solver_mapping = []
        for world in (0, 1):
            for newton_joint_id in guide_joint_ids[world]:
                host_joint_hits = np.flatnonzero(newton_joint_to_host[world] == newton_joint_id)
                assert len(host_joint_hits) == 1, (
                    f"world {world} Newton guide joint {newton_joint_id} maps to host joints "
                    f"{host_joint_hits.tolist()}"
                )
                host_joint_id = int(host_joint_hits[0])
                host_dof_id = int(host.jnt_dofadr[host_joint_id])
                expected_newton_dof = int(newton_qd_start[newton_joint_id])
                assert int(host_dof_to_newton[world, host_joint_id]) == expected_newton_dof
                dof_hits = np.flatnonzero(newton_dof_to_host[world] == expected_newton_dof)
                assert dof_hits.tolist() == [host_dof_id], (
                    f"world {world} Newton guide DOF {expected_newton_dof} maps to host DOFs "
                    f"{dof_hits.tolist()}, expected [{host_dof_id}]"
                )
                host_guide_joint_ids[world].append(host_joint_id)
                host_guide_dof_ids[world].append(host_dof_id)
                solver_mapping.append({
                    "world": world,
                    "newton_joint_label": joint_labels[newton_joint_id],
                    "newton_joint_id": int(newton_joint_id),
                    "newton_dof_id": expected_newton_dof,
                    "host_joint_id": host_joint_id,
                    "host_joint_name": mujoco.mj_id2name(
                        host, mujoco.mjtObj.mjOBJ_JOINT, host_joint_id,
                    ),
                    "host_dof_id": host_dof_id,
                })
        print(f"guide_solver_mapping={solver_mapping}", flush=True)
        all_host_guide_joint_ids = set(host_guide_joint_ids[0] + host_guide_joint_ids[1])
        actuator_joint_ids = set(_to_numpy(host.actuator_trnid).reshape(-1, 2)[:, 0].astype(int))
        equality_types = _to_numpy(host.eq_type).reshape(-1).astype(np.int64)
        equality_obj1 = _to_numpy(host.eq_obj1id).reshape(-1).astype(np.int64)
        equality_obj2 = _to_numpy(host.eq_obj2id).reshape(-1).astype(np.int64)
        joint_equality_rows = equality_types == int(mujoco.mjtEq.mjEQ_JOINT)
        equality_joint_ids = set(equality_obj1[joint_equality_rows].tolist())
        equality_joint_ids.update(equality_obj2[joint_equality_rows].tolist())
        assert all_host_guide_joint_ids.isdisjoint(actuator_joint_ids), "guide wheel has an actuator"
        assert all_host_guide_joint_ids <= equality_joint_ids, "coupled guide wheel lacks a mimic/equality row"
        assert int(host.neq) == 14
        for side, guide in zip(("left", "right"), guide_specs, strict=True):
            host_joint_id = host_guide_joint_ids[0][guide_specs.index(guide)]
            hits = np.flatnonzero(
                joint_equality_rows
                & (equality_obj1 == host_joint_id)
            )
            assert len(hits) == 1, f"{side} guide should have one follower equality, got {hits.tolist()}"
            equality_id = int(hits[0])
            ratio = ROBOT_SPEC.wheel_radius_m / guide.radius_m
            leader_name = "left_wheel_0_wheel_joint" if side == "left" else "right_wheel_0_wheel_joint"
            leader_newton_hits = [index for index, label in enumerate(joint_labels)
                                  if joint_world[index] == 0
                                  and label.rsplit("/", 1)[-1] == leader_name]
            assert len(leader_newton_hits) == 1
            leader_host_hits = np.flatnonzero(
                newton_joint_to_host[0] == leader_newton_hits[0]
            )
            assert len(leader_host_hits) == 1
            leader_joint_id = int(leader_host_hits[0])
            assert equality_obj2[equality_id] == leader_joint_id
            np.testing.assert_allclose(
                _to_numpy(host.eq_data)[equality_id, :5], [0.0, ratio, 0.0, 0.0, 0.0],
                rtol=0.0, atol=1e-7,
            )

        # Newton/MJWarp's generalized actuator force must remain zero on the
        # guide DOFs even while the driven wheels are commanded.
        qfrc_actuator = _to_numpy(solver.mjw_data.qfrc_actuator)
        assert qfrc_actuator.shape == (2, int(host.nv))

        contact_materials = env.drive_config["physics_contact"]["actual_wheel_ground_shape_materials"]
        guide_materials = contact_materials["passive_front_guide"]
        assert guide_materials["mu"] == pytest.approx([0.8])
        assert guide_materials["kf"] == pytest.approx([1000.0])
        assert set(env.drive_config["physics_contact"]["front_guide_contact_body_names"]) == {
            guide.link_name for guide in guide_specs
        }

        obs, _ = env.reset(seed=92041, options={"mode": "search", "random_start": True})
        assert all(torch.isfinite(value).all() for value in obs["policy"].values())
        roots = _to_numpy(env.rover.data.root_link_pose_w.torch)
        assert np.isfinite(roots).all()
        origins = _to_numpy(env.scene.env_origins)
        body_poses = _to_numpy(env.rover.data.body_link_pose_w.torch)
        for world, task in enumerate(env.tasks):
            support_ids = [env.rover.body_names.index(f"{wheel.name}_wheel_link")
                           for wheel in ROBOT_SPEC.wheel_specs]
            guide_ids = [env.rover.body_names.index(guide.link_name) for guide in guide_specs]
            support_pose = body_poses[world, support_ids]
            support_xy = support_pose[:, :2] - origins[world, None, :2]
            support_z = support_pose[:, 2] - origins[world, 2] - ROBOT_SPEC.wheel_radius_m
            support_clearance = support_z - task.scene.terrain.surface_height(support_xy)
            assert float(np.min(support_clearance)) >= -0.01
            assert float(np.max(support_clearance)) <= 0.03
            guide_pose = body_poses[world, guide_ids]
            guide_xy = guide_pose[:, :2] - origins[world, None, :2]
            guide_z = guide_pose[:, 2] - origins[world, 2]
            guide_bottom = np.asarray([wheel.radius_m for wheel in guide_specs])
            guide_clearance = guide_z - guide_bottom - task.scene.terrain.surface_height(guide_xy)
            assert float(np.min(guide_clearance)) >= -0.01
            # Raised guide wheels may be airborne on level/low-curvature terrain.
            assert float(np.max(guide_clearance)) <= 0.09

        zero_actions = torch.zeros((2, 4), dtype=torch.float32, device=env.device)
        for _ in range(20):
            env.step(zero_actions)
        # World 0 drives both tracks forward; world 1 commands an opposing
        # left/right pair so straight and pivot motion both exercise the guides.
        actions = torch.tensor([[0.5, 0.5, 0.0, 0.0], [-0.5, 0.5, 0.0, 0.0]],
                               dtype=torch.float32, device=env.device)
        guide_q_rows = []
        guide_qd_rows = []
        support_q_rows = []
        support_qd_rows = []
        track_command_rows = []
        for _ in range(40):
            _, rewards, terminated, truncated, _ = env.step(actions)
            q_now = _to_numpy(env.rover.data.joint_pos.torch)
            qd_now = _to_numpy(env.rover.data.joint_vel.torch)
            guide_q_rows.append(np.stack([
                q_now[:, env.rover.joint_names.index(guide.joint_name)]
                for guide in guide_specs
            ], axis=1))
            guide_qd_rows.append(np.stack([
                qd_now[:, env.rover.joint_names.index(guide.joint_name)]
                for guide in guide_specs
            ], axis=1))
            support_q_rows.append(np.stack([
                q_now[:, env.rover.joint_names.index(f"{side}_wheel_0_wheel_joint")]
                for side in ("left", "right")
            ], axis=1))
            support_qd_rows.append(np.stack([
                qd_now[:, env.rover.joint_names.index(f"{side}_wheel_0_wheel_joint")]
                for side in ("left", "right")
            ], axis=1))
            track_command_rows.append(_to_numpy(env.track_command).copy())
        guide_q_rows = np.asarray(guide_q_rows)
        guide_qd_rows = np.asarray(guide_qd_rows)
        support_q_rows = np.asarray(support_q_rows)
        support_qd_rows = np.asarray(support_qd_rows)
        track_command_rows = np.asarray(track_command_rows)
        assert torch.isfinite(rewards).all()
        assert not bool(terminated.any()) and not bool(truncated.any())
        assert len(env.diagnostic_step_infos) == 2
        assert env.observation_config["feedback_dim"] == 37
        for world, info in enumerate(env.diagnostic_step_infos):
            assert info["body_stability"]["enabled"] is True
            assert len(info["body_stability"]["body_angular_velocity_rad_s"]) == 3
            assert info["body_stability_summary"]["sample_count"] == 60
            assert info["termination_detail"] == {
                "boundary": False,
                "overtilt_roll": False,
                "overtilt_pitch": False,
            }
            assert np.isfinite(env.tasks[world].body_angular_velocity).all()
            assert info is not env._last_results[world][4]
            assert info["reward_components"] is not env._last_results[world][4]["reward_components"]
            component_total = sum(info["reward_components"].values())
            assert component_total == pytest.approx(float(rewards[world]), abs=1e-5, rel=1e-6)

        # Keep the established support-mimic residual thresholds for the new
        # guide-to-wheel-0 equalities across all sampled policy steps.
        guide_residuals = {}
        for guide_index, guide in enumerate(guide_specs):
            side_index = 0 if guide.side == "left" else 1
            ratio = ROBOT_SPEC.wheel_radius_m / guide.radius_m
            q_error = guide_q_rows[:, :, guide_index] - ratio * support_q_rows[:, :, side_index]
            qd_error = guide_qd_rows[:, :, guide_index] - ratio * support_qd_rows[:, :, side_index]
            assert float(np.max(np.abs(q_error))) <= 0.02
            assert float(np.percentile(np.abs(qd_error), 95.0)) <= 0.0926
            assert float(np.max(np.abs(qd_error))) <= 0.463
            guide_residuals[guide.side] = {
                "q_abs_max_rad": float(np.max(np.abs(q_error))),
                "qd_abs_p95_rad_s": float(np.percentile(np.abs(qd_error), 95.0)),
                "qd_abs_max_rad_s": float(np.max(np.abs(qd_error))),
                "ratio": float(ratio),
            }
        np.testing.assert_allclose(
            track_command_rows[-1, 0, 0], track_command_rows[-1, 0, 1],
            rtol=0.0, atol=1e-6,
        )
        assert track_command_rows[-1, 1, 0] * track_command_rows[-1, 1, 1] < 0.0
        assert float(np.mean(np.abs(support_qd_rows[:, 1, 0]))) > 0.1
        assert float(np.mean(np.abs(support_qd_rows[:, 1, 1]))) > 0.1
        print(f"guide_policy_step_residuals={guide_residuals}", flush=True)

        # Validate the diagnostic's rates against the current Newton root
        # angular velocity rotated from world axes into the root body frame.
        from isaaclab.utils.math import matrix_from_quat

        root_pose = env.rover.data.root_link_pose_w.torch
        root_velocity = env.rover.data.root_com_vel_w.torch
        root_rotation = matrix_from_quat(root_pose[:, 3:])
        expected_body_angular_velocity = torch.bmm(
            root_rotation.transpose(1, 2), root_velocity[:, 3:6].unsqueeze(-1),
        ).squeeze(-1)
        diagnostic_body_angular_velocity = torch.as_tensor(
            [info["body_stability"]["body_angular_velocity_rad_s"]
             for info in env.diagnostic_step_infos],
            dtype=expected_body_angular_velocity.dtype,
            device=expected_body_angular_velocity.device,
        )
        torch.testing.assert_close(
            diagnostic_body_angular_velocity,
            expected_body_angular_velocity,
            rtol=1e-6,
            atol=1e-6,
        )

        qfrc_actuator = _to_numpy(solver.mjw_data.qfrc_actuator)
        for world in (0, 1):
            np.testing.assert_allclose(
                qfrc_actuator[world, host_guide_dof_ids[world]],
                0.0, atol=1e-9, rtol=0.0,
            )
        applied_effort = env.rover.actuators.applied_effort.torch[:, env._wheel_ids]
        assert torch.isfinite(applied_effort).all()
        assert float(applied_effort.abs().max()) <= 2.0 + 1e-5

        # Partial reset clears stability accumulation only in the reset world.
        env._reset_idx(torch.tensor([0], dtype=torch.long, device=env.device))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.obs_buf = env._compute_observations()
        assert env.tasks[0].body_stability_stats.summary()["sample_count"] == 0
        assert env.tasks[1].body_stability_stats.summary()["sample_count"] == 60
        assert np.array_equal(env.tasks[0].body_angular_velocity, np.zeros(3))
    finally:
        env.close()


def test_native_free_front_guides_remain_unconstrained_without_drive_actuators():
    """The diagnostic free-guide setting keeps the historical 12 wheel rows."""
    import mujoco

    from isaac_field.assets import FrontGuideSpec, ROBOT_SPEC
    from isaac_field.env import make_env
    from isaac_field.task import Config

    guides = FrontGuideSpec(rotation_coupling="free")
    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=False,
        validate_scene=False,
        max_time=2.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    env = make_env(
        cfg,
        mode="search",
        num_envs=1,
        seed=92043,
        auto_reset=False,
        front_guide_spec=guides,
    )
    try:
        coupling = env.drive_config["same_side_wheel_coupling"]
        assert coupling["relations_per_world"] == 12
        assert coupling["wheel_relations_per_world"] == 12
        assert coupling["guide_relations_per_world"] == 0
        assert env._same_side_mimic_model_audit["relations_per_world"] == 12
        assert env._same_side_mimic_model_audit["guide_relations_per_world"] == 0
        guide_meta = env.drive_config["front_guide"]
        assert guide_meta["rotation_coupling"] == "free"
        assert guide_meta["model"] == (
            "passive_free_rotating_guide_wheels_separate_proxy_not_continuous_track"
        )
        assert guide_meta["coupling"]["enabled"] is False
        assert guide_meta["coupling"]["relations_per_world"] == 0
        assert guide_meta["drive_motor_count"] == 0
        assert guide_meta["added_drive_torque_nm"] == 0.0

        solver = env.sim.physics_manager._solver
        host = solver.mj_model
        model = env.sim.physics_manager.get_model()
        labels = [str(label) for label in model.joint_label]
        worlds = _to_numpy(model.joint_world).reshape(-1).astype(np.int64)
        mapping = _to_numpy(solver.mjc_jnt_to_newton_jnt).astype(np.int64)
        host_joint_ids = set()
        for guide in guides.wheel_specs(ROBOT_SPEC):
            newton_hits = [joint_id for joint_id, label in enumerate(labels)
                           if worlds[joint_id] == 0
                           and label.rsplit("/", 1)[-1] == guide.joint_name]
            assert len(newton_hits) == 1
            host_hits = np.flatnonzero(mapping[0] == newton_hits[0])
            assert len(host_hits) == 1
            host_joint_ids.add(int(host_hits[0]))
        actuator_joint_ids = set(_to_numpy(host.actuator_trnid).reshape(-1, 2)[:, 0].astype(int))
        assert host_joint_ids.isdisjoint(actuator_joint_ids)
        equality_types = _to_numpy(host.eq_type).reshape(-1).astype(np.int64)
        equality_rows = equality_types == int(mujoco.mjtEq.mjEQ_JOINT)
        equality_joint_ids = set(_to_numpy(host.eq_obj1id).reshape(-1)[equality_rows].astype(int))
        equality_joint_ids.update(_to_numpy(host.eq_obj2id).reshape(-1)[equality_rows].astype(int))
        assert host_joint_ids.isdisjoint(equality_joint_ids)
        assert int(host.neq) == 12
        assert len(env._same_side_mimic_hook.relations) == 12
    finally:
        env.close()


def test_native_single_field_adapter_reset_reports_real_rough_terrain_seed():
    """Verify the evaluator exposes the seed carried by the reset Scene."""
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.task import Config

    cfg = Config(
        size=100.0,
        plants=0,
        terrain_enabled=True,
        terrain_amplitude=0.35,
        terrain_wavelength_min=4.0,
        terrain_wavelength_max=10.0,
        rock_density=0.0,
        validate_scene=False,
        max_time=2.0,
        lidar_rays=400,
        depth_height=160,
        depth_width=88,
    )
    env = make_env(
        cfg,
        mode="search",
        num_envs=1,
        seed=92042,
        auto_reset=False,
        front_guide_spec=FrontGuideSpec(),
        body_stability_config=BodyStabilityConfig(),
    )
    adapter = SingleFieldAdapter(env)
    try:
        _, info = adapter.reset(
            seed=92042,
            options={"mode": "search", "random_start": True},
        )
        scene_seed = adapter.task.scene.terrain_seed
        assert scene_seed is not None
        assert info["terrain_seed"] == scene_seed
        assert scene_seed == env.terrain.seed
        assert info["mode"] == "search"
    finally:
        adapter.close()
