"""Bounded native PhysX probes, without training or an FPS benchmark."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _upstream_probe(steps):
    import torch
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg
    from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 7
    cfg.sim.physics = PhysxCfg()
    cfg.sim.visualizer_cfgs = []
    env = CartpoleEnv(cfg)
    try:
        env.reset()
        for _ in range(steps):
            obs, reward, _, _, _ = env.step(torch.zeros((4, 1), device=env.device))
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(reward).all()
        return {"probe": "official_cartpole", "steps": steps, "num_envs": 4,
                "device": str(env.device), "observation_shape": list(obs["policy"].shape),
                "manager": env.sim.physics_manager.__module__}
    finally:
        env.close()


def _field_probe(steps, plants):
    import torch
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.env import make_env, ray_ranges
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.physx_drive import coupling_relations
    from isaac_field.assets import ROBOT_SPEC
    from isaac_field.task import Config

    cfg = Config(size=12.0, plants=plants, terrain_enabled=False, max_time=8.0,
                 validate_scene=False, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    env = make_env(cfg, mode="goto", num_envs=2, seed=97_041, auto_reset=False,
                   front_guide_spec=FrontGuideSpec(), rsl=True)
    result = {"probe": "field", "task_config": asdict(cfg), "checks": {}}
    try:
        env.reset()
        expected = {"lidar": (2, 1, 400), "depth": (2, 1, 160, 88),
                    "detections": (2, 16, 8), "state": (2, 33), "feedback": (2, 37)}
        samples = {}
        coupling_residual = 0.0
        coupling_samples = []
        position_residual = 0.0
        previous_position = env.rover.data.joint_pos.torch.clone()
        interval_velocity_residual = 0.0
        max_effort = 0.0
        for _ in range(steps):
            actions = torch.zeros((2, 4), device=env.device)
            actions[0, :2] = 0.025
            actions[1, :2] = -0.025
            obs, reward, _, _, _ = env.step(actions)
            assert set(obs) == set(expected), list(obs)
            for key, shape in expected.items():
                tensor = obs[key]
                assert tuple(tensor.shape) == shape, (key, tensor.shape)
                assert torch.isfinite(tensor).all(), key
                samples[key] = {"shape": list(tensor.shape), "min": tensor.min().item(),
                                "max": tensor.max().item()}
            assert torch.isfinite(reward).all()
            velocity = env.rover.data.joint_vel.torch
            position = env.rover.data.joint_pos.torch
            step_residual = 0.0
            for row in coupling_relations(ROBOT_SPEC, env.front_guide_spec):
                leader = env.rover.joint_names.index(row['leader'])
                follower = env.rover.joint_names.index(row['follower'])
                residual = (velocity[:, follower] - row['ratio'] * velocity[:, leader]).abs().max().item()
                coupling_residual = max(coupling_residual, residual)
                step_residual = max(step_residual, residual)
                position_residual = max(position_residual,
                    (position[:, follower] - row['ratio'] * position[:, leader]).abs().max().item())
                relative_delta = ((position[:, follower] - previous_position[:, follower])
                    - row['ratio'] * (position[:, leader] - previous_position[:, leader]))
                interval_velocity_residual = max(interval_velocity_residual,
                    relative_delta.abs().max().item() / env.step_dt)
            coupling_samples.append(step_residual)
            previous_position = position.clone()
            effort = env.rover.actuators.applied_effort.torch[:, env._wheel_ids]
            assert torch.isfinite(effort).all()
            max_effort = max(max_effort, effort.abs().max().item())
        assert max_effort <= 2.00001, max_effort
        result['motion_coupling_max_velocity_residual_rad_s'] = coupling_residual
        result['motion_coupling_velocity_residual_by_step_rad_s'] = coupling_samples
        result['motion_coupling_max_position_residual_rad'] = position_residual
        result['motion_coupling_control_interval_velocity_residual_rad_s'] = interval_velocity_residual
        result['max_post_clamp_wheel_effort_nm'] = max_effort
        result['coupling_import_audit'] = env._same_side_mimic_model_audit
        result["observations"] = samples
        assert samples['depth']['max'] > 0.0, 'RTX depth never produced a nonzero metric sample'
        result["checks"]["five_live_observation_streams_finite"] = True
        mass = env.rover.data.body_mass.torch.sum(dim=1)
        torch.testing.assert_close(mass, torch.full_like(mass, 30.0), rtol=0, atol=1e-4)
        result["checks"]["mass_30kg_per_world"] = True
        result["drive_config"] = env.drive_config
        assert env.drive_config["actuator_effort_limit_nm"] == 2.0
        coupling = env.drive_config["same_side_wheel_coupling"]
        assert coupling["relations_per_world"] == 14
        assert env.drive_config["front_guide"]["drive_motor_count"] == 0
        result["checks"]["wheel_and_passive_guide_contract"] = True
        if plants:
            # A stem is visual-only in the contact model, but a horizontal
            # range beam must still see it. Move one physical kinematic root
            # through the public API; do not change the rangecaster output.
            from isaaclab.utils.math import matrix_from_quat
            mount = env.rover.data.body_link_pose_w.torch[0, env._mount_id]
            forward = matrix_from_quat(mount[None, 3:])[0, :, 0]
            pose = env.plants.data.body_link_pose_w.torch[:1, :1].clone()
            pose[0, 0, :2] = mount[:2] + 1.5 * forward[:2]
            pose[0, 0, 2] = env.scene.env_origins[0, 2]
            sensor = env.scene['rangefinder']
            other_range_before = ray_ranges(sensor.data)[1].clone()
            env.plants.write_body_pose_to_sim_index(body_poses=pose,
                env_ids=torch.tensor([0], device=env.device), body_ids=[0])
            env.scene.write_data_to_sim()
            env.sim.forward()
            sensor.update(0., force_recompute=True)
            ranges = ray_ranges(sensor.data)
            distance = float(ranges[0, 0])
            assert 0.2 < distance < 1.55, distance
            torch.testing.assert_close(ranges[1], other_range_before, rtol=0, atol=1e-5)
            result['visual_plant_range_m'] = distance
            result['checks']['visual_plant_remains_range_visible_after_pose_write'] = True
            # Force fresh RTX frames at the moved pose and exercise the same
            # instance-ID parser consumed by training, not synthetic detections.
            detections = []
            for _ in range(3):
                env.sim.render()
                env.scene['detector'].update(0., force_recompute=True)
                env._sensor_cache.clear()
                detections = env._capture(0, 'camera')['detections']
                if detections:
                    break
            result['targeted_leaf_detections'] = detections
            result['targeted_instance_metadata'] = env.scene['detector'].data.info['instance_segmentation']
            assert detections, 'RTX leaf instance IDs did not produce a training detection'
            result['checks']['visible_leaf_detection_from_rtx_instance_ids'] = True
        before = env.rover.data.root_link_pose_w.torch[1].clone()
        plant_before = env.plant_positions()[1].clone()
        env._reset_idx(torch.tensor([0], device=env.device, dtype=torch.long))
        after = env.rover.data.root_link_pose_w.torch[1].clone()
        torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
        torch.testing.assert_close(env.plant_positions()[1], plant_before, rtol=0, atol=1e-6)
        result["checks"]["partial_reset_world_isolation"] = True
        # Stress actual actuator commands, then read the post-limit effort.
        # Joint state is never overwritten to manufacture this clamp result.
        env.wheel_target_velocity.fill_(110.)
        env._drive_pi.feedforward.fill_(2.)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        saturated = env.rover.actuators.applied_effort.torch[:, env._wheel_ids].abs()
        result['stress_post_clamp_wheel_effort_nm'] = saturated.max().item()
        assert torch.isfinite(saturated).all() and saturated.max() <= 2.00001
        assert saturated.max() >= 1.99, 'stress command did not exercise the configured clamp'
        result['checks']['actual_2nm_actuator_clamp_under_stress'] = True
        result.update(backend_identity=env.backend_identity, steps=steps)
        # PhysX mimic is a joint-position constraint. Verify that equation and
        # its finite-difference velocity over the actual 50 ms control interval.
        # Keep the last-substep solver velocity separately: it has contact
        # impulses and is still the unchanged raw feedback source in training.
        result['mimic_acceptance'] = {
            'position_residual_limit_rad': 0.002,
            'control_interval_velocity_residual_limit_rad_s': 0.05,
            'control_interval_s': float(env.step_dt),
            'instantaneous_solver_velocity': 'reported separately; raw training feedback remains unchanged',
        }
        assert position_residual < 0.002, position_residual
        assert interval_velocity_residual < 0.05, interval_velocity_residual
        result['checks']['mimic_position_and_control_interval_kinematics'] = True
        return result
    except BaseException as error:
        error.field_probe_result = result
        raise
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--plants", type=int, default=0)
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument('--portable-root', type=Path, help='reuse a short Kit cache root for serial probes')
    args = parser.parse_args(argv)
    if args.steps < 1 or not 0 <= args.plants <= 8:
        parser.error("steps must be positive; plants must be between zero and eight")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"passed": False}
    from isaac_field.runtime import launch_app, simulator_identity

    kit_args = ''
    if args.portable_root is not None:
        portable = args.portable_root.resolve().as_posix()
        if any(character.isspace() for character in portable):
            parser.error('portable-root must not contain spaces')
        kit_args = f'--portable-root {portable}'
    with launch_app(kit_args=kit_args):
        try:
            import torch
            assert torch.cuda.is_available(), "CUDA PyTorch is required"
            actual = (torch.arange(8, device="cuda:0", dtype=torch.float32) ** 2).sum().item()
            assert actual == 140.0, actual
            result.update(_upstream_probe(args.steps) if args.upstream else _field_probe(args.steps, args.plants))
            result.update(simulator_identity(), passed=True, cuda_computation=True, torch=torch.__version__)
        except BaseException as error:
            result.update(getattr(error, 'field_probe_result', {}))
            result.update(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
