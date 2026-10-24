"""Bounded native correctness check; no policy, optimizer or training."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / os.environ.get('TENSOR_NATIVE_REPORT', 'native_smoke.json')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    steps = int(os.environ.get('TENSOR_NATIVE_STEPS', '16'))
    report = {'status': 'running', 'training_started': False,
              'policy_loaded': False, 'optimizer_started': False,
              'fps_benchmark_started': False, 'num_envs': 4, 'steps': steps}
    files = ['isaac_field/env.py', 'isaac_field/task.py',
             'isaac_field/tensor_task.py', 'isaac_field/tensor_sensors.py']
    sha = lambda: {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    report['source_sha256'] = sha()
    phases = {}
    def persist_report():
        # KitLauncher.close exits the process, including on normal shutdown.
        report['source_sha256_after'] = sha()
        report['sources_unchanged'] = report['source_sha256'] == report['source_sha256_after']
        report['host_wall_phases'] = {k: {'inclusive_seconds': ns / 1e9, 'calls': n} for k, (ns, n) in phases.items()}
        report['timing_scope'] = 'Correctness run, host wall only; nested/inclusive and startup/reset affected, not a sustained speed or FPS benchmark.'
        with OUTPUT.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps({'status': report['status'], 'report': str(OUTPUT), 'error': report.get('error')}, ensure_ascii=False), flush=True)
    def instrument(obj, name, label):
        original = getattr(obj, name)
        def measured(*args, **kwargs):
            before = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            except BaseException as exc:
                report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
                # The upstream launcher exits the process on exceptions.
                with OUTPUT.open('x', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                raise
            finally:
                total, count = phases.get(label, (0, 0))
                phases[label] = (total + time.perf_counter_ns() - before, count + 1)
        setattr(obj, name, measured)
    try:
        import numpy as np
        import torch
        from isaac_field.runtime import launch_app
        with launch_app(device='cuda:0', headless=True):
            from isaac_field.rsl_rl_cfg import FieldFastStopRslEnvCfg
            from isaac_field.rsl_rl_env import IsaacFieldRslEnv
            cfg = FieldFastStopRslEnvCfg()
            cfg.task.size = 12.0
            cfg.task.terrain_enabled = False
            cfg.scene.num_envs = 4
            cfg.scene.env_spacing = 22.0
            cfg.task_backend = 'torch_batched'
            cfg.sim.visualizer_cfgs = []
            cfg.sim.default_visualizer_cfg = None
            env = IsaacFieldRslEnv(cfg)
            try:
                report['physics'] = {'dt': cfg.sim.dt, 'decimation': cfg.decimation,
                    'position_iterations': cfg.sim.physics.max_position_iteration_count,
                    'task_backend': env.task_backend, 'device': str(env.device)}
                assert not env.sim.visualizers
                stage = env.sim.stage
                prim = stage.GetPrimAtPath(cfg.sim.physics_prim_path)
                report['physics']['gpu_dynamics'] = prim.GetAttribute('physxScene:enableGPUDynamics').Get()
                report['physics']['broadphase'] = prim.GetAttribute('physxScene:broadphaseType').Get()
                assert report['physics']['gpu_dynamics'] is True
                assert report['physics']['broadphase'] == 'GPU'
                if os.environ.get('TENSOR_PROFILE_INTERNALS') == '1':
                    import carb
                    import omni.physx
                    settings = carb.settings.get_settings()
                    report['runtime_settings'] = {key: settings.get(key) for key in (
                        '/app/runLoops/main/rateLimitEnabled', '/app/runLoops/main/rateLimitFrequency',
                        '/physics/fabricUpdateTransformations', '/physics/fabricUpdateVelocities',
                        '/physics/fabricUseGPUInterop', '/physics/suppressReadback')}
                    interface = omni.physx.get_physx_simulation_interface()
                    class Proxy:
                        def __getattr__(self, key):
                            return getattr(interface, key)
                    proxy = Proxy()
                    for method in ('simulate', 'fetch_results'):
                        setattr(proxy, method, getattr(interface, method))
                        instrument(proxy, method, 'physx_' + method)
                    omni.physx.get_physx_simulation_interface = lambda: proxy
                    instrument(torch.cuda, 'set_device', 'cuda_set_device')
                obs, _ = env.reset(seed=69041)
                if os.environ.get('TENSOR_FABRIC_ON_DEMAND') == '1':
                    import carb
                    settings = carb.settings.get_settings()
                    settings.set_bool('/physics/fabricUpdateTransformations', False)
                    settings.set_bool('/physics/fabricUpdateVelocities', False)
                    report['fabric_diagnostic_override'] = True
                expected = {'lidar': (4, 1, 400), 'depth': (4, 1, 160, 88),
                            'detections': (4, 16, 8), 'state': (4, 33), 'feedback': (4, 37)}
                max_errors = {key: 0.0 for key in expected}
                def check(observation):
                    assert set(observation) == set(expected)
                    for key, shape in expected.items():
                        value = observation[key]
                        assert tuple(value.shape) == shape
                        assert value.device.type == 'cuda' and value.dtype == torch.float32
                        assert bool(torch.isfinite(value).all())
                    for lane, task in enumerate(env.tasks):
                        reference = copy.copy(task)
                        reference._tensor_observation_callback = None
                        reference.frames = {k: task._materialize_sensor_frame(v) for k, v in task.frames.items()}
                        original = reference._observation()
                        for key in expected:
                            actual = observation[key][lane].detach().cpu().numpy()
                            error = float(np.max(np.abs(actual - original[key])))
                            max_errors[key] = max(error, max_errors[key])
                            np.testing.assert_allclose(actual, original[key], atol=2e-6, rtol=1e-6)
                check(obs)
                for obj, name, label in [(env.sim, 'step', 'physics_step'),
                    (env.scene, 'write_data_to_sim', 'scene_write'),
                    (env.scene, 'update', 'scene_update'),
                    (env.sim, 'render', 'render_dispatch'),
                    (env, '_get_dones', 'task_dones_inclusive'),
                    (env, '_get_observations', 'observations_inclusive')]:
                    instrument(obj, name, label)
                held = obs['depth']
                held_copy = held.clone()
                actions = torch.zeros((4, 4), device=env.device)
                actions[:, :2] = 0.08
                actions[:, 2] = 0.02
                if os.environ.get('TENSOR_FABRIC_ON_DEMAND') == '1':
                    actions[:, 3] = 0.1
                camera_positions_checked = 0
                last_depth_serials = [t._serial['depth'] for t in env.tasks]
                for step in range(steps):
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    assert bool(torch.isfinite(reward).all())
                    assert reward.device.type == 'cuda'
                    check(obs)
                    if os.environ.get('TENSOR_FABRIC_ON_DEMAND') == '1':
                        assert settings.get('/physics/fabricUpdateTransformations') is False
                    serials = [t._serial['depth'] for t in env.tasks]
                    captured_ids = [i for i, (a, b) in enumerate(zip(serials, last_depth_serials)) if a != b]
                    if os.environ.get('TENSOR_CAMERA_POSE_CHECK') == '1' and captured_ids:
                            camera = env.scene['depth'].data
                            torch.testing.assert_close(
                                camera.pos_w.torch[captured_ids],
                                env.rover.data.body_link_pose_w.torch[captured_ids, env._mount_id, :3],
                                atol=2e-5, rtol=1e-5)
                            camera_positions_checked += 1
                    last_depth_serials = serials
                    torch.testing.assert_close(held, held_copy, atol=0, rtol=0)
                    if step == 7:
                        env._reset_idx(torch.tensor([0], dtype=torch.int32, device=env.device))
                        obs = env._compute_observations()
                        check(obs)
                        last_depth_serials = [t._serial['depth'] for t in env.tasks]
                report.update(status='passed', max_reference_observation_error=max_errors,
                              camera_positions_checked=camera_positions_checked,
                              partial_reset_checked=True, old_observation_immutable=True,
                              output_shapes={k: list(v.shape) for k, v in obs.items()},
                              device_frames_retained=all(
                                  task.frames[k]['_tensor_sensor_frame'] is True
                                  and task.frames[k]['valid'].device.type == 'cuda'
                                  for task in env.tasks for k in ('lidar', 'depth')))
                persist_report()
            finally:
                env.close()
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    report['source_sha256_after'] = sha()
    report['sources_unchanged'] = report['source_sha256'] == report['source_sha256_after']
    report['host_wall_phases'] = {k: {'inclusive_seconds': ns / 1e9, 'calls': n} for k, (ns, n) in phases.items()}
    report['timing_scope'] = 'Correctness run, host wall only; nested/inclusive and startup/reset affected, not a sustained speed or FPS benchmark.'
    if not OUTPUT.exists():
                if not OUTPUT.exists():
                    persist_report()
    print(json.dumps({'status': report['status'], 'report': str(OUTPUT), 'error': report.get('error')}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'passed' and report['sources_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
