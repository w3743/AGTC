from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from .env import Config, FieldEnv


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stats(samples):
    x = np.asarray(samples, dtype=float)
    return {'count': len(x), 'mean_ms': float(x.mean()), 'p50_ms': float(np.median(x)),
            'p95_ms': float(np.percentile(x, 95)), 'p99_ms': float(np.percentile(x, 99)),
            'throughput_hz': float(1000 / x.mean())}


def benchmark(cfg, args):
    from .policy import PolicyRunner
    import torch
    env = FieldEnv(cfg, args.mode)
    t0 = time.perf_counter()
    obs, info = env.reset(seed=args.seed)
    setup_ms = (time.perf_counter() - t0) * 1000
    rng = np.random.default_rng(args.seed + 100)
    data = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(cfg),
            'hardware': {'os': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(),
                         'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu', 'torch_threads': 1},
            'initial_reset_and_validation_ms': setup_ms, 'validation': env.validation,
            'scope': 'Windows/host CPU; headless, no browser/HTTP serialization, no RK3588 or real perception inference',
            'competence_claim': False}
    action = np.zeros(4, np.float32)
    observations = []
    samples, resets = [], 0
    for i in range(args.steps + 32):
        if i % 10 == 0:
            action = rng.uniform(-1, 1, 4).astype(np.float32)
        t0 = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= 32:
            samples.append(elapsed)
            if len(observations) < 256:
                observations.append({k: v.copy() for k, v in obs.items()})
        if terminated or truncated:
            resets += 1
            obs, _ = env.reset(seed=args.seed + resets)
    data['simulation_step'] = stats(samples)
    data['simulation_step']['real_time_factor'] = data['simulation_step']['throughput_hz'] * cfg.dt
    data['simulation_step']['resets_excluded_from_timing'] = resets

    runner = PolicyRunner(env, args.checkpoint)
    data['policy_source'] = runner.label
    data['model_parameters'] = sum(p.numel() for p in runner.model.policy.parameters())
    for caching in (True, False):
        runner.model.policy.features_extractor.cache_inference = caching
        runner.model.policy.features_extractor._cache.clear()
        runner.reset()
        times = []
        for i in range(args.policy_steps + 32):
            ob = observations[i % len(observations)]
            t0 = time.perf_counter()
            runner.action(ob)
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= 32:
                times.append(elapsed)
        data['policy_cached' if caching else 'policy_uncached'] = stats(times)
    runner.model.policy.features_extractor.cache_inference = True
    runner.reset()
    obs, _ = env.reset(seed=args.seed + 1000)
    combined, sim_ms, infer_ms = [], [], []
    for i in range(min(args.steps, 600) + 32):
        t0 = time.perf_counter()
        action = runner.action(obs)
        t1 = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        t2 = time.perf_counter()
        if i >= 32:
            combined.append((t2 - t0) * 1000)
            infer_ms.append((t1 - t0) * 1000)
            sim_ms.append((t2 - t1) * 1000)
        if terminated or truncated:
            obs, _ = env.reset()
            runner.reset()
    data['closed_loop_unpaced'] = stats(combined)
    data['closed_loop_unpaced']['inference'] = stats(infer_ms)
    data['closed_loop_unpaced']['environment'] = stats(sim_ms)
    if args.paced_seconds > 0:
        runner.reset()
        obs, _ = env.reset(seed=args.seed + 2000)
        begin = time.perf_counter()
        deadline = begin
        durations, timestamps = [], []
        count = max(2, int(args.paced_seconds / cfg.dt))
        missed = 0
        for i in range(count):
            time.sleep(max(0, deadline - time.perf_counter()))
            actual_start = time.perf_counter()
            action = runner.action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            finish = time.perf_counter()
            timestamps.append(actual_start)
            durations.append((finish - actual_start) * 1000)
            deadline += cfg.dt
            if finish > deadline:
                missed += 1
            if terminated or truncated:
                obs, _ = env.reset()
                runner.reset()
        data['paced_loop'] = {**stats(durations), 'target_hz': 1 / cfg.dt,
                              'actual_start_hz': (count - 1) / (timestamps[-1] - timestamps[0]),
                              'deadline_misses': missed, 'steps': count,
                              'start_interval_p95_ms': float(np.percentile(np.diff(timestamps) * 1000, 95))}
    save(args.output, data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print(f'Benchmark saved: {Path(args.output).resolve()}')


class MixedEnv(FieldEnv):
    """Short-range curriculum only for smoke training, not a full-field benchmark."""
    def __init__(self, cfg, seed=7):
        super().__init__(cfg)
        self.task_rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        mode = 'goto' if self.task_rng.random() < .6 else 'search'
        obs, info = super().reset(seed=seed, options={'mode': mode, **(options or {})})
        if self.mode == 'goto':
            for _ in range(100):
                goal = self.robot[:2] + self.task_rng.uniform(1, 5, 2)
                if self.scene.navigable(goal):
                    self.set_goal(goal)
                    break
            obs, info = self._observation(), self._info()
        return obs, info


def train(cfg, args):
    from .policy import make_model
    from stable_baselines3.common.monitor import Monitor
    import torch
    from sb3_contrib import RecurrentPPO
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = MixedEnv(cfg, args.seed) if args.mode == 'mixed' else FieldEnv(cfg, args.mode)
    monitored = Monitor(env)
    model = make_model(monitored, seed=args.seed, n_steps=128, verbose=1)
    before = {k: p.detach().clone() for k, p in model.policy.named_parameters()}
    begin = time.perf_counter()
    model.learn(total_timesteps=args.steps)
    elapsed = time.perf_counter() - begin
    finite = all(torch.isfinite(p).all().item() for p in model.policy.parameters())
    change = sum(float((p.detach() - before[k]).square().sum()) for k, p in model.policy.named_parameters())
    path = out / 'checkpoint.zip'
    model.save(path)
    reloaded = RecurrentPPO.load(path, device='cpu')
    # Execute full held-out short episodes, maintaining hidden states throughout.
    evaluation = []
    for mode in ('search', 'goto'):
        for seed in (10007, 10008):
            test = FieldEnv(cfg, mode)
            obs, info = test.reset(seed=seed)
            state = None
            episode_start = True
            total_reward = 0
            steps = 0
            while not test.done:
                action, state = reloaded.predict(obs, state=state, episode_start=np.array([episode_start]), deterministic=True)
                episode_start = False
                obs, reward, terminated, truncated, info = test.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            evaluation.append({'mode': mode, 'seed': seed, 'steps': steps, 'reward': total_reward, **info})
    summary = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'requested_steps': args.steps,
               'actual_steps': model.num_timesteps, 'training_wall_seconds': elapsed,
               'training_steps_per_second': model.num_timesteps / elapsed, 'config': asdict(cfg),
               'mode': args.mode, 'algorithm': 'RecurrentPPO', 'policy': 'SensorEncoder + LSTM128',
               'device': 'cpu', 'parameters': sum(p.numel() for p in model.policy.parameters()),
               'weights_finite': finite, 'weight_delta_squared': change, 'checkpoint_reload_passed': True,
               'evaluation': evaluation, 'training_success_claim': False,
               'note': 'Smoke training verifies gradients/checkpoint/execution, not field task mastery. Reports automatic; no learned report/finish head or adversarial teacher yet.',
               'checkpoint': str(path.resolve())}
    save(out / 'train_summary.json', summary)
    env.close()
    if not finite or change <= 0:
        raise RuntimeError('training did not produce finite updated parameters')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Lightweight field MVP: no Gazebo, no image rendering.')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('serve', 'benchmark', 'train'):
        p = sub.add_parser(command)
        p.add_argument('--size', type=float, default=16 if command == 'train' else 100)
        p.add_argument('--plants', type=int, default=12 if command == 'train' else 400)
        p.add_argument('--mode', choices=('search', 'goto', 'mixed') if command == 'train' else ('search', 'goto'), default='mixed' if command == 'train' else 'search')
        p.add_argument('--seed', type=int, default=7)
        p.add_argument('--hz', type=float, default=20)
        p.add_argument('--max-time', type=float, default=20 if command == 'train' else 900)
        p.add_argument('--max-speed', type=float, default=15 / 3.6, help='m/s; default 15 km/h')
        p.add_argument('--plant-height', type=float, default=1.0, help='maize height in metres')
        p.add_argument('--height-spread', type=float, default=0.05, help='height variation +/- metres')
        p.add_argument('--lidar-rays', type=int, default=400, help='samples per complete laser scan')
        p.add_argument('--lidar-hz', type=float, default=10.0, help='laser scan refresh rate')
        p.add_argument('--lidar-range', type=float, default=8.0, help='maximum laser range in metres')
        p.add_argument('--lidar-min-range', type=float, default=0.12, help='minimum laser range in metres')
        p.add_argument('--depth-width', type=int, default=88, help='shared policy and preview depth width')
        p.add_argument('--depth-height', type=int, default=160, help='shared policy and preview depth height')
        p.add_argument('--flat-ground', action='store_true', help='disable terrain perturbations and rocks')
        p.add_argument('--terrain-amplitude', type=float, default=0.35, help='ground height amplitude bound in metres')
        p.add_argument('--terrain-wavelength-min', type=float, default=4.0,
                       help='minimum terrain wavelength in metres (at least 1 m)')
        p.add_argument('--terrain-wavelength-max', type=float, default=10.0,
                       help='maximum terrain wavelength in metres')
        p.add_argument('--rock-density', type=float, default=0.01, help='rocks per square metre')
        p.add_argument('--no-validate', action='store_true', help='development/benchmark option, recorded in output')
        if command == 'serve':
            p.add_argument('--port', type=int, default=8765)
            p.add_argument('--checkpoint', type=Path)
        elif command == 'benchmark':
            p.add_argument('--steps', type=int, default=2000)
            p.add_argument('--policy-steps', type=int, default=1000)
            p.add_argument('--paced-seconds', type=float, default=5)
            p.add_argument('--checkpoint', type=Path)
            p.add_argument('--output', type=Path, default=Path('results/field_mvp/benchmark.json'))
        else:
            p.add_argument('--steps', type=int, default=2048)
            p.add_argument('--output', type=Path, default=Path('results/field_mvp/training'))
    args = parser.parse_args()
    if not np.isfinite(args.hz) or not 1 <= args.hz <= 200:
        parser.error('--hz must be 1..200')
    if not np.isfinite(args.lidar_hz) or args.lidar_hz <= 0:
        parser.error('--lidar-hz must be finite and positive')
    if args.command == 'benchmark' and (args.steps < 1 or args.policy_steps < 1 or args.paced_seconds < 0):
        parser.error('benchmark steps must be positive and paced duration nonnegative')
    cfg = Config(size=args.size, plants=args.plants, dt=1 / args.hz, max_time=args.max_time,
                 max_speed=args.max_speed, plant_height=args.plant_height,
                 plant_height_spread=args.height_spread, lidar_rays=args.lidar_rays,
                 lidar_period=1 / args.lidar_hz, lidar_range=args.lidar_range,
                 lidar_min_range=args.lidar_min_range, depth_width=args.depth_width,
                 depth_height=args.depth_height, terrain_enabled=not args.flat_ground,
                 terrain_amplitude=args.terrain_amplitude,
                 terrain_wavelength_min=args.terrain_wavelength_min,
                 terrain_wavelength_max=args.terrain_wavelength_max,
                 rock_density=args.rock_density,
                 validate_scene=not args.no_validate)
    # Small matrix workloads should not fan out into dozens of BLAS workers.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.command == 'serve':
            from .server import serve
            serve(cfg, mode=args.mode, seed=args.seed, port=args.port, checkpoint=args.checkpoint)
        elif args.command == 'benchmark':
            benchmark(cfg, args)
        else:
            train(cfg, args)


if __name__ == '__main__':
    main()
