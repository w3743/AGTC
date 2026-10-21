"""Train and benchmark the recurrent field policy on Gazebo observations.

The Gazebo adapter is intentionally imported lazily: importing this module or
asking for ``--help`` never starts Gazebo.  The policy sees only the same four
observation entries as ``FieldEnv``; task generation may inspect the scene at
reset time, but no map or plant labels are added to policy observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
import traceback
from typing import Any
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np

from field_sim.env import Config
from field_sim.policy import PolicyRunner, make_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "gazebo_field_training"


def _json_value(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_monitor_port_available(port: int) -> None:
    """Fail early if another process already owns the monitor endpoint."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Windows otherwise permits two SO_REUSEADDR listeners on one port.
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            probe.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            # Match Unix server sockets: allow a fast restart after the last
            # monitor connection's TIME_WAIT, while an active listener still
            # prevents bind because SO_REUSEPORT is never enabled.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", int(port)))
    except OSError as exc:
        raise OSError(f"training monitor port {port} is already in use or unavailable") from exc
    finally:
        probe.close()


def _physics_runtime_settings(driver) -> dict:
    """Read physics values from the runtime SDF actually passed to Gazebo."""
    result = {"physics_step_size_s": None, "real_time_factor": None,
              "real_time_update_rate_hz": None, "physics_settings_source": "unavailable"}
    if driver is None:
        return result
    result["physics_step_size_s"] = getattr(driver, "step_size", None)
    runtime_sdf = getattr(driver, "_runtime_sdf", None)
    if runtime_sdf is None or not Path(runtime_sdf).is_file():
        return result
    physics = ET.parse(runtime_sdf).getroot().find(".//world/physics")
    if physics is None:
        return result
    for element, key in (("max_step_size", "physics_step_size_s"),
                         ("real_time_factor", "real_time_factor"),
                         ("real_time_update_rate", "real_time_update_rate_hz")):
        text = physics.findtext(element)
        if text is not None:
            result[key] = float(text)
    result["physics_settings_source"] = "runtime_sdf"
    return result


def _threadpool_limit_context():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        # The WSL training overlay does not require threadpoolctl; OMP/BLAS
        # environment limits are set by its launcher and Torch can be capped
        # directly for its own intra-op workers.
        try:
            import torch
        except ImportError:
            return nullcontext()
        torch.set_num_threads(1)
        return nullcontext()
    return threadpool_limits(limits=1)


def _percentile_stats(samples, *, unit="ms") -> dict:
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "unit": unit, "mean": None, "p50": None, "p95": None,
                "p99": None, "min": None, "max": None}
    return {"count": int(values.size), "unit": unit,
            "mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def _observation_is_finite(observation) -> bool:
    return isinstance(observation, dict) and all(
        np.all(np.isfinite(np.asarray(value))) for value in observation.values())


def _make_env(cfg: Config, mode: str, *, backend: str, output_dir: Path,
              gui: bool = False):
    if backend == "fieldsim":
        from field_sim.env import FieldEnv
        return FieldEnv(cfg, mode)
    from simulation.gazebo_env import GazeboFieldEnv
    return GazeboFieldEnv(cfg, mode, output_dir=output_dir, gui=gui)


class MixedTaskWrapper(gym.Wrapper):
    """Gym-compatible reset wrapper with a small 60/40 goto/search curriculum."""

    def __init__(self, env, seed: int, *, task_mode: str = "mixed", random_start: bool = False):
        super().__init__(env)
        if task_mode not in ("mixed", "search", "goto"):
            raise ValueError("task_mode must be mixed, search, or goto")
        self.task_mode = task_mode
        self.random_start = bool(random_start)
        self.task_rng = np.random.default_rng(seed)
        self.mode = "search"
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        if self.task_mode == "mixed":
            if self.reset_count == 0:
                self.mode = "goto" if self.task_rng.random() < 0.6 else "search"
            elif self.reset_count == 1:
                self.mode = "search" if self.mode == "goto" else "goto"
            else:
                self.mode = "goto" if self.task_rng.random() < 0.6 else "search"
        else:
            self.mode = self.task_mode
        self.reset_count += 1
        supplied_options = options or {}
        reset_options = {"mode": self.mode, **supplied_options}
        if (self.random_start and "start" not in supplied_options
                and "random_start" not in supplied_options):
            reset_options["random_start"] = True
        observation, info = self.env.reset(seed=seed, options=reset_options)
        base = self.unwrapped
        self.mode = getattr(base, "mode", self.mode)
        if self.mode == "goto":
            scene = getattr(base, "scene", None)
            robot = getattr(base, "robot", None)
            if scene is not None and robot is not None and hasattr(base, "set_goal"):
                position = np.asarray(robot[:2], dtype=np.float64)
                for _ in range(100):
                    radius = float(self.task_rng.uniform(1.0, 5.0))
                    angle = float(self.task_rng.uniform(-math.pi, math.pi))
                    goal = position + radius * np.array([math.cos(angle), math.sin(angle)])
                    if scene.navigable(goal):
                        base.set_goal(goal)
                        # Match FieldEnv's reset contract after assigning a task.
                        observation, info = base._observation(), base._info()
                        break
        return observation, info

    def step(self, action):
        return self.env.step(action)


class MetricsWrapper(gym.Wrapper):
    """Collect reset/step timing without changing the observation or reward."""

    def __init__(self, env, *, monitor=None, phase="training", total_steps=0):
        super().__init__(env)
        self.monitor = monitor
        self.phase = phase
        self.total_steps = int(total_steps)
        self.step_count = 0
        self.monitor_errors: list[str] = []
        self.reset_seconds: list[float] = []
        self.step_ms: list[float] = []
        self.sim_step_seconds: list[float] = []
        self.episode_modes: list[str] = []
        self.episode_ends: list[dict] = []
        self.sensor_fresh_counts = Counter()
        self.last_info = {}
        self.last_sensor_snapshot = {}
        self._previous_sensor_sequences = {}
        self._mode = getattr(env, "mode", "search")

    def reset(self, *, seed=None, options=None):
        begin = time.perf_counter()
        result = self.env.reset(seed=seed, options=options)
        self.reset_seconds.append(time.perf_counter() - begin)
        self._mode = getattr(self.unwrapped, "mode", getattr(self.env, "mode", "search"))
        self.episode_modes.append(str(self._mode))
        if isinstance(result, tuple) and len(result) == 2:
            self.last_info = result[1]
            self._previous_sensor_sequences = _sensor_sequences(result[1])
            self.last_sensor_snapshot = _sensor_snapshot(self, result[1])
            self._publish_monitor(force=True)
        return result

    def step(self, action):
        begin = time.perf_counter()
        result = self.env.step(action)
        self.step_ms.append((time.perf_counter() - begin) * 1000.0)
        self.step_count += 1
        _, reward, terminated, truncated, info = result
        if isinstance(info, dict):
            sequences = _sensor_sequences(info)
            _sequence_deltas(self._previous_sensor_sequences, sequences,
                             self.sensor_fresh_counts)
            self._previous_sensor_sequences.update(sequences)
            self.last_info = info
            self.last_sensor_snapshot = _sensor_snapshot(self, info)
            sim_step = info.get("step_sim_s")
        else:
            sim_step = None
        if not isinstance(sim_step, (int, float)) or not math.isfinite(float(sim_step)):
            sim_step = getattr(getattr(self.unwrapped, "cfg", None), "dt", 0.0)
        self.sim_step_seconds.append(max(0.0, float(sim_step)))
        if terminated or truncated:
            self.episode_ends.append({"mode": str(self._mode), "reward": float(reward),
                                      "reason": info.get("reason"),
                                      "terminated": bool(terminated),
                                      "truncated": bool(truncated)})
        self._publish_monitor(step_ms=self.step_ms[-1])
        return result

    def _publish_monitor(self, *, force=False, policy_ms=None, step_ms=None):
        if self.monitor is None:
            return
        try:
            self.monitor.publish(self.unwrapped, phase=self.phase,
                                 steps=self.step_count, total=self.total_steps,
                                 force=force, policy_ms=policy_ms, step_ms=step_ms)
        except Exception as exc:  # Monitoring is read-only and must not break a rollout.
            self.monitor_errors.append(f"{type(exc).__name__}: {exc}")

    def close(self):
        return self.env.close()


def _timed_env(cfg, mode, *, backend, output_dir, seed, gui=False,
               monitor=None, phase="smoke", total_steps=0):
    base = _make_env(cfg, "search" if mode == "mixed" else mode,
                     backend=backend, output_dir=output_dir, gui=gui)
    task_env = MixedTaskWrapper(base, seed) if mode == "mixed" else base
    return base, MetricsWrapper(task_env, monitor=monitor, phase=phase,
                                total_steps=total_steps)


def _base_observation_space(env):
    return env.observation_space


def _sensor_sequences(info) -> dict[str, float]:
    if not isinstance(info, dict):
        return {}
    values = info.get("sensor_sequences")
    if not isinstance(values, dict):
        return {}
    result = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _sensor_snapshot(env, info) -> dict:
    base = env.unwrapped
    now = float(getattr(base, "time", 0.0) or 0.0)
    ages = {}
    fresh_flags = {}
    frames = getattr(base, "frames", None)
    if isinstance(frames, dict):
        for key, frame in frames.items():
            if isinstance(frame, dict):
                capture = frame.get("capture")
                if capture is not None:
                    try:
                        ages[key] = max(0.0, now - float(capture))
                    except (TypeError, ValueError):
                        pass
        new = getattr(base, "_new", None)
        if isinstance(new, dict):
            fresh_flags = {str(k): bool(v) for k, v in new.items()}
    if isinstance(info, dict):
        candidate = info.get("sensor_age_seconds") or info.get("sensor_age")
        if isinstance(candidate, dict):
            ages.update({str(k): float(v) for k, v in candidate.items()
                         if isinstance(v, (int, float)) and math.isfinite(v)})
        candidate = info.get("sensor_fresh")
        if isinstance(candidate, dict):
            fresh_flags.update({str(k): bool(v) for k, v in candidate.items()})
    return {"simulation_time_seconds": now, "sequences": _sensor_sequences(info),
            "frame_age_seconds": ages, "latest_step_fresh_flags": fresh_flags}


def _sequence_deltas(previous: dict, current: dict, totals: Counter) -> dict:
    deltas = {}
    for key, value in current.items():
        old = previous.get(key, value)
        delta = max(0.0, value - old)
        deltas[key] = delta
        totals[key] += int(delta)
    return deltas


def _robot_state(env):
    base = env.unwrapped
    robot = getattr(base, "robot", None)
    gimbal = getattr(base, "gimbal", None)
    if robot is None or gimbal is None:
        return None
    return np.asarray(robot, dtype=np.float64).copy(), np.asarray(gimbal, dtype=np.float64).copy()


def _save_observation(output: Path, observation, depth_range: float) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "last_delivered_observation.npz",
                        **{key: np.asarray(value) for key, value in observation.items()})
    depth = np.asarray(observation["depth"], dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] < 2:
        raise ValueError(f"Expected depth observation (value, mask, H, W), got {depth.shape}")
    normalized = np.clip(np.nan_to_num(depth[0], nan=0.0, posinf=0.0), 0.0, 1.0)
    valid = np.isfinite(depth[0]) & (depth[1] > 0.5)
    x = normalized * max(float(depth_range), 1e-6) / 5.0
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save the smoke depth preview") from exc
    Image.fromarray(rgb, mode="RGB").save(output / "last_delivered_depth_preview.png")
    return {"observation_file": str(output / "last_delivered_observation.npz"),
            "depth_preview_file": str(output / "last_delivered_depth_preview.png"),
            "depth_shape": list(depth.shape), "valid_depth_pixels": int(np.count_nonzero(valid)),
            "depth_pixels": int(valid.size)}


def _make_config(args) -> Config:
    return Config(size=args.size, plants=args.plants, dt=args.dt, max_time=args.max_time,
                  depth_width=args.depth_width, depth_height=args.depth_height,
                  lidar_rays=args.lidar_rays, lidar_period=args.lidar_period,
                  terrain_enabled=not args.flat_ground)


def _smoke(cfg, args, output: Path) -> dict:
    begin = time.perf_counter()
    base = None
    measured = None
    task_reports = []
    seq_totals = Counter()
    final_observation = None
    last_info = {}
    try:
        base, measured = _timed_env(cfg, "search", backend=args.backend,
                                    output_dir=output / "search_world", seed=args.seed,
                                    monitor=getattr(args, "monitor", None), phase="smoke",
                                    total_steps=64)
        for index, task in enumerate(("search", "goto")):
            options = {"mode": task}
            obs, info = measured.reset(seed=args.seed + index, options=options)
            if not _base_observation_space(measured).contains(obs) or not _observation_is_finite(obs):
                raise AssertionError(f"{task} reset returned an invalid observation")
            initial_observation = {key: np.asarray(value).copy() for key, value in obs.items()}
            initial_state = _robot_state(measured)
            before_sequences = _sensor_sequences(info)
            previous_sequences = dict(before_sequences)
            task_fresh = Counter()
            reward_sum = 0.0
            actions = np.tile(np.array([0.08, 0.05, 0.04, 0.03], dtype=np.float32), (32, 1))
            for action in actions:
                obs, reward, terminated, truncated, info = measured.step(action)
                if not _base_observation_space(measured).contains(obs) or not _observation_is_finite(obs):
                    raise AssertionError(f"{task} step returned an invalid observation")
                if not math.isfinite(float(reward)):
                    raise AssertionError(f"{task} step returned a non-finite reward")
                reward_sum += float(reward)
                current = _sensor_sequences(info)
                _sequence_deltas(previous_sequences, current, task_fresh)
                previous_sequences.update(current)
                if terminated or truncated:
                    raise AssertionError(f"{task} smoke episode ended before 32 active actions: {info}")
            state_after_drive = _robot_state(measured)
            moved = False
            gimbal_moved = False
            if initial_state is not None and state_after_drive is not None:
                moved = bool(np.linalg.norm(state_after_drive[0][:2] - initial_state[0][:2]) > 1e-4)
                gimbal_moved = bool(np.linalg.norm(state_after_drive[1] - initial_state[1]) > 1e-4)
            else:
                state_before = np.asarray(initial_observation["state"])
                state_after = np.asarray(obs["state"])
                moved = bool(np.linalg.norm(state_after[4:6] - state_before[4:6]) > 1e-5
                             or np.linalg.norm(state_after[6:8] - state_before[6:8]) > 1e-5)
                gimbal_moved = bool(np.linalg.norm(state_after[10:13] - state_before[10:13]) > 1e-5)
            if not moved or not gimbal_moved:
                raise AssertionError(f"{task} smoke failed to move vehicle and gimbal (vehicle={moved}, gimbal={gimbal_moved})")
            missing_sensors = [name for name in ("lidar", "depth") if task_fresh.get(name, 0) <= 0]
            if missing_sensors:
                raise AssertionError(f"{task} smoke sensor sequences did not advance for {missing_sensors}; got {dict(task_fresh)}")

            goal_test = None
            if task == "goto":
                base_env = measured.unwrapped
                robot = getattr(base_env, "robot", None)
                set_goal = getattr(base_env, "set_goal", None)
                if robot is None or set_goal is None:
                    raise RuntimeError("GazeboFieldEnv must expose robot pose and set_goal for the goto hold check")
                set_goal(np.asarray(robot[:2], dtype=np.float64).copy())
                obs, info = base_env._observation(), base_env._info()
                hold_reward = 0.0
                steps = 0
                max_hold_steps = max(80, int(math.ceil((cfg.goal_hold + 2.0) / cfg.dt)))
                for _ in range(max_hold_steps):
                    obs, reward, terminated, truncated, info = measured.step(np.zeros(4, np.float32))
                    steps += 1
                    hold_reward += float(reward)
                    current = _sensor_sequences(info)
                    _sequence_deltas(previous_sequences, current, task_fresh)
                    previous_sequences.update(current)
                    if terminated or truncated:
                        break
                if not info.get("is_success", False) or info.get("reason") != "goal_reached":
                    raise AssertionError(f"zero-action goto hold did not reach the goal: {info}")
                if hold_reward <= 0:
                    raise AssertionError(f"goal hold produced no positive completion reward: {hold_reward}")
                goal_test = {"steps": steps, "reward": hold_reward, "reason": info.get("reason"),
                             "is_success": bool(info.get("is_success"))}

            seq_totals.update(task_fresh)
            task_reports.append({"mode": task, "active_steps": 32,
                                 "reward_sum_during_motion": reward_sum,
                                 "sensor_sequence_start": before_sequences,
                                 "sensor_frames_advanced": dict(task_fresh),
                                 "vehicle_moved": moved, "gimbal_moved": gimbal_moved,
                                 "goto_goal_hold": goal_test})
            final_observation, last_info = obs, info
        artifacts = _save_observation(output, final_observation, cfg.depth_range)
        summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "backend": args.backend, "mode": "search+goto", "seed": args.seed,
                   "monitor_port": args.monitor_port,
                   "config": _json_value(vars(cfg)), "tasks": task_reports,
                   "sensor_frames_advanced_total": dict(seq_totals),
                   "last_delivered_sensor_snapshot": _sensor_snapshot(measured, last_info),
                   "monitor_errors": measured.monitor_errors,
                   "artifacts": artifacts, "environment_reset_seconds": measured.reset_seconds,
                   "smoke_wall_seconds_including_startup": time.perf_counter() - begin,
                   "competence_claim": False,
                   "limitations": ["smoke control is a small fixed action, not a learned closed loop",
                                   "sensor sequence advancement confirms messages, not sensor calibration"]}
        _write_json(output / "smoke_summary.json", summary)
        return summary
    finally:
        if measured is not None:
            measured.close()
        elif base is not None:
            base.close()


def _training_env(cfg, args, output: Path):
    base = _make_env(cfg, "search" if args.mode == "mixed" else args.mode,
                     backend=args.backend, output_dir=output / "world", gui=False)
    task_env = MixedTaskWrapper(base, args.seed, task_mode=args.mode, random_start=True)
    measured = MetricsWrapper(task_env, monitor=getattr(args, "monitor", None),
                              phase="training", total_steps=args.steps)
    from stable_baselines3.common.monitor import Monitor
    monitored = Monitor(measured, filename=str(output / "monitor"))
    return base, measured, monitored


def _train(cfg, args, output: Path) -> tuple[dict, Path]:
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingMonitorCallback(BaseCallback):
        def __init__(self, metrics: MetricsWrapper, verbose=0):
            super().__init__(verbose=verbose)
            self.metrics = metrics

        def _on_rollout_start(self):
            self.metrics.phase = "training"
            self.metrics._publish_monitor(force=True)

        def _on_rollout_end(self):
            self.metrics.phase = "optimizing"
            self.metrics._publish_monitor(force=True)

        def _on_step(self):
            return True

    output.mkdir(parents=True, exist_ok=True)
    wall_begin = time.perf_counter()
    base = measured = monitored = model = None
    try:
        base, measured, monitored = _training_env(cfg, args, output)
        model = make_model(monitored, seed=args.seed, n_steps=args.n_steps, verbose=args.verbose)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.policy.named_parameters()}
        initial_checkpoint = output / "initial_checkpoint.zip"
        model.save(initial_checkpoint)
        learn_begin = time.perf_counter()
        callback = TrainingMonitorCallback(measured) if measured.monitor is not None else None
        model.learn(total_timesteps=args.steps, callback=callback)
        learn_seconds = time.perf_counter() - learn_begin
        train_wall_seconds = time.perf_counter() - wall_begin
        measured.phase = "finished"
        measured._publish_monitor(force=True)
        finite = all(bool(torch.isfinite(parameter).all().item())
                     for parameter in model.policy.parameters())
        weight_delta_squared = sum(
            float((parameter.detach() - before[name]).square().sum().item())
            for name, parameter in model.policy.named_parameters())
        checkpoint = output / "checkpoint.zip"
        model.save(checkpoint)
        reloaded = RecurrentPPO.load(checkpoint, env=monitored, device="cpu")
        base_obs = getattr(base, "_last_observation", None)
        if base_obs is None and hasattr(base, "_observation"):
            base_obs = base._observation()
        if base_obs is None:
            raise RuntimeError("GazeboFieldEnv must expose the last delivered observation or _observation() for reload verification")
        test_action, _ = reloaded.predict(base_obs, deterministic=True)
        reload_action_finite = bool(np.all(np.isfinite(test_action)))
        if not finite or weight_delta_squared <= 0 or not reload_action_finite:
            raise RuntimeError("training verification failed: non-finite/unchanged weights or invalid reloaded action")
        actual_steps = int(model.num_timesteps)
        reset_stats = _percentile_stats(np.asarray(measured.reset_seconds) * 1000.0)
        rollout_step_stats = _percentile_stats(measured.step_ms)
        step_metrics = getattr(base, "step_metrics", None)
        if callable(step_metrics):
            step_metrics = step_metrics()
        if step_metrics is None and hasattr(base, "_last_step_wall_s"):
            step_metrics = {"last_step_wall_seconds": float(base._last_step_wall_s),
                            "last_step_sim_seconds": float(getattr(base, "_last_sim_step_s", 0.0))}
        summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "backend": args.backend, "config": _json_value(vars(cfg)), "mode": args.mode,
                   "requested_steps": args.steps, "actual_steps": actual_steps,
                   "monitor_port": args.monitor_port,
                   "n_steps": args.n_steps, "n_epochs": 3,
                   "training_wall_seconds_including_env_startup_export_and_resets": train_wall_seconds,
                   "model_learn_seconds": learn_seconds,
                   "training_steps_per_second": actual_steps / max(train_wall_seconds, 1e-9),
                   "rollout_step_wall_ms": rollout_step_stats,
                   "reset_wall_ms": reset_stats,
                   "reset_count": len(measured.reset_seconds),
                   "episode_modes": measured.episode_modes,
                   "episode_mode_counts": dict(Counter(measured.episode_modes)),
                   "completed_episodes": measured.episode_ends,
                   "effective_simulated_seconds": float(sum(measured.sim_step_seconds)),
                   "effective_simulated_seconds_per_wall_second": float(sum(measured.sim_step_seconds)) / max(train_wall_seconds, 1e-9),
                   "environment_step_metrics": step_metrics,
                   "sensor_frames_advanced": dict(measured.sensor_fresh_counts),
                   "sensor_snapshot": measured.last_sensor_snapshot,
                   "monitor_errors": measured.monitor_errors,
                   "algorithm": "RecurrentPPO", "policy": "SensorEncoder + LSTM128",
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "parameters": sum(p.numel() for p in model.policy.parameters()),
                   "weights_finite": finite, "weight_delta_squared": weight_delta_squared,
                   "checkpoint_reload_passed": True, "reload_prediction_finite": reload_action_finite,
                   "initial_checkpoint": str(initial_checkpoint.resolve()),
                   "initial_checkpoint_note": "Untrained initialization snapshot for parameter comparison only; it is not a task-competence baseline.",
                   "training_success_claim": False,
                   "note": "This run verifies optimization, changed finite weights and checkpoint execution only; it does not establish task competence.",
                   "checkpoint": str(checkpoint.resolve())}
        _write_json(output / "train_summary.json", summary)
        return summary, checkpoint
    finally:
        if monitored is not None:
            monitored.close()
        elif measured is not None:
            measured.close()
        elif base is not None:
            base.close()


def _policy_speed(runner: PolicyRunner, observation, *, caching: bool, repeats=200) -> dict:
    extractor = runner.model.policy.features_extractor
    extractor.cache_inference = caching
    extractor._cache.clear()
    runner.reset()
    for _ in range(20):
        runner.action(observation)
    runner.reset()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        action = runner.action(observation)
        samples.append((time.perf_counter() - begin) * 1000.0)
        if not np.all(np.isfinite(action)):
            raise AssertionError("policy returned a non-finite action in inference benchmark")
    runner.reset()
    extractor._cache.clear()
    return _percentile_stats(samples)


def _benchmark_unpaced(env, runner, cfg, args):
    wall_begin = time.perf_counter()
    obs, info = env.reset(seed=args.seed)
    if not env.observation_space.contains(obs) or not _observation_is_finite(obs):
        raise AssertionError("benchmark reset returned an invalid observation")
    policy_cached = _policy_speed(runner, obs, caching=True)
    policy_uncached = _policy_speed(runner, obs, caching=False)
    runner.reset()
    action_ms, env_step_ms, combined_ms = [], [], []
    reset_interrupts = []
    sensor_fresh = Counter()
    previous_sequences = _sensor_sequences(info)
    last_info = info
    final_obs = obs
    for step_index in range(args.benchmark_steps):
        begin = time.perf_counter()
        action = runner.action(obs)
        after_policy = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        finish = time.perf_counter()
        if not _observation_is_finite(obs) or not math.isfinite(float(reward)):
            raise AssertionError("benchmark rollout produced a non-finite observation or reward")
        action_ms.append((after_policy - begin) * 1000.0)
        env_step_ms.append((finish - after_policy) * 1000.0)
        combined_ms.append((finish - begin) * 1000.0)
        current = _sensor_sequences(info)
        _sequence_deltas(previous_sequences, current, sensor_fresh)
        previous_sequences.update(current)
        last_info = info
        final_obs = obs
        if terminated or truncated:
            reset_begin = time.perf_counter()
            obs, reset_info = env.reset(seed=args.seed + 1 + len(reset_interrupts))
            reset_interrupts.append({"step": step_index + 1,
                                     "wall_seconds": time.perf_counter() - reset_begin,
                                     "mode": getattr(env.unwrapped, "mode", None),
                                     "reason": info.get("reason")})
            runner.reset()
            previous_sequences = _sensor_sequences(reset_info)
            last_info = reset_info
            final_obs = obs
    return {"policy_cached_inference_200": policy_cached,
            "policy_uncached_inference_200": policy_uncached,
            "unpaced_rollout_steps": args.benchmark_steps,
            "unpaced_wall_seconds_including_initial_and_episode_resets": time.perf_counter() - wall_begin,
            "policy_inference_ms_per_decision": _percentile_stats(action_ms),
            "environment_step_ms_per_decision": _percentile_stats(env_step_ms),
            "combined_decision_ms": _percentile_stats(combined_ms),
            "decision_fps_policy_only": 1000.0 / max(float(np.mean(action_ms)), 1e-9),
            "decision_fps_environment_step_only": 1000.0 / max(float(np.mean(env_step_ms)), 1e-9),
            "decision_fps_combined": 1000.0 / max(float(np.mean(combined_ms)), 1e-9),
            "episode_reset_interruptions_excluded_from_step_samples": reset_interrupts,
            "reset_wall_ms": _percentile_stats(np.asarray(env.reset_seconds) * 1000.0),
            "sensor_fresh_counts": dict(sensor_fresh),
            "last_sensor_snapshot": _sensor_snapshot(env, last_info),
            "last_observation": final_obs}


def _benchmark_paced(env, runner, cfg, args) -> dict:
    period = cfg.dt
    count = max(2, int(round(args.paced_seconds / period)))
    reset_begin = time.perf_counter()
    obs, info = env.reset(seed=args.seed + 10_000)
    initial_reset_seconds = time.perf_counter() - reset_begin
    runner.reset()
    previous_sequences = _sensor_sequences(info)
    sensor_fresh = Counter()
    begin = time.perf_counter()
    deadline = begin
    starts = []
    actions_ms, env_ms, combined_ms = [], [], []
    active_misses = 0
    reset_interrupts = []
    for index in range(count):
        delay = deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        active_start = time.perf_counter()
        starts.append(active_start)
        action_begin = time.perf_counter()
        action = runner.action(obs)
        action_end = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_end = time.perf_counter()
        actions_ms.append((action_end - action_begin) * 1000.0)
        env_ms.append((step_end - action_end) * 1000.0)
        combined_ms.append((step_end - active_start) * 1000.0)
        current = _sensor_sequences(info)
        _sequence_deltas(previous_sequences, current, sensor_fresh)
        previous_sequences.update(current)
        next_deadline = deadline + period
        if step_end > next_deadline:
            active_misses += 1
        if terminated or truncated:
            reset_begin = time.perf_counter()
            obs, reset_info = env.reset(seed=args.seed + 10_001 + len(reset_interrupts))
            reset_seconds = time.perf_counter() - reset_begin
            reset_interrupts.append({"step": index + 1, "wall_seconds": reset_seconds,
                                     "reason": info.get("reason")})
            next_deadline += reset_seconds
            runner.reset()
            previous_sequences = _sensor_sequences(reset_info)
        deadline = next_deadline
    finish = time.perf_counter()
    reset_total = sum(row["wall_seconds"] for row in reset_interrupts)
    wall_seconds = finish - begin
    active_wall_seconds = max(1e-9, wall_seconds - reset_total)
    raw_intervals = np.diff(np.asarray(starts, dtype=np.float64))
    interval_without_resets = raw_intervals.copy()
    # Each reset occurs between its step's start and the following decision start.
    for row in reset_interrupts:
        following = int(row["step"])
        if 0 <= following - 1 < len(interval_without_resets):
            interval_without_resets[following - 1] = max(
                0.0, interval_without_resets[following - 1] - row["wall_seconds"])
    return {"target_hz": 1.0 / period, "steps": count,
            "requested_paced_seconds": args.paced_seconds,
            "wall_seconds_including_resets": wall_seconds,
            "initial_reset_wall_seconds": initial_reset_seconds,
            "wall_seconds_including_initial_and_episode_resets": wall_seconds + initial_reset_seconds,
            "active_wall_seconds_excluding_resets": active_wall_seconds,
            "actual_start_hz_active": (count - 1) / max(float(np.sum(interval_without_resets)), 1e-9),
            "actual_start_hz_wall": (count - 1) / max(float(raw_intervals.sum()), 1e-9),
            "start_interval_ms_p95_active": float(np.percentile(interval_without_resets * 1000.0, 95)),
            "start_interval_ms_p95_wall": float(np.percentile(raw_intervals * 1000.0, 95)),
            "active_deadline_misses": active_misses,
            "policy_inference_ms_per_decision": _percentile_stats(actions_ms),
            "environment_step_ms_per_decision": _percentile_stats(env_ms),
            "combined_decision_ms": _percentile_stats(combined_ms),
            "episode_reset_interruptions": reset_interrupts,
            "environment_resets_total": len(env.reset_seconds),
            "sensor_fresh_counts": dict(sensor_fresh),
            "last_sensor_snapshot": _sensor_snapshot(env, info)}


def _benchmark(cfg, args, output: Path, checkpoint: Path) -> dict:
    import torch

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Benchmark checkpoint does not exist: {checkpoint}")
    begin = time.perf_counter()
    base = measured = None
    try:
        base, measured = _timed_env(cfg, args.mode, backend=args.backend,
                                    output_dir=output / "world", seed=args.seed,
                                    monitor=getattr(args, "monitor", None), phase="benchmark",
                                    total_steps=args.benchmark_steps + int(round(args.paced_seconds / cfg.dt)))
        runner = PolicyRunner(measured, checkpoint)
        setup_seconds = time.perf_counter() - begin
        unpaced = _benchmark_unpaced(measured, runner, cfg, args)
        unpaced.pop("last_observation")
        paced = _benchmark_paced(measured, runner, cfg, args)
        environment_metrics = getattr(base, "step_metrics", None)
        if callable(environment_metrics):
            environment_metrics = environment_metrics()
        if environment_metrics is None and hasattr(base, "_last_step_wall_s"):
            environment_metrics = {"last_step_wall_seconds": float(base._last_step_wall_s),
                                   "last_step_sim_seconds": float(getattr(base, "_last_sim_step_s", 0.0))}
        driver = getattr(base, "driver", None)
        gazebo_version = (getattr(base, "gazebo_version", None)
                          or getattr(driver, "gazebo_version", None)
                          or os.environ.get("GZ_SIM_VERSION"))
        summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "backend": args.backend, "mode": args.mode,
                   "config": _json_value(vars(cfg)), "checkpoint": str(checkpoint.resolve()),
                   "monitor_port": args.monitor_port,
                   "policy_source": runner.label,
                   "benchmark_setup_wall_seconds_including_env_startup_export_reload": setup_seconds,
                   "hardware": _hardware_versions(torch),
                   "gui_enabled": False,
                   "gazebo_expected_abi": "Gazebo Harmonic / gz-sim 8",
                   "gazebo_reported_version": gazebo_version,
                   "ros_distribution": os.environ.get("ROS_DISTRO"),
                   **_physics_runtime_settings(driver),
                   "unpaced": unpaced, "paced_control": paced,
                   "environment_step_metrics": environment_metrics,
                   "final_sensor_snapshot": measured.last_sensor_snapshot,
                   "competence_claim": False,
                   "decision_rate_note": "Decision Hz is measured at policy/environment loop boundaries; physics iterations/Hz are not reported as policy decision Hz."}
        _write_json(output / "benchmark_summary.json", summary)
        return summary
    finally:
        if measured is not None:
            measured.close()
        elif base is not None:
            base.close()


def _hardware_versions(torch_module=None) -> dict:
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").is_file():
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware")) and ":" in line:
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    versions = {name: _distribution_version(name) for name in (
        "numpy", "torch", "stable-baselines3", "sb3-contrib", "Pillow")}
    if torch_module is not None:
        versions["torch"] = getattr(torch_module, "__version__", versions["torch"])
    return {"os": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
            "packages": versions, "device": "cpu",
            "torch_threads": (torch_module.get_num_threads() if torch_module is not None else None)}


def _run_all(cfg, args, output):
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "backend": args.backend, "mode": args.mode, "output": str(output.resolve())}
    smoke_args = argparse.Namespace(**vars(args))
    smoke = _smoke(cfg, smoke_args, output / "smoke")
    train_args = argparse.Namespace(**vars(args))
    train_summary, checkpoint = _train(cfg, train_args, output / "train")
    benchmark_args = argparse.Namespace(**vars(args))
    benchmark = _benchmark(cfg, benchmark_args, output / "benchmark", checkpoint)
    summary.update({"smoke": smoke, "train": train_summary,
                    "benchmark": benchmark,
                    "task_mode_coverage": _task_mode_coverage(smoke, train_summary),
                    "training_success_claim": False})
    _write_json(output / "all_summary.json", summary)
    return summary


def _task_mode_coverage(smoke_summary: dict, train_summary: dict) -> dict:
    smoke_modes = sorted({str(task.get("mode")) for task in smoke_summary.get("tasks", [])
                          if task.get("mode")})
    training_modes = [str(mode) for mode in train_summary.get("episode_modes", [])]
    training_set = set(training_modes)
    smoke_set = set(smoke_modes)
    required = {"search", "goto"}
    return {"training_episode_modes": training_modes,
            "training_episode_mode_counts": dict(Counter(training_modes)),
            "training_rollout_covers_search_and_goto": required.issubset(training_set),
            "smoke_task_self_check_modes": smoke_modes,
            "smoke_self_check_covers_search_and_goto": required.issubset(smoke_set),
            "combined_observed_modes": sorted(training_set | smoke_set),
            "interpretation": "Smoke exercises both task types with fixed actions; training coverage is reported separately from its actual reset modes."}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and benchmark the recurrent field policy through GazeboFieldEnv.")
    parser.add_argument("command", nargs="?", choices=("all", "smoke", "train", "benchmark"), default="all")
    parser.add_argument("--backend", choices=("gazebo", "fieldsim"), default="gazebo")
    parser.add_argument("--mode", choices=("mixed", "search", "goto"), default="mixed")
    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--plants", type=int, default=400)
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--depth-width", type=int, default=88)
    parser.add_argument("--depth-height", type=int, default=160)
    parser.add_argument("--lidar-rays", type=int, default=400)
    parser.add_argument("--lidar-period", type=float, default=0.1)
    parser.add_argument("--flat-ground", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--benchmark-steps", type=int, default=128)
    parser.add_argument("--paced-seconds", type=float, default=5.0)
    parser.add_argument("--monitor-port", type=int,
                        help="optional read-only Gazebo training monitor endpoint port")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args(argv)
    if args.size <= 0 or args.plants < 0 or args.max_time <= 0 or args.dt <= 0:
        parser.error("size/max-time/dt must be positive and plants nonnegative")
    if args.steps <= 0 or args.n_steps <= 0 or args.benchmark_steps <= 0:
        parser.error("steps, n-steps and benchmark-steps must be positive")
    if args.paced_seconds <= 0:
        parser.error("paced-seconds must be positive")
    if args.monitor_port is not None and not 1024 <= args.monitor_port <= 65535:
        parser.error("monitor-port must be 1024..65535")
    if args.depth_width <= 0 or args.depth_height <= 0 or args.lidar_rays <= 0 or args.lidar_period <= 0:
        parser.error("sensor dimensions/rates must be positive")
    if args.output is None:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        args.output = DEFAULT_OUTPUT_ROOT / stamp
    args.output = args.output.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = _make_config(args)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    monitor = None
    try:
        if args.monitor_port is not None:
            _check_monitor_port_available(args.monitor_port)
            from simulation.gazebo_monitor import TrainingMonitor
            monitor = TrainingMonitor(args.monitor_port)
        args.monitor = monitor
        with _threadpool_limit_context():
            if args.command == "smoke":
                result = _smoke(cfg, args, output)
            elif args.command == "train":
                result, _ = _train(cfg, args, output)
            elif args.command == "benchmark":
                checkpoint = args.checkpoint
                if checkpoint is None:
                    checkpoint = output.parent / "train" / "checkpoint.zip"
                result = _benchmark(cfg, args, output, checkpoint)
            else:
                result = _run_all(cfg, args, output)
    except BaseException as exc:
        failure = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "command": args.command, "backend": args.backend,
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "output": str(output)}
        _write_json(output / "failure.json", failure)
        raise
    finally:
        if monitor is not None:
            monitor.close()
    print(json.dumps(_json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved under: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
