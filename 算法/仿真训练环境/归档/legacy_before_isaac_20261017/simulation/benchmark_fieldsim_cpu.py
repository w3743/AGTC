#!/usr/bin/env python3
"""Benchmark the existing FieldSim Gym step loop with a stationary action.

Default protocol matches the Gazebo stationary comparison: seed 7, one
simulated second of warmup, then 20 simulated seconds measured over 3 fresh
environments. This measures the FieldSim Gym environment, including its
simulated sensor/detection/reward work; it is not an equal-feature Gazebo
training-throughput comparison.
"""
from __future__ import annotations

import os

# Keep numerical backends on the same single-thread budget as the Gazebo run.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from field_sim.env import Config, FieldEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "gazebo_field_runs" / "field_stationary_comparison.json"
SENSOR_KINDS = ("lidar", "depth", "camera")


def _steps_for(seconds: float, dt: float, label: str) -> int:
    exact = seconds / dt
    steps = round(exact)
    if steps <= 0 or not math.isclose(exact, steps, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{label} must be a positive multiple of Config.dt={dt}")
    return int(steps)


def _run_once(repeat: int, seed: int, warmup_seconds: float,
              measure_seconds: float) -> dict:
    config = Config()
    warmup_steps = _steps_for(warmup_seconds, config.dt, "warmup seconds")
    measured_steps = _steps_for(measure_seconds, config.dt, "measurement seconds")
    action = np.zeros(4, dtype=np.float32)
    env = FieldEnv(config, "search")
    reset_started = time.perf_counter()
    try:
        observation, reset_info = env.reset(seed=seed)
        reset_wall = time.perf_counter() - reset_started
        warmup_started = time.perf_counter()
        warmup_reward = 0.0
        for _ in range(warmup_steps):
            observation, reward, terminated, truncated, info = env.step(action)
            warmup_reward += float(reward)
            if terminated or truncated:
                raise RuntimeError(f"episode ended during warmup: {info.get('reason')}")
        warmup_wall = time.perf_counter() - warmup_started

        measure_start_sim = float(env.time)
        initial_serial = {key: int(env._serial[key]) for key in SENSOR_KINDS}
        initial_dropped = {key: int(env._dropped[key]) for key in SENSOR_KINDS}
        capture_timestamps = {key: [] for key in SENSOR_KINDS}
        measured_reward = 0.0
        measured_started = time.perf_counter()
        for _ in range(measured_steps):
            serial_before = {key: int(env._serial[key]) for key in SENSOR_KINDS}
            observation, reward, terminated, truncated, info = env.step(action)
            measured_reward += float(reward)
            for key in SENSOR_KINDS:
                if int(env._serial[key]) > serial_before[key]:
                    capture_timestamps[key].append(float(env.frames[key]["capture"]))
            if terminated or truncated:
                raise RuntimeError(f"episode ended during measurement: {info.get('reason')}")
        measured_wall = time.perf_counter() - measured_started
        measure_end_sim = float(env.time)
        measured_sim = measure_end_sim - measure_start_sim
        if measured_wall <= 0 or measured_sim <= 0:
            raise RuntimeError("non-positive benchmark duration")

        fresh_frames = {
            key: int(env._serial[key]) - initial_serial[key]
            for key in SENSOR_KINDS
        }
        dropped_frames = {
            key: int(env._dropped[key]) - initial_dropped[key]
            for key in SENSOR_KINDS
        }
        sensor_metrics = {}
        for key in SENSOR_KINDS:
            stamps = capture_timestamps[key]
            unique = sorted(set(round(stamp, 9) for stamp in stamps))
            expected_period = {
                "lidar": config.lidar_period,
                "depth": config.depth_period,
                "camera": config.camera_period,
            }[key]
            age = float(env.time - env.frames[key]["capture"])
            sensor_metrics[key] = {
                "fresh_frames": fresh_frames[key],
                "fresh_hz_sim": fresh_frames[key] / measured_sim,
                "expected_period_s": expected_period,
                "expected_hz": 1.0 / expected_period,
                "unique_capture_timestamps": len(unique),
                "capture_timestamp_span_s": (
                    unique[-1] - unique[0] if len(unique) > 1 else 0.0
                ),
                "dropped_frames": dropped_frames[key],
                "final_frame_age_s": age,
                "capture_times_monotonic": all(
                    a < b for a, b in zip(stamps, stamps[1:])
                ),
            }

        shapes = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in observation.items()
        }
        return {
            "repeat": repeat,
            "seed": seed,
            "mode": env.mode,
            "scene_id": int(env.scene_id),
            "terrain_seed": env.scene.terrain_seed,
            "terrain_enabled": bool(config.terrain_enabled),
            "reset_wall_seconds_excluded": reset_wall,
            "warmup_sim_seconds": float(env.cfg.dt * warmup_steps),
            "warmup_steps": warmup_steps,
            "warmup_wall_seconds_excluded": warmup_wall,
            "warmup_reward_excluded": warmup_reward,
            "action": {"values": [0.0, 0.0, 0.0, 0.0], "meaning": "stationary zero command"},
            "measurement": {
                "sim_seconds": measured_sim,
                "wall_seconds": measured_wall,
                "steps": measured_steps,
                "steps_per_wall_second": measured_steps / measured_wall,
                "rtf_sim_over_wall": measured_sim / measured_wall,
                "reward_sum": measured_reward,
                "reward_per_step": measured_reward / measured_steps,
                "sim_start_s": measure_start_sim,
                "sim_end_s": measure_end_sim,
            },
            "sensor_freshness": sensor_metrics,
            "final_episode_info": info,
            "observation_contract": shapes,
        }
    finally:
        env.close()


def _aggregate(repeats: list[dict]) -> dict:
    metric_paths = (
        ("measurement", "steps_per_wall_second"),
        ("measurement", "rtf_sim_over_wall"),
        ("measurement", "wall_seconds"),
    )
    aggregate = {}
    for parent, key in metric_paths:
        values = [float(run[parent][key]) for run in repeats]
        aggregate[f"{parent}.{key}"] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    for kind in SENSOR_KINDS:
        values = [float(run["sensor_freshness"][kind]["fresh_hz_sim"])
                  for run in repeats]
        aggregate[f"sensor_freshness.{kind}.fresh_hz_sim"] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return aggregate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--measure-seconds", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.repeats < 1:
        parser.error("seed must be non-negative and repeats must be positive")

    results = [
        _run_once(index + 1, args.seed, args.warmup_seconds, args.measure_seconds)
        for index in range(args.repeats)
    ]
    config = Config()
    document = {
        "schema": "fieldsim-stationary-cpu-benchmark-v1",
        "comparison_scope": (
            "FieldSim Gym step throughput with its simulated sensors, detections, and reward; "
            "Gazebo's scene/sensor smoke run has no matching full Gym observation/reward loop, "
            "so this is not an equal-feature training-throughput ranking."
        ),
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "mode": "search",
            "action": [0.0, 0.0, 0.0, 0.0],
            "warmup_sim_seconds": args.warmup_seconds,
            "measurement_sim_seconds": args.measure_seconds,
            "cpu_thread_environment": {
                key: os.environ[key] for key in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
            "config": {
                "dt_s": config.dt,
                "lidar_period_s": config.lidar_period,
                "depth_period_s": config.depth_period,
                "camera_period_s": config.camera_period,
                "plants": config.plants,
                "leaves_per_plant": config.leaves_per_plant,
                "lidar_rays": config.lidar_rays,
                "depth_shape_hw": [config.depth_height, config.depth_width],
                "validate_scene": config.validate_scene,
                "terrain_enabled": config.terrain_enabled,
            },
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "field_sim_module": FieldEnv.__module__,
        },
        "repeats_results": results,
        "aggregate": _aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "aggregate": document["aggregate"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
