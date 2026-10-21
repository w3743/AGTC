"""One Gazebo goto reachability probe with a simple point-goal controller.

This is a control/dynamics diagnostic on an evaluator-selected clear route. It
is not a learned-policy or obstacle-avoidance result and does not change the
environment's four-dimensional action mapping.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from field_sim.env import Config, wrap
from simulation.evaluate_field_gazebo import _scene_identity, _select_safe_goto_goal
from simulation.gazebo_env import GazeboFieldEnv


def _point_goal_action(env):
    """Return the requested normalized four-channel action and P terms."""
    c = env.cfg
    delta = np.asarray(env.goal, dtype=np.float64) - np.asarray(env.robot[:2], dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = wrap(bearing - float(env.robot[2]))
    desired_v = min(0.75, 0.6 * distance) * max(0.0, 1.0 - abs(heading_error) / 1.3)
    desired_w = float(np.clip(1.5 * heading_error, -c.max_turn_rate, c.max_turn_rate))
    action = np.asarray([desired_v / c.max_speed,
                         desired_w / c.max_turn_rate,
                         0.0, 0.0], dtype=np.float32)
    target_v = float(env.robot[3] + np.clip(desired_v - env.robot[3],
                                           -c.acceleration * c.dt, c.acceleration * c.dt))
    target_w = float(env.robot[4] + np.clip(desired_w - env.robot[4],
                                           -c.angular_acceleration * c.dt,
                                           c.angular_acceleration * c.dt))
    return action, {
        "distance_m": distance,
        "heading_error_rad": heading_error,
        "desired_linear_mps": float(desired_v),
        "desired_yaw_rate_radps": desired_w,
        "acceleration_limited_linear_mps": target_v,
        "acceleration_limited_yaw_rate_radps": target_w,
    }


def _pose_record(env):
    return {
        "xy_field_m": [float(env.robot[0]), float(env.robot[1])],
        "z_m": float(getattr(env, "_body_height", math.nan)),
        "yaw_rad": float(env.robot[2]),
        "linear_speed_mps": float(env.robot[3]),
        "yaw_rate_radps": float(env.robot[4]),
        "roll_rad": float(getattr(env, "_body_roll", math.nan)),
        "pitch_rad": float(getattr(env, "_body_pitch", math.nan)),
        "gimbal_rad": [float(v) for v in np.asarray(env.gimbal).reshape(-1)[:2]],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def run_probe(output: Path, *, seed: int = 10007) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config(max_time=20.0)
    env = GazeboFieldEnv(cfg, mode="goto", output_dir=output / "gazebo_episode", gui=False)
    rows: list[dict] = []
    result: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "gazebo",
        "probe": "simple P point-goal control on the evaluator's clear local route",
        "learned_policy": False,
        "obstacle_avoidance_controller": False,
        "uses_original_four_channel_action_contract": True,
        "mode": "goto",
        "seed": int(seed),
        "config": {"size_m": cfg.size, "plants": cfg.plants,
                    "dt_s": cfg.dt, "episode_budget_s": cfg.max_time,
                    "max_speed_mps": cfg.max_speed,
                    "max_yaw_rate_radps": cfg.max_turn_rate,
                    "goal_tolerance_m": cfg.goal_tolerance,
                    "goal_hold_s": cfg.goal_hold,
                    "acceleration_mps2": cfg.acceleration,
                    "angular_acceleration_radps2": cfg.angular_acceleration},
        "controller": {
            "linear_speed_mps": "min(0.75, 0.6 * distance) * max(0, 1 - abs(heading_error) / 1.3)",
            "yaw_rate_radps": "clip(1.5 * heading_error, -max_turn_rate, max_turn_rate)",
            "normalized_action": "[linear_speed/max_speed, yaw_rate/max_yaw_rate, 0, 0]",
            "gimbal_rates": [0.0, 0.0],
        },
        "pose_source": "Gazebo PosePublisher-backed GazeboFieldEnv robot pose; XY translated to FieldSim field frame",
        "score_interpretation": "Rule-based control/dynamics reachability probe only; not AI performance, not obstacle avoidance, not a learned-policy evaluation.",
    }
    env_info = {}
    episode_started = None
    error = None
    shutdown: dict = {}
    try:
        obs, env_info = env.reset(seed=seed)
        identity = _scene_identity(env)
        goal, route = _select_safe_goto_goal(env)
        env.set_goal(goal)
        obs = env._observation()
        env_info = env._info()
        start_xy = np.asarray(env.robot[:2], dtype=np.float64).copy()
        initial_distance = float(np.linalg.norm(env.goal - start_xy))
        result.update({"scene": identity, "goal_route": route,
                       "goal_field_xy_m": np.asarray(goal, dtype=np.float64).tolist(),
                       "initial_pose": _pose_record(env),
                       "initial_goal_distance_m": initial_distance,
                       "reset_wall_s": float(env_info.get("reset_wall_s", math.nan)),
                       "episode_dir": str(env._episode_dir)})
        path_length = 0.0
        episode_started = time.perf_counter()
        max_steps = int(math.ceil(cfg.max_time / cfg.dt))
        terminated = truncated = False
        for step_index in range(max_steps):
            before_xy = np.asarray(env.robot[:2], dtype=np.float64).copy()
            action, command = _point_goal_action(env)
            obs, reward, terminated, truncated, env_info = env.step(action)
            path_length += float(np.linalg.norm(np.asarray(env.robot[:2]) - before_xy))
            row = {
                "step": step_index + 1,
                "sim_time_s": float(env.time),
                "action_normalized": action.astype(float).tolist(),
                "command": command,
                "pose": _pose_record(env),
                "goal_distance_m": float(env_info.get("distance", math.inf)),
                "reward": float(reward),
                "total_reward": float(env.total_reward),
                "step_wall_s": float(env_info.get("step_wall_s", math.nan)),
                "step_sim_s": float(env_info.get("step_sim_s", math.nan)),
                "reason": env_info.get("reason", ""),
                "physical_truncation": env_info.get("physical_truncation"),
                "damage_count": int(env_info.get("damage", len(env.damaged))),
                "collision_count": int(env_info.get("collisions", env.collisions)),
            }
            rows.append(row)
            if terminated or truncated:
                break
        episode_wall_s = time.perf_counter() - episode_started
        result.update({
            "steps": len(rows),
            "sim_seconds": float(env.time),
            "episode_wall_s": episode_wall_s,
            "termination_reason": env_info.get("reason", ""),
            "physical_reason": env_info.get("physical_truncation"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "goal_success": env_info.get("reason") == "goal_reached",
            "arrival_time_s": float(env.time) if env_info.get("reason") == "goal_reached" else None,
            "initial_goal_distance_m": initial_distance,
            "final_goal_distance_m": float(env_info.get("distance", math.inf)),
            "goal_progress_m": initial_distance - float(env_info.get("distance", math.inf)),
            "path_length_m": path_length,
            "net_displacement_m": float(np.linalg.norm(np.asarray(env.robot[:2]) - start_xy)),
            "final_pose": _pose_record(env),
            "final_speed_mps": float(env.robot[3]),
            "final_yaw_rate_radps": float(env.robot[4]),
            "crushed_plant_count": int(env_info.get("damage", len(env.damaged))),
            "rock_collision_count": int(env_info.get("collisions", env.collisions)),
            "reward_sum": float(env.total_reward),
            "sensor_actual_frames": env_info.get("sensor_actual_frames"),
            "sensor_actual_timestamps_s": env_info.get("sensor_actual_timestamps_s"),
        })
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["error"] = error
    finally:
        driver = getattr(env, "driver", None)
        process = getattr(driver, "process", None) if driver is not None else None
        shutdown["gz_process_pid"] = getattr(process, "pid", None)
        shutdown["gz_process_poll_before"] = process.poll() if process is not None else None
        driver_close_started = time.perf_counter()
        if driver is not None:
            try:
                driver.close()
            except BaseException as exc:
                shutdown["driver_close_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                shutdown["driver_close_wall_s"] = time.perf_counter() - driver_close_started
                shutdown["shutdown_method"] = getattr(driver, "shutdown_method", "owned_process_signal")
                shutdown["shutdown_response"] = getattr(driver, "shutdown_response", None)
                shutdown["shutdown_error"] = getattr(driver, "shutdown_error", None)
                shutdown["gz_process_exit_code"] = process.poll() if process is not None else None
                env.driver = None
        else:
            shutdown["driver_close_wall_s"] = time.perf_counter() - driver_close_started
        env_close_started = time.perf_counter()
        try:
            env.close()
        except BaseException as exc:
            shutdown["env_close_error"] = f"{type(exc).__name__}: {exc}"
        shutdown["env_close_after_driver_s"] = time.perf_counter() - env_close_started
        result["shutdown"] = shutdown
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["trajectory_rows"] = len(rows)
        if rows:
            _write_json(output / "trajectory.json", {"coordinate_frame": "field XY m, Gazebo-derived pose",
                                                       "rows": rows})
        if getattr(env, "_snapshot", None) is not None:
            try:
                _write_json(output / "final_state.json", env.state_payload())
            except BaseException as exc:
                result["final_state_write_error"] = f"{type(exc).__name__}: {exc}"
        if error:
            result["status"] = "failed"
        elif result.get("goal_success"):
            result["status"] = "goal_reached"
        else:
            result["status"] = "completed_without_goal_reached"
        _write_json(output / "probe.json", result)
    if error:
        raise RuntimeError(f"Gazebo goto probe failed; evidence saved under {output}: {error}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10007)
    args = parser.parse_args(argv)
    result = run_probe(args.output, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
