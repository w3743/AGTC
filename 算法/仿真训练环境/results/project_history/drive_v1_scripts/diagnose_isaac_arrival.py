"""Capture step-by-step evidence for short goto failures in two policy checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "results/isaac_field_migration/field_policy_16384_latest.zip"
AFTER = ROOT / "results/isaac_field_improvement/short_goto_20261017_175932/field_policy_after.zip"
OUTPUT = ROOT / "results/isaac_field_diagnosis/arrival_trace_20261017"
SEEDS = (51001, 51002, 51003, 51004)


def _policy_metadata(model: Any, path: Path) -> dict[str, Any]:
    learning_rate = getattr(model, "learning_rate", None)
    if callable(learning_rate):
        learning_rate = f"callable:{getattr(learning_rate, '__name__', type(learning_rate).__name__)}"
    return {
        "checkpoint": str(path),
        "model_class": type(model).__name__,
        "num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "gamma": float(model.gamma),
        "learning_rate": learning_rate,
        "target_kl": None if model.target_kl is None else float(model.target_kl),
        "goal_gain": float(model.policy.features_extractor.goal_gain),
    }


def _episode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    start_distance = float(rows[0]["distance_before_m"])
    final = rows[-1]
    closest = min(rows, key=lambda row: row["distance_after_m"])
    far_bearing = [row for row in rows if row["distance_before_m"] > 0.6 and abs(row["goal_bearing_rad"]) > math.radians(10)]
    turn_samples = [row for row in far_bearing if abs(row["action_4"][1]) > 0.005]
    tolerance_rows = [row for row in rows if row["distance_after_m"] <= 0.4]
    abs_v_error = [abs(row["actual_v_w_after"][0] - row["command_v_w"][0]) for row in rows]
    abs_w_error = [abs(row["actual_v_w_after"][1] - row["command_v_w"][1]) for row in rows]
    goal_observation_error = [row["goal_observation_error_m"] for row in rows]
    turn_action_aligned = [
        math.copysign(1.0, row["action_4"][1]) == math.copysign(1.0, row["goal_bearing_rad"])
        for row in turn_samples
    ]
    actual_yaw_deltas = [
        (row["pose_after_xy_yaw"][2] - row["pose_before_xy_yaw"][2] + math.pi) % (2.0 * math.pi) - math.pi
        for row in rows
    ]
    dt = float(rows[0].get("dt_s", 0.05))
    actual_turn_samples = [
        (row, yaw_delta)
        for row, yaw_delta in zip(rows, actual_yaw_deltas)
        if abs(row["command_v_w"][1]) > 0.01 and abs(yaw_delta) > 1e-5
    ]
    commanded_yaw_integral = sum(abs(row["command_v_w"][1]) * dt for row in rows)
    actual_yaw_travel = sum(abs(delta) for delta in actual_yaw_deltas)
    return {
        "steps": len(rows),
        "success": bool(final["terminated"] and final["reason"] == "goal_reached"),
        "reason": final["reason"],
        "start_distance_m": start_distance,
        "final_distance_m": float(final["distance_after_m"]),
        "closest_distance_m": float(closest["distance_after_m"]),
        "closest_step": int(closest["step"]),
        "closest_step_action_4": closest["action_4"],
        "closest_step_command_v_w": closest["command_v_w"],
        "closest_step_actual_v_w": closest["actual_v_w_after"],
        "within_goal_tolerance_steps": len(tolerance_rows),
        "mean_abs_forward_tracking_error_mps": float(np.mean(abs_v_error)),
        "mean_abs_yaw_rate_tracking_error_radps": float(np.mean(abs_w_error)),
        "mean_abs_command_yaw_rate_radps": float(np.mean([abs(row["command_v_w"][1]) for row in rows])),
        "mean_abs_actual_yaw_rate_radps": float(np.mean([abs(row["actual_v_w_after"][1]) for row in rows])),
        "commanded_abs_yaw_integral_rad": float(commanded_yaw_integral),
        "measured_abs_yaw_travel_rad": float(actual_yaw_travel),
        "measured_to_commanded_yaw_travel_ratio": (
            None if commanded_yaw_integral <= 1e-12 else float(actual_yaw_travel / commanded_yaw_integral)
        ),
        "max_goal_observation_error_m": float(max(goal_observation_error)),
        "mean_goal_observation_error_m": float(np.mean(goal_observation_error)),
        "turn_action_alignment_fraction_when_far": (
            float(np.mean(turn_action_aligned)) if turn_action_aligned else None
        ),
        "actual_yaw_delta_alignment_fraction_with_command": (
            float(np.mean([
                math.copysign(1.0, yaw_delta) == math.copysign(1.0, row["command_v_w"][1])
                for row, yaw_delta in actual_turn_samples
            ])) if actual_turn_samples else None
        ),
        "mean_abs_action_forward_when_within_1m": (
            float(np.mean([abs(row["action_4"][0]) for row in rows if row["distance_before_m"] <= 1.0]))
            if any(row["distance_before_m"] <= 1.0 for row in rows) else None
        ),
        "mean_abs_actual_speed_when_within_tolerance_mps": (
            float(np.mean([abs(row["actual_v_w_after"][0]) for row in tolerance_rows]))
            if tolerance_rows else None
        ),
    }


def _run_episode(adapter, model, seed: int) -> dict[str, Any]:
    observation, _reset_info = adapter.reset(seed=seed)
    recurrent_state = None
    episode_start = np.ones(1, dtype=bool)
    rows: list[dict[str, Any]] = []
    cfg = adapter.cfg
    engine = adapter.engine
    for step in range(1, int(math.ceil(cfg.max_time / cfg.dt)) + 1):
        state = np.asarray(observation["state"], dtype=np.float32).reshape(44)
        pose_before = np.asarray(adapter.robot, dtype=np.float64).copy()
        goal_world = np.asarray(adapter.goal, dtype=np.float64).copy()
        goal_delta = goal_world - pose_before[:2]
        goal_world_frame = np.array([
            math.cos(pose_before[2]) * goal_delta[0] + math.sin(pose_before[2]) * goal_delta[1],
            -math.sin(pose_before[2]) * goal_delta[0] + math.cos(pose_before[2]) * goal_delta[1],
        ])
        goal_robot_frame = state[2:4].astype(np.float64) * float(cfg.size)
        goal_error = float(np.linalg.norm(goal_robot_frame - goal_world_frame))
        distance_before = float(np.linalg.norm(goal_delta))
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        recurrent_reset = bool(episode_start[0])
        episode_start = np.zeros(1, dtype=bool)
        action = np.asarray(action, dtype=np.float32).reshape(4)
        observation, reward, terminated, truncated, info = adapter.step(action)
        command = engine.command[0].detach().cpu().numpy().astype(np.float64)
        actual = np.asarray(adapter.robot[3:5], dtype=np.float64).copy()
        pose_after = np.asarray(adapter.robot[:3], dtype=np.float64).copy()
        state_after = np.asarray(observation["state"], dtype=np.float32).reshape(44)
        rows.append({
            "step": step,
            "time_s": float(adapter.time),
            "dt_s": float(cfg.dt),
            "lstm_reset": recurrent_reset,
            "state_44_before": state.astype(float).tolist(),
            "goal_robot_frame_normalized": state[2:4].astype(float).tolist(),
            "goal_robot_frame_m": goal_robot_frame.astype(float).tolist(),
            "goal_bearing_rad": float(math.atan2(goal_robot_frame[1], goal_robot_frame[0])),
            "goal_observation_error_m": goal_error,
            "distance_before_m": distance_before,
            "action_4": action.astype(float).tolist(),
            "desired_v_w": [float(action[0] * cfg.max_speed), float(action[1] * cfg.max_turn_rate)],
            "command_v_w": command.tolist(),
            "actual_v_w_before": pose_before[3:5].astype(float).tolist(),
            "actual_v_w_after": actual.astype(float).tolist(),
            "pose_before_xy_yaw": pose_before[:3].astype(float).tolist(),
            "pose_after_xy_yaw": pose_after.astype(float).tolist(),
            "distance_after_m": float(info.get("distance", np.linalg.norm(goal_world - pose_after[:2]))),
            "reward_raw": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reason": str(info.get("reason", "")),
            "state_44_after": state_after.astype(float).tolist(),
        })
        if terminated or truncated:
            break
    final = rows[-1]
    return {
        "seed": int(seed),
        "goal_world_xy": goal_world.astype(float).tolist(),
        "start_pose_xy_yaw": rows[0]["pose_before_xy_yaw"],
        "initial_goal_robot_frame_m": rows[0]["goal_robot_frame_m"],
        "raw_return": float(sum(row["reward_raw"] for row in rows)),
        "damage": int(info.get("damage", 0)),
        "collisions": int(info.get("collisions", 0)),
        "summary": _episode_summary(rows),
        "steps": rows,
    }


def run(*, device: str = "cuda:0", output: Path = OUTPUT) -> dict[str, Any]:
    from sb3_contrib import RecurrentPPO
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.task import Config

    checkpoint_paths = {"policy_16384": INITIAL, "policy_20480": AFTER}
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        plants=16,
        layout="rows",
        row_spacing=0.8,
        plant_spacing=0.3,
        terrain_enabled=False,
        max_time=8.0,
    )
    engine = make_env(
        cfg=cfg,
        mode="goto",
        num_envs=1,
        seed=80_003,
        auto_reset=False,
        goal_distance=(0.6, 1.2),
        goal_angle=math.radians(30),
        distance_cost=0.0,
    )
    adapter = SingleFieldAdapter(engine)
    results: dict[str, Any] = {
        "schema": "isaac-field-arrival-trace-v1",
        "task": {
            "mode": "goto",
            "plants": cfg.plants,
            "terrain_enabled": False,
            "max_time_s": cfg.max_time,
            "max_steps_per_episode": int(math.ceil(cfg.max_time / cfg.dt)),
            "goal_distance_m": [0.6, 1.2],
            "goal_angle_rad": math.radians(30),
            "distance_cost": 0.0,
            "seeds": list(SEEDS),
            "scene_seed": 80_003,
        },
        "checkpoint_metadata": {},
        "episodes": {},
        "training_kwargs_load_behavior": (
            "The current train_stage checkpoint-load path passes learning_rate, gamma, and target_kl to "
            "RecurrentPPO.load and explicitly synchronizes the loaded model learning-rate schedule, optimizer, "
            "gamma/rollout-buffer gamma, and target_kl. This trace only loads policies for inference; it does not "
            "exercise training-stage overrides."
        ),
        "status": "running",
    }
    report_path = output / "arrival_trace.json"
    try:
        for policy_name, path in checkpoint_paths.items():
            model = RecurrentPPO.load(str(path), device=device)
            results["checkpoint_metadata"][policy_name] = _policy_metadata(model, path)
            results["episodes"][policy_name] = {}
            for seed in SEEDS:
                print(f"Trace {policy_name} seed={seed}", flush=True)
                episode = _run_episode(adapter, model, seed)
                results["episodes"][policy_name][str(seed)] = episode
                results["status"] = f"running:{policy_name}:seed={seed}"
                report_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            del model
        results["status"] = "complete"
    finally:
        adapter.close()
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(report_path),
        "checkpoint_metadata": results["checkpoint_metadata"],
        "episode_summaries": {
            model_name: {seed: episode["summary"] for seed, episode in episodes.items()}
            for model_name, episodes in results["episodes"].items()
        },
    }, ensure_ascii=False, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(device=args.device, output=args.output)


if __name__ == "__main__":
    main()
