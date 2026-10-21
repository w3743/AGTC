#!/usr/bin/env python3
"""Read-only CPU verification and analysis for the saved GAE treatment."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIAG_DIR = ROOT / "results/isaac_field_diagnosis/goal_conditioning_20261017"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DIAG_DIR))

from sb3_contrib import RecurrentPPO
import policy_goal_conditioning_cpu as goal_analysis


CHECKPOINT = HERE / "treatment/field_policy_after.zip"
CONTROL = ROOT / "results/isaac_field_improvement/exploration_gsde_20261017/gaussian/field_policy_after.zip"
PROTOCOL = HERE / "protocol.json"
EVALUATION = HERE / "evaluation_gae_lambda_099.json"
GOAL_OUTPUT = HERE / "gae_policy_goal_conditioning_analysis.json"
DIRECTION_OUTPUT = HERE / "directionality_gae_lambda_099.json"


def checkpoint_metadata() -> dict:
    model = RecurrentPPO.load(str(CHECKPOINT), device="cpu")
    control = RecurrentPPO.load(str(CONTROL), device="cpu")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    hp = protocol["training"]["hyperparameters"]
    extractor = model.policy.features_extractor
    drive = getattr(model, "field_drive_config", None)
    obs = getattr(model, "field_observation_config", None)
    control_drive = getattr(control, "field_drive_config", None)
    control_obs = getattr(control, "field_observation_config", None)
    optimizer_lrs = sorted({float(group["lr"]) for group in model.policy.optimizer.param_groups})
    values = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "num_timesteps": int(model.num_timesteps),
        "n_envs": int(model.n_envs),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "learning_rate_schedule_value": float(model.learning_rate),
        "optimizer_group_lrs": optimizer_lrs,
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "rollout_buffer_gamma": float(model.rollout_buffer.gamma),
        "rollout_buffer_gae_lambda": float(model.rollout_buffer.gae_lambda),
        "use_sde": bool(model.use_sde),
        "goal_gain": float(getattr(extractor, "goal_gain", float("nan"))),
        "drive_config": drive,
        "observation_config": obs,
        "control_checkpoint": str(CONTROL.relative_to(ROOT)),
        "drive_config_matches_gaussian_control": drive == control_drive,
        "observation_config_matches_gaussian_control": obs == control_obs,
        "protocol_parameter_checks": {
            "num_timesteps_matches": int(model.num_timesteps) == 69632,
            "gae_lambda_model_is_0p99": math.isclose(float(model.gae_lambda), 0.99, abs_tol=1e-12),
            "gae_lambda_rollout_buffer_is_0p99": math.isclose(float(model.rollout_buffer.gae_lambda), 0.99, abs_tol=1e-12),
            "gamma_is_0p995": math.isclose(float(model.gamma), 0.995, abs_tol=1e-12),
            "goal_gain_is_100": math.isclose(float(getattr(extractor, "goal_gain", float("nan"))), 100.0),
            "gaussian_not_sde": not bool(model.use_sde),
            "n_envs_is_4": int(model.n_envs) == 4,
            "n_steps_is_128": int(model.n_steps) == 128,
            "batch_size_is_128": int(model.batch_size) == 128,
            "n_epochs_is_5": int(model.n_epochs) == 5,
            "learning_rate_is_1e4": all(math.isclose(value, 1e-4, rel_tol=1e-7, abs_tol=1e-12) for value in optimizer_lrs),
            "drive_metadata_matches_control": drive == control_drive,
            "observation_metadata_matches_control": obs == control_obs,
            "training_result_source_is_not_used_for_parameters": True,
        },
    }
    values["all_protocol_parameter_checks_pass"] = all(values["protocol_parameter_checks"].values())
    return values


def bearing_sign(angle: float, threshold: float = 0.05) -> int:
    return 1 if angle > threshold else -1 if angle < -threshold else 0


def mean(values):
    return None if not values else float(np.mean(values))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def episode_summary(row: dict, tolerance: float, hold_threshold: float, dt: float) -> dict:
    trajectory = row["trajectory"]
    if not trajectory:
        raise ValueError(f"seed {row.get('seed')} has no saved trajectory")
    first = trajectory[0]
    initial_bearing = float(first["before"]["goal_bearing_rad"])
    group_sign = bearing_sign(initial_bearing)
    group = "positive" if group_sign > 0 else "negative" if group_sign < 0 else "near_zero"
    actions = [float(step["action_4"][1]) for step in trajectory]
    first_actions = actions[: min(20, len(actions))]
    track_diffs = [float(step["track_command_lr_mps"][1] - step["track_command_lr_mps"][0]) for step in trajectory]
    first_track_diffs = track_diffs[: min(20, len(track_diffs))]
    yaw_deltas = []
    previous_yaw = float(first["before"]["pose_xy_yaw"][2])
    for step in trajectory:
        yaw = float(step["after"]["pose_xy_yaw"][2])
        yaw_deltas.append(wrap(yaw - previous_yaw))
        previous_yaw = yaw
    first_yaw_deltas = yaw_deltas[: min(20, len(yaw_deltas))]
    bearing_matches_initial = [bearing_sign(action, 1e-3) == group_sign for action in actions]
    instantaneous_matches = [
        bearing_sign(action, 1e-3) == bearing_sign(float(step["before"]["goal_bearing_rad"]))
        for action, step in zip(actions, trajectory)
    ]
    yaw_matches = [
        bearing_sign(delta, 0.0) == bearing_sign(float(step["before"]["goal_bearing_rad"]))
        for delta, step in zip(yaw_deltas, trajectory)
    ]
    samples = [first["before"]] + [step["after"] for step in trajectory]
    distances = [float(sample["goal_distance_m"]) for sample in samples]
    damage = float(row["damage"])
    collisions = float(row["collisions"])
    clean_success = bool(row["success"] and damage == 0.0 and collisions == 0.0)
    return {
        "seed": int(row["seed"]),
        "initial_bearing_rad": initial_bearing,
        "bearing_group": group,
        "success": bool(row["success"]),
        "clean_success": clean_success,
        "damage": damage,
        "damaged": damage > 0.0,
        "collisions": collisions,
        "steps": int(row["steps"]),
        "mean_diff_action_first_1s": mean(first_actions),
        "mean_diff_action_full": mean(actions),
        "positive_diff_action_steps": sum(value > 1e-3 for value in actions),
        "negative_diff_action_steps": sum(value < -1e-3 for value in actions),
        "mean_diff_sign_match_initial_bearing_fraction": mean([float(value) for value in bearing_matches_initial]),
        "mean_diff_sign_match_instantaneous_bearing_fraction": mean([float(value) for value in instantaneous_matches]),
        "mean_track_right_minus_left_first_1s_mps": mean(first_track_diffs),
        "mean_track_right_minus_left_full_mps": mean(track_diffs),
        "mean_yaw_increment_first_1s_rad": float(np.sum(first_yaw_deltas)),
        "mean_yaw_increment_full_rad": float(np.sum(yaw_deltas)),
        "mean_abs_yaw_increment_full_rad": abs(float(np.sum(yaw_deltas))),
        "mean_yaw_sign_match_instantaneous_bearing_fraction": mean([float(value) for value in yaw_matches]),
        "min_distance_m": float(min(distances)),
        "final_distance_m": float(distances[-1]),
        "entered_0p4m_tolerance": bool(any(value <= tolerance for value in distances)),
        "reached_hold": bool(max(float(step["after"]["hold_s"]) for step in trajectory) >= hold_threshold),
    }


def summarize(episodes: list[dict]) -> dict:
    def aggregate(group: list[dict]) -> dict:
        keys = (
            "initial_bearing_rad", "mean_diff_action_first_1s", "mean_diff_action_full",
            "mean_diff_sign_match_initial_bearing_fraction", "mean_diff_sign_match_instantaneous_bearing_fraction",
            "mean_track_right_minus_left_first_1s_mps", "mean_track_right_minus_left_full_mps",
            "mean_yaw_increment_first_1s_rad", "mean_yaw_increment_full_rad",
            "mean_abs_yaw_increment_full_rad", "mean_yaw_sign_match_instantaneous_bearing_fraction",
            "min_distance_m", "final_distance_m",
        )
        out = {"episodes": len(group)}
        for key in keys:
            out["mean_" + key.removeprefix("mean_")] = mean([row[key] for row in group])
        out.update({
            "clean_successes": sum(row["clean_success"] for row in group),
            "successes": sum(row["success"] for row in group),
            "damaged_episodes": sum(row["damaged"] for row in group),
            "mean_damage": mean([row["damage"] for row in group]),
            "collisions": sum(row["collisions"] for row in group),
            "entered_0p4m_tolerance": sum(row["entered_0p4m_tolerance"] for row in group),
            "reached_hold": sum(row["reached_hold"] for row in group),
            "positive_diff_action_steps": sum(row["positive_diff_action_steps"] for row in group),
            "negative_diff_action_steps": sum(row["negative_diff_action_steps"] for row in group),
        })
        return out

    groups = {
        label: [row for row in episodes if row["bearing_group"] == label]
        for label in ("positive", "negative", "near_zero")
    }
    return {
        "overall": aggregate(episodes),
        "by_initial_bearing": {label: aggregate(rows) for label, rows in groups.items()},
    }


def main() -> None:
    if GOAL_OUTPUT.exists() or DIRECTION_OUTPUT.exists():
        raise FileExistsError("refusing to overwrite existing analysis output")
    import torch
    torch.set_num_threads(1)
    metadata = checkpoint_metadata()
    if not metadata["all_protocol_parameter_checks_pass"]:
        raise RuntimeError("checkpoint metadata does not match the requested training conditions")

    goal_analysis.CHECKPOINT_STEPS["gae_lambda_099"] = 69632
    obs, capture_manifest = goal_analysis.read_observations()
    goal_result = goal_analysis.run_model("gae_lambda_099", CHECKPOINT, obs)
    goal_report = {
        "schema": "isaac-field-goal-conditioning-cpu-analysis-v1",
        "checkpoint_metadata_verified_from_saved_model": metadata,
        "native_capture": str(goal_analysis.INPUT.relative_to(ROOT)),
        "capture_manifest": str(goal_analysis.MANIFEST.relative_to(ROOT)),
        "capture_seed": capture_manifest["environment"]["episode_reset_seed"],
        "angles_deg": np.asarray(obs["angles_deg"]).tolist(),
        "observation_invariance": {
            name: capture_manifest["non_goal_observation_invariance"][name]["max_abs_diff_from_angle_-90"]
            for name in ("lidar", "depth", "detections", "state")
        },
        "model": goal_result,
        "conclusion": "The recorded goal input reaches the policy; this counterfactual does not establish closed-loop ability.",
    }
    GOAL_OUTPUT.write_text(json.dumps(goal_report, indent=2) + "\n", encoding="utf-8")

    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if len(evaluation["rows"]) != 24 or not evaluation.get("trajectory_schema"):
        raise RuntimeError("treatment heldout data is incomplete")
    tolerance = float(protocol["config"]["goal_tolerance"])
    hold = float(protocol["config"]["goal_hold"])
    dt = float(protocol["config"]["dt"])
    episodes = [episode_summary(row, tolerance, hold, dt) for row in evaluation["rows"]]
    gl = float(metadata["gamma"] * metadata["gae_lambda"])
    directionality = {
        "schema": "isaac-field-gae-directionality-cpu-v1",
        "source": "evaluation_gae_lambda_099.json",
        "method": "CPU-only aggregate of the saved compact trajectories; no simulation or model inference.",
        "trajectory_timestep_s": dt,
        "first_second_steps": 20,
        "bearing_group_threshold_rad": 0.05,
        "clean_success_definition": "success and damage == 0 and collisions == 0",
        "checkpoint_metadata": {
            "num_timesteps": metadata["num_timesteps"],
            "gamma": metadata["gamma"],
            "gae_lambda": metadata["gae_lambda"],
            "rollout_buffer_gae_lambda": metadata["rollout_buffer_gae_lambda"],
            "goal_gain": metadata["goal_gain"],
        },
        "learning_signal_context": {
            "gamma_lambda": gl,
            "residual_weight_e_fold_steps": -1.0 / math.log(gl),
            "residual_weight_e_fold_seconds": -dt / math.log(gl),
            "residual_weight_half_life_steps": math.log(0.5) / math.log(gl),
            "residual_weight_half_life_seconds": dt * math.log(0.5) / math.log(gl),
            "interpretation": "This is a characteristic GAE residual weighting scale, not a hard credit horizon; value bootstrapping can carry information farther.",
        },
        "evaluation_summary": evaluation["summary"],
        **summarize(episodes),
        "episodes": episodes,
    }
    DIRECTION_OUTPUT.write_text(json.dumps(directionality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "metadata": metadata,
        "goal_output": str(GOAL_OUTPUT),
        "differential_action_by_goal_angle": goal_result["step1_differential_action_by_angle"],
        "jacobian_diff_action_by_goal_channel": goal_result["jacobian_raw_state_goal_xy_to_distribution_mean"]["differential_action_row_by_goal_channel"],
        "direction_output": str(DIRECTION_OUTPUT),
        "evaluation_summary": evaluation["summary"],
        "directionality_summary": directionality["by_initial_bearing"],
    }, indent=2))


if __name__ == "__main__":
    main()
