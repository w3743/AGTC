#!/usr/bin/env python3
"""Read-only diagnosis of the four approved minimal-input trial evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_minimal_inputs_20261020"
OUT = Path(__file__).resolve().parent
EVALUATIONS = {
    "field_baseline": "evaluation_field_baseline_82001_82004.json",
    "field_final": "evaluation_field_final_82001_82004.json",
    "free_baseline": "evaluation_free_baseline_87001_87004.json",
    "free_final": "evaluation_free_final_87001_87004.json",
}
EXPECTED_SEEDS = {
    "field_baseline": {82001, 82002, 82003, 82004},
    "field_final": {82001, 82002, 82003, 82004},
    "free_baseline": {87001, 87002, 87003, 87004},
    "free_final": {87001, 87002, 87003, 87004},
}


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def trajectory_stats(row: dict, max_track_speed: float) -> dict:
    trajectory = row["trajectory"]
    if len(trajectory) != int(row["steps"]):
        raise ValueError(f"seed {row['seed']}: trajectory/step count mismatch")

    first, last = trajectory[0], trajectory[-1]
    distances = [float(trajectory[0]["before"]["goal_distance_m"])]
    distances.extend(float(step["after"]["goal_distance_m"]) for step in trajectory)
    distance_steps = [
        float(step["after"]["goal_distance_m"] - step["before"]["goal_distance_m"])
        for step in trajectory
    ]
    actions = [step["action_4"] for step in trajectory]
    commands = [step["command_v_w"] for step in trajectory]
    track_commands = [step["track_command_lr_mps"] for step in trajectory]
    actual = [step["after"]["actual_v_w"] for step in trajectory]
    poses = [first["before"]["pose_xy_yaw"]]
    poses.extend(step["after"]["pose_xy_yaw"] for step in trajectory)

    path_length = sum(
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(poses, poses[1:])
    )
    net_displacement = math.hypot(
        float(poses[-1][0]) - float(poses[0][0]),
        float(poses[-1][1]) - float(poses[0][1]),
    )
    yaw_changes = [
        wrap_angle(float(b[2]) - float(a[2]))
        for a, b in zip(poses, poses[1:])
    ]
    cumulative_abs_yaw = sum(abs(value) for value in yaw_changes)
    signed_yaw = sum(yaw_changes)
    distance_min = min(distances)
    distance_min_index = distances.index(distance_min)
    final_distance = float(row["final_distance"])
    tail_from_min = distances[distance_min_index:]
    positive_v_steps = [
        index for index, command in enumerate(commands) if float(command[0]) > 0.02
    ]
    negative_v_steps = [
        index for index, command in enumerate(commands) if float(command[0]) < -0.02
    ]
    positive_v_away = [index for index in positive_v_steps if distance_steps[index] > 1e-5]
    positive_v_toward = [index for index in positive_v_steps if distance_steps[index] < -1e-5]
    saturated_action_steps = [
        index for index, action in enumerate(actions)
        if any(abs(float(value)) >= 0.95 for value in action)
    ]
    saturated_track_steps = [
        index for index, track in enumerate(track_commands)
        if any(abs(float(value)) >= 0.99 * max_track_speed for value in track)
    ]
    near_stationary_steps = [
        index for index, velocity in enumerate(actual)
        if abs(float(velocity[0])) < 0.01 and abs(float(velocity[1])) < 0.01
    ]
    all_abs_action = [abs(float(value)) for action in actions for value in action]
    command_v = [float(value[0]) for value in commands]
    command_w = [float(value[1]) for value in commands]
    actual_v = [float(value[0]) for value in actual]
    actual_w = [float(value[1]) for value in actual]
    abs_track_commands = [abs(float(value)) for pair in track_commands for value in pair]
    yaw_command_actual_ratio = (
        mean(abs(value) for value in command_w) / mean(abs(value) for value in actual_w)
        if mean(abs(value) for value in actual_w) > 1e-12 else None
    )

    return {
        "seed": int(row["seed"]),
        "steps": int(row["steps"]),
        "duration_s": float(last["after"]["time_s"]),
        "success": bool(row["success"]),
        "raw_return": float(row["raw_return"]),
        "damage": float(row["damage"]),
        "collisions": int(row["collisions"]),
        "timed_out": bool(row["timed_out"]),
        "reason": row["reason"],
        "start_distance_m": distances[0],
        "final_distance_m": final_distance,
        "min_distance_m": distance_min,
        "min_distance_step": distance_min_index,
        "distance_reduction_m": distances[0] - final_distance,
        "post_min_rebound_m": final_distance - distance_min,
        "post_min_max_rebound_m": max(tail_from_min) - distance_min,
        "distance_increase_step_fraction": sum(delta > 1e-5 for delta in distance_steps) / len(distance_steps),
        "distance_decrease_step_fraction": sum(delta < -1e-5 for delta in distance_steps) / len(distance_steps),
        "positive_v_step_fraction": len(positive_v_steps) / len(commands),
        "negative_v_step_fraction": len(negative_v_steps) / len(commands),
        "positive_v_away_fraction": (len(positive_v_away) / len(positive_v_steps)) if positive_v_steps else None,
        "positive_v_toward_fraction": (len(positive_v_toward) / len(positive_v_steps)) if positive_v_steps else None,
        "mean_command_v_mps": mean(command_v),
        "max_abs_command_v_mps": max(abs(value) for value in command_v),
        "mean_command_abs_yaw_rate_rad_s": mean(abs(value) for value in command_w),
        "max_abs_command_yaw_rate_rad_s": max(abs(value) for value in command_w),
        "command_to_actual_abs_yaw_rate_ratio": yaw_command_actual_ratio,
        "max_abs_track_command_mps": max(abs_track_commands),
        "mean_abs_left_track_command_mps": mean(abs(float(value[0])) for value in track_commands),
        "mean_abs_right_track_command_mps": mean(abs(float(value[1])) for value in track_commands),
        "mean_actual_v_mps": mean(actual_v),
        "max_abs_actual_v_mps": max(abs(value) for value in actual_v),
        "mean_actual_abs_yaw_rate_rad_s": mean(abs(value) for value in actual_w),
        "near_stationary_step_fraction": len(near_stationary_steps) / len(actual),
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "path_to_net_ratio": (path_length / net_displacement) if net_displacement > 1e-9 else None,
        "net_yaw_change_rad_wrapped": wrap_angle(signed_yaw),
        "cumulative_abs_yaw_change_rad": cumulative_abs_yaw,
        "cumulative_yaw_turns_equivalent": cumulative_abs_yaw / (2.0 * math.pi),
        "max_abs_action_component": max(all_abs_action),
        "action_component_max_abs": [
            max(abs(float(action[index])) for action in actions) for index in range(4)
        ],
        "action_saturated_steps_fraction_abs_ge_0_95": len(saturated_action_steps) / len(actions),
        "track_command_saturated_steps_fraction": len(saturated_track_steps) / len(track_commands),
        "first_command_v_w": first["command_v_w"],
        "first_action_4": first["action_4"],
        "last_command_v_w": last["command_v_w"],
        "last_action_4": last["action_4"],
        "initial_goal_bearing_rad": float(first["before"]["goal_bearing_rad"]),
        "final_goal_bearing_rad": float(last["after"]["goal_bearing_rad"]),
        "first_goal_frame_xy_m": first["before"]["goal_robot_frame_m"],
        "last_goal_frame_xy_m": last["after"]["goal_robot_frame_m"],
        "snapshots": [
            {
                "step": int(trajectory[index]["step"]),
                "time_s": float(trajectory[index]["time_s"]),
                "goal_distance_m": float(trajectory[index]["after"]["goal_distance_m"]),
                "goal_bearing_rad": float(trajectory[index]["after"]["goal_bearing_rad"]),
                "command_v_w": trajectory[index]["command_v_w"],
                "track_command_lr_mps": trajectory[index]["track_command_lr_mps"],
                "actual_v_w": trajectory[index]["after"]["actual_v_w"],
                "pose_xy_yaw": trajectory[index]["after"]["pose_xy_yaw"],
                "action_4": trajectory[index]["action_4"],
            }
            for index in (0, 20, 40, 80, 160, 240, 320, 399)
        ],
    }


def summarize_eval(path: Path, name: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = {int(row["seed"]) for row in data["rows"]}
    if seeds != EXPECTED_SEEDS[name]:
        raise ValueError(f"{name}: unexpected seed set {sorted(seeds)}")
    max_speed = float(data["drive_config"]["max_track_speed_mps"])
    return {
        "schema": data["schema"],
        "checkpoint": data["checkpoint"],
        "mode": data["mode"],
        "goal_distance_m": data["goal_distance"],
        "summary": data["summary"],
        "rows": [trajectory_stats(row, max_speed) for row in data["rows"]],
    }


def summarize_training() -> dict:
    stages = {}
    for stage_name in ("stage_1_near", "stage_2_steer30", "stage_3_steer60", "stage_4_turn90"):
        data = json.loads((TRIAL / stage_name / "progress.json").read_text(encoding="utf-8"))
        stages[stage_name] = {
            "stage": data["stage"],
            "start_num_timesteps": data["start_num_timesteps"],
            "end_num_timesteps": data["end_num_timesteps"],
            "aggregate_rows": data["aggregate_rows"],
            "last_completed_episodes": data["last_completed_episodes"],
        }
    log_metrics = []
    for line in (TRIAL / "launch_stdout.log").read_text(encoding="utf-8", errors="replace").splitlines():
        if "recent_success=" in line and "recent_mean_damage=" in line:
            prefix, rest = line.split("timesteps=", 1)
            timestep = int(rest.split()[0])
            success_part = rest.split("recent_success=", 1)[1].split()[0]
            success_count, episode_count = (int(part) for part in success_part.split("/"))
            damage = float(rest.split("recent_mean_damage=", 1)[1].split()[0])
            log_metrics.append({
                "stage_label": prefix.split("]", 1)[0].lstrip("["),
                "timesteps": timestep,
                "recent_successes": success_count,
                "recent_episode_count": episode_count,
                "recent_success_rate": success_count / episode_count,
                "recent_mean_damage": damage,
            })
    return {"stages": stages, "stdout_recent_episode_metrics": log_metrics}


def verify_pairing() -> dict:
    pairs = {}
    for pair_name, baseline_name, final_name in (
        ("field", "field_baseline", "field_final"),
        ("free", "free_baseline", "free_final"),
    ):
        baseline = json.loads((TRIAL / EVALUATIONS[baseline_name]).read_text(encoding="utf-8"))
        final = json.loads((TRIAL / EVALUATIONS[final_name]).read_text(encoding="utf-8"))
        pair_rows = {}
        for seed in sorted(EXPECTED_SEEDS[baseline_name]):
            left = next(row for row in baseline["rows"] if int(row["seed"]) == seed)
            right = next(row for row in final["rows"] if int(row["seed"]) == seed)
            initial_equal = left["reset_initial_condition"] == right["reset_initial_condition"]
            goal_equal = left["goal"] == right["goal"]
            pair_rows[str(seed)] = {
                "initial_condition_equal": initial_equal,
                "goal_equal": goal_equal,
                "baseline_trajectory_matches_reset": bool(left["trajectory_start_matches_reset"]),
                "final_trajectory_matches_reset": bool(right["trajectory_start_matches_reset"]),
                "plant_count": int(left["reset_initial_condition"]["plant_count"]),
                "plant_layout_sha_equal": (
                    left["reset_initial_condition"]["plant_xy_sha256"]
                    == right["reset_initial_condition"]["plant_xy_sha256"]
                ),
            }
            if not all((
                pair_rows[str(seed)]["initial_condition_equal"],
                pair_rows[str(seed)]["goal_equal"],
                pair_rows[str(seed)]["baseline_trajectory_matches_reset"],
                pair_rows[str(seed)]["final_trajectory_matches_reset"],
                pair_rows[str(seed)]["plant_layout_sha_equal"],
            )):
                raise ValueError(f"{pair_name} seed {seed}: baseline/final pairing mismatch")
        pair_metadata = {
            "action_config_equal": baseline["action_config"] == final["action_config"],
            "drive_config_equal": baseline["drive_config"] == final["drive_config"],
            "damage_config_equal": baseline["damage_config"] == final["damage_config"],
            "observation_config_equal": baseline["observation_config"] == final["observation_config"],
            "damage_metric_equal": baseline["damage_metric"] == final["damage_metric"],
            "rows": pair_rows,
        }
        if not all(pair_metadata[key] for key in (
            "action_config_equal", "drive_config_equal", "damage_config_equal",
            "observation_config_equal", "damage_metric_equal",
        )):
            raise ValueError(f"{pair_name}: baseline/final contract mismatch")
        pairs[pair_name] = pair_metadata
    return pairs


def main() -> None:
    evaluations = {
        name: summarize_eval(TRIAL / filename, name)
        for name, filename in EVALUATIONS.items()
    }
    training = summarize_training()
    report = {
        "schema": "minimal-inputs-trial-diagnostic-v1",
        "scope": {
            "trial": str(TRIAL),
            "evaluation_files": list(EVALUATIONS.values()),
            "seeds": sorted(EXPECTED_SEEDS["field_baseline"] | EXPECTED_SEEDS["free_baseline"]),
            "read_only": True,
            "no_new_evaluation": True,
        },
        "pairing_checks": verify_pairing(),
        "evaluations": evaluations,
        "training_recent_episodes": training,
    }
    json_path = OUT / "diagnostic_metrics.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json_path)
    for name, evaluation in evaluations.items():
        print(f"{name}: {json.dumps(evaluation['summary'], ensure_ascii=False, sort_keys=True)}")
        for row in evaluation["rows"]:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    for stage, data in training["stages"].items():
        print(f"{stage} progress:")
        for row in data["aggregate_rows"]:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        print("last10:", json.dumps(data["last_completed_episodes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
