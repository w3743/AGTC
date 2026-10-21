"""Read-only CPU audit of the saved 2026-10-17 exploration trajectories."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "results/isaac_field_improvement/exploration_gsde_20261017"
STAGED = PROJECT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable"
AUDIT_DIR = PROJECT / "results/isaac_field_diagnosis"
BASE_OUTPUT = AUDIT_DIR / "reward_motion_audit_20261017.json"
MODELS = ("baseline", "gaussian", "gsde")


def wrap_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def rounded(value: float | None) -> float | None:
    return round(value, 7) if value is not None else None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": rounded(mean(values)),
        "median": rounded(median(values)),
        "min": rounded(min(values)) if values else None,
        "max": rounded(max(values)) if values else None,
    }


def classify(row: dict[str, Any]) -> str:
    if int(row["damage"]) > 0:
        return "damaged"
    if bool(row["success"]) and int(row["collisions"]) == 0:
        return "clean_success"
    if bool(row["timed_out"]) and int(row["collisions"]) == 0:
        return "timeout_intact"
    return "other"


def analyze_episode(row: dict[str, Any], dt: float, progress_budget: float,
                    goal_tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = row["trajectory"]
    initial = trajectory[0]["before"]
    states = [initial] + [sample["after"] for sample in trajectory]
    first_sec_samples = [sample for sample in trajectory
                         if float(sample["time_s"]) <= 1.0 + 1e-8]
    first_sec_state = min(states, key=lambda state: abs(float(state["time_s"]) - 1.0))
    if abs(float(first_sec_state["time_s"]) - 1.0) > dt / 2 + 1e-8:
        first_sec_state = None

    poses = [state["pose_xy_yaw"] for state in states]
    unwrapped_yaw = [float(poses[0][2])]
    unwrapped_bearing = [float(states[0]["goal_bearing_rad"])]
    for previous, current, prev_state, curr_state in zip(
            poses, poses[1:], states, states[1:]):
        unwrapped_yaw.append(unwrapped_yaw[-1] + wrap_delta(float(current[2]), float(previous[2])))
        unwrapped_bearing.append(
            unwrapped_bearing[-1] + wrap_delta(
                float(curr_state["goal_bearing_rad"]), float(prev_state["goal_bearing_rad"])))

    d0 = float(initial["goal_distance_m"])
    progress_scale = float(progress_budget) / max(d0, float(goal_tolerance))
    d1 = float(first_sec_state["goal_distance_m"]) if first_sec_state else None
    df = float(states[-1]["goal_distance_m"])
    xys = [(float(pose[0]), float(pose[1])) for pose in poses]
    path_length = sum(math.dist(a, b) for a, b in zip(xys, xys[1:]))
    net_displacement = math.dist(xys[0], xys[-1])
    net_yaw = unwrapped_yaw[-1] - unwrapped_yaw[0]
    total_abs_yaw = sum(abs(b - a) for a, b in zip(unwrapped_yaw, unwrapped_yaw[1:]))
    net_bearing = unwrapped_bearing[-1] - unwrapped_bearing[0]

    first_actions = [float(sample["action_4"][0]) for sample in first_sec_samples]
    all_actions = [float(sample["action_4"][0]) for sample in trajectory]
    first_commands = [float(sample["command_v_w"][0]) for sample in first_sec_samples]
    all_commands = [float(sample["command_v_w"][0]) for sample in trajectory]
    first_actual_v = [float(sample["after"]["actual_v_w"][0]) for sample in first_sec_samples]
    all_actual_v = [float(sample["after"]["actual_v_w"][0]) for sample in trajectory]
    first_progress = d0 - d1 if d1 is not None else None

    metrics = {
        "forward_action_normalized_first_1s_mean": mean(first_actions),
        "forward_action_normalized_episode_mean": mean(all_actions),
        "command_forward_mps_first_1s_mean": mean(first_commands),
        "command_forward_mps_episode_mean": mean(all_commands),
        "actual_v_first_1s_mean_mps": mean(first_actual_v),
        "actual_v_episode_mean_mps": mean(all_actual_v),
        "initial_goal_distance_m": d0,
        "goal_progress_first_1s_m": first_progress,
        "goal_progress_episode_m": d0 - df,
        "progress_reward_first_1s": progress_scale * first_progress if first_progress is not None else None,
        "progress_reward_episode": progress_scale * (d0 - df),
        "progress_scale_per_m": progress_scale,
        "final_goal_distance_m": df,
        "net_xy_displacement_m": net_displacement,
        "xy_path_length_m": path_length,
        "net_yaw_change_rad": net_yaw,
        "abs_net_yaw_change_rad": abs(net_yaw),
        "cumulative_abs_yaw_change_rad": total_abs_yaw,
        "net_goal_bearing_change_rad": net_bearing,
        "cumulative_abs_goal_bearing_change_rad": sum(
            abs(b - a) for a, b in zip(unwrapped_bearing, unwrapped_bearing[1:])),
        "first_1s_sample_count": len(first_sec_samples),
        "steps": len(trajectory),
    }
    summary = {
        "seed": int(row["seed"]),
        "category": classify(row),
        "success": bool(row["success"]),
        "timed_out": bool(row["timed_out"]),
        "reason": row["reason"],
        "raw_return": float(row["raw_return"]),
        "damage": int(row["damage"]),
        "collisions": int(row["collisions"]),
        **metrics,
    }
    internals = {
        "seed": int(row["seed"]),
        "trajectory": trajectory,
        "states": states,
        "unwrapped_yaw": unwrapped_yaw,
        "unwrapped_bearing": unwrapped_bearing,
        "initial_distance": d0,
        "progress_scale": progress_scale,
    }
    return summary, internals


def turn_windows(internals: list[dict[str, Any]], window_steps: int,
                 yaw_threshold: float, displacement_threshold: float,
                 bearing_threshold: float, budget: float) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    max_abs_yaw = 0.0
    max_abs_bearing = 0.0
    min_displacement = math.inf
    window_count = 0
    for episode in internals:
        states = episode["states"]
        trajectory = episode["trajectory"]
        yaw = episode["unwrapped_yaw"]
        bearing = episode["unwrapped_bearing"]
        scale = episode["progress_scale"]
        for start in range(max(0, len(trajectory) - window_steps + 1)):
            end = start + window_steps
            start_state, end_state = states[start], states[end]
            p0, p1 = start_state["pose_xy_yaw"], end_state["pose_xy_yaw"]
            displacement = math.dist(p0[:2], p1[:2])
            delta_yaw = yaw[end] - yaw[start]
            delta_bearing = bearing[end] - bearing[start]
            max_abs_yaw = max(max_abs_yaw, abs(delta_yaw))
            max_abs_bearing = max(max_abs_bearing, abs(delta_bearing))
            min_displacement = min(min_displacement, displacement)
            window_count += 1
            if (abs(delta_yaw) < yaw_threshold
                    or displacement > displacement_threshold
                    or abs(delta_bearing) < bearing_threshold):
                continue
            segment = trajectory[start:end]
            d_start = float(start_state["goal_distance_m"])
            d_end = float(end_state["goal_distance_m"])
            found.append({
                "seed": int(episode["seed"]),
                "start_time_s": float(segment[0]["before"]["time_s"]),
                "end_time_s": float(segment[-1]["after"]["time_s"]),
                "yaw_change_rad": delta_yaw,
                "goal_bearing_change_rad": delta_bearing,
                "net_xy_displacement_m": displacement,
                "goal_distance_progress_m": d_start - d_end,
                "progress_reward": scale * (d_start - d_end),
                "raw_reward_sum": sum(float(step["reward_raw"]) for step in segment),
                "damage_delta": int(end_state["damage"]) - int(start_state["damage"]),
                "collision_delta": int(end_state["collisions"]) - int(start_state["collisions"]),
            })
    return {
        "window_steps": window_steps,
        "window_seconds": window_steps * 0.05,
        "sliding_window_count": window_count,
        "criteria": {
            "absolute_yaw_change_at_least_rad": yaw_threshold,
            "net_xy_displacement_at_most_m": displacement_threshold,
            "absolute_goal_bearing_change_at_least_rad": bearing_threshold,
        },
        "qualifying_window_count": len(found),
        "max_absolute_yaw_change_in_any_window_rad": max_abs_yaw,
        "max_absolute_goal_bearing_change_in_any_window_rad": max_abs_bearing,
        "minimum_net_xy_displacement_in_any_window_m": (
            min_displacement if math.isfinite(min_displacement) else None),
        "qualifying_windows": found,
        "interpretation": (
            "No qualifying segments; do not estimate local turn reward from these trajectories. "
            "Use episode-level displacement and yaw summaries instead."
            if not found else
            "Qualifying windows are overlapping descriptive samples, not independent episodes."
        ),
    }


def stage_metadata() -> dict[str, Any]:
    report_path = STAGED / "stages_report.json"
    log_path = STAGED / "run.log"
    if not report_path.exists():
        return {"available": False, "exact_completed_episode_total": None}
    stages = json.loads(report_path.read_text(encoding="utf-8"))
    progress = {}
    final_tail_sizes = {}
    for stage in stages:
        name = stage["name"]
        progress[name] = stage.get("aggregate_progress", [])
        final_tail_sizes[name] = len(stage.get("last_completed_episodes", []))
    return {
        "available": True,
        "source_files": [str(report_path.relative_to(PROJECT)), str(log_path.relative_to(PROJECT))],
        "training_timesteps_by_stage": {
            stage["name"]: stage.get("trained_timesteps") for stage in stages
        },
        "progress_snapshot_recent_episode_counts": {
            name: [int(snapshot["recent_episode_count"]) for snapshot in snapshots]
            for name, snapshots in progress.items()
        },
        "last_completed_episodes_tail_sizes_by_stage": final_tail_sizes,
        "exact_completed_episode_total": None,
        "reason_exact_total_unavailable": (
            "The log and stages_report store overlapping recent-episode windows and only a final "
            "tail list, not a complete episode ledger; summing rolling counts would double-count."
        ),
    }


def main() -> None:
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    task_cfg = protocol["config"]
    eval_cfg = protocol["evaluation"]
    dt = float(task_cfg["dt"])
    budget = float(task_cfg["goal_progress_budget"])
    tolerance = float(task_cfg["goal_tolerance"])
    episode_aggregates: dict[str, Any] = {}
    internals_by_model: dict[str, list[dict[str, Any]]] = {}

    for model in MODELS:
        source_path = RUN / f"evaluation_{model}.json"
        evaluation = json.loads(source_path.read_text(encoding="utf-8"))
        rows = evaluation["rows"]
        episodes, internals = zip(*(analyze_episode(row, dt, budget, tolerance) for row in rows))
        episode_rows = list(episodes)
        internals_by_model[model] = list(internals)
        categories: dict[str, Any] = {}
        for category in ("clean_success", "timeout_intact", "damaged", "other"):
            group = [episode for episode in episode_rows if episode["category"] == category]
            categories[category] = {
                "episodes": len(group),
                "mean_raw_return": rounded(mean([e["raw_return"] for e in group])),
                "median_raw_return": rounded(median([e["raw_return"] for e in group])),
                "mean_damage_per_episode": rounded(mean([float(e["damage"]) for e in group])),
                "total_collisions": sum(int(e["collisions"]) for e in group),
            }
        metric_names = (
            "forward_action_normalized_first_1s_mean",
            "forward_action_normalized_episode_mean",
            "command_forward_mps_first_1s_mean",
            "command_forward_mps_episode_mean",
            "actual_v_first_1s_mean_mps",
            "actual_v_episode_mean_mps",
            "goal_progress_first_1s_m",
            "goal_progress_episode_m",
            "progress_reward_first_1s",
            "progress_reward_episode",
            "net_xy_displacement_m",
            "xy_path_length_m",
            "net_yaw_change_rad",
            "abs_net_yaw_change_rad",
            "cumulative_abs_yaw_change_rad",
            "net_goal_bearing_change_rad",
            "cumulative_abs_goal_bearing_change_rad",
        )
        per_metric = {
            name: summarize([float(e[name]) for e in episode_rows if e[name] is not None])
            for name in metric_names
        }
        episode_aggregates[model] = {
            "source_file": str(source_path.relative_to(PROJECT)),
            "episodes": len(rows),
            "summary_mean_raw_return": float(evaluation["summary"]["mean_raw_return"]),
            "outcome_categories": categories,
            "macro_mean_metrics_per_episode": per_metric,
            "episodes_detail": episode_rows,
            "turn_without_translation": turn_windows(
                list(internals), window_steps=40, yaw_threshold=0.25,
                displacement_threshold=0.10, bearing_threshold=0.15, budget=budget),
        }

    baseline_rows = json.loads((RUN / "evaluation_baseline.json").read_text(encoding="utf-8"))["rows"]
    unique_initial_distances = []
    for row in baseline_rows:
        unique_initial_distances.append(float(row["trajectory"][0]["before"]["goal_distance_m"]))

    result = {
        "schema": "isaac-field-reward-motion-audit-v1",
        "scope": "CPU-only analysis of saved compact evaluation trajectories and existing training metadata; no simulator, training, FPS test, or source/protocol/result mutation.",
        "inputs": {
            "protocol": str((RUN / "protocol.json").relative_to(PROJECT)),
            "evaluation_files": [str((RUN / f"evaluation_{m}.json").relative_to(PROJECT)) for m in MODELS],
            "training_metadata": [
                str((STAGED / "stages_report.json").relative_to(PROJECT)),
                str((STAGED / "run.log").relative_to(PROJECT)),
            ],
            "paired_evaluation_seeds": eval_cfg["seeds"],
            "pairing_valid": json.loads((RUN / "run_status.json").read_text(encoding="utf-8"))["pairing_valid"],
        },
        "analysis_definitions": {
            "episode_metrics": "Macro-average per-episode values, so successful short episodes have the same weight as 20 s timeouts.",
            "forward_action": "action_4[0], the normalized policy forward channel; command_v_w[0] is also reported in m/s.",
            "actual_forward_speed": "Saved after.actual_v_w[0] series; first-second mean uses all transitions ending at t <= 1.0 s, full mean uses all transitions.",
            "distance_progress": "Initial goal_distance_m minus the nearest saved distance at t=1.0 s or the final saved distance; positive means closer.",
            "outcome_categories": {
                "clean_success": "success=true, damage=0, collisions=0",
                "timeout_intact": "timed_out=true, damage=0, collisions=0",
                "damaged": "damage>0, regardless of success/timeout",
                "other": "remaining cases, including collision without plant damage",
            },
            "turn_window": "Overlapping 2.0 s windows; clear turn requires |unwrapped yaw change| >= 0.25 rad, net XY displacement <= 0.10 m, and |cumulative goal-bearing change| >= 0.15 rad. Reward is summed saved reward_raw over the same window. If no windows qualify, no local turn-reward inference is made.",
        },
        "run_context": {
            "source_checkpoint_timesteps": protocol["source_checkpoint"]["num_timesteps"],
            "timesteps_trained_each_variant": protocol["training"]["timesteps_per_variant"],
            "goal_distance_range_m": eval_cfg["goal_distance_m"],
            "episode_limit_s": task_cfg["max_time"],
            "dt_s": dt,
            "damage_penalty": task_cfg["damage_penalty"],
            "goal_progress_budget": budget,
            "progress_weight_legacy_fallback_not_active": task_cfg["progress_weight"],
            "goal_tolerance_m": tolerance,
            "distance_cost_evaluation": eval_cfg["distance_cost"],
            "initial_goal_distance_unique_seeds_m": summarize(unique_initial_distances),
            "progress_scale_budget_over_initial_distance": summarize(
                [budget / max(distance, tolerance) for distance in unique_initial_distances]),
            "reward_formula_context_from_frozen_task_source": {
                "per_step_time_cost": -float(task_cfg["time_penalty"]) * dt,
                "damage_event_cost": -float(task_cfg["damage_penalty"]),
                "distance_progress_term": "goal_progress_budget / max(initial_goal_distance, goal_tolerance) * (old_distance - new_distance)",
                "goal_success_bonus": 5.0,
                "note": "Raw reward also includes collision/boundary terms; compact trajectories do not expose a boundary flag.",
            },
        },
        "models": episode_aggregates,
        "staged_training_episode_count": stage_metadata(),
        "interpretation": [
            "Per-model forward action, commanded forward speed, measured actual forward speed, and goal-distance progress are reported at one second and over each full episode; use these saved signals rather than inferring motion from reward alone.",
            "The 2 s strict turn-with-negligible-translation test may be uninformative when policies do not execute such turns; its zero/low sample count is a limitation, not evidence that turn reward is absent.",
            "The evaluation setup combines 1-3 m targets, a 20 s horizon, a 5-unit per-goal distance-progress budget, a +5 terminal success bonus, and a 20-unit damage penalty. A damage event is four times the full nominal progress budget, while immediate progress is scaled by initial distance. This can plausibly make safe early motion/turn exploration difficult, but does not establish reward design as the sole cause of the observed behavior.",
            "These data can inform whether to test basic motion/turn exploration before extending credit assignment, but this audit does not choose or implement a reward, curriculum, or geometry change.",
        ],
    }

    output = BASE_OUTPUT
    suffix = 2
    while output.exists():
        output = BASE_OUTPUT.with_name(f"{BASE_OUTPUT.stem}_{suffix}{BASE_OUTPUT.suffix}")
        suffix += 1
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(PROJECT))


if __name__ == "__main__":
    main()
