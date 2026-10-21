#!/usr/bin/env python3
"""Evaluate saved curriculum checkpoints on the four paired free-space seeds."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.train_isaac_foundation_fresh import ENV_SEED, _config_dict, _evaluate
from isaac_field.task import Config


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_minimal_inputs_20261020"
OUT = Path(__file__).resolve().parent / "stage_pairing"
FREE_SEEDS = (87001, 87002, 87003, 87004)
STAGES = (
    ("stage01_near", "stage_01_near.zip", 32768),
    ("stage02_steer30", "stage_02_steer30.zip", 65536),
    ("stage03_steer60", "stage_03_steer60.zip", 131072),
)
REFERENCE_EVALS = (
    TRIAL / "evaluation_free_baseline_87001_87004.json",
    TRIAL / "evaluation_free_final_87001_87004.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_row(row: dict[str, Any], report_name: str) -> dict[str, Any]:
    trajectory = row["trajectory"]
    if len(trajectory) != int(row["steps"]):
        raise ValueError(f"seed {row['seed']}: trajectory length mismatch")
    last = trajectory[-1]
    actual = [sample["after"]["actual_v_w"] for sample in trajectory]
    commands = [sample["command_v_w"] for sample in trajectory]
    tracks = [sample["track_command_lr_mps"] for sample in trajectory]
    action = [sample["action_4"] for sample in trajectory]
    mean_abs_command_yaw = sum(abs(float(v[1])) for v in commands) / len(commands)
    mean_abs_actual_yaw = sum(abs(float(v[1])) for v in actual) / len(actual)
    return {
        "seed": int(row["seed"]),
        "evaluation_file": report_name,
        "success": bool(row["success"]),
        "raw_return": float(row["raw_return"]),
        "final_distance_m": float(row["final_distance"]),
        "damage": float(row["damage"]),
        "collisions": int(row["collisions"]),
        "reason": row["reason"],
        "steps": int(row["steps"]),
        "timed_out": bool(row["timed_out"]),
        "reset_initial_condition": row["reset_initial_condition"],
        "goal": row["goal"],
        "trajectory_start_matches_reset": bool(row["trajectory_start_matches_reset"]),
        "start_distance_m": float(trajectory[0]["before"]["goal_distance_m"]),
        "min_distance_m": min(float(item["after"]["goal_distance_m"]) for item in trajectory),
        "mean_command_v_mps": sum(float(v[0]) for v in commands) / len(commands),
        "mean_command_yaw_rate_rad_s": sum(float(v[1]) for v in commands) / len(commands),
        "mean_abs_command_yaw_rate_rad_s": mean_abs_command_yaw,
        "mean_actual_v_mps": sum(float(v[0]) for v in actual) / len(actual),
        "mean_actual_yaw_rate_rad_s": sum(float(v[1]) for v in actual) / len(actual),
        "mean_abs_actual_yaw_rate_rad_s": mean_abs_actual_yaw,
        "command_to_actual_abs_yaw_rate_ratio": (
            mean_abs_command_yaw / mean_abs_actual_yaw if mean_abs_actual_yaw > 1e-12 else None
        ),
        "mean_left_track_command_mps": sum(float(v[0]) for v in tracks) / len(tracks),
        "mean_right_track_command_mps": sum(float(v[1]) for v in tracks) / len(tracks),
        "max_abs_action_component": max(abs(float(x)) for values in action for x in values),
        "final_pose_xy_yaw": last["after"]["pose_xy_yaw"],
        "final_actual_v_w": last["after"]["actual_v_w"],
        "final_command_v_w": last["command_v_w"],
        "final_track_command_lr_mps": last["track_command_lr_mps"],
        "final_goal_frame_xy_m": last["after"]["goal_robot_frame_m"],
        "final_goal_bearing_rad": float(last["after"]["goal_bearing_rad"]),
    }


def check_pairing(stage_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    references = {str(path.name): json_file(path) for path in REFERENCE_EVALS}
    for name, report in references.items():
        observed = {int(row["seed"]) for row in report["rows"]}
        if observed != set(FREE_SEEDS):
            raise ValueError(f"reference {name} has unexpected seed set {sorted(observed)}")

    all_reports = {**references, **stage_reports}
    first_name = next(iter(all_reports))
    first = {int(row["seed"]): row for row in all_reports[first_name]["rows"]}
    initial_conditions = {}
    for seed in FREE_SEEDS:
        seed_rows = {}
        reference = first[seed]
        for name, report in all_reports.items():
            row = next(row for row in report["rows"] if int(row["seed"]) == seed)
            reset_equal = row["reset_initial_condition"] == reference["reset_initial_condition"]
            goal_equal = row["goal"] == reference["goal"]
            trajectory_matches = bool(row["trajectory_start_matches_reset"])
            seed_rows[name] = {
                "reset_initial_condition": row["reset_initial_condition"],
                "goal": row["goal"],
                "reset_equal_to_reference": reset_equal,
                "goal_equal_to_reference": goal_equal,
                "trajectory_start_matches_reset": trajectory_matches,
            }
            if not (reset_equal and goal_equal and trajectory_matches):
                raise ValueError(f"seed {seed}: evaluation initial conditions are not paired ({name})")
        initial_conditions[str(seed)] = seed_rows

    reference_name = next(iter(references))
    reference_report = references[reference_name]
    contract_fields = (
        "action_config", "drive_config", "damage_config", "damage_metric", "observation_config",
    )
    contract_equal = {}
    for name, report in all_reports.items():
        contract_equal[name] = {
            field: report[field] == reference_report[field]
            for field in contract_fields
        }
        if not all(contract_equal[name].values()):
            raise ValueError(f"evaluation contract differs from original paired free evaluation: {name}")
    return {
        "reference_evaluations": list(references),
        "all_same_reset_and_goal_for_each_seed": True,
        "all_trajectories_start_at_recorded_reset": True,
        "contracts_equal_to_original_free_baseline": contract_equal,
        "complete_initial_conditions_by_seed": initial_conditions,
    }


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing stage-pairing outputs: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(**_config_dict())
    stage_reports: dict[str, dict[str, Any]] = {}
    checkpoint_rows = []
    for name, checkpoint_name, timesteps in STAGES:
        checkpoint = TRIAL / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        label = f"{name}_free_87001_87004"
        report = _evaluate(
            OUT,
            cfg,
            checkpoint,
            label,
            FREE_SEEDS,
            plants=0,
            angle_deg=90.0,
            device="cuda:0",
        )
        stage_reports[report[f"checkpoint"]] = report
        output_name = f"evaluation_{label}.json"
        rows = [summarize_row(row, output_name) for row in report["rows"]]
        checkpoint_rows.append({
            "stage": name,
            "cumulative_training_timesteps": timesteps,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_file": output_name,
            "summary": report["summary"],
            "episode_summaries": rows,
            "action_config": report["action_config"],
            "drive_config": report["drive_config"],
            "damage_config": report["damage_config"],
            "damage_metric": report["damage_metric"],
            "observation_config": report["observation_config"],
            "runtime_assets": report.get("runtime_assets", []),
        })
        print(json.dumps({
            "stage": name,
            "checkpoint_sha256": checkpoint_hash,
            "summary": report["summary"],
        }, ensure_ascii=False, sort_keys=True), flush=True)

    pairing = check_pairing(stage_reports)
    result = {
        "schema": "minimal-inputs-stage-checkpoint-paired-evaluation-v1",
        "claim_limit": "Four development seeds only; diagnostic, not blind validation or generalization evidence.",
        "evaluation_config": {
            "mode": "goto",
            "plants": 0,
            "terrain_enabled": bool(cfg.terrain_enabled),
            "goal_distance_m": [1.0, 3.0],
            "goal_angle_deg": 90.0,
            "distance_cost": 0.0,
            "deterministic": True,
            "environment_seed": ENV_SEED,
            "evaluation_seeds": list(FREE_SEEDS),
            "model_device": "cuda:0",
            "record_trajectories": True,
        },
        "pairing": pairing,
        "stages": checkpoint_rows,
    }
    summary_path = OUT / "stage_pairing_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
