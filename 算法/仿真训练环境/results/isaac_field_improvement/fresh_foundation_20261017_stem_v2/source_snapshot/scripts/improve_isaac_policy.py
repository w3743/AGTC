"""Run reproducible smoke or staged-goto RecurrentPPO fine-tuning checks.

These are ability checks, not benchmarks. Each Isaac SimulationContext is
closed before the next one is created so evaluation and training do not overlap.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import asdict
import json
from pathlib import Path
import math

from isaac_field.learning import DEFAULT_INITIAL_CHECKPOINT, evaluate_model, train_stage
from isaac_field.task import (
    Config,
    DAMAGE_METRIC_ROOT_PROTECTION_V1,
    DAMAGE_METRIC_STEM_CONTACT_V2,
)


def _evaluate(
    checkpoint: Path,
    *,
    cfg: Config,
    seeds: tuple[int, ...],
    device: str,
    goal_distance: tuple[float, float] = (0.6, 1.2),
    goal_angle: float = math.radians(30),
    record_trajectories: bool = False,
):
    from isaac_field.env import SingleFieldAdapter, make_env

    engine = make_env(
        cfg=cfg,
        mode="goto",
        num_envs=1,
        seed=80_003,
        auto_reset=False,
        goal_distance=goal_distance,
        goal_angle=goal_angle,
        distance_cost=0.0,
    )
    try:
        adapter = SingleFieldAdapter(engine)
        return evaluate_model(
            adapter,
            checkpoint,
            seeds,
            goal_distance=goal_distance,
            goal_angle=goal_angle,
            mode="goto",
            deterministic=True,
            device=device,
            record_trajectories=record_trajectories,
        )
    finally:
        engine.close()


def run_comparison(
    *,
    checkpoint: str | Path = DEFAULT_INITIAL_CHECKPOINT,
    output_dir: str | Path | None = None,
    timesteps: int = 4096,
    eval_episodes: int = 4,
    device: str = "cuda:0",
    training_seed: int = 31_001,
    record_trajectories: bool = False,
) -> dict:
    """Fine-tune briefly and compare on four fixed, training-disjoint seeds."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if timesteps < 128 or timesteps % 128:
        raise ValueError("timesteps must be a positive multiple of the saved 128-step rollout")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")

    heldout_seeds = tuple(range(51_001, 51_001 + eval_episodes))
    if training_seed in heldout_seeds:
        raise ValueError("training_seed must be disjoint from held-out evaluation seeds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_dir) if output_dir is not None else Path(
        "results/isaac_field_improvement"
    ) / f"short_goto_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output / "field_policy_after.zip"

    # Keep engine/observation geometry constant while making this a fast,
    # flat-terrain navigation check representative of the agreed crop spacing.
    cfg = Config(
        plants=16,
        layout="rows",
        row_spacing=0.8,
        plant_spacing=0.3,
        terrain_enabled=False,
        max_time=8.0,
    )
    evaluation = {
        "mode": "goto",
        "goal_distance_m": [0.6, 1.2],
        "goal_angle_rad": math.radians(30),
        "max_time_s": cfg.max_time,
        "distance_cost": 0.0,
        "deterministic": True,
        "record_trajectories": bool(record_trajectories),
        "plants": cfg.plants,
        "layout": cfg.layout,
        "row_spacing_m": cfg.row_spacing,
        "plant_spacing_m": cfg.plant_spacing,
        "terrain_enabled": cfg.terrain_enabled,
        "heldout_seeds": list(heldout_seeds),
        "training_seed": training_seed,
    }

    print("Starting baseline evaluation", flush=True)
    before = _evaluate(
        checkpoint, cfg=cfg, seeds=heldout_seeds, device=device,
        record_trajectories=record_trajectories,
    )
    (output / "before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from isaac_field.env import make_env

    print("Starting short fine-tuning", flush=True)
    train_env = make_env(
        cfg=cfg,
        mode="goto",
        num_envs=2,
        seed=training_seed,
        auto_reset=True,
        goal_distance=(0.6, 1.2),
        goal_angle=math.radians(30),
        distance_cost=1.0,
    )
    try:
        stage = train_stage(
            train_env,
            checkpoint,
            timesteps,
            final_checkpoint,
            seed=training_seed,
            device=device,
            goal_gain=100.0,
            n_steps=128,
            batch_size=128,
            n_epochs=5,
            verbose=1,
        )
    finally:
        train_env.close()

    print("Starting post-training evaluation", flush=True)
    after = _evaluate(
        stage.checkpoint, cfg=cfg, seeds=heldout_seeds, device=device,
        record_trajectories=record_trajectories,
    )
    (output / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "schema": "isaac-field-short-policy-improvement-v1",
        "algorithm": "SB3-Contrib RecurrentPPO",
        "checkpoint_before": str(checkpoint),
        "checkpoint_after": str(stage.checkpoint),
        "evaluation": evaluation,
        "training": {
            "requested_timesteps": stage.requested_timesteps,
            "trained_timesteps": stage.trained_timesteps,
            "seed": stage.seed,
            "num_envs": 2,
            "distance_cost": 1.0,
            "n_steps": stage.n_steps,
            "batch_size": stage.batch_size,
            "n_epochs": stage.n_epochs,
        },
        "heldout_before_summary": before["summary"],
        "heldout_after_summary": after["summary"],
        "heldout_before_after_rows": [
            {"seed": seed, "before": before["rows"][i], "after": after["rows"][i]}
            for i, seed in enumerate(heldout_seeds)
        ],
        "claim_limit": (
            "This is a short flat-terrain, 16-plant smoke comparison on four fixed seeds; "
            "it does not establish general navigation competence or statistically reliable improvement."
        ),
    }
    summary_path = output / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "before": before["summary"],
                      "after": after["summary"], "summary": str(summary_path)},
                     ensure_ascii=False, indent=2), flush=True)
    return summary


def _make_progress_callback(num_envs: int, native_env, interval: int = 2048):
    """Print sparse task outcomes in aggregate transitions, without FPS timing."""
    from stable_baselines3.common.callbacks import BaseCallback

    class AggregateProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.recent_episodes = []
            self.log_rows = []

        def _on_step(self) -> bool:
            calls_per_log = max(1, int(interval) // int(num_envs))
            if self.n_calls % calls_per_log == 0:
                self._refresh_episodes()
                recent = self.recent_episodes[-10:]
                row = {
                    "timesteps": int(self.num_timesteps),
                    "recent_episode_count": len(recent),
                    "recent_successes": sum(int(item["success"]) for item in recent),
                    "recent_mean_damage": (
                        None if not recent else
                        sum(item["damage"] for item in recent) / len(recent)
                    ),
                }
                self.log_rows.append(row)
                success_text = (
                    "n/a" if not recent else
                    f"{row['recent_successes']}/{len(recent)}"
                )
                damage_text = (
                    "n/a" if row["recent_mean_damage"] is None else
                    f"{row['recent_mean_damage']:.2f}"
                )
                print(
                    f"[staged-goto] timesteps={row['timesteps']} "
                    f"recent_success={success_text} recent_mean_damage={damage_text}",
                    flush=True,
                )
            return True

        def _refresh_episodes(self):
            # The official fast SB3 wrapper deliberately omits custom task info.
            # Read the native task's completed episodes instead of treating
            # missing wrapper keys as zero successes.
            self.recent_episodes = [
                {"success": bool(info.get("is_success", False)),
                 "damage": int(info.get("damage", 0))}
                for info in native_env.episode_results[-10:]
            ]

        def _on_training_end(self):
            self._refresh_episodes()

    return AggregateProgressCallback()


def _add_safety_summary(report: dict) -> dict:
    """Add crop-priority counts derived directly from held-out episode rows."""
    result = dict(report)
    summary = dict(report["summary"])
    rows = report["rows"]
    summary["clean_successes"] = sum(
        bool(row.get("success", False))
        and int(row.get("damage", 0)) == 0
        and int(row.get("collisions", 0)) == 0
        for row in rows
    )
    summary["damaged_episodes"] = sum(int(row.get("damage", 0)) > 0 for row in rows)
    summary["total_collisions"] = sum(int(row.get("collisions", 0)) for row in rows)
    result["summary"] = summary
    return result


def _safe_goto_assessment(before: dict, after: dict) -> str:
    """Apply the predeclared clean-success and crop-safety comparison rule."""
    def damage_metric(report: dict) -> str | None:
        config = report.get("damage_config")
        metric = config.get("metric") if isinstance(config, dict) else None
        if metric is None:
            metric = report.get("damage_metric")
        rows = report.get("rows")
        row_metrics = (
            {row.get("damage_metric") for row in rows if isinstance(row, dict)
             and row.get("damage_metric") is not None}
            if isinstance(rows, list)
            else set()
        )
        if len(row_metrics) > 1:
            return None
        if row_metrics:
            row_metric = next(iter(row_metrics))
            if metric is not None and metric != row_metric:
                return None
            metric = row_metric
        # Pre-metadata held-out reports used the original root-protection rule.
        if metric is None:
            return DAMAGE_METRIC_ROOT_PROTECTION_V1
        metric = str(metric)
        return metric if metric in (
            DAMAGE_METRIC_ROOT_PROTECTION_V1,
            DAMAGE_METRIC_STEM_CONTACT_V2,
        ) else None

    before_metric, after_metric = damage_metric(before), damage_metric(after)
    if before_metric is None or after_metric is None or before_metric != after_metric:
        return "incomparable_damage_metric"
    b, a = before["summary"], after["summary"]
    safety_not_worse = (
        a["damaged_episodes"] <= b["damaged_episodes"]
        and a["mean_damage"] <= b["mean_damage"]
        and a["total_collisions"] <= b["total_collisions"]
    )
    safety_strictly_better = (
        a["damaged_episodes"] < b["damaged_episodes"]
        or a["mean_damage"] < b["mean_damage"]
        or a["total_collisions"] < b["total_collisions"]
    )
    if a["clean_successes"] > b["clean_successes"] and safety_not_worse:
        return "improved_under_predeclared_safety_criterion"
    if (
        a["clean_successes"] <= b["clean_successes"]
        and safety_not_worse
        and safety_strictly_better
    ):
        return "more_conservative_without_clean_success_gain"
    if a["clean_successes"] > b["clean_successes"]:
        return "success_safety_tradeoff"
    return "no_improvement"


def run_staged_goto(
    *,
    checkpoint: str | Path = DEFAULT_INITIAL_CHECKPOINT,
    output_dir: str | Path | None = None,
    timesteps: int = 32_768,
    heldout_seed_base: int | None = None,
    device: str = "cuda:0",
    record_trajectories: bool = False,
    reward_profile: str = "legacy",
) -> dict:
    """Run the existing two-stage goto curriculum with a selectable reward profile."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if reward_profile not in ("legacy", "protect_plants"):
        raise ValueError("reward_profile must be 'legacy' or 'protect_plants'")
    safe_profile = reward_profile == "protect_plants"
    if heldout_seed_base is None:
        heldout_seed_base = 81_001 if safe_profile else 76_001
    if safe_profile:
        # The safe protocol always records trajectories, irrespective of CLI flags.
        record_trajectories = True
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or timesteps < 1024:
        raise ValueError("staged-goto timesteps must be an integer of at least 1024")
    stage_timesteps = int(timesteps) // 2
    if int(timesteps) % 2 or stage_timesteps % 512:
        raise ValueError("each staged-goto stage must be a multiple of 512 transitions")
    if isinstance(heldout_seed_base, bool) or int(heldout_seed_base) != heldout_seed_base or heldout_seed_base < 0:
        raise ValueError("heldout_seed_base must be a non-negative integer")

    heldout_count = 24 if safe_profile else 12
    heldout_seeds = tuple(range(int(heldout_seed_base), int(heldout_seed_base) + heldout_count))
    stage_seeds = (66_001, 66_002) if safe_profile else (46_001, 46_002)
    if set(heldout_seeds).intersection(stage_seeds):
        raise ValueError("held-out seeds must not overlap staged-goto training seeds")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(output_dir) if output_dir is not None else Path(
        "results/isaac_field_improvement"
    ) / f"{'safe_goto' if safe_profile else 'staged_goto'}_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        plants=16,
        layout="rows",
        row_spacing=0.8,
        plant_spacing=0.3,
        terrain_enabled=False,
        max_time=20.0,
        damage_penalty=20.0 if safe_profile else 4.0,
        goal_progress_budget=5.0 if safe_profile else None,
    )
    evaluation_goal_distance = (1.0, 3.0)
    evaluation_goal_angle = math.radians(60)
    stages = (
        {"name": "near", "seed": stage_seeds[0], "goal_distance_m": [0.45, 1.25],
         "goal_angle_deg": 45.0},
        {"name": "medium", "seed": stage_seeds[1], "goal_distance_m": [1.0, 3.0],
         "goal_angle_deg": 60.0},
    )
    num_envs = 4
    training_distance_cost = 0.0 if safe_profile else 1.0
    stage_hyperparameters = {
        "timesteps_per_stage": stage_timesteps,
        "num_envs": num_envs,
        "learning_rate": 1e-4,
        "n_steps": 128,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.995,
        "target_kl": 0.03,
        "log_std_init_if_fresh": -2.0,
        "warm_start_exploration": "preserved from the loaded checkpoint",
        "verbose": 0,
        "distance_cost": training_distance_cost,
        "goal_gain": 100.0,
        "algorithm": "SB3-Contrib RecurrentPPO",
        "device": device,
        "checkpoint_aggregate_interval": 4096,
        "progress_aggregate_interval": 2048,
        "checkpoint_callback_save_freq_calls": 4096 // num_envs,
    }
    protocol = {
        "schema": "isaac-field-staged-goto-protocol-v1",
        "protocol": "safe-goto" if safe_profile else "staged-goto",
        "reward_profile": reward_profile,
        "model_source": str(checkpoint),
        "total_timesteps": int(timesteps),
        "stage_count": len(stages),
        "config": asdict(cfg),
        "evaluation": {
            "mode": "goto",
            "goal_distance_m": list(evaluation_goal_distance),
            "goal_angle_deg": 60.0,
            "goal_angle_rad": evaluation_goal_angle,
            "distance_cost": 0.0,
            "deterministic": True,
            "heldout_seed_base": int(heldout_seed_base),
            "heldout_seeds": list(heldout_seeds),
            "record_trajectories": bool(record_trajectories),
            "plants": cfg.plants,
            "layout": cfg.layout,
            "terrain_enabled": cfg.terrain_enabled,
            "claim_limit": (
                "Local 16-plant flat-terrain goto evaluation only; this does not establish "
                "full-field navigation or search capability."
            ),
        },
        "training": {
            "distance_cost": training_distance_cost,
            "stage_seeds": list(stage_seeds),
            "stage_timesteps": stage_timesteps,
            "hyperparameters": stage_hyperparameters,
        },
        "predeclared_assessment": {
            "improvement_requires": (
                "clean_successes increase while damaged_episodes, mean_damage, and "
                "total_collisions do not increase"
            ),
            "clean_success_definition": (
                "successful episode with zero damaged plants and zero collisions"
            ),
            "otherwise": (
                "report a safety-success tradeoff or a more-conservative result; "
                "do not claim ability improvement"
            ),
            "raw_return_comparison": "compare only within this same reward profile",
        },
        "training_seed_policy": {
            "stage_seeds": list(stage_seeds),
            "heldout_disjoint": True,
        },
        "stages": [
            {
                **stage,
                "checkpoint_source": (
                    str(checkpoint) if index == 0 else
                    str(output / "stage_01_near" / "field_policy_near.zip")
                ),
                "goal_angle_rad": math.radians(stage["goal_angle_deg"]),
                "hyperparameters": stage_hyperparameters,
            }
            for index, stage in enumerate(stages)
        ],
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Starting staged-goto baseline evaluation", flush=True)
    before = _evaluate(
        checkpoint,
        cfg=cfg,
        seeds=heldout_seeds,
        device=device,
        goal_distance=evaluation_goal_distance,
        goal_angle=evaluation_goal_angle,
        record_trajectories=record_trajectories,
    )
    before = _add_safety_summary(before)
    (output / "before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from isaac_field.env import make_env

    stage_reports = []
    active_checkpoint = checkpoint
    for index, stage_spec in enumerate(stages):
        stage_dir = output / f"stage_{index + 1:02d}_{stage_spec['name']}"
        checkpoint_dir = stage_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stage_checkpoint = (
            output / "field_policy_after.zip"
            if index == len(stages) - 1
            else stage_dir / f"field_policy_{stage_spec['name']}.zip"
        )
        goal_angle = math.radians(stage_spec["goal_angle_deg"])
        train_env = make_env(
            cfg=cfg,
            mode="goto",
            num_envs=num_envs,
            seed=stage_spec["seed"],
            auto_reset=True,
            goal_distance=tuple(stage_spec["goal_distance_m"]),
            goal_angle=goal_angle,
            distance_cost=training_distance_cost,
        )
        progress_callback = _make_progress_callback(num_envs=num_envs, native_env=train_env, interval=2048)
        checkpoint_callback = CheckpointCallback(
            save_freq=4096 // num_envs,
            save_path=str(checkpoint_dir),
            name_prefix=f"field_policy_{stage_spec['name']}",
            verbose=0,
        )
        try:
            print(f"Starting staged-goto stage {stage_spec['name']}", flush=True)
            result = train_stage(
                train_env,
                active_checkpoint,
                stage_timesteps,
                stage_checkpoint,
                callback=CallbackList([checkpoint_callback, progress_callback]),
                seed=stage_spec["seed"],
                device=device,
                goal_gain=100.0,
                n_steps=128,
                batch_size=128,
                n_epochs=5,
                learning_rate=1e-4,
                gamma=0.995,
                target_kl=0.03,
                verbose=0,
            )
        finally:
            train_env.close()
        active_checkpoint = result.checkpoint
        stage_reports.append({
            "name": stage_spec["name"],
            "seed": int(stage_spec["seed"]),
            "goal_distance_m": list(stage_spec["goal_distance_m"]),
            "goal_angle_deg": float(stage_spec["goal_angle_deg"]),
            "checkpoint_in": str(stage_reports[-1]["checkpoint_out"] if stage_reports else checkpoint),
            "checkpoint_out": str(result.checkpoint),
            "requested_timesteps": int(result.requested_timesteps),
            "trained_timesteps": int(result.trained_timesteps),
            "aggregate_progress": progress_callback.log_rows,
            "last_completed_episodes": progress_callback.recent_episodes[-10:],
            "periodic_checkpoints": sorted(
                str(path) for path in checkpoint_dir.glob("*.zip")
            ),
        })

    (output / "stages_report.json").write_text(
        json.dumps(stage_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_checkpoint = Path(active_checkpoint)

    print("Starting staged-goto post-training evaluation", flush=True)
    after = _evaluate(
        final_checkpoint,
        cfg=cfg,
        seeds=heldout_seeds,
        device=device,
        goal_distance=evaluation_goal_distance,
        goal_angle=evaluation_goal_angle,
        record_trajectories=record_trajectories,
    )
    after = _add_safety_summary(after)
    (output / "after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "schema": "isaac-field-staged-goto-comparison-v1",
        "protocol": str(protocol_path),
        "checkpoint_before": str(checkpoint),
        "checkpoint_after": str(final_checkpoint),
        "stages_report": str(output / "stages_report.json"),
        "heldout_before_report": str(output / "before.json"),
        "heldout_after_report": str(output / "after.json"),
        "heldout_before_summary": before["summary"],
        "heldout_after_summary": after["summary"],
        "assessment": _safe_goto_assessment(before, after) if safe_profile else None,
        "heldout_before_after_rows": [
            {
                "seed": seed,
                "before": {k: v for k, v in before["rows"][i].items() if k != "trajectory"},
                "after": {k: v for k, v in after["rows"][i].items() if k != "trajectory"},
            }
            for i, seed in enumerate(heldout_seeds)
        ],
        "claim_limit": (
            f"This is a {heldout_count}-seed local flat-terrain goto evaluation with "
            "16 plants; it does not establish navigation or search capability across "
            "the full 100 m field."
        ),
    }
    summary_path = output / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "before": before["summary"],
                      "after": after["summary"], "checkpoint": str(final_checkpoint),
                      "summary": str(summary_path)}, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("smoke", "staged-goto", "safe-goto"), default="smoke")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=31_001)
    parser.add_argument("--heldout-seed-base", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--record-trajectories", action="store_true",
        help="store compact per-step action, pose, velocity, and goal-relative data in before/after JSON",
    )
    args = parser.parse_args()
    if args.protocol == "smoke":
        run_comparison(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            timesteps=4096 if args.timesteps is None else args.timesteps,
            eval_episodes=args.eval_episodes,
            training_seed=args.training_seed,
            device=args.device,
            record_trajectories=args.record_trajectories,
        )
    else:
        safe_profile = args.protocol == "safe-goto"
        run_staged_goto(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            timesteps=(65_536 if safe_profile else 32_768)
            if args.timesteps is None else args.timesteps,
            heldout_seed_base=args.heldout_seed_base,
            device=args.device,
            record_trajectories=args.record_trajectories or safe_profile,
            reward_profile="protect_plants" if safe_profile else "legacy",
        )


if __name__ == "__main__":
    main()
