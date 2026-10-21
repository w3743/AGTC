"""Prepare or launch the fresh two-stage Isaac Field foundation curriculum."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_isaac_exploration import (
    SOURCE_FILES as RUNTIME_FILES,
    _json,
    _pair_initial_conditions,
    _sha256,
    _source_manifest,
)

RUNNER = "scripts/train_isaac_foundation.py"
BASELINE = ROOT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip"
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/fresh_foundation_20261017_stem_v2"
TRAIN_SEED = 69_001
FREE_SEEDS = tuple(range(87_001, 87_025))
PLANT_SEEDS = tuple(range(82_001, 82_025))
ENV_SEED = 80_003
STAGES = (
    {"name": "near", "goal_distance_m": [0.6, 1.2], "goal_angle_deg": 15.0, "timesteps": 32_768},
    {"name": "turn", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0, "timesteps": 131_072},
)


def _snapshot(output: Path) -> dict:
    manifest = _source_manifest(output, copy_files=True)
    source = ROOT / RUNNER
    relative = f"source_snapshot/{RUNNER}"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest["files"].append({"path": RUNNER, "snapshot_path": relative, "sha256": _sha256(source)})
    return manifest


def prepare(output: Path) -> dict:
    from isaac_field.task import Config
    from scripts.compare_isaac_exploration import _checkpoint_steps

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse prepared output: {output}")
    if not BASELINE.is_file() or _checkpoint_steps(BASELINE) != 53_248:
        raise ValueError(f"expected the recorded 53,248-step development baseline: {BASELINE}")
    cfg = Config(plants=0, terrain_enabled=False, max_time=20.0, damage_penalty=20.0,
                 goal_progress_budget=5.0, damage_metric="stem-contact-v2")
    output.mkdir(parents=True)
    snapshot = _snapshot(output)
    if len(RUNTIME_FILES) != 19 or len(snapshot["files"]) != 20:
        raise RuntimeError(
            f"expected 19 frozen runtime files plus this runner, got {len(snapshot['files'])} files"
        )
    protocol = {
        "schema": "isaac-field-foundation-curriculum-v1",
        "status": "prepared",
        "source_snapshot": snapshot,
        "baseline_checkpoint": {
            "path": BASELINE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(BASELINE),
            "num_timesteps": 53_248,
        },
        "initialization": "fresh RecurrentPPO; no checkpoint warm-start",
        "config": asdict(cfg),
        "training": {
            "algorithm": "SB3-Contrib RecurrentPPO",
            "num_envs": 4,
            "training_seeds": [TRAIN_SEED, TRAIN_SEED + 1],
            "timesteps_total": sum(stage["timesteps"] for stage in STAGES),
            "stages": [dict(stage) for stage in STAGES],
            "mode": "goto",
            "distance_cost": 0.0,
            "goal_gain": 100.0,
            "n_steps": 128,
            "batch_size": 128,
            "n_epochs": 5,
            "learning_rate": 3e-4,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "target_kl": 0.03,
            "log_std_init": -2.0,
            "use_sde": False,
            "checkpoint_interval_transitions": 8192,
        },
        "evaluation": {
            "device": "selected at launch",
            "environment_seed": ENV_SEED,
            "distance_cost": 0.0,
            "deterministic": True,
            "record_trajectories": True,
            "free_space": {"plants": 0, "seeds": list(FREE_SEEDS),
                           "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0},
            "plants": {"plants": 16, "seeds": list(PLANT_SEEDS),
                       "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0,
                       "baseline_and_final_paired": True},
            "dataset_role": "development, not blind validation",
            "claim_limit": "Flat local tasks only; free-space success does not establish reliable field navigation or search.",
        },
    }
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": _sha256(output / "protocol.json")
    })
    return {"output": str(output), "snapshot_files": len(protocol["source_snapshot"]["files"]),
            "baseline_sha256": protocol["baseline_checkpoint"]["sha256"]}


def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --launch")
    protocol, status = (json.loads(path.read_text(encoding="utf-8")) for path in (protocol_path, status_path))
    if status.get("status") != "prepared" or status.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("run is not an unchanged prepared experiment")
    if (protocol.get("schema") != "isaac-field-foundation-curriculum-v1"
            or protocol.get("config", {}).get("damage_metric") != "stem-contact-v2"):
        raise ValueError("runner requires the foundation protocol with damage_metric='stem-contact-v2'")
    if len(RUNTIME_FILES) != 19 or len(protocol["source_snapshot"]["files"]) != 20:
        raise ValueError("expected the frozen 19-file runtime plus this runner")
    for row in protocol["source_snapshot"]["files"]:
        live, snapshot = ROOT / row["path"], output / row["snapshot_path"]
        if not live.is_file() or not snapshot.is_file() or _sha256(live) != row["sha256"] or _sha256(snapshot) != row["sha256"]:
            raise ValueError(f"source snapshot mismatch: {row['path']}")
    baseline = ROOT / protocol["baseline_checkpoint"]["path"]
    if not baseline.is_file() or _sha256(baseline) != protocol["baseline_checkpoint"]["sha256"]:
        raise ValueError("development baseline checkpoint changed")
    return protocol


def _safety_summary(report: dict) -> dict:
    from scripts.improve_isaac_policy import _add_safety_summary
    return _add_safety_summary(report)


def _evaluate(output: Path, cfg, checkpoint: Path, label: str, seeds: tuple[int, ...], angle: float, device: str) -> dict:
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.learning import evaluate_model

    _json(output / "run_status.json", {"status": "running", "phase": f"evaluating_{label}"})
    engine = make_env(cfg=cfg, mode="goto", num_envs=1, seed=ENV_SEED, auto_reset=False,
                      goal_distance=(1.0, 3.0), goal_angle=angle, distance_cost=0.0)
    adapter = SingleFieldAdapter(engine)
    try:
        report = evaluate_model(adapter, checkpoint, seeds, goal_distance=(1.0, 3.0),
                                goal_angle=angle, mode="goto", device=device,
                                record_trajectories=True)
    finally:
        adapter.close()
    report = _safety_summary(report)
    _json(output / f"evaluation_{label}.json", report)
    return report


def launch(output: Path, device: str) -> dict:
    protocol = preflight(output)
    output = output.resolve()
    _json(output / "run_status.json", {"status": "running", "phase": "initializing"})
    try:
        from isaac_field.env import make_env
        from isaac_field.learning import CurriculumStage, train_curriculum
        from isaac_field.task import Config
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

        cfg = Config(**protocol["config"])
        stages = tuple(CurriculumStage(
            row["name"], "goto", tuple(row["goal_distance_m"]), math.radians(row["goal_angle_deg"])
        ) for row in protocol["training"]["stages"])
        latest_env = {"env": None}
        progress_by_stage = {}

        def env_factory(**kwargs):
            stage_name = "near" if tuple(kwargs["goal_distance"]) == (0.6, 1.2) else "turn"
            _json(output / "run_status.json", {"status": "running", "phase": f"creating_{stage_name}_env"})
            kwargs["num_envs"] = 4 if kwargs["auto_reset"] else 1
            latest_env["env"] = make_env(**kwargs)
            return latest_env["env"]

        def callback_factory(stage):
            index = next(i for i, item in enumerate(stages) if item.name == stage.name)
            stage_dir = output / f"stage_{index + 1}_{stage.name}"
            checkpoint_dir = stage_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _json(output / "run_status.json", {"status": "running", "phase": f"training_{stage.name}"})
            checkpoint = CheckpointCallback(
                save_freq=8192 // 4, save_path=str(checkpoint_dir),
                name_prefix=f"field_policy_{stage.name}", verbose=0,
            )
            progress = _make_progress_callback(4, latest_env["env"], interval=8192)
            progress_by_stage[stage.name] = progress

            class SaveProgressCallback(BaseCallback):
                def _on_step(self):
                    return True

                def _on_training_end(self):
                    _json(stage_dir / "progress.json", {
                        "aggregate_rows": progress.log_rows,
                        "last_completed_episodes": progress.recent_episodes[-10:],
                    })

            return CallbackList([checkpoint, progress, SaveProgressCallback()])

        training = protocol["training"]
        curriculum = train_curriculum(
            cfg, output, checkpoint=None,
            timesteps_per_stage=tuple(spec["timesteps"] for spec in training["stages"]),
            training_seed=TRAIN_SEED, heldout_seeds=FREE_SEEDS, stages=stages,
            evaluate_before_after=False, goal_gain=100.0, device=device,
            n_epochs=5, batch_size=128, callback_factory=callback_factory,
            env_factory=env_factory,
        )
        final_checkpoint = curriculum.checkpoint
        _json(output / "training_progress.json", {
            name: {"aggregate_rows": callback.log_rows,
                  "last_completed_episodes": callback.recent_episodes[-10:]}
            for name, callback in progress_by_stage.items()
        })
        _json(output / "run_status.json", {"status": "running", "phase": "evaluating_free_space"})
        free_cfg = replace(cfg, plants=0)
        free = _evaluate(output, free_cfg, final_checkpoint, "free_after_870xx", FREE_SEEDS,
                         math.pi / 2.0, device)

        field_cfg = replace(cfg, plants=16)
        baseline_path = ROOT / protocol["baseline_checkpoint"]["path"]
        before = _evaluate(output, field_cfg, baseline_path, "field_baseline_820xx",
                           PLANT_SEEDS, math.pi / 3.0, device)
        after = _evaluate(output, field_cfg, final_checkpoint, "field_after_820xx",
                          PLANT_SEEDS, math.pi / 3.0, device)
        pairing = _pair_initial_conditions({"baseline": before, "after": after})
        assessment = _safe_goto_assessment(before, after) if pairing["valid"] else "invalid_initial_conditions"
        summary = {
            "schema": "isaac-field-foundation-development-summary-v1",
            "curriculum_summary": str(curriculum.summary_path),
            "checkpoint_final": str(final_checkpoint),
            "free_space_summary": free["summary"],
            "field_baseline_summary": before["summary"],
            "field_after_summary": after["summary"],
            "field_pairing": pairing,
            "field_assessment": assessment,
            "drive_metadata": {
                "baseline_matches_runtime": before["drive_config"] == before["checkpoint_drive_config"],
                "final_matches_runtime": after["drive_config"] == after["checkpoint_drive_config"],
                "interpretation": "The baseline is re-evaluated on current physics; a legacy drive-metadata mismatch is recorded, not reused as an old report.",
            },
            "evaluation_reports": ["evaluation_free_after_870xx.json",
                                   "evaluation_field_baseline_820xx.json",
                                   "evaluation_field_after_820xx.json"],
            "claim_limit": protocol["evaluation"]["claim_limit"],
        }
        _json(output / "comparison_summary.json", summary)
        _json(output / "run_status.json", {"status": "completed", "summary": "comparison_summary.json"})
        return summary
    except Exception as exc:
        _json(output / "run_status.json", {"status": "failed", "exception": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = prepare(args.output) if args.prepare else launch(args.output, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
