"""Prepare and run a fresh, frozen RecurrentPPO foundation curriculum."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/fresh_depth_radar_20261019"
SCHEMA = "isaac-field-fresh-foundation-v1"
ENCODER_SCHEMA = "depth64-lidar-circular1d-v1"
INPUT_SCHEMA = "sensor-depth-only-v3"
TRAIN_SEED = 69_001
ENV_SEED = 80_003
DEV_FIELD_SEEDS = tuple(range(82_001, 82_005))
DEV_FREE_SEEDS = tuple(range(87_001, 87_005))

OBSERVATION_SHAPES = {
    "lidar": (2, 400), "depth": (1, 160, 88),
    "detections": (16, 10), "state": (43,), "action": (4,),
}
STAGES = (
    {"name": "near", "goal_distance_m": [0.6, 1.2], "goal_angle_deg": 15.0, "timesteps": 32_768},
    {"name": "steer30", "goal_distance_m": [1.0, 2.0], "goal_angle_deg": 30.0, "timesteps": 32_768},
    {"name": "steer60", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0, "timesteps": 65_536},
    {"name": "turn90", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0, "timesteps": 65_536},
)
TRAINING = {
    "algorithm": "SB3-Contrib RecurrentPPO", "mode": "goto", "num_envs": 4,
    "training_seed": TRAIN_SEED,
    "stage_seeds": [TRAIN_SEED + i for i in range(len(STAGES))],
    "stages": [dict(stage) for stage in STAGES],
    "timesteps_total": sum(stage["timesteps"] for stage in STAGES),
    "goal_gain": 1.0, "n_steps": 128, "batch_size": 128, "n_epochs": 5,
    "learning_rate": 3e-4, "gamma": 0.995, "gae_lambda": 0.95,
    "target_kl": 0.03, "log_std_init": -2.0, "use_sde": False,
    "checkpoint_interval_transitions": 8192,
    "initialization": "new random RecurrentPPO; no old checkpoint or weight transfer",
}
SOURCE_HELPERS = (
    "scripts/train_isaac_foundation.py",
    "scripts/improve_isaac_policy.py",
    "scripts/compare_isaac_exploration.py",
)
FIXED_SOURCES = ("scripts/isaaclab_python.sh", *SOURCE_HELPERS,
                 "scripts/train_isaac_foundation_fresh.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _protocol_sha(path: Path) -> str:
    return _sha256(path)


def _field_encoder_config() -> dict:
    from isaac_field import policy

    describe = getattr(policy, "_field_encoder_config", None)
    if not callable(describe):
        raise ValueError("policy does not expose its saved field_encoder_config")
    return dict(describe())


def _experiment_architecture() -> dict:
    from isaac_field import policy

    return {
        "field_encoder_config": _field_encoder_config(),
        "depth_value_range": [0.0, 1.0],
        "depth_space_bounds": [-1.0, 1.0],
        "input_contract": INPUT_SCHEMA,
        "observation_shapes": {key: list(value) for key, value in policy.OBSERVATION_SHAPES.items()},
        "action_shape": list(policy.ACTION_SHAPE),
        "goal_gain": 1.0,
        "lstm_hidden_size": 128,
        "architecture_is_searched": False,
    }


def _runtime_contract() -> dict:
    from isaac_field import policy

    actual_schema = getattr(policy, "ENCODER_SCHEMA", None)
    actual_shapes = getattr(policy, "OBSERVATION_SHAPES", None)
    expected_shapes = {key: tuple(value) for key, value in OBSERVATION_SHAPES.items() if key != "action"}
    if actual_schema != ENCODER_SCHEMA:
        raise ValueError(f"expected encoder {ENCODER_SCHEMA!r}, found {actual_schema!r}")
    if actual_shapes != expected_shapes or tuple(getattr(policy, "ACTION_SHAPE", ())) != (4,):
        raise ValueError(f"observation/action contract mismatch: {actual_shapes!r}")
    config = _field_encoder_config()
    if (config.get("schema") != ENCODER_SCHEMA
            or config.get("features_dim") != 160
            or config.get("branch_dims") != {"lidar": 32, "depth": 64, "detections": 32, "state": 32}
            or config.get("depth_pool") != [8, 6]
            or config.get("radar_pool") != 50
            or TRAINING["goal_gain"] != 1.0):
        raise ValueError(f"unexpected live field_encoder_config: {config!r}")
    return {"encoder_schema": actual_schema, "field_encoder_config": config,
            "observation_shapes": {key: list(value) for key, value in actual_shapes.items()},
            "goal_gain": 1.0}


def _config_dict() -> dict:
    from isaac_field.task import Config

    return asdict(Config(plants=0, terrain_enabled=False, max_time=20.0,
                         damage_penalty=20.0, goal_progress_budget=5.0,
                         damage_metric="stem-contact-v2"))


def _source_paths() -> tuple[str, ...]:
    package_files = sorted((ROOT / "isaac_field").rglob("*.py"))
    package_files += sorted((ROOT / "isaac_field").rglob("*.html"))
    paths = {path.relative_to(ROOT).as_posix() for path in package_files}
    paths.update(FIXED_SOURCES)
    missing = sorted(path for path in paths if not (ROOT / path).is_file())
    if missing:
        raise FileNotFoundError(f"fresh runner source files missing: {missing}")
    return tuple(sorted(paths))


def _source_manifest(output: Path, *, copy_files: bool) -> dict:
    rows = []
    for relative in _source_paths():
        source = ROOT / relative
        snapshot_path = f"source_snapshot/{relative}"
        digest = _sha256(source)
        if copy_files:
            target = output / snapshot_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rows.append({"path": relative, "snapshot_path": snapshot_path, "sha256": digest})
    return {"files": rows}


def _verify_sources(output: Path, manifest: dict) -> None:
    expected = set(_source_paths())
    rows = manifest.get("files", [])
    if {row.get("path") for row in rows} != expected:
        raise ValueError("frozen source manifest is incomplete or has unexpected files")
    for row in rows:
        live = ROOT / row["path"]
        snapshot = output / row["snapshot_path"]
        if not live.is_file() or not snapshot.is_file():
            raise FileNotFoundError(f"missing frozen source: {row['path']}")
        if _sha256(snapshot) != row["sha256"] or _sha256(live) != row["sha256"]:
            raise ValueError(f"source snapshot/live hash mismatch: {row['path']}")


def _package_versions() -> dict:
    names = ("torch", "stable-baselines3", "sb3-contrib", "gymnasium", "numpy", "warp-lang")
    versions = {name: None for name in names}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "packages": versions}


def _isaac_commit() -> str | None:
    result = subprocess.run(["git", "-C", str(ROOT / "external/IsaacLab"), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse fresh experiment output: {output}")
    _runtime_contract()
    config = _config_dict()
    if config.get("plants") != 0 or config.get("terrain_enabled") is not False:
        raise ValueError("fresh foundation training requires plants=0 and flat terrain")
    if config.get("max_time") != 20.0 or config.get("damage_penalty") != 20.0:
        raise ValueError("fresh foundation task configuration differs from the protocol")
    snapshot = _source_manifest(output, copy_files=False)
    output.mkdir(parents=True)
    snapshot = _source_manifest(output, copy_files=True)
    protocol = {
        "schema": SCHEMA,
        "status": "prepared",
        "architecture": _experiment_architecture(),
        "config": config,
        "robot_geometry": _robot_geometry(),
        "training": TRAINING,
        "initialization": {"checkpoint": None, "baseline_path": "untrained_baseline.zip",
                           "baseline_num_timesteps": 0, "weight_transfer": False},
        "evaluation": {
            "environment_seed": ENV_SEED, "mode": "goto", "distance_cost": 0.0,
            "deterministic": True, "record_trajectories": True,
            "free_space": {"plants": 0, "seeds": list(DEV_FREE_SEEDS),
                           "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0},
            "field": {"plants": 16, "seeds": list(DEV_FIELD_SEEDS),
                      "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0},
            "dataset_role": "development only; these four seeds do not establish generalization",
            "blind_seeds": {"range_label": "830xx", "used": False, "observed": False},
            "claim_limit": "This is not an architecture ablation or blind validation.",
        },
        "runtime": {"versions": _package_versions(), "isaaclab_commit": _isaac_commit()},
        "source_snapshot": snapshot,
    }
    _json(output / "protocol.json", protocol)
    digest = _protocol_sha(output / "protocol.json")
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": digest, "exit_code": 0,
        "num_timesteps": 0,
    })
    return {"output": str(output), "protocol_sha256": digest,
            "source_files": len(snapshot["files"]), "timesteps_total": TRAINING["timesteps_total"]}


def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --launch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    digest = _protocol_sha(protocol_path)
    if status.get("status") != "prepared" or status.get("protocol_sha256") != digest:
        raise ValueError("fresh trial is not an unchanged prepared experiment")
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported fresh protocol schema")
    if protocol.get("architecture") != _experiment_architecture() or protocol.get("training") != TRAINING:
        raise ValueError("prepared encoder or training contract differs from this runner")
    if protocol.get("config") != _config_dict():
        raise ValueError("prepared task config differs from the current Config")
    _runtime_contract()
    _verify_sources(output, protocol.get("source_snapshot", {}))
    return protocol


def _robot_geometry() -> dict:
    from isaac_field.assets import ROBOT_SPEC, robot_asset_spec_sha256

    return {"source": "isaac_field.assets.ROBOT_SPEC", "spec": asdict(ROBOT_SPEC),
            "sha256": robot_asset_spec_sha256(ROBOT_SPEC)}


def _checkpoint_steps(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read("data"))["num_timesteps"])


def _checkpoint_manifest(output: Path) -> list[dict]:
    rows = []
    for path in sorted(output.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        rows.append({"path": path.relative_to(output).as_posix(),
                     "sha256": _sha256(path), "num_timesteps": _checkpoint_steps(path),
                     "adam_optimizer_state_present": "policy.optimizer.pth" in names})
    _json(output / "checkpoint_manifest.json", {"checkpoints": rows})
    return rows


def _status(output: Path, runtime: dict, status: str = "running", **extra) -> None:
    _json(output / "run_status.json", {
        "status": status, "protocol_sha256": runtime.get("protocol_sha256"),
        "num_timesteps": int(runtime.get("num_timesteps", 0)),
        "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
    })


def _evaluate(output: Path, cfg, checkpoint: Path, label: str, seeds: tuple[int, ...],
              plants: int, angle_deg: float, device: str) -> dict:
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field.learning import evaluate_model
    from scripts.improve_isaac_policy import _add_safety_summary

    eval_cfg = replace(cfg, plants=plants)
    angle = math.radians(angle_deg)
    engine = make_env(cfg=eval_cfg, mode="goto", num_envs=1, seed=ENV_SEED,
                      auto_reset=False, goal_distance=(1.0, 3.0),
                      goal_angle=angle, distance_cost=0.0)
    adapter = SingleFieldAdapter(engine)
    try:
        report = evaluate_model(adapter, checkpoint, seeds,
                                goal_distance=(1.0, 3.0), goal_angle=angle,
                                mode="goto", deterministic=True, device=device,
                                record_trajectories=True)
    finally:
        adapter.close()
    report = _add_safety_summary(report)
    _json(output / f"evaluation_{label}.json", report)
    return report


def _paired_dev_conditions(reports: dict[str, dict], seeds: tuple[int, ...]) -> dict:
    from scripts.compare_isaac_exploration import _pair_initial_conditions

    result = _pair_initial_conditions(reports)
    expected = set(seeds)
    exact_rows = True
    for report in reports.values():
        rows = report.get("rows", [])
        observed = [row.get("seed") for row in rows]
        if len(rows) != len(seeds) or set(observed) != expected:
            exact_rows = False
            break
    result["helper_valid_24_seed_rule"] = bool(result.get("valid"))
    result["expected_seed_count"] = len(seeds)
    result["valid"] = bool(
        not result.get("mismatches")
        and result.get("paired_seed_count") == len(seeds)
        and exact_rows
    )
    return result


def launch(output: Path, device: str = "cuda:0") -> dict:
    protocol = preflight(output)
    output = output.resolve()
    runtime = {"protocol_sha256": _protocol_sha(output / "protocol.json"), "num_timesteps": 0}
    _status(output, runtime, phase="initializing")
    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from isaac_field.env import make_env
        from isaac_field.learning import CurriculumStage, train_curriculum
        from isaac_field.task import Config
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment

        cfg = Config(**protocol["config"])
        stages = tuple(CurriculumStage(
            row["name"], "goto", tuple(row["goal_distance_m"]),
            math.radians(row["goal_angle_deg"]),
        ) for row in protocol["training"]["stages"])
        latest = {"env": None}
        progress_by_stage = {}
        baseline_path = output / protocol["initialization"]["baseline_path"]

        def env_factory(**kwargs):
            _verify_sources(output, protocol["source_snapshot"])
            kwargs["num_envs"] = 4 if kwargs["auto_reset"] else 1
            latest["env"] = make_env(**kwargs)
            return latest["env"]

        def callback_factory(stage):
            index = next(i for i, item in enumerate(stages) if item.name == stage.name)
            stage_dir = output / f"stage_{index + 1}_{stage.name}"
            checkpoint_dir = stage_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _verify_sources(output, protocol["source_snapshot"])
            progress = _make_progress_callback(4, latest["env"], interval=8192)
            progress_by_stage[stage.name] = progress
            checkpoint = CheckpointCallback(save_freq=8192 // 4,
                                            save_path=str(checkpoint_dir),
                                            name_prefix=f"field_policy_{stage.name}", verbose=0)

            class InitialBaselineCallback(BaseCallback):
                def _on_training_start(self):
                    if baseline_path.exists():
                        raise FileExistsError(f"refusing to replace initial model: {baseline_path}")
                    if int(self.model.num_timesteps) != 0:
                        raise RuntimeError("new foundation model must start at zero timesteps")
                    extractor = self.model.policy.features_extractor
                    actor_lstm = getattr(self.model.policy, "lstm_actor", None)
                    if int(getattr(extractor, "features_dim", -1)) != 160:
                        raise ValueError("live encoder fusion dimension is not 160")
                    if float(getattr(extractor, "goal_gain", math.nan)) != 1.0:
                        raise ValueError("live encoder goal_gain is not 1.0")
                    if actor_lstm is None or int(getattr(actor_lstm, "hidden_size", -1)) != 128:
                        raise ValueError("live recurrent policy does not use LSTM hidden size 128")
                    field_encoder_config = _field_encoder_config()
                    if getattr(self.model, "field_encoder_config", None) != field_encoder_config:
                        raise ValueError("new model field_encoder_config differs from the current policy")
                    self.model.save(str(baseline_path.with_suffix("")))
                    if _checkpoint_steps(baseline_path) != 0:
                        raise RuntimeError("untrained baseline checkpoint is not at zero steps")
                    _json(output / "initialization.json", {
                        "checkpoint": baseline_path.name, "sha256": _sha256(baseline_path),
                        "num_timesteps": 0, "protocol_sha256": runtime["protocol_sha256"],
                        "field_encoder_config": field_encoder_config,
                        "protocol_architecture": protocol["architecture"],
                        "source_initialization": "new RecurrentPPO, no old checkpoint loaded",
                    })
                    _checkpoint_manifest(output)

                def _on_step(self):
                    return True

            class PersistStageProgress(BaseCallback):
                def _on_training_start(self):
                    self.stage_start = int(self.model.num_timesteps)
                    runtime["num_timesteps"] = self.stage_start
                    _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)

                def _on_step(self):
                    runtime["num_timesteps"] = int(self.num_timesteps)
                    if (runtime["num_timesteps"] - self.stage_start) % 8192 == 0:
                        _checkpoint_manifest(output)
                        _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    return True

                def _on_training_end(self):
                    progress._refresh_episodes()
                    runtime["num_timesteps"] = int(self.model.num_timesteps)
                    _json(stage_dir / "progress.json", {
                        "stage": stage.name, "start_num_timesteps": self.stage_start,
                        "end_num_timesteps": runtime["num_timesteps"],
                        "aggregate_rows": progress.log_rows,
                        "last_completed_episodes": progress.recent_episodes[-10:],
                    })
                    _status(output, runtime, phase=f"completed_{stage.name}", stage=stage.name)

            callbacks = [checkpoint, progress, PersistStageProgress()]
            if index == 0:
                callbacks.insert(0, InitialBaselineCallback())
            return CallbackList(callbacks)

        curriculum = train_curriculum(
            cfg, output, checkpoint=None,
            timesteps_per_stage=tuple(row["timesteps"] for row in TRAINING["stages"]),
            training_seed=TRAIN_SEED, heldout_seeds=DEV_FIELD_SEEDS,
            stages=stages, evaluate_before_after=False, goal_gain=1.0,
            device=device, n_epochs=5, batch_size=128,
            callback_factory=callback_factory, env_factory=env_factory,
        )
        runtime["num_timesteps"] = int(curriculum.summary["stages"][-1]["total_timesteps"])
        final_checkpoint = curriculum.checkpoint
        if runtime["num_timesteps"] != TRAINING["timesteps_total"]:
            raise RuntimeError(f"expected {TRAINING['timesteps_total']} steps, got {runtime['num_timesteps']}")
        if not baseline_path.is_file():
            raise RuntimeError("the untrained baseline was not saved by the first-stage callback")
        checkpoints = _checkpoint_manifest(output)
        _status(output, runtime, phase="evaluating_dev_free")
        _verify_sources(output, protocol["source_snapshot"])
        free_base = _evaluate(output, cfg, baseline_path, "free_baseline_87001_87004",
                              DEV_FREE_SEEDS, 0, 90.0, device)
        _verify_sources(output, protocol["source_snapshot"])
        free_final = _evaluate(output, cfg, final_checkpoint, "free_final_87001_87004",
                               DEV_FREE_SEEDS, 0, 90.0, device)
        free_pair = _paired_dev_conditions(
            {"baseline": free_base, "final": free_final}, DEV_FREE_SEEDS)
        free_safety = (_safe_goto_assessment(free_base, free_final)
                       if free_pair["valid"] else "invalid_initial_conditions")
        _status(output, runtime, phase="evaluating_dev_field")
        _verify_sources(output, protocol["source_snapshot"])
        field_base = _evaluate(output, cfg, baseline_path, "field_baseline_82001_82004",
                               DEV_FIELD_SEEDS, 16, 60.0, device)
        _verify_sources(output, protocol["source_snapshot"])
        field_final = _evaluate(output, cfg, final_checkpoint, "field_final_82001_82004",
                                DEV_FIELD_SEEDS, 16, 60.0, device)
        field_pair = _paired_dev_conditions(
            {"baseline": field_base, "final": field_final}, DEV_FIELD_SEEDS)
        field_safety = (_safe_goto_assessment(field_base, field_final)
                        if field_pair["valid"] else "invalid_initial_conditions")
        summary = {
            "schema": "isaac-field-fresh-foundation-development-summary-v1",
            "checkpoint_initial": str(baseline_path), "checkpoint_final": str(final_checkpoint),
            "actual_final_num_timesteps": runtime["num_timesteps"],
            "checkpoint_count": len(checkpoints),
            "free_space_pairing": free_pair, "free_space_assessment": free_safety,
            "field_pairing": field_pair, "field_assessment": field_safety,
            "evaluation_reports": [
                "evaluation_free_baseline_87001_87004.json", "evaluation_free_final_87001_87004.json",
                "evaluation_field_baseline_82001_82004.json", "evaluation_field_final_82001_82004.json",
            ],
            "claim_limit": "Development comparison only; not an architecture ablation, blind validation, or full-field navigation result.",
            "blind_830xx": "not evaluated or observed",
        }
        _json(output / "comparison_summary.json", summary)
        _status(output, runtime, "completed", phase="completed",
                summary="comparison_summary.json", exit_code=0)
        return summary
    except KeyboardInterrupt as exc:
        _status(output, runtime, "interrupted", phase="interrupted",
                exception=f"{type(exc).__name__}: {exc}", exit_code=130)
        raise
    except BaseException as exc:
        _status(output, runtime, "failed", phase="failed",
                exception=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(launch(args.output, args.device), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
