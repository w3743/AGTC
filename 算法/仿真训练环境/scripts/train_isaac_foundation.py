"""Prepare or launch a fresh or recovered Isaac Field foundation curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_isaac_exploration import (
    LEGACY_FOUNDATION_SOURCE_FILES as RUNTIME_FILES,
    _json,
    _pair_initial_conditions,
    _sha256,
    _source_manifest,
)

RUNNER = "scripts/train_isaac_foundation.py"
BASELINE = ROOT / "results/isaac_field_improvement/staged_goto_20261017_drive_stable/field_policy_after.zip"
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive"
RESUME_SCHEMA = "isaac-field-foundation-curriculum-resume-v1"
TRAIN_SEED = 69_001
FREE_SEEDS = tuple(range(87_001, 87_025))
PLANT_SEEDS = tuple(range(82_001, 82_025))
ENV_SEED = 80_003
STAGES = (
    {"name": "near", "goal_distance_m": [0.6, 1.2], "goal_angle_deg": 15.0, "timesteps": 32_768},
    {"name": "steer30", "goal_distance_m": [1.0, 2.0], "goal_angle_deg": 30.0, "timesteps": 32_768},
    {"name": "steer60", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0, "timesteps": 65_536},
    {"name": "turn90", "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0, "timesteps": 65_536},
)


def _jsonable(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_config_dict() -> dict:
    from isaac_field.task import Config

    cfg = Config(plants=0, terrain_enabled=False, max_time=20.0, damage_penalty=20.0,
                 goal_progress_budget=5.0, damage_metric="stem-contact-v2")
    return _jsonable(asdict(cfg))


def _robot_geometry_dict() -> dict:
    from isaac_field.assets import ROBOT_SPEC, robot_asset_spec_sha256

    return _jsonable({
        "source": "isaac_field.assets.ROBOT_SPEC",
        "actual_robot_spec": asdict(ROBOT_SPEC),
        "overall_width_m": float(ROBOT_SPEC.vehicle_width_m),
        "track_center_separation_m": float(ROBOT_SPEC.track_separation_m),
        "track_width_m": float(ROBOT_SPEC.wheel_width_m),
        "robot_asset_spec_sha256": robot_asset_spec_sha256(ROBOT_SPEC),
    })


def _training_protocol() -> dict:
    return {
        "algorithm": "SB3-Contrib RecurrentPPO",
        "num_envs": 4,
        "training_seeds": [TRAIN_SEED + index for index in range(len(STAGES))],
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
    }


def _resume_stage_timesteps(starting_num_timesteps: int) -> tuple[int, ...]:
    if (isinstance(starting_num_timesteps, bool)
            or int(starting_num_timesteps) != starting_num_timesteps
            or not 0 <= int(starting_num_timesteps) < STAGES[0]["timesteps"]):
        raise ValueError("resume checkpoint must be before the end of the near stage")
    return (STAGES[0]["timesteps"] - int(starting_num_timesteps),) + tuple(
        stage["timesteps"] for stage in STAGES[1:]
    )


def _source_rows(manifest: dict) -> dict[str, dict]:
    rows = manifest.get("files", [])
    by_path = {row.get("path"): row for row in rows}
    expected = set(RUNTIME_FILES) | {RUNNER}
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)) or len(rows) != len(expected) or set(by_path) != expected:
        raise ValueError("expected the complete current runtime source set plus the foundation runner")
    return by_path


def _validate_source_snapshot(
    source_root: Path,
    snapshot_root: Path,
    manifest: dict,
    *,
    compare_runner_to_live: bool,
) -> None:
    rows = _source_rows(manifest)
    for relative, row in rows.items():
        source = source_root / relative
        snapshot = snapshot_root / row["snapshot_path"]
        if not source.is_file() or not snapshot.is_file() or _sha256(snapshot) != row["sha256"]:
            raise ValueError(f"source snapshot mismatch: {relative}")
        if relative in RUNTIME_FILES or compare_runner_to_live:
            if _sha256(source) != row["sha256"]:
                raise ValueError(f"live source changed since snapshot: {relative}")


def _validate_parent_source_snapshot(parent: Path, manifest: dict) -> list[dict]:
    """Check the parent's frozen files; permit only an explicitly recorded viewer-only drift."""
    rows = _source_rows(manifest)
    differences = []
    for relative, row in rows.items():
        live, snapshot = ROOT / relative, parent / row["snapshot_path"]
        if not live.is_file() or not snapshot.is_file() or _sha256(snapshot) != row["sha256"]:
            raise ValueError(f"parent source snapshot mismatch: {relative}")
        if relative == RUNNER:
            continue
        live_sha = _sha256(live)
        if live_sha == row["sha256"]:
            continue
        if relative != "isaac_field/viewer.py":
            raise ValueError(f"parent training/runtime source changed since snapshot: {relative}")
        differences.append({
            "path": relative,
            "parent_snapshot_sha256": row["sha256"],
            "current_sha256": live_sha,
            "reason": "viewer-only UI/idle updates; training environment and policy sources remain unchanged",
        })
    return differences


def _validate_parent_protocol(protocol: dict) -> None:
    if protocol.get("schema") != "isaac-field-foundation-curriculum-v2":
        raise ValueError("resume parent is not a prepared foundation v2 trial")
    if protocol.get("status") != "prepared":
        raise ValueError("resume parent protocol is not the original prepared protocol")
    if protocol.get("config") != _field_config_dict():
        raise ValueError("resume parent field config differs from the current trial")
    if protocol.get("robot_geometry") != _robot_geometry_dict():
        raise ValueError("resume parent robot geometry differs from the current RobotAssetSpec")
    if protocol.get("training") != _training_protocol():
        raise ValueError("resume parent curriculum or training hyperparameters differ")
    baseline = protocol.get("baseline_checkpoint", {})
    baseline_path = ROOT / baseline.get("path", "")
    if (baseline.get("path") != BASELINE.relative_to(ROOT).as_posix()
            or not baseline_path.is_file() or _sha256(baseline_path) != baseline.get("sha256")
            or baseline.get("num_timesteps") != 53_248):
        raise ValueError("resume parent development baseline is missing or changed")


def _validate_resume_parent(parent: Path, checkpoint: Path) -> dict:
    parent = parent.resolve()
    try:
        parent_relative = parent.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("resume parent must be inside the project workspace") from exc
    protocol_path, status_path = parent / "protocol.json", parent / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("resume parent must contain protocol.json and run_status.json")
    protocol_bytes, status_bytes = protocol_path.read_bytes(), status_path.read_bytes()
    parent_protocol = json.loads(protocol_bytes)
    parent_status = json.loads(status_bytes)
    if parent_status.get("status") not in ("failed", "interrupted"):
        raise ValueError("resume parent must be terminal failed or interrupted")
    final_artifacts = (
        parent / "stage_04_turn90.zip", parent / "curriculum_summary.json",
        parent / "comparison_summary.json",
    )
    if any(path.exists() for path in final_artifacts) or any(
        parent_status.get(key) for key in ("summary", "checkpoint_final", "final_checkpoint")
    ):
        raise ValueError("resume parent already has a final checkpoint or comparison")
    _validate_parent_protocol(parent_protocol)
    source_differences = _validate_parent_source_snapshot(
        parent, parent_protocol.get("source_snapshot", {})
    )

    checkpoint = checkpoint.resolve()
    try:
        checkpoint_relative_to_parent = checkpoint.relative_to(parent)
        checkpoint_relative = checkpoint.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("resume checkpoint must be inside its parent trial") from exc
    if (len(checkpoint_relative_to_parent.parts) != 3
            or checkpoint_relative_to_parent.parts[0] != "stage_1_near"
            or checkpoint_relative_to_parent.parts[1] != "checkpoints"
            or checkpoint.name != "field_policy_near_8192_steps.zip"
            or checkpoint.suffix != ".zip"):
        raise ValueError(
            "resume checkpoint must be parent stage_1_near/checkpoints/field_policy_near_8192_steps.zip"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    before = checkpoint.stat()
    checkpoint_sha = _sha256(checkpoint)
    from scripts.compare_isaac_exploration import _checkpoint_steps
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError("resume checkpoint ZIP failed its integrity check")
        entry_sizes = {entry.filename: entry.file_size for entry in archive.infolist()}
        required_entries = {"data", "policy.pth", "policy.optimizer.pth"}
        if not required_entries.issubset(entry_sizes):
            raise ValueError("resume checkpoint is missing policy or Adam optimizer state")
        if entry_sizes["policy.pth"] <= 0 or entry_sizes["policy.optimizer.pth"] <= 0:
            raise ValueError("resume checkpoint policy and Adam optimizer state must be non-empty")
    num_timesteps = _checkpoint_steps(checkpoint)
    after = checkpoint.stat()
    stable = (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
              and checkpoint_sha == _sha256(checkpoint))
    if not stable:
        raise ValueError("resume checkpoint changed while it was being verified")
    if num_timesteps != 8192:
        raise ValueError(f"expected the stable 8,192-step near checkpoint, got {num_timesteps}")
    return {
        "parent": {
            "path": parent_relative,
            "protocol_sha256": _sha256(protocol_path),
            "run_status_sha256": _sha256(status_path),
            "run_status": parent_status,
        },
        "checkpoint": {
            "path": checkpoint_relative,
            "sha256": checkpoint_sha,
            "num_timesteps": num_timesteps,
            "size_bytes": before.st_size,
            "zip_integrity": True,
            "stable_during_verification": stable,
        },
        "parent_config_sha256": _json_sha256(parent_protocol["config"]),
        "parent_robot_geometry_sha256": _json_sha256(parent_protocol["robot_geometry"]),
        "parent_training_sha256": _json_sha256(parent_protocol["training"]),
        "parent_runtime_source_differences": source_differences,
    }


def _snapshot(output: Path) -> dict:
    manifest = _source_manifest(output, copy_files=True, source_files=RUNTIME_FILES)
    source = ROOT / RUNNER
    relative = f"source_snapshot/{RUNNER}"
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest["files"].append({"path": RUNNER, "snapshot_path": relative, "sha256": _sha256(source)})
    return manifest


def prepare(
    output: Path,
    *,
    resume_parent: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> dict:
    raise RuntimeError(
        "this frozen entrypoint creates Newton/SB3 protocols and is disabled for new runs; "
        "use scripts/train_isaac_foundation_fresh.py to prepare an Isaac Sim PhysX RSL-RL trial"
    )
    from isaac_field.task import Config
    from scripts.compare_isaac_exploration import _checkpoint_steps

    output = output.resolve()
    if (resume_parent is None) != (resume_checkpoint is None):
        raise ValueError("--resume-parent and --resume-checkpoint must be provided together")
    if output.exists():
        raise FileExistsError(f"refusing to reuse prepared output: {output}")
    if not BASELINE.is_file() or _checkpoint_steps(BASELINE) != 53_248:
        raise ValueError(f"expected the recorded 53,248-step development baseline: {BASELINE}")
    resume_info = None
    if resume_parent is not None:
        resume_parent = resume_parent.resolve()
        if output.parent != resume_parent:
            raise ValueError("resume output must be a new direct child of the parent trial")
        resume_info = _validate_resume_parent(resume_parent, resume_checkpoint)

    cfg = Config(**_field_config_dict())
    output.mkdir(parents=True)
    snapshot = _snapshot(output)
    expected_sources = set(RUNTIME_FILES) | {RUNNER}
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)) or {
        row.get("path") for row in snapshot.get("files", [])
    } != expected_sources:
        raise RuntimeError(
            "snapshot does not contain the complete current runtime source set plus this runner"
        )
    _validate_source_snapshot(ROOT, output, snapshot, compare_runner_to_live=True)
    if resume_info is not None:
        current_resume_info = _validate_resume_parent(resume_parent, resume_checkpoint)
        if current_resume_info != resume_info:
            raise ValueError("resume parent or checkpoint changed while preparing the child")
    training = _training_protocol()
    protocol = {
        "schema": RESUME_SCHEMA if resume_info is not None else "isaac-field-foundation-curriculum-v2",
        "status": "prepared",
        "source_snapshot": snapshot,
        "baseline_checkpoint": {
            "path": BASELINE.relative_to(ROOT).as_posix(),
            "sha256": _sha256(BASELINE),
            "num_timesteps": 53_248,
        },
        "initialization": (
            "RecurrentPPO warm-start from the recorded parent checkpoint"
            if resume_info is not None else "fresh RecurrentPPO; no checkpoint warm-start"
        ),
        "config": asdict(cfg),
        "robot_geometry": _robot_geometry_dict(),
        "training": training,
        "evaluation": {
            "device": "selected at launch",
            "environment_seed": ENV_SEED,
            "distance_cost": 0.0,
            "deterministic": True,
            "record_trajectories": True,
            "free_space": {"plants": 0, "seeds": list(FREE_SEEDS),
                           "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 90.0,
                           "baseline_and_final_paired": True},
            "plants": {"plants": 16, "seeds": list(PLANT_SEEDS),
                       "goal_distance_m": [1.0, 3.0], "goal_angle_deg": 60.0,
                       "baseline_and_final_paired": True},
            "dataset_role": "development, not blind validation",
            "claim_limit": "Flat local tasks only; free-space success does not establish reliable field navigation or search.",
        },
    }
    if resume_info is not None:
        timesteps_this_run = _resume_stage_timesteps(resume_info["checkpoint"]["num_timesteps"])
        recovery_manifest = {
            "schema": "isaac-field-foundation-recovery-manifest-v1",
            **resume_info,
            "training_plan": {
                "starting_num_timesteps": resume_info["checkpoint"]["num_timesteps"],
                "canonical_stage_timesteps": [int(row["timesteps"]) for row in STAGES],
                "timesteps_per_stage_this_run": list(timesteps_this_run),
                "additional_timesteps": sum(timesteps_this_run),
                "expected_final_num_timesteps": sum(timesteps_this_run)
                + resume_info["checkpoint"]["num_timesteps"],
                "training_seeds": list(training["training_seeds"]),
            },
            "continuation_semantics": {
                "weights_and_adam_optimizer_state": "loaded from the verified checkpoint and continued",
                "global_timestep": "retained by RecurrentPPO.learn(reset_num_timesteps=False)",
                "environment_rng_observations_recurrent_state_and_partial_rollout": (
                    "reinitialized in a new environment; per-stage RNG is reseeded from the canonical stage seed"
                ),
                "bitwise_resume": False,
            },
        }
        _json(output / "recovery_manifest.json", recovery_manifest)
        protocol["recovery_manifest"] = {
            "path": "recovery_manifest.json",
            "sha256": _sha256(output / "recovery_manifest.json"),
        }
    _json(output / "protocol.json", protocol)
    _json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": _sha256(output / "protocol.json")
    })
    return {"output": str(output), "snapshot_files": len(protocol["source_snapshot"]["files"]),
            "baseline_sha256": protocol["baseline_checkpoint"]["sha256"],
            "resume": resume_info is not None}


def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --launch")
    protocol, status = (json.loads(path.read_text(encoding="utf-8")) for path in (protocol_path, status_path))
    if status.get("status") != "prepared" or status.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("run is not an unchanged prepared experiment")
    schema = protocol.get("schema")
    if (schema not in ("isaac-field-foundation-curriculum-v2", RESUME_SCHEMA)
            or protocol.get("config", {}).get("damage_metric") != "stem-contact-v2"):
        raise ValueError("runner requires the foundation protocol with damage_metric='stem-contact-v2'")
    if protocol.get("config") != _field_config_dict():
        raise ValueError("prepared field configuration differs from the current foundation protocol")
    if protocol.get("robot_geometry") != _robot_geometry_dict():
        raise ValueError("prepared protocol robot geometry or asset specification changed")
    training = protocol.get("training", {})
    if training != _training_protocol():
        raise ValueError("prepared curriculum stages or training seeds differ from this runner")
    _validate_source_snapshot(ROOT, output, protocol.get("source_snapshot", {}), compare_runner_to_live=True)
    baseline = ROOT / protocol["baseline_checkpoint"]["path"]
    if (protocol["baseline_checkpoint"]["path"] != BASELINE.relative_to(ROOT).as_posix()
            or protocol["baseline_checkpoint"].get("num_timesteps") != 53_248
            or not baseline.is_file()
            or _sha256(baseline) != protocol["baseline_checkpoint"]["sha256"]):
        raise ValueError("development baseline checkpoint changed")
    if schema == RESUME_SCHEMA:
        recovery_ref = protocol.get("recovery_manifest", {})
        recovery_path = output / recovery_ref.get("path", "")
        if (recovery_ref.get("path") != "recovery_manifest.json"
                or not recovery_path.is_file()
                or _sha256(recovery_path) != recovery_ref.get("sha256")):
            raise ValueError("recovery manifest changed after prepare")
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        plan = recovery.get("training_plan", {})
        start = recovery.get("checkpoint", {}).get("num_timesteps")
        if start != 8192:
            raise ValueError("recovery checkpoint must start at 8,192 timesteps")
        expected_stage_steps = _resume_stage_timesteps(start)
        if (recovery.get("schema") != "isaac-field-foundation-recovery-manifest-v1"
                or plan.get("starting_num_timesteps") != start
                or plan.get("canonical_stage_timesteps") != [int(row["timesteps"]) for row in STAGES]
                or plan.get("timesteps_per_stage_this_run") != list(expected_stage_steps)
                or plan.get("additional_timesteps") != sum(expected_stage_steps)
                or plan.get("expected_final_num_timesteps") != start + sum(expected_stage_steps)
                or plan.get("training_seeds") != training["training_seeds"]
                or sum(expected_stage_steps) != 188_416
                or start + sum(expected_stage_steps) != 196_608):
            raise ValueError("recovery timestep plan does not match the verified 8,192-step resume")
        if output.parent.resolve() != ROOT / recovery.get("parent", {}).get("path", ""):
            raise ValueError("resume output is no longer a direct child of its recorded parent")
        current = _validate_resume_parent(
            ROOT / recovery["parent"]["path"], ROOT / recovery["checkpoint"]["path"]
        )
        for key in ("parent", "checkpoint", "parent_config_sha256",
                    "parent_robot_geometry_sha256", "parent_training_sha256",
                    "parent_runtime_source_differences"):
            if recovery.get(key) != current.get(key):
                raise ValueError(f"resume parent or checkpoint changed after prepare: {key}")
        semantics = recovery.get("continuation_semantics", {})
        if (semantics.get("weights_and_adam_optimizer_state") != "loaded from the verified checkpoint and continued"
                or semantics.get("global_timestep") != "retained by RecurrentPPO.learn(reset_num_timesteps=False)"
                or semantics.get("bitwise_resume") is not False):
            raise ValueError("recovery continuation semantics are missing or changed")
    elif "recovery_manifest" in protocol:
        raise ValueError("fresh protocol must not contain a recovery manifest")
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
    raise RuntimeError(
        "this frozen entrypoint resumes/trains SB3 .zip checkpoints and cannot create a PhysX RSL run; "
        "use scripts/train_isaac_foundation_fresh.py with a new protocol"
    )
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
            _json(output / "run_status.json", {"status": "running", "phase": "creating_training_env"})
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
        recovery = None
        checkpoint = None
        stage_timesteps = tuple(spec["timesteps"] for spec in training["stages"])
        if protocol.get("schema") == RESUME_SCHEMA:
            recovery_path = output / protocol["recovery_manifest"]["path"]
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            checkpoint = ROOT / recovery["checkpoint"]["path"]
            stage_timesteps = tuple(recovery["training_plan"]["timesteps_per_stage_this_run"])
        curriculum = train_curriculum(
            cfg, output, checkpoint=checkpoint,
            timesteps_per_stage=stage_timesteps,
            training_seed=TRAIN_SEED, heldout_seeds=FREE_SEEDS, stages=stages,
            evaluate_before_after=False, goal_gain=100.0, device=device,
            n_epochs=5, batch_size=128, callback_factory=callback_factory,
            env_factory=env_factory,
        )
        final_checkpoint = curriculum.checkpoint
        if recovery is not None:
            from scripts.compare_isaac_exploration import _checkpoint_steps

            actual_final_timesteps = _checkpoint_steps(final_checkpoint)
            expected_final_timesteps = recovery["training_plan"]["expected_final_num_timesteps"]
            if actual_final_timesteps != expected_final_timesteps:
                raise RuntimeError(
                    f"recovery ended at {actual_final_timesteps} steps; expected {expected_final_timesteps}"
                )
        _json(output / "training_progress.json", {
            name: {"aggregate_rows": callback.log_rows,
                  "last_completed_episodes": callback.recent_episodes[-10:]}
            for name, callback in progress_by_stage.items()
        })
        baseline_path = ROOT / protocol["baseline_checkpoint"]["path"]
        _json(output / "run_status.json", {"status": "running", "phase": "evaluating_free_space"})
        free_cfg = replace(cfg, plants=0)
        free_baseline = _evaluate(output, free_cfg, baseline_path, "free_baseline_870xx", FREE_SEEDS,
                                   math.pi / 2.0, device)
        free_after = _evaluate(output, free_cfg, final_checkpoint, "free_after_870xx", FREE_SEEDS,
                               math.pi / 2.0, device)
        free_pairing = _pair_initial_conditions({"baseline": free_baseline, "after": free_after})
        free_assessment = (
            _safe_goto_assessment(free_baseline, free_after)
            if free_pairing["valid"] else "invalid_initial_conditions"
        )

        field_cfg = replace(cfg, plants=16)
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
            "recovery_manifest": protocol.get("recovery_manifest"),
            "actual_final_num_timesteps": (
                _checkpoint_steps(final_checkpoint) if recovery is not None else None
            ),
            "free_space_baseline_summary": free_baseline["summary"],
            "free_space_after_summary": free_after["summary"],
            "free_space_pairing": free_pairing,
            "free_space_assessment": free_assessment,
            "field_baseline_summary": before["summary"],
            "field_after_summary": after["summary"],
            "field_pairing": pairing,
            "field_assessment": assessment,
            "drive_metadata": {
                "baseline_matches_runtime": before["drive_config"] == before["checkpoint_drive_config"],
                "final_matches_runtime": after["drive_config"] == after["checkpoint_drive_config"],
                "interpretation": "The baseline is re-evaluated on current physics; a legacy drive-metadata mismatch is recorded, not reused as an old report.",
            },
            "evaluation_reports": ["evaluation_free_baseline_870xx.json",
                                   "evaluation_free_after_870xx.json",
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume-parent", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = (
            args.resume_parent / "resume_8192"
            if args.prepare and args.resume_parent is not None else DEFAULT_OUTPUT
        )
    if args.launch and (args.resume_parent is not None or args.resume_checkpoint is not None):
        parser.error("resume options are accepted only by --prepare; --launch reads the prepared protocol")
    if (args.resume_parent is None) != (args.resume_checkpoint is None):
        parser.error("--resume-parent and --resume-checkpoint must be provided together")
    result = (
        prepare(output, resume_parent=args.resume_parent, resume_checkpoint=args.resume_checkpoint)
        if args.prepare else launch(output, args.device)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
