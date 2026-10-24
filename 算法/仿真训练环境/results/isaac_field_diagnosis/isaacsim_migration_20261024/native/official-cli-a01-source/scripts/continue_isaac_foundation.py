"""Continue the verified coupled-track foundation checkpoint in an isolated trial."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_isaac_foundation_fresh as fresh

PARENT = ROOT / "results/isaac_field_improvement/fresh_coupled_tracks_20261020"
DEFAULT_OUTPUT = ROOT / "results/isaac_field_improvement/continue_coupled_tracks_20261020"
RUNNER = "scripts/continue_isaac_foundation.py"
SCHEMA = "isaac-field-coupled-track-continuation-v1"
PARENT_PROTOCOL_SHA256 = "2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6"
PARENT_CHECKPOINT_SHA256 = "6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5"
PARENT_CPU_REPORT_SHA256 = "166283ba98336b371c421814e6fc888e4e4fb2c9defab2f3f32eb3158b8561cd"
PARENT_TIMESTEPS = 196_608
ADDED_TIMESTEPS = 196_608
FINAL_TIMESTEPS = PARENT_TIMESTEPS + ADDED_TIMESTEPS
TRAIN_SEED = 69_005
KNOWN_PARENT_UI_DRIFT = {
    "isaac_field/viewer.py",
    "isaac_field/visualization/THIRD_PARTY_NOTICES.md",
    "isaac_field/visualization/network_2d.html",
    "isaac_field/visualization/network_2d.py",
}
EXPECTED_ADAM = {
    "class": "Adam", "state_parameter_count": 37, "step_entry_count": 37,
    "min_step": 7607, "max_step": 7607, "all_steps_positive": True,
    "floating_state_tensor_count": 111, "all_finite": True, "empty": False,
}


def _write_json(path: Path, value: dict) -> None:
    fresh._json(path, value)


def _current_source_manifest(output: Path, *, copy_files: bool) -> dict:
    manifest = fresh._source_manifest(output, copy_files=copy_files)
    source = ROOT / RUNNER
    snapshot_path = f"source_snapshot/{RUNNER}"
    digest = fresh._sha256(source)
    if copy_files:
        target = output / snapshot_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest["files"].append({"path": RUNNER, "snapshot_path": snapshot_path, "sha256": digest})
    manifest["files"].sort(key=lambda row: row["path"])
    return manifest


def _verify_child_sources(output: Path, manifest: dict) -> None:
    rows = manifest.get("files", [])
    base_paths = set(fresh._source_paths())
    if len(rows) != len(base_paths) + 1 or {row.get("path") for row in rows} != base_paths | {RUNNER}:
        raise ValueError("continuation frozen source manifest must contain exactly 43 sources")
    fresh._verify_sources(output, {"files": [row for row in rows if row["path"] != RUNNER]})
    own = next(row for row in rows if row["path"] == RUNNER)
    if fresh._sha256(ROOT / RUNNER) != own["sha256"] or fresh._sha256(output / own["snapshot_path"]) != own["sha256"]:
        raise ValueError("continuation runner differs from its frozen source copy")


def _parent_audit(parent: Path) -> tuple[dict, Path, dict]:
    if parent.resolve() != PARENT.resolve():
        raise ValueError(f"only the verified parent trial may be continued: {PARENT}")
    protocol_path = parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if fresh._protocol_sha(protocol_path) != PARENT_PROTOCOL_SHA256:
        raise ValueError("parent protocol SHA does not match the verified coupled-track trial")
    status = json.loads((parent / "run_status.json").read_text(encoding="utf-8"))
    process_exit = json.loads((parent / "process_exit.json").read_text(encoding="utf-8"))
    cpu_report_path = parent / "final_cpu_validation/final_cpu_validation.json"
    cpu_report = json.loads(cpu_report_path.read_text(encoding="utf-8"))
    cpu_exit = json.loads((cpu_report_path.parent / "final_cpu_validation.exit.json").read_text(encoding="utf-8"))
    if (status.get("status") != "completed" or status.get("exit_code") != 0
            or status.get("num_timesteps") != PARENT_TIMESTEPS
            or process_exit.get("exit_code") != 0
            or (parent / "python.exit_code").read_text(encoding="utf-8").strip() != "0"):
        raise ValueError("parent training lacks verified completion and real zero exit")
    if (fresh._sha256(cpu_report_path) != PARENT_CPU_REPORT_SHA256
            or cpu_exit.get("actual_exit_code") != 0
            or cpu_exit.get("report_sha256") != PARENT_CPU_REPORT_SHA256
            or cpu_exit.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
            or cpu_exit.get("protocol_sha256") != PARENT_PROTOCOL_SHA256):
        raise ValueError("parent CPU validation/real-exit evidence changed")
    checks = cpu_report.get("checks", {})
    required_checks = (
        "protocol_sha_matches_expected_and_stable", "final_checkpoint_sha_stable_and_matches_manifest",
        "final_timestep_count_matches_protocol", "baseline_and_checkpoint_metadata_match_protocol",
        "42_live_sources_and_snapshot_copies_match", "all_weights_finite_and_shapes_match",
        "adam_optimizer_state_finite_with_positive_steps", "loaded_cpu_only",
        "live_runtime_contract_matches_v7",
    )
    if (cpu_report.get("status") != "passed" or cpu_report.get("exit_code") != 0
            or cpu_report.get("checkpoint", {}).get("optimizer") != EXPECTED_ADAM
            or not all(checks.get(name) is True for name in required_checks)):
        raise ValueError("parent CPU checkpoint/Adam validation does not satisfy the pinned audit")

    checkpoint = parent / "stage_04_turn90.zip"
    if fresh._sha256(checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent final checkpoint SHA changed")
    manifest = json.loads((parent / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    row = next((item for item in manifest.get("checkpoints", []) if item.get("path") == checkpoint.name), None)
    if (row is None or row.get("sha256") != PARENT_CHECKPOINT_SHA256
            or row.get("num_timesteps") != PARENT_TIMESTEPS
            or row.get("adam_optimizer_state_present") is not True):
        raise ValueError("parent checkpoint manifest does not confirm the final optimizer checkpoint")
    with zipfile.ZipFile(checkpoint) as archive:
        names = set(archive.namelist())
        data = json.loads(archive.read("data"))
    if ("policy.pth" not in names or "policy.optimizer.pth" not in names
            or data.get("num_timesteps") != PARENT_TIMESTEPS
            or data.get("field_action_config") != fresh._action_config()
            or data.get("field_encoder_config") != fresh._field_encoder_config()):
        raise ValueError("parent checkpoint lacks matching model, optimizer, timestep, or policy metadata")
    if protocol.get("schema") != fresh.SCHEMA or protocol.get("training") != fresh.TRAINING:
        raise ValueError("parent protocol is not the pinned fresh foundation curriculum")
    if protocol.get("action_config") != fresh._action_config() or protocol.get("config") != fresh._config_dict():
        raise ValueError("parent action/task config differs from the current coupled-track contract")

    rows = protocol.get("source_snapshot", {}).get("files", [])
    expected = set(fresh._source_paths())
    if {item.get("path") for item in rows} != expected or len(rows) != 42:
        raise ValueError("parent protocol must contain the verified 42-source snapshot")
    drift = set()
    for item in rows:
        source = ROOT / item["path"]
        snapshot = parent / item["snapshot_path"]
        if not snapshot.is_file() or fresh._sha256(snapshot) != item["sha256"]:
            raise ValueError(f"parent snapshot hash mismatch: {item['path']}")
        if not source.is_file() or fresh._sha256(source) != item["sha256"]:
            drift.add(item["path"])
    if drift != KNOWN_PARENT_UI_DRIFT:
        raise ValueError(f"unexpected current source drift from verified parent: {sorted(drift)}")
    return protocol, checkpoint, manifest


def prepare(parent: Path, output: Path, max_time: float = 20.0) -> dict:
    raise RuntimeError(
        "this frozen continuation is pinned to an SB3/Newton .zip parent; it cannot be relabeled as PhysX RSL-RL. "
        "Use an explicit weights-transfer workflow from a compatible RSL .pt checkpoint."
    )
    parent, output = parent.resolve(), output.resolve()
    if max_time not in (20.0, 60.0):
        raise ValueError("max_time must be 20 or 60 seconds")
    if output.exists():
        raise FileExistsError(f"refusing to reuse continuation output: {output}")
    parent_protocol, parent_checkpoint, _ = _parent_audit(parent)
    current_config = fresh._config_dict()
    for key, value in parent_protocol["config"].items():
        if key != "max_time" and current_config.get(key) != value:
            raise ValueError(f"current task config changed from parent: {key}")
    config = dict(parent_protocol["config"], max_time=float(max_time))

    snapshot = _current_source_manifest(output, copy_files=False)
    output.mkdir(parents=True)
    snapshot = _current_source_manifest(output, copy_files=True)
    baseline = output / "parent_final_baseline.zip"
    shutil.copyfile(parent_checkpoint, baseline)
    if fresh._sha256(baseline) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("copied parent baseline differs from its verified checkpoint")

    training = dict(parent_protocol["training"])
    training.update({
        "training_seed": TRAIN_SEED,
        "stage_seeds": [TRAIN_SEED + i for i in range(len(fresh.STAGES))],
        "stages": [dict(item) for item in fresh.STAGES],
        "timesteps_total": ADDED_TIMESTEPS,
        "initialization": "continue parent RecurrentPPO weights and Adam; reset environment/RNG/rollout state",
        "checkpoint_interval_transitions": 8192,
    })
    protocol = {
        "schema": SCHEMA, "status": "prepared", "parent": {
            "path": str(parent), "protocol_sha256": PARENT_PROTOCOL_SHA256,
            "checkpoint": parent_checkpoint.name, "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "num_timesteps": PARENT_TIMESTEPS, "optimizer": EXPECTED_ADAM,
        },
        "architecture": parent_protocol["architecture"], "action_config": parent_protocol["action_config"],
        "runtime_contract": parent_protocol["runtime_contract"], "config": config,
        "robot_geometry": parent_protocol["robot_geometry"], "training": training,
        "initialization": {
            "baseline_path": baseline.name, "baseline_sha256": PARENT_CHECKPOINT_SHA256,
            "baseline_num_timesteps": PARENT_TIMESTEPS, "loads_policy_weights": True,
            "loads_adam_optimizer_state": True, "weight_transfer": False,
            "expected_final_num_timesteps": FINAL_TIMESTEPS,
            "continuation_semantics": {
                "exact_policy_and_adam_state_from_parent": True,
                "environment_rng_lstm_state_and_rollout_buffer_reinitialized": True,
                "bitwise_resume": False,
                "max_time_note": ("20-second task input and episode limit retained" if max_time == 20.0 else
                                  "60-second task input and episode limit applied to both parent baseline and child final; episode_time_ratio changes"),
            },
        },
        "evaluation": parent_protocol["evaluation"],
        "runtime": {"versions": fresh._package_versions(), "isaaclab_commit": fresh._isaac_commit()},
        "source_snapshot": snapshot,
    }
    _write_json(output / "protocol.json", protocol)
    digest = fresh._protocol_sha(output / "protocol.json")
    _write_json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": digest,
        "num_timesteps": PARENT_TIMESTEPS, "exit_code": None,
    })
    return {"output": str(output), "protocol_sha256": digest, "source_count": len(snapshot["files"]),
            "starting_num_timesteps": PARENT_TIMESTEPS, "additional_timesteps": ADDED_TIMESTEPS,
            "expected_final_num_timesteps": FINAL_TIMESTEPS}


def preflight(output: Path) -> dict:
    output = output.resolve()
    protocol_path = output / "protocol.json"
    status_path = output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --preflight or --launch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    digest = fresh._protocol_sha(protocol_path)
    if (protocol.get("schema") != SCHEMA or status.get("status") != "prepared"
            or status.get("protocol_sha256") != digest):
        raise ValueError("continuation must be an unchanged, not-yet-launched prepared trial")
    if (protocol.get("parent", {}).get("protocol_sha256") != PARENT_PROTOCOL_SHA256
            or protocol.get("parent", {}).get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
            or protocol.get("initialization", {}).get("expected_final_num_timesteps") != FINAL_TIMESTEPS
            or protocol.get("training", {}).get("stage_seeds") != [69005, 69006, 69007, 69008]
            or protocol.get("config", {}).get("max_time") not in (20.0, 60.0)):
        raise ValueError("continuation protocol differs from the pinned training contract")
    _verify_child_sources(output, protocol.get("source_snapshot", {}))
    parent = Path(protocol["parent"]["path"])
    _parent_audit(parent)
    baseline = output / protocol["initialization"]["baseline_path"]
    if fresh._sha256(baseline) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("child baseline is not byte-identical to the verified parent final")
    return protocol


def _load_checkpoint_member(checkpoint: Path, name: str):
    import torch

    with zipfile.ZipFile(checkpoint) as archive:
        data = archive.read(name)
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(io.BytesIO(data), map_location="cpu")


def _tree_tensors(value, prefix=""):
    import torch

    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tree_tensors(item, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tree_tensors(item, f"{prefix}/{index}")


def _tree_equal(left, right) -> bool:
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _tensor_digest(tree) -> tuple[str, int, int]:
    digest, tensors, elements = hashlib.sha256(), 0, 0
    for name, tensor in _tree_tensors(tree):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not bool(value.isfinite().all()):
            raise ValueError(f"non-finite checkpoint tensor: {name}")
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
        tensors += 1; elements += value.numel()
    return digest.hexdigest(), tensors, elements


def _adam_audit(state: dict) -> dict:
    states = state.get("state", {})
    steps = []
    floating_tensors = 0
    for _, tensor in _tree_tensors(states):
        if tensor.is_floating_point():
            floating_tensors += 1
            if not bool(tensor.isfinite().all()):
                raise ValueError("parent/live Adam state has non-finite values")
    for item in states.values():
        step = item.get("step")
        if step is not None:
            steps.append(int(step.detach().cpu().item() if hasattr(step, "detach") else step))
    report = {
        "class": "Adam", "state_parameter_count": len(states), "step_entry_count": len(steps),
        "min_step": min(steps) if steps else None, "max_step": max(steps) if steps else None,
        "all_steps_positive": bool(steps) and all(step > 0 for step in steps),
        "floating_state_tensor_count": floating_tensors, "all_finite": True, "empty": not states,
    }
    if report != EXPECTED_ADAM:
        raise ValueError(f"live Adam state differs from the verified parent audit: {report}")
    return report


def _initialization_audit(model, env, protocol: dict, baseline: Path, output: Path) -> dict:
    if int(model.num_timesteps) != PARENT_TIMESTEPS:
        raise ValueError(f"resumed model starts at {model.num_timesteps}, expected {PARENT_TIMESTEPS}")
    if fresh._sha256(baseline) != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent baseline changed after launch preflight")
    live = fresh._live_environment_contract(env)
    from isaac_field.policy import _validate_model_action_contract

    _validate_model_action_contract(model)
    encoder = fresh._field_encoder_config()
    if (model.field_action_config != protocol["action_config"]
            or model.field_encoder_config != encoder
            or int(model.policy.features_extractor.features_dim) != 192
            or int(model.policy.lstm_actor.hidden_size) != 128):
        raise ValueError("resumed model architecture/action contract differs from the parent")
    expected_policy = _load_checkpoint_member(baseline, "policy.pth")
    expected_adam = _load_checkpoint_member(baseline, "policy.optimizer.pth")
    policy = model.policy.state_dict()
    adam = model.policy.optimizer.state_dict()
    if type(model.policy.optimizer).__name__ != "Adam":
        raise ValueError(f"resumed optimizer is not Adam: {type(model.policy.optimizer).__name__}")
    policy_hash, tensor_count, elements = _tensor_digest(policy)
    _tensor_digest(expected_policy)
    if not _tree_equal(policy, expected_policy):
        raise ValueError("resumed policy weights differ from the byte-verified parent checkpoint")
    adam_hash, adam_tensor_count, adam_elements = _tensor_digest(adam)
    _tensor_digest(expected_adam)
    if not _tree_equal(adam, expected_adam):
        raise ValueError("resumed Adam optimizer state differs from the byte-verified parent checkpoint")
    adam_report = _adam_audit(adam)
    result = {
        "status": "passed", "checkpoint": baseline.name, "checkpoint_sha256": fresh._sha256(baseline),
        "parent_protocol_sha256": protocol["parent"]["protocol_sha256"],
        "num_timesteps": int(model.num_timesteps), "policy_weights_equal_parent": True,
        "policy_state_sha256": policy_hash, "policy_tensor_count": tensor_count,
        "policy_element_count": elements, "adam_state_equal_parent": True,
        "adam_state_sha256": adam_hash, "adam_tensor_count": adam_tensor_count,
        "adam_element_count": adam_elements, "adam": adam_report,
        "live_drive_config": live["drive_config"], "live_observation_config": live["observation_config"],
        "continuation_semantics": protocol["initialization"]["continuation_semantics"],
        "reference_load_rng_side_effect": False,
    }
    _write_json(output / "initialization.json", result)
    return result


def _status(output: Path, runtime: dict, status: str = "running", **extra) -> None:
    _write_json(output / "run_status.json", {
        "status": status, "protocol_sha256": runtime["protocol_sha256"],
        "num_timesteps": int(runtime.get("num_timesteps", PARENT_TIMESTEPS)),
        "updated_at": datetime.now(timezone.utc).isoformat(), **extra,
    })


def launch(output: Path, device: str = "cuda:0") -> dict:
    raise RuntimeError(
        "this frozen continuation requires the Newton/SB3 optimizer and cannot resume with the PhysX RSL runner"
    )
    protocol = preflight(output)
    output = output.resolve()
    lock = output / "launch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
    except FileExistsError as exc:
        raise ValueError("continuation has already been launched or claimed") from exc
    protocol = preflight(output)
    runtime = {"protocol_sha256": fresh._protocol_sha(output / "protocol.json"),
               "num_timesteps": PARENT_TIMESTEPS}
    _status(output, runtime, phase="initializing")
    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from isaac_field.env import make_env
        from isaac_field.learning import CurriculumStage, train_curriculum
        from isaac_field.task import Config
        from scripts.improve_isaac_policy import _make_progress_callback, _safe_goto_assessment

        cfg = Config(**protocol["config"])
        stages = tuple(CurriculumStage(row["name"], "goto", tuple(row["goal_distance_m"]),
                                       math.radians(row["goal_angle_deg"]))
                       for row in protocol["training"]["stages"])
        latest, assets, progress_by_stage = {"env": None}, [], {}
        baseline = output / protocol["initialization"]["baseline_path"]

        def env_factory(**kwargs):
            _verify_child_sources(output, protocol["source_snapshot"])
            kwargs["num_envs"] = 4 if kwargs["auto_reset"] else 1
            env = make_env(**kwargs)
            try:
                fresh._live_environment_contract(env)
                assets.append({"environment_index": len(assets), "seed": kwargs.get("seed"),
                               "assets": fresh._runtime_assets(env)})
                _write_json(output / "runtime_assets.json", {"environments": assets})
            except BaseException:
                env.close(); raise
            latest["env"] = env
            return env

        def callback_factory(stage):
            index = next(i for i, item in enumerate(stages) if item.name == stage.name)
            stage_dir = output / f"stage_{index + 1:02d}_{stage.name}"
            checkpoint_dir = stage_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _verify_child_sources(output, protocol["source_snapshot"])
            progress = _make_progress_callback(4, latest["env"], interval=8192)
            progress_by_stage[stage.name] = progress
            checkpoint = CheckpointCallback(save_freq=8192 // 4, save_path=str(checkpoint_dir),
                                            name_prefix=f"field_policy_{stage.name}", verbose=0)

            class PersistStage(BaseCallback):
                def _on_training_start(self):
                    self.stage_start = int(self.model.num_timesteps)
                    runtime["num_timesteps"] = self.stage_start
                    _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    if index == 0:
                        _initialization_audit(self.model, latest["env"], protocol, baseline, output)

                def _on_step(self):
                    runtime["num_timesteps"] = int(self.num_timesteps)
                    if (runtime["num_timesteps"] - self.stage_start) % 8192 == 0:
                        fresh._checkpoint_manifest(output)
                        _status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    return True

                def _on_training_end(self):
                    progress._refresh_episodes()
                    runtime["num_timesteps"] = int(self.model.num_timesteps)
                    _write_json(stage_dir / "progress.json", {
                        "stage": stage.name, "start_num_timesteps": self.stage_start,
                        "end_num_timesteps": runtime["num_timesteps"],
                        "aggregate_rows": progress.log_rows,
                        "last_completed_episodes": progress.recent_episodes[-10:],
                    })
                    _status(output, runtime, phase=f"completed_{stage.name}", stage=stage.name)

            return CallbackList([checkpoint, progress, PersistStage()])

        result = train_curriculum(
            cfg, output, checkpoint=baseline,
            timesteps_per_stage=tuple(row["timesteps"] for row in protocol["training"]["stages"]),
            training_seed=TRAIN_SEED, heldout_seeds=tuple(fresh.DEV_FIELD_SEEDS), stages=stages,
            evaluate_before_after=False, goal_gain=1.0, device=device,
            n_epochs=5, batch_size=128, callback_factory=callback_factory, env_factory=env_factory,
        )
        runtime["num_timesteps"] = int(result.summary["stages"][-1]["total_timesteps"])
        final = result.checkpoint
        if runtime["num_timesteps"] != FINAL_TIMESTEPS:
            raise RuntimeError(f"expected cumulative {FINAL_TIMESTEPS} steps, got {runtime['num_timesteps']}")
        if not (output / "initialization.json").is_file():
            raise RuntimeError("first-stage resumed model initialization audit was not recorded")
        checkpoints = fresh._checkpoint_manifest(output)

        evaluation = protocol["evaluation"]
        free_seeds = tuple(evaluation["free_space"]["seeds"])
        field_seeds = tuple(evaluation["field"]["seeds"])
        _status(output, runtime, phase="evaluating_dev_free")
        _verify_child_sources(output, protocol["source_snapshot"])
        free_base = fresh._evaluate(output, cfg, baseline, "free_baseline_87001_87004", free_seeds, 0, 90.0, device)
        _verify_child_sources(output, protocol["source_snapshot"])
        free_final = fresh._evaluate(output, cfg, final, "free_final_87001_87004", free_seeds, 0, 90.0, device)
        free_pair = fresh._paired_dev_conditions({"baseline": free_base, "final": free_final}, free_seeds)
        free_assessment = _safe_goto_assessment(free_base, free_final) if free_pair["valid"] else "invalid_initial_conditions"
        _status(output, runtime, phase="evaluating_dev_field")
        _verify_child_sources(output, protocol["source_snapshot"])
        field_base = fresh._evaluate(output, cfg, baseline, "field_baseline_82001_82004", field_seeds, 16, 60.0, device)
        _verify_child_sources(output, protocol["source_snapshot"])
        field_final = fresh._evaluate(output, cfg, final, "field_final_82001_82004", field_seeds, 16, 60.0, device)
        field_pair = fresh._paired_dev_conditions({"baseline": field_base, "final": field_final}, field_seeds)
        field_assessment = _safe_goto_assessment(field_base, field_final) if field_pair["valid"] else "invalid_initial_conditions"
        summary = {
            "schema": SCHEMA, "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "checkpoint_initial": str(baseline), "checkpoint_final": str(final),
            "starting_num_timesteps": PARENT_TIMESTEPS, "actual_final_num_timesteps": runtime["num_timesteps"],
            "checkpoint_count": len(checkpoints), "free_space_pairing": free_pair,
            "free_space_assessment": free_assessment, "field_pairing": field_pair,
            "field_assessment": field_assessment,
            "evaluation_reports": ["evaluation_free_baseline_87001_87004.json",
                                   "evaluation_free_final_87001_87004.json",
                                   "evaluation_field_baseline_82001_82004.json",
                                   "evaluation_field_final_82001_82004.json"],
            "claim_limit": "Development paired comparison only; not blind validation or full-field navigation evidence.",
            "blind_830xx": "not evaluated or observed",
        }
        _write_json(output / "comparison_summary.json", summary)
        _status(output, runtime, "completed", phase="completed", summary="comparison_summary.json", exit_code=0)
        return summary
    except KeyboardInterrupt as exc:
        _status(output, runtime, "interrupted", phase="interrupted", exception=f"{type(exc).__name__}: {exc}", exit_code=130)
        raise
    except BaseException as exc:
        _status(output, runtime, "failed", phase="failed", exception=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-time", type=float, choices=(20.0, 60.0), default=20.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            result = prepare(args.parent, args.output, args.max_time)
        elif args.preflight:
            protocol = preflight(args.output)
            result = {"status": "ready", "protocol_sha256": fresh._protocol_sha(args.output / "protocol.json"),
                      "source_count": len(protocol["source_snapshot"]["files"])}
        else:
            result = launch(args.output, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
