"""Prepare and run an auditable continuation of the fast-stop rough-field trial.

The runner only consumes the pinned 57,344-step parent checkpoint. It keeps the
parent trial immutable and writes every continuation artifact to a new child.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_isaac_foundation_fresh as fresh

PARENT = ROOT / "results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020"
PARENT_RELATIVE = PARENT.relative_to(ROOT).as_posix()
PARENT_PROTOCOL_SHA256 = "3ca58fcc05b9d7ac34798f836704c91b5ec608d62f7424c3897ed291451a620c"
PARENT_CHECKPOINT_RELATIVE = (
    PARENT_RELATIVE
    + "/stage_1_global_random/checkpoints/field_policy_global_random_57344_steps.zip"
)
PARENT_CHECKPOINT_SHA256 = "db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a"
DEFAULT_OUTPUT = ROOT / (
    "results/isaac_field_improvement/"
    "random_global_rough_fast_stop_lstm512_resume_20261021_attempt2"
)
RUNNER = "scripts/resume_fast_stop.py"
SCHEMA = "isaac-field-fast-stop-resume-v1"
PROFILE = fresh.PROFILE_RANDOM_ROUGH_FAST_STOP
SOURCE_COUNT = 45
START_TIMESTEPS = 57_344
FINAL_TIMESTEPS = 196_608
ADDITIONAL_TIMESTEPS = FINAL_TIMESTEPS - START_TIMESTEPS
CHECKPOINT_INTERVAL = 8_192
TRAIN_SEED = 69_009
REQUIRED_GATE_SOURCE = "scripts/lassc_consumer.py"
FIRST_PERIODIC_TIMESTEPS = START_TIMESTEPS + CHECKPOINT_INTERVAL
ALLOWED_EXTERNAL_DEPENDENCIES = (
    "../视觉算法/软件/数值深度接口候选/app/depth_adapter.py",
    "../视觉算法/软件/数值深度接口候选/app/depth_wire.py",
    "../视觉算法/深度接口契约.md",
)
CONTINUATION_INITIALIZATION = (
    "continue verified RecurrentPPO policy and Adam from the parent checkpoint"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _workspace_file(value: str | Path, *, label: str) -> tuple[Path, str]:
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError(f"{label} must not traverse outside the project workspace")
    resolved = (raw if raw.is_absolute() else ROOT / raw).resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} resolves outside the project workspace") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {relative}")
    return resolved, relative


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.lower()
    )


def _normalize_hash_rows(value: Any, *, label: str) -> list[dict[str, str]]:
    if isinstance(value, dict):
        rows = [{"path": path, "sha256": digest} for path, digest in value.items()]
    elif isinstance(value, list):
        rows = value
    else:
        raise TypeError(f"gate report {label} must be an object or list")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"gate report {label} entries must be objects")
        path_value = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path_value, str) or not _valid_sha(digest):
            raise ValueError(f"gate report {label} entry requires path and SHA-256")
        _, relative = _workspace_file(path_value, label=f"gate {label} path")
        if relative in seen:
            raise ValueError(f"gate report {label} contains duplicate path: {relative}")
        seen.add(relative)
        normalized.append({"path": relative, "sha256": digest.lower()})
    if not normalized:
        raise ValueError(f"gate report must contain at least one {label} hash")
    return sorted(normalized, key=lambda row: row["path"])


def _verify_hash_rows(rows: list[dict[str, str]], *, label: str) -> None:
    for row in rows:
        path, relative = _workspace_file(row["path"], label=f"gate {label} path")
        actual = _sha256(path)
        if actual != row["sha256"]:
            raise ValueError(
                f"gate {label} SHA-256 changed for {relative}: "
                f"expected {row['sha256']}, got {actual}"
            )


def _normalize_external_dependency_hashes(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(ALLOWED_EXTERNAL_DEPENDENCIES):
        raise ValueError(
            "gate external_dependency_hashes must contain exactly the three approved paths"
        )
    normalized: dict[str, str] = {}
    for relative in ALLOWED_EXTERNAL_DEPENDENCIES:
        digest = value[relative]
        if not _valid_sha(digest):
            raise ValueError(f"external dependency hash must be a SHA-256: {relative}")
        resolved = (ROOT / relative).resolve()
        expected = Path(os.path.abspath(ROOT / relative))
        if resolved != expected:
            raise ValueError(f"external dependency resolves to an unexpected file: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(f"approved external dependency is missing: {relative}")
        actual = _sha256(resolved)
        if actual != digest.lower():
            raise ValueError(
                f"external dependency SHA-256 changed for {relative}: "
                f"expected {digest.lower()}, got {actual}"
            )
        normalized[relative] = digest.lower()
    return normalized


def _read_gate_report(path: Path) -> dict[str, Any]:
    report_path, report_relative = _workspace_file(path, label="gate report")
    report = _load_json(report_path)
    if report.get("status") != "passed":
        raise ValueError("interface QA gate report must explicitly set status='passed'")
    qa_reference = report.get("qa_reference")
    if not isinstance(qa_reference, str) or not qa_reference.strip():
        raise ValueError("interface QA gate report must include a nonempty qa_reference")
    sources = _normalize_hash_rows(report.get("source_hashes"), label="source_hashes")
    if REQUIRED_GATE_SOURCE not in {row["path"] for row in sources}:
        raise ValueError(f"gate source_hashes must include {REQUIRED_GATE_SOURCE}")
    evidence = _normalize_hash_rows(report.get("evidence"), label="evidence")
    _verify_hash_rows(sources, label="source")
    _verify_hash_rows(evidence, label="evidence")
    external = _normalize_external_dependency_hashes(
        report.get("external_dependency_hashes")
    )
    return {
        "source_path": report_relative,
        "source_sha256": _sha256(report_path),
        "qa_reference": qa_reference.strip(),
        "source_hashes": sources,
        "evidence_hashes": evidence,
        "external_dependency_hashes": external,
    }


def _expected_contracts() -> dict[str, Any]:
    config = fresh._profile_config(PROFILE)
    architecture = fresh._experiment_architecture(512)
    robot_geometry = fresh._robot_geometry()
    return {
        "profile": PROFILE,
        "config": config,
        "architecture": architecture,
        "action_config": fresh._action_config(),
        "runtime_contract": fresh._runtime_contract(),
        "robot_geometry": robot_geometry,
        "profile_physics_contract": fresh._profile_physics_contract(
            PROFILE, config, robot_geometry
        ),
        "training": fresh._profile_training(PROFILE),
        "goal_objective_contract": fresh._fast_stop_objective_contract(config),
    }


def _validate_parent_contract(
    protocol: dict[str, Any], *, require_parent_schema: bool = True
) -> None:
    if require_parent_schema and protocol.get("schema") != fresh.SCHEMA:
        raise ValueError("parent protocol is not the frozen fresh-foundation schema")
    expected = _expected_contracts()
    expected_training = expected["training"]
    if not require_parent_schema:
        expected_training = json.loads(json.dumps(expected_training))
        expected_training["initialization"] = CONTINUATION_INITIALIZATION
    expected["training"] = expected_training
    for key, value in expected.items():
        if not fresh._json_contract_equal(protocol.get(key), value):
            raise ValueError(f"parent {key} differs from the fast-stop continuation contract")
    training = protocol["training"]
    if (training.get("num_envs") != 4
            or training.get("training_seed") != TRAIN_SEED
            or training.get("timesteps_total") != FINAL_TIMESTEPS
            or training.get("gamma") != 1.0
            or training.get("checkpoint_interval_transitions") != CHECKPOINT_INTERVAL
            or training.get("stages") != [dict(fresh.RANDOM_ROUGH_STAGE)]):
        raise ValueError("parent training budget or random-rough stage differs from the pinned trial")
    architecture = protocol["architecture"]
    if (architecture.get("lstm_hidden_size") != 512
            or architecture.get("lstm_num_layers") != 1
            or architecture.get("input_contract") != fresh.INPUT_SCHEMA
            or architecture.get("action_shape") != [4]):
        raise ValueError("parent policy architecture is not the expected 512-unit four-action policy")
    objective = protocol["goal_objective_contract"]
    stop = objective.get("goal_stop", {})
    if (objective.get("discount_gamma") != 1.0
            or stop.get("distance_tolerance_m") != 0.4
            or stop.get("linear_speed_limit_m_s") != 0.05
            or stop.get("angular_speed_limit_rad_s") != 0.1
            or stop.get("hold_required_s") != 0.5
            or stop.get("terminal_orientation_required") is not False):
        raise ValueError("parent strict-stop objective differs from the approved fast-stop contract")
    physics = protocol["profile_physics_contract"]
    guides = physics.get("front_guide_spec", {})
    stability = physics.get("body_stability_config", {})
    if (guides.get("rotation_coupling") != "same_side_track_speed"
            or physics.get("front_guide_drive_coupling", {}).get("guide_actuator_count") != 0
            or stability.get("tilt_weight") != 0.0
            or stability.get("angular_rate_weight") != 0.0):
        raise ValueError("parent front-guide or fast-stop stability physics differs")


def _verify_parent_snapshot(parent: Path, protocol: dict[str, Any]) -> None:
    rows = protocol.get("source_snapshot", {}).get("files", [])
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(rows) != SOURCE_COUNT or len(set(paths)) != SOURCE_COUNT:
        raise ValueError(f"parent protocol must contain its {SOURCE_COUNT}-source snapshot")
    for row in rows:
        relative = row.get("snapshot_path")
        digest = row.get("sha256")
        if not isinstance(relative, str) or ".." in Path(relative).parts or not _valid_sha(digest):
            raise ValueError("parent source snapshot manifest is malformed")
        snapshot = (parent / relative).resolve()
        try:
            snapshot.relative_to(parent.resolve())
        except ValueError as exc:
            raise ValueError("parent source snapshot resolves outside its trial") from exc
        if not snapshot.is_file() or _sha256(snapshot) != digest.lower():
            raise ValueError(f"parent source snapshot changed: {row.get('path')}")


def _checkpoint_info(path: Path, parent_protocol: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    sha_before = _sha256(path)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"checkpoint ZIP integrity failed at {bad_member}")
        entries = {item.filename: item.file_size for item in archive.infolist()}
        required = {"data", "policy.pth", "policy.optimizer.pth"}
        if not required.issubset(entries):
            raise ValueError("checkpoint must contain data, policy weights, and Adam optimizer state")
        if entries["policy.pth"] <= 0 or entries["policy.optimizer.pth"] <= 0:
            raise ValueError("checkpoint policy weights and Adam state must be nonempty")
        data = json.loads(archive.read("data"))
    expected_encoder = parent_protocol["architecture"]["field_encoder_config"]
    expected_action = parent_protocol["action_config"]
    policy_kwargs = data.get("policy_kwargs", {})
    if (data.get("num_timesteps") != START_TIMESTEPS
            or data.get("n_steps") != 128
            or data.get("n_envs") != 4
            or data.get("gamma") != 1.0
            or data.get("field_action_config") != expected_action
            or data.get("field_encoder_config") != expected_encoder
            or policy_kwargs.get("lstm_hidden_size") != 512
            or policy_kwargs.get("n_lstm_layers") != 1):
        raise ValueError("checkpoint timestep, PPO, encoder, action, or LSTM metadata differs")
    after = path.stat()
    sha_after = _sha256(path)
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or sha_before != sha_after):
        raise ValueError("checkpoint changed while it was being verified")
    if sha_before != PARENT_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 differs from the pinned parent: {sha_before}")
    return {
        "path": PARENT_CHECKPOINT_RELATIVE,
        "sha256": sha_before,
        "num_timesteps": START_TIMESTEPS,
        "size_bytes": before.st_size,
        "zip_integrity": True,
        "policy_weights_present": True,
        "adam_optimizer_state_present": True,
    }


def _audit_parent() -> dict[str, Any]:
    protocol_path = PARENT / "protocol.json"
    status_path = PARENT / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("pinned parent trial is missing protocol or status evidence")
    protocol_sha = _sha256(protocol_path)
    if protocol_sha != PARENT_PROTOCOL_SHA256:
        raise ValueError(f"parent protocol SHA-256 changed: {protocol_sha}")
    protocol = _load_json(protocol_path)
    _validate_parent_contract(protocol)
    _verify_parent_snapshot(PARENT, protocol)
    checkpoint = ROOT / PARENT_CHECKPOINT_RELATIVE
    checkpoint_info = _checkpoint_info(checkpoint, protocol)
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    return {
        "path": PARENT_RELATIVE,
        "protocol_sha256": protocol_sha,
        "status_path": (PARENT / "run_status.json").relative_to(ROOT).as_posix(),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "status_snapshot": status,
        "checkpoint": checkpoint_info,
        "protocol": protocol,
    }


def _capture_sources(output: Path) -> dict[str, Any]:
    base_paths = fresh._source_paths()
    if len(base_paths) != SOURCE_COUNT or len(set(base_paths)) != SOURCE_COUNT:
        raise ValueError(f"current training source set must contain exactly {SOURCE_COUNT} files")
    if RUNNER in base_paths:
        raise ValueError("resume runner must be frozen as an extra source, separate from the 45-source set")
    manifest = fresh._source_manifest(output, copy_files=True)
    rows = manifest.get("files", [])
    if len(rows) != SOURCE_COUNT or {row.get("path") for row in rows} != set(base_paths):
        raise ValueError("fresh runner returned an incomplete or unexpected 45-source snapshot")
    runner_path = ROOT / RUNNER
    runner_snapshot = output / "source_snapshot" / RUNNER
    runner_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runner_path, runner_snapshot)
    rows.append({"path": RUNNER, "snapshot_path": f"source_snapshot/{RUNNER}",
                 "sha256": _sha256(runner_path)})
    rows.sort(key=lambda row: row["path"])
    return {"files": rows, "base_source_count": SOURCE_COUNT,
            "extra_runner": RUNNER, "total_source_count": SOURCE_COUNT + 1}


def _verify_child_sources(output: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("files", [])
    if (manifest.get("base_source_count") != SOURCE_COUNT
            or manifest.get("extra_runner") != RUNNER
            or manifest.get("total_source_count") != SOURCE_COUNT + 1
            or len(rows) != SOURCE_COUNT + 1):
        raise ValueError("child source snapshot must contain 45 current sources plus this runner")
    base_rows = [row for row in rows if row.get("path") != RUNNER]
    runner_rows = [row for row in rows if row.get("path") == RUNNER]
    if len(runner_rows) != 1:
        raise ValueError("child source snapshot must freeze the resume runner exactly once")
    fresh._verify_sources(output, {"files": base_rows})
    row = runner_rows[0]
    snapshot = output / row["snapshot_path"]
    source = ROOT / RUNNER
    if (not snapshot.is_file() or not source.is_file()
            or not _valid_sha(row.get("sha256"))
            or _sha256(snapshot) != row["sha256"]
            or _sha256(source) != row["sha256"]):
        raise ValueError("resume runner differs from its frozen child snapshot")


def _validate_gate_snapshot(output: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol.get("interface_gate", {})
    report_relative = gate.get("source_path")
    copied_report = output / gate.get("snapshot_path", "")
    if (not isinstance(report_relative, str)
            or gate.get("snapshot_path") != "interface_gate_report.json"
            or not copied_report.is_file()
            or _sha256(copied_report) != gate.get("source_sha256")):
        raise ValueError("copied interface QA gate report is missing or changed")
    original, _ = _workspace_file(report_relative, label="original gate report")
    if _sha256(original) != gate.get("source_sha256"):
        raise ValueError("original interface QA gate report changed after prepare")
    report = _load_json(copied_report)
    if (report.get("status") != "passed"
            or report.get("qa_reference") != gate.get("qa_reference")):
        raise ValueError("saved interface gate report no longer records an explicit pass and QA reference")
    sources = _normalize_hash_rows(report.get("source_hashes"), label="source_hashes")
    evidence = _normalize_hash_rows(report.get("evidence"), label="evidence")
    external = _normalize_external_dependency_hashes(
        report.get("external_dependency_hashes")
    )
    if (sources != gate.get("source_hashes")
            or evidence != gate.get("evidence_hashes")
            or external != gate.get("external_dependency_hashes")
            or REQUIRED_GATE_SOURCE not in {row["path"] for row in sources}):
        raise ValueError("interface gate source/evidence hashes differ from the prepared protocol")
    _verify_hash_rows(sources, label="source")
    _verify_hash_rows(evidence, label="evidence")
    return gate


def prepare(output: Path, gate_report: Path) -> dict[str, Any]:
    output = output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"continuation output must be the independent child path: {DEFAULT_OUTPUT}")
    if output.exists():
        raise FileExistsError(f"refusing to reuse or overwrite continuation output: {output}")
    parent_info = _audit_parent()
    gate_info = _read_gate_report(gate_report)
    contract = _expected_contracts()
    contract["training"] = json.loads(json.dumps(contract["training"]))
    contract["training"]["initialization"] = CONTINUATION_INITIALIZATION
    parent_architecture = parent_info["protocol"]["architecture"]
    observation_contract = {
        "schema": "parent-isaac-policy-observation-contract-v1",
        "input_contract": parent_architecture["input_contract"],
        "observation_shapes": parent_architecture["observation_shapes"],
        "simulation_contract": "the original Isaac simulation policy observation contract",
        "rk_interface_connected": False,
        "consumer_interface_gate_changes_training_observation": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = _capture_sources(output)
    baseline = output / "parent_checkpoint_baseline.zip"
    shutil.copyfile(ROOT / PARENT_CHECKPOINT_RELATIVE, baseline)
    if _sha256(baseline) != parent_info["checkpoint"]["sha256"]:
        raise RuntimeError("child baseline copy differs from the verified parent checkpoint")
    gate_source = ROOT / gate_info["source_path"]
    gate_snapshot = output / "interface_gate_report.json"
    shutil.copyfile(gate_source, gate_snapshot)
    if _sha256(gate_snapshot) != gate_info["source_sha256"]:
        raise RuntimeError("copied interface gate report differs from its verified source")

    recovery = {
        "schema": "isaac-field-fast-stop-recovery-v1",
        "parent": {key: value for key, value in parent_info.items() if key != "protocol"},
        "checkpoint_copy": {
            "path": "parent_checkpoint_baseline.zip",
            "sha256": _sha256(baseline),
            "num_timesteps": START_TIMESTEPS,
            "adam_optimizer_state_present": True,
        },
        "training_plan": {
            "starting_num_timesteps": START_TIMESTEPS,
            "additional_timesteps": ADDITIONAL_TIMESTEPS,
            "expected_final_num_timesteps": FINAL_TIMESTEPS,
            "checkpoint_interval_transitions": CHECKPOINT_INTERVAL,
            "training_seed": TRAIN_SEED,
            "num_envs": 4,
            "n_steps": 128,
        },
        "continuation_semantics": {
            "policy_weights_and_adam_optimizer": "load and continue from the copied verified checkpoint",
            "global_timestep": "retained by train_stage.learn(reset_num_timesteps=False)",
            "environment_rng_lstm_state_and_partial_rollout": "reinitialized in the child environment",
            "observation_contract": "original Isaac simulation policy contract; RK interface is not connected",
            "consumer_interface_gate": "QA evidence only; it does not alter training observations",
            "bitwise_resume": False,
            "development_evaluation": "not run by this continuation",
            "blind_830xx": "not evaluated or observed",
        },
    }
    _write_json(output / "recovery_manifest.json", recovery)
    recovery_path = output / "recovery_manifest.json"
    protocol = {
        "schema": SCHEMA,
        "status": "prepared",
        **contract,
        "parent": {key: value for key, value in parent_info.items() if key != "protocol"},
        "initialization": {
            "checkpoint_path": "parent_checkpoint_baseline.zip",
            "checkpoint_sha256": _sha256(baseline),
            "starting_num_timesteps": START_TIMESTEPS,
            "loads_policy_weights": True,
            "loads_adam_optimizer_state": True,
            "weight_transfer": False,
            "expected_final_num_timesteps": FINAL_TIMESTEPS,
            "initial_random_baseline_created": False,
        },
        "training_plan": recovery["training_plan"],
        "observation_contract": observation_contract,
        "first_periodic_checkpoint": {
            "expected_global_num_timesteps": FIRST_PERIODIC_TIMESTEPS,
            "weights_and_adam_must_differ_from_parent_baseline": True,
        },
        "interface_gate": {
            **gate_info,
            "snapshot_path": "interface_gate_report.json",
        },
        "recovery_manifest": {
            "path": "recovery_manifest.json",
            "sha256": _sha256(recovery_path),
        },
        "source_snapshot": source_snapshot,
        "runtime": {"versions": fresh._package_versions(),
                    "isaaclab_commit": fresh._isaac_commit()},
    }
    _write_json(output / "protocol.json", protocol)
    protocol_sha = _sha256(output / "protocol.json")
    _write_json(output / "run_status.json", {
        "status": "prepared", "protocol_sha256": protocol_sha,
        "num_timesteps": START_TIMESTEPS, "exit_code": None,
    })
    preflight(output)
    return {
        "output": str(output),
        "protocol_sha256": protocol_sha,
        "parent_protocol_sha256": parent_info["protocol_sha256"],
        "checkpoint_sha256": parent_info["checkpoint"]["sha256"],
        "source_files": source_snapshot["total_source_count"],
        "qa_reference": gate_info["qa_reference"],
        "starting_num_timesteps": START_TIMESTEPS,
        "additional_timesteps": ADDITIONAL_TIMESTEPS,
        "expected_final_num_timesteps": FINAL_TIMESTEPS,
    }


def preflight(output: Path, *, allow_launch_lock: bool = False) -> dict[str, Any]:
    output = output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"preflight output must be the independent child path: {DEFAULT_OUTPUT}")
    protocol_path, status_path = output / "protocol.json", output / "run_status.json"
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("run --prepare before --preflight or --launch")
    protocol = _load_json(protocol_path)
    status = _load_json(status_path)
    protocol_sha = _sha256(protocol_path)
    if (protocol.get("schema") != SCHEMA or protocol.get("status") != "prepared"
            or status.get("status") != "prepared"
            or status.get("protocol_sha256") != protocol_sha):
        raise ValueError("continuation must be an unchanged, not-yet-launched prepared trial")
    _validate_parent_contract(protocol, require_parent_schema=False)
    expected_contracts = _expected_contracts()
    expected_contracts["training"] = {
        **expected_contracts["training"],
        "initialization": CONTINUATION_INITIALIZATION,
    }
    if (protocol.get("profile") != PROFILE
            or any(not fresh._json_contract_equal(protocol.get(key), expected)
                   for key, expected in expected_contracts.items())):
        raise ValueError("child protocol differs from current fast-stop training/runtime contracts")
    plan = protocol.get("training_plan", {})
    if (plan.get("starting_num_timesteps") != START_TIMESTEPS
            or plan.get("additional_timesteps") != ADDITIONAL_TIMESTEPS
            or plan.get("expected_final_num_timesteps") != FINAL_TIMESTEPS
            or plan.get("checkpoint_interval_transitions") != CHECKPOINT_INTERVAL
            or plan.get("training_seed") != TRAIN_SEED
            or plan.get("num_envs") != 4 or plan.get("n_steps") != 128
            or START_TIMESTEPS + ADDITIONAL_TIMESTEPS != FINAL_TIMESTEPS):
        raise ValueError("continuation timestep plan differs from the pinned 196,608-step budget")
    parent_architecture = _expected_contracts()["architecture"]
    expected_observation_contract = {
        "schema": "parent-isaac-policy-observation-contract-v1",
        "input_contract": parent_architecture["input_contract"],
        "observation_shapes": parent_architecture["observation_shapes"],
        "simulation_contract": "the original Isaac simulation policy observation contract",
        "rk_interface_connected": False,
        "consumer_interface_gate_changes_training_observation": False,
    }
    if protocol.get("observation_contract") != expected_observation_contract:
        raise ValueError("child policy observation contract changed or claims RK integration")
    first_periodic = protocol.get("first_periodic_checkpoint", {})
    if (first_periodic.get("expected_global_num_timesteps") != FIRST_PERIODIC_TIMESTEPS
            or first_periodic.get("weights_and_adam_must_differ_from_parent_baseline") is not True):
        raise ValueError("child protocol lacks the expected updated first periodic checkpoint gate")
    _verify_child_sources(output, protocol.get("source_snapshot", {}))
    _validate_gate_snapshot(output, protocol)
    current_parent = _audit_parent()
    parent_ref = protocol.get("parent", {})
    current_parent_ref = {key: value for key, value in current_parent.items() if key != "protocol"}
    if parent_ref != current_parent_ref:
        raise ValueError("parent protocol/status/checkpoint evidence changed after prepare")
    baseline = output / "parent_checkpoint_baseline.zip"
    if _sha256(baseline) != protocol.get("initialization", {}).get("checkpoint_sha256"):
        raise ValueError("child baseline checkpoint is not byte-identical to the pinned parent checkpoint")
    recovery_ref = protocol.get("recovery_manifest", {})
    recovery_path = output / recovery_ref.get("path", "")
    if (recovery_ref.get("path") != "recovery_manifest.json"
            or not recovery_path.is_file()
            or _sha256(recovery_path) != recovery_ref.get("sha256")):
        raise ValueError("continuation recovery manifest is missing or changed")
    recovery = _load_json(recovery_path)
    if (recovery.get("training_plan") != plan
            or recovery.get("parent", {}).get("protocol_sha256") != PARENT_PROTOCOL_SHA256
            or recovery.get("checkpoint_copy", {}).get("sha256") != _sha256(baseline)
            or recovery.get("continuation_semantics", {}).get("observation_contract")
            != "original Isaac simulation policy contract; RK interface is not connected"
            or recovery.get("continuation_semantics", {}).get("consumer_interface_gate")
            != "QA evidence only; it does not alter training observations"):
        raise ValueError("recovery manifest differs from the prepared continuation contract")
    training_artifacts = (
        "process_exit.json", "python.exit_code", "initialization.json",
        "first_periodic_checkpoint.json", "runtime_assets.json", "resume_summary.json",
        "stage_1_global_random", "stage_01_global_random",
    )
    supervisor_artifacts = ("launch.lock", "launch.log", "launcher.pid", "python.pid")
    for artifact in training_artifacts:
        if (output / artifact).exists():
            raise ValueError(f"continuation launch artifact already exists: {artifact}")
    if allow_launch_lock:
        _validate_supervisor_artifacts(output)
    else:
        for artifact in supervisor_artifacts:
            if (output / artifact).exists():
                raise ValueError(f"continuation launch artifact already exists: {artifact}")
    return protocol


def _validate_supervisor_artifacts(output: Path) -> dict[str, int]:
    """Validate only the files created by the current launch supervisor."""
    required = ("launch.lock", "launch.log", "launcher.pid", "python.pid")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"worker launch is missing supervisor artifacts: {missing}")
    try:
        lock = _load_json(output / "launch.lock")
        launcher_pid = int((output / "launcher.pid").read_text(encoding="ascii").strip())
        worker_pid = int((output / "python.pid").read_text(encoding="ascii").strip())
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("worker supervisor PID/lock records are malformed") from exc
    if (type(lock.get("pid")) is not int or lock["pid"] <= 0
            or launcher_pid != lock["pid"] or worker_pid <= 0):
        raise ValueError("worker supervisor PID records do not agree")
    return {"launcher_pid": launcher_pid, "worker_pid": worker_pid}


def _load_checkpoint_member(checkpoint: Path, name: str):
    import torch

    with zipfile.ZipFile(checkpoint) as archive:
        payload = archive.read(name)
    try:
        return torch.load(__import__("io").BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(__import__("io").BytesIO(payload), map_location="cpu")


def _audit_first_periodic_checkpoint(output: Path, baseline: Path) -> dict[str, Any]:
    checkpoint = (output / "stage_1_global_random" / "checkpoints"
                  / f"field_policy_global_random_{FIRST_PERIODIC_TIMESTEPS}_steps.zip")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"first periodic checkpoint at global step {FIRST_PERIODIC_TIMESTEPS} is missing"
        )
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError("first periodic checkpoint ZIP integrity failed")
        data = json.loads(archive.read("data"))
    if (data.get("num_timesteps") != FIRST_PERIODIC_TIMESTEPS
            or data.get("field_action_config") != fresh._action_config()):
        raise ValueError("first periodic checkpoint has the wrong global step or action contract")
    baseline_policy = _load_checkpoint_member(baseline, "policy.pth")
    periodic_policy = _load_checkpoint_member(checkpoint, "policy.pth")
    baseline_adam = _load_checkpoint_member(baseline, "policy.optimizer.pth")
    periodic_adam = _load_checkpoint_member(checkpoint, "policy.optimizer.pth")
    policy_changed = not _tree_equal(baseline_policy, periodic_policy)
    adam_changed = not _tree_equal(baseline_adam, periodic_adam)
    if not policy_changed or not adam_changed:
        raise ValueError(
            "first periodic checkpoint must contain policy and Adam updates relative to the parent"
        )
    return {
        "status": "passed",
        "path": checkpoint.relative_to(output).as_posix(),
        "sha256": _sha256(checkpoint),
        "global_num_timesteps": FIRST_PERIODIC_TIMESTEPS,
        "parent_baseline_sha256": _sha256(baseline),
        "policy_weights_changed_from_parent": policy_changed,
        "adam_optimizer_changed_from_parent": adam_changed,
        "policy_state_sha256": _tree_digest(periodic_policy),
        "adam_state_sha256": _tree_digest(periodic_adam),
    }


def _verify_first_periodic_checkpoint_record(output: Path, baseline: Path) -> dict[str, Any]:
    record_path = output / "first_periodic_checkpoint.json"
    if not record_path.is_file():
        raise ValueError("first periodic global-step update audit is missing")
    saved = _load_json(record_path)
    current = _audit_first_periodic_checkpoint(output, baseline)
    if saved != current:
        raise ValueError("first periodic checkpoint or its recorded update audit changed")
    return saved


def _tree_equal(left: Any, right: Any) -> bool:
    import torch

    if torch.is_tensor(left) and torch.is_tensor(right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _tree_digest(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            if tensor.is_floating_point() and not bool(tensor.isfinite().all()):
                raise ValueError(f"non-finite tensor in loaded checkpoint state: {prefix}")
            digest.update(prefix.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{prefix}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, nested in enumerate(item):
                visit(nested, f"{prefix}/{index}")
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value, "root")
    return digest.hexdigest()


def _initialization_audit(model: Any, env: Any, protocol: dict[str, Any], baseline: Path,
                          output: Path) -> dict[str, Any]:
    from isaac_field.policy import _validate_model_action_contract

    if int(model.num_timesteps) != START_TIMESTEPS:
        raise ValueError(f"loaded RecurrentPPO starts at {model.num_timesteps}, expected {START_TIMESTEPS}")
    if int(getattr(model, "n_steps", -1)) != 128 or int(getattr(model, "n_envs", -1)) != 4:
        raise ValueError("loaded RecurrentPPO rollout or environment count differs from protocol")
    if float(getattr(model, "gamma", math.nan)) != 1.0:
        raise ValueError("loaded RecurrentPPO gamma differs from strict-stop protocol")
    policy = model.policy
    if (int(policy.features_extractor.features_dim) != 192
            or int(policy.lstm_actor.hidden_size) != 512
            or int(policy.lstm_critic.hidden_size) != 512
            or int(policy.lstm_actor.num_layers) != 1
            or int(policy.lstm_critic.num_layers) != 1):
        raise ValueError("loaded RecurrentPPO architecture differs from the 512-unit protocol")
    _validate_model_action_contract(model)
    live_contract = fresh._validate_rough_profile_environment(
        env, protocol["profile_physics_contract"]
    )
    expected_policy = _load_checkpoint_member(baseline, "policy.pth")
    expected_adam = _load_checkpoint_member(baseline, "policy.optimizer.pth")
    live_policy = policy.state_dict()
    live_adam = policy.optimizer.state_dict()
    if type(policy.optimizer).__name__ != "Adam":
        raise ValueError(f"loaded optimizer is {type(policy.optimizer).__name__}, expected Adam")
    if not _tree_equal(live_policy, expected_policy):
        raise ValueError("loaded policy tensors differ from the verified parent checkpoint")
    if not _tree_equal(live_adam, expected_adam):
        raise ValueError("loaded Adam tensors differ from the verified parent checkpoint")
    if fresh._sha256(baseline) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("baseline checkpoint changed during initialization audit")
    result = {
        "status": "passed",
        "checkpoint_path": "parent_checkpoint_baseline.zip",
        "checkpoint_sha256": fresh._sha256(baseline),
        "num_timesteps": int(model.num_timesteps),
        "policy_weights_equal_checkpoint": True,
        "policy_state_sha256": _tree_digest(live_policy),
        "adam_optimizer": type(policy.optimizer).__name__,
        "adam_state_equal_checkpoint": True,
        "adam_state_sha256": _tree_digest(live_adam),
        "live_environment_contract": live_contract,
        "environment_rng_reinitialized": True,
        "bitwise_resume": False,
    }
    _write_json(output / "initialization.json", result)
    return result


def _write_status(output: Path, runtime: dict[str, Any], status: str = "running", **extra) -> None:
    _write_json(output / "run_status.json", {
        "status": status,
        "protocol_sha256": runtime["protocol_sha256"],
        "num_timesteps": int(runtime.get("num_timesteps", START_TIMESTEPS)),
        "updated_at": datetime.now(UTC).isoformat(),
        **extra,
    })


def _launch_training(output: Path, device: str) -> dict[str, Any]:
    protocol = preflight(output, allow_launch_lock=True)
    runtime = {"protocol_sha256": _sha256(output / "protocol.json"),
               "num_timesteps": START_TIMESTEPS}
    _write_status(output, runtime, phase="initializing")
    try:
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )

        from isaac_field.env import make_env
        from isaac_field.learning import CurriculumStage, train_curriculum
        from isaac_field.task import Config
        from scripts.improve_isaac_policy import _make_progress_callback

        cfg = Config(**protocol["config"])
        training = protocol["training"]
        training_plan = protocol["training_plan"]
        guide_spec, stability_config = fresh._rough_profile_components(PROFILE)
        if (fresh.asdict(guide_spec) != protocol["profile_physics_contract"]["front_guide_spec"]
                or fresh.asdict(stability_config)
                != protocol["profile_physics_contract"]["body_stability_config"]):
            raise ValueError("live front-guide/stability settings differ from the frozen protocol")
        stage_rows = training["stages"]
        stages = tuple(CurriculumStage(
            row["name"], "goto", row.get("goal_distance_m"),
            math.radians(float(row["goal_angle_deg"])),
        ) for row in stage_rows)
        latest: dict[str, Any] = {"env": None}
        assets: list[dict[str, Any]] = []
        progress_by_stage: dict[str, Any] = {}
        baseline = output / protocol["initialization"]["checkpoint_path"]

        def env_factory(**kwargs):
            _verify_child_sources(output, protocol["source_snapshot"])
            kwargs["num_envs"] = 4 if kwargs["auto_reset"] else 1
            kwargs["front_guide_spec"] = guide_spec
            kwargs["body_stability_config"] = stability_config
            env = make_env(**kwargs)
            try:
                fresh._validate_rough_profile_environment(
                    env, protocol["profile_physics_contract"]
                )
                assets.append({
                    "environment_index": len(assets),
                    "seed": kwargs.get("seed"),
                    "assets": fresh._runtime_assets(env),
                })
                _write_json(output / "runtime_assets.json", {"environments": assets})
            except BaseException:
                env.close()
                raise
            latest["env"] = env
            return env

        def callback_factory(stage):
            index = next(i for i, item in enumerate(stages) if item.name == stage.name)
            stage_dir = output / f"stage_{index + 1}_{stage.name}"
            checkpoint_dir = stage_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _verify_child_sources(output, protocol["source_snapshot"])
            progress = _make_progress_callback(4, latest["env"], interval=CHECKPOINT_INTERVAL)
            progress_by_stage[stage.name] = progress
            checkpoint_callback = CheckpointCallback(
                save_freq=CHECKPOINT_INTERVAL // 4,
                save_path=str(checkpoint_dir),
                name_prefix=f"field_policy_{stage.name}",
                verbose=0,
            )

            class PersistStage(BaseCallback):
                def _on_training_start(self):
                    self.stage_start = int(self.model.num_timesteps)
                    if self.stage_start != START_TIMESTEPS:
                        raise ValueError(
                            f"training did not start from {START_TIMESTEPS}: {self.stage_start}"
                        )
                    runtime["num_timesteps"] = self.stage_start
                    _write_status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    if index == 0:
                        _initialization_audit(self.model, latest["env"], protocol, baseline, output)

                def _on_step(self):
                    runtime["num_timesteps"] = int(self.num_timesteps)
                    if runtime["num_timesteps"] == FIRST_PERIODIC_TIMESTEPS:
                        audit = _audit_first_periodic_checkpoint(output, baseline)
                        _write_json(output / "first_periodic_checkpoint.json", audit)
                    if (runtime["num_timesteps"] - self.stage_start) % CHECKPOINT_INTERVAL == 0:
                        fresh._checkpoint_manifest(output)
                        _write_status(output, runtime, phase=f"training_{stage.name}", stage=stage.name)
                    return True

                def _on_training_end(self):
                    progress._refresh_episodes()
                    runtime["num_timesteps"] = int(self.model.num_timesteps)
                    _write_json(stage_dir / "progress.json", {
                        "stage": stage.name,
                        "start_num_timesteps": self.stage_start,
                        "end_num_timesteps": runtime["num_timesteps"],
                        "aggregate_rows": progress.log_rows,
                        "last_completed_episodes": progress.recent_episodes[-10:],
                    })
                    _write_status(output, runtime, phase=f"completed_{stage.name}", stage=stage.name)

            callbacks = [checkpoint_callback, progress, PersistStage()]
            from isaac_field.training_diagnostics import make_sb3_diagnostics_callback

            callbacks.append(make_sb3_diagnostics_callback(
                stage_dir / "training_diagnostics.jsonl",
                stage=stage.name,
                info_provider=lambda: getattr(latest.get("env"), "diagnostic_step_infos", ()),
            ))
            return CallbackList(callbacks)

        _write_status(output, runtime, phase="creating_training_environment")
        result = train_curriculum(
            cfg,
            output,
            checkpoint=baseline,
            timesteps_per_stage=(int(training_plan["additional_timesteps"]),),
            training_seed=int(training_plan["training_seed"]),
            heldout_seeds=tuple(fresh.DEV_FIELD_SEEDS),
            stages=stages,
            evaluate_before_after=False,
            goal_gain=1.0,
            gamma=float(training["gamma"]),
            device=device,
            n_epochs=int(training["n_epochs"]),
            batch_size=int(training["batch_size"]),
            callback_factory=callback_factory,
            env_factory=env_factory,
            lstm_hidden_size=int(protocol["architecture"]["lstm_hidden_size"]),
        )
        actual_final = int(result.summary["stages"][-1]["total_timesteps"])
        if actual_final != FINAL_TIMESTEPS:
            raise RuntimeError(f"expected {FINAL_TIMESTEPS} cumulative steps, got {actual_final}")
        if not (output / "initialization.json").is_file():
            raise RuntimeError("resumed policy/Adam initialization audit was not recorded")
        first_periodic_audit = _verify_first_periodic_checkpoint_record(output, baseline)
        checkpoints = fresh._checkpoint_manifest(output)
        final_path = Path(result.checkpoint).resolve()
        final_row = next((row for row in checkpoints
                          if (output / row["path"]).resolve() == final_path), None)
        if (final_row is None or final_row["num_timesteps"] != FINAL_TIMESTEPS
                or final_row["adam_optimizer_state_present"] is not True):
            raise RuntimeError("final checkpoint manifest lacks the expected cumulative steps or Adam state")
        _verify_child_sources(output, protocol["source_snapshot"])
        runtime["num_timesteps"] = actual_final
        summary = {
            "schema": SCHEMA,
            "status": "training_completed_no_evaluation",
            "parent_protocol_sha256": protocol["parent"]["protocol_sha256"],
            "parent_checkpoint_sha256": protocol["parent"]["checkpoint"]["sha256"],
            "checkpoint_start": str(baseline),
            "checkpoint_final": str(final_path),
            "checkpoint_final_sha256": final_row["sha256"],
            "starting_num_timesteps": START_TIMESTEPS,
            "additional_timesteps": ADDITIONAL_TIMESTEPS,
            "actual_final_num_timesteps": actual_final,
            "initialization_audit": "initialization.json",
            "first_periodic_checkpoint_audit": "first_periodic_checkpoint.json",
            "first_periodic_checkpoint_num_timesteps": first_periodic_audit["global_num_timesteps"],
            "checkpoint_count": len(checkpoints),
            "development_evaluation": "not run",
            "blind_830xx": "not evaluated or observed",
            "claim_limit": "Continuation completion is not evidence of navigation improvement.",
        }
        _write_json(output / "resume_summary.json", summary)
        _write_status(output, runtime, "completed", phase="completed",
                      summary="resume_summary.json", exit_code=0,
                      checkpoint_final=str(final_path), checkpoint_final_sha256=final_row["sha256"])
        return summary
    except KeyboardInterrupt as exc:
        _write_status(output, runtime, "interrupted", phase="interrupted",
                      exception=f"{type(exc).__name__}: {exc}", exit_code=130)
        raise
    except BaseException as exc:
        _write_status(output, runtime, "failed", phase="failed",
                      exception=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise


def launch(output: Path, device: str = "cuda:0") -> dict[str, Any]:
    output = output.resolve()
    preflight(output)
    lock = output / "launch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(),
                                     "started_at": datetime.now(UTC).isoformat()}))
    except FileExistsError as exc:
        raise ValueError("continuation launch has already been claimed") from exc
    if (output / "launch.log").exists():
        raise FileExistsError("refusing to overwrite continuation launch.log")
    started = datetime.now(UTC).isoformat()
    (output / "launcher.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
               "--output", str(output), "--device", device]
    with (output / "launch.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (output / "python.pid").write_text(f"{process.pid}\n", encoding="ascii")
        actual_exit_code = process.wait()
    finished = datetime.now(UTC).isoformat()
    (output / "python.exit_code").write_text(f"{actual_exit_code}\n", encoding="ascii")
    exit_record = {
        "launcher_pid": os.getpid(),
        "python_pid": process.pid,
        "exit_code": actual_exit_code,
        "actual_exit_code": actual_exit_code,
        "started_utc": started,
        "finished_utc": finished,
        "source": "resume_fast_stop_supervisor_waited_for_worker",
    }
    _write_json(output / "process_exit.json", exit_record)
    if actual_exit_code != 0:
        status_path = output / "run_status.json"
        status = _load_json(status_path) if status_path.is_file() else {}
        if status.get("status") in ("prepared", "running"):
            _write_json(output / "run_status.json", {
                **status, "status": "failed", "exit_code": actual_exit_code,
                "phase": "worker_exit", "updated_at": finished,
            })
    return exit_record


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Internal continuation training worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if not (output / "launch.lock").is_file():
            raise ValueError("worker requires the launch supervisor lock")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                supervisor = _validate_supervisor_artifacts(output)
            except ValueError:
                supervisor = None
            if supervisor is not None and supervisor["worker_pid"] == os.getpid():
                break
            if time.monotonic() >= deadline:
                raise ValueError("worker PID is not recorded by its launch supervisor")
            time.sleep(0.05)
        preflight(output, allow_launch_lock=True)
        _launch_training(output, args.device)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001 - CLI worker boundary must record unexpected failures.
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        return _worker_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if args.gate_report is None:
                parser.error("--prepare requires --gate-report")
            result = prepare(args.output, args.gate_report)
        elif args.preflight:
            protocol = preflight(args.output)
            result = {"status": "ready", "protocol_sha256": _sha256(args.output / "protocol.json"),
                      "source_files": protocol["source_snapshot"]["total_source_count"],
                      "starting_num_timesteps": START_TIMESTEPS,
                      "additional_timesteps": ADDITIONAL_TIMESTEPS,
                      "expected_final_num_timesteps": FINAL_TIMESTEPS,
                      "qa_reference": protocol["interface_gate"]["qa_reference"]}
        else:
            result = launch(args.output, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(result.get("actual_exit_code", 0))
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001 - CLI boundary prints diagnostics and returns a nonzero code.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
