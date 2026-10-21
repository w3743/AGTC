from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

import scripts.resume_fast_stop as resume
from scripts import train_isaac_foundation_fresh as fresh


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | dict]:
    root = tmp_path / "workspace"
    root.mkdir()
    parent = root / "results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020"
    child = root / "results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_resume_20261021"
    evidence = root / "results/qa/interface_contract.json"
    gate_path = root / "results/qa/interface_gate.json"

    source_paths = tuple(
        f"isaac_field/frozen_{index:02d}.py" for index in range(resume.SOURCE_COUNT)
    )
    for relative in (*source_paths, resume.REQUIRED_GATE_SOURCE):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"frozen current source: {relative}\n", encoding="utf-8")
    runner_copy = root / resume.RUNNER
    runner_copy.parent.mkdir(parents=True, exist_ok=True)
    runner_copy.write_bytes(Path(resume.__file__).read_bytes())
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("interface QA evidence\n", encoding="utf-8")
    external_dependencies = {}
    for relative in resume.ALLOWED_EXTERNAL_DEPENDENCIES:
        external = (root / relative).resolve()
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_text(f"approved external contract source: {relative}\n", encoding="utf-8")
        external_dependencies[relative] = _sha(external)

    monkeypatch.setattr(resume, "ROOT", root)
    monkeypatch.setattr(resume, "PARENT", parent)
    monkeypatch.setattr(resume, "PARENT_RELATIVE", parent.relative_to(root).as_posix())
    monkeypatch.setattr(resume, "PARENT_CHECKPOINT_RELATIVE", (
        parent.relative_to(root).as_posix()
        + "/stage_1_global_random/checkpoints/field_policy_global_random_57344_steps.zip"
    ))
    monkeypatch.setattr(resume, "DEFAULT_OUTPUT", child)
    monkeypatch.setattr(fresh, "ROOT", root)
    monkeypatch.setattr(fresh, "_source_paths", lambda: source_paths)
    monkeypatch.setattr(fresh, "_profile_config", lambda _profile: {"max_time": 60.0, "plants": 0})
    monkeypatch.setattr(fresh, "_experiment_architecture", lambda _width: {
        "field_encoder_config": {"schema": "test-encoder-v1", "features_dim": 192},
        "input_contract": "sensor-minimal-v6",
        "observation_shapes": {"depth": [88, 160], "lidar": [64], "state": [32]},
        "action_shape": [4],
        "goal_gain": 1.0, "lstm_hidden_size": 512, "lstm_num_layers": 1,
    })
    action = {"schema": "direct-track-speed-v1", "components": ["left", "right", "yaw", "pitch"]}
    robot = {"sha256": "robot-test-sha"}
    physics = {
        "front_guide_spec": {"rotation_coupling": "same_side_track_speed"},
        "front_guide_drive_coupling": {"guide_actuator_count": 0},
        "body_stability_config": {"tilt_weight": 0.0, "angular_rate_weight": 0.0},
    }
    objective = {
        "discount_gamma": 1.0,
        "goal_stop": {"distance_tolerance_m": 0.4, "linear_speed_limit_m_s": 0.05,
                      "angular_speed_limit_rad_s": 0.1, "hold_required_s": 0.5,
                      "terminal_orientation_required": False},
    }
    training = {
        "algorithm": "SB3-Contrib RecurrentPPO", "mode": "goto", "num_envs": 4,
        "training_seed": resume.TRAIN_SEED, "timesteps_total": resume.FINAL_TIMESTEPS,
        "stages": [dict(fresh.RANDOM_ROUGH_STAGE)], "n_steps": 128, "batch_size": 128,
        "n_epochs": 5, "learning_rate": 3e-4, "gamma": 1.0,
        "checkpoint_interval_transitions": resume.CHECKPOINT_INTERVAL,
    }
    runtime = {"encoder_schema": "depth64-lidar-single1d-minimal-v3"}
    monkeypatch.setattr(fresh, "_action_config", lambda: action)
    monkeypatch.setattr(fresh, "_runtime_contract", lambda: runtime)
    monkeypatch.setattr(fresh, "_robot_geometry", lambda: robot)
    monkeypatch.setattr(fresh, "_profile_physics_contract", lambda *_args: physics)
    monkeypatch.setattr(fresh, "_profile_training", lambda _profile: training)
    monkeypatch.setattr(fresh, "_fast_stop_objective_contract", lambda _config: objective)
    monkeypatch.setattr(fresh, "_package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(fresh, "_isaac_commit", lambda: "test-commit")

    parent.mkdir(parents=True)
    parent_source_rows = []
    for relative in source_paths:
        source = root / relative
        snapshot_path = f"source_snapshot/{relative}"
        snapshot = parent / snapshot_path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        parent_source_rows.append({"path": relative, "snapshot_path": snapshot_path,
                                   "sha256": _sha(source)})
    protocol = {
        "schema": fresh.SCHEMA,
        "profile": resume.PROFILE,
        "config": {"max_time": 60.0, "plants": 0},
        "architecture": fresh._experiment_architecture(512),
        "action_config": action,
        "runtime_contract": runtime,
        "robot_geometry": robot,
        "profile_physics_contract": physics,
        "training": training,
        "goal_objective_contract": objective,
        "source_snapshot": {"files": parent_source_rows},
    }
    protocol_path = parent / "protocol.json"
    _write_json(protocol_path, protocol)
    status_path = parent / "run_status.json"
    _write_json(status_path, {"status": "running", "phase": "stale-pause-record"})
    checkpoint = parent / "stage_1_global_random/checkpoints/field_policy_global_random_57344_steps.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({
            "num_timesteps": resume.START_TIMESTEPS,
            "n_steps": 128,
            "n_envs": 4,
            "gamma": 1.0,
            "field_action_config": action,
            "field_encoder_config": fresh._experiment_architecture(512)["field_encoder_config"],
            "policy_kwargs": {"lstm_hidden_size": 512, "n_lstm_layers": 1},
        }))
        archive.writestr("policy.pth", b"test-policy-tensors")
        archive.writestr("policy.optimizer.pth", b"test-adam-tensors")
    monkeypatch.setattr(resume, "PARENT_PROTOCOL_SHA256", _sha(protocol_path))
    monkeypatch.setattr(resume, "PARENT_CHECKPOINT_SHA256", _sha(checkpoint))

    gate = {
        "status": "passed",
        "qa_reference": "interface-qa-2026-10-21#accepted",
        "source_hashes": {resume.REQUIRED_GATE_SOURCE: _sha(root / resume.REQUIRED_GATE_SOURCE)},
        "evidence": [{"path": evidence.relative_to(root).as_posix(), "sha256": _sha(evidence)}],
        "external_dependency_hashes": external_dependencies,
    }
    _write_json(gate_path, gate)
    return {"root": root, "parent": parent, "child": child, "gate": gate_path,
            "evidence": evidence, "checkpoint": checkpoint, "protocol": protocol_path,
            "status": status_path, "source_paths": source_paths}


def test_prepare_records_pinned_parent_gate_and_46_frozen_sources(trial, monkeypatch) -> None:
    root = trial["root"]
    parent = trial["parent"]
    child = trial["child"]
    parent_checkpoint_sha = _sha(trial["checkpoint"])
    parent_status_sha = _sha(trial["status"])

    result = resume.prepare(child, trial["gate"])
    protocol = json.loads((child / "protocol.json").read_text(encoding="utf-8"))
    recovery = json.loads((child / "recovery_manifest.json").read_text(encoding="utf-8"))
    assert result["starting_num_timesteps"] == 57_344
    assert result["additional_timesteps"] == 139_264
    assert result["expected_final_num_timesteps"] == 196_608
    assert protocol["source_snapshot"]["base_source_count"] == 45
    assert protocol["source_snapshot"]["total_source_count"] == 46
    assert protocol["initialization"]["loads_policy_weights"] is True
    assert protocol["initialization"]["loads_adam_optimizer_state"] is True
    assert protocol["interface_gate"]["qa_reference"].endswith("#accepted")
    assert protocol["interface_gate"]["source_hashes"][0]["path"] == resume.REQUIRED_GATE_SOURCE
    assert set(protocol["interface_gate"]["external_dependency_hashes"]) == set(
        resume.ALLOWED_EXTERNAL_DEPENDENCIES
    )
    assert protocol["observation_contract"]["rk_interface_connected"] is False
    assert protocol["observation_contract"]["consumer_interface_gate_changes_training_observation"] is False
    assert recovery["continuation_semantics"]["bitwise_resume"] is False
    assert recovery["continuation_semantics"]["environment_rng_lstm_state_and_partial_rollout"] == (
        "reinitialized in the child environment"
    )
    assert _sha(child / "parent_checkpoint_baseline.zip") == parent_checkpoint_sha
    assert _sha(trial["status"]) == parent_status_sha
    assert trial["parent"] == parent
    assert (child / "run_status.json").is_file()
    assert (root / "scripts/resume_fast_stop.py").is_file()
    assert resume.preflight(child)["schema"] == resume.SCHEMA


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "pending"), ("qa_reference", "")],
)
def test_prepare_rejects_gate_without_explicit_pass_and_reference(trial, field, value) -> None:
    report = json.loads(trial["gate"].read_text(encoding="utf-8"))
    report[field] = value
    _write_json(trial["gate"], report)
    with pytest.raises(ValueError, match="gate report"):
        resume.prepare(trial["child"], trial["gate"])
    assert not trial["child"].exists()


@pytest.mark.parametrize("changed_artifact", ["checkpoint", "evidence", "source"])
def test_preflight_rejects_changed_parent_or_gate_evidence(trial, changed_artifact) -> None:
    resume.prepare(trial["child"], trial["gate"])
    if changed_artifact == "checkpoint":
        with zipfile.ZipFile(trial["checkpoint"], "a") as archive:
            archive.writestr("post-prepare-change", b"tamper")
    elif changed_artifact == "evidence":
        trial["evidence"].write_text("changed QA evidence\n", encoding="utf-8")
    else:
        source = trial["root"] / resume.REQUIRED_GATE_SOURCE
        source.write_text("changed source after QA\n", encoding="utf-8")
    with pytest.raises(ValueError):
        resume.preflight(trial["child"])


def test_prepare_refuses_to_overwrite_an_existing_child(trial) -> None:
    resume.prepare(trial["child"], trial["gate"])
    protocol_before = _sha(trial["child"] / "protocol.json")
    with pytest.raises(FileExistsError, match="refusing to reuse or overwrite"):
        resume.prepare(trial["child"], trial["gate"])
    assert _sha(trial["child"] / "protocol.json") == protocol_before


def test_preflight_rejects_changed_approved_external_dependency(trial) -> None:
    resume.prepare(trial["child"], trial["gate"])
    relative = resume.ALLOWED_EXTERNAL_DEPENDENCIES[0]
    external = (trial["root"] / relative).resolve()
    external.write_text("changed approved interface dependency\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external dependency SHA-256 changed"):
        resume.preflight(trial["child"])


def test_prepare_rejects_non_allowlisted_external_dependency_path(trial) -> None:
    report = json.loads(trial["gate"].read_text(encoding="utf-8"))
    report["external_dependency_hashes"]["../unapproved/dependency.py"] = "0" * 64
    _write_json(trial["gate"], report)
    with pytest.raises(ValueError, match="exactly the three approved paths"):
        resume.prepare(trial["child"], trial["gate"])


def test_worker_preflight_allows_only_supervisor_artifacts(trial, monkeypatch) -> None:
    resume.prepare(trial["child"], trial["gate"])
    child = trial["child"]
    (child / "launch.lock").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    (child / "launch.log").write_text("supervisor launch log\n", encoding="utf-8")
    (child / "launcher.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    (child / "python.pid").write_text(f"{os.getpid()}\n", encoding="ascii")

    launched = []
    monkeypatch.setattr(
        resume, "_launch_training", lambda output, device: launched.append((output, device))
    )
    assert resume._worker_main(["--output", str(child), "--device", "cpu"]) == 0
    assert launched == [(child.resolve(), "cpu")]
    with pytest.raises(ValueError, match="launch artifact already exists"):
        resume.preflight(child)

    checkpoint_dir = child / "stage_1_global_random" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "preexisting.zip").write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="stage_1_global_random"):
        resume.preflight(child, allow_launch_lock=True)


def test_first_periodic_audit_requires_updated_policy_and_adam(trial, monkeypatch) -> None:
    resume.prepare(trial["child"], trial["gate"])
    baseline = trial["child"] / "parent_checkpoint_baseline.zip"
    checkpoint = (trial["child"] / "stage_1_global_random" / "checkpoints"
                  / f"field_policy_global_random_{resume.FIRST_PERIODIC_TIMESTEPS}_steps.zip")
    checkpoint.parent.mkdir(parents=True)
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({
            "num_timesteps": resume.FIRST_PERIODIC_TIMESTEPS,
            "field_action_config": fresh._action_config(),
        }))
        archive.writestr("policy.pth", b"updated policy")
        archive.writestr("policy.optimizer.pth", b"updated adam")

    monkeypatch.setattr(
        resume, "_load_checkpoint_member",
        lambda path, name: f"{'baseline' if path == baseline else 'updated'}:{name}",
    )
    monkeypatch.setattr(resume, "_tree_equal", lambda left, right: left == right)
    monkeypatch.setattr(resume, "_tree_digest", lambda value: f"digest:{value}")
    audit = resume._audit_first_periodic_checkpoint(trial["child"], baseline)
    assert audit["global_num_timesteps"] == 65_536
    assert audit["policy_weights_changed_from_parent"] is True
    assert audit["adam_optimizer_changed_from_parent"] is True

    monkeypatch.setattr(
        resume, "_load_checkpoint_member", lambda _path, name: f"same:{name}"
    )
    with pytest.raises(ValueError, match="policy and Adam updates"):
        resume._audit_first_periodic_checkpoint(trial["child"], baseline)


def test_live_profile_contracts_roundtrip_through_protocol_json() -> None:
    pytest.importorskip("stable_baselines3")
    contracts = resume._expected_contracts()
    roundtripped = json.loads(json.dumps(contracts, ensure_ascii=False, allow_nan=False))
    assert roundtripped["robot_geometry"] != contracts["robot_geometry"]
    assert fresh._json_contract_equal(
        roundtripped["robot_geometry"], contracts["robot_geometry"]
    )
    assert fresh._json_contract_equal(
        roundtripped["profile_physics_contract"], contracts["profile_physics_contract"]
    )
    assert all(fresh._json_contract_equal(roundtripped[key], value)
               for key, value in contracts.items())


def test_preflight_rejects_incompatible_checkpoint_optimizer_or_policy_contract(trial) -> None:
    checkpoint = trial["checkpoint"]
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({"num_timesteps": resume.START_TIMESTEPS,
                                               "n_steps": 128, "n_envs": 4, "gamma": 1.0}))
        archive.writestr("policy.pth", b"policy")
        archive.writestr("policy.optimizer.pth", b"")
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(resume, "PARENT_CHECKPOINT_SHA256", _sha(checkpoint))
        with pytest.raises(ValueError, match="nonempty|metadata"):
            resume.prepare(trial["child"], trial["gate"])
    finally:
        monkeypatch.undo()
