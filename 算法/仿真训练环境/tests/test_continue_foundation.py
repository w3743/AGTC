"""CPU-only contract tests for the pinned coupled-track continuation runner."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import scripts.continue_isaac_foundation as runner
import scripts.train_isaac_foundation_fresh as fresh


UI_DRIFT = (
    "isaac_field/policy.py", "isaac_field/env.py", "isaac_field/viewer.py",
    "isaac_field/visualization/THIRD_PARTY_NOTICES.md",
    "isaac_field/visualization/network_2d.html", "isaac_field/visualization/network_2d.py",
)
FILES = UI_DRIFT + tuple(f"isaac_field/test_source_{index:02d}.py" for index in range(36))
CONFIG = {"plants": 0, "terrain_enabled": False, "max_time": 20.0, "damage_penalty": 20.0}
ACTION = {"schema": "test-action"}
ENCODER = {"schema": fresh.ENCODER_SCHEMA, "features_dim": 192}
TRAINING = {"stages": [], "timesteps_total": 196608}
CHECKS = {
    "protocol_sha_matches_expected_and_stable": True,
    "final_checkpoint_sha_stable_and_matches_manifest": True,
    "final_timestep_count_matches_protocol": True,
    "baseline_and_checkpoint_metadata_match_protocol": True,
    "42_live_sources_and_snapshot_copies_match": True,
    "all_weights_finite_and_shapes_match": True,
    "adam_optimizer_state_finite_with_positive_steps": True,
    "loaded_cpu_only": True,
    "live_runtime_contract_matches_v7": True,
}


def _zip_checkpoint(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({
            "num_timesteps": runner.PARENT_TIMESTEPS,
            "field_action_config": ACTION, "field_encoder_config": ENCODER,
        }))
        archive.writestr("policy.pth", b"weights")
        archive.writestr("policy.optimizer.pth", b"adam")


@pytest.fixture
def parent_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    parent = root / "parent"
    parent.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "PARENT", parent)
    monkeypatch.setattr(runner, "RUNNER", "scripts/continue_isaac_foundation.py")
    monkeypatch.setattr(runner, "KNOWN_PARENT_UI_DRIFT", set(UI_DRIFT[2:]))
    monkeypatch.setattr(fresh, "ROOT", root)
    monkeypatch.setattr(fresh, "_source_paths", lambda: FILES)
    monkeypatch.setattr(fresh, "_action_config", lambda: copy.deepcopy(ACTION))
    monkeypatch.setattr(fresh, "_field_encoder_config", lambda: copy.deepcopy(ENCODER))
    monkeypatch.setattr(fresh, "_config_dict", lambda: copy.deepcopy(CONFIG))
    monkeypatch.setattr(fresh, "_runtime_contract", lambda: {"runtime": "coupled-v7"})
    monkeypatch.setattr(fresh, "_experiment_architecture", lambda: {"architecture": "LSTM128"})
    monkeypatch.setattr(fresh, "_robot_geometry", lambda: {"robot": "test"})
    monkeypatch.setattr(fresh, "_package_versions", lambda: {"torch": "test"})
    monkeypatch.setattr(fresh, "_isaac_commit", lambda: "test")
    monkeypatch.setattr(fresh, "SCHEMA", "fresh-test-schema")
    monkeypatch.setattr(fresh, "TRAINING", copy.deepcopy(TRAINING))
    own_runner = root / runner.RUNNER
    own_runner.parent.mkdir(parents=True, exist_ok=True)
    own_runner.write_text("test continuation runner source\n", encoding="utf-8")

    source_rows = []
    for relative in FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = f"source_snapshot/{relative}"
        snapshot = parent / snapshot_path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        old = f"parent:{relative}\n"
        snapshot.write_text(old, encoding="utf-8")
        path.write_text((f"current:{relative}\n" if relative in UI_DRIFT[2:] else old), encoding="utf-8")
        source_rows.append({"path": relative, "snapshot_path": snapshot_path,
                            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})

    checkpoint = parent / "stage_04_turn90.zip"
    _zip_checkpoint(checkpoint)
    checkpoint_sha = fresh._sha256(checkpoint)
    (parent / "checkpoint_manifest.json").write_text(json.dumps({"checkpoints": [{
        "path": checkpoint.name, "sha256": checkpoint_sha,
        "num_timesteps": runner.PARENT_TIMESTEPS, "adam_optimizer_state_present": True,
    }]}), encoding="utf-8")
    protocol = {
        "schema": fresh.SCHEMA, "training": copy.deepcopy(TRAINING), "action_config": ACTION,
        "runtime_contract": {"runtime": "coupled-v7"}, "architecture": {"architecture": "LSTM128"},
        "config": copy.deepcopy(CONFIG), "robot_geometry": {"robot": "test"},
        "source_snapshot": {"files": source_rows},
        "evaluation": {"free_space": {"seeds": [87001, 87002, 87003, 87004], "plants": 0},
                       "field": {"seeds": [82001, 82002, 82003, 82004], "plants": 16}},
    }
    protocol_path = parent / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = fresh._protocol_sha(protocol_path)
    validation = {
        "status": "passed", "exit_code": 0,
        "checks": copy.deepcopy(CHECKS),
        "checkpoint": {"optimizer": copy.deepcopy(runner.EXPECTED_ADAM)},
    }
    validation_path = parent / "final_cpu_validation/final_cpu_validation.json"
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    validation_sha = fresh._sha256(validation_path)
    (validation_path.parent / "final_cpu_validation.exit.json").write_text(json.dumps({
        "actual_exit_code": 0, "report_sha256": validation_sha,
        "checkpoint_sha256": checkpoint_sha, "protocol_sha256": protocol_sha,
    }), encoding="utf-8")
    (parent / "run_status.json").write_text(json.dumps({
        "status": "completed", "exit_code": 0, "num_timesteps": runner.PARENT_TIMESTEPS,
    }), encoding="utf-8")
    (parent / "process_exit.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    (parent / "python.exit_code").write_text("0\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PARENT_PROTOCOL_SHA256", protocol_sha)
    monkeypatch.setattr(runner, "PARENT_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(runner, "PARENT_CPU_REPORT_SHA256", validation_sha)
    return root, parent


def _prepared_legacy_child(root: Path, parent: Path) -> Path:
    """Build a frozen historical child fixture without invoking disabled prepare."""
    output = root / "child"
    output.mkdir()
    parent_protocol = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    for relative in (*fresh._source_paths(), runner.RUNNER):
        source = root / relative
        snapshot_relative = f"source_snapshot/{relative}"
        snapshot = output / snapshot_relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        rows.append({"path": relative, "snapshot_path": snapshot_relative,
                     "sha256": fresh._sha256(source)})

    baseline = output / "parent_final_baseline.zip"
    parent_checkpoint = parent / "stage_04_turn90.zip"
    baseline.write_bytes(parent_checkpoint.read_bytes())
    protocol = {
        "schema": runner.SCHEMA,
        "status": "prepared",
        "parent": {
            "path": str(parent),
            "protocol_sha256": runner.PARENT_PROTOCOL_SHA256,
            "checkpoint": parent_checkpoint.name,
            "checkpoint_sha256": runner.PARENT_CHECKPOINT_SHA256,
            "num_timesteps": runner.PARENT_TIMESTEPS,
            "optimizer": runner.EXPECTED_ADAM,
        },
        "architecture": parent_protocol["architecture"],
        "action_config": parent_protocol["action_config"],
        "runtime_contract": parent_protocol["runtime_contract"],
        "config": {**CONFIG, "max_time": 20.0},
        "robot_geometry": parent_protocol["robot_geometry"],
        "training": {
            **copy.deepcopy(TRAINING),
            "training_seed": runner.TRAIN_SEED,
            "stage_seeds": [runner.TRAIN_SEED + i for i in range(len(fresh.STAGES))],
        },
        "initialization": {
            "baseline_path": baseline.name,
            "baseline_sha256": runner.PARENT_CHECKPOINT_SHA256,
            "expected_final_num_timesteps": runner.FINAL_TIMESTEPS,
        },
        "source_snapshot": {"files": rows},
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    (output / "run_status.json").write_text(json.dumps({
        "status": "prepared", "protocol_sha256": fresh._protocol_sha(protocol_path),
    }), encoding="utf-8")
    return output


@pytest.mark.parametrize("max_time", [20.0, 60.0])
def test_prepare_refuses_to_relabel_a_historical_newton_parent(parent_trial, max_time):
    root, parent = parent_trial
    output = root / "child"
    parent_bytes = (parent / "stage_04_turn90.zip").read_bytes()
    with pytest.raises(RuntimeError, match="pinned to an SB3/Newton .zip parent"):
        runner.prepare(parent, output, max_time)
    assert not output.exists()
    assert (parent / "stage_04_turn90.zip").read_bytes() == parent_bytes


def test_prepare_rejects_unapproved_parent_drift(parent_trial):
    root, parent = parent_trial
    (root / FILES[0]).write_text("unexpected training drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected current source drift"):
        runner._parent_audit(parent)


@pytest.mark.parametrize("tamper", ["baseline", "runner_source"])
def test_preflight_rejects_tampered_child_artifacts_and_launch_stays_legacy_only(parent_trial, tamper):
    root, parent = parent_trial
    output = _prepared_legacy_child(root, parent)
    if tamper == "baseline":
        with (output / "parent_final_baseline.zip").open("ab") as stream:
            stream.write(b"tamper")
        expected = "byte-identical"
    else:
        (root / runner.RUNNER).write_text("changed after freeze\n", encoding="utf-8")
        expected = "runner differs"
    with pytest.raises(ValueError, match=expected):
        runner.preflight(output)
    status_path = output / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "running"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires the Newton/SB3 optimizer"):
        runner.launch(output)
    assert not (output / "launch.lock").exists()


def test_launch_refuses_existing_claim_before_starting_simulator(parent_trial):
    root, parent = parent_trial
    output = _prepared_legacy_child(root, parent)
    lock = output / "launch.lock"
    lock.write_text("claimed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires the Newton/SB3 optimizer"):
        runner.launch(output)
    assert lock.read_text(encoding="utf-8") == "claimed\n"


def test_checkpoint_state_load_does_not_advance_torch_rng(tmp_path: Path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "state.zip"
    import io
    buffer = io.BytesIO()
    torch.save({"layer.weight": torch.arange(4, dtype=torch.float32)}, buffer)
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("policy.pth", buffer.getvalue())
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    state = runner._load_checkpoint_member(checkpoint, "policy.pth")
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(state["layer.weight"], torch.arange(4, dtype=torch.float32))


def test_adam_audit_matches_verified_parent_step_count():
    torch = pytest.importorskip("torch")
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(37)]
    optimizer = torch.optim.Adam(parameters)
    for parameter in parameters:
        optimizer.state[parameter] = {
            "step": torch.tensor(7607.0), "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    assert runner._adam_audit(optimizer.state_dict()) == runner.EXPECTED_ADAM
    optimizer.state[parameters[0]]["step"] = torch.tensor(7606.0)
    with pytest.raises(ValueError, match="verified parent audit"):
        runner._adam_audit(optimizer.state_dict())
