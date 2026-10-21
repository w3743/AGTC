"""CPU-only protocol and pairing checks for the fresh foundation runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import scripts.train_isaac_foundation_fresh as runner


FAKE_SOURCES = (
    "isaac_field/policy.py",
    "scripts/isaaclab_python.sh",
    "scripts/train_isaac_foundation.py",
    "scripts/improve_isaac_policy.py",
    "scripts/compare_isaac_exploration.py",
    "scripts/train_isaac_foundation_fresh.py",
)
FAKE_ARCHITECTURE = {
    "field_encoder_config": {
        "schema": runner.ENCODER_SCHEMA,
        "branch_dims": {"lidar": 32, "depth": 64, "detections": 32, "state": 32, "feedback": 32},
        "depth_pool": [8, 6], "radar_pool": 50, "features_dim": 192,
    },
    "input_contract": runner.INPUT_SCHEMA,
    "action_config": runner._action_config(),
    "goal_gain": 1.0,
}
FAKE_CONFIG = {
    "plants": 0, "terrain_enabled": False, "max_time": 20.0,
    "damage_penalty": 20.0, "goal_progress_budget": 5.0,
    "damage_metric": "stem-contact-v2",
}


@pytest.fixture
def trial_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    for relative in FAKE_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source: {relative}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "_source_paths", lambda: FAKE_SOURCES)
    monkeypatch.setattr(runner, "_runtime_contract", lambda: {
        "valid": True, "action_config": runner._action_config(),
    })
    monkeypatch.setattr(runner, "_experiment_architecture", lambda: copy.deepcopy(FAKE_ARCHITECTURE))
    monkeypatch.setattr(runner, "_config_dict", lambda: copy.deepcopy(FAKE_CONFIG))
    monkeypatch.setattr(runner, "_robot_geometry", lambda: {"spec": "test"})
    monkeypatch.setattr(runner, "_package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(runner, "_isaac_commit", lambda: "test-commit")
    return root


def test_prepare_snapshots_sources_and_preflight_accepts_unchanged_trial(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh"
    result = runner.prepare(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))

    assert protocol["initialization"]["checkpoint"] is None
    assert protocol["initialization"]["baseline_path"] == "untrained_baseline.zip"
    assert protocol["training"]["timesteps_total"] == 196_608
    assert protocol["training"]["stage_seeds"] == [69_001, 69_002, 69_003, 69_004]
    assert protocol["action_config"] == runner._action_config()
    assert protocol["runtime_contract"]["action_config"] == runner._action_config()
    assert protocol["architecture"]["action_config"] == runner._action_config()
    assert protocol["architecture"]["input_contract"] == "sensor-minimal-v6"
    assert protocol["evaluation"]["blind_seeds"] == {
        "range_label": "830xx", "used": False, "observed": False,
    }
    assert status["protocol_sha256"] == result["protocol_sha256"]
    assert runner.preflight(output)["schema"] == runner.SCHEMA


def test_preflight_rejects_live_source_drift(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh"
    runner.prepare(output)
    (trial_root / FAKE_SOURCES[0]).write_text("changed policy\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source snapshot/live hash mismatch"):
        runner.preflight(output)


def test_live_environment_contract_requires_pi_v6_drive_interface_and_v6_observation():
    env = SimpleNamespace(
        action_config=runner._action_config(),
        drive_config={"interface": "independent_left_right_track_speed_explicit_pi_v6"},
        observation_config={"schema": "sensor-minimal-v6"},
    )

    contract = runner._live_environment_contract(env)
    assert contract["action_config"] == runner._action_config()
    assert contract["drive_config"]["interface"] == "independent_left_right_track_speed_explicit_pi_v6"
    assert contract["observation_config"]["schema"] == "sensor-minimal-v6"
    for old_interface in (
        "independent_left_right_track_speed_explicit_v5",
        "independent_left_right_track_speed_explicit_v4",
        "forward_and_differential_track_speed_v2",
    ):
        env.drive_config = {"interface": old_interface}
        with pytest.raises(ValueError, match="unexpected live drive interface"):
            runner._live_environment_contract(env)


@pytest.mark.parametrize("checkpoint_action,accepted", [
    (None, False),
    ({"schema": "forward-differential-v2"}, False),
    ("current", True),
])
def test_checkpoint_manifest_requires_and_records_action_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint_action, accepted: bool,
):
    monkeypatch.setattr(runner, "_action_config", lambda: "current")
    output = tmp_path / "trial"
    output.mkdir()
    path = output / "checkpoint.zip"
    data = {"num_timesteps": 0}
    if checkpoint_action is not None:
        data["field_action_config"] = checkpoint_action
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
        archive.writestr("policy.optimizer.pth", b"optimizer")

    if accepted:
        manifest = runner._checkpoint_manifest(output)
        assert manifest[0]["action_config"] == "current"
    else:
        with pytest.raises(ValueError, match="checkpoint action contract"):
            runner._checkpoint_manifest(output)


@pytest.mark.parametrize("field,value", [
    ("input_contract", "old-depth-two-channel-v2"),
    ("action_config", {"schema": "forward-differential-v2"}),
    ("goal_gain", 100.0),
    ("field_encoder_config", {"schema": "old-encoder", "features_dim": 128}),
])
def test_preflight_rejects_old_input_gain_or_encoder(
    trial_root: Path, field: str, value: object,
) -> None:
    output = trial_root / "results" / "fresh"
    runner.prepare(output)
    protocol_path = output / "protocol.json"
    status_path = output / "run_status.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["architecture"][field] = value
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["protocol_sha256"] = runner._sha256(protocol_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared encoder"):
        runner.preflight(output)


def _paired_report(seeds: tuple[int, ...], *, variant: str) -> dict:
    rows = []
    for seed in seeds:
        rows.append({
            "seed": seed,
            "reset_initial_condition": {
                "seed": seed,
                "plant_xy_sha256": "same-plants",
                "start_pose_xy_yaw": [1.0, 2.0, 0.1],
                "goal_xy": [3.0, 4.0],
            },
            "trajectory_start_matches_reset": True,
            "variant": variant,
        })
    return {"rows": rows}


@pytest.mark.parametrize("case,expected", [
    ("same", True), ("missing", False), ("duplicate", False), ("different", False),
])
def test_four_seed_pairing_checks_exact_rows_and_initial_conditions(
    case: str, expected: bool,
) -> None:
    seeds = tuple(range(82_001, 82_005))
    baseline = _paired_report(seeds, variant="baseline")
    final = _paired_report(seeds, variant="final")
    if case == "missing":
        final["rows"].pop()
    elif case == "duplicate":
        final["rows"][-1]["seed"] = seeds[0]
    elif case == "different":
        final["rows"][0]["reset_initial_condition"]["goal_xy"] = [3.01, 4.0]

    result = runner._paired_dev_conditions({"baseline": baseline, "final": final}, seeds)
    assert result["valid"] is expected
    assert result["expected_seed_count"] == 4


def test_main_refusing_existing_output_preserves_protocol_and_status(
    trial_root: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = trial_root / "existing"
    output.mkdir()
    protocol = output / "protocol.json"
    status = output / "run_status.json"
    protocol.write_text('{"immutable":true}\n', encoding="utf-8")
    status.write_text('{"status":"completed","exit_code":0}\n', encoding="utf-8")
    before_protocol, before_status = protocol.read_bytes(), status.read_bytes()

    assert runner.main(["--prepare", "--output", str(output)]) == 1
    capsys.readouterr()
    assert protocol.read_bytes() == before_protocol
    assert status.read_bytes() == before_status


def test_main_launch_preflight_refusal_preserves_running_status(
    trial_root: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = trial_root / "in_progress"
    output.mkdir()
    protocol = output / "protocol.json"
    status = output / "run_status.json"
    protocol.write_text('{}\n', encoding="utf-8")
    status.write_text('{"status":"running","phase":"training"}\n', encoding="utf-8")
    before_protocol, before_status = protocol.read_bytes(), status.read_bytes()

    assert runner.main(["--launch", "--output", str(output)]) == 1
    capsys.readouterr()
    assert protocol.read_bytes() == before_protocol
    assert status.read_bytes() == before_status
