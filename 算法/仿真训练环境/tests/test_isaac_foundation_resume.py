from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import scripts.compare_isaac_exploration as compare
import scripts.train_isaac_foundation as runner


def _zip_checkpoint(path: Path, steps: int, *, optimizer_payload: bytes | None = b"adam-state") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps}))
        archive.writestr("policy.pth", b"policy-state")
        if optimizer_payload is not None:
            archive.writestr("policy.optimizer.pth", optimizer_payload)


@pytest.fixture
def trial_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    for relative in (*runner.RUNTIME_FILES, runner.RUNNER):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"frozen source: {relative}\n", encoding="utf-8")

    baseline = root / "results" / "baseline.zip"
    _zip_checkpoint(baseline, 53_248)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "BASELINE", baseline)
    monkeypatch.setattr(compare, "ROOT", root)
    return root


def _legacy_protocol(root: Path, output: Path, *, run_status: str = "prepared") -> Path:
    output.mkdir(parents=True)
    snapshot = runner._snapshot(output)
    protocol = {
        "schema": "isaac-field-foundation-curriculum-v2",
        "status": "prepared",
        "source_snapshot": snapshot,
        "baseline_checkpoint": {
            "path": runner.BASELINE.relative_to(root).as_posix(),
            "sha256": runner._sha256(runner.BASELINE),
            "num_timesteps": 53_248,
        },
        "config": runner._field_config_dict(),
        "robot_geometry": runner._robot_geometry_dict(),
        "training": runner._training_protocol(),
    }
    runner._json(output / "protocol.json", protocol)
    runner._json(output / "run_status.json", {
        "status": run_status,
        "protocol_sha256": runner._sha256(output / "protocol.json"),
    })
    return output


def _prepared_parent(root: Path) -> tuple[Path, Path]:
    parent = _legacy_protocol(root, root / "results" / "failed_parent", run_status="failed")
    checkpoint = parent / "stage_1_near" / "checkpoints" / "field_policy_near_8192_steps.zip"
    _zip_checkpoint(checkpoint, 8192)
    # The historical audit permits viewer-only drift and does not compare the
    # frozen runner source with today's disabled entrypoint.
    (root / runner.RUNNER).write_text("updated recovery runner\n", encoding="utf-8")
    (root / "isaac_field" / "viewer.py").write_text("viewer idle-loop fix\n", encoding="utf-8")
    return parent, checkpoint


def test_resume_schedule_uses_only_the_remaining_near_steps() -> None:
    steps = runner._resume_stage_timesteps(8192)
    assert steps == (24_576, 32_768, 65_536, 65_536)
    assert sum(steps) == 188_416
    assert 8192 + sum(steps) == 196_608
    with pytest.raises(ValueError):
        runner._resume_stage_timesteps(32_768)


def test_legacy_training_entrypoints_fail_clearly_without_creating_protocols(
    trial_root: Path,
) -> None:
    output = trial_root / "results" / "disabled_newton_trial"
    with pytest.raises(RuntimeError, match="Newton/SB3 protocols.*disabled for new runs"):
        runner.prepare(output)
    assert not output.exists()
    with pytest.raises(RuntimeError, match="SB3 .zip checkpoints.*cannot create a PhysX RSL run"):
        runner.launch(output, "cpu")


def test_legacy_preflight_remains_read_only_for_existing_protocols(trial_root: Path) -> None:
    output = _legacy_protocol(trial_root, trial_root / "results" / "historical")
    before_protocol = (output / "protocol.json").read_bytes()
    before_status = (output / "run_status.json").read_bytes()

    protocol = runner.preflight(output)

    assert protocol["schema"] == "isaac-field-foundation-curriculum-v2"
    assert protocol["training"]["algorithm"] == "SB3-Contrib RecurrentPPO"
    assert (output / "protocol.json").read_bytes() == before_protocol
    assert (output / "run_status.json").read_bytes() == before_status


def test_resume_forensics_reads_original_zip_and_source_snapshot_without_rewriting(
    trial_root: Path,
) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    protocol_before = (parent / "protocol.json").read_bytes()
    checkpoint_before = checkpoint.read_bytes()

    evidence = runner._validate_resume_parent(parent, checkpoint)

    assert evidence["checkpoint"]["num_timesteps"] == 8192
    assert evidence["checkpoint"]["zip_integrity"] is True
    assert evidence["checkpoint"]["stable_during_verification"] is True
    assert len(evidence["parent_runtime_source_differences"]) == 1
    assert evidence["parent_runtime_source_differences"][0]["path"] == "isaac_field/viewer.py"
    assert (parent / "protocol.json").read_bytes() == protocol_before
    assert checkpoint.read_bytes() == checkpoint_before
    assert compare._checkpoint_steps(checkpoint) == 8192


def test_resume_audit_rejects_a_running_parent_and_final_artifacts(trial_root: Path) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    (parent / "run_status.json").write_text(
        json.dumps({"status": "running", "phase": "training"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="terminal failed or interrupted"):
        runner._validate_resume_parent(parent, checkpoint)

    (parent / "run_status.json").write_text(
        json.dumps({"status": "failed", "exception": "test interruption"}), encoding="utf-8"
    )
    (parent / "comparison_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already has a final checkpoint or comparison"):
        runner._validate_resume_parent(parent, checkpoint)


@pytest.mark.parametrize("checkpoint_case", ["wrong_filename", "missing_optimizer", "empty_optimizer"])
def test_resume_forensics_requires_the_designated_checkpoint_and_adam_payload(
    trial_root: Path, checkpoint_case: str
) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    if checkpoint_case == "wrong_filename":
        checkpoint = checkpoint.with_name("field_policy_near_8192_steps_copy.zip")
        _zip_checkpoint(checkpoint, 8192)
    elif checkpoint_case == "missing_optimizer":
        _zip_checkpoint(checkpoint, 8192, optimizer_payload=None)
    else:
        _zip_checkpoint(checkpoint, 8192, optimizer_payload=b"")

    with pytest.raises(ValueError, match="checkpoint|Adam optimizer"):
        runner._validate_resume_parent(parent, checkpoint)
