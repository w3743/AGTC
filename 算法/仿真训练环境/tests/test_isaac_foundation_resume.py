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
    monkeypatch.setattr(compare, "SOURCE_FILES", runner.RUNTIME_FILES)
    return root


def _prepared_parent(root: Path) -> tuple[Path, Path]:
    parent = root / "results" / "failed_parent"
    runner.prepare(parent)
    (parent / "run_status.json").write_text(
        json.dumps({"status": "failed", "exception": "test interruption"}), encoding="utf-8"
    )
    checkpoint = parent / "stage_1_near" / "checkpoints" / "field_policy_near_8192_steps.zip"
    _zip_checkpoint(checkpoint, 8192)
    # The parent runner snapshot is expected to remain valid even though the new
    # recovery child freezes the current runner version.
    (root / runner.RUNNER).write_text("updated recovery runner\n", encoding="utf-8")
    # Only the explicitly allowed viewer-only source drift may be accepted.
    (root / "isaac_field" / "viewer.py").write_text(
        "viewer idle-loop fix\n", encoding="utf-8"
    )
    return parent, checkpoint


def test_resume_schedule_uses_only_the_remaining_near_steps() -> None:
    steps = runner._resume_stage_timesteps(8192)
    assert steps == (24_576, 32_768, 65_536, 65_536)
    assert sum(steps) == 188_416
    assert 8192 + sum(steps) == 196_608
    with pytest.raises(ValueError):
        runner._resume_stage_timesteps(32_768)


def test_fresh_prepare_keeps_the_existing_protocol_schema(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh"
    prepared = runner.prepare(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert prepared["resume"] is False
    assert protocol["schema"] == "isaac-field-foundation-curriculum-v2"
    assert protocol["training"]["timesteps_total"] == 196_608
    assert "recovery_manifest" not in protocol
    frozen_paths = {row["path"] for row in protocol["source_snapshot"]["files"]}
    assert set(runner.RUNTIME_FILES) | {runner.RUNNER} == frozen_paths
    assert "isaac_field/action_contract.py" in frozen_paths
    assert runner.preflight(output)["schema"] == protocol["schema"]


def test_resume_child_records_hashes_and_preflights(trial_root: Path) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    child = parent / "resume_8192"
    result = runner.prepare(child, resume_parent=parent, resume_checkpoint=checkpoint)
    protocol = json.loads((child / "protocol.json").read_text(encoding="utf-8"))
    recovery_path = child / protocol["recovery_manifest"]["path"]
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))

    assert result["resume"] is True
    assert protocol["schema"] == runner.RESUME_SCHEMA
    assert protocol["training"]["timesteps_total"] == 196_608
    assert recovery["parent"]["protocol_sha256"]
    assert recovery["parent"]["run_status_sha256"]
    assert recovery["checkpoint"]["sha256"]
    assert recovery["parent_config_sha256"]
    assert recovery["parent_robot_geometry_sha256"]
    source_differences = recovery["parent_runtime_source_differences"]
    assert len(source_differences) == 1
    assert source_differences[0]["path"] == "isaac_field/viewer.py"
    assert source_differences[0]["parent_snapshot_sha256"] != source_differences[0]["current_sha256"]
    assert source_differences[0]["reason"] == (
        "viewer-only UI/idle updates; training environment and policy sources remain unchanged"
    )
    assert source_differences[0]["current_sha256"] == runner._sha256(
        trial_root / "isaac_field" / "viewer.py"
    )
    assert recovery["training_plan"]["timesteps_per_stage_this_run"] == [
        24_576, 32_768, 65_536, 65_536
    ]
    assert recovery["training_plan"]["additional_timesteps"] == 188_416
    assert recovery["training_plan"]["expected_final_num_timesteps"] == 196_608
    assert recovery["continuation_semantics"]["bitwise_resume"] is False
    assert runner.preflight(child)["schema"] == runner.RESUME_SCHEMA


@pytest.mark.parametrize("tampered_file", ["checkpoint", "parent_protocol", "parent_status"])
def test_resume_preflight_rejects_changed_parent_evidence(
    trial_root: Path, tampered_file: str
) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    child = parent / "resume_8192"
    runner.prepare(child, resume_parent=parent, resume_checkpoint=checkpoint)
    assert runner.preflight(child)["schema"] == runner.RESUME_SCHEMA

    if tampered_file == "checkpoint":
        with zipfile.ZipFile(checkpoint, "a") as archive:
            archive.writestr("tampered", b"changed")
    elif tampered_file == "parent_protocol":
        with (parent / "protocol.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        (parent / "run_status.json").write_text(
            json.dumps({"status": "running", "phase": "unexpected"}), encoding="utf-8"
        )

    with pytest.raises((ValueError, zipfile.BadZipFile)):
        runner.preflight(child)


def test_prepare_rejects_running_parent_and_unpaired_resume_arguments(
    trial_root: Path,
) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    child = parent / "resume_running"
    (parent / "run_status.json").write_text(
        json.dumps({"status": "running", "phase": "training"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="terminal failed or interrupted"):
        runner.prepare(child, resume_parent=parent, resume_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="provided together"):
        runner.prepare(trial_root / "results" / "unpaired", resume_parent=parent)


def test_prepare_rejects_a_parent_that_already_has_final_artifacts(trial_root: Path) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    (parent / "comparison_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already has a final checkpoint or comparison"):
        runner.prepare(parent / "resume_final", resume_parent=parent, resume_checkpoint=checkpoint)


def test_prepare_rejects_non_viewer_runtime_drift(trial_root: Path) -> None:
    parent, checkpoint = _prepared_parent(trial_root)
    (trial_root / "isaac_field" / "task.py").write_text(
        "unapproved training source change\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="training/runtime source changed"):
        runner.prepare(parent / "resume_changed_runtime", resume_parent=parent,
                       resume_checkpoint=checkpoint)


@pytest.mark.parametrize("checkpoint_case", ["wrong_filename", "missing_optimizer", "empty_optimizer"])
def test_resume_requires_the_designated_checkpoint_and_adam_payload(
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
