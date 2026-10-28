from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import scripts.evaluate_fast_stop_resume as evaluation


def _sample(distance: float, hold: float, *, time_s: float = 0.05,
            source: str = "root_com_vel_w") -> dict:
    return {
        "time_s": time_s,
        "distance_m": distance,
        "linear_speed_xy_norm_m_s": 0.02,
        "angular_speed_xyz_norm_rad_s": 0.03,
        "hold_s": hold,
        "speed_source": source,
    }


def _step(before_distance: float, after_distance: float,
          before_hold: float, after_hold: float,
          before_time: float = 0.0, after_time: float = 0.05) -> dict:
    return {
        "before": {"goal_distance_m": before_distance, "hold_s": before_hold,
                   "time_s": before_time},
        "after": {"goal_distance_m": after_distance, "hold_s": after_hold,
                  "time_s": after_time},
    }


def test_path_metrics_include_before_and_after_samples() -> None:
    row = {
        "trajectory": [
            _step(0.36, 0.50, 0.30, 0.10),
            _step(0.50, 0.60, 0.10, 0.00, before_time=0.05, after_time=0.10),
        ],
        "goal_stop": {
            "schema": "goal_stop_v1", "profile": "fast_stop_v1",
            "linear_speed_m_s": 0.02, "angular_speed_rad_s": 0.03, "hold_s": 0.00,
        },
    }
    metrics = evaluation.summarize_trajectory(
        row, [_sample(0.50, 0.10), _sample(0.60, 0.00, time_s=0.10)],
        distance_tolerance_m=0.4,
    )
    assert metrics["min_distance_m"] == pytest.approx(0.36)
    assert metrics["entered_radius"] is True
    assert metrics["max_hold_s"] == pytest.approx(0.30)
    assert metrics["kinematic_samples"] == 2
    assert row["trajectory"][0]["root_com_kinematics"]["speed_source"] == "root_com_vel_w"


def test_path_metrics_reject_misaligned_after_hold_and_timestamp() -> None:
    row = {
        "trajectory": [_step(0.7, 0.5, 0.0, 0.1)],
        "goal_stop": {
            "schema": "goal_stop_v1", "profile": "fast_stop_v1",
            "linear_speed_m_s": 0.02, "angular_speed_rad_s": 0.03, "hold_s": 0.1,
        },
    }
    with pytest.raises(ValueError, match="hold_s are misaligned"):
        evaluation.summarize_trajectory(
            row, [_sample(0.5, 0.0)], distance_tolerance_m=0.4,
        )

    row["trajectory"] = [_step(0.7, 0.5, 0.0, 0.0)]
    with pytest.raises(ValueError, match="time_s are misaligned"):
        evaluation.summarize_trajectory(
            row, [_sample(0.5, 0.0, time_s=0.06)], distance_tolerance_m=0.4,
        )


def test_path_metrics_reject_missing_snapshots_or_root_speed_fallback() -> None:
    row = {
        "trajectory": [_step(0.7, 0.5, 0.0, 0.0)],
        "goal_stop": {
            "schema": "goal_stop_v1", "profile": "fast_stop_v1",
            "linear_speed_m_s": 0.02, "angular_speed_rad_s": 0.03, "hold_s": 0.0,
        },
    }
    del row["trajectory"][0]["after"]["goal_distance_m"]
    with pytest.raises(ValueError, match="invalid distance, hold, or time"):
        evaluation.summarize_trajectory(row, [_sample(0.5, 0.0)], distance_tolerance_m=0.4)

    row["trajectory"] = [_step(0.7, 0.5, 0.0, 0.0)]
    with pytest.raises(ValueError, match="native root COM velocity"):
        evaluation.summarize_trajectory(
            row, [_sample(0.5, 0.0, source="task_forward_speed_fallback")],
            distance_tolerance_m=0.4,
        )


def test_paired_comparison_rejects_mismatched_conditions() -> None:
    with pytest.raises(ValueError, match="paired evaluation conditions are invalid"):
        evaluation._require_valid_pairing(
            {"valid": False, "paired_seed_count": 4,
             "mismatches": [{"seed": 87001, "differences": ["goal_xy"]}]},
            evaluation.SPLIT_SEEDS["rough_a"], {"rows": []}, {"rows": []},
        )


def _write_checkpoint(path: Path, *, steps: int, action: dict, encoder: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data", json.dumps({
            "num_timesteps": steps, "n_steps": 128, "n_envs": 4, "gamma": 1.0,
            "field_action_config": action, "field_encoder_config": encoder,
            "policy_kwargs": {"lstm_hidden_size": 512, "n_lstm_layers": 1},
        }))
        archive.writestr("policy.pth", b"policy")
        archive.writestr("policy.optimizer.pth", b"adam")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_checkpoint_selection_uses_summary_artifact_not_same_step_periodic(tmp_path: Path) -> None:
    control = tmp_path / "control"
    action = {"schema": "direct-track-speed-v1"}
    encoder = {"schema": "sensor-minimal-v6"}
    parent_protocol = {"action_config": action, "architecture": {"field_encoder_config": encoder}}
    final_path = control / "stage_01_global_random.zip"
    final_sha = _write_checkpoint(final_path, steps=evaluation.EXPECTED_TIMESTEPS,
                                  action=action, encoder=encoder)
    periodic_path = control / "stage_1_global_random" / "checkpoints" / "field_policy_global_random_196608_steps.zip"
    periodic_sha = _write_checkpoint(periodic_path, steps=evaluation.EXPECTED_TIMESTEPS,
                                     action=action, encoder=encoder)
    manifest = {"checkpoints": [
        {"path": "stage_01_global_random.zip", "sha256": final_sha,
         "num_timesteps": evaluation.EXPECTED_TIMESTEPS, "adam_optimizer_state_present": True},
        {"path": "stage_1_global_random/checkpoints/field_policy_global_random_196608_steps.zip",
         "sha256": periodic_sha, "num_timesteps": evaluation.EXPECTED_TIMESTEPS,
         "adam_optimizer_state_present": True},
    ]}
    summary = {"checkpoint_final": str(final_path), "checkpoint_final_sha256": final_sha,
               "actual_final_num_timesteps": evaluation.EXPECTED_TIMESTEPS}
    selected = evaluation.select_final_checkpoint(control, summary, manifest, parent_protocol)
    assert Path(selected["path"]) == final_path
    assert selected["sha256"] == final_sha

    summary["checkpoint_final"] = str(periodic_path)
    summary["checkpoint_final_sha256"] = periodic_sha
    with pytest.raises(ValueError, match="must not be a periodic checkpoint"):
        evaluation.select_final_checkpoint(control, summary, manifest, parent_protocol)


def test_completion_gate_rejects_nonzero_real_supervisor_exit(tmp_path: Path, monkeypatch) -> None:
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    for name, data in {
        "process_exit.json": {"source": "resume_fast_stop_supervisor_waited_for_worker",
                              "actual_exit_code": 143, "exit_code": 143},
        "python.exit_code": None,
        "run_status.json": {}, "resume_summary.json": {}, "protocol.json": {},
        "checkpoint_manifest.json": {}, "initialization.json": {},
    }.items():
        path = control / name
        if name == "python.exit_code":
            path.write_text("143\n", encoding="ascii")
        else:
            path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="real supervisor-recorded exit code 0"):
        evaluation.validate_completed_trial(control)
