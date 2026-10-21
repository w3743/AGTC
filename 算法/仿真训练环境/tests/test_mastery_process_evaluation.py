from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.evaluate_fast_stop_mastery as sampler
import scripts.evaluate_fast_stop_resume as trajectory_runner


def _context(root: Path) -> dict:
    trial = root / "trial"
    return {
        "trial": trial,
        "protocol_sha256": "p" * 64,
        "source_file_count": 47,
        "source_sha256": "s" * 64,
        "source_hashes": [],
        "run_status": {"status": "running"},
        "transfer_config": {"terrain_enabled": True, "max_time": 60.0, "plants": 0},
        "checkpoint": {"path": str(trial / "checkpoints" / "latest.zip"),
                       "sha256": "c" * 64, "num_timesteps": 106496, "gamma": 1.0,
                       "policy_architecture": {"lstm_hidden_size": 512, "n_lstm_layers": 1}},
        "protocol": {
            "action_config": {"schema": "test-action"},
            "config": {"damage_metric": "stem-contact-v2"},
            "profile_physics_contract": {"terrain_config": {"terrain_enabled": True}},
            "goal_objective_contract": {"goal_stop": {"distance_tolerance_m": 0.4}},
        },
    }


def _member() -> dict:
    return {"name": "rough_a", "plants": 0, "goal_distance": None,
            "goal_angle": 3.141592653589793, "environment_seed": 80003,
            "seeds": [87001, 87002, 87003, 87004]}


def test_evaluator_reuses_frozen_rough_config_and_member_seeds(tmp_path, monkeypatch):
    context, member = _context(tmp_path), _member()
    seen = {}

    def fake_config(**values):
        seen["config"] = values
        return SimpleNamespace(**values)

    def fake_evaluate(output, cfg, checkpoint, label, seeds, plants, angle, env_seed,
                      device, physics, objective):
        seen.update(output=output, checkpoint=checkpoint, label=label, seeds=seeds,
                     plants=plants, angle=angle, environment_seed=env_seed, device=device,
                     physics=physics, objective=objective)
        return {"ok": True}

    monkeypatch.setattr(sampler.mastery, "Config", fake_config)
    monkeypatch.setattr(sampler.trajectory_runner, "_evaluate_checkpoint", fake_evaluate)
    context["protocol"].update(profile_physics_contract={"terrain_config": {"terrain_enabled": True}},
                               goal_objective_contract={"schema": "frozen-objective"})
    result = sampler._evaluate_member(context, tmp_path / "out", member, "cpu")

    assert result == {"ok": True}
    assert seen["config"] == context["transfer_config"]
    assert seen["config"]["terrain_enabled"] is True
    assert seen["seeds"] == tuple(member["seeds"])
    assert seen["environment_seed"] == member["environment_seed"]
    assert seen["physics"] == context["protocol"]["profile_physics_contract"]
    assert seen["objective"] == context["protocol"]["goal_objective_contract"]


def test_transfer_config_rejects_a_noncanonical_terrain_setting(monkeypatch):
    canonical = {"environment_config": {"terrain_enabled": True}, "members": []}
    monkeypatch.setattr(sampler.mastery, "_fast_stop_transfer_probe", lambda: canonical)
    with pytest.raises(ValueError, match="differs from the frozen"):
        sampler._transfer_config({"transfer_probe": {
            "environment_config": {"terrain_enabled": False}, "members": []
        }})


def _report(context: dict, member: dict, *, source: str = "root_com_vel_w",
            before: dict | None = None) -> dict:
    protocol = context["protocol"]
    step = {
        "before": before or {"goal_distance_m": 0.36, "hold_s": 0.30, "time_s": 0.0},
        "after": {"goal_distance_m": 0.50, "hold_s": 0.10, "time_s": 0.05},
        "root_com_kinematics": {
            "time_s": 0.05, "distance_m": 0.50, "linear_speed_xy_norm_m_s": 0.02,
            "angular_speed_xyz_norm_rad_s": 0.03, "hold_s": 0.10,
            "speed_source": source,
        },
    }
    rows = [{"seed": seed, "reason": "time_limit", "success": False,
             "trajectory": [copy.deepcopy(step)], "goal_stop": {"schema": "goal_stop_v1",
                 "profile": "fast_stop_v1", "linear_speed_m_s": 0.02,
                 "angular_speed_rad_s": 0.03, "hold_s": 0.10}}
            for seed in member["seeds"]]
    shared = {"schema": "test", "action_config": protocol["action_config"],
              "checkpoint_action_config": protocol["action_config"],
              "checkpoint": context["checkpoint"]["path"], "distance_cost": 0.0,
              "goal_distance": None, "goal_angle_rad": member["goal_angle"],
              "mode": "goto", "deterministic": True, "seeds": member["seeds"],
              "drive_config": {"drive": "same"}, "checkpoint_drive_config": {"drive": "same"},
              "observation_config": {"observation": "same"},
              "checkpoint_observation_config": {"observation": "same"},
              "damage_config": {"metric": "stem-contact-v2"},
              "checkpoint_damage_config": {"metric": "stem-contact-v2"},
              "damage_metric": "stem-contact-v2", "rows": rows,
              "evaluation_config": {"environment_seed": member["environment_seed"],
                  "plants": member["plants"], "goal_distance_m": None,
                  "goal_angle_deg": 180.0, "goal_sampling_profile": "random_rough",
                  "terrain_config": protocol["profile_physics_contract"]["terrain_config"],
                  "goal_objective_contract": protocol["goal_objective_contract"]}}
    for row in rows:
        if source == "root_com_vel_w":
            row["path_metrics"] = trajectory_runner.summarize_trajectory(
                row, [row["trajectory"][0]["root_com_kinematics"]], distance_tolerance_m=0.4,
            )
        else:
            row["path_metrics"] = {"min_distance_m": 0.36, "entered_radius": True,
                                   "distance_tolerance_m": 0.4, "max_hold_s": 0.30,
                                   "kinematic_samples": 1}
    return shared


def test_report_accepts_initial_entry_then_leaving_and_keeps_peak_hold(tmp_path):
    context, member = _context(tmp_path), _member()
    report = _report(context, member)
    # Episode 1 enters at its initial sample and later leaves; episode 2 begins
    # outside, enters on its terminal sample, and must not inherit episode 1's hold.
    first = report["rows"][0]
    first["trajectory"] = [{
        "before": {"goal_distance_m": 0.36, "hold_s": 0.30, "time_s": 0.0},
        "after": {"goal_distance_m": 0.50, "hold_s": 0.10, "time_s": 0.05},
        "root_com_kinematics": {"time_s": 0.05, "distance_m": 0.50,
            "linear_speed_xy_norm_m_s": 0.02, "angular_speed_xyz_norm_rad_s": 0.03,
            "hold_s": 0.10, "speed_source": "root_com_vel_w"},
    }]
    second = report["rows"][1]
    second["trajectory"] = [{
        "before": {"goal_distance_m": 0.70, "hold_s": 0.0, "time_s": 0.0},
        "after": {"goal_distance_m": 0.39, "hold_s": 0.20, "time_s": 0.05},
        "root_com_kinematics": {"time_s": 0.05, "distance_m": 0.39,
            "linear_speed_xy_norm_m_s": 0.02, "angular_speed_xyz_norm_rad_s": 0.03,
            "hold_s": 0.20, "speed_source": "root_com_vel_w"},
    }]
    first["goal_stop"]["hold_s"] = 0.10
    second["goal_stop"]["hold_s"] = 0.20
    for row in (first, second):
        row["path_metrics"] = trajectory_runner.summarize_trajectory(
            row, [row["trajectory"][0]["root_com_kinematics"]], distance_tolerance_m=0.4,
        )

    sampler._validate_report(report, context, member)
    assert first["path_metrics"]["min_distance_m"] == pytest.approx(0.36)
    assert first["path_metrics"]["entered_radius"] is True
    assert first["path_metrics"]["max_hold_s"] == pytest.approx(0.30)
    assert second["path_metrics"]["min_distance_m"] == pytest.approx(0.39)
    assert second["path_metrics"]["entered_radius"] is True
    assert second["path_metrics"]["max_hold_s"] == pytest.approx(0.20)


def test_report_rejects_mismatched_evaluation_contract(tmp_path):
    context, member = _context(tmp_path), _member()
    report = _report(context, member)
    report["evaluation_config"]["environment_seed"] += 1
    with pytest.raises(ValueError, match="differs from frozen config/objective/seeds"):
        sampler._validate_report(report, context, member)


@pytest.mark.parametrize("source,missing_before,error", [
    ("task_speed_fallback", False, "native root COM velocity"),
    ("root_com_vel_w", True, "missing before snapshot"),
])
def test_report_rejects_fallback_or_missing_before_sample(tmp_path, source, missing_before, error):
    context, member = _context(tmp_path), _member()
    report = _report(context, member, source=source)
    if missing_before:
        report["rows"][0]["trajectory"][0]["before"] = None
    with pytest.raises(ValueError, match=error):
        sampler._validate_report(report, context, member)


def test_check_is_read_only_and_never_allows_launch(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "check_trial", lambda *_a, **_k: context)
    output = tmp_path / "new_evaluation"
    result = sampler.run(context["trial"], output, launch=False)
    assert result["launch_allowed"] is False
    assert not output.exists()


def test_incomplete_trial_refuses_launch_without_creating_output(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "check_trial", lambda *_a, **_k: context)

    def incomplete(*_args, **_kwargs):
        raise ValueError("trial has not completed with real exit 0")

    monkeypatch.setattr(sampler.goal_viewer, "validate_completed_trial", incomplete)
    output = tmp_path / "new_evaluation"
    with pytest.raises(ValueError, match="not completed"):
        sampler.run(context["trial"], output, launch=True)
    assert not output.exists()


def test_parallel_simulation_refuses_launch_before_output_creation(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "check_trial", lambda *_a, **_k: context)
    monkeypatch.setattr(sampler.goal_viewer, "validate_completed_trial",
                        lambda *_a, **_k: {"protocol_sha256": context["protocol_sha256"]})
    monkeypatch.setattr(sampler, "_assert_no_parallel_simulation",
                        lambda: (_ for _ in ()).throw(RuntimeError("parallel Isaac process")))
    output = tmp_path / "new_evaluation"
    with pytest.raises(RuntimeError, match="parallel Isaac process"):
        sampler.run(context["trial"], output, launch=True)
    assert not output.exists()


@pytest.mark.parametrize("argv", [
    ["/usr/bin/python", "scripts/train_isaac_mastery.py", "--launch"],
    ["/venv/bin/python", "scripts/evaluate_fast_stop_resume.py"],
    ["python", "scripts/view_fast_stop_goal.py", "--check"],
    ["python", "-m", "scripts.evaluate_fast_stop_mastery"],
    ["python", "-m", "scripts.train_isaac_mastery"],
    ["python", "-m", "scripts.view_fast_stop_goal"],
    ["python", "-m", "isaac_field", "viewer"],
    ["python", "-m", "isaac_field", "smoke"],
    ["python", "-m", "isaac_field", "train"],
    ["python", "-m", "isaac_field", "evaluate"],
    ["python", "-m", "isaac_field", "adversary"],
    ["python", "-m", "isaac_field.viewer"],
])
def test_parallel_process_classifier_blocks_python_simulation_argv(argv):
    assert sampler._is_parallel_simulation_argv(argv, 12345, current_pid=54321)


def test_parallel_process_classifier_ignores_wrapper_self_and_lmstudio():
    assert not sampler._is_parallel_simulation_argv(
        ["/usr/bin/bash", "scripts/isaaclab_python.sh",
         "scripts/evaluate_fast_stop_mastery.py", "--launch"],
        12345, current_pid=54321,
    )
    assert not sampler._is_parallel_simulation_argv(
        ["/venv/bin/python", "scripts/train_isaac_mastery.py", "--launch"],
        sampler.os.getpid(),
    )
    assert not sampler._is_parallel_simulation_argv(["/opt/lmstudio/LM Studio"], 12345,
                                                     current_pid=54321)


@pytest.mark.parametrize("argv", [
    ["python", "-m", "external_scripts.train_isaac_mastery"],
    ["python", "-m", "scripts.view_policy_netron"],
])
def test_parallel_process_classifier_ignores_non_simulation_modules(argv):
    assert not sampler._is_parallel_simulation_argv(argv, 12345, current_pid=54321)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "check_trial", lambda *_a, **_k: context)
    output = tmp_path / "existing_evaluation"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sampler.run(context["trial"], output, launch=True)
    assert marker.read_text(encoding="utf-8") == "preserve"
