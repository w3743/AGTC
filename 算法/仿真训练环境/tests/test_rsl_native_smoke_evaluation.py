"""CPU-only reconstruction and report tests for the RSL native smoke path."""
from __future__ import annotations

from dataclasses import asdict
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.evaluate_fast_stop_mastery as evaluator


def _contract(profile="fast_stop_v1", *, remaining=None):
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from scripts import train_isaac_foundation_fresh as fresh

    task = fresh._config_dict()
    task.update(fresh.FAST_STOP_CONFIG)
    task["goto_reward_profile"] = profile
    if remaining is not None:
        task["fast_stop_failure_remaining_time_cost"] = remaining
    task.update(max_time=60.0, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA, rock_density=0.0)
    guides, stability = fresh._rough_profile_components(fresh.PROFILE_RANDOM_ROUGH_FAST_STOP)
    return {
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "task_config": task,
        "front_guide_spec": asdict(guides),
        "body_stability_config": stability.as_dict(),
        "observation_config": {"schema": ABSOLUTE_GOAL_INPUT_SCHEMA},
    }


def test_checkpoint_task_reconstruction_rejects_incomplete_or_drifted_metadata():
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA

    contract = _contract()
    task, guides, stability = evaluator._validate_rsl_task_contract(contract)
    assert task.input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert task.goto_reward_profile == "fast_stop_v1"
    assert task.max_time == 60.0
    assert task.plants == 0 and task.terrain_enabled is False and task.rock_density == 0.0
    assert guides.rotation_coupling == "same_side_track_speed"
    assert stability.tilt_weight == 0.0 and stability.angular_rate_weight == 0.0

    for missing in ("task_config", "front_guide_spec", "body_stability_config"):
        broken = dict(contract)
        del broken[missing]
        with pytest.raises(ValueError, match="缺少可复现环境字段"):
            evaluator._validate_rsl_task_contract(broken)

    broken_task = dict(contract)
    broken_task["task_config"] = dict(contract["task_config"], max_time=30.0)
    with pytest.raises(ValueError, match="strict fast-stop 60 s absolute-goal full-map profile"):
        evaluator._validate_rsl_task_contract(broken_task)


def test_historical_v1_contract_without_remaining_time_flag_stays_v1():
    contract = _contract()
    contract["task_config"].pop("fast_stop_failure_remaining_time_cost")

    task, _, _ = evaluator._validate_rsl_task_contract(contract)

    assert task.goto_reward_profile == "fast_stop_v1"
    assert task.fast_stop_failure_remaining_time_cost is False


def test_v2_contract_requires_and_preserves_remaining_time_flag():
    contract = _contract("fast_stop_v2", remaining=True)

    task, _, _ = evaluator._validate_rsl_task_contract(contract)

    assert task.goto_reward_profile == "fast_stop_v2"
    assert task.fast_stop_failure_remaining_time_cost is True


@pytest.mark.parametrize("profile", ["legacy", "fast_stop_v3", "", None])
def test_unsupported_task_profiles_are_rejected(profile):
    contract = _contract(profile)

    with pytest.raises(ValueError, match="unsupported fast-stop reward profile"):
        evaluator._validate_rsl_task_contract(contract)


@pytest.mark.parametrize("remaining", [False, None])
def test_v2_without_true_remaining_time_flag_is_rejected(remaining):
    contract = _contract("fast_stop_v2", remaining=remaining)

    with pytest.raises(ValueError, match="fast_stop_v2 requires.*fast_stop_failure_remaining_time_cost=True"):
        evaluator._validate_rsl_task_contract(contract)


def test_factory_uses_checkpoint_task_guide_and_stability_without_native_sim(monkeypatch):
    contract = _contract()
    captured = {}

    class FakeEngine:
        pass

    engine = FakeEngine()

    def fake_make_env(*, cfg, **kwargs):
        captured["task"] = cfg
        captured.update(kwargs)
        return engine

    adapter = object()
    monkeypatch.setitem(sys.modules, "isaac_field.env", SimpleNamespace(
        make_env=fake_make_env,
        SingleFieldAdapter=lambda value: adapter if value is engine else None,
    ))
    actual_engine, actual_adapter = evaluator._create_rsl_evaluation_adapter(contract, 87_019)

    assert actual_engine is engine and actual_adapter is adapter
    assert captured["task"].max_time == 60.0
    assert captured["mode"] == "goto"
    assert captured["num_envs"] == 1 and captured["seed"] == 87_019
    assert captured["auto_reset"] is False and captured["distance_cost"] == 0.0
    assert captured["goal_distance"] is None
    assert captured["front_guide_spec"].rotation_coupling == "same_side_track_speed"
    assert captured["body_stability_config"].tilt_weight == 0.0


@pytest.mark.parametrize("profile", ["fast_stop_v1", "fast_stop_v2"])
def test_short_native_report_never_claims_success_rate_or_competence(tmp_path, monkeypatch, profile):
    checkpoint = tmp_path / "field.pt"
    checkpoint.write_bytes(b"RSL checkpoint test fixture")
    contract = _contract(profile, remaining=True if profile == "fast_stop_v2" else None)
    closed = []
    engine = SimpleNamespace(close=lambda: closed.append(True))
    adapter = object()
    monkeypatch.setattr(evaluator, "_create_rsl_evaluation_adapter", lambda *_args: (engine, adapter))
    monkeypatch.setattr(evaluator.goal_viewer, "_verify_rsl_environment_contract", lambda *_args: None)
    monkeypatch.setattr(evaluator, "_checkpoint_sha256", lambda *_args: "fixture-sha256")
    monkeypatch.setattr(evaluator, "_assert_no_parallel_simulation", lambda: None)
    monkeypatch.setitem(sys.modules, "isaac_field.rsl_rl_policy", SimpleNamespace(
        load_rsl_checkpoint_payload=lambda *_args, **_kwargs: {"field_policy_contract": contract},
    ))

    def fake_evaluate(_adapter, model, seeds, **kwargs):
        assert model == checkpoint.resolve()
        assert seeds == [87_001, 87_002]
        assert kwargs["max_steps"] == 2
        assert kwargs["device"] == "cpu"
        return {
            "model_class": "RslPredictionModel",
            "summary": {"success_rate": 1.0},
            "rows": [
                {"seed": seed, "steps": 2, "reason": "time_limit", "success": True,
                 "raw_return": 99.0, "final_distance": 1.0, "damage": 0, "collisions": 0}
                for seed in seeds
            ],
        }

    monkeypatch.setitem(sys.modules, "isaac_field.learning", SimpleNamespace(evaluate_model=fake_evaluate))
    result = evaluator.run_rsl_smoke(checkpoint, scenario_count=2, max_steps=2)

    assert result["status"] == "smoke_completed"
    assert result["assessment"] == "not_assessed"
    assert "success_rate" not in result and "success" not in result["rows"][0]
    assert "99.0" not in str(result)
    assert result["scenario_seeds"] == [87_001, 87_002]
    assert result["checkpoint_sha256"] == "fixture-sha256"
    assert result["task_profile"] == profile
    assert closed == [True]


@pytest.mark.parametrize("scenario_count,max_steps", [(0, 1), (3, 1), (1, 0), (1, 3)])
def test_native_smoke_bounds_reject_long_or_broad_runs(tmp_path, scenario_count, max_steps):
    checkpoint = tmp_path / "field.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="scenario_count|max_steps"):
        evaluator.run_rsl_smoke(
            checkpoint, scenario_count=scenario_count, max_steps=max_steps
        )


def test_legacy_zip_cannot_enter_new_rsl_smoke_path(tmp_path):
    checkpoint = tmp_path / "old.zip"
    checkpoint.touch()
    with pytest.raises(ValueError, match="requires an RSL .pt checkpoint"):
        evaluator.run_rsl_smoke(checkpoint, max_steps=1)


def test_rsl_full_evaluation_is_not_implicitly_smoke_capped(tmp_path, monkeypatch):
    checkpoint = tmp_path / "field.pt"
    checkpoint.touch()
    captured = {}

    def fake_evaluate(checkpoint, **kwargs):
        captured.update(checkpoint=checkpoint, **kwargs)
        return {"backend": "rsl_rl", "rows": []}, {}, Path(checkpoint), []

    monkeypatch.setattr(evaluator, "_evaluate_rsl_checkpoint", fake_evaluate)
    result = evaluator.run_rsl_evaluation(checkpoint)

    assert result["backend"] == "rsl_rl"
    assert captured["scenario_count"] == 10
    assert captured["max_steps"] is None


def test_checkpoint_identity_change_during_rsl_evaluation_is_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "field.pt"
    checkpoint.write_bytes(b"checkpoint")
    contract = _contract()
    engine = SimpleNamespace(close=lambda: None)
    hashes = iter(("before", "after"))
    monkeypatch.setattr(evaluator, "_checkpoint_sha256", lambda *_args: next(hashes))
    monkeypatch.setattr(evaluator, "_assert_no_parallel_simulation", lambda: None)
    monkeypatch.setattr(evaluator, "_create_rsl_evaluation_adapter", lambda *_args: (engine, object()))
    monkeypatch.setattr(evaluator.goal_viewer, "_verify_rsl_environment_contract", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "isaac_field.rsl_rl_policy", SimpleNamespace(
        load_rsl_checkpoint_payload=lambda *_args, **_kwargs: {"field_policy_contract": contract},
    ))
    monkeypatch.setitem(sys.modules, "isaac_field.learning", SimpleNamespace(
        evaluate_model=lambda *_args, **_kwargs: {"model_class": "RslPredictionModel", "rows": []},
    ))

    with pytest.raises(RuntimeError, match="checkpoint changed during evaluation"):
        evaluator.run_rsl_evaluation(checkpoint, scenario_count=1)


def test_output_collision_is_refused_before_native_environment_creation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "field.pt"
    checkpoint.touch()
    report = tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")
    called = []
    monkeypatch.setattr(evaluator, "run_rsl_smoke", lambda *_args, **_kwargs: called.append(True))

    assert evaluator.main([
        "--checkpoint", str(checkpoint), "--smoke", "--max-steps", "2",
        "--output", str(report),
    ]) == 2
    assert called == []
    assert report.read_text(encoding="utf-8") == "keep"


def test_old_trial_path_requires_explicit_legacy_switch(monkeypatch):
    monkeypatch.setattr(evaluator, "run", lambda *_args, **_kwargs: pytest.fail("legacy run must not be implicit"))
    with pytest.raises(SystemExit) as result:
        evaluator.main(["--check"])
    assert result.value.code == 2
