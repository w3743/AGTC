"""CPU-only tests for the full-episode native RSL curriculum evaluator."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import scripts.evaluate_rsl_curriculum as evaluator


@pytest.fixture(autouse=True)
def _keep_evaluator_tests_cpu_only(monkeypatch):
    from contextlib import nullcontext
    from isaac_field import runtime as isaac_runtime

    # Every test in this module uses fake checkpoint/environment/evaluation
    # components. Exercise workflow behavior without launching Isaac Sim.
    monkeypatch.setattr(isaac_runtime, "launch_app", lambda *args, **kwargs: nullcontext())


def _full_contract():
    from isaac_field.assets import FrontGuideSpec
    from isaac_field.body_stability import BodyStabilityConfig
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from isaac_field.task import Config
    from scripts.train_isaac_foundation_fresh import FAST_STOP_CONFIG

    cfg = Config(
        **FAST_STOP_CONFIG,
        size=100.0,
        max_time=60.0,
        plants=0,
        terrain_enabled=False,
        rock_density=0.0,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    guide = FrontGuideSpec(x_m=0.43)
    stability = BodyStabilityConfig(tilt_weight=2.5)
    return {
        "backend": "rsl_rl",
        "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
        "observation_config": {"schema": ABSOLUTE_GOAL_INPUT_SCHEMA, "state_dim": 33},
        "task_config": asdict(cfg),
        "front_guide_spec": asdict(guide),
        "body_stability_config": asdict(stability),
    }


def _install_fake_runtime(monkeypatch, checkpoint: Path, *, success_count=13,
                          damage_at=None, collision_at=None):
    from isaac_field import learning
    from isaac_field import runtime as isaac_runtime
    from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
    from scripts import evaluate_fast_stop_mastery, view_fast_stop_goal

    contract = _full_contract()
    # This CPU test substitutes the simulator, environment, policy loader, and
    # evaluator below. Keep the workflow lifecycle in-process and never launch Kit.
    monkeypatch.setattr(isaac_runtime, "launch_app", lambda *args, **kwargs: nullcontext())
    cfg, guide, stability = view_fast_stop_goal._rsl_environment_configs(contract)
    opened = []
    env_calls = []
    eval_calls = []
    loaded_models = []

    class FakeTask:
        def __init__(self):
            self.time = 0.0
            self.robot = (4.0, 5.0)
            self.goal = (4.1, 5.1)

        def _goal_stop_state(self):
            return {
                "speed_source": "root_com_vel_w",
                "linear_speed_m_s": 0.01,
                "angular_speed_rad_s": 0.02,
                "hold_s": 0.6,
                "stop_speed_limit_m_s": 0.05,
                "stop_angular_speed_limit_rad_s": 0.1,
                "hold_required_s": 0.5,
                "distance_tolerance_m": 0.4,
                "within_stop_limits": True,
            }

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.field_cfg = kwargs["cfg"]
            self.task = FakeTask()
            self.closed = False
            env_calls.append(kwargs)
            opened.append(self)

        def close(self):
            self.closed = True

    class FakeSingleFieldAdapter:
        def __init__(self, env):
            self.engine = env
            self.task = env.task
            self.cfg = env.field_cfg
            self.closed = False

        def reset(self, *, seed=None, options=None):
            self.task.time = 0.0
            return None, {"seed": seed}

        def step(self, action):
            self.task.time += cfg.dt
            return None, 0.0, False, False, {}

        def close(self):
            self.closed = True
            self.engine.close()

    class FakeRslPredictionModel:
        def __init__(self, path, *, device, input_schema):
            assert Path(path) == checkpoint
            assert device == "cpu"
            assert input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
            self.backend = "rsl_rl"
            self.input_schema = ABSOLUTE_GOAL_INPUT_SCHEMA
            self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            self.contract = contract
            self.num_timesteps = 40_960
            loaded_models.append(self)

    def fake_make_env(**kwargs):
        assert kwargs["cfg"] == cfg
        assert kwargs["front_guide_spec"] == guide
        assert kwargs["body_stability_config"] == stability
        return FakeEngine(**kwargs)

    def fake_evaluate_model(adapter, model_or_checkpoint, seeds, **kwargs):
        assert model_or_checkpoint is loaded_models[0]
        assert list(seeds) == list(range(87_010, 87_010 + len(seeds)))
        assert kwargs["max_steps"] is None
        assert kwargs["record_trajectories"] is True
        assert kwargs["record_initial_conditions"] is True
        assert kwargs["deterministic"] is True
        assert kwargs["device"] == "cpu"
        assert kwargs["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
        assert kwargs["expected_observation_config"] == contract["observation_config"]
        eval_calls.append(kwargs)
        rows = []
        for index, seed in enumerate(seeds):
            adapter.reset(seed=seed)
            adapter.step([0.0])
            adapter.step([0.0])
            success = index < success_count
            rows.append({
                "seed": int(seed),
                "steps": 2,
                "success": success,
                "reason": "goal_reached" if success else "timeout",
                "damage": int(index == damage_at),
                "collisions": int(index == collision_at),
                "trajectory": [
                    {"track_command_lr_mps": [1.0, 0.5]},
                    {"track_command_lr_mps": [-1.0, 0.5]},
                ],
                "reset_initial_condition": {
                    "start_pose_xy_yaw": [1.0, 2.0, 0.3],
                    "goal_xy": [4.0, 5.0],
                },
                "trajectory_start_matches_reset": True,
            })
        return {
            "model_class": "RslPredictionModel",
            "input_schema": ABSOLUTE_GOAL_INPUT_SCHEMA,
            "rows": rows,
            "summary": {"native_episode_count": len(rows), "native_success_rate": success_count / len(rows)},
            "initial_conditions": [row["reset_initial_condition"] for row in rows],
        }

    fake_inference = ModuleType("isaac_field.rsl_rl_inference")
    fake_inference.RslPredictionModel = FakeRslPredictionModel
    fake_env = ModuleType("isaac_field.env")
    fake_env.SingleFieldAdapter = FakeSingleFieldAdapter
    fake_env.make_env = fake_make_env
    monkeypatch.setitem(sys.modules, "isaac_field.rsl_rl_inference", fake_inference)
    monkeypatch.setitem(sys.modules, "isaac_field.env", fake_env)
    monkeypatch.setattr(learning, "evaluate_model", fake_evaluate_model)
    monkeypatch.setattr(view_fast_stop_goal, "_verify_rsl_environment_contract", lambda *_: None)
    monkeypatch.setattr(evaluate_fast_stop_mastery, "_assert_no_parallel_simulation", lambda: None)
    return SimpleNamespace(env_calls=env_calls, eval_calls=eval_calls, cfg=cfg,
                           guide=guide, stability=stability, loaded_models=loaded_models)


def test_run_evaluation_reconstructs_checkpoint_task_and_records_full_episode_data(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"native RSL checkpoint fixture")
    runtime = _install_fake_runtime(monkeypatch, checkpoint)
    output = tmp_path / "evaluation"

    result = evaluator.run_evaluation(checkpoint, output)

    assert result["status"] == "completed"
    assert result["assessment"] == "strictpass"
    assert len(runtime.env_calls) == 1
    call = runtime.env_calls[0]
    assert call["mode"] == "goto"
    assert call["num_envs"] == 1
    assert call["auto_reset"] is False
    assert call["goal_distance"] is None
    assert call["goal_angle"] == pytest.approx(3.141592653589793)
    assert call["cfg"].max_time == 60.0
    assert call["front_guide_spec"] == runtime.guide
    assert call["body_stability_config"] == runtime.stability
    call = runtime.eval_calls[0]
    assert call["max_steps"] is None
    assert call["goal_distance"] is None
    assert call["goal_angle"] == pytest.approx(3.141592653589793)

    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert report["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["sampling"]["scenario_seeds"] == list(range(87_010, 87_026))
    assert report["sampling"]["max_steps_per_episode"] == 1200
    assert report["sampling"]["max_steps_cap_applied"] is False
    assert report["sampling"]["goal_sampling"] == "global"
    assert report["native_metrics"] == {"native_episode_count": 16, "native_success_rate": 13 / 16}
    assert "strictpass" not in report["native_metrics"]
    assert report["assessment"] == "strictpass"
    assert report["summary"]["clean_successes"] == 13
    assert report["summary"]["damage_total"] == 0
    assert report["summary"]["collision_total"] == 0
    assert report["rows"][0]["terminal_stop_state"]["speed_source"] == "root_com_vel_w"
    assert report["rows"][0]["terminal_stop_safe"] is True
    assert report["rows"][0]["track_direction_reversals"]["left_track"] == 1
    assert len(report["rows"][0]["goal_stop_trace"]) == 2
    assert report["initial_conditions"][0]["reset_initial_condition"]["start_pose_xy_yaw"] == [1.0, 2.0, 0.3]
    assert report["initial_conditions"][0]["seed"] == 87_010
    assert report["initial_conditions"][0]["trajectory_start_matches_reset"] is True
    assert manifest["status"] == "completed"
    assert manifest["record_trajectories"] is True
    assert manifest["record_initial_conditions"] is True
    assert "830xx" in report["claim_limit"]


@pytest.mark.parametrize(
    "success_count,damage_at,collision_at,expected",
    (
        (13, None, None, "strictpass"),
        (12, None, None, "strictfail"),
        (13, 15, None, "strictfail"),
        (13, None, 15, "strictfail"),
    ),
)
def test_strictpass_requires_thirteen_clean_episodes_and_zero_damage_and_collisions(
    tmp_path, monkeypatch, success_count, damage_at, collision_at, expected
):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"RSL checkpoint")
    _install_fake_runtime(monkeypatch, checkpoint, success_count=success_count,
                          damage_at=damage_at, collision_at=collision_at)

    result = evaluator.run_evaluation(checkpoint, tmp_path / "out")

    assert result["assessment"] == expected
    report = json.loads((tmp_path / "out" / "evaluation.json").read_text(encoding="utf-8"))
    assert report["summary"]["strictpass"] is (expected == "strictpass")


def test_non_sixteen_scenario_run_is_unassessed_and_distance_range_is_recorded(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"RSL checkpoint")
    _install_fake_runtime(monkeypatch, checkpoint, success_count=3)

    result = evaluator.run_evaluation(
        checkpoint, tmp_path / "out", scenario_count=3, goal_distance=(1.0, 8.0),
        goal_angle_deg=90.0,
    )

    assert result["assessment"] == "not_assessed"
    report = json.loads((tmp_path / "out" / "evaluation.json").read_text(encoding="utf-8"))
    assert report["sampling"]["goal_distance_m"] == [1.0, 8.0]
    assert report["sampling"]["goal_sampling"] == "uniform_distance_range"
    assert report["sampling"]["goal_angle_deg"] == 90.0
    assert report["summary"]["strictpass_eligible"] is False


def test_goal_success_without_measured_distance_and_stop_limits_is_unsafe():
    evaluated = {
        "rows": [{
            "seed": 7,
            "success": True,
            "reason": "goal_reached",
            "damage": 0,
            "collisions": 0,
            "trajectory": [{"track_command_lr_mps": [0.2, 0.2]}],
        }],
        "summary": {},
    }
    trace = [[{
        "time_s": 60.0,
        "distance_m": 0.8,
        "distance_tolerance_m": 0.4,
        "linear_speed_xy_norm_m_s": 0.01,
        "angular_speed_xyz_norm_rad_s": 0.02,
        "hold_s": 0.6,
        "linear_speed_limit_m_s": 0.05,
        "angular_speed_limit_rad_s": 0.1,
        "hold_required_s": 0.5,
        "within_stop_limits": True,
        "speed_source": "root_com_vel_w",
    }]]

    report = evaluator._finalize_episode_rows(evaluated, trace)

    assert report["rows"][0]["terminal_stop_safe"] is False
    assert report["rows"][0]["unsafe"] is True
    assert report["rows"][0]["unsafe_reasons"] == ["goal_reached_without_measured_stop"]


@pytest.mark.parametrize(
    "hold_s,expected_safe",
    ((0.49999999999999994, True), (0.499999998, False)),
)
def test_terminal_hold_tolerates_only_float_accumulation_noise(hold_s, expected_safe):
    evaluated = {
        "rows": [{
            "seed": 7,
            "success": True,
            "reason": "goal_reached",
            "damage": 0,
            "collisions": 0,
            "trajectory": [{"track_command_lr_mps": [0.2, 0.2]}],
        }],
        "summary": {},
    }
    stop_sample = {
        "time_s": 60.0,
        "distance_m": 0.1,
        "distance_tolerance_m": 0.4,
        "linear_speed_xy_norm_m_s": 0.01,
        "angular_speed_xyz_norm_rad_s": 0.02,
        "hold_s": hold_s,
        "linear_speed_limit_m_s": 0.05,
        "angular_speed_limit_rad_s": 0.1,
        "hold_required_s": 0.5,
        "within_stop_limits": True,
        "speed_source": "root_com_vel_w",
    }

    report = evaluator._finalize_episode_rows(evaluated, [[stop_sample]])

    assert report["rows"][0]["terminal_stop_safe"] is expected_safe
    assert report["rows"][0]["clean_success"] is expected_safe


def test_existing_output_is_refused_without_modification(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"RSL checkpoint")
    output = tmp_path / "already-exists"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.run_evaluation(checkpoint, output)

    assert marker.read_text(encoding="utf-8") == "original"


def test_stop_trace_requires_real_root_com_and_full_angular_kinematics():
    class BaseAdapter:
        def __init__(self, env):
            self.task = env.task

        def reset(self, *, seed=None, options=None):
            self.task.time = 0.0
            return None, {}

        def step(self, action):
            self.task.time += 0.05
            return None, 0.0, False, False, {}

    class Task:
        time = 0.0
        robot = (0.0, 0.0)
        goal = (1.0, 0.0)

        def _goal_stop_state(self):
            return {
                "speed_source": "root_com_vel_w",
                "linear_speed_m_s": 0.0,
                "angular_speed_rad_s": 0.0,
                "hold_s": 0.5,
                "stop_speed_limit_m_s": 0.05,
                "stop_angular_speed_limit_rad_s": 0.1,
                "hold_required_s": 0.5,
                "distance_tolerance_m": 0.4,
                "within_stop_limits": True,
            }

    adapter = evaluator._goal_stop_trace_adapter(BaseAdapter)(SimpleNamespace(task=Task()))
    adapter.reset(seed=1)
    adapter.step([0.0])
    assert adapter.finish_stop_traces()[0][0]["speed_source"] == "root_com_vel_w"

    Task._goal_stop_state = lambda self: {
        "speed_source": "planar_estimate", "linear_speed_m_s": 0.0,
        "angular_speed_rad_s": 0.0, "hold_s": 0.0,
        "stop_speed_limit_m_s": 0.05, "stop_angular_speed_limit_rad_s": 0.1,
        "hold_required_s": 0.5, "distance_tolerance_m": 0.4,
        "within_stop_limits": False,
    }
    adapter = evaluator._goal_stop_trace_adapter(BaseAdapter)(SimpleNamespace(task=Task()))
    adapter.reset(seed=1)
    with pytest.raises(ValueError, match="per-step root COM"):
        adapter.step([0.0])


def test_distance_sampling_validation_rejects_invalid_ranges():
    assert evaluator._validate_sampling(1, 1, None, 180.0, 100.0)[0] is None
    with pytest.raises(ValueError, match="MIN/MAX"):
        evaluator._validate_sampling(1, 1, (2.0, 1.0), 180.0, 100.0)
    with pytest.raises(ValueError, match="field diagonal"):
        evaluator._validate_sampling(1, 1, (0.0, 200.0), 180.0, 100.0)
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        evaluator._validate_sampling(1, 1, None, 181.0, 100.0)


@pytest.mark.parametrize(
    "seed_start,scenario_count,rejected",
    (
        (82_999, 1, False),
        (82_999, 2, True),
        (83_000, 1, True),
        (83_998, 2, True),
        (83_999, 1, True),
        (84_000, 1, False),
    ),
)
def test_development_seed_bank_cannot_intersect_reserved_83000_range(
    seed_start, scenario_count, rejected
):
    if rejected:
        with pytest.raises(ValueError, match="reserved curriculum seed bank"):
            evaluator._validate_sampling(seed_start, scenario_count, None, 180.0, 100.0)
    else:
        distance, _ = evaluator._validate_sampling(
            seed_start, scenario_count, None, 180.0, 100.0
        )
        assert distance is None


@pytest.mark.parametrize("missing_key", ("damage", "collisions"))
def test_episode_missing_native_safety_metric_is_rejected(missing_key):
    row = {
        "seed": 7,
        "success": True,
        "reason": "goal_reached",
        "damage": 0,
        "collisions": 0,
        "trajectory": [{"track_command_lr_mps": [0.2, 0.2]}],
    }
    del row[missing_key]
    evaluated = {"rows": [row], "summary": {}}
    stop_sample = {
        "time_s": 60.0,
        "distance_m": 0.1,
        "distance_tolerance_m": 0.4,
        "linear_speed_xy_norm_m_s": 0.01,
        "angular_speed_xyz_norm_rad_s": 0.02,
        "hold_s": 0.6,
        "linear_speed_limit_m_s": 0.05,
        "angular_speed_limit_rad_s": 0.1,
        "hold_required_s": 0.5,
        "within_stop_limits": True,
        "speed_source": "root_com_vel_w",
    }

    with pytest.raises(ValueError, match="missing native episode safety metrics"):
        evaluator._finalize_episode_rows(evaluated, [[stop_sample]])


def test_cli_defaults_to_full_development_sample_and_global_goal_sampling(monkeypatch, capsys):
    captured = {}

    def fake_run(checkpoint, output, **kwargs):
        captured.update(checkpoint=checkpoint, output=output, **kwargs)
        return {"status": "completed", "assessment": "not_assessed"}

    monkeypatch.setattr(evaluator, "run_evaluation", fake_run)
    assert evaluator.main(["--checkpoint", "model.pt", "--output", "new-output"]) == 0

    assert captured["seed_start"] == 87_010
    assert captured["scenario_count"] == 16
    assert captured["goal_distance"] is None
    assert captured["goal_angle_deg"] == 180.0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_contract_reconstruction_rejects_missing_complete_task_metadata():
    from scripts import view_fast_stop_goal

    with pytest.raises(ValueError, match="task_config|可复现环境字段"):
        view_fast_stop_goal._rsl_environment_configs({
            "backend": "rsl_rl",
            "input_schema": "absolute_goal_v1",
            "task_config": {},
        })
