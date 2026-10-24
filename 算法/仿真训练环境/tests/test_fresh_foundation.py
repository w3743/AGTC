"""CPU-only protocol and pairing checks for the fresh foundation runner."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest
import numpy as np

import scripts.train_isaac_foundation_fresh as runner
from isaac_field.task import Config, Scene, sample_episode_start_goal, wrap

_REAL_ROBOT_GEOMETRY = runner._robot_geometry


FAKE_SOURCES = (
    "isaac_field/policy.py",
    "isaac_field/learning.py",
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
    "lstm_hidden_size": 512,
    "lstm_num_layers": 1,
    "architecture_is_searched": False,
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
    monkeypatch.setattr(
        runner, "_experiment_architecture",
        lambda lstm_hidden_size=512: {
            **copy.deepcopy(FAKE_ARCHITECTURE), "lstm_hidden_size": lstm_hidden_size,
        },
    )
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
    assert protocol["initialization"]["baseline_path"] == "untrained_baseline.pt"
    assert protocol["profile"] == runner.PROFILE_LEGACY_FOUNDATION
    assert protocol["training"]["timesteps_total"] == 196_608
    assert protocol["training"]["stage_seeds"] == [69_001, 69_002, 69_003, 69_004]
    assert protocol["action_config"] == runner._action_config()
    assert protocol["runtime_contract"]["action_config"] == runner._action_config()
    assert protocol["architecture"]["action_config"] == runner._action_config()
    assert protocol["architecture"]["input_contract"] == "sensor-minimal-v6"
    assert protocol["architecture"]["lstm_hidden_size"] == 512
    assert protocol["architecture"]["lstm_num_layers"] == 1
    assert protocol["evaluation"]["blind_seeds"] == {
        "range_label": "830xx", "used": False, "observed": False,
    }
    assert status["protocol_sha256"] == result["protocol_sha256"]
    assert runner.preflight(output)["schema"] == runner.SCHEMA


def test_prepare_freezes_512_lstm_width_and_preflight_accepts_it(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh-512"
    runner.prepare(output, profile=runner.PROFILE_RANDOM_ROUGH, lstm_hidden_size=512)

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["architecture"]["lstm_hidden_size"] == 512
    assert protocol["architecture"]["lstm_num_layers"] == 1
    assert runner.preflight(output)["architecture"]["lstm_hidden_size"] == 512


@pytest.mark.parametrize("value", [0, -1, True, 512.0, "512"])
def test_prepare_rejects_non_positive_or_non_integer_lstm_width(trial_root: Path, value) -> None:
    output = trial_root / "results" / "invalid-width"
    with pytest.raises(ValueError, match="lstm_hidden_size must be a positive integer"):
        runner.prepare(output, lstm_hidden_size=value)
    assert not output.exists()


def test_preflight_rejects_tampered_frozen_lstm_architecture(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh-512"
    runner.prepare(output, profile=runner.PROFILE_RANDOM_ROUGH, lstm_hidden_size=512)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["architecture"]["lstm_num_layers"] = 2
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    status_path = output / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["protocol_sha256"] = runner._sha256(protocol_path)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="prepared encoder"):
        runner.preflight(output)


def test_main_rejects_lstm_width_outside_prepare() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--launch", "--lstm-hidden-size", "512"])


def test_random_rough_profile_freezes_global_goals_terrain_and_physics_contract(trial_root: Path) -> None:
    output = trial_root / "results" / "random_rough"
    result = runner.prepare(output, profile=runner.PROFILE_RANDOM_ROUGH)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))

    assert protocol["profile"] == runner.PROFILE_RANDOM_ROUGH
    assert result["timesteps_total"] == 196_608
    assert protocol["initialization"] == {
        "checkpoint": None, "baseline_path": "untrained_baseline.pt",
        "baseline_num_timesteps": 0, "weight_transfer": False,
    }
    assert protocol["config"]["plants"] == 0
    assert protocol["config"]["terrain_enabled"] is True
    assert protocol["config"]["terrain_amplitude"] == 0.35
    assert protocol["config"]["terrain_wavelength_min"] == 4.0
    assert protocol["config"]["terrain_wavelength_max"] == 10.0
    assert protocol["config"]["rock_density"] == 0.0
    assert protocol["config"]["max_time"] == 60.0
    assert protocol["training"]["training_seed"] == 69009
    assert protocol["training"]["stage_seeds"] == [69009]
    assert protocol["training"]["stages"] == [{
        "name": "global_random", "goal_distance_m": None,
        "goal_angle_deg": 180.0, "timesteps": 196_608,
    }]
    assert set(protocol["evaluation"]) >= {"rough_a", "rough_b"}
    assert not ({"free_space", "field"} & set(protocol["evaluation"]))
    for split, seeds, environment_seed in (
        ("rough_a", [87001, 87002, 87003, 87004], 80003),
        ("rough_b", [82001, 82002, 82003, 82004], 80004),
    ):
        evaluation = protocol["evaluation"][split]
        assert evaluation["plants"] == 0
        assert evaluation["goal_distance_m"] is None
        assert evaluation["goal_angle_deg"] == 180.0
        assert evaluation["seeds"] == seeds
        assert evaluation["environment_seed"] == environment_seed
    physics = protocol["profile_physics_contract"]
    assert physics["front_guide_spec"] == {
        "x_m": 0.42, "z_offset_from_support_m": 0.05, "radius_m": 0.045,
        "width_m": 0.06, "mass_per_wheel_kg": 0.25,
        "rotation_coupling": "same_side_track_speed",
    }
    assert physics["body_stability_config"]["excess_tilt_threshold_rad"] == pytest.approx(0.6108652381980153)
    assert physics["body_stability_config"]["tilt_weight"] == 2.0
    assert physics["body_stability_config"]["angular_rate_weight"] == 0.02
    assert len(physics["front_guide_wheels"]) == 2
    assert physics["front_guide_drive_coupling"] == {
        "enabled": True,
        "relation": "q_guide = (R_main/R_guide)*q_same_side_wheel0",
        "relations_per_world": 2,
        "leader_joint_by_side": {
            "left": "left_wheel_0_wheel_joint",
            "right": "right_wheel_0_wheel_joint",
        },
        "ratio_by_side": {"left": 1.0, "right": 1.0},
        "guide_actuator_count": 0,
        "added_pid_count": 0,
    }
    assert physics["same_side_wheel_coupling_counts"] == {
        "relations_per_world": 14, "wheel_relations_per_world": 12,
        "guide_relations_per_world": 2,
    }
    assert "not a continuous track" in physics["interpretation"]["front_guides"]
    assert len(physics["contract_sha256"]) == 64
    assert runner.preflight(output)["profile"] == runner.PROFILE_RANDOM_ROUGH


def test_fast_stop_profile_freezes_objective_gamma_zero_stability_and_512_width(trial_root: Path) -> None:
    output = trial_root / "results" / "fast-stop"
    runner.prepare(output, profile=runner.PROFILE_RANDOM_ROUGH_FAST_STOP, lstm_hidden_size=512)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))

    assert protocol["profile"] == runner.PROFILE_RANDOM_ROUGH_FAST_STOP
    assert protocol["architecture"]["lstm_hidden_size"] == 512
    assert protocol["initialization"]["checkpoint"] is None
    assert protocol["initialization"]["weight_transfer"] is False
    assert {key: protocol["config"][key] for key in runner.FAST_STOP_CONFIG} == runner.FAST_STOP_CONFIG
    assert protocol["config"]["plants"] == 0
    assert protocol["config"]["terrain_enabled"] is True
    assert protocol["config"]["terrain_amplitude"] == 0.35
    assert protocol["config"]["terrain_wavelength_min"] == 4.0
    assert protocol["config"]["terrain_wavelength_max"] == 10.0
    assert protocol["config"]["rock_density"] == 0.0
    assert protocol["config"]["max_time"] == 60.0
    assert protocol["training"]["timesteps_total"] == 196_608
    assert protocol["training"]["training_seed"] == 69_009
    assert protocol["training"]["gamma"] == 1.0
    assert protocol["training"]["goal_objective_contract"] == protocol["goal_objective_contract"]
    assert protocol["evaluation"]["goal_objective_contract"] == protocol["goal_objective_contract"]
    assert protocol["goal_objective_contract"]["reward"]["success"] == 10.0
    assert protocol["goal_objective_contract"]["reward"]["failure"] == -10.0
    assert protocol["goal_objective_contract"]["termination"]["timeout_is_terminal"] is True
    assert protocol["goal_objective_contract"]["discount_gamma"] == 1.0
    assert protocol["training"]["diagnostics"]["enabled"] is True
    stability = protocol["profile_physics_contract"]["body_stability_config"]
    assert stability["tilt_weight"] == 0.0
    assert stability["angular_rate_weight"] == 0.0
    assert runner.preflight(output)["profile"] == runner.PROFILE_RANDOM_ROUGH_FAST_STOP


def test_fast_stop_profile_uses_new_isaacsim_physx_output_namespace() -> None:
    assert runner._default_output_for_profile(runner.PROFILE_RANDOM_ROUGH_FAST_STOP).name == (
        "isaacsim_physx_rsl_rough_fast_stop_20261024"
    )


def test_cli_prepare_honors_explicit_output(
    trial_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = trial_root / "user-selected-output"
    monkeypatch.setattr(runner, "DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT", trial_root / "unused-default")
    assert runner.main(["--prepare", "--output", str(output)]) == 0
    capsys.readouterr()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["profile"] == runner.PROFILE_RANDOM_ROUGH


def test_cli_prepare_defaults_to_coupled_rough_output(
    trial_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    default_output = trial_root / "random_global_rough_coupled_guides"
    monkeypatch.setattr(runner, "DEFAULT_RANDOM_ROUGH_COUPLED_GUIDES_OUTPUT", default_output)
    assert runner.DEFAULT_RANDOM_ROUGH_OUTPUT.name == "isaacsim_physx_rsl_rough_20261024"
    assert runner.main(["--prepare"]) == 0
    capsys.readouterr()
    protocol = json.loads((default_output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["profile"] == runner.PROFILE_RANDOM_ROUGH


def test_preflight_accepts_real_robot_spec_after_protocol_json_roundtrip(
    trial_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = trial_root / "results" / "json-roundtrip"
    runner.prepare(output, profile=runner.PROFILE_RANDOM_ROUGH)
    geometry = _REAL_ROBOT_GEOMETRY()
    serialized_geometry = json.loads(json.dumps(geometry, ensure_ascii=False))
    assert serialized_geometry != geometry

    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["robot_geometry"] = serialized_geometry
    protocol["profile_physics_contract"] = runner._profile_physics_contract(
        runner.PROFILE_RANDOM_ROUGH,
        runner._profile_config(runner.PROFILE_RANDOM_ROUGH),
        geometry,
    )
    runner._json(protocol_path, protocol)
    status_path = output / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["protocol_sha256"] = runner._protocol_sha(protocol_path)
    runner._json(status_path, status)

    monkeypatch.setattr(runner, "_robot_geometry", lambda: geometry)
    assert runner.preflight(output)["profile"] == runner.PROFILE_RANDOM_ROUGH


def test_global_goal_sampling_can_reach_far_behind_targets_with_connectivity_validation():
    config = Config(size=24.0, cell=0.5, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(config, np.random.default_rng(3), None, np.array([22.0, 22.0]))
    observed_far_behind = False
    for seed in range(24):
        start, goal = sample_episode_start_goal(
            scene, np.random.default_rng(seed), random_start=True, mode="goto",
            goal_distance=None, goal_angle=math.pi,
        )
        displacement = goal - start[:2]
        relative_angle = wrap(math.atan2(displacement[1], displacement[0]) - start[2])
        assert scene.navigable(start[:2])
        assert scene.navigable(goal)
        assert scene.validate_connectivity(start[:2], goal)["connected"] is True
        if np.linalg.norm(displacement) > 3.0 and abs(relative_angle) > math.pi / 2:
            observed_far_behind = True
            break
    assert observed_far_behind


def test_rough_evaluation_summarizes_posture_and_separates_boundary_from_overtilt():
    report = {"rows": [
        {"reason": "outside_or_overturned", "body_stability": _body_stability_summary(
            max_abs_roll_rad=0.4, max_abs_pitch_rad=0.3, max_tilt_rad=0.5,
            roll_pitch_angular_rate_rms_rad_s=0.2,
        ), "termination_detail": {"boundary": True, "overtilt_roll": False,
                                  "overtilt_pitch": False}},
        {"reason": "outside_or_overturned", "body_stability": _body_stability_summary(
            max_abs_roll_rad=0.7, max_abs_pitch_rad=0.2, max_tilt_rad=0.75,
            roll_pitch_angular_rate_rms_rad_s=0.6,
        ), "termination_detail": {"boundary": False, "overtilt_roll": True,
                                  "overtilt_pitch": False}},
    ], "summary": {}}
    result = runner._add_rough_evaluation_summaries(report)
    assert result["summary"]["body_stability"]["metrics"]["max_tilt_rad"] == {
        "max": 0.75, "mean": 0.625,
    }
    assert result["summary"]["body_stability"]["metrics"]["roll_pitch_angular_rate_rms_rad_s"] == {
        "max": 0.6, "mean": 0.4,
    }
    assert result["summary"]["termination_events"] == {
        "boundary": 1, "overtilt_roll": 1, "overtilt_pitch": 0,
        "posture_failure_episodes": 1, "outside_or_overturned_reason": 2,
    }
    gate = runner._rough_termination_safety(result, result)
    assert gate["not_increased"] is True
    regressed = copy.deepcopy(result)
    regressed["summary"]["termination_events"]["posture_failure_episodes"] += 1
    assert runner._rough_termination_safety(result, regressed)["not_increased"] is False


def _body_stability_summary(**updates) -> dict:
    result = {
        "sample_count": 12, "duration_s": 0.6,
        "max_abs_roll_rad": 0.4, "max_abs_pitch_rad": 0.3, "max_tilt_rad": 0.5,
        "excess_tilt_duration_s": 0.0, "roll_pitch_rate_sq_integral": 0.024,
        "penalty_sum": 0.001, "roll_pitch_angular_rate_rms_rad_s": 0.2,
        "sampling": "policy-step samples; not physics-substep extrema",
    }
    result.update(updates)
    return result


@pytest.mark.parametrize("field,value", [
    ("penalty_sum", float("nan")), ("roll_pitch_rate_sq_integral", float("inf")),
])
def test_rough_evaluation_rejects_non_finite_unaggregated_posture_summary_fields(
    field: str, value: float,
) -> None:
    report = {"rows": [{
        "reason": "time_limit",
        "body_stability": _body_stability_summary(**{field: value}),
        "termination_detail": {"boundary": False, "overtilt_roll": False,
                               "overtilt_pitch": False},
    }], "summary": {}}

    with pytest.raises(ValueError, match=field):
        runner._add_rough_evaluation_summaries(report)


def test_rough_evaluation_rejects_missing_numeric_posture_summary_field() -> None:
    summary = _body_stability_summary()
    del summary["penalty_sum"]
    report = {"rows": [{
        "reason": "time_limit", "body_stability": summary,
        "termination_detail": {"boundary": False, "overtilt_roll": False,
                               "overtilt_pitch": False},
    }], "summary": {}}

    with pytest.raises(ValueError, match="penalty_sum"):
        runner._add_rough_evaluation_summaries(report)


@pytest.mark.parametrize(
    "distance,environment_seed,expected_distance,expected_seed",
    [(None, 80003, None, 80003), ("default", None, (1.0, 3.0), runner.ENV_SEED)],
)
def test_evaluate_forwards_global_goal_and_environment_seed_without_changing_defaults(
    trial_root: Path, monkeypatch: pytest.MonkeyPatch, distance, environment_seed, expected_distance, expected_seed,
) -> None:
    class Adapter:
        def __init__(self, engine):
            self.engine = engine

        def close(self):
            pass

    calls = {"make_env": None, "evaluate": None}
    env_module = ModuleType("isaac_field.env")
    env_module.SingleFieldAdapter = Adapter
    env_module.make_env = lambda **kwargs: calls.update(make_env=kwargs) or object()
    learning_module = ModuleType("isaac_field.learning")
    learning_module.evaluate_model = lambda *args, **kwargs: calls.update(evaluate=kwargs) or {"rows": [], "summary": {}}
    policy_module = ModuleType("scripts.improve_isaac_policy")
    policy_module._add_safety_summary = lambda report: report
    monkeypatch.setitem(sys.modules, "isaac_field.env", env_module)
    monkeypatch.setitem(sys.modules, "isaac_field.learning", learning_module)
    monkeypatch.setitem(sys.modules, "scripts.improve_isaac_policy", policy_module)
    monkeypatch.setattr(runner, "_live_environment_contract", lambda _: {})
    monkeypatch.setattr(runner, "_runtime_assets", lambda _: [])

    @dataclass
    class ConfigForEval:
        plants: int = 1

    cfg = ConfigForEval()
    kwargs = {} if distance == "default" else {"goal_distance": distance, "environment_seed": environment_seed}
    report = runner._evaluate(
        trial_root, cfg, trial_root / "checkpoint.pt", "test", (87001,),
        0, 180.0, "cpu", **kwargs,
    )

    assert calls["make_env"]["seed"] == expected_seed
    assert calls["make_env"]["goal_distance"] == expected_distance
    assert calls["evaluate"]["goal_distance"] == expected_distance
    assert report["evaluation_config"] == {
        "environment_seed": expected_seed, "plants": 0,
        "goal_distance_m": None if expected_distance is None else list(expected_distance),
        "goal_angle_deg": 180.0,
    }


def test_rough_evaluation_records_actual_terrain_config(
    trial_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        def __init__(self, engine):
            self.engine = engine

        def close(self):
            pass

    env_module = ModuleType("isaac_field.env")
    env_module.SingleFieldAdapter = Adapter
    env_module.make_env = lambda **kwargs: object()
    learning_module = ModuleType("isaac_field.learning")
    learning_module.evaluate_model = lambda *args, **kwargs: {"rows": [], "summary": {}}
    policy_module = ModuleType("scripts.improve_isaac_policy")
    policy_module._add_safety_summary = lambda report: report
    monkeypatch.setitem(sys.modules, "isaac_field.env", env_module)
    monkeypatch.setitem(sys.modules, "isaac_field.learning", learning_module)
    monkeypatch.setitem(sys.modules, "scripts.improve_isaac_policy", policy_module)
    monkeypatch.setattr(runner, "_validate_rough_profile_environment", lambda *_: {})
    monkeypatch.setattr(runner, "_runtime_assets", lambda _: [
        {"asset": "field", "path": "/tmp/terrain.usda", "sha256": "a" * 64, "size_bytes": 123},
    ])
    monkeypatch.setattr(runner, "_add_rough_evaluation_summaries", lambda report: report)

    @dataclass
    class ConfigForEval:
        plants: int = 16
        terrain_enabled: bool = True
        terrain_amplitude: float = 0.35
        terrain_wavelength_min: float = 4.0
        terrain_wavelength_max: float = 10.0
        rock_density: float = 0.0
        max_time: float = 60.0

    terrain_config = dict(runner.RANDOM_ROUGH_TERRAIN_CONFIG)
    report = runner._evaluate(
        trial_root, ConfigForEval(), trial_root / "checkpoint.pt", "rough", (87_001,),
        0, 180.0, "cpu", goal_distance=None, environment_seed=80_003,
        profile_physics_contract={"terrain_config": terrain_config},
    )

    assert report["evaluation_config"] == {
        "environment_seed": 80_003, "plants": 0, "goal_distance_m": None,
        "goal_angle_deg": 180.0, "goal_sampling_profile": runner.PROFILE_RANDOM_ROUGH,
        "terrain_config": terrain_config,
    }


def test_preflight_rejects_live_source_drift(trial_root: Path) -> None:
    output = trial_root / "results" / "fresh"
    runner.prepare(output)
    (trial_root / FAKE_SOURCES[0]).write_text("changed policy\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source snapshot/live hash mismatch"):
        runner.preflight(output)


def test_live_environment_contract_requires_coupled_pi_v7_and_physx_contract():
    current_drive = {
        "interface": "independent_left_right_track_speed_coupled_pi_v7",
        "same_side_wheel_coupling": {"enabled": True},
        "physics_contact": {
            "simulator": "isaac_sim",
            "physics_backend": "physx",
            "learning_backend": "rsl_rl",
            "backend": "PhysxCfg",
            "solver": "TGS",
            "physics_dt_s": 0.001,
            "decimation": 50,
            "policy_dt_s": 0.05,
            "physics_manager": "isaaclab_physx.physics_manager.PhysxManager",
            "usd_authored_materials": {
                "/World/envs/env_0/Field/material": {
                    "static_friction": 0.7, "dynamic_friction": 0.6, "restitution": 0.0,
                },
                "/World/envs/env_0/Robot/material": {
                    "static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0,
                },
            },
            "contact_offset_m": 0.005,
            "rest_offset_m": 0.0,
            "calibrated_hardware": False,
        },
    }
    env = SimpleNamespace(
        backend_identity={"simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl"},
        num_envs=1,
        action_config=runner._action_config(),
        drive_config=copy.deepcopy(current_drive),
        observation_config={"schema": "sensor-minimal-v6"},
    )

    contract = runner._live_environment_contract(env)
    assert contract["action_config"] == runner._action_config()
    assert contract["drive_config"]["interface"] == "independent_left_right_track_speed_coupled_pi_v7"
    assert contract["observation_config"]["schema"] == "sensor-minimal-v6"
    assert contract["backend_identity"] == env.backend_identity
    assert contract["physics_contact"]["solver"] == "TGS"
    for old_interface in (
        "independent_left_right_track_speed_explicit_pi_v6",
        "independent_left_right_track_speed_explicit_v5",
        "independent_left_right_track_speed_explicit_v4",
        "forward_and_differential_track_speed_v2",
    ):
        env.drive_config = {"interface": old_interface}
        with pytest.raises(ValueError, match="unexpected live drive interface"):
            runner._live_environment_contract(env)

    env.drive_config = copy.deepcopy(current_drive)
    env.drive_config["same_side_wheel_coupling"]["enabled"] = False
    with pytest.raises(ValueError, match="unexpected live same-side wheel coupling"):
        runner._live_environment_contract(env)

    for key, wrong in (
        ("physics_dt_s", 0.002),
        ("decimation", 25),
        ("policy_dt_s", 0.04),
        ("backend", "NewtonCfg"),
        ("solver", "PGS"),
        ("physics_manager", "isaaclab_newton.physics_manager.NewtonManager"),
        ("contact_offset_m", 0.01),
    ):
        env.drive_config = copy.deepcopy(current_drive)
        env.drive_config["physics_contact"][key] = wrong
        with pytest.raises(ValueError, match="unexpected live"):
            runner._live_environment_contract(env)

    env.drive_config = copy.deepcopy(current_drive)
    env.drive_config["physics_contact"]["usd_authored_materials"] = {}
    with pytest.raises(ValueError, match="USD contact materials"):
        runner._live_environment_contract(env)

    env.drive_config = copy.deepcopy(current_drive)
    env.backend_identity = {"simulator": "isaac_sim", "physics_backend": "newton", "learning_backend": "rsl_rl"}
    with pytest.raises(ValueError, match="backend_identity"):
        runner._live_environment_contract(env)


@pytest.mark.parametrize("profile", [runner.PROFILE_RANDOM_ROUGH, runner.PROFILE_RANDOM_ROUGH_FAST_STOP])
def test_rough_profile_live_contract_requires_same_side_guide_coupling(
    trial_root: Path, profile: str,
) -> None:
    front_spec, body_stability = runner._rough_profile_components(profile)
    physics = runner._profile_physics_contract(
        profile, runner._profile_config(profile),
        runner._robot_geometry(),
    )
    guide_coupling = physics["front_guide_drive_coupling"]
    drive = {
        "interface": "independent_left_right_track_speed_coupled_pi_v7",
        "robot_asset_spec_sha256": physics["robot_asset_spec_sha256"],
        "same_side_wheel_coupling": {
            "enabled": True,
            "relations_per_world": 14,
            "wheel_relations_per_world": 12,
            "guide_relations_per_world": 2,
            "front_guide_relation": guide_coupling["relation"],
        },
        "front_guide": {
            "enabled": True,
            "model": runner.COUPLED_FRONT_GUIDE_MODEL,
            "wheels": physics["front_guide_wheels"],
            "drive_motor_count": 0,
            "added_drive_torque_nm": 0.0,
            "rotation_coupling": "same_side_track_speed",
            "coupling": guide_coupling,
        },
        "body_stability": {
            "enabled": True,
            "config": asdict(body_stability),
            "rate_source": "root_link_com_angular_velocity_world_rotated_to_body",
            "policy_observation_included": False,
            "hard_tilt_termination_rad": 1.2,
        },
        "physics_contact": {
            "simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl",
            "backend": "PhysxCfg", "solver": "TGS",
            "physics_manager": "isaaclab_physx.physics_manager.PhysxManager",
            "physics_dt_s": 0.001, "decimation": 50, "policy_dt_s": 0.05,
            "contact_offset_m": 0.005, "rest_offset_m": 0.0, "calibrated_hardware": False,
            "usd_authored_materials": {
                "/World/envs/env_0/Field/material": {
                    "static_friction": 0.7, "dynamic_friction": 0.6, "restitution": 0.0,
                },
                "/World/envs/env_0/Robot/material": {
                    "static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0,
                },
            },
        },
    }
    env = SimpleNamespace(
        backend_identity={"simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl"},
        num_envs=1,
        action_config=runner._action_config(), drive_config=drive,
        observation_config={"schema": runner.INPUT_SCHEMA},
        front_guide_spec=front_spec, body_stability_config=body_stability,
    )
    assert runner._validate_rough_profile_environment(env, physics)["drive_config"] == drive
    if profile == runner.PROFILE_RANDOM_ROUGH_FAST_STOP:
        assert body_stability.tilt_weight == 0.0
        assert body_stability.angular_rate_weight == 0.0

    invalid_cases = (
        ("model", lambda d: d["front_guide"].update(model="passive_free_rotating_guide_wheels_separate_proxy_not_continuous_track")),
        ("rotation coupling", lambda d: d["front_guide"].update(rotation_coupling="free")),
        ("ratio", lambda d: d["front_guide"]["coupling"]["ratio_by_side"].update(left=0.5)),
        ("actuator", lambda d: d["front_guide"]["coupling"].update(guide_actuator_count=1)),
        ("motor count", lambda d: d["front_guide"].update(drive_motor_count=False)),
        ("torque", lambda d: d["front_guide"].update(added_drive_torque_nm=0.1)),
        ("relation count", lambda d: d["same_side_wheel_coupling"].update(relations_per_world=12)),
        ("guide count", lambda d: d["same_side_wheel_coupling"].update(guide_relations_per_world=0)),
    )
    for label, mutate in invalid_cases:
        env.drive_config = copy.deepcopy(drive)
        mutate(env.drive_config)
        with pytest.raises(ValueError, match="coupled front-guide|same-side wheel/guide"):
            runner._validate_rough_profile_environment(env, physics)


def test_fast_stop_evaluation_records_native_terminal_stop_contract_and_times() -> None:
    objective = runner._fast_stop_objective_contract(
        runner._profile_config(runner.PROFILE_RANDOM_ROUGH_FAST_STOP)
    )

    def row(seed: int, success: bool, time_s: float, linear: float, angular: float, hold: float):
        return {
            "seed": seed, "success": success, "episode_time_s": time_s,
            "final_distance": 0.3 if success else 2.0,
            "goal_stop": {
                "schema": "goal_stop_v1", "profile": "fast_stop_v1",
                "linear_speed_m_s": linear, "angular_speed_rad_s": angular, "hold_s": hold,
                "stop_speed_limit_m_s": 0.05, "stop_angular_speed_limit_rad_s": 0.1,
                "hold_required_s": 0.5, "distance_tolerance_m": 0.4,
                "within_stop_limits": linear < 0.05 and angular < 0.1,
                "speed_source": "root_com_vel_w",
            },
        }

    report = {
        "rows": [row(1, True, 12.0, 0.049, 0.099, 0.5), row(2, False, 60.0, 0.2, 0.4, 0.1)],
        "summary": {"episodes": 2, "successes": 1},
    }
    result = runner._add_fast_stop_evaluation_summaries(report, objective)
    assert result["summary"]["mean_success_time_s"] == 12.0
    assert result["summary"]["median_success_time_s"] == 12.0
    assert result["summary"]["goal_stop_contract"] == objective["goal_stop"]
    assert report["summary"] == {"episodes": 2, "successes": 1}

    fallback = copy.deepcopy(report)
    fallback["rows"][0]["goal_stop"]["speed_source"] = "task_forward_speed_fallback"
    with pytest.raises(ValueError, match="native root_com_vel_w"):
        runner._add_fast_stop_evaluation_summaries(fallback, objective)
    threshold = copy.deepcopy(report)
    threshold["rows"][0]["goal_stop"]["linear_speed_m_s"] = 0.05
    threshold["rows"][0]["goal_stop"]["within_stop_limits"] = False
    with pytest.raises(ValueError, match="violates its frozen goal-stop contract"):
        runner._add_fast_stop_evaluation_summaries(threshold, objective)
    inconsistent_limit_flag = copy.deepcopy(report)
    inconsistent_limit_flag["rows"][0]["goal_stop"]["within_stop_limits"] = False
    with pytest.raises(ValueError, match="inconsistent with strict speed thresholds"):
        runner._add_fast_stop_evaluation_summaries(inconsistent_limit_flag, objective)


def test_fast_stop_paired_time_gate_requires_same_success_seeds_and_no_safety_regression() -> None:
    before = {
        "rows": [
            {"seed": 1, "success": True, "episode_time_s": 12.0, "damage": 0, "collisions": 0},
            {"seed": 2, "success": True, "episode_time_s": 15.0, "damage": 0, "collisions": 0},
            {"seed": 3, "success": True, "episode_time_s": 18.0, "damage": 1, "collisions": 1},
        ],
        "summary": {"damaged_episodes": 1, "mean_damage": 1.0, "total_collisions": 1},
    }
    after = copy.deepcopy(before)
    after["rows"][0]["episode_time_s"] = 11.0
    after["rows"][1]["episode_time_s"] = 15.0
    result = runner._fast_stop_time_assessment(
        before, after, pairing_valid=True, termination_safety_not_increased=True,
    )
    assert result["status"] == "paired_time_improvement"
    assert result["time_improvement"] is True
    assert result["baseline_clean_success_seeds"] == [1, 2]
    assert [row["seed"] for row in result["paired_times_s"]] == [1, 2]

    changed = copy.deepcopy(after)
    changed["rows"][1]["success"] = False
    result = runner._fast_stop_time_assessment(
        before, changed, pairing_valid=True, termination_safety_not_increased=True,
    )
    assert result["status"] == "ineligible_success_count_changed"
    regressed = copy.deepcopy(after)
    regressed["summary"]["total_collisions"] = 2
    result = runner._fast_stop_time_assessment(
        before, regressed, pairing_valid=True, termination_safety_not_increased=True,
    )
    assert result["status"] == "ineligible_safety_regression"

    missing_metrics = copy.deepcopy(before)
    missing_metrics["rows"][0].pop("collisions")
    result = runner._fast_stop_time_assessment(
        missing_metrics, after, pairing_valid=True, termination_safety_not_increased=True,
    )
    assert result["status"] == "ineligible_missing_row_safety_metrics"


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
    path = output / "checkpoint.pt"
    path.write_bytes(b"synthetic RSL checkpoint")
    from isaac_field import rsl_rl_policy

    payload = {"num_timesteps": 0, "field_policy_contract": {}}
    if checkpoint_action is not None:
        payload["field_policy_contract"]["action_config"] = checkpoint_action
    monkeypatch.setattr(rsl_rl_policy, "load_rsl_checkpoint_payload", lambda *_args, **_kwargs: payload)

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


def _rough_paired_report(seeds: tuple[int, ...], *, variant: str) -> dict:
    report = _paired_report(seeds, variant=variant)
    report["evaluation_config"] = {
        "environment_seed": 80_003, "plants": 0,
        "goal_distance_m": None, "goal_angle_deg": 180.0,
        "goal_sampling_profile": runner.PROFILE_RANDOM_ROUGH,
        "terrain_config": dict(runner.RANDOM_ROUGH_TERRAIN_CONFIG),
    }
    report["runtime_assets"] = [{
        "asset": "field", "path": "/tmp/rough/terrain.usda",
        "sha256": "a" * 64, "size_bytes": 456,
    }]
    for row in report["rows"]:
        row["reset_info"] = {"terrain_seed": 80_003}
    return report


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


def test_rough_pairing_requires_equal_evaluation_and_terrain_seeds() -> None:
    seeds = tuple(range(87_001, 87_005))
    baseline = _rough_paired_report(seeds, variant="baseline")
    final = _rough_paired_report(seeds, variant="final")
    final["rows"][2]["reset_info"]["terrain_seed"] = 80_004

    result = runner._paired_dev_conditions(
        {"baseline": baseline, "final": final}, seeds, require_rough_terrain_pair=True,
    )
    assert result["valid"] is False
    assert any(row["variant"] == "terrain_seed" for row in result["mismatches"])


@pytest.mark.parametrize("fault,expected_variant", [
    ("missing_seeds", "terrain_seed"),
    ("one_sided_seed", "terrain_seed"),
    ("missing_field", "field_asset"),
    ("one_sided_field", "field_asset"),
    ("different_field_hash", "field_asset"),
])
def test_rough_pairing_fails_closed_on_missing_or_different_terrain_evidence(
    fault: str, expected_variant: str,
) -> None:
    seeds = tuple(range(87_001, 87_005))
    baseline = _rough_paired_report(seeds, variant="baseline")
    final = _rough_paired_report(seeds, variant="final")
    if fault == "missing_seeds":
        for report in (baseline, final):
            for row in report["rows"]:
                row["reset_info"].pop("terrain_seed")
    elif fault == "one_sided_seed":
        final["rows"][2]["reset_info"].pop("terrain_seed")
    elif fault == "missing_field":
        baseline["runtime_assets"] = []
    elif fault == "one_sided_field":
        final["runtime_assets"] = []
    elif fault == "different_field_hash":
        final["runtime_assets"][0]["sha256"] = "b" * 64

    result = runner._paired_dev_conditions(
        {"baseline": baseline, "final": final}, seeds, require_rough_terrain_pair=True,
    )
    assert result["valid"] is False
    assert any(row["variant"] == expected_variant for row in result["mismatches"])


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
