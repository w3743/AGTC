"""CPU-only checks for the compatible fast-stop mastery candidate profile."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import scripts.train_isaac_mastery as runner


FAKE_SOURCES = (
    "isaac_field/frozen_component.py",
    "scripts/train_isaac_mastery.py",
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    for relative in FAKE_SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen: {relative}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "_sources", lambda: FAKE_SOURCES)
    monkeypatch.setattr(runner.foundation, "_package_versions", lambda: {"python": "test"})
    monkeypatch.setattr(runner.foundation, "_isaac_commit", lambda: "test-commit")
    monkeypatch.setattr(runner.foundation, "_robot_geometry", lambda: {"spec": "test-proxy"})
    monkeypatch.setattr(runner.foundation, "_action_config", lambda: {"schema": "test", "dimensions": 4})
    monkeypatch.setattr(runner.foundation, "_experiment_architecture",
                        lambda *_args, **_kwargs: {"input_contract": "sensor-minimal-v6"})
    return root


def _prepare(root: Path, name: str, **kwargs) -> Path:
    output = root / "results" / name
    runner.prepare(output, budget=2_560, block_steps=512, **kwargs)
    return output


def test_legacy_default_profile_remains_compatible_with_explicit_legacy(workspace):
    default = _prepare(workspace, "default")
    explicit = _prepare(workspace, "legacy", profile=runner.PROFILE_LEGACY)
    default_protocol = json.loads((default / "protocol.json").read_text(encoding="utf-8"))
    explicit_protocol = json.loads((explicit / "protocol.json").read_text(encoding="utf-8"))

    assert default_protocol == explicit_protocol
    assert default_protocol["profile"] == "legacy"
    assert default_protocol["stages"] == list(runner.STAGES)
    assert default_protocol["hyperparameters"]["gamma"] == 0.995
    assert "policy_architecture" not in default_protocol
    assert runner.preflight(default)["method"] == "mastery"


def test_fast_stop_512_freezes_course_contract_policy_and_safe_promotion(workspace):
    output = _prepare(workspace, "fast_stop", profile=runner.PROFILE_FAST_STOP_512)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))

    assert protocol["schema"] == runner.SCHEMA
    assert protocol["profile"] == "fast_stop_512"
    expected_config = runner.foundation._profile_config(
        runner.foundation.PROFILE_RANDOM_ROUGH_FAST_STOP)
    expected_config.update(plants=0, terrain_enabled=False, rock_density=0.0)
    assert protocol["config"] == json.loads(json.dumps(expected_config))
    assert protocol["config"]["goto_reward_profile"] == "fast_stop_v1"
    assert protocol["config"]["goal_tolerance"] == 0.4
    assert protocol["config"]["goal_stop_speed"] == 0.05
    assert protocol["config"]["goal_stop_angular_speed"] == 0.1
    assert protocol["config"]["goal_hold"] == 0.5
    assert protocol["config"]["max_time"] == 60.0
    assert protocol["config"]["plants"] == 0
    assert protocol["config"]["terrain_enabled"] is False
    assert protocol["config"]["rock_density"] == 0.0
    assert protocol["goal_objective_contract"] == runner.foundation._fast_stop_objective_contract(
        runner._profile_config(runner.PROFILE_FAST_STOP_512))

    stages = protocol["stages"]
    assert [stage["name"] for stage in stages] == ["near", "turn30", "turn60", "turn90", "range"]
    assert [stage["goal_distance"] for stage in stages] == [
        [0.6, 1.2], [0.6, 1.2], [0.6, 1.2], [0.6, 1.2], [1.0, 3.0]
    ]
    assert [stage["goal_angle"] for stage in stages] == pytest.approx(
        [0.2617993877991494, 0.5235987755982988, 1.0471975511965976,
         1.5707963267948966, 1.5707963267948966]
    )
    assert {stage["plants"] for stage in stages} == {0}
    assert protocol["policy_architecture"] == {"lstm_hidden_size": 512, "n_lstm_layers": 1}
    assert protocol["hyperparameters"]["lstm_hidden_size"] == 512
    assert protocol["hyperparameters"]["gamma"] == 1.0
    assert protocol["hyperparameters"]["gae_lambda"] == 0.95
    assert protocol["mastery"]["episodes_per_stage"] == 16
    assert protocol["mastery"]["min_clean_success_rate"] == 0.8
    assert protocol["mastery"]["required_consecutive_passes"] == 2
    assert protocol["mastery"]["max_damaged_episodes"] == 0
    assert protocol["mastery"]["max_total_collisions"] == 0
    assert protocol["mastery"]["active_blocks_per_replay"] == 3
    assert set(protocol["mastery"]["seed_banks"]) == {stage["name"] for stage in stages}
    guide = protocol["profile_physics_contract"]["front_guide_drive_coupling"]
    assert guide["enabled"] is True
    assert guide["relation"] == "q_guide = (R_main/R_guide)*q_same_side_wheel0"
    assert protocol["profile_physics_contract"]["same_side_wheel_coupling_counts"] == {
        "relations_per_world": 14, "wheel_relations_per_world": 12, "guide_relations_per_world": 2,
    }
    assert runner.preflight(output)["profile"] == "fast_stop_512"


def test_fast_stop_512_rejects_noncanonical_discount(workspace):
    with pytest.raises(ValueError, match="requires gamma=1.0"):
        _prepare(workspace, "bad_gamma", profile=runner.PROFILE_FAST_STOP_512, gamma=0.995)
    with pytest.raises(ValueError, match="default gae_lambda=0.95"):
        _prepare(workspace, "bad_gae", profile=runner.PROFILE_FAST_STOP_512, gae_lambda=0.9)
    with pytest.raises(ValueError, match="requires method=mastery"):
        _prepare(workspace, "bad_method", profile=runner.PROFILE_FAST_STOP_512, method="fixed")


@pytest.mark.parametrize("mutation", ("source", "protocol"))
def test_fast_stop_512_preflight_rejects_frozen_source_and_protocol_mutations(
    workspace, mutation
):
    output = _prepare(workspace, f"drift_{mutation}", profile=runner.PROFILE_FAST_STOP_512)
    if mutation == "source":
        snapshot = output / "source_snapshot" / FAKE_SOURCES[0]
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
        expected = "frozen source mismatch"
    else:
        protocol_path = output / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["config"]["terrain_enabled"] = True
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Model a rewritten status digest too: profile semantic checks must still reject it.
        status_path = output / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["protocol_sha256"] = runner.foundation._sha256(protocol_path)
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        expected = "task config differs"
    with pytest.raises(ValueError, match=expected):
        runner.preflight(output)


def test_fast_stop_512_global_probe_uses_canonical_global_rough_transfer_suites(
    workspace, monkeypatch
):
    output = _prepare(workspace, "global_probe", profile=runner.PROFILE_FAST_STOP_512)
    protocol = runner.preflight(output)
    calls = []

    def evaluate(_output, protocol_arg, checkpoint, stage, seeds, *, environment_seed, task_config):
        assert protocol_arg is protocol
        calls.append((stage, list(seeds), environment_seed, task_config))
        return {
            "action_config": {"schema": "test"},
            "drive_config": {"front_guide": {"rotation_coupling": "same_side_track_speed"}},
            "observation_config": {"schema": "sensor-minimal-v6"},
            "damage_config": {"metric": "root-protection-v1"},
            "damage_metric": "root-protection-v1",
            "distance_cost": 0.0,
            "goal_distance": stage["goal_distance"],
            "goal_angle_rad": stage["goal_angle"],
            "mode": stage["mode"],
            "deterministic": True,
            "seeds": list(seeds),
            "rows": [{
                "seed": seed, "success": False, "damage": 0, "collisions": 0,
                "damage_metric": "root-protection-v1",
                "reset_initial_condition": {
                    "plant_xy_sha256": f"plants-{seed}",
                    "start_pose_xy_yaw": [float(seed % 5), 1.0, 0.0],
                    "goal_xy": [2.0, 3.0],
                },
                "trajectory_start_matches_reset": True,
            } for seed in seeds],
            "evaluation_config": {"environment_seed": environment_seed,
                                  "task": task_config, "stage": stage},
        }

    monkeypatch.setattr(runner, "_evaluate_stage", evaluate)
    report = runner._global_probe(output, protocol, Path("checkpoint.zip"))

    assert len(calls) == 2
    assert [call[0]["name"] for call in calls] == ["rough_a", "rough_b"]
    assert all(call[0]["goal_distance"] is None for call in calls)
    assert all(call[0]["plants"] == 0 and call[0]["mode"] == "goto" for call in calls)
    assert all(call[0]["goal_angle"] == pytest.approx(3.141592653589793) for call in calls)
    assert all(call[3]["max_time"] == 60.0 and call[3]["terrain_enabled"] is True for call in calls)
    assert report["probe"]["probe_stage"] == "global_suite"
    assert report["evaluation_config"]["claim_limit"] == protocol["transfer_probe"]["claim_limit"]
    assert report["evaluation_config"]["suite"] == protocol["transfer_probe"]["purpose"]
    assert [member["evaluation_config"]["stage"]["name"]
            for member in report["evaluation_config"]["members"]] == ["rough_a", "rough_b"]
    assert report["seeds"] == [seed for member in protocol["transfer_probe"]["members"]
                                for seed in member["seeds"]]

    controller = runner._controller(protocol, runner._mastery_specs(protocol, report), report)
    assert controller.validate_best_probe(report)["probe_stage"] == "global_suite"
    candidate = runner._global_probe(output, protocol, Path("candidate.zip"))
    comparison = runner._select_best(protocol, candidate, report)
    assert comparison["comparable"] is True
    assert comparison["paired_seed_count"] == len(report["seeds"])
    assert [member["stage"] for member in comparison["member_checks"]] == ["rough_a", "rough_b"]


def test_evaluate_stage_forwards_unbounded_goal_distance_as_none(workspace, monkeypatch):
    output = _prepare(workspace, "none_goal_distance", profile=runner.PROFILE_FAST_STOP_512)
    protocol = runner.preflight(output)
    protocol["device"] = "cpu"
    member = protocol["transfer_probe"]["members"][0]
    stage = {key: member[key] for key in ("name", "plants", "goal_distance", "goal_angle")}
    stage["mode"] = "goto"
    engine = SimpleNamespace(field_cfg=runner.Config(**protocol["transfer_probe"]["environment_config"]))
    engine.closed = False
    engine.close = lambda: setattr(engine, "closed", True)
    opened = []
    evaluated = []

    def open_env(_output, protocol_arg, stage_arg, *, training, seed, task_config=None):
        assert protocol_arg is protocol
        assert stage_arg["goal_distance"] is None
        assert training is False
        assert seed == member["environment_seed"]
        assert task_config == protocol["transfer_probe"]["environment_config"]
        opened.append(stage_arg)
        return engine

    def evaluate_model(adapter, checkpoint, seeds, **kwargs):
        assert adapter is engine
        evaluated.append(kwargs)
        return {"seeds": list(seeds), "goal_distance": kwargs["goal_distance"]}

    env_module = ModuleType("isaac_field.env")
    env_module.SingleFieldAdapter = lambda value: value
    learning_module = ModuleType("isaac_field.learning")
    learning_module.evaluate_model = evaluate_model
    monkeypatch.setitem(sys.modules, "isaac_field.env", env_module)
    monkeypatch.setitem(sys.modules, "isaac_field.learning", learning_module)
    monkeypatch.setattr(runner, "_open_env", open_env)

    report = runner._evaluate_stage(
        output, protocol, Path("checkpoint.zip"), stage, member["seeds"],
        environment_seed=member["environment_seed"],
        task_config=protocol["transfer_probe"]["environment_config"],
    )

    assert opened == [stage]
    assert evaluated[0]["goal_distance"] is None
    assert report["goal_distance"] is None
    assert report["evaluation_config"]["stage"] == stage
    assert engine.closed is True
