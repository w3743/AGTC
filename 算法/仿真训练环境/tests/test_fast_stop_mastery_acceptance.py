"""Offline acceptance checks for the fast-stop mastery protocol.

Synthetic probe rows below exercise the real protocol/controller/selection
logic. They are fixtures, not simulator results or evidence of learned skill.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import scripts.train_isaac_mastery as runner
from test_curriculum_control import _complete_block, _report


@pytest.fixture(scope="module")
def fast_stop_trial(tmp_path_factory):
    output = tmp_path_factory.mktemp("fast-stop-mastery") / "trial"
    runner.prepare(
        output,
        profile=runner.PROFILE_FAST_STOP_512,
        budget=196_608,
        block_steps=8_192,
    )
    return output, runner.preflight(output)


def _synthetic_baseline(protocol: dict) -> dict:
    transfer_seeds = [
        seed
        for member in protocol["transfer_probe"]["members"]
        for seed in member["seeds"]
    ]
    return {
        "action_config": protocol["action_config"],
        "drive_config": {"fixture": "fast-stop rough drive contract"},
        "observation_config": {"schema": runner.foundation.INPUT_SCHEMA},
        "damage_config": _synthetic_damage_config(protocol),
        "damage_metric": protocol["config"]["damage_metric"],
        "distance_cost": 0.0,
        "goal_distance": None,
        "goal_angle_rad": None,
        "mode": "goto_suite",
        "deterministic": True,
        "evaluation_config": {"fixture": "synthetic baseline contract"},
        "seeds": transfer_seeds,
    }


def _synthetic_damage_config(protocol: dict) -> dict:
    """Return a stable schema-valid fixture; no physical contact was simulated."""
    return {
        "metric": protocol["config"]["damage_metric"],
        "penalty_per_plant": float(protocol["config"]["damage_penalty"]),
        "track_geometry": "synthetic fixture",
        "track_contact_length_m": 0.595,
        "track_band_width_m": 0.06,
        "track_center_separation_m": 0.41,
        "track_outer_width_m": 0.47,
        "track_center_separation_status": "synthetic fixture",
        "track_vertical_model": "synthetic fixture",
        "stem_radius_source": "synthetic fixture",
        "chassis_geometry": "synthetic fixture",
        "chassis_length_m": 0.8,
        "chassis_width_m": 0.3,
        "chassis_height_m": 0.3,
        "vertical_model": "synthetic fixture",
    }


def _mastery_report(protocol: dict, stage_name: str, *, clean_count: int) -> dict:
    synthetic_baseline = _synthetic_baseline(protocol)
    spec = runner._mastery_specs(protocol, synthetic_baseline)[stage_name]
    return _report(
        "mastery",
        stage_name,
        spec.seeds,
        spec.config,
        clean_count=clean_count,
    )


def test_fast_stop_mastery_prepares_real_frozen_offline_contract(fast_stop_trial):
    output, protocol = fast_stop_trial
    assert protocol["profile"] == "fast_stop_512"
    assert protocol["method"] == "mastery"
    assert protocol["budget"] == 196_608
    assert protocol["block_steps"] == 8_192
    assert protocol["config"]["goto_reward_profile"] == "fast_stop_v1"
    assert protocol["config"]["goal_tolerance"] == 0.4
    assert protocol["config"]["goal_stop_speed"] == 0.05
    assert protocol["config"]["goal_stop_angular_speed"] == 0.1
    assert protocol["config"]["goal_hold"] == 0.5
    assert protocol["config"]["max_time"] == 60.0
    assert protocol["config"]["terrain_enabled"] is False
    assert protocol["config"]["rock_density"] == 0.0

    assert [stage["name"] for stage in protocol["stages"]] == [
        "near", "turn30", "turn60", "turn90", "range"
    ]
    assert [stage["goal_distance"] for stage in protocol["stages"]] == [
        [0.6, 1.2], [0.6, 1.2], [0.6, 1.2], [0.6, 1.2], [1.0, 3.0]
    ]
    assert {stage["plants"] for stage in protocol["stages"]} == {0}
    assert protocol["policy_architecture"] == {
        "lstm_hidden_size": 512, "n_lstm_layers": 1
    }
    assert protocol["hyperparameters"]["gamma"] == 1.0
    assert protocol["hyperparameters"]["gae_lambda"] == 0.95

    transfer = protocol["transfer_probe"]
    assert transfer["environment_config"]["terrain_enabled"] is True
    assert [member["name"] for member in transfer["members"]] == ["rough_a", "rough_b"]
    assert all(member["goal_distance"] is None for member in transfer["members"])
    assert all(member["plants"] == 0 for member in transfer["members"])
    assert protocol["selection"]["seed_banks"] == [
        member["seeds"] for member in transfer["members"]
    ]

    mastery_seeds = {
        seed for bank in protocol["mastery"]["seed_banks"].values() for seed in bank
    }
    transfer_seeds = {
        seed for member in transfer["members"] for seed in member["seeds"]
    }
    training_seeds = {
        protocol["training_seed_base"] + index
        for index in range(protocol["budget"] // 512 + 1)
    }
    assert mastery_seeds.isdisjoint(transfer_seeds)
    assert mastery_seeds.isdisjoint(training_seeds)
    assert transfer_seeds.isdisjoint(training_seeds)
    assert all(not 83_000 <= seed <= 83_999
               for seed in mastery_seeds | transfer_seeds | training_seeds)
    assert protocol["blind_seeds"] == {"range": [83_000, 83_999], "used": False}
    # Environment seeds identify scene generation; they are not episode seed-bank entries.
    assert all(member["environment_seed"] not in transfer_seeds
               for member in transfer["members"])

    status = (output / "run_status.json").read_text(encoding="utf-8")
    assert '"status": "prepared"' in status
    assert '"num_timesteps": 0' in status
    assert not (output / "launch.log").exists()
    assert not (output / "untrained_baseline.zip").exists()


def test_fast_stop_mastery_uses_consecutive_probes_and_replays_mastered_stage(
    fast_stop_trial,
):
    _output, protocol = fast_stop_trial
    baseline = _synthetic_baseline(protocol)
    specs = runner._mastery_specs(protocol, baseline)
    controller = runner._controller(protocol, specs, baseline)

    _complete_block(controller)
    first = controller.record_probe(_mastery_report(protocol, "near", clean_count=16))
    assert first["active_stage"]["name"] == "near"
    assert first["consecutive_passes"] == 1
    with pytest.raises(RuntimeError, match="controller marks one due"):
        controller.record_probe(_mastery_report(protocol, "near", clean_count=16))

    _complete_block(controller)
    failed = controller.record_probe(_mastery_report(protocol, "near", clean_count=12))
    assert failed["active_stage"]["name"] == "near"
    assert failed["consecutive_passes"] == 0

    _complete_block(controller)
    pass_one = controller.record_probe(_mastery_report(protocol, "near", clean_count=13))
    assert pass_one["active_stage"]["name"] == "near"
    assert pass_one["consecutive_passes"] == 1
    _complete_block(controller)
    pass_two = controller.record_probe(_mastery_report(protocol, "near", clean_count=13))
    assert pass_two["active_stage"]["name"] == "turn30"
    assert pass_two["last_event"]["promotion"] == "advanced"
    assert pass_two["consumed_timesteps"] == 32_768

    replay_plan = controller.next_training_plan(max_timesteps=8_192)
    assert replay_plan["sampled_stage"] == "near"
    assert replay_plan["long_run_cycle"]["block_order"] == [
        "near", "turn30", "turn30", "turn30"
    ]
    assert replay_plan["long_run_cycle"]["target_distribution"] == [
        {"stage": pass_two["long_run_cycle"]["target_distribution"][0]["stage"],
         "weight": 0.25},
        {"stage": pass_two["long_run_cycle"]["target_distribution"][1]["stage"],
         "weight": 0.75},
    ]
    controller.record_training(
        replay_plan["timesteps"],
        sampled_stage=replay_plan["sampled_stage"],
        actual_sampling=replay_plan["sampling"],
    )
    after_replay = controller.decision()
    assert after_replay["active_stage"]["name"] == "turn30"
    assert after_replay["active_stage_timesteps"] == 0
    assert after_replay["probe_due"] is False
    assert controller.next_training_plan(max_timesteps=8_192)["sampled_stage"] == "turn30"


def test_fast_stop_mastery_minimum_stage_timesteps_means_remaining_budget(
    fast_stop_trial, tmp_path,
):
    _output, _full_protocol = fast_stop_trial
    output = tmp_path / "short-budget"
    runner.prepare(
        output,
        profile=runner.PROFILE_FAST_STOP_512,
        budget=24_576,
        block_steps=8_192,
    )
    protocol = runner.preflight(output)
    baseline = _synthetic_baseline(protocol)
    controller = runner._controller(
        protocol, runner._mastery_specs(protocol, baseline), baseline
    )

    for _ in range(2):
        _complete_block(controller)
        decision = controller.record_probe(
            _mastery_report(protocol, "near", clean_count=16)
        )
    assert decision["consumed_timesteps"] == 16_384
    assert decision["remaining_timesteps"] == 8_192
    assert decision["active_stage"]["name"] == "near"
    assert decision["last_event"]["promotion"] == "blocked_by_budget"
    assert decision["promotion_blocked_reason"] == "insufficient_global_budget_for_next_stage"


def _synthetic_member_report(
    protocol: dict,
    member: dict,
    *,
    clean_count: int,
    checkpoint: str,
) -> dict:
    stage = {
        "name": member["name"],
        "mode": "goto",
        "goal_distance": member["goal_distance"],
        "goal_angle": member["goal_angle"],
        "plants": member["plants"],
    }
    contract = {
        "action_config": protocol["action_config"],
        "drive_config": {"fixture": "same synthetic rough drive contract"},
        "observation_config": {"schema": runner.foundation.INPUT_SCHEMA},
        "damage_config": _synthetic_damage_config(protocol),
        "damage_metric": protocol["config"]["damage_metric"],
        "distance_cost": 0.0,
        "goal_distance": member["goal_distance"],
        "goal_angle_rad": member["goal_angle"],
        "mode": "goto",
        "deterministic": True,
        "evaluation_config": {
            "task": protocol["transfer_probe"]["environment_config"],
            "environment_seed": member["environment_seed"],
            "stage": stage,
        },
    }
    report = _report(
        "best_checkpoint",
        member["name"],
        member["seeds"],
        contract,
        clean_count=clean_count,
    )
    report["seeds"] = list(member["seeds"])
    report["checkpoint"] = checkpoint
    return report


def test_fast_stop_selection_requires_each_real_transfer_member_not_to_regress(
    fast_stop_trial, monkeypatch,
):
    output, protocol = fast_stop_trial
    outcomes = {
        "baseline": {"rough_a": 2, "rough_b": 2},
        "regressed": {"rough_a": 1, "rough_b": 4},
        "improved": {"rough_a": 2, "rough_b": 3},
    }

    def synthetic_evaluation(_output, protocol_arg, checkpoint, stage, seeds,
                             *, environment_seed, task_config):
        assert protocol_arg is protocol
        member = next(item for item in protocol["transfer_probe"]["members"]
                      if item["name"] == stage["name"])
        assert list(seeds) == member["seeds"]
        assert environment_seed == member["environment_seed"]
        assert task_config == protocol["transfer_probe"]["environment_config"]
        return _synthetic_member_report(
            protocol,
            member,
            clean_count=outcomes[Path(checkpoint).stem][member["name"]],
            checkpoint=str(checkpoint),
        )

    # Only the hardware/simulator evaluation boundary is replaced. Aggregation,
    # probe validation, comparison, and member-level selection stay production code.
    monkeypatch.setattr(runner, "_evaluate_stage", synthetic_evaluation)
    best = runner._global_probe(output, protocol, Path("baseline.zip"))
    assert best["probe"]["probe_stage"] == "global_suite"
    controller = runner._controller(protocol, runner._mastery_specs(protocol, best), best)
    assert controller.validate_best_probe(best)["probe_stage"] == "global_suite"

    more_total_but_rough_a_regressed = runner._global_probe(
        output, protocol, Path("regressed.zip")
    )
    rejected = runner._select_best(protocol, more_total_but_rough_a_regressed, best)
    assert rejected["comparable"] is True
    assert rejected["candidate_metrics"]["clean_successes"] > rejected["best_metrics"]["clean_successes"]
    assert rejected["assessment"] == "suite_member_regression"
    assert rejected["replace_best"] is False
    checks = {row["stage"]: row for row in rejected["member_checks"]}
    assert checks["rough_a"]["non_regression"] is False
    assert checks["rough_b"]["non_regression"] is True

    safe_candidate = runner._global_probe(output, protocol, Path("improved.zip"))
    accepted = runner._select_best(protocol, safe_candidate, best)
    assert accepted["replace_best"] is True
    assert all(row["non_regression"] for row in accepted["member_checks"])
