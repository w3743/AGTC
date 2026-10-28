import json

import pytest

from isaac_field.action_contract import ACTION_COMPONENTS, ACTION_SCHEMA
from isaac_field.training_diagnostics import TrainingDiagnosticsAccumulator


def _step(*, actions, clipped_actions, rewards, dones, infos):
    return {
        "actions": actions,
        "clipped_actions": clipped_actions,
        "rewards": rewards,
        "dones": dones,
        "infos": infos,
    }


def test_rollout_separates_sampled_and_vecenv_actions_and_tracks_episode_return():
    accumulator = TrainingDiagnosticsAccumulator(stage="near")
    accumulator.observe_step(_step(
        actions=[[1.4, -0.2], [0.5, -1.7]],
        clipped_actions=[[1.0, -0.2], [0.5, -1.0]],
        rewards=[1.0, -0.5],
        dones=[False, True],
        infos=[{}, {"reason": "time_limit", "damage": 0, "collisions": 2}],
    ))
    accumulator.observe_step(_step(
        actions=[[0.3, 0.4], [0.1, -0.2]],
        clipped_actions=[[0.3, 0.4], [0.1, -0.2]],
        rewards=[2.0, 0.25],
        dones=[True, False],
        infos=[{"reason": "goal_reached", "is_success": True, "damage": 0}, {}],
    ))

    record = accumulator.finish_rollout(
        num_timesteps=8,
        logger_metrics={"train/approx_kl": 0.02, "train/clip_fraction": 0.1},
    )

    assert record["stage"] == "near"
    assert record["num_timesteps"] == 8
    assert record["callback_vector_steps"] == 2
    assert record["sampled_policy_actions"]["count"] == 8
    assert record["actions_passed_to_vec_env"]["count"] == 8
    assert record["action_clip_observation"]["changed_elements"] == 2
    assert record["action_clip_observation"]["paired_finite_elements"] == 8
    assert record["action_clip_observation"]["fraction_changed_by_sb3_action_clip"] == pytest.approx(0.25)
    assert "actuator" in record["action_clip_observation"]["note"]
    assert record["completed_episode_returns"]["count"] == 2
    assert record["completed_episode_returns"]["sum"] == pytest.approx(2.5)
    assert [row["env_index"] for row in record["completed_episodes"]] == [1, 0]
    assert [row["callback_reward_return"] for row in record["completed_episodes"]] == [-0.5, 3.0]
    assert record["completed_episodes"][0]["terminal_info"] == {
        "reason": "time_limit", "damage": 0, "collisions": 2,
    }
    assert record["completed_episodes"][1]["terminal_info"] == {
        "reason": "goal_reached", "is_success": True, "damage": 0,
    }
    assert [row["reward_components"]["status"] for row in record["completed_episodes"]] == [
        "missing", "missing",
    ]
    assert record["reward_components"]["status"] == "not_provided_by_environment"
    assert record["logger_metrics"] == {"train/approx_kl": 0.02, "train/clip_fraction": 0.1}
    assert record["logger_metric_alignment"]["sampled_at"] == (
        "on_rollout_end_before_current_rollout_optimizer_update"
    )
    assert record["logger_metric_alignment"]["expected_lag_rollouts"] == 1
    assert record["logger_metric_alignment"]["source_rollout_index"] is None
    assert record["logger_metric_alignment"]["status"] == "source_rollout_unknown_before_callback"
    assert record["logger_metric_alignment"]["snapshot_status"] == "numeric_values_observed"
    assert "logger.dump clears name_to_value" in record["logger_metric_alignment"]["dump_order_note"]
    json.dumps(record, allow_nan=False)


def test_reward_components_are_observed_only_when_supplied_and_logger_maps_to_prior_rollout():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[0.0], clipped_actions=[0.0], rewards=[0.2], dones=[False],
        infos=[{"reward_components": {"progress": 0.3, "time": -0.1}}],
    ), info_source="environment_info_provider")
    first = accumulator.finish_rollout(num_timesteps=4, logger_metrics=None)
    assert first["terminal_info_source"] == "environment_info_provider"
    assert first["reward_components"]["status"] == "observed"
    assert first["reward_components"]["totals"]["progress"]["sum"] == pytest.approx(0.3)
    assert first["reward_components"]["per_step_reconciliation"]["status"] == "matched"
    assert first["logger_metrics"] is None
    assert first["logger_metric_alignment"]["snapshot_status"] == "empty_or_non_numeric_snapshot"
    assert first["logger_metric_alignment"]["status"] == "no_numeric_metrics_available"

    accumulator.observe_step(_step(
        actions=[0.0], clipped_actions=[0.0], rewards=[0.1], dones=[False], infos=[{}],
    ))
    second = accumulator.finish_rollout(num_timesteps=8, logger_metrics={"train/value_loss": 1.5})
    assert second["logger_metric_alignment"]["source_rollout_index"] == 1
    assert second["logger_metric_alignment"]["source_num_timesteps"] == 4
    assert second["logger_metric_alignment"]["expected_lag_rollouts"] == 1

    accumulator.observe_step(_step(
        actions=[0.0], clipped_actions=[0.0], rewards=[0.1], dones=[False], infos=[{}],
    ))
    empty_snapshot = accumulator.finish_rollout(num_timesteps=12, logger_metrics={})
    assert empty_snapshot["logger_metrics"] is None
    assert empty_snapshot["logger_metric_alignment"]["source_rollout_index"] == 2
    assert empty_snapshot["logger_metric_alignment"]["status"] == "no_numeric_metrics_available"
    assert empty_snapshot["logger_metric_alignment"]["snapshot_status"] == "empty_or_non_numeric_snapshot"

    final = accumulator.finish_training(
        num_timesteps=12,
        logger_metrics={"train/value_loss": 0.75, "ignored": float("nan")},
    )
    assert final["event_type"] == "final_optimizer_metrics"
    assert final["logger_metrics"] == {"train/value_loss": 0.75}
    assert final["logger_metric_alignment"]["source_rollout_index"] == 3
    assert final["logger_metric_alignment"]["source_num_timesteps"] == 12
    assert final["logger_metric_alignment"]["expected_lag_rollouts"] == 0
    assert final["logger_metric_alignment"]["snapshot_status"] == "numeric_values_observed"
    json.dumps(final, allow_nan=False)


def test_missing_and_partial_components_and_action_shape_mismatch_are_explicit():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[1.1, 2.2]], clipped_actions=[1.0, 2.0], rewards=[float("nan")],
        dones=[False], infos=[{}],
    ))
    record = accumulator.finish_rollout(num_timesteps=1, logger_metrics={"train/approx_kl": float("inf")})

    assert record["action_clip_observation"]["paired_finite_elements"] == 0
    assert record["action_clip_observation"]["pairing_status"] == "action_array_size_mismatch"
    assert record["action_clip_observation"]["fraction_changed_by_sb3_action_clip"] is None
    assert record["callback_rewards"]["count"] == 0
    assert record["callback_rewards"]["non_finite_count"] == 1
    assert record["reward_components"]["status"] == "not_provided_by_environment"
    assert record["logger_metrics"] is None
    assert record["reward_components"]["per_step_reconciliation"]["status"] == "unavailable"
    json.dumps(record, allow_nan=False)


def test_reward_components_report_partial_vector_steps():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[0.0], [0.0]], clipped_actions=[[0.0], [0.0]], rewards=[0.0, 1.0],
        dones=[False, False], infos=[{"reward_components": {"progress": 0.0}}, {}],
    ))
    record = accumulator.finish_rollout(num_timesteps=2)
    assert record["reward_components"]["status"] == "partial"
    assert record["reward_components"]["observed_env_steps"] == 1
    assert record["reward_components"]["expected_env_steps"] == 2
    assert record["reward_components"]["per_step_reconciliation"]["status"] == "partial"


def test_episode_return_accumulator_survives_rollout_boundaries():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[0.0], clipped_actions=[0.0], rewards=[1.25], dones=[False], infos=[{}],
    ))
    first = accumulator.finish_rollout(num_timesteps=1)
    assert first["completed_episode_returns"]["count"] == 0

    accumulator.observe_step(_step(
        actions=[0.0], clipped_actions=[0.0], rewards=[-0.25], dones=[True],
        infos=[{"reason": "time_limit"}],
    ))
    second = accumulator.finish_rollout(num_timesteps=2)
    assert second["completed_episode_returns"]["count"] == 1
    assert second["completed_episode_returns"]["sum"] == pytest.approx(1.0)
    assert second["completed_episodes"][0]["reward_components"]["status"] == "missing"
    final = accumulator.finish_training(num_timesteps=2, logger_metrics=None)
    assert final["logger_metrics"] is None
    assert final["logger_metric_alignment"]["status"] == "no_numeric_metrics_available"


def test_four_dimensional_action_channels_report_contract_order_stats_and_bound_fractions():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[0.95, -0.95, 1.2, -1.0], [0.0, 1.0, -0.2, 0.5]],
        clipped_actions=[[1.0, -1.0, 1.0, -1.0], [0.0, 1.0, -0.2, 0.5]],
        rewards=[0.0, 0.1], dones=[False, False], infos=[{}, {}],
    ))
    record = accumulator.finish_rollout(num_timesteps=2)

    for key in ("sampled_policy_action_components", "actions_passed_to_vec_env_components"):
        summary = record[key]
        assert summary["status"] == "observed"
        assert summary["action_schema"] == ACTION_SCHEMA
        assert summary["component_order"] == list(ACTION_COMPONENTS)
        assert summary["expected_shape"] == "[num_env, 4]"
        assert summary["normalized_bound"] == [-1.0, 1.0]
        assert summary["absolute_value_threshold"] == 0.95

    raw = record["sampled_policy_action_components"]["components"]
    assert [item["name"] for item in raw] == list(ACTION_COMPONENTS)
    assert raw[0]["statistics"]["mean"] == pytest.approx(0.475)
    assert raw[0]["statistics"]["std"] == pytest.approx(0.475)
    assert raw[0]["absolute_value_at_least_threshold"]["count"] == 1
    assert raw[0]["absolute_value_at_least_threshold"]["fraction"] == pytest.approx(0.5)
    assert raw[0]["exact_bound"]["count"] == 0
    assert raw[0]["over_bound"]["count"] == 0
    assert raw[1]["absolute_value_at_least_threshold"]["count"] == 2
    assert raw[1]["exact_bound"]["count"] == 1
    assert raw[2]["statistics"]["min"] == pytest.approx(-0.2)
    assert raw[2]["statistics"]["max"] == pytest.approx(1.2)
    assert raw[2]["over_bound"]["count"] == 1
    assert raw[3]["exact_bound"]["count"] == 1

    clipped = record["actions_passed_to_vec_env_components"]["components"]
    assert clipped[0]["exact_bound"]["count"] == 1
    assert clipped[1]["exact_bound"]["count"] == 2
    assert clipped[2]["exact_bound"]["count"] == 1
    assert clipped[3]["exact_bound"]["count"] == 1
    assert record["sampled_policy_actions"]["count"] == 8
    assert record["actions_passed_to_vec_env"]["count"] == 8


def test_action_component_stats_keep_aggregate_for_missing_or_invalid_shape():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[0.1, 0.2], [0.3, 0.4]],
        clipped_actions=[[0.1, 0.2], [0.3, 0.4]],
        rewards=[0.0, 0.0], dones=[False, False], infos=[{}, {}],
    ))
    accumulator.observe_step(_step(
        actions=None, clipped_actions=None, rewards=[0.0, 0.0],
        dones=[False, False], infos=[{}, {}],
    ))
    record = accumulator.finish_rollout(num_timesteps=4)

    assert record["sampled_policy_actions"]["count"] == 4
    assert record["actions_passed_to_vec_env"]["count"] == 4
    sampled = record["sampled_policy_action_components"]
    assert sampled["status"] == "unavailable_invalid_shape"
    assert sampled["components"] is None
    assert sampled["invalid_shape_steps"] == 1
    assert sampled["missing_steps"] == 1
    assert sampled["invalid_shapes"] == [[2, 2]]
    assert record["actions_passed_to_vec_env_components"]["components"] is None
    assert record["actions_passed_to_vec_env_components"]["status"] == "unavailable_invalid_shape"


def test_reward_component_reconciliation_uses_float32_tolerance_and_counts_mismatches():
    accumulator = TrainingDiagnosticsAccumulator()
    callback_rewards = [0.300004, 0.75, 0.4]
    accumulator.observe_step(_step(
        actions=[[0.0] * 4 for _ in callback_rewards],
        clipped_actions=[[0.0] * 4 for _ in callback_rewards],
        rewards=callback_rewards, dones=[False] * 3,
        infos=[
            {"reward_components": {"progress": 0.1, "alive": 0.200005}},
            {"reward_components": {"progress": 0.2, "time": -0.1}},
            {},
        ],
    ))
    record = accumulator.finish_rollout(num_timesteps=3)
    check = record["reward_components"]["per_step_reconciliation"]

    assert check["absolute_tolerance"] == pytest.approx(1e-5)
    assert check["status"] == "partial_with_mismatches"
    assert check["expected_env_steps"] == 3
    assert check["compared_env_steps"] == 2
    assert check["within_tolerance_env_steps"] == 1
    assert check["mismatch_env_steps"] == 1
    assert check["unavailable_env_steps"] == 1
    assert check["unavailable_reasons"] == {"reward_components_missing": 1}
    assert check["mismatches"][0]["env_index"] == 1
    assert check["mismatches"][0]["callback_reward"] == pytest.approx(0.75)
    assert check["mismatches"][0]["reward_components_sum"] == pytest.approx(0.1)
    assert record["callback_rewards"]["sum"] == pytest.approx(sum(callback_rewards))
    json.dumps(record, allow_nan=False)


def test_episode_component_totals_accumulate_across_rollouts_and_mark_missing_values():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[0.0] * 4], clipped_actions=[[0.0] * 4], rewards=[0.2], dones=[False],
        infos=[{"reward_components": {"progress": 0.1, "time": 0.1}}],
    ))
    accumulator.finish_rollout(num_timesteps=1)

    accumulator.observe_step(_step(
        actions=[[0.0] * 4], clipped_actions=[[0.0] * 4], rewards=[0.2], dones=[False],
        infos=[{"reason": "still_running"}],
    ))
    accumulator.finish_rollout(num_timesteps=2)

    accumulator.observe_step(_step(
        actions=[[0.0] * 4], clipped_actions=[[0.0] * 4], rewards=[0.2], dones=[True],
        infos=[{"reason": "time_limit", "reward_components": {"progress": 0.2}}],
    ))
    record = accumulator.finish_rollout(num_timesteps=3)
    episode_components = record["completed_episodes"][0]["reward_components"]

    assert episode_components["status"] == "partial"
    assert episode_components["expected_env_steps"] == 3
    assert episode_components["provided_env_steps"] == 2
    assert episode_components["missing_env_steps"] == 1
    assert episode_components["components"]["progress"]["total"] is None
    assert episode_components["components"]["progress"]["observed_sum"] == pytest.approx(0.3)
    assert episode_components["components"]["progress"]["valid_steps"] == 2
    assert episode_components["components"]["progress"]["missing_steps"] == 1
    assert episode_components["components"]["time"]["total"] is None
    assert episode_components["components"]["time"]["observed_sum"] == pytest.approx(0.1)
    assert episode_components["components"]["time"]["missing_steps"] == 2
    json.dumps(record, allow_nan=False)


def test_complete_episode_component_totals_are_summed_across_rollouts():
    accumulator = TrainingDiagnosticsAccumulator()
    accumulator.observe_step(_step(
        actions=[[0.0] * 4], clipped_actions=[[0.0] * 4], rewards=[0.2], dones=[False],
        infos=[{"reward_components": {"progress": 0.1, "time": 0.1}}],
    ))
    accumulator.finish_rollout(num_timesteps=1)

    accumulator.observe_step(_step(
        actions=[[0.0] * 4], clipped_actions=[[0.0] * 4], rewards=[0.3], dones=[True],
        infos=[{"reward_components": {"progress": 0.2, "time": 0.1}}],
    ))
    record = accumulator.finish_rollout(num_timesteps=2)
    episode_components = record["completed_episodes"][0]["reward_components"]

    assert episode_components["status"] == "complete"
    assert episode_components["expected_env_steps"] == 2
    assert episode_components["missing_env_steps"] == 0
    assert episode_components["components"]["progress"]["total"] == pytest.approx(0.3)
    assert episode_components["components"]["time"]["total"] == pytest.approx(0.2)
    assert episode_components["components"]["progress"]["missing_steps"] == 0
