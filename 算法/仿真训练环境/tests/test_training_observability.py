"""Reward accounting and lightweight pairing must preserve task semantics."""
from copy import deepcopy

import numpy as np
import pytest

from isaac_field.task import Config, FieldTask


def _task(**kwargs):
    cfg = Config(size=12, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=4, depth_height=3, **kwargs)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=4, options={"mode": "goto", "start": [3, 3, 0],
                                  "goal": [4, 3], "random_start": False})

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, cfg.lidar_range, np.float32),
                    "valid": np.zeros(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), cfg.depth_range, np.float32),
                    "valid": np.zeros((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": []}

    task.attach_sensors(capture)
    task.initialize_sensors()
    return task


@pytest.mark.parametrize("budget", [None, 5.0])
def test_component_accounting_does_not_change_progress_or_penalties(budget):
    task = _task(goal_progress_budget=budget)
    before = task.robot.copy()
    task.robot[0] += 0.2
    _, reward, terminated, truncated, info = task.finish(
        np.zeros(4, np.float32), before, 1.0, True, True,
    )
    progress = (task.cfg.progress_weight if budget is None else budget) * (1.0 - 0.8)
    expected = -task.cfg.time_penalty * task.cfg.dt - task.cfg.collision_penalty
    expected -= 0.1 * task.cfg.dt
    expected += progress
    assert reward == pytest.approx(expected, abs=1e-12)
    assert sum(info["reward_components"].values()) == pytest.approx(reward, abs=1e-12)
    assert info["reward_components"]["collision"] == -task.cfg.collision_penalty
    assert not terminated and not truncated
    # Returned infos are snapshots, not aliases to the next transition's terms.
    saved = deepcopy(info["reward_components"])
    task.reward_components["time"] = 999
    assert info["reward_components"] == saved


def test_success_hold_and_timeout_remain_distinct_in_reward_accounting():
    task = _task(max_time=1.0, goal_progress_budget=5.0)
    task.robot[:2] = task.goal
    rewards = []
    for _ in range(6):
        _, reward, terminated, truncated, info = task.finish(
            np.zeros(4, np.float32), task.robot.copy(), 0.0,
        )
        rewards.append(reward)
        assert sum(info["reward_components"].values()) == pytest.approx(reward)
    assert terminated and not truncated and info["reason"] == "goal_reached"
    assert info["reward_components"]["goal_success"] == 5.0
    assert sum(rewards) == pytest.approx(5.0 - task.cfg.time_penalty * 0.3)

    timeout_task = _task(max_time=0.05)
    _, reward, terminated, truncated, info = timeout_task.finish(
        np.zeros(4, np.float32), timeout_task.robot.copy(), 1.0,
    )
    assert not terminated and truncated
    assert info["reason"] == "time_limit"
    assert info["reward_components"]["goal_success"] == 0.0
    assert sum(info["reward_components"].values()) == pytest.approx(reward)


def test_lightweight_initial_condition_recording_without_trajectory():
    # Reuse the established orchestration fixture, including its real task-like
    # state transitions; this mode must retain pairing evidence without pixels
    # or the full trajectory payload.
    from test_isaac_field_learning import _FakeAdapter, _FakeModel
    from isaac_field import learning
    from isaac_field.action_contract import action_config
    from types import SimpleNamespace

    engine = SimpleNamespace(
        cfg=SimpleNamespace(distance_cost=0.0, mode="goto", goal_distance=None, goal_angle=3.14),
        action_config=action_config(), close=lambda: None,
    )
    report = learning.evaluate_model(
        _FakeAdapter(engine), _FakeModel(), (12, 13), record_initial_conditions=True,
    )
    for row in report["rows"]:
        assert row["trajectory_start_matches_reset"] is True
        assert len(row["reset_initial_condition"]["plant_xy_sha256"]) == 64
        assert "trajectory" not in row


@pytest.mark.parametrize("load_zip", [False, True])
def test_gae_override_updates_live_buffer_and_saved_checkpoint(tmp_path, monkeypatch, load_zip):
    from test_isaac_field_learning import _TinyRecurrentPolicyEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO
    from isaac_field import learning, training
    from isaac_field.action_contract import action_config
    from isaac_field.policy import make_model

    env = DummyVecEnv([_TinyRecurrentPolicyEnv])
    env.action_config = action_config()
    monkeypatch.setattr(training, "_sb3_wrapper", lambda value: value)
    try:
        source = make_model(env, seed=31, device="cpu", n_steps=128, batch_size=128,
                            n_epochs=1, gae_lambda=0.95, lstm_hidden_size=16,
                            net_arch={"pi": [16], "vf": [16]})
        source_path = tmp_path / "baseline.zip"
        source.save(str(source_path))
        result = learning.train_stage(env, source_path if load_zip else source, 128,
            tmp_path / "trained.zip", device="cpu", n_epochs=1, gae_lambda=0.99)
        assert result.model.gae_lambda == 0.99
        assert result.model.rollout_buffer.gae_lambda == 0.99
        restored = RecurrentPPO.load(result.checkpoint, device="cpu")
        assert restored.gae_lambda == 0.99
        assert restored.num_timesteps == 128
    finally:
        env.close()
