"""Focused contracts for reset-time navigation curriculum; no training or Gazebo."""
from __future__ import annotations

import math

import numpy as np
import pytest

from field_sim.env import Config, FieldEnv
from field_sim.policy import make_model
from simulation.improve_field_policy import (NavigationCurriculum, _make_or_load_model,
                                             _make_training_env, _parse_args,
                                             _validation_score, config_for_stage)


def _small_env_config():
    return Config(size=100.0, plants=0, terrain_enabled=False,
                  validate_scene=False, lidar_rays=8, depth_width=4,
                  depth_height=4, max_detections=2, max_time=20.0)


def test_stage_config_changes_scene_only_and_keeps_policy_contract():
    easy = config_for_stage("easy")
    turns = config_for_stage("turns")
    field = config_for_stage("field")

    for cfg in (easy, turns, field):
        env = FieldEnv(cfg, mode="goto")
        assert env.observation_space["lidar"].shape == (2, 400)
        assert env.observation_space["depth"].shape == (2, 160, 88)
        assert env.observation_space["state"].shape == (44,)
        assert env.action_space.shape == (4,)
        assert cfg.max_time == 20.0
        assert cfg.progress_weight == 1.0
        assert cfg.time_penalty == 0.02
        env.close()

    assert (easy.plants, easy.terrain_enabled, easy.validate_scene) == (0, False, False)
    assert (turns.plants, turns.terrain_enabled, turns.validate_scene) == (0, False, False)
    assert (field.plants, field.terrain_enabled, field.validate_scene) == (400, True, True)
    assert easy.goal_tolerance == field.goal_tolerance == 0.4
    assert easy.goal_hold == field.goal_hold == 0.3
    assert easy.max_speed == field.max_speed


@pytest.mark.parametrize(("stage", "low", "high", "angle"), [
    ("easy", 0.8, 2.5, math.pi / 4),
    ("turns", 1.0, 5.0, math.pi),
])
def test_curriculum_samples_goal_in_phase_range_without_changing_observation(stage, low, high, angle):
    env = NavigationCurriculum(FieldEnv(_small_env_config()), stage=stage, seed=91)
    try:
        for reset_seed in range(8):
            observation, _ = env.reset(seed=reset_seed)
            task = env.last_task
            assert set(observation) == {"lidar", "depth", "detections", "state"}
            assert env.observation_space.contains(observation)
            assert task["stage"] == stage
            start = np.asarray(task["start"])
            goal = np.asarray(task["goal"])
            distance = float(np.linalg.norm(goal - start[:2]))
            heading = abs(task["heading_offset_rad"])
            assert low <= distance <= high
            assert heading <= angle + 1e-7
            assert env.unwrapped.mode == "goto"
            assert env.unwrapped.scene.navigable(goal)
            if not np.allclose(start[:2], [2.0, 2.0]):
                assert 8.0 <= start[0] <= 92.0
                assert 8.0 <= start[1] <= 92.0
    finally:
        env.close()


def test_curriculum_accepts_short_goal_range_and_heading_limit():
    env = NavigationCurriculum(FieldEnv(_small_env_config()), stage="easy", seed=123,
                               goal_min=0.45, goal_max=1.05, goal_angle_deg=10.0)
    try:
        for reset_seed in range(6):
            observation, _ = env.reset(seed=reset_seed)
            task = env.last_task
            assert env.observation_space.contains(observation)
            assert 0.45 <= task["distance_m"] <= 1.05
            assert abs(task["heading_offset_rad"]) <= math.radians(10.0) + 1e-7
            assert task["goal_min_m"] == pytest.approx(0.45)
            assert task["goal_max_m"] == pytest.approx(1.05)
            assert task["goal_angle_deg"] == pytest.approx(10.0)
    finally:
        env.close()


def test_field_goal_sampler_keeps_eighty_twenty_route_intent_and_no_collision_goal():
    env = NavigationCurriculum(FieldEnv(_small_env_config()), stage="field", seed=18)
    route_types = []
    try:
        for reset_seed in range(40):
            observation, _ = env.reset(seed=reset_seed)
            assert env.observation_space.contains(observation)
            task = env.last_task
            route_types.append(task["route_type"])
            assert 1.0 <= task["distance_m"] <= 5.0
            assert env.unwrapped.scene.navigable(np.asarray(task["goal"]))
        clear_fraction = sum(kind == "clear_straight" for kind in route_types) / len(route_types)
        assert 0.6 <= clear_fraction <= 0.95
        assert all(kind in {"clear_straight", "general_navigable",
                            "general_navigable_fallback"} for kind in route_types)
    finally:
        env.close()


def test_validation_score_prioritizes_safe_arrival_then_progress_and_safety():
    safer = [
        {"complete": True, "safe_success": True, "progress_fraction": 0.9,
         "damage": 0, "collisions": 0},
        {"complete": True, "safe_success": False, "progress_fraction": 0.8,
         "damage": 0, "collisions": 0},
    ]
    more_progress = [
        {"complete": True, "safe_success": False, "progress_fraction": 1.0,
         "damage": 0, "collisions": 0},
        {"complete": True, "safe_success": False, "progress_fraction": 1.0,
         "damage": 0, "collisions": 0},
    ]
    assert _validation_score(safer) > _validation_score(more_progress)
    assert _validation_score([*safer, {"complete": False}]) is None


def test_cli_defaults_are_small_easy_curriculum_with_four_field_sim_envs():
    args = _parse_args([])
    assert args.stage == "easy"
    assert args.steps == 32768
    assert args.chunk_steps == 8192
    assert args.wall_seconds == 240.0
    assert args.n_env == 4
    assert args.n_steps == 128
    assert args.batch_size == 128
    assert args.n_epochs == 3
    assert args.goal_gain == 12.5
    assert args.gamma == 0.995
    assert args.use_sde is False
    assert args.sde_sample_freq == -1
    assert args.log_std_init == 0.0
    assert args.squash_output is False
    assert args.target_kl is None
    assert args.goal_min is None
    assert args.goal_max is None
    assert args.goal_angle_deg is None
    assert args.episode_steps is None
    assert args.device == "cpu"
    assert args.vec_backend == "dummy"
    assert args.distance_cost == pytest.approx(0.0)
    assert args.stop_on_two_zero_success_negative_progress is False
    assert args.output.is_absolute()
    assert args.checkpoint is None

    sde_args = _parse_args(["--use-sde", "--sde-sample-freq", "32",
                            "--log-std-init", "-1", "--squash-output",
                            "--target-kl", "0.03"])
    assert sde_args.use_sde is True
    assert sde_args.sde_sample_freq == 32
    assert sde_args.log_std_init == -1.0
    assert sde_args.squash_output is True
    assert sde_args.target_kl == pytest.approx(0.03)

    range_args = _parse_args(["--goal-min", "0.45", "--goal-max", "1.05",
                              "--goal-angle-deg", "10", "--episode-steps", "160"])
    assert range_args.goal_min == pytest.approx(0.45)
    assert range_args.goal_max == pytest.approx(1.05)
    assert range_args.goal_angle_deg == pytest.approx(10.0)
    assert range_args.episode_steps == 160
    custom_backend = _parse_args(["--device", "auto", "--vec-backend", "subproc",
                                  "--distance-cost", "1.0",
                                  "--stop-on-two-zero-success-negative-progress"])
    assert custom_backend.device == "auto"
    assert custom_backend.vec_backend == "subproc"
    assert custom_backend.distance_cost == pytest.approx(1.0)
    assert custom_backend.stop_on_two_zero_success_negative_progress is True
    with pytest.raises(SystemExit):
        _parse_args(["--distance-cost", "-0.01"])
    with pytest.raises(SystemExit):
        _parse_args(["--goal-min", "1.2", "--goal-max", "1.0"])
    with pytest.raises(SystemExit):
        _parse_args(["--episode-steps", "0"])


def test_four_env_vector_reset_keeps_original_observation_and_action_shapes():
    vec_env = _make_training_env(_small_env_config(), n_env=4, seed=7, stage="easy")
    try:
        observation = vec_env.reset()
        assert vec_env.num_envs == 4
        assert observation["lidar"].shape == (4, 2, 8)
        assert observation["depth"].shape == (4, 2, 4, 4)
        assert observation["detections"].shape == (4, 2, 10)
        assert observation["state"].shape == (4, 44)
        assert vec_env.action_space.shape == (4,)
        assert all(np.isfinite(value).all() for value in observation.values())
    finally:
        vec_env.close()


def test_time_limit_truncates_training_episode_without_changing_config_timeout():
    cfg = _small_env_config()
    vec_env = _make_training_env(cfg, n_env=1, seed=12, stage="easy",
                                 goal_min=0.45, goal_max=1.05,
                                 goal_angle_deg=10.0, episode_steps=3)
    try:
        observation = vec_env.reset()
        assert vec_env.observation_space.contains({key: value[0] for key, value in observation.items()})
        assert vec_env.envs[0].unwrapped.cfg.max_time == pytest.approx(20.0)
        action = np.zeros((1, 4), dtype=np.float32)
        for index in range(3):
            observation, rewards, dones, infos = vec_env.step(action)
            assert np.isfinite(rewards).all()
            if index < 2:
                assert not dones[0]
            else:
                assert dones[0]
                assert infos[0].get("TimeLimit.truncated") is True
                # DummyVecEnv has already reset for the next episode.
                assert vec_env.envs[0].unwrapped.time == pytest.approx(0.0)
    finally:
        vec_env.close()


def test_distance_cost_changes_only_training_reward_and_uses_dt_scale():
    cfg = _small_env_config()
    plain = _make_training_env(cfg, n_env=1, seed=31, stage="easy",
                               goal_min=0.45, goal_max=1.05,
                               goal_angle_deg=10.0, episode_steps=160)
    dense = _make_training_env(cfg, n_env=1, seed=31, stage="easy",
                               goal_min=0.45, goal_max=1.05,
                               goal_angle_deg=10.0, episode_steps=160,
                               distance_cost=2.0)
    try:
        plain.seed(7001)
        dense.seed(7001)
        obs_plain = plain.reset()
        obs_dense = dense.reset()
        assert all(np.array_equal(obs_plain[key], obs_dense[key]) for key in obs_plain)
        action = np.zeros((1, 4), dtype=np.float32)
        next_plain, reward_plain, done_plain, info_plain = plain.step(action)
        next_dense, reward_dense, done_dense, info_dense = dense.step(action)
        assert not done_plain[0] and not done_dense[0]
        assert all(np.array_equal(next_plain[key], next_dense[key]) for key in next_plain)
        assert info_plain[0]["distance"] == pytest.approx(info_dense[0]["distance"])
        expected = 2.0 * info_plain[0]["distance"] * cfg.dt
        assert reward_plain[0] - reward_dense[0] == pytest.approx(expected, abs=1e-6)
        assert plain.envs[0].unwrapped.cfg.max_time == pytest.approx(20.0)
        assert dense.envs[0].unwrapped.cfg.max_time == pytest.approx(20.0)
    finally:
        plain.close()
        dense.close()


def test_subproc_vec_backend_spawn_smoke_preserves_observation_contract():
    vec_env = _make_training_env(_small_env_config(), n_env=1, seed=21, stage="easy",
                                 goal_min=0.45, goal_max=1.05,
                                 goal_angle_deg=10.0, episode_steps=4,
                                 vec_backend="subproc")
    try:
        assert type(vec_env).__name__ == "SubprocVecEnv"
        observation = vec_env.reset()
        assert observation["lidar"].shape == (1, 2, 8)
        assert observation["depth"].shape == (1, 2, 4, 4)
        assert observation["detections"].shape == (1, 2, 10)
        assert observation["state"].shape == (1, 44)
        observation, rewards, dones, infos = vec_env.step(np.zeros((1, 4), np.float32))
        assert np.isfinite(rewards).all()
        assert not dones[0]
        assert all(np.isfinite(value).all() for value in observation.values())
    finally:
        vec_env.close()


def test_resume_load_keeps_checkpoint_goal_gain_instead_of_cli_default(tmp_path):
    vec_env = _make_training_env(_small_env_config(), n_env=1, seed=11, stage="easy")
    try:
        original = make_model(vec_env, seed=11, n_steps=4, batch_size=4,
                              n_epochs=1, goal_gain=3.5, n_env=1,
                              use_sde=True, sde_sample_freq=32,
                              log_std_init=-1.0, squash_output=True,
                              target_kl=0.03)
        checkpoint = tmp_path / "gain_3_5.zip"
        original.save(checkpoint)

        args = _parse_args(["--checkpoint", str(checkpoint), "--n-env", "1",
                            "--goal-gain", "12.5"])
        resumed = _make_or_load_model(vec_env, args)
        assert str(original.device) == "cpu"
        assert str(resumed.device) == "cpu"
        assert resumed.policy.features_extractor.goal_gain == pytest.approx(3.5)
        assert resumed.use_sde is True
        assert resumed.sde_sample_freq == 32
        assert resumed.target_kl == pytest.approx(0.03)
        assert resumed.policy_kwargs["log_std_init"] == pytest.approx(-1.0)
        assert resumed.policy_kwargs["squash_output"] is True
        assert args.goal_gain == pytest.approx(12.5)
        assert args.use_sde is False
    finally:
        vec_env.close()

