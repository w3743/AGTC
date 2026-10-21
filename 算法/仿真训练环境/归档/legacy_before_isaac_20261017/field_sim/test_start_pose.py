"""A sampled start must propagate to every initial sensor and policy state."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from field_sim.env import Config, FieldEnv, Scene


def config():
    return Config(size=12, plants=0, terrain_enabled=False, validate_scene=False,
                  lidar_rays=8, depth_width=2, depth_height=2, position_noise=0)


def test_randomized_start_is_used_by_scene_sensors_and_goal_observation():
    env = FieldEnv(config(), 'goto')
    start = np.array([6, 7, math.pi])
    obs, _ = env.reset(seed=7, options={'start': start, 'goal': [8, 7]})
    np.testing.assert_allclose(env.robot[:2], start[:2])
    for frame in env.frames.values():
        np.testing.assert_allclose(frame['pose'][:2], start[:2])
        np.testing.assert_allclose(frame['origin'][:2], start[:2])
    np.testing.assert_allclose(obs['state'][2:4], [-2/12, 0], atol=1e-7)
    np.testing.assert_allclose(obs['state'][4:6], [0, 1/6], atol=1e-7)
    assert env.observation_space.contains(obs)
    start[0] = 9
    assert env.robot[0] == 6
    env.close()


@pytest.mark.parametrize('start', [[0, 2, 0], [12, 2, 0], [2, 2], [2, 2, float('nan')]])
def test_invalid_start_cannot_put_vehicle_outside_field(start):
    env = FieldEnv(config())
    with pytest.raises(ValueError, match='start'):
        env.reset(options={'start': start})
    env.close()


def test_random_start_replays_with_seed_and_moves_on_later_resets():
    first, replay = FieldEnv(config()), FieldEnv(config())
    try:
        first.reset(seed=31, options={'random_start': True})
        replay.reset(seed=31, options={'random_start': True})
        np.testing.assert_array_equal(first.robot[:3], replay.robot[:3])
        previous = first.robot[:3].copy()
        first.reset(options={'random_start': True})
        assert not np.array_equal(first.robot[:3], previous)
    finally:
        first.close()
        replay.close()


def test_random_start_is_blank_of_plants_and_all_rocks_including_small_rocks():
    cfg = Config(size=20, plants=2, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=2, depth_height=2)
    rng = np.random.default_rng(19)
    scene = Scene(cfg, np.array([[6.0, 6.0], [14.0, 14.0]]), np.zeros(2, bool), rng)
    scene.terrain = SimpleNamespace(
        rock_xy=np.array([[10.0, 10.0], [4.0, 15.0]], dtype=np.float32),
        rock_radius=np.array([0.12, 0.25], dtype=np.float32),
        rock_blocking=np.array([False, True]),
    )
    clearance = math.hypot(cfg.robot_length, cfg.robot_width) / 2 + 0.05
    for _ in range(200):
        start = scene.sample_start(rng)
        assert scene.navigable(start[:2])
        assert np.all(np.linalg.norm(scene.xy - start[:2], axis=1) > scene.radius + clearance)
        assert np.all(np.linalg.norm(scene.terrain.rock_xy - start[:2], axis=1)
                      > scene.terrain.rock_radius + clearance)


def test_random_start_reports_no_available_blank_pose():
    cfg = Config(size=12, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=2, depth_height=2)
    scene = Scene(cfg, np.empty((0, 2)), np.empty(0, bool), np.random.default_rng(2))
    scene.terrain = SimpleNamespace(
        rock_xy=np.array([[6.0, 6.0]], dtype=np.float32),
        rock_radius=np.array([100.0], dtype=np.float32),
        rock_blocking=np.array([False]),
    )
    with pytest.raises(ValueError, match='no blank random start'):
        scene.sample_start(np.random.default_rng(3), max_attempts=20)


def test_random_start_updates_initial_sensor_origins_and_rejects_explicit_start_conflict():
    env = FieldEnv(config(), 'goto')
    try:
        observation, _ = env.reset(seed=41, options={'random_start': True, 'goal': [10, 10]})
        for frame in env.frames.values():
            np.testing.assert_allclose(frame['pose'][:2], env.robot[:2])
            np.testing.assert_allclose(frame['origin'][:2], env.robot[:2])
        assert env.observation_space.contains(observation)
        with pytest.raises(ValueError, match='cannot be supplied together'):
            env.reset(options={'start': [5.0, 5.0, 0.0], 'random_start': True})
    finally:
        env.close()
