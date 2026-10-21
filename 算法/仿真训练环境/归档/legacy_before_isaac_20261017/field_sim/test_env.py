"""Low-cost behavioral checks for the array-based field environment."""

import math
import threading
import time

import numpy as np
import pytest

from field_sim.env import Config, FieldEnv, Scene


def small_config(**overrides):
    values = {
        "size": 20.0,
        "cell": 0.5,
        "plants": 0,
        "leaves_per_plant": 1,
        "validate_scene": False,
        "lidar_rays": 8,
        "lidar_relative_error": 0.0,
        "depth_height": 2,
        "depth_width": 2,
        "max_detections": 8,
        "max_time": 60.0,
        "detection_drop": 0.0,
        "class_error": 0.0,
        "terrain_enabled": False,
    }
    values.update(overrides)
    return Config(**values)


def make_scene(cfg, xy=(), dry=(), seed=1):
    positions = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    targets = np.asarray(dry, dtype=bool).reshape(-1)
    return Scene(cfg, positions, targets, np.random.default_rng(seed))


def make_env(cfg=None, *, mode="search", scene=None, goal=None, seed=7):
    cfg = cfg or small_config()
    env = FieldEnv(cfg, mode=mode)
    options = {}
    if scene is not None:
        options["scene"] = scene
    if goal is not None:
        options["goal"] = goal
    observation, info = env.reset(seed=seed, options=options)
    return env, observation, info


def set_rocks(scene, rocks):
    """Install deterministic rock geometry for focused sensor/motion checks."""
    terrain = scene.terrain
    assert terrain is not None
    terrain.rock_xy = np.asarray([r[:2] for r in rocks], dtype=np.float32).reshape(-1, 2)
    terrain.rock_radius = np.asarray([r[2] for r in rocks], dtype=np.float32)
    terrain.rock_height = np.asarray([r[3] for r in rocks], dtype=np.float32)
    terrain.rock_blocking = np.asarray([r[4] for r in rocks], dtype=bool)
    terrain.rock_base_z = np.asarray(terrain.height(terrain.rock_xy), dtype=np.float32)


def test_body_pose_cache_exact_key_and_explicit_scene_and_terrain_invalidation(monkeypatch):
    cfg = small_config(terrain_enabled=True, terrain_amplitude=0.0, rock_density=0.0)
    scene = make_scene(cfg)
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[:3] = [10.0, 10.0, 0.0]
    env.clear_body_pose_cache()
    expected = env._body_pose()  # Baseline with an empty cache.

    calls = 0
    original = scene.terrain.surface_height

    def counted_surface_height(points):
        nonlocal calls
        calls += 1
        return original(points)

    monkeypatch.setattr(scene.terrain, "surface_height", counted_surface_height)
    env.clear_body_pose_cache()
    np.testing.assert_allclose(env._body_pose(), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(env._body_pose(), expected, rtol=0.0, atol=0.0)
    assert calls == 2  # Two track rows once; the repeated pose is cached.

    env.robot[0] += 0.1
    env._body_pose()
    assert calls == 4  # Position change.
    env.robot[2] = 0.1
    env._body_pose()
    assert calls == 6  # Heading change.
    cfg.robot_length += 0.1
    env._body_pose()
    assert calls == 8  # Footprint change.

    replacement_scene = make_scene(cfg, seed=2)
    replacement_scene.terrain = scene.terrain
    env.scene = replacement_scene
    env._body_pose()
    assert calls == 10  # Scene identity change, even with the same terrain object.

    replacement_terrain = make_scene(cfg, seed=3).terrain
    env.scene.terrain = replacement_terrain
    replacement_original = replacement_terrain.surface_height

    def counted_replacement_height(points):
        nonlocal calls
        calls += 1
        return replacement_original(points)

    monkeypatch.setattr(replacement_terrain, "surface_height", counted_replacement_height)
    env._body_pose()
    assert calls == 12  # Terrain identity change within the same scene.


def test_body_pose_cache_clear_keeps_small_rock_bump_semantics():
    cfg = small_config(terrain_enabled=True, terrain_amplitude=0.0, rock_density=0.0)
    scene = make_scene(cfg)
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[:3] = [10.0, 10.0, 0.0]
    env.clear_body_pose_cache()
    flat_pose = env._body_pose()
    assert flat_pose[0] == pytest.approx(0.0)

    set_rocks(scene, [(10.4, 10.15, 0.08, 0.2, False)])
    env.clear_body_pose_cache()
    bump_pose = env._body_pose()
    assert bump_pose[0] > flat_pose[0]
    assert bump_pose[1] > flat_pose[1]


def test_seeded_reset_is_reproducible_and_observation_contains_masks_without_truth_ids():
    cfg = small_config(size=20.0, plants=8, dry_fraction=0.5)
    first, obs1, _ = make_env(cfg, seed=123)
    second, obs2, _ = make_env(cfg, seed=123)

    np.testing.assert_array_equal(first.scene.xy, second.scene.xy)
    np.testing.assert_array_equal(first.scene.targets, second.scene.targets)
    np.testing.assert_array_equal(first.scene.centers, second.scene.centers)
    assert obs1.keys() == {"lidar", "depth", "detections", "state"}
    assert first.observation_space.contains(obs1)
    assert second.observation_space.contains(obs2)
    for key in obs1:
        np.testing.assert_array_equal(obs1[key], obs2[key])

    assert obs1["lidar"].shape == (2, cfg.lidar_rays)
    assert obs1["depth"].shape == (2, cfg.depth_height, cfg.depth_width)
    assert set(np.unique(obs1["lidar"][1])).issubset({0.0, 1.0})
    assert set(np.unique(obs1["depth"][1])).issubset({0.0, 1.0})
    assert obs1["detections"].shape == (cfg.max_detections, 10)
    forbidden = {"truth", "targets", "target_ids", "plant_ids", "leaf_ids", "ids"}
    assert not (forbidden & set(obs1))


def test_four_continuous_actions_obey_vehicle_and_gimbal_limits():
    cfg = small_config(max_time=100.0)
    env, _, _ = make_env(cfg)

    assert env.action_space.shape == (4,)
    np.testing.assert_array_equal(env.action_space.low, np.full(4, -1.0, np.float32))
    np.testing.assert_array_equal(env.action_space.high, np.full(4, 1.0, np.float32))
    assert cfg.robot_length == pytest.approx(0.8)
    assert cfg.robot_width == pytest.approx(0.3)
    assert cfg.max_speed == pytest.approx(15.0 / 3.6)
    assert cfg.camera_height == pytest.approx(0.4)
    assert cfg.pitch_min == pytest.approx(math.radians(-25))
    assert cfg.pitch_max == pytest.approx(math.radians(60))

    assert env.gimbal[1] == pytest.approx(0.0)
    env.step([2.0, 2.0, 0.0, 2.0])  # Finite out-of-range commands are clipped.
    assert env.robot[3] == pytest.approx(cfg.acceleration * cfg.dt)
    assert env.robot[4] == pytest.approx(cfg.angular_acceleration * cfg.dt)
    assert env.gimbal[1] == pytest.approx(cfg.gimbal_pitch_rate * cfg.dt)

    for _ in range(60):
        env.step([1.0, 0.0, 0.0, 1.0])
    assert env.robot[3] == pytest.approx(cfg.max_speed)
    assert env.robot[3] <= 15.0 / 3.6 + 1e-6
    assert env.gimbal[1] == pytest.approx(cfg.pitch_max)

    for _ in range(30):
        env.step([0.0, 0.0, 0.0, -1.0])
    assert env.gimbal[1] == pytest.approx(cfg.pitch_min)


def test_step_motion_and_finish_hooks_keep_shared_step_pipeline():
    class HookRecordingEnv(FieldEnv):
        def _advance_motion(self, action):
            self.motion_action = action.copy()
            self.motion_result = super()._advance_motion(action)
            return self.motion_result

        def _finish_step(self, action, before, old_distance, rock_collision, boundary):
            self.finish_arguments = (action.copy(), before.copy(), old_distance,
                                     rock_collision, boundary)
            return super()._finish_step(action, before, old_distance,
                                        rock_collision, boundary)

    cfg = small_config()
    env = HookRecordingEnv(cfg, mode="goto")
    env.reset(seed=71, options={"goal": [18.0, 18.0]})
    before = env.robot.copy()
    clipped_action = np.array([1.0, -1.0, 0.5, 1.0], np.float32)

    result = env.step([3.0, -3.0, 0.5, 1.0])

    assert len(env.motion_result) == 4
    np.testing.assert_array_equal(env.motion_action, clipped_action)
    np.testing.assert_array_equal(env.finish_arguments[0], clipped_action)
    np.testing.assert_array_equal(env.finish_arguments[1], before)
    assert env.finish_arguments[2] == pytest.approx(np.linalg.norm(np.array([18.0, 18.0]) - before[:2]))
    assert env.finish_arguments[3:] == env.motion_result[2:]
    assert len(result) == 5
    assert env.time == pytest.approx(cfg.dt)
    np.testing.assert_allclose(env.last_gimbal_rates,
                               clipped_action[2:] * [cfg.gimbal_yaw_rate, cfg.gimbal_pitch_rate])


def test_lidar_rays_form_a_full_uniform_circle_in_the_yawed_pitched_scan_plane():
    cfg = Config(plants=0, validate_scene=False, terrain_enabled=False,
                 depth_height=2, depth_width=2, max_detections=8)
    assert cfg.lidar_model == "X2"
    assert cfg.lidar_rays == 400
    assert cfg.lidar_period == pytest.approx(0.1)
    env, _, _ = make_env(cfg)
    env.robot[2] = 0.37
    env.gimbal[:] = [0.71, 0.43]

    rays = env._capture("lidar")["directions"]
    forward, right, _ = env._basis()
    left = -right
    normal = np.cross(forward, left)

    assert rays.shape == (400, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(rays @ normal, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        rays[[0, 100, 200, 300]],
        np.stack((-forward, -left, forward, left)),
        atol=1e-6,
    )

    azimuth = np.mod(np.arctan2(rays @ left, rays @ forward), 2 * math.pi)
    azimuth.sort()
    circular_gaps = np.diff(np.r_[azimuth, azimuth[0] + 2 * math.pi])
    np.testing.assert_allclose(circular_gaps, 2 * math.pi / cfg.lidar_rays, atol=1e-6)


def test_default_lidar_range_and_empty_horizontal_scan_normalization():
    cfg = Config(
        plants=0,
        validate_scene=False,
        depth_height=2,
        depth_width=2,
        max_detections=8,
        range_noise=0.0,
        terrain_enabled=False,
    )
    assert cfg.lidar_range == pytest.approx(8.0)
    assert cfg.lidar_min_range == pytest.approx(0.12)
    assert cfg.lidar_relative_error == pytest.approx(0.05)
    assert cfg.lidar_model == "X2"
    assert cfg.lidar_rays == 400
    assert cfg.lidar_period == pytest.approx(0.1)
    env, _, _ = make_env(cfg)
    env.gimbal[1] = 0.0

    frame = env._capture("lidar")
    env.frames["lidar"] = frame
    observation = env._observation()

    np.testing.assert_allclose(frame["directions"][:, 2], 0.0, atol=1e-7)
    np.testing.assert_allclose(frame["ranges"], cfg.lidar_range)
    assert not np.any(frame["valid"])
    lidar_meta = env.state_payload()["lidar"]
    assert lidar_meta["max_range"] == pytest.approx(8.0)
    assert lidar_meta["min_range"] == pytest.approx(0.12)
    assert lidar_meta["relative_error"] == pytest.approx(0.05)
    assert lidar_meta["model"] == cfg.lidar_model
    assert lidar_meta["rays"] == cfg.lidar_rays
    assert lidar_meta["period"] == pytest.approx(cfg.lidar_period)
    np.testing.assert_allclose(observation["lidar"][0], 1.0)
    assert not np.any(observation["lidar"][1])


def test_explicit_12hz_scheduler_keeps_absolute_phase_at_20hz_physics():
    cfg = small_config(lidar_rays=400, lidar_period=1 / 12, dt=0.05, range_noise=0.0)
    env, observation, _ = make_env(cfg, seed=31)
    assert observation["lidar"].shape == (2, 400)
    capture = env._capture
    capture_times = [float(env.frames["lidar"]["capture"])]

    def record_lidar(kind):
        frame = capture(kind)
        if kind == "lidar":
            capture_times.append(float(frame["capture"]))
        return frame

    env._capture = record_lidar
    for _ in range(200):
        env.step(np.zeros(4, np.float32))

    assert env._serial["lidar"] == 121  # 120 scheduled samples in 10 s plus t=0.
    assert len(capture_times) == 121
    assert capture_times[0] == pytest.approx(0.0)
    assert capture_times[-1] == pytest.approx(10.0)
    intervals = np.round(np.diff(capture_times), 2)
    assert np.count_nonzero(intervals == 0.05) == 40
    assert np.count_nonzero(intervals == 0.10) == 80
    assert env._dropped["lidar"] == 0
    assert env.state_payload()["metrics"]["sensor_dropped"]["lidar"] == 0


def test_x2_scheduler_records_skipped_deadlines_without_duplicate_frame_capture():
    cfg = small_config(dt=0.25, lidar_rays=400, lidar_period=1 / 12, range_noise=0.0)
    env, _, _ = make_env(cfg, seed=32)
    capture = env._capture
    capture_times = []

    def record_lidar(kind):
        frame = capture(kind)
        if kind == "lidar":
            capture_times.append(float(frame["capture"]))
        return frame

    env._capture = record_lidar
    env.step(np.zeros(4, np.float32))

    assert capture_times == pytest.approx([0.25])
    assert env._serial["lidar"] == 2
    assert env._dropped["lidar"] == 2
    assert env.state_payload()["metrics"]["sensor_dropped"]["lidar"] == 2


def test_default_10hz_lidar_captures_100_samples_in_ten_seconds_plus_initial():
    cfg = Config(plants=0, validate_scene=False, depth_height=2, depth_width=2,
                 max_detections=8, range_noise=0.0, lidar_relative_error=0.0,
                 terrain_enabled=False)
    env, _, _ = make_env(cfg, seed=33)
    assert cfg.lidar_period == pytest.approx(0.1)
    assert cfg.lidar_rays == 400
    capture = env._capture
    capture_times = [float(env.frames["lidar"]["capture"])]

    def record_lidar(kind):
        frame = capture(kind)
        if kind == "lidar":
            capture_times.append(float(frame["capture"]))
        return frame

    env._capture = record_lidar
    for _ in range(200):
        env.step(np.zeros(4, np.float32))

    assert env._serial["lidar"] == 101
    assert len(capture_times) == 101
    assert capture_times[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(np.diff(capture_times), 0.1, atol=1e-9)
    assert env._dropped["lidar"] == 0


def test_lidar_blind_zone_is_masked_as_no_return_at_default_range():
    cfg = small_config(size=20.0, lidar_rays=400, lidar_range=8.0,
                       lidar_min_range=0.12, range_noise=0.0)
    scene = make_scene(cfg, xy=[[10.06, 10.0]], dry=[False])
    scene.height[:] = 1.0
    scene.stem_radius[:] = 0.012
    scene.centers[:] = [19.0, 19.0, 10.0]
    scene.axes[:] = 0.001
    env, _, _ = make_env(cfg, scene=scene, seed=34)
    env.robot[:3] = [10.0, 10.0, 0.0]
    frame = env._capture("lidar")
    env.frames["lidar"] = frame
    forward_index = 200

    ideal_ranges, truth_hits = scene.cast(env._origin(), frame["directions"], cfg.lidar_range)
    assert 0 < ideal_ranges[forward_index] < cfg.lidar_min_range
    assert truth_hits[forward_index] == -2
    assert not frame["valid"][forward_index]
    assert frame["hit_kind"][forward_index] == 0
    assert frame["ranges"][forward_index] == pytest.approx(cfg.lidar_range)
    assert env.state_payload()["lidar"]["ranges"][forward_index] == pytest.approx(cfg.lidar_range)


def test_lidar_relative_error_stays_within_bound_without_additive_noise():
    cfg = small_config(size=30.0, lidar_rays=400, lidar_range=8.0,
                       lidar_min_range=0.12, lidar_relative_error=0.05, range_noise=0.0)
    scene = make_scene(cfg, xy=[[15.0, 10.0], [5.0, 10.0]], dry=[False, False], seed=4)
    scene.height[:] = 1.0
    scene.stem_radius[:] = 0.012
    scene.centers[:] = [29.0, 29.0, 10.0]
    scene.axes[:] = 0.001
    env, _, _ = make_env(cfg, scene=scene, seed=35)
    env.robot[:3] = [10.0, 10.0, 0.0]
    frame = env._capture("lidar")

    for index in (0, 200):
        ideal, hit = scene.cast(env._origin(), frame["directions"][index:index + 1], cfg.lidar_range)
        assert hit[0] in (-2, -3)
        assert frame["valid"][index]
        assert abs(float(frame["ranges"][index]) - float(ideal[0])) <= cfg.lidar_relative_error * ideal[0] + 1e-6


def test_large_lidar_gaussian_noise_is_capped_and_invalid_returns_keep_range_semantics():
    cfg = small_config(lidar_rays=8, lidar_range=8.0, lidar_min_range=0.12,
                       lidar_relative_error=0.05, range_noise=0.5)
    env, _, _ = make_env(cfg, seed=36)

    def fixture_cast(_origin, directions, max_range):
        ideal = np.full(len(directions), max_range, dtype=np.float32)
        hits = np.full(len(directions), -1, dtype=np.int32)
        ideal[0], hits[0] = 1.0, 0
        ideal[1], hits[1] = 0.1201, 0
        ideal[2], hits[2] = 0.06, -2  # Plant return inside the blind zone.
        return ideal, hits

    env.scene.cast = fixture_cast
    env.np_random = np.random.default_rng(0)  # Large positive Gaussian draw.
    frame = env._capture("lidar")

    assert frame["valid"][0] and frame["hit_kind"][0] == 2
    assert frame["ranges"][0] == pytest.approx(1.05, abs=1e-6)
    assert abs(frame["ranges"][0] - 1.0) <= cfg.lidar_relative_error * 1.0 + 1e-6
    assert frame["valid"][1] and frame["ranges"][1] >= cfg.lidar_min_range
    for index in (2, 3):  # Blind-zone hit and no echo are both masked as no return.
        assert not frame["valid"][index]
        assert frame["hit_kind"][index] == 0
        assert frame["ranges"][index] == pytest.approx(cfg.lidar_range)

    def minimum_cast(_origin, directions, max_range):
        ideal = np.full(len(directions), max_range, dtype=np.float32)
        hits = np.full(len(directions), -1, dtype=np.int32)
        ideal[0], hits[0] = 0.1201, 0
        return ideal, hits

    env.scene.cast = minimum_cast
    env.np_random = np.random.default_rng(0)
    clipped = env._capture("lidar")
    assert clipped["valid"][0]
    assert clipped["ranges"][0] == pytest.approx(cfg.lidar_min_range)


@pytest.mark.parametrize(
    "overrides",
    [
        {"lidar_min_range": 0.0},
        {"lidar_min_range": float("nan")},
        {"lidar_min_range": 8.0},
        {"lidar_relative_error": -0.01},
        {"lidar_relative_error": 1.0},
        {"lidar_relative_error": float("inf")},
    ],
)
def test_config_rejects_invalid_lidar_minimum_and_relative_error(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"terrain_wavelength_min": 0.5},
        {"terrain_wavelength_min": float("nan")},
        {"terrain_wavelength_max": float("inf")},
        {"terrain_wavelength_min": 8.0, "terrain_wavelength_max": 4.0},
    ],
)
def test_config_rejects_invalid_terrain_wavelength_ranges(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_reset_starts_with_horizontal_gimbal_and_lidar_pose():
    cfg = small_config(lidar_rays=96, range_noise=0.0)
    env, _, _ = make_env(cfg)

    assert env.gimbal[1] == pytest.approx(0.0)
    assert env.frames["lidar"]["pose"][4] == pytest.approx(0.0)
    np.testing.assert_allclose(env.frames["lidar"]["directions"][:, 2], 0.0, atol=1e-7)


def test_lidar_front_back_hits_and_tilted_ground_no_return_markers():
    cfg = small_config(size=30.0, lidar_rays=96, lidar_range=20.0, range_noise=0.0)
    scene = make_scene(cfg, xy=[[15.0, 10.0], [5.0, 10.0]], dry=[False, False])
    scene.height[:] = 1.0
    scene.stem_radius[:] = 0.012
    scene.centers[:] = [29.0, 29.0, 10.0]  # Keep leaves out of the direct-ray fixture.
    scene.axes[:] = 0.001
    env, _, _ = make_env(cfg, scene=scene, seed=13)
    env.robot[:3] = [10.0, 10.0, 0.0]
    env.gimbal[0] = 0.0
    front_index, rear_index = 48, 0

    def capture_at(pitch):
        env.gimbal[1] = pitch
        frame = env._capture("lidar")
        env.frames["lidar"] = frame
        state_lidar = env.state_payload()["lidar"]
        return frame, state_lidar

    horizontal, horizontal_meta = capture_at(0.0)
    assert horizontal["hit_kind"].dtype == np.uint8
    for index in (front_index, rear_index):
        assert horizontal["valid"][index]
        assert horizontal["hit_kind"][index] == 2
        assert horizontal["ranges"][index] == pytest.approx(5.0 - 0.012, abs=1e-3)
    assert horizontal_meta["scan_pitch"] == pytest.approx(0.0)
    assert horizontal_meta["valid"] == horizontal["valid"].tolist()
    assert horizontal_meta["hit_kind"] == horizontal["hit_kind"].tolist()
    assert set(horizontal_meta["hit_kind"]).issubset({0, 1, 2})
    assert "hits" not in horizontal_meta and "plant_ids" not in horizontal_meta

    for pitch, ground_index, no_return_index in (
        (math.radians(25), rear_index, front_index),
        (math.radians(-25), front_index, rear_index),
    ):
        tilted, metadata = capture_at(pitch)
        assert tilted["valid"][ground_index]
        assert tilted["hit_kind"][ground_index] == 1
        assert tilted["ranges"][ground_index] == pytest.approx(
            cfg.camera_height / abs(tilted["directions"][ground_index, 2]), abs=1e-5
        )
        assert not tilted["valid"][no_return_index]
        assert tilted["hit_kind"][no_return_index] == 0
        assert tilted["ranges"][no_return_index] == pytest.approx(cfg.lidar_range)
        assert metadata["scan_pitch"] == pytest.approx(pitch)


def test_default_maize_proxy_has_short_similar_plants_and_distinct_stem_and_root_radii():
    cfg = Config(validate_scene=False, terrain_enabled=False)
    assert cfg.plant_height == pytest.approx(1.0)
    assert cfg.plant_height_spread == pytest.approx(0.05)
    assert cfg.stem_radius == pytest.approx(0.012)
    assert cfg.leaves_per_plant == 8

    xy = np.array([[5.0, 5.0], [8.0, 7.0], [13.0, 14.0]], np.float32)
    scene = Scene(cfg, xy, np.array([False, True, False]), np.random.default_rng(42))
    assert np.all((scene.height >= 0.95) & (scene.height <= 1.05))
    assert np.ptp(scene.height) <= 0.10
    assert np.all((scene.stem_radius >= 0.012 * 0.9) & (scene.stem_radius <= 0.012 * 1.1))
    assert np.all((scene.radius >= 0.09) & (scene.radius <= 0.15))
    assert np.all(scene.radius > scene.stem_radius * 5)
    assert scene.leaf_plant.shape == (3 * cfg.leaves_per_plant,)
    assert np.all(scene.axes[:, 0] > scene.axes[:, 1])

    levels = np.tile(np.linspace(0.14, 0.86, cfg.leaves_per_plant), len(xy))
    leaf_plants = scene.leaf_plant
    nodes = scene.centers - 0.85 * scene.axes[:, :1] * scene.u
    np.testing.assert_allclose(nodes[:, :2], scene.xy[leaf_plants], atol=1e-6)
    np.testing.assert_allclose(nodes[:, 2], scene.height[leaf_plants] * levels, atol=1e-6)
    expected_length = scene.height[leaf_plants] * (0.16 + 0.14 * np.sin(np.pi * levels))
    assert np.all(scene.axes[:, 0] >= expected_length * 0.95)
    assert np.all(scene.axes[:, 0] <= expected_length * 1.05)
    assert np.all(scene.axes[:, 1] >= scene.height[leaf_plants] * 0.025)
    assert np.all(scene.axes[:, 1] <= scene.height[leaf_plants] * 0.04)
    assert np.all(np.sum(scene.u[:-1] * scene.u[1:], axis=1)[:cfg.leaves_per_plant - 1] < -0.8)

    leaf_top = (scene.centers[:, 2] + scene.axes[:, 0] * np.abs(scene.u[:, 2])
                + scene.axes[:, 1] * np.abs(scene.v[:, 2]))
    leaf_bottom = (scene.centers[:, 2] - scene.axes[:, 0] * np.abs(scene.u[:, 2])
                   - scene.axes[:, 1] * np.abs(scene.v[:, 2]))
    leaf_height = scene.height[scene.leaf_plant]
    assert np.all(leaf_bottom >= 0)
    assert np.all(leaf_top <= leaf_height + 1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"plant_height": 0.0},
        {"plant_height": float("nan")},
        {"plant_height": float("inf")},
        {"plant_height_spread": -0.01},
        {"plant_height_spread": float("inf")},
        {"plant_height": 0.05, "plant_height_spread": 0.05},
        {"stem_radius": 0.0},
        {"stem_radius": float("nan")},
        {"stem_radius": float("inf")},
    ],
)
def test_config_rejects_invalid_maize_geometry_parameters(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_fast_swept_motion_damages_each_crossed_plant_once_and_keeps_moving():
    cfg = small_config(
        dt=0.5,
        max_speed=10.0,
        acceleration=100.0,
        max_time=5.0,
        camera_range=3.0,
        lidar_period=10.0,
        depth_period=10.0,
        camera_period=10.0,
    )
    scene = make_scene(cfg, xy=[[4.0, 2.0], [6.0, 2.0]], dry=[False, False])
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[2] = 0.0  # Straight test path from x=2 to x=7 in one step.

    _, first_reward, terminated, truncated, first_info = env.step([1.0, 0.0, 0.0, 0.0])
    assert not terminated and not truncated
    assert env.robot[0] == pytest.approx(7.0)
    assert env.damaged == {0, 1}
    assert first_info["damage"] == 2

    _, second_reward, terminated, truncated, second_info = env.step([1.0, 0.0, 0.0, 0.0])
    assert not terminated and not truncated
    assert env.robot[0] > 7.0
    assert second_info["damage"] == 2
    assert second_reward - first_reward == pytest.approx(2 * cfg.damage_penalty)


def test_goto_requires_low_speed_and_continuous_hold_near_goal():
    cfg = small_config(dt=0.05, max_speed=10.0, acceleration=4.0, goal_hold=0.3, goal_tolerance=0.4)
    env, _, _ = make_env(cfg, mode="goto", goal=[2.0, 2.0])
    env.robot[3] = 0.5
    env.robot[4] = 0.0

    for _ in range(6):
        _, _, terminated, truncated, info = env.step([0.05, 0.0, 0.0, 0.0])
        assert not terminated and not truncated
        assert info["reason"] != "goal_reached"
    assert env.robot[3] > 0.2

    for _ in range(7):
        _, _, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["reason"] == "goal_reached"


def test_cloud_ray_hits_foreground_leaf_and_cannot_see_through_it():
    cfg = small_config(camera_range=8.0, depth_range=8.0)
    scene = make_scene(cfg, xy=[[4.0, 2.0], [6.0, 2.0]], dry=[False, True])
    scene.centers[:] = [[4.0, 2.0, 0.6], [6.0, 2.0, 0.8]]
    scene.normals[:] = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    scene.u[:] = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    scene.v[:] = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    scene.axes[:] = [[0.15, 0.15], [0.2, 0.2]]
    scene.stem_radius[:] = 0.05
    scene.height[:] = 0.05

    env, _, _ = make_env(cfg, scene=scene, seed=2)
    env.robot[2] = 0.0
    origin = env._origin()
    target_ray = scene.centers[1] - origin
    target_ray /= np.linalg.norm(target_ray)
    _, hit = scene.cast(origin, target_ray[None, :], cfg.camera_range)
    assert hit[0] == 0  # The first finite, two-sided leaf patch blocks the rear target.
    captured = env._capture("camera")
    assert 0 in {d["leaf"] for d in captured["detections"]}
    assert 1 not in {d["leaf"] for d in captured["detections"]}

    # At the same camera pose, moving the foreground patch aside exposes the rear leaf.
    scene.centers[0, 1] = 3.5
    _, clear_hit = scene.cast(origin, target_ray[None, :], cfg.camera_range)
    assert clear_hit[0] == 1
    visible = env._capture("camera")
    assert 1 in {d["leaf"] for d in visible["detections"]}


def test_tapered_blade_ray_shape_and_thin_stem_use_separate_geometry():
    cfg = small_config(plants=0, camera_range=8.0, lidar_range=8.0)
    scene = make_scene(cfg, xy=[[15.0, 15.0]], dry=[False])
    scene.centers[0] = [5.0, 5.0, 1.0]
    scene.normals[0] = [0.0, 0.0, 1.0]
    scene.u[0] = [1.0, 0.0, 0.0]
    scene.v[0] = [0.0, 1.0, 0.0]
    scene.axes[0] = [1.0, 0.2]

    # In the tapered tip, y^2 <= (1-x^2)(1-|x|); just outside it misses.
    inside_origin = np.array([[5.9, 5.02, 2.0]], dtype=np.float32)
    outside_origin = np.array([[5.9, 5.04, 2.0]], dtype=np.float32)
    down = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    _, inside_hit = scene.cast(inside_origin[0], down, cfg.camera_range)
    _, outside_hit = scene.cast(outside_origin[0], down, cfg.camera_range)
    assert inside_hit[0] == 0
    assert outside_hit[0] != 0

    # A horizontal ray intersects the stem at its thin radius, not its wider
    # root/damage/navigation radius.
    scene.centers[0] = [15.0, 15.0, 1.0]
    scene.xy[0] = [5.0, 0.0]
    scene.stem_radius[0] = 0.012
    scene.radius[0] = 0.15
    distance, hit = scene.cast(np.array([0.0, 0.0, 0.5], np.float32),
                               np.array([[1.0, 0.0, 0.0]], np.float32), 8.0)
    assert hit[0] == -2
    assert distance[0] == pytest.approx(5.0 - scene.stem_radius[0], abs=1e-3)


def test_sensor_frames_keep_capture_pose_and_age_while_latency_is_pending():
    cfg = small_config(
        dt=0.1,
        max_speed=10.0,
        acceleration=100.0,
        lidar_period=10.0,
        depth_period=10.0,
        camera_period=0.2,
        camera_latency=0.3,
    )
    env, observation, _ = make_env(cfg, seed=3)
    initial_pose = env.frames["camera"]["pose"].copy()
    assert observation["state"][34] == pytest.approx(0.0)

    for _ in range(4):
        observation, _, _, _, _ = env.step([0.05, 0.0, 0.0, 0.0])
    pending = env.state_payload()
    assert pending["metrics"]["sensor_sequences"]["camera"] == 1
    assert observation["state"][34] * 2 == pytest.approx(0.4)
    np.testing.assert_array_equal(env.frames["camera"]["pose"], initial_pose)

    observation, _, _, _, _ = env.step([0.05, 0.0, 0.0, 0.0])
    delivered = env.state_payload()
    frame = env.frames["camera"]
    assert delivered["metrics"]["sensor_sequences"]["camera"] == 2
    assert frame["capture"] == pytest.approx(0.2)
    assert env.time - frame["capture"] == pytest.approx(0.3)
    assert observation["state"][34] * 2 == pytest.approx(0.3)
    expected_capture_xy = np.array([2.0, 2.0]) + 0.1 * np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    np.testing.assert_allclose(frame["pose"][:2], expected_capture_xy, atol=1e-6)
    assert np.linalg.norm(frame["pose"][:2] - env.robot[:2]) > 0.1


def test_depth_payload_exposes_received_frame_minimum_capture_time_and_pose():
    cfg = small_config(depth_period=10.0, depth_latency=0.0)
    assert cfg.depth_min_range == pytest.approx(0.3)
    env, observation, _ = make_env(cfg, seed=17)
    frame = env.frames["depth"]
    captured_pose = frame["pose"].copy()

    env.robot[:3] = [4.0, 6.0, -0.3]
    env.gimbal[:] = [0.4, 0.25]
    payload = env.state_payload()["depth"]

    assert payload["min_range"] == pytest.approx(cfg.depth_min_range)
    assert payload["max_range"] == pytest.approx(cfg.depth_range)
    assert payload["unit"] == "m"
    assert payload["convention"] == "camera_z"
    assert payload["projection"] == "pinhole"
    assert payload["resolution"] == [cfg.depth_width, cfg.depth_height]
    assert payload["shape"] == [cfg.depth_height, cfg.depth_width]
    assert payload["native_resolution"] == [352, 640]
    assert payload["calibration_id"] == "bcb9e5c82b93"
    assert payload["calibration_source"].endswith("LightStereo源码/models/portrait_geometry.npz")
    assert payload["capture_time"] == pytest.approx(frame["capture"])
    np.testing.assert_array_equal(payload["pose"], captured_pose)
    assert not np.allclose(payload["pose"], [*env.robot[:3], *env.gimbal])
    assert set(observation) == {"lidar", "depth", "detections", "state"}

    # The debug payload and policy observation describe the exact delivered frame.
    current_observation = env._observation()
    values = np.asarray(payload["values"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=np.float32)
    np.testing.assert_allclose(current_observation["depth"][0], values / cfg.depth_range)
    np.testing.assert_array_equal(current_observation["depth"][1], valid)
    assert np.isfinite(values).all()
    assert np.all(values[valid == 0] == pytest.approx(cfg.depth_range))


def test_goal_change_preserves_scene_time_and_robot_state():
    cfg = small_config()
    scene = make_scene(cfg, xy=[[5.0, 5.0]], dry=[False])
    env, _, _ = make_env(cfg, scene=scene)
    env.step([0.1, 0.0, 0.0, 0.0])
    old_scene = env.scene
    old_scene_id = env.scene_id
    old_time = env.time
    old_robot = env.robot.copy()

    env.set_goal([10.0, 10.0])

    assert env.mode == "goto"
    np.testing.assert_array_equal(env.goal, [10.0, 10.0])
    assert env.scene is old_scene
    assert env.scene_id == old_scene_id
    assert env.time == pytest.approx(old_time)
    np.testing.assert_array_equal(env.robot, old_robot)


def test_total_reward_accumulates_across_steps_and_goal_change_then_clears_on_reset():
    cfg = small_config()
    env, _, _ = make_env(cfg)

    _, first_reward, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0])
    _, second_reward, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0])
    accumulated = first_reward + second_reward
    assert env.total_reward == pytest.approx(accumulated)

    env.set_goal([10.0, 10.0])
    assert env.total_reward == pytest.approx(accumulated)

    env.reset(seed=11)
    assert env.total_reward == pytest.approx(0.0)


def test_reset_environment_applies_runtime_terrain_settings_and_clears_episode_metrics():
    from field_sim.server import Application

    cfg = small_config(size=20.0, plants=0, terrain_enabled=True,
                       terrain_amplitude=0.35, terrain_wavelength_min=4.0,
                       terrain_wavelength_max=10.0, rock_density=0.0)
    app = Application(cfg, seed=3)
    app.controller = "manual"

    class Runner:
        resets = 0

        def reset(self):
            self.resets += 1

    runner = Runner()
    app.runner = runner
    app.times.append(1.0)
    app.policy_times.append(2.0)
    app.loop_times.append(3.0)
    old_scene_id = app.env.scene_id
    old_env = app.env

    app.reset_environment({
        "seed": 9,
        "terrain_amplitude": 0.3,
        "terrain_wavelength_min": 2.0,
        "terrain_wavelength_max": 6.0,
    })

    assert app.env is not old_env
    assert app.env.cfg.terrain_amplitude == pytest.approx(0.3)
    assert app.env.cfg.terrain_wavelength_min == pytest.approx(2.0)
    assert app.env.cfg.terrain_wavelength_max == pytest.approx(6.0)
    assert app.env.scene.terrain.wavelength_range == (2.0, 6.0)
    assert app.env.scene_id == old_scene_id + 1
    assert app.seed == 9
    assert app.controller == "manual"
    assert app.runner is runner and runner.resets == 1
    assert not app.times and not app.policy_times and not app.loop_times
    assert app.scene_cache["config"]["terrain_wavelength_min"] == pytest.approx(2.0)


def test_invalid_runtime_terrain_reset_preserves_environment_scene_and_metrics():
    from field_sim.server import Application

    cfg = small_config(size=20.0, plants=0, terrain_enabled=True,
                       terrain_amplitude=0.35, rock_density=0.0)
    app = Application(cfg, seed=4)

    class Runner:
        resets = 0

        def reset(self):
            self.resets += 1

    runner = Runner()
    app.runner = runner
    app.times.append(1.0)
    app.policy_times.append(2.0)
    app.loop_times.append(3.0)
    app.error = "existing error"
    old_env = app.env
    old_scene = app.env.scene
    old_obs = app.obs
    old_scene_cache = app.scene_cache
    old_seed = app.seed
    invalid_requests = (
        {"terrain_amplitude": "not-a-number"},
        {"terrain_amplitude": float("nan")},
        {"terrain_wavelength_min": 8.0, "terrain_wavelength_max": 4.0},
    )

    for request in invalid_requests:
        with pytest.raises(ValueError):
            app.reset_environment(request)
        assert app.env is old_env and app.env.scene is old_scene
        assert app.obs is old_obs and app.scene_cache is old_scene_cache
        assert app.seed == old_seed and app.error == "existing error"
        assert list(app.times) == [1.0]
        assert list(app.policy_times) == [2.0]
        assert list(app.loop_times) == [3.0]
        assert runner.resets == 0


def test_worker_survives_controller_exception_and_stops_cleanly():
    from field_sim.server import Application

    app = Application(small_config(), seed=1)
    controller_called = threading.Event()

    class BrokenController:
        def action(self, _env):
            controller_called.set()
            raise RuntimeError("controller probe failure")

    app._demo = BrokenController()
    worker = threading.Thread(target=app.run, daemon=True)
    worker.start()
    try:
        assert controller_called.wait(timeout=2.0), "worker never invoked its controller"
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with app.lock:
                paused, error = app.paused, app.error
            if paused and error:
                break
            time.sleep(0.005)

        with app.lock:
            assert app.paused
            assert "controller probe failure" in app.error
            assert app.state()["performance"]["loop_hz"] == 0
        assert worker.is_alive(), "controller error terminated the worker loop"
    finally:
        app.stop.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive(), "worker did not exit after stop was requested"


@pytest.mark.parametrize(
    "goal",
    [
        [0.0, 0.0],
        [5.0, 5.0],
        [float("nan"), 2.0],
        [3.0],
    ],
)
def test_set_goal_rejects_outside_occupied_or_malformed_goals(goal):
    cfg = small_config()
    scene = make_scene(cfg, xy=[[5.0, 5.0]], dry=[False])
    env, _, _ = make_env(cfg, scene=scene)
    old_goal = env.goal.copy()

    with pytest.raises(ValueError):
        env.set_goal(goal)

    np.testing.assert_array_equal(env.goal, old_goal)
    assert env.mode == "search"


@pytest.mark.parametrize(
    "action",
    [
        [0.0, 0.0, 0.0],
        [float("nan"), 0.0, 0.0, 0.0],
        [0.0, float("inf"), 0.0, 0.0],
    ],
)
def test_step_rejects_wrong_shape_and_non_finite_actions(action):
    env, _, _ = make_env()
    old_time = env.time
    old_robot = env.robot.copy()

    with pytest.raises(ValueError):
        env.step(action)

    assert env.time == pytest.approx(old_time)
    np.testing.assert_array_equal(env.robot, old_robot)


def test_terrain_has_independent_reproducible_seed_and_does_not_change_plant_rng_draws():
    assert Config().terrain_enabled is True
    assert Config().terrain_amplitude == pytest.approx(0.35)
    assert Config().terrain_wavelength_min == pytest.approx(4.0)
    assert Config().terrain_wavelength_max == pytest.approx(10.0)
    cfg = small_config(size=20.0, plants=6, dry_fraction=0.5, terrain_enabled=True,
                       rock_density=0.04, large_rock_fraction=0.5)
    first = Scene.generate(cfg, np.random.default_rng(120), np.array([2.0, 2.0]), np.array([18.0, 18.0]))
    second = Scene.generate(cfg, np.random.default_rng(120), np.array([2.0, 2.0]), np.array([18.0, 18.0]))
    np.testing.assert_array_equal(first.terrain.wave_vectors4x2, second.terrain.wave_vectors4x2)
    np.testing.assert_array_equal(first.terrain.wave_amplitudes4, second.terrain.wave_amplitudes4)
    np.testing.assert_array_equal(first.terrain.rock_xy, second.terrain.rock_xy)
    np.testing.assert_array_equal(first.terrain.rock_radius, second.terrain.rock_radius)
    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.radius, second.radius)
    np.testing.assert_array_equal(first.height, second.height)
    np.testing.assert_array_equal(first.axes, second.axes)

    flat_cfg = small_config(size=20.0, plants=6, dry_fraction=0.5, terrain_enabled=False)
    flat = Scene.generate(flat_cfg, np.random.default_rng(120), np.array([2.0, 2.0]), np.array([18.0, 18.0]))
    np.testing.assert_array_equal(first.xy, flat.xy)
    np.testing.assert_array_equal(first.radius, flat.radius)
    np.testing.assert_array_equal(first.height, flat.height)
    np.testing.assert_array_equal(first.axes, flat.axes)
    np.testing.assert_array_equal(first.u, flat.u)
    np.testing.assert_array_equal(first.leaf_dry, flat.leaf_dry)

    changed_terrain_cfg = small_config(
        size=20.0, plants=6, dry_fraction=0.5, terrain_enabled=True,
        terrain_amplitude=0.2, terrain_wavelength_min=12.0,
        terrain_wavelength_max=32.0, rock_density=0.04, large_rock_fraction=0.5,
    )
    changed = Scene.generate(changed_terrain_cfg, np.random.default_rng(120),
                             np.array([2.0, 2.0]), np.array([18.0, 18.0]))
    np.testing.assert_array_equal(first.xy, changed.xy)
    np.testing.assert_array_equal(first.radius, changed.radius)
    np.testing.assert_array_equal(first.height, changed.height)
    np.testing.assert_array_equal(first.axes, changed.axes)
    np.testing.assert_array_equal(first.terrain.rock_xy, changed.terrain.rock_xy)

    env, observation, _ = make_env(cfg, seed=120)
    assert observation["state"].shape == (44,)
    assert env.observation_space.contains(observation)
    assert set(observation) == {"lidar", "depth", "detections", "state"}
    assert not ({"terrain", "rocks", "wave_vectors4x2", "targets"} & set(observation))


def test_body_tilt_uses_forward_and_left_support_signs_and_frame_keeps_capture_origin():
    cfg = small_config(size=20.0, terrain_enabled=True, terrain_amplitude=0.25,
                       rock_density=0.0, plants=0, camera_height=0.4)
    scene = make_scene(cfg)
    terrain = scene.terrain
    terrain.wave_vectors4x2[:] = [[0.0, 0.2], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    terrain.wave_amplitudes4[:] = [0.25, 0.0, 0.0, 0.0]
    terrain.wave_phases4[:] = 0.0
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[:3] = [5.0, 5.0, 0.0]

    body_z, body_pitch, body_roll = env._body_pose()
    assert body_z > 0.0
    assert body_pitch == pytest.approx(0.0, abs=1e-6)
    assert body_roll > 0.0  # Left is +Y when yaw is zero.
    first_frame = env._capture("lidar")
    np.testing.assert_allclose(first_frame["origin"], env._origin(), atol=1e-7)
    np.testing.assert_allclose(first_frame["body_attitude"], [body_roll, body_pitch], atol=1e-7)
    assert first_frame["origin"][2] > cfg.camera_height

    env.robot[1] = 13.0
    second_frame = env._capture("lidar")
    assert not np.allclose(first_frame["origin"], second_frame["origin"])
    env.frames["lidar"] = first_frame
    metadata = env.state_payload()["lidar"]
    np.testing.assert_array_equal(metadata["origin"], first_frame["origin"])
    np.testing.assert_array_equal(metadata["body_attitude"], first_frame["body_attitude"])
    assert len(env.state_payload()["robot"]) == 8


def test_default_terrain_roots_and_camera_pose_share_ground_height():
    cfg = small_config(size=20.0, terrain_enabled=True, rock_density=0.0, plants=1)
    scene = make_scene(cfg, xy=[[5.0, 5.0]], dry=[False])
    env, _, _ = make_env(cfg, scene=scene)

    assert Config().terrain_amplitude == pytest.approx(0.35)
    assert cfg.terrain_amplitude == pytest.approx(0.35)
    np.testing.assert_allclose(scene.root_z, scene.terrain.height(scene.xy), atol=1e-7)
    env.robot[:3] = [5.0, 5.0, 0.0]
    body_z, body_pitch, body_roll = env._body_pose()
    frame = env._capture("camera")
    np.testing.assert_allclose(frame["origin"], env._origin(), atol=1e-7)
    np.testing.assert_allclose(frame["body_attitude"], [body_roll, body_pitch], atol=1e-7)
    assert frame["origin"][2] == pytest.approx(body_z + cfg.camera_height * np.cos(body_pitch)
                                                * np.cos(body_roll), abs=1e-6)


def test_small_rock_is_traversable_and_raises_the_left_track_without_collision():
    cfg = small_config(size=20.0, dt=0.5, max_speed=2.0, acceleration=100.0,
                       terrain_enabled=True, terrain_amplitude=0.0, rock_density=0.0,
                       max_time=5.0)
    scene = make_scene(cfg)
    set_rocks(scene, [(3.0, 2.15, 0.09, 0.08, False)])
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[2] = 0.0

    _, _, terminated, truncated, info = env.step([1.0, 0.0, 0.0, 0.0])
    assert not terminated and not truncated
    assert env.robot[0] == pytest.approx(3.0)
    assert info["collisions"] == 0
    assert env._body_pose()[2] > 0.0
    assert env._body_pose()[1] == pytest.approx(0.0, abs=1e-6)
    assert env._body_pose()[2] > 0.0  # Positive roll: the left track is raised.
    assert env._origin()[2] > cfg.camera_height


def test_blocking_rock_stops_fast_sweep_and_vehicle_can_reverse_away():
    cfg = small_config(size=20.0, dt=0.5, max_speed=10.0, acceleration=100.0,
                       terrain_enabled=True, terrain_amplitude=0.0, rock_density=0.0,
                       collision_penalty=0.2, max_time=5.0)
    scene = make_scene(cfg)
    set_rocks(scene, [(4.0, 2.0, 0.2, 0.25, True)])
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[2] = 0.0

    _, reward, terminated, truncated, info = env.step([1.0, 0.0, 0.0, 0.0])
    assert not terminated and not truncated
    assert info["collisions"] == 1
    assert env.robot[0] < 3.5  # The full 5 m candidate sweep was intercepted.
    assert env.robot[0] > 2.0
    assert env.robot[3] == pytest.approx(0.0)
    assert env.robot[4] == pytest.approx(0.0)
    assert reward <= -cfg.collision_penalty

    previous_x = float(env.robot[0])
    _, _, _, _, reverse_info = env.step([-1.0, 0.0, 0.0, 0.0])
    assert env.robot[0] < previous_x
    assert reverse_info["collisions"] == 1


def test_in_place_rotation_is_swept_against_blocking_rock_footprint():
    cfg = small_config(size=20.0, dt=1.0, max_turn_rate=math.pi / 2,
                       angular_acceleration=10.0, terrain_enabled=True,
                       terrain_amplitude=0.0, rock_density=0.0, max_time=5.0)
    scene = make_scene(cfg)
    set_rocks(scene, [(2.35, 2.35, 0.15, 0.2, True)])
    env, _, _ = make_env(cfg, scene=scene)
    env.robot[2] = 0.0

    _, _, terminated, truncated, info = env.step([0.0, 1.0, 0.0, 0.0])
    assert not terminated and not truncated
    assert info["collisions"] == 1
    assert 0.0 < env.robot[2] < math.pi / 2
    assert env.robot[3] == pytest.approx(0.0)
    assert env.robot[4] == pytest.approx(0.0)


def test_generated_blocking_rocks_clear_start_goal_and_preserve_navigable_connectivity():
    cfg = small_config(size=20.0, cell=0.5, plants=0, terrain_enabled=True,
                       terrain_amplitude=0.03, rock_density=0.02,
                       large_rock_fraction=1.0, terrain_seed=91)
    start, goal = np.array([2.0, 2.0]), np.array([18.0, 18.0])
    scene = Scene.generate(cfg, np.random.default_rng(3), start, goal)
    report = scene.validate(start, goal)

    assert scene.terrain is not None and np.any(scene.terrain.rock_blocking)
    assert np.all(np.linalg.norm(scene.terrain.rock_xy - start, axis=1)
                  >= 1.0 + scene.terrain.rock_radius - 1e-6)
    assert np.all(np.linalg.norm(scene.terrain.rock_xy - goal, axis=1)
                  >= 1.0 + scene.terrain.rock_radius - 1e-6)
    assert scene.navigable(start) and scene.navigable(goal)
    assert report["connected"]
