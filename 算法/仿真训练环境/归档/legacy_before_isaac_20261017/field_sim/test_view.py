"""Debug geometry and camera-view payload checks; no renderer dependency."""

import numpy as np
import pytest

from field_sim.env import Config, FieldEnv, Scene


def make_two_plant_env(seed=7):
    cfg = Config(
        size=20.0,
        plants=2,
        leaves_per_plant=2,
        validate_scene=False,
        lidar_rays=8,
        depth_height=2,
        depth_width=2,
        max_detections=8,
        terrain_enabled=False,
    )
    scene = Scene(
        cfg,
        np.array([[7.0, 7.0], [12.0, 9.0]], dtype=np.float32),
        np.array([True, False]),
        np.random.default_rng(123),
    )
    env = FieldEnv(cfg)
    observation, _ = env.reset(seed=seed, options={"scene": scene})
    return env, scene, observation


def set_rocks(scene, rocks):
    terrain = scene.terrain
    assert terrain is not None
    terrain.rock_xy = np.asarray([r[:2] for r in rocks], dtype=np.float32).reshape(-1, 2)
    terrain.rock_radius = np.asarray([r[2] for r in rocks], dtype=np.float32)
    terrain.rock_height = np.asarray([r[3] for r in rocks], dtype=np.float32)
    terrain.rock_blocking = np.asarray([r[4] for r in rocks], dtype=bool)
    terrain.rock_base_z = np.asarray(terrain.height(terrain.rock_xy), dtype=np.float32)


def test_scene_debug_payload_exports_scene_leaf_geometry_without_policy_leakage():
    env, scene, observation = make_two_plant_env()

    payload = env.scene_payload()
    assert payload["schema"] == "field-geometry-debug-v1"
    assert payload["coordinate_frame"] == "field_xyz_z_up_m"
    assert payload["truth_debug_only"] is True
    assert payload["plant_model"] == "maize-proxy-v1"

    geometry = payload["leaf_geometry"]
    assert geometry["shape"] == "tapered_blade_v1"
    expected = {
        "centers": scene.centers,
        "normals": scene.normals,
        "u": scene.u,
        "v": scene.v,
        "axes": scene.axes,
        "dry": scene.leaf_dry,
        "plant": scene.leaf_plant,
    }
    assert len(scene.xy) == 2
    assert len(scene.leaf_dry) == 4
    for key, values in expected.items():
        np.testing.assert_array_equal(np.asarray(geometry[key]), values)

    for i, plant in enumerate(payload["plants"]):
        assert plant["stem_radius"] == scene.stem_radius[i]
        assert plant["root_damage_radius"] == scene.radius[i]
        assert plant["radius"] == scene.radius[i]
        assert plant["height"] == scene.height[i]

    assert set(observation) == {"lidar", "depth", "detections", "state"}
    assert env.observation_space.contains(observation)
    assert "leaf_geometry" not in observation
    assert "truth_debug_only" not in observation


def test_state_view_pose_matches_camera_basis_and_tracks_robot_and_gimbal():
    env, _, _ = make_two_plant_env()

    def check_view_pose():
        forward, _, up = env._basis()
        pose = env.state_payload()["view_pose"]
        np.testing.assert_allclose(pose["origin"], env._origin(), atol=1e-7)
        np.testing.assert_allclose(pose["origin"][2], env.cfg.camera_height, atol=1e-7)
        np.testing.assert_allclose(pose["forward"], forward, atol=1e-7)
        np.testing.assert_allclose(pose["up"], up, atol=1e-7)
        return np.asarray(pose["forward"]), np.asarray(pose["up"])

    first_forward, first_up = check_view_pose()

    env.robot[2] = -0.6
    env.gimbal[:] = [0.8, 0.65]
    second_forward, second_up = check_view_pose()
    assert not np.allclose(second_forward, first_forward)
    assert not np.allclose(second_up, first_up)


def test_default_depth_frame_uses_scaled_lightstereo_pinhole_geometry_and_masks_no_return():
    cfg = Config(plants=0, validate_scene=False, range_noise=0.0, terrain_enabled=False)
    assert cfg.depth_height == 160
    assert cfg.depth_width == 88
    env = FieldEnv(cfg)
    observation, _ = env.reset(seed=11)
    assert observation["depth"].shape == (2, 160, 88)
    assert env.observation_space.contains(observation)

    native_k = np.array([
        [393.29918354972216, 0.0, 177.54501597086588],
        [0.0, 393.29918354972216, 323.73144276936847],
        [0.0, 0.0, 1.0],
    ])
    sx, sy = cfg.depth_width / 352, cfg.depth_height / 640
    expected_k = native_k.copy()
    expected_k[0, 0] *= sx
    expected_k[0, 2] = (native_k[0, 2] + 0.5) * sx - 0.5
    expected_k[1, 1] *= sy
    expected_k[1, 2] = (native_k[1, 2] + 0.5) * sy - 0.5

    recorded = {}

    def no_return_cast(_origin, directions, max_range):
        recorded["directions"] = directions.copy()
        recorded["max_range"] = max_range
        return np.full(len(directions), max_range, np.float32), np.full(len(directions), -1, np.int32)

    env.scene.cast = no_return_cast
    frame = env._capture("depth")
    env.frames["depth"] = frame
    directions = recorded["directions"]
    assert directions.shape == (160 * 88, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(frame["intrinsics"], expected_k, atol=1e-5)

    # Increasing image X points camera-right; increasing image Y points down,
    # opposite the world-space camera-up vector.
    _, right, up = env._basis()
    row = int(round(expected_k[1, 2]))
    left = directions[row * cfg.depth_width]
    right_pixel = directions[row * cfg.depth_width + cfg.depth_width - 1]
    assert np.dot(right_pixel, right) > np.dot(left, right)
    col = int(round(expected_k[0, 2]))
    top = directions[col]
    bottom = directions[(cfg.depth_height - 1) * cfg.depth_width + col]
    assert np.dot(bottom, up) < np.dot(top, up)

    forward, _, _ = env._basis()
    expected_cast_range = cfg.depth_range / np.min(directions @ forward)
    assert recorded["max_range"] == pytest.approx(expected_cast_range, rel=1e-6)
    assert not np.any(frame["valid"])
    np.testing.assert_array_equal(frame["values"], np.full((160, 88), cfg.depth_range, np.float32))
    np.testing.assert_array_equal(env._observation()["depth"][0], np.ones((160, 88), np.float32))
    np.testing.assert_array_equal(env._observation()["depth"][1], np.zeros((160, 88), np.float32))


def test_depth_values_are_camera_z_not_slant_range_and_ground_sentinel_is_a_hit():
    cfg = Config(size=20, plants=0, validate_scene=False, depth_height=9,
                 depth_width=11, depth_range=3.5, range_noise=0.0,
                 terrain_enabled=False)
    env = FieldEnv(cfg)
    env.reset(seed=13)
    recorded = {}

    def constant_camera_z_plane(_origin, directions, max_range):
        forward, _, _ = env._basis()
        cosines = directions @ forward
        slant_ranges = 3.0 / cosines
        recorded["directions"] = directions.copy()
        recorded["ranges"] = slant_ranges
        recorded["max_range"] = max_range
        # -1 is also Scene.cast's ground sentinel. A range before the maximum
        # ray distance must still count as a valid ground return.
        return slant_ranges.astype(np.float32), np.full(len(directions), -1, np.int32)

    env.scene.cast = constant_camera_z_plane
    frame = env._capture("depth")
    forward, _, _ = env._basis()
    cosines = recorded["directions"] @ forward
    np.testing.assert_allclose(recorded["max_range"], cfg.depth_range / cosines.min(), rtol=1e-6)
    assert np.ptp(recorded["ranges"]) > 0.1  # Off-axis slant distance grows.
    assert np.all(frame["valid"])  # But all rays intersect the same Z=3 m plane.
    np.testing.assert_allclose(frame["values"], 3.0, atol=1e-6)


def test_wave_ground_changes_lidar_and_depth_returns():
    common = dict(size=20.0, plants=0, validate_scene=False, depth_height=5,
                  depth_width=5, depth_range=5.0, lidar_rays=8,
                  range_noise=0.0, lidar_relative_error=0.0,
                  terrain_amplitude=0.03, rock_density=0.0)
    rough_cfg = Config(**common, terrain_enabled=True)
    rough_scene = Scene(rough_cfg, np.empty((0, 2), np.float32),
                        np.empty(0, bool), np.random.default_rng(10))
    terrain = rough_scene.terrain
    terrain.wave_vectors4x2[:] = [[0.2, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    terrain.wave_amplitudes4[:] = [0.03, 0.0, 0.0, 0.0]
    terrain.wave_phases4[:] = 0.0
    rough = FieldEnv(rough_cfg)
    rough.reset(seed=2, options={"scene": rough_scene})
    rough.robot[:3] = [10.0, 10.0, 0.0]
    rough.gimbal[1] = -0.15
    rough_lidar = rough._capture("lidar")
    rough_depth = rough._capture("depth")

    flat_cfg = Config(**common, terrain_enabled=False)
    flat_scene = Scene(flat_cfg, np.empty((0, 2), np.float32),
                       np.empty(0, bool), np.random.default_rng(10))
    flat = FieldEnv(flat_cfg)
    flat.reset(seed=2, options={"scene": flat_scene})
    flat.robot[:3] = [10.0, 10.0, 0.0]
    flat.gimbal[1] = -0.15
    flat_lidar = flat._capture("lidar")
    flat_depth = flat._capture("depth")

    assert rough_lidar["hit_kind"][4] == 1
    assert flat_lidar["hit_kind"][4] == 1
    assert abs(float(rough_lidar["ranges"][4] - flat_lidar["ranges"][4])) > 0.01
    assert rough_depth["valid"][2, 2] and flat_depth["valid"][2, 2]
    assert abs(float(rough_depth["values"][2, 2] - flat_depth["values"][2, 2])) > 0.01


def test_rock_hits_lidar_and_depth_and_occludes_camera_target():
    cfg = Config(
        size=20.0, plants=1, leaves_per_plant=1, validate_scene=False,
        terrain_enabled=True, terrain_amplitude=0.0, rock_density=0.0,
        camera_height=0.2, lidar_rays=8, lidar_range=8.0,
        lidar_relative_error=0.0, range_noise=0.0,
        depth_height=5, depth_width=5, depth_range=5.0,
        camera_range=8.0, detection_drop=0.0, class_error=0.0,
        max_detections=8,
    )
    scene = Scene(cfg, np.array([[6.0, 2.0]], np.float32), np.array([True]),
                  np.random.default_rng(4))
    set_rocks(scene, [(4.0, 2.0, 0.3, 0.5, False)])
    scene.centers[0] = [6.0, 2.0, 0.2]
    scene.normals[0] = [1.0, 0.0, 0.0]
    scene.u[0] = [0.0, 1.0, 0.0]
    scene.v[0] = [0.0, 0.0, 1.0]
    scene.axes[0] = [0.15, 0.15]

    env = FieldEnv(cfg)
    observation, _ = env.reset(seed=9, options={"scene": scene})
    env.robot[:3] = [2.0, 2.0, 0.0]
    env.gimbal[:] = [0.0, 0.0]
    lidar = env._capture("lidar")
    depth = env._capture("depth")
    camera = env._capture("camera")
    env.frames.update(lidar=lidar, depth=depth, camera=camera)

    assert lidar["valid"][4] and lidar["hit_kind"][4] == 3
    assert lidar["ranges"][4] == pytest.approx(2.0 - 0.3 * np.sqrt(1 - (0.2 / 0.5) ** 2), abs=1e-3)
    assert depth["valid"][2, 2]
    assert depth["values"][2, 2] < cfg.depth_range
    assert camera["detections"] == []

    hit_distance, hit_id = scene.cast(np.array([2.0, 2.0, 0.2], np.float32),
                                      np.array([[1.0, 0.0, 0.0]], np.float32), 8.0)
    assert hit_distance[0] < 3.0 and hit_id[0] <= -1_000_000
    assert env.observation_space.contains(observation)
    assert set(observation) == {"lidar", "depth", "detections", "state"}
    payload = env.scene_payload()["terrain"]
    assert np.asarray(payload["wave_vectors4x2"]).shape == (4, 2)
    assert len(payload["rocks"]) == 1
    assert payload["rocks"][0]["blocking"] is False
    assert payload["rocks"][0]["z"] == pytest.approx(0.0)
