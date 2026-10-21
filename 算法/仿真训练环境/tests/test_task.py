"""Regression coverage for the simulator-independent Isaac field task."""
from __future__ import annotations

import math

import numpy as np
import pytest

from isaac_field.task import (
    Config,
    DAMAGE_METRIC_ROOT_PROTECTION_V1,
    DAMAGE_METRIC_STEM_CONTACT_V2,
    FAST_STOP_FAILURE_REWARD,
    FAST_STOP_SUCCESS_REWARD,
    FieldTask,
    InvalidEpisodePoseError,
    Scene,
    sample_episode_start_goal,
)
from isaac_field.feedback import FeedbackSample, FEEDBACK_VALID_SLICE
from isaac_field.navigation import has_grid_anchor
from isaac_field.assets import ROBOT_SPEC
from isaac_field.terrain import Terrain


def _small_config(**updates):
    values = dict(
        size=24.0,
        cell=0.5,
        plants=5,
        terrain_enabled=True,
        validate_scene=False,
        terrain_amplitude=0.0,
        rock_density=0.02,
        large_rock_fraction=0.0,
        lidar_rays=8,
        depth_width=4,
        depth_height=3,
    )
    values.update(updates)
    return Config(**values)


def _attach_empty_sensors(task, cfg):
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


def _step_task_to(task, xy):
    before = task.robot.copy()
    old_distance = float(np.linalg.norm(task.goal - task.robot[:2]))
    task.robot[:2] = np.asarray(xy, dtype=np.float64)
    return task.finish(np.zeros(4, dtype=np.float32), before, old_distance)


def test_prepare_reuses_fixed_terrain_and_random_spawn_clears_full_footprint():
    cfg = _small_config()
    terrain = Terrain(24.0, seed=9, ground_amplitude=0.0, rock_density=0.04,
                      large_rock_fraction=0.0)
    task = FieldTask(cfg, mode="goto")

    scene = task.prepare(seed=123, options={"terrain": terrain, "goal": [22.0, 22.0]})

    assert scene.terrain is terrain
    assert not np.allclose(task.robot[:2], [2.0, 2.0])
    clearance = math.hypot(cfg.robot_length, cfg.robot_width) / 2 + 0.05
    assert np.all(task.robot[:2] >= clearance)
    assert np.all(task.robot[:2] <= cfg.size - clearance)
    assert np.all(np.linalg.norm(scene.xy - task.robot[:2], axis=1) > scene.radius + clearance)
    if len(terrain.rock_xy):
        assert np.all(np.linalg.norm(terrain.rock_xy - task.robot[:2], axis=1)
                      > terrain.rock_radius + clearance)
        plant_rock_distance = np.linalg.norm(
            scene.xy[:, None, :] - terrain.rock_xy[None, :, :], axis=2
        )
        assert np.all(plant_rock_distance > terrain.rock_radius[None, :] + 0.15)


def test_fast_connectivity_uses_vehicle_inflated_grid_without_cast(monkeypatch):
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)
    scene = Scene.generate(cfg, np.random.default_rng(1), None, np.array([10.0, 10.0]))

    def forbidden_cast(*_args, **_kwargs):
        pytest.fail("fast connectivity must not call geometric sensor casts")

    monkeypatch.setattr(scene, "cast", forbidden_cast)
    result = scene.validate_connectivity(np.array([2.0, 2.0]), np.array([10.0, 10.0]))

    assert result["connected"] is True
    assert result["all_free_connected"] is True
    assert result["start_goal_connected"] is True
    assert result["reachable_cells"] == result["free_cells"]
    assert result["free_pose_components"] == 1
    assert "unobservable_targets" not in result
    assert "0.1 m grid" in result["scope"]


def test_row_gap_clearance_respects_wider_envelope_and_grid_resolution():
    cfg = _small_config(size=10.0, plants=6, terrain_enabled=False, validate_scene=False,
                        row_spacing=0.8, plant_spacing=0.3)
    xy = np.array([[x, y] for y in (4.0, 4.8) for x in (4.7, 5.0, 5.3)])
    scene = Scene(cfg, xy, np.zeros(len(xy), dtype=bool), np.random.default_rng(11))
    scene.radius = np.full(len(xy), 0.12, dtype=np.float32)

    assert scene.pose_clear([5.0, 4.4], 0.0)
    assert not scene.pose_clear([5.0, 4.4], math.pi / 2)
    result = scene.validate_connectivity(np.array([1.0, 1.0]), np.array([9.0, 9.0]))

    assert result["connected"] is True
    assert result["all_free_connected"] is True
    assert result["start_goal_connected"] is True
    assert result["free_pose_components"] == 1
    # The continuous center pose clears, but the 0.1 m navigation grid has no
    # center in the remaining 0.08 m aisle once plant and margin dilation apply.
    assert not has_grid_anchor(scene, [5.0, 4.4])
    with pytest.raises(InvalidEpisodePoseError, match="explicit start footprint.*requested yaw"):
        sample_episode_start_goal(
            scene, np.random.default_rng(12),
            start=np.array([5.0, 4.4, math.pi / 2]), goal=np.array([9.0, 9.0]),
            random_start=False,
        )
    with pytest.raises(InvalidEpisodePoseError, match="explicit start footprint.*requested yaw"):
        FieldTask(cfg, mode="goto").prepare(
            seed=12,
            options={"scene": scene, "start": [5.0, 4.4, math.pi / 2],
                     "goal": [9.0, 9.0], "random_start": False},
        )


def test_fast_connectivity_rejects_enclosed_pose_island():
    cfg = _small_config(size=10.0, plants=32, terrain_enabled=False, validate_scene=False)
    xy = np.array([[5.0 + 1.5 * math.cos(a), 5.0 + 1.5 * math.sin(a)]
                   for a in np.linspace(0.0, 2 * math.pi, 32, endpoint=False)])
    scene = Scene(cfg, xy, np.zeros(len(xy), dtype=bool), np.random.default_rng(9))
    scene.radius = np.full(len(xy), 0.22, dtype=np.float32)

    result = scene.validate_connectivity(np.array([5.0, 5.0]), np.array([1.0, 1.0]))

    assert result["start_goal_connected"] is False
    assert result["all_free_connected"] is False
    assert result["connected"] is False
    assert result["free_pose_components"] > 1


def test_seed178_unanchored_random_start_retries_within_the_fixed_scene():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(cfg, np.random.default_rng(2), None, np.array([10.0, 10.0]))
    first = scene.sample_start(np.random.default_rng(178))

    # Seed 178's first continuous sample is clear, but its floor-indexed 0.1 m
    # navigation cell has no safe center near the field boundary.
    assert first.tolist() == pytest.approx([0.8169629475706089, 0.47642173739899957,
                                            1.638827950435279])
    assert not has_grid_anchor(scene, first[:2])

    task = FieldTask(cfg, mode="goto")
    prepared = task.prepare(seed=178, options={"scene": scene, "goal": [6.0, 6.0]})

    assert prepared is scene
    assert has_grid_anchor(scene, task.robot[:2])
    assert task.validation["connected"] is True
    assert task.validation["start_goal_connected"] is True


def test_explicit_unanchored_start_and_goal_are_rejected_clearly():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(cfg, np.random.default_rng(2), None, np.array([10.0, 10.0]))
    bad = np.array([0.8169629475706089, 0.47642173739899957, 1.638827950435279])

    with pytest.raises(InvalidEpisodePoseError, match="explicit start.*grid anchor"):
        FieldTask(cfg, mode="goto").prepare(
            seed=178,
            options={"scene": scene, "start": bad, "goal": [6.0, 6.0], "random_start": False},
        )
    with pytest.raises(InvalidEpisodePoseError, match="explicit goal.*grid anchor"):
        FieldTask(cfg, mode="goto").prepare(
            seed=178,
            options={"scene": scene, "start": [6.0, 6.0, 0.0], "goal": bad[:2],
                     "random_start": False},
        )


def test_validated_pair_is_the_final_random_spawn_and_curriculum_goal():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(cfg, np.random.default_rng(3), None, np.array([10.0, 10.0]))
    rng = np.random.default_rng(178)
    start, goal = sample_episode_start_goal(
        scene, rng, random_start=True, spawn_near_crops=0.0,
        mode="goto", goal_distance=(1.0, 2.0), goal_angle=math.pi,
    )
    first = scene.sample_start(np.random.default_rng(178))

    assert has_grid_anchor(scene, start[:2])
    assert has_grid_anchor(scene, goal)
    assert 1.0 <= np.linalg.norm(goal - start[:2]) <= 2.0
    assert not np.allclose(start, first)  # rejected candidates advance the same RNG

    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=178, options={
        "scene": scene, "start": start, "goal": goal, "random_start": False,
    })

    np.testing.assert_allclose(task.robot[:3], start)
    np.testing.assert_allclose(task.goal, goal)
    assert task.validation["connected"] is True


@pytest.mark.parametrize("seed", range(24))
def test_random_start_sampling_handles_multiple_grid_boundary_seeds(seed):
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(cfg, np.random.default_rng(2), None, np.array([10.0, 10.0]))
    task = FieldTask(cfg, mode="goto")

    task.prepare(seed=seed, options={"scene": scene, "goal": [6.0, 6.0]})

    assert has_grid_anchor(scene, task.robot[:2]), f"seed={seed}, start={task.robot[:3].tolist()}"
    assert task.validation["connected"] is True


def test_pose_clear_uses_full_047m_vehicle_width_and_spawn_checks_all_rocks():
    cfg = _small_config(size=10.0, plants=0, terrain_enabled=False, validate_scene=False)
    assert cfg.robot_width == pytest.approx(0.47)
    terrain = Terrain(10.0, seed=13, ground_amplitude=0.0, rock_density=0.0)
    terrain.rock_xy = np.array([[5.0, 5.0]], dtype=np.float32)
    terrain.rock_radius = np.array([0.08], dtype=np.float32)
    terrain.rock_height = np.array([0.05], dtype=np.float32)
    terrain.rock_blocking = np.array([False], dtype=bool)
    terrain.rock_base_z = np.asarray(terrain.height(terrain.rock_xy), dtype=np.float32)
    scene = Scene(cfg, np.empty((0, 2)), np.empty((0,), dtype=bool),
                  np.random.default_rng(1), terrain=terrain)

    # The 0.47 m-wide H safety envelope fits 0.4 m from the small rock; a
    # circumscribed disk would reject this useful row-side clearance.
    assert scene.pose_clear([5.0, 5.4], 0.0, include_small_rocks=True)
    assert not scene.pose_clear([5.0, 5.4], math.pi / 2, include_small_rocks=True)
    # This pose would clear under the old 0.30 m body-only width, but the
    # confirmed track outer edge and small-rock margin make it unsafe now.
    assert not scene.pose_clear([5.0, 5.30], 0.0, include_small_rocks=True)
    assert not scene.pose_clear([5.0, 0.25], 0.0, include_small_rocks=True)
    assert scene.pose_clear([5.0, 0.27], 0.0, include_small_rocks=True)

    pose = scene.sample_start(np.random.default_rng(14))

    assert pose.shape == (3,)
    assert scene.pose_clear(pose[:2], pose[2], include_small_rocks=True)


def test_prepare_uses_fast_validation_and_keeps_fixed_scene(monkeypatch):
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=True)
    scene = Scene.generate(cfg, np.random.default_rng(2), None, np.array([10.0, 10.0]))
    monkeypatch.setattr(scene, "validate", lambda *_args: pytest.fail("reset must not run full visibility validation"))
    task = FieldTask(cfg)

    prepared = task.prepare(seed=4, options={"scene": scene, "goal": [10.0, 10.0]})

    assert prepared is scene
    assert task.validation["connected"] is True
    assert task.robot.shape == (5,)


def test_full_offline_validation_checks_a_real_dry_target(monkeypatch):
    cfg = _small_config(
        size=12.0,
        plants=1,
        dry_fraction=1.0,
        leaves_per_plant=3,
        terrain_enabled=False,
        validate_scene=False,
    )
    start = np.array([2.0, 2.0])
    goal = np.array([10.0, 10.0])
    scene = Scene.generate(cfg, np.random.default_rng(17), start, goal)
    cast = scene.cast
    cast_calls = 0

    def count_cast(*args, **kwargs):
        nonlocal cast_calls
        cast_calls += 1
        return cast(*args, **kwargs)

    monkeypatch.setattr(scene, "cast", count_cast)

    assert scene.targets.tolist() == [True]
    assert scene.leaf_dry.any()
    result = scene.validate(start, goal)

    assert cast_calls > 0
    assert isinstance(result["connected"], bool)
    assert result["free_cells"] > 0
    assert result["unobservable_targets"] in ([], [0])


def test_engine_frames_drive_observation_and_finish_without_scene_cast(monkeypatch):
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=5, options={"start": [2.0, 2.0, 0.4], "goal": [10.0, 10.0]})
    monkeypatch.setattr(task.scene, "cast", lambda *_args: pytest.fail("runtime sensing must not cast scene rays"))
    task.robot = np.array([2.0, 2.0, 0.4, 0.0, 0.0], dtype=np.float64)
    task.gimbal = np.array([0.1, -0.05], dtype=np.float64)
    task.body_pose = (0.17, -0.04, 0.03)
    task.view_origin = np.array([2.1, 2.0, 0.55], dtype=np.float32)
    task.view_basis = (
        np.array([0.99, 0.0, 0.1], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.1, 0.0, -0.99], dtype=np.float32),
    )
    captured = []

    def capture(kind):
        captured.append(kind)
        if kind == "lidar":
            ranges = np.full(cfg.lidar_rays, 2.0, dtype=np.float32)
            valid = np.ones(cfg.lidar_rays, dtype=bool)
            # A valid over-range hit saturates at 1; an explicitly invalid
            # finite-distance override remains the reserved policy value 0.
            ranges[0] = cfg.lidar_range * 2.0
            ranges[1] = 3.0
            valid[1] = False
            return {"ranges": ranges, "valid": valid}
        if kind == "depth":
            values = np.full((cfg.depth_height, cfg.depth_width), 2.5, dtype=np.float32)
            valid = np.ones((cfg.depth_height, cfg.depth_width), dtype=bool)
            values[0, 0] = cfg.depth_range
            valid[0, 0] = False
            values[0, 1] = cfg.depth_range
            return {"values": values, "valid": valid}
        return {"detections": [{
            "box": [0.5, 0.4, 0.2, 0.3],
            "class_id": 1,
            "confidence": 0.8,
            # A noisy negative range is still a candidate with a real box;
            # range validity is carried separately from candidate presence.
            "range": -0.1,
        }]}

    task.attach_sensors(capture)
    observation, info = task.initialize_sensors()

    assert captured == ["lidar", "depth", "camera"]
    assert observation["lidar"].shape == (1, cfg.lidar_rays)
    assert observation["depth"].shape == (1, cfg.depth_height, cfg.depth_width)
    assert observation["depth"].dtype == np.float32
    assert observation["detections"].shape == (cfg.max_detections, 8)
    assert observation["state"].shape == (33,)
    assert observation["state"].dtype == np.float32
    assert observation["feedback"].shape == (37,)
    assert observation["feedback"].dtype == np.float32
    np.testing.assert_array_equal(observation["feedback"][:21], np.zeros(21, dtype=np.float32))
    np.testing.assert_array_equal(observation["feedback"][21:29], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(observation["feedback"][29:37], np.ones(8, dtype=np.float32))
    assert observation["lidar"][0, 0] == 1.0
    assert observation["lidar"][0, 1] == 0.0
    assert task.frames["lidar"]["ranges"][0] == cfg.lidar_range
    assert task.frames["lidar"]["valid"][0]
    assert task.frames["lidar"]["ranges"][1] == pytest.approx(3.0)
    assert not task.frames["lidar"]["valid"][1]
    np.testing.assert_allclose(observation["lidar"][0, 2:], 2.0 / cfg.lidar_range)
    np.testing.assert_allclose(
        observation["detections"][0],
        [0.5, 0.4, 0.2, 0.3, 1.0, 0.8, -0.1 / cfg.depth_range, 1.0],
    )
    np.testing.assert_array_equal(
        observation["detections"][1:], np.zeros_like(observation["detections"][1:])
    )
    assert observation["depth"][0, 0, 0] == 0.0  # invalid pixels use the reserved policy value
    assert observation["depth"][0, 0, 1] == 1.0  # a valid max-range return remains distinguishable
    assert not bool(task.frames["depth"]["valid"][0, 0])
    assert bool(task.frames["depth"]["valid"][0, 1])
    np.testing.assert_allclose(observation["depth"][0, 1:], 2.5 / cfg.depth_range)
    # The compact state starts with task mode and goal, followed by position
    # and body-heading features; gimbal and IMU readings stay in feedback.
    assert observation["state"][0] == 1.0  # goto
    np.testing.assert_allclose(observation["state"][5:7], [0.0, 1.0], atol=1e-3)
    assert task._body_pose() == (0.17, -0.04, 0.03)
    np.testing.assert_allclose(task._origin(), [2.1, 2.0, 0.55])
    np.testing.assert_allclose(task.frames["lidar"]["body_attitude"], [0.03, -0.04])
    np.testing.assert_allclose(task.frames["lidar"]["pose"], [2.0, 2.0, 0.4, 0.1, -0.05])
    np.testing.assert_allclose(task.frames["lidar"]["estimated_pose"][2:], [0.0, 0.0, 0.0])
    assert info["sensor_sequences"] == {"lidar": 1, "depth": 1, "camera": 1}

    before = task.robot.copy()
    old_distance = np.linalg.norm(task.goal - task.robot[:2])
    task.robot[0] += 0.01
    result = task.finish(np.zeros(4, dtype=np.float32), before, old_distance)

    assert len(result) == 5
    assert result[0]["state"].shape == (33,)
    assert task.time == pytest.approx(cfg.dt)
    assert result[2] is False and result[3] is False


def test_last_action_state_does_not_leak_truth_motion_and_resets():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=5, options={"start": [2.0, 2.0, 0.4], "goal": [10.0, 10.0]})

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, 2.0, np.float32),
                    "valid": np.ones(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), 2.5, np.float32),
                    "valid": np.ones((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": []}

    task.attach_sensors(capture)
    task.initialize_sensors()
    task.robot[3:] = [0.25, 0.4]
    action = np.array([0.75, -0.6, 0.2, -0.3], dtype=np.float32)
    before = task.robot.copy()
    old_distance = float(np.linalg.norm(task.goal - task.robot[:2]))

    observation, *_ = task.finish(action, before, old_distance)

    assert observation["state"].shape == (33,)
    np.testing.assert_allclose(observation["state"][5:7], [0.0, 1.0], atol=1e-3)
    np.testing.assert_allclose(observation["state"][7:11], action)
    np.testing.assert_array_equal(task.last_action, action)
    action[:] = 0.0
    np.testing.assert_allclose(task.last_action, [0.75, -0.6, 0.2, -0.3])
    np.testing.assert_allclose(observation["state"][7:11], [0.75, -0.6, 0.2, -0.3])

    # With no valid attitude feedback, changing plant truth yaw/roll/pitch
    # cannot alter policy observation fields. Repeated GETs are stable too.
    baseline = task._observation()
    task.robot[2] = -1.25
    task.body_pose = (0.17, 0.55, -0.42)
    changed_truth = task._observation()
    np.testing.assert_array_equal(changed_truth["state"], baseline["state"])
    np.testing.assert_array_equal(changed_truth["feedback"], baseline["feedback"])
    repeated = task._observation()
    np.testing.assert_array_equal(repeated["state"], changed_truth["state"])
    np.testing.assert_array_equal(repeated["feedback"], changed_truth["feedback"])

    task.prepare(seed=6, options={"start": [2.0, 2.0, 0.4], "goal": [10.0, 10.0]})
    np.testing.assert_array_equal(task.last_action, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(task._feedback_snapshot.feedback[FEEDBACK_VALID_SLICE], np.zeros(8, dtype=np.float32))


def test_minimal_state_contract_encodes_mode_goal_and_previous_action_slots():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False,
                        validate_scene=False, position_noise=0.0)
    goto = FieldTask(cfg, mode="goto")
    goto.prepare(seed=21, options={"start": [2.0, 2.0, 0.0], "goal": [10.0, 8.0]})
    _attach_empty_sensors(goto, cfg)

    initial = goto._observation()["state"]
    assert initial.shape == (33,)
    assert initial[0] == 1.0
    np.testing.assert_allclose(initial[1:3], [8.0 / cfg.size, 6.0 / cfg.size], atol=1e-7)
    np.testing.assert_allclose(initial[3:5], [2.0 / cfg.size * 2 - 1] * 2, atol=1e-7)
    np.testing.assert_allclose(initial[5:7], [0.0, 1.0], atol=1e-7)
    np.testing.assert_array_equal(initial[7:11], np.zeros(4, dtype=np.float32))

    action = np.asarray([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    before = goto.robot.copy()
    old_distance = float(np.linalg.norm(goto.goal - goto.robot[:2]))
    after, *_ = goto.finish(action, before, old_distance)
    np.testing.assert_array_equal(after["state"][7:11], action)

    search = FieldTask(cfg, mode="search")
    search.prepare(seed=22, options={"start": [2.0, 2.0, 0.0], "goal": [10.0, 8.0]})
    _attach_empty_sensors(search, cfg)
    search_state = search._observation()["state"]
    assert search_state[0] == 0.0
    np.testing.assert_array_equal(search_state[1:3], np.zeros(2, dtype=np.float32))


def test_camera_callback_rejects_candidate_with_nonpositive_box_dimensions():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=23, options={"start": [2.0, 2.0, 0.0], "goal": [10.0, 8.0]})

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, cfg.lidar_range, np.float32),
                    "valid": np.zeros(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), cfg.depth_range, np.float32),
                    "valid": np.zeros((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": [{
            "box": [0.5, 0.5, 0.0, 0.2], "class_id": 1,
            "confidence": 0.9, "range": None,
        }]}

    task.attach_sensors(capture)
    with pytest.raises(ValueError, match="box width and height must be positive"):
        task.initialize_sensors()


def test_feedback_sample_drives_policy_pose_and_is_exposed_without_truth_fallback():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=31, options={"start": [2.0, 2.0, -0.7], "goal": [10.0, 8.0]})
    task.gimbal[:] = [0.3, -0.2]
    quat = [0.0, 0.0, math.sin(.25), math.cos(.25)]  # xyzw; firmware yaw estimate is +0.5 rad
    task.update_feedback(FeedbackSample(
        sample_time=0.0,
        attitude_euler_rad=[0.0, 0.0, .5],
        attitude_quat_xyzw=quat,
        gimbal_q_rad=[0.0, 0.0],
        gimbal_qd_rps=[0.0, 0.0],
        motor_iq_a=[1.0, -2.0],
        motor_rpm=[100.0, -200.0],
    ))

    _attach_empty_sensors(task, cfg)
    observation, _ = task.initialize_sensors()
    assert observation["state"].shape == (33,)
    assert observation["feedback"].shape == (37,)
    assert observation["feedback"][FEEDBACK_VALID_SLICE.start + 3] == 1.0
    # The simulated gimbal encoder has a documented 0.0005 rad white-noise
    # model; body-heading compensation uses measured q, while q/qd remain in
    # feedback rather than duplicating gimbal state fields.
    np.testing.assert_allclose(observation["state"][5:7], [math.sin(.5), math.cos(.5)], atol=2e-3)
    np.testing.assert_allclose(observation["feedback"][13:17], np.zeros(4), atol=2e-3)
    assert task.frames["lidar"]["pose"][2] == pytest.approx(-.7)
    assert task.frames["lidar"]["estimated_pose"][2] == pytest.approx(.5, abs=2e-3)
    assert task.frames["lidar"]["estimated_pose"][3] == pytest.approx(0.0, abs=1e-3)


def test_uwb_position_cache_has_independent_step_noise_without_motion_oracle():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False)

    def make_task(seed):
        task = FieldTask(cfg, mode="goto")
        task.prepare(seed=seed, options={"start": [2.0, 2.0, 0.0], "goal": [10.0, 8.0]})
        geometry_rng_before = repr(task.np_random.bit_generator.state)
        _attach_empty_sensors(task, cfg)
        assert repr(task.np_random.bit_generator.state) == geometry_rng_before
        return task

    task = make_task(61)
    same_seed = make_task(61)
    other_seed = make_task(62)
    assert task.estimated_pose_config["position_noise_std_m"] == cfg.position_noise
    assert "independent measurement" in task.estimated_pose_config["sampling"]
    np.testing.assert_array_equal(task._pose_noise, same_seed._pose_noise)
    assert not np.array_equal(task._pose_noise, other_seed._pose_noise)

    captured = task.frames["lidar"]["estimated_pose"].copy()
    prior_truth_xy = task.robot[:2].copy()
    task.robot[:2] += np.array([0.2, -0.1])
    task.time = cfg.dt
    task._sense()
    current = task._estimated_pose()
    true_delta = task.robot[:2] - prior_truth_xy
    measured_delta = current[:2] - captured[:2]
    assert np.linalg.norm(measured_delta - true_delta) > 1e-4

    rng_state = repr(task._uwb_rng.bit_generator.state)
    first_observation = task._observation()
    second_observation = task._observation()
    np.testing.assert_array_equal(first_observation["state"], second_observation["state"])
    assert repr(task._uwb_rng.bit_generator.state) == rng_state

    task.robot[:2] += np.array([0.2, 0.0])
    task.time = 2 * cfg.dt
    task._sense()
    fresh_frame = task._capture("lidar")
    np.testing.assert_array_equal(fresh_frame["estimated_pose"], task._estimated_pose())


def test_goto_progress_budget_is_normalized_and_cumulative_with_legacy_default():
    legacy_cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False,
                               time_penalty=0.0, collision_penalty=0.0, progress_weight=0.15,
                               goal_progress_budget=None)
    legacy = FieldTask(legacy_cfg, mode="goto")
    assert legacy_cfg.damage_penalty == 4.0
    legacy.prepare(seed=20, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
    _attach_empty_sensors(legacy, legacy_cfg)
    _, legacy_reward, *_ = _step_task_to(legacy, [3.0, 2.0])
    assert legacy_reward == pytest.approx(0.15)
    assert legacy._progress_reference is None

    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False,
                        time_penalty=0.0, collision_penalty=0.0, goal_progress_budget=5.0,
                        goal_hold=10.0)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=21, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
    _attach_empty_sensors(task, cfg)

    rewards = [_step_task_to(task, [3.0, 2.0])[1],  # d=2 -> 1: +2.5
               _step_task_to(task, [2.0, 2.0])[1],  # d=1 -> 2: -2.5
               _step_task_to(task, [4.0, 2.0])[1]]  # d=2 -> 0: +5

    assert rewards == pytest.approx([2.5, -2.5, 5.0])
    assert sum(rewards) == pytest.approx(cfg.goal_progress_budget)
    assert task._progress_reference == pytest.approx(2.0)
    assert not task.done  # no hard stop from reaching the pose before the dwell is satisfied


def _make_fast_stop_task(**updates):
    values = dict(
        size=12.0, plants=0, terrain_enabled=False, validate_scene=False,
        dt=0.1, time_penalty=0.1, collision_penalty=0.0,
        goto_reward_profile="fast_stop_v1", goal_progress_budget=5.0,
        goal_tolerance=0.4, goal_hold=0.5,
        goal_stop_speed=0.05, goal_stop_angular_speed=0.1,
    )
    values.update(updates)
    cfg = _small_config(**values)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=421, options={"start": [2.0, 2.0, 0.0], "goal": [2.2, 2.0]})
    _attach_empty_sensors(task, cfg)
    return task


def test_fast_stop_requires_world_xy_speed_and_full_body_angular_hold_for_half_second():
    task = _make_fast_stop_task()
    task.robot[3:] = 0.0  # forward/yaw projections alone would incorrectly declare a stop
    task.root_com_velocity_xy = np.array([0.0, 0.06])  # lateral slide is real motion
    task.body_angular_velocity = np.zeros(3)
    for _ in range(3):
        _, _, terminated, truncated, info = _step_task_to(task, [2.2, 2.0])
        assert not terminated and not truncated
        assert info["goal_stop"]["hold_s"] == 0.0
        assert info["goal_stop"]["linear_speed_m_s"] == pytest.approx(0.06)
        assert info["goal_stop"]["speed_source"] == "root_com_vel_w"

    task.root_com_velocity_xy = np.array([0.049, 0.0])
    task.body_angular_velocity = np.array([0.0, 0.0, 0.099])
    _step_task_to(task, [2.2, 2.0])
    _, _, _, _, info = _step_task_to(task, [2.2, 2.0])
    assert info["goal_stop"]["hold_s"] == pytest.approx(0.2)

    # Roll/pitch angular velocity counts just like yaw; equality at either
    # threshold also fails the strict "below threshold" condition.
    task.body_angular_velocity = np.array([0.0, 0.101, 0.0])
    _step_task_to(task, [2.2, 2.0])
    assert task._hold == 0.0
    task.body_angular_velocity = np.array([0.0, 0.1, 0.0])
    task.root_com_velocity_xy = np.array([0.05, 0.0])
    _step_task_to(task, [2.2, 2.0])
    assert task._hold == 0.0

    task.root_com_velocity_xy = np.array([0.049, 0.0])
    task.body_angular_velocity = np.array([0.0, 0.099, 0.0])
    terminal = None
    for _ in range(5):
        terminal = _step_task_to(task, [2.2, 2.0])
    _, reward, terminated, truncated, info = terminal
    assert (terminated, truncated) == (True, False)
    assert info["reason"] == "goal_reached"
    assert info["goal_stop"]["hold_s"] == pytest.approx(0.5)
    assert info["goal_stop"]["hold_required_s"] == pytest.approx(0.5)
    assert info["goal_stop"]["stop_speed_limit_m_s"] == pytest.approx(0.05)
    assert info["goal_stop"]["stop_angular_speed_limit_rad_s"] == pytest.approx(0.1)
    assert info["reward_components"]["goal_success"] == FAST_STOP_SUCCESS_REWARD
    assert sum(info["reward_components"].values()) == pytest.approx(reward)


def test_fast_stop_task_fallback_uses_forward_and_yaw_speed_and_rejects_nonfinite_values():
    task = _make_fast_stop_task()
    task.root_com_velocity_xy = None
    task.robot[3:] = [0.06, 0.0]
    _, _, terminated, truncated, info = _step_task_to(task, [2.2, 2.0])
    assert not terminated and not truncated
    assert info["goal_stop"]["speed_source"] == "task_forward_speed_fallback"
    assert info["goal_stop"]["linear_speed_m_s"] == pytest.approx(0.06)
    assert info["goal_stop"]["hold_s"] == 0.0

    task.robot[3:] = [0.0, 0.11]
    _step_task_to(task, [2.2, 2.0])
    assert task._hold == 0.0
    task.robot[4] = float("nan")
    with pytest.raises(ValueError, match="fallback stop kinematics must be finite"):
        task._info()


def test_fast_stop_terminal_outcomes_cancel_progress_and_timeout_is_terminated():
    task = _make_fast_stop_task(max_time=0.3, goal_hold=0.5)
    task.prepare(seed=422, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
    _attach_empty_sensors(task, task.cfg)
    rewards = []
    progress_rewards = []
    terminal_info = None
    for point in ([2.5, 2.0], [3.0, 2.0], [3.5, 2.0]):
        _, reward, terminated, truncated, info = _step_task_to(task, point)
        rewards.append(reward)
        progress_rewards.append(info["reward_components"]["goal_progress"])
        assert sum(info["reward_components"].values()) == pytest.approx(reward)
        if info["reason"]:
            terminal_info = info
            assert (terminated, truncated) == (True, False)
    assert terminal_info is not None
    assert terminal_info["reason"] == "time_limit"
    assert terminal_info["reward_components"]["terminal_failure"] == FAST_STOP_FAILURE_REWARD
    assert sum(progress_rewards) == pytest.approx(0.0)
    assert sum(rewards) == pytest.approx(-10.0 - 0.1 * task.time)


@pytest.mark.parametrize("failure", ["boundary", "overtilt_roll", "overtilt_pitch"])
def test_fast_stop_physics_failure_wins_over_same_step_goal_success(failure):
    task = _make_fast_stop_task(dt=0.1, goal_hold=0.1, time_penalty=0.0)
    task.root_com_velocity_xy = np.zeros(2)
    task.body_angular_velocity = np.zeros(3)
    boundary = failure == "boundary"
    if failure == "overtilt_roll":
        task.body_pose = (0.0, 0.0, 1.21)
    elif failure == "overtilt_pitch":
        task.body_pose = (0.0, 1.21, 0.0)
    _, reward, terminated, truncated, info = task.finish(
        np.zeros(4, dtype=np.float32), task.robot.copy(),
        float(np.linalg.norm(task.goal - task.robot[:2])), boundary=boundary,
    )
    assert (terminated, truncated) == (True, False)
    assert info["reason"] == "outside_or_overturned"
    assert info["reward_components"]["goal_success"] == 0.0
    assert info["reward_components"]["terminal_failure"] == FAST_STOP_FAILURE_REWARD
    assert info["goal_stop"]["hold_s"] == 0.0
    assert reward == pytest.approx(FAST_STOP_FAILURE_REWARD)
    assert sum(info["reward_components"].values()) == pytest.approx(reward)
    assert task.physics_failure
    task.prepare(seed=423, options={"start": [2.0, 2.0, 0.0], "goal": [2.2, 2.0]})
    assert not task.physics_failure


def test_fast_stop_legacy_profile_keeps_forward_yaw_hold_and_legacy_default():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False,
                        dt=0.1, time_penalty=0.0, collision_penalty=0.0,
                        goal_tolerance=0.4, goal_hold=0.1)
    assert cfg.goto_reward_profile == "legacy"
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=424, options={"start": [2.0, 2.0, 0.0], "goal": [2.2, 2.0]})
    _attach_empty_sensors(task, cfg)
    task.robot[3:] = 0.0
    task.root_com_velocity_xy = np.array([0.0, 1.0])
    task.body_angular_velocity = np.array([1.0, 0.0, 0.0])
    _, reward, terminated, truncated, info = _step_task_to(task, [2.2, 2.0])
    assert (terminated, truncated) == (True, False)
    assert info["reason"] == "goal_reached"
    assert info["goal_stop"]["linear_speed_m_s"] == 0.0
    assert info["goal_stop"]["angular_speed_rad_s"] == 0.0
    assert info["goal_stop"]["stop_speed_limit_m_s"] == 0.2
    assert info["goal_stop"]["speed_source"] == "root_forward_projection_and_world_z_rate"
    assert reward == pytest.approx(5.03)


def test_progress_reference_uses_first_post_reset_distance_and_restarts_on_goal_or_prepare():
    cfg = _small_config(size=12.0, plants=0, terrain_enabled=False, validate_scene=False,
                        time_penalty=0.0, collision_penalty=0.0, goal_progress_budget=5.0,
                        goal_hold=10.0)
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=22, options={"start": [1.0, 1.0, 0.0], "goal": [10.0, 1.0]})
    # IsaacFieldEnv finalizes randomized spawn and curriculum goal after prepare().
    task.robot[:2] = [2.0, 2.0]
    task.goal = np.array([6.0, 2.0])
    assert task._progress_reference is None
    _attach_empty_sensors(task, cfg)
    _, reward, *_ = _step_task_to(task, [3.0, 2.0])
    assert reward == pytest.approx(1.25)  # first real distance was 4 m, not prepare's 9 m
    assert task._progress_reference == pytest.approx(4.0)

    task.set_goal([8.0, 2.0])
    assert task._progress_reference is None
    _, reward, *_ = _step_task_to(task, [4.0, 2.0])
    assert reward == pytest.approx(1.0)  # new target starts at 5 m
    assert task._progress_reference == pytest.approx(5.0)

    task.prepare(seed=23, options={"start": [1.0, 1.0, 0.0], "goal": [10.0, 1.0]})
    assert task._progress_reference is None
    # Again simulate the final spawn/goal written by IsaacFieldEnv after prepare().
    task.robot[:2] = [2.0, 2.0]
    task.goal = np.array([6.0, 2.0])
    _attach_empty_sensors(task, cfg)
    _, reward, *_ = _step_task_to(task, [3.0, 2.0])
    assert reward == pytest.approx(1.25)
    assert task._progress_reference == pytest.approx(4.0)


def test_safe_progress_budget_rewards_intact_arrival_and_allows_penalized_crushing():
    cfg = _small_config(size=12.0, plants=1, terrain_enabled=False, validate_scene=False,
                        dt=0.1, time_penalty=0.0, collision_penalty=0.0,
                        goal_progress_budget=5.0, damage_penalty=20.0,
                        goal_tolerance=0.2, goal_hold=0.1)

    def make_scene():
        scene = Scene(cfg, np.array([[2.7, 2.125]], dtype=np.float32), np.array([False]),
                      np.random.default_rng(31))
        scene.radius[:] = 0.15
        return scene

    intact = FieldTask(cfg, mode="goto")
    intact.prepare(seed=24, options={"scene": make_scene(), "start": [2.0, 2.5, 0.0],
                                     "goal": [4.0, 2.5]})
    _attach_empty_sensors(intact, cfg)
    _, arrival_reward, terminated, truncated, info = _step_task_to(intact, [4.0, 2.5])
    assert (terminated, truncated) == (True, False)
    assert info["damage"] == 0
    assert info["reason"] == "goal_reached"
    assert arrival_reward == pytest.approx(10.0)  # 5 progress budget + 5 terminal bonus

    crushing = FieldTask(cfg, mode="goto")
    crushing.prepare(seed=25, options={"scene": make_scene(), "start": [2.0, 2.0, 0.0],
                                       "goal": [4.0, 2.0]})
    _attach_empty_sensors(crushing, cfg)
    _, crush_reward, terminated, truncated, info = _step_task_to(crushing, [2.5, 2.0])
    assert (terminated, truncated) == (False, False)
    assert info["damage"] == 1
    assert crush_reward == pytest.approx(1.25 - 20.0)
    _, continued_reward, terminated, truncated, info = _step_task_to(crushing, [2.5, 2.0])
    assert (terminated, truncated) == (False, False)
    assert info["damage"] == 1
    assert continued_reward == pytest.approx(0.0)  # one penalty per newly damaged plant


def _damage_test_task(xy, *, metric=DAMAGE_METRIC_STEM_CONTACT_V2,
                      body_z=0.15, stem_height=0.5, protection_radius=0.4):
    cfg = _small_config(plants=0, terrain_enabled=False, validate_scene=False,
                        damage_metric=metric)
    scene = Scene(cfg, np.array([[8.0, 8.0]], dtype=np.float32), np.array([False]),
                  np.random.default_rng(130))
    task = FieldTask(cfg, mode="goto")
    task.prepare(seed=131, options={"scene": scene, "start": [2.0, 2.0, 0.0],
                                    "goal": [20.0, 20.0]})
    # Put the plant at a controlled location after prepare's independent pose
    # validation; the broad radius remains available to prove v2 ignores it.
    scene.xy[0] = np.asarray(xy, dtype=np.float32)
    scene.radius[:] = protection_radius
    scene.stem_radius[:] = 0.012
    scene.root_z[:] = 0.0
    scene.height[:] = stem_height
    task.body_pose = (body_z, 0.0, 0.0)
    return task


def _sweep_stationary(task, *, xy=(2.0, 2.0), yaw=0.0):
    pose = np.array([xy[0], xy[1], yaw, 0.0, 0.0], dtype=np.float64)
    task._damage_sweep(pose, pose)


def test_damage_metric_defaults_and_legacy_checkpoint_config_resolution():
    assert Config().damage_metric == DAMAGE_METRIC_STEM_CONTACT_V2
    assert Config().robot_width == pytest.approx(ROBOT_SPEC.vehicle_width_m)
    old_config = Config.from_checkpoint_config({"size": 12.0, "plants": 0,
                                                "terrain_enabled": False})
    assert old_config.damage_metric == DAMAGE_METRIC_ROOT_PROTECTION_V1
    assert old_config.robot_width == pytest.approx(ROBOT_SPEC.body_width_m)
    saved_old_config = Config.from_checkpoint_config({"size": 12.0, "plants": 0,
                                                      "terrain_enabled": False,
                                                      "robot_width": 0.3})
    assert saved_old_config.robot_width == pytest.approx(0.3)
    explicit = Config.from_checkpoint_config({"size": 12.0, "plants": 0,
                                              "terrain_enabled": False,
                                              "damage_metric": DAMAGE_METRIC_STEM_CONTACT_V2})
    assert explicit.damage_metric == DAMAGE_METRIC_STEM_CONTACT_V2


def test_stem_contact_v2_ignores_root_protection_circle_and_legacy_keeps_it():
    # This point is outside the 40 mm track bands and the chassis rectangle,
    # but overlaps the old 150 mm root-protection disk expanded by the body.
    v2 = _damage_test_task((2.0, 2.27), protection_radius=0.15)
    _sweep_stationary(v2)
    assert v2.damaged == set()
    assert v2._info()["damage_metric"] == DAMAGE_METRIC_STEM_CONTACT_V2
    assert v2.damage_config["track_contact_length_m"] == pytest.approx(0.595)
    assert v2.damage_config["track_band_width_m"] == pytest.approx(0.06)
    assert v2.damage_config["track_center_separation_m"] == pytest.approx(0.41)
    assert v2.damage_config["track_outer_width_m"] == pytest.approx(0.47)
    assert v2.damage_config["chassis_width_m"] == pytest.approx(0.3)
    assert v2.damage_config["penalty_per_plant"] == pytest.approx(v2.cfg.damage_penalty)
    assert "user-confirmed 0.47 m overall width" in v2.damage_config["track_center_separation_status"]

    legacy = _damage_test_task((2.0, 2.27), metric=DAMAGE_METRIC_ROOT_PROTECTION_V1,
                               protection_radius=0.15)
    _sweep_stationary(legacy)
    assert legacy.damaged == {0}
    assert legacy._info()["damage_metric"] == DAMAGE_METRIC_ROOT_PROTECTION_V1
    assert legacy.damage_config["plant_geometry"] == "Scene.radius root-protection disk"


def test_stem_contact_v2_uses_exact_track_length_and_reuses_asset_track_width():
    # A short stem within the specified 60 mm track band is crushed at flat
    # ground level even though it does not reach the chassis body.
    hit = _damage_test_task((2.20, 2.225), body_z=0.15, stem_height=0.04,
                            protection_radius=0.4)
    _sweep_stationary(hit)
    assert hit.damaged == {0}

    # 42.5 mm beyond the exact 595 mm track end is outside its stem-radius
    # dilation. The 45 mm roller radius must not silently enlarge contact span.
    clear = _damage_test_task((2.34, 2.225), body_z=0.15, stem_height=0.04,
                              protection_radius=0.4)
    _sweep_stationary(clear)
    assert clear.damaged == set()


def test_stem_contact_v2_checks_chassis_stem_vertical_overlap_and_sweeps_rotation():
    # Centerline stems lie between the two track bands, so only body/main-stem
    # contact can damage them. Current root height determines vertical overlap.
    low_body = _damage_test_task((2.0, 2.0), body_z=0.15, stem_height=0.5,
                                 protection_radius=0.4)
    _sweep_stationary(low_body)
    assert low_body.damaged == {0}

    high_body = _damage_test_task((2.0, 2.0), body_z=2.0, stem_height=0.5,
                                  protection_radius=0.4)
    _sweep_stationary(high_body)
    assert high_body.damaged == set()

    # End orientations miss this stem, but the left track crosses it while
    # rotating. The sweep remains a finely sampled geometric approximation.
    turning = _damage_test_task((2.0, 2.225),
                                body_z=0.15, stem_height=0.04, protection_radius=0.4)
    before = np.array([2.0, 2.0, -math.pi / 3, 0.0, 0.0])
    after = np.array([2.0, 2.0, math.pi / 3, 0.0, 0.0])
    turning._damage_sweep(before, before)
    assert turning.damaged == set()
    turning._damage_sweep(after, after)
    assert turning.damaged == set()
    turning._damage_sweep(before, after)
    assert turning.damaged == {0}


def test_stem_contact_v2_catches_a_thin_stem_crossed_between_fast_endpoint_poses():
    task = _damage_test_task((2.5, 2.225), body_z=0.15, stem_height=0.04,
                             protection_radius=0.4)
    before = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    after = np.array([3.0, 2.0, 0.0, 0.0, 0.0])
    task._damage_sweep(before, before)
    assert task.damaged == set()
    task._damage_sweep(after, after)
    assert task.damaged == set()
    task._damage_sweep(before, after)
    assert task.damaged == {0}


def test_stem_contact_v2_does_not_count_body_or_tracks_when_both_clear_the_maize():
    task = _damage_test_task((2.20, 2.225), body_z=2.0, stem_height=0.5,
                             protection_radius=0.4)
    _sweep_stationary(task)
    assert task.damaged == set()


@pytest.mark.parametrize(("name", "value"), [
    ("goal_progress_budget", -1.0),
    ("goal_progress_budget", float("inf")),
    ("goal_progress_budget", True),
    ("goal_tolerance", 0.0),
    ("goal_tolerance", float("nan")),
    ("goal_hold", 0.0),
    ("goal_hold", -1.0),
    ("damage_penalty", -1.0),
    ("time_penalty", float("nan")),
    ("progress_weight", -0.1),
    ("goto_reward_profile", "unknown"),
    ("goal_stop_speed", -0.01),
    ("goal_stop_angular_speed", float("inf")),
])
def test_reward_config_rejects_invalid_values(name, value):
    with pytest.raises(ValueError):
        _small_config(**{name: value})


def test_fast_stop_profile_requires_a_progress_budget():
    with pytest.raises(ValueError, match="fast_stop_v1 requires goal_progress_budget"):
        _small_config(goto_reward_profile="fast_stop_v1", goal_progress_budget=None)
