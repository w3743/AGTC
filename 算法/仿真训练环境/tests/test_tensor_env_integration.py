"""CPU integration checks for the opt-in empty-field Torch task backend."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaac_field.body_stability import BodyStabilityAccumulator, BodyStabilityConfig, body_stability_sample
from isaac_field.env import FieldEnvCfg, IsaacFieldEnv
from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.task import Config, FieldTask
from test_task import _attach_empty_sensors, _make_fast_stop_task


def _flat12_config(**updates):
    values = dict(
        size=12.0,
        plants=0,
        terrain_enabled=False,
        rock_density=0.0,
        validate_scene=False,
        dt=0.05,
        max_time=60.0,
        time_penalty=0.1,
        goal_tolerance=0.4,
        goal_hold=0.5,
        goal_stop_speed=0.05,
        goal_stop_angular_speed=0.1,
        goal_progress_budget=5.0,
        goto_reward_profile="fast_stop_v2",
        fast_stop_failure_remaining_time_cost=True,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
        position_noise=0.025,
        lidar_rays=8,
        depth_width=4,
        depth_height=3,
        lidar_range=8.0,
        lidar_min_range=0.12,
        lidar_relative_error=0.01,
        lidar_period=0.1,
        lidar_latency=0.03,
        depth_range=5.0,
        depth_min_range=0.3,
        range_noise=0.03,
        depth_period=0.4,
        depth_latency=0.3,
        camera_period=0.2,
        camera_latency=0.12,
        max_detections=16,
    )
    values.update(updates)
    return Config(**values)


def _backend_cfg(task, *, mode="goto"):
    return SimpleNamespace(
        task=task,
        mode=mode,
        sim=SimpleNamespace(
            dt=0.001,
            render_interval=50,
            physics=SimpleNamespace(
                solver_type=1,
                min_position_iteration_count=40,
                max_position_iteration_count=40,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=1,
            ),
        ),
        decimation=50,
    )


def test_backend_guard_is_opt_in_and_rejects_unsupported_profiles():
    assert FieldEnvCfg().task_backend == "reference"
    task = _flat12_config(lidar_rays=400, depth_width=88, depth_height=160)
    IsaacFieldEnv._validate_task_backend(_backend_cfg(task), "torch_batched")
    IsaacFieldEnv._validate_task_backend(SimpleNamespace(), "reference")

    with pytest.raises(ValueError, match="mode='search'"):
        IsaacFieldEnv._validate_task_backend(_backend_cfg(task, mode="search"), "torch_batched")
    with pytest.raises(ValueError, match="input_schema"):
        IsaacFieldEnv._validate_task_backend(
            _backend_cfg(_flat12_config(
                lidar_rays=400, depth_width=88, depth_height=160,
                input_schema="sensor-relative-goal-v1",
            )),
            "torch_batched",
        )


def test_task_keeps_tensor_sensor_frames_on_device_and_materializes_only_for_debug(monkeypatch):
    task = _make_fast_stop_task(
        goto_reward_profile="fast_stop_v2",
        fast_stop_failure_remaining_time_cost=True,
    )
    lidar = torch.tensor([1.0, 8.0, 3.0, 8.0, 2.0, 8.0, 4.0, 8.0])
    lidar_valid = torch.tensor([True, False, True, False, True, False, True, False])
    depth = torch.full((3, 4), 2.0)
    depth_valid = torch.ones((3, 4), dtype=torch.bool)
    task.attach_sensors(lambda kind: (
        {"ranges": lidar, "valid": lidar_valid, "_tensor_sensor_frame": True}
        if kind == "lidar" else
        {"values": depth, "valid": depth_valid, "intrinsics": np.eye(3),
         "_tensor_sensor_frame": True}
    ))
    original_asarray = np.asarray

    def reject_tensor_asarray(value, *args, **kwargs):
        if isinstance(value, torch.Tensor):
            raise AssertionError("large tensor sensor payload crossed into NumPy")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", reject_tensor_asarray)
    lidar_frame = task._capture("lidar")
    depth_frame = task._capture("depth")
    assert lidar_frame["ranges"] is lidar
    assert lidar_frame["valid"] is lidar_valid
    assert depth_frame["values"] is depth
    assert depth_frame["valid"] is depth_valid
    task.frames = {"lidar": lidar_frame}
    task._tensor_observation_callback = lambda: {"batched": True}
    assert task._observation() == {"batched": True}

    materialized = FieldTask._materialize_sensor_frame(lidar_frame)
    assert isinstance(materialized["ranges"], np.ndarray)
    assert isinstance(materialized["valid"], np.ndarray)
    assert materialized["ranges"].tolist() == lidar.tolist()
    assert "_tensor_sensor_frame" not in materialized  # only the debug copy drops the marker
    assert lidar_frame["_tensor_sensor_frame"] is True


def test_sensor_batch_preserves_due_clocks_latency_rng_and_partial_reset():
    cfg = _flat12_config()
    tasks = []
    for seed in (31, 32):
        task = FieldTask(cfg, mode="goto")
        task.prepare(seed=seed, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
        tasks.append(task)

    lidar_ranges = torch.tensor([
        [1.0, 2.0, 8.0, float("nan"), 0.1, 7.0, 3.0, 8.0],
        [4.0, 5.0, 8.0, 1.5, 0.2, 6.0, 2.0, 8.0],
    ])
    ray_hits = torch.zeros((2, 8, 3), dtype=torch.float32)
    ray_hits[:, :, 0] = lidar_ranges.nan_to_num(nan=0.0)
    ray_pos = torch.zeros((2, 3), dtype=torch.float32)
    depth_values = torch.tensor([
        [[[0.2], [0.3], [1.0], [2.0]], [[3.0], [4.0], [5.0], [6.0]], [[float("nan")], [2.5], [0.4], [1.4]]],
        [[[0.5], [1.5], [2.5], [3.5]], [[4.5], [5.0], [5.1], [0.1]], [[1.0], [2.0], [3.0], [4.0]]],
    ], dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32).expand(2, 3, 3).clone()
    env = IsaacFieldEnv.__new__(IsaacFieldEnv)
    env.task_backend = "torch_batched"
    # Isaac Lab 3 exposes DirectRLEnv.device as the string "cpu"/"cuda:0";
    # sensor tensors expose torch.device, so the environment must normalize it.
    env.sim = SimpleNamespace(device="cpu")
    env.field_cfg = cfg
    env.tasks = tasks
    env._sensor_cache = {}
    env._tensor_sensor_frame_cache = {}
    env._tensor_capture_env_ids = (0, 1)
    env._tensor_capture_initial_env_ids = frozenset((0, 1))
    class _SensorScene(dict):
        num_envs = 2

    env.scene = _SensorScene({
        "lidar": SimpleNamespace(data=SimpleNamespace(
            ray_hits_w=SimpleNamespace(torch=ray_hits), pos_w=SimpleNamespace(torch=ray_pos),
        )),
        "depth": SimpleNamespace(data=SimpleNamespace(
            output={"distance_to_image_plane": SimpleNamespace(torch=depth_values)},
            intrinsic_matrices=SimpleNamespace(torch=intrinsics),
        )),
    })
    for env_id, task in enumerate(tasks):
        task.attach_sensors(lambda kind, i=env_id: env._capture(i, kind))
        task.update_uwb_pose_noise()
    for task in tasks:
        task.initialize_sensors(build_observation=False)
    assert all(task.frames["lidar"]["_tensor_sensor_frame"] for task in tasks)
    assert all(task.frames["depth"]["_tensor_sensor_frame"] for task in tasks)
    assert all(task.frames["camera"]["detections"] == [] for task in tasks)

    env._tensor_goal_xy = torch.as_tensor(np.stack([task.goal for task in tasks]), dtype=torch.float64)
    env._tensor_episode_time = torch.zeros((2,), dtype=torch.float64)
    env._tensor_last_actions = torch.zeros((2, 4), dtype=torch.float32)
    env._tensor_lidar_ranges = torch.full((2, cfg.lidar_rays), cfg.lidar_range)
    env._tensor_lidar_valid = torch.zeros((2, cfg.lidar_rays), dtype=torch.bool)
    env._tensor_depth_values = torch.full((2, cfg.depth_height, cfg.depth_width), cfg.depth_range)
    env._tensor_depth_valid = torch.zeros((2, cfg.depth_height, cfg.depth_width), dtype=torch.bool)
    env._tensor_lidar_frame_seq = [None, None]
    env._tensor_depth_frame_seq = [None, None]
    packed = env._build_tensor_observations()
    expected_observations = []
    for task in tasks:
        device_frames = task.frames
        task.frames = {
            name: FieldTask._materialize_sensor_frame(frame)
            for name, frame in device_frames.items()
        }
        expected_observations.append(task._observation())
        task.frames = device_frames
    for env_id, expected in enumerate(expected_observations):
        for key, value in expected.items():
            np.testing.assert_allclose(packed[key][env_id].numpy(), value, rtol=1e-6, atol=1e-6)
    lidar_buffer_id = env._tensor_lidar_ranges.data_ptr()
    depth_buffer_id = env._tensor_depth_values.data_ptr()
    old_lidar_policy = packed["lidar"]
    old_depth_policy = packed["depth"]
    old_lidar_snapshot = old_lidar_policy.clone()
    old_depth_snapshot = old_depth_policy.clone()
    unchanged = env._build_tensor_observations()
    assert unchanged["lidar"] is old_lidar_policy
    assert unchanged["depth"] is old_depth_policy
    assert env._tensor_lidar_ranges.data_ptr() == lidar_buffer_id
    assert env._tensor_depth_values.data_ptr() == depth_buffer_id

    before_states = [copy.deepcopy(task.np_random.bit_generator.state) for task in tasks]
    reference_rngs = []
    for state in before_states:
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state)
        reference_rngs.append(rng)
    raw_lidar = torch.linalg.vector_norm(ray_hits - ray_pos[:, None, :], dim=-1).numpy()
    raw_depth = depth_values[..., 0].numpy()
    expected_lidar = []
    expected_depth = []
    for env_id, rng in enumerate(reference_rngs):
        raw = raw_lidar[env_id]
        valid = np.isfinite(raw) & (raw >= cfg.lidar_min_range) & (raw < cfg.lidar_range)
        noisy = np.full(cfg.lidar_rays, cfg.lidar_range, dtype=np.float32)
        noise = rng.uniform(-cfg.lidar_relative_error, cfg.lidar_relative_error, size=int(valid.sum()))
        noisy[valid] = np.clip(
            raw[valid] * (1.0 + noise), cfg.lidar_min_range, cfg.lidar_range,
        )
        expected_lidar.append(noisy)
        raw = raw_depth[env_id]
        valid = np.isfinite(raw) & (raw >= cfg.depth_min_range) & (raw <= cfg.depth_range)
        noisy_depth = np.full(raw.shape, cfg.depth_range, dtype=np.float32)
        noise = rng.normal(0.0, cfg.range_noise, size=int(valid.sum()))
        noisy_depth[valid] = np.clip(
            raw[valid] + noise, cfg.depth_min_range, cfg.depth_range,
        )
        expected_depth.append(noisy_depth)

    for task in tasks:
        task.time = 0.4
    env._sensor_cache.clear()
    env._tensor_sensor_frame_cache.clear()
    env._tensor_capture_initial_env_ids = frozenset()
    for task in tasks:
        task.update_uwb_pose_noise()
    for task in tasks:
        task._sense()

    for env_id, task in enumerate(tasks):
        lidar_frame = task._pending["lidar"][-1][1]
        depth_frame = task._pending["depth"][-1][1]
        np.testing.assert_array_equal(lidar_frame["ranges"].numpy(), expected_lidar[env_id])
        np.testing.assert_array_equal(depth_frame["values"].numpy(), expected_depth[env_id])
        assert lidar_frame["capture"] == pytest.approx(0.4)
        assert depth_frame["capture"] == pytest.approx(0.4)
        assert task._pending["lidar"][-1][0] == pytest.approx(0.43)
        assert task._pending["depth"][-1][0] == pytest.approx(0.7)
        assert task._dropped == {"lidar": 3, "depth": 0, "camera": 1}
        assert task.np_random.bit_generator.state == reference_rngs[env_id].bit_generator.state

    # The delayed LiDAR frame captured at t=.4 is delivered at t=.45. The new
    # normalized batch must not mutate the previous observation storage.
    for task in tasks:
        task.time = 0.45
        task.update_uwb_pose_noise()
    env._sensor_cache.clear()
    env._tensor_sensor_frame_cache.clear()
    for task in tasks:
        task._sense()
    next_observation = env._build_tensor_observations()
    assert next_observation["lidar"] is not old_lidar_policy
    assert next_observation["depth"] is old_depth_policy
    torch.testing.assert_close(old_lidar_policy, old_lidar_snapshot)
    torch.testing.assert_close(old_depth_policy, old_depth_snapshot)
    before_reset_lidar = next_observation["lidar"]
    before_reset_depth = next_observation["depth"]
    before_reset_lidar_snapshot = before_reset_lidar.clone()
    before_reset_depth_snapshot = before_reset_depth.clone()

    untouched_state = copy.deepcopy(tasks[0].np_random.bit_generator.state)
    untouched_serials = dict(tasks[0]._serial)
    untouched_lidar_row = env._tensor_lidar_ranges[0].clone()
    untouched_depth_row = env._tensor_depth_values[0].clone()
    tasks[1].prepare(seed=99, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
    env._tensor_lidar_frame_seq[1] = None
    env._tensor_depth_frame_seq[1] = None
    env._tensor_lidar_policy_cache = None
    env._tensor_depth_policy_cache = None
    env._sensor_cache.clear()
    env._tensor_sensor_frame_cache.clear()
    env._tensor_capture_env_ids = (1,)
    env._tensor_capture_initial_env_ids = frozenset((1,))
    tasks[1].update_uwb_pose_noise()
    tasks[1].initialize_sensors(build_observation=False)
    assert tasks[0]._serial == untouched_serials
    assert tasks[0].np_random.bit_generator.state == untouched_state
    assert tasks[1].frames["depth"]["_tensor_sensor_frame"]
    env._build_tensor_observations()
    torch.testing.assert_close(env._tensor_lidar_ranges[0], untouched_lidar_row)
    torch.testing.assert_close(env._tensor_depth_values[0], untouched_depth_row)
    torch.testing.assert_close(env._tensor_lidar_ranges[1], tasks[1].frames["lidar"]["ranges"])
    torch.testing.assert_close(env._tensor_depth_values[1], tasks[1].frames["depth"]["values"])
    torch.testing.assert_close(before_reset_lidar, before_reset_lidar_snapshot)
    torch.testing.assert_close(before_reset_depth, before_reset_depth_snapshot)
    env._tensor_capture_env_ids = None
    env._tensor_capture_initial_env_ids = frozenset()


class _TaskWithoutLegacyFinish:
    def __init__(self, *, done=False, time=0.0, hold=0.0, xy=(2.0, 2.0)):
        self.done = done
        self.mode = "goto"
        self.time = time
        self._hold = hold
        self._initial_goal_distance = 2.0
        self.physics_failure = False
        self.goal = np.asarray([4.0, 2.0], dtype=np.float64)
        self.robot = np.asarray([*xy, 0.0, 0.0, 0.0], dtype=np.float64)
        self.body_pose = (0.0, 0.0, 0.0)
        self.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
        self.body_angular_velocity = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.reward = 0.0
        self.total_reward = 0.0
        self.reward_components = {}
        self.frames = {"lidar": {}, "depth": {}, "camera": {}}
        self._new = {"lidar": False, "depth": False, "camera": False}
        self.body_stability_sample = None
        self.body_stability_stats = BodyStabilityAccumulator()
        self.trail = []
        self.sense_calls = 0
        self.scene = SimpleNamespace()
        self.reason = "already_done" if done else ""

    def finish(self, *args, **kwargs):
        raise AssertionError("torch_batched path must never call FieldTask.finish")

    def update_uwb_pose_noise(self):
        pass

    def _sense(self):
        self.sense_calls += 1

    def _info(self):
        return {
            "distance": float(np.linalg.norm(self.goal - self.robot[:2])),
            "reason": self.reason,
            "is_success": self.reason == "goal_reached",
            "reward_components": self.reward_components.copy(),
            "goal_stop": {},
            "sensor_sequences": {},
            "sensor_dropped": {},
        }


def test_batched_dones_match_reference_and_preserve_partial_cache_without_finish():
    reference = _make_fast_stop_task(
        goto_reward_profile="fast_stop_v2",
        fast_stop_failure_remaining_time_cost=True,
        dt=0.05,
        max_time=60.0,
    )
    reference.prepare(seed=426, options={"start": [2.0, 2.0, 0.0], "goal": [4.0, 2.0]})
    _attach_empty_sensors(reference, reference.cfg)
    reference.time = 0.45
    reference._hold = 0.45
    before = reference.robot.copy()
    reference.robot[:2] = [3.8, 2.0]
    reference.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
    reference.body_angular_velocity = np.zeros(3, dtype=np.float64)
    reference.body_pose = (0.0, 0.7, 0.0)
    reference_result = reference.finish(np.zeros(4, dtype=np.float32), before, 2.0)
    assert reference_result[2] and not reference_result[3]
    stability_config = BodyStabilityConfig()
    expected_stability = body_stability_sample(
        roll=0.0, pitch=0.7, body_angular_velocity=np.zeros(3), dt=0.05,
        config=stability_config,
    )
    expected_reward = reference.reward - expected_stability["penalty"]

    cfg = _flat12_config(lidar_rays=400, depth_width=88, depth_height=160)
    env = IsaacFieldEnv.__new__(IsaacFieldEnv)
    env.task_backend = "torch_batched"
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.scene = SimpleNamespace(num_envs=2)
    env.cfg = SimpleNamespace(auto_reset=False, sim=SimpleNamespace(dt=0.001), decimation=50)
    env.field_cfg = cfg
    env.body_stability_config = stability_config
    env.tasks = [
        _TaskWithoutLegacyFinish(time=0.45, hold=0.45, xy=(3.8, 2.0)),
        _TaskWithoutLegacyFinish(done=True, time=0.2, hold=0.0, xy=(2.1, 2.0)),
    ]
    env.actions = torch.zeros((2, 4), dtype=torch.float32)
    env._tensor_goal_xy = torch.tensor([[4.0, 2.0], [4.0, 2.0]], dtype=torch.float64)
    env._tensor_episode_time = torch.tensor([0.45, 0.2], dtype=torch.float64)
    env._tensor_hold = torch.tensor([0.45, 0.0], dtype=torch.float64)
    env._tensor_initial_goal_distance = torch.tensor([2.0, 2.0], dtype=torch.float64)
    env._tensor_previous_physics_failure = torch.zeros((2,), dtype=torch.bool)
    env._tensor_last_actions = torch.zeros((2, 4), dtype=torch.float32)
    env._tensor_old_distance = torch.tensor([2.0, 2.0], dtype=torch.float64)
    env._tensor_sensor_frame_cache = {}
    env._tensor_capture_env_ids = None
    env._tensor_capture_initial_env_ids = frozenset()
    env._sensor_cache = {}
    env._state_observation_fresh = False
    env._feedback_host_cache = None
    env._finished_observation_cache_mask = None
    env._tensor_finished_observation_cache = None
    env._tensor_observation_cache = None
    env._last_results = [None, None]
    env.episode_results = []
    env._reward = torch.zeros((2,), dtype=torch.float32)
    env._build_calls = 0

    root = torch.tensor([
        [3.8, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.1, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    state = {
        "root": root,
        "origin_offsets": torch.zeros((2, 3), dtype=torch.float32),
        "angles": torch.tensor([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        "root_com_velocity_xy": torch.zeros((2, 2), dtype=torch.float32),
        "body_angular_velocity": torch.zeros((2, 3), dtype=torch.float32),
    }

    def refresh_state(*, include_feedback=False):
        env._sim_tensor_cache = state
        env._state_observation_fresh = True

    def build_observations():
        env._build_calls += 1
        return {"state": torch.full((2, 1), float(env._build_calls))}

    env._refresh_state = refresh_state
    env._refresh_feedback = lambda *, after_step: None
    env._build_tensor_observations = build_observations

    terminated, truncated = env._get_tensor_batched_dones()
    assert not terminated.any() and not truncated.any()  # auto_reset=False masks outward flags
    assert env.tasks[0].done and env.tasks[0].reason == "goal_reached"
    assert env.tasks[1].reason == "already_done"
    assert env.tasks[0].reward == pytest.approx(expected_reward, abs=2e-6)
    assert env.tasks[0].total_reward == pytest.approx(expected_reward, abs=2e-6)
    assert sum(env.tasks[0].reward_components.values()) == pytest.approx(expected_reward, abs=2e-6)
    assert env.tasks[0].body_stability_sample["penalty"] == pytest.approx(expected_stability["penalty"])
    assert env.tasks[0].sense_calls == 1 and env.tasks[1].sense_calls == 0
    assert env._last_results[0][2] is True
    assert env._last_results[1] is None
    assert env._finished_observation_cache_mask == (True, False)

    observation = env._get_tensor_batched_observations()["policy"]["state"]
    assert observation[:, 0].tolist() == [1.0, 2.0]
    assert env._build_calls == 2
    assert env._finished_observation_cache_mask is None
