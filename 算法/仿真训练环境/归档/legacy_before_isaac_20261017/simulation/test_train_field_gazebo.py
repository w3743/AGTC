"""Runner contracts that do not launch Gazebo."""
from __future__ import annotations

import json
import socket
import time
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from field_sim.env import Config, FieldEnv
from simulation.gazebo_monitor import TrainingMonitor
from simulation.train_field_gazebo import (MetricsWrapper, MixedTaskWrapper, _check_monitor_port_available,
                                           _parse_args, _benchmark_paced, _task_mode_coverage, _training_env,
                                           _physics_runtime_settings)


def test_cli_defaults_to_isolated_mixed_gazebo_run_with_absolute_output():
    args = _parse_args([])

    assert args.command == "all"
    assert args.backend == "gazebo"
    assert args.mode == "mixed"
    assert args.steps == 512
    assert args.n_steps == 128
    assert args.output.is_absolute()
    assert args.monitor_port is None


def test_mixed_reset_wrapper_selects_modes_and_samples_nearby_navigable_goals():
    cfg = Config(size=20, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=2, depth_height=2, max_detections=2)
    base = FieldEnv(cfg)
    env = MixedTaskWrapper(base, seed=7)
    modes = []
    try:
        for seed in range(40):
            observation, _ = env.reset(seed=seed)
            mode = base.mode
            modes.append(mode)
            assert set(observation) == {"lidar", "depth", "detections", "state"}
            assert base.observation_space.contains(observation)
            if mode == "goto":
                distance = float(np.linalg.norm(base.goal - base.robot[:2]))
                assert 1.0 <= distance <= 5.0
                assert base.scene.navigable(base.goal)
        assert set(modes[:2]) == {"goto", "search"}
        assert "goto" in modes and "search" in modes
        assert 0.4 <= modes.count("goto") / len(modes) <= 0.8
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["mixed", "search", "goto"])
def test_training_env_enables_random_spawn_for_mock_gazebo_backend(tmp_path, monkeypatch, mode):
    class MockGazeboEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "state": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)})
            self.mode = "search"
            self.robot = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
            self.scene = SimpleNamespace(navigable=lambda point: True)
            self.reset_options = None

        def reset(self, *, seed=None, options=None):
            self.reset_options = dict(options or {})
            self.mode = self.reset_options.get("mode", self.mode)
            return self._observation(), {}

        def _observation(self):
            return {"state": np.zeros(1, dtype=np.float32)}

        def _info(self):
            return {}

        def set_goal(self, goal):
            self.goal = np.asarray(goal, dtype=np.float64)

    base = MockGazeboEnv()
    monkeypatch.setattr(
        "simulation.train_field_gazebo._make_env",
        lambda cfg, requested_mode, **kwargs: base)
    output = tmp_path / mode
    output.mkdir()
    args = SimpleNamespace(mode=mode, seed=17, backend="gazebo", monitor=None, steps=32)
    _, measured, monitored = _training_env(Config(size=12, plants=0, terrain_enabled=False,
                                                  validate_scene=False), args, output)
    try:
        monitored.reset(seed=29)
        assert base.reset_options["random_start"] is True
        assert base.reset_options["mode"] in ("search", "goto")
        if mode != "mixed":
            assert base.reset_options["mode"] == mode
    finally:
        measured.close()


def test_explicit_training_start_suppresses_automatic_random_start():
    class ResetCaptureEnv(gym.Env):
        action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        observation_space = spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)})

        def reset(self, *, seed=None, options=None):
            self.reset_options = dict(options or {})
            self.mode = self.reset_options.get("mode", "search")
            self.robot = np.array([5.0, 5.0, 0.0])
            self.scene = SimpleNamespace(navigable=lambda point: True)
            return {"state": np.zeros(1, dtype=np.float32)}, {}

        def _observation(self):
            return {"state": np.zeros(1, dtype=np.float32)}

        def _info(self):
            return {}

    base = ResetCaptureEnv()
    wrapper = MixedTaskWrapper(base, seed=2, task_mode="search", random_start=True)
    wrapper.reset(options={"start": [4.0, 4.0, 0.0]})
    assert "random_start" not in base.reset_options


def test_metrics_wrapper_records_actual_sensor_progress_and_sim_time():
    cfg = Config(size=20, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=2, depth_height=2, max_detections=2,
                 lidar_relative_error=0.0, range_noise=0.0)
    measured = MetricsWrapper(FieldEnv(cfg), phase="benchmark", total_steps=4)
    try:
        obs, info = measured.reset(seed=3)
        for _ in range(4):
            obs, _, terminated, truncated, info = measured.step(np.zeros(4, np.float32))
            assert not terminated and not truncated
        assert measured.step_count == 4
        assert measured.sim_step_seconds == [cfg.dt] * 4
        assert measured.sensor_fresh_counts["lidar"] > 0
        assert measured.last_sensor_snapshot["sequences"]["lidar"] > 1
        assert measured.last_sensor_snapshot["frame_age_seconds"]["lidar"] >= 0
        assert len(measured.step_ms) == 4
    finally:
        measured.close()


def test_metrics_wrapper_publishes_read_only_training_state_from_same_env():
    cfg = Config(size=8, plants=0, terrain_enabled=False, validate_scene=False,
                 lidar_rays=8, depth_width=2, depth_height=2, max_detections=2)
    base = FieldEnv(cfg)
    monitor = TrainingMonitor(port=0)
    env = MetricsWrapper(base, monitor=monitor, phase="training", total_steps=4)
    url = "http://127.0.0.1:" + str(monitor.server.server_address[1])
    try:
        env.reset(seed=9)
        time.sleep(0.21)  # TrainingMonitor throttles ordinary step updates to 5 Hz.
        env.step(np.zeros(4, np.float32))
        with urlopen(url + "/api/state") as response:
            state = json.load(response)
        assert state["read_only"] is True
        assert state["training"]["phase"] == "training"
        assert state["training"]["same_environment"] is True
        assert state["training"]["steps"] == 1
    finally:
        env.close()
        monitor.close()


def test_monitor_port_preflight_rejects_an_already_bound_port():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        busy_port = server.getsockname()[1]
        with pytest.raises(OSError, match="already in use or unavailable"):
            _check_monitor_port_available(busy_port)
    finally:
        server.close()


def test_physics_summary_reads_the_runtime_sdf_values(tmp_path):
    runtime_sdf = tmp_path / "runtime.sdf"
    runtime_sdf.write_text(
        "<sdf><world name='field'><physics><max_step_size>0.005</max_step_size>"
        "<real_time_factor>0</real_time_factor><real_time_update_rate>0</real_time_update_rate>"
        "</physics></world></sdf>", encoding="utf-8")
    values = _physics_runtime_settings(SimpleNamespace(step_size=0.005, _runtime_sdf=runtime_sdf))

    assert values == {"physics_step_size_s": 0.005, "real_time_factor": 0.0,
                      "real_time_update_rate_hz": 0.0,
                      "physics_settings_source": "runtime_sdf"}


def test_paced_control_target_rate_comes_from_decision_dt():
    cfg = Config(size=12, plants=0, dt=0.1, max_time=10, terrain_enabled=False,
                 validate_scene=False, lidar_rays=8, depth_width=2, depth_height=2,
                 max_detections=2)
    env = MetricsWrapper(FieldEnv(cfg))

    class ZeroRunner:
        def reset(self):
            pass

        def action(self, observation):
            return np.zeros(4, dtype=np.float32)

    try:
        result = _benchmark_paced(
            env, ZeroRunner(), cfg, SimpleNamespace(paced_seconds=0.1, seed=7))
        assert result["target_hz"] == 1.0 / cfg.dt
        assert result["steps"] == 2
    finally:
        env.close()


def test_all_run_reports_smoke_coverage_separately_from_actual_training_modes():
    coverage = _task_mode_coverage(
        {"tasks": [{"mode": "search"}, {"mode": "goto"}]},
        {"episode_modes": ["search", "search"]})

    assert coverage["training_episode_modes"] == ["search", "search"]
    assert coverage["training_rollout_covers_search_and_goto"] is False
    assert coverage["smoke_self_check_covers_search_and_goto"] is True
    assert coverage["combined_observed_modes"] == ["goto", "search"]
