"""CPU checks for the opt-in absolute field-goal observation schema."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_field.body_stability import BodyStabilityAccumulator
from isaac_field.feedback import (
    ABSOLUTE_GOAL_INPUT_SCHEMA,
    FEEDBACK_DIM,
    INPUT_SCHEMA,
    STATE_DIM,
    feedback_config,
)
from isaac_field.task import (
    ABSOLUTE_GOAL_STATE_FIELDS,
    Config,
    FieldTask,
    STATE_FIELDS,
    state_fields_for_schema,
)


def _config(**overrides) -> Config:
    values = {
        "size": 24.0,
        "plants": 0,
        "terrain_enabled": False,
        "validate_scene": False,
        "rock_density": 0.0,
        "lidar_rays": 8,
        "depth_width": 4,
        "depth_height": 3,
        "position_noise": 0.0,
    }
    values.update(overrides)
    return Config(**values)


def _initialize_task(task: FieldTask, *, start=(2.0, 2.0, math.pi / 4), goal=(22.0, 22.0)):
    cfg = task.cfg

    def capture(kind):
        if kind == "lidar":
            return {"ranges": np.full(cfg.lidar_rays, cfg.lidar_range, np.float32),
                    "valid": np.zeros(cfg.lidar_rays, dtype=bool)}
        if kind == "depth":
            return {"values": np.full((cfg.depth_height, cfg.depth_width), cfg.depth_range, np.float32),
                    "valid": np.zeros((cfg.depth_height, cfg.depth_width), dtype=bool)}
        return {"detections": []}

    task.prepare(seed=51, options={
        "random_start": False,
        "start": list(start),
        "goal": list(goal),
    })
    task.attach_sensors(capture)
    task.initialize_sensors()
    # Make the estimated pose deterministic and explicitly exercise an angle
    # where the old body-relative feature clips before the field goal does.
    task._feedback_snapshot = SimpleNamespace(
        feedback=np.zeros(FEEDBACK_DIM, dtype=np.float32),
        pose_valid=True,
        estimated_heading_rad=math.pi / 4,
        estimated_gimbal_q_rad=np.zeros(2, dtype=np.float64),
    )
    return task


def _observation_config(cfg: Config):
    from isaac_field.env import IsaacFieldEnv

    probe = SimpleNamespace(
        field_cfg=cfg,
        tasks=[SimpleNamespace(estimated_pose_config={"source": "test"})],
        cfg=SimpleNamespace(
            imu_offset_pos=(0.0, 0.0, 0.0),
            imu_offset_rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    return IsaacFieldEnv.observation_config.fget(probe)


def test_legacy_schema_remains_the_default_and_absolute_schema_has_distinct_field_identity():
    assert Config().input_schema == INPUT_SCHEMA == "sensor-minimal-v6"
    assert state_fields_for_schema(INPUT_SCHEMA) is STATE_FIELDS
    assert state_fields_for_schema(ABSOLUTE_GOAL_INPUT_SCHEMA) == ABSOLUTE_GOAL_STATE_FIELDS
    assert len(STATE_FIELDS) == len(ABSOLUTE_GOAL_STATE_FIELDS) == STATE_DIM == 33
    assert ABSOLUTE_GOAL_STATE_FIELDS[:7] == (
        "mode_0_search_1_goto",
        "goal_world_x_normalized",
        "goal_world_y_normalized",
        "estimated_x_normalized",
        "estimated_y_normalized",
        "heading_sin",
        "heading_cos",
    )
    assert ABSOLUTE_GOAL_STATE_FIELDS[3:] == STATE_FIELDS[3:]
    assert feedback_config() == feedback_config(input_schema=INPUT_SCHEMA)
    assert feedback_config()["input_schema"] == INPUT_SCHEMA
    assert feedback_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)["input_schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    with pytest.raises(ValueError, match="unsupported input_schema"):
        feedback_config(input_schema="sensor-unknown-v0")
    with pytest.raises(ValueError, match="unsupported input_schema"):
        state_fields_for_schema("sensor-unknown-v0")


@pytest.mark.parametrize("bad_size", [float("nan"), float("inf"), 0.0, -1.0])
def test_config_rejects_nonfinite_or_nonpositive_field_size(bad_size):
    with pytest.raises(ValueError, match="size must be finite"):
        _config(size=bad_size, input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)


def test_config_and_field_task_reject_unknown_input_schema_early():
    with pytest.raises(ValueError, match="unsupported input_schema"):
        _config(input_schema="sensor-unknown-v0")

    cfg = _config()
    cfg.input_schema = "sensor-unknown-v0"
    with pytest.raises(ValueError, match="unsupported input_schema"):
        FieldTask(cfg, mode="goto")


def test_absolute_goal_uses_real_field_coordinates_and_preserves_all_other_state_values():
    legacy = _initialize_task(FieldTask(_config(input_schema=INPUT_SCHEMA), mode="goto"))
    absolute = _initialize_task(FieldTask(_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA), mode="goto"))

    legacy_observation = legacy._observation()
    absolute_observation = absolute._observation()
    legacy_state = legacy_observation["state"]
    absolute_state = absolute_observation["state"]

    np.testing.assert_allclose(absolute_state[1:3], np.asarray([22.0, 22.0]) / 24.0 * 2.0 - 1.0)
    # The legacy body-relative channel saturates at +1 here. The new channels
    # must come from task.goal, not by trying to invert that clipped value.
    np.testing.assert_allclose(legacy_state[1:3], [1.0, 0.0], atol=1e-6)
    assert np.max(np.abs(absolute_state[1:3])) < 1.0
    np.testing.assert_array_equal(absolute_state[:1], legacy_state[:1])
    np.testing.assert_array_equal(absolute_state[3:], legacy_state[3:])
    np.testing.assert_array_equal(absolute_observation["feedback"], legacy_observation["feedback"])
    assert absolute_state.shape == (STATE_DIM,)
    assert np.isfinite(absolute_state).all()
    assert np.all((-1.0 <= absolute_state) & (absolute_state <= 1.0))


def test_absolute_goal_channels_remain_zero_in_search_mode():
    task = _initialize_task(
        FieldTask(_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA), mode="search"),
    )
    observation = task._observation()
    np.testing.assert_array_equal(observation["state"][1:3], np.zeros(2, dtype=np.float32))


def test_environment_contract_names_absolute_goal_frame_scale_and_legacy_default():
    legacy = _observation_config(_config())
    absolute = _observation_config(_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA))

    assert legacy["schema"] == INPUT_SCHEMA
    assert legacy["state_fields"] == list(STATE_FIELDS)
    assert legacy["feedback"] == feedback_config()
    assert "absolute_goal_coordinates" not in legacy

    assert absolute["schema"] == ABSOLUTE_GOAL_INPUT_SCHEMA
    assert absolute["state_dim"] == STATE_DIM
    assert absolute["state_fields"] == list(ABSOLUTE_GOAL_STATE_FIELDS)
    assert absolute["observation_shapes"]["state"] == [STATE_DIM]
    assert absolute["observation_shapes"]["feedback"] == [FEEDBACK_DIM]
    assert absolute["feedback"] == feedback_config(input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
    assert absolute["absolute_goal_coordinates"] == {
        "source": "task.goal",
        "coordinate_frame": "field_world_xy",
        "origin_xy_m": [0.0, 0.0],
        "unit": "m",
        "field_size_m": 24.0,
        "normalization": "world_xy / field_size_m * 2 - 1",
        "state_clip": [-1.0, 1.0],
    }


def test_absolute_goal_terminal_snapshot_and_single_adapter_share_the_task_observation():
    import torch

    from isaac_field.env import IsaacFieldEnv, SingleFieldAdapter

    cfg = _config(
        dt=0.05,
        max_time=0.05,
        input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
    )
    task = _initialize_task(FieldTask(cfg, mode="goto"), start=(2.0, 2.0, 0.0), goal=(20.0, 18.0))
    task.body_stability_stats = BodyStabilityAccumulator()
    task.body_stability_sample = None

    class MockNativeEnv:
        _get_dones = IsaacFieldEnv._get_dones
        observation_config = _observation_config(cfg)

        def _refresh_state(self, *, include_feedback=False):
            pass

        def _refresh_feedback(self, *, after_step):
            pass

    engine = MockNativeEnv()
    engine.tasks = [task]
    engine.actions = torch.zeros((1, 4), dtype=torch.float32)
    engine._before = [task.robot.copy()]
    engine._old_distance = [float(np.linalg.norm(task.goal - task.robot[:2]))]
    engine._last_results = [None]
    engine._reward = None
    engine.device = "cpu"
    engine.field_cfg = cfg
    engine.cfg = SimpleNamespace(auto_reset=False, distance_cost=0.0)
    engine.step_dt = cfg.dt
    engine.body_stability_config = None
    engine.episode_results = []

    terminated, truncated = IsaacFieldEnv._get_dones(engine)
    assert terminated.tolist() == [False]
    assert truncated.tolist() == [False]  # auto_reset=False keeps reset flags external
    cached_observation = engine._last_results[0][0]
    np.testing.assert_allclose(cached_observation["state"][1:3], np.asarray(task.goal) / cfg.size * 2 - 1)
    assert engine._last_results[0][3] is True

    adapter = object.__new__(SingleFieldAdapter)
    adapter.engine = engine
    assert adapter.observation_config is engine.observation_config
    current_observation = adapter.current_observation()
    np.testing.assert_allclose(current_observation["state"][1:3], cached_observation["state"][1:3])
