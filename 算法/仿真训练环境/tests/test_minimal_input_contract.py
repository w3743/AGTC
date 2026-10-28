"""Contract regressions for the compact v6 policy observation."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaac_field.feedback import (
    FEEDBACK_AGE_SLICE,
    FEEDBACK_DIM,
    FEEDBACK_GROUPS,
    FEEDBACK_SCHEMA,
    FEEDBACK_VALID_SLICE,
    FEEDBACK_VALUE_SLICES,
    INPUT_SCHEMA,
    STATE_DIM,
    feedback_config,
)
from isaac_field.policy import (
    ENCODER_SCHEMA,
    FEATURES_DIM,
    OBSERVATION_SHAPES,
    SensorEncoder,
    _field_encoder_config,
    detection_presence_mask,
    validate_policy_spaces,
)
from isaac_field.task import STATE_FIELDS
from isaac_field.visualization.activation_maps import DETECTION_LABELS, STATE_LABELS


def _space(*, lidar=(1, 400), detections=(16, 8), state=(33,), feedback=(37,)):
    return gym.spaces.Dict({
        "lidar": gym.spaces.Box(0.0, 1.0, shape=lidar, dtype=np.float32),
        "depth": gym.spaces.Box(0.0, 1.0, shape=(1, 160, 88), dtype=np.float32),
        "detections": gym.spaces.Box(-1.0, 1.0, shape=detections, dtype=np.float32),
        "state": gym.spaces.Box(-1.0, 1.0, shape=state, dtype=np.float32),
        "feedback": gym.spaces.Box(-np.inf, np.inf, shape=feedback, dtype=np.float32),
    })


def test_v6_observation_metadata_and_field_orders_are_self_consistent():
    assert INPUT_SCHEMA == "sensor-minimal-v6"
    assert OBSERVATION_SHAPES == {
        "lidar": (1, 400),
        "depth": (1, 160, 88),
        "detections": (16, 8),
        "state": (33,),
        "feedback": (37,),
    }
    assert STATE_DIM == len(STATE_FIELDS) == len(STATE_LABELS) == 33
    assert STATE_FIELDS[:12] == (
        "mode_0_search_1_goto",
        "goal_local_x_over_field_size",
        "goal_local_y_over_field_size",
        "estimated_x_normalized",
        "estimated_y_normalized",
        "heading_sin",
        "heading_cos",
        "previous_left_track_speed",
        "previous_right_track_speed",
        "previous_gimbal_yaw_rate",
        "previous_gimbal_pitch_rate",
        "episode_time_ratio",
    )
    assert len(DETECTION_LABELS) == 8
    assert FEEDBACK_SCHEMA == "gimbal-imu-drive-feedback-v2"
    assert FEEDBACK_DIM == len(feedback_config()["labels"]) == 37
    assert FEEDBACK_GROUPS == (
        "accel", "gyro", "mag", "attitude", "gimbal_yaw", "gimbal_pitch", "motor_left", "motor_right",
    )
    assert FEEDBACK_VALUE_SLICES == {
        "accel": slice(0, 3),
        "gyro": slice(3, 6),
        "mag": slice(6, 9),
        "attitude": slice(9, 13),
        "gimbal_yaw": slice(13, 15),
        "gimbal_pitch": slice(15, 17),
        "motor_left": slice(17, 19),
        "motor_right": slice(19, 21),
    }
    assert FEEDBACK_VALID_SLICE == slice(21, 29)
    assert FEEDBACK_AGE_SLICE == slice(29, 37)
    assert not any("欧拉" in label or "温度" in label for label in feedback_config()["labels"])

    encoder_config = _field_encoder_config()
    assert ENCODER_SCHEMA == "depth64-lidar-single1d-minimal-v3"
    assert encoder_config["schema"] == ENCODER_SCHEMA
    assert encoder_config["features_dim"] == FEATURES_DIM == 192
    assert encoder_config["lidar_channels"] == ["normalized_range_invalid_zero"]
    assert encoder_config["detection_presence"] == "width>0_and_height>0"


def test_v5_observation_shapes_and_encoder_metadata_are_rejected():
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    old_v5 = _space(lidar=(2, 400), detections=(16, 10), state=(39,), feedback=(41,))
    with pytest.raises(ValueError):
        validate_policy_spaces(old_v5, action_space)

    with pytest.raises(ValueError, match="unsupported encoder_schema"):
        SensorEncoder(_space(), encoder_schema="depth64-lidar-circular1d-feedback-v2")


def test_candidate_padding_mask_uses_positive_box_geometry_only():
    detections = torch.zeros((1, 16, 8), dtype=torch.float32)
    # A real box remains present when range noise is negative and range_valid
    # says the independent measurement is unavailable.
    detections[0, 0] = torch.tensor([0.5, 0.4, 0.12, 0.08, 1.0, 0.9, -0.02, 0.0])
    # A width or height of zero is padding even if other columns are nonzero.
    detections[0, 1] = torch.tensor([0.2, 0.2, 0.0, 0.1, 1.0, 0.8, 0.2, 1.0])
    detections[0, 2] = torch.tensor([0.2, 0.2, 0.1, 0.0, 1.0, 0.8, 0.2, 1.0])
    detections[0, 3] = torch.tensor([0.3, 0.6, 0.1, 0.2, 2.0, 0.7, -0.01, 1.0])

    mask = detection_presence_mask(detections)

    assert mask.shape == (1, 16, 1)
    assert mask.dtype == torch.bool
    assert mask[0, 0, 0]
    assert not mask[0, 1, 0]
    assert not mask[0, 2, 0]
    assert mask[0, 3, 0]
    assert not mask[0, 4:, 0].any()
    assert detections[0, 0, 6] < 0 and detections[0, 0, 7] == 0
    assert detections[0, 3, 6] < 0 and detections[0, 3, 7] == 1
