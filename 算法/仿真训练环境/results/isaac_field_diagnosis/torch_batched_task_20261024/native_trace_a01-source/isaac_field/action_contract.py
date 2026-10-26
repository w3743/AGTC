"""Stable, lightweight action contract shared by training and visualization."""
from __future__ import annotations

ACTION_SCHEMA = "direct-track-speed-v1"
ACTION_COMPONENTS = (
    "left_track_speed",
    "right_track_speed",
    "gimbal_yaw_rate",
    "gimbal_pitch_rate",
)
ACTION_LABELS = (
    "左履带速度指令",
    "右履带速度指令",
    "云台偏航角速度指令",
    "云台俯仰角速度指令",
)


def action_config() -> dict[str, object]:
    """Return the JSON-safe normalized action contract stored with checkpoints."""
    return {
        "schema": ACTION_SCHEMA,
        "components": list(ACTION_COMPONENTS),
        "normalized_range": [-1.0, 1.0],
    }
