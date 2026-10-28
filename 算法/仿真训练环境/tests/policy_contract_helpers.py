"""Small SB3-independent helpers shared by field policy contract tests."""
from __future__ import annotations

from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from isaac_field.task import state_fields_for_schema


def observation_config(input_schema: str) -> dict:
    return {
        "schema": input_schema,
        "state_dim": 33,
        "state_fields": list(state_fields_for_schema(input_schema)),
        "observation_shapes": {
            "lidar": [1, 400],
            "depth": [1, 160, 88],
            "detections": [16, 8],
            "state": [33],
            "feedback": [37],
        },
        "feedback": {"input_schema": input_schema},
        **(
            {
                "absolute_goal_coordinates": {
                    "source": "task.goal",
                    "coordinate_frame": "field_world_xy",
                    "origin_xy_m": [0.0, 0.0],
                    "unit": "m",
                    "field_size_m": 100.0,
                    "normalization": "world_xy / field_size_m * 2 - 1",
                    "state_clip": [-1.0, 1.0],
                }
            }
            if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA
            else {}
        ),
    }
