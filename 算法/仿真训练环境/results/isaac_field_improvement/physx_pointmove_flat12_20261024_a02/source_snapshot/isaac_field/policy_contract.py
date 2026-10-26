"""Pure policy input/encoder contracts shared by SB3 and RSL-RL adapters."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .action_contract import action_config
from .feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, FEEDBACK_DIM, INPUT_SCHEMA, STATE_DIM


OBSERVATION_SHAPES = {
    "lidar": (1, 400),
    "depth": (1, 160, 88),
    "detections": (16, 8),
    "state": (STATE_DIM,),
    "feedback": (FEEDBACK_DIM,),
}
ACTION_SHAPE = (4,)
ENCODER_SCHEMA = "depth64-lidar-single1d-minimal-v3"
ABSOLUTE_GOAL_ENCODER_SCHEMA = "depth64-lidar-single1d-absolute-goal-v4"
BRANCH_DIMS = {"lidar": 32, "depth": 64, "detections": 32, "state": 32, "feedback": 32}
FEATURES_DIM = sum(BRANCH_DIMS.values())

# New protocol/checkpoint identities.  Historical RSL checkpoints predate
# these fields and remain readable as historical artifacts; active PhysX
# evaluation must compare all three fields before it uses a policy.
BACKEND_IDENTITY = {
    "simulator": "isaac_sim",
    "physics_backend": "physx",
    "learning_backend": "rsl_rl",
}
BACKEND_IDENTITY_KEYS = tuple(BACKEND_IDENTITY)


def validate_backend_identity(value: Mapping[str, Any], *, allow_legacy: bool = False) -> dict[str, str] | None:
    """Validate the explicit simulator/physics/learner identity in new artifacts."""
    if value is None and allow_legacy:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("backend_identity must be a mapping")
    identity = {key: value.get(key) for key in BACKEND_IDENTITY_KEYS}
    if identity != BACKEND_IDENTITY:
        raise ValueError(
            "backend_identity must identify Isaac Sim + PhysX + RSL-RL; "
            f"got {identity!r}"
        )
    return identity


def recorded_backend_identity(value: Mapping[str, Any]) -> dict[str, str]:
    """Read a historical identity for explicit lineage without endorsing it.

    New active artifacts must use :func:`validate_backend_identity`. This helper
    is only for recording the source side of an explicit weights transfer.
    """
    if not isinstance(value, Mapping):
        raise ValueError("recorded backend identity must be a mapping")
    present = [value.get(key) is not None for key in BACKEND_IDENTITY_KEYS]
    if not any(present):
        return {
            "simulator": "legacy_unrecorded",
            "physics_backend": "legacy_unrecorded",
            "learning_backend": "legacy_unrecorded",
        }
    identity = {key: value.get(key) for key in BACKEND_IDENTITY_KEYS}
    if not all(present) or any(not isinstance(item, str) or not item.strip() for item in identity.values()):
        raise ValueError("recorded backend identity must have three non-empty string fields")
    return identity


def stamp_backend_identity(value: dict[str, Any]) -> dict[str, Any]:
    """Add the active backend identity to a newly prepared artifact in place."""
    if not isinstance(value, dict):
        raise TypeError("new protocol must be a dictionary")
    existing = {key: value[key] for key in BACKEND_IDENTITY_KEYS if key in value}
    if existing and existing != BACKEND_IDENTITY:
        raise ValueError(f"cannot relabel an artifact prepared for another backend: {existing!r}")
    value.update(BACKEND_IDENTITY)
    return value


def require_backend_identity(value: Mapping[str, Any]) -> dict[str, str]:
    """Reject legacy or differently targeted protocol files before launch."""
    if not isinstance(value, Mapping):
        raise TypeError("protocol must be a mapping")
    identity = {key: value.get(key) for key in BACKEND_IDENTITY_KEYS}
    return validate_backend_identity(identity)


def _encoder_schema_for_input(input_schema: str) -> str:
    if input_schema == INPUT_SCHEMA:
        return ENCODER_SCHEMA
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        return ABSOLUTE_GOAL_ENCODER_SCHEMA
    raise ValueError(
        f"unsupported input_schema {input_schema!r}; expected {INPUT_SCHEMA!r} or "
        f"{ABSOLUTE_GOAL_INPUT_SCHEMA!r}"
    )


def _state_fields_for_schema(input_schema: str) -> tuple[str, ...]:
    # Keep task imports lazy so loading the policy contract does not pull in
    # simulator/task modules until state-field validation actually needs them.
    from .task import state_fields_for_schema

    return tuple(state_fields_for_schema(input_schema))


def _field_encoder_config(input_schema: str = INPUT_SCHEMA) -> dict[str, Any]:
    """Return the JSON-safe encoder description stored with each checkpoint."""
    encoder_schema = _encoder_schema_for_input(input_schema)
    config = {
        "schema": encoder_schema,
        "branch_dims": dict(BRANCH_DIMS),
        "depth_pool": [8, 6],
        "radar_pool": 50,
        "lidar_channels": ["normalized_range_invalid_zero"],
        "detection_fields": ["cx", "cy", "width", "height", "class_id", "confidence", "range", "range_valid"],
        "detection_presence": "width>0_and_height>0",
        "features_dim": FEATURES_DIM,
    }
    if input_schema != INPUT_SCHEMA:
        config["input_schema"] = input_schema
        config["state_fields"] = list(_state_fields_for_schema(input_schema))
    return config


def _json_safe(value: Any) -> Any:
    """Normalize observation metadata for stable checkpoint comparisons."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_observation_config(
    observation_config: Mapping[str, Any],
    input_schema: str,
    *,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the same-shaped state semantics represented in checkpoint metadata."""
    _encoder_schema_for_input(input_schema)
    config = _json_safe(observation_config)
    if not isinstance(config, dict):
        raise ValueError("observation_config must be a mapping")
    expected_fields = list(_state_fields_for_schema(input_schema))
    if config.get("schema") != input_schema:
        raise ValueError(
            f"observation_config schema {config.get('schema')!r} does not match input_schema {input_schema!r}"
        )
    if config.get("state_dim") != STATE_DIM or config.get("state_fields") != expected_fields:
        raise ValueError(
            f"observation_config state fields do not match {input_schema} ({STATE_DIM} fields)"
        )
    shapes = config.get("observation_shapes")
    if not isinstance(shapes, dict) or shapes.get("state") != [STATE_DIM]:
        raise ValueError("observation_config observation_shapes must declare state as [33]")
    feedback = config.get("feedback")
    if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        if not isinstance(feedback, dict) or feedback.get("input_schema") != input_schema:
            raise ValueError("absolute-goal observation_config must bind feedback to its input schema")
        coordinates = config.get("absolute_goal_coordinates")
        if not isinstance(coordinates, dict):
            raise ValueError("absolute-goal observation_config must declare absolute_goal_coordinates")
        fixed_coordinates = {
            "source": "task.goal",
            "coordinate_frame": "field_world_xy",
            "origin_xy_m": [0.0, 0.0],
            "unit": "m",
            "normalization": "world_xy / field_size_m * 2 - 1",
            "state_clip": [-1.0, 1.0],
        }
        mismatched_coordinates = {
            key: (coordinates.get(key), expected)
            for key, expected in fixed_coordinates.items()
            if coordinates.get(key) != expected
        }
        if mismatched_coordinates:
            details = ", ".join(
                f"{key}={actual!r} (expected {expected!r})"
                for key, (actual, expected) in mismatched_coordinates.items()
            )
            raise ValueError(f"absolute_goal_coordinates contract is missing or incompatible: {details}")
        field_size = coordinates.get("field_size_m")
        try:
            numeric_field_size = float(field_size)
        except (TypeError, ValueError, OverflowError):
            numeric_field_size = math.nan
        if (
            isinstance(field_size, bool)
            or not isinstance(field_size, (int, float))
            or not math.isfinite(numeric_field_size)
            or numeric_field_size < 6.0
        ):
            raise ValueError("absolute_goal_coordinates field_size_m must be finite and at least 6 m")
    if expected_observation_config is not None:
        expected = _json_safe(expected_observation_config)
        if not isinstance(expected, dict) or expected != config:
            raise ValueError("observation_config does not match expected_observation_config")
    return config


def _observation_config_is_complete(observation_config: Any) -> bool:
    return (
        isinstance(observation_config, Mapping)
        and "schema" in observation_config
        and "state_dim" in observation_config
        and "state_fields" in observation_config
        and isinstance(observation_config.get("observation_shapes"), Mapping)
        and all(key in observation_config["observation_shapes"] for key in OBSERVATION_SHAPES)
    )


def _validate_legacy_partial_observation_config(observation_config: Mapping[str, Any]) -> None:
    """Keep older v6 callers working when they stored only a schema marker."""
    config = _json_safe(observation_config)
    if not isinstance(config, dict):
        raise ValueError("observation_config must be a mapping")
    if "schema" in config and config["schema"] != INPUT_SCHEMA:
        raise ValueError(f"observation_config schema does not match legacy input_schema {INPUT_SCHEMA!r}")
    expected_fields = list(_state_fields_for_schema(INPUT_SCHEMA))
    if "state_dim" in config and config["state_dim"] != STATE_DIM:
        raise ValueError("legacy observation_config state_dim does not match 33 fields")
    if "state_fields" in config and config["state_fields"] != expected_fields:
        raise ValueError("legacy observation_config state fields do not match sensor-minimal-v6")
    shapes = config.get("observation_shapes")
    if shapes is not None and (not isinstance(shapes, dict) or shapes.get("state") != [STATE_DIM]):
        raise ValueError("legacy observation_config observation_shapes must declare state as [33]")


def validate_observation_space(observation_space) -> None:
    """Validate the five sensor tensor dimensions without depending on an RL library."""
    spaces = getattr(observation_space, "spaces", None)
    if spaces is None:
        raise ValueError("policy expects a Dict observation space from Sb3VecEnvWrapper")
    missing = set(OBSERVATION_SHAPES) - set(spaces)
    if missing:
        raise ValueError(f"observation space is missing keys: {sorted(missing)}")
    mismatched = {
        key: (tuple(spaces[key].shape), expected)
        for key, expected in OBSERVATION_SHAPES.items()
        if tuple(spaces[key].shape) != expected
    }
    if mismatched:
        details = ", ".join(
            f"{key}={actual} (expected {expected})" for key, (actual, expected) in mismatched.items()
        )
        raise ValueError(
            f"observation dimensions do not match the field policy contract: {details}. "
            "The current contract requires one-channel LiDAR range, single-channel depth, 16x8 detections, "
            "33 task fields and 37 feedback fields; older sensor contracts require retraining."
        )


def validate_policy_spaces(
    observation_space,
    action_space=None,
    *,
    input_schema: str = INPUT_SCHEMA,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> None:
    """Validate field observation and optional normalized continuous action spaces."""
    _encoder_schema_for_input(input_schema)
    validate_observation_space(observation_space)
    spaces = observation_space.spaces
    if expected_observation_config is not None:
        config = _validate_observation_config(expected_observation_config, input_schema)
        configured_shapes = config.get("observation_shapes", {})
        shape_mismatches = {
            key: (configured_shapes.get(key), list(spaces[key].shape))
            for key in OBSERVATION_SHAPES
            if configured_shapes.get(key) != list(spaces[key].shape)
        }
        if shape_mismatches:
            details = ", ".join(
                f"{key}={actual} (observation_config declares {declared})"
                for key, (declared, actual) in shape_mismatches.items()
            )
            raise ValueError(f"observation_config dimensions do not match policy spaces: {details}")
    if action_space is not None:
        shape = tuple(getattr(action_space, "shape", ()))
        if shape != ACTION_SHAPE:
            raise ValueError(f"action space shape {shape} does not match {ACTION_SHAPE}")
        if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
            raise ValueError("action space must be a continuous Box with finite normalized bounds [-1, 1]")
        low = np.asarray(action_space.low, dtype=np.float64)
        high = np.asarray(action_space.high, dtype=np.float64)
        normalized_low, normalized_high = action_config()["normalized_range"]
        expected_low = np.full(ACTION_SHAPE, normalized_low, dtype=np.float64)
        expected_high = np.full(ACTION_SHAPE, normalized_high, dtype=np.float64)
        if (
            not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or not np.array_equal(low, expected_low)
            or not np.array_equal(high, expected_high)
        ):
            raise ValueError(
                "action space bounds must be finite and match the action contract's normalized range [-1, 1]"
            )
