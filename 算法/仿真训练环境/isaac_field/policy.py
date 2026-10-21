"""SB3 recurrent policies for the Isaac Lab field environment.

The v6 relative-goal and v1 absolute-goal inputs share the same 33-field shape,
so checkpoints bind both the input and encoder schemas. Sensor inputs use one
depth channel, one LiDAR range channel and detections whose presence is encoded
by positive box dimensions. The action contract maps its first two outputs to
independent left/right track speeds. Incompatible contracts require retraining;
loading never silently reshapes or reinterprets weights.
This module is self contained and does not depend on ``field_sim``.
"""
from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .action_contract import ACTION_SCHEMA, action_config
from .feedback import (
    ABSOLUTE_GOAL_INPUT_SCHEMA,
    FEEDBACK_DIM,
    INPUT_SCHEMA,
    STATE_DIM,
)


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
    from .task import state_fields_for_schema

    return tuple(state_fields_for_schema(input_schema))


def _field_encoder_config(
    input_schema: str = INPUT_SCHEMA,
) -> dict[str, Any]:
    """Return the JSON-safe description stored alongside each new checkpoint."""
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
    if shapes is not None and (
        not isinstance(shapes, dict) or shapes.get("state") != [STATE_DIM]
    ):
        raise ValueError("legacy observation_config observation_shapes must declare state as [33]")


def _environment_input_schema(env) -> str | None:
    """Find schema markers on native envs and the small adapter wrappers used here."""
    pending = [env]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        observation_config = getattr(candidate, "observation_config", None)
        if isinstance(observation_config, Mapping):
            schema = observation_config.get("schema")
            if schema in (INPUT_SCHEMA, ABSOLUTE_GOAL_INPUT_SCHEMA):
                return schema
        for name in ("cfg", "field_cfg"):
            config = getattr(candidate, name, None)
            schema = getattr(config, "input_schema", None)
            if schema in (INPUT_SCHEMA, ABSOLUTE_GOAL_INPUT_SCHEMA):
                return schema
        for name in ("engine", "env", "unwrapped", "task"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
        children = getattr(candidate, "envs", None)
        if isinstance(children, (list, tuple)):
            pending.extend(children)
    return None


def _resolve_observation_config(
    env,
    input_schema: str,
    observation_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Select and validate schema metadata before model construction or loading."""
    _encoder_schema_for_input(input_schema)
    env_schema = _environment_input_schema(env)
    if env_schema is not None and env_schema != input_schema:
        raise ValueError(
            f"environment input_schema {env_schema!r} does not match requested {input_schema!r}"
        )
    env_config = getattr(env, "observation_config", None)
    if observation_config is None:
        observation_config = env_config
        if (
            input_schema == INPUT_SCHEMA
            and observation_config is not None
            and not _observation_config_is_complete(observation_config)
        ):
            if not isinstance(observation_config, Mapping):
                raise ValueError("observation_config must be a mapping")
            _validate_legacy_partial_observation_config(observation_config)
            return None
    elif env_config is not None and _json_safe(observation_config) != _json_safe(env_config):
        raise ValueError("observation_config does not match env.observation_config")
    if observation_config is None:
        if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            raise ValueError(
                "absolute-goal policies require an explicit observation_config or env.observation_config"
            )
        return None
    if not isinstance(observation_config, Mapping):
        raise ValueError("observation_config must be a mapping")
    return _validate_observation_config(observation_config, input_schema)


def detection_presence_mask(value: torch.Tensor) -> torch.Tensor:
    """Return a singleton-channel mask for boxes with positive width and height."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("detection_presence_mask expects a torch.Tensor")
    if value.ndim < 2 or value.shape[-1] != 8:
        raise ValueError(f"detections must end in 8 fields, got shape {tuple(value.shape)}")
    return (value[..., 2:3] > 0) & (value[..., 3:4] > 0)


class SensorEncoder(BaseFeaturesExtractor):
    """Encode five sensor/task branches under an explicit input-state schema."""

    branch_dims = BRANCH_DIMS

    def __init__(
        self,
        observation_space,
        cache_inference: bool = True,
        goal_gain: float = 1.0,
        encoder_schema: str | None = None,
        input_schema: str = INPUT_SCHEMA,
    ):
        validate_policy_spaces(observation_space)
        super().__init__(observation_space, features_dim=FEATURES_DIM)
        goal_gain = float(goal_gain)
        if not math.isfinite(goal_gain) or goal_gain != 1.0:
            raise ValueError("goal_gain must be 1.0: extra goal-input scaling has been removed")
        expected_encoder_schema = _encoder_schema_for_input(input_schema)
        if encoder_schema is None:
            encoder_schema = expected_encoder_schema
        if encoder_schema != expected_encoder_schema:
            raise ValueError(
                f"unsupported encoder_schema {encoder_schema!r} for {input_schema!r}; "
                f"expected {expected_encoder_schema!r}"
            )
        self.cache_inference = bool(cache_inference)
        self.goal_gain = goal_gain
        self.encoder_schema = encoder_schema
        self.input_schema = input_schema
        self.branch_dims = dict(BRANCH_DIMS)
        self.laser = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50),
            nn.Flatten(),
            nn.Linear(32 * 50, self.branch_dims["lidar"]),
            nn.ReLU(),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 6)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 6, self.branch_dims["depth"]),
            nn.ReLU(),
        )
        self.boxes = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.proprioception = nn.Sequential(nn.Linear(observation_space["state"].shape[0], 32), nn.ReLU())
        self.feedback = nn.Sequential(nn.Linear(FEEDBACK_DIM, 32), nn.ReLU())
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def train(self, mode: bool = True):
        # Cached rollout embeddings cannot survive a parameter update.
        if mode:
            self._cache.clear()
        return super().train(mode)

    def _branch(self, name: str, value: torch.Tensor, function: nn.Module) -> torch.Tensor:
        caching = (
            self.cache_inference
            and not self.training
            and not torch.is_grad_enabled()
            and value.shape[0] == 1
        )
        if caching:
            cached = self._cache.get(name)
            if cached is not None and cached[0].device == value.device and torch.equal(cached[0], value):
                return cached[1]
        result = function(value)
        if caching:
            self._cache[name] = (value.detach().clone(), result.detach())
        return result

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        def encode_boxes(value: torch.Tensor) -> torch.Tensor:
            features = self.boxes(value) * detection_presence_mask(value).to(dtype=value.dtype)
            return features.amax(dim=1)

        return torch.cat(
            (
                self._branch("lidar", obs["lidar"], self.laser),
                self._branch("depth", obs["depth"], self.depth),
                self._branch("detections", obs["detections"], encode_boxes),
                self.proprioception(obs["state"]),
                self.feedback(obs["feedback"]),
            ),
            dim=1,
        )


def validate_policy_spaces(
    observation_space,
    action_space=None,
    *,
    input_schema: str = INPUT_SCHEMA,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> None:
    """Fail early if an env does not expose the migrated sensor contract."""
    _encoder_schema_for_input(input_schema)
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
        details = ", ".join(f"{key}={actual} (expected {expected})" for key, (actual, expected) in mismatched.items())
        raise ValueError(
            f"observation dimensions do not match the field policy contract: {details}. "
            "The current contract requires one-channel LiDAR range, single-channel depth, 16x8 detections, "
            "33 task fields and 37 feedback fields; "
            "older sensor contracts require retraining."
        )
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


def make_model(
    env,
    *,
    input_schema: str = INPUT_SCHEMA,
    observation_config: Mapping[str, Any] | None = None,
    seed: int = 7,
    n_steps: int = 128,
    verbose: int = 0,
    goal_gain: float = 1.0,
    encoder_schema: str | None = None,
    batch_size: int = 64,
    n_epochs: int = 3,
    learning_rate: float = 3e-4,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
    n_env: int | None = None,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    log_std_init: float = 0.0,
    squash_output: bool = False,
    target_kl: float | None = None,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    net_arch: dict[str, list[int]] | None = None,
    device: str | torch.device = "cpu",
    tensorboard_log: str | Path | None = None,
):
    """Build the schema-bound sensor encoder with SB3-Contrib RecurrentPPO."""
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    expected_encoder_schema = _encoder_schema_for_input(input_schema)
    if encoder_schema is None:
        encoder_schema = expected_encoder_schema
    elif encoder_schema != expected_encoder_schema:
        raise ValueError(
            f"encoder_schema {encoder_schema!r} does not match input_schema {input_schema!r}; "
            f"expected {expected_encoder_schema!r}"
        )
    resolved_observation_config = _resolve_observation_config(env, input_schema, observation_config)
    actual_n_env = int(getattr(env, "num_envs", 1))
    if n_env is not None and int(n_env) != actual_n_env:
        raise ValueError(f"n_env={n_env} does not match environment num_envs={actual_n_env}")
    validate_policy_spaces(
        env.observation_space,
        env.action_space,
        input_schema=input_schema,
        expected_observation_config=resolved_observation_config,
    )
    if not math.isfinite(float(log_std_init)):
        raise ValueError("log_std_init must be finite")
    if int(sde_sample_freq) == 0 or int(sde_sample_freq) < -1:
        raise ValueError("sde_sample_freq must be -1 or a positive integer")
    if target_kl is not None and (not math.isfinite(float(target_kl)) or float(target_kl) <= 0):
        raise ValueError("target_kl must be finite and positive when provided")
    if int(n_steps) < 2 or int(batch_size) < 1 or int(n_epochs) < 1:
        raise ValueError("n_steps must be >= 2 and batch_size/n_epochs must be positive")
    if int(batch_size) > int(n_steps) * actual_n_env:
        raise ValueError("batch_size cannot exceed n_steps * num_envs")
    if net_arch is None:
        net_arch = {"pi": [64], "vf": [64]}
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        seed=seed,
        device=device,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        verbose=int(verbose),
        use_sde=bool(use_sde),
        sde_sample_freq=int(sde_sample_freq),
        target_kl=target_kl,
        tensorboard_log=None if tensorboard_log is None else str(tensorboard_log),
        policy_kwargs={
            "features_extractor_class": SensorEncoder,
            "features_extractor_kwargs": {
                "goal_gain": float(goal_gain),
                "encoder_schema": encoder_schema,
                "input_schema": input_schema,
            },
            "lstm_hidden_size": int(lstm_hidden_size),
            "n_lstm_layers": int(n_lstm_layers),
            "net_arch": net_arch,
            "log_std_init": float(log_std_init),
            "squash_output": bool(squash_output),
            "normalize_images": False,
        },
    )
    model.field_encoder_config = _field_encoder_config(input_schema)
    if resolved_observation_config is not None:
        model.field_observation_config = resolved_observation_config
    model.field_action_config = action_config()
    return model


def _validate_model_encoder_contract(
    model,
    *,
    input_schema: str = INPUT_SCHEMA,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> None:
    """Reject checkpoints whose saved or instantiated encoder contract is not current."""
    expected_encoder_schema = _encoder_schema_for_input(input_schema)
    policy_kwargs = getattr(model, "policy_kwargs", {})
    extractor_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    if (
        policy_kwargs.get("features_extractor_class") is not SensorEncoder
        or not isinstance(extractor_kwargs, dict)
        or extractor_kwargs.get("encoder_schema") != expected_encoder_schema
        or extractor_kwargs.get("input_schema", INPUT_SCHEMA) != input_schema
        or not isinstance(extractor, SensorEncoder)
        or getattr(extractor, "encoder_schema", None) != expected_encoder_schema
        or getattr(extractor, "input_schema", INPUT_SCHEMA) != input_schema
        or int(getattr(extractor, "features_dim", -1)) != FEATURES_DIM
        or getattr(model, "field_encoder_config", None) != _field_encoder_config(input_schema)
    ):
        raise ValueError(
            "Checkpoint sensor encoder is missing or does not match "
            f"{expected_encoder_schema} (features_dim={FEATURES_DIM}); "
            "this structure requires a newly trained checkpoint and old weights are not automatically converted."
        )
    model_observation_config = getattr(model, "field_observation_config", None)
    if model_observation_config is None:
        if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            raise ValueError(
                "Checkpoint field_observation_config is missing or incompatible; "
                "absolute-goal checkpoints require matching state-field metadata."
            )
    else:
        if not isinstance(model_observation_config, Mapping):
            raise ValueError("Checkpoint field_observation_config is missing or incompatible")
        if (
            input_schema == INPUT_SCHEMA
            and not _observation_config_is_complete(model_observation_config)
        ):
            _validate_legacy_partial_observation_config(model_observation_config)
        else:
            _validate_observation_config(
                model_observation_config,
                input_schema,
                expected_observation_config=expected_observation_config,
            )


def _validate_model_action_contract(model) -> None:
    """Reject models whose four outputs have missing or obsolete meanings."""
    if getattr(model, "field_action_config", None) != action_config():
        raise ValueError(
            "Checkpoint action contract is missing or incompatible with "
            f"{ACTION_SCHEMA}; this action contract requires a newly trained checkpoint because its "
            "left/right track-speed meanings changed, "
            "and old weights are not automatically converted."
        )


def _read_checkpoint_metadata(checkpoint: str | Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read("data"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{checkpoint} is not a readable SB3 checkpoint") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{checkpoint} does not contain an SB3 checkpoint metadata object")
    return data


def _validate_checkpoint_action_metadata(checkpoint: str | Path) -> None:
    """Check action semantics before SB3 can load same-shaped, incompatible weights."""
    data = _read_checkpoint_metadata(checkpoint)
    if data.get("field_action_config") != action_config():
        raise ValueError(
            "Checkpoint action contract is missing or incompatible with "
            f"{ACTION_SCHEMA}; this action contract requires a newly trained checkpoint because its "
            "left/right track-speed meanings changed, "
            "and old weights are not automatically converted."
        )


def _validate_checkpoint_encoder_metadata(
    checkpoint: str | Path,
    *,
    input_schema: str = INPUT_SCHEMA,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> None:
    """Check the explicit schema before SB3 attempts to instantiate old weights."""
    expected_encoder_schema = _encoder_schema_for_input(input_schema)
    data = _read_checkpoint_metadata(checkpoint)
    policy_kwargs = data.get("policy_kwargs")
    extractor_kwargs = (
        policy_kwargs.get("features_extractor_kwargs")
        if isinstance(policy_kwargs, dict)
        else None
    )
    if (
        not isinstance(extractor_kwargs, dict)
        or extractor_kwargs.get("encoder_schema") != expected_encoder_schema
        or extractor_kwargs.get("input_schema", INPUT_SCHEMA) != input_schema
    ):
        raise ValueError(
            "Checkpoint sensor encoder schema is missing or incompatible; "
            f"the current {expected_encoder_schema} structure requires a newly trained checkpoint, "
            "and old weights are not automatically converted."
        )
    if data.get("field_encoder_config") != _field_encoder_config(input_schema):
        raise ValueError(
            "Checkpoint field_encoder_config is missing or incompatible; "
            f"the current {expected_encoder_schema} structure requires a newly trained checkpoint, "
            "and old weights are not automatically converted."
        )
    checkpoint_observation_config = data.get("field_observation_config")
    if checkpoint_observation_config is None:
        if input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            raise ValueError(
                "Checkpoint field_observation_config is missing or incompatible; "
                "a matching state-field schema is required before SB3 can load the checkpoint."
            )
    elif not isinstance(checkpoint_observation_config, Mapping):
        raise ValueError("Checkpoint field_observation_config is missing or incompatible")
    else:
        if (
            input_schema == INPUT_SCHEMA
            and not _observation_config_is_complete(checkpoint_observation_config)
        ):
            _validate_legacy_partial_observation_config(checkpoint_observation_config)
        else:
            _validate_observation_config(
                checkpoint_observation_config,
                input_schema,
                expected_observation_config=expected_observation_config,
            )


def make_exploration_variant(
    source_model,
    env,
    *,
    use_sde: bool,
    latent_norm_sq_mean: float | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
    sde_sample_freq: int | None = None,
):
    """Create a fresh Gaussian or gSDE model initialized from compatible weights.

    The source must be this module's canonical Gaussian RecurrentPPO policy. All
    compatible actor, critic, encoder, and recurrent weights are copied exactly.
    gSDE has a different ``log_std`` shape, so that one tensor is initialized
    from the source standard deviation and the supplied actor-latent calibration.
    For gSDE, ``sde_sample_freq`` defaults to 8 environment transitions: this
    resamples the noise matrix, not a held action.
    Optimizer state, rollout state, and environment RNG state are intentionally
    not copied; this is a weight warm start, not a bitwise training resume.
    """
    from gymnasium.spaces import Box
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.distributions import DiagGaussianDistribution

    if not isinstance(source_model, RecurrentPPO):
        raise ValueError("source_model must be an sb3-contrib RecurrentPPO")
    source_encoder_config = getattr(source_model, "field_encoder_config", None)
    source_input_schema = (
        source_encoder_config.get("input_schema", INPUT_SCHEMA)
        if isinstance(source_encoder_config, Mapping)
        else INPUT_SCHEMA
    )
    source_observation_config = getattr(source_model, "field_observation_config", None)
    if source_observation_config is None:
        source_observation_config = getattr(env, "observation_config", None)
    resolved_observation_config = _resolve_observation_config(
        env, source_input_schema, source_observation_config
    )
    _validate_model_encoder_contract(
        source_model,
        input_schema=source_input_schema,
        expected_observation_config=resolved_observation_config,
    )
    _validate_model_action_contract(source_model)
    source_policy = source_model.policy
    policy_kwargs = source_model.policy_kwargs
    if source_model.use_sde or getattr(source_policy, "use_sde", False):
        raise ValueError("source_model must use the supported Gaussian distribution, not gSDE")
    if not isinstance(source_policy.action_dist, DiagGaussianDistribution):
        raise ValueError("source_model must use DiagGaussianDistribution")
    if not isinstance(source_model.action_space, Box) or tuple(source_model.action_space.shape) != ACTION_SHAPE:
        raise ValueError(f"source_model must have a continuous {ACTION_SHAPE} action space")
    if not isinstance(env.action_space, Box) or tuple(env.action_space.shape) != ACTION_SHAPE:
        raise ValueError(f"env must have a continuous {ACTION_SHAPE} action space")
    validate_policy_spaces(
        source_model.observation_space,
        source_model.action_space,
        input_schema=source_input_schema,
        expected_observation_config=resolved_observation_config,
    )
    validate_policy_spaces(
        env.observation_space,
        env.action_space,
        input_schema=source_input_schema,
        expected_observation_config=resolved_observation_config,
    )
    if not (
        np.array_equal(source_model.action_space.low, env.action_space.low)
        and np.array_equal(source_model.action_space.high, env.action_space.high)
    ):
        raise ValueError("source and target action bounds must match")

    # make_model intentionally supports the project's one recurrent sensor-policy
    # layout. Reject extensions it cannot faithfully recreate instead of silently
    # training a differently shaped policy.
    supported_kwargs = {
        "features_extractor_class",
        "features_extractor_kwargs",
        "lstm_hidden_size",
        "n_lstm_layers",
        "net_arch",
        "log_std_init",
        "squash_output",
        "normalize_images",
    }
    if set(policy_kwargs) - supported_kwargs:
        raise ValueError("source policy has unsupported custom policy_kwargs")
    if policy_kwargs.get("features_extractor_class") is not SensorEncoder:
        raise ValueError("source policy must use this module's SensorEncoder")
    extractor_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
    if set(extractor_kwargs) - {"goal_gain", "encoder_schema", "input_schema"}:
        raise ValueError("source SensorEncoder has unsupported custom settings")
    if policy_kwargs.get("squash_output", False) or source_policy.squash_output:
        raise ValueError("source policy must use the supported unsquashed action distribution")
    if policy_kwargs.get("normalize_images", False):
        raise ValueError("source policy must use normalize_images=False")
    if (
        source_policy.shared_lstm
        or not source_policy.enable_critic_lstm
        or not source_policy.share_features_extractor
        or not isinstance(source_policy.features_extractor, SensorEncoder)
        or source_policy.optimizer_class is not torch.optim.Adam
    ):
        raise ValueError("source policy has an unsupported recurrent, extractor, or optimizer structure")
    if not isinstance(source_policy.lstm_actor, nn.LSTM) or not isinstance(source_policy.lstm_critic, nn.LSTM):
        raise ValueError("source policy must have separate actor and critic LSTMs")
    if source_policy.lstm_actor.hidden_size != source_policy.lstm_critic.hidden_size:
        raise ValueError("source actor and critic LSTM sizes do not match")
    if source_policy.lstm_actor.num_layers != source_policy.lstm_critic.num_layers:
        raise ValueError("source actor and critic LSTM layer counts do not match")
    if source_policy.optimizer_kwargs != {"eps": 1e-5}:
        raise ValueError("source policy has unsupported Adam settings")
    if (
        float(source_model.ent_coef) != 0.0
        or float(source_model.vf_coef) != 0.5
        or float(source_model.gae_lambda) != 0.95
        or float(source_model.max_grad_norm) != 0.5
        or source_model.clip_range_vf is not None
        or any(not math.isclose(float(source_model.clip_range(value)), 0.2) for value in (0.0, 0.5, 1.0))
    ):
        raise ValueError("source model uses unsupported PPO training settings")
    if not isinstance(source_model.learning_rate, (int, float)):
        raise ValueError("source model must use a constant learning rate")
    if int(source_model.n_envs) != int(getattr(env, "num_envs", 1)):
        raise ValueError("source and target environment counts must match")

    source_log_std = source_policy.log_std.detach()
    if tuple(source_log_std.shape) != ACTION_SHAPE:
        raise ValueError(f"source Gaussian log_std must have shape {ACTION_SHAPE}")
    if use_sde:
        if latent_norm_sq_mean is None:
            raise ValueError("gSDE initialization requires latent_norm_sq_mean from policy rollouts")
        latent_norm_sq_mean = float(latent_norm_sq_mean)
        if not math.isfinite(latent_norm_sq_mean) or latent_norm_sq_mean <= 0:
            raise ValueError("latent_norm_sq_mean must be finite and positive")
        resolved_sde_sample_freq = 8 if sde_sample_freq is None else int(sde_sample_freq)
    else:
        if latent_norm_sq_mean is not None:
            raise ValueError("latent_norm_sq_mean is only used for gSDE")
        resolved_sde_sample_freq = source_model.sde_sample_freq if sde_sample_freq is None else int(sde_sample_freq)
    if resolved_sde_sample_freq == 0 or resolved_sde_sample_freq < -1:
        raise ValueError("sde_sample_freq must be -1 or a positive integer")

    net_arch = policy_kwargs.get("net_arch")
    if not isinstance(net_arch, dict) or set(net_arch) != {"pi", "vf"}:
        raise ValueError("source policy must use separate pi/vf net_arch lists")
    if any(not isinstance(net_arch[key], list) for key in ("pi", "vf")):
        raise ValueError("source policy net_arch must contain pi/vf lists")
    if (
        int(policy_kwargs.get("lstm_hidden_size", -1)) != source_policy.lstm_actor.hidden_size
        or int(policy_kwargs.get("n_lstm_layers", -1)) != source_policy.lstm_actor.num_layers
    ):
        raise ValueError("source policy LSTM metadata does not match its modules")
    target = make_model(
        env,
        input_schema=source_input_schema,
        observation_config=resolved_observation_config,
        seed=source_model.seed if seed is None else seed,
        n_steps=source_model.n_steps,
        verbose=source_model.verbose,
        goal_gain=float(extractor_kwargs.get("goal_gain", 1.0)),
        encoder_schema=str(extractor_kwargs["encoder_schema"]),
        batch_size=source_model.batch_size,
        n_epochs=source_model.n_epochs,
        learning_rate=float(source_model.learning_rate),
        gamma=float(source_model.gamma),
        use_sde=bool(use_sde),
        sde_sample_freq=resolved_sde_sample_freq,
        log_std_init=float(source_log_std.mean().cpu()),
        squash_output=False,
        target_kl=source_model.target_kl,
        lstm_hidden_size=source_policy.lstm_actor.hidden_size,
        n_lstm_layers=source_policy.lstm_actor.num_layers,
        net_arch=net_arch,
        device=source_model.device if device is None else device,
        tensorboard_log=source_model.tensorboard_log,
    )

    source_state = source_policy.state_dict()
    target_state = target.policy.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("source and target policy parameter names do not match")
    with torch.no_grad():
        for name, target_value in target_state.items():
            source_value = source_state[name]
            if use_sde and name == "log_std":
                continue
            if source_value.shape != target_value.shape:
                raise ValueError(f"unsupported parameter shape change for {name}")
            target_value.copy_(source_value.detach().to(device=target_value.device, dtype=target_value.dtype))
    target.policy.load_state_dict(target_state, strict=True)

    if use_sde:
        if (
            not getattr(target.policy.action_dist, "full_std", False)
            or getattr(target.policy.action_dist, "use_expln", True)
        ):
            raise ValueError("gSDE target must use full_std=True and use_expln=False")
        if tuple(target.policy.log_std.shape) != (target.policy.mlp_extractor.latent_dim_pi, ACTION_SHAPE[0]):
            raise ValueError("gSDE target log_std has an unsupported shape")
        if tuple(target.policy.log_std.shape) == tuple(source_log_std.shape):
            raise ValueError("gSDE target log_std unexpectedly matches the Gaussian shape")
        calibrated = source_log_std.to(target.policy.log_std.device) - 0.5 * math.log(latent_norm_sq_mean)
        with torch.no_grad():
            target.policy.log_std.copy_(calibrated.reshape(1, -1).expand_as(target.policy.log_std))
    if target.policy.squash_output:
        raise ValueError("target policy must use squash_output=False")
    target.num_timesteps = int(source_model.num_timesteps)
    return target


@contextmanager
def capture_actor_latent_pi(policy) -> Iterator[list[torch.Tensor]]:
    """Capture the actual post-LSTM actor MLP output during normal policy calls."""
    actor_net = getattr(getattr(policy, "mlp_extractor", None), "policy_net", None)
    if not isinstance(actor_net, nn.Module):
        raise ValueError("policy has no supported mlp_extractor.policy_net actor module")
    captured: list[torch.Tensor] = []

    def save_output(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("actor policy_net did not return a tensor")
        captured.append(output.detach())

    handle = actor_net.register_forward_hook(save_output)
    try:
        yield captured
    finally:
        handle.remove()


def _legacy_policy_kwargs(checkpoint: str | Path) -> dict[str, Any]:
    """Read portable SB3 policy settings and replace the old extractor class."""
    with zipfile.ZipFile(checkpoint) as archive:
        try:
            data = json.loads(archive.read("data"))
        except KeyError as exc:
            raise ValueError(f"{checkpoint} is not an SB3 checkpoint (missing data entry)") from exc
    old = data.get("policy_kwargs")
    if not isinstance(old, dict):
        raise ValueError(f"{checkpoint} does not contain policy_kwargs")

    defaults: dict[str, Any] = {
        "features_extractor_kwargs": {"goal_gain": 1.0},
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": {"pi": [64], "vf": [64]},
        "log_std_init": 0.0,
        "squash_output": False,
        "normalize_images": False,
    }
    # SB3's data.json keeps these JSON-compatible values alongside its opaque
    # cloudpickle payload, so the old module is never imported during migration.
    for key in (
        "features_extractor_kwargs",
        "lstm_hidden_size",
        "n_lstm_layers",
        "net_arch",
        "log_std_init",
        "squash_output",
        "normalize_images",
        "share_features_extractor",
        "optimizer_kwargs",
        "shared_lstm",
        "enable_critic_lstm",
    ):
        if key in old:
            defaults[key] = old[key]
    kwargs: dict[str, Any] = {"features_extractor_class": SensorEncoder, **defaults}
    return kwargs


def _reject_absolute_goal_legacy_conversion(checkpoint: str | Path) -> None:
    """Keep the old extractor migration from relabeling absolute-goal inputs."""
    data = _read_checkpoint_metadata(checkpoint)
    policy_kwargs = data.get("policy_kwargs")
    extractor_kwargs = (
        policy_kwargs.get("features_extractor_kwargs")
        if isinstance(policy_kwargs, dict)
        else None
    )
    encoder_config = data.get("field_encoder_config")
    observation_config = data.get("field_observation_config")
    if observation_config is None:
        observation_config = data.get("observation_config")
    schema_markers = [data.get("input_schema"), data.get("field_input_schema")]
    if isinstance(extractor_kwargs, dict):
        schema_markers.extend((extractor_kwargs.get("input_schema"), extractor_kwargs.get("encoder_schema")))
    if isinstance(encoder_config, Mapping):
        schema_markers.extend((encoder_config.get("input_schema"), encoder_config.get("schema")))
    if isinstance(observation_config, Mapping):
        schema_markers.append(observation_config.get("schema"))
        feedback_config = observation_config.get("feedback")
        if isinstance(feedback_config, Mapping):
            schema_markers.append(feedback_config.get("input_schema"))
    if ABSOLUTE_GOAL_INPUT_SCHEMA in schema_markers or ABSOLUTE_GOAL_ENCODER_SCHEMA in schema_markers:
        raise ValueError(
            "absolute-goal checkpoints cannot be converted to the legacy v6 policy contract"
        )


def convert_legacy_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    env=None,
    *,
    device: str | torch.device = "cpu",
):
    """Map the old ``field_sim`` extractor to this module, save, and reload.

    The source checkpoint remains untouched. ``env`` is optional because SB3
    stores its spaces in the archive; pass the native wrapper when available to
    validate the migrated observation/action contract. ``output`` is the
    destination checkpoint filename; SB3 appends ``.zip`` when it is omitted.
    """
    from sb3_contrib import RecurrentPPO

    source = Path(checkpoint)
    destination = Path(output)
    if source.resolve() == destination.resolve() or source.resolve() == Path(f"{destination}.zip").resolve():
        raise ValueError("converted checkpoint output must be different from the legacy source")
    _reject_absolute_goal_legacy_conversion(source)
    if env is not None:
        _resolve_observation_config(env, INPUT_SCHEMA, None)
    _validate_checkpoint_action_metadata(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = RecurrentPPO.load(
        str(source),
        env=env,
        device=device,
        custom_objects={"policy_kwargs": _legacy_policy_kwargs(source)},
    )
    model.save(str(destination))
    saved_path = destination if destination.suffix == ".zip" else Path(f"{destination}.zip")
    return RecurrentPPO.load(str(saved_path), env=env, device=device)


class PolicyRunner:
    """Stateful deterministic policy inference for one or more vector rows."""

    def __init__(
        self,
        env,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "cpu",
        input_schema: str = INPUT_SCHEMA,
        expected_observation_config: Mapping[str, Any] | None = None,
    ):
        from sb3_contrib import RecurrentPPO

        torch.set_num_threads(1)
        expected_encoder_schema = _encoder_schema_for_input(input_schema)
        resolved_observation_config = _resolve_observation_config(
            env, input_schema, expected_observation_config
        )
        validate_policy_spaces(
            env.observation_space,
            env.action_space,
            input_schema=input_schema,
            expected_observation_config=resolved_observation_config,
        )
        if checkpoint:
            _validate_checkpoint_action_metadata(checkpoint)
            _validate_checkpoint_encoder_metadata(
                checkpoint,
                input_schema=input_schema,
                expected_observation_config=resolved_observation_config,
            )
        try:
            self.model = (
                RecurrentPPO.load(str(checkpoint), env=env, device=device)
                if checkpoint
                else make_model(
                    env,
                    input_schema=input_schema,
                    observation_config=resolved_observation_config,
                    device=device,
                )
            )
        except ValueError as exc:
            if checkpoint and "spaces do not match" in str(exc):
                raise ValueError(
                    "Checkpoint observation dimensions do not match this environment "
                    f"(LiDAR: {env.observation_space['lidar'].shape}, "
                    f"depth: {env.observation_space['depth'].shape}, "
                    f"detections: {env.observation_space['detections'].shape}, "
                    f"state: {env.observation_space['state'].shape}). "
                    f"The {input_schema} 33-state / 37-feedback contract requires a newly trained checkpoint; "
                    "old weights are not automatically converted."
                ) from exc
            raise
        except RuntimeError as exc:
            message = str(exc).lower()
            if checkpoint and ("size mismatch" in message or "size of tensor" in message):
                raise ValueError(
                    "Checkpoint encoder weights are incompatible with the current "
                    f"{expected_encoder_schema} policy; this structure requires a newly trained checkpoint, "
                    "and old weights are not automatically converted."
                ) from exc
            raise
        _validate_model_encoder_contract(
            self.model,
            input_schema=input_schema,
            expected_observation_config=resolved_observation_config,
        )
        _validate_model_action_contract(self.model)
        self.input_schema = input_schema
        self.observation_config = resolved_observation_config
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.state = None
        self.episode_start = np.ones(self.num_envs, dtype=bool)
        self.label = "checkpoint loaded; task competence is not implied" if checkpoint else "random initialization; not trained"

    def reset(self, done: np.ndarray | None = None) -> None:
        if done is None:
            self.state = None
            self.episode_start = np.ones(self.num_envs, dtype=bool)
        else:
            done = np.asarray(done, dtype=bool).reshape(-1)
            if done.shape != (self.num_envs,):
                raise ValueError(f"done must have shape ({self.num_envs},)")
            if self.state is not None:
                states = [np.asarray(value).copy() for value in self.state]
                for value in states:
                    value[:, done, :] = 0.0
                self.state = tuple(states)
            self.episode_start = done

    def action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        action, self.state = self.model.predict(
            observation,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start = np.zeros(self.num_envs, dtype=bool)
        return action
