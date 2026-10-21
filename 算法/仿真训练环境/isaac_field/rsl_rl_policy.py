"""RSL-RL adapter for the field sensor policy.

The sensor encoder is the existing :class:`isaac_field.policy.SensorEncoder`.
RSL-RL owns the recurrent models, distributions, PPO update and runner; this
module only adapts TensorDict inputs, assigns one owner to the shared encoder,
and tags checkpoints with the field policy contract.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel

from .action_contract import action_config
from .feedback import INPUT_SCHEMA
from .policy import (
    BRANCH_DIMS,
    FEATURES_DIM,
    OBSERVATION_SHAPES,
    SensorEncoder,
    _validate_observation_config,
    _encoder_schema_for_input,
    _field_encoder_config,
)


RSL_FIELD_CHECKPOINT_FORMAT = "agtc.rsl_rl.field_policy"
RSL_FIELD_CHECKPOINT_VERSION = 1
FIELD_OBSERVATION_KEYS = tuple(OBSERVATION_SHAPES)
FIELD_ACTION_DIM = 4
FIELD_LSTM_HIDDEN_DIM = 512
FIELD_LSTM_NUM_LAYERS = 1
FIELD_HEAD_DIMS = (64,)
FIELD_HEAD_ACTIVATION = "tanh"
FIELD_LOG_STD_INIT = -2.0
FIELD_GAUSSIAN_INIT_STD = math.exp(FIELD_LOG_STD_INIT)
FIELD_GAUSSIAN_CLASS = "rsl_rl.modules.distribution.GaussianDistribution"
FIELD_GAUSSIAN_STD_RANGE = (1e-6, 1e6)
_RUNTIME_CONTRACT_KEYS = ("num_envs", "num_steps_per_env")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        hasattr(value, "keys") and hasattr(value, "__getitem__")
    )


def _field_observation_tree(obs: Mapping[str, Any] | TensorDict) -> Mapping[str, Any] | TensorDict:
    """Return sensor leaves from either a root TensorDict or Isaac Lab's policy group."""
    if "policy" in obs:
        policy = obs["policy"]
        if _is_mapping(policy):
            return policy
    return obs


def _field_observation_shapes(obs: Mapping[str, Any] | TensorDict) -> dict[str, tuple[int, ...]]:
    leaves = _field_observation_tree(obs)
    missing = set(FIELD_OBSERVATION_KEYS) - set(leaves.keys())
    if missing:
        raise ValueError(f"field observations are missing keys: {sorted(missing)}")
    shapes: dict[str, tuple[int, ...]] = {}
    for key in FIELD_OBSERVATION_KEYS:
        value = leaves[key]
        expected = tuple(OBSERVATION_SHAPES[key])
        if not isinstance(value, torch.Tensor) or value.ndim < len(expected) + 1:
            raise ValueError(f"field observation {key!r} must include batch and feature axes")
        shape = tuple(value.shape[-len(expected) :])
        if shape != expected:
            raise ValueError(f"field observation {key!r} has shape {shape}, expected {expected}")
        shapes[key] = shape
    leading_shapes = {
        tuple(leaves[key].shape[: -len(OBSERVATION_SHAPES[key])])
        for key in FIELD_OBSERVATION_KEYS
    }
    if len(leading_shapes) != 1:
        raise ValueError("all field observation leaves must have matching batch/sequence axes")
    return shapes


def _sensor_space(shapes: Mapping[str, Sequence[int]]) -> gym.spaces.Dict:
    spaces = {
        key: gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(shapes[key]),
            dtype=np.float32,
        )
        for key in FIELD_OBSERVATION_KEYS
    }
    return gym.spaces.Dict(spaces)


def _json_safe(value: Any) -> Any:
    """Convert contract metadata to plain JSON values and reject opaque objects."""
    # np.float64 can also satisfy isinstance(value, float), so normalize NumPy
    # scalars before Python's native scalar checks to keep torch.save payloads
    # compatible with weights_only=True.
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint contract cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"checkpoint contract value is not JSON serializable: {type(value).__name__}")


def _resolve_env_contract_value(env: Any, name: str, fallback: Any = None) -> Any:
    source = getattr(env, "unwrapped", env)
    value = getattr(source, name, fallback)
    return value() if callable(value) else value


def _normalized_gaussian_config(value: Any) -> dict[str, Any]:
    """Validate the supported Gaussian semantics across dot/colon class paths."""
    if not isinstance(value, Mapping):
        raise ValueError("field actor must use a Gaussian distribution config")
    config = _json_safe(value)
    class_name = config.get("class_name")
    if not isinstance(class_name, str):
        raise ValueError("field Gaussian distribution class is missing")
    class_name = class_name.replace(":", ".")
    if class_name != FIELD_GAUSSIAN_CLASS:
        raise ValueError("field actor distribution must be RSL-RL GaussianDistribution")
    init_std = config.get("init_std")
    std_type = config.get("std_type")
    std_range = config.get("std_range", list(FIELD_GAUSSIAN_STD_RANGE))
    if (
        isinstance(init_std, bool)
        or not isinstance(init_std, (int, float))
        or float(init_std) != FIELD_GAUSSIAN_INIT_STD
        or std_type != "log"
        or std_range != list(FIELD_GAUSSIAN_STD_RANGE)
        or set(config) - {"class_name", "init_std", "std_type", "std_range"}
    ):
        raise ValueError("field Gaussian distribution must keep log_std=-2 and the default std_range")
    return {
        "class_name": FIELD_GAUSSIAN_CLASS,
        "init_std": FIELD_GAUSSIAN_INIT_STD,
        "std_type": "log",
        "std_range": list(FIELD_GAUSSIAN_STD_RANGE),
    }


class RslFieldRNNModel(RNNModel):
    """Existing five-sensor encoder followed by RSL-RL's LSTM and MLP head.

    Configure this class for both actor and critic. When RSL-RL's
    ``share_cnn_encoders`` option is enabled, it passes ``actor.cnns`` into the
    critic. The actor registers the encoder; the critic stores only a Python
    reference so its parameters are optimized and clipped exactly once.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = FIELD_HEAD_DIMS,
        activation: str = FIELD_HEAD_ACTIVATION,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = FIELD_LSTM_HIDDEN_DIM,
        rnn_num_layers: int = FIELD_LSTM_NUM_LAYERS,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        input_schema: str = INPUT_SCHEMA,
        encoder_schema: str | None = None,
        goal_gain: float = 1.0,
        cache_inference: bool = True,
        observation_shapes: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        if obs_set not in ("actor", "critic"):
            raise ValueError("RslFieldRNNModel obs_set must be 'actor' or 'critic'")
        if int(output_dim) != (FIELD_ACTION_DIM if obs_set == "actor" else 1):
            raise ValueError("actor output_dim must be 4 and critic output_dim must be 1")
        if tuple(hidden_dims) != FIELD_HEAD_DIMS or activation != FIELD_HEAD_ACTIVATION:
            raise ValueError("field PPO heads must be one 64-unit tanh layer")
        if obs_normalization:
            raise ValueError("field policy requires obs_normalization=False")
        if rnn_type.lower() != "lstm" or int(rnn_hidden_dim) != FIELD_LSTM_HIDDEN_DIM or int(rnn_num_layers) != FIELD_LSTM_NUM_LAYERS:
            raise ValueError("field recurrent policy requires an LSTM with hidden_dim=512 and num_layers=1")
        if obs_set == "actor" and distribution_cfg is None:
            raise ValueError("field actor requires a Gaussian distribution")
        if obs_set == "critic" and distribution_cfg is not None:
            raise ValueError("field critic must be deterministic")
        if obs_set == "actor":
            _normalized_gaussian_config(distribution_cfg)

        actual_shapes = _field_observation_shapes(obs)
        if observation_shapes is not None:
            declared_shapes = {key: tuple(value) for key, value in observation_shapes.items()}
            if declared_shapes != actual_shapes:
                raise ValueError("declared field observation_shapes differ from the live TensorDict")

        expected_encoder_schema = _encoder_schema_for_input(input_schema)
        if encoder_schema is None:
            encoder_schema = expected_encoder_schema
        if encoder_schema != expected_encoder_schema:
            raise ValueError(
                f"encoder_schema {encoder_schema!r} does not match input_schema {input_schema!r}; "
                f"expected {expected_encoder_schema!r}"
            )
        if float(goal_gain) != 1.0 or not math.isfinite(float(goal_gain)):
            raise ValueError("goal_gain must be 1.0")

        # Keep constructor metadata independent of RSL's config resolution.
        distribution_config = _json_safe(distribution_cfg)

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
        )

        self.input_schema = input_schema
        self.encoder_schema = encoder_schema
        self.goal_gain = float(goal_gain)
        self.cache_inference = bool(cache_inference)
        self.obs_set = obs_set
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation_name = activation
        self.obs_normalization = False
        self.rnn_type = "lstm"
        self.rnn_hidden_dim = FIELD_LSTM_HIDDEN_DIM
        self.rnn_num_layers = FIELD_LSTM_NUM_LAYERS
        self.distribution_config = distribution_config

        if cnns is None:
            encoder = SensorEncoder(
                _sensor_space(actual_shapes),
                cache_inference=self.cache_inference,
                goal_gain=self.goal_gain,
                encoder_schema=self.encoder_schema,
                input_schema=self.input_schema,
            )
            # This is the only registered owner of the shared encoder.
            self.cnns = nn.ModuleDict({"field": encoder})
        else:
            if "field" not in cnns or not isinstance(cnns["field"], SensorEncoder):
                raise ValueError("shared cnns must contain the existing SensorEncoder under key 'field'")
            encoder = cnns["field"]
            if (
                encoder.encoder_schema != self.encoder_schema
                or encoder.input_schema != self.input_schema
                or int(encoder.features_dim) != FEATURES_DIM
                or encoder.branch_dims != dict(BRANCH_DIMS)
            ):
                raise ValueError("shared SensorEncoder differs from the requested field encoder contract")
            # Bypass nn.Module registration: actor owns this ModuleDict and the
            # critic's optimizer/state_dict must not list the same parameters.
            object.__setattr__(self, "_shared_cnns_ref", cnns)

    @property
    def sensor_encoder(self) -> SensorEncoder:
        if "cnns" in self._modules:
            return self._modules["cnns"]["field"]  # type: ignore[return-value]
        return self.__dict__["_shared_cnns_ref"]["field"]  # type: ignore[return-value]

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active = obs_groups[obs_set]
        if not active:
            raise ValueError(f"the {obs_set!r} observation set cannot be empty")
        leaves = _field_observation_tree(obs)
        _field_observation_shapes(obs)
        if "policy" in obs and _is_mapping(obs["policy"]):
            if active != ["policy"]:
                raise ValueError("nested field observations must use obs_groups=[\"policy\"]")
        elif set(active) != set(FIELD_OBSERVATION_KEYS):
            raise ValueError(
                "root field observations must use obs_groups containing lidar, depth, detections, state, and feedback"
            )
        return active, FEATURES_DIM

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        leaves = _field_observation_tree(obs)
        leading_shape = tuple(leaves["lidar"].shape[:-2])
        batch = math.prod(leading_shape)
        encoder_inputs = {
            key: leaves[key].reshape(batch, *OBSERVATION_SHAPES[key])
            for key in FIELD_OBSERVATION_KEYS
        }
        features = self.sensor_encoder(encoder_inputs).reshape(*leading_shape, FEATURES_DIM)
        features = self.obs_normalizer(features)
        return self.rnn(features, masks, hidden_state).squeeze(0)

    def constructor_config(self) -> dict[str, Any]:
        """Return JSON-ready keyword arguments sufficient to rebuild this model."""
        config = {
            "output_dim": self.output_dim,
            "obs_set": self.obs_set,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation_name,
            "obs_normalization": False,
            "distribution_cfg": copy.deepcopy(self.distribution_config),
            "rnn_type": self.rnn_type,
            "rnn_hidden_dim": self.rnn_hidden_dim,
            "rnn_num_layers": self.rnn_num_layers,
            "input_schema": self.input_schema,
            "encoder_schema": self.encoder_schema,
            "goal_gain": self.goal_gain,
            "cache_inference": self.cache_inference,
            "observation_shapes": {
                key: list(shape) for key, shape in OBSERVATION_SHAPES.items()
            },
        }
        return _json_safe(config)

    def as_jit(self) -> nn.Module:
        from .rsl_rl_export import field_model_as_jit

        return field_model_as_jit(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        from .rsl_rl_export import field_model_as_onnx

        return field_model_as_onnx(self, verbose=verbose)


def build_field_policy_contract(
    actor: RslFieldRNNModel,
    critic: RslFieldRNNModel,
    *,
    obs_groups: Mapping[str, Sequence[str]],
    env: Any,
    num_steps_per_env: int | None = None,
) -> dict[str, Any]:
    """Build the JSON-safe identity attached to every RSL field checkpoint."""
    observation_config = _resolve_env_contract_value(env, "observation_config")
    configured_action = _resolve_env_contract_value(env, "action_config", action_config())
    if configured_action != action_config():
        raise ValueError("live environment action_config differs from the field action contract")
    encoder = actor.sensor_encoder
    if observation_config is None:
        raise ValueError("RSL field environments must expose a complete observation_config")
    config: dict[str, Any] = {
        "backend": "rsl_rl",
        "format": RSL_FIELD_CHECKPOINT_FORMAT,
        "version": RSL_FIELD_CHECKPOINT_VERSION,
        "input_schema": actor.input_schema,
        "encoder_config": {
            "class_name": "isaac_field.policy:SensorEncoder",
            **_field_encoder_config(actor.input_schema),
            "goal_gain": actor.goal_gain,
            "cache_inference": actor.cache_inference,
        },
        "observation_shapes": {
            key: list(shape) for key, shape in OBSERVATION_SHAPES.items()
        },
        "observation_config": observation_config,
        "action_config": configured_action,
        "actor_config": actor.constructor_config(),
        "critic_config": critic.constructor_config(),
        "obs_groups": {key: list(value) for key, value in obs_groups.items()},
        "lstm": {
            "type": actor.rnn_type,
            "hidden_dim": actor.rnn_hidden_dim,
            "num_layers": actor.rnn_num_layers,
            "actor_critic_separate": True,
        },
        "heads": {
            "actor": {"hidden_dims": list(actor.hidden_dims), "activation": actor.activation_name},
            "critic": {"hidden_dims": list(critic.hidden_dims), "activation": critic.activation_name},
        },
        "distribution": actor.distribution_config,
        "action_dim": FIELD_ACTION_DIM,
        "num_envs": int(getattr(env, "num_envs", 0)),
        "num_steps_per_env": None if num_steps_per_env is None else int(num_steps_per_env),
    }
    for name in ("drive_config", "damage_config"):
        value = _resolve_env_contract_value(env, name)
        if value is not None:
            config[name] = value
    return _json_safe(config)


def validate_field_policy_contract(
    contract: Mapping[str, Any], expected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a checkpoint marker and the fixed field model/action contract."""
    safe = _json_safe(contract)
    if not isinstance(safe, dict):
        raise ValueError("field_policy_contract must be a JSON object")
    if (
        safe.get("backend") != "rsl_rl"
        or safe.get("format") != RSL_FIELD_CHECKPOINT_FORMAT
        or safe.get("version") != RSL_FIELD_CHECKPOINT_VERSION
    ):
        raise ValueError("checkpoint is not a supported AGTC RSL-RL field policy")
    schema = safe.get("input_schema")
    expected_encoder = _field_encoder_config(schema)
    encoder_cfg = safe.get("encoder_config")
    if not isinstance(encoder_cfg, dict):
        raise ValueError("checkpoint field encoder config is missing")
    if (
        encoder_cfg.get("class_name") != "isaac_field.policy:SensorEncoder"
        or any(encoder_cfg.get(key) != value for key, value in expected_encoder.items())
        or encoder_cfg.get("schema") != expected_encoder["schema"]
        or encoder_cfg.get("features_dim") != FEATURES_DIM
        or encoder_cfg.get("goal_gain") != 1.0
        or not isinstance(encoder_cfg.get("cache_inference"), bool)
    ):
        raise ValueError("checkpoint field encoder config/schema is incompatible")
    if safe.get("observation_shapes") != {
        key: list(shape) for key, shape in OBSERVATION_SHAPES.items()
    }:
        raise ValueError("checkpoint field observation shapes are incompatible")
    observation_config = safe.get("observation_config")
    try:
        checked_observation_config = _validate_observation_config(observation_config, schema)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint field observation config is incompatible") from exc
    if checked_observation_config.get("observation_shapes") != {
        key: list(shape) for key, shape in OBSERVATION_SHAPES.items()
    }:
        raise ValueError("checkpoint observation_config sensor shapes are incompatible")
    if safe.get("action_config") != action_config() or safe.get("action_dim") != FIELD_ACTION_DIM:
        raise ValueError("checkpoint field action contract is incompatible")
    expected_groups = {
        "actor": list(FIELD_OBSERVATION_KEYS),
        "critic": list(FIELD_OBSERVATION_KEYS),
    }
    if safe.get("obs_groups") != expected_groups:
        raise ValueError("checkpoint field observation groups are incompatible")
    actor_cfg = safe.get("actor_config")
    critic_cfg = safe.get("critic_config")
    if not isinstance(actor_cfg, dict) or not isinstance(critic_cfg, dict):
        raise ValueError("checkpoint is missing actor/critic constructor configs")
    if critic_cfg.get("distribution_cfg") is not None:
        raise ValueError("checkpoint field critic must be deterministic")
    try:
        actor_distribution = _normalized_gaussian_config(actor_cfg.get("distribution_cfg"))
        contract_distribution = _normalized_gaussian_config(safe.get("distribution"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint field Gaussian distribution config is incompatible") from exc
    if actor_distribution != contract_distribution:
        raise ValueError("checkpoint field Gaussian distribution config is incompatible")
    for name, cfg, output_dim, obs_set in (
        ("actor", actor_cfg, FIELD_ACTION_DIM, "actor"),
        ("critic", critic_cfg, 1, "critic"),
    ):
        if (
            cfg.get("output_dim") != output_dim
            or cfg.get("obs_set") != obs_set
            or cfg.get("hidden_dims") != list(FIELD_HEAD_DIMS)
            or cfg.get("activation") != FIELD_HEAD_ACTIVATION
            or cfg.get("obs_normalization") is not False
            or cfg.get("rnn_type") != "lstm"
            or cfg.get("rnn_hidden_dim") != FIELD_LSTM_HIDDEN_DIM
            or cfg.get("rnn_num_layers") != FIELD_LSTM_NUM_LAYERS
            or cfg.get("input_schema") != schema
            or cfg.get("encoder_schema") != expected_encoder["schema"]
            or cfg.get("goal_gain") != 1.0
            or not isinstance(cfg.get("cache_inference"), bool)
            or cfg.get("observation_shapes") != {
                key: list(shape) for key, shape in OBSERVATION_SHAPES.items()
            }
        ):
            raise ValueError(f"checkpoint {name} model config is incompatible")
    expected_lstm = {
        "type": "lstm",
        "hidden_dim": FIELD_LSTM_HIDDEN_DIM,
        "num_layers": FIELD_LSTM_NUM_LAYERS,
        "actor_critic_separate": True,
    }
    expected_heads = {
        "actor": {"hidden_dims": list(FIELD_HEAD_DIMS), "activation": FIELD_HEAD_ACTIVATION},
        "critic": {"hidden_dims": list(FIELD_HEAD_DIMS), "activation": FIELD_HEAD_ACTIVATION},
    }
    if safe.get("lstm") != expected_lstm or safe.get("heads") != expected_heads:
        raise ValueError("checkpoint field recurrent/head metadata is incompatible")
    if (
        not isinstance(safe.get("num_envs"), int)
        or safe["num_envs"] < 1
        or (
            safe.get("num_steps_per_env") is not None
            and (not isinstance(safe["num_steps_per_env"], int) or safe["num_steps_per_env"] < 1)
        )
    ):
        raise ValueError("checkpoint field rollout metadata is invalid")
    if expected is not None:
        expected_safe = _json_safe(expected)
        for key in _RUNTIME_CONTRACT_KEYS:
            safe.pop(key, None)
            expected_safe.pop(key, None)
        if safe != expected_safe:
            raise ValueError("checkpoint field_policy_contract differs from the live field environment/model")
    return safe


class RslFieldPPO(PPO):
    """Official RSL-RL PPO plus strict field checkpoint identity metadata."""

    field_policy_contract: dict[str, Any] | None = None

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: Any, cfg: dict, device: str) -> "RslFieldPPO":
        # Keep the upstream model, storage, PPO, sharing, and compile setup.
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(algorithm, RslFieldPPO):
            raise TypeError("algorithm.class_name must resolve to RslFieldPPO")
        actor = algorithm._raw_actor
        critic = algorithm._raw_critic
        if not isinstance(actor, RslFieldRNNModel) or not isinstance(critic, RslFieldRNNModel):
            raise TypeError("RslFieldPPO requires RslFieldRNNModel actor and critic")
        if actor.sensor_encoder is not critic.sensor_encoder:
            raise ValueError("field actor and critic must share the same SensorEncoder instance")
        algorithm.field_policy_contract = build_field_policy_contract(
            actor,
            critic,
            obs_groups=cfg["obs_groups"],
            env=env,
            num_steps_per_env=cfg.get("num_steps_per_env"),
        )
        validate_field_policy_contract(algorithm.field_policy_contract)
        return algorithm

    def save(self) -> dict[str, Any]:
        if self.field_policy_contract is None:
            raise ValueError("cannot save RslFieldPPO before the environment policy contract is bound")
        saved = super().save()
        saved["field_policy_contract"] = copy.deepcopy(self.field_policy_contract)
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        contract = loaded_dict.get("field_policy_contract") if isinstance(loaded_dict, dict) else None
        if not isinstance(contract, Mapping):
            raise ValueError("checkpoint lacks field_policy_contract; SB3 .zip checkpoints cannot resume in RSL-RL")
        validate_field_policy_contract(contract, expected=self.field_policy_contract)
        return super().load(loaded_dict, load_cfg, strict)


def load_rsl_checkpoint_payload(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and validate an RSL field checkpoint payload without initializing Isaac Lab."""
    checkpoint = Path(path)
    if checkpoint.suffix.lower() == ".zip":
        raise ValueError("SB3 .zip checkpoints are replayed by the SB3 backend and cannot be loaded as RSL-RL")
    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError(f"{checkpoint} is not a readable RSL-RL field checkpoint") from exc
    if not isinstance(payload, dict):
        raise ValueError("RSL field checkpoint payload must be a dictionary")
    contract = payload.get("field_policy_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint lacks field_policy_contract; SB3 .zip checkpoints cannot resume in RSL-RL")
    validate_field_policy_contract(contract)
    return payload
