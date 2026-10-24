"""Load RSL checkpoints for the existing, backend-independent episode evaluator.

Only the prediction boundary is adapted. Training and recurrent computation
remain in RSL-RL; the evaluator continues to own resets and trajectory records.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from .action_contract import action_config
from .feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from .policy_contract import (
    BACKEND_IDENTITY,
    BACKEND_IDENTITY_KEYS,
    _field_encoder_config,
    _json_safe,
    _validate_observation_config,
    recorded_backend_identity,
    validate_backend_identity,
)


def _checkpoint_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RslPredictionModel:
    """Explicit RSL actor predictor with caller-owned LSTM state.

    The ``predict`` signature lets ``learning.evaluate_model`` reuse its real
    episode loop. This object is not an SB3 model and never converts SB3 ZIPs.
    """

    backend = "rsl_rl"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        input_schema: str = ABSOLUTE_GOAL_INPUT_SCHEMA,
        expected_observation_config: Mapping[str, Any] | None = None,
        expected_backend_identity: Mapping[str, Any] | None = None,
        allow_backend_transfer: bool = False,
    ):
        from .rsl_rl_policy import RslFieldRNNModel, load_rsl_checkpoint_payload

        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = _checkpoint_sha(self.checkpoint)
        payload = load_rsl_checkpoint_payload(self.checkpoint, map_location="cpu")
        if _checkpoint_sha(self.checkpoint) != self.checkpoint_sha256:
            raise ValueError("RSL checkpoint changed while loading")
        self.contract = payload["field_policy_contract"]
        if type(allow_backend_transfer) is not bool:
            raise ValueError("allow_backend_transfer must be boolean")
        self.allow_backend_transfer = allow_backend_transfer
        self.validate_contract(
            input_schema,
            expected_observation_config,
            expected_backend_identity,
            allow_backend_transfer=allow_backend_transfer,
        )
        self.input_schema = input_schema
        self.field_observation_config = _json_safe(self.contract["observation_config"])
        self.field_action_config = _json_safe(self.contract["action_config"])
        self.field_encoder_config = _field_encoder_config(input_schema)
        self.field_drive_config = self.contract.get("drive_config")
        self.field_damage_config = self.contract.get("damage_config")
        self.shapes = {
            key: tuple(self.field_observation_config["observation_shapes"][key])
            for key in ("lidar", "depth", "detections", "state", "feedback")
        }
        observations = TensorDict(
            {key: torch.zeros((1, *shape), device=self.device) for key, shape in self.shapes.items()},
            batch_size=[1],
        )
        actor_config = dict(self.contract["actor_config"])
        output_dim = actor_config.pop("output_dim", 4)
        obs_set = actor_config.pop("obs_set", "actor")
        actor_config.pop("class_name", None)
        if output_dim != 4 or obs_set != "actor":
            raise ValueError("RSL checkpoint must identify a four-action actor")
        self.actor = RslFieldRNNModel(
            observations, self.contract["obs_groups"], obs_set, output_dim, **actor_config,
        ).to(self.device)
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.actor.eval()
        for parameter in self.actor.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError("RSL actor checkpoint contains nonfinite parameters")
        # An iteration index alone is not a verified transition count.
        self.num_timesteps = payload.get("num_timesteps")
        self._critic = None

    @property
    def critic(self):
        """Load the real critic only when a diagnostic explicitly requests it."""
        if self._critic is None:
            from .rsl_rl_policy import RslFieldRNNModel, load_rsl_checkpoint_payload
            if _checkpoint_sha(self.checkpoint) != self.checkpoint_sha256:
                raise ValueError("RSL checkpoint changed since actor loading")
            payload = load_rsl_checkpoint_payload(self.checkpoint, map_location='cpu')
            if _checkpoint_sha(self.checkpoint) != self.checkpoint_sha256:
                raise ValueError("RSL checkpoint changed while loading the critic")
            observations = TensorDict({key: torch.zeros((1, *shape), device=self.device)
                                       for key, shape in self.shapes.items()}, batch_size=[1])
            config = dict(self.contract['critic_config'])
            output_dim, obs_set = config.pop('output_dim'), config.pop('obs_set')
            config.pop('class_name', None)
            if output_dim != 1 or obs_set != 'critic':
                raise ValueError('RSL checkpoint does not identify the critic')
            critic = RslFieldRNNModel(observations, self.contract['obs_groups'], obs_set,
                                      output_dim, cnns=self.actor.cnns, **config).to(self.device)
            critic.load_state_dict(payload['critic_state_dict'], strict=True)
            critic.eval()
            if any(not torch.isfinite(parameter).all() for parameter in critic.parameters()):
                raise ValueError('RSL critic checkpoint contains nonfinite parameters')
            self._critic = critic
        return self._critic

    def validate_contract(
        self, input_schema: str, expected_observation_config: Mapping[str, Any] | None,
        expected_backend_identity: Mapping[str, Any] | None = None,
        *, allow_backend_transfer: bool = False,
    ) -> None:
        if self.contract.get("backend") != "rsl_rl":
            raise ValueError("checkpoint does not identify the RSL-RL backend")
        if self.contract.get("input_schema") != input_schema:
            raise ValueError("RSL checkpoint input_schema differs from the live environment")
        _validate_observation_config(
            self.contract["observation_config"], input_schema,
            expected_observation_config=expected_observation_config,
        )
        if self.contract.get("action_config") != action_config():
            raise ValueError("RSL checkpoint action contract differs from direct-track-speed-v1")
        actual_backend_identity = {
            key: self.contract.get(key) for key in BACKEND_IDENTITY_KEYS
        }
        if any(value is not None for value in actual_backend_identity.values()):
            if allow_backend_transfer:
                recorded_backend_identity(actual_backend_identity)
            else:
                validate_backend_identity(actual_backend_identity)
        if expected_backend_identity is not None:
            expected = validate_backend_identity(expected_backend_identity)
            if actual_backend_identity != expected and not allow_backend_transfer:
                raise ValueError(
                    "RSL checkpoint simulator/physics identity differs from the live environment"
                )

    def _predict(self, model, observation, state=None, episode_start=None, *, deterministic=True, clip=False):
        if not isinstance(observation, Mapping) or set(observation) != set(self.shapes):
            raise ValueError("RSL prediction requires the five sensor observation leaves")
        values = {}
        batch = None
        unbatched = None
        for key, shape in self.shapes.items():
            value = torch.as_tensor(observation[key], dtype=torch.float32, device=self.device)
            single = tuple(value.shape) == shape
            if single:
                value = value.unsqueeze(0)
            if value.ndim != len(shape) + 1 or tuple(value.shape[1:]) != shape:
                raise ValueError(f"RSL observation {key} has an incompatible shape")
            if batch is not None and (batch != value.shape[0] or unbatched != single):
                raise ValueError("RSL observation leaves have inconsistent batch axes")
            if not torch.isfinite(value).all():
                raise ValueError(f"RSL observation {key} contains nonfinite values")
            batch, unbatched = value.shape[0], single
            values[key] = value
        rnn = model.rnn.rnn
        hidden_shape = (rnn.num_layers, batch, rnn.hidden_size)
        if state is None:
            hidden = tuple(torch.zeros(hidden_shape, device=self.device) for _ in range(2))
        else:
            if not isinstance(state, (tuple, list)) or len(state) != 2:
                raise ValueError("RSL LSTM state must contain h and c")
            hidden = tuple(torch.as_tensor(value, device=self.device, dtype=torch.float32).clone() for value in state)
            if any(tuple(value.shape) != hidden_shape or not torch.isfinite(value).all() for value in hidden):
                raise ValueError("RSL LSTM state has incompatible shape or nonfinite values")
        if episode_start is not None:
            starts = torch.as_tensor(episode_start, device=self.device, dtype=torch.bool).reshape(-1)
            if starts.shape != (batch,):
                raise ValueError("episode_start must identify each observation row")
            for value in hidden:
                value[:, starts, :] = 0
        previous = model.get_hidden_state()
        try:
            model.reset(hidden_state=hidden)
            with torch.inference_mode():
                actions = model(TensorDict(values, batch_size=[batch]), stochastic_output=not deterministic)
                next_state = tuple(value.detach().cpu().numpy().copy() for value in model.get_hidden_state())
                if clip:
                    actions = actions.clamp(-1, 1)
                actions = actions.detach().cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError("RSL actor produced nonfinite actions")
        finally:
            model.reset(hidden_state=previous)
        return (actions[0] if unbatched else actions), next_state

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return self._predict(self.actor, observation, state, episode_start, deterministic=deterministic, clip=True)

    def predict_values(self, observation, state=None, episode_start=None):
        return self._predict(self.critic, observation, state, episode_start)


class RslPolicyRunner:
    """Stateful field policy boundary shared by production and paused viewers."""

    backend = 'rsl_rl'

    def __init__(self, env, checkpoint, *, device='cpu', input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA,
                 expected_observation_config=None, allow_backend_transfer=False):
        if checkpoint is None:
            raise ValueError('RSL inference requires an explicit .pt checkpoint')
        expected = expected_observation_config
        if expected is None:
            expected = getattr(env, 'observation_config', None)
        self.model = RslPredictionModel(checkpoint, device=device, input_schema=input_schema,
                                        expected_observation_config=expected,
                                        expected_backend_identity=BACKEND_IDENTITY,
                                        allow_backend_transfer=allow_backend_transfer)
        self.input_schema = input_schema
        self.observation_config = self.model.field_observation_config
        self.num_envs = int(getattr(env, 'num_envs', 1))
        self.state = None
        self.episode_start = np.ones(self.num_envs, dtype=bool)
        self.label = 'RSL checkpoint loaded; control competence is not established'

    def reset(self, done=None):
        if done is None:
            self.state = None
            self.episode_start = np.ones(self.num_envs, dtype=bool)
            return
        done = np.asarray(done, dtype=bool).reshape(-1)
        if done.shape != (self.num_envs,):
            raise ValueError(f'done must have shape ({self.num_envs},)')
        if self.state is not None:
            self.state = tuple(np.asarray(value).copy() for value in self.state)
            for value in self.state:
                value[:, done, :] = 0
        self.episode_start = done.copy()

    def action(self, observation):
        action, state = self.model.predict(observation, state=self.state,
                                           episode_start=self.episode_start, deterministic=True)
        self.state = state
        self.episode_start = np.zeros(self.num_envs, dtype=bool)
        return action
