"""Load RSL checkpoints for the existing, backend-independent episode evaluator.

Only the prediction boundary is adapted. Training and recurrent computation
remain in RSL-RL; the evaluator continues to own resets and trajectory records.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from .action_contract import action_config
from .feedback import ABSOLUTE_GOAL_INPUT_SCHEMA
from .policy import _field_encoder_config, _json_safe, _validate_observation_config


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
    ):
        from .rsl_rl_policy import RslFieldRNNModel, load_rsl_checkpoint_payload

        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        payload = load_rsl_checkpoint_payload(self.checkpoint, map_location="cpu")
        self.contract = payload["field_policy_contract"]
        self.validate_contract(input_schema, expected_observation_config)
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

    def validate_contract(
        self, input_schema: str, expected_observation_config: Mapping[str, Any] | None,
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

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
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
        rnn = self.actor.rnn.rnn
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
        previous = self.actor.get_hidden_state()
        try:
            self.actor.reset(hidden_state=hidden)
            with torch.inference_mode():
                actions = self.actor(TensorDict(values, batch_size=[batch]), stochastic_output=not deterministic)
                next_state = tuple(value.detach().cpu().numpy().copy() for value in self.actor.get_hidden_state())
                actions = actions.clamp(-1, 1).detach().cpu().numpy()
            if not np.isfinite(actions).all():
                raise ValueError("RSL actor produced nonfinite actions")
        finally:
            self.actor.reset(hidden_state=previous)
        return (actions[0] if unbatched else actions), next_state
