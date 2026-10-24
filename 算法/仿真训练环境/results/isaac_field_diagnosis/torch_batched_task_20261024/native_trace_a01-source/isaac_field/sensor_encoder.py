"""SB3-independent PyTorch field sensor encoder shared across RL backends."""
from __future__ import annotations

import math

import torch
from torch import nn

from .feedback import FEEDBACK_DIM, INPUT_SCHEMA
from .policy_contract import BRANCH_DIMS, FEATURES_DIM, _encoder_schema_for_input, validate_observation_space


def detection_presence_mask(value: torch.Tensor) -> torch.Tensor:
    """Return a singleton-channel mask for boxes with positive width and height."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("detection_presence_mask expects a torch.Tensor")
    if value.ndim < 2 or value.shape[-1] != 8:
        raise ValueError(f"detections must end in 8 fields, got shape {tuple(value.shape)}")
    return (value[..., 2:3] > 0) & (value[..., 3:4] > 0)


class _SensorEncoderImplementation:
    """Shared module layout and forward path, installed directly on each encoder."""

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def _initialize_sensor_encoder(
        self,
        observation_space,
        *,
        cache_inference: bool,
        goal_gain: float,
        encoder_schema: str | None,
        input_schema: str,
    ) -> None:
        expected_encoder_schema = _encoder_schema_for_input(input_schema)
        validate_observation_space(observation_space)
        goal_gain = float(goal_gain)
        if not math.isfinite(goal_gain) or goal_gain != 1.0:
            raise ValueError("goal_gain must be 1.0: extra goal-input scaling has been removed")
        if encoder_schema is None:
            encoder_schema = expected_encoder_schema
        if encoder_schema != expected_encoder_schema:
            raise ValueError(
                f"unsupported encoder_schema {encoder_schema!r} for {input_schema!r}; "
                f"expected {expected_encoder_schema!r}"
            )

        self._features_dim = FEATURES_DIM
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


class SensorEncoderCore(_SensorEncoderImplementation, nn.Module):
    """Pure ``torch.nn.Module`` encoder for RSL-RL and other PyTorch callers.

    The five branch modules are registered directly on this instance. Their
    names and tensor shapes match the long-standing SB3 ``policy.SensorEncoder``
    layout so existing checkpoint tensors can be loaded without remapping or
    changing encoder semantics.
    """

    branch_dims = BRANCH_DIMS

    def __init__(
        self,
        observation_space,
        cache_inference: bool = True,
        goal_gain: float = 1.0,
        encoder_schema: str | None = None,
        input_schema: str = INPUT_SCHEMA,
    ) -> None:
        nn.Module.__init__(self)
        self._initialize_sensor_encoder(
            observation_space,
            cache_inference=cache_inference,
            goal_gain=goal_gain,
            encoder_schema=encoder_schema,
            input_schema=input_schema,
        )


__all__ = ["SensorEncoderCore", "detection_presence_mask"]
