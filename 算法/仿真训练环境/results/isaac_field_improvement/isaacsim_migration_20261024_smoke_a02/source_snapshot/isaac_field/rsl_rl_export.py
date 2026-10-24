"""Export the field RSL-RL actor with its five sensor inputs.

RSL-RL's built-in recurrent exporters accept one already-fused observation
tensor.  The field actor receives five separate sensor tensors, so these small
wrappers include the existing sensor branches before delegating to the copied
RSL-RL normalizer, LSTM, MLP, and deterministic Gaussian output module.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
from torch import nn

from .policy_contract import ACTION_SHAPE, FEATURES_DIM
from .sensor_encoder import SensorEncoderCore as SensorEncoder


def _validate_actor(model: nn.Module) -> tuple[SensorEncoder, nn.LSTM, nn.Module, nn.Module, nn.Module]:
    """Check that ``model`` is the supported recurrent Gaussian actor."""
    try:
        from rsl_rl.models.rnn_model import RNNModel
        from rsl_rl.modules.distribution import GaussianDistribution
    except ImportError as exc:
        raise RuntimeError("RSL-RL is required to export the field actor") from exc

    if not isinstance(model, RNNModel):
        raise TypeError("model must be an RSL-RL RNNModel actor")
    if getattr(model, "obs_set", None) != "actor":
        raise ValueError("field export accepts the actor only, not a critic model")
    encoder = getattr(model, "sensor_encoder", None)
    if not isinstance(encoder, SensorEncoder):
        raise TypeError("model.sensor_encoder must be isaac_field.sensor_encoder.SensorEncoderCore")
    recurrent = getattr(getattr(model, "rnn", None), "rnn", None)
    if not isinstance(recurrent, nn.LSTM):
        raise TypeError("field export supports an LSTM actor")
    distribution = getattr(model, "distribution", None)
    if not isinstance(distribution, GaussianDistribution):
        raise TypeError("field export supports the deterministic mean of a Gaussian actor")
    if distribution.output_dim != ACTION_SHAPE[0]:
        raise ValueError(f"actor Gaussian must have {ACTION_SHAPE[0]} outputs")
    if int(getattr(encoder, "features_dim", -1)) != FEATURES_DIM:
        raise ValueError(f"sensor encoder must produce {FEATURES_DIM} features")
    if recurrent.input_size != FEATURES_DIM:
        raise ValueError("actor LSTM input size does not match the sensor encoder")
    mlp = getattr(model, "mlp", None)
    normalizer = getattr(model, "obs_normalizer", None)
    if not isinstance(mlp, nn.Module) or not isinstance(normalizer, nn.Module):
        raise TypeError("actor must expose the RSL-RL MLP and observation normalizer")
    return encoder, recurrent, mlp, normalizer, distribution


class _FieldSensorBranches(nn.Module):
    """Scriptable copies of the five existing SensorEncoder branches."""

    def __init__(self, encoder: SensorEncoder) -> None:
        super().__init__()
        # Copy only learned branches; the rollout cache and SB3 base class are
        # intentionally excluded from the portable inference graph.
        self.laser = copy.deepcopy(encoder.laser)
        self.depth = copy.deepcopy(encoder.depth)
        self.boxes = copy.deepcopy(encoder.boxes)
        self.proprioception = copy.deepcopy(encoder.proprioception)
        self.feedback = copy.deepcopy(encoder.feedback)

    def forward(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        detections: torch.Tensor,
        state: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        lidar_features = self.laser(lidar)
        depth_features = self.depth(depth)
        box_features = self.boxes(detections)
        # Match detection_presence_mask: a row is present iff both dimensions
        # are positive; masked box features are max-pooled across detections.
        present = (detections[..., 2:3] > 0) & (detections[..., 3:4] > 0)
        box_features = (box_features * present.to(dtype=detections.dtype)).amax(dim=1)
        state_features = self.proprioception(state)
        feedback_features = self.feedback(feedback)
        return torch.cat(
            (lidar_features, depth_features, box_features, state_features, feedback_features), dim=1
        )


class _FieldTorchLSTM(nn.Module):
    """Stateful, single-environment JIT export matching RSL-RL's JIT contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        encoder, recurrent, mlp, normalizer, distribution = _validate_actor(model)
        self.sensor_encoder = _FieldSensorBranches(encoder)
        self.obs_normalizer = copy.deepcopy(normalizer)
        self.rnn = copy.deepcopy(recurrent)
        self.mlp = copy.deepcopy(mlp)
        self.deterministic_output = distribution.as_deterministic_output_module()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.eval()

    def forward(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        detections: torch.Tensor,
        state: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sensor_encoder(lidar, depth, detections, state, feedback)
        x = self.obs_normalizer(x)
        x, (hidden, cell) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = hidden
        self.cell_state[:] = cell
        x = self.mlp(x.squeeze(0))
        return self.deterministic_output(x)

    @torch.jit.export
    def reset(self) -> None:
        """Clear the single deployed environment's recurrent state."""
        self.hidden_state.zero_()
        self.cell_state.zero_()


class _FieldOnnxLSTM(nn.Module):
    """Stateless ONNX step with explicit recurrent state tensors."""

    is_recurrent: bool = True

    def __init__(self, model: nn.Module, verbose: bool = False) -> None:
        super().__init__()
        encoder, recurrent, mlp, normalizer, distribution = _validate_actor(model)
        self.verbose = bool(verbose)
        self.sensor_encoder = _FieldSensorBranches(encoder)
        self.obs_normalizer = copy.deepcopy(normalizer)
        self.rnn = copy.deepcopy(recurrent)
        self.mlp = copy.deepcopy(mlp)
        self.deterministic_output = distribution.as_deterministic_output_module()
        self.num_layers = recurrent.num_layers
        self.hidden_size = recurrent.hidden_size
        self.eval()

    def forward(
        self,
        lidar: torch.Tensor,
        depth: torch.Tensor,
        detections: torch.Tensor,
        state: torch.Tensor,
        feedback: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.sensor_encoder(lidar, depth, detections, state, feedback)
        x = self.obs_normalizer(x)
        x, (h_out, c_out) = self.rnn(x.unsqueeze(0), (h_in, c_in))
        mean = self.deterministic_output(self.mlp(x.squeeze(0)))
        return mean, h_out, c_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return CPU tensors shaped like one field-environment observation."""
        return (
            torch.zeros(1, 1, 400),
            torch.zeros(1, 1, 160, 88),
            torch.zeros(1, 16, 8),
            torch.zeros(1, 33),
            torch.zeros(1, 37),
            torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["lidar", "depth", "detections", "state", "feedback", "h_in", "c_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out", "c_out"]


def field_model_as_jit(model: nn.Module) -> nn.Module:
    """Build the stateful field actor wrapper consumed by the RSL-RL exporter."""
    return _FieldTorchLSTM(model)


def field_model_as_onnx(model: nn.Module, verbose: bool = False) -> nn.Module:
    """Build the explicit-state field actor wrapper consumed by RSL-RL export."""
    return _FieldOnnxLSTM(model, verbose=verbose)
