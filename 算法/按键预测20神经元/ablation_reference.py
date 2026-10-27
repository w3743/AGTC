"""Exact online RTRL reference for small 20-unit UORO ablations.

This class carries the shared-parameter state sensitivity of the recurrent
state. It is an online recurrence: after an optimizer update, the old state
and sensitivity are advanced with the post-update parameters. It is not the
gradient through the optimizer or through an Adam-updated historical
trajectory.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from online_model import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANK,
    DEFAULT_SPARSITY_STRENGTH,
    INPUT_COUNT,
    NEURON_COUNT,
    PARAMETER_SIZE,
    STATE_SIZE,
    OnlinePlasticRNN,
    _validate_digit,
    functional_logits,
    functional_state_step,
)


_INPUT_WEIGHT_SIZE = NEURON_COUNT * INPUT_COUNT
_RECURRENT_WEIGHT_SIZE = NEURON_COUNT * NEURON_COUNT
_BIAS_START = _INPUT_WEIGHT_SIZE + _RECURRENT_WEIGHT_SIZE
_ETA_INDEX = _BIAS_START + NEURON_COUNT
_DECAY_INDEX = _ETA_INDEX + 1


def _rtrl_transition(
    digit: int,
    state: Tensor,
    theta: Tensor,
    sensitivity: Tensor,
) -> tuple[Tensor, Tensor]:
    """Advance ``state`` and its exact first-order shared-parameter RTRL map.

    ``sensitivity`` has shape ``[420, 623]`` and stores d(state)/d(theta).
    The local recurrence is expanded in blocks so each step costs roughly
    O(20^2 * 623), rather than multiplying a dense 420x420 matrix.
    """

    digit = _validate_digit(digit)
    if tuple(state.shape) != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},)")
    if tuple(theta.shape) != (PARAMETER_SIZE,):
        raise ValueError(f"theta must have shape ({PARAMETER_SIZE},)")
    if tuple(sensitivity.shape) != (STATE_SIZE, PARAMETER_SIZE):
        raise ValueError(
            f"sensitivity must have shape ({STATE_SIZE}, {PARAMETER_SIZE})"
        )

    n = NEURON_COUNT
    old_h = state[:n]
    old_fast = state[n:].view(n, n)
    input_weights = theta[:_INPUT_WEIGHT_SIZE].view(n, INPUT_COUNT)
    recurrent = theta[_INPUT_WEIGHT_SIZE : _INPUT_WEIGHT_SIZE + _RECURRENT_WEIGHT_SIZE].view(n, n)
    bias = theta[_BIAS_START : _BIAS_START + n]
    eta_raw = theta[_ETA_INDEX]
    decay_logit = theta[_DECAY_INDEX]
    eta = 0.5 * torch.tanh(eta_raw)
    d_eta = 0.5 * (1.0 - torch.tanh(eta_raw).square())
    decay = torch.sigmoid(decay_logit)
    d_decay = decay * (1.0 - decay)

    one_hot = F.one_hot(
        torch.tensor(digit, dtype=torch.long, device=theta.device),
        num_classes=INPUT_COUNT,
    ).to(dtype=theta.dtype)
    effective_recurrent = recurrent + eta * old_fast
    next_state = functional_state_step(digit, state, theta)
    next_h = next_state[:n]
    slope = 1.0 - next_h.square()

    # Local derivative of h' with respect to old h, followed by the direct
    # parameter derivative of h'.
    dh_dh = slope[:, None] * effective_recurrent
    dh_dtheta = theta.new_zeros((n, PARAMETER_SIZE))
    dh_dtheta[:, :_INPUT_WEIGHT_SIZE] = (
        torch.eye(n, dtype=theta.dtype, device=theta.device)[:, :, None]
        * slope[:, None, None]
        * one_hot[None, None, :]
    ).reshape(n, _INPUT_WEIGHT_SIZE)
    r_start = _INPUT_WEIGHT_SIZE
    r_end = r_start + _RECURRENT_WEIGHT_SIZE
    dh_dtheta[:, r_start:r_end] = (
        torch.eye(n, dtype=theta.dtype, device=theta.device)[:, :, None]
        * slope[:, None, None]
        * old_h[None, None, :]
    ).reshape(n, _RECURRENT_WEIGHT_SIZE)
    dh_dtheta[:, _BIAS_START : _BIAS_START + n] = torch.diag(slope)
    dh_dtheta[:, _ETA_INDEX] = slope * d_eta * (old_fast @ old_h)

    old_dh = sensitivity[:n]
    old_dfast = sensitivity[n:].view(n, n, PARAMETER_SIZE)
    # d(h')/d(F) is row-local: d h'_i / d F_ij = slope_i * eta * h_j.
    fast_weighted_sensitivity = torch.einsum("j,ijp->ip", old_h, old_dfast)
    next_dh = (
        dh_dh @ old_dh
        + (slope * eta)[:, None] * fast_weighted_sensitivity
        + dh_dtheta
    )

    # F' = decay*F + (1-decay)*(h' outer h), including the direct derivative
    # through h', the old source activity, and the decay logit.
    next_dfast = (
        decay * old_dfast
        + (1.0 - decay)
        * (
            next_dh[:, None, :] * old_h[None, :, None]
            + next_h[:, None, None] * old_dh[None, :, :]
        )
    )
    old_outer = torch.outer(next_h, old_h)
    old_fast_for_decay = old_fast - old_outer
    next_dfast[:, :, _DECAY_INDEX] += d_decay * old_fast_for_decay

    next_sensitivity = torch.cat(
        (next_dh, next_dfast.reshape(_RECURRENT_WEIGHT_SIZE, PARAMETER_SIZE)),
        dim=0,
    )
    return next_state.detach(), next_sensitivity.detach()


class ExactOnlineRNN(OnlinePlasticRNN):
    """Persistent exact-RTRL reference with the production online API.

    ``rank`` is accepted so an ablation harness can construct this model with
    the same configuration as UORO. It does not affect this full-sensitivity
    estimator.
    """

    def __init__(
        self,
        seed: int = 0,
        rank: int = DEFAULT_RANK,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        sparsity_strength: float = DEFAULT_SPARSITY_STRENGTH,
        weight_decay: float = 0.0,
    ) -> None:
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        super().__init__(
            seed=seed,
            rank=rank,
            learning_rate=learning_rate,
            sparsity_strength=sparsity_strength,
            exact_readout_step=True,
        )
        self.weight_decay = float(weight_decay)
        self.optimizer.param_groups[0]["weight_decay"] = self.weight_decay
        self.state_sensitivity = torch.zeros(
            STATE_SIZE,
            PARAMETER_SIZE,
            dtype=self.input_weights.dtype,
            device=self.input_weights.device,
        )
        self._pending_sensitivity: Tensor | None = None

    def _cache_pending_transition(self, digit: int) -> None:
        """Cache the pre-transition state and parameters for consistency checks."""

        self._pending_base_state = self._state_vector().detach().clone()
        self._pending_uoro_a = self.uoro_a.detach().clone()
        self._pending_uoro_b = self.uoro_b.detach().clone()
        self._pending_digit = _validate_digit(digit)
        self._pending_theta = self._pack_parameters().detach().clone()
        self._pending_sensitivity = None

    def _advance_uoro_and_state(
        self,
        digit: int,
        random_signs: Tensor | Sequence[Any] | None = None,
    ) -> None:
        """Advance state and the full sensitivity using current parameters."""

        del random_signs
        digit = _validate_digit(digit)
        state = self._state_vector().detach()
        theta = self._pack_parameters().detach()
        next_state, next_sensitivity = _rtrl_transition(
            digit, state, theta, self.state_sensitivity.detach()
        )
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (next_state, next_sensitivity)
        ):
            raise FloatingPointError("non-finite recurrent state or RTRL sensitivity")
        self.h = next_state[:NEURON_COUNT].detach().clone()
        self.fast_edges = next_state[NEURON_COUNT:].view(
            NEURON_COUNT, NEURON_COUNT
        ).detach().clone()
        self.state_sensitivity = next_sensitivity
        if (
            self._pending_digit == digit
            and self._pending_base_state is not None
            and torch.equal(self._pending_base_state, state)
            and self._pending_theta is not None
            and torch.equal(self._pending_theta, theta)
        ):
            self._pending_sensitivity = next_sensitivity.detach().clone()

    def _estimate_current_loss_gradient(self, target: int) -> tuple[float, Tensor]:
        """Return the current CE gradient from the carried RTRL sensitivity."""

        target = _validate_digit(target)
        if any(
            value is None
            for value in (
                self._pending_base_state,
                self._pending_digit,
                self._pending_theta,
                self._pending_sensitivity,
            )
        ):
            raise RuntimeError("missing cached transition for exact RTRL sensitivity")

        theta = self._pack_parameters().detach().requires_grad_(True)
        if not torch.equal(theta.detach(), self._pending_theta):
            raise RuntimeError("parameters changed since the pending prediction transition")
        state = self._state_vector().detach().requires_grad_(True)
        expected_state = functional_state_step(self._pending_digit, self._pending_base_state, theta)
        if not torch.equal(expected_state.detach(), state.detach()):
            raise RuntimeError("cached transition does not match the carried state")
        if not torch.equal(self._pending_sensitivity, self.state_sensitivity):
            raise RuntimeError("cached sensitivity does not match the carried state")

        logits = functional_logits(state, theta)
        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor.unsqueeze(0))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite training loss")
        direct_gradient, state_gradient = torch.autograd.grad(loss, (theta, state))
        gradient = direct_gradient.detach() + state_gradient.detach() @ self.state_sensitivity
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("non-finite exact RTRL gradient")
        return float(loss.detach().item()), gradient.detach()

    def config(self) -> dict[str, Any]:
        config = super().config()
        config.update(
            {
                "mode": "exact_rtrl",
                "gradient_method": "shared_parameter_sensitivity_online_recurrence",
                "rtrl_state_sensitivity_shape": [STATE_SIZE, PARAMETER_SIZE],
                "rank_used": False,
                "weight_decay": self.weight_decay,
            }
        )
        return config

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        active_bytes = int(
            self.state_sensitivity.numel() * self.state_sensitivity.element_size()
        )
        pending_bytes = int(
            self._pending_sensitivity.numel() * self._pending_sensitivity.element_size()
        ) if self._pending_sensitivity is not None else 0
        unused_uoro_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                self.uoro_a,
                self.uoro_b,
                self._pending_uoro_a,
                self._pending_uoro_b,
            )
            if value is not None
        )
        pending_exact_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                self._pending_base_state,
                self._pending_theta,
                self._pending_sensitivity,
            )
            if value is not None
        )
        diagnostics.update(
            {
                "mode": "exact_rtrl",
                "rank_used": False,
                "weight_decay": self.weight_decay,
                "sensitivity_bytes": active_bytes,
                "pending_sensitivity_bytes": pending_bytes,
                "unused_uoro_buffer_bytes": int(unused_uoro_bytes),
                "factor_bytes": active_bytes + pending_bytes,
                "pending_exact_cache_bytes": int(pending_exact_bytes),
                "exact_state_sensitivity_norm": float(
                    torch.linalg.vector_norm(self.state_sensitivity).item()
                ),
                "finite": bool(diagnostics["finite"])
                and bool(torch.isfinite(self.state_sensitivity).all().item())
                and (
                    self._pending_sensitivity is None
                    or bool(torch.isfinite(self._pending_sensitivity).all().item())
                ),
            }
        )
        return diagnostics

    def checkpoint_state(self) -> dict[str, Any]:
        checkpoint = super().checkpoint_state()
        checkpoint["config"] = self.config()
        checkpoint["runtime"]["state_sensitivity"] = (
            self.state_sensitivity.detach().cpu().clone()
        )
        checkpoint["runtime"]["pending_sensitivity"] = (
            self._pending_sensitivity.detach().cpu().clone()
            if self._pending_sensitivity is not None
            else None
        )
        return checkpoint

    def load_checkpoint_state(self, checkpoint: Mapping[str, Any]) -> None:
        config = checkpoint.get("config")
        runtime = checkpoint.get("runtime")
        if not isinstance(config, Mapping) or config.get("mode") != "exact_rtrl":
            raise ValueError("checkpoint is not for ExactOnlineRNN")
        if not isinstance(runtime, Mapping):
            raise ValueError("checkpoint is missing runtime state")
        weight_decay = config.get("weight_decay")
        if (
            not isinstance(weight_decay, (int, float))
            or not math.isfinite(float(weight_decay))
            or weight_decay < 0
        ):
            raise ValueError("checkpoint has an invalid weight decay")

        super().load_checkpoint_state(checkpoint)
        self.weight_decay = float(weight_decay)
        self.optimizer.param_groups[0]["weight_decay"] = self.weight_decay

        def restore_sensitivity(value: Any, name: str) -> Tensor:
            if not isinstance(value, Tensor) or tuple(value.shape) != (
                STATE_SIZE,
                PARAMETER_SIZE,
            ):
                raise ValueError(f"checkpoint field {name!r} has the wrong shape")
            restored = value.to(device=self.h.device, dtype=self.h.dtype).detach().clone()
            if not bool(torch.isfinite(restored).all().item()):
                raise ValueError(f"checkpoint field {name!r} is non-finite")
            return restored

        self.state_sensitivity = restore_sensitivity(
            runtime.get("state_sensitivity"), "state_sensitivity"
        )
        pending_value = runtime.get("pending_sensitivity")
        if pending_value is None:
            self._pending_sensitivity = None
            if self.observed_steps > 0:
                raise ValueError("checkpoint is missing pending exact sensitivity")
        else:
            self._pending_sensitivity = restore_sensitivity(
                pending_value, "pending_sensitivity"
            )
            if not torch.equal(self._pending_sensitivity, self.state_sensitivity):
                raise ValueError("checkpoint pending sensitivity does not match carried state")


class TruncatedOnlineRNN(OnlinePlasticRNN):
    """Continuous-state online model with an exact finite credit horizon.

    Each transition is saved at its actual forward point. For the current loss,
    local VJPs are applied in reverse over the most recent ``credit_horizon``
    transitions and their parameter contributions are summed. The initial
    state at the horizon boundary is treated as constant. This is a truncated
    shared-parameter sensitivity recurrence, not a replay of the input window
    under current parameters and not differentiation through optimizer steps.
    """

    def __init__(
        self,
        seed: int = 0,
        rank: int = DEFAULT_RANK,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        sparsity_strength: float = DEFAULT_SPARSITY_STRENGTH,
        weight_decay: float = 0.0,
        credit_horizon: int = 32,
    ) -> None:
        if type(credit_horizon) is not int or credit_horizon < 1:
            raise ValueError("credit_horizon must be a positive integer")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        super().__init__(
            seed=seed,
            rank=rank,
            learning_rate=learning_rate,
            sparsity_strength=sparsity_strength,
            exact_readout_step=True,
        )
        self.weight_decay = float(weight_decay)
        self.optimizer.param_groups[0]["weight_decay"] = self.weight_decay
        self.credit_horizon = credit_horizon
        self._transition_history: deque[tuple[int, Tensor, Tensor]] = deque(
            maxlen=credit_horizon
        )

    def _advance_uoro_and_state(
        self,
        digit: int,
        random_signs: Tensor | Sequence[Any] | None = None,
    ) -> None:
        """Advance persistent state and record the exact forward transition."""

        del random_signs
        digit = _validate_digit(digit)
        pre_state = self._state_vector().detach().clone()
        theta_used = self._pack_parameters().detach().clone()
        next_state = functional_state_step(digit, pre_state, theta_used).detach()
        if not bool(torch.isfinite(next_state).all().item()):
            raise FloatingPointError("non-finite recurrent state")
        self.h = next_state[:NEURON_COUNT].detach().clone()
        self.fast_edges = next_state[NEURON_COUNT:].view(
            NEURON_COUNT, NEURON_COUNT
        ).detach().clone()
        self._transition_history.append((digit, pre_state, theta_used))

    def _estimate_current_loss_gradient(self, target: int) -> tuple[float, Tensor]:
        """Backpropagate the current loss through the retained local transitions."""

        target = _validate_digit(target)
        if any(
            value is None
            for value in (
                self._pending_base_state,
                self._pending_digit,
                self._pending_theta,
            )
        ):
            raise RuntimeError("missing cached transition for truncated RTRL")
        theta = self._pack_parameters().detach().requires_grad_(True)
        if not torch.equal(theta.detach(), self._pending_theta):
            raise RuntimeError("parameters changed since the pending prediction transition")

        state = self._state_vector().detach().requires_grad_(True)
        expected_state = functional_state_step(
            self._pending_digit, self._pending_base_state, theta
        )
        if not torch.equal(expected_state.detach(), state.detach()):
            raise RuntimeError("cached transition does not match the carried state")

        logits = functional_logits(state, theta)
        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor.unsqueeze(0))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite training loss")
        direct_gradient, state_gradient = torch.autograd.grad(loss, (theta, state))
        gradient = direct_gradient.detach().clone()
        adjoint = state_gradient.detach()

        transitions = list(self._transition_history)
        if len(transitions) > self.credit_horizon:
            raise RuntimeError("truncated transition history exceeds its credit horizon")
        for index in range(len(transitions) - 1, -1, -1):
            digit, pre_state, theta_used = transitions[index]
            output, pullback = torch.func.vjp(
                lambda old_state, parameters: functional_state_step(
                    digit, old_state, parameters
                ),
                pre_state,
                theta_used,
            )
            expected_output = (
                transitions[index + 1][1]
                if index + 1 < len(transitions)
                else state.detach()
            )
            if not torch.equal(output.detach(), expected_output):
                raise RuntimeError("truncated transition history is not state-contiguous")
            state_adjoint, parameter_gradient = pullback(adjoint)
            gradient.add_(parameter_gradient.detach())
            adjoint = state_adjoint.detach()

        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("non-finite truncated RTRL gradient")
        return float(loss.detach().item()), gradient.detach()

    def config(self) -> dict[str, Any]:
        config = super().config()
        config.update(
            {
                "mode": "exact_truncated",
                "gradient_method": "truncated_shared_parameter_sensitivity_online_recurrence",
                "credit_horizon": self.credit_horizon,
                "rank_used": False,
                "weight_decay": self.weight_decay,
            }
        )
        return config

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        history_bytes = sum(
            pre_state.numel() * pre_state.element_size()
            + theta.numel() * theta.element_size()
            for _, pre_state, theta in self._transition_history
        )
        unused_uoro_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                self.uoro_a,
                self.uoro_b,
                self._pending_uoro_a,
                self._pending_uoro_b,
            )
            if value is not None
        )
        pending_exact_bytes = sum(
            value.numel() * value.element_size()
            for value in (self._pending_base_state, self._pending_theta)
            if value is not None
        )
        history_finite = all(
            bool(torch.isfinite(value).all().item())
            for _, pre_state, theta in self._transition_history
            for value in (pre_state, theta)
        )
        diagnostics.update(
            {
                "mode": "exact_truncated",
                "rank_used": False,
                "weight_decay": self.weight_decay,
                "credit_horizon": self.credit_horizon,
                "transition_history_length": len(self._transition_history),
                "transition_history_bytes": int(history_bytes),
                "unused_uoro_buffer_bytes": int(unused_uoro_bytes),
                "factor_bytes": int(history_bytes),
                "pending_exact_cache_bytes": int(pending_exact_bytes),
                "gradient_method": "truncated_shared_parameter_sensitivity_online_recurrence",
                "finite": bool(diagnostics["finite"]) and history_finite,
            }
        )
        return diagnostics

    def checkpoint_state(self) -> dict[str, Any]:
        checkpoint = super().checkpoint_state()
        checkpoint["config"] = self.config()
        checkpoint["runtime"]["transition_history"] = [
            {
                "digit": digit,
                "pre_state": pre_state.detach().cpu().clone(),
                "theta_used": theta.detach().cpu().clone(),
            }
            for digit, pre_state, theta in self._transition_history
        ]
        return checkpoint

    def load_checkpoint_state(self, checkpoint: Any) -> None:
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        runtime = checkpoint.get("runtime") if isinstance(checkpoint, dict) else None
        if not isinstance(config, dict) or config.get("mode") != "exact_truncated":
            raise ValueError("checkpoint is not for TruncatedOnlineRNN")
        horizon = config.get("credit_horizon")
        weight_decay = config.get("weight_decay")
        history = runtime.get("transition_history") if isinstance(runtime, dict) else None
        if type(horizon) is not int or horizon < 1:
            raise ValueError("checkpoint has an invalid credit horizon")
        if not isinstance(weight_decay, (int, float)) or not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("checkpoint has an invalid weight decay")
        if not isinstance(history, list) or len(history) > horizon:
            raise ValueError("checkpoint has an invalid truncated transition history")

        super().load_checkpoint_state(checkpoint)
        self.credit_horizon = horizon
        self.weight_decay = float(weight_decay)
        self.optimizer.param_groups[0]["weight_decay"] = self.weight_decay
        restored: deque[tuple[int, Tensor, Tensor]] = deque(maxlen=horizon)
        for index, entry in enumerate(history):
            if not isinstance(entry, dict):
                raise ValueError("checkpoint transition history entry must be a mapping")
            digit = _validate_digit(entry.get("digit"))
            pre_state = entry.get("pre_state")
            theta_used = entry.get("theta_used")
            if not isinstance(pre_state, Tensor) or tuple(pre_state.shape) != (STATE_SIZE,):
                raise ValueError("checkpoint transition history has an invalid pre-state")
            if not isinstance(theta_used, Tensor) or tuple(theta_used.shape) != (PARAMETER_SIZE,):
                raise ValueError("checkpoint transition history has invalid parameters")
            pre_state = pre_state.to(device=self.h.device, dtype=self.h.dtype).detach().clone()
            theta_used = theta_used.to(device=self.h.device, dtype=self.h.dtype).detach().clone()
            if not bool(torch.isfinite(pre_state).all().item()) or not bool(torch.isfinite(theta_used).all().item()):
                raise ValueError("checkpoint transition history contains non-finite values")
            if restored:
                previous_digit, previous_state, previous_theta = restored[-1]
                previous_output = functional_state_step(previous_digit, previous_state, previous_theta)
                if not torch.equal(previous_output, pre_state):
                    raise ValueError(f"checkpoint transition history is discontinuous at item {index}")
            restored.append((digit, pre_state, theta_used))
        if restored:
            last_digit, last_pre_state, last_theta = restored[-1]
            if not torch.equal(
                functional_state_step(last_digit, last_pre_state, last_theta),
                self._state_vector(),
            ):
                raise ValueError("checkpoint transition history does not reach carried state")
        elif self.observed_steps > 0:
            raise ValueError("checkpoint is missing truncated transition history")
        self._transition_history = restored
