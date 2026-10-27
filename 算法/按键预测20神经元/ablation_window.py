"""Configurable recent-window control for the 20-neuron key predictor.

This module is deliberately isolated from the app's production model. It
inherits the existing PlasticRNN transition and changes only the online
training controls needed by the ablation harness.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import copy
import math
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

try:  # Support both direct script execution and package-style imports.
    from .model import CONTEXT_LENGTH, INPUT_COUNT, NEURON_COUNT, PlasticRNN
except ImportError:  # pragma: no cover - selected by direct script execution
    from model import CONTEXT_LENGTH, INPUT_COUNT, NEURON_COUNT, PlasticRNN


WINDOW_CHECKPOINT_FORMAT_VERSION = 1
GRADIENT_CLIP_MAX_NORM = 1.0


class WindowControlRNN(PlasticRNN):
    """PlasticRNN with independently selectable window-training controls.

    ``observe`` appends to an internal context and trains on the target just
    observed. The first observation only seeds the context. ``predict_next``
    reads the cached context and replays it from a clean state; it does not
    mutate context, optimizer state, or model parameters.
    """

    def __init__(
        self,
        seed: int = 0,
        learning_rate: float = 0.015,
        updates_per_press: int = 3,
        weight_decay: float = 1e-4,
        sparsity_strength: float = 0.0,
        loss_scope: str = "all",
        training_context_alignment: str = "legacy_including_target",
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(weight_decay)) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(float(sparsity_strength)) or sparsity_strength < 0:
            raise ValueError("sparsity_strength must be finite and non-negative")
        if loss_scope not in {"all", "latest"}:
            raise ValueError("loss_scope must be 'all' or 'latest'")
        if training_context_alignment not in {
            "legacy_including_target",
            "sealed_prefix",
        }:
            raise ValueError(
                "training_context_alignment must be 'legacy_including_target' "
                "or 'sealed_prefix'"
            )

        super().__init__(
            seed=seed,
            context_length=context_length,
            updates_per_press=updates_per_press,
            learning_rate=learning_rate,
        )
        self.seed = int(seed)
        self.weight_decay = float(weight_decay)
        self.sparsity_strength = float(sparsity_strength)
        self.loss_scope = loss_scope
        self.training_context_alignment = training_context_alignment
        self.optimizer.param_groups[0]["weight_decay"] = self.weight_decay

        self.history: list[int] = []
        self.observed_steps = 0
        self.clipped_updates = 0
        self.last_gradient_norm: float | None = None
        self.last_applied_gradient_norm: float | None = None
        self.last_gradient_was_clipped: bool | None = None
        self.last_training_ms = 0.0
        self.last_training_context: list[int] = []

    def observe(self, digit: int) -> float | None:
        """Append an observed digit and update from the recent window."""

        digit = self._validate_digit(digit)
        sealed_prefix = list(self.history)
        self.history.append(digit)
        if len(self.history) > self.context_length:
            del self.history[: len(self.history) - self.context_length]
        self.observed_steps += 1

        started = time.perf_counter()
        if self.training_context_alignment == "sealed_prefix":
            loss = self._train_on_sealed_prefix(sealed_prefix, digit)
        else:
            loss = self.train_on_history(self.history) if len(self.history) > 1 else None
        self.last_training_ms = (time.perf_counter() - started) * 1000.0
        if loss is None:
            self.last_gradient_norm = None
            self.last_applied_gradient_norm = None
            self.last_gradient_was_clipped = None
        return loss

    def train_on_history(self, history: Iterable[int]) -> float | None:
        """Train with the selected loss scope over the bounded recent window."""

        context = [self._validate_digit(value) for value in history][
            -self.context_length :
        ]
        if len(context) < 2:
            self.last_training_loss = None
            self.last_training_pairs = 0
            self.last_training_context = list(context)
            return None

        return self._train_on_context(context)

    def _train_on_sealed_prefix(self, prefix: Iterable[int], target: int) -> float | None:
        """Train on the exact prefix used to score the newly observed target."""

        input_prefix = [self._validate_digit(value) for value in prefix][
            -self.context_length :
        ]
        target = self._validate_digit(target)
        if not input_prefix:
            self.last_training_loss = None
            self.last_training_pairs = 0
            self.last_training_context = []
            return None
        return self._train_on_context([*input_prefix, target])

    def _train_on_context(self, context: list[int]) -> float:
        self.last_training_context = list(context)
        self.last_training_pairs = len(context) - 1 if self.loss_scope == "all" else 1
        final_loss = 0.0

        for _ in range(self.updates_per_press):
            state = self.initial_state()
            if self.loss_scope == "all":
                losses: list[Tensor] = []
                for source_digit, target_digit in zip(context[:-1], context[1:]):
                    logits, state = self.step(source_digit, state)
                    target = torch.tensor(target_digit, dtype=torch.long)
                    losses.append(
                        F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                    )
                loss = torch.stack(losses).mean()
            else:
                # Replay every source in the same latest-32 context, then
                # score only its final transition.
                logits = torch.zeros(INPUT_COUNT, dtype=self.input_weights.dtype)
                for source_digit in context[:-1]:
                    logits, state = self.step(source_digit, state)
                target = torch.tensor(context[-1], dtype=torch.long)
                loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite training loss")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad.detach().reshape(-1).to(dtype=torch.float64)
                for parameter in self.parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("training loss produced no gradients")
            norm_value = float(torch.linalg.vector_norm(torch.cat(gradients)).item())
            if not math.isfinite(norm_value):
                raise FloatingPointError("non-finite gradient norm")
            clip_scale = (
                min(1.0, GRADIENT_CLIP_MAX_NORM / norm_value)
                if norm_value > 0
                else 1.0
            )
            clipped = norm_value > GRADIENT_CLIP_MAX_NORM
            self.last_gradient_norm = norm_value
            self.last_gradient_was_clipped = clipped
            if clipped:
                self.clipped_updates += 1
            for parameter in self.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(clip_scale)
            self.last_applied_gradient_norm = norm_value * clip_scale

            self.optimizer.step()
            self.optimizer_updates += 1
            self._proximal_sparsify_input_weights()
            final_loss = float(loss.detach().item())

        self.last_training_loss = final_loss
        return final_loss

    def _proximal_sparsify_input_weights(self) -> None:
        threshold = float(self.optimizer.param_groups[0]["lr"]) * self.sparsity_strength
        if threshold <= 0:
            return
        with torch.no_grad():
            weights = self.input_weights
            weights.copy_(weights.sign() * torch.clamp(weights.abs() - threshold, min=0.0))

    def _state_for_history(self):
        if self.history:
            _, state = self.replay(self.history)
            return state
        return self.initial_state()

    def predict_next(
        self, history: Iterable[int] | None = None
    ) -> tuple[list[float], list[float], list[list[float]]]:
        """Return a pure prediction from the internal replay window.

        The optional argument is accepted to share the OnlinePlasticRNN API;
        this model's history is managed by ``observe`` and the argument is
        intentionally ignored.
        """

        del history
        with torch.no_grad():
            if self.history:
                logits, state = self.replay(self.history)
            else:
                state = self.initial_state()
                logits = torch.zeros(INPUT_COUNT, dtype=self.input_weights.dtype)
            if self.optimizer_updates == 0 or len(self.history) < 2:
                probabilities = [1.0 / INPUT_COUNT] * INPUT_COUNT
            else:
                probabilities = torch.softmax(logits, dim=0).cpu().tolist()
            effective_edges = self.recurrent_weights + self.eta * state.fast_edges
        return (
            [float(value) for value in probabilities],
            [float(value) for value in state.activity.detach().cpu().tolist()],
            [
                [float(value) for value in row]
                for row in effective_edges.detach().cpu().tolist()
            ],
        )

    def parameter_vector(self) -> Tensor:
        """Return the detached flat 623-value vector in OnlinePlasticRNN order."""

        return torch.cat(
            [parameter.detach().reshape(-1) for parameter in self.parameters()]
        ).clone()

    def state_vector(self) -> Tensor:
        """Return the replayed activity and fast-edge trace (420 values)."""

        with torch.no_grad():
            state = self._state_for_history()
            return torch.cat(
                (state.activity.reshape(-1), state.fast_edges.reshape(-1))
            ).detach().clone()

    def config(self) -> dict[str, Any]:
        return {
            "mode": "window",
            "input_count": INPUT_COUNT,
            "neuron_count": NEURON_COUNT,
            "context_length": self.context_length,
            "updates_per_press": self.updates_per_press,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "optimizer": "Adam",
            "weight_decay": self.weight_decay,
            "gradient_clip_max_norm": GRADIENT_CLIP_MAX_NORM,
            "sparsity_strength": self.sparsity_strength,
            "sparsity_method": "post_adam_elementwise_soft_threshold",
            "proximal_threshold": float(
                self.optimizer.param_groups[0]["lr"] * self.sparsity_strength
            ),
            "loss_scope": self.loss_scope,
            "training_context_alignment": self.training_context_alignment,
            "clip_rule": "production_float64_global_norm_no_epsilon",
            "seed": self.seed,
        }

    def diagnostics(self) -> dict[str, Any]:
        with torch.no_grad():
            state = self._state_for_history()
            parameters = self.parameter_vector()
            weights = self.input_weights.detach()
            nonzero_by_digit = (weights != 0).sum(dim=0).cpu().tolist()
            finite_tensors = [parameters, state.activity, state.fast_edges]
            finite = all(bool(torch.isfinite(value).all().item()) for value in finite_tensors)
            updates = int(self.optimizer_updates)
            norms = {
                "activity": float(torch.linalg.vector_norm(state.activity).item()),
                "fast_edges": float(torch.linalg.vector_norm(state.fast_edges).item()),
                "parameters": float(torch.linalg.vector_norm(parameters).item()),
                "uoro_a": [],
                "uoro_b": [],
            }
            nonzero_total = int((weights != 0).sum().item())
        return {
            "mode": "window",
            "loss_scope": self.loss_scope,
            "observed_steps": int(self.observed_steps),
            "history_length": len(self.history),
            "optimizer_updates": updates,
            "input_nonzero_total": nonzero_total,
            "input_nonzero_by_digit": [int(value) for value in nonzero_by_digit],
            "sparsity_strength": self.sparsity_strength,
            "finite": finite,
            "norms": norms,
            "last_training_loss": self.last_training_loss,
            "last_training_pairs": int(self.last_training_pairs),
            "last_gradient_norm": self.last_gradient_norm,
            "last_applied_gradient_norm": self.last_applied_gradient_norm,
            "last_gradient_was_clipped": self.last_gradient_was_clipped,
            "clipped_updates": int(self.clipped_updates),
            "clip_fraction": (
                float(self.clipped_updates / updates) if updates else 0.0
            ),
            "last_training_ms": float(self.last_training_ms),
            "history": list(self.history),
            "training_context_alignment": self.training_context_alignment,
            "last_training_context": list(self.last_training_context),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return parameters, optimizer and online-window runtime state."""

        return {
            "format_version": WINDOW_CHECKPOINT_FORMAT_VERSION,
            "config": self.config(),
            "model": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "runtime": {
                "history": list(self.history),
                "observed_steps": int(self.observed_steps),
                "optimizer_updates": int(self.optimizer_updates),
                "last_training_loss": self.last_training_loss,
                "last_training_pairs": int(self.last_training_pairs),
                "last_gradient_norm": self.last_gradient_norm,
                "last_applied_gradient_norm": self.last_applied_gradient_norm,
                "last_gradient_was_clipped": self.last_gradient_was_clipped,
                "clipped_updates": int(self.clipped_updates),
                "last_training_ms": float(self.last_training_ms),
                "last_training_context": list(self.last_training_context),
            },
        }

    def load_checkpoint_state(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore a checkpoint returned by :meth:`checkpoint_state`."""

        if int(checkpoint.get("format_version", -1)) != WINDOW_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported window-model checkpoint format")
        config = checkpoint.get("config")
        model_state = checkpoint.get("model")
        optimizer_state = checkpoint.get("optimizer")
        runtime = checkpoint.get("runtime")
        if not all(
            isinstance(value, Mapping)
            for value in (config, model_state, optimizer_state, runtime)
        ):
            raise ValueError("checkpoint is missing config, model, optimizer, or runtime")
        if config != self.config():
            raise ValueError("checkpoint config does not match this WindowControlRNN")
        self.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        history = runtime.get("history")
        if not isinstance(history, list) or len(history) > self.context_length:
            raise ValueError("checkpoint history is invalid")
        self.history = [self._validate_digit(value) for value in history]
        self.observed_steps = int(runtime.get("observed_steps", 0))
        self.optimizer_updates = int(runtime.get("optimizer_updates", 0))
        self.last_training_loss = runtime.get("last_training_loss")
        self.last_training_pairs = int(runtime.get("last_training_pairs", 0))
        self.last_gradient_norm = runtime.get("last_gradient_norm")
        self.last_applied_gradient_norm = runtime.get("last_applied_gradient_norm")
        self.last_gradient_was_clipped = runtime.get("last_gradient_was_clipped")
        self.clipped_updates = int(runtime.get("clipped_updates", 0))
        self.last_training_ms = float(runtime.get("last_training_ms", 0.0))
        training_context = runtime.get("last_training_context", [])
        if not isinstance(training_context, list) or len(training_context) > self.context_length + 1:
            raise ValueError("checkpoint last_training_context is invalid")
        self.last_training_context = [
            self._validate_digit(value) for value in training_context
        ]


__all__ = ["WindowControlRNN"]
