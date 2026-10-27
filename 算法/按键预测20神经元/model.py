"""A small, fixed-size plastic recurrent network for next-digit prediction.

The 10 digit inputs are external ports. The network has exactly 20 activity
units and a full 20 by 20 directed recurrent matrix. Recurrent matrix rows are
destination units and columns are source units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


INPUT_COUNT = 10
NEURON_COUNT = 20
CONTEXT_LENGTH = 32
TRAINING_UPDATES_PER_PRESS = 3
LEARNING_RATE = 0.015
INPUT_SPARSITY = 0.02


@dataclass(frozen=True)
class NetworkState:
    """Dynamic state while replaying a context window."""

    activity: Tensor
    fast_edges: Tensor


class PlasticRNN(nn.Module):
    """20-unit recurrent network with a local, stateful fast-weight trace.

    At a time step, ``activity`` is the old hidden activity and ``fast_edges``
    is the old local edge trace. The next activity is calculated first, then
    every directed edge trace is updated from its destination/source activity
    pair. The first ten activities are the ten output logits after a shared
    positive scale; there is no separate readout layer.
    """

    def __init__(
        self,
        seed: int = 0,
        context_length: int = CONTEXT_LENGTH,
        updates_per_press: int = TRAINING_UPDATES_PER_PRESS,
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        super().__init__()
        if context_length < 2:
            raise ValueError("context_length must be at least 2")
        if updates_per_press < 1:
            raise ValueError("updates_per_press must be at least 1")

        self.context_length = int(context_length)
        self.updates_per_press = int(updates_per_press)
        # Fixed production setting; kept on the instance so QA can set it to
        # zero for a bitwise legacy-window comparison.
        self.input_sparsity = INPUT_SPARSITY

        # Keep initialization deterministic for this model without changing
        # the global torch RNG used by a host application.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_weights = nn.Parameter(torch.empty(NEURON_COUNT, INPUT_COUNT))
            self.recurrent_weights = nn.Parameter(torch.empty(NEURON_COUNT, NEURON_COUNT))
            self.bias = nn.Parameter(torch.zeros(NEURON_COUNT))
            nn.init.xavier_uniform_(self.input_weights, gain=0.35)
            nn.init.orthogonal_(self.recurrent_weights)
            with torch.no_grad():
                self.recurrent_weights.mul_(0.35)

        self.eta_raw = nn.Parameter(torch.tensor(0.0))
        self.decay_logit = nn.Parameter(torch.tensor(math.log(0.8 / 0.2)))
        self.scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.5))))
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.optimizer_updates = 0
        self.last_training_loss: float | None = None
        self.last_training_pairs = 0

    @property
    def eta(self) -> Tensor:
        """Shared fast-edge learning gain, constrained to [-0.5, 0.5]."""

        return 0.5 * torch.tanh(self.eta_raw)

    @property
    def trace_decay(self) -> Tensor:
        """Shared edge-trace decay, constrained to [0, 1]."""

        return torch.sigmoid(self.decay_logit)

    @property
    def output_scale(self) -> Tensor:
        """Positive shared scale applied directly to the first ten activities."""

        return F.softplus(self.scale_raw)

    def initial_state(self) -> NetworkState:
        return NetworkState(
            activity=torch.zeros(NEURON_COUNT, dtype=self.input_weights.dtype),
            fast_edges=torch.zeros(
                NEURON_COUNT, NEURON_COUNT, dtype=self.input_weights.dtype
            ),
        )

    @staticmethod
    def _validate_digit(digit: int) -> int:
        if type(digit) is not int or not 0 <= digit < INPUT_COUNT:
            raise ValueError("digit must be an integer from 0 to 9")
        return digit

    def step(self, digit: int, state: NetworkState) -> tuple[Tensor, NetworkState]:
        """Consume one digit and return next-digit logits plus the new state."""

        digit = self._validate_digit(digit)
        digit_input = F.one_hot(
            torch.tensor(digit, dtype=torch.long), num_classes=INPUT_COUNT
        ).to(dtype=self.input_weights.dtype)

        # For matrix indices [destination, source], multiplication by the
        # old activity vector sends every source to every destination.
        effective_edges = self.recurrent_weights + self.eta * state.fast_edges
        new_activity = torch.tanh(
            self.input_weights @ digit_input
            + effective_edges @ state.activity
            + self.bias
        )

        # Local Hebbian trace: the edge from old source j to new destination i
        # is updated from h_new[i] * h_old[j].
        new_fast_edges = (
            self.trace_decay * state.fast_edges
            + (1.0 - self.trace_decay)
            * torch.outer(new_activity, state.activity)
        )
        logits = self.output_scale * new_activity[:INPUT_COUNT]
        return logits, NetworkState(new_activity, new_fast_edges)

    def replay(self, digits: Iterable[int]) -> tuple[Tensor, NetworkState]:
        """Replay digits from a clean state and return the last next-digit logits."""

        state = self.initial_state()
        logits = torch.zeros(INPUT_COUNT, dtype=self.input_weights.dtype)
        for digit in digits:
            logits, state = self.step(digit, state)
        return logits, state

    def train_on_history(self, history: Iterable[int]) -> float | None:
        """Train on adjacent observed pairs from the bounded recent context.

        For each pair ``(d[t], d[t+1])``, the network consumes only the
        observed prefix ending at ``d[t]`` before predicting ``d[t+1]``. No
        current/future value is fed into a prediction before that prediction is
        scored. Gradients are backpropagated through at most ``context_length``
        observed digits.
        """

        context = [self._validate_digit(d) for d in history][-self.context_length :]
        if len(context) < 2:
            self.last_training_loss = None
            self.last_training_pairs = 0
            return None

        targets = [
            torch.tensor(target, dtype=torch.long) for target in context[1:]
        ]
        self.last_training_pairs = len(targets)
        loss_value = 0.0

        for _ in range(self.updates_per_press):
            state = self.initial_state()
            losses: list[Tensor] = []
            for source_digit, target in zip(context[:-1], targets):
                logits, state = self.step(source_digit, state)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))

            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
            self.optimizer.step()
            threshold = float(self.optimizer.param_groups[0]["lr"]) * self.input_sparsity
            if threshold > 0.0:
                with torch.no_grad():
                    weights = self.input_weights
                    weights.copy_(weights.sign() * torch.clamp(weights.abs() - threshold, min=0.0))
            self.optimizer_updates += 1
            loss_value = float(loss.detach().item())

        self.last_training_loss = loss_value
        return loss_value

    def predict_next(self, history: Iterable[int]) -> tuple[list[float], list[float], list[list[float]]]:
        """Return probabilities, neuron activities, and effective 20x20 edges.

        Until one observed transition pair has trained the network, predictions
        remain exactly uniform. This avoids presenting random initialization as
        learned behavior.
        """

        context = [self._validate_digit(d) for d in history][-self.context_length :]
        if context:
            with torch.no_grad():
                logits, state = self.replay(context)
                activity = state.activity.detach().cpu().tolist()
                effective = (
                    self.recurrent_weights + self.eta * state.fast_edges
                ).detach().cpu().tolist()
        else:
            state = self.initial_state()
            activity = state.activity.detach().cpu().tolist()
            effective = self.recurrent_weights.detach().cpu().tolist()
            logits = torch.zeros(INPUT_COUNT, dtype=self.input_weights.dtype)

        if len(context) < 2 or self.optimizer_updates == 0:
            probabilities = [1.0 / INPUT_COUNT] * INPUT_COUNT
        else:
            with torch.no_grad():
                probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()

        return (
            [float(value) for value in probabilities],
            [float(value) for value in activity],
            [[float(value) for value in row] for row in effective],
        )
