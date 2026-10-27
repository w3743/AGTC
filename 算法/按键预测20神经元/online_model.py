"""Persistent 20-unit plastic RNN trained online with rank-r UORO.

The model consumes one digit at a time. Its dynamic state is exactly the 20
activities and 400 local fast-edge traces. It does not replay a history window;
one recent transition is cached for the exact readout gradient, while older
influences are tracked by UORO. The pure flat-tensor transition below is used
by both the model and the UORO JVP/VJP calculations, so no state-by-parameter
Jacobian is formed.

The input-weight sparsity option is a practical elementwise proximal step:
after Adam, Win is soft-thresholded by learning_rate * sparsity_strength. This
can set weights to exact zero and later gradients can make them nonzero again.
The Adam-plus-prox procedure as a whole is not claimed to be unbiased or to
have a general convergence guarantee.
"""

from __future__ import annotations

import math
import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


INPUT_COUNT = 10
NEURON_COUNT = 20
STATE_SIZE = NEURON_COUNT + NEURON_COUNT * NEURON_COUNT
PARAMETER_SIZE = (
    NEURON_COUNT * INPUT_COUNT
    + NEURON_COUNT * NEURON_COUNT
    + NEURON_COUNT
    + 3
)
DEFAULT_RANK = 4
DEFAULT_LEARNING_RATE = 0.003
DEFAULT_SPARSITY_STRENGTH = 0.02
GRADIENT_CLIP_MAX_NORM = 1.0
CHECKPOINT_FORMAT_VERSION = 2

_INPUT_WEIGHT_SIZE = NEURON_COUNT * INPUT_COUNT
_RECURRENT_WEIGHT_SIZE = NEURON_COUNT * NEURON_COUNT
_BIAS_SIZE = NEURON_COUNT
_ETA_INDEX = _INPUT_WEIGHT_SIZE + _RECURRENT_WEIGHT_SIZE + _BIAS_SIZE
_DECAY_INDEX = _ETA_INDEX + 1
_SCALE_INDEX = _DECAY_INDEX + 1


def _validate_digit(digit: int) -> int:
    if type(digit) is not int or not 0 <= digit < INPUT_COUNT:
        raise ValueError("digit must be an integer from 0 to 9")
    return digit


def functional_state_step(digit: int, state: Tensor, theta: Tensor) -> Tensor:
    """Pure recurrent transition on ``state[420]`` and ``theta[623]``.

    ``theta`` is concatenated in this order: input weights, recurrent weights,
    bias, eta_raw, decay_logit, scale_raw. The output is the next flattened
    ``(h, F)`` state. This function is suitable for ``torch.func.jvp`` and
    ``torch.func.vjp`` and does not materialize a Jacobian.
    """

    digit = _validate_digit(digit)
    if state.ndim != 1 or state.numel() != STATE_SIZE:
        raise ValueError(f"state must be a flat tensor with {STATE_SIZE} values")
    if theta.ndim != 1 or theta.numel() != PARAMETER_SIZE:
        raise ValueError(f"theta must be a flat tensor with {PARAMETER_SIZE} values")

    input_weights = theta[:_INPUT_WEIGHT_SIZE].view(NEURON_COUNT, INPUT_COUNT)
    recurrent_start = _INPUT_WEIGHT_SIZE
    recurrent_weights = theta[
        recurrent_start : recurrent_start + _RECURRENT_WEIGHT_SIZE
    ].view(NEURON_COUNT, NEURON_COUNT)
    bias_start = recurrent_start + _RECURRENT_WEIGHT_SIZE
    bias = theta[bias_start : bias_start + _BIAS_SIZE]
    eta = 0.5 * torch.tanh(theta[_ETA_INDEX])
    decay = torch.sigmoid(theta[_DECAY_INDEX])

    activity = state[:NEURON_COUNT]
    fast_edges = state[NEURON_COUNT:].view(NEURON_COUNT, NEURON_COUNT)
    one_hot = F.one_hot(
        torch.tensor(digit, dtype=torch.long, device=theta.device),
        num_classes=INPUT_COUNT,
    ).to(dtype=theta.dtype)

    new_activity = torch.tanh(
        input_weights @ one_hot
        + (recurrent_weights + eta * fast_edges) @ activity
        + bias
    )
    new_fast_edges = (
        decay * fast_edges
        + (1.0 - decay) * torch.outer(new_activity, activity)
    )
    return torch.cat((new_activity, new_fast_edges.reshape(-1)))


def functional_logits(state: Tensor, theta: Tensor) -> Tensor:
    """Pure 10-logit readout from ``state[420]`` and ``theta[623]``."""

    if state.ndim != 1 or state.numel() != STATE_SIZE:
        raise ValueError(f"state must be a flat tensor with {STATE_SIZE} values")
    if theta.ndim != 1 or theta.numel() != PARAMETER_SIZE:
        raise ValueError(f"theta must be a flat tensor with {PARAMETER_SIZE} values")
    output_scale = F.softplus(theta[_SCALE_INDEX])
    return output_scale * state[:INPUT_COUNT]


class OnlinePlasticRNN(nn.Module):
    """A persistent 20-neuron RNN with an online rank-r UORO estimator.

    ``observe(digit)`` first scores the new digit and estimates its gradient
    from the old carried ``(h, F, a, b)`` state using the parameters that
    generated that state. It then performs at most one Adam-plus-prox update.
    Finally, it consumes ``digit`` from that same old ``(h, F)`` state and
    advances both the recurrent state and UORO factors using the same
    post-update parameter vector. The first digit has no preceding trainable
    transition and therefore does not trigger an update. In
    ``compute_estimate``, the optimizer step is skipped, so scoring, consuming,
    and factor propagation all use the same frozen parameter vector.

    The optimizer path clips the UORO estimate to max norm 1.0 before Adam;
    clipping can bias the applied estimate. ``compute_estimate`` returns the
    unclipped estimate for fixed-parameter checks.

    ``compute_estimate`` follows the same sequence and advances the state and
    UORO factors, but leaves parameters and optimizer state unchanged. It is a
    QA hook for fixed-parameter Monte Carlo checks. ``random_signs`` can inject
    the rank-by-420 Rademacher vectors used while advancing that step.

    By default, the readout loss is differentiated exactly through the most
    recent input transition (memory-1 UORO); earlier state sensitivity remains
    represented by rank-r factors. Set ``exact_readout_step=False`` to use the
    previous pure-readout estimate for a variance ablation.
    """

    def __init__(
        self,
        seed: int = 0,
        rank: int = DEFAULT_RANK,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        sparsity_strength: float = DEFAULT_SPARSITY_STRENGTH,
        exact_readout_step: bool = True,
    ) -> None:
        super().__init__()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(rank) is not int or rank < 1:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(sparsity_strength)) or sparsity_strength < 0:
            raise ValueError("sparsity_strength must be finite and non-negative")
        if type(exact_readout_step) is not bool:
            raise ValueError("exact_readout_step must be a bool")

        self.seed = seed
        self.rank = rank
        self.sparsity_strength = float(sparsity_strength)
        self.exact_readout_step = exact_readout_step

        # Match the comparison model's deterministic initialization without
        # perturbing the host process's global PyTorch random generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_weights = nn.Parameter(torch.empty(NEURON_COUNT, INPUT_COUNT))
            self.recurrent_weights = nn.Parameter(
                torch.empty(NEURON_COUNT, NEURON_COUNT)
            )
            self.bias = nn.Parameter(torch.zeros(NEURON_COUNT))
            nn.init.xavier_uniform_(self.input_weights, gain=0.35)
            nn.init.orthogonal_(self.recurrent_weights)
            with torch.no_grad():
                self.recurrent_weights.mul_(0.35)

        self.eta_raw = nn.Parameter(torch.tensor(0.0))
        self.decay_logit = nn.Parameter(torch.tensor(math.log(0.8 / 0.2)))
        self.scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.5))))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(learning_rate))
        self.optimizer_updates = 0
        self.last_training_loss: float | None = None
        self.last_training_pairs = 0
        self.last_gradient_norm: float | None = None
        self.last_applied_gradient_norm: float | None = None
        self.last_gradient_was_clipped: bool | None = None
        self.observed_steps = 0

        self.h = torch.zeros(NEURON_COUNT, dtype=self.input_weights.dtype)
        self.fast_edges = torch.zeros(
            NEURON_COUNT, NEURON_COUNT, dtype=self.input_weights.dtype
        )
        self.uoro_a = torch.zeros(
            rank, STATE_SIZE, dtype=self.input_weights.dtype
        )
        self.uoro_b = torch.zeros(
            rank, PARAMETER_SIZE, dtype=self.input_weights.dtype
        )
        self._pending_base_state: Tensor | None = None
        self._pending_uoro_a: Tensor | None = None
        self._pending_uoro_b: Tensor | None = None
        self._pending_digit: int | None = None
        self._pending_theta: Tensor | None = None

        # Keep UORO randomness independent from parameter initialization and
        # from PyTorch's global RNG. A checkpoint saves the complete generator
        # state so a resumed run draws the same subsequent signs.
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed((seed ^ 0x5DEECE66D) & 0x7FFFFFFFFFFFFFFF)

    @property
    def eta(self) -> Tensor:
        """Shared fast-edge learning gain constrained to [-0.5, 0.5]."""

        return 0.5 * torch.tanh(self.eta_raw)

    @property
    def trace_decay(self) -> Tensor:
        """Shared edge-trace decay constrained to [0, 1]."""

        return torch.sigmoid(self.decay_logit)

    @property
    def output_scale(self) -> Tensor:
        """Positive shared scale applied directly to the first ten activities."""

        return F.softplus(self.scale_raw)

    def _parameter_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.input_weights,
            self.recurrent_weights,
            self.bias,
            self.eta_raw,
            self.decay_logit,
            self.scale_raw,
        )

    def _pack_parameters(self) -> Tensor:
        return torch.cat([parameter.reshape(-1) for parameter in self._parameter_tensors()])

    def _state_vector(self) -> Tensor:
        return torch.cat((self.h.reshape(-1), self.fast_edges.reshape(-1)))

    def parameter_vector(self) -> Tensor:
        """Return a detached flat copy of the 623 trainable parameters."""

        return self._pack_parameters().detach().clone()

    def state_vector(self) -> Tensor:
        """Return a detached flat copy of the carried 420-value ``(h, F)``."""

        return self._state_vector().detach().clone()

    def _cache_pending_transition(self, digit: int) -> None:
        """Save the pre-transition state/factors and exact parameter vector."""

        self._pending_base_state = self._state_vector().detach().clone()
        self._pending_uoro_a = self.uoro_a.detach().clone()
        self._pending_uoro_b = self.uoro_b.detach().clone()
        self._pending_digit = _validate_digit(digit)
        self._pending_theta = self._pack_parameters().detach().clone()

    def _draw_rademacher(self) -> Tensor:
        signs = torch.randint(
            0,
            2,
            (self.rank, STATE_SIZE),
            generator=self._rng,
            dtype=torch.int64,
        )
        return (signs.mul_(2).sub_(1)).to(
            device=self.h.device, dtype=self.h.dtype
        )

    def _coerce_random_signs(self, random_signs: Tensor | Sequence[Any] | None) -> Tensor:
        if random_signs is None:
            return self._draw_rademacher()
        signs = torch.as_tensor(random_signs, device=self.h.device, dtype=self.h.dtype)
        if signs.ndim == 1 and self.rank == 1:
            signs = signs.unsqueeze(0)
        if tuple(signs.shape) != (self.rank, STATE_SIZE):
            raise ValueError(
                f"random_signs must have shape ({self.rank}, {STATE_SIZE})"
            )
        if not bool(torch.isfinite(signs).all().item()) or not bool(
            ((signs == 1) | (signs == -1)).all().item()
        ):
            raise ValueError("random_signs values must be finite Rademacher signs (-1 or 1)")
        return signs.detach()

    @staticmethod
    def _norm(value: Tensor) -> Tensor:
        return torch.linalg.vector_norm(value.reshape(-1))

    def _advance_uoro_and_state(
        self,
        digit: int,
        random_signs: Tensor | Sequence[Any] | None = None,
    ) -> None:
        """Consume one input and advance each rank-one state Jacobian estimate."""

        signs = self._coerce_random_signs(random_signs)
        state = self._state_vector().detach()
        theta = self._pack_parameters().detach()
        next_state = functional_state_step(digit, state, theta).detach()

        # The VJP closure computes nu^T B directly for each injected sign
        # vector; the full 420-by-623 state Jacobian is never constructed.
        _, parameter_pullback = torch.func.vjp(
            lambda parameters: functional_state_step(digit, state, parameters),
            theta,
        )
        next_a: list[Tensor] = []
        next_b: list[Tensor] = []

        for index in range(self.rank):
            old_a = self.uoro_a[index].detach()
            old_b = self.uoro_b[index].detach()
            nu = signs[index]
            _, propagated_a = torch.func.jvp(
                lambda old_state: functional_state_step(digit, old_state, theta),
                (state,),
                (old_a,),
            )
            direct_b = parameter_pullback(nu)[0].detach()

            propagated_state_norm = self._norm(propagated_a)
            old_parameter_norm = self._norm(old_b)
            direct_parameter_norm = self._norm(direct_b)
            sign_norm = self._norm(nu)

            propagated_is_zero = bool(
                (propagated_state_norm == 0).item() or (old_parameter_norm == 0).item()
            )
            direct_is_zero = bool((direct_parameter_norm == 0).item())

            if propagated_is_zero and direct_is_zero:
                # Exact zero estimate: avoid 0/0 and leave both factors zero.
                a_new = torch.zeros_like(old_a)
                b_new = torch.zeros_like(old_b)
            elif propagated_is_zero:
                # Only nu outer (nu^T B) remains; it is already rank one.
                a_new = nu.clone()
                b_new = direct_b
            elif direct_is_zero:
                # The direct projected term vanishes for this draw; preserve
                # the old rank-one term without adding unnecessary noise.
                a_new = propagated_a.detach()
                b_new = old_b.clone()
            else:
                # Rank-one trick with variance-balancing normalizers. All four
                # norms are strictly positive in this branch.
                rho0 = torch.sqrt(old_parameter_norm / propagated_state_norm)
                rho1 = torch.sqrt(direct_parameter_norm / sign_norm)
                a_new = rho0 * propagated_a + rho1 * nu
                b_new = old_b / rho0 + direct_b / rho1

            next_a.append(a_new.detach())
            next_b.append(b_new.detach())

        next_h = next_state[:NEURON_COUNT].detach().clone()
        next_fast_edges = next_state[NEURON_COUNT:].view(
            NEURON_COUNT, NEURON_COUNT
        ).detach().clone()
        next_a_tensor = torch.stack(next_a).detach()
        next_b_tensor = torch.stack(next_b).detach()
        if not all(
            bool(torch.isfinite(tensor).all().item())
            for tensor in (next_h, next_fast_edges, next_a_tensor, next_b_tensor)
        ):
            raise FloatingPointError("non-finite recurrent state or UORO factor")
        self.h = next_h
        self.fast_edges = next_fast_edges
        self.uoro_a = next_a_tensor
        self.uoro_b = next_b_tensor

    def _estimate_current_loss_gradient(self, target: int) -> tuple[float, Tensor]:
        """Score the carried prediction and return its flat UORO gradient."""

        target = _validate_digit(target)
        theta = self._pack_parameters().detach().requires_grad_(True)
        if self.exact_readout_step:
            if any(
                value is None
                for value in (
                    self._pending_base_state,
                    self._pending_uoro_a,
                    self._pending_uoro_b,
                    self._pending_digit,
                    self._pending_theta,
                )
            ):
                raise RuntimeError("missing cached transition for exact readout step")
            if not torch.equal(theta.detach(), self._pending_theta):
                raise RuntimeError(
                    "parameters changed since the pending prediction transition"
                )

            base_state = self._pending_base_state.detach().requires_grad_(True)
            prediction_state = functional_state_step(
                self._pending_digit, base_state, theta
            )
            if not torch.equal(prediction_state.detach(), self._state_vector().detach()):
                raise RuntimeError("cached transition does not match the carried state")
            logits = functional_logits(prediction_state, theta)
            sensitivity_a = self._pending_uoro_a.detach()
            sensitivity_b = self._pending_uoro_b.detach()
            differentiated_state = base_state
        else:
            prediction_state = self._state_vector().detach().requires_grad_(True)
            logits = functional_logits(prediction_state, theta)
            sensitivity_a = self.uoro_a.detach()
            sensitivity_b = self.uoro_b.detach()
            differentiated_state = prediction_state

        target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor.unsqueeze(0))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite training loss")

        direct_gradient, state_gradient = torch.autograd.grad(
            loss, (theta, differentiated_state)
        )
        rank_scalars = torch.mv(sensitivity_a, state_gradient.detach())
        carried_gradient = (rank_scalars.unsqueeze(1) * sensitivity_b).mean(dim=0)
        gradient = direct_gradient.detach() + carried_gradient
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("non-finite UORO gradient estimate")
        return float(loss.detach().item()), gradient.detach()

    def _apply_gradient(self, gradient: Tensor) -> None:
        if gradient.ndim != 1 or gradient.numel() != PARAMETER_SIZE:
            raise ValueError(f"gradient must be a flat tensor with {PARAMETER_SIZE} values")
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError("non-finite UORO gradient estimate")

        raw_norm = float(
            torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float64)).item()
        )
        if not math.isfinite(raw_norm):
            raise FloatingPointError("non-finite UORO gradient norm")
        clip_scale = (
            min(1.0, GRADIENT_CLIP_MAX_NORM / raw_norm) if raw_norm > 0 else 1.0
        )
        self.last_gradient_norm = raw_norm
        self.last_gradient_was_clipped = clip_scale < 1.0
        clipped_gradient = gradient * clip_scale
        self.last_applied_gradient_norm = raw_norm * clip_scale
        self.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for parameter in self._parameter_tensors():
            size = parameter.numel()
            parameter_gradient = clipped_gradient[offset : offset + size].view_as(parameter)
            parameter.grad = parameter_gradient.detach().clone()
            offset += size
        self.optimizer.step()
        self.optimizer_updates += 1
        self._proximal_sparsify_input_weights()

    def _proximal_sparsify_input_weights(self) -> None:
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        threshold = learning_rate * self.sparsity_strength
        if threshold <= 0:
            return
        with torch.no_grad():
            weights = self.input_weights
            weights.copy_(weights.sign() * torch.clamp(weights.abs() - threshold, min=0.0))

    def observe(self, digit: int) -> float | None:
        """Score ``digit`` against the prior prediction, update once, then consume it."""

        digit = _validate_digit(digit)
        loss_value: float | None = None
        if self.observed_steps > 0:
            loss_value, gradient = self._estimate_current_loss_gradient(digit)
            self.last_training_loss = loss_value
            self.last_training_pairs = 1
            self._apply_gradient(gradient)
        else:
            self.last_training_loss = None
            self.last_training_pairs = 0
            self.last_gradient_norm = None
            self.last_applied_gradient_norm = None
            self.last_gradient_was_clipped = None

        self._cache_pending_transition(digit)
        self._advance_uoro_and_state(digit)
        self.observed_steps += 1
        return loss_value

    def compute_estimate(
        self,
        digit: int,
        *,
        random_signs: Tensor | Sequence[Any] | None = None,
    ) -> tuple[float | None, Tensor | None]:
        """Advance one sample without changing weights or optimizer state.

        The returned gradient is a detached flat 623-vector, or ``None`` for
        the first input, which has no trainable transition. When supplied,
        ``random_signs`` must be a ``(rank, 420)`` matrix of ±1 values and is
        used only for the UORO factor update caused by consuming this digit.
        """

        digit = _validate_digit(digit)
        if self.observed_steps > 0:
            loss_value, gradient = self._estimate_current_loss_gradient(digit)
            gradient_norm = float(
                torch.linalg.vector_norm(gradient.to(dtype=torch.float64)).item()
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite UORO gradient norm")
            self.last_training_loss = loss_value
            self.last_training_pairs = 1
            self.last_gradient_norm = gradient_norm
            self.last_applied_gradient_norm = None
            self.last_gradient_was_clipped = None
        else:
            loss_value = None
            gradient = None
            self.last_training_loss = None
            self.last_training_pairs = 0
            self.last_gradient_norm = None
            self.last_applied_gradient_norm = None
            self.last_gradient_was_clipped = None

        self._cache_pending_transition(digit)
        self._advance_uoro_and_state(digit, random_signs=random_signs)
        self.observed_steps += 1
        return loss_value, gradient.detach().clone() if gradient is not None else None

    def predict_next(
        self, history: Sequence[int] | None = None
    ) -> tuple[list[float], list[float], list[list[float]]]:
        """Return probabilities, carried activity, and effective 20x20 edges.

        ``history`` is accepted for callers shared with the baseline interface,
        but is never replayed; prediction depends only on the persistent state.
        """

        del history
        with torch.no_grad():
            state = self._state_vector()
            theta = self._pack_parameters()
            logits = functional_logits(state, theta)
            if self.optimizer_updates == 0:
                probabilities = [1.0 / INPUT_COUNT] * INPUT_COUNT
            else:
                probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            effective_edges = self.recurrent_weights + self.eta * self.fast_edges

        return (
            [float(value) for value in probabilities],
            [float(value) for value in self.h.detach().cpu().tolist()],
            [
                [float(value) for value in row]
                for row in effective_edges.detach().cpu().tolist()
            ],
        )

    def config(self) -> dict[str, Any]:
        """Return the fixed architecture and active online-training settings."""

        return {
            "mode": "uoro",
            "input_count": INPUT_COUNT,
            "neuron_count": NEURON_COUNT,
            "state_size": STATE_SIZE,
            "parameter_count": PARAMETER_SIZE,
            "uoro_rank": self.rank,
            "exact_readout_step": self.exact_readout_step,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "optimizer": "Adam",
            "gradient_clip_max_norm": GRADIENT_CLIP_MAX_NORM,
            "sparsity_strength": self.sparsity_strength,
            "sparsity_method": "post_adam_elementwise_soft_threshold",
            "proximal_threshold": float(
                self.optimizer.param_groups[0]["lr"] * self.sparsity_strength
            ),
            "seed": self.seed,
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return current parameters, sparse-input counts, and UORO health data."""

        with torch.no_grad():
            finite_tensors = [
                *self._parameter_tensors(),
                self.h,
                self.fast_edges,
                self.uoro_a,
                self.uoro_b,
                *[
                    value
                    for value in (
                        self._pending_base_state,
                        self._pending_uoro_a,
                        self._pending_uoro_b,
                        self._pending_theta,
                    )
                    if value is not None
                ],
            ]
            finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in finite_tensors)
            input_weights = self.input_weights.detach()
            input_nonzero_by_digit_tensor = (input_weights != 0).sum(dim=0)
            parameter_norm = torch.linalg.vector_norm(self._pack_parameters())
            norms = {
                "activity": float(torch.linalg.vector_norm(self.h).item()),
                "fast_edges": float(torch.linalg.vector_norm(self.fast_edges).item()),
                "uoro_a": [
                    float(torch.linalg.vector_norm(row).item()) for row in self.uoro_a
                ],
                "uoro_b": [
                    float(torch.linalg.vector_norm(row).item()) for row in self.uoro_b
                ],
                "parameters": float(parameter_norm.item()),
                "pending_base_state": (
                    float(torch.linalg.vector_norm(self._pending_base_state).item())
                    if self._pending_base_state is not None
                    else None
                ),
            }
            input_weights_list = input_weights.cpu().tolist()

        factor_bytes = int(
            self.uoro_a.numel() * self.uoro_a.element_size()
            + self.uoro_b.numel() * self.uoro_b.element_size()
        )
        pending_factor_bytes = sum(
            value.numel() * value.element_size()
            for value in (self._pending_uoro_a, self._pending_uoro_b)
            if value is not None
        )
        pending_cache_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                self._pending_base_state,
                self._pending_uoro_a,
                self._pending_uoro_b,
                self._pending_theta,
            )
            if value is not None
        )
        return {
            "mode": "uoro",
            "uoro_rank": self.rank,
            "exact_readout_step": self.exact_readout_step,
            "observed_steps": int(self.observed_steps),
            "optimizer_updates": int(self.optimizer_updates),
            "input_nonzero_total": int((input_weights != 0).sum().item()),
            "input_nonzero_by_digit": [
                int(value) for value in input_nonzero_by_digit_tensor.cpu().tolist()
            ],
            "input_weights": [
                [float(value) for value in row] for row in input_weights_list
            ],
            "sparsity_strength": self.sparsity_strength,
            "trace_decay": float(self.trace_decay.detach().item()),
            "eta": float(self.eta.detach().item()),
            "factor_bytes": factor_bytes + int(pending_factor_bytes),
            "pending_exact_cache_bytes": int(pending_cache_bytes),
            "finite": finite,
            "norms": norms,
            "last_gradient_norm": self.last_gradient_norm,
            "last_applied_gradient_norm": self.last_applied_gradient_norm,
            "last_gradient_was_clipped": self.last_gradient_was_clipped,
            "last_training_loss": self.last_training_loss,
            "last_training_pairs": int(self.last_training_pairs),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return all model, optimizer, recurrent, UORO, and RNG state."""

        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": self.config(),
            "model": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "runtime": {
                "h": self.h.detach().cpu().clone(),
                "fast_edges": self.fast_edges.detach().cpu().clone(),
                "uoro_a": self.uoro_a.detach().cpu().clone(),
                "uoro_b": self.uoro_b.detach().cpu().clone(),
                "observed_steps": int(self.observed_steps),
                "optimizer_updates": int(self.optimizer_updates),
                "last_training_loss": self.last_training_loss,
                "last_training_pairs": int(self.last_training_pairs),
                "last_gradient_norm": self.last_gradient_norm,
                "last_applied_gradient_norm": self.last_applied_gradient_norm,
                "last_gradient_was_clipped": self.last_gradient_was_clipped,
                "pending_transition": (
                    {
                        "base_state": self._pending_base_state.detach().cpu().clone(),
                        "uoro_a": self._pending_uoro_a.detach().cpu().clone(),
                        "uoro_b": self._pending_uoro_b.detach().cpu().clone(),
                        "digit": self._pending_digit,
                        "theta": self._pending_theta.detach().cpu().clone(),
                    }
                    if self._pending_base_state is not None
                    else None
                ),
                "rng_state": self._rng.get_state().clone(),
            },
        }

    def load_checkpoint_state(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore a value returned by :meth:`checkpoint_state` in place."""

        if int(checkpoint.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported online-model checkpoint format")
        config = checkpoint.get("config")
        runtime = checkpoint.get("runtime")
        model_state = checkpoint.get("model")
        optimizer_state = checkpoint.get("optimizer")
        if not all(isinstance(value, Mapping) for value in (config, runtime, model_state)):
            raise ValueError("checkpoint is missing config, runtime, or model state")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("checkpoint is missing optimizer state")

        rank = config.get("uoro_rank")
        if type(rank) is not int or rank < 1:
            raise ValueError("checkpoint has an invalid UORO rank")
        sparsity_strength = float(config.get("sparsity_strength"))
        if not math.isfinite(sparsity_strength) or sparsity_strength < 0:
            raise ValueError("checkpoint has an invalid sparsity strength")
        exact_readout_step = config.get("exact_readout_step")
        if type(exact_readout_step) is not bool:
            raise ValueError("checkpoint has an invalid exact_readout_step setting")
        seed = config.get("seed", self.seed)
        if type(seed) is not int:
            raise ValueError("checkpoint has an invalid seed")

        expected_model_keys = set(self.state_dict())
        if set(model_state) != expected_model_keys:
            raise ValueError("checkpoint parameter keys do not match this model")
        self.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)

        self.seed = seed
        self.rank = rank
        self.sparsity_strength = sparsity_strength
        self.exact_readout_step = exact_readout_step
        device = self.input_weights.device
        dtype = self.input_weights.dtype

        def restore_tensor(value: Any, name: str, expected_shape: tuple[int, ...]) -> Tensor:
            if not isinstance(value, Tensor):
                raise ValueError(f"checkpoint runtime field {name!r} must be a tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"checkpoint runtime field {name!r} has the wrong shape")
            restored = value.to(device=device, dtype=dtype).detach().clone()
            if not bool(torch.isfinite(restored).all().item()):
                raise ValueError(f"checkpoint runtime field {name!r} is non-finite")
            return restored

        def tensor_field(name: str, expected_shape: tuple[int, ...]) -> Tensor:
            return restore_tensor(runtime.get(name), name, expected_shape)

        self.h = tensor_field("h", (NEURON_COUNT,))
        self.fast_edges = tensor_field("fast_edges", (NEURON_COUNT, NEURON_COUNT))
        self.uoro_a = tensor_field("uoro_a", (rank, STATE_SIZE))
        self.uoro_b = tensor_field("uoro_b", (rank, PARAMETER_SIZE))
        self.observed_steps = int(runtime.get("observed_steps", 0))
        self.optimizer_updates = int(runtime.get("optimizer_updates", 0))
        self.last_training_loss = runtime.get("last_training_loss")
        self.last_training_pairs = int(runtime.get("last_training_pairs", 0))
        self.last_gradient_norm = runtime.get("last_gradient_norm")
        self.last_applied_gradient_norm = runtime.get("last_applied_gradient_norm")
        self.last_gradient_was_clipped = runtime.get("last_gradient_was_clipped")

        pending = runtime.get("pending_transition")
        if pending is None:
            self._pending_base_state = None
            self._pending_uoro_a = None
            self._pending_uoro_b = None
            self._pending_digit = None
            self._pending_theta = None
            if self.observed_steps > 0:
                raise ValueError("checkpoint is missing the pending prediction transition")
        elif isinstance(pending, Mapping):
            self._pending_base_state = restore_tensor(
                pending.get("base_state"), "pending_transition.base_state", (STATE_SIZE,)
            )
            self._pending_uoro_a = restore_tensor(
                pending.get("uoro_a"), "pending_transition.uoro_a", (rank, STATE_SIZE)
            )
            self._pending_uoro_b = restore_tensor(
                pending.get("uoro_b"),
                "pending_transition.uoro_b",
                (rank, PARAMETER_SIZE),
            )
            pending_digit = pending.get("digit")
            if type(pending_digit) is not int:
                raise ValueError("checkpoint pending transition has an invalid digit")
            self._pending_digit = _validate_digit(pending_digit)
            self._pending_theta = restore_tensor(
                pending.get("theta"), "pending_transition.theta", (PARAMETER_SIZE,)
            )
            if self.observed_steps < 1:
                raise ValueError("checkpoint has a pending transition with zero observed steps")
        else:
            raise ValueError("checkpoint pending_transition must be a mapping or null")

        if self.exact_readout_step and self.observed_steps > 0:
            current_theta = self._pack_parameters().detach()
            if not torch.equal(current_theta, self._pending_theta):
                raise ValueError("checkpoint pending transition parameters do not match")
            expected_state = functional_state_step(
                self._pending_digit, self._pending_base_state, current_theta
            )
            if not torch.equal(expected_state, self._state_vector()):
                raise ValueError("checkpoint pending transition does not match carried state")

        rng_state = runtime.get("rng_state")
        if not isinstance(rng_state, Tensor):
            raise ValueError("checkpoint runtime field 'rng_state' must be a tensor")
        self._rng.set_state(rng_state.detach().cpu())

    def _apply(self, fn: Any, recurse: bool = True) -> OnlinePlasticRNN:
        # These persistent tensors are not Parameters, so nn.Module's default
        # device/dtype conversion would otherwise leave them behind.
        super()._apply(fn, recurse=recurse)
        for name in (
            "h",
            "fast_edges",
            "uoro_a",
            "uoro_b",
            "_pending_base_state",
            "_pending_uoro_a",
            "_pending_uoro_b",
            "_pending_theta",
        ):
            if hasattr(self, name):
                value = getattr(self, name)
                if value is not None:
                    setattr(self, name, fn(value))
        return self
