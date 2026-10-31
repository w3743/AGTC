"""PlasticRNN experiment with a trainable linear output readout."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
from model import INPUT_COUNT, NEURON_COUNT, PlasticRNN, NetworkState


class LinearReadoutRNN(PlasticRNN):
    """Plastic recurrent core with a fully trainable 10-by-width readout.

    Width 20 preserves every legacy core initialization bitwise. Other widths
    use the same seeded Xavier/orthogonal initialization recipe at that width.
    Readout starts as the legacy positive scale times the first ten activity
    coordinates, with zero weights on any remaining coordinates.
    """

    def __init__(self, seed: int = 0, width: int = NEURON_COUNT,
                 context_length: int = 32, updates_per_press: int = 3,
                 learning_rate: float = 0.015) -> None:
        if width < INPUT_COUNT:
            raise ValueError("width must be at least 10")
        if context_length < 2 or updates_per_press < 1:
            raise ValueError("invalid training configuration")
        nn.Module.__init__(self)
        self.context_length = int(context_length)
        self.updates_per_press = int(updates_per_press)
        self.input_sparsity = 0.02
        self.width = int(width)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input_weights = nn.Parameter(torch.empty(width, INPUT_COUNT))
            self.recurrent_weights = nn.Parameter(torch.empty(width, width))
            self.bias = nn.Parameter(torch.zeros(width))
            nn.init.xavier_uniform_(self.input_weights, gain=0.35)
            nn.init.orthogonal_(self.recurrent_weights)
            with torch.no_grad():
                self.recurrent_weights.mul_(0.35)
        self.eta_raw = nn.Parameter(torch.tensor(0.0))
        self.decay_logit = nn.Parameter(torch.full((width, 1), math.log(0.8 / 0.2)))
        legacy_scale = F.softplus(torch.tensor(math.log(math.expm1(1.5))))
        initial_weight = torch.zeros(INPUT_COUNT, width)
        initial_weight[:, :INPUT_COUNT] = torch.eye(INPUT_COUNT) * legacy_scale
        self.W_out = nn.Parameter(initial_weight)
        self.b_out = nn.Parameter(torch.zeros(INPUT_COUNT))
        self.optimizer = torch.optim.Adam(
            [{"params": [p for name, p in self.named_parameters() if name != "decay_logit"]},
             {"params": [self.decay_logit], "weight_decay": 0.0}],
            lr=learning_rate, weight_decay=1e-4)
        self.optimizer_updates = 0
        self.last_training_loss = None
        self.last_training_pairs = 0

    @property
    def trace_decay(self) -> Tensor:
        """Per-destination retention coefficients, shaped (width, 1)."""
        return torch.sigmoid(self.decay_logit)

    def initial_state(self) -> NetworkState:
        return NetworkState(torch.zeros(self.width, dtype=self.input_weights.dtype),
                            torch.zeros(self.width, self.width, dtype=self.input_weights.dtype))

    def step(self, digit: int, state: NetworkState) -> tuple[Tensor, NetworkState]:
        digit = self._validate_digit(digit)
        digit_input = F.one_hot(torch.tensor(digit, dtype=torch.long), num_classes=INPUT_COUNT).to(self.input_weights.dtype)
        effective_edges = self.recurrent_weights + self.eta * state.fast_edges
        new_activity = torch.tanh(self.input_weights @ digit_input + effective_edges @ state.activity + self.bias)
        new_fast_edges = (self.trace_decay * state.fast_edges
                          + (1.0 - self.trace_decay) * torch.outer(new_activity, state.activity))
        logits = self.W_out @ new_activity + self.b_out
        return logits, NetworkState(new_activity, new_fast_edges)


def validate_mechanism(previous_path: str | Path) -> dict[str, object]:
    """Validate all custom task/seed initial checkpoints against this model.

    ``previous_path`` is the decimal benchmark run directory containing
    ``custom_{task}_seed{seed}/checkpoint_000.pt``.
    """
    from decimal_tasks import TASKS, SEEDS

    root = Path(previous_path)
    paths = sorted(root.glob("custom_*_seed*/checkpoint_000.pt"))
    expected = len(TASKS) * len(SEEDS)
    if len(paths) != expected:
        raise AssertionError(f"expected {expected} custom checkpoints, found {len(paths)} under {root}")
    checked = 0
    for path in paths:
        run = path.parent.name
        seed = int(run.rsplit("_seed", 1)[1])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        old = payload["model"]
        model = LinearReadoutRNN(seed=seed)
        for name, value in old.items():
            if name == "scale_raw":
                continue
            if name not in model.state_dict() or not torch.equal(value, model.state_dict()[name]):
                raise AssertionError(f"{path}: mismatch in {name}")
        if not torch.equal(model.decay_logit, old["decay_logit"]):
            raise AssertionError(f"{path}: retention logits differ")
        if model.decay_logit.shape != (model.width, 1):
            raise AssertionError("retention shape must be (width, 1)")
        if hasattr(model, "scale_raw"):
            raise AssertionError("legacy scale_raw must be removed")
        # The initialized linear map reproduces legacy logits and core state.
        from per_neuron_retention import PerNeuronRetentionRNN
        legacy = PerNeuronRetentionRNN(seed=seed)
        for name, value in old.items():
            if name != "scale_raw" and not torch.equal(value, legacy.state_dict()[name]):
                raise AssertionError(f"{path}: source checkpoint differs from PerNeuronRetentionRNN at {name}")
        history = [0, 1, 0, 1, 1, 0]
        with torch.no_grad():
            legacy_logits, legacy_state = legacy.replay(history)
            logits, state = model.replay(history)
        if not torch.equal(legacy_logits, logits):
            raise AssertionError(f"{path}: initial logits differ")
        if not torch.equal(legacy_state.activity, state.activity) or not torch.equal(legacy_state.fast_edges, state.fast_edges):
            raise AssertionError(f"{path}: initial state differs")
        groups = model.optimizer.param_groups
        optimized = [p for group in groups for p in group["params"]]
        if len(optimized) != len({id(p) for p in optimized}) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
            raise AssertionError("optimizer parameter coverage mismatch")
        if groups[0]["weight_decay"] != 1e-4 or groups[1]["weight_decay"] != 0.0 or groups[1]["params"] != [model.decay_logit]:
            raise AssertionError("optimizer weight decay grouping mismatch")
        before = model.decay_logit.detach().clone()
        for _ in range(10):
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            model.optimizer.step()
        if not torch.equal(before, model.decay_logit.detach()):
            raise AssertionError("zero-gradient retention logits drifted")
        checked += 1
    return {"passed": True, "checkpoints_checked": checked, "parameter_count": sum(p.numel() for p in LinearReadoutRNN(seed=0).parameters()),
            "retention_shape": [20, 1], "checkpoint_source": str(root), "initial_logits_and_state_equal": True,
            "optimizer_coverage": True, "zero_gradient_retention_drift": 0.0}
