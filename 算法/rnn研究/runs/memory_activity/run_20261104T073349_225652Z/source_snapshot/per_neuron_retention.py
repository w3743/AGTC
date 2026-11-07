"""Experimental PlasticRNN variant with destination-specific trace retention.

This module is intentionally separate from the production model. Each row of
``decay_logit`` controls the short-term trace retention of one receiving
neuron; the 20x20 fast edge traces remain independent as in the base model.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
from model import NEURON_COUNT, NetworkState, PlasticRNN


class PerNeuronRetentionRNN(PlasticRNN):
    """PlasticRNN with one learnable decay logit per receiving neuron."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old_scalar = self.decay_logit.detach().clone()
        self.decay_logit = torch.nn.Parameter(old_scalar.expand(NEURON_COUNT, 1).clone())
        # Match the baseline groups: ordinary parameters retain 1e-4 WD;
        # all retention logits are excluded from WD.
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer = torch.optim.Adam(
            [
                {"params": [p for name, p in self.named_parameters() if name != "decay_logit"]},
                {"params": [self.decay_logit], "weight_decay": 0.0},
            ],
            lr=learning_rate,
            weight_decay=1e-4,
        )

    @property
    def trace_decay(self) -> Tensor:
        """Per-destination retention coefficients, shape (20, 1)."""
        return torch.sigmoid(self.decay_logit)


def validate_mechanism() -> dict[str, object]:
    """Run cheap deterministic invariants suitable for an experiment precheck."""
    torch.manual_seed(0)
    shared = PlasticRNN(seed=73)
    independent = PerNeuronRetentionRNN(seed=73)
    # Other parameters and initial dynamics must be exactly those of baseline.
    for name, value in shared.state_dict().items():
        if name != "decay_logit":
            assert torch.equal(value, independent.state_dict()[name]), name
    assert torch.equal(shared.decay_logit.detach().expand_as(independent.decay_logit), independent.decay_logit)
    history = [0, 1, 1, 0, 1]
    shared_logits, shared_state = shared.replay(history)
    row_logits, row_state = independent.replay(history)
    assert torch.equal(shared_logits, row_logits)
    assert torch.equal(shared_state.activity, row_state.activity)
    assert torch.equal(shared_state.fast_edges, row_state.fast_edges)

    # Changing one receiver retention changes only its trace row on the next step.
    a, b = PerNeuronRetentionRNN(seed=73), PerNeuronRetentionRNN(seed=73)
    _, state = a.replay(history)
    _, state_b = b.replay(history)
    with torch.no_grad():
        b.decay_logit[7, 0] += 0.7
    _, next_a = a.step(0, state)
    _, next_b = b.step(0, state_b)
    changed_rows = (next_a.fast_edges != next_b.fast_edges).any(dim=1)
    assert changed_rows.nonzero().flatten().tolist() == [7]

    # At eta=0, prediction loss is independent of fast-edge decay; at eta!=0,
    # all receiving rows get a finite, nonzero task gradient on a varied stream.
    zero_eta = PerNeuronRetentionRNN(seed=11)
    with torch.no_grad():
        zero_eta.eta_raw.zero_()
    zlogits, _ = zero_eta.replay([0, 1, 0, 1, 1, 0])
    torch.nn.functional.cross_entropy(zlogits.unsqueeze(0), torch.tensor([1])).backward()
    grad_zero = zero_eta.decay_logit.grad
    assert grad_zero is not None and torch.equal(grad_zero, torch.zeros_like(grad_zero))
    active = PerNeuronRetentionRNN(seed=11)
    with torch.no_grad():
        active.eta_raw.fill_(0.8)
    alogits, _ = active.replay([0, 1, 0, 1, 1, 0])
    torch.nn.functional.cross_entropy(alogits.unsqueeze(0), torch.tensor([1])).backward()
    grad_rows = active.decay_logit.grad
    assert grad_rows is not None and torch.isfinite(grad_rows).all()
    assert torch.all(grad_rows.abs().sum(dim=1) > 0)
    scalar = PlasticRNN(seed=11)
    with torch.no_grad():
        scalar.eta_raw.fill_(0.8)
    slogits, _ = scalar.replay([0, 1, 0, 1, 1, 0])
    torch.nn.functional.cross_entropy(slogits.unsqueeze(0), torch.tensor([1])).backward()
    assert scalar.decay_logit.grad is not None
    assert torch.allclose(grad_rows.sum(), scalar.decay_logit.grad, atol=1e-6, rtol=0)

    # Fresh Adam steps with absent task gradients do not move retention logits.
    drift = PerNeuronRetentionRNN(seed=12)
    initial = drift.decay_logit.detach().clone()
    for _ in range(10):
        for parameter in drift.parameters():
            parameter.grad = torch.zeros_like(parameter)
        drift.optimizer.step()
    assert torch.equal(initial, drift.decay_logit.detach())

    groups = drift.optimizer.param_groups
    params = [p for group in groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {id(p) for p in drift.parameters()}
    assert groups[0]["weight_decay"] == 1e-4 and groups[1]["weight_decay"] == 0.0
    assert groups[1]["params"] == [drift.decay_logit]
    assert sum(p.numel() for p in drift.parameters()) == 642
    return {"passed": True, "parameter_count": 642, "retention_shape": [20, 1],
            "initial_shared_dynamics_bitwise_equal": True,
            "single_logit_changes_only_receiver_row": 7,
            "eta_zero_gradient_max": float(grad_zero.abs().max()),
            "eta_nonzero_gradient_rows": int((grad_rows.abs().sum(dim=1) > 0).sum()),
            "gradient_sum_error": float((grad_rows.sum() - scalar.decay_logit.grad).abs()),
            "zero_task_gradient_steps": 10, "optimizer_groups": 2}


if __name__ == "__main__":
    print(json.dumps(validate_mechanism(), ensure_ascii=False, allow_nan=False))
