from __future__ import annotations

"""Explicit single-layer recurrent baselines for next-digit experiments."""

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SequenceRNN(nn.Module):
    """Vanilla RNN, GRU, or LSTM with explicit, auditable cell equations.

    Each input is converted to a ten-dimensional one-hot vector. The returned
    row at time t predicts the digit following ``digits[t]``. Calls always
    begin with zero hidden (and cell) state.
    """

    KINDS = ("rnn", "gru", "lstm")

    def __init__(self, kind: str, hidden_size: int, seed: int = 0) -> None:
        super().__init__()
        kind = kind.lower()
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        self.kind: Literal["rnn", "gru", "lstm"] = kind  # type: ignore[assignment]
        self.hidden_size = hidden_size
        self.input_size = 10
        self.gates = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
        # Use PyTorch cell's standard uniform range for all cell and readout
        # parameters; keep construction from perturbing the caller's RNG.
        bound = 1.0 / math.sqrt(hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.weight_ih = nn.Parameter(torch.empty(self.gates * hidden_size, 10))
            self.weight_hh = nn.Parameter(torch.empty(self.gates * hidden_size, hidden_size))
            self.bias_ih = nn.Parameter(torch.empty(self.gates * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(self.gates * hidden_size))
            self.readout = nn.Linear(hidden_size, 10)
            for parameter in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh):
                nn.init.uniform_(parameter, -bound, bound)
            nn.init.uniform_(self.readout.weight, -bound, bound)
            nn.init.uniform_(self.readout.bias, -bound, bound)
        self.config = {
            "kind": kind,
            "input_size": 10,
            "hidden_size": hidden_size,
            "num_layers": 1,
            "readout_classes": 10,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "equations": {"rnn": "tanh", "gru": "pytorch_r_z_n_reset_after_hidden_projection", "lstm": "i_f_g_o_no_peepholes"}[kind],
        }

    def forward(self, digits: list[int]) -> Tensor:
        if not isinstance(digits, list):
            raise TypeError("digits must be a list of integers")
        if any(type(d) is not int or not 0 <= d < 10 for d in digits):
            raise ValueError("each digit must be an integer from 0 to 9")
        if not digits:
            return self.weight_ih.new_empty((0, 10))
        x = F.one_hot(torch.tensor(digits, device=self.weight_ih.device), num_classes=10).to(self.weight_ih.dtype)
        h = x.new_zeros(self.hidden_size)
        if self.kind == "lstm":
            c = x.new_zeros(self.hidden_size)
        outputs: list[Tensor] = []
        H = self.hidden_size
        for xt in x:
            input_projection = F.linear(xt, self.weight_ih, self.bias_ih)
            hidden_projection = F.linear(h, self.weight_hh, self.bias_hh)
            if self.kind == "rnn":
                h = torch.tanh(input_projection + hidden_projection)
            elif self.kind == "gru":
                ir, iz, inn = input_projection.chunk(3, dim=-1)
                hr, hz, hn = hidden_projection.chunk(3, dim=-1)
                reset = torch.sigmoid(ir + hr)
                update = torch.sigmoid(iz + hz)
                candidate = torch.tanh(inn + reset * hn)
                h = candidate + update * (h - candidate)
            else:
                ii, iff, ig, io = input_projection.chunk(4, dim=-1)
                hi, hf, hg, ho = hidden_projection.chunk(4, dim=-1)
                i = torch.sigmoid(ii + hi)
                f = torch.sigmoid(iff + hf)
                g = torch.tanh(ig + hg)
                o = torch.sigmoid(io + ho)
                c = f * c + i * g
                h = o * torch.tanh(c)
            outputs.append(F.linear(h, self.readout.weight, self.readout.bias))
        return torch.stack(outputs, dim=0)


def validate() -> dict[str, object]:
    """Cheap equivalence, gradient, causality, and RNG checks."""
    results: dict[str, object] = {}
    for kind, Cell in (("rnn", nn.RNNCell), ("gru", nn.GRUCell), ("lstm", nn.LSTMCell)):
        torch.manual_seed(913)
        rng_before = torch.random.get_rng_state().clone()
        model = SequenceRNN(kind, 7, seed=29)
        assert torch.equal(rng_before, torch.random.get_rng_state()), "constructor changed global RNG"
        same = SequenceRNN(kind, 7, seed=29)
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), same.parameters()))
        cell = Cell(10, 7)
        if kind == "rnn":
            keys = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
            with torch.no_grad():
                for key in keys:
                    getattr(cell, key).copy_(getattr(model, key))
            h = torch.zeros(7)
            ref = []
            for d in [2, 1, 9, 2]:
                h = cell(F.one_hot(torch.tensor(d), 10).float(), h)
                ref.append(F.linear(h, model.readout.weight, model.readout.bias))
        elif kind == "gru":
            with torch.no_grad():
                for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(cell, key).copy_(getattr(model, key))
            h = torch.zeros(7)
            ref = []
            for d in [2, 1, 9, 2]:
                h = cell(F.one_hot(torch.tensor(d), 10).float(), h)
                ref.append(F.linear(h, model.readout.weight, model.readout.bias))
        else:
            with torch.no_grad():
                for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(cell, key).copy_(getattr(model, key))
            h, c = torch.zeros(7), torch.zeros(7)
            ref = []
            for d in [2, 1, 9, 2]:
                h, c = cell(F.one_hot(torch.tensor(d), 10).float(), (h, c))
                ref.append(F.linear(h, model.readout.weight, model.readout.bias))
        out = model([2, 1, 9, 2])
        assert torch.allclose(out, torch.stack(ref), atol=1e-6, rtol=1e-6), kind
        loss = out.square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert torch.equal(model([3, 4, 5])[:2], model([3, 4, 8])[:2]), "future digit affected earlier output"
        assert model([]).shape == (0, 10)
        results[kind] = {"passed": True, "parameter_count": model.config["parameter_count"]}
    results["passed"] = True
    return results


if __name__ == "__main__":
    import json
    print(json.dumps(validate(), allow_nan=False))
