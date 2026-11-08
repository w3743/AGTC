"""Small causal sequence models for compute-matched digit prediction studies.

Each forward pass starts from an empty context and returns one set of logits per
input position.  Row ``t`` predicts the digit after the prefix ending at ``t``.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WindowMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.hidden = nn.Linear(32 * 10, width)
        self.hidden2 = nn.Linear(width, 2 * width)
        self.output = nn.Linear(2 * width, 10)
        self.config = {"architecture": "window_mlp", "window": 32,
                       "input_dim": 320, "hidden_dims": [width, 2 * width], "output_dim": 10,
                       "activation": "tanh_each_hidden", "padding": "zero_left"}

    def forward(self, digits: Sequence[int]) -> Tensor:
        x = _one_hot(digits, self.output.weight.device)
        if x.shape[0] == 0:
            return x.new_empty((0, 10))
        # Window at t is digits[max(0,t-31):t+1], left padded to 32.
        padded = F.pad(x, (0, 0, 31, 0))
        windows = padded.unfold(0, 32, 1).permute(0, 2, 1).reshape(-1, 320)
        return self.output(torch.tanh(self.hidden2(torch.tanh(self.hidden(windows)))))


class CausalTCN(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input = nn.Linear(10, width)
        self.dilations = (1, 2, 4, 8)
        channels = (width, width, width, width + 1)
        self.layers = nn.ModuleList(nn.Conv1d(width if i == 0 else channels[i - 1], out_channels,
                                              kernel_size=3, dilation=d)
                                    for i, (d, out_channels) in enumerate(zip(self.dilations, channels)))
        self.output = nn.Linear(width + 1, 10)
        self.config = {"architecture": "causal_tcn", "input_dim": 10,
                       "width": width, "channels": [width, width, width, width + 1],
                       "layers": 4, "kernel_size": 3,
                       "dilations": list(self.dilations), "receptive_field": 31,
                       "activation": "relu", "residual": False, "output_dim": 10,
                       "implementation": "explicit_dilated_windows"}

    def forward(self, digits: Sequence[int]) -> Tensor:
        x = _one_hot(digits, self.output.weight.device)
        if x.shape[0] == 0:
            return x.new_empty((0, 10))
        h = self.input(x).transpose(0, 1).unsqueeze(0)
        for dilation, conv in zip(self.dilations, self.layers):
            batch, channels, length = h.shape
            padded = F.pad(h, (2 * dilation, 0))
            taps = torch.stack(
                [padded[:, :, tap * dilation:tap * dilation + length] for tap in range(3)],
                dim=-1,
            )
            windows = taps.permute(0, 2, 1, 3).reshape(batch, length, channels * 3)
            weight = conv.weight.reshape(conv.out_channels, channels * 3)
            h = F.relu(F.linear(windows, weight, conv.bias).transpose(1, 2))
        return self.output(h.squeeze(0).transpose(0, 1))


class CausalTransformer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.input = nn.Linear(10, width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, 4 * width)
        self.ff2 = nn.Linear(4 * width, width)
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, 10)
        self.register_buffer("position", _sinusoidal_positions(32, width), persistent=True)
        self.config = {"architecture": "causal_transformer", "input_dim": 10,
                       "width": width, "layers": 1, "heads": 1,
                       "head_dim": width, "ffn_dim": 4 * width,
                       "max_positions": 32, "position_encoding": "fixed_sinusoidal",
                       "normalization": "layer_norm", "activation": "relu",
                       "dropout": 0.0, "output_dim": 10}

    def forward(self, digits: Sequence[int]) -> Tensor:
        x = _one_hot(digits, self.output.weight.device)
        length = x.shape[0]
        if length == 0:
            return x.new_empty((0, 10))
        if length > self.position.shape[0]:
            raise ValueError("causal_transformer supports at most 32 positions")
        h = self.input(x).unsqueeze(0) + self.position[:length].unsqueeze(0)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.width)
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                             device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = self.attention_out(torch.matmul(weights, v))
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ff2(F.relu(self.ff1(h))))
        return self.output(h.squeeze(0))


def _one_hot(digits: Sequence[int], device: torch.device) -> Tensor:
    values = torch.as_tensor(list(digits), dtype=torch.long, device=device)
    if values.ndim != 1 or (values.numel() and ((values < 0).any() or (values > 9).any())):
        raise ValueError("digits must be a one-dimensional sequence of integers from 0 to 9")
    return F.one_hot(values, num_classes=10).to(dtype=torch.float32)


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    even_dims = torch.arange(0, width, 2, dtype=torch.float32)
    rates = torch.exp(-math.log(10000.0) * even_dims / width)
    angles = positions * rates.unsqueeze(0)
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if width > 1:
        encoding[:, 1::2] = torch.cos(angles[:, :width // 2])
    return encoding


def make_model(kind: str, width: int, seed: int = 0) -> nn.Module:
    """Build a model with isolated, deterministic initialization."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("width must be a positive integer")
    constructors = {"mlp": WindowMLP, "tcn": CausalTCN, "transformer": CausalTransformer}
    if kind not in constructors:
        raise ValueError(f"unknown model kind {kind!r}; choose from {tuple(constructors)}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = constructors[kind](width)
    model.config["kind"] = kind
    model.config["seed"] = seed
    return model


def validate() -> bool:
    """Run lightweight mechanism checks; this does not run training experiments."""
    global_before = torch.random.get_rng_state().clone()
    kinds = ("mlp", "tcn", "transformer")
    models = {kind: make_model(kind, 8, seed=17) for kind in kinds}
    assert torch.equal(global_before, torch.random.get_rng_state()), "factory changed global RNG"
    prefix = [1, 4, 2, 0, 7]
    for model in models.values():
        model.eval()
        assert model([]).shape == (0, 10)
        a = model(prefix)
        b = model(prefix + [9, 9, 9, 9])
        assert a.shape == (len(prefix), 10) and b.shape[0] == len(prefix) + 4
        assert torch.allclose(a, b[:len(prefix)], atol=1e-6, rtol=0), model.config
        loss = model(prefix).square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.isfinite(p).all() for p in model.parameters())
    tcn = models["tcn"]
    probe = torch.randn(1, tcn.config["width"], 11)
    for dilation, conv in zip(tcn.dilations, tcn.layers):
        batch, channels, length = probe.shape
        padded = F.pad(probe, (2 * dilation, 0))
        taps = torch.stack([padded[:, :, tap * dilation:tap * dilation + length]
                            for tap in range(3)], dim=-1)
        windows = taps.permute(0, 2, 1, 3).reshape(batch, length, channels * 3)
        explicit = F.linear(windows, conv.weight.reshape(conv.out_channels, channels * 3),
                            conv.bias).transpose(1, 2)
        reference = conv(padded)
        assert torch.allclose(explicit, reference, atol=1e-6, rtol=0), dilation
        probe = F.relu(explicit)
    return True


if __name__ == "__main__":
    validate()
    print("representative_sequence mechanism checks passed")
