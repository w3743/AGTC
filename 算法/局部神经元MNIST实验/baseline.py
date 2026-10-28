"""A compact, ordinary fully-connected MNIST baseline."""

from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Two-hidden-layer MLP used as a reference, not a compute-matched control."""

    def __init__(self, hidden_dim: int = 256, second_hidden_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0 or second_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.second_hidden_dim = int(second_hidden_dim)
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden_dim, self.second_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.second_hidden_dim, 10),
        )

    def forward(self, images: torch.Tensor, plastic: bool = False) -> torch.Tensor:
        # ``plastic`` is accepted so the baseline can use the shared runner API.
        del plastic
        return self.layers(images)
