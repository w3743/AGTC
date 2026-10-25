"""Minimal actor-critic + PPO for the M0 throughput prototype.

The sensor encoder is imported read-only from the repository
(`isaac_field/sensor_encoder.py`, a pure torch nn.Module) so the branch layout and
feature dimension (192 = 32+64+32+32+32) are identical to the real policy. The
policy head is a minimal Gaussian MLP (the real agent is LSTM-512; M0 only needs
a differentiable consumer of the five observation tensors).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaac_field.sensor_encoder import SensorEncoderCore  # noqa: E402


class ActorCritic(nn.Module):
    def __init__(self, observation_space, *, action_dim: int = 4, hidden: int = 64,
                 init_log_std: float = -0.5):
        super().__init__()
        self.encoder = SensorEncoderCore(observation_space)
        self.features_dim = int(self.encoder.features_dim)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.features_dim, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(self.features_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.action_dim = action_dim

    def forward(self, obs):
        features = self.encoder(obs)
        mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features).squeeze(-1)
        return mean, value

    def distribution(self, obs):
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs):
        dist, value = self.distribution(obs)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    def evaluate(self, obs, actions):
        dist, value = self.distribution(obs)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
