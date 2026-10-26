"""Real LSTM-512 policy for the native WSL stack (aligned with the repository architecture).

SensorEncoderCore(5 observation channels) -> single-layer LSTM(hidden=512) -> actor head
Linear(512->64)+Tanh+Linear(64->4) (action mean) / critic head Linear(512->64)+Tanh+Linear(64->1).
log_std defaults to log(0.05) matching the repository init_std=0.05.

PPO with recurrent state is supported through `evaluate_sequence`, which replays the rollout
with the stored initial hidden state (and the same per-step done resets) to recompute
log-probabilities and values instead of treating each step as an independent sample.
"""
from __future__ import annotations

import math
import sys

import torch
from torch import nn

REPO_ROOT = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaac_field.sensor_encoder import SensorEncoderCore  # noqa: E402


class LSTMActorCritic(nn.Module):
    def __init__(self, observation_space, *, action_dim: int = 4, hidden_size: int = 512,
                 num_layers: int = 1, init_std: float = 0.05, head_hidden: int = 64):
        super().__init__()
        self.encoder = SensorEncoderCore(observation_space)
        self.features_dim = int(self.encoder.features_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.action_dim = int(action_dim)
        self.lstm = nn.LSTM(self.features_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden), nn.Tanh(), nn.Linear(head_hidden, self.action_dim))
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(math.log(init_std))))

    # ------------------------------------------------------------------ helpers
    def init_hidden(self, batch: int, device=None):
        device = device or self.log_std.device
        zeros = torch.zeros(self.num_layers, int(batch), self.hidden_size, device=device)
        return zeros, zeros.clone()

    @staticmethod
    def reset_hidden(hidden, done_mask):
        """Zero the recurrent state of finished envs (done_mask: bool [B])."""
        h, c = hidden
        if done_mask is None or not bool(done_mask.any()):
            return h, c
        mask = done_mask.to(dtype=h.dtype).view(1, -1, 1)
        return h * (1.0 - mask), c * (1.0 - mask)

    def _step(self, obs, hidden):
        features = self.encoder(obs).unsqueeze(1)          # [B, 1, F]
        out, hidden = self.lstm(features, hidden)
        feat = out.squeeze(1)                              # [B, H]
        mean = torch.tanh(self.actor_mean(feat))
        value = self.critic(feat).squeeze(-1)
        return mean, value, hidden

    # ------------------------------------------------------------------ policy api
    def forward(self, obs, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(int(next(iter(obs.values())).shape[0]),
                                      next(iter(obs.values())).device)
        return self._step(obs, hidden)

    def distribution(self, obs, hidden=None):
        mean, value, hidden = self._step(obs, hidden)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value, hidden

    @torch.no_grad()
    def act(self, obs, hidden=None):
        dist, value, hidden = self.distribution(obs, hidden)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, hidden

    def evaluate_sequence(self, obs_steps, actions, hidden0, done_steps=None):
        """Replay a rollout of T steps from the stored initial hidden state.

        obs_steps: list/tuple of T observation dicts (each tensor [B, ...])
        actions:   [T, B, A]
        done_steps: optional [T, B] bool; after step t the corresponding envs reset.
        Returns (logp [T,B], values [T,B], entropy [T,B]).
        """
        hidden = hidden0
        logps, values, entropies = [], [], []
        T = len(obs_steps)
        std = self.log_std.exp()
        for t in range(T):
            mean, value, hidden = self._step(obs_steps[t], hidden)
            dist = torch.distributions.Normal(mean, std.expand_as(mean))
            logps.append(dist.log_prob(actions[t]).sum(-1))
            values.append(value)
            entropies.append(dist.entropy().sum(-1))
            if done_steps is not None:
                hidden = self.reset_hidden(hidden, done_steps[t])
        return torch.stack(logps), torch.stack(values), torch.stack(entropies)

    def policy_meta(self) -> dict:
        return {"policy_type": "lstm", "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "features_dim": self.features_dim,
                "init_std": float(self.log_std.exp().mean().item()),
                "params_total": int(sum(p.numel() for p in self.parameters())),
                "params_lstm": int(sum(p.numel() for p in self.lstm.parameters())),
                "params_encoder": int(sum(p.numel() for p in self.encoder.parameters())),
                "params_heads": int(sum(p.numel() for p in self.actor_mean.parameters())
                                    + sum(p.numel() for p in self.critic.parameters()))}


def ppo_update_lstm(policy, optimizer, obs_steps, actions, old_logp, advantages, returns,
                    hidden0, done_steps, *, epochs: int = 4, clip: float = 0.2,
                    value_coef: float = 0.5, entropy_coef: float = 0.01,
                    max_grad_norm: float = 0.5) -> float:
    """Recurrent PPO update: replay each rollout from its stored initial hidden state.

    obs_steps: list of T observation dicts ([B, ...]); actions/old_logp/advantages/returns
    are [T, B, ...]; done_steps [T, B] bool marks env resets after each step.
    """
    actions = torch.as_tensor(actions)
    old_logp = torch.as_tensor(old_logp)
    advantages = torch.as_tensor(advantages)
    returns = torch.as_tensor(returns)
    loss_value = 0.0
    for _ in range(int(epochs)):
        logp, values, entropy = policy.evaluate_sequence(obs_steps, actions, hidden0, done_steps)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        loss_value = float(loss.item())
    return loss_value
