"""Optional SB3 recurrent PPO adapter; all memory features are learned."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SensorEncoder(BaseFeaturesExtractor):
    """Small branches, no explicit map. Cache only during single-env inference."""
    def __init__(self, observation_space, cache_inference=True, goal_gain=1.0):
        super().__init__(observation_space, features_dim=128)
        goal_gain = float(goal_gain)
        if not math.isfinite(goal_gain) or goal_gain <= 0:
            raise ValueError("goal_gain must be finite and positive")
        self.cache_inference = cache_inference
        self.goal_gain = goal_gain
        self.laser = nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(observation_space['lidar'].shape)), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.depth = nn.Sequential(nn.Conv2d(2, 8, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(), nn.Linear(96, 32), nn.ReLU())
        self.boxes = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.proprioception = nn.Sequential(nn.Linear(observation_space['state'].shape[0], 32), nn.ReLU())
        self._cache = {}

    def train(self, mode=True):
        # Rollout embeddings cannot survive a parameter update.
        if mode:
            self._cache.clear()
        return super().train(mode)

    def _branch(self, name, value, function):
        caching = self.cache_inference and not self.training and not torch.is_grad_enabled() and value.shape[0] == 1
        if caching:
            cached = self._cache.get(name)
            if cached is not None and cached[0].device == value.device and torch.equal(cached[0], value):
                return cached[1]
        result = function(value)
        if caching:
            self._cache[name] = (value.detach().clone(), result.detach())
        return result

    def forward(self, obs):
        def encode_boxes(x):
            f = self.boxes(x) * x[:, :, 8:9]
            return f.amax(dim=1)
        state = obs['state']
        if self.goal_gain != 1.0:
            # Keep the observation contract and tensor untouched; only the
            # features sent to the existing proprioception MLP are rescaled.
            state = state.clone()
            state[:, 2:4] = state[:, 2:4] * self.goal_gain
            state[:, 19] = state[:, 19] * self.goal_gain
        return torch.cat((self._branch('lidar', obs['lidar'], self.laser),
                          self._branch('depth', obs['depth'], self.depth),
                          self._branch('detections', obs['detections'], encode_boxes),
                          self.proprioception(state)), dim=1)


def make_model(env, *, seed=7, n_steps=128, verbose=0, goal_gain=1.0,
               batch_size=64, n_epochs=3, learning_rate=3e-4, gamma=0.999,
               n_env=None, use_sde=False, sde_sample_freq=-1,
               log_std_init=0.0, squash_output=False, target_kl=None,
               device='cpu'):
    from sb3_contrib import RecurrentPPO
    torch.set_num_threads(1)
    actual_n_env = int(getattr(env, 'num_envs', 1))
    if n_env is not None and int(n_env) != actual_n_env:
        raise ValueError(f"n_env={n_env} does not match environment num_envs={actual_n_env}")
    if not math.isfinite(float(log_std_init)):
        raise ValueError("log_std_init must be finite")
    if int(sde_sample_freq) == 0 or int(sde_sample_freq) < -1:
        raise ValueError("sde_sample_freq must be -1 or a positive integer")
    if target_kl is not None and (not math.isfinite(float(target_kl)) or float(target_kl) <= 0):
        raise ValueError("target_kl must be finite and positive when provided")
    return RecurrentPPO('MultiInputLstmPolicy', env, seed=seed, device=device,
                        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                        learning_rate=learning_rate, gamma=gamma, verbose=verbose,
                        use_sde=bool(use_sde), sde_sample_freq=int(sde_sample_freq),
                        target_kl=target_kl,
                        policy_kwargs={'features_extractor_class': SensorEncoder,
                                       'features_extractor_kwargs': {'goal_gain': goal_gain},
                                       'lstm_hidden_size': 128, 'n_lstm_layers': 1,
                                       'net_arch': {'pi': [64], 'vf': [64]},
                                       'log_std_init': float(log_std_init),
                                       'squash_output': bool(squash_output),
                                       'normalize_images': False})


class PolicyRunner:
    def __init__(self, env, checkpoint=None, *, device='cpu'):
        from sb3_contrib import RecurrentPPO
        torch.set_num_threads(1)
        try:
            self.model = (RecurrentPPO.load(str(checkpoint), env=env, device=device)
                          if checkpoint else make_model(env, device=device))
        except ValueError as exc:
            if checkpoint and 'spaces do not match' in str(exc):
                raise ValueError(f'Checkpoint observation dimensions do not match this environment '
                                 f'(LiDAR: {env.observation_space["lidar"].shape}, '
                                 f'depth: {env.observation_space["depth"].shape}, '
                                 f'state: {env.observation_space["state"].shape}). '
                                 'Use a checkpoint trained for the current sensor dimensions and depth convention.') from exc
            raise
        self.state = None
        self.episode_start = True
        self.label = 'short-trained checkpoint; task competence not established' if checkpoint else 'random initialization; not trained'

    def reset(self):
        self.state = None
        self.episode_start = True

    def action(self, observation):
        action, self.state = self.model.predict(observation, state=self.state,
                                               episode_start=np.array([self.episode_start]), deterministic=True)
        self.episode_start = False
        return action



