"""RSL-RL boundary adapter for the nested Isaac Field policy observations."""
from __future__ import annotations

import gymnasium as gym

from .env import IsaacFieldEnv


def expose_policy_observation_space(env) -> None:
    """Expose the policy Dict as the vector environment's root observation space.

    Isaac Lab's native RSL wrapper reads ``env.unwrapped.obs_buf`` directly, so
    an outer Gym wrapper cannot adapt reset observations. Keep this small helper
    separate so its spaces can be validated without constructing a simulator.
    """
    single_spaces = env.single_observation_space
    if not isinstance(single_spaces, gym.spaces.Dict) or "policy" not in single_spaces.spaces:
        raise TypeError("Isaac Field RSL adapter expects a single_observation_space['policy'] Dict")
    policy_space = single_spaces["policy"]
    if not isinstance(policy_space, gym.spaces.Dict):
        raise TypeError("Isaac Field policy observations must be a Dict of tensor leaves")

    env.single_observation_space = policy_space
    env.observation_space = gym.vector.utils.batch_space(policy_space, env.num_envs)


class IsaacFieldRslEnv(IsaacFieldEnv):
    """Expose the five policy tensors at the root for RSL-RL's rollout storage."""

    def __init__(self, cfg=None, render_mode=None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        expose_policy_observation_space(self)

    def _compute_observations(self):
        # Delegate to DirectRLEnv so observation noise has exactly the upstream
        # semantics for both ordinary and terminal observations.
        observations = super()._compute_observations()
        if not isinstance(observations, dict) or "policy" not in observations:
            raise TypeError("Isaac Field observations must contain the 'policy' group")
        policy = observations["policy"]
        if not isinstance(policy, dict):
            raise TypeError("Isaac Field policy observations must be a Dict of tensor leaves")
        return policy
