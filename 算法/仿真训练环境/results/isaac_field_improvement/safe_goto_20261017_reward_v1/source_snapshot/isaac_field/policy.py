"""SB3 recurrent policy for the Isaac Lab field environment.

The observation tensor shapes are retained, but ``state[13:17]`` now stores
the previous normalized policy command. Checkpoints trained with the earlier
measured-motion semantics in those slots are warm starts, not contract-equivalent
weights. This module is self contained and does not depend on ``field_sim``.
"""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


OBSERVATION_SHAPES = {
    "lidar": (2, 400),
    "depth": (2, 160, 88),
    "detections": (16, 10),
    "state": (44,),
}
ACTION_SHAPE = (4,)


class SensorEncoder(BaseFeaturesExtractor):
    """Encode LiDAR, depth, detections, and proprioception into 128 features."""

    def __init__(self, observation_space, cache_inference: bool = True, goal_gain: float = 1.0):
        super().__init__(observation_space, features_dim=128)
        goal_gain = float(goal_gain)
        if not math.isfinite(goal_gain) or goal_gain <= 0:
            raise ValueError("goal_gain must be finite and positive")
        self.cache_inference = bool(cache_inference)
        self.goal_gain = goal_gain
        self.laser = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(observation_space["lidar"].shape)), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 3)),
            nn.Flatten(),
            nn.Linear(96, 32),
            nn.ReLU(),
        )
        self.boxes = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.proprioception = nn.Sequential(nn.Linear(observation_space["state"].shape[0], 32), nn.ReLU())
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def train(self, mode: bool = True):
        # Cached rollout embeddings cannot survive a parameter update.
        if mode:
            self._cache.clear()
        return super().train(mode)

    def _branch(self, name: str, value: torch.Tensor, function: nn.Module) -> torch.Tensor:
        caching = (
            self.cache_inference
            and not self.training
            and not torch.is_grad_enabled()
            and value.shape[0] == 1
        )
        if caching:
            cached = self._cache.get(name)
            if cached is not None and cached[0].device == value.device and torch.equal(cached[0], value):
                return cached[1]
        result = function(value)
        if caching:
            self._cache[name] = (value.detach().clone(), result.detach())
        return result

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        def encode_boxes(value: torch.Tensor) -> torch.Tensor:
            features = self.boxes(value) * value[:, :, 8:9]
            return features.amax(dim=1)

        state = obs["state"]
        if self.goal_gain != 1.0:
            # Scale only relative goal and tolerance features without mutating the input.
            state = state.clone()
            state[:, 2:4] = state[:, 2:4] * self.goal_gain
            state[:, 19] = state[:, 19] * self.goal_gain
        return torch.cat(
            (
                self._branch("lidar", obs["lidar"], self.laser),
                self._branch("depth", obs["depth"], self.depth),
                self._branch("detections", obs["detections"], encode_boxes),
                self.proprioception(state),
            ),
            dim=1,
        )


def validate_policy_spaces(observation_space, action_space=None) -> None:
    """Fail early if an env does not expose the migrated sensor contract."""
    spaces = getattr(observation_space, "spaces", None)
    if spaces is None:
        raise ValueError("policy expects a Dict observation space from Sb3VecEnvWrapper")
    missing = set(OBSERVATION_SHAPES) - set(spaces)
    if missing:
        raise ValueError(f"observation space is missing keys: {sorted(missing)}")
    mismatched = {
        key: (tuple(spaces[key].shape), expected)
        for key, expected in OBSERVATION_SHAPES.items()
        if tuple(spaces[key].shape) != expected
    }
    if mismatched:
        details = ", ".join(f"{key}={actual} (expected {expected})" for key, (actual, expected) in mismatched.items())
        raise ValueError(f"observation dimensions do not match the field policy contract: {details}")
    if action_space is not None and tuple(action_space.shape) != ACTION_SHAPE:
        raise ValueError(f"action space shape {tuple(action_space.shape)} does not match {ACTION_SHAPE}")


def make_model(
    env,
    *,
    seed: int = 7,
    n_steps: int = 128,
    verbose: int = 0,
    goal_gain: float = 1.0,
    batch_size: int = 64,
    n_epochs: int = 3,
    learning_rate: float = 3e-4,
    gamma: float = 0.999,
    n_env: int | None = None,
    use_sde: bool = False,
    sde_sample_freq: int = -1,
    log_std_init: float = 0.0,
    squash_output: bool = False,
    target_kl: float | None = None,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    net_arch: dict[str, list[int]] | None = None,
    device: str | torch.device = "cpu",
    tensorboard_log: str | Path | None = None,
):
    """Build the existing sensor encoder with SB3-Contrib RecurrentPPO."""
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(1)
    actual_n_env = int(getattr(env, "num_envs", 1))
    if n_env is not None and int(n_env) != actual_n_env:
        raise ValueError(f"n_env={n_env} does not match environment num_envs={actual_n_env}")
    validate_policy_spaces(env.observation_space, env.action_space)
    if not math.isfinite(float(log_std_init)):
        raise ValueError("log_std_init must be finite")
    if int(sde_sample_freq) == 0 or int(sde_sample_freq) < -1:
        raise ValueError("sde_sample_freq must be -1 or a positive integer")
    if target_kl is not None and (not math.isfinite(float(target_kl)) or float(target_kl) <= 0):
        raise ValueError("target_kl must be finite and positive when provided")
    if int(n_steps) < 2 or int(batch_size) < 1 or int(n_epochs) < 1:
        raise ValueError("n_steps must be >= 2 and batch_size/n_epochs must be positive")
    if int(batch_size) > int(n_steps) * actual_n_env:
        raise ValueError("batch_size cannot exceed n_steps * num_envs")
    if net_arch is None:
        net_arch = {"pi": [64], "vf": [64]}
    return RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        seed=seed,
        device=device,
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
        learning_rate=float(learning_rate),
        gamma=float(gamma),
        verbose=int(verbose),
        use_sde=bool(use_sde),
        sde_sample_freq=int(sde_sample_freq),
        target_kl=target_kl,
        tensorboard_log=None if tensorboard_log is None else str(tensorboard_log),
        policy_kwargs={
            "features_extractor_class": SensorEncoder,
            "features_extractor_kwargs": {"goal_gain": float(goal_gain)},
            "lstm_hidden_size": int(lstm_hidden_size),
            "n_lstm_layers": int(n_lstm_layers),
            "net_arch": net_arch,
            "log_std_init": float(log_std_init),
            "squash_output": bool(squash_output),
            "normalize_images": False,
        },
    )


def _legacy_policy_kwargs(checkpoint: str | Path) -> dict[str, Any]:
    """Read portable SB3 policy settings and replace the old extractor class."""
    with zipfile.ZipFile(checkpoint) as archive:
        try:
            data = json.loads(archive.read("data"))
        except KeyError as exc:
            raise ValueError(f"{checkpoint} is not an SB3 checkpoint (missing data entry)") from exc
    old = data.get("policy_kwargs")
    if not isinstance(old, dict):
        raise ValueError(f"{checkpoint} does not contain policy_kwargs")

    defaults: dict[str, Any] = {
        "features_extractor_kwargs": {"goal_gain": 1.0},
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": {"pi": [64], "vf": [64]},
        "log_std_init": 0.0,
        "squash_output": False,
        "normalize_images": False,
    }
    # SB3's data.json keeps these JSON-compatible values alongside its opaque
    # cloudpickle payload, so the old module is never imported during migration.
    for key in (
        "features_extractor_kwargs",
        "lstm_hidden_size",
        "n_lstm_layers",
        "net_arch",
        "log_std_init",
        "squash_output",
        "normalize_images",
        "share_features_extractor",
        "optimizer_kwargs",
        "shared_lstm",
        "enable_critic_lstm",
    ):
        if key in old:
            defaults[key] = old[key]
    kwargs: dict[str, Any] = {"features_extractor_class": SensorEncoder, **defaults}
    return kwargs


def convert_legacy_checkpoint(
    checkpoint: str | Path,
    output: str | Path,
    env=None,
    *,
    device: str | torch.device = "cpu",
):
    """Map the old ``field_sim`` extractor to this module, save, and reload.

    The source checkpoint remains untouched. ``env`` is optional because SB3
    stores its spaces in the archive; pass the native wrapper when available to
    validate the migrated observation/action contract. ``output`` is the
    destination checkpoint filename; SB3 appends ``.zip`` when it is omitted.
    """
    from sb3_contrib import RecurrentPPO

    source = Path(checkpoint)
    destination = Path(output)
    if source.resolve() == destination.resolve() or source.resolve() == Path(f"{destination}.zip").resolve():
        raise ValueError("converted checkpoint output must be different from the legacy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = RecurrentPPO.load(
        str(source),
        env=env,
        device=device,
        custom_objects={"policy_kwargs": _legacy_policy_kwargs(source)},
    )
    model.save(str(destination))
    saved_path = destination if destination.suffix == ".zip" else Path(f"{destination}.zip")
    return RecurrentPPO.load(str(saved_path), env=env, device=device)


class PolicyRunner:
    """Stateful deterministic policy inference for one or more vector rows."""

    def __init__(self, env, checkpoint: str | Path | None = None, *, device: str | torch.device = "cpu"):
        from sb3_contrib import RecurrentPPO

        torch.set_num_threads(1)
        validate_policy_spaces(env.observation_space, env.action_space)
        try:
            self.model = (
                RecurrentPPO.load(str(checkpoint), env=env, device=device)
                if checkpoint
                else make_model(env, device=device)
            )
        except ValueError as exc:
            if checkpoint and "spaces do not match" in str(exc):
                raise ValueError(
                    "Checkpoint observation dimensions do not match this environment "
                    f"(LiDAR: {env.observation_space['lidar'].shape}, "
                    f"depth: {env.observation_space['depth'].shape}, "
                    f"detections: {env.observation_space['detections'].shape}, "
                    f"state: {env.observation_space['state'].shape})."
                ) from exc
            raise
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.state = None
        self.episode_start = np.ones(self.num_envs, dtype=bool)
        self.label = "checkpoint loaded; task competence is not implied" if checkpoint else "random initialization; not trained"

    def reset(self, done: np.ndarray | None = None) -> None:
        if done is None:
            self.state = None
            self.episode_start = np.ones(self.num_envs, dtype=bool)
        else:
            done = np.asarray(done, dtype=bool).reshape(-1)
            if done.shape != (self.num_envs,):
                raise ValueError(f"done must have shape ({self.num_envs},)")
            if self.state is not None:
                states = [np.asarray(value).copy() for value in self.state]
                for value in states:
                    value[:, done, :] = 0.0
                self.state = tuple(states)
            self.episode_start = done

    def action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        action, self.state = self.model.predict(
            observation,
            state=self.state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start = np.zeros(self.num_envs, dtype=bool)
        return action
