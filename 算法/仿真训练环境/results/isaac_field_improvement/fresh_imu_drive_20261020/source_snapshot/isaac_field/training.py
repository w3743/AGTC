"""SB3 training and performance measurements for batched Isaac Lab envs."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_field.action_contract import action_config
from isaac_field.policy import (
    _validate_model_action_contract,
    make_model,
    validate_policy_spaces,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Small, configurable defaults for checking the Isaac Lab training path."""

    seed: int = 7
    n_steps: int = 32
    batch_size: int = 32
    n_epochs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.995
    goal_gain: float = 1.0
    use_sde: bool = False
    sde_sample_freq: int = -1
    log_std_init: float = 0.0
    squash_output: bool = False
    target_kl: float | None = 0.03
    lstm_hidden_size: int = 128
    n_lstm_layers: int = 1
    net_arch: tuple[tuple[str, tuple[int, ...]], ...] = (("pi", (64,)), ("vf", (64,)))
    verbose: int = 0
    progress_bar: bool = False
    tensorboard_log: str | Path | None = None

    def policy_architecture(self) -> dict[str, list[int]]:
        return {key: list(layers) for key, layers in self.net_arch}


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    checkpoint: Path
    requested_timesteps: int
    actual_timesteps: int
    wrapper_wall_seconds: float
    model_setup_wall_seconds: float
    learn_wall_seconds: float
    save_wall_seconds: float
    total_wall_seconds: float


def _sb3_wrapper(env):
    """Wrap a native batched DirectRLEnv once with Isaac Lab's official adapter."""
    from stable_baselines3.common.vec_env import VecEnv
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    if isinstance(env, Sb3VecEnvWrapper):
        return env
    if isinstance(env, VecEnv):
        raise TypeError(
            "pass a native batched Isaac Lab env or Sb3VecEnvWrapper; "
            "DummyVecEnv/SubprocVecEnv would create separate SimulationContexts"
        )
    return Sb3VecEnvWrapper(env, fast_variant=True)


def _sync_device(device: Any) -> None:
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError):
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _sync_devices(*devices: Any) -> None:
    seen: set[str] = set()
    for device in devices:
        key = str(device)
        if key not in seen:
            seen.add(key)
            _sync_device(device)


def _saved_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else Path(f"{path}.zip")


def train(
    env,
    output: str | Path,
    timesteps: int = 256,
    device: str | torch.device = "cpu",
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train RecurrentPPO on one native, batched DirectRLEnv instance.

    The returned ``learn_wall_seconds`` measures SB3's ``learn`` call, including
    rollout simulation and optimization, and excludes wrapper/model setup and
    checkpoint serialization.
    """
    if int(timesteps) < 1:
        raise ValueError("timesteps must be positive")
    config = config or TrainingConfig()
    total_start = time.perf_counter()

    wrap_start = time.perf_counter()
    vec_env = _sb3_wrapper(env)
    wrapper_wall = time.perf_counter() - wrap_start
    validate_policy_spaces(vec_env.observation_space, vec_env.action_space)
    action_source = env
    live_action_config = None
    while action_source is not None:
        live_action_config = getattr(action_source, "action_config", None)
        if live_action_config is not None:
            break
        action_source = getattr(action_source, "env", None)
    if live_action_config != action_config():
        raise ValueError(
            "training environment action_config is missing or differs from the "
            f"current action contract: {live_action_config!r}"
        )

    n_envs = int(vec_env.num_envs)
    rollout_size = int(config.n_steps) * n_envs
    if config.n_steps < 2 or config.batch_size < 1 or config.batch_size > rollout_size:
        raise ValueError(
            f"invalid rollout settings: n_steps={config.n_steps}, batch_size={config.batch_size}, "
            f"num_envs={n_envs} (require 1 <= batch_size <= n_steps * num_envs and n_steps >= 2)"
        )
    if config.n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(config.gamma) or not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    model_start = time.perf_counter()
    model = make_model(
        vec_env,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        goal_gain=config.goal_gain,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        log_std_init=config.log_std_init,
        squash_output=config.squash_output,
        target_kl=config.target_kl,
        lstm_hidden_size=config.lstm_hidden_size,
        n_lstm_layers=config.n_lstm_layers,
        net_arch=config.policy_architecture(),
        verbose=config.verbose,
        device=device,
        tensorboard_log=config.tensorboard_log,
    )
    _validate_model_action_contract(model)
    # Preserve the simulator and observation contracts in the saved checkpoint.
    # Current checkpoints use single-channel depth, 39 task and 41 feedback fields.
    # Older input dimensions are rejected rather than silently adapted.
    if hasattr(env, "drive_config"):
        model.field_drive_config = dict(env.drive_config)
    if hasattr(env, "observation_config"):
        model.field_observation_config = dict(env.observation_config)
    damage_config = getattr(env, "damage_config", None)
    if isinstance(damage_config, dict):
        model.field_damage_config = dict(damage_config)
    model_setup_wall = time.perf_counter() - model_start

    sim_device = getattr(vec_env, "sim_device", "cpu")
    _sync_devices(model.device, sim_device)
    learn_start = time.perf_counter()
    model.learn(total_timesteps=int(timesteps), progress_bar=config.progress_bar)
    _sync_devices(model.device, sim_device)
    learn_wall = time.perf_counter() - learn_start

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _sync_device(model.device)
    save_start = time.perf_counter()
    model.save(str(output_path))
    save_wall = time.perf_counter() - save_start
    saved_path = _saved_checkpoint_path(output_path)
    return TrainingResult(
        model=model,
        checkpoint=saved_path,
        requested_timesteps=int(timesteps),
        actual_timesteps=int(model.num_timesteps),
        wrapper_wall_seconds=wrapper_wall,
        model_setup_wall_seconds=model_setup_wall,
        learn_wall_seconds=learn_wall,
        save_wall_seconds=save_wall,
        total_wall_seconds=time.perf_counter() - total_start,
    )


def _latency_stats(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _single_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value[:1] for key, value in observation.items()}


def benchmark(env, model, steps: int, *, warmup_steps: int = 4) -> dict[str, Any]:
    """Measure one-row policy latency and native batched control throughput.

    The one-row latency times only ``model.predict`` on the first row. Batched
    throughput times batched ``predict`` plus one official-wrapper environment
    step, reporting both vector steps/second and environment transitions/second.
    All CUDA timing windows synchronize the policy and simulator devices.
    """
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    if int(warmup_steps) < 0:
        raise ValueError("warmup_steps cannot be negative")
    vec_env = _sb3_wrapper(env)
    validate_policy_spaces(vec_env.observation_space, vec_env.action_space)
    num_envs = int(vec_env.num_envs)
    policy_device = getattr(model, "device", "cpu")
    sim_device = getattr(vec_env, "sim_device", "cpu")

    observation = vec_env.reset()
    batch_state = None
    single_state = None
    batch_episode_start = np.ones(num_envs, dtype=bool)
    single_episode_start = np.ones(1, dtype=bool)
    for _ in range(int(warmup_steps)):
        single_obs = _single_observation(observation)
        _, single_state = model.predict(
            single_obs,
            state=single_state,
            episode_start=single_episode_start,
            deterministic=True,
        )
        actions, batch_state = model.predict(
            observation,
            state=batch_state,
            episode_start=batch_episode_start,
            deterministic=True,
        )
        observation, _, dones, _ = vec_env.step(actions)
        batch_episode_start = np.asarray(dones, dtype=bool)
        single_episode_start = batch_episode_start[:1]

    observation = vec_env.reset()
    batch_state = None
    single_state = None
    batch_episode_start = np.ones(num_envs, dtype=bool)
    single_episode_start = np.ones(1, dtype=bool)
    single_samples: list[float] = []
    batch_policy_samples: list[float] = []
    env_step_samples: list[float] = []
    batch_wall_seconds = 0.0

    for _ in range(int(steps)):
        single_obs = _single_observation(observation)
        _sync_device(policy_device)
        start = time.perf_counter()
        _, single_state = model.predict(
            single_obs,
            state=single_state,
            episode_start=single_episode_start,
            deterministic=True,
        )
        _sync_device(policy_device)
        single_samples.append((time.perf_counter() - start) * 1000.0)

        _sync_devices(policy_device, sim_device)
        batch_start = time.perf_counter()
        policy_start = time.perf_counter()
        actions, batch_state = model.predict(
            observation,
            state=batch_state,
            episode_start=batch_episode_start,
            deterministic=True,
        )
        _sync_device(policy_device)
        batch_policy_samples.append((time.perf_counter() - policy_start) * 1000.0)

        env_start = time.perf_counter()
        observation, _, dones, _ = vec_env.step(actions)
        _sync_devices(policy_device, sim_device)
        env_step_samples.append((time.perf_counter() - env_start) * 1000.0)
        batch_wall_seconds += time.perf_counter() - batch_start
        batch_episode_start = np.asarray(dones, dtype=bool)
        single_episode_start = batch_episode_start[:1]

    return {
        "steps": int(steps),
        "num_envs": num_envs,
        "single_env_latency": _latency_stats(single_samples),
        "batched_policy_latency": _latency_stats(batch_policy_samples),
        "batched_env_step_latency": _latency_stats(env_step_samples),
        "batched_control": {
            "wall_seconds": float(batch_wall_seconds),
            "vector_steps_per_second": float(steps / batch_wall_seconds),
            "environment_transitions_per_second": float(steps * num_envs / batch_wall_seconds),
        },
    }
