"""Official RSL-RL training and policy timing for Isaac Lab environments."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingConfig:
    """Small, configurable defaults for checking the Isaac Lab training path."""

    seed: int = 7
    n_steps: int = 128
    batch_size: int = 128
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    goal_gain: float = 1.0
    use_sde: bool = False
    sde_sample_freq: int = -1
    log_std_init: float = 0.0
    squash_output: bool = False
    target_kl: float | None = 0.03
    lstm_hidden_size: int = 512
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
    """Legacy SB3 wrapper used only by explicitly selected archived benchmarks."""
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
    return path if path.suffix == ".pt" else Path(f"{path}.pt")


def train(
    env,
    output: str | Path,
    timesteps: int = 512,
    device: str | torch.device = "cpu",
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train through Isaac Lab's official RSL-RL ``OnPolicyRunner``.

    The input must be constructed with ``runtime.create_env(..., rsl=True)``;
    this utility preserves the old result shape while returning an RSL ``.pt``.
    """
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or int(timesteps) < 1:
        raise ValueError("timesteps must be a positive integer")
    config = config or TrainingConfig()
    total_start = time.perf_counter()
    unsupported = []
    if config.use_sde:
        unsupported.append("use_sde")
    if int(config.sde_sample_freq) != -1:
        unsupported.append("sde_sample_freq")
    if config.squash_output:
        unsupported.append("squash_output")
    if int(config.n_lstm_layers) != 1:
        unsupported.append("n_lstm_layers")
    if config.policy_architecture() != {"pi": [64], "vf": [64]}:
        unsupported.append("net_arch")
    if config.progress_bar or config.verbose:
        unsupported.append("SB3 progress_bar/verbose")
    if config.tensorboard_log is not None:
        unsupported.append("tensorboard_log (RSL logs are colocated with the checkpoint)")
    if unsupported:
        raise ValueError(f"unsupported legacy SB3 training settings: {unsupported}")

    from isaac_field.learning import train_stage

    output_path = _saved_checkpoint_path(output)
    result = train_stage(
        env, None, int(timesteps), output_path,
        seed=int(config.seed), device=str(device),
        goal_gain=float(config.goal_gain), n_steps=config.n_steps,
        batch_size=int(config.batch_size), n_epochs=int(config.n_epochs),
        learning_rate=float(config.learning_rate), gamma=float(config.gamma),
        gae_lambda=float(config.gae_lambda), log_std_init=float(config.log_std_init),
        target_kl=config.target_kl, lstm_hidden_size=int(config.lstm_hidden_size),
    )
    model = result.model
    saved_path = result.checkpoint
    return TrainingResult(
        model=model,
        checkpoint=saved_path,
        requested_timesteps=int(timesteps),
        actual_timesteps=int(result.trained_timesteps),
        wrapper_wall_seconds=0.0,
        model_setup_wall_seconds=0.0,
        learn_wall_seconds=time.perf_counter() - total_start,
        save_wall_seconds=0.0,
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


def benchmark(
    env, model, steps: int, *, warmup_steps: int = 4, legacy_sb3: bool = False,
) -> dict[str, Any]:
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
    if legacy_sb3 is not True:
        raise ValueError("SB3 policy benchmarking is legacy-only; pass legacy_sb3=True explicitly")
    from isaac_field.policy import validate_policy_spaces

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
