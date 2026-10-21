"""Sequential recurrent-policy curriculum training and held-out evaluation.

Training uses Isaac Lab's official SB3 vector-environment wrapper. Each stage
owns one SimulationContext at a time; evaluation always uses the unshaped task
reward (``distance_cost=0``) and records per-seed rows for reproducibility.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from isaac_field.task import Config


DEFAULT_INITIAL_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "isaac_field_migration"
    / "field_policy_16384_latest.zip"
)
DEFAULT_HELDOUT_SEEDS = tuple(range(10_001, 10_011))


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    mode: str
    goal_distance: tuple[float, float] | None
    goal_angle: float


DEFAULT_CURRICULUM = (
    CurriculumStage("near", "goto", (0.45, 1.05), math.radians(15)),
    CurriculumStage("medium", "goto", (1.0, 3.0), math.pi / 3.0),
    CurriculumStage("far", "goto", (3.0, 8.0), math.pi),
    CurriculumStage("mixed", "mixed", None, math.pi),
)


@dataclass(frozen=True)
class StageTrainingResult:
    model: Any
    checkpoint: Path
    requested_timesteps: int
    trained_timesteps: int
    total_timesteps: int
    seed: int
    goal_gain: float
    n_steps: int
    batch_size: int
    n_epochs: int


@dataclass(frozen=True)
class CurriculumResult:
    checkpoint: Path
    summary_path: Path
    summary: dict[str, Any]


def _make_env_factory():
    from isaac_field.env import make_env

    return make_env


def _make_adapter_factory():
    from isaac_field.env import SingleFieldAdapter

    return SingleFieldAdapter


def _saved_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else Path(f"{path}.zip")


def _existing_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    zipped = Path(f"{path}.zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(path)


def _validate_seeds(seeds: Sequence[int], *, name: str) -> tuple[int, ...]:
    values = tuple(seeds)
    if not values:
        raise ValueError(f"{name} must contain at least one seed")
    if any(isinstance(seed, bool) or int(seed) != seed or int(seed) < 0 for seed in values):
        raise ValueError(f"{name} must contain non-negative integer seeds")
    normalized = tuple(int(seed) for seed in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return normalized


def _validate_goal_sampling(
    goal_distance: tuple[float, float] | None,
    goal_angle: float,
) -> tuple[tuple[float, float] | None, float]:
    if goal_distance is None:
        normalized_distance = None
    else:
        if len(goal_distance) != 2:
            raise ValueError("goal_distance must be a (minimum, maximum) pair")
        lo, hi = (float(value) for value in goal_distance)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo < 0 or hi < lo:
            raise ValueError("goal_distance must be finite, non-negative, and ordered")
        normalized_distance = (lo, hi)
    angle = float(goal_angle)
    if not math.isfinite(angle) or not 0.0 <= angle <= math.pi:
        raise ValueError("goal_angle must be in [0, pi] radians")
    return normalized_distance, angle


def _adapter_engine_cfg(adapter):
    engine = getattr(adapter, "engine", None)
    cfg = getattr(engine, "cfg", None)
    if cfg is None:
        raise TypeError("adapter must be a SingleFieldAdapter with engine.cfg")
    return engine, cfg


def _model_or_checkpoint(model_or_checkpoint, *, device: str):
    if hasattr(model_or_checkpoint, "predict"):
        return model_or_checkpoint, None
    from sb3_contrib import RecurrentPPO

    path = _existing_checkpoint_path(model_or_checkpoint)
    return RecurrentPPO.load(str(path), device=device), path


def _metric_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trajectory_task_snapshot(task) -> dict[str, Any]:
    robot = np.asarray(task.robot, dtype=np.float64).reshape(-1)
    goal = np.asarray(task.goal, dtype=np.float64).reshape(-1)
    if robot.size < 5 or goal.size < 2:
        raise ValueError("compact trajectories require task.robot=[x,y,yaw,v,w] and task.goal=[x,y]")
    delta = goal[:2] - robot[:2]
    cosine, sine = math.cos(float(robot[2])), math.sin(float(robot[2]))
    relative = np.asarray(
        [delta[0] * cosine + delta[1] * sine,
         -delta[0] * sine + delta[1] * cosine],
        dtype=np.float64,
    )
    damage = getattr(task, "damaged", ())
    try:
        damage_count = len(damage)
    except TypeError:
        damage_count = int(damage)
    return {
        "time_s": _metric_float(getattr(task, "time", None)),
        "pose_xy_yaw": robot[:3].tolist(),
        "actual_v_w": robot[3:5].tolist(),
        "goal_robot_frame_m": relative.tolist(),
        "goal_distance_m": float(np.linalg.norm(delta)),
        "goal_bearing_rad": float(math.atan2(relative[1], relative[0])),
        "damage": int(damage_count),
        "collisions": int(getattr(task, "collisions", 0)),
        "hold_s": _metric_float(getattr(task, "_hold", None)),
    }


def _trajectory_vector(engine, name: str) -> list[float] | None:
    value = getattr(engine, name, None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim > 1:
        array = array[0]
    return array.reshape(-1).tolist()


def evaluate_model(
    adapter,
    model_or_checkpoint,
    seeds: Sequence[int],
    *,
    goal_distance: tuple[float, float] | None = None,
    goal_angle: float = math.pi,
    mode: str | None = None,
    options_factory: Callable[[int], Mapping[str, Any] | None] | None = None,
    deterministic: bool = True,
    max_steps: int | None = None,
    device: str = "cuda:0",
    record_trajectories: bool = False,
) -> dict[str, Any]:
    """Evaluate complete episodes on fixed seeds and return JSON-safe rows.

    ``adapter`` is the existing ``SingleFieldAdapter`` around a native
    ``num_envs=1, auto_reset=False`` Isaac Lab environment. Recurrent state is
    reset for every seed. The environment's distance shaping must be disabled,
    so row and aggregate returns are the original task rewards. Optional
    compact trajectories contain task state and actions, never sensor images
    or observations.
    """
    seeds = _validate_seeds(seeds, name="seeds")
    goal_distance, goal_angle = _validate_goal_sampling(goal_distance, goal_angle)
    if max_steps is not None and (
        isinstance(max_steps, bool) or int(max_steps) != max_steps or int(max_steps) < 1
    ):
        raise ValueError("max_steps must be a positive integer when provided")
    if options_factory is not None and not callable(options_factory):
        raise TypeError("options_factory must be callable")
    if not isinstance(record_trajectories, (bool, np.bool_)):
        raise ValueError("record_trajectories must be a boolean")
    record_trajectories = bool(record_trajectories)
    engine, env_cfg = _adapter_engine_cfg(adapter)
    if record_trajectories and getattr(adapter, "task", None) is None:
        raise TypeError("compact trajectory recording requires an adapter exposing its live task")
    distance_cost = float(getattr(env_cfg, "distance_cost", math.nan))
    if not math.isfinite(distance_cost) or not math.isclose(distance_cost, 0.0, abs_tol=1e-12):
        raise ValueError("held-out evaluation requires distance_cost=0.0 (raw task reward)")
    if mode is not None:
        if mode not in ("mixed", "goto", "search"):
            raise ValueError("mode must be mixed, goto, search, or None")
        env_cfg.mode = mode
    env_cfg.goal_distance = goal_distance
    env_cfg.goal_angle = goal_angle

    model, checkpoint_path = _model_or_checkpoint(model_or_checkpoint, device=device)
    if max_steps is None:
        task_cfg = getattr(adapter, "cfg", None)
        dt = float(getattr(task_cfg, "dt", 0.0))
        max_time = float(getattr(task_cfg, "max_time", 0.0))
        max_steps = max(1, int(math.ceil(max_time / dt))) if dt > 0 and max_time > 0 else 100_000

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        options = None if options_factory is None else options_factory(seed)
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("options_factory must return a mapping or None")
        observation, reset_info = adapter.reset(seed=seed, options=None if options is None else dict(options))
        state = None
        episode_start = np.ones(1, dtype=bool)
        raw_return = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = dict(reset_info or {})
        steps = 0
        trajectory: list[dict[str, Any]] = []
        for steps in range(1, int(max_steps) + 1):
            action, state = model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=bool(deterministic),
            )
            action_row = (
                np.asarray(action, dtype=np.float64).reshape(-1)
                if record_trajectories else None
            )
            before = _trajectory_task_snapshot(adapter.task) if record_trajectories else None
            episode_start = np.zeros(1, dtype=bool)
            observation, reward, terminated, truncated, info = adapter.step(action)
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError(f"non-finite reward in held-out episode seed={seed}")
            raw_return += reward
            if record_trajectories:
                after = _trajectory_task_snapshot(adapter.task)
                trajectory.append(
                    {
                        "step": int(steps),
                        "time_s": after["time_s"],
                        "action_4": action_row.tolist(),
                        "command_v_w": _trajectory_vector(engine, "command"),
                        "track_command_lr_mps": _trajectory_vector(engine, "track_command"),
                        "before": before,
                        "after": after,
                        "reward_raw": reward,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "reason": str(info.get("reason", "")),
                    }
                )
            if terminated or truncated:
                break
        step_limit_reached = not (terminated or truncated)
        if step_limit_reached:
            truncated = True
        reason = str(info.get("reason", "time_limit" if truncated else ""))
        timed_out = step_limit_reached or reason == "time_limit"
        success = bool(
            info.get("is_success", reason in ("goal_reached", "all_targets_found_by_evaluator"))
        )
        final_distance = _metric_float(info.get("distance"))
        task = getattr(adapter, "task", None)
        if final_distance is None:
            if task is not None and hasattr(task, "goal") and hasattr(task, "robot"):
                final_distance = _metric_float(np.linalg.norm(task.goal - task.robot[:2]))
        goal = info.get("goal")
        if isinstance(goal, np.ndarray):
            goal = goal.tolist()
        elif goal is None and task is not None and hasattr(task, "goal"):
            goal = np.asarray(task.goal, dtype=np.float64).tolist()
        row = {
            "seed": int(seed),
            "mode": str(info.get("mode", getattr(task, "mode", getattr(env_cfg, "mode", "")))),
            "raw_return": float(raw_return),
            "success": success,
            "damage": int(info.get("damage", 0)),
            "collisions": int(info.get("collisions", 0)),
            "final_distance": final_distance,
            "reason": reason,
            "steps": int(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timed_out": bool(timed_out),
            "goal": goal,
            "reset_info": {
                key: reset_info[key]
                for key in ("mode", "targets", "terrain_seed")
                if reset_info and key in reset_info and isinstance(reset_info[key], (str, int, float, bool, type(None)))
            },
        }
        if record_trajectories:
            row["trajectory"] = trajectory
        rows.append(row)

    success_count = sum(int(row["success"]) for row in rows)

    def mean(key: str) -> float | None:
        values = [_metric_float(row[key]) for row in rows]
        present = [value for value in values if value is not None]
        return None if not present else float(np.mean(present))

    report = {
        "schema": "isaac-field-heldout-v1",
        "reward": "raw_task_reward_distance_cost_zero",
        "distance_cost": 0.0,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "model_class": type(model).__name__,
        "drive_config": getattr(engine, "drive_config", None),
        "checkpoint_drive_config": getattr(model, "field_drive_config", None),
        "observation_config": getattr(engine, "observation_config", None),
        "checkpoint_observation_config": getattr(model, "field_observation_config", None),
        "goal_distance": None if goal_distance is None else list(goal_distance),
        "goal_angle_rad": float(goal_angle),
        "mode": getattr(env_cfg, "mode", None),
        "deterministic": bool(deterministic),
        "seeds": list(seeds),
        "summary": {
            "episodes": len(rows),
            "successes": success_count,
            "success_rate": float(success_count / len(rows)),
            "mean_raw_return": mean("raw_return"),
            "mean_damage": mean("damage"),
            "mean_collisions": mean("collisions"),
            "mean_final_distance": mean("final_distance"),
            "timed_out_episodes": sum(int(row["timed_out"]) for row in rows),
        },
        "rows": rows,
    }
    if record_trajectories:
        report["trajectory_schema"] = "isaac-field-compact-trajectory-v1"
    return report


def train_stage(
    env,
    checkpoint: str | Path | Any | None,
    timesteps: int,
    output: str | Path,
    callback=None,
    *,
    seed: int = 7,
    device: str = "cuda:0",
    goal_gain: float = 100.0,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    log_std_init: float = -2.0,
    target_kl: float | None = 0.03,
    verbose: int = 0,
) -> StageTrainingResult:
    """Continue or initialize RecurrentPPO on the official Isaac Lab wrapper.

    ``log_std_init`` applies only when creating a fresh policy. Loading a model
    keeps its learned distribution parameters while the stage-level optimizer
    and discount settings are explicitly overridden.
    """
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or int(timesteps) < 1:
        raise ValueError("timesteps must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if int(n_steps) != 128:
        # Rollout length is intentionally kept stable across saved PPO stages.
        raise ValueError("curriculum stages require n_steps=128 to match the checkpoint rollout buffer")
    if int(batch_size) < 1 or int(batch_size) > int(n_steps) or int(n_epochs) < 1:
        raise ValueError("require 1 <= batch_size <= n_steps and n_epochs >= 1")
    if not math.isfinite(float(goal_gain)) or float(goal_gain) <= 0:
        raise ValueError("goal_gain must be finite and positive")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if target_kl is not None and (
        not math.isfinite(float(target_kl)) or float(target_kl) <= 0.0
    ):
        raise ValueError("target_kl must be finite and positive when provided")

    from sb3_contrib import RecurrentPPO
    from isaac_field.policy import make_model, validate_policy_spaces
    from isaac_field.training import _sb3_wrapper

    if hasattr(env, "engine"):
        raise TypeError("train_stage requires a native batched Isaac Lab env, not SingleFieldAdapter")
    vec_env = _sb3_wrapper(env)
    validate_policy_spaces(vec_env.observation_space, vec_env.action_space)
    if hasattr(checkpoint, "predict"):
        model = checkpoint
        model.set_env(vec_env)
    elif checkpoint is None:
        model = make_model(
            vec_env,
            seed=int(seed),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            n_epochs=int(n_epochs),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            goal_gain=float(goal_gain),
            log_std_init=float(log_std_init),
            target_kl=target_kl,
            device=device,
            verbose=verbose,
        )
    else:
        checkpoint_path = _existing_checkpoint_path(checkpoint)
        model = RecurrentPPO.load(
            str(checkpoint_path),
            env=vec_env,
            device=device,
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            target_kl=target_kl,
        )

    if int(getattr(model, "n_steps", -1)) != int(n_steps):
        raise ValueError(
            f"checkpoint n_steps={getattr(model, 'n_steps', None)} does not match required n_steps={n_steps}"
        )
    extractor = getattr(model.policy, "features_extractor", None)
    actual_goal_gain = getattr(extractor, "goal_gain", None)
    if actual_goal_gain is None or not math.isclose(float(actual_goal_gain), float(goal_gain), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"checkpoint goal_gain={actual_goal_gain!r} does not match curriculum goal_gain={goal_gain}"
        )

    model.n_epochs = int(n_epochs)
    model.batch_size = int(batch_size)
    model.verbose = int(verbose)
    # RecurrentPPO.load accepts constructor overrides, but an already-created
    # model needs its schedule and rollout buffer updated in place as well.
    # Sync the optimizer's current value too; SB3 will continue applying this
    # schedule at each policy update.
    from stable_baselines3.common.utils import FloatSchedule

    actual_learning_rate = float(learning_rate)
    model.learning_rate = actual_learning_rate
    model.lr_schedule = FloatSchedule(actual_learning_rate)
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = actual_learning_rate
    model.gamma = float(gamma)
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None and hasattr(rollout_buffer, "gamma"):
        rollout_buffer.gamma = float(gamma)
    model.target_kl = target_kl
    # A migrated checkpoint contains the previous simulator's rolling stats.
    # They must not appear as successes measured in this Isaac Lab stage.
    for buffer_name in ('ep_info_buffer','ep_success_buffer'):
        buffer=getattr(model,buffer_name,None)
        if buffer is not None:
            buffer.clear()
    model.set_random_seed(int(seed))
    # Save the action/actuator and observation contracts with the weights.
    # Older checkpoints remain usable as warm starts, but may encode previous
    # commands and measured motion differently in state[13:17].
    if hasattr(env, "drive_config"):
        model.field_drive_config = dict(env.drive_config)
    if hasattr(env, "observation_config"):
        model.field_observation_config = dict(env.observation_config)
    before = int(model.num_timesteps)
    model.learn(
        total_timesteps=int(timesteps),
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    after = int(model.num_timesteps)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    return StageTrainingResult(
        model=model,
        checkpoint=_saved_checkpoint_path(output_path),
        requested_timesteps=int(timesteps),
        trained_timesteps=after - before,
        total_timesteps=after,
        seed=int(seed),
        goal_gain=float(goal_gain),
        n_steps=int(model.n_steps),
        batch_size=int(model.batch_size),
        n_epochs=int(model.n_epochs),
    )


def _default_factory_kwargs(
    cfg: Config,
    *,
    mode: str,
    seed: int,
    goal_distance: tuple[float, float] | None,
    goal_angle: float,
    auto_reset: bool,
) -> dict[str, Any]:
    return {
        "cfg": cfg,
        "mode": mode,
        "num_envs": 1,
        "seed": int(seed),
        "auto_reset": auto_reset,
        "goal_distance": goal_distance,
        "goal_angle": float(goal_angle),
        "distance_cost": 0.0,
    }


def _close_context(adapter_or_env) -> None:
    close = getattr(adapter_or_env, "close", None)
    if callable(close):
        close()
        return
    engine = getattr(adapter_or_env, "engine", None)
    close = getattr(engine, "close", None)
    if callable(close):
        close()


def _open_evaluation_adapter(
    cfg: Config,
    *,
    env_factory: Callable[..., Any],
    adapter_factory: Callable[[Any], Any],
    seed: int,
) -> tuple[Any, Any]:
    engine = env_factory(
        **_default_factory_kwargs(
            cfg,
            mode="mixed",
            seed=seed,
            goal_distance=None,
            goal_angle=math.pi,
            auto_reset=False,
        )
    )
    try:
        adapter = adapter_factory(engine)
    except BaseException:
        _close_context(engine)
        raise
    return adapter, engine


def _summary_result(result: StageTrainingResult, stage: CurriculumStage) -> dict[str, Any]:
    return {
        "stage": asdict(stage),
        "checkpoint": str(result.checkpoint),
        "requested_timesteps": int(result.requested_timesteps),
        "trained_timesteps": int(result.trained_timesteps),
        "total_timesteps": int(result.total_timesteps),
        "seed": int(result.seed),
        "goal_gain": float(result.goal_gain),
        "n_steps": int(result.n_steps),
        "batch_size": int(result.batch_size),
        "n_epochs": int(result.n_epochs),
    }


def train_curriculum(
    cfg: Config,
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = DEFAULT_INITIAL_CHECKPOINT,
    timesteps_per_stage: int | Sequence[int] = 128,
    training_seed: int = 7,
    heldout_seeds: Sequence[int] = DEFAULT_HELDOUT_SEEDS,
    stages: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
    evaluate_before_after: bool = True,
    goal_gain: float = 100.0,
    device: str = "cuda:0",
    n_epochs: int = 5,
    batch_size: int = 128,
    callback_factory: Callable[[CurriculumStage], Any] | None = None,
    options_factory: Callable[[int], Mapping[str, Any] | None] | None = None,
    env_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[[Any], Any] | None = None,
) -> CurriculumResult:
    """Train stages sequentially, closing every physics context before opening another.

    The default stage seeds are consecutive from ``training_seed``. Held-out
    evaluation is run before and after all stages with the same fixed seeds, on
    separate mixed-mode evaluation contexts and with raw reward only.
    """
    stages = tuple(stages)
    if not stages:
        raise ValueError("stages must contain at least one curriculum stage")
    if isinstance(training_seed, bool) or int(training_seed) != training_seed or int(training_seed) < 0:
        raise ValueError("training_seed must be a non-negative integer")
    heldout = _validate_seeds(heldout_seeds, name="heldout_seeds")
    training_seeds = tuple(int(training_seed) + index for index in range(len(stages)))
    overlap = sorted(set(training_seeds).intersection(heldout))
    if overlap:
        raise ValueError(f"held-out seeds overlap training seeds: {overlap}")
    if isinstance(timesteps_per_stage, Sequence) and not isinstance(timesteps_per_stage, (str, bytes)):
        stage_timesteps = tuple(timesteps_per_stage)
        if len(stage_timesteps) != len(stages):
            raise ValueError("timesteps_per_stage sequence must match the number of stages")
    else:
        stage_timesteps = (timesteps_per_stage,) * len(stages)
    if any(isinstance(value, bool) or int(value) != value or int(value) < 1 for value in stage_timesteps):
        raise ValueError("every curriculum timestep count must be a positive integer")
    for stage in stages:
        if stage.mode not in ("mixed", "goto", "search"):
            raise ValueError(f"invalid mode for curriculum stage {stage.name!r}: {stage.mode!r}")
        _validate_goal_sampling(stage.goal_distance, stage.goal_angle)
    if not math.isfinite(float(goal_gain)) or float(goal_gain) <= 0:
        raise ValueError("goal_gain must be finite and positive")

    env_factory = env_factory or _make_env_factory()
    adapter_factory = adapter_factory or _make_adapter_factory()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    active_checkpoint: str | Path | Any | None = checkpoint
    if checkpoint is not None:
        checkpoint = _existing_checkpoint_path(checkpoint)

    evaluation_context_seed = max((*heldout, *training_seeds), default=0) + 1_000_003
    initial_evaluation: dict[str, Any] | None = None
    if evaluate_before_after:
        adapter, engine = _open_evaluation_adapter(
            cfg,
            env_factory=env_factory,
            adapter_factory=adapter_factory,
            seed=evaluation_context_seed,
        )
        try:
            if active_checkpoint is None:
                from isaac_field.policy import make_model
                from isaac_field.training import _sb3_wrapper

                active_checkpoint = make_model(
                    _sb3_wrapper(engine),
                    seed=int(training_seed),
                    n_steps=128,
                    batch_size=int(batch_size),
                    n_epochs=int(n_epochs),
                    goal_gain=float(goal_gain),
                    device=device,
                )
            initial_evaluation = evaluate_model(
                adapter,
                active_checkpoint,
                heldout,
                goal_distance=None,
                goal_angle=math.pi,
                mode="mixed",
                options_factory=options_factory,
                device=device,
            )
        finally:
            _close_context(adapter)

    stage_results: list[dict[str, Any]] = []
    for index, (stage, stage_timesteps) in enumerate(zip(stages, stage_timesteps)):
        stage_seed = training_seeds[index]
        engine = env_factory(
            **_default_factory_kwargs(
                cfg,
                mode=stage.mode,
                seed=stage_seed,
                goal_distance=stage.goal_distance,
                goal_angle=stage.goal_angle,
                auto_reset=True,
            )
        )
        stage_output = output / f"stage_{index + 1:02d}_{stage.name}"
        callback = None if callback_factory is None else callback_factory(stage)
        try:
            result = train_stage(
                engine,
                active_checkpoint,
                int(stage_timesteps),
                stage_output,
                callback=callback,
                seed=stage_seed,
                device=device,
                goal_gain=goal_gain,
                n_epochs=n_epochs,
                batch_size=batch_size,
            )
            stage_results.append(_summary_result(result, stage))
            active_checkpoint = result.checkpoint
        finally:
            _close_context(engine)

    if not isinstance(active_checkpoint, (str, Path)):
        raise RuntimeError("curriculum training did not produce a saved checkpoint")
    final_checkpoint = Path(active_checkpoint)
    final_evaluation: dict[str, Any] | None = None
    if evaluate_before_after:
        adapter, _engine = _open_evaluation_adapter(
            cfg,
            env_factory=env_factory,
            adapter_factory=adapter_factory,
            seed=evaluation_context_seed,
        )
        try:
            final_evaluation = evaluate_model(
                adapter,
                final_checkpoint,
                heldout,
                goal_distance=None,
                goal_angle=math.pi,
                mode="mixed",
                options_factory=options_factory,
                device=device,
            )
        finally:
            _close_context(adapter)

    summary: dict[str, Any] = {
        "schema": "isaac-field-curriculum-v1",
        "algorithm": "SB3-Contrib RecurrentPPO",
        "checkpoint_start": None if checkpoint is None else str(checkpoint),
        "checkpoint_final": str(final_checkpoint),
        "goal_gain": float(goal_gain),
        "stage_seeds": list(training_seeds),
        "heldout_seeds": list(heldout),
        "heldout_seed_policy": "fixed seeds, disjoint from the curriculum training seed range",
        "distance_cost": 0.0,
        "stages": stage_results,
        "heldout_before": initial_evaluation,
        "heldout_after": final_evaluation,
        "claim_limit": "Training completion alone does not establish policy competence or improvement.",
    }
    summary_path = output / "curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return CurriculumResult(checkpoint=final_checkpoint, summary_path=summary_path, summary=summary)

