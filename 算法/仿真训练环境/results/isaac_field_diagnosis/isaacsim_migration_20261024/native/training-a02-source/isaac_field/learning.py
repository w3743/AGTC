"""Sequential RSL-RL curriculum training and held-out evaluation.

Training delegates rollout collection, PPO, and checkpoint handling to Isaac
Lab's official RSL-RL ``OnPolicyRunner``. Each stage closes its environment
while the workflow keeps one Isaac Sim Kit runtime open; evaluation records
per-seed rows using the unshaped task reward (``distance_cost=0``).
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from isaac_field.feedback import ABSOLUTE_GOAL_INPUT_SCHEMA, INPUT_SCHEMA
from isaac_field.task import Config


DEFAULT_INITIAL_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "isaac_field_improvement"
    / "isaacsim_physx_rsl_foundation_20261024"
    / "stage_04_turn90.pt"
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
    from isaac_field.runtime import create_env

    return create_env


def _make_adapter_factory():
    from isaac_field.env import SingleFieldAdapter

    return SingleFieldAdapter


def _keep_runtime_for_workflow(function):
    """Hold one Isaac Sim Kit context across all stages and paired evaluations."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from .runtime import launch_app

        with launch_app(device=kwargs.get("device", "cuda:0"), headless=True):
            return function(*args, **kwargs)

    return wrapped


def _saved_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".pt" else Path(f"{path}.pt")


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


def _damage_config(source) -> dict[str, Any] | None:
    """Find the live task's damage contract without changing evaluation config."""
    candidates = (getattr(source, "task", None), source, getattr(source, "engine", None))
    for candidate in candidates:
        config = getattr(candidate, "damage_config", None) if candidate is not None else None
        if isinstance(config, Mapping):
            return dict(config)
    return None


def _live_observation_config(source) -> dict[str, Any] | None:
    """Find the native observation contract through adapters and wrappers."""
    pending = [source]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        config = getattr(candidate, "observation_config", None)
        if isinstance(config, Mapping):
            return dict(config)
        for name in ("engine", "env", "unwrapped"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return None


def _live_input_schema(source) -> str | None:
    """Read the schema carried by a task config, if this environment exposes one."""
    pending = [source]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        for name in ("cfg", "field_cfg"):
            config = getattr(candidate, name, None)
            schema = getattr(config, "input_schema", None)
            if isinstance(schema, str):
                return schema
        for name in ("engine", "env", "unwrapped"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return None


def _live_backend_identity(source) -> dict[str, str] | None:
    """Resolve the simulator identity through adapters and Gym wrappers."""
    from .policy_contract import BACKEND_IDENTITY_KEYS

    pending = [source]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        value = getattr(candidate, "backend_identity", None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return {key: value.get(key) for key in BACKEND_IDENTITY_KEYS}
        flattened = {key: getattr(candidate, key, None) for key in BACKEND_IDENTITY_KEYS}
        if all(item is not None for item in flattened.values()):
            return flattened
        for name in ("engine", "env", "unwrapped"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
    return None


def _model_backend_identity(model) -> dict[str, str]:
    """Return checkpoint provenance separately from the active eval backend."""
    from isaac_field.policy_contract import recorded_backend_identity

    contract = getattr(model, "contract", None)
    if isinstance(contract, Mapping):
        return recorded_backend_identity({
            key: contract.get(key)
            for key in ("simulator", "physics_backend", "learning_backend")
        })
    return recorded_backend_identity({})


def _model_backend_name(model) -> str:
    if getattr(model, "backend", None) == "rsl_rl":
        return "rsl_rl"
    module = type(model).__module__
    if module.startswith(("stable_baselines3", "sb3_contrib")):
        return "sb3_legacy"
    return "custom_or_test_model"


def _observation_config_is_complete(config: Mapping[str, Any] | None) -> bool:
    if not isinstance(config, Mapping):
        return False
    fields = config.get("state_fields")
    shapes = config.get("observation_shapes")
    return (
        config.get("state_dim") == 33
        and isinstance(fields, (list, tuple))
        and len(fields) == 33
        and isinstance(shapes, Mapping)
        and shapes.get("state") == [33]
    )


def _policy_contract(
    source,
    *,
    input_schema: str | None = None,
    expected_observation_config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve and cross-check the live schema before model use or checkpoint loading."""
    from isaac_field.feedback import SUPPORTED_INPUT_SCHEMAS

    live_schema = _live_input_schema(source)
    actual_config = _live_observation_config(source)
    actual_schema = actual_config.get("schema") if actual_config is not None else None
    # Older adapters and tests can carry unrelated, pre-versioned display metadata.
    # Only a supported schema is authoritative when no Config exposes one.
    if live_schema is None and actual_schema in SUPPORTED_INPUT_SCHEMAS:
        live_schema = actual_schema
    schema = live_schema if input_schema is None and live_schema is not None else (
        INPUT_SCHEMA if input_schema is None else input_schema
    )
    if schema not in SUPPORTED_INPUT_SCHEMAS:
        raise ValueError(
            f"unsupported input_schema {schema!r}; expected one of {SUPPORTED_INPUT_SCHEMAS!r}"
        )
    if live_schema is not None and live_schema != schema:
        raise ValueError(
            f"live environment input_schema {live_schema!r} does not match requested {schema!r}"
        )
    if actual_config is not None:
        config_schema = actual_config.get("schema")
        if config_schema in SUPPORTED_INPUT_SCHEMAS and config_schema != schema:
            raise ValueError(
                f"live observation_config schema {config_schema!r} does not match requested {schema!r}"
            )
    if expected_observation_config is not None:
        expected = dict(expected_observation_config)
        if actual_config is not None and actual_config != expected:
            raise ValueError("expected_observation_config differs from the live environment contract")
        actual_config = expected
    if schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        if not _observation_config_is_complete(actual_config):
            raise ValueError(
                "absolute-goal training and evaluation require the complete live observation_config"
            )
    elif not _observation_config_is_complete(actual_config):
        # Preserve v6 compatibility with old minimal test adapters and archived
        # checkpoints. Full contracts are still forwarded whenever available.
        actual_config = None
    return schema, actual_config


def _model_or_checkpoint(
    model_or_checkpoint,
    *,
    device: str,
    input_schema: str,
    expected_observation_config: Mapping[str, Any] | None,
    expected_backend_identity: Mapping[str, Any] | None,
    legacy_sb3: bool,
):
    # RSL checkpoints retain their own model/metadata format. Adapt prediction
    # only, so both backends use the same episode and physical-trace evaluator.
    if getattr(model_or_checkpoint, "backend", None) == "rsl_rl":
        model_or_checkpoint.validate_contract(
            input_schema, expected_observation_config, expected_backend_identity,
        )
        return model_or_checkpoint, getattr(model_or_checkpoint, "checkpoint", None)
    if isinstance(model_or_checkpoint, (str, Path)) and Path(model_or_checkpoint).suffix.lower() in (".pt", ".pth"):
        from isaac_field.rsl_rl_inference import RslPredictionModel

        path = _existing_checkpoint_path(model_or_checkpoint)
        model = RslPredictionModel(
            path, device=device, input_schema=input_schema,
            expected_observation_config=expected_observation_config,
            expected_backend_identity=expected_backend_identity,
        )
        return model, path
    if isinstance(model_or_checkpoint, (str, Path)) and not legacy_sb3:
        raise ValueError("SB3 policy loading is legacy-only; pass legacy_sb3=True explicitly")
    if hasattr(model_or_checkpoint, "predict") and not legacy_sb3:
        module = type(model_or_checkpoint).__module__
        if module.startswith(("stable_baselines3", "sb3_contrib")):
            raise ValueError("SB3 policy evaluation is legacy-only; pass legacy_sb3=True explicitly")
        if input_schema != INPUT_SCHEMA and getattr(model_or_checkpoint, "backend", None) != "rsl_rl":
            raise ValueError("absolute-goal evaluation requires a native RSL-RL .pt checkpoint")
        if not hasattr(model_or_checkpoint, "policy"):
            from isaac_field.action_contract import action_config

            if getattr(model_or_checkpoint, "field_action_config", None) != action_config():
                raise ValueError("Checkpoint action contract is missing or incompatible")
            # Lightweight prediction doubles and simple policy adapters do not
            # depend on the optional SB3 package. Real SB3 policy objects have
            # a `policy` member and use the explicit legacy validator below.
            return model_or_checkpoint, None
    from isaac_field.policy import (
        _validate_checkpoint_action_metadata,
        _validate_checkpoint_encoder_metadata,
        _validate_model_encoder_contract,
        _validate_model_action_contract,
    )

    if hasattr(model_or_checkpoint, "predict"):
        _validate_model_action_contract(model_or_checkpoint)
        # Lightweight fake models used by legacy orchestration tests have no
        # policy metadata. Real models always receive the schema validation;
        # the absolute-goal path never permits a metadata-free model.
        if input_schema != INPUT_SCHEMA or hasattr(model_or_checkpoint, "policy"):
            _validate_model_encoder_contract(
                model_or_checkpoint,
                input_schema=input_schema,
                expected_observation_config=expected_observation_config,
            )
        return model_or_checkpoint, None
    if not legacy_sb3:
        raise ValueError("SB3 policy loading is legacy-only; pass legacy_sb3=True explicitly")
    from sb3_contrib import RecurrentPPO

    path = _existing_checkpoint_path(model_or_checkpoint)
    _validate_checkpoint_action_metadata(path)
    _validate_checkpoint_encoder_metadata(
        path,
        input_schema=input_schema,
        expected_observation_config=expected_observation_config,
    )
    model = RecurrentPPO.load(str(path), device=device)
    _validate_model_action_contract(model)
    _validate_model_encoder_contract(
        model,
        input_schema=input_schema,
        expected_observation_config=expected_observation_config,
    )
    return model, path


def _live_action_config(engine) -> dict[str, Any]:
    from isaac_field.action_contract import action_config

    expected = action_config()
    candidate = engine
    actual = None
    while candidate is not None:
        actual = getattr(candidate, "action_config", None)
        if actual is not None:
            break
        candidate = getattr(candidate, "env", None)
    if actual != expected:
        raise ValueError(
            "evaluation environment action_config is missing or differs from the "
            f"current action contract: {actual!r}"
        )
    return expected


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
    result = {
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
    body_pose = getattr(task, "body_pose", None)
    if body_pose is not None:
        pose = np.asarray(body_pose, dtype=np.float64)
        if pose.shape == (3,) and np.isfinite(pose).all():
            result.update(body_z_m=float(pose[0]), body_pitch_rad=float(pose[1]),
                          body_roll_rad=float(pose[2]))
    angular = getattr(task, "body_angular_velocity", None)
    if angular is not None:
        rates = np.asarray(angular, dtype=np.float64)
        if rates.shape == (3,) and np.isfinite(rates).all():
            result["body_angular_velocity_rad_s"] = rates.tolist()
    return result


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


def _reset_initial_condition(task, seed: int) -> dict[str, Any]:
    """Hash the actual generated plants and pose/goal after an evaluation reset."""
    import hashlib

    scene = getattr(task, "scene", None)
    if scene is None or not hasattr(scene, "xy"):
        raise TypeError("reset condition recording requires task.scene.xy")
    plant_xy = np.ascontiguousarray(scene.xy, dtype="<f4")
    digest = hashlib.sha256(
        np.asarray(plant_xy.shape, dtype="<i8").tobytes() + plant_xy.tobytes()
    ).hexdigest()
    robot = np.asarray(task.robot, dtype=np.float64).reshape(-1)
    goal = np.asarray(task.goal, dtype=np.float64).reshape(-1)
    if robot.size < 3 or goal.size < 2:
        raise ValueError("reset condition recording requires robot pose and goal")
    return {
        "seed": int(seed),
        "start_pose_xy_yaw": robot[:3].tolist(),
        "goal_xy": goal[:2].tolist(),
        "plant_count": int(len(plant_xy)),
        "plant_xy_sha256": digest,
    }


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
    record_initial_conditions: bool = False,
    input_schema: str | None = None,
    expected_observation_config: Mapping[str, Any] | None = None,
    legacy_sb3: bool = False,
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
    if not isinstance(record_initial_conditions, (bool, np.bool_)):
        raise ValueError("record_initial_conditions must be a boolean")
    record_initial_conditions = bool(record_initial_conditions or record_trajectories)
    engine, env_cfg = _adapter_engine_cfg(adapter)
    current_action_config = _live_action_config(engine)
    policy_input_schema, current_observation_config = _policy_contract(
        adapter,
        input_schema=input_schema,
        expected_observation_config=expected_observation_config,
    )
    if record_initial_conditions and getattr(adapter, "task", None) is None:
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

    model, checkpoint_path = _model_or_checkpoint(
        model_or_checkpoint,
        device=device,
        input_schema=policy_input_schema,
        expected_observation_config=current_observation_config,
        expected_backend_identity=_live_backend_identity(engine),
        legacy_sb3=legacy_sb3,
    )
    current_damage_config = _damage_config(adapter)
    current_damage_metric = (
        current_damage_config.get("metric") if current_damage_config is not None else None
    )
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
        reset_initial_condition = (
            _reset_initial_condition(adapter.task, seed) if record_initial_conditions else None
        )
        state = None
        episode_start = np.ones(1, dtype=bool)
        raw_return = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = dict(reset_info or {})
        steps = 0
        trajectory: list[dict[str, Any]] = []
        start_matches_reset = False
        for steps in range(1, int(max_steps) + 1):
            if steps == 1 and record_initial_conditions:
                # Test the actual pre-action state even without retaining a full
                # trajectory. This is evidence for paired checkpoint selection.
                start_matches_reset = bool(np.allclose(
                    np.asarray(adapter.task.robot)[:3],
                    reset_initial_condition["start_pose_xy_yaw"], rtol=0.0, atol=1e-8,
                ))
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
        episode_time_s = _metric_float(info.get("episode_time_s"))
        if episode_time_s is None and task is not None:
            episode_time_s = _metric_float(getattr(task, "time", None))
        if episode_time_s is None:
            episode_time_s = float(steps) * float(getattr(env_cfg, "dt", 0.0))
        row = {
            "seed": int(seed),
            "mode": str(info.get("mode", getattr(task, "mode", getattr(env_cfg, "mode", "")))),
            "raw_return": float(raw_return),
            "success": success,
            "damage": int(info.get("damage", 0)),
            "damage_metric": str(
                info.get("damage_metric") or current_damage_metric or "unrecorded"
            ),
            "collisions": int(info.get("collisions", 0)),
            "final_distance": final_distance,
            "reason": reason,
            "steps": int(steps),
            "episode_time_s": float(episode_time_s),
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
        if isinstance(info.get("body_stability_summary"), Mapping):
            row["body_stability"] = dict(info["body_stability_summary"])
        if isinstance(info.get("termination_detail"), Mapping):
            row["termination_detail"] = dict(info["termination_detail"])
        if isinstance(info.get("goal_stop"), Mapping):
            row["goal_stop"] = dict(info["goal_stop"])
        if record_initial_conditions:
            row["reset_initial_condition"] = reset_initial_condition
            row["trajectory_start_matches_reset"] = start_matches_reset
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
        # `backend_identity` historically identifies the environment used for
        # this evaluation. Keep it and spell out the two sides so a Newton-era
        # checkpoint evaluated in a PhysX environment cannot inherit PhysX
        # provenance by implication.
        "backend_identity": _live_backend_identity(engine),
        "environment_backend_identity": _live_backend_identity(engine),
        "model_backend": _model_backend_name(model),
        "model_backend_identity": _model_backend_identity(model),
        "model_class": type(model).__name__,
        "action_config": current_action_config,
        "input_schema": policy_input_schema,
        "checkpoint_action_config": getattr(model, "field_action_config", None),
        "damage_config": current_damage_config,
        "damage_metric": current_damage_metric or "unrecorded",
        "checkpoint_damage_config": getattr(model, "field_damage_config", None),
        "checkpoint_damage_metric": (
            getattr(model, "field_damage_config", {}).get("metric", "legacy/unrecorded")
            if isinstance(getattr(model, "field_damage_config", None), Mapping)
            else "legacy/unrecorded"
        ),
        "drive_config": getattr(engine, "drive_config", None),
        "checkpoint_drive_config": getattr(model, "field_drive_config", None),
        "observation_config": getattr(engine, "observation_config", None),
        "policy_observation_contract": current_observation_config,
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
    if any("goal_stop" in row for row in rows):
        success_times = [float(row["episode_time_s"]) for row in rows if row["success"]]
        report["summary"]["mean_success_time_s"] = (
            None if not success_times else float(np.mean(success_times))
        )
        report["summary"]["median_success_time_s"] = (
            None if not success_times else float(np.median(success_times))
        )
    if record_trajectories:
        report["trajectory_schema"] = "isaac-field-compact-trajectory-v1"
    return report


def _validate_lstm_hidden_size(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("lstm_hidden_size must be a positive integer")
    return value


def _validate_model_lstm_hidden_size(model, expected: int | None) -> None:
    if expected is None:
        return
    policy = getattr(model, "policy", None)
    for name in ("lstm_actor", "lstm_critic"):
        module = getattr(policy, name, None)
        actual = getattr(module, "hidden_size", None)
        if type(actual) is not int or actual != expected:
            raise ValueError(
                f"model {name} hidden_size={actual!r} does not match requested lstm_hidden_size={expected}"
            )


def _train_stage_sb3_legacy(
    env,
    checkpoint: str | Path | Any | None,
    timesteps: int,
    output: str | Path,
    callback=None,
    *,
    seed: int = 7,
    device: str = "cuda:0",
    goal_gain: float = 1.0,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    log_std_init: float = -2.0,
    target_kl: float | None = 0.03,
    verbose: int = 0,
    lstm_hidden_size: int | None = None,
    input_schema: str | None = None,
) -> StageTrainingResult:
    """Frozen historical SB3 implementation retained for forensic reference.

    ``log_std_init`` applies only when creating a fresh policy. Loading a model
    keeps its learned distribution parameters while the stage-level optimizer
    and discount settings are explicitly overridden.
    """
    raise RuntimeError(
        "SB3 training is disabled after the Isaac Sim migration; use the official RSL-RL train_stage adapter"
    )
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or int(timesteps) < 1:
        raise ValueError("timesteps must be a positive integer")
    lstm_hidden_size = _validate_lstm_hidden_size(lstm_hidden_size)
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if int(n_steps) != 128:
        # Rollout length is intentionally kept stable across saved PPO stages.
        raise ValueError("curriculum stages require n_steps=128 to match the checkpoint rollout buffer")
    if int(batch_size) < 1 or int(batch_size) > int(n_steps) or int(n_epochs) < 1:
        raise ValueError("require 1 <= batch_size <= n_steps and n_epochs >= 1")
    if not math.isfinite(float(goal_gain)) or float(goal_gain) != 1.0:
        raise ValueError("goal_gain must be 1.0: extra goal-input scaling has been removed")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not math.isfinite(float(gae_lambda)) or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0, 1]")
    if target_kl is not None and (
        not math.isfinite(float(target_kl)) or float(target_kl) <= 0.0
    ):
        raise ValueError("target_kl must be finite and positive when provided")

    from sb3_contrib import RecurrentPPO
    from isaac_field.policy import (
        _validate_checkpoint_action_metadata,
        _validate_checkpoint_encoder_metadata,
        _validate_model_encoder_contract,
        _validate_model_action_contract,
        make_model,
        validate_policy_spaces,
    )
    from isaac_field.training import _sb3_wrapper

    if hasattr(env, "engine"):
        raise TypeError("train_stage requires a native batched Isaac Lab env, not SingleFieldAdapter")
    policy_input_schema, observation_config = _policy_contract(
        env,
        input_schema=input_schema,
    )
    from isaac_field.action_contract import action_config

    expected_action_config = action_config()
    action_source = env
    live_action_config = None
    while action_source is not None:
        live_action_config = getattr(action_source, "action_config", None)
        if live_action_config is not None:
            break
        action_source = getattr(action_source, "env", None)
    if live_action_config != expected_action_config:
        raise ValueError(
            "training environment action_config is missing or differs from the "
            f"current action contract: {live_action_config!r}"
        )
    vec_env = _sb3_wrapper(env)
    if policy_input_schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
        validate_policy_spaces(
            vec_env.observation_space,
            vec_env.action_space,
            input_schema=policy_input_schema,
            expected_observation_config=observation_config,
        )
    else:
        validate_policy_spaces(vec_env.observation_space, vec_env.action_space)
    if hasattr(checkpoint, "predict"):
        _validate_model_lstm_hidden_size(checkpoint, lstm_hidden_size)
        _validate_model_action_contract(checkpoint)
        if policy_input_schema != INPUT_SCHEMA or hasattr(checkpoint, "policy"):
            _validate_model_encoder_contract(
                checkpoint,
                input_schema=policy_input_schema,
                expected_observation_config=observation_config,
            )
        model = checkpoint
        model.set_env(vec_env)
    elif checkpoint is None:
        model_kwargs = {
            "seed": int(seed),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "n_epochs": int(n_epochs),
            "learning_rate": float(learning_rate),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "goal_gain": float(goal_gain),
            "log_std_init": float(log_std_init),
            "target_kl": target_kl,
            "device": device,
            "verbose": verbose,
            "input_schema": policy_input_schema,
        }
        if observation_config is not None:
            model_kwargs["observation_config"] = observation_config
        if lstm_hidden_size is not None:
            model_kwargs["lstm_hidden_size"] = lstm_hidden_size
        model = make_model(
            vec_env,
            **model_kwargs,
        )
    else:
        checkpoint_path = _existing_checkpoint_path(checkpoint)
        _validate_checkpoint_action_metadata(checkpoint_path)
        _validate_checkpoint_encoder_metadata(
            checkpoint_path,
            input_schema=policy_input_schema,
            expected_observation_config=observation_config,
        )
        model = RecurrentPPO.load(
            str(checkpoint_path),
            env=vec_env,
            device=device,
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            target_kl=target_kl,
        )
    _validate_model_lstm_hidden_size(model, lstm_hidden_size)
    _validate_model_action_contract(model)
    if policy_input_schema != INPUT_SCHEMA or hasattr(model, "policy"):
        _validate_model_encoder_contract(
            model,
            input_schema=policy_input_schema,
            expected_observation_config=observation_config,
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
    model.gae_lambda = float(gae_lambda)
    rollout_buffer = getattr(model, "rollout_buffer", None)
    if rollout_buffer is not None and hasattr(rollout_buffer, "gamma"):
        rollout_buffer.gamma = float(gamma)
    if rollout_buffer is not None and hasattr(rollout_buffer, "gae_lambda"):
        rollout_buffer.gae_lambda = float(gae_lambda)
    model.target_kl = target_kl
    # A migrated checkpoint contains the previous simulator's rolling stats.
    # They must not appear as successes measured in this Isaac Lab stage.
    for buffer_name in ('ep_info_buffer','ep_success_buffer'):
        buffer=getattr(model,buffer_name,None)
        if buffer is not None:
            buffer.clear()
    model.set_random_seed(int(seed))
    # Preserve the validated action semantics; they cannot be rewritten to
    # relabel an old model whose same-shaped outputs meant something else.
    if hasattr(env, "drive_config"):
        model.field_drive_config = dict(env.drive_config)
    if hasattr(env, "observation_config"):
        model.field_observation_config = dict(env.observation_config)
    damage_config = _damage_config(env)
    if damage_config is not None:
        model.field_damage_config = damage_config
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


def train_stage(
    env,
    checkpoint: str | Path | Any | None,
    timesteps: int,
    output: str | Path,
    callback=None,
    *,
    seed: int = 7,
    device: str = "cuda:0",
    goal_gain: float = 1.0,
    n_steps: int = 128,
    batch_size: int = 128,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    log_std_init: float = -2.0,
    target_kl: float | None = 0.03,
    verbose: int = 0,
    lstm_hidden_size: int | None = None,
    input_schema: str | None = None,
    resume_optimizer_state: bool = True,
    weights_transfer: bool = False,
) -> StageTrainingResult:
    """Train a stage with Isaac Lab's official RSL-RL ``OnPolicyRunner``.

    This adapter only builds the supported field runner config and delegates
    rollout collection, PPO updates, checkpoint loading, and saving to the
    upstream runner. SB3 callbacks and ZIP checkpoints are rejected because
    they cannot be applied to the native RSL path. ``log_std_init`` applies
    only to a fresh model; checkpoint workflows retain the parent's recorded
    Gaussian construction config and restore its learned state.
    """
    if callback is not None:
        raise TypeError("RSL-RL training does not accept SB3 callbacks; use runner save_interval/logging")
    if isinstance(timesteps, bool) or int(timesteps) != timesteps or int(timesteps) < 0:
        raise ValueError("timesteps must be a non-negative integer")
    if type(resume_optimizer_state) is not bool:
        raise ValueError("resume_optimizer_state must be boolean")
    if type(weights_transfer) is not bool:
        raise ValueError("weights_transfer must be boolean")
    if weights_transfer and checkpoint is None:
        raise ValueError("weights_transfer requires an explicit RSL .pt parent checkpoint")
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint) if isinstance(checkpoint, (str, Path)) else None
        if checkpoint_path is None:
            raise TypeError("RSL stage resume requires an explicit .pt checkpoint path")
        if checkpoint_path.suffix.lower() in (".zip", ".pkl"):
            raise ValueError("SB3 checkpoints are legacy-only and cannot train an Isaac Sim RSL stage")
        checkpoint = _existing_checkpoint_path(checkpoint_path)
    checkpoint_payload = None
    if checkpoint is not None:
        from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

        checkpoint_payload = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    lstm_hidden_size = _validate_lstm_hidden_size(lstm_hidden_size)
    if not math.isfinite(float(goal_gain)) or float(goal_gain) != 1.0:
        raise ValueError("goal_gain must be 1.0: extra goal-input scaling has been removed")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not math.isfinite(float(gae_lambda)) or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0, 1]")
    if target_kl is not None and (
        not math.isfinite(float(target_kl)) or float(target_kl) <= 0.0
    ):
        raise ValueError("target_kl must be finite and positive when provided")
    if (
        isinstance(batch_size, bool)
        or int(batch_size) != batch_size
        or int(batch_size) < 1
        or isinstance(n_epochs, bool)
        or int(n_epochs) != n_epochs
        or int(n_epochs) < 1
    ):
        raise ValueError("batch_size and num_learning_epochs must be positive integers")
    batch_size = int(batch_size)
    n_epochs = int(n_epochs)

    from isaac_field.policy_contract import BACKEND_IDENTITY, _encoder_schema_for_input
    from isaac_field.rsl_rl_cfg import FieldFastStopRslRunnerCfg
    from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import checkpoint_load_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version as package_version
    from rsl_rl.runners import OnPolicyRunner

    if hasattr(env, "engine"):
        raise TypeError("train_stage requires a native Isaac Lab batched environment, not SingleFieldAdapter")
    engine = getattr(env, "unwrapped", env)
    identity = getattr(engine, "backend_identity", None)
    if callable(identity):
        identity = identity()
    if identity != BACKEND_IDENTITY:
        raise ValueError(
            "training requires an IsaacFieldRslEnv with backend_identity="
            f"{BACKEND_IDENTITY!r}; got {identity!r}"
        )
    expected_observation_keys = {"lidar", "depth", "detections", "state", "feedback"}
    observation_buffer = getattr(engine, "obs_buf", None)
    observation_space = getattr(engine, "single_observation_space", None)
    observation_leaves = getattr(observation_space, "spaces", None)
    has_initialized_observations = (
        isinstance(observation_buffer, Mapping)
        and set(observation_buffer) == expected_observation_keys
    )
    has_root_observation_space = (
        isinstance(observation_leaves, Mapping)
        and set(observation_leaves) == expected_observation_keys
    )
    if not (has_initialized_observations or has_root_observation_space):
        raise TypeError(
            "training environment must expose the five root RSL observation leaves in "
            "obs_buf or single_observation_space"
        )
    policy_input_schema, observation_config = _policy_contract(env, input_schema=input_schema)
    if not _observation_config_is_complete(observation_config):
        raise ValueError("RSL training requires a complete observation_config")
    if getattr(env, "num_envs", None) is None or int(env.num_envs) < 1:
        raise ValueError("RSL training environment must expose a positive num_envs")

    output_path = Path(output)
    if output_path.suffix.lower() != ".pt":
        raise ValueError("native RSL checkpoints must use the .pt extension")
    log_dir = output_path.parent / f"{output_path.stem}_logs"
    if output_path.exists() or log_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing RSL checkpoint or log directory: {output_path}"
        )
    if lstm_hidden_size is None:
        if checkpoint is None:
            lstm_hidden_size = 512
        else:
            recorded_hidden = checkpoint_payload["field_policy_contract"]["lstm"]["hidden_dim"]
            lstm_hidden_size = _validate_lstm_hidden_size(recorded_hidden)

    rollout_transitions = int(env.num_envs) * int(n_steps)
    if int(timesteps) % rollout_transitions:
        raise ValueError(
            f"timesteps={timesteps} must be divisible by num_envs*num_steps_per_env={rollout_transitions}"
        )
    iterations = int(timesteps) // rollout_transitions
    if int(batch_size) > rollout_transitions:
        raise ValueError(
            f"batch_size={batch_size} cannot exceed rollout transitions={rollout_transitions}"
        )
    if rollout_transitions % int(batch_size):
        raise ValueError(
            f"rollout transitions={rollout_transitions} must be divisible by batch_size={batch_size}"
        )
    from isaac_field.runtime import simulator_identity

    if simulator_identity() != BACKEND_IDENTITY:
        raise RuntimeError("runtime simulator identity differs from the supported PhysX RSL backend")
    runner_cfg = FieldFastStopRslRunnerCfg()
    runner_cfg.seed = int(seed)
    runner_cfg.num_steps_per_env = int(n_steps)
    runner_cfg.max_iterations = int(iterations)
    num_mini_batches = max(1, rollout_transitions // int(batch_size))
    runner_cfg.save_interval = max(1, min(int(getattr(runner_cfg, "save_interval", 16)), iterations or 1))
    runner_cfg.device = str(device)
    runner_cfg.run_name = output_path.stem
    runner_cfg.actor.input_schema = policy_input_schema
    runner_cfg.critic.input_schema = policy_input_schema
    runner_cfg.actor.encoder_schema = _encoder_schema_for_input(policy_input_schema)
    runner_cfg.critic.encoder_schema = _encoder_schema_for_input(policy_input_schema)
    runner_cfg.actor.observation_shapes = {
        key: list(value) for key, value in observation_config["observation_shapes"].items()
    }
    runner_cfg.critic.observation_shapes = dict(runner_cfg.actor.observation_shapes)
    runner_cfg.actor.rnn_hidden_dim = int(lstm_hidden_size)
    runner_cfg.critic.rnn_hidden_dim = int(lstm_hidden_size)
    runner_cfg.algorithm.num_learning_epochs = int(n_epochs)
    runner_cfg.algorithm.num_mini_batches = int(num_mini_batches)
    runner_cfg.algorithm.learning_rate = float(learning_rate)
    runner_cfg.algorithm.gamma = float(gamma)
    runner_cfg.algorithm.lam = float(gae_lambda)
    runner_cfg.algorithm.desired_kl = target_kl
    if log_std_init is not None and not math.isfinite(float(log_std_init)):
        raise ValueError("log_std_init must be finite")
    if checkpoint_payload is not None:
        # Constructor metadata must match the parent contract for strict resume
        # and explicit weights transfer. The checkpoint state restores the
        # learned standard deviation; this only preserves its initial config.
        runner_cfg.actor.distribution_cfg.init_std = float(
            checkpoint_payload["field_policy_contract"]["distribution"]["init_std"]
        )
    elif log_std_init is not None:
        # RSL-RL records standard deviation in linear space; retain the public
        # parameter's log-space meaning when creating a fresh model.
        runner_cfg.actor.distribution_cfg.init_std = math.exp(float(log_std_init))
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, package_version("rsl-rl-lib"))

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(wrapped_env, runner_cfg.to_dict(), log_dir=str(log_dir), device=str(device))
    transfer_record = None
    if checkpoint is not None:
        if weights_transfer:
            from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload
            from isaac_field.policy_contract import (
                recorded_backend_identity,
                require_backend_identity,
            )

            payload = load_rsl_checkpoint_payload(checkpoint, map_location="cpu")
            source_contract = payload["field_policy_contract"]
            target_contract = runner.alg.field_policy_contract
            source_backend_identity = recorded_backend_identity({
                key: source_contract.get(key) for key in BACKEND_IDENTITY
            })
            target_backend_identity = require_backend_identity(target_contract)
            compatible_keys = (
                "input_schema", "encoder_config", "observation_shapes", "observation_config",
                "action_config", "actor_config", "critic_config", "obs_groups", "lstm",
                "heads", "distribution", "action_dim",
            )
            differences = [
                key for key in compatible_keys if source_contract.get(key) != target_contract.get(key)
            ]
            if differences:
                raise ValueError(
                    "weights transfer source model structure differs from the target; "
                    f"mismatched contract fields: {differences}"
                )
            runner.alg._raw_actor.load_state_dict(payload["actor_state_dict"], strict=True)
            runner.alg._raw_critic.load_state_dict(payload["critic_state_dict"], strict=True)
            runner.alg.optimizer.state.clear()
            runner.alg.num_timesteps = 0
            with Path(checkpoint).open("rb") as stream:
                parent_checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            transfer_record = {
                "mode": "weights_transfer",
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_backend_identity": source_backend_identity,
                "target_backend_identity": target_backend_identity,
                "parent_task_config": source_contract.get("task_config"),
                "optimizer": "reinitialized",
                "timestep_counter": "reset_to_zero",
            }
        else:
            runner.load(
                str(checkpoint),
                load_cfg=checkpoint_load_cfg(
                    bool(resume_optimizer_state), runner_class=runner_cfg.class_name,
                ),
                map_location=str(device),
            )
    before = int(getattr(runner.alg, "num_timesteps", 0))
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
    after = int(getattr(runner.alg, "num_timesteps", before))
    if after - before != int(timesteps):
        raise RuntimeError(
            f"official RSL runner trained {after - before} transitions; expected {timesteps}"
        )
    runner.save(str(output_path))
    if transfer_record is not None:
        import torch

        payload = torch.load(output_path, map_location="cpu", weights_only=True)
        payload["training_initialization"] = transfer_record
        torch.save(payload, output_path)
    return StageTrainingResult(
        model=runner.alg,
        checkpoint=output_path,
        requested_timesteps=int(timesteps),
        trained_timesteps=after - before,
        total_timesteps=after,
        seed=int(seed),
        goal_gain=float(goal_gain),
        n_steps=int(n_steps),
        batch_size=int(batch_size),
        n_epochs=int(n_epochs),
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
        "rsl": True,
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


@_keep_runtime_for_workflow
def train_curriculum(
    cfg: Config,
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    timesteps_per_stage: int | Sequence[int] = 128,
    training_seed: int = 7,
    heldout_seeds: Sequence[int] = DEFAULT_HELDOUT_SEEDS,
    stages: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
    evaluate_before_after: bool = True,
    goal_gain: float = 1.0,
    gamma: float = 0.995,
    device: str = "cuda:0",
    n_epochs: int = 5,
    batch_size: int = 128,
    callback_factory: Callable[[CurriculumStage], Any] | None = None,
    options_factory: Callable[[int], Mapping[str, Any] | None] | None = None,
    env_factory: Callable[..., Any] | None = None,
    adapter_factory: Callable[[Any], Any] | None = None,
    lstm_hidden_size: int | None = None,
) -> CurriculumResult:
    """Train stages sequentially, closing every physics context before opening another.

    The default stage seeds are consecutive from ``training_seed``. Held-out
    evaluation is run before and after all stages with the same fixed seeds, on
    separate mixed-mode evaluation contexts and with raw reward only.
    """
    stages = tuple(stages)
    lstm_hidden_size = _validate_lstm_hidden_size(lstm_hidden_size)
    if not stages:
        raise ValueError("stages must contain at least one curriculum stage")
    if callback_factory is not None:
        raise TypeError("SB3 callback_factory is not supported by RSL-RL; use runner save_interval/logging")
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
    if not math.isfinite(float(goal_gain)) or float(goal_gain) != 1.0:
        raise ValueError("goal_gain must be 1.0: extra goal-input scaling has been removed")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")

    env_factory = env_factory or _make_env_factory()
    adapter_factory = adapter_factory or _make_adapter_factory()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    active_checkpoint: str | Path | Any | None = checkpoint
    if checkpoint is not None:
        checkpoint = _existing_checkpoint_path(checkpoint)

    evaluation_context_seed = max((*heldout, *training_seeds), default=0) + 1_000_003
    initial_evaluation: dict[str, Any] | None = None
    if active_checkpoint is None:
        baseline_env = env_factory(
            **_default_factory_kwargs(
                cfg,
                mode="mixed",
                seed=evaluation_context_seed,
                goal_distance=None,
                goal_angle=math.pi,
                auto_reset=False,
            )
        )
        try:
            initialized = train_stage(
                baseline_env,
                None,
                0,
                output / "untrained_baseline.pt",
                seed=int(training_seed),
                device=device,
                goal_gain=goal_gain,
                gamma=float(gamma),
                n_epochs=n_epochs,
                batch_size=batch_size,
                lstm_hidden_size=lstm_hidden_size,
                input_schema=cfg.input_schema,
            )
            active_checkpoint = initialized.checkpoint
        finally:
            _close_context(baseline_env)
    if evaluate_before_after:
        adapter, engine = _open_evaluation_adapter(
            cfg,
            env_factory=env_factory,
            adapter_factory=adapter_factory,
            seed=evaluation_context_seed,
        )
        try:
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
        stage_output = output / f"stage_{index + 1:02d}_{stage.name}.pt"
        try:
            stage_kwargs = {
                "seed": stage_seed,
                "device": device,
                "goal_gain": goal_gain,
                "gamma": float(gamma),
                "n_epochs": n_epochs,
                "batch_size": batch_size,
            }
            if lstm_hidden_size is not None:
                stage_kwargs["lstm_hidden_size"] = lstm_hidden_size
            result = train_stage(
                engine,
                active_checkpoint,
                int(stage_timesteps),
                stage_output,
                **stage_kwargs,
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
        "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner",
        "backend_identity": {
            "simulator": "isaac_sim", "physics_backend": "physx", "learning_backend": "rsl_rl",
        },
        "input_schema": cfg.input_schema,
        "checkpoint_start": None if checkpoint is None else str(checkpoint),
        "checkpoint_final": str(final_checkpoint),
        "goal_gain": float(goal_gain),
        "gamma": float(gamma),
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

