"""One same-context, naive-minimax teacher/student training round."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _weights(model) -> dict[str, np.ndarray]:
    policy = getattr(model, "policy", None)
    if policy is None:
        policy = getattr(model, "actor", None)
    if policy is None and getattr(model, "model", None) is not None:
        policy = getattr(model.model, "actor", model.model)
    if policy is None:
        policy = model
    return {k: np.asarray(v.detach().cpu() if hasattr(v, "detach") else v).copy()
            for k, v in policy.state_dict().items()}


def _changed(before, model) -> bool:
    after = _weights(model)
    return set(before) != set(after) or any(not np.array_equal(before[k], after[k]) for k in before)


def _score(adapter, runner, scene, seed, task_cfg, goal_distance, goal_angle):
    obs, _ = adapter.reset(seed=seed, options={"scene": scene, "random_start": True, "mode": "goto"})
    runner.reset()
    max_steps = max(1, math.ceil(float(task_cfg.max_time) / float(task_cfg.dt)))
    score, info, terminated, truncated = 0.0, {}, False, False
    for steps in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = adapter.step(runner.action(obs))
        score += float(reward)
        if terminated or truncated:
            break
    truncated = bool(truncated or not terminated)
    task = getattr(adapter, "task", None)
    reason = str(info.get("reason", "time_limit" if truncated else ""))
    distance = info.get("distance")
    if distance is None and task is not None:
        distance = float(np.linalg.norm(task.goal - task.robot[:2]))
    return {"seed": int(seed), "student_score": float(score), "raw_return": float(score),
            "success": bool(info.get("is_success", reason == "goal_reached")),
            "damage": int(info.get("damage", 0)), "collisions": int(info.get("collisions", 0)),
            "final_distance": None if distance is None else float(distance), "reason": reason,
            "steps": int(steps), "terminated": bool(terminated), "truncated": truncated,
            "goal_distance": list(goal_distance), "goal_angle_rad": float(goal_angle)}


def _initial_teacher_model(env, *, seed, device, ppo_kwargs):
    """Construct the same official RSL MLP actor used by train_designer."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel

    settings = dict(ppo_kwargs)
    observation, _ = env.reset(seed=int(seed))
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    observations = TensorDict({"obs": obs}, batch_size=[1])
    return MLPModel(
        observations,
        {"actor": ["obs"], "critic": ["obs"]},
        "actor",
        3,
        hidden_dims=list(settings.get("hidden_dims", (64, 64))),
        activation=str(settings.get("activation", "tanh")),
        obs_normalization=False,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": math.exp(float(settings.get("log_std_init", 0.0))),
            "std_type": "log",
        },
    )


def run_round(
    env,
    student_checkpoint: str | Path,
    output: str | Path,
    teacher_steps: int = 16,
    student_steps: int = 256,
    seed: int = 7,
    *,
    edit_budget: int = 8,
    goal_distance: tuple[float, float] = (1.0, 3.0),
    goal_angle: float = math.pi,
    teacher_device: str = "cpu",
    student_device: str = "cuda:0",
    goal_gain: float = 1.0,
    weights_transfer: bool = False,
    teacher_ppo_kwargs: Mapping[str, Any] | None = None,
    student_train_kwargs: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the original one-sided map-designer/student loop with official RSL-RL."""

    teacher_steps, student_steps = int(teacher_steps), int(student_steps)
    edit_budget, seed = int(edit_budget), int(seed)
    if type(weights_transfer) is not bool:
        raise ValueError("weights_transfer must be boolean")
    if teacher_steps < edit_budget or student_steps < 1:
        raise ValueError("teacher_steps must cover one teacher episode and student_steps must be positive")
    if not 8 <= edit_budget <= 16 or seed < 0:
        raise ValueError("edit_budget must be 8..16 and seed must be non-negative")
    lo, hi = map(float, goal_distance)
    if not math.isfinite(lo + hi) or lo < 0 or hi < lo or not 0 <= goal_angle <= math.pi:
        raise ValueError("invalid goal sampling limits")

    deps = dict(dependencies or {})
    if not all(key in deps for key in ("adapter_factory", "runner_factory", "designer_factory",
                                       "teacher_trainer", "student_trainer", "row_positions")):
        from isaac_field.adversary import MapDesignerEnv, train_designer
        from isaac_field.env import SingleFieldAdapter
        from isaac_field.learning import train_stage
        from isaac_field.rsl_rl_inference import RslPolicyRunner
        from isaac_field.layouts import row_positions
        for key, value in (("adapter_factory", SingleFieldAdapter), ("runner_factory", RslPolicyRunner),
                           ("designer_factory", MapDesignerEnv), ("teacher_trainer", train_designer),
                           ("student_trainer", train_stage), ("row_positions", row_positions)):
            deps.setdefault(key, value)
    Adapter, Runner, Designer = (deps[k] for k in ("adapter_factory", "runner_factory", "designer_factory"))
    train_teacher, train_student = deps["teacher_trainer"], deps["student_trainer"]
    row_positions = deps["row_positions"]
    make_initial_teacher = deps.get("initial_teacher_model_factory", _initial_teacher_model)
    from isaac_field.policy_contract import BACKEND_IDENTITY, recorded_backend_identity
    from isaac_field.adversary import TEACHER_RUNTIME_IDENTITY
    production_rsl_student = getattr(Runner, "__name__", "") == "RslPolicyRunner"
    environment_identity = getattr(env, "backend_identity", None)
    if callable(environment_identity):
        environment_identity = environment_identity()
    if not isinstance(environment_identity, Mapping):
        environment_identity = None
    else:
        environment_identity = dict(environment_identity)
    student_physx_evaluated = bool(
        production_rsl_student and environment_identity == BACKEND_IDENTITY
    )
    evaluator_kind = (
        "physx_student_policy_rollout"
        if student_physx_evaluated else "test_or_custom_score_callback"
    )
    student_path = Path(student_checkpoint).expanduser().resolve()
    if production_rsl_student:
        if student_path.suffix.lower() != ".pt":
            raise ValueError("new adversarial rounds require an RSL .pt student checkpoint; SB3 .zip is legacy-only")
        if not student_path.is_file():
            raise FileNotFoundError(student_path)

    if int(getattr(env, "num_envs", 0)) != 1:
        raise ValueError("run_round requires a native Isaac Lab environment with num_envs=1")
    cfg, task_cfg = env.cfg, env.field_cfg
    cfg.mode, cfg.auto_reset, cfg.distance_cost = "goto", False, 0.0
    cfg.goal_distance, cfg.goal_angle = (lo, hi), float(goal_angle)
    env.layout_sampler = None
    adapter = Adapter(env)
    adapter.reset(seed=seed, options={"mode": "goto", "random_start": True})
    template = copy.deepcopy(adapter.scene)
    if template is None:
        raise RuntimeError("native environment did not create a scene template")

    student_identity = recorded_backend_identity({})
    source_sha256 = None
    source_hidden_dim = None
    student_contract = None
    task_values = asdict(task_cfg) if is_dataclass(task_cfg) else (
        dict(vars(task_cfg)) if hasattr(task_cfg, "__dict__") else task_cfg
    )
    transfer = weights_transfer
    if production_rsl_student:
        from isaac_field.rsl_rl_policy import load_rsl_checkpoint_payload

        student_payload = load_rsl_checkpoint_payload(student_path, map_location="cpu")
        student_contract = student_payload["field_policy_contract"]
        student_identity = recorded_backend_identity({
            key: student_contract.get(key)
            for key in ("simulator", "physics_backend", "learning_backend")
        })
        lstm_contract = student_contract.get("lstm")
        source_hidden_dim = lstm_contract.get("hidden_dim") if isinstance(lstm_contract, Mapping) else None
        if type(source_hidden_dim) is not int or source_hidden_dim < 1:
            raise ValueError("student checkpoint lacks a positive recorded LSTM hidden_dim")
        task_contract_differs = student_contract.get("task_config") != task_values
        backend_differs = student_identity != BACKEND_IDENTITY
        if (task_contract_differs or backend_differs) and not weights_transfer:
            reasons = []
            if task_contract_differs:
                reasons.append("task_config")
            if backend_differs:
                reasons.append("simulator/backend identity")
            raise ValueError(
                "student checkpoint differs in " + " and ".join(reasons)
                + "; pass weights_transfer=True to initialize a new trial from weights"
            )
        import hashlib

        source_sha256 = hashlib.sha256(student_path.read_bytes()).hexdigest()

    if production_rsl_student:
        from isaac_field.feedback import INPUT_SCHEMA

        runner = Runner(
            adapter, student_path, device=student_device,
            input_schema=getattr(task_cfg, "input_schema", None) or INPUT_SCHEMA,
            expected_observation_config=getattr(env, "observation_config", None),
            allow_backend_transfer=weights_transfer,
        )
    else:
        runner = Runner(adapter, student_checkpoint, device=student_device)
    student_before = _weights(runner.model)
    rollouts, teacher_scenes = [], []
    rollout_seed = seed + 1_000_003

    def evaluate(scene):
        nonlocal rollout_seed
        if np.asarray(scene.xy).shape != np.asarray(template.xy).shape or not np.array_equal(scene.targets, template.targets):
            raise ValueError("teacher layout changed fixed plant or target shape")
        metrics = _score(adapter, runner, scene, rollout_seed, task_cfg, (lo, hi), goal_angle)
        rollout_seed += 1
        rollouts.append({**metrics, "layout_xy": np.asarray(scene.xy).tolist(),
                         "targets": np.asarray(scene.targets, dtype=bool).tolist(),
                         "terrain_seed": getattr(scene, "terrain_seed", None)})
        teacher_scenes.append(copy.deepcopy(scene))
        return metrics["student_score"]

    designer_kwargs = {
        "seed": seed,
        "layout_template": template,
        "terrain": getattr(env, "terrain", template.terrain),
        "edit_budget": edit_budget,
    }
    if getattr(Designer, "__module__", "") == "isaac_field.adversary":
        designer_kwargs.update(
            evaluator_kind=evaluator_kind,
            student_physx_evaluated=student_physx_evaluated,
        )
    designer = Designer(task_cfg, evaluate, **designer_kwargs)
    ppo = {"n_steps": min(64, teacher_steps), "batch_size": min(32, teacher_steps),
           "n_epochs": 3, "learning_rate": 3e-4, "gamma": 0.99, "verbose": 0}
    # Use an exact divisor when possible so the 16-step smoke round scores two
    # complete 8-edit teacher episodes without PPO silently extending it.
    while ppo["n_steps"] > 1 and teacher_steps % ppo["n_steps"]:
        ppo["n_steps"] -= 1
        ppo["batch_size"] = min(32, ppo["n_steps"])
    ppo.update(dict(teacher_ppo_kwargs or {}))
    initial_teacher = make_initial_teacher(designer, seed=seed, device=teacher_device, ppo_kwargs=ppo)
    teacher_before = _weights(initial_teacher)
    del initial_teacher

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    protected_outputs = (
        protocol_path,
        out / "round_summary.json",
        out / "teacher_policy.pt",
        out / "teacher_policy.summary.json",
        out / "teacher_policy.episodes.json",
        out / "student_policy.pt",
    )
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(f"refusing to reuse an existing adversarial round directory: {out}")
    protocol = {
        "schema": "isaac-field-adversarial-round-protocol-v2",
        "target_student_backend_identity": dict(BACKEND_IDENTITY),
        "objective": "naive_minimax_teacher_reward_equals_negative_student_raw_return",
        "paired": False,
        "student_checkpoint": str(student_path),
        "student_checkpoint_sha256": source_sha256,
        "student_backend_identity": student_identity,
        "task_config": task_values,
        "teacher": {
            "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner PPO with MLPModel",
            "runtime_identity": dict(TEACHER_RUNTIME_IDENTITY),
            "evaluator_kind": evaluator_kind,
            "student_physx_evaluated": student_physx_evaluated,
            "target_student_backend_identity": dict(BACKEND_IDENTITY),
            "timesteps": teacher_steps,
            "edit_budget": edit_budget,
            "hyperparameters": ppo,
        },
        "student": {
            "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner",
            "timesteps": student_steps,
            "weights_transfer": transfer,
            "optimizer": "reinitialized" if transfer else "resumed",
            "initial_policy_evaluation": {
                "model_backend_identity": student_identity,
                "environment_backend_identity": environment_identity,
                "cross_backend_evaluation": student_identity != BACKEND_IDENTITY,
            },
            "initialization": {
                "parent_checkpoint_sha256": source_sha256,
                "parent_backend_identity": student_identity,
                "target_backend_identity": BACKEND_IDENTITY,
            },
        },
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    teacher_result = train_teacher(designer, out / "teacher_policy", teacher_steps,
                                   seed=seed, device=teacher_device, ppo_kwargs=ppo)
    if not teacher_scenes:
        raise RuntimeError("teacher training did not produce a scored layout")

    def layout_sampler(rng):
        if teacher_scenes and rng.random() >= 0.5:
            return copy.deepcopy(teacher_scenes[int(rng.integers(len(teacher_scenes)))])
        for _ in range(32):
            try:
                xy = row_positions(task_cfg, rng, terrain=designer.terrain)
                scene = designer._scene_at_positions(xy)
            except (TypeError, ValueError):
                continue
            valid, _ = designer._layout_valid(xy, scene=scene)
            if valid:
                return copy.deepcopy(scene)
        return None

    layout_sampler.teacher_scene_probability = 0.5
    cfg.auto_reset = True
    env.layout_sampler = layout_sampler
    episode_results = getattr(env, "episode_results", [])
    first_student_episode = len(episode_results)
    student_kwargs = dict(student_train_kwargs or {})
    if "weights_transfer" in student_kwargs:
        raise ValueError("run_round selects strict resume or explicit task-config weights transfer itself")
    student_kwargs["weights_transfer"] = transfer
    if production_rsl_student:
        student_kwargs.setdefault("lstm_hidden_size", source_hidden_dim)
    student_result = train_student(env, student_checkpoint, student_steps, out / "student_policy",
                                   seed=seed + 1, device=student_device, goal_gain=goal_gain,
                                   **student_kwargs)
    summary = {
        "schema": "isaac-field-adversarial-round-v1", "paired": False,
        "algorithm": "Isaac Lab official RSL-RL OnPolicyRunner PPO",
        "target_student_backend_identity": dict(BACKEND_IDENTITY),
        "student_environment_backend_identity": environment_identity,
        "teacher_runtime_identity": dict(TEACHER_RUNTIME_IDENTITY),
        "teacher_evaluator_kind": evaluator_kind,
        "student_physx_evaluated": student_physx_evaluated,
        "objective": "naive_minimax: teacher_reward = -student raw task return",
        "limitation": "Short loop-closure self-check only; changed weights do not establish competence or improvement; not PAIRED.",
        "seed": seed, "teacher_steps_requested": teacher_steps,
        "teacher_steps_actual": int(getattr(teacher_result, "actual_timesteps",
                                             getattr(teacher_result.model, "num_timesteps", teacher_steps))),
        "teacher_checkpoint": str(getattr(teacher_result, "checkpoint", out / "teacher_policy.pt")),
        "teacher_weights_changed": _changed(teacher_before, teacher_result.model),
        "teacher_rollouts": rollouts,
        "layout_sampler": "0.5 fresh valid rows / 0.5 actually teacher-scored layouts",
        "student_steps_requested": student_steps,
        "student_steps_actual": int(getattr(student_result, "trained_timesteps", student_steps)),
        "student_total_timesteps": int(getattr(student_result, "total_timesteps", 0)),
        "student_checkpoint": str(getattr(student_result, "checkpoint", out / "student_policy.pt")),
        "student_weights_transfer": transfer,
        "student_weights_changed": _changed(student_before, student_result.model),
        "student_episodes": getattr(env, "episode_results", [])[first_student_episode:],
    }
    (out / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

