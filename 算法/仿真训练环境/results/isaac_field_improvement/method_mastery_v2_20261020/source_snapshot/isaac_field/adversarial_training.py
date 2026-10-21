"""One same-context, naive-minimax teacher/student training round."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _weights(model) -> dict[str, np.ndarray]:
    policy = getattr(model, "policy", model)
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
    from stable_baselines3 import PPO

    return PPO("MlpPolicy", env, seed=seed, device=device, **dict(ppo_kwargs))


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
    teacher_ppo_kwargs: Mapping[str, Any] | None = None,
    student_train_kwargs: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score teacher-edited layouts with a frozen student, then train the student.

    The caller owns the native one-row environment. Its SimulationContext is
    reused for teacher rollouts and student training. Defaults are a wiring
    self-check only and do not establish task competence or improvement.

    ``dependencies`` optionally overrides ``adapter_factory``, ``runner_factory``,
    ``designer_factory``, ``teacher_trainer``, ``student_trainer`` and
    ``initial_teacher_model_factory`` for simulator-free orchestration tests.
    """
    teacher_steps, student_steps = int(teacher_steps), int(student_steps)
    edit_budget, seed = int(edit_budget), int(seed)
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
        from isaac_field.policy import PolicyRunner
        from isaac_field.layouts import row_positions
        for key, value in (("adapter_factory", SingleFieldAdapter), ("runner_factory", PolicyRunner),
                           ("designer_factory", MapDesignerEnv), ("teacher_trainer", train_designer),
                           ("student_trainer", train_stage), ("row_positions", row_positions)):
            deps.setdefault(key, value)
    Adapter, Runner, Designer = (deps[k] for k in ("adapter_factory", "runner_factory", "designer_factory"))
    train_teacher, train_student = deps["teacher_trainer"], deps["student_trainer"]
    row_positions = deps["row_positions"]
    make_initial_teacher = deps.get("initial_teacher_model_factory", _initial_teacher_model)

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

    designer = Designer(task_cfg, evaluate, seed=seed, layout_template=template,
                        terrain=getattr(env, "terrain", template.terrain), edit_budget=edit_budget)
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
    student_result = train_student(env, student_checkpoint, student_steps, out / "student_policy",
                                   seed=seed + 1, device=student_device, goal_gain=goal_gain,
                                   **dict(student_train_kwargs or {}))
    summary = {
        "schema": "isaac-field-adversarial-round-v1", "paired": False,
        "objective": "naive_minimax: teacher_reward = -student raw task return",
        "limitation": "Short loop-closure self-check only; changed weights do not establish competence or improvement; not PAIRED.",
        "seed": seed, "teacher_steps_requested": teacher_steps,
        "teacher_steps_actual": int(getattr(teacher_result.model, "num_timesteps", teacher_steps)),
        "teacher_checkpoint": str(getattr(teacher_result, "checkpoint", out / "teacher_policy.zip")),
        "teacher_weights_changed": _changed(teacher_before, teacher_result.model),
        "teacher_rollouts": rollouts,
        "layout_sampler": "0.5 fresh valid rows / 0.5 actually teacher-scored layouts",
        "student_steps_requested": student_steps,
        "student_steps_actual": int(getattr(student_result, "trained_timesteps", student_steps)),
        "student_total_timesteps": int(getattr(student_result, "total_timesteps", 0)),
        "student_checkpoint": str(getattr(student_result, "checkpoint", out / "student_policy.zip")),
        "student_weights_changed": _changed(student_before, student_result.model),
        "student_episodes": getattr(env, "episode_results", [])[first_student_episode:],
    }
    (out / "round_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

