"""Small staged FieldSim curriculum for the recurrent field navigation policy.

Task sampling is confined to this training wrapper. It uses scene truth only to
choose reset start/goal locations; observations passed to the policy retain the
unchanged FieldEnv contract.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

import gymnasium as gym
import numpy as np

from field_sim.env import Config, FieldEnv, wrap
from field_sim.policy import make_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "field_policy_improvement"
VALIDATION_SEEDS = (20007, 20008, 20009)


def _json_value(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def config_for_stage(stage: str, *, size: float = 100.0, max_time: float = 20.0,
                     dt: float = 0.05, lidar_rays: int = 400,
                     depth_width: int = 88, depth_height: int = 160) -> Config:
    """Return the shared observation/action configuration for one curriculum phase."""
    if stage not in ("easy", "turns", "field"):
        raise ValueError("stage must be easy, turns, or field")
    field = stage == "field"
    return Config(
        size=size,
        plants=400 if field else 0,
        terrain_enabled=field,
        validate_scene=field,
        max_time=max_time,
        dt=dt,
        lidar_rays=lidar_rays,
        depth_width=depth_width,
        depth_height=depth_height,
        progress_weight=1.0,
        time_penalty=0.02,
    )


def _route_clearance(env: FieldEnv, start: np.ndarray, goal: np.ndarray) -> float:
    """Truth-only straight-corridor clearance used to bias easy field episodes."""
    points = []
    radii = []
    if len(env.scene.xy):
        points.append(np.asarray(env.scene.xy, dtype=np.float64))
        radii.append(np.asarray(env.scene.radius, dtype=np.float64))
    terrain = env.scene.terrain
    if terrain is not None and len(terrain.rock_xy):
        # Include traversable rocks here too: an easy sample should be visibly
        # and physically clear, even though the simulator permits small rocks.
        points.append(np.asarray(terrain.rock_xy, dtype=np.float64))
        radii.append(np.asarray(terrain.rock_radius, dtype=np.float64))
    if not points:
        return math.inf
    centers = np.concatenate(points, axis=0)
    obstacles = np.concatenate(radii, axis=0)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    delta = goal - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-12:
        distances = np.linalg.norm(centers - start, axis=1)
    else:
        along = np.clip(((centers - start) @ delta) / denominator, 0.0, 1.0)
        nearest = start[None, :] + along[:, None] * delta[None, :]
        distances = np.linalg.norm(centers - nearest, axis=1)
    footprint = math.hypot(env.cfg.robot_length, env.cfg.robot_width) / 2 + 0.05
    return float(np.min(distances - obstacles - footprint))


class NavigationCurriculum(gym.Wrapper):
    """Reset-time start/goal sampler; never adds privileged data to observations."""

    def __init__(self, env: FieldEnv, *, stage: str = "easy", seed: int = 7,
                 goal_min: float | None = None, goal_max: float | None = None,
                 goal_angle_deg: float | None = None):
        super().__init__(env)
        if stage not in ("easy", "turns", "field"):
            raise ValueError("stage must be easy, turns, or field")
        self.stage = stage
        default_min, default_max = (0.8, 2.5) if stage == "easy" else (1.0, 5.0)
        default_angle = 45.0 if stage == "easy" else 180.0
        self.goal_min = default_min if goal_min is None else float(goal_min)
        self.goal_max = default_max if goal_max is None else float(goal_max)
        self.goal_angle_deg = default_angle if goal_angle_deg is None else float(goal_angle_deg)
        if (not math.isfinite(self.goal_min) or self.goal_min <= 0
                or not math.isfinite(self.goal_max) or self.goal_max < self.goal_min):
            raise ValueError("goal range must be finite, positive, and ordered")
        if (not math.isfinite(self.goal_angle_deg)
                or not 0.0 <= self.goal_angle_deg <= 180.0):
            raise ValueError("goal_angle_deg must be finite and in [0, 180]")
        self.task_rng = np.random.default_rng(seed)
        self.reset_count = 0
        self.last_task: dict[str, Any] | None = None

    def _sample_start(self) -> np.ndarray:
        size = float(self.unwrapped.cfg.size)
        default_start = np.array([2.0, 2.0, math.pi / 4], dtype=np.float64)
        if self.task_rng.random() < 0.2:
            return default_start
        margin = min(8.0, size * 0.08)
        lo, hi = margin, size - margin
        yaw = float(self.task_rng.uniform(-math.pi, math.pi))
        return np.array([self.task_rng.uniform(lo, hi),
                         self.task_rng.uniform(lo, hi), yaw], dtype=np.float64)

    def _goal_range(self) -> tuple[float, float]:
        return self.goal_min, self.goal_max

    def _sample_goal(self, start: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        env = self.unwrapped
        min_distance, max_distance = self._goal_range()
        start_xy = np.asarray(start[:2], dtype=np.float64)
        yaw = float(start[2])
        want_clear_route = self.stage == "field" and self.task_rng.random() < 0.8
        angle_limit = math.radians(self.goal_angle_deg)
        for _ in range(5000):
            distance = float(self.task_rng.uniform(min_distance, max_distance))
            offset = float(self.task_rng.uniform(-angle_limit, angle_limit))
            angle = yaw + offset
            goal = start_xy + distance * np.array([math.cos(angle), math.sin(angle)])
            if not env.scene.navigable(goal):
                continue
            clearance = _route_clearance(env, start_xy, goal)
            if want_clear_route and clearance < 0.05:
                continue
            return goal, {
                "distance_m": float(np.linalg.norm(goal - start_xy)),
                "heading_offset_rad": float(wrap(math.atan2(goal[1] - start_xy[1],
                                                              goal[0] - start_xy[0]) - yaw)),
                "route_type": "clear_straight" if want_clear_route else "general_navigable",
                "corridor_clearance_m": clearance if math.isfinite(clearance) else None,
            }
        if want_clear_route:
            # Dense fields can have no short clear route for a sampled start.
            # Preserve the 80/20 intent where possible, but don't fail resets.
            # A local retry with the same angle/distance contract, without the
            # extra corridor constraint, is a valid navigable sample.
            for _ in range(5000):
                distance = float(self.task_rng.uniform(min_distance, max_distance))
                offset = float(self.task_rng.uniform(-angle_limit, angle_limit))
                angle = yaw + offset
                goal = start_xy + distance * np.array([math.cos(angle), math.sin(angle)])
                if env.scene.navigable(goal):
                    clearance = _route_clearance(env, start_xy, goal)
                    return goal, {"distance_m": float(np.linalg.norm(goal - start_xy)),
                                  "heading_offset_rad": float(wrap(math.atan2(goal[1] - start_xy[1],
                                                                              goal[0] - start_xy[0]) - yaw)),
                                  "route_type": "general_navigable_fallback",
                                  "corridor_clearance_m": clearance if math.isfinite(clearance) else None,
                                  "clear_route_fallback": True}
        raise RuntimeError(f"could not sample a navigable {self.stage} goal after 5000 attempts")

    def reset(self, *, seed=None, options=None):
        start = self._sample_start()
        reset_options = dict(options or {})
        reset_options.update({"start": start.tolist(), "mode": "goto"})
        observation, info = self.env.reset(seed=seed, options=reset_options)
        goal, task = self._sample_goal(start)
        self.unwrapped.set_goal(goal)
        self.last_task = {"stage": self.stage, "start": start.tolist(),
                          "goal": goal.tolist(), "goal_min_m": self.goal_min,
                          "goal_max_m": self.goal_max,
                          "goal_angle_deg": self.goal_angle_deg, **task}
        self.reset_count += 1
        # The goal vector is the only policy-facing signal for this task.
        return self.unwrapped._observation(), self.unwrapped._info()


def _make_training_env(cfg: Config, *, n_env: int, seed: int, stage: str,
                       goal_min: float | None = None, goal_max: float | None = None,
                       goal_angle_deg: float | None = None,
                       episode_steps: int | None = None,
                       vec_backend: str = "dummy", distance_cost: float = 0.0):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from gymnasium.wrappers import TimeLimit, TransformReward

    if n_env < 1:
        raise ValueError("n_env must be positive")
    if episode_steps is not None and episode_steps < 1:
        raise ValueError("episode_steps must be positive when provided")
    if vec_backend not in ("dummy", "subproc"):
        raise ValueError("vec_backend must be dummy or subproc")
    if not math.isfinite(float(distance_cost)) or distance_cost < 0:
        raise ValueError("distance_cost must be finite and non-negative")
    factories = []
    for rank in range(n_env):
        task_seed = int(seed + 1009 * (rank + 1))

        def create_env(task_seed=task_seed):
            env = FieldEnv(cfg, mode="goto")
            if episode_steps is not None:
                env = TimeLimit(env, max_episode_steps=episode_steps)
            env = NavigationCurriculum(env, stage=stage, seed=task_seed,
                                       goal_min=goal_min, goal_max=goal_max,
                                       goal_angle_deg=goal_angle_deg)
            if distance_cost > 0:
                task_env = env.unwrapped
                env = TransformReward(
                    env,
                    lambda reward, task_env=task_env: float(reward)
                    - float(distance_cost) * float(task_env._info()["distance"]) * cfg.dt)
            return env

        factories.append(create_env)
    if vec_backend == "subproc":
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def _validation_score(episodes: list[dict]) -> tuple[float, float, float, float] | None:
    """Compare only complete fixed-seed validation sets, safety first."""
    if not episodes or any(not row.get("complete", False) for row in episodes):
        return None
    count = len(episodes)
    safe_arrivals = sum(bool(row["safe_success"]) for row in episodes)
    progress = float(np.mean([row["progress_fraction"] for row in episodes]))
    damage = float(np.mean([row["damage"] for row in episodes]))
    collisions = float(np.mean([row["collisions"] for row in episodes]))
    return safe_arrivals / count, progress, -damage, -collisions


def _validation_record(cfg: Config, model, *, deadline: float | None = None) -> dict:
    """Run the three fixed full-field seeds with recurrent state through terminal."""
    from simulation.evaluate_field_gazebo import _select_safe_goto_goal

    episodes = []
    for seed in VALIDATION_SEEDS:
        if deadline is not None and time.perf_counter() >= deadline:
            episodes.append({"seed": seed, "complete": False, "skipped_wall_limit": True})
            continue
        env = FieldEnv(cfg, mode="goto")
        try:
            wall_begin = time.perf_counter()
            observation, info = env.reset(seed=seed, options={"mode": "goto"})
            initial_distance = float(np.linalg.norm(env.goal - env.robot[:2]))
            goal, route = _select_safe_goto_goal(env, nominal_distance=3.0)
            env.set_goal(goal)
            observation, info = env._observation(), env._info()
            initial_distance = float(info["distance"])
            recurrent_state = None
            episode_start = True
            total_return = 0.0
            steps = 0
            timed_out = False
            normalized_actions = []
            body_motion = []
            max_steps = int(math.ceil(cfg.max_time / cfg.dt))
            for _ in range(max_steps):
                action, recurrent_state = model.predict(
                    observation, state=recurrent_state,
                    episode_start=np.asarray([episode_start], dtype=bool),
                    deterministic=True)
                episode_start = False
                action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1, 4)[0], -1.0, 1.0)
                normalized_actions.append(action.copy())
                observation, reward, terminated, truncated, info = env.step(action)
                body_motion.append((float(env.robot[3]), float(env.robot[4])))
                total_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    timed_out = True
                    break
            final_distance = float(info.get("distance", np.linalg.norm(env.goal - env.robot[:2])))
            complete = bool(terminated or truncated) if steps else False
            damage = int(info.get("damage", len(env.damaged)))
            collisions = int(info.get("collisions", env.collisions))
            success = info.get("reason") == "goal_reached"
            progress_fraction = ((initial_distance - final_distance) / max(initial_distance, 1e-9))
            action_array = np.asarray(normalized_actions, dtype=np.float64).reshape(-1, 4)
            physical_scales = np.asarray([cfg.max_speed, cfg.max_turn_rate,
                                          cfg.gimbal_yaw_rate, cfg.gimbal_pitch_rate],
                                         dtype=np.float64)
            physical_actions = action_array * physical_scales[None, :]
            motion_array = np.asarray(body_motion, dtype=np.float64).reshape(-1, 2)
            episodes.append({
                "seed": seed, "complete": complete, "timed_out": timed_out,
                "mode": "goto", "steps": steps, "simulated_seconds": float(env.time),
                "initial_distance_m": initial_distance, "final_distance_m": final_distance,
                "progress_m": initial_distance - final_distance,
                "progress_fraction": float(progress_fraction),
                "goal_reached": bool(success),
                "safe_success": bool(success and damage == 0 and collisions == 0),
                "damage": damage, "collisions": collisions,
                "return": float(total_return), "reason": info.get("reason"),
                "normalized_action_mean_4d": action_array.mean(axis=0).tolist(),
                "normalized_action_std_4d": action_array.std(axis=0).tolist(),
                "physical_control_units": ["m/s", "rad/s", "rad/s", "rad/s"],
                "physical_control_mean_4d": physical_actions.mean(axis=0).tolist(),
                "physical_control_std_4d": physical_actions.std(axis=0).tolist(),
                "mean_abs_v_mps": float(np.mean(np.abs(motion_array[:, 0]))),
                "final_v_mps": float(env.robot[3]),
                "final_w_radps": float(env.robot[4]),
                "route": route, "wall_seconds": time.perf_counter() - wall_begin,
            })
        finally:
            env.close()
    score = _validation_score(episodes)
    return {"seeds": list(VALIDATION_SEEDS), "complete": score is not None,
            "score": list(score) if score is not None else None,
            "episodes": episodes,
            "score_order": ["safe_arrival_rate", "mean_progress_fraction",
                            "negative_mean_damage", "negative_mean_collisions"]}


def _copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _make_or_load_model(vec_env, args):
    """Construct a fresh configured model or resume without rewriting its encoder gain."""
    if args.checkpoint is None:
        return make_model(vec_env, seed=args.seed, n_steps=args.n_steps,
                          verbose=args.verbose, goal_gain=args.goal_gain,
                          batch_size=args.batch_size, n_epochs=args.n_epochs,
                          learning_rate=args.learning_rate, gamma=args.gamma,
                          n_env=args.n_env, use_sde=args.use_sde,
                          sde_sample_freq=args.sde_sample_freq,
                          log_std_init=args.log_std_init,
                          squash_output=args.squash_output,
                          target_kl=args.target_kl, device=args.device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO.load(str(args.checkpoint), env=vec_env, device=args.device)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Improve field navigation with staged pure-FieldSim curriculum.")
    parser.add_argument("--stage", choices=("easy", "turns", "field"), default="easy")
    parser.add_argument("--steps", type=int, default=32768,
                        help="additional training timesteps, excluding validation")
    parser.add_argument("--chunk-steps", type=int, default=8192)
    parser.add_argument("--wall-seconds", type=float, default=240.0,
                        help="hard stop checked inside each training chunk")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="optional RecurrentPPO checkpoint to continue")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-env", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--goal-gain", type=float, default=12.5)
    parser.add_argument("--goal-min", type=float,
                        help="optional curriculum goal-distance minimum in meters")
    parser.add_argument("--goal-max", type=float,
                        help="optional curriculum goal-distance maximum in meters")
    parser.add_argument("--goal-angle-deg", type=float,
                        help="optional maximum goal heading offset from the start yaw")
    parser.add_argument("--episode-steps", type=int,
                        help="optional Gymnasium TimeLimit per training episode; Config.max_time is unchanged")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="PyTorch device for fresh or resumed RecurrentPPO model")
    parser.add_argument("--vec-backend", choices=("dummy", "subproc"), default="dummy",
                        help="SB3 vector environment implementation")
    parser.add_argument("--distance-cost", type=float, default=0.0,
                        help="training-only dense reward coefficient: subtract coefficient * goal distance * dt")
    parser.add_argument("--stop-on-two-zero-success-negative-progress", action="store_true",
                        help="stop after two consecutive chunks with zero train successes and negative validation progress")
    parser.add_argument("--use-sde", action="store_true",
                        help="use SB3 generalized state-dependent exploration")
    parser.add_argument("--sde-sample-freq", type=int, default=-1,
                        help="resample gSDE noise every N steps; -1 means once per rollout")
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--squash-output", action="store_true",
                        help="squash continuous policy output to action bounds")
    parser.add_argument("--target-kl", type=float,
                        help="optional PPO target KL stop threshold")
    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--lidar-rays", type=int, default=400)
    parser.add_argument("--depth-width", type=int, default=88)
    parser.add_argument("--depth-height", type=int, default=160)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.chunk_steps <= 0 or args.wall_seconds <= 0:
        parser.error("steps, chunk-steps, and wall-seconds must be positive")
    if args.n_env < 1 or args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        parser.error("n-env, n-steps, batch-size, and n-epochs must be positive")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.gamma) or not 0 < args.gamma <= 1
            or not math.isfinite(args.goal_gain) or args.goal_gain <= 0):
        parser.error("learning-rate/goal-gain must be positive and gamma in (0, 1]")
    if (args.sde_sample_freq == 0 or args.sde_sample_freq < -1
            or not math.isfinite(args.log_std_init)):
        parser.error("sde-sample-freq must be -1 or positive and log-std-init must be finite")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0):
        parser.error("target-kl must be finite and positive")
    if (not math.isfinite(args.size) or args.size < 6
            or not math.isfinite(args.max_time) or args.max_time <= 0
            or not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("size >= 6 and max-time/dt > 0 required")
    if args.lidar_rays < 8 or args.depth_width < 2 or args.depth_height < 2:
        parser.error("sensor dimensions are too small")
    if args.goal_min is not None and (not math.isfinite(args.goal_min) or args.goal_min <= 0):
        parser.error("goal-min must be finite and positive")
    if args.goal_max is not None and (not math.isfinite(args.goal_max) or args.goal_max <= 0):
        parser.error("goal-max must be finite and positive")
    default_goal_min, default_goal_max = (0.8, 2.5) if args.stage == "easy" else (1.0, 5.0)
    effective_goal_min = default_goal_min if args.goal_min is None else args.goal_min
    effective_goal_max = default_goal_max if args.goal_max is None else args.goal_max
    if effective_goal_max < effective_goal_min:
        parser.error("goal-max must be greater than or equal to goal-min")
    if (args.goal_angle_deg is not None
            and (not math.isfinite(args.goal_angle_deg)
                 or not 0 <= args.goal_angle_deg <= 180)):
        parser.error("goal-angle-deg must be finite and in [0, 180]")
    if args.episode_steps is not None and args.episode_steps < 1:
        parser.error("episode-steps must be positive")
    if not math.isfinite(args.distance_cost) or args.distance_cost < 0:
        parser.error("distance-cost must be finite and non-negative")
    if args.output is None:
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        args.output = DEFAULT_OUTPUT_ROOT / stamp
    args.output = args.output.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    return args


def train(args) -> dict:
    """Run chunked training, validation, checkpointing, and summary emission."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    class ChunkCallback(BaseCallback):
        def __init__(self, deadline: float, verbose=0):
            super().__init__(verbose=verbose)
            self.deadline = deadline
            self.episode_rows = []
            self.timed_out = False

        def _on_step(self):
            infos = self.locals.get("infos", ())
            dones = self.locals.get("dones", ())
            for done, info in zip(dones, infos):
                if done and isinstance(info, dict):
                    self.episode_rows.append({
                        "success": bool(info.get("is_success", False)),
                        "reason": info.get("reason"),
                        "damage": int(info.get("damage", 0)),
                        "collisions": int(info.get("collisions", 0)),
                        "distance_m": float(info.get("distance", 0.0)),
                    })
            self.timed_out = time.perf_counter() >= self.deadline
            return not self.timed_out

    args.output.mkdir(parents=True, exist_ok=False)
    cfg = config_for_stage(args.stage, size=args.size, max_time=args.max_time,
                           dt=args.dt, lidar_rays=args.lidar_rays,
                           depth_width=args.depth_width, depth_height=args.depth_height)
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    vec_env = None
    model = None
    summary = {
        "schema": "field-policy-curriculum-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "config": cfg.__dict__.copy(),
        "task_distribution": {
            "goal_min_m_requested": args.goal_min,
            "goal_max_m_requested": args.goal_max,
            "goal_angle_deg_requested": args.goal_angle_deg,
            "goal_min_m_effective": (0.8 if args.stage == "easy" else 1.0)
                                    if args.goal_min is None else args.goal_min,
            "goal_max_m_effective": (2.5 if args.stage == "easy" else 5.0)
                                    if args.goal_max is None else args.goal_max,
            "goal_angle_deg_effective": (45.0 if args.stage == "easy" else 180.0)
                                        if args.goal_angle_deg is None else args.goal_angle_deg,
            "episode_steps": args.episode_steps,
            "episode_time_limit_seconds": (args.episode_steps * args.dt
                                           if args.episode_steps is not None else None),
            "config_max_time_seconds": cfg.max_time,
        },
        "training_reward_transform": {
            "distance_cost": args.distance_cost,
            "formula": ("r' = r - coefficient * ||goal - robot_xy|| * cfg.dt"
                        if args.distance_cost > 0 else None),
            "scope": "training environments only; fixed full-field validation uses unmodified FieldEnv reward",
        },
        "device_requested": args.device,
        "vec_backend_requested": args.vec_backend,
        "stop_on_two_zero_success_negative_progress": args.stop_on_two_zero_success_negative_progress,
        "training_parameters_requested": {
            "steps": args.steps, "chunk_steps": args.chunk_steps,
            "wall_seconds": args.wall_seconds, "n_env": args.n_env,
            "n_steps": args.n_steps, "batch_size": args.batch_size,
            "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
            "gamma": args.gamma, "goal_gain": args.goal_gain, "seed": args.seed,
            "goal_min_m": args.goal_min, "goal_max_m": args.goal_max,
            "goal_angle_deg": args.goal_angle_deg,
            "episode_steps": args.episode_steps,
            "device": args.device,
            "vec_backend": args.vec_backend,
            "distance_cost": args.distance_cost,
            "use_sde": args.use_sde, "sde_sample_freq": args.sde_sample_freq,
            "log_std_init": args.log_std_init, "squash_output": args.squash_output,
            "target_kl": args.target_kl,
        },
        "checkpoint_source": str(args.checkpoint) if args.checkpoint else None,
        "validation_seeds": list(VALIDATION_SEEDS),
        "validation_log": [], "chunks": [], "best_checkpoint": None,
        "training_completed": False, "timed_out": False,
        "limitation": "Short curriculum optimization and fixed-seed validation do not establish real-world competence.",
    }
    latest_path = args.output / "checkpoint.zip"
    best_path = args.output / "best_checkpoint.zip"
    best_score = None
    extra_steps = 0
    try:
        vec_env = _make_training_env(
            cfg, n_env=args.n_env, seed=args.seed, stage=args.stage,
            goal_min=args.goal_min, goal_max=args.goal_max,
            goal_angle_deg=args.goal_angle_deg, episode_steps=args.episode_steps,
            vec_backend=args.vec_backend, distance_cost=args.distance_cost)
        model = _make_or_load_model(vec_env, args)
        torch.set_num_threads(1)
        extractor = model.policy.features_extractor
        effective_gain = float(getattr(extractor, "goal_gain", 1.0))
        summary["effective_goal_gain"] = effective_gain
        summary["device_effective"] = str(model.device)
        summary["vec_backend_effective"] = type(vec_env).__name__
        summary["goal_gain_note"] = (
            "Loaded checkpoint feature semantics are preserved; requested goal_gain is not applied during resume."
            if args.checkpoint else "Configured when the new policy was constructed.")
        summary["effective_model_hyperparameters"] = {
            "n_steps": int(model.n_steps), "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs), "gamma": float(model.gamma),
            "learning_rate": float(model.learning_rate), "num_timesteps_at_start": int(model.num_timesteps),
            "use_sde": bool(model.use_sde),
            "sde_sample_freq": int(model.sde_sample_freq),
            "target_kl": model.target_kl,
        }
        effective_policy_kwargs = getattr(model, "policy_kwargs", {}) or {}
        summary["effective_exploration"] = {
            "log_std_init": float(effective_policy_kwargs.get("log_std_init", 0.0)),
            "squash_output": bool(effective_policy_kwargs.get("squash_output", False)),
        }
        summary["exploration_resume_note"] = (
            "Loaded exploration settings are preserved; requested CLI exploration options do not mutate a resumed checkpoint."
            if args.checkpoint else "Exploration settings were applied during fresh model construction.")
        summary["observation_shapes"] = {
            key: list(space.shape) for key, space in vec_env.observation_space.spaces.items()}
        summary["action_shape"] = list(vec_env.action_space.shape)

        model.save(latest_path)
        initial_path = args.output / "initial_checkpoint.zip"
        _copy_checkpoint(latest_path, initial_path)
        summary["initial_checkpoint"] = str(initial_path.resolve())
        summary["initial_checkpoint_note"] = "Untrained or resumed starting weights; saved for comparison, not used as a validation claim."
        _write_json(args.output / "training_summary.json", summary)

        while extra_steps < args.steps and time.perf_counter() < deadline:
            requested = min(args.chunk_steps, args.steps - extra_steps)
            chunk_begin = time.perf_counter()
            start_steps = int(model.num_timesteps)
            callback = ChunkCallback(deadline)
            model.learn(total_timesteps=requested, callback=callback,
                        reset_num_timesteps=False, progress_bar=False)
            end_steps = int(model.num_timesteps)
            actual = max(0, end_steps - start_steps)
            extra_steps += actual
            training_success_rate = (float(np.mean([row["success"] for row in callback.episode_rows]))
                                     if callback.episode_rows else None)
            chunk = {
                "index": len(summary["chunks"]) + 1,
                "requested_steps": requested,
                "actual_steps": actual,
                "cumulative_additional_steps": extra_steps,
                "wall_seconds": time.perf_counter() - chunk_begin,
                "training_completed_episodes": len(callback.episode_rows),
                "training_success_rate": training_success_rate,
                "training_episode_rows": callback.episode_rows,
                "timed_out": callback.timed_out,
            }
            model.save(latest_path)
            chunk_validation = _validation_record(
                config_for_stage("field", size=args.size, max_time=20.0, dt=args.dt,
                                 lidar_rays=args.lidar_rays, depth_width=args.depth_width,
                                 depth_height=args.depth_height),
                model, deadline=deadline)
            chunk["validation"] = chunk_validation
            if chunk_validation["score"] is not None:
                score = tuple(float(value) for value in chunk_validation["score"])
                if best_score is None or score > best_score:
                    best_score = score
                    _copy_checkpoint(latest_path, best_path)
                    summary["best_checkpoint"] = str(best_path.resolve())
                    summary["best_validation_score"] = list(best_score)
            summary["chunks"].append(chunk)
            summary["validation_log"].append({"kind": "chunk", "chunk": chunk["index"], **chunk_validation})
            elapsed = time.perf_counter() - started
            successes = sum(row["success"] for row in callback.episode_rows)
            rate_text = (f"{successes}/{len(callback.episode_rows)}"
                         if callback.episode_rows else "n/a")
            print(f"[chunk {chunk['index']}] actual_steps={actual} cumulative={extra_steps}/{args.steps} "
                  f"training_success={rate_text} elapsed={elapsed:.1f}s", flush=True)
            should_stop_early = False
            if args.stop_on_two_zero_success_negative_progress and len(summary["chunks"]) >= 2:
                recent = summary["chunks"][-2:]
                zero_train_success = all(
                    row["training_completed_episodes"] > 0
                    and not any(episode["success"] for episode in row["training_episode_rows"])
                    for row in recent)
                negative_validation_progress = all(
                    row["validation"].get("complete")
                    and float(np.mean([episode["progress_m"]
                                       for episode in row["validation"]["episodes"]])) < 0
                    for row in recent)
                if zero_train_success and negative_validation_progress:
                    should_stop_early = True
                    summary["stopped_early"] = True
                    summary["stop_reason"] = (
                        "Two consecutive chunks had zero training successes and negative mean fixed full-field validation progress.")
            _write_json(args.output / "training_summary.json", summary)
            if should_stop_early:
                break
            if callback.timed_out or actual == 0:
                break

        summary["actual_additional_steps"] = extra_steps
        summary["model_num_timesteps"] = int(model.num_timesteps)
        summary["timed_out"] = time.perf_counter() >= deadline
        summary["training_completed"] = extra_steps >= args.steps
        summary["training_wall_seconds_including_validation"] = time.perf_counter() - started
        summary["policy_parameters_finite"] = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in model.policy.parameters())
        summary["checkpoint"] = str(latest_path.resolve())
        summary["validation_claim"] = False
        summary["training_claim"] = False
        _write_json(args.output / "training_summary.json", summary)
        return summary
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary["actual_additional_steps"] = extra_steps
        summary["training_wall_seconds_including_validation"] = time.perf_counter() - started
        if model is not None:
            try:
                model.save(latest_path)
                summary["checkpoint"] = str(latest_path.resolve())
            except Exception as save_exc:
                summary["checkpoint_save_error"] = f"{type(save_exc).__name__}: {save_exc}"
        _write_json(args.output / "training_summary.json", summary)
        raise
    finally:
        if vec_env is not None:
            vec_env.close()


def main(argv=None) -> int:
    args = _parse_args(argv)
    summary = train(args)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved under: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
