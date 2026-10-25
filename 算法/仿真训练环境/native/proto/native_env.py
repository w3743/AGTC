"""Gymnasium interface for the M0 native-stack throughput prototype.

Batched, vectorised environment: num_envs (default 64) MuJoCo CPU worlds stepped
together at 20 Hz policy rate (dt=1 ms x decimation=50). The observation dict keys
and shapes match isaac_field's policy_contract exactly:
    lidar (n,1,400), depth (n,1,160,88), detections (n,16,8), state (n,33), feedback (n,37)
Actions: (n,4) in [-1,1] = left track speed, right track speed, gimbal yaw rate,
gimbal pitch rate (isaac_field/action_contract.py).

Task for M0 is a deliberately simplified goto task (progress + time penalty, goal
disks, flip/time terminations). It is NOT the training task and must not be used
for capability conclusions (see observation_contract.json).
"""
from __future__ import annotations

import math

import gymnasium as gym
import json
import numpy as np
from gymnasium import spaces

from observation_provider import (DEPTH_H, DEPTH_W, FEEDBACK_DIM, LIDAR_RAYS,
                                  MAX_DETECTIONS, STATE_DIM, ObservationProvider)
from physics_backend import MAX_TRACK_SPEED, MujocoCPUBackend

GOAL_TOLERANCE = 0.4
GOAL_HOLD_S = 0.3
GOAL_STOP_LINEAR = 0.2
GOAL_STOP_ANGULAR = 0.3
FLIP_COS_LIMIT = math.cos(1.2)  # isaac_field hard tilt termination 1.2 rad


class NativeFieldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = 64, *, seed: int = 0, field_size: float = 100.0,
                 dt: float = 1e-3, decimation: int = 50, episode_length_s: float = 60.0,
                 reward_progress_weight: float = 0.15, reward_time_penalty: float = 0.01,
                 physics: str = "mujoco_cpu", lidar_backend: str = "numpy",
                 physics_cuda_graph: bool = False,
                 device: str = "cpu", task_backend: str = "simplified",
                 spawn_clearance_m: float = 0.01, feedback_backend: str = "direct",
                 feedback_noise: bool = True, depth_backend: str = "constant",
                 depth_device: str = "cpu", depth_noise: bool = True,
                 camera_backend: str = "constant", camera_noise: bool = True,
                 camera_device: str = "cpu", auto_reset: bool = True,
                 profile: bool = False):
        super().__init__()
        self.num_envs = int(num_envs)
        self.field_size = float(field_size)
        self.dt = float(dt)
        self.decimation = int(decimation)
        self.policy_dt = self.dt * self.decimation
        if abs(self.policy_dt - 0.05) > 1e-9:
            raise ValueError(f"sim_dt*decimation must equal the 0.05 s policy step, got {self.policy_dt}")
        self.experimental_physics = abs(self.dt - 1e-3) > 1e-12 or self.decimation != 50
        self.episode_length_s = float(episode_length_s)
        self.reward_progress_weight = float(reward_progress_weight)
        self.reward_time_penalty = float(reward_time_penalty)
        self.spawn_clearance_m = float(spawn_clearance_m)
        # evaluators disable auto-reset to inspect terminal states; training keeps True
        self.auto_reset = bool(auto_reset)
        # non-finite state/action diagnostics (kept small; consumed by train_wsl for failure.json)
        self.nonfinite_events = []
        self.nonfinite_count = 0
        self._last_nonfinite = np.zeros(self.num_envs, dtype=bool)
        self._last_action_nonfinite = np.zeros(self.num_envs, dtype=bool)
        if task_backend not in ("simplified", "fast_stop_v2"):
            raise ValueError(f"unknown task_backend {task_backend!r}")
        self.task_backend = task_backend
        self._fast_stop = None
        if physics_cuda_graph and physics != "newton_mujoco_warp":
            raise ValueError("physics_cuda_graph requires newton_mujoco_warp")
        if physics == "mujoco_cpu":
            self.backend = MujocoCPUBackend(nworld=self.num_envs, dt=self.dt,
                                            decimation=self.decimation, seed=seed)
        elif physics == "newton_xpbd":
            from newton_backend import NewtonXPBDGBackend
            self.backend = NewtonXPBDGBackend(nworld=self.num_envs, dt=self.dt,
                                              decimation=self.decimation, seed=seed,
                                              device=device if device != "cpu" else "cpu")
        elif physics == "newton_mujoco_warp":
            from newton_mujoco_backend import NewtonMuJoCoWarpBackend
            self.backend = NewtonMuJoCoWarpBackend(
                nworld=self.num_envs, dt=self.dt, decimation=self.decimation,
                seed=seed, device=device, use_cuda_graph=physics_cuda_graph)
        else:
            raise ValueError(f"unknown physics backend {physics!r}")
        self.physics_name = physics
        if task_backend == "fast_stop_v2":
            from reward_fast_stop import FastStopV2Adapter
            self._fast_stop = FastStopV2Adapter(self.num_envs, config={
                "dt": self.policy_dt, "max_time": self.episode_length_s,
                "goal_tolerance": GOAL_TOLERANCE, "goal_hold": GOAL_HOLD_S,
                "goal_stop_speed": GOAL_STOP_LINEAR, "goal_stop_angular_speed": GOAL_STOP_ANGULAR,
                "time_penalty": self.reward_time_penalty, "goal_progress_budget": 1.0,
            })
        self.provider = ObservationProvider(self.num_envs, field_size=self.field_size, seed=seed,
                                           lidar_backend=lidar_backend,
                                           feedback_backend=feedback_backend,
                                           feedback_noise=feedback_noise,
                                           depth_backend=depth_backend,
                                           depth_device=depth_device,
                                           depth_noise=depth_noise,
                                           camera_backend=camera_backend,
                                           camera_noise=camera_noise,
                                           camera_device=camera_device,
                                           profile=profile)
        self.rng = np.random.default_rng(seed)
        obs_shapes = self.provider.observation_shapes()
        self.single_observation_space = spaces.Dict({
            "lidar": spaces.Box(0.0, 1.0, obs_shapes["lidar"], np.float32),
            "depth": spaces.Box(-1.0, 1.0, obs_shapes["depth"], np.float32),
            "detections": spaces.Box(-1.0, 1.0, obs_shapes["detections"], np.float32),
            "state": spaces.Box(-1.0, 1.0, obs_shapes["state"], np.float32),
            "feedback": spaces.Box(-np.inf, np.inf, obs_shapes["feedback"], np.float32),
        })
        self.observation_space = spaces.Dict({
            k: spaces.Box(-np.inf if k == "feedback" else (0.0 if k == "lidar" else -1.0),
                          np.inf if k == "feedback" else 1.0,
                          (self.num_envs,) + obs_shapes[k], np.float32)
            for k in obs_shapes
        })
        self.single_action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.num_envs, 4), np.float32)
        self.step_count = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_time = np.zeros(self.num_envs)
        self.hold_time = np.zeros(self.num_envs)
        self.goal_xy = np.zeros((self.num_envs, 2))
        self.prev_distance = np.zeros(self.num_envs)
        self.last_actions = np.zeros((self.num_envs, 4))
        self._last_read = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    def _sample_start_goal(self, ids):
        n = len(ids)
        margin = 5.0
        start = self.rng.uniform(margin, self.field_size - margin, (n, 2))
        yaw = self.rng.uniform(-math.pi, math.pi, n)
        goal = self.rng.uniform(margin, self.field_size - margin, (n, 2))
        for _ in range(20):
            far = np.linalg.norm(goal - start, axis=1) > 8.0
            if far.all():
                break
            goal[~far] = self.rng.uniform(margin, self.field_size - margin, ((~far).sum(), 2))
        poses = np.zeros((n, 4))
        poses[:, 0:2] = start
        poses[:, 2] = yaw
        poses[:, 3] = self.spawn_clearance_m  # isaac_field env.py spawns with +1 cm clearance
        return poses, goal

    def _reset_envs(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        poses, goal = self._sample_start_goal(ids)
        self.backend.reset(env_ids=ids, poses=poses)
        self.goal_xy[ids] = goal
        self.provider.reset(env_ids=ids, goal_xy=goal)
        self.step_count[ids] = 0
        self.episode_time[ids] = 0.0
        self.hold_time[ids] = 0.0
        self.last_actions[ids] = 0.0
        self.prev_distance[ids] = np.linalg.norm(goal - poses[:, 0:2], axis=1)
        if self._fast_stop is not None:
            self._fast_stop.reset(env_ids=ids, initial_distance=self.prev_distance[ids])

    def reset_with_poses(self, poses: np.ndarray, goals: np.ndarray):
        """Reset all envs to explicit reference-sampled poses (M2 reset-equivalence check).

        poses: (n,4) x, y, yaw, clearance; goals: (n,2). The poses are expected to come
        from isaac_field.task.sample_episode_start_goal and pass scene.pose_clear.
        """
        poses = np.asarray(poses, np.float64).reshape(self.num_envs, 4)
        goals = np.asarray(goals, np.float64).reshape(self.num_envs, 2)
        all_ids = np.arange(self.num_envs)
        self.backend.reset(env_ids=all_ids, poses=poses)
        self.goal_xy[all_ids] = goals
        self.provider.reset(env_ids=all_ids, goal_xy=goals)
        self.step_count[all_ids] = 0
        self.episode_time[all_ids] = 0.0
        self.hold_time[all_ids] = 0.0
        self.last_actions[all_ids] = 0.0
        self.prev_distance[all_ids] = np.linalg.norm(goals - poses[:, 0:2], axis=1)
        if self._fast_stop is not None:
            self._fast_stop.reset(env_ids=all_ids, initial_distance=self.prev_distance[all_ids])
        read = self.backend.read()
        obs = self.provider.observe(read, actions=self.last_actions)
        self._last_read = read
        return obs

    # ------------------------------------------------------------------
    _FINITE_FIELDS = ("pos", "quat_wxyz", "lin_vel", "ang_vel", "gimbal_q", "wheel_qd")

    def _nonfinite_masks(self, read, action=None):
        """Per-env non-finite masks for the state fields (and optionally the action)."""
        bad = np.zeros(self.num_envs, dtype=bool)
        detail = {}
        for name in self._FINITE_FIELDS:
            value = read.get(name)
            if value is None:
                continue
            value = np.asarray(value)
            env_bad = ~np.isfinite(value).reshape(self.num_envs, -1).all(axis=1)
            if env_bad.any():
                detail[name] = int(env_bad.sum())
            bad |= env_bad
        if action is not None:
            act_bad = ~np.isfinite(np.asarray(action)).reshape(self.num_envs, -1).all(axis=1)
            if act_bad.any():
                detail["action"] = int(act_bad.sum())
                bad |= act_bad
        return bad, detail

    def _record_nonfinite(self, read, event, *, auto_recovered):
        if len(self.nonfinite_events) < 50:
            self.nonfinite_events.append({
                "step": int(self.step_count.max()) if self.step_count.size else 0,
                "env_ids": [int(v) for v in np.flatnonzero(event["bad"])],
                "fields": event["detail"],
                "auto_recovered": bool(auto_recovered),
                "qpos_sample": np.asarray(read.get("pos", np.zeros((0, 3))))[event["bad"]].tolist(),
            })
        self.nonfinite_count += int(event["bad"].sum())
        print("NONFINITE_STATE " + json.dumps({
            "step": int(self.step_count.max()) if self.step_count.size else 0,
            "env_ids": [int(v) for v in np.flatnonzero(event["bad"])][:8],
            "fields": event["detail"], "auto_recovered": bool(auto_recovered)}))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        all_ids = np.arange(self.num_envs)
        self._reset_envs(all_ids)
        read = self.backend.read()
        bad, detail = self._nonfinite_masks(read)
        if bad.any():
            self._record_nonfinite(read, {"bad": bad, "detail": detail}, auto_recovered=False)
        obs = self.provider.observe(read, actions=self.last_actions)
        self._last_read = read
        return obs, {"reset": True}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (self.num_envs, 4):
            raise ValueError(f"action must be ({self.num_envs}, 4), got {action.shape}")
        # action non-finiteness must never reach the physics backend
        act_bad = ~np.isfinite(action).all(axis=1)
        if act_bad.any():
            self._record_nonfinite({"pos": np.zeros((self.num_envs, 3))},
                                   {"bad": act_bad, "detail": {"action": int(act_bad.sum())}},
                                   auto_recovered=True)
            action = action.copy()
            action[act_bad] = 0.0
        self.backend.step(action)
        physics_diagnostics = getattr(self.backend, "last_solver_diagnostics", None)
        if isinstance(physics_diagnostics, dict):
            physics_diagnostics = {key: value.copy() if isinstance(value, list) else value
                                   for key, value in physics_diagnostics.items()}
        read = self.backend.read()
        bad, detail = self._nonfinite_masks(read, action)
        recovered = False
        if bad.any():
            recovered = bool(self.auto_reset)
            self._record_nonfinite(read, {"bad": bad, "detail": detail}, auto_recovered=recovered)
            if self.auto_reset:
                # active training: reset the diverged envs so the run can continue
                self._reset_envs(np.flatnonzero(bad))
                read = self.backend.read()
                post_bad, post_detail = self._nonfinite_masks(read)
                if post_bad.any():
                    self._record_nonfinite(read, {"bad": post_bad, "detail": post_detail},
                                           auto_recovered=False)
        obs = self.provider.observe(read, actions=action)
        self._last_read = read
        self.step_count += 1
        self.episode_time += self.policy_dt
        self.last_actions = action.astype(np.float32)
        self._last_nonfinite = bad
        self._last_action_nonfinite = act_bad

        pos = read["pos"]
        lin_speed = np.linalg.norm(read["lin_vel"][:, :2], axis=1)
        ang_speed = np.linalg.norm(read["ang_vel"], axis=1)
        distance = np.linalg.norm(self.goal_xy - pos[:, :2], axis=1)
        progress = self.prev_distance - distance
        if self._fast_stop is not None:
            from reward_fast_stop import pitch_roll_from_quat_wxyz
            tilt = np.stack([pitch_roll_from_quat_wxyz(q) for q in read["quat_wxyz"]], axis=0)
            ref = self._fast_stop.step(
                position_xy=pos[:, :2], old_distance=self.prev_distance.copy(),
                goal_xy=self.goal_xy, root_com_velocity_xy=read["lin_vel"][:, :2],
                body_angular_velocity=read["ang_vel"], body_tilt_pitch_roll=tilt,
                episode_time=self.episode_time.copy())
            reward = ref["rewards"].astype(np.float64)
            terminated = ref["terminated"].astype(bool)
            truncated = np.zeros(self.num_envs, dtype=bool)  # fast_stop_v2: 60 s is a terminal failure
            success = ref["reason_codes"] == 1
            flipped = ref["reason_codes"] == 2
            self.prev_distance = distance
            done = ref["done"].astype(bool)
            if done.any() and self.auto_reset:
                ids = np.flatnonzero(done)
                self._reset_envs(ids)
                read = self.backend.read()
                obs = self.provider.observe(read, actions=self.last_actions)
                self._last_read = read
            info = {"success": success, "flipped": flipped, "distance": distance,
                    "lin_speed": lin_speed, "policy_step": self.step_count.copy(),
                    "reason_codes": ref["reason_codes"], "reward_components": ref["components"],
                    "nonfinite": self._last_nonfinite.copy(), "action_nonfinite": self._last_action_nonfinite.copy(),
                    "physics_solver_diagnostics": physics_diagnostics}
            return obs, reward.astype(np.float32), terminated, truncated, info
        self.prev_distance = distance
        # up-vector z from body quat -> flip / hard tilt check (1.2 rad)
        up_z = np.zeros(self.num_envs)
        for i, (w, x, y, z) in enumerate(read["quat_wxyz"]):
            up_z[i] = 1.0 - 2.0 * (x * x + y * y)
        flipped = up_z < FLIP_COS_LIMIT
        stopped = (lin_speed < GOAL_STOP_LINEAR) & (ang_speed < GOAL_STOP_ANGULAR)
        in_disk = distance <= GOAL_TOLERANCE
        self.hold_time = np.where(in_disk & stopped, self.hold_time + self.policy_dt, 0.0)
        success = self.hold_time >= GOAL_HOLD_S
        truncated = self.episode_time >= self.episode_length_s
        terminated = flipped | success

        reward = (self.reward_progress_weight * progress
                  - self.reward_time_penalty * self.policy_dt
                  + 10.0 * success.astype(np.float64)
                  - 10.0 * flipped.astype(np.float64))

        done = terminated | truncated
        if done.any() and self.auto_reset:
            ids = np.flatnonzero(done)
            self._reset_envs(ids)
            read = self.backend.read()
            obs = self.provider.observe(read, actions=self.last_actions)
            self._last_read = read
        info = {"success": success, "flipped": flipped,
                "distance": distance, "lin_speed": lin_speed,
                "policy_step": self.step_count.copy(),
                "nonfinite": self._last_nonfinite.copy(),
                "action_nonfinite": self._last_action_nonfinite.copy(),
                "physics_solver_diagnostics": physics_diagnostics}
        return obs, reward.astype(np.float32), terminated, truncated, info

    # ------------------------------------------------------------------
    def consume_nonfinite(self):
        """Return and clear the non-finite event log/counters (training failure reporting)."""
        out = {"count": int(self.nonfinite_count), "events": self.nonfinite_events,
               "last_mask": [int(v) for v in np.flatnonzero(self._last_nonfinite)],
               "sensor_frames": getattr(self.provider, "nonfinite_frames", 0)}
        self.nonfinite_events = []
        self.nonfinite_count = 0
        return out

    def consume_sensor_times(self):
        """Per-sensor seconds since the last call (None unless profile=True)."""
        return self.provider.consume_timings()

    def task_config(self) -> dict:
        """Describe the task implementation and every task-side parameter used by step/reset."""
        common = {
            "backend": self.task_backend,
            "episode_length_s": self.episode_length_s,
            "reset_sampling": {
                "field_size_m": self.field_size,
                "start_goal_margin_m": 5.0,
                "minimum_start_goal_distance_m": 8.0,
                "max_goal_resamples": 20,
                "initial_yaw_range_rad": [-math.pi, math.pi],
                "spawn_clearance_m": self.spawn_clearance_m,
            },
        }
        if self.task_backend == "fast_stop_v2":
            common["implementation"] = "isaac_field.tensor_task.fast_stop_v2_step"
            common["reward_termination"] = dict(self._fast_stop.cfg)
            common["reward_termination"]["timeout_behavior"] = "terminal_failure"
        else:
            common["implementation"] = "native_env.simplified_goto"
            common["reward_termination"] = {
                "progress_weight": self.reward_progress_weight,
                "time_penalty_per_policy_second": self.reward_time_penalty,
                "goal_tolerance_m": GOAL_TOLERANCE,
                "goal_hold_s": GOAL_HOLD_S,
                "goal_stop_linear_speed_mps": GOAL_STOP_LINEAR,
                "goal_stop_angular_speed_radps": GOAL_STOP_ANGULAR,
                "flip_tilt_limit_rad": 1.2,
                "success_reward": 10.0,
                "flip_penalty": -10.0,
                "timeout_behavior": "truncated",
            }
        return common

    def _physics_identity(self) -> dict:
        identity = dict(self.backend.identity())
        identity["gravity_mps2"] = [0.0, 0.0, -9.81]
        if self.physics_name == "mujoco_cpu":
            identity["solver"] = {
                "kind": "mujoco_timestep_newton",
                "iterations": 40,
                "line_search_iterations": 8,
                "cone": "elliptic",
                "impratio": 10.0,
            }
            identity["contact"] = {
                "ground_friction": [float(self.backend.mu), 0.005, 0.0001],
                "wheel_friction": [float(self.backend.mu), 0.005, 0.0001],
            }
        elif self.physics_name == "newton_xpbd":
            identity["solver"] = {
                "kind": "newton_xpbd",
                "iterations": 10,
                "shape_ke_values_per_world": identity["shape_material_ke_values_per_world"],
                "shape_kd_values_per_world": identity["shape_material_kd_values_per_world"],
            }
            identity["contact"] = {
                "shape_mu_values_per_world": identity["shape_material_mu_values_per_world"]
            }
        return identity

    def observation_config(self) -> dict:
        cfg = self.provider.contract(self.policy_dt)
        cfg["action_components"] = ["left_track_speed", "right_track_speed",
                                    "gimbal_yaw_rate", "gimbal_pitch_rate"]
        cfg["task_backend"] = self.task_backend
        cfg["task"] = self.task_config()
        cfg["physics"] = self._physics_identity()
        cfg["physics_dt_s"] = self.dt
        cfg["decimation"] = self.decimation
        cfg["policy_dt_s"] = self.policy_dt
        cfg["physics_experimental"] = bool(self.experimental_physics)
        cfg["physics_note"] = ("default 1 ms / decimation 50" if not self.experimental_physics
                               else "experimental configuration; results are not comparable with 1 ms/50 runs")
        cfg["profile_enabled"] = bool(self.provider.profile)
        cfg["reward"] = self.task_config()["reward_termination"]
        return cfg

    def identity(self) -> dict:
        return {"num_envs": self.num_envs, "policy_dt_s": self.policy_dt,
                "sim_dt_s": self.dt, "decimation": self.decimation,
                "experimental_physics": bool(self.experimental_physics),
                "episode_length_s": self.episode_length_s,
                "obs_shapes": self.provider.observation_shapes(),
                "action_dim": 4, "max_track_speed_mps": MAX_TRACK_SPEED,
                "task_backend": self.task_backend, "task": self.task_config(),
                "physics": self._physics_identity()}
