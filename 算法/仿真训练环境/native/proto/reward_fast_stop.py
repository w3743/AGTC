"""M2.3: reference reward/termination adapter using isaac_field.tensor_task.fast_stop_v2_step.

The prototype env can switch to this adapter (`task_backend="fast_stop_v2"`); it is a
thin numpy<->torch boundary around the repository function so the reward/termination
math is exactly the reference implementation (torch CPU in M2).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from isaac_field.tensor_task import fast_stop_v2_step  # noqa: E402

DEFAULTS = dict(dt=0.05, max_time=60.0, goal_tolerance=0.4, goal_hold=0.3,
                goal_stop_speed=0.2, goal_stop_angular_speed=0.3,
                time_penalty=0.01, goal_progress_budget=1.0)


class FastStopV2Adapter:
    """Stateful numpy wrapper; state tensors stay small (batch x scalars)."""

    def __init__(self, num_envs: int, *, config: dict | None = None):
        self.num_envs = int(num_envs)
        self.cfg = dict(DEFAULTS)
        if config:
            self.cfg.update(config)
        self.hold = torch.zeros(self.num_envs, dtype=torch.float64)
        self.physics_failure = torch.zeros(self.num_envs, dtype=torch.bool)
        self.initial_distance = torch.zeros(self.num_envs, dtype=torch.float64)
        self.episode_time = torch.zeros(self.num_envs, dtype=torch.float64)

    def reset(self, env_ids=None, initial_distance: np.ndarray | None = None):
        ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids)
        self.hold[ids] = 0.0
        self.physics_failure[ids] = False
        self.episode_time[ids] = 0.0
        if initial_distance is not None:
            self.initial_distance[ids] = torch.as_tensor(initial_distance, dtype=torch.float64)

    def step(self, *, position_xy, old_distance, goal_xy, root_com_velocity_xy,
             body_angular_velocity, body_tilt_pitch_roll, episode_time, active_mask=None,
             boundary=None, physics_failure_event=None):
        n = self.num_envs
        dev = torch.device("cpu")
        kw = dict(
            position_xy=torch.as_tensor(np.asarray(position_xy), dtype=torch.float32, device=dev),
            old_distance=torch.as_tensor(np.asarray(old_distance), dtype=torch.float32, device=dev),
            goal_xy=torch.as_tensor(np.asarray(goal_xy), dtype=torch.float32, device=dev),
            root_com_velocity_xy=torch.as_tensor(np.asarray(root_com_velocity_xy), dtype=torch.float32, device=dev),
            body_angular_velocity=torch.as_tensor(np.asarray(body_angular_velocity), dtype=torch.float32, device=dev),
            body_tilt_pitch_roll=torch.as_tensor(np.asarray(body_tilt_pitch_roll), dtype=torch.float32, device=dev),
            boundary=torch.as_tensor(np.zeros(n) if boundary is None else boundary, dtype=torch.bool, device=dev),
            physics_failure_event=torch.as_tensor(
                np.zeros(n) if physics_failure_event is None else physics_failure_event, dtype=torch.bool, device=dev),
            previous_physics_failure=self.physics_failure.clone(),
            episode_time=self.episode_time.clone(),
            hold_elapsed=self.hold.clone(),
            initial_goal_distance=self.initial_distance.clone(),
            active_mask=torch.as_tensor(np.ones(n) if active_mask is None else active_mask,
                                        dtype=torch.bool, device=dev),
            **{k: v for k, v in self.cfg.items()},
        )
        out = fast_stop_v2_step(**kw)
        self.hold = out["next_hold"].clone()
        self.physics_failure = out["next_physics_failure"].clone()
        self.episode_time = out["next_time"].clone()
        return {
            "rewards": out["rewards"].numpy(),
            "components": {k: v.numpy() for k, v in out["components"].items()},
            "done": out["done"].numpy(),
            "terminated": out["terminated"].numpy(),
            "truncated": out["truncated"].numpy(),
            "reason_codes": out["reason_codes"].numpy(),
            "next_time": out["next_time"].numpy(),
            "next_hold": out["next_hold"].numpy(),
            "next_physics_failure": out["next_physics_failure"].numpy(),
        }

    def call_direct(self, **kwargs):
        """Same call without the adapter state cache (for element-wise verification)."""
        n = self.num_envs
        dev = torch.device("cpu")
        return fast_stop_v2_step(
            position_xy=torch.as_tensor(np.asarray(kwargs["position_xy"]), dtype=torch.float32),
            old_distance=torch.as_tensor(np.asarray(kwargs["old_distance"]), dtype=torch.float32),
            goal_xy=torch.as_tensor(np.asarray(kwargs["goal_xy"]), dtype=torch.float32),
            root_com_velocity_xy=torch.as_tensor(np.asarray(kwargs["root_com_velocity_xy"]), dtype=torch.float32),
            body_angular_velocity=torch.as_tensor(np.asarray(kwargs["body_angular_velocity"]), dtype=torch.float32),
            body_tilt_pitch_roll=torch.as_tensor(np.asarray(kwargs["body_tilt_pitch_roll"]), dtype=torch.float32),
            boundary=torch.as_tensor(np.zeros(n), dtype=torch.bool),
            physics_failure_event=torch.as_tensor(np.zeros(n), dtype=torch.bool),
            previous_physics_failure=torch.as_tensor(np.asarray(kwargs["previous_physics_failure"]), dtype=torch.bool),
            episode_time=torch.as_tensor(np.asarray(kwargs["episode_time"]), dtype=torch.float64),
            hold_elapsed=torch.as_tensor(np.asarray(kwargs["hold_elapsed"]), dtype=torch.float64),
            initial_goal_distance=torch.as_tensor(np.asarray(kwargs["initial_goal_distance"]), dtype=torch.float64),
            active_mask=torch.as_tensor(np.ones(n), dtype=torch.bool),
            **{k: v for k, v in self.cfg.items()},
        )


def pitch_roll_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Return [pitch, roll] in radians (same convention as task.body_pose)."""
    w, x, y, z = q
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return np.array([pitch, roll], dtype=np.float64)
