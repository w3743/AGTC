"""Batched Torch math for the empty-field goto fast-stop task.

This module deliberately covers only ``fast_stop_v2`` reward/termination and
the absolute-goal goto policy observation. It does not simulate plants, rocks,
damage, search, sensors, or feedback; callers supply already captured sensor
frames and the existing sensor/VQF-derived pose and feedback tensors.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

FAST_STOP_SUCCESS_REWARD = 10.0
FAST_STOP_FAILURE_REWARD = -10.0

REASON_ONGOING = 0
REASON_GOAL_REACHED = 1
REASON_OUTSIDE_OR_OVERTURNED = 2
REASON_TIME_LIMIT = 3

_SENSOR_ORDER = ("lidar", "depth", "camera")


def _require_batch(name: str, value: Tensor, tail_shape: tuple[int, ...], *, device=None) -> int:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(tail_shape) + 1 or tuple(value.shape[1:]) != tail_shape:
        raise ValueError(f"{name} must have shape [N,{','.join(map(str, tail_shape))}]")
    if value.shape[0] < 1:
        raise ValueError(f"{name} batch must not be empty")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    return int(value.shape[0])


def _require_vector(name: str, value: Tensor, batch: int, *, device, dtype=None) -> Tensor:
    _require_batch(name, value, (), device=device)
    if value.shape[0] != batch:
        raise ValueError(f"{name} batch dimension must be {batch}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}")
    return value


def _config_value(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _goal_set_progress_potential(
    initial_distance: Tensor, distance: Tensor, tolerance: float, budget: float
) -> Tensor:
    """Torch equivalent of task._goal_set_progress_potential, in float64."""
    initial_error = torch.clamp(initial_distance - tolerance, min=0.0)
    error = torch.clamp(distance - tolerance, min=0.0)
    denominator = torch.maximum(initial_error, torch.full_like(initial_error, tolerance))
    progress = (initial_error - error) / denominator
    return budget * torch.clamp(progress, min=-1.0, max=1.0)


def fast_stop_v2_step(
    *,
    position_xy: Tensor,
    old_distance: Tensor,
    goal_xy: Tensor,
    root_com_velocity_xy: Tensor,
    body_angular_velocity: Tensor,
    body_tilt_pitch_roll: Tensor,
    boundary: Tensor,
    physics_failure_event: Tensor,
    previous_physics_failure: Tensor,
    episode_time: Tensor,
    hold_elapsed: Tensor,
    initial_goal_distance: Tensor,
    active_mask: Tensor,
    dt: float,
    max_time: float,
    goal_tolerance: float,
    goal_hold: float,
    goal_stop_speed: float,
    goal_stop_angular_speed: float,
    time_penalty: float,
    goal_progress_budget: float,
) -> dict[str, Any]:
    """Compute one batch of empty-field ``fast_stop_v2`` task transitions.

    Position/velocity/tilt are engine truth for reward only. ``active_mask`` is
    false for already-finished lanes in auto-reset-disabled batches; those
    lanes report no new reward/done and retain their three persistent values.
    ``body_tilt_pitch_roll`` is ``[pitch, roll]`` in radians, and the stop gate
    uses full world XY and full body angular-velocity magnitudes.
    """
    batch = _require_batch("position_xy", position_xy, (2,))
    device = position_xy.device
    for name, value, tail in (
        ("old_distance", old_distance, ()),
        ("goal_xy", goal_xy, (2,)),
        ("root_com_velocity_xy", root_com_velocity_xy, (2,)),
        ("body_angular_velocity", body_angular_velocity, (3,)),
        ("body_tilt_pitch_roll", body_tilt_pitch_roll, (2,)),
    ):
        if _require_batch(name, value, tail, device=device) != batch:
            raise ValueError(f"{name} batch dimension must be {batch}")
    for name, value in (("episode_time", episode_time), ("hold_elapsed", hold_elapsed),
                        ("initial_goal_distance", initial_goal_distance)):
        _require_vector(name, value, batch, device=device)
    for name, value in (("boundary", boundary), ("physics_failure_event", physics_failure_event),
                        ("previous_physics_failure", previous_physics_failure),
                        ("active_mask", active_mask)):
        _require_vector(name, value, batch, device=device, dtype=torch.bool)

    dt = _config_value("dt", dt, positive=True)
    max_time = _config_value("max_time", max_time, positive=True)
    goal_tolerance = _config_value("goal_tolerance", goal_tolerance, positive=True)
    goal_hold = _config_value("goal_hold", goal_hold, positive=True)
    goal_stop_speed = _config_value("goal_stop_speed", goal_stop_speed, positive=True)
    goal_stop_angular_speed = _config_value("goal_stop_angular_speed", goal_stop_angular_speed, positive=True)
    time_penalty = _config_value("time_penalty", time_penalty, nonnegative=True)
    goal_progress_budget = _config_value("goal_progress_budget", goal_progress_budget, nonnegative=True)

    # Keep the accumulated clocks and threshold arithmetic in the same
    # precision as FieldTask's NumPy/Python-double reference.
    position = position_xy.to(dtype=torch.float64)
    goal = goal_xy.to(dtype=torch.float64)
    current_time = episode_time.to(dtype=torch.float64)
    current_hold = hold_elapsed.to(dtype=torch.float64)
    initial_distance = initial_goal_distance.to(dtype=torch.float64)
    next_time = torch.where(active_mask, current_time + dt, current_time)
    current_distance = torch.linalg.vector_norm(position - goal, dim=1)
    linear_speed = torch.linalg.vector_norm(root_com_velocity_xy.to(dtype=torch.float64), dim=1)
    angular_speed = torch.linalg.vector_norm(body_angular_velocity.to(dtype=torch.float64), dim=1)
    tilt_failure = torch.any(torch.abs(body_tilt_pitch_roll.to(dtype=torch.float64)) > 1.2, dim=1)
    next_physics_failure = (previous_physics_failure | physics_failure_event | boundary | tilt_failure)
    next_physics_failure = torch.where(active_mask, next_physics_failure, previous_physics_failure)
    physics_terminal = active_mask & next_physics_failure

    within_stop_state = (
        (current_distance <= goal_tolerance)
        & (linear_speed < goal_stop_speed)
        & (angular_speed < goal_stop_angular_speed)
    )
    advancing_hold = torch.where(within_stop_state, current_hold + dt, torch.zeros_like(current_hold))
    next_hold = torch.where(
        active_mask,
        torch.where(physics_terminal, current_hold, advancing_hold),
        current_hold,
    )
    goal_terminal = active_mask & ~physics_terminal & (next_hold + 1e-9 >= goal_hold)
    time_terminal = active_mask & ~physics_terminal & ~goal_terminal & (next_time + 1e-9 >= max_time)
    done = physics_terminal | goal_terminal | time_terminal

    reason_codes = torch.zeros((batch,), dtype=torch.int64, device=device)
    reason_codes = torch.where(physics_terminal, REASON_OUTSIDE_OR_OVERTURNED, reason_codes)
    reason_codes = torch.where(goal_terminal, REASON_GOAL_REACHED, reason_codes)
    reason_codes = torch.where(time_terminal, REASON_TIME_LIMIT, reason_codes)

    before_phi = _goal_set_progress_potential(
        initial_distance, old_distance.to(dtype=torch.float64), goal_tolerance, goal_progress_budget
    )
    after_phi = _goal_set_progress_potential(
        initial_distance, current_distance, goal_tolerance, goal_progress_budget
    )
    zero64 = torch.zeros((batch,), dtype=torch.float64, device=device)
    active64 = active_mask.to(dtype=torch.float64)
    time_component = active64 * (-time_penalty * dt)
    progress_component = active64 * (after_phi - before_phi)
    success_component = goal_terminal.to(dtype=torch.float64) * FAST_STOP_SUCCESS_REWARD
    failure_component = (physics_terminal | time_terminal).to(dtype=torch.float64) * FAST_STOP_FAILURE_REWARD
    remaining_time = torch.clamp(max_time - next_time, min=0.0)
    remaining_component = (physics_terminal | time_terminal).to(dtype=torch.float64) * (-time_penalty * remaining_time)

    components64 = {
        "time": time_component,
        "damage": zero64,
        "collision": zero64,
        "boundary": zero64,
        "goal_progress": progress_component,
        "search_detection": zero64,
        "goal_success": success_component,
        "search_timeout": zero64,
        "distance_cost_shaping": zero64,
        "outside_or_overturned": zero64,
        "terminal_failure": failure_component,
        "failure_remaining_time_cost": remaining_component,
    }
    components = {name: value.to(dtype=torch.float32) for name, value in components64.items()}
    rewards = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return {
        "rewards": rewards,
        "components": components,
        "done": done,
        "terminated": done,
        "truncated": torch.zeros_like(done),
        "reason_codes": reason_codes,
        "next_time": next_time,
        "next_hold": next_hold,
        "next_physics_failure": next_physics_failure,
    }


def _wrap_tensor(angle: Tensor) -> Tensor:
    pi = math.pi
    return torch.remainder(angle + pi, 2.0 * pi) - pi


def pack_emptyfield_goto_observation(
    *,
    lidar_ranges: Tensor,
    lidar_valid: Tensor,
    depth_values: Tensor,
    depth_valid: Tensor,
    empty_detections: Tensor,
    frame_capture_times: Tensor,
    frame_estimated_poses: Tensor,
    frame_new: Tensor,
    estimated_pose: Tensor,
    goal_xy: Tensor,
    last_actions: Tensor,
    episode_time: Tensor,
    feedback: Tensor,
    size: float,
    lidar_range: float,
    depth_range: float,
    max_time: float,
    max_detections: int = 16,
    normalized_lidar: Tensor | None = None,
    normalized_depth: Tensor | None = None,
) -> dict[str, Tensor]:
    """Build the five batched absolute-goal goto policy tensors on-device.

    Sensor order in frame metadata is fixed as lidar, depth, camera. Pose inputs
    are the already-estimated noisy UWB/VQF/gimbal states, never engine truth.
    Detections must be empty; this helper does not claim detector/search support.
    Optional normalized sensor tensors are read-only, caller-owned cache entries;
    when supplied, their exact objects are returned without recomputation or copy.
    """
    if not isinstance(lidar_ranges, Tensor) or lidar_ranges.ndim != 2:
        raise ValueError("lidar_ranges must have shape [N,R]")
    batch = _require_batch("lidar_ranges", lidar_ranges, (int(lidar_ranges.shape[1]),))
    device = lidar_ranges.device
    if lidar_ranges.shape[1] < 1:
        raise ValueError("lidar_ranges must contain at least one ray")
    rays = int(lidar_ranges.shape[1])
    if _require_batch("lidar_valid", lidar_valid, (rays,), device=device) != batch:
        raise ValueError("lidar_valid batch dimension must match lidar_ranges")
    if lidar_valid.dtype != torch.bool:
        raise TypeError("lidar_valid must be boolean")
    if not isinstance(depth_values, Tensor) or depth_values.ndim != 3 or depth_values.shape[0] != batch:
        raise ValueError("depth_values must have shape [N,H,W]")
    height, width = int(depth_values.shape[1]), int(depth_values.shape[2])
    _require_batch("depth_values", depth_values, (height, width), device=device)
    if _require_batch("depth_valid", depth_valid, (height, width), device=device) != batch:
        raise ValueError("depth_valid batch dimension must match depth_values")
    if depth_valid.dtype != torch.bool:
        raise TypeError("depth_valid must be boolean")
    if normalized_lidar is not None:
        if _require_batch("normalized_lidar", normalized_lidar, (1, rays), device=device) != batch:
            raise ValueError(f"normalized_lidar batch dimension must be {batch}")
        if normalized_lidar.dtype != torch.float32:
            raise TypeError("normalized_lidar must use dtype torch.float32")
    if normalized_depth is not None:
        if _require_batch("normalized_depth", normalized_depth, (1, height, width), device=device) != batch:
            raise ValueError(f"normalized_depth batch dimension must be {batch}")
        if normalized_depth.dtype != torch.float32:
            raise TypeError("normalized_depth must use dtype torch.float32")
    expected_shapes = (
        ("empty_detections", empty_detections, (0, 8)),
        ("frame_capture_times", frame_capture_times, (len(_SENSOR_ORDER),)),
        ("frame_estimated_poses", frame_estimated_poses, (len(_SENSOR_ORDER), 5)),
        ("frame_new", frame_new, (len(_SENSOR_ORDER),)),
        ("estimated_pose", estimated_pose, (5,)),
        ("goal_xy", goal_xy, (2,)),
        ("last_actions", last_actions, (4,)),
        ("episode_time", episode_time, ()),
        ("feedback", feedback, (37,)),
    )
    for name, value, tail in expected_shapes:
        if _require_batch(name, value, tail, device=device) != batch:
            raise ValueError(f"{name} batch dimension must be {batch}")
    for name, value in (("depth_valid", depth_valid), ("frame_new", frame_new)):
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
    if isinstance(max_detections, bool) or int(max_detections) != max_detections or max_detections < 1:
        raise ValueError("max_detections must be a positive integer")
    size = _config_value("size", size, positive=True)
    lidar_range = _config_value("lidar_range", lidar_range, positive=True)
    depth_range = _config_value("depth_range", depth_range, positive=True)
    max_time = _config_value("max_time", max_time, positive=True)

    # Normalize image/sensor tensors without leaving their device.
    lidar = (torch.where(
        lidar_valid,
        lidar_ranges.to(dtype=torch.float32) / lidar_range,
        torch.zeros_like(lidar_ranges, dtype=torch.float32),
    ).unsqueeze(1) if normalized_lidar is None else normalized_lidar)
    depth = (torch.where(
        depth_valid,
        depth_values.to(dtype=torch.float32) / depth_range,
        torch.zeros_like(depth_values, dtype=torch.float32),
    ).unsqueeze(1) if normalized_depth is None else normalized_depth)
    detections = torch.zeros((batch, int(max_detections), 8), dtype=torch.float32, device=device)

    # Reproduce FieldTask._observation arithmetic in float64, then return the
    # float32 policy state contract.
    time64 = episode_time.to(dtype=torch.float64)
    estimate64 = estimated_pose.to(dtype=torch.float64)
    goals64 = goal_xy.to(dtype=torch.float64)
    goal_state = goals64 / size * 2.0 - 1.0
    estimated_xy = estimate64[:, :2]
    estimated_xy_state = estimated_xy / size * 2.0 - 1.0
    heading = estimate64[:, 2]
    head_features = torch.stack((torch.sin(heading), torch.cos(heading)), dim=1)
    time_ratio = torch.clamp(time64 / max_time, max=1.0).unsqueeze(1)
    state_prefix = torch.cat((
        torch.ones((batch, 1), dtype=torch.float64, device=device),
        goal_state,
        estimated_xy_state,
        head_features,
        last_actions.to(dtype=torch.float64),
        time_ratio,
    ), dim=1)

    frame_poses64 = frame_estimated_poses.to(dtype=torch.float64)
    age = torch.clamp((time64.unsqueeze(1) - frame_capture_times.to(dtype=torch.float64)) / 2.0, max=1.0)
    fresh = frame_new.to(dtype=torch.float64)
    motion = (estimated_xy.unsqueeze(1) - frame_poses64[:, :, :2]) / lidar_range
    heading_delta = _wrap_tensor(heading.unsqueeze(1) - frame_poses64[:, :, 2]) / math.pi
    gimbal_yaw_delta = _wrap_tensor(estimate64[:, 3].unsqueeze(1) - frame_poses64[:, :, 3]) / math.pi
    gimbal_pitch_delta = (estimate64[:, 4].unsqueeze(1) - frame_poses64[:, :, 4]) / math.pi
    frame_features = torch.stack((
        age,
        fresh,
        motion[:, :, 0],
        motion[:, :, 1],
        heading_delta,
        gimbal_yaw_delta,
        gimbal_pitch_delta,
    ), dim=2).reshape(batch, 21)
    state = torch.clamp(torch.cat((state_prefix, frame_features), dim=1), min=-1.0, max=1.0)
    if state.shape[1] != 33:
        raise RuntimeError(f"state builder produced {state.shape[1]} values, expected 33")
    return {
        "lidar": lidar,
        "depth": depth,
        "detections": detections,
        "state": state.to(dtype=torch.float32),
        "feedback": feedback.to(dtype=torch.float32),
    }
