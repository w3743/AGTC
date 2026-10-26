"""Small batched side-integral helper for the explicit proxy-wheel drive."""
from __future__ import annotations

import math

import torch


def _validated_side_ids(side_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    raw_sides = torch.as_tensor(side_ids, device=device)
    if raw_sides.ndim != 1 or raw_sides.numel() == 0:
        raise ValueError("side_ids must be a non-empty one-dimensional tensor")
    if raw_sides.dtype == torch.bool or raw_sides.is_complex():
        raise ValueError("side_ids must contain integer wheel-side ids")
    if raw_sides.is_floating_point() and (
        not torch.isfinite(raw_sides).all() or not torch.equal(raw_sides, raw_sides.round())
    ):
        raise ValueError("side_ids must contain finite integer wheel-side ids")
    sides = raw_sides.to(dtype=torch.long)
    if not torch.all((sides == 0) | (sides == 1)):
        raise ValueError("side ids must be 0 (left) or 1 (right)")
    if not torch.any(sides == 0) or not torch.any(sides == 1):
        raise ValueError("both left and right sides must contain at least one wheel")
    return sides


def side_mean(values: torch.Tensor, side_ids: torch.Tensor) -> torch.Tensor:
    """Reduce wheel values to left/right means, preserving the batch axis."""
    if values.ndim != 2:
        raise ValueError("wheel values must have shape (batch, wheels)")
    sides = _validated_side_ids(side_ids, device=values.device)
    if values.shape[1] != sides.numel():
        raise ValueError("wheel values and side_ids must contain the same wheel count")
    return torch.stack([values[:, sides == side].mean(dim=1) for side in (0, 1)], dim=1)


def broadcast_side_values(side_values: torch.Tensor, side_ids: torch.Tensor) -> torch.Tensor:
    """Broadcast two per-side values back to wheel order."""
    if side_values.ndim != 2 or side_values.shape[1] != 2:
        raise ValueError("side_values must have shape (batch, 2)")
    sides = _validated_side_ids(side_ids, device=side_values.device)
    return side_values.index_select(1, sides)


def _conditional_integral_step_math(
    error: torch.Tensor,
    integral: torch.Tensor,
    p_estimate: torch.Tensor,
    *,
    ki: float,
    dt: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = p_estimate + integral
    blocked = ((total > limit) & (error > 0.0)) | ((total < -limit) & (error < 0.0))
    delta = torch.where(blocked, torch.zeros_like(error), error * (ki * dt))
    updated = torch.clamp(integral + delta, min=-limit, max=limit)
    return updated, blocked


def _validate_controller_parameters(*, kp: float, ki: float, dt: float, limit: float) -> None:
    for name, value, allow_zero in (
        ("kp", kp, False), ("ki", ki, True), ("dt", dt, False), ("limit", limit, False),
    ):
        if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
            suffix = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {suffix}")


def conditional_integral_step(
    error: torch.Tensor,
    integral: torch.Tensor,
    p_estimate: torch.Tensor,
    *,
    ki: float,
    dt: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate side speed error with conditional anti-windup and bounded I."""
    if error.shape != integral.shape or error.shape != p_estimate.shape:
        raise ValueError("error, integral and P estimate must have matching shapes")
    if not (torch.isfinite(error).all() and torch.isfinite(integral).all()
            and torch.isfinite(p_estimate).all()):
        raise ValueError("PI inputs must be finite")
    if not math.isfinite(ki) or ki < 0.0:
        raise ValueError("ki must be finite and non-negative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("limit must be finite and positive")
    return _conditional_integral_step_math(
        error, integral, p_estimate, ki=ki, dt=dt, limit=limit,
    )


class SideIntegralDriveController:
    """Two-state left/right integral FF; native per-wheel IdealPD remains the P loop."""

    def __init__(
        self,
        num_envs: int,
        side_ids: torch.Tensor,
        *,
        device: torch.device | str,
        kp: float = 0.04,
        ki: float = 0.1,
        dt: float = 0.05,
        limit: float = 2.0,
    ) -> None:
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        _validate_controller_parameters(kp=kp, ki=ki, dt=dt, limit=limit)
        requested_device = torch.device(device)
        validated_side_ids = _validated_side_ids(side_ids, device=requested_device)
        self.side_ids = validated_side_ids.clone()
        # Private copies make the validated wheel layout fixed after construction.
        self._wheel_side_ids = validated_side_ids.clone()
        self._left_wheel_indices = torch.nonzero(
            self._wheel_side_ids == 0, as_tuple=False,
        ).flatten()
        self._right_wheel_indices = torch.nonzero(
            self._wheel_side_ids == 1, as_tuple=False,
        ).flatten()
        self._wheel_count = int(self._wheel_side_ids.numel())
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.limit = float(limit)
        self.integral = torch.zeros((int(num_envs), 2), device=requested_device, dtype=torch.float32)
        # Canonicalize CUDA aliases to the actual allocation device.
        self.device = self.integral.device
        self.last_error = torch.zeros_like(self.integral)
        self.last_p_estimate = torch.zeros_like(self.integral)
        self.last_blocked = torch.zeros_like(self.integral, dtype=torch.bool)

    @property
    def feedforward(self) -> torch.Tensor:
        return self.integral.index_select(1, self._wheel_side_ids)

    def _side_mean(self, values: torch.Tensor) -> torch.Tensor:
        left_mean = values.index_select(1, self._left_wheel_indices).mean(dim=1)
        right_mean = values.index_select(1, self._right_wheel_indices).mean(dim=1)
        return torch.stack((left_mean, right_mean), dim=1)

    def update(self, target_wheel_qd: torch.Tensor, actual_wheel_qd: torch.Tensor) -> None:
        _validate_controller_parameters(kp=self.kp, ki=self.ki, dt=self.dt, limit=self.limit)
        expected = (self.integral.shape[0], self._wheel_count)
        if tuple(target_wheel_qd.shape) != expected or tuple(actual_wheel_qd.shape) != expected:
            raise ValueError(f"target and actual wheel qd must both have shape {expected}")
        if target_wheel_qd.device != self.device or actual_wheel_qd.device != self.device:
            raise ValueError("target and actual wheel qd must be on the controller device")
        finite = (
            torch.isfinite(target_wheel_qd).all()
            & torch.isfinite(actual_wheel_qd).all()
            & torch.isfinite(self.integral).all()
        )
        error = self._side_mean(target_wheel_qd) - self._side_mean(actual_wheel_qd)
        p_estimate = self.kp * error
        finite = finite & torch.isfinite(error).all() & torch.isfinite(p_estimate).all()
        if not bool(finite):
            raise ValueError("PI inputs must be finite")
        self.last_error = error
        self.last_p_estimate = p_estimate
        self.integral, self.last_blocked = _conditional_integral_step_math(
            error, self.integral, p_estimate, ki=self.ki, dt=self.dt, limit=self.limit,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear all or selected environment rows, matching DirectRLEnv reset semantics."""
        index = slice(None) if env_ids is None else env_ids
        self.integral[index] = 0.0
        self.last_error[index] = 0.0
        self.last_p_estimate[index] = 0.0
        self.last_blocked[index] = False

