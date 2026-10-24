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
    total = p_estimate + integral
    blocked = ((total > limit) & (error > 0.0)) | ((total < -limit) & (error < 0.0))
    delta = torch.where(blocked, torch.zeros_like(error), error * (ki * dt))
    updated = torch.clamp(integral + delta, min=-limit, max=limit)
    return updated, blocked


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
        for name, value, allow_zero in (
            ("kp", kp, False), ("ki", ki, True), ("dt", dt, False), ("limit", limit, False),
        ):
            if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
                suffix = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {suffix}")
        requested_device = torch.device(device)
        self.side_ids = _validated_side_ids(side_ids, device=requested_device)
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
        return broadcast_side_values(self.integral, self.side_ids)

    def update(self, target_wheel_qd: torch.Tensor, actual_wheel_qd: torch.Tensor) -> None:
        expected = (self.integral.shape[0], self.side_ids.numel())
        if tuple(target_wheel_qd.shape) != expected or tuple(actual_wheel_qd.shape) != expected:
            raise ValueError(f"target and actual wheel qd must both have shape {expected}")
        if target_wheel_qd.device != self.device or actual_wheel_qd.device != self.device:
            raise ValueError("target and actual wheel qd must be on the controller device")
        if not (torch.isfinite(target_wheel_qd).all() and torch.isfinite(actual_wheel_qd).all()):
            raise ValueError("target and actual wheel qd must be finite")
        self.last_error = (
            side_mean(target_wheel_qd, self.side_ids)
            - side_mean(actual_wheel_qd, self.side_ids)
        )
        self.last_p_estimate = self.kp * self.last_error
        self.integral, self.last_blocked = conditional_integral_step(
            self.last_error,
            self.integral,
            self.last_p_estimate,
            ki=self.ki,
            dt=self.dt,
            limit=self.limit,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear all or selected environment rows, matching DirectRLEnv reset semantics."""
        index = slice(None) if env_ids is None else env_ids
        self.integral[index] = 0.0
        self.last_error[index] = 0.0
        self.last_p_estimate[index] = 0.0
        self.last_blocked[index] = False

