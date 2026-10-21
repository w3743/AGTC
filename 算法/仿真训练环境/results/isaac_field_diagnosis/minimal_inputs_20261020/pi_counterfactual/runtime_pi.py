"""Small CPU-testable runtime PI integral helper for the isolated probe only."""
from __future__ import annotations

import torch


def side_mean(values: torch.Tensor, side_ids: torch.Tensor) -> torch.Tensor:
    """Reduce wheel values to left/right side means, preserving the batch axis."""
    if values.ndim != 2 or side_ids.ndim != 1 or values.shape[1] != side_ids.numel():
        raise ValueError("wheel values must be (batch, wheels) with one side id per wheel")
    sides = side_ids.to(device=values.device, dtype=torch.long)
    if not torch.all((sides == 0) | (sides == 1)):
        raise ValueError("side ids must be 0 (left) or 1 (right)")
    means = []
    for side in (0, 1):
        mask = sides == side
        if not mask.any():
            raise ValueError(f"side {side} has no proxy wheels")
        means.append(values[:, mask].mean(dim=1))
    return torch.stack(means, dim=1)


def broadcast_side_values(side_values: torch.Tensor, side_ids: torch.Tensor) -> torch.Tensor:
    """Broadcast two per-side values to wheel order using the 0/1 side map."""
    if side_values.ndim != 2 or side_values.shape[1] != 2 or side_ids.ndim != 1:
        raise ValueError("side values must be (batch, 2) and side ids must be one-dimensional")
    sides = side_ids.to(device=side_values.device, dtype=torch.long)
    if not torch.all((sides == 0) | (sides == 1)):
        raise ValueError("side ids must be 0 (left) or 1 (right)")
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
    """Integrate unless estimated P+I is saturated and error pushes outward."""
    if error.shape != integral.shape or error.shape != p_estimate.shape:
        raise ValueError("error, integral and P estimate must have matching shapes")
    if not (torch.isfinite(error).all() and torch.isfinite(integral).all()
            and torch.isfinite(p_estimate).all()):
        raise ValueError("PI inputs must be finite")
    total = p_estimate + integral
    blocked = ((total > limit) & (error > 0.0)) | ((total < -limit) & (error < 0.0))
    delta = torch.where(blocked, torch.zeros_like(error), error * (ki * dt))
    updated = torch.clamp(integral + delta, min=-limit, max=limit)
    return updated, blocked


class RuntimePIController:
    """Small batched integral controller; P remains the actuator's native IdealPD."""

    def __init__(
        self,
        num_envs: int,
        num_wheels: int,
        *,
        device: torch.device | str,
        kp: float = 0.04,
        ki: float = 0.1,
        dt: float = 0.05,
        limit: float = 2.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.limit = float(limit)
        self.integral = torch.zeros((num_envs, num_wheels), device=device, dtype=torch.float32)
        self.last_error = torch.zeros_like(self.integral)
        self.last_p_estimate = torch.zeros_like(self.integral)
        self.last_blocked = torch.zeros_like(self.integral, dtype=torch.bool)

    @property
    def feedforward(self) -> torch.Tensor:
        return self.integral

    def update(self, target: torch.Tensor, actual: torch.Tensor) -> None:
        if target.shape != self.integral.shape or actual.shape != self.integral.shape:
            raise ValueError("target and actual wheel velocities must match PI state shape")
        if not (torch.isfinite(target).all() and torch.isfinite(actual).all()):
            raise ValueError("target and actual wheel velocities must be finite")
        self.last_error = target - actual
        self.last_p_estimate = self.kp * self.last_error
        self.integral, self.last_blocked = conditional_integral_step(
            self.last_error,
            self.integral,
            self.last_p_estimate,
            ki=self.ki,
            dt=self.dt,
            limit=self.limit,
        )

    def reset(self, env_ids=None) -> None:
        """Clear only selected environment rows, matching DirectRLEnv reset semantics."""
        if env_ids is None:
            index = slice(None)
        else:
            index = env_ids
        self.integral[index] = 0.0
        self.last_error[index] = 0.0
        self.last_p_estimate[index] = 0.0
        self.last_blocked[index] = False
