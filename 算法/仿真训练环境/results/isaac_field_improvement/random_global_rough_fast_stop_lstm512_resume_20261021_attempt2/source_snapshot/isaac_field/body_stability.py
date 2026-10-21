"""Policy-time body stability objectives and diagnostics, independent of sensors.

Simulator truth is used for rewards and evaluation only. These values are not
additional policy observations, and the trial coefficients are not hardware
safety ratings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class BodyStabilityConfig:
    excess_tilt_threshold_rad: float = math.radians(35.0)
    tilt_weight: float = 2.0
    angular_rate_weight: float = 0.02

    def __post_init__(self) -> None:
        if (not math.isfinite(self.excess_tilt_threshold_rad)
                or not 0 <= self.excess_tilt_threshold_rad < math.pi / 2):
            raise ValueError("excess tilt threshold must be finite and in [0, pi/2)")
        if any(not math.isfinite(value) or value < 0
               for value in (self.tilt_weight, self.angular_rate_weight)):
            raise ValueError("body stability weights must be finite and nonnegative")

    def as_dict(self) -> dict:
        return asdict(self)


def body_stability_sample(
    roll: float,
    pitch: float,
    body_angular_velocity: Sequence[float],
    dt: float,
    config: BodyStabilityConfig | None = None,
) -> dict:
    """Calculate an integrated penalty without penalizing normal slope following.

    Tilt is the body-up/world-up angle. Only tilt above the soft threshold is
    charged. The rate term uses physical angular velocity about body X/Y;
    Euler angle derivatives and moving-gimbal gyro values are not equivalent.
    Passing None preserves the original reward, while still exposing metrics.
    """
    roll, pitch, dt = float(roll), float(pitch), float(dt)
    angular = tuple(float(value) for value in body_angular_velocity)
    if (len(angular) != 3 or dt <= 0
            or not all(math.isfinite(value) for value in (roll, pitch, dt, *angular))):
        raise ValueError("body stability requires finite attitude/rates and positive dt")
    if config is not None and not isinstance(config, BodyStabilityConfig):
        raise TypeError("body stability config must be BodyStabilityConfig or None")
    parameters = config or BodyStabilityConfig()
    tilt = math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch))))
    excess = max(0.0, tilt - parameters.excess_tilt_threshold_rad)
    rate_sq = angular[0] ** 2 + angular[1] ** 2
    penalty = (dt * (parameters.tilt_weight * excess ** 2
                     + parameters.angular_rate_weight * rate_sq)
               if config is not None else 0.0)
    return {
        "enabled": config is not None, "roll_rad": roll, "pitch_rad": pitch,
        "tilt_rad": tilt, "excess_tilt_rad": excess,
        "body_angular_velocity_rad_s": list(angular),
        "roll_pitch_rate_sq": rate_sq, "penalty": penalty,
    }


@dataclass
class BodyStabilityAccumulator:
    """Accumulate samples at policy times; this is not a physics-substep maximum."""

    sample_count: int = 0
    duration_s: float = 0.0
    max_abs_roll_rad: float = 0.0
    max_abs_pitch_rad: float = 0.0
    max_tilt_rad: float = 0.0
    excess_tilt_duration_s: float = 0.0
    roll_pitch_rate_sq_integral: float = 0.0
    penalty_sum: float = 0.0

    def update(self, sample: dict, dt: float) -> None:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("stability sample duration must be finite and positive")
        self.sample_count += 1
        self.duration_s += dt
        self.max_abs_roll_rad = max(self.max_abs_roll_rad, abs(sample["roll_rad"]))
        self.max_abs_pitch_rad = max(self.max_abs_pitch_rad, abs(sample["pitch_rad"]))
        self.max_tilt_rad = max(self.max_tilt_rad, sample["tilt_rad"])
        self.excess_tilt_duration_s += dt if sample["excess_tilt_rad"] > 0 else 0.0
        self.roll_pitch_rate_sq_integral += dt * sample["roll_pitch_rate_sq"]
        self.penalty_sum += sample["penalty"]

    def summary(self) -> dict:
        result = asdict(self)
        result["roll_pitch_angular_rate_rms_rad_s"] = math.sqrt(
            self.roll_pitch_rate_sq_integral / self.duration_s
        ) if self.duration_s else 0.0
        result["sampling"] = "policy-step samples; not physics-substep extrema"
        return result
