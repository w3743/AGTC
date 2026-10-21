"""Typed sensor feedback contract and per-environment IMU fusion model.

The simulation model below adds small, explicit sensor-noise assumptions to
native Newton/actuator signals. Its motor-current input is a model signal in
simulation, not calibrated or measured drive current.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


INPUT_SCHEMA = "sensor-imu-drive-v5"
STATE_DIM = 39
FEEDBACK_SCHEMA = "gimbal-imu-drive-feedback-v1"
FEEDBACK_DIM = 41
FEEDBACK_VALUE_DIM = 25
FEEDBACK_VALID_OFFSET = FEEDBACK_VALUE_DIM
FEEDBACK_AGE_OFFSET = FEEDBACK_VALID_OFFSET + 8
FEEDBACK_MAX_AGE_S = 2.0
FEEDBACK_TIMEOUT_S = 0.25
MAG_COUNTS_SCALE = 32768.0
G0_MPS2 = 9.80665
PI = math.pi
GIMBAL_PITCH_AXIS_SIGN = -1.0  # assets.py pitch joint rotates about local -Y.
DEFAULT_SENSOR_MOUNT_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)

FEEDBACK_GROUPS = (
    "accel", "gyro", "mag_temperature", "attitude", "gimbal_yaw",
    "gimbal_pitch", "motor_left", "motor_right",
)

FEEDBACK_VALUE_SLICES = {
    "accel": slice(0, 3),
    "gyro": slice(3, 6),
    "mag_temperature": slice(6, 10),
    "attitude": slice(10, 17),
    "gimbal_yaw": slice(17, 19),
    "gimbal_pitch": slice(19, 21),
    "motor_left": slice(21, 23),
    "motor_right": slice(23, 25),
}
FEEDBACK_VALID_SLICE = slice(FEEDBACK_VALID_OFFSET, FEEDBACK_AGE_OFFSET)
FEEDBACK_AGE_SLICE = slice(FEEDBACK_AGE_OFFSET, FEEDBACK_DIM)

FEEDBACK_LABELS = (
    "云台IMU加速度X / 9.80665（g）", "云台IMU加速度Y / 9.80665（g）", "云台IMU加速度Z / 9.80665（g）",
    "云台IMU角速度X / π（rad/s）", "云台IMU角速度Y / π（rad/s）", "云台IMU角速度Z / π（rad/s）",
    "磁场X计数 / 32768", "磁场Y计数 / 32768", "磁场Z计数 / 32768", "IMU温度（°C / 100）",
    "云台IMU姿态横滚 / π（rad）", "云台IMU姿态俯仰 / π（rad）", "云台IMU姿态偏航 / π（rad）",
    "云台IMU四元数X", "云台IMU四元数Y", "云台IMU四元数Z", "云台IMU四元数W",
    "云台偏航关节角 / π（rad）", "云台偏航关节速率 / π（rad/s）",
    "云台俯仰关节角 / π（rad）", "云台俯仰关节速率 / π（rad/s）",
    "左侧等效Iq / 100（A）", "左侧机械转速 / 10000（rpm）",
    "右侧等效Iq / 100（A）", "右侧机械转速 / 10000（rpm）",
    *(f"有效标记：{group}" for group in FEEDBACK_GROUPS),
    *(f"数据年龄 / 2秒：{group}" for group in FEEDBACK_GROUPS),
)

# Noise values are explicit simulator assumptions, not measured ATK-IMU901 or
# motor-controller characteristics. Signal values are not clipped.
SIM_SENSOR_MODEL = {
    "name": "simulated-gimbal-imu-drive-feedback-v1",
    "dt_s": 0.05,
    "filter_rate_hz": 20.0,
    "filter": "VQF 2.1.2 online VQF; uses only accelerometer, gyroscope, magnetometer",
    "acceleration_semantics": "proper acceleration in sensor body frame, m/s^2; zero is valid free-fall data",
    "magnetometer_semantics": "raw signed counts divided by 32768 for storage/VQF; no uT calibration evidence",
    "magnetic_reference": "ENU with magnetic north along +Y",
    "sensor_mount_assumption": "IMU axes aligned with pitch_link; no mount rotation or bias calibration available",
    "sensor_mount_rot_xyzw": list(DEFAULT_SENSOR_MOUNT_ROT_XYZW),
    "gimbal_kinematics": "R_body_sensor = Rz(yaw) * Ry(-pitch); pitch joint positive direction is local -Y",
    "attitude_source_simulation": "VQF 2.1.2 estimated from noisy accel/gyro/magnetometer only",
    "attitude_source_hardware": "ATK-IMU901 firmware Euler and quaternion fields are passed through independently",
    "accel_white_noise_std_mps2": 0.025,
    "accel_episode_bias_std_mps2": 0.05,
    "gyro_white_noise_std_rad_s": 0.0015,
    "gyro_episode_bias_std_rad_s": 0.003,
    "mag_white_noise_std_counts": 12.0,
    "mag_episode_bias_std_counts": 24.0,
    "temperature_nominal_degC": 25.0,
    "temperature_white_noise_std_degC": 0.05,
    "temperature_episode_offset_std_degC": 0.15,
    "temperature_random_walk_std_degC_sqrt_s": 0.002,
    "gimbal_position_white_noise_std_rad": 0.0005,
    "gimbal_rate_white_noise_std_rad_s": 0.003,
    "motor_iq_white_noise_std_A": 0.02,
    "motor_rpm_white_noise_std_rpm": 1.0,
    "motor_iq_semantics": (
        "simulation equivalent Iq only when supplied by the explicit actuator-effort model; "
        "assumed Kt=0.035 Nm/A and efficiency=1, uncalibrated and not measured current"
    ),
    "timeout_s": FEEDBACK_TIMEOUT_S,
    "age_max_s": FEEDBACK_MAX_AGE_S,
    "value_scaling": {
        "accel": {"factor": 1.0 / G0_MPS2, "unit": "g (m/s^2 divided by 9.80665)"},
        "gyro": {"factor": 1.0 / PI, "unit": "rad/s divided by pi"},
        "mag_counts": {"factor": 1.0 / MAG_COUNTS_SCALE, "unit": "raw signed counts / 32768"},
        "temperature": {"factor": 1.0 / 100.0, "unit": "degC / 100"},
        "attitude_euler": {"factor": 1.0 / PI, "unit": "rad / pi"},
        "attitude_quaternion": {"factor": 1.0, "unit": "unitless xyzw"},
        "gimbal_position": {"factor": 1.0 / PI, "unit": "rad / pi"},
        "gimbal_rate": {"factor": 1.0 / PI, "unit": "rad/s / pi"},
        "motor_iq": {"factor": 1.0 / 100.0, "unit": "A / 100; simulation value is model-equivalent"},
        "motor_rpm": {"factor": 1.0 / 10000.0, "unit": "mechanical rpm / 10000"},
        "age": {"factor": 1.0 / FEEDBACK_MAX_AGE_S, "unit": "s / age_max_s; bounded, not signal-clipped"},
    },
}

_SOURCE_GROUPS = FEEDBACK_GROUPS


def feedback_config() -> dict[str, Any]:
    """Return a JSON-safe, labeled description of the policy feedback contract."""
    return {
        "input_schema": INPUT_SCHEMA,
        "state_dim": STATE_DIM,
        "schema": FEEDBACK_SCHEMA,
        "dimension": FEEDBACK_DIM,
        "value_dimension": FEEDBACK_VALUE_DIM,
        "value_order": list(FEEDBACK_LABELS[:FEEDBACK_VALUE_DIM]),
        "valid_order": list(FEEDBACK_GROUPS),
        "age_order": list(FEEDBACK_GROUPS),
        "group_slices": {
            key: {"start": value.start, "stop": value.stop}
            for key, value in FEEDBACK_VALUE_SLICES.items()
        },
        "valid_slice": {"start": FEEDBACK_VALID_SLICE.start, "stop": FEEDBACK_VALID_SLICE.stop},
        "age_slice": {"start": FEEDBACK_AGE_SLICE.start, "stop": FEEDBACK_AGE_SLICE.stop},
        "labels": list(FEEDBACK_LABELS),
        "simulation_assumptions": dict(SIM_SENSOR_MODEL),
        "missing_or_expired": "group values are zero and its valid flag is zero; age is bounded",
    }


def _optional_vec(value: Sequence[float] | np.ndarray | None, size: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return arr.copy()


@dataclass(frozen=True)
class FeedbackSample:
    """One policy-step sample with per-group source timestamps in episode seconds.

    Optional arrays are deliberately raw engineering values. A missing group is
    held from its previous sample until its timeout; an explicit `valid[group]`
    false clears that group's cache immediately.
    """

    sample_time: float
    accel_mps2: Sequence[float] | np.ndarray | None = None
    gyro_rps: Sequence[float] | np.ndarray | None = None
    mag_counts: Sequence[float] | np.ndarray | None = None
    temperature_c: float | None = None
    attitude_euler_rad: Sequence[float] | np.ndarray | None = None
    attitude_quat_xyzw: Sequence[float] | np.ndarray | None = None
    gimbal_q_rad: Sequence[float] | np.ndarray | None = None
    gimbal_qd_rps: Sequence[float] | np.ndarray | None = None
    motor_iq_a: Sequence[float] | np.ndarray | None = None
    motor_rpm: Sequence[float] | np.ndarray | None = None
    valid: Mapping[str, bool] | None = None
    source_times: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.sample_time)) or float(self.sample_time) < 0:
            raise ValueError("sample_time must be finite and non-negative episode time")
        for name, size in (("accel_mps2", 3), ("gyro_rps", 3), ("mag_counts", 3),
                           ("attitude_euler_rad", 3), ("attitude_quat_xyzw", 4),
                           ("gimbal_q_rad", 2), ("gimbal_qd_rps", 2),
                           ("motor_iq_a", 2), ("motor_rpm", 2)):
            value = _optional_vec(getattr(self, name), size, name)
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.temperature_c is not None and not math.isfinite(float(self.temperature_c)):
            raise ValueError("temperature_c must be finite when present")
        if self.valid is not None:
            unknown = set(self.valid) - set(_SOURCE_GROUPS)
            if unknown:
                raise ValueError(f"unknown feedback validity groups: {sorted(unknown)}")
            object.__setattr__(self, "valid", MappingProxyType({str(k): bool(v) for k, v in self.valid.items()}))
        if self.source_times is not None:
            unknown = set(self.source_times) - set(_SOURCE_GROUPS)
            if unknown:
                raise ValueError(f"unknown feedback timestamp groups: {sorted(unknown)}")
            source_times = {str(k): float(v) for k, v in self.source_times.items()}
            if not all(math.isfinite(v) for v in source_times.values()):
                raise ValueError("feedback source times must be finite")
            object.__setattr__(self, "source_times", MappingProxyType(source_times))


@dataclass(frozen=True)
class FeedbackSnapshot:
    """Immutable feedback vector and derived sensor-only body orientation."""

    feedback: np.ndarray
    sample_time: float
    estimated_heading_rad: float
    estimated_roll_rad: float
    estimated_pitch_rad: float
    estimated_quaternion_xyzw: np.ndarray
    estimated_gimbal_q_rad: np.ndarray
    pose_valid: bool

    def __post_init__(self) -> None:
        vector = np.asarray(self.feedback, dtype=np.float32).reshape(FEEDBACK_DIM).copy()
        quat = np.asarray(self.estimated_quaternion_xyzw, dtype=np.float32).reshape(4).copy()
        gimbal = np.asarray(self.estimated_gimbal_q_rad, dtype=np.float32).reshape(2).copy()
        vector.setflags(write=False)
        quat.setflags(write=False)
        gimbal.setflags(write=False)
        object.__setattr__(self, "feedback", vector)
        object.__setattr__(self, "estimated_quaternion_xyzw", quat)
        object.__setattr__(self, "estimated_gimbal_q_rad", gimbal)


class SimFeedbackEstimator:
    """Per-environment VQF plus explicit simulated sensor noise/cache model."""

    def __init__(self, seed: int | None = None, *,
                 config: Mapping[str, Any] | None = None,
                 vqf_factory: Callable[..., Any] | None = None,
                 noise_enabled: bool = True):
        self.seed = None if seed is None else int(seed)
        self.config = dict(SIM_SENSOR_MODEL)
        if config is not None:
            self.config.update(config)
        mount_q = np.asarray(self.config["sensor_mount_rot_xyzw"], dtype=np.float64)
        if mount_q.shape != (4,) or not np.all(np.isfinite(mount_q)) or np.linalg.norm(mount_q) < 1e-12:
            raise ValueError("sensor_mount_rot_xyzw must be a finite non-zero xyzw quaternion")
        self.config["sensor_mount_rot_xyzw"] = (mount_q / np.linalg.norm(mount_q)).tolist()
        self._vqf_factory = vqf_factory
        self.noise_enabled = bool(noise_enabled)
        self.reset(now=0.0)

    @staticmethod
    def _make_vqf(factory: Callable[..., Any] | None, dt: float):
        if factory is None:
            try:
                from vqf import VQF  # type: ignore[import-not-found]
                factory = VQF
            except ImportError:
                try:
                    from vqf import PyVQF  # type: ignore[import-not-found]
                    factory = PyVQF
                except ImportError as exc:
                    raise RuntimeError("VQF 2.1.2 is required for sensor attitude fusion") from exc
        return factory(dt)

    def reset(self, now: float = 0.0) -> FeedbackSnapshot:
        if not math.isfinite(float(now)) or float(now) < 0:
            raise ValueError("reset time must be finite and non-negative")
        self.rng = np.random.default_rng(self.seed)
        # Delay importing/constructing VQF until an actual IMU sample arrives.
        self._vqf = None
        self._sensor: dict[str, dict[str, Any]] = {
            group: {"values": None, "timestamp": None, "valid": False}
            for group in _SOURCE_GROUPS
        }
        self._last_filter_timestamps: dict[str, float | None] = {
            key: None for key in ("accel", "gyro", "mag_temperature")
        }
        self._filter_initialized = False
        self._mag_filter_seen = False
        self._external_attitude_expected = False
        self._temperature_drift = 0.0
        self._temperature_offset = float(
            self.rng.normal(0, self.config["temperature_episode_offset_std_degC"])
            if self.noise_enabled else 0.0
        )
        self._episode_accel_bias = None
        self._episode_gyro_bias = None
        self._episode_mag_bias = None
        self._last_update_time: float | None = None
        self._last_now = float(now)
        self._snapshot = self._build_snapshot(float(now))
        return self._snapshot

    def _noise(self, std: float, size: int | None = None):
        if not self.noise_enabled or std == 0:
            return 0.0 if size is None else np.zeros(size, dtype=np.float64)
        return self.rng.normal(0.0, float(std), size=size)

    def _prepare_group(self, group: str, value: np.ndarray, timestamp: float) -> np.ndarray:
        if group == "accel":
            bias = self._episode_accel_bias
            return value + bias + self._noise(self.config["accel_white_noise_std_mps2"], 3)
        if group == "gyro":
            bias = self._episode_gyro_bias
            return value + bias + self._noise(self.config["gyro_white_noise_std_rad_s"], 3)
        if group == "mag_temperature":
            mag = value[:3] + self._episode_mag_bias + self._noise(self.config["mag_white_noise_std_counts"], 3)
            dt = max(0.0, float(self.config["dt_s"]))
            self._temperature_drift += float(self._noise(
                float(self.config["temperature_random_walk_std_degC_sqrt_s"]) * math.sqrt(dt)))
            raw_temp = value[3] if math.isfinite(float(value[3])) else float(self.config["temperature_nominal_degC"])
            temp = (raw_temp + self._temperature_offset + self._temperature_drift
                    + float(self._noise(self.config["temperature_white_noise_std_degC"])))
            return np.array([*mag / MAG_COUNTS_SCALE, temp], dtype=np.float64)
        if group in ("gimbal_yaw", "gimbal_pitch"):
            q_noise = self._noise(self.config["gimbal_position_white_noise_std_rad"])
            qd_noise = self._noise(self.config["gimbal_rate_white_noise_std_rad_s"])
            return value + np.array([q_noise, qd_noise], dtype=np.float64)
        if group.startswith("motor_"):
            return value + np.array([
                self._noise(self.config["motor_iq_white_noise_std_A"]),
                self._noise(self.config["motor_rpm_white_noise_std_rpm"]),
            ], dtype=np.float64)
        if group == "attitude":
            # Firmware-fused Euler/quaternion values are raw measured outputs.
            return value.copy()
        return value.copy()

    @property
    def snapshot(self) -> FeedbackSnapshot:
        return self._snapshot

    def update(self, now: float, accel_native: Sequence[float] | np.ndarray | None,
               gyro_native: Sequence[float] | np.ndarray | None,
               mag_counts_sensor: Sequence[float] | np.ndarray | None,
               gimbal_q: Sequence[float] | np.ndarray | None,
               gimbal_qd: Sequence[float] | np.ndarray | None,
               motor_iq_model: Sequence[float] | np.ndarray | None,
               motor_rpm_model: Sequence[float] | np.ndarray | None,
               *, temperature_c: float | None = None,
               attitude_euler_rad: Sequence[float] | np.ndarray | None = None,
               attitude_quat_xyzw: Sequence[float] | np.ndarray | None = None,
               valid: Mapping[str, bool] | None = None,
               source_times: Mapping[str, float] | None = None) -> FeedbackSnapshot:
        """Ingest one step; a repeated `now` returns the identical cached snapshot.

        `accel_native` is proper acceleration in m/s^2 in the sensor frame.
        Motor values are pass-through signals; their simulation provenance is
        documented in ``SIM_SENSOR_MODEL``.
        """
        now = float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("now must be finite and non-negative")
        if self._last_update_time is not None:
            if now == self._last_update_time:
                return self._snapshot
            if now < self._last_update_time:
                raise ValueError("feedback update time must be monotonic; reset between episodes")
        if self._last_update_time is None and now < self._last_now:
            raise ValueError("feedback update time precedes reset time")

        data = {
            "accel": _optional_vec(accel_native, 3, "accel_native"),
            "gyro": _optional_vec(gyro_native, 3, "gyro_native"),
            "mag_temperature": None,
            "attitude": None,
            "gimbal_yaw": None,
            "gimbal_pitch": None,
            "motor_left": None,
            "motor_right": None,
        }
        mag = _optional_vec(mag_counts_sensor, 3, "mag_counts_sensor")
        if mag is not None:
            temp = float(self.config["temperature_nominal_degC"] if temperature_c is None else temperature_c)
            if not math.isfinite(temp):
                raise ValueError("temperature_c must be finite when present")
            data["mag_temperature"] = np.array([*mag, temp], dtype=np.float64)
        attitude_euler = _optional_vec(attitude_euler_rad, 3, "attitude_euler_rad")
        attitude_quat = _optional_vec(attitude_quat_xyzw, 4, "attitude_quat_xyzw")
        if attitude_euler is not None and attitude_quat is not None:
            data["attitude"] = np.array([*attitude_euler, *attitude_quat], dtype=np.float64)
        q = _optional_vec(gimbal_q, 2, "gimbal_q")
        qd = _optional_vec(gimbal_qd, 2, "gimbal_qd")
        if q is not None and qd is not None:
            data["gimbal_yaw"] = np.array([q[0], qd[0]], dtype=np.float64)
            data["gimbal_pitch"] = np.array([q[1], qd[1]], dtype=np.float64)
        iq = _optional_vec(motor_iq_model, 2, "motor_iq_model")
        rpm = _optional_vec(motor_rpm_model, 2, "motor_rpm_model")
        if iq is not None and rpm is not None:
            data["motor_left"] = np.array([iq[0], rpm[0]], dtype=np.float64)
            data["motor_right"] = np.array([iq[1], rpm[1]], dtype=np.float64)

        validity = dict(valid or {})
        if ((attitude_euler is None) != (attitude_quat is None)
                or (attitude_quat is not None and float(np.linalg.norm(attitude_quat)) < 1e-12)
                or (data["attitude"] is None and validity.get("attitude", False))):
            # Euler and quaternion are one atomic feedback group. A partial or
            # zero quaternion packet invalidates this group without poisoning
            # the rest of the sensor sample.
            validity["attitude"] = False
        unknown = set(validity) - set(_SOURCE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feedback validity groups: {sorted(unknown)}")
        times = dict(source_times or {})
        unknown = set(times) - set(_SOURCE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feedback timestamp groups: {sorted(unknown)}")
        if "attitude" in validity or data["attitude"] is not None:
            self._external_attitude_expected = True

        if self._episode_accel_bias is None:
            self._episode_accel_bias = self._noise(self.config["accel_episode_bias_std_mps2"], 3)
            self._episode_gyro_bias = self._noise(self.config["gyro_episode_bias_std_rad_s"], 3)
            self._episode_mag_bias = self._noise(self.config["mag_episode_bias_std_counts"], 3)

        fresh: dict[str, np.ndarray] = {}
        for group in _SOURCE_GROUPS:
            supplied = data[group]
            if group in validity and not validity[group]:
                invalid_time = float(times.get(group, now))
                self._sensor[group] = {"values": None, "timestamp": invalid_time, "valid": False}
                continue
            if supplied is None:
                if group in validity and validity[group]:
                    raise ValueError(f"feedback group {group!r} marked valid without values")
                # A missing update holds the last sample until its own timeout.
                continue
            timestamp = float(times.get(group, now))
            if not math.isfinite(timestamp):
                raise ValueError(f"source timestamp for {group!r} must be finite")
            if timestamp > now + 1e-9 or timestamp < 0:
                # A future/out-of-epoch measurement is invalid; zero its group.
                self._sensor[group] = {"values": None, "timestamp": timestamp, "valid": False}
                continue
            previous_ts = self._sensor[group]["timestamp"]
            if previous_ts is not None and timestamp <= previous_ts:
                continue
            measured = self._prepare_group(group, supplied, timestamp)
            self._sensor[group] = {"values": measured, "timestamp": timestamp, "valid": True}
            fresh[group] = measured

        # VQF 2.1.2 expects sensor-frame rad/s, m/s^2 and arbitrary-unit mag.
        # Update only new source timestamps; do not synthesize samples by holding.
        if "gyro" in fresh:
            if self._vqf is None:
                self._vqf = self._make_vqf(self._vqf_factory, float(self.config["dt_s"]))
            self._vqf.updateGyr(np.asarray(fresh["gyro"], dtype=np.float64))
            self._last_filter_timestamps["gyro"] = self._sensor["gyro"]["timestamp"]
        if "accel" in fresh:
            if self._vqf is None:
                self._vqf = self._make_vqf(self._vqf_factory, float(self.config["dt_s"]))
            self._vqf.updateAcc(np.asarray(fresh["accel"], dtype=np.float64))
            self._last_filter_timestamps["accel"] = self._sensor["accel"]["timestamp"]
        if "mag_temperature" in fresh:
            if self._vqf is None:
                self._vqf = self._make_vqf(self._vqf_factory, float(self.config["dt_s"]))
            self._vqf.updateMag(np.asarray(fresh["mag_temperature"][:3], dtype=np.float64))
            self._last_filter_timestamps["mag_temperature"] = self._sensor["mag_temperature"]["timestamp"]
            self._mag_filter_seen = True
        # A board-provided attitude packet is a separate atomic source group.
        # It does not affect VQF; the filter remains sensor-only in simulation.
        if self._sensor["gyro"]["valid"] and self._sensor["accel"]["valid"]:
            self._filter_initialized = True

        self._last_update_time = now
        self._last_now = now
        self._snapshot = self._build_snapshot(now)
        return self._snapshot

    def update_sample(self, sample: FeedbackSample) -> FeedbackSnapshot:
        """Adapt the public typed sample to the raw measurement update API."""
        if not isinstance(sample, FeedbackSample):
            raise TypeError("sample must be FeedbackSample")
        iq = sample.motor_iq_a
        rpm = sample.motor_rpm
        if (iq is None) != (rpm is None):
            raise ValueError("motor_iq_a and motor_rpm must be supplied together")
        q = sample.gimbal_q_rad
        qd = sample.gimbal_qd_rps
        if (q is None) != (qd is None):
            raise ValueError("gimbal_q_rad and gimbal_qd_rps must be supplied together")
        return self.update(
            now=sample.sample_time,
            accel_native=sample.accel_mps2,
            gyro_native=sample.gyro_rps,
            mag_counts_sensor=sample.mag_counts,
            gimbal_q=q,
            gimbal_qd=qd,
            motor_iq_model=iq,
            motor_rpm_model=rpm,
            temperature_c=sample.temperature_c,
            attitude_euler_rad=sample.attitude_euler_rad,
            attitude_quat_xyzw=sample.attitude_quat_xyzw,
            valid=sample.valid,
            source_times=sample.source_times,
        )

    def _source_age(self, group: str, now: float) -> float:
        cached = self._sensor[group]
        timestamp = cached["timestamp"]
        if timestamp is None:
            return FEEDBACK_MAX_AGE_S
        return float(np.clip(max(0.0, now - float(timestamp)), 0.0, FEEDBACK_MAX_AGE_S))

    def _is_valid(self, group: str, now: float) -> bool:
        cached = self._sensor[group]
        return bool(cached["valid"] and self._source_age(group, now) <= FEEDBACK_TIMEOUT_S)

    def _body_rotation(self, sensor_quaternion_xyzw: np.ndarray, gimbal_q: np.ndarray) -> Rotation:
        sensor_to_world = Rotation.from_quat(sensor_quaternion_xyzw)
        yaw = Rotation.from_rotvec(np.array([0.0, 0.0, float(gimbal_q[0])]))
        pitch = Rotation.from_rotvec(np.array([0.0, GIMBAL_PITCH_AXIS_SIGN * float(gimbal_q[1]), 0.0]))
        mount = Rotation.from_quat(np.asarray(self.config["sensor_mount_rot_xyzw"], dtype=np.float64))
        body_to_sensor = yaw * pitch * mount
        return sensor_to_world * body_to_sensor.inv()

    def _build_snapshot(self, now: float) -> FeedbackSnapshot:
        values = np.zeros(FEEDBACK_DIM, dtype=np.float32)
        flags = np.zeros(8, dtype=np.float32)
        ages = np.full(8, FEEDBACK_MAX_AGE_S, dtype=np.float32)

        source_to_vector = {
            "accel": "accel", "gyro": "gyro", "mag_temperature": "mag_temperature",
            "gimbal_yaw": "gimbal_yaw", "gimbal_pitch": "gimbal_pitch",
            "motor_left": "motor_left", "motor_right": "motor_right",
        }
        for index, group in enumerate(FEEDBACK_GROUPS):
            if group == "attitude":
                continue
            age = self._source_age(group, now)
            ages[index] = age
            if not self._is_valid(group, now):
                continue
            vector_group = source_to_vector[group]
            block = self._sensor[group]["values"]
            scaled = np.asarray(block, dtype=np.float64).copy()
            if group == "accel":
                scaled /= G0_MPS2
            elif group == "gyro":
                scaled /= PI
            elif group == "mag_temperature":
                scaled[3] /= 100.0
            elif group in ("gimbal_yaw", "gimbal_pitch"):
                scaled /= PI
            elif group.startswith("motor_"):
                scaled[0] /= 100.0
                scaled[1] /= 10000.0
            values[FEEDBACK_VALUE_SLICES[vector_group]] = np.asarray(scaled, dtype=np.float32)
            flags[index] = 1.0

        if self._external_attitude_expected:
            attitude_valid = self._is_valid("attitude", now)
            ages[FEEDBACK_GROUPS.index("attitude")] = self._source_age("attitude", now)
        else:
            attitude_valid = (
                self._filter_initialized
                and self._is_valid("accel", now)
                and self._is_valid("gyro", now)
                and self._is_valid("mag_temperature", now)
            )
            imu_ages = [self._source_age(k, now) for k in ("accel", "gyro", "mag_temperature")]
            ages[FEEDBACK_GROUPS.index("attitude")] = max(imu_ages)
        sensor_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        sensor_euler = np.zeros(3, dtype=np.float64)
        body_euler = np.zeros(3, dtype=np.float64)
        if self._external_attitude_expected and self._is_valid("attitude", now):
            board = np.asarray(self._sensor["attitude"]["values"], dtype=np.float64)
            sensor_euler = board[:3].copy()
            sensor_quat_xyzw = board[3:7].copy()
            if attitude_valid and self._is_valid("gimbal_yaw", now) and self._is_valid("gimbal_pitch", now):
                gimbal_q = np.array([
                    self._sensor["gimbal_yaw"]["values"][0],
                    self._sensor["gimbal_pitch"]["values"][0],
                ], dtype=np.float64)
                body_euler = self._body_rotation(sensor_quat_xyzw, gimbal_q).as_euler("xyz")
        elif not self._external_attitude_expected and self._filter_initialized:
            quat_wxyz = np.asarray(
                self._vqf.getQuat9D() if self._mag_filter_seen else self._vqf.getQuat6D(),
                dtype=np.float64,
            ).reshape(4)
            norm = float(np.linalg.norm(quat_wxyz))
            if math.isfinite(norm) and norm > 1e-12:
                sensor_quat_xyzw = quat_wxyz[[1, 2, 3, 0]] / norm
                sensor_rotation = Rotation.from_quat(sensor_quat_xyzw)
                sensor_euler = sensor_rotation.as_euler("xyz")
                if attitude_valid and self._is_valid("gimbal_yaw", now) and self._is_valid("gimbal_pitch", now):
                    gimbal_q = np.array([
                        self._sensor["gimbal_yaw"]["values"][0],
                        self._sensor["gimbal_pitch"]["values"][0],
                    ], dtype=np.float64)
                    body_euler = self._body_rotation(sensor_quat_xyzw, gimbal_q).as_euler("xyz")
        if attitude_valid:
            i = FEEDBACK_GROUPS.index("attitude")
            values[FEEDBACK_VALUE_SLICES["attitude"]] = np.asarray(
                [*(sensor_euler / PI), *sensor_quat_xyzw], dtype=np.float32
            )
            flags[i] = 1.0

        values[FEEDBACK_VALID_SLICE] = flags
        values[FEEDBACK_AGE_SLICE] = ages / FEEDBACK_MAX_AGE_S
        # No clipping of valid sensor values: only ages are bounded.
        if not np.all(np.isfinite(values)):
            raise ValueError("feedback vector became non-finite; input exceeds float32 representable range")
        pose_valid = bool(
            attitude_valid and self._is_valid("gimbal_yaw", now) and self._is_valid("gimbal_pitch", now)
        )
        gimbal_q_est = np.array([
            self._sensor["gimbal_yaw"]["values"][0] if self._is_valid("gimbal_yaw", now) else 0.0,
            self._sensor["gimbal_pitch"]["values"][0] if self._is_valid("gimbal_pitch", now) else 0.0,
        ], dtype=np.float32)
        return FeedbackSnapshot(
            feedback=values,
            sample_time=float(now),
            estimated_heading_rad=float(body_euler[2]) if pose_valid else 0.0,
            estimated_roll_rad=float(body_euler[0]) if pose_valid else 0.0,
            estimated_pitch_rad=float(body_euler[1]) if pose_valid else 0.0,
            estimated_quaternion_xyzw=sensor_quat_xyzw.astype(np.float32),
            estimated_gimbal_q_rad=gimbal_q_est,
            pose_valid=pose_valid,
        )
