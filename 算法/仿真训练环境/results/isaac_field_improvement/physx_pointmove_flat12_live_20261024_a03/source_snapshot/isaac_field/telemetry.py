"""Pure stream decoders and a typed sample adapter for external telemetry.

This module performs no serial or CAN I/O. ATK-IMU901 packet definitions are
from the saved vendor guide at ``docs/hardware/atk_imu901/manual_text.txt``.
Drive CAN vectors are copied into the local test fixture from the current FOC
module's C codec golden vectors; tests do not depend on that external path.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import time
from typing import Any, Mapping


G0_M_S2 = 9.80665
ATK_HEADER = b"\x55\x55"
ATK_DATA_LENGTHS = {0x01: 6, 0x02: 8, 0x03: 12, 0x04: 8, 0x05: 10, 0x06: 8}
ATK_MAX_FRAME_BYTES = 5 + max(ATK_DATA_LENGTHS.values())
ATK_MAX_BUFFER_BYTES = ATK_MAX_FRAME_BYTES * 2

DRIVE_CAN_DLC = 8
DRIVE_FEEDBACK_STALE_S = 0.100
DEFAULT_SENSOR_TIMEOUT_S = 0.250
CAN_CRC_POLY = 0x1021
CAN_CRC_INIT = 0xFFFF

DRIVE_QUALITY_HALL_VALID = 1 << 0
DRIVE_QUALITY_RPM_VALID = 1 << 1
DRIVE_QUALITY_IQ_VALID = 1 << 2
DRIVE_QUALITY_ELECTRICAL_ANGLE_VALID = 1 << 3
DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED = 1 << 4
DRIVE_QUALITY_LOCAL_LIMIT_ACTIVE = 1 << 5
DRIVE_QUALITY_FAULT_LATCHED = 1 << 6
DRIVE_QUALITY_APPLIED_COMMAND_VALID = 1 << 7
DRIVE_INVALID_I16 = -0x8000

DRIVE_BOOT_IDS = {0x4D1: "left", 0x4D2: "right"}
DRIVE_STATE_IDS = {0x521: "left", 0x522: "right"}
DRIVE_QUALITY_IDS = {0x541: "left", 0x542: "right"}
DRIVE_SPEED_IDS = {"left": 0x501, "right": 0x502}
DRIVE_ID_BASES = {"boot": 0x4D0, "begin": 0x4E0, "close": 0x4F0,
                  "speed": 0x500, "state": 0x520, "quality": 0x540}


class TelemetryProtocolError(ValueError):
    """A malformed, corrupt, or out-of-window telemetry frame."""


def _finite_time(value: float | None, *, name: str) -> float:
    result = time.monotonic() if value is None else float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _signed_i16_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], "little", signed=True)


def _quality_dict(flags: int) -> dict[str, bool]:
    return {
        "hall_valid": bool(flags & DRIVE_QUALITY_HALL_VALID),
        "rpm_valid": bool(flags & DRIVE_QUALITY_RPM_VALID),
        "iq_valid": bool(flags & DRIVE_QUALITY_IQ_VALID),
        "electrical_angle_valid": bool(flags & DRIVE_QUALITY_ELECTRICAL_ANGLE_VALID),
        # This warning is independent of the RPM/Iq measurement-valid bits.
        "low_speed_or_reversal_degraded": bool(
            flags & DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED
        ),
        "local_limit_active": bool(flags & DRIVE_QUALITY_LOCAL_LIMIT_ACTIVE),
        "fault_latched": bool(flags & DRIVE_QUALITY_FAULT_LATCHED),
        "applied_command_valid": bool(flags & DRIVE_QUALITY_APPLIED_COMMAND_VALID),
    }


class ATKIMU901StreamDecoder:
    """Bounded, resynchronizing decoder for ATK-IMU901 upload packets.

    ``received_monotonic`` timestamps packets at host receipt. One feed call
    may contain any number of partial or complete packets. Invalid packets are
    skipped and counted; no command is sent to the IMU.
    """

    def __init__(self, *, accel_range_g: int = 4, gyro_range_dps: int = 2000):
        if accel_range_g not in (2, 4, 8, 16):
            raise ValueError("accel_range_g must be one of 2, 4, 8, 16")
        if gyro_range_dps not in (250, 500, 1000, 2000):
            raise ValueError("gyro_range_dps must be one of 250, 500, 1000, 2000")
        self.accel_range_g = int(accel_range_g)
        self.gyro_range_dps = int(gyro_range_dps)
        self._buffer = bytearray()
        self.stats = {
            "bytes_discarded": 0,
            "bad_lengths": 0,
            "bad_checksums": 0,
            "unknown_ids": 0,
            "truncated_frames": 0,
            "buffer_overflows": 0,
            "ignored_pressure_frames": 0,
        }

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def flush(self) -> int:
        """Discard a final incomplete packet and return the number of bytes."""
        count = len(self._buffer)
        if count:
            self.stats["truncated_frames"] += 1
            self.stats["bytes_discarded"] += count
            self._buffer.clear()
        return count

    def feed(self, data: bytes | bytearray | memoryview,
             received_monotonic: float | None = None) -> list[dict[str, Any]]:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        timestamp = _finite_time(received_monotonic, name="received_monotonic")
        decoded: list[dict[str, Any]] = []
        for value in data:
            self._buffer.append(int(value))
            if len(self._buffer) > ATK_MAX_BUFFER_BYTES:
                self._buffer.pop(0)
                self.stats["buffer_overflows"] += 1
                self.stats["bytes_discarded"] += 1
            decoded.extend(self._drain(timestamp))
        return decoded

    def _drain(self, timestamp: float) -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = []
        while True:
            header_at = self._buffer.find(ATK_HEADER)
            if header_at < 0:
                keep = 1 if self._buffer and self._buffer[-1] == 0x55 else 0
                discard = len(self._buffer) - keep
                if discard:
                    del self._buffer[:discard]
                    self.stats["bytes_discarded"] += discard
                return decoded
            if header_at:
                del self._buffer[:header_at]
                self.stats["bytes_discarded"] += header_at
            if len(self._buffer) < 4:
                return decoded
            frame_id = self._buffer[2]
            data_len = self._buffer[3]
            expected_len = ATK_DATA_LENGTHS.get(frame_id)
            if expected_len is None:
                self.stats["unknown_ids"] += 1
                del self._buffer[0]
                self.stats["bytes_discarded"] += 1
                continue
            if data_len != expected_len:
                self.stats["bad_lengths"] += 1
                del self._buffer[0]
                self.stats["bytes_discarded"] += 1
                continue
            frame_len = 5 + data_len
            if len(self._buffer) < frame_len:
                return decoded
            packet = bytes(self._buffer[:frame_len])
            if (sum(packet[:-1]) & 0xFF) != packet[-1]:
                self.stats["bad_checksums"] += 1
                # Preserve possible nested 0x55 0x55 sync in corrupted data.
                del self._buffer[0]
                self.stats["bytes_discarded"] += 1
                continue
            del self._buffer[:frame_len]
            if frame_id == 0x05:
                # Pressure/altitude and its temperature are intentionally not
                # surfaced to the policy or FeedbackSample.
                self.stats["ignored_pressure_frames"] += 1
                continue
            decoded.extend(self._decode_packet(frame_id, packet[4:-1], timestamp, packet))

    def _decode_packet(self, frame_id: int, payload: bytes, timestamp: float,
                       packet: bytes) -> list[dict[str, Any]]:
        raw_hex = packet.hex(" ").upper()
        common = {
            "source": "atk_imu901",
            "timestamp": timestamp,
            "timestamps": {"host_receive_monotonic_s": timestamp},
            "valid": True,
            "frame_id": frame_id,
            "raw_hex": raw_hex,
        }
        if frame_id == 0x01:
            euler = [
                _signed_i16_le(payload, offset) * math.pi / 32768.0
                for offset in (0, 2, 4)
            ]
            return [{
                **common,
                "group": "attitude",
                "units": {"attitude_euler_rad": "rad"},
                "values": {"attitude_euler_rad": euler,
                           "order": ["roll_x", "pitch_y", "yaw_z"]},
            }]
        if frame_id == 0x02:
            # Vendor order is q0=w, q1, q2, q3; preserve every signed value.
            q_wxyz = [_signed_i16_le(payload, offset) / 32768.0 for offset in (0, 2, 4, 6)]
            q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
            return [{
                **common,
                "valid": any(component != 0.0 for component in q_xyzw),
                "group": "attitude",
                "units": {"attitude_quat_xyzw": "unitless"},
                "values": {"attitude_quat_xyzw": q_xyzw,
                           "attitude_quat_wxyz": q_wxyz},
            }]
        if frame_id == 0x03:
            accel_raw = [_signed_i16_le(payload, offset) for offset in (0, 2, 4)]
            gyro_raw = [_signed_i16_le(payload, offset) for offset in (6, 8, 10)]
            accel = [value / 32768.0 * self.accel_range_g * G0_M_S2
                     for value in accel_raw]
            gyro = [value / 32768.0 * self.gyro_range_dps * math.pi / 180.0
                    for value in gyro_raw]
            return [
                {
                    **common,
                    "group": "accel",
                    "units": {"accel_mps2": "m/s^2"},
                    "values": {"accel_mps2": accel, "raw_counts": accel_raw,
                               "configured_range_g": self.accel_range_g},
                },
                {
                    **common,
                    "group": "gyro",
                    "units": {"gyro_rps": "rad/s"},
                    "values": {"gyro_rps": gyro, "raw_counts": gyro_raw,
                               "configured_range_dps": self.gyro_range_dps},
                },
            ]
        if frame_id == 0x04:
            mag = [_signed_i16_le(payload, offset) for offset in (0, 2, 4)]
            temperature = _signed_i16_le(payload, 6) / 100.0
            return [{
                **common,
                "group": "mag_temperature",
                "units": {"mag_counts": "raw signed counts; no uT calibration in guide",
                          "temperature_c": "degC"},
                "values": {"mag_counts": mag, "temperature_c": temperature},
            }]
        if frame_id == 0x06:
            ports = [int.from_bytes(payload[offset:offset + 2], "little")
                     for offset in (0, 2, 4, 6)]
            return [{
                **common,
                "group": "ports",
                "units": {"port_counts": "raw unsigned counts"},
                "values": {"ports": ports},
                "diagnostic_only": True,
            }]
        raise AssertionError(f"unexpected decoded ATK-IMU901 frame id: {frame_id}")


def drive_crc16_ccitt_false(can_id: int, dlc: int, payload_prefix: bytes) -> int:
    """CRC16/CCITT-FALSE over CAN ID little-endian, DLC, and six payload bytes."""
    if isinstance(can_id, bool) or not isinstance(can_id, int) or not 0 <= can_id <= 0x7FF:
        raise TelemetryProtocolError("CAN ID must be an 11-bit standard identifier")
    if dlc != DRIVE_CAN_DLC or len(payload_prefix) != 6:
        raise TelemetryProtocolError("drive-v1 CRC requires DLC=8 and six covered data bytes")
    stream = bytes((can_id & 0xFF, (can_id >> 8) & 0xFF, dlc)) + bytes(payload_prefix)
    crc = CAN_CRC_INIT
    for value in stream:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ CAN_CRC_POLY) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _normalize_node(node: str | int) -> str:
    if isinstance(node, bool):
        raise TelemetryProtocolError("drive node must be left/1 or right/2")
    if (type(node) is int and node == 1) or (isinstance(node, str) and node in ("left", "LEFT")):
        return "left"
    if (type(node) is int and node == 2) or (isinstance(node, str) and node in ("right", "RIGHT")):
        return "right"
    raise TelemetryProtocolError("drive node must be left/1 or right/2")


def encode_speed_command(node: str | int, *, session_id: int, sequence: int,
                         target_motor_rpm: int, ttl_10ms: int) -> dict[str, Any]:
    """Encode a v1 speed request; this returns bytes and never writes a CAN bus."""
    side = _normalize_node(node)
    for name, value, minimum, maximum in (("session_id", session_id, 1, 0xFFFF),
                                          ("sequence", sequence, 0, 0xFF),
                                          ("target_motor_rpm", target_motor_rpm, -0x7FFF, 0x7FFF),
                                          ("ttl_10ms", ttl_10ms, 1, 10)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TelemetryProtocolError(f"{name} must be an integer")
        if value < minimum or value > maximum:
            raise TelemetryProtocolError(f"{name} is outside the v1 range")
    if target_motor_rpm == DRIVE_INVALID_I16:
        raise TelemetryProtocolError("target motor RPM -32768 is reserved")
    # Negative mechanical RPM is signed little-endian; 0x8000 is reserved.
    rpm_bytes = int(target_motor_rpm).to_bytes(2, "little", signed=True)
    prefix = (session_id.to_bytes(2, "little") + bytes((sequence,)) + rpm_bytes + bytes((ttl_10ms,)))
    can_id = DRIVE_SPEED_IDS[side]
    crc = drive_crc16_ccitt_false(can_id, DRIVE_CAN_DLC, prefix)
    data = prefix + crc.to_bytes(2, "little")
    return {
        "can_id": can_id,
        "dlc": DRIVE_CAN_DLC,
        "data": data,
        "data_hex": data.hex(" ").upper(),
        "extended": False,
        "remote": False,
    }


def decode_drive_feedback_frame(can_id: int, data: bytes | bytearray | memoryview, *,
                                dlc: int | None = None, extended: bool = False,
                                remote: bool = False,
                                received_monotonic: float | None = None) -> dict[str, Any] | None:
    """Decode boot/state/quality CAN frames. Unknown protocol IDs return None."""
    if isinstance(can_id, bool) or not isinstance(can_id, int) or not 0 <= can_id <= 0x7FF:
        raise TelemetryProtocolError("CAN ID must be an 11-bit standard identifier")
    if extended:
        raise TelemetryProtocolError("drive-v1 rejects extended CAN frames")
    if remote:
        raise TelemetryProtocolError("drive-v1 rejects remote CAN frames")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    payload = bytes(data)
    effective_dlc = len(payload) if dlc is None else dlc
    if isinstance(effective_dlc, bool) or not isinstance(effective_dlc, int):
        raise TelemetryProtocolError("DLC must be an integer")
    if effective_dlc != DRIVE_CAN_DLC or len(payload) != DRIVE_CAN_DLC:
        raise TelemetryProtocolError("drive-v1 requires exactly DLC=8 and eight data bytes")

    if can_id in DRIVE_BOOT_IDS:
        kind, side = "boot_epoch", DRIVE_BOOT_IDS[can_id]
    elif can_id in DRIVE_STATE_IDS:
        kind, side = "motor_state", DRIVE_STATE_IDS[can_id]
    elif can_id in DRIVE_QUALITY_IDS:
        kind, side = "motor_quality", DRIVE_QUALITY_IDS[can_id]
    else:
        return None

    timestamp = _finite_time(received_monotonic, name="received_monotonic")
    expected_crc = drive_crc16_ccitt_false(can_id, effective_dlc, payload[:6])
    received_crc = int.from_bytes(payload[6:8], "little")
    if expected_crc != received_crc:
        raise TelemetryProtocolError("drive-v1 CRC mismatch")

    if kind == "boot_epoch":
        boot_epoch = int.from_bytes(payload[0:4], "little")
        reserved = int.from_bytes(payload[4:6], "little")
        if boot_epoch == 0:
            raise TelemetryProtocolError("boot_epoch must be nonzero")
        if reserved != 0:
            raise TelemetryProtocolError("boot epoch reserved field must be zero")
        units = {"boot_epoch": "u32 device boot generation"}
        values = {"boot_epoch": boot_epoch}
        group = "boot_epoch"
    elif kind == "motor_state":
        sample_seq = int.from_bytes(payload[0:2], "little")
        rpm = _signed_i16_le(payload, 2)
        iq_centiamp = _signed_i16_le(payload, 4)
        units = {"mechanical_rpm": "motor-shaft mechanical rpm",
                 "iq_centiamp": "0.01 A per count"}
        values = {"sample_sequence": sample_seq, "mechanical_rpm_i16": rpm,
                  "iq_centiamp": iq_centiamp}
        group = f"motor_{side}"
    else:
        sample_seq = int.from_bytes(payload[0:2], "little")
        device_tick = int.from_bytes(payload[2:4], "little")
        flags = payload[4]
        applied_seq = payload[5]
        quality = _quality_dict(flags)
        if not quality["applied_command_valid"] and applied_seq != 0xFF:
            raise TelemetryProtocolError("invalid applied command must carry seq 0xFF")
        units = {"device_tick_ms_low16": "device-local milliseconds, uint16 wrap",
                 "quality_flags": "bit flags"}
        values = {"sample_sequence": sample_seq, "device_tick_ms_low16": device_tick,
                  "quality_flags": flags, "quality": quality,
                  "applied_command_sequence": applied_seq}
        group = f"motor_{side}"
    return {
        "source": f"foc_drive_{side}",
        "group": group,
        "timestamp": timestamp,
        "timestamps": {"host_receive_monotonic_s": timestamp},
        "units": units,
        "valid": True,
        "kind": kind,
        "can_id": can_id,
        "dlc": effective_dlc,
        "raw_hex": payload.hex(" ").upper(),
        "values": values,
    }


def _seq16_newer(candidate: int, reference: int) -> bool:
    distance = (candidate - reference) & 0xFFFF
    return 0 < distance < 0x8000


def _validate_feedback_pair(state: Mapping[str, Any], quality: Mapping[str, Any]) -> dict[str, Any]:
    state_values = state["values"]
    quality_values = quality["values"]
    if state_values["sample_sequence"] != quality_values["sample_sequence"]:
        raise TelemetryProtocolError("state and quality sample sequences do not match")
    flags = quality_values["quality_flags"]
    rpm = state_values["mechanical_rpm_i16"]
    iq = state_values["iq_centiamp"]
    q = quality_values["quality"]
    if q["rpm_valid"] == (rpm == DRIVE_INVALID_I16):
        raise TelemetryProtocolError("RPM sentinel and RPM_VALID bit disagree")
    if q["iq_valid"] == (iq == DRIVE_INVALID_I16):
        raise TelemetryProtocolError("Iq sentinel and IQ_VALID bit disagree")
    return {
        "source": state["source"],
        "group": state["group"],
        "timestamp": max(state["timestamp"], quality["timestamp"]),
        "timestamps": {
            "state_host_receive_monotonic_s": state["timestamp"],
            "quality_host_receive_monotonic_s": quality["timestamp"],
        },
        "units": {"mechanical_rpm": "motor-shaft mechanical rpm", "iq_A": "A"},
        "valid": bool(q["rpm_valid"] and q["iq_valid"]),
        "kind": "motor_feedback_pair",
        "values": {
            "sample_sequence": state_values["sample_sequence"],
            "mechanical_rpm": rpm if q["rpm_valid"] else None,
            "iq_A": iq / 100.0 if q["iq_valid"] else None,
            "quality_flags": flags,
            "quality": dict(q),
            "device_tick_ms_low16": quality_values["device_tick_ms_low16"],
            "applied_command_sequence": (
                quality_values["applied_command_sequence"] if q["applied_command_valid"] else None
            ),
        },
    }


class TelemetryAssembler:
    """Pair drive feedback and assemble read-only values into FeedbackSample.

    Host monotonic receipt times are kept for freshness checks. ``sample_time``
    passed to :meth:`assemble` is episode-relative; per-source times are mapped
    from host-age into that episode clock. No CAN/serial port is opened or
    written by this class.
    """

    def __init__(self, *, accel_range_g: int = 4, gyro_range_dps: int = 2000,
                 sensor_timeout_s: float = DEFAULT_SENSOR_TIMEOUT_S,
                 motor_timeout_s: float = DRIVE_FEEDBACK_STALE_S,
                 reorder_window: int = 16, seen_window: int = 64):
        if not math.isfinite(sensor_timeout_s) or sensor_timeout_s <= 0:
            raise ValueError("sensor_timeout_s must be finite and positive")
        if not math.isfinite(motor_timeout_s) or not 0 < motor_timeout_s <= DRIVE_FEEDBACK_STALE_S:
            raise ValueError("motor_timeout_s must be in (0, 0.100] seconds")
        if not 1 <= reorder_window < 0x8000:
            raise ValueError("reorder_window must be in [1, 32767]")
        if seen_window < reorder_window + 1:
            raise ValueError("seen_window must exceed reorder_window")
        self.imu = ATKIMU901StreamDecoder(accel_range_g=accel_range_g,
                                         gyro_range_dps=gyro_range_dps)
        self.sensor_timeout_s = float(sensor_timeout_s)
        self.motor_timeout_s = float(motor_timeout_s)
        self.reorder_window = int(reorder_window)
        self.seen_window = int(seen_window)
        self._imu_latest: dict[str, tuple[Any, float]] = {}
        self._attitude_euler: tuple[list[float], float] | None = None
        self._attitude_quat: tuple[list[float], float] | None = None
        self._gimbal: dict[str, tuple[float, float, float]] = {}
        self._boot_epoch: dict[str, int] = {}
        self._pending: dict[str, dict[str, OrderedDict[int, dict[str, Any]]]] = {
            side: {"motor_state": OrderedDict(), "motor_quality": OrderedDict()}
            for side in ("left", "right")
        }
        self._seen: dict[tuple[str, str], OrderedDict[int, bytes]] = {}
        self._newest_sequence: dict[tuple[str, str], int] = {}
        self._latest_motor_pair: dict[str, dict[str, Any]] = {}
        self._last_device_tick: dict[str, int] = {}
        self.last_tick_delta_ms: dict[str, int] = {}
        self.stats = {
            "duplicate_drive_frames": 0,
            "out_of_order_drive_frames": 0,
            "stale_drive_frames": 0,
            "unmatched_drive_frames": 0,
            "expired_drive_pairs": 0,
            "paired_drive_samples": 0,
        }

    def feed_imu_bytes(self, data: bytes | bytearray | memoryview,
                       received_monotonic: float | None = None) -> list[dict[str, Any]]:
        frames = self.imu.feed(data, received_monotonic)
        for frame in frames:
            group = frame["group"]
            timestamp = frame["timestamp"]
            values = frame["values"]
            if group in ("accel", "gyro", "mag_temperature"):
                payload_key = {"accel": "accel_mps2", "gyro": "gyro_rps",
                               "mag_temperature": "mag_counts"}[group]
                cached = {payload_key: values[payload_key]}
                cache_group = group
                if group == "mag_temperature":
                    # Keep the decoder's wire-frame name for diagnostics, but
                    # publish magnetometer freshness independently of temperature.
                    cached["temperature_c"] = values["temperature_c"]
                    cache_group = "mag"
                self._imu_latest[cache_group] = (cached, timestamp)
            elif group == "attitude":
                if not frame["valid"]:
                    # A zero quaternion cannot initialize FeedbackSample. Clear
                    # the atomic factory attitude pair instead of raising or
                    # silently keeping an older value alive.
                    self._attitude_euler = None
                    self._attitude_quat = None
                    continue
                if "attitude_euler_rad" in values:
                    self._attitude_euler = (list(values["attitude_euler_rad"]), timestamp)
                if "attitude_quat_xyzw" in values:
                    self._attitude_quat = (list(values["attitude_quat_xyzw"]), timestamp)
        return frames

    def update_gimbal_axis(self, axis: str, q_rad: float, qd_rad_s: float,
                           received_monotonic: float | None = None) -> dict[str, Any]:
        axis = str(axis).lower()
        if axis not in ("yaw", "pitch"):
            raise ValueError("axis must be 'yaw' or 'pitch'")
        q = float(q_rad)
        qd = float(qd_rad_s)
        if not math.isfinite(q) or not math.isfinite(qd):
            raise ValueError("gimbal q and qd must be finite")
        timestamp = _finite_time(received_monotonic, name="received_monotonic")
        self._gimbal[axis] = (q, qd, timestamp)
        return {
            "source": f"gimbal_{axis}",
            "group": f"gimbal_{axis}",
            "timestamp": timestamp,
            "timestamps": {"host_receive_monotonic_s": timestamp},
            "units": {"q_rad": "rad", "qd_rad_s": "rad/s"},
            "valid": True,
            "values": {"q_rad": q, "qd_rad_s": qd},
        }

    def feed_can_frame(self, can_id: int, data: bytes | bytearray | memoryview, *,
                       dlc: int | None = None, extended: bool = False,
                       remote: bool = False,
                       received_monotonic: float | None = None) -> dict[str, Any] | None:
        frame = decode_drive_feedback_frame(
            can_id, data, dlc=dlc, extended=extended, remote=remote,
            received_monotonic=received_monotonic,
        )
        if frame is None:
            return None
        if frame["kind"] == "boot_epoch":
            side = frame["source"].removeprefix("foc_drive_")
            new_epoch = frame["values"]["boot_epoch"]
            old_epoch = self._boot_epoch.get(side)
            if old_epoch != new_epoch:
                self._clear_motor_state(side)
                self._boot_epoch[side] = new_epoch
            return frame
        side = frame["source"].removeprefix("foc_drive_")
        if side not in self._boot_epoch:
            raise TelemetryProtocolError("drive state/quality arrived before boot epoch")
        kind = frame["kind"]
        sequence = frame["values"]["sample_sequence"]
        raw = bytes.fromhex(frame["raw_hex"])
        if not self._remember_sequence(side, kind, sequence, raw):
            frame["duplicate"] = True
            return frame
        self._pending[side][kind][sequence] = frame
        self._trim_pending(side, kind)
        other_kind = "motor_quality" if kind == "motor_state" else "motor_state"
        other = self._pending[side][other_kind].get(sequence)
        if other is None:
            self.stats["unmatched_drive_frames"] += 1
            return frame
        state = frame if kind == "motor_state" else other
        quality = frame if kind == "motor_quality" else other
        pair = _validate_feedback_pair(state, quality)
        now = frame["timestamp"]
        state_age = now - state["timestamp"]
        quality_age = now - quality["timestamp"]
        if (state_age < 0 or quality_age < 0 or state_age > self.motor_timeout_s or
                quality_age > self.motor_timeout_s):
            self.stats["expired_drive_pairs"] += 1
            self._pending[side]["motor_state"].pop(sequence, None)
            self._pending[side]["motor_quality"].pop(sequence, None)
            return frame
        self._pending[side]["motor_state"].pop(sequence, None)
        self._pending[side]["motor_quality"].pop(sequence, None)
        pair["values"]["boot_epoch"] = self._boot_epoch[side]
        prior_pair = self._latest_motor_pair.get(side)
        prior_sequence = None if prior_pair is None else prior_pair["values"]["sample_sequence"]
        if prior_sequence is None or _seq16_newer(sequence, prior_sequence):
            previous_tick = self._last_device_tick.get(side)
            tick = pair["values"]["device_tick_ms_low16"]
            if previous_tick is not None:
                self.last_tick_delta_ms[side] = (tick - previous_tick) & 0xFFFF
            self._last_device_tick[side] = tick
            self._latest_motor_pair[side] = pair
            self.stats["paired_drive_samples"] += 1
        else:
            self.stats["out_of_order_drive_frames"] += 1
        return frame

    def _clear_motor_state(self, side: str) -> None:
        self._pending[side] = {"motor_state": OrderedDict(), "motor_quality": OrderedDict()}
        self._latest_motor_pair.pop(side, None)
        self._last_device_tick.pop(side, None)
        self.last_tick_delta_ms.pop(side, None)
        for key in ((side, "motor_state"), (side, "motor_quality")):
            self._seen.pop(key, None)
            self._newest_sequence.pop(key, None)

    def _remember_sequence(self, side: str, kind: str, sequence: int, raw: bytes) -> bool:
        key = (side, kind)
        seen = self._seen.setdefault(key, OrderedDict())
        prior = seen.get(sequence)
        if prior is not None:
            if prior != raw:
                raise TelemetryProtocolError("conflicting duplicate drive sample sequence")
            self.stats["duplicate_drive_frames"] += 1
            return False
        newest = self._newest_sequence.get(key)
        if newest is None:
            self._newest_sequence[key] = sequence
        elif _seq16_newer(sequence, newest):
            self._newest_sequence[key] = sequence
        else:
            back = (newest - sequence) & 0xFFFF
            if back == 0 or back > self.reorder_window:
                self.stats["stale_drive_frames"] += 1
                raise TelemetryProtocolError("drive sample sequence is stale or outside reorder window")
            self.stats["out_of_order_drive_frames"] += 1
        seen[sequence] = raw
        seen.move_to_end(sequence)
        while len(seen) > self.seen_window:
            seen.popitem(last=False)
        return True

    def _trim_pending(self, side: str, kind: str) -> None:
        pending = self._pending[side][kind]
        newest = self._newest_sequence.get((side, kind))
        if newest is None:
            return
        for sequence in list(pending):
            back = (newest - sequence) & 0xFFFF
            if 0 < back <= 0x8000 and back > self.reorder_window:
                pending.pop(sequence, None)
                self.stats["stale_drive_frames"] += 1

    def get_motor_feedback(self, node: str | int,
                           now_monotonic: float | None = None) -> dict[str, Any] | None:
        side = _normalize_node(node)
        now = _finite_time(now_monotonic, name="now_monotonic")
        pair = self._latest_motor_pair.get(side)
        if pair is None:
            return None
        state_time = pair["timestamps"]["state_host_receive_monotonic_s"]
        quality_time = pair["timestamps"]["quality_host_receive_monotonic_s"]
        if (now < state_time or now < quality_time or now - state_time > self.motor_timeout_s or
                now - quality_time > self.motor_timeout_s):
            return None
        return pair

    def assemble(self, sample_time: float, now_monotonic: float | None = None):
        """Return one `FeedbackSample`, with each source group independently timed."""
        from .feedback import FeedbackSample

        episode_time = float(sample_time)
        if not math.isfinite(episode_time) or episode_time < 0:
            raise ValueError("sample_time must be finite and non-negative episode seconds")
        now = _finite_time(now_monotonic, name="now_monotonic")
        from .feedback import FEEDBACK_GROUPS

        valid: dict[str, bool] = {key: False for key in FEEDBACK_GROUPS}
        source_times: dict[str, float] = {}
        values: dict[str, Any] = {
            "accel_mps2": None,
            "gyro_rps": None,
            "mag_counts": None,
            "temperature_c": None,
            "attitude_euler_rad": None,
            "attitude_quat_xyzw": None,
        }

        def source_episode_time(received: float, timeout: float) -> float | None:
            age = now - received
            if age < 0 or age > timeout:
                return None
            result = episode_time - age
            # Float subtraction at an episode's first sample can yield a tiny
            # negative value (for example 0.05 - (20.05 - 20.0)).
            if result >= -1e-9:
                return max(0.0, result)
            return None

        for group, field_names in (
            ("accel", ("accel_mps2",)),
            ("gyro", ("gyro_rps",)),
            ("mag", ("mag_counts",)),
        ):
            cached = self._imu_latest.get(group)
            if cached is None:
                continue
            cached_values, received = cached
            source_time = source_episode_time(received, self.sensor_timeout_s)
            if source_time is None:
                continue
            for field in field_names:
                values[field] = list(cached_values[field]) if isinstance(cached_values[field], list) else cached_values[field]
            if group == "mag" and "temperature_c" in cached_values:
                # Temperature remains available to callers for diagnostics only.
                values["temperature_c"] = float(cached_values["temperature_c"])
            valid[group] = True
            source_times[group] = source_time

        euler_time = None if self._attitude_euler is None else self._attitude_euler[1]
        quat_time = None if self._attitude_quat is None else self._attitude_quat[1]
        if euler_time is not None and source_episode_time(euler_time, self.sensor_timeout_s) is not None:
            values["attitude_euler_rad"] = list(self._attitude_euler[0])
        if quat_time is not None:
            attitude_source_time = source_episode_time(quat_time, self.sensor_timeout_s)
            if attitude_source_time is not None:
                values["attitude_quat_xyzw"] = list(self._attitude_quat[0])
                valid["attitude"] = True
                source_times["attitude"] = attitude_source_time

        q_values = [0.0, 0.0]
        qd_values = [0.0, 0.0]
        q_any = False
        for index, axis in enumerate(("yaw", "pitch")):
            group = f"gimbal_{axis}"
            cached = self._gimbal.get(axis)
            if cached is None:
                continue
            q_rad, qd_rad_s, received = cached
            source_time = source_episode_time(received, self.sensor_timeout_s)
            if source_time is None:
                continue
            q_values[index] = q_rad
            qd_values[index] = qd_rad_s
            valid[group] = True
            source_times[group] = source_time
            q_any = True
        if q_any:
            values["gimbal_q_rad"] = q_values
            values["gimbal_qd_rps"] = qd_values

        iq_values = [0.0, 0.0]
        rpm_values = [0.0, 0.0]
        motor_any = False
        for index, side in enumerate(("left", "right")):
            pair = self.get_motor_feedback(side, now)
            if pair is None or not pair["valid"]:
                continue
            motor_values = pair["values"]
            iq_values[index] = float(motor_values["iq_A"])
            rpm_values[index] = float(motor_values["mechanical_rpm"])
            valid[f"motor_{side}"] = True
            pair_time = min(
                pair["timestamps"]["state_host_receive_monotonic_s"],
                pair["timestamps"]["quality_host_receive_monotonic_s"],
            )
            source_time = source_episode_time(pair_time, self.motor_timeout_s)
            if source_time is None:
                valid[f"motor_{side}"] = False
                continue
            source_times[f"motor_{side}"] = source_time
            motor_any = True
        if motor_any:
            values["motor_iq_a"] = iq_values
            values["motor_rpm"] = rpm_values

        return FeedbackSample(
            sample_time=episode_time,
            **values,
            valid=valid,
            source_times=source_times,
        )


def make_hardware_feedback_estimator(*, seed: int | None = None, **kwargs: Any):
    """Construct the existing feedback estimator with hardware noise disabled."""
    from .feedback import SimFeedbackEstimator

    if "noise_enabled" in kwargs:
        raise ValueError("hardware telemetry always disables simulated sensor noise")
    return SimFeedbackEstimator(seed=seed, noise_enabled=False, **kwargs)
