import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from isaac_field.feedback import FEEDBACK_GROUPS, FEEDBACK_VALID_SLICE, FEEDBACK_VALUE_SLICES
from isaac_field.telemetry import (
    ATKIMU901StreamDecoder,
    DRIVE_INVALID_I16,
    DRIVE_QUALITY_APPLIED_COMMAND_VALID,
    DRIVE_QUALITY_HALL_VALID,
    DRIVE_QUALITY_IQ_VALID,
    DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED,
    DRIVE_QUALITY_RPM_VALID,
    TelemetryAssembler,
    TelemetryProtocolError,
    decode_drive_feedback_frame,
    drive_crc16_ccitt_false,
    encode_speed_command,
    make_hardware_feedback_estimator,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
GOLDEN_PATH = FIXTURE_DIR / "foc_drive_feedback_v1_golden_vectors.json"
PROVENANCE_PATH = FIXTURE_DIR / "foc_drive_feedback_v1_golden_vectors.provenance.json"
ATK_G = 9.80665


def _i16(value: int) -> bytes:
    return int(value).to_bytes(2, "little", signed=True)


def _atk_packet(frame_id: int, payload: bytes, *, bad_checksum: bool = False) -> bytes:
    packet = b"\x55\x55" + bytes((frame_id, len(payload))) + payload
    checksum = sum(packet) & 0xFF
    if bad_checksum:
        checksum ^= 0x01
    return packet + bytes((checksum,))


def _can_frame(can_id: int, first_six: bytes) -> bytes:
    assert len(first_six) == 6
    crc = drive_crc16_ccitt_false(can_id, 8, first_six)
    return first_six + crc.to_bytes(2, "little")


def _boot_frame(can_id: int, epoch: int) -> bytes:
    return _can_frame(can_id, struct.pack("<IH", epoch, 0))


def _state_frame(can_id: int, sequence: int, rpm: int, iq_centiamp: int) -> bytes:
    return _can_frame(can_id, struct.pack("<Hhh", sequence, rpm, iq_centiamp))


def _quality_frame(can_id: int, sequence: int, tick: int, flags: int, applied: int) -> bytes:
    return _can_frame(can_id, struct.pack("<HHBB", sequence, tick, flags, applied))


def _flags(*, rpm=True, iq=True, low_speed=False, applied=True):
    value = DRIVE_QUALITY_HALL_VALID
    if rpm:
        value |= DRIVE_QUALITY_RPM_VALID
    if iq:
        value |= DRIVE_QUALITY_IQ_VALID
    if low_speed:
        value |= DRIVE_QUALITY_LOW_SPEED_OR_REVERSAL_DEGRADED
    if applied:
        value |= DRIVE_QUALITY_APPLIED_COMMAND_VALID
    return value


def _feed_pair(assembler, side, sequence, *, rpm=1200, iq_centiamp=1250,
               tick=1000, flags=None, applied=7, state_time=1.0,
               quality_time=None):
    is_left = side == "left"
    state_id = 0x521 if is_left else 0x522
    quality_id = 0x541 if is_left else 0x542
    quality_time = state_time if quality_time is None else quality_time
    if flags is None:
        flags = _flags()
    assembler.feed_can_frame(state_id, _state_frame(state_id, sequence, rpm, iq_centiamp),
                             received_monotonic=state_time)
    return assembler.feed_can_frame(
        quality_id,
        _quality_frame(quality_id, sequence, tick, flags, applied if flags & DRIVE_QUALITY_APPLIED_COMMAND_VALID else 0xFF),
        received_monotonic=quality_time,
    )


def test_c_golden_vectors_crc_and_speed_command_interoperate():
    fixture_bytes = GOLDEN_PATH.read_bytes()
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture_bytes).hexdigest().upper() == provenance["sha256"]
    vectors = json.loads(fixture_bytes)
    assert vectors["protocol"] == "foc-drive-feedback-v1"

    for vector in vectors["vectors"]:
        can_id = int(vector["id"], 16)
        data = bytes.fromhex(vector["data_hex"])
        assert vector["dlc"] == 8
        assert len(data) == 8
        assert drive_crc16_ccitt_false(can_id, 8, data[:6]) == int.from_bytes(data[6:], "little")

    left = encode_speed_command("left", session_id=0x1234, sequence=0x7E,
                                target_motor_rpm=1200, ttl_10ms=10)
    right = encode_speed_command(2, session_id=0xBEEF, sequence=0,
                                 target_motor_rpm=-1500, ttl_10ms=5)
    by_name = {item["name"]: item for item in vectors["vectors"]}
    assert left["can_id"] == int(by_name["speed_left_forward"]["id"], 16)
    assert left["data"] == bytes.fromhex(by_name["speed_left_forward"]["data_hex"])
    assert right["can_id"] == int(by_name["speed_right_reverse"]["id"], 16)
    assert right["data"] == bytes.fromhex(by_name["speed_right_reverse"]["data_hex"])
    assert left["extended"] is False and left["remote"] is False

    boot = by_name["boot_epoch_left"]
    decoded_boot = decode_drive_feedback_frame(
        int(boot["id"], 16), bytes.fromhex(boot["data_hex"]), received_monotonic=1.0
    )
    assert decoded_boot["values"]["boot_epoch"] == boot["meaning"]["boot_epoch"]
    state = by_name["state_left"]
    decoded_state = decode_drive_feedback_frame(
        int(state["id"], 16), bytes.fromhex(state["data_hex"]), received_monotonic=1.0
    )
    assert decoded_state["values"]["mechanical_rpm_i16"] == state["meaning"]["mechanical_rpm"]
    assert decoded_state["values"]["iq_centiamp"] == round(state["meaning"]["iq_A"] * 100)
    quality = by_name["quality_left"]
    decoded_quality = decode_drive_feedback_frame(
        int(quality["id"], 16), bytes.fromhex(quality["data_hex"]), received_monotonic=1.0
    )
    assert decoded_quality["values"]["quality_flags"] == quality["meaning"]["quality_flags"]
    assert decoded_quality["values"]["quality"]["low_speed_or_reversal_degraded"] is True


def test_atk_known_packets_convert_units_preserve_quaternion_and_ignore_pressure():
    euler_payload = _i16(32767) + _i16(-16384) + _i16(-8192)
    quat_payload = _i16(-16384) + _i16(8192) + _i16(-4096) + _i16(16384)
    imu_payload = b"".join(_i16(v) for v in (16384, -8192, 32767, -16384, 8192, -32768))
    mag_payload = b"".join(_i16(v) for v in (123, -456, 32767, -1234))
    ports_payload = struct.pack("<HHHH", 0, 1, 4096, 65535)
    pressure_payload = struct.pack("<Iih", 101325, -123, 2500)
    stream = b"".join((
        _atk_packet(0x01, euler_payload),
        _atk_packet(0x02, quat_payload),
        _atk_packet(0x03, imu_payload),
        _atk_packet(0x04, mag_payload),
        _atk_packet(0x05, pressure_payload),
        _atk_packet(0x06, ports_payload),
    ))
    decoder = ATKIMU901StreamDecoder()
    frames = decoder.feed(stream, received_monotonic=12.5)
    assert [frame["group"] for frame in frames] == [
        "attitude", "attitude", "accel", "gyro", "mag_temperature", "ports"
    ]

    euler = frames[0]["values"]["attitude_euler_rad"]
    np.testing.assert_allclose(euler, [32767 / 32768 * math.pi, -math.pi / 2, -math.pi / 4])
    quaternion = frames[1]["values"]["attitude_quat_xyzw"]
    np.testing.assert_allclose(quaternion, [0.25, -0.125, 0.5, -0.5])
    assert frames[1]["values"]["attitude_quat_wxyz"] == [-0.5, 0.25, -0.125, 0.5]

    accel = next(frame for frame in frames if frame["group"] == "accel")
    gyro = next(frame for frame in frames if frame["group"] == "gyro")
    np.testing.assert_allclose(accel["values"]["accel_mps2"], [2 * ATK_G, -ATK_G, 32767 / 32768 * 4 * ATK_G])
    np.testing.assert_allclose(gyro["values"]["gyro_rps"], np.deg2rad([-1000, 500, -2000]))
    mag = next(frame for frame in frames if frame["group"] == "mag_temperature")["values"]
    assert mag["mag_counts"] == [123, -456, 32767]
    assert mag["temperature_c"] == pytest.approx(-12.34)
    port_frame = next(frame for frame in frames if frame["group"] == "ports")
    assert port_frame["values"]["ports"] == [0, 1, 4096, 65535]
    assert port_frame["diagnostic_only"] is True
    assert all(frame["timestamp"] == 12.5 for frame in frames)
    assert decoder.stats["ignored_pressure_frames"] == 1
    assert all(frame["frame_id"] != 0x05 for frame in frames)


def test_atk_decoder_handles_fragmentation_bad_checksum_bad_length_noise_and_flush():
    valid = _atk_packet(0x04, b"".join(_i16(value) for value in (1, 2, 3, 400)))
    decoder = ATKIMU901StreamDecoder()
    assert decoder.feed(b"noise\x54\x55" + valid[:4], received_monotonic=1.0) == []
    assert decoder.buffered_bytes <= 17
    result = decoder.feed(valid[4:], received_monotonic=1.1)
    assert len(result) == 1
    assert result[0]["values"]["mag_counts"] == [1, 2, 3]
    assert result[0]["timestamp"] == 1.1

    malformed_length = b"\x55\x55\x04\x07" + b"\x00" * 9
    recovered = decoder.feed(malformed_length + _atk_packet(0x01, _i16(0) * 3),
                             received_monotonic=2.0)
    assert len(recovered) == 1 and recovered[0]["frame_id"] == 0x01
    assert decoder.stats["bad_lengths"] >= 1

    corrupted = _atk_packet(0x01, _i16(1) * 3, bad_checksum=True)
    recovered = decoder.feed(corrupted + _atk_packet(0x01, _i16(2) * 3),
                             received_monotonic=3.0)
    assert len(recovered) == 1
    assert recovered[0]["values"]["attitude_euler_rad"][0] == pytest.approx(2 / 32768 * math.pi)
    assert decoder.stats["bad_checksums"] == 1

    assert decoder.feed(b"\x55\x55\x01", received_monotonic=4.0) == []
    assert decoder.flush() == 3
    assert decoder.stats["truncated_frames"] == 1


def test_zero_factory_quaternion_is_marked_invalid_and_clears_attitude_cache():
    assembler = TelemetryAssembler()
    assembler.feed_imu_bytes(_atk_packet(0x01, _i16(0) * 3) +
                              _atk_packet(0x02, _i16(0) * 4),
                              received_monotonic=1.0)
    sample = assembler.assemble(0.1, now_monotonic=1.0)
    assert sample.valid["attitude"] is False
    assert sample.attitude_euler_rad is None
    assert sample.attitude_quat_xyzw is None


def test_drive_frame_validation_rejects_bad_crc_dlc_rtr_extended_and_bad_ids():
    boot_data = _boot_frame(0x4D1, 0x12345678)
    with pytest.raises(TelemetryProtocolError, match="CRC"):
        decode_drive_feedback_frame(0x4D1, boot_data[:-1] + bytes((boot_data[-1] ^ 0x80,)))
    with pytest.raises(TelemetryProtocolError, match="DLC"):
        decode_drive_feedback_frame(0x4D1, boot_data[:-1])
    with pytest.raises(TelemetryProtocolError, match="extended"):
        decode_drive_feedback_frame(0x4D1, boot_data, extended=True)
    with pytest.raises(TelemetryProtocolError, match="remote"):
        decode_drive_feedback_frame(0x4D1, boot_data, remote=True)
    for can_id in (-1, 0x800, True):
        with pytest.raises(TelemetryProtocolError, match="11-bit"):
            decode_drive_feedback_frame(can_id, boot_data)
    with pytest.raises(TelemetryProtocolError, match="boot_epoch"):
        decode_drive_feedback_frame(0x4D1, _boot_frame(0x4D1, 0))
    assert decode_drive_feedback_frame(0x123, b"\0" * 8) is None

    for node in (True, 1.0, 0, 3, "center"):
        with pytest.raises(TelemetryProtocolError):
            encode_speed_command(node, session_id=1, sequence=0,
                                 target_motor_rpm=0, ttl_10ms=1)
    for rpm, ttl in ((-32768, 1), (32768, 1), (0, 0), (0, 11)):
        with pytest.raises(TelemetryProtocolError):
            encode_speed_command("left", session_id=1, sequence=0,
                                 target_motor_rpm=rpm, ttl_10ms=ttl)


def test_drive_pairing_requires_boot_and_exact_sample_sequence_and_expires_over_100ms():
    assembler = TelemetryAssembler()
    state_id, quality_id = 0x521, 0x541
    state = _state_frame(state_id, 10, 100, 200)
    quality = _quality_frame(quality_id, 11, 5, _flags(), 3)
    with pytest.raises(TelemetryProtocolError, match="boot epoch"):
        assembler.feed_can_frame(state_id, state, received_monotonic=1.0)

    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 123), received_monotonic=1.0)
    assembler.feed_can_frame(state_id, state, received_monotonic=1.01)
    assembler.feed_can_frame(quality_id, quality, received_monotonic=1.02)
    assert assembler.get_motor_feedback("left", 1.02) is None
    # A later matching quality frame completes only the exact sequence pair.
    matching = _quality_frame(quality_id, 10, 6, _flags(), 3)
    assembler.feed_can_frame(quality_id, matching, received_monotonic=1.03)
    pair = assembler.get_motor_feedback("left", 1.03)
    assert pair["values"]["sample_sequence"] == 10
    assert pair["values"]["boot_epoch"] == 123
    assert assembler.stats["paired_drive_samples"] == 1

    assembler2 = TelemetryAssembler()
    assembler2.feed_can_frame(0x4D1, _boot_frame(0x4D1, 124), received_monotonic=2.0)
    _feed_pair(assembler2, "left", 1, state_time=2.0, quality_time=2.100001)
    assert assembler2.get_motor_feedback("left", 2.2) is None
    assert assembler2.stats["expired_drive_pairs"] == 1


def test_drive_duplicates_out_of_order_tick_wrap_and_quality_bits_are_independent():
    assembler = TelemetryAssembler()
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 1), received_monotonic=10.0)
    _feed_pair(assembler, "left", 0xFFFF, rpm=1000, iq_centiamp=250,
               tick=0xFFFA, state_time=10.0)
    first_pair = assembler.get_motor_feedback("left", 10.0)
    assert first_pair["valid"] is True

    duplicate = assembler.feed_can_frame(
        0x541, _quality_frame(0x541, 0xFFFF, 0xFFFA, _flags(), 7),
        received_monotonic=10.001,
    )
    assert duplicate["duplicate"] is True
    assert assembler.stats["duplicate_drive_frames"] == 1

    _feed_pair(assembler, "left", 0, rpm=900, iq_centiamp=-425,
               tick=3, state_time=10.01)
    assert assembler.last_tick_delta_ms["left"] == 9
    assert assembler.get_motor_feedback("left", 10.01)["values"]["sample_sequence"] == 0

    # RPM remains valid while Iq is invalid; low-speed warning is independent
    # of both measurement-valid bits and never invalidates them on its own.
    invalid_iq_flags = _flags(rpm=True, iq=False, low_speed=True)
    _feed_pair(assembler, "left", 1, rpm=700, iq_centiamp=DRIVE_INVALID_I16,
               tick=9, flags=invalid_iq_flags, state_time=10.02)
    partial = assembler.get_motor_feedback("left", 10.02)
    assert partial["valid"] is False
    assert partial["values"]["quality"]["rpm_valid"] is True
    assert partial["values"]["quality"]["iq_valid"] is False
    assert partial["values"]["quality"]["low_speed_or_reversal_degraded"] is True
    assert partial["values"]["mechanical_rpm"] == 700
    assert partial["values"]["iq_A"] is None

    # The low-speed/reversal bit alone is only a warning; it leaves both
    # measurement-valid flags and the atomic motor pair valid.
    warning_only = TelemetryAssembler()
    warning_only.feed_can_frame(0x4D1, _boot_frame(0x4D1, 1), received_monotonic=10.0)
    _feed_pair(warning_only, "left", 1, rpm=20, iq_centiamp=-10,
               flags=_flags(low_speed=True), state_time=10.0)
    warning_pair = warning_only.get_motor_feedback("left", 10.0)
    assert warning_pair["valid"] is True
    assert warning_pair["values"]["quality"]["rpm_valid"] is True
    assert warning_pair["values"]["quality"]["iq_valid"] is True
    assert warning_pair["values"]["quality"]["low_speed_or_reversal_degraded"] is True

    # A matching but older sequence can be paired inside the reorder window,
    # but it cannot replace the newest sample.
    _feed_pair(assembler, "left", 3, rpm=111, iq_centiamp=222,
               tick=10, state_time=10.03)
    _feed_pair(assembler, "left", 2, rpm=112, iq_centiamp=223,
               tick=11, state_time=10.04)
    latest = assembler.get_motor_feedback("left", 10.04)
    assert latest["values"]["sample_sequence"] == 3
    assert assembler.stats["out_of_order_drive_frames"] >= 1


def test_conflicting_duplicates_and_sequences_outside_reorder_window_are_rejected():
    assembler = TelemetryAssembler(reorder_window=4, seen_window=8)
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 1), received_monotonic=1.0)
    first = _state_frame(0x521, 100, 10, 20)
    assembler.feed_can_frame(0x521, first, received_monotonic=1.0)
    with pytest.raises(TelemetryProtocolError, match="conflicting duplicate"):
        assembler.feed_can_frame(0x521, _state_frame(0x521, 100, 11, 20),
                                 received_monotonic=1.001)
    with pytest.raises(TelemetryProtocolError, match="stale"):
        assembler.feed_can_frame(0x521, _state_frame(0x521, 95, 10, 20),
                                 received_monotonic=1.002)

    # A changed boot epoch fences the previous sequence window.
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 2), received_monotonic=1.003)
    accepted_after_boot = assembler.feed_can_frame(
        0x521, _state_frame(0x521, 95, 10, 20), received_monotonic=1.004
    )
    assert accepted_after_boot["values"]["sample_sequence"] == 95


def test_sentinels_must_agree_with_quality_bits_and_invalid_applied_seq_is_ff():
    assembler = TelemetryAssembler()
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 7), received_monotonic=1.0)
    state_id, quality_id = 0x521, 0x541
    assembler.feed_can_frame(state_id, _state_frame(state_id, 1, DRIVE_INVALID_I16, 20),
                             received_monotonic=1.0)
    flags = _flags(rpm=True, iq=True)
    with pytest.raises(TelemetryProtocolError, match="sentinel"):
        assembler.feed_can_frame(quality_id, _quality_frame(quality_id, 1, 1, flags, 3),
                                 received_monotonic=1.0)

    bad_applied = _quality_frame(quality_id, 2, 2, 0, 3)
    with pytest.raises(TelemetryProtocolError, match="0xFF"):
        decode_drive_feedback_frame(quality_id, bad_applied, received_monotonic=1.0)


class _PassThroughVQF:
    def __init__(self, dt):
        self.dt = dt
        self.gyr = []
        self.acc = []
        self.mag = []

    def updateGyr(self, value):
        self.gyr.append(np.asarray(value).copy())

    def updateAcc(self, value):
        self.acc.append(np.asarray(value).copy())

    def updateMag(self, value):
        self.mag.append(np.asarray(value).copy())

    def getQuat6D(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def getQuat9D(self):
        return self.getQuat6D()


def test_assembler_preserves_raw_measurements_factory_attitude_gimbal_and_drive_in_feedbacksample():
    assembler = TelemetryAssembler()
    received = 100.0
    euler_counts = (2048, -1024, 8192)
    quat_counts_wxyz = (30000, -2000, 4000, -6000)
    accel_gyro_counts = (8192, -4096, 16384, 100, -200, 300)
    mag_counts = (-10, 20, 30, 2250)
    packets = b"".join((
        _atk_packet(0x01, b"".join(_i16(v) for v in euler_counts)),
        _atk_packet(0x02, b"".join(_i16(v) for v in quat_counts_wxyz)),
        _atk_packet(0x03, b"".join(_i16(v) for v in accel_gyro_counts)),
        _atk_packet(0x04, b"".join(_i16(v) for v in mag_counts)),
        _atk_packet(0x05, struct.pack("<Iih", 100000, 1234, 2500)),
    ))
    assembler.feed_imu_bytes(packets, received_monotonic=received)
    assembler.update_gimbal_axis("yaw", 0.25, -0.5, received_monotonic=received + 0.01)
    assembler.update_gimbal_axis("pitch", -0.75, 0.125, received_monotonic=received + 0.01)
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 0xAABBCCDD),
                             received_monotonic=received + 0.01)
    assembler.feed_can_frame(0x4D2, _boot_frame(0x4D2, 0xAABBCCDE),
                             received_monotonic=received + 0.01)
    _feed_pair(assembler, "left", 1, rpm=1234, iq_centiamp=-425,
               tick=10, state_time=received + 0.02)
    _feed_pair(assembler, "right", 1, rpm=-2345, iq_centiamp=678,
               tick=11, state_time=received + 0.02)

    sample = assembler.assemble(sample_time=0.05, now_monotonic=received + 0.05)
    assert sample.sample_time == pytest.approx(0.05)
    assert sample.valid == {group: True for group in FEEDBACK_GROUPS}
    np.testing.assert_allclose(sample.accel_mps2,
                               np.array(accel_gyro_counts[:3]) / 32768 * 4 * ATK_G)
    np.testing.assert_allclose(sample.gyro_rps,
                               np.array(accel_gyro_counts[3:]) / 32768 * 2000 * math.pi / 180)
    assert list(sample.mag_counts) == [-10, 20, 30]
    assert sample.temperature_c == pytest.approx(22.5)
    np.testing.assert_allclose(sample.attitude_euler_rad,
                               np.array(euler_counts) / 32768 * math.pi)
    expected_quat_xyzw = np.array(quat_counts_wxyz[1:] + quat_counts_wxyz[:1]) / 32768
    np.testing.assert_allclose(sample.attitude_quat_xyzw, expected_quat_xyzw)
    np.testing.assert_allclose(sample.gimbal_q_rad, [0.25, -0.75])
    np.testing.assert_allclose(sample.gimbal_qd_rps, [-0.5, 0.125])
    np.testing.assert_allclose(sample.motor_iq_a, [-4.25, 6.78])
    np.testing.assert_allclose(sample.motor_rpm, [1234, -2345])
    assert sample.source_times["accel"] == pytest.approx(0.0)
    assert sample.source_times["attitude"] == pytest.approx(0.0)
    assert sample.source_times["mag"] == pytest.approx(0.0)
    assert sample.source_times["motor_left"] == pytest.approx(0.02)

    estimator = make_hardware_feedback_estimator(seed=42, vqf_factory=_PassThroughVQF)
    assert estimator.noise_enabled is False
    snapshot = estimator.update_sample(sample)
    np.testing.assert_allclose(snapshot.feedback[FEEDBACK_VALUE_SLICES["attitude"]],
                               expected_quat_xyzw)
    np.testing.assert_allclose(snapshot.feedback[FEEDBACK_VALUE_SLICES["mag"]],
                               np.array([-10, 20, 30]) / 32768)
    np.testing.assert_array_equal(snapshot.feedback[FEEDBACK_VALID_SLICE], np.ones(8))


def test_assembler_accepts_quaternion_without_euler_and_ignores_euler_only_for_policy_pose():
    quat_payload = b"".join(_i16(v) for v in (30000, -2000, 4000, -6000))
    euler_payload = b"".join(_i16(v) for v in (2048, -1024, 8192))

    quaternion_only = TelemetryAssembler()
    quaternion_only.feed_imu_bytes(_atk_packet(0x02, quat_payload), received_monotonic=5.0)
    quat_sample = quaternion_only.assemble(0.05, now_monotonic=5.0)
    assert quat_sample.valid["attitude"] is True
    assert quat_sample.attitude_euler_rad is None
    assert quat_sample.attitude_quat_xyzw is not None
    quat_estimator = make_hardware_feedback_estimator(seed=1, vqf_factory=_PassThroughVQF)
    quat_snapshot = quat_estimator.update_sample(quat_sample)
    assert quat_snapshot.feedback[FEEDBACK_VALID_SLICE.start + FEEDBACK_GROUPS.index("attitude")] == 1.0
    np.testing.assert_allclose(quat_snapshot.feedback[FEEDBACK_VALUE_SLICES["attitude"]],
                               quat_sample.attitude_quat_xyzw)

    euler_only = TelemetryAssembler()
    euler_only.feed_imu_bytes(_atk_packet(0x01, euler_payload), received_monotonic=6.0)
    euler_sample = euler_only.assemble(0.05, now_monotonic=6.0)
    assert euler_sample.valid["attitude"] is False
    assert euler_sample.attitude_euler_rad is not None  # retained for diagnostics
    assert euler_sample.attitude_quat_xyzw is None
    euler_estimator = make_hardware_feedback_estimator(seed=2, vqf_factory=_PassThroughVQF)
    euler_snapshot = euler_estimator.update_sample(euler_sample)
    assert euler_snapshot.feedback[FEEDBACK_VALID_SLICE.start + FEEDBACK_GROUPS.index("attitude")] == 0.0
    np.testing.assert_array_equal(euler_snapshot.feedback[FEEDBACK_VALUE_SLICES["attitude"]], np.zeros(4))
    assert euler_snapshot.pose_valid is False


def test_assembler_invalidates_expired_drive_and_does_not_substitute_command_values():
    assembler = TelemetryAssembler()
    assembler.feed_can_frame(0x4D1, _boot_frame(0x4D1, 1), received_monotonic=20.0)
    _feed_pair(assembler, "left", 1, rpm=500, iq_centiamp=200,
               state_time=20.0)
    fresh = assembler.assemble(0.05, now_monotonic=20.05)
    assert fresh.valid["motor_left"] is True
    expired = assembler.assemble(0.2, now_monotonic=20.101)
    assert expired.valid["motor_left"] is False
    assert expired.motor_iq_a is None
    assert expired.motor_rpm is None

