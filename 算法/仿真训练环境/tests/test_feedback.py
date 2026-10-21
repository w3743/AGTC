import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from isaac_field.feedback import (
    FEEDBACK_AGE_SLICE,
    FEEDBACK_DIM,
    FEEDBACK_GROUPS,
    FEEDBACK_LABELS,
    FEEDBACK_VALID_SLICE,
    FEEDBACK_VALUE_SLICES,
    SIM_SENSOR_MODEL,
    SimFeedbackEstimator,
    FeedbackSample,
    feedback_config,
)


class _FixedVQF:
    def __init__(self, dt, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.dt = dt
        self.quat = np.asarray(quat_wxyz, dtype=np.float64)
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
        return self.quat.copy()

    def getQuat9D(self):
        return self.quat.copy()


def _estimator(quat_wxyz=(1.0, 0.0, 0.0, 0.0), **kwargs):
    factory = lambda dt: _FixedVQF(dt, quat_wxyz)
    return SimFeedbackEstimator(seed=123, noise_enabled=False, vqf_factory=factory, **kwargs)


def _full_sample(sample_time=0.05, **overrides):
    fields = dict(
        sample_time=sample_time,
        accel_mps2=[19.6133, 0.0, 0.0],
        gyro_rps=[math.pi, 0.0, 0.0],
        mag_counts=[0.0, 32768.0, -32768.0],
        temperature_c=25.0,
        gimbal_q_rad=[math.pi / 2, math.pi / 4],
        gimbal_qd_rps=[math.pi / 4, math.pi / 2],
        motor_iq_a=[250.0, -150.0],
        motor_rpm=[16000.0, -15000.0],
    )
    fields.update(overrides)
    return FeedbackSample(**fields)


def test_feedback_contract_scale_order_and_valid_static_values():
    estimator = _estimator()
    snapshot = estimator.update_sample(_full_sample())
    vector = snapshot.feedback

    assert vector.shape == (FEEDBACK_DIM,) == (37,)
    assert len(FEEDBACK_LABELS) == FEEDBACK_DIM
    assert FEEDBACK_VALID_SLICE == slice(21, 29)
    assert FEEDBACK_AGE_SLICE == slice(29, 37)
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["accel"]], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["gyro"]], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["mag"]], [0.0, 1.0, -1.0])
    # Sensor quaternion is scalar-first in VQF, policy contract is xyzw only.
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["attitude"]], [0, 0, 0, 1])
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["gimbal_yaw"]], [.5, .25])
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["gimbal_pitch"]], [.25, .5])
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["motor_left"]], [2.5, 1.6])
    np.testing.assert_allclose(vector[FEEDBACK_VALUE_SLICES["motor_right"]], [-1.5, -1.5])
    np.testing.assert_array_equal(vector[FEEDBACK_VALID_SLICE], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(vector[FEEDBACK_AGE_SLICE], np.zeros(8, dtype=np.float32))
    assert snapshot.pose_valid
    assert SIM_SENSOR_MODEL["filter_rate_hz"] == 20.0
    assert feedback_config()["dimension"] == 37
    assert feedback_config()["input_schema"] == "sensor-minimal-v6"
    assert feedback_config()["schema"] == "gimbal-imu-drive-feedback-v2"
    assert feedback_config()["simulation_assumptions"]["motor_iq_semantics"].find("not measured") >= 0


def test_magnetometer_does_not_require_temperature_and_vqf_gets_scaled_mag_only():
    estimator = _estimator()
    snapshot = estimator.update_sample(_full_sample(temperature_c=None))

    np.testing.assert_allclose(snapshot.feedback[FEEDBACK_VALUE_SLICES["mag"]], [0.0, 1.0, -1.0])
    assert snapshot.feedback[FEEDBACK_VALID_SLICE.start + FEEDBACK_GROUPS.index("mag")] == 1.0
    np.testing.assert_allclose(estimator._vqf.mag[-1], [0.0, 1.0, -1.0])


def test_missing_and_stale_groups_are_zero_flagged_and_age_is_bounded():
    estimator = _estimator()
    empty = estimator.update_sample(FeedbackSample(sample_time=0.0))
    np.testing.assert_array_equal(empty.feedback[:21], np.zeros(21, dtype=np.float32))
    np.testing.assert_array_equal(empty.feedback[FEEDBACK_VALID_SLICE], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(empty.feedback[FEEDBACK_AGE_SLICE], np.ones(8, dtype=np.float32))
    assert not empty.pose_valid

    estimator.reset()
    estimator.update_sample(_full_sample(sample_time=0.0))
    held = estimator.update_sample(FeedbackSample(sample_time=0.1))
    assert held.feedback[FEEDBACK_VALID_SLICE.start + FEEDBACK_GROUPS.index("motor_left")] == 1.0
    np.testing.assert_allclose(held.feedback[FEEDBACK_VALUE_SLICES["motor_left"]], [2.5, 1.6])
    assert held.feedback[FEEDBACK_AGE_SLICE.start + FEEDBACK_GROUPS.index("motor_left")] == pytest.approx(.05)

    stale = estimator.update_sample(FeedbackSample(sample_time=0.4))
    motor_index = FEEDBACK_GROUPS.index("motor_left")
    np.testing.assert_array_equal(stale.feedback[FEEDBACK_VALUE_SLICES["motor_left"]], [0.0, 0.0])
    assert stale.feedback[FEEDBACK_VALID_SLICE.start + motor_index] == 0.0
    assert stale.feedback[FEEDBACK_AGE_SLICE.start + motor_index] == pytest.approx(.2)


def test_independent_source_timestamps_and_explicit_invalidity():
    estimator = _estimator()
    sample = _full_sample(
        sample_time=.05,
        attitude_euler_rad=[0.0, 0.0, 0.0],
        attitude_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        source_times={
            "accel": 0.0, "gyro": .04, "mag": .03,
            "attitude": .02, "gimbal_yaw": .01, "gimbal_pitch": .05,
            "motor_left": .02, "motor_right": .05,
        },
    )
    snap = estimator.update_sample(sample)
    ages = snap.feedback[FEEDBACK_AGE_SLICE]
    np.testing.assert_allclose(ages, [.025, .005, .01, .015, .02, 0.0, .015, 0.0], atol=1e-7)

    invalid = estimator.update_sample(FeedbackSample(
        sample_time=.10,
        valid={"motor_left": False},
        source_times={"motor_left": .08},
    ))
    i = FEEDBACK_GROUPS.index("motor_left")
    assert invalid.feedback[FEEDBACK_VALID_SLICE.start + i] == 0.0
    np.testing.assert_array_equal(invalid.feedback[FEEDBACK_VALUE_SLICES["motor_left"]], [0.0, 0.0])
    assert invalid.feedback[FEEDBACK_AGE_SLICE.start + i] == pytest.approx(.01)


def test_zero_proper_acceleration_is_a_valid_freefall_sample():
    estimator = _estimator()
    sample = _full_sample(accel_mps2=[0.0, 0.0, 0.0])
    snap = estimator.update_sample(sample)
    assert snap.feedback[FEEDBACK_VALID_SLICE.start] == 1.0
    np.testing.assert_array_equal(snap.feedback[FEEDBACK_VALUE_SLICES["accel"]], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(estimator._vqf.acc[-1], [0.0, 0.0, 0.0])


def test_pure_gimbal_rotation_and_mount_offset_cancel_from_body_attitude():
    yaw, pitch = 0.42, 0.31
    mount = Rotation.from_euler("xyz", [0.08, -0.04, 0.12])
    gimbal_rotation = (Rotation.from_euler("z", yaw)
                       * Rotation.from_euler("y", -pitch)
                       * mount)
    xyzw = gimbal_rotation.as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    estimator = _estimator(
        quat_wxyz=wxyz,
        config={"sensor_mount_rot_xyzw": mount.as_quat().tolist()},
    )
    snap = estimator.update_sample(_full_sample(
        gimbal_q_rad=[yaw, pitch],
        gimbal_qd_rps=[0.4, -0.2],
    ))
    assert snap.pose_valid
    assert snap.estimated_heading_rad == pytest.approx(0.0, abs=1e-6)
    assert snap.estimated_roll_rad == pytest.approx(0.0, abs=1e-6)
    assert snap.estimated_pitch_rad == pytest.approx(0.0, abs=1e-6)


def test_firmware_quaternion_only_is_valid_and_stales_without_vqf_or_truth_fallback():
    euler = np.array([0.12, -0.18, 0.37])
    quat = Rotation.from_euler("xyz", euler).as_quat()
    estimator = _estimator()
    direct = FeedbackSample(
        sample_time=.05,
        attitude_quat_xyzw=quat,
        gimbal_q_rad=[0.0, 0.0],
        gimbal_qd_rps=[0.0, 0.0],
        valid={"attitude": True},
    )
    snap = estimator.update_sample(direct)
    attitude_slice = FEEDBACK_VALUE_SLICES["attitude"]
    np.testing.assert_allclose(snap.feedback[attitude_slice], quat)
    assert snap.pose_valid

    stale = estimator.update_sample(FeedbackSample(
        sample_time=.5,
        accel_mps2=[0, 0, 9.8], gyro_rps=[0, 0, 0], mag_counts=[0, 32768, 0],
        gimbal_q_rad=[0, 0], gimbal_qd_rps=[0, 0],
        valid={"attitude": False},
    ))
    i = FEEDBACK_GROUPS.index("attitude")
    assert stale.feedback[FEEDBACK_VALID_SLICE.start + i] == 0.0
    np.testing.assert_array_equal(stale.feedback[attitude_slice], np.zeros(4, dtype=np.float32))
    assert not stale.pose_valid

    partial = estimator.update_sample(FeedbackSample(
        sample_time=.55,
        attitude_euler_rad=[0.0, 0.0, .2],
        valid={"attitude": True},
        source_times={"attitude": .53},
    ))
    assert partial.feedback[FEEDBACK_VALID_SLICE.start + i] == 0.0
    np.testing.assert_array_equal(partial.feedback[attitude_slice], np.zeros(4, dtype=np.float32))


def test_euler_only_packet_cannot_create_policy_quaternion_or_pose():
    estimator = _estimator()
    snap = estimator.update_sample(FeedbackSample(
        sample_time=.05,
        accel_mps2=[0.0, 0.0, 9.80665],
        gyro_rps=[0.0, 0.0, 0.2],
        mag_counts=[0.0, 32768.0, 0.0],
        attitude_euler_rad=[0.0, 0.0, 1.2],
        gimbal_q_rad=[0.0, 0.0],
        gimbal_qd_rps=[0.0, 0.0],
        valid={"attitude": True, "accel": True, "gyro": True, "mag": True,
               "gimbal_yaw": True, "gimbal_pitch": True},
    ))

    i = FEEDBACK_GROUPS.index("attitude")
    np.testing.assert_array_equal(snap.feedback[FEEDBACK_VALUE_SLICES["attitude"]], np.zeros(4))
    assert snap.feedback[FEEDBACK_VALID_SLICE.start + i] == 0.0
    assert not snap.pose_valid


def test_repeated_sample_time_is_idempotent_and_does_not_consume_rng():
    estimator = _estimator()
    first = estimator.update_sample(_full_sample(sample_time=.05))
    rng_state = estimator.rng.bit_generator.state
    second = estimator.update_sample(_full_sample(
        sample_time=.05, accel_mps2=[0, 0, 0], motor_iq_a=[0, 0], motor_rpm=[0, 0],
    ))
    assert second is first
    assert estimator.rng.bit_generator.state == rng_state
