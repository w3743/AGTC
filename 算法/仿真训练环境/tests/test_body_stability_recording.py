"""Trajectory recording contract for simulator-truth body-stability data."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaac_field.learning import _trajectory_task_snapshot


def _task(**extra):
    values = {
        "robot": np.asarray([1.0, 2.0, 0.3, 0.4, -0.2], dtype=np.float64),
        "goal": np.asarray([4.0, 6.0], dtype=np.float64),
        "time": 1.25,
        "damaged": set(),
        "collisions": 2,
        "_hold": 0.15,
    }
    values.update(extra)
    return SimpleNamespace(**values)


def test_snapshot_maps_z_pitch_roll_in_meters_and_radians_and_keeps_all_true_angular_rates():
    snapshot = _trajectory_task_snapshot(
        _task(
            body_pose=(0.37, -0.62, 0.81),
            body_angular_velocity=(1.2, -2.3, 3.4),
        )
    )

    assert snapshot["body_z_m"] == pytest.approx(0.37)
    assert snapshot["body_pitch_rad"] == pytest.approx(-0.62)
    assert snapshot["body_roll_rad"] == pytest.approx(0.81)
    assert snapshot["body_angular_velocity_rad_s"] == pytest.approx([1.2, -2.3, 3.4])
    assert len(snapshot["body_angular_velocity_rad_s"]) == 3


def test_snapshot_remains_compatible_when_body_stability_fields_are_missing():
    snapshot = _trajectory_task_snapshot(_task())

    assert snapshot["pose_xy_yaw"] == pytest.approx([1.0, 2.0, 0.3])
    assert snapshot["actual_v_w"] == pytest.approx([0.4, -0.2])
    assert "body_z_m" not in snapshot
    assert "body_pitch_rad" not in snapshot
    assert "body_roll_rad" not in snapshot
    assert "body_angular_velocity_rad_s" not in snapshot


@pytest.mark.parametrize(
    "body_pose",
    [
        (float("nan"), 0.1, -0.2),
        (0.3, float("inf"), -0.2),
        (0.3, 0.1, float("-inf")),
    ],
)
def test_snapshot_omits_non_finite_body_pose_as_a_group(body_pose):
    snapshot = _trajectory_task_snapshot(_task(body_pose=body_pose))

    assert not {"body_z_m", "body_pitch_rad", "body_roll_rad"}.intersection(snapshot)


@pytest.mark.parametrize(
    "body_angular_velocity",
    [
        (float("nan"), 0.2, 0.3),
        (0.1, float("inf"), 0.3),
        (0.1, 0.2, float("-inf")),
    ],
)
def test_snapshot_omits_non_finite_body_angular_velocity(body_angular_velocity):
    snapshot = _trajectory_task_snapshot(_task(body_angular_velocity=body_angular_velocity))

    assert "body_angular_velocity_rad_s" not in snapshot
