import math

import pytest

from isaac_field.body_stability import (
    BodyStabilityAccumulator, BodyStabilityConfig, body_stability_sample,
)


def test_steady_slope_is_not_a_horizontal_attitude_tracking_target():
    sample = body_stability_sample(0, math.radians(25), (0, 0, 0), .05,
                                   BodyStabilityConfig())
    assert sample["tilt_rad"] == pytest.approx(math.radians(25))
    assert sample["penalty"] == 0


def test_combined_roll_and_pitch_can_exceed_soft_limit():
    sample = body_stability_sample(math.pi / 6, math.pi / 6, (0, 0, 0), .05,
                                   BodyStabilityConfig())
    assert sample["tilt_rad"] == pytest.approx(math.acos(.75))
    assert sample["excess_tilt_rad"] > 0
    assert sample["penalty"] > 0


def test_rate_penalty_excludes_pure_yaw_and_integrates_time():
    cfg = BodyStabilityConfig()
    assert body_stability_sample(0, 0, (0, 0, 10), .05, cfg)["penalty"] == 0
    short = body_stability_sample(0, 0, (.8, -.6, 10), .05, cfg)
    long = body_stability_sample(0, 0, (.8, -.6, 10), .1, cfg)
    assert short["penalty"] == pytest.approx(.001)
    assert long["penalty"] == pytest.approx(2 * short["penalty"])
    assert body_stability_sample(1, 1, (2, 3, 4), .05)["penalty"] == 0


def test_episode_statistics_are_time_weighted_and_reset_with_a_new_accumulator():
    stats = BodyStabilityAccumulator()
    stats.update(body_stability_sample(0, 0, (1, 0, 0), .1), .1)
    stats.update(body_stability_sample(0, 0, (0, 0, 0), .3), .3)
    assert stats.summary()["roll_pitch_angular_rate_rms_rad_s"] == pytest.approx(.5)
    assert stats.summary()["sample_count"] == 2
    assert BodyStabilityAccumulator().summary()["duration_s"] == 0


@pytest.mark.parametrize("roll,pitch,rates,dt", [
    (float("nan"), 0, (0, 0, 0), .05),
    (0, 0, (0, float("inf"), 0), .05),
    (0, 0, (0, 0), .05),
    (0, 0, (0, 0, 0), 0),
])
def test_invalid_physical_samples_fail_explicitly(roll, pitch, rates, dt):
    with pytest.raises(ValueError):
        body_stability_sample(roll, pitch, rates, dt)
