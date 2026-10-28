"""CPU regressions for the v6 per-side integral drive controller."""
from __future__ import annotations

import pytest
import torch

from isaac_field.drive_control import (
    SideIntegralDriveController,
    broadcast_side_values,
    conditional_integral_step,
    side_mean,
)


SIDE_IDS = torch.tensor([0] * 7 + [1] * 7, dtype=torch.long)


def _reference_drive_update(target, actual, integral, side_ids, *, kp, ki, dt, limit):
    """Original public-helper path used as a frozen controller reference."""
    error = side_mean(target, side_ids) - side_mean(actual, side_ids)
    p_estimate = kp * error
    next_integral, blocked = conditional_integral_step(
        error, integral, p_estimate, ki=ki, dt=dt, limit=limit,
    )
    return error, p_estimate, next_integral, blocked


def test_side_mean_and_integral_feedforward_are_broadcast_to_the_correct_wheels():
    target = torch.tensor([[
        2., 4., 3., 5., 1., 2., 4., 10., 12., 14., 16., 18., 20., 22.,
    ]])
    actual = torch.tensor([[
        1., 2., 1., 4., 1., 1., 4., 9., 12., 13., 15., 18., 19., 22.,
    ]])
    torch.testing.assert_close(side_mean(target, SIDE_IDS), torch.tensor([[3., 16.]]))
    torch.testing.assert_close(side_mean(actual, SIDE_IDS), torch.tensor([[2., 15.428571]]))

    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    assert controller.kp == pytest.approx(0.04)
    assert controller.ki == pytest.approx(0.1)
    assert controller.dt == pytest.approx(0.05)
    assert controller.limit == pytest.approx(2.0)
    controller.update(target, actual)

    assert controller.integral.shape == (1, 2)
    torch.testing.assert_close(controller.last_error, torch.tensor([[1., 4. / 7.]]))
    torch.testing.assert_close(controller.integral, torch.tensor([[0.005, 0.02 / 7.]]))
    assert controller.feedforward.shape == (1, 14)
    torch.testing.assert_close(controller.feedforward[:, :7], controller.integral[:, :1].expand(-1, 7))
    torch.testing.assert_close(controller.feedforward[:, 7:], controller.integral[:, 1:2].expand(-1, 7))
    torch.testing.assert_close(broadcast_side_values(controller.integral, SIDE_IDS), controller.feedforward)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_conditional_antiwindup_blocks_outward_error_and_allows_unwinding(sign: float):
    error = torch.tensor([[sign]])
    integral = torch.tensor([[sign * 1.99]])
    p_estimate = 0.04 * error
    updated, blocked = conditional_integral_step(
        error, integral, p_estimate, ki=0.1, dt=0.05, limit=2.0,
    )
    assert blocked.item()
    torch.testing.assert_close(updated, integral)

    unwind_error = torch.tensor([[-sign]])
    unwound, blocked = conditional_integral_step(
        unwind_error,
        integral,
        0.04 * unwind_error,
        ki=0.1,
        dt=0.05,
        limit=2.0,
    )
    assert not blocked.item()
    assert abs(unwound.item()) < abs(integral.item())


def test_zero_integral_gain_is_allowed_and_keeps_feedforward_zero():
    controller = SideIntegralDriveController(2, SIDE_IDS, device="cpu", ki=0.0)
    target = torch.ones((2, 14), dtype=torch.float32)
    actual = torch.zeros_like(target)
    controller.update(target, actual)
    torch.testing.assert_close(controller.integral, torch.zeros((2, 2)))
    torch.testing.assert_close(controller.feedforward, torch.zeros((2, 14)))
    assert not controller.last_blocked.any()


def test_integral_state_is_bounded_and_conditional_antiwindup_holds_at_limit():
    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    target = torch.full((1, 14), 10.0)
    actual = torch.zeros_like(target)

    for _ in range(100):
        controller.update(target, actual)

    assert torch.isfinite(controller.integral).all()
    assert torch.all(controller.integral <= 2.0)
    assert torch.all(controller.integral >= -2.0)
    assert torch.all(controller.feedforward.abs() <= 2.0)
    assert controller.last_blocked.all()
    torch.testing.assert_close(controller.integral, torch.full((1, 2), 1.65))


def test_controller_matches_frozen_public_helper_reference_across_saturation_and_partial_reset():
    side_ids = SIDE_IDS.clone()
    controller = SideIntegralDriveController(
        3, side_ids, device="cpu", kp=0.01, ki=0.2, dt=0.05, limit=0.12,
    )
    reference_integral = torch.zeros((3, 2))
    reference_error = torch.zeros_like(reference_integral)
    reference_p = torch.zeros_like(reference_integral)
    reference_blocked = torch.zeros_like(reference_integral, dtype=torch.bool)
    target = torch.tensor([
        [1.0] * 7 + [0.5] * 7,
        [-1.0] * 7 + [-2.0] * 7,
        [2.0] * 7 + [1.0] * 7,
    ])
    actual = torch.zeros_like(target)
    saw_blocked = False
    saw_unwinding = False

    for step in range(18):
        if step == 8:
            env_ids = torch.tensor([1], dtype=torch.long)
            controller.reset(env_ids)
            reference_integral[env_ids] = 0.0
            reference_error[env_ids] = 0.0
            reference_p[env_ids] = 0.0
            reference_blocked[env_ids] = False
        if step == 12:
            target = -target

        before_integral = reference_integral.clone()
        error, p_estimate, next_integral, blocked = _reference_drive_update(
            target, actual, reference_integral, side_ids,
            kp=controller.kp, ki=controller.ki, dt=controller.dt, limit=controller.limit,
        )
        controller.update(target, actual)
        reference_integral = next_integral
        reference_error = error
        reference_p = p_estimate
        reference_blocked = blocked
        saw_blocked = saw_blocked or bool(blocked.any())
        if step >= 12:
            saw_unwinding = saw_unwinding or bool((reference_integral.abs() < before_integral.abs()).any())

        torch.testing.assert_close(controller.last_error, reference_error, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_p_estimate, reference_p, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.integral, reference_integral, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_blocked, reference_blocked, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.feedforward, broadcast_side_values(reference_integral, side_ids))

    assert saw_blocked
    assert saw_unwinding


def test_controller_clones_fixed_side_topology_at_construction():
    side_ids = SIDE_IDS.clone()
    controller = SideIntegralDriveController(1, side_ids, device="cpu")
    side_ids.zero_()
    controller.side_ids.zero_()
    target = torch.tensor([[2.0] * 7 + [4.0] * 7])
    controller.update(target, torch.zeros_like(target))

    torch.testing.assert_close(controller.last_error, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(controller.feedforward[:, :7], controller.integral[:, :1].expand(-1, 7))
    torch.testing.assert_close(controller.feedforward[:, 7:], controller.integral[:, 1:2].expand(-1, 7))


def test_partial_reset_clears_only_selected_environment_integral_and_broadcast():
    controller = SideIntegralDriveController(2, SIDE_IDS, device="cpu")
    target = torch.tensor([[2.] * 7 + [4.] * 7, [-3.] * 7 + [5.] * 7])
    controller.update(target, torch.zeros_like(target))
    controller.integral[0] = torch.tensor([0.25, -0.5])
    controller.integral[1] = torch.tensor([-0.3, 0.4])
    before_integral = controller.integral.clone()
    before_feedforward = controller.feedforward.clone()

    controller.reset(torch.tensor([1], dtype=torch.long))

    torch.testing.assert_close(controller.integral[0], before_integral[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(controller.feedforward[0], before_feedforward[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(controller.integral[1], torch.zeros(2))
    torch.testing.assert_close(controller.feedforward[1], torch.zeros(14))
    torch.testing.assert_close(controller.last_error[1], torch.zeros(2))
    torch.testing.assert_close(controller.last_p_estimate[1], torch.zeros(2))
    assert not controller.last_blocked[1].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kp": 0.0},
        {"ki": -0.1},
        {"ki": float("nan")},
        {"dt": 0.0},
        {"dt": float("inf")},
        {"limit": 0.0},
    ],
)
def test_controller_rejects_invalid_gains_timebase_and_limit(kwargs):
    with pytest.raises(ValueError):
        SideIntegralDriveController(1, SIDE_IDS, device="cpu", **kwargs)


def test_controller_rejects_invalid_environment_count_side_map_shapes_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="num_envs"):
        SideIntegralDriveController(0, SIDE_IDS, device="cpu")
    with pytest.raises(ValueError, match="both left and right"):
        SideIntegralDriveController(1, torch.zeros(14, dtype=torch.long), device="cpu")
    with pytest.raises(ValueError, match="wheel values"):
        side_mean(torch.ones((1, 13)), SIDE_IDS)

    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        controller.update(torch.ones((1, 13)), torch.zeros((1, 13)))
    with pytest.raises(ValueError, match="finite"):
        controller.update(torch.full((1, 14), float("nan")), torch.zeros((1, 14)))


@pytest.mark.parametrize(
    "side_ids, message",
    [
        (torch.tensor([False] * 7 + [True] * 7), "integer wheel-side"),
        (torch.tensor([0.0] * 7 + [1.0] * 6 + [1.5]), "finite integer"),
        (torch.tensor([0] * 7 + [1] * 6 + [2]), "side ids must be 0"),
    ],
)
def test_controller_rejects_noninteger_and_out_of_range_topology(side_ids, message):
    with pytest.raises(ValueError, match=message):
        SideIntegralDriveController(1, side_ids, device="cpu")


@pytest.mark.parametrize(
    "target_value, actual_value",
    [(float("nan"), 0.0), (0.0, float("inf"))],
)
def test_controller_rejects_nonfinite_target_or_actual_without_mutating_state(target_value, actual_value):
    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    target = torch.full((1, 14), target_value)
    actual = torch.full((1, 14), actual_value)
    before = (controller.integral.clone(), controller.last_error.clone(), controller.last_p_estimate.clone())

    with pytest.raises(ValueError, match="finite"):
        controller.update(target, actual)

    torch.testing.assert_close(controller.integral, before[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(controller.last_error, before[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(controller.last_p_estimate, before[2], rtol=0.0, atol=0.0)


def test_controller_rejects_nonfinite_integral_and_finite_inputs_that_overflow_error():
    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    target = torch.ones((1, 14))
    actual = torch.zeros_like(target)
    controller.integral[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        controller.update(target, actual)
    controller.integral.zero_()

    large = torch.finfo(torch.float32).max
    target[:, :7] = large
    actual[:, :7] = -large
    with pytest.raises(ValueError, match="finite"):
        controller.update(target, actual)


@pytest.mark.parametrize(
    "name, invalid_value",
    [("kp", float("nan")), ("ki", -1.0), ("dt", 0.0), ("limit", float("inf"))],
)
def test_controller_revalidates_mutated_scalar_parameters(name, invalid_value):
    controller = SideIntegralDriveController(1, SIDE_IDS, device="cpu")
    setattr(controller, name, invalid_value)
    with pytest.raises(ValueError, match=name):
        controller.update(torch.ones((1, 14)), torch.zeros((1, 14)))


@pytest.mark.parametrize(
    "error, integral, p_estimate",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("nan")),
    ],
)
def test_public_integral_helper_keeps_strict_nonfinite_validation(error, integral, p_estimate):
    with pytest.raises(ValueError, match="PI inputs must be finite"):
        conditional_integral_step(
            torch.tensor([[error]]),
            torch.tensor([[integral]]),
            torch.tensor([[p_estimate]]),
            ki=0.1,
            dt=0.05,
            limit=2.0,
        )

