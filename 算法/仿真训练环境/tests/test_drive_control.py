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

