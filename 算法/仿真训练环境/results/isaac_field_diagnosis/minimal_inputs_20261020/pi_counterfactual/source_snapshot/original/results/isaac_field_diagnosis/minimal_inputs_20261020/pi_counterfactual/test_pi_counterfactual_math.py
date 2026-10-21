from __future__ import annotations

import pytest
import torch

from runtime_pi import (
    RuntimePIController,
    broadcast_side_values,
    conditional_integral_step,
    side_mean,
)


def test_conditional_antiwindup_blocks_outward_integral_at_positive_saturation():
    error = torch.tensor([[1.0]], dtype=torch.float32)
    integral = torch.tensor([[1.99]], dtype=torch.float32)
    p_estimate = 0.04 * error
    updated, blocked = conditional_integral_step(
        error, integral, p_estimate, ki=0.1, dt=0.05, limit=2.0
    )
    assert blocked.item()
    torch.testing.assert_close(updated, integral)


def test_conditional_antiwindup_blocks_outward_integral_at_negative_saturation():
    error = torch.tensor([[-1.0]], dtype=torch.float32)
    integral = torch.tensor([[-1.99]], dtype=torch.float32)
    p_estimate = 0.04 * error
    updated, blocked = conditional_integral_step(
        error, integral, p_estimate, ki=0.1, dt=0.05, limit=2.0
    )
    assert blocked.item()
    torch.testing.assert_close(updated, integral)


def test_integral_can_unwind_and_remains_clipped_to_effort_limit():
    error = torch.tensor([[-0.4]], dtype=torch.float32)
    integral = torch.tensor([[1.9]], dtype=torch.float32)
    p_estimate = 0.04 * error
    unwound, blocked = conditional_integral_step(
        error, integral, p_estimate, ki=0.1, dt=0.05, limit=2.0
    )
    assert not blocked.item()
    assert unwound.item() < integral.item()

    large_error = torch.tensor([[0.4]], dtype=torch.float32)
    clipped, blocked = conditional_integral_step(
        large_error,
        torch.zeros_like(large_error),
        0.04 * large_error,
        ki=100.0,
        dt=0.05,
        limit=2.0,
    )
    assert not blocked.item()
    assert clipped.item() == pytest.approx(2.0)


def test_controller_partial_reset_clears_only_selected_environment():
    controller = RuntimePIController(2, 3, device="cpu")
    target = torch.ones((2, 3), dtype=torch.float32)
    actual = torch.zeros_like(target)
    controller.update(target, actual)
    controller.integral[0] = torch.tensor([0.25, -0.5, 0.75])
    controller.integral[1] = torch.tensor([-0.3, 0.4, -0.5])
    controller.reset(torch.tensor([1], dtype=torch.long))
    torch.testing.assert_close(controller.integral[0], torch.tensor([0.25, -0.5, 0.75]))
    torch.testing.assert_close(controller.integral[1], torch.zeros(3))
    torch.testing.assert_close(controller.last_error[0], torch.ones(3))
    torch.testing.assert_close(controller.last_error[1], torch.zeros(3))
    assert not controller.last_blocked[1].any()


def test_controller_rejects_nonfinite_velocity_inputs():
    controller = RuntimePIController(1, 2, device="cpu")
    with pytest.raises(ValueError, match="finite"):
        controller.update(torch.tensor([[float("nan"), 0.0]]), torch.zeros((1, 2)))


def test_per_side_state_uses_wheel_means_and_broadcasts_same_effort_per_side():
    sides = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    target_wheel_qd = torch.tensor([[
        2.0, 4.0, 3.0, 5.0, 1.0, 2.0, 4.0,
        10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
    ]])
    actual_wheel_qd = torch.tensor([[
        1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 4.0,
        9.0, 12.0, 13.0, 15.0, 18.0, 19.0, 22.0,
    ]])
    target_side = side_mean(target_wheel_qd, sides)
    actual_side = side_mean(actual_wheel_qd, sides)
    torch.testing.assert_close(target_side, torch.tensor([[3.0, 16.0]]))
    torch.testing.assert_close(actual_side, torch.tensor([[2.0, 15.428571]]))

    controller = RuntimePIController(1, 2, device="cpu", ki=0.1)
    controller.update(target_side, actual_side)
    assert controller.integral[0, 0] > 0.0
    assert controller.integral[0, 1] > 0.0
    wheel_ff = broadcast_side_values(controller.feedforward, sides)
    assert wheel_ff.shape == (1, 14)
    torch.testing.assert_close(wheel_ff[:, :7], controller.feedforward[:, :1].expand(-1, 7))
    torch.testing.assert_close(wheel_ff[:, 7:], controller.feedforward[:, 1:2].expand(-1, 7))


def test_per_side_integral_partial_reset_only_clears_selected_environment():
    controller = RuntimePIController(2, 2, device="cpu", ki=0.1)
    target_side = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    controller.update(target_side, torch.zeros_like(target_side))
    saved = controller.integral[0].clone()
    controller.reset(torch.tensor([1], dtype=torch.long))
    torch.testing.assert_close(controller.integral[0], saved)
    torch.testing.assert_close(controller.integral[1], torch.zeros(2))
    torch.testing.assert_close(
        broadcast_side_values(controller.feedforward, torch.tensor([0, 0, 1, 1])),
        torch.tensor([[saved[0], saved[0], saved[1], saved[1]], [0.0, 0.0, 0.0, 0.0]]),
    )


def test_per_side_helpers_reject_missing_side_and_invalid_map():
    with pytest.raises(ValueError, match="side 1 has no proxy wheels"):
        side_mean(torch.ones((1, 2)), torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="side ids must be 0"):
        broadcast_side_values(torch.ones((1, 2)), torch.tensor([0, 2]))
