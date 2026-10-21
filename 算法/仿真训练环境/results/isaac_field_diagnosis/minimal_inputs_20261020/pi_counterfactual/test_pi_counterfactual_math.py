from __future__ import annotations

import pytest
import torch

from runtime_pi import (
    RuntimePIController,
    broadcast_side_values,
    conditional_integral_step,
    side_mean,
)
from pi_counterfactual_probe import select_cases
from compare_pi_counterfactual_runs import compare_initial_state_fields, evaluate_brake_tail


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


def test_case_filter_defaults_to_the_original_eight_actions_and_two_brakes():
    actions = [{"name": f"action_{index}"} for index in range(8)]
    brakes = [{"name": "brake_straight"}, {"name": "brake_turn"}]
    selected_actions, selected_brakes, names = select_cases(None, actions, brakes)
    assert selected_actions == actions
    assert selected_brakes == brakes
    assert names == [row["name"] for row in actions + brakes]


def test_case_filter_accepts_repeated_and_comma_separated_names():
    actions = [
        {"name": "diff_left_plus_0.05"},
        {"name": "diff_left_plus_0.10"},
        {"name": "other_action"},
    ]
    brakes = [
        {"name": "brake_straight_0.20_to_zero"},
        {"name": "brake_diff_left_plus_0.80_to_zero"},
    ]
    selected_actions, selected_brakes, names = select_cases(
        ["diff_left_plus_0.05,diff_left_plus_0.10", "brake_straight_0.20_to_zero", "brake_diff_left_plus_0.80_to_zero"],
        actions,
        brakes,
    )
    assert [row["name"] for row in selected_actions] == [
        "diff_left_plus_0.05", "diff_left_plus_0.10"
    ]
    assert [row["name"] for row in selected_brakes] == [
        "brake_straight_0.20_to_zero", "brake_diff_left_plus_0.80_to_zero"
    ]
    assert names == [row["name"] for row in selected_actions + selected_brakes]


def test_case_filter_rejects_duplicates_unknown_and_empty_entries():
    actions = [{"name": "known"}]
    brakes = [{"name": "brake"}]
    with pytest.raises(ValueError, match="duplicate"):
        select_cases(["known,known"], actions, brakes)
    with pytest.raises(ValueError, match="unknown"):
        select_cases(["missing"], actions, brakes)
    with pytest.raises(ValueError, match="non-empty"):
        select_cases(["known,"], actions, brakes)


def test_confirmation_initial_state_tolerance_reports_actual_maxima():
    p_state = {
        "position_xy_m": [50.0, 50.0],
        "wheel_actual_qd_rad_s": [0.0, 0.0],
        "track_command_mps": [0.0, 0.0],
    }
    pi_state = {
        "position_xy_m": [50.00005, 50.0],
        "wheel_actual_qd_rad_s": [0.0005, 0.0],
        "track_command_mps": [0.0, 0.0],
    }
    deltas, passed = compare_initial_state_fields(
        p_state,
        pi_state,
        {"position_xy_m": 1e-4, "wheel_actual_qd_rad_s": 1e-3, "track_command_mps": 0.0},
    )
    assert deltas["position_xy_m"] == pytest.approx(5e-5)
    assert deltas["wheel_actual_qd_rad_s"] == pytest.approx(5e-4)
    assert passed == {
        "position_xy_m": True,
        "wheel_actual_qd_rad_s": True,
        "track_command_mps": True,
    }
    pi_state["wheel_actual_qd_rad_s"][0] = 0.0011
    _, passed = compare_initial_state_fields(
        p_state, pi_state, {"wheel_actual_qd_rad_s": 1e-3}
    )
    assert passed["wheel_actual_qd_rad_s"] is False


def test_confirmation_brake_tail_requires_zero_command_and_strict_velocity_limits():
    rows = [
        {"step": index, "track_command_mps": [0.0, 0.0], "actual_v_w": [0.1, -0.2]}
        for index in range(1, 7)
    ]
    result = evaluate_brake_tail(
        rows,
        tail_count=6,
        command_target=[0.0, 0.0],
        speed_limit_mps=0.2,
        yaw_limit_rad_s=0.3,
    )
    assert result["passed"] is True
    rows[-1]["track_command_mps"] = [0.0, 1e-6]
    result = evaluate_brake_tail(
        rows,
        tail_count=6,
        command_target=[0.0, 0.0],
        speed_limit_mps=0.2,
        yaw_limit_rad_s=0.3,
    )
    assert result["passed"] is False
    assert result["track_command_exact_zero"] is False
