"""Reconstructed CUDA equivalence check for SideIntegralDriveController.

This script was recreated after the original inline execution from the
recorded check description. It has not been executed as part of the
documentation update. Run from the 仿真训练环境 directory only when a CUDA
validation is separately authorized and no Kit is running.
"""
from __future__ import annotations

import torch

from isaac_field.drive_control import (
    SideIntegralDriveController,
    broadcast_side_values,
    conditional_integral_step,
    side_mean,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda:0")
    side_ids = torch.tensor([0] * 7 + [1] * 7, dtype=torch.long, device=device)
    controller = SideIntegralDriveController(
        3, side_ids, device=device, kp=0.01, ki=0.2, dt=0.05, limit=0.12,
    )
    reference_integral = torch.zeros((3, 2), device=device)
    target = torch.tensor(
        [
            [1.0] * 7 + [0.5] * 7,
            [-1.0] * 7 + [-2.0] * 7,
            [2.0] * 7 + [1.0] * 7,
        ],
        device=device,
    )
    actual = torch.zeros_like(target)
    saw_blocked = False
    saw_unwinding = False

    for step in range(18):
        if step == 8:
            env_ids = torch.tensor([1], dtype=torch.long, device=device)
            controller.reset(env_ids)
            reference_integral[env_ids] = 0.0
        if step == 12:
            target = -target

        before_integral = reference_integral.clone()
        reference_error = side_mean(target, side_ids) - side_mean(actual, side_ids)
        reference_p = controller.kp * reference_error
        reference_integral, reference_blocked = conditional_integral_step(
            reference_error,
            reference_integral,
            reference_p,
            ki=controller.ki,
            dt=controller.dt,
            limit=controller.limit,
        )

        controller.update(target, actual)
        torch.testing.assert_close(controller.last_error, reference_error, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_p_estimate, reference_p, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.integral, reference_integral, rtol=0.0, atol=0.0)
        torch.testing.assert_close(controller.last_blocked, reference_blocked, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            controller.feedforward,
            broadcast_side_values(reference_integral, side_ids),
            rtol=0.0,
            atol=0.0,
        )
        saw_blocked = saw_blocked or bool(reference_blocked.any())
        if step >= 12:
            saw_unwinding = saw_unwinding or bool(
                (reference_integral.abs() < before_integral.abs()).any()
            )

    assert saw_blocked
    assert saw_unwinding

    mutable_side_ids = side_ids.clone()
    topology_probe = SideIntegralDriveController(1, mutable_side_ids, device=device)
    mutable_side_ids.zero_()
    topology_probe.side_ids.zero_()
    probe_target = torch.tensor([[2.0] * 7 + [4.0] * 7], device=device)
    topology_probe.update(probe_target, torch.zeros_like(probe_target))
    torch.testing.assert_close(
        topology_probe.last_error,
        torch.tensor([[2.0, 4.0]], device=device),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(topology_probe.feedforward[:, :7], topology_probe.integral[:, :1].expand(-1, 7))
    assert torch.equal(topology_probe.feedforward[:, 7:], topology_probe.integral[:, 1:2].expand(-1, 7))

    print(
        "CUDA equivalent: device=cuda:0, steps=18, partial_reset=true, "
        "saturation=true, topology_clone=true"
    )


if __name__ == "__main__":
    main()
