"""Small Warp kernel shared by opt-in native Newton effort regression probes."""
from __future__ import annotations

import warp as wp


@wp.kernel
def record_two_env_wheel_qd(
    state_qd: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    tape: wp.array2d(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    capacity: int,
):
    slot = wp.atomic_add(counter, 0, 1)
    if slot < capacity:
        for world in range(2):
            for wheel in range(14):
                tape[slot, world * 14 + wheel] = state_qd[offsets[world * 14 + wheel]]


@wp.kernel
def record_wheel_applied_effort(
    applied: wp.array2d(dtype=wp.float32),
    wheel_ids: wp.array(dtype=wp.int32),
    tape: wp.array2d(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    capacity: int,
):
    slot = wp.atomic_add(counter, 0, 1)
    if slot < capacity:
        for wheel in range(14):
            tape[slot, wheel] = applied[0, wheel_ids[wheel]]
