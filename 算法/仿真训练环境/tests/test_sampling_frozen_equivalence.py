"""CPU equivalence check for the new batched sampler against a02's frozen methods."""
from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from isaac_field import env as env_module


def _load_cache_helpers():
    helper_path = Path(__file__).with_name("test_sampling_copy_cache.py")
    spec = importlib.util.spec_from_file_location("_sampling_copy_cache_helpers", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_frozen_methods():
    sim_root = Path(__file__).resolve().parents[1]
    source = sim_root / "results" / "isaac_field_improvement" / "physx_pointmove_flat12_20261024_a02" / "source_snapshot" / "isaac_field" / "env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacFieldEnv")
    names = {"_refresh_state", "_refresh_feedback"}
    methods = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert {node.name for node in methods} == names
    namespace = {
        "torch": torch,
        "np": np,
        "math": math,
        "matrix_from_quat": env_module.matrix_from_quat,
        "euler_xyz_from_quat": env_module.euler_xyz_from_quat,
        "wrap": env_module.wrap,
        "FeedbackSample": env_module.FeedbackSample,
        "ROBOT_SPEC": env_module.ROBOT_SPEC,
    }
    module_ast = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(module_ast, str(source), "exec"), namespace)
    return namespace


def _quat_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _configured_env(helpers):
    env = helpers._bare_env(num_envs=2)
    offset_q = _quat_xyz(0.21, -0.33, 0.47)
    env.cfg.imu_offset_rot_xyzw = offset_q
    env.cfg.drive_efficiency_assumption = 0.91
    env.cfg.motor_torque_constant_nm_per_a = 0.041
    env.cfg.auto_reset = True
    spec = env_module.ROBOT_SPEC
    env._torque_to_iq = spec.drive_sprocket_radius_m / (
        spec.wheel_radius_m * spec.drive_reduction_ratio
        * env.cfg.drive_efficiency_assumption * env.cfg.motor_torque_constant_nm_per_a
    )
    env._omega_to_motor_rpm = (spec.wheel_radius_m / spec.drive_sprocket_radius_m
                               * spec.drive_reduction_ratio * 60.0 / (2.0 * np.pi))
    env._wheel_track_ids = torch.tensor([0] * 7 + [1] * 7, dtype=torch.long)
    env._wheel_side_masks = (np.arange(14) < 7, np.arange(14) >= 7)
    env._imu_offset_rotation = env_module.matrix_from_quat(
        torch.tensor(offset_q, dtype=torch.float32).reshape(1, 4)
    )[0].numpy().copy()

    root = env.rover.data.root_link_pose_w.torch
    mount = env.rover.data.body_link_pose_w.torch[:, 0, :]
    root[0, 3:7] = torch.tensor(_quat_xyz(0.34, -0.17, 0.82), dtype=torch.float32)
    root[1, 3:7] = torch.tensor(_quat_xyz(-0.28, 0.23, -0.61), dtype=torch.float32)
    mount[0, 3:7] = torch.tensor(_quat_xyz(0.44, -0.09, 1.02), dtype=torch.float32)
    mount[1, 3:7] = torch.tensor(_quat_xyz(-0.11, 0.37, -0.42), dtype=torch.float32)
    env.scene["imu"].data.lin_acc_b.torch[:] = torch.tensor(
        [[1.7, -2.1, 9.3], [-0.8, 0.4, 10.2]], dtype=torch.float32
    )
    env.scene["imu"].data.ang_vel_b.torch[:] = torch.tensor(
        [[0.16, -0.27, 0.31], [-0.22, 0.18, -0.13]], dtype=torch.float32
    )
    return env


def _assert_same_value(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, (float, int, np.number)):
        assert actual == pytest.approx(expected, abs=1e-7, rel=1e-7)
    else:
        actual_array = np.asarray(actual)
        expected_array = np.asarray(expected)
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_allclose(actual_array, expected_array, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("after_step", [False, True])
def test_batched_sampling_matches_frozen_truth_and_feedback_for_multiple_envs(after_step):
    helpers = _load_cache_helpers()
    old_methods = _load_frozen_methods()
    current = _configured_env(helpers)
    frozen = _configured_env(helpers)

    current._refresh_state(include_feedback=True)
    current._refresh_feedback(after_step=after_step)
    old_methods["_refresh_state"](frozen)
    old_methods["_refresh_feedback"](frozen, after_step=after_step)

    assert len(current.tasks) == len(frozen.tasks) == 2
    for actual_task, expected_task in zip(current.tasks, frozen.tasks, strict=True):
        for name in (
            "robot", "gimbal", "root_com_velocity_xy", "body_angular_velocity", "view_origin",
        ):
            _assert_same_value(getattr(actual_task, name), getattr(expected_task, name))
        _assert_same_value(actual_task.body_pose, expected_task.body_pose)
        _assert_same_value(np.stack(actual_task.view_basis), np.stack(expected_task.view_basis))
        assert len(actual_task.feedback_samples) == len(expected_task.feedback_samples) == 1
        actual_sample = actual_task.feedback_samples[0]
        expected_sample = expected_task.feedback_samples[0]
        for field in fields(env_module.FeedbackSample):
            _assert_same_value(getattr(actual_sample, field.name), getattr(expected_sample, field.name))

    assert current._feedback_host_cache is None
