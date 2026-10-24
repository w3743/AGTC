"""CPU checks for batched Isaac state sampling and partial-reset invalidation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaac_field import env as env_module
from isaac_field.assets import ROBOT_SPEC


class _Task:
    def __init__(self, *, time=0.0, frames=True):
        self.robot = np.zeros(5, dtype=np.float64)
        self.gimbal = np.zeros(2, dtype=np.float64)
        self.body_pose = (0.0, 0.0, 0.0)
        self.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
        self.body_angular_velocity = np.zeros(3, dtype=np.float64)
        self.view_origin = np.zeros(3, dtype=np.float32)
        self.view_basis = tuple(np.zeros(3, dtype=np.float32) for _ in range(3))
        self.done = False
        self.time = time
        self.frames = [object()] if frames else []
        self.feedback_samples = []

    def update_feedback(self, sample):
        self.feedback_samples.append(sample)

    def _observation(self):
        return {"sample": np.zeros(1, dtype=np.float32)}

    def initialize_sensors(self):
        self.frames.append(object())
        return {"sample": np.ones(1, dtype=np.float32)}, None


def _bare_env(num_envs=2):
    env = env_module.IsaacFieldEnv.__new__(env_module.IsaacFieldEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._mount_id = 0
    env._gimbal_ids = torch.tensor([14, 15])
    env._wheel_ids = torch.arange(14)
    env._wheel_side_masks = (np.arange(14) < 7, np.arange(14) >= 7)
    env._imu_offset_rotation = np.eye(3, dtype=np.float32)
    env._torque_to_iq = (ROBOT_SPEC.drive_sprocket_radius_m / (
        ROBOT_SPEC.wheel_radius_m * ROBOT_SPEC.drive_reduction_ratio * 0.035
    ))
    env._omega_to_motor_rpm = (ROBOT_SPEC.wheel_radius_m / ROBOT_SPEC.drive_sprocket_radius_m
                               * ROBOT_SPEC.drive_reduction_ratio * 60.0 / (2.0 * np.pi))
    env._state_observation_fresh = False
    env._feedback_host_cache = None
    env.cfg = SimpleNamespace(auto_reset=True, sim=SimpleNamespace(dt=0.001), decimation=50)
    env.tasks = [_Task(time=0.05 * i) for i in range(num_envs)]

    root_pose = torch.zeros((num_envs, 7), dtype=torch.float32)
    root_pose[:, :2] = torch.tensor([[11.0, 22.0], [101.0, 202.0]], dtype=torch.float32)[:num_envs]
    root_pose[:, 2] = 0.3
    root_pose[:, 6] = 1.0
    root_pose[0, 3:7] = torch.tensor([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    if num_envs > 1:
        root_pose[1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    root_velocity = torch.tensor([
        [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        [-2.0, 4.0, -1.0, -0.4, 0.5, -0.6],
    ], dtype=torch.float32)[:num_envs]
    mount_pose = root_pose.clone()
    mount_pose[:, 0] += 0.25
    joint_pos = torch.arange(num_envs * 16, dtype=torch.float32).reshape(num_envs, 16) / 10
    joint_vel = torch.arange(num_envs * 16, dtype=torch.float32).reshape(num_envs, 16) / 5
    effort = torch.arange(num_envs * 16, dtype=torch.float32).reshape(num_envs, 16) + 1
    env.rover = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pose_w=SimpleNamespace(torch=root_pose),
            root_com_vel_w=SimpleNamespace(torch=root_velocity),
            body_link_pose_w=SimpleNamespace(torch=mount_pose[:, None, :]),
            joint_pos=SimpleNamespace(torch=joint_pos),
            joint_vel=SimpleNamespace(torch=joint_vel),
        ),
        actuators=SimpleNamespace(applied_effort=SimpleNamespace(torch=effort)),
    )
    origins = torch.tensor([[10.0, 20.0, 0.0], [100.0, 200.0, 0.0]], dtype=torch.float32)[:num_envs]
    imu_acc = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=torch.float32)[:num_envs]
    imu_gyro = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=torch.float32)[:num_envs]
    class _Scene(dict):
        def __init__(self, *args, env_origins, **kwargs):
            super().__init__(*args, **kwargs)
            self.env_origins = env_origins

    env.scene = _Scene({
        "imu": SimpleNamespace(data=SimpleNamespace(
            lin_acc_b=SimpleNamespace(torch=imu_acc),
            ang_vel_b=SimpleNamespace(torch=imu_gyro),
        )),
    }, env_origins=origins)
    return env


def test_state_and_feedback_share_one_host_transfer_and_preserve_physical_values(monkeypatch):
    env = _bare_env()
    env.tasks[0].time = 0.05
    cpu_calls = []
    original_cpu = torch.Tensor.cpu

    def count_cpu(tensor, *args, **kwargs):
        cpu_calls.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", count_cpu)
    env._refresh_state(include_feedback=True)
    env._refresh_feedback(after_step=True)

    assert len(cpu_calls) == 1
    first, second = env.tasks
    assert first.robot == pytest.approx([1.0, 2.0, np.pi / 2, 2.0, 0.3])
    assert second.robot == pytest.approx([1.0, 2.0, 0.0, -2.0, -0.6])
    assert first.body_pose == pytest.approx((0.3, 0.0, 0.0))
    assert first.root_com_velocity_xy == pytest.approx([1.0, 2.0])
    assert first.body_angular_velocity == pytest.approx([0.2, -0.1, 0.3])
    assert first.view_origin == pytest.approx([1.25, 2.0, 0.3])
    np.testing.assert_allclose(
        np.stack(first.view_basis),
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-6,
    )

    sample = first.feedback_samples[-1]
    assert sample.sample_time == pytest.approx(0.1)
    assert sample.accel_mps2 == pytest.approx([1.0, 2.0, 3.0])
    assert sample.gyro_rps == pytest.approx([0.1, 0.2, 0.3])
    assert sample.mag_counts == pytest.approx([3000.0, 0.0, -6000.0])
    assert sample.temperature_c == pytest.approx(25.0)
    assert sample.gimbal_q_rad == pytest.approx(env.rover.data.joint_pos.torch[0, 14:16].numpy())
    assert sample.gimbal_qd_rps == pytest.approx(env.rover.data.joint_vel.torch[0, 14:16].numpy())
    torque = env.rover.actuators.applied_effort.torch[0, :14].numpy()
    speed = env.rover.data.joint_vel.torch[0, :14].numpy()
    assert sample.motor_iq_a == pytest.approx([
        torque[:7].sum() * env._torque_to_iq,
        torque[7:].sum() * env._torque_to_iq,
    ])
    assert sample.motor_rpm == pytest.approx([
        speed[:7].mean() * env._omega_to_motor_rpm,
        speed[7:].mean() * env._omega_to_motor_rpm,
    ])
    assert env._feedback_host_cache is None


def test_partial_reset_observation_invalidates_cached_state_before_sensor_render():
    env = _bare_env()
    env.tasks[1].frames = []
    env._state_observation_fresh = True
    env._feedback_host_cache = {"stale": True}
    calls = []

    class _Scene(dict):
        env_origins = torch.zeros((2, 3))

        def write_data_to_sim(self):
            calls.append("write")

    class _Sensor:
        def update(self, dt, *, force_recompute):
            calls.append((dt, force_recompute))

    class _Sim:
        device = torch.device("cpu")

        def forward(self):
            calls.append("forward")

        def render(self):
            calls.append("render")

    env.scene = _Scene({name: _Sensor() for name in ("lidar", "rangefinder", "depth", "detector")})
    env.sim = _Sim()

    def refresh_state(*, include_feedback=False):
        calls.append(("state", env._state_observation_fresh, include_feedback))
        env._state_observation_fresh = True
        env._feedback_host_cache = {"fresh": True} if include_feedback else None

    def refresh_feedback(*, after_step):
        calls.append(("feedback", after_step, env._feedback_host_cache))
        env._feedback_host_cache = None

    env._refresh_state = refresh_state
    env._refresh_feedback = refresh_feedback

    output = env._get_observations()

    assert calls[0:3] == ["write", "forward", "render"]
    assert calls[3:7] == [(0.0, True)] * 4
    state_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "state")
    assert state_call == ("state", False, True)
    assert ("feedback", False, {"fresh": True}) in calls
    assert output["policy"]["sample"].shape == (2, 1)
    assert not env._state_observation_fresh
    assert env._feedback_host_cache is None

    # The post-step state is already sampled by _get_dones; observation reads
    # consume it without another simulator-state transfer.
    calls.clear()
    env._state_observation_fresh = True
    env._feedback_host_cache = None
    env._get_observations()
    assert calls == []
    assert not env._state_observation_fresh
