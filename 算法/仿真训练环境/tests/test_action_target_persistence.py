"""CPU coverage for Isaac Lab action-target persistence across decimation."""
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from isaac_field import env as env_module
from isaac_field.drive_control import SideIntegralDriveController
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.actuators import actuator_kernels
from isaaclab.actuators.actuator_collection import ActuatorTargetCommand
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray


class _CpuActuatorControl:
    """Minimal CPU selector/validation bridge for the real SDK command class."""

    def __init__(self, num_envs, num_joints):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.staged = []

    def _indices(self, indices, size):
        if indices is None:
            values = range(size)
        elif isinstance(indices, wp.array):
            return indices
        elif isinstance(indices, torch.Tensor):
            assert indices.device.type == "cpu"
            values = indices.tolist()
        elif isinstance(indices, slice):
            values = range(size)[indices]
        else:
            values = indices
        return wp.array(list(values), dtype=wp.int32, device="cpu")

    def resolve_env_ids(self, indices):
        return self._indices(indices, self.num_envs)

    def resolve_joint_ids(self, indices):
        return self._indices(indices, self.num_joints)

    def assert_shape_and_dtype(self, value, shape, dtype, name):
        assert tuple(value.shape) == shape
        if isinstance(value, wp.array):
            assert value.dtype == dtype
        else:
            assert value.dtype == torch.float32
            assert value.device.type == "cpu"

    def stage_user_command(self, command_name, *_args):
        self.staged.append(command_name)


class _CpuTargetBuffers:
    """Real Warp command storage with the SDK's real target-command setters."""

    def __init__(self, num_envs=2, num_joints=16):
        self.num_instances = num_envs
        self.num_joints = num_joints
        self.device = "cpu"
        self._control = _CpuActuatorControl(num_envs, num_joints)
        for name in ("position", "velocity", "effort"):
            short_name = {"position": "pos", "velocity": "vel", "effort": "effort"}[name]
            buffer = wp.zeros((num_envs, num_joints), dtype=wp.float32, device="cpu")
            setattr(self, f"_joint_{short_name}_target", buffer)
            setattr(self, f"_joint_{short_name}_target_ta", ProxyArray(buffer))
        self.target_command = ActuatorTargetCommand(self)

    def snapshot(self):
        return {
            name: getattr(self, f"_joint_{short_name}_target_ta").torch.clone()
            for name, short_name in {"position": "pos", "velocity": "vel", "effort": "effort"}.items()
        }


def _bare_env(targets):
    env = env_module.IsaacFieldEnv.__new__(env_module.IsaacFieldEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.001), decimation=50)
    env.field_cfg = SimpleNamespace(
        max_speed=1.0,
        acceleration=20.0,
        gimbal_yaw_rate=0.4,
        gimbal_pitch_rate=0.3,
        pitch_min=-0.7,
        pitch_max=0.7,
    )
    env.tasks = [
        SimpleNamespace(robot=np.zeros(5), goal=np.array([1.0, 0.0])),
        SimpleNamespace(robot=np.zeros(5), goal=np.array([0.0, 1.0])),
    ]
    env.track_command = torch.zeros((2, 2), dtype=torch.float32)
    env.command = torch.zeros((2, 2), dtype=torch.float32)
    env.wheel_target_velocity = torch.zeros((2, 14), dtype=torch.float32)
    env.gimbal_target = torch.zeros((2, 2), dtype=torch.float32)
    env._wheel_ids = torch.arange(14, dtype=torch.long)
    env._gimbal_ids = torch.tensor([14, 15], dtype=torch.long)
    env._wheel_track_ids = torch.tensor([0] * 7 + [1] * 7, dtype=torch.long)
    env._drive_pi = SideIntegralDriveController(
        2,
        torch.tensor([0] * 7 + [1] * 7, dtype=torch.long),
        device="cpu",
        kp=0.04,
        ki=0.1,
        dt=env.step_dt,
        limit=2.0,
    )
    env.rover = SimpleNamespace(
        data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.zeros((2, 16), dtype=torch.float32))),
        actuators=SimpleNamespace(target_command=targets.target_command),
    )
    env._sensor_cache = {}
    return env


def _implicit_cpu_state():
    shape = (2, 16)
    joint_pos = wp.zeros(shape, dtype=wp.float32, device="cpu")
    joint_vel = wp.zeros(shape, dtype=wp.float32, device="cpu")
    stiffness = wp.zeros(shape, dtype=wp.float32, device="cpu")
    damping = wp.zeros(shape, dtype=wp.float32, device="cpu")
    effort_limit = wp.full(shape, 1.0, dtype=wp.float32, device="cpu")
    velocity_limit = wp.full((2, 2), 3.0, dtype=wp.float32, device="cpu")
    wp.to_torch(stiffness)[:, 14:16] = 1.0
    wp.to_torch(damping)[:, 14:16] = 0.1
    batch_shape = (2, 2)
    outputs = [
        wp.zeros(batch_shape, dtype=wp.float32, device="cpu"),
        wp.zeros(batch_shape, dtype=wp.float32, device="cpu"),
        *(wp.zeros(shape, dtype=wp.float32, device="cpu") for _ in range(6)),
    ]
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "stiffness": stiffness,
        "damping": damping,
        "effort_limit": effort_limit,
        "velocity_limit": velocity_limit,
        "joint_indices": wp.array([14, 15], dtype=wp.int32, device="cpu"),
        "outputs": outputs,
    }


def _compute_implicit(targets, state):
    wp.launch(
        actuator_kernels.compute_implicit_actuator_batch,
        dim=(2, 2),
        inputs=[
            targets._joint_pos_target,
            targets._joint_vel_target,
            targets._joint_effort_target,
            state["joint_pos"],
            state["joint_vel"],
            state["stiffness"],
            state["damping"],
            state["effort_limit"],
            state["velocity_limit"],
            state["joint_indices"],
        ],
        outputs=state["outputs"],
        device="cpu",
    )


def test_policy_targets_persist_for_fifty_substeps_reset_and_next_step_updates():
    targets = _CpuTargetBuffers()
    env = _bare_env(targets)
    actions = torch.tensor([[0.4, 0.2, 0.5, -0.5], [-0.3, 0.6, -0.4, 0.4]])

    env._pre_physics_step(actions)
    assert targets._control.staged == ["velocity", "position", "effort"]
    initial = targets.snapshot()
    assert torch.count_nonzero(initial["velocity"][:, :14]) == 28
    assert torch.count_nonzero(initial["position"][:, 14:16]) == 4
    assert torch.count_nonzero(initial["effort"][:, :14]) > 0

    wheel_ids = env._wheel_ids
    actuator = IdealPDActuator(
        IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.04,
            actuator_effort_limit=2.0,
            actuator_velocity_limit=110.0,
        ),
        joint_names=[f"wheel_{index}" for index in range(14)],
        joint_ids=wheel_ids,
        num_envs=2,
        device="cpu",
        stiffness=0.0,
        damping=0.04,
        actuator_effort_limit=2.0,
        actuator_velocity_limit=110.0,
    )
    implicit = _implicit_cpu_state()
    wheel_torques = []
    gimbal_torques = []

    for substep in range(50):
        env._apply_action()
        wheel_qd = targets.target_command.velocity.torch[:, wheel_ids] + 0.1 * substep
        gimbal_qd = torch.tensor([[0.01 * substep, -0.007 * substep]] * 2)
        env.rover.data.joint_vel.torch[:, :14] = wheel_qd
        env.rover.data.joint_vel.torch[:, 14:16] = gimbal_qd
        state_joint_vel = wp.to_torch(implicit["joint_vel"])
        state_joint_vel[:, :14] = wheel_qd
        state_joint_vel[:, 14:16] = gimbal_qd

        wheel_action = ArticulationActions(
            joint_positions=torch.zeros((2, 14), dtype=torch.float32),
            joint_velocities=targets.target_command.velocity.torch[:, wheel_ids],
            joint_efforts=targets.target_command.effort.torch[:, wheel_ids],
            joint_indices=wheel_ids,
        )
        actuator.compute(
            wheel_action,
            joint_pos=torch.zeros((2, 14), dtype=torch.float32),
            joint_vel=wheel_qd,
        )
        wheel_torques.append(wheel_action.joint_efforts[0, 0].item())

        _compute_implicit(targets, implicit)
        gimbal_torques.append(wp.to_torch(implicit["outputs"][0])[0, 0].item())

    after_substeps = targets.snapshot()
    for name in initial:
        assert torch.equal(after_substeps[name], initial[name]), name
    assert targets._control.staged == ["velocity", "position", "effort"]
    assert wheel_torques[0] > wheel_torques[-1]
    assert gimbal_torques[0] != gimbal_torques[-1]

    # Match _reset_idx: write zero commands only to the selected environment row.
    zeros = torch.zeros((1, 16), dtype=torch.float32)
    targets.target_command.set_position_index(value=zeros, env_ids=[1])
    targets.target_command.set_velocity_index(value=zeros, env_ids=[1])
    targets.target_command.set_effort_index(
        value=torch.zeros((1, 14), dtype=torch.float32), joint_ids=wheel_ids, env_ids=[1]
    )
    reset = targets.snapshot()
    for name in reset:
        assert torch.count_nonzero(reset[name][1]) == 0, name
        assert torch.equal(reset[name][0], initial[name][0]), name

    env.track_command[1] = 0.0
    env.wheel_target_velocity[1] = 0.0
    env.gimbal_target[1] = 0.0
    env.command[1] = 0.0
    env._drive_pi.reset(torch.tensor([1]))
    env.rover.data.joint_vel.torch[1] = 0.0
    _compute_implicit(targets, implicit)
    after_reset_compute = targets.snapshot()
    for name in reset:
        assert torch.equal(after_reset_compute[name], reset[name]), name

    env._pre_physics_step(torch.tensor([[0.1, -0.2, 0.2, -0.2], [0.5, -0.4, 0.3, -0.3]]))
    next_step = targets.snapshot()
    assert not torch.equal(next_step["velocity"][1, :14], reset["velocity"][1, :14])
    assert not torch.equal(next_step["position"][1, 14:16], reset["position"][1, 14:16])
    assert not torch.equal(next_step["effort"][1, :14], reset["effort"][1, :14])
    assert targets._control.staged == [
        "velocity", "position", "effort",
        "position", "velocity", "effort",
        "velocity", "position", "effort",
    ]
