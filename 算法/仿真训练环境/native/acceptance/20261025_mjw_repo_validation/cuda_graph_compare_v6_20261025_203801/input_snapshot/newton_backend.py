"""Newton XPBD backend with independent per-world state (same interface as MujocoCPUBackend).

Model per world is identical to physics_backend.build_mjcf:
  19 bodies / 19 joints / 24 DOF (6-DOF free base + 14 wheels + 2 gimbal + 2 guides)
  + 14 mimic relations via ModelBuilder.set_joint_mimic.

Drive semantics (prototype-level deviation, documented):
  * wheel velocity servo through `control.joint_target_qd` with joint target_kd=0.04.
  * collision contacts are refreshed before every XPBD physics substep.
  * gimbal position servo through `control.joint_target_q` (target_ke=1, target_kd=0.1)
  * the CPU backend's explicit per-wheel torque + side PI feed-forward is NOT applied
    here; target velocities already include the 1.5 m/s^2 slew and action mapping.
  * terrain/plant physics identical (flat ground plane), lidar uses its own model.

Only imported when `--physics newton_xpbd` is requested, so the CPU default path never
imports newton/warp.
"""
from __future__ import annotations

import json
import math

import numpy as np

from physics_backend import (CONTACT_LENGTH, DRIVE_KP, EFFORT_LIMIT, GIMBAL_JOINT_NAMES,
                             GIMBAL_OFFSET_Z, GIMBAL_PITCH_RATE, GIMBAL_YAW_RATE, GUIDE_MASS,
                             GUIDE_RADIUS, GUIDE_RATIO, GUIDE_X, GUIDE_Z, MAX_TRACK_SPEED,
                             PITCH_MAX, PITCH_MIN, TRACK_ACCELERATION, TRACK_SEPARATION,
                             WHEEL_HALF_WIDTH, WHEEL_JOINT_NAMES, WHEEL_MASS, WHEEL_RADIUS,
                             WHEEL_Z, WHEELS_PER_SIDE, YAW_LIMIT, SideIntegralPI)


def _diag_inertia(ixx, iyy, izz):
    import warp as wp
    return wp.mat33(np.array([[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]], dtype=np.float32))


# explicit inertias (isaac_field/assets.py formulas); avoids Newton inertia
# validation re-computation for shapes authored with density=0 (which produced NaN)
CHASSIS_I = (26.4 * (0.15 ** 2 + 0.075 ** 2) / 12.0,
             26.4 * (0.4 ** 2 + 0.075 ** 2) / 12.0,
             26.4 * (0.4 ** 2 + 0.15 ** 2) / 12.0)
WHEEL_I = (0.25 * (3 * 0.045 ** 2 + 0.06 ** 2) / 12.0,
           0.5 * 0.25 * 0.045 ** 2,
           0.25 * (3 * 0.045 ** 2 + 0.06 ** 2) / 12.0)
GIMBAL_I = (0.05 * (0.06 ** 2 + 0.04 ** 2) / 12.0,
            0.05 * (0.06 ** 2 + 0.04 ** 2) / 12.0,
            0.05 * (0.06 ** 2 + 0.06 ** 2) / 12.0)
GUIDE_I = WHEEL_I


def _robot_builder(mu: float = 0.8, *, chassis_inertia=None, pitch_inertia=None,
                   shape_gap: float | None = None, shape_margin: float | None = None):
    import newton
    import warp as wp

    chassis_inertia = CHASSIS_I if chassis_inertia is None else tuple(chassis_inertia)
    pitch_inertia = GIMBAL_I if pitch_inertia is None else tuple(pitch_inertia)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    b.default_shape_cfg.mu = float(mu)
    b.default_shape_cfg.ke = 2.5e3
    b.default_shape_cfg.kd = 100.0

    def shape_cfg():
        values = {"density": 0.0, "ke": 2.5e3, "kd": 100.0, "mu": float(mu)}
        if shape_gap is not None:
            values["gap"] = float(shape_gap)
        if shape_margin is not None:
            values["margin"] = float(shape_margin)
        return newton.ModelBuilder.ShapeConfig(**values)

    b.add_ground_plane(cfg=shape_cfg())
    base = b.add_link(xform=wp.transform(p=(0.0, 0.0, 0.15), q=wp.quat_identity()),
                      mass=26.4, inertia=_diag_inertia(*chassis_inertia), label="base")
    b.add_shape_box(base, hx=0.4, hy=0.15, hz=0.075,
                    cfg=shape_cfg(), label="chassis")
    joints = [b.add_joint_free(child=base, label="base_free")]
    # gimbal yaw (z) -> pitch (y)
    yaw = b.add_link(xform=wp.transform(p=(0.0, 0.0, 0.15 + GIMBAL_OFFSET_Z), q=wp.quat_identity()),
                     mass=0.05, inertia=_diag_inertia(*GIMBAL_I), label="gimbal_yaw")
    b.add_shape_box(yaw, hx=0.03, hy=0.03, hz=0.02,
                    cfg=shape_cfg(), label="gimbal_yaw_geom")
    joints.append(b.add_joint_revolute(
        parent=base, child=yaw, axis=newton.Axis.Z,
        parent_xform=wp.transform(p=(0.0, 0.0, GIMBAL_OFFSET_Z), q=wp.quat_identity()),
        target_ke=1.0, target_kd=0.1, effort_limit=1.0,
        limit_lower=-YAW_LIMIT, limit_upper=YAW_LIMIT, label="gimbal_yaw_joint"))
    pitch = b.add_link(xform=wp.transform(p=(0.0, 0.0, 0.15 + GIMBAL_OFFSET_Z), q=wp.quat_identity()),
                       mass=0.05, inertia=_diag_inertia(*pitch_inertia), label="gimbal_pitch")
    b.add_shape_box(pitch, hx=0.02, hy=0.02, hz=0.02,
                    cfg=shape_cfg(), label="gimbal_pitch_geom")
    joints.append(b.add_joint_revolute(
        parent=yaw, child=pitch, axis=newton.Axis.Y,
        target_ke=1.0, target_kd=0.1, effort_limit=1.0,
        limit_lower=PITCH_MIN, limit_upper=PITCH_MAX, label="gimbal_pitch_joint"))

    wheel_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.5707963267948966)
    wheel_joints = {"left": [], "right": []}
    for side, y in (("left", TRACK_SEPARATION / 2), ("right", -TRACK_SEPARATION / 2)):
        for i in range(WHEELS_PER_SIDE):
            x = (i - (WHEELS_PER_SIDE - 1) / 2.0) * (CONTACT_LENGTH / (WHEELS_PER_SIDE - 1))
            link = b.add_link(xform=wp.transform(p=(x, y, 0.15 + WHEEL_Z), q=wp.quat_identity()),
                              mass=WHEEL_MASS, inertia=_diag_inertia(*WHEEL_I), label=f"{side}_wheel_{i}_link")
            b.add_shape_cylinder(link, xform=wp.transform(p=(0.0, 0.0, 0.0), q=wheel_quat),
                                 radius=WHEEL_RADIUS, half_height=WHEEL_HALF_WIDTH,
                                 cfg=shape_cfg(),
                                 label=f"{side}_wheel_{i}_geom")
            j = b.add_joint_revolute(parent=base, child=link, axis=newton.Axis.Y,
                                     parent_xform=wp.transform(p=(x, y, WHEEL_Z), q=wp.quat_identity()),
                                     target_ke=0.0, target_kd=DRIVE_KP,
                                     effort_limit=EFFORT_LIMIT, damping=0.2,
                                     label=f"{side}_wheel_{i}_wheel_joint")
            joints.append(j)
            wheel_joints[side].append(j)
    guide_joints = []
    for side, y in (("left", TRACK_SEPARATION / 2), ("right", -TRACK_SEPARATION / 2)):
        link = b.add_link(xform=wp.transform(p=(GUIDE_X, y, 0.15 + GUIDE_Z), q=wp.quat_identity()),
                          mass=GUIDE_MASS, inertia=_diag_inertia(*GUIDE_I), label=f"{side}_front_guide_link")
        b.add_shape_cylinder(link, xform=wp.transform(p=(0.0, 0.0, 0.0), q=wheel_quat),
                             radius=GUIDE_RADIUS, half_height=WHEEL_HALF_WIDTH,
                             cfg=shape_cfg(),
                             label=f"{side}_front_guide_geom")
        j = b.add_joint_revolute(parent=base, child=link, axis=newton.Axis.Y,
                                 parent_xform=wp.transform(p=(GUIDE_X, y, GUIDE_Z), q=wp.quat_identity()),
                                 damping=0.2, label=f"{side}_front_guide_wheel_joint")
        joints.append(j)
        guide_joints.append(j)
    b.add_articulation(joints, label="rover")
    relations = 0
    for side in ("left", "right"):
        leader = wheel_joints[side][0]
        for follower in wheel_joints[side][1:]:
            b.set_joint_mimic(follower, leader, coeffs=(0.0, 1.0))
            relations += 1
    for side, j in zip(("left", "right"), guide_joints):
        b.set_joint_mimic(j, wheel_joints[side][0], coeffs=(0.0, GUIDE_RATIO))
        relations += 1
    b.color()
    return b, wheel_joints, guide_joints, relations


class NewtonXPBDGBackend:
    """Batched Newton XPBD backend; the requested device is used for every model buffer."""

    def _robot_builder_options(self) -> dict:
        """Return an optional model profile; the XPBD default keeps its historical parameters."""
        return {}

    def _build_model(self, newton):
        robot, _, _, relations = _robot_builder(mu=self.mu, **self._robot_builder_options())
        self._mimic_relations = int(relations)
        main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        main.add_world(robot, label_prefix="w0_")
        if self.nworld > 1:
            main.replicate(robot, world_count=self.nworld - 1,
                           label_prefixes=[f"w{i}_" for i in range(1, self.nworld)])
        return main.finalize(device=self.device)

    def _build_solver(self, newton):
        return newton.solvers.SolverXPBD(self.model, iterations=10)

    def _build_contact_system(self, newton):
        pipeline = newton.CollisionPipeline(self.model)
        return pipeline, pipeline.contacts()

    def _initialize_solver_state(self) -> None:
        """Hook for solver-specific observables initialized before the first reset."""
        return None

    def __init__(self, nworld: int = 64, *, dt: float = 1e-3, decimation: int = 50,
                 effort_limit: float = EFFORT_LIMIT, kp: float = DRIVE_KP, ki: float = 0.1,
                 mu: float = 0.8, seed: int = 0, device: str | None = None):
        import newton
        import warp as wp

        self._newton = newton
        self._wp = wp
        newton.use_coord_layout_targets = True
        wp.init()
        self.nworld = int(nworld)
        self.dt = float(dt)
        self.decimation = int(decimation)
        self.policy_dt = self.dt * self.decimation
        self.effort_limit = float(effort_limit)
        self.mu = float(mu)
        if device is None:
            device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
        self.device = wp.get_device(device)

        self.model = self._build_model(newton)

        if str(self.model.device) != str(self.device):
            raise RuntimeError(
                f"requested Newton device {self.device}, but model finalized on {self.model.device}"
            )
        self.solver = self._build_solver(newton)
        self.collision_pipeline, self.contacts = self._build_contact_system(newton)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.joint_q = wp.zeros_like(self.model.joint_q)
        self.joint_qd = wp.zeros_like(self.model.joint_qd)
        self._initial_joint_q = self.model.joint_q.numpy().copy()
        self._initial_joint_qd = self.model.joint_qd.numpy().copy()
        self.joint_q.assign(self._initial_joint_q)
        self.joint_qd.assign(self._initial_joint_qd)
        newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state_1)

        joint_labels = [str(x) for x in self.model.joint_label]
        joint_world = self.model.joint_world.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        target_q_start = self.model.joint_target_q_start.numpy()
        self.world_joint_idx = {}
        self.world_joint_names = {}
        self.world_joint_slices = {}
        for w in range(self.nworld):
            indices = np.flatnonzero(joint_world == w).tolist()
            by_name = {}
            for j in indices:
                name = joint_labels[j].rsplit("/", 1)[-1]
                if name in by_name:
                    raise RuntimeError(f"world {w}: duplicate joint label {name!r}")
                by_name[name] = int(j)
            if not indices:
                raise RuntimeError(f"world {w}: no joints found from model.joint_world")
            self.world_joint_idx[w] = [int(j) for j in indices]
            self.world_joint_names[w] = by_name
            self.world_joint_slices[w] = [
                (int(q_start[j]), int(q_start[j + 1]),
                 int(qd_start[j]), int(qd_start[j + 1]))
                for j in indices
            ]

        self.wheel_joint_idx = {
            side: [[self.world_joint_names[w][f"{side}_wheel_{k}_wheel_joint"]
                    for k in range(WHEELS_PER_SIDE)] for w in range(self.nworld)]
            for side in ("left", "right")
        }
        self.guide_joint_idx = [
            [self.world_joint_names[w][f"{side}_front_guide_wheel_joint"]
             for side in ("left", "right")]
            for w in range(self.nworld)
        ]
        self.gimbal_joint_idx = [
            [self.world_joint_names[w][name] for name in GIMBAL_JOINT_NAMES]
            for w in range(self.nworld)
        ]
        self.base_joint_idx = [self.world_joint_names[w]["base_free"] for w in range(self.nworld)]
        self.wheel_dof = {
            side: [[int(qd_start[j]) for j in self.wheel_joint_idx[side][w]]
                   for w in range(self.nworld)]
            for side in ("left", "right")
        }
        # Velocity targets use DOF indices; position targets use target-coordinate indices.
        self.wheel_target_dof = {side: [indices.copy() for indices in table]
                                 for side, table in self.wheel_dof.items()}
        self.gimbal_q_coord = [
            [int(q_start[j]) for j in self.gimbal_joint_idx[w]]
            for w in range(self.nworld)
        ]
        self.gimbal_target_q_coord = [
            [int(target_q_start[j]) for j in self.gimbal_joint_idx[w]]
            for w in range(self.nworld)
        ]
        self.gimbal_qd_dof = [
            [int(qd_start[j]) for j in self.gimbal_joint_idx[w]]
            for w in range(self.nworld)
        ]
        self.base_coord = [int(q_start[j]) for j in self.base_joint_idx]

        articulation_world = self.model.articulation_world.numpy()
        self.world_articulation_idx = {}
        for w in range(self.nworld):
            indices = np.flatnonzero(articulation_world == w).tolist()
            if len(indices) != 1:
                raise RuntimeError(f"world {w}: expected one articulation, got {indices}")
            self.world_articulation_idx[w] = [int(i) for i in indices]

        body_labels = [str(x) for x in self.model.body_label]
        body_world = self.model.body_world.numpy()
        self.world_body_idx = {
            w: np.flatnonzero(body_world == w).astype(np.int64).tolist()
            for w in range(self.nworld)
        }
        self.base_body = []
        for w in range(self.nworld):
            matches = [i for i, lab in enumerate(body_labels)
                       if int(body_world[i]) == w and lab.rsplit("/", 1)[-1] == "base"]
            if len(matches) != 1:
                raise RuntimeError(f"world {w}: expected one base body, got {matches}")
            self.base_body.append(matches[0])
        self.policy_kp = float(kp)
        self.policy_ki = float(ki)
        self.pi = SideIntegralPI(self.nworld, kp=kp, ki=ki, policy_dt=self.policy_dt,
                                 limit=self.effort_limit)
        self.rng = np.random.default_rng(seed)
        self.track_command = np.zeros((self.nworld, 2))
        self.gimbal_target = np.zeros((self.nworld, 2))
        self.wheel_target_qd = np.zeros((self.nworld, 2 * WHEELS_PER_SIDE))
        self.last_effort = np.zeros((self.nworld, 2 * WHEELS_PER_SIDE))
        self.step_count = np.zeros(self.nworld, dtype=np.int64)
        self._initialize_solver_state()
        self.reset()

    def _sync_reset_joint_coordinates(self, ids: np.ndarray, q: np.ndarray, qd: np.ndarray) -> None:
        """Update only selected worlds in both reduced-coordinate ping-pong states."""
        for state in (self.state_0, self.state_1):
            state_q = state.joint_q.numpy().copy()
            state_qd = state.joint_qd.numpy().copy()
            for world_value in ids:
                world = int(world_value)
                for q0, q1, qd0, qd1 in self.world_joint_slices[world]:
                    state_q[q0:q1] = q[q0:q1]
                    state_qd[qd0:qd1] = qd[qd0:qd1]
            state.joint_q.assign(state_q)
            state.joint_qd.assign(state_qd)

    # ------------------------------------------------------------------
    def reset(self, env_ids=None, poses: np.ndarray | None = None) -> None:
        ids = np.arange(self.nworld, dtype=np.int64) if env_ids is None else np.asarray(env_ids, np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.nworld):
            raise ValueError(f"env_ids must be a one-dimensional subset of [0, {self.nworld})")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("env_ids must not contain duplicates")
        if poses is None:
            poses = np.zeros((len(ids), 4), dtype=np.float64)
        else:
            poses = np.asarray(poses, dtype=np.float64)
        if poses.shape != (len(ids), 4):
            raise ValueError(f"poses must have shape ({len(ids)}, 4), got {poses.shape}")
        if not len(ids):
            return

        # Inverse kinematics starts from the live state so untouched worlds keep their
        # current joint values. eval_fk below writes only the selected articulations.
        self._newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        q = self.joint_q.numpy().copy()
        qd = self.joint_qd.numpy().copy()
        target_qd = self.control.joint_target_qd.numpy().copy()
        target_q = self.control.joint_target_q.numpy().copy()
        articulation_ids = []
        for row, env_id_value in enumerate(ids):
            env_id = int(env_id_value)
            articulation_ids.extend(self.world_articulation_idx[env_id])
            for q0, q1, qd0, qd1 in self.world_joint_slices[env_id]:
                q[q0:q1] = self._initial_joint_q[q0:q1]
                qd[qd0:qd1] = self._initial_joint_qd[qd0:qd1]
            x, y, yaw = (float(poses[row, i]) for i in range(3))
            z = 0.15 + float(poses[row, 3])
            c = self.base_coord[env_id]
            q[c:c + 7] = [x, y, z, 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
            for side in ("left", "right"):
                for dof in self.wheel_target_dof[side][env_id]:
                    target_qd[dof] = 0.0
            for g, coord in enumerate(self.gimbal_target_q_coord[env_id]):
                target_q[coord] = self._initial_joint_q[self.gimbal_q_coord[env_id][g]]
            self.track_command[env_id] = 0.0
            self.gimbal_target[env_id] = 0.0
            self.wheel_target_qd[env_id] = 0.0
            self.last_effort[env_id] = 0.0
            self.step_count[env_id] = 0

        self.joint_q.assign(q)
        self.joint_qd.assign(qd)
        indices = self._wp.array(np.asarray(articulation_ids, dtype=np.int32),
                                 dtype=self._wp.int32, device=self.model.device)
        self._newton.eval_fk(self.model, self.joint_q, self.joint_qd,
                             self.state_0, indices=indices)
        self._newton.eval_fk(self.model, self.joint_q, self.joint_qd,
                             self.state_1, indices=indices)
        self._sync_reset_joint_coordinates(ids, q, qd)
        self.control.joint_target_qd.assign(target_qd)
        self.control.joint_target_q.assign(target_q)
        self.pi.reset(ids)

    def step(self, actions: np.ndarray) -> None:
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        desired = actions[:, :2] * MAX_TRACK_SPEED
        change = TRACK_ACCELERATION * self.policy_dt
        self.track_command += np.clip(desired - self.track_command, -change, change)
        self.wheel_target_qd = (self.track_command / WHEEL_RADIUS)[:, self.pi.wheel_side]
        actual = self.read()["wheel_qd"]
        ff = self.pi.update(self.wheel_target_qd, actual)
        self.gimbal_target += actions[:, 2:4] * np.array([GIMBAL_YAW_RATE, GIMBAL_PITCH_RATE]) * self.policy_dt
        self.gimbal_target[:, 0] = (self.gimbal_target[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
        self.gimbal_target[:, 1] = np.clip(self.gimbal_target[:, 1], PITCH_MIN, PITCH_MAX)
        # Write each world's targets through its actual Newton joint/coordinate map.
        qd_target = self.control.joint_target_qd.numpy().copy()
        q_target = self.control.joint_target_q.numpy().copy()
        for w in range(self.nworld):
            for side in ("left", "right"):
                base = 0 if side == "left" else WHEELS_PER_SIDE
                for k, dof in enumerate(self.wheel_target_dof[side][w]):
                    qd_target[dof] = self.wheel_target_qd[w, base + k]
            for g, coord in enumerate(self.gimbal_target_q_coord[w]):
                q_target[coord] = self.gimbal_target[w, g]
        self.control.joint_target_qd.assign(qd_target)
        self.control.joint_target_q.assign(q_target)
        for _ in range(self.decimation):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.step_count += 1
        self.last_effort = ff

    # ------------------------------------------------------------------
    def read(self) -> dict[str, np.ndarray]:
        self._newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        jq = self.joint_q.numpy()
        jqd = self.joint_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        n = self.nworld
        pos = np.zeros((n, 3))
        quat = np.zeros((n, 4))
        lin = np.zeros((n, 3))
        ang = np.zeros((n, 3))
        wheel_qd = np.zeros((n, 2 * WHEELS_PER_SIDE))
        gimbal_q = np.zeros((n, 2))
        gimbal_qd = np.zeros((n, 2))
        for w in range(n):
            b = self.base_body[w]
            pos[w] = body_q[b, 0:3]
            qx, qy, qz, qw = body_q[b, 3:7]
            quat[w] = [qw, qx, qy, qz]
            lin[w] = body_qd[b, 0:3]
            ang[w] = body_qd[b, 3:6]
            for side, base in (("left", 0), ("right", WHEELS_PER_SIDE)):
                for k, dof in enumerate(self.wheel_dof[side][w]):
                    wheel_qd[w, base + k] = jqd[dof]
            for g, coord in enumerate(self.gimbal_q_coord[w]):
                gimbal_q[w, g] = jq[coord]
                gimbal_qd[w, g] = jqd[self.gimbal_qd_dof[w][g]]
        return {"pos": pos, "quat_wxyz": quat, "lin_vel": lin, "ang_vel": ang,
                "wheel_qd": wheel_qd, "gimbal_q": gimbal_q, "gimbal_qd": gimbal_qd,
                "effort": self.last_effort.copy(), "step_count": self.step_count.copy(),
                "track_command": self.track_command.copy()}

    def identity(self) -> dict:
        shape_world = self.model.shape_world.numpy()
        material_arrays = {field: getattr(self.model, f"shape_material_{field}").numpy()
                           for field in ("mu", "ke", "kd")}
        def material_values(field: str, world: int) -> list[float]:
            values = material_arrays[field]
            selected = values[shape_world == world]
            if selected.size == 0:
                raise RuntimeError(f"world {world} has no shape material values for {field}")
            return sorted({float(value) for value in selected})
        materials = {field: material_values(field, 0) for field in ("mu", "ke", "kd")}
        for world in range(1, self.nworld):
            if any(material_values(field, world) != values
                   for field, values in materials.items()):
                raise RuntimeError(f"world {world} material values differ from world 0")
        return {"engine": "newton_xpbd", "backend_revision": "newton_xpbdg_v3_contacts_worldmap_materialcfg",
                "device": str(self.model.device),
                "newton": getattr(self._newton, "__version__", "?"),
                "warp": self._wp.config.version, "mujoco": None,
                "nworld": self.nworld, "dt_s": self.dt, "decimation": self.decimation,
                "policy_dt_s": self.policy_dt, "dof": int(self.model.joint_dof_count),
                "joints": int(self.model.joint_count), "bodies": int(self.model.body_count),
                "mimic_relations_per_world": int(self._mimic_relations),
                "effort_limit_nm": self.effort_limit, "drive_kp": self.policy_kp,
                "drive_ki": self.policy_ki, "mu_requested": self.mu,
                "shape_material_mu_values_per_world": materials["mu"],
                "shape_material_ke_values_per_world": materials["ke"],
                "shape_material_kd_values_per_world": materials["kd"],
                "max_track_speed_mps": MAX_TRACK_SPEED,
                "drive_semantics": "velocity targets (joint_target_qd) with target_kd=0.04; side PI not applied",
                "note": "World-indexed Newton XPBD backend; model, state, controls, and contacts share the requested device"}
