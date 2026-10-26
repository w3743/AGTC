"""MuJoCo CPU physics backend for the native-stack M0 throughput prototype.

Model (per world), matching isaac_field's ROBOT_SPEC / FrontGuideSpec / drive semantics:
  * 6-DOF free base (chassis box 0.8 x 0.3 x 0.15 m, 26.4 kg)
  * 14 support-wheel hinge joints (7 per side, 45 mm radius, 0.25 kg each)
  * 2 gimbal hinges (yaw about z, pitch about y, joint effort limit 1 Nm)
  * 2 front-guide hinges (radius 45 mm, same-side track-speed coupled)
  * 14 mimic equalities: q_wheel_i = 1.0*q_wheel_0 per side (12) and
    q_guide = (R_main/R_guide)*q_wheel_0 per side (2)
  => 19 bodies / 19 joints / 24 DOF, exactly the口径 fixed in INTEGRATION_ASSESSMENT.md.

Control semantics copied from env.py + drive_control.py:
  * action[0:2] in [-1,1] -> desired track speed (max_speed = 15 km/h)
  * first-order acceleration slew 1.5 m/s^2 per policy step
  * wheel target qd = track speed / 0.045 m
  * per-wheel P (kp=0.04) + side-mean integral feed-forward (ki=0.1, limit 2 Nm),
    conditional anti-windup, effort clamped to +/-2 Nm
  * gimbal target integrated from action[2:4] * (1.8, 1.2) rad/s * 0.05 s
  * physics dt = 1 ms, decimation = 50 (policy 20 Hz), mu = 0.8

Prototype deviations (记录在 observation_contract / REPORT):
  * effort is recomputed each physics substep from the current wheel qd with the
    integral held for the policy step (Isaac Lab actuator also runs per substep,
    so this is equivalent in spirit; integral updates once per policy step).
  * ground is a single MuJoCo plane per MjData; terrain undulation is not modeled
    in physics for M0 (lidar uses its own analytic terrain model).
"""
from __future__ import annotations

import json
import math

import mujoco
import numpy as np

# --- isaac_field/assets.py ROBOT_SPEC + FrontGuideSpec (documented literals) ---
BODY_LENGTH = 0.8
BODY_WIDTH = 0.3
BODY_HEIGHT = 0.15
CHASSIS_MASS = 26.4
WHEEL_MASS = 0.25
WHEEL_RADIUS = 0.045
WHEEL_HALF_WIDTH = 0.03
WHEEL_Z = -0.105
TRACK_SEPARATION = 0.41
WHEELS_PER_SIDE = 7
CONTACT_LENGTH = 0.595
WHEEL_SPACING = CONTACT_LENGTH / (WHEELS_PER_SIDE - 1)
GIMBAL_MASS = 0.05
GIMBAL_OFFSET_Z = 0.25
GUIDE_X = 0.42
GUIDE_Z = WHEEL_Z + 0.05
GUIDE_RADIUS = 0.045
GUIDE_MASS = 0.25
GUIDE_RATIO = WHEEL_RADIUS / GUIDE_RADIUS

# --- task / drive constants (isaac_field/env.py, task.py, drive_control.py) ---
MAX_TRACK_SPEED = 15.0 / 3.6       # m/s
TRACK_ACCELERATION = 1.5           # m/s^2, per-policy-step slew
DRIVE_KP = 0.04
DRIVE_KI = 0.1
EFFORT_LIMIT = 2.0                 # Nm per wheel
GIMBAL_YAW_RATE = 1.8              # rad/s command scale
GIMBAL_PITCH_RATE = 1.2
PITCH_MIN = math.radians(-25.0)
PITCH_MAX = math.radians(60.0)
YAW_LIMIT = math.pi
MOTOR_TORQUE_CONSTANT = 0.035      # Nm/A (model parameter, uncalibrated)
DRIVE_REDUCTION = 4.0

WHEEL_JOINT_NAMES = [
    f"{side}_wheel_{i}_wheel_joint"
    for side, _ in (("left", 1), ("right", -1))
    for i in range(WHEELS_PER_SIDE)
]
GUIDE_JOINT_NAMES = [f"{side}_front_guide_wheel_joint" for side in ("left", "right")]
GIMBAL_JOINT_NAMES = ["gimbal_yaw_joint", "gimbal_pitch_joint"]
LEFT_WHEEL_IDS = np.arange(0, WHEELS_PER_SIDE)
RIGHT_WHEEL_IDS = np.arange(WHEELS_PER_SIDE, 2 * WHEELS_PER_SIDE)
WHEEL_SITE_IDS = np.concatenate([LEFT_WHEEL_IDS, RIGHT_WHEEL_IDS])


def build_mjcf(mu: float = 0.8, iterations: int = 40, ls_iterations: int = 8,
               dt: float = 1e-3, timestep_solver: str = "Newton") -> str:
    quat = "0.70710678 0.70710678 0 0"  # cylinder local z -> y
    base = [f'<body name="base" pos="0 0 0.15"><freejoint name="base_free"/>',
            f'<geom name="chassis" type="box" size="{BODY_LENGTH/2} {BODY_WIDTH/2} {BODY_HEIGHT/2}" '
            f'mass="{CHASSIS_MASS}"/>']
    # gimbal: yaw (z) -> pitch (y); pitch limits from assets.py
    base.append(
        f'<body name="gimbal_yaw_link" pos="0 0 {GIMBAL_OFFSET_Z}">'
        f'<joint name="gimbal_yaw_joint" type="hinge" axis="0 0 1" range="{-YAW_LIMIT} {YAW_LIMIT}" damping="0.1"/>'
        f'<geom name="gimbal_yaw_geom" type="box" size="0.03 0.03 0.02" mass="{GIMBAL_MASS}"/>'
        f'<body name="gimbal_pitch_link" pos="0 0 0">'
        f'<joint name="gimbal_pitch_joint" type="hinge" axis="0 1 0" range="{PITCH_MIN} {PITCH_MAX}" damping="0.1"/>'
        f'<geom name="gimbal_pitch_geom" type="box" size="0.02 0.02 0.02" mass="{GIMBAL_MASS}"/>'
        f'</body></body>')
    for side, y in (("left", TRACK_SEPARATION / 2), ("right", -TRACK_SEPARATION / 2)):
        for i in range(WHEELS_PER_SIDE):
            x = (i - (WHEELS_PER_SIDE - 1) / 2.0) * WHEEL_SPACING
            base.append(
                f'<body name="{side}_wheel_{i}_link" pos="{x:.6f} {y:.6f} {WHEEL_Z}">'
                f'<joint name="{side}_wheel_{i}_wheel_joint" type="hinge" axis="0 1 0" '
                f'range="-1000 1000" damping="0.2"/>'
                f'<geom name="{side}_wheel_{i}_geom" type="cylinder" size="{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}" '
                f'mass="{WHEEL_MASS}" quat="{quat}" friction="{mu} 0.005 0.0001" contype="1" conaffinity="1"/>'
                f'</body>')
        base.append(
            f'<body name="{side}_front_guide_link" pos="{GUIDE_X:.6f} {y:.6f} {GUIDE_Z:.6f}">'
            f'<joint name="{side}_front_guide_wheel_joint" type="hinge" axis="0 1 0" '
            f'range="-1000 1000" damping="0.2"/>'
            f'<geom name="{side}_front_guide_geom" type="cylinder" size="{GUIDE_RADIUS} {WHEEL_HALF_WIDTH}" '
            f'mass="{GUIDE_MASS}" quat="{quat}" friction="{mu} 0.005 0.0001" contype="1" conaffinity="1"/>'
            f'</body>')
    base.append('</body>')

    eq = []
    solref, solimp = "0.004 1", "0.999 0.9999 0.001 0.5 2"
    for side in ("left", "right"):
        leader = f"{side}_wheel_0_wheel_joint"
        for i in range(1, WHEELS_PER_SIDE):
            eq.append(f'<joint joint1="{side}_wheel_{i}_wheel_joint" joint2="{leader}" '
                      f'polycoef="0 1 0 0 0" solref="{solref}" solimp="{solimp}"/>')
        eq.append(f'<joint joint1="{side}_front_guide_wheel_joint" joint2="{leader}" '
                  f'polycoef="0 {GUIDE_RATIO:.6f} 0 0 0" solref="{solref}" solimp="{solimp}"/>')

    act = [f'<motor joint="{name}" gear="1" ctrlrange="{-EFFORT_LIMIT} {EFFORT_LIMIT}"/>'
           for name in WHEEL_JOINT_NAMES]
    act.append(f'<position joint="gimbal_yaw_joint" kp="1" kv="0.1" '
               f'ctrlrange="{-YAW_LIMIT} {YAW_LIMIT}" forcerange="-1 1"/>')
    act.append(f'<position joint="gimbal_pitch_joint" kp="1" kv="0.1" '
               f'ctrlrange="{PITCH_MIN} {PITCH_MAX}" forcerange="-1 1"/>')
    return f'''<mujoco model="native_rover18">
  <option timestep="{dt}" gravity="0 0 -9.81" iterations="{iterations}" ls_iterations="{ls_iterations}"
          solver="{timestep_solver}" cone="elliptic" impratio="10"/>
  <default><geom rgba="0.6 0.6 0.6 1"/></default>
  <worldbody>
    <geom name="ground" type="plane" size="200 200 0.1" rgba=".35 .35 .35 1"
          friction="{mu} 0.005 0.0001" contype="1" conaffinity="1"/>
    {"".join(base)}
  </worldbody>
  <equality>{"".join(eq)}</equality>
  <actuator>{"".join(act)}</actuator>
</mujoco>'''


class SideIntegralPI:
    """Two-state left/right integral feed-forward (drive_control.py semantics)."""

    def __init__(self, num_envs: int, *, kp: float, ki: float, policy_dt: float, limit: float):
        self.num_envs = int(num_envs)
        self.kp = float(kp)
        self.ki = float(ki)
        self.policy_dt = float(policy_dt)
        self.limit = float(limit)
        self.integral = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.wheel_side = np.concatenate([np.zeros(WHEELS_PER_SIDE, np.int64),
                                          np.ones(WHEELS_PER_SIDE, np.int64)])
        self.left_idx = np.flatnonzero(self.wheel_side == 0)
        self.right_idx = np.flatnonzero(self.wheel_side == 1)

    def reset(self, env_ids=None):
        idx = slice(None) if env_ids is None else np.asarray(env_ids, np.int64)
        self.integral[idx] = 0.0

    def update(self, target_qd: np.ndarray, actual_qd: np.ndarray) -> np.ndarray:
        """target/actual: (num_envs, 14) -> per-wheel feed-forward (num_envs, 14)."""
        left_target = target_qd[:, self.left_idx].mean(axis=1)
        right_target = target_qd[:, self.right_idx].mean(axis=1)
        left_actual = actual_qd[:, self.left_idx].mean(axis=1)
        right_actual = actual_qd[:, self.right_idx].mean(axis=1)
        error = np.stack([left_target - left_actual, right_target - right_actual], axis=1)
        p_estimate = self.kp * error
        total = p_estimate + self.integral
        blocked = ((total > self.limit) & (error > 0.0)) | ((total < -self.limit) & (error < 0.0))
        delta = np.where(blocked, 0.0, error * (self.ki * self.policy_dt))
        self.integral = np.clip(self.integral + delta, -self.limit, self.limit)
        return self.integral[:, self.wheel_side]


class MujocoCPUBackend:
    """64 independent MjData sharing one MjModel (MuJoCo CPU, single thread)."""

    def __init__(self, nworld: int = 64, *, dt: float = 1e-3, decimation: int = 50,
                 effort_limit: float = EFFORT_LIMIT, kp: float = DRIVE_KP, ki: float = DRIVE_KI,
                 mu: float = 0.8, seed: int = 0):
        self.nworld = int(nworld)
        self.dt = float(dt)
        self.decimation = int(decimation)
        self.policy_dt = self.dt * self.decimation
        self.effort_limit = float(effort_limit)
        self.mu = float(mu)
        self.mjm = mujoco.MjModel.from_xml_string(build_mjcf(mu=mu, dt=self.dt))
        self.datas = [mujoco.MjData(self.mjm) for _ in range(self.nworld)]
        if int(self.mjm.nv) != 24:
            raise RuntimeError(f"expected 24 DOF (6 base + 18 joints), got {self.mjm.nv}")
        if int(self.mjm.neq) != 14:
            raise RuntimeError(f"expected 14 mimic equalities, got {self.mjm.neq}")
        self.policy_dt = float(dt) * int(decimation)
        self._kp = float(kp)
        self._ki = float(ki)
        self.pi = SideIntegralPI(self.nworld, kp=kp, ki=ki, policy_dt=self.policy_dt,
                                 limit=self.effort_limit)
        self.rng = np.random.default_rng(seed)
        self.track_command = np.zeros((self.nworld, 2), dtype=np.float64)
        self.gimbal_target = np.zeros((self.nworld, 2), dtype=np.float64)
        self.wheel_target_qd = np.zeros((self.nworld, 2 * WHEELS_PER_SIDE), dtype=np.float64)
        self.last_effort = np.zeros((self.nworld, 2 * WHEELS_PER_SIDE), dtype=np.float64)
        self.step_count = np.zeros(self.nworld, dtype=np.int64)
        # qvel layout: 0:3 base linear, 3:6 base angular, then joint dofs in joint order
        self._base_qvel = slice(0, 6)
        self.reset()

    # ---- helpers -------------------------------------------------------
    def _joint_qpos_adr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        return int(self.mjm.jnt_qposadr[jid])

    def _joint_dof_adr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        return int(self.mjm.jnt_dofadr[jid])

    def _wheel_dof_adrs(self) -> np.ndarray:
        return np.asarray([self._joint_dof_adr(n) for n in WHEEL_JOINT_NAMES], dtype=np.int64)

    def _wheel_qpos_adrs(self) -> np.ndarray:
        return np.asarray([self._joint_qpos_adr(n) for n in WHEEL_JOINT_NAMES], dtype=np.int64)

    def _gimbal_dof_adrs(self) -> np.ndarray:
        return np.asarray([self._joint_dof_adr(n) for n in GIMBAL_JOINT_NAMES], dtype=np.int64)

    # ---- API -----------------------------------------------------------
    def reset(self, env_ids=None, poses: np.ndarray | None = None) -> None:
        """poses: (n,4) = x, y, yaw, z_extra (default zeros)."""
        ids = np.arange(self.nworld) if env_ids is None else np.asarray(env_ids, np.int64)
        if poses is None:
            poses = np.zeros((len(ids), 4), dtype=np.float64)
        for row, env_id in enumerate(ids):
            d = self.datas[env_id]
            x, y, yaw = float(poses[row, 0]), float(poses[row, 1]), float(poses[row, 2])
            z = 0.15 + float(poses[row, 3])
            d.qpos[:] = self.mjm.qpos0
            d.qpos[0], d.qpos[1], d.qpos[2] = x, y, z
            d.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]  # wxyz
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(self.mjm, d)
            self.track_command[env_id] = 0.0
            self.gimbal_target[env_id] = 0.0
            self.wheel_target_qd[env_id] = 0.0
            self.last_effort[env_id] = 0.0
            self.step_count[env_id] = 0
        self.pi.reset(ids)

    def step(self, actions: np.ndarray) -> None:
        """actions: (nworld, 4) in [-1, 1]; advances one 20 Hz policy step."""
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if actions.shape != (self.nworld, 4):
            raise ValueError(f"actions must be ({self.nworld}, 4), got {actions.shape}")
        desired = actions[:, :2] * MAX_TRACK_SPEED
        change = TRACK_ACCELERATION * self.policy_dt
        self.track_command += np.clip(desired - self.track_command, -change, change)
        self.wheel_target_qd = (self.track_command / WHEEL_RADIUS)[:, self.pi.wheel_side]
        actual = self.read()["wheel_qd"]
        ff = self.pi.update(self.wheel_target_qd, actual)
        self.gimbal_target += actions[:, 2:4] * np.array([GIMBAL_YAW_RATE, GIMBAL_PITCH_RATE]) * self.policy_dt
        self.gimbal_target[:, 0] = (self.gimbal_target[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
        self.gimbal_target[:, 1] = np.clip(self.gimbal_target[:, 1], PITCH_MIN, PITCH_MAX)
        wheel_dof = self._wheel_dof_adrs()
        for _ in range(self.decimation):
            for env_id, d in enumerate(self.datas):
                qd = d.qvel[wheel_dof]
                effort = self._kp * (self.wheel_target_qd[env_id] - qd) + ff[env_id]
                effort = np.clip(effort, -self.effort_limit, self.effort_limit)
                d.ctrl[: 2 * WHEELS_PER_SIDE] = effort
                d.ctrl[2 * WHEELS_PER_SIDE:] = self.gimbal_target[env_id]
                mujoco.mj_step(self.mjm, d)
                self.last_effort[env_id] = effort
            self.step_count += 1

    def read(self) -> dict[str, np.ndarray]:
        n = self.nworld
        pos = np.zeros((n, 3))
        quat_wxyz = np.zeros((n, 4))
        lin_vel = np.zeros((n, 3))
        ang_vel = np.zeros((n, 3))
        wheel_qd = np.zeros((n, 2 * WHEELS_PER_SIDE))
        gimbal_q = np.zeros((n, 2))
        gimbal_qd = np.zeros((n, 2))
        wheel_dof = self._wheel_dof_adrs()
        gimbal_dof = self._gimbal_dof_adrs()
        gimbal_qpos = np.asarray([self._joint_qpos_adr(nm) for nm in GIMBAL_JOINT_NAMES])
        for env_id, d in enumerate(self.datas):
            pos[env_id] = d.qpos[0:3]
            quat_wxyz[env_id] = d.qpos[3:7]
            lin_vel[env_id] = d.qvel[0:3]
            ang_vel[env_id] = d.qvel[3:6]
            wheel_qd[env_id] = d.qvel[wheel_dof]
            gimbal_q[env_id] = d.qpos[gimbal_qpos]
            gimbal_qd[env_id] = d.qvel[gimbal_dof]
        return {"pos": pos, "quat_wxyz": quat_wxyz, "lin_vel": lin_vel, "ang_vel": ang_vel,
                "wheel_qd": wheel_qd, "gimbal_q": gimbal_q, "gimbal_qd": gimbal_qd,
                "effort": self.last_effort.copy(), "step_count": self.step_count.copy(),
                "track_command": self.track_command.copy()}

    def identity(self) -> dict:
        return {"engine": "mujoco_cpu", "mujoco": mujoco.__version__,
                "nworld": self.nworld, "dt_s": self.dt, "decimation": self.decimation,
                "policy_dt_s": self.policy_dt, "dof": int(self.mjm.nv), "joints": int(self.mjm.njnt),
                "equalities": int(self.mjm.neq), "actuators": int(self.mjm.nu),
                "effort_limit_nm": self.effort_limit, "drive_kp": self._kp, "drive_ki": self._ki,
                "mu": self.mu, "max_track_speed_mps": MAX_TRACK_SPEED,
                "mimic_relations": 14}
