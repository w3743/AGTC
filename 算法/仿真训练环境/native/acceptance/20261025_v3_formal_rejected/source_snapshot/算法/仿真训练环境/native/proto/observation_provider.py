"""Observation provider for the M0 throughput prototype.

Channels (see observation_contract.json written by train_m0.py):
  lidar      : REAL  - analytic raycast (terrain sinusoid + plant stem cylinders +
                       canopy ellipsoids), same geometry families as isaac_field/task.py:431-499.
  state      : REAL  - 33 fields, ABSOLUTE_GOAL_INPUT_SCHEMA layout copied from
                       isaac_field/task.py:65-72, computed from simulator state.
  feedback   : REAL  - 37 fields (21 values + 8 valid + 8 ages) per isaac_field/feedback.py,
                       computed from simulator IMU/gimbal/motor state with the documented sim noise.
  depth      : PLACEHOLDER - constant 0.5 (NOT equivalent to the RTX depth camera).
  detections : PLACEHOLDER - zeros (NOT equivalent to the instance-segmentation detector).

Sensor clocks follow isaac_field/task.py:Config: lidar 10 Hz (30 ms latency),
depth 2.5 Hz (300 ms), camera 5 Hz (120 ms); policy step is 20 Hz.
Only the REAL lidar channel models its latency (integer policy-step delay queue);
placeholder channels keep a constant frame and only track their age/period flags.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque

import numpy as np

FIELD_SIZE = 100.0
POLICY_DT = 0.05
LIDAR_RAYS = 400
LIDAR_RANGE = 8.0
LIDAR_MIN_RANGE = 0.12
LIDAR_REL_ERROR = 0.05
DEPTH_H, DEPTH_W = 160, 88
CAMERA_HEIGHT = 0.4  # isaac_field Config.camera_height (camera origin = root z + camera_height)
MAX_DETECTIONS = 16
STATE_DIM = 33
FEEDBACK_DIM = 37
G0 = 9.80665
MAG_REFERENCE = np.array([0.0, 3000.0, -6000.0])  # ENU, magnetic north +Y
MOTOR_KT = 0.035
DRIVE_REDUCTION = 4.0
MAX_TRACK_SPEED = 15.0 / 3.6

# analytic scene (cost prototype scene; not the full isaac_field plant/rock model)
PLANTS = 400
STEM_RADIUS = 0.012
PLANT_HEIGHT = 1.0
CANOPY_RX, CANOPY_RY, CANOPY_RZ = 0.12, 0.12, 0.35
CANOPY_Z = 0.6
TERRAIN_AMPLITUDE = 0.35


def terrain_height(x, y):
    return (TERRAIN_AMPLITUDE * 0.6 * np.sin(x * 2.0 * math.pi / 8.0) * np.cos(y * 2.0 * math.pi / 9.0)
            + TERRAIN_AMPLITUDE * 0.4 * np.sin((x + y) * 2.0 * math.pi / 4.5))


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / max(np.linalg.norm(q), 1e-12)


def yaw_from_quat_wxyz(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WarpLidarCaster:
    """Warp kernel version of the analytic lidar (CUDA preferred; CPU fallback for validation).

    Same geometry as the NumPy path: all 400 plants are tested per ray (no cull), which is
    cheap on GPU. Imported/initialised only when lidar_backend="warp".
    """

    def __init__(self, az: np.ndarray, plant_xy: np.ndarray, plant_base_z: np.ndarray,
                 device: str | None = None):
        import warp as wp

        self.wp = wp
        wp.init()
        self.device = device or ("cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")
        self.az = np.asarray(az, dtype=np.float32)
        self.plant_xy = np.asarray(plant_xy, dtype=np.float32)
        self.plant_base_z = np.asarray(plant_base_z, dtype=np.float32)

        @wp.kernel
        def cast_kernel(origin: wp.array(dtype=wp.vec3), direction: wp.array(dtype=wp.vec3),
                        pxy: wp.array(dtype=wp.vec2), bz: wp.array(dtype=wp.float32),
                        n_plants: int, max_range: float, out_t: wp.array(dtype=wp.float32)):
            i = wp.tid()
            o = origin[i]
            d = direction[i]
            best = max_range
            for p in range(n_plants):
                px = pxy[p][0]
                py = pxy[p][1]
                base = bz[p]
                qx = o[0] - px
                qy = o[1] - py
                a = d[0] * d[0] + d[1] * d[1]
                if a > 1.0e-8:
                    b = 2.0 * (d[0] * qx + d[1] * qy)
                    c = qx * qx + qy * qy - STEM_RADIUS * STEM_RADIUS
                    disc = b * b - 4.0 * a * c
                    if disc >= 0.0:
                        t = (-b - wp.sqrt(disc)) / (2.0 * a)
                        z = o[2] + t * d[2]
                        if t > LIDAR_MIN_RANGE and t < best and z >= base and z <= base + PLANT_HEIGHT:
                            best = t
                cz = base + CANOPY_Z
                ox = (o[0] - px) / CANOPY_RX
                oy = (o[1] - py) / CANOPY_RY
                oz = (o[2] - cz) / CANOPY_RZ
                dx = d[0] / CANOPY_RX
                dy = d[1] / CANOPY_RY
                dz = d[2] / CANOPY_RZ
                a2 = dx * dx + dy * dy + dz * dz
                b2 = 2.0 * (ox * dx + oy * dy + oz * dz)
                c2 = ox * ox + oy * oy + oz * oz - 1.0
                disc2 = b2 * b2 - 4.0 * a2 * c2
                if disc2 >= 0.0:
                    t2 = (-b2 - wp.sqrt(disc2)) / (2.0 * a2)
                    if t2 > LIDAR_MIN_RANGE and t2 < best:
                        best = t2
            out_t[i] = best

        self._kernel = cast_kernel

    def cast(self, origins: np.ndarray, headings: np.ndarray, gimbal_yaw: np.ndarray):
        wp = self.wp
        n = origins.shape[0]
        az = self.az[None, :] + (headings + gimbal_yaw)[:, None]
        dirs = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=-1).astype(np.float32)
        o = wp.array(np.ascontiguousarray(origins.astype(np.float32)), dtype=wp.vec3, device=self.device)
        d = wp.array(dirs.reshape(-1, 3), dtype=wp.vec3, device=self.device)
        pxy = wp.array(self.plant_xy, dtype=wp.vec2, device=self.device)
        bz = wp.array(self.plant_base_z, dtype=wp.float32, device=self.device)
        out = wp.array(np.zeros(n * self.az.size, dtype=np.float32), dtype=wp.float32, device=self.device)
        wp.launch(self._kernel, dim=n * self.az.size,
                  inputs=[o, d, pxy, bz, int(self.plant_xy.shape[0]),
                          LIDAR_RANGE, out], device=self.device)
        return out.numpy().reshape(n, self.az.size)


class ObservationProvider:
    """Builds the five policy observation tensors for num_envs environments."""

    def __init__(self, num_envs: int, field_size: float = FIELD_SIZE, seed: int = 0,
                 lidar_backend: str = "numpy", device: str = "cpu",
                 feedback_backend: str = "direct", feedback_noise: bool = True,
                 depth_backend: str = "constant", depth_device: str = "cpu",
                 depth_noise: bool = True, camera_backend: str = "constant",
                 camera_noise: bool = True, camera_device: str = "cpu",
                 profile: bool = False):
        self.num_envs = int(num_envs)
        self.field_size = float(field_size)
        self.profile = bool(profile)
        self.nonfinite_frames = 0  # env-frames skipped because the state was non-finite
        self.timings = ({"lidar": 0.0, "depth": 0.0, "detections": 0.0,
                         "state": 0.0, "feedback": 0.0} if self.profile else None)
        self.feedback_backend = feedback_backend
        self.depth_backend = depth_backend
        self.depth_noise = bool(depth_noise)
        self._depth_scene = None
        self._dc = None
        self._depth_intr = None
        self._depth_cam_xy = None
        self._depth_device = "cpu"
        if depth_backend == "analytic":
            import sys as _sys
            _repo = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            import depth_caster as _dc
            from isaac_field.task import Config as _Config
            self._dc = _dc
            self._depth_scene = _dc.build_reference_scene()
            self._depth_intr, self._depth_cam_xy = _dc.camera_model()
            _cfg = _Config()
            self._depth_range_noise = float(_cfg.range_noise)
            self._depth_rng = np.random.default_rng(seed + 77)
            self._depth_device = "cuda:0" if str(depth_device).startswith("cuda") else "cpu"
        elif depth_backend != "constant":
            raise ValueError(f"unknown depth_backend {depth_backend!r}")
        self.camera_backend = camera_backend
        self.camera_noise = bool(camera_noise)
        self._dct = None
        self._det_scene = None
        self._det_warp = False
        self._det_device = "cpu"
        if camera_backend in ("analytic", "analytic_warp"):
            import sys as _sys
            _repo = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            import detections_caster as _dct
            self._dct = _dct
            self._det_scene = self._depth_scene if self._depth_scene is not None else _dct.dc.build_reference_scene()
            self._det_rng = np.random.default_rng(seed + 91)
            self._det_warp = camera_backend == "analytic_warp"
            self._det_device = "cuda:0" if str(camera_device).startswith("cuda") else "cpu"
        elif camera_backend != "constant":
            raise ValueError(f"unknown camera_backend {camera_backend!r}")
        self._vqf_estimators = None
        if feedback_backend == "vqf":
            import sys as _sys
            _repo = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
            if _repo not in _sys.path:
                _sys.path.insert(0, _repo)
            from isaac_field.feedback import SimFeedbackEstimator  # read-only reference estimator
            self._vqf_estimators = [SimFeedbackEstimator(seed=seed * 100000 + e,
                                                         noise_enabled=bool(feedback_noise))
                                    for e in range(self.num_envs)]
        elif feedback_backend != "direct":
            raise ValueError(f"unknown feedback_backend {feedback_backend!r}")
        self.rng = np.random.default_rng(seed)
        rng_scene = np.random.default_rng(seed + 1_000_003)
        self.plant_xy = rng_scene.uniform(0.0, field_size, (PLANTS, 2)).astype(np.float64)
        self.plant_base_z = terrain_height(self.plant_xy[:, 0], self.plant_xy[:, 1])
        az = np.linspace(-math.pi, math.pi, LIDAR_RAYS, endpoint=False)
        self.lidar_az = az
        self.lidar_local = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)
        self.lidar_backend = lidar_backend
        self._warp_caster = None
        if lidar_backend == "warp":
            self._warp_caster = WarpLidarCaster(az, self.plant_xy, self.plant_base_z,
                                                device=None if device != "cpu" else "cpu")
        elif lidar_backend != "numpy":
            raise ValueError(f"unknown lidar_backend {lidar_backend!r}")
        self.lidar_period = 0.1
        self.depth_period = 0.4
        self.camera_period = 0.2
        self.lidar_latency_steps = max(1, int(math.ceil(0.03 / POLICY_DT)))  # 1 policy step
        self.step = np.zeros(self.num_envs, dtype=np.int64)
        self.last_lidar = np.zeros(self.num_envs, dtype=np.int64)
        self.last_depth = np.zeros(self.num_envs, dtype=np.int64)
        self.last_camera = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_time = np.zeros(self.num_envs)
        self.prev_pose = np.zeros((self.num_envs, 3))
        self.lidar_latest = np.zeros((self.num_envs, LIDAR_RAYS))
        self.lidar_queue = deque([np.zeros((self.num_envs, LIDAR_RAYS))
                                  for _ in range(self.lidar_latency_steps + 1)],
                                 maxlen=self.lidar_latency_steps + 1)
        self.lidar_frame = np.zeros((self.num_envs, LIDAR_RAYS))
        self.depth_latency_steps = max(1, int(math.ceil(0.3 / POLICY_DT)))  # 6 policy steps (isaac 300 ms)
        self.depth_latest = np.zeros((self.num_envs, DEPTH_H, DEPTH_W))
        self.depth_queue = deque([np.zeros((self.num_envs, DEPTH_H, DEPTH_W))
                                  for _ in range(self.depth_latency_steps + 1)],
                                 maxlen=self.depth_latency_steps + 1)
        self.depth_frame = (np.zeros((self.num_envs, DEPTH_H, DEPTH_W)) if depth_backend == "analytic"
                            else np.full((self.num_envs, DEPTH_H, DEPTH_W), 0.5))
        self.camera_latency_steps = max(1, int(math.ceil(0.12 / POLICY_DT)))  # 3 policy steps (isaac 120 ms)
        self.det_latest = np.zeros((self.num_envs, MAX_DETECTIONS, 8), np.float32)
        self.det_queue = deque([np.zeros((self.num_envs, MAX_DETECTIONS, 8), np.float32)
                                for _ in range(self.camera_latency_steps + 1)],
                               maxlen=self.camera_latency_steps + 1)
        self.det_frame = self.det_queue[0].copy()
        self.new_lidar = np.zeros(self.num_envs)
        self.new_depth = np.zeros(self.num_envs)
        self.new_camera = np.zeros(self.num_envs)
        self.motion = np.zeros((self.num_envs, 3, 3))  # sensor x (dx, dy, dyaw)
        self.frame_pose = np.zeros((self.num_envs, 3, 5))  # per sensor: x, y, yaw, gimbal_yaw, gimbal_pitch
        self.goal_xy = np.zeros((self.num_envs, 2))
        self.prev_action = np.zeros((self.num_envs, 4))

    # ------------------------------------------------------------------
    def reset(self, env_ids=None, goal_xy=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, np.int64)
        self.step[ids] = 0
        self.last_lidar[ids] = 0
        self.last_depth[ids] = 0
        self.last_camera[ids] = 0
        self.episode_time[ids] = 0.0
        self.prev_action[ids] = 0.0
        self.new_lidar[ids] = 0.0
        self.new_depth[ids] = 0.0
        self.new_camera[ids] = 0.0
        self.motion[ids] = 0.0
        for q in self.depth_queue:
            q[ids] = 0.0
        self.depth_latest[ids] = 0.0
        self.depth_frame[ids] = 0.0
        for q in self.det_queue:
            q[ids] = 0.0
        self.det_latest[ids] = 0.0
        self.det_frame[ids] = 0.0
        if self._vqf_estimators is not None:
            # episode time restarts at 0: reset the per-env VQF estimator so its
            # monotonic-time check accepts the new episode
            for e in ids:
                self._vqf_estimators[int(e)].reset(now=0.0)
        if goal_xy is not None:
            self.goal_xy[ids] = goal_xy

    # ------------------------------------------------------------------
    def _lidar_cast(self, origins: np.ndarray, headings: np.ndarray, gimbal_yaw: np.ndarray):
        """Vectorised analytic raycast: plant stems (cylinders) + canopies (ellipsoids).

        Terrain is a low-amplitude sinusoid; horizontal rays from 0.55 m never hit it,
        so terrain is intentionally not intersected here (documented in the report).
        """
        n = origins.shape[0]
        out = np.full((n, LIDAR_RAYS), LIDAR_RANGE, dtype=np.float64)
        ang = headings + gimbal_yaw
        # NOTE: use the azimuth angle itself (earlier bug fed cos(az) into cos() again)
        dx_all = np.cos(self.lidar_az[None, :] + ang[:, None])
        dy_all = np.sin(self.lidar_az[None, :] + ang[:, None])
        for e in range(n):
            ox, oy, oz = origins[e]
            near = np.sum((self.plant_xy - [ox, oy]) ** 2, axis=1) < (LIDAR_RANGE + 0.5) ** 2
            ids = np.flatnonzero(near)
            if not ids.size:
                continue
            dx, dy = dx_all[e][:, None], dy_all[e][:, None]
            px = self.plant_xy[ids, 0][None, :]
            py = self.plant_xy[ids, 1][None, :]
            base_z = self.plant_base_z[ids][None, :]
            qx, qy = ox - px, oy - py
            a = dx * dx + dy * dy
            b = 2.0 * (dx * qx + dy * qy)
            c = qx * qx + qy * qy - STEM_RADIUS ** 2
            disc = b * b - 4.0 * a * c
            with np.errstate(invalid="ignore", divide="ignore"):
                t_stem = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
            z_stem = oz + t_stem * 0.0
            ok_stem = ((disc >= 0.0) & (t_stem > LIDAR_MIN_RANGE) & (t_stem < LIDAR_RANGE)
                       & (z_stem >= base_z) & (z_stem <= base_z + PLANT_HEIGHT))
            cz = base_z + CANOPY_Z
            oxn, oyn, ozn = (ox - px) / CANOPY_RX, (oy - py) / CANOPY_RY, (oz - cz) / CANOPY_RZ
            dxn, dyn = dx / CANOPY_RX, dy / CANOPY_RY
            a2 = dxn * dxn + dyn * dyn
            b2 = 2.0 * (oxn * dxn + oyn * dyn)
            c2 = oxn * oxn + oyn * oyn + ozn * ozn - 1.0
            disc2 = b2 * b2 - 4.0 * a2 * c2
            with np.errstate(invalid="ignore", divide="ignore"):
                t_can = (-b2 - np.sqrt(np.maximum(disc2, 0.0))) / (2.0 * a2)
            ok_can = (disc2 >= 0.0) & (t_can > LIDAR_MIN_RANGE) & (t_can < LIDAR_RANGE)
            t = np.where(ok_stem, np.where(ok_can, np.minimum(t_stem, t_can), t_stem),
                         np.where(ok_can, t_can, np.inf))
            best = np.min(t, axis=1)
            out[e] = np.minimum(out[e], best)
        return out

    def _lidar_noise(self, ranges: np.ndarray) -> np.ndarray:
        valid = np.isfinite(ranges) & (ranges >= LIDAR_MIN_RANGE) & (ranges < LIDAR_RANGE)
        noise = self.rng.uniform(-LIDAR_REL_ERROR, LIDAR_REL_ERROR, ranges.shape)
        values = np.clip(np.where(valid, ranges * (1.0 + noise), LIDAR_RANGE),
                         LIDAR_MIN_RANGE, LIDAR_RANGE)
        normalized = values / LIDAR_RANGE
        normalized[~valid] = 0.0
        return normalized

    # ------------------------------------------------------------------
    def observe(self, read: dict, actions: np.ndarray | None = None) -> dict[str, np.ndarray]:
        n = self.num_envs
        pos = read["pos"]
        quat = read["quat_wxyz"]
        yaw = np.asarray([yaw_from_quat_wxyz(q) for q in quat])
        gimbal_q = read["gimbal_q"]
        if actions is not None:
            self.prev_action = np.asarray(actions, dtype=np.float64)

        self.step += 1
        self.episode_time += POLICY_DT
        # non-finite state guard: never let NaN reach the analytic casters / VQF, keep the
        # previous sensor frames for the bad envs and report them as invalid
        finite = (np.isfinite(pos).reshape(n, -1).all(axis=1)
                  & np.isfinite(quat).reshape(n, -1).all(axis=1)
                  & np.isfinite(gimbal_q).reshape(n, -1).all(axis=1))
        if not finite.all():
            self.nonfinite_frames += int((~finite).sum())
        lidar_due = ((self.step % 2) == 0) & finite
        depth_due = ((self.step % 8) == 0) & finite
        camera_due = ((self.step % 4) == 0) & finite

        # REAL lidar; latency modelled as a one-policy-step delay queue
        _t = time.perf_counter() if self.timings is not None else None
        if lidar_due.any():
            ids = np.flatnonzero(lidar_due)
            origins = pos[ids] + np.array([0.0, 0.0, 0.40])
            if self._warp_caster is not None:
                ranges = self._warp_caster.cast(origins, yaw[ids], gimbal_q[ids, 0])
            else:
                ranges = self._lidar_cast(origins, yaw[ids], gimbal_q[ids, 0])
            frame = self.lidar_latest.copy()
            frame[ids] = self._lidar_noise(ranges)
            self.lidar_latest = frame
            self.frame_pose[ids, 0] = np.stack([pos[ids, 0], pos[ids, 1], yaw[ids],
                                               gimbal_q[ids, 0], gimbal_q[ids, 1]], axis=1)
            self.motion[ids, 0, 0] = pos[ids, 0] - self.frame_pose[ids, 0, 0]
            self.motion[ids, 0, 1] = pos[ids, 1] - self.frame_pose[ids, 0, 1]
            self.motion[ids, 0, 2] = yaw[ids] - self.frame_pose[ids, 0, 2]
            self.last_lidar[ids] = self.step[ids]
        self.lidar_queue.append(self.lidar_latest.copy())
        self.lidar_frame = self.lidar_queue[0]
        if _t is not None:
            self.timings["lidar"] += time.perf_counter() - _t

        _t = time.perf_counter() if self.timings is not None else None
        if depth_due.any():
            ids = np.flatnonzero(depth_due)
            self.last_depth[ids] = self.step[ids]
            if self._dc is not None:
                camera_z = pos[ids, 2] + CAMERA_HEIGHT
                poses = np.stack([pos[ids, 0], pos[ids, 1], camera_z, yaw[ids],
                                  gimbal_q[ids, 0], gimbal_q[ids, 1]], axis=1)
                dep, val, _device = self._dc.cast_warp(poses, scene=self._depth_scene,
                                                       intrinsics=self._depth_intr,
                                                       camera_xy=self._depth_cam_xy,
                                                       device=self._depth_device)
                # reference semantics: noise is added to valid metric pixels, then clipped
                if self.depth_noise:
                    noise = self._depth_rng.normal(0.0, self._depth_range_noise, dep.shape)
                    dep = np.clip(dep + noise, self._dc.DEPTH_MIN_RANGE, self._dc.DEPTH_RANGE)
                frame = np.where(val, dep, 0.0) / self._dc.DEPTH_RANGE
                self.depth_latest[ids] = frame
            self.frame_pose[ids, 1] = np.stack([pos[ids, 0], pos[ids, 1], yaw[ids],
                                               gimbal_q[ids, 0], gimbal_q[ids, 1]], axis=1)
        self.depth_queue.append(self.depth_latest.copy())
        self.depth_frame = self.depth_queue[0]
        if _t is not None:
            self.timings["depth"] += time.perf_counter() - _t

        _t = time.perf_counter() if self.timings is not None else None
        if camera_due.any():
            ids = np.flatnonzero(camera_due)
            self.last_camera[ids] = self.step[ids]
            if self._dct is not None:
                poses = np.stack([pos[ids, 0], pos[ids, 1], pos[ids, 2] + CAMERA_HEIGHT, yaw[ids],
                                  gimbal_q[ids, 0], gimbal_q[ids, 1]], axis=1)
                masks = (self._dct.cast_mask_warp_batch(poses, scene=self._det_scene,
                                                        device=self._det_device)[0]
                         if self._det_warp else
                         self._dct.cast_mask(poses, scene=self._det_scene))
                center = self._dct.cast_center_distance_batch(poses, scene=self._det_scene)
                rows, _dets = self._dct.observation_from_masks_batch(
                    masks, center, self._det_scene.leaf_dry,
                    rng=self._det_rng, noise=self.camera_noise)
                self.det_latest[ids] = rows
            self.frame_pose[ids, 2] = np.stack([pos[ids, 0], pos[ids, 1], yaw[ids],
                                               gimbal_q[ids, 0], gimbal_q[ids, 1]], axis=1)
        self.det_queue.append(self.det_latest.copy())
        self.det_frame = self.det_queue[0]
        if _t is not None:
            self.timings["detections"] += time.perf_counter() - _t

        self.prev_pose[:, 0] = pos[:, 0]
        self.prev_pose[:, 1] = pos[:, 1]
        self.prev_pose[:, 2] = yaw
        self.new_lidar = lidar_due.astype(np.float64)
        self.new_depth = depth_due.astype(np.float64)
        self.new_camera = camera_due.astype(np.float64)

        _t = time.perf_counter() if self.timings is not None else None
        state = self._state(pos, yaw, read).astype(np.float32)
        if not finite.all():
            state[~finite] = 0.0
        if _t is not None:
            self.timings["state"] += time.perf_counter() - _t
        _t = time.perf_counter() if self.timings is not None else None
        feedback = (self._feedback_vqf(quat, gimbal_q, read, finite=finite)
                    if self._vqf_estimators is not None
                    else self._feedback(quat, gimbal_q, read, finite=finite)).astype(np.float32)
        if not finite.all():
            feedback[~finite] = 0.0
        if _t is not None:
            self.timings["feedback"] += time.perf_counter() - _t

        return {
            "lidar": self.lidar_frame[:, None, :].astype(np.float32),
            "depth": self.depth_frame[:, None, :, :].astype(np.float32),
            "detections": (self.det_frame.copy() if self._dct is not None
                           else np.zeros((n, MAX_DETECTIONS, 8), dtype=np.float32)),
            "state": state,
            "feedback": feedback,
        }

    def consume_timings(self) -> dict | None:
        """Return accumulated per-sensor seconds and reset them (profile mode only)."""
        if self.timings is None:
            return None
        out = {k: round(float(v), 6) for k, v in self.timings.items()}
        out["total"] = round(float(sum(self.timings.values())), 6)
        for k in self.timings:
            self.timings[k] = 0.0
        return out

    def _feedback_vqf(self, quat, gimbal_q, read, finite=None) -> np.ndarray:
        """Real VQF path: reuse isaac_field's SimFeedbackEstimator (python vqf 2.1.2) per env.

        Signals passed are engine-truth sensor values; the estimator applies its documented
        sim noise and VQF attitude fusion, exactly like IsaacFieldEnv does. Cost is measured
        separately (~1.26 ms/env/update on this CPU).
        """
        n = self.num_envs
        out = np.zeros((n, FEEDBACK_DIM))
        gimbal_qd = read["gimbal_qd"]
        for e in range(n):
            if finite is not None and not finite[e]:
                continue  # keep the estimator state untouched for non-finite envs
            r_sensor = np.array([[math.cos(gimbal_q[e, 0]), -math.sin(gimbal_q[e, 0]), 0.0],
                                 [math.sin(gimbal_q[e, 0]), math.cos(gimbal_q[e, 0]), 0.0],
                                 [0.0, 0.0, 1.0]])
            # sensor frame = base yaw * gimbal yaw * Ry(-pitch); true gravity/mag/angular rate
            r_base = quat_wxyz_to_matrix(quat[e])
            cy, sy = math.cos(gimbal_q[e, 0]), math.sin(gimbal_q[e, 0])
            cp, sp = math.cos(gimbal_q[e, 1]), math.sin(gimbal_q[e, 1])
            r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            r_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
            r_sensor = r_base @ r_yaw @ r_pitch
            acc = r_sensor.T @ np.array([0.0, 0.0, G0])
            gyro = r_sensor.T @ read["ang_vel"][e]
            mag = r_sensor.T @ MAG_REFERENCE
            effort = read["effort"][e]
            iq = np.array([float(effort[0:7].mean()) / MOTOR_KT, float(effort[7:14].mean()) / MOTOR_KT])
            rpm = np.array([float(read["wheel_qd"][e, 0:7].mean()) * DRIVE_REDUCTION * 60.0 / (2.0 * math.pi),
                            float(read["wheel_qd"][e, 7:14].mean()) * DRIVE_REDUCTION * 60.0 / (2.0 * math.pi)])
            snap = self._vqf_estimators[e].update(
                float(self.episode_time[e]), acc, gyro, mag,
                gimbal_q[e].copy(), gimbal_qd[e].copy(), iq, rpm)
            out[e] = snap.feedback
        return out

    # ------------------------------------------------------------------
    def _state(self, pos, yaw, read) -> np.ndarray:
        """33-dim state; mirrors isaac_field/task.py FieldTask._observation exactly.

        Field order (ABSOLUTE_GOAL_INPUT_SCHEMA): mode, goal_xy_norm(2), est_xy_norm(2),
        sin/cos heading, last_action(4, raw), time/max_time, then per sensor
        [age/2, new_frame, dx/lidar_range, dy/lidar_range, dyaw/pi, dgimbal_yaw/pi,
         dgimbal_pitch/pi]. Final vector is clipped to [-1, 1] like the reference.
        """
        n = self.num_envs
        s = np.zeros((n, STATE_DIM))
        s[:, 0] = 1.0  # goto
        s[:, 1] = self.goal_xy[:, 0] / self.field_size * 2.0 - 1.0
        s[:, 2] = self.goal_xy[:, 1] / self.field_size * 2.0 - 1.0
        s[:, 3] = pos[:, 0] / self.field_size * 2.0 - 1.0
        s[:, 4] = pos[:, 1] / self.field_size * 2.0 - 1.0
        s[:, 5] = np.sin(yaw)
        s[:, 6] = np.cos(yaw)
        s[:, 7:11] = self.prev_action  # reference uses the raw last action (not the slewed command)
        s[:, 11] = np.clip(self.episode_time / 60.0, 0.0, 1.0)
        for k, (last, new_flag) in enumerate((
                (self.last_lidar, self.new_lidar),
                (self.last_depth, self.new_depth),
                (self.last_camera, self.new_camera))):
            base = 12 + 7 * k
            fx, fy, fyaw, fgy, fgp = (self.frame_pose[:, k, i] for i in range(5))
            s[:, base + 0] = np.clip((self.step - last) * POLICY_DT / 2.0, 0.0, 1.0)
            s[:, base + 1] = new_flag
            s[:, base + 2] = (pos[:, 0] - fx) / LIDAR_RANGE
            s[:, base + 3] = (pos[:, 1] - fy) / LIDAR_RANGE
            s[:, base + 4] = _wrap(yaw - fyaw) / math.pi
            s[:, base + 5] = _wrap(read["gimbal_q"][:, 0] - fgy) / math.pi
            s[:, base + 6] = (read["gimbal_q"][:, 1] - fgp) / math.pi
        return np.clip(np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    def _feedback(self, quat, gimbal_q, read, finite=None) -> np.ndarray:
        n = self.num_envs
        out = np.zeros((n, FEEDBACK_DIM))
        gimbal_qd = read["gimbal_qd"]
        for e in range(n):
            if finite is not None and not finite[e]:
                continue
            r_base = quat_wxyz_to_matrix(quat[e])
            cy, sy = math.cos(gimbal_q[e, 0]), math.sin(gimbal_q[e, 0])
            cp, sp = math.cos(gimbal_q[e, 1]), math.sin(gimbal_q[e, 1])
            r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            r_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
            r_sensor = r_base @ r_yaw @ r_pitch
            acc = r_sensor.T @ np.array([0.0, 0.0, G0]) + self.rng.normal(0.0, 0.025, 3)
            gyro = r_sensor.T @ read["ang_vel"][e] + self.rng.normal(0.0, 0.0015, 3)
            mag = r_sensor.T @ MAG_REFERENCE + self.rng.normal(0.0, 12.0, 3)
            out[e, 0:3] = acc / G0
            out[e, 3:6] = gyro / math.pi
            out[e, 6:9] = mag / 32768.0
            out[e, 9:13] = matrix_to_quat_wxyz(r_sensor)
            out[e, 13] = gimbal_q[e, 0] / math.pi
            out[e, 14] = gimbal_qd[e, 0] / math.pi
            out[e, 15] = gimbal_q[e, 1] / math.pi
            out[e, 16] = gimbal_qd[e, 1] / math.pi
            effort = read["effort"][e]
            left_iq = float(effort[0:7].mean() / MOTOR_KT)
            right_iq = float(effort[7:14].mean() / MOTOR_KT)
            left_rpm = float(read["wheel_qd"][e, 0:7].mean() * DRIVE_REDUCTION * 60.0 / (2.0 * math.pi))
            right_rpm = float(read["wheel_qd"][e, 7:14].mean() * DRIVE_REDUCTION * 60.0 / (2.0 * math.pi))
            out[e, 17] = left_iq / 100.0
            out[e, 18] = left_rpm / 10000.0
            out[e, 19] = right_iq / 100.0
            out[e, 20] = right_rpm / 10000.0
            out[e, 21:29] = 1.0  # valid flags (all sensor groups refreshed each step)
            out[e, 29:37] = 0.0  # ages
        return np.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)

    # ------------------------------------------------------------------
    def observation_shapes(self):
        return {"lidar": (1, LIDAR_RAYS), "depth": (1, DEPTH_H, DEPTH_W),
                "detections": (MAX_DETECTIONS, 8), "state": (STATE_DIM,),
                "feedback": (FEEDBACK_DIM,)}

    def contract(self, policy_dt: float = POLICY_DT) -> dict:
        return {
            "run_id": "m0_prototype",
            "phase": "prototype_throughput_only",
            "policy_dt_s": policy_dt,
            "sensor_schedule": {
                "lidar_period_s": self.lidar_period, "depth_period_s": self.depth_period,
                "camera_period_s": self.camera_period,
                "latency_s": {"lidar": 0.03, "depth": 0.3, "camera": 0.12},
                "latency_model": ("lidar: one policy-step delay queue; depth: 6-step (300 ms); "
                                  "detections: 3-step (120 ms) queues with analytic frames"
                                  if (self._dc is not None or self._dct is not None) else
                                  "lidar: one policy-step delay queue; depth/camera: placeholder frames only"),
            },
            "channels": {
                "lidar": {"shape": [1, LIDAR_RAYS], "mode": "real",
                          "backend": "warp_kernel" if self._warp_caster is not None else "numpy",
                          "source": "analytic raycast (stem cylinders + canopy ellipsoids; terrain not intersected)",
                          "verified_vs": None,
                          "verified_note": "not yet compared ray-by-ray with isaac_field Scene.cast / PhysX raycasts",
                          "usable_for_capability": False},
                "depth": {"shape": [1, DEPTH_H, DEPTH_W], "mode": "real" if self._dc is not None else "placeholder",
                          "backend": "analytic_warp" if self._dc is not None else "constant",
                          "source": ("analytic raycast (leaf patches + stems + terrain heightfield), camera-axial Z, "
                                     "normalised where(valid,v,0)/5.0"
                                     if self._dc is not None else "constant 0.5"),
                          "verified_vs": ("m1_depth_verification.json (independent numpy caster)"
                                          if self._dc is not None else None),
                          "verified_note": ("warp kernel == numpy caster: valid-mask mismatch 0, max 2.06e-4 m, "
                                            "100% pixels <=1e-3 m; flat-plane axial-Z check 7.8e-5 m; "
                                            "RTX domain gap registered, not pixel-compared"
                                            if self._dc is not None else "placeholder"),
                          "usable_for_capability": False},
                "detections": {"shape": [MAX_DETECTIONS, 8], "mode": "real" if self._dct is not None else "placeholder",
                               "backend": ("analytic_detector_warp" if self._det_warp else "analytic_detector")
                               if self._dct is not None else "constant",
                               "source": ("analytic 64x64 instance mask -> env.py box/confidence/class/range rules, "
                                          "packed per task.py"
                                          if self._dct is not None else "zeros"),
                               "verified_vs": ("detections_verification.json (FieldTask._observation packing)"
                                               if self._dct is not None else None),
                               "verified_note": ("analytic detections packing == reference FieldTask element-wise "
                                                 "(max abs error 0.0); mask/box rules re-implemented from env.py; "
                                                 "RTX domain gap registered, not pixel-compared"
                                                 if self._dct is not None else "placeholder"),
                               "usable_for_capability": False},
                "state": {"shape": [STATE_DIM], "mode": "real",
                          "source": "ABSOLUTE_GOAL_INPUT_SCHEMA layout (isaac_field/task.py), computed from simulator state",
                          "verified_vs": None,
                          "verified_note": "layout matches; numeric parity vs Isaac env not yet measured",
                          "usable_for_capability": False},
                "feedback": {"shape": [FEEDBACK_DIM], "mode": "real",
                             "backend": "vqf_reference" if self._vqf_estimators is not None else "direct_analytic",
                             "source": ("isaac_field/feedback.py SimFeedbackEstimator (vqf 2.1.2), fed with engine-truth "
                                        "accel/gyro/mag/gimbal/motor signals"
                                        if self._vqf_estimators is not None else
                                        "direct quaternion/IMU computation (no VQF)"),
                             "verified_vs": "m1_verification.feedback (see m1_vqf_verification.json)",
                             "verified_note": ("VQF path: attitude equals the reference estimator; direct path differs by 0.977"
                                               if self._vqf_estimators is not None else
                                               "direct attitude differs from reference VQF by 0.977 (see m1_verification)"),
                             "usable_for_capability": False},
            },
            "capability_metrics_allowed": False,
            "reason": ("all five channels are real and algorithmically verified; capability metrics stay disabled until "
                       "the RTX domain gap (depth/detections) is accepted"
                       if (self._dc is not None or self._dct is not None) else
                       "depth/detections placeholders and no cross-validation -> throughput measurements only"),
            "capability_blockers": (["RTX domain gap not pixel-compared (depth, detections)"]
                                    if (self._dc is not None or self._dct is not None) else
                                    ["depth placeholder (constant 0.5)", "detections placeholder (16x8 zeros)"]),
        }

    def contract_sha256(self, policy_dt: float = POLICY_DT) -> str:
        blob = json.dumps(self.contract(policy_dt), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
