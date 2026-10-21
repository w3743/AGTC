"""Gym adapter that drives a paused Gazebo world through real pose/sensors.

Plant detection and damage remain field_sim geometry proxies, but are evaluated
from Gazebo's actual body and gimbal pose. LiDAR and depth observations always
come from Gazebo messages; this adapter never falls back to analytic ray casts.
"""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from field_sim.env import (DEPTH_GEOMETRY_ID, DEPTH_GEOMETRY_SOURCE,
                           DEPTH_NATIVE_HEIGHT, DEPTH_NATIVE_WIDTH,
                           FieldEnv, wrap)


ROOT = Path(__file__).resolve().parents[1]


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _pose7(value):
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        return None
    norm = float(np.linalg.norm(pose[3:]))
    if norm < 1e-9:
        return None
    pose[3:] /= norm
    return pose


def _rpy(pose):
    """Return standard Gazebo roll, pitch, yaw from [x,y,z,qx,qy,qz,qw]."""
    x, y, z, w = pose[3:]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1, 1)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _body_rotation(roll, source_pitch, yaw):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(source_pitch), math.sin(source_pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]], dtype=np.float64)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


class GazeboFieldEnv(FieldEnv):
    """FieldEnv rewards/observations backed by a per-episode Gazebo process."""

    def __init__(self, cfg=None, mode="search", *,
                 output_dir: Path | str = ROOT / "results" / "gazebo_field" / "episodes",
                 gui: bool = False, driver_factory=None):
        super().__init__(cfg, mode)
        self.output_dir = Path(output_dir)
        self.gui = bool(gui)
        self.driver_factory = driver_factory
        self.driver = None
        self._episode_index = 0
        self._preparing = False
        self._snapshot = None
        self._sim_origin = 0.0
        self._last_sensor_stamp = {"lidar": None, "depth": None}
        self._actual_frames = {"lidar": 0, "depth": 0}
        self._actual_timestamps = {"lidar": None, "depth": None}
        self._gimbal_target = np.zeros(2, dtype=np.float64)
        self._motion_path = []
        self._invalid_physical_reason = ""
        self._last_reset_wall_s = 0.0
        self._last_step_wall_s = 0.0
        self._last_sim_step_s = 0.0
        self._episode_dir = None

    def _new_episode_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_index += 1
        name = f"episode_{self._episode_index:06d}_{uuid.uuid4().hex[:8]}"
        path = self.output_dir / name
        path.mkdir()
        return path

    def _write_source_payloads(self, episode_dir):
        scene_path = episode_dir / "scene.json"
        state_path = episode_dir / "state.json"
        scene = self.scene_payload()
        state = self.state_payload()
        if scene.get("scene_id") != state.get("scene_id"):
            raise RuntimeError("scene/state payload IDs differ before Gazebo export")
        scene_path.write_text(json.dumps(scene, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        state_path.write_text(json.dumps(state, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        return scene_path, state_path

    def _make_driver(self, world_path, episode_dir):
        factory = self.driver_factory
        if factory is None:
            from simulation.gazebo_driver import GazeboDriver
            factory = GazeboDriver
        return factory(world_path, episode_dir, dt=self.cfg.dt, step_size=0.005,
                       gui=self.gui,
                       gimbal_rates=(self.cfg.gimbal_yaw_rate, self.cfg.gimbal_pitch_rate))

    def reset(self, *, seed=None, options=None):
        started = time.perf_counter()
        if self.driver is not None:
            self.driver.close()
            self.driver = None
        # Let FieldEnv derive the source spawn height from the new terrain,
        # rather than carrying the last episode's Gazebo pose across resets.
        self._snapshot = None
        self._body_height = None
        self._last_step_wall_s = 0.0
        self._last_sim_step_s = 0.0
        self._motion_path = []
        self._preparing = True
        try:
            # Reuse FieldEnv's scene generation, validation and initial camera
            # initialization. Its synthetic range frames are discarded below.
            _, _ = super().reset(seed=seed, options=options)
        finally:
            self._preparing = False

        self._episode_dir = self._new_episode_dir()
        from simulation.field_world import export_scene
        scene_path, state_path = self._write_source_payloads(self._episode_dir)
        if self.scene_id != self.state_payload()["scene_id"]:
            raise RuntimeError("scene_id changed during Gazebo payload export")
        try:
            exported = export_scene(scene_path, state_path, self._episode_dir / "export")
            self.driver = self._make_driver(Path(exported["world"]), self._episode_dir)
            snapshot = self.driver.warmup()
        except Exception:
            if self.driver is not None:
                self.driver.close()
                self.driver = None
            raise
        self._snapshot = snapshot
        self._sim_origin = float(_get(snapshot, "sim_time", 0.0))
        self._install_snapshot(snapshot, dt_hint=self.cfg.dt)

        # Clear all synthetic initial frames, pending deliveries and evaluator
        # state, then bootstrap only from actual Gazebo LiDAR/depth frames.
        self.frames = {}
        self._pending = {k: deque() for k in ("lidar", "depth", "camera")}
        self._serial = {k: 0 for k in ("lidar", "depth", "camera")}
        self._dropped = {k: 0 for k in self._serial}
        self._new = {k: False for k in self._serial}
        self._last_sensor_stamp = {"lidar": None, "depth": None}
        self._actual_frames = {"lidar": 0, "depth": 0}
        self._actual_timestamps = {"lidar": None, "depth": None}
        self._next = {"lidar": self.time + self.cfg.lidar_period,
                      "depth": self.time + self.cfg.depth_period,
                      "camera": self.time + self.cfg.camera_period}
        self._pose_noise = np.zeros(2)
        self._next_uwb = self.time + .1
        self.reported.clear()
        self.seen.clear()
        self.damaged.clear()
        self.false_reports.clear()
        self._counts = (0, 0, 0)
        self._hold = 0.0
        self.collisions = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.done, self.reason = False, ""
        self.trail.clear()
        self.trail.append(self.robot[:2].tolist())
        self._gimbal_target[:] = self.gimbal
        self._invalid_physical_reason = self._physical_status()
        self._ingest_actual_sensor("lidar", _get(snapshot, "lidar"), bootstrap=True)
        self._ingest_actual_sensor("depth", _get(snapshot, "depth"), bootstrap=True)
        self._deliver_camera(self._capture("camera"))
        self._last_reset_wall_s = time.perf_counter() - started
        info = self._info()
        info["reset_wall_s"] = self._last_reset_wall_s
        info["episode_dir"] = str(self._episode_dir)
        return self._observation(), info

    def _install_snapshot(self, snapshot, dt_hint=None):
        pose = _pose7(_get(snapshot, "pose"))
        if pose is None:
            raise RuntimeError("Gazebo snapshot has no finite vehicle pose")
        prior = getattr(self, "_snapshot", None)
        prior_pose = _pose7(_get(prior, "pose")) if prior is not None else None
        sim_time = float(_get(snapshot, "sim_time", self._sim_origin))
        roll, sdf_pitch, yaw = _rpy(pose)
        source_pitch = -sdf_pitch
        half = self.cfg.size / 2
        dt = float(dt_hint or self.cfg.dt)
        v = w = 0.0
        if prior_pose is not None:
            previous_time = float(_get(prior, "sim_time", sim_time - dt))
            elapsed = max(sim_time - previous_time, 1e-9)
            delta = pose[:2] - prior_pose[:2]
            previous_yaw = _rpy(prior_pose)[2]
            mid_yaw = previous_yaw + 0.5 * wrap(yaw - previous_yaw)
            v = float(np.dot(delta, [math.cos(mid_yaw), math.sin(mid_yaw)]) / elapsed)
            w = wrap(yaw - previous_yaw) / elapsed
            dt = elapsed
        self.robot[:] = [pose[0] + half, pose[1] + half, yaw, v, w]
        gimbal = _get(snapshot, "gimbal")
        if gimbal is not None:
            gy, gp = np.asarray(gimbal, dtype=np.float64).reshape(-1)[:2]
            if np.isfinite(gy) and np.isfinite(gp):
                self.gimbal[:] = [wrap(float(gy)), float(gp)]
        self._body_height = float(pose[2])
        self._body_roll = float(roll)
        self._body_pitch = float(source_pitch)
        self._snapshot = snapshot
        self._snapshot_dt = dt

    def _body_pose(self):
        if getattr(self, "_body_height", None) is not None:
            return self._body_height, self._body_pitch, self._body_roll
        return super()._body_pose()

    def _frame_context(self, sensor):
        raw_pose = _pose7(_get(sensor, "pose"))
        if raw_pose is None:
            raw_pose = _pose7(_get(self._snapshot, "pose"))
        if raw_pose is None:
            raise RuntimeError("Gazebo sensor frame has no matching vehicle pose")
        raw_gimbal = _get(sensor, "gimbal")
        if raw_gimbal is None:
            raw_gimbal = _get(self._snapshot, "gimbal", self.gimbal)
        gimbal = np.asarray(raw_gimbal, dtype=np.float64).reshape(-1)
        if gimbal.size < 2 or not np.all(np.isfinite(gimbal[:2])):
            raise RuntimeError("Gazebo sensor frame has no matching gimbal angles")
        roll, sdf_pitch, yaw = _rpy(raw_pose)
        source_pitch = -sdf_pitch
        gy, gp = float(gimbal[0]), float(gimbal[1])
        half = self.cfg.size / 2
        field_pose = np.array([raw_pose[0] + half, raw_pose[1] + half, yaw, gy, gp], dtype=np.float32)
        rotation = _body_rotation(roll, source_pitch, yaw)
        origin = raw_pose[:3] + rotation @ np.array([0.0, 0.0, self.cfg.camera_height])
        # Gazebo model poses are centered on the source world; browser/map
        # coordinates are centered on the field. Keep sensor origins in the
        # same field frame as field_pose and the analytic geometry.
        origin[:2] += half
        f_local = np.array([math.cos(gp) * math.cos(gy), math.cos(gp) * math.sin(gy), math.sin(gp)])
        right_local = np.array([math.sin(gy), -math.cos(gy), 0.0])
        f, right, up = rotation @ f_local, rotation @ right_local, None
        up = np.cross(right, f)
        return field_pose, np.asarray(origin, dtype=np.float32), np.asarray([roll, source_pitch], dtype=np.float32), f, right, up

    def _ingest_actual_sensor(self, kind, sensor, *, bootstrap=False):
        if sensor is None:
            raise RuntimeError(f"Gazebo did not provide an initial {kind} frame")
        stamp = _get(sensor, "sim_time", _get(sensor, "stamp"))
        if stamp is None or not math.isfinite(float(stamp)):
            raise RuntimeError(f"Gazebo {kind} frame has no finite simulation timestamp")
        stamp = float(stamp)
        previous = self._last_sensor_stamp[kind]
        if previous is not None and stamp <= previous + 1e-9:
            return False
        self._last_sensor_stamp[kind] = stamp
        pose, origin, attitude, forward, right, _ = self._frame_context(sensor)
        capture = stamp - self._sim_origin
        self._serial[kind] += 1
        seq = self._serial[kind]
        if kind == "lidar":
            ranges = np.asarray(_get(sensor, "ranges"), dtype=np.float32).reshape(-1)
            if ranges.size != self.cfg.lidar_rays:
                raise ValueError(f"Gazebo LiDAR has {ranges.size} rays; expected {self.cfg.lidar_rays}")
            valid = np.isfinite(ranges) & (ranges >= self.cfg.lidar_min_range) & (ranges < self.cfg.lidar_range - 1e-4)
            output_ranges = np.where(valid, np.clip(ranges, self.cfg.lidar_min_range, self.cfg.lidar_range), self.cfg.lidar_range).astype(np.float32)
            az = np.linspace(-math.pi, math.pi, self.cfg.lidar_rays, endpoint=False, dtype=np.float32)
            directions = (np.cos(az)[:, None] * forward[None, :]
                          - np.sin(az)[:, None] * right[None, :]).astype(np.float32)
            # Keep the transport/state mask boolean. FieldEnv._observation stacks
            # it with float sensor values and explicitly casts the policy input
            # back to float32, so this does not change the model contract.
            frame = {"ranges": output_ranges, "valid": valid,
                     "hit_kind": np.zeros(self.cfg.lidar_rays, dtype=np.uint8),
                     "directions": directions, "scan_pitch": float(math.asin(float(np.clip(forward[2], -1, 1)))),
                     "origin": origin, "body_attitude": attitude,
                     "model": self.cfg.lidar_model, "min_range": self.cfg.lidar_min_range,
                     "max_range": self.cfg.lidar_range}
        else:
            values = np.asarray(_get(sensor, "values"), dtype=np.float32)
            expected = (self.cfg.depth_height, self.cfg.depth_width)
            if values.shape != expected:
                raise ValueError(f"Gazebo depth has shape {values.shape}; expected {expected}")
            valid = np.isfinite(values) & (values >= self.cfg.depth_min_range) & (values <= self.cfg.depth_range)
            values = np.where(valid, np.clip(values, self.cfg.depth_min_range, self.cfg.depth_range), self.cfg.depth_range).astype(np.float32)
            frame = {"values": values, "valid": valid, "unit": "m",
                     "convention": "camera_z", "projection": "pinhole",
                     "resolution": [self.cfg.depth_width, self.cfg.depth_height],
                     "shape": [self.cfg.depth_height, self.cfg.depth_width],
                     "native_resolution": [DEPTH_NATIVE_WIDTH, DEPTH_NATIVE_HEIGHT],
                     "intrinsics": self.depth_intrinsics.copy(),
                     "calibration_id": DEPTH_GEOMETRY_ID,
                     "calibration_source": DEPTH_GEOMETRY_SOURCE,
                     "min_range": self.cfg.depth_min_range, "max_range": self.cfg.depth_range,
                     "origin": origin, "body_attitude": attitude}
        frame.update({"seq": seq, "source_seq": _get(sensor, "sequence"),
                      "capture": capture, "pose": pose, "source_stamp": stamp,
                      "stamp_source": _get(sensor, "stamp_source", "gazebo_sim_time")})
        if bootstrap:
            self.frames[kind] = frame
            self._new[kind] = True
            self._actual_frames[kind] = 1
            self._actual_timestamps[kind] = stamp
            return True
        latency = self.cfg.lidar_latency if kind == "lidar" else self.cfg.depth_latency
        self._pending[kind].append((capture + latency, frame))
        self._actual_frames[kind] += 1
        self._actual_timestamps[kind] = stamp
        return True

    def _deliver_camera(self, frame):
        self.frames["camera"] = frame
        self._new["camera"] = True
        for detection in frame["detections"]:
            self.seen.add(detection["plant"])
            if detection["class_id"] == 1 and detection["confidence"] >= .68 and detection["range"] is not None:
                (self.reported if self.scene.targets[detection["plant"]] else self.false_reports).add(detection["plant"])

    def _sense(self, initial=False):
        if self._preparing:
            return super()._sense(initial=initial)
        for kind in ("lidar", "depth"):
            self._new[kind] = False
        self._new["camera"] = False
        current_snapshot = self.driver.get_snapshot()
        self._snapshot = current_snapshot
        for kind in ("lidar", "depth"):
            self._ingest_actual_sensor(kind, _get(current_snapshot, kind))

        if self.time + 1e-9 >= self._next["camera"]:
            deadline = self._next["camera"]
            periods = max(1, math.floor((self.time - deadline + 1e-9) / self.cfg.camera_period) + 1)
            self._dropped["camera"] += periods - 1
            frame = self._capture("camera")
            frame["capture"] = self.time
            self._pending["camera"].append((self.time + self.cfg.camera_latency, frame))
            self._next["camera"] = deadline + periods * self.cfg.camera_period

        for kind in ("lidar", "depth", "camera"):
            pending = self._pending[kind]
            while pending and pending[0][0] <= self.time + 1e-9:
                _, frame = pending.popleft()
                self.frames[kind] = frame
                self._new[kind] = True
                if kind == "camera":
                    self._deliver_camera(frame)
        if self.time + 1e-9 >= self._next_uwb:
            self._pose_noise = self.np_random.normal(0, self.cfg.position_noise, 2)
            self._next_uwb = self.time + .1

    def _trajectory_robot_states(self, snapshot, fallback_before):
        rows = list(_get(snapshot, "trajectory", []) or [])
        states = []
        for row in rows:
            pose = _pose7(_get(row, "pose"))
            if pose is None:
                continue
            roll, sdf_pitch, yaw = _rpy(pose)
            states.append(np.array([pose[0] + self.cfg.size / 2,
                                    pose[1] + self.cfg.size / 2, yaw, 0.0, 0.0], dtype=np.float64))
        if not states or np.linalg.norm(states[0][:2] - fallback_before[:2]) > 1e-5:
            states.insert(0, fallback_before.copy())
        if not states or np.linalg.norm(states[-1][:2] - self.robot[:2]) > 1e-5 or abs(wrap(states[-1][2] - self.robot[2])) > 1e-5:
            states.append(self.robot.copy())
        return states

    def _advance_motion(self, a):
        c = self.cfg
        snapshot_before = self._snapshot
        before = self.robot.copy()
        old_distance = float(np.linalg.norm(self.goal - before[:2]))
        target_v = float(self.robot[3] + np.clip(a[0] * c.max_speed - self.robot[3],
                                                   -c.acceleration * c.dt, c.acceleration * c.dt))
        target_w = float(self.robot[4] + np.clip(a[1] * c.max_turn_rate - self.robot[4],
                                                   -c.angular_acceleration * c.dt, c.angular_acceleration * c.dt))
        self.last_gimbal_rates = np.asarray(a[2:], dtype=np.float64) * [c.gimbal_yaw_rate, c.gimbal_pitch_rate]
        self._gimbal_target[0] = wrap(self._gimbal_target[0] + self.last_gimbal_rates[0] * c.dt)
        self._gimbal_target[1] = np.clip(self._gimbal_target[1] + self.last_gimbal_rates[1] * c.dt,
                                         c.pitch_min, c.pitch_max)
        start_sim = float(_get(snapshot_before, "sim_time", self._sim_origin + self.time))
        started = time.perf_counter()
        snapshot = self.driver.step(target_v, target_w,
                                    float(self._gimbal_target[0]), float(self._gimbal_target[1]),
                                    duration=c.dt)
        self._last_step_wall_s = time.perf_counter() - started
        end_sim = float(_get(snapshot, "sim_time", start_sim + c.dt))
        self._last_sim_step_s = max(0.0, end_sim - start_sim)
        self._snapshot = snapshot_before
        self._install_snapshot(snapshot, dt_hint=self._last_sim_step_s or c.dt)
        states = self._trajectory_robot_states(snapshot, before)
        self._motion_path = states

        rock_collision = False
        for left, right in zip(states, states[1:]):
            hit, _ = self._rock_sweep(left, right)
            rock_collision = rock_collision or hit
        if rock_collision:
            self.collisions += 1
        radius = math.hypot(c.robot_length, c.robot_width) / 2
        boundary = bool(np.any(self.robot[:2] < radius) or np.any(self.robot[:2] > c.size - radius))
        # FieldEnv's shared finisher advances by cfg.dt; align its clock to the
        # actual stepped simulation time without kinematic position integration.
        elapsed = end_sim - self._sim_origin
        self.time = max(0.0, elapsed - c.dt)
        self._invalid_physical_reason = self._physical_status()
        return before, old_distance, rock_collision, boundary

    def _physical_status(self):
        if not np.all(np.isfinite(self.robot)) or not np.isfinite(getattr(self, "_body_height", math.nan)):
            return "invalid_pose"
        if abs(self._body_roll) >= math.pi / 2 or abs(self._body_pitch) >= math.pi / 2:
            return "tipped"
        try:
            ground = float(self.scene.terrain.surface_height(self.robot[:2])) if self.scene.terrain is not None else 0.0
        except Exception:
            ground = 0.0
        if self._body_height < ground - .25:
            return "below_terrain"
        return ""

    def _damage_sweep(self, before, after):
        states = getattr(self, "_motion_path", None)
        if states and len(states) > 1:
            for left, right in zip(states, states[1:]):
                super()._damage_sweep(left, right)
        else:
            super()._damage_sweep(before, after)

    def _finish_step(self, a, before, old_distance, rock_collision, boundary):
        result = super()._finish_step(a, before, old_distance, rock_collision, boundary)
        obs, reward, terminated, truncated, info = result
        reason = self._invalid_physical_reason
        if boundary:
            reason = "boundary"
        if reason:
            self.done, self.reason = True, reason
            terminated, truncated = False, True
        info.update({"backend": "gazebo", "reason": self.reason,
                     "is_success": self.reason in ("goal_reached", "all_targets_found_by_evaluator"),
                     "physical_truncation": reason or None})
        return obs, reward, terminated, truncated, info

    def _info(self):
        info = super()._info()
        info.update({"backend": "gazebo",
                     "sim_time_s": float(_get(self._snapshot, "sim_time", self._sim_origin + self.time)),
                     "sensor_actual_timestamps_s": self._actual_timestamps.copy(),
                     "sensor_actual_frames": self._actual_frames.copy(),
                     "sensor_capture_timestamps_s": {k: (None if k not in self.frames else self.frames[k].get("source_stamp"))
                                                      for k in ("lidar", "depth")},
                     "step_wall_s": self._last_step_wall_s,
                     "step_sim_s": self._last_sim_step_s,
                     "reset_wall_s": self._last_reset_wall_s,
                     "episode_dir": None if self._episode_dir is None else str(self._episode_dir)})
        return info

    def step(self, action):
        started = time.perf_counter()
        result = super().step(action)
        self._last_step_wall_s = max(self._last_step_wall_s, time.perf_counter() - started)
        obs, reward, terminated, truncated, info = result
        info.update({"step_wall_s": self._last_step_wall_s,
                     "step_sim_s": self._last_sim_step_s,
                     "sensor_actual_timestamps_s": self._actual_timestamps.copy(),
                     "sensor_actual_frames": self._actual_frames.copy()})
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None


__all__ = ["GazeboFieldEnv"]
