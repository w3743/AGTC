"""Fast adapter tests with a deterministic fake driver (never launches Gazebo)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from field_sim.env import Config
from simulation.gazebo_env import GazeboFieldEnv


def _pose(x, y, z, yaw):
    return [float(x), float(y), float(z), 0.0, 0.0,
            math.sin(yaw / 2), math.cos(yaw / 2)]


class FakeDriver:
    """Physical-motion contract stub with timestamped raw sensor arrays."""

    def __init__(self, world_path, log_dir, *, dt, step_size, gui, gimbal_rates):
        self.dt = dt
        manifest = json.loads((Path(world_path).parent / "manifest.json").read_text(encoding="utf-8"))
        start = manifest["state_initialization"]["robot"]
        self.pose = _pose(float(start["x"]) - 4, float(start["y"]) - 4,
                          float(start["z"]), float(start["yaw"]))
        self.gimbal = [0.0, 0.0]
        self.time = 0.0
        self.last_command = None
        self.next_lidar = None
        self.next_depth = None
        self.lidar = None
        self.depth = None
        self.closed = False

    def _sensor(self, kind, stamp):
        common = {"sim_time": float(stamp), "pose": list(self.pose),
                  "gimbal": list(self.gimbal), "sequence": 1}
        if kind == "lidar":
            common["ranges"] = np.full(400, 2.0, dtype=np.float32)
            common["ranges"][0] = np.nan
        else:
            common["values"] = np.full((160, 88), 2.0, dtype=np.float32)
            common["values"][0, 0] = np.nan
        return common

    def _snapshot(self, trajectory=None):
        return {"sim_time": float(self.time), "pose": list(self.pose),
                "gimbal": list(self.gimbal), "trajectory": list(trajectory or []),
                "lidar": self.lidar, "depth": self.depth}

    def warmup(self):
        self.time = 1.2
        self.lidar = self._sensor("lidar", self.time)
        self.depth = self._sensor("depth", self.time)
        self.next_lidar = self.time + .1
        self.next_depth = self.time + .4
        return self._snapshot()

    def step(self, v, w, gy, gp, duration=None):
        duration = self.dt if duration is None else duration
        old_time, old_pose = self.time, list(self.pose)
        yaw = 2 * math.atan2(self.pose[5], self.pose[6])
        self.pose[0] += float(v) * math.cos(yaw) * duration
        self.pose[1] += float(v) * math.sin(yaw) * duration
        yaw += float(w) * duration
        self.pose[3:] = _pose(0, 0, 0, yaw)[3:]
        self.gimbal = [float(gy), float(gp)]
        self.time += duration
        if self.time + 1e-9 >= self.next_lidar:
            self.lidar = self._sensor("lidar", self.time)
            self.next_lidar += .1
        if self.time + 1e-9 >= self.next_depth:
            self.depth = self._sensor("depth", self.time)
            self.next_depth += .4
        trajectory = [{"sim_time": old_time, "pose": old_pose},
                      {"sim_time": self.time, "pose": list(self.pose)}]
        self.last_command = (float(v), float(w), float(gy), float(gp), float(duration))
        return self._snapshot(trajectory)

    def get_snapshot(self):
        return self._snapshot()

    def close(self):
        self.closed = True


class GazeboFieldEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gazebo_field_env_test_")
        self.cfg = Config(size=8, plants=1, leaves_per_plant=1, terrain_enabled=False,
                          validate_scene=False, rock_density=0, dt=.05, max_time=2)
        self.env = GazeboFieldEnv(self.cfg, "goto", output_dir=self.temp.name,
                                  driver_factory=FakeDriver)

    def tearDown(self):
        self.env.close()
        self.temp.cleanup()

    def test_real_sensor_payload_and_physical_action_contract(self):
        obs, info = self.env.reset(seed=7, options={"goal": [6, 6]})
        self.assertEqual(info["backend"], "gazebo")
        self.assertEqual(obs["lidar"].shape, (2, 400))
        self.assertEqual(obs["depth"].shape, (2, 160, 88))
        self.assertEqual(obs["lidar"].dtype, np.float32)
        self.assertEqual(obs["depth"].dtype, np.float32)
        self.assertEqual(obs["lidar"][1, 0], 0.0)
        self.assertEqual(obs["depth"][1, 0, 0], 0.0)
        self.assertTrue(np.allclose(obs["depth"][0, 1:, :], 2.0 / self.cfg.depth_range))
        self.assertEqual(obs["depth"][0, 0, 0], 1.0)  # Invalid pixels use max-range sentinel.
        self.assertTrue(np.all(obs["depth"][1, 1:, :] == 1))
        self.assertEqual(info["sensor_actual_frames"], {"lidar": 1, "depth": 1})
        self.assertEqual(self.env.frames["lidar"]["valid"].dtype, np.bool_)
        self.assertEqual(self.env.frames["depth"]["valid"].dtype, np.bool_)

        state = self.env.state_payload()
        self.assertEqual(state["lidar"]["ranges"][0], self.cfg.lidar_range)
        self.assertTrue(np.allclose(state["lidar"]["ranges"][1:], 2.0))
        self.assertIs(state["lidar"]["valid"][0], False)
        self.assertIs(state["lidar"]["valid"][1], True)
        self.assertEqual(set(state["lidar"]["hit_kind"]), {0})
        self.assertIs(state["depth"]["valid"][0][0], False)
        self.assertIs(state["depth"]["valid"][0][1], True)
        for sensor_name in ("lidar", "depth"):
            sensor_origin = np.asarray(state[sensor_name]["origin"], dtype=np.float64)
            np.testing.assert_allclose(sensor_origin[:2], [state["robot"]["x"], state["robot"]["y"]], atol=1e-5)
            self.assertAlmostEqual(sensor_origin[2], self.env.driver.pose[2] + self.cfg.camera_height, places=5)
        self.assertEqual(state["depth"]["unit"], "m")
        self.assertEqual(state["depth"]["convention"], "camera_z")
        self.assertEqual(state["depth"]["resolution"], [88, 160])
        self.assertEqual(state["depth"]["calibration_id"], "bcb9e5c82b93")
        self.assertTrue({"directions", "valid", "hit_kind", "scan_pitch", "origin",
                         "body_attitude", "gimbal_pitch", "age", "seq"}.issubset(state["lidar"]))
        self.assertTrue({"values", "valid", "projection", "shape", "native_resolution",
                         "intrinsics", "calibration_source", "min_range", "max_range",
                         "capture_time", "pose", "origin", "body_attitude", "age", "seq"}
                        .issubset(state["depth"]))
        self.assertEqual(state["metrics"]["backend"], "gazebo")

        x_before = float(self.env.robot[0])
        _, _, _, _, step_info = self.env.step(np.array([1, 0, 1, 0], dtype=np.float32))
        command = self.env.driver.last_command
        self.assertAlmostEqual(command[0], self.cfg.acceleration * self.cfg.dt, places=6)
        self.assertAlmostEqual(command[2], self.cfg.gimbal_yaw_rate * self.cfg.dt, places=6)
        self.assertGreater(float(self.env.robot[0]), x_before)
        self.assertEqual(step_info["backend"], "gazebo")
        self.assertEqual(step_info["sensor_actual_frames"]["lidar"], 1)
        self.env.step(np.zeros(4, dtype=np.float32))
        self.assertEqual(self.env._actual_frames["lidar"], 2)
        self.assertGreater(self.env._actual_timestamps["lidar"], 1.2)

    def test_boundary_is_truncated_without_pose_clamp(self):
        self.env.reset(seed=3, options={"goal": [6, 6]})
        radius = math.hypot(self.cfg.robot_length, self.cfg.robot_width) / 2
        source_x = self.cfg.size - radius - .001
        self.env.driver.pose[0] = source_x - self.cfg.size / 2
        self.env._snapshot = self.env.driver.get_snapshot()
        self.env._install_snapshot(self.env._snapshot)
        self.env.robot[3:5] = 0

        _, _, terminated, truncated, info = self.env.step(np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "boundary")
        self.assertGreater(self.env.robot[0], self.cfg.size - radius)


if __name__ == "__main__":
    unittest.main()
