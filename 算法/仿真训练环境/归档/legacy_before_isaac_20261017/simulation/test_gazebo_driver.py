"""Unit tests for the Gazebo driver helpers; these never start Gazebo."""
from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from simulation.gazebo_driver import (
    _decode_depth,
    _find_robot_pose,
    _joint_angles,
    _match_context,
    _stamp,
    _write_runtime_sdf,
)


class _Time:
    def __init__(self, sec=0, nsec=0):
        self.sec, self.nsec = sec, nsec


class _Header:
    def __init__(self, stamp=None):
        self.stamp = stamp or _Time()
        self.present = stamp is not None

    def HasField(self, name):
        return name == "stamp" and self.present


class _Stamped:
    def __init__(self, stamp=None):
        self.header = _Header(stamp)

    def HasField(self, name):
        return name == "header"


class GazeboDriverHelperTests(unittest.TestCase):
    def test_unset_stamp_is_absent_and_present_stamp_is_read(self):
        self.assertIsNone(_stamp(_Stamped()))
        self.assertAlmostEqual(_stamp(_Stamped(_Time(7, 250_000_000))), 7.25)

    def test_robot_pose_selects_named_model(self):
        def pose(name, x):
            return type("Pose", (), {
                "name": name,
                "position": type("V", (), {"x": x, "y": 2.0, "z": 3.0})(),
                "orientation": type("Q", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})(),
            })()
        message = type("PoseV", (), {"pose": [pose("other", -1), pose("field_robot", 4)]})()
        self.assertEqual(_find_robot_pose(message, "field_robot"), [4.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])

    def test_joint_angle_extraction_uses_axis1(self):
        def axis(position):
            return type("Axis", (), {
                "position": position,
                "HasField": lambda self, name: name == "position",
            })()
        joints = [
            type("Joint", (), {"name": "gimbal_yaw_joint", "axis1": axis(-0.3),
                                "HasField": lambda self, name: name == "axis1"})(),
            type("Joint", (), {"name": "field_robot::gimbal_pitch_joint", "axis1": axis(0.2),
                                "HasField": lambda self, name: name == "axis1"})(),
        ]
        model = type("Model", (), {"joint": joints})()
        self.assertEqual(_joint_angles(model), [-0.3, 0.2])

    def test_depth_decode_honors_row_stride(self):
        payload = struct.pack("<ff", 1.0, 2.0) + b"\0" * 4 + struct.pack("<ff", 3.0, 4.0) + b"\0" * 4
        message = type("Image", (), {"width": 2, "height": 2, "step": 12, "data": payload})()
        image = _decode_depth(message)
        self.assertEqual(image.shape, (2, 2))
        self.assertEqual(image.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_sensor_context_matches_nearest_pose_and_gimbal(self):
        context = _match_context(1.04,
                                [{"sim_time": 1.0, "pose": [1]}, {"sim_time": 1.1, "pose": [2]}],
                                [{"sim_time": 1.02, "gimbal": [0.1, 0.2]}])
        self.assertEqual(context["pose"], [1])
        self.assertEqual(context["gimbal"], [0.1, 0.2])
        self.assertAlmostEqual(context["pose_match_error_s"], 0.04)

    def test_runtime_sdf_is_a_temp_copy_with_step_and_pose_publisher(self):
        xml = '''<sdf version="1.9"><world name="field_training"><physics>
          <max_step_size>0.004</max_step_size><real_time_factor>1</real_time_factor>
        </physics><model name="field_robot"><plugin name="gz::sim::systems::JointPositionController">
          <joint_name>gimbal_yaw_joint</joint_name><cmd_max>0.5</cmd_max>
        </plugin><plugin name="gz::sim::systems::JointPositionController">
          <joint_name>gimbal_pitch_joint</joint_name><cmd_max>0.5</cmd_max>
        </plugin></model></world></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "world.sdf"
            source.write_text(xml, encoding="utf-8")
            runtime = _write_runtime_sdf(source, "field_robot", 0.005, (1.8, 1.2))
            try:
                world = ET.parse(runtime).getroot().find(".//world")
                physics = world.find("physics")
                self.assertEqual(physics.findtext("max_step_size"), "0.005")
                self.assertEqual(physics.findtext("real_time_factor"), "0")
                robot = world.find("model[@name='field_robot']")
                controllers = {p.findtext("joint_name"): p.findtext("cmd_max")
                               for p in robot.findall("plugin")
                               if p.findtext("joint_name")}
                self.assertEqual(controllers, {"gimbal_yaw_joint": "1.8",
                                               "gimbal_pitch_joint": "1.2"})
                publisher = robot.find("plugin[@name='gz::sim::systems::PosePublisher']")
                self.assertIsNotNone(publisher)
                self.assertEqual(publisher.findtext("publish_model_pose"), "true")
                self.assertEqual(publisher.findtext("publish_link_pose"), "true")
                self.assertEqual(source.read_text(encoding="utf-8"), xml)
            finally:
                runtime.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
