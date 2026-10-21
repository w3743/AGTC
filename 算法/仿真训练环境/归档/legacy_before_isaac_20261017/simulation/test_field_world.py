"""Focused contract tests for the offline Gazebo field export."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from simulation import field_world as exporter


def _payloads(tmp: Path):
    scene = {
        "schema": "field-geometry-debug-v1",
        "scene_id": 42,
        "coordinate_frame": "field_xyz_z_up_m",
        "size": 100.0,
        "config": {
            "size": 100.0, "plants": 2, "leaves_per_plant": 2,
            "robot_length": 0.8, "robot_width": 0.3, "camera_height": 0.4,
            "pitch_min": -0.4363323129985824, "pitch_max": 1.0471975511965976,
            "lidar_rays": 8, "lidar_period": 0.1, "lidar_range": 8.0,
            "lidar_min_range": 0.12, "depth_width": 4, "depth_height": 2,
            "depth_period": 0.4, "depth_min_range": 0.3, "depth_range": 5.0,
        },
        "terrain": {
            "enabled": True, "base_height": 0.2,
            "wave_vectors4x2": [[0.1, 0.0], [0.0, 0.2], [0.05, 0.05], [-0.04, 0.03]],
            "wave_amplitudes4": [0.2, 0.1, 0.05, 0.05],
            "wave_phases4": [0.0, 0.3, 0.5, -0.4],
            "seed": 91,
            "rocks": [
                {"x": 25.0, "y": 75.0, "z": 0.1, "base_z": 0.1,
                 "radius": 0.2, "height": 0.3, "blocking": True},
                {"x": 60.0, "y": 10.0, "z": 0.1, "base_z": 0.1,
                 "radius": 0.08, "height": 0.12, "blocking": False},
            ],
        },
        "plants": [
            {"x": 10.0, "y": 20.0, "z": 0.0, "base_z": 0.0, "radius": 0.09,
             "height": 1.0, "stem_radius": 0.01, "root_damage_radius": 0.09, "dry": False},
            {"x": 85.0, "y": 30.0, "z": 0.0, "base_z": 0.0, "radius": 0.08,
             "height": 0.9, "stem_radius": 0.01, "root_damage_radius": 0.08, "dry": True},
        ],
        "leaf_geometry": {
            "shape": "tapered_blade_v1",
            "centers": [[10.0, 20.0, 0.4], [10.0, 20.1, 0.7],
                        [85.0, 30.0, 0.4], [85.0, 30.1, 0.7]],
            "normals": [[0.0, 0.0, 1.0]] * 4,
            "u": [[1.0, 0.0, 0.0]] * 4,
            "v": [[0.0, 1.0, 0.0]] * 4,
            "axes": [[0.4, 0.06]] * 4,
            "plant": [0, 0, 1, 1],
            "dry": [False, False, True, True],
        },
    }
    state = {
        "scene_id": 42,
        "robot": {"x": 12.0, "y": 23.0, "z": 0.3, "yaw": 0.4, "roll": 0.1, "pitch": 0.2},
        "gimbal": {"yaw": 0.3, "pitch": -0.1},
        "depth": {"resolution": [4, 2],
                  "intrinsics": [[2.5, 0.0, 1.5], [0.0, 2.5, 0.5], [0.0, 0.0, 1.0]]},
    }
    scene_path = tmp / "scene.json"
    state_path = tmp / "state.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return scene, state, scene_path, state_path


class FieldWorldExportTests(unittest.TestCase):
    def test_shared_geometry_centering_heightmap_and_no_plant_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene, _, scene_path, state_path = _payloads(root)
            out = root / "export"
            result = exporter.export_scene(scene_path, state_path, out)

            tree = ET.parse(result["world"])
            world = tree.find("world")
            self.assertEqual(world.attrib["name"], "field_training")
            vegetation = world.find("model[@name='field_vegetation']")
            veg_link = vegetation.find("link")
            self.assertEqual([v.attrib["name"] for v in veg_link.findall("visual")],
                             ["healthy_leaves", "dry_leaves", "stems"])
            self.assertEqual(veg_link.findall("collision"), [])
            self.assertEqual(len(world.findall("model[@name='field_robot']")), 1)
            self.assertEqual(len(world.findall("model")), 5)
            self.assertIsNotNone(world.find("plugin[@name='gz::sim::systems::Imu']"))
            self.assertIsNotNone(world.find("plugin[@name='gz::sim::systems::Magnetometer']"))

            assets = out / "assets"
            for asset in ("terrain.png", "soil_diffuse.png", "flat_normal.png",
                          "healthy_leaves.obj", "dry_leaves.obj", "stems.obj", "rock_cap.obj"):
                self.assertTrue((assets / asset).is_file(), asset)
            image = np.asarray(Image.open(assets / "terrain.png"))
            self.assertEqual(image.shape, (513, 513))
            self.assertEqual(image[0].tolist(), exporter.terrain_samples(scene)[1][-1].tolist())
            self.assertEqual(int(image.min()), 0)
            self.assertEqual(int(image.max()), 255)
            _, pixels, low, height_span, error = exporter.terrain_samples(scene)
            self.assertLessEqual(error, height_span / 510.0 + 1e-12)
            expected_corner = 0.2 + sum(a * math.sin(p) for a, p in zip(
                scene["terrain"]["wave_amplitudes4"], scene["terrain"]["wave_phases4"]))
            self.assertAlmostEqual(exporter.terrain_samples(scene)[0][0, 0], expected_corner)
            self.assertGreaterEqual(low, 0.2 - sum(scene["terrain"]["wave_amplitudes4"]) - 1e-12)
            terrain_visual_hm = world.find("model[@name='field_terrain']/link/visual/geometry/heightmap")
            terrain_collision_hm = world.find("model[@name='field_terrain']/link/collision/geometry/heightmap")
            terrain_link = world.find("model[@name='field_terrain']/link")
            terrain_visual = world.find("model[@name='field_terrain']/link/visual")
            terrain_collision = world.find("model[@name='field_terrain']/link/collision")
            self.assertEqual(terrain_link.findtext("pose"), "0 0 0 0 0 0")
            self.assertEqual(terrain_visual.findtext("pose"), "0 0 0 0 0 0")
            self.assertAlmostEqual(float(terrain_collision.findtext("pose").split()[2]), low, places=6)
            self.assertAlmostEqual(float(terrain_visual_hm.findtext("pos").split()[2]), low, places=6)
            self.assertEqual(terrain_collision_hm.findtext("pos"), "0 0 0")
            self.assertEqual(terrain_visual_hm.findtext("texture/diffuse"), "assets/soil_diffuse.png")
            self.assertEqual(terrain_visual_hm.findtext("texture/normal"), "assets/flat_normal.png")
            self.assertIsNone(terrain_collision_hm.find("texture"))
            self.assertEqual(world.findtext("spherical_coordinates/latitude_deg"), "30")
            self.assertEqual(world.findtext("spherical_coordinates/longitude_deg"), "120")
            tracked_vehicle = world.find("model[@name='field_robot']/plugin[@name='gz::sim::systems::TrackedVehicle']")
            self.assertEqual(float(tracked_vehicle.findtext("track_height")), 0.11)
            self.assertIsNone(tracked_vehicle.find("tracks_height"))
            controllers = world.findall("model[@name='field_robot']/plugin[@name='gz::sim::systems::JointPositionController']")
            self.assertEqual(len(controllers), 2)
            for controller in controllers:
                self.assertEqual(controller.findtext("use_velocity_commands"), "true")
                self.assertEqual(controller.findtext("cmd_max"), "0.5")
                self.assertIsNone(controller.find("p_gain"))

            # OBJ vertices are the source tapered-blade boundary translated into centered XY.
            first = next(line for line in (assets / "healthy_leaves.obj").read_text().splitlines()
                         if line.startswith("v "))
            xyz = [float(x) for x in first.split()[1:]]
            expected = exporter._leaf_ring(scene["leaf_geometry"]["centers"][0],
                                           scene["leaf_geometry"]["u"][0],
                                           scene["leaf_geometry"]["v"][0], 0.4, 0.06)[0]
            np.testing.assert_allclose(xyz, [expected[0] - 50.0, expected[1] - 50.0, expected[2]], atol=1e-8)
            stem_vertex = next(line for line in (assets / "stems.obj").read_text().splitlines()
                               if line.startswith("v "))
            stem_xyz = [float(x) for x in stem_vertex.split()[1:]]
            np.testing.assert_allclose(stem_xyz, [-39.99, -30.0, 0.0], atol=1e-8)

            # Rock visuals and colliders share their dimensions and scene recentering.
            rock = world.find("model[@name='rock_000']")
            pose = [float(v) for v in rock.findtext("pose").split()]
            self.assertEqual(pose[:3], [-25.0, 25.0, 0.1])
            visual_mesh = rock.find("link/visual/geometry/mesh")
            collision_mesh = rock.find("link/collision/geometry/mesh")
            self.assertEqual(visual_mesh.findtext("scale"), collision_mesh.findtext("scale"))
            self.assertEqual(visual_mesh.findtext("scale"), "0.2 0.2 0.3")
            rock_obj = (assets / "rock_cap.obj").read_text(encoding="ascii").splitlines()
            normals = [line for line in rock_obj if line.startswith("vn ")]
            faces = [line for line in rock_obj if line.startswith("f ")]
            self.assertEqual(len(normals), 241)
            self.assertEqual(len(faces), 384)
            for face in faces:
                references = face.split()[1:]
                self.assertTrue(all("//" in ref for ref in references))
                self.assertTrue(all(1 <= int(ref.split("//")[0]) <= 217 for ref in references))
                self.assertTrue(all(1 <= int(ref.split("//")[1]) <= len(normals) for ref in references))

            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["geometry_counts"]["merged_plant_visual_meshes"], 3)
            self.assertEqual(manifest["geometry_counts"]["plant_collision_meshes"], 0)
            self.assertEqual(manifest["geometry_counts"]["unit_rock_mesh_faces"], 384)
            self.assertEqual(manifest["plant_damage_adapter"]["plant_adapter"][1]["leaf_ids"], [2, 3])
            self.assertFalse(manifest["uwb"]["enabled"])
            self.assertTrue(manifest["terrain"]["runtime_trailing_datum_crop_observed"])
            self.assertAlmostEqual(manifest["terrain"]["heightmap_collision_pose_z_m"], low)
            self.assertAlmostEqual(manifest["terrain"]["heightmap_visual_pos_z_m"], low)
            self.assertAlmostEqual(manifest["terrain"]["heightmap_size_z_m"], height_span)
            self.assertEqual(manifest["vehicle"]["gimbal"]["control_mode"], "bounded_velocity_commands")
            self.assertIn("not geographic ground truth", manifest["magnetometer_world_reference"]["status"])

    def test_vehicle_sensor_axes_and_optional_uwb(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene, _, scene_path, state_path = _payloads(root)
            out = root / "uwb_export"
            result = exporter.export_scene(scene_path, state_path, out, mass=30.0,
                                           com_height=0.15, with_uwb=True,
                                           uwb_anchor_clearance_m=2.0)
            world = ET.parse(result["world"]).find("world")
            robot = world.find("model[@name='x500_field_robot']")
            model_pose = [float(v) for v in robot.findtext("pose").split()]
            self.assertAlmostEqual(model_pose[4], -0.2)
            self.assertAlmostEqual(model_pose[5], 0.4)

            pitch_joint = robot.find("joint[@name='gimbal_pitch_joint']")
            self.assertEqual(pitch_joint.findtext("axis/xyz"), "0 -1 0")
            self.assertEqual(pitch_joint.find("pose").attrib["relative_to"], "gimbal_yaw_link")
            self.assertEqual(robot.find("joint[@name='gimbal_yaw_joint']/pose").attrib["relative_to"], "base_link")
            self.assertEqual(robot.find("joint[@name='left_track_fixed_joint']/pose").attrib["relative_to"], "base_link")
            pitch_link = robot.find("link[@name='gimbal_pitch_link']")
            self.assertEqual(pitch_link.findtext("sensor[@name='field_lidar']/pose"), "0 0 0 0 0 0")
            self.assertEqual(pitch_link.findtext("sensor[@name='field_depth_camera']/pose"), "0 0 0 0 0 0")
            camera = pitch_link.find("sensor[@name='field_depth_camera']/camera")
            self.assertEqual(camera.findtext("image/width"), "4")
            self.assertEqual(camera.findtext("image/height"), "2")
            self.assertEqual(camera.findtext("camera_info_topic"), exporter.TOPICS["depth_camera_info"])
            self.assertEqual(camera.findtext("lens/intrinsics/fx"), "2.5")
            self.assertAlmostEqual(float(pitch_link.findtext("pose").split()[2]), 0.4)
            lidar_horizontal = pitch_link.find("sensor[@name='field_lidar']/lidar/scan/horizontal")
            self.assertEqual(lidar_horizontal.findtext("samples"), "8")
            self.assertAlmostEqual(float(lidar_horizontal.findtext("max_angle")), math.pi - math.pi / 4)

            plugins = robot.findall("plugin")
            self.assertEqual(sum(p.attrib["name"] == "gz::sim::systems::TrackController" for p in plugins), 2)
            self.assertEqual(sum(p.attrib["name"] == "gz::sim::systems::JointPositionController" for p in plugins), 2)
            mass_map = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["vehicle"]["mass_distribution_kg"]
            self.assertAlmostEqual(sum(mass_map.values()), 30.0)
            self.assertEqual(mass_map["uwb_tag_1"], 0.01)

            anchor_models = [m for m in world.findall("model")
                             if m.attrib["name"].startswith("r1_rover_anchor_")]
            self.assertEqual(len(anchor_models), 9)
            anchors_meta = exporter.uwb_anchor_layout(scene, 2.0)
            for anchor_meta in anchors_meta:
                anchor = world.find(f"model[@name='{anchor_meta['model']}']")
                self.assertEqual(anchor.find("link").attrib["name"], anchor_meta["link"])
                pose = [float(value) for value in anchor.findtext("pose").split()]
                np.testing.assert_allclose(pose[:3], anchor_meta["world_xyz_m"], atol=1e-8)
                expected_source = [10.0, 50.0, 90.0][(anchor_meta["id"] - 1) % 3]
                expected_source_y = [10.0, 50.0, 90.0][(anchor_meta["id"] - 1) // 3]
                self.assertAlmostEqual(anchor_meta["source_xy_m"][0], expected_source)
                self.assertAlmostEqual(anchor_meta["source_xy_m"][1], expected_source_y)
                self.assertAlmostEqual(anchor_meta["terrain_z_m"], exporter.terrain_height_at(
                    scene, *anchor_meta["source_xy_m"]))
            self.assertIsNotNone(robot.find("link[@name='uwb_tag_1']"))
            uwb_plugin = world.find("plugin[@name='custom::UWBGazeboSystem']")
            self.assertIsNotNone(uwb_plugin)
            self.assertEqual(uwb_plugin.findtext("enable_nlos_dropout"), "false")
            self.assertEqual(uwb_plugin.findtext("min_noise_stddev_cm"), "5")
            self.assertEqual(uwb_plugin.findtext("max_noise_stddev_cm"), "5")
            self.assertEqual(uwb_plugin.findtext("min_dropout_probability"), "0.02")
            self.assertEqual(uwb_plugin.findtext("max_dropout_probability"), "0.02")
            self.assertGreater(float(uwb_plugin.findtext("los_max_thickness_m")), 140.0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["uwb"]["enabled"])
            self.assertEqual(manifest["vehicle"]["model_name"], "x500_field_robot")
            self.assertFalse(manifest["uwb"]["runtime_verified"])
            self.assertFalse(manifest["uwb"]["anchor_layout_calibrated"])
            self.assertEqual(len(manifest["uwb"]["anchors"]), 9)
            self.assertEqual(manifest["uwb"]["anchor_clearance_m"], 2.0)
            self.assertFalse(manifest["uwb"]["nlos_enabled"])
            topics_doc = json.loads(Path(result["topics"]).read_text(encoding="utf-8"))
            self.assertEqual(len(topics_doc["uwb_range_pairs"]), 9)
            self.assertEqual(topics_doc["uwb_range_pairs"][0], {
                "anchor_id": 1, "tag_id": 1, "pair_key": "a1t1",
                "range_topic": "/uwb_gz_simulator/distances/a1t1",
                "ground_truth_topic": "/uwb_gz_simulator/distances_ground_truth/a1t1",
                "message_type": "gz.msgs.Double", "unit": "cm"})

    def test_scene_state_mismatch_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, state, scene_path, state_path = _payloads(root)
            state["scene_id"] = 43
            state_path.write_text(json.dumps(state), encoding="utf-8")
            out = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "scene/state scene_id mismatch"):
                exporter.export_scene(scene_path, state_path, out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
