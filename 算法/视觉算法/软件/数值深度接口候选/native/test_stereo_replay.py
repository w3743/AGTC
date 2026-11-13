"""Compare the C++ SGBM replay to the current Python pipeline on recorded frames."""
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
import unittest
from email.parser import BytesParser
from email.policy import default

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
from fast_depth import FastSGBMDepth
import depth_wire
from probe_depth import DepthProbeCollector


class StereoReplayTest(unittest.TestCase):
    def compare_recorded_pair(self, capture_name, size):
        exe = os.environ.get("AGTC_STEREO_FIXTURE")
        if not exe:
            self.skipTest("Set AGTC_STEREO_FIXTURE to the built replay fixture")
        capture = HERE.parent.parent / "stereo" / "captures" / capture_name
        inputs = [cv2.imdecode(np.fromfile(capture / (side + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
                  for side in ("left", "right")]
        self.assertTrue(all(image is not None for image in inputs))
        if size == (640, 360):
            inputs = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in inputs]
        calibration = HERE.parent / "app" / "calibration.npz"
        engine = FastSGBMDepth(calibration)
        rectified = engine.rectify(*inputs)
        expected_disp, _ = engine.disparities(*rectified)
        expected_depth, expected_valid = engine.metric_values(expected_disp, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            yaml = root / "calibration.yml"
            subprocess.run([sys.executable, "-B", str(HERE / "export_calibration.py"), str(calibration), str(yaml)], check=True)
            for name, image in zip(("left", "right"), inputs):
                image.tofile(root / (name + ".raw"))
            prefix = root / "output"
            env = os.environ.copy()
            if env.get("AGTC_OPENCV_BIN"):
                env["PATH"] = env["AGTC_OPENCV_BIN"] + os.pathsep + env["PATH"]
            proc = subprocess.run([exe, str(yaml), str(root / "left.raw"), str(root / "right.raw"),
                                   str(size[0]), str(size[1]), str(prefix)], env=env, capture_output=True, text=True, check=True)
            lines = proc.stdout.splitlines()
            self.assertEqual(lines[0], hashlib.sha256(calibration.read_bytes()).hexdigest())
            self.assertEqual(len(lines), 2)
            disparity = np.fromfile(str(prefix) + ".disparity", np.float32).reshape(240, 131)
            depth = np.fromfile(str(prefix) + ".depth", np.float32).reshape(240, 131)
            valid = np.fromfile(str(prefix) + ".valid", np.uint8).reshape(240, 131).astype(bool)
            left_rect = np.fromfile(str(prefix) + ".left_rectified", np.uint8).reshape(240, 131)
            right_rect = np.fromfile(str(prefix) + ".right_rectified", np.uint8).reshape(240, 131)
            left_resized = np.fromfile(str(prefix) + ".left_resized", np.uint8).reshape(135, 240)
            np.testing.assert_array_equal(left_resized, cv2.resize(inputs[0], (240, 135), interpolation=cv2.INTER_AREA), err_msg="left resize")
            native_map_x = np.fromfile(str(prefix) + ".left_map_x", np.float32).reshape(135, 240)
            native_map_y = np.fromfile(str(prefix) + ".left_map_y", np.float32).reshape(135, 240)
            published = pathlib.Path(str(prefix) + ".published").read_bytes()
            metadata = depth_wire.parse_depth_metadata(pathlib.Path(str(prefix) + ".published.json").read_text(encoding="ascii"))
            expected_payload, _, _ = depth_wire.encode_depth_payload(np.ascontiguousarray(expected_depth))
            self.assertEqual(published, expected_payload)
            header_bytes, http_body = pathlib.Path(str(prefix) + ".http").read_bytes().split(b"\r\n\r\n", 1)
            status_line, headers = header_bytes.split(b"\r\n", 1)
            self.assertEqual(status_line, b"HTTP/1.1 200 OK")
            parsed_headers = BytesParser(policy=default).parsebytes(headers + b"\r\n\r\n")
            collector = DepthProbeCollector()
            try:
                now = time.perf_counter_ns()
                row = collector.ingest_response(status=200, headers=parsed_headers.items(), body=http_body,
                                                host_send_monotonic_ns=now, host_receive_monotonic_ns=now)
                self.assertEqual(row["validation_errors"], [])
                self.assertEqual(http_body, published)
            finally:
                collector.close_archive()
            self.assertEqual(metadata["seq"], 1)
            self.assertEqual(metadata["pipeline_epoch"], 1)
            self.assertEqual((metadata["left_pts_ns"], metadata["right_pts_ns"]), (1_000_000_000, 1_010_000_000))
            self.assertEqual(metadata["calibration_id"], engine.identifier)
            self.assertIsNone(metadata["extrinsics_id"])
            self.assertIsNone(metadata["source_mono_ns"])
            expected_geometry = depth_wire.fast_geometry_descriptor(engine, calibration)
            np.testing.assert_allclose(metadata["K"], expected_geometry["K"], rtol=0, atol=1e-12)
            metrics = {"left_remap_mismatches": int(np.count_nonzero(left_rect != rectified[0])),
                       "right_remap_mismatches": int(np.count_nonzero(right_rect != rectified[1])),
                       "disparity_mismatches": int(np.count_nonzero(disparity != expected_disp)),
                       "valid_mask_mismatches": int(np.count_nonzero(valid != expected_valid))}
            print("OpenCV Python", cv2.__version__, "C++", lines[1], metrics)
            evidence_dir = os.environ.get("AGTC_STEREO_EVIDENCE_DIR")
            if evidence_dir:
                directory = pathlib.Path(evidence_dir)
                directory.mkdir(parents=True, exist_ok=True)
                sources = [calibration, capture / "left.png", capture / "right.png", HERE / "stereo_sgbm.cpp",
                           HERE / "depth_filter.cpp", HERE.parent / "app" / "fast_depth.py", pathlib.Path(exe)]
                report = {"platform": platform.platform(), "python": platform.python_version(),
                          "opencv_python": cv2.__version__, "opencv_native": lines[1],
                          "capture": capture_name, "input_size": list(size), "pixels": int(disparity.size),
                          "metrics": metrics, "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
                report_name = f"stereo-{cv2.__version__}-{capture_name}-{size[0]}x{size[1]}.json"
                with (directory / report_name).open("x", encoding="utf-8") as output:
                    json.dump(report, output, indent=2)
                    output.write("\n")
            np.testing.assert_array_equal(native_map_x, engine.maps[0][0], err_msg="left map x")
            np.testing.assert_array_equal(native_map_y, engine.maps[0][1], err_msg="left map y")
            np.testing.assert_array_equal(left_rect, rectified[0], err_msg="left rectification")
            np.testing.assert_array_equal(right_rect, rectified[1], err_msg="right rectification")
            np.testing.assert_array_equal(disparity, expected_disp)
            np.testing.assert_array_equal(valid, expected_valid)
            np.testing.assert_allclose(depth, expected_depth, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_first_pair_live_size(self):
        self.compare_recorded_pair("20240825T221623918473Z-946d97fc", (640, 360))

    def test_first_pair_full_sensor_size(self):
        self.compare_recorded_pair("20240825T221623918473Z-946d97fc", (3840, 2160))

    def test_second_pair_live_size(self):
        self.compare_recorded_pair("20240825T221620622446Z-de013c47", (640, 360))

    def test_rejects_invalid_calibration_and_image_size(self):
        exe = os.environ.get("AGTC_STEREO_FIXTURE")
        if not exe:
            self.skipTest("Set AGTC_STEREO_FIXTURE to the built replay fixture")
        env = os.environ.copy()
        if env.get("AGTC_OPENCV_BIN"):
            env["PATH"] = env["AGTC_OPENCV_BIN"] + os.pathsep + env["PATH"]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            yaml = root / "calibration.yml"
            calibration = HERE.parent / "app" / "calibration.npz"
            subprocess.run([sys.executable, "-B", str(HERE / "export_calibration.py"), str(calibration), str(yaml)], check=True)
            bad = root / "invalid.yml"
            bad.write_text(yaml.read_text(encoding="utf-8").replace("geometry_pass_metric_accuracy_unverified", "unverified"), encoding="utf-8")
            args = [exe, str(bad), str(root / "missing-left"), str(root / "missing-right"), "640", "360", str(root / "out")]
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid calibration", result.stderr)
            args[1], args[4], args[5] = str(yaml), "639", "360"
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid input dimensions", result.stderr)
            self.assertFalse((root / "out.depth").exists())


if __name__ == "__main__":
    unittest.main()
