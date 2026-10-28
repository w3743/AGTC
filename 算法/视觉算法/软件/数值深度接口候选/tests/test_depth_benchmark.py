import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import benchmark_depth as benchmark


class FakeEngine:
    def __init__(self, calibration_path, *, scale, threads, input_size):
        self.scale = scale
        self.threads = threads
        self.input_size = input_size
        self.preprocess_backend = "cpu"
        self.width = 8
        self.height = 6
        self.raw_width = 12
        self.raw_height = 9
        self.num_disparities = 16
        self.matcher = SimpleNamespace(getBlockSize=lambda: 5)
        self.small_masks = (np.ones((6, 8), bool), np.ones((6, 8), bool))
        self.identifier = "fixture-calibration"
        self.baseline_mm = 95.0
        self.depth_factor = 1.0
        self.disparity_offset = 0.0

    def compute(self, left, right):
        self.last_input_shape = left.shape
        depth = np.full((6, 8), .75, dtype=np.float32)
        depth[:, 4:] = np.nan
        details = {"stage_ms": {"resize": 1.0, "remap_rotate_crop": 2.0,
                                 "match": 3.0, "depth_filter": 4.0, "colorize": 1.0}}
        return np.zeros((6, 8, 3), np.uint8), depth, details

    def disparities(self, left, right):
        return np.full(left.shape, 8.0, dtype=np.float32), None

    def metric_values(self, disparity, _reverse, _left):
        depth = np.full(disparity.shape, self.depth_factor / disparity[0, 0], dtype=np.float32)
        valid = np.ones(disparity.shape, dtype=bool)
        return depth, valid


class DepthBenchmarkTests(unittest.TestCase):
    def test_cli_value_parsers_and_source_resolutions(self):
        self.assertEqual(benchmark.parse_scales(".375, .5,1"), (.375, .5, 1.0))
        self.assertEqual(benchmark.parse_threads("2,4"), (2, 4))
        self.assertEqual(benchmark.parse_modes("cpu,opencl"), ("cpu", "opencl"))
        self.assertEqual(benchmark.parse_shifts("8,24"), (8, 24))
        self.assertEqual(benchmark.parse_input_sizes("640x360,source", 3840, 2160),
                         ((640, 360), (3840, 2160)))
        with self.assertRaises(ValueError):
            benchmark.parse_input_sizes("640x480", 3840, 2160)

    def test_same_pair_resize_and_analytic_shift(self):
        left = np.arange(64 * 36, dtype=np.uint8).reshape(36, 64)
        right = np.flip(left, axis=1).copy()
        small_left, small_right = benchmark.prepare_pair(left, right, (32, 18))
        self.assertEqual(small_left.shape, (18, 32))
        self.assertEqual(small_right.shape, (18, 32))
        synth_left, synth_right = benchmark.synthetic_shift_pair((16, 40), 7, seed=19)
        np.testing.assert_array_equal(synth_right[:, :-7], synth_left[:, 7:])
        self.assertTrue(np.all(synth_right[:, -7:] == 0))

    def test_synthetic_depth_check_uses_analytic_shift_not_physical_truth(self):
        engine = FakeEngine(None, scale=.375, threads=2, input_size=(64, 32))
        engine.width = 64
        engine.height = 32
        result = benchmark.check_synthetic_shift(engine, 8)
        self.assertEqual(result["expected_z_m"], .125)
        self.assertEqual(result["median_measured_z_m"], .125)
        self.assertEqual(result["median_absolute_error_mm"], 0.0)
        self.assertEqual(result["valid_core_fraction"], 1.0)
        self.assertIn("not physical accuracy", result["truth_scope"])

    def test_coverage_separates_halves_and_mask_boundary(self):
        depth = np.ones((10, 12), dtype=np.float32)
        depth[:, 6:] = np.nan
        support = np.ones((10, 12), dtype=bool)
        support[:, 0] = False
        result = benchmark.coverage_summary(depth, support, boundary_band=1)
        self.assertEqual(result["valid_fraction_all_pixels"], .5)
        self.assertEqual(result["valid_fraction_left_half"], 1.0)
        self.assertEqual(result["valid_fraction_right_half"], 0.0)
        self.assertEqual(result["rectification_support_fraction"], 11 / 12)
        self.assertGreater(result["mask_boundary_pixels"], 0)
        with self.assertRaises(ValueError):
            benchmark.coverage_summary(depth, np.ones((9, 12), dtype=bool))

    def test_report_keeps_truth_null_and_records_each_input_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            left_path = base / "left.png"
            right_path = base / "right.png"
            calibration_path = base / "calibration.npz"
            output_path = base / "benchmark.json"
            left = np.zeros((36, 64), np.uint8)
            right = np.full((36, 64), 127, np.uint8)
            self.assertTrue(cv2.imwrite(str(left_path), left))
            self.assertTrue(cv2.imwrite(str(right_path), right))
            calibration_path.write_bytes(b"fixture-calibration")
            report = benchmark.run_benchmark(
                left_path, right_path, calibration_path, output_path,
                scales=(.375,), threads=(2,), input_sizes=((32, 18),),
                preprocess_modes=("cpu",), warmup=1, runs=2,
                engine_factory=FakeEngine)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], "fast-sgbm-profile-benchmark-v1")
        self.assertEqual(len(saved["profiles"]), 1)
        profile = saved["profiles"][0]
        self.assertEqual(profile["input_resolution_wh"], [32, 18])
        self.assertEqual(profile["process_ms"].keys(), {"p50", "p95"})
        self.assertEqual(profile["stage_ms"]["match"], {"p50": 3.0, "p95": 3.0})
        self.assertIsNone(profile["accuracy"]["independent_ground_truth"])
        self.assertIsNone(profile["accuracy"]["median_error_mm"])
        self.assertTrue(saved["comparison_notes"]["coverage_is_not_accuracy"])


if __name__ == "__main__":
    unittest.main()
