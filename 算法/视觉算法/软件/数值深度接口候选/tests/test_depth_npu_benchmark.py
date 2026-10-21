import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

import benchmark_npu


class NpuBenchmarkSummaryTests(unittest.TestCase):
    def test_profile_summary_separates_latency_from_combined_throughput(self):
        summary = benchmark_npu.summarize_profile(
            "three_sessions_core1_2_4",
            [1, 2, 4],
            [[10.0, 20.0], [8.0, 16.0], [5.0, 15.0]],
            wall_clock_s=2.0,
        )
        self.assertEqual(summary["measured_inference_count"], 6)
        self.assertEqual(summary["per_inference_latency_ms"], {"p50": 12.5, "p95": 19.0})
        self.assertEqual(summary["aggregate_throughput_fps"], 3.0)
        self.assertEqual([item["core_mask"] for item in summary["per_session"]], [1, 2, 4])
        self.assertIn("concurrent sessions", summary["throughput_semantics"])

    def test_reference_comparison_uses_content_roi_and_reports_pixel_delta(self):
        expected = np.array(
            [[0, 0, 100, 100], [1, 2, 100, 100], [3, 4, 100, 100]],
            dtype=np.float32,
        )
        prediction = np.array(
            [[0, 0, -100, -100], [2, 2, -100, -100], [1, 6, -100, -100]],
            dtype=np.float32,
        )
        result = benchmark_npu.compare_reference_output(
            prediction, expected, content_width=2, pad_top=1
        )
        self.assertTrue(result["shape_match"])
        self.assertTrue(result["prediction_all_finite"])
        self.assertTrue(result["expected_all_finite"])
        self.assertEqual(result["comparison_region"]["finite_pixels"], 4)
        self.assertEqual(
            result["absolute_disparity_delta_px"],
            {"mean": 1.25, "p50": 1.5, "p95": 2.0, "max": 2.0},
        )
        self.assertFalse(result["within_reference_tolerance"])

    def test_reference_comparison_reports_shape_or_finite_failures(self):
        expected = np.ones((2, 3), dtype=np.float32)
        mismatch = benchmark_npu.compare_reference_output(
            np.ones((3, 2), dtype=np.float32), expected, content_width=3, pad_top=0
        )
        self.assertFalse(mismatch["shape_match"])
        self.assertIsNone(mismatch["absolute_disparity_delta_px"])

        prediction = np.ones((2, 3), dtype=np.float32)
        prediction[0, 0] = np.nan
        nonfinite = benchmark_npu.compare_reference_output(
            prediction, expected, content_width=3, pad_top=0
        )
        self.assertFalse(nonfinite["prediction_all_finite"])
        self.assertEqual(nonfinite["comparison_region"]["finite_pixels"], 5)
        self.assertFalse(nonfinite["within_reference_tolerance"])

    def test_help_is_available_without_importing_rknnlite(self):
        self.assertNotIn("rknnlite", sys.modules)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                benchmark_npu.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--runtime", output.getvalue())
        self.assertNotIn("rknnlite", sys.modules)


if __name__ == "__main__":
    unittest.main()
