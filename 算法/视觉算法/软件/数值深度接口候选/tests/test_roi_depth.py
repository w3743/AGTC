import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from roi_depth import ROIDepthError, ROISGBMDepth  # noqa: E402


K = np.array([[400.0, 0.0, 95.5],
              [0.0, 400.0, 63.5],
              [0.0, 0.0, 1.0]], dtype=np.float64)


def synthetic_pair(disparity, height=128, width=192, seed=81):
    rng = np.random.default_rng(seed)
    # High-frequency, deterministic texture gives SGBM a known integer shift.
    left = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0.45)
    right = np.zeros_like(left)
    if disparity >= 0:
        right[:, :width - disparity] = left[:, disparity:]
    else:
        shift = -disparity
        right[:, shift:] = left[:, :width - shift]
    left_valid = np.ones((height, width), dtype=np.bool_)
    right_valid = np.ones((height, width), dtype=np.bool_)
    if disparity >= 0:
        left_valid[:, :disparity] = False
        right_valid[:, width - disparity:] = False
    else:
        shift = -disparity
        left_valid[:, width - shift:] = False
        right_valid[:, :shift] = False
    return left, right, left_valid, right_valid


def matcher(**overrides):
    options = dict(
        K=K,
        fB_px_m=240.0,
        disparity_offset_px=2.0,
        min_disparity=0,
        num_disparities=32,
        block_size=5,
        depth_range_m=(0.3, 50.0),
        speckle_window_size=0,
    )
    options.update(overrides)
    return ROISGBMDepth(**options)


class ROIDepthTests(unittest.TestCase):
    def test_known_shift_depth_offset_roi_k_and_separate_timings(self):
        left, right, left_valid, right_valid = synthetic_pair(10)
        roi = (48, 32, 64, 48)
        result = matcher().compute(
            left, right, left_valid, right_valid, roi,
            full_frame_rectification_ms=12.5,
        )
        self.assertEqual(result.depth_z_m.shape, (48, 64))
        self.assertEqual(result.depth_z_m.dtype, np.float32)
        self.assertEqual(result.valid.dtype, np.bool_)
        self.assertGreater(float(result.valid.mean()), 0.5)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 10.0, delta=0.4)
        # Explicit nonzero disparity offset is respected: 240 / (10 - 2) = 30 m.
        self.assertAlmostEqual(float(np.nanmedian(result.depth_z_m)), 30.0, delta=1.5)
        self.assertEqual(result.roi_xywh, roi)
        self.assertAlmostEqual(result.K_roi[0, 2], K[0, 2] - roi[0])
        self.assertAlmostEqual(result.K_roi[1, 2], K[1, 2] - roi[1])
        np.testing.assert_array_equal(
            result.roi_to_source,
            np.array([[1.0, 0.0, roi[0]], [0.0, 1.0, roi[1]], [0.0, 0.0, 1.0]])
        )
        self.assertIsNone(result.physical_accuracy_claim_m)
        self.assertEqual(result.timings_ms["full_frame_rectification_ms"], 12.5)
        self.assertGreaterEqual(result.timings_ms["matcher_compute_ms"], 0.0)
        self.assertGreaterEqual(result.timings_ms["mask_and_depth_ms"], 0.0)
        self.assertGreaterEqual(result.timings_ms["roi_total_ms_excluding_rectification"], 0.0)

    def test_lr_check_is_default_and_reports_full_canvas_reverse_cost(self):
        left, right, left_valid, right_valid = synthetic_pair(10)
        roi = (48, 32, 64, 48)
        result = matcher().compute(left, right, left_valid, right_valid, roi)

        self.assertTrue(result.compute_geometry["lr_consistency_enabled"])
        self.assertEqual(result.compute_geometry["lr_max_diff_px"], 1.0)
        self.assertGreater(float(result.valid.mean()), 0.5)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 10.0, delta=0.4)
        reverse = result.compute_geometry["reverse_match"]
        self.assertEqual(reverse["scope"], "full_rectified_canvas")
        self.assertEqual(reverse["input_shape_hw"][0], left.shape[0])
        self.assertEqual(reverse["input_area_px"],
                         reverse["input_shape_hw"][0] * reverse["input_shape_hw"][1])
        self.assertGreaterEqual(result.timings_ms["reverse_canvas_prepare_ms"], 0.0)
        self.assertGreaterEqual(result.timings_ms["reverse_matcher_compute_ms"], 0.0)
        self.assertGreaterEqual(result.timings_ms["lr_consistency_filter_ms"], 0.0)

        # LR can be disabled only for an explicit benchmark/ablation; its cost
        # is then absent rather than reported as zero work.
        forward_only = matcher(lr_consistency=False).compute(
            left, right, left_valid, right_valid, roi
        )
        self.assertFalse(forward_only.compute_geometry["lr_consistency_enabled"])
        self.assertIsNone(forward_only.compute_geometry["reverse_match"])
        self.assertIsNone(forward_only.timings_ms["reverse_canvas_prepare_ms"])
        self.assertIsNone(forward_only.timings_ms["reverse_matcher_compute_ms"])
        self.assertIsNone(forward_only.timings_ms["lr_consistency_filter_ms"])

    def test_lr_check_rejects_occluded_and_random_mismatches(self):
        # Start from a known 10 px shift, then replace the matching right-image
        # patch with unrelated texture. Both masks remain true so this tests LR
        # consistency itself rather than source-window masking.
        left, right, left_valid, right_valid = synthetic_pair(10, seed=42)
        rng = np.random.default_rng(104)
        x0, y0, x1, y1 = 70, 42, 105, 88
        right[y0:y1, x0 - 10:x1 - 10] = rng.integers(
            0, 256, size=(y1 - y0, x1 - x0), dtype=np.uint8
        )
        roi = (48, 32, 64, 48)
        forward_only = matcher(lr_consistency=False).compute(
            left, right, left_valid, right_valid, roi
        )
        checked = matcher(lr_consistency=True, lr_max_diff_px=1.0).compute(
            left, right, left_valid, right_valid, roi
        )
        local_region = np.s_[y0 - roi[1]:y1 - roi[1], x0 - roi[0]:x1 - roi[0]]
        occluded_forward_coverage = float(forward_only.valid[local_region].mean())
        occluded_checked_coverage = float(checked.valid[local_region].mean())
        self.assertGreater(occluded_forward_coverage, 0.2)
        self.assertLess(occluded_checked_coverage, occluded_forward_coverage - 0.1)

        # Completely unrelated random views can still produce plausible local
        # SGBM matches; explicit LR checking must reject most of those pixels.
        unrelated_right = rng.integers(0, 256, size=left.shape, dtype=np.uint8)
        unrelated_forward = matcher(lr_consistency=False).compute(
            left, unrelated_right, left_valid, right_valid, roi
        )
        unrelated_checked = matcher(lr_consistency=True, lr_max_diff_px=1.0).compute(
            left, unrelated_right, left_valid, right_valid, roi
        )
        self.assertGreater(float(unrelated_forward.valid.mean()), 0.2)
        self.assertLess(float(unrelated_checked.valid.mean()),
                        float(unrelated_forward.valid.mean()) - 0.1)

    def test_canvas_contains_entire_right_correspondence_window_and_reports_cost(self):
        left, right, left_valid, right_valid = synthetic_pair(10)
        roi = (48, 32, 64, 48)
        result = matcher().compute(left, right, left_valid, right_valid, roi)
        geom = result.compute_geometry
        support_min, support_max = geom["right_search_support_x_inclusive"]
        crop_x, _, crop_w, _ = geom["matcher_source_crop_xywh"]
        self.assertLessEqual(crop_x, max(0, support_min))
        self.assertGreaterEqual(crop_x + crop_w - 1, min(left.shape[1] - 1, support_max))
        h, w = geom["matcher_input_shape_hw"]
        self.assertEqual(geom["matcher_input_area_px"], h * w)
        self.assertLess(geom["matcher_input_area_px"], geom["full_source_area_px"])
        self.assertEqual(geom["padding_is_synthetic_and_invalid"], True)

    def test_native_640x320_d64_roi_has_reported_search_cost_and_expected_z(self):
        left, right, left_valid, right_valid = synthetic_pair(30, height=320, width=640, seed=17)
        native_k = np.array([[2359.795, 0.0, 320.0],
                             [0.0, 2359.795, 160.0],
                             [0.0, 0.0, 1.0]])
        result = matcher(
            K=native_k,
            fB_px_m=227.21656,
            disparity_offset_px=0.0,
            num_disparities=64,
            depth_range_m=(5.0, 50.0),
        ).compute(left, right, left_valid, right_valid, (64, 0, 576, 320),
                  full_frame_rectification_ms=18.0)
        self.assertGreater(float(result.valid.mean()), 0.7)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 30.0, delta=0.5)
        self.assertAlmostEqual(float(np.nanmedian(result.depth_z_m)), 227.21656 / 30.0, delta=0.2)
        self.assertEqual(result.compute_geometry["matcher_input_shape_hw"][0], 320)
        self.assertLessEqual(result.compute_geometry["matcher_input_shape_hw"][1], 642)
        self.assertEqual(result.timings_ms["full_frame_rectification_ms"], 18.0)

    def test_left_padding_keeps_low_x_roi_search_and_right_mask_rejects_match(self):
        left, right, left_valid, right_valid = synthetic_pair(8)
        roi = (10, 32, 48, 40)
        result = matcher().compute(left, right, left_valid, right_valid, roi)
        self.assertGreater(float(result.valid.mean()), 0.2)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 8.0, delta=0.5)
        self.assertGreater(result.compute_geometry["matcher_padding_ltrb"][0], 0)

        # Remove a patch of right-camera source support at the exact x-d match.
        invalid_right = right_valid.copy()
        y_abs, x_abs = 50, 40
        invalid_right[y_abs - 3:y_abs + 4, x_abs - 8 - 3:x_abs - 8 + 4] = False
        masked = matcher().compute(left, right, left_valid, invalid_right, roi)
        local_y, local_x = y_abs - roi[1], x_abs - roi[0]
        self.assertFalse(masked.valid[local_y, local_x])
        self.assertTrue(np.isnan(masked.depth_z_m[local_y, local_x]))

    def test_nonzero_min_disparity_preserves_roi_correspondence_support(self):
        left, right, left_valid, right_valid = synthetic_pair(10)
        result = matcher(min_disparity=8).compute(
            left, right, left_valid, right_valid, (48, 32, 64, 48)
        )
        self.assertGreater(float(result.valid.mean()), 0.5)
        self.assertAlmostEqual(float(np.nanmedian(result.disparity_px)), 10.0, delta=0.5)
        self.assertEqual(result.compute_geometry["right_search_support_x_inclusive"], [9, 103])

    def test_out_of_frame_right_correspondence_stays_invalid_without_fill(self):
        left, right, left_valid, right_valid = synthetic_pair(16)
        roi = (8, 28, 32, 32)
        result = matcher().compute(left, right, left_valid, right_valid, roi)
        # Source left x=8 and disparity 16 would correspond to x=-8. Padding is
        # available to SGBM but cannot make that physical source point valid.
        self.assertFalse(result.valid[:, 0].any())
        self.assertTrue(np.isnan(result.depth_z_m[:, 0]).all())

    def test_rejects_bad_inputs_geometry_and_ranges(self):
        left, right, left_valid, right_valid = synthetic_pair(10)
        with self.assertRaises(ROIDepthError):
            matcher(K=[[400.0, 0.1, 95.5], [0, 400, 63.5], [0, 0, 1]])
        with self.assertRaises(ROIDepthError):
            matcher(fB_px_m=float("nan"))
        with self.assertRaises(ROIDepthError):
            matcher(num_disparities=24)
        with self.assertRaises(ROIDepthError):
            matcher(depth_range_m=(50.0, 0.3))
        with self.assertRaises(ROIDepthError):
            matcher().compute(left, right, left_valid.astype(np.uint8), right_valid,
                              (48, 32, 64, 48))
        with self.assertRaises(ROIDepthError):
            matcher().compute(left, right, left_valid, right_valid, (180, 32, 32, 48))
        with self.assertRaises(ROIDepthError):
            matcher().compute(left[:, :-1], right, left_valid, right_valid, (20, 20, 40, 40))


if __name__ == "__main__":
    unittest.main()
