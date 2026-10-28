"""Metric units, rotation/disparity sign, and invalid correspondence checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from depth import MetricDepth

PARAMETERS = Path(__file__).with_name('calibration') / 'current' / 'parameters.npz'


class DepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.matcher = MetricDepth(PARAMETERS)

    def test_projection_depth_after_ccw_and_resize(self):
        m = self.matcher
        with np.load(PARAMETERS, allow_pickle=False) as p:
            for z in (400., 1000., 3000.):
                xyz = p['R1'] @ np.array([35., -20., z])
                homogeneous = np.append(xyz, 1)
                a, b = p['P1'] @ homogeneous, p['P2'] @ homogeneous
                a, b = a[:2]/a[2], b[:2]/b[2]
                # CCW x' is old y; resized pixel-center shifts cancel in disparity.
                disparity = (a[1]-b[1]) * m.RAW_SIZE[0]/m.SENSOR_SIZE[0]
                estimate_m = m.depth_factor/(disparity-m.disparity_offset)
                self.assertAlmostEqual(estimate_m, xyz[2]/1000, places=10)
                self.assertGreater(disparity, 0)

    def test_textured_plane_and_mismatched_right_image(self):
        m = self.matcher
        rng = np.random.default_rng(3588)
        left = rng.integers(0, 256, (m.height, m.width), dtype=np.uint8)
        shift = round(m.depth_factor)
        right = rng.integers(0, 256, left.shape, dtype=np.uint8)
        right[:, :-shift] = left[:, shift:]
        dl, dr = m.disparities(left, right)
        depth, valid = m.metric_values(dl, dr, left)
        self.assertGreater(valid.mean(), .15)
        expected = m.depth_factor/(shift-m.disparity_offset)
        self.assertLess(abs(float(np.nanmedian(depth))-expected), .005)
        self.assertTrue(np.isnan(depth[~valid]).all())
        _, failed = m.metric_values(dl, np.zeros_like(dr), left)
        self.assertLess(failed.mean(), .001)
        _, zero = m.metric_values(np.zeros_like(dl), np.zeros_like(dr), left)
        self.assertFalse(zero.any())

    def test_reject_draft_and_wrong_frame_geometry(self):
        with np.load(PARAMETERS, allow_pickle=False) as source:
            values = dict(source)
        values['status'] = 'needs_review_not_for_mapping'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draft.npz'
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, '尚未通过'):
                MetricDepth(path)
        with self.assertRaisesRegex(ValueError, '3840'):
            self.matcher.rectify(np.zeros((640, 360), np.uint8), np.zeros((640, 360), np.uint8))

    def test_auto_color_and_statistics_preserve_metric_depth(self):
        m = self.matcher
        image = np.random.default_rng(7).integers(0, 256, (m.height, m.width), dtype=np.uint8)
        dl = np.full(image.shape, 40, np.float32)
        dr = -dl
        with patch.object(m, 'rectify', return_value=(image, image)), patch.object(m, 'disparities', return_value=(dl, dr)):
            automatic, depth, stats = m.compute(None, None)
        frozen = depth.copy()
        fixed = m.colorize(depth, m.RANGE_M)
        self.assertTrue(np.array_equal(depth, frozen, equal_nan=True))
        self.assertFalse(np.array_equal(automatic, fixed))
        self.assertAlmostEqual(stats['depth_min_m'], m.depth_factor/40, places=3)
        self.assertEqual(stats['depth_min_m'], stats['depth_max_m'])
        self.assertGreater(stats['depth_color_range_m'][1]-stats['depth_color_range_m'][0], .20)
        self.assertTrue((automatic[~np.isfinite(depth)] == 0).all())
        with patch.object(m, 'rectify', return_value=(image, image)), patch.object(m, 'disparities', return_value=(dl*0, dr*0)):
            color, _, empty_stats = m.compute(None, None)
        self.assertFalse(color.any())
        self.assertIsNone(empty_stats['depth_min_m'])
        self.assertIsNone(empty_stats['depth_max_m'])

    def test_patch_contrast_retains_smooth_texture_but_rejects_flat(self):
        m = self.matcher
        x = np.arange(m.width, dtype=np.float32)
        image = np.tile(100+12*np.sin(x/4), (m.height, 1)).astype(np.uint8)
        dl = np.full(image.shape, 40, np.float32)
        _, valid = m.metric_values(dl, -dl, image)
        self.assertGreater(valid.mean(), .2)
        _, flat = m.metric_values(dl, -dl, np.full_like(image, 100))
        self.assertFalse(flat.any())


if __name__ == '__main__':
    unittest.main()
