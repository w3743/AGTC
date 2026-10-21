import sys
import unittest
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from calibration import load_model  # noqa: E402
from pointcloud import PointCloudStream  # noqa: E402


SHAPES = ((240, 131), (320, 176), (480, 264), (640, 352))


class PointCloudGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration_path = APP / "calibration.npz"
        cls.model = load_model(cls.calibration_path)
        cls.projector = PointCloudStream(cls.calibration_path)
        cls.p1 = np.asarray(cls.model["P1"], dtype=np.float64)
        cls.p2 = np.asarray(cls.model["P2"], dtype=np.float64)

    def expected_scaled_k_from_p1(self, shape):
        height, _ = shape
        sensor_scale = height / 3840.0
        crop = max(1, int(round(4 * height / 640.0)))
        resize = np.array([[sensor_scale, 0.0, (sensor_scale - 1.0) / 2.0],
                           [0.0, sensor_scale, (sensor_scale - 1.0) / 2.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        scaled_p1 = resize @ self.p1
        # Rectified sensor coordinates are turned CCW, then crop is removed
        # from the resulting horizontal axis.
        return np.array([
            [scaled_p1[1, 1], 0.0, scaled_p1[1, 2] - crop],
            [0.0, scaled_p1[0, 0], height - 1.0 - scaled_p1[0, 2]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def test_scaled_k_matches_actual_p1_resize_rotation_and_rounded_crop(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                actual = self.projector.scaled_intrinsics(shape)
                expected = self.expected_scaled_k_from_p1(shape)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)

    def test_q_reprojection_z_and_inverse_k_rays_agree_at_all_scales(self):
        z_expected = 1.25
        for shape in SHAPES:
            height, width = shape
            sensor_scale = height / 3840.0
            k = self.projector.scaled_intrinsics(shape)
            q = self.projector.scaled_q(shape)

            # Independently scale P1/P2's baseline disparity terms rather than
            # obtaining them from the Q helper under test.
            factor = -float(self.p2[1, 3]) * sensor_scale / 1000.0
            offset = float(self.p1[1, 2] - self.p2[1, 2]) * sensor_scale
            q_factor, q_offset = self.projector._depth_conversion(shape, None, None)
            self.assertAlmostEqual(q_factor, factor, places=10)
            self.assertAlmostEqual(q_offset, offset, places=10)

            pixels = ((width // 2, height // 2),
                      (min(width - 1, 7), min(height - 1, 9)),
                      (width - 3, height - 5))
            disparity = factor / z_expected + offset
            for x, y in pixels:
                with self.subTest(shape=shape, pixel=(x, y)):
                    homogeneous = q @ np.array([float(x), float(y), disparity, 1.0])
                    xyz_q = homogeneous[:3] / homogeneous[3]
                    ray = np.linalg.solve(k, np.array([float(x), float(y), 1.0]))
                    xyz_k = ray * (z_expected / ray[2])
                    self.assertAlmostEqual(float(xyz_q[2]), z_expected, places=10)
                    np.testing.assert_allclose(xyz_q, xyz_k, rtol=0.0, atol=1e-9)

            # Validate the actual pixel_to_xyz path, which consumes scaled_q.
            depth = np.full(shape, z_expected, dtype=np.float32)
            x, y = pixels[0]
            xyz_pixel = self.projector.pixel_to_xyz(depth, x, y)
            self.assertIsNotNone(xyz_pixel)
            self.assertAlmostEqual(xyz_pixel[2], z_expected, places=5)
            ray = np.linalg.solve(k, np.array([float(x), float(y), 1.0]))
            xyz_k = ray * (z_expected / ray[2])
            np.testing.assert_allclose(xyz_pixel, xyz_k, rtol=0.0, atol=1e-5)

    def test_wrong_portrait_width_is_rejected(self):
        with self.assertRaises(ValueError):
            self.projector.scaled_intrinsics((240, 120))


if __name__ == "__main__":
    unittest.main()
