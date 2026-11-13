"""Differential host checks for the SGBM metric-filter migration."""
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
from fast_depth import FastSGBMDepth


class FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        library = pathlib.Path(cls.tmp.name) / ("filter.dll" if sys.platform == "win32" else "libfilter.so")
        link = ["-static-libstdc++", "-static-libgcc"] if sys.platform == "win32" else []
        subprocess.run(["g++", "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-shared",
                        str(HERE / "depth_filter.cpp"), *link, "-o", str(library)], check=True)
        cls.dll_dir = os.add_dll_directory(str(pathlib.Path(shutil.which("g++")).parent)) if sys.platform == "win32" else None
        cls.lib = ctypes.CDLL(str(library))
        cls.func = cls.lib.agtc_metric_depth_f32
        fptr, bptr = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
        cls.func.argtypes = [fptr, bptr, bptr, ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_double, ctypes.c_double, fptr, bptr]
        cls.func.restype = ctypes.c_int
        cls.workspace_size = cls.lib.agtc_metric_depth_workspace_size
        cls.workspace_size.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        cls.workspace_size.restype = ctypes.c_size_t
        cls.with_workspace = cls.lib.agtc_metric_depth_f32_with_workspace
        cls.with_workspace.argtypes = cls.func.argtypes + [bptr, ctypes.c_size_t]
        cls.with_workspace.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        if sys.platform == "win32":
            import _ctypes
            _ctypes.FreeLibrary(cls.lib._handle)
            cls.dll_dir.close()
        cls.tmp.cleanup()

    def run_case(self, disparity, masks, offset, factor):
        height, width = disparity.shape
        engine = FastSGBMDepth.__new__(FastSGBMDepth)
        engine.width, engine.height = width, height
        engine.xx, engine.yy = np.indices((height, width))[1], np.indices((height, width))[0]
        engine.window_masks = tuple(mask.astype(bool) for mask in masks)
        engine.disparity_offset, engine.depth_factor = offset, factor
        with np.errstate(invalid="ignore", over="ignore"):
            expected_depth, expected_valid = engine.metric_values(disparity, None, None)
        output = np.empty_like(disparity)
        valid = np.empty_like(masks[0])
        ptr = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
        rc = self.func(ptr(disparity, ctypes.c_float), ptr(masks[0], ctypes.c_uint8),
                       ptr(masks[1], ctypes.c_uint8), width, height, offset, factor,
                       ptr(output, ctypes.c_float), ptr(valid, ctypes.c_uint8))
        self.assertEqual(rc, 0)
        np.testing.assert_array_equal(valid.astype(bool), expected_valid)
        np.testing.assert_allclose(output, expected_depth, rtol=1e-6, atol=1e-6, equal_nan=True)
        scratch = np.full(self.workspace_size(width, height), 0xA5, np.uint8)
        reused_depth, reused_valid = np.empty_like(output), np.empty_like(valid)
        rc = self.with_workspace(ptr(disparity, ctypes.c_float), ptr(masks[0], ctypes.c_uint8),
            ptr(masks[1], ctypes.c_uint8), width, height, offset, factor,
            ptr(reused_depth, ctypes.c_float), ptr(reused_valid, ctypes.c_uint8),
            ptr(scratch, ctypes.c_uint8), scratch.size)
        self.assertEqual(rc, 0)
        np.testing.assert_array_equal(reused_valid, valid)
        np.testing.assert_array_equal(reused_depth, output)

    def test_random_and_boundary_data(self):
        rng = np.random.default_rng(13011)
        for height, width in ((1, 1), (2, 4), (11, 17), (240, 131)):
            for offset, factor in ((0.0, 8.0), (0.233456789, 27.3456789)):
                d = rng.uniform(-2, 35, (height, width)).astype(np.float32)
                if d.size >= 8:
                    d.flat[:8] = [np.nan, np.inf, -np.inf, 0, offset + .01,
                                  offset + factor / .3, offset + factor / 5, 1.5]
                masks = tuple((rng.random((height, width)) > 0.1).astype(np.uint8) for _ in range(2))
                self.run_case(d, masks, offset, factor)

    def test_rejects_invalid_buffers(self):
        self.assertEqual(self.func(None, None, None, 0, 0, 0, 1, None, None), -1)
        max_size = ctypes.c_size_t(-1).value
        self.assertEqual(self.workspace_size(0, 1), 0)
        self.assertEqual(self.workspace_size(1, max_size), 0)
        self.assertEqual(self.workspace_size(max_size, 2), 0)

    def test_workspace_reuse_and_short_buffer(self):
        height, width = 4, 12
        disparity = np.full((height, width), 2.0, np.float32)
        support = np.ones((height, width), np.uint8)
        depth = np.full_like(disparity, -7.0)
        valid = np.full_like(support, 77)
        scratch = np.full(self.workspace_size(width, height), 0xA5, np.uint8)
        ptr = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
        args = (ptr(disparity, ctypes.c_float), ptr(support, ctypes.c_uint8),
                ptr(support, ctypes.c_uint8), width, height, 0.0, 2.0,
                ptr(depth, ctypes.c_float), ptr(valid, ctypes.c_uint8), ptr(scratch, ctypes.c_uint8))
        self.assertEqual(self.with_workspace(*args, scratch.size - 1), -1)
        self.assertTrue(np.all(depth == -7.0))
        self.assertTrue(np.all(valid == 77))
        self.assertEqual(self.with_workspace(*args, scratch.size), 0)
        self.assertTrue(np.any(valid))
        disparity.fill(np.nan)
        self.assertEqual(self.with_workspace(*args, scratch.size), 0)
        self.assertTrue(np.all(valid == 0))
        self.assertTrue(np.all(np.isnan(depth)))

    def test_range_and_disparity_boundaries_with_support(self):
        height, width = 12, 131
        masks = (np.ones((height, width), np.uint8), np.ones((height, width), np.uint8))
        for offset, factor in ((0.0, 8.0), (0.233456789, 27.3456789)):
            points = [offset + .01, offset + factor / .3, offset + factor / 5.0,
                      0, 1, 50, 100, 130]
            for point in points:
                p = np.float32(point)
                for value in (np.nextafter(p, np.float32(-np.inf)), p,
                              np.nextafter(p, np.float32(np.inf))):
                    self.run_case(np.full((height, width), value, dtype=np.float32), masks, offset, factor)

    def test_recorded_stereo_disparity(self):
        captures = HERE.parent.parent / "stereo" / "captures"
        capture = captures / "20240825T221623918473Z-946d97fc"
        left = cv2.imdecode(np.fromfile(capture / "left.png", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        right = cv2.imdecode(np.fromfile(capture / "right.png", dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        engine = FastSGBMDepth(HERE.parent / "app" / "calibration.npz")
        left_rectified, right_rectified = engine.rectify(left, right)
        disparity, _ = engine.disparities(left_rectified, right_rectified)
        disparity = np.ascontiguousarray(disparity)
        masks = tuple(np.ascontiguousarray(mask, dtype=np.uint8) for mask in engine.window_masks)
        self.run_case(disparity, masks, engine.disparity_offset, engine.depth_factor)


if __name__ == "__main__":
    unittest.main()
