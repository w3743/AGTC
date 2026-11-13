"""Host-side differential test for the native candidate; no device deployment."""
import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
import depth_wire


class NativeEncoderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        library = pathlib.Path(cls.tmp.name) / ("depth.dll" if sys.platform == "win32" else "libdepth.so")
        tune = ["-march=native"] if os.environ.get("AGTC_BENCH_NATIVE") == "1" else []
        subprocess.run(["g++", "-std=c++17", "-O3", *tune, "-Wall", "-Wextra", "-Werror", "-shared", "-o", str(library),
                        str(HERE / "depth_encode.cpp")], check=True)
        cls.lib = ctypes.CDLL(str(library))
        for suffix, ctype in (("f32", ctypes.c_float), ("f64", ctypes.c_double)):
            function = getattr(cls.lib, "agtc_depth_encode_" + suffix)
            function.argtypes = [ctypes.POINTER(ctype), ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
            function.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        if sys.platform == "win32":
            import _ctypes
            _ctypes.FreeLibrary(cls.lib._handle)
        cls.tmp.cleanup()

    def test_matches_python_reference(self):
        rng = np.random.default_rng(20261113)
        for dtype, suffix, ctype in ((np.float32, "f32", ctypes.c_float),
                                     (np.float64, "f64", ctypes.c_double)):
            boundaries = np.array([np.nan, np.inf, -np.inf, 0, 0.3, 5, 5.001,
                                   0.3005, 1.2345, 1.2355], dtype=dtype)
            endpoints = np.array([0.3, 5.0], dtype=dtype)
            ties = ((np.arange(300, 5000, dtype=np.float64) + 0.5) / 1000).astype(dtype)
            values = np.concatenate((boundaries, np.nextafter(endpoints, dtype(-np.inf)),
                                     np.nextafter(endpoints, dtype(np.inf)),
                                     ties, np.nextafter(ties, dtype(-np.inf)),
                                     np.nextafter(ties, dtype(np.inf)),
                                     rng.uniform(-1, 7, 200000).astype(dtype)))
            values = np.ascontiguousarray(values.reshape(1, -1))
            expected, _, _ = depth_wire.encode_depth_payload(values)
            output = (ctypes.c_uint8 * (values.size * 2))()
            func = getattr(self.lib, "agtc_depth_encode_" + suffix)
            self.assertEqual(func(values.ctypes.data_as(ctypes.POINTER(ctype)), values.size,
                                  output, len(output)), 0)
            self.assertEqual(bytes(output), expected, suffix)

    def test_rejects_invalid_arguments(self):
        func = self.lib.agtc_depth_encode_f32
        value = (ctypes.c_float * 1)(1.0)
        out = (ctypes.c_uint8 * 2)(42, 42)
        for ptr, count, dest, capacity in ((None, 1, out, 2), (value, 0, out, 2),
                                            (value, 1, None, 2), (value, 1, out, 1)):
            self.assertEqual(func(ptr, count, dest, capacity), -1)
        self.assertEqual(bytes(out), bytes([42, 42]))


if __name__ == "__main__":
    unittest.main()
