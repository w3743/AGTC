"""Reproducible host microbenchmark; does not represent device latency."""
import ctypes
import json
import os
import platform
import statistics
import time

import numpy as np
from test_depth_encode import NativeEncoderTest, depth_wire

N = 1000
NativeEncoderTest.setUpClass()
try:
    rng = np.random.default_rng(20261113)
    depth = rng.uniform(0.1, 6.0, (240, 131)).astype(np.float32)
    depth.ravel()[::7] = np.nan
    expected = depth_wire.encode_depth_payload(depth)[0]
    out = (ctypes.c_uint8 * (depth.size * 2))()
    func = NativeEncoderTest.lib.agtc_depth_encode_f32
    ptr = depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    assert func(ptr, depth.size, out, len(out)) == 0 and bytes(out) == expected
    def measure(call):
        for _ in range(50):
            call()
        samples = []
        for _ in range(N):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1000)
        samples.sort()
        return {"median_us": statistics.median(samples), "p95_us": samples[int(N * .95) - 1],
                "p99_us": samples[int(N * .99) - 1], "max_us": samples[-1]}
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "numpy": np.__version__,
              "shape": list(depth.shape), "dtype": str(depth.dtype), "iterations": N,
              "compiler_flags": "-O3 -march=native" if os.environ.get("AGTC_BENCH_NATIVE") == "1" else "-O3",
              "native_output": "preallocated", "baseline_output": "new bytes each call",
              "baseline": measure(lambda: depth_wire.encode_depth_payload(depth)),
              "native": measure(lambda: func(ptr, depth.size, out, len(out)))}
    print(json.dumps(result, indent=2))
finally:
    NativeEncoderTest.tearDownClass()
