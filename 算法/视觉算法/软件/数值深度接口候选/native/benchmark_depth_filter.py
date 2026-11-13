"""Host-only filter microbenchmark on recorded stereo disparity."""
import ctypes
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from test_depth_filter import FilterTest, FastSGBMDepth, HERE

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, help="new JSON evidence file; refuses to overwrite")
options = parser.parse_args()
FilterTest.setUpClass()
try:
    capture = HERE.parent.parent / "stereo" / "captures" / "20240825T221623918473Z-946d97fc"
    left, right = (cv2.imdecode(np.fromfile(capture / (name + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
                   for name in ("left", "right"))
    engine = FastSGBMDepth(HERE.parent / "app" / "calibration.npz")
    left, right = engine.rectify(left, right)
    disparity, _ = engine.disparities(left, right)
    disparity = np.ascontiguousarray(disparity)
    masks = tuple(np.ascontiguousarray(mask, dtype=np.uint8) for mask in engine.window_masks)
    result = np.empty_like(disparity)
    valid = np.empty_like(masks[0])
    ptr = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
    args = (ptr(disparity, ctypes.c_float), ptr(masks[0], ctypes.c_uint8),
            ptr(masks[1], ctypes.c_uint8), disparity.shape[1], disparity.shape[0],
            engine.disparity_offset, engine.depth_factor,
            ptr(result, ctypes.c_float), ptr(valid, ctypes.c_uint8))
    baseline = lambda: engine.metric_values(disparity, None, None)
    native = lambda: FilterTest.func(*args)
    workspace = np.empty(FilterTest.workspace_size(disparity.shape[1], disparity.shape[0]), np.uint8)
    workspace_pointer = ptr(workspace, ctypes.c_uint8)
    native_reuse = lambda: FilterTest.with_workspace(*args, workspace_pointer, workspace.size)
    expected, expected_valid = baseline()
    assert native() == 0
    np.testing.assert_array_equal(valid.astype(bool), expected_valid)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert native_reuse() == 0
    np.testing.assert_array_equal(valid.astype(bool), expected_valid)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    def measure(call):
        for _ in range(50): call()
        samples = []
        for _ in range(1000):
            start = time.perf_counter_ns(); call(); samples.append((time.perf_counter_ns() - start) / 1000)
        ordered = sorted(samples)
        return {"median_us": statistics.median(samples), "p95_us": ordered[949], "p99_us": ordered[989],
                "samples_us": samples}
    sources = [HERE / "depth_filter.cpp", HERE / "depth_filter.h", HERE.parent / "app" / "fast_depth.py",
               HERE.parent / "app" / "calibration.npz", capture / "left.png", capture / "right.png"]
    hashes = {str(path.relative_to(HERE.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "compiler_flags": "g++ -O3", "shape": list(disparity.shape), "samples": 1000,
        "input_and_source_sha256": hashes,
        "baseline": measure(baseline), "native": measure(native), "native_reused_workspace": measure(native_reuse),
        "caveat": "baseline also calculates filter statistics; native uses preallocated outputs and omits statistics; native wrapper allocates scratch per call while reuse supplies scratch; host temperature, frequency and power were not recorded"}
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"output": str(options.output) if options.output else None,
                      "baseline": {k: v for k, v in report["baseline"].items() if k != "samples_us"},
                      "native": {k: v for k, v in report["native"].items() if k != "samples_us"},
                      "native_reused_workspace": {k: v for k, v in report["native_reused_workspace"].items() if k != "samples_us"},
                      "caveat": report["caveat"]}, indent=2))
finally:
    FilterTest.tearDownClass()
