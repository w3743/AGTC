"""Host-only recorded-frame benchmark, one fixed profile; output is new evidence."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))
from fast_depth import FastSGBMDepth

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=pathlib.Path, required=True)
parser.add_argument("--opencv-bin", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Refusing to overwrite existing evidence")
capture = HERE.parent.parent / "stereo" / "captures" / "20240825T221623918473Z-946d97fc"
calibration = HERE.parent / "app" / "calibration.npz"
images = [cv2.imdecode(np.fromfile(capture / (side + ".png"), np.uint8), cv2.IMREAD_GRAYSCALE)
          for side in ("left", "right")]
if any(image is None for image in images):
    raise SystemExit("Recorded images unavailable")
images = [cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA) for image in images]
engine = FastSGBMDepth(calibration, threads=2)
def python_frame():
    left, right = engine.rectify(*images)
    disparity, _ = engine.disparities(left, right)
    depth, valid = engine.metric_values(disparity, None, None)
    return disparity, depth, valid
for _ in range(5):
    python_frame()
python_samples = []
for _ in range(100):
    begin = time.perf_counter_ns()
    expected_disparity, expected_depth, expected_valid = python_frame()
    python_samples.append(time.perf_counter_ns() - begin)
with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    yaml = root / "calibration.yml"
    subprocess.run([sys.executable, "-B", str(HERE / "export_calibration.py"), str(calibration), str(yaml)], check=True)
    for name, image in zip(("left", "right"), images):
        image.tofile(root / (name + ".raw"))
    prefix = root / "output"
    env = os.environ.copy()
    env["PATH"] = str(args.opencv_bin) + os.pathsep + env["PATH"]
    run = subprocess.run([str(args.fixture), str(yaml), str(root / "left.raw"), str(root / "right.raw"),
                          "640", "360", str(prefix), "100"], env=env, capture_output=True, text=True, check=True)
    native_version = run.stdout.splitlines()[1]
    if cv2.__version__ != native_version:
        raise SystemExit("Refusing mismatched OpenCV versions")
    native_disparity = np.fromfile(str(prefix) + ".disparity", np.float32).reshape(240, 131)
    native_depth = np.fromfile(str(prefix) + ".depth", np.float32).reshape(240, 131)
    native_valid = np.fromfile(str(prefix) + ".valid", np.uint8).reshape(240, 131).astype(bool)
    np.testing.assert_array_equal(native_disparity, expected_disparity)
    np.testing.assert_array_equal(native_valid, expected_valid)
    np.testing.assert_allclose(native_depth, expected_depth, rtol=1e-6, atol=1e-6, equal_nan=True)
    native_samples = [int(s) for s in (root / "output.timings_ns").read_text().splitlines()]
def summary(values):
    ordered = sorted(values)
    return {"median_ms": statistics.median(values) / 1e6, "p95_ms": ordered[94] / 1e6,
            "p99_ms": ordered[98] / 1e6, "samples_ns": values}
sources = [HERE / "stereo_sgbm.cpp", HERE / "depth_filter.cpp", HERE.parent / "app" / "fast_depth.py",
           calibration, capture / "left.png", capture / "right.png", args.fixture]
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
          "processor": platform.processor(), "opencv_python": cv2.__version__, "opencv_native": native_version,
          "numpy": np.__version__, "threads": 2, "warmup": 5, "sample_count": 100,
          "input_size": [640, 360], "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          "python": summary(python_samples), "native": summary(native_samples),
          "caveat": "sequential processes; Python filter calculates statistics, native does not; no temperature/frequency/power monitoring; no camera, HTTP, board or long-duration timing"}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print(json.dumps({"output": str(args.output),
                  "python": {k: v for k, v in report["python"].items() if k != "samples_ns"},
                  "native": {k: v for k, v in report["native"].items() if k != "samples_ns"},
                  "caveat": report["caveat"]}, indent=2))
