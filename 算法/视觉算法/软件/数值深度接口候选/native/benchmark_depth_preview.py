"""Host-only fixed-colormap/JPEG microbenchmark on synthetic depth values."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=pathlib.Path, required=True)
parser.add_argument("--opencv-bin", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Refusing to overwrite existing evidence")
cv2.setNumThreads(2)
indexes = np.arange(131 * 240, dtype=np.uint32).reshape(240, 131)
mm = np.where(indexes % 7 == 0, 0, 300 + indexes % 4701).astype(np.uint16)
depth = mm.astype(np.float32) / 1000
def python_frame():
    valid = mm != 0
    shade = np.zeros(depth.shape, np.uint8)
    shade[valid] = np.clip((5.0 - depth[valid]) / 4.7 * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    ok, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok: raise RuntimeError("Python JPEG encoding failed")
    return encoded
for _ in range(50): python_frame()
baseline = []
for _ in range(500):
    start = time.perf_counter_ns(); python_frame(); baseline.append(time.perf_counter_ns() - start)
env = os.environ.copy()
env["PATH"] = str(args.opencv_bin) + os.pathsep + env["PATH"]
completed = subprocess.run([str(args.fixture)], env=env, capture_output=True, text=True, check=True)
samples = {"fresh": [], "cached": []}
for line in completed.stdout.splitlines():
    kind, value = line.split()
    samples[kind].append(int(value))
if any(len(values) != 500 for values in samples.values()):
    raise SystemExit("Fixture did not emit all samples")
def summary(values):
    ordered = sorted(values)
    return {"median_us": statistics.median(values) / 1000, "p95_us": ordered[474] / 1000,
            "p99_us": ordered[494] / 1000, "samples_ns": values}
sources = [HERE / "depth_preview.cpp", HERE.parent / "app" / "depth.py", args.fixture]
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
          "processor": platform.processor(), "opencv_python": cv2.__version__, "numpy": np.__version__,
          "input": "synthetic 240x131 uint16 mm; every seventh pixel invalid", "quality": 85,
          "warmup": 50, "sample_count": 500, "threads": 2,
          "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
          "python_fresh": summary(baseline), "native_fresh": summary(samples["fresh"]),
          "native_cached": summary(samples["cached"]),
          "caveat": "sequential processes; OpenCV builds and JPEG libraries may differ; fixed scale from quantized data; no temperature/frequency/power, camera, HTTP or board timing"}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2); stream.write("\n")
print(json.dumps({key: {k: v for k, v in report[key].items() if k != "samples_ns"}
                  for key in ("python_fresh", "native_fresh", "native_cached")}, indent=2))
