"""Host-only JPEG comparison using synthetic uniform left/right images."""
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
if args.output.exists(): raise SystemExit("Refusing to overwrite existing evidence")
cv2.setNumThreads(2)
left, right = np.full((360, 640), 42, np.uint8), np.full((360, 640), 99, np.uint8)
def python_frame():
    ok, jpeg = cv2.imencode(".jpg", np.hstack((left, right)), [cv2.IMWRITE_JPEG_QUALITY, 55])
    if not ok: raise RuntimeError("JPEG encoding failed")
    return jpeg
for _ in range(25): python_frame()
baseline = []
for _ in range(250):
    started = time.perf_counter_ns(); python_frame(); baseline.append(time.perf_counter_ns() - started)
env = os.environ.copy(); env["PATH"] = str(args.opencv_bin) + os.pathsep + env["PATH"]
completed = subprocess.run([str(args.fixture)], env=env, capture_output=True, text=True, check=True)
samples = {"fresh": [], "cached": [], "concat": [], "jpeg": []}
for line in completed.stdout.splitlines():
    label, elapsed = line.split(); samples[label].append(int(elapsed))
if any(len(values) != 250 for values in samples.values()): raise RuntimeError("Incomplete fixture samples")
def summary(values):
    ordered = sorted(values)
    return {"median_us": statistics.median(values) / 1000, "p95_us": ordered[237] / 1000,
            "p99_us": ordered[247] / 1000, "samples_ns": values}
sources = [HERE / "stereo_preview.cpp", HERE.parent / "app" / "server.py", args.fixture]
report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
          "processor": platform.processor(), "opencv_python": cv2.__version__, "numpy": np.__version__,
          "input": "synthetic uniform 640x360 gray images", "jpeg_quality": 55, "warmup": 25,
          "sample_count": 250, "threads": 2,
          "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
          "python_fresh": summary(baseline), "native_fresh": summary(samples["fresh"]),
          "native_cached": summary(samples["cached"]), "native_concat_only": summary(samples["concat"]),
          "native_jpeg_only": summary(samples["jpeg"]),
          "caveat": "sequential processes, different OpenCV/JPEG builds; excludes capture, frame copy and network; no thermal, power or board timing; uniform images are not representative of camera content"}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2); stream.write("\n")
print(json.dumps({key: {k: v for k, v in report[key].items() if k != "samples_ns"}
                  for key in ("python_fresh", "native_fresh", "native_cached", "native_concat_only", "native_jpeg_only")}, indent=2))
