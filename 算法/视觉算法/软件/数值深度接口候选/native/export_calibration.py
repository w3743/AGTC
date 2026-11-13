"""Offline NPZ-to-OpenCV YAML conversion; never runs on the vehicle."""
import argparse
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("Refusing to overwrite existing output")
raw = args.source.read_bytes()
with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
    model = {key: saved[key].copy() for key in saved.files}
if str(model.get("status")) != "geometry_pass_metric_accuracy_unverified":
    raise SystemExit("Calibration geometry status is not accepted")
if tuple(model["image_size"]) != (3840, 2160) or "mm" not in str(model.get("coordinate_convention", "")):
    raise SystemExit("Unexpected image dimensions or units")
shapes = {"K1": (3, 3), "K2": (3, 3), "R1": (3, 3), "R2": (3, 3),
          "P1": (3, 4), "P2": (3, 4), "T": (3, 1)}
for key, shape in shapes.items():
    if model[key].shape != shape or not np.isfinite(model[key]).all():
        raise SystemExit("Invalid calibration matrix: " + key)
for key in ("D1", "D2"):
    if model[key].size not in (4, 5, 8, 12, 14) or not np.isfinite(model[key]).all():
        raise SystemExit("Invalid distortion coefficients: " + key)
if abs(model["P2"][1, 3]) <= abs(model["P2"][0, 3]) or model["P2"][1, 3] >= 0:
    raise SystemExit("Calibration does not provide the required vertical baseline")
fs = cv2.FileStorage("", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML)
if not fs.isOpened():
    raise SystemExit("Could not create output")
fs.write("schema", "agtc-stereo-calibration-v1")
fs.write("source_sha256", hashlib.sha256(raw).hexdigest())
fs.write("status", str(model["status"]))
fs.write("image_width", 3840)
fs.write("image_height", 2160)
fs.write("unit", "mm")
for key in (*shapes, "D1", "D2"):
    fs.write(key, np.asarray(model[key], dtype=np.float64))
serialized = fs.releaseAndGetString()
with args.output.open("x", encoding="utf-8", newline="\n") as output:
    output.write(serialized)
