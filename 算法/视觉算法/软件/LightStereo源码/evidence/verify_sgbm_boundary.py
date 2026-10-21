import json
import sys
import warnings

sys.path.insert(0, "/opt/rk-lightstereo/runtime/vendor")
sys.path.insert(0, "/opt/rk-lightstereo/app")

import cv2
import numpy as np
from fast_depth import FastSGBMDepth

engine = FastSGBMDepth(
    "/opt/rk-lightstereo/app/calibration.npz",
    scale=0.375,
    num_disparities=64,
    threads=2,
    input_size=(640, 360),
)
left = cv2.imread("/opt/rk-lightstereo/validation/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("/opt/rk-lightstereo/validation/right.png", cv2.IMREAD_GRAYSCALE)
assert left is not None and right is not None, "saved stereo pair missing"
_, depth, _ = engine.compute(left, right)
valid = np.isfinite(depth)
stats = {
    "valid_fraction": float(valid.mean()),
    "left_x_0_63": float(valid[:, :64].mean()),
    "right_x_64_130": float(valid[:, 64:].mean()),
}
assert stats["valid_fraction"] > 0.5 and stats["left_x_0_63"] > 0.3, stats

synthetic = []
rng = np.random.default_rng(3588)
for shift in (8, 24):
    left = rng.integers(0, 256, (engine.height, engine.width), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    right[:, :-shift] = left[:, shift:]
    disparity, _ = engine.disparities(left, right)
    measured, mask = engine.metric_values(disparity, None, left)
    core = np.zeros_like(mask)
    core[8:-8, shift + 8:-8] = True
    expected = engine.depth_factor / (shift - engine.disparity_offset)
    median = float(np.nanmedian(measured[core]))
    coverage = float(mask[core].mean())
    assert coverage > 0.8, (shift, coverage)
    assert abs(median - expected) < 0.01, (shift, median, expected)
    assert not mask[:, :shift + 2].any(), f"accepted out-of-bounds shift {shift}"
    synthetic.append({
        "shift_px": shift,
        "expected_depth_m": expected,
        "median_depth_m": median,
        "core_valid_fraction": coverage,
    })

with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    for value in (float("nan"), float("inf"), -float("inf"), -1.0, 0.0):
        _, mask = engine.metric_values(
            np.full((engine.height, engine.width), value, dtype=np.float32), None, None
        )
        assert not mask.any(), f"accepted invalid disparity {value}"

print(json.dumps({
    "ok": True,
    "resolution": [engine.width, engine.height],
    "num_disparities": engine.num_disparities,
    "saved_pair": stats,
    "synthetic": synthetic,
    "invalid_disparities_rejected": True,
}, separators=(",", ":")))