"""M1 detections warp-kernel verification: identical mask/box/range/confidence vs numpy path.

CPU run (warp device='cpu'); GPU timing is measured separately in a GPU window.
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import detections_caster as dct  # noqa: E402
import depth_caster as dc  # noqa: E402


def peak_rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def build_poses(scene, n=6):
    poses = np.zeros((n, 6))
    terrain = scene.terrain
    for i in range(n):
        px, py = scene.xy[i]
        ang = i * (2 * np.pi / n) + 0.4
        x, y = px + 3.0 * np.cos(ang), py + 3.0 * np.sin(ang)
        z = float(terrain.surface_height(np.array([[x, y]]))[0]) + dc.ROOT_OFFSET + dc.CAMERA_HEIGHT
        poses[i] = [x, y, z, ang + np.pi, 0.0, 0.03]
    return poses


def main():
    scene = dc.build_reference_scene()
    poses = build_poses(scene, 8)
    center = dct.cast_center_distance(poses, scene=scene)

    t0 = time.perf_counter()
    masks_np = dct.cast_mask(poses, scene=scene)
    t_np = time.perf_counter() - t0
    t1 = time.perf_counter()
    masks_wp, dev = dct.cast_mask_warp(poses, scene=scene, device="cpu")
    t_wp = time.perf_counter() - t1
    t1b = time.perf_counter()
    masks_bt, devb = dct.cast_mask_warp_batch(poses, scene=scene, device="cpu")
    t_bt = time.perf_counter() - t1b

    mask_mismatch = int((masks_np != masks_wp).sum())
    mask_batch_mismatch = int((masks_np != masks_bt).sum())
    rows_np, dets_np = dct.observation_from_masks(masks_np, center, scene.leaf_dry, noise=False)
    rows_wp, dets_wp = dct.observation_from_masks(masks_wp, center, scene.leaf_dry, noise=False)
    rows_bt, dets_bt = dct.observation_from_masks(masks_bt, center, scene.leaf_dry, noise=False)
    diff = np.abs(rows_np.astype(np.float64) - rows_wp.astype(np.float64))
    diff_bt = np.abs(rows_np.astype(np.float64) - rows_bt.astype(np.float64))
    field_max = [round(float(diff[:, :, i].max()), 6) for i in range(8)]
    result = {
        "phase": "M1_detections_warp_kernel",
        "poses": int(poses.shape[0]),
        "pixels_per_frame": dct.DETECTOR_SIZE ** 2,
        "device": dev,
        "mask_mismatch_pixels": mask_mismatch,
        "mask_batch_mismatch_pixels": mask_batch_mismatch,
        "mask_total_pixels": int(masks_np.size),
        "leaf_hit_ratio_numpy": round(float((masks_np >= 0).mean()), 4),
        "active_rows_numpy": int(sum(len(d) for d in dets_np)),
        "active_rows_warp": int(sum(len(d) for d in dets_wp)),
        "active_rows_batch": int(sum(len(d) for d in dets_bt)),
        "rows_max_abs_error": round(float(diff.max()), 9),
        "rows_mean_abs_error": round(float(diff.mean()), 9),
        "batch_rows_max_abs_error": round(float(diff_bt.max()), 9),
        "per_field_max_abs_error": {
            "cx": field_max[0], "cy": field_max[1], "w": field_max[2], "h": field_max[3],
            "class_id": field_max[4], "confidence": field_max[5],
            "range_norm": field_max[6], "range_valid": field_max[7]},
        "timing": {
            "numpy_mask_s_per_frame": round(t_np / poses.shape[0], 5),
            "warp_cpu_mask_s_per_frame": round(t_wp / poses.shape[0], 5),
            "warp_batch_cpu_s_per_frame": round(t_bt / poses.shape[0], 5),
            "note": "warp CPU excludes a one-off compile in this run; GPU timing measured in a GPU window",
        },
        "peak_rss_mb": peak_rss(),
        "verdict": ("warp mask (per-pose and single-launch batch) equals the numpy mask and "
                    "detection rows are identical"
                    if mask_mismatch == 0 and mask_batch_mismatch == 0
                    and float(diff.max()) == 0.0 and float(diff_bt.max()) == 0.0 else "MISMATCH - inspect"),
    }
    with open(f"{HERE}/detections_warp_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DETECTIONS_WARP " + json.dumps({k: result[k] for k in
          ("poses", "device", "mask_mismatch_pixels", "mask_batch_mismatch_pixels",
           "leaf_hit_ratio_numpy", "active_rows_numpy", "active_rows_warp",
           "active_rows_batch", "rows_max_abs_error", "batch_rows_max_abs_error",
           "per_field_max_abs_error", "timing", "verdict")}))
    print("DETECTIONS_WARP_DONE")


if __name__ == "__main__":
    main()
