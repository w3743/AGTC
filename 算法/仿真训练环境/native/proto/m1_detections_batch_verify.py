"""Batch-vs-per-pose detections pipeline verification (CPU): centre ray + assembly.

Compares batched `cast_center_distance_batch` / `assemble_batch` against the original
per-pose implementations on the same 64-pose batch, and times both.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import detections_caster as dct  # noqa: E402
import depth_caster as dc  # noqa: E402
from m1_detections_warp_verify import build_poses  # noqa: E402


def main():
    scene = dc.build_reference_scene()
    poses = build_poses(scene, 64)

    t0 = time.perf_counter()
    masks_np = dct.cast_mask(poses, scene=scene)
    t_mask = time.perf_counter() - t0
    t0 = time.perf_counter()
    center_loop = dct.cast_center_distance(poses, scene=scene)
    t_center_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    center_batch = dct.cast_center_distance_batch(poses, scene=scene)
    t_center_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows_loop, _d1 = dct.observation_from_masks(masks_np, center_loop, scene.leaf_dry, noise=False)
    t_asm_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows_batch, _d2 = dct.observation_from_masks_batch(masks_np, center_batch, scene.leaf_dry, noise=False)
    t_asm_batch = time.perf_counter() - t0

    center_diff = np.abs(center_loop.astype(np.float64) - center_batch.astype(np.float64))
    center_finite = np.isfinite(center_diff)
    rows_diff = np.abs(rows_loop.astype(np.float64) - rows_batch.astype(np.float64))
    result = {
        "phase": "M2_detections_batch_pipeline",
        "poses": int(poses.shape[0]),
        "center_distance": {
            "max_abs_diff_all": round(float(np.nanmax(center_diff)), 9),
            "max_abs_diff_finite": round(float(center_diff[center_finite].max()) if center_finite.any() else 0.0, 9),
            "inf_mismatch": int((np.isinf(center_loop) != np.isinf(center_batch)).sum()),
            "per_pose_s_loop": round(t_center_loop / poses.shape[0], 6),
            "per_pose_s_batch": round(t_center_batch / poses.shape[0], 6),
        },
        "assembly": {
            "max_abs_diff": round(float(rows_diff.max()), 9),
            "mean_abs_diff": round(float(rows_diff.mean()), 9),
            "per_pose_s_loop": round(t_asm_loop / poses.shape[0], 6),
            "per_pose_s_batch": round(t_asm_batch / poses.shape[0], 6),
        },
        "total_pipeline": {
            "old_s_per_batch": round(t_mask + t_center_loop + t_asm_loop, 4),
            "batch_s_per_batch": round(t_mask + t_center_batch + t_asm_batch, 4),
            "numpy_mask_s_per_batch": round(t_mask, 4),
            "note": "CPU numpy path; the warp mask is measured on GPU separately",
        },
        "verdict": ("batched centre-ray and assembly match the per-pose implementations"
                    if result_ok(center_diff, rows_diff, center_loop, center_batch)
                    else "MISMATCH - inspect"),
    }
    with open(f"{HERE}/detections_batch_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DETECTIONS_BATCH " + json.dumps(result))


def result_ok(center_diff, rows_diff, center_loop, center_batch):
    inf_mismatch = int((np.isinf(center_loop) != np.isinf(center_batch)).sum())
    finite = np.isfinite(center_diff)
    max_center = float(center_diff[finite].max()) if finite.any() else 0.0
    return inf_mismatch == 0 and max_center <= 1e-3 and float(rows_diff.max()) <= 1e-7


if __name__ == "__main__":
    main()
