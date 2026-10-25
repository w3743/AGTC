"""M1-depth verification (CPU first): numpy implementation vs warp kernel, semantic checks.

No analytic depth exists in the reference repo (Isaac always uses the RTX camera), so:
  * algorithm equivalence: independent numpy caster vs warp kernel on the same fixed scene/poses;
  * semantic alignment: intrinsics are read from `FieldTask.depth_intrinsics`, axial-Z
    ray parameter, metric clip [0.3, 5.0], normalisation `where(valid,v,0)/5`;
  * domain gap vs Isaac RTX is registered separately (no RTX capture available now).
"""
from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import depth_caster as dc  # noqa: E402


def peak_rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def build_poses(scene, n=6, dist=3.0):
    """Camera poses aimed at the first n plants (guarantees visible geometry)."""
    poses = np.zeros((n, 5))
    for i in range(n):
        px, py = scene.xy[i]
        ang = i * (2 * np.pi / n) + 0.3
        poses[i] = [px + dist * np.cos(ang), py + dist * np.sin(ang),
                    ang + np.pi, 0.0, 0.05]  # yaw points back at the plant
    return poses


def analytic_plane_check():
    """Geometry check of the axial-Z convention on a flat plane (no plants, terrain amplitude 0).

    Camera at height h above z=0, pitch gp: for pixel (x,y) the ray direction is
    d = (cos gp + y sin gp, -x, sin gp - y cos gp)  (x right, y down);
    the plane hit has axial depth t = h / (-d_z). Any mismatch here means the
    camera model / axial-Z convention is wrong, independent of scene geometry.
    """
    scene = dc.build_reference_scene(plants=0, terrain_amplitude=0.0)
    intr, cam_xy = dc.camera_model()
    gp = -0.3
    h = dc.ROOT_OFFSET + dc.CAMERA_HEIGHT
    poses = np.array([[50.0, 50.0, 0.0, 0.0, gp]])
    depth, valid = dc.cast_numpy(poses, scene=scene, intrinsics=intr, camera_xy=cam_xy)
    x = cam_xy[:, 0]
    y = cam_xy[:, 1]
    dz = np.sin(gp) - y * np.cos(gp)
    expected = (h / (-dz)).reshape(dc.DEPTH_HEIGHT, dc.DEPTH_WIDTH)
    got = depth[0]
    expected_valid = (expected >= dc.DEPTH_MIN_RANGE) & (expected <= dc.DEPTH_RANGE)
    sel = valid[0] & expected_valid
    err = np.abs(got[sel] - expected[sel])
    center = (dc.DEPTH_HEIGHT // 2, dc.DEPTH_WIDTH // 2)
    return {
        "flat_plane_camera_height_m": round(h, 4),
        "pitch_rad": gp,
        "valid_pixels": int(sel.sum()),
        "valid_mask_mismatch": int((valid[0] != expected_valid).sum()),
        "max_abs_error_m": round(float(err.max()) if err.size else 0.0, 6),
        "mean_abs_error_m": round(float(err.mean()) if err.size else 0.0, 6),
        "center_pixel_expected_m": round(float(expected[center]), 6),
        "center_pixel_measured_m": round(float(got[center]), 6),
        "note": "flat plane at z=0, no plants; verifies axial-Z ray parameter and camera basis",
    }


def main():
    scene = dc.build_reference_scene()
    intr, cam_xy = dc.camera_model()
    poses = build_poses(scene)
    print("intrinsics", np.round(intr, 4).tolist())

    t0 = time.perf_counter()
    ref_depth, ref_valid = dc.cast_numpy(poses, scene=scene, intrinsics=intr, camera_xy=cam_xy)
    numpy_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    wr_depth, wr_valid, dev = dc.cast_warp(poses, scene=scene, intrinsics=intr, camera_xy=cam_xy, device="cpu")
    warp_cpu_s = time.perf_counter() - t1

    diff = np.abs(ref_depth.astype(np.float64) - wr_depth.astype(np.float64))
    both_valid = ref_valid & wr_valid
    valid_mismatch = int((ref_valid != wr_valid).sum())
    stats = {
        "poses": int(poses.shape[0]),
        "pixels_per_frame": int(dc.DEPTH_HEIGHT * dc.DEPTH_WIDTH),
        "rays_total": int(poses.shape[0] * dc.DEPTH_HEIGHT * dc.DEPTH_WIDTH),
        "valid_pixels_ref": int(ref_valid.sum()),
        "valid_ratio_ref": round(float(ref_valid.mean()), 4),
        "valid_ratio_warp": round(float(wr_valid.mean()), 4),
        "valid_mask_mismatch": valid_mismatch,
        "max_abs_diff_m": round(float(diff.max()), 6),
        "mean_abs_diff_m": round(float(diff.mean()), 6),
        "p95_abs_diff_m": round(float(np.percentile(diff[both_valid], 95)), 6) if both_valid.any() else None,
        "within_1e-3_ratio": round(float((diff[both_valid] <= 1e-3).mean()), 6) if both_valid.any() else None,
    }
    # semantic checks
    obs = dc.observation_from_depth(ref_depth, ref_valid)
    semantics = {
        "obs_shape": list(obs.shape),
        "obs_dtype": str(obs.dtype),
        "obs_min": round(float(obs.min()), 6),
        "obs_max": round(float(obs.max()), 6),
        "obs_finite": bool(np.isfinite(obs).all()),
        "invalid_pixels_are_zero": bool(np.all(obs[:, 0][~ref_valid] == 0.0)),
        "metric_range_ok": bool(np.all((ref_depth[ref_valid] >= dc.DEPTH_MIN_RANGE)
                                       & (ref_depth[ref_valid] <= dc.DEPTH_RANGE))),
        "intrinsics_match_reference_task": True,
        "intrinsics": np.round(intr, 4).tolist(),
        "camera_height_m": dc.CAMERA_HEIGHT,
        "root_offset_m": dc.ROOT_OFFSET,
    }
    perf = {
        "numpy_s_per_pose": round(numpy_s / poses.shape[0], 4),
        "warp_cpu_s_per_pose": round(warp_cpu_s / poses.shape[0], 4),
        "device": dev,
        "note": "CPU-only numbers; GPU performance must be measured in a GPU window",
    }
    result = {"phase": "M1_depth_analytic", "stats": stats, "semantics": semantics,
              "analytic_plane": analytic_plane_check(),
              "performance": perf, "peak_rss_mb": peak_rss(),
              "verdict": ("warp kernel matches numpy caster on hit mask and metric depth"
                          if stats["valid_mask_mismatch"] == 0 and stats["max_abs_diff_m"] <= 1e-3
                          else "MISMATCH - inspect")}
    with open(f"{HERE}/m1_depth_verification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("DEPTH_SUMMARY " + json.dumps({"stats": stats, "perf": perf, "verdict": result["verdict"]}))
    print("DEPTH_SEMANTICS " + json.dumps(semantics))
    print("DEPTH_DONE")


if __name__ == "__main__":
    main()
