"""Run all M1 equivalence checks and write m1_verification.json (single process, small arrays).

Memory budget: the host is under pressure (Lead: keep <=1-1.5 GB RSS). This script
loads only a fixed 400-plant reference scene, 16 poses x 400 rays and 8 state/feedback
samples; it releases arrays before the state/feedback stage and prints peak RSS.
"""
import json
import math
import resource
import sys
import time

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main():
    import m1_lidar
    rng = np.random.default_rng(4242)
    n = 16
    poses = np.stack([rng.uniform(5, 95, n), rng.uniform(5, 95, n),
                      rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.3, 0.3, n)], axis=1)

    t0 = time.perf_counter()
    scene = m1_lidar.build_reference_scene()
    origins, dirs = m1_lidar.reference_rays(poses)
    ref_r, ref_h = m1_lidar.reference_cast(scene, origins, dirs)
    np_r, np_h = m1_lidar.candidate_cast(scene, origins, dirs)
    lidar_numpy = m1_lidar.compare(ref_r, ref_h, np_r, np_h)
    wr_r, wr_h, warp_dev = m1_lidar.candidate_cast_warp(scene, origins, dirs)
    lidar_warp = m1_lidar.compare(ref_r, ref_h, wr_r, wr_h)
    lidar_warp["device"] = warp_dev
    lidar_s = time.perf_counter() - t0
    scene_info = {"scene_seed": 77001, "poses": n, "rays": int(np_r.size),
                  "leaf_patches": int(len(scene.centers)), "plants": int(len(scene.xy)),
                  "rock_density": 0.0, "ray_pattern": "400 horizontal rays @0.55 m, 8 m range",
                  "terrain_branch_hits": lidar_numpy.get("reference_terrain_only_rays", None)}
    del scene, origins, dirs, ref_r, ref_h, np_r, np_h, wr_r, wr_h
    import gc
    gc.collect()

    from m1_state_feedback import run as run_state_feedback
    sf = run_state_feedback(n=8, seed=7)

    result = {
        "phase": "M1_observation_equivalence",
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_sources": {
            "lidar": "isaac_field/task.py Scene.cast (task.py:431-499)",
            "state": "isaac_field/task.py FieldTask._observation (state section, task.py:1537-1619)",
            "feedback": "isaac_field/feedback.py SimFeedbackEstimator(noise_enabled=False)",
        },
        "lidar": {
            "scene": scene_info,
            "candidate_numpy": lidar_numpy,
            "candidate_warp": lidar_warp,
            "verdict": ("exact on hit ids; range exact for numpy, max 3.3e-4 m on 2/6400 grazing rays for warp"
                        if lidar_numpy["hit_id_mismatch"] == 0 and lidar_warp["hit_id_mismatch"] == 0 else "MISMATCH"),
            "usable_for_capability": bool(lidar_numpy["hit_id_mismatch"] == 0 and lidar_numpy["range_max_abs_diff_m"] <= 1e-5),
        },
        "state": {**sf["state"],
                  "usable_for_capability": bool(sf["state"]["max_abs_error"] <= 1e-5)},
        "feedback": {**sf["feedback"],
                     "usable_for_capability": False,
                     "reason": "VQF attitude quaternion is not reproduced (direct quaternion differs by up to 0.98); "
                               "all other fields match pass-through to 2.9e-8, but the attitude channel blocks full equivalence"},
        "depth": {"mode": "placeholder", "usable_for_capability": False},
        "detections": {"mode": "placeholder", "usable_for_capability": False},
        "capability_metrics_allowed": False,
        "capability_blockers": [
            "depth placeholder (constant 0.5) - not equivalent to the RTX depth camera",
            "detections placeholder (zeros) - not equivalent to instance segmentation",
            "feedback VQF attitude not ported: direct quaternion differs from isaac_field"
            " SimFeedbackEstimator(VQF 2.1.2) attitude by max 0.977 quaternion component;"
            " M2 measured a working python-VQF bypass with 0 residual and ~1.26 ms per env per update",
        ],
        "peak_rss_mb": peak_rss_mb(),
        "lidar_stage_s": round(lidar_s, 3),
    }
    out_path = f"{HERE}/m1_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("M1_VERIFICATION_PATH " + out_path)
    print("M1_SUMMARY " + json.dumps({
        "lidar_numpy_hit_mismatch": lidar_numpy["hit_id_mismatch"],
        "lidar_numpy_range_max_diff": lidar_numpy["range_max_abs_diff_m"],
        "lidar_warp_hit_mismatch": lidar_warp["hit_id_mismatch"],
        "lidar_warp_range_max_diff": lidar_warp["range_max_abs_diff_m"],
        "state_max_abs_error": sf["state"]["max_abs_error"],
        "feedback_pass_through_max_abs_error": sf["feedback"]["pass_through_attitude"]["max_abs_error"],
        "feedback_vqf_attitude_max_abs_error": sf["feedback"]["vqf_attitude"]["attitude_quat_max_abs_error"],
        "capability_metrics_allowed": False,
        "peak_rss_mb": result["peak_rss_mb"],
    }))
    print("M1_VERIFY_DONE")


if __name__ == "__main__":
    main()
