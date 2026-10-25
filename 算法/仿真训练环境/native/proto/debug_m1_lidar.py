"""Debug the single mismatching ray in M1.1 lidar comparison."""
import math
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from m1_lidar import (LIDAR_RANGE, RAY_COUNT, build_reference_scene, candidate_cast,
                      reference_cast, reference_rays)

rng = np.random.default_rng(4242)
n = 16
poses = np.stack([rng.uniform(5, 95, n), rng.uniform(5, 95, n),
                  rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.3, 0.3, n)], axis=1)
scene = build_reference_scene()
origins, dirs = reference_rays(poses)
ref_r, ref_h = reference_cast(scene, origins, dirs)
cand_r, cand_h = candidate_cast(scene, origins, dirs)
diff = np.abs(ref_r.astype(np.float64) - cand_r.astype(np.float64))
e, r = np.unravel_index(np.argmax(diff), diff.shape)
print("worst env", e, "ray", r, "ref", ref_r[e, r], "hit", ref_h[e, r],
      "cand", cand_r[e, r], "hit", cand_h[e, r], "diff", diff[e, r])
print("origin", origins[e], "dir", dirs[e * RAY_COUNT + r])
print("all mismatch rays:", np.argwhere(ref_h != cand_h).tolist())
for env, ray in np.argwhere(diff > 1e-3):
    print("env", env, "ray", ray, "ref", ref_r[env, ray], ref_h[env, ray],
          "cand", cand_r[env, ray], cand_h[env, ray])
