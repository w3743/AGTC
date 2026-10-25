"""Debug the warp lidar kernel vs numpy reference on a single known-hit ray."""
import math
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from m1_lidar import (RAY_COUNT, build_reference_scene, candidate_cast, reference_cast,
                      reference_rays)

rng = np.random.default_rng(4242)
n = 16
poses = np.stack([rng.uniform(5, 95, n), rng.uniform(5, 95, n),
                  rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.3, 0.3, n)], axis=1)
scene = build_reference_scene()
origins, dirs = reference_rays(poses)
ref_r, ref_h = reference_cast(scene, origins, dirs)
rs, cs = np.unravel_index(np.argmax(ref_h > -1), ref_h.shape)  # first hit
print("known hit env/ray", rs, cs, "ref_t", ref_r[rs, cs], "ref_hit", ref_h[rs, cs])
o = origins[rs]
d = dirs[rs * RAY_COUNT + cs]
print("origin", o, "dir", d)

# stage 1: a trivial kernel that just checks the cull works
import warp as wp

wp.init()
dev = "cpu"


@wp.kernel
def probe(origin: wp.array(dtype=wp.vec3), pxy: wp.array(dtype=wp.vec2),
          n_plants: int, reach: float, out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    o = origin[i]
    count = int(0)
    for p in range(n_plants):
        dx = o[0] - pxy[p][0]
        dy = o[1] - pxy[p][1]
        if dx * dx + dy * dy < reach * reach:
            count = count + 1
    out[i] = count


o_wp = wp.array(np.asarray([o], np.float32), dtype=wp.vec3, device=dev)
px_wp = wp.array(np.asarray(scene.xy, np.float32), dtype=wp.vec2, device=dev)
out = wp.array(np.zeros(1, np.int32), dtype=wp.int32, device=dev)
wp.launch(probe, dim=1, inputs=[o_wp, px_wp, int(len(scene.xy)), np.float32(8.5), out], device=dev)
print("cull count (warp):", int(out.numpy()[0]),
      "expected:", int((np.sum((np.asarray(scene.xy) - o[:2]) ** 2, axis=1) < 8.5 ** 2).sum()))
