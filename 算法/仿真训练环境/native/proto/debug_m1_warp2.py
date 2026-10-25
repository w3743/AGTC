"""Staged debug of the warp lidar kernel conditions."""
import math
import sys

import numpy as np

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from m1_lidar import (LIDAR_RANGE, RAY_COUNT, build_reference_scene, reference_cast,
                      reference_rays)

import warp as wp

wp.init()
dev = "cpu"
rng = np.random.default_rng(4242)
n = 16
poses = np.stack([rng.uniform(5, 95, n), rng.uniform(5, 95, n),
                  rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.3, 0.3, n)], axis=1)
scene = build_reference_scene()
origins, dirs = reference_rays(poses)
ref_r, ref_h = reference_cast(scene, origins, dirs)
rs, cs = np.unravel_index(np.argmax(ref_h > -1), ref_h.shape)
o = np.asarray(origins[rs], np.float32)
d = np.asarray(dirs[rs * RAY_COUNT + cs], np.float32)
print("known hit", ref_r[rs, cs], ref_h[rs, cs])

centers = np.asarray(scene.centers, np.float32)
normals = np.asarray(scene.normals, np.float32)
uu = np.asarray(scene.u, np.float32)
vv = np.asarray(scene.v, np.float32)
axes = np.asarray(scene.axes, np.float32)
pxy = np.asarray(scene.xy, np.float32)
stem_r = np.asarray(scene.stem_radius, np.float32)
root = np.asarray(scene.root_z, np.float32)
hgt = np.asarray(scene.height, np.float32)
plants = pxy.shape[0]
leaves = centers.shape[0] // plants


@wp.kernel
def staged(origin: wp.array(dtype=wp.vec3), direction: wp.array(dtype=wp.vec3),
           centers_: wp.array(dtype=wp.vec3), normals_: wp.array(dtype=wp.vec3),
           uu_: wp.array(dtype=wp.vec3), vv_: wp.array(dtype=wp.vec3),
           axes_: wp.array(dtype=wp.vec2), pxy_: wp.array(dtype=wp.vec2),
           stem_r_: wp.array(dtype=wp.float32), root_: wp.array(dtype=wp.float32),
           hgt_: wp.array(dtype=wp.float32), n_plants: int, n_leaves: int,
           max_range: float, counts: wp.array(dtype=wp.int32), out_t: wp.array(dtype=wp.float32),
           out_hit: wp.array(dtype=wp.int32)):
    i = wp.tid()
    o_ = origin[i]
    d_ = direction[i]
    reach = max_range + 0.5
    den_ok = int(0)
    t_ok = int(0)
    inside_ok = int(0)
    best = max_range
    hit = int(-1)
    for p in range(n_plants):
        dx0 = o_[0] - pxy_[p][0]
        dy0 = o_[1] - pxy_[p][1]
        if dx0 * dx0 + dy0 * dy0 > reach * reach:
            continue
        for k in range(n_leaves):
            idx = p * n_leaves + k
            c = centers_[idx]
            nrm = normals_[idx]
            den = wp.dot(d_, nrm)
            if wp.abs(den) > 1.0e-7:
                den_ok = den_ok + 1
                t = wp.dot(c - o_, nrm) / den
                if t > 1.0e-4 and t <= max_range:
                    t_ok = t_ok + 1
                    u = uu_[idx]
                    v = vv_[idx]
                    pu = wp.dot(o_ - c, u) + t * wp.dot(d_, u)
                    pv = wp.dot(o_ - c, v) + t * wp.dot(d_, v)
                    sx = pu / axes_[idx][0]
                    sy = pv / axes_[idx][1]
                    taper = (1.0 - sx * sx) * (1.0 - wp.abs(sx))
                    if wp.abs(sx) <= 1.0 and sy * sy <= taper:
                        inside_ok = inside_ok + 1
                        if t < best:
                            best = t
                            hit = idx
        qx = dx0
        qy = dy0
        a = d_[0] * d_[0] + d_[1] * d_[1]
        if a > 1.0e-8:
            b = 2.0 * (d_[0] * qx + d_[1] * qy)
            cc = qx * qx + qy * qy - stem_r_[p] * stem_r_[p]
            disc = b * b - 4.0 * a * cc
            if disc >= 0.0:
                t = (-b - wp.sqrt(disc)) / (2.0 * a)
                z = o_[2] + t * d_[2]
                if t > 1.0e-4 and t < best and z >= root_[p] and z <= root_[p] + hgt_[p]:
                    best = t
                    hit = -2 - p
    counts[0] = den_ok
    counts[1] = t_ok
    counts[2] = inside_ok
    out_t[i] = best
    out_hit[i] = hit


o_wp = wp.array(np.asarray([o]), dtype=wp.vec3, device=dev)
d_wp = wp.array(np.asarray([d]), dtype=wp.vec3, device=dev)
c_wp = wp.array(centers.reshape(-1, 3), dtype=wp.vec3, device=dev)
n_wp = wp.array(normals.reshape(-1, 3), dtype=wp.vec3, device=dev)
u_wp = wp.array(uu.reshape(-1, 3), dtype=wp.vec3, device=dev)
v_wp = wp.array(vv.reshape(-1, 3), dtype=wp.vec3, device=dev)
ax_wp = wp.array(axes.reshape(-1, 2), dtype=wp.vec2, device=dev)
px_wp = wp.array(pxy.reshape(-1, 2), dtype=wp.vec2, device=dev)
sr_wp = wp.array(stem_r.reshape(-1), dtype=wp.float32, device=dev)
rz_wp = wp.array(root.reshape(-1), dtype=wp.float32, device=dev)
hg_wp = wp.array(hgt.reshape(-1), dtype=wp.float32, device=dev)
counts = wp.array(np.zeros(3, np.int32), dtype=wp.int32, device=dev)
out_t = wp.array(np.zeros(1, np.float32), dtype=wp.float32, device=dev)
out_h = wp.array(np.zeros(1, np.int32), dtype=wp.int32, device=dev)
wp.launch(staged, dim=1,
          inputs=[o_wp, d_wp, c_wp, n_wp, u_wp, v_wp, ax_wp, px_wp, sr_wp, rz_wp, hg_wp,
                  int(plants), int(leaves), np.float32(LIDAR_RANGE), counts, out_t, out_h],
          device=dev)
wp.synchronize()
print("warp counts den_ok/t_ok/inside_ok", counts.numpy().tolist(),
      "t", float(out_t.numpy()[0]), "hit", int(out_h.numpy()[0]))

# numpy side for the same ray
o64 = o.astype(np.float64)
d64 = d.astype(np.float64)
den = normals.astype(np.float64) @ d64
num = np.sum((centers - o64) * normals, axis=1)
t = num / np.where(np.abs(den) > 1e-7, den, np.inf)
print("numpy den_ok", int((np.abs(den) > 1e-7).sum()),
      "t_ok", int(((t > 1e-4) & (t <= LIDAR_RANGE)).sum()))
