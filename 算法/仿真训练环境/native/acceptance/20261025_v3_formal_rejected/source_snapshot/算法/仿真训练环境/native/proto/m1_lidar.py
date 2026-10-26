"""M1.1: lidar equivalence against isaac_field's reference analytic raycast.

Reference implementation: `isaac_field.task.Scene.cast` (task.py:431-499)
  * terrain heightfield (Terrain.cast) + leaf patches (ellipse on plane) +
    stem cylinders + rocks.
Candidate implementation (below): a standalone numpy vectorisation of the same
  leaf/stem math, driven by the SAME Scene arrays (centers/normals/u/v/axes,
  xy/stem_radius/root_z/height). Terrain is not intersected because the
  400-beam LiDAR is horizontal at 0.55 m while terrain amplitude is 0.35 m; the
  harness verifies that the reference's terrain branch never contributes a hit
  for this ray set (it reports how many rays hit terrain).

The comparison uses a fixed scene (seed), fixed sensor poses and the exact ray
pattern used by the prototype (400 horizontal rays, 8 m range).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

LIDAR_RANGE = 8.0
LIDAR_MIN_RANGE = 0.12
RAY_COUNT = 400


def build_reference_scene(*, seed: int = 77001, plants: int = 400, size: float = 100.0,
                          terrain_amplitude: float = 0.35, rock_density: float = 0.0):
    """Fixed scene built with the repository's own reference classes (read-only import)."""
    from isaac_field.task import Config, Scene
    from isaac_field.terrain import Terrain

    cfg = Config(size=size, plants=plants, dry_fraction=0.25, leaves_per_plant=8,
                 terrain_enabled=True, terrain_amplitude=terrain_amplitude,
                 terrain_wavelength_min=4.0, terrain_wavelength_max=10.0,
                 rock_density=rock_density, large_rock_fraction=0.2,
                 terrain_seed=seed)
    rng = np.random.default_rng(seed)
    terrain = Terrain(cfg.size, seed=seed, ground_amplitude=cfg.terrain_amplitude,
                      wavelength_range=(cfg.terrain_wavelength_min, cfg.terrain_wavelength_max),
                      rock_density=cfg.rock_density, large_rock_fraction=cfg.large_rock_fraction)
    xy = rng.uniform(5.0, size - 5.0, (plants, 2)).astype(np.float32)
    dry = (rng.random(plants) < cfg.dry_fraction)
    scene = Scene(cfg, xy, dry, np.random.default_rng(seed + 2), terrain=terrain)
    return scene


def reference_rays(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """poses: (n, 4) = x, y, yaw, gimbal_yaw. Returns origins (n,3), dirs (n*400,3) world frame."""
    az = np.linspace(-math.pi, math.pi, RAY_COUNT, endpoint=False).astype(np.float64)
    n = poses.shape[0]
    ang = (poses[:, 2] + poses[:, 3])[:, None]
    dirs = np.stack([np.cos(az[None, :] + ang), np.sin(az[None, :] + ang),
                     np.zeros_like(ang + az[None, :])], axis=-1).reshape(-1, 3)
    origins = np.zeros((n, 3))
    origins[:, 0] = poses[:, 0]
    origins[:, 1] = poses[:, 1]
    origins[:, 2] = 0.15 + 0.40
    return origins, dirs


def reference_cast(scene, origins: np.ndarray, dirs: np.ndarray):
    """Per-pose reference cast; returns ranges (n,400) and hit codes (n,400)."""
    n = origins.shape[0]
    ranges = np.zeros((n, RAY_COUNT), dtype=np.float32)
    hits = np.zeros((n, RAY_COUNT), dtype=np.int64)
    for e in range(n):
        out, hit = scene.cast(origins[e], dirs[e * RAY_COUNT:(e + 1) * RAY_COUNT], LIDAR_RANGE)
        ranges[e] = out
        hits[e] = hit
    return ranges, hits


def candidate_cast(scene, origins: np.ndarray, dirs: np.ndarray):
    """Standalone numpy implementation of the same leaf/stem math (no terrain).

    NOTE: the reference casts origin/directions to float32 and keeps scene arrays
    float32; the candidate uses the same dtypes, otherwise grazing rays (disc ~ 0)
    flip between hit and miss.
    """
    n = origins.shape[0]
    ranges = np.full((n, RAY_COUNT), np.float32(LIDAR_RANGE), dtype=np.float32)
    hits = np.full((n, RAY_COUNT), -1, dtype=np.int64)

    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    plant_xy = np.asarray(scene.xy, np.float32)
    stem_radius = np.asarray(scene.stem_radius, np.float32)
    root_z = np.asarray(scene.root_z, np.float32)
    height = np.asarray(scene.height, np.float32)

    for e in range(n):
        o = np.asarray(origins[e], np.float32)
        d = np.asarray(dirs[e * RAY_COUNT:(e + 1) * RAY_COUNT], np.float32).reshape(-1, 3)
        best = np.full(len(d), np.float32(LIDAR_RANGE), dtype=np.float32)
        hit = np.full(len(d), -1, dtype=np.int64)
        # ---- leaf patches (cull by leaf centre distance, same as reference) ----
        leaf_padding = max(0.5, float(leaf_axes.max(initial=0.0)))
        near = np.sum((leaf_centers[:, :2] - o[:2]) ** 2, axis=1) < (LIDAR_RANGE + leaf_padding) ** 2
        ids = np.flatnonzero(near)
        if len(ids):
            c, nrm, u, v, axes = leaf_centers[ids], leaf_normals[ids], leaf_u[ids], leaf_v[ids], leaf_axes[ids]
            den = d @ nrm.T
            num = np.sum((c - o) * nrm, axis=1)
            t = np.divide(num[None, :], den, out=np.full_like(den, np.inf), where=np.abs(den) > 1e-7)
            pu = np.sum((o - c) * u, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d @ u.T)
            pv = np.sum((o - c) * v, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d @ v.T)
            shape_x, shape_y = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - shape_x ** 2) * (1.0 - np.abs(shape_x))
            inside = (np.abs(shape_x) <= 1.0) & (shape_y ** 2 <= taper)
            ts = np.where(inside & (t > 1e-4) & (t <= LIDAR_RANGE), t, np.inf)
            j = np.argmin(ts, axis=1)
            cand = ts[np.arange(len(d)), j]
            take = cand < best
            best[take] = cand[take]
            hit[take] = ids[j[take]]
        # ---- stems (cylinders) ----
        stems = np.flatnonzero(np.sum((plant_xy - o[:2]) ** 2, axis=1) < (LIDAR_RANGE + 0.2) ** 2)
        if len(stems):
            q = o[:2] - plant_xy[stems]
            a = np.sum(d[:, :2] ** 2, axis=1)[:, None]
            b = 2 * (d[:, :2] @ q.T)
            cterm = np.sum(q * q, axis=1)[None, :] - stem_radius[stems] ** 2
            disc = b * b - 4 * a * cterm
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.divide(-b - np.sqrt(np.maximum(disc, 0)), 2 * a,
                              out=np.full_like(b, np.inf), where=a > 1e-8)
            z = o[2] + np.where(np.isfinite(t), t, 0) * d[:, 2, None]
            t = np.where((disc >= 0) & (t > 1e-4) & (z >= root_z[stems])
                         & (z <= root_z[stems] + height[stems]), t, np.inf)
            j = np.argmin(t, axis=1)
            cand = t[np.arange(len(d)), j]
            take = cand < best
            best[take] = cand[take]
            hit[take] = -2 - stems[j[take]]
        ranges[e] = best
        hits[e] = hit
    return ranges, hits


def candidate_cast_warp(scene, origins: np.ndarray, dirs: np.ndarray, device: str | None = None):
    """Warp-kernel version of the same leaf/stem math (CUDA when available, CPU fallback)."""
    import warp as wp

    wp.init()
    dev = device or ("cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")
    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    plant_xy = np.asarray(scene.xy, np.float32)
    stem_radius = np.asarray(scene.stem_radius, np.float32)
    root_z = np.asarray(scene.root_z, np.float32)
    height = np.asarray(scene.height, np.float32)
    plants = plant_xy.shape[0]
    leaves = leaf_centers.shape[0] // plants

    @wp.kernel
    def cast_kernel(origin: wp.array(dtype=wp.vec3), direction: wp.array(dtype=wp.vec3),
                    centers: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                    uu: wp.array(dtype=wp.vec3), vv: wp.array(dtype=wp.vec3),
                    axes: wp.array(dtype=wp.vec2), pxy: wp.array(dtype=wp.vec2),
                    stem_r: wp.array(dtype=wp.float32), root: wp.array(dtype=wp.float32),
                    hgt: wp.array(dtype=wp.float32), n_plants: int, n_leaves: int,
                    max_range: float, out_t: wp.array(dtype=wp.float32),
                    out_hit: wp.array(dtype=wp.int32)):
        i = wp.tid()
        o = origin[i]
        d = direction[i]
        best = max_range
        hit = int(-1)
        reach = max_range + 0.5
        for p in range(n_plants):
            px = pxy[p][0]
            py = pxy[p][1]
            dx0 = o[0] - px
            dy0 = o[1] - py
            if dx0 * dx0 + dy0 * dy0 > reach * reach:
                continue
            for k in range(n_leaves):
                idx = p * n_leaves + k
                c = centers[idx]
                nrm = normals[idx]
                den = wp.dot(d, nrm)
                if wp.abs(den) > 1.0e-7:
                    t = wp.dot(c - o, nrm) / den
                    if t > 1.0e-4 and t <= max_range:
                        u = uu[idx]
                        v = vv[idx]
                        pu = wp.dot(o - c, u) + t * wp.dot(d, u)
                        pv = wp.dot(o - c, v) + t * wp.dot(d, v)
                        sx = pu / axes[idx][0]
                        sy = pv / axes[idx][1]
                        taper = (1.0 - sx * sx) * (1.0 - wp.abs(sx))
                        if wp.abs(sx) <= 1.0 and sy * sy <= taper and t < best:
                            best = t
                            hit = idx
            qx = dx0
            qy = dy0
            a = d[0] * d[0] + d[1] * d[1]
            if a > 1.0e-8:
                b = 2.0 * (d[0] * qx + d[1] * qy)
                cc = qx * qx + qy * qy - stem_r[p] * stem_r[p]
                disc = b * b - 4.0 * a * cc
                if disc >= 0.0:
                    t = (-b - wp.sqrt(disc)) / (2.0 * a)
                    z = o[2] + t * d[2]
                    if t > 1.0e-4 and t < best and z >= root[p] and z <= root[p] + hgt[p]:
                        best = t
                        hit = -2 - p
        out_t[i] = best
        out_hit[i] = hit

    n = origins.shape[0]
    # one origin per ray (kernel indexes origin[i] for i in [0, n*RAY_COUNT)); an
    # earlier version passed only n origins and read out of bounds, which silently
    # produced "no hit" for every ray.
    o_all = np.repeat(np.asarray(origins, np.float32), RAY_COUNT, axis=0).reshape(-1, 3)
    d_all = np.asarray(dirs, np.float32).reshape(-1, 3)
    o_wp = wp.array(o_all, dtype=wp.vec3, device=dev)
    d_wp = wp.array(d_all, dtype=wp.vec3, device=dev)
    c_wp = wp.array(leaf_centers.reshape(-1, 3), dtype=wp.vec3, device=dev)
    n_wp = wp.array(leaf_normals.reshape(-1, 3), dtype=wp.vec3, device=dev)
    u_wp = wp.array(leaf_u.reshape(-1, 3), dtype=wp.vec3, device=dev)
    v_wp = wp.array(leaf_v.reshape(-1, 3), dtype=wp.vec3, device=dev)
    ax_wp = wp.array(leaf_axes.reshape(-1, 2), dtype=wp.vec2, device=dev)
    px_wp = wp.array(plant_xy.reshape(-1, 2), dtype=wp.vec2, device=dev)
    sr_wp = wp.array(stem_radius.reshape(-1), dtype=wp.float32, device=dev)
    rz_wp = wp.array(root_z.reshape(-1), dtype=wp.float32, device=dev)
    hg_wp = wp.array(height.reshape(-1), dtype=wp.float32, device=dev)
    out_t = wp.array(np.zeros(n * RAY_COUNT, np.float32), dtype=wp.float32, device=dev)
    out_h = wp.array(np.zeros(n * RAY_COUNT, np.int32), dtype=wp.int32, device=dev)
    wp.launch(cast_kernel, dim=n * RAY_COUNT,
              inputs=[o_wp, d_wp, c_wp, n_wp, u_wp, v_wp, ax_wp, px_wp, sr_wp, rz_wp, hg_wp,
                      int(plants), int(leaves), np.float32(LIDAR_RANGE), out_t, out_h],
              device=dev)
    wp.synchronize()
    return (out_t.numpy().reshape(n, RAY_COUNT).astype(np.float32),
            out_h.numpy().reshape(n, RAY_COUNT).astype(np.int64), str(dev))


def compare(ranges_ref, hits_ref, ranges_cand, hits_cand, *, range_tol=1e-5):
    diff = np.abs(np.asarray(ranges_ref, np.float32).astype(np.float64)
                  - np.asarray(ranges_cand, np.float32).astype(np.float64))
    hit_same = (np.asarray(hits_ref) == np.asarray(hits_cand))
    return {
        "rays": int(diff.size),
        "hit_id_mismatch": int((~hit_same).sum()),
        "hit_id_mismatch_ratio": round(float((~hit_same).mean()), 6),
        "range_max_abs_diff_m": round(float(diff.max()), 9),
        "range_mean_abs_diff_m": round(float(diff.mean()), 9),
        "range_within_tol": int((diff <= range_tol).sum()),
        "range_within_tol_ratio": round(float((diff <= range_tol).mean()), 6),
        "range_tolerance_m": range_tol,
        "hit_rays": int((np.asarray(hits_cand) >= 0).sum() + (np.asarray(hits_cand) <= -2).sum()),
        "empty_scene_rays": int((np.asarray(hits_cand) == -1).sum()),
    }


def run(poses: np.ndarray, *, seed: int = 77001, label: str = "m1_lidar") -> dict:
    scene = build_reference_scene(seed=seed)
    origins, dirs = reference_rays(np.asarray(poses, np.float64))
    ref_ranges, ref_hits = reference_cast(scene, origins, dirs)
    terrain_only = (ref_hits == -1) & (ref_ranges < LIDAR_RANGE - 1e-6)
    cand_ranges, cand_hits = candidate_cast(scene, origins, dirs)
    stats = compare(ref_ranges, ref_hits, cand_ranges, cand_hits)
    stats.update({
        "scene_seed": seed, "label": label,
        "poses": int(len(poses)),
        "ray_pattern": f"{RAY_COUNT} horizontal rays, az -pi..pi, range {LIDAR_RANGE} m",
        "terrain_amplitude_m": 0.35, "sensor_height_m": 0.55,
        "reference_terrain_only_rays": int(terrain_only.sum()),
        "leaf_patches": int(len(scene.centers)), "plants": int(len(scene.xy)),
        "rock_density": 0.0,
        "terrain_note": "reference terrain branch contributes no hit for horizontal rays at 0.55 m (checked)",
    })
    return {"stats": stats, "scene": scene, "origins": origins, "dirs": dirs,
            "ref_ranges": ref_ranges, "ref_hits": ref_hits, "cand_ranges": cand_ranges, "cand_hits": cand_hits}


if __name__ == "__main__":
    rng = np.random.default_rng(4242)
    n = 16
    poses = np.stack([rng.uniform(5, 95, n), rng.uniform(5, 95, n),
                      rng.uniform(-math.pi, math.pi, n), rng.uniform(-0.3, 0.3, n)], axis=1)
    out = run(poses)
    print("RESULT " + json.dumps(out["stats"]))
