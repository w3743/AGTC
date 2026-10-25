"""Analytic depth caster: 160x88 camera-Z depth from the reference scene geometry.

Semantics match `isaac_field` (no analytic reference depth exists in that repo -- the
reference always renders with the Isaac RTX camera):
  * camera model = `FieldTask.depth_intrinsics` (scaled `DEPTH_NATIVE_INTRINSICS`),
    pixel rays (x, y, 1) in the camera frame, axial-Z = ray parameter;
  * camera pose = body yaw * gimbal yaw * gimbal pitch, origin = root_z + camera_height;
  * metric depth clipped to [depth_min_range=0.3, depth_range=5.0]; invalid pixels
    become 0.0 after the `where(valid, value, 0)/depth_range` normalisation;
  * observation channel (1, 160, 88) float32, normalised to [0, 1].

Two implementations are provided so the warp kernel can be validated on CPU:
  `cast_numpy`  -- independent vectorised numpy implementation (reference for the kernel)
  `cast_warp`   -- warp kernel (CPU or CUDA), same math and dtypes
"""
from __future__ import annotations

import math
import sys

import numpy as np

HERE = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from m1_lidar import build_reference_scene  # noqa: E402

DEPTH_MIN_RANGE = 0.3
DEPTH_RANGE = 5.0
DEPTH_HEIGHT = 160
DEPTH_WIDTH = 88
CAMERA_HEIGHT = 0.4          # isaac_field Config.camera_height
ROOT_OFFSET = 0.15           # prototype MuJoCo base-to-ground offset (body origin height)


def camera_model():
    """Exact intrinsics/pixel grid used by the reference task (read-only import)."""
    from isaac_field.task import Config, FieldTask
    cfg = Config(size=100.0, plants=0, dry_fraction=0.0, terrain_enabled=False)
    task = FieldTask(cfg, "goto")
    return np.asarray(task.depth_intrinsics, np.float64), np.asarray(task.depth_camera_xy, np.float64)


def camera_basis(yaw, gimbal_yaw, gimbal_pitch):
    """Returns (forward, right, down) world-frame unit vectors (isaac_field _flat_basis convention)."""
    f_local = np.array([math.cos(gimbal_pitch) * math.cos(gimbal_yaw),
                        math.cos(gimbal_pitch) * math.sin(gimbal_yaw),
                        math.sin(gimbal_pitch)])
    right_local = np.array([math.sin(gimbal_yaw), -math.cos(gimbal_yaw), 0.0])
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    forward = rot @ f_local
    right = rot @ right_local
    down = np.cross(forward, right)   # camera +Y (down) = F x X
    return forward, right, down


def camera_rays(yaw, gimbal_yaw, gimbal_pitch, camera_xy):
    """World ray directions with axial-normalised length (Z_cam component == 1)."""
    forward, right, down = camera_basis(yaw, gimbal_yaw, gimbal_pitch)
    x = camera_xy[:, 0:1]
    y = camera_xy[:, 1:2]
    dirs = forward[None, :] + x * right[None, :] + y * down[None, :]
    return dirs


def terrain_march(terrain, origins, dirs, max_range, *, step=0.02):
    """First crossing of the heightfield, marching in axial-t (dirs not normalised).

    Returns distance array (inf where no hit). Linear bisection refinement between
    the two bracketing samples keeps the error below the march step.
    """
    origins = np.asarray(origins, np.float64)
    dirs = np.asarray(dirs, np.float64)
    n_rays = origins.shape[0]
    best = np.full(n_rays, np.inf)
    if terrain is None:
        return best
    ts = np.arange(step, max_range + 1e-9, step, dtype=np.float64)
    oxy, oz = origins[:, :2], origins[:, 2]
    dxy, dz = dirs[:, :2], dirs[:, 2]
    for t in ts:
        pxy = oxy + t * dxy
        pz = oz + t * dz
        below = pz <= terrain_height(terrain, pxy)
        new = below & ~np.isfinite(best)
        best[new] = t
    finite = np.isfinite(best)
    if finite.any():
        idx = np.flatnonzero(finite)
        a = np.maximum(best[idx] - step, 0.0)
        b = best[idx].copy()
        for _ in range(8):
            mid = 0.5 * (a + b)
            pxy = oxy[idx] + mid[:, None] * dxy[idx]
            pz = oz[idx] + mid * dz[idx]
            below = pz <= terrain_height(terrain, pxy)
            b = np.where(below, mid, b)
            a = np.where(below, a, mid)
        best[idx] = b
    return best


def terrain_height(terrain, xy):
    if terrain is None:
        return np.zeros(xy.shape[0])
    return np.asarray(terrain.surface_height(xy), np.float64)


def _pose_origin(poses_e, terrain):
    """(x,y) or (x,y,z) pose row -> (origin3, yaw, gimbal_yaw, gimbal_pitch)."""
    if len(poses_e) == 6:
        return (np.array([poses_e[0], poses_e[1], poses_e[2]], np.float64),
                poses_e[3], poses_e[4], poses_e[5])
    x, y = float(poses_e[0]), float(poses_e[1])
    z = float(terrain_height(terrain, np.array([[x, y]]))[0]) + ROOT_OFFSET + CAMERA_HEIGHT
    return np.array([x, y, z], np.float64), poses_e[2], poses_e[3], poses_e[4]


def cast_numpy(poses, *, scene=None, intrinsics=None, camera_xy=None, chunk=None):
    """poses: (n,5) x,y,yaw,gimbal_yaw,gimbal_pitch or (n,6) with explicit camera z.

    -> metric depth (n,160,88) in metres (axial Z, clipped to [0.3, 5.0]).
    """
    scene = scene or build_reference_scene()
    if intrinsics is None:
        intrinsics, camera_xy = camera_model()
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    plant_xy = np.asarray(scene.xy, np.float32)
    stem_radius = np.asarray(scene.stem_radius, np.float32)
    root_z = np.asarray(scene.root_z, np.float32)
    height = np.asarray(scene.height, np.float32)
    terrain = scene.terrain

    out = np.full((n, DEPTH_HEIGHT * DEPTH_WIDTH), DEPTH_RANGE, np.float32)
    for e in range(n):
        origin, yaw, gy, gp = _pose_origin(poses[e], terrain)
        dirs = camera_rays(float(yaw), float(gy), float(gp), camera_xy).astype(np.float64)
        origins_rays = np.broadcast_to(origin[None, :], dirs.shape)
        best = terrain_march(terrain, origins_rays, dirs, DEPTH_RANGE).astype(np.float32)
        # leaf/stem math in float32 (same dtype path as the warp kernel; float64 here
        # flips patch-edge inclusion on a handful of grazing pixels)
        o32 = origin.astype(np.float32)
        d32 = dirs.astype(np.float32)
        # ---- leaf patches near the camera (cull radius = range + padding) ----
        padding = max(0.5, float(leaf_axes.max(initial=0.0)))
        near = np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1) < (DEPTH_RANGE + padding) ** 2
        ids = np.flatnonzero(near)
        if len(ids):
            c, nrm, u, v, axes = (leaf_centers[ids], leaf_normals[ids], leaf_u[ids],
                                  leaf_v[ids], leaf_axes[ids])
            den = d32 @ nrm.T
            num = np.sum((c - o32) * nrm, axis=1)
            t = np.divide(num[None, :], den, out=np.full_like(den, np.inf),
                          where=np.abs(den) > 1e-7)
            pu = np.sum((o32 - c) * u, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d32 @ u.T)
            pv = np.sum((o32 - c) * v, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d32 @ v.T)
            sx, sy = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - sx ** 2) * (1.0 - np.abs(sx))
            inside = (np.abs(sx) <= 1.0) & (sy ** 2 <= taper)
            ts = np.where(inside & (t > 1e-4) & (t <= DEPTH_RANGE), t, np.inf)
            best = np.minimum(best, ts.min(axis=1).astype(np.float32))
        # ---- stems (vertical cylinders) ----
        stems = np.flatnonzero(np.sum((plant_xy - origin[:2]) ** 2, axis=1) < (DEPTH_RANGE + 0.2) ** 2)
        if len(stems):
            q = (o32[:2] - plant_xy[stems])
            a = (d32[:, 0] ** 2 + d32[:, 1] ** 2)[:, None]
            b = 2.0 * (d32[:, :2] @ q.T)
            cterm = np.sum(q * q, axis=1)[None, :] - stem_radius[stems] ** 2
            disc = b * b - 4.0 * a * cterm
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.divide(-b - np.sqrt(np.maximum(disc, 0)), 2.0 * a,
                              out=np.full_like(b, np.inf), where=a > 1e-8)
            z = o32[2] + np.where(np.isfinite(t), t, 0) * d32[:, 2, None]
            t = np.where((disc >= 0) & (t > 1e-4) & (z >= root_z[stems])
                         & (z <= root_z[stems] + height[stems]), t, np.inf)
            best = np.minimum(best, t.min(axis=1).astype(np.float32))
        out[e] = np.clip(best, DEPTH_MIN_RANGE, DEPTH_RANGE)
    valid = np.isfinite(out) & (out >= DEPTH_MIN_RANGE) & (out < DEPTH_RANGE)
    return out.reshape(n, DEPTH_HEIGHT, DEPTH_WIDTH), valid.reshape(n, DEPTH_HEIGHT, DEPTH_WIDTH)


def observation_from_depth(depth, valid):
    """observation channel: where(valid, value, 0)/depth_range, shape (n, 1, 160, 88)."""
    return (np.where(valid, depth, 0.0) / DEPTH_RANGE)[:, None, :, :].astype(np.float32)


def cast_warp(poses, *, scene=None, intrinsics=None, camera_xy=None, device=None):
    """Warp kernel implementation used by the native stack (CPU or CUDA)."""
    import warp as wp

    scene = scene or build_reference_scene()
    if intrinsics is None:
        intrinsics, camera_xy = camera_model()
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
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
    terrain = scene.terrain
    pupil = np.asarray(camera_xy, np.float64)
    npix = DEPTH_HEIGHT * DEPTH_WIDTH

    @wp.kernel
    def depth_kernel(origin: wp.array(dtype=wp.vec3), forward: wp.array(dtype=wp.vec3),
                     right: wp.array(dtype=wp.vec3), down: wp.array(dtype=wp.vec3),
                     pupil_xy: wp.array(dtype=wp.vec2),
                     centers: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                     uu: wp.array(dtype=wp.vec3), vv: wp.array(dtype=wp.vec3),
                     axes: wp.array(dtype=wp.vec2), pxy: wp.array(dtype=wp.vec2),
                     stem_r: wp.array(dtype=wp.float32), root: wp.array(dtype=wp.float32),
                     hgt: wp.array(dtype=wp.float32),
                     wave_k: wp.array(dtype=wp.vec2), wave_a: wp.array(dtype=wp.float32),
                     wave_ph: wp.array(dtype=wp.float32), n_waves: int, base_h: float,
                     march_step: float, march_n: int,
                     n_patches: int, n_stems: int, n_pix: int, max_range: float,
                     min_range: float, out_depth: wp.array(dtype=wp.float32)):
        i = wp.tid()
        pix = i
        o = origin[0]
        uv = pupil_xy[pix]
        d = forward[0] + uv[0] * right[0] + uv[1] * down[0]
        best = max_range
        # ---- terrain heightfield: march axial-t, then bisect the first bracket ----
        for k in range(march_n):
            t = march_step * wp.float32(k + 1)
            px = o[0] + t * d[0]
            py = o[1] + t * d[1]
            pz = o[2] + t * d[2]
            h = base_h
            for w in range(n_waves):
                h = h + wave_a[w] * wp.sin(wave_k[w][0] * px + wave_k[w][1] * py + wave_ph[w])
            if pz <= h:
                lo = t - march_step
                hi = t
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    mx = o[0] + mid * d[0]
                    my = o[1] + mid * d[1]
                    mz = o[2] + mid * d[2]
                    mh = base_h
                    for w in range(n_waves):
                        mh = mh + wave_a[w] * wp.sin(wave_k[w][0] * mx + wave_k[w][1] * my + wave_ph[w])
                    if mz <= mh:
                        hi = mid
                    else:
                        lo = mid
                best = hi
                break
        for p in range(n_patches):
            c = centers[p]
            nrm = normals[p]
            den = wp.dot(d, nrm)
            if wp.abs(den) > 1.0e-7:
                t = wp.dot(c - o, nrm) / den
                if t > 1.0e-4 and t < best:
                    u = uu[p]
                    v = vv[p]
                    pu = wp.dot(o - c, u) + t * wp.dot(d, u)
                    pv = wp.dot(o - c, v) + t * wp.dot(d, v)
                    sx = pu / axes[p][0]
                    sy = pv / axes[p][1]
                    taper = (1.0 - sx * sx) * (1.0 - wp.abs(sx))
                    if wp.abs(sx) <= 1.0 and sy * sy <= taper:
                        best = t
        for s in range(n_stems):
            qx = o[0] - pxy[s][0]
            qy = o[1] - pxy[s][1]
            a = d[0] * d[0] + d[1] * d[1]
            if a > 1.0e-8:
                b = 2.0 * (d[0] * qx + d[1] * qy)
                cc = qx * qx + qy * qy - stem_r[s] * stem_r[s]
                disc = b * b - 4.0 * a * cc
                if disc >= 0.0:
                    t = (-b - wp.sqrt(disc)) / (2.0 * a)
                    z = o[2] + t * d[2]
                    if t > 1.0e-4 and t < best and z >= root[s] and z <= root[s] + hgt[s]:
                        best = t
        if best < min_range:
            best = min_range
        out_depth[i] = best

    depth_out = np.full((n, npix), DEPTH_RANGE, np.float32)
    # terrain wave parameters (analytic heightfield, same formula as Terrain.height)
    base_h = float(getattr(terrain, "base_height", 0.0)) if terrain is not None else 0.0
    if terrain is not None:
        wave_k = np.asarray(terrain.wave_vectors4x2, np.float32).reshape(-1, 2)
        wave_a = np.asarray(terrain.wave_amplitudes4, np.float32).reshape(-1)
        wave_ph = np.asarray(terrain.wave_phases4, np.float32).reshape(-1)
    else:
        wave_k = np.zeros((1, 2), np.float32)
        wave_a = np.zeros(1, np.float32)
        wave_ph = np.zeros(1, np.float32)
    n_waves = int(wave_a.shape[0])
    march_step = 0.02
    march_n = int(np.ceil(DEPTH_RANGE / march_step))
    wk_wp = wp.array(wave_k.reshape(-1, 2), dtype=wp.vec2, device=dev)
    wa_wp = wp.array(wave_a, dtype=wp.float32, device=dev)
    wp_wp = wp.array(wave_ph, dtype=wp.float32, device=dev)
    origins = np.zeros((n, 3), np.float32)
    forwards = np.zeros((n, 3), np.float32)
    rights = np.zeros((n, 3), np.float32)
    downs = np.zeros((n, 3), np.float32)
    for e in range(n):
        origin, yaw, gy, gp = _pose_origin(poses[e], terrain)
        origins[e] = origin
        f, r, d = camera_basis(float(yaw), float(gy), float(gp))
        forwards[e], rights[e], downs[e] = f, r, d
    o_wp = wp.array(origins.reshape(-1, 3), dtype=wp.vec3, device=dev)
    f_wp = wp.array(forwards.reshape(-1, 3), dtype=wp.vec3, device=dev)
    r_wp = wp.array(rights.reshape(-1, 3), dtype=wp.vec3, device=dev)
    d_wp = wp.array(downs.reshape(-1, 3), dtype=wp.vec3, device=dev)
    uv_wp = wp.array(pupil.astype(np.float32).reshape(-1, 2), dtype=wp.vec2, device=dev)

    for e in range(n):
        origin = origins[e]
        padding = max(0.5, float(leaf_axes.max(initial=0.0)))
        near = np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1) < (DEPTH_RANGE + padding) ** 2
        patch_ids = np.flatnonzero(near)
        stems = np.flatnonzero(np.sum((plant_xy - origin[:2]) ** 2, axis=1) < (DEPTH_RANGE + 0.2) ** 2)
        if len(patch_ids) == 0 and len(stems) == 0:
            continue
        c_wp = wp.array(leaf_centers[patch_ids], dtype=wp.vec3, device=dev)
        n_wp = wp.array(leaf_normals[patch_ids], dtype=wp.vec3, device=dev)
        u_wp = wp.array(leaf_u[patch_ids], dtype=wp.vec3, device=dev)
        v_wp = wp.array(leaf_v[patch_ids], dtype=wp.vec3, device=dev)
        a_wp = wp.array(leaf_axes[patch_ids].reshape(-1, 2), dtype=wp.vec2, device=dev)
        px_wp = wp.array(plant_xy[stems].reshape(-1, 2), dtype=wp.vec2, device=dev)
        sr_wp = wp.array(stem_radius[stems], dtype=wp.float32, device=dev)
        rz_wp = wp.array(root_z[stems], dtype=wp.float32, device=dev)
        hg_wp = wp.array(height[stems], dtype=wp.float32, device=dev)
        out_wp = wp.array(depth_out[e], dtype=wp.float32, device=dev)
        # one-pose arrays: the kernel renders exactly one pose per launch
        o_e = wp.array(origins[e:e + 1].reshape(-1, 3), dtype=wp.vec3, device=dev)
        f_e = wp.array(forwards[e:e + 1].reshape(-1, 3), dtype=wp.vec3, device=dev)
        r_e = wp.array(rights[e:e + 1].reshape(-1, 3), dtype=wp.vec3, device=dev)
        d_e = wp.array(downs[e:e + 1].reshape(-1, 3), dtype=wp.vec3, device=dev)
        src = [o_e, f_e, r_e, d_e, uv_wp, c_wp, n_wp, u_wp, v_wp, a_wp, px_wp,
               sr_wp, rz_wp, hg_wp, wk_wp, wa_wp, wp_wp, int(n_waves), np.float32(base_h),
               np.float32(march_step), int(march_n),
               int(len(patch_ids)), int(len(stems)), int(npix),
               np.float32(DEPTH_RANGE), np.float32(DEPTH_MIN_RANGE), out_wp]
        wp.launch(depth_kernel, dim=npix, inputs=src, device=dev,
                  block_dim=256)
        wp.synchronize()
        depth_out[e] = out_wp.numpy()

    valid = depth_out < DEPTH_RANGE
    return (depth_out.reshape(n, DEPTH_HEIGHT, DEPTH_WIDTH),
            valid.reshape(n, DEPTH_HEIGHT, DEPTH_WIDTH), str(dev))


def poses_from_state(states, yaw=None, gimbal_yaw=None, gimbal_pitch=None):
    """Convenience: build poses (n,5) from arrays of x, y, yaw, gy, gp."""
    states = np.asarray(states, np.float64)
    return np.stack([states[:, 0], states[:, 1], yaw, gimbal_yaw, gimbal_pitch], axis=1)
