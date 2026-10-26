"""Analytic detections 16x8 matching isaac_field's detector semantics (CPU, small).

Reference semantics extracted from `isaac_field/env.py` `_capture(kind='detector')`:
  * detector camera: 64x64, f = 64 / (2 tan(pi/6)) = 55.4256, cx = cy = 31.5,
    clipping (0.12, camera_range=7.0), same mount pose as the depth camera;
  * instance segmentation mask: one instance per leaf mesh path `/Corn_<p>/Leaf_<j>`
    (stems are not matched by the reference regex, so they are never detected);
  * circular FOV: (u-31.5)^2 + (v-31.5)^2 <= 32^2;
  * per instance: skip if pixels < 2 or rand < detection_drop (0.08);
    confidence = clip(0.70 + 0.28*min(1, n_pix/20) + N(0, 0.015), 0, 0.99);
    class_id = leaf_dry XOR (rand < class_error (0.015));
    range only when this instance is the centre pixel (image[32,32]) and
    0.12 <= rangefinder distance <= camera_range (rangefinder = 1-ray at the mount);
    box = [(xmin+xmax+1)/128, (ymin+ymax+1)/128, (xmax-xmin+1)/64, (ymax-ymin+1)/64];
  * detections sorted by confidence descending, truncated to max_detections=16;
  * `isaac_field/task.py` packs each row as
    [cx, cy, w, h, class_id, confidence, min(1, range/depth_range), range_valid].

The analytic version renders the mask with the same raycast geometry as the depth caster
(leaf patches; stems/terrain only contribute to the 1-ray rangefinder distance).
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

import depth_caster as dc  # noqa: E402

DETECTOR_SIZE = 64
DETECTOR_FOCAL = DETECTOR_SIZE / (2.0 * math.tan(math.pi / 6.0))  # 55.4256...
DETECTOR_CX = 31.5
DETECTOR_CY = 31.5
DETECTOR_CLIP_MIN = 0.12
DETECTOR_CLIP_MAX = 7.0          # Config.camera_range
MAX_DETECTIONS = 16
DEPTH_RANGE = 5.0                # Config.depth_range (row range normalisation)
CAMERA_HEIGHT = dc.CAMERA_HEIGHT  # detector and depth camera share the mount pose

DETECTION_NOISE = dict(drop=0.08, class_error=0.015, conf_std=0.015,
                       range_std=0.01, conf_base=0.70, conf_pix=0.28, conf_pix_cap=20)


def detector_pixel_xy():
    v, u = np.meshgrid(np.arange(DETECTOR_SIZE, dtype=np.float64),
                       np.arange(DETECTOR_SIZE, dtype=np.float64), indexing="ij")
    x = (u - DETECTOR_CX) / DETECTOR_FOCAL
    y = (v - DETECTOR_CY) / DETECTOR_FOCAL
    return np.stack([x.ravel(), y.ravel()], axis=1)


def detector_basis(yaw, gimbal_yaw, gimbal_pitch):
    """Same mount basis as the depth camera (forward, right, down)."""
    return dc.camera_basis(yaw, gimbal_yaw, gimbal_pitch)


def cast_mask(poses, *, scene=None, pixel_xy=None, chunk=4096):
    """Render per-leaf instance masks: (n,64,64) int32 (-1 = no leaf hit)."""
    scene = scene or dc.build_reference_scene()
    pixel_xy = detector_pixel_xy() if pixel_xy is None else pixel_xy
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    leaf_plant = np.asarray(scene.leaf_plant, np.int64)
    masks = np.full((n, DETECTOR_SIZE * DETECTOR_SIZE), -1, np.int64)
    padding = max(0.5, float(leaf_axes.max(initial=0.0)))
    for e in range(n):
        origin, yaw, gy, gp = dc._pose_origin(poses[e], None)  # explicit z: pose row must be 6 cols
        if len(poses[e]) == 5:  # 5-col poses are not supported here (depth camera z needed)
            raise ValueError("cast_mask requires poses with explicit camera z (6 columns)")
        forward, right, down = detector_basis(float(yaw), float(gy), float(gp))
        dirs = (forward[None, :] + pixel_xy[:, 0:1] * right[None, :]
                + pixel_xy[:, 1:2] * down[None, :]).astype(np.float64)
        best = np.full(dirs.shape[0], np.inf, np.float32)
        hit = np.full(dirs.shape[0], -1, np.int64)
        near = np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1) < (DETECTOR_CLIP_MAX + padding) ** 2
        ids = np.flatnonzero(near)
        if len(ids):
            c, nrm, u, v, axes = (leaf_centers[ids], leaf_normals[ids], leaf_u[ids],
                                  leaf_v[ids], leaf_axes[ids])
            o32 = origin.astype(np.float32)
            d32 = dirs.astype(np.float32)
            den = d32 @ nrm.T
            num = np.sum((c - o32) * nrm, axis=1)
            t = np.divide(num[None, :], den, out=np.full_like(den, np.inf),
                          where=np.abs(den) > 1e-7)
            pu = np.sum((o32 - c) * u, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d32 @ u.T)
            pv = np.sum((o32 - c) * v, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d32 @ v.T)
            sx, sy = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - sx ** 2) * (1.0 - np.abs(sx))
            inside = (np.abs(sx) <= 1.0) & (sy ** 2 <= taper)
            ts = np.where(inside & (t > DETECTOR_CLIP_MIN) & (t <= DETECTOR_CLIP_MAX), t, np.inf)
            j = np.argmin(ts, axis=1)
            cand = ts[np.arange(len(dirs)), j]
            valid = np.isfinite(cand)
            best[valid] = cand[valid]
            hit[valid] = ids[j[valid]]
        masks[e] = hit
    return masks.reshape(n, DETECTOR_SIZE, DETECTOR_SIZE)


def cast_mask_warp(poses, *, scene=None, pixel_xy=None, device=None):
    """Warp-kernel version of `cast_mask` (CPU or CUDA); identical mask semantics.

    One kernel launch per pose (64x64 = 4,096 rays), leaf patches culled on the host
    exactly like the numpy path. Returns (masks int64, device string).
    """
    import warp as wp

    scene = scene or dc.build_reference_scene()
    pixel_xy = detector_pixel_xy() if pixel_xy is None else pixel_xy
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
    wp.init()
    dev = device or ("cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")

    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    padding = max(0.5, float(leaf_axes.max(initial=0.0)))
    npix = DETECTOR_SIZE * DETECTOR_SIZE

    @wp.kernel
    def mask_kernel(origin: wp.array(dtype=wp.vec3), forward: wp.array(dtype=wp.vec3),
                    right: wp.array(dtype=wp.vec3), down: wp.array(dtype=wp.vec3),
                    pupil_xy: wp.array(dtype=wp.vec2),
                    centers: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                    uu: wp.array(dtype=wp.vec3), vv: wp.array(dtype=wp.vec3),
                    axes: wp.array(dtype=wp.vec2), patch_ids: wp.array(dtype=wp.int32),
                    n_patches: int, clip_min: float, clip_max: float,
                    out_mask: wp.array(dtype=wp.int32)):
        i = wp.tid()
        uv = pupil_xy[i]
        o = origin[0]
        d = forward[0] + uv[0] * right[0] + uv[1] * down[0]
        best = clip_max
        hit = int(-1)
        for p in range(n_patches):
            c = centers[p]
            nrm = normals[p]
            den = wp.dot(d, nrm)
            if wp.abs(den) > 1.0e-7:
                t = wp.dot(c - o, nrm) / den
                if t > clip_min and t < best:
                    u = uu[p]
                    v = vv[p]
                    pu = wp.dot(o - c, u) + t * wp.dot(d, u)
                    pv = wp.dot(o - c, v) + t * wp.dot(d, v)
                    sx = pu / axes[p][0]
                    sy = pv / axes[p][1]
                    taper = (1.0 - sx * sx) * (1.0 - wp.abs(sx))
                    if wp.abs(sx) <= 1.0 and sy * sy <= taper:
                        best = t
                        hit = patch_ids[p]
        out_mask[i] = hit

    pupil_wp = wp.array(pixel_xy.astype(np.float32).reshape(-1, 2), dtype=wp.vec2, device=dev)
    masks = np.full((n, npix), -1, np.int64)
    for e in range(n):
        origin, yaw, gy, gp = dc._pose_origin(poses[e], None)
        forward, right, down = detector_basis(float(yaw), float(gy), float(gp))
        near = np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1) < (DETECTOR_CLIP_MAX + padding) ** 2
        patch_ids = np.flatnonzero(near).astype(np.int32)
        o_wp = wp.array(origin.astype(np.float32).reshape(1, 3), dtype=wp.vec3, device=dev)
        f_wp = wp.array(forward.astype(np.float32).reshape(1, 3), dtype=wp.vec3, device=dev)
        r_wp = wp.array(right.astype(np.float32).reshape(1, 3), dtype=wp.vec3, device=dev)
        d_wp = wp.array(down.astype(np.float32).reshape(1, 3), dtype=wp.vec3, device=dev)
        if len(patch_ids) == 0:
            continue
        c_wp = wp.array(leaf_centers[patch_ids], dtype=wp.vec3, device=dev)
        n_wp = wp.array(leaf_normals[patch_ids], dtype=wp.vec3, device=dev)
        u_wp = wp.array(leaf_u[patch_ids], dtype=wp.vec3, device=dev)
        v_wp = wp.array(leaf_v[patch_ids], dtype=wp.vec3, device=dev)
        a_wp = wp.array(leaf_axes[patch_ids].reshape(-1, 2), dtype=wp.vec2, device=dev)
        pid_wp = wp.array(patch_ids, dtype=wp.int32, device=dev)
        out_wp = wp.array(np.full(npix, -1, np.int32), dtype=wp.int32, device=dev)
        wp.launch(mask_kernel, dim=npix,
                  inputs=[o_wp, f_wp, r_wp, d_wp, pupil_wp, c_wp, n_wp, u_wp, v_wp,
                          a_wp, pid_wp, int(len(patch_ids)),
                          np.float32(DETECTOR_CLIP_MIN), np.float32(DETECTOR_CLIP_MAX), out_wp],
                  device=dev)
        wp.synchronize()
        masks[e] = out_wp.numpy()
    return masks.reshape(n, DETECTOR_SIZE, DETECTOR_SIZE).astype(np.int64), str(dev)


def cast_mask_warp_batch(poses, *, scene=None, pixel_xy=None, device=None):
    """Single-launch batched version of `cast_mask_warp` (n poses x 4096 pixels).

    Candidate leaf patches are culled per pose on the host and packed into a padded
    (n, max_candidates) id table; the kernel loops only those candidates, so one
    launch covers the whole batch (removes the per-pose host launch overhead).
    """
    import warp as wp

    scene = scene or dc.build_reference_scene()
    pixel_xy = detector_pixel_xy() if pixel_xy is None else pixel_xy
    poses = np.asarray(poses, np.float64)
    n = poses.shape[0]
    wp.init()
    dev = device or ("cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")

    leaf_centers = np.asarray(scene.centers, np.float32)
    leaf_normals = np.asarray(scene.normals, np.float32)
    leaf_u = np.asarray(scene.u, np.float32)
    leaf_v = np.asarray(scene.v, np.float32)
    leaf_axes = np.asarray(scene.axes, np.float32)
    padding = max(0.5, float(leaf_axes.max(initial=0.0)))
    npix = DETECTOR_SIZE * DETECTOR_SIZE

    origins = np.zeros((n, 3), np.float32)
    forwards = np.zeros((n, 3), np.float32)
    rights = np.zeros((n, 3), np.float32)
    downs = np.zeros((n, 3), np.float32)
    cand_lists = []
    for e in range(n):
        origin, yaw, gy, gp = dc._pose_origin(poses[e], None)
        forward, right, down = detector_basis(float(yaw), float(gy), float(gp))
        origins[e] = origin
        forwards[e], rights[e], downs[e] = forward, right, down
        near = np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1) < (DETECTOR_CLIP_MAX + padding) ** 2
        cand_lists.append(np.flatnonzero(near).astype(np.int32))
    max_cand = max(1, max((len(c) for c in cand_lists), default=1))
    cand = np.full((n, max_cand), -1, np.int32)
    for e, ids in enumerate(cand_lists):
        cand[e, :len(ids)] = ids

    @wp.kernel
    def mask_batch_kernel(origin: wp.array(dtype=wp.vec3), forward: wp.array(dtype=wp.vec3),
                          right: wp.array(dtype=wp.vec3), down: wp.array(dtype=wp.vec3),
                          pupil_xy: wp.array(dtype=wp.vec2),
                          centers: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                          uu: wp.array(dtype=wp.vec3), vv: wp.array(dtype=wp.vec3),
                          axes: wp.array(dtype=wp.vec2), cand: wp.array2d(dtype=wp.int32),
                          max_cand: int, n_pix: int, clip_min: float, clip_max: float,
                          out_mask: wp.array(dtype=wp.int32)):
        i = wp.tid()
        env = i / n_pix
        pix = i % n_pix
        uv = pupil_xy[pix]
        o = origin[env]
        d = forward[env] + uv[0] * right[env] + uv[1] * down[env]
        best = clip_max
        hit = int(-1)
        for p in range(max_cand):
            cid = cand[env, p]
            if cid < 0:
                continue
            c = centers[cid]
            nrm = normals[cid]
            den = wp.dot(d, nrm)
            if wp.abs(den) > 1.0e-7:
                t = wp.dot(c - o, nrm) / den
                if t > clip_min and t < best:
                    u = uu[cid]
                    v = vv[cid]
                    pu = wp.dot(o - c, u) + t * wp.dot(d, u)
                    pv = wp.dot(o - c, v) + t * wp.dot(d, v)
                    sx = pu / axes[cid][0]
                    sy = pv / axes[cid][1]
                    taper = (1.0 - sx * sx) * (1.0 - wp.abs(sx))
                    if wp.abs(sx) <= 1.0 and sy * sy <= taper:
                        best = t
                        hit = cid
        out_mask[i] = hit

    o_wp = wp.array(origins, dtype=wp.vec3, device=dev)
    f_wp = wp.array(forwards, dtype=wp.vec3, device=dev)
    r_wp = wp.array(rights, dtype=wp.vec3, device=dev)
    d_wp = wp.array(downs, dtype=wp.vec3, device=dev)
    pupil_wp = wp.array(pixel_xy.astype(np.float32).reshape(-1, 2), dtype=wp.vec2, device=dev)
    c_wp = wp.array(leaf_centers, dtype=wp.vec3, device=dev)
    n_wp = wp.array(leaf_normals, dtype=wp.vec3, device=dev)
    u_wp = wp.array(leaf_u, dtype=wp.vec3, device=dev)
    v_wp = wp.array(leaf_v, dtype=wp.vec3, device=dev)
    a_wp = wp.array(leaf_axes.reshape(-1, 2), dtype=wp.vec2, device=dev)
    cand_wp = wp.array(cand, dtype=wp.int32, device=dev)
    out_wp = wp.array(np.full(n * npix, -1, np.int32), dtype=wp.int32, device=dev)
    wp.launch(mask_batch_kernel, dim=n * npix,
              inputs=[o_wp, f_wp, r_wp, d_wp, pupil_wp, c_wp, n_wp, u_wp, v_wp, a_wp,
                      cand_wp, int(max_cand), int(npix), np.float32(DETECTOR_CLIP_MIN),
                      np.float32(DETECTOR_CLIP_MAX), out_wp],
              device=dev)
    wp.synchronize()
    masks = out_wp.numpy().reshape(n, DETECTOR_SIZE, DETECTOR_SIZE).astype(np.int64)
    return masks, str(dev)


def cast_center_distance(poses, *, scene=None):
    """One-ray rangefinder: axial distance along the camera forward axis (leaves + stems + terrain)."""
    scene = scene or dc.build_reference_scene()
    poses = np.asarray(poses, np.float64)
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
    out = np.full(poses.shape[0], np.inf, np.float32)
    for e in range(poses.shape[0]):
        origin, yaw, gy, gp = dc._pose_origin(poses[e], None)
        forward, _, _ = detector_basis(float(yaw), float(gy), float(gp))
        best = float(dc.terrain_march(terrain, np.broadcast_to(origin[None, :], (1, 3)),
                                      forward[None, :], DETECTOR_CLIP_MAX)[0])
        near = leaf_centers[leaf_axes.max(axis=1) > 0]  # keep arrays aligned; cull below
        ids = np.flatnonzero(np.sum((leaf_centers[:, :2] - origin[:2]) ** 2, axis=1)
                             < (DETECTOR_CLIP_MAX + 0.5) ** 2)
        if len(ids):
            c, nrm, u, v, axes = (leaf_centers[ids], leaf_normals[ids], leaf_u[ids],
                                  leaf_v[ids], leaf_axes[ids])
            o32 = origin.astype(np.float32)
            d32 = forward.astype(np.float32)
            den = d32 @ nrm.T
            num = np.sum((c - o32) * nrm, axis=1)
            t = np.divide(num, den, out=np.full_like(den, np.inf), where=np.abs(den) > 1e-7)
            pu = np.sum((o32 - c) * u, axis=1) + t * (d32 @ u.T)
            pv = np.sum((o32 - c) * v, axis=1) + t * (d32 @ v.T)
            sx, sy = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - sx ** 2) * (1.0 - np.abs(sx))
            inside = (np.abs(sx) <= 1.0) & (sy ** 2 <= taper)
            ts = np.where(inside & (t > DETECTOR_CLIP_MIN) & (t <= DETECTOR_CLIP_MAX), t, np.inf)
            if len(ts):
                best = min(best, float(np.min(ts)))
        stems = np.flatnonzero(np.sum((plant_xy - origin[:2]) ** 2, axis=1) < (DETECTOR_CLIP_MAX + 0.2) ** 2)
        if len(stems):
            q = (origin[:2].astype(np.float32) - plant_xy[stems])
            dxy = forward[:2].astype(np.float32)
            a = float(dxy @ dxy)
            if a > 1e-8:
                b = 2.0 * (dxy @ q.T)
                cterm = np.sum(q * q, axis=1) - stem_radius[stems] ** 2
                disc = b * b - 4.0 * a * cterm
                ok = disc >= 0.0
                if ok.any():
                    tt = (-b[ok] - np.sqrt(np.maximum(disc[ok], 0))) / (2.0 * a)
                    z = origin[2] + tt * forward[2]
                    good = (tt > DETECTOR_CLIP_MIN) & (tt <= DETECTOR_CLIP_MAX) \
                        & (z >= root_z[stems][ok]) & (z <= (root_z[stems] + height[stems])[ok])
                    if good.any():
                        best = min(best, float(np.min(tt[good])))
        out[e] = best if best <= DETECTOR_CLIP_MAX else np.inf
    return out


def cast_center_distance_batch(poses, *, scene=None):
    """Batched 1-ray rangefinder: batched terrain march + per-pose leaf/stem tests.

    Uses the same math as `cast_center_distance`; the terrain march is done once for all
    n rays instead of looping poses, which was the dominant cost in the old path.
    """
    scene = scene or dc.build_reference_scene()
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
    origins = np.zeros((n, 3), np.float64)
    forwards = np.zeros((n, 3), np.float64)
    for e in range(n):
        origin, yaw, gy, gp = dc._pose_origin(poses[e], None)
        forward, _, _ = detector_basis(float(yaw), float(gy), float(gp))
        origins[e] = origin
        forwards[e] = forward
    best = np.asarray(dc.terrain_march(terrain, origins, forwards, DETECTOR_CLIP_MAX), np.float64)
    for e in range(n):
        o32 = origins[e].astype(np.float32)
        d32 = forwards[e].astype(np.float32)
        ids = np.flatnonzero(np.sum((leaf_centers[:, :2] - o32[:2]) ** 2, axis=1)
                             < (DETECTOR_CLIP_MAX + 0.5) ** 2)
        if len(ids):
            c, nrm, u, v, axes = (leaf_centers[ids], leaf_normals[ids], leaf_u[ids],
                                  leaf_v[ids], leaf_axes[ids])
            den = d32 @ nrm.T
            num = np.sum((c - o32) * nrm, axis=1)
            t = np.divide(num, den, out=np.full_like(den, np.inf), where=np.abs(den) > 1e-7)
            pu = np.sum((o32 - c) * u, axis=1) + t * (d32 @ u.T)
            pv = np.sum((o32 - c) * v, axis=1) + t * (d32 @ v.T)
            sx, sy = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - sx ** 2) * (1.0 - np.abs(sx))
            inside = (np.abs(sx) <= 1.0) & (sy ** 2 <= taper)
            ts = np.where(inside & (t > DETECTOR_CLIP_MIN) & (t <= DETECTOR_CLIP_MAX), t, np.inf)
            if ts.size:
                best[e] = min(best[e], float(np.min(ts)))
        stems = np.flatnonzero(np.sum((plant_xy - o32[:2]) ** 2, axis=1) < (DETECTOR_CLIP_MAX + 0.2) ** 2)
        if len(stems):
            q = o32[:2] - plant_xy[stems]
            dxy = d32[:2]
            a = float(dxy @ dxy)
            if a > 1e-8:
                b = 2.0 * (dxy @ q.T)
                cterm = np.sum(q * q, axis=1) - stem_radius[stems] ** 2
                disc = b * b - 4.0 * a * cterm
                ok = disc >= 0.0
                if ok.any():
                    tt = (-b[ok] - np.sqrt(np.maximum(disc[ok], 0))) / (2.0 * a)
                    z = o32[2] + tt * d32[2]
                    good = (tt > DETECTOR_CLIP_MIN) & (tt <= DETECTOR_CLIP_MAX) \
                        & (z >= root_z[stems][ok]) & (z <= (root_z[stems] + height[stems])[ok])
                    if good.any():
                        best[e] = min(best[e], float(np.min(tt[good])))
    return np.where(best <= DETECTOR_CLIP_MAX, best, np.inf).astype(np.float32)


def assemble(masks, center_distance, leaf_dry, *, rng=None, noise=False,
             noise_cfg=None) -> list[list[dict]]:
    """Reference detector rules -> per-env detection dict lists (sorted, capped at 16)."""
    cfg = dict(DETECTION_NOISE)
    if noise_cfg:
        cfg.update(noise_cfg)
    rng = rng or np.random.default_rng(0)
    n = masks.shape[0]
    out = []
    yy, xx = np.indices((DETECTOR_SIZE, DETECTOR_SIZE))
    cone = (xx - DETECTOR_CX) ** 2 + (yy - DETECTOR_CY) ** 2 <= 32.0 ** 2
    for e in range(n):
        image = masks[e]
        detections = []
        # reference uses image[32,32] exactly (int(31.5) would be 31)
        central_id = int(image[32, 32])
        for instance in np.unique(image[cone]):
            instance = int(instance)
            if instance < 0:
                continue
            pixels = (image == instance) & cone
            ys, xs = np.nonzero(pixels)
            if len(xs) < 2:
                continue
            if noise and rng.random() < cfg["drop"]:
                continue
            confidence = float(np.clip(
                cfg["conf_base"] + cfg["conf_pix"] * min(1.0, len(xs) / cfg["conf_pix_cap"])
                + (rng.normal(0.0, cfg["conf_std"]) if noise else 0.0), 0.0, 0.99))
            predicted = int(bool(leaf_dry[instance]) ^ ((rng.random() < cfg["class_error"]) if noise else False))
            distance = float(center_distance[e])
            measured = None
            if instance == central_id and DETECTOR_CLIP_MIN <= distance <= DETECTOR_CLIP_MAX:
                measured = distance + (rng.normal(0.0, cfg["range_std"]) if noise else 0.0)
            detections.append({
                "box": [(xs.min() + xs.max() + 1) / 128.0, (ys.min() + ys.max() + 1) / 128.0,
                        (xs.max() - xs.min() + 1) / 64.0, (ys.max() - ys.min() + 1) / 64.0],
                "class_id": predicted, "confidence": confidence, "range": measured,
                "plant": int(instance // 8), "leaf": int(instance % 8),
            })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        out.append(detections[:MAX_DETECTIONS])
    return out


def pack(detections, *, depth_range=DEPTH_RANGE, max_detections=MAX_DETECTIONS) -> np.ndarray:
    """isaac_field/task.py packing: (max_detections, 8) float32."""
    rows = np.zeros((max_detections, 8), np.float32)
    for i, d in enumerate(detections[:max_detections]):
        box = np.asarray(d["box"], np.float32)
        raw_range = d["range"]
        range_value = 0.0 if raw_range is None else float(raw_range)
        rows[i] = np.asarray([*box, float(d["class_id"]), float(d["confidence"]),
                              min(1.0, range_value / depth_range), float(raw_range is not None)],
                             np.float32)
    return rows


def observation_from_masks(masks, center_distance, leaf_dry, *, rng=None, noise=False,
                            noise_cfg=None) -> np.ndarray:
    dets = assemble(masks, center_distance, leaf_dry, rng=rng, noise=noise, noise_cfg=noise_cfg)
    return np.stack([pack(d) for d in dets], axis=0), dets


def assemble_batch(masks, center_distance, leaf_dry, *, rng=None, noise=False,
                   noise_cfg=None) -> list[list[dict]]:
    """Vectorised `assemble`: groups all (pose, instance) pixel sets at once.

    Pixels are flattened across poses, cone-masked, grouped by (pose, instance) with a
    single `np.unique`, and bounding boxes are produced with `reduceat`; the per-instance
    confidence/class/range rules are then identical to `assemble`.
    """
    cfg = dict(DETECTION_NOISE)
    if noise_cfg:
        cfg.update(noise_cfg)
    rng = rng or np.random.default_rng(0)
    n = masks.shape[0]
    yy, xx = np.indices((DETECTOR_SIZE, DETECTOR_SIZE))
    cone = (xx - DETECTOR_CX) ** 2 + (yy - DETECTOR_CY) ** 2 <= 32.0 ** 2
    flat = np.where(cone[None, :, :], masks, -1).reshape(n, -1)
    pose_idx, pix_idx = np.nonzero(flat >= 0)
    inst = flat[pose_idx, pix_idx]
    ys, xs = np.divmod(pix_idx, DETECTOR_SIZE)
    width = int(leaf_dry.shape[0])
    keys = pose_idx.astype(np.int64) * (width + 1) + inst
    order = np.argsort(keys, kind="stable")
    keys_s, xs_s, ys_s = keys[order], xs[order], ys[order]
    pose_s, inst_s = pose_idx[order], inst[order]
    starts = np.flatnonzero(np.r_[True, keys_s[1:] != keys_s[:-1]]) if len(keys_s) else np.zeros(0, int)
    counts = np.diff(np.r_[starts, len(keys_s)]) if len(keys_s) else np.zeros(0, int)
    outputs: list[list[dict]] = [[] for _ in range(n)]
    if not len(starts):
        return outputs
    xmin = np.minimum.reduceat(xs_s, starts)
    xmax = np.maximum.reduceat(xs_s, starts)
    ymin = np.minimum.reduceat(ys_s, starts)
    ymax = np.maximum.reduceat(ys_s, starts)
    central_ids = masks[:, 32, 32]
    for g in range(len(starts)):
        if counts[g] < 2:
            continue
        pos = int(pose_s[starts[g]])
        inst_id = int(inst_s[starts[g]])
        if noise and rng.random() < cfg["drop"]:
            continue
        confidence = float(np.clip(
            cfg["conf_base"] + cfg["conf_pix"] * min(1.0, counts[g] / cfg["conf_pix_cap"])
            + (rng.normal(0.0, cfg["conf_std"]) if noise else 0.0), 0.0, 0.99))
        predicted = int(bool(leaf_dry[inst_id]) ^ ((rng.random() < cfg["class_error"]) if noise else False))
        distance = float(center_distance[pos])
        measured = None
        if inst_id == int(central_ids[pos]) and DETECTOR_CLIP_MIN <= distance <= DETECTOR_CLIP_MAX:
            measured = distance + (rng.normal(0.0, cfg["range_std"]) if noise else 0.0)
        outputs[pos].append({
            "box": [(xmin[g] + xmax[g] + 1) / 128.0, (ymin[g] + ymax[g] + 1) / 128.0,
                    (xmax[g] - xmin[g] + 1) / 64.0, (ymax[g] - ymin[g] + 1) / 64.0],
            "class_id": predicted, "confidence": confidence, "range": measured,
            "plant": int(inst_id // 8), "leaf": int(inst_id % 8)})
    for pos in range(n):
        outputs[pos].sort(key=lambda d: d["confidence"], reverse=True)
        outputs[pos] = outputs[pos][:MAX_DETECTIONS]
    return outputs


def observation_from_masks_batch(masks, center_distance, leaf_dry, *, rng=None, noise=False,
                                  noise_cfg=None):
    dets = assemble_batch(masks, center_distance, leaf_dry, rng=rng, noise=noise, noise_cfg=noise_cfg)
    return np.stack([pack(d) for d in dets], axis=0), dets
