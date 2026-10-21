"""Conservative two-heading connectivity validation for field task resets.

The grid here is a validation aid only. It does not produce a map for the
policy and it does not plan paths. Translation is restricted to the two
longitudinal headings (0 and 90 degrees); heading changes are allowed only at
cells clear for the vehicle's circumscribed disk.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components


GRID_RESOLUTION = 0.1
CLEARANCE_MARGIN = 0.025


def _point(value, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite xy point")
    return point


def _window(center: float, extent: float, n: int) -> slice:
    """Return a safely over-inclusive local grid slice for one obstacle."""
    lo = max(0, int(math.floor((center - extent) / GRID_RESOLUTION)) - 1)
    hi = min(n, int(math.ceil((center + extent) / GRID_RESOLUTION)) + 2)
    return slice(lo, max(lo, hi))


def _obstacles(scene):
    for xy, radius in zip(scene.xy, scene.radius):
        yield np.asarray(xy, dtype=np.float64), float(radius)
    terrain = scene.terrain
    if terrain is not None:
        blocking = np.asarray(terrain.rock_blocking, dtype=bool)
        for xy, radius in zip(terrain.rock_xy[blocking], terrain.rock_radius[blocking]):
            yield np.asarray(xy, dtype=np.float64), float(radius)


def _free_masks(scene):
    """Rasterize H, V and rotate-clear masks, updating only local plant boxes."""
    cfg = scene.cfg
    n = int(math.ceil(float(cfg.size) / GRID_RESOLUTION))
    axis = (np.arange(n, dtype=np.float64) + 0.5) * GRID_RESOLUTION
    half_length = float(cfg.robot_length) / 2.0
    half_width = float(cfg.robot_width) / 2.0

    pivot_radius = math.hypot(half_length, half_width) + CLEARANCE_MARGIN
    boundary_clear = ((axis >= pivot_radius)
                      & (axis <= cfg.size - pivot_radius))
    # Keep only poses whose center is inside the common, turn-safe boundary.
    # Otherwise the rectangular corner slivers along the field edge become
    # unrotatable pose components and make every otherwise-open map fail the
    # all-components-connected requirement.
    free_h = boundary_clear[:, None] & boundary_clear[None, :]
    free_v = free_h.copy()
    clear_pivot = free_h.copy()

    for point, root_radius in _obstacles(scene):
        expanded = root_radius + CLEARANCE_MARGIN
        h_xs = _window(point[0], half_length + expanded, n)
        h_ys = _window(point[1], half_width + expanded, n)
        xs, ys = axis[h_xs], axis[h_ys]
        dx = np.maximum(np.abs(xs[None, :] - point[0]) - half_length, 0.0)
        dy = np.maximum(np.abs(ys[:, None] - point[1]) - half_width, 0.0)
        free_h[h_ys, h_xs] &= dx * dx + dy * dy > expanded * expanded

        v_xs = _window(point[0], half_width + expanded, n)
        v_ys = _window(point[1], half_length + expanded, n)
        xs, ys = axis[v_xs], axis[v_ys]
        dx = np.maximum(np.abs(xs[None, :] - point[0]) - half_width, 0.0)
        dy = np.maximum(np.abs(ys[:, None] - point[1]) - half_length, 0.0)
        free_v[v_ys, v_xs] &= dx * dx + dy * dy > expanded * expanded

        pivot_extent = root_radius + pivot_radius
        px = _window(point[0], pivot_extent, n)
        py = _window(point[1], pivot_extent, n)
        ddx = axis[px][None, :] - point[0]
        ddy = axis[py][:, None] - point[1]
        clear_pivot[py, px] &= ddx * ddx + ddy * ddy > pivot_extent * pivot_extent

    return axis, free_h, free_v, clear_pivot


def _grid_index(point: np.ndarray, size: float, n: int) -> tuple[int, int] | None:
    if np.any(point < 0.0) or np.any(point >= size):
        return None
    xy = np.floor(point / GRID_RESOLUTION).astype(np.int64)
    xy = np.minimum(xy, n - 1)
    return int(xy[1]), int(xy[0])


def validate_scene(scene, start, goal) -> dict[str, object]:
    """Check all free H/V pose components and the requested start-goal pair."""
    start_xy = _point(start, name="start")
    goal_xy = _point(goal, name="goal")
    _, free_h, free_v, clear_pivot = _free_masks(scene)
    n = free_h.shape[0]

    # H poses move only along X; V poses move only along Y.
    horizontal = np.zeros((3, 3), dtype=bool)
    horizontal[1, :] = True
    vertical = np.zeros((3, 3), dtype=bool)
    vertical[:, 1] = True
    cross = np.zeros((3, 3), dtype=bool)
    cross[1, :] = True
    cross[:, 1] = True
    labels_h, count_h = ndimage.label(free_h, structure=horizontal)
    labels_v, count_v = ndimage.label(free_v, structure=vertical)
    pivot_labels, pivot_count = ndimage.label(
        clear_pivot & (labels_h > 0) & (labels_v > 0), structure=cross
    )
    pivot_offset = int(count_h + count_v)
    node_count = pivot_offset + int(pivot_count)
    if node_count:
        orientation_edges, p_edges = [], []
        for pivot_id, region_slice in enumerate(ndimage.find_objects(pivot_labels), start=1):
            if region_slice is None:
                continue
            local_pivot = pivot_labels[region_slice] == pivot_id
            h_ids = np.unique(labels_h[region_slice][local_pivot])
            v_ids = np.unique(labels_v[region_slice][local_pivot])
            h_ids = h_ids[h_ids > 0].astype(np.int64) - 1
            v_ids = v_ids[v_ids > 0].astype(np.int64) - 1 + int(count_h)
            pivot_node = pivot_offset + pivot_id - 1
            if len(h_ids):
                orientation_edges.append(h_ids)
                p_edges.append(np.full(len(h_ids), pivot_node, dtype=np.int64))
            if len(v_ids):
                orientation_edges.append(v_ids)
                p_edges.append(np.full(len(v_ids), pivot_node, dtype=np.int64))
        if orientation_edges:
            orientation_nodes = np.concatenate(orientation_edges)
            pivot_nodes = np.concatenate(p_edges)
            rows = np.concatenate((orientation_nodes, pivot_nodes))
            cols = np.concatenate((pivot_nodes, orientation_nodes))
            graph = sparse.coo_matrix(
                (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                shape=(node_count, node_count),
            ).tocsr()
        else:
            graph = sparse.csr_matrix((node_count, node_count), dtype=np.uint8)
        component_count, node_components = connected_components(
            graph, directed=False, return_labels=True
        )
    else:
        component_count = 0
        node_components = np.zeros(0, dtype=np.int32)

    def nodes_at(point):
        ij = _grid_index(point, float(scene.cfg.size), n)
        if ij is None:
            return np.empty(0, dtype=np.int64)
        row, col = ij
        h_label, v_label = int(labels_h[row, col]), int(labels_v[row, col])
        nodes = []
        if h_label:
            nodes.append(h_label - 1)
        if v_label:
            nodes.append(int(count_h) + v_label - 1)
        pivot_label = int(pivot_labels[row, col])
        if pivot_label:
            nodes.append(pivot_offset + pivot_label - 1)
        return np.asarray(nodes, dtype=np.int64)

    start_nodes = nodes_at(start_xy)
    goal_nodes = nodes_at(goal_xy)
    start_components = (np.unique(node_components[start_nodes]) if len(start_nodes)
                        else np.empty(0, dtype=np.int32))
    goal_components = (np.unique(node_components[goal_nodes]) if len(goal_nodes)
                       else np.empty(0, dtype=np.int32))
    start_goal_connected = bool(
        len(start_components)
        and len(goal_components)
        and np.intersect1d(start_components, goal_components).size
        and scene.navigable(start_xy)
        and scene.navigable(goal_xy)
    )
    all_free_connected = bool(node_count > 0 and component_count == 1)

    if len(start_components):
        h_reachable = np.zeros(int(count_h) + 1, dtype=bool)
        v_reachable = np.zeros(int(count_v) + 1, dtype=bool)
        h_reachable[1:] = np.isin(node_components[:count_h], start_components)
        v_reachable[1:] = np.isin(
            node_components[count_h:count_h + count_v], start_components
        )
        reachable = ((labels_h > 0) & h_reachable[labels_h]) | ((labels_v > 0) & v_reachable[labels_v])
        reachable_cells = int(reachable.sum())
    else:
        reachable_cells = 0

    free_cells = int((free_h | free_v).sum())
    return {
        "connected": bool(all_free_connected and start_goal_connected),
        "all_free_connected": all_free_connected,
        "start_goal_connected": start_goal_connected,
        "free_cells": free_cells,
        "reachable_cells": reachable_cells,
        "free_pose_components": int(component_count),
        "scope": (
            "0.1 m grid; H=0 and V=90 degree rectangular vehicle poses; "
            "translations connect only along their longitudinal axis; rotations require "
            "a 4-connected clear circumscribed-disk pivot region; all free pose components and start-to-goal "
            "must connect; validation only, no policy map or path planning"
        ),
    }
