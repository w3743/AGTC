"""Procedural plant layouts for the Isaac Field simulator."""
from __future__ import annotations

import math

import numpy as np


def _xy_point(point, *, name: str) -> np.ndarray | None:
    if point is None:
        return None
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size < 2 or not np.isfinite(value[:2]).all():
        raise ValueError(f"{name} must contain finite x and y coordinates")
    return value[:2].copy()


def row_positions(
    cfg,
    rng: np.random.Generator,
    terrain=None,
    start=None,
    goal=None,
) -> np.ndarray:
    """Sample a localized, realistically spaced crop patch.

    Plants are placed in stable row-major order. The default 400-plant scene
    uses about 20 rows by 20 plants per row, 0.30 m within-row spacing and
    0.80 m between rows. A 1.5 m clear aisle is inserted after each seven
    within-row plant slots. The patch is translated randomly and rotated by
    either zero or 90 degrees; it does not imply a field-wide crop density.

    Rock intersections and the 1 m start/goal exclusion disks leave holes in
    their nominal slots. Extra slots are scanned along the same rows to fill
    the exact configured plant budget. Exhausting a bounded number of patches
    raises ``ValueError`` rather than silently returning fewer plants.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    try:
        count = int(cfg.plants)
        if count != cfg.plants or count < 0:
            raise ValueError
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("cfg.plants must be a non-negative integer") from exc
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)

    size = float(cfg.size)
    plant_spacing = float(getattr(cfg, "plant_spacing", 0.30))
    row_spacing = float(getattr(cfg, "row_spacing", 0.80))
    aisle_every = int(getattr(cfg, "row_aisle_every", 7))
    aisle_width = float(getattr(cfg, "row_aisle_width", 1.5))
    headland = float(getattr(cfg, "row_headland", 2.0))
    reserved_clearance = float(getattr(cfg, "row_start_goal_clearance", 1.0))
    max_attempts = int(getattr(cfg, "row_layout_attempts", 128))
    plant_rock_clearance = float(getattr(cfg, "plant_rock_clearance", 0.18))
    if (
        not math.isfinite(size)
        or size <= 0
        or not math.isfinite(plant_spacing)
        or plant_spacing <= 0
        or not math.isfinite(row_spacing)
        or row_spacing <= 0
        or aisle_every < 1
        or not math.isfinite(aisle_width)
        or aisle_width < 0
        or not math.isfinite(headland)
        or headland < 2.0
        or not math.isfinite(reserved_clearance)
        or reserved_clearance < 0
        or max_attempts < 1
        or not math.isfinite(plant_rock_clearance)
        or plant_rock_clearance < 0
    ):
        raise ValueError("invalid crop-row geometry settings")

    start_xy = _xy_point(start, name="start")
    goal_xy = _xy_point(goal, name="goal")
    reserved_points = [point for point in (start_xy, goal_xy) if point is not None]
    if terrain is None:
        rock_xy = np.empty((0, 2), dtype=np.float64)
        rock_radius = np.empty((0,), dtype=np.float64)
    else:
        rock_xy = np.asarray(getattr(terrain, "rock_xy", ()), dtype=np.float64).reshape(-1, 2)
        rock_radius = np.asarray(getattr(terrain, "rock_radius", ()), dtype=np.float64).reshape(-1)
        if len(rock_xy) != len(rock_radius):
            raise ValueError("terrain rock_xy and rock_radius arrays must have equal length")
        if not np.isfinite(rock_xy).all() or not np.isfinite(rock_radius).all() or np.any(rock_radius < 0):
            raise ValueError("terrain rock geometry must be finite with non-negative radii")

    rows = int(math.ceil(math.sqrt(count)))
    columns = int(math.ceil(count / rows))
    row_counts = [columns] * rows
    row_counts[-1] = count - columns * (rows - 1)
    nominal_width = (columns - 1) * plant_spacing + ((columns - 1) // aisle_every) * aisle_width
    nominal_height = (rows - 1) * row_spacing
    if nominal_width + 2 * headland >= size or nominal_height + 2 * headland >= size:
        raise ValueError(
            f"{count} plants in {rows} rows do not fit a {size:g} m field with "
            f"{headland:g} m headlands; increase field size or reduce row spacing/count"
        )

    # Local positions are centered so either orthogonal orientation has the
    # same translation bounds. Additional rock/start skips extend rows only
    # toward their positive local-x direction and remain clipped by headlands.
    local_y = (np.arange(rows, dtype=np.float64) * row_spacing) - nominal_height / 2.0
    local_x_origin = -nominal_width / 2.0
    for _ in range(max_attempts):
        rotate_90 = bool(rng.integers(0, 2))
        if rotate_90:
            half_x, half_y = nominal_height / 2.0, nominal_width / 2.0
        else:
            half_x, half_y = nominal_width / 2.0, nominal_height / 2.0
        center = np.array(
            [
                rng.uniform(headland + half_x, size - headland - half_x),
                rng.uniform(headland + half_y, size - headland - half_y),
            ],
            dtype=np.float64,
        )

        positions: list[np.ndarray] = []
        success = True
        for row_index, needed in enumerate(row_counts):
            accepted = 0
            slot_index = 0
            # A row can extend to either field edge, leaving the requested
            # headland. The attempt limit bounds work even for dense rocks.
            max_slots = int(math.ceil((size - 2 * headland) / plant_spacing)) + columns
            while accepted < needed and slot_index < max_slots:
                x = local_x_origin + slot_index * plant_spacing
                x += (slot_index // aisle_every) * aisle_width
                local = np.array([x, local_y[row_index]], dtype=np.float64)
                if rotate_90:
                    local = np.array([-local[1], local[0]], dtype=np.float64)
                point = center + local
                slot_index += 1
                if np.any(point < headland) or np.any(point > size - headland):
                    continue
                if reserved_points and any(
                    np.linalg.norm(point - reserve) < reserved_clearance + 1e-4
                    for reserve in reserved_points
                ):
                    continue
                if len(rock_xy) and np.any(
                    np.linalg.norm(rock_xy - point, axis=1)
                    <= rock_radius + plant_rock_clearance + 1e-4
                ):
                    continue
                positions.append(point.astype(np.float32))
                accepted += 1
            if accepted < needed:
                success = False
                break

        if success and len(positions) == count:
            result = np.asarray(positions, dtype=np.float32).reshape(count, 2)
            if not np.isfinite(result).all():
                raise RuntimeError("row layout produced non-finite positions")
            return result

    raise ValueError(
        f"could not place the exact {count}-plant row layout after {max_attempts} patches; "
        "check field size, rocks, start/goal clearance, and row geometry"
    )


def scatter_positions(
    cfg,
    rng: np.random.Generator,
    terrain=None,
    start=None,
    goal=None,
) -> np.ndarray:
    """Generate the former sparse, field-wide layout with an exact plant budget."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    count = int(cfg.plants)
    if count != cfg.plants or count < 0:
        raise ValueError("cfg.plants must be a non-negative integer")
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    start_xy = _xy_point(start, name="start")
    goal_xy = _xy_point(goal, name="goal")
    size = float(cfg.size)
    spacing = max(1.4, math.hypot(cfg.robot_length, cfg.robot_width) + 0.85)
    margin = max(1.2, spacing * 0.7)
    cell = float(cfg.cell)
    max_draws = int(getattr(cfg, "scatter_layout_attempts", max(1000, count * 500)))
    if margin * 2 >= size or cell <= 0 or max_draws < count:
        raise ValueError("scatter layout cannot fit the configured plant count and vehicle clearance")
    rock_xy = np.empty((0, 2), dtype=np.float64) if terrain is None else np.asarray(terrain.rock_xy, dtype=np.float64).reshape(-1, 2)
    rock_radius = np.empty((0,), dtype=np.float64) if terrain is None else np.asarray(terrain.rock_radius, dtype=np.float64).reshape(-1)
    positions: list[np.ndarray] = []
    for _ in range(max_draws):
        if len(positions) == count:
            break
        point = rng.uniform(margin, size - margin, size=2)
        point = np.floor(point / cell) * cell + cell / 2.0
        if np.any(point < margin) or np.any(point > size - margin):
            continue
        if start_xy is not None and np.linalg.norm(point - start_xy) < 1.5:
            continue
        if goal_xy is not None and np.linalg.norm(point - goal_xy) < 1.5:
            continue
        if len(rock_xy) and np.any(np.linalg.norm(rock_xy - point, axis=1) <= rock_radius + 0.15):
            continue
        if positions and np.any(np.linalg.norm(np.asarray(positions) - point, axis=1) < spacing):
            continue
        positions.append(point.astype(np.float32))
    if len(positions) != count:
        raise ValueError(
            f"could not place the exact {count}-plant scatter layout; placed {len(positions)} "
            f"after {max_draws} samples; reduce density or increase field size"
        )
    return np.asarray(positions, dtype=np.float32)


def generate_layout(
    cfg,
    rng: np.random.Generator,
    *,
    start=None,
    goal=None,
    terrain=None,
) -> np.ndarray:
    """Dispatch the configured rows/scatter layout used by ``Scene.generate``."""
    layout = str(getattr(cfg, "layout", "rows")).lower()
    if layout == "rows":
        return row_positions(cfg, rng, terrain=terrain, start=start, goal=goal)
    if layout == "scatter":
        return scatter_positions(cfg, rng, terrain=terrain, start=start, goal=goal)
    raise ValueError("cfg.layout must be 'rows' or 'scatter'")

