"""Pure plant-layout geometry tests; they do not create simulator environments."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaac_field.layouts import generate_layout, row_positions


def _cfg(**overrides):
    values = {
        "plants": 400,
        "size": 100.0,
        "cell": 0.5,
        "robot_length": 0.8,
        "robot_width": 0.47,
        "layout": "rows",
        "plant_spacing": 0.3,
        "row_spacing": 0.8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Terrain:
    def __init__(self, rock_xy, rock_radius):
        self.rock_xy = np.asarray(rock_xy, dtype=np.float64).reshape(-1, 2)
        self.rock_radius = np.asarray(rock_radius, dtype=np.float64).reshape(-1)


def test_rows_are_local_dense_patch_with_real_crop_spacing_and_orthogonal_heading():
    positions = row_positions(_cfg(), np.random.default_rng(42))
    assert positions.shape == (400, 2)
    assert positions.dtype == np.float32
    assert np.isfinite(positions).all()
    assert positions.min() >= 2.0
    assert positions.max() <= 98.0

    # Stable row-major ordering exposes the requested within-row plant spacing,
    # 1.5 m clear cross-aisle after each seven plant slots, and 0.8 m row pitch.
    deltas = np.diff(positions[:20], axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    np.testing.assert_allclose(distances[[0, 1, 4, 7, 8, 11, 14, 15, 18]], 0.3, atol=1e-5)
    np.testing.assert_allclose(distances[[6, 13]], 1.8, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(positions[20] - positions[0]), 0.8, atol=1e-5)
    assert np.allclose(deltas[:, 0], 0.0, atol=1e-5) or np.allclose(deltas[:, 1], 0.0, atol=1e-5)

    # The 400 plants occupy a small patch inside a 100x100 m field, not a
    # fabricated 400-plant field-wide density.
    patch_span = np.sort(np.ptp(positions, axis=0))
    np.testing.assert_allclose(patch_span, [8.7, 15.2], atol=1e-4)


def test_rows_avoid_rocks_and_keep_one_metre_start_goal_clearance_while_filling_budget():
    baseline = row_positions(_cfg(), np.random.default_rng(9))
    terrain = _Terrain([baseline[5]], [0.25])
    positions = row_positions(
        _cfg(),
        np.random.default_rng(9),
        terrain=terrain,
        start=baseline[0],
        goal=baseline[205],
    )

    assert positions.shape == (400, 2)
    rock_distances = np.linalg.norm(positions - terrain.rock_xy[0], axis=1)
    assert np.all(rock_distances > 0.25 + 0.18)
    assert np.all(np.linalg.norm(positions - baseline[0], axis=1) >= 1.0)
    assert np.all(np.linalg.norm(positions - baseline[205], axis=1) >= 1.0)


def test_configured_dispatch_defaults_to_rows_and_keeps_scatter_available():
    cfg = _cfg()
    first = generate_layout(cfg, np.random.default_rng(31))
    second = generate_layout(cfg, np.random.default_rng(31))
    np.testing.assert_array_equal(first, second)

    scattered = generate_layout(_cfg(plants=8, layout="scatter"), np.random.default_rng(31))
    assert scattered.shape == (8, 2)
    assert np.min(np.linalg.norm(scattered[:, None] - scattered[None, :], axis=2) + np.eye(8) * 100) >= 1.4


def test_unfillable_layout_raises_after_bounded_attempts():
    cfg = _cfg(row_layout_attempts=2)
    terrain = _Terrain([[50.0, 50.0]], [1_000.0])
    with pytest.raises(ValueError, match="exact 400-plant row layout after 2 patches"):
        row_positions(cfg, np.random.default_rng(5), terrain=terrain)

