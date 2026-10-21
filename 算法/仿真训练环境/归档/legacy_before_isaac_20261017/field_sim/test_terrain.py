import math

import numpy as np
import pytest

from field_sim.terrain import Terrain


def flat_terrain() -> Terrain:
    terrain = Terrain(20.0, seed=7, ground_amplitude=0.0, rock_density=0.0)
    terrain.rock_xy = np.array([[5.0, 5.0], [15.0, 15.0]], dtype=np.float32)
    terrain.rock_radius = np.array([0.5, 0.25], dtype=np.float32)
    terrain.rock_height = np.array([0.4, 0.2], dtype=np.float32)
    terrain.rock_blocking = np.array([True, False])
    terrain.rock_base_z = np.array([0.0, 0.0], dtype=np.float32)
    return terrain


def cast_with_direct_ground_residual(terrain, origin, directions, max_range):
    """Run the production cast using the original per-sample height path."""
    direct_residual = Terrain._ground_residual.__get__(terrain, Terrain)

    def reference_residual(o, d, ids, t, *_precomputed):
        return direct_residual(o, d, ids, t)

    terrain._ground_residual = reference_residual
    try:
        return terrain.cast(np.asarray(origin, dtype=np.float64).copy(),
                            np.asarray(directions, dtype=np.float64).copy(), max_range)
    finally:
        del terrain._ground_residual


def test_seeded_generation_and_clearance_are_reproducible():
    kwargs = dict(size=60.0, seed=18, rock_density=0.04,
                  clear_disks=[([10.0, 10.0], 3.0), ([50.0, 50.0], 4.0)])
    first = Terrain(**kwargs)
    second = Terrain(**kwargs)

    np.testing.assert_array_equal(first.wave_vectors4x2, second.wave_vectors4x2)
    np.testing.assert_array_equal(first.wave_amplitudes4, second.wave_amplitudes4)
    np.testing.assert_array_equal(first.wave_phases4, second.wave_phases4)
    np.testing.assert_array_equal(first.rock_xy, second.rock_xy)
    np.testing.assert_array_equal(first.rock_radius, second.rock_radius)
    np.testing.assert_array_equal(first.rock_height, second.rock_height)
    np.testing.assert_array_equal(first.rock_blocking, second.rock_blocking)

    for center, clear_radius in first.clear_disks:
        assert np.all(np.linalg.norm(first.rock_xy - center, axis=1)
                      >= first.rock_radius + clear_radius)
    assert len(first.rock_xy) == len(first.rock_radius) == len(first.rock_height)
    assert len(first.rock_xy) == len(first.rock_blocking)
    assert first.seed == 18


def test_ground_amplitude_slope_and_normals_stay_bounded():
    terrain = Terrain(100.0, seed=21)
    points = np.random.default_rng(4).uniform(0.0, 100.0, (5000, 2)).astype(np.float32)

    heights = terrain.height(points)
    gradients = terrain.slope(points)
    normals = terrain.normal(points)
    assert np.max(np.abs(heights - terrain.base_height)) <= terrain.ground_amplitude + 1e-6
    assert np.max(np.linalg.norm(gradients, axis=1)) <= terrain.slope_bound + 1e-6
    assert terrain.slope_bound < 0.56
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert np.all(normals[:, 2] > 0.85)


def test_surface_height_and_cast_distinguish_ground_rock_and_miss():
    terrain = flat_terrain()
    points = np.array([[5.0, 5.0], [5.5, 5.0], [5.51, 5.0], [15.0, 15.0]], dtype=np.float32)
    np.testing.assert_allclose(terrain.surface_height(points), [0.4, 0.0, 0.0, 0.2], atol=1e-6)
    assert terrain.blocking_at(points).tolist() == [True, True, False, False]

    origin = np.array([5.0, 5.0, 2.0])
    directions = np.array([
        [0.0, 0.0, -2.0],  # Normalized internally; rock apex is 0.4 m high.
        [2.0, 0.0, -2.0],  # Misses the rock and reaches flat ground.
        [1.0, 0.0, 0.0],   # No ground return before max range.
    ])
    distance, kind = terrain.cast(origin, directions, 10.0)
    np.testing.assert_allclose(distance, [1.6, 2.0 * np.sqrt(2.0), 10.0], atol=2e-5)
    np.testing.assert_array_equal(kind, [Terrain.ROCK, Terrain.GROUND, Terrain.MISS])


def test_downward_ray_starting_inside_rock_returns_positive_exit():
    terrain = flat_terrain()
    distance, kind = terrain.cast(
        np.array([5.0, 5.0, 0.2]), np.array([[0.0, 0.0, -1.0]]), 5.0
    )
    assert distance[0] == pytest.approx(0.2, abs=1e-6)
    assert kind[0] == Terrain.ROCK


def test_rough_ground_cast_finds_first_crossing_for_grazing_ray():
    terrain = Terrain(100.0, seed=8, rock_density=0.0)
    terrain.wave_vectors4x2[:] = np.array([[0.5, 0.0], [0.0, 0.2], [0.2, 0.2], [0.4, -0.2]])
    terrain.wave_amplitudes4[:] = [0.03, 0.0, 0.0, 0.0]
    terrain.wave_phases4[:] = 0.0
    terrain.slope_bound = 0.015
    origin = np.array([0.0, 50.0, 0.01])
    direction = np.array([[1.0, 0.0, -0.005]])

    distance, kind = terrain.cast(origin, direction, 5.0)
    assert kind[0] == Terrain.GROUND
    assert distance[0] < 1.0
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    hit = origin + distance[0] * direction[0]
    residual = abs(hit[2] - terrain.height(hit[:2]))
    assert residual <= 1e-4


@pytest.mark.parametrize("seed", [3, 8, 19])
def test_precomputed_ground_phase_matches_direct_reference_with_random_rays_and_rocks(seed):
    terrain = Terrain(40.0, seed=seed, ground_amplitude=0.35,
                      wavelength_range=(4.0, 10.0), rock_density=0.04)
    rng = np.random.default_rng(1000 + seed)
    count = 300
    yaw = rng.uniform(-math.pi, math.pi, count)
    pitch = rng.uniform(-0.55, 0.55, count)
    pitch[: count // 3] = rng.uniform(-0.004, 0.004, count // 3)
    cp = np.cos(pitch)
    directions = np.column_stack((cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)))

    for origin in ([20.0, 20.0, 0.4], [2.0, 2.0, 0.02], [30.0, 20.0, 0.7]):
        reference_distance, reference_kind = cast_with_direct_ground_residual(
            terrain, origin, directions, 8.0
        )
        distance, kind = terrain.cast(origin, directions.copy(), 8.0)
        np.testing.assert_array_equal(kind, reference_kind)
        np.testing.assert_allclose(distance, reference_distance, rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    ("origin_z", "expected_kind", "expected_distance"),
    [(8.0, Terrain.GROUND, 8.0), (8.001, Terrain.MISS, 8.0),
     (7.999, Terrain.GROUND, 7.999)],
)
def test_precomputed_ground_phase_preserves_max_range_boundary(
        origin_z, expected_kind, expected_distance):
    terrain = Terrain(20.0, seed=1, ground_amplitude=0.0, rock_density=0.0)
    origin = np.array([10.0, 10.0, origin_z])
    directions = np.array([[0.0, 0.0, -1.0]])

    reference_distance, reference_kind = cast_with_direct_ground_residual(
        terrain, origin, directions, 8.0
    )
    distance, kind = terrain.cast(origin, directions, 8.0)

    np.testing.assert_array_equal(kind, reference_kind)
    np.testing.assert_allclose(distance, reference_distance, rtol=0.0, atol=1e-3)
    assert kind[0] == expected_kind
    assert distance[0] == pytest.approx(expected_distance, abs=1e-3)


def _single_wave_terrain() -> Terrain:
    terrain = Terrain(40.0, seed=5, ground_amplitude=0.35,
                      wavelength_range=(4.0, 10.0), rock_density=0.0)
    terrain.wave_vectors4x2[:] = [[2.0 * np.pi / 4.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    terrain.wave_amplitudes4[:] = [0.35, 0.0, 0.0, 0.0]
    terrain.wave_phases4[:] = 0.0
    terrain.slope_bound = 0.35 * 2.0 * np.pi / 4.0
    return terrain


def test_horizontal_ray_hits_uphill_and_surface_origin_avoids_self_hit():
    terrain = _single_wave_terrain()
    origin = np.array([0.0, 10.0, 0.2])
    distance, kind = terrain.cast(origin, np.array([[1.0, 0.0, 0.0]]), 5.0)

    expected = np.arcsin(0.2 / 0.35) / (2.0 * np.pi / 4.0)
    assert kind[0] == Terrain.GROUND
    assert distance[0] == pytest.approx(expected, abs=2e-4)
    assert terrain._ground_sample_spacing == pytest.approx(4.0 / 32.0)

    on_ground, on_ground_kind = terrain.cast(
        np.array([0.0, 10.0, 0.0]), np.array([[0.0, 0.0, -1.0]]), 5.0
    )
    assert on_ground_kind[0] == Terrain.MISS
    assert on_ground[0] == pytest.approx(5.0)


def test_slightly_upward_ray_still_hits_a_rising_slope():
    terrain = _single_wave_terrain()
    origin = np.array([0.0, 10.0, 0.05])
    direction = np.array([[1.0, 0.0, 0.03]])
    distance, kind = terrain.cast(origin, direction, 5.0)

    direction /= np.linalg.norm(direction, axis=1)[:, None]
    assert kind[0] == Terrain.GROUND
    point = origin + distance[0] * direction[0]
    assert abs(point[2] - terrain.height(point[:2])) <= 1e-4
    assert distance[0] < 1.0


def test_nonmonotone_ray_keeps_first_hill_hit_when_endpoint_is_above_ground():
    terrain = _single_wave_terrain()
    origin = np.array([0.0, 10.0, 0.1])
    direction = np.array([[1.0, 0.0, -0.005]])
    max_range = 4.0  # End after two complete 4 m waves, clear of the hills.
    distance, kind = terrain.cast(origin, direction, max_range)

    direction /= np.linalg.norm(direction, axis=1)[:, None]
    endpoint = origin + max_range * direction[0]
    assert endpoint[2] - terrain.height(endpoint[:2]) > 0.0
    assert kind[0] == Terrain.GROUND
    point = origin + distance[0] * direction[0]
    assert abs(point[2] - terrain.height(point[:2])) <= 1e-4
    assert distance[0] < 2.0


def test_default_ground_amplitude_and_wavelength_range_are_configurable():
    terrain = Terrain(20.0, seed=1, rock_density=0.0)
    assert terrain.ground_amplitude == pytest.approx(0.35)
    assert terrain.wavelength_range == (4.0, 10.0)


def test_explicit_legacy_wavelength_range_preserves_seeded_angle_phase_and_rock_draws():
    current = Terrain(40.0, seed=18, rock_density=0.04,
                      wavelength_range=(4.0, 10.0))
    legacy = Terrain(40.0, seed=18, rock_density=0.04,
                     wavelength_range=(12.0, 32.0))

    current_angles = current.wave_vectors4x2 / np.linalg.norm(current.wave_vectors4x2, axis=1)[:, None]
    legacy_angles = legacy.wave_vectors4x2 / np.linalg.norm(legacy.wave_vectors4x2, axis=1)[:, None]
    np.testing.assert_allclose(current_angles, legacy_angles, atol=1e-7)
    np.testing.assert_array_equal(current.wave_phases4, legacy.wave_phases4)
    np.testing.assert_array_equal(current.wave_amplitudes4, legacy.wave_amplitudes4)
    np.testing.assert_array_equal(current.rock_xy, legacy.rock_xy)
    np.testing.assert_array_equal(current.rock_radius, legacy.rock_radius)


@pytest.mark.parametrize(
    "wavelength_range",
    [(0.5, 4.0), (4.0, 3.0), (float("nan"), 10.0), (4.0, float("inf"))],
)
def test_terrain_rejects_invalid_wavelength_ranges(wavelength_range):
    with pytest.raises(ValueError):
        Terrain(20.0, seed=1, rock_density=0.0, wavelength_range=wavelength_range)


def test_rock_size_ranges_and_blocking_class_are_configurable():
    terrain = Terrain(
        100.0, seed=3, rock_density=0.03, large_rock_fraction=0.35,
        small_rock_radius=(0.02, 0.04), small_rock_height=(0.01, 0.03),
        large_rock_radius=(0.2, 0.25), large_rock_height=(0.3, 0.35),
    )
    assert terrain.rock_blocking.any()
    assert (~terrain.rock_blocking).any()
    assert np.all((terrain.rock_radius[terrain.rock_blocking] >= 0.2)
                  & (terrain.rock_radius[terrain.rock_blocking] <= 0.25))
    assert np.all((terrain.rock_height[terrain.rock_blocking] >= 0.3)
                  & (terrain.rock_height[terrain.rock_blocking] <= 0.35))
    assert np.all((terrain.rock_radius[~terrain.rock_blocking] >= 0.02)
                  & (terrain.rock_radius[~terrain.rock_blocking] <= 0.04))
    assert np.all((terrain.rock_height[~terrain.rock_blocking] >= 0.01)
                  & (terrain.rock_height[~terrain.rock_blocking] <= 0.03))

