"""Seeded, low-cost ground and rock geometry for the field simulator.

The terrain is a sum of a few long-wave sinusoids.  Its gradient has a known
upper bound, so ordinary downward rays can be bracketed directly and solved
with vectorized bisection instead of marching over every ray.  Rocks are
upper-half ellipsoids resting on the base terrain at their centers.

``cast`` returns metric ray distances and a public surface kind (0 = miss,
1 = ground, 3 = rock).  It does not expose rock indices or any plant truth.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class Terrain:
    """Procedural terrain and rock geometry.

    Args:
        size: Side length of the square field, in metres.
        seed: Seed for a private NumPy generator.  Use a dedicated terrain seed
            so adding rocks never changes the plant layout.
        rng: Optional generator; mutually exclusive with ``seed``.
        clear_disks: ``[(xy, radius), ...]`` areas in which rock centers are
            excluded, usually disks around the start and goal.
        ground_amplitude: Sum-of-wave amplitude budget in metres.  Since each
            component amplitude is nonnegative and sums to this value,
            ``height`` stays within this distance of ``base_height``.
        wavelength_range: Inclusive lower/upper wavelength bounds in metres.
        rock_density: Expected number of rocks per square metre.
        large_rock_fraction: Fraction of rocks drawn from the blocking class.
        small_rock_radius / small_rock_height: Range for traversable rocks.
        large_rock_radius / large_rock_height: Range for blocking rocks.

    The four wave arrays are public for visualization/serialization:
    ``wave_vectors4x2`` has angular wave vectors in rad/m, while
    ``wave_amplitudes4`` and ``wave_phases4`` contain metres and radians.
    """

    GROUND = np.uint8(1)
    ROCK = np.uint8(3)
    MISS = np.uint8(0)
    _GROUND_DISTANCE_TOLERANCE = 1e-4
    _GROUND_SAMPLE_SPACING = 0.5

    def __init__(
        self,
        size: float,
        *,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
        clear_disks: Sequence[tuple[Sequence[float], float]] | None = None,
        base_height: float = 0.0,
        ground_amplitude: float = 0.35,
        wavelength_range: tuple[float, float] = (4.0, 10.0),
        rock_density: float = 0.01,
        large_rock_fraction: float = 0.2,
        small_rock_radius: tuple[float, float] = (0.04, 0.10),
        small_rock_height: tuple[float, float] = (0.03, 0.08),
        large_rock_radius: tuple[float, float] = (0.15, 0.30),
        large_rock_height: tuple[float, float] = (0.15, 0.30),
    ) -> None:
        self.size = float(size)
        self.base_height = float(base_height)
        self.ground_amplitude = float(ground_amplitude)
        try:
            self.wavelength_range = tuple(map(float, wavelength_range))
        except (TypeError, ValueError) as exc:
            raise ValueError("wavelength_range must be a positive (min, max) pair") from exc
        self.rock_density = float(rock_density)
        self.large_rock_fraction = float(large_rock_fraction)
        self.small_rock_radius = tuple(map(float, small_rock_radius))
        self.small_rock_height = tuple(map(float, small_rock_height))
        self.large_rock_radius = tuple(map(float, large_rock_radius))
        self.large_rock_height = tuple(map(float, large_rock_height))
        self.clear_disks = self._parse_clear_disks(clear_disks)
        self.seed = seed if isinstance(seed, (int, np.integer)) else None
        self._validate_parameters()
        if rng is not None and seed is not None:
            raise ValueError("pass either seed or rng, not both")
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        # Four seeded sine waves make smooth, repeatable relief. A 35 cm total
        # amplitude budget over 4 m minimum wavelengths bounds grade by 0.55.
        wave_count = 4
        angle = self.rng.uniform(-math.pi, math.pi, wave_count)
        wavelength = self.rng.uniform(*self.wavelength_range, wave_count)
        self.wave_vectors4x2 = (
            np.stack((np.cos(angle), np.sin(angle)), axis=1)
            * (2.0 * math.pi / wavelength)[:, None]
        ).astype(np.float32)
        weights = self.rng.uniform(0.5, 1.5, wave_count)
        self.wave_amplitudes4 = (self.ground_amplitude * weights / weights.sum()).astype(np.float32)
        self.wave_phases4 = self.rng.uniform(-math.pi, math.pi, wave_count).astype(np.float32)
        self.slope_bound = float(np.sum(
            self.wave_amplitudes4 * np.linalg.norm(self.wave_vectors4x2, axis=1)
        ))
        self._ground_sample_spacing = min(
            self._GROUND_SAMPLE_SPACING, self.wavelength_range[0] / 32.0
        )

        self._generate_rocks()

    def _validate_parameters(self) -> None:
        if not math.isfinite(self.size) or self.size <= 0:
            raise ValueError("size must be finite and positive")
        if not math.isfinite(self.base_height):
            raise ValueError("base_height must be finite")
        if not math.isfinite(self.ground_amplitude) or self.ground_amplitude < 0:
            raise ValueError("ground_amplitude must be finite and nonnegative")
        if (len(self.wavelength_range) != 2
                or not all(math.isfinite(v) for v in self.wavelength_range)
                or self.wavelength_range[0] < 1.0
                or self.wavelength_range[1] < self.wavelength_range[0]):
            raise ValueError("wavelength_range must be finite with 1 <= min <= max metres")
        if not math.isfinite(self.rock_density) or self.rock_density < 0:
            raise ValueError("rock_density must be finite and nonnegative")
        if not math.isfinite(self.large_rock_fraction) or not 0 <= self.large_rock_fraction <= 1:
            raise ValueError("large_rock_fraction must be in [0, 1]")
        for name, bounds in (
            ("small_rock_radius", self.small_rock_radius),
            ("small_rock_height", self.small_rock_height),
            ("large_rock_radius", self.large_rock_radius),
            ("large_rock_height", self.large_rock_height),
        ):
            if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError(f"{name} must be a positive (min, max) pair")

    @staticmethod
    def _parse_clear_disks(
        clear_disks: Sequence[tuple[Sequence[float], float]] | None,
    ) -> tuple[tuple[np.ndarray, float], ...]:
        parsed: list[tuple[np.ndarray, float]] = []
        for center, radius in clear_disks or ():
            xy = np.asarray(center, dtype=np.float64).reshape(-1)
            radius = float(radius)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not math.isfinite(radius) or radius < 0:
                raise ValueError("clear_disks entries must be (finite xy, nonnegative radius)")
            parsed.append((xy, radius))
        return tuple(parsed)

    def _generate_rocks(self) -> None:
        expected = self.rock_density * self.size * self.size
        count = int(self.rng.poisson(expected)) if expected else 0
        is_large = self.rng.random(count) < self.large_rock_fraction
        radius = np.empty(count, dtype=np.float32)
        height = np.empty(count, dtype=np.float32)
        for mask, radius_range, height_range in (
            (is_large, self.large_rock_radius, self.large_rock_height),
            (~is_large, self.small_rock_radius, self.small_rock_height),
        ):
            n = int(mask.sum())
            radius[mask] = self.rng.uniform(*radius_range, n)
            height[mask] = self.rng.uniform(*height_range, n)

        # Rejection placement avoids overlapping rock footprints and respects
        # start/goal clearance disks. At the default density, rejection is rare.
        accepted_xy: list[np.ndarray] = []
        accepted_radius: list[float] = []
        accepted_indices: list[int] = []
        edge = min(max((*self.small_rock_radius, *self.large_rock_radius)), self.size / 2.0)
        low, high = edge, self.size - edge
        for i in range(count):
            r = float(radius[i])
            for _ in range(200):
                xy = self.rng.uniform(low, high, 2) if high > low else np.full(2, self.size / 2.0)
                if any(np.linalg.norm(xy - center) < r + clear_r for center, clear_r in self.clear_disks):
                    continue
                if accepted_xy and any(
                    np.linalg.norm(xy - other) < r + other_r
                    for other, other_r in zip(accepted_xy, accepted_radius)
                ):
                    continue
                accepted_xy.append(xy.astype(np.float32))
                accepted_radius.append(r)
                accepted_indices.append(i)
                break

        # Keep generated geometry compact; if clearance prevented placement,
        # retain the class attributes corresponding to accepted rocks.
        self.rock_xy = np.asarray(accepted_xy, dtype=np.float32).reshape(-1, 2)
        accepted = np.asarray(accepted_indices, dtype=np.intp)
        self.rock_radius = radius[accepted].copy()
        self.rock_height = height[accepted].copy()
        self.rock_blocking = is_large[accepted].astype(bool, copy=True)
        self.rock_base_z = self.height(self.rock_xy).astype(np.float32, copy=False)

    @staticmethod
    def _xy_array(xy: np.ndarray | Sequence[float]) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(xy, dtype=np.float32)
        if arr.shape == () or arr.shape[-1] != 2:
            raise ValueError("xy must have final dimension 2")
        if not np.all(np.isfinite(arr)):
            raise ValueError("xy must be finite")
        return arr.reshape(-1, 2), arr.shape[:-1]

    def height(self, xy: np.ndarray | Sequence[float]) -> np.ndarray | float:
        """Base ground height (excluding stones), preserving leading shape."""
        points, shape = self._xy_array(xy)
        phase = points @ self.wave_vectors4x2.T + self.wave_phases4
        values = self.base_height + np.sin(phase) @ self.wave_amplitudes4
        values = values.reshape(shape)
        return float(values) if not shape else values

    def slope(self, xy: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return ``(dh/dx, dh/dy)`` for each point."""
        points, shape = self._xy_array(xy)
        phase = points @ self.wave_vectors4x2.T + self.wave_phases4
        grad = (np.cos(phase) * self.wave_amplitudes4) @ self.wave_vectors4x2
        return grad.reshape((*shape, 2))

    def normal(self, xy: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return upward unit normals to the base ground."""
        gradient = self.slope(xy)
        normal = np.concatenate((-gradient, np.ones((*gradient.shape[:-1], 1), dtype=np.float32)), axis=-1)
        return normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    def surface_height(self, xy: np.ndarray | Sequence[float]) -> np.ndarray | float:
        """Ground or stone-cap height, whichever is higher at each point."""
        points, shape = self._xy_array(xy)
        ground = np.asarray(self.height(points), dtype=np.float32).reshape(-1)
        top = ground.copy()
        for rock_id in self._rock_candidates(points):
            center = self.rock_xy[rock_id]
            radius = self.rock_radius[rock_id]
            rock_h = self.rock_height[rock_id]
            base_z = self.rock_base_z[rock_id]
            delta = points - center
            rho2 = np.sum(delta * delta, axis=1) / float(radius * radius)
            inside = rho2 <= 1.0
            cap = base_z + rock_h * np.sqrt(np.maximum(0.0, 1.0 - rho2))
            top[inside] = np.maximum(top[inside], cap[inside])
        top = top.reshape(shape)
        return float(top) if not shape else top

    def blocking_at(self, xy: np.ndarray | Sequence[float]) -> np.ndarray | bool:
        """Whether points overlap the horizontal footprint of a blocking rock."""
        points, shape = self._xy_array(xy)
        blocked = np.zeros(len(points), dtype=bool)
        for rock_id in self._rock_candidates(points, blocking_only=True):
            center = self.rock_xy[rock_id]
            radius = self.rock_radius[rock_id]
            delta = points - center
            blocked |= np.sum(delta * delta, axis=1) <= radius * radius
        blocked = blocked.reshape(shape)
        return bool(blocked) if not shape else blocked

    def _rock_candidates(self, points: np.ndarray, *, blocking_only: bool = False) -> np.ndarray:
        """Cull rocks whose circles cannot meet the query points' AABB."""
        if not len(points) or not len(self.rock_xy):
            return np.empty(0, dtype=np.intp)
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        centers = self.rock_xy
        nearest = np.maximum(np.maximum(lower[None, :] - centers, 0.0), centers - upper[None, :])
        intersects = np.sum(nearest * nearest, axis=1) <= self.rock_radius * self.rock_radius
        if blocking_only:
            intersects &= self.rock_blocking
        return np.flatnonzero(intersects)

    def cast(
        self,
        origin: np.ndarray | Sequence[float],
        directions: np.ndarray | Sequence[Sequence[float]],
        max_range: float,
        *,
        ray_chunk_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest ray distance and surface kind for each ray.

        Direction vectors are normalized internally, so each returned value
        is a distance in metres even if callers provide non-unit vectors.
        Misses have distance ``max_range`` and kind ``0``; ground and rocks
        use kinds ``1`` and ``3``. Monotone descending ground rays use a
        bounded bracket and bisection. Other rays that can meet the terrain
        height slab are sampled at horizontal spacing no greater than
        ``min(0.5 m, minimum_wavelength / 32)`` before bisection; this finite
        sampling can miss tangencies or multiple crossings within one sample
        interval. Rocks use analytic ellipsoid
        roots and are filtered against the base ground, which occludes the
        millimetric mismatch between the cap base and local undulating ground.
        """
        o = np.asarray(origin, dtype=np.float64).reshape(-1)
        d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        max_range = float(max_range)
        if o.shape != (3,) or not np.all(np.isfinite(o)):
            raise ValueError("origin must be a finite 3-vector")
        if not np.all(np.isfinite(d)):
            raise ValueError("directions must be finite")
        if not math.isfinite(max_range) or max_range <= 0:
            raise ValueError("max_range must be finite and positive")
        if ray_chunk_size < 1:
            raise ValueError("ray_chunk_size must be positive")
        lengths = np.linalg.norm(d, axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("directions must have nonzero length")
        d /= lengths[:, None]

        distance = np.full(len(d), max_range, dtype=np.float32)
        kind = np.full(len(d), self.MISS, dtype=np.uint8)
        if len(d):
            self._cast_ground(o, d, np.arange(len(d)), max_range, distance, kind)
        if len(self.rock_xy) and len(d):
            self._cast_rocks(o, d, max_range, distance, kind, ray_chunk_size)
        return distance, kind

    def _ground_residual(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        ids: np.ndarray,
        t: np.ndarray,
        phase_origin: np.ndarray | None = None,
        phase_rate: np.ndarray | None = None,
    ) -> np.ndarray:
        if phase_origin is None or phase_rate is None:
            # Keep the direct evaluation path as a small numerical reference;
            # casts pass precomputed float32 phase terms below.
            xy = origin[:2] + t[:, None] * directions[ids, :2]
            ground = np.asarray(self.height(xy), dtype=np.float64).reshape(-1)
        else:
            # Terrain.height quantizes xy, wave vectors, phases, and amplitudes
            # to float32. Keep that precision while using phase linearity along
            # each ray to avoid rebuilding xy and multiplying by wave vectors
            # in every bracket/sampling/bisection iteration.
            phase = phase_origin[None, :] + t.astype(np.float32, copy=False)[:, None] * phase_rate[ids]
            ground = np.asarray(
                self.base_height + np.sin(phase) @ self.wave_amplitudes4,
                dtype=np.float64,
            ).reshape(-1)
        return origin[2] + t * directions[ids, 2] - ground

    def _cast_ground(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        ids: np.ndarray,
        max_range: float,
        distance: np.ndarray,
        kind: np.ndarray,
    ) -> None:
        if not len(ids):
            return

        start_residual = float(origin[2] - self.height(origin[:2]))
        # A ray starting under the ground is occluded. A ray starting exactly
        # on it is allowed to travel onward, but t=0 itself is never a return.
        if start_residual < -1e-6:
            return

        ray_directions = directions[ids]
        dz = ray_directions[:, 2]
        horizontal_speed = np.linalg.norm(ray_directions[:, :2], axis=1)
        lower = self.base_height - self.ground_amplitude
        upper = self.base_height + self.ground_amplitude
        lo = np.zeros(len(ids), dtype=np.float64)
        hi = np.full(len(ids), max_range, dtype=np.float64)
        moving = np.abs(dz) > 1e-12
        if np.any(moving):
            t0 = (lower - origin[2]) / dz[moving]
            t1 = (upper - origin[2]) / dz[moving]
            lo[moving] = np.maximum(0.0, np.minimum(t0, t1))
            hi[moving] = np.minimum(max_range, np.maximum(t0, t1))
        horizontal = ~moving
        outside_slab = horizontal & ((origin[2] < lower) | (origin[2] > upper))
        hi[outside_slab] = -1.0
        candidate = (lo <= hi) & (hi > 1e-5)
        work = ids[candidate]
        if not len(work):
            return
        lo = lo[candidate]
        hi = hi[candidate]
        horizontal_speed = horizontal_speed[candidate]
        phase_origin = (np.asarray(origin[:2], dtype=np.float32) @ self.wave_vectors4x2.T
                        + self.wave_phases4)
        phase_rate = directions[:, :2].astype(np.float32) @ self.wave_vectors4x2.T
        flo = self._ground_residual(origin, directions, work, lo, phase_origin, phase_rate)
        fhi = self._ground_residual(origin, directions, work, hi, phase_origin, phase_rate)
        dz = dz[candidate]
        monotone = dz < -self.slope_bound * horizontal_speed - 1e-9

        root = np.full(len(work), np.inf, dtype=np.float64)
        # The gradient bound proves that the residual strictly decreases on
        # these rays, so only they can use the endpoint bracket fast path.
        simple = np.flatnonzero(
            monotone & (flo >= -1e-10) & (fhi <= 1e-10)
            & ~((lo <= 1e-5) & (flo <= 1e-6))
        )
        if len(simple):
            a, b = lo[simple].copy(), hi[simple].copy()
            steps = self._bisection_steps(float(np.max(b - a, initial=0.0)))
            for _ in range(steps):
                mid = 0.5 * (a + b)
                fm = self._ground_residual(
                    origin, directions, work[simple], mid, phase_origin, phase_rate
                )
                upper_half = fm > 0.0
                a[upper_half] = mid[upper_half]
                b[~upper_half] = mid[~upper_half]
            root[simple] = b

        # Near-grazing, horizontal, and upward rays can have a non-monotone
        # residual. Keep every such ray that intersects the global height slab;
        # an endpoint test would discard rays that hit a hill and then leave it.
        grazing = np.flatnonzero(~monotone)
        if len(grazing):
            span = hi[grazing] - lo[grazing]
            sample_count = np.maximum(
                1,
                np.ceil(span * horizontal_speed[grazing] / self._ground_sample_spacing).astype(np.int64),
            )
            active = np.ones(len(grazing), dtype=bool)
            prev_t = lo[grazing].copy()
            prev_f = flo[grazing].copy()
            bracket_lo = np.zeros(len(grazing), dtype=np.float64)
            bracket_hi = np.zeros(len(grazing), dtype=np.float64)
            boundary_hit = (lo[grazing] > 1e-5) & (prev_f <= 0.0)
            root[grazing[boundary_hit]] = lo[grazing[boundary_hit]]
            active[boundary_hit] = False
            for step in range(1, int(np.max(sample_count, initial=0)) + 1):
                sampling = np.flatnonzero(active & (sample_count >= step))
                if not len(sampling):
                    break
                cur_t = (lo[grazing[sampling]]
                         + span[sampling] * (step / sample_count[sampling]))
                cur_f = self._ground_residual(
                    origin, directions, work[grazing[sampling]], cur_t,
                    phase_origin, phase_rate,
                )
                crossed = (prev_f[sampling] > 0.0) & (cur_f <= 0.0)
                found_now = sampling[crossed]
                bracket_lo[found_now] = prev_t[found_now]
                bracket_hi[found_now] = cur_t[crossed]
                active[found_now] = False
                continuing = sampling[~crossed]
                prev_t[continuing] = cur_t[~crossed]
                prev_f[continuing] = cur_f[~crossed]
            found = np.flatnonzero(~active & ~boundary_hit)
            if len(found):
                a, b = bracket_lo[found], bracket_hi[found]
                ray_ids = work[grazing[found]]
                steps = self._bisection_steps(float(np.max(b - a, initial=0.0)))
                for _ in range(steps):
                    mid = 0.5 * (a + b)
                    fm = self._ground_residual(
                        origin, directions, ray_ids, mid, phase_origin, phase_rate
                    )
                    upper_half = fm > 0.0
                    a[upper_half] = mid[upper_half]
                    b[~upper_half] = mid[~upper_half]
                root[grazing[found]] = b

        hit = np.isfinite(root) & (root > 1e-5) & (root <= max_range + 1e-7)
        target = work[hit]
        distance[target] = root[hit].astype(np.float32)
        kind[target] = self.GROUND

    def _bisection_steps(self, max_width: float) -> int:
        """Iterations needed to bound range error by 0.1 mm."""
        if max_width <= self._GROUND_DISTANCE_TOLERANCE:
            return 0
        return int(math.ceil(math.log2(max_width / self._GROUND_DISTANCE_TOLERANCE)))

    def _cast_rocks(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        max_range: float,
        distance: np.ndarray,
        kind: np.ndarray,
        chunk_size: int,
    ) -> None:
        # A radius around the origin bounds all rock centers that any ray can
        # reach; per-ray cylinder roots below remove the remaining false pairs.
        reach = max_range + float(np.max(self.rock_radius, initial=0.0))
        near_ids = np.flatnonzero(np.sum((self.rock_xy - origin[:2]) ** 2, axis=1) <= reach * reach)
        if not len(near_ids):
            return
        centers = self.rock_xy[near_ids].astype(np.float64)
        radii = self.rock_radius[near_ids].astype(np.float64)
        heights = self.rock_height[near_ids].astype(np.float64)
        bases = self.rock_base_z[near_ids].astype(np.float64)
        ox, oy, oz = origin
        eps = 1e-8

        for start in range(0, len(directions), chunk_size):
            end = min(len(directions), start + chunk_size)
            ray_ids = np.arange(start, end)
            rays = directions[start:end]
            dx, dy, dz = rays[:, 0:1], rays[:, 1:2], rays[:, 2:3]
            rx = ox - centers[:, 0][None, :]
            ry = oy - centers[:, 1][None, :]
            a_xy = dx * dx + dy * dy
            b_xy = 2.0 * (dx * rx + dy * ry)
            c_xy = rx * rx + ry * ry - radii[None, :] ** 2
            disc_xy = b_xy * b_xy - 4.0 * a_xy * c_xy
            square_xy = np.sqrt(np.maximum(disc_xy, 0.0))
            denom_xy = np.where(a_xy > eps, 2.0 * a_xy, 1.0)
            enter = (-b_xy - square_xy) / denom_xy
            leave = (-b_xy + square_xy) / denom_xy
            vertical_inside = (a_xy <= eps) & (c_xy <= 0.0)
            enter = np.where(vertical_inside, 0.0, enter)
            leave = np.where(vertical_inside, max_range, leave)
            cyl_valid = (vertical_inside | (disc_xy >= 0.0)) & (leave >= 1e-6) & (enter <= max_range)

            # Full ellipsoid quadratic, then retain only its upper half.
            radius2 = radii[None, :] ** 2
            height2 = heights[None, :] ** 2
            zrel = oz - bases[None, :]
            qa = a_xy / radius2 + dz * dz / height2
            qb = 2.0 * (dx * rx + dy * ry) / radius2 + 2.0 * dz * zrel / height2
            qc = (rx * rx + ry * ry) / radius2 + zrel * zrel / height2 - 1.0
            disc = qb * qb - 4.0 * qa * qc
            root_disc = np.sqrt(np.maximum(disc, 0.0))
            t0 = (-qb - root_disc) / (2.0 * qa)
            t1 = (-qb + root_disc) / (2.0 * qa)
            # If a ray starts inside the half-ellipsoid and exits through its
            # flat base, that disk is also a valid positive exit surface.
            tbase = np.divide(
                bases[None, :] - oz, dz,
                out=np.full_like(dz + np.zeros_like(c_xy), np.inf),
                where=dz < -1e-8,
            )
            safe_tbase = np.where(np.isfinite(tbase), tbase, 0.0)
            tx = ox + safe_tbase * dx
            ty = oy + safe_tbase * dy
            base_inside = ((tx - centers[:, 0][None, :]) ** 2
                           + (ty - centers[:, 1][None, :]) ** 2 <= radius2 + 1e-9)

            candidates = np.stack((t0, t1, tbase), axis=2)
            candidate_valid = np.stack((
                (disc >= 0.0) & (t0 > 1e-6) & (t0 >= np.maximum(enter, 0.0) - 1e-6)
                & (t0 <= leave + 1e-6) & (t0 <= max_range)
                & (oz + t0 * dz >= bases[None, :] - 1e-6),
                (disc >= 0.0) & (t1 > 1e-6) & (t1 >= np.maximum(enter, 0.0) - 1e-6)
                & (t1 <= leave + 1e-6) & (t1 <= max_range)
                & (oz + t1 * dz >= bases[None, :] - 1e-6),
                (tbase > 1e-6) & (tbase <= max_range) & base_inside,
            ), axis=2)
            candidate_valid &= cyl_valid[:, :, None]
            candidates = np.where(candidate_valid, candidates, np.inf)
            best_pair = np.min(candidates, axis=2)

            hit_rows, hit_rocks = np.where(np.isfinite(best_pair))
            if not len(hit_rows):
                continue
            hit_t = best_pair[hit_rows, hit_rocks]
            hit_x = ox + hit_t * rays[hit_rows, 0]
            hit_y = oy + hit_t * rays[hit_rows, 1]
            hit_z = oz + hit_t * rays[hit_rows, 2]
            local_ground = np.asarray(self.height(np.column_stack((hit_x, hit_y))), dtype=np.float64)
            visible = hit_z >= local_ground - 1e-6
            hit_rows, hit_rocks, hit_t = hit_rows[visible], hit_rocks[visible], hit_t[visible]
            if not len(hit_rows):
                continue
            global_rays = ray_ids[hit_rows]
            # Rocks win equality: their underside and adjacent ground share a
            # boundary, while their visible top must remain a rock return.
            closer = hit_t <= distance[global_rays].astype(np.float64) + 1e-6
            global_rays, hit_t = global_rays[closer], hit_t[closer]
            distance[global_rays] = hit_t.astype(np.float32)
            kind[global_rays] = self.ROCK

