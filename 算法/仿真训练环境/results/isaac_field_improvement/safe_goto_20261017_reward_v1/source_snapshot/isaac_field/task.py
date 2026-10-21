"""Task state and field geometry shared by simulator-backed environments."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy import ndimage

from .layouts import generate_layout
from .navigation import validate_scene as validate_navigation_scene
from .terrain import Terrain


# Snapshot of LightStereo's cropped, rectified portrait geometry. The source
# is models/portrait_geometry.npz (SHA-256 prefix bcb9e5c82b93); keep the
# simulator independent of the external project/package at runtime.
DEPTH_NATIVE_WIDTH = 352
DEPTH_NATIVE_HEIGHT = 640
DEPTH_NATIVE_INTRINSICS = np.array([
    [393.29918354972216, 0.0, 177.54501597086588],
    [0.0, 393.29918354972216, 323.73144276936847],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DEPTH_GEOMETRY_ID = "bcb9e5c82b93"
DEPTH_GEOMETRY_SOURCE = "算法/视觉算法/软件/LightStereo源码/models/portrait_geometry.npz"


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Config:
    size: float = 100.0
    cell: float = 0.5
    plants: int = 400
    dry_fraction: float = 0.25
    leaves_per_plant: int = 8
    layout: str = "rows"
    row_spacing: float = 0.8
    plant_spacing: float = 0.3
    plant_height: float = 1.0
    plant_height_spread: float = 0.05
    stem_radius: float = 0.012
    robot_length: float = 0.8
    robot_width: float = 0.3
    max_speed: float = 15 / 3.6
    max_turn_rate: float = 1.5
    acceleration: float = 1.5  # assumption; configurable, not a measured chassis model
    angular_acceleration: float = 4.0
    camera_height: float = 0.4
    pitch_min: float = math.radians(-25)
    pitch_max: float = math.radians(60)
    gimbal_yaw_rate: float = 1.8
    gimbal_pitch_rate: float = 1.2
    dt: float = 0.05
    max_time: float = 900.0
    lidar_model: str = "X2"
    lidar_rays: int = 400
    lidar_range: float = 8.0
    lidar_min_range: float = 0.12
    lidar_relative_error: float = 0.05
    lidar_period: float = 0.1
    lidar_latency: float = 0.03
    depth_height: int = 160
    depth_width: int = 88
    depth_min_range: float = 0.3
    depth_range: float = 5.0
    depth_period: float = 0.4
    depth_latency: float = 0.3
    camera_range: float = 7.0
    camera_period: float = 0.2
    camera_latency: float = 0.12
    max_detections: int = 16
    position_noise: float = 0.025
    range_noise: float = 0.01
    detection_drop: float = 0.08
    class_error: float = 0.015
    goal_tolerance: float = 0.4
    goal_hold: float = 0.3
    damage_penalty: float = 4.0
    collision_penalty: float = 0.1
    time_penalty: float = 0.01
    progress_weight: float = 0.15
    stop_on_damage: bool = False
    validate_scene: bool = True
    terrain_enabled: bool = True
    terrain_amplitude: float = 0.35
    # The one-metre floor keeps the ray sampling interval practical at λ/32.
    terrain_wavelength_min: float = 4.0
    terrain_wavelength_max: float = 10.0
    rock_density: float = 0.01
    large_rock_fraction: float = 0.2
    terrain_seed: int | None = None
    # Optional per-goal progress budget. None preserves progress_weight semantics.
    # Keep this at the end to preserve existing positional Config arguments.
    goal_progress_budget: float | None = None

    def __post_init__(self):
        if self.size < 6 or self.cell <= 0 or self.dt <= 0 or self.max_time <= 0:
            raise ValueError("size >= 6, cell/dt/max_time > 0 required")
        if self.plants < 0 or not 0 <= self.dry_fraction <= 1:
            raise ValueError("invalid plant count or dry fraction")
        if self.layout not in ("rows", "scatter"):
            raise ValueError("layout must be 'rows' or 'scatter'")
        if (not math.isfinite(self.row_spacing) or self.row_spacing <= 0
                or not math.isfinite(self.plant_spacing) or self.plant_spacing <= 0):
            raise ValueError("row and plant spacing must be finite and positive")
        if not math.isfinite(self.plant_height) or self.plant_height <= 0:
            raise ValueError("plant height must be finite and positive")
        if (not math.isfinite(self.plant_height_spread)
                or not 0 <= self.plant_height_spread < self.plant_height):
            raise ValueError("plant height spread must be finite and in [0, plant height)")
        if not math.isfinite(self.stem_radius) or self.stem_radius <= 0:
            raise ValueError("stem radius must be finite and positive")
        if self.robot_length <= 0 or self.robot_width <= 0 or self.max_speed <= 0:
            raise ValueError("vehicle dimensions/speed must be positive")
        if self.depth_height < 2 or self.depth_width < 2 or self.lidar_rays < 8:
            raise ValueError("sensor sampling dimensions too small")
        if self.pitch_min >= self.pitch_max or self.camera_height <= 0:
            raise ValueError("invalid camera height or pitch limits")
        if self.leaves_per_plant < 1 or self.max_detections < 1:
            raise ValueError("leaf/detection count must be positive")
        if min(self.lidar_period, self.depth_period, self.camera_period) <= 0:
            raise ValueError("sensor periods must be positive")
        if min(self.lidar_latency, self.depth_latency, self.camera_latency) < 0:
            raise ValueError("sensor latency cannot be negative")
        if (not math.isfinite(self.depth_min_range)
                or not 0 < self.depth_min_range < self.depth_range):
            raise ValueError("depth min range must be finite, positive, and below max range")
        if (not math.isfinite(self.lidar_min_range)
                or not 0 < self.lidar_min_range < self.lidar_range):
            raise ValueError("lidar min range must be finite, positive, and below max range")
        if (not math.isfinite(self.lidar_relative_error)
                or not 0 <= self.lidar_relative_error < 1):
            raise ValueError("lidar relative error must be finite and in [0, 1)")
        if (not math.isfinite(self.terrain_amplitude)
                or self.terrain_amplitude < 0):
            raise ValueError("terrain amplitude must be finite and non-negative")
        if (not math.isfinite(self.terrain_wavelength_min)
                or not math.isfinite(self.terrain_wavelength_max)
                or self.terrain_wavelength_min < 1.0
                or self.terrain_wavelength_max < self.terrain_wavelength_min):
            raise ValueError("terrain wavelengths must be finite with 1 <= min <= max metres")
        if (not math.isfinite(self.rock_density) or self.rock_density < 0
                or not math.isfinite(self.large_rock_fraction)
                or not 0 <= self.large_rock_fraction <= 1):
            raise ValueError("rock density must be non-negative and large-rock fraction in [0, 1]")
        if self.terrain_seed is not None and (
                isinstance(self.terrain_seed, bool) or int(self.terrain_seed) != self.terrain_seed
                or self.terrain_seed < 0):
            raise ValueError("terrain seed must be a non-negative integer or None")
        if not math.isfinite(self.collision_penalty) or self.collision_penalty < 0:
            raise ValueError("collision penalty must be finite and non-negative")
        if not math.isfinite(self.goal_tolerance) or self.goal_tolerance <= 0:
            raise ValueError("goal tolerance must be finite and positive")
        if not math.isfinite(self.goal_hold) or self.goal_hold <= 0:
            raise ValueError("goal hold must be finite and positive")
        for name, value in (("damage penalty", self.damage_penalty),
                            ("time penalty", self.time_penalty),
                            ("progress weight", self.progress_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.goal_progress_budget is not None and (
                isinstance(self.goal_progress_budget, bool)
                or not math.isfinite(self.goal_progress_budget)
                or self.goal_progress_budget < 0):
            raise ValueError("goal progress budget must be finite and non-negative or None")


def _terrain_seed_from_rng(rng: np.random.Generator) -> int:
    """Derive a stable terrain seed without drawing from the plant RNG."""
    state = repr(rng.bit_generator.state).encode("utf-8")
    return int.from_bytes(hashlib.sha256(state).digest()[:8], "little")


def _make_terrain(cfg: Config, seed: int, clear_disks) -> Terrain:
    return Terrain(
        cfg.size,
        seed=int(seed),
        clear_disks=clear_disks,
        ground_amplitude=cfg.terrain_amplitude,
        wavelength_range=(cfg.terrain_wavelength_min, cfg.terrain_wavelength_max),
        rock_density=cfg.rock_density,
        large_rock_fraction=cfg.large_rock_fraction,
    )


class Scene:
    """Truth geometry is private to simulation/evaluation, never an observation."""
    _ROCK_HIT = -1_000_000

    def __init__(self, cfg: Config, xy: np.ndarray, dry: np.ndarray, rng: np.random.Generator,
                 *, terrain: Terrain | None = None, terrain_seed: int | None = None):
        self.cfg = cfg
        self.xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        self.targets = np.asarray(dry, dtype=bool).reshape(-1)
        if len(self.xy) != len(self.targets):
            raise ValueError("position and target counts differ")
        if terrain is not None or cfg.terrain_enabled:
            if terrain is None:
                seed = cfg.terrain_seed if terrain_seed is None else terrain_seed
                if seed is None:
                    seed = _terrain_seed_from_rng(rng)
                clear_disks = [(p, 0.5) for p in self.xy]
                terrain = _make_terrain(cfg, seed, clear_disks)
            self.terrain = terrain
            chosen_seed = terrain_seed if terrain_seed is not None else terrain.seed
            self.terrain_seed = int(chosen_seed) if chosen_seed is not None else None
        else:
            self.terrain = None
            self.terrain_seed = None
        n, k = len(self.xy), cfg.leaves_per_plant
        self.root_z = (np.asarray(self.terrain.height(self.xy), dtype=np.float32).reshape(-1)
                       if self.terrain is not None else np.zeros(n, dtype=np.float32))
        self.radius = rng.uniform(0.09, 0.15, n).astype(np.float32)
        self.stem_radius = rng.uniform(cfg.stem_radius * 0.9, cfg.stem_radius * 1.1, n).astype(np.float32)
        self.height = rng.uniform(cfg.plant_height - cfg.plant_height_spread,
                                  cfg.plant_height + cfg.plant_height_spread, n).astype(np.float32)
        self.leaf_plant = np.repeat(np.arange(n), k)
        self.centers = np.empty((n * k, 3), dtype=np.float32)
        self.normals = np.empty((n * k, 3), dtype=np.float32)
        self.u = np.empty((n * k, 3), dtype=np.float32)
        self.v = np.empty((n * k, 3), dtype=np.float32)
        self.axes = np.empty((n * k, 2), dtype=np.float32)
        levels = np.linspace(0.14, 0.86, k)
        for plant in range(n):
            base_az = rng.uniform(-math.pi, math.pi)
            height = float(self.height[plant])
            for layer, level in enumerate(levels):
                index = plant * k + layer
                az = base_az + layer * math.pi + rng.normal(0.0, 0.08)
                elevation = -0.1 + 0.4 * float(level)
                half_length = height * (0.16 + 0.14 * math.sin(math.pi * level)) * rng.uniform(0.95, 1.05)
                half_width = height * rng.uniform(0.025, 0.04)
                node_z = height * float(level)
                if elevation > 0:
                    max_sin = min(1.0, max(0.0, (height - node_z) / (1.85 * half_length)))
                    elevation = min(elevation, math.asin(max_sin))
                u = np.array([math.cos(elevation) * math.cos(az),
                              math.cos(elevation) * math.sin(az), math.sin(elevation)], np.float32)
                v = np.array([-math.sin(az), math.cos(az), 0.0], np.float32)
                self.u[index], self.v[index] = u, v
                self.normals[index] = np.cross(u, v)
                node = np.array([*self.xy[plant], float(self.root_z[plant]) + node_z], np.float32)
                self.centers[index] = node + 0.85 * half_length * u
                self.axes[index] = [half_length, half_width]
        self.leaf_dry = np.zeros((n, k), bool)
        for i in np.flatnonzero(self.targets):
            self.leaf_dry[i, rng.choice(k, size=max(1, k // 3), replace=False)] = True
        self.leaf_dry = self.leaf_dry.reshape(-1)

    @classmethod
    def generate(cls, cfg: Config, rng: np.random.Generator, start: np.ndarray | None,
                 goal: np.ndarray, *, terrain: Terrain | None = None) -> "Scene":
        # A separate seed makes terrain reproducible without advancing the plant
        # geometry generator. The seed is derived before any plant draws.
        terrain_seed = cfg.terrain_seed if cfg.terrain_seed is not None else _terrain_seed_from_rng(rng)
        xy = generate_layout(cfg, rng, start=start, goal=goal, terrain=terrain)
        dry = rng.random(cfg.plants) < cfg.dry_fraction
        if cfg.plants and cfg.dry_fraction > 0:
            dry[rng.integers(cfg.plants)] = True
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if terrain is None and cfg.terrain_enabled:
            clear_disks = []
            if start is not None:
                clear_disks.append((np.asarray(start, dtype=np.float64), 1.0))
            clear_disks.append((np.asarray(goal, dtype=np.float64), 1.0))
            clear_disks.extend((p, 0.5) for p in xy)
            terrain = _make_terrain(cfg, terrain_seed, clear_disks)
        if terrain is not None and terrain.seed is not None:
            terrain_seed = int(terrain.seed)
        return cls(cfg, xy, dry, rng, terrain=terrain, terrain_seed=terrain_seed)

    def sample_start(self, rng: np.random.Generator, *, max_attempts: int = 2000) -> np.ndarray:
        """Sample an in-bounds yaw and reject its full rectangular footprint.

        Spawn checks include all rocks, including small traversable ones, so
        every reset begins without a plant or rock intersecting the chassis.
        """
        c = self.cfg
        margin = 0.025
        half_length, half_width = c.robot_length / 2, c.robot_width / 2
        if max_attempts <= 0 or min(c.size - c.robot_length - 2 * margin,
                                    c.size - c.robot_width - 2 * margin) <= 0:
            raise ValueError("no blank random start is available for this field and vehicle footprint")
        for _ in range(max_attempts):
            yaw = float(rng.uniform(-math.pi, math.pi))
            cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
            extent_x = cosine * half_length + sine * half_width
            extent_y = sine * half_length + cosine * half_width
            low = np.array([extent_x + margin, extent_y + margin])
            high = c.size - low
            if np.any(high <= low):
                continue
            xy = rng.uniform(low, high)
            if self.pose_clear(xy, yaw, include_small_rocks=True):
                return np.array([xy[0], xy[1], yaw], dtype=np.float64)
        raise ValueError(
            f"no blank random start found after {max_attempts} samples; "
            "the map may be too dense for the rectangular vehicle footprint"
        )

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> tuple[np.ndarray, np.ndarray]:
        """Offline geometric cast used only by conservative scene validation.

        Runtime sensor frames must come from the attached engine sensors.
        """
        o = np.asarray(origin, np.float32)
        d = np.asarray(directions, np.float32).reshape(-1, 3)
        if self.terrain is not None:
            out, terrain_kind = self.terrain.cast(o, d, max_range)
            out = np.asarray(out, dtype=np.float32).copy()
            terrain_kind = np.asarray(terrain_kind, dtype=np.uint8)
            hit = np.full(len(d), -1, dtype=np.int32)
            rock_rays = np.flatnonzero(terrain_kind == 3)
            if len(rock_rays):
                unit = d[rock_rays] / np.maximum(np.linalg.norm(d[rock_rays], axis=1), 1e-12)[:, None]
                points = o[None, :] + out[rock_rays, None] * unit
                delta = points[:, None, :2] - self.terrain.rock_xy[None, :, :]
                radial = np.sum(delta * delta, axis=2) / (self.terrain.rock_radius[None, :] ** 2)
                zfrac = ((points[:, None, 2] - self.terrain.rock_base_z[None, :])
                         / self.terrain.rock_height[None, :])
                on_rock = (radial <= 1.0 + 1e-3) & (zfrac >= -1e-3) & (zfrac <= 1.0 + 1e-3)
                score = np.where(on_rock, np.abs(radial + zfrac * zfrac - 1.0), np.inf)
                fallback = np.argmin(radial, axis=1)
                rock_ids = np.where(np.any(on_rock, axis=1), np.argmin(score, axis=1), fallback)
                hit[rock_rays] = self._ROCK_HIT - rock_ids.astype(np.int32)
        else:
            out = np.full(len(d), max_range, dtype=np.float32)
            hit = np.full(len(d), -1, dtype=np.int32)
            down = d[:, 2] < -1e-6
            ground = np.full(len(d), np.inf, np.float32)
            ground[down] = -o[2] / d[down, 2]
            take = (ground > 0) & (ground < out)
            out[take] = ground[take]
        leaf_padding = max(0.5, float(self.axes.max(initial=0.0)))
        near = np.sum((self.centers[:, :2] - o[:2]) ** 2, axis=1) < (max_range + leaf_padding) ** 2
        ids = np.flatnonzero(near)
        if len(ids):
            c, n, u, v, axes = self.centers[ids], self.normals[ids], self.u[ids], self.v[ids], self.axes[ids]
            den = d @ n.T
            num = np.sum((c - o) * n, axis=1)
            t = np.divide(num[None, :], den, out=np.full_like(den, np.inf), where=np.abs(den) > 1e-7)
            # No 3D hit-point tensor: two matrix products give patch coordinates.
            pu = np.sum((o - c) * u, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d @ u.T)
            pv = np.sum((o - c) * v, axis=1)[None, :] + np.where(np.isfinite(t), t, 0) * (d @ v.T)
            shape_x, shape_y = pu / axes[:, 0], pv / axes[:, 1]
            taper = (1.0 - shape_x ** 2) * (1.0 - np.abs(shape_x))
            inside = (np.abs(shape_x) <= 1.0) & (shape_y ** 2 <= taper)
            ts = np.where(inside & (t > 1e-4) & (t <= max_range), t, np.inf)
            j = np.argmin(ts, axis=1)
            best = ts[np.arange(len(d)), j]
            take = best < out
            out[take], hit[take] = best[take], ids[j[take]]
        stems = np.flatnonzero(np.sum((self.xy - o[:2]) ** 2, axis=1) < (max_range + 0.2) ** 2)
        if len(stems):
            q = o[:2] - self.xy[stems]
            a = np.sum(d[:, :2] ** 2, axis=1)[:, None]
            b = 2 * (d[:, :2] @ q.T)
            c = np.sum(q * q, axis=1)[None, :] - self.stem_radius[stems] ** 2
            disc = b * b - 4 * a * c
            t = np.divide(-b - np.sqrt(np.maximum(disc, 0)), 2 * a, out=np.full_like(b, np.inf), where=a > 1e-8)
            z = o[2] + np.where(np.isfinite(t), t, 0) * d[:, 2, None]
            root_z = self.root_z[stems][None, :]
            t = np.where((disc >= 0) & (t > 1e-4) & (z >= root_z)
                         & (z <= root_z + self.height[stems]), t, np.inf)
            j = np.argmin(t, axis=1)
            best = t[np.arange(len(d)), j]
            take = best < out
            out[take], hit[take] = best[take], -2 - stems[j[take]]
        return out, hit

    def pose_clear(self, xy: np.ndarray, yaw: float, include_small_rocks: bool = False) -> bool:
        """Test the oriented 0.8 m by 0.3 m footprint against root/rock disks."""
        point = np.asarray(xy, dtype=np.float64).reshape(-1)
        angle = float(yaw)
        if point.shape != (2,) or not np.all(np.isfinite(point)) or not math.isfinite(angle):
            return False

        cfg = self.cfg
        half_length, half_width = cfg.robot_length / 2, cfg.robot_width / 2
        cosine, sine = math.cos(angle), math.sin(angle)
        extent_x = abs(cosine) * half_length + abs(sine) * half_width
        extent_y = abs(sine) * half_length + abs(cosine) * half_width
        margin = 0.025
        if (point[0] < extent_x + margin or point[0] > cfg.size - extent_x - margin
                or point[1] < extent_y + margin or point[1] > cfg.size - extent_y - margin):
            return False

        def overlaps_disks(centers, radii):
            if not len(centers):
                return False
            delta = np.asarray(centers, dtype=np.float64) - point
            local_x = cosine * delta[:, 0] + sine * delta[:, 1]
            local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
            dx = np.maximum(np.abs(local_x) - half_length, 0.0)
            dy = np.maximum(np.abs(local_y) - half_width, 0.0)
            reach = np.asarray(radii, dtype=np.float64) + margin
            return bool(np.any(dx * dx + dy * dy <= reach * reach))

        if overlaps_disks(self.xy, self.radius):
            return False
        if self.terrain is not None and len(self.terrain.rock_xy):
            rock_mask = (np.ones(len(self.terrain.rock_xy), dtype=bool) if include_small_rocks
                         else np.asarray(self.terrain.rock_blocking, dtype=bool))
            if overlaps_disks(self.terrain.rock_xy[rock_mask], self.terrain.rock_radius[rock_mask]):
                return False
        return True

    def navigable(self, point: np.ndarray) -> bool:
        """Whether either allowed longitudinal heading clears the true footprint."""
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            return False
        return self.pose_clear(p, 0.0) or self.pose_clear(p, math.pi / 2)

    def _offline_navigation_labels(self, start: np.ndarray):
        """Legacy circle-clear grid used only by the full offline validator."""
        cfg = self.cfg
        start = np.asarray(start, dtype=np.float64).reshape(-1)
        if start.shape != (2,) or not np.all(np.isfinite(start)):
            raise ValueError("start must be a finite xy point")
        n = int(math.ceil(cfg.size / cfg.cell))
        axis = (np.arange(n) + 0.5) * cfg.cell
        r = math.hypot(cfg.robot_length, cfg.robot_width) / 2 + 0.05
        free = ((axis[:, None] >= r) & (axis[:, None] <= cfg.size - r)
                & (axis[None, :] >= r) & (axis[None, :] <= cfg.size - r))
        for p, pr in zip(self.xy, self.radius):
            reach = r + pr
            lo = np.maximum(0, np.floor((p - reach) / cfg.cell).astype(int))
            hi = np.minimum(n, np.ceil((p + reach) / cfg.cell).astype(int) + 1)
            xs, ys = axis[lo[0]:hi[0]], axis[lo[1]:hi[1]]
            free[lo[1]:hi[1], lo[0]:hi[0]] &= (xs[None, :] - p[0]) ** 2 + (ys[:, None] - p[1]) ** 2 > reach ** 2
        if self.terrain is not None:
            for p, pr in zip(self.terrain.rock_xy[self.terrain.rock_blocking],
                             self.terrain.rock_radius[self.terrain.rock_blocking]):
                reach = r + float(pr)
                lo = np.maximum(0, np.floor((p - reach) / cfg.cell).astype(int))
                hi = np.minimum(n, np.ceil((p + reach) / cfg.cell).astype(int) + 1)
                xs, ys = axis[lo[0]:hi[0]], axis[lo[1]:hi[1]]
                free[lo[1]:hi[1], lo[0]:hi[0]] &= (xs[None, :] - p[0]) ** 2 + (ys[:, None] - p[1]) ** 2 > reach ** 2
        labels, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(2, 1))
        ij = np.clip((start / cfg.cell).astype(int), 0, n - 1)
        start_label = int(labels[ij[1], ij[0]])
        reached = labels == start_label if start_label else np.zeros_like(free)
        return free, labels, reached

    def validate_connectivity(self, start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
        """Fast conservative two-heading connectivity check for task resets."""
        return validate_navigation_scene(self, start, goal)

    def validate(self, start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
        """Full offline validation, including sampled dry-leaf visibility.

        The vehicle grid here intentionally retains the older circumscribed
        disk approximation; reset-time reachability uses navigation.py.
        """
        cfg = self.cfg
        free, _, reached = self._offline_navigation_labels(start)
        goal_xy = np.asarray(goal, dtype=np.float64).reshape(-1)
        radius = math.hypot(cfg.robot_length, cfg.robot_width) / 2 + 0.05
        goal_circle_clear = (goal_xy.shape == (2,) and np.all(np.isfinite(goal_xy))
                             and np.all(goal_xy >= radius)
                             and np.all(goal_xy <= cfg.size - radius)
                             and (not len(self.xy) or np.all(
                                 np.linalg.norm(self.xy - goal_xy, axis=1) > self.radius + radius
                             )))
        if goal_circle_clear and self.terrain is not None:
            blocking = np.asarray(self.terrain.rock_blocking, dtype=bool)
            if np.any(blocking):
                goal_circle_clear = bool(np.all(
                    np.linalg.norm(self.terrain.rock_xy[blocking] - goal_xy, axis=1)
                    > self.terrain.rock_radius[blocking] + radius
                ))
        all_connected = int(reached.sum()) == int(free.sum()) and bool(goal_circle_clear)
        invisible = []
        for plant in np.flatnonzero(self.targets):
            visible = False
            for leaf in np.flatnonzero((self.leaf_plant == plant) & self.leaf_dry):
                c = self.centers[leaf]
                for az in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                    xy = self.xy[plant] + 1.3 * np.array([math.cos(az), math.sin(az)])
                    if not self.navigable(xy):
                        continue
                    ground_z = float(self.terrain.surface_height(xy)) if self.terrain is not None else 0.0
                    o = np.array([*xy, ground_z + cfg.camera_height])
                    delta = c - o
                    dist = np.linalg.norm(delta)
                    pitch = math.atan2(delta[2], np.linalg.norm(delta[:2]))
                    if not cfg.pitch_min <= pitch <= cfg.pitch_max or dist > cfg.depth_range:
                        continue
                    _, h = self.cast(o, (delta / dist)[None, :], cfg.camera_range)
                    if h[0] == leaf:
                        visible = True
                        break
                if visible:
                    break
            if not visible:
                invisible.append(int(plant))
        return {"connected": bool(all_connected), "free_cells": int(free.sum()),
                "reachable_cells": int(reached.sum()), "unobservable_targets": invisible,
                "scope": "conservative circumscribed vehicle disk; dry-leaf center samples, not exhaustive visibility proof"}



class FieldTask:
    """Simulator-independent task state, scoring, and observation assembly."""

    _SENSOR_KINDS = ("lidar", "depth", "camera")

    def __init__(self, cfg: Config | None = None, mode: str = "search"):
        self.cfg = cfg or Config()
        if mode not in ("search", "goto"):
            raise ValueError("mode must be search or goto")
        self.mode = mode
        c = self.cfg
        az = np.linspace(-math.pi, math.pi, c.lidar_rays, endpoint=False)
        self.laser_local = np.stack((np.cos(az), np.sin(az), np.zeros_like(az)), -1).astype(np.float32)
        sx, sy = c.depth_width / DEPTH_NATIVE_WIDTH, c.depth_height / DEPTH_NATIVE_HEIGHT
        depth_k = DEPTH_NATIVE_INTRINSICS.copy()
        depth_k[0, 0] *= sx
        depth_k[0, 2] = (depth_k[0, 2] + 0.5) * sx - 0.5
        depth_k[1, 1] *= sy
        depth_k[1, 2] = (depth_k[1, 2] + 0.5) * sy - 0.5
        self.depth_intrinsics = depth_k.astype(np.float32)
        pixel_u, pixel_v = np.meshgrid(np.arange(c.depth_width, dtype=np.float32),
                                       np.arange(c.depth_height, dtype=np.float32), indexing="xy")
        x = (pixel_u - self.depth_intrinsics[0, 2]) / self.depth_intrinsics[0, 0]
        y = (pixel_v - self.depth_intrinsics[1, 2]) / self.depth_intrinsics[1, 1]
        self.depth_camera_xy = np.stack((x.ravel(), y.ravel()), axis=-1).astype(np.float32)
        self.scene_id = 0
        self._capture_callback = None
        self.np_random = np.random.default_rng()
        self.robot = np.array([2.0, 2.0, math.pi / 4, 0.0, 0.0], np.float64)
        self.gimbal = np.zeros(2, dtype=np.float64)
        self.body_pose = (0.0, 0.0, 0.0)
        self.view_origin = np.array([2.0, 2.0, c.camera_height], dtype=np.float32)
        self.view_basis = (np.array([1.0, 0.0, 0.0], np.float32),
                           np.array([0.0, -1.0, 0.0], np.float32),
                           np.array([0.0, 0.0, 1.0], np.float32))
        self._progress_reference: float | None = None

    def attach_sensors(self, capture):
        """Attach a callback returning a real engine frame for each sensor kind."""
        if not callable(capture):
            raise TypeError("capture must be callable")
        self._capture_callback = capture

    def prepare(self, seed=None, options=None):
        """Create/reset task geometry and bookkeeping without sampling sensors."""
        opts = dict(options or {})
        random_start = opts.get("random_start", "start" not in opts)
        if not isinstance(random_start, (bool, np.bool_)):
            raise ValueError("random_start must be a boolean")
        if random_start and "start" in opts:
            raise ValueError("start and random_start=True cannot be supplied together")
        if "mode" in opts:
            self.mode = opts["mode"]
        if self.mode not in ("search", "goto"):
            raise ValueError("unknown task")
        self.np_random = np.random.default_rng(seed)
        c = self.cfg
        self.time = 0.0
        # IsaacFieldEnv may replace the spawn and sampled goto target after
        # prepare(); lock the distance only on the first subsequent finish().
        self._progress_reference = None
        start = np.asarray(opts.get("start", [2.0, 2.0, math.pi / 4]), dtype=np.float64)
        radius = math.hypot(c.robot_length, c.robot_width) / 2
        if (start.shape != (3,) or not np.all(np.isfinite(start))
                or np.any(start[:2] < radius) or np.any(start[:2] > c.size - radius)):
            raise ValueError("start must contain finite x, y, yaw with the vehicle inside the field")
        self.robot = np.array([start[0], start[1], wrap(float(start[2])), 0.0, 0.0], np.float64)
        self.gimbal = np.zeros(2, dtype=np.float64)
        self.body_pose = (0.0, 0.0, 0.0)
        self.goal = np.asarray(opts.get("goal", [c.size - 2, c.size - 2]), dtype=np.float64)
        if self.goal.shape != (2,) or not np.all(np.isfinite(self.goal)):
            raise ValueError("goal must contain two finite coordinates")
        self.reported, self.seen, self.damaged, self.false_reports = set(), set(), set(), set()
        self.trail = deque(maxlen=600)
        self.trail.append(self.robot[:2].tolist())
        self.last_action = np.zeros(4, np.float32)
        self.last_gimbal_rates = np.zeros(2, dtype=np.float64)
        self.done, self.reason, self.reward = False, "", 0.0
        self.total_reward = 0.0
        self._hold = 0.0
        self.collisions = 0
        self._serial = {k: 0 for k in self._SENSOR_KINDS}
        self._dropped = {k: 0 for k in self._SENSOR_KINDS}
        self._next = {k: 0.0 for k in self._SENSOR_KINDS}
        self._pending = {k: deque() for k in self._SENSOR_KINDS}
        self.frames = {}
        self._new = {k: False for k in self._SENSOR_KINDS}
        self.validation = None
        self._pose_noise = np.zeros(2, dtype=np.float64)
        self._next_uwb = 0.0
        supplied_scene = opts.get("scene")
        supplied_terrain = opts.get("terrain")
        if supplied_scene is not None and supplied_terrain is not None:
            if supplied_scene.terrain is not supplied_terrain:
                raise ValueError("scene and terrain options must refer to the same terrain object")
        spawn_error = None
        for attempt in range(12):
            self.scene = supplied_scene if supplied_scene is not None else Scene.generate(
                c, self.np_random, None if random_start else self.robot[:2], self.goal,
                terrain=supplied_terrain)
            if random_start:
                try:
                    self.robot[:3] = self.scene.sample_start(self.np_random)
                    spawn_error = None
                except ValueError as exc:
                    spawn_error = exc
                    if supplied_scene is not None:
                        raise
                    if attempt == 11:
                        break
                    continue
                self.trail.clear()
                self.trail.append(self.robot[:2].tolist())
            if not c.validate_scene:
                break
            self.validation = self.scene.validate_connectivity(self.robot[:2], self.goal)
            if self.validation["connected"]:
                break
            if supplied_scene is not None:
                raise ValueError(f"invalid supplied scene: {self.validation}")
        else:
            raise ValueError(f"unable to generate validated scene: {self.validation}")
        if random_start and spawn_error is not None:
            raise ValueError(f"unable to generate a scene with a blank random start: {spawn_error}")
        self.scene_id += 1
        z, pitch, roll = self._body_pose()
        self.view_origin = np.asarray([self.robot[0], self.robot[1], z + c.camera_height], dtype=np.float32)
        self.view_basis = self._flat_basis()
        self._counts = (0, 0, 0)
        return self.scene

    def initialize_sensors(self):
        """Capture initial engine frames and return the first policy observation."""
        if self._capture_callback is None:
            raise RuntimeError("attach_sensors(capture) must be called before initialize_sensors()")
        self._sense(initial=True)
        return self._observation(), self._info()

    def _body_pose(self):
        z, pitch, roll = self.body_pose
        return float(z), float(pitch), float(roll)

    def _origin(self):
        return np.asarray(self.view_origin, dtype=np.float32).reshape(3)

    def _basis(self):
        if len(self.view_basis) != 3:
            raise ValueError("view_basis must be (forward, right, up)")
        return tuple(np.asarray(axis, dtype=np.float32).reshape(3) for axis in self.view_basis)

    def _flat_basis(self):
        yaw = float(self.robot[2])
        gy, gp = (float(value) for value in self.gimbal)
        f_local = np.array([math.cos(gp) * math.cos(gy), math.cos(gp) * math.sin(gy), math.sin(gp)], np.float32)
        right_local = np.array([math.sin(gy), -math.cos(gy), 0.0], np.float32)
        up_local = np.cross(right_local, f_local)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotate = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], np.float32)
        return tuple(rotate @ axis for axis in (f_local, right_local, up_local))

    def _capture(self, kind):
        if kind not in self._SENSOR_KINDS:
            raise ValueError(f"unknown sensor kind: {kind}")
        if self._capture_callback is None:
            raise RuntimeError("sensor capture callback is not attached")
        result = self._capture_callback(kind)
        if not isinstance(result, dict):
            raise TypeError(f"capture({kind!r}) must return a frame dictionary")
        frame = dict(result)
        self._serial[kind] += 1
        z, pitch, roll = self._body_pose()
        frame.setdefault("seq", self._serial[kind])
        frame.setdefault("capture", self.time)
        frame.setdefault("pose", np.concatenate((self.robot[:3], self.gimbal)).copy())
        frame.setdefault("origin", self._origin().copy())
        frame.setdefault("body_attitude", np.asarray([roll, pitch], dtype=np.float32))
        frame["seq"] = int(frame["seq"])
        frame["capture"] = float(frame["capture"])
        frame["pose"] = np.asarray(frame["pose"], dtype=np.float32).reshape(5)
        frame["origin"] = np.asarray(frame["origin"], dtype=np.float32).reshape(3)
        frame["body_attitude"] = np.asarray(frame["body_attitude"], dtype=np.float32).reshape(2)
        if kind == "lidar":
            if "ranges" not in frame:
                raise ValueError("lidar engine frame requires ranges")
            ranges = np.asarray(frame["ranges"], dtype=np.float32).reshape(-1)
            if ranges.shape != (self.cfg.lidar_rays,):
                raise ValueError(f"lidar ranges must have shape ({self.cfg.lidar_rays},)")
            finite = np.isfinite(ranges)
            ranges = np.nan_to_num(ranges, nan=self.cfg.lidar_range, posinf=self.cfg.lidar_range,
                                   neginf=self.cfg.lidar_min_range)
            ranges = np.clip(ranges, self.cfg.lidar_min_range, self.cfg.lidar_range)
            valid = np.asarray(frame.get("valid", finite & (ranges < self.cfg.lidar_range)), dtype=bool).reshape(-1)
            if valid.shape != ranges.shape:
                raise ValueError("lidar valid mask must match ranges")
            frame["ranges"] = ranges
            frame["valid"] = valid & finite
            # Semantic hit labels are optional; zero means unknown, as in the UI schema.
            frame["hit_kind"] = np.asarray(frame.get("hit_kind", np.zeros(len(ranges), np.uint8)), dtype=np.uint8).reshape(-1)
            if frame["hit_kind"].shape != ranges.shape:
                raise ValueError("lidar hit_kind must match ranges")
            forward, right, _ = self._basis()
            frame.setdefault("directions", self.laser_local[:, :1] * forward + self.laser_local[:, 1:2] * (-right))
            frame["directions"] = np.asarray(frame["directions"], dtype=np.float32).reshape(self.cfg.lidar_rays, 3)
            frame.setdefault("scan_pitch", float(math.asin(float(np.clip(forward[2], -1.0, 1.0)))) )
            frame["scan_pitch"] = float(frame["scan_pitch"])
        elif kind == "depth":
            if "values" not in frame:
                raise ValueError("depth engine frame requires camera_z values in metres")
            values = np.asarray(frame["values"], dtype=np.float32).reshape(self.cfg.depth_height, self.cfg.depth_width)
            finite = np.isfinite(values)
            values = np.nan_to_num(values, nan=self.cfg.depth_range, posinf=self.cfg.depth_range,
                                   neginf=self.cfg.depth_range)
            values = np.clip(values, self.cfg.depth_min_range, self.cfg.depth_range)
            valid = np.asarray(frame.get("valid", finite & (values < self.cfg.depth_range)), dtype=bool)
            valid = valid.reshape(self.cfg.depth_height, self.cfg.depth_width) & finite
            frame.update(values=values, valid=valid)
            frame.setdefault("unit", "m")
            frame.setdefault("convention", "camera_z")
            frame.setdefault("projection", "pinhole")
            frame.setdefault("resolution", [self.cfg.depth_width, self.cfg.depth_height])
            frame.setdefault("shape", [self.cfg.depth_height, self.cfg.depth_width])
            frame.setdefault("native_resolution", [DEPTH_NATIVE_WIDTH, DEPTH_NATIVE_HEIGHT])
            frame.setdefault("intrinsics", self.depth_intrinsics.copy())
            frame.setdefault("calibration_id", DEPTH_GEOMETRY_ID)
            frame.setdefault("calibration_source", DEPTH_GEOMETRY_SOURCE)
            frame.setdefault("min_range", self.cfg.depth_min_range)
            frame.setdefault("max_range", self.cfg.depth_range)
            frame["intrinsics"] = np.asarray(frame["intrinsics"], dtype=np.float32).reshape(3, 3)
        else:
            if "detections" not in frame:
                raise ValueError("camera engine frame requires detections from its perception pipeline")
            frame["detections"] = list(frame["detections"])
        return frame

    def finish(self, action, before, old_distance, rock_collision=False, boundary=False):
        """Finish a physics step after the engine has updated the actual pose."""
        if self.done:
            raise RuntimeError("episode ended; prepare() a new episode")
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError("action must contain four finite values")
        a = np.clip(a, -1.0, 1.0)
        before = np.asarray(before, dtype=np.float64)
        if before.shape != (5,) or not np.all(np.isfinite(before)):
            raise ValueError("before must contain finite x, y, yaw, v, w")
        if not math.isfinite(float(old_distance)) or float(old_distance) < 0:
            raise ValueError("old_distance must be finite and non-negative")
        if rock_collision:
            self.collisions += 1
        return self._finish_step(a, before, float(old_distance), bool(rock_collision), bool(boundary))



    def set_goal(self, goal):
        p = np.asarray(goal, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)) or not self.scene.navigable(p):
            raise ValueError("goal must be inside the field and clear for the vehicle")
        self.goal = p
        self.mode = "goto"
        self._hold = 0.0
        # A changed viewer goal starts a fresh normalized progress period.
        self._progress_reference = None
        self.done, self.reason = False, ""

    def _sense(self, initial=False):
        c = self.cfg
        for kind, period, latency in (("lidar", c.lidar_period, c.lidar_latency), ("depth", c.depth_period, c.depth_latency), ("camera", c.camera_period, c.camera_latency)):
            self._new[kind] = False
            due = initial or self.time + 1e-9 >= self._next[kind]
            if due:
                frame = self._capture(kind)
                if initial:
                    self._next[kind] = self.time + period
                else:
                    # Keep the absolute sensor clock. At most one frame is
                    # sampled at this physics instant; overdue whole periods
                    # are counted as dropped instead of replayed at one pose.
                    deadline = self._next[kind]
                    due_periods = max(1, math.floor((self.time - deadline + 1e-9) / period) + 1)
                    self._dropped[kind] += due_periods - 1
                    self._next[kind] = deadline + due_periods * period
                self._pending[kind].append((self.time if initial else self.time + latency, frame))
            while self._pending[kind] and self._pending[kind][0][0] <= self.time + 1e-9:
                _, frame = self._pending[kind].popleft()
                self.frames[kind] = frame
                self._new[kind] = True
                if kind == "camera":
                    for d in frame["detections"]:
                        plant = d.get("plant")
                        if plant is None or not 0 <= int(plant) < len(self.scene.targets):
                            continue
                        plant = int(plant)
                        self.seen.add(plant)
                        # MVP: automatic detector report, NOT a learned report/finish head.
                        if d["class_id"] == 1 and d["confidence"] >= .68 and d["range"] is not None:
                            (self.reported if self.scene.targets[plant] else self.false_reports).add(plant)
        if initial or self.time + 1e-9 >= self._next_uwb:
            self._pose_noise = self.np_random.normal(0, c.position_noise, 2)
            self._next_uwb = self.time + .1

    def _damage_sweep(self, before, after):
        c = self.cfg
        displacement = np.linalg.norm(after[:2] - before[:2])
        dyaw = wrap(after[2] - before[2])
        subdivisions = max(1, math.ceil(displacement / min(c.robot_width / 3, .1)), math.ceil(abs(dyaw) / .08))
        for t in np.linspace(0, 1, subdivisions + 1):
            p = before[:2] + t * (after[:2] - before[:2])
            a = before[2] + t * dyaw
            delta = self.scene.xy - p
            x = delta[:, 0] * math.cos(a) + delta[:, 1] * math.sin(a)
            y = -delta[:, 0] * math.sin(a) + delta[:, 1] * math.cos(a)
            dx = np.maximum(np.abs(x) - c.robot_length / 2, 0)
            dy = np.maximum(np.abs(y) - c.robot_width / 2, 0)
            self.damaged.update(np.flatnonzero(dx * dx + dy * dy <= self.scene.radius ** 2).tolist())

    def _finish_step(self, a, before, old_distance, rock_collision, boundary):
        """Apply shared damage, sensing, reward, and episode bookkeeping."""
        c = self.cfg
        self._damage_sweep(before, self.robot)
        self.time += c.dt
        self.trail.append(self.robot[:2].tolist())
        self._sense()
        count = (len(self.reported), len(self.false_reports), len(self.damaged))
        delta = np.array(count) - self._counts
        self._counts = count
        reward = (-c.time_penalty * c.dt - c.damage_penalty * delta[2]
                  - c.collision_penalty * int(rock_collision) - .1 * boundary * c.dt)
        distance = np.linalg.norm(self.goal - self.robot[:2])
        if self.mode == "search":
            reward += float(delta[0]) - .25 * delta[1]
            if self.scene.targets.sum() > 0 and len(self.reported) == int(self.scene.targets.sum()):
                self.done, self.reason = True, "all_targets_found_by_evaluator"
                reward += 5
        else:
            if c.goal_progress_budget is None:
                reward += c.progress_weight * (old_distance - distance)
            else:
                if self._progress_reference is None:
                    self._progress_reference = float(old_distance)
                progress_scale = c.goal_progress_budget / max(
                    self._progress_reference, c.goal_tolerance
                )
                reward += progress_scale * (old_distance - distance)
            self._hold = self._hold + c.dt if distance <= c.goal_tolerance and abs(self.robot[3]) < .2 and abs(self.robot[4]) < .3 else 0
            if self._hold + 1e-9 >= c.goal_hold:
                self.done, self.reason = True, "goal_reached"
                reward += 5
        if c.stop_on_damage and delta[2] > 0:
            self.done, self.reason = True, "plant_damage"
        truncated = self.time + 1e-9 >= c.max_time and not self.done
        if truncated:
            self.done, self.reason = True, "time_limit"
            if self.mode == "search":
                reward -= 2 * (1 - len(self.reported) / max(1, int(self.scene.targets.sum())))
        # Measured motion is already present at state[8:10]. Keep the prior
        # policy command separately so the recurrent policy can observe its
        # own command after actuator lag and contact dynamics.
        self.last_action = np.array(a, dtype=np.float32, copy=True)
        self.reward = float(reward)
        self.total_reward += self.reward
        return self._observation(), self.reward, self.done and not truncated, bool(truncated), self._info()

    def _observation(self):
        c = self.cfg
        lf, df, cf = (self.frames[k] for k in ("lidar", "depth", "camera"))
        det = np.zeros((c.max_detections, 10), np.float32)
        for i, d in enumerate(cf["detections"]):
            det[i] = [*d["box"], d["class_id"], d["confidence"], min(1, (d["range"] or 0) / c.depth_range), float(d["range"] is not None), 1, 0]
        estimated_xy = self.robot[:2] + self._pose_noise
        delta = self.goal - estimated_xy
        theta = self.robot[2]
        relative = np.array([math.cos(theta) * delta[0] + math.sin(theta) * delta[1], -math.sin(theta) * delta[0] + math.cos(theta) * delta[1]]) / c.size
        state = [float(self.mode == "search"), float(self.mode == "goto"), *(relative if self.mode == "goto" else [0, 0]),
                 *(estimated_xy / c.size * 2 - 1), math.sin(theta), math.cos(theta), self.robot[3] / c.max_speed,
                 self.robot[4] / c.max_turn_rate, math.sin(self.gimbal[0]), math.cos(self.gimbal[0]), self.gimbal[1] / math.pi,
                 *self.last_action, min(1, self.time / c.max_time), c.dt, c.goal_tolerance / c.size]
        for k in ("lidar", "depth", "camera"):
            f = self.frames[k]
            age = self.time - f["capture"]
            motion = self.robot[:2] - f["pose"][:2]
            state.extend([min(age / 2, 1), float(self._new[k]), *(motion / c.lidar_range),
                          wrap(theta - f["pose"][2]) / math.pi, wrap(self.gimbal[0] - f["pose"][3]) / math.pi,
                          (self.gimbal[1] - f["pose"][4]) / math.pi])
        state.append(c.position_noise / c.size)
        _, body_pitch, body_roll = self._body_pose()
        state.extend([body_roll / math.pi, body_pitch / math.pi])
        return {"lidar": np.stack((lf["ranges"] / c.lidar_range, lf["valid"])).astype(np.float32),
                "depth": np.stack((df["values"] / c.depth_range, df["valid"])).astype(np.float32),
                "detections": det, "state": np.clip(np.asarray(state, np.float32), -1, 1)}

    def _info(self):
        return {"time": self.time, "found": len(self.reported), "targets": int(self.scene.targets.sum()),
                "false_reports": len(self.false_reports), "damage": len(self.damaged),
                "collisions": self.collisions,
                "distance": float(np.linalg.norm(self.goal - self.robot[:2])), "reason": self.reason,
                "is_success": self.reason in ("goal_reached", "all_targets_found_by_evaluator"),
                "sensor_sequences": {k: self.frames[k]["seq"] for k in self.frames},
                "sensor_dropped": self._dropped.copy()}

    def scene_payload(self):
        s = self.scene
        if s.terrain is None:
            terrain_payload = {"enabled": False, "seed": None, "base_height": 0.0,
                               "wave_vectors4x2": [], "wave_amplitudes4": [], "wave_phases4": [],
                               "rocks": []}
        else:
            t = s.terrain
            terrain_payload = {
                "enabled": True, "seed": s.terrain_seed, "base_height": float(t.base_height),
                "wave_vectors4x2": np.asarray(t.wave_vectors4x2).tolist(),
                "wave_amplitudes4": np.asarray(t.wave_amplitudes4).tolist(),
                "wave_phases4": np.asarray(t.wave_phases4).tolist(),
                "rocks": [{"x": float(xy[0]), "y": float(xy[1]),
                           "z": float(t.rock_base_z[i]), "base_z": float(t.rock_base_z[i]),
                           "radius": float(t.rock_radius[i]), "height": float(t.rock_height[i]),
                           "blocking": bool(t.rock_blocking[i])}
                          for i, xy in enumerate(t.rock_xy)],
            }
        return {"scene_id": self.scene_id, "size": self.cfg.size, "config": asdict(self.cfg), "validation": self.validation,
                "truth_debug_only": True,
                "schema": "field-geometry-debug-v1", "coordinate_frame": "field_xyz_z_up_m",
                "plant_model": "maize-proxy-v1",
                "terrain": terrain_payload,
                "leaf_geometry": {"centers": s.centers.tolist(), "normals": s.normals.tolist(),
                                  "u": s.u.tolist(), "v": s.v.tolist(), "axes": s.axes.tolist(),
                                  "dry": s.leaf_dry.tolist(), "plant": s.leaf_plant.tolist(),
                                  "shape": "tapered_blade_v1"},
                "plants": [{"x": float(x), "y": float(y), "z": float(s.root_z[i]), "base_z": float(s.root_z[i]),
                            "radius": float(s.radius[i]), "height": float(s.height[i]),
                            "stem_radius": float(s.stem_radius[i]), "root_damage_radius": float(s.radius[i]),
                            "dry": bool(s.targets[i]), "leaves": [{"center": c.tolist(), "dry": bool(d)} for c, d in zip(s.centers[s.leaf_plant == i], s.leaf_dry[s.leaf_plant == i])]}
                           for i, (x, y) in enumerate(s.xy)]}

    def state_payload(self):
        lf, df, cf = (self.frames[k] for k in ("lidar", "depth", "camera"))
        forward, _, up = self._basis()
        body_z, body_pitch, body_roll = self._body_pose()
        return {"scene_id": self.scene_id, "mode": self.mode, "time": self.time, "dt": self.cfg.dt,
                "view_pose": {"origin": self._origin().tolist(), "forward": forward.tolist(),
                              "up": up.tolist(), "half_angle": math.pi / 6},
                "robot": {"x": float(self.robot[0]), "y": float(self.robot[1]), "yaw": float(self.robot[2]),
                          "z": body_z, "roll": body_roll, "pitch": body_pitch,
                          "v": float(self.robot[3]), "w": float(self.robot[4])},
                "gimbal": dict(zip(("yaw", "pitch"), self.gimbal.tolist())), "goal": self.goal.tolist(),
                "reported": sorted(self.reported), "seen": sorted(self.seen), "damaged": sorted(self.damaged), "trail": list(self.trail),
                "lidar": {"model": self.cfg.lidar_model, "rays": self.cfg.lidar_rays,
                          "period": self.cfg.lidar_period, "min_range": self.cfg.lidar_min_range,
                          "relative_error": self.cfg.lidar_relative_error,
                          "ranges": lf["ranges"].tolist(), "directions": lf["directions"].tolist(),
                          "valid": lf["valid"].tolist(), "hit_kind": lf["hit_kind"].tolist(),
                          "scan_pitch": float(lf["scan_pitch"]), "gimbal_pitch": float(lf["pose"][4]),
                          "max_range": self.cfg.lidar_range,
                          "age": self.time - lf["capture"], "seq": lf["seq"], "origin": lf["origin"].tolist(),
                          "body_attitude": lf["body_attitude"].tolist()},
                "depth": {"values": df["values"].tolist(), "valid": df["valid"].tolist(),
                          "unit": df["unit"], "convention": df["convention"],
                          "projection": df["projection"], "resolution": df["resolution"],
                          "shape": df["shape"],
                          "native_resolution": df["native_resolution"],
                          "intrinsics": df["intrinsics"].tolist(),
                          "calibration_id": df["calibration_id"],
                          "calibration_source": df["calibration_source"],
                          "min_range": df["min_range"], "max_range": df["max_range"],
                          "capture_time": float(df["capture"]), "pose": df["pose"].tolist(),
                          "origin": df["origin"].tolist(), "body_attitude": df["body_attitude"].tolist(),
                          "age": self.time - df["capture"], "seq": df["seq"]},
                "detections": [{k: d[k] for k in ("box", "class_id", "confidence", "range")} for d in cf["detections"]],
                "metrics": self._info(), "reward": self.reward, "total_reward": self.total_reward,
                "done": self.done, "reason": self.reason}
