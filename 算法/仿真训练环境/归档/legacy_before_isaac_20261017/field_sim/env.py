"""Array-based field environment. No renderer, SLAM, or image detector in step()."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from field_sim.terrain import Terrain


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

    def __post_init__(self):
        if self.size < 6 or self.cell <= 0 or self.dt <= 0 or self.max_time <= 0:
            raise ValueError("size >= 6, cell/dt/max_time > 0 required")
        if self.plants < 0 or not 0 <= self.dry_fraction <= 1:
            raise ValueError("invalid plant count or dry fraction")
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
        if cfg.terrain_enabled:
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
                 goal: np.ndarray) -> "Scene":
        # A separate seed makes terrain reproducible without advancing the plant
        # geometry generator. The seed is derived before any plant draws.
        terrain_seed = cfg.terrain_seed if cfg.terrain_seed is not None else _terrain_seed_from_rng(rng)
        positions: list[np.ndarray] = []
        spacing = max(1.4, math.hypot(cfg.robot_length, cfg.robot_width) + 0.85)
        margin = max(1.2, spacing * 0.7)
        if margin * 2 >= cfg.size:
            raise ValueError("vehicle too large for field")
        for _ in range(max(1000, cfg.plants * 150)):
            if len(positions) >= cfg.plants:
                break
            p = rng.uniform(margin, cfg.size - margin, 2)
            p = np.floor(p / cfg.cell) * cfg.cell + cfg.cell / 2
            if ((start is not None and np.linalg.norm(p - start) < 1.5)
                    or np.linalg.norm(p - goal) < 1.5):
                continue
            if positions and np.min(np.linalg.norm(np.asarray(positions) - p, axis=1)) < spacing:
                continue
            positions.append(p)
        if len(positions) != cfg.plants:
            raise ValueError(f"Cannot place {cfg.plants} plants at this size/vehicle clearance; placed {len(positions)}. Reduce density.")
        dry = rng.random(cfg.plants) < cfg.dry_fraction
        if cfg.plants and cfg.dry_fraction > 0:
            dry[rng.integers(cfg.plants)] = True
        xy = np.asarray(positions).reshape(-1, 2)
        terrain = None
        if cfg.terrain_enabled:
            clear_disks = []
            if start is not None:
                clear_disks.append((np.asarray(start, dtype=np.float64), 1.0))
            clear_disks.append((np.asarray(goal, dtype=np.float64), 1.0))
            clear_disks.extend((p, 0.5) for p in xy)
            terrain = _make_terrain(cfg, terrain_seed, clear_disks)
        return cls(cfg, xy, dry, rng, terrain=terrain, terrain_seed=terrain_seed)

    def sample_start(self, rng: np.random.Generator, *, max_attempts: int = 2000) -> np.ndarray:
        """Uniformly sample a blank vehicle pose inside this fixed scene.

        The candidate is rejected against the same circumscribed vehicle
        clearance used by ``navigable``. Small, traversable rocks are also
        excluded so every random spawn begins on an empty patch of ground.
        The returned float64 array is an independent ``[x, y, yaw]`` pose.
        """
        clearance = math.hypot(self.cfg.robot_length, self.cfg.robot_width) / 2 + 0.05
        if max_attempts <= 0 or 2 * clearance >= self.cfg.size:
            raise ValueError("no blank random start is available for this field and vehicle footprint")
        for _ in range(max_attempts):
            xy = rng.uniform(clearance, self.cfg.size - clearance, size=2)
            if not self.navigable(xy):
                continue
            if (self.terrain is not None and len(self.terrain.rock_xy)
                    and np.any(np.linalg.norm(self.terrain.rock_xy - xy, axis=1)
                               <= self.terrain.rock_radius + clearance)):
                continue
            return np.array([xy[0], xy[1], rng.uniform(-math.pi, math.pi)], dtype=np.float64)
        raise ValueError(
            f"no blank random start found after {max_attempts} samples; "
            "the map may be too dense for the vehicle clearance"
        )

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> tuple[np.ndarray, np.ndarray]:
        """First hit on finite tapered leaves, stems, terrain, or rocks."""
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

    def navigable(self, point: np.ndarray) -> bool:
        r = math.hypot(self.cfg.robot_length, self.cfg.robot_width) / 2 + 0.05
        clear = (np.all(point >= r) and np.all(point <= self.cfg.size - r)
                 and (not len(self.xy) or np.all(np.linalg.norm(self.xy - point, axis=1) > self.radius + r)))
        if not clear or self.terrain is None:
            return bool(clear)
        blocking = self.terrain.rock_blocking
        if np.any(blocking):
            rock_distance = np.linalg.norm(self.terrain.rock_xy[blocking] - point, axis=1)
            clear = clear and bool(np.all(rock_distance > self.terrain.rock_radius[blocking] + r))
        return bool(clear)

    def validate(self, start: np.ndarray, goal: np.ndarray) -> dict[str, Any]:
        """Conservative disk-clearance connectivity; never passed to policy."""
        cfg = self.cfg
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
        ij = np.clip((start / cfg.cell).astype(int), 0, n - 1)
        reached = np.zeros_like(free)
        todo = deque([(int(ij[1]), int(ij[0]))]) if free[ij[1], ij[0]] else deque()
        if todo:
            reached[todo[0]] = True
        while todo:
            y, x = todo.popleft()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < n and 0 <= xx < n and free[yy, xx] and not reached[yy, xx]:
                    reached[yy, xx] = True
                    todo.append((yy, xx))
        all_connected = int(reached.sum()) == int(free.sum()) and self.navigable(goal)
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


class FieldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None, mode: str = "search"):
        super().__init__()
        self.cfg = cfg or Config()
        if mode not in ("search", "goto"):
            raise ValueError("mode must be search or goto")
        self.mode = mode
        c = self.cfg
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = spaces.Dict({
            "lidar": spaces.Box(0, 1, (2, c.lidar_rays), np.float32),
            "depth": spaces.Box(0, 1, (2, c.depth_height, c.depth_width), np.float32),
            "detections": spaces.Box(-1, 1, (c.max_detections, 10), np.float32),
            "state": spaces.Box(-1, 1, (44,), np.float32),
        })
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
        self._body_pose_cache_scene = None
        self._body_pose_cache_terrain = None
        self._body_pose_cache_key = None
        self._body_pose_cache_value = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        random_start = opts.get("random_start", False)
        if not isinstance(random_start, (bool, np.bool_)):
            raise ValueError("random_start must be a boolean")
        if random_start and "start" in opts:
            raise ValueError("start and random_start=True cannot be supplied together")
        if "mode" in opts:
            self.mode = opts["mode"]
        if self.mode not in ("search", "goto"):
            raise ValueError("unknown task")
        c = self.cfg
        self.time = 0.0
        start = np.asarray(opts.get("start", [2.0, 2.0, math.pi / 4]), dtype=np.float64)
        radius = math.hypot(c.robot_length, c.robot_width) / 2
        if (start.shape != (3,) or not np.all(np.isfinite(start)) or
                np.any(start[:2] < radius) or np.any(start[:2] > c.size - radius)):
            raise ValueError("start must contain finite x, y, yaw with the vehicle inside the field")
        self.robot = np.array([start[0], start[1], wrap(float(start[2])), 0.0, 0.0], np.float64)
        self.gimbal = np.array([0.0, 0.0], np.float64)
        self.goal = np.array(opts.get("goal", [c.size - 2, c.size - 2]), dtype=np.float64)
        if self.goal.shape != (2,) or not np.all(np.isfinite(self.goal)):
            raise ValueError("goal must contain two finite coordinates")
        self.reported, self.seen, self.damaged, self.false_reports = set(), set(), set(), set()
        self.trail = deque(maxlen=600)
        self.trail.append(self.robot[:2].tolist())
        self.last_action = np.zeros(4, np.float32)
        self.last_gimbal_rates = np.zeros(2)
        self.done, self.reason, self.reward = False, "", 0.0
        self.total_reward = 0.0
        self._hold = 0.0
        self.collisions = 0
        self._serial = {k: 0 for k in ("lidar", "depth", "camera")}
        self._dropped = {k: 0 for k in self._serial}
        self._next = {k: 0.0 for k in self._serial}
        self._pending = {k: deque() for k in self._serial}
        self.frames = {}
        self._new = {k: False for k in self._serial}
        self.validation = None
        supplied = opts.get("scene")
        spawn_error = None
        for attempt in range(12):
            self.scene = supplied if supplied is not None else Scene.generate(
                c, self.np_random, None if random_start else self.robot[:2], self.goal)
            if random_start:
                try:
                    self.robot[:3] = self.scene.sample_start(self.np_random)
                    spawn_error = None
                except ValueError as exc:
                    spawn_error = exc
                    if supplied is not None:
                        raise
                    if attempt == 11:
                        break
                    continue
                self.trail.clear()
                self.trail.append(self.robot[:2].tolist())
            if not c.validate_scene:
                break
            self.validation = self.scene.validate(self.robot[:2], self.goal)
            if self.validation["connected"] and not self.validation["unobservable_targets"]:
                break
            if supplied is not None:
                raise ValueError(f"invalid supplied scene: {self.validation}")
        else:
            raise ValueError(f"unable to generate validated scene: {self.validation}")
        if random_start and spawn_error is not None:
            raise ValueError(f"unable to generate a scene with a blank random start: {spawn_error}")
        self.scene_id += 1
        self._pose_noise = np.zeros(2)
        self._next_uwb = 0.0
        self._sense(initial=True)
        self._counts = (0, 0, 0)
        return self._observation(), self._info()

    def set_goal(self, goal):
        p = np.asarray(goal, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)) or not self.scene.navigable(p):
            raise ValueError("goal must be inside the field and clear for the vehicle")
        self.goal = p
        self.mode = "goto"
        self._hold = 0.0
        self.done, self.reason = False, ""

    def _basis(self):
        _, body_pitch, body_roll = self._body_pose()
        yaw = float(self.robot[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(body_pitch), math.sin(body_pitch)
        cr, sr = math.cos(body_roll), math.sin(body_roll)
        # Vehicle coordinates are forward, left, up. Positive pitch raises the
        # nose and positive roll raises the left side.
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], np.float32)
        ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], np.float32)
        body = rz @ ry @ rx
        gy, gp = float(self.gimbal[0]), float(self.gimbal[1])
        f_local = np.array([math.cos(gp) * math.cos(gy),
                            math.cos(gp) * math.sin(gy), math.sin(gp)], np.float32)
        right_local = np.array([math.sin(gy), -math.cos(gy), 0.0], np.float32)
        up_local = np.cross(right_local, f_local)
        return body @ f_local, body @ right_local, body @ up_local

    def clear_body_pose_cache(self):
        """Invalidate the support-pose cache after in-place terrain edits.

        Scene terrain is static during normal episodes. Call this if a caller
        deliberately changes terrain geometry without replacing the scene.
        """
        self._body_pose_cache_scene = None
        self._body_pose_cache_terrain = None
        self._body_pose_cache_key = None
        self._body_pose_cache_value = None

    def _body_pose(self):
        """Return support-ground Z, nose-up pitch, and left-up roll.

        This is a quasi-static track-contact proxy; it does not model suspension
        or chassis dynamics. Closely spaced samples let small rocks perturb pose.
        """
        if self.scene is None or self.scene.terrain is None:
            self.clear_body_pose_cache()
            return 0.0, 0.0, 0.0
        c = self.cfg
        yaw = float(self.robot[2])
        terrain = self.scene.terrain
        key = (float(self.robot[0]), float(self.robot[1]), yaw,
               float(c.robot_length), float(c.robot_width))
        if (self._body_pose_cache_scene is self.scene
                and self._body_pose_cache_terrain is terrain
                and self._body_pose_cache_key == key):
            return self._body_pose_cache_value
        forward = np.array([math.cos(yaw), math.sin(yaw)], np.float64)
        left = np.array([-math.sin(yaw), math.cos(yaw)], np.float64)
        longitudinal = np.linspace(-c.robot_length / 2, c.robot_length / 2,
                                   max(3, int(math.ceil(c.robot_length / 0.05)) + 1))
        track_heights = []
        for side in (1.0, -1.0):
            points = (self.robot[:2][None, :]
                      + longitudinal[:, None] * forward[None, :]
                      + (side * c.robot_width / 2) * left[None, :])
            heights = np.asarray(terrain.surface_height(points), dtype=np.float64).reshape(-1)
            track_heights.append(heights)
        # Each short track segment uses its highest contact sample. This gives
        # a small stone on one track a visible quasi-static bump, including at
        # the vehicle midpoint where four-corner sampling would miss it.
        segment_count = 4
        left_h, right_h = [np.asarray([np.max(part) for part in np.array_split(h, segment_count)])
                           for h in track_heights]
        front = 0.5 * (left_h[-1] + right_h[-1])
        rear = 0.5 * (left_h[0] + right_h[0])
        z = float(np.mean(np.concatenate((left_h, right_h))))
        pitch = math.atan2(float(front - rear), c.robot_length)
        roll = math.atan2(float(np.mean(left_h) - np.mean(right_h)), c.robot_width)
        result = (z, pitch, roll)
        self._body_pose_cache_scene = self.scene
        self._body_pose_cache_terrain = terrain
        self._body_pose_cache_key = key
        self._body_pose_cache_value = result
        return result

    def _origin(self):
        z, pitch, roll = self._body_pose()
        yaw = float(self.robot[2])
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], np.float32)
        ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], np.float32)
        body = rz @ ry @ rx
        center = np.array([self.robot[0], self.robot[1], z], np.float32)
        return center + body @ np.array([0.0, 0.0, self.cfg.camera_height], np.float32)

    def _capture(self, kind):
        c, s = self.cfg, self.scene
        self._serial[kind] += 1
        _, body_pitch, body_roll = self._body_pose()
        frame = {"seq": self._serial[kind], "capture": self.time,
                 "pose": np.concatenate((self.robot[:3], self.gimbal)).copy(),
                 "origin": self._origin().copy(),
                 "body_attitude": np.asarray([body_roll, body_pitch], dtype=np.float32)}
        origin = frame["origin"]
        f, right, up = self._basis()
        if kind == "lidar":
            # Laser local x points along camera optical axis, local y to its left.
            d = self.laser_local[:, :1] * f + self.laser_local[:, 1:2] * (-right)
            ideal_ranges, hits = s.cast(origin, d, c.lidar_range)
            valid = (ideal_ranges >= c.lidar_min_range) & (ideal_ranges < c.lidar_range - 1e-4)
            hit_kind = np.zeros(len(hits), dtype=np.uint8)
            hit_kind[valid & (hits == -1)] = 1  # ground
            hit_kind[valid & (hits != -1) & (hits > Scene._ROCK_HIT)] = 2  # plant geometry
            hit_kind[valid & (hits <= Scene._ROCK_HIT)] = 3  # rock; truth ids stay private
            ranges = np.full(len(ideal_ranges), c.lidar_range, dtype=np.float32)
            if np.any(valid):
                error_bound = ideal_ranges[valid] * c.lidar_relative_error
                relative_noise = self.np_random.uniform(-1.0, 1.0, int(valid.sum())) * error_bound
                additive_noise = self.np_random.normal(0, c.range_noise, int(valid.sum()))
                total_noise = np.clip(relative_noise + additive_noise, -error_bound, error_bound)
                measured = ideal_ranges[valid] + total_noise
                ranges[valid] = np.clip(measured, c.lidar_min_range, c.lidar_range)
            frame.update(ranges=ranges.astype(np.float32), valid=valid, hit_kind=hit_kind, directions=d,
                         scan_pitch=float(math.asin(float(np.clip(f[2], -1.0, 1.0)))))
        elif kind == "depth":
            # Rectified stereo uses a rectangular pinhole image. Image Y points
            # down, which is opposite the field's camera-up basis vector.
            d = (f + self.depth_camera_xy[:, :1] * right
                 - self.depth_camera_xy[:, 1:2] * up)
            d /= np.linalg.norm(d, axis=1)[:, None]
            forward_cos = d @ f
            max_ray_range = c.depth_range / float(np.min(forward_cos))
            ranges, hits = s.cast(origin, d, max_ray_range)
            z_depth = ranges * forward_cos
            # Scene.cast uses -1 for both ground and no-return; a ground return
            # is distinguished by its range being strictly before the ray limit.
            returned = (hits != -1) | (ranges < max_ray_range - 1e-4)
            valid = (returned & (z_depth >= c.depth_min_range)
                     & (z_depth <= c.depth_range + 1e-6))
            values = np.full(len(ranges), c.depth_range, dtype=np.float32)
            if np.any(valid):
                measured = z_depth[valid] + self.np_random.normal(0, c.range_noise, int(valid.sum()))
                values[valid] = np.clip(measured, c.depth_min_range, c.depth_range)
            frame.update(values=values.reshape(c.depth_height, c.depth_width),
                         valid=valid.reshape(c.depth_height, c.depth_width),
                         unit="m", convention="camera_z", projection="pinhole",
                         resolution=[c.depth_width, c.depth_height],
                         shape=[c.depth_height, c.depth_width],
                         native_resolution=[DEPTH_NATIVE_WIDTH, DEPTH_NATIVE_HEIGHT],
                         intrinsics=self.depth_intrinsics.copy(),
                         calibration_id=DEPTH_GEOMETRY_ID,
                         calibration_source=DEPTH_GEOMETRY_SOURCE,
                         min_range=c.depth_min_range, max_range=c.depth_range)
        else:
            delta = s.centers - origin
            dist = np.linalg.norm(delta, axis=1)
            forward = delta @ f
            ids = np.flatnonzero((dist > 0.3) & (dist <= c.camera_range) & (forward > dist * math.cos(math.pi / 6)))
            detections = []
            for leaf in ids:
                # Five points are enough for the MVP's explicit, approximate occlusion model.
                center = s.centers[leaf]
                sample = center + np.array([[0, 0], [.6, 0], [-.6, 0], [0, .6], [0, -.6]], np.float32) @ np.stack((s.u[leaf] * s.axes[leaf, 0], s.v[leaf] * s.axes[leaf, 1]))
                rays = sample - origin
                lengths = np.linalg.norm(rays, axis=1)
                _, hits = s.cast(origin, rays / lengths[:, None], c.camera_range)
                view = (rays @ f) > lengths * math.cos(math.pi / 6)
                visible = np.mean((hits == leaf) & view)
                facing = abs(float(np.dot(s.normals[leaf], delta[leaf] / dist[leaf])))
                if visible < .4 or facing < .08 or self.np_random.random() < c.detection_drop:
                    continue
                quality = visible * (0.6 + 0.4 * facing) * max(.1, 1 - dist[leaf] / c.camera_range)
                confidence = float(np.clip(.5 + .49 * quality + self.np_random.normal(0, .015), 0, .99))
                actual_dry = bool(s.leaf_dry[leaf])
                predicted = int(actual_dry ^ (self.np_random.random() < c.class_error))
                x = .5 + float(np.dot(delta[leaf], right) / forward[leaf]) / (2 * math.tan(math.pi / 6))
                y = .5 - float(np.dot(delta[leaf], up) / forward[leaf]) / (2 * math.tan(math.pi / 6))
                w = min(.9, 2 * s.axes[leaf, 0] / (forward[leaf] * 2 * math.tan(math.pi / 6)))
                h = min(.9, 2 * s.axes[leaf, 1] / (forward[leaf] * 2 * math.tan(math.pi / 6)))
                # Object IDs below are scorer-only, never exported to observations.
                detections.append({"box": [x, y, float(w), float(h)], "class_id": predicted,
                                   "confidence": confidence, "range": float(dist[leaf]) if dist[leaf] <= c.depth_range else None,
                                   "relative": np.array([np.dot(delta[leaf], f), np.dot(delta[leaf], right), np.dot(delta[leaf], up)]),
                                   "plant": int(s.leaf_plant[leaf]), "leaf": int(leaf)})
            detections.sort(key=lambda a: a["confidence"], reverse=True)
            frame["detections"] = detections[:c.max_detections]
        return frame

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
                        self.seen.add(d["plant"])
                        # MVP: automatic detector report, NOT a learned report/finish head.
                        if d["class_id"] == 1 and d["confidence"] >= .68 and d["range"] is not None:
                            (self.reported if self.scene.targets[d["plant"]] else self.false_reports).add(d["plant"])
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

    def _rock_sweep(self, before, after) -> tuple[bool, float]:
        """Return whether the swept vehicle footprint hits a blocking rock.

        The returned fraction is the last sampled safe pose, so a fast step
        cannot tunnel through an obstacle between its endpoints.
        """
        terrain = self.scene.terrain
        if terrain is None or not np.any(terrain.rock_blocking):
            return False, 1.0
        c = self.cfg
        rocks = terrain.rock_xy[terrain.rock_blocking]
        radii = terrain.rock_radius[terrain.rock_blocking]
        displacement = float(np.linalg.norm(after[:2] - before[:2]))
        dyaw = wrap(float(after[2] - before[2]))
        subdivisions = max(1, math.ceil(displacement / 0.075), math.ceil(abs(dyaw) / 0.05))
        corner_radius = math.hypot(c.robot_length, c.robot_width) / 2
        # A corner travels farther than the chassis center while yawing. Check
        # each interval midpoint with its half-interval motion bound; this
        # covers the continuous sweep while leaving the initial pose uninflated.
        sample_margin = 0.5 * (displacement + corner_radius * abs(dyaw)) / subdivisions
        swept_radii = radii + sample_margin
        def intersects(t, test_radii):
            p = before[:2] + t * (after[:2] - before[:2])
            yaw = before[2] + t * dyaw
            delta = rocks - p
            x = delta[:, 0] * math.cos(yaw) + delta[:, 1] * math.sin(yaw)
            y = -delta[:, 0] * math.sin(yaw) + delta[:, 1] * math.cos(yaw)
            dx = np.maximum(np.abs(x) - c.robot_length / 2, 0.0)
            dy = np.maximum(np.abs(y) - c.robot_width / 2, 0.0)
            return bool(np.any(dx * dx + dy * dy <= test_radii * test_radii))

        if intersects(0.0, radii):
            return True, 0.0
        for index in range(1, subdivisions + 1):
            midpoint = (index - 0.5) / subdivisions
            if intersects(midpoint, swept_radii):
                return True, (index - 1) / subdivisions
        return False, 1.0

    def _advance_motion(self, a):
        """Advance vehicle and gimbal motion for one clipped action.

        Subclasses can override this hook to supply externally simulated motion
        while leaving the shared sensing, reward, and episode logic intact.
        Returns ``(before, old_distance, rock_collision, boundary)``.
        """
        c = self.cfg
        before = self.robot.copy()
        old_distance = np.linalg.norm(self.goal - before[:2])
        self.robot[3] += np.clip(a[0] * c.max_speed - self.robot[3], -c.acceleration * c.dt, c.acceleration * c.dt)
        self.robot[4] += np.clip(a[1] * c.max_turn_rate - self.robot[4], -c.angular_acceleration * c.dt, c.angular_acceleration * c.dt)
        candidate = self.robot.copy()
        mid = before[2] + candidate[4] * c.dt / 2
        candidate[:2] = before[:2] + candidate[3] * c.dt * np.array([math.cos(mid), math.sin(mid)])
        candidate[2] = wrap(before[2] + candidate[4] * c.dt)
        rock_collision, safe_fraction = self._rock_sweep(before, candidate)
        if rock_collision:
            dyaw = wrap(float(candidate[2] - before[2]))
            self.robot[:2] = before[:2] + safe_fraction * (candidate[:2] - before[:2])
            self.robot[2] = wrap(before[2] + safe_fraction * dyaw)
            self.robot[3:5] = 0.0
            self.collisions += 1
        else:
            self.robot[:] = candidate
        r = math.hypot(c.robot_length, c.robot_width) / 2
        boundary = bool(np.any(self.robot[:2] < r) or np.any(self.robot[:2] > c.size - r))
        self.robot[:2] = np.clip(self.robot[:2], r, c.size - r)
        if boundary:
            self.robot[3] = 0
        self.last_gimbal_rates = a[2:] * [c.gimbal_yaw_rate, c.gimbal_pitch_rate]
        self.gimbal[0] = wrap(self.gimbal[0] + self.last_gimbal_rates[0] * c.dt)
        self.gimbal[1] = np.clip(self.gimbal[1] + self.last_gimbal_rates[1] * c.dt, c.pitch_min, c.pitch_max)

        return before, old_distance, rock_collision, boundary

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
            reward += c.progress_weight * (old_distance - distance)
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
        self.last_action = np.array([self.robot[3] / c.max_speed, self.robot[4] / c.max_turn_rate, *a[2:]], np.float32)
        self.reward = float(reward)
        self.total_reward += self.reward
        return self._observation(), self.reward, self.done and not truncated, bool(truncated), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("episode ended; call reset()")
        a = np.asarray(action, np.float32)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError("action must contain four finite values")
        a = np.clip(a, -1, 1)
        before, old_distance, rock_collision, boundary = self._advance_motion(a)
        return self._finish_step(a, before, old_distance, rock_collision, boundary)

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
