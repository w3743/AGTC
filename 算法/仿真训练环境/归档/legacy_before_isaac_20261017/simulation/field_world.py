"""Offline exporter for the field_sim geometry payload to Gazebo Harmonic.

The exporter consumes the already-materialized scene truth. It never draws a
new random scene, and it deliberately exports the crop as three visual meshes
plus one mesh per rock, instead of creating thousands of leaf models.

Run from the repository root with::

    python -m simulation.field_world --scene-json results/gazebo_field_source/current_scene.json \
        --state-json results/gazebo_field_source/current_state.json --output results/gazebo_field/current

Only stdlib, NumPy, and Pillow are required. Gazebo itself is not launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HEIGHTMAP_RESOLUTION = 513  # 2^n + 1; 0.1953125 m / datum for a 100 m field.
LEAF_BOUNDARY_STEPS = 24  # Same boundary samples as field_sim/web/index.html.
STEM_SIDES = 8
ROCK_LATITUDE_STEPS = 8
ROCK_LONGITUDE_STEPS = 24

TOPICS = {
    "cmd_vel": "/field/cmd_vel",
    "odom": "/field/odom",
    "tf": "/field/tf",
    "joint_states": "/field/joint_states",
    "lidar": "/field/lidar/scan",
    "depth": "/field/depth/image",
    "depth_camera_info": "/field/depth/camera_info",
    "imu": "/field/imu/data",
    "magnetometer": "/field/magnetometer/data",
    "gimbal_yaw_cmd": "/field/gimbal/yaw/cmd_pos",
    "gimbal_pitch_cmd": "/field/gimbal/pitch/cmd_pos",
    "track_left_velocity": "/field/track/left/velocity",
    "track_right_velocity": "/field/track/right/velocity",
    "track_left_center_of_rotation": "/field/track/left/center_of_rotation",
    "track_right_center_of_rotation": "/field/track/right/center_of_rotation",
}


def _fmt(value: float | int) -> str:
    """Compact deterministic float formatting for SDF and OBJ."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), ".9g")


def _triple(values) -> str:
    return " ".join(_fmt(v) for v in values)


def _child(parent: ET.Element, name: str, value: str | float | int | None = None,
           attrs: dict[str, str] | None = None) -> ET.Element:
    node = ET.SubElement(parent, name, attrs or {})
    if value is not None:
        node.text = str(value)
    return node


def _pose(parent: ET.Element, xyz, rpy=(0.0, 0.0, 0.0), *, relative_to: str | None = None) -> ET.Element:
    attrs = {"relative_to": relative_to} if relative_to else None
    return _child(parent, "pose", _triple((*xyz, *rpy)), attrs)


def _material(parent: ET.Element, color: tuple[float, float, float, float]) -> None:
    mat = _child(parent, "material")
    _child(mat, "ambient", _triple(color))
    _child(mat, "diffuse", _triple(color))
    _child(mat, "specular", "0.05 0.05 0.05 1")


def _geometry_mesh(parent: ET.Element, uri: str, scale=(1.0, 1.0, 1.0)) -> None:
    geometry = _child(parent, "geometry")
    mesh = _child(geometry, "mesh")
    _child(mesh, "uri", uri)
    if tuple(scale) != (1.0, 1.0, 1.0):
        _child(mesh, "scale", _triple(scale))


def _geometry_box(parent: ET.Element, size) -> None:
    geometry = _child(parent, "geometry")
    box = _child(geometry, "box")
    _child(box, "size", _triple(size))


def _visual(link: ET.Element, name: str, uri: str | None = None, *, color=None,
            pose=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), box=None,
            cylinder=None, sphere=None, scale=(1.0, 1.0, 1.0)) -> ET.Element:
    visual = _child(link, "visual", attrs={"name": name})
    _pose(visual, pose, rpy)
    if uri:
        _geometry_mesh(visual, uri, scale)
    elif box:
        _geometry_box(visual, box)
    elif cylinder:
        geometry = _child(visual, "geometry")
        cyl = _child(geometry, "cylinder")
        _child(cyl, "radius", _fmt(cylinder[0]))
        _child(cyl, "length", _fmt(cylinder[1]))
    elif sphere:
        geometry = _child(visual, "geometry")
        sph = _child(geometry, "sphere")
        _child(sph, "radius", _fmt(sphere))
    if color:
        _material(visual, color)
    _child(visual, "cast_shadows", "false")
    return visual


def _collision(link: ET.Element, name: str, *, uri: str | None = None, scale=(1.0, 1.0, 1.0),
               pose=(0.0, 0.0, 0.0), box=None) -> ET.Element:
    collision = _child(link, "collision", attrs={"name": name})
    _pose(collision, pose)
    if uri:
        _geometry_mesh(collision, uri, scale)
    else:
        _geometry_box(collision, box)
    return collision


def _inertial(link: ET.Element, mass: float, com_z: float, size) -> None:
    inertia_node = _child(link, "inertial")
    _pose(inertia_node, (0.0, 0.0, com_z))
    _child(inertia_node, "mass", _fmt(mass))
    x, y, z = map(float, size)
    inertia = _child(inertia_node, "inertia")
    for tag, value in (
        ("ixx", mass * (y * y + z * z) / 12.0),
        ("iyy", mass * (x * x + z * z) / 12.0),
        ("izz", mass * (x * x + y * y) / 12.0),
        ("ixy", 0.0), ("ixz", 0.0), ("iyz", 0.0),
    ):
        _child(inertia, tag, _fmt(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payloads(scene_path: Path, state_path: Path | None):
    scene_raw = scene_path.read_bytes()
    scene = json.loads(scene_raw.decode("utf-8"))
    state = None
    state_raw = None
    if state_path is not None:
        state_raw = state_path.read_bytes()
        state = json.loads(state_raw.decode("utf-8"))
        if state.get("scene_id") != scene.get("scene_id"):
            raise ValueError(
                f"scene/state scene_id mismatch: {scene.get('scene_id')} != {state.get('scene_id')}"
            )
    validate_scene_payload(scene)
    return scene, state, hashlib.sha256(scene_raw).hexdigest(), (
        hashlib.sha256(state_raw).hexdigest() if state_raw is not None else None
    )


def validate_scene_payload(scene: dict) -> None:
    if scene.get("schema") != "field-geometry-debug-v1":
        raise ValueError("expected field-geometry-debug-v1 scene payload")
    if scene.get("coordinate_frame") != "field_xyz_z_up_m":
        raise ValueError("scene payload must use field_xyz_z_up_m")
    size = float(scene.get("size", 0.0))
    if not math.isfinite(size) or size <= 0:
        raise ValueError("scene size must be finite and positive")
    config = scene.get("config") or {}
    plants = scene.get("plants") or []
    geometry = scene.get("leaf_geometry") or {}
    centers = np.asarray(geometry.get("centers", []), dtype=np.float64)
    normals = np.asarray(geometry.get("normals", []), dtype=np.float64)
    axes = np.asarray(geometry.get("axes", []), dtype=np.float64)
    u = np.asarray(geometry.get("u", []), dtype=np.float64)
    v = np.asarray(geometry.get("v", []), dtype=np.float64)
    plant_ids = np.asarray(geometry.get("plant", []), dtype=np.int64)
    dry = np.asarray(geometry.get("dry", []), dtype=bool)
    count = len(centers)
    if geometry.get("shape") != "tapered_blade_v1":
        raise ValueError("only tapered_blade_v1 leaf geometry is supported")
    if centers.shape != (count, 3) or normals.shape != (count, 3) or u.shape != (count, 3) or v.shape != (count, 3):
        raise ValueError("leaf centers, normals, u and v must be matching N x 3 arrays")
    if axes.shape != (count, 2) or plant_ids.shape != (count,) or dry.shape != (count,):
        raise ValueError("leaf axes, plant ids and dry states must match leaf centers")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("leaf geometry contains non-finite values")
    if np.any(plant_ids < 0) or np.any(plant_ids >= len(plants)) or np.any(axes <= 0):
        raise ValueError("leaf geometry has invalid plant ids or axes")
    if len(plants) != int(config.get("plants", len(plants))):
        raise ValueError("plant count differs from config")
    expected = int(config.get("leaves_per_plant", 0))
    if expected and count != len(plants) * expected:
        raise ValueError("leaf count differs from plants x leaves_per_plant")
    ids, counts = np.unique(plant_ids, return_counts=True)
    if len(ids) != len(plants) or np.any(counts != expected):
        raise ValueError("every plant must have the configured number of leaves")
    if geometry.get("plant") != [int(v) for v in plant_ids]:
        # JSON IDs are already in global order; this is a helpful but nonfatal
        # consistency guard against a malformed payload with reordered IDs.
        raise ValueError("leaf plant IDs are malformed")
    terrain = scene.get("terrain") or {}
    if terrain.get("enabled"):
        waves = np.asarray(terrain.get("wave_vectors4x2", []), dtype=np.float64)
        amp = np.asarray(terrain.get("wave_amplitudes4", []), dtype=np.float64)
        phase = np.asarray(terrain.get("wave_phases4", []), dtype=np.float64)
        if waves.shape != (4, 2) or amp.shape != (4,) or phase.shape != (4,):
            raise ValueError("enabled terrain must contain exactly four 2D sine waves")
        if not np.all(np.isfinite(waves)) or not np.all(np.isfinite(amp)) or not np.all(np.isfinite(phase)) or np.any(amp < 0):
            raise ValueError("terrain wave parameters must be finite with nonnegative amplitudes")
        for rock in terrain.get("rocks", []):
            values = [rock.get(k) for k in ("x", "y", "z", "radius", "height")]
            if not all(math.isfinite(float(v)) for v in values) or float(rock["radius"]) <= 0 or float(rock["height"]) <= 0:
                raise ValueError("rock geometry contains invalid values")


def terrain_samples(scene: dict, resolution: int = HEIGHTMAP_RESOLUTION):
    """Return an exact wave sample grid and its normalized sampled-height encoding."""
    size = float(scene["size"])
    terrain = scene["terrain"]
    base_z = float(terrain.get("base_height", 0.0))
    if terrain.get("enabled"):
        vectors = np.asarray(terrain["wave_vectors4x2"], dtype=np.float64)
        amplitudes = np.asarray(terrain["wave_amplitudes4"], dtype=np.float64)
        phases = np.asarray(terrain["wave_phases4"], dtype=np.float64)
    else:
        vectors = np.zeros((4, 2), dtype=np.float64)
        amplitudes = np.zeros(4, dtype=np.float64)
        phases = np.zeros(4, dtype=np.float64)
    axis = np.linspace(0.0, size, resolution, dtype=np.float64)
    xs, ys = np.meshgrid(axis, axis, indexing="xy")
    heights = np.full((resolution, resolution), base_z, dtype=np.float64)
    for wave, amplitude, phase in zip(vectors, amplitudes, phases):
        heights += amplitude * np.sin(xs * wave[0] + ys * wave[1] + phase)
    # Use the extrema actually represented by the PNG. Ogre2 rescales sampled
    # heights using the image min / max; matching those extrema also keeps the
    # Bullet collision heightfield and rendered terrain on the same datum.
    low, high = float(np.min(heights)), float(np.max(heights))
    if high == low:
        pixels = np.zeros((resolution, resolution), dtype=np.uint8)
        z_size = 0.0
    else:
        normalized = np.clip((heights - low) / (high - low), 0.0, 1.0)
        pixels = np.rint(normalized * 255.0).astype(np.uint8)
        z_size = high - low
    reconstructed = low + pixels.astype(np.float64) / 255.0 * z_size if z_size else heights.copy()
    max_quant_error = float(np.max(np.abs(reconstructed - heights))) if heights.size else 0.0
    return heights, pixels, low, z_size, max_quant_error


def terrain_height_at(scene: dict, x: float, y: float) -> float:
    """Evaluate the source four-wave terrain at field-local XY coordinates."""
    terrain = scene["terrain"]
    height = float(terrain.get("base_height", 0.0))
    if not terrain.get("enabled"):
        return height
    vectors = terrain["wave_vectors4x2"]
    amplitudes = terrain["wave_amplitudes4"]
    phases = terrain["wave_phases4"]
    return height + sum(float(a) * math.sin(float(x) * float(v[0]) +
                                              float(y) * float(v[1]) + float(p))
                        for v, a, p in zip(vectors, amplitudes, phases))


def uwb_anchor_layout(scene: dict, clearance_m: float = 1.8) -> list[dict]:
    """Create a 3x3 field-spanning anchor grid at a constant terrain clearance."""
    size = float(scene["size"])
    half = size / 2.0
    layout = []
    coordinates = [(fraction * size, fraction * size)
                   for fraction in (0.1, 0.5, 0.9)]
    index = 0
    for source_y in (0.1 * size, 0.5 * size, 0.9 * size):
        for source_x in (0.1 * size, 0.5 * size, 0.9 * size):
            index += 1
            ground_z = terrain_height_at(scene, source_x, source_y)
            layout.append({
                "id": index,
                "model": f"r1_rover_anchor_{index}",
                "link": f"uwb_anchor_{index}",
                "source_xy_m": [source_x, source_y],
                "world_xyz_m": [source_x - half, source_y - half,
                                ground_z + float(clearance_m)],
                "terrain_z_m": ground_z,
                "clearance_m": float(clearance_m),
            })
    return layout


def _write_terrain_png(scene: dict, path: Path):
    heights, pixels, low, z_size, max_error = terrain_samples(scene)
    # Image top row represents source y=size. This north-up convention is
    # recorded explicitly; Gazebo's imported orientation remains unverified.
    Image.fromarray(np.flipud(pixels), mode="L").save(path, optimize=True)
    return {
        "enabled": bool(scene["terrain"].get("enabled", False)),
        "source": "four_sine_waves" if scene["terrain"].get("enabled") else "flat_base_height",
        "resolution_px": [HEIGHTMAP_RESOLUTION, HEIGHTMAP_RESOLUTION],
        "horizontal_size_m": [float(scene["size"]), float(scene["size"])],
        "datum_spacing_m": float(scene["size"]) / (HEIGHTMAP_RESOLUTION - 1),
        "grayscale_bits": 8,
        "grayscale_range": [0, 255],
        "heightmap_size_z_m": z_size,
        # Harmonic's Bullet adapter ignores HeightmapSdf::Position(), while
        # Ogre2 heightmap rendering uses it and does not apply the visual pose.
        # Therefore put the low datum on collision pose and visual heightmap pos.
        "heightmap_collision_pose_z_m": low,
        "heightmap_visual_pos_z_m": low,
        "heightmap_link_pose_z_m": 0.0,
        "vertical_mapping": "world_z = low_sample + gray/255 * sampled_span",
        "sampled_wave_min_m": float(np.min(heights)),
        "sampled_wave_max_m": float(np.max(heights)),
        "max_sample_quantization_error_m": max_error,
        "runtime_trailing_datum_crop_observed": True,
        "image_axes": {
            "column_0_source_x_m": 0.0,
            "column_last_source_x_m": float(scene["size"]),
            "row_0_source_y_m": float(scene["size"]),
            "row_last_source_y_m": 0.0,
            "world_xy": "source_xy - size/2; Gazebo heightmap image orientation not runtime-verified",
        },
        "original_wave_parameters": {
            "base_height_m": float(scene["terrain"].get("base_height", 0.0)),
            "wave_vectors4x2_rad_per_m": scene["terrain"].get("wave_vectors4x2", []),
            "amplitudes4_m": scene["terrain"].get("wave_amplitudes4", []),
            "phases4_rad": scene["terrain"].get("wave_phases4", []),
        },
        "continuous_surface_match": "sampled; Gazebo interpolation and axis orientation require runtime comparison",
    }


def _write_heightmap_textures(assets: Path) -> dict:
    """Write local single-color heightmap diffuse and flat-normal maps."""
    soil_rgb = (0x57, 0x4F, 0x33)
    Image.new("RGB", (8, 8), soil_rgb).save(assets / "soil_diffuse.png", optimize=True)
    Image.new("RGB", (8, 8), (128, 128, 255)).save(assets / "flat_normal.png", optimize=True)
    return {"diffuse": "assets/soil_diffuse.png", "normal": "assets/flat_normal.png",
            "diffuse_rgb": list(soil_rgb), "normal_rgb": [128, 128, 255],
            "tile_size_m": 10.0, "single_layer": True}


def _leaf_ring(center, u, v, half_length, half_width):
    points = []
    for side in (1.0, -1.0):
        steps = range(LEAF_BOUNDARY_STEPS + 1) if side > 0 else range(LEAF_BOUNDARY_STEPS, -1, -1)
        for step in steps:
            x = -1.0 + 2.0 * step / LEAF_BOUNDARY_STEPS
            y = side * math.sqrt(max(0.0, (1.0 - x * x) * (1.0 - abs(x))))
            points.append(np.asarray(center, dtype=np.float64) +
                          np.asarray(u, dtype=np.float64) * half_length * x +
                          np.asarray(v, dtype=np.float64) * half_width * y)
    return points


def _write_leaf_obj(path: Path, geometry: dict, leaf_ids: list[int], center_offset: float = 0.0) -> int:
    centers = geometry["centers"]
    normals = geometry["normals"]
    us = geometry["u"]
    vs = geometry["v"]
    axes = geometry["axes"]
    face_count = 0
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Static field_sim tapered_blade_v1 surface; double-sided per triangle.\n")
        vertex_base = 1
        normal_base = 1
        for leaf_id in leaf_ids:
            ring = _leaf_ring(centers[leaf_id], us[leaf_id], vs[leaf_id], *axes[leaf_id])
            for point in ring:
                stream.write(f"v {_fmt(point[0] - center_offset)} {_fmt(point[1] - center_offset)} {_fmt(point[2])}\n")
            center = np.asarray(centers[leaf_id], dtype=np.float64)
            stream.write(f"v {_fmt(center[0] - center_offset)} {_fmt(center[1] - center_offset)} {_fmt(center[2])}\n")
            normal = np.asarray(normals[leaf_id], dtype=np.float64)
            length = float(np.linalg.norm(normal))
            if length <= 1e-12:
                raise ValueError(f"leaf {leaf_id} has a zero normal")
            normal /= length
            stream.write(f"vn {_fmt(normal[0])} {_fmt(normal[1])} {_fmt(normal[2])}\n")
            stream.write(f"vn {_fmt(-normal[0])} {_fmt(-normal[1])} {_fmt(-normal[2])}\n")
            center_index = vertex_base + len(ring)
            for i in range(len(ring)):
                current = vertex_base + i
                nxt = vertex_base + ((i + 1) % len(ring))
                # Leaf boundary order is clockwise in (u,v); this winding
                # points the front face along the source normal u x v.
                stream.write(f"f {center_index}//{normal_base} {nxt}//{normal_base} {current}//{normal_base}\n")
                stream.write(f"f {center_index}//{normal_base + 1} {current}//{normal_base + 1} {nxt}//{normal_base + 1}\n")
                face_count += 2
            vertex_base += len(ring) + 1
            normal_base += 2
    return face_count


def _write_stem_obj(path: Path, plants: list[dict], center_offset: float = 0.0) -> int:
    face_count = 0
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Visual-only segmented stem cylinders; no plant collider.\n")
        vertex_base = 1
        for plant in plants:
            cx, cy = float(plant["x"]) - center_offset, float(plant["y"]) - center_offset
            z0, height, radius = float(plant.get("base_z", plant["z"])), float(plant["height"]), float(plant["stem_radius"])
            rings = []
            for z, r in ((z0, radius), (z0 + height, radius * 0.72)):
                ring = []
                for i in range(STEM_SIDES):
                    angle = 2 * math.pi * i / STEM_SIDES
                    point = (cx + r * math.cos(angle), cy + r * math.sin(angle), z)
                    ring.append(point)
                    stream.write(f"v {_fmt(point[0])} {_fmt(point[1])} {_fmt(point[2])}\n")
                rings.append(ring)
            bottom_center = vertex_base + 2 * STEM_SIDES
            top_center = bottom_center + 1
            stream.write(f"v {_fmt(cx)} {_fmt(cy)} {_fmt(z0)}\n")
            stream.write(f"v {_fmt(cx)} {_fmt(cy)} {_fmt(z0 + height)}\n")
            for i in range(STEM_SIDES):
                j = (i + 1) % STEM_SIDES
                b0, b1 = vertex_base + i, vertex_base + j
                t0, t1 = vertex_base + STEM_SIDES + i, vertex_base + STEM_SIDES + j
                # The side and two cap triangles are intentionally double sided.
                for a, b, c in ((b0, b1, t1), (b0, t1, t0),
                                (bottom_center, b1, b0), (top_center, t0, t1)):
                    stream.write(f"f {a} {b} {c}\n")
                    stream.write(f"f {a} {c} {b}\n")
                    face_count += 2
            vertex_base += 2 * STEM_SIDES + 2
    return face_count


def _write_rock_obj(path: Path) -> int:
    faces = 0
    rings = ROCK_LATITUDE_STEPS
    sides = ROCK_LONGITUDE_STEPS
    vertex_count = (rings + 1) * sides
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# Closed unit upper-ellipsoid cap, radius 1 and height 1; explicit collision normals.\n")
        for lat in range(rings + 1):
            theta = (math.pi / 2) * lat / rings
            z = math.sin(theta)
            radius = math.cos(theta)
            for lon in range(sides):
                phi = 2 * math.pi * lon / sides
                stream.write(f"v {_fmt(radius * math.cos(phi))} {_fmt(radius * math.sin(phi))} {_fmt(z)}\n")
        bottom_center = vertex_count + 1
        stream.write("v 0 0 0\n")
        for lat in range(rings + 1):
            theta = (math.pi / 2) * lat / rings
            z = math.sin(theta)
            radius = math.cos(theta)
            for lon in range(sides):
                phi = 2 * math.pi * lon / sides
                stream.write(f"vn {_fmt(radius * math.cos(phi))} {_fmt(radius * math.sin(phi))} {_fmt(z)}\n")
        bottom_normal_base = vertex_count + 1
        for _ in range(sides + 1):
            stream.write("vn 0 0 -1\n")
        for lat in range(rings):
            for lon in range(sides):
                a = lat * sides + lon + 1
                b = lat * sides + (lon + 1) % sides + 1
                c = (lat + 1) * sides + (lon + 1) % sides + 1
                d = (lat + 1) * sides + lon + 1
                if lat == rings - 1:
                    # Adjacent pole vertices coincide; retain one outward triangle per sector.
                    stream.write(f"f {a}//{a} {b}//{b} {d}//{d}\n")
                    faces += 1
                else:
                    stream.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                    stream.write(f"f {a}//{a} {c}//{c} {d}//{d}\n")
                    faces += 2
        # Close the equator with a downward-facing disk so Bullet receives a watertight solid.
        bottom_center_normal = bottom_normal_base + sides
        for lon in range(sides):
            a = lon + 1
            b = (lon + 1) % sides + 1
            stream.write(f"f {bottom_center}//{bottom_center_normal} {b}//{bottom_normal_base + b} {a}//{bottom_normal_base + a}\n")
            faces += 1
    return faces


def _add_heightmap_model(world: ET.Element, size: float, terrain_meta: dict) -> None:
    model = _child(world, "model", attrs={"name": "field_terrain"})
    _child(model, "static", "true")
    link = _child(model, "link", attrs={"name": "terrain_link"})
    _pose(link, (0.0, 0.0, terrain_meta["heightmap_link_pose_z_m"]))
    for name in ("collision", "visual"):
        parent = _child(link, name, attrs={"name": f"terrain_{name}"})
        # Physics reads collision pose; Ogre2 uses heightmap position directly
        # and ignores visual-parent pose. Keep these offsets separate by design.
        if name == "collision":
            _pose(parent, (0.0, 0.0, terrain_meta["heightmap_collision_pose_z_m"]))
        else:
            _pose(parent, (0.0, 0.0, 0.0))
        geometry = _child(parent, "geometry")
        hm = _child(geometry, "heightmap")
        _child(hm, "uri", "assets/terrain.png")
        _child(hm, "size", _triple((size, size, max(terrain_meta["heightmap_size_z_m"], 1e-6))))
        hm_pos_z = terrain_meta["heightmap_visual_pos_z_m"] if name == "visual" else 0.0
        _child(hm, "pos", _triple((0.0, 0.0, hm_pos_z)))
        _child(hm, "sampling", "1")
        if name == "visual":
            texture = _child(hm, "texture")
            _child(texture, "size", _fmt(terrain_meta["render_texture"]["tile_size_m"]))
            _child(texture, "diffuse", terrain_meta["render_texture"]["diffuse"])
            _child(texture, "normal", terrain_meta["render_texture"]["normal"])
            blend = _child(hm, "blend")
            _child(blend, "min_height", _fmt(terrain_meta["heightmap_visual_pos_z_m"]))
            _child(blend, "fade_dist", _fmt(max(terrain_meta["heightmap_size_z_m"], 0.01)))


def _add_rocks(world: ET.Element, rocks: list[dict], field_size: float) -> dict:
    blocking = 0
    small = 0
    for rock_id, rock in enumerate(rocks):
        is_blocking = bool(rock.get("blocking", False))
        blocking += int(is_blocking)
        small += int(not is_blocking)
        model = _child(world, "model", attrs={"name": f"rock_{rock_id:03d}"})
        _child(model, "static", "true")
        size = float(rock["radius"])
        height = float(rock["height"])
        # Scene coordinates are re-centered into a world from -size/2..+size/2.
        offset = field_size / 2.0
        _pose(model, (float(rock["x"]) - offset, float(rock["y"]) - offset,
                      float(rock.get("base_z", rock["z"]))))
        link = _child(model, "link", attrs={"name": "rock_link"})
        collision = _collision(link, "rock_collision", uri="assets/rock_cap.obj",
                               scale=(size, size, height))
        friction = _child(collision, "surface")
        friction_node = _child(friction, "friction")
        ode = _child(friction_node, "ode")
        _child(ode, "mu", "0.7")
        _child(ode, "mu2", "0.7")
        _visual(link, "rock_visual", "assets/rock_cap.obj", scale=(size, size, height),
                color=(0x89 / 255, 0x91 / 255, 0x95 / 255, 1.0))
    return {"models": len(rocks), "blocking_collision_models": blocking,
            "traversable_collision_models": small, "visual_mesh_asset": "assets/rock_cap.obj"}


def _joint(model: ET.Element, name: str, parent: str, child: str, joint_type: str,
           pose, axis=None, limits=None, *, relative_to=None) -> ET.Element:
    joint = _child(model, "joint", attrs={"name": name, "type": joint_type})
    _child(joint, "parent", parent)
    _child(joint, "child", child)
    _pose(joint, pose[:3], pose[3:], relative_to=relative_to)
    if axis is not None:
        axis_node = _child(joint, "axis")
        _child(axis_node, "xyz", _triple(axis))
        if limits is not None:
            limit = _child(axis_node, "limit")
            _child(limit, "lower", _fmt(limits[0]))
            _child(limit, "upper", _fmt(limits[1]))
            _child(limit, "effort", "20")
            _child(limit, "velocity", "3")
    return joint


def _add_track_link(model: ET.Element, side: str, y: float, mass: float, size) -> None:
    link = _child(model, "link", attrs={"name": f"{side}_track"})
    _pose(link, (0.0, y, 0.055))
    _inertial(link, mass, 0.0, size)
    collision = _collision(link, f"{side}_track_collision", box=size)
    surface = _child(collision, "surface")
    friction = _child(surface, "friction")
    ode = _child(friction, "ode")
    _child(ode, "mu", "0.7")
    _child(ode, "mu2", "150")
    _child(ode, "fdir1", "1 0 0")
    _visual(link, f"{side}_track_visual", color=(0.10, 0.12, 0.13, 1.0), box=size)
    _joint(model, f"{side}_track_fixed_joint", "base_link", f"{side}_track", "fixed",
           (0, y, 0.055, 0, 0, 0), relative_to="base_link")


def _add_sensor(parent: ET.Element, *, name: str, sensor_type: str, topic: str,
                rate: float, pose=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    sensor = _child(parent, "sensor", attrs={"name": name, "type": sensor_type})
    _pose(sensor, pose, rpy)
    _child(sensor, "always_on", "true")
    _child(sensor, "update_rate", _fmt(rate))
    _child(sensor, "visualize", "false")
    _child(sensor, "topic", topic)
    return sensor


def _add_robot(world: ET.Element, scene: dict, state: dict | None,
               mass: float, com_height: float, model_name: str = "field_robot",
               with_uwb: bool = False) -> dict:
    config = scene["config"]
    total = float(mass)
    scale = total / 30.0
    track_mass = 1.0 * scale
    yaw_mass = 0.6 * scale
    pitch_mass = 0.4 * scale
    tag_mass = 0.01 if with_uwb else 0.0
    base_mass = total - 2 * track_mass - yaw_mass - pitch_mass - tag_mass
    track_z = 0.055
    yaw_z = float(config.get("camera_height", 0.4))
    pitch_z = yaw_z
    tag_z = 0.43
    base_com_z = (com_height * total - 2 * track_mass * track_z - yaw_mass * yaw_z -
                  pitch_mass * pitch_z - tag_mass * tag_z) / base_mass
    if base_mass <= 0 or not math.isfinite(base_com_z) or base_com_z < 0:
        raise ValueError("mass/com-height combination puts the base center of mass below its link origin")

    robot = _child(world, "model", attrs={"name": model_name})
    if state and state.get("robot"):
        pose_state = state["robot"]
        field_half = float(scene["size"]) / 2.0
        model_xyz = (float(pose_state["x"]) - field_half, float(pose_state["y"]) - field_half,
                     float(pose_state.get("z", 0.0)))
        model_rpy = (float(pose_state.get("roll", 0.0)), -float(pose_state.get("pitch", 0.0)),
                     float(pose_state.get("yaw", 0.0)))
    else:
        model_xyz, model_rpy = (0.0, 0.0, 0.1), (0.0, 0.0, 0.0)
    _pose(robot, model_xyz, model_rpy)

    body_size = (0.72, 0.14, 0.13)
    track_size = (float(config.get("robot_length", 0.8)), 0.08, 0.11)
    track_offset = (float(config.get("robot_width", 0.3)) - track_size[1]) / 2.0
    base = _child(robot, "link", attrs={"name": "base_link"})
    _inertial(base, base_mass, base_com_z, body_size)
    _collision(base, "chassis_collision", pose=(0.0, 0.0, 0.16), box=body_size)
    _visual(base, "chassis_visual", color=(0.23, 0.27, 0.28, 1.0), pose=(0.0, 0.0, 0.16), box=body_size)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        _add_track_link(robot, side, sign * track_offset, track_mass, track_size)

    gimbal = state.get("gimbal", {}) if state else {}
    yaw_angle = float(gimbal.get("yaw", 0.0))
    pitch_angle = float(gimbal.get("pitch", 0.0))
    yaw_link = _child(robot, "link", attrs={"name": "gimbal_yaw_link"})
    _pose(yaw_link, (0.0, 0.0, yaw_z))
    _inertial(yaw_link, yaw_mass, 0.0, (0.12, 0.12, 0.03))
    _visual(yaw_link, "turret_visual", color=(0.14, 0.24, 0.25, 1.0),
            pose=(0.0, 0.0, 0.0), cylinder=(0.07, 0.025))
    _joint(robot, "gimbal_yaw_joint", "base_link", "gimbal_yaw_link", "revolute",
           (0.0, 0.0, yaw_z, 0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
           limits=(-math.pi, math.pi), relative_to="base_link")

    pitch_link = _child(robot, "link", attrs={"name": "gimbal_pitch_link"})
    _pose(pitch_link, (0.0, 0.0, pitch_z))
    _inertial(pitch_link, pitch_mass, 0.0, (0.10, 0.06, 0.05))
    _joint(robot, "gimbal_pitch_joint", "gimbal_yaw_link", "gimbal_pitch_link", "revolute",
           (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), axis=(0.0, -1.0, 0.0),
           limits=(float(config.get("pitch_min", -math.radians(25))),
                   float(config.get("pitch_max", math.radians(60)))), relative_to="gimbal_yaw_link")

    # Gazebo's TrackController assumes the belt advances along the transformed
    # +X axis. The simple rectangular track links therefore stay unrotated.
    tracked = _child(robot, "plugin", attrs={
        "name": "gz::sim::systems::TrackedVehicle", "filename": "gz-sim-tracked-vehicle-system"})
    for side in ("left", "right"):
        item = _child(tracked, f"{side}_track")
        _child(item, "link", f"{side}_track")
        _child(item, "velocity_topic", TOPICS[f"track_{side}_velocity"])
        _child(item, "center_of_rotation_topic", TOPICS[f"track_{side}_center_of_rotation"])
    _child(tracked, "tracks_separation", _fmt(2 * track_offset))
    _child(tracked, "track_height", _fmt(track_size[2]))
    _child(tracked, "steering_efficiency", "0.5")
    _child(tracked, "topic", TOPICS["cmd_vel"])
    _child(tracked, "odom_publish_frequency", "20")
    _child(tracked, "odom_topic", TOPICS["odom"])
    _child(tracked, "tf_topic", TOPICS["tf"])
    _child(tracked, "frame_id", "field_odom")
    _child(tracked, "child_frame_id", f"{model_name}/base_link")

    for side in ("left", "right"):
        track_controller = _child(robot, "plugin", attrs={
            "name": "gz::sim::systems::TrackController",
            "filename": "gz-sim-track-controller-system"})
        _child(track_controller, "link", f"{side}_track")
        _child(track_controller, "track_orientation", "0 0 0")
        _child(track_controller, "velocity_topic", TOPICS[f"track_{side}_velocity"])
        _child(track_controller, "center_of_rotation_topic", TOPICS[f"track_{side}_center_of_rotation"])
        _child(track_controller, "min_velocity", _fmt(-float(config.get("max_speed", 4.1667))))
        _child(track_controller, "max_velocity", _fmt(float(config.get("max_speed", 4.1667))))

    for joint_name, topic, initial, bounds in (
        ("gimbal_yaw_joint", TOPICS["gimbal_yaw_cmd"], yaw_angle, (-math.pi, math.pi)),
        ("gimbal_pitch_joint", TOPICS["gimbal_pitch_cmd"], pitch_angle,
         (float(config.get("pitch_min", -math.radians(25))), float(config.get("pitch_max", math.radians(60))))),
    ):
        controller = _child(robot, "plugin", attrs={
            "name": "gz::sim::systems::JointPositionController",
            "filename": "gz-sim-joint-position-controller-system"})
        _child(controller, "joint_name", joint_name)
        _child(controller, "topic", topic)
        _child(controller, "initial_position", _fmt(initial))
        # Camera gimbal dynamics are not identified. Use the stock controller's
        # bounded velocity mode to avoid high-gain torque oscillation on the
        # deliberately low-inertia proxy links.
        _child(controller, "use_velocity_commands", "true")
        _child(controller, "cmd_max", "0.5")

    joint_state = _child(robot, "plugin", attrs={
        "name": "gz::sim::systems::JointStatePublisher", "filename": "gz-sim-joint-state-publisher-system"})
    _child(joint_state, "topic", TOPICS["joint_states"])

    lidar_rate = 1.0 / float(config.get("lidar_period", 0.1))
    rays = int(config.get("lidar_rays", 400))
    lidar = _add_sensor(pitch_link, name="field_lidar", sensor_type="gpu_lidar",
                        topic=TOPICS["lidar"], rate=lidar_rate, pose=(0.0, 0.0, 0.0))
    lidar_node = _child(lidar, "lidar")
    scan = _child(lidar_node, "scan")
    horizontal = _child(scan, "horizontal")
    _child(horizontal, "samples", str(rays))
    _child(horizontal, "resolution", "1")
    _child(horizontal, "min_angle", _fmt(-math.pi))
    _child(horizontal, "max_angle", _fmt(math.pi - 2 * math.pi / rays))
    scan_range = _child(lidar_node, "range")
    _child(scan_range, "min", _fmt(float(config.get("lidar_min_range", 0.12))))
    _child(scan_range, "max", _fmt(float(config.get("lidar_range", 8.0))))
    _child(scan_range, "resolution", "0.01")

    depth_width = int(config.get("depth_width", 88))
    depth_height = int(config.get("depth_height", 160))
    if state and state.get("depth", {}).get("intrinsics"):
        intrinsics = np.asarray(state["depth"]["intrinsics"], dtype=np.float64)
    else:
        native_width, native_height = 352, 640
        native_intrinsics = np.array(
            [[393.29918354972216, 0.0, 177.54501597086588],
             [0.0, 393.29918354972216, 323.73144276936847],
             [0.0, 0.0, 1.0]], dtype=np.float64)
        sx, sy = depth_width / native_width, depth_height / native_height
        intrinsics = native_intrinsics.copy()
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] = (intrinsics[0, 2] + 0.5) * sx - 0.5
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] = (intrinsics[1, 2] + 0.5) * sy - 0.5
    if intrinsics.shape != (3, 3):
        raise ValueError("depth intrinsics must be a 3x3 matrix")
    if not np.all(np.isfinite(intrinsics)) or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("depth intrinsics must be finite with positive focal lengths")
    if state and state.get("depth", {}).get("resolution"):
        state_resolution = tuple(int(v) for v in state["depth"]["resolution"])
        if state_resolution != (depth_width, depth_height):
            raise ValueError("state depth resolution differs from scene camera configuration")
    depth_period = float(config.get("depth_period", 0.4))
    depth = _add_sensor(pitch_link, name="field_depth_camera", sensor_type="depth_camera",
                        topic=TOPICS["depth"], rate=1.0 / depth_period, pose=(0.0, 0.0, 0.0))
    camera = _child(depth, "camera")
    _child(camera, "horizontal_fov", _fmt(2 * math.atan(depth_width / (2 * intrinsics[0, 0]))))
    image = _child(camera, "image")
    _child(image, "width", str(depth_width))
    _child(image, "height", str(depth_height))
    _child(image, "format", "R_FLOAT32")
    clip = _child(camera, "clip")
    _child(clip, "near", _fmt(float(config.get("depth_min_range", 0.3))))
    _child(clip, "far", _fmt(float(config.get("depth_range", 5.0))))
    lens = _child(camera, "lens")
    _child(lens, "type", "gnomonical")
    lens_intrinsics = _child(lens, "intrinsics")
    for name, val in (("fx", intrinsics[0, 0]), ("fy", intrinsics[1, 1]),
                      ("cx", intrinsics[0, 2]), ("cy", intrinsics[1, 2]),
                      ("s", intrinsics[0, 1])):
        _child(lens_intrinsics, name, _fmt(val))
    depth_camera = _child(camera, "depth_camera")
    _child(depth_camera, "output", "depths")
    depth_clip = _child(depth_camera, "clip")
    _child(depth_clip, "near", _fmt(float(config.get("depth_min_range", 0.3))))
    _child(depth_clip, "far", _fmt(float(config.get("depth_range", 5.0))))
    _child(camera, "camera_info_topic", TOPICS["depth_camera_info"])

    for name, sensor_type, topic, rate, sensor_tag in (
        ("field_imu", "imu", TOPICS["imu"], 100.0, "imu"),
        ("field_magnetometer", "magnetometer", TOPICS["magnetometer"], 20.0, "magnetometer"),
    ):
        sensor = _add_sensor(base, name=name, sensor_type=sensor_type, topic=topic, rate=rate,
                             pose=(0.0, 0.0, 0.30))
        _child(sensor, sensor_tag)

    if with_uwb:
        tag = _child(robot, "link", attrs={"name": "uwb_tag_1"})
        _pose(tag, (0.0, 0.0, tag_z))
        _inertial(tag, tag_mass, 0.0, (0.08, 0.08, 0.08))
        _visual(tag, "uwb_tag_visual", color=(1.0, 0.2, 0.1, 1.0), sphere=0.04)
        _joint(robot, "uwb_tag_joint", "base_link", "uwb_tag_1", "fixed",
               (0.0, 0.0, tag_z, 0.0, 0.0, 0.0), relative_to="base_link")

    return {
        "model_name": model_name, "mass_kg": total, "center_of_mass_height_m": com_height,
        "center_of_mass_status": "assumed_configurable_proxy",
        "mass_distribution_kg": {"base_link": base_mass, "left_track": track_mass,
                                  "right_track": track_mass, "gimbal_yaw_link": yaw_mass,
                                  "gimbal_pitch_link": pitch_mass,
                                  **({"uwb_tag_1": tag_mass} if with_uwb else {})},
        "dimensions_m": {"length": float(config.get("robot_length", 0.8)),
                          "width": float(config.get("robot_width", 0.3)),
                          "track_length_axis": "+X", "track_orientation_rpy": [0.0, 0.0, 0.0]},
        "gimbal": {"yaw_joint": "gimbal_yaw_joint", "pitch_joint": "gimbal_pitch_joint",
                   "pitch_axis_in_yaw_link": [0.0, -1.0, 0.0],
                   "pitch_positive_direction": "raises camera +X forward vector",
                   "pitch_limits_rad": [float(config.get("pitch_min", -math.radians(25))),
                                        float(config.get("pitch_max", math.radians(60)))],
                   "initial_position_rad": {"yaw": yaw_angle, "pitch": pitch_angle},
                   "initialization": "controller target only; link poses use zero joint angles",
                   "control_mode": "bounded_velocity_commands",
                   "max_commanded_joint_speed_rad_s": 0.5,
                   "actuator_dynamics": "kinematic position tracking; physical actuator dynamics are not identified",
                   "lidar_and_depth_camera_share_pitch_link": True},
        "sensors": {
            "lidar": {"topic": TOPICS["lidar"], "type": "gpu_lidar", "horizontal_samples": rays,
                      "min_angle_rad": -math.pi, "max_angle_rad": math.pi - 2 * math.pi / rays,
                      "endpoint_policy": f"no duplicate +/-pi endpoint; uniform 2pi/{rays} azimuth steps",
                      "range_m": [float(config.get("lidar_min_range", 0.12)), float(config.get("lidar_range", 8.0))],
                      "update_rate_hz": lidar_rate, "mount": "gimbal_pitch_link"},
            "depth": {"topic": TOPICS["depth"], "camera_info_topic": TOPICS["depth_camera_info"],
                      "type": "depth_camera", "format": "R_FLOAT32", "convention": "camera_z",
                      "resolution_wh": [depth_width, depth_height], "intrinsics_K": intrinsics.tolist(),
                      "range_m": [float(config.get("depth_min_range", 0.3)), float(config.get("depth_range", 5.0))],
                      "update_rate_hz": 1.0 / depth_period, "mount": "gimbal_pitch_link",
                      "rgb_enabled": False},
            "imu": {"topic": TOPICS["imu"], "mount": "base_link"},
            "magnetometer": {"topic": TOPICS["magnetometer"], "mount": "base_link"},
        },
    }


def _write_world(scene: dict, state: dict | None, terrain_meta: dict,
                 rocks_meta: dict, vehicle_meta: dict, output: Path, unlimited: bool,
                 with_uwb: bool, uwb_anchor_clearance_m: float) -> Path:
    size = float(scene["size"])
    root = ET.Element("sdf", {"version": "1.9"})
    world = _child(root, "world", attrs={"name": "field_training"})
    physics = _child(world, "physics", attrs={"name": "field_4ms", "type": "dart"})
    _child(physics, "max_step_size", "0.004")
    _child(physics, "real_time_factor", "0" if unlimited else "1")
    dart = _child(physics, "dart")
    _child(dart, "collision_detector", "bullet")
    _child(world, "gravity", "0 0 -9.81")
    spherical = _child(world, "spherical_coordinates")
    _child(spherical, "surface_model", "EARTH_WGS84")
    _child(spherical, "world_frame_orientation", "ENU")
    _child(spherical, "latitude_deg", "30")
    _child(spherical, "longitude_deg", "120")
    _child(spherical, "elevation", "0")
    _child(spherical, "heading_deg", "0")
    for name, filename in (
        ("gz::sim::systems::Physics", "gz-sim-physics-system"),
        ("gz::sim::systems::UserCommands", "gz-sim-user-commands-system"),
        ("gz::sim::systems::SceneBroadcaster", "gz-sim-scene-broadcaster-system"),
        ("gz::sim::systems::Sensors", "gz-sim-sensors-system"),
        ("gz::sim::systems::Imu", "gz-sim-imu-system"),
        ("gz::sim::systems::Magnetometer", "gz-sim-magnetometer-system"),
    ):
        plugin = _child(world, "plugin", attrs={"name": name, "filename": filename})
        if "Sensors" in name:
            _child(plugin, "render_engine", "ogre2")
    scene_node = _child(world, "scene")
    _child(scene_node, "ambient", "0.55 0.58 0.52 1")
    _child(scene_node, "background", "0.12 0.15 0.18 1")
    _child(scene_node, "shadows", "false")
    light = _child(world, "light", attrs={"name": "sun", "type": "directional"})
    _pose(light, (0.0, 0.0, 12.0))
    _child(light, "cast_shadows", "false")
    _child(light, "diffuse", "0.9 0.9 0.86 1")
    _child(light, "direction", "-0.45 0.2 -1")

    _add_heightmap_model(world, size, terrain_meta)
    rocks_meta.update(_add_rocks(world, scene.get("terrain", {}).get("rocks", []), size))
    if with_uwb:
        uwb_plugin = _child(world, "plugin", attrs={
            "name": "custom::UWBGazeboSystem", "filename": "libUWBGazeboPlugin.so"})
        _child(uwb_plugin, "topic", "/uwb_gz_simulator/distances")
        _child(uwb_plugin, "ground_truth_topic", "/uwb_gz_simulator/distances_ground_truth")
        # The upstream plugin has no heightmap exclusion, so neutralize its
        # thickness-dependent output for this field-only compatibility preset.
        _child(uwb_plugin, "min_noise_stddev_cm", "5")
        _child(uwb_plugin, "max_noise_stddev_cm", "5")
        _child(uwb_plugin, "min_dropout_probability", "0.02")
        _child(uwb_plugin, "max_dropout_probability", "0.02")
        _child(uwb_plugin, "enable_nlos_dropout", "false")
        _child(uwb_plugin, "los_max_thickness_m",
               _fmt(max(1000.0, 2.0 * size + uwb_anchor_clearance_m + 1.0)))
        _child(uwb_plugin, "apply_pair_bias", "false")
        for item in uwb_anchor_layout(scene, uwb_anchor_clearance_m):
            anchor = _child(world, "model", attrs={"name": item["model"]})
            _child(anchor, "static", "true")
            _pose(anchor, item["world_xyz_m"])
            link = _child(anchor, "link", attrs={"name": item["link"]})
            _visual(link, "anchor_visual", color=(1.0, 0.6, 0.05, 1.0), sphere=0.1)
    veg = _child(world, "model", attrs={"name": "field_vegetation"})
    _child(veg, "static", "true")
    veg_link = _child(veg, "link", attrs={"name": "vegetation_visuals"})
    for name, color in (("healthy_leaves", (0x4A / 255, 0x86 / 255, 0x39 / 255, 1.0)),
                        ("dry_leaves", (0xB1 / 255, 0x7B / 255, 0x42 / 255, 1.0)),
                        ("stems", (0x66 / 255, 0x8B / 255, 0x36 / 255, 1.0))):
        _visual(veg_link, name, f"assets/{name}.obj", color=color)
    robot_meta = _add_robot(world, scene, state, float(vehicle_meta["mass_kg"]),
                            float(vehicle_meta["center_of_mass_height_m"]),
                            "x500_field_robot" if with_uwb else "field_robot", with_uwb)
    vehicle_meta.clear()
    vehicle_meta.update(robot_meta)
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    output_path = output / "world.sdf"
    output_path.write_text('<?xml version="1.0"?>\n' + xml + "\n", encoding="utf-8")
    return output_path


def _hash_input(path: Path | None, digest: str | None):
    return None if path is None else {"path": str(path.resolve()), "sha256": digest}


def export_scene(scene_path: Path, state_path: Path | None, output: Path, *,
                 mass: float = 30.0, com_height: float = 0.15, unlimited: bool = False,
                 with_uwb: bool = False, uwb_anchor_clearance_m: float = 1.8) -> dict:
    scene_path = Path(scene_path)
    state_path = Path(state_path) if state_path is not None else None
    output = Path(output)
    if not math.isfinite(float(mass)) or mass <= 3.0:
        raise ValueError("mass must exceed 3 kg to retain positive track and gimbal link masses")
    if not math.isfinite(float(com_height)) or com_height <= 0:
        raise ValueError("com-height must be finite and positive")
    if not math.isfinite(float(uwb_anchor_clearance_m)) or uwb_anchor_clearance_m <= 0:
        raise ValueError("uwb-anchor-clearance must be finite and positive")
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if state_path is not None and not state_path.is_file():
        raise FileNotFoundError(state_path)
    scene, state, scene_sha, state_sha = _load_payloads(scene_path, state_path)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    geometry = scene["leaf_geometry"]
    leaf_dry = np.asarray(geometry["dry"], dtype=bool)
    healthy_ids = np.flatnonzero(~leaf_dry).tolist()
    dry_ids = np.flatnonzero(leaf_dry).tolist()
    center_offset = float(scene["size"]) / 2.0
    leaf_faces_healthy = _write_leaf_obj(assets / "healthy_leaves.obj", geometry, healthy_ids, center_offset)
    leaf_faces_dry = _write_leaf_obj(assets / "dry_leaves.obj", geometry, dry_ids, center_offset)
    stem_faces = _write_stem_obj(assets / "stems.obj", scene["plants"], center_offset)
    rock_faces = _write_rock_obj(assets / "rock_cap.obj")
    terrain_meta = _write_terrain_png(scene, assets / "terrain.png")
    terrain_meta["render_texture"] = _write_heightmap_textures(assets)
    terrain_meta["runtime_heightmap_warning"] = (
        "Gazebo Harmonic probe reported cropping of the final heightmap row and column; "
        "effective sample-axis coverage remains runtime-comparison work."
    )
    rock_list = scene.get("terrain", {}).get("rocks", [])
    leaf_ids_by_plant: dict[int, list[int]] = {i: [] for i in range(len(scene["plants"]))}
    for leaf_id, plant_id in enumerate(geometry["plant"]):
        leaf_ids_by_plant[int(plant_id)].append(leaf_id)
    plant_adapter = []
    for plant_id, plant in enumerate(scene["plants"]):
        plant_adapter.append({
            "plant_id": plant_id,
            "source_xy_m": [float(plant["x"]), float(plant["y"])],
            "world_xy_m": [float(plant["x"]) - float(scene["size"]) / 2,
                           float(plant["y"]) - float(scene["size"]) / 2],
            "root_z_m": float(plant.get("base_z", plant["z"])),
            "root_damage_radius_m": float(plant.get("root_damage_radius", plant["radius"])),
            "plant_height_m": float(plant["height"]),
            "stem_radius_m": float(plant["stem_radius"]),
            "dry_plant_state": bool(plant["dry"]),
            "leaf_ids": leaf_ids_by_plant[plant_id],
        })
    leaf_ordinal_by_plant = [0] * len(scene["plants"])
    leaf_adapter = []
    for leaf_id, plant_id_raw in enumerate(geometry["plant"]):
        plant_id = int(plant_id_raw)
        leaf_adapter.append({"leaf_id": leaf_id, "plant_id": plant_id,
                             "leaf_index_within_plant": leaf_ordinal_by_plant[plant_id],
                             "dry_leaf_state": bool(geometry["dry"][leaf_id])})
        leaf_ordinal_by_plant[plant_id] += 1
    terrain_meta.update({"rock_count": len(rock_list),
                         "rocks": {"mesh": "assets/rock_cap.obj", "unit_cap_radii": [1.0, 1.0],
                                   "unit_cap_height": 1.0, "collision_policy": "all rocks have static collision; blocking flag preserved in manifest"},
                         "scene_seed": scene.get("terrain", {}).get("seed")})
    vehicle_meta = {"mass_kg": float(mass), "center_of_mass_height_m": float(com_height)}
    rocks_meta = {"mesh_face_count": rock_faces}
    world_path = _write_world(scene, state, terrain_meta, rocks_meta, vehicle_meta, output,
                              unlimited, with_uwb, float(uwb_anchor_clearance_m))
    topics_path = output / "world.topics.json"
    topics_doc = {
        "world": "field_training", "model": vehicle_meta["model_name"], "topics": dict(TOPICS),
        "message_types": {
            "cmd_vel": "gz.msgs.Twist", "odom": "gz.msgs.Odometry", "tf": "gz.msgs.Pose_V",
            "joint_states": "gz.msgs.Model", "lidar": "gz.msgs.LaserScan",
            "depth": "gz.msgs.Image", "depth_camera_info": "gz.msgs.CameraInfo",
            "imu": "gz.msgs.IMU", "magnetometer": "gz.msgs.Magnetometer",
            "gimbal_yaw_cmd": "gz.msgs.Double", "gimbal_pitch_cmd": "gz.msgs.Double",
            "track_left_velocity": "gz.msgs.Double", "track_right_velocity": "gz.msgs.Double",
            "track_left_center_of_rotation": "gz.msgs.Vector3d",
            "track_right_center_of_rotation": "gz.msgs.Vector3d",
        },
    }
    uwb_anchors = uwb_anchor_layout(scene, uwb_anchor_clearance_m) if with_uwb else []
    if with_uwb:
        topics_doc["topics"].update({
            "uwb_ranges_base": "/uwb_gz_simulator/distances",
            "uwb_ground_truth_base": "/uwb_gz_simulator/distances_ground_truth",
        })
        topics_doc["message_types"].update({
            "uwb_ranges": "gz.msgs.Double",
            "uwb_ground_truth": "gz.msgs.Double",
        })
        topics_doc["uwb_range_pairs"] = [
            {"anchor_id": item["id"], "tag_id": 1, "pair_key": f"a{item['id']}t1",
             "range_topic": f"/uwb_gz_simulator/distances/a{item['id']}t1",
             "ground_truth_topic": f"/uwb_gz_simulator/distances_ground_truth/a{item['id']}t1",
             "message_type": "gz.msgs.Double", "unit": "cm"}
            for item in uwb_anchors
        ]
    topics_path.write_text(json.dumps(topics_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = scene["config"]
    manifest = {
        "schema": "gazebo-field-export-v1",
        "exporter": "simulation.field_world",
        "scene_id": scene.get("scene_id"),
        "inputs": {"scene_json": _hash_input(scene_path, scene_sha),
                   "state_json": _hash_input(state_path, state_sha)},
        "world": {"name": "field_training", "sdf": "world.sdf", "topics": "world.topics.json",
                  "target": "Gazebo Harmonic / gz-sim 8", "runtime_loaded_or_validated": False},
        "coordinate_frame": {
            "source": "field_xyz_z_up_m; x,y in [0,size]",
            "world": "Gazebo X/Y centered at field center; world_xy = source_xy - size/2; world center=(0,0)",
            "height": "source Z is retained in world Z",
        },
        "physics": {"engine": "dart", "collision_detector": "bullet", "max_step_size_s": 0.004,
                    "real_time_factor": 0 if unlimited else 1, "shadows": False},
        "magnetometer_world_reference": {
            "surface_model": "EARTH_WGS84", "world_frame_orientation": "ENU",
            "latitude_deg": 30.0, "longitude_deg": 120.0, "elevation_m": 0.0,
            "status": "proxy coordinates for magnetic-field modeling only; not geographic ground truth",
        },
        "terrain": terrain_meta,
        "geometry_counts": {
            "plants": len(scene["plants"]), "leaves": len(geometry["centers"]),
            "healthy_leaves": len(healthy_ids), "dry_leaves": len(dry_ids),
            "stems": len(scene["plants"]), "rocks": len(rock_list),
            "blocking_rocks": sum(bool(r.get("blocking", False)) for r in rock_list),
            "nonblocking_rocks": sum(not bool(r.get("blocking", False)) for r in rock_list),
            "merged_plant_visual_meshes": 3, "plant_collision_meshes": 0,
            "leaf_boundary_samples_per_side": LEAF_BOUNDARY_STEPS + 1,
            "leaf_mesh_faces": {"healthy": leaf_faces_healthy, "dry": leaf_faces_dry},
            "stem_mesh_faces": stem_faces, "unit_rock_mesh_faces": rock_faces,
        },
        "plant_damage_adapter": {
            "plant_key": "plant_id (source scene plant array index)",
            "leaf_key": "leaf_id (global source leaf array index)",
            "association": "leaf_adapter[].plant_id and plant_adapter[].leaf_ids",
            "leaf_state_key": "dry_leaf_state",
            "plant_root_damage_key": "root_damage_radius_m",
            "plant_adapter": plant_adapter,
            "leaf_adapter": leaf_adapter,
        },
        "vehicle": vehicle_meta,
        "uwb": ({"enabled": True, "plugin_filename": "libUWBGazeboPlugin.so",
                 "plugin_name": "custom::UWBGazeboSystem", "robot_name_prefix": "x500_",
                 "anchors": uwb_anchors,
                 "anchor_layout": "3x3 grid at source field fractions x,y=[0.1,0.5,0.9]",
                 "anchor_clearance_m": float(uwb_anchor_clearance_m),
                 "anchor_layout_calibrated": False,
                 "tag_link": "uwb_tag_1", "tag_mass_in_total_vehicle_mass": True,
                 "range_topic_base": "/uwb_gz_simulator/distances",
                 "ground_truth_topic_base": "/uwb_gz_simulator/distances_ground_truth",
                 "pair_topic_template": "/uwb_gz_simulator/distances/a{anchor_id}t1",
                 "pair_message_type": "gz.msgs.Double", "range_unit": "cm",
                 "nlos_enabled": False,
                 "nlos_compatibility_preset": {
                     "enable_nlos_dropout": False,
                     "min_noise_stddev_cm": 5.0, "max_noise_stddev_cm": 5.0,
                     "min_dropout_probability": 0.02, "max_dropout_probability": 0.02,
                     "los_max_thickness_m": max(1000.0, 2.0 * float(scene["size"]) +
                                                   float(uwb_anchor_clearance_m) + 1.0),
                     "effect": "constant 5 cm Gaussian noise and 2% random dropout neutralize thickness-dependent AABB effects; the plugin's >75m distance blackout remains"
                 },
                 "runtime_verified": False,
                 "note": "Compatibility preset only: upstream plugin has no terrain exclusion, so geometric NLOS and body/vegetation/rock occlusion are not modeled. Anchor deployment is a proxy and is not calibrated to the real site."}
                if with_uwb else
                {"enabled": False, "reason": "optional UWB integration disabled by default"}),
        "state_initialization": None if state is None else {
            "scene_id": state.get("scene_id"), "robot": state.get("robot"), "gimbal": state.get("gimbal"),
            "spawn_world_xy_m": [float(state["robot"]["x"]) - float(scene["size"]) / 2,
                                 float(state["robot"]["y"]) - float(scene["size"]) / 2],
            "state_frame_replayed": False,
        },
        "topics": topics_doc,
        "proxy_limits": [
            "Plant leaves and stems are merged static visual meshes with no colliders so the vehicle can pass through them.",
            "Leaf mesh samples the source tapered_blade_v1 boundary at 24 intervals per side; it does not model bending, breakage, or per-leaf dynamics.",
            "The visual meshes preserve plant and leaf IDs in manifest.json for a future damage/detection adapter.",
            "Rocks are separate static models; large-rock blocking state is preserved in metadata and every rock has collision.",
            "The depth camera emits depth only. No detector plugin, trained detector, or closed-loop field policy is included.",
            "The heightmap samples and quantizes the four source waves; Gazebo image row direction, interpolation, and contact fidelity need runtime comparison.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"world": str(world_path), "manifest": str(manifest_path), "topics": str(topics_path),
            "assets": str(assets), "scene_id": scene.get("scene_id"), "geometry_counts": manifest["geometry_counts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an existing field_sim scene payload to a Gazebo Harmonic world.")
    parser.add_argument("--scene-json", type=Path,
                        default=ROOT / "results" / "gazebo_field_source" / "current_scene.json")
    parser.add_argument("--state-json", type=Path, default=None,
                        help="Optional matching field_sim state payload; scene_id must match.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gazebo_field")
    parser.add_argument("--mass", type=float, default=30.0, help="Total robot mass in kg (default: 30).")
    parser.add_argument("--com-height", type=float, default=0.15,
                        help="Assumed center of mass height above support ground, in metres.")
    parser.add_argument("--unlimited", action="store_true",
                        help="Set real_time_factor=0 for an unpaced benchmark world.")
    parser.add_argument("--with-uwb", action="store_true",
                        help="Include the local UWBGazeboPlugin integration (requires its plugin on the runtime system).")
    parser.add_argument("--uwb-anchor-clearance", type=float, default=1.8,
                        help="With --with-uwb, place anchors this far above the source terrain (default: 1.8 m proxy).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_scene(args.scene_json, args.state_json, args.output,
                              mass=args.mass, com_height=args.com_height, unlimited=args.unlimited,
                              with_uwb=args.with_uwb, uwb_anchor_clearance_m=args.uwb_anchor_clearance)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"field_world: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
