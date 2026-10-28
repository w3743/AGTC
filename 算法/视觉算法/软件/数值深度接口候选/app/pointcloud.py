"""Export one LightStereo metric-depth frame as a portable colored point cloud."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import cv2
import numpy as np


APP = Path(__file__).resolve().parent
ROOT = APP.parent
MAX_VIEW_POINTS = 25000
DEPTH_RANGE_M = (0.3, 5.0)
VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _projection_geometry(calibration_path=None) -> dict[str, np.ndarray]:
    """Derive upright K/Q from the same calibration consumed by depth."""
    from calibration import load_model, portrait_geometry
    return portrait_geometry(load_model(calibration_path or APP / 'calibration.npz'))


def export(depth: np.ndarray, output_dir: str | Path, calibration_id: str,
           source: dict | list | tuple | str, *, calibration_path: str | Path | None = None,
           geometry: dict | None = None) -> dict:
    """Write full-resolution PLY and bounded HTML sample for one depth image."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (640, 352):
        raise ValueError(f"Expected portrait depth shape (640, 352), got {depth.shape}")
    if geometry is None:
        if calibration_path is None:
            raise ValueError("calibration_path is required to load projection geometry")
        if _sha256(Path(calibration_path)) != _sha256(APP / "calibration.npz"):
            raise ValueError("Custom calibration requires matching cropped projection geometry")
        geometry = _projection_geometry()
    k = np.asarray(geometry["K"], dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Projection K must be a finite 3x3 matrix")

    valid = np.isfinite(depth) & (depth >= DEPTH_RANGE_M[0]) & (depth <= DEPTH_RANGE_M[1])
    ys, xs = np.nonzero(valid)
    if not len(xs):
        raise ValueError("No finite depth points in the 0.3–5 m range")
    z = depth[ys, xs].astype(np.float64)
    rays = np.linalg.solve(k, np.vstack((xs, ys, np.ones(len(xs)))))
    xyz = (rays * z).T

    # Check pixel-to-ray roundtrip on a bounded, evenly-spaced subset.
    check_idx = np.linspace(0, len(xyz) - 1, min(len(xyz), 2048), dtype=np.int64)
    projected = xyz[check_idx] @ k.T
    uv = projected[:, :2] / projected[:, 2:3]
    expected = np.column_stack((xs[check_idx], ys[check_idx]))
    max_roundtrip_px = float(np.max(np.abs(uv - expected)))
    assert max_roundtrip_px < 1e-3, f"Pixel reprojection error {max_roundtrip_px} px"

    near, far = np.percentile(z, [5, 95])
    if far - near < 0.01:
        near, far = max(0.0, near - 0.01), far + 0.01
    shade = np.clip((far - z) / (far - near) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(shade[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
    vertices = np.empty(len(xyz), dtype=VERTEX_DTYPE)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, i]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ply_path = out / "pointcloud.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment units metres; camera X right Y down Z forward\n"
        "comment colors represent depth, not camera RGB\n"
        "comment source one stereo pair; no fusion\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    ply_path.write_bytes(header + vertices.tobytes())
    ply_bytes = ply_path.read_bytes()
    assert ply_bytes[:len(header)] == header
    assert len(ply_bytes) - len(header) == len(vertices) * VERTEX_DTYPE.itemsize
    restored = np.frombuffer(ply_bytes, dtype=VERTEX_DTYPE, offset=len(header))
    assert len(restored) == len(vertices) and np.array_equal(restored, vertices)

    if isinstance(source, dict):
        source_pair = {str(key): str(value) for key, value in source.items()}
    elif isinstance(source, (tuple, list)):
        source_pair = {"left": str(source[0]), "right": str(source[1])}
    else:
        source_pair = {"pair": str(source)}
    bounds = {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()}
    meta = {
        "source": source_pair,
        "count": int(len(xyz)),
        "calibration_id": str(calibration_id),
        "bounds": bounds,
        "color_range_m": [float(near), float(far)],
        "coordinate_system": "metres; left rectified portrait camera; X right, Y down, Z forward",
        "filter": "finite metric depth 0.3-5m; no hole filling",
        "scope": "one stereo pair; no multi-frame fusion",
        "projection_validation": {"max_pixel_roundtrip_px": max_roundtrip_px},
        "ply_sha256": _sha256(ply_path),
    }
    if calibration_path is not None:
        cal_path = Path(calibration_path)
        meta["calibration_sha256"] = _sha256(cal_path)
    (out / "pointcloud.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The binary PLY keeps every point; HTML embeds at most 25,000 uniform samples.
    step = max(1, int(np.ceil(len(xyz) / MAX_VIEW_POINTS)))
    payload = dict(meta, source="LightStereo · 已保存的双目照片对",
                   points=np.column_stack((xyz[::step].round(5), rgb[::step])).tolist())
    template_path = APP / "pointcloud_viewer.html"
    template = template_path.read_text(encoding="utf-8")
    assert template.count("__POINTCLOUD_DATA__") == 1
    html = template.replace("__POINTCLOUD_DATA__", json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"))
    (out / "pointcloud.html").write_text(html, encoding="utf-8")
    return meta


class PointCloudStream:
    """Project each fresh metric-depth frame into a compact camera-frame packet."""

    HEADER = struct.Struct("<4sIHH")
    RECORD = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<u2"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])

    def __init__(self, calibration_path: str | Path, *, max_points: int = 12000,
                 voxel_m: float = 0.01):
        calibration_path = Path(calibration_path)
        self.calibration_id = _sha256(calibration_path)[:12]
        self.geometry = _projection_geometry(calibration_path)
        self.K = np.asarray(self.geometry["K"], dtype=np.float64)
        self.Q = np.asarray(self.geometry["Q"], dtype=np.float64)
        if self.Q.shape != (4, 4) or not np.isfinite(self.Q).all():
            raise ValueError("Projection Q must be a finite 4x4 matrix")
        self.max_points = min(15000, max(5000, int(max_points)))
        self.voxel_m = max(0.005, float(voxel_m))

    def scaled_intrinsics(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        if height <= 0 or width <= 0:
            raise ValueError(f"Depth shape does not match calibrated portrait geometry: {shape}")
        scale = height / 640.0
        crop = max(1, int(round(4 * scale)))
        if abs(width - (360 * scale - 2 * crop)) > 2:
            raise ValueError(f"Depth shape does not match calibrated portrait geometry: {shape}")
        # Match the exact resize -> CCW rotation -> rounded horizontal crop
        # used to derive scaled_q. The reference K already includes crop=4 at
        # 352x640; the shift maps that cropped reference grid onto this one.
        horizontal_shift = scale * (4 + 0.5) - 0.5 - crop
        transform = np.array([[scale, 0, horizontal_shift],
                              [0, scale, (scale - 1) / 2],
                              [0, 0, 1]], dtype=np.float64)
        return transform @ self.K

    def scaled_q(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        scale = height / 640.0
        crop = max(1, int(round(4 * scale)))
        # Convert the scaled, cropped pixel/disparity coordinates back to the
        # 352x640 reference grid consumed by the packaged Q matrix.
        horizontal_shift = scale * (4 + 0.5) - 0.5 - crop
        low_to_reference = np.array([
            [1 / scale, 0, 0, -horizontal_shift / scale],
            [0, 1 / scale, 0, (1 - scale) / (2 * scale)],
            [0, 0, 1 / scale, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        q = self.Q @ low_to_reference
        if not np.isfinite(q).all() or abs(width - (360 * scale - 2 * crop)) > 2:
            raise ValueError(f"Depth shape does not match packaged Q geometry: {shape}")
        return q

    def _depth_conversion(self, shape: tuple[int, int],
                          depth_factor: float | None,
                          disparity_offset: float | None) -> tuple[float, float]:
        scale = shape[0] / 640.0
        q_factor = float(self.Q[2, 3] / self.Q[3, 2])
        q_offset = float(-self.Q[3, 3] / self.Q[3, 2])
        factor = q_factor * scale if depth_factor is None else float(depth_factor)
        offset = q_offset * scale if disparity_offset is None else float(disparity_offset)
        return factor, offset

    def pixel_to_xyz(self, depth: np.ndarray, x: int, y: int,
                     depth_factor: float | None = None,
                     disparity_offset: float | None = None) -> tuple[float, float, float] | None:
        depth = np.asarray(depth, dtype=np.float32)
        if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
            return None
        radius = 4
        x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        good = patch[np.isfinite(patch) & (patch >= DEPTH_RANGE_M[0]) & (patch <= DEPTH_RANGE_M[1])]
        if good.size == 0:
            return None
        z = float(np.median(good))
        q = self.scaled_q(depth.shape)
        factor, offset = self._depth_conversion(depth.shape, depth_factor, disparity_offset)
        disparity = factor / z + offset
        homogeneous = q @ np.array([float(x), float(y), disparity, 1.0])
        if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-9:
            return None
        xyz = homogeneous[:3] / homogeneous[3]
        return tuple(float(value) for value in xyz)

    def encode(self, depth: np.ndarray, gray: np.ndarray, sequence: int,
               depth_factor: float | None = None,
               disparity_offset: float | None = None) -> tuple[bytes, int]:
        depth = np.asarray(depth, dtype=np.float32)
        gray = np.asarray(gray)
        if depth.ndim != 2 or gray.shape != depth.shape or gray.dtype != np.uint8:
            raise ValueError("Point cloud input must be a matching depth/GRAY8 pair")
        valid = np.isfinite(depth) & (depth >= DEPTH_RANGE_M[0]) & (depth <= DEPTH_RANGE_M[1])
        support = cv2.boxFilter(valid.astype(np.uint8), cv2.CV_16U, (3, 3),
                                normalize=False, borderType=cv2.BORDER_CONSTANT)
        valid &= support >= 3
        # Keep up to 15k samples at the smaller 240x131 mode; use 2x2 sampling
        # for 320x176 and larger depth maps before voxel filtering.
        step = 1 if depth.size <= 45000 else 2
        grid = np.zeros(depth.shape, dtype=bool)
        grid[::step, ::step] = True
        ys, xs = np.nonzero(valid & grid)
        if len(xs):
            if len(xs) > self.max_points:
                selected = np.linspace(0, len(xs) - 1, self.max_points, dtype=np.int64)
                xs, ys = xs[selected], ys[selected]
            z = depth[ys, xs].astype(np.float64)
            q = self.scaled_q(depth.shape)
            factor, offset = self._depth_conversion(depth.shape, depth_factor,
                                                     disparity_offset)
            disparity = factor / z + offset
            # This is a 4x4 projection over only a few thousand samples. Avoid
            # dispatching it through threaded BLAS: on RK3588 that startup can
            # take longer than the arithmetic and stalls the depth worker.
            x_values = xs.astype(np.float64, copy=False)
            y_values = ys.astype(np.float64, copy=False)
            hom_w = (q[3, 0] * x_values + q[3, 1] * y_values
                     + q[3, 2] * disparity + q[3, 3])
            xyz = np.column_stack((
                (q[0, 0] * x_values + q[0, 1] * y_values
                 + q[0, 2] * disparity + q[0, 3]) / hom_w,
                (q[1, 0] * x_values + q[1, 1] * y_values
                 + q[1, 2] * disparity + q[1, 3]) / hom_w,
                (q[2, 0] * x_values + q[2, 1] * y_values
                 + q[2, 2] * disparity + q[2, 3]) / hom_w,
            ))
            voxels = np.floor(xyz / self.voxel_m).astype(np.int32)
            voxel_min = voxels.min(axis=0)
            relative = voxels - voxel_min
            spans = relative.max(axis=0).astype(np.int64) + 1
            voxel_keys = ((relative[:, 0].astype(np.int64) * spans[1]
                           + relative[:, 1]) * spans[2] + relative[:, 2])
            _, keep = np.unique(voxel_keys, return_index=True)
            keep.sort()
            if len(keep) > self.max_points:
                keep = keep[np.linspace(0, len(keep) - 1, self.max_points, dtype=np.int64)]
            xyz, xs, ys = xyz[keep], xs[keep], ys[keep]
            camera = gray[ys, xs]
            records = np.empty(len(xyz), dtype=self.RECORD)
            records["x"] = np.clip(np.rint(xyz[:, 0] * 1000), -32768, 32767).astype(np.int16)
            records["y"] = np.clip(np.rint(xyz[:, 1] * 1000), -32768, 32767).astype(np.int16)
            records["z"] = np.clip(np.rint(xyz[:, 2] * 1000), 0, 65535).astype(np.uint16)
            records["red"] = camera
            records["green"] = camera
            records["blue"] = camera
            payload = records.tobytes()
        else:
            payload = b""
        count = len(payload) // self.RECORD.itemsize
        return self.HEADER.pack(b"PC01", int(sequence) & 0xFFFFFFFF,
                                count, self.RECORD.itemsize) + payload, count


def stream_self_test(calibration_path: Path) -> dict:
    projector = PointCloudStream(calibration_path)
    depth = np.ones((320, 176), dtype=np.float32)
    gray = np.tile(np.arange(176, dtype=np.uint8), (320, 1))
    packet, count = projector.encode(depth, gray, 1234)
    magic, sequence, encoded_count, stride = projector.HEADER.unpack_from(packet)
    if (magic != b"PC01" or sequence != 1234 or encoded_count != count
            or stride != projector.RECORD.itemsize or not 0 < count <= 15000
            or len(packet) != projector.HEADER.size + count * stride):
        raise RuntimeError("Realtime point-cloud packet contract failed")
    xyz = projector.pixel_to_xyz(depth, 88, 160)
    if xyz is None or abs(xyz[2] - 1.0) > .001:
        raise RuntimeError(f"Camera-frame projection check failed: {xyz}")
    return dict(ok=True, packet_bytes=len(packet), points=count, depth_sequence=sequence,
                stride_bytes=stride, camera_xyz_m=list(xyz),
                coordinate_system="metres; camera X right, Y down, Z forward",
                calibration_id=projector.calibration_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one LightStereo depth frame as PLY and HTML")
    parser.add_argument("--left", type=Path, default=ROOT / "validation" / "left.png")
    parser.add_argument("--right", type=Path, default=ROOT / "validation" / "right.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--calibration", type=Path, default=APP / "calibration.npz")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the live binary stream and camera-frame projection")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(stream_self_test(args.calibration), ensure_ascii=False,
                         separators=(",", ":")))
        return

    from lightstereo_depth import MetricDepth

    left = cv2.imread(str(args.left), cv2.IMREAD_UNCHANGED)
    right = cv2.imread(str(args.right), cv2.IMREAD_UNCHANGED)
    if left is None or right is None:
        raise FileNotFoundError(f"Could not read input pair: {args.left}, {args.right}")
    if (left.ndim != 2 or right.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8
            or left.shape != (2160, 3840) or right.shape != left.shape):
        raise ValueError("Expected matching raw 3840x2160 uint8 grayscale left/right images")
    engine = MetricDepth(args.calibration)
    try:
        color, depth, details = engine.compute(left, right)
    finally:
        engine.close()
    del color  # Export is based on metric depth, with depth pseudo-coloring.
    calibration_id = getattr(engine, "identifier", None)
    if calibration_id is None and isinstance(details, dict):
        calibration_id = details.get("calibration_id", details.get("identifier"))
    if not calibration_id:
        calibration_id = _sha256(args.calibration)
    meta = export(depth, args.output_dir, str(calibration_id),
                  {"left": str(args.left), "right": str(args.right)},
                  calibration_path=args.calibration)
    print(json.dumps({"count": meta["count"], "ply": str(args.output_dir / "pointcloud.ply"),
                      "html": str(args.output_dir / "pointcloud.html"),
                      "calibration_id": meta["calibration_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
