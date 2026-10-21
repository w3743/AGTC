"""Export one LightStereo metric-depth frame as a portable colored point cloud."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def _projection_geometry() -> dict[str, np.ndarray]:
    """Return the experiment's upright, 4-pixel-cropped portrait K/Q geometry."""
    packaged = ROOT / "models" / "portrait_geometry.npz"
    with np.load(packaged, allow_pickle=False) as data:
        if not np.array_equal(data["image_size"], [352, 640]):
            raise ValueError("portrait_geometry.npz must describe 352x640 pixels")
        k, q = np.array(data["K"], np.float64), np.array(data["Q"], np.float64)
    if k.shape != (3, 3) or q.shape != (4, 4):
        raise ValueError("Invalid K/Q in portrait_geometry.npz")
    if not np.isfinite(k).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite K/Q in portrait_geometry.npz")
    return {"K": k, "Q": q}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one LightStereo depth frame as PLY and HTML")
    parser.add_argument("--left", type=Path, default=ROOT / "validation" / "left.png")
    parser.add_argument("--right", type=Path, default=ROOT / "validation" / "right.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--calibration", type=Path, default=APP / "calibration.npz")
    args = parser.parse_args()

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
