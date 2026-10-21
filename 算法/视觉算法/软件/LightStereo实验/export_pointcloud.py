"""Export the verified portrait RKNN snapshot as metric, depth-colored points."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate import _depth_from_disparity
from portrait import CroppedMetric, projection_geometry, verify_geometry

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    calibration = HERE.parent / 'stereo/calibration/current/parameters.npz'
    report = json.loads((RESULTS / 'portrait-report.json').read_text(encoding='utf-8'))
    metric = CroppedMetric(calibration)
    if metric.identifier != report['calibration_id']:
        raise ValueError('Calibration changed; rerun portrait inference before exporting')
    verify_geometry(calibration)
    paths = [RESULTS / f'portrait-npu-{direction}.npz' for direction in ('forward', 'reverse')]
    with np.load(paths[0], allow_pickle=False) as data:
        disparity = data['disparity'].astype(np.float32)
    with np.load(paths[1], allow_pickle=False) as data:
        reverse = np.ascontiguousarray(data['disparity'][:, ::-1], dtype=np.float32)
    if disparity.shape != (640, 352) or reverse.shape != disparity.shape:
        raise ValueError('Expected matching native portrait NPU outputs')
    if not np.isfinite(disparity).all() or not np.isfinite(reverse).all():
        raise ValueError('Nonfinite disparity prediction')
    depth, _, _, valid, _ = _depth_from_disparity(metric, disparity, reverse)
    geometry = projection_geometry(calibration)
    xyz_image = cv2.reprojectImageTo3D(disparity, geometry['Q'])
    valid &= np.isfinite(xyz_image).all(axis=2) & (xyz_image[:, :, 2] > 0)
    ys, xs = np.nonzero(valid)
    xyz = xyz_image[valid]
    if len(xyz) == 0:
        raise ValueError('No valid points to export')
    # Validate Q against the independent K inverse ray and metric Z formula.
    rays = np.linalg.solve(geometry['K'], np.vstack([xs, ys, np.ones(len(xs))])).T
    independent = rays * depth[valid, None]
    error = float(np.max(np.abs(xyz-independent)))
    if error > 2e-5:
        raise ValueError(f'Point projection mismatch: {error} m')
    uvw = xyz.astype(np.float64) @ geometry['K'].T
    uv = uvw[:, :2] / uvw[:, 2:]
    reprojection = float(np.max(np.abs(uv-np.column_stack([xs, ys]))))
    if reprojection > 0.001:
        raise ValueError(f'Pixel roundtrip mismatch: {reprojection} px')
    near, far = np.percentile(xyz[:, 2], [5, 95])
    if far-near < 0.01:
        near, far = max(0.0, near-0.01), far+0.01
    shade = np.clip((far-xyz[:, 2])/(far-near)*255, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(shade[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices = np.empty(len(xyz), dtype=dtype)
    for index, name in enumerate(('x', 'y', 'z')):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = rgb[:, index]
    capture = report['heldout_ids'][0]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment units metres; camera X right Y down Z forward\n'
              'comment colors represent depth, not camera RGB\n'
              f'comment source capture {capture}\n'
              f'element vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')
    ply = RESULTS / 'pointcloud.ply'
    ply.write_bytes(header + vertices.tobytes())
    restored = np.frombuffer(ply.read_bytes(), dtype=dtype, offset=len(header))
    assert len(restored) == len(vertices) and np.array_equal(restored, vertices)
    bounds = {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()}
    meta = {'source': f'LightStereo RKNN · 已保存照片 {capture}',
            'capture_id': capture, 'calibration_id': metric.identifier,
            'count': len(xyz), 'bounds': bounds, 'color_range_m': [float(near), float(far)],
            'coordinate_system': 'metres; X right, Y down, Z forward; left rectified camera',
            'filter': 'rectification support in both eyes, LR <=1px, depth 0.3-5m; no hole filling',
            'scope': 'one saved stereo pair; no multi-frame fusion or independent accuracy claim',
            'projection_validation': {'max_xyz_difference_m': error, 'max_pixel_roundtrip_px': reprojection},
            'input_sha256': {str(path.relative_to(HERE)): sha256(path) for path in [calibration, *paths] if path.is_relative_to(HERE)},
            'calibration_sha256': sha256(calibration), 'ply_sha256': sha256(ply)}
    (RESULTS / 'pointcloud.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    # PLY contains every valid point; the portable viewer receives a uniform sample.
    step = max(1, int(np.ceil(len(xyz)/25000)))
    payload = dict(meta, points=np.column_stack([xyz[::step].round(5), rgb[::step]]).tolist())
    template = (HERE / 'pointcloud_viewer.html').read_text(encoding='utf-8')
    assert template.count('__POINTCLOUD_DATA__') == 1
    html = template.replace('__POINTCLOUD_DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c'))
    (RESULTS / 'pointcloud.html').write_text(html, encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
