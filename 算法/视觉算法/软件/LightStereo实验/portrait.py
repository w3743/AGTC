"""Shared crop geometry: 360x640 -> 352x640, no image rescaling or padding."""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stereo.depth import MetricDepth


class CroppedMetric(MetricDepth):
    CROP = 4

    def __init__(self, path):
        super().__init__(path)
        self.width -= 2*self.CROP
        self.masks = [m[:, self.CROP:-self.CROP].copy() for m in self.masks]
        self.yy, self.xx = np.indices((self.height, self.width))

    def rectify(self, left, right):
        return [np.ascontiguousarray(im[:, self.CROP:-self.CROP])
                for im in super().rectify(left, right)]


def projection_geometry(path, crop=4):
    with np.load(path, allow_pickle=False) as m:
        scale = MetricDepth.RAW_SIZE[0]/MetricDepth.SENSOR_SIZE[0]
        resize = np.array([[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2], [0, 0, 1.]])
        rotate_image = np.array([[0, 1, 0], [-1, 0, 639], [0, 0, 1.]])
        translate = np.array([[1, 0, -crop], [0, 1, 0], [0, 0, 1.]])
        rotate_camera = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]])
        camera_inverse = np.eye(4)
        camera_inverse[:3, :3] = rotate_camera.T
        ps = [translate @ rotate_image @ resize @ m[f'P{i}'] @ camera_inverse for i in (1, 2)]
    for p in ps:
        p[:, 3] /= 1000  # Projection accepts XYZ in metres, calibration stored mm.
    p1, p2 = ps
    f, cx, cy, cx2 = p1[0, 0], p1[0, 2], p1[1, 2], p2[0, 2]
    tx = p2[0, 3]/f
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f],
                  [0, 0, -1/tx, (cx-cx2)/tx]])
    return dict(P1=p1, P2=p2, K=p1[:, :3], Q=q, image_size=np.array([360-2*crop, 640]),
                crop_left_right=np.array([crop, crop]), depth_unit=np.array('m'),
                axes=np.array('upright rectified camera: X right, Y down, Z forward'))


def verify_geometry(path):
    full, crop = projection_geometry(path, 0), projection_geometry(path, 4)
    metric = CroppedMetric(path)
    largest = 0.
    for x, y, z in ((.03, .05, .4), (-.12, .08, 1.), (.4, -.2, 3.)):
        point = np.array([x, y, z, 1.])
        pixels = []
        for key in ('P1', 'P2'):
            old, new = full[key] @ point, crop[key] @ point
            old, new = old[:2]/old[2], new[:2]/new[2]
            assert np.allclose(new, old-[4, 0])
            pixels.append(new)
        d = pixels[0][0]-pixels[1][0]
        zz = metric.depth_factor/(d-metric.disparity_offset)
        restored = crop['Q'] @ np.array([*pixels[0], d, 1.])
        restored = restored[:3]/restored[3]
        largest = max(largest, float(np.max(np.abs(restored-point[:3]))), abs(zz-z))
    assert largest < 1e-10
    return {'crop_projection_and_Q_max_error_m': largest, 'view_width_lost_fraction': 8/360}
