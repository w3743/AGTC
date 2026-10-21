"""Calibrated upright stereo depth for the fixed, sideways-mounted camera pair."""
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np


class MetricDepth:
    # Rectify in sensor coordinates, then rotate both images CCW into horizontal stereo.
    SENSOR_SIZE = (3840, 2160)
    RAW_SIZE = (640, 360)
    RANGE_M = (.3, 5.)

    def __init__(self, path):
        path = Path(path)
        self.identifier = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        with np.load(path, allow_pickle=False) as saved:
            model = {key: saved[key].copy() for key in saved.files}
        if str(model.get('status')) != 'geometry_pass_metric_accuracy_unverified':
            raise ValueError('标定参数尚未通过几何验收')
        if tuple(model['image_size']) != self.SENSOR_SIZE:
            raise ValueError('标定分辨率与相机不一致')
        if 'mm' not in str(model.get('coordinate_convention', '')):
            raise ValueError('标定参数缺少毫米单位说明')
        for key in ('K1', 'K2', 'D1', 'D2', 'R1', 'R2', 'P1', 'P2', 'T'):
            if not np.isfinite(model[key]).all():
                raise ValueError('标定参数存在无效数值')
        p1, p2 = model['P1'], model['P2']
        if abs(p2[1, 3]) <= abs(p2[0, 3]) or p2[1, 3] >= 0:
            raise ValueError('当前算法需要原图竖直基线、转正后正视差的标定参数')
        scale = self.RAW_SIZE[0] / self.SENSOR_SIZE[0]
        resize = np.array([[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2], [0, 0, 1.]])
        self.maps, self.masks = [], []
        for n in (1, 2):
            k = resize @ model[f'K{n}']
            p = resize @ model[f'P{n}']
            maps = cv2.initUndistortRectifyMap(k, model[f'D{n}'], model[f'R{n}'], p,
                                              self.RAW_SIZE, cv2.CV_32FC1)
            self.maps.append(maps)
            mask = cv2.remap(np.full(self.RAW_SIZE[::-1], 255, np.uint8), *maps, cv2.INTER_LINEAR)
            self.masks.append(cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE) > 254)
        # For vertical stereo d=y_left-y_right. CCW rotation makes x'=y, preserving d.
        self.depth_factor = -float(p2[1, 3]) * scale / 1000  # focal_px * baseline_m
        self.disparity_offset = float(p1[1, 2]-p2[1, 2]) * scale
        self.baseline_mm = float(np.linalg.norm(model['T']))
        self.width, self.height = self.RAW_SIZE[1], self.RAW_SIZE[0]
        self.span = int(np.ceil((self.depth_factor/self.RANGE_M[0] + abs(self.disparity_offset))/16)*16)
        if self.span >= self.width or self.span > 256:
            raise ValueError('标定视差搜索范围异常')
        args = dict(numDisparities=self.span, blockSize=5, P1=200, P2=800,
                    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=80,
                    speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, **args)
        self.right_min = -self.span+1
        self.right_matcher = cv2.StereoSGBM_create(minDisparity=self.right_min, **args)
        self.yy, self.xx = np.indices((self.height, self.width))
        self.auto_range = None
        self.filter_stats = {}

    def rectify(self, left, right):
        result = []
        for gray, maps in zip((left, right), self.maps):
            if gray.shape != self.SENSOR_SIZE[::-1] or gray.dtype != np.uint8:
                raise ValueError('米制深度需要未经旋转裁剪的3840×2160灰度原图')
            small = cv2.resize(gray, self.RAW_SIZE, interpolation=cv2.INTER_AREA)
            result.append(cv2.rotate(cv2.remap(small, *maps, cv2.INTER_LINEAR), cv2.ROTATE_90_COUNTERCLOCKWISE))
        return result

    def metric_values(self, dl, dr, left):
        xr = np.rint(self.xx-dl).astype(np.int32)
        safe = xr.clip(0, self.width-1)
        valid = (xr >= 0) & (xr < self.width) & (dl > 0) & (dl < self.span)
        matched = float(valid.mean())
        valid &= dr[self.yy, safe] > self.right_min-1
        valid &= np.abs(dl+dr[self.yy, safe]) <= 1.0
        valid &= self.masks[0] & self.masks[1][self.yy, safe]
        consistent = float(valid.mean())
        # Window contrast retains low-frequency texture, not just Laplacian edges.
        # Keep real image support, uniqueness and left-right checks; never fill holes.
        gray = left.astype(np.float32)
        variance = cv2.boxFilter(gray*gray, -1, (7, 7)) - cv2.boxFilter(gray, -1, (7, 7))**2
        valid &= variance > 4.0  # local standard deviation > 2 grayscale levels
        textured = float(valid.mean())
        denom = dl-self.disparity_offset
        depth = np.full(dl.shape, np.nan, np.float32)
        np.divide(self.depth_factor, denom, out=depth, where=denom > 0)
        valid &= np.isfinite(depth) & (depth >= self.RANGE_M[0]) & (depth <= self.RANGE_M[1])
        depth[~valid] = np.nan
        self.filter_stats = dict(matched_fraction=matched, consistent_fraction=consistent,
                                 textured_fraction=textured, range_fraction=float(valid.mean()))
        return depth, valid

    def disparities(self, left, right):
        # Avoid SGBM discarding a whole search-width strip at each image edge.
        # metric_values still rejects every match outside the real source images.
        # Each direction needs padding only on its search boundary, not both sides.
        l, r = [cv2.copyMakeBorder(im, 0, 0, self.span, 0, cv2.BORDER_CONSTANT, value=0)
                for im in (left, right)]
        dl = self.left_matcher.compute(l, r).astype(np.float32)[:, self.span:]/16
        r, l = [cv2.copyMakeBorder(im, 0, 0, 0, self.span, cv2.BORDER_CONSTANT, value=0)
                for im in (right, left)]
        dr = self.right_matcher.compute(r, l).astype(np.float32)[:, :self.width]/16
        return dl, dr

    @staticmethod
    def colorize(depth, bounds):
        valid = np.isfinite(depth)
        near, far = bounds
        shade = np.zeros(depth.shape, np.uint8)
        shade[valid] = np.clip((far-depth[valid]) / max(far-near, .001)*255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        return color

    def compute(self, left, right):
        started = time.perf_counter()
        l, r = self.rectify(left, right)
        rectified = time.perf_counter()
        dl, dr = self.disparities(l, r)
        matched = time.perf_counter()
        depth, valid = self.metric_values(dl, dr, l)
        if valid.any():
            minimum, p05, median, p95, maximum = np.percentile(depth[valid], [0, 5, 50, 95, 100]).tolist()
            span = max(.25, (p95-p05)*1.2)
            center = (p05+p95)/2
            target = np.array([max(self.RANGE_M[0], center-span/2), min(self.RANGE_M[1], center+span/2)])
            self.auto_range = target if self.auto_range is None else self.auto_range*.8 + target*.2
        else:
            minimum = p05 = median = p95 = maximum = None
        bounds = self.auto_range.tolist() if self.auto_range is not None else list(self.RANGE_M)
        color = self.colorize(depth, bounds)
        finished = time.perf_counter()
        rounded = lambda value: None if value is None else round(float(value), 3)
        details = dict(depth_unit='m', depth_range_m=list(self.RANGE_M),
                       depth_color_range_m=[round(v, 3) for v in bounds], depth_color_mode='auto',
                       depth_min_m=rounded(minimum), depth_max_m=rounded(maximum),
                       depth_p05_m=rounded(p05), depth_p95_m=rounded(p95), depth_median_m=rounded(median),
                       process_ms=round((finished-started)*1000, 1),
                       stage_ms=dict(rectify=round((rectified-started)*1000, 1),
                                     match=round((matched-rectified)*1000, 1),
                                     filter_color=round((finished-matched)*1000, 1)),
                       filter_stats=self.filter_stats,
                       valid_fraction=float(valid.mean()), baseline_mm=round(self.baseline_mm, 3),
                       calibration_id=self.identifier, depth_resolution=[self.width, self.height])
        return color, depth, details
