"""Online pinhole calibration in sensor coordinates, with millimetre object points."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from uuid import uuid4

import cv2
import numpy as np

SIZE = (3840, 2160)


@dataclass(frozen=True)
class BoardConfig:
    cols: int = 9
    rows: int = 9
    square_mm: float = 2.0

    def __post_init__(self):
        if (isinstance(self.cols, bool) or isinstance(self.rows, bool)
                or int(self.cols) != self.cols or int(self.rows) != self.rows
                or not 3 <= self.cols <= 24 or not 3 <= self.rows <= 24
                or not math.isfinite(float(self.square_mm)) or not .1 <= self.square_mm <= 200):
            raise ValueError('内角点行列须为 3–24 的整数，格长须为 0.1–200 mm 的有限数值')


def object_grid(config):
    obj = np.zeros((config.cols * config.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:config.cols, 0:config.rows].T.reshape(-1, 2)
    return obj * float(config.square_mm)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def validate_model(model):
    shapes = dict(K1=(3, 3), K2=(3, 3), R=(3, 3), R1=(3, 3), R2=(3, 3),
                  P1=(3, 4), P2=(3, 4), Q=(4, 4))
    for key, shape in shapes.items():
        a = np.asarray(model.get(key, []))
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('标定矩阵无效: ' + key)
    for key in ('T', 'D1', 'D2'):
        a = np.asarray(model.get(key, []))
        if not np.isfinite(a).all() or (a.size != 3 if key == 'T' else a.size not in (4, 5, 8)):
            raise ValueError('标定向量无效: ' + key)
    if tuple(model.get('image_size', [])) != SIZE or 'mm' not in str(model.get('coordinate_convention', '')):
        raise ValueError('标定必须是 3840×2160 原图坐标，物点/平移单位为 mm')
    if str(model.get('status')) != 'geometry_pass_metric_accuracy_unverified':
        raise ValueError('候选标定尚未通过几何质量检查')
    for key in ('K1', 'K2'):
        k = model[key]
        if not (100 < k[0, 0] < 30000 and 100 < k[1, 1] < 30000
                and 0 < k[0, 2] < SIZE[0] and 0 < k[1, 2] < SIZE[1]):
            raise ValueError('焦距/主点超出合理范围: ' + key)
    for key in ('R', 'R1', 'R2'):
        if not np.allclose(model[key].T @ model[key], np.eye(3), atol=1e-4) or abs(np.linalg.det(model[key]) - 1) > 1e-4:
            raise ValueError('旋转矩阵不是有效的正旋转: ' + key)
    if not 30 <= np.linalg.norm(model['T']) <= 500:
        raise ValueError('双目基线应为 30–500 mm，请检查格长和角点顺序')
    p2 = model['P2']
    if abs(p2[1, 3]) <= abs(p2[0, 3]) or p2[1, 3] >= 0:
        raise ValueError('基线方向不符合当前竖直双目/逆时针转正的运行约定')


def portrait_geometry(model):
    """Derive the reference 352x640 portrait K/Q from the active sensor NPZ."""
    p1, p2 = model['P1'], model['P2']
    scale, crop = 1. / 6, 4
    fx, fy = float(p1[1, 1]) * scale, float(p1[0, 0]) * scale
    cx = float(p1[1, 2]) * scale + (scale - 1) / 2 - crop
    cy = 639 - (float(p1[0, 2]) * scale + (scale - 1) / 2)
    baseline_m = -float(p2[1, 3]) / float(p2[1, 1]) / 1000
    delta = float(p1[1, 2] - p2[1, 2]) * scale
    if not np.isfinite([fx, fy, cx, cy, baseline_m]).all() or min(fx, fy, baseline_m) <= 0:
        raise ValueError('肖像投影几何无效')
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    q = np.array([[1, 0, 0, -cx], [0, fx / fy, 0, -cy * fx / fy],
                  [0, 0, 0, fx], [0, 0, 1 / baseline_m, -delta / baseline_m]])
    return {'K': k, 'Q': q}


def epipolar_errors(samples, model):
    errors = []
    for pair in samples:
        rect = [cv2.undistortPoints(np.asarray(pair[key], np.float64), model['K' + str(n)],
                                    model['D' + str(n)], R=model['R' + str(n)], P=model['P' + str(n)])
                for key, n in (('left', 1), ('right', 2))]
        errors.extend(np.abs(rect[0][:, 0, 0] - rect[1][:, 0, 0]).tolist())
    return np.asarray(errors)


def coverage(samples, size=SIZE):
    counts = [[0] * 9, [0] * 9]
    for pair in samples:
        for eye, name in enumerate(('left', 'right')):
            center = np.asarray(pair[name]).reshape(-1, 2).mean(axis=0)
            x, y = np.clip((center / size * 3).astype(int), 0, 2)
            counts[eye][int(y * 3 + x)] += 1
    return counts


def solve_samples(samples, config, size=SIZE):
    if tuple(size) != SIZE:
        raise ValueError('只能使用 3840×2160 原图求解')
    if len(samples) < 30:
        raise ValueError('至少需要 30 组有效样本（25 组求解 + 至少 5 组留出验证）')
    obj = object_grid(config)
    for pair in samples:
        for name in ('left', 'right'):
            a = np.asarray(pair[name])
            if a.shape != (len(obj), 1, 2) or not np.isfinite(a).all():
                raise ValueError('样本角点尺寸或数值无效')
    counts = coverage(samples, size)
    reasons = []
    for eye, grid in zip(('左', '右'), counts):
        if sum(v > 0 for v in grid) < 6 or any(grid[i] == 0 for i in (0, 2, 4, 6, 8)):
            reasons.append(eye + '目覆盖不足：需中心、四角及至少 6 个区域')
    unique = {np.asarray(s['left']).round(1).tobytes() for s in samples}
    if len(unique) < 25:
        reasons.append('有效姿态不足，存在大量重复样本')
    if reasons:
        return None, dict(passed=False, reasons=reasons, coverage=counts, count=len(samples))
    hold_count = min(len(samples) - 25, len(samples) // 5)
    hold_indices = set(list(range(4, len(samples), 5))[:hold_count])
    train = [s for i, s in enumerate(samples) if i not in hold_indices]
    hold = [s for i, s in enumerate(samples) if i in hold_indices]
    objpoints = [obj.copy() for _ in train]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 70, 1e-7)
    outputs = [cv2.calibrateCamera(objpoints, [np.asarray(s[name], np.float32) for s in train],
                                  size, None, None, criteria=criteria) for name in ('left', 'right')]
    (rms_l, k1, d1, rv, _), (rms_r, k2, d2, _, _) = outputs
    rms_s, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        objpoints, [np.asarray(s['left'], np.float32) for s in train],
        [np.asarray(s['right'], np.float32) for s in train], k1, d1, k2, d2, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k1, d1, k2, d2, size, r, t,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    model = dict(K1=k1, D1=d1, K2=k2, D2=d2, R=r, T=t, R1=r1, R2=r2, P1=p1, P2=p2, Q=q,
                 image_size=np.array(size), coordinate_convention=np.array('sensor coordinates; T in mm; CCW upright'),
                 status=np.array('geometry_pass_metric_accuracy_unverified'), square_mm=np.array(config.square_mm))
    errors = epipolar_errors(hold, model)
    mean, p95 = float(errors.mean()), float(np.percentile(errors, 95))
    report = dict(passed=False, count=len(samples), train_count=len(train), holdout_count=len(hold),
                  rms_left_px=float(rms_l), rms_right_px=float(rms_r), rms_stereo_px=float(rms_s),
                  epipolar_mean_px=mean, epipolar_p95_px=p95, baseline_mm=float(np.linalg.norm(t)),
                  coverage=counts, board=asdict(config), reasons=reasons,
                  distance_accuracy='几何验收；实际距离精度需独立卷尺/测距仪验证')
    if not np.isfinite([rms_l, rms_r, rms_s, mean, p95]).all() or max(rms_l, rms_r, rms_s) >= .5 or mean >= .30 or p95 >= .80:
        reasons.append('重投影/留出极线误差未达标，补采清晰静止且不同角度的样本')
    rotations = np.array([v.ravel() for v in rv])
    if max(np.ptp(rotations, axis=0)) < .15:
        reasons.append('倾斜姿态变化不足，需改变棋盘俯仰与偏航')
    try:
        validate_model(model)
        portrait_geometry(model)
    except ValueError as exc:
        reasons.append(str(exc))
    report['passed'] = not reasons
    if reasons:
        model['status'] = np.array('geometry_failed')
    return model, report


def detect_corners(gray, config, hint=None):
    """Bound searches to small images/ROIs; refine on untouched sensor pixels."""
    pattern = (int(config.cols), int(config.rows))
    # FAST_CHECK rejects low-contrast gold boards even when classic succeeds.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    scale = min(1., 720. / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    def search(image):
        ok, points = cv2.findChessboardCorners(image, pattern, flags)
        if not ok and hasattr(cv2, 'findChessboardCornersSB'):
            ok, points = cv2.findChessboardCornersSB(image, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
        return points if ok else None

    c = None
    # Reuse a successfully located board before considering clutter elsewhere.
    regions = []
    height, width = gray.shape[:2]
    if hint is not None:
        points = np.asarray(hint).reshape(-1, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        pad = np.maximum((high - low) * .35, 32)
        regions.append((*np.maximum(low - pad, 0).astype(int),
                        *np.minimum(high + pad, (width, height)).astype(int)))
    if not regions:
        c = search(small)
        if c is not None:
            c /= scale
    if c is None:
        # Dense clusters of image corners propose at most three local searches.
        # Constant padding prevents image-edge reflections from inflating scores.
        features = cv2.goodFeaturesToTrack(small, 1000, .03, max(2, round(4 * scale / .25)))
        if features is not None:
            heat = np.zeros(small.shape, np.uint8)
            xy = features[:, 0, :].astype(int)
            heat[xy[:, 1], xy[:, 0]] = 1
            window = max(20, round(400 * scale))
            density = cv2.boxFilter(heat, cv2.CV_32F, (window, window), normalize=False,
                                    borderType=cv2.BORDER_CONSTANT)
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(density)
                if score < 12:
                    break
                radius = max(10, round(320 * scale))
                density[max(0, y-radius):y+radius, max(0, x-radius):x+radius] = 0
                roi_w, roi_h = min(640, width), min(640, height)
                x0 = max(0, min(width-roi_w, round(x/scale)-roi_w//2))
                y0 = max(0, min(height-roi_h, round(y/scale)-roi_h//2))
                regions.append((x0, y0, x0+roi_w, y0+roi_h))
        for x0, y0, x1, y1 in regions:
            roi = gray[y0:y1, x0:x1]
            factor = min(1., 720. / max(roi.shape))
            points = search(cv2.resize(roi, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA))
            if points is not None:
                c = points / factor + np.array([x0, y0], np.float32)
                break
    if c is None:
        return None
    pitch = np.median(np.linalg.norm(np.diff(c.reshape(config.rows, config.cols, 2), axis=1), axis=2))
    window = max(1, min(5, int(pitch / 4)))
    c = cv2.cornerSubPix(gray, c.astype(np.float32), (window, window), (-1, -1), (3, 30, .01))
    # A square chessboard allows 90-degree ambiguity: canonicalize both eyes.
    grid = c.reshape(int(config.rows), int(config.cols), 2)
    variants = [grid, grid[::-1, ::-1]]
    if config.rows == config.cols:
        variants += [np.rot90(grid, 1), np.rot90(grid, 3)]
    grid = max(variants, key=lambda g: float((g[0, -1] - g[0, 0])[0]))
    return np.ascontiguousarray(grid.reshape(-1, 1, 2), dtype=np.float32)


class CalibrationSession:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.directory = None
        self.config = BoardConfig()
        self.samples = []
        self.report = None
        self.state = 'idle'
        latest = self.root / 'latest.json'
        if latest.is_file():
            self.resume(json.loads(latest.read_text(encoding='utf-8'))['id'])

    def safe_dir(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('会话 ID 无效')
        return self.root / identifier

    def resume(self, identifier):
        directory = self.safe_dir(identifier)
        data = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
        self.directory = directory
        self.config = BoardConfig(**data['board'])
        self.samples = []
        for p in sorted(directory.glob('corners_*.npz')):
            d = load_model(p)
            self.samples.append({'left': d['left'], 'right': d['right']})
        self.report = json.loads((directory / 'report.json').read_text(encoding='utf-8')) if (directory / 'report.json').is_file() else None
        self.state = 'review' if self.report else 'idle'

    def start(self, config, resume=False):
        with self.lock:
            if not resume or self.directory is None:
                self.directory = self.safe_dir(uuid4().hex)
                self.directory.mkdir()
                self.samples, self.report = [], None
                self.config = config
                atomic_json(self.directory / 'session.json', dict(board=asdict(config), created_at=datetime.now(timezone.utc).isoformat()))
                atomic_json(self.root / 'latest.json', {'id': self.directory.name})
            elif config != self.config:
                raise ValueError('继续会话必须使用原棋盘规格；改变规格请新建会话')
            self.state = 'collecting'

    def add(self, pair, corners):
        with self.lock:
            if self.state != 'collecting':
                raise ValueError('请先开始采集；求解期间不能采集')
            if any(c is None for c in corners):
                raise ValueError('左右都需识别完整棋盘，检查内角点行列、清晰度和反光')
            for c in corners:
                if c.shape != (self.config.cols * self.config.rows, 1, 2) or not np.isfinite(c).all():
                    raise ValueError('角点无效')
            if any(np.sqrt(np.mean((corners[0] - s['left']) ** 2)) < 3 for s in self.samples):
                raise ValueError('姿态与已采样本太接近，请移动或倾斜棋盘')
            number = len(self.samples)
            # Corners are the commit marker: incomplete image writes are never sampled.
            for name, frame in zip(('left', 'right'), pair):
                if frame.shape != SIZE[::-1] or frame.dtype != np.uint8:
                    raise ValueError('样本必须是 4K 原始灰度图')
                if not cv2.imwrite(str(self.directory / ('%s_%03d.png' % (name, number))), frame):
                    raise OSError('保存标定原图失败')
            temp = self.directory / 'corners.tmp.npz'
            np.savez_compressed(temp, left=corners[0], right=corners[1])
            os.replace(str(temp), str(self.directory / ('corners_%03d.npz' % number)))
            self.samples.append({'left': corners[0].copy(), 'right': corners[1].copy()})
            # New samples invalidate any previously reviewed candidate.
            self.report = None
            for name in ('candidate.npz', 'report.json'):
                p = self.directory / name
                if p.exists():
                    p.unlink()

    def snapshot(self):
        with self.lock:
            return dict(id=self.directory.name if self.directory else None, board=asdict(self.config),
                        state=self.state, count=len(self.samples), coverage=coverage(self.samples), report=self.report)

    def solve(self):
        with self.lock:
            if self.state not in ('collecting', 'review') or len(self.samples) < 30:
                raise ValueError('请先采集至少 30 组有效样本')
            self.state = 'solving'
        try:
            model, report = solve_samples(self.samples, self.config)
            if model is not None:
                np.savez_compressed(self.directory / 'candidate.npz', **model)
            atomic_json(self.directory / 'report.json', report)
            with self.lock:
                self.report, self.state = report, 'review'
            return report
        except Exception as exc:
            with self.lock:
                self.report = dict(passed=False, reasons=[str(exc)])
                self.state = 'review'
            atomic_json(self.directory / 'report.json', self.report)
            raise
