"""Camera-mode coordination and online calibration HTTP boundary."""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re
import threading
import time
import zipfile

import cv2
import numpy as np

from calibration import (BoardConfig, CalibrationSession, SIZE, detect_corners,
                         load_model, validate_model, portrait_geometry)
from calibration_apply import digest, prepare_transaction, launch_transaction


class CalibrationLive:
    def __init__(self, live):
        self.live = live
        self.root = Path(__file__).resolve().parent.parent
        self.session = CalibrationSession(self.root / 'calibration' / 'sessions')
        self.lock = threading.RLock()
        self.active = False
        self.recovery_needed = False
        self.heartbeat = 0
        self.previous_size = None
        self.detected = None
        self.hints = [None, None]
        self.jpeg = None
        self.raw_jpeg = None
        self.raw_at = 0
        self.raw_sequence = 0
        self.detectors = ThreadPoolExecutor(max_workers=2, thread_name_prefix='calibration-eye')
        self.message = '确认棋盘规格后开始校准；默认 9×9 内角点、每格 2 mm'
        self.thread = threading.Thread(target=self.work, name='calibration-detection', daemon=True)
        self.thread.start()
        threading.Thread(target=self.raw_work, name='calibration-preview', daemon=True).start()
        threading.Thread(target=self.watchdog, name='calibration-watchdog', daemon=True).start()

    def clear_outputs(self):
        live = self.live
        with live.condition:
            live.jpeg = live.jpeg_fixed = live.measure_jpeg = live.pointcloud_packet = None
            live.measure_input = live.pointcloud_input = None
            live.measure_history.clear()
            live.updated = None
            live.state.update(mode='calibrating' if self.active else 'waiting', fps=None,
                              depth_median_m=None, valid_fraction=None,
                              message='校准中：深度、测距、点云暂停' if self.active else '校准已退出，等待新画面')
            live.condition.notify_all()

    def start(self, payload):
        config = BoardConfig(payload.get('cols', 9), payload.get('rows', 9), float(payload.get('square_mm', 2)))
        with self.lock:
            if self.active:
                raise ValueError('正在校准，改变规格请先退出再新建会话')
            if self.session.state == 'solving':
                raise ValueError('上一会话仍在求解，请等待计算结束')
            self.session.start(config, bool(payload.get('resume', False)))
            if not self.live.store.lock.acquire(blocking=False):
                raise ValueError('正在拍照，请稍后开始校准')
            try:
                with self.live.capture_control_lock:
                    self.previous_size = (self.live.stream_width, self.live.stream_height)
                    with self.live.condition:
                        self.active = self.live.calibration_active = True
                        self.live.calibration_generation += 1
                    self.live.npu_preview.until = 0
                    self.clear_outputs()
                    self.live._stop_capture_pipeline()
                    try:
                        self.live._start_capture_pipeline(*SIZE)
                    except Exception:
                        self.recovery_needed = True
                        try:
                            self.live._start_capture_pipeline(*self.previous_size)
                            self.active = self.live.calibration_active = False
                            self.recovery_needed = False
                        except Exception:
                            self.message = '采集启动失败，正在重试恢复正常管线'
                        raise
                    self.detected, self.jpeg = None, None
                    self.raw_jpeg = None
                    self.hints = [None, None]
                    self.heartbeat = time.monotonic()
                    self.message = '4K 原图已启动；移动到中心/四角、改变距离和倾角，摆板停稳后采集'
            finally:
                self.live.store.lock.release()
        return self.status()

    def exit(self):
        with self.lock:
            if not self.active:
                return self.status()
            with self.live.capture_control_lock:
                with self.live.condition:
                    self.live.calibration_generation += 1
                self.recovery_needed = True
                self.clear_outputs()
                try:
                    self.live._stop_capture_pipeline()
                    self.live._start_capture_pipeline(*self.previous_size)
                except Exception:
                    self.message = '恢复正常管线失败，后台将重试；也可再次点击退出'
                    raise
                with self.live.condition:
                    self.active = self.live.calibration_active = False
                    self.recovery_needed = False
                self.clear_outputs()
            self.message = '已恢复正常运行；会话与候选结果已保存'
            self.detected, self.jpeg = None, None
            self.raw_jpeg = None
        return self.status()

    def raw_work(self):
        last = None
        while not self.live.stopped.wait(.08):
            if not self.active or self.recovery_needed:
                continue
            generation = self.live.calibration_generation
            with self.live.condition:
                pair = self.live.select_pair(require_sync=False)
            if pair is None or pair[0][0] == last:
                continue
            last = pair[0][0]
            try:
                eyes = [cv2.rotate(cv2.resize(item[1], (640, 360)), cv2.ROTATE_90_COUNTERCLOCKWISE)
                        for item in pair]
                ok, jpeg = cv2.imencode('.jpg', np.hstack(eyes), [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self.lock:
                    if ok and self.active and generation == self.live.calibration_generation:
                        self.raw_jpeg = jpeg.tobytes()
                        self.raw_at = time.monotonic()
                        self.raw_sequence += 1
            except Exception as exc:
                self.message = '预览异常：' + str(exc)

    def watchdog(self):
        while not self.live.stopped.wait(1):
            if self.recovery_needed or (self.active and time.monotonic() - self.heartbeat > 30):
                try:
                    self.exit()
                    self.message = '校准页面已离开或断线，自动恢复正常运行；可继续已保存会话'
                except Exception as exc:
                    self.message = '恢复采集失败：' + str(exc)

    def work(self):
        last = None
        while not self.live.stopped.wait(.2):
            if not self.active or self.recovery_needed or self.session.state == 'solving':
                continue
            generation = self.live.calibration_generation
            with self.live.condition:
                pair = self.live.select_pair()
            if pair is None or pair[0][0] == last:
                continue
            last = pair[0][0]
            try:
                started = time.monotonic()
                tasks = [self.detectors.submit(detect_corners, frame[1], self.session.config, hint)
                         for frame, hint in zip(pair, self.hints)]
                corners = [task.result() for task in tasks]
                eyes, zooms = [], []
                for item, points in zip(pair, corners):
                    gray = item[1]
                    small = cv2.cvtColor(cv2.resize(gray, (960, 540)), cv2.COLOR_GRAY2BGR)
                    if points is not None:
                        cv2.drawChessboardCorners(small, (self.session.config.cols, self.session.config.rows), points / 4, True)
                        low = np.maximum(points.reshape(-1, 2).min(axis=0).astype(int) - 20, 0)
                        high = np.minimum(points.reshape(-1, 2).max(axis=0).astype(int) + 21, SIZE)
                        crop = cv2.cvtColor(gray[low[1]:high[1], low[0]:high[0]], cv2.COLOR_GRAY2BGR)
                        cv2.drawChessboardCorners(crop, (self.session.config.cols, self.session.config.rows), points - low, True)
                        zoom = cv2.resize(crop, (300, 300))
                    else:
                        zoom = np.zeros((300, 300, 3), np.uint8)
                    eyes.append(cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE))
                    zooms.append(zoom)
                canvas = np.hstack(eyes)
                zoom_canvas = np.hstack(zooms)
                ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                zok, zencoded = cv2.imencode('.jpg', zoom_canvas)
                with self.lock:
                    if generation != self.live.calibration_generation or not self.active:
                        continue
                    self.detected = (pair, corners, time.monotonic())
                    self.hints = corners
                    self.detection_ms = round((time.monotonic() - started) * 1000)
                    if ok and zok:
                        self.jpeg, self.zoom_jpeg = encoded.tobytes(), zencoded.tobytes()
            except Exception as exc:
                self.message = '检测异常：' + str(exc)
            self.live.stopped.wait(.1)

    def capture(self):
        with self.lock:
            if not self.active or self.detected is None:
                raise ValueError('请开始校准并等待棋盘识别')
            pair, corners, detected_at = self.detected
            if time.monotonic() - detected_at > 5 or max(time.monotonic() - frame[2] for frame in pair) > 5:
                raise ValueError('检测图像已过期，请等待新一组角点画面')
            if abs(pair[0][0] - pair[1][0]) > 25_000_000:
                raise ValueError('左右时间偏差过大，请保持标定板静止')
            if all(c is not None for c in corners):
                dirs = [c[self.session.config.cols - 1, 0] - c[0, 0] for c in corners]
                angle = np.dot(dirs[0], dirs[1]) / max(1e-8, np.linalg.norm(dirs[0]) * np.linalg.norm(dirs[1]))
                if angle < .9:
                    raise ValueError('左右棋盘方向不一致，请减小倾斜，防止对称棋盘角点顺序歧义')
                for c in corners:
                    spacing = np.linalg.norm(np.diff(c.reshape(self.session.config.rows, self.session.config.cols, 2), axis=1), axis=2)
                    if np.median(spacing) < 4:
                        raise ValueError('每格在原图中不足 4 像素，请靠近并保持左右都能看到完整棋盘')
            if self.session.state == 'review':
                self.session.state = 'collecting'
            self.session.add([item[1] for item in pair], corners)
            self.message = '已保存 %d 组；继续移动棋盘到不同区域与倾角' % len(self.session.samples)
        return self.status()

    def solve(self):
        with self.lock:
            if not self.active or self.session.state == 'solving':
                raise ValueError('请在校准模式下计算，且同一时间只能运行一次')
            if len(self.session.samples) < 30:
                raise ValueError('至少需要 30 组有效采集样本')
            # Reserve state before launching to prevent two HTTP requests racing.
            self.session.state = 'solving'
            def run():
                try:
                    from calibration import solve_samples, atomic_json
                    model, report = solve_samples(self.session.samples, self.session.config)
                    if model is not None:
                        np.savez_compressed(self.session.directory / 'candidate.npz', **model)
                    atomic_json(self.session.directory / 'report.json', report)
                    with self.session.lock:
                        self.session.report, self.session.state = report, 'review'
                    self.message = '计算结束，请查看报告和校正预览；通过后可手动应用'
                except Exception as exc:
                    from calibration import atomic_json
                    with self.session.lock:
                        self.session.report = dict(passed=False, reasons=[str(exc)])
                        self.session.state = 'review'
                    atomic_json(self.session.directory / 'report.json', self.session.report)
                    self.message = '标定求解失败：' + str(exc)
            threading.Thread(target=run, name='calibration-solver', daemon=True).start()
        return self.status()

    def history(self):
        directory = self.root / 'calibration' / 'history'
        entries = []
        if directory.is_dir():
            for path in sorted(directory.glob('*.json'), reverse=True):
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if re.fullmatch(r'[a-f0-9]{32}', str(data['id'])) and path.with_suffix('.npz').is_file():
                        entries.append(data)
                except (OSError, ValueError, KeyError):
                    continue
        return entries

    def status(self, heartbeat=False):
        if heartbeat:
            self.heartbeat = time.monotonic()
        result = self.session.snapshot()
        with self.lock:
            pair, corners, at = self.detected if self.detected else (None, [None, None], None)
            result.update(active=self.active, message=self.message, detected=[c is not None for c in corners],
                          detection_age_ms=round((time.monotonic() - at) * 1000) if at else None,
                          detection_ms=getattr(self, 'detection_ms', None),
                          preview_sequence=self.raw_sequence,
                          preview_age_ms=round((time.monotonic()-self.raw_at)*1000) if self.raw_at else None,
                          image_age_ms=round(max(time.monotonic() - f[2] for f in pair) * 1000) if pair else None,
                          skew_ms=round(abs(pair[0][0] - pair[1][0]) / 1e6, 2) if pair else None,
                          current_calibration_id=digest(self.root / 'app' / 'calibration.npz'), history=self.history())
        latest = self.root / 'calibration' / 'latest-transaction.json'
        if latest.is_file():
            identifier = json.loads(latest.read_text(encoding='utf-8')).get('id', '')
            if re.fullmatch(r'[a-f0-9]{32}', identifier):
                tx = self.root / 'calibration' / 'transactions' / (identifier + '.json')
                if tx.is_file():
                    data = json.loads(tx.read_text(encoding='utf-8'))
                    result['transaction'] = {k: data.get(k) for k in ('id', 'state', 'expected', 'previous', 'error')}
        return result

    def preview(self, zoom=False, rectified=False):
        if rectified:
            directory = self.session.directory
            if directory is None or not (directory / 'candidate.npz').is_file():
                raise ValueError('尚无候选标定')
            model = load_model(directory / 'candidate.npz')
            eyes = []
            for n, name in ((1, 'left'), (2, 'right')):
                paths = sorted(directory.glob(name + '_*.png'))
                if not paths:
                    raise ValueError('暂无样本图像')
                gray = cv2.imread(str(paths[-1]), cv2.IMREAD_GRAYSCALE)
                scale = 1. / 4
                transform = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1.]])
                maps = cv2.initUndistortRectifyMap(transform @ model['K' + str(n)], model['D' + str(n)],
                    model['R' + str(n)], transform @ model['P' + str(n)], (960, 540), cv2.CV_32FC1)
                eye = cv2.remap(cv2.resize(gray, (960, 540)), *maps, cv2.INTER_LINEAR)
                eyes.append(cv2.cvtColor(cv2.rotate(eye, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_GRAY2BGR))
            image = np.hstack(eyes)
            for y in range(0, image.shape[0], 40):
                cv2.line(image, (0, y), (image.shape[1], y), (0, 220, 0), 1)
            return cv2.imencode('.jpg', image)[1].tobytes()
        if not self.active or self.detected is None or time.monotonic() - self.detected[2] > 5:
            raise ValueError('等待新鲜 4K 棋盘检测画面')
        return self.zoom_jpeg if zoom else self.jpeg

    def download(self):
        with self.lock:
            if self.session.directory is None or self.session.state == 'solving':
                raise ValueError('暂无会话或正在求解，请稍候')
            # Stream from a file, not an in-memory ZIP containing all 4K images.
            target = self.root / 'calibration' / 'session-download.zip'
            with zipfile.ZipFile(str(target), 'w', zipfile.ZIP_STORED) as archive:
                for p in self.session.directory.iterdir():
                    if p.suffix in ('.png', '.npz', '.json') and not p.name.endswith('.tmp'):
                        archive.write(str(p), p.name)
            return target

    def apply(self, history_id=None):
        with self.lock:
            if self.session.state == 'solving':
                raise ValueError('正在求解，请等待完成')
            if history_id is not None:
                if not isinstance(history_id, str) or not re.fullmatch(r'[a-f0-9]{32}', history_id):
                    raise ValueError('历史标定 ID 无效')
                candidate = self.root / 'calibration' / 'history' / (history_id + '.npz')
            else:
                if not self.session.report or not self.session.report.get('passed'):
                    raise ValueError('仅能应用通过质量检查的候选标定')
                candidate = self.session.directory / 'candidate.npz'
            model = load_model(candidate)
            validate_model(model)
            portrait_geometry(model)
            from fast_depth import FastSGBMDepth
            from range4k import Range4K
            FastSGBMDepth(candidate, input_size=(640, 360))
            Range4K(candidate)
            from pointcloud import PointCloudStream
            PointCloudStream(candidate)
            transaction = prepare_transaction(self.root, candidate)
            identifier = launch_transaction(transaction)
            self.message = '正在应用标定并重新启动服务；连接恢复后检查新标定 ID'
            return {'ok': True, 'transaction_id': identifier, 'message': self.message}

    def handle_get(self, handler, path, query):
        if path == '/calibration':
            return handler.send(Path(__file__).with_name('calibration.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/calibration/status':
            return handler.json_response(self.status(query.get('heartbeat') == ['1']))
        if path == '/calibration/raw.jpg':
            with self.lock:
                if not self.active:
                    raise ValueError('请开始校准')
                if self.raw_jpeg is None or time.monotonic()-self.raw_at > 1:
                    raise ValueError('等待新鲜 4K 左右原图')
                jpeg = self.raw_jpeg
            return handler.send(jpeg, 'image/jpeg')
        if path == '/calibration/session.zip':
            target = self.download()
            handler.send_response(200)
            handler.send_header('Content-Type', 'application/zip')
            handler.send_header('Content-Length', str(target.stat().st_size))
            handler.send_header('Content-Disposition', 'attachment; filename="calibration-session.zip"')
            handler.end_headers()
            with target.open('rb') as file:
                while True:
                    block = file.read(65536)
                    if not block:
                        break
                    handler.wfile.write(block)
            return
        if path in ('/calibration/preview.jpg', '/calibration/zoom.jpg', '/calibration/rectified.jpg'):
            return handler.send(self.preview(path.endswith('zoom.jpg'), path.endswith('rectified.jpg')), 'image/jpeg')
        return handler.send_error(404)

    def handle_post(self, handler, path, payload):
        if path == '/calibration/start':
            result = self.start(payload)
        elif path == '/calibration/capture':
            result = self.capture()
        elif path == '/calibration/solve':
            result = self.solve()
        elif path == '/calibration/exit':
            result = self.exit()
        elif path == '/calibration/apply':
            result = self.apply()
        elif path == '/calibration/restore':
            result = self.apply(payload.get('id', ''))
        else:
            return handler.send_error(404)
        return handler.json_response(result)
