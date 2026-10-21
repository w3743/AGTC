#!/usr/bin/env python3
"""On-board dual-camera capture and calibrated pseudocolor depth."""
import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import signal
import threading
import time
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4
import zipfile

import cv2
import numpy as np
from depth import MetricDepth

WIDTH, HEIGHT = 640, 360
CAPTURE_WIDTH, CAPTURE_HEIGHT = 3840, 2160
MAX_SKEW_NS = 25_000_000
MAX_FRAME_AGE = .75
CAPTURE_TARGET = dict(
    id='screen-checkerboard-10x7', inner_corners=[9, 6], square_size_mm=24.2,
    measurement='User measured five cells horizontally and vertically: 121 mm each',
    source='User-selected target; not automatically detected',
)


class CaptureStore:
    """Only complete, explicitly requested stereo pairs become visible."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.last_stamps = None

    def entries(self):
        result = []
        for path in sorted(self.directory.glob('*/meta.json')):
            if not re.fullmatch(r'\d{8}T\d{12}Z-[a-f0-9]{8}', path.parent.name):
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if (data.get('id') == path.parent.name and
                        all((path.parent / name).is_file() for name in ('left.png', 'right.png'))):
                    result.append(data)
            except (OSError, ValueError):
                continue
        return result

    def save(self, pair, devices):
        a, b = pair
        now = time.monotonic()
        if max(now-a[2], now-b[2]) > MAX_FRAME_AGE:
            raise ValueError('图像已过期，请等待两路实时画面恢复')
        if abs(a[0]-b[0]) > MAX_SKEW_NS:
            raise ValueError('左右时间差过大，请保持板子静止并稍后重试')
        if self.last_stamps and any(s <= old for s, old in zip((a[0], b[0]), self.last_stamps)):
            raise ValueError('这组图像已保存，请等待新画面')
        for frame in (a[1], b[1]):
            if frame.shape != (CAPTURE_HEIGHT, CAPTURE_WIDTH) or frame.dtype != np.uint8:
                raise ValueError('采集分辨率异常，未保存照片')
        stamp = datetime.now(timezone.utc)
        identifier = stamp.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:8]
        temporary = self.directory / ('.'+identifier+'.tmp')
        destination = self.directory / identifier
        metadata = dict(id=identifier, captured_at=stamp.isoformat(),
                        width=CAPTURE_WIDTH, height=CAPTURE_HEIGHT,
                        skew_ms=round(abs(a[0]-b[0])/1e6, 3),
                        left_pts_ns=int(a[0]), right_pts_ns=int(b[0]),
                        left_device=devices[0], right_device=devices[1],
                        pixel_format='GRAY8', rectified=False,
                        sync='GStreamer running-time pairing; hardware exposure sync unverified',
                        square_size_reference_mm=CAPTURE_TARGET['square_size_mm'], baseline_reference_mm=95,
                        calibration_target=CAPTURE_TARGET.copy(),
                        calibrated=False)
        temporary.mkdir()
        try:
            for name, frame in (('left.png', a[1]), ('right.png', b[1])):
                if not cv2.imwrite(str(temporary/name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError('无法写入 '+name)
            (temporary/'meta.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.last_stamps = (a[0], b[0])
        return metadata


class Matcher:
    def __init__(self):
        self.transforms = None
        self.detail = {}

    @staticmethod
    def correspondences(left, right):
        detector = cv2.SIFT_create(nfeatures=1800)
        k1, d1 = detector.detectAndCompute(left, None)
        k2, d2 = detector.detectAndCompute(right, None)
        if d1 is None or d2 is None:
            raise ValueError('纹理不足，请让两路相机同时看到有纹理的静止物体')
        matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .75*n.distance]
        if len(good) < 28:
            raise ValueError('共同纹理不足，请固定相机并让两路画面包含更多清晰纹理')
        p1 = np.float32([k1[m.queryIdx].pt for m in good])
        p2 = np.float32([k2[m.trainIdx].pt for m in good])
        return p1, p2

    @staticmethod
    def check_geometry(p1, p2, transforms):
        if len(p1) < 28:
            raise ValueError('共同纹理不足，请固定相机并让两路画面包含更多清晰纹理')
        for points in (p1, p2):
            area = cv2.contourArea(cv2.convexHull(points)) / (WIDTH*HEIGHT)
            if area < .12:
                raise ValueError('匹配纹理覆盖太小，请固定相机并拍摄分布更广的纹理')
        q1 = cv2.perspectiveTransform(p1[:, None, :], transforms[0])[:, 0]
        q2 = cv2.perspectiveTransform(p2[:, None, :], transforms[1])[:, 0]
        error = float(np.percentile(np.abs(q1[:, 1]-q2[:, 1]), 90))
        if not np.isfinite(error) or error > 2.0:
            raise ValueError('独立复核的对齐误差超过 2 像素')
        return error

    @classmethod
    def verify(cls, left, right, transforms):
        p1, p2 = cls.correspondences(left, right)
        return cls.check_geometry(p1, p2, transforms)

    def align(self, left, right):
        p1, p2 = self.correspondences(left, right)
        f, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.2, .999)
        inliers = 0 if mask is None else int(mask.sum())
        if (f is None or f.shape != (3, 3) or mask is None or inliers < 22
                or inliers / len(p1) < .60):
            raise ValueError('无法可靠估计双目几何，请固定相机并改变观察场景')
        inlier_mask = mask.ravel() != 0
        ip1, ip2 = p1[inlier_mask], p2[inlier_mask]
        for p in (ip1, ip2):
            area = cv2.contourArea(cv2.convexHull(p)) / (WIDTH*HEIGHT)
            if area < .12:
                raise ValueError('匹配纹理覆盖太小，请固定相机并拍摄分布更广的纹理')
        ok, h1, h2 = cv2.stereoRectifyUncalibrated(ip1, ip2, f, (WIDTH, HEIGHT))
        if not ok or not np.isfinite(h1).all() or not np.isfinite(h2).all():
            raise ValueError('图像对齐失败')
        error = self.check_geometry(p1, p2, (h1, h2))
        ones = np.full((HEIGHT, WIDTH), 255, np.uint8)
        masks = [cv2.warpPerspective(ones, h, (WIDTH, HEIGHT)) > 250 for h in (h1, h2)]
        if (masks[0] & masks[1]).mean() < .35:
            raise ValueError('校正后共同视野太小')
        q1 = cv2.perspectiveTransform(ip1[:, None, :], h1)[:, 0]
        q2 = cv2.perspectiveTransform(ip2[:, None, :], h2)[:, 0]
        ds = q1[:, 0]-q2[:, 0]
        lo = int(np.floor(np.percentile(ds, 2)/16)*16)-16
        hi = int(np.ceil(np.percentile(ds, 98)/16)*16)+16
        span = max(64, int(np.ceil((hi-lo)/16))*16)
        if span > 256 or abs(lo) > WIDTH//2:
            raise ValueError('视差范围异常，请调整相机朝向')
        args = dict(numDisparities=span, blockSize=5, P1=200, P2=800,
                    disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=80,
                    speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.a = cv2.StereoSGBM_create(minDisparity=lo, **args)
        self.rlo = -lo-span+1
        self.b = cv2.StereoSGBM_create(minDisparity=self.rlo, **args)
        self.lo, self.span = lo, span
        self.transforms, self.masks = (h1, h2), masks
        self.detail = {'alignment_inliers':inliers, 'alignment_p90_px':round(error, 2),
                       'disparity_min_px':lo, 'disparity_max_px':lo+span}
        return self.detail

    def compute(self, left, right):
        l, r = [cv2.warpPerspective(im, h, (WIDTH, HEIGHT))
                for im, h in zip((left, right), self.transforms)]
        dl = self.a.compute(l, r).astype(np.float32)/16
        dr = self.b.compute(r, l).astype(np.float32)/16
        yy, xx = np.indices(dl.shape)
        xr = np.rint(xx-dl).astype(np.int32)
        inside = (xr >= 0) & (xr < WIDTH)
        safe = xr.clip(0, WIDTH-1)
        valid = inside & (dl > self.lo-1) & (dr[yy, safe] > self.rlo-1)
        valid &= np.abs(dl+dr[yy, safe]) <= 1.5
        valid &= self.masks[0] & self.masks[1][yy, safe]
        valid &= cv2.boxFilter(cv2.Laplacian(l, cv2.CV_32F)**2, -1, (5,5)) > 20
        shade = np.clip((dl-self.lo)/self.span*255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        return color, float(valid.mean())


class Live:
    def __init__(self, left, right):
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GstVideo', '1.0')
        from gi.repository import Gst, GstVideo
        self.Gst = Gst
        self.GstVideo = GstVideo
        Gst.init(None)
        cv2.setNumThreads(4)
        self.devices = (left, right)
        self.stopped = threading.Event()
        self.store = CaptureStore(Path(__file__).with_name('captures'))
        self.capture_error = None
        self.preview_jpeg = None
        self.preview_seq = 0
        self.preview_pair = None
        self.preview_until = time.monotonic() + 5
        self.depth_until = 0.0
        self.condition = threading.Condition()
        self.frames = [deque(maxlen=6), deque(maxlen=6)]
        self.matcher = Matcher()
        calibration_path = Path(__file__).with_name('calibration.npz')
        self.metric = MetricDepth(calibration_path) if calibration_path.exists() else None
        self.realign = threading.Event()
        self.realign.set()
        self.jpeg = None
        self.jpeg_fixed = None
        self.seq = 0
        self.output_times = deque(maxlen=30)
        self.updated = None
        self.state = dict(mode='waiting', message='正在启动两路相机', fps=None,
                          pair_skew_ms=None, valid_fraction=None, frames=0,
                          auto_alignment=True, alignment_state='collecting',
                          alignment_message='正在等待共同纹理以估计未标定校正',
                          alignment_checks=0, alignment_p90_px=None,
                          mapping='未启动：缺少双目标定，不能生成米制地图',
                          sync='软件时间戳配对；未经硬件曝光同步验证')
        if self.metric is not None:
            self.state.update(auto_alignment=False, alignment_state='calibrated',
                              calibration_id=self.metric.identifier, baseline_mm=round(self.metric.baseline_mm, 3),
                              depth_unit='m', depth_range_m=list(self.metric.RANGE_M),
                              alignment_message='采用已验收的固定双目标定参数',
                              mapping='深度试验已就绪；实际测距精度待验证，尚未启动建图')
        branches = []
        for i, device in enumerate((left, right)):
            branches.append(f'v4l2src device={device} ! video/x-raw,format=NV12,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT} '
                            f'! appsink name=sink{i} '
                            'max-buffers=2 drop=true sync=false')
        self.pipeline = Gst.parse_launch(' '.join(branches))
        self.pipeline.set_state(Gst.State.PLAYING)
        for i in range(2):
            threading.Thread(target=self.capture, args=(i,), daemon=True).start()
        threading.Thread(target=self.preview, daemon=True).start()
        threading.Thread(target=self.work, daemon=True).start()

    def capture(self, index):
        gst = self.Gst
        sink = self.pipeline.get_by_name(f'sink{index}')
        video_info = None
        while not self.stopped.is_set():
            sample = sink.emit('try-pull-sample', gst.SECOND)
            if sample is None:
                continue
            if video_info is None:
                video_info = self.GstVideo.VideoInfo()
                if not video_info.from_caps(sample.get_caps()):
                    self.capture_error = '无法读取相机图像布局'
                    return
            buf = sample.get_buffer()
            ok, info = buf.map(gst.MapFlags.READ)
            if not ok:
                continue
            try:
                # Preserve the sensor's NV12 luma plane; no expensive colour conversion.
                meta = self.GstVideo.buffer_get_video_meta(buf)
                stride = meta.stride[0] if meta else video_info.stride[0]
                offset = meta.offset[0] if meta else video_info.offset[0]
                frame = np.ndarray((CAPTURE_HEIGHT, CAPTURE_WIDTH), np.uint8,
                                   buffer=info.data, offset=offset, strides=(stride, 1)).copy()
                stamp = sample.get_segment().to_running_time(gst.Format.TIME, buf.pts)
                if stamp != gst.CLOCK_TIME_NONE:
                    with self.condition:
                        self.frames[index].append((stamp, frame, time.monotonic()))
                        self.condition.notify_all()
            finally:
                buf.unmap(info)

    def select_pair(self, require_sync=True):
        """Caller holds condition. Never return a disconnected camera's old frame."""
        now = time.monotonic()
        if not self.frames[0] or not self.frames[1]:
            return None
        for a in reversed(self.frames[0]):
            if now-a[2] > MAX_FRAME_AGE:
                break
            fresh = [b for b in self.frames[1] if now-b[2] <= MAX_FRAME_AGE]
            if not fresh:
                return None
            b = min(fresh, key=lambda item: abs(item[0]-a[0]))
            if not require_sync or abs(a[0]-b[0]) <= MAX_SKEW_NS:
                return a, b
        return None

    def preview(self):
        last = None
        while not self.stopped.is_set():
            started = time.monotonic()
            if started > self.preview_until:
                self.stopped.wait(.2)
                continue
            with self.condition:
                pair = self.select_pair(require_sync=False)
            if pair is not None:
                stamps = (pair[0][0], pair[1][0])
                if stamps != last:
                    try:
                        canvas = np.hstack([cv2.resize(item[1], (960, 540), interpolation=cv2.INTER_AREA) for item in pair])
                        ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            with self.condition:
                                self.preview_jpeg = encoded.tobytes()
                                self.preview_pair = pair
                                self.preview_seq += 1
                                self.condition.notify_all()
                            last = stamps
                    except Exception as exc:
                        self.capture_error = '预览处理异常：'+str(exc)
            self.stopped.wait(max(.01, .2-(time.monotonic()-started)))

    def take_photo(self):
        if not self.store.lock.acquire(blocking=False):
            raise ValueError('正在保存上一组照片，请稍候')
        try:
            with self.condition:
                pair = self.select_pair()
            if pair is None:
                raise ValueError('尚无新鲜的双路匹配图像，请等待画面恢复')
            metadata = self.store.save(pair, self.devices)
            self.capture_error = None
            return metadata
        except Exception as exc:
            self.capture_error = str(exc)
            raise
        finally:
            self.store.lock.release()

    def publish(self, color, fixed_color=None, **state):
        ok, encoded = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 85])
        fixed = encoded
        if fixed_color is not None:
            fixed_ok, fixed = cv2.imencode('.jpg', fixed_color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok = ok and fixed_ok
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.jpeg_fixed = fixed.tobytes()
                self.seq += 1
                self.updated = time.monotonic()
                self.state.update(state, frames=self.seq)
                self.condition.notify_all()

    def work(self):
        if self.metric is not None:
            return self.metric_work()
        last_stamp, retry_at = None, 0
        retry_delay = 1.0
        pending = None
        pending_stamp = None
        verify_count = 0
        last_verify = 0.0
        last_check_stamp = None

        def waiting(message, alignment_state='collecting', checks=0, p90=None):
            fields = dict(mode='waiting', message=message, valid_fraction=None,
                          fps=None, alignment_state=alignment_state,
                          alignment_message=message, alignment_checks=checks,
                          alignment_p90_px=p90)
            self.publish(np.zeros((HEIGHT, WIDTH, 3), np.uint8), **fields)

        def reset(message):
            nonlocal pending, pending_stamp, verify_count, last_verify, last_check_stamp
            self.matcher = Matcher()
            pending = None
            pending_stamp = None
            verify_count = 0
            last_verify = 0.0
            last_check_stamp = None
            self.output_times.clear()
            self.state.update(alignment_inliers=None, disparity_min_px=None, disparity_max_px=None)
            waiting(message)

        while not self.stopped.is_set():
            started = time.monotonic()
            if started > self.depth_until:
                self.stopped.wait(.2)
                continue
            a = b = None
            with self.condition:
                pair = self.select_pair()
                if pair and (last_stamp is None or pair[0][0] > last_stamp):
                    a, b = pair
            if a is None:
                if self.updated is not None and started-self.updated > 1:
                    self.state.update(mode='waiting', message='等待时间戳相近的左右图像')
                time.sleep(.02)
                continue
            last_stamp = a[0]
            skew = abs(a[0]-b[0])/1e6
            a = (a[0], cv2.resize(a[1], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            b = (b[0], cv2.resize(b[1], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
            self.state['pair_skew_ms'] = round(skew, 2)
            try:
                if self.realign.is_set():
                    self.realign.clear()
                    retry_at = 0
                    retry_delay = 1.0
                    reset('正在自动估计未标定校正')
                if self.matcher.transforms is None and pending is None:
                    if started < retry_at:
                        time.sleep(.1)
                        continue
                    try:
                        candidate = Matcher()
                        candidate.align(a[1], b[1])
                        pending, pending_stamp = candidate, (a[0], b[0])
                        verify_count = 0
                        last_check_stamp = None
                        retry_delay = 1.0
                        waiting('新候选已生成，等待两次独立画面复核', 'verifying')
                    except (ValueError, cv2.error) as exc:
                        retry_at = started + retry_delay
                        retry_delay = min(8.0, retry_delay * 2)
                        waiting(str(exc))
                        continue

                if pending is not None:
                    stamps = (a[0], b[0])
                    reference = last_check_stamp or pending_stamp
                    distinct = (a[0]-reference[0] >= .4e9 and b[0]-reference[1] >= .4e9)
                    if distinct:
                        try:
                            error = Matcher.verify(a[1], b[1], pending.transforms)
                            verify_count += 1
                            last_check_stamp = stamps
                            if verify_count >= 2:
                                self.matcher = pending
                                pending = None
                                pending_stamp = None
                                last_verify = started
                                self.state.update(self.matcher.detail)
                                waiting('候选通过两次独立复核，开始跟踪；未标定相对视差',
                                        'tracking', verify_count, round(error, 2))
                            else:
                                waiting('候选通过首次复核，等待另一帧再次确认',
                                        'verifying', verify_count, round(error, 2))
                        except (ValueError, cv2.error) as exc:
                            pending = None
                            pending_stamp = None
                            verify_count = 0
                            retry_at = started + retry_delay
                            retry_delay = min(8.0, retry_delay * 2)
                            waiting(f'候选复核未通过：{exc}')
                    else:
                        waiting('等待与候选不同且间隔足够的画面复核', 'verifying', verify_count,
                                self.state.get('alignment_p90_px'))
                    if self.matcher.transforms is None:
                        continue

                if started-last_verify >= 2.5:
                    try:
                        error = Matcher.verify(a[1], b[1], self.matcher.transforms)
                        last_verify = time.monotonic()
                        verify_count += 1
                        self.state.update(alignment_state='tracking',
                                          alignment_message='自动未标定校正持续通过独立复核',
                                          alignment_checks=verify_count,
                                          alignment_p90_px=round(error, 2))
                    except (ValueError, cv2.error) as exc:
                        reset(f'跟踪复核失败，已停用旧校正：{exc}')
                        retry_at = started + retry_delay
                        retry_delay = min(8.0, retry_delay * 2)
                        continue

                color, fraction = self.matcher.compute(a[1], b[1])
                now = time.monotonic()
                self.output_times.append(now)
                rate = ((len(self.output_times)-1)/(now-self.output_times[0])) if len(self.output_times)>1 else None
                message = ('有效匹配较少，请固定双目并朝向有纹理、明暗适中的共同场景；黑色为无效区域'
                           if fraction < .1 else '未标定相对视差，仅用于匹配试验；黑色为无效区域')
                self.publish(color, mode='uncalibrated',
                             message=message, valid_fraction=fraction,
                             fps=None if rate is None else round(rate, 1),
                             alignment_state='tracking',
                             alignment_message='自动未标定校正持续通过独立复核')
            except (ValueError, cv2.error) as exc:
                reset(str(exc))
                retry_at = started + retry_delay
                retry_delay = min(8.0, retry_delay * 2)
            except Exception as exc:
                reset(f'采集处理异常：{exc}')
                self.state.update(mode='error')
            time.sleep(max(.005, .125-(time.monotonic()-started)))

    def metric_work(self):
        last_stamp = None
        blank = np.zeros((self.metric.height, self.metric.width, 3), np.uint8)
        while not self.stopped.is_set():
            started = time.monotonic()
            if started > self.depth_until:
                self.output_times.clear()
                self.stopped.wait(.2)
                continue
            with self.condition:
                pair = self.select_pair()
            if pair is None or pair[0][0] == last_stamp:
                if self.updated is not None and started-self.updated > 1:
                    self.state.update(mode='waiting', message='等待新鲜且时间戳相近的左右图像', depth_median_m=None)
                self.stopped.wait(.02)
                continue
            last_stamp = pair[0][0]
            try:
                color, depth, details = self.metric.compute(pair[0][1], pair[1][1])
                fixed_color = self.metric.colorize(depth, self.metric.RANGE_M)
                now = time.monotonic()
                source_age = max(now-item[2] for item in pair)
                if source_age > 1.5:
                    raise ValueError('深度处理延迟过大，等待新画面')
                self.output_times.append(now)
                rate = (len(self.output_times)-1)/(now-self.output_times[0]) if len(self.output_times) > 1 else None
                self.publish(color, fixed_color=fixed_color, **details, mode='calibrated', fps=round(rate, 1) if rate else None,
                             pair_skew_ms=round(abs(pair[0][0]-pair[1][0])/1e6, 2),
                             depth_source_age_ms=round(source_age*1000), alignment_state='calibrated',
                             message='已标定米制深度；黑色为无有效深度，实际距离精度待实测',
                             alignment_message='固定标定校正；请勿改变两镜头相对位置')
            except Exception as exc:
                self.publish(blank, mode='error', message='深度计算失败：'+str(exc),
                             valid_fraction=None, depth_median_m=None, fps=None)
            self.stopped.wait(max(.005, .1-(time.monotonic()-started)))

    def status(self):
        captures = self.store.entries()
        with self.condition:
            now = time.monotonic()
            pair = self.select_pair()
            preview_age = None if self.preview_pair is None else max(now-item[2] for item in self.preview_pair)*1000
            return dict(self.state, age_ms=None if self.updated is None else round((now-self.updated)*1000),
                        capture_ready=pair is not None and preview_age is not None and preview_age <= 1500 and not self.store.lock.locked(),
                        preview_age_ms=None if preview_age is None else round(preview_age),
                        capture_skew_ms=None if pair is None else round(abs(pair[0][0]-pair[1][0])/1e6, 2),
                        camera_age_ms=[None if not frames else round((now-frames[-1][2])*1000) for frames in self.frames],
                        camera_pts_ns=[[int(frame[0]) for frame in frames] for frames in self.frames],
                        capture_count=len(captures), capture_busy=self.store.lock.locked(),
                        capture_error=self.capture_error, capture_width=CAPTURE_WIDTH,
                        capture_height=CAPTURE_HEIGHT, capture_directory=str(self.store.directory),
                        calibration_target=CAPTURE_TARGET.copy(),
                        last_capture=captures[-1] if captures else None)


def serve(args):
    live = Live(args.left, args.right)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, content, kind='application/json; charset=utf-8', status=200):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def json_response(self, data, status=200):
            return self.send(json.dumps(data, ensure_ascii=False).encode(), status=status)

        def do_GET(self):
            path = urlsplit(self.path).path
            fixed_scale = parse_qs(urlsplit(self.path).query).get('scale', ['auto'])[0] == 'fixed'
            if path == '/':
                return self.send(Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/status':
                status = live.status()
                if live.metric is not None:
                    status['depth_color_mode'] = 'fixed' if fixed_scale else 'auto'
                    if fixed_scale:
                        status['depth_color_range_m'] = list(live.metric.RANGE_M)
                return self.json_response(status)
            if path in ('/stereo.jpg', '/depth.jpg'):
                stereo = path == '/stereo.jpg'
                if stereo:
                    live.preview_until = time.monotonic() + 5
                else:
                    live.depth_until = time.monotonic() + 5
                with live.condition:
                    jpeg = live.preview_jpeg if stereo else (live.jpeg_fixed if fixed_scale else live.jpeg)
                if jpeg is None:
                    return self.json_response({'message': '图像尚未就绪'}, 503)
                return self.send(jpeg, 'image/jpeg')
            if path == '/captures':
                return self.json_response({'captures':live.store.entries()})
            if path == '/captures.zip':
                entries = live.store.entries()
                if not entries:
                    return self.json_response({'message':'还没有照片，请先拍摄一组'}, 404)
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', 'attachment; filename="stereo-captures.zip"')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    with zipfile.ZipFile(self.wfile, 'w', compression=zipfile.ZIP_STORED) as archive:
                        for entry in entries:
                            for name in ('left.png', 'right.png', 'meta.json'):
                                archive.write(live.store.directory/entry['id']/name, entry['id']+'/'+name)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            match = re.fullmatch(r'/captures/(\d{8}T\d{12}Z-[a-f0-9]{8})/(left|right)\.png', path)
            if match:
                file = live.store.directory/match[1]/(match[2]+'.png')
                if not (file.parent/'meta.json').is_file() or not file.is_file():
                    return self.send_error(404)
                return self.send(file.read_bytes(), 'image/png')
            if path not in ('/depth.mjpg', '/stereo.mjpg'):
                return self.send_error(404)
            stereo = path == '/stereo.mjpg'
            if stereo:
                live.preview_until = time.monotonic()+5
            else:
                live.depth_until = time.monotonic()+5
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            seq = -1
            try:
                while not live.stopped.is_set():
                    if stereo:
                        live.preview_until = time.monotonic()+5
                    else:
                        live.depth_until = time.monotonic()+5
                    with live.condition:
                        live.condition.wait_for(lambda: (live.preview_seq if stereo else live.seq) != seq or live.stopped.is_set(), timeout=2)
                        current = live.preview_seq if stereo else live.seq
                        if current == seq:
                            continue
                        seq = current
                        jpeg = live.preview_jpeg if stereo else live.jpeg
                    if jpeg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            origin = self.headers.get('Origin')
            if (self.headers.get('Sec-Fetch-Site') == 'cross-site' or
                    (origin and origin not in ('http://'+self.headers.get('Host', ''), 'https://'+self.headers.get('Host', '')))):
                return self.json_response({'message':'请从本机拍照界面操作'}, 403)
            path = urlsplit(self.path).path
            if path == '/capture':
                try:
                    capture = live.take_photo()
                    return self.json_response({'message':'左右原图已成对保存', 'capture':capture,
                                               'capture_count':len(live.store.entries())}, 201)
                except ValueError as exc:
                    return self.json_response({'message':str(exc)}, 409)
                except Exception as exc:
                    return self.json_response({'message':'保存失败：'+str(exc)}, 500)
            if path != '/rectify':
                return self.send_error(404)
            if live.metric is not None:
                return self.json_response({'message':'已启用固定双目标定，不使用临时自动对齐；若镜头移动需重新标定'}, 409)
            live.realign.set()
            self.send(json.dumps({'message':'正在重新估计对齐，请保持两只相机相对位置固定'}, ensure_ascii=False).encode())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    def stop(signum, frame):
        live.stopped.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f'Stereo trial listening on {args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        live.stopped.set()
        with live.condition:
            live.condition.notify_all()
        live.pipeline.set_state(live.Gst.State.NULL)
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='/dev/video33')
    parser.add_argument('--right', default='/dev/video42')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--host', default='127.0.0.1')
    serve(parser.parse_args())
