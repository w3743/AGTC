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
from lightstereo_depth import MetricDepth

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
        self.store = CaptureStore(Path(__file__).resolve().parent.parent / 'captures')
        self.capture_error = None
        self.preview_jpeg = None
        self.preview_seq = 0
        self.preview_pair = None
        self.preview_until = time.monotonic() + 5
        self.depth_until = 0.0
        self.condition = threading.Condition()
        self.frames = [deque(maxlen=6), deque(maxlen=6)]
        calibration_path = Path(__file__).with_name('calibration.npz')
        self.metric = MetricDepth(calibration_path)
        self.realign = threading.Event()
        self.realign.set()
        self.jpeg = None
        self.jpeg_fixed = None
        self.seq = 0
        self.output_times = deque(maxlen=30)
        self.updated = None
        self.state = dict(algorithm='LightStereo FP16 NPU', mode='waiting', message='正在启动两路相机', fps=None,
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
        self.depth_thread = threading.Thread(target=self.metric_work, daemon=True)
        self.depth_thread.start()

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
    print(f'LightStereo listening on {args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        live.stopped.set()
        with live.condition:
            live.condition.notify_all()
        live.pipeline.set_state(live.Gst.State.NULL)
        live.depth_thread.join(timeout=5)
        if not live.depth_thread.is_alive():
            live.metric.close()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='/dev/video33')
    parser.add_argument('--right', default='/dev/video42')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--host', default='127.0.0.1')
    serve(parser.parse_args())
