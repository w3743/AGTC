#!/usr/bin/env python3
"""On-board dual-camera capture and calibrated pseudocolor depth."""
import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
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
from fast_depth import FastSGBMDepth
from lightstereo_depth import MetricDepth as NpuMetricDepth
from pointcloud import PointCloudStream
from realtime import HardwareSampler
from hardware_video import HardwareVideo
from calibration_live import CalibrationLive
from depth_wire import (CONVENTION as DEPTH_WIRE_CONVENTION,
                        FRAME_ID as DEPTH_WIRE_FRAME_ID,
                        MAX_MM as DEPTH_WIRE_MAX_MM,
                        MIN_MM as DEPTH_WIRE_MIN_MM,
                        SCHEMA as DEPTH_WIRE_SCHEMA,
                        SOURCE_HEIGHT as DEPTH_WIRE_HEIGHT,
                        SOURCE_WIDTH as DEPTH_WIRE_WIDTH,
                        build_depth_record, encode_depth_payload,
                        fast_geometry_descriptor, parse_requested_sequence,
                        select_depth_record)

WIDTH, HEIGHT = 640, 360
CAPTURE_WIDTH, CAPTURE_HEIGHT = 3840, 2160
PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 360
PREVIEW_INTERVAL = 1.0 / 30.0
PREVIEW_QUALITY = max(35, min(90, int(os.environ.get("LIGHTSTEREO_PREVIEW_QUALITY", "55"))))
MAX_SKEW_NS = 25_000_000
MAX_FRAME_AGE = .75
CAPTURE_TARGET = dict(
    id='screen-checkerboard-10x7', inner_corners=[9, 6], square_size_mm=24.2,
    measurement='User measured five cells horizontally and vertically: 121 mm each',
    source='User-selected target; not automatically detected',
)


def make_metric_engine(calibration_path, input_size):
    mode = os.environ.get("LIGHTSTEREO_DEPTH_MODE", "fast").strip().lower()
    if mode in ("fast", "sgbm"):
        return FastSGBMDepth(calibration_path, input_size=input_size)
    if mode in ("npu", "lightstereo"):
        metric = NpuMetricDepth(calibration_path)
        metric.algorithm = "LightStereo FP16 NPU (dual direction)"
        return metric
    raise ValueError("LIGHTSTEREO_DEPTH_MODE must be fast or npu")


def monotonic_clock_domain(session_id):
    """Identify CLOCK_MONOTONIC's boot domain; fallback is process-session scoped."""
    try:
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
        if re.fullmatch(r'[A-Za-z0-9._:-]{1,128}', boot_id):
            return 'linux-boot:' + boot_id + ':CLOCK_MONOTONIC'
    except (OSError, UnicodeError):
        pass
    return 'session:' + session_id + ':CLOCK_MONOTONIC'


def rotate_pipeline_epoch(live):
    """Caller holds live.condition; invalidate frames and in-flight old work."""
    live.pipeline_epoch += 1
    live.measure_history.clear()
    live.frames = [deque(maxlen=6), deque(maxlen=6)]
    live.measure_input = None
    live.pointcloud_input = None
    live.condition.notify_all()


def activate_capture_pipeline(live, generation_stop):
    """Publish a new epoch only after its GStreamer pipeline is playing."""
    with live.condition:
        live.capture_pipeline_stop = generation_stop
        rotate_pipeline_epoch(live)
        live.capture_running = True


def invalidate_capture_pipeline(live):
    """Stop accepting old-pipeline frames before teardown begins."""
    with live.condition:
        live.capture_running = False
        rotate_pipeline_epoch(live)


def depth_work_is_current(live, metric, calibration_generation, pipeline_epoch):
    return (not live.calibration_active
            and metric is live.metric
            and calibration_generation == live.calibration_generation
            and pipeline_epoch == live.pipeline_epoch)


def append_capture_frame(live, index, generation_stop, frame):
    """Append only if this sample belongs to the currently running pipeline."""
    with live.condition:
        if (not live.capture_running or live.stopped.is_set()
                or generation_stop.is_set()
                or generation_stop is not live.capture_pipeline_stop):
            return False
        live.frames[index].append(frame)
        live.condition.notify_all()
        return True


def removed_feature_response(handler):
    return handler.json_response({
        'ok': False,
        'message': '独立远距、NPU预览、测距标注与点云接口已移除；请使用统一深度图和 /depth.bin 数值接口',
    }, 410)


class DisabledNpuPreviewState:
    """No-worker compatibility state for CalibrationLive's demand reset."""
    def __init__(self):
        self.until = 0.0


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
        cv2.setNumThreads(2)
        self.devices = (left, right)
        self.stopped = threading.Event()
        self.capture_control_lock = threading.RLock()
        self.publish_lock = threading.Lock()
        self.capture_running = False
        self.calibration_active = False
        self.calibration_generation = 0
        self.process_session_id = uuid4().hex
        self.monotonic_clock_id = monotonic_clock_domain(self.process_session_id)
        self.pipeline_epoch = 0
        self.hardware_video = None
        self.hardware_video_error = None
        configured_mode = os.environ.get("LIGHTSTEREO_DEPTH_MODE", "fast").strip().lower()
        if configured_mode in ("fast", "sgbm"):
            self.stream_width = int(os.environ.get("LIGHTSTEREO_LIVE_WIDTH", "640"))
            self.stream_height = int(os.environ.get("LIGHTSTEREO_LIVE_HEIGHT", "360"))
        else:
            self.stream_width, self.stream_height = CAPTURE_WIDTH, CAPTURE_HEIGHT
        if (self.stream_width < 32 or self.stream_height < 32
                or self.stream_width > CAPTURE_WIDTH or self.stream_height > CAPTURE_HEIGHT
                or self.stream_width % 8 or self.stream_height % 8):
            raise ValueError("Live NV12 size must be <= 3840x2160 and divisible by 8")
        self.store = CaptureStore(Path(__file__).resolve().parent.parent / 'captures')
        self.capture_error = None
        self.preview_jpeg = None
        self.preview_seq = 0
        self.preview_pair = None
        self.preview_times = deque(maxlen=30)
        self.preview_stats = dict(fps=0.0, resize_ms=None, jpeg_ms=None,
                                  jpeg_bytes=0, resolution=None)
        self.preview_until = time.monotonic() + 5
        self.depth_until = 0.0
        self.condition = threading.Condition()
        self.frames = [deque(maxlen=6), deque(maxlen=6)]
        calibration_path = Path(__file__).with_name('calibration.npz')
        self.calibration_path = calibration_path
        self.metric = make_metric_engine(calibration_path,
                                         (self.stream_width, self.stream_height))
        self.depth_wire_geometry = None
        self.depth_wire_error = '数值深度接口仅支持当前已标定的 131×240 快模式'
        if isinstance(self.metric, FastSGBMDepth):
            try:
                self.depth_wire_geometry = fast_geometry_descriptor(self.metric, calibration_path)
                self.depth_wire_error = None
            except Exception as exc:
                # Keep the preview alive if this profile cannot be proven to
                # match the current source calibration.
                self.depth_wire_error = str(exc)
        self.pointcloud_stream = None
        self.pointcloud_error = None
        try:
            self.pointcloud_stream = PointCloudStream(calibration_path)
        except Exception as exc:
            self.pointcloud_error = str(exc)
        self.measure_jpeg = None
        self.measure_input = None
        self.pointcloud_packet = None
        self.pointcloud_input = None
        self.measure_history = deque(maxlen=180)
        self.hardware_sampler = HardwareSampler()
        self.hardware_lock = threading.Lock()
        self.realign = threading.Event()
        self.realign.set()
        self.jpeg = None
        self.jpeg_fixed = None
        self.seq = 0
        self.output_times = deque(maxlen=30)
        self.updated = None
        self.state = dict(algorithm=getattr(self.metric, 'algorithm', 'LightStereo FP16 NPU'),
                          depth_mode=os.environ.get("LIGHTSTEREO_DEPTH_MODE", "fast"),
                          mode='waiting', message='正在启动两路相机', fps=None,
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
        self.state['pointcloud_calibration_id'] = (
            self.pointcloud_stream.calibration_id if self.pointcloud_stream else None)
        self.state['pointcloud_error'] = self.pointcloud_error
        self._start_capture_pipeline(self.stream_width, self.stream_height)
        threading.Thread(target=self.preview, daemon=True).start()
        self.depth_thread = threading.Thread(target=self.metric_work, daemon=True)
        self.depth_thread.start()
        self.npu_preview = DisabledNpuPreviewState()
        self.calibration = CalibrationLive(self)

    def _start_capture_pipeline(self, width, height):
        with self.condition:
            self.capture_running = False
        branches = []
        # Live mode negotiates its requested NV12 size directly on the camera
        # nodes (640x360 by default). Separate 4K MPP streams require opt-in.
        native = (os.environ.get('LIGHTSTEREO_4K', '0') == '1'
                  and (width, height) == (640, 360)
                  and self.devices == ('/dev/video33', '/dev/video42'))
        video = HardwareVideo(self.Gst, os.environ.get('LIGHTSTEREO_4K_BPS', '10000000')) if native else None
        for index, device in enumerate(self.devices):
            capture_device = ('/dev/video34', '/dev/video43')[index] if native else device
            branches.append(
                f'v4l2src device={capture_device} ! '
                f'video/x-raw,format=NV12,width={width},height={height},framerate=30/1 '
                f'! appsink name=sink{index} max-buffers=2 drop=true sync=false')
            if video:
                branches.append(video.branch(device, index))
        pipeline = None
        try:
            pipeline = self.Gst.parse_launch(' '.join(branches))
            result = pipeline.set_state(self.Gst.State.PLAYING)
            if result == self.Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f'无法启动 NV12 摄像头管线 {width}×{height}')
        except Exception:
            if pipeline is not None:
                pipeline.set_state(self.Gst.State.NULL)
            raise
        self.pipeline = pipeline
        self.hardware_video = video
        if video:
            video.start(pipeline)
        self.stream_width, self.stream_height = int(width), int(height)
        self.capture_pipeline_stop = threading.Event()
        activate_capture_pipeline(self, self.capture_pipeline_stop)
        self.capture_threads = []
        for index in range(2):
            thread = threading.Thread(target=self.capture,
                                      args=(index, self.capture_pipeline_stop,
                                            self.stream_width, self.stream_height),
                                      daemon=True)
            thread.start()
            self.capture_threads.append(thread)

    def _stop_capture_pipeline(self):
        # Invalidate work already outside the lock before the old pipeline is torn down.
        invalidate_capture_pipeline(self)
        stop_event = getattr(self, 'capture_pipeline_stop', None)
        if stop_event is not None:
            stop_event.set()
        video, self.hardware_video = self.hardware_video, None
        if video:
            video.close()
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is not None:
            pipeline.set_state(self.Gst.State.NULL)
        for thread in getattr(self, 'capture_threads', []):
            thread.join(timeout=2.0)
        self.capture_threads = []

    def capture(self, index, generation_stop, width, height):
        gst = self.Gst
        sink = self.pipeline.get_by_name(f'sink{index}')
        video_info = None
        while not self.stopped.is_set() and not generation_stop.is_set():
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
                copy_started = time.perf_counter()
                frame = np.ndarray((height, width), np.uint8,
                                   buffer=info.data, offset=offset, strides=(stride, 1)).copy()
                copy_ms = (time.perf_counter() - copy_started) * 1000
                stamp = sample.get_segment().to_running_time(gst.Format.TIME, buf.pts)
                if stamp != gst.CLOCK_TIME_NONE:
                    append_capture_frame(
                        self, index, generation_stop,
                        (stamp, frame, time.monotonic(), copy_ms))
            finally:
                buf.unmap(info)

    def select_pair(self, require_sync=True):
        """Caller holds condition. Never return a disconnected camera's old frame."""
        if not self.capture_running:
            return None
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
                        resize_start = time.perf_counter()
                        eyes = [item[1] if item[1].shape == (PREVIEW_HEIGHT, PREVIEW_WIDTH)
                                else cv2.resize(item[1], (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                                                interpolation=cv2.INTER_LINEAR)
                                for item in pair]
                        canvas = np.hstack(eyes)
                        resize_ms = (time.perf_counter() - resize_start) * 1000
                        jpeg_start = time.perf_counter()
                        ok, encoded = cv2.imencode('.jpg', canvas,
                                                    [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
                        jpeg_ms = (time.perf_counter() - jpeg_start) * 1000
                        if ok:
                            with self.condition:
                                self.preview_jpeg = encoded.tobytes()
                                self.preview_pair = pair
                                self.preview_seq += 1
                                self.preview_times.append(time.monotonic())
                                fps = ((len(self.preview_times)-1) /
                                       (self.preview_times[-1]-self.preview_times[0])
                                       if len(self.preview_times) > 1 else 0.0)
                                self.preview_stats = dict(fps=round(fps, 2),
                                                          resize_ms=round(resize_ms, 2),
                                                          jpeg_ms=round(jpeg_ms, 2),
                                                          jpeg_bytes=len(self.preview_jpeg),
                                                          resolution=[canvas.shape[1], canvas.shape[0]])
                                self.condition.notify_all()
                            last = stamps
                    except Exception as exc:
                        self.capture_error = '预览处理异常：'+str(exc)
            self.stopped.wait(max(.001, PREVIEW_INTERVAL-(time.monotonic()-started)))

    def take_photo(self):
        if self.calibration_active:
            raise ValueError('校准中，请使用校准界面的采集按钮')
        if not self.store.lock.acquire(blocking=False):
            raise ValueError('正在保存上一组照片，请稍候')
        previous_size = None
        try:
            with self.capture_control_lock:
                current_size = (self.stream_width, self.stream_height)
                if current_size != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
                    previous_size = current_size
                    self._stop_capture_pipeline()
                    # The new GStreamer clock epoch restarts at zero; older PTS
                    # values cannot identify duplicate frames across the switch.
                    self.store.last_stamps = None
                    self._start_capture_pipeline(CAPTURE_WIDTH, CAPTURE_HEIGHT)
            deadline = time.monotonic() + 6.0
            pair = None
            while time.monotonic() < deadline and not self.stopped.is_set():
                with self.condition:
                    candidate = self.select_pair()
                    if (candidate is not None and
                            all(frame[1].shape == (CAPTURE_HEIGHT, CAPTURE_WIDTH)
                                for frame in candidate)):
                        pair = candidate
                        break
                    self.condition.wait(timeout=.1)
            if pair is None:
                raise ValueError('高分辨率双目照片未在 6 秒内同步，请稍后重试')
            metadata = self.store.save(pair, self.devices)
            self.capture_error = None
            return metadata
        except Exception as exc:
            self.capture_error = str(exc)
            raise
        finally:
            try:
                if previous_size is not None and not self.stopped.is_set():
                    with self.capture_control_lock:
                        self._stop_capture_pipeline()
                        self._start_capture_pipeline(*previous_size)
            finally:
                self.store.lock.release()

    def publish(self, color, fixed_color=None, depth=None, rectified_left=None, **state):
        # JPEG/depth preparation happens without the shared state lock. This
        # separate lock serializes publishers so sequence reservation and every
        # sequence-indexed input/history update can be committed atomically.
        with self.publish_lock:
            return Live._publish_serialized(self, color, fixed_color, depth, rectified_left, **state)

    def _publish_serialized(self, color, fixed_color=None, depth=None, rectified_left=None, **state):
        generation = state.pop('worker_generation', self.calibration_generation)
        pipeline_epoch = state.pop('worker_pipeline_epoch', self.pipeline_epoch)
        worker_metric = state.pop('worker_metric', self.metric)
        depth_metadata_context = state.pop('depth_metadata_context', None)
        with self.condition:
            if not depth_work_is_current(self, worker_metric, generation, pipeline_epoch):
                return
        started = time.perf_counter()
        ok, encoded = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 85])
        fixed = encoded
        if fixed_color is not None:
            fixed_ok, fixed = cv2.imencode('.jpg', fixed_color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok = ok and fixed_ok
        state['jpeg_encode_ms'] = round((time.perf_counter() - started) * 1000, 2)
        state['jpeg_bytes'] = int(encoded.nbytes + fixed.nbytes) if ok else 0
        history = None
        wire_payload = None
        wire_width = wire_height = None
        millimetres = None
        if depth is not None and rectified_left is not None:
            if depth_metadata_context is not None:
                try:
                    wire_payload, wire_width, wire_height = encode_depth_payload(depth)
                    if (wire_width, wire_height) != (DEPTH_WIRE_WIDTH, DEPTH_WIRE_HEIGHT):
                        raise ValueError('depth output shape does not match the 131x240 wire profile')
                except (TypeError, ValueError) as exc:
                    wire_payload = None
                    state['depth_wire_error'] = str(exc)
            if wire_payload is None:
                millimetres = np.zeros(depth.shape, dtype=np.uint16)
                valid = np.isfinite(depth) & (depth >= .3) & (depth <= 5.0)
                millimetres[valid] = np.clip(np.rint(depth[valid] * 1000), 1, 5000).astype(np.uint16)
        state['depth_publish_ms'] = round((time.perf_counter() - started) * 1000, 2)
        if isinstance(state.get('stage_ms'), dict):
            state['stage_ms'].update(jpeg_encode=state['jpeg_encode_ms'])
        if ok:
            with self.condition:
                if not depth_work_is_current(self, worker_metric, generation, pipeline_epoch):
                    return
                # Sequence and all sequence-indexed values are bound together
                # only after the final generation/epoch check, under this lock.
                next_sequence = self.seq + 1
                if wire_payload is not None and depth_metadata_context is not None:
                    metadata = dict(depth_metadata_context)
                    metadata.update(
                        schema=DEPTH_WIRE_SCHEMA,
                        seq=next_sequence,
                        width=wire_width,
                        height=wire_height,
                        encoding='16UC1',
                        byte_order='little',
                        unit='mm',
                        convention=DEPTH_WIRE_CONVENTION,
                        frame_id=DEPTH_WIRE_FRAME_ID,
                        min_mm=DEPTH_WIRE_MIN_MM,
                        max_mm=DEPTH_WIRE_MAX_MM,
                        publish_mono_ns=time.monotonic_ns(),
                        source_mono_ns=None,
                        clock_mapping_id=None,
                        clock_quality=None,
                        source_time_reference='left_pts')
                    try:
                        record = build_depth_record(
                            wire_payload, metadata,
                            expected_shape=(DEPTH_WIRE_HEIGHT, DEPTH_WIRE_WIDTH))
                        history = record.history_entry()
                        state.pop('depth_wire_error', None)
                    except (TypeError, ValueError, UnicodeError) as exc:
                        state['depth_wire_error'] = str(exc)
                        millimetres = np.zeros(depth.shape, dtype=np.uint16)
                        valid = np.isfinite(depth) & (depth >= .3) & (depth <= 5.0)
                        millimetres[valid] = np.clip(np.rint(depth[valid] * 1000), 1, 5000).astype(np.uint16)
                        history = (next_sequence, millimetres, state.get('source_pts_ns'))
                self.jpeg = encoded.tobytes()
                self.jpeg_fixed = fixed.tobytes()
                self.seq = next_sequence
                if depth is not None and rectified_left is not None:
                    if history is None and millimetres is not None:
                        history = (next_sequence, millimetres, state.get('source_pts_ns'))
                if history is not None:
                    self.measure_history.append(history)
                else:
                    self.measure_history.clear()
                self.updated = time.monotonic()
                self.state.update(state, frames=self.seq, depth_seq=self.seq)
                self.condition.notify_all()


    def metric_work(self):
        last_stamp = None
        last_epoch = None
        while not self.stopped.is_set():
            started = time.monotonic()
            with self.condition:
                active = (self.capture_running and not self.calibration_active
                          and started <= self.depth_until)
                generation = self.calibration_generation
                epoch = self.pipeline_epoch
                metric = self.metric
                geometry = self.depth_wire_geometry if metric is self.metric else None
                session_id = self.process_session_id
                monotonic_id = self.monotonic_clock_id
                pair = self.select_pair() if active else None
            if epoch != last_epoch:
                last_stamp = None
                self.output_times.clear()
                last_epoch = epoch
            if not active:
                self.output_times.clear()
                self.stopped.wait(.2)
                continue
            if metric is None:
                self.stopped.wait(.2)
                continue
            blank = np.zeros((metric.height, metric.width, 3), np.uint8)
            if pair is None or pair[0][0] == last_stamp:
                with self.condition:
                    if (not self.calibration_active and generation == self.calibration_generation
                            and epoch == self.pipeline_epoch and metric is self.metric
                            and self.updated is not None and started-self.updated > 1):
                        self.state.update(mode='waiting', message='等待新鲜且时间戳相近的左右图像', depth_median_m=None)
                self.stopped.wait(.02)
                continue
            last_stamp = pair[0][0]
            try:
                color, depth, details = metric.compute(pair[0][1], pair[1][1])
                fixed_color = metric.colorize(depth, metric.RANGE_M)
                rectified_left = getattr(metric, 'last_rectified_left', None)
                now = time.monotonic()
                source_age = max(now-item[2] for item in pair)
                if source_age > 1.5:
                    raise ValueError('深度处理延迟过大，等待新画面')
                self.output_times.append(now)
                rate = (len(self.output_times)-1)/(now-self.output_times[0]) if len(self.output_times) > 1 else None
                metadata_context = None
                if geometry is not None:
                    metadata_context = dict(
                        session_id=session_id,
                        pipeline_epoch=epoch,
                        width=geometry['width'],
                        height=geometry['height'],
                        K=list(geometry['K']),
                        calibration_id=geometry['calibration_id'],
                        extrinsics_id=geometry['extrinsics_id'],
                        preprocessing_id=geometry['preprocessing_id'],
                        frame_id=geometry['frame_id'],
                        monotonic_clock_id=monotonic_id,
                        left_pts_ns=int(pair[0][0]),
                        right_pts_ns=int(pair[1][0]),
                        left_arrival_mono_ns=int(round(pair[0][2] * 1_000_000_000)),
                        right_arrival_mono_ns=int(round(pair[1][2] * 1_000_000_000)))
                self.publish(color, fixed_color=fixed_color, depth=depth,
                             rectified_left=rectified_left, **details,
                             worker_generation=generation,
                              worker_pipeline_epoch=epoch,
                              worker_metric=metric,
                              depth_metadata_context=metadata_context,
                             mode='calibrated', fps=round(rate, 1) if rate else None,
                             source_pts_ns=int(pair[0][0]),
                             pair_skew_ms=round(abs(pair[0][0]-pair[1][0])/1e6, 2),
                             depth_source_age_ms=round(source_age*1000), alignment_state='calibrated',
                             message='已标定米制深度；黑色为无有效深度，实际距离精度待实测',
                             alignment_message='固定标定校正；请勿改变两镜头相对位置')
            except Exception as exc:
                self.publish(blank, worker_generation=generation,
                             worker_pipeline_epoch=epoch, worker_metric=metric,
                             mode='error', message='深度计算失败：'+str(exc),
                             valid_fraction=None, depth_median_m=None, fps=None)
            self.stopped.wait(max(.001, .002-(time.monotonic()-started)))

    def hardware(self):
        with self.hardware_lock:
            return self.hardware_sampler.snapshot(os.getpid())

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
                        camera_capture_fps=[
                            round((len(frames)-1)*1e9/(frames[-1][0]-frames[0][0]), 2)
                            if len(frames) > 1 and frames[-1][0] > frames[0][0] else None
                            for frames in self.frames],
                        camera_pts_interval_ms=[
                            round(sum(frames[index][0]-frames[index-1][0]
                                      for index in range(1, len(frames))) /
                                  max(1, len(frames)-1) / 1e6, 2)
                            if len(frames) > 1 else None
                            for frames in self.frames],
                        camera_age_ms=[None if not frames else round((now-frames[-1][2])*1000) for frames in self.frames],
                        camera_pts_ns=[[int(frame[0]) for frame in frames] for frames in self.frames],
                        capture_count=len(captures), capture_busy=self.store.lock.locked(),
                        preview_stats=self.preview_stats.copy(),
                        camera_copy_ms=[None if not frames else round(frames[-1][3], 2)
                                        for frames in self.frames],
                        camera_copy_ms_p50=[
                            round(float(np.median([frame[3] for frame in frames])), 2)
                            if frames else None for frames in self.frames],
                        camera_format='NV12',
                        hardware_video=(self.hardware_video.status() if self.hardware_video else
                                        dict(enabled=False, error=self.hardware_video_error)),
                        camera_stream_resolution=[self.stream_width, self.stream_height],
                        capture_resolution=[CAPTURE_WIDTH, CAPTURE_HEIGHT],
                        uncompressed_input_bytes_per_second=[
                            round(self.stream_width * self.stream_height * 1.5 * fps, 0)
                            if fps is not None else None
                            for fps in [
                                ((len(frames)-1)*1e9/(frames[-1][0]-frames[0][0])
                                 if len(frames) > 1 and frames[-1][0] > frames[0][0] else None)
                                for frames in self.frames]],
                        pointcloud_error=self.pointcloud_error,
                        capture_error=self.capture_error, capture_width=CAPTURE_WIDTH,
                        capture_height=CAPTURE_HEIGHT, capture_directory=str(self.store.directory),
                        calibration_target=CAPTURE_TARGET.copy(),
                        last_capture=captures[-1] if captures else None)


def serve_depth_bin(handler, live, query, wait_timeout=1.0):
    """Send one locked immutable snapshot; query is parse_qs(..., keep_blank_values=True)."""
    try:
        sequence = parse_requested_sequence(query)
    except ValueError as exc:
        return handler.json_response({'ok': False, 'message': str(exc)}, 400)
    with live.condition:
        if live.calibration_active:
            return handler.json_response({'ok': False, 'message': '校准中：数值深度已暂停'}, 409)
        if not live.capture_running:
            return handler.json_response({'ok': False, 'message': '相机采集管线未运行'}, 503)
        if live.depth_wire_geometry is None:
            return handler.json_response({
                'ok': False,
                'message': '当前后端/标定未提供已验证的数值深度几何',
                'detail': live.depth_wire_error,
            }, 503)
        request_epoch = live.pipeline_epoch
        # A numeric-only client activates the fast depth worker.
        live.depth_until = time.monotonic() + 5

        def ready():
            if (live.stopped.is_set() or not live.capture_running
                    or live.calibration_active or live.pipeline_epoch != request_epoch):
                return True
            history = live.measure_history
            if not history:
                return False
            if sequence is None:
                return len(history[-1]) > 3
            if any(entry[0] == sequence for entry in history):
                return True
            return history[-1][0] >= sequence

        if not ready():
            live.condition.wait_for(ready, timeout=wait_timeout)
        if live.calibration_active:
            return handler.json_response({'ok': False, 'message': '校准中：数值深度已暂停'}, 409)
        if not live.capture_running or live.pipeline_epoch != request_epoch:
            status = 409 if sequence is not None else 503
            return handler.json_response({'ok': False, 'message': '采集管线已切换，请重新读取'}, status)
        record, status, message = select_depth_record(live.measure_history, sequence)
        if status == 503 and sequence is not None:
            status = 409
    if status != 200 or record is None:
        return handler.json_response({'ok': False, 'message': message,
                                      'sequence': sequence}, status)
    return handler.send(record.payload, 'application/octet-stream',
                        headers={'X-Depth-Metadata': record.metadata_ascii})


def serve(args):
    live = Live(args.left, args.right)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, content, kind='application/json; charset=utf-8', status=200, headers=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(content)

        def json_response(self, data, status=200):
            return self.send(json.dumps(data, ensure_ascii=False).encode(), status=status)

        def redirect_depth(self):
            self.send_response(302)
            self.send_header('Location', '/?view=depth')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def removed_feature(self):
            return removed_feature_response(self)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/measure', '/pointcloud', '/far'):
                return self.redirect_depth()
            if path in ('/stereo4k', '/measure4k'):
                self.send_response(302)
                self.send_header('Location', '/?view=raw')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path in ('/measure.jpg', '/pointcloud.bin', '/range4k',
                        '/far.jpg', '/far/status', '/far/point', '/npu.jpg',
                        '/video/0.mp4', '/video/1.mp4'):
                return self.removed_feature()
            if path in ('/ui.css', '/ui.js'):
                kind = 'text/css; charset=utf-8' if path.endswith('.css') else 'application/javascript; charset=utf-8'
                return self.send(Path(__file__).with_name(path[1:]).read_bytes(), kind)
            if path == '/calibration_nav.js':
                return self.send(Path(__file__).with_name('calibration_nav.js').read_bytes(), 'application/javascript; charset=utf-8')
            if path == '/calibration' or path.startswith('/calibration/'):
                try:
                    return live.calibration.handle_get(self, path, parse_qs(urlsplit(self.path).query))
                except (ValueError, OSError) as exc:
                    return self.json_response({'message': str(exc)}, 409)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if live.calibration_active and path in ('/depth.jpg', '/depth.bin'):
                return self.json_response({'message': '校准中：深度接口已暂停'}, 409)
            fixed_scale = parse_qs(urlsplit(self.path).query).get('scale', ['auto'])[0] == 'fixed'
            if path == '/':
                return self.send(Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
            page_names = {'/hardware': 'hardware.html'}
            if path in page_names:
                page = Path(__file__).with_name(page_names[path])
                if not page.is_file():
                    return self.json_response({'message': '页面文件不存在：'+page.name}, 503)
                return self.send(page.read_bytes(), 'text/html; charset=utf-8')
            if path == '/status':
                status = live.status()
                if live.metric is not None:
                    status['depth_color_mode'] = 'fixed' if fixed_scale else 'auto'
                    if fixed_scale:
                        status['depth_color_range_m'] = list(live.metric.RANGE_M)
                return self.json_response(status)
            if path == '/depth.bin':
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                return serve_depth_bin(self, live, query)
            if path == '/hardware/status':
                return self.json_response(live.hardware())
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
                        jpeg = (live.preview_jpeg if stereo else
                                live.jpeg_fixed if fixed_scale else live.jpeg)
                    if jpeg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = urlsplit(self.path).path
            if path in ('/measure-point', '/measure-point4k', '/far/point'):
                return self.removed_feature()
            origin = self.headers.get('Origin')
            if (self.headers.get('Sec-Fetch-Site') == 'cross-site' or
                    (origin and origin not in ('http://'+self.headers.get('Host', ''), 'https://'+self.headers.get('Host', '')))):
                return self.json_response({'message':'请从本机拍照界面操作'}, 403)
            if path.startswith('/calibration/'):
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 <= length <= 4096:
                        raise ValueError('请求长度无效')
                    payload = json.loads(self.rfile.read(length)) if length else {}
                    if not isinstance(payload, dict):
                        raise ValueError('请求必须是 JSON 对象')
                    return live.calibration.handle_post(self, path, payload)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    return self.json_response({'message': str(exc)}, 409)
                except Exception as exc:
                    return self.json_response({'message': '校准操作失败：' + str(exc)}, 500)
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
        live._stop_capture_pipeline()
        live.depth_thread.join(timeout=5)
        close_metric = getattr(live.metric, 'close', None)
        if close_metric:
            close_metric()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='/dev/video33')
    parser.add_argument('--right', default='/dev/video42')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--host', default='127.0.0.1')
    serve(parser.parse_args())
