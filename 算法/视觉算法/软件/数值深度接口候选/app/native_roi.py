"""Low-rate, independent rectified-ROI depth experiment for the 4K cameras.

This branch is deliberately demand driven. It drains the two GStreamer appsinks
at all times, but only maps/remaps images and runs SGBM while a caller has an
active request. It does not own or stop the shared video pipeline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Optional

import cv2
import numpy as np

from roi_depth import ROISGBMDepth


RAW_WIDTH = 3840
RAW_HEIGHT = 2160
PAIR_TOLERANCE_NS = 25_000_000
REQUEST_SECONDS = 8.0
HISTORY_SIZE = 12
CAPTURE_QUEUE_SIZE = 3
PAIR_MAX_ARRIVAL_AGE_NS = 750_000_000


def _pts_bucket(pts_ns: int, rate: int) -> int:
    """Return a stable PTS-time bucket used for per-eye remap throttling."""
    if isinstance(pts_ns, bool) or not isinstance(pts_ns, int) or pts_ns < 0:
        raise ValueError("pts_ns must be a non-negative integer")
    if isinstance(rate, bool) or not isinstance(rate, int) or not 1 <= rate <= 30:
        raise ValueError("rate must be an integer from 1 to 30 fps")
    return pts_ns // (1_000_000_000 // rate)


@dataclass(frozen=True)
class _GrayFrame:
    pts_ns: int
    arrival_mono_ns: int
    gray: np.ndarray
    valid: np.ndarray
    rectification_ms: float
    demand_generation: int = 0


@dataclass(frozen=True)
class NativeDepthRecord:
    """Immutable ROI output; arrays are backed by immutable ``bytes``."""

    sequence: int
    depth_z_m: np.ndarray
    valid: np.ndarray
    preview_jpeg: bytes
    metadata_json: str

    def metadata(self) -> dict[str, Any]:
        """Return a detached JSON metadata snapshot."""
        return json.loads(self.metadata_json)


def _readonly_copy(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"metadata value {type(value).__name__} is not JSON serializable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeROIDepth:
    """Small rectified ROI SGBM experiment sourced from the 4K raw video tees.

    ``rate`` is the GStreamer raw branch rate.  The class has its own lifetime
    event and never mutates the shared preview/depth server's stop state.
    """

    def __init__(self, Gst, GstVideo, calibration_path, rate=10,
                 canvas_wh=(640, 320), range_m=(5.0, 50.0), near_guard=None):
        if isinstance(rate, bool) or not isinstance(rate, int) or not 1 <= rate <= 30:
            raise ValueError("rate must be an integer from 1 to 30 fps")
        if len(canvas_wh) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in canvas_wh):
            raise ValueError("canvas_wh must be integer (width, height)")
        self.canvas_w, self.canvas_h = canvas_wh
        if self.canvas_w < 128 or self.canvas_h < 64:
            raise ValueError("canvas dimensions are too small for the 64-pixel search ROI")
        if (len(range_m) != 2 or not all(math.isfinite(float(v)) and float(v) > 0 for v in range_m)
                or float(range_m[0]) >= float(range_m[1])):
            raise ValueError("range_m must be a finite increasing positive pair")
        self.range_m = (float(range_m[0]), float(range_m[1]))
        self.rate = rate
        self._bucket_span_ns = 1_000_000_000 // rate
        if near_guard is not None and not callable(near_guard):
            raise ValueError("near_guard must be callable or None")
        self.near_guard = near_guard
        self.Gst = Gst
        self.GstVideo = GstVideo
        # Resolve PyGObject constants and VideoInfo descriptors in the caller
        # thread before capture workers can race lazy GI enum initialization.
        self._gst_runtime_available = Gst is not None and GstVideo is not None
        if self._gst_runtime_available:
            self._gst_second = int(Gst.SECOND)
            self._gst_clock_time_none = int(Gst.CLOCK_TIME_NONE)
            self._gst_format_time = Gst.Format.TIME
            self._gst_map_read = Gst.MapFlags.READ
            self._gstvideo_buffer_get_video_meta = GstVideo.buffer_get_video_meta
            self._gstvideo_info_type = GstVideo.VideoInfo
            self._gstvideo_info_from_caps = self._gstvideo_info_type.from_caps
        else:
            # Saved-image / offline geometry use does not need GStreamer.
            self._gst_second = self._gst_clock_time_none = None
            self._gst_format_time = self._gst_map_read = None
            self._gstvideo_buffer_get_video_meta = None
            self._gstvideo_info_type = self._gstvideo_info_from_caps = None
        self.calibration_path = Path(calibration_path)
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._pipeline = None
        self._lifecycle_generation = 0
        self._sinks = [None, None]
        self._queues = [deque(maxlen=CAPTURE_QUEUE_SIZE), deque(maxlen=CAPTURE_QUEUE_SIZE)]
        self._last_used_pts = [-1, -1]
        self._last_remap_bucket: list[Optional[int]] = [None, None]
        self._history: deque[NativeDepthRecord] = deque(maxlen=HISTORY_SIZE)
        self._request_until = 0.0
        self._request_generation = 0
        self._pipeline_epoch: Optional[int] = None
        self._session_id = str(uuid.uuid4())
        self._sequence = 0
        self._error: Optional[str] = None
        self._capture_dropped = [0, 0]
        self._processed_pairs = 0
        self._last_pair_delta_ns: Optional[int] = None
        self._last_timings: dict[str, Optional[float]] = {}
        self._load_geometry()

    def _load_geometry(self) -> None:
        if not self.calibration_path.is_file():
            raise FileNotFoundError(self.calibration_path)
        with np.load(self.calibration_path, allow_pickle=False) as calibration:
            required = ("K1", "D1", "R1", "P1", "K2", "D2", "R2", "P2", "image_size")
            if any(key not in calibration for key in required):
                raise ValueError("calibration is missing a left/right K/D/R/P matrix or image_size")
            image_size = np.asarray(calibration["image_size"]).reshape(-1)
            if image_size.size != 2 or tuple(map(int, image_size)) != (RAW_WIDTH, RAW_HEIGHT):
                raise ValueError("native ROI requires calibration image_size=[3840,2160]")
            self._K = [np.asarray(calibration[f"K{i}"], dtype=np.float64).copy() for i in (1, 2)]
            self._D = [np.asarray(calibration[f"D{i}"], dtype=np.float64).copy() for i in (1, 2)]
            self._R = [np.asarray(calibration[f"R{i}"], dtype=np.float64).copy() for i in (1, 2)]
            self._P1 = np.asarray(calibration["P1"], dtype=np.float64).copy()
            self._P2 = np.asarray(calibration["P2"], dtype=np.float64).copy()
        if (any(matrix.shape != (3, 3) for matrix in self._K + self._R)
                or any(matrix.ndim != 2 or matrix.shape[0] != 1 for matrix in self._D)
                or self._P1.shape != (3, 4) or self._P2.shape != (3, 4)
                or not all(np.isfinite(a).all() for a in self._K + self._D + self._R + [self._P1, self._P2])):
            raise ValueError("calibration matrices have invalid dimensions or non-finite values")

        # The source is 3840x2160 landscape. The exposed canvas is a centered
        # crop of its 90-degree CCW upright view; its raw rectangular precursor
        # is small even though the source maps still refer to the full sensor.
        upright_w, upright_h = RAW_HEIGHT, RAW_WIDTH
        self.origin_x = (upright_w - self.canvas_w) // 2
        self.origin_y = (upright_h - self.canvas_h) // 2
        if self.origin_x < 0 or self.origin_y < 0:
            raise ValueError("requested canvas does not fit inside the upright sensor image")
        raw_u0 = RAW_WIDTH - self.origin_y - self.canvas_h
        raw_v0 = self.origin_x
        raw_crop_w, raw_crop_h = self.canvas_h, self.canvas_w
        self.raw_roi_xywh = (raw_u0, raw_v0, raw_crop_w, raw_crop_h)

        crop_transform = np.array([[1.0, 0.0, -raw_u0],
                                   [0.0, 1.0, -raw_v0],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)
        self._maps_x, self._maps_y, self._valid_canvases = [], [], []
        for eye, projection in enumerate((self._P1, self._P2)):
            shifted_projection = crop_transform @ projection
            map_x, map_y = cv2.initUndistortRectifyMap(
                self._K[eye], self._D[eye], self._R[eye], shifted_projection[:, :3],
                (raw_crop_w, raw_crop_h), cv2.CV_32FC1)
            source_supported = (
                np.isfinite(map_x) & np.isfinite(map_y)
                & (map_x >= 0.0) & (map_x < RAW_WIDTH - 1.0)
                & (map_y >= 0.0) & (map_y < RAW_HEIGHT - 1.0)
            )
            valid_canvas = cv2.rotate(source_supported.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)
            self._maps_x.append(map_x)
            self._maps_y.append(map_y)
            self._valid_canvases.append(valid_canvas)
        expected_shape = (self.canvas_h, self.canvas_w)
        if any(mask.shape != expected_shape for mask in self._valid_canvases):
            raise RuntimeError(f"internal ROI rotation produced {[m.shape for m in self._valid_canvases]}, expected {expected_shape}")
        for array in self._maps_x + self._maps_y + self._valid_canvases:
            array.setflags(write=False)

        # P1/P2 are sensor-space rectified projection matrices. Rotation and
        # crop are applied analytically so this K matches the remapped canvas.
        source_fx = float(self._P1[0, 0])
        source_fy = float(self._P1[1, 1])
        source_cx = float(self._P1[0, 2])
        source_cy = float(self._P1[1, 2])
        self.K_canvas = np.array([
            [source_fy, 0.0, source_cy - self.origin_x],
            [0.0, source_fx, RAW_WIDTH - 1.0 - source_cx - self.origin_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        fB_px_m = -float(self._P2[1, 3]) / 1000.0
        disparity_offset = float(self._P1[1, 2] - self._P2[1, 2])
        if not math.isfinite(fB_px_m) or fB_px_m <= 0:
            raise ValueError("calibration has no positive vertical stereo baseline in P2")
        self._roi_xywh = (64, 0, self.canvas_w - 64, self.canvas_h)
        self._matcher = ROISGBMDepth(
            K=self.K_canvas,
            fB_px_m=fB_px_m,
            min_disparity=0,
            num_disparities=64,
            disparity_offset_px=disparity_offset,
            depth_range_m=self.range_m,
            block_size=5,
            p1=200,
            p2=800,
        )
        self._calibration_sha256 = _sha256_file(self.calibration_path)
        self._source_sha256 = {
            "native_roi.py": _sha256_file(Path(__file__)),
            "roi_depth.py": _sha256_file(Path(__file__).with_name("roi_depth.py")),
        }
        geometry_descriptor = {
            "calibration_sha256": self._calibration_sha256,
            "source_sha256": self._source_sha256,
            "source_wh": [RAW_WIDTH, RAW_HEIGHT],
            "canvas_wh": [self.canvas_w, self.canvas_h],
            "origin_xy": [self.origin_x, self.origin_y],
            "raw_roi_xywh": list(self.raw_roi_xywh),
            "K_canvas": self.K_canvas.tolist(),
            "roi_xywh": list(self._roi_xywh),
            "depth_range_m": list(self.range_m),
            "min_disparity": 0,
            "num_disparities": 64,
            "block_size": 5,
            "P1": 200,
            "P2": 800,
            "algorithm": "OpenCV StereoSGBM SGBM_3WAY roi_depth.ROISGBMDepth",
            "opencv": cv2.__version__,
        }
        self.preprocessing_id = hashlib.sha256(
            json.dumps(geometry_descriptor, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        self._geometry_descriptor = geometry_descriptor

    def start(self, pipeline, pipeline_epoch):
        """Attach to an already-running shared pipeline and launch three workers."""
        if not self._gst_runtime_available:
            raise RuntimeError("GStreamer bindings are required to start native ROI capture")
        if isinstance(pipeline_epoch, bool) or not isinstance(pipeline_epoch, int):
            raise ValueError("pipeline_epoch must be an integer")
        with self._condition:
            if self._pipeline is not None or self._threads:
                raise RuntimeError("NativeROIDepth has already been started")
            self._pipeline = pipeline
            self._lifecycle_generation += 1
            generation = self._lifecycle_generation
            self._pipeline_epoch = pipeline_epoch
            self._session_id = str(uuid.uuid4())
            self._stop_event.clear()
            self._request_until = 0.0
            self._request_generation += 1
            self._error = None
            self._sequence = 0
            self._history.clear()
            self._queues = [deque(maxlen=CAPTURE_QUEUE_SIZE), deque(maxlen=CAPTURE_QUEUE_SIZE)]
            self._last_used_pts = [-1, -1]
            self._last_remap_bucket = [None, None]
            self._capture_dropped = [0, 0]
            self._processed_pairs = 0
            self._last_pair_delta_ns = None
            try:
                self._sinks = [pipeline.get_by_name("far0"), pipeline.get_by_name("far1")]
                if any(sink is None for sink in self._sinks):
                    raise RuntimeError("pipeline is missing far0/far1 raw ROI appsinks")
            except Exception as exc:
                self._error = str(exc)
                self._pipeline = None
                self._sinks = [None, None]
                return self.status()
            self._threads = [
                threading.Thread(target=self._capture_worker, args=(eye, generation),
                                 name=f"native-roi-capture-{eye}", daemon=True)
                for eye in range(2)
            ]
            self._threads.append(threading.Thread(target=self._compute_worker, args=(generation,),
                                                  name="native-roi-compute", daemon=True))
            for thread in self._threads:
                thread.start()
        return self.status()

    def request(self):
        """Request demand for up to eight seconds, extending an active request."""
        with self._condition:
            now = time.monotonic()
            if self._request_until <= now:
                # Do not publish a buffered pair from an earlier request after
                # a long idle interval; wait for fresh live samples instead.
                self._queues = [deque(maxlen=CAPTURE_QUEUE_SIZE), deque(maxlen=CAPTURE_QUEUE_SIZE)]
                self._request_generation += 1
            self._request_until = max(self._request_until, now + REQUEST_SECONDS)
            self._condition.notify_all()
            return {"requested": True, "request_until_mono": self._request_until}

    def _active_demand_generation(self, generation: int) -> Optional[int]:
        with self._condition:
            if (generation != self._lifecycle_generation or self._stop_event.is_set()
                    or time.monotonic() >= self._request_until):
                return None
            return self._request_generation

    def _sample_running_pts(self, sample) -> Optional[int]:
        buffer = sample.get_buffer()
        pts = int(buffer.pts)
        if pts == self._gst_clock_time_none:
            return None
        segment = sample.get_segment()
        if segment is None:
            return None
        running = segment.to_running_time(self._gst_format_time, pts)
        if running is None or int(running) == self._gst_clock_time_none:
            return None
        return int(running)

    def _claim_remap_bucket(self, eye: int, pts_ns: int, lifecycle_generation: int,
                            demand_generation: int) -> Optional[int]:
        """Claim one monotonic PTS bucket for one eye before mapping its buffer."""
        if eye not in (0, 1):
            raise ValueError("eye must be 0 or 1")
        bucket = _pts_bucket(pts_ns, self.rate)
        with self._condition:
            if (lifecycle_generation != self._lifecycle_generation
                    or demand_generation != self._request_generation
                    or self._stop_event.is_set()
                    or time.monotonic() >= self._request_until):
                return None
            last_bucket = self._last_remap_bucket[eye]
            # PTS should be ordered on each appsink. Reject duplicate buckets
            # and any late sample that would revisit an already-used bucket.
            if last_bucket is not None and bucket <= last_bucket:
                return None
            self._last_remap_bucket[eye] = bucket
            return bucket

    def _capture_worker(self, eye: int, generation: int) -> None:
        sink = self._sinks[eye]
        try:
            while generation == self._lifecycle_generation and not self._stop_event.is_set():
                sample = sink.emit("try-pull-sample", self._gst_second // 2)
                if sample is None:
                    continue
                arrival_ns = time.monotonic_ns()
                demand_generation = self._active_demand_generation(generation)
                if demand_generation is None:
                    with self._condition:
                        if generation == self._lifecycle_generation:
                            self._capture_dropped[eye] += 1
                    continue
                pts_ns = self._sample_running_pts(sample)
                if pts_ns is None:
                    with self._condition:
                        if generation == self._lifecycle_generation:
                            self._capture_dropped[eye] += 1
                    continue
                if self._claim_remap_bucket(eye, pts_ns, generation, demand_generation) is None:
                    with self._condition:
                        if generation == self._lifecycle_generation:
                            self._capture_dropped[eye] += 1
                    continue
                gray, valid, rectification_ms = self._sample_to_canvas(sample, eye)
                frame = _GrayFrame(pts_ns, arrival_ns, gray, valid, rectification_ms,
                                   demand_generation)
                with self._condition:
                    if (generation != self._lifecycle_generation or self._stop_event.is_set()
                            or demand_generation != self._request_generation
                            or time.monotonic() >= self._request_until):
                        if generation == self._lifecycle_generation:
                            self._capture_dropped[eye] += 1
                        continue
                    self._queues[eye].append(frame)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if generation == self._lifecycle_generation:
                    self._error = f"capture eye {eye}: {exc}"
                self._condition.notify_all()

    def _sample_to_canvas(self, sample, eye: int):
        caps = sample.get_caps()
        if caps is None or caps.get_size() < 1:
            raise RuntimeError("raw appsink sample has no caps")
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        pixel_format = structure.get_string("format")
        if (width, height, pixel_format) != (RAW_WIDTH, RAW_HEIGHT, "NV12"):
            raise RuntimeError(f"expected {RAW_WIDTH}x{RAW_HEIGHT} NV12, received {width}x{height} {pixel_format}")
        buffer = sample.get_buffer()
        mapped, map_info = buffer.map(self._gst_map_read)
        if not mapped:
            raise RuntimeError("unable to map NV12 GstBuffer")
        try:
            meta = self._gstvideo_buffer_get_video_meta(buffer)
            if meta is not None:
                stride = int(meta.stride[0])
                offset = int(meta.offset[0])
            else:
                info = self._gstvideo_info_type()
                if not self._gstvideo_info_from_caps(info, caps):
                    raise RuntimeError("unable to read NV12 plane layout")
                stride = int(info.stride[0])
                offset = int(info.offset[0])
            if stride < RAW_WIDTH or offset < 0:
                raise RuntimeError("invalid NV12 luma plane stride/offset")
            mapped_data = map_info.data
            data_size = len(mapped_data)
            last_required = offset + (RAW_HEIGHT - 1) * stride + RAW_WIDTH
            if last_required > data_size:
                raise RuntimeError("mapped GstBuffer is shorter than the NV12 luma plane")
            source = np.ndarray((RAW_HEIGHT, RAW_WIDTH), dtype=np.uint8,
                                buffer=mapped_data, offset=offset,
                                strides=(stride, 1))
            return self._rectify_luma(source, eye)
        finally:
            buffer.unmap(map_info)

    def _rectify_luma(self, source, eye=0):
        if eye not in (0, 1):
            raise ValueError("eye must be 0 or 1")
        source = np.asarray(source)
        if source.shape != (RAW_HEIGHT, RAW_WIDTH) or source.dtype != np.uint8:
            raise ValueError("luma plane must be uint8 with shape (2160,3840)")
        start = time.perf_counter_ns()
        raw_crop = cv2.remap(source, self._maps_x[eye], self._maps_y[eye], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        canvas = cv2.rotate(raw_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = np.ascontiguousarray(canvas, dtype=np.uint8)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        # Copy masks so records never retain calibration map references or a
        # mapped GstBuffer's lifetime.
        valid = np.array(self._valid_canvases[eye], dtype=bool, copy=True)
        return gray, valid, elapsed_ms

    def _select_pair_locked(self):
        cutoff_ns = time.monotonic_ns() - PAIR_MAX_ARRIVAL_AGE_NS
        self._queues = [
            deque((frame for frame in queue if frame.arrival_mono_ns >= cutoff_ns),
                  maxlen=CAPTURE_QUEUE_SIZE)
            for queue in self._queues
        ]
        lefts = [frame for frame in self._queues[0] if frame.pts_ns > self._last_used_pts[0]]
        rights = [frame for frame in self._queues[1] if frame.pts_ns > self._last_used_pts[1]]
        candidates = []
        for left in lefts:
            for right in rights:
                delta = abs(left.pts_ns - right.pts_ns)
                if delta <= PAIR_TOLERANCE_NS:
                    candidates.append((max(left.pts_ns, right.pts_ns), min(left.pts_ns, right.pts_ns),
                                       -delta, left, right))
        if not candidates:
            return None
        _, _, neg_delta, left, right = max(candidates, key=lambda row: row[:3])
        self._last_used_pts = [left.pts_ns, right.pts_ns]
        self._queues[0] = deque((f for f in self._queues[0] if f.pts_ns > left.pts_ns), maxlen=CAPTURE_QUEUE_SIZE)
        self._queues[1] = deque((f for f in self._queues[1] if f.pts_ns > right.pts_ns), maxlen=CAPTURE_QUEUE_SIZE)
        return left, right, -neg_delta

    def _compute_worker(self, generation: int) -> None:
        while generation == self._lifecycle_generation and not self._stop_event.is_set():
            with self._condition:
                pending = None
                while (generation == self._lifecycle_generation and not self._stop_event.is_set()
                       and pending is None):
                    if time.monotonic() < self._request_until:
                        pending = self._select_pair_locked()
                    if pending is None:
                        self._condition.wait(timeout=0.5)
                if generation != self._lifecycle_generation or self._stop_event.is_set():
                    return
            left, right, delta_ns = pending
            try:
                result = self._matcher.compute(
                    left.gray, right.gray, left.valid, right.valid, self._roi_xywh)
                self._publish(left, right, delta_ns, result, generation)
            except Exception as exc:
                with self._condition:
                    if generation == self._lifecycle_generation:
                        self._error = f"compute: {exc}"
                    self._condition.notify_all()

    def _publish(self, left: _GrayFrame, right: _GrayFrame, delta_ns: int, result,
                 generation: int) -> None:
        depth_source = np.asarray(result.depth_z_m, dtype=np.float64)
        valid = np.asarray(result.valid, dtype=bool)
        if depth_source.shape != valid.shape or depth_source.ndim != 2:
            raise RuntimeError("ROI matcher returned mismatched depth and validity arrays")
        near, far = self.range_m
        valid = np.array(valid & np.isfinite(depth_source)
                         & (depth_source >= near) & (depth_source <= far),
                         dtype=bool, copy=True)
        depth = np.array(depth_source, dtype=np.float32, copy=True)
        K_roi = np.asarray(result.K_roi, dtype=np.float64).reshape(3, 3).copy()
        with self._condition:
            if (generation != self._lifecycle_generation
                    or left.demand_generation != right.demand_generation
                    or left.demand_generation != self._request_generation
                    or self._stop_event.is_set()
                    or time.monotonic() >= self._request_until
                    or self._pipeline_epoch is None):
                return
            pipeline_epoch = self._pipeline_epoch
            near_guard = self.near_guard

        guard_status = "not_available"
        guard_verification = "unverified"
        guard_metadata: dict[str, Any] = {"status": "not_available"}
        keep_mask = np.ones(valid.shape, dtype=bool)
        if near_guard is not None:
            try:
                callback_result = near_guard(
                    left.pts_ns, right.pts_ns, pipeline_epoch, K_roi.copy(), depth.shape)
                if not isinstance(callback_result, tuple) or len(callback_result) != 2:
                    raise ValueError("near_guard must return (keep_bool_mask, metadata_dict)")
                raw_keep, raw_metadata = callback_result
                raw_keep = np.asarray(raw_keep)
                if raw_keep.dtype != np.bool_ or raw_keep.shape != valid.shape:
                    raise ValueError("near_guard mask must be bool and match the depth shape")
                if not isinstance(raw_metadata, dict):
                    raise ValueError("near_guard metadata must be a dict")
                # Verify metadata before accepting the mask, and detach it from
                # caller-owned mutable objects before the record is published.
                guard_metadata = json.loads(json.dumps(raw_metadata, allow_nan=False,
                                                       default=_json_default))
                keep_mask = np.array(raw_keep, dtype=bool, copy=True)
                metadata_status = str(guard_metadata.get("status", "")).strip().lower()
                unavailable_statuses = {"unavailable", "not_available", "unverified", "no_near_frame", "error"}
                if guard_metadata.get("available") is False or metadata_status in unavailable_statuses:
                    guard_status = "unverified"
                else:
                    guard_status = "applied"
                    guard_verification = "mask_applied"
            except Exception as exc:
                keep_mask = np.zeros(valid.shape, dtype=bool)
                guard_status = "error"
                guard_metadata = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:240],
                }
        valid &= keep_mask
        depth[~valid] = np.nan
        preview_jpeg = self._preview(depth, valid)
        with self._condition:
            if (generation != self._lifecycle_generation
                    or left.demand_generation != self._request_generation
                    or self._stop_event.is_set()
                    or time.monotonic() >= self._request_until
                    or self._pipeline_epoch != pipeline_epoch):
                return
            self._sequence += 1
            sequence = self._sequence
            publish_ns = time.monotonic_ns()
            roi_to_canvas = np.asarray(result.roi_to_source, dtype=np.float64).reshape(3, 3)
            metadata = {
                "schema": "native-roi-depth-experimental-v1",
                "session_id": self._session_id,
                "pipeline_epoch": pipeline_epoch,
                "seq": sequence,
                "width": int(depth.shape[1]),
                "height": int(depth.shape[0]),
                "encoding": "32FC1",
                "byte_order": "array_only",
                "storage": "immutable in-process NumPy snapshot; no wire encoding is defined",
                "unit": "m",
                "convention": "rectified_left_optical_z",
                "frame_id": "left_rectified_upright_roi",
                "K": K_roi.ravel().tolist(),
                "K_canvas": self.K_canvas.ravel().tolist(),
                "roi_to_canvas": roi_to_canvas.ravel().tolist(),
                "calibration_id": self._calibration_sha256,
                "extrinsics_id": None,
                "preprocessing_id": self.preprocessing_id,
                "range_m": list(self.range_m),
                "canvas_wh": [self.canvas_w, self.canvas_h],
                "canvas_origin_upright_xy": [self.origin_x, self.origin_y],
                "raw_roi_xywh": list(self.raw_roi_xywh),
                "roi_xywh_in_canvas": list(self._roi_xywh),
                "left_pts_ns": left.pts_ns,
                "right_pts_ns": right.pts_ns,
                "pts_delta_ns": delta_ns,
                "pts_reference": "GStreamer segment running-time",
                "source_time_reference": "left_pts",
                "source_mono_ns": None,
                "monotonic_clock_id": None,
                "clock_mapping_id": None,
                "clock_quality": "unready",
                "left_arrival_mono_ns": left.arrival_mono_ns,
                "right_arrival_mono_ns": right.arrival_mono_ns,
                "publish_mono_ns": publish_ns,
                "software_arrival_to_publish_ms": (publish_ns - min(left.arrival_mono_ns, right.arrival_mono_ns)) / 1e6,
                "synchronization": "PTS pair within 25 ms; hardware exposure sync unverified",
                "physical_accuracy_claim_m": None,
                "near_guard": guard_status,
                "near_guard_verification": guard_verification,
                "near_guard_metadata": guard_metadata,
                "valid_fraction": float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
                "compute_geometry": result.compute_geometry,
                "timings_ms": {
                    **dict(result.timings_ms),
                    "native_roi_remap_left_ms": left.rectification_ms,
                    "native_roi_remap_right_ms": right.rectification_ms,
                    "native_roi_remap_pair_max_ms": max(left.rectification_ms, right.rectification_ms),
                },
            }
            record = NativeDepthRecord(
                sequence=sequence,
                depth_z_m=_readonly_copy(depth, np.dtype(np.float32)),
                valid=_readonly_copy(valid, np.dtype(np.bool_)),
                preview_jpeg=bytes(preview_jpeg),
                metadata_json=json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
            )
            self._history.append(record)
            self._processed_pairs += 1
            self._last_pair_delta_ns = delta_ns
            self._last_timings = dict(metadata["timings_ms"])
            self._condition.notify_all()

    def _preview(self, depth: np.ndarray, valid: np.ndarray) -> bytes:
        near, far = self.range_m
        safe_depth = np.where(valid, depth, near)
        scaled = np.clip((safe_depth - near) * (255.0 / (far - near)), 0.0, 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
        preview[~valid] = 0
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            raise RuntimeError("unable to encode native ROI preview")
        return encoded.tobytes()

    def get_record(self, sequence=None):
        """Return the latest (or requested) immutable record, if still retained."""
        with self._condition:
            if not self._history:
                return None
            if sequence is None:
                return self._history[-1]
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
                raise ValueError("sequence must be a positive integer")
            return next((record for record in self._history if record.sequence == sequence), None)

    def status(self):
        with self._condition:
            age_ms = None
            if self._history:
                age_ms = max(0.0, (time.monotonic_ns() - self._history[-1].metadata()["publish_mono_ns"]) / 1e6)
            return {
                "enabled": self._pipeline is not None,
                "running": bool(self._threads) and not self._stop_event.is_set(),
                "requested": time.monotonic() < self._request_until,
                "request_remaining_ms": max(0, int((self._request_until - time.monotonic()) * 1000)),
                "pipeline_epoch": self._pipeline_epoch,
                "session_id": self._session_id,
                "canvas_wh": [self.canvas_w, self.canvas_h],
                "roi_xywh": list(self._roi_xywh),
                "range_m": list(self.range_m),
                "history_count": len(self._history),
                "latest_sequence": self._history[-1].sequence if self._history else None,
                "latest_age_ms": age_ms,
                "processed_pairs": self._processed_pairs,
                "last_pair_delta_ms": self._last_pair_delta_ns / 1e6 if self._last_pair_delta_ns is not None else None,
                "capture_dropped": list(self._capture_dropped),
                "last_timings_ms": dict(self._last_timings),
                "preprocessing_id": self.preprocessing_id,
                "calibration_id": self._calibration_sha256,
                "hardware_sync_verified": False,
                "physical_accuracy_claim_m": None,
                "error": self._error,
            }

    def close(self):
        """Stop only this adapter's workers and clear its private state."""
        with self._condition:
            self._lifecycle_generation += 1
            self._request_generation += 1
            self._stop_event.set()
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=1.5)
        with self._condition:
            self._threads.clear()
            self._pipeline = None
            self._sinks = [None, None]
            self._queues = [deque(maxlen=CAPTURE_QUEUE_SIZE), deque(maxlen=CAPTURE_QUEUE_SIZE)]
            self._last_remap_bucket = [None, None]
            self._history.clear()
            self._request_until = 0.0
            self._condition.notify_all()

