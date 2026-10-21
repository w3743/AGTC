"""MetricDepth backend using LightStereo RKNN inference on the RK3588 NPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from depth import MetricDepth as BaseDepth


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "lightstereo_640_352.rknn"
RUNTIME_PATH = ROOT / "runtime" / "librknnrt.so"
FORWARD_REFERENCE = ROOT / "validation" / "reference.npz"
REVERSE_REFERENCE = ROOT / "validation" / "reference-reverse.npz"
CROP = 4


def _isolated_rknn_class():
    try:
        import rknnlite.api.rknn_lite as lite
        from rknnlite.api.rknn_runtime import RKNNRuntime
    except Exception as exc:
        raise RuntimeError(f"无法导入 rknnlite2: {exc}") from exc

    runtime_path = str(RUNTIME_PATH)

    class IsolatedRuntime(RKNNRuntime):
        def _get_rknn_api_lib_path(self):
            return runtime_path

    # RKNNLite imports this symbol when constructing a session. Replacing it
    # here scopes the override to this Python process and leaves system files alone.
    lite.RKNNRuntime = IsolatedRuntime
    from rknnlite.api import RKNNLite
    return RKNNLite


class MetricDepth(BaseDepth):
    """Drop-in server backend; outputs metric depth and standard server metadata."""

    CROP = CROP

    def __init__(self, path, *, input_size=None):
        self.input_size = self.SENSOR_SIZE if input_size is None else tuple(input_size)
        self._forward_rknn = None
        self._reverse_rknn = None
        self._executor = None
        self.inference_timing = {}
        self._closed = False
        self._close_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        try:
            super().__init__(path)
            self.width -= 2 * self.CROP
            self.masks = [m[:, self.CROP:-self.CROP].copy() for m in self.masks]
            self.yy, self.xx = np.indices((self.height, self.width))
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"LightStereo 模型不存在: {MODEL_PATH}")
            if not RUNTIME_PATH.is_file():
                raise FileNotFoundError(f"隔离版 RKNN runtime 不存在: {RUNTIME_PATH}")
            RKNNLite = _isolated_rknn_class()
            for attr, label, core_mask in (
                ("_forward_rknn", "正向", 1), ("_reverse_rknn", "反向", 2),
            ):
                session = RKNNLite(verbose=False)
                setattr(self, attr, session)
                if session.load_rknn(str(MODEL_PATH)) != 0:
                    raise RuntimeError(f"LightStereo {label} RKNN 模型加载失败")
                if session.init_runtime(core_mask=core_mask) != 0:
                    raise RuntimeError(
                        f"LightStereo {label} NPU runtime 初始化失败(core_mask={core_mask})")
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lightstereo-npu")
        except Exception:
            self.close()
            raise

    def rectify(self, left, right):
        images = super().rectify(left, right)
        start = time.perf_counter()
        cropped = [np.ascontiguousarray(im[:, self.CROP:-self.CROP]) for im in images]
        self.crop_timing_ms = (time.perf_counter() - start) * 1000
        return cropped

    @staticmethod
    def _as_model_image(gray):
        if gray.shape != (640, 352) or gray.dtype != np.uint8:
            raise ValueError(f"LightStereo 输入必须为 640x352 uint8 灰度图，实际 {gray.shape} {gray.dtype}")
        # The exported RKNN graph includes normalization. Keep input uint8 NHWC.
        return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2)[None, ...])

    def _infer(self, left, right, session=None, timing_key=None):
        session = self._forward_rknn if session is None else session
        if self._closed or session is None:
            raise RuntimeError("LightStereo NPU session 未初始化或已关闭")
        started = time.perf_counter()
        outputs = session.inference(
            inputs=[self._as_model_image(left), self._as_model_image(right)],
            data_format=["nhwc", "nhwc"],
        )
        inference_ms = (time.perf_counter() - started) * 1000
        if timing_key is not None:
            timings = getattr(self, 'inference_timing', None)
            if timings is None:
                timings = self.inference_timing = {}
            timings[timing_key+'_ms'] = inference_ms
        if not outputs or outputs[0] is None:
            raise RuntimeError("LightStereo 推理没有返回视差输出")
        disparity = np.asarray(outputs[0]).squeeze()
        if disparity.shape != (640, 352) or not np.isfinite(disparity).all():
            raise RuntimeError(f"LightStereo 输出无效: shape={disparity.shape}")
        return disparity.astype(np.float32, copy=True)

    def disparities(self, left, right):
        with self._inference_lock:
            if self._closed or self._executor is None:
                raise RuntimeError("LightStereo NPU backend 已关闭")
            # Snapshot the full frame before queuing either worker, so a caller
            # reusing its camera buffers cannot change this pair mid-inference.
            forward_left = np.ascontiguousarray(left).copy()
            forward_right = np.ascontiguousarray(right).copy()
            reverse_left = np.ascontiguousarray(right[:, ::-1]).copy()
            reverse_right = np.ascontiguousarray(left[:, ::-1]).copy()
            forward_future = self._executor.submit(
                self._infer, forward_left, forward_right, self._forward_rknn, 'forward')
            reverse_future = self._executor.submit(
                self._infer, reverse_left, reverse_right, self._reverse_rknn, 'reverse')
            results = {}
            errors = []
            for label, future in (("正向", forward_future), ("反向", reverse_future)):
                try:
                    results[label] = future.result()
                except Exception as exc:
                    errors.append((label, exc))
            if errors:
                label, exc = errors[0]
                raise RuntimeError(f"LightStereo {label}并行推理失败: {exc}") from exc
            forward = results["正向"]
            reverse_flipped = results["反向"]
        # Reverse disparity has the opposite sign; unflip it into right-image coordinates.
        start = time.perf_counter()
        reverse = -np.ascontiguousarray(reverse_flipped[:, ::-1]).copy()
        timings = getattr(self, 'inference_timing', {})
        timings['reverse_unflip_ms'] = (time.perf_counter() - start) * 1000
        self.inference_timing = timings
        return forward, reverse

    def compute(self, left, right):
        result = super().compute(left, right)
        result[2]["npu_execution"] = dict(mode="dual_session_parallel",
                                          forward_core_mask=1, reverse_core_mask=2)
        return result

    def metric_values(self, dl, dr, left):
        consistency_start = time.perf_counter()
        xr = np.rint(self.xx - dl).astype(np.int32)
        inside = (xr >= 0) & (xr < self.width)
        safe = xr.clip(0, self.width - 1)
        rows = self.yy
        valid = inside & np.isfinite(dl) & (dl > 0)
        matched_fraction = float(valid.mean())
        reverse_at_match = dr[rows, safe]
        valid &= np.isfinite(reverse_at_match)
        valid &= np.abs(dl + reverse_at_match) <= 1.0
        valid &= self.masks[0] & self.masks[1][rows, safe]
        consistent_fraction = float(valid.mean())

        consistency_ms = (time.perf_counter() - consistency_start) * 1000
        filter_start = time.perf_counter()
        denominator = dl - self.disparity_offset
        depth = np.full(dl.shape, np.nan, dtype=np.float32)
        np.divide(self.depth_factor, denominator, out=depth,
                  where=np.isfinite(denominator) & (denominator > 0))
        valid &= np.isfinite(depth) & (depth >= self.RANGE_M[0]) & (depth <= self.RANGE_M[1])
        depth[~valid] = np.nan
        valid_fraction = float(valid.mean())
        # Keep keys consumed by the existing server and status UI. LightStereo
        # has no SGBM texture filter, so textured_fraction mirrors consistency support.
        self.filter_stats = dict(matched_fraction=matched_fraction,
                                 consistent_fraction=consistent_fraction,
                                 textured_fraction=consistent_fraction,
                                 range_fraction=valid_fraction)
        self.filter_timing = dict(consistency_ms=consistency_ms,
                                  depth_filter_ms=(time.perf_counter() - filter_start) * 1000)
        return depth, valid

    def close(self):
        with self._inference_lock:
            with self._close_lock:
                if self._closed:
                    return
                self._closed = True
                executor, self._executor = self._executor, None
                sessions = (self._forward_rknn, self._reverse_rknn)
                self._forward_rknn = None
                self._reverse_rknn = None
        if executor is not None:
            executor.shutdown(wait=True)
        for session in sessions:
            if session is not None:
                session.release()


def _reference_report(session, reference_path, runs):
    with np.load(reference_path, allow_pickle=False) as reference:
        left = np.ascontiguousarray(reference["left"])
        right = np.ascontiguousarray(reference["right"])
        expected = np.asarray(reference["disparity"]).squeeze().astype(np.float32)
        width = int(reference["content_width"]) if "content_width" in reference else expected.shape[1]
        pad_top = int(reference["pad_top"]) if "pad_top" in reference else 0
    if left.shape != (1, 640, 352, 3) or right.shape != left.shape:
        raise ValueError(f"参考输入尺寸错误: {reference_path}")
    timings = []
    prediction = None
    for index in range(runs + 2):
        start = time.perf_counter()
        output = session.inference(inputs=[left, right], data_format=["nhwc", "nhwc"])
        if not output or output[0] is None:
            raise RuntimeError(f"参考推理无输出: {reference_path.name}")
        prediction = np.asarray(output[0]).squeeze().astype(np.float32)
        if prediction.shape != expected.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"参考推理输出无效: {reference_path.name}, {prediction.shape}")
        if index >= 2:
            timings.append((time.perf_counter() - start) * 1000)
    delta = np.abs(prediction[pad_top:, :width] - expected[pad_top:, :width])
    stats = dict(mean=float(delta.mean()), p95=float(np.percentile(delta, 95)),
                 max=float(delta.max()))
    if not (stats["mean"] < 0.2 and stats["p95"] < 0.6 and stats["max"] < 5.0):
        raise RuntimeError(f"{reference_path.name} 数值回归超限: {stats}")
    return dict(reference=reference_path.name, shape=list(prediction.shape), runs=runs,
                inference_ms=dict(median=float(np.median(timings)), p95=float(np.percentile(timings, 95))),
                reference_delta_px=stats)


def self_test(runs=10):
    if runs < 1:
        raise ValueError("--runs 必须至少为 1")
    for path in (MODEL_PATH, RUNTIME_PATH, FORWARD_REFERENCE, REVERSE_REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(f"自检所需包内文件不存在: {path}")
    RKNNLite = _isolated_rknn_class()
    session = None
    try:
        session = RKNNLite(verbose=False)
        if session.load_rknn(str(MODEL_PATH)) != 0:
            raise RuntimeError("RKNN 模型加载失败")
        if session.init_runtime(core_mask=7) != 0:
            raise RuntimeError("RK3588 NPU runtime 初始化失败(core_mask=7)")
        reports = [_reference_report(session, path, runs)
                   for path in (FORWARD_REFERENCE, REVERSE_REFERENCE)]
        mapped = Path("/proc/self/maps")
        if mapped.is_file():
            mappings = mapped.read_text(errors="replace")
            if str(RUNTIME_PATH) not in mappings:
                raise RuntimeError(f"实际进程未加载包内 RKNN runtime: {RUNTIME_PATH}")
        session.release()
        session = None
        metric = MetricDepth(ROOT / "app" / "calibration.npz")
        try:
            left_path, right_path = ROOT / "validation" / "left.png", ROOT / "validation" / "right.png"
            left_raw = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            right_raw = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
            if left_raw is None or right_raw is None:
                raise RuntimeError("validation/left.png 或 right.png 无法读取")
            rectify_start = time.perf_counter()
            left_rect, right_rect = metric.rectify(left_raw, right_raw)
            rectify_ms = (time.perf_counter() - rectify_start) * 1000
            if left_rect.shape != (640, 352) or right_rect.shape != (640, 352):
                raise RuntimeError(f"rectify 尺寸异常: {left_rect.shape}, {right_rect.shape}")
            for rectified, label in ((left_rect, "left"), (right_rect, "right")):
                with np.load(FORWARD_REFERENCE, allow_pickle=False) as data:
                    expected = np.asarray(data[label]).squeeze()
                if not np.array_equal(np.repeat(rectified[..., None], 3, axis=2), expected):
                    raise RuntimeError(f"校正图与参考三通道输入不一致: {label}")
            full_start = time.perf_counter()
            dl, dr = metric.disparities(left_rect, right_rect)
            depth, valid = metric.metric_values(dl, dr, left_rect)
            full_pipeline_ms = (time.perf_counter() - full_start) * 1000
            valid_count = int(valid.sum())
            finite_count = int(np.isfinite(depth).sum())
            if valid_count != 128585 or finite_count != 128585:
                raise RuntimeError(
                    f"并行完整流水线有效点异常: valid={valid_count}, finite={finite_count}, "
                    "expected=128585")
            pipeline = dict(shape=list(depth.shape), valid_count=valid_count,
                            finite_count=finite_count, valid_fraction=float(valid.mean()),
                            lr_consistency_threshold_px=1.0,
                            rectify_ms=rectify_ms, match_filter_ms=full_pipeline_ms,
                            offline_total_ms=rectify_ms+full_pipeline_ms)
        finally:
            metric.close()
        return dict(ok=True, model=str(MODEL_PATH), runtime=str(RUNTIME_PATH),
                    runtime_isolated=True, forward=reports[0], reverse=reports[1],
                    full_pipeline=pipeline)
    finally:
        if session is not None:
            session.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightStereo NPU forward/reverse reference check")
    parser.add_argument("--self-test", action="store_true", help="运行包内正反向参考与性能对照")
    parser.add_argument("--runs", type=int, default=10, help="每份参考预热后计时次数")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("请指定 --self-test")
    try:
        print(json.dumps(self_test(args.runs), ensure_ascii=False, separators=(",", ":")), flush=True)
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=False, separators=(",", ":")), flush=True)
        raise SystemExit(1)
