"""Low-latency metric stereo backend using calibrated CPU SGBM."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from depth import MetricDepth as BaseDepth


APP = Path(__file__).resolve().parent
ROOT = APP.parent


class FastSGBMDepth(BaseDepth):
    """Half-resolution rectification and SGBM, returning camera-axis metres."""

    ALGORITHM = "Calibrated SGBM CPU (configurable resolution)"

    def __init__(self, path, *, scale=None, num_disparities=None, threads=None,
                 input_size=None):
        super().__init__(path)
        if scale is None:
            scale = float(os.environ.get("LIGHTSTEREO_FAST_SCALE", "0.375"))
        if not 0.25 <= float(scale) <= 1.0:
            raise ValueError("LIGHTSTEREO_FAST_SCALE must be between 0.25 and 1.0")
        self.scale = float(scale)
        if input_size is None:
            input_size = (int(os.environ.get("LIGHTSTEREO_LIVE_WIDTH", "640")),
                          int(os.environ.get("LIGHTSTEREO_LIVE_HEIGHT", "360")))
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2 or min(self.input_size) < 32:
            raise ValueError("The live camera input size must be at least 32x32")
        self.raw_width = int(round(self.RAW_SIZE[0] * self.scale))
        self.raw_height = int(round(self.RAW_SIZE[1] * self.scale))
        self.map_scale = self.raw_width / self.SENSOR_SIZE[0]
        self.crop = max(1, int(round(4 * self.map_scale / (640 / 3840))))
        resize = np.array([[self.map_scale, 0, (self.map_scale - 1) / 2],
                           [0, self.map_scale, (self.map_scale - 1) / 2],
                           [0, 0, 1.0]], dtype=np.float64)
        with np.load(path, allow_pickle=False) as saved:
            model = {key: saved[key].copy() for key in saved.files}
        self.maps = []
        self.small_masks = []
        projections = []
        for index in (1, 2):
            k = resize @ model[f"K{index}"]
            p = resize @ model[f"P{index}"]
            projections.append(p)
            maps = cv2.initUndistortRectifyMap(
                k, model[f"D{index}"], model[f"R{index}"], p,
                (self.raw_width, self.raw_height), cv2.CV_32FC1)
            self.maps.append(maps)
            mask = cv2.remap(np.full((self.raw_height, self.raw_width), 255, np.uint8),
                             *maps, cv2.INTER_LINEAR)
            upright = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.small_masks.append(upright[:, self.crop:-self.crop] > 254)

        self.depth_factor = -float(projections[1][1, 3]) / 1000.0
        self.disparity_offset = float(projections[0][1, 2] - projections[1][1, 2])
        self.width = self.raw_height - 2 * self.crop
        self.height = self.raw_width
        self.yy, self.xx = np.indices((self.height, self.width))
        self.auto_range = None
        self.filter_stats = {}
        self.inference_timing = {}
        self.rectify_timing = {}
        self.filter_timing = {}
        if threads is None:
            threads = int(os.environ.get("LIGHTSTEREO_FAST_THREADS", "2"))
        cv2.setNumThreads(max(1, int(threads)))
        # At 640x360 input, the board profile showed CPU resize/remap had lower
        # and steadier latency than the OpenCL upload/download path. Keep OpenCL
        # available as an explicit override for BSPs with a faster UMat path.
        requested_preprocess = os.environ.get("LIGHTSTEREO_FAST_PREPROCESS", "cpu").strip().lower()
        self.preprocess_backend = "cpu"
        self.opencl_error = None
        self.maps_umat = None
        if requested_preprocess == "opencl" and cv2.ocl.haveOpenCL():
            try:
                cv2.ocl.setUseOpenCL(True)
                self.maps_umat = [(cv2.UMat(x), cv2.UMat(y)) for x, y in self.maps]
                probe = cv2.UMat(np.zeros((16, 16), dtype=np.uint8))
                if cv2.resize(probe, (8, 8), interpolation=cv2.INTER_AREA).get().shape != (8, 8):
                    raise RuntimeError("OpenCL resize probe returned an invalid shape")
                self.preprocess_backend = "opencl"
            except Exception as exc:
                self.opencl_error = str(exc)
                self.maps_umat = None
                cv2.ocl.setUseOpenCL(False)
        else:
            cv2.ocl.setUseOpenCL(False)

        required = int(math.ceil(self.disparity_offset + self.depth_factor / self.RANGE_M[0] + 4))
        calculated = max(16, int(math.ceil(required / 16.0)) * 16)
        requested = num_disparities
        if requested is None and os.environ.get("LIGHTSTEREO_FAST_DISPARITIES"):
            requested = int(os.environ["LIGHTSTEREO_FAST_DISPARITIES"])
        self.num_disparities = calculated if requested is None else int(requested)
        self.num_disparities = max(16, int(math.ceil(self.num_disparities / 16.0)) * 16)
        maximum = ((self.width - 16) // 16) * 16
        if maximum < 16:
            raise ValueError("Rectified image is too narrow for SGBM")
        self.num_disparities = min(self.num_disparities, maximum)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=5,
            P1=200,
            P2=800,
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=8,
            speckleWindowSize=40,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        window = np.ones((self.matcher.getBlockSize(), self.matcher.getBlockSize()),
                         dtype=np.uint8)
        self.window_masks = tuple(
            cv2.erode(mask.astype(np.uint8), window,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            for mask in self.small_masks)
        self.algorithm = f"Calibrated SGBM CPU ({self.height}x{self.width})"

    def rectify(self, left, right):
        result = []
        resize_ms = 0.0
        rectify_ms = 0.0
        for gray, maps in zip((left, right), self.maps):
            if (gray.shape not in (self.SENSOR_SIZE[::-1], self.input_size[::-1])
                    or gray.dtype != np.uint8):
                raise ValueError("深度输入必须为标定原图或当前 NV12 灰度流分辨率")
            if self.preprocess_backend == "opencl":
                started = time.perf_counter()
                source = cv2.UMat(gray)
                small = cv2.resize(source, (self.raw_width, self.raw_height),
                                   interpolation=cv2.INTER_AREA)
                remapped = cv2.remap(small, *self.maps_umat[len(result)], cv2.INTER_LINEAR)
                upright = cv2.rotate(remapped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                output = upright.get()
                result.append(np.ascontiguousarray(output[:, self.crop:-self.crop]))
                # OpenCL queues these kernels asynchronously; report their synchronized
                # upload-to-download wall time as one combined preprocessing stage.
                rectify_ms += (time.perf_counter() - started) * 1000
            else:
                started = time.perf_counter()
                small = cv2.resize(gray, (self.raw_width, self.raw_height),
                                   interpolation=cv2.INTER_AREA)
                resize_ms += (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                remapped = cv2.remap(small, *maps, cv2.INTER_LINEAR)
                upright = cv2.rotate(remapped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                result.append(np.ascontiguousarray(upright[:, self.crop:-self.crop]))
                rectify_ms += (time.perf_counter() - started) * 1000
        self.rectify_timing = dict(resize_ms=resize_ms, remap_rotate_ms=rectify_ms)
        return result

    def disparities(self, left, right):
        started = time.perf_counter()
        pad = self.num_disparities
        # SGBM drops its full search-width border; metric_values checks the
        # resulting matches against the original, unpadded image support.
        left_padded = cv2.copyMakeBorder(left, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        right_padded = cv2.copyMakeBorder(right, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        disparity = self.matcher.compute(left_padded, right_padded).astype(np.float32) / 16.0
        disparity = disparity[:, pad:pad + left.shape[1]]
        self.inference_timing = {"forward_ms": (time.perf_counter() - started) * 1000,
                                 "reverse_ms": 0.0}
        return disparity, None

    def metric_values(self, disparity, _reverse, _left):
        filter_start = time.perf_counter()
        d = np.asarray(disparity, dtype=np.float32)
        denominator = d - self.disparity_offset
        depth = np.full(d.shape, np.nan, dtype=np.float32)
        matched = np.isfinite(d) & (d > self.disparity_offset + 0.01)
        xr = self.xx - d
        finite_xr = np.where(np.isfinite(xr), xr, 0)
        xr0 = np.floor(finite_xr).astype(np.int32)
        xr1 = np.ceil(finite_xr).astype(np.int32)
        right_in_bounds = (xr0 >= 0) & (xr1 < self.width)
        xr0 = np.clip(xr0, 0, self.width - 1)
        xr1 = np.clip(xr1, 0, self.width - 1)
        right_support = (self.window_masks[1][self.yy, xr0]
                         & self.window_masks[1][self.yy, xr1])
        valid = matched & self.window_masks[0] & right_in_bounds & right_support
        np.divide(self.depth_factor, denominator, out=depth,
                  where=valid & np.isfinite(denominator) & (denominator > 0))
        valid &= np.isfinite(depth) & (depth >= self.RANGE_M[0]) & (depth <= self.RANGE_M[1])

        # Drop isolated matches while keeping thin object edges for click-to-measure.
        support = cv2.boxFilter(valid.astype(np.uint8), cv2.CV_16U, (3, 3),
                                normalize=False, borderType=cv2.BORDER_CONSTANT)
        valid &= support >= 3
        depth[~valid] = np.nan
        self.filter_timing = {"consistency_ms": 0.0,
                              "depth_filter_ms": (time.perf_counter() - filter_start) * 1000}
        self.filter_stats = dict(matched_fraction=float(matched.mean()),
                                 consistent_fraction=float(valid.mean()),
                                 textured_fraction=float(valid.mean()),
                                 range_fraction=float(valid.mean()),
                                 consistency_method="SGBM disp12MaxDiff=1 plus 3x3 support and real 5x5 stereo windows")
        return depth, valid

    def compute(self, left, right):
        color, depth, details = super().compute(left, right)
        details["algorithm"] = self.algorithm
        details["npu_execution"] = {"mode": "idle; fast CPU SGBM path"}
        details["preprocess_backend"] = self.preprocess_backend
        details["opencl_error"] = self.opencl_error
        if self.preprocess_backend == "opencl":
            details["stage_ms"]["resize"] = None
            details["stage_ms"]["remap_rotate_crop"] = round(
                self.rectify_timing["remap_rotate_ms"], 2)
            details["stage_ms"]["preprocess_opencl_combined"] = round(
                self.rectify_timing["remap_rotate_ms"], 2)
        details["depth_resolution"] = [self.width, self.height]
        details["stage_ms"]["inference_forward"] = round(
            self.inference_timing.get("forward_ms", 0.0), 2)
        details["stage_ms"]["inference_reverse"] = 0.0
        return color, depth, details


def _checkerboard_accuracy(depth: np.ndarray, raw_left: np.ndarray, calibration_path: Path):
    """Compare a known held-out target patch; this is not an independent range truth."""
    small = cv2.resize(raw_left, (640, 360), interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(
        small, (9, 6), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        small, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
    with np.load(calibration_path, allow_pickle=False) as saved:
        k, d, r, p = (saved[name].copy() for name in ("K1", "D1", "R1", "P1"))
    points = cv2.undistortPoints(corners * 6.0, k, d, R=r, P=p).reshape(-1, 2)
    scale = depth.shape[0] / 3840.0
    transform = np.array([[scale, 0, (scale - 1) / 2],
                          [0, scale, (scale - 1) / 2],
                          [0, 0, 1]], dtype=np.float64)
    rectified = points @ transform[:2, :2].T + transform[:2, 2]
    crop = max(1, int(round(4 * depth.shape[0] / 640.0)))
    polygon = np.rint(np.column_stack((rectified[:, 1] - crop,
                                      depth.shape[0] - 1 - rectified[:, 0]))).astype(np.int32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, x1 = np.percentile(xs, [25, 75])
    y0, y1 = np.percentile(ys, [25, 75])
    yy, xx = np.indices(depth.shape)
    core = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (mask > 0)
    values = depth[core & np.isfinite(depth)]
    if values.size < 20:
        return None
    measurement = dict(depth_p10_p50_p90_m=[float(v) for v in np.percentile(values, [10, 50, 90])],
                       valid_pixels=int(values.size))
    metadata_path = ROOT / "validation" / "meta.json"
    report_path = ROOT / "evidence" / "calibration-report.json"
    if metadata_path.is_file() and report_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record = next(item for group in ("training_metrics", "heldout_metrics")
                          for item in report.get(group, [])
                          if item.get("id") == metadata.get("id"))
            reference = float(record["board_distance_mm"]) / 1000.0
            measured = measurement["depth_p10_p50_p90_m"][1]
            measurement.update(calibration_estimated_target_m=reference,
                               median_difference_mm=round((measured - reference) * 1000, 1),
                               target_id=metadata["id"],
                               scope="calibration-derived target estimate; not an external tape/laser truth")
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
            pass
    return measurement


def self_test(runs=12):
    if runs < 1:
        raise ValueError("--runs must be >= 1")
    left_path, right_path = ROOT / "validation" / "left.png", ROOT / "validation" / "right.png"
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("validation/left.png and right.png are required")
    engine = FastSGBMDepth(APP / "calibration.npz")
    live_left = cv2.resize(left, engine.input_size, interpolation=cv2.INTER_AREA)
    live_right = cv2.resize(right, engine.input_size, interpolation=cv2.INTER_AREA)
    try:
        totals = []
        stage_totals = {name: [] for name in ("resize", "remap_rotate_crop",
                                                "preprocess_opencl_combined", "match",
                                                "depth_filter", "colorize")}
        valid_counts = []
        last_depth = None
        for index in range(runs + 2):
            started = time.perf_counter()
            _color, depth, details = engine.compute(live_left, live_right)
            last_depth = depth
            elapsed = (time.perf_counter() - started) * 1000
            if depth.shape != (engine.height, engine.width):
                raise RuntimeError(f"Unexpected depth shape: {depth.shape}")
            valid_count = int(np.isfinite(depth).sum())
            if valid_count < max(500, depth.size // 30):
                raise RuntimeError(f"Too few valid metric pixels: {valid_count}")
            if index >= 2:
                totals.append(elapsed)
                valid_counts.append(valid_count)
                for name in stage_totals:
                    value = details["stage_ms"].get(name)
                    if value is not None:
                        stage_totals[name].append(float(value))
        percentile = lambda values, q: float(np.percentile(values, q))
        return dict(ok=True, algorithm=engine.algorithm, runs=runs,
                    simulated_input_resolution=list(engine.input_size),
                    simulated_input_source="saved 4K pair resized to match tested GStreamer/ISP NV12 caps",
                    resolution=[engine.width, engine.height],
                    num_disparities=engine.num_disparities,
                    total_ms={"p50": percentile(totals, 50), "p95": percentile(totals, 95)},
                    fps_from_total_p50=1000.0 / percentile(totals, 50),
                    stage_ms={name: ({"p50": percentile(values, 50),
                                      "p95": percentile(values, 95)} if values else None)
                              for name, values in stage_totals.items()},
                    valid_pixels_p50=int(percentile(valid_counts, 50)),
                    valid_fraction_p50=percentile(valid_counts, 50) / (engine.width * engine.height),
                    calibration_id=engine.identifier,
                    baseline_mm=engine.baseline_mm,
                    checkerboard=_checkerboard_accuracy(
                        last_depth, left, APP / "calibration.npz"),
                    accuracy_scope="single saved pair; compare against a known target before safety use")
    finally:
        del engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the low-latency calibrated SGBM path")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--runs", type=int, default=12)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("specify --self-test")
    try:
        print(json.dumps(self_test(args.runs), ensure_ascii=False, separators=(",", ":")), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
