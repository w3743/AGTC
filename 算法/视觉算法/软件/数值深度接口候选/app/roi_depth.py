"""Experimental full-resolution ROI stereo depth for already-rectified pairs.

This is an independent experiment, not the existing 5 m depth profile. The
caller owns camera synchronization, rectification, calibration validity, and
any physical-accuracy assessment. Invalid output depth is NaN and is always
paired with an explicit boolean validity mask.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from numbers import Real
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np


class ROIDepthError(ValueError):
    """The pair, calibration, search range, or ROI is not usable."""


@dataclass(frozen=True)
class ROIDepthResult:
    """Depth and coordinate metadata for one independently computed ROI."""

    depth_z_m: np.ndarray
    valid: np.ndarray
    disparity_px: np.ndarray
    roi_xywh: Tuple[int, int, int, int]
    K_source: np.ndarray
    K_roi: np.ndarray
    roi_to_source: np.ndarray
    timings_ms: dict[str, Optional[float]]
    compute_geometry: dict[str, Any]
    depth_range_m: Tuple[float, float]
    physical_accuracy_claim_m: None = None


def _numeric_matrix(value: Any, label: str) -> np.ndarray:
    raw = value.tolist() if isinstance(value, np.ndarray) else value
    try:
        if isinstance(raw, (list, tuple)):
            values = np.asarray(raw, dtype=object).reshape(-1).tolist()
            if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
                raise ValueError("matrix values must be real numbers")
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ROIDepthError(f"{label} must contain finite real numbers") from exc
    return matrix


def _validate_intrinsics(value: Any) -> np.ndarray:
    k = _numeric_matrix(value, "K")
    if k.shape == (9,):
        k = k.reshape(3, 3)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(k[0, 1]) > 1e-9 or abs(k[1, 0]) > 1e-9
            or not np.allclose(k[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)):
        raise ROIDepthError("K must be a finite rectified pinhole matrix with positive fx/fy")
    return k


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ROIDepthError(f"{label} must be an integer")
    return int(value)


def _parse_roi(roi_xywh: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    if not isinstance(roi_xywh, (tuple, list, np.ndarray)) or len(roi_xywh) != 4:
        raise ROIDepthError("roi_xywh must be (x, y, width, height)")
    x, y, roi_w, roi_h = (_exact_int(v, "ROI coordinates") for v in roi_xywh)
    if x < 0 or y < 0 or roi_w <= 0 or roi_h <= 0 or x + roi_w > width or y + roi_h > height:
        raise ROIDepthError("ROI must be a non-empty rectangle inside the rectified frame")
    return x, y, roi_w, roi_h


class ROISGBMDepth:
    """Reusable SGBM matcher that computes a padded canvas around one left ROI.

    ``fB_px_m`` is f_px * baseline_m. Z is computed as
    ``fB_px_m / (disparity_px - disparity_offset_px)``. ``K`` is the calibrated
    intrinsic matrix of the already-rectified full raster; it is transformed to
    ROI-local pixel coordinates in each result.

    The canvas contains the left ROI plus the complete right correspondence
    interval ``x_right = x_left - d`` for the configured disparity search, with
    block and context margins. Synthetic left padding is added only when the
    cropped ROI would otherwise fall inside SGBM's left search border; strict
    masks and original-frame bounds reject any correspondence that lands in
    padding or outside the physical source.
    SGBM's semi-global paths are bounded by this canvas, so this is an
    experimental ROI computation rather than a bit-identical crop of a full
    frame SGBM result. The forward matcher input area is reported. With the
    default explicit LR check, a second SGBM pass runs on the full rectified
    canvas (plus search padding); its input area and preparation/compute time
    are reported separately, so ROI timing does not hide this cost.
    """

    def __init__(
        self,
        *,
        K: Sequence[float],
        fB_px_m: float,
        min_disparity: int,
        num_disparities: int,
        disparity_offset_px: float = 0.0,
        block_size: int = 5,
        depth_range_m: Tuple[float, float] = (0.3, 50.0),
        context_px: Optional[int] = None,
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        disp12_max_diff: int = 1,
        uniqueness_ratio: int = 8,
        speckle_window_size: int = 40,
        speckle_range: int = 2,
        lr_consistency: bool = True,
        lr_max_diff_px: float = 1.0,
    ) -> None:
        self.K = _validate_intrinsics(K)
        if isinstance(fB_px_m, bool) or not isinstance(fB_px_m, Real):
            raise ROIDepthError("fB_px_m must be a finite positive real number")
        self.fB_px_m = float(fB_px_m)
        if not math.isfinite(self.fB_px_m) or self.fB_px_m <= 0:
            raise ROIDepthError("fB_px_m must be a finite positive real number")
        if isinstance(disparity_offset_px, bool) or not isinstance(disparity_offset_px, Real):
            raise ROIDepthError("disparity_offset_px must be finite")
        self.disparity_offset_px = float(disparity_offset_px)
        if not math.isfinite(self.disparity_offset_px):
            raise ROIDepthError("disparity_offset_px must be finite")
        self.min_disparity = _exact_int(min_disparity, "min_disparity")
        self.num_disparities = _exact_int(num_disparities, "num_disparities")
        if self.num_disparities < 16 or self.num_disparities % 16:
            raise ROIDepthError("num_disparities must be a positive multiple of 16, at least 16")
        self.max_disparity_exclusive = self.min_disparity + self.num_disparities
        self.max_disparity = self.max_disparity_exclusive - 1
        self.block_size = _exact_int(block_size, "block_size")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ROIDepthError("block_size must be an odd integer >= 3")
        self.radius = self.block_size // 2
        near, far = depth_range_m
        if (isinstance(near, bool) or isinstance(far, bool)
                or not isinstance(near, Real) or not isinstance(far, Real)):
            raise ROIDepthError("depth_range_m must contain two finite positive numbers")
        self.depth_range_m = (float(near), float(far))
        if (not all(math.isfinite(v) and v > 0 for v in self.depth_range_m)
                or self.depth_range_m[0] >= self.depth_range_m[1]):
            raise ROIDepthError("depth_range_m must be finite, positive, and increasing")
        if context_px is None:
            self.context_px = self.block_size * 2
        else:
            self.context_px = _exact_int(context_px, "context_px")
            if self.context_px < 0:
                raise ROIDepthError("context_px must be non-negative")

        default_p1 = 8 * self.block_size * self.block_size
        default_p2 = 32 * self.block_size * self.block_size
        self.p1 = default_p1 if p1 is None else _exact_int(p1, "p1")
        self.p2 = default_p2 if p2 is None else _exact_int(p2, "p2")
        if self.p1 <= 0 or self.p2 <= self.p1:
            raise ROIDepthError("SGBM penalties require 0 < p1 < p2")
        if not isinstance(lr_consistency, bool):
            raise ROIDepthError("lr_consistency must be bool")
        self.lr_consistency = lr_consistency
        if isinstance(lr_max_diff_px, bool) or not isinstance(lr_max_diff_px, Real):
            raise ROIDepthError("lr_max_diff_px must be finite and non-negative")
        self.lr_max_diff_px = float(lr_max_diff_px)
        if not math.isfinite(self.lr_max_diff_px) or self.lr_max_diff_px < 0:
            raise ROIDepthError("lr_max_diff_px must be finite and non-negative")
        matcher_start = time.perf_counter()
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=_exact_int(disp12_max_diff, "disp12_max_diff"),
            preFilterCap=31,
            uniquenessRatio=_exact_int(uniqueness_ratio, "uniqueness_ratio"),
            speckleWindowSize=_exact_int(speckle_window_size, "speckle_window_size"),
            speckleRange=_exact_int(speckle_range, "speckle_range"),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.matcher_create_ms = (time.perf_counter() - matcher_start) * 1000.0

    def compute(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_valid: np.ndarray,
        right_valid: np.ndarray,
        roi_xywh: Sequence[int],
        *,
        full_frame_rectification_ms: Optional[float] = None,
    ) -> ROIDepthResult:
        """Compute only one ROI canvas from a pre-rectified grayscale pair.

        ``full_frame_rectification_ms`` is an optional measurement supplied by
        the caller. It remains a separate timing and is never included in the
        adapter's own total or reported SGBM compute time.
        """
        total_start = time.perf_counter()
        validate_start = total_start
        left = np.asarray(left_gray)
        right = np.asarray(right_gray)
        if (left.ndim != 2 or right.shape != left.shape or left.dtype != np.uint8
                or right.dtype != np.uint8 or left.size == 0):
            raise ROIDepthError("left/right must be same-size non-empty uint8 rectified grayscale images")
        height, width = left.shape
        lmask = np.asarray(left_valid)
        rmask = np.asarray(right_valid)
        if (lmask.dtype != np.bool_ or rmask.dtype != np.bool_
                or lmask.shape != left.shape or rmask.shape != left.shape):
            raise ROIDepthError("left_valid/right_valid must be bool masks matching the input HxW")
        roi = _parse_roi(roi_xywh, width, height)
        x, y, roi_w, roi_h = roi
        if full_frame_rectification_ms is not None:
            if (isinstance(full_frame_rectification_ms, bool)
                    or not isinstance(full_frame_rectification_ms, Real)
                    or not math.isfinite(float(full_frame_rectification_ms))
                    or float(full_frame_rectification_ms) < 0):
                raise ROIDepthError("full_frame_rectification_ms must be non-negative or null")
            rectification_ms: Optional[float] = float(full_frame_rectification_ms)
        else:
            rectification_ms = None
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        validate_ms = (time.perf_counter() - validate_start) * 1000.0

        # Include both left block support and every possible right block support
        # for x_right=x_left-d. Expand again for SGBM path context. Bounds here
        # are in the original rectified frame, before synthetic padding.
        right_x_min = x - self.max_disparity
        right_x_max = x + roi_w - 1 - self.min_disparity
        support_x_min = min(x, right_x_min) - self.radius
        support_x_max_exclusive = max(x + roi_w - 1, right_x_max) + self.radius + 1
        crop_x0 = max(0, support_x_min - self.context_px)
        crop_x1 = min(width, support_x_max_exclusive + self.context_px)
        crop_y0 = max(0, y - self.radius - self.context_px)
        crop_y1 = min(height, y + roi_h + self.radius + self.context_px)
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise ROIDepthError("ROI search canvas is empty")

        prepare_start = time.perf_counter()
        left_crop = left[crop_y0:crop_y1, crop_x0:crop_x1]
        right_crop = right[crop_y0:crop_y1, crop_x0:crop_x1]
        # Like the fast path, pad left by a full disparity range so the matcher
        # does not discard low-x ROI candidates at its search border. Right pad
        # also accommodates a configured negative-disparity search.
        roi_before_padding_x = x - crop_x0
        pad_left = max(0, self.min_disparity + self.num_disparities - roi_before_padding_x)
        pad_right = max(self.radius, -self.min_disparity + self.radius)
        pad_top = pad_bottom = 0
        left_canvas = cv2.copyMakeBorder(
            left_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
        right_canvas = cv2.copyMakeBorder(
            right_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
        roi_local_x = pad_left + x - crop_x0
        roi_local_y = pad_top + y - crop_y0
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        match_start = time.perf_counter()
        disparity_canvas = self.matcher.compute(left_canvas, right_canvas).astype(np.float32) / 16.0
        matcher_compute_ms = (time.perf_counter() - match_start) * 1000.0
        disparity = np.ascontiguousarray(
            disparity_canvas[roi_local_y:roi_local_y + roi_h,
                             roi_local_x:roi_local_x + roi_w]
        )

        reverse_disparity = None
        reverse_prepare_ms: Optional[float] = None
        reverse_matcher_compute_ms: Optional[float] = None
        reverse_geometry = None
        if self.lr_consistency:
            # Compute a reverse positive disparity on the complete rectified
            # input canvas: flip(R) is its left image, flip(L) its right image.
            # This allows LR checks to sample every physical right coordinate
            # corresponding to the forward ROI, without confusing the local
            # forward crop's synthetic padding with source pixels.
            reverse_prepare_start = time.perf_counter()
            reverse_pad_left = max(0, self.min_disparity + self.num_disparities)
            reverse_pad_right = max(self.radius, -self.min_disparity + self.radius)
            reverse_left = np.ascontiguousarray(right[:, ::-1])
            reverse_right = np.ascontiguousarray(left[:, ::-1])
            reverse_left_canvas = cv2.copyMakeBorder(
                reverse_left, 0, 0, reverse_pad_left, reverse_pad_right,
                cv2.BORDER_CONSTANT, value=0,
            )
            reverse_right_canvas = cv2.copyMakeBorder(
                reverse_right, 0, 0, reverse_pad_left, reverse_pad_right,
                cv2.BORDER_CONSTANT, value=0,
            )
            reverse_prepare_ms = (time.perf_counter() - reverse_prepare_start) * 1000.0

            reverse_start = time.perf_counter()
            reverse_canvas = self.matcher.compute(
                reverse_left_canvas, reverse_right_canvas
            ).astype(np.float32) / 16.0
            reverse_matcher_compute_ms = (time.perf_counter() - reverse_start) * 1000.0
            reverse_padded_slice = reverse_canvas[:, reverse_pad_left:reverse_pad_left + width]
            reverse_disparity = np.ascontiguousarray(reverse_padded_slice[:, ::-1])
            reverse_geometry = {
                "scope": "full_rectified_canvas",
                "input_shape_hw": list(reverse_left_canvas.shape),
                "input_area_px": int(reverse_left_canvas.shape[0] * reverse_left_canvas.shape[1]),
                "padding_ltrb": [reverse_pad_left, 0, reverse_pad_right, 0],
            }

        filter_start = time.perf_counter()
        kernel = np.ones((self.block_size, self.block_size), dtype=np.uint8)
        left_window_valid = cv2.erode(
            lmask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)
        right_window_valid = cv2.erode(
            rmask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(bool)

        source_x = x + np.arange(roi_w, dtype=np.float32)[None, :]
        source_y = y + np.arange(roi_h, dtype=np.int32)[:, None]
        raw_d = disparity
        matched = (np.isfinite(raw_d)
                   & (raw_d >= float(self.min_disparity))
                   & (raw_d < float(self.max_disparity_exclusive)))
        denominator = raw_d - self.disparity_offset_px
        z = np.full((roi_h, roi_w), np.nan, dtype=np.float32)
        positive_denominator = np.isfinite(denominator) & (denominator > 0.0)
        np.divide(self.fB_px_m, denominator, out=z,
                  where=matched & positive_denominator)
        z_in_range = np.isfinite(z) & (z >= self.depth_range_m[0]) & (z <= self.depth_range_m[1])

        # Check right support in original-image coordinates after disparity
        # search, never against synthetic canvas pixels or clipped indices.
        xr = source_x - raw_d
        finite_xr = np.isfinite(xr)
        xr0 = np.zeros(xr.shape, dtype=np.int32)
        xr1 = np.zeros(xr.shape, dtype=np.int32)
        xr0[finite_xr] = np.floor(xr[finite_xr]).astype(np.int32)
        xr1[finite_xr] = np.ceil(xr[finite_xr]).astype(np.int32)
        right_in_bounds = finite_xr & (xr0 >= 0) & (xr1 < width)
        # Safe lookup indices are used only behind the explicit bounds mask.
        safe_x0 = np.clip(xr0, 0, width - 1)
        safe_x1 = np.clip(xr1, 0, width - 1)
        yy = np.broadcast_to(source_y, (roi_h, roi_w))
        left_support = left_window_valid[y:y + roi_h, x:x + roi_w]
        right_support = (right_window_valid[yy, safe_x0]
                         & right_window_valid[yy, safe_x1])
        valid = matched & positive_denominator & z_in_range & left_support & right_in_bounds & right_support

        lr_filter_ms: Optional[float] = None
        if self.lr_consistency:
            lr_start = time.perf_counter()
            assert reverse_disparity is not None
            reverse_matched = (
                np.isfinite(reverse_disparity)
                & (reverse_disparity >= float(self.min_disparity))
                & (reverse_disparity < float(self.max_disparity_exclusive))
            )
            reverse_x_left = (np.arange(width, dtype=np.float32)[None, :]
                              + reverse_disparity)
            reverse_finite_x = np.isfinite(reverse_x_left)
            reverse_x0 = np.zeros(reverse_x_left.shape, dtype=np.int32)
            reverse_x1 = np.zeros(reverse_x_left.shape, dtype=np.int32)
            reverse_x0[reverse_finite_x] = np.floor(
                reverse_x_left[reverse_finite_x]
            ).astype(np.int32)
            reverse_x1[reverse_finite_x] = np.ceil(
                reverse_x_left[reverse_finite_x]
            ).astype(np.int32)
            reverse_in_bounds = (reverse_finite_x & (reverse_x0 >= 0)
                                 & (reverse_x1 < width))
            safe_reverse_x0 = np.clip(reverse_x0, 0, width - 1)
            safe_reverse_x1 = np.clip(reverse_x1, 0, width - 1)
            reverse_y = np.broadcast_to(
                np.arange(height, dtype=np.int32)[:, None], (height, width)
            )
            # In reverse orientation, the left support is the original right
            # mask and the right correspondence is the original left mask.
            reverse_left_support = right_window_valid
            reverse_right_support = (
                left_window_valid[reverse_y, safe_reverse_x0]
                & left_window_valid[reverse_y, safe_reverse_x1]
            )
            reverse_valid = (reverse_matched & reverse_in_bounds
                             & reverse_left_support & reverse_right_support)

            reverse_d0 = reverse_disparity[yy, safe_x0]
            reverse_d1 = reverse_disparity[yy, safe_x1]
            reverse_ok0 = reverse_valid[yy, safe_x0]
            reverse_ok1 = reverse_valid[yy, safe_x1]
            lr_consistent = (
                right_in_bounds & reverse_ok0 & reverse_ok1
                & (np.abs(raw_d - reverse_d0) <= self.lr_max_diff_px)
                & (np.abs(raw_d - reverse_d1) <= self.lr_max_diff_px)
            )
            valid &= lr_consistent
            lr_filter_ms = (time.perf_counter() - lr_start) * 1000.0
        z[~valid] = np.nan
        disparity = disparity.astype(np.float32, copy=False)
        disparity[~valid] = np.nan
        filter_ms = (time.perf_counter() - filter_start) * 1000.0

        roi_to_source = np.array([[1.0, 0.0, float(x)],
                                  [0.0, 1.0, float(y)],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
        K_roi = np.array([[1.0, 0.0, -float(x)],
                          [0.0, 1.0, -float(y)],
                          [0.0, 0.0, 1.0]], dtype=np.float64) @ self.K
        total_ms = (time.perf_counter() - total_start) * 1000.0
        K_source = self.K.copy()
        for array in (z, valid, disparity, roi_to_source, K_roi, K_source):
            array.setflags(write=False)
        clipped_right_min = max(0, right_x_min)
        clipped_right_max = min(width - 1, right_x_max)
        clipped_right_support = (
            [clipped_right_min, clipped_right_max]
            if clipped_right_min <= clipped_right_max else None
        )
        compute_geometry = {
            "source_shape_hw": [height, width],
            "roi_xywh": list(roi),
            "roi_output_shape_hw": [roi_h, roi_w],
            "matcher_source_crop_xywh": [crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0],
            "matcher_padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
            "matcher_input_shape_hw": list(left_canvas.shape),
            "matcher_input_area_px": int(left_canvas.shape[0] * left_canvas.shape[1]),
            "full_source_area_px": int(width * height),
            "right_search_support_x_inclusive": [right_x_min, right_x_max],
            "right_search_support_x_clipped": clipped_right_support,
            "min_disparity": self.min_disparity,
            "max_disparity_exclusive": self.max_disparity_exclusive,
            "fB_px_m": self.fB_px_m,
            "disparity_offset_px": self.disparity_offset_px,
            "depth_range_m": list(self.depth_range_m),
            "block_size": self.block_size,
            "context_px": self.context_px,
            "input_is_already_rectified": True,
            "padding_is_synthetic_and_invalid": True,
            "sgm_context_is_roi_bounded": True,
            "lr_consistency_enabled": self.lr_consistency,
            "lr_max_diff_px": self.lr_max_diff_px,
            "lr_sample_rule": (
                "both floor/ceil reverse samples at x_right=x_left-d must be valid "
                "and within threshold"
            ) if self.lr_consistency else None,
            "reverse_match": reverse_geometry,
        }
        timings = {
            "full_frame_rectification_ms": rectification_ms,
            "matcher_create_ms_once": self.matcher_create_ms,
            "input_validation_ms": validate_ms,
            "roi_canvas_prepare_ms": prepare_ms,
            "matcher_compute_ms": matcher_compute_ms,
            "reverse_canvas_prepare_ms": reverse_prepare_ms,
            "reverse_matcher_compute_ms": reverse_matcher_compute_ms,
            "lr_consistency_filter_ms": lr_filter_ms,
            "mask_and_depth_ms": filter_ms,
            "roi_total_ms_excluding_rectification": total_ms,
        }
        return ROIDepthResult(
            depth_z_m=z,
            valid=valid,
            disparity_px=disparity,
            roi_xywh=roi,
            K_source=K_source,
            K_roi=K_roi,
            roi_to_source=roi_to_source,
            timings_ms=timings,
            compute_geometry=compute_geometry,
            depth_range_m=self.depth_range_m,
            physical_accuracy_claim_m=None,
        )

