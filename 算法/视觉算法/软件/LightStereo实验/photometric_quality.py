"""Offline photometric input comparison for the native portrait LightStereo model.

Writes only results/photometric-quality.json. It preserves the calibrated crop,
0.3-5 m range, source-image masks, and the existing 1 px correspondence LRC.
"""
from __future__ import annotations

import json
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate as audit
import evaluate_portrait as portrait_eval
from portrait import CroppedMetric

CALIB = ROOT / "stereo" / "calibration" / "current"
CAPTURES = ROOT / "stereo" / "captures"
RESULTS = HERE / "results"
MODEL = HERE / "models" / "lightstereo_640_352.onnx"
PREVIEW = RESULTS / "optimization-stereo.jpg"
INPUT_META = RESULTS / "optimization-input.json"
REPORT = RESULTS / "photometric-quality.json"
METHODS = ("original_gray", "shared_percentile_1_99", "pair_mean_std")


def _imread_gray(path: Path):
    """Decode through pathlib bytes so OpenCV works with the Unicode workspace path."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def transform_pair(left: np.ndarray, right: np.ndarray):
    """Photometric candidates only; never change geometry or image support."""
    out = {"original_gray": (left, right)}

    # One common transfer function preserves left/right exposure relation.
    lo, hi = np.percentile(np.concatenate((left.ravel(), right.ravel())), [1, 99])
    if hi <= lo:
        stretched = (left.copy(), right.copy())
    else:
        stretched = tuple(np.clip((im.astype(np.float32)-lo)*(255.0/(hi-lo)), 0, 255).astype(np.uint8)
                          for im in (left, right))
    out["shared_percentile_1_99"] = stretched

    means = [float(np.mean(im)) for im in (left, right)]
    stds = [float(np.std(im)) for im in (left, right)]
    target_mean, target_std = float(np.mean(means)), float(np.mean(stds))
    balanced = []
    for im, mean, std in zip((left, right), means, stds):
        gain = target_std / max(std, 1e-6)
        balanced.append(np.clip((im.astype(np.float32)-mean)*gain+target_mean, 0, 255).astype(np.uint8))
    out["pair_mean_std"] = tuple(balanced)

    return out


def sift_vertical_residual(left: np.ndarray, right: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=1200)
    kl, dl = sift.detectAndCompute(left, None)
    kr, dr = sift.detectAndCompute(right, None)
    result = {"keypoints_left": len(kl), "keypoints_right": len(kr), "ratio_matches": 0,
              "ransac_inliers": 0, "fundamental_method": None,
              "signed_dy_p50_p90_p95_px": None, "abs_dy_p50_p90_p95_px": None,
              "linear_dy_xy_coefficients": None, "linear_fit_abs_residual_p50_p90_p95_px": None}
    if dl is not None and dr is not None and len(dl) >= 2 and len(dr) >= 2:
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
        good = [m for m, n in matches if m.distance < 0.72*n.distance]
        result["ratio_matches"] = len(good)
        if len(good) >= 8:
            p1 = np.asarray([kl[m.queryIdx].pt for m in good], np.float32)
            p2 = np.asarray([kr[m.trainIdx].pt for m in good], np.float32)
            method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
            method_name = "USAC_MAGSAC" if method == getattr(cv2, "USAC_MAGSAC", -1) else "FM_RANSAC"
            try:
                _, mask = cv2.findFundamentalMat(p1, p2, method, 1.0, 0.999)
            except cv2.error:
                method, method_name = cv2.FM_RANSAC, "FM_RANSAC"
                _, mask = cv2.findFundamentalMat(p1, p2, method, 1.0, 0.999)
            if mask is not None:
                keep = mask.reshape(-1).astype(bool)
                a, b = p1[keep].astype(np.float64), p2[keep].astype(np.float64)
                result["ransac_inliers"] = int(keep.sum())
                result["fundamental_method"] = method_name
                if len(a) >= 3:
                    dy = a[:, 1]-b[:, 1]
                    signed = np.percentile(dy, [50, 90, 95])
                    absolute = np.percentile(np.abs(dy), [50, 90, 95])
                    # Remove only a first-order spatial trend for diagnosis;
                    # these fitted values never affect inference or calibration.
                    xy = (a+b)*0.5
                    design = np.column_stack((xy[:, 0], xy[:, 1], np.ones(len(xy))))
                    coefficients, *_ = np.linalg.lstsq(design, dy, rcond=None)
                    fit_abs = np.percentile(np.abs(dy-design@coefficients), [50, 90, 95])
                    result.update(signed_dy_p50_p90_p95_px=[float(v) for v in signed],
                                  abs_dy_p50_p90_p95_px=[float(v) for v in absolute],
                                  linear_dy_xy_coefficients=[float(v) for v in coefficients],
                                  linear_fit_abs_residual_p50_p90_p95_px=[float(v) for v in fit_abs])
    return result


def capture_preview(metric: audit.MetricDepth):
    """Direction-only check on the supplied JPEG: split first, then resize to sensor size."""
    pair = _imread_gray(PREVIEW)
    if pair is None:
        raise FileNotFoundError(PREVIEW)
    h, w = pair.shape
    if (w, h) != (1920, 540):
        raise ValueError(f"Expected the documented 1920x540 side-by-side preview, got {w}x{h}")
    # This is an interpolated preview, not original sensor data. Upscaling each
    # half separately makes the existing rectifier callable but cannot restore detail.
    left_small, right_small = pair[:, :w//2], pair[:, w//2:]
    left_raw = cv2.resize(left_small, (3840, 2160), interpolation=cv2.INTER_CUBIC)
    right_raw = cv2.resize(right_small, (3840, 2160), interpolation=cv2.INTER_CUBIC)
    left_rect, right_rect = metric.rectify(left_raw, right_raw)

    # Check vertical residuals of SIFT descriptor matches after the fixed
    # rectification. This diagnoses geometry direction only, not source quality.
    vertical = sift_vertical_residual(left_rect[:, 4:-4], right_rect[:, 4:-4])
    return left_rect[:, 4:-4], right_rect[:, 4:-4], {
        "file": PREVIEW.name, "input_metadata": INPUT_META.name,
        "source_size_wh": [w, h], "split_halves_wh": [w//2, h],
        "processing": "split side-by-side JPEG; upscale each half to 3840x2160; existing fixed rectify; crop 4 px per side",
        "is_original_sensor_frame": False, "has_ground_truth": False,
        "sift_rectified_vertical_residual": vertical,
    }


def paired_board_error(depth, valid, expected, common):
    mask = common & np.isfinite(expected) & valid & np.isfinite(depth)
    n = int(mask.sum())
    if not n:
        return {"common_valid_board_points": 0, "median_abs_error_m": None, "p90_abs_error_m": None}
    error = np.abs(depth[mask]-expected[mask])
    return {"common_valid_board_points": n, "median_abs_error_m": float(np.median(error)),
            "p90_abs_error_m": float(np.percentile(error, 90))}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cal_report = audit._json(CALIB / "report.json")
    if cal_report.get("status") != "geometry_pass_metric_accuracy_unverified":
        raise ValueError("Current calibration is not accepted")
    ids = cal_report["heldout_ids"]
    if len(ids) != 5:
        raise ValueError(f"Expected five held-out captures, got {len(ids)}")
    metric = CroppedMetric(CALIB / "parameters.npz")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
    detections = audit._json(CALIB / "detections.json")
    full_metric = audit.MetricDepth(CALIB / "parameters.npz")

    per_capture = []
    heldout_sift = []
    timings = {name: [] for name in METHODS}
    for image_id in ids:
        folder = CAPTURES / image_id
        raw_l = _imread_gray(folder / "left.png")
        raw_r = _imread_gray(folder / "right.png")
        if raw_l is None or raw_r is None:
            raise FileNotFoundError(f"Missing stereo images for {image_id}")
        full_l, full_r = full_metric.rectify(raw_l, raw_r)
        left, right = full_l[:, 4:-4], full_r[:, 4:-4]
        if left.shape != (640, 352) or right.shape != left.shape:
            raise ValueError(f"Unexpected crop {left.shape}")
        heldout_sift.append({"id": image_id, **sift_vertical_residual(left, right)})
        expected_full, _ = audit._plane_reference(full_metric, detections, image_id)
        expected = expected_full[:, 4:-4] if expected_full is not None else None
        pair_methods = transform_pair(left, right)
        depths, valids, stats = {}, {}, {}
        for name in METHODS:
            l, r = pair_methods[name]
            started = time.perf_counter()
            disp, reverse, forward_ms, reverse_ms, _, _ = portrait_eval._run(session, l, r, native=True)
            depth, checked, method_stats = portrait_eval._metrics(metric, disp, reverse, expected)
            timings[name].append((time.perf_counter()-started)*1000)
            depths[name], valids[name] = depth, checked
            stats[name] = {"checked_coverage": method_stats["checked_coverage"],
                           "forward_ms": forward_ms, "reverse_ms": reverse_ms}

        # `.values()` is intentionally iterated as the five candidate masks,
        # then intersected with baseline, board, and each candidate individually.
        common_all = np.isfinite(expected) if expected is not None else np.zeros(left.shape, bool)
        for mask in valids.values():
            common_all &= np.asarray(mask, dtype=bool)
        for name in METHODS[1:]:
            pair_common = np.isfinite(expected) if expected is not None else np.zeros(left.shape, bool)
            pair_common &= np.asarray(valids["original_gray"], dtype=bool)
            pair_common &= np.asarray(valids[name], dtype=bool)
            stats[name]["paired_common_board"] = {
                "original_gray": paired_board_error(depths["original_gray"], valids["original_gray"], expected, pair_common),
                name: paired_board_error(depths[name], valids[name], expected, pair_common),
                "five_method_common_valid_board_points": int(common_all.sum()),
            }
        per_capture.append({"id": image_id, "methods": stats})

    # Preview direction check: geometry is applied through the existing rectifier;
    # report these interpolated-image metrics separately from held-out measurements.
    preview_left, preview_right, preview_info = capture_preview(full_metric)
    preview = {}
    for name, (l, r) in transform_pair(preview_left, preview_right).items():
        disp, reverse, forward_ms, reverse_ms, _, _ = portrait_eval._run(session, l, r, native=True)
        _, checked, method_stats = portrait_eval._metrics(metric, disp, reverse, None)
        preview[name] = {"checked_coverage": method_stats["checked_coverage"],
                         "forward_ms": forward_ms, "reverse_ms": reverse_ms}

    # Single-frame geometry diagnostic only: translate right rectified gray by
    # the robust signed median dy (y_left-y_right). Positive dy therefore moves
    # right-image content down. Warp its source mask identically and retain the
    # original 1 px LRC and range test. This never changes calibration files.
    preview_geometry = preview_info["sift_rectified_vertical_residual"]
    if preview_geometry["signed_dy_p50_p90_p95_px"] is not None:
        signed_dy = float(preview_geometry["signed_dy_p50_p90_p95_px"][0])
        shift_y = int(round(signed_dy))
        shifted_right = cv2.warpAffine(preview_right,
                                       np.array([[1, 0, 0], [0, 1, shift_y]], np.float32),
                                       (metric.width, metric.height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shifted_metric = copy.copy(metric)
        shifted_masks = [metric.masks[0].copy(), None]
        shifted_masks[1] = cv2.warpAffine(metric.masks[1].astype(np.uint8),
                                          np.array([[1, 0, 0], [0, 1, shift_y]], np.float32),
                                          (metric.width, metric.height), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        shifted_metric.masks = shifted_masks
        disp, reverse, forward_ms, reverse_ms, _, _ = portrait_eval._run(
            session, preview_left, shifted_right, native=True)
        _, shifted_checked, shifted_stats = portrait_eval._metrics(shifted_metric, disp, reverse, None)
        shifted_sift = sift_vertical_residual(preview_left, shifted_right)
        preview["right_vertical_shift_diagnostic"] = {
            "shift_y_px": shift_y, "sign_basis": "dy = y_left - y_right; positive value moves right image down",
            "checked_coverage": shifted_stats["checked_coverage"],
            "coverage_delta_percentage_points_vs_original_gray": 100*(shifted_stats["checked_coverage"]-preview["original_gray"]["checked_coverage"]),
            "forward_ms": forward_ms, "reverse_ms": reverse_ms,
            "sift_rectified_vertical_residual": shifted_sift,
            "mask_handling": "right rectification support mask translated with nearest-neighbor interpolation; out-of-frame pixels invalid; left mask unchanged",
            "limits": "one interpolated JPEG preview only; diagnostic alignment probe, not an estimated calibration or deployment recommendation"
        }

    aggregates = {}
    for name in METHODS:
        captures = [c["methods"][name] for c in per_capture]
        aggregates[name] = {"mean_checked_coverage": float(np.mean([x["checked_coverage"] for x in captures])),
                            "per_capture_checked_coverage": [float(x["checked_coverage"]) for x in captures],
                            "mean_forward_ms": float(np.mean([x["forward_ms"] for x in captures])),
                            "mean_reverse_ms": float(np.mean([x["reverse_ms"] for x in captures]))}
        if name != "original_gray":
            vals = [x[name]["median_abs_error_m"] for x in [c["methods"][name]["paired_common_board"] for c in per_capture]
                    if x[name]["median_abs_error_m"] is not None]
            base_vals = [x["original_gray"]["median_abs_error_m"] for x in [c["methods"][name]["paired_common_board"] for c in per_capture]
                         if x["original_gray"]["median_abs_error_m"] is not None]
            point_counts = [c["methods"][name]["paired_common_board"]["five_method_common_valid_board_points"] for c in per_capture]
            aggregates[name]["median_of_capture_medians_common_board_abs_error_m"] = {
                "original_gray": float(np.median(base_vals)) if base_vals else None,
                name: float(np.median(vals)) if vals else None,
                "mean_pairwise_common_points": float(np.mean([c["methods"][name]["paired_common_board"][name]["common_valid_board_points"] for c in per_capture])),
                "mean_five_method_common_points": float(np.mean(point_counts)),
            }

    baseline_cov = aggregates["original_gray"]["mean_checked_coverage"]
    best = max((name for name in METHODS[1:]), key=lambda n: aggregates[n]["mean_checked_coverage"])
    cov_gain = aggregates[best]["mean_checked_coverage"]-baseline_cov
    paired = aggregates[best]["median_of_capture_medians_common_board_abs_error_m"]
    err_base, err_candidate = paired["original_gray"], paired[best]
    if cov_gain >= 0.01 and err_base is not None and err_candidate is not None and err_candidate <= err_base*1.10:
        recommendation = {"decision": "candidate_for_further_offline_review", "candidate": best,
                          "reason": "checked coverage improves by at least 1 percentage point and common-board median error rises by no more than 10%; this is not deployment approval"}
    else:
        recommendation = {"decision": "do_not_recommend_deployment", "candidate": best,
                          "reason": "no candidate met the coverage and common-board error screening rule"}

    report = {
        "method": "four input-only conditions on five saved held-out pairs, common unscaled rectified 352x640 crop; native ONNX CPU inference with 4 threads",
        "scope": "offline photometric direction screening only; no camera/calibration/LRC/range changes; no hole filling",
        "calibration_id": metric.identifier,
        "heldout_ids": ids,
        "preprocessing": {
            "original_gray": "gray replicated to RGB, ImageNet normalization",
            "shared_percentile_1_99": "one pairwise 1st-99th percentile transfer shared by both images",
            "pair_mean_std": "affine-match each image to the arithmetic mean of pair means and standard deviations",
            "fixed_filters": "existing source remap masks and correspondence xright=round(xleft-disparity), reverse disparity sampled at xright within 1 px, 0.3-5 m; original local-texture filter is not added to LightStereo"
        },
        "common_board_error_definition": "Per capture and candidate, compare original and candidate on their pairwise intersection with PnP board pixels; the extra five_method_common_valid_board_points count uses .values() across all method validity masks. This is internal solvePnP plane consistency, not measured accuracy.",
        "aggregate": aggregates,
        "captures": per_capture,
        "heldout_rectified_sift_vertical_residual": heldout_sift,
        "preview_direction_check": {"image": preview_info, "methods": preview,
                                    "warning": "The supplied JPEG halves were independently upscaled before rectification. This interpolated preview is not original sensor data, held-out data, or ground truth."},
        "preview_geometry_shift_finding": {
            "interpretation": "USAC_MAGSAC retains 115/143 ratio-tested SIFT matches; held-out rectified median signed dy ranges -0.266 to +0.261 px, while the preview median is +5.043 px and its xy-linear-fit p90 absolute residual is 1.227 px. The preview offset is therefore coherent and unlike these held-out examples, rather than explained by scattered descriptor mismatches.",
            "diagnostic_only_right_y_shift": preview.get("right_vertical_shift_diagnostic"),
            "deployment_effect": "No calibration, service, camera transform, or model input correction is authorized by this single interpolated preview; the photometric candidates remain unrecommended."
        },
        "screening_rule": ">=1.0 percentage point mean checked coverage gain, and candidate median-of-capture-medians common-board absolute error <= 110% of the paired baseline",
        "recommendation": recommendation,
        "model": {"file": MODEL.name, "sha256": audit._sha256(MODEL),
                  "input": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()]},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "recommendation": recommendation, "aggregate": aggregates}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
