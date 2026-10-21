"""Compare SGBM, padded 576x960, and native 640x352 on shared portrait crops."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate as audit  # shared calibration, plane, and reporting helpers
from portrait import CroppedMetric, verify_geometry

CALIB = ROOT / "stereo" / "calibration" / "current"
CAPTURES = ROOT / "stereo" / "captures"
RESULTS = HERE / "results"
MODELS = HERE / "upstream" / "models"
NATIVE_MODEL = HERE / "models" / "lightstereo_640_352.onnx"


def _run(session, left, right, native=False):
    in0, in1 = session.get_inputs()
    h, w = map(int, in0.shape[-2:])
    if native:
        if (h, w) != (640, 352) or left.shape != (640, 352):
            raise ValueError("Native branch requires the unscaled 352x640 crop")
        def prep(gray):
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            nchw = np.transpose(rgb[None], (0, 3, 1, 2)).astype(np.float32) / 255.
            return (nchw-audit.RGB_MEAN)/audit.RGB_STD, rgb[None]
        x_l, u8_l = prep(left)
        x_r, u8_r = prep(right)
        sy = sx = 1.0
        top = 0
    else:
        def prep(gray):
            h0, w0 = gray.shape
            scale = min(w/w0, h/h0)
            rw, rh = int(w0*scale), int(h0*scale)
            resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_LINEAR)
            top = h-rh
            padded = cv2.copyMakeBorder(resized, top, 0, 0, w-rw, cv2.BORDER_REPLICATE)
            rgb = np.repeat(padded[:, :, None], 3, axis=2)
            nchw = np.transpose(rgb[None], (0, 3, 1, 2)).astype(np.float32)/255.
            return (nchw-audit.RGB_MEAN)/audit.RGB_STD, rgb[None], rw/w0, rh/h0, top, rw, rh
        x_l, u8_l, sx, sy, top, rw, rh = prep(left)
        x_r, u8_r, sx2, sy2, top2, rw2, rh2 = prep(right)
        if (sx, sy, top, rw, rh) != (sx2, sy2, top2, rw2, rh2):
            raise ValueError("Left/right preprocessing mismatch")

    def infer(a, b):
        t0 = time.perf_counter()
        raw = session.run([session.get_outputs()[0].name], {in0.name: a, in1.name: b})[0]
        return np.asarray(raw[0, 0], np.float32), (time.perf_counter()-t0)*1000

    raw, fwd = infer(x_l, x_r)
    if native:
        disp = raw.copy()
    else:
        disp = raw[top:top+rh, :rw] / sx
        disp = cv2.resize(disp, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Swap the eyes and horizontally flip each to obtain reverse disparity in
    # original right-image coordinates, sampling later at x_right.
    if native:
        xr, ur = prep(cv2.flip(right, 1))
        xl, ul = prep(cv2.flip(left, 1))
        rev_raw, rev_ms = infer(xr, xl)
        reverse = cv2.flip(rev_raw, 1)
    else:
        xr, ur, rsx, rsy, rtop, rrw, rrh = prep(cv2.flip(right, 1))
        xl, ul, lsx, lsy, ltop, lrw, lrh = prep(cv2.flip(left, 1))
        if (rsx, rsy, rtop, rrw, rrh) != (lsx, lsy, ltop, lrw, lrh):
            raise ValueError("Reverse preprocessing mismatch")
        rev_raw, rev_ms = infer(xr, xl)
        reverse = rev_raw[rtop:rtop+rrh, :rrw] / rsx
        reverse = cv2.resize(reverse, (right.shape[1], right.shape[0]), interpolation=cv2.INTER_LINEAR)
        reverse = cv2.flip(reverse, 1)
    reference = (u8_l, u8_r, raw, left.shape[1], top)
    return disp, reverse, fwd, rev_ms, reference, rev_raw


def _metrics(metric, disp, reverse, expected):
    depth, dense, support, checked, lr_error = audit._depth_from_disparity(metric, disp, reverse)
    board, _ = audit._board_stats(depth, checked, expected)
    return depth, checked, {
        "dense_coverage": float(dense.mean()),
        "supported_coverage": float(support.mean()),
        "checked_coverage": float(checked.mean()),
        "lr_consistency_within_1px": float(checked.sum()/max(support.sum(), 1)),
        "board_plane_consistency_only_not_ground_truth": board,
    }


def _common_board_stats(depths, valids, expected):
    if expected is None:
        return {name: {"common_valid_n": 0, "median_abs_error_m": None, "p90_abs_error_m": None}
                for name in depths}
    board = np.isfinite(expected)
    common = board.copy()
    for valid in valids.values():
        common &= np.asarray(valid, dtype=bool)
    n = int(common.sum())
    if n == 0:
        raise AssertionError("No common valid checkerboard pixels across all methods")
    result = {}
    for name, depth in depths.items():
        if not np.isfinite(depth[common]).all():
            raise AssertionError(f"Nonfinite depth in common valid pixels: {name}")
        error = np.abs(depth[common]-expected[common])
        if not np.isfinite(error).all():
            raise AssertionError(f"Nonfinite common checkerboard error: {name}")
        result[name] = {"common_valid_n": n,
                        "median_abs_error_m": float(np.median(error)),
                        "p90_abs_error_m": float(np.percentile(error, 90))}
    return result


def _distance_legend(width, bounds):
    height = 48
    out = np.zeros((height, width, 3), np.uint8)
    bar_w, bar_h = min(280, width-40), 14
    x0, y0 = (width-bar_w)//2, 21
    shades = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :]
    colors = cv2.applyColorMap(shades, cv2.COLORMAP_TURBO)
    out[y0:y0+bar_h, x0:x0+bar_w] = colors
    far_text = f"far  {bounds[1]:.2f} m"
    near_text = f"near  {bounds[0]:.2f} m"
    cv2.putText(out, far_text, (x0, 15), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (235, 235, 235), 1, cv2.LINE_AA)
    size = cv2.getTextSize(near_text, cv2.FONT_HERSHEY_SIMPLEX, .42, 1)[0]
    cv2.putText(out, near_text, (x0+bar_w-size[0], 15), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(out, f"display range {bounds[0]:.2f}-{bounds[1]:.2f} m",
                (x0, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cal_report = audit._json(CALIB/"report.json")
    if cal_report.get("status") != "geometry_pass_metric_accuracy_unverified":
        raise ValueError("Current calibration is not accepted")
    ids = cal_report["heldout_ids"]
    if len(ids) != 5:
        raise ValueError(f"Expected five held-out captures, got {len(ids)}")
    metric = CroppedMetric(CALIB/"parameters.npz")
    geometry = verify_geometry(CALIB/"parameters.npz")
    model_paths = {
        "576x960 padded": MODELS/"lightstereo_s_sceneflow_general_opt_576_960.onnx",
        "640x352 native": NATIVE_MODEL,
    }
    sessions, model_info = {}, {}
    for name, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        sessions[name] = session
        model_info[name] = {"file": path.name, "sha256": audit._sha256(path),
                            "input": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
                            "output": {"name": session.get_outputs()[0].name,
                                       "shape": session.get_outputs()[0].shape}}

    original_metric = audit.MetricDepth(CALIB/"parameters.npz")
    det = audit._json(CALIB/"detections.json")
    captures, first = [], None
    ref_saved = False
    for image_id in ids:
        folder = CAPTURES/image_id
        raw_l = cv2.imread(str(folder/"left.png"), cv2.IMREAD_GRAYSCALE)
        raw_r = cv2.imread(str(folder/"right.png"), cv2.IMREAD_GRAYSCALE)
        if raw_l is None or raw_r is None:
            raise FileNotFoundError(f"Missing stereo images for {image_id}")
        start = time.perf_counter()
        full_l, full_r = original_metric.rectify(raw_l, raw_r)
        rectify_ms = (time.perf_counter()-start)*1000
        left, right = full_l[:, 4:-4], full_r[:, 4:-4]
        if left.shape != (640, 352) or right.shape != left.shape:
            raise ValueError(f"Unexpected crop shape {left.shape}")
        # Plane reference must be formed in the original 360x640 coordinates,
        # then shifted into the cropped frame.
        full_expected, hull = audit._plane_reference(original_metric, det, image_id)
        expected = full_expected[:, 4:-4].copy() if full_expected is not None else None

        t0 = time.perf_counter()
        dl, dr = metric.disparities(left, right)
        s_depth, s_valid = metric.metric_values(dl, dr, left)
        sgbm_ms = (time.perf_counter()-t0)*1000
        s_board, _ = audit._board_stats(s_depth, s_valid, expected)
        s_stats = {"dense_coverage": float(s_valid.mean()),
                   "supported_coverage": float(metric.filter_stats["consistent_fraction"]),
                   "checked_coverage": float(s_valid.mean()),
                   "lr_consistency_within_1px": float(metric.filter_stats["consistent_fraction"] / max(metric.filter_stats["matched_fraction"], 1e-12)),
                   "board_plane_consistency_only_not_ground_truth": s_board,
                   "mean_ms": sgbm_ms}
        methods = {"SGBM": (s_depth, s_valid, s_stats, sgbm_ms)}
        frame_models = {}
        for name, session in sessions.items():
            disp, reverse, fwd, rev, reference, reverse_raw = _run(
                session, left, right, native=name.startswith("640x352"))
            depth, checked, stats = _metrics(metric, disp, reverse, expected)
            stats.update(forward_ms=fwd, reverse_ms=rev)
            frame_models[name] = stats
            methods[name] = (depth, checked, stats, fwd)
            if not ref_saved and name.startswith("640x352"):
                u8_l, u8_r, raw_disp, content_width, pad_top = reference
                forward_common = dict(left=u8_l.astype(np.uint8), right=u8_r.astype(np.uint8),
                                      content_width=np.int32(352), pad_top=np.int32(0))
                np.savez_compressed(RESULTS/"portrait-reference.npz", **forward_common,
                                    disparity=raw_disp.astype(np.float32))
                # Keep the swapped, flipped inputs and raw output in network
                # coordinates so NPU output can be compared directly.
                rev_left = np.flip(u8_r, axis=2).copy()
                rev_right = np.flip(u8_l, axis=2).copy()
                reverse_common = dict(left=rev_left, right=rev_right,
                                      content_width=np.int32(352), pad_top=np.int32(0))
                np.savez_compressed(RESULTS/"portrait-reference-reverse.npz", **reverse_common,
                                    disparity=reverse_raw.astype(np.float32))
                ref_saved = True

        common_errors = _common_board_stats(
            {k: v[0] for k, v in methods.items()}, {k: v[1] for k, v in methods.items()}, expected)
        for name, error in common_errors.items():
            methods[name][2]["common_valid_board_points"] = error
        captures.append({"id": image_id, "rectify_ms": rectify_ms,
                         "sgbm": s_stats, "models": frame_models})
        if first is None:
            first = (image_id, methods)

    def median_or_none(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.median(values)) if values else None

    def mean_or_none(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else None

    aggregate = {}
    for name in ["SGBM", *model_paths]:
        items = [c["sgbm"] if name == "SGBM" else c["models"][name] for c in captures]
        aggregate[name] = {
            "mean_checked_coverage": float(np.mean([i["checked_coverage"] for i in items])),
            "mean_board_coverage": mean_or_none([i["board_plane_consistency_only_not_ground_truth"]["coverage"] for i in items]),
            "median_board_abs_error_m": median_or_none([i["board_plane_consistency_only_not_ground_truth"]["median_abs_error_m"] for i in items]),
            "mean_common_valid_board_points": mean_or_none([i["common_valid_board_points"]["common_valid_n"] for i in items]),
            "median_common_valid_abs_error_m": median_or_none([i["common_valid_board_points"]["median_abs_error_m"] for i in items]),
        }
        if name != "SGBM":
            aggregate[name]["mean_forward_ms"] = float(np.mean([i["forward_ms"] for i in items]))
            aggregate[name]["mean_reverse_ms"] = float(np.mean([i["reverse_ms"] for i in items]))
        else:
            aggregate[name]["mean_ms"] = float(np.mean([i["mean_ms"] for i in items]))

    image_id, methods = first
    pooled_depth = np.concatenate([
        depth[valid & np.isfinite(depth)].astype(np.float64)
        for depth, valid, _, _ in methods.values()
        if np.any(valid & np.isfinite(depth))
    ])
    p05, p95 = np.percentile(pooled_depth, [5, 95]).tolist()
    margin = max((p95-p05)*0.05, 0.02)
    color_bounds = [max(metric.RANGE_M[0], p05-margin), min(metric.RANGE_M[1], p95+margin)]
    if color_bounds[1] <= color_bounds[0]:
        color_bounds = [max(metric.RANGE_M[0], p05-0.02), min(metric.RANGE_M[1], p95+0.02)]

    report = {
        "method": "five held-out captures; same rectified 352x640 crop; SGBM vs existing padded 576x960 vs native 640x352",
        "interpretation": "Plane consistency uses solvePnP with this calibration and is not independent distance ground truth. Coverage measures valid checked pixels.",
        "geometry_verification": geometry,
        "input": {"upright_shape_wh": [352, 640], "crop_left_right_px": [4, 4], "external_padding": 0,
                  "native_preprocess": "gray replicated to RGB, ImageNet normalization, no resize or padding",
                  "576_preprocess": "uniform resize, TOP/RIGHT EDGE replicate padding; disparity divided by actual rounded horizontal resize ratio before restoring to 352x640",
                  "disparity_validity": "depth 0.3-5m; xright=round(xleft-disparity); both rectification masks at correspondence; reverse disparity sampled at xright within 1px"},
        "calibration_id": metric.identifier, "heldout_ids": ids,
        "comparison_visualization": {"capture_id": image_id,
                                      "pooled_valid_depth_p05_p95_m": [float(p05), float(p95)],
                                      "shared_color_range_m": [float(v) for v in color_bounds],
                                      "range_margin_m": float(margin),
                                      "valid_depth_count": int(pooled_depth.size),
                                      "mapping": "Turbo; near is red, far is dark purple; invalid pixels are black"},
        "models": model_info, "aggregate": aggregate, "captures": captures,
    }
    (RESULTS/"portrait-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    panes = []
    for name in ("SGBM", "576x960 padded", "640x352 native"):
        depth, valid = methods[name][0], methods[name][1]
        view = depth.copy()
        view[~valid] = np.nan
        panes.append(audit._pane(audit._color_depth(view, tuple(color_bounds)), name))
    comparison = np.vstack([np.hstack(panes), _distance_legend(352*len(panes), color_bounds)])
    audit._write_png(RESULTS/"portrait-comparison.png", comparison)
    print(json.dumps({"aggregate": aggregate, "reference_saved": ref_saved}, indent=2))


if __name__ == "__main__":
    main()
