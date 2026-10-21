"""Offline LightStereo vs SGBM audit on calibration held-out captures.

Run from this directory with the workspace Python environment:
    python -B evaluate.py
Only writes this file's sibling results/ directory.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from stereo.depth import MetricDepth  # noqa: E402


CALIB = ROOT / "stereo" / "calibration" / "current"
CAPTURES = ROOT / "stereo" / "captures"
DETECTIONS_PATH = CALIB / "detections.json"
RESULTS = HERE / "results"
MODELS = HERE / "upstream" / "models"
MODEL_FILES = {
    "opt256x512": MODELS / "lightstereo_s_sceneflow_general_opt_256_512.onnx",
    "opt576x960": MODELS / "lightstereo_s_sceneflow_general_opt_576_960.onnx",
}
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
MAX_BOARD_POINTS = 10000


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _percentiles(values: np.ndarray):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "median": None, "p90": None, "p95": None}
    p = np.percentile(values, [50, 90, 95])
    return {"count": int(values.size), "median": float(p[0]), "p90": float(p[1]), "p95": float(p[2])}


def _prepare(gray: np.ndarray, height: int, width: int):
    """Mirror upstream: CV_8UC3, uniform resize, TOP/RIGHT EDGE padding."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected upright uint8 grayscale input")
    h, w = gray.shape
    scale = min(width / w, height / h)
    resized_w, resized_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top, left = height - resized_h, 0
    bottom, right = 0, width - resized_w
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REPLICATE)
    # Explicitly replicate grayscale to RGB. The per-channel normalization below
    # is the ImageNet normalization expected by the ONNX model and RKNN converter.
    rgb = np.repeat(padded[:, :, None], 3, axis=2)
    nchw_u8 = np.transpose(rgb, (2, 0, 1))[None]
    nchw = (nchw_u8.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
    return nchw, rgb[None], scale, (top, left, resized_h, resized_w)


def _infer_frame(session, gray_left, gray_right):
    shp = session.get_inputs()[0].shape
    h, w = int(shp[2]), int(shp[3])
    x_l, u8_l, scale_l, crop_l = _prepare(gray_left, h, w)
    x_r, u8_r, scale_r, crop_r = _prepare(gray_right, h, w)
    if scale_l != scale_r or crop_l != crop_r:
        raise ValueError("Left/right preprocessing mismatch")
    started = time.perf_counter()
    raw = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: x_l,
        session.get_inputs()[1].name: x_r,
    })[0]
    forward_ms = (time.perf_counter() - started) * 1000
    disp = np.asarray(raw[0, 0], dtype=np.float32)
    top, left, rh, rw = crop_l
    # Upstream postprocess divides by the resize scale, crops the image support,
    # and interpolates back to the original image dimensions.
    disp = disp / scale_l
    disp = disp[top:top + rh, left:left + rw]
    disp = cv2.resize(disp, (gray_left.shape[1], gray_left.shape[0]), interpolation=cv2.INTER_LINEAR)

    started = time.perf_counter()
    rev_raw = session.run([session.get_outputs()[0].name], {
        session.get_inputs()[0].name: _prepare(cv2.flip(gray_right, 1), h, w)[0],
        session.get_inputs()[1].name: _prepare(cv2.flip(gray_left, 1), h, w)[0],
    })[0]
    reverse_ms = (time.perf_counter() - started) * 1000
    rev = np.asarray(rev_raw[0, 0], dtype=np.float32) / scale_l
    rev = rev[top:top + rh, left:left + rw]
    rev = cv2.resize(rev, (gray_left.shape[1], gray_left.shape[0]), interpolation=cv2.INTER_LINEAR)
    rev = cv2.flip(rev, 1)
    return disp, rev, forward_ms, reverse_ms, (u8_l, u8_r, raw[0, 0].astype(np.float32), rw, top)


def _depth_from_disparity(metric: MetricDepth, disp, reverse_disp):
    denom = disp - metric.disparity_offset
    depth = np.full(disp.shape, np.nan, np.float32)
    np.divide(metric.depth_factor, denom, out=depth, where=denom > 0)
    dense = (np.isfinite(depth) & (depth >= metric.RANGE_M[0]) & (depth <= metric.RANGE_M[1])
             & (disp > 0) & metric.masks[0])
    yy, xx = np.indices(disp.shape)
    xr = np.rint(xx - disp).astype(np.int32)
    inside = (xr >= 0) & (xr < metric.width)
    safe = xr.clip(0, metric.width - 1)
    # Require both original camera remap masks at the actual correspondence.
    supported = inside & metric.masks[0] & metric.masks[1][yy, safe]
    support = dense & supported
    # reverse_disp is in the original right-image coordinates after undoing the
    # horizontal flip. Sample at x_right, not at the left pixel coordinate.
    lr_error = np.abs(disp - reverse_disp[yy, safe])
    checked = support & np.isfinite(lr_error) & (lr_error <= 1.0)
    return depth, dense, support, checked, lr_error


def _verify_lr_consistency_logic():
    """Synthetic constant and slanted fields catch wrong-coordinate LR checks."""
    from types import SimpleNamespace
    shape = (4, 100)
    metric = SimpleNamespace(width=shape[1], height=shape[0], depth_factor=20.0,
                             disparity_offset=0.0, RANGE_M=(0.3, 5.0),
                             masks=[np.ones(shape, bool), np.ones(shape, bool)])
    constant = np.full(shape, 4.0, np.float32)
    _, _, support, checked, _ = _depth_from_disparity(metric, constant, constant.copy())
    assert support.any() and np.array_equal(checked, support), "constant field should pass L/R check"
    _, _, _, rejected, _ = _depth_from_disparity(metric, constant, constant + 2.0)
    assert not rejected.any(), "offset reverse field should fail L/R check"
    x = np.arange(shape[1], dtype=np.float32)[None, :]
    slanted = np.broadcast_to(4.0 + 0.5 * x, shape).copy()
    reverse_at_right_x = np.broadcast_to(8.0 + x, shape).copy()
    _, _, slanted_support, slanted_checked, sampled_error = _depth_from_disparity(
        metric, slanted, reverse_at_right_x)
    # At correspondence xR=x-d, reverse(xR)=d(x); same-x comparison differs
    # by several pixels and would incorrectly reject this valid slanted field.
    sampled = slanted_support & (sampled_error <= 1.0)
    assert sampled.any() and np.array_equal(slanted_checked, sampled), "slanted correspondence check failed"
    return {"constant_field_pass": int(checked.sum()), "offset_field_pass_should_be_zero": int(rejected.sum()),
            "slanted_field_pass": int(slanted_checked.sum()), "same_coordinate_would_fail_slanted": True}


def _plane_reference(metric, detections, image_id):
    """PnP plane from raw left checkerboard detections, sampled in upright rectified pixels."""
    capture = next((c for c in detections["captures"] if c["id"] == image_id), None)
    if capture is None or not capture.get("left", {}).get("ok"):
        return None, None
    raw_points = np.asarray(capture["left"]["points"], np.float32).reshape(-1, 2)
    pattern_w, pattern_h = map(int, detections["pattern_size"])
    square_mm = float(detections["square_mm"])
    obj = np.zeros((pattern_h * pattern_w, 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern_w, 0:pattern_h].T.reshape(-1, 2) * square_mm
    model = np.load(CALIB / "parameters.npz", allow_pickle=False)
    ok, rvec, tvec = cv2.solvePnP(obj, raw_points, model["K1"], model["D1"], flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    # Scale the calibrated camera matrices exactly as MetricDepth.rectify does.
    sx = MetricDepth.RAW_SIZE[0] / MetricDepth.SENSOR_SIZE[0]
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sx, (sx - 1) / 2], [0, 0, 1.]], np.float64)
    k_small = resize @ model["K1"]
    p_small = resize @ model["P1"]
    pts_small = raw_points * sx + (sx - 1) / 2
    rect_uv = cv2.undistortPoints(pts_small.reshape(-1, 1, 2), k_small, model["D1"],
                                  R=model["R1"], P=p_small[:, :3]).reshape(-1, 2)
    # Sensor-small image is 640x360; MetricDepth rotates it CCW to 360x640.
    upright = np.column_stack((rect_uv[:, 1], MetricDepth.RAW_SIZE[0] - 1 - rect_uv[:, 0]))
    hull = cv2.convexHull(np.rint(upright).astype(np.int32))
    mask = np.zeros((metric.height, metric.width), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0

    # Transform the PnP board plane into the rectified left camera coordinates.
    r_rect = model["R1"] @ rmat
    t_rect = model["R1"] @ tvec.reshape(3)
    normal = r_rect[:, 2]
    plane_d = -float(normal @ t_rect)
    yy, xx = np.where(mask)
    if not len(xx):
        return None, hull.reshape(-1, 2)
    # Undo the CCW image rotation to coordinates in the 640x360 rectified view.
    u = MetricDepth.RAW_SIZE[0] - 1 - yy.astype(np.float64)
    v = xx.astype(np.float64)
    pix = np.vstack((u, v, np.ones_like(u)))
    rays = np.linalg.inv(p_small[:, :3]) @ pix
    denominator = normal @ rays
    lam = np.full(denominator.shape, np.nan, np.float64)
    np.divide(-plane_d, denominator, out=lam, where=np.abs(denominator) > 1e-9)
    expected = (lam / 1000).astype(np.float32)
    expected_map = np.full((metric.height, metric.width), np.nan, np.float32)
    in_range = (expected >= metric.RANGE_M[0]) & (expected <= metric.RANGE_M[1])
    expected_map[yy, xx] = np.where(in_range, expected, np.nan)
    return expected_map, hull.reshape(-1, 2)


def _board_stats(depth, valid, expected):
    if expected is None:
        return {"n": 0, "coverage": None, "median_abs_error_m": None, "p90_abs_error_m": None}, np.zeros(depth.shape, bool)
    board = np.isfinite(expected)
    measured = board & valid & np.isfinite(depth)
    error = np.abs(depth[measured] - expected[measured])
    return {
        "n": int(board.sum()),
        "valid_n": int(measured.sum()),
        "coverage": float(measured.sum() / max(board.sum(), 1)),
        "median_abs_error_m": float(np.median(error)) if error.size else None,
        "p90_abs_error_m": float(np.percentile(error, 90)) if error.size else None,
        "median_plane_z_m": float(np.nanmedian(expected[board])) if board.any() else None,
        "median_predicted_z_m": float(np.nanmedian(depth[measured])) if measured.any() else None,
    }, board


def _color_depth(depth, bounds=(0.3, 5.0)):
    valid = np.isfinite(depth)
    shade = np.zeros(depth.shape, np.uint8)
    shade[valid] = np.clip((bounds[1] - depth[valid]) / (bounds[1] - bounds[0]) * 255, 0, 255).astype(np.uint8)
    result = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
    result[~valid] = 0
    return result


def _pane(img, label):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 34), (20, 20, 20), -1)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_png(path: Path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"PNG encoding failed: {path}")
    # pathlib.write_bytes handles Unicode Windows paths; cv2.imwrite does not
    # reliably do so on all OpenCV builds.
    path.write_bytes(encoded.tobytes())


def _live_preview(metric, sessions):
    """Read the existing JPEG endpoint only; never call the capture endpoint."""
    try:
        with urlopen("http://127.0.0.1:8877/stereo.jpg", timeout=3) as response:
            jpeg = response.read()
        # The first request may wake a cached JPEG; wait briefly and fetch once
        # more before treating it as a current scene preview.
        time.sleep(0.3)
        with urlopen("http://127.0.0.1:8877/status", timeout=3) as response:
            live_status = json.loads(response.read().decode("utf-8"))
        with urlopen("http://127.0.0.1:8877/stereo.jpg", timeout=3) as response:
            jpeg = response.read()
        mosaic = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
        if mosaic is None or mosaic.shape[0] != 540 or mosaic.shape[1] != 1920:
            return {"status": "skipped", "reason": "JPEG did not have the expected 1920x540 side-by-side layout"}
        left, right = mosaic[:, :960], mosaic[:, 960:]
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # Match the stated preview experiment: restore each 960x540 JPEG eye to
        # the calibration sensor canvas before MetricDepth applies its maps.
        raw_l = cv2.resize(left, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        raw_r = cv2.resize(right, (3840, 2160), interpolation=cv2.INTER_LINEAR)
        rect_l, rect_r = metric.rectify(raw_l, raw_r)
        s_start = time.perf_counter()
        dl, dr = metric.disparities(rect_l, rect_r)
        s_depth, s_valid = metric.metric_values(dl, dr, rect_l)
        s_ms = (time.perf_counter() - s_start) * 1000
        age = live_status.get("preview_age_ms")
        result = {"status": "preview_only_not_calibration_or_ground_truth" if age is not None and age < 1500 else "cached_preview",
                  "url": "http://127.0.0.1:8877/stereo.jpg", "preview_age_ms": age,
                  "left_right_skew_ms": live_status.get("pair_skew_ms", live_status.get("capture_skew_ms")),
                  "jpeg_bytes": len(jpeg), "per_eye_jpeg_wh": [960, 540],
                  "method": "split side-by-side JPEG, resize each eye to 3840x2160, then MetricDepth.rectify",
                  "sgbm_ms": s_ms, "sgbm_coverage": float(s_valid.mean()), "models": {}}
        panes = [_pane(cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR), "Current JPEG left"),
                 _pane(_color_depth(s_depth), "SGBM checked (m)")]
        for name, session in sessions.items():
            disp, rev, forward_ms, reverse_ms, _ = _infer_frame(session, rect_l, rect_r)
            depth, dense, support, checked, lr_error = _depth_from_disparity(metric, disp, rev)
            consistency = float((support & (lr_error <= 1.0)).sum() / max(support.sum(), 1))
            result["models"][name] = {"forward_ms": forward_ms, "reverse_ms": reverse_ms,
                                      "dense_prediction_coverage": float(dense.mean()),
                                      "lr_remap_supported_coverage": float(support.mean()),
                                      "checked_coverage": float(checked.mean()),
                                      "lr_consistency_within_1px": consistency}
            checked_depth = depth.copy()
            checked_depth[~checked] = np.nan
            panes.append(_pane(_color_depth(checked_depth), f"{name} preview checked"))
        _write_png(RESULTS / "current-preview.png", np.hstack(panes))
        return result
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report_file = CALIB / "report.json"
    report_cal = _json(report_file)
    if report_cal.get("status") != "geometry_pass_metric_accuracy_unverified":
        raise ValueError(f"Current calibration is not accepted: {report_cal.get('status')}")
    ids = report_cal["heldout_ids"]
    if len(ids) != 5:
        raise ValueError(f"Expected five held-out captures, got {len(ids)}")
    detections = _json(DETECTIONS_PATH)
    metric = MetricDepth(CALIB / "parameters.npz")
    sessions = {}
    model_info = {}
    for name, path in MODEL_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if len(sess.get_inputs()) != 2 or len(sess.get_outputs()) != 1:
            raise ValueError(f"Unexpected ONNX I/O contract for {name}")
        sessions[name] = sess
        model_info[name] = {
            "file": path.name, "sha256": _sha256(path),
            "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()],
            "output": {"name": sess.get_outputs()[0].name, "shape": sess.get_outputs()[0].shape,
                       "type": sess.get_outputs()[0].type},
        }

    rows = []
    per_model = {name: {"fwd_ms": [], "reverse_ms": [], "dense_coverage": [], "support_coverage": [],
                        "checked_coverage": [], "lr_consistency_1px": [],
                        "lr_median_abs_difference_px": [], "board": []} for name in MODEL_FILES}
    sgbm_stats = {"ms": [], "coverage": [], "board": []}
    representative = None
    reference_saved = False
    reference_576_saved = False

    for image_id in ids:
        capture_dir = CAPTURES / image_id
        raw_l = cv2.imread(str(capture_dir / "left.png"), cv2.IMREAD_GRAYSCALE)
        raw_r = cv2.imread(str(capture_dir / "right.png"), cv2.IMREAD_GRAYSCALE)
        if raw_l is None or raw_r is None:
            raise FileNotFoundError(f"Missing stereo images for {image_id}")
        rect_start = time.perf_counter()
        gray_l, gray_r = metric.rectify(raw_l, raw_r)
        rectify_ms = (time.perf_counter() - rect_start) * 1000
        if gray_l.shape != (640, 360) or gray_r.shape != (640, 360):
            raise ValueError(f"Unexpected rectified shape: {gray_l.shape}, {gray_r.shape}")

        sgbm_start = time.perf_counter()
        dl, dr = metric.disparities(gray_l, gray_r)
        s_depth, s_valid = metric.metric_values(dl, dr, gray_l)
        sgbm_ms = (time.perf_counter() - sgbm_start) * 1000
        expected, hull = _plane_reference(metric, detections, image_id)
        s_board, _ = _board_stats(s_depth, s_valid, expected)
        sgbm_stats["ms"].append(sgbm_ms)
        sgbm_stats["coverage"].append(float(s_valid.mean()))
        sgbm_stats["board"].append(s_board)

        frame = {"id": image_id, "rectify_ms": rectify_ms, "sgbm_ms": sgbm_ms,
                 "sgbm_coverage": float(s_valid.mean()), "sgbm_plane_consistency": s_board,
                 "models": {}}
        model_outputs = {}
        for name, session in sessions.items():
            disp, rev_disp, fwd_ms, rev_ms, reference = _infer_frame(session, gray_l, gray_r)
            depth, dense, support, checked, lr_error = _depth_from_disparity(metric, disp, rev_disp)
            consistency = float((support & (lr_error <= 1.0)).sum() / max(support.sum(), 1))
            lrdiff = lr_error[support]
            board, _ = _board_stats(depth, checked, expected)
            board_dense, _ = _board_stats(depth, dense, expected)
            frame["models"][name] = {
                "forward_ms": fwd_ms, "reverse_ms": rev_ms,
                "dense_prediction_coverage": float(dense.mean()),
                "lr_remap_supported_coverage": float(support.mean()),
                "checked_coverage": float(checked.mean()),
                "lr_consistency_within_1px": consistency,
                "lr_abs_difference_px": _percentiles(lrdiff),
                "plane_consistency_only_not_ground_truth": {"lr_checked": board, "dense_only": board_dense},
            }
            per_model[name]["fwd_ms"].append(fwd_ms)
            per_model[name]["reverse_ms"].append(rev_ms)
            per_model[name]["dense_coverage"].append(float(dense.mean()))
            per_model[name]["support_coverage"].append(float(support.mean()))
            per_model[name]["checked_coverage"].append(float(checked.mean()))
            per_model[name]["lr_consistency_1px"].append(consistency)
            per_model[name]["lr_median_abs_difference_px"].append(float(np.median(lrdiff)) if lrdiff.size else None)
            per_model[name]["board"].append(board)
            model_outputs[name] = (disp, depth, dense, checked)

            if name == "opt256x512" and not reference_saved:
                # Explicit NHWC uint8 pair for matching converter/runtime input;
                # disparity is the raw, pre-crop ONNX output in model coordinates.
                u8_left, u8_right, raw_disp, content_width, pad_top = reference
                np.savez_compressed(RESULTS / "reference.npz",
                                    left=u8_left.astype(np.uint8),
                                    right=u8_right.astype(np.uint8),
                                    disparity=raw_disp.astype(np.float32),
                                    content_width=np.int32(content_width), pad_top=np.int32(pad_top))
                reference_saved = True
            if name == "opt576x960" and not reference_576_saved:
                u8_left, u8_right, raw_disp, content_width, pad_top = reference
                np.savez_compressed(RESULTS / "reference_576.npz",
                                    left=u8_left.astype(np.uint8),
                                    right=u8_right.astype(np.uint8),
                                    disparity=raw_disp.astype(np.float32),
                                    content_width=np.int32(content_width), pad_top=np.int32(pad_top))
                reference_576_saved = True

        rows.append(frame)
        if representative is None:
            representative = (image_id, gray_l, s_depth, model_outputs, hull)

    summary = {
        "method": "five current calibration held-out rectified captures; no deployment or board run",
        "interpretation": "Coverage is support/range availability, not accuracy. Plane errors are internal chessboard plane consistency from solvePnP using this calibration, not independent distance ground truth.",
        "input": {"shape_after_rectify_wh": [360, 640], "source": "MetricDepth.rectify(raw 3840x2160 grayscale)",
                  "model_color": "replicate gray to RGB", "normalization": "(gray/255 - ImageNet RGB mean)/std",
                  "resize": "uniform min scale; OpenCV linear", "padding": "TOP and RIGHT sides with EDGE replicate",
                  "output": "divide disparity by resize scale; remove pad; resize to 360x640",
                  "checked_validity": "depth in range, xright=round(xleft-disparity) in bounds, both remap masks valid at correspondence, and swapped+flipped reverse disparity agrees within 1px"},
        "synthetic_lr_check": _verify_lr_consistency_logic(),
        "calibration": {"file": "stereo/calibration/current/parameters.npz", "id": metric.identifier,
                        "status": report_cal["status"], "heldout_ids": ids},
        "models": model_info,
        "aggregate": {"sgbm": {
            "mean_ms": float(np.mean(sgbm_stats["ms"])), "median_ms": float(np.median(sgbm_stats["ms"])),
            "mean_coverage": float(np.mean(sgbm_stats["coverage"])),
            "mean_plane_consistency_coverage": float(np.mean([x["coverage"] for x in sgbm_stats["board"] if x["coverage"] is not None])) if any(x["coverage"] is not None for x in sgbm_stats["board"]) else None,
            "median_plane_abs_error_m": float(np.median([x["median_abs_error_m"] for x in sgbm_stats["board"] if x["median_abs_error_m"] is not None])) if any(x["median_abs_error_m"] is not None for x in sgbm_stats["board"]) else None,
        }, "models": {}},
        "captures": rows,
    }
    for name, stats in per_model.items():
        board_cov = [b["coverage"] for b in stats["board"] if b["coverage"] is not None]
        board_err = [b["median_abs_error_m"] for b in stats["board"] if b["median_abs_error_m"] is not None]
        summary["aggregate"]["models"][name] = {
            "mean_forward_ms": float(np.mean(stats["fwd_ms"])),
            "median_forward_ms": float(np.median(stats["fwd_ms"])),
            "mean_reverse_ms": float(np.mean(stats["reverse_ms"])),
            "mean_dense_prediction_coverage": float(np.mean(stats["dense_coverage"])),
            "mean_lr_remap_supported_coverage": float(np.mean(stats["support_coverage"])),
            "mean_checked_coverage": float(np.mean(stats["checked_coverage"])),
            "mean_lr_consistency_within_1px": float(np.mean(stats["lr_consistency_1px"])),
            "mean_lr_median_abs_difference_px": float(np.mean([x for x in stats["lr_median_abs_difference_px"] if x is not None])) if any(x is not None for x in stats["lr_median_abs_difference_px"]) else None,
            "mean_plane_consistency_coverage": float(np.mean(board_cov)) if board_cov else None,
            "median_plane_abs_error_m": float(np.median(board_err)) if board_err else None,
        }
    summary["current_preview_jpeg"] = _live_preview(metric, sessions)
    (RESULTS / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if representative is not None:
        image_id, gray, s_depth, model_outputs, hull = representative
        gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panes = [_pane(gray_color, f"Left {image_id[:8]}") , _pane(_color_depth(s_depth), "SGBM checked (m)")]
        for name in MODEL_FILES:
            depth, dense, checked = model_outputs[name][1], model_outputs[name][2], model_outputs[name][3]
            dense_depth = depth.copy()
            dense_depth[~dense] = np.nan
            checked_depth = depth.copy()
            checked_depth[~checked] = np.nan
            short_name = "256" if name == "opt256x512" else "576"
            panes.append(_pane(_color_depth(dense_depth), f"{short_name} dense"))
            panes.append(_pane(_color_depth(checked_depth), f"{short_name} checked"))
        if hull is not None:
            for pane in panes[1:]:
                cv2.polylines(pane, [np.rint(hull).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        _write_png(RESULTS / "comparison.png", np.hstack(panes))
    print(f"Wrote {RESULTS / 'report.json'}")
    print(f"Wrote {RESULTS / 'comparison.png'}")
    print(f"Wrote {RESULTS / 'reference.npz'}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
