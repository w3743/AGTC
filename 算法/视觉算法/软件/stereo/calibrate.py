"""Offline calibration of frozen screen-checkerboard captures; never deploys.

Run from the 算法/视觉算法/软件 directory.
Usage: python -B stereo/calibrate.py detect|fit|validate|batch [--output DIR]
All pixel errors and saved parameters use raw 3840x2160 sensor coordinates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "calibration"
SIZE = (3840, 2160)
PATTERN = (9, 6)
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
cv2.setNumThreads(4)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ordered_corners(gray):
    upright = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    small = cv2.resize(upright, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, points = cv2.findChessboardCornersSB(small, PATTERN, flags)
    if not ok:
        small = cv2.createCLAHE(2., (8, 8)).apply(small)
        ok, points = cv2.findChessboardCornersSB(small, PATTERN, flags)
    if not ok:
        return {"ok": False, "reason": "complete_9x6_grid_not_found"}
    points = points * 2 + .5
    cv2.cornerSubPix(cv2.GaussianBlur(upright, (0, 0), .8), points,
                     (7, 7), (-1, -1), (3, 50, .01))
    grid = points.reshape(6, 9, 2)
    u = np.median(np.diff(grid, axis=1).reshape(-1, 2), axis=0)
    v = np.median(np.diff(grid, axis=0).reshape(-1, 2), axis=0)
    if u[0] * v[1] - u[1] * v[0] < 0:
        grid = grid[::-1].copy()
    # A 10x7 board with a black top-left cell has an unambiguous color phase.
    # Use local quadrilateral centers, avoiding a homography's distortion bias.
    centers = (grid[:-1, :-1] + grid[1:, :-1] + grid[:-1, 1:] + grid[1:, 1:]) / 4
    colors = []
    for x, y in centers.reshape(-1, 2):
        x, y = int(round(float(x))), int(round(float(y)))
        colors.append(float(np.median(upright[y-2:y+3, x-2:x+3])))
    colors = np.array(colors).reshape(5, 8)
    parity = np.indices((5, 8)).sum(axis=0) % 2
    contrast = float(np.median(colors[parity == 1]) - np.median(colors[parity == 0]))
    if abs(contrast) < 35:
        return {"ok": False, "reason": "checker_color_phase_ambiguous", "contrast": contrast}
    if contrast < 0:
        grid = grid[::-1, ::-1].copy()
    sensor = np.stack((SIZE[0] - 1 - grid[..., 1], grid[..., 0]), axis=-1).reshape(-1, 2)
    return {"ok": True, "contrast": abs(contrast), "points": sensor.tolist()}


def detect():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8-sig"))
    result = {"image_size": SIZE, "pattern_size": PATTERN, "square_mm": manifest["square_mm"],
              "coordinates": "unrotated sensor; board first inner corner adjacent to black top-left cell",
              "captures": []}
    for i, meta in enumerate(manifest["captures"]):
        row = {"id": meta["id"], "skew_ms": meta["skew_ms"]}
        for eye in ("left", "right"):
            path = ROOT / "captures" / meta["id"] / (eye + ".png")
            raw = path.read_bytes()
            gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            assert gray.shape == SIZE[::-1], (path, gray.shape)
            row[eye] = ordered_corners(gray)
            row[eye]["sha256"] = hashlib.sha256(raw).hexdigest()
        row["ok"] = row["left"]["ok"] and row["right"]["ok"]
        result["captures"].append(row)
        write_json("detections.json", result)
        print(f'{i+1}/{len(manifest["captures"])} {row["id"]}: L={row["left"]["ok"]} R={row["right"]["ok"]}', flush=True)


def image_points(row, eye):
    return np.array(row[eye]["points"], dtype=np.float32).reshape(-1, 1, 2)


def solve(rows, obj):
    objects = [obj.copy() for _ in rows]
    eyes = [[image_points(row, eye) for row in rows] for eye in ("left", "right")]
    mono = [cv2.calibrateCamera(objects, pts, SIZE, None, None, criteria=CRITERIA) for pts in eyes]
    stereo = cv2.stereoCalibrate(objects, *eyes, mono[0][1], mono[0][2],
                                mono[1][1], mono[1][2], SIZE,
                                criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)
    rms, k1, d1, k2, d2, rotation, translation, essential, fundamental = stereo
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k1, d1, k2, d2, SIZE, rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return dict(K1=k1, D1=d1, K2=k2, D2=d2, R=rotation, T=translation, E=essential, F=fundamental,
                R1=r1, R2=r2, P1=p1, P2=p2, Q=q, roi1=roi1, roi2=roi2,
                mono_rms=[float(mono[0][0]), float(mono[1][0])], stereo_rms=float(rms))


def rms(points):
    return float(np.sqrt(np.mean(np.sum(np.square(points), axis=-1))))


def evaluate(rows, obj, model):
    result = []
    vertical = abs(model["P2"][1, 3]) > abs(model["P2"][0, 3])
    for row in rows:
        mono_errors, poses, rectified = [], [], []
        for eye, n in (("left", 1), ("right", 2)):
            pts = image_points(row, eye)
            k, d = model[f"K{n}"], model[f"D{n}"]
            ok, rv, tv = cv2.solvePnP(obj, pts, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise RuntimeError("PnP failed: " + row["id"])
            reproj = cv2.projectPoints(obj, rv, tv, k, d)[0]
            mono_errors.append(rms((reproj - pts).reshape(-1, 2)))
            poses.append((cv2.Rodrigues(rv)[0], tv))
            rectified.append(cv2.undistortPoints(pts, k, d, R=model[f"R{n}"], P=model[f"P{n}"]).reshape(-1, 2))
        r_left, t_left = poses[0]
        r_right = model["R"] @ r_left
        t_right = model["R"] @ t_left + model["T"]
        predicted = cv2.projectPoints(obj, cv2.Rodrigues(r_right)[0], t_right, model["K2"], model["D2"])[0]
        cross_error = rms((predicted - image_points(row, "right")).reshape(-1, 2))
        residuals = abs(rectified[0][:, 0 if vertical else 1] - rectified[1][:, 0 if vertical else 1])
        # Independent per-camera board poses also expose a moving stereo mount.
        pair_r = poses[1][0] @ r_left.T
        pair_t = poses[1][1] - pair_r @ t_left
        result.append(dict(id=row["id"], mono_rms_px=mono_errors, crossview_rms_px=cross_error,
                           epipolar_px=residuals.tolist(), implied_baseline_mm=float(np.linalg.norm(pair_t)),
                           relative_rotation_vector_deg=(cv2.Rodrigues(pair_r)[0].ravel()*180/np.pi).tolist(),
                           relative_translation_mm=pair_t.ravel().tolist(),
                           board_distance_mm=float(t_left[2, 0]),
                           board_normal=r_left[:, 2].tolist()))
    return result


def summarize(rows):
    residuals = np.concatenate([row["epipolar_px"] for row in rows])
    return dict(pairs=len(rows), mono_rms_px=np.sqrt(np.mean(np.square([r["mono_rms_px"] for r in rows]), axis=0)).tolist(),
                crossview_rms_px=float(np.sqrt(np.mean(np.square([r["crossview_rms_px"] for r in rows])))),
                epipolar_median_px=float(np.median(residuals)), epipolar_p95_px=float(np.percentile(residuals, 95)),
                epipolar_max_px=float(np.max(residuals)))


def camera_sanity(k, d):
    corners = np.float64([[0, 0], [SIZE[0]-1, 0], [0, SIZE[1]-1], [SIZE[0]-1, SIZE[1]-1]])
    radius = np.max(np.linalg.norm((corners-k[:2, 2]) / np.diag(k)[:2], axis=1))
    r = np.linspace(0, radius, 200)
    k1, k2, _, _, k3 = d.reshape(-1)[:5]
    derivative = 1 + 3*k1*r*r + 5*k2*r**4 + 7*k3*r**6
    return dict(positive_focal=bool(k[0, 0] > 0 and k[1, 1] > 0),
                principal_point_in_sensor=bool(0 < k[0, 2] < SIZE[0] and 0 < k[1, 2] < SIZE[1]),
                radial_derivative_min=float(derivative.min()),
                radial_monotonic_over_sensor_proxy=bool(np.all(derivative > 0)))


def make_review(rows, model, prefix=""):
    # Raw corner overlays prove indexing; rectified samples show epipolar alignment.
    cards, rect_cards = [], []
    chosen = [rows[0], rows[len(rows)//2], rows[-1]]
    for row in chosen:
        views, rect_views = [], []
        for eye, n in (("left", 1), ("right", 2)):
            gray = cv2.imread(str(ROOT / "captures" / row["id"] / (eye + ".png")), 0)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            pts = image_points(row, eye)
            cv2.drawChessboardCorners(rgb, PATTERN, pts, True)
            x, y = pts[0, 0].round().astype(int)
            cv2.circle(rgb, (x, y), 16, (0, 0, 255), 4)
            upright = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            upright = cv2.resize(upright, (324, 576), interpolation=cv2.INTER_AREA)
            origin = (np.array([y, SIZE[0]-1-x]) * .15).round().astype(int)
            cv2.putText(upright, "0", tuple(origin + (5, -7)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            views.append(upright)
            mx, my = cv2.initUndistortRectifyMap(model[f"K{n}"], model[f"D{n}"], model[f"R{n}"], model[f"P{n}"], SIZE, cv2.CV_32FC1)
            rect = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
            rect = cv2.rotate(rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rect_views.append(cv2.cvtColor(cv2.resize(rect, (324, 576)), cv2.COLOR_GRAY2BGR))
        card = np.hstack(views)
        cv2.putText(card, row["id"][9:22], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cards.append(card)
        rect = np.hstack(rect_views)
        if abs(model["P2"][1, 3]) > abs(model["P2"][0, 3]):
            for y in range(30, 576, 36):
                cv2.line(rect, (0, y), (647, y), (0, 220, 0), 1)
        else:
            for x in range(20, 324, 36):
                for offset in (0, 324):
                    cv2.line(rect, (x+offset, 0), (x+offset, 575), (0, 220, 0), 1)
        rect_cards.append(rect)
    cv2.imwrite(str(OUT / (prefix + "corner-review.jpg")), np.vstack(cards))
    cv2.imwrite(str(OUT / (prefix + "rectified-validation.jpg")), np.vstack(rect_cards))


def validate():
    """Validate new captures against an unchanged parameter file, without fitting."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    frozen_path = OUT / "parameters.npz"
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    model = dict(np.load(frozen_path, allow_pickle=False))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8-sig"))
    old_ids = {row["id"] for row in manifest["captures"]}
    with urllib.request.urlopen("http://127.0.0.1:8877/captures", timeout=15) as stream:
        available = json.load(stream)["captures"]
    new = [row for row in available if row["id"] not in old_ids
           and row["id"] > max(old_ids)
           and row.get("calibration_target", {}).get("id") == "screen-checkerboard-10x7"]
    if not new:
        raise RuntimeError("No new screen-checkerboard captures")

    def fetch(meta):
        assert (meta["width"], meta["height"]) == SIZE
        assert meta["calibration_target"]["square_size_mm"] == float(model["square_mm"])
        assert meta["calibration_target"]["inner_corners"] == list(PATTERN)
        assert (meta["left_device"], meta["right_device"]) == ("/dev/video33", "/dev/video42")
        dest = ROOT / "captures" / meta["id"]
        dest.mkdir(exist_ok=True)
        for eye in ("left", "right"):
            path = dest / (eye + ".png")
            if not path.exists():
                with urllib.request.urlopen(f'http://127.0.0.1:8877/captures/{meta["id"]}/{eye}.png', timeout=45) as stream:
                    data = stream.read()
                gray = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
                assert gray is not None and gray.shape == SIZE[::-1], "Invalid capture download"
                path.write_bytes(data)
        (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, new))
    detections = []
    for meta in new:
        row = dict(id=meta["id"], skew_ms=meta["skew_ms"])
        for eye in ("left", "right"):
            raw = (ROOT / "captures" / meta["id"] / (eye + ".png")).read_bytes()
            row[eye] = ordered_corners(cv2.imdecode(np.frombuffer(raw, np.uint8), 0))
            row[eye]["sha256"] = hashlib.sha256(raw).hexdigest()
        row["ok"] = row["left"]["ok"] and row["right"]["ok"]
        detections.append(row)
        print(f'New capture {len(detections)}/{len(new)}: complete={row["ok"]}', flush=True)
    good = [row for row in detections if row["ok"]]
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * float(model["square_mm"])
    metrics = evaluate(good, obj, model) if good else []
    summary = summarize(metrics) if metrics else None
    checks = dict(at_least_five_new_pairs=len(new) >= 5, all_pairs_detected=len(good) == len(new),
                  new_epipolar_p95_under_2px=bool(summary and summary["epipolar_p95_px"] < 2),
                  new_crossview_rms_under_3px=bool(summary and summary["crossview_rms_px"] < 3))
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    result = dict(status="new_capture_geometry_pass" if all(checks.values()) else "new_capture_geometry_fail",
                  parameters_sha256=frozen_hash, parameters_modified=False, refitted=False, deployed=False,
                  image_size=SIZE, checks=checks, summary=summary, metrics=metrics, captures=detections,
                  note="New captures used only for validation. Scale accuracy and mapping accuracy remain unverified.")
    # After recording the frozen-model failure, diagnose only the relative pose.
    # This refit is explicitly separate from the independent acceptance result.
    recovery = None
    if len(good) >= 5 and result["status"] == "new_capture_geometry_fail":
        def relative_model(rows):
            stereo = cv2.stereoCalibrate([obj] * len(rows),
                [image_points(row, "left") for row in rows],
                [image_points(row, "right") for row in rows],
                model["K1"].copy(), model["D1"].copy(), model["K2"].copy(), model["D2"].copy(),
                SIZE, criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)
            candidate = model.copy()
            candidate.update(dict(zip(("K1", "D1", "K2", "D2", "R", "T", "E", "F"), stereo[1:])))
            rect = cv2.stereoRectify(candidate["K1"], candidate["D1"], candidate["K2"], candidate["D2"],
                SIZE, candidate["R"], candidate["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
            candidate.update(dict(zip(("R1", "R2", "P1", "P2", "Q"), rect[:5])))
            return candidate, float(stereo[0])

        candidate, fit_rms = relative_model(good)
        folds = []
        for i, row in enumerate(good):
            folded, _ = relative_model(good[:i] + good[i+1:])
            folds.extend(evaluate([row], obj, folded))
        cv_summary = summarize(folds)
        delta_deg = float(np.linalg.norm(cv2.Rodrigues(candidate["R"] @ model["R"].T)[0]) * 180 / np.pi)
        recovery = dict(status="diagnostic_refit_requires_new_acceptance", independent_acceptance=False,
                        intrinsics_changed=False, stereo_fit_rms_px=fit_rms,
                        baseline_mm=float(np.linalg.norm(candidate["T"])), relative_rotation_change_deg=delta_deg,
                        leave_one_pose_out=cv_summary, fold_metrics=folds,
                        training=summarize(evaluate(good, obj, candidate)),
                        parameters={key: candidate[key].tolist() for key in ("K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2", "Q")})
        candidate["status"] = np.array(recovery["status"])
        candidate["source_parameters_sha256"] = np.array(frozen_hash)
        candidate["fit_capture_ids"] = np.array([row["id"] for row in good])
        np.savez_compressed(OUT / "parameters-current-candidate.npz", **candidate)
        result["relative_pose_diagnostic"] = recovery
    write_json("acceptance.json", result)
    if good:
        make_review(good, model, prefix="acceptance-")
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash
    summary_text = "未获得完整角点，无法计算误差。" if summary is None else (
        f"旧参数在新照片上的左右对齐误差中位数为{summary['epipolar_median_px']:.2f} px，"
        f"95分位为{summary['epipolar_p95_px']:.2f} px；左图预测右图RMS为{summary['crossview_rms_px']:.2f} px。"
        f"左右单目RMS为{summary['mono_rms_px'][0]:.3f} / {summary['mono_rms_px'][1]:.3f} px。")
    recovery_text = ""
    if recovery:
        recovery_text = f'''
### 双目相对姿态诊断（不是独立验收通过）

保持旧内参和畸变不变，仅使用新照片重新估计两镜头的相对旋转和平移：

- 相对旧参数的旋转变化量：{recovery['relative_rotation_change_deg']:.2f}度。
- 新估计光心间距：{recovery['baseline_mm']:.2f} mm。
- 新照片拟合双目RMS：{recovery['stereo_fit_rms_px']:.3f} px。
- 每次留出一整组、用其余组估计的交叉验证：对齐95分位{cv_summary['epipolar_p95_px']:.3f} px，跨相机预测RMS {cv_summary['crossview_rms_px']:.3f} px。

新候选参数存于 `parameters-current-candidate.npz`，仅为诊断草案，没有覆盖原 `parameters.npz`，没有部署。新照片已用于这次重估，不能再称为新候选参数的独立验收集。交叉验证尚不能满足原先的2 px对齐门槛，不能只凭较低的拟合误差宣称通过。

证据支持当前双目相对几何不同于前一批；具体是重新调整、支架形变还是采集运动，尚未由照片独立确认。先确认固定方式及是否有过调整，再决定补采，避免混合不同固定状态反复标定。
'''
    section = f'''
## 新拍照片独立验收

读取{len(new)}组此前未参与拟合的屏幕棋盘照片，其中{len(good)}组完整检出。验收使用固定原参数，SHA256：`{frozen_hash}`，未修改或重新拟合原参数。

验收状态：**{result['status']}**。{summary_text}以上均为3840×2160原图像素。

详细逐组角点、原图哈希、误差和检查结果见 `acceptance.json`；原参数校正效果见 `acceptance-rectified-validation.jpg`。实物距离和地图精度仍未验收。
{recovery_text}
复现本项验收：`python -B stereo/calibrate.py validate`。原始照片全部保留。
'''
    report_path = OUT / "report.md"
    previous = report_path.read_text(encoding="utf-8").split("\n## 新拍照片独立验收")[0]
    report_path.write_text(previous.rstrip() + "\n" + section, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "checks", "summary")}, indent=2), flush=True)
    if recovery:
        print(json.dumps({k: recovery[k] for k in ("status", "stereo_fit_rms_px", "baseline_mm", "relative_rotation_change_deg", "leave_one_pose_out")}, indent=2), flush=True)


def fit():
    detection = json.loads((OUT / "detections.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8-sig"))
    assert [r["id"] for r in detection["captures"]] == [r["id"] for r in manifest["captures"]], "Detection incomplete"
    assert detection["square_mm"] == manifest["square_mm"] == 24.2
    selected, excluded = [], []
    for row in detection["captures"]:
        for eye in ("left", "right"):
            source = ROOT / "captures" / row["id"] / (eye + ".png")
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row[eye]["sha256"], "Source image changed"
        if not row["ok"]:
            excluded.append(dict(id=row["id"], reason="incomplete_detection"))
            continue
        distances = [max(float(np.median(np.linalg.norm(image_points(row, eye)-image_points(other, eye), axis=2)))
                         for eye in ("left", "right")) for other in selected]
        if distances and min(distances) < 12:
            excluded.append(dict(id=row["id"], reason="near_duplicate_under_12px", reference=selected[int(np.argmin(distances))]["id"]))
        else:
            selected.append(row)
    if len(selected) < 15:
        raise RuntimeError("Need at least 15 complete, distinct pairs")
    # Pose-level holdout is frozen before any model fitting or residual inspection.
    heldout = selected[4::5]
    heldout_ids = {row["id"] for row in heldout}
    train = [row for row in selected if row["id"] not in heldout_ids]
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * detection["square_mm"]
    print(f"Whole-batch diagnostic: {len(train)} train / {len(heldout)} held out", flush=True)
    model = solve(train, obj)
    batch_metrics = evaluate(train, obj, model)
    whole_batch = dict(stereo_rms_px=model["stereo_rms"], mono_rms_px=model["mono_rms"],
                       baseline_mm=float(np.linalg.norm(model["T"])), training_metrics=batch_metrics,
                       used_for_parameter_export=False)
    start_id = manifest.get("fit_start_capture")
    if start_id:
        assert start_id in {r["id"] for r in selected}, "Unknown segment start"
        excluded.extend(dict(id=row["id"], reason="earlier_mount_state_or_transition") for row in selected if row["id"] < start_id)
        train = [row for row in train if row["id"] >= start_id]
        heldout = [row for row in heldout if row["id"] >= start_id]
        assert len(train) >= 15 and len(heldout) >= 4, "Insufficient segment support"
        model = solve(train, obj)
    print(f"Selected segment: {len(train)} train / {len(heldout)} held out", flush=True)
    initial = evaluate(train, obj, model)
    scores = np.array([r["crossview_rms_px"] for r in initial])
    threshold = max(2.5, float(np.median(scores) + 3 * 1.4826 * np.median(abs(scores - np.median(scores)))))
    rejected_ids = {r["id"] for r in initial if r["crossview_rms_px"] > threshold}
    # One training-only robust pass; never remove validation poses using residuals.
    if rejected_ids and len(rejected_ids) <= len(train)//4:
        excluded.extend(dict(id=ident, reason="training_crossview_outlier", threshold_px=threshold) for ident in sorted(rejected_ids))
        train = [r for r in train if r["id"] not in rejected_ids]
        print(f"Refit after {len(rejected_ids)} training outliers (threshold {threshold:.3f}px)", flush=True)
        model = solve(train, obj)
    train_metrics = evaluate(train, obj, model)
    held_metrics = evaluate(heldout, obj, model)
    train_summary, held_summary = summarize(train_metrics), summarize(held_metrics)
    sanity = [camera_sanity(model[f"K{n}"], model[f"D{n}"]) for n in (1, 2)]
    baseline = float(np.linalg.norm(model["T"]))
    # Explicit experiment gates, not a universal calibration accuracy guarantee.
    checks = dict(mono_training_under_1px=max(model["mono_rms"]) < 1,
                  stereo_training_under_1px=model["stereo_rms"] < 1,
                  heldout_epipolar_p95_under_2px=held_summary["epipolar_p95_px"] < 2,
                  heldout_crossview_rms_under_3px=held_summary["crossview_rms_px"] < 3,
                  camera_model_sanity=all(s["positive_focal"] and s["principal_point_in_sensor"] and s["radial_monotonic_over_sensor_proxy"] for s in sanity))
    report = dict(status="geometry_pass_metric_accuracy_unverified" if all(checks.values()) else "needs_review_not_for_mapping",
                  deployed=False, image_size=SIZE, square_mm=detection["square_mm"],
                  fit_start_capture=start_id, selection_reason=manifest.get("selection_reason"),
                  whole_batch_diagnostic=whole_batch,
                  coordinate_convention="raw sensor x-right y-down; X_right=R*X_left+T; T and Q depths in mm",
                  distortion_model="OpenCV Brown-Conrady k1,k2,p1,p2,k3",
                  raw_epipolar_axis="vertical" if abs(model["P2"][1, 3]) > abs(model["P2"][0, 3]) else "horizontal",
                  baseline_mm=baseline, baseline_reference_mm=95, baseline_reference_used_in_fit=False,
                  capture_count=len(detection["captures"]), detected_pairs=sum(r["ok"] for r in detection["captures"]),
                  checks=checks, camera_sanity=sanity, fit_mono_rms_px=model["mono_rms"], fit_stereo_rms_px=model["stereo_rms"],
                  training=train_summary, heldout=held_summary,
                  train_ids=[r["id"] for r in train], heldout_ids=[r["id"] for r in heldout],
                  exclusions=excluded, initial_training_metrics=initial,
                  training_metrics=train_metrics, heldout_metrics=held_metrics,
                  limitations=["Screen scale is user measured, not independently verified", "Software timestamp pairing; hardware exposure synchronization unverified", "Holdout images were not fitted, but were inspected during diagnosis; new untouched acceptance captures are required", "No independent known-distance or mapping validation", "Parameters are valid only for unchanged lenses, focus, mounting and sensor image geometry"],
                  parameters={key: value.tolist() for key, value in model.items() if isinstance(value, np.ndarray)})
    write_json("report.json", report)
    arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(OUT / "parameters.npz", **arrays, image_size=SIZE, square_mm=detection["square_mm"],
                        status=report["status"], coordinate_convention=report["coordinate_convention"])
    make_review(heldout, model)
    md = f'''# 双目标定结果（待验收草案）

状态：{report['status']}。参数已保存，未部署到板端，尚不能声明准确建图。

## 本次数据与方法

- 46组屏幕棋盘照片均检出左右各54个角点；旧拓竹板照片未参与。
- 棋盘为10×7格、9×6内角点，格距24.2 mm（用户横纵各5格实测121 mm）。
- 按原始序列每第5组预留验证；验证图不参加参数拟合和训练异常筛选。
- 全批双目RMS为{whole_batch['stereo_rms_px']:.3f}像素，数据不能直接混合。{manifest.get('selection_reason', '')}
- 当前参数只拟合最后一段：{len(train)}组拟合、{len(heldout)}组保留验证。全部原图保留。
- 参数采用五项畸变模型，先单目标定，再固定内参估计双目关系；没有强制输入手测基线。

## 结果（全部误差为3840×2160原始图像像素）

| 指标 | 结果 |
|---|---:|
| 左 / 右单目拟合RMS | {model['mono_rms'][0]:.3f} / {model['mono_rms'][1]:.3f} px |
| 双目拟合RMS | {model['stereo_rms']:.3f} px |
| 保留验证左右对齐误差中位数 | {held_summary['epipolar_median_px']:.3f} px |
| 保留验证左右对齐误差95分位 | {held_summary['epipolar_p95_px']:.3f} px |
| 保留验证左右对齐误差最大值 | {held_summary['epipolar_max_px']:.3f} px |
| 左图推算右图的验证RMS | {held_summary['crossview_rms_px']:.3f} px |
| 算法估计两光心间距 | {baseline:.2f} mm |

预设实验门槛：单目和双目拟合RMS均小于1 px，验证对齐95分位小于2 px、跨相机预测RMS小于3 px，并通过内参和畸变合理性检查。
通过情况：{json.dumps(checks, ensure_ascii=False)}。
这些门槛只用于本次筛选，不等于测距精度承诺。验证图已在诊断中查看，下一步验收应使用全新照片。

## 使用限制与下一步

1. 先检查两镜头的刚性连接和锁紧状态。分段变化支持相对姿态变化这一判断，但照片不能单独确定是支架、碰动还是其他采集因素。
2. 保持当前镜头、焦距、屏幕缩放不变；停稳2秒，再补拍5组全新验证照片，覆盖近远及不同倾角。
3. 对已知距离的物体做独立测距验证后，才能评估实际尺度准确度。本批没有做建图或实物距离验收。
4. 参数坐标为未旋转的3840×2160原图，X_right=R*X_left+T，T单位mm。原图极线方向为{report['raw_epipolar_axis']}；不能直接交给只支持水平视差的SGBM。缩放或旋转输入时，必须同步转换内参与校正坐标。

## 文件

- parameters.npz：K1、D1、K2、D2、R、T、E、F、R1、R2、P1、P2、Q，以及尺寸、单位说明和草案状态。
- report.json：完整参数、逐组误差、选择理由、全批诊断及保留验证结果。
- detections.json：原图角点、左右图SHA256校验值、检测结果。
- corner-review.jpg：验证图同名角点核对，红色0为同一物理起点。
- rectified-validation.jpg：同批验证图校正后的极线检查，转正显示，绿线用于目视比较。
- manifest.json：本次固定照片清单与取段配置。

复现：在项目的 算法/视觉算法/软件 目录运行 `python -B stereo/calibrate.py detect`，然后 `python -B stereo/calibrate.py fit`。依赖NumPy与OpenCV；程序不改原图，不启动或重启板端服务。
'''
    (OUT / "report.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "baseline_mm", "raw_epipolar_axis", "checks", "training", "heldout", "camera_sanity")}, indent=2), flush=True)


def batch():
    """Fit one new fixed-mount session with a prespecified untouched test set."""
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8-sig"))
    train_ids, test_ids = set(manifest["train_ids"]), set(manifest["heldout_ids"])
    assert not train_ids & test_ids and len(train_ids) >= 10 and len(test_ids) >= 5
    assert train_ids | test_ids == {r["id"] for r in manifest["captures"]}
    for meta in manifest["captures"]:
        assert (meta["width"], meta["height"]) == SIZE and not meta["rectified"]
        assert meta["calibration_target"]["inner_corners"] == list(PATTERN)
        assert meta["calibration_target"]["square_size_mm"] == manifest["square_mm"] == 24.2
        assert (meta["left_device"], meta["right_device"]) == ("/dev/video33", "/dev/video42")
    detect()
    rows = json.loads((OUT / "detections.json").read_text(encoding="utf-8"))["captures"]
    if not all(r["ok"] for r in rows):
        raise RuntimeError("Incomplete detection; preserve frozen split and inspect failed captures")
    train = [r for r in rows if r["id"] in train_ids]
    test = [r for r in rows if r["id"] in test_ids]
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * manifest["square_mm"]
    model = solve(train, obj)
    training, heldout = evaluate(train, obj, model), evaluate(test, obj, model)
    train_summary, test_summary = summarize(training), summarize(heldout)
    sanity = [camera_sanity(model[f"K{n}"], model[f"D{n}"]) for n in (1, 2)]
    checks = dict(mono_training_under_1px=max(model["mono_rms"]) < 1,
                  stereo_training_under_1px=model["stereo_rms"] < 1,
                  heldout_epipolar_p95_under_2px=test_summary["epipolar_p95_px"] < 2,
                  heldout_crossview_rms_under_3px=test_summary["crossview_rms_px"] < 3,
                  camera_model_sanity=all(s["positive_focal"] and s["principal_point_in_sensor"] and s["radial_monotonic_over_sensor_proxy"] for s in sanity))
    report = dict(status="geometry_pass_metric_accuracy_unverified" if all(checks.values()) else "needs_review_not_for_mapping",
                  deployed=False, image_size=SIZE, square_mm=manifest["square_mm"],
                  baseline_mm=float(np.linalg.norm(model["T"])), baseline_reference_used_in_fit=False,
                  split_policy=manifest["split_policy"], train_ids=manifest["train_ids"], heldout_ids=manifest["heldout_ids"],
                  camera_sanity=sanity, checks=checks, fit_mono_rms_px=model["mono_rms"], fit_stereo_rms_px=model["stereo_rms"],
                  training=train_summary, heldout=test_summary, training_metrics=training, heldout_metrics=heldout,
                  raw_epipolar_axis="vertical" if abs(model["P2"][1, 3]) > abs(model["P2"][0, 3]) else "horizontal",
                  coordinate_convention="raw sensor x-right y-down; X_right=R*X_left+T; T and Q depths in mm",
                  parameters={k: v.tolist() for k, v in model.items() if isinstance(v, np.ndarray)},
                  limitations=["No independent physical distance verification", "Hardware exposure synchronization unverified", "No mapping accuracy test", "Valid only while mounting, focus and image geometry remain unchanged"])
    write_json("report.json", report)
    arrays = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(OUT / "parameters.npz", **arrays, image_size=SIZE, square_mm=manifest["square_mm"],
                        status=report["status"], coordinate_convention=report["coordinate_convention"])
    make_review(test, model)
    md = f'''# 新固定状态标定与独立验收

状态：**{report['status']}**。参数未部署，实物距离和建图精度尚未验证。

本次实际收到{len(rows)}组新屏幕棋盘照片，左右各54个角点全部检出。检测和拟合前已固定分组：前{len(train)}组计算，后{len(test)}组独立验收。没有混入历史照片、按验收误差删图或重新拟合验收图。格距24.2 mm。

| 指标 | 结果 |
|---|---:|
| 左 / 右单目拟合RMS | {model['mono_rms'][0]:.3f} / {model['mono_rms'][1]:.3f} px |
| 双目拟合RMS | {model['stereo_rms']:.3f} px |
| 独立验收左右对齐误差中位数 | {test_summary['epipolar_median_px']:.3f} px |
| 独立验收左右对齐误差95分位 | {test_summary['epipolar_p95_px']:.3f} px |
| 独立验收左右对齐误差最大值 | {test_summary['epipolar_max_px']:.3f} px |
| 独立验收跨相机预测RMS | {test_summary['crossview_rms_px']:.3f} px |
| 估计光心间距 | {report['baseline_mm']:.2f} mm |

误差均为3840×2160原图像素。预设门槛沿用上一批：单目和双目RMS小于1 px、独立验收对齐95分位小于2 px、跨相机预测RMS小于3 px，且内参/畸变合理。结果：`{json.dumps(checks, ensure_ascii=False)}`。

采用五项畸变模型，先以本批训练图计算内参，再固定内参估计双目关系。手工测量的基线未参与约束。图片SHA256、角点、逐组误差和全部参数分别保存于detections.json与report.json；parameters.npz可由程序读取。

原图极线方向为{report['raw_epipolar_axis']}，参数必须与原始图像坐标一起使用；输入旋转或缩放时需同步转换，不可直接套到旧SGBM输入。角点及校正预览分别为corner-review.jpg、rectified-validation.jpg。

复现：`python -B stereo/calibrate.py batch --output stereo/calibration/current`。原始照片全部保留，旧参数与服务未改变。
'''
    (OUT / "report.md").write_text(md, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "baseline_mm", "checks", "training", "heldout", "camera_sanity")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["detect", "fit", "validate", "batch"])
    parser.add_argument("--output", type=Path, help="Directory containing the frozen manifest and results")
    args = parser.parse_args()
    if args.output:
        OUT = args.output.resolve()
    if args.action == "detect":
        detect()
    elif args.action == "fit":
        fit()
    elif args.action == "validate":
        validate()
    else:
        batch()
