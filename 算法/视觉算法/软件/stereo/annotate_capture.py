"""Review-only annotations for the second capture; never writes calibration.

The visible top/left board edges establish the physical grid indices below.
These anchors are specific to this capture, not reusable for a moving board.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CAPTURE = "20240825T204140845288Z-1964342b"
FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_LARGER
CONFIG = {"left": (1120, 2070, (98.6, 134.3)), "right": (160, 2150, (83, 95))}


def detect(image, sigma):
    processed = cv2.GaussianBlur(image, (0, 0), sigma)
    ok, points, meta = cv2.findChessboardCornersSBWithMeta(processed, (7, 5), FLAGS)
    if not ok:
        raise RuntimeError("Chessboard detection failed; no annotations exported")
    return points.reshape(*meta.shape, 2)


def main():
    report = dict(schema=2, capture_id=CAPTURE, status="draft_boundary_anchored_review",
                  calibration_valid=False, square_mm=2.5,
                  origin_definition="Visible top-left of board in CCW upright view. First internal intersection is (col=1,row=1); labels are (col,row).",
                  method="CLAHE + Gaussian smoothing + OpenCV SB; physical indexing anchored manually to visible board edges, not detector order.",
                  limitations=["One pose only; not full calibration", "No code decoding", "Physical indexing requires visual review", "Smoothing repeatability is not absolute corner accuracy"],
                  views={}, correspondences=[])
    panels = []
    all_points = {}
    for side, (x, y, seed) in CONFIG.items():
        path = ROOT / "captures" / CAPTURE / (side + ".png")
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != (2160, 3840):
            raise RuntimeError("Missing or unexpected source image")
        upright = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        roi = upright[y:y + 600, x:x + 600]
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
        grid = detect(enhanced, 2.0)
        if side == "left":
            # SB enumerates this view along the opposite axis; normalize it.
            grid = grid[:, ::-1].transpose(1, 0, 2)
            first_col, first_row = 1, 1
        else:
            first_col, first_row = 2, 2
        if not (grid[0, 1, 0] > grid[0, 0, 0] and grid[1, 0, 1] > grid[0, 0, 1]):
            raise RuntimeError("Unexpected detector orientation; review anchors")
        # An independent smoothing setting checks whether localization is stable.
        alternate = detect(enhanced, 1.5).reshape(-1, 2)
        points = {}
        shifts = []
        for row in range(2, 9):
            for col in range(2, 9):
                p = grid[row - first_row, col - first_col].astype(float)
                shift = float(np.linalg.norm(alternate - p, axis=1).min())
                if shift > 1.5:
                    raise RuntimeError(f"Unstable {side} point ({col},{row}): {shift:.2f}px")
                points[col, row] = p
                shifts.append(shift)
        anchor = np.array(seed, np.float32).reshape(1, 1, 2)
        cv2.cornerSubPix(cv2.GaussianBlur(enhanced, (0, 0), 2), anchor,
                         (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
        anchor = anchor.reshape(2)
        # Verify that detected start and board-edge anchor differ by whole cells.
        gx = np.median(grid[:, 1:] - grid[:, :-1], axis=(0, 1))
        gy = np.median(grid[1:] - grid[:-1], axis=(0, 1))
        anchor_expected = grid[0, 0] - gx * (first_col - 1) - gy * (first_row - 1)
        anchor_error = float(np.linalg.norm(anchor - anchor_expected))
        if anchor_error > 3:
            raise RuntimeError(f"Board boundary anchor inconsistent: {side} {anchor_error}")
        report["views"][side] = dict(image=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=[3840, 2160],
            roi_upright=[x, y, 600, 600], detected_start_col_row=[first_col, first_row],
            anchor_1_1_upright=(anchor + [x, y]).tolist(), anchor_error_px=anchor_error,
            smoothing_repeatability_max_px=max(shifts))
        all_points[side] = {key: dict(upright=(p + [x, y]).tolist(),
            sensor=[float(3839 - (p[1] + y)), float(p[0] + x)]) for key, p in points.items()}
        # Show enough surrounding board to verify the physical origin.
        panel = cv2.cvtColor(cv2.resize(enhanced[:360, :360], (900, 900)), cv2.COLOR_GRAY2BGR)
        scale = 2.5
        square = np.array([anchor - gx - gy, anchor - gy, anchor, anchor - gx])
        cv2.polylines(panel, [np.round(square * scale).astype(np.int32)], True, (255, 180, 0), 2)
        a = tuple(np.round(anchor * scale).astype(int))
        cv2.drawMarker(panel, a, (60, 60, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(panel, "A (1,1)", (a[0] - 20, a[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (60, 60, 255), 2)
        for (col, row), p in points.items():
            q = tuple(np.round(p * scale).astype(int))
            cv2.circle(panel, q, 4, (0, 255, 255), -1)
            cv2.putText(panel, f"{col},{row}", (q[0] + 5, q[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 255, 255), 1, cv2.LINE_AA)
        panel = cv2.copyMakeBorder(panel, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(panel, f"{side.upper()} | (col,row) | DRAFT", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, .85, (240, 240, 240), 2)
        panels.append(panel)
    for col, row in all_points["left"]:
        report["correspondences"].append(dict(col=col, row=row,
            object_mm=[col * 2.5, row * 2.5, 0], left=all_points["left"][col, row],
            right=all_points["right"][col, row]))
    report["paired_candidates"] = len(report["correspondences"])
    (ROOT / "diagnostic" / "corner_sample.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cv2.imwrite(str(ROOT / "diagnostic" / "checkerboard.png"), np.hstack(panels))
    print(json.dumps({"paired_candidates": report["paired_candidates"], "views": report["views"]}, indent=2))


if __name__ == "__main__":
    main()
