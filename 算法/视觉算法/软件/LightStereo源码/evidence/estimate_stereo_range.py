#!/usr/bin/env python3
"""Reproduce a first-order stereo range geometry estimate from the saved calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOFTWARE_DIR = HERE.parent.parent
CALIBRATION_PATH = SOFTWARE_DIR / "数值深度接口候选" / "app" / "calibration.npz"
HARDWARE_EVIDENCE_PATH = HERE / "rk3588-hardware-live.json"
OUTPUT_PATH = HERE / "stereo-range-geometry.json"

FAST_SCALE = 0.0625
RANGE_M = (10.0, 20.0, 30.0, 50.0)
OBJECT_WIDTH_MM = 24.0
ASSUMED_DISPARITY_SIGMA_PX = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_matrix_match(name: str, actual: np.ndarray, expected: object) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected_array.shape or not np.allclose(
        actual, expected_array, rtol=0.0, atol=1e-9
    ):
        raise SystemExit(f"{name} in calibration.npz does not match live evidence")


def main() -> None:
    calibration_sha256 = sha256_file(CALIBRATION_PATH)
    hardware_sha256 = sha256_file(HARDWARE_EVIDENCE_PATH)
    hardware = json.loads(HARDWARE_EVIDENCE_PATH.read_text(encoding="utf-8"))
    snapshot = hardware["snapshot"]
    live_calibration = snapshot["calibration"]
    expected_calibration_sha256 = live_calibration["sha256"].lower()
    if calibration_sha256 != expected_calibration_sha256:
        raise SystemExit(
            "calibration.npz SHA-256 does not match snapshot.calibration.sha256: "
            f"{calibration_sha256} != {expected_calibration_sha256}"
        )

    with np.load(CALIBRATION_PATH, allow_pickle=False) as calibration:
        p1 = np.asarray(calibration["P1"], dtype=np.float64)
        p2 = np.asarray(calibration["P2"], dtype=np.float64)
        image_size = [int(value) for value in calibration["image_size"].tolist()]

    require_matrix_match("P1", p1, live_calibration["P1"])
    require_matrix_match("P2", p2, live_calibration["P2"])
    if image_size != [int(value) for value in live_calibration["image_size"]]:
        raise SystemExit("image_size in calibration.npz does not match live evidence")

    # This calibration's P2[1,3] is the vertical baseline term, in pixel*mm.
    # Convert mm to m, then use its magnitude for positive disparity/depth values.
    p2_1_3_signed_px_m = float(p2[1, 3]) / 1000.0
    native_fb_px_m = abs(p2_1_3_signed_px_m)
    fast_fb_px_m = FAST_SCALE * native_fb_px_m
    fx_px = float(p1[0, 0])
    object_width_m = OBJECT_WIDTH_MM / 1000.0

    service_status = snapshot["status"]
    depth_range_m = [float(value) for value in service_status["depth_range_m"]]
    npu_mode = service_status.get("npu_preview", {}).get("mode")

    estimates = []
    for distance_m in RANGE_M:
        native_disparity_px = native_fb_px_m / distance_m
        fast_disparity_px = fast_fb_px_m / distance_m
        object_width_native_px = fx_px * object_width_m / distance_m
        object_width_fast_px = FAST_SCALE * object_width_native_px

        # Z = fB / d, so |dZ/dd| * sigma_d = Z^2 * sigma_d / fB.
        sigma_z_native_m = (
            distance_m**2 * ASSUMED_DISPARITY_SIGMA_PX / native_fb_px_m
        )
        sigma_z_fast_m = (
            distance_m**2 * ASSUMED_DISPARITY_SIGMA_PX / fast_fb_px_m
        )
        estimates.append(
            {
                "distance_m": distance_m,
                "native_rectified_grid": {
                    "grid_scale": 1.0,
                    "effective_fB_px_m": native_fb_px_m,
                    "disparity_px": native_disparity_px,
                    "object_width_24mm_horizontal_px": object_width_native_px,
                    "assumed_disparity_sigma_px": ASSUMED_DISPARITY_SIGMA_PX,
                    "sigma_z_first_order_m": sigma_z_native_m,
                },
                "fast_grid": {
                    "grid_scale": FAST_SCALE,
                    "effective_fB_px_m": fast_fb_px_m,
                    "disparity_px": fast_disparity_px,
                    "object_width_24mm_horizontal_px": object_width_fast_px,
                    "assumed_disparity_sigma_px": ASSUMED_DISPARITY_SIGMA_PX,
                    "sigma_z_first_order_m": sigma_z_fast_m,
                },
            }
        )

    output = {
        "title": "Stereo range geometry estimate (not a range-accuracy test)",
        "inputs": {
            "calibration_npz": {
                "path_from_software_dir": "数值深度接口候选/app/calibration.npz",
                "sha256": calibration_sha256,
                "matches_live_evidence_calibration_sha256": True,
            },
            "hardware_live_evidence": {
                "path": "rk3588-hardware-live.json",
                "sha256": hardware_sha256,
                "recorded_at_asia_shanghai": hardware.get(
                    "recorded_at_asia_shanghai"
                ),
                "calibration_sha256": expected_calibration_sha256,
            },
        },
        "calibration": {
            "image_size_px": image_size,
            "P1": p1.tolist(),
            "P2": p2.tolist(),
            "P2_1_3_signed_px_m": p2_1_3_signed_px_m,
            "native_effective_fB_px_m": native_fb_px_m,
            "fast_scale": FAST_SCALE,
            "fast_effective_fB_px_m": fast_fb_px_m,
            "horizontal_fx_px_from_P1_0_0": fx_px,
            "baseline_mm_reported_in_live_evidence_not_used_to_rebuild_fB": float(
                live_calibration["baseline_mm"]
            ),
        },
        "current_service_snapshot": {
            "algorithm": service_status.get("algorithm"),
            "depth_mode": service_status.get("depth_mode"),
            "depth_range_m": depth_range_m,
            "npu_preview_mode": npu_mode,
            "note": (
                "The live snapshot reports the service depth range as 0.3-5.0 m. "
                "The 10-50 m rows below are outside that reported range."
            ),
        },
        "method": {
            "native_fB": "abs(P2[1,3]) / 1000; use the calibrated P2 term directly, not an fx*T reconstruction",
            "fast_fB": "fast_scale * native_fB",
            "disparity": "d_px = effective_fB_px_m / Z_m",
            "object_projection": "width_px = P1[0,0] * 0.024 m / Z_m; assumed fronto-parallel horizontal 24 mm width",
            "first_order_depth_uncertainty": "sigma_Z_m ~= Z_m^2 * sigma_d_px / effective_fB_px_m",
            "pixel_grid_warning": (
                "sigma_d=0.25 px is applied separately in each grid: native rectified-image px "
                "for native results, and fast-grid px for fast results. A fast-grid 0.25 px "
                "is not a native-image 0.25 px."
            ),
        },
        "assumptions_and_limits": [
            "The fast grid is modeled as isotropic 1/16 scaling (0.0625) of rectified pixel coordinates.",
            "The 24 mm object is an ideal fronto-parallel horizontal width; real orientation, blur, occlusion, matching and segmentation are not modeled.",
            "The 0.25 px disparity error is an assumed sensitivity input, not a measured error, uncertainty distribution or validated algorithm result.",
            "These geometric values do not prove object recognition, detection range, depth accuracy or successful operation at 10-50 m.",
            "The live evidence reports the current service depth range as 0.3-5.0 m; the longer distances are extrapolations beyond that reported range.",
        ],
        "estimates": estimates,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
