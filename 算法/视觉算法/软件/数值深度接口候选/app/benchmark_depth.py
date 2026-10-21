#!/usr/bin/env python3
"""Compare fresh FastSGBMDepth profiles on one fixed stereo image pair.

This tool is measurement-only: it does not change matcher settings or the depth
wire contract. Independent physical depth truth is never inferred from
calibration metadata; use --synthetic-shifts only for an analytic matcher check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from fast_depth import FastSGBMDepth


APP = Path(__file__).resolve().parent
ROOT = APP.parent
DEFAULT_LEFT = ROOT / "validation" / "left.png"
DEFAULT_RIGHT = ROOT / "validation" / "right.png"
DEFAULT_CALIBRATION = APP / "calibration.npz"
DEFAULT_SCALES = (0.375, 0.5, 0.75, 1.0)
DEFAULT_THREADS = (2, 4)
DEFAULT_PREPROCESS = ("cpu", "opencl")
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_scales(value: str) -> Tuple[float, ...]:
    try:
        result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from exc
    if not result or any(not 0.25 <= scale <= 1.0 for scale in result):
        raise argparse.ArgumentTypeError("scales must be between 0.25 and 1.0")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("scales must not repeat")
    return result


def parse_threads(value: str) -> Tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threads must be comma-separated integers") from exc
    if not result or any(count < 1 or count > 64 for count in result):
        raise argparse.ArgumentTypeError("threads must be between 1 and 64")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("thread counts must not repeat")
    return result


def parse_modes(value: str) -> Tuple[str, ...]:
    result = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not result or any(mode not in ("cpu", "opencl") for mode in result):
        raise argparse.ArgumentTypeError("preprocess modes may contain cpu and/or opencl")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("preprocess modes must not repeat")
    return result


def parse_shifts(value: str) -> Tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("synthetic shifts must be comma-separated integers") from exc
    if any(shift <= 0 for shift in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("synthetic shifts must be unique positive integers")
    return result


def parse_input_sizes(value: str, source_width: int, source_height: int
                      ) -> Tuple[Tuple[int, int], ...]:
    sizes = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "source":
            size = (int(source_width), int(source_height))
        else:
            try:
                width_text, height_text = token.split("x", 1)
                size = (int(width_text), int(height_text))
            except (ValueError, TypeError) as exc:
                raise ValueError("input sizes must look like 640x360 or source") from exc
        width, height = size
        if width < 32 or height < 32 or width > source_width or height > source_height:
            raise ValueError("input size must be >=32 and no larger than the source pair")
        if not np.isclose(width / height, source_width / source_height, rtol=0, atol=1e-4):
            raise ValueError("input size must preserve the source stereo field-of-view aspect ratio")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise ValueError("at least one input size is required")
    return tuple(sizes)


def percentile_summary(samples: Sequence[float]) -> Optional[Dict[str, float]]:
    if not samples:
        return None
    values = np.asarray(samples, dtype=np.float64)
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def coverage_summary(depth: np.ndarray, support: np.ndarray,
                     boundary_band: int = 5) -> Dict[str, float]:
    """Report finite-depth coverage and valid rate near rectification-mask edges."""
    depth = np.asarray(depth)
    support = np.asarray(support, dtype=bool)
    if depth.ndim != 2 or support.shape != depth.shape:
        raise ValueError("depth and rectification support must be matching 2-D arrays")
    valid = np.isfinite(depth)
    height, width = depth.shape
    middle = width // 2
    kernel_size = max(3, boundary_band * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    eroded = cv2.erode(support.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    boundary = support & ~eroded
    interior = support & eroded

    def fraction(mask: np.ndarray) -> Optional[float]:
        count = int(np.count_nonzero(mask))
        return float(np.count_nonzero(valid & mask) / count) if count else None

    return {
        "valid_fraction_all_pixels": float(valid.mean()),
        "valid_fraction_left_half": float(valid[:, :middle].mean()) if middle else None,
        "valid_fraction_right_half": float(valid[:, middle:].mean()) if middle < width else None,
        "rectification_support_fraction": float(support.mean()),
        "mask_boundary_band_px": int(boundary_band),
        "valid_fraction_mask_boundary": fraction(boundary),
        "valid_fraction_mask_interior": fraction(interior),
        "mask_boundary_pixels": int(np.count_nonzero(boundary)),
        "valid_pixels": int(np.count_nonzero(valid)),
        "raster_pixels": int(height * width),
    }


def summarize_coverage(rows: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    keys = tuple(rows[0]) if rows else ()
    result: Dict[str, object] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = percentile_summary(values)
    return result


def prepare_pair(left: np.ndarray, right: np.ndarray,
                 size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    width, height = size
    if left.shape != right.shape or left.ndim != 2 or right.ndim != 2:
        raise ValueError("benchmark requires matching grayscale stereo images")
    if (left.shape[1], left.shape[0]) == size:
        return left.copy(), right.copy()
    interpolation = cv2.INTER_AREA if width <= left.shape[1] and height <= left.shape[0] else cv2.INTER_LINEAR
    return (cv2.resize(left, size, interpolation=interpolation),
            cv2.resize(right, size, interpolation=interpolation))


def synthetic_shift_pair(shape: Tuple[int, int], disparity: int,
                         seed: int = 173) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic already-rectified pair with integer disparity."""
    height, width = (int(value) for value in shape)
    if disparity <= 0 or disparity >= width:
        raise ValueError("synthetic disparity must be positive and smaller than image width")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .55)
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]
    return left, right


def check_synthetic_shift(engine: FastSGBMDepth, disparity: int) -> Dict[str, object]:
    left, right = synthetic_shift_pair((engine.height, engine.width), disparity)
    disp, reverse = engine.disparities(left, right)
    depth, valid = engine.metric_values(disp, reverse, left)
    window_radius = int(engine.matcher.getBlockSize()) // 2
    margin = max(disparity + window_radius + 2, window_radius + 2)
    core = np.zeros(valid.shape, dtype=bool)
    if engine.width > 2 * margin:
        core[:, margin:engine.width - margin] = True
    else:
        core[:] = True
    selected = valid & core & np.isfinite(depth)
    expected_depth = engine.depth_factor / (float(disparity) - engine.disparity_offset)
    if np.any(selected):
        measured = depth[selected].astype(np.float64)
        error_mm = np.abs(measured - expected_depth) * 1000.0
        median_depth = float(np.median(measured))
        median_error = float(np.median(error_mm))
    else:
        median_depth = median_error = None
    count = int(np.count_nonzero(selected))
    return {
        "synthetic_disparity_px": int(disparity),
        "expected_z_m": float(expected_depth),
        "median_measured_z_m": median_depth,
        "median_absolute_error_mm": median_error,
        "valid_core_pixels": count,
        "valid_core_fraction": float(count / max(1, int(core.sum()))),
        "truth_scope": "analytic integer shift on an already rectified synthetic pair; not physical accuracy",
    }


def _load_images(left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("could not read both grayscale source images")
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left/right source image sizes must match")
    return left, right


def _cpu_model() -> Optional[str]:
    if platform.processor():
        return platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    except OSError:
        pass
    return None


def _stage_samples(details: Mapping[str, object]) -> Dict[str, float]:
    stage = details.get("stage_ms", {})
    if not isinstance(stage, Mapping):
        return {}
    keys = ("resize", "remap_rotate_crop", "preprocess_opencl_combined", "match",
            "inference_forward", "depth_filter", "colorize")
    result = {}
    for key in keys:
        value = stage.get(key)
        if value is not None:
            result[key] = float(value)
    return result


def _run_profile(left_original: np.ndarray, right_original: np.ndarray,
                 calibration_path: Path, input_size: Tuple[int, int], scale: float,
                 threads: int, preprocess: str, warmup: int, runs: int,
                 synthetic_shifts: Sequence[int],
                 engine_factory: Callable[..., FastSGBMDepth]) -> Dict[str, object]:
    left, right = prepare_pair(left_original, right_original, input_size)
    previous_mode = os.environ.get("LIGHTSTEREO_FAST_PREPROCESS")
    os.environ["LIGHTSTEREO_FAST_PREPROCESS"] = preprocess
    engine = None
    try:
        engine = engine_factory(calibration_path, scale=scale, threads=threads,
                                input_size=input_size)
        actual_preprocess = str(getattr(engine, "preprocess_backend", "cpu"))
        if actual_preprocess != preprocess:
            raise RuntimeError("requested %s preprocessing but engine selected %s" %
                               (preprocess, actual_preprocess))
        process_samples: List[float] = []
        stage_samples: Dict[str, List[float]] = {}
        coverage_samples: List[Dict[str, float]] = []
        for index in range(warmup + runs):
            start = time.perf_counter()
            _color, depth, details = engine.compute(left, right)
            wall_ms = (time.perf_counter() - start) * 1000.0
            if depth.shape != (engine.height, engine.width):
                raise RuntimeError("engine depth shape does not match configured output raster")
            if index < warmup:
                continue
            process_samples.append(wall_ms)
            for name, value in _stage_samples(details).items():
                stage_samples.setdefault(name, []).append(value)
            support = np.asarray(engine.small_masks[0], dtype=bool) & np.asarray(engine.small_masks[1], dtype=bool)
            coverage_samples.append(coverage_summary(depth, support))

        synthetic = [check_synthetic_shift(engine, shift) for shift in synthetic_shifts]
        source_height, source_width = left_original.shape
        input_pixels = int(input_size[0] * input_size[1])
        nominal_live_pixels = 640 * 360
        profile = {
            "input_resolution_wh": [input_size[0], input_size[1]],
            "input_pixels_per_eye": input_pixels,
            "input_pixels_vs_live_640x360": float(input_pixels / nominal_live_pixels),
            "original_source_resolution_wh": [source_width, source_height],
            "algorithm_scale": float(scale),
            "algorithm_rectification_resolution_wh": [int(engine.raw_width), int(engine.raw_height)],
            "algorithm_rectification_pixels_per_eye": int(engine.raw_width * engine.raw_height),
            "output_depth_resolution_wh": [int(engine.width), int(engine.height)],
            "threads": int(threads),
            "preprocess": actual_preprocess,
            "num_disparities": int(engine.num_disparities),
            "block_size": int(engine.matcher.getBlockSize()),
            "warmup_runs": int(warmup),
            "measured_runs": int(runs),
            "process_ms": percentile_summary(process_samples),
            "stage_ms": {name: percentile_summary(values)
                         for name, values in sorted(stage_samples.items())},
            "coverage": summarize_coverage(coverage_samples),
            "accuracy": {
                "independent_ground_truth": None,
                "median_error_mm": None,
                "scope": "no independent physical range truth supplied",
            },
            "synthetic_shift_checks": synthetic,
            "calibration_id": str(engine.identifier),
            "baseline_mm": float(engine.baseline_mm),
        }
        return profile
    finally:
        if engine is not None:
            del engine
        if previous_mode is None:
            os.environ.pop("LIGHTSTEREO_FAST_PREPROCESS", None)
        else:
            os.environ["LIGHTSTEREO_FAST_PREPROCESS"] = previous_mode


def run_benchmark(left_path: Path, right_path: Path, calibration_path: Path,
                  output_path: Path, *,
                  scales: Sequence[float] = DEFAULT_SCALES,
                  threads: Sequence[int] = DEFAULT_THREADS,
                  input_sizes: Optional[Sequence[Tuple[int, int]]] = None,
                  preprocess_modes: Sequence[str] = DEFAULT_PREPROCESS,
                  warmup: int = DEFAULT_WARMUP,
                  runs: int = DEFAULT_RUNS,
                  synthetic_shifts: Sequence[int] = (),
                  engine_factory: Callable[..., FastSGBMDepth] = FastSGBMDepth
                  ) -> Dict[str, object]:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >=0 and runs must be >=1")
    left_path, right_path, calibration_path = map(Path, (left_path, right_path, calibration_path))
    left, right = _load_images(left_path, right_path)
    source_height, source_width = left.shape
    if input_sizes is None:
        input_sizes = parse_input_sizes("640x360,source", source_width, source_height)
    modes = tuple(preprocess_modes)
    if not modes or any(mode not in ("cpu", "opencl") for mode in modes):
        raise ValueError("preprocess_modes must select cpu and/or opencl")
    original_threads = cv2.getNumThreads()
    opencl_was_enabled = bool(cv2.ocl.useOpenCL())
    opencl_available = bool(cv2.ocl.haveOpenCL())
    profiles: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    try:
        for input_size in input_sizes:
            # Prepare each resolution once; every profile still gets a fresh engine.
            prepare_pair(left, right, input_size)
            for scale in scales:
                for thread_count in threads:
                    for mode in modes:
                        descriptor = dict(input_resolution_wh=list(input_size), scale=float(scale),
                                          threads=int(thread_count), preprocess_requested=mode)
                        if mode == "opencl" and not opencl_available:
                            skipped.append(dict(descriptor, reason="OpenCL not reported available by OpenCV"))
                            continue
                        try:
                            profile = _run_profile(left, right, calibration_path, input_size,
                                                   float(scale), int(thread_count), mode,
                                                   warmup, runs, synthetic_shifts, engine_factory)
                            profiles.append(profile)
                        except RuntimeError as exc:
                            if mode == "opencl" and "selected cpu" in str(exc):
                                skipped.append(dict(descriptor, reason=str(exc)))
                            else:
                                errors.append(dict(descriptor, error=str(exc)))
                        except Exception as exc:
                            errors.append(dict(descriptor, error=type(exc).__name__ + ": " + str(exc)))
    finally:
        cv2.setNumThreads(original_threads)
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(opencl_was_enabled)
    if not profiles:
        raise RuntimeError("no benchmark profile completed; inspect skipped/errors in the exception context")

    output = Path(output_path)
    source_path = Path(__file__).with_name("fast_depth.py").resolve()
    report: Dict[str, object] = {
        "schema": "fast-sgbm-profile-benchmark-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pair": {
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_sha256": sha256_file(left_path),
            "right_sha256": sha256_file(right_path),
            "resolution_wh": [source_width, source_height],
            "independent_ground_truth": None,
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
        },
        "implementation": {
            "fast_depth_path": str(source_path),
            "fast_depth_sha256": sha256_file(source_path),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": _cpu_model(),
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "opencv_threads_before": original_threads,
            "opencv_opencl_available": opencl_available,
        },
        "comparison_notes": {
            "same_original_pair_for_all_profiles": True,
            "input_sizes_are_resized_from_source_with_inter_area_when_downscaling": True,
            "different_input_pixel_counts_are_reported_separately": True,
            "fresh_engine_per_profile": True,
            "coverage_is_not_accuracy": True,
            "depth_profile_m": [0.3, 5.0],
            "source_photo_is_not_independent_range_truth": True,
        },
        "profiles": profiles,
        "skipped_profiles": skipped,
        "profile_errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fresh FastSGBMDepth profiles on a fixed stereo pair")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT,
                        help="fixed raw left grayscale source image (default: validation/left.png)")
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT,
                        help="fixed raw right grayscale source image (default: validation/right.png)")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, required=True, help="JSON destination")
    parser.add_argument("--scales", type=parse_scales, default=DEFAULT_SCALES)
    parser.add_argument("--threads", type=parse_threads, default=DEFAULT_THREADS)
    parser.add_argument("--input-sizes", default="640x360,source",
                        help="comma-separated WxH values and/or source; default 640x360,source")
    parser.add_argument("--preprocess", type=parse_modes, default=DEFAULT_PREPROCESS,
                        help="cpu,opencl; unavailable OpenCL is reported and skipped")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--synthetic-shifts", type=parse_shifts, default=(),
                        help="optional analytic already-rectified disparity tests, e.g. 8,24")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1")
    try:
        left, right = _load_images(args.left, args.right)
        expected = FastSGBMDepth.SENSOR_SIZE[::-1]
        if left.shape != expected:
            raise ValueError("fixed calibrated source pair must be %dx%d grayscale" %
                             (expected[1], expected[0]))
        sizes = parse_input_sizes(args.input_sizes, left.shape[1], left.shape[0])
        report = run_benchmark(
            args.left, args.right, args.calibration, args.output,
            scales=args.scales, threads=args.threads, input_sizes=sizes,
            preprocess_modes=args.preprocess, warmup=args.warmup,
            runs=args.runs, synthetic_shifts=args.synthetic_shifts)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"ok": True, "output": str(args.output),
                      "profiles": len(report["profiles"]),
                      "skipped": len(report["skipped_profiles"]),
                      "errors": len(report["profile_errors"])}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
