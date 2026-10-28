"""Camera-free RKNN forward benchmark for the board's packaged model/runtime.

The module deliberately imports rknnlite only when a real benchmark starts, so
--help and the pure summary helpers work on a development host without RKNN.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np


DEFAULT_MODEL = "/opt/rk-lightstereo/models/lightstereo_640_352.rknn"
DEFAULT_REFERENCE = "/opt/rk-lightstereo/validation/reference.npz"
DEFAULT_RUNTIME = "/opt/rk-lightstereo/runtime/librknnrt.so"
DEFAULT_OUTPUT = "/tmp/rk-lightstereo-npu-forward-benchmark.json"
EXPECTED_INPUT_SHAPE = (1, 640, 352, 3)
REFERENCE_TOLERANCE = {"mean_lt_px": 0.2, "p95_lt_px": 0.6, "max_lt_px": 5.0}
PROFILES = (
    ("single_session_core1", (1,)),
    ("single_session_core7", (7,)),
    ("three_sessions_core1_2_4", (1, 2, 4)),
)
SCOPE = (
    "RKNN forward inference only on the saved NHWC reference tensors; excludes "
    "camera capture, rectification, reverse inference, consistency/post-processing, "
    "and UI/transport."
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed RKNN forward-only profiles with no camera access. "
            "The three-session profile reports combined throughput separately "
            "from per-inference latency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="RKNN model file")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="NPZ containing left/right NHWC tensors and reference disparity")
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME,
                        help="isolated librknnrt.so path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON report path")
    parser.add_argument("--runs", type=positive_int, default=6,
                        help="measured inferences per session")
    parser.add_argument("--warmup", type=nonnegative_int, default=2,
                        help="unmeasured warmup inferences per session")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def file_metadata(path: Path) -> dict:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def load_reference(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as data:
        required = ("left", "right", "disparity")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("reference NPZ is missing: " + ", ".join(missing))
        left = np.asarray(data["left"])
        right = np.asarray(data["right"])
        expected = np.asarray(data["disparity"]).squeeze()
        content_width = int(np.asarray(data["content_width"]).item()) \
            if "content_width" in data else int(expected.shape[1])
        pad_top = int(np.asarray(data["pad_top"]).item()) \
            if "pad_top" in data else 0

    if left.shape != EXPECTED_INPUT_SHAPE or right.shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            "reference inputs must both have shape %r; received left=%r right=%r"
            % (EXPECTED_INPUT_SHAPE, left.shape, right.shape)
        )
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("reference left/right tensors must be uint8")
    if expected.ndim != 2:
        raise ValueError("reference disparity must be a 2-D tensor after squeeze")
    if not 0 < content_width <= expected.shape[1]:
        raise ValueError("reference content_width is outside the disparity tensor")
    if not 0 <= pad_top < expected.shape[0]:
        raise ValueError("reference pad_top is outside the disparity tensor")

    return {
        "left": np.ascontiguousarray(left),
        "right": np.ascontiguousarray(right),
        "expected": np.ascontiguousarray(expected),
        "content_width": content_width,
        "pad_top": pad_top,
    }


def percentile_summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("percentile summary requires a non-empty 1-D sequence")
    if not np.isfinite(array).all():
        raise ValueError("percentile inputs must all be finite")
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def compare_reference_output(prediction, expected, content_width: int, pad_top: int) -> dict:
    predicted = np.asarray(prediction).squeeze()
    reference = np.asarray(expected).squeeze()
    prediction_numeric = bool(np.issubdtype(predicted.dtype, np.number))
    reference_numeric = bool(np.issubdtype(reference.dtype, np.number))
    prediction_finite = bool(prediction_numeric and np.isfinite(predicted).all())
    reference_finite = bool(reference_numeric and np.isfinite(reference).all())
    shape_match = predicted.shape == reference.shape
    result = {
        "prediction_shape": list(predicted.shape),
        "expected_shape": list(reference.shape),
        "shape_match": bool(shape_match),
        "prediction_dtype": str(predicted.dtype),
        "expected_dtype": str(reference.dtype),
        "prediction_all_finite": prediction_finite,
        "expected_all_finite": reference_finite,
        "comparison_region": {
            "content_width": int(content_width),
            "pad_top": int(pad_top),
        },
        "absolute_disparity_delta_px": None,
        "within_reference_tolerance": False,
        "tolerance": dict(REFERENCE_TOLERANCE),
    }
    if not shape_match or predicted.ndim != 2 or not prediction_numeric or not reference_numeric:
        return result
    if not 0 < int(content_width) <= reference.shape[1]:
        raise ValueError("content_width is outside the reference tensor")
    if not 0 <= int(pad_top) < reference.shape[0]:
        raise ValueError("pad_top is outside the reference tensor")

    pred_roi = predicted[int(pad_top):, :int(content_width)].astype(np.float64, copy=False)
    ref_roi = reference[int(pad_top):, :int(content_width)].astype(np.float64, copy=False)
    finite = np.isfinite(pred_roi) & np.isfinite(ref_roi)
    finite_count = int(finite.sum())
    total_count = int(finite.size)
    result["comparison_region"].update({
        "finite_pixels": finite_count,
        "total_pixels": total_count,
        "finite_fraction": float(finite_count / total_count) if total_count else 0.0,
    })
    if finite_count == 0:
        return result

    delta = np.abs(pred_roi[finite] - ref_roi[finite])
    stats = {
        "mean": float(delta.mean()),
        "p50": float(np.percentile(delta, 50)),
        "p95": float(np.percentile(delta, 95)),
        "max": float(delta.max()),
    }
    result["absolute_disparity_delta_px"] = stats
    result["within_reference_tolerance"] = bool(
        prediction_finite and reference_finite
        and stats["mean"] < REFERENCE_TOLERANCE["mean_lt_px"]
        and stats["p95"] < REFERENCE_TOLERANCE["p95_lt_px"]
        and stats["max"] < REFERENCE_TOLERANCE["max_lt_px"]
    )
    return result


def summarize_profile(name: str, core_masks, session_latencies_ms, wall_clock_s: float) -> dict:
    masks = [int(mask) for mask in core_masks]
    sessions = [list(values) for values in session_latencies_ms]
    if not masks or len(masks) != len(sessions):
        raise ValueError("each profile core mask must have one session latency list")
    if any(not values for values in sessions):
        raise ValueError("each session must contain at least one measured inference")
    if not np.isfinite(float(wall_clock_s)) or float(wall_clock_s) <= 0:
        raise ValueError("wall_clock_s must be finite and positive")

    all_latencies = [float(value) for values in sessions for value in values]
    if any(not np.isfinite(value) or value < 0 for value in all_latencies):
        raise ValueError("latencies must be finite and non-negative")
    inference_count = len(all_latencies)
    per_session = [
        {
            "core_mask": mask,
            "runs": len(values),
            "latency_ms": percentile_summary(values),
        }
        for mask, values in zip(masks, sessions)
    ]
    return {
        "name": str(name),
        "session_count": len(masks),
        "core_masks": masks,
        "runs_per_session": [len(values) for values in sessions],
        "measured_inference_count": inference_count,
        "per_inference_latency_ms": percentile_summary(all_latencies),
        "per_session": per_session,
        "measured_wall_clock_s": float(wall_clock_s),
        "aggregate_throughput_fps": float(inference_count / float(wall_clock_s)),
        "throughput_semantics": (
            "completed forward calls across all profile sessions, including concurrent sessions, "
            "divided by measured wall time"
        ),
    }


def _load_rknn_lite_class(runtime_path: Path):
    try:
        import rknnlite.api.rknn_lite as lite
        from rknnlite.api.rknn_runtime import RKNNRuntime
    except Exception as exc:
        raise RuntimeError("rknnlite2 is required only for a real board benchmark: %s" % exc) from exc

    isolated_path = str(Path(runtime_path).resolve(strict=True))

    class IsolatedRuntime(RKNNRuntime):
        def _get_rknn_api_lib_path(self):
            return isolated_path

    # This override is scoped to this standalone process; it does not touch disk.
    lite.RKNNRuntime = IsolatedRuntime
    from rknnlite.api import RKNNLite
    return RKNNLite


def _verify_runtime_mapping(runtime_path: Path) -> dict:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        raise RuntimeError("cannot verify the loaded RKNN runtime via /proc/self/maps")
    maps_text = maps_path.read_text(encoding="utf-8", errors="replace")
    requested = str(Path(runtime_path).absolute())
    resolved = str(Path(runtime_path).resolve(strict=True))
    mapped_lines = [line for line in maps_text.splitlines() if "librknnrt.so" in line]
    requested_mapped = any(requested in line or resolved in line for line in mapped_lines)
    system_runtime_mapped = any(
        "/usr/lib/" in line and "librknnrt.so" in line
        for line in mapped_lines
    )
    if not requested_mapped:
        raise RuntimeError("requested isolated runtime is not present in this process maps")
    if system_runtime_mapped:
        raise RuntimeError("system /usr/lib/librknnrt.so was also mapped")
    return {
        "requested_path": requested,
        "resolved_path": resolved,
        "requested_runtime_mapped": True,
        "system_runtime_mapped": False,
        "mapped_runtime_lines": mapped_lines,
    }


def _create_session(RKNNLite, model_path: Path, core_mask: int):
    session = RKNNLite(verbose=False)
    try:
        load_result = session.load_rknn(str(model_path))
        if load_result != 0:
            raise RuntimeError("RKNN model load failed for core mask %d: %s" % (core_mask, load_result))
        init_result = session.init_runtime(core_mask=int(core_mask))
        if init_result != 0:
            raise RuntimeError("RKNN runtime init failed for core mask %d: %s" % (core_mask, init_result))
        return session
    except Exception:
        try:
            session.release()
        except Exception:
            pass
        raise


def _first_output(result):
    if result is None:
        raise RuntimeError("RKNN inference returned no outputs")
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    if len(outputs) == 0 or outputs[0] is None:
        raise RuntimeError("RKNN inference returned no first output")
    return outputs[0]


def run_profile(name: str, core_masks, RKNNLite, model_path: Path, runtime_path: Path,
                reference: dict, runs: int, warmup: int) -> dict:
    sessions = []
    init_times_ms = []
    try:
        for core_mask in core_masks:
            start = time.perf_counter()
            sessions.append(_create_session(RKNNLite, model_path, core_mask))
            init_times_ms.append((time.perf_counter() - start) * 1000.0)

        runtime_mapping = _verify_runtime_mapping(runtime_path)
        session_inputs = [
            [reference["left"].copy(), reference["right"].copy()]
            for _ in sessions
        ]
        warmup_start = time.perf_counter()
        for session, inputs in zip(sessions, session_inputs):
            for _ in range(warmup):
                _first_output(session.inference(inputs=inputs, data_format=["nhwc", "nhwc"]))
        warmup_total_ms = (time.perf_counter() - warmup_start) * 1000.0

        barrier = threading.Barrier(len(sessions) + 1)

        def measure_session(session, inputs):
            barrier.wait()
            latencies = []
            first = None
            for index in range(runs):
                start = time.perf_counter()
                output = session.inference(inputs=inputs, data_format=["nhwc", "nhwc"])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                first_output = _first_output(output)
                latencies.append(elapsed_ms)
                if index == 0:
                    first = np.array(first_output, copy=True)
            return latencies, first

        with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            futures = [
                executor.submit(measure_session, session, inputs)
                for session, inputs in zip(sessions, session_inputs)
            ]
            measurement_start = time.perf_counter()
            barrier.wait()
            results = [future.result() for future in futures]
            wall_clock_s = time.perf_counter() - measurement_start

        latencies_by_session = [item[0] for item in results]
        first_outputs = [item[1] for item in results]
        summary = summarize_profile(name, core_masks, latencies_by_session, wall_clock_s)
        summary["session_initialization_ms"] = [float(value) for value in init_times_ms]
        summary["warmup_runs_per_session"] = int(warmup)
        summary["warmup_total_ms_excluded"] = float(warmup_total_ms)
        summary["initialization_excluded_from_measurement"] = True
        summary["warmup_excluded_from_measurement"] = True
        summary["runtime_isolation"] = runtime_mapping
        summary["first_output_reference_checks"] = [
            {
                "core_mask": int(core_mask),
                **compare_reference_output(
                    first_output,
                    reference["expected"],
                    reference["content_width"],
                    reference["pad_top"],
                ),
            }
            for core_mask, first_output in zip(core_masks, first_outputs)
        ]
        if len(sessions) > 1:
            summary["concurrency_note"] = (
                "aggregate throughput combines concurrent sessions; latency percentiles "
                "remain per inference and are not single-profile wall-clock latency"
            )
        return summary
    finally:
        for session in sessions:
            try:
                session.release()
            except Exception:
                pass


def _existing_file(path: str, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError("%s does not exist or is not a file: %s" % (label, candidate))
    return candidate.resolve(strict=True)


def run_benchmark(model: str, reference_path: str, runtime: str, output: str,
                  runs: int = 6, warmup: int = 2) -> dict:
    if int(runs) < 1 or int(warmup) < 0:
        raise ValueError("runs must be >= 1 and warmup must be >= 0")
    model_path = _existing_file(model, "model")
    reference_file = _existing_file(reference_path, "reference")
    runtime_path = _existing_file(runtime, "runtime")
    output_path = Path(output).resolve(strict=False)
    if output_path in {model_path, reference_file, runtime_path}:
        raise ValueError("output path must not overwrite model, reference, or runtime")

    reference = load_reference(reference_file)
    RKNNLite = _load_rknn_lite_class(runtime_path)
    profiles = [
        run_profile(name, masks, RKNNLite, model_path, runtime_path, reference, runs, warmup)
        for name, masks in PROFILES
    ]
    report = {
        "schema": "rknn-forward-runtime-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "process_id": int(os.getpid()),
        "output_path": str(output_path),
        "inputs": {
            "model": file_metadata(model_path),
            "reference_file": file_metadata(reference_file),
            "runtime": file_metadata(runtime_path),
            "reference_tensors": {
                name: {
                    "shape": list(reference[name].shape),
                    "dtype": str(reference[name].dtype),
                    "sha256": sha256_array(reference[name]),
                }
                for name in ("left", "right", "expected")
            },
            "content_width": int(reference["content_width"]),
            "pad_top": int(reference["pad_top"]),
        },
        "settings": {
            "runs_per_session": int(runs),
            "warmup_per_session": int(warmup),
            "profiles": [{"name": name, "core_masks": list(masks)} for name, masks in PROFILES],
        },
        "scope": SCOPE,
        "comparison_notes": {
            "latency_ms": "per-call RKNN forward latency; p50/p95 are pooled across profile sessions",
            "throughput_fps": "completed forward calls divided by measured wall time",
            "three_session_profile": (
                "throughput is combined across core masks 1/2/4 and must not be read as "
                "single-frame latency or single-session FPS"
            ),
            "initialization_and_warmup": "excluded from measured latency and throughput",
            "reference_tolerance": dict(REFERENCE_TOLERANCE),
        },
        "profiles": profiles,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            model=args.model,
            reference_path=args.reference,
            runtime=args.runtime,
            output=args.output,
            runs=args.runs,
            warmup=args.warmup,
        )
    except Exception as exc:
        parser.exit(2, "benchmark_npu: %s: %s\n" % (type(exc).__name__, exc))
    print(json.dumps(report, separators=(",", ":"), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
