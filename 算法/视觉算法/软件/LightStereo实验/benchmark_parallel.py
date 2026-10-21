"""Board-only LightStereo serial vs two-session NPU benchmark."""
from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import rknnlite.api.rknn_lite as lite
from rknnlite.api.rknn_runtime import RKNNRuntime


ROOT = Path("/opt/rk-lightstereo")
MODEL = ROOT / "models/lightstereo_640_352.rknn"
RUNTIME = ROOT / "runtime/librknnrt.so"
REFERENCES = {
    "forward": ROOT / "validation/reference.npz",
    "reverse": ROOT / "validation/reference-reverse.npz",
}
OUTPUT = Path("/tmp/parallel-npu.json")
WARMUP = 2
RUNS = 6


def make_rknn_class():
    class IsolatedRuntime(RKNNRuntime):
        def _get_rknn_api_lib_path(self):
            return str(RUNTIME)

    # RKNNLite reads this class when constructing each session. The override
    # is process-local and points to the same package runtime as the service.
    lite.RKNNRuntime = IsolatedRuntime
    from rknnlite.api import RKNNLite
    return RKNNLite


def load_reference(path):
    with np.load(path, allow_pickle=False) as data:
        left = np.ascontiguousarray(data["left"])
        right = np.ascontiguousarray(data["right"])
        expected = np.asarray(data["disparity"], dtype=np.float32).squeeze()
        width = int(data["content_width"]) if "content_width" in data else expected.shape[1]
        top = int(data["pad_top"]) if "pad_top" in data else 0
    if left.shape != (1, 640, 352, 3) or right.shape != left.shape:
        raise ValueError("unexpected reference input shape: " + str(path))
    if left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("reference inputs must be uint8")
    return {"inputs": [left, right], "expected": expected, "width": width, "top": top}


def create_session(RKNNLite, mask):
    session = RKNNLite(verbose=False)
    if session.load_rknn(str(MODEL)) != 0:
        session.release()
        raise RuntimeError("model load failed")
    if session.init_runtime(core_mask=mask) != 0:
        session.release()
        raise RuntimeError("runtime init failed for mask " + str(mask))
    runtime_check()
    return session


def infer(session, ref):
    outputs = session.inference(inputs=ref["inputs"], data_format=["nhwc", "nhwc"])
    if not outputs or outputs[0] is None:
        raise RuntimeError("inference returned no output")
    prediction = np.asarray(outputs[0], dtype=np.float32).squeeze()
    if prediction.shape != (640, 352) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid output shape or values: " + str(prediction.shape))
    return prediction


def region(ref, values):
    return values[ref["top"]:, :ref["width"]]


def stats(values):
    values = np.abs(np.asarray(values, dtype=np.float32)).reshape(-1)
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def delta(reference, prediction):
    return stats(region(reference, prediction - reference["expected"]))


def timing(values):
    return {"median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}


def runtime_check():
    maps = Path("/proc/self/maps").read_text(errors="replace")
    if str(RUNTIME) not in maps:
        raise RuntimeError("package librknnrt.so is not mapped")
    if "/usr/lib/librknnrt.so" in maps:
        raise RuntimeError("unexpected system librknnrt.so is mapped")


def serial_case(RKNNLite, refs):
    session = create_session(RKNNLite, 7)
    try:
        last = {}
        for _ in range(WARMUP):
            for name in refs:
                last[name] = infer(session, refs[name])
        elapsed = []
        for _ in range(RUNS):
            start = time.perf_counter()
            for name in refs:
                last[name] = infer(session, refs[name])
            elapsed.append((time.perf_counter() - start) * 1000)
        return {"pair_ms": timing(elapsed), "outputs": last}
    finally:
        session.release()


def parallel_case(RKNNLite, refs, masks):
    sessions = {name: create_session(RKNNLite, masks[name]) for name in refs}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            last = {}
            for _ in range(WARMUP):
                futures = {name: pool.submit(infer, sessions[name], refs[name]) for name in refs}
                last = {name: future.result() for name, future in futures.items()}
            elapsed = []
            for _ in range(RUNS):
                start = time.perf_counter()
                futures = {name: pool.submit(infer, sessions[name], refs[name]) for name in refs}
                last = {name: future.result() for name, future in futures.items()}
                elapsed.append((time.perf_counter() - start) * 1000)
        return {"pair_ms": timing(elapsed), "outputs": last, "masks": masks}
    finally:
        for session in sessions.values():
            session.release()


def report_deltas(refs, serial, parallel):
    result = {}
    for name in refs:
        result[name] = {
            "serial_vs_onnx_px": delta(refs[name], serial[name]),
            "parallel_vs_onnx_px": delta(refs[name], parallel[name]),
            "parallel_vs_serial_px": stats(region(refs[name], parallel[name] - serial[name])),
        }
    return result


def thresholds_pass(differences):
    return all(values["mean"] < 0.2 and values["p95"] < 0.6 and values["max"] < 5.0
               for values in differences.values())


def main():
    if not MODEL.is_file() or not RUNTIME.is_file():
        raise FileNotFoundError("model or package runtime missing")
    refs = {name: load_reference(path) for name, path in REFERENCES.items()}
    RKNNLite = make_rknn_class()

    serial = serial_case(RKNNLite, refs)
    pair12 = parallel_case(RKNNLite, refs, {"forward": 1, "reverse": 2})
    report = {
        "model": str(MODEL), "runtime": str(RUNTIME), "warmup_pairs": WARMUP,
        "timed_pairs": RUNS, "input": "pre-normalized uint8 NHWC references",
        "serial_mask7": {"pair_ms": serial["pair_ms"]},
        "parallel_mask1_2": {"pair_ms": pair12["pair_ms"], "masks": pair12["masks"]},
        "parallel_mask1_2_speedup": float(serial["pair_ms"]["median_ms"] /
                                           pair12["pair_ms"]["median_ms"]),
        "differences_px": report_deltas(refs, serial["outputs"], pair12["outputs"]),
        "thresholds_px": {"onnx_mean_lt": 0.2, "onnx_p95_lt": 0.6, "onnx_max_lt": 5.0},
        "scope": "NPU inference only; no camera, service, depth.jpg, frequency or network access",
    }
    report["thresholds_pass"] = {
        "serial_vs_onnx": thresholds_pass({name: report["differences_px"][name]["serial_vs_onnx_px"]
                                            for name in refs}),
        "parallel_mask1_2_vs_onnx": thresholds_pass({name: report["differences_px"][name]["parallel_vs_onnx_px"]
                                                      for name in refs}),
    }

    # A single additional mask3/mask4 comparison is useful only when the
    # separate-core pair failed to beat the serial baseline by at least 5%.
    if pair12["pair_ms"]["median_ms"] >= serial["pair_ms"]["median_ms"] * 0.95:
        pair34 = parallel_case(RKNNLite, refs, {"forward": 3, "reverse": 4})
        report["parallel_mask3_4"] = {"pair_ms": pair34["pair_ms"], "masks": pair34["masks"]}
        report["parallel_mask3_4_speedup"] = float(serial["pair_ms"]["median_ms"] /
                                                   pair34["pair_ms"]["median_ms"])
        report["differences_px"]["mask3_4"] = report_deltas(refs, serial["outputs"], pair34["outputs"])
        report["thresholds_pass"]["parallel_mask3_4_vs_onnx"] = thresholds_pass(
            {name: report["differences_px"]["mask3_4"][name]["parallel_vs_onnx_px"] for name in refs})
    else:
        report["mask3_4"] = "skipped: mask1/2 pair was at least 5% faster than serial"

    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
