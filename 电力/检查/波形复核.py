"""Extract evidence from a synchronized bench CSV for one test and one bus island."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


EVENTS = ("main_req", "main_low_r", "load_valid")
COLUMNS = {
    "precharge": ("time_s", "pack_v", "bus_v", "precharge_a", "pre_resistor_v", "main_a", *EVENTS),
    "brake": ("time_s", "bus_v", "regen_w", "load_w", "battery_w", "supercap_w", "dump_v", "dump_a"),
}


def read_samples(raw: bytes, mode: str) -> list[dict[str, float]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError("duplicate CSV column names")
    missing = set(COLUMNS[mode]) - set(headers)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    columns = COLUMNS[mode]
    if mode == "precharge" and "pre_pass_v" in headers:
        columns = (*columns, "pre_pass_v")

    samples: list[dict[str, float]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"row {row_number}: more values than columns")
        try:
            sample = {name: float(row[name]) for name in columns}
        except (TypeError, ValueError) as error:
            raise ValueError(f"row {row_number}: missing or invalid number") from error
        if not all(math.isfinite(value) for value in sample.values()):
            raise ValueError(f"row {row_number}: non-finite sample")
        if samples and sample["time_s"] <= samples[-1]["time_s"]:
            raise ValueError(f"row {row_number}: time_s must strictly increase")
        if mode == "precharge" and any(sample[name] not in (0, 1) for name in EVENTS):
            raise ValueError(f"row {row_number}: event levels must be 0 or 1")
        samples.append(sample)
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    return samples


def integrate(time_s: list[float], power_w: list[float]) -> tuple[float, float]:
    """Return final and peak cumulative energy using piecewise-linear power."""
    total = peak = 0.0
    for start, end, p_start, p_end in zip(time_s, time_s[1:], power_w, power_w[1:]):
        duration = end - start
        if p_start > 0 > p_end:
            positive_duration = duration * (p_start / (p_start - p_end))
            peak = max(peak, total + positive_duration * p_start / 2)
        total += duration * (p_start / 2 + p_end / 2)
        peak = max(peak, total)
    if not math.isfinite(total) or not math.isfinite(peak):
        raise ValueError("energy integral overflowed")
    return total, peak


def precharge_metrics(time_s: list[float], samples: list[dict[str, float]]) -> dict:
    report = {
        "event_rises": {
            name: {
                "initial_level": int(samples[0][name]),
                "edges": [
                    {"previous_sample_s": previous["time_s"], "sample_s": sample["time_s"],
                     "sampled_pack_minus_bus_v": sample["pack_v"] - sample["bus_v"]}
                    for previous, sample in zip(samples, samples[1:])
                    if previous[name] == 0 and sample[name] == 1
                ],
            }
            for name in EVENTS
        },
        "peak_abs_main_current_a": max(abs(sample["main_a"]) for sample in samples),
        "peak_abs_precharge_current_a": max(abs(sample["precharge_a"]) for sample in samples),
        "precharge_resistor_energy_j": integrate(
            time_s, [sample["pre_resistor_v"] * sample["precharge_a"] for sample in samples]
        )[0],
    }
    if "pre_pass_v" in samples[0]:
        power_w = [sample["pre_pass_v"] * sample["precharge_a"] for sample in samples]
        peak = max(range(len(samples)), key=power_w.__getitem__)
        report["precharge_pass_fet_energy_j"] = integrate(time_s, power_w)[0]
        report["precharge_pass_fet_peak_sample"] = {
            "time_s": time_s[peak],
            "vds_v": samples[peak]["pre_pass_v"],
            "current_a": samples[peak]["precharge_a"],
            "power_w": power_w[peak],
        }
    return report


def precharge_intervals(samples: list[dict[str, float]], threshold_a: float) -> list[dict]:
    """Bracket observed current pulses without assuming a crossing time between samples."""
    intervals = []
    first_high = None
    above_threshold = [abs(sample["precharge_a"]) >= threshold_a for sample in samples]
    for index, active in enumerate([*above_threshold, False]):
        if active:
            if first_high is None:
                first_high = index
            continue
        if first_high is not None:
            first_s = samples[first_high]["time_s"]
            last_s = samples[index - 1]["time_s"]
            before_s = samples[first_high - 1]["time_s"] if first_high else None
            after_s = samples[index]["time_s"] if index < len(samples) else None
            intervals.append({
                "previous_below_s": before_s,
                "first_at_or_above_s": first_s,
                "last_at_or_above_s": last_s,
                "next_below_s": after_s,
                "observed_span_s": last_s - first_s,
                "bracket_span_s": after_s - before_s if before_s is not None and after_s is not None else None,
            })
            first_high = None
    return intervals


def brake_metrics(time_s: list[float], samples: list[dict[str, float]]) -> dict:
    dump_a = [sample["dump_a"] for sample in samples]
    dump_w = [sample["dump_v"] * sample["dump_a"] for sample in samples]
    residual_w = [
        sample["regen_w"] - sample["load_w"] - sample["battery_w"] - sample["supercap_w"] - power
        for sample, power in zip(samples, dump_w)
    ]
    residual_j, peak_residual_j = integrate(time_s, residual_w)
    dump_i2t = sum(
        (end - start) * (i_start * i_start + i_start * i_end + i_end * i_end) / 3
        for start, end, i_start, i_end in zip(time_s, time_s[1:], dump_a, dump_a[1:])
    )
    if not math.isfinite(dump_i2t):
        raise ValueError("dump resistor I²t integral overflowed")
    return {
        "peak_sampled_bus_v": max(sample["bus_v"] for sample in samples),
        "dump_resistor_peak_abs_current_a": max(map(abs, dump_a)),
        "dump_resistor_i2t_a2s": dump_i2t,
        "dump_resistor_energy_j": integrate(time_s, dump_w)[0],
        "residual_energy_end_j": residual_j,
        "residual_energy_peak_j": peak_residual_j,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=COLUMNS)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--precharge-current-threshold-a", type=float,
                        help="report sampled precharge current intervals at or above this magnitude")
    parser.add_argument("--brake-start-s", type=float,
                        help="start brake integration and peak calculation at this CSV sample time")
    args = parser.parse_args()
    threshold = args.precharge_current_threshold_a
    brake_start = args.brake_start_s
    if threshold is not None and (args.mode != "precharge" or not math.isfinite(threshold) or threshold <= 0):
        parser.error("the current threshold must be finite, positive and used in precharge mode")
    if brake_start is not None and (args.mode != "brake" or not math.isfinite(brake_start)):
        parser.error("the brake start time must be finite and used in brake mode")
    try:
        raw = args.csv_file.read_bytes()
        samples = read_samples(raw, args.mode)
        time_s = [sample["time_s"] for sample in samples]
        start = 0
        if brake_start is not None:
            if brake_start not in time_s or brake_start == time_s[-1]:
                raise ValueError("brake start time must match a CSV sample before the final sample")
            start = time_s.index(brake_start)
        metrics = precharge_metrics if args.mode == "precharge" else brake_metrics
        report = {
            "mode": args.mode,
            "qualification": "evidence_only",
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "sample_count": len(samples),
            "first_sample_s": time_s[0],
            "last_sample_s": time_s[-1],
            "max_sample_gap_s": max(end - start for start, end in zip(time_s, time_s[1:])),
            **metrics(time_s[start:], samples[start:]),
        }
        if brake_start is not None:
            report["brake_start_s"] = time_s[start]
            report["brake_window_sample_count"] = len(samples) - start
        if threshold is not None:
            report["precharge_current_threshold_a"] = threshold
            report["precharge_current_intervals"] = precharge_intervals(samples, threshold)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
