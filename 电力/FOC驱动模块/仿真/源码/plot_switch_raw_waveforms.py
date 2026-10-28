#!/usr/bin/env python3
"""Plot existing ngspice binary raw traces; this script does not rerun circuits."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "结果"
CASES = {
    "drive_positive": {
        "name": "switch_positive_filtered",
        "title": "Positive drive request (+1.8 kW command)",
        "color": "#1565c0",
    },
    "regen_negative": {
        "name": "switch_filtered_tvs",
        "title": "Regenerative request (−1.8 kW command)",
        "color": "#b45309",
    },
}
SAMPLE_PERIOD_S = 1e-6


def load_binary_raw(path: Path) -> tuple[dict[str, int], np.ndarray]:
    with path.open("rb") as f:
        header = f.read(128 * 1024)
    marker = b"Binary:\n"
    marker_at = header.find(marker)
    if marker_at < 0:
        raise ValueError(f"missing ngspice Binary marker: {path}")
    text = header[:marker_at].decode("ascii")
    offset = marker_at + len(marker)
    nvars = int(re.search(r"^No\. Variables:\s*(\d+)", text, re.M).group(1))
    npoints = int(re.search(r"^No\. Points:\s*(\d+)", text, re.M).group(1))
    variables: dict[str, int] = {}
    in_vars = False
    for line in text.splitlines():
        if line.strip() == "Variables:":
            in_vars = True
            continue
        if in_vars:
            match = re.match(r"\s*(\d+)\s+([^\s]+)\s+[^\s]+", line)
            if match:
                variables[match.group(2).lower()] = int(match.group(1))
    if len(variables) != nvars:
        raise ValueError(f"raw header declares {nvars} variables, parsed {len(variables)}")
    expected_size = offset + npoints * nvars * 8
    if path.stat().st_size != expected_size:
        raise ValueError(f"binary payload size mismatch for {path}")
    points = np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=(npoints, nvars))
    return variables, points


def get_vector(points: np.ndarray, variables: dict[str, int], name: str) -> np.ndarray:
    return np.asarray(points[:, variables[name.lower()]], dtype=np.float64)


def parse_log(path: Path) -> dict[str, tuple[float, float | None]]:
    pattern = re.compile(
        r"^([a-zA-Z0-9_]+)\s*=\s*([+-]?[0-9.]+(?:e[+-]?\d+)?)"
        r"(?:\s+at=\s*([+-]?[0-9.]+(?:e[+-]?\d+)?))?\s*$",
        re.I,
    )
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            out[match.group(1).lower()] = (
                float(match.group(2)), float(match.group(3)) if match.group(3) else None
            )
    return out


def load_case(case_id: str, spec: dict) -> tuple[dict, dict[str, np.ndarray]]:
    name = spec["name"]
    folder = RESULTS / name
    raw_path = folder / f"{name}_summary.raw"
    params = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    result = json.loads((RESULTS / f"{name}_results.json").read_text(encoding="utf-8"))["case"]["measures"]
    variables, points = load_binary_raw(raw_path)

    t = get_vector(points, variables, "time")
    vbus = get_vector(points, variables, "v(bus)")
    sw = np.column_stack([get_vector(points, variables, f"v(sw{x})") for x in "abc"])
    phases = np.column_stack([get_vector(points, variables, f"i(vsense{x})") for x in "abc"])
    ilink = get_vector(points, variables, "i(llink)")
    ilhf_a = get_vector(points, variables, "i(lhfa)")

    # The raw does not store nsA/B/C; low-side VDS is approximated by abs(SW),
    # with ns≈0. At the 150 A trip boundary, the omitted shunt drop is 75 mV.
    vds_env = np.maximum(
        np.max(np.abs(vbus[:, None] - sw), axis=1),
        np.max(np.abs(sw), axis=1),
    )
    log = parse_log(folder / "ngspice.log")
    netlist_text = (folder / "model.cir").read_text(encoding="ascii", errors="replace")
    start_match = re.search(r"^\.param TSTART=([^\s]+) TEND=([^\s]+)", netlist_text, re.M)
    if not start_match:
        raise ValueError(f"TSTART/TEND missing from archived netlist: {folder / 'model.cir'}")
    active_start = float(start_match.group(1))
    active_end = float(start_match.group(2))
    time_sample = np.arange(active_start, active_end + SAMPLE_PERIOD_S / 2, SAMPLE_PERIOD_S)
    sample = {
        "vbus_v": np.interp(time_sample, t, vbus),
        "vds_max_envelope_approx_v": np.interp(time_sample, t, vds_env),
        "ia_a": np.interp(time_sample, t, phases[:, 0]),
        "ib_a": np.interp(time_sample, t, phases[:, 1]),
        "ic_a": np.interp(time_sample, t, phases[:, 2]),
        "ilink_a": np.interp(time_sample, t, ilink),
        "ilhf_a_a": np.interp(time_sample, t, ilhf_a),
    }
    meta = {
        "case_id": case_id,
        "name": name,
        "title": spec["title"],
        "color": spec["color"],
        "time_s": t,
        "full_raw": {
            "vbus_v": vbus,
            "vds_max_envelope_approx_v": vds_env,
            "ia_a": phases[:, 0],
            "ib_a": phases[:, 1],
            "ic_a": phases[:, 2],
        },
        "time_sample_s": time_sample,
        "sample": sample,
        "log": log,
        "result": result,
        "raw_points": len(t),
        "raw_file": f"结果/{name}/{name}_summary.raw",
        "raw_bytes": raw_path.stat().st_size,
    }
    return meta, sample


def write_csv(cases: list[dict]) -> Path:
    out = RESULTS / "switch_short_waveform_samples.csv"
    fields = [
        "case_id", "t_ms", "vbus_v", "vds_max_envelope_approx_v",
        "ia_a", "ib_a", "ic_a", "i_link_bulk_a", "i_hf_cap_leg_a",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for idx, t in enumerate(case["time_sample_s"]):
                writer.writerow({
                    "case_id": case["case_id"],
                    "t_ms": f"{t * 1e3:.6f}",
                    "vbus_v": f"{case['sample']['vbus_v'][idx]:.6f}",
                    "vds_max_envelope_approx_v": f"{case['sample']['vds_max_envelope_approx_v'][idx]:.6f}",
                    "ia_a": f"{case['sample']['ia_a'][idx]:.6f}",
                    "ib_a": f"{case['sample']['ib_a'][idx]:.6f}",
                    "ic_a": f"{case['sample']['ic_a'][idx]:.6f}",
                    "i_link_bulk_a": f"{case['sample']['ilink_a'][idx]:.6f}",
                    "i_hf_cap_leg_a": f"{case['sample']['ilhf_a_a'][idx]:.6f}",
                })
    return out


def exact_phase_peak(case: dict) -> tuple[float, float]:
    candidates = []
    log = case["log"]
    for phase in "abc":
        for suffix in ("max", "min"):
            value, t = log[f"i{phase}_{suffix}"]
            candidates.append((abs(value), value, t))
    _, value, t = max(candidates)
    return value, t


def plot_case(case: dict) -> Path:
    name = case["name"]
    log = case["log"]
    result = case["result"]
    ts = case["time_sample_s"] * 1e3
    s = case["sample"]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.13, hspace=0.25)

    axes[0].plot(ts, s["vbus_v"], color="#1565c0", lw=1.1, label="VBUS (raw)")
    axes[0].plot(ts, s["vds_max_envelope_approx_v"], color="#d97706", lw=1.0,
                 label="max VDS envelope (derived from raw switch nodes)")
    vbus_peak, vbus_peak_t = log["vbus_max"]
    vds_peak, vds_peak_t = log["vds_any_peak"]
    axes[0].scatter([vbus_peak_t * 1e3], [vbus_peak], marker="v", color="#0d47a1", zorder=4,
                    label=f"ngspice full-trace VBUS max={vbus_peak:.2f} V")
    axes[0].scatter([vds_peak_t * 1e3], [vds_peak], marker="^", color="#b45309", zorder=4,
                    label=f"ngspice full-trace VDS max={vds_peak:.2f} V")
    axes[0].axhline(75, color="#a52222", ls="--", lw=0.9, label="75 V recommended bus ceiling")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Bus voltage and maximum device-voltage envelope")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].grid(alpha=0.25)

    colors = ("#1565c0", "#d97706", "#2e8b57")
    for phase, color in zip("abc", colors):
        axes[1].plot(ts, s[f"i{phase}_a"], color=color, lw=0.9, label=f"phase {phase.upper()}")
    axes[1].axhline(120, color="#a52222", ls="--", lw=1, label="+120 A screening line")
    axes[1].axhline(-120, color="#a52222", ls="--", lw=1, label="−120 A screening line")
    phase_peak, phase_t = exact_phase_peak(case)
    axes[1].scatter([phase_t * 1e3], [phase_peak], color="black", marker="o", s=24, zorder=5,
                    label=f"ngspice full-trace peak={phase_peak:+.2f} A")
    axes[1].set_ylabel("Phase current (A)")
    axes[1].set_title("Three-phase currents (raw vectors)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(ts, s["ilink_a"], color="#6a3d9a", lw=1.0, label="I(Llink): bulk DC-link branch")
    axes[2].plot(ts, s["ilhf_a_a"], color="#c2410c", lw=0.9, label="I(LhfA): phase-A HF capacitor branch")
    source_peak = result["isource_max"]
    bus_peak = result["ibus_max"]
    bus_min = result["ibus_min"]
    axes[2].set_ylabel("Local capacitor current (A)")
    axes[2].set_xlabel("Simulation time (ms)")
    axes[2].set_title(
        f"Capacitor ripple; separate ngspice scalar measures: source Imax={source_peak:.2f} A; "
        f"bridge iBUS={bus_min:.2f}…{bus_peak:.2f} A"
    )
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.25)

    axes[2].set_xlim(ts[0], ts[-1])
    fig.suptitle(f"{case['title']} — saved ngspice-47 transient", fontsize=12)
    fig.text(
        0.01, 0.015,
        "Saved ngspice transient; max step=20 ns. Curves/CSV use 1 µs samples; full-resolution .meas defines extrema.\n"
        "Raw omits I(Vsense), I(Vbus_sense), and nsA/B/C. Source and bridge iBUS are shown as scalar measures.\n"
        "VDS uses low-side ns≈0 (≤75 mV shunt drop). I(LhfA) is the phase-A bypass branch only.",
        fontsize=7.3, ha="left", va="bottom",
    )
    out = RESULTS / name / f"{case['case_id']}_waveforms.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    cases = []
    for case_id, spec in CASES.items():
        meta, _ = load_case(case_id, spec)
        cases.append(meta)
    csv_path = write_csv(cases)
    outputs = [plot_case(case) for case in cases]
    for case, png in zip(cases, outputs):
        print(json.dumps({
            "case": case["case_id"],
            "png": str(png),
            "raw_points": case["raw_points"],
            "csv_rows_for_case": len(case["time_sample_s"]),
            "source_i_max_scalar_a": case["result"]["isource_max"],
            "bridge_ibus_max_scalar_a": case["result"]["ibus_max"],
            "bridge_ibus_min_scalar_a": case["result"]["ibus_min"],
        }, ensure_ascii=False))
    print(json.dumps({
        "csv": str(csv_path),
        "rows": sum(len(case["time_sample_s"]) for case in cases),
        "sample_period_us": SAMPLE_PERIOD_S * 1e6,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
