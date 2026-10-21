#!/usr/bin/env python3
"""Build a compact, auditable comparison from the saved ngspice switch runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "结果"


def read_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    legacy_path = RESULTS / "switch_profile_results.json"
    legacy = read_result("switch_profile_results.json") if legacy_path.exists() else None
    regen = read_result("switch_filtered_nodump_results.json")
    regen_tvs = read_result("switch_filtered_tvs_results.json")
    drive = read_result("switch_positive_filtered_results.json")

    def case(label: str, filename: str, result: dict, status: str) -> dict:
        outer = result.get("case", {})
        params = outer.get("parameters", {})
        m = outer.get("measures", {})
        model_path = RESULTS / params.get("name", "") / "model.cir"
        duration = params.get("duration_s")
        if duration is None:
            duration = params.get("load_duration_s")
        return {
            "label": label,
            "result_file": f"结果/{filename}",
            "netlist_file": f"结果/{params.get('name', '')}/model.cir",
            "netlist_sha256": outer.get("netlist_sha256") or (sha256(model_path) if model_path.exists() else None),
            "status": status,
            "parameters": params,
            "active_duration_s": duration,
            "metrics": m,
        }

    cases = []
    if legacy is not None:
        cases.append(case("legacy_direct_bus_feedback", "switch_profile_results.json", legacy,
                          "INVALID_AS_HARDWARE_PREDICTION: instantaneous PWM duty division by switching bus voltage; retained only as pre-fix comparison"))
    cases.extend([
        case("regen_filtered_no_tvs", "switch_filtered_nodump_results.json", regen,
             "OPEN_LOOP_SCREEN: 100us filtered bus feedforward; not current-loop validation"),
        case("regen_filtered_smcj40a_proxy", "switch_filtered_tvs_results.json", regen_tvs,
             "OPEN_LOOP_SCREEN: fixed linear TVS proxy carried zero current below its 49.1V intercept; not a real-device or temperature clamp validation"),
        case("drive_filtered_no_tvs", "switch_positive_filtered_results.json", drive,
             "OPEN_LOOP_SCREEN: positive electrical input profile; not a 1s power qualification"),
    ])
    summary = {
        "title": "Three-phase switch-level short-window current-case summary",
        "legacy_case_included": legacy is not None,
        "legacy_case_note": "The invalid pre-fix direct-feedback case is included only when its saved result exists; current cases are summarized without fabricating a legacy input.",
        "ngspice": "47",
        "scope": "behavioral switch-level screening only; no manufacturer MOS macro, no closed-loop FOC, no 1s motor-power proof",
        "interpretation": {
            "pac": "measured three-phase motor-terminal power; requested setpoint is not substituted for measured power",
            "pbus": "measured inverter DC-port power, separate from battery/source current",
            "isource": "measured through the Thevenin battery source; local DC-link capacitors carry switching ripple",
            "ibus": "instantaneous inverter DC-port current, including local capacitor pulse current; not an external-source current limit measurement",
            "regen": "fixed-speed back-EMF source represents a mechanical energy reservoir; reverse battery acceptance is blocked in these switch cases",
            "comparison": "The optional invalid legacy case and current filtered cases may have different durations; none is a motor qualification.",
        },
        "cases": cases,
    }
    (RESULTS / "switch_short_cases_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Plot all three current cases. If the invalid pre-fix result is available,
    # include it only as an optional historical comparison and omit one regen
    # member from the compact plot; all current cases remain in the JSON.
    if legacy is not None:
        plot_cases = [cases[0], cases[2], cases[3]]
        labels = ["Legacy direct\nfeedback (invalid)", "Filtered regen\n+ SMCJ proxy", "Filtered drive\n(no TVS)"]
        colors = ["#9b4f4f", "#3979a8", "#40916c"]
    else:
        plot_cases = cases
        labels = ["Filtered regen\n(no TVS)", "Filtered regen\n+ SMCJ proxy", "Filtered drive\n(no TVS)"]
        colors = ["#3979a8", "#5981af", "#40916c"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)

    x = np.arange(len(plot_cases))
    pac = [c["metrics"]["pac_avg"] for c in plot_cases]
    pbus = [c["metrics"]["pbus_avg"] for c in plot_cases]
    width = 0.35
    axes[0].bar(x - width / 2, pac, width, label="$P_{AC}$ motor terminals", color=colors)
    axes[0].bar(x + width / 2, pbus, width, label="$P_{BUS}$ inverter port", color=colors, alpha=0.55, hatch="//")
    axes[0].axhline(1800, color="black", linewidth=0.8, linestyle="--")
    axes[0].axhline(-1800, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Average power during reported measurement window (W)")
    axes[0].set_title("Measured power ports")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(axis="y", alpha=0.25)

    phase_peak = [max(abs(c["metrics"][k]) for k in ("ia_max", "ia_min", "ib_max", "ib_min", "ic_max", "ic_min")) for c in plot_cases]
    axes[1].bar(x, phase_peak, color=colors)
    axes[1].axhline(120, color="#a52222", linewidth=1.2, linestyle="--", label="120 A screening boundary")
    axes[1].set_ylabel("Maximum absolute phase current (A)")
    axes[1].set_title("Phase current stress")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    vbus = [c["metrics"]["vbus_max"] for c in plot_cases]
    vds = [c["metrics"]["vds_any_peak"] for c in plot_cases]
    axes[2].bar(x - width / 2, vbus, width, label="$V_{BUS}$ maximum", color=colors)
    axes[2].bar(x + width / 2, vds, width, label="$V_{DS}$ maximum", color=colors, alpha=0.55, hatch="//")
    axes[2].axhline(75, color="#be7b00", linewidth=1.0, linestyle="--", label="75 V recommended bus ceiling")
    axes[2].axhline(100, color="#a52222", linewidth=1.0, linestyle=":", label="100 V MOS rating")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].set_title("Voltage stress")
    axes[2].legend(fontsize=7, loc="upper left")
    axes[2].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
    fig.suptitle("ngspice short-window screening — durations differ; not hardware qualification", fontsize=11)
    fig.savefig(RESULTS / "switch_short_cases_comparison.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "summary": str(RESULTS / "switch_short_cases_summary.json"),
        "plot": str(RESULTS / "switch_short_cases_comparison.png"),
        "cases": len(cases),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
