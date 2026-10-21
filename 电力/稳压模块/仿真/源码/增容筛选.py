#!/usr/bin/env python3
"""双向恒功率、端电压与ESR筛选。仅为集中参数候选模型，不写CAD或现行BOM。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2]
OUT = MODULE / "仿真/结果/增容筛选.json"

SOURCES = {
    "SC0350-270-RSS": "https://www.licaptech.com/pdfs/datasheets/ultracapacitors/60862a8f5979e5f969df83ba_LICAP_SC0350_270_RSS_Datasheet_061120.pdf",
}
# EOL capacitance 80%, ESR 200% are sensitivity assumptions using the
# manufacturers' endurance criteria, not a qualified 0..60C guarantee.
CANDIDATES = [
    ("current_8S1P_350F", "SC0350-270-RSS", 350., .0028, 1, 17.6, 19.6, math.sqrt((19.6**2 + 17.6**2)/2)),
    ("8S2P_350F_same_window", "SC0350-270-RSS", 350., .0028, 2, 17.6, 19.6, 18.7),
    ("8S2P_350F_balanced_800W_candidate", "SC0350-270-RSS", 350., .0028, 2, 13.6, 19.6, 17.2),
    ("8S2P_350F_support_1kW_prepared", "SC0350-270-RSS", 350., .0028, 2, 13.6, 19.6, 19.2),
    ("8S2P_350F_absorb_1kW_prepared", "SC0350-270-RSS", 350., .0028, 2, 13.6, 19.6, 17.1),
]


def simulate(cap_f, esr_ohm, initial_v, power_w, eta, direction, dt):
    """Solve the low-current quadratic branch and integrate ideal internal V.

    Discharge: Pbus/eta = I*(Vinternal-I*R).
    Charge: Pbus*eta = I*(Vinternal+I*R).
    Terminal power excludes no additional series losses: those are included
    in R. Capacitor ESR and interconnect loss are separately energy-accounted.
    """
    v = initial_v
    sign = -1 if direction == "support" else 1
    port_power = power_w / eta if sign == -1 else power_w * eta
    low, high, peak_i, joules_r = math.inf, -math.inf, 0., 0.
    steps = round(1. / dt)
    for step in range(steps + 1):
        discriminant = v*v + sign*4*esr_ohm*port_power
        if discriminant <= 0:
            return {"quadratic_solution_exists": False, "failure_time_s": step*dt}
        i = 2*port_power/(v + math.sqrt(discriminant))
        terminal = v + sign*i*esr_ohm
        low, high, peak_i = min(low, terminal), max(high, terminal), max(peak_i, i)
        if step < steps:
            joules_r += i*i*esr_ohm*dt
            v += sign*i/cap_f*dt
    stored_change = .5*cap_f*(v*v - initial_v*initial_v)
    residual = stored_change - (port_power - joules_r if sign == 1 else -port_power - joules_r)
    return {"quadratic_solution_exists": True, "initial_internal_V": initial_v,
            "end_internal_V": v, "minimum_terminal_V": low, "maximum_terminal_V": high,
            "maximum_average_port_current_A": peak_i, "resistive_loss_J": joules_r,
            "stored_energy_change_J": stored_change, "energy_balance_residual_J": residual}


def main():
    rows = []
    for key, mpn, nominal_c, initial_r, parallel, low, high, bias in CANDIDATES:
        c = nominal_c*.8*parallel/8
        for stress in (False, True):
            eta = .90 if stress else .95
            # Sensitivity only: cold multiplier and 5mOhm interconnect have
            # not been measured, so a pass is never a hardware qualification.
            r = 8*initial_r*2/parallel*(1.3 if stress else 1.) + (.005 if stress else 0.)
            bias_error = .1 if stress else 0.
            reserve = .1 if stress else 0.
            for p in (800., 1000.):
                results = {}
                for direction in ("support", "absorb"):
                    start = bias-bias_error if direction == "support" else bias+bias_error
                    a = simulate(c, r, start, p, eta, direction, .0001)
                    b = simulate(c, r, start, p, eta, direction, .00005)
                    if not a["quadratic_solution_exists"]:
                        a["conditional_screen_pass"] = False
                    else:
                        a["terminal_voltage_timestep_difference_V"] = max(
                            abs(a[f]-b[f]) for f in ("minimum_terminal_V", "maximum_terminal_V"))
                        # Ideal buck/boost ripple upper bound Vbus/(4Lf),
                        # using candidate Lmin=12uH and f=50kHz. These are
                        # not validated hot/bias inductance or PWM settings.
                        a["ideal_ripple_half_peak_A"] = 37.8/(8*12e-6*50e3)
                        a["estimated_inductor_peak_A"] = a["maximum_average_port_current_A"] + a["ideal_ripple_half_peak_A"]
                        a["conditional_screen_pass"] = (
                            a["minimum_terminal_V"] >= low+reserve and
                            a["maximum_terminal_V"] <= high-reserve and
                            a["estimated_inductor_peak_A"] <= 80.)
                    results[direction] = a
                rows.append({"candidate": key, "mpn": mpn, "cell_quantity": 8*parallel,
                             "parallel_count": parallel, "total_EOL_capacitance_F": c,
                             "total_model_series_resistance_ohm": r, "efficiency_assumption": eta,
                             "terminal_window_V": [low, high], "idle_internal_target_V": bias,
                             "unmeasured_bias_error_assumption_V": bias_error,
                             "additional_terminal_reserve_V": reserve, "bus_power_W": p,
                             "duration_s": 1., "scenario": "unqualified_extra_loss_sensitivity" if stress else "EOL_25C_parameter_screen",
                             "results": results,
                             "both_directions_pass": all(x["conditional_screen_pass"] for x in results.values())})
    completed = [x for row in rows for x in row["results"].values() if x["quadratic_solution_exists"]]
    infeasible = [dict(candidate=row["candidate"], scenario=row["scenario"], power_W=row["bus_power_W"], direction=direction,
                       failure_time_s=x["failure_time_s"])
                  for row in rows for direction, x in row["results"].items() if not x["quadratic_solution_exists"]]
    numerical_ok = bool(completed) and all(abs(x["energy_balance_residual_J"]) < .1 and
                       x["terminal_voltage_timestep_difference_V"] < .001 for x in completed)
    result = {"date": "2026-10-21", "status": "CANDIDATE_MODEL_ONLY_NO_HARDWARE_RELEASE",
              "sources": SOURCES, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "latest_requirement": "500元成本上限已取消；电容选型固定LICAP SC0350-270-RSS，最多16颗；800–1000W/1s作为设计目标，按状态核算可用额度；全电路最少选型、最小总成本与性价比优先。",
              "conditions": ["8 series groups are identical; no cell imbalance or individual fault is simulated.",
                             "EOL C=80% and ESR=200% are data-sheet endurance sensitivities; temperature corners remain unqualified.",
                             "Power is a total BUS allocation, not an independent allowance to each of the two FOC boards.",
                             "80A screens average SC current plus an ideal Vbus/(8Lf) half-ripple estimate at Lmin=12uH/f=50kHz. Hot biased inductance, actual PWM, trip tolerance, response and SOA remain unqualified.",
                             "An idle-voltage target is no-load internal voltage; ADC under current sees terminal voltage plus the shunt reference offset.",
                             "A wider voltage window requires a new per-cell UV/control/inductor qualification; it does not modify the current 8S1P design.",
                             "Measurements, end-to-end shutdown, firmware, balancing/leakage and mechanical/thermal design remain mandatory.",
                             "Only the user's fixed SC0350-270-RSS selection is evaluated. Other cell types are excluded.",
                             "Parallel groups reuse the current corrected footprint; mechanical fit, placement and fault-energy coordination need new checks."],
              "numerical_verification": {"completed_runs": len(completed),
                  "completed_runs_energy_balance_and_step_convergence_pass": numerical_ok,
                  "infeasible_runs": infeasible,
                  "all_runs_complete_one_second": not infeasible,
                  "meaning": "Numerical checks apply only to completed trajectories. Infeasible constant-power runs fail the candidate screen and are never counted as numerical passes."},
              "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf8")
    for row in rows:
        if row["scenario"] == "unqualified_extra_loss_sensitivity":
            print(row["candidate"], f"{row['bus_power_W']:.0f}W/1s",
                  "support=", row["results"]["support"]["conditional_screen_pass"],
                  "absorb=", row["results"]["absorb"]["conditional_screen_pass"],
                  "support_min_V=", row["results"]["support"].get("minimum_terminal_V"),
                  "absorb_max_V=", row["results"]["absorb"].get("maximum_terminal_V"))
    print("completed_runs_energy_balance_and_step_convergence=", numerical_ok,
          "completed_runs=", len(completed), "infeasible_runs=", len(infeasible),
          "all_runs_complete_one_second=", not infeasible)
    return 0 if numerical_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
