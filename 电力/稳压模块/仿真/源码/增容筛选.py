#!/usr/bin/env python3
"""Current 8S1P SC energy and terminal-voltage sensitivities; not hardware qualification."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

MODULE = Path(__file__).resolve().parents[2]
CONFIG = MODULE / "原理图准备" / "AFE启动配置.json"
OUT = MODULE / "仿真" / "结果" / "增容筛选.json"

SOURCES = {
    "SC0350-270-RSS": "https://www.licaptech.com/pdfs/datasheets/ultracapacitors/60862a8f5979e5f969df83ba_LICAP_SC0350_270_RSS_Datasheet_061120.pdf",
}


def derive_supercapacitor_model(config: dict[str, Any]) -> dict[str, float | int | str]:
    """Derive the model only for the currently selected physical 8S1P stack."""
    series = int(config["series_groups"])
    parallel = int(config["parallel_cells_per_group"])
    quantity = int(config["cell_count"])
    if series != 8 or parallel != 1 or quantity != 8 or quantity != series * parallel:
        raise ValueError(
            "current supercapacitor model is fixed at 8S1P / 8 cells; "
            f"received series={series}, parallel={parallel}, quantity={quantity}"
        )
    if str(config.get("cell_mpn", "")).strip() != "SC0350-270-RSS":
        raise ValueError("current model is restricted to the user's fixed SC0350-270-RSS selection")

    part = config["supercapacitor_model"]
    c_cell_nom = float(part["nominal_capacitance_per_cell_F"])
    c_eol_fraction = float(part["eol_capacitance_fraction"])
    r_cell_nom = float(part["nominal_esr_per_cell_ohm"])
    r_eol_multiplier = float(part["eol_esr_multiplier"])
    extra_r_multiplier = float(part["unqualified_extra_esr_multiplier"])
    interconnect_r = float(part["unqualified_interconnect_series_resistance_ohm"])
    if not all(math.isfinite(x) for x in (c_cell_nom, c_eol_fraction, r_cell_nom,
                                          r_eol_multiplier, extra_r_multiplier, interconnect_r)):
        raise ValueError("capacitance and ESR assumptions must be finite")
    if c_cell_nom != 350.0:
        raise ValueError("SC0350-270-RSS nameplate capacitance must remain 350F per cell")
    if c_cell_nom <= 0 or not 0 < c_eol_fraction <= 1 or r_cell_nom < 0 or r_eol_multiplier < 1:
        raise ValueError("invalid capacitance or ESR assumptions in AFE startup configuration")
    if extra_r_multiplier < 1 or interconnect_r < 0:
        raise ValueError("invalid unqualified ESR sensitivity in AFE startup configuration")

    c_group_eol = c_cell_nom * parallel * c_eol_fraction
    r_group_eol = r_cell_nom * r_eol_multiplier / parallel
    return {
        "manufacturer_part": str(config["cell_mpn"]),
        "series_groups": series,
        "parallel_cells_per_group": parallel,
        "cell_count": quantity,
        "nominal_capacitance_per_cell_F": c_cell_nom,
        "eol_capacitance_per_group_F": c_group_eol,
        "eol_stack_equivalent_capacitance_F": c_group_eol / series,
        "nominal_esr_per_cell_ohm": r_cell_nom,
        "eol_esr_per_group_ohm": r_group_eol,
        "eol_stack_series_resistance_ohm": r_group_eol * series,
        "extra_loss_stack_series_resistance_ohm": r_group_eol * series * extra_r_multiplier + interconnect_r,
        "eol_capacitance_fraction": c_eol_fraction,
        "eol_esr_multiplier": r_eol_multiplier,
        "unqualified_extra_esr_multiplier": extra_r_multiplier,
        "unqualified_interconnect_series_resistance_ohm": interconnect_r,
    }


def simulate(
    cap_f: float,
    esr_ohm: float,
    initial_v: float,
    power_w: float,
    eta: float,
    direction: str,
    dt: float,
    duration_s: float = 1.0,
) -> dict[str, float | bool]:
    """Integrate a constant-power capacitor/ESR model for an explicit duration.

    The optional one-second default preserves old callers. Current model entry
    points pass duration_s explicitly. This is an energy-flow sensitivity only;
    it does not simulate a converter, PWM, switching, or its protection logic.
    """
    values = (cap_f, esr_ohm, initial_v, power_w, eta, dt, duration_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("simulation inputs must be finite")
    if cap_f <= 0 or esr_ohm < 0 or initial_v < 0 or power_w <= 0 or not 0 < eta <= 1:
        raise ValueError("invalid capacitance, ESR, voltage, power, or efficiency")
    if direction not in ("support", "absorb"):
        raise ValueError("direction must be 'support' or 'absorb'")
    if dt <= 0 or duration_s <= 0:
        raise ValueError("time step and duration must be positive")

    v = initial_v
    sign = -1 if direction == "support" else 1
    port_power = power_w / eta if sign == -1 else power_w * eta
    low, high, peak_i, joules_r = math.inf, -math.inf, 0.0, 0.0
    steps = math.ceil(duration_s / dt)
    for step in range(steps + 1):
        time_s = min(step * dt, duration_s)
        discriminant = v * v + sign * 4 * esr_ohm * port_power
        if discriminant <= 0:
            return {
                "quadratic_solution_exists": False,
                "trajectory_complete": False,
                "failure_time_s": time_s,
                "internal_voltage_at_failure_V": v,
                "integrated_duration_s": time_s,
            }
        root = math.sqrt(discriminant)
        denominator = v + root
        if denominator <= 0:
            return {
                "quadratic_solution_exists": False,
                "trajectory_complete": False,
                "failure_time_s": time_s,
                "internal_voltage_at_failure_V": v,
                "integrated_duration_s": time_s,
            }
        current = 2 * port_power / denominator
        terminal = v + sign * current * esr_ohm
        low, high = min(low, terminal), max(high, terminal)
        peak_i = max(peak_i, current)

        if step < steps:
            step_s = min(dt, duration_s - time_s)
            if step_s <= 0:
                break
            joules_r += current * current * esr_ohm * step_s
            v += sign * current / cap_f * step_s

    stored_change = 0.5 * cap_f * (v * v - initial_v * initial_v)
    expected_stored_change = sign * port_power * duration_s - joules_r
    residual = stored_change - expected_stored_change
    return {
        "quadratic_solution_exists": True,
        "trajectory_complete": True,
        "initial_internal_V": initial_v,
        "end_internal_V": v,
        "minimum_terminal_V": low,
        "maximum_terminal_V": high,
        "maximum_average_port_current_A": peak_i,
        "resistive_loss_J": joules_r,
        "stored_energy_change_J": stored_change,
        "energy_balance_residual_J": residual,
        "requested_duration_s": duration_s,
        "integrated_duration_s": duration_s,
        "time_step_s": dt,
    }


def verify_simulation_api() -> dict[str, bool]:
    """Check duration validation and compatibility for the former call shape."""
    rejected_nonpositive_duration = True
    for duration in (0.0, -0.1):
        try:
            simulate(35.0, 0.05, 17.45, 100.0, 0.95, "support", 0.5, duration)
        except ValueError:
            continue
        rejected_nonpositive_duration = False
    legacy_call = simulate(35.0, 0.05, 17.45, 100.0, 0.95, "support", 0.5)
    # Independent zero-ESR closed form, including a fractional last time step.
    # Use the forward-Euler global error bound, rather than an arbitrary
    # tolerance tighter than the first-order integration method can provide.
    duration, coefficient = 0.037123, 800.0 / 0.95 / 35.0
    exact_v = math.sqrt(18.0 ** 2 - 2 * coefficient * duration)
    lipschitz, curvature = coefficient / exact_v ** 2, coefficient ** 2 / exact_v ** 3
    errors, within_bound = [], True
    for step in (0.0001, 0.00005, 0.000025):
        row = simulate(35.0, 0.0, 18.0, 800.0, 0.95, "support", step, duration)
        error = abs(row["end_internal_V"] - exact_v)
        bound = step * curvature * math.expm1(lipschitz * duration) / (2 * lipschitz)
        within_bound &= error <= bound + 1e-12 and row["integrated_duration_s"] == duration
        errors.append(error)
    return {
        "nonpositive_duration_rejected": rejected_nonpositive_duration,
        "legacy_call_defaults_to_one_second": legacy_call["trajectory_complete"] and legacy_call["integrated_duration_s"] == 1.0,
        "fractional_duration_closed_form_within_Euler_error_bound": within_bound,
        "closed_form_first_order_step_convergence": all(1.9 < a / b < 2.1 for a, b in zip(errors, errors[1:])),
    }


def verify_cancelled_configuration_rejected(config: dict[str, Any]) -> bool:
    """Keep the cancelled 8S2P / 16-cell option out of the active model."""
    expanded = dict(config)
    expanded["parallel_cells_per_group"] = 2
    expanded["cell_count"] = 16
    try:
        derive_supercapacitor_model(expanded)
    except ValueError:
        return True
    return False


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    model = derive_supercapacitor_model(config)
    software = config["software_contract"]
    sensitivity = config["energy_sensitivity"]
    series = model["series_groups"]
    group_cap = model["eol_capacitance_per_group_F"]
    ripple_half_peak = float(sensitivity["ripple_bus_voltage_V"]) / (
        8 * float(sensitivity["inductor_min_H"]) * float(sensitivity["switching_frequency_Hz"])
    )
    measurement_error = float(software["cell_measurement_error_allocation_V"])
    group_deviation = float(software["max_true_group_deviation_from_mean_allocation_V"])
    soft_uv = float(software["normal_cell_soft_uv_V"])
    soft_ov = float(software["normal_cell_soft_ov_V"])
    current_limit = float(software["provisional_peak_current_screen_A"])
    idle = float(software["idle_internal_target_V"])
    voltage_low_stack = series * (soft_uv + measurement_error + group_deviation)
    voltage_high_stack = series * (soft_ov - measurement_error - group_deviation)
    if voltage_low_stack >= voltage_high_stack:
        raise ValueError("measurement/imbalance allocations leave no valid cell-voltage window")

    scenarios = [
        {
            "name": "eol_parameter_sensitivity",
            "efficiency": float(sensitivity["nominal_efficiency_assumption"]),
            "resistance": float(model["eol_stack_series_resistance_ohm"]),
            "start_voltage_error_V": 0.0,
            "additional_terminal_reserve_V": 0.0,
            "qualification": "PARAMETER_SENSITIVITY_ONLY",
        },
        {
            "name": "unqualified_extra_loss_sensitivity",
            "efficiency": float(sensitivity["unqualified_stress_efficiency_assumption"]),
            "resistance": float(model["extra_loss_stack_series_resistance_ohm"]),
            "start_voltage_error_V": float(sensitivity["unqualified_start_voltage_error_V"]),
            "additional_terminal_reserve_V": float(sensitivity["unqualified_additional_terminal_reserve_V"]),
            "qualification": "UNQUALIFIED_SENSITIVITY_ONLY",
        },
    ]
    powers = [float(value) for value in sensitivity["illustrative_bus_power_W"]]
    durations = [float(value) for value in sensitivity["illustrative_durations_s"]]
    if not powers or not durations or any(value <= 0 for value in powers + durations):
        raise ValueError("illustrative energy sensitivity powers and durations must be positive")

    rows = []
    for scenario in scenarios:
        for power_w in powers:
            for duration_s in durations:
                trajectories = {}
                for direction in ("support", "absorb"):
                    start_error = scenario["start_voltage_error_V"]
                    initial_v = idle + (-start_error if direction == "support" else start_error)
                    coarse = simulate(group_cap / series, scenario["resistance"], initial_v,
                                      power_w, scenario["efficiency"], direction, 0.0001, duration_s)
                    fine = simulate(group_cap / series, scenario["resistance"], initial_v,
                                    power_w, scenario["efficiency"], direction, 0.00005, duration_s)
                    if coarse["trajectory_complete"] and fine["trajectory_complete"]:
                        difference = max(
                            abs(coarse[key] - fine[key])
                            for key in ("minimum_terminal_V", "maximum_terminal_V")
                        )
                        coarse["terminal_voltage_timestep_difference_V"] = difference
                        coarse["timestep_convergence_pass"] = difference < 0.001
                        coarse["refined_energy_balance_residual_J"] = fine["energy_balance_residual_J"]
                        peak = float(coarse["maximum_average_port_current_A"]) + ripple_half_peak
                        coarse["ideal_ripple_half_peak_A"] = ripple_half_peak
                        coarse["estimated_inductor_peak_A"] = peak
                        lower_limit = voltage_low_stack + float(scenario["additional_terminal_reserve_V"])
                        upper_limit = voltage_high_stack - float(scenario["additional_terminal_reserve_V"])
                        coarse["screen_voltage_window_V"] = [lower_limit, upper_limit]
                        coarse["screen_current_peak_limit_A"] = current_limit
                        coarse["conditional_screen_pass"] = bool(
                            coarse["timestep_convergence_pass"]
                            and coarse["minimum_terminal_V"] >= lower_limit
                            and coarse["maximum_terminal_V"] <= upper_limit
                            and peak <= current_limit
                        )
                    else:
                        coarse["terminal_voltage_timestep_difference_V"] = None
                        coarse["timestep_convergence_pass"] = False
                        coarse["ideal_ripple_half_peak_A"] = ripple_half_peak
                        coarse["estimated_inductor_peak_A"] = None
                        coarse["conditional_screen_pass"] = False
                        coarse["screen_failure_reason"] = "constant-power branch has no real solution before requested end"
                    trajectories[direction] = coarse

                rows.append({
                    "configuration": "current_8S1P_350F",
                    "manufacturer_part": model["manufacturer_part"],
                    "cell_count": model["cell_count"],
                    "series_groups": series,
                    "parallel_cells_per_group": model["parallel_cells_per_group"],
                    "eol_stack_equivalent_capacitance_F": model["eol_stack_equivalent_capacitance_F"],
                    "model_series_resistance_ohm": scenario["resistance"],
                    "efficiency_assumption": scenario["efficiency"],
                    "start_voltage_target_V": idle,
                    "start_voltage_error_assumption_V": scenario["start_voltage_error_V"],
                    "screen_voltage_window_V": [
                        voltage_low_stack + float(scenario["additional_terminal_reserve_V"]),
                        voltage_high_stack - float(scenario["additional_terminal_reserve_V"]),
                    ],
                    "bus_power_sensitivity_W": power_w,
                    "duration_s": duration_s,
                    "duration_role": "one_second_reference_only" if math.isclose(duration_s, 1.0) else "illustrative_sensitivity_only",
                    "scenario": scenario["name"],
                    "qualification": scenario["qualification"],
                    "results": trajectories,
                    "both_directions_conditional_screen_pass": all(
                        item["conditional_screen_pass"] for item in trajectories.values()
                    ),
                })

    api_validation = verify_simulation_api()
    api_validation["cancelled_16_cell_configuration_rejected"] = verify_cancelled_configuration_rejected(config)
    completed = [
        x for row in rows for x in row["results"].values()
        if x["trajectory_complete"]
    ]
    failed = [
        {
            "scenario": row["scenario"],
            "power_W": row["bus_power_sensitivity_W"],
            "duration_s": row["duration_s"],
            "direction": direction,
            "failure_time_s": trajectory.get("failure_time_s"),
        }
        for row in rows for direction, trajectory in row["results"].items()
        if not trajectory["trajectory_complete"]
    ]
    numerical_ok = (
        bool(completed)
        and all(abs(float(x["energy_balance_residual_J"])) < 0.1 for x in completed)
        and all(abs(float(x["refined_energy_balance_residual_J"])) < 0.1 for x in completed)
        and all(x["timestep_convergence_pass"] for x in completed)
        and all(api_validation.values())
    )
    result = {
        "date": "2026-10-21",
        "status": "CURRENT_8S1P_SENSITIVITY_MODEL_ONLY_NO_HARDWARE_RELEASE",
        "configuration_path": str(CONFIG.relative_to(MODULE)).replace("\\", "/"),
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "current_requirement": "LICAP SC0350-270-RSS fixed at 8 cells / 8S1P for startup transient assist; actual power and duration are unspecified; no 1s requirement.",
        "supercapacitor_model": model,
        "energy_sensitivity": {
            "requirement_status": sensitivity["requirement_status"],
            "illustrative_bus_power_W": powers,
            "illustrative_durations_s": durations,
            "cell_soft_uv_V": soft_uv,
            "cell_soft_ov_V": soft_ov,
            "measurement_error_allocation_V": measurement_error,
            "true_group_deviation_allocation_V": group_deviation,
            "current_peak_screen_A": current_limit,
            "ideal_ripple_half_peak_A": ripple_half_peak,
            "rows": rows,
        },
        "conditions": [
            "Only the current eight-cell 8S1P selection is evaluated; the cancelled 16-cell option is rejected by the configuration validator and is not an active candidate.",
            "Power and duration pairs are illustrative sensitivities only. The 1s duration is a reference, not a current user requirement or release gate.",
            "EOL C fraction and ESR multiplier are endurance sensitivities, not a qualified temperature/age guarantee.",
            "The constant-power quadratic model includes idealized stack capacitance and series resistance only; it does not simulate a switching converter, control loop, PWM, bootstrap, minimum pulse, reverse-current semiconductor paths, or power-off backfeed.",
            "The 80A peak screen adds an ideal Vbus/(8Lf) half-ripple estimate. Hot biased inductance, actual PWM, current trip tolerance, response, SOA and thermal limits remain unqualified.",
            "The cell voltage window includes configured soft limits plus measurement and group-deviation allocations; those allocations require physical qualification.",
            "Individual cell imbalance, balancing/leakage, converter startup, fault injection, mechanical fit and energy-path release are not established by this model.",
        ],
        "simulation_api_validation": api_validation,
        "numerical_verification": {
            "completed_trajectories": len(completed),
            "infeasible_sensitivity_trajectories": failed,
            "energy_balance_residual_max_abs_J": max(abs(float(x["energy_balance_residual_J"])) for x in completed),
            "all_completed_trajectories_energy_balance_and_step_convergence_pass": numerical_ok,
            "meaning": "Numerical checks validate integration only; conditional screen results are sensitivities, not hardware release or a startup requirement.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        "8S1P-SC-SENSITIVITY: "
        f"cells={model['cell_count']}; scenarios={len(rows)}; "
        f"completed_trajectories={len(completed)}; infeasible={len(failed)}; "
        f"energy_balance_and_convergence={'PASS' if numerical_ok else 'FAIL'}; "
        "power_duration_pairs=illustrative_only; hardware_release=NOT_ESTABLISHED"
    )
    return 0 if numerical_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
