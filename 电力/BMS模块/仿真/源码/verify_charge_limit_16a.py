#!/usr/bin/env python3
"""Reproducible parameter screen for the 9S4P net-charge upper bound.

This is a calculation/sensitivity model, not a firmware, circuit, or vehicle
validation. Run from any directory with Python 3.10+; standard library only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "BMS模块" / "仿真" / "输入" / "16A净充电上限验证_input.json"
OUTPUT = ROOT / "BMS模块" / "仿真" / "结果" / "16A净充电上限验证_result.json"
CHECK_OUTPUT = ROOT / "BMS模块" / "检查" / "16A净充电上限验证.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_finite_number(value: Any) -> bool:
    """Reject bool explicitly: Python otherwise treats True/False as 1/0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def round_down(value: float, quantum: float) -> float:
    return math.floor((value + 1e-12) / quantum) * quantum


def current_error_bound_A(p: dict[str, Any]) -> float:
    bq = p["bq76952"]
    n_lsb = (
        bq["cc_inl_abs_max_lsb"]
        + bq["cc_offset_uncalibrated_abs_lsb"]
        + bq["cc_offset_drift_abs_lsb_per_C"]
        * abs(bq["bms_temperature_max_C_for_offset_drift"] - bq["bms_calibration_temperature_C"])
        + bq["cc_dnl_worst_case_engineering_assumption_lsb"]
        + bq["cc2_stability_margin_lsb_engineering_assumption"]
    )
    voltage_error_V = n_lsb * bq["cc_lsb_uV"] * 1e-6
    # The calculation uses the nominal programmed shunt conversion. The
    # measured-shunt and Kelvin path low corners are accounted for in K below.
    scale = min(x["relative_scale"] for x in bq["software_current_gain_scale_scenarios"])
    return voltage_error_V / p["shunt"]["nominal_ohm"] * scale


def gain_lower_bound(p: dict[str, Any], temp_C: float, path_deficit_uohm: float) -> float:
    bq, shunt = p["bq76952"], p["shunt"]
    gain_ratio = bq["cc_gain_lsb_per_v_min"] / bq["cc_gain_lsb_per_v_typ"]
    scale = min(x["relative_scale"] for x in bq["software_current_gain_scale_scenarios"])
    dtemp = abs(temp_C - shunt["calibration_temperature_C"])
    r_min = shunt["nominal_ohm"] * (
        1.0 - shunt["tolerance_fraction"]
        - shunt["temperature_coefficient_ppm_per_C"] * 1e-6 * dtemp
    )
    r_effective = r_min - path_deficit_uohm * 1e-6
    return gain_ratio * scale * r_effective / shunt["nominal_ohm"]


def cell_current_limit_A(p: dict[str, Any], cell_temp_upper_C: float) -> float:
    """Apply only the manufacturer's demonstrated max-rate test window."""
    pack = p["pack"]
    low, high = pack["cell_max_charge_test_temperature_C"]
    if low <= cell_temp_upper_C <= high:
        return pack["cell_max_charge_A"]
    return pack["cell_max_charge_outside_test_window_A"]


def compute_ccl(p: dict[str, Any], alpha: float, temp_C: float,
                path_deficit_uohm: float, overshoot_A: float,
                cell_temp_upper_C: float = 25.0,
                request_A: float | None = None) -> dict[str, Any]:
    pack = p["pack"]
    request = pack["user_net_charge_ceiling_A"] if request_A is None else max(0.0, request_A)
    cell_A = cell_current_limit_A(p, cell_temp_upper_C)
    cell_limited_pack_A = cell_A / alpha if alpha > 0 else math.inf
    k_min = gain_lower_bound(p, temp_C, path_deficit_uohm)
    error_A = current_error_bound_A(p)
    occ = occ_bounds(p)
    occ_peak_cap = max(
        0.0,
        occ["early_trip_current_A"] - occ["normal_operation_peak_planning_margin_A"],
    )
    applicable_pack_limit_A = min(
        pack["user_net_charge_ceiling_A"], cell_limited_pack_A, occ_peak_cap
    )
    raw = max(0.0, k_min * (applicable_pack_limit_A - overshoot_A) - error_A)
    quantum = p["normal_ccl_model"]["command_quantization_A"]
    command = min(round_down(request, quantum), round_down(raw, quantum))
    predicted_peak = (command + error_A) / k_min + overshoot_A if k_min > 0 else math.inf
    return {
        "requested_CCL_A": request,
        "alpha_max_branch_share": alpha,
        "cell_temperature_upper_bound_C": cell_temp_upper_C,
        "cell_max_charge_rate_allowed_A": cell_A,
        "cell_limited_pack_ceiling_A": cell_limited_pack_A,
        "applicable_pack_ceiling_A": applicable_pack_limit_A,
        "shunt_temperature_C": temp_C,
        "sense_path_underread_uohm": path_deficit_uohm,
        "bounded_source_overshoot_A": overshoot_A,
        "measurement_gain_lower_bound": k_min,
        "one_sided_measurement_underread_bound_A": error_A,
        "raw_CCL_command_bound_A": raw,
        "quantized_CCL_candidate_A": command,
        "model_predicted_pack_peak_A": predicted_peak,
        "occ_early_trip_current_A_design_candidate": occ["early_trip_current_A"],
        "occ_peak_planning_margin_A_engineering_assumption": occ["normal_operation_peak_planning_margin_A"],
        "occ_peak_cap_after_planning_margin_A": occ_peak_cap,
        "ccl_candidate_peak_margin_to_occ_early_corner_A": occ["early_trip_current_A"] - predicted_peak,
        "ccl_candidate_peak_within_occ_early_corner": predicted_peak <= occ["early_trip_current_A"] + 1e-9,
        "ccl_candidate_peak_meets_occ_planning_margin": predicted_peak <= occ_peak_cap + 1e-9,
        "predicted_max_branch_current_A": alpha * predicted_peak,
        "predicted_max_branch_within_cell_rate_limit": alpha * predicted_peak <= cell_A + 1e-9,
    }


def occ_bounds(p: dict[str, Any]) -> dict[str, Any]:
    o = p["occ_protection"]
    low_mV = o["nominal_threshold_mV"] + o["threshold_accuracy_small_setting_min_mV"]
    high_mV = o["nominal_threshold_mV"] + o["threshold_accuracy_small_setting_max_mV"]
    shunt = p["shunt"]
    dtemp = abs(o["occ_shunt_temperature_limit_C"] - shunt["calibration_temperature_C"])
    tcr = shunt["temperature_coefficient_ppm_per_C"] * 1e-6 * dtemp
    r_low = shunt["nominal_ohm"] * (1 - shunt["tolerance_fraction"] - tcr)
    r_high = shunt["nominal_ohm"] * (1 + shunt["tolerance_fraction"] + tcr)
    early_trip = low_mV * 1e-3 / r_high
    margin = o["normal_operation_peak_planning_margin_A"]
    return {
        "register_code_design_candidate": o["register_code_candidate"],
        "configuration_written_or_readback_verified": o["configuration_written_or_readback_verified"],
        "configuration_status": "design candidate only; actual register write/readback not verified",
        "threshold_early_corner_mV": low_mV,
        "threshold_late_corner_mV": high_mV,
        "shunt_temperature_limit_C_candidate": o["occ_shunt_temperature_limit_C"],
        "shunt_low_corner_ohm_including_TCR": r_low,
        "shunt_high_corner_ohm_including_TCR": r_high,
        "nominal_trip_current_A": o["nominal_threshold_mV"] * 1e-3
                                 / p["shunt"]["nominal_ohm"],
        "early_trip_current_A": early_trip,
        "late_trip_current_A": high_mV * 1e-3 / r_low,
        "programmed_delay_ms": 3.3 * (2 + o["delay_setting_candidate"]),
        "normal_operation_peak_planning_margin_A": margin,
        "normal_operation_peak_cap_after_margin_A": max(0.0, early_trip - margin),
        "margin_status": "engineering planning assumption; not a BQ specification or verified protection coordination margin",
    }


def default_algorithm_state(p: dict[str, Any], *, alpha_validated: bool) -> dict[str, Any]:
    """A fully qualified hypothetical state for deterministic model tests."""
    return {
        # Bounds cover the coldest and warmest cell across all 36 cells.
        "cell_temperature_lower_bound_C": 25.0,
        "cell_temperature_upper_bound_C": 25.0,
        "cell_temperature_bounds_qualified_all_36_cells": True,
        "temperature_input_fresh": True,
        "cell_voltage_headroom_valid": True,
        "bq_current_calibration_readback_valid": True,
        "occ_code3_readback_valid": True,
        "source_local_overshoot_bound_qualified": True,
        "source_capability_fresh": True,
        "shunt_thermal_bound_valid": True,
        "shunt_temperature_upper_bound_C": p["shunt"]["thermal_release_limit_C_candidate"],
        "sense_path_underread_bound_uohm": p["shunt"]["reference_algorithm_sense_path_underread_uohm_limit"],
        "source_current_overshoot_bound_A": p["normal_ccl_model"]["source_current_overshoot_A_design_budget"],
        "alpha_upper_bound_validated": alpha_validated,
        "alpha_max_branch_share": p["pack"]["alpha_design_assumption"],
        # Needed for alpha=1 fallback: no branch current reverses or circulates.
        "no_counterflow_current_qualified": True,
    }


def reference_ccl_algorithm(p: dict[str, Any], request_A: float,
                            state: dict[str, Any]) -> dict[str, Any]:
    """Executable fail-closed reference. This is not deployed firmware."""
    if not is_finite_number(request_A):
        return {
            "status": "FAIL_CLOSED_CCL_ZERO",
            "requested_CCL_A": None,
            "command_CCL_A": 0.0,
            "reason_codes": ["INVALID_CCL_REQUEST"],
            "actuation_status": "reference_algorithm_only_not_deployed",
        }
    required_true = (
        "cell_temperature_bounds_qualified_all_36_cells",
        "temperature_input_fresh",
        "cell_voltage_headroom_valid",
        "bq_current_calibration_readback_valid",
        "occ_code3_readback_valid",
        "source_local_overshoot_bound_qualified",
        "source_capability_fresh",
        "shunt_thermal_bound_valid",
    )
    reasons = [f"INVALID_OR_MISSING_{key.upper()}" for key in required_true if state.get(key) is not True]
    cell_temp_low = state.get("cell_temperature_lower_bound_C")
    cell_temp_high = state.get("cell_temperature_upper_bound_C")
    shunt_temp = state.get("shunt_temperature_upper_bound_C")
    path_uohm = state.get("sense_path_underread_bound_uohm")
    overshoot = state.get("source_current_overshoot_bound_A")
    max_path = p["shunt"]["reference_algorithm_sense_path_underread_uohm_limit"]
    max_overshoot = p["normal_ccl_model"]["source_current_overshoot_A_design_budget"]
    max_shunt_temp = p["shunt"]["thermal_release_limit_C_candidate"]
    low_valid = is_finite_number(cell_temp_low)
    high_valid = is_finite_number(cell_temp_high)
    if not low_valid:
        reasons.append("INVALID_CELL_TEMPERATURE_LOWER_BOUND")
    if not high_valid:
        reasons.append("INVALID_CELL_TEMPERATURE_UPPER_BOUND")
    if low_valid and high_valid:
        test_low, test_high = p["pack"]["cell_max_charge_test_temperature_C"]
        if cell_temp_low > cell_temp_high:
            reasons.append("CELL_TEMPERATURE_BOUNDS_REVERSED")
        elif cell_temp_low < test_low or cell_temp_high > test_high:
            reasons.append("CELL_TEMPERATURE_BOUNDS_OUTSIDE_QUALIFIED_MAX_RATE_TEST_WINDOW")
    if not is_finite_number(shunt_temp):
        reasons.append("INVALID_SHUNT_TEMPERATURE_UPPER_BOUND")
    elif shunt_temp > max_shunt_temp:
        reasons.append("SHUNT_TEMPERATURE_BOUND_EXCEEDS_DESIGN_LIMIT")
    if not is_finite_number(path_uohm) or path_uohm < 0:
        reasons.append("INVALID_SENSE_PATH_UNDERREAD_BOUND")
    elif path_uohm > max_path:
        reasons.append("SENSE_PATH_UNDERREAD_BOUND_EXCEEDS_DESIGN_LIMIT")
    if not is_finite_number(overshoot) or overshoot < 0:
        reasons.append("INVALID_SOURCE_OVERSHOOT_BOUND")
    elif overshoot > max_overshoot:
        reasons.append("SOURCE_OVERSHOOT_BOUND_EXCEEDS_DESIGN_BUDGET")
    if reasons:
        return {
            "status": "FAIL_CLOSED_CCL_ZERO",
            "requested_CCL_A": max(0.0, request_A),
            "command_CCL_A": 0.0,
            "reason_codes": reasons,
            "actuation_status": "reference_algorithm_only_not_deployed",
        }

    alpha_validated = state.get("alpha_upper_bound_validated") is True
    if alpha_validated:
        alpha = state.get("alpha_max_branch_share")
        minimum_alpha = 1.0 / p["pack"]["parallel_cells"]
        if not is_finite_number(alpha) or not minimum_alpha <= alpha <= 1:
            return {
                "status": "FAIL_CLOSED_CCL_ZERO",
                "requested_CCL_A": max(0.0, request_A),
                "command_CCL_A": 0.0,
                "reason_codes": ["INVALID_VALIDATED_ALPHA_BOUND"],
                "actuation_status": "reference_algorithm_only_not_deployed",
            }
    else:
        if state.get("no_counterflow_current_qualified") is not True:
            return {
                "status": "FAIL_CLOSED_CCL_ZERO",
                "requested_CCL_A": max(0.0, request_A),
                "command_CCL_A": 0.0,
                "reason_codes": ["ALPHA_FALLBACK_REQUIRES_QUALIFIED_NO_COUNTERFLOW"],
                "actuation_status": "reference_algorithm_only_not_deployed",
            }
        alpha = p["pack"]["alpha_unvalidated_fallback"]

    calculation = compute_ccl(
        p, alpha, max_shunt_temp, path_uohm, overshoot,
        cell_temp_upper_C=cell_temp_high, request_A=request_A,
    )
    return {
        "status": "PASS_LIMITED_BY_REFERENCE_ALGORITHM",
        "command_CCL_A": calculation["quantized_CCL_candidate_A"],
        "reason_codes": (["ALPHA_VALIDATED_BOUND_INCLUDES_WORST_BRANCH_CURRENT"] if alpha_validated else [
            "ALPHA_UNVALIDATED_FALLBACK_TO_1",
            "NO_COUNTERFLOW_CURRENT_QUALIFIED",
        ]),
        "alpha_upper_bound_used": alpha,
        "alpha_was_validated": alpha_validated,
        "cell_temperature_lower_bound_C": cell_temp_low,
        "cell_temperature_upper_bound_C": cell_temp_high,
        "temperature_bounds_cover_all_36_cells": True,
        "no_counterflow_current_qualified": state.get("no_counterflow_current_qualified") is True,
        "shunt_temperature_C_used": max_shunt_temp,
        "sense_path_underread_uohm_used": path_uohm,
        "source_overshoot_bound_A_used": overshoot,
        "calculation": calculation,
        "actuation_status": "reference_algorithm_only_not_deployed",
    }


def bus_case(p: dict[str, Any], regen_W: float, stop_delay_ms: float,
             ramp_ms: float, sc: dict[str, Any], sc_delay_ms: float) -> dict[str, Any]:
    b = p["bus_energy_screen"]
    dt_s = b["integration_step_ms"] * 1e-3
    end_s = b["simulation_horizon_ms"] * 1e-3
    e_limit = 0.5 * b["bus_capacitance_F"] * (
        b["bus_voltage_limit_V"] ** 2 - b["bus_initial_voltage_V"] ** 2
    )
    available_sc_J = sc["available_energy_J"]
    sc_used_J = 0.0
    bus_energy_J = 0.0
    crossed_at_ms = None
    for n in range(int(round(end_s / dt_s))):
        t = n * dt_s
        t_ms = t * 1000.0
        # A local source stop request is delayed, then both modeled sources
        # linearly ramp to zero. No CAN timing is used in this model.
        if t_ms <= stop_delay_ms:
            fraction = 1.0
        elif ramp_ms <= 0:
            fraction = 0.0
        else:
            fraction = max(0.0, 1.0 - (t_ms - stop_delay_ms) / ramp_ms)
        source_W = (regen_W + b["solar_power_W"]) * fraction
        absorber_W = 0.0
        if t_ms >= sc_delay_ms and sc_used_J < available_sc_J:
            absorber_W = min(sc["power_limit_W"], source_W)
            absorber_W = min(absorber_W, (available_sc_J - sc_used_J) / dt_s)
        net_W = max(0.0, source_W - absorber_W - b["bus_load_W"])
        bus_energy_J += net_W * dt_s
        sc_used_J += absorber_W * dt_s
        if bus_energy_J >= e_limit and crossed_at_ms is None:
            crossed_at_ms = t_ms + b["integration_step_ms"]
            break
        if source_W <= 0 and t_ms > stop_delay_ms + ramp_ms:
            break
    v_end = math.sqrt(b["bus_initial_voltage_V"] ** 2
                      + 2.0 * bus_energy_J / b["bus_capacitance_F"])
    return {
        "regen_W": regen_W,
        "solar_W": b["solar_power_W"],
        "local_source_stop_delay_ms": stop_delay_ms,
        "linear_source_ramp_down_ms": ramp_ms,
        "absorber_case": sc["name"],
        "absorber_response_delay_ms": sc_delay_ms,
        "absorber_energy_used_J": sc_used_J,
        "bus_injected_energy_J": bus_energy_J,
        "bus_voltage_at_end_or_limit_V": min(v_end, b["bus_voltage_limit_V"]),
        "bus_limit_crossed": crossed_at_ms is not None,
        "bus_limit_crossing_time_ms": crossed_at_ms,
        "available_bus_energy_headroom_J": e_limit,
    }


def verify_artifacts(p: dict[str, Any]) -> dict[str, Any]:
    enet_path = ROOT / p["evidence_files"]["formal_netlist"]
    bom_path = ROOT / p["evidence_files"]["bom"]
    old_path = ROOT / "BMS模块" / "检查" / "行进充电与回收支持审查.json"
    enet = load_json(enet_path)
    r = enet["components"]["RSHUNT"]
    props = r["props"]
    pin_info = r["pinInfoMap"]
    with bom_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bom_rows = [x for x in rows if x.get("位号") == "RSHUNT"]
    formal_ok = (
        props.get("Manufacturer Part") == p["shunt"]["manufacturer_part"]
        and props.get("Supplier Part") == p["shunt"]["supplier_part"]
        and pin_info.get("1", {}).get("net") == "B0"
        and pin_info.get("2", {}).get("net") == "PACK-"
    )
    bom_ok = (
        len(bom_rows) == 1
        and bom_rows[0].get("MPN") == p["shunt"]["manufacturer_part"]
        and bom_rows[0].get("位号") == "RSHUNT"
    )
    old = load_json(old_path)
    old_wording = old.get("battery_boundary", {}).get("bms_oc_current_protection", "")
    old_file_text = old_path.read_text(encoding="utf-8")
    # Scan the complete review file, not only one field, so stale copied prose
    # cannot be hidden by a path/field mismatch.
    typo_patterns = (
        "3mV阈值名义12A",
        "3 mV阈值名义12A",
        "3mV阈值，名义12A",
        "3 mV阈值，名义12A",
    )
    typo_detected = any(pattern in old_file_text for pattern in typo_patterns)
    corrected_text_present = (
        "OCC寄存器 code 3（实际写入/读回未验证）= 6mV" in old_wording
        and "不是恒流调节或16A硬上限" in old_wording
        and "早动作角约7.92A" in old_wording
    )
    return {
        "formal_netlist": {
            "path": str(enet_path.relative_to(ROOT)),
            "sha256": sha256(enet_path),
            "mpn": props.get("Manufacturer Part"),
            "supplier_part": props.get("Supplier Part"),
            "pin1_net": pin_info.get("1", {}).get("net"),
            "pin2_net": pin_info.get("2", {}).get("net"),
            "matches_model_and_expected_nets": formal_ok,
        },
        "bom": {
            "path": str(bom_path.relative_to(ROOT)),
            "sha256": sha256(bom_path),
            "rshunt_rows": len(bom_rows),
            "mpn": bom_rows[0].get("MPN") if bom_rows else None,
            "purchase_status": bom_rows[0].get("采购状态") if bom_rows else None,
            "matches_model": bom_ok,
        },
        "old_active_review": {
            "path": str(old_path.relative_to(ROOT)),
            "sha256": sha256(old_path),
            "contains_threshold_typo": typo_detected,
            "corrected_wording_present": corrected_text_present,
            "current_text": old_wording,
            "recommended_correction": (
                "当前设计候选OCC寄存器 code 3（实际写入/读回未验证）= 6mV，名义12A；"
                "小阈值负误差−2mV与0.505mΩ分流器高角组合时早动作角约7.92A；"
                "小阈值正误差+2.65mV与0.495mΩ分流器低角组合时晚动作角约17.5A。"
                "OCC仅作延时过流切断，不是恒流调节或16A硬上限。"
            ),
            "correction_applied_before_model_run": corrected_text_present,
        },
    }


def main() -> None:
    p = load_json(INPUT)
    artifacts = verify_artifacts(p)
    err_A = current_error_bound_A(p)
    alpha = p["pack"]["alpha_design_assumption"]
    overshoot = p["normal_ccl_model"]["conditional_model_case_source_current_overshoot_A"]
    sensitivity = [
        compute_ccl(p, a, t, path, ios)
        for a in p["pack"]["alpha_sweep"]
        for t in p["shunt"]["shunt_temperature_sweep_C"]
        for path in p["shunt"]["sense_path_underread_uohm_sweep"]
        for ios in p["normal_ccl_model"]["source_current_overshoot_A_sweep"]
    ]
    alpha_validated_state = default_algorithm_state(p, alpha_validated=True)
    alpha_unvalidated_state = default_algorithm_state(p, alpha_validated=False)
    validated_16a_request = reference_ccl_algorithm(p, 16.0, alpha_validated_state)
    fallback_16a_request = reference_ccl_algorithm(p, 16.0, alpha_unvalidated_state)
    low_request = reference_ccl_algorithm(p, 1.0, alpha_validated_state)
    temp_fault_state = dict(alpha_validated_state, cell_temperature_upper_bound_C=28.0)
    low_temp_fault_state = dict(alpha_validated_state, cell_temperature_lower_bound_C=5.0)
    reversed_temp_state = dict(alpha_validated_state, cell_temperature_lower_bound_C=26.0,
                               cell_temperature_upper_bound_C=25.0)
    nan_temp_state = dict(alpha_validated_state, cell_temperature_lower_bound_C=float("nan"))
    missing_temp_lower_state = {
        k: v for k, v in alpha_validated_state.items() if k != "cell_temperature_lower_bound_C"
    }
    stale_temp_state = dict(alpha_validated_state, temperature_input_fresh=False)
    source_fault_state = dict(alpha_validated_state, source_current_overshoot_bound_A=1.01)
    path_fault_state = dict(alpha_validated_state, sense_path_underread_bound_uohm=20.01)
    cell_headroom_fault_state = dict(alpha_validated_state, cell_voltage_headroom_valid=False)
    invalid_alpha_state = dict(alpha_validated_state, alpha_max_branch_share=1.01)
    below_minimum_alpha_state = dict(alpha_validated_state, alpha_max_branch_share=0.01)
    bool_request = reference_ccl_algorithm(p, True, alpha_validated_state)
    bool_shunt_temp_state = dict(alpha_validated_state, shunt_temperature_upper_bound_C=True)
    bool_path_state = dict(alpha_validated_state, sense_path_underread_bound_uohm=True)
    bool_overshoot_state = dict(alpha_validated_state, source_current_overshoot_bound_A=True)
    bool_alpha_state = dict(alpha_validated_state, alpha_max_branch_share=True)
    no_counterflow_unqualified_state = dict(alpha_unvalidated_state, no_counterflow_current_qualified=False)
    fault_cases = {
        "outside_manufacturer_max_rate_test_window": reference_ccl_algorithm(p, 16.0, temp_fault_state),
        "one_or_more_cells_below_test_window": reference_ccl_algorithm(p, 16.0, low_temp_fault_state),
        "temperature_bounds_reversed": reference_ccl_algorithm(p, 16.0, reversed_temp_state),
        "temperature_bound_nan": reference_ccl_algorithm(p, 16.0, nan_temp_state),
        "temperature_lower_bound_missing": reference_ccl_algorithm(p, 16.0, missing_temp_lower_state),
        "stale_temperature_input": reference_ccl_algorithm(p, 16.0, stale_temp_state),
        "source_overshoot_exceeds_design_budget": reference_ccl_algorithm(p, 16.0, source_fault_state),
        "sense_path_bound_exceeds_design_limit": reference_ccl_algorithm(p, 16.0, path_fault_state),
        "cell_voltage_headroom_invalid": reference_ccl_algorithm(p, 16.0, cell_headroom_fault_state),
        "invalid_claimed_alpha_bound": reference_ccl_algorithm(p, 16.0, invalid_alpha_state),
        "claimed_alpha_below_four_branch_mathematical_minimum": reference_ccl_algorithm(
            p, 16.0, below_minimum_alpha_state
        ),
        "boolean_request_rejected": bool_request,
        "boolean_shunt_temperature_rejected": reference_ccl_algorithm(p, 16.0, bool_shunt_temp_state),
        "boolean_sense_path_bound_rejected": reference_ccl_algorithm(p, 16.0, bool_path_state),
        "boolean_source_overshoot_bound_rejected": reference_ccl_algorithm(p, 16.0, bool_overshoot_state),
        "boolean_alpha_bound_rejected": reference_ccl_algorithm(p, 16.0, bool_alpha_state),
        "missing_temperature_qualification": reference_ccl_algorithm(
            p, 16.0, {k: v for k, v in alpha_validated_state.items()
                       if k != "cell_temperature_bounds_qualified_all_36_cells"}
        ),
        "unvalidated_alpha_without_no_counterflow_qualification": reference_ccl_algorithm(
            p, 16.0, no_counterflow_unqualified_state
        ),
    }
    comparison_alpha = {
        str(a): compute_ccl(
            p, a, p["shunt"]["thermal_release_limit_C_candidate"],
            p["shunt"]["conditional_model_case_sense_path_underread_uohm"],
            overshoot,
        )
        for a in [0.25, alpha, 0.4, 1.0]
    }
    b = p["bus_energy_screen"]
    bus_cases = [
        bus_case(p, regen, delay, ramp, sc, sc_delay)
        for regen in b["regen_power_W_sweep"]
        for delay in b["source_stop_delay_ms_sweep"]
        for ramp in b["source_power_ramp_down_ms_sweep"]
        for sc in b["supercap_cases"]
        for sc_delay in b["supercap_response_delay_ms_sweep"]
    ]
    headroom_J = 0.5 * b["bus_capacitance_F"] * (
        b["bus_voltage_limit_V"] ** 2 - b["bus_initial_voltage_V"] ** 2
    )
    occ = occ_bounds(p)
    validated_case = validated_16a_request["calculation"]
    fallback_case = fallback_16a_request["calculation"]
    checks = {
        "netlist_and_bom_shunt_match": (
            artifacts["formal_netlist"]["matches_model_and_expected_nets"]
            and artifacts["bom"]["matches_model"]
        ),
        "net_charge_command_never_exceeds_user_16A_ceiling": all(
            x["quantized_CCL_candidate_A"] <= p["pack"]["user_net_charge_ceiling_A"]
            for x in sensitivity
        ),
        "modeled_peak_respects_user_cell_and_OCC_constraints": all(
            x["model_predicted_pack_peak_A"] <= x["applicable_pack_ceiling_A"] + 1e-9
            and x["predicted_max_branch_within_cell_rate_limit"]
            and x["ccl_candidate_peak_meets_occ_planning_margin"]
            for x in sensitivity
        ),
        "OCC_early_and_late_corners_include_shunt_TCR_at_75C": (
            occ["shunt_high_corner_ohm_including_TCR"] > p["shunt"]["nominal_ohm"]
            and occ["shunt_low_corner_ohm_including_TCR"] < p["shunt"]["nominal_ohm"]
            and occ["early_trip_current_A"] < 7.9208
            and occ["late_trip_current_A"] > 16.0
        ),
        "validated_alpha_16A_request_is_limited_under_OCC_margin": (
            validated_16a_request["command_CCL_A"] < 16.0
            and validated_case["ccl_candidate_peak_meets_occ_planning_margin"]
            and validated_case["predicted_max_branch_within_cell_rate_limit"]
        ),
        "unvalidated_alpha_uses_full_single_branch_fallback": (
            fallback_16a_request["alpha_upper_bound_used"] == 1.0
            and fallback_16a_request["no_counterflow_current_qualified"] is True
            and fallback_case["predicted_max_branch_within_cell_rate_limit"]
            and fallback_case["ccl_candidate_peak_meets_occ_planning_margin"]
        ),
        "all_36_cell_temperature_lower_and_upper_bounds_required": (
            validated_16a_request["temperature_bounds_cover_all_36_cells"]
            and validated_16a_request["cell_temperature_lower_bound_C"] >= 23.0
            and validated_16a_request["cell_temperature_upper_bound_C"] <= 27.0
            and all(fault_cases[name]["command_CCL_A"] == 0.0 for name in (
                "one_or_more_cells_below_test_window", "temperature_bounds_reversed", "temperature_bound_nan"
            ))
        ),
        "validated_alpha_respects_four_branch_mathematical_minimum": (
            fault_cases["claimed_alpha_below_four_branch_mathematical_minimum"]["command_CCL_A"] == 0.0
            and fault_cases["claimed_alpha_below_four_branch_mathematical_minimum"]["reason_codes"]
            == ["INVALID_VALIDATED_ALPHA_BOUND"]
        ),
        "boolean_numeric_inputs_fail_closed": (
            bool_request["command_CCL_A"] == 0.0
            and all(fault_cases[name]["command_CCL_A"] == 0.0 for name in (
                "boolean_shunt_temperature_rejected", "boolean_sense_path_bound_rejected",
                "boolean_source_overshoot_bound_rejected", "boolean_alpha_bound_rejected",
            ))
        ),
        "alpha_fallback_requires_qualified_no_counterflow": (
            fallback_16a_request["command_CCL_A"] > 0.0
            and no_counterflow_unqualified_state["no_counterflow_current_qualified"] is False
            and fault_cases["unvalidated_alpha_without_no_counterflow_qualification"]["command_CCL_A"] == 0.0
        ),
        "requested_current_is_never_raised": low_request["command_CCL_A"] <= 1.0,
        "all_fault_cases_fail_closed_to_zero": all(
            case["status"] == "FAIL_CLOSED_CCL_ZERO" and case["command_CCL_A"] == 0.0
            for case in fault_cases.values()
        ),
        "OCC_design_candidate_write_readback_status_unverified": (
            not occ["configuration_written_or_readback_verified"]
        ),
        "old_3mV_wording_removed": not artifacts["old_active_review"]["contains_threshold_typo"],
        "corrected_OCC_wording_present": artifacts["old_active_review"]["corrected_wording_present"],
        "alpha_is_unvalidated_assumption": not p["pack"]["alpha_assumption_validated"],
        "release_blocked_by_unresolved_gates": any(
            value is False for value in p["release_gates_current_status"].values()
        ),
    }
    result = {
        "schema_version": 2,
        "title": p["title"],
        "model_status": "PASS executable reference algorithm and deterministic fail-safe tests; NOT firmware/hardware/vehicle acceptance",
        "release_status": "BLOCKED pending physical input qualification, firmware implementation/readback, branch-current, local source-response, and sink validation",
        "provenance": {
            "input_path": str(INPUT.relative_to(ROOT)),
            "input_sha256": sha256(INPUT),
            "model_source_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "model_source_sha256": sha256(Path(__file__)),
            "artifacts": artifacts,
            "native_design_binding_note": (
                "Parameters were cross-checked to the current formal BMS-9S4P.enet and BOM at run time. "
                "The source/input are evidence-bound by hashes in this result only; this does not claim "
                "binding to a future native-design revision."
            ),
        },
        "measurement_error_model": {
            "one_sided_underread_lsb_sum": 22.3 + 1.0 + 0.03 * 35.0 + 1.0 + 1.0,
            "one_sided_underread_bound_A": err_A,
            "assumptions": p["normal_ccl_model"]["assumed_underread_error_components"],
            "important_limit": (
                "DNL 1 LSB and CC2 stability reserve 1 LSB are explicit conservative/model assumptions; "
                "the datasheet's 0.1 LSB DNL is typical and ±1 LSB stability is not a hard maximum. "
                "No installed board gain/offset readback or calibration is proven."
            ),
        },
        "ccl": {
            "formula": "I_CCL <= floor_q(min(I_request, max(0, K_min * (min(16A, I_cell_max(T)/alpha, I_OCC_early(Tshunt)-M_OCC) - I_os_bound) - E_underread)))",
            "K_min_definition": (
                "(BQ CC_GAIN_min / CC_GAIN_typ) * software_gain_scale_low * "
                "(Rshunt_low_corner - one_sided_sense_path_deficit) / Rshunt_nominal"
            ),
            "I_os_definition": p["normal_ccl_model"]["source_current_overshoot_definition"],
            "16A_is_user_net_charge_ceiling_not_minimum_delivery": True,
            "validated_alpha_1_over_3_hypothetical_16A_request": validated_16a_request,
            "unvalidated_alpha_full_single_branch_fallback_16A_request": fallback_16a_request,
            "fault_injection_results": fault_cases,
            "conditional_case_interpretation": (
                "The former 10.5A cell/measurement-only command is not used. The executable reference algorithm "
                "takes the minimum of the 16A user ceiling, temperature-valid cell ceiling and OCC early-corner "
                "planning boundary, then subtracts the worst-case dynamic overshoot and measurement underread. "
            "For alpha=1.0 fallback it also requires a qualified no-counterflow condition; otherwise it returns CCL=0. "
            "Cell temperature qualification requires all 36 cells covered by conservative minimum and maximum bounds."
            ),
            "alpha_comparison_at_75C_20uohm_path_1A_overshoot": comparison_alpha,
            "parameter_sweep_case_count": len(sensitivity),
            "parameter_sweep": sensitivity,
            "reference_algorithm_status": (
                "Executable Python design reference only, not deployed firmware. CCL remains zero unless all "
                "required validity/readback/source-bound gates pass. The 1A dynamic term is a designed acceptance "
            "budget, not measured source behavior; release remains blocked until local source loops demonstrate it. "
            "Alpha=1 fallback requires qualified no-counterflow current; any validated alpha must bound worst "
            "branch current, including circulating-current effects. Temperature bounds cover all 36 cells."
            ),
        },
        "occ": {
            **occ,
            "normal_CCL_compatibility_rule": (
                "Keep predicted peak <= early trip corner - explicit engineering planning margin. "
                "Do not rely on delayed OCC trip as normal current regulation."
            ),
            "interpretation": (
                "BQ76952 OCC is a threshold comparator plus programmed delay and autonomous CHG-FET trip "
                "when configured. Code 3 is only a design candidate, write/readback is unverified. "
                "It cannot regulate current and cannot guarantee a 16A ceiling."
            ),
        },
        "bus_energy": {
            "available_bus_headroom_J": headroom_J,
            "scenario_count": len(bus_cases),
            "cases": bus_cases,
            "interpretation": (
                "At CCL=0, energy must be handled by local source reduction/inhibit and/or a validated "
                "supercap DC/DC sink. The supercap is not added to the public-bus capacitance. Cases are "
                "parameterized screens, not measured source latency or installed sink capability."
            ),
        },
        "cell_and_pack": {
            "cell_charge_max_A": p["pack"]["cell_max_charge_A"],
            "manufacturer_test_temperature_C": p["pack"]["cell_max_charge_test_temperature_C"],
            "manufacturer_charge_temperature_range_C": p["pack"]["cell_charge_ambient_range_C"],
            "manufacturer_charge_surface_temperature_range_C": p["pack"]["cell_charge_surface_temperature_range_C"],
            "manufacturer_rating_temperature_note": p["pack"]["cell_current_rating_temperature_note"],
            "full_rate_outside_manufacturer_test_window": "UNVERIFIED; CCL=0 pending manufacturer temperature curve or pack qualification.",
            "algorithm_temperature_rule": "4A/cell max-rate model only when qualified conservative lower and upper temperature bounds cover all 36 cells, satisfy lower<=upper, and each interval lies wholly in [23,27]C; CCL=0 for missing, stale, invalid, reversed, uncovered, or out-of-window bounds until curve/qualification evidence exists.",
            "alpha_definition": p["pack"]["alpha_definition"],
            "alpha_design_assumption": alpha,
            "alpha_assumption_rationale": p["pack"]["alpha_design_assumption_rationale"],
            "alpha_assumption_validated": False,
            "ideal_equal_share_at_16A_per_cell_A": 16.0 / p["pack"]["parallel_cells"],
            "cell_headroom_at_16A_ideal_share_A": p["pack"]["cell_max_charge_A"] - 16.0 / p["pack"]["parallel_cells"],
            "pack_limit_at_alpha_1_over_3_before_measurement_and_dynamic_reserve_A": 4.0 / alpha,
            "qualification_statement": (
                "EVE 30P 4.0A is a manufacturer maximum-charge test at 25C, not an all-temperature rating. "
                "At ideal 4P balance 16A is exactly 4A/cell with zero branch-sharing margin; alpha<=1/3 "
                "is an unvalidated engineering assumption, not proof of 16A cell compliance."
            ),
        },
        "fault_behavior": p["control_fault_policy"],
        "release_gates": p["release_gates_current_status"],
        "reference_algorithm": {
            "required_state_fields": list(default_algorithm_state(p, alpha_validated=False).keys()),
            "fail_closed_policy": "Any missing/invalid all-cell lower/upper temperature bound, voltage headroom, calibration/readback, source bound/freshness, shunt thermal bound or path bound returns CCL=0.",
            "alpha_policy": "Use a validated worst-branch share when qualified. Alpha=1.0 fallback requires a qualified no-counterflow condition; otherwise CCL=0.",
            "temperature_policy": "Require conservative lower and upper bounds covering all 36 cells, with every cell interval inside 23..27C; missing, invalid, NaN, reversed, stale, or out-of-range bounds return CCL=0.",
            "dynamic_policy": "Source current overshoot bound must be qualified and <=1.0A; design acceptance uses max(0,dI/dt)*local_response_delay + additional_peak_overshoot <=1.0A.",
            "implementation_status": "Python reference is executable and tested; firmware/source actuation remains unimplemented and unverified.",
        },
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check_report = {
        "schema_version": 2,
        "title": p["title"],
        "model_status": result["model_status"],
        "release_status": result["release_status"],
        "result_file": str(OUTPUT.relative_to(ROOT)),
        "result_sha256": sha256(OUTPUT),
        "source_file": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__)),
        "input_file": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "checks": checks,
        "validated_alpha_1_over_3_16A_request_candidate": validated_16a_request,
        "unvalidated_alpha_fallback_16A_request_candidate": fallback_16a_request,
        "fault_injection_results": fault_cases,
        "occ_bounds": occ,
        "release_gates": p["release_gates_current_status"],
        "old_review_correction": artifacts["old_active_review"],
        "native_design_changed": False,
        "bom_changed": False,
        "limitations": [
            "计算模型与参数化能量筛查，不是固件、硬件、台架或整车验收。",
            "当前 BOM/netlist 交叉核对为运行时快照；后续原生工程修改后须重跑并重新绑定哈希。",
            "1A源超调和20uΩ采样路径误差是设计接受预算，不是实测或当前硬件保证。",
            "alpha<=1/3仍是未验证假设；参考算法在验证前按alpha=1.0回退，但只有在无支路反向/环流资格成立时才允许该回退，否则CCL=0。",
            "EVE 4A/cell最大测试电流只在25±2°C模型窗口使用；必须以覆盖全部36颗电芯的保守温度下界和上界确认每颗均在23..27°C。窗口外缺温度曲线所以CCL=0。",
            "OCC是过流延时切断，不是恒流调节；不增加独立比较器。",
            "本轮没有改动CAD、BOM、项目要求、风扇文档或其他模块。",
        ],
    }
    CHECK_OUTPUT.write_text(json.dumps(check_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(OUTPUT.relative_to(ROOT)),
        "underread_A": err_A,
        "validated_alpha_1_over_3_16A_command_A": validated_16a_request["command_CCL_A"],
        "unvalidated_alpha_fallback_16A_command_A": fallback_16a_request["command_CCL_A"],
        "OCC": occ,
        "bus_headroom_J": headroom_J,
        "bus_scenarios": len(bus_cases),
        "checks": checks,
        "release_status": result["release_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
