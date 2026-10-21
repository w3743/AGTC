#!/usr/bin/env python3
"""Current V1.6-BUSFIX per-cell protection reference qualification model.

Standard-library only. This is an algorithmic/analytical model, not firmware,
analog hardware, switching-power-stage, PCB, or physical qualification.
"""
from __future__ import annotations

import hashlib
import copy
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "稳压模块"
INPUT = MODULE / "仿真" / "输入" / "V1.6-BUSFIX" / "model_input.json"
OUTPUT = MODULE / "仿真" / "结果" / "V1.6-BUSFIX" / "cell_protection_qualification.json"
QUALIFICATION_KEYS = (
    "end_to_end_E100mV_validated",
    "80A_maximum_charge_current_validated",
    "trip_to_zero_current_T50ms_validated",
    "local_firmware_implementation_verified",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parallel(*resistances: float) -> float:
    return 1.0 / sum(1.0 / r for r in resistances)


def resistor_relative_corner(tol: float, tcr_ppm: float, delta_c: float,
                             tol_sign: int, tcr_sign: int) -> float:
    return 1.0 + tol_sign * tol + tcr_sign * tcr_ppm * abs(delta_c) / 1_000_000.0


def tap_error_corners(cfg: dict[str, Any], cell_voltage: float, tap_index: int,
                      temp_c: float) -> dict[str, float]:
    """Return max absolute reconstructed cumulative tap error at equal cell V."""
    r = cfg["resistor_network"]
    nominal_t = r["nominal_temperature_C"]
    delta = temp_c - nominal_t
    tol = r["nominal_tolerance_fraction"]
    values: list[float] = []
    tap_v = tap_index * cell_voltage
    if tap_index <= 7:
        khat = 1.0 + r["tap_high_ohm"] / r["tap_low_ohm"] + r["tap_high_ohm"] / r["tap_test_ohm"]
        for signs in itertools.product((-1, 1), repeat=6):
            hi = r["tap_high_ohm"] * resistor_relative_corner(tol, r["tap_high_tcr_ppm_per_C"], delta, signs[0], signs[1])
            lo = r["tap_low_ohm"] * resistor_relative_corner(tol, r["tap_low_tcr_ppm_per_C"], delta, signs[2], signs[3])
            rt = r["tap_test_ohm"] * resistor_relative_corner(tol, r["tap_test_tcr_ppm_per_C"], delta, signs[4], signs[5])
            va = (tap_v / hi) / (1.0 / hi + 1.0 / lo + 1.0 / rt)
            values.append(khat * va - tap_v)
    else:
        khat = 1.0 + r["simple_tap_high_ohm"] / r["simple_tap_low_ohm"]
        for signs in itertools.product((-1, 1), repeat=4):
            hi = r["simple_tap_high_ohm"] * resistor_relative_corner(tol, r["tap_high_tcr_ppm_per_C"], delta, signs[0], signs[1])
            lo = r["simple_tap_low_ohm"] * resistor_relative_corner(tol, r["tap_low_tcr_ppm_per_C"], delta, signs[2], signs[3])
            va = (tap_v / hi) / (1.0 / hi + 1.0 / lo)
            values.append(khat * va - tap_v)
    return {"min_error_V": min(values), "max_error_V": max(values),
            "max_abs_error_V": max(abs(min(values)), abs(max(values))),
            "tap_voltage_V": tap_v}


class LocalChargeProtection:
    """Small fail-closed state model; diagnostics and CAN cannot refresh samples."""
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.sample: list[float] | None = None
        self.sample_ms: float | None = None
        self.data_valid = False
        self.sample_test_low = False
        self.charge_request = False
        self.ov_latched = False
        gate = cfg.get("qualification_gate", {})
        self.qualification_ready = all(gate.get(key) is True for key in QUALIFICATION_KEYS)

    def boot(self) -> None:
        self.__init__(self.cfg)

    def submit_working_sample(self, cells: list[float], now_ms: float,
                              test_low: bool = True, complete: bool = True) -> None:
        if not test_low or not complete or len(cells) != self.cfg["series_cells"]:
            self.data_valid = False
            self.charge_request = False
            return
        try:
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in cells):
                raise ValueError("ADC samples must be numeric values, not bool/string coercions")
            vals = [float(v) for v in cells]
        except (TypeError, ValueError):
            self.data_valid = False
            self.charge_request = False
            return
        if any(not math.isfinite(v) for v in vals):
            self.data_valid = False
            self.charge_request = False
            return
        trip = self.cfg["cell_trip_measured_voltage_V"]
        if any(v >= trip for v in vals):
            self.ov_latched = True
            self.data_valid = False
            self.charge_request = False
            return
        if any(v < 0.0 or v > 3.0 for v in vals):
            self.data_valid = False
            self.charge_request = False
            return
        self.sample = vals
        self.sample_ms = now_ms
        self.data_valid = True
        self.sample_test_low = True

    def diagnostic_sample(self, _cells: list[float], _now_ms: float,
                          _test_low: bool) -> None:
        # Intentionally no update to working sample/timestamp/protection state.
        return

    def can_enable_command(self) -> None:
        # CAN requests are coordination only; local protection needs local request API.
        return

    def fresh_and_safe(self, now_ms: float, test_low: bool = True) -> bool:
        if not test_low or not self.sample_test_low or not self.data_valid:
            return False
        if self.sample is None or self.sample_ms is None:
            return False
        age = now_ms - self.sample_ms
        # Freshness/validity is independent of the latch: an explicit clear
        # must be possible once healthy samples arrive, while output gating
        # and re-arming still separately reject an active latch.
        return 0.0 <= age <= self.cfg["freshness_timeout_ms"]

    def request_local_charge(self, now_ms: float, test_low: bool = True) -> bool:
        if self.qualification_ready and self.fresh_and_safe(now_ms, test_low) and not self.ov_latched:
            self.charge_request = True
        else:
            self.charge_request = False
        return self.charge_request

    def clear_ov_latch(self, now_ms: float, test_low: bool,
                        explicit_clear: bool) -> bool:
        if not explicit_clear or self.charge_request or not self.fresh_and_safe(now_ms, test_low):
            return False
        assert self.sample is not None
        if not all(v < self.cfg["cell_latch_clear_measured_voltage_V"] for v in self.sample):
            return False
        self.ov_latched = False
        return True

    def output_enabled(self, now_ms: float, test_low: bool = True) -> bool:
        if not self.qualification_ready or not test_low or not self.fresh_and_safe(now_ms, test_low):
            self.charge_request = False
            return False
        return self.charge_request and not self.ov_latched


def run_state_tests(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    # The delivered input keeps every physical/firmware qualification gate
    # closed.  A copied candidate config exercises the conditional state
    # behavior only; it does not change the actual default-off result.
    candidate_cfg = copy.deepcopy(cfg)
    for key in QUALIFICATION_KEYS:
        candidate_cfg.setdefault("qualification_gate", {})[key] = True
    cells_target = [2.40] * cfg["series_cells"]
    cells_clear = [2.49] * cfg["series_cells"]
    p = LocalChargeProtection(cfg)
    tests: list[tuple[str, bool, str]] = []

    tests.append(("boot_default_charge_off", not p.output_enabled(0), "new instance starts disabled"))
    p.submit_working_sample(cells_target, 0)
    tests.append(("unqualified_input_never_enables_charge",
                  not p.request_local_charge(0) and not p.output_enabled(0),
                  "actual input qualification gates are false; default must remain off"))

    cfg = candidate_cfg
    cells_target = [2.40] * cfg["series_cells"]
    cells_clear = [2.49] * cfg["series_cells"]
    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    tests.append(("fresh_sample_does_not_auto_enable", not p.output_enabled(0), "explicit local request required"))
    tests.append(("explicit_local_request_enables_with_fresh_data", p.request_local_charge(0) and p.output_enabled(0), "complete fresh sample + TEST low"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    tests.append(("sample_age_20ms_accepted", p.output_enabled(20), "freshness boundary is inclusive at 20ms"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    tests.append(("sample_age_over_20ms_fails_closed", not p.output_enabled(20.001) and not p.charge_request,
                  "stale data cancels charge request"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    p.diagnostic_sample([2.60] * cfg["series_cells"], 19, _test_low=False)
    tests.append(("diagnostic_frame_does_not_refresh_working_sample",
                  not p.output_enabled(21) and p.sample_ms == 0,
                  "diagnostic sample cannot postpone staleness"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    tests.append(("test_high_cancels_charge", not p.output_enabled(1, test_low=False) and not p.charge_request,
                  "TEST must be confirmed low for working protection"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target[:7], 0, complete=False)
    tests.append(("incomplete_scan_cannot_enable", not p.request_local_charge(0), "all eight cells required"))
    p = LocalChargeProtection(cfg)
    p.submit_working_sample([2.4] * 7 + [float("nan")], 0)
    tests.append(("nonfinite_cell_cannot_enable", not p.request_local_charge(0), "NaN invalidates complete scan"))
    p = LocalChargeProtection(cfg)
    p.submit_working_sample([2.4] * 7 + [True], 0)
    tests.append(("boolean_sample_cannot_enable", not p.request_local_charge(0), "bool is not accepted as numeric ADC voltage"))
    p = LocalChargeProtection(cfg)
    p.submit_working_sample([2.4] * 7 + ["2.4"], 0)
    tests.append(("string_sample_cannot_enable", not p.request_local_charge(0), "numeric strings are not accepted as ADC voltage"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    trip_cells = list(cells_target)
    trip_cells[4] = cfg["cell_trip_measured_voltage_V"]
    p.submit_working_sample(trip_cells, 10)
    tests.append(("any_cell_at_trip_latches_off", p.ov_latched and not p.output_enabled(10),
                  "single-cell >=2.56V latches charge off"))

    # At the 19.6V planning/work-zone boundary, one cell can be at 2.66V
    # while seven others are 2.42V.  A -100mV measurement error places that
    # cell exactly at the 2.56V candidate trip threshold; total voltage alone
    # would not reach the retained 20.8V SC_OV threshold.
    actual_imbalance = [2.66] + [(19.6 - 2.66) / 7.0] * 7
    measured_imbalance = [2.56] + actual_imbalance[1:]
    p = LocalChargeProtection(cfg)
    p.submit_working_sample(measured_imbalance, 0)
    tests.append(("single_cell_imbalance_latches_at_19_6V_total",
                  abs(sum(actual_imbalance) - 19.6) < 1e-12 and p.ov_latched,
                  "actual 2.66V cell with -100mV measurement error trips at 2.56V while total is 19.60V < 20.8V SC_OV"))

    p = LocalChargeProtection(cfg)
    p.submit_working_sample(cells_target, 0)
    p.request_local_charge(0)
    p.submit_working_sample([2.56] + cells_target[1:], 10)
    p.charge_request = True  # adversarial state: reset API must refuse while requested
    p.submit_working_sample(cells_clear, 20)
    p.charge_request = True
    tests.append(("latch_clear_refused_while_charge_requested",
                  not p.clear_ov_latch(20, True, explicit_clear=True) and p.ov_latched,
                  "explicit clear cannot overlap a charge request"))

    p.charge_request = False
    tests.append(("explicit_clear_requires_healthy_below_clear_samples",
                  p.clear_ov_latch(20, True, explicit_clear=True) and not p.output_enabled(20),
                  "clear only removes latch; it does not re-arm power"))
    tests.append(("rearm_requires_new_local_request",
                  p.request_local_charge(20) and p.output_enabled(20), "manual local request after clear"))

    before = p.sample_ms
    p.can_enable_command()
    tests.append(("CAN_enable_command_cannot_override_or_refresh",
                  p.sample_ms == before and p.output_enabled(20), "CAN does not alter local sample or protection"))

    p.boot()
    tests.append(("reboot_defaults_charge_off", not p.output_enabled(0) and not p.charge_request,
                  "reset returns to inhibited state"))
    return [{"name": name, "pass": bool(passed), "evidence": evidence}
            for name, passed, evidence in tests]


def calculate_error_budget(cfg: dict[str, Any]) -> dict[str, Any]:
    series = cfg["series_cells"]
    hard = cfg["cell_hard_max_voltage_V"]
    adc = cfg["adc_gpio_stress_inputs"]
    r = cfg["resistor_network"]
    k = 1.0 + r["tap_high_ohm"] / r["tap_low_ohm"] + r["tap_high_ohm"] / r["tap_test_ohm"]
    c = r["tap_high_ohm"] / r["tap_test_ohm"]
    k8 = 1.0 + r["simple_tap_high_ohm"] / r["simple_tap_low_ohm"]
    rth = parallel(r["tap_high_ohm"], r["tap_low_ohm"], r["tap_test_ohm"])
    rth8 = parallel(r["simple_tap_high_ohm"], r["simple_tap_low_ohm"])
    lsb = adc["nominal_vref_V"] / (2 ** adc["adc_resolution_bits"])
    lsb_term = adc["stm32g431_single_adc_full_temperature_TUE_stress_lsb"] * lsb
    tap_adc = k * lsb_term
    tap_test_adc = c * cfg["test_feedback_divider_ratio"] * lsb_term
    t8_adc = k8 * lsb_term
    tap_leak = k * rth * adc["gpio_leakage_TT_a_A"]
    t8_leak = k8 * rth8 * adc["gpio_leakage_TT_a_A"]
    test_rth = 1.0 / (1.0 / 10_000.0 + 1.0 / 10_000.0)
    test_leak = c * cfg["test_feedback_divider_ratio"] * test_rth * adc["gpio_leakage_FT_a_A"]
    temp_results: dict[str, Any] = {}
    for temp in r["corner_temperatures_C"]:
        taps = {str(j): tap_error_corners(cfg, hard, j, temp) for j in range(1, series + 1)}
        cell_resistor = {"1": taps["1"]["max_abs_error_V"]}
        for j in range(2, series + 1):
            cell_resistor[str(j)] = taps[str(j)]["max_abs_error_V"] + taps[str(j - 1)]["max_abs_error_V"]
        cell_stress = {}
        shunt = cfg["sc_minus_candidate"]["uncompensated_offset_bound_mV"] / 1000.0
        for j in range(1, series + 1):
            resistor = cell_resistor[str(j)]
            if j == 1:
                adc_term = tap_adc + tap_test_adc
                leak_term = tap_leak + test_leak + shunt
            elif j < series:
                # The common TEST_FB correction cancels for adjacent T1..T7 cells.
                adc_term = 2.0 * tap_adc
                leak_term = 2.0 * tap_leak
            else:
                adc_term = t8_adc + tap_adc + tap_test_adc
                leak_term = t8_leak + tap_leak + test_leak
            cell_stress[str(j)] = {
                "resistor_corner_mV": resistor * 1000.0,
                "adc_TUE_conditional_mV": adc_term * 1000.0,
                "GPIO_leakage_conditional_mV": leak_term * 1000.0,
                "additive_known_stress_mV": (resistor + adc_term + leak_term) * 1000.0,
                "is_complete_hard_bound": False
            }
        temp_results[str(int(temp))] = {
            "cumulative_tap_resistor_error": taps,
            "cell_resistor_error_bound_mV": {key: value * 1000.0 for key, value in cell_resistor.items()},
            "cell_additive_known_stress": cell_stress
        }
    max_stress = max(v["additive_known_stress_mV"]
                     for temp in temp_results.values()
                     for v in temp["cell_additive_known_stress"].values())
    return {
        "nominal_coefficients": {"K_T1_to_T7": k, "c_TEST": c, "K_T8": k8,
                                 "Rth_T1_to_T7_ohm": rth, "Rth_T8_ohm": rth8},
        "conditions": "Equal cells at 2.7V, TEST low, BOM initial tolerance +/-0.1% with independent TCR corners; TCR considered to 125C. ADC and pad leakage terms are conditional data-sheet stress arithmetic.",
        "adc_stress": {"LSB_mV": lsb * 1000.0, "7_5_LSB_mV": lsb_term * 1000.0,
                       "tap_T1_to_T7_mV": tap_adc * 1000.0,
                       "TEST_FB_reconstruction_contribution_mV": tap_test_adc * 1000.0,
                       "tap_T8_mV": t8_adc * 1000.0,
                       "common_TEST_term_cancels_for_cells_2_to_7": True},
        "gpio_leakage_stress": {"T1_to_T7_tap_pin_per_cell_mV": tap_leak * 1000.0,
                                 "T8_pin_mV": t8_leak * 1000.0,
                                 "TEST_FB_per_use_mV": test_leak * 1000.0,
                                 "T1_to_T7_Rth_ohm": rth, "T8_Rth_ohm": rth8,
                                 "TEST_FB_Rth_ohm": test_rth},
        "corner_results_by_temperature_C": temp_results,
        "maximum_additive_known_stress_mV_not_a_closed_total_error": max_stress,
        "E100mV_qualification_candidate_vs_conditional_stress": "FAIL" if max_stress > 100.0 else "CONDITIONAL_ONLY",
        "unclosed_terms": cfg["unknown_or_unclosed_measurement_terms"]
    }


def calculate_protection(cfg: dict[str, Any]) -> dict[str, Any]:
    current = cfg["charge_current_bound_for_slew_A"]
    capacitance = cfg["cell_capacitance_min_EOL_F"]
    dt_s = cfg["response_time_qualification_candidate_ms"] / 1000.0
    dv_dt = current / capacitance
    dynamic = dv_dt * dt_s
    target = cfg["planning_charge_target_voltage_V"]
    error = cfg["measurement_error_qualification_abs_V"]
    trip = cfg["cell_trip_measured_voltage_V"]
    hard = cfg["cell_hard_max_voltage_V"]
    no_false_trip_margin = trip - (target + error)
    post_trip_bound = trip + error + dynamic
    hard_margin = hard - post_trip_bound
    return {
        "measurement_error_requirement_abs_V": error,
        "target_measured_V": target,
        "trip_measured_V": trip,
        "reset_measured_V": cfg["cell_latch_clear_measured_voltage_V"],
        "target_to_trip_no_false_trip_margin_mV": no_false_trip_margin * 1000.0,
        "current_A": current,
        "minimum_EOL_capacitance_per_cell_F": capacitance,
        "dv_dt_V_per_s": dv_dt,
        "response_budget_ms_unmeasured_qualification_limit": cfg["response_time_qualification_candidate_ms"],
        "dynamic_voltage_rise_mV": dynamic * 1000.0,
        "worst_true_voltage_after_trip_V": post_trip_bound,
        "margin_to_2_7V_hard_max_mV": hard_margin * 1000.0,
        "conditional_math_pass": no_false_trip_margin > 0.0 and hard_margin >= 0.0,
        "conditions": [
            "All eight cell-voltage estimates meet a validated end-to-end absolute error bound <=100mV over operating temperature, VREF/VDDA, input slew, and wiring conditions.",
            "Charge current never exceeds 80A up to current zero.",
            "Trip-to-zero-current response is measured/qualified <=50ms including scan wait, execution, PWM/driver turn-off and current decay.",
            "Freshness/TEST handling, latched shutdown, reset default, and no CAN override behave as this model."
        ],
        "separate_MCU_fault_boundary": "Independent hard protection against arbitrary MCU stuck PWM/GPIO/peripheral or firmware failure is not proved by ADC logic, reset defaults, or IWDG."
    }


def calculate_energy(cfg: dict[str, Any]) -> dict[str, Any]:
    n = cfg["series_cells"]
    c_eol = cfg["cell_capacitance_min_EOL_F"]
    low = cfg["planning_lower_voltage_V"]
    target = cfg["planning_charge_target_voltage_V"]
    retention = cfg["initial_capacitance_retention_at_EOL"]
    eta = cfg["bus_conversion_efficiency_ideal"]
    duration = cfg["energy_target_duration_s"]
    error = cfg["measurement_error_qualification_abs_V"]
    mid = math.sqrt((target * target + low * low) / 2.0)
    def stored_j(vhi: float, cap: float) -> float:
        return 0.5 * n * cap * (vhi * vhi - low * low)
    midpoint_stored = stored_j(mid, c_eol)
    midpoint_bus = midpoint_stored * eta
    actual_worst_at_measured_target = target - error
    target_worst_stored = stored_j(actual_worst_at_measured_target, c_eol)
    target_worst_bus = target_worst_stored * eta
    target_nominal_stored = stored_j(target, c_eol)
    target_nominal_bus = target_nominal_stored * eta
    energy_rows = []
    for power in cfg["energy_target_power_W"]:
        required_bus = power * duration
        midpoint_required_eol = required_bus / (eta * 0.5 * n * (mid * mid - low * low))
        fullwindow_required_eol = required_bus / (eta * 0.5 * n * (target * target - low * low))
        worst_target_required_eol = required_bus / (eta * 0.5 * n * (actual_worst_at_measured_target ** 2 - low * low))
        energy_rows.append({
            "power_W": power,
            "duration_s": duration,
            "required_bus_energy_J": required_bus,
            "required_EOL_cell_capacitance_F_at_energy_midpoint": midpoint_required_eol,
            "required_initial_cell_capacitance_F_at_energy_midpoint_if_80pct_EOL": midpoint_required_eol / retention,
            "required_EOL_cell_capacitance_F_from_full_2_45V_start": fullwindow_required_eol,
            "required_initial_cell_capacitance_F_from_full_2_45V_start_if_80pct_EOL": fullwindow_required_eol / retention,
            "required_EOL_cell_capacitance_F_at_measured_target_worst_actual": worst_target_required_eol,
            "current_280F_midpoint_pass": midpoint_bus >= required_bus,
            "current_280F_at_measured_target_worst_actual_pass": target_worst_bus >= required_bus,
            "ideal_energy_shortfall_J_at_midpoint": max(0.0, required_bus - midpoint_bus),
            "ideal_energy_shortfall_J_at_measured_target_worst_actual": max(0.0, required_bus - target_worst_bus)
        })
    return {
        "lower_voltage_V": low,
        "upper_planning_target_voltage_V": target,
        "energy_midpoint_voltage_V": mid,
        "minimum_EOL_capacitance_F_per_cell": c_eol,
        "conversion_efficiency_assumption": eta,
        "midpoint_storage_energy_J": midpoint_stored,
        "midpoint_ideal_bus_energy_J": midpoint_bus,
        "measured_target_worst_actual_voltage_V_if_error_100mV": actual_worst_at_measured_target,
        "worst_actual_target_storage_energy_J": target_worst_stored,
        "worst_actual_target_ideal_bus_energy_J": target_worst_bus,
        "nominal_actual_2_45V_storage_energy_J": target_nominal_stored,
        "nominal_actual_2_45V_ideal_bus_energy_J": target_nominal_bus,
        "energy_targets": energy_rows,
        "limitations": ["Ideal capacitor energy only; excludes ESR, converter dynamic loss beyond eta, wiring, temperature/cold derating and control reserve.",
                        "The 280F value is minimum end-of-life cell capacitance (350F initial nameplate x 80% retention), not a measured cell.",
                        "No topology or component selection is made; capacitance/pre-bias options require verified physical parts and cost."],
        "status": "FAIL_AT_ENERGY_MIDPOINT_AND_AT_E100MV_WORST_CASE_CHARGE_TARGET"
    }


def main() -> int:
    cfg = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    tests = run_state_tests(cfg)
    error_budget = calculate_error_budget(cfg)
    protection = calculate_protection(cfg)
    energy = calculate_energy(cfg)
    source_manifest_path = INPUT.parent / "source_snapshot_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))
    snapshot_checks = []
    for item in source_manifest:
        path = INPUT.parent / "source_snapshot" / item["staged"]
        actual = sha256(path)
        snapshot_checks.append({"file": item["staged"], "expected_sha256": item["source_sha256"],
                               "actual_sha256": actual, "matches_official_snapshot": actual.lower() == item["source_sha256"].lower()})
    result = {
        "model_id": cfg["model_id"],
        "status": "MODEL_ONLY_PARTIAL_PASS_PHYSICAL_QUALIFICATION_FAIL",
        "runtime": {"python": platform.python_version(), "dependencies": "Python standard library only"},
        "inputs": {"path": str(INPUT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(INPUT)},
        "source_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(Path(__file__).resolve())},
        "source_snapshot_checks": snapshot_checks,
        "state_machine_tests": {"pass_count": sum(1 for t in tests if t["pass"]), "fail_count": sum(1 for t in tests if not t["pass"]), "tests": tests},
        "qualification_gate": {
            "input_values": cfg["qualification_gate"],
            "all_required_gates_open": all(cfg["qualification_gate"].get(key) is True for key in QUALIFICATION_KEYS),
            "default_behavior": "charge output remains inhibited unless all four qualification gates are explicitly true",
            "candidate_state_tests": "conditional state behavior is tested on an in-memory copied input with gates true; persisted input remains false"
        },
        "protection_window": protection,
        "measurement_error_budget": error_budget,
        "energy": energy,
        "acceptance": {
            "algorithmic_state_model": "PASS" if all(t["pass"] for t in tests) else "FAIL",
            "current_input_charge_enable": "FAIL_CLOSED_OFF_QUALIFICATION_GATES_FALSE",
            "2_7V_single_cell_guard": "CONDITIONAL_PASS_ONLY_IF_E100_AND_T50MS_QUALIFIED" if protection["conditional_math_pass"] else "FAIL",
            "end_to_end_E100mV_measurement_qualification": error_budget["E100mV_qualification_candidate_vs_conditional_stress"],
            "80A_and_50ms_physical_response": "NOT_MEASURED",
            "800W_1000W_1s_energy_target": "FAIL",
            "20_8V_total_SC_OV_individual_cell_proof": "NO; total-voltage threshold is not per-cell protection",
            "hardware_MCU_failure_independence": "FAIL_OPEN",
            "hardware_or_firmware_acceptance": "NOT_ESTABLISHED"
        },
        "boundaries": [
            "No CAD, BOM or firmware change was made.",
            "The persisted qualification gate remains closed; conditional candidate state tests do not enable the current input or prove firmware/hardware qualification.",
            "The 20.8V total SC_OV value is retained; total voltage cannot prove per-cell <=2.7V.",
            "V1.5 LICAP energy/SPICE outputs remain historical and are not current BUSFIX simulation results.",
            "CAN cannot refresh local ADC working samples or override the local protection state."
        ]
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_snapshot_ok = all(x["matches_official_snapshot"] for x in snapshot_checks)
    state_pass = result["state_machine_tests"]["pass_count"]
    state_total = len(tests)
    print(
        "BUSFIX-CELL-PROTECTION: "
        f"state_tests={state_pass}/{state_total} {'PASS' if state_pass == state_total else 'FAIL'}; "
        f"T2.56_E100_T50ms_margin={protection['margin_to_2_7V_hard_max_mV']:.3f}mV; "
        f"raw_conditional_error_stress_max={error_budget['maximum_additive_known_stress_mV_not_a_closed_total_error']:.3f}mV "
        f"(E100={'PASS' if error_budget['E100mV_qualification_candidate_vs_conditional_stress'] != 'FAIL' else 'FAIL'}); "
        f"energy_midpoint={energy['midpoint_ideal_bus_energy_J']:.2f}J; "
        f"C_EOL@midpoint_800W/1000W="
        f"{energy['energy_targets'][0]['required_EOL_cell_capacitance_F_at_energy_midpoint']:.1f}/"
        f"{energy['energy_targets'][1]['required_EOL_cell_capacitance_F_at_energy_midpoint']:.1f}F_per_cell; "
        f"snapshot_hashes={'PASS' if all_snapshot_ok else 'FAIL'}; "
        f"actual_input_gate={'OPEN' if result['qualification_gate']['all_required_gates_open'] else 'CLOSED_OFF'}; "
        "overall=MODEL_ONLY_PARTIAL_PASS_PHYSICAL_QUALIFICATION_FAIL"
    )
    return 0 if state_pass == state_total and all_snapshot_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
