"""Paper screening for the existing eight-tap G431 measurement network.

This script does not alter the native schematic or procurement BOM.  Values are
deliberately explicit so the generated JSON can be reviewed independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "检查" / "低成本逐节监测核算.json"


def divider_ratio(r_hi: float, r_lo: float) -> float:
    return 1.0 + r_hi / r_lo


def parallel(*values: float) -> float:
    return 1.0 / sum(1.0 / value for value in values)


def ratio_tolerance_fraction(r_hi: float, r_lo: float, resistor_tol: float) -> float:
    # For K = 1 + Rhi/Rlo, the first-order worst ratio error is
    # (Rhi/(Rhi+Rlo)) * (|dRhi| + |dRlo|).
    return (r_hi / (r_hi + r_lo)) * (2.0 * resistor_tol)


def ratio_thermal_fraction(r_hi: float, r_lo: float, tcr_ppm_c: float, delta_c: float) -> float:
    # Bound opposite-sign TCR on the upper and equivalent lower resistances.
    return (r_hi / (r_hi + r_lo)) * (2.0 * tcr_ppm_c * 1e-6 * abs(delta_c))


def main() -> None:
    n_cells = 8
    cell_normal_trip_v = 2.30
    cell_warning_v = 2.20
    cell_hard_limit_v = 2.70
    cell_cap_nominal_f = 350.0
    cell_cap_eol_screen_f = 280.0
    charge_peak_a = 50.0
    failed_sample_window_s = 0.1454
    shunt_ohm = 0.0005
    bus_min_v = 22.5
    aux_converter_min_v = 18.0
    adc_vref_v = 3.3
    adc_bits = 12
    adc_et_max_lsb = 7.5
    adc_inl_max_lsb = 4.4
    adc_clock_hz = 60e6
    adc_max_sample_cycles = 640.5
    adc_12bit_conversion_cycles = 653.0
    adc_slow_source_max_ohm = 33_000.0
    ambient_cal_c = 25.0
    ambient_min_c = 0.0
    ambient_max_c = 60.0
    resistor_tcr_ppm_c = 25.0
    vrefint_tcr_ppm_c = 50.0
    vrefint_factory_cal_c = 30.0
    vrefint_factory_cal_uncertainty_c = 5.0
    resistor_tolerance = 0.001

    r_hi = 68_000.0
    r_lo = 10_000.0
    r_test = 47_000.0
    # The schematic connects T1..T7 AD nodes to PB10/TEST through 47 kOhm.
    # Holding TEST low makes that branch a known additional lower-leg path.
    r_lo_test_low = parallel(r_lo, r_test)
    r_lo_t8 = r_lo
    ratio_test_low = divider_ratio(r_hi, r_lo_test_low)
    ratio_t8 = divider_ratio(r_hi, r_lo_t8)
    rth_test_low = parallel(r_hi, r_lo, r_test)
    rth_t8 = parallel(r_hi, r_lo_t8)

    adc_lsb_v = adc_vref_v / (2**adc_bits)
    age_rise_v = charge_peak_a * failed_sample_window_s / cell_cap_eol_screen_f
    shunt_drop_v = charge_peak_a * shunt_ohm
    ratio_tol_7 = ratio_tolerance_fraction(r_hi, r_lo_test_low, resistor_tolerance)
    ratio_tol_8 = ratio_tolerance_fraction(r_hi, r_lo_t8, resistor_tolerance)
    ratio_tcr_7 = ratio_thermal_fraction(
        r_hi,
        r_lo_test_low,
        resistor_tcr_ppm_c,
        max(abs(ambient_min_c - ambient_cal_c), abs(ambient_max_c - ambient_cal_c)),
    )
    ratio_tcr_8 = ratio_thermal_fraction(
        r_hi,
        r_lo_t8,
        resistor_tcr_ppm_c,
        max(abs(ambient_min_c - ambient_cal_c), abs(ambient_max_c - ambient_cal_c)),
    )
    vref_residual_fraction = vrefint_tcr_ppm_c * 1e-6 * max(
        abs(ambient_min_c - vrefint_factory_cal_c),
        abs(ambient_max_c - vrefint_factory_cal_c),
    ) + vrefint_tcr_ppm_c * 1e-6 * vrefint_factory_cal_uncertainty_c

    tap_ratios = [ratio_test_low] * 7 + [ratio_t8]
    cumulative_at_trip = [(idx + 1) * cell_normal_trip_v for idx in range(n_cells)]
    tap_adc_v_at_trip = [v / ratio for v, ratio in zip(cumulative_at_trip, tap_ratios)]
    tap_adc_v_at_hard = [
        (idx + 1) * cell_hard_limit_v / ratio for idx, ratio in enumerate(tap_ratios)
    ]
    tap_lsb_equiv_v = [adc_lsb_v * ratio for ratio in tap_ratios]
    tap_rth_ohm = [rth_test_low] * 7 + [rth_t8]

    cell_rows = []
    for i in range(n_cells):
        upper_idx = i
        lower_idx = i - 1

        # Single-ended slow-channel total error per tap, conservatively summed
        # for adjacent taps. For cell 1, the lower endpoint is SC_MINUS/GND_SYS.
        adc_et = adc_et_max_lsb * tap_lsb_equiv_v[upper_idx]
        if lower_idx >= 0:
            adc_et += adc_et_max_lsb * tap_lsb_equiv_v[lower_idx]

        upper_tol = ratio_tol_8 if upper_idx == 7 else ratio_tol_7
        upper_tcr = ratio_tcr_8 if upper_idx == 7 else ratio_tcr_7
        divider_tol = upper_tol * cumulative_at_trip[upper_idx]
        divider_tcr = upper_tcr * cumulative_at_trip[upper_idx]
        if lower_idx >= 0:
            lower_tol = ratio_tol_8 if lower_idx == 7 else ratio_tol_7
            lower_tcr = ratio_tcr_8 if lower_idx == 7 else ratio_tcr_7
            divider_tol += lower_tol * cumulative_at_trip[lower_idx]
            divider_tcr += lower_tcr * cumulative_at_trip[lower_idx]

        # VREFINT is factory-characterized per die at 30 C; its specified max
        # average tempco is used as the residual after calibration. This is a
        # screening estimate, not a production guarantee for a complete board.
        vref_err = cell_normal_trip_v * vref_residual_fraction
        # The 0.5 mOhm SC_MINUS/GND_SYS offset is common to upper taps. Use the
        # requested 0..50 A one-polarity bound for the first cell and for change
        # between adjacent reads; current reversal during a pair would need a
        # separate +/-50 mV bound and is intentionally not combined here.
        ground_err = shunt_drop_v
        age_err = age_rise_v

        raw_at_25 = adc_et + divider_tol + vref_err + ground_err
        calibrated_screen = (
            adc_inl_max_lsb * tap_lsb_equiv_v[upper_idx]
            + 0.5 * tap_lsb_equiv_v[upper_idx]
            + divider_tcr
            + vref_err
            + ground_err
        )
        if lower_idx >= 0:
            calibrated_screen += (
                adc_inl_max_lsb * tap_lsb_equiv_v[lower_idx]
                + 0.5 * tap_lsb_equiv_v[lower_idx]
            )

        # Recalculate calibrated ADC terms after the optional lower tap is
        # accounted for above; the total adds sample-age rise last.
        cell_rows.append(
            {
                "cell": i + 1,
                "upper_tap_adc_v_at_2p30": round(tap_adc_v_at_trip[upper_idx], 6),
                "upper_tap_adc_v_at_2p70": round(tap_adc_v_at_hard[upper_idx], 6),
                "adc_equivalent_lsb_mV_at_upper_tap": round(
                    tap_lsb_equiv_v[upper_idx] * 1000.0, 4
                ),
                "adc_ET_pair_max_mV": round(adc_et * 1000.0, 3),
                "divider_tolerance_pair_max_mV_at_2p30": round(divider_tol * 1000.0, 3),
                "divider_temperature_pair_max_mV_after_25C_cal_0_to_60C": round(
                    divider_tcr * 1000.0, 3
                ),
                "VREFINT_common_gain_residual_mV": round(vref_err * 1000.0, 3),
                "ground_delta_bound_mV": round(ground_err * 1000.0, 3),
                "tail_age_rise_mV": round(age_err * 1000.0, 3),
                "uncalibrated_25C_total_screen_mV_including_tail": round(
                    (raw_at_25 + age_err) * 1000.0, 3
                ),
                "calibrated_components_total_screen_mV_including_tail": round(
                    (calibrated_screen + age_err) * 1000.0, 3
                ),
                "uncalibrated_25C_actual_cell_v_at_2p30_trip_bound": round(
                    cell_normal_trip_v + raw_at_25 + age_err, 6
                ),
                "calibrated_components_actual_cell_v_at_2p30_trip_bound": round(
                    cell_normal_trip_v + calibrated_screen + age_err, 6
                ),
            }
        )

    worst_raw = max(cell_rows, key=lambda row: row["uncalibrated_25C_total_screen_mV_including_tail"])
    worst_cal = max(
        cell_rows,
        key=lambda row: row["calibrated_components_total_screen_mV_including_tail"],
    )
    tau_280_s = 120.0 * cell_cap_eol_screen_f
    tau_350_s = 120.0 * cell_cap_nominal_f
    bleed_rows = []
    for v_cell in (cell_warning_v, cell_normal_trip_v, 2.45):
        i_bleed = v_cell / 120.0
        p_bleed = v_cell**2 / 120.0
        bleed_rows.append(
            {
                "cell_voltage_V": v_cell,
                "current_per_cell_mA": round(i_bleed * 1000.0, 3),
                "power_per_cell_mW": round(p_bleed * 1000.0, 3),
                "power_all_8_cells_W": round(p_bleed * n_cells, 4),
            }
        )

    scan_sample_time_s = adc_max_sample_cycles / adc_clock_hz
    scan_conversion_time_s = adc_12bit_conversion_cycles / adc_clock_hz
    max_adc_input_unpowered_v = tap_adc_v_at_hard[-1]
    unpowered_injection_t8_a = max(
        0.0, (max_adc_input_unpowered_v - 0.3) / rth_t8
    )
    worst_unpowered_injection_sum_a = sum(
        max(0.0, (v - 0.3) / r) for v, r in zip(tap_adc_v_at_hard, tap_rth_ohm)
    )

    energy_headroom = {
        "280F_screen_8S_from_2p2_to_2p45_J": round(
            0.5 * cell_cap_eol_screen_f * n_cells * (2.45**2 - 2.2**2), 1
        ),
        "280F_screen_8S_from_2p3_to_2p45_J": round(
            0.5 * cell_cap_eol_screen_f * n_cells * (2.45**2 - 2.3**2), 1
        ),
        "280F_screen_8S_from_2p3_to_2p7_J_theoretical": round(
            0.5 * cell_cap_eol_screen_f * n_cells * (2.7**2 - 2.3**2), 1
        ),
        "usable_energy_warning": "Theoretical capacitor energy only; mismatch, ESR, converter power and thermal limits are not included.",
    }

    result = {
        "document": "Existing eight cumulative-tap low-cost cell-monitoring screening",
        "date": "2026-10-29",
        "status": "PAPER_SCREENING_ONLY_NOT_SCHEMATIC_OR_BOM_CHANGE",
        "native_or_procurement_bom_modified": False,
        "preferred_path": "Keep the existing eight independent 68k/10k cumulative dividers and G431 ADCs if the power-sequencing contract guarantees the MCU remains powered until SC isolation is complete. Accuracy alone does not require a new monitor IC at 2.2-2.3 V/cell trip settings.",
        "schematic_binding": {
            "source": "检查/当前原理图证据.json",
            "tap_nets": "T1..T8 cumulative taps; AD_T1..AD_T8 connect to G431 ADC pins",
            "components": "R_T1..R_T8_HI=68k ±0.1%; R_T1..R_T8_LO=10k ±0.1%; T1..T7 each also have 47k to TEST/PB10; C_T1..C_T8_FILTER=1nF C0G",
            "calculation_assumption": "PB10/TEST is held low during ADC sampling, so T1..T7 bottom leg is 10k || 47k; T8 remains 10k.",
            "test_state_requirement": "The 47k TEST branch changes divider ratio. Firmware must hold TEST in the stated fixed state and use that actual ratio, or calibrate the actual pin network. TEST high can bias zero-voltage readings upward; an uncontrolled TEST state invalidates this calibration.",
        },
        "inputs": {
            "cells": n_cells,
            "cell_warning_V": cell_warning_v,
            "software_trip_candidate_V": cell_normal_trip_v,
            "hard_absolute_cell_limit_V": cell_hard_limit_v,
            "cell_capacitance_nominal_F": cell_cap_nominal_f,
            "cell_capacitance_EOL_screen_F": cell_cap_eol_screen_f,
            "charge_peak_budget_A": charge_peak_a,
            "sample_failure_tail_s": failed_sample_window_s,
            "SC_MINUS_to_GND_SYS_shunt_ohm": shunt_ohm,
            "bus_min_V": bus_min_v,
            "existing_aux_converter_input_min_V": aux_converter_min_v,
            "ADC_VREF_V_assumed": adc_vref_v,
            "resistor_tolerance": resistor_tolerance,
            "resistor_TCR_ppm_per_C": resistor_tcr_ppm_c,
            "ambient_C": [ambient_min_c, ambient_max_c],
            "calibration_temperature_C": ambient_cal_c,
        },
        "network": {
            "T1_to_T7_effective_lower_ohm_TEST_low": round(r_lo_test_low, 3),
            "T1_to_T7_ratio_TEST_low": round(ratio_test_low, 9),
            "T8_ratio": round(ratio_t8, 9),
            "T1_to_T7_source_resistance_ohm": round(rth_test_low, 3),
            "T8_source_resistance_ohm": round(rth_t8, 3),
            "T1_to_T7_ADC_V_at_2p7V_per_cell": round(tap_adc_v_at_hard[6], 6),
            "T8_ADC_V_at_2p7V_per_cell": round(tap_adc_v_at_hard[7], 6),
            "all_taps_zero_cell_stack_ADC_V": 0.0,
            "T8_input_valid_with_G431_VDDA_3p3V": tap_adc_v_at_hard[7] < adc_vref_v,
            "ADC_source_resistance_limit_ohm_12bit_640p5_cycles_slow": adc_slow_source_max_ohm,
            "ADC_source_resistance_within_limit": max(tap_rth_ohm) < adc_slow_source_max_ohm,
            "1nF_filter_tau_max_us": round(max(tap_rth_ohm) * 1e-9 * 1e6, 3),
            "5tau_settle_max_us": round(max(tap_rth_ohm) * 1e-9 * 5e6, 3),
            "640p5_cycle_sample_time_us_at_60MHz": round(scan_sample_time_s * 1e6, 3),
            "eight_sequential_sample_time_us_at_60MHz": round(scan_sample_time_s * 8e6, 3),
            "12bit_total_conversion_time_us_at_60MHz": round(scan_conversion_time_s * 1e6, 3),
            "eight_sequential_12bit_conversion_time_us_at_60MHz": round(
                scan_conversion_time_s * 8e6, 3
            ),
            "eight_conversion_time_fraction_of_145p4ms_window": round(
                scan_conversion_time_s * 8.0 / failed_sample_window_s, 8
            ),
        },
        "errors": {
            "ADC_ET_max_single_ended_slow_channel_LSB": adc_et_max_lsb,
            "ADC_INL_max_single_ended_LSB": adc_inl_max_lsb,
            "ADC_LSB_at_3p3V_mV": round(adc_lsb_v * 1000.0, 5),
            "ADC_error_table_scope": "STM32G431 datasheet Table 64: after internal ADC calibration, characterized/not production-tested; no oversampling. Pair budgets sum worst signed endpoints and do not assume correlated error cancellation.",
            "application_calibration": "Calibrate ADC1 and ADC2 gain/offset separately; use per-die VREFINT factory value to track common VREF; calibrate or characterize each cumulative divider at a known temperature. The conservative raw screening still retains full datasheet ADC ET; the calibrated component screen retains channel INL, quantization, resistor TCR, VREFINT temperature drift, shunt-reference and sample age.",
            "vrefint_residual_fraction_after_factory_calibration_screen": round(vref_residual_fraction, 8),
            "vrefint_residual_basis": "50 ppm/C max average coefficient from the 30 C factory calibration point to the 0..60 C ambient envelope, using 35 C max delta; no claim about unspecified board thermal gradients.",
            "oversampling_credit_for_INL": 0.0,
            "ground_offset_bound_V": round(shunt_drop_v, 6),
            "ground_offset_boundary": "25 mV for one-polarity 0..50 A or the first cell; common offset cancels for adjacent taps only when current is effectively unchanged between their samples. A charge/discharge sign reversal between paired reads could double this term to 50 mV and requires a current-synchronous bound.",
            "sample_age_rise_V_at_50A_280F": round(age_rise_v, 8),
            "worst_uncalibrated_cell": worst_raw["cell"],
            "worst_uncalibrated_2p3_threshold_actual_plus_tail_V": worst_raw[
                "uncalibrated_25C_actual_cell_v_at_2p30_trip_bound"
            ],
            "worst_calibrated_components_cell": worst_cal["cell"],
            "worst_calibrated_components_2p3_threshold_actual_plus_tail_V": worst_cal[
                "calibrated_components_actual_cell_v_at_2p30_trip_bound"
            ],
            "per_cell": cell_rows,
            "interpretation": "Under the stated stable TEST state and 0..60 C screening assumptions, a 2.30 V measured software trip plus 145.4 ms at 50 A remains below 2.70 V even using the full characterized ADC ET and independent worst-case divider errors. The old ~213.8/339.2 mV bounds are conservative subtraction budgets, not proof that the chain cannot be used; the 125 C case is outside the 0..60 C ambient requirement and is not used as the operating target.",
        },
        "supply_and_power": {
            "BUS_min_minus_aux_converter_min_V": round(bus_min_v - aux_converter_min_v, 3),
            "SC_total_at_2p2_V": round(n_cells * 2.2, 3),
            "SC_total_at_2p3_V": round(n_cells * 2.3, 3),
            "BUS_headroom_at_2p2_V": round(bus_min_v - n_cells * 2.2, 3),
            "BUS_headroom_at_2p3_V": round(bus_min_v - n_cells * 2.3, 3),
            "BUS_headroom_at_2p7_V": round(bus_min_v - n_cells * 2.7, 3),
            "buck_duty_screen_at_2p2": round(n_cells * 2.2 / bus_min_v, 6),
            "buck_duty_screen_at_2p3": round(n_cells * 2.3 / bus_min_v, 6),
            "buck_duty_screen_at_2p7": round(n_cells * 2.7 / bus_min_v, 6),
            "power_stage_conclusion": "Static voltage headroom supports charging to 2.2-2.3 V/cell at the 22.5 V BUS minimum. The 2.7 V/cell absolute limit leaves only 0.9 V from BUS to SC and is not a usable commanded charge endpoint. Voltage headroom is not a qualification of 50 A thermal/current capability; 50 A is retained only as the requested safety-rise budget.",
            "startup_to_2p3_ideal_at_20A_s": round(
                (cell_cap_eol_screen_f / n_cells) * (n_cells * 2.3) / 20.0, 3
            ),
            "energy_headroom": energy_headroom,
        },
        "balancing": {
            "method": "Fixed 120 ohm resistor directly across each of eight cells; reuse existing resistor instances if rewire is allowed.",
            "per_cell": bleed_rows,
            "tau_280F_h": round(tau_280_s / 3600.0, 3),
            "tau_350F_h": round(tau_350_s / 3600.0, 3),
            "280F_time_from_2p3V_to_1V_h_ideal_RC": round(
                tau_280_s / 3600.0 * math.log(2.3), 3
            ),
            "350F_time_from_2p3V_to_1V_h_ideal_RC": round(
                tau_350_s / 3600.0 * math.log(2.3), 3
            ),
            "conclusion": "Slow passive equalization only. It continuously drains every cell, dissipates about 0.35 W at 2.30 V/cell and 0.40 W at 2.45 V/cell, cannot balance main charge current, and does not diagnose an open/short bleed path.",
        },
        "power_off_input": {
            "status": "FAIL_IF_G431_VDD_OR_VDDA_IS_ZERO_WHILE_SC_TAPS_REMAIN_LIVE",
            "ADC_pin_types": "The current tap pins PA0..PA6 and PB14 are TT_a analog-capable pins, not powered-off-tolerant analog inputs.",
            "datasheet_limit": "STM32G431 Table 17: TT_x input voltage is VSS-0.3 to VDD+0.3 V; Table 15 lists positive current injection as 0 mA for TT_x pins.",
            "T8_live_at_2p7_stack_V": round(max_adc_input_unpowered_v, 6),
            "T8_thevenin_ohm": round(rth_t8, 3),
            "T8_estimated_current_if_clamped_at_0p3V_mA": round(unpowered_injection_t8_a * 1000.0, 3),
            "sum_of_eight_estimated_pin_currents_if_each_clamps_at_0p3V_mA": round(
                worst_unpowered_injection_sum_a * 1000.0, 3
            ),
            "interpretation": "The divider is valid from a 0 V stack while G431 analog supply is valid. It is not legal to call the same inputs safe when G431 supply collapses while SC remains charged: T8 alone can exceed the powered-off input limit and inject about 0.28 mA under a simple clamp estimate. Do not rely on this estimated current as a permitted datasheet condition.",
            "least_change_resolution": "Zero new parts only if the existing 5V_EXT/3V3_CORE power sequence is guaranteed to keep G431 and the external scan watchdog alive until the physical SC isolation has completed. If that contract cannot be guaranteed, the minimum integrated isolation candidate to evaluate is one TMUX7308F (powered-off/fault-protected Sx inputs) plus a 20 V rail from BUS using a high-voltage LDO such as TPS7A4001; 22.5 V BUS gives 2.5 V headroom and its normal-mode condition is VIN >= VOUT + 2 V. This is a conditional fallback, not added to the design; at the 2.2-2.3 V trip window the 18.4 V maximum normal stack is below a 20 V mux rail. It does not create an independent per-cell hardware comparator.",
        },
        "protection_boundary": {
            "retained_system_protection": [
                "Independent total SC/BUS overvoltage comparator",
                "Independent current comparators",
                "Physical SC power isolation",
                "External scan watchdog, permission and fault latch",
            ],
            "sampling_failure": "A watchdog serviced only after a complete, fresh, valid eight-tap scan can revoke isolation permission within the defined 145.4 ms stale/missed-scan window.",
            "not_provided": "Without independent per-cell hardware comparators, this is not an independent analog hard-OV guarantee at 2.70 V/cell. A plausible but wrong ADC/MUX value that still passes scan checks and refreshes the heartbeat is outside the specific stale/missed-scan shutdown function. Total voltage and current comparators do not identify a single high cell.",
            "screening_only": "No formal safety qualification, firmware implementation, schematic edit, BOM edit, device characterization, or hardware test is claimed.",
        },
        "official_sources": [
            {
                "name": "STM32G431 datasheet DS12589 Rev 6",
                "url": "https://www.st.com/resource/en/datasheet/stm32g431cb.pdf",
                "used_for": "TT_a input limits, injection, ADC ET/INL, VREFINT, source-resistance and sample-time limits.",
            },
            {
                "name": "TMUX7308F datasheet",
                "url": "https://www.ti.com/lit/ds/symlink/tmux7308f.pdf",
                "used_for": "Conditional single-chip mux power-off/fault protection fallback only.",
            },
            {
                "name": "TPS7A4001 datasheet",
                "url": "https://www.ti.com/lit/ds/symlink/tps7a4001.pdf",
                "used_for": "Conditional 20 V mux rail dropout/input range fallback only.",
            },
            {
                "name": "YAGEO RT resistor series data",
                "url": "https://www.yageogroup.com/component-documentation/download/specsheet/RT0603BRD0768KL",
                "used_for": "68 kOhm resistor ±0.1%, ±25 ppm/C; project evidence identifies this exact high-side MPN.",
            },
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        "worst 2.30 V trip screen: "
        f"raw={worst_raw['uncalibrated_25C_actual_cell_v_at_2p30_trip_bound']:.3f} V, "
        "calibrated-components="
        f"{worst_cal['calibrated_components_actual_cell_v_at_2p30_trip_bound']:.3f} V"
    )


if __name__ == "__main__":
    main()
