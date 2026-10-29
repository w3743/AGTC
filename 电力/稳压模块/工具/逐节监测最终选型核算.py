"""Paper-final calculation for an 8S supercapacitor cell monitor.

This script writes only 检查/逐节监测最终选型核算.json. It does not edit the
native schematic, procurement BOM, or either design document.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "检查" / "逐节监测最终选型核算.json"


def parallel(*values: float) -> float:
    return 1.0 / sum(1.0 / value for value in values)


def tap_gain(r_hi: float, r_lo: float, r_pd: float, r_on: float) -> float:
    """Gain from cumulative stack tap to MCU D/ADC bus."""
    r_th = parallel(r_hi, r_lo)
    divider_gain = r_lo / (r_hi + r_lo)
    return divider_gain * r_pd / (r_pd + r_th + r_on)


def gain_extrema() -> tuple[float, float, float]:
    r_hi, r_lo, r_pd, tol = 68_000.0, 10_000.0, 10_000.0, 0.001
    nominal = tap_gain(r_hi, r_lo, r_pd, 4.5)
    values = []
    for e_hi, e_lo, e_pd in itertools.product((-1, 1), repeat=3):
        for r_on in (0.0, 4.5):
            values.append(
                tap_gain(
                    r_hi * (1 + e_hi * tol),
                    r_lo * (1 + e_lo * tol),
                    r_pd * (1 + e_pd * tol),
                    r_on,
                )
            )
    return nominal, min(values) / nominal - 1.0, max(values) / nominal - 1.0


def tcr_extrema() -> tuple[float, float]:
    """Worst gain drift from 25 C to 0..60 C using ±25 ppm/C for all Rs."""
    r_hi, r_lo, r_pd, r_on = 68_000.0, 10_000.0, 10_000.0, 4.5
    base = tap_gain(r_hi, r_lo, r_pd, r_on)
    delta = 25e-6 * 35.0
    values = []
    for e_hi, e_lo, e_pd in itertools.product((-1, 1), repeat=3):
        values.append(
            tap_gain(
                r_hi * (1 + e_hi * delta),
                r_lo * (1 + e_lo * delta),
                r_pd * (1 + e_pd * delta),
                r_on,
            )
            / base
            - 1.0
        )
    return min(values), max(values)


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Small Gaussian elimination solver (8x8 maximum here)."""
    n = len(rhs)
    work = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        if abs(scale) < 1e-30:
            raise ArithmeticError("singular matrix")
        for j in range(col, n + 1):
            work[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = work[row][col]
            for j in range(col, n + 1):
                work[row][j] -= factor * work[col][j]
    return [work[i][n] for i in range(n)]


def node_conductance(
    r_balance: list[float], r_div: float, cell_leak_conductance: list[float]
) -> list[list[float]]:
    """Nodal resistor matrix: cell bleeders/leakage plus cumulative tap dividers."""
    n = len(r_balance)
    g = [[0.0] * n for _ in range(n)]
    for cell in range(n):
        conductance = 1.0 / r_balance[cell] + cell_leak_conductance[cell]
        node = cell  # node0 is cell1's upper node; stack bottom is ground.
        g[node][node] += conductance
        if cell > 0:
            lower = cell - 1
            g[lower][lower] += conductance
            g[node][lower] -= conductance
            g[lower][node] -= conductance
    for tap_node in range(n):
        g[tap_node][tap_node] += 1.0 / r_div
    return g


def capacitance_matrix(cell_capacitances: list[float]) -> list[list[float]]:
    """Nodal C matrix for capacitors between adjacent stack nodes."""
    n = len(cell_capacitances)
    cmat = [[0.0] * n for _ in range(n)]
    for cell, cap in enumerate(cell_capacitances):
        upper_node = cell
        cmat[upper_node][upper_node] += cap
        if cell > 0:
            lower_node = cell - 1
            cmat[lower_node][lower_node] += cap
            cmat[upper_node][lower_node] -= cap
            cmat[lower_node][upper_node] -= cap
    return cmat


def inverse_times_matrix(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    n = len(left)
    result = [[0.0] * n for _ in range(n)]
    for col in range(n):
        column = solve_linear(left, [right[row][col] for row in range(n)])
        for row in range(n):
            result[row][col] = column[row]
    return result


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def cell_voltages(node_voltages: list[float]) -> list[float]:
    values = []
    previous = 0.0
    for node_voltage in node_voltages:
        values.append(node_voltage - previous)
        previous = node_voltage
    return values


def parking_simulation(
    capacitances_f: list[float],
    balance_resistors_ohm: list[float],
    leak_conductance_s: list[float],
    initial_cell_v: float = 2.2,
    dt_s: float = 30.0,
    days: int = 8,
) -> dict:
    """RK4 solve of M*d(nodeV)/dt=-G*nodeV during parked discharge."""
    n = len(capacitances_f)
    cmat = capacitance_matrix(capacitances_f)
    gmat = node_conductance(balance_resistors_ohm, 78_000.0, leak_conductance_s)
    a = inverse_times_matrix(cmat, gmat)
    x = [initial_cell_v * (i + 1) for i in range(n)]
    day_marks = {1, 2, 3, 5, 7, days}
    at_day: dict[str, list[float]] = {}
    min_v = math.inf
    min_time = 0.0
    total_s = days * 86_400.0
    steps = int(total_s / dt_s)

    def deriv(state: list[float]) -> list[float]:
        return [-value for value in matvec(a, state)]

    for step in range(steps):
        t0 = step * dt_s
        k1 = deriv(x)
        k2 = deriv([v + dt_s * k / 2.0 for v, k in zip(x, k1)])
        k3 = deriv([v + dt_s * k / 2.0 for v, k in zip(x, k2)])
        k4 = deriv([v + dt_s * k for v, k in zip(x, k3)])
        x = [
            v + dt_s * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
            for v, a1, a2, a3, a4 in zip(x, k1, k2, k3, k4)
        ]
        vcell = cell_voltages(x)
        current_min = min(vcell)
        if current_min < min_v:
            min_v = current_min
            min_time = t0 + dt_s
        day = int((t0 + dt_s) / 86_400.0)
        if day in day_marks and str(day) not in at_day:
            at_day[str(day)] = vcell

    return {
        "initial_cell_voltage_V": initial_cell_v,
        "capacitances_F": capacitances_f,
        "balance_resistors_ohm": balance_resistors_ohm,
        "leak_model": "each cell leakage as parallel conductance; 0.3mA at 2.7V => 111.1uS, zero leakage at 0V; sensitivity model, not a vendor I-V guarantee",
        "cell_voltages_by_park_day_V": at_day,
        "unclamped_linear_minimum_V": min_v,
        "unclamped_linear_minimum_day": min_time / 86_400.0,
        "physical_reading": "No material reverse voltage in the modeled 0/+20% capacity, bounded-leakage cases. A few microvolts of negative linear tail occurs only after the entire stack is already at microvolt levels and is below model resolution; it is not evidence of a physical reverse bias. LICAP does not specify a reverse-voltage operating model, so do not extrapolate the linear model below zero.",
    }


def steady_state_boundary() -> dict:
    """Max spread with direct bleeders at the low-current DC equilibrium."""
    n = 8
    r_nominal = 120.0
    tolerance = 0.01
    max_leak_g = 0.0003 / 2.7
    best = None
    # Enumerate ±1% resistor corners and 0/max leakage corners per cell.
    for rmask in range(1 << n):
        resistors = [
            r_nominal * (1.01 if (rmask >> i) & 1 else 0.99) for i in range(n)
        ]
        for lmask in range(1 << n):
            leaks = [max_leak_g if (lmask >> i) & 1 else 0.0 for i in range(n)]
            gmat = node_conductance(resistors, 78_000.0, leaks)
            unit_current = [0.0] * n
            unit_current[-1] = 1.0
            node_v_per_amp = solve_linear(gmat, unit_current)
            cell_v_per_amp = cell_voltages(node_v_per_amp)
            max_per_amp = max(cell_v_per_amp)
            current_a = 2.15 / max_per_amp
            cell_v = [value * current_a for value in cell_v_per_amp]
            spread = max(cell_v) - min(cell_v)
            if best is None or spread > best["spread_V"]:
                best = {
                    "spread_V": spread,
                    "current_A": current_a,
                    "cell_voltages_V": cell_v,
                    "r_balance_ohm": resistors,
                    "cell_leak_conductance_S": leaks,
                }
    assert best is not None
    return {
        "definition": "steady DC charge current through stack; each cell's capacitive current is zero; node KCL includes each direct 120R shunt and all Tn-to-GND 78k tap branches",
        "largest_cell_set_to_V": 2.15,
        "worst_corner": best,
        "interpretation": "A 120R direct shunt only produces a normal-window DC equilibrium when stack charge current is about 18mA, not at a 20A-class ramp. At the low-current equilibrium, resistor tolerance, cumulative-divider return currents and leakage can leave about 157mV cell spread. Capacity affects the time to equilibrium, not its DC voltage; it can make transient spread worse.",
    }


def build_report() -> dict:
    cells = 8
    r_hi = 68_000.0
    r_lo = 10_000.0
    r_pd = 10_000.0
    r_on = 4.5
    r_th = parallel(r_hi, r_lo)
    nominal_gain, gain_min_error, gain_max_error = gain_extrema()
    tcr_min_error, tcr_max_error = tcr_extrema()
    lsb_v = 3.3 / 4096.0
    cell_equiv_lsb_v = lsb_v / nominal_gain
    et_lsb = 7.5
    adc_pair_error_v = 2.0 * et_lsb * cell_equiv_lsb_v
    adc_cal_pair_v = 2.0 * (4.4 + 0.5) * cell_equiv_lsb_v
    adc_clock_hz = 60e6
    adc_sample_cycles = 640.5
    adc_conversion_cycles = 653.0
    adc_sample_s = adc_sample_cycles / adc_clock_hz
    adc_conversion_s = adc_conversion_cycles / adc_clock_hz
    settle_s = 100e-6
    output_tau_s = (parallel(r_pd, r_th + r_on)) * 1e-9
    slot_s = settle_s + adc_conversion_s
    four_pair_scan_s = 4.0 * slot_s

    # IWDG PR=/4, RLR=700; STM32G431 LSI min/max are 29.5/34 kHz.
    iwdg_r_load = 700
    iwdg_prescaler = 4
    lsi_min_hz = 29_500.0
    lsi_max_hz = 34_000.0
    iwdg_max_s = (iwdg_r_load + 1) * iwdg_prescaler / lsi_min_hz
    iwdg_min_s = (iwdg_r_load + 1) * iwdg_prescaler / lsi_max_hz
    max_cell_sample_age_s = iwdg_max_s + four_pair_scan_s
    # Screening point requested by the parent: 80 A at maximum stale-sample age.
    # 280 F is an extra-conservative sensitivity below the LICAP 350 F minimum.
    charge_current_a = 80.0
    cell_capacity_screen_f = 280.0
    age_rise_v = charge_current_a * max_cell_sample_age_s / cell_capacity_screen_f
    age_rise_50a_v = 50.0 * max_cell_sample_age_s / cell_capacity_screen_f
    age_rise_80a_350f_v = 80.0 * max_cell_sample_age_s / 350.0
    ground_error_v = charge_current_a * 0.0005
    # Use the ST table's 7.5 mV max temperature spread relative to the
    # individually factory-calibrated VREFINT (1.212 V typ), plus its 1000 ppm
    # max long-term stability and 1200 ppm/V supply coefficient over 0.3 V.
    vref_error_fraction = 7.5e-3 / 1.212 + 1000e-6 + 1200e-6 * 0.3

    # Conservative raw budget includes uncalibrated divider tolerance and full ADC ET.
    # The adjacent cumulative-tap pair error scales as (2*cell-1)*Vcell.
    # VREF error is common and therefore contributes one-cell voltage, not 8+7 cells.
    threshold_rows = []
    for threshold in (2.00, 2.15, 2.20):
        i = 8
        cumulative_multiplier = 2 * i - 1
        div_tol_v = cumulative_multiplier * threshold * max(abs(gain_min_error), abs(gain_max_error))
        div_tcr_v = cumulative_multiplier * threshold * max(abs(tcr_min_error), abs(tcr_max_error))
        vref_v = threshold * vref_error_fraction
        raw_error_v = adc_pair_error_v + div_tol_v + div_tcr_v + vref_v + ground_error_v + age_rise_v
        threshold_rows.append(
            {
                "measured_cell_threshold_V": threshold,
                "adc_ET_pair_mV": adc_pair_error_v * 1000.0,
                "divider_tolerance_pair_mV": div_tol_v * 1000.0,
                "divider_TCR_pair_mV_after_25C_cal": div_tcr_v * 1000.0,
                "VREFINT_residual_mV": vref_v * 1000.0,
                "SC_MINUS_GND_shunt_mV": ground_error_v * 1000.0,
                "maximum_sample_age_rise_mV": age_rise_v * 1000.0,
                "raw_total_error_mV": raw_error_v * 1000.0,
                "raw_actual_upper_bound_V": threshold + raw_error_v,
                "margin_to_2p70V_mV": (2.70 - threshold - raw_error_v) * 1000.0,
                "50A_280F_age_rise_mV": age_rise_50a_v * 1000.0,
                "50A_280F_upper_bound_V_same_other_errors": threshold + raw_error_v - age_rise_v + age_rise_50a_v,
                "80A_350F_age_rise_mV_spec_min_cap": age_rise_80a_350f_v * 1000.0,
                "80A_350F_upper_bound_V_same_other_errors": threshold + raw_error_v - age_rise_v + age_rise_80a_350f_v,
            }
        )

    # Calibrated budget: per-tap 25C gain/offset calibration removes initial resistor tolerance.
    # Keep full G431 INL + quantization, resistor TCR, VREFINT, ground, age, and
    # a full 4.5-ohm cross-device RON-drift bound at the T4/T5 bank boundary.
    ron_drift_cell8_v = (15 * 2.40 * 0.28 / (r_pd + r_th + r_on))
    ron_drift_cross_bank_v = (9 * 2.40 * 4.5 / (r_pd + r_th + r_on))
    ron_drift_v = max(ron_drift_cell8_v, ron_drift_cross_bank_v)
    cal_threshold_rows = []
    for threshold in (2.00, 2.15, 2.20):
        cumulative_multiplier = 15
        div_tcr_v = cumulative_multiplier * threshold * max(abs(tcr_min_error), abs(tcr_max_error))
        vref_v = threshold * vref_error_fraction
        ron_v = ron_drift_v * threshold / 2.40
        cal_error_v = adc_cal_pair_v + div_tcr_v + vref_v + ground_error_v + age_rise_v + ron_v
        cal_threshold_rows.append(
            {
                "measured_cell_threshold_V": threshold,
                "ADC_INL_plus_quant_pair_mV": adc_cal_pair_v * 1000.0,
                "divider_TCR_pair_mV_after_per_tap_25C_cal": div_tcr_v * 1000.0,
                "VREFINT_residual_mV": vref_v * 1000.0,
                "MUX_RON_temperature_mismatch_bound_mV": ron_v * 1000.0,
                "SC_MINUS_GND_shunt_mV": ground_error_v * 1000.0,
                "maximum_sample_age_rise_mV": age_rise_v * 1000.0,
                "calibrated_total_error_mV": cal_error_v * 1000.0,
                "calibrated_actual_upper_bound_V": threshold + cal_error_v,
                "margin_to_2p70V_mV": (2.70 - threshold - cal_error_v) * 1000.0,
            }
        )

    # Bus divider: source-side 300k/10k node loaded through mux by host-side 10k.
    bus_high = 300_000.0
    bus_low = 10_000.0
    bus_pd = 10_000.0
    bus_rth = parallel(bus_high, bus_low)
    bus_ratio_old = (bus_high + bus_low) / bus_low
    bus_ratio_loaded = bus_ratio_old * (bus_pd + bus_rth + r_on) / bus_pd
    bus_source_adc_80_v = 80.0 / bus_ratio_old
    bus_d_adc_80_v = 80.0 / bus_ratio_loaded
    bus_lsb_mV = lsb_v * bus_ratio_loaded * 1000.0
    imon_gain = 10_000.0 / (10_000.0 + 22.0 + r_on)

    tap_rows = []
    for tap in range(1, cells + 1):
        tap_source_at_27 = tap * 2.7 * r_lo / (r_hi + r_lo)
        adc_at_27 = tap * 2.7 * nominal_gain
        tap_rows.append(
            {
                "tap": tap,
                "source_AD_Tn_at_2p7V_per_cell_V": tap_source_at_27,
                "bank_ADC_D_at_2p7V_per_cell_V": adc_at_27,
                "safe_with_TMUV1511_power_off_3p6V_limit": tap_source_at_27 < 3.6,
            }
        )

    divider_load_rows = []
    for voltage in (2.00, 2.15, 2.20, 2.70):
        cell_currents = []
        for cell in range(1, cells + 1):
            current = sum(tap * voltage / (r_hi + r_lo) for tap in range(cell, cells + 1))
            cell_currents.append(current)
        direct_bleed_current = voltage / 120.0
        direct_bleed_power = voltage * voltage / 120.0
        divider_total_power = sum((tap * voltage) ** 2 / (r_hi + r_lo) for tap in range(1, cells + 1))
        active_t8_current = 8 * voltage / (r_hi + parallel(r_lo, r_pd + r_on))
        base_t8_current = 8 * voltage / (r_hi + r_lo)
        divider_load_rows.append(
            {
                "cell_voltage_V": voltage,
                "tap_only_cell_currents_bottom_to_top_mA": [value * 1000.0 for value in cell_currents],
                "tap_only_stack_power_mW": divider_total_power * 1000.0,
                "direct_120R_current_per_cell_mA": direct_bleed_current * 1000.0,
                "direct_120R_power_per_cell_mW": direct_bleed_power * 1000.0,
                "direct_120R_power_8_cells_W": direct_bleed_power * 8.0,
                "bottom_total_bleed_plus_tap_mA": (direct_bleed_current + cell_currents[0]) * 1000.0,
                "top_total_bleed_plus_tap_mA": (direct_bleed_current + cell_currents[-1]) * 1000.0,
                "T8_added_current_while_selected_uA": (active_t8_current - base_t8_current) * 1e6,
            }
        )

    rdiv = r_hi + r_lo
    tau_balance_rows = []
    for cap in (280.0, 350.0, 420.0):
        tau_h = 120.0 * cap / 3600.0
        tau_balance_rows.append(
            {
                "capacitance_F": cap,
                "120R_only_RC_tau_h": tau_h,
                "voltage_after_8h_from_2p2V_ideal_V": 2.2 * math.exp(-8.0 / tau_h),
                "voltage_after_24h_from_2p2V_ideal_V": 2.2 * math.exp(-24.0 / tau_h),
            }
        )

    # Parking matrix cases, with worst low-cap bottom cell and high-cap upper cells.
    max_leak_g = 0.0003 / 2.7
    cap_corner = [350.0] + [420.0] * 7
    r_corner = [118.8] + [121.2] * 7
    leak_corner = [max_leak_g] + [0.0] * 7
    parking_direct = parking_simulation(cap_corner, r_corner, leak_corner)
    parking_divider_only = parking_simulation(
        cap_corner,
        [1e30] * 8,
        leak_corner,
    )

    # Steady-state boundary enumerates direct R_BAL tolerances and leakage corners.
    balance_boundary = steady_state_boundary()

    return {
        "schema": "sc8_cell_monitor_final_selection/v1",
        "date": "2026-10-29",
        "status": "PAPER_FINAL_SELECTION_NOT_NATIVE_BOM_OR_HARDWARE_VALIDATION",
        "scope": "Independent 8S supercapacitor 0V-capable cumulative tap measurement, ADC routing, thresholds, passive 120R discharge and parking RC screening.",
        "native_schematic_modified": False,
        "procurement_bom_modified": False,
        "design_documents_modified": False,
        "decision": {
            "cell_measurement": "Keep the existing eight cumulative 68k/10k dividers. Add two four-channel TMUX1511 banks; within each package tie D1..D4 to one independent ADC bus and use the four SPST channels as one-hot tap selectors. This is not an addressed 8:1 mux; it uses one GPIO per channel.",
            "signal_isolation": "Use a third TMUX1511 for BUS_ADC and DRV_SO/IMON powered-off isolation. Its BUS and IMON channels are held on while 3V3_CORE is valid; when the mux supply is off, both MCU-side 10k pulldowns hold the ADC/comparator nodes low.",
            "MAX17853_MAX17877": "WITHDRAWN_NOT_ADOPTED",
            "cell_charge_control": "Use 2.00V measured maximum-cell threshold as the charge/standby target, 2.15V measured as warning/derate, and 2.20V measured as regenerative-charge stop. Safety bound uses uncalibrated divider tolerance, full STM32G431 ADC ET, VREFINT drift/aging/supply bounds, 80A stale-sample rise, and SC_MINUS ground return; per-tap route calibration remains required for useful telemetry but is not credited to the safety threshold.",
            "passive_120R": "Retain R_BAL1..R_BAL8 as permanently connected cell shunts, treated as slow discharge/voltage-spread decay, not an active or capacity-tolerance balancer. Their continuous drain and limited ability at high charge current are explicit operating boundaries.",
        },
        "evidence_reviewed": [
            "../AGENTS.md",
            "设计需求.md (opening sections)",
            "设计细节.md (opening sections; prior MAX17853/77 adoption is withdrawn)",
            "原理图准备/BOM.csv",
            "检查/当前原理图证据.json",
            "检查/低成本逐节监测核算.json (historical direct-ADC/test-branch baseline; not adopted)",
            "检查/初充与掉电接口核算.json (historical candidate only; not adopted as a whole)",
            "参考资料/TMUX1511_Official.pdf",
            "检查/未通过项修复/BUSFIX_逐节保护资格复查.json",
        ],
        "cell_mux": {
            "device": {
                "manufacturer": "Texas Instruments",
                "mpn": "TMUX1511PWR",
                "package": "TSSOP-14",
                "supplier": "LCSC",
                "supplier_part": "C2866750",
                "vdd": "3V3_CORE",
                "gnd": "GND_SYS",
                "local_bypass": "100nF from VDD to GND per device; reuse C_BAL refs listed below.",
                "datasheet_limits": {
                    "supply_V": [1.5, 5.5],
                    "powered_signal_max_V": 5.5,
                    "powered_off_signal_V": [0.0, 3.6],
                    "powered_off_leakage_max_uA_per_io": 2.0,
                    "ron_max_ohm": 4.5,
                    "logic_high_min_V": 1.2,
                    "logic_low_max_V": 0.45,
                    "internal_sel_pulldown_ohm": 6_000_000,
                },
            },
            "U_CELL_MUX1": {
                "channels": [
                    {"S_pin": 2, "S_net": "AD_T1", "D_pin": 3, "D_net": "SC_ADC_A", "SEL_pin": 1, "GPIO": "PC6 / P1-9"},
                    {"S_pin": 5, "S_net": "AD_T2", "D_pin": 6, "D_net": "SC_ADC_A", "SEL_pin": 4, "GPIO": "PA15 / P1-15"},
                    {"S_pin": 9, "S_net": "AD_T3", "D_pin": 8, "D_net": "SC_ADC_A", "SEL_pin": 10, "GPIO": "PC10 / P1-16"},
                    {"S_pin": 12, "S_net": "AD_T4", "D_pin": 11, "D_net": "SC_ADC_A", "SEL_pin": 13, "GPIO": "PC11 / P1-17"},
                ],
                "D_tied_pins": [3, 6, 8, 11],
                "adc": "PB14 / ADC1_IN5",
                "pulldown_ref": "R_ADC_A_PD",
                "filter_cap_ref": "C_T1_FILTER moved to SC_ADC_A-GND_SYS",
            },
            "U_CELL_MUX2": {
                "channels": [
                    {"S_pin": 2, "S_net": "AD_T5", "D_pin": 3, "D_net": "SC_ADC_B", "SEL_pin": 1, "GPIO": "PB3 / P1-18"},
                    {"S_pin": 5, "S_net": "AD_T6", "D_pin": 6, "D_net": "SC_ADC_B", "SEL_pin": 4, "GPIO": "PB4 / P1-19"},
                    {"S_pin": 9, "S_net": "AD_T7", "D_pin": 8, "D_net": "SC_ADC_B", "SEL_pin": 10, "GPIO": "PB5 / P1-20 (repurpose old PERMIT GPIO)"},
                    {"S_pin": 12, "S_net": "AD_T8", "D_pin": 11, "D_net": "SC_ADC_B", "SEL_pin": 13, "GPIO": "PB9 / P1-24"},
                ],
                "D_tied_pins": [3, 6, 8, 11],
                "adc": "PB15 / ADC2_IN15",
                "pulldown_ref": "R_ADC_B_PD",
                "filter_cap_ref": "C_T2_FILTER moved to SC_ADC_B-GND_SYS",
            },
            "selection_contract": {
                "bank_A_one_hot": "SEL1..SEL4 mutually exclusive; all-low between selections.",
                "bank_B_one_hot": "SEL1..SEL4 mutually exclusive; all-low between selections.",
                "paired_scan_order": [["T1", "T5"], ["T2", "T6"], ["T3", "T7"], ["T4", "T8"]],
                "settle_after_select_us": 100.0,
                "fresh_scan_period_ms": 20.0,
                "compute_cell_voltage": "Vcell[n]=Vtap[n]-Vtap[n-1], T0=GND_SYS.",
                "unsafe_state": "If two SEL channels in the same bank are high, different taps are tied together through two switch channels. Samples are invalid; firmware must enforce all-low then one-hot.",
            },
        },
        "signal_isolation_mux": {
            "ref": "U_SIGNAL_ISO",
            "mpn": "TMUX1511PWR",
            "supplier_part": "LCSC C2866750",
            "pin_map": {
                "VDD_pin14": "3V3_CORE",
                "GND_pin7": "GND_SYS",
                "SEL1_pin1": "3V3_CORE; channel1 on while core rail valid",
                "S1_pin2_to_D1_pin3": "BUS_ADC_SOURCE -> BUS_ADC_MCU / PB0 ADC1_IN15",
                "SEL2_pin4": "3V3_CORE; channel2 on while core rail valid",
                "S2_pin5_to_D2_pin6": "IMON_SOURCE -> IMON_MCU / PB1, PA7, PA0 comparator inputs",
                "SEL3_pin10_SEL4_pin13": "GND_SYS; unused off",
                "S3_pin9_S4_pin12": "GND_SYS",
                "D3_pin8_D4_pin11": "NC",
            },
            "host_pulldowns": [
                {"ref": "R_TEST_HI", "function": "BUS_ADC_MCU to GND_SYS"},
                {"ref": "R_TEST_LO", "function": "IMON_MCU to GND_SYS"},
            ],
            "source_filters_retained": ["C_BUS_ADC on BUS_ADC_SOURCE", "R_DRV_SO_SER=22ohm before IMON_SOURCE", "C_DRV_SO=100pF on IMON_SOURCE"],
        },
        "gpio_allocation": {
            "protected_reservations": [
                {"GPIO": "PA9", "function": "TIM1_CH2 hardware ARM hold output -> G07A; reset/fault state low"},
                {"GPIO": "PB12", "function": "TIM1_BKIN plus readback of actual active-low DRV_FAULT_N"},
                {"GPIO": "PA8/PB13", "function": "main complementary PWM"},
                {"GPIO": "PA6/PA2", "function": "COMP1/COMP2 open-drain fault outputs"},
                {"GPIO": "PA0/PB1/PA7", "function": "COMP3+/COMP1+/COMP2+ from IMON_MCU"},
            ],
            "monitor_pins": [
                {"GPIO": "PB0", "function": "BUS_ADC_MCU / ADC1_IN15"},
                {"GPIO": "PB14", "function": "SC_ADC_A / ADC1_IN5"},
                {"GPIO": "PB15", "function": "SC_ADC_B / ADC2_IN15"},
                {"GPIO": "PB11", "function": "NTC_MAIN ADC"},
                {"GPIO": "PC4", "function": "NTC_SC / ADC2_IN5"},
                {"GPIO": "PA3", "function": "NTC_DUMP / ADC1_IN4 (moved off PB12)"},
            ],
            "selector_gpio": ["PC6", "PA15", "PC10", "PC11", "PB3", "PB4", "PB5", "PB9"],
            "unused_or_optional": ["PA10 optional DRV_FAULT_N readback; otherwise free", "PB10 old TEST_CTRL released"],
            "board_variant_check": "PA9/PB6 connected-core-board solder bridges and actual core-board pinout must be read back before implementation; no carrier/core-board change is included here.",
        },
        "bom_delta_for_measurement_and_balance": {
            "new_physical_instances": [
                {"refs": ["U_CELL_MUX1", "U_CELL_MUX2", "U_SIGNAL_ISO"], "qty": 3, "mpn": "TMUX1511PWR", "manufacturer": "Texas Instruments", "supplier": "LCSC", "supplier_part": "C2866750", "reason": "four independent SPST channels, 3.6V powered-off protection, low Ron, signal isolation"},
                {"refs": ["R_ADC_A_PD", "R_ADC_B_PD"], "qty": 2, "mpn": "PTFR0805B10K0N9", "value": "10k 0.1%", "manufacturer": "RESI", "supplier": "LCSC", "supplier_part": "C328436", "reason": "one host-side D-bus pull-down per cell bank"},
            ],
            "moved_or_rewired": [
                {"refs": ["R_T1_TEST", "R_T2_TEST", "R_T3_TEST", "R_T4_TEST", "R_T5_TEST", "R_T6_TEST", "R_T7_TEST"], "part": "PTFR0805B47K0N9 47k 0.1%", "new_function": "SEL1..SEL7 external pull-downs to GND_SYS; one per cell-mux channel"},
                {"refs": ["R_QP_GS"], "part": "RMC060347K1%N 47k 1%", "new_function": "SEL8 external pull-down to GND_SYS; Q_BUS_P is retired"},
                {"refs": ["R_TEST_HI", "R_TEST_LO"], "part": "PTFR0805B10K0N9 10k 0.1%", "new_function": "U_SIGNAL_ISO BUS_ADC_MCU and IMON_MCU D-side pull-downs"},
                {"refs": ["C_T1_FILTER", "C_T2_FILTER"], "part": "CC0805CG100V102GN 1nF 100V C0G", "new_function": "cell-bank A/B D-output filters to GND_SYS"},
                {"refs": ["C_BAL1", "C_BAL2", "C_BAL3"], "part": "C0805C104K5RAC7800 100nF 50V X7R", "new_function": "local VDD bypass for U_CELL_MUX1, U_CELL_MUX2, U_SIGNAL_ISO"},
                {"refs": ["C_BAL4", "C_BAL5"], "part": "C0805C104K5RAC7800 100nF 50V X7R", "new_function": "reserve for parent-selected G07 and LTC local bypass"},
                {"refs": ["R_BAL1", "R_BAL2", "R_BAL3", "R_BAL4", "R_BAL5", "R_BAL6", "R_BAL7", "R_BAL8"], "part": "MFR25SFRE52-120R 120ohm 1% 0.25W", "new_function": "permanent resistor directly across each cell: R_BAL1=T1..SC_MINUS, R_BAL2=T2..T1, ... R_BAL8=SC_PLUS..T7"},
            ],
            "retired": [
                {"refs": ["Q_BUS_P"], "part": "IXTH76P10T", "reason": "parent-selected bus architecture removes this P-MOS; reclaim R_QP_GS for SEL8"},
                {"refs": ["U_BAL1", "U_BAL2", "U_BAL3", "U_BAL4", "U_BAL5", "U_BAL6", "U_BAL7", "U_BAL8"], "part": "BW6101-C1AP", "reason": "parent-selected always-connected 120R bleed; no controller IC"},
                {"refs": ["C_T3_FILTER", "C_T4_FILTER", "C_T5_FILTER", "C_T6_FILTER", "C_T7_FILTER", "C_T8_FILTER", "C_TEST_FB"], "reason": "old per-tap and TEST_FB capacitors no longer used; T1/T2 capacitors moved to bank outputs"},
                {"refs": ["C_BAL6", "C_BAL7", "C_BAL8"], "part": "C0805C104K5RAC7800", "reason": "released; no local-bypass assignment in the adopted count"},
            ],
            "retained_unchanged": [
                "R_T1_HI..R_T8_HI=RT0603BRD0768KL 68k ±0.1%",
                "R_T1_LO..R_T8_LO=PTFR0805B10K0N9 10k ±0.1%; all eight remain to limit mux S inputs",
                "R_BUS_HI2/R_BUS_HI=150k each in series; R_BUS_LO=10k",
                "R_DRV_SO_SER=22ohm; C_DRV_SO=100pF; C_BUS_ADC=1nF on source sides",
            ],
            "quantity_note": "Only this measurement/balance slice adds three mux ICs and two 10k resistors; five 100nF and two 1nF caps are moved, not newly procured. This does not replace the parent's global 22-add/19-retire BOM count.",
        },
        "analog_network_calculation": {
            "cell_divider": {
                "R_hi_ohm": r_hi,
                "R_lo_ohm": r_lo,
                "D_pull_ohm": r_pd,
                "mux_ron_screen_ohm": r_on,
                "source_Rth_ohm": r_th,
                "tap_source_ratio": 7.8,
                "tap_to_bank_ADC_gain": nominal_gain,
                "effective_cumulative_tap_division_ratio": 1.0 / nominal_gain,
                "divider_gain_tolerance_min_fraction": gain_min_error,
                "divider_gain_tolerance_max_fraction": gain_max_error,
                "25C_cal_to_0_60C_TCR_min_fraction": tcr_min_error,
                "25C_cal_to_0_60C_TCR_max_fraction": tcr_max_error,
                "bank_output_source_impedance_ohm": parallel(r_pd, r_th + r_on),
                "bank_output_tau_us_with_1nF": output_tau_s * 1e6,
                "tap_values_at_2p7V_each": tap_rows,
                "must_keep_R_Tn_LO": "If any R_Tn_LO is omitted/moved, corresponding mux S sees raw cumulative stack voltage up to 21.6V, exceeding 5.5V signal maximum and 3.6V powered-off range. Keep all eight 10k lower legs.",
            },
            "bus_divider": {
                "existing_upper_series_ohm": 300_000,
                "existing_lower_ohm": 10_000,
                "prior_ratio": bus_ratio_old,
                "loaded_ratio_with_reused_R_TEST_HI_and_mux": bus_ratio_loaded,
                "80V_source_side_ADC_V": bus_source_adc_80_v,
                "80V_MCU_D_side_ADC_V": bus_d_adc_80_v,
                "bus_mV_per_12bit_count_at_3p3V": bus_lsb_mV,
                "calibration_note": "Use about 61.014 V/V after mux and host-side loading; prior 31 V/V no longer applies to PB0.",
            },
            "imon": {
                "DRV_SO_series_resistor_ohm": 22.0,
                "mux_ron_max_ohm": 4.5,
                "MCU_side_pull_ohm": 10_000,
                "source_to_MCU_gain": imon_gain,
                "attenuation_percent": (1.0 - imon_gain) * 100.0,
                "max_3p05V_source_to_MCU_V": 3.05 * imon_gain,
                "note": "About 0.264% attenuation is calibratable; apply the same gain to ADC and comparator threshold interpretation.",
            },
            "cell_scan_timing": {
                "ADC_clock_MHz": adc_clock_hz / 1e6,
                "sample_cycles": adc_sample_cycles,
                "sample_time_us": adc_sample_s * 1e6,
                "conversion_cycles_total": adc_conversion_cycles,
                "conversion_time_us_total": adc_conversion_s * 1e6,
                "100us_settle_in_tau": settle_s / output_tau_s,
                "four_parallel_bank_pairs_scan_us": four_pair_scan_s * 1e6,
                "fresh_scan_period_ms_contract": 20.0,
                "max_ADC_source_ohm_reference": 33_000,
                "source_within_reference": parallel(r_pd, r_th + r_on) < 33_000,
            },
        },
        "power_off_and_static_load": {
            "powered_off": {
                "max_cell_mux_S_input_V_at_2p7V_each": tap_rows[-1]["source_AD_Tn_at_2p7V_per_cell_V"],
                "below_TMUX1511_3p6V_poweroff_limit": tap_rows[-1]["source_AD_Tn_at_2p7V_per_cell_V"] < 3.6,
                "cell_bank_worst_four_off_channel_leakage_uA": 8.0,
                "cell_bank_D_node_worst_first_order_voltage_mV_with_10k": 80.0,
                "third_mux_each_D_node_one_channel_leakage_uA": 2.0,
                "third_mux_each_D_node_worst_first_order_voltage_mV_with_10k": 20.0,
                "limitation": "Mux isolates MCU pins from the tap voltage when core power is off; it does not stop the continuous R_Tn divider current or direct R_BAL bleed from discharging the supercapacitors.",
            },
            "tap_and_bleed_current_tables": divider_load_rows,
            "direct_bleed_time_constants": tau_balance_rows,
            "capacity_mismatch": {
                "LICAP_capacitance_range_F": [350.0, 420.0],
                "at_charge_standby_target_low_cap_cell_V": 2.00,
                "equal_charge_high_cap_cell_V": 2.00 * 350.0 / 420.0,
                "voltage_spread_V": 2.00 - 2.00 * 350.0 / 420.0,
                "time_to_2p00_at_20A_from_0_for_350F_s": 350.0 * 2.00 / 20.0,
                "initial_charge_time_range_at_20A_s_for_280_350_420F": [280.0 * 2.00 / 20.0, 350.0 * 2.00 / 20.0, 420.0 * 2.00 / 20.0],
                "stored_energy_at_mismatch_target_J": 0.5 * (350.0 * 2.00**2 + 7 * 420.0 * (2.00 * 350.0 / 420.0) ** 2),
                "interpretation": "120R bleed current is under 0.1% of a 20A ramp and cannot correct the 0/+20% capacitance mismatch. It only dissipates energy and slowly reduces voltage differences; no active balancing claim is made.",
            },
            "steady_state_spread": balance_boundary,
            "parking_rc_matrix": {
                "model": "For node voltages X1..X8, M*dX/dt=-G*X; M is the series-capacitor nodal matrix, G contains each direct R_BAL resistor, each 78k cumulative tap resistor to GND_SYS, and an equivalent leakage conductance per cell. No source current during parking.",
                "direct_120R_worst_mismatch_case": parking_direct,
                "78k_tap_network_only_comparison": parking_divider_only,
                "takeaway": "With direct 120R per cell, storage is intentionally lost quickly (low-cap corner is about 24mV at day 2 and about 2mV at day 3 from a 2.2V start). The cumulative-tap return path did not create material reverse voltage in 0/+20% capacitance and bounded linear-leakage corners; numerical negative tail is only microvolts after all cells have already decayed to microvolts. This is a model result, not a reverse-voltage qualification.",
            },
        },
        "threshold_error_budget": {
            "conditions": {
                "ambient_C": [0, 60],
                "calibration_C": 25,
                "resistor_tolerance_fraction": 0.001,
                "resistor_TCR_bound_ppm_per_C": 25,
                "ADC_Vref_V": 3.3,
                "ADC_resolution_bits": 12,
                "ADC_ET_max_LSB_per_endpoint": 7.5,
                "ADC_INL_max_LSB_per_endpoint": 4.4,
                "quantization_bound_LSB_per_endpoint": 0.5,
                "VREFINT_residual_fraction": vref_error_fraction,
                "VREFINT_error_basis": "ST per-device factory VREFINT calibration at 30C +/-5C; worst full-temperature spread 7.5mV divided by 1.212V typical, plus 1000ppm/1000h maximum long-term stability and 1200ppm/V maximum supply coefficient over 0.3V.",
                "SC_MINUS_to_GND_SYS_shunt_ohm": 0.0005,
                "charge_current_A_for_age_and_shunt": charge_current_a,
                "cell_capacitance_F_for_age_sensitivity": cell_capacity_screen_f,
                "age_basis": "IWDG max timeout plus full scan duration; primary screen is 80A/280F (more conservative than the LICAP 350F minimum), with 50A/280F and 80A/350F comparison values included per threshold.",
                "ADC_error_scope": "G431 ET/INL values are characterized, not production-tested. Raw ET bound sums both endpoints without cancellation. Calibrated bound requires per-tap route gain/offset calibration and per-ADC calibration.",
            },
            "LSB_mV": lsb_v * 1000.0,
            "one_endpoint_cell_equivalent_LSB_mV": cell_equiv_lsb_v * 1000.0,
            "raw_conservative_all_cells_at_thresholds": threshold_rows,
            "per_channel_calibrated_all_cells_at_thresholds": cal_threshold_rows,
            "decision": {
                "2p00_charge_standby_target": "PASS in raw bound with substantial voltage headroom; normal initial charge at 20A takes 35s at 350F (28s at 280F and 42s at 420F).",
                "2p15_warning_derate": "PASS in raw bound; upper cell remains below 2.70V under the 80A/280F maximum-age screen.",
                "2p20_regeneration_stop": "PASS in raw uncalibrated screen with more than 110mV margin to 2.70V at 80A/280F, including 40mV ground-return error. the result relies on the listed datasheet characterization limits and the modeled 0.5mOhm SC_MINUS return. Per-tap route calibration is required for telemetry but not credited here.",
                "hard_protection_boundary": "These MCU/ADC thresholds do not create an independent hard overvoltage trip. Direct120R is too weak to stop a failed high-current charger. Native schematic and hardware are unchanged, so no hardware-protection PASS is claimed here.",
            },
        },
        "watchdog_and_freshness_contract": {
            "scan_period_ms": 20.0,
            "refresh_rule": "Only refresh IWDG after all four paired bank slots complete, both ADC results are fresh, no ADC overrun/invalid conversion occurred, and the 8 reconstructed cell results pass range/freshness checks. Any incomplete or invalid scan does not feed IWDG.",
            "IWDG_source": "STM32G431 LSI; datasheet range 29.5..34kHz over VDD/temp.",
            "prescaler_divisor": iwdg_prescaler,
            "reload_value": iwdg_r_load,
            "IWDG_min_timeout_ms": iwdg_min_s * 1000.0,
            "IWDG_max_timeout_ms": iwdg_max_s * 1000.0,
            "full_scan_duration_ms": four_pair_scan_s * 1000.0,
            "max_last_valid_sample_to_reset_ms": max_cell_sample_age_s * 1000.0,
            "age_rise_at_50A_280F_mV": age_rise_50a_v * 1000.0,
            "age_rise_at_80A_280F_mV": age_rise_v * 1000.0,
            "age_rise_at_80A_350F_mV": age_rise_80a_350f_v * 1000.0,
            "limitation": "The age bound covers MCU non-progress or a missing/invalid scan when the refresh rule is implemented. It does not cover a silently plausible but wrong ADC value or a power-stage shutdown path that ignores MCU reset; those are separate hardware verification items.",
        },
        "sources": [
            {"name": "TI TMUX1511 Rev B datasheet", "url": "https://www.ti.com/lit/ds/symlink/tmux1511.pdf"},
            {"name": "TI DRV8161 datasheet", "url": "https://www.ti.com/lit/ds/symlink/drv8161.pdf"},
            {"name": "ST STM32G431 datasheet", "url": "https://www.st.com/resource/en/datasheet/stm32g431cb.pdf"},
            {"name": "ST RM0440 reference manual", "url": "https://www.st.com/resource/en/reference_manual/dm00355726-stm32g4-series-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf"},
            {"name": "LICAP SC0350-270-RSS datasheet", "url": "https://www.licaptech.com/pdfs/datasheets/ultracapacitors/60862a8f5979e5f969df83ba_LICAP_SC0350_270_RSS_Datasheet_061120.pdf"},
            {"name": "Bitwell BW6101 product/spec page (existing devices retired in parent-selected direct-120R architecture)", "url": "https://bitwell.cn/productview.aspx?CateID=607&ID=269"},
            {"name": "YAGEO 68k RT0603BRD0768KL spec sheet", "url": "https://www.yageogroup.com/component-documentation/download/specsheet/RT0603BRD0768KL"},
            {"name": "YAGEO 150k RT1206BRD07150KL spec sheet", "url": "https://yageogroup.com/component-documentation/download/specsheet/RT1206BRD07150KL"},
            {"name": "RESI PTFR series datasheet", "url": "https://www.resistor.today/upload/file/20200703/RESI-CATALOG-2021.pdf"},
            {"name": "KEMET C0805C104K5RAC7800 spec alias", "url": "https://search.kemet.com/download/specsheet/C0805C104K5RAC7800"},
        ],
        "open_items": [
            "Paper selection only: add these refs/nets to a separately authorized native schematic revision and regenerate a fresh ENET/BOM; no such implementation occurred in this task.",
            "Per-tap calibration coefficients and G431 ADC1/ADC2 temperature verification are required for useful cell telemetry; the 2.20V threshold safety screen does not credit per-tap route calibration, but still relies on characterized ADC ET limits and must be closed by hardware temperature/current qualification.",
            "The 120R direct bleeders consume stored energy continuously; parked stack intentionally decays and must be recharged/initialized before the supercapacitor is counted for braking.",
            "The 0/+20% capacity and leakage parking result is a linear lumped model. LICAP has no reverse-voltage operating guarantee; preserve a nonnegative-cell policy in firmware and verify real cells if parking discharge is part of the release qualification.",
        ],
    }


def main() -> None:
    report = build_report()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps(report["threshold_error_budget"]["decision"], ensure_ascii=False))
    print(json.dumps(report["watchdog_and_freshness_contract"], ensure_ascii=False))
    print(json.dumps(report["power_off_and_static_load"]["steady_state_spread"]["worst_corner"], ensure_ascii=False))


if __name__ == "__main__":
    main()
