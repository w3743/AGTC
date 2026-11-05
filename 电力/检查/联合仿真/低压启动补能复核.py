"""Rerunnable low-voltage launch / braking reserve comparison for AGTC.

This is a bounded, average-model screen. It deliberately separates known
mechanical inputs, manufacturer limits, comparison assumptions, and HOLDs.
It does not model the switched power stage or qualify a vehicle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUT = HERE / "低压启动补能复核.json"

# User/project comparison inputs (not all are qualified ratings).
MASS_KG = 30.0
TARGET_SPEED_KMH = 15.0
WHEEL_RADIUS_M = 0.030
GEAR_RATIO = 4.0
MOTOR_COUNT = 2
DUAL_FOC_PEAK_INPUT_W = 3600.0
PACK_V_MIN_V = 22.5
PACK_V_MAX_V = 37.8
PACK_CURRENT_CANDIDATE_A = 130.0
PACK_SERIES = 9
PACK_PARALLEL = 5
EVE_CELL_MAX_DISCHARGE_A = 30.0

# LICAP manufacturer endurance screen: 350 F rated, <=20% capacitance loss,
# <=100% ESR rise from 2.8 mOhm maximum initial ESR.
SC_CELLS = 8
SC_CELL_C_RATED_F = 350.0
SC_EOL_C_F = 280.0
SC_CELL_ESR_INITIAL_MAX_OHM = 0.0028
SC_CELL_ESR_EOL_SCREEN_OHM = 0.0056
SC_CELL_STARTS_V = (2.0, 2.15, 2.2)
SC_CELL_MAX_V = 2.2
SC_CHARGE_CURRENT_SCREEN_A = 20.0

# Regulator baseline parts. L and PWM are screen values, not a manufacturer-
# guaranteed operating point. 20 kHz is the saved G431 PWM configuration.
INDUCTOR_COUNT = 2
INDUCTOR_EACH_H = 10e-6
INDUCTOR_L_BIASED_SCREEN_H = 12.8e-6
INDUCTOR_L_NO_BIAS_MIN_H = 16e-6
INDUCTOR_L_NO_BIAS_MAX_H = 24e-6
PWM_FREQUENCY_HZ = 20_000.0
INDUCTOR_PEAK_CURRENT_SCREEN_A = 80.0
MOSFET_RDS_MAX_OHM_25C = 0.0043
MOSFET_COUNT_MAIN = 2
CONVERTER_EFF_SCREEN = 0.95

# Joint braking comparison: battery charge and central dump deliberately get
# zero credit; only the supercapacitor path and connected bus capacitance count.
STOP_DISTANCE_M = 3.0
BRAKE_RESPONSE_DELAY_S = 0.050
BUS_CAP_LOWER_SCREEN_F = 0.05286
BUS_START_V = 37.8
BUS_LIMIT_COMPARISON_V = 39.009
BRAKE_DT_S = 0.001
LAUNCH_DT_S = 0.0005
JOINT_WINDOW_GRID_V = 0.01
LAUNCH_POWER_HOLD_DURATIONS_S = (0.5, 1.0, 1.39, 1.5, 2.0, 2.5, 3.0)
JOINT_WINDOW_DURATIONS_S = (0.5, 1.0, 1.39, 1.5, 2.0, 2.5, 3.0)

LICAP_URL = "https://www.licaptech.com/pdfs/datasheets/ultracapacitors/60862a8f5979e5f969df83ba_LICAP_SC0350_270_RSS_Datasheet_061120.pdf"
VISHAY_URL = "https://www.vishay.com/docs/34681/ihxl2000vz-3a.pdf"
INFINEON_URL = "https://www.infineon.com/dgdl/irlb4030pbf.pdf?fileId=5546d462533600a4015356604640258"


def sha256_file(relative: str) -> dict[str, str]:
    path = REPO / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sc_series_values(cell_v: float) -> tuple[float, float, float]:
    """Return OCV, C_eq, and ESR for eight equal series cells at EOL screen."""
    return (
        SC_CELLS * cell_v,
        SC_EOL_C_F / SC_CELLS,
        SC_CELLS * SC_CELL_ESR_EOL_SCREEN_OHM,
    )


def boost_ripple_a(v_terminal: float, bus_v: float, inductance_h: float, frequency_hz: float) -> float:
    if v_terminal <= 0 or bus_v <= v_terminal or inductance_h <= 0 or frequency_hz <= 0:
        return 0.0
    duty = 1.0 - v_terminal / bus_v
    return v_terminal * duty / (inductance_h * frequency_hz)


def boost_current_for_bus_power(
    p_bus_w: float,
    voc_v: float,
    resistance_ohm: float,
    bus_v: float,
    inductance_h: float,
    frequency_hz: float,
    peak_limit_a: float,
    efficiency: float,
) -> dict[str, float | bool]:
    """Solve constant-power boost input current, including triangular ripple.

    ESR ripple heating is included as R*deltaI^2/12.  This is an average
    CCM screen, not a switching waveform or a semiconductor current rating.
    """
    target_port_w = p_bus_w / efficiency

    def state(current_a: float) -> tuple[float, float, float]:
        terminal_v = voc_v - current_a * resistance_ohm
        ripple_a = boost_ripple_a(terminal_v, bus_v, inductance_h, frequency_hz)
        available_w = current_a * terminal_v - resistance_ohm * ripple_a * ripple_a / 12.0
        return available_w, ripple_a, terminal_v

    # Estimate the greatest average current whose ripple peak stays below the
    # 80 A screen. Two updates account for the ripple change with terminal V.
    max_average_a = peak_limit_a
    for _ in range(3):
        _, ripple, _ = state(max_average_a)
        max_average_a = max(0.0, peak_limit_a - ripple / 2.0)
    max_power_w, max_ripple_a, max_terminal_v = state(max_average_a)
    if max_power_w < target_port_w:
        return {
            "ok": False,
            "current_avg_a": max_average_a,
            "current_peak_a": max_average_a + max_ripple_a / 2.0,
            "ripple_pp_a": max_ripple_a,
            "terminal_v": max_terminal_v,
            "available_bus_w": max_power_w * efficiency,
        }

    # Small-root constant-power solution, refined for ripple ESR heating.
    requested_with_ripple_w = target_port_w
    current_a = 0.0
    for _ in range(3):
        discriminant = voc_v * voc_v - 4.0 * resistance_ohm * requested_with_ripple_w
        if discriminant < 0.0:
            return {
                "ok": False,
                "current_avg_a": max_average_a,
                "current_peak_a": max_average_a + max_ripple_a / 2.0,
                "ripple_pp_a": max_ripple_a,
                "terminal_v": max_terminal_v,
                "available_bus_w": max_power_w * efficiency,
            }
        current_a = (voc_v - math.sqrt(discriminant)) / (2.0 * resistance_ohm)
        _, ripple, _ = state(current_a)
        requested_with_ripple_w = target_port_w + resistance_ohm * ripple * ripple / 12.0
    available_w, ripple_a, terminal_v = state(current_a)
    if current_a > max_average_a or current_a + ripple_a / 2.0 > peak_limit_a:
        return {
            "ok": False,
            "current_avg_a": current_a,
            "current_peak_a": current_a + ripple_a / 2.0,
            "ripple_pp_a": ripple_a,
            "terminal_v": terminal_v,
            "available_bus_w": available_w * efficiency,
        }
    return {
        "ok": True,
        "current_avg_a": current_a,
        "current_peak_a": current_a + ripple_a / 2.0,
        "ripple_pp_a": ripple_a,
        "terminal_v": terminal_v,
        "available_bus_w": available_w * efficiency,
    }


def launch_full_peak(
    initial_cell_v: float,
    duration_s: float,
    inductance_h: float = INDUCTOR_L_BIASED_SCREEN_H,
    frequency_hz: float = PWM_FREQUENCY_HZ,
    dt_s: float = LAUNCH_DT_S,
) -> dict:
    """Stress envelope: hold 3.6 kW bus draw while travelling to 15 km/h.

    Battery is held at 22.5 V and capped at 130 A; this gives the SC a constant
    675 W bus supplement. No battery voltage sag or FOC transient is modeled.
    """
    voc_v, ceq_f, esr_ohm = sc_series_values(initial_cell_v)
    pack_power_w = PACK_V_MIN_V * PACK_CURRENT_CANDIDATE_A
    gap_bus_w = max(0.0, DUAL_FOC_PEAK_INPUT_W - pack_power_w)
    elapsed = 0.0
    peak_average_a = 0.0
    peak_ripple_a = 0.0
    peak_switch_a = 0.0
    min_voc_v = voc_v
    while elapsed < duration_s:
        step = min(dt_s, duration_s - elapsed)
        result = boost_current_for_bus_power(
            gap_bus_w, voc_v, esr_ohm, PACK_V_MIN_V,
            inductance_h, frequency_hz,
            INDUCTOR_PEAK_CURRENT_SCREEN_A, CONVERTER_EFF_SCREEN,
        )
        if not result["ok"]:
            return {
                "ok": False,
                "failure_time_s": elapsed,
                "reason": "full_peak_bus_supplement_exceeds_80A_peak_screen_or_power_limit",
                "initial_cell_v": initial_cell_v,
                "cell_v_at_failure": voc_v / SC_CELLS,
                "current_avg_a": result["current_avg_a"],
                "current_peak_a": result["current_peak_a"],
                "ripple_pp_a": result["ripple_pp_a"],
                "available_bus_w": result["available_bus_w"],
                "requested_bus_w": gap_bus_w,
                "inductance_h": inductance_h,
                "frequency_hz": frequency_hz,
            }
        current_a = float(result["current_avg_a"])
        ripple_a = float(result["ripple_pp_a"])
        peak_average_a = max(peak_average_a, current_a)
        peak_ripple_a = max(peak_ripple_a, ripple_a)
        peak_switch_a = max(peak_switch_a, float(result["current_peak_a"]))
        voc_v -= current_a * step / ceq_f
        min_voc_v = min(min_voc_v, voc_v)
        elapsed += step
    return {
        "ok": True,
        "initial_cell_v": initial_cell_v,
        "duration_s": duration_s,
        "final_cell_ocv_v": voc_v / SC_CELLS,
        "minimum_cell_ocv_v": min_voc_v / SC_CELLS,
        "max_sc_avg_current_a": peak_average_a,
        "max_inductor_ripple_pp_a": peak_ripple_a,
        "max_estimated_switch_peak_a": peak_switch_a,
        "low_voltage_gap_bus_w": gap_bus_w,
        "sc_converter_input_w_at_95pct": gap_bus_w / CONVERTER_EFF_SCREEN,
        "inductance_h": inductance_h,
        "frequency_hz": frequency_hz,
    }


def max_full_peak_duration(initial_cell_v: float, max_duration_s: float = 5.0) -> dict:
    result = launch_full_peak(initial_cell_v, max_duration_s)
    duration = result.get("failure_time_s", max_duration_s)
    return {
        "initial_cell_v": initial_cell_v,
        "max_duration_at_0p5ms_model_step_s": duration,
        "limiting_step": result if not result["ok"] else None,
    }


def brake_3m_sc_only(
    initial_cell_v: float,
    response_delay_s: float = BRAKE_RESPONSE_DELAY_S,
    dt_s: float = BRAKE_DT_S,
) -> dict:
    """3 m flat stop; zero battery/dump credit; 20 A SC charge screen only."""
    speed = TARGET_SPEED_KMH / 3.6
    v_init = speed
    braking_distance = STOP_DISTANCE_M - speed * response_delay_s
    if braking_distance <= 0:
        return {"ok": False, "reason": "response_delay_consumes_stop_distance"}
    decel = speed * speed / (2.0 * braking_distance)
    active_stop_s = speed / decel
    total_stop_s = response_delay_s + active_stop_s
    peak_mechanical_w = MASS_KG * decel * speed
    each_motor_torque_nm = MASS_KG * decel * WHEEL_RADIUS_M / (MOTOR_COUNT * GEAR_RATIO)

    voc_v, ceq_f, esr_ohm = sc_series_values(initial_cell_v)
    elapsed = 0.0
    bus_extra_j = 0.0
    peak_bus_extra_j = 0.0
    peak_bus_v = BUS_START_V
    max_terminal_cell_v = initial_cell_v
    max_charge_current_a = 0.0
    while elapsed < total_stop_s:
        step = min(dt_s, total_stop_s - elapsed)
        if elapsed < response_delay_s:
            mechanical_w = 0.0
        else:
            active_t = elapsed - response_delay_s + step / 2.0
            velocity = max(0.0, speed - decel * active_t)
            mechanical_w = MASS_KG * decel * velocity

        demand_w = mechanical_w + bus_extra_j / step
        max_i_by_cell_voltage = max(
            0.0,
            (SC_CELLS * SC_CELL_MAX_V - voc_v) / (esr_ohm + step / ceq_f),
        )
        current_limit = min(SC_CHARGE_CURRENT_SCREEN_A, max_i_by_cell_voltage)
        # During regeneration the BUS is the converter input, so its required
        # power is SC-terminal charging power divided by converter efficiency.
        max_absorb_bus_w = current_limit * (voc_v + current_limit * esr_ohm) / CONVERTER_EFF_SCREEN
        if demand_w <= 0.0 or current_limit <= 0.0:
            current_a = 0.0
        elif demand_w >= max_absorb_bus_w:
            current_a = current_limit
        else:
            # Solve I * (Voc + I*R) / eta = requested BUS absorption.
            current_a = 2.0 * CONVERTER_EFF_SCREEN * demand_w / (
                voc_v + math.sqrt(voc_v * voc_v + 4.0 * esr_ohm * CONVERTER_EFF_SCREEN * demand_w)
            )
        absorbed_bus_w = current_a * (voc_v + current_a * esr_ohm) / CONVERTER_EFF_SCREEN
        bus_extra_j = max(0.0, bus_extra_j + (mechanical_w - absorbed_bus_w) * step)
        peak_bus_extra_j = max(peak_bus_extra_j, bus_extra_j)
        voc_v += current_a * step / ceq_f
        terminal_cell_v = (voc_v + current_a * esr_ohm) / SC_CELLS
        max_terminal_cell_v = max(max_terminal_cell_v, terminal_cell_v)
        peak_bus_v = max(
            peak_bus_v,
            math.sqrt(BUS_START_V * BUS_START_V + 2.0 * bus_extra_j / BUS_CAP_LOWER_SCREEN_F),
        )
        max_charge_current_a = max(max_charge_current_a, current_a)
        elapsed += step

    return {
        "ok": peak_bus_v <= BUS_LIMIT_COMPARISON_V and max_terminal_cell_v <= SC_CELL_MAX_V + 1e-9,
        "response_delay_s": response_delay_s,
        "stop_distance_from_request_m": STOP_DISTANCE_M,
        "distance_during_delay_m": speed * response_delay_s,
        "active_braking_distance_m": braking_distance,
        "active_braking_time_s": active_stop_s,
        "total_request_to_stop_s": total_stop_s,
        "constant_deceleration_m_s2": decel,
        "peak_mechanical_braking_power_w": peak_mechanical_w,
        "ideal_each_motor_braking_torque_nm": each_motor_torque_nm,
        "translation_energy_j": 0.5 * MASS_KG * speed * speed,
        "initial_cell_ocv_v": initial_cell_v,
        "final_cell_ocv_v": voc_v / SC_CELLS,
        "maximum_cell_terminal_v": max_terminal_cell_v,
        "max_sc_charge_current_a": max_charge_current_a,
        "peak_unabsorbed_bus_energy_j": peak_bus_extra_j,
        "bus_peak_v": peak_bus_v,
        "bus_limit_comparison_v": BUS_LIMIT_COMPARISON_V,
        "bus_margin_v": BUS_LIMIT_COMPARISON_V - peak_bus_v,
        "bus_energy_headroom_at_start_j": 0.5 * BUS_CAP_LOWER_SCREEN_F * (BUS_LIMIT_COMPARISON_V**2 - BUS_START_V**2),
    }


def joint_window(duration_s: float, start_low_v: float = 1.0, start_high_v: float = 2.2) -> dict:
    """Find 10 mV-grid starting cell-voltage interval passing launch plus stop."""
    # Skip a long scan if even the best permitted starting voltage misses the
    # full-peak duration; the result would otherwise be an empty grid.
    best_launch = launch_full_peak(start_high_v, duration_s)
    if not best_launch["ok"]:
        return {"duration_s": duration_s, "status": "NO_FULL_PEAK_LAUNCH_AT_2P2V_SCREEN", "best_case": best_launch}
    count = round((start_high_v - start_low_v) / JOINT_WINDOW_GRID_V)
    passing = []
    for n in range(count + 1):
        initial = start_low_v + JOINT_WINDOW_GRID_V * n
        launch = launch_full_peak(initial, duration_s)
        if not launch["ok"]:
            continue
        stop = brake_3m_sc_only(float(launch["final_cell_ocv_v"]))
        if stop["ok"]:
            passing.append({
                "initial_cell_v": initial,
                "post_launch_cell_v": launch["final_cell_ocv_v"],
                "post_brake_cell_v": stop["final_cell_ocv_v"],
                "bus_peak_v": stop["bus_peak_v"],
                "launch_max_current_peak_a": launch["max_estimated_switch_peak_a"],
            })
    if not passing:
        return {"duration_s": duration_s, "status": "NO_COMMON_WINDOW_ON_10mV_GRID", "points_tested": count + 1}
    return {
        "duration_s": duration_s,
        "status": "candidate_window_only",
        "initial_cell_v_low_grid": passing[0]["initial_cell_v"],
        "initial_cell_v_high_grid": passing[-1]["initial_cell_v"],
        "grid_step_v": JOINT_WINDOW_GRID_V,
        "passing_points": len(passing),
        "points_tested": count + 1,
        "lowest_point": passing[0],
        "highest_point": passing[-1],
        "midpoint_sample": passing[len(passing) // 2],
    }


def constant_acceleration_traction(duration_s: float) -> dict[str, float]:
    speed = TARGET_SPEED_KMH / 3.6
    accel = speed / duration_s
    total_force_n = MASS_KG * accel
    each_motor_torque_nm = total_force_n * WHEEL_RADIUS_M / (MOTOR_COUNT * GEAR_RATIO)
    mech_peak_w = total_force_n * speed
    return {
        "duration_s": duration_s,
        "constant_acceleration_m_s2": accel,
        "minimum_tire_friction_coefficient_flat": accel / 9.80665,
        "total_wheel_force_n": total_force_n,
        "each_motor_shaft_torque_nm_ideal_gears": each_motor_torque_nm,
        "mechanical_peak_total_power_w": mech_peak_w,
        "mechanical_energy_j": 0.5 * MASS_KG * speed * speed,
        "input_peak_total_w_at_90pct_drive_efficiency_screen": mech_peak_w / 0.90,
        "input_peak_total_w_at_80pct_drive_efficiency_screen": mech_peak_w / 0.80,
    }


def parse_power_bom() -> dict:
    path = REPO / "电力/稳压模块/原理图准备/选型BOM.csv"
    counts = {"IHXL2000VZEB100M3A": 0, "IRLB4030PBF": 0, "SC0350-270-RSS": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mpn = row.get("Manufacturer part", "")
            for target in counts:
                if mpn == target:
                    counts[target] += int(row.get("Quantity") or 1)
    return {"path": "电力/稳压模块/原理图准备/选型BOM.csv", "counts_by_exact_mpn": counts}


def build_result() -> dict:
    speed = TARGET_SPEED_KMH / 3.6
    gap_bus_w = max(0.0, DUAL_FOC_PEAK_INPUT_W - PACK_V_MIN_V * PACK_CURRENT_CANDIDATE_A)
    k_t_nm_a_ideal = 60.0 / (2.0 * math.pi * 270.0)
    stop_no_delay = brake_3m_sc_only(2.0, response_delay_s=0.0)
    stop_50ms = brake_3m_sc_only(2.0, response_delay_s=BRAKE_RESPONSE_DELAY_S)

    duration_checks = []
    for duration in LAUNCH_POWER_HOLD_DURATIONS_S:
        launch = launch_full_peak(2.0, duration)
        if launch["ok"]:
            post_start_brake = brake_3m_sc_only(float(launch["final_cell_ocv_v"]))
            launch["braking_from_post_launch"] = post_start_brake
        duration_checks.append(launch)

    max_duration = [max_full_peak_duration(v) for v in (2.0, 2.15, 2.2)]
    windows = [joint_window(t) for t in JOINT_WINDOW_DURATIONS_S]
    traction = [constant_acceleration_traction(t) for t in (0.5, 1.0, 2.0, 3.0)]
    cell_discharge = PACK_CURRENT_CANDIDATE_A / PACK_PARALLEL
    full_peak_cell_discharge = (DUAL_FOC_PEAK_INPUT_W / PACK_V_MIN_V) / PACK_PARALLEL

    # Loss-only static sensitivity; actual bridge duty, current distribution,
    # switching loss, diode/deadtime and thermal impedance are not supplied.
    mos_loss_1_fet_52a = 52.0**2 * MOSFET_RDS_MAX_OHM_25C
    mos_loss_1_fet_80a = 80.0**2 * MOSFET_RDS_MAX_OHM_25C
    mos_loss_two_fet_80a = 2.0 * mos_loss_1_fet_80a
    dcr_hot_2_inductors = 2.0 * 0.00086 * (1.0 + 0.00393 * (100.0 - 25.0))

    sources = [
        "电力/基本设计.md",
        "电力/FOC驱动模块/设计需求.md",
        "电力/FOC驱动模块/设计细节.md",
        "电力/稳压模块/设计需求.md",
        "电力/稳压模块/设计细节.md",
        "电力/稳压模块/原理图准备/选型BOM.csv",
        "电力/检查/联合仿真/起步停车与下坡包络.py",
        "电力/参考资料/电池/11-_EVE_INR18650-30P_SPEC._PBRI-INR1865030P-D06-01-B-2023.8.4.pdf",
        "电力/稳压模块/参考资料/LICAP_SC0350-270-RSS_英文2021.pdf",
    ]
    return {
        "schema_version": "agtc-low-voltage-launch-braking-window.v2",
        "status": "HOLD_NOT_QUALIFIED",
        "run_date_local": "2026-11-05",
        "scope": "15 km/h flat start-to-speed electrical supply envelope plus 15 km/h, 3 m flat stop with zero battery-charge and zero central-dump credit",
        "source_hashes": [sha256_file(s) for s in sources],
        "manufacturer_evidence": {
            "LICAP_SC0350_270_RSS": {
                "url": LICAP_URL,
                "rated_capacitance_f": 350.0,
                "capacitance_tolerance": "0/+20%",
                "maximum_initial_dc_esr_ohm": SC_CELL_ESR_INITIAL_MAX_OHM,
                "endurance_condition": "rated voltage, 65°C, 1500 h",
                "endurance_change_limits": {"capacitance_drop_max_fraction": 0.20, "esr_increase_max_fraction_of_max_initial": 1.0},
                "cycle_life_condition": "rated voltage to 50% voltage, constant current, 25°C",
                "projected_cycle_life": 1_000_000,
                "cycle_change_limits": {"capacitance_drop_max_fraction": 0.20, "esr_increase_max_fraction_of_max_initial": 1.0},
                "screen_mapping": {"cell_capacitance_f": SC_EOL_C_F, "cell_esr_ohm": SC_CELL_ESR_EOL_SCREEN_OHM},
                "qualification_limit": "datasheet life projections/limits do not predict the actual service life or validate a pack; balancing, temperature, ripple, cycle profile, aging and cell mismatch remain untested",
            },
            "Vishay_IHXL2000VZEB100M3A": {
                "url": VISHAY_URL,
                "per_part_inductance_h": 10e-6,
                "tolerance_at_100khz": 0.20,
                "per_part_dcr_25c_typ_ohm": 0.00082,
                "per_part_dcr_25c_max_ohm": 0.00086,
                "per_part_heat_rating_typ_a_at_40c_rise": 83.0,
                "per_part_heat_rating_typ_a_at_80c_rise": 118.0,
                "per_part_saturation_current_typ_a_at_20pct_inductance_drop": 83.0,
                "per_part_saturation_current_typ_a_at_30pct_inductance_drop": 123.0,
                "max_part_temperature_c": 155.0,
                "existing_series_pair_nominal_h": 20e-6,
                "project_bias_screen_at_80a_h": INDUCTOR_L_BIASED_SCREEN_H,
                "qualification_limit": "manufacturer heat/saturation currents are typical datasheet points, not guaranteed design current in this assembly; ambient 60°C, core loss, ripple, layout, thermal coupling and cooling still require qualification",
            },
            "Infineon_IRLB4030PBF": {
                "url": INFINEON_URL,
                "rds_on_max_ohm_at_vgs10_tj25": MOSFET_RDS_MAX_OHM_25C,
                "max_continuous_drain_current_at_25c_a": 180.0,
                "max_continuous_drain_current_at_100c_a": 130.0,
                "max_junction_temperature_c": 175.0,
                "qualification_limit": "single-device absolute/current ratings do not qualify the two-switch converter, PCB, gate-drive voltage, switching loss, SOA, cooling or fault response",
            },
            "EVE_INR18650_30P_local_manufacturer_spec": {
                "source_file": "电力/参考资料/电池/11-_EVE_INR18650-30P_SPEC._PBRI-INR1865030P-D06-01-B-2023.8.4.pdf",
                "max_cell_discharge_current_a": EVE_CELL_MAX_DISCHARGE_A,
                "temperature_note": "maximum discharge specification states 30A (10C), 75°C cut-off; operating discharge temperature lists -20 to 60°C ambient, with recommended release below 60°C after re-discharge",
                "five_parallel_cell_current_at_130a_pack_a": cell_discharge,
                "five_parallel_cell_current_at_160a_pack_a": full_peak_cell_discharge,
                "qualification_limit": "30A cell value does not establish pack/BMS/cell-tab thermal qualification; current sharing, low-SOC DC resistance and pack sag are not modeled",
            },
        },
        "comparison_assumptions": {
            "mass_kg": MASS_KG,
            "target_speed_kmh": TARGET_SPEED_KMH,
            "wheel_radius_m": WHEEL_RADIUS_M,
            "gear_ratio": GEAR_RATIO,
            "motor_count": MOTOR_COUNT,
            "pack_voltage_v": [PACK_V_MIN_V, PACK_V_MAX_V],
            "pack_current_a": PACK_CURRENT_CANDIDATE_A,
            "pack_current_status": "130A is an unimplemented engineering candidate; current BMS source is not the 9S5P/130A configuration",
            "dual_foc_input_peak_w": DUAL_FOC_PEAK_INPUT_W,
            "sc_cells_series": SC_CELLS,
            "sc_eol_screen_capacitance_f_per_cell": SC_EOL_C_F,
            "sc_eol_screen_esr_ohm_per_cell": SC_CELL_ESR_EOL_SCREEN_OHM,
            "sc_charge_current_a": SC_CHARGE_CURRENT_SCREEN_A,
            "converter_efficiency": {"value": CONVERTER_EFF_SCREEN, "status": "assumed comparison only; actual power-stage efficiency unknown"},
            "inductor_pair_bias_screen_h": INDUCTOR_L_BIASED_SCREEN_H,
            "pwm_frequency_hz": {"value": PWM_FREQUENCY_HZ, "source": "existing regulator firmware BSP configuration", "status": "configuration value, not a measured switching waveform"},
            "inductor_peak_current_a": {"value": INDUCTOR_PEAK_CURRENT_SCREEN_A, "status": "screen point; not a qualified converter limit"},
            "bus_capacitance_f": BUS_CAP_LOWER_SCREEN_F,
            "bus_start_v": BUS_START_V,
            "bus_limit_comparison_v": BUS_LIMIT_COMPARISON_V,
            "bus_limit_status": "low-side FOC BOV comparison threshold, not a safe maximum rating",
            "braking_response_delay_s": BRAKE_RESPONSE_DELAY_S,
            "braking_sources_credited": {"battery_net_charge": 0.0, "central_dump": 0.0, "mechanical_brake": 0.0, "sc_dc_dc": "only; efficiency and current are assumed"},
            "not_modeled": ["battery sag/DCIR and BMS trip/thermal", "actual motor torque-speed-current curve", "tire friction/weight transfer", "rotor and gearbox inertia", "drive efficiency", "rolling resistance/aero", "switched converter and control delay", "MOS/diode switching and reverse-recovery losses", "PCB/connector/inductor thermal network", "cell imbalance and temperature gradients", "mechanical brake (none exists)"],
        },
        "derived_electrical_baseline": {
            "battery_power_at_22p5v_130a_w": PACK_V_MIN_V * PACK_CURRENT_CANDIDATE_A,
            "dual_foc_peak_gap_bus_w": gap_bus_w,
            "supercap_terminal_input_required_at_95pct_w": gap_bus_w / CONVERTER_EFF_SCREEN,
            "ideal_full_peak_pack_current_at_22p5v_a": DUAL_FOC_PEAK_INPUT_W / PACK_V_MIN_V,
            "ideal_full_peak_pack_current_at_37p8v_a": DUAL_FOC_PEAK_INPUT_W / PACK_V_MAX_V,
            "battery_cell_current_at_130a_pack_a": cell_discharge,
            "battery_cell_current_at_160a_pack_a": full_peak_cell_discharge,
            "motor_torque_constant_ideal_nm_per_a": k_t_nm_a_ideal,
            "each_motor_torque_from_80a_if_current_definition_matches_kv_a": 80.0 * k_t_nm_a_ideal,
            "motor_rpm_at_15kmh_with_60mm_wheel_and_4to1": speed / (2.0 * math.pi * WHEEL_RADIUS_M) * 60.0 * GEAR_RATIO,
            "maximum_full_peak_duration_by_initial_cell_voltage": max_duration,
            "full_peak_hold_checks_starting_at_2v_cell": duration_checks,
            "traction_constant_acceleration_profiles": traction,
            "interpretation": "3.6kW is a peak supply-capability envelope, not a physical constant load from zero speed. Traction profile table gives inertial lower-bound force/power for chosen ramps; actual ramp cannot be uniquely qualified without friction, torque/current map, losses and inertia.",
        },
        "power_stage_screen": {
            "bom_counts": parse_power_bom(),
            "inductor_pair_nominal_h": INDUCTOR_COUNT * INDUCTOR_EACH_H,
            "inductor_no_bias_tolerance_total_h": [INDUCTOR_L_NO_BIAS_MIN_H, INDUCTOR_L_NO_BIAS_MAX_H],
            "inductor_pair_dcr_max_at_25c_ohm": 2.0 * 0.00086,
            "inductor_pair_dcr_estimate_at_100c_ohm": dcr_hot_2_inductors,
            "inductor_pair_copper_loss_at_52a_100c_est_w": 52.0**2 * dcr_hot_2_inductors,
            "inductor_pair_copper_loss_at_80a_100c_est_w": 80.0**2 * dcr_hot_2_inductors,
            "inductor_ripple_peak_screen_by_frequency": [
                {"frequency_hz": f, "ripple_pp_a_at_worst_duty_22p5v_12p8uH": PACK_V_MIN_V / (4.0 * INDUCTOR_L_BIASED_SCREEN_H * f)}
                for f in (10_000.0, 20_000.0, 40_000.0)
            ],
            "ir lb4030_static_loss_screen": {
                "one_device_at_52a_w_25c_rds_max": mos_loss_1_fet_52a,
                "one_device_at_80a_w_25c_rds_max": mos_loss_1_fet_80a,
                "two_device_full_current_simultaneous_upper_sensitivity_at_80a_w": mos_loss_two_fet_80a,
                "interpretation": "ideal complementary half-bridge total conduction loss depends on current duty split; static bound uses RDS(on) maximum at 25°C and omits temperature increase, switching/deadtime/diode/core/contact losses. The datasheet 180A/130A ratings are not board ratings.",
            },
            "power_stage_conclusion": "Existing parts are the zero-addition candidate. At 2.0V/cell, the 675W bus gap is conditionally supportable for a bounded interval in this average screen; no full converter capability is qualified because switch model, control, temperature and component/package/PCB thermal limits remain open.",
        },
        "joint_launch_braking_window": {
            "start_to_15kmh_full_peak_duration_s": windows,
            "three_meter_stop_at_2v_cell_no_added_delay": stop_no_delay,
            "three_meter_stop_at_2v_cell_with_50ms_response_delay": stop_50ms,
            "interpretation": "A common window exists only for the specified modeled launch duration and starting cell-voltage band, under the stated 80A/20A/95% assumptions. Passing this averaged screen does not qualify physical stopping, which also needs measured torque, grip, response and working hardware.",
        },
        "cost_paths": {
            "recommended_lowest_candidate": {
                "path": "reuse the currently selected 8S1P LICAP bank and the two selected 10uH inductors/two main MOSFETs; use software power limiting when the measured SC reserve or temperature/current margin is insufficient",
                "board_bom_delta": {"supercap_cells": 0, "inductors": 0, "main_mosfets": 0},
                "verified_incremental_cost_cny": 0.0,
                "meaning": "zero new BOM items relative to current selected design, not zero total vehicle procurement cost and not evidence that parts are in stock; complete purchase value remains unknown",
                "qualification": "HOLD; must measure complete switching/thermal envelope and stop reserve before assigning launch power",
            },
            "add_eight_cells_for_8s2p": {
                "board_bom_delta": {"supercap_cells": 8},
                "price_status": "not costable at a valid per-vehicle purchase price from saved evidence: project source records a 256-piece MOQ / USD 7.81-per-piece bulk-page quote, while 8 are needed; no current domestic delivered quote/tax/freight was verified",
                "decision": "not selected; do not treat this as a required change until current 8S1P time-domain result fails the user-selected launch profile",
            },
            "inductor_or_mosfet_upsize": {
                "board_bom_delta": "unknown",
                "price_status": "no exact replacement MPN with verified thermal/current fit and same-channel delivered price was established in this scoped review",
                "decision": "do not add before measured current/temperature shows the saved pair cannot meet the requirement",
            },
            "cost_conclusion": "The least-cost route supported by evidence is reuse plus a conditional launch power cap. No hardware change is justified by the current model alone, and no net savings claim is made. If the required full-peak duration exceeds the modeled common window, actual duration and part pricing must be resolved before proposing a BOM change.",
        },
        "qualification_gates": [
            "HOLD: 9S5P/130A BMS is an unimplemented candidate; pack sag, cell sharing, connector, busbar and BMS thermal behavior at low SOC are absent.",
            "HOLD: motor 1.8kW input peak and 80A seller figure do not define current convention, duration, torque-speed curve or motor temperature; 15km/h traction cannot be derived from these values alone.",
            "HOLD: 95% converter efficiency and 20A SC charging limit are comparison inputs, not measured/guaranteed stage performance.",
            "HOLD: 80A peak-current screen and 12.8uH biased inductance are not qualified limits; manufacturer heat currents are typical points and assembly thermal impedance is unknown.",
            "HOLD: EOL cell ESR/capacitance values are datasheet condition limits, not a service-life prediction; cell mismatch and balancing are not modeled.",
            "HOLD: braking assumes 50ms delay as an explicit planning stress input, ideal mechanical-to-BUS recovery, zero battery/dump and a constant-deceleration 3m stop; no physical stopping guarantee follows.",
            "HOLD: 39.009V is a comparator threshold reference, not a permitted bus maximum or safe design voltage.",
        ],
        "sources_and_limits": {
            "PWM_source": "固件/regulator_driver/Bsp/Inc/bsp_board.h:138; configuration value only",
            "LICAP_original_datasheet": LICAP_URL,
            "Vishay_original_datasheet": VISHAY_URL,
            "Infineon_original_datasheet": INFINEON_URL,
            "local_EVE_cell_spec": "电力/参考资料/电池/11-_EVE_INR18650-30P_SPEC._PBRI-INR1865030P-D06-01-B-2023.8.4.pdf",
            "existing_brake_envelope": "电力/检查/联合仿真/起步停车与下坡包络.py; this file retains it as reference and adds 50ms response-delay and joint launch/brake simulation",
        },
    }


def main() -> None:
    result = build_result()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.name}: {result['status']}")


if __name__ == "__main__":
    main()
