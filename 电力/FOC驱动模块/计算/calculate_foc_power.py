#!/usr/bin/env python3
"""Auditable first-pass FOC inverter power-stage calculations (stdlib only).

Run from any directory with Python 3:
    python calculate_foc_power.py

Writes foc_power_results.json beside this script.  Values are selection
screens under the explicit assumptions documented in 1秒脉冲能力核算.md, not a
hardware qualification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


OUT = Path(__file__).with_name("foc_power_results.json")

P_MOTOR_W = 1800.0          # electrical power at one motor/inverter output, if that interpretation applies
T_PULSE_S = 1.0
N_BOARDS = 2
V_BUS = (22.5, 37.8)
I_BUS_CANDIDATE_A = 100.0
I_PHASE_PK_CANDIDATE_A = 120.0
N_FET = 6
FSW_HZ = (10_000.0, 20_000.0, 30_000.0, 40_000.0)
EDGE_SCALE = (0.5, 1.0, 2.0)
R_SHUNT_OHM = 0.0005
SHUNT_GAIN = 20.0

MOSFETS = {
    "IRLB4030PbF": {
        "vds_v": 100.0,
        "rds25_max_ohm": 0.0043,
        "rds_temp_factor_175_graph_range": [2.4, 2.5],
        "rds_temp_factor_used": 2.5,
        "rds_hot_used_ohm": 0.0043 * 2.5,
        "tr_ns": 330.0,
        "tf_ns": 170.0,
        "tr_tf_test_condition": "VDD=65 V, ID=110 A, RG=2.7 ohm, VGS=4.5 V (datasheet test; not the proposed driver waveform)",
        "qg_max_nc": 130.0,
        "tj_max_c": 175.0,
        "thermal_model": "manufacturer RC network, Figure 13; see calculation below",
        "zth_rc_c_per_w": [0.0477, 0.1631, 0.1893],
        "tau_rc_s": [71e-6, 881e-6, 7.457e-3],
        "zth_1s_c_per_w": sum([0.0477, 0.1631, 0.1893])
        - sum(r * math.exp(-1.0 / tau) for r, tau in zip([0.0477, 0.1631, 0.1893], [71e-6, 881e-6, 7.457e-3])),
    },
    "STL130N6F7": {
        "vds_v": 60.0,
        "rds25_max_ohm": 0.0035,
        "rds_temp_factor_175_graph_range": [1.8, 1.9],
        "rds_temp_factor_used": 1.9,
        "rds_hot_used_ohm": 0.0035 * 1.9,
        "tr_ns": 44.0,
        "tf_ns": 24.0,
        "tr_tf_test_condition": "VDD=30 V, ID=26 A, RG=4.7 ohm, VGS=10 V (datasheet test; not the proposed driver waveform)",
        "qg_typ_nc": 42.0,
        "tj_max_c": 175.0,
        "rth_jc_c_per_w": 1.2,
        "zth_1s_method": "not extrapolated from the datasheet plot; RthJC=1.2 C/W used only as a conservative asymptotic upper envelope for a conditional screen, not 1 s proof",
    },
}


def zth_rc_at_s(r_values: list[float], tau_values: list[float], t_s: float) -> float:
    return sum(r * (1.0 - math.exp(-t_s / tau)) for r, tau in zip(r_values, tau_values))


def capacitor_rms_factor(m: float, pf: float) -> float:
    """Kolar/Infineon 3-phase sinusoidal PWM DC-link RMS factor.

    m=V_phase,peak/(Vdc/2), linear SVPWM range 0..2/sqrt(3).
    pf is fundamental displacement power factor, 0..1.
    """
    a = math.sqrt(3.0) / (4.0 * math.pi)
    b = math.sqrt(3.0) / math.pi
    q = pf * pf
    f2 = 2.0 * m * (a + q * (b - 9.0 * m / 16.0))
    return math.sqrt(max(0.0, f2))


def capacitor_rms_factor_max(pf: float) -> tuple[float, float]:
    m_max = 2.0 / math.sqrt(3.0)
    a = math.sqrt(3.0) / (4.0 * math.pi)
    b = math.sqrt(3.0) / math.pi
    q = pf * pf
    m_star = m_max if q == 0.0 else (a + q * b) / (2.0 * q * 9.0 / 16.0)
    m_star = min(max(m_star, 0.0), m_max)
    return capacitor_rms_factor(m_star, pf), m_star


def round_tree(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: round_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_tree(v) for v in obj]
    return obj


def main() -> None:
    phase_rms_min = {}
    for v in V_BUS:
        irms = math.sqrt(2.0 / 3.0) * P_MOTOR_W / v
        ipk = math.sqrt(2.0) * irms
        phase_rms_min[str(v)] = {
            "bus_current_at_1800w_dc_input_a": P_MOTOR_W / v,
            "phase_rms_min_at_full_linear_svpwm_unity_pf_a": irms,
            "phase_peak_min_at_full_linear_svpwm_unity_pf_a": ipk,
            "phase_power_boundary_at_120a_peak_w": math.sqrt(3.0) / 2.0 * v * I_PHASE_PK_CANDIDATE_A,
            "dc_power_boundary_at_100a_w": I_BUS_CANDIDATE_A * v,
            "combined_boundary_w": min(math.sqrt(3.0) / 2.0 * v * I_PHASE_PK_CANDIDATE_A, I_BUS_CANDIDATE_A * v),
            "phase_peak_equivalent_at_100a_bus_full_linear_svpwm_unity_pf_a": 2.0 / math.sqrt(3.0) * I_BUS_CANDIDATE_A,
        }

    sine_irms = I_PHASE_PK_CANDIDATE_A / math.sqrt(2.0)
    locked_vector = [I_PHASE_PK_CANDIDATE_A, -I_PHASE_PK_CANDIDATE_A / 2.0, -I_PHASE_PK_CANDIDATE_A / 2.0]
    locked_abs_sum = sum(abs(x) for x in locked_vector)
    sine_abs_mean_sum = 6.0 / math.pi * I_PHASE_PK_CANDIDATE_A

    loss_cases = {}
    for name, mos in MOSFETS.items():
        r_hot = mos["rds_hot_used_ohm"]
        tr_tf_s = (mos["tr_ns"] + mos["tf_ns"]) * 1e-9
        # For balanced three-phase sinusoidal current, sum over the six FETs
        # using Iavg|sin| = 2/pi * Ipk; 6/pi is a deliberately conservative
        # all-six-device overlap upper screen. The half factor is sensitivity.
        pcond_sine_total = 3.0 * sine_irms**2 * r_hot
        per_fet_cond_sine = pcond_sine_total / N_FET
        pcond_lock_total = sum(i * i for i in locked_vector) * r_hot
        max_lock_fet_cond = I_PHASE_PK_CANDIDATE_A**2 * r_hot
        case = {
            "rds_hot_screen_ohm": r_hot,
            "rds_temp_factor_used_at_175c": mos["rds_temp_factor_used"],
            "phase_rms_candidate_a": sine_irms,
            "sine_conduction_total_w": pcond_sine_total,
            "sine_conduction_average_per_fet_w": per_fet_cond_sine,
            "locked_phase_vector_a": locked_vector,
            "locked_conduction_total_w": pcond_lock_total,
            "locked_hottest_fet_conduction_w": max_lock_fet_cond,
            "switching_model": "Psw_upper = Vdc * fsw * (tr+tf) * sum_phase(mean(abs(Iphase))); sine sum=6/pi*Ipk; locked vector [120,-60,-60] sum=240 A. Sensitivity half is included because complementary leg commutation may make the six-device overlap approximation conservative.",
            "switching_sensitivity": {},
            "thermal_1s": {},
        }
        for v in V_BUS:
            vk = str(v)
            case["switching_sensitivity"][vk] = {"sine": [], "locked_vector": []}
            for fsw in FSW_HZ:
                for scale in EDGE_SCALE:
                    p_sw_sine_upper = v * fsw * tr_tf_s * scale * sine_abs_mean_sum
                    p_sw_lock_upper = v * fsw * tr_tf_s * scale * locked_abs_sum
                    case["switching_sensitivity"][vk]["sine"].append({
                        "fsw_hz": fsw,
                        "edge_scale": scale,
                        "p_sw_upper_w": p_sw_sine_upper,
                        "p_sw_half_sensitivity_w": p_sw_sine_upper / 2.0,
                    })
                    case["switching_sensitivity"][vk]["locked_vector"].append({
                        "fsw_hz": fsw,
                        "edge_scale": scale,
                        "p_sw_upper_total_w": p_sw_lock_upper,
                        "p_sw_hottest_fet_upper_w": 0.5 * v * fsw * tr_tf_s * scale * I_PHASE_PK_CANDIDATE_A,
                        "p_sw_half_sensitivity_total_w": p_sw_lock_upper / 2.0,
                        "p_sw_half_sensitivity_hottest_fet_w": 0.25 * v * fsw * tr_tf_s * scale * I_PHASE_PK_CANDIDATE_A,
                    })

            fsw_nom = 20_000.0
            p_sw_sine = v * fsw_nom * tr_tf_s * sine_abs_mean_sum
            p_sw_lock = v * fsw_nom * tr_tf_s * locked_abs_sum
            p_die_sine = per_fet_cond_sine + p_sw_sine / N_FET
            p_die_lock = max_lock_fet_cond + 0.5 * v * fsw_nom * tr_tf_s * I_PHASE_PK_CANDIDATE_A
            if "zth_1s_c_per_w" in mos:
                zth = zth_rc_at_s(mos["zth_rc_c_per_w"], mos["tau_rc_s"], T_PULSE_S)
                thermal_status = "RC-network extrapolation to 1 s from the manufacturer-provided fit; case held fixed, losses at the 175 C Rds screen. Model only."
            else:
                zth = mos["rth_jc_c_per_w"]
                thermal_status = "conservative asymptotic RthJC upper envelope only; datasheet transient plot does not provide a 1 s qualification point. Not 1 s proof."
            case["thermal_1s"][vk] = {
                "fsw_hz": fsw_nom,
                "edge_scale": 1.0,
                "zth_used_c_per_w": zth,
                "zth_method": thermal_status,
                "sine_total_bridge_loss_w": pcond_sine_total + p_sw_sine,
                "sine_per_fet_loss_w": p_die_sine,
                "sine_tj_screen_at_tc60_c": 60.0 + zth * p_die_sine,
                "sine_tc_limit_at_tj150_c": 150.0 - zth * p_die_sine,
                "sine_tc_limit_at_tj175_c": 175.0 - zth * p_die_sine,
                "locked_total_bridge_loss_w": pcond_lock_total + p_sw_lock,
                "locked_hottest_fet_loss_w": p_die_lock,
                "locked_tj_screen_at_tc60_c": 60.0 + zth * p_die_lock,
                "locked_tc_limit_at_tj150_c": 150.0 - zth * p_die_lock,
                "locked_tc_limit_at_tj175_c": 175.0 - zth * p_die_lock,
                "locked_phase_a_shunt_power_w": I_PHASE_PK_CANDIDATE_A**2 * R_SHUNT_OHM,
            }
        loss_cases[name] = case

    m_max = 2.0 / math.sqrt(3.0)
    cap_factor_pf1, cap_m_pf1 = capacitor_rms_factor_max(1.0)
    cap_factor_pf08, cap_m_pf08 = capacitor_rms_factor_max(0.8)
    cap_i = sine_irms * cap_factor_pf1
    cap_n = 6
    cap_each_uF = 900.0
    cap_esr_hot_mohm = 10.6
    cap_bank_esr = cap_esr_hot_mohm / 1000.0 / cap_n
    cap_bank_w = cap_i**2 * cap_bank_esr
    cap_bank_c_f = cap_n * cap_each_uF * 1e-6
    cap_energy = 0.5 * cap_bank_c_f * (V_BUS[1] ** 2 - V_BUS[0] ** 2)
    vesc_bulk_c = 4 * 220e-6
    vesc_energy = 0.5 * vesc_bulk_c * (V_BUS[1] ** 2 - V_BUS[0] ** 2)
    dc_link = {
        "rms_formula": "Ic_rms/Iphase_rms = sqrt(2*m*(sqrt(3)/(4*pi) + PF^2*(sqrt(3)/pi - 9*m/16)))",
        "m_definition": "Vphase_peak/(Vdc/2); linear SVPWM range 0..2/sqrt(3)",
        "pf_definition": "fundamental displacement power factor; sinusoidal phase current, ideal DC source and idealized PWM assumptions",
        "max_factor_pf1": cap_factor_pf1,
        "m_at_max_pf1": cap_m_pf1,
        "max_factor_pf0_8": cap_factor_pf08,
        "m_at_max_pf0_8": cap_m_pf08,
        "phase_rms_at_120a_peak_a": sine_irms,
        "worst_cap_rms_at_120a_peak_pf1_a": cap_i,
        "peh226ml3900qe4": {
            "count_candidate": cap_n,
            "capacitance_nominal_f": cap_bank_c_f,
            "single_cap_uF": cap_each_uF,
            "voltage_rating_v": 63.0,
            "ripple_current_rated_natural_convection_a_at_5khz_125c": 8.1,
            "ripple_current_rated_life_condition": "continuous operation at natural convection; 4000 h for Ø20/Ø18 case",
            "ripple_current_max_natural_convection_a_at_5khz_125c": 10.2,
            "ripple_current_max_life_condition": "maximum at natural convection; 2000 h for Ø20/Ø18 case",
            "ripple_current_max_with_heatsink_a_at_5khz_125c": 22.2,
            "natural_convection_bank_rated_ideal_share_a": 6 * 8.1,
            "natural_convection_bank_max_ideal_share_a": 6 * 10.2,
            "heatsink_bank_max_ideal_share_a": 6 * 22.2,
            "hot_esr_max_mohm_at_5_to_100khz_125_to_150c": cap_esr_hot_mohm,
            "bank_esr_ideal_parallel_ohm": cap_bank_esr,
            "bank_esr_loss_at_worst_cap_rms_w": cap_bank_w,
            "esr_loss_per_cap_equal_share_w": cap_bank_w / cap_n,
            "stored_energy_37_8_to_22_5v_j_nominal": cap_energy,
            "fraction_of_1800j_pulse": cap_energy / (P_MOTOR_W * T_PULSE_S),
            "ideal_c_needed_to_supply_1800j_from_37_8_to_22_5_f": 2 * P_MOTOR_W * T_PULSE_S / (V_BUS[1] ** 2 - V_BUS[0] ** 2),
        },
        "vesc6_author_bulk_reference": {
            "bulk_electrolytics": "4 x 220 uF / 63 V from C48-C51; only these bus bulk caps are counted here",
            "nominal_c_f": vesc_bulk_c,
            "stored_energy_37_8_to_22_5v_j": vesc_energy,
            "fraction_of_1800j_pulse": vesc_energy / (P_MOTOR_W * T_PULSE_S),
            "ceramic_parts_not_added_to_bulk_total": "six 4.7 uF parts near the switching stage; schematic inventory alone is not used to count all six as DC-bus-parallel capacitance",
        },
        "minimum_cap_constraints": [
            "ripple-current thermal rating at actual ambient, frequency spectrum, mounting and 1 s pulse duration",
            "capacitance from the actual PWM ripple charge integral: C >= max|integral(iC dt)|/allowed_deltaV; not I*dt without waveform",
            "ESR ripple: deltaV_ESR approximately delta_i*ESR_eq; P_ESR=Ic_rms^2*ESR_eq; share/temperature must be verified",
            "ESL switching edge: deltaV_ESL approximately L_eq*di/dt; requires physical loop/parasitic and switching waveform",
        ],
    }

    shunts = {
        "source_topology": "three 0.5 mOhm / 6 W / 2512 phase shunts R24/R29/R32; author firmware marks HW_HAS_3_SHUNTS and HW_HAS_PHASE_SHUNTS",
        "phase_sense_gain_v_per_v": SHUNT_GAIN,
        "at_120a_phase_peak": {
            "shunt_peak_voltage_mv": 1000.0 * I_PHASE_PK_CANDIDATE_A * R_SHUNT_OHM,
            "amplifier_output_delta_v_at_gain20": I_PHASE_PK_CANDIDATE_A * R_SHUNT_OHM * SHUNT_GAIN,
            "sine_rms_current_a": sine_irms,
            "sine_resistor_power_w_per_phase": sine_irms**2 * R_SHUNT_OHM,
            "locked_phase_a_120a_dc_power_w": I_PHASE_PK_CANDIDATE_A**2 * R_SHUNT_OHM,
            "locked_phase_a_120a_dc_energy_1s_j": I_PHASE_PK_CANDIDATE_A**2 * R_SHUNT_OHM * T_PULSE_S,
            "locked_phase_b_c_60a_dc_power_w_each": (I_PHASE_PK_CANDIDATE_A / 2.0) ** 2 * R_SHUNT_OHM,
            "initial_nameplate_candidate_floor_w": 10.0,
            "selection_note": "10 W is a screening nameplate floor; a 6 W part can only be conditionally retained after its manufacturer 1 s pulse curve and measured temperature/current protection are checked.",
        },
        "hypothetical_single_0_5mohm_dc_bus_shunt_at_100a": {
            "drop_mv": 1000.0 * I_BUS_CANDIDATE_A * R_SHUNT_OHM,
            "dissipation_w": I_BUS_CANDIDATE_A**2 * R_SHUNT_OHM,
            "note": "not the VESC phase-shunt topology",
        },
        "limits": "60 mV x gain20 is signal swing only; amplifier REF/common-mode/ADC headroom and actual 1 s resistor pulse capability are not established by the schematic marking.",
    }

    result = {
        "meta": {
            "purpose": "per-board 1 s power-stage screening for 9S 5065 270KV hall BLDC; not hardware qualification",
            "input_interpretation": "main P/V calculations assume 1.8 kW at DC bus; motor-terminal AC interpretation is separately tabulated using candidate bridge losses",
            "direction": "motoring and regenerative current magnitudes use the same synchronous 6-MOS bridge loss screen; actual battery/DC bus energy acceptance is not demonstrated",
            "single_pulse_initial_condition": "one 1 s event; case temperature held at stated value, no repeated interval known",
            "limitations": [
                "motor phase resistance, inductance, pole-pair count, speed, power factor and actual PWM/gate waveform unknown",
                "MOSFET normalized Rds(T) figures are visually read typical graph factors multiplied by 25 C maximum Rds; hot absolute max is not guaranteed",
                "switching loss uses datasheet tr/tf under different test currents/voltages/gate resistors; sensitivity includes 0.5x/1x/2x edges and half-loss case",
                "Coss/Eoss, reverse recovery, deadtime/body-diode conduction, gate-driver switching waveforms, copper, shunts, PCB, connectors, source and clamp losses excluded from MOSFET total",
                "rated ID and pulse ID are not used as proof; case temperature must remain within specified design boundary over the pulse",
            ],
        },
        "power_and_current": {
            "one_board": {
                "p_electrical_w": P_MOTOR_W,
                "duration_s": T_PULSE_S,
                "energy_j": P_MOTOR_W * T_PULSE_S,
                "dc_bus_ideal_current_and_phase_minimum": phase_rms_min,
            },
            "two_identical_boards_simultaneous": {
                "power_w": P_MOTOR_W * N_BOARDS,
                "energy_j": P_MOTOR_W * T_PULSE_S * N_BOARDS,
                "bus_current_at_22_5v_a": P_MOTOR_W * N_BOARDS / V_BUS[0],
                "bus_current_at_37_8v_a": P_MOTOR_W * N_BOARDS / V_BUS[1],
            },
            "motor_terminal_ac_power_case": {
                "p_ac_per_motor_w": P_MOTOR_W,
                "dc_power_estimate_formula": "Pdc = Pac + Pbridge; candidate 120 A phase-current bridge-loss screen only, other losses omitted",
                "per_candidate_and_bus_voltage": {},
            },
            "candidate_boundaries_are_independent_not_frozen": {
                "phase_peak_a": I_PHASE_PK_CANDIDATE_A,
                "dc_bus_a": I_BUS_CANDIDATE_A,
                "max_dc_power_by_dc_current_w": {str(v): v * I_BUS_CANDIDATE_A for v in V_BUS},
                "max_ac_power_by_phase_current_at_full_linear_svpwm_unity_pf_w": {
                    str(v): math.sqrt(3.0) / 2.0 * v * I_PHASE_PK_CANDIDATE_A for v in V_BUS
                },
                "combined_power_cap_w": {
                    str(v): min(v * I_BUS_CANDIDATE_A, math.sqrt(3.0) / 2.0 * v * I_PHASE_PK_CANDIDATE_A)
                    for v in V_BUS
                },
                "note": "120 A phase peak and 100 A DC bus are separate candidate limits; at full modulation, unity PF, 120 A phase peak corresponds to about 103.9 A DC current.",
            },
            "regeneration": {
                "symmetric_power_stage_screen_w": -P_MOTOR_W,
                "energy_requested_per_board_j": P_MOTOR_W * T_PULSE_S,
                "bridge_losses_same_magnitude_assumption": True,
                "source_acceptance_proven": False,
                "note": "no separate energy source or storage is assumed; the local DC-link is not an energy sink for 1.8 kJ.",
            },
        },
        "mosfet_candidates": loss_cases,
        "dc_link": dc_link,
        "shunts": shunts,
    }

    for name, mos_case in loss_cases.items():
        for v in V_BUS:
            k = str(v)
            bridge_loss = mos_case["thermal_1s"][k]["sine_total_bridge_loss_w"]
            lock_loss = mos_case["thermal_1s"][k]["locked_total_bridge_loss_w"]
            result["power_and_current"]["motor_terminal_ac_power_case"]["per_candidate_and_bus_voltage"].setdefault(name, {})[k] = {
                "dc_bus_power_with_sine_candidate_losses_w": P_MOTOR_W + bridge_loss,
                "dc_bus_current_with_sine_candidate_losses_a": (P_MOTOR_W + bridge_loss) / v,
                "dc_bus_power_with_locked_vector_loss_w": P_MOTOR_W + lock_loss,
                "dc_bus_current_with_locked_vector_loss_a": (P_MOTOR_W + lock_loss) / v,
            }

    OUT.write_text(json.dumps(round_tree(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"1 s energy per board: {P_MOTOR_W*T_PULSE_S:.0f} J; two boards: {P_MOTOR_W*T_PULSE_S*N_BOARDS:.0f} J")
    print(f"120 A sine phase RMS: {sine_irms:.2f} A; max DC-link RMS factor PF=1: {cap_factor_pf1:.3f} at m={cap_m_pf1:.3f}; Icap={cap_i:.2f} A")
    for name in MOSFETS:
        print(f"{name}: see JSON for hot-Rds, switching, loss and 1 s thermal screens")


if __name__ == "__main__":
    main()
