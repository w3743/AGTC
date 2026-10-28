#!/usr/bin/env python3
"""Reproducible conditional thermal screening for the selected IPT015N10N5 bridge.

Run from any directory with Python 3:
    python selected_ipt015_thermal_screen.py

The script writes selected_ipt015_thermal_screen.json beside itself. This is a
steady-state R-theta screening calculation, not a transient thermal validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


MPN = "IPT015N10N5ATMA1"
VDS_RATING_V = 100.0
RDS_MAX_AT_6V_OHM = 0.002
HOT_FACTOR = 2.5
RDS_HOT_SCREEN_OHM = RDS_MAX_AT_6V_OHM * HOT_FACTOR
PHASE_PEAK_A = 120.0
F_SWITCH_HZ = 20_000.0
T_CASE_MAX_C = 60.0
T_J_START_MAX_C = 60.0
R_TH_JC_MAX_C_PER_W = 0.4
ONE_SECOND_S = 1.0
BUS_VOLTAGES_V = [22.5, 32.4, 37.8]
EDGE_SUM_NS_CASES = [300.0, 500.0, 1000.0]
N_PHASES = 3


def round_value(value: float, digits: int = 6) -> float:
    return round(value, digits)


def switching_loss_bridge_upper_w(vbus_v: float, edge_sum_ns: float) -> float:
    """Conservative all-six-device sinusoidal overlap upper screen (6/pi)."""
    edge_sum_s = edge_sum_ns * 1e-9
    return (6.0 / math.pi) * vbus_v * PHASE_PEAK_A * edge_sum_s * F_SWITCH_HZ


def switching_loss_hottest_device_w(vbus_v: float, edge_sum_ns: float) -> float:
    """Single-device overlap term used with the 120 A full-cycle conduction bound."""
    edge_sum_s = edge_sum_ns * 1e-9
    return vbus_v * PHASE_PEAK_A / 2.0 * edge_sum_s * F_SWITCH_HZ


def calculate() -> dict:
    p_conduction_bridge_w = 1.5 * PHASE_PEAK_A**2 * RDS_HOT_SCREEN_OHM
    p_conduction_hottest_device_w = PHASE_PEAK_A**2 * RDS_HOT_SCREEN_OHM

    bus_cases = []
    for vbus_v in BUS_VOLTAGES_V:
        p_switch_bridge_w = switching_loss_bridge_upper_w(vbus_v, 500.0)
        p_bridge_w = p_conduction_bridge_w + p_switch_bridge_w
        p_switch_device_w = switching_loss_hottest_device_w(vbus_v, 500.0)
        p_hottest_device_w = p_conduction_hottest_device_w + p_switch_device_w
        tj_upper_c = T_CASE_MAX_C + p_hottest_device_w * R_TH_JC_MAX_C_PER_W
        bus_cases.append(
            {
                "vbus_v": vbus_v,
                "input_current_at_1p8kw_a": round_value(1800.0 / vbus_v),
                "bridge_conduction_w": round_value(p_conduction_bridge_w),
                "bridge_switching_upper_w_500ns": round_value(p_switch_bridge_w),
                "bridge_total_upper_w_500ns": round_value(p_bridge_w),
                "bridge_energy_j_1s_500ns": round_value(p_bridge_w * ONE_SECOND_S),
                "hottest_device_conduction_w": round_value(p_conduction_hottest_device_w),
                "hottest_device_switching_w_500ns": round_value(p_switch_device_w),
                "hottest_device_total_w_500ns": round_value(p_hottest_device_w),
                "hottest_device_energy_j_1s_500ns": round_value(
                    p_hottest_device_w * ONE_SECOND_S
                ),
                "tj_upper_c_under_case_and_start_assumptions": round_value(tj_upper_c),
                "tj_margin_to_175c_c_under_assumptions": round_value(175.0 - tj_upper_c),
            }
        )

    edge_sensitivity = []
    for vbus_v in (22.5, 37.8):
        for edge_sum_ns in EDGE_SUM_NS_CASES:
            p_sw_bridge = switching_loss_bridge_upper_w(vbus_v, edge_sum_ns)
            p_total_bridge = p_conduction_bridge_w + p_sw_bridge
            p_sw_device = switching_loss_hottest_device_w(vbus_v, edge_sum_ns)
            p_total_device = p_conduction_hottest_device_w + p_sw_device
            edge_sensitivity.append(
                {
                    "vbus_v": vbus_v,
                    "tr_plus_tf_ns": edge_sum_ns,
                    "bridge_switching_upper_w": round_value(p_sw_bridge),
                    "bridge_total_upper_w": round_value(p_total_bridge),
                    "bridge_energy_j_1s": round_value(p_total_bridge * ONE_SECOND_S),
                    "hottest_device_switching_w": round_value(p_sw_device),
                    "hottest_device_total_w": round_value(p_total_device),
                    "hottest_device_tj_upper_c": round_value(
                        T_CASE_MAX_C + p_total_device * R_TH_JC_MAX_C_PER_W
                    ),
                }
            )

    result = {
        "title": "Selected IPT015N10N5ATMA1 conditional thermal screen",
        "status": "calculation_screen_only_not_1s_thermal_acceptance",
        "device": {
            "manufacturer": "Infineon",
            "mpn": MPN,
            "quantity_per_bridge": 6,
            "package": "PG-HSOF-8 / TOLL",
            "vds_rating_v": VDS_RATING_V,
            "rds_max_at_6v_25c_ohm": RDS_MAX_AT_6V_OHM,
            "rds_hot_screen_factor": HOT_FACTOR,
            "rds_hot_screen_ohm": RDS_HOT_SCREEN_OHM,
            "rtheta_jc_max_c_per_w": R_TH_JC_MAX_C_PER_W,
            "tj_max_c": 175.0,
            "official_datasheet": "https://www.infineon.com/assets/row/public/documents/24/49/infineon-ipt015n10n5-datasheet-en.pdf",
            "datasheet_basis": "Rev. 2.4, Table 4: RDS(on) max 2.0 mOhm at VGS=6 V, ID=75 A; Table 3: RthetaJC max 0.4 C/W; maximum Tj 175 C.",
        },
        "inputs": {
            "phase_peak_a": PHASE_PEAK_A,
            "switching_frequency_hz": F_SWITCH_HZ,
            "tr_plus_tf_nominal_ns": 500.0,
            "tr_plus_tf_sensitivity_ns": EDGE_SUM_NS_CASES,
            "bus_voltage_cases_v": BUS_VOLTAGES_V,
            "input_power_for_bus_current_a": 1800.0,
            "initial_tj_max_assumption_c": T_J_START_MAX_C,
            "case_temperature_max_throughout_pulse_assumption_c": T_CASE_MAX_C,
            "pulse_duration_s": ONE_SECOND_S,
        },
        "formulas": {
            "rds_hot_screen": "2 mOhm * 2.5 = 5 mOhm",
            "bridge_conduction": "Pcond_bridge = 3/2 * Iphase_pk^2 * Rds_hot_screen",
            "bridge_switching_upper": "Psw_bridge_upper = 6/pi * VBUS * Iphase_pk * (tr+tf) * fsw; conservative all-six-device sinusoidal overlap screen",
            "hottest_device_conduction": "Pcond_device_upper = Iphase_pk^2 * Rds_hot_screen; assumes 120 A continuously through one device for the conservative bound",
            "hottest_device_switching": "Psw_device_upper = VBUS * Iphase_pk/2 * (tr+tf) * fsw",
            "temperature_bound": "Tj_upper = Tc_max + (Pcond_device_upper + Psw_device_upper) * RthetaJC_max",
            "one_second_energy": "E_1s = P * 1 s",
            "bus_current": "Ibus = 1800 W / VBUS; ideal input-current arithmetic, no efficiency allowance",
        },
        "bridge_conduction_w": round_value(p_conduction_bridge_w),
        "hottest_device_conduction_w": round_value(p_conduction_hottest_device_w),
        "nominal_500ns_bus_cases": bus_cases,
        "edge_time_sensitivity": edge_sensitivity,
        "limitations": [
            "The 2.5 hot factor is a project screening assumption applied to the 6 V, 25 C maximum RDS(on); it is not an Infineon guaranteed value for this assembly.",
            "The 6/pi bridge switching term is a conservative all-six-device overlap upper screen; complementary commutation may reduce the sum. No reduction is used in the headline values.",
            "The hottest-device conduction term assumes one device carries the full 120 A continuously; it is a conservative local-hotspot screen, not sinusoidal per-device average sharing.",
            "The temperature calculation uses steady-state RthetaJC as a coarse 1 s screen. It is not the manufacturer's transient Zth curve or a measured 1 s junction-temperature result.",
            "It assumes starting Tj <= 60 C and case temperature Tc <= 60 C throughout the pulse. Case temperature, mounting, PCB copper, package interface and actual thermal path must be verified.",
            "It excludes diode reverse recovery, output-capacitance loss, gate-drive loss, ringing, deadtime, current ripple, unequal current sharing, control-induced recirculation and surrounding-component heating.",
            "The 1.8 kW bus current values are ideal arithmetic only; they do not establish motor operating point, inverter efficiency, bus-current limit or system acceptance.",
        ],
        "sources": [
            {
                "title": "Infineon IPT015N10N5 official datasheet, Rev. 2.4",
                "url": "https://www.infineon.com/assets/row/public/documents/24/49/infineon-ipt015n10n5-datasheet-en.pdf",
                "relevant_sections": "Table 3 thermal characteristics; Table 4 static characteristics; Tj maximum ratings.",
            }
        ],
    }

    # Guard the specifically requested nominal outputs against accidental drift.
    at_22v5 = bus_cases[0]
    at_37v8 = bus_cases[2]
    assert abs(at_22v5["bridge_total_upper_w_500ns"] - 159.566202) < 1e-6
    assert abs(at_37v8["bridge_total_upper_w_500ns"] - 194.631219) < 1e-6
    assert abs(at_37v8["hottest_device_total_w_500ns"] - 94.68) < 1e-9
    assert abs(at_37v8["tj_upper_c_under_case_and_start_assumptions"] - 97.872) < 1e-9
    return result


def main() -> None:
    output_path = Path(__file__).with_name("selected_ipt015_thermal_screen.json")
    result = calculate()
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    for case in result["nominal_500ns_bus_cases"]:
        print(
            f"VBUS={case['vbus_v']:g} V: Ibus@1.8kW={case['input_current_at_1p8kw_a']:.3f} A, "
            f"Pbridge={case['bridge_total_upper_w_500ns']:.3f} W, "
            f"Pdevice={case['hottest_device_total_w_500ns']:.3f} W, "
            f"Tj upper={case['tj_upper_c_under_case_and_start_assumptions']:.3f} C"
        )


if __name__ == "__main__":
    main()
