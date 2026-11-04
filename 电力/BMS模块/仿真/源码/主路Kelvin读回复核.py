#!/usr/bin/env python3
"""Budget the readback tap without changing the precharge comparator path."""

import argparse
import json
import math
from pathlib import Path

from 预充供电与主路实证复核 import BOARD, read_candidate


def interface():
    frozen = read_candidate()
    path = BOARD / "归档/精简预充合闸候选/精简候选_P1.enet"
    parts = json.loads(path.read_bytes())["components"]
    core = json.loads((BOARD / "检查/主控头引脚核对.json").read_bytes())["pin_map"]
    routes = {
        "P1": {10: ("PA8", "KELVIN_I2C2_SDA"), 11: ("PA9", "KELVIN_I2C2_SCL")},
        "P2": {15: ("PA3", "KELVIN_DRDY_N")},
    }
    for header, pins in routes.items():
        for pin, (gpio, _) in pins.items():
            assert parts[f"J_CORE_{header}"]["pinInfoMap"][str(pin)]["net"] == ""
            assert next(item["mcu_gpio"] for item in core[header] if item["pin"] == pin) == gpio
    taps = {
        "U_PRE_SENSE": {"1": "PRE_SENSE_OUT", "3": "BATT+_FUSED", "4": "PACK+",
                        "5": "PRE_REF_BUF", "6": "VOUT_5V"},
        "U_PRE_REF": {"1": "PRE_REF_BUF"},
        "U_PRE_AND": {"3": "PRE_DIFF_IN", "6": "PRE_RAW", "11": "PRE_DONE"},
        "U_AND2": {"4": "MAIN_CLR_N", "6": "MAIN_OR"},
        "U_SEQ": {"1": "MAIN_CLR_N", "5": "FET_RELEASE"},
        "U1": {"30": "DFETOFF"},
    }
    for ref, pins in taps.items():
        for pin, net in pins.items():
            assert parts[ref]["pinInfoMap"][pin]["net"] == net
    return {"frozen_sha256": frozen["sha256"], "taps": taps, "NC_header_routes": routes}


def screening(stage_b):
    # A conditional budget. Datasheet characterization at other operating
    # points and the new board-level limits still require qualification.
    g_min = .2 * .999
    supply_max = 5.25
    ref_max = supply_max * 1.0135 / (1.0135 + .9865)
    r_tap_max = 499000 * 1.01
    r_tap_min = 499000 * .99
    tap_tau = r_tap_max * 20e-12
    sense_envelope = .260
    trip_reading = stage_b["readback_threshold_V"]
    error_limit = stage_b["differential_total_error_budget_V"]
    input_errors = {
        "INA600A_offset": .005 / g_min,
        "INA600A_CMRR_91dB": 37.8 / 10**(91 / 20) / g_min,
        "INA600A_ref_transfer_reservation": ref_max * .0005 / g_min,
        "INA600A_PSRR_reservation": .75 * 15e-6 / g_min,
        "INA600A_signal_gain_and_drift": sense_envelope * .001,
        "two_buffer_errors": .000040 / g_min,
        "two_buffer_input_bias_and_board_leakage": 2 * r_tap_max * 300e-12 / g_min,
        "ADC_additive_including_protection_resistors": .000250 / g_min,
        "ADC_gain_reference_drift": sense_envelope * .0055,
        "tap_perturbation_of_legacy_differential_nodes": .000100 / g_min,
        "aperture_skew_at_0p30V_per_s": .0075 * .30,
    }
    total = sum(input_errors.values())
    assert total < error_limit
    assert stage_b["healthy_indicated_drop_ceiling_V"] < trip_reading
    assert trip_reading < stage_b["bad_indicated_drop_floor_V_including_precharge_parallel"]
    return {
        "input_referred_error_contributions_V": input_errors,
        "input_referred_total_V": total,
        "remaining_budget_V": error_limit - total,
        "INA_nominal_delta_at_80mV_V": .2 * trip_reading,
        "reference_upper_V": ref_max,
        "tap_fault_current_ceiling_per_node_A": supply_max / r_tap_min,
        "tap_RC_s_at_20pF": tap_tau,
        "tap_RC_only_settle_s_for_5p25V_to_0p1mV": tap_tau * math.log(supply_max / .000100),
        "ADC_full_scale_differential_V": .256,
        "ADC_quantization_input_referred_half_LSB_V": .256 / 32768 / 2 / g_min,
        "ADC_conversion_period_s_at_slowest_oscillator": .001 / .98,
        "independent_bus_3p3k_pullup_rise_s_at_100pF": .8473 * 3300 * 1.01 * 100e-12,
        "qualified_max_aperture_skew_s": .0075,
        "qualified_max_gap_slew_V_per_s": .30,
        "allocated_actual_stop_chain_s": [.0001, .0001, .0008],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    previous = json.loads((BOARD / "仿真/结果/主路两阶段取舍复核.json").read_bytes())
    wiring = interface()
    assert wiring["frozen_sha256"] == previous["netlist_evidence"]["BMS_frozen_sha256"]
    result = {
        "status": "HOLD_CONDITIONAL_READBACK_BUDGET_INDEPENDENT_HARD_STOP_OPEN",
        "interface": wiring,
        "readback": screening(previous["handoff"]["B"]),
        "readback_only_parts": {"shared_buffered_tap": 12, "independent_INA_and_reference": 14},
        "sources": {
            "INA600A": "https://www.ti.com/lit/gpn/ina600",
            "OPA2333": "https://www.ti.com/lit/ds/symlink/opa2333.pdf",
            "ADS112C04": "https://www.ti.com/lit/ds/symlink/ads112c04.pdf",
            "STM32G431": "https://www.st.com/resource/en/datasheet/stm32g431cb.pdf",
            "TLV9032": "https://www.ti.com/lit/gpn/tlv9032",
            "BQ76952": "https://www.ti.com/lit/gpn/bq76952",
            "BQ76952_TRM": "https://www.ti.com/lit/pdf/sluuby2",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
