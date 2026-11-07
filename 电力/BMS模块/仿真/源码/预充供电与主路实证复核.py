#!/usr/bin/env python3
"""Screen the 5 V interlock and observability of the main path in the compact candidate."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from 精简预充候选复核 import BOARD, differential, timing


def read_candidate():
    baseline = BOARD / "检查/预充合闸候选基线"
    enet = baseline / "精简候选_P1.enet"
    data = enet.read_bytes()
    record = json.loads((baseline / "官方读回与针网差异.json").read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    assert digest == record["deliverable_sha256"][enet.name]
    parts = json.loads(data)["components"]
    expected = {
        "U_PRE_5V_POR": {"2": "PRE_SENSE5_OK", "3": "VOUT_5V"},
        "U_PRE_AND": {"8": "PRE_RAILS_OK", "9": "POR_OK_N", "10": "PRE_SENSE5_OK"},
        "U_PRE_ARM": {"1": "PRE_REQ", "6": "PRE_RUN_OK"},
        "U_SEQ": {"1": "MAIN_CLR_N", "5": "FET_RELEASE", "11": "PRE_REQ"},
        "J_CORE_P1": {"15": "TMR_ACTIVE", "22": "MAIN_REQ"},
        "J_CORE_P2": {"5": "BQ_I2C_SCL", "6": "BQ_I2C_SDA", "14": "MAIN_PROVEN", "17": "PRE_DONE"},
        "U1": {"18": "SRP", "20": "SRN", "42": "BQ_PACK", "43": "DSG",
               "45": "CHG", "48": "VC16_FILT"},
        "R_PACK": {"1": "PACK+", "2": "BQ_PACK"},
        "R_VC16": {"1": "TAP_B9", "2": "VC16_FILT"},
        "RSHUNT": {"1": "B0", "2": "PACK-"},
    }
    for ref, pins in expected.items():
        for pin, net in pins.items():
            assert parts[ref]["pinInfoMap"][pin]["net"] == net
    sense_endpoints = sorted(
        f"{ref}.{pin}" for ref, part in parts.items() for pin, info in part["pinInfoMap"].items()
        if info.get("net") == "PRE_SENSE_OUT"
    )
    assert sense_endpoints == ["U_PRE_CMP.3", "U_PRE_CMP.6", "U_PRE_SENSE.1"]
    return {"sha256": digest, "pin_nets": expected, "sense_endpoints": sense_endpoints}


def power_window():
    # TPS389001 uses independent ±1% threshold bounds. The SENSE current is an
    # application limit to verify because the 100 nA table entry is at 5 V.
    sense_corners = (.9965, 1.0035)
    sense_current = (-100e-9, 100e-9)

    def rail_bounds(thresholds):
        values = []
        for v, top, bottom, bias in itertools.product(thresholds, sense_corners, sense_corners, sense_current):
            upper = 30100 * top
            lower = 10000 * bottom
            values.append(v * (1 + upper / lower) + bias * upper)
        return [min(values), max(values)]

    fall = rail_bounds((1.15 * .99, 1.15 * 1.01))
    rise_min, rise_max = rail_bounds((
        min(1.157 * .99, 1.15 * .99 * 1.00325),
        max(1.157 * 1.01, 1.15 * 1.01 * 1.00825),
    ))
    assert 4.5 < fall[0] < rise_max < 4.75

    # The push-pull Schmitt buffer isolates the ordinary AND input from the
    # open-drain node. Check DC corners separately from edge qualification.
    logic_corners = (.9865, 1.0135)
    node_current = 5.25e-6

    def node_values(rail, bias):
        for up, down in itertools.product(logic_corners, repeat=2):
            pull_up, pull_down = 15000 * up, 15000 * down
            yield (rail / pull_up + bias) / (1 / pull_up + 1 / pull_down)

    reset_sink_max = 5.25 / (15000 * min(logic_corners)) + node_current
    assert reset_sink_max < .0004

    base = differential()
    normal_cases = {}
    for rail in (4.5, 4.75):
        # The original tap ratio is unchanged. Include the rail during
        # the turn-off response, all the way down to the INA600A floor.
        tap_min = base["tap_to_center_V"][0] * rail / base["screened_supply_V"][0]
        inner = (tap_min - sum(base["output_error_contributions_V"].values())) / .2002
        screened = timing({"input_inner_to_guarantee_reading_pass_V": inner})
        normal_cases[str(rail)] = {
            "inner_gap_V": inner,
            "charge_time_s": screened["case_at_allocated_current"]["time_to_target_s"],
            "remaining_time_s": screened["remaining_charge_time_at_allocated_current_s"],
            "constant_current_limit_A": screened["load_limit_442ohm_target_A"],
            "capacitance_limit_F_at_allocated_current": screened["cap_limit_at_allocated_current_F"],
        }
    droop = fall[0] - 4.5
    return {
        "status": "CONDITIONAL_STATIC_ONLY_DYNAMIC_OPEN",
        "as_built_4p63_rising_max_V": base["supervisor_worst_rising_release_threshold_V"],
        "pin_compatible_4p55_falling_min_V": 4.55 * .98,
        "pin_compatible_4p55_rising_max_V": 4.55 * 1.02 * 1.015,
        "falling_threshold_V": fall,
        "rising_threshold_V": [rise_min, rise_max],
        "rising_margin_at_4p75_V": 4.75 - rise_max,
        "falling_drop_plus_error_limit_V": droop,
        "turnoff_example_s_at_1V_per_ms_after_10mV_error": (droop - .010) / 1000,
        "ct_charge_only_s": [.18e-6 * 1.17 / 1.40e-6, .24e-6 * 1.29 / .85e-6],
        "open_drain_logic_high_min_at_4p75_V": min(node_values(4.75, -node_current)),
        "passive_logic_node_max_at_1p5_V": max(node_values(1.5, node_current)),
        "reset_sink_upper_A": reset_sink_max,
        "allocation_A": screened["design_allocation_A"],
        "model_available_charge_time_s": screened["effective_charge_time_s"],
        "model_identified_cap_upper_F": screened["identified_cap_upper_F"],
        "normal_timing_model_by_rail_V": normal_cases,
    }, screened


def main_path_proof(model):
    allocation = model["design_allocation_A"]
    cold_precharge_resistance = 37.8 / model["short_peak_A"]
    return {
        "status": "FAIL_EXISTING_SIGNALS",
        "common_observation_at_equalized_zero_load": {
            "main_FET_drive_requested": True, "PRE_DONE": True, "TMR_ACTIVE": True,
            "stack_and_pack_ideal_V": [37.8, 37.8], "total_shunt_current_A": 0,
            "consistent_physical_main_path_states": ["low_resistance", "open_with_precharge_on"],
        },
        "precharge_only_steady_drop_V_at_allocated_current": [
            allocation * cold_precharge_resistance, allocation * model["hot_model_r_ohm"],
        ],
        "example_2mohm_main_path_drop_V_at_allocated_current": allocation * .002,
        "common_0p5mohm_shunt_drop_V_at_allocated_current": allocation * .0005,
        "BQ_worst_case_absolute_difference_error_V_minus40_to85C": .5 + .5,
        "fixture_reference_only": model["fixture_discrimination_only"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidate = read_candidate()
    window, model = power_window()
    result = {
        "status": "ISOLATED_HOLD",
        "candidate": candidate,
        "power_window": window,
        "main_path_proof": main_path_proof(model),
        "sources": {
            "TPS3890": "https://www.ti.com/lit/ds/symlink/tps3890.pdf",
            "TLV809E": "https://www.ti.com/lit/ds/symlink/tlv809e.pdf",
            "INA600": "https://www.ti.com/lit/gpn/ina600",
            "SN74LVC08A": "https://www.ti.com/lit/ds/symlink/sn74lvc08a.pdf",
            "SN74LVC1G17": "https://www.ti.com/lit/ds/symlink/sn74lvc1g17.pdf",
            "BQ76952": "https://www.ti.com/lit/gpn/bq76952",
            "BQ76952_TRM": "https://www.ti.com/lit/pdf/sluuby2",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
