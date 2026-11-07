#!/usr/bin/env python3
"""Screen a separately switched diagnostic load before LOAD_POWER_VALID."""

import argparse
import json
import math
from pathlib import Path

from 精简预充候选复核 import BOARD, differential, timing
from 预充供电与主路实证复核 import read_candidate


def interface():
    frozen = read_candidate()
    enet = BOARD / "检查/预充合闸候选基线/精简候选_P1.enet"
    parts = json.loads(enet.read_bytes())["components"]
    core = json.loads((BOARD / "检查/主控头引脚核对.json").read_bytes())["pin_map"]["P2"]
    routes = {
        9: ("PB0", "CHG_VGS_OK"), 10: ("PB1", "DSG_VGS_OK"), 11: ("PA7", "TEST_REQ"),
        15: ("PA3", "DIFF_OUT_ADC"), 16: ("PA4", "DIFF_REF_ADC"), 18: ("PA2", "TEST_I_ADC"),
    }
    for pin, (gpio, _) in routes.items():
        assert parts["J_CORE_P2"]["pinInfoMap"][str(pin)]["net"] == ""
        assert next(item["mcu_gpio"] for item in core if item["pin"] == pin) == gpio
    taps = {
        "U_GATE1": {"4": "PRECHARGE_WINDOW"},
        "U_PRE_REF": {"1": "PRE_REF_BUF"},
        "Q_CHG1": {"1": "CHG_G1", "2": "BATT+_FUSED", "9": "FET_MID"},
        "Q_DSG1": {"1": "DSG_G1", "2": "PACK+", "9": "FET_MID"},
    }
    for ref, pins in taps.items():
        for pin, net in pins.items():
            assert parts[ref]["pinInfoMap"][pin]["net"] == net
    return {
        "frozen_netlist_sha256": frozen["sha256"],
        "confirmed_local_taps": taps,
        "P2_NC_WeAct_reference_only": routes,
        "existing_P2_14": "MAIN_PROVEN",
        "existing_P2_17": "PRE_DONE",
        "existing_P1_15": "TMR_ACTIVE",
    }


def switched_load(precharge_timeout):
    # Include resistor initial tolerance, temperature and allowance for life.
    # The 3-ohm relay ceiling is an application qualification target.
    load = (180 * .94, 180 * 1.06)
    shunt = (4.7 * .985, 4.7 * 1.015)
    v_min, v_max = 22.1, 37.8
    i_min = v_min / (load[1] + shunt[1] + 2 * 3)
    i_max = v_max / (load[0] + shunt[0])
    power_max = i_max**2 * load[0]
    hot_rating = 10 * (250 - 60) / (250 - 25)
    relay_current_at_60 = .4 - .004 * (60 - 25)
    relay_power_at_60 = .240 - .0024 * (60 - 25)
    led_i_min = (4.75 - 1.6 - .4) / (510 * 1.015)
    led_i_max = 5.25 / (510 * .985)
    assert .005 < led_i_min < led_i_max < .016
    assert i_min - .004 > .104
    assert power_max < hot_rating
    assert i_max < relay_current_at_60
    assert i_max**2 * 3 < relay_power_at_60
    return {
        "load_and_shunt_resistance_ohm": [load, shunt],
        "pulse_bus_V": [v_min, v_max],
        "test_current_A": [i_min, i_max],
        "branch_shunt_voltage_V": [
            v_min * shunt[0] / (load[1] + shunt[0] + 6),
            v_max * shunt[1] / (load[0] + shunt[1]),
        ],
        "LED_drive_A": [led_i_min, led_i_max],
        "load_max_W": power_max,
        "load_reference_rating_at_60C_W": hot_rating,
        "each_relay_max_screened_W": i_max**2 * 3,
        "each_relay_derated_absolute_max_at_60C_W": relay_power_at_60,
        "normal_40ms_load_energy_J": power_max * .040,
        "normal_load_energy_with_2ms_each_edge_J": power_max * .044,
        "stuck_request_or_switch_load_energy_before_tail_J": power_max * precharge_timeout,
        "whole_branch_bypassed_independent_of_both_switches": "REJECT; no branch isolation",
    }


def separation(model, current_upper):
    cap = model["identified_cap_upper_F"]
    allocated = model["design_allocation_A"]
    r_pre_min = 37.8 / model["short_peak_A"]
    r_pre_max = model["hot_model_r_ohm"]
    duration = model["fixture_discrimination_only"]["pulse_s"]
    current_lower = model["fixture_discrimination_only"]["pulse_A"]
    error = model["fixture_discrimination_only"]["increment_total_error_budget_V"]
    threshold = model["fixture_discrimination_only"]["increment_measured_acceptance_ceiling_V"]

    def step(r):
        return -math.expm1(-duration / (r * cap))

    # At positive initial gap, natural precharge decay offsets the stimulus.
    # For I*R > D0 the response grows with R, so use the lower R corner.
    worst_open = (current_lower * r_pre_min - .260) * step(r_pre_min)
    dynamic_error = .001
    healthy_r = .009
    healthy_baseline_tail = .260 * math.exp(-.005 / (healthy_r * cap))
    good_max = (current_upper + allocated) * healthy_r + healthy_baseline_tail
    assert good_max + error + dynamic_error < threshold < worst_open - error - dynamic_error

    # This weak path passes the old increment threshold at the low pack corner.
    weak_r = 1 / (1 / .1 + 1 / r_pre_max)
    weak_path_response = .110 * weak_r * step(weak_r)
    assert weak_path_response < threshold

    # Quantitative example for a future absolute Kelvin channel. Split R_eq
    # to bound the worst-case surviving -5mV baseline contribution.
    bad_r_eq = 1 / (1 / .030 + 1 / r_pre_min)
    breakpoint = .100
    bad_min = min(
        current_lower * bad_r_eq * step(bad_r_eq) - .005 * (1 - step(breakpoint)),
        current_lower * breakpoint * step(breakpoint) - .005,
    )
    absolute_error_plus_nuisance = .0004
    good_absolute_max = (current_upper + allocated) * healthy_r + .005 * (1 - step(healthy_r))
    assert good_absolute_max + absolute_error_plus_nuisance < .0025 < bad_min - absolute_error_plus_nuisance
    return {
        "precharge_resistance_ohm": [r_pre_min, r_pre_max],
        "main_open_true_increment_min_V": worst_open,
        "main_open_indicated_increment_min_V": worst_open - error - dynamic_error,
        "healthy_indicated_abs_increment_max_V": good_max + error + dynamic_error,
        "weak_100mohm_main_pass_counterexample_V": weak_path_response,
        "INA600A_offset_alone_referred_to_input_V": .005 / .2,
        "example_precision_absolute_check": {
            "known_healthy_main_max_ohm": healthy_r,
            "must_reject_main_ohm_at_or_above": .030,
            "absolute_baseline_V_max": .005,
            "indicated_healthy_end_max_V": good_absolute_max + absolute_error_plus_nuisance,
            "indicated_bad_end_min_V": bad_min - absolute_error_plus_nuisance,
            "absolute_acceptance_ceiling_V": .0025,
            "absolute_measurement_error_V": .0002,
            "source_and_model_error_V": .0002,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = timing(differential())
    stimulus = switched_load(model["TPL5111_DRVn_s"][1])
    stages_ms = {
        "MAIN_REQ_to_both_source_referenced_VGS_valid": 10,
        "settled_zero_current_baseline": 5,
        "switch_on_and_current_readback": 2,
        "verified_current_duration": 40,
        "switch_off_and_current_readback": 2,
        "decision_and_MAIN_PROVEN": 1,
    }
    assert sum(stages_ms.values()) == 60
    result = {
        "status": "HOLD_NO_VEHICLE_MAIN_PROOF",
        "interface": interface(),
        "load": stimulus,
        "separation": separation(model, stimulus["test_current_A"][1]),
        "handoff_stage_limits_ms": stages_ms,
        "single_switch_fault_containment": {
            "both_controllable_off": "normal",
            "either_open": "insufficient measured pulse, reject",
            "request_switch_stuck_on": "off-phase current fails; independent window opens arm switch",
            "arm_switch_stuck_on": "request switch still interrupts load; latent fault",
        },
        "sources": {
            "TLP3475W": "https://toshiba.semicon-storage.com/info/docget.jsp?did=153823&prodName=TLP3475W",
            "SN74LVC1G06": "https://www.ti.com/lit/ds/symlink/sn74lvc1g06.pdf",
            "CW010": "https://www.vishay.com/docs/30215/cw.pdf",
            "INA600A": "https://www.ti.com/lit/gpn/ina600",
            "STM32G431": "https://www.st.com/resource/en/datasheet/stm32g431cb.pdf",
            "BQ76952": "https://www.ti.com/lit/gpn/bq76952",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
