#!/usr/bin/env python3
"""Compare a limited handoff with a full-load authorization."""

import argparse
import hashlib
import json
from pathlib import Path

from 预充供电与主路实证复核 import BOARD, read_candidate


def netlist_evidence():
    bms = read_candidate()
    result_path = BOARD / "仿真/结果/主路车载脉冲复核.json"
    previous = json.loads(result_path.read_bytes())
    assert previous["interface"]["frozen_netlist_sha256"] == bms["sha256"]

    foc_path = BOARD.parent / "FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet"
    data = foc_path.read_bytes()
    foc_sha = hashlib.sha256(data).hexdigest()
    bus = json.loads((BOARD / "仿真/结果/跨板母线预充.json").read_bytes())
    assert foc_sha == bus["source_population"]["FOC_per_board"]["sha256"]
    parts = {part["props"]["Designator"]: part for part in json.loads(data)["components"].values()}
    foc_nets = {
        "J_LOAD_POWER_VALID": {"1": "LOAD_EN_BIAS_EXT", "2": "LOAD_POWER_VALID"},
        "U1": {"28": "BREAK_N", "33": "DRV_ENABLE", "47": "BUCK_RT_SD"},
        "Q_BUCK_CLAMP": {"1": "BUCK_CLAMP_BASE", "3": "BUCK_RT_SD"},
        "Q_BUCK_RELEASE": {"1": "BUCK_RELEASE_BASE", "3": "BUCK_CLAMP_BASE"},
    }
    for ref, pins in foc_nets.items():
        for pin, net in pins.items():
            assert parts[ref]["pinInfoMap"][pin]["net"] == net
    return {
        "BMS_frozen_sha256": bms["sha256"],
        "FOC_adopted_sha256": foc_sha,
        "FOC_pin_nets": foc_nets,
        "BMS_current_and_Kelvin_taps": bms["pin_nets"]["U1"],
        "BMS_unrouted_differential_endpoints": bms["sense_endpoints"],
    }, previous["separation"]["precharge_resistance_ohm"][0]


def rated_comparison():
    # Two 3-device parallel banks. The hot factor is a sensitivity only.
    mos_cold = 2 * .0028 / 3
    mos_hot = 2 * mos_cold
    resistances = {
        "cold_MOS_only": mos_cold,
        "hot_MOS_only_x2_sensitivity": mos_hot,
        "healthy_Kelvin_budget": .006,
        "fault_30mohm": .030,
        "fault_100mohm": .100,
        "separate_negative_shunt": .0005,
    }
    return {
        name: {
            "ohm": r,
            "60A": {"V": 60 * r, "W": 60**2 * r},
            "130A_comparison": {"V": 130 * r, "W": 130**2 * r},
        }
        for name, r in resistances.items()
    }


def handoff(precharge_min, rated):
    gap = .260
    voltage = 37.8
    hard_duration = .060
    current_error = .025
    nonpath_current = .005
    accept_current = .100
    precharge_max_current = gap / precharge_min
    open_indicated_max = precharge_max_current + nonpath_current + current_error
    assert open_indicated_max < accept_current
    main_current_min = accept_current - open_indicated_max
    trial_current_cap = .250
    stage_a = {
        "status": "REJECT_ON_FROZEN_INTERFACE_AND_LOAD_GUARANTEES",
        "true_precharge_only_current_ceiling_A": precharge_max_current,
        "other_shunt_current_budget_A": nonpath_current,
        "CC2_total_error_budget_A": current_error,
        "open_indicated_max_A": open_indicated_max,
        "indicated_acceptance_floor_A": accept_current,
        "healthy_true_shunt_current_for_worst_error_pass_A": accept_current + current_error,
        "main_branch_current_proven_min_A_at_acceptance": main_current_min,
        "largest_instantaneous_apparent_R_main_ohm_at_acceptance": gap / main_current_min,
        "dual_buck_ideal_input_A_at_rated_output_and_37p8V": 2 * 5 * .350 / voltage,
        "nominal_dual_buck_2p4W_ideal_input_A_at_37p8V": 2.4 / voltage,
        "current_cap_A_required_at_actual_aux_Bus_feed": trial_current_cap,
        "physical_timeout_s_required": hard_duration,
        "aux_supplied_energy_cap_J_if_current_and_time_enforced": voltage * trial_current_cap * hard_duration,
        "main_path_energy_J_if_gap_valid_plus_1ms_full_voltage_tail": (
            gap * trial_current_cap * hard_duration + voltage * trial_current_cap * .001
        ),
    }

    # Include the precharge branch in parallel even if it remains on.
    # A shorted bypass of R_PRE invalidates the model and the proof.
    good_r = rated["healthy_Kelvin_budget"]["ohm"]
    bad_r = rated["fault_30mohm"]["ohm"]
    bad_parallel_r = 1 / (1 / bad_r + 1 / precharge_min)
    min_i, max_i = 4.3, 5.0
    min_path_current = min_i - nonpath_current
    abs_error = .045
    pass_voltage = .080
    max_good_reading = good_r * max_i + abs_error
    min_bad_reading = bad_parallel_r * min_path_current - abs_error
    assert max_good_reading < pass_voltage < min_bad_reading
    equivalent_limit = (pass_voltage + abs_error) / min_path_current
    stage_b = {
        "status": "CONDITIONAL_SEPARATION_CURRENT_AND_TIMING_INTERLOCK_MISSING",
        "true_pack_current_range_A": [min_i, max_i],
        "true_Kelvin_parallel_path_current_floor_A": min_path_current,
        "CC2_error_budget_A": .25,
        "CC2_indicated_floor_A": min_i + .25,
        "differential_total_error_budget_V": abs_error,
        "readback_threshold_V": pass_voltage,
        "healthy_indicated_drop_ceiling_V": max_good_reading,
        "bad_indicated_drop_floor_V_including_precharge_parallel": min_bad_reading,
        "main_ohm_ceiling_at_pass": 1 / (1 / equivalent_limit - 1 / precharge_min),
        "min_current_A_for_existing_BQ_1V_gap_error_to_separate_6mohm_and_30mohm": 2 / (bad_r - good_r),
        "example_trial_timeout_s_requires_independent_hardware": hard_duration,
        "energy_J_with_current_limit_and_timeout_only": voltage * max_i * hard_duration,
        "energy_J_with_260mV_guard_and_1ms_full_voltage_tail": gap * max_i * hard_duration + voltage * max_i * .001,
        "voltage_error_ohm_at_60A_130A_excluding_current_error": [abs_error / 60, abs_error / 130],
    }
    return {"A": stage_a, "B": stage_b}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evidence, precharge_min = netlist_evidence()
    rated = rated_comparison()
    result = {
        "status": "HOLD_REJECT_ZERO_NEW_HARDWARE",
        "netlist_evidence": evidence,
        "rated_reference_only": rated,
        "handoff": handoff(precharge_min, rated),
        "minimum_new_online_observation": "protected synchronous BATT+_FUSED minus PACK+ Kelvin voltage",
        "sources": {
            "BQ76952": "https://www.ti.com/lit/gpn/bq76952",
            "BQ76952_TRM": "https://www.ti.com/lit/pdf/sluuby2",
            "DRV8353": "https://www.ti.com/lit/gpn/drv8353",
            "LM5008A": "https://www.ti.com/lit/ds/symlink/lm5008a.pdf",
            "INA600A": "https://www.ti.com/lit/gpn/ina600",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
