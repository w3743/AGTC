#!/usr/bin/env python3
"""Screen one LM5069-1 precharge branch against the frozen resistor candidate."""

import argparse
import json
import math
from pathlib import Path

from 预充供电与主路实证复核 import BOARD, read_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frozen = read_candidate()
    prior = json.loads((BOARD / "仿真/结果/精简预充边界与分阶段预算.json").read_bytes())
    timing = prior["timing_and_allocation"]
    cap = timing["identified_cap_upper_F"]
    target_gap = prior["differential"]["input_inner_to_guarantee_reading_pass_V"]
    pack = 37.8
    load = .100
    r_hot = .30
    r_sense = [.068 * .985, .068 * 1.015]
    cc_lo, cc_hi = .0485 / r_sense[1], .0615 / r_sense[0]
    knee_gap = r_hot * cc_lo
    assert load * r_hot < target_gap < knee_gap
    t_cc = cap * (pack - knee_gap) / (cc_lo - load)
    t_tail = cap * r_hot * math.log((knee_gap - load * r_hot) / (target_gap - load * r_hot))
    ct_required_min = t_cc * 120e-6 / 3.76
    ct_lo, ct_hi = 170e-6, 200e-6
    t_insertion = [ct_lo * 3.76 / 8e-6, ct_hi * 4.16 / 3e-6]
    t_fault = [ct_lo * 3.76 / 120e-6, ct_hi * 4.16 / 51e-6]
    assert t_insertion[0] > timing["TPL5111_DRVn_s"][1]
    assert t_fault[0] > t_cc

    r_pre = [pack / timing["short_peak_A"], timing["hot_model_r_ohm"]]
    r_probe = [r_pre[0] + 2.2 * .95, r_pre[1] + 2.2 * 1.05 + .20]
    probe_i = [22.1 / r_probe[1], pack / r_probe[0]]
    probe_time = .040
    main_i_floor = 1.70 - .100 - .020 - .005
    healthy_drop = probe_i[1] * .006
    weak_drop = main_i_floor * .030
    assert healthy_drop + .010 < .035 < weak_drop - .010
    result = {
        "status": "HOLD_REJECT_LM5069_COLD_INSERTION_TIMER",
        "frozen_sha256": frozen["sha256"],
        "frozen_reference": {
            "allocated_noncapacitive_A": timing["design_allocation_A"],
            "constant_load_model_limit_A": timing["load_limit_442ohm_target_A"],
            "precharge_window_s": timing["TPL5111_DRVn_s"],
            "usable_charge_s": timing["effective_charge_time_s"],
            "short_A": timing["short_peak_A"],
            "short_energy_before_tail_J": timing["short_single_energy_J_before_tail"],
        },
        "linear_precharge_conditional": {
            "charge_C_upper_F": cap,
            "known_dual_FOC_resistive_nominal_A_only": .00204,
            "noncapacitive_comparison_A_not_measured": load,
            "required_ideal_current_A": cap * (pack - target_gap) / timing["effective_charge_time_s"] + load,
            "LM5069_current_limit_A_at_spec_conditions": [cc_lo, cc_hi],
            "assumed_hot_dropout_path_ohm": r_hot,
            "CC_and_dropout_s": [t_cc, t_tail],
            "charge_s_excluding_insertion_and_power_limiting": t_cc + t_tail,
            "effective_CT_minimum_for_CC_F": ct_required_min,
            "effective_CT_maximum_for_75ms_worst_insertion_F": .075 * 3e-6 / 4.16,
            "effective_CT_screen_F": [ct_lo, ct_hi],
            "PWR_330kohm_nominal_model_W_at_37p8V": (330000 / 130000 + .00118 * pack) / .068,
            "cold_insertion_s": t_insertion,
            "internal_fault_time_s": t_fault,
            "short_regulated_upper_W": pack * cc_hi,
            "short_J_with_independent_long_timer_excluding_tail": pack * cc_hi * timing["TPL5111_DRVn_s"][1],
            "breaker_threshold_A_not_peak": [.080 / r_sense[1], .130 / r_sense[0]],
            "SOA_case_rise_budget_at_60C_ambient_and_100C_case_degC_per_W": 40 / (pack * cc_hi),
        },
        "probe_reusing_R_PRE_conditional": {
            "probe_duration_s": probe_time,
            "series_resistance_ohm": r_probe,
            "probe_current_A": probe_i,
            "probe_energy_J": pack * probe_i[1] * probe_time,
            "open_bus_drop_V_at_22p24V_and_40ms": 22.24 * -math.expm1(-probe_time / (r_probe[1] * cap)),
            "main_current_floor_A_at_1p70A_BQ_readback": main_i_floor,
            "weak_path_max_R_using_260mV_comparator_only_ohm": .260 / main_i_floor,
            "healthy_readback_upper_V_at_10mV_error": healthy_drop + .010,
            "bad_readback_lower_V_at_10mV_error": weak_drop - .010,
            "example_absolute_readback_threshold_V": .035,
        },
        "count_per_vehicle_for_screened_power_and_control": {
            "precharge_regulator_added": 9,
            "precharge_isolation_logic_added": 4,
            "probe_switch_series_resistor_and_enable_gate_added": 13,
            "R_PRE_relocated": 1,
            "net_added_before_measurement_and_60ms_probe_watchdog": 26,
            "price": "not verified",
        },
        "sources": {
            "LM5069": "https://www.ti.com/lit/ds/symlink/lm5069.pdf",
            "CSD19536KTT": "https://www.ti.com/lit/gpn/csd19536ktt",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
