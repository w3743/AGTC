#!/usr/bin/env python3
"""Screen the compact PRE_DONE candidate and its cross-board acceptance contract."""

import argparse
import itertools
import json
import math
from pathlib import Path

from 预充合闸候选复核 import logic_cases
from 跨板母线预充 import at_constant_current


BOARD = Path(__file__).resolve().parents[2]


def differential():
    tap = []
    corners = (.9865, 1.0135)
    # TLV809E 4.63 V option has ±2% falling threshold and ≤1.5% hysteresis.
    falling_min = 4.63 * .98
    rising_max = 4.63 * 1.02 * 1.015
    for rail, top, hi, lo, bottom in itertools.product((falling_min, 5.25), *([corners] * 4)):
        total = 33200 * (top + bottom) + 442 * (hi + lo)
        tap.extend((rail * 442 * hi / total, rail * 442 * lo / total))

    # Sum signs independently. CM/PS errors use the full excitation screen.
    output_errors = {
        "INA600A_full_temperature_offset": .005,
        "INA600A_common_mode_91dB": 37.8 / 10**(91 / 20),
        "INA600A_supply_sensitivity": .000010,
        "OPA333_buffer_including_bias_and_drift_screen": .000030,
        "TLV9032_offset_5V_spec_point": .002,
        "TLV9032_common_mode_60dB_5V_spec_point": 2.625 / 10**(60 / 20),
        "TLV9032_supply_sensitivity": .000070,
        "TLV9032_half_hysteresis_assumption_to_qualify": .003,
    }
    error = sum(output_errors.values())
    inner = (min(tap) - error) / .2002
    outer = (max(tap) + error) / .1998
    allowed = .260
    assert outer + .010 < allowed < outer + .020
    return {
        "status": "CONDITIONAL at the 1ms transfer error budget; original 2ms comparison fails",
        "center_leg_ohm": 442,
        "screened_supply_V": [falling_min, 5.25],
        "supervisor_worst_rising_release_threshold_V": rising_max,
        "supervisor_release_guaranteed_at_4p75V": rising_max <= 4.75,
        "tap_to_center_V": [min(tap), max(tap)],
        "output_error_contributions_V": output_errors,
        "input_inner_to_guarantee_reading_pass_V": inner,
        "input_outer_static_V": outer,
        "input_outer_at_10V_per_s_1ms_V": outer + .010,
        "input_outer_at_10V_per_s_2ms_V": outer + .020,
        "conditional_true_limit_V": allowed,
        "remaining_combined_Kelvin_noise_model_margin_at_1ms_V": allowed - outer - .010,
        "minimum_loop_resistance_for_example_130A_ohm": allowed / 130,
    }


def timing(diff):
    bus = json.loads((BOARD / "仿真/结果/跨板母线预充.json").read_bytes())
    r = bus["constant_current_sweeps"]["hot_R_sensitivity"]["r_total_ohm"]
    cap = bus["upper_initial_tolerance_F"]
    short = bus["shorted_bus"]
    pack = 37.8

    # TI TPL5111 table 1, 5 s < tIP <= 10 s. Same error sources as prior screening.
    def tip(r_set):
        scaled = r_set / 100
        return (-.1284 * scaled**2 + 46.9861 * scaled - 2651.8889) / 100

    timer_errors = (.006, 25e-6 * 1.1, .005, .014, .004 * 1.1, .0024)
    lo_tip = tip(9310 * .99 * .9975)
    hi_tip = tip(9310 * 1.01 * 1.0035)
    assert 5 < lo_tip <= hi_tip <= 10
    lo = lo_tip * math.prod(1 - e for e in timer_errors) - .050
    hi = hi_tip * math.prod(1 + e for e in timer_errors) - .050
    reserved = .326
    charge_time = lo - reserved
    attenuation = math.exp(-charge_time / (r * cap))
    inner = diff["input_inner_to_guarantee_reading_pass_V"]

    def load_limit(gap):
        return (gap - pack * attenuation) / (r * (1 - attenuation))

    allocated_A = .00425
    case = at_constant_current(pack, r, cap, charge_time, allocated_A, inner)
    cap_limit = charge_time / (r * math.log((pack - r * allocated_A) / (inner - r * allocated_A)))
    assert case["time_to_target_s"] < charge_time
    assert .010 * r > inner
    short_a = short["max_A_no_other_drops"]
    r_pre_min = pack / short_a
    proof_pulse_s = .040
    proof_pulse_A = .100
    short_energy = short["resistor_power_W"] * hi
    return {
        "TPL5111_DRVn_s": [lo, hi],
        "physical_turn_on_dwell_and_handoff_budget_s": reserved,
        "budget_includes_turn_on_s": .075,
        "budget_includes_max_dwell_s": .191,
        "budget_includes_proof_and_handoff_s": .060,
        "effective_charge_time_s": charge_time,
        "hot_model_r_ohm": r,
        "identified_cap_upper_F": cap,
        "load_limit_old_110p8mV_target_A": load_limit(.11081633091750204),
        "load_limit_442ohm_target_A": load_limit(inner),
        "design_allocation_A": allocated_A,
        "allocation_buckets_A": {"FOC_A": .00150, "FOC_B": .00150,
                                   "regulator_always_on_protection": .00060,
                                   "pump_disabled": .00045, "BMS_sense_MPPT_other": .00020},
        "case_at_allocated_current": case,
        "cap_limit_at_allocated_current_F": cap_limit,
        "remaining_charge_time_at_allocated_current_s": charge_time - case["time_to_target_s"],
        "short_peak_A": short_a,
        "short_resistor_W": short["resistor_power_W"],
        "short_single_energy_J_before_tail": short_energy,
        "short_double_energy_J_before_tails": 2 * short_energy,
        "available_fault_tail_against_7s_s": 7 - hi,
        "fixture_discrimination_only": {
            "pulse_A": proof_pulse_A, "pulse_s": proof_pulse_s,
            "minimum_precharge_only_increment_V": proof_pulse_A * r_pre_min * -math.expm1(-proof_pulse_s / (r_pre_min * cap)),
            "increment_measured_acceptance_ceiling_V": .020,
            "increment_total_error_budget_V": .010,
            "also_require_both_main_bank_VGS": True,
            "autonomous_proof_source_implemented": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # The board retains its direct BQ-alert input on U_AND2.1. U_PRE_RUN
    # already includes that input, the local permit, and both power monitors.
    for alert, local, rails, raw, proof in itertools.product((False, True), repeat=5):
        before = alert and local and rails and (raw or proof)
        after = alert and (alert and local and rails) and (raw or proof)
        assert before == after
    diff = differential()
    result = {
        "status": "ISOLATED_HOLD",
        "logic_equivalent_truth_combinations": 32,
        "logic_cases": logic_cases(),
        "differential": diff,
        "timing_and_allocation": timing(diff),
        "open": ["FOC and all other branch input envelopes and total effective capacitance",
                 "comparator and INA worst hysteresis, saturation recovery and unpowered behavior",
                 "actual proof generator, main path R/L, and physical propagation",
                 "hot SOA and Q_PRE current-to-zero tail; rearm thermal recovery",
                 "full procurement coverage"],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
