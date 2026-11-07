#!/usr/bin/env python3
"""Screen the isolated PRE_DONE candidate without changing the adopted design."""

import argparse
import itertools
import json
import math
from pathlib import Path

from 跨板母线预充 import at_constant_current


RESULTS = Path(__file__).resolve().parents[1] / "结果"


def logic_cases():
    # Ideal logic after rails and propagation settle. Physical proof is an input.
    class Sequence:
        armed = False
        timer_used = False
        main = False

        def step(self, *, good=True, pre_edge=False, main_edge=False, drv=False,
                 differential=False, settled=False, proof=False):
            if not good:
                self.armed = False
            if pre_edge:
                self.armed = good and not self.timer_used
                self.timer_used = True
            pre = self.armed and good and drv
            raw = pre and differential
            done = raw and settled and not proof
            if not (good and (raw or proof)):
                self.main = False
            elif main_edge:
                self.main = done
            return pre, done, self.main, self.main and proof

    cases = {}
    seq = Sequence()
    cases["early_edge"] = seq.step(pre_edge=True, main_edge=True, drv=True)
    cases["stale_high_request_no_edge"] = seq.step(drv=True, differential=True, settled=True)
    cases["new_qualified_request"] = seq.step(main_edge=True, drv=True, differential=True, settled=True)
    cases["physical_proof_arrives_after_release"] = seq.step(drv=True, differential=True, settled=True, proof=True)
    cases["timer_expires_after_physical_proof"] = seq.step(proof=True)
    cases["brownout_or_cancel_after_proof"] = seq.step(good=False, proof=True)
    cases["cancel_then_permit_recovery_within_window"] = seq.step(drv=True, differential=True)
    cases["second_pre_req_same_power_session"] = seq.step(pre_edge=True, drv=True, differential=True)

    seq = Sequence()
    seq.step(pre_edge=True, drv=True)
    cases["rail_transient_window_without_dwell"] = seq.step(main_edge=True, drv=True, differential=True)
    seq.step(main_edge=True, drv=True, differential=True, settled=True)
    cases["differential_lost_before_physical_proof"] = seq.step(drv=True)
    seq.step(main_edge=True, drv=True, differential=True, settled=True)
    cases["timeout_before_physical_proof"] = seq.step()
    cases["request_edge_after_timeout"] = seq.step(main_edge=True, differential=True, settled=True)

    seq = Sequence()
    seq.step(pre_edge=True, drv=True, proof=True)
    cases["stale_high_physical_proof_blocks_new_main_request"] = seq.step(
        main_edge=True, drv=True, differential=True, settled=True, proof=True)

    assert cases == {
        "early_edge": (True, False, False, False),
        "stale_high_request_no_edge": (True, True, False, False),
        "new_qualified_request": (True, True, True, False),
        "physical_proof_arrives_after_release": (True, False, True, True),
        "timer_expires_after_physical_proof": (False, False, True, True),
        "brownout_or_cancel_after_proof": (False, False, False, False),
        "cancel_then_permit_recovery_within_window": (False, False, False, False),
        "second_pre_req_same_power_session": (False, False, False, False),
        "rail_transient_window_without_dwell": (True, False, False, False),
        "differential_lost_before_physical_proof": (True, False, False, False),
        "timeout_before_physical_proof": (False, False, False, False),
        "request_edge_after_timeout": (False, False, False, False),
        "stale_high_physical_proof_blocks_new_main_request": (True, False, False, False),
    }
    return {"model": "settled logic only", "columns": ["pre", "done", "fet_release", "lpv"], "cases": cases}


def electrical():
    bus = json.loads((RESULTS / "跨板母线预充.json").read_bytes())
    start = json.loads((RESULTS / "正常起电与5V预载.json").read_bytes())["startup"]["comparison"]
    cap = bus["upper_initial_tolerance_F"]
    r = bus["constant_current_sweeps"]["hot_R_sensitivity"]["r_total_ohm"]
    pack = 37.8
    # Four discrete 1% resistors. Additional ±0.35% temperature sensitivity.
    tap = []
    for v, t, a, b, g in itertools.product((4.60, 5.25), *((.9865, 1.0135),) * 4):
        total = 33200 * (t + g) + 510 * (a + b)
        tap.extend((v * 510 * a / total, v * 510 * b / total))
    offset = {
        "INA600_output_offset_and_drift": .00335,
        "INA600_common_mode_screen": pack / 10**(89 / 20),
        "TLV9001_reference_buffer_offset": .002,
        "TLV9032_offset_screen": .002,
        "TLV9032_hysteresis_assumed": .003,
    }
    error = sum(offset.values())
    largest_static = (max(tap) + error) / .1998
    smallest_static = (min(tap) - error) / .2002
    # Conservative sensitivity; actual Schmitt thresholds and Ceff need validation.
    min_dwell = -.099 * .6 * math.log1p(-1.50 / 3.60)
    max_dwell = -.101 * 1.2 * math.log1p(-2.30 / 2.90)
    turn_on_budget = .075
    dwell_and_transfer_reserve = max_dwell + .060
    timer = start["TPL5111_9p53k_datasheet_fit_only_s"]
    charge_s = timer["DRVn_min"] - turn_on_budget - dwell_and_transfer_reserve
    charge = at_constant_current(pack, r, cap, charge_s, .010, smallest_static)
    attenuation = math.exp(-charge_s / (r * cap))
    load_limit = (smallest_static - pack * attenuation) / (r * (1 - attenuation))
    constant_drop = .010 * r
    cap_limit = charge_s / (r * math.log((pack - constant_drop) / (smallest_static - constant_drop)))
    short_w = bus["shorted_bus"]["resistor_power_W"]
    short_a = bus["shorted_bus"]["max_A_no_other_drops"]
    energy = short_w * timer["DRVn_max"]
    return {
        "status": "HOLD: semiconductor and whole-system qualification open",
        "differential": {"tap_to_center_bounds_V": [min(tap), max(tap)],
            "error_terms_output_V": offset, "symmetric_static_acceptance_screen_V": smallest_static,
            "static_acceptance_outer_bound_V": largest_static,
            "outer_bound_with_10V_per_s_and_2ms_V": largest_static + .020,
            "example_true_allowed_V": .260,
            "confirmed_bounds_missing": ["saturation recovery and false-window dwell at all rail/CM corners",
                "actual hysteresis, input protection transients and Kelvin routing error", "source direction and unpowered backfeed"]},
        "timing": {"TPL5111_DRVn_window_s": [timer["DRVn_min"], timer["DRVn_max"]],
            "RC_dwell_s_sensitivity": [min_dwell, max_dwell],
            "physical_turn_on_budget_s": turn_on_budget,
            "remaining_transfer_after_dwell_s": .060, "charging_budget_s": charge_s,
            "charge_screen_at_10mA": charge, "hot_R_ohm": r, "identified_C_upper_F": cap,
            "load_limit_at_identified_C_A": load_limit, "C_limit_at_10mA_F": cap_limit,
            "dwell_earliest_requires_initial_cap_voltage_at_B0": True},
        "short_fault": {"peak_current_A": short_a, "resistor_power_W": short_w,
            "single_long_DRVn_pulse_energy_J_before_tail": energy,
            "two_pulses_no_cooldown_energy_J_before_tails": 2 * energy,
            "additional_resistor_energy_per_ms_tail_J": short_w / 1000,
            "extra_time_before_7s_comparison_s": 7 - timer["DRVn_max"],
            "QPRE_each_hot_R_assumed_ohm": .125, "QPRE_each_conduction_W": short_a**2 * .125,
            "automatic_retry_model_pass": True, "physical_tail_and_rearm_cooldown": "OPEN"},
        "handoff": {"proof_input_required_low_before_arming": True,
            "stale_high_proof_blocks_fresh_main_request_in_logic_model": True,
            "proving_observation": "Both bank VGS and bounded low-impedance conduction response before power permission.",
            "physical_proof_source_and_validity": "OPEN"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = {"logic": logic_cases(), "electrical": electrical()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
