#!/usr/bin/env python3
"""Evaluate the passive 5 V preload and conditional shared-bus startup window."""

import hashlib
import json
import math
from pathlib import Path

from 跨板母线预充 import at_constant_current


BOARD = Path(__file__).resolve().parents[2]
RESULTS = BOARD / "仿真/结果"
CURRENT = BOARD / "检查/当前源"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preload(qty, nominal_ohm, rated_w):
    # Initial tolerance, full 25-to-155 C TCR span and 1000 h load-life
    # resistance change. Their combination is a conservative sizing budget.
    high = nominal_ohm * 1.01 * 1.013 * 1.01 + .05
    low = nominal_ohm * .99 * .987 * .99 - .05
    min_v, max_v, local_ambient = 4.75, 5.25, 100
    derated_w = rated_w * (155 - local_ambient) / (155 - 70)
    minimum_after_open = (qty - 1) * min_v / high
    per_part_max = max_v**2 / low
    return {
        "qty": qty, "nominal_ohm": nominal_ohm, "equivalent_ohm": nominal_ohm / qty,
        "R_low_ohm": low, "R_high_ohm": high, "nominal_5V_A": 5 * qty / nominal_ohm,
        "all_branches_4p75V_min_A": qty * min_v / high,
        "one_open_4p75V_min_A": minimum_after_open, "one_open_4p75V_min_W": min_v * minimum_after_open,
        "min_V_for_60mA_and_0p30W_one_open": max(.06 * high / (qty - 1), math.sqrt(.3 * high / (qty - 1))),
        "per_part_max_W_at_5p25V": per_part_max, "group_max_W_at_5p25V": qty * per_part_max,
        "group_max_A_at_5p25V": qty * max_v / low,
        "local_ambient_design_limit_C": local_ambient, "derated_per_part_W_at_design_limit": derated_w,
        "per_part_W_margin_at_design_limit": derated_w - per_part_max,
    }


def startup(bus, main_close):
    cap = bus["upper_initial_tolerance_F"]
    cold = bus["constant_current_sweeps"]["cap_tolerance_upper"]["r_total_ohm"]
    hot = bus["constant_current_sweeps"]["hot_R_sensitivity"]["r_total_ohm"]
    timing = main_close["measurement_example_only"]
    target = timing["worst_case_true_gap_for_acceptance_V"]
    time_reserve = .25
    charge_time = 6 - time_reserve
    pack = 37.8

    def limit_load(r, time, gap):
        remaining = math.exp(-time / (r * cap))
        return (gap - pack * remaining) / (r * (1 - remaining))

    short = bus["shorted_bus"]
    short_comparison_time = 7
    short_a = short["max_A_no_other_drops"]
    qpre_hot_r_assumed = .125
    qpre_per_die_power = short_a**2 * qpre_hot_r_assumed
    budget = .010
    hot_10ma = at_constant_current(pack, hot, cap, charge_time, budget, target)
    assert 0 < hot_10ma["end_gap_V"] < target
    assert limit_load(cold, bus["timing_s"]["min_pulse"], .5) < 0
    # TPL5111 Table 1, 5 < tIP <= 10 s, using the same error sources as
    # the existing timer audit but the larger supply deviation on both sides.
    def pulse_from_fit(resistor):
        rd = resistor / 100
        return (-.1284 * rd**2 + 46.9861 * rd - 2651.8889) / 100

    timer_errors = (.006, 25e-6 * 1.1, .005, .014, .004 * 1.1, .0024)
    timer_low = pulse_from_fit(9530 * .99 * .9975) * math.prod(1 - v for v in timer_errors) - .050
    timer_high = pulse_from_fit(9530 * 1.01 * 1.0035) * math.prod(1 + v for v in timer_errors) - .050
    return {
        "status": "CONDITIONAL_MODEL_ONLY",
        "common_BUS_max_C_F": cap, "pack_max_V": pack, "bus_initial_V": 0,
        "precharge_hot_R_sensitivity_ohm": hot,
        "existing_min_window_s": bus["timing_s"]["min_pulse"],
        "existing_max_window_s": bus["timing_s"]["max_pulse"],
        "existing_min_window_max_load_for_0p5V_A": limit_load(cold, bus["timing_s"]["min_pulse"], .5),
        "comparison": {
            "required_actual_gap_to_guarantee_acceptance_V": target, "physical_window_min_s": 6,
            "reserved_qualification_and_transfer_s": time_reserve, "maximum_total_silent_load_contract_A": budget,
            "max_load_within_RC_budget_at_cold_R_A": limit_load(cold, charge_time, target),
            "max_load_within_RC_budget_at_hot_R_A": limit_load(hot, charge_time, target),
            "hot_10mA_result": hot_10ma,
            "remaining_load_after_two_FOC_nominal_resistors_A": budget - .00204,
            "central_12V_hard_protection_requirement": "Keep available before overvoltage exposure; qualify its BUS ramp startup with all source injection inhibited until protection is valid.",
            "max_physical_fault_conduction_comparison_s": short_comparison_time,
            "resistor_short_energy_J": short["resistor_power_W"] * short_comparison_time,
            "Q_PRE_assumed_hot_Rds_ohm_each": qpre_hot_r_assumed,
            "Q_PRE_short_conduction_W_each": qpre_per_die_power,
            "Q_PRE_max_Zth_C_per_W_from_85C_to_125C": (125 - 85) / qpre_per_die_power,
            "TPL5111_9p53k_datasheet_fit_only_s": {"DRVn_min":timer_low, "DRVn_max":timer_high,
                "remaining_before_6s_min_for_physical_turn_on":timer_low-6,
                "remaining_before_7s_fault_max_for_physical_turn_off":7-timer_high},
        },
        "load_power_valid": {
            "bias_population": "FOC-A 1k; FOC-B DNP; one shared floating receiver pair",
            "max_combined_Vf_plus_sink_V_for_5mA_from_4p75V": 4.75 - .005 * 515.1,
            "estimated_high_min_V": 3.192, "estimated_threshold_V": 3.047,
            "required_coupler_IC_mA_at_VCE_0p4V": 1.425,
            "one_receiver_dark_current_budget_A_at_0p3V_base": .3 / 101350,
            "status": "static design budgets only; full-temperature off leakage, sink behavior and turnoff ordering remain open",
        },
        "BOTHOFF_candidate": {
            "BQ_VIL_from_REG18_min_V": .33 * 1.6, "BQ_VIH_from_REG18_max_V": .66 * 2,
            "47k_pulldown_high_ohm": 47000 * 1.0135,
            "allowable_sum_positive_leakage_A": .33 * 1.6 / (47000 * 1.0135),
            "BQ_leakage_spec_scope": "1 uA for listed pins in SHUTDOWN only",
            "status": "HOLD: no guaranteed Q leakage/state over slow/zero PRE_CTRL_3V3, no configuration or open-pulldown coverage",
        },
    }


def main():
    bus_path = RESULTS / "跨板母线预充.json"
    close_path = RESULTS / "主路合闸准入.json"
    bus, close = (json.loads(p.read_bytes()) for p in (bus_path, close_path))
    enet = CURRENT / "BMS-9S4P_P1_图面检查.enet"
    parts = json.loads(enet.read_bytes())["components"]
    components = {c["props"]["Designator"]: c for c in parts.values()}
    module = components["PSU5_EXT"]
    assert module["props"]["Manufacturer Part"] == "URB4805YMD-10WR3"
    assert module["pinInfoMap"]["2"]["net"] == "TAP_B9"
    preload_refs = {ref for ref in components if ref.startswith("R_PSU_MINLOAD")}
    assert preload_refs == {f"R_PSU_MINLOAD{i}" for i in range(1, 6)}
    for ref in preload_refs:
        c = components[ref]
        props = c["props"]
        assert props["Manufacturer Part"] == "1206W4F2700T5E" and props["Supplier Part"] == "C4466"
        assert props["FootprintName"] == "R1206"
        assert props["Add into BOM"] == props["Convert to PCB"] == "yes"
        assert {k:v["net"] for k,v in c["pinInfoMap"].items()} == {"1":"VOUT_5V", "2":"B0"}
    original, alternate, selected = preload(7, 510, .1), preload(5, 300, .25), preload(5, 270, .25)
    assert selected["one_open_4p75V_min_A"] >= .06 and selected["one_open_4p75V_min_W"] >= .3
    assert alternate["one_open_4p75V_min_W"] < .3
    assert selected["per_part_W_margin_at_design_limit"] > 0
    assert .3 + selected["group_max_A_at_5p25V"] < 2
    report = {
        "source": {"official_ENET_sha256": digest(enet), "cross_board_model_sha256": digest(bus_path),
                   "main_close_model_sha256": digest(close_path)},
        "preload": {"status":"Schematic envelope pass at specified conditions",
                    "original":original, "five_300ohm_alternate":alternate, "selected":selected,
                    "single_fault": {"one_open":"60mA and 0.30W satisfied; no credit for variable loads",
                                     "one_short":"Both topologies directly short VOUT_5V-B0. Existing HLK protection applies; LPV supply loss and physical main isolation remain separate system checks."},
                    "baseline_300mA_plus_preload_A": .3 + selected["group_max_A_at_5p25V"],
                    "minimum_load_source":"https://hlktech.net/index.php?id=980",
                    "resistor_source":"https://www.uni-royal.cn/en/images/userfile/file/1753752986c56505e6d9ab55c7.pdf"},
        "startup": startup(bus, close),
    }
    (RESULTS / "正常起电与5V预载.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
