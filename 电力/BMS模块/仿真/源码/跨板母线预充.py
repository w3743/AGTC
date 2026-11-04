#!/usr/bin/env python3
"""Bound the shared-bus precharge with installed populations and load sweeps."""

import hashlib
import json
import math
from pathlib import Path

BOARD = Path(__file__).resolve().parents[2]
POWER = BOARD.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_population(path, refs, bus):
    netlist = json.loads(path.read_text(encoding="utf-8-sig"))
    parts = {c["props"]["Designator"]: c for c in netlist["components"].values()}
    for ref, mpn in refs.items():
        c = parts[ref]
        assert c["props"]["Manufacturer Part"] == mpn, (ref, c["props"]["Manufacturer Part"])
        assert {p["net"] for p in c["pinInfoMap"].values()} == {bus, "GND_SYS"}, ref
    return {"path": path.relative_to(POWER).as_posix(), "sha256": sha(path), "bulk_refs": refs}


def at_constant_current(pack_v, r, capacitance, t, load_a, target_gap=0.5):
    steady_gap = r * load_a
    end_gap = steady_gap + (pack_v - steady_gap) * math.exp(-t / (r * capacitance))
    target_s = None
    if steady_gap < target_gap:
        target_s = r * capacitance * math.log((pack_v - steady_gap) / (target_gap - steady_gap))
    return {"load_A": load_a, "end_gap_V": end_gap, "steady_gap_V": steady_gap, "time_to_target_s": target_s}


def power_step(pack_v, r, capacitance, duration, steps, dt=0.0002):
    # UVLO-like ramp followed by a bounded constant input-power comparison.
    # Ramping over 1 V avoids a solver discontinuity. Actual start-up is unknown.
    v = 0.0
    first_target = None
    count = math.ceil(duration / dt)
    dt = duration / count
    switches = [{"V": threshold, "W": power} for threshold, power in steps]
    def dv(bus):
        load_w = sum(power * max(0, min(1, bus - threshold)) for threshold, power in steps)
        load_a = 0.002 + load_w / max(1, bus)
        return ((pack_v - bus) / r - load_a) / capacitance
    for i in range(count):
        a = dv(v)
        b = dv(v + a * dt / 2)
        c = dv(v + b * dt / 2)
        d = dv(v + c * dt)
        v += dt / 6 * (a + 2*b + 2*c + d)
        if first_target is None and pack_v - v <= 0.5:
            first_target = (i + 1) * dt
    return {"input_power_steps": switches, "static_background_A": 0.002,
            "end_gap_V": pack_v-v, "time_to_0p5V_s": first_target}


def main():
    sources = {
        "FOC_per_board": verify_population(
            POWER / "FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_P1.enet",
            {f"CBUS{i}": "PEH200PJ5100MB2" for i in range(1, 4)}, "BUS"),
        "regulator_saved_official_enet": verify_population(
            POWER / "稳压模块/检查/本轮磁盘读回.enet",
            {f"C_BUS_BULK{i}": "PEH226NL3900QE4" for i in range(1, 7)}, "BUS_EXT"),
        "pump": verify_population(
            POWER / "水泵驱动模块/eda工程文件/导出/当前交付/水泵驱动.enet",
            {"C_VM_BULK": "UHW2A471MHD", "C_5V_IN": "EMK1KM330GB0D00T"}, "BUS_M"),
    }
    # Exact electrolytic tolerance corners, plus nominal direct small caps.
    # The latter use +20% as a sensitivity. Values are not lifetime guarantees.
    population = [
        {"board": "FOC-A and B", "bulk_nominal_uF": 60000, "bulk_upper_uF": 72000,
         "direct_small_nominal_uF": 117.8},
        {"board": "regulator", "bulk_nominal_uF": 5400, "bulk_upper_uF": 7020,
         "direct_small_nominal_uF": 15.6},
        {"board": "pump", "bulk_nominal_uF": 503, "bulk_upper_uF": 603.6,
         "direct_small_nominal_uF": 0.12},
    ]
    nominal = sum(x["bulk_nominal_uF"] + x["direct_small_nominal_uF"] for x in population) * 1e-6
    high = sum(x["bulk_upper_uF"] + 1.2*x["direct_small_nominal_uF"] for x in population) * 1e-6
    timing = json.loads((BOARD / "仿真/结果/预充Layout闭合.json").read_text(encoding="utf-8-sig"))["timer"]
    shortest = timing["drv_pulse_min_s"]
    longest = timing["drv_pulse_max_s"]
    latest_falling_edge = timing["ff_b_power_to_drv_low_max_s"]
    pack = 37.8
    r_slow = 8.2*1.05 + 2*0.058
    r_hot_sensitivity = 8.2*1.05*(1+200e-6*(350-25)) + 2*0.125
    r_short = 8.2*0.95*(1-200e-6*(350-25))
    main_path_mohm = 2 * 2.8 / 3 + .5
    cases = {key: {
        "capacitance_F": cap, "r_total_ohm": r,
        "constant_load_cases": [at_constant_current(v, r, cap, shortest, load) for load in (0, .002, .025, .05, .075, .10)]
    } for key, v, r, cap in (
        ("legacy_72mF", pack, r_slow, .072), ("nominal_population", pack, r_slow, nominal),
        ("cap_tolerance_upper", pack, r_slow, high), ("hot_R_sensitivity", pack, r_hot_sensitivity, high),
        ("low_pack_cap_tolerance_upper", 22.5, r_slow, high),
    )}
    assert abs(cases["legacy_72mF"]["constant_load_cases"][0]["end_gap_V"] - .9047) < .003
    assert cases["cap_tolerance_upper"]["constant_load_cases"][0]["end_gap_V"] > .5
    ideal = power_step(pack, r_slow, high, shortest, [])
    exact = at_constant_current(pack, r_slow, high, shortest, .002)
    assert abs(ideal["end_gap_V"] - exact["end_gap_V"]) < 1e-5
    steps = {
        "regulator_aux_comparison": [(18, .75)],
        "pump_buck_budget_comparison": [(18, 1.875)],
        "both_aux_comparisons": [(18, .75), (18, 1.875)],
        "load_enable_order_sensitivity": [(18, 2.625), (32, 2.0)],
    }
    load_scenarios = {k: power_step(pack, r_slow, high, shortest, v) for k,v in steps.items()}
    result = {
        "scope": "Reduced R-C and bounded load sensitivities, not measured input current or MOS SOA",
        "source_population": sources, "populations_uF": population,
        "nominal_F": nominal, "upper_initial_tolerance_F": high,
        "timing_s": {"min_pulse": shortest, "max_pulse": longest,
                     "ff_b_power_to_drv_low_max": latest_falling_edge,
                     "scope": "DRVn is low during the first REXT reading (TI TPL5111 Figure 10); physical turnoff tail is separate"},
        "constant_current_sweeps": cases,
        "power_step_sensitivities_37p8V": load_scenarios,
        "allowable_total_load_at_0p5V_gap_A": .5/r_slow,
        "allowable_total_load_at_0p5V_gap_W": {str(v): (v-.5)*.5/r_slow for v in (22.5,37.8)},
        "differential_gate": {"actual_gap_comparison_V": .5, "BQ_stack_plus_pack_uncombined_accuracy_bound_V": 1.0,
          "formula": "abs(measured difference)+error bound <= allowed true gap",
          "engineering_example_calibrated_error_V": .1,
          "engineering_example_measured_threshold_V": .4,
          "engineering_example_true_gap_to_guarantee_acceptance_V": .3,
          "target_0p3V_at_25mA": at_constant_current(pack,r_slow,high,shortest,.025,.3)},
        "main_inrush_sensitivity": {str(milliohm): .5/(milliohm*.001) for milliohm in (2,3,5,10,20)},
        "main_inrush_cold_resistance_comparison": {
            "two_3_parallel_banks_plus_shunt_mohm": main_path_mohm,
            "initial_A_at_0p5V": .5 / (main_path_mohm * .001),
            "true_gap_V_at_130A": 130 * main_path_mohm * .001,
            "scope": "25 C max stated RDS(on) per MOS, nominal shunt, ideal sharing; no ESR, ESL or gate ramp",
        },
        "shorted_bus": {"R_pre_low_ohm": r_short, "max_A_no_other_drops": pack/r_short,
           "resistor_power_W": pack**2/r_short,
           "resistor_energy_J_max_drv_pulse": pack**2/r_short*longest,
           "resistor_energy_J_if_full_elapsed_time_conducted": pack**2/r_short*latest_falling_edge,
           "single_FET_load_line_peak_W": pack**2/(4*r_short),
           "single_FET_load_line_peak_V_A": [pack/2,pack/(2*r_short)]},
        "bus_reject_energy_sensitivity": {
            "37p8_to_40p3V_at_initial_tolerance_low_bulk_J": .5*(.048+.00486+.0004024)*(40.3**2-37.8**2),
            "constant_495p3W_30ms_J": 495.3*.03,
            "0p82ohm_dump_at_40p3V_A": 40.3/.82, "0p82ohm_dump_at_40p3V_W": 40.3**2/.82,
            "8S_350F_20A_0_to_2V_per_cell_ideal_s": (350/8)*16/20,
        },
    }
    output = BOARD / "仿真/结果/跨板母线预充.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
