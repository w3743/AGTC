#!/usr/bin/env python3
"""Reproduce the BMS main-close feasibility screens and missing inputs."""

import hashlib
import json
import math
from pathlib import Path

from 跨板母线预充 import at_constant_current


BOARD = Path(__file__).resolve().parents[2]
POWER = BOARD.parent
RESULTS = BOARD / "仿真/结果"
CURRENT = BOARD / "检查/当前源"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def series_rlc_pulse(gap, r, inductance, capacitance, threshold):
    """First current peak and time above a positive comparison threshold."""
    if inductance == 0:
        current = gap / r
        duration = r * capacitance * math.log(current / threshold) if current > threshold else 0.0
        return 0.0, current, duration
    alpha = r / (2 * inductance)
    omega_sq = 1 / (inductance * capacitance) - alpha**2
    if math.isclose(omega_sq, 0, abs_tol=alpha**2 * 1e-10):
        t_peak = 1 / alpha

        def response(t):
            return gap / inductance * t * math.exp(-alpha * t)

        tail = t_peak * 2
    elif omega_sq > 0:
        omega = math.sqrt(omega_sq)
        t_peak = math.atan2(omega, alpha) / omega

        def response(t):
            return gap / (inductance * omega) * math.exp(-alpha * t) * math.sin(omega * t)

        tail = math.pi / omega
    else:
        beta = math.sqrt(-omega_sq)
        fast = -(alpha + beta)
        slow = 1 / (inductance * capacitance * fast)
        t_peak = math.log(fast / slow) / (slow - fast)

        def response(t):
            return gap * (math.exp(slow * t) - math.exp(fast * t)) / (inductance * (slow - fast))

        tail = t_peak * 2
    current = response(t_peak)
    assert 0 < current <= min(gap / r, gap * math.sqrt(capacitance / inductance)) * (1 + 1e-10)
    if current <= threshold:
        return t_peak, current, 0.0

    while response(tail) > threshold:
        tail *= 2

    def crossing(left, right):
        left_above = response(left) > threshold
        for _ in range(48):
            center = (left + right) / 2
            if (response(center) > threshold) == left_above:
                left = center
            else:
                right = center
        return (left + right) / 2

    return t_peak, current, crossing(t_peak, tail) - crossing(0, t_peak)


def main():
    bus_path = RESULTS / "跨板母线预充.json"
    bus = json.loads(bus_path.read_bytes())
    c_max = bus["upper_initial_tolerance_F"]
    r_pre = bus["constant_current_sweeps"]["cap_tolerance_upper"]["r_total_ohm"]
    t_min = bus["timing_s"]["min_pulse"]
    pinproof = json.loads((CURRENT / "本轮图面与官方复核.json").read_bytes())
    p1 = BOARD / "sch/Schematic1/P1.esch2"
    enet = CURRENT / "BMS-9S4P_P1_图面检查.enet"
    assert digest(p1) == pinproof["source_sha256_after"][p1.relative_to(BOARD).as_posix()]
    assert digest(enet) == pinproof["netlist"]["new_export_sha256"]
    components = {c["props"]["Designator"]: c for c in json.loads(enet.read_bytes())["components"].values()}
    timing_pins = {
        "U_SEQ": {"1": "MAIN_CLR_N", "2": "PRE_CTRL_3V3", "3": "MAIN_REQ", "5": "FET_RELEASE",
                  "9": "TMR_TRIG", "11": "PRE_REQ", "13": "POR_OK_N"},
        "U_AND2": {"4": "MAIN_CLR_N", "6": "PRE_CTRL_3V3"},
        "U_GATE1": {"3": "BMS_LOCAL_OK", "4": "PRECHARGE_WINDOW", "6": "TMR_ACTIVE"},
        "U1": {"30": "DFETOFF", "42": "BQ_PACK", "48": "VC16_FILT"},
        "RSHUNT": {"1": "B0", "2": "PACK-"},
    }
    for ref, pins in timing_pins.items():
        actual = components[ref]["pinInfoMap"]
        assert all(actual[pin]["net"] == net for pin, net in pins.items()), ref

    # 130 A is a comparison target. Do not treat it as transient qualification.
    i_compare = 130
    shunt_max = .0005 * (1 + .01 + 50e-6 * 50) + 10e-6
    scd_min_v = .100 * (1 - .15)
    scd_compare = scd_min_v / shunt_max
    cold_path = bus["main_inrush_cold_resistance_comparison"]["two_3_parallel_banks_plus_shunt_mohm"]
    rlc = []
    for r_mohm in (.45, 1, 2, cold_path, 3, 5):
        for l_uh in (0, .1, 1, 5):
            t, i, above = series_rlc_pulse(.5, r_mohm * .001, l_uh * 1e-6, c_max, scd_compare)
            rlc.append({"R_mohm": r_mohm, "L_uH": l_uh, "true_gap_V": .5,
                        "peak_A": i, "peak_time_us": t * 1e6,
                        "first_pulse_above_static_SCD_early_us": above * 1e6,
                        "shunt_sense_mV_at_peak": i * shunt_max * 1000})

    # Illustrative error allocations, each referred to the pack side.
    # Max slope and elapsed time must be guaranteed during the entire transfer.
    gap_limit_example = i_compare * .002
    static_example = .05
    slope_example = 10
    latency_example = .002
    total_error = static_example + slope_example * latency_example
    threshold = gap_limit_example - total_error
    gap_to_guarantee_acceptance = threshold - total_error
    acceptance = [at_constant_current(37.8, r_pre, c_max, t_min, i, gap_to_guarantee_acceptance)
                  for i in (0, .01, .02, .025, .05)]
    assert acceptance[0]["time_to_target_s"] > t_min
    assert acceptance[-1]["time_to_target_s"] is None
    amplifier_gain = .2
    amplifier_out_error = .003 + 10e-6 * 35 + 37.8 / 10**(89/20)
    comparator_error = .002

    report = {
        "status": "HOLD",
        "sources": {"saved_9S4P_P1_sha256": digest(p1), "official_ENET_sha256": digest(enet),
                    "cross_board_calculation_sha256": digest(bus_path),
                    "target_pack": "9S5P; same 9S voltage, minimum source impedance must be re-established"},
        "cross_board_source_match": {
            name: {"sha256": entry["sha256"], "matches_disk": digest(POWER / entry["path"]) == entry["sha256"]}
            for name, entry in bus["source_population"].items()
        },
        "verified_timing_pins": timing_pins,
        "passive_closure_only": {
            "assumptions": "one fixed source; zero additional load or injection; positive series R; zero initial incremental I; ideal step gate; lumped C and L",
            "C_F": c_max, "rlc_sensitivities": rlc,
            "bounds": "peak <= min(abs(gap)/R_min, abs(gap)*sqrt(C_max/L_min)); use each bound only with its verified lower limit",
            "mismatch_energy_J_at_0p5V": .5 * c_max * .5**2,
            "energy_note": "Only incremental mismatch energy; does not bound per-die SOA, short-circuit current, or sustained load/source power.",
        },
        "protection": {"SCD_low_sense_mV": scd_min_v * 1000, "effective_shunt_high_mohm": shunt_max * 1000,
                       "static_SCD_early_A": scd_compare,
                       "R_min_mohm_to_stay_below_static_SCD_at_0p5V": .5 / scd_compare * 1000,
                       "R_min_mohm_for_130A_at_0p5V": .5 / i_compare * 1000,
                       "SCD_nominal_delay_us": 15,
                       "qualification": "Apply noise, pre-existing current, comparator/filter tolerance and timing separately; SCD nominal delay is not current-zero time."},
        "measurement_example_only": {"BQ_uncorrelated_differential_bound_V": 1.0,
            "R_min_assumption_mohm": 2, "current_comparison_A": i_compare,
            "true_gap_limit_V": gap_limit_example, "static_error_budget_V": static_example,
            "max_gap_slew_V_per_s": slope_example, "total_age_to_low_resistance_s": latency_example,
            "measured_assert_threshold_V": threshold, "worst_case_true_gap_for_acceptance_V": gap_to_guarantee_acceptance,
            "maximum_steady_load_for_guaranteed_acceptance_A": gap_to_guarantee_acceptance / r_pre,
            "RC_sweep_at_worst_case_true_gap": acceptance},
        "precharge_short": bus["shorted_bus"],
        "candidate_only": {
            "INA600AIDBVR_and_TLV9032DDFR": {
                "gain": amplifier_gain, "illustrative_output_window_half_width_V": amplifier_gain * threshold,
                "selected_device_terms_input_referred_V": (amplifier_out_error + comparator_error) / amplifier_gain,
                "residual_static_error_budget_V": static_example - (amplifier_out_error + comparator_error) / amplifier_gain,
                "scope": "Preliminary terms only. Add real supply/VCM range, threshold and REF errors, hysteresis, noise, input network, saturation recovery, and fail-off evidence.",
            },
            "close_window_condition": "min physical Q_PRE on-time >= worst RC time + dwell + remaining measurement and main-gate transfer margin",
            "fault_window_condition": "max physical short pulse incl turn-off tail and initial Tj <= verified R_PRE and Q_PRE thermal and pulse-energy limits",
            "sequence": "local armed -> one PRE_REQ -> stable signed PRE_DONE in open timing window -> new MAIN_REQ edge -> confirmed main conduction -> receiver power permission",
            "disable": "Invalid or stale data, prebias source unknown, timeout or fault inhibits MAIN_REQ, clears FF-A, and revokes precharge. No automatic retry.",
        },
        "inputs_required": [
            "effective C, ESR/ESL, max precharge I(V,t,T), gate-on load steps, prebias and source isolation for all attached devices",
            "Kelvin loop minimum R and L, pack minimum impedance for 9S4P and 9S5P, coupled branches and harness",
            "simultaneous signed differential error including offsets, gain, noise, CM, contact and shunt drop; sample age, qualification delay and worst gap slew",
            "hot individual MOS I_D, V_DS, V_GS and Tj trajectories, current hogging, SOA and Zth under pulse/retry, current-zero and clamp response",
            "guaranteed fault shutoff and restart across reset, power loss and externally prebiased bus; verified FOC/auxiliary sequencing",
        ],
    }
    (RESULTS / "主路合闸准入.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
