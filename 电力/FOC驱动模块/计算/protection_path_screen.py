"""Reproducible paper screen for FOC protection margins and failure paths.

Python 3.10+, standard library only. Reads the existing current-window corner
report and writes protection_path_screen.json beside this script. It does not
modify the schematic, source netlist, candidate BOM, or existing current-window
calculation.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
CURRENT_JSON = MODULE / "原理图准备" / "电流窗口误差核算.json"
OUT = HERE / "protection_path_screen.json"


def safe_threshold_screen() -> dict:
    rtop_nom = 27_400.0
    rbot_nom = 10_000.0
    rtol = 0.01
    c_nom = 1e-9
    ctol = 0.05
    vrefint_min, vrefint_typ, vrefint_max = 1.182, 1.212, 1.232
    scaler_abs_max = 0.010
    comp_offset_min, comp_offset_max = -0.009, 0.003
    comp_pin_leakage_abs_max = 150e-9

    eff_min = vrefint_min - scaler_abs_max + comp_offset_min
    eff_typ = vrefint_typ
    eff_max = vrefint_max + scaler_abs_max + comp_offset_max

    resistor_corners = [
        (rtop_nom * (1 + s_top * rtol), rbot_nom * (1 + s_bot * rtol))
        for s_top, s_bot in itertools.product((-1, 1), repeat=2)
    ]
    ratios = [(top + bot) / bot for top, bot in resistor_corners]
    ratio_nom = (rtop_nom + rbot_nom) / rbot_nom
    # COMP4 is on a TT GPIO input. At the crossing, SAFE_ADC is below valid VDD,
    # so include Table 53's ±150 nA pad-leakage bound in the divider threshold.
    v_trip_min = min(
        (eff_min - comp_pin_leakage_abs_max * (top * bot / (top + bot)))
        * ((top + bot) / bot)
        for top, bot in resistor_corners
    )
    v_trip_max = max(
        (eff_max + comp_pin_leakage_abs_max * (top * bot / (top + bot)))
        * ((top + bot) / bot)
        for top, bot in resistor_corners
    )
    v_trip_typ = eff_typ * ratio_nom
    safe_adc_nom_at_5v = 5.0 / ratio_nom

    time_corners = []
    for top, bot in resistor_corners:
        rth = top * bot / (top + bot)
        ratio = (top + bot) / bot
        for cap, vth, ileak in itertools.product(
            (c_nom * (1 - ctol), c_nom * (1 + ctol)),
            (eff_min, eff_max),
            (-comp_pin_leakage_abs_max, comp_pin_leakage_abs_max),
        ):
            # Conditional first-order response to an ideal SAFE_5V 5 V -> 0 V step.
            # Real regulator/output-capacitor decay is not a step and is not included.
            v_initial = 5.0 / ratio + ileak * rth
            v_final = ileak * rth
            t_us = rth * cap * math.log((v_initial - v_final) / (vth - v_final)) * 1e6
            time_corners.append(t_us)

    # TPS7A4001 divider source screen. Applying its overall ±2.5% output accuracy to
    # this resistor-programmed output is an explicit screening assumption, not a
    # datasheet-guaranteed full-temperature bound for this exact circuit.
    fb_top_nom, fb_bot_nom = 97_600.0, 30_100.0
    fb_rtol = 0.01
    fb_top_min = fb_top_nom * (1 - fb_rtol)
    fb_bot_max = fb_bot_nom * (1 + fb_rtol)
    ldo_nom = 1.173 * (1 + fb_top_nom / fb_bot_nom)
    ldo_min_screen = (
        1.173 * (1 - 0.025) * (1 + fb_top_min / fb_bot_max)
        - 0.1e-6 * fb_top_min
    )

    return {
        "status": "screen_only_not_hardware_guarantee",
        "planned_network": {
            "R_SAFEADC_TOP_ohm": rtop_nom,
            "R_SAFEADC_BOT_ohm": rbot_nom,
            "resistor_tolerance_each": 0.01,
            "C_SAFEADC_F": c_nom,
            "capacitor_tolerance_assumed": 0.05,
            "SAFE_ADC_nominal_ratio": 1 / ratio_nom,
            "SAFE_ADC_at_SAFE5_5V_nom_V": safe_adc_nom_at_5v,
            "nominal_tau_us": (rtop_nom * rbot_nom / (rtop_nom + rbot_nom)) * c_nom * 1e6,
            "tau_corner_us": {
                "min": min(
                    (top * bot / (top + bot)) * cap * 1e6
                    for top, bot in resistor_corners
                    for cap in (c_nom * (1 - ctol), c_nom * (1 + ctol))
                ),
                "max": max(
                    (top * bot / (top + bot)) * cap * 1e6
                    for top, bot in resistor_corners
                    for cap in (c_nom * (1 - ctol), c_nom * (1 + ctol))
                ),
            },
            "COMP4_trip_SAFE5_V": {
                "nominal": v_trip_typ,
                "corner_min": v_trip_min,
                "corner_max": v_trip_max,
                "calculation": "(VREFINT + VSC + Voffset_COMP ± ILEAK*Rth) * (Rtop + Rbottom) / Rbottom",
                "VREFINT_min_typ_max_V": [vrefint_min, vrefint_typ, vrefint_max],
                "VSC_bound_mV": [-10, 10],
                "COMP_offset_bound_mV": [-9, 3],
                "PB0_TT_input_leakage_abs_max_nA": 150,
                "corner_method": "independent VREFINT/VSC/Voffset/TT pad leakage extremes and Rtop/Rbottom each ±1%",
                "omissions": [
                    "HYST=000 has only a typical 0 mV entry in DS Table 73; min/max are not specified",
                    "uses TT pad leakage ±150 nA as an input-current bound at a valid VDD; verify pin configuration and analog input behavior",
                    "resistor temperature coefficient and aging",
                    "SAFE5 source ramp/decay, output-capacitor/load interaction, wiring noise",
                ],
            },
            "healthy_node_at_SAFE5_5V_V": 5.0 / ratio_nom,
            "COMP4_healthy_overdrive_at_SAFE5_5V_nominal_mV": (5.0 / ratio_nom - vrefint_typ) * 1e3,
            "ideal_source_step_to_COMP4_trip_us": {
                "min": min(time_corners),
                "max": max(time_corners),
                "scenario": "ideal SAFE5 source step 5 V to 0 V; divider RC only",
                "not_total_fault_response": True,
            },
            "TPS7A4001_source_screen": {
                "R_FB_TOP_ohm": fb_top_nom,
                "R_FB_BOTTOM_ohm": fb_bot_nom,
                "nominal_output_V_from_VREF_1p173": ldo_nom,
                "output_floor_screen_V": ldo_min_screen,
                "screen_margin_over_COMP_trip_max_V": ldo_min_screen - v_trip_max,
                "assumption": "apply ±2.5% overall-output accuracy to programmed-divider output; do not treat as full-temperature guaranteed min for this exact circuit",
                "feedback_current_bound_uA": 0.1,
                "feedback_current_floor_term_V": 0.1e-6 * fb_top_min,
                "divider_current_nominal_uA": ldo_nom / (fb_top_nom + fb_bot_nom) * 1e6,
                "data_limit": "TI Table 6.5 conditions say FB tied to OUT unless otherwise noted; Table 6.5 VREF min/max are at 25°C. The programmable output formula is application guidance. The calculated margin is screening only.",
            },
        },
        "COMP4_configuration_candidate": {
            "input_plus": "PB0 / COMP4_INP / ADC1_IN15 = SAFE_ADC",
            "input_minus": "INMSEL=011 (VREFINT); SCALEN=1; BRGEN=0 for full VREFINT input",
            "output_polarity": "COMP_CSR.POLARITY=1 (inverted): healthy SAFE_ADC above threshold -> COMP output low; undervoltage -> high",
            "timer_route": "TIM1_AF1.BKCMP4E=1; BKE=1; AOE=0; BKF=0",
            "break_polarity_tuple": "BKP=1, BKINP=1 for active-low external BREAK_N; BKCMP1P/2P/3P/4P=0 for active-high fault comparator outputs (COMP4 itself inverted as above)",
            "startup_wait_us": {
                "scaler_max": 200,
                "comparator_max": 5,
                "combined_minimum_before_arm": 205,
            },
            "prop_delay_limit": "G431 DS Table 73 max 31 ns only at VDDA≥2.7 V for 200 mV input step, 100 mV overdrive and 50 pF output load; not a slow-ramp or full path delay guarantee",
            "register_note": "G431 CMSIS COMP_CSR has no OUTSEL field; COMP4 is routed internally through TIM1_AF1.BKCMP4E",
        },
        "partial_power_so_to_tlv_screen": {
            "input_resistors_each_ohm": 1000,
            "resistor_min_ohm": 990,
            "conditional_bound": "Assume VREF_CORE≤3.6 V, DRV SO absolute max≤VREF_CORE+0.3 V, SAFE5=0 V, and TLV1704 clamp to V+ +0.5 V; I=(3.6+0.3-0.5)/990.",
            "per_channel_clamp_current_upper_mA_under_assumptions": (3.6 + 0.3 - 0.5) / 990 * 1e3,
            "three_channel_sum_mA_under_assumptions": 3 * (3.6 + 0.3 - 0.5) / 990 * 1e3,
            "interpretation": "below TI TLV1704 ±10 mA per-input absolute current limit under the stated rail assumptions; does not guarantee safe-rail backfeed, comparator output state, or function while SAFE5 is off",
        },
    }


def current_window_and_slope_screen() -> dict:
    if not CURRENT_JSON.exists():
        raise FileNotFoundError(f"Required existing current-window calculation not found: {CURRENT_JSON}")
    prior = json.loads(CURRENT_JSON.read_text(encoding="utf-8"))
    case = prior["cases"]["root_selected_gain10_nominal_plus_minus_150A"]
    pos = case["positive_trip"]
    neg = case["negative_trip_magnitude"]
    rneg_top_max = 26_700.0 * 1.01
    rneg_bottom_max = 10_000.0 * 1.01
    rneg_th_max = rneg_top_max * rneg_bottom_max / (rneg_top_max + rneg_bottom_max)
    rseries_max = 1_000.0 * 1.01
    min_sense_slope = 0.000492525 * 9.7
    negative_input_bias_bound_A = 20e-9 * (rneg_th_max + rseries_max) / min_sense_slope
    positive_pin_leakage_bound_A = 150e-9 * 47.0 / min_sense_slope
    pos_min, pos_max = pos["min_trip_A"], pos["max_trip_A"]
    neg_min, neg_max = neg["min_trip_A"], neg["max_trip_A"]
    margins = {
        "positive_A": pos_min - 120.0 - positive_pin_leakage_bound_A,
        "negative_magnitude_A": neg_min - 120.0 - negative_input_bias_bound_A,
    }
    slopes = [1, 5, 10, 25]
    return {
        "source_file": str(CURRENT_JSON),
        "source_status": prior["status"],
        "selected_nominal_window_A": case["nominal_window_A"],
        "source_corner_trip_min_max_A_before_input_current": {
            "positive": [pos_min, pos_max],
            "negative_magnitude": [neg_min, neg_max],
        },
        "positive_trip_min_max_A_including_added_input_current_bound": [pos_min - positive_pin_leakage_bound_A, pos_max + positive_pin_leakage_bound_A],
        "negative_magnitude_min_max_A_including_added_input_current_bound": [neg_min - negative_input_bias_bound_A, neg_max + negative_input_bias_bound_A],
        "additional_input_current_error_bound_A": {
            "positive_COMP_pin_TT_leakage_at_47ohm_source": positive_pin_leakage_bound_A,
            "negative_TLV1704_bias_on_1k_signal_and_26p7k_10k_reference": negative_input_bias_bound_A,
            "note": "conservative additive bound beyond the existing static-corner JSON; assumes the comparator inputs share these stated source resistances",
        },
        "minimum_static_margin_over_120A": margins,
        "pre_threshold_margin_time_us_by_current_slope": {
            f"{slope}_A_per_us": {
                "positive": margins["positive_A"] / slope,
                "negative_magnitude": margins["negative_magnitude_A"] / slope,
            }
            for slope in slopes
        },
        "time_formula": "(minimum static trip magnitude - 120 A) / |dI/dt|",
        "interpretation": "budget from 120 A to the minimum static trip threshold only; excludes comparator/Break/gate delay and therefore is not total overcurrent duration or allowed overshoot",
        "omissions_in_source_calculation": prior["assumptions"]["omitted_effects"],
    }


def build_report() -> dict:
    return {
        "title": "FOC protection path and fail-state screening",
        "date": "2026-10-11",
        "status": "engineering_screen_not_schematic_or_hardware_acceptance",
        "scope": ["P0-1 power loss/default-off/recovery", "P0-2 threshold error/path delay"],
        "thermal_scope": "excluded; PCB layout and thermal design/validation are user responsibilities",
        "safe5_comp4": safe_threshold_screen(),
        "current_window": current_window_and_slope_screen(),
        "current_path_detail": {
            "CSA_output_slope_nominal_mV_per_A": 0.5e-3 * 10.0 * 1e3,
            "negative_path": {
                "series_resistor_each_ohm": 1000,
                "series_resistor_tolerance": 0.01,
                "TLV1704_input_bias_current_max_nA": 20,
                "series_resistor_only_bias_drop_uV": 20 * 1e-9 * 1000 * 1.01 * 1e6,
                "VNEG_REF_divider_Rtop_ohm": 26700,
                "VNEG_REF_divider_Rbottom_ohm": 10000,
                "VNEG_REF_divider_tolerance_each": 0.01,
                "VNEG_REF_divider_max_thevenin_ohm": (26700 * 1.01) * (10000 * 1.01) / ((26700 * 1.01) + (10000 * 1.01)),
                "combined_differential_bias_voltage_bound_uV": 20e-9 * (((26700 * 1.01) * (10000 * 1.01) / ((26700 * 1.01) + (10000 * 1.01))) + (1000 * 1.01)) * 1e6,
                "combined_equivalent_current_error_upper_A": (20e-9 * (((26700 * 1.01) * (10000 * 1.01) / ((26700 * 1.01) + (10000 * 1.01))) + (1000 * 1.01))) / (0.000492525 * 9.7),
                "TLV1704_VOS_max_mV_full_temp": 5.5,
                "TLV1704_propagation_delay_typ_ns": {
                    "output_high_to_low": 460,
                    "output_low_to_high": 560,
                },
                "delay_test_condition": "100 mV overdrive, 15 pF load, 5.1 kOhm pull-up; typical only",
                "input_filter": "No intentional capacitor is fitted on the 1 kOhm-to-TLV1704 comparator input branch; do not infer a guaranteed RC delay without its input capacitance/source model.",
            },
            "positive_path": {
                "SO_ADC_series_resistor_ohm": 47,
                "SO_ADC_shunt_capacitance_F": 220e-12,
                "COMP_TT_pin_leakage_abs_max_nA": 150,
                "COMP_pin_leakage_voltage_error_upper_uV_if_tapped_after_47ohm": 150e-9 * 47 * 1e6,
                "equivalent_current_error_upper_A_if_tapped_after_47ohm": (150e-9 * 47) / (0.000492525 * 9.7),
                "nominal_RC_tau_ns": 47 * 220e-12 * 1e9,
                "nominal_10_90_rise_ns": 2.197224577 * 47 * 220e-12 * 1e9,
                "limitation": "This is the external ADC-node RC screen only. Confirm the exact comparator tap and include CSA output impedance, ADC sample-and-hold loading and selected sample time in firmware/bench validation.",
            },
            "CSA_response": {
                "settling_time_typ_ns": 500,
                "condition": "DRV835x §7.5 typical value for a 0.5 V CSA output step at GCSA=10",
                "guaranteed_maximum": None,
                "limitation": "Settling time, bandwidth and slew rate do not provide a guaranteed maximum response for the selected small fault overdrive or current slope.",
            },
            "DRV8353_input_to_gate_propagation": {
                "listed_delay_ns": 200,
                "datasheet_locator": "Table 7.5, single listed value; no min/max limits stated",
                "use": "context only; not a guaranteed max and not a VGS turn-off or gate-discharge bound",
            },
        },
        "required_schematic_delta_candidate": [
            "R_SAFEADC_TOP 47k -> 27.4k 1%; R_SAFEADC_BOT 47k -> 10k 1%; C_SAFEADC 10nF -> 1nF C0G",
            "Move SAFE_ADC to PB0 (COMP4_INP/ADC1_IN15); move TEMP_MOS_ADC to PA2 (ADC1_IN3); update coreboard connector pin mapping consistently",
            "No extra IC. The working root is editing the native source; this report does not assert those edits are saved in current source/evidence.",
        ],
        "power_loss_assessment": {
            "safe5_lost_while_mcu_3v3_alive": "With the COMP4 candidate configured and MCU/VDDA valid, SAFE_ADC falling below VREFINT is a hardware Break request independent of CPU execution. U3 TLV1704 and U4 dump driver are unpowered, so local dump is unavailable. The three 1k SO resistors limit comparator-input current under stated source/rail assumptions; they do not prove an unpowered U3 safe output, prevent rail backfeed, or create an energy sink.",
            "main5_or_mcu_3v3_lost_while_safe5_alive": "COMP4/MCU no longer supplies a valid control function. DRV_ENABLE has the existing external 10k pull-down. DRV8353 Table 7.5 lists 100k internal pull-downs for INHx/INLx without min/max limits; do not claim six board-level PWM inputs are guaranteed low from that single listed value. Configure BOR level 4 and verify option-byte readback; firmware must set all six timer outputs LOW, MOE=0/AOE=0 before raising ENABLE or ARM. No six PWM input pull-downs are added in this candidate. SAFE-powered bus dump may remain available while BUS/SAFE5 stay in range.",
            "both_rails_lost": "Driver enters UVLO/sleep-dependent behavior; local dump is unavailable. Body-diode regen can still charge BUS after PWM/gates turn off. A source sink, braking path, or physical isolation/energy handling must close this system-level path.",
            "partial_power_mcu_adc": "STM32G431 TT inputs include PB0 and PA2. DS Table 15 says positive injection is not possible up to the stated maximum and gives TT positive injection 0 mA; Table 14 absolute max is 4.0 V. This is not a positive clamp-current path to calculate, and it does not guarantee ADC/comparator functionality with VDD/VDDA off or below its operating range. Measure VREF_CORE, MCU_3V3, SAFE_ADC and each SO_RAW/SO_ADC node together under independent-rail loss; do not infer normal operation or zero rail interaction from the injection footnote alone.",
            "cpu_hang_after_setup": "A configured hardware COMP4-to-TIM1 Break path can remain active independent of CPU execution, provided VDD/VDDA remain valid and setup was completed. Fault comparator stuck-healthy, timer configuration upset, or core power loss remains a single-point limitation.",
            "dump_device_faults": "Q7 drain-source short cannot be cleared by a gate-off command; resistor open prevents energy absorption. Thermal switch/firmware alone does not prove a disconnect or residual-energy path. Bench/vehicle system must add/validate independent interruption and absorption behavior.",
            "rearm": "Keep TIM1 AOE=0 and MOE=0 on reset/fault. No automatic retry. Require explicit DISARM, validate configuration/readback, SAFE5 healthy, zero-current/reference checks, and explicit ARM after fault recovery.",
        },
        "delay_assessment": {
            "full_chain_formula": "T_fault = sensor/CSA response + applicable external RC + comparator + TIM1 Break/output-state update + DRV8353/gate discharge. SAFE undervoltage additionally includes SAFE5 rail decay and SAFE_ADC RC.",
            "finite_guaranteed_total_max": None,
            "reason": "TI DRV8353 CSA 500 ns settling is typical for a 0.5 V output step, not a guaranteed max; TLV1704 460/560 ns are typical at 100 mV overdrive, 15 pF and 5.1 kOhm pull-up; DRV8353 input-to-gate delay 200 ns is a single listed value without min/max; ST COMP max 31/35 ns is conditional on a 200 mV input step, 100 mV overdrive and 50 pF. TIM1 transfer, output idle/deadtime behavior, actual gate discharge, SAFE5/LDO decay, current slope and loaded board waveforms have no end-to-end guaranteed bound here.",
            "measurement": "Capture current step/CSA_SO_RAW, comparator output or BREAK_N, all six INHx/INLx at U1, all six VGS, phase current, BUS current/voltage, SAFE5, MCU_3V3, VREF_CORE, DRV_ENABLE and nFAULT on one synchronized acquisition across positive/negative overcurrent, SAFE5 loss, main5/MCU loss and BUS overvoltage. Verify TIM1 OIS/OISN, OSSI/OSSR, deadtime, BKE/BKP/BKINP/BKCMPxP and AOE/MOE readback.",
        },
        "acceptance_state": {
            "P0_1": "NOT_CLOSED: COMP4 candidate can request Break only while MCU_3V3/VDDA and timer path are valid; active dump is unavailable on SAFE5 failure; body-diode regeneration/energy sink is a system-level open item; default PWM-low is not board-level guaranteed without bench evidence.",
            "P0_2": "NOT_CLOSED: static threshold and conditional RC screens are calculable, but no guaranteed total response maximum or allowed overcurrent overshoot is available without current-slope and hardware timing evidence.",
            "firmware_minimum": "BOR level 4 option bytes with readback; COMP VREFINT scaler and COMP4 startup wait >=205 us before ARM; all six PWM low, MOE=0, AOE=0, safe TIM1 Break idle/deadtime configuration and readback; ENABLE remains low until all checks pass; explicit ARM only; no automatic restart.",
        },
        "sources": [
            {"document": "ST STM32G431x6/x8/xB DS12589 Rev 6", "url": "https://www.st.com/resource/en/datasheet/stm32g431r8.pdf", "pages": "Table 12 pp.52-53; Tables 14-17 pp.68-71; Table 19 p.72; Table 20 p.74; Table 53 p.110; Table 73 p.143"},
            {"document": "ST RM0440 Rev 9", "url": "https://www.st.com/resource/en/reference_manual/dm00355726.pdf", "pages": "Tables 199-200 (COMP input routing); TIM1_AF1 and break polarity section"},
            {"document": "ST CMSIS STM32G431 device header", "url": "https://raw.githubusercontent.com/STMicroelectronics/cmsis-device-g4/master/Include/stm32g431xx.h", "pages": "COMP_CSR fields and TIM1_AF1_BKCMP4E/BKCMP4P definitions; no COMP_CSR_OUTSEL"},
            {"document": "ST STM32G4 LL COMP header", "url": "https://raw.githubusercontent.com/STMicroelectronics/stm32g4xx-hal-driver/master/Inc/stm32g4xx_ll_comp.h", "pages": "VREFINT selection uses INMSEL=011 and SCALEN; no BRGEN for full VREFINT; delay constants 200/5 us"},
            {"document": "TI DRV835x Rev A", "url": "https://www.ti.com/lit/ds/symlink/drv8353.pdf", "pages": "Table 7.5 p.12 (100 kΩ INHx/INLx pull-down and 200 ns tPD are single listed values without min/max); §7.5 pp.14-15 (CSA gain/VOS/typical settling and drift); §8.4.1.1 p.53 (sleep mode)"},
            {"document": "TI TLV1704 Rev D", "url": "https://www.ti.com/lit/ds/symlink/tlv1704.pdf", "pages": "Table 7.1 p.6 (input absolute maximum/clamp); Tables 7.6/7.7 p.7 (bias current, typical switching characteristics)"},
            {"document": "TI TPS7A4001 Rev B", "url": "https://www.ti.com/lit/ds/symlink/tps7a4001.pdf", "pages": "Table 6.5 p.5; adjustable-output equation and guidance pp.9-10"},
        ],
    }


if __name__ == "__main__":
    report = build_report()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    safe = report["safe5_comp4"]["planned_network"]
    current = report["current_window"]
    print(json.dumps({
        "safe_trip_V": safe["COMP4_trip_SAFE5_V"],
        "safe_tau_us": safe["tau_corner_us"],
        "safe_ideal_step_crossing_us": safe["ideal_source_step_to_COMP4_trip_us"],
        "current_trip_min_A": {
            "positive": current["positive_trip_min_max_A_including_added_input_current_bound"][0],
            "negative_magnitude": current["negative_magnitude_min_max_A_including_added_input_current_bound"][0],
        },
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))
