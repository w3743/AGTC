"""Calculate the selected shared-supply interlock and its qualification budgets."""

from itertools import product
import json
from math import log
from pathlib import Path

RESISTOR_BODY_DELTA_C = 60  # 0..85 C body relative to 25 C reference.
PRECISION_RESISTOR_ERROR = 0.001 + 25e-6 * RESISTOR_BODY_DELTA_C


def source_corners():
    values = []
    for top_error, bottom_error, reference, bias in product(
            (-PRECISION_RESISTOR_ERROR, PRECISION_RESISTOR_ERROR),
            (-PRECISION_RESISTOR_ERROR, PRECISION_RESISTOR_ERROR),
            (1.237, 1.313), (-100e-9, 100e-9)):
        top, bottom = 84_500 * (1 + top_error), 10_000 * (1 + bottom_error)
        values.append(reference * (1 + top / bottom) + bias * top)
    return min(values), max(values)


def sense_corners():
    falling, rising = [], []
    for top_error, bottom_error, threshold, bias, hysteresis_error in product(
            (-PRECISION_RESISTOR_ERROR, PRECISION_RESISTOR_ERROR),
            (-PRECISION_RESISTOR_ERROR, PRECISION_RESISTOR_ERROR), (0.792, 0.808),
            (-100e-9, 100e-9), (-0.015, 0.015)):
        top = 120_000 * (1 + top_error)
        bottom = 10_000 * (1 + bottom_error)
        gain = 1 + top / bottom
        falling.append(threshold * gain + bias * top)
        rising.append((threshold + 0.016 * (1 + hysteresis_error)) * gain + bias * top)
    return {'fall_v': [min(falling), max(falling)],
            'release_v': [min(rising), max(rising)]}


def main():
    thresholds = sense_corners()
    comparator_budget = json.loads(Path(__file__).with_name('独立泄放比较器门限角.json').read_text(encoding='utf-8'))
    latest_dump_trip = comparator_budget['rising_range_v'][1]
    source_min, source_max = source_corners()
    cap_min = 10e-6
    analogue_current_budget = 3e-3
    rail_current_budget = 16e-3
    slew_error_budget = 0.01 + 200e-6 * RESISTOR_BODY_DELTA_C
    resistance_min, resistance_max = 44 * (1 - slew_error_budget), 44 * (1 + slew_error_budget)
    driver_min = source_min - rail_current_budget * resistance_max
    gate_step = 300e-9 / cap_min
    # Conservative full-load GND-current specification; not an efficiency fit.
    source_ground_current_budget = 3.3e-3
    source_power = (39.009 - source_min) * rail_current_budget + 39.009 * source_ground_current_budget
    ctr_min = -log(0.31) * 877e3 * 1.98e-9
    ctr_max = -log(0.25) * 1147e3 * 2.42e-9 + 40e-6
    full_discharge = 0.05 * ctr_max
    # The 17 us bound applies after 20% input overdrive, with CTS unpopulated.
    deep_trip_v = 0.8 * thresholds['fall_v'][0]
    diode_drop_budget = 1.25
    analogue_at_deep_trip = deep_trip_v - diode_drop_budget
    analogue_floor = 6.0
    hold_time = cap_min * (analogue_at_deep_trip - analogue_floor) / analogue_current_budget
    protected_time = full_discharge + 17e-6
    recovery_energy = 0.5 * 0.05286 * (latest_dump_trip**2 - 14**2)
    cold_energy = 0.5 * 0.05286 * (latest_dump_trip**2 - 22.5**2)
    recovery_budget = 0.05
    ctr_with_start = ctr_max + 0.002
    # Slow corner: include MAXIMUM bank capacitance as well as the previously
    # checked minimum capacitance for slew/hold. Treat both banks as charging
    # from zero to HW voltage, conservatively ignoring the diode voltage offset.
    # X7R positive temperature allowance is separate from initial tolerance.
    bank_cap_max = (4 * 22e-6 + 0.3e-6) * 1.2 * 1.15
    source_cap_max = 2.2e-6 * 1.2 * 1.15
    ref_charge_max = 2.2e-6 * 1.2 * 1.15 * 2.50275
    # Before READY, the gate is held low: the 300 nC * 20 kHz = 6 mA
    # switching allocation is zero. Retain the other full 10 mA allocation.
    startup_load_budget = rail_current_budget - 300e-9 * 20_000
    source_current_floor = 0.05
    transition_voltage = source_min - resistance_max * source_current_floor
    final_rail = source_min - resistance_max * startup_load_budget
    last_net_current = (source_min - thresholds['release_v'][1]) / resistance_max - startup_load_budget
    current_limited_charge = bank_cap_max * transition_voltage / (source_current_floor - startup_load_budget)
    taper_charge = resistance_max * bank_cap_max * log(
        (final_rail - transition_voltage) / (final_rail - thresholds['release_v'][1]))
    direct_source_cap_charge = source_cap_max * source_min / (source_current_floor - startup_load_budget)
    reference_charge_delay = ref_charge_max / last_net_current
    # A design qualification budget, not an LT3010 guaranteed startup maximum.
    intrinsic_source_start_budget = 0.002
    recovery_upper_comparison = (current_limited_charge + taper_charge + direct_source_cap_charge
                                 + reference_charge_delay + ctr_with_start + intrinsic_source_start_budget)
    ref_error = 0.0005 + 3e-6 * 100 + 5e-6 * (12.4386 - 6.0)
    # 0.11% total remains a qualification envelope, including effects with no maximum.
    paper_checks = {
        'shared_supply_low_voltage_order': 2.7 < 3.45,
        'release_headroom_with_ideal_gate_step':
            driver_min - gate_step > thresholds['release_v'][1],
        'rise_slew_below_100mv_per_us':
            13.2 / (resistance_min * cap_min) < 100_000,
        'analogue_hold_exceeds_full_ctr_discharge_and_detection': hold_time > protected_time,
        'reference_initial_temp_line_inside_total_budget': ref_error < 0.0011,
        'recovery_budget_inside_450w_energy_window': recovery_budget < recovery_energy / 450.03,
        'ctr_and_internal_start_fit_recovery_budget': ctr_with_start < recovery_budget,
        'maximum_capacitance_charge_and_reset_fit_recovery_budget': recovery_upper_comparison < recovery_budget,
    }
    if not all(paper_checks.values()):
        raise ValueError(f'Interlock design-budget failure: {paper_checks}')
    result = {
        'status': 'EIGHT_LOCAL_BUDGETS_PASS_EXTERNAL_READY_DESIGN_OPEN_HOLD',
        'external_ready_design_open': {
            'failure_path': 'With old 12V_PROT healthy and TPS3760 VDD below its 2.7 V guaranteed RESET-low region, undefined HW_READY can bias the series NPN and falsely illuminate the external READY optocoupler.',
            'existing_uvlo_scope': 'UCC UVLO inhibits its own output in this region; it does not prove that the separately powered external READY LED remains off.',
            'required_resolution': 'Qualify the external READY sink against low VDD, retaining GUARD_OK and the main-only BMS interlock without blocking the precharge bootstrap. Exact parts and native implementation remain under review.',
            'eight_checks_scope': 'The eight local rail/timing/energy comparisons below do not close this external READY path.',
        },
        'scope': {
            'defined_running_fault': 'BMS main and old 12 V/CORE/CAN withdrawn; downstream BUS positive and return conductors retained. The independent supply remains connected.',
            'supply_input_loss_recovery': 'BUS falls then rises through 14 V; measure all source, reference and supervisor startup time.',
            'cold_high_bus_start': 'Hard READY handshake must prevent traction and commanded regen until the independent chain is ready. A cold 37.8 V source plus immediate 450 W injection has insufficient startup energy margin.',
            'excluded_additional_fault': 'Independent regulator forced to zero with BUS still healthy is a separate internal supply fault. No 3 m stopping credit is assigned to it.',
            'vehicle_stop_qualification': 'HOLD; neither this calculation nor the reset timer proves motor torque or a 3 m stop.',
        },
        'selected_parts': {
            'supply': 'LT3010EMS8E#PBF',
            'supervisor': 'TPS3760A012DYYR',
            'reference': 'REF5025ID, high grade; AI standard grade is not an equivalent',
            'comparator': 'TLV1811DBVR',
            'ctr': '2.2 nF C0G, total effective 1.98–2.42 nF on CTR/MR pin 9; CTS pin 10 unpopulated',
            'feedforward': '2.2 nF C0G, same SKU as CTR; minimum 1.98 nF > 1.592 nF required by the LT3010 application condition',
            'rail_slew': 'Two FH RAG-06K22R0FT, 22 ohm 1% 0.5 W, in series; >=10 uF effective at the common driver/supervisor node',
            'uv_divider_ohms': [120_000, 10_000],
            'hard_ready': 'Reuse the existing GUARD-driven READY LED sink, with a series BC847C enabled by HW_READY through 10k and a 100k base pull-down. HW_READY pull-up is 4.7k, external pull-down 100k.',
            'analogue_hold': 'Diode and >=10 uF effective supply both REF5025 and TLV1811',
            'resistor_temperature': 'Source feedback and POR divider <=25 ppm/C, body 0..85 C; slew resistance budget <=200 ppm/C. Initial tolerance and drift are added, not conflated.',
        },
        'paper_checks': paper_checks,
        'threshold_corners': thresholds,
        'rail': {
            'source_v': [source_min, source_max],
            'continuous_output_current_budget_ma': rail_current_budget * 1e3,
            'minimum_driver_supply_v': driver_min,
            'ideal_300nc_gate_step_v': gate_step,
            'minimum_margin_to_release_v': driver_min - gate_step - thresholds['release_v'][1],
            'maximum_rising_slew_v_per_us': 13.2 / (resistance_min * cap_min) / 1e6,
            'per_slew_resistor_peak_voltage_v': 13.2 * (1 + slew_error_budget) / 2,
            'per_slew_resistor_five_second_overload_voltage_v': 2.5 * (0.5 * 22 * 0.99)**0.5,
            'slew_resistor_scope': 'Peak-voltage comparison only against the vendor five-second short-overload test condition. Full high-capacitance startup current, pulse duration/energy, hot-board behavior and repeated starts remain qualifications; the LT3010 50 mA rating is not a peak-current upper bound.',
            'ready_low_max_v': 0.3,
            'ucc_low_guarantee_v': 1.0,
            'required_ready_high_v': 2.4,
            'maximum_supervisor_sink_ma': 13.2 / (4700 * (1 - slew_error_budget)) * 1e3,
            'hard_ready_base_current_measurement_min_ma': 0.5,
            'hard_ready_led_current_measurement_ma': [5.0, 6.9],
            'ready_load_note': 'High-level loading and measured READY must meet the UCC input threshold, without treating its typical internal pull-down as a guarantee.',
        },
        'source_thermal': {
            'output_current_budget_a': rail_current_budget,
            'ground_current_budget_a': source_ground_current_budget,
            'input_current_budget_a': rail_current_budget + source_ground_current_budget,
            'maximum_comparison_input_v': 39.009,
            'dissipation_w': source_power,
            'ambient_max_c': 60.0,
            'operating_junction_max_c': 125.0,
            'effective_theta_ja_upper_bound_without_margin_c_per_w': (125 - 60) / source_power,
            'junction_at_62c_per_w_reference_c': 60 + 62 * source_power,
            'scope': 'DC sizing only. The exposed pad must be soldered to GND copper. Actual PCB thermal resistance, source pulse/short behavior, input peaks and capacitor charging remain qualifications; the 62 C/W copper reference is not a finished-board guarantee.',
        },
        'timing': {
            'ctr_min_s': ctr_min, 'ctr_max_s': ctr_max,
            'maximum_cold_start_plus_ctr_s': ctr_with_start,
            'full_discharge_fault_duration_lower_bound_s': full_discharge,
            'deep_overdrive_supply_v': deep_trip_v,
            'analogue_min_at_deep_trip_v_with_diode_budget': analogue_at_deep_trip,
            'analogue_floor_v': analogue_floor,
            'analogue_current_budget_ma': analogue_current_budget * 1e3,
            'conditional_hold_s': hold_time,
            'hold_margin_s': hold_time - protected_time,
            'hold_scope': 'Requires verified diode drop/leakage, effective capacitance, load and recharge. For repeated glitches, measure the held node minimum and enforce the qualified waveform envelope.',
            'reference_establishment': 'Measure REF within 2.49725–2.50275 V before every READY rise, across zero/residual initial charge, actual load and 0–60 C ambient with component self-heating included.',
        },
        'energy_recovery': {
            'capacitance_f': 0.05286, 'comparison_net_regen_w': 450.03,
            'latest_dump_trip_budget_v': latest_dump_trip,
            'bus14_to_latest_dump_threshold_j': recovery_energy,
            'bus22p5_to_latest_dump_threshold_j': cold_energy,
            'bus14_to_ready_measurement_max_s': recovery_budget,
            'available_time_at_450w_s': recovery_energy / 450.03,
            'margin_at_50ms_s': recovery_energy / 450.03 - recovery_budget,
            'maximum_net_regen_for_50ms_recovery_w': recovery_energy / recovery_budget,
            'cold_37p8v_to_threshold_at_450w_s':
                0.5 * 0.05286 * (latest_dump_trip**2 - 37.8**2) / 450.03,
            'ready_to_actual_current_max_s': 0.0005,
            'scope': 'Power, effective BUS capacitance and response bounds must be replaced by qualified system corners. Recovery does not establish vehicle stopping capability.',
        },
        'maximum_capacitance_startup': {
            'bank_capacitance_high_f': bank_cap_max,
            'direct_source_capacitance_high_f': source_cap_max,
            'source_available_current_floor_a': source_current_floor,
            'load_budget_during_charge_a': startup_load_budget,
            'pre_ready_load_reason': 'The 16 mA running budget includes 6 mA gate charge at 300 nC and 20 kHz. RESET holds IN+ low before READY, so only the other full 10 mA is allocated during charging. Reference output charging is additionally budgeted as charge, not hidden in DC current.',
            'current_limited_phase_s': current_limited_charge,
            'constant_voltage_taper_s': taper_charge,
            'direct_source_cap_charge_allowance_s': direct_source_cap_charge,
            'reference_output_charge_allowance_s': reference_charge_delay,
            'intrinsic_source_turn_on_delay_budget_s': intrinsic_source_start_budget,
            'full_recovery_upper_comparison_s': recovery_upper_comparison,
            'margin_to_50ms_s': recovery_budget - recovery_upper_comparison,
            'old_22nf_counterexample_min_charge_plus_timer_s':
                current_limited_charge + taper_charge + direct_source_cap_charge +
                (-log(0.25) * 1147e3 * 24.2e-9 + 40e-6 + 0.002),
            'scope': 'Conditional charge bound with zero initial charge, initial tolerance +20%, X7R temperature +15%, and pre-READY 10 mA load. Source turn-on <=2 ms and >=50 mA available charging current are qualification conditions, not measured or full-startup datasheet guarantees. Full BUS14-to-READY <=50 ms is the governing physical criterion. The old 22 nF timer fails this allocated worst corner; changing to 2.2 nF adds no components.',
        },
        'measurements': [
            'Actual REF settling before READY, driver threshold, held analogue supply and reset recovery over the specified cold/residual/brownout waveform envelope.',
            'Source startup: BUS rising through 14 V to valid READY <=50 ms at the stated energy corner; cold BUS step to 22.5 V with concurrent injection separately.',
            'Old-rail loss: new source and return remain functional, without unintended backfeed or a new reset during the defined main fault.',
            'Qualified independent chain: threshold to branch current matching net regen <=0.5 ms, near-bridge voltage and release current/energy within approved bounds.',
        ],
        'sources': [
            'https://www.ti.com/lit/ds/symlink/tps3760.pdf',
            'https://www.ti.com/lit/ds/symlink/ucc27511a.pdf',
            'https://www.ti.com/lit/ds/symlink/ref50.pdf',
            'https://www.ti.com/lit/ds/symlink/tlv1811.pdf',
            'https://www.analog.com/media/en/technical-documentation/data-sheets/lt3010-3010-5.pdf',
        ],
    }
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['status'])


if __name__ == '__main__':
    main()
