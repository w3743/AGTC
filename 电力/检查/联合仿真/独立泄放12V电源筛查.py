"""Size the independent dump-driver supply. Outputs are design comparisons."""

import json
from itertools import product
from pathlib import Path

from 独立泄放互锁角核算 import divider_range


def main():
    bus_v = 39.009
    feedback_top = 93_100
    feedback_bottom = 10_000
    tolerance = 0.001
    rail_values = []
    # +/-3% is a conservative design allowance, not a high-input guarantee.
    for ref, top_error, bottom_error, bias in product(
            (1.173 * 0.97, 1.173 * 1.03), (-tolerance, tolerance),
            (-tolerance, tolerance), (-0.1e-6, 0.1e-6)):
        hi = feedback_top * (1 + top_error)
        lo = feedback_bottom * (1 + bottom_error)
        rail_values.append(ref * (1 + hi / lo) + bias * hi)
    rail_min, rail_max = min(rail_values), max(rail_values)
    uv_assert = divider_range(267_000, 10_000, tolerance, (0.396, 0.404))
    uv_release = divider_range(267_000, 10_000, tolerance, (0.3995, 0.414))
    gate_charge_c = 300e-9
    switching_hz = 20_000
    static_a = 0.008  # Includes divider, logic, static driver and on-state gate resistor.
    out_cap_f = 4.7e-6
    supply_a = static_a + gate_charge_c * switching_hz
    power_w = (bus_v - rail_min) * supply_a + bus_v * 65e-6
    ideal_droop_v = gate_charge_c / out_cap_f
    assert rail_min - ideal_droop_v > uv_release[1]
    # LT3010E adjustable full-temperature reference range, plus resistor and bias corners.
    alternate_min = 1.237 * (1 + 84_500 * 0.999 / (10_000 * 1.001))
    alternate_max = 1.313 * (1 + 84_500 * 1.001 / (10_000 * 0.999)) + 0.1e-6 * 84_500 * 1.001
    alternate_power = (bus_v - alternate_min) * supply_a + bus_v * 3.3e-3
    assert alternate_min - ideal_droop_v > uv_release[1]
    result = {
        'status': 'HOLD_CIRCUIT_AND_BENCH',
        'source': 'TPS7A4001DGNR, same type as FOC U2/U6, powered from BMS downstream BUS',
        'feedback_ohms': [feedback_top, feedback_bottom],
        'feedback_resistor_tolerance': tolerance,
        'nominal_12_hw_v': 1.173 * (1 + feedback_top / feedback_bottom),
        'rail_design_range_v': [rail_min, rail_max],
        'supervisor_divider_comparison': {
            'parts_ohms': [267_000, 10_000],
            'tolerance': tolerance,
            'fall_v': uv_assert,
            'release_v': uv_release,
            'minimum_rail_margin_to_release_after_ideal_gate_step_v':
                rail_min - ideal_droop_v - uv_release[1],
            'scope': 'TPS386000 12 V sense corner only. The interlock is not qualified.',
        },
        'design_budgets': {
            'static_load_ma': static_a * 1e3,
            'combined_gate_charge_nc': gate_charge_c * 1e9,
            'event_frequency_khz': switching_hz / 1e3,
            'total_output_ma': supply_a * 1e3,
            'effective_input_cap_min_uf': 1,
            'effective_output_cap_min_uf': out_cap_f * 1e6,
            'ideal_gate_step_v': ideal_droop_v,
            'power_at_39p009v_w': power_w,
            'estimated_tj_at_60c_using_jedec_c': 60 + power_w * 66.7,
            'energy_in_10uf_at_12v_mj': 0.5 * 10e-6 * 12**2 * 1e3,
        },
        'shutdown_paper_limit': {
            'max_out_minus_in_v': 0.3,
            'counterexample_v': rail_max,
            'reason': 'With abrupt BUS hard-low and bare LDO, CIN follows BUS while COUT retains voltage.',
            'decision': 'Exclude the bare TPS7A4001 topology from hard-drop qualification.',
        },
        'reverse_safe_source_candidate': {
            'mpn': 'LT3010EMS8E#PBF',
            'feedback_ohms': [84_500, 10_000],
            'nominal_v_without_adjust_bias': 1.275 * 9.45,
            'steady_voltage_corner_v': [alternate_min, alternate_max],
            'minimum_rail_margin_to_release_after_ideal_gate_step_v':
                alternate_min - ideal_droop_v - uv_release[1],
            'power_comparison_w': alternate_power,
            'estimated_tj_at_60c_with_62c_per_w_copper_reference_c': 60 + alternate_power * 62,
            'ground_current_budget_ma': 3.3,
            'scope': 'Vendor supports input grounded with output held. The voltage corner uses the vendor ADJ reference range and 100 nA screening allowance for ADJ bias; verify actual capacitance, dropout, reverse leakage, input inrush and package thermals.',
            'cost': 'ADI family 1k-unit list starting at USD 2.63; actual MPN and delivered pricing pending.',
        },
        'bench_dependencies': [
            '12 V load and total gate charge at actual gate voltage, gate frequency and temperature',
            '12 V ramp, ripple and minimum, effective capacitors and board temperature',
            'inrush with existing precharge, BUS collapse, residual-voltage restart and old-rail fault isolation',
        ],
        'sources': [
            'https://www.ti.com/lit/ds/symlink/tps7a4001.pdf',
            'https://www.ti.com/lit/ds/symlink/tps386000.pdf',
            'https://www.ti.com/lit/ds/symlink/ucc27511a.pdf',
            'https://www.hlktech.net/index.php?id=980',
            'https://www.ti.com/product/LM5164',
            'https://www.analog.com/en/products/lt3010.html',
            'https://www.analog.com/media/en/technical-documentation/data-sheets/lt3010-3010-5.pdf',
        ],
    }
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['status'])


if __name__ == '__main__':
    main()
