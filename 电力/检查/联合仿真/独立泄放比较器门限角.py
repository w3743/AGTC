"""Recalculate the actual series divider, feedback and component-temperature budget."""
from itertools import product
import json
from pathlib import Path


def main():
    # Body temperature is a PCB/qualification constraint, not ambient = junction.
    temperature_delta = 60  # 0..85 C body; furthest from the 25 C reference.
    rising, falling = [], []
    for etop, esmall, ebottom, efeedback, eref, eoffset, ebias, eout in product((-1, 1), repeat=8):
        upper = (143_000 * (1 + etop * (0.001 + 25e-6 * temperature_delta))
                 + 600 * (1 + esmall * (0.01 + 100e-6 * temperature_delta)))
        bottom = 10_000 * (1 + ebottom * (0.001 + 25e-6 * temperature_delta))
        feedback = 6.8e6 * (1 + efeedback * (0.01 + 200e-6 * temperature_delta))
        sense = 2.5 * (1 + eref * 0.0011) + eoffset * 0.004
        gain = 1 + upper / bottom + upper / feedback
        bias_term = ebias * 1.2e-9 * upper
        low_output = 0 if eout < 0 else 0.25
        # Max high is the separate transient qualification envelope. Min high
        # uses the held-supply floor minus the 4 mA output-drop allowance.
        high_output = 6 - 0.25 if eout < 0 else 13.2
        rising.append(sense * gain - low_output * upper / feedback + bias_term)
        falling.append(sense * gain - high_output * upper / feedback + bias_term)
    maximum_trip, minimum_release = max(rising), min(falling)
    bus_cap = 0.05286
    ceiling = 39.009
    full_resistance_low = 0.82 * 0.95
    release_energy = 0.5 * bus_cap * (minimum_release**2 - 37.8**2)
    result = {
        'status': 'CONDITIONAL_COMPONENT_BUDGET_NOT_HARDWARE_QUALIFICATION',
        'body_temperature_constraint_c': [0, 85],
        'ambient_requirement_c': [0, 60],
        'resistors': [
            {'function': 'bus_upper_main', 'ohm': 143000, 'initial_tolerance': 0.001, 'tcr_budget_ppm_per_c': 25, 'mpn': 'PTFR0805B143KP9'},
            {'function': 'bus_upper_trim', 'ohm': 600, 'initial_tolerance': 0.01, 'tcr_budget_ppm_per_c': 100, 'mpn': 'CRCW0805600RFKEAHP'},
            {'function': 'bus_bottom', 'ohm': 10000, 'initial_tolerance': 0.001, 'tcr_budget_ppm_per_c': 25, 'mpn': 'HoAR0603-1/10W-10KR-0.1%-TCR25'},
            {'function': 'feedback', 'ohm': 6800000, 'initial_tolerance': 0.01, 'tcr_budget_ppm_per_c': 200, 'mpn': 'RC0603FR-076M8L'},
        ],
        'reference_total_error_budget_fraction': 0.0011,
        'comparator_offset_budget_v': 0.004,
        'comparator_input_bias_budget_a': 1.2e-9,
        'rising_range_v': [min(rising), maximum_trip],
        'release_range_v': [minimum_release, max(falling)],
        'latest_trip_to_39p009_energy_j': 0.5 * bus_cap * (ceiling**2 - maximum_trip**2),
        'no_absorption_time_at_450p03w_s': 0.5 * bus_cap * (ceiling**2 - maximum_trip**2) / 450.03,
        'zero_regen_min_release_to_37p8_energy_j': release_energy,
        'ideal_full_current_release_delay_budget_s': release_energy / (minimum_release**2 / full_resistance_low),
        'conditions': [
            'Exact resistor TCRs and body temperature, including adjacent power-device heating, must meet these allocations. No assumption that 60 C ambient is the component temperature.',
            'TLV1811 has no internal hysteresis. The external 6.8 Mohm positive feedback is included, with 1% and 200 ppm/C allocations.',
            'Offset/output-drop bounds are datasheet test-condition values used as an application budget. Actual 6 V held-rail floor through operating supply, common mode, temperature, load and reference errors require complete threshold sweeps.',
            'Output low 0..0.25 V and output high 5.75..13.2 V are qualification envelopes, not a claim that every supply is 13.2 V.',
            'Latest trip differs slightly from the historical 38.7382808 V comparison because the actual 600 ohm resistor is 1% and temperature coefficients are explicit. Do not rewrite frozen old simulation results as current evidence.',
            'The zero-regen release time is an ideal energy comparison, not a physical switch-off guarantee. Measure comparator/driver/gate, branch current and battery current after net regen is removed.',
        ],
        'sources': [
            'https://www.ti.com/document-viewer/TLV1811/datasheet/GUID-XXXXXXXX-SF0T-XXXX-XXXX-000000283538',
            'https://www.ti.com/document-viewer/TLV1811/datasheet/GUID-21EE9DF7-0441-4429-8C7D-65598439B616',
        ],
    }
    assert maximum_trip < ceiling and minimum_release > 37.8
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rising_v': result['rising_range_v'], 'release_v': result['release_range_v'],
                      'release_delay_comparison_us': result['ideal_full_current_release_delay_budget_s'] * 1e6}))


if __name__ == '__main__':
    main()
