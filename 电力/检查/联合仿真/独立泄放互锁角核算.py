"""Screen the selected TPS386000 interlock before bench qualification."""

import json
from itertools import product
from pathlib import Path


def divider_range(top, bottom, tolerance, threshold):
    values = []
    for top_error, bottom_error, sense_v, bias in product(
            (-tolerance, tolerance), (-tolerance, tolerance), threshold, (-25e-9, 25e-9)):
        hi = top * (1 + top_error)
        lo = bottom * (1 + bottom_error)
        values.append(sense_v * (1 + hi / lo) + bias * hi)
    return [min(values), max(values)]


def main():
    # TPS386000 data-sheet limits, not a simulation of startup output current.
    fall, uv_release = (0.396, 0.404), (0.3995, 0.414)
    full_ref_min_v, no_false_trip_floor_v = 2.5*(1-0.0011), 2.47
    window = {
        'supply_12_uv_assert_v': divider_range(270_000, 10_000, 0.01, fall),
        'supply_12_uv_release_v': divider_range(270_000, 10_000, 0.01, uv_release),
        'reference_uv_assert_v': divider_range(49_900, 10_000, 0.001, fall),
        'reference_uv_release_v': divider_range(49_900, 10_000, 0.001, uv_release),
    }
    # Necessary conditions for one passive divider to provide both limits,
    # even before resistor error and input bias are added.
    ratio_for_fast_ref_inhibit = no_false_trip_floor_v / fall[0]
    ratio_for_reference_to_enable = full_ref_min_v / uv_release[-1]
    ramp_us_per_v = 99*1e-6/(3.3*1.04)*1e6
    assert ratio_for_fast_ref_inhibit > ratio_for_reference_to_enable
    assert ramp_us_per_v > 15
    assert window['reference_uv_release_v'][1] < full_ref_min_v
    result = {
        'qualification': 'HOLD',
        'selected_prototype': 'Reuse TPS386000RGPR and TLV76033DBZR types on an independent domain; wire-OR resets, same filtered VDD pulls READY up, external pull-down, no READY reservoir capacitor.',
        'windows': window,
        'startup': {
            'ct_open_release_ms_min_typ_max': [14, 20, 24],
            'datasheet_por_ramp_min_us_per_v': 15,
            '100ohm_1uF_effective_min_ramp_us_per_v_for_3p432V_ideal_step': ramp_us_per_v,
            'reset_load_at_0p9v_0p2v_ready_with_10k_uA': (0.9-0.2)/10_000*1e6,
            'datasheet_test_current_at_0p9v_uA': 15,
            'ready_high_v_typical_3p3_with_200k_input': 3.3/(1+10_000/100_000+10_000/200_000),
            'ucc_in_plus_guaranteed_low_and_high_v': [1.0, 2.4],
            'qualification': 'Filtered ramp satisfies a slew comparison, not output-current guarantee during POR.',
        },
        'reference_counterexample': {
            'acceptable_ref_lower_bound_screen_v': no_false_trip_floor_v,
            'valid_ref_min_screen_v': full_ref_min_v,
            'required_divider_factor_at_least': ratio_for_fast_ref_inhibit,
            'allowed_divider_factor_at_most': ratio_for_reference_to_enable,
            'conclusion': 'No overlapping divider factor even with ideal resistors. A 2.445 V plateau can count as valid on some units. For combined supply brownout, verify 12 V UV disarms before REF exits the safe range and restarts the whole delay. Independent REF drift is not covered by this comparator threshold alone.',
        },
        'sources': [
            'https://www.ti.com/lit/ds/symlink/tps386000.pdf',
            'https://www.ti.com/lit/ds/symlink/ucc27511a.pdf',
            'https://www.ti.com/lit/ds/symlink/tlv760.pdf',
        ],
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('TPS386000 corner comparison: release ratio conflict reproduced, HOLD')


if __name__ == '__main__':
    main()
