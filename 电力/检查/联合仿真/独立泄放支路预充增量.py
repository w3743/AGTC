"""Bound the new branch's addition to the existing candidate precharge budget."""
from hashlib import sha256
import json
from math import log
from pathlib import Path

HERE = Path(__file__).resolve().parent


def charge_time(vpack, capacitance, load, limit, resistance, gap):
    if gap <= resistance * load or limit <= load:
        return None
    transition_gap = resistance * limit
    constant_current = capacitance * max(0, vpack - transition_gap) / (limit - load)
    taper = resistance * capacitance * log((transition_gap - resistance * load) /
                                          (gap - resistance * load))
    return constant_current + taper


def main():
    inputs = HERE / '联合行为模型输入.json'
    p = json.loads(inputs.read_text(encoding='utf-8-sig'))['parameters']['precharge']
    rail = json.loads((HERE / '同域泄放互锁核算.json').read_text(encoding='utf-8'))
    extra_load = rail['source_thermal']['input_current_budget_a']
    # An allocation, not a measured inventory: all downstream capacitors are
    # conservatively charged to the full source maximum, including the 2.5 V cap.
    input_cap_high = 2.2e-6 * 1.2 * 1.15
    output_cap_allocation_high = (4 * 22e-6 + 2 * 2.2e-6 + 0.3e-6) * 1.2 * 1.15
    output_charge = output_cap_allocation_high * rail['rail']['source_v'][1]
    old_load_allocation = 0.099
    total_load = old_load_allocation + extra_load
    limit = p['tps2490_candidate_current_min_a']
    resistance = p['tps2490_candidate_series_resistance_ohm']
    gap = p['tps2490_candidate_gap_comparison_v']
    minimum_net_current_before_gap = min(limit, gap / resistance) - total_load
    charge_delay_bound = output_charge / minimum_net_current_before_gap
    overhead = p['tps2490_startup_reserve_s'] + sum(p['main_proof_time_components_s'].values())
    cases = []
    for pack in (22.5, 37.8):
        before = charge_time(pack, p['whole_bus_cap_initial_tolerance_high_f'],
                             old_load_allocation, limit, resistance, gap)
        after = charge_time(pack, p['whole_bus_cap_initial_tolerance_high_f'] + input_cap_high,
                            total_load, limit, resistance, gap)
        sequence = after + charge_delay_bound + overhead
        cases.append({'pack_v': pack, 'old_charge_time_s': before,
                      'new_charge_time_s': after,
                      'extra_cap_charge_delay_upper_bound_s': charge_delay_bound,
                      'sequence_upper_comparison_s': sequence,
                      'shortest_timer_s': p['tps2490_candidate_window_s'][0],
                      'timer_margin_s': p['tps2490_candidate_window_s'][0] - sequence})
    result = {
        'status': 'CANDIDATE_PRECHARGE_BUDGET_ONLY_HOLD',
        'source_sha256': {inputs.name: sha256(inputs.read_bytes()).hexdigest()},
        'original_noncapacitive_allocation_a': old_load_allocation,
        'independent_branch_input_current_budget_a': extra_load,
        'new_total_noncapacitive_allocation_a': total_load,
        'new_direct_input_capacitance_high_f': input_cap_high,
        'downstream_capacitance_high_allocation_f': output_cap_allocation_high,
        'downstream_charging_input_charge_upper_bound_c': output_charge,
        'downstream_charging_energy_from_37p8v_upper_bound_j': 37.8 * output_charge,
        'additional_direct_input_cap_energy_at_37p8v_j': 0.5 * input_cap_high * 37.8**2,
        'minimum_net_precharge_current_before_target_a': minimum_net_current_before_gap,
        'cases': cases,
        'paper_timer_comparison_met': all(row['timer_margin_s'] > 0 for row in cases),
        'bound_conditions': [
            '99 mA is the old candidate allocation, not a measured whole-vehicle load inventory.',
            'New 19.3 mA already includes the 16 mA output budget and 3.3 mA GND-current allowance; do not add analogue current twice.',
            'The output-bank charging charge is added separately, not falsely treated as direct BUS capacitance. Current limit and series resistance are the existing TPS2490 candidate assumptions.',
            'Capacitor allocation and total DC load must cover the final native BOM; any larger replacement must rerun this calculation.',
            'Capacitance high corners include initial +20% and X7R temperature +15%; no credit is taken for DC-bias reduction.',
            'New input protection DC resistance, source turn-on profile, minimum TPS current, FET SOA, current-limit peak and actual load enable timing still need qualified evidence.',
            'Existing safe main-closure proof and READY handshake remain mandatory. Positive timer margin is not main-path proof or precharge qualification.',
            'This is a local supply-budget calculation; it is not the final all-board stopping simulation.',
        ],
    }
    if not result['paper_timer_comparison_met']:
        raise ValueError('The extra branch consumes the candidate precharge timer budget')
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'timer_margins_s': [c['timer_margin_s'] for c in cases]}))


if __name__ == '__main__':
    main()
