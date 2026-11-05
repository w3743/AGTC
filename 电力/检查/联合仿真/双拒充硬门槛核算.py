"""比较独立推挽比较器的门槛、接管能量窗与退出时序。"""

from itertools import product
import json
from pathlib import Path


BUS_CAP_F = 0.05286
BUS_FULL_V = 37.8
BUS_CEILING_V = 39.009
MIN_DUMP_R_OHM = 0.779


def energy_between(low, high):
    return 0.5 * BUS_CAP_F * (high**2 - low**2)


def screen_divider(high_ohm):
    rising, falling, hysteresis = [], [], []
    for high_error, low_error, feedback_error, ref_error, offset_v, bias_a, output_low, output_high in product(
        (-0.0025, 0.0025), (-0.0025, 0.0025), (-0.015, 0.015), (-0.0011, 0.0011),
        (-0.004, 0.004), (-2e-9, 2e-9), (0, 0.25), (11.15, 13.2),
    ):
        high = high_ohm * (1 + high_error)
        low = 10_000 * (1 + low_error)
        feedback = 6_800_000 * (1 + feedback_error)
        reference = 2.5 * (1 + ref_error) + offset_v
        common = reference * (1 + high / low + high / feedback) + bias_a * high
        rising.append(common - output_low * high / feedback)
        falling.append(common - output_high * high / feedback)
        hysteresis.append(rising[-1] - falling[-1])
    # This off-time screen assumes zero regeneration and full conduction at the cold low resistance.
    maximum_power_at_falling_corner = min(falling)**2 / MIN_DUMP_R_OHM
    return {
        'divider_high_ohm': high_ohm, 'rising_v': [min(rising), max(rising)],
        'falling_v': [min(falling), max(falling)], 'same_unit_min_hysteresis_v': min(hysteresis),
        'latest_on_energy_j': energy_between(max(rising), BUS_CEILING_V),
        'earliest_release_to_full_battery_energy_j': energy_between(BUS_FULL_V, min(falling)),
        'off_time_to_full_battery_no_regen_us': max(0, energy_between(BUS_FULL_V, min(falling))) / maximum_power_at_falling_corner * 1e6,
    }


def main():
    speed = 15 / 3.6
    stops = []
    for delay_s in (0, 0.02, 0.05, 0.10):
        distance = 3.0 - speed * delay_s
        acceleration = speed**2 / (2 * distance)
        stops.append({'torque_build_delay_s': delay_s, 'distance_available_m': distance,
                      'net_brake_force_n_translation_only': 30 * acceleration,
                      'initial_power_w_translation_only': 30 * acceleration * speed,
                      'motor_torque_nm_each_ideal': 30 * acceleration * 0.03 / 8})
    dividers = [screen_divider(high) for high in (143_000, 143_300, 143_600, 144_000)]
    baseline = dividers[2]
    result = {
        'qualification': 'HOLD',
        'scope': 'TLV1811 + 高精度 REF5025 候选静态角与简化能量筛查。仅独立反馈，不经共享 DUMP_OFF。未含精确料号、局部母线差值、参考热滞/寿命实测、失电带压、电阻和线束热、真实门极时序。',
        'ti_comparator': 'https://www.ti.com/lit/gpn/tlv1811',
        'ti_reference': 'https://www.ti.com/lit/gpn/REF50',
        'comparison': {'bus_cap_f': BUS_CAP_F, 'bus_full_v': BUS_FULL_V, 'bus_ceiling_v': BUS_CEILING_V,
                       'dump_resistance_low_ohm': MIN_DUMP_R_OHM, 'current_unproven': True},
        'parameter_notes': '分压上下臂各总误差±0.25%，反馈±1.5%。REF总误差暂取±0.11%，输入失调±4mV、偏流±2nA。推挽低0–0.25V，高11.15–13.2V。所有电阻实际公差和温漂待绑定。',
        'divider_comparisons': dividers,
        'launch_speed_stop_delay_comparisons': stops,
        'example_143p6k_margin_after_1ms_390w_j': baseline['latest_on_energy_j'] - 0.390,
        'blocking': '供电/参考资格、带电 BUS 失电反灌、真正接管时间、低压释放时间、每次硬泄放脉冲及重复热量均须闭合。',
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
