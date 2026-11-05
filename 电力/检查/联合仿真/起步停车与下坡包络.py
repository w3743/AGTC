"""使用现有机械基线和超容寿命角，比较起步供电与电制动包络。"""

import json
from math import hypot, sqrt
from pathlib import Path


MASS = 30.0
TOP_SPEED = 15.0 / 3.6
BUS_CAP = 0.05286
BUS_START = 37.8
BUS_LIMIT = 39.009
SC_CAP = 280.0 / 8
SC_ESR = 8 * 0.0056
SC_START = 8 * 2.0
SC_TERMINAL_LIMIT = 8 * 2.2
SC_CHARGE_A = 20.0
EFFICIENCY = 0.95
DT = 0.001


def brake(speed, distance, grade=0.0, descent_time=0.0, dt=DT):
    deceleration = speed**2 / (2 * distance)
    stop_time = speed / deceleration
    gravity_force = MASS * 9.80665 * grade / hypot(1, grade)
    peak_w = (MASS * deceleration + gravity_force) * speed
    sc_v = SC_START
    bus_extra_j = 0.0
    peak_bus_j = 0.0
    # An ideal power controller services ongoing regeneration and excess BUS energy.
    for phase_s, power in ((descent_time, lambda _: gravity_force * speed),
                           (stop_time, lambda t: peak_w * (1 - t / stop_time))):
        elapsed = 0.0
        while elapsed < phase_s:
            step = min(dt, phase_s - elapsed)
            demand_w = power(elapsed + step / 2) + bus_extra_j / step
            demand_a = 2 * EFFICIENCY * demand_w / (sc_v + sqrt(sc_v**2 + 4 * SC_ESR * EFFICIENCY * demand_w))
            current_a = max(0.0, min(SC_CHARGE_A, (SC_TERMINAL_LIMIT - sc_v) / (SC_ESR + step / SC_CAP), demand_a))
            next_sc_v = sc_v + current_a * step / SC_CAP
            absorbed_j = current_a * ((sc_v + next_sc_v) / 2 + SC_ESR * current_a) * step / EFFICIENCY
            bus_extra_j = max(0.0, demand_w * step - absorbed_j)
            peak_bus_j = max(peak_bus_j, bus_extra_j)
            sc_v = next_sc_v
            elapsed += step
    return {
        'grade': grade, 'steady_s': descent_time, 'speed_kmh': speed * 3.6,
        'braking_distance_m': distance, 'braking_time_s': stop_time,
        'steady_gravity_w': gravity_force * speed, 'initial_braking_w': peak_w,
        'steady_motor_torque_nm_ideal_each': gravity_force * 0.03 / 8,
        'peak_motor_torque_nm_ideal_each': (MASS * deceleration + gravity_force) * 0.03 / 8,
        'stop_mechanical_j': 0.5 * MASS * speed**2 + gravity_force * distance,
        'bus_peak_v': sqrt(BUS_START**2 + 2 * peak_bus_j / BUS_CAP),
        'sc_final_rest_cell_v': sc_v / 8,
    }


def downhill_window(grade, dt=DT):
    low, high = 0.0, 30.0
    for _ in range(14):
        trial = (low + high) / 2
        if brake(5 / 3.6, 1.0, grade, trial, dt)['bus_peak_v'] <= BUS_LIMIT:
            low = trial
        else:
            high = trial
    return brake(5 / 3.6, 1.0, grade, low, dt)


def launch_gap_window(start_v, battery_a, inductance_h, dt=0.0001):
    bus_v = 22.5
    gap_w = 3600 - bus_v * battery_a
    terminal_w = gap_w / EFFICIENCY
    voltage = start_v
    elapsed = 0.0
    while voltage**2 > 4 * SC_ESR * terminal_w:
        current = 2 * terminal_w / (voltage + sqrt(voltage**2 - 4 * SC_ESR * terminal_w))
        terminal_v = voltage - current * SC_ESR
        duty = 1 - terminal_v / bus_v
        ripple = terminal_v * duty / (inductance_h * 20_000)
        if current + ripple / 2 >= 80:
            break
        voltage -= current * dt / SC_CAP
        elapsed += dt
    return {'start_cell_v': start_v / 8, 'battery_current_comparison_a': battery_a,
            'sc_output_required_w': gap_w, 'inductance_h': inductance_h,
            'time_until_80a_peak_comparison_s': elapsed, 'end_cell_rest_v': voltage / 8}


def main():
    flat = [brake(TOP_SPEED, distance) for distance in (2.0, 3.0, 5.0)]
    downhill = [downhill_window(grade) for grade in (0.2, 0.3)]
    for case in flat:
        refined = brake(TOP_SPEED, case['braking_distance_m'], dt=DT / 2)
        assert abs(case['bus_peak_v'] - refined['bus_peak_v']) < 0.005
    for case in downhill:
        refined = downhill_window(case['grade'], dt=DT / 2)
        assert abs(case['steady_s'] - refined['steady_s']) < 0.02
    supply_gap_w = 3600 - 22.5 * 130
    sc_terminal_w = supply_gap_w / EFFICIENCY
    sc_discharge_a = 2 * sc_terminal_w / (SC_START + sqrt(SC_START**2 - 4 * SC_ESR * sc_terminal_w))
    result = {
        'qualification': 'HOLD',
        'assumptions': '30kg，60mm主动轮，4:1，无损机械传动且双机均分。仅平动、坡重力与恒定减速度。电池和中央泄放记零，假设双FOC可控且全部机械制动功率进BUS，超容充放效率取95%比较。逐节均衡，每节280F/5.6mΩ，限制充电20A与每节2.2V端电压。模型不含轮地附着、转动惯量、转换器动态、温升、恢复、反拖尾流或掉电制动。',
        'licap_source': 'https://www.licaptech.com/pdfs/datasheets/ultracapacitors/60862a8f5979e5f969df83ba_LICAP_SC0350_270_RSS_Datasheet_061120.pdf',
        'bus_limit_comparison_v': BUS_LIMIT,
        'launch_power_lower_bounds': {
            'target_speed_kmh': 15, 'dual_foc_input_w': 3600, 'current_at_37p8_v_a': 3600 / 37.8,
            'current_at_22p5_v_a': 3600 / 22.5, 'candidate_battery_limit_a': 130,
            'min_sc_bus_power_at_22p5_v_w': supply_gap_w,
            'initial_sc_discharge_a_16v_esr_95pct': sc_discharge_a,
            'translation_only_energy_j': 0.5 * MASS * TOP_SPEED**2,
            'launch_duration_comparisons': [
                {'s': t, 'wheel_force_n': MASS * TOP_SPEED / t,
                 'each_motor_torque_nm_ideal': MASS * TOP_SPEED / t * 0.03 / 8,
                 'sc_bus_energy_if_gap_held_j': supply_gap_w * t}
                for t in (1, 2, 3)
            ],
            'constant_gap_scope': '理想 CCM Boost，母线固定22.5V，电感20kHz，80A为比较峰值。未计温升、控制动态、实际加速轨迹及其他负载。',
            'constant_gap_supply_windows': [
                launch_gap_window(start_v, battery_a, inductance)
                for start_v in (16.0, 17.2) for battery_a in (100, 130)
                for inductance in (12.8e-6, 20e-6, 24e-6)
            ],
        },
        'flat_braking_sc_only': flat,
        'downhill_sc_only_energy_window_with_one_meter_stop': downhill,
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
