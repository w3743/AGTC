"""Tail-braking sensitivity using the existing low-side bridge state."""

import json
import math
from pathlib import Path

from FOC双机低阶扭矩筛查 import CASES, GEAR_RATIO, MASS_KG, POLE_PAIRS, WHEEL_RADIUS_M


OUT = Path(__file__).with_suffix('.json')
RESOLUTION_KMH = 0.01
CURRENT_SCREEN_A = 20.0


def short_point(speed, motor, bridge_resistance):
    """Quasi-steady, symmetric PMSM model under a balanced three-phase short."""
    omega_m = GEAR_RATIO * speed / WHEEL_RADIUS_M
    omega_e = POLE_PAIRS * omega_m
    resistance = motor['r_phase_ohm'] + bridge_resistance
    reactance = omega_e * motor['lq_h']
    flux = motor['kt_nm_per_a'] / (1.5 * POLE_PAIRS)
    iq = omega_e * flux * resistance / (resistance**2 + reactance**2)
    id_current = reactance * iq / resistance
    return math.hypot(iq, id_current), motor['kt_nm_per_a'] * iq


def tail_case(name, motor, bridge_resistance, initial_kmh, mass_factor, initial_iq_a):
    speed = initial_kmh / 3.6
    target = RESOLUTION_KMH / 3.6
    effective_mass = MASS_KG * mass_factor
    elapsed, distance, peak_current = 0.0, 0.0, abs(initial_iq_a)
    id_a, iq_a = 0.0, initial_iq_a
    r = motor['r_phase_ohm'] + bridge_resistance
    l = motor['lq_h']
    flux = motor['kt_nm_per_a'] / (1.5 * POLE_PAIRS)
    initial_energy = 0.5 * effective_mass * speed**2
    initial_current, initial_torque = short_point(speed, motor, bridge_resistance)
    while speed > target and elapsed < 2:
        omega_e = POLE_PAIRS * GEAR_RATIO * speed / WHEEL_RADIUS_M
        torque = -motor['kt_nm_per_a'] * iq_a
        force = 2 * torque * GEAR_RATIO / WHEEL_RADIUS_M
        decel = force / effective_mass
        step = min(0.000005, (speed - target) / decel) if decel > 0 else 0.000005
        id_a, iq_a = (id_a + step * (-r * id_a + omega_e*l*iq_a) / l,
                      iq_a + step * (-r * iq_a - omega_e*(l*id_a + flux)) / l)
        next_speed = max(target, speed - decel * step)
        distance += 0.5 * (speed + next_speed) * step
        elapsed += step
        peak_current = max(math.hypot(id_a, iq_a), peak_current)
        speed = next_speed
        if step < 1e-9:
            break
    return {
        'motor_comparison': name, 'bridge_series_ohm_per_phase': bridge_resistance,
        'initial_speed_kmh': initial_kmh, 'effective_mass_factor': mass_factor,
        'initial_kinetic_energy_j': initial_energy, 'starting_iq_a_phase_peak_model': initial_iq_a,
        'short_steady_initial_speed_current_a_phase_peak_model': initial_current,
        'short_steady_initial_speed_torque_nm_per_motor': initial_torque,
        'transient_peak_current_a_phase_peak_model': peak_current,
        'above_20a_comparison': peak_current > CURRENT_SCREEN_A,
        'distance_to_resolution_m': distance, 'time_to_resolution_s': elapsed,
        'reached_resolution_model': speed <= target + 1e-8,
    }


def main():
    hall_spacing_m = 2 * math.pi * WHEEL_RADIUS_M / (6 * POLE_PAIRS * GEAR_RATIO)
    rows = [tail_case(name, motor, bridge_r, initial, mass, iq)
            for name, motor in CASES.items()
            for bridge_r in (0.0, 0.008)
            for initial in (0.1, 0.3, 0.5)
            for mass in (1.0, 1.5)
            for iq in (0.0, -10.69)]
    assert len(rows) == 72 and any(row['above_20a_comparison'] for row in rows)
    result = {
        'qualification': 'HOLD',
        'scope': '双机平地三低侧短接下的对称电机瞬态和纵向耦合，包含两个切换初始电流。电机参数、MOS 串联电阻、等效质量为敏感性。未计 PWM/互锁、实际绕组和 MOS 热、齿隙、履带滑动或地面阻力。',
        'hall': {'project_geometry_only': True, 'spacing_m': hall_spacing_m,
                 'seconds_per_edge_at_0p5_kmh': hall_spacing_m / (0.5 / 3.6),
                 'seconds_per_edge_at_0p1_kmh': hall_spacing_m / (0.1 / 3.6),
                 'ideal_minimum_q_projection_with_30deg_error': math.cos(math.pi / 6)},
        'resolution_kmh_model_only': RESOLUTION_KMH,
        'current_a_comparison_only': CURRENT_SCREEN_A,
        'qualified_tail_torque_nm': 0.0,
        'verdict': '低侧短接力矩随速度趋零。0.01 km/h 只是积分报告分辨率，真实停稳由独立地面位移判定。切换速度必须由相电流瞬态和实际减速共同限制。',
        'cases': rows,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(rows)} low-speed short comparisons, HOLD')


if __name__ == '__main__':
    main()
