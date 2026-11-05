"""Compare KV conventions before claiming that the unchanged bridge needs weakening."""
import json
import math
from pathlib import Path


def main():
    kv, poles = 270.0, 7
    speed, radius, gear = 15 / 3.6, 0.03, 4
    omega_m = speed * gear / radius
    # Both remain hypotheses until measured back-EMF and torque/current mapping.
    constants = {
        'legacy_assumed_phase_torque_constant_9p55_over_kv': 60 / (2 * math.pi * kv),
        'dc_bus_kv_with_sinusoidal_phase_peak_current': math.sqrt(3) / 2 * 60 / (2 * math.pi * kv),
    }
    cbus, initial_bus, delay = 0.05286, 22.5, 0.05
    source_input_budget = 0.0193
    # Both healthy FOC logic loads remain powered. Other old central rails are
    # withdrawn in this event; no claim of zero control consumption is made.
    auxiliary_power = 2 * 1.2 + initial_bus * source_input_budget
    buses = []
    for shed in (0.0, 0.02, 0.05):
        energy = 70 * shed + 5 * (delay - shed) + auxiliary_power * delay
        bus = math.sqrt(initial_bus**2 - 2 * energy / cbus)
        buses.append({'pump_70w_duration_s': shed, 'pump_then_w': 5,
                      'additional_control_and_source_budget_w': auxiliary_power,
                      'energy_draw_before_buildup_j': energy, 'bus_after_50ms_v': bus})
    rows = []
    for name, kt in constants.items():
        flux = kt / (1.5 * poles)
        # The 0.9 factor deliberately discounts credited wheel braking force.
        # It is NOT a claim that reverse gearbox efficiency divides shaft power.
        # No unmeasured gearbox friction is credited as an extra brake.
        for force_factor in (1.0, 0.9):
            acceleration = speed**2 / (2 * (3 - speed * delay))
            torque = 30 * acceleration * radius / (2 * gear * force_factor)
            iq = -torque / kt
            for inductance in (20e-6, 50e-6, 100e-6):
                vd = -poles * omega_m * inductance * iq
                vq = 0.020 * iq + poles * omega_m * flux
                voltage = math.hypot(vd, vq)
                required_bus = math.sqrt(3) * voltage / 0.95
                for bus in [initial_bus] + [b['bus_after_50ms_v'] for b in buses]:
                    rows.append({'kv_interpretation': name, 'kt_per_phase_peak_a_nm_per_a': kt,
                                 'psi_f_wb': flux, 'credited_wheel_force_fraction_assumption': force_factor,
                                 'phase_resistance_assumption_ohm': 0.020,
                                 'lq_assumption_h': inductance, 'torque_each_nm': torque,
                                 'iq_phase_peak_a': iq, 'id_a': 0.0, 'bus_v': bus,
                                 'required_phase_peak_v': voltage,
                                 'required_bus_at_95pct_modulation_v': required_bus,
                                 'id0_voltage_comparison_met': required_bus <= bus,
                                 'under_20a_exploration_line': abs(iq) <= 20,
                                 'qualification': 'UNMEASURED_COMPARISON_ONLY'})
    # Commanded Id=0 is not actual Id=0 when the angle estimate is wrong.
    # Link EMF and torque through the SAME flux; check both signs of error.
    # Increase commanded current by 1/cos(error) to retain required torque.
    kt = constants['dc_bus_kv_with_sinusoidal_phase_peak_current']
    flux = kt / (1.5 * poles)
    torque = 30 * speed**2 / (2 * (3 - speed * delay)) * radius / (2 * gear * 0.9)
    angle_rows = []
    for angle in (0, 2, 5, 10, 30):
        magnitude = torque / (kt * math.cos(math.radians(angle)))
        points = []
        for sign in (-1, 1):
            error = math.radians(sign * angle)
            actual_d = magnitude * math.sin(error)
            actual_q = -magnitude * math.cos(error)
            vd = 0.020 * actual_d - poles * omega_m * 50e-6 * actual_q
            vq = 0.020 * actual_q + poles * omega_m * (50e-6 * actual_d + flux)
            points.append({'actual_id_phase_peak_a': actual_d,
                           'actual_iq_phase_peak_a': actual_q,
                           'required_bus_v': math.sqrt(3) * math.hypot(vd, vq) / 0.95})
        worst_bus = max(p['required_bus_v'] for p in points)
        angle_rows.append({'angle_error_bound_deg': angle,
                           'commanded_id_a': 0, 'commanded_iq_phase_peak_a': -magnitude,
                           'actual_shaft_braking_torque_min_nm': torque,
                           'both_angle_error_signs': points,
                           'worst_required_bus_v': worst_bus,
                           'pump_profiles': [{'shed_s': b['pump_70w_duration_s'],
                                              'bus_v': b['bus_after_50ms_v'],
                                              'voltage_budget_met': b['bus_after_50ms_v'] >= worst_bus}
                                             for b in buses]})
    low_speed_rows = []
    for speed_kmh in (5, 1, 0.5, 0.1):
        for angle in (30, 35):
            magnitude = torque / (kt * math.cos(math.radians(angle)))
            omega_e = poles * speed_kmh / 3.6 * gear / radius
            points = []
            for sign in (-1, 1):
                actual_d = magnitude * math.sin(math.radians(sign * angle))
                actual_q = -magnitude * math.cos(math.radians(sign * angle))
                vd = 0.020 * actual_d - omega_e * 50e-6 * actual_q
                vq = 0.020 * actual_q + omega_e * (50e-6 * actual_d + flux)
                points.append(math.sqrt(3) * math.hypot(vd, vq) / 0.95)
            low_speed_rows.append({'speed_kmh': speed_kmh, 'angle_error_bound_deg': angle,
                                   'commanded_current_phase_peak_a': magnitude,
                                   'commanded_current_phase_rms_a': magnitude / math.sqrt(2),
                                   'actual_id_magnitude_bound_a': magnitude * math.sin(math.radians(angle)),
                                   'actual_iq_phase_peak_a': -torque / kt,
                                   'actual_torque_each_nm': torque,
                                   'worst_required_bus_v': max(points),
                                   'under_20a_exploration_line': magnitude <= 20,
                                   'qualification': 'LOW_SPEED_CURRENT_AND_ANGLE_BUDGET_ONLY_NO_STOP_DETECTOR'})
    result = {
        'status': 'KV_CONVENTION_COMPARISON_ONLY_HOLD',
        'selected_motor_label': 'BDUAV N5065 270KV, 7 pole pairs; label convention not measured',
        'constants': constants,
        'peak_vs_rms': 'Iq and dq voltage are sinusoidal phase peak. Phase RMS = phase peak / sqrt(2). Bus current is a separate quantity.',
        'buildup_delay_s': delay,
        'bus_capacitance_f_assumption': cbus,
        'initial_bus_v': initial_bus,
        'pump_load_sensitivity': buses,
        'rows': rows,
        'angle_error_central_point_sensitivity': angle_rows,
        'low_speed_sector_angle_sensitivity': low_speed_rows,
        'conclusions': [
            'The old 9.55/KV phase-current sensitivity is not a measured conversion of the vendor label. A failure under it is not proof that the selected motor cannot brake at 22.5 V.',
            'If KV means no-load rpm per DC-bus volt under the sinusoidal convention, phase-peak Kt is about 8.27/KV. Then the central Rs/Lq point has an Id=0 candidate at 22.5 V and after the 20 ms pump-shed comparison, without adding inverter components.',
            'Lq=100 uH, available voltage, delay, real efficiency and actual motor constants can still fail the Id=0 window. Do not claim this comparison qualifies torque, source current or a 3 m stop.',
            'Measure line-line back-EMF waveform and speed, derive phase-neutral peak EMF and flux, and independently measure signed torque versus phase-peak Iq. Do not select whichever convention produces a passing result.',
            'Keep weakening disabled until actual negative-Id/demagnetization and thermal limits exist. Prioritize correct units and existing pump/auxiliary load shedding before proposing hardware or mechanical changes.',
            'The 0.9 wheel-force factor is a conservative reduction of credited braking effectiveness, not a backdrive efficiency model. A later energy-conserving joint model must use the actual direction-dependent drivetrain losses.',
            'This is a local voltage/torque feasibility comparison. It neither executes nor replaces the final all-board vehicle model.',
            'Angle-error rows retain torque by increasing commanded current and check actual positive/negative Id. Disabling intentional weakening does not make real Id zero under angle error. Qualify angle/current/voltage together; a nominal 0-degree result is insufficient.',
        ],
        'primary_sources': [
            'https://newdocs.odriverobotics.com/v/latest/component-selection.html',
            'https://docs.simplefoc.com/voltage_torque_control',
        ],
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    center = [r for r in rows if r['kv_interpretation'].startswith('dc_bus') and r['credited_wheel_force_fraction_assumption'] == 0.9 and r['lq_assumption_h'] == 50e-6]
    print(json.dumps({'kt_phase_peak': constants['dc_bus_kv_with_sinusoidal_phase_peak_current'], 'central_id0_comparisons': center}))


if __name__ == '__main__':
    main()
