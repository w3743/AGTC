"""Find the least negative d-current for the AGTC braking demand."""
from hashlib import sha256
import json
import math
from pathlib import Path

FILE = Path(__file__).resolve()
SOURCE = FILE.parents[1] / 'FOC双机低阶扭矩筛查.json'
VEHICLE_SPEED = 15 / 3.6
MASS, RADIUS, GEAR, POLE_PAIRS = 30, 0.03, 4, 7
BUS_CAP_F = 0.05286
CAP_AMP_A = 60.0
ID_COMPARISON_A = 20.0


def bus_at_torque_build(pump_shed_s):
    # Zero regeneration during the 50 ms waiting interval, BMS main open.
    energy = 70 * pump_shed_s + 5 * (0.05 - pump_shed_s)
    return math.sqrt(22.5**2 - 2 * energy / BUS_CAP_F)


def required_point(bus, modulation, ld, motor, torque_nm):
    kt, resistance = motor['kt_nm_per_a'], motor['r_phase_ohm']
    omega_m = VEHICLE_SPEED * GEAR / RADIUS
    omega_e = POLE_PAIRS * omega_m
    q = torque_nm / kt
    flux = kt / (1.5 * POLE_PAIRS)
    d_zero = omega_e * motor['lq_h'] * q
    q_zero = omega_e * flux - resistance * q
    d_slope, q_slope = resistance, omega_e * ld
    voltage_limit = modulation * bus / math.sqrt(3)
    quadratic = d_slope**2 + q_slope**2
    center = d_slope * d_zero + q_slope * q_zero
    excess = d_zero**2 + q_zero**2 - voltage_limit**2
    discriminant = center**2 - quadratic * excess
    result = {
        'bus_v': bus, 'modulation_fraction': modulation, 'ld_h': ld,
        'voltage_limit_phase_peak_v': voltage_limit,
        'voltage_required_id0_phase_peak_v': math.hypot(d_zero, q_zero),
        'requested_iq_peak_a': -q, 'qualified': False,
    }
    if excess > 0 and (discriminant < 0 or center <= 0):
        return {**result, 'required_id_peak_a': None, 'status': 'NO_DQ_SOLUTION_AT_REQUESTED_IQ'}

    d = 0 if excess <= 0 else excess / (center + math.sqrt(discriminant))
    current = math.hypot(d, q)
    copper = 1.5 * resistance * current**2
    signed_bus = torque_nm * omega_m - copper
    return {
        **result,
        'required_id_peak_a': -d, 'total_current_phase_peak_a': current,
        'voltage_after_weakening_phase_peak_v': math.hypot(d_zero - d_slope*d, q_zero - q_slope*d),
        'copper_loss_w_per_motor': copper, 'signed_bus_regen_w_per_motor_before_inverter_loss': signed_bus,
        'under_60a_20a_comparison': current <= CAP_AMP_A and d <= ID_COMPARISON_A,
        'status': 'SENSITIVITY_ONLY',
    }


def main():
    source = SOURCE.read_bytes()
    motor = json.loads(source)['assumed_parameter_cases_not_measurements']['central_sensitivity']
    torque = MASS * VEHICLE_SPEED**2 / (2 * (3 - 0.1 * VEHICLE_SPEED)) * RADIUS / (2 * GEAR)
    bus50, bus20 = bus_at_torque_build(0.05), bus_at_torque_build(0.02)
    rows = [required_point(bus, margin, ld, motor, torque)
            for bus in (22.5, bus20, bus50, 18)
            for margin in (1.0, 0.95, 0.90)
            for ld in (20e-6, 50e-6, 100e-6)]
    assert len(rows) == 36
    assert any(row.get('under_60a_20a_comparison') is False for row in rows)
    result = {
        'qualification': 'HOLD',
        'scope': '现有桥臂受限弱磁对照，15 km/h，30 kg，按100 ms建扭预算的轮端制动力折算每机扭矩。相峰值电流口径，id和iq均取负，母线调制余量、电机Ld及总电流均为筛查，不是获准值。电机磁体温度/退磁限值、实际电流环及母线暂态未知。',
        'torque_nm_required_each_model': torque, 'known_qualified_negative_id_a': None,
        'bus_after_50ms_with_70w_until_20ms_v': bus20, 'bus_after_50ms_with_70w_all_50ms_v': bus50,
        'cutoff_model_only_v': 18, 'total_current_comparison_a': CAP_AMP_A,
        'negative_id_comparison_a': ID_COMPARISON_A,
        'motor_source': SOURCE.name, 'motor_source_sha256': sha256(source).hexdigest(),
        'equations': 'Vd=R·Id−ωe·Lq·Iq, Vq=R·Iq+ωe·(Ld·Id+ψ); |V|≤modulation·Vbus/√3.',
        'rows': rows,
        'limitation': '无独立已知负Id退磁许可；参数扫描失败不排除其他电机参数或控制方式，存在数学解也不建立安全性或停车资格。',
    }
    FILE.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'{len(rows)} fixed-torque weak-field comparisons, HOLD')


if __name__ == '__main__':
    main()
