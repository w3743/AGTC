"""Low-order, parameterized two-FOC braking screen; assumptions are not measurements."""

import json
import math
from pathlib import Path

MASS_KG = 30.0                 # project baseline, not weighed
WHEEL_RADIUS_M = 0.030         # inferred from 60 mm drive wheel, pitch radius unqualified
GEAR_RATIO = 4.0               # project baseline
GEAR_EFFICIENCY = 1.0          # sensitivity assumption; loss data unavailable
KV_RPM_PER_V = 270.0           # seller/user-selected label; convention unverified
POLE_PAIRS = 7
V0_MPS = 15 / 3.6
TARGET_STOP_M = 3.0
BUS_VALUES_V = (22.5, 33.3, 37.8)
DT_S = 0.001

# These are explicit sensitivity points, not measured motor bounds or ratings.
CASES = {
    "optimistic_sensitivity": {"kt_nm_per_a": 0.050, "r_phase_ohm": 0.010, "lq_h": 20e-6, "iq_peak_a": 100.0},
    "central_sensitivity": {"kt_nm_per_a": 60 / (2 * math.pi * KV_RPM_PER_V), "r_phase_ohm": 0.020, "lq_h": 50e-6, "iq_peak_a": 60.0},
    "conservative_sensitivity": {"kt_nm_per_a": 0.025, "r_phase_ohm": 0.040, "lq_h": 100e-6, "iq_peak_a": 20.0},
}
INERTIA_FACTORS = (1.0, 1.5, 2.0)  # effective translational mass multiplier, not inferred inertia


def motor_rpm(vehicle_speed_mps):
    wheel_rpm = vehicle_speed_mps / (2 * math.pi * WHEEL_RADIUS_M) * 60
    return wheel_rpm * GEAR_RATIO


def electrical_point(speed_mps, bus_v, p):
    rpm = motor_rpm(speed_mps)
    omega_m = rpm * 2 * math.pi / 60
    omega_e = omega_m * POLE_PAIRS
    kt = p["kt_nm_per_a"]
    # Explicit convention: torque = Kt * |Iq|; q-current is treated as phase-peak
    # equivalent here. If measured Kt uses RMS/line conventions, remap before use.
    psi = kt / (1.5 * POLE_PAIRS)
    vmax_phase_peak = bus_v / math.sqrt(3)  # ideal SVPWM linear range; no deadtime/margin
    r, lq = p["r_phase_ohm"], p["lq_h"]
    back_emf = omega_e * psi
    cross_coupling = omega_e * lq
    quadratic = cross_coupling**2 + r**2
    center = back_emf * r
    discriminant = center**2 - quadratic * (back_emf**2 - vmax_phase_peak**2)
    max_i = 0.0
    if discriminant >= 0:
        half_span = math.sqrt(discriminant)
        lower = max(0.0, (center - half_span) / quadratic)
        upper = min(p["iq_peak_a"], (center + half_span) / quadratic)
        if upper >= lower:
            max_i = upper
    torque = kt * max_i
    p_mech = torque * omega_m
    p_cu = 1.5 * r * max_i * max_i
    p_bus_signed = p_mech - p_cu
    return {
        "motor_rpm": rpm,
        "electrical_hz": rpm * POLE_PAIRS / 60,
        "v_phase_peak_limit_ideal": vmax_phase_peak,
        "v_phase_peak_at_iq_limit": math.hypot(cross_coupling * max_i, back_emf - r * max_i),
        "iq_peak_available_a_model": max_i,
        "motor_brake_torque_nm_per_motor_upper_model": torque,
        "wheel_force_n_two_motors_upper_model": 2 * torque * GEAR_RATIO * GEAR_EFFICIENCY / WHEEL_RADIUS_M,
        "shaft_mechanical_power_W_extracted": p_mech,
        "three_phase_copper_loss_W_model": p_cu,
        "net_regen_power_to_bus_W_before_inverter_loss": max(0.0, p_bus_signed),
        "signed_bus_power_per_motor_W_before_inverter_loss": p_bus_signed,
        "torque_lower_qualified_Nm_per_motor": 0.0,
        "hall_torque_availability_factor_qualified": None,
    }


def stop_integral(bus_v, p, inertia_factor, step=DT_S):
    # Upper-envelope integration only. Hall availability and sink acceptance are
    # deliberately not credited; these are separately reported as zero-qualified.
    x, v, t, peak_bus_power, bus_energy = 0.0, V0_MPS, 0.0, 0.0, 0.0
    mass_eff = MASS_KG * inertia_factor
    while v > 0 and x < 100:
        ep = electrical_point(v, bus_v, p)
        force = ep["wheel_force_n_two_motors_upper_model"]
        if force <= 0:
            return {"stop_distance_upper_envelope_m": None, "stop_time_s": None,
                    "status": "NO_STOP_IN_UPPER_MODEL", "distance_so_far_m": x}
        a = force / mass_eff
        step_s = min(step, v / a)
        dv = min(v, a * step_s)
        v_next = v - dv
        ds = (v + v_next) * 0.5 * step_s
        x += ds
        t += step_s
        power = 2 * ep["signed_bus_power_per_motor_W_before_inverter_loss"]
        peak_bus_power = max(peak_bus_power, power)
        bus_energy += power * step_s
        v = v_next
    return {"stop_distance_upper_envelope_m": x, "stop_time_s": t,
            "status": "MODEL_ONLY_UNQUALIFIED", "peak_net_regen_bus_W": peak_bus_power,
            "signed_bus_energy_J_before_inverter_loss": bus_energy, "effective_mass_kg": mass_eff}


def main():
    speeds = [round(i * 0.1, 3) for i in range(43)] + [V0_MPS]
    speed_rows = []
    for name, p in CASES.items():
        for bus in BUS_VALUES_V:
            for speed in speeds:
                ep = electrical_point(speed, bus, p)
                speed_rows.append({"case": name, "bus_v": bus, "vehicle_speed_mps": speed,
                                   "vehicle_speed_kmh": speed * 3.6, **ep})
    stop_rows = []
    for name, p in CASES.items():
        for bus in BUS_VALUES_V:
            for inertia in INERTIA_FACTORS:
                stop_rows.append({"case": name, "bus_v": bus, "effective_mass_factor": inertia,
                                  **stop_integral(bus, p, inertia)})

    kt = 60 / (2 * math.pi * KV_RPM_PER_V)
    required_a = V0_MPS**2 / (2 * TARGET_STOP_M)
    required_f = MASS_KG * required_a
    required_motor_nm = required_f * WHEEL_RADIUS_M / (2 * GEAR_RATIO * GEAR_EFFICIENCY)
    tstart_s = 0.05
    omega_m_top = motor_rpm(V0_MPS) * 2 * math.pi / 60
    result = {
        "status": "HOLD_MODEL_SCREEN_ONLY",
        "model": "quasi-steady PMSM dq voltage feasibility plus longitudinal upper-envelope integration; not a measured motor, firmware, vehicle, or sink model",
        "known_project_inputs": {
            "selected_motor": "BDUAV N5065 270KV, 7 pole pairs",
            "seller_motor_image": {"voltage_recommendation_V": [24, 36], "power_marking_W": 1800,
                                   "current_marking_A": 80, "current_definition": "unknown",
                                   "Hall_supply_V": 5, "Hall_current_mA": 50, "Hall_angle_deg": 60},
            "mechanical_baseline": {"mass_kg": 30, "top_speed_kmh": 15, "gear_ratio": 4,
                                    "drive_wheel_diameter_mm": 60,
                                    "qualification": "project inputs, not weighed/measured"},
            "bus_window_V": [22.5, 37.8],
        },
        "unknowns_kept_unqualified": ["phase/line resistance and inductance definitions and variation",
                                      "KV/Ke/Kt convention and measured torque constant",
                                      "80 A measurement point, RMS/peak, continuous/pulse duration",
                                      "rotor/electrical phase calibration and Hall output type/edges",
                                      "motor, wheel, gear and track equivalent inertia; gear losses",
                                      "low-speed torque control and stopping delay",
                                      "battery/supercapacitor/dump acceptance and thermal transient capability",
                                      "tire/track traction, grade, load, actual pitch radius"],
        "assumed_parameter_cases_not_measurements": CASES,
        "assumption_notes": {"kt": "central Kt=60/(2πKV) is a sensitivity convention only; endpoint Kt values are deliberately broad assumptions",
                             "r_l": "10/20/40mΩ and 20/50/100µH are sensitivity points; project has older 17mΩ/20µH simulation assumptions but no motor measurements",
                             "iq": "20/60/100A are illustrative q-current ceilings; not derived from seller's undefined 80A or a qualified board rating",
                             "dq_voltage": "vd=ωe Lq Iq, vq=ωe ψ−R Iq, ψ=Kt/(1.5p), ideal SVPWM Vphase-peak=Vbus/√3; ignores Rs inductance saliency, deadtime, current ripple, thermal and control delays",
                             "copper": "Pcu=1.5 Rphase Iq_peak² under stated peak-current convention; inverter and iron losses omitted",
                             "hall": "no Hall-based torque credit in the lower bound; at standstill edge-derived speed is unavailable; modeled upper torque does not establish low-speed closed-loop ability",
                             "sink": "regen power is demand to the DC bus before inverter losses, not absorbed energy; bus acceptance is not modeled or qualified"},
        "3m_flat_stop_required_scale_only": {
            "speed_mps": V0_MPS, "distance_m": TARGET_STOP_M,
            "constant_deceleration_required_m_s2": required_a,
            "translation_force_required_N": required_f,
            "equal_each_side_motor_torque_required_Nm_ideal": required_motor_nm,
            "each_motor_iq_A_if_central_Kt_convention": required_motor_nm / kt,
            "translation_energy_J": 0.5 * MASS_KG * V0_MPS**2,
            "initial_total_mechanical_power_W": required_f * V0_MPS,
            "motor_rpm_at_15kmh": motor_rpm(V0_MPS),
            "motor_electrical_hz_at_15kmh": motor_rpm(V0_MPS) * POLE_PAIRS / 60,
            "extra_distance_from_50ms_constant_speed_delay_m": V0_MPS * tstart_s,
            "scope": "equal share, level ground, translation-only; no delay allowance in 3m torque benchmark; not an achievable-performance claim"},
        "joint_sim_schema": {
            "lookup_key": ["case", "bus_v", "vehicle_speed_mps"],
            "fields": ["motor_brake_torque_nm_per_motor_upper_model", "torque_lower_qualified_Nm_per_motor",
                       "wheel_force_n_two_motors_upper_model", "net_regen_power_to_bus_W_before_inverter_loss",
                       "signed_bus_power_per_motor_W_before_inverter_loss", "three_phase_copper_loss_W_model",
                       "hall_torque_availability_factor_qualified"],
            "array": "speed_torque_power_rows",
            "sink_instruction": "consumer applies its own per-board allocation/acceptance; absent proved sink the credited sustainable braking torque lower bound remains zero"},
        "stop_integrations_upper_envelope": stop_rows,
        "speed_torque_power_rows": speed_rows,
        "qualified_conclusion": "HOLD: no positive lower-bound braking torque, no qualified Hall low-speed control, no measured motor R/L/Kt, and no qualified dual-rejection sink. Upper-envelope sensitivity cases are not evidence that 3 m can be met.",
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
