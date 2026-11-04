"""Conditional dual-FOC braking and sink handover envelope for the connected bus."""
import hashlib
import json
import math
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOC_ENET = ROOT / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
BUS_C_LOW_F = 0.048 + 0.00486
LOCAL_BOV_LOW_V = 39.209
V_START_MPS = 15 / 3.6
MASS_KG = 30.0
DECEL_MPS2 = 3.9624


def braking_case(slope_deg):
    duration = V_START_MPS / DECEL_MPS2
    force = MASS_KG * (DECEL_MPS2 + 9.80665 * math.sin(math.radians(slope_deg)))
    power = force * V_START_MPS
    return {"slope_deg": slope_deg, "t_s": duration, "distance_m": V_START_MPS * duration / 2,
            "P_mech_initial_W": power, "E_mech_J": power * duration / 2}


def remaining_sink_demand(initial_w, duration_s, retained_w):
    """Area of the triangular residual after a constant qualified sink."""
    peak = max(0.0, initial_w - retained_w)
    return peak, 0.5 * duration_s * peak**2 / initial_w


def window(arm_v, guard_v, residual_w):
    limit = LOCAL_BOV_LOW_V - guard_v
    headroom = 0.5 * BUS_C_LOW_F * (limit**2 - arm_v**2)
    return {"actual_arm_V": arm_v, "reserve_V": guard_v, "comparison_limit_V": limit,
            "residual_W": residual_w, "headroom_J": headroom,
            "max_latency_ms_zero_other_reservations": 1000 * headroom / residual_w,
            "in_flight_J_by_delay_ms": {str(ms): residual_w * ms / 1000 for ms in (1, 2, 5, 20, 50)}}


def main():
    brakes = [braking_case(angle) for angle in (0, 5, 10, 20, 30)]
    cases = []
    for brake, (scenario, retained_w) in product(brakes, (
            ("battery_refuses_SC_unqualified", 0), ("battery_refuses_SC_qualified_300W_example", 300),
            ("SC_refuses_BMS_current_allocation", 0), ("SC_refuses_valid_CCL_16A_at_22p5V_example", 360),
            ("SC_refuses_valid_CCL_16A_at_37p8V_example", 604.8), ("both_refuse", 0))):
        peak, energy = remaining_sink_demand(brake["P_mech_initial_W"], brake["t_s"], retained_w)
        cases.append({"scenario": scenario, "slope_deg": brake["slope_deg"], "retained_sink_W": retained_w,
                      "central_peak_min_W": peak, "central_stop_energy_min_J": energy,
                      "mean_thermal_spacing_s_by_qualified_average_W": {
                          str(power): energy / power for power in (100, 200, 300)},
                      "ideal_peak_duty_at_38V_Rmax": peak / (38**2 / 0.861)})
    report = {
        "status": "CONDITIONAL_CONNECTED_BUS_ENVELOPE",
        "FOC_ENET_SHA256": hashlib.sha256(FOC_ENET.read_bytes()).hexdigest(),
        "mass_kg": MASS_KG, "speed_km_h": V_START_MPS * 3.6, "deceleration_m_s2": DECEL_MPS2,
        "mechanical_cases": brakes, "central_cases": cases,
        "handover_windows": [window(arm, guard, power) for arm, guard, power in
                             product((37.8, 38.0, 38.5), (0, 0.2), (250, 495.3, 707.9, 914.6, 1200))],
        "C_connected_low_initial_tolerance_F": BUS_C_LOW_F,
        "central_full_on_at_38V_ideal": {"min_W": 38**2/0.861, "max_W": 38**2/0.779,
                                        "min_A": 38/0.861, "max_A": 38/0.779},
        "conditions": [
            "The connected bus lower initial-tolerance comparison counts two local electrolytic groups and regulator electrolytics; it excludes capacitor aging, distributed ESL/ESR, MPPT, wiring and small bypass caps.",
            "Mechanical cases assume 100% net conversion of translation and slope terms and omit rotation, external push, traction and motor/drive power limits. The envelope is for two motors in total.",
            "Retained sink examples are constant and available immediately. Actual battery allocation is currently zero. A qualified current contract must subtract already committed current and transition reserves. SC-side 20 A initial-charge comparator is not a BUS absorption rating.",
            "Latency allowance is for net residual energy. Subtract retained inductor/winding energy, auxiliary sources, ripple/voltage error and other known delayed contributions before dividing by net power. Avoid counting the same transient twice.",
            "Central peak duty is a static resistive comparison. Spacing addresses mean heat only. A real pulse envelope, starting temperature, PWM and SOA, hot resistance, gate and line current buildup, and proven initial/steady sink are required.",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
