"""Parametric, rerunnable low-voltage launch / battery-only power screen.

Ideal constant-acceleration translation model; not a measured vehicle or motor
map. Distinguishes theoretical 9S voltage, BMS hardware configuration, firmware
gate, target pack, and supercapacitor charge/discharge permission.
Run: python 电力/检查/联合仿真/低压起步接管.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "低压起步接管.json"
V_MIN, V_MAX = 22.5, 37.8
TARGET_KMH = 15.0
R_WHEEL_M, GEAR, MOTORS = 0.030, 4.0, 2
ETA_SCREEN = 0.90  # comparison input only; no measured drivetrain efficiency
MASSES_KG = (15.0, 30.0, 45.0)
DURATIONS_S = (0.5, 1.0, 2.0, 3.0)
BATTERY_CURRENT_CASES_A = (60.0, 100.0, 130.0)

EVIDENCE = {
    "foc_firmware": "固件/foc_driver/BusinessLayer/Inc/bl_motor.h",
    "bms_native_design": "电力/BMS模块/设计细节.md",
    "bms_native_project": "电力/BMS模块/eda工程文件/BMS-9S4P/BMS-9S4P.eprj3",
    "bms_native_bom": "电力/BMS模块/原理图准备/BOM.csv",
    "bms_target_requirements": "电力/BMS模块/设计需求.md",
    "battery_cell_spec": "电力/参考资料/电池/11-_EVE_INR18650-30P_SPEC._PBRI-INR1865030P-D06-01-B-2023.8.4.pdf",
    "motor_requirements": "电力/FOC驱动模块/设计需求.md",
    "motor_detail": "电力/FOC驱动模块/设计细节.md",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory(mass: float, duration: float) -> dict:
    vf = TARGET_KMH / 3.6
    a = vf / duration
    force = mass * a
    torque = force * R_WHEEL_M / (MOTORS * GEAR)
    mech_peak = force * vf
    input_peak = mech_peak / ETA_SCREEN
    points = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = duration * fraction
        speed = vf * fraction
        p_mech = force * speed
        p_electrical = p_mech / ETA_SCREEN
        points.append({
            "time_s": round(t, 6), "speed_kmh": round(speed * 3.6, 6),
            "mechanical_power_w": round(p_mech, 6),
            "input_power_w_at_90pct_screen": round(p_electrical, 6),
            "battery_current_a_at_22p5v_screen": round(p_electrical / V_MIN, 6),
        })
    return {
        "mass_kg": mass, "duration_s": duration,
        "constant_acceleration_m_s2": a,
        "wheel_force_n": force,
        "ideal_each_motor_shaft_torque_nm": torque,
        "friction_coefficient_minimum_flat_no_losses": a / 9.80665,
        "translation_energy_j": 0.5 * mass * vf * vf,
        "mechanical_peak_power_w": mech_peak,
        "input_peak_power_w_at_90pct_screen": input_peak,
        "peak_battery_current_a_at_22p5v_screen": input_peak / V_MIN,
        "samples": points,
    }


def main() -> None:
    profiles = [trajectory(m, t) for m in MASSES_KG for t in DURATIONS_S]
    power_cases = []
    for current in BATTERY_CURRENT_CASES_A:
        power = V_MIN * current
        power_cases.append({
            "pack_current_a": current,
            "bus_power_at_22p5v_w": power,
            "ideal_cell_current_a": {"4P": current / 4, "5P": current / 5},
            "constant_3p6_kw_shortfall_w": max(0.0, 3600.0 - power),
            "status": {
                60.0: "continuous_design_input_only_not_qualified",
                100.0: "native_main_fuse_label_and_legacy_comparison_only_not_discharge_rating",
                130.0: "9S5P_engineering_candidate_not_implemented_or_qualified",
            }[current],
            "max_mechanical_traction_power_w_at_90pct_screen": power * ETA_SCREEN,
        })
    result = {
        "schema": "agtc-low-voltage-launch-takeover-v1",
        "status": "MODEL_SCREEN_ONLY_HOLD",
        "model": "flat ground, ideal constant acceleration, translation only; no rolling/aero/rotational inertia, bus sag, motor map, torque/current control, thermal model, or uncertainty margin",
        "assumptions": {
            "target_speed_kmh": TARGET_KMH, "wheel_radius_m": R_WHEEL_M,
            "gear_ratio": GEAR, "motor_count": MOTORS,
            "drivetrain_efficiency_screen": ETA_SCREEN,
            "efficiency_is_measured": False,
            "initial_speed_kmh": 0.0,
            "power_trajectory": "P_mech(t)=m*(v_final/T)^2*t; P_electrical=P_mech/eta",
        },
        "configuration_layers": {
            "theoretical_9s_hardware_voltage": {
                "range_v": [V_MIN, V_MAX],
                "meaning": "pack electrochemical/configuration range only; does not prove load voltage, available current, or FOC compatibility",
            },
            "native_bms_hardware": {
                "series_parallel": "9S4P", "main_mosfet_banks": "3+3",
                "main_fuse_label_a": 100,
                "discharge_current_rating_a": None,
                "evidence_paths": [EVIDENCE["bms_native_project"], EVIDENCE["bms_native_design"], EVIDENCE["bms_native_bom"]],
                "status": "existing source is 9S4P; 100A fuse is not a BMS discharge rating; 130A not established",
            },
            "current_foc_software": {
                "bus_min_v": 30.0, "bus_max_v": 58.0,
                "source_path": EVIDENCE["foc_firmware"],
                "status": "22.5V is below configured software bus minimum; low-voltage launch unavailable in current firmware as written",
                "additional_blocker": "source comments state no speed loop, torque command, or reliable self-start strategy; voltage-mode ramp is not a battery power limiter",
            },
            "user_target_battery": {
                "series_parallel": "9S5P", "cells": 45,
                "peak_current_candidate_a": 130,
                "candidate_is_implemented": False,
                "status": "design target/candidate only; not current hardware or qualified rating",
                "source_paths": [EVIDENCE["bms_target_requirements"], EVIDENCE["bms_native_design"]],
            },
        },
        "supercapacitor_state_semantics": {
            "regen_charge_rejected_or_charge_limit_zero": "means no credit for accepting regenerative current; does not itself prohibit discharging the capacitor through the bidirectional converter",
            "startup_discharge": "separate permission and limit, requiring valid cell voltages, converter direction/state, temperature, and local current limit",
            "zero_or_unavailable_supercap": "battery-only power-limited launch is the intended fallback, but current FOC 30V minimum and missing closed-loop limits prevent claiming it works at 22.5V today",
        },
        "battery_power_comparison_at_22p5v": power_cases,
        "cell_current_screen": "EVE 30P specifies 30 A maximum discharge in the cited cell datasheet. Ideal sharing is only an arithmetic screen. Pack temperature, imbalance, interconnect and protection limits remain open.",
        "constant_acceleration_profiles": profiles,
        "blocking_inputs": [
            "measured loaded vehicle mass and effective wheel radius",
            "motor torque-speed and phase/bus-current maps including reliable zero-speed start",
            "measured BMS/battery low-SOC cold/hot V-I limit and bus sag",
            "qualified current limit, duration, sharing, fuse/wiring/connector and thermal envelope",
            "FOC undervoltage threshold requalification plus closed-loop torque/power ramp and fault behavior",
            "auxiliary startup load and supercapacitor discharge-direction limits",
        ],
        "source_hashes": [{"path": path, "sha256": sha(REPO / path)} for path in EVIDENCE.values()],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"profiles={len(profiles)}; source_hashes={len(result['source_hashes'])}")


if __name__ == "__main__":
    main()
