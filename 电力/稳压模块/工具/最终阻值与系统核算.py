"""本轮采用阻值和系统工作点；不替用户限定电阻功率等级。"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def main():
    mass, speed, decel = 30.0, 15 / 3.6, 3.9624
    power = mass * speed * decel
    resistance, tolerance = 0.82, 0.05
    bus_min, bus_max = 22.5, 42.0
    result = {
        "date": "2026-10-29",
        "status": "ADOPTED_RESISTANCE_AND_ANALYTICAL_WORKING_POINTS",
        "scope": "阻值、支路电流、比较工况；不是电阻热/脉冲额定选型或停车实测",
        "resistor": {
            "ref": "R_DUMP_EXT", "quantity": 1,
            "resistance_ohm": resistance, "tolerance_fraction": tolerance,
            "power_rating_W": None,
            "power_rating_reason": "用户明确只要求阻值，功率和短时承受能力由用户负责；不沿用500W、1kJ或400W限制",
            "thermal_fuse_added": False,
            "maximum_R_for_reference_initial_power_ohm": bus_min**2 / power,
            "power_at_22p5V_high_R_W": bus_min**2 / (resistance*(1+tolerance)),
            "instantaneous_current_at_42V_low_R_A": bus_max / (resistance*(1-tolerance)),
            "instantaneous_electrical_power_at_42V_low_R_W": bus_max**2 / (resistance*(1-tolerance)),
            "control": "正常由BUS反馈PWM调节；硬BUS过压可强制导通；温度不串断最终耗能支路",
        },
        "braking_comparison": {
            "assumed_mass_kg": mass, "assumed_speed_kmh": 15,
            "reference_deceleration_m_s2": decel,
            "kinetic_energy_J": mass*speed**2/2,
            "initial_power_W": power,
            "time_s": speed/decel, "distance_m": speed**2/(2*decel),
            "interpretation": "M1A1参考减速度的工程比较，非用户指定1秒停车；不包含未知旋转惯量/坡度/附着或实车制动力资格",
        },
        "battery_peak_comparison": {
            "series": 9, "parallel": 5, "cells": 45,
            "user_cell_price_CNY": 13, "additional_cells": 9,
            "additional_cell_cost_CNY": 117,
            "two_motor_peak_input_W": 3600,
            "battery_peak_comparison_A_not_a_frozen_rating": 130,
            "battery_input_at_min_bus_W": 22.5*130,
            "minimum_missing_power_at_min_bus_W": 3600-22.5*130,
            "SC_current_for_missing_power_at_17p2V_eff90_A": (3600-22.5*130)/(17.2*.9),
            "empty_SC": "允许限功率慢加速，不把BMS130A比较值写成已经达标的额定",
        },
        "passive_balance": {
            "existing_resistors_rewired": 8, "resistance_each_ohm": 120,
            "current_at_2p2V_mA": 2.2/120*1000,
            "pack_dissipation_at_2p2V_each_W": 8*2.2**2/120,
            "time_constant_hours_280_350_420F": [c*120/3600 for c in [280,350,420]],
            "meaning": "持续慢泄放、允许长停自放电；不同容量停车不保持等压，不能纠正一次几十秒初充中的容量差异，充电按最高单体终止",
        },
    }
    cases = []
    for cell_c in [280,350,420]:
        for esr in [0.0028,0.0056]:
            voc, dt, terminal_max, imax, energy = 16.0, 0.0001, 2.0, 0.0, 0.0
            duration = speed/decel
            t = 0.0
            while t < duration:
                step = min(dt,duration-t)
                p = max(0.0, mass*decel*(speed-decel*(t+step/2))) * 0.95
                r = esr*8
                current = 2*p/(voc+math.sqrt(voc*voc+4*r*p))
                terminal_max = max(terminal_max,voc/8+current*esr)
                imax = max(imax,current)
                voc += current/(cell_c/8)*step
                energy += p*step
                t += step
            cases.append({'cell_C_F':cell_c,'cell_ESR_ohm':esr,'initial_OCV_each_V':2.0,
                          'max_terminal_each_V':terminal_max,'peak_charge_A':imax,
                          'final_OCV_each_V':voc/8,'SC_input_energy_J':energy})
    result['regen_window_comparison']={'model':'equal cells; constant deceleration; 95% bus-to-SC efficiency; no unknown rotating inertia/slope; not qualification',
                                      'measured_stop_V':2.2,'cases':cases}
    assert max(x['max_terminal_each_V'] for x in cases) < 2.2
    assert result['resistor']['power_at_22p5V_high_R_W'] > power
    assert result['resistor']['instantaneous_current_at_42V_low_R_A'] < 55
    out = ROOT / '检查/最终阻值与系统核算.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({"resistance_ohm": resistance, "power_at_low_bus_W": result['resistor']['power_at_22p5V_high_R_W'], "max_branch_current_A": result['resistor']['instantaneous_current_at_42V_low_R_A'], "additional_cell_cost_CNY":117}, ensure_ascii=False))


if __name__ == '__main__':
    main()
