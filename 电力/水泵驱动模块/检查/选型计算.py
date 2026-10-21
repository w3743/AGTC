"""DRV8262 水泵驱动初步选型计算；运行后在本目录生成选型计算.json。

此文件仅复现算术估算，不是仿真、器件实测或硬件验收结果。
"""

from __future__ import annotations

import json
from pathlib import Path


# 设计输入 / 估算假设
VLOGIC_3V3_V = 3.3  # WeAct 主控板 3V3 电源；不是 DRV8262 DVDD
VREF_DIV_TOP_OHM = 10_000.0
VREF_DIV_BOTTOM_OHM = 10_000.0
VREF_V = VLOGIC_3V3_V * VREF_DIV_BOTTOM_OHM / (
    VREF_DIV_TOP_OHM + VREF_DIV_BOTTOM_OHM
)
RIPROPI_OHM = 1_500.0
AIPROPI_A_PER_A = 212e-6  # TI datasheet typical current-mirror gain
IMOTOR_A = 3.0            # RMS for thermal estimate; DC level for IPROPI example
VMOTOR_RATED_V = 21.0
VBUS_MIN_V = 22.5
VBUS_MAX_V = 37.8

# 9S 母线采样：两个 110 kΩ 串联作上臂，10 kΩ 下臂。
R_DIV_TOP_EACH_OHM = 110_000.0
R_DIV_BOTTOM_OHM = 10_000.0

# TI single-H-bridge hot typical RDS(on) is 43 mΩ HS and 40 mΩ LS at 150°C.
# For a compact, slightly conservative estimate, use 43 mΩ for both FETs.
R_FET_HOT_EST_OHM = 0.043
R_FET_ROOM_OHM = 0.025
VM_V = VBUS_MAX_V
PWM_HZ = 20_000.0
VD_BODY_DIODE_V = 1.0
T_DEAD_S = 300e-9
T_RF_S = 110e-9
IVM_EXTERNAL_TYP_A = 5e-3
RTH_JA_DDW_C_PER_W = 22.2

SOURCES = [
    {
        "title": "TI DRV8262 Rev.C data sheet",
        "url": "https://www.ti.com/lit/ds/symlink/drv8262.pdf",
        "used_for": [
            "single-H-bridge AIPROPI=212 µA/A and current-regulation equation",
            "single-H-bridge truth table and input defaults",
            "RDS(on), tD=300 ns, tRF=110 ns, IVM, RθJA and recommended capacitors",
        ],
    },
    {
        "title": "TI E2E: DRV8262 current regulation in single H-bridge mode",
        "url": "https://e2e.ti.com/support/motor-drivers-group/motor-drivers/f/motor-drivers-forum/1336342/drv8262-current-regulation-in-single-h-bridge-mode",
        "used_for": [
            "clarification that tying both IPROPI outputs sums their currents; using two equal resistors makes the effective resistance half",
        ],
    },
]


def calc() -> dict:
    ipropi_v = IMOTOR_A * AIPROPI_A_PER_A * RIPROPI_OHM
    itrip_a = VREF_V / (AIPROPI_A_PER_A * RIPROPI_OHM)
    divider_ratio = R_DIV_BOTTOM_OHM / (
        2 * R_DIV_TOP_EACH_OHM + R_DIV_BOTTOM_OHM
    )

    def duty_for_bus(vbus: float) -> float:
        return VMOTOR_RATED_V / vbus

    # Single bridge: one conducting high-side + one low-side path. The 43 mΩ
    # hot estimate is deliberately used for both FETs; this is not a worst case.
    p_cond_hot_w = IMOTOR_A**2 * (2 * R_FET_HOT_EST_OHM)
    p_dead_w = 2 * VD_BODY_DIODE_V * IMOTOR_A * T_DEAD_S * PWM_HZ
    p_switch_w = VM_V * IMOTOR_A * T_RF_S * PWM_HZ
    p_quiescent_typ_w = VM_V * IVM_EXTERNAL_TYP_A
    p_total_est_w = p_cond_hot_w + p_dead_w + p_switch_w + p_quiescent_typ_w
    p_cond_room_w = IMOTOR_A**2 * (2 * R_FET_ROOM_OHM)
    p_total_room_r_est_w = (
        p_cond_room_w + p_dead_w + p_switch_w + p_quiescent_typ_w
    )

    return {
        "status": "初步选型估算；不是仿真、器件实测或正式验收",
        "device_candidate": "TI DRV8262DDWR",
        "inputs": {
            "motor_rated_voltage_v": VMOTOR_RATED_V,
            "bus_voltage_min_v": VBUS_MIN_V,
            "bus_voltage_max_v": VBUS_MAX_V,
            "motor_current_for_ipropi_example_a": IMOTOR_A,
            "motor_rms_current_for_thermal_estimate_a": IMOTOR_A,
            "vref_v": VREF_V,
            "vref_source": "WeAct 主控板 3.3 V 经 10 kΩ/10 kΩ 分压得到；不取 DRV8262 DVDD",
            "vref_divider_input_v": VLOGIC_3V3_V,
            "vref_divider_top_ohm": VREF_DIV_TOP_OHM,
            "vref_divider_bottom_ohm": VREF_DIV_BOTTOM_OHM,
            "r_ipropi_ohm_single_resistor": RIPROPI_OHM,
            "a_ipropi_a_per_a_typical": AIPROPI_A_PER_A,
            "single_bridge_ipropi_connection": "IPROPI1 与 IPROPI2 相连；连接点仅接一个 RIPROPI 至 GND",
            "bus_divider_top_ohm": [R_DIV_TOP_EACH_OHM, R_DIV_TOP_EACH_OHM],
            "bus_divider_bottom_ohm": R_DIV_BOTTOM_OHM,
            "thermal_assumptions": {
                "bus_voltage_v": VM_V,
                "pwm_frequency_hz": PWM_HZ,
                "hot_rds_each_fet_ohm_estimate": R_FET_HOT_EST_OHM,
                "room_rds_each_fet_ohm_typical": R_FET_ROOM_OHM,
                "body_diode_voltage_v": VD_BODY_DIODE_V,
                "dead_time_s": T_DEAD_S,
                "output_rise_fall_time_s": T_RF_S,
                "external_vcc_vm_quiescent_current_typical_a": IVM_EXTERNAL_TYP_A,
                "ddw_jeDEC_board_rtheta_ja_c_per_w": RTH_JA_DDW_C_PER_W,
                "ambient_temperature_c_project_limit": 60.0,
                "duty_for_thermal_estimate": "conduction model uses motor RMS current; switching estimate uses TI PSW=VM*I*trf*f formula",
            },
        },
        "results": {
            "ipropi_voltage_at_3a_v": ipropi_v,
            "nominal_itrip_a": itrip_a,
            "current_mirror_gain_total_a_per_a": AIPROPI_A_PER_A,
            "nominal_duty_for_21v_motor": {
                "at_21v_bus": duty_for_bus(21.0),
                "at_22_5v_bus": duty_for_bus(VBUS_MIN_V),
                "at_37_8v_bus": duty_for_bus(VBUS_MAX_V),
            },
            "bus_divider_ratio_vout_over_vin": divider_ratio,
            "bus_divider_sensed_voltage_v": {
                "at_22_5v_bus": VBUS_MIN_V * divider_ratio,
                "at_37_8v_bus": VBUS_MAX_V * divider_ratio,
            },
            "thermal_estimate_20khz_3arms_at_37_8v": {
                "conduction_hot_fet_resistance_estimate_w": p_cond_hot_w,
                "dead_time_w": p_dead_w,
                "switching_w": p_switch_w,
                "quiescent_external_vcc_typical_w": p_quiescent_typ_w,
                "total_hot_r_plus_typical_other_terms_w": p_total_est_w,
                "junction_rise_using_JEDEC_rtheta_ja_c_estimate": p_total_est_w * RTH_JA_DDW_C_PER_W,
                "junction_temperature_at_25c_ambient_estimate": 25.0 + p_total_est_w * RTH_JA_DDW_C_PER_W,
                "junction_temperature_at_60c_ambient_estimate": 60.0 + p_total_est_w * RTH_JA_DDW_C_PER_W,
                "total_room_r_typical_other_terms_w": p_total_room_r_est_w,
                "interpretation": "约1.25 W 为热态典型导通电阻估算值与典型开关/静态损耗相加；不是最坏值。RθJA仅对应 TI 指定 JEDEC 板条件，实际值依 PCB 铜面积、层数和热过孔变化。",
            },
        },
        "formula_notes": [
            "ITRIP = VREF / (RIPROPI × AIPROPI); single mode ties IPROPI pins and uses one resistor, so gain remains 212 µA/A against total motor current.",
            "VREF = 3.3 V × 10 kΩ / (10 kΩ + 10 kΩ) = 1.65 V; source is the WeAct board 3V3 rail, not DRV8262 DVDD.",
            "VIPROPI = I × AIPROPI × RIPROPI.",
            "Dideal = 21 V / VBUS; use as an ideal duty ceiling estimate, then validate using motor-terminal voltage and actual bus voltage.",
            "VADC/VBUS = 10k / (110k + 110k + 10k) = 1/23.",
            "Pcond ≈ Irms² × (RHS + RLS); for this estimate both hot FETs use 43 mΩ.",
            "Pdead ≈ 2 × VD × I × tD × fPWM; Pswitch ≈ VM × I × tRF × fPWM; Pq ≈ VM × IVM.",
            "The 5.19 A trip point is nominal only. Mirror gain accuracy, reference and resistor tolerances mean it is not a guaranteed 5 A hard ceiling; it does not current-limit commanded high-side braking.",
        ],
        "sources": SOURCES,
    }


if __name__ == "__main__":
    result = calc()
    output_path = Path(__file__).with_name("选型计算.json")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["results"], ensure_ascii=False, indent=2))
    print(f"已写入：{output_path}")
