#!/usr/bin/env python3
"""
BMS 模块 9S5P / 130A + 复用预充主路探测 + 全隔离 CAN/LPV 闭合复算脚本。

验证范围（基于 BMS-9S4P.eprj3 官方发布源 224 器件 + 复用预充合并施工合同.json）：
1. 9S5P 4+4 主功率 MOS (HYG022N10NS1TA) 60A 连续 / 130A 1s 导通损耗、分流与 BQ76952 OCD1/SCD 门限校核。
2. 11.0kΩ TPL5111 定时窗 (8.953..10.226s) 覆盖 79.787mF 慢角预充 + 判稳 + 820Ω/3W 4.1s 主路开路探测 (MAIN_PROVEN) 与短路热边界。
3. 全车统一光耦 LTV-817S-TA1-C (C109227, C档 CTR 200..400%) 三路独立 LPV (FOC-A / FOC-B / 稳压) 驱动电流与饱和裕量校核。
4. 隔离 CAN 副边地 PACK- -> GND_CAN (零增件) 与 J_EARLY_READY_IN 接收接口校验。
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT_DIR = pathlib.Path(__file__).resolve().parent
BMS_ENET = ROOT / "电力/BMS模块/检查/主工程图面验收/主工程.enet"
MERGED_CONTRACT = ROOT / "电力/BMS模块/原理图准备/复用预充合并施工合同.json"
COREBOARD_PIN_AUDIT = ROOT / "电力/BMS模块/检查/主控头引脚核对.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_9s5p_power_chain() -> dict:
    rds_max_25c = 2.8e-3  # HYG022N10NS1TA max @ VGS=10V, ID=50A
    r_3p3 = 2.0 * (rds_max_25c / 3.0)
    r_4p4 = 2.0 * (rds_max_25c / 4.0)
    r_shunt = 0.5e-3  # HoLRS2512-6W-0.5mR-1%

    # BQ76952 OCD1=80mV (2mV/step -> code 40), SCD=100mV (code 5: 100mV)
    ocd1_v = 0.080
    scd_v = 0.100
    ocd1_i_nom = ocd1_v / r_shunt
    scd_i_nom = scd_v / r_shunt
    # 含 1% 分流容差 + 50ppm*60K + BQ 门限容差保守 10%
    ocd1_i_min = (ocd1_v * 0.90) / (r_shunt * 1.013)

    return {
        "mos_mpn": "HYG022N10NS1TA (HUAYI, C2925050)",
        "topology": "4+4 common-drain back-to-back high-side N-MOS (原3+3增Q_CHG4/Q_DSG4及独立门阻/泄放阻共+6件, 0新SKU)",
        "total_path_rds_25c_max_mOhm": {"old_3p3": round(r_3p3 * 1e3, 3), "new_4p4": round(r_4p4 * 1e3, 3)},
        "loss_60A_W": {"old_3p3": round(60**2 * r_3p3, 2), "new_4p4": round(60**2 * r_4p4, 2), "per_mos_new_2x_hot_W": round(60**2 * r_4p4 * 2 / 8, 2)},
        "loss_130A_W": {"old_3p3": round(130**2 * r_3p3, 2), "new_4p4": round(130**2 * r_4p4, 2), "per_mos_current_A": round(130 / 4, 2)},
        "shunt_130A_power_W": round(130**2 * r_shunt, 2),
        "bq_protection_candidate": {
            "OCD1_threshold_mV": 80,
            "OCD1_nom_A": ocd1_i_nom,
            "OCD1_conservative_min_A": round(ocd1_i_min, 1),
            "margin_above_130A_peak_A": round(ocd1_i_min - 130.0, 1),
            "SCD_threshold_mV": 100,
            "SCD_nom_A": scd_i_nom,
            "offboard_fuse_F1_upgrade_candidate": "Littelfuse JLLN150.VXP (150A Class T, 125VDC, 由人类按9S5P峰值与线束核定)",
        },
        "gate_charge_note": "4管典型Qg=642nC (2x工程筛查1.284µC), Ciss典型36.1nF (>BQ 20nF表征点), 首板须示波器实测关断波形与止流时长。",
    }


def verify_precharge_and_probe() -> dict:
    # 慢角预充: 37.8V, 79.787mF, R_PRE_hot=9.41965Ω, I_leak=45.736mA (6x7mA电解 + 3.736mA双FOC)
    v_pack = 37.8
    c_bus = 79.787e-3
    r_pre_max = 9.41965
    i_leak = 45.736e-3
    v_target_diff = 0.5588  # 正向上阈保证通过 0.5688V - 10mV动态
    v_inf = v_pack - i_leak * r_pre_max
    # V_diff(t) = v_pack - v_inf*(1 - exp(-t/RC)) = i_leak*r_pre + v_inf*exp(-t/RC)
    t_charge_s = -r_pre_max * c_bus * math.log((v_target_diff - i_leak * r_pre_max) / v_inf)
    t_other_s = 0.355  # 启动/判稳/请求服务预算
    t_probe_s = 4.100  # 820Ω/3W 主路开路探测时长
    t_total_slow_s = t_charge_s + t_other_s + t_probe_s
    t_timer_min_s = 8.953  # 11.0kΩ TPL5111 最短窗
    t_timer_max_s = 10.226  # 11.0kΩ TPL5111 最长窗

    # 820Ω/3W 开路探测校验: PACK+ 初始 22.1V (保守差压 -0.262V -> 需放至 +0.8709V, ΔV=1.1329V)
    # Q_PROBE VCE <= 2V, R_probe_max = 820 * 1.03 = 844.6Ω -> I_min = (22.1 - 2.0) / 844.6 = 23.798mA
    # 79.787mF * 1.1329V / 23.798mA = 3.798s < 4.1s
    r_probe_max = 820.0 * 1.03
    i_probe_min = (22.1 - 2.0) / r_probe_max
    dv_open_clear = 0.8709 - (-0.262)
    t_open_detect_s = c_bus * dv_open_clear / i_probe_min
    p_probe_max_w = (37.8**2) / (820.0 * 0.99)

    # 37.8V 全压短路 (R_PRE_min = 7.284Ω)
    p_short_w = (37.8**2) / 7.284
    e_short_kj = p_short_w * t_timer_max_s / 1e3

    return {
        "timer_resistor": "R_TIMER_SET = 11.0kΩ (替换原7.32kΩ)",
        "timer_window_s": [t_timer_min_s, t_timer_max_s],
        "slow_corner_charge_s": round(t_charge_s, 3),
        "slow_corner_total_with_4p1s_probe_s": round(t_total_slow_s, 3),
        "margin_inside_timer_min_s": round(t_timer_min_s - t_total_slow_s, 3),
        "pass_slow_corner": t_total_slow_s < t_timer_min_s,
        "probe_open_detect_worst_s": round(t_open_detect_s, 3),
        "probe_duration_s": t_probe_s,
        "pass_open_main_detect": t_open_detect_s < t_probe_s,
        "probe_resistor_max_power_W": round(p_probe_max_w, 2),
        "probe_resistor_rating_W": 3.0,
        "short_circuit_37p8v_power_W": round(p_short_w, 1),
        "short_circuit_rpre_60c_derated_W": 253.3,
        "short_circuit_energy_max_kJ": round(e_short_kj, 3),
        "pass_rpre_power_derated": p_short_w < 253.3,
    }


def verify_three_lpv_and_opto() -> dict:
    # 510Ω ±1% (0603WAF5100T5E, C23193), VOUT_5V = 4.75..5.25V, Q_DRV VDS <= 0.1V
    # LTV-817S-TA1-C (C109227, C档): VF 1.2V typ / 1.4V max @20mA (保守取1.65V低温角), CTR C档 200..400% @5mA
    if_min_ma = (4.75 - 1.65 - 0.10) / (510.0 * 1.015) * 1e3
    if_nom_ma = (5.00 - 1.20 - 0.05) / 510.0 * 1e3
    if_max_each_ma = (5.25 - 0.90) / (510.0 * 0.985) * 1e3
    total_3ch_max_ma = if_max_each_ma * 3.0
    # FOC 侧接收端: 1kΩ 偏置至 3.3V/5V -> 饱和下拉电流 <= 3.6mA (若接5V <= 5.2mA, 实为1.425mA分压负载)
    # 保守 50% 全温/老化折减后 CTR_min = 100% -> IC_sat >= if_min_ma = 5.80mA
    ctr_derated_min = 1.00
    ic_capable_min_ma = if_min_ma * ctr_derated_min
    foc_load_max_ma = 1.50
    return {
        "unified_opto_mpn": "LTV-817S-TA1-C (LITEON, C109227, 全车4颗同料替代原 UNVERIFIED_2X VO617A)",
        "led_series_resistor": "0603WAF5100T5E (UNI-ROYAL, 510Ω ±1%, C23193) ×3",
        "if_min_cold_mA": round(if_min_ma, 2),
        "if_nom_mA": round(if_nom_ma, 2),
        "total_3ch_max_mA": round(total_3ch_max_ma, 2),
        "ic_capable_at_50pct_ctr_derating_mA": round(ic_capable_min_ma, 2),
        "foc_single_board_sink_max_mA": foc_load_max_ma,
        "saturation_margin_ratio": round(ic_capable_min_ma / foc_load_max_ma, 2),
        "pass": ic_capable_min_ma > foc_load_max_ma * 2.0,
    }


def verify_free_mcu_pin() -> dict:
    c = json.loads(MERGED_CONTRACT.read_text(encoding="utf-8"))
    p2 = c["mergedPinNets"]["J_CORE_P2"]["pins"]
    assert p2.get("12") is None, f"J_CORE_P2.12 is not free: {p2.get('12')}"
    coreboard = json.loads(COREBOARD_PIN_AUDIT.read_text(encoding="utf-8"))
    gpio = next(row["mcu_gpio"] for row in coreboard["pin_map"]["P2"] if row["pin"] == 12)
    assert gpio == "PC4", f"WeAct P2.12 is {gpio}, expected PC4"
    return {
        "assigned_pin": f"J_CORE_P2.12 ({gpio}, 原为 NC)",
        "coreboard_schematic": coreboard["board"]["official_schematic"],
        "net": "EARLY_GUARD_READY_N",
        "pullup": "R_EARLY_READY_PU = 10kΩ (FH RS-03K1002FT C115324) -> CORE_3V3",
        "connector": "J_EARLY_READY_IN (HanElectricity 2541WV-2X01P C41430884): pin1=EARLY_GUARD_READY_N, pin2=B0",
        "interlock_rule": "严格不接入 U_AND2.3 异步清主锁存链；仅作为新预充/新合闸硬件/固件准入资格，运行中丢失触发撤运动许可而非瞬时硬断主路。",
    }


def verify_p0_buck(contract: dict) -> dict:
    rows = {row["ref"]: row for row in contract["drawing_components"]}
    assert rows["PSU5_EXT"]["lcsc"] == "C2901894"
    assert rows["PSU5_EXT"]["pins"]["2"] == "TAP_B9"
    assert rows["R_PSU5_EN_SW"]["pins"] == {"1": "CTRL_MODULE", "2": "PSU5_EN"}
    assert rows["C_CAN_ISOIN_1U"]["pins"] == {"1": "CAN_ISO_5V", "2": "GND_CAN"}
    assert not any(f"R_PSU_MINLOAD{i}" in rows for i in range(1, 6))
    resistor_ratio = 95.3 / 30.1
    # Independent 0.1% initial tolerances plus 25 ppm/K = 0.1875% per arm at ΔT=35 K.
    error = .001 + 25e-6 * 35
    low = 1.176 * (1 + resistor_ratio * (1 - error) / (1 + error))
    high = 1.224 * (1 + resistor_ratio * (1 + error) / (1 - error))
    nominal = 1.2 * (1 + resistor_ratio)
    ripple = nominal * (1 - nominal / 37.8) / (47e-6 * .8 * 300e3)
    peak = .401 + ripple / 2
    assert low > 4.737 and high < 5.25 and peak < 1.2
    return {
        "output_nominal_v": nominal, "output_static_v": [low, high],
        "previous_preload_output_w": 5 * 25 / 270,
        "max_comparison_ripple_pp_a": ripple, "peak_at_401ma_a": peak,
        "en_open_delay_typ_ms": 1000 * 10e-9 * 1.24 / .35e-6,
        "tapped_input_c_nominal_uf": 9.5,
        "input_cap_energy_at_37v8_j": .5 * 9.5e-6 * 37.8**2,
        "scope": "静态及理想连续导通筛查。入口电流与I²t取真实线束，反灌与掉电时序按合同核验。",
        "abnormal_boundaries": contract["p0"]["buck"]["abnormal_boundaries"],
    }


def main() -> None:
    contract_path = OUT_DIR / "BMS_9S5P施工合同.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = {
        "module": "BMS模块",
        "candidate_id": "BMS_9S5P_CLOSURE_20261123",
        "status": "BMS_SUBCIRCUIT_MODEL_ONLY",
        "source_bindings": {
            "bms_official_enet": str(BMS_ENET.relative_to(ROOT)),
            "bms_official_enet_sha256": sha256_file(BMS_ENET),
            "merged_precharge_contract": str(MERGED_CONTRACT.relative_to(ROOT)),
            "merged_precharge_contract_sha256": sha256_file(MERGED_CONTRACT),
            "p0_contract_sha256": sha256_file(contract_path),
        },
        "power_9s5p_4p4": verify_9s5p_power_chain(),
        "precharge_and_probe": verify_precharge_and_probe(),
        "three_isolated_lpv": verify_three_lpv_and_opto(),
        "early_guard_ready_in": verify_free_mcu_pin(),
        "p0_5v_and_visoin": verify_p0_buck(contract),
        "isolated_can_fix": {
            "part_delta": 1,
            "visoin_bypass": "C_CAN_ISOIN_1U (C15849) 紧贴 U_CAN.10/.12",
            "net_rename": "PACK- -> GND_CAN on CAN secondary isolated domain",
            "affected_pins": [
                "U_CAN.9 (GND2)",
                "U_CAN.12 (GNDP2)",
                "D_CAN_TVS.3 (GND)",
                "J_CAN.3 (GND)",
                "C_CAN_PWR_OUT1.2",
                "C_CAN_PWR_OUT2.2",
            ],
            "harness_pinout_warning": "BMS J_CAN (DEGSON 5.08 3P) 针序为 1=CANH, 2=CANL, 3=GND_CAN；FOC/稳压/水泵 (JST XH 3P) 为 1=GND_CAN, 2=CANH, 3=CANL，做线束时必须按信号名逐针对应，严禁直通混插。",
        },
        "bom_encoding_fix": {
            "ref": "C_VC1",
            "fix": "Manufacturer 从乱码 'FH(华)' 修正为 'FH(风华)' (MPN 0603B224K500NT / C83626 不变)",
        },
    }
    joint_path = ROOT / "电力/检查/联合仿真/降本后典型故障联合仿真.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    report["localization_v1"] = {
        "joint_evidence": str(joint_path.relative_to(ROOT)),
        "precharge_after": joint["comparison"]["after"]["2_precharge"],
        "main_path_after": joint["comparison"]["after"]["5_protection"],
        "tap_fuse_after": joint["5_tap_fuse"],
        "baseline_note": "原字段precharge_and_probe为降本前基线；4+4主功率保留。",
    }
    (OUT_DIR / "BMS_9S5P闭合复算.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
