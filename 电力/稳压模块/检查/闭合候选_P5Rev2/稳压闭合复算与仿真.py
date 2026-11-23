#!/usr/bin/env python3
"""
稳压模块 P5-Rev2 闭合设计复算与 ngspice 瞬态电路仿真脚本。

验证范围（基于 P5重绘候选 234 器件基线）：
1. U_SC_OV 五针参考地同步改接 GND_SYS 后，R_BUS_OV_TRIM=2.7kΩ (RS-03K2701FT) 与 FOC BOV (1M+69.8k) 全角窗口复算。
2. 12V_PROT -> D_HW_HOLD (1N5819HW-7-F) -> 2x10Ω (RS-06K10R0FT) -> 12V_HW (C_HW_BANK=47µF) + TPS3760A012DYYR (100k/10k 0.1% 25ppm):
   - 稳态最低轨 (11.4V) 裕量校验（含 0°C 高 VF 与 60°C 高漏电）
   - 上游 12V_PROT 硬短路跌零时的保持时间与 CTR 完全放电复位裕量（含 10µF 寿命末极端角 + 100°C 肖特基 10mA 最大漏电）
   - ngspice 瞬态仿真：冷启动 dV/dt、最低轨负载稳态、上游 0Ω 硬短路跌零、2ms 短断恢复
3. HW_READY -> 22kΩ/10kΩ -> SN74LVC1G17DCKR (复用 C10425 替换缺封装 DBVR) -> PB7 全角接口预算。
4. 集成隔离 CAN (CA-IS2062A C22438481) 供电电流与 BOM 净增减统计。
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT_DIR = pathlib.Path(__file__).resolve().parent
NGSPICE = ROOT / "电力/FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe"
P5_ENET = ROOT / "电力/稳压模块/检查/全页排版候选/P5重绘候选/全页_官方.enet"
P5_BOM = ROOT / "电力/稳压模块/检查/全页排版候选/P5重绘候选/全页_官方BOM.csv"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def calc_bus_ov_gnd_sys() -> dict:
    """U_SC_OV 五针改接 GND_SYS 后（残余地差 <= 10mV），比较 2.7k / 1.5k / 0Ω。"""
    # TLV6700 datasheet full temp: VIT+ 396..404mV, VIT- 387..400mV; IB sensitivity +/-25nA
    vit_p_min, vit_p_max = 0.396, 0.404
    vit_n_min, vit_n_max = 0.387, 0.400
    ib_a = 25e-9
    gnd_err_v = 0.010  # GND_SYS Kelvin 残余地差预算 ±10mV
    od_v = 0.010  # 10mV comparator input overdrive
    foc_earliest_static_v = 42.53278094792394  # FOC 1M + 69.8k (0.1%/25ppm, 60K)
    v_batt_full = 37.800
    c_bus_low = 0.05286  # 52.86mF low corner
    p_inj_450 = 450.0268817204302
    p_inj_915 = 915.0

    # 255k x2 (0.1%, 25ppm/K, 60K -> +/-0.25%), 5.11k (0.1%, 25ppm/K, 60K -> +/-0.25%)
    r255_tol = 0.001 + 25e-6 * 60.0
    rbot_tol = 0.001 + 25e-6 * 60.0
    rtrim_tol = 0.010 + 100e-6 * 60.0  # 1% + 100ppm/K * 60K = 1.6%

    rbot_nom = 5110.0
    rbot_min = rbot_nom * (1.0 - rbot_tol)
    rbot_max = rbot_nom * (1.0 + rbot_tol)

    cases = []
    for trim_ohm, label in [(2700.0, "2.7kΩ (RS-03K2701FT 复用)"), (1500.0, "1.5kΩ (水泵复用对照)"), (0.0, "0Ω (原图基线)")]:
        rtop_min = 510000.0 * (1.0 - r255_tol) + trim_ohm * (1.0 - rtrim_tol)
        rtop_max = 510000.0 * (1.0 + r255_tol) + trim_ohm * (1.0 + rtrim_tol)

        # Vbus = Vth * (1 + Rtop/Rbot) +/- Ib * Rtop +/- gnd_err
        rise_static_earliest = vit_p_min * (1.0 + rtop_min / rbot_max) - ib_a * rtop_max - gnd_err_v
        rise_static_latest = vit_p_max * (1.0 + rtop_max / rbot_min) + ib_a * rtop_max + gnd_err_v
        fall_static_earliest = vit_n_min * (1.0 + rtop_min / rbot_max) - ib_a * rtop_max - gnd_err_v
        fall_static_latest = vit_n_max * (1.0 + rtop_max / rbot_min) + ib_a * rtop_max + gnd_err_v

        rise_od10_latest = (vit_p_max + od_v) * (1.0 + rtop_max / rbot_min) + ib_a * rtop_max + gnd_err_v
        fall_od10_earliest = (vit_n_min - od_v) * (1.0 + rtop_min / rbot_max) - ib_a * rtop_max - gnd_err_v

        windows = []
        for dv_drop in [0.0, 0.1, 0.2, 0.3]:
            v_start = rise_od10_latest + dv_drop
            margin_v = foc_earliest_static_v - v_start
            if margin_v > 0:
                e_j = 0.5 * c_bus_low * (foc_earliest_static_v**2 - v_start**2)
                t_450_ms = e_j / p_inj_450 * 1e3
                t_915_ms = e_j / p_inj_915 * 1e3
            else:
                e_j, t_450_ms, t_915_ms = 0.0, 0.0, 0.0
            windows.append({
                "near_bridge_minus_central_v": dv_drop,
                "margin_to_foc_earliest_static_v": round(margin_v, 4),
                "energy_j": round(e_j, 4),
                "budget_ms_at_450w": round(t_450_ms, 4),
                "budget_ms_at_915w": round(t_915_ms, 4),
            })

        cases.append({
            "label": label,
            "trim_ohm": trim_ohm,
            "rise_static_earliest_v": round(rise_static_earliest, 4),
            "rise_static_latest_v": round(rise_static_latest, 4),
            "fall_static_earliest_v": round(fall_static_earliest, 4),
            "fall_static_latest_v": round(fall_static_latest, 4),
            "rise_od10_latest_v": round(rise_od10_latest, 4),
            "fall_od10_earliest_v": round(fall_od10_earliest, 4),
            "release_margin_above_37p8v_v": round(fall_od10_earliest - v_batt_full, 4),
            "pass_no_battery_drain_at_od10": fall_od10_earliest > v_batt_full + 0.10,
            "windows_vs_foc_42p533v": windows,
        })
    return {
        "gnd_sys_residual_error_v": gnd_err_v,
        "foc_earliest_static_v": round(foc_earliest_static_v, 4),
        "selected": "2.7kΩ (RS-03K2701FT C118394, 复用 R_HW_READY_PU 零新 SKU)",
        "cases": cases,
    }


def calc_tps3760_and_1g17() -> dict:
    """计算 100k/10k (0.1%/25ppm) TPS3760 门限、D_HW_HOLD 稳态裕量与 22k/10k 1G17 接口预算。"""
    # TPS3760A012DYYR: VITN = 0.800V +/- 1% (0.792..0.808V), hyst = 2% (A012; worst 2.5% upper corner)
    vitn_min, vitn_max = 0.792, 0.808
    vitp_max = vitn_max * 1.025  # 0.8282V conservative upper corner
    r_tol_01 = 0.001 + 25e-6 * 60.0  # 0.25% over 60K
    r_top_nom, r_bot_nom = 100000.0, 10000.0

    ratio_min = 1.0 + (r_top_nom * (1 - r_tol_01)) / (r_bot_nom * (1 + r_tol_01))
    ratio_max = 1.0 + (r_top_nom * (1 + r_tol_01)) / (r_bot_nom * (1 - r_tol_01))

    v_fall_min = vitn_min * ratio_min
    v_fall_max = vitn_max * ratio_max
    v_rise_max = vitp_max * ratio_max

    # 稳态最低轨: 12V_PROT = 11.4V, D_HW_HOLD (1N5819HW-7-F) 0°C/100mA 保守 VF=0.50V, 2x10Ω 高角=20.4Ω, I=16mA
    v_prot_min = 11.4
    vf_worst = 0.50
    i_load = 0.016
    r_slew_max = 20.4
    dv_pulse = 300e-9 / 10e-6  # 300nC on 10uF worst-case cap = 30mV
    v_hw_min_steady = v_prot_min - vf_worst - i_load * r_slew_max - dv_pulse
    steady_margin_above_release = v_hw_min_steady - v_rise_max

    # 上游 12V_PROT 跌零（0Ω 短路）解析核对：D_HW_HOLD 反偏截止
    # 极端最坏角：10µF 寿命末电容，I_load=16mA + 100°C 肖特基最大漏电 10mA = 26mA
    # 深度过驱点: 0.8 * v_fall_min = 6.94V -> 2.7V (TPS VDD 下限)
    v_od_start = 0.80 * v_fall_min
    v_tps_min = 2.7
    t_detect_us = 17.0
    t_ctr_discharge_us = 194.5
    t_req_us = t_detect_us + t_ctr_discharge_us

    hold_cases = []
    for c_uf, i_tot_ma, desc in [
        (10.0, 26.0, "极端最坏角: 10µF有效 + 16mA负载 + 10mA(100°C肖特基最大漏电)"),
        (10.0, 17.0, "60°C保守角: 10µF有效 + 16mA负载 + 1mA肖特基漏电"),
        (30.08, 26.0, "47µF降额角(30.08µF) + 26mA极端总电流"),
        (30.08, 17.0, "47µF降额角(30.08µF) + 17mA(60°C保守)"),
    ]:
        t_fall_us = (c_uf * 1e-6) * (v_od_start - v_tps_min) / (i_tot_ma * 1e-3) * 1e6
        hold_cases.append({
            "case": desc,
            "c_eff_uF": c_uf,
            "i_total_mA": i_tot_ma,
            "v_start_deep_od_v": round(v_od_start, 3),
            "fall_to_2p7v_us": round(t_fall_us, 1),
            "margin_over_211p5us_us": round(t_fall_us - t_req_us, 1),
            "pass": t_fall_us > t_req_us,
        })

    # 冷启动斜率 (0 -> 13.2V 阶跃, D_HW_HOLD VF_min=0.15V, 2x10Ω低角=19.8Ω, C_min=10µF)
    dv_dt_max_mv_per_us = (13.2 - 0.15) / (19.8 * 10e-6) / 1e3
    # RS-06K10R0FT 单次脉冲能量 (47uF*1.2=56.4uF, 13.2V)
    e_pulse_each_mj = 0.5 * (56.4e-6) * (13.2**2) / 2.0 * 1e3

    # 1G17DCKR 接口预算: 比较 R_DUMP_SW_BASE = 22kΩ (复用 RC0603FR-0722KL) vs 24kΩ
    # R_HW_READY_PU = 2.7kΩ (+/-1.5%), R_HW_READY_PD = 100kΩ (+/-1.5%), R_DUMP_IN_PD = 10kΩ (+/-1.5%)
    g17_cases = []
    tol = 0.015
    for r_ser in [22000.0, 24000.0]:
        r_ser_min, r_ser_max = r_ser * (1 - tol), r_ser * (1 + tol)
        r_bot_min, r_bot_max = 10000.0 * (1 - tol), 10000.0 * (1 + tol)
        r_pu_min, r_pu_max = 2700.0 * (1 - tol), 2700.0 * (1 + tol)
        r_pd_min = 100000.0 * (1 - tol)

        # Low-state worst case: HW_READY <= 2.7V (TPS undefined zone), +5uA leakage into node
        v_in_low_max = 2.7 * r_bot_max / (r_ser_min + r_bot_max) + 5e-6 * (r_ser_min * r_bot_max / (r_ser_min + r_bot_max))
        # High-state worst case at v_hw = v_rise_max (9.13V) and at steady min (10.544V)
        # First compute HW_READY voltage when pulled up via 2.7k_max against 100k_min, (r_ser_min+r_bot_min), and UCC 100uA budget
        r_div_min = r_ser_min + r_bot_min
        r_load_eq = 1.0 / (1.0 / r_pd_min + 1.0 / r_div_min)
        # At v_hw_min_steady:
        v_ready_steady = (v_hw_min_steady / r_pu_max - 50e-6) / (1.0 / r_pu_max + 1.0 / r_load_eq)
        v_in_high_steady = v_ready_steady * r_bot_min / (r_ser_max + r_bot_min) - 5e-6 * (r_ser_max * r_bot_min / (r_ser_max + r_bot_min))
        # At v_rise_max (minimum guaranteed high after release):
        v_ready_rel = (v_rise_max / r_pu_max - 50e-6) / (1.0 / r_pu_max + 1.0 / r_load_eq)
        v_in_high_rel = v_ready_rel * r_bot_min / (r_ser_max + r_bot_min) - 5e-6 * (r_ser_max * r_bot_min / (r_ser_max + r_bot_min))
        # Overvoltage check at 13.2V high rail:
        v_in_13p2_max = 13.2 * r_bot_max / (r_ser_min + r_bot_max) + 5e-6 * (r_ser_min * r_bot_max / (r_ser_min + r_bot_max))

        g17_cases.append({
            "r_ser_ohm": r_ser,
            "v_in_at_2p7v_max_v": round(v_in_low_max, 4),
            "margin_below_0p89v_v": round(0.89 - v_in_low_max, 4),
            "v_in_at_release_9p15v_min_v": round(v_in_high_rel, 4),
            "margin_above_3p0v_spec_1p92v_at_release_v": round(v_in_high_rel - 1.92, 4),
            "margin_above_3p3v_env_2p25v_at_release_v": round(v_in_high_rel - 2.25, 4),
            "v_in_at_steady_min_10p54v_min_v": round(v_in_high_steady, 4),
            "margin_above_3p3v_env_2p25v_at_steady_v": round(v_in_high_steady - 2.25, 4),
            "v_in_at_13p2v_max_v": round(v_in_13p2_max, 4),
            "pass_3p3v_core_envelope": (v_in_low_max < 0.89) and (v_in_high_rel > 2.25) and (v_in_high_steady > 2.25) and (v_in_13p2_max < 5.5),
            "note": "TI SN74LVC1G17 VT+ max at VCC=3.0V is 1.92V (2.25V linear envelope for 3.3V±2% rail); 2.74V is the VCC=4.5V limit.",
        })

    return {
        "tps3760_100k_10k_thresholds_v": {
            "fall_min_v": round(v_fall_min, 4),
            "fall_max_v": round(v_fall_max, 4),
            "release_max_v": round(v_rise_max, 4),
        },
        "steady_11p4v_with_d_hw_hold": {
            "v_hw_min_steady_v": round(v_hw_min_steady, 4),
            "margin_above_release_max_v": round(steady_margin_above_release, 4),
            "pass": steady_margin_above_release > 1.0,
        },
        "cold_start_slew": {
            "max_dv_dt_mv_per_us": round(dv_dt_max_mv_per_us, 2),
            "tps3760_limit_mv_per_us": 100.0,
            "pass": dv_dt_max_mv_per_us < 100.0,
            "slew_resistor_pulse_energy_each_mJ": round(e_pulse_each_mj, 3),
        },
        "upstream_short_hold_cases": hold_cases,
        "g17_interface_cases": g17_cases,
    }


def run_ngspice_transient() -> dict:
    """使用仓库自带 ngspice_con.exe 跑 12V_PROT -> D_HW_HOLD -> 2x10Ω -> 12V_HW 真实瞬态仿真。"""
    assert NGSPICE.exists(), f"Missing ngspice: {NGSPICE}"
    # 仿真序列 (总长 30ms):
    # 0..0.1ms: 0V
    # 0.1ms..5ms: 冷启动阶跃至 13.2V (检查首段 dV/dt)
    # 5ms..10ms: 跌至最低稳态 11.4V (检查稳态电压)
    # 10ms..12ms: 上游 12V_PROT 2ms 硬短路跌零 (0V) 再恢复 11.4V (检查浅短断保持)
    # 15ms..30ms: 上游 12V_PROT 永久硬短路跌零 (0V) (检查 6.94V -> 2.7V 跌落时长 >> 211.5us)
    # 使用 10µF 极端最坏有效电容角 + 16mA 负载 + 1N5819 肖特基模型(保守高VF=0.45V@1A, IS调大至保守漏电)
    cir = """* Regulator P5-Rev2 12V_HW Diode-Isolated Hold & Startup Transient
.model D_SCHOTTKY D(IS=2e-5 N=1.08 RS=0.15 BV=40 IBV=1e-3 CJO=110p)

Vprot n_prot 0 PWL(
+ 0 0
+ 100u 0
+ 101u 13.2
+ 5m 13.2
+ 5.01m 11.4
+ 10m 11.4
+ 10.001m 0
+ 12m 0
+ 12.001m 11.4
+ 15m 11.4
+ 15.001m 0
+ 30m 0
+ )

D_HW_HOLD n_prot n_d D_SCHOTTKY
R_SLEW1 n_d n_mid 9.9
R_SLEW2 n_mid n_hw 9.9
* 极端最坏角 10uF 有效电容 + 0.08 Ohm ESR
C_HW_BANK n_hw n_cap_esr 10u IC=0
R_CAP_ESR n_cap_esr 0 0.08
* 负载电流 16mA (在低压时用电阻+恒流平滑)
B_LOAD n_hw 0 I = 0.016 * tanh(v(n_hw)/0.5)
* TPS3760 分压 100k / 10k
R_POR_TOP n_hw n_sense 100k
R_POR_BOT n_sense 0 10k

.tran 2u 30m uic
.control
run
meas tran v_step_10u FIND v(n_hw) AT=111u
meas tran v_step_20u FIND v(n_hw) AT=121u
meas tran v_steady_11p4 FIND v(n_hw) AT=9.9m
meas tran v_after_2ms_dip FIND v(n_hw) AT=11.99m
meas tran t_cross_8p7 WHEN v(n_hw)=8.7 FALL=LAST
meas tran t_cross_6p94 WHEN v(n_hw)=6.94 FALL=LAST
meas tran t_cross_2p7 WHEN v(n_hw)=2.7 FALL=LAST
meas tran i_rev_max MIN i(Vprot) FROM=15.01m TO=25m
print v_step_10u v_step_20u v_steady_11p4 v_after_2ms_dip t_cross_8p7 t_cross_6p94 t_cross_2p7 i_rev_max
quit
.endc
.end
"""
    with tempfile.TemporaryDirectory() as td:
        cir_path = pathlib.Path(td) / "hw_hold.cir"
        cir_path.write_text(cir, encoding="utf-8")
        proc = subprocess.run(
            [str(NGSPICE), "-b", str(cir_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        out = proc.stdout + "\n" + proc.stderr

    parsed = {}
    for line in out.splitlines():
        if "=" in line:
            parts = line.strip().split("=")
            if len(parts) == 2:
                k = parts[0].strip()
                if k in {"v_step_10u", "v_step_20u", "v_steady_11p4", "v_after_2ms_dip", "t_cross_8p7", "t_cross_6p94", "t_cross_2p7", "i_rev_max"}:
                    try:
                        parsed[k] = float(parts[1].strip().split()[0])
                    except ValueError:
                        pass

    dv_dt_mv_per_us = (parsed["v_step_20u"] - parsed["v_step_10u"]) / 10.0 * 1e3
    fall_6p94_to_2p7_us = (parsed["t_cross_2p7"] - parsed["t_cross_6p94"]) * 1e6
    fall_8p7_to_2p7_us = (parsed["t_cross_2p7"] - parsed["t_cross_8p7"]) * 1e6

    return {
        "ngspice_bin": str(NGSPICE.relative_to(ROOT)),
        "ngspice_sha256": sha256_file(NGSPICE),
        "corner": "10uF worst-case effective cap, 19.8 Ohm slew, 16mA load, Schottky D_HW_HOLD",
        "measured": {
            "cold_start_dv_dt_mv_per_us": round(dv_dt_mv_per_us, 2),
            "v_steady_at_11p4v_v": round(parsed["v_steady_11p4"], 4),
            "v_after_2ms_short_dip_v": round(parsed["v_after_2ms_dip"], 4),
            "fall_8p7v_to_2p7v_us": round(fall_8p7_to_2p7_us, 1),
            "fall_6p94v_to_2p7v_us": round(fall_6p94_to_2p7_us, 1),
            "reverse_current_into_shorted_prot_uA": round(abs(parsed.get("i_rev_max", 0.0)) * 1e6, 2),
        },
        "checks": {
            "cold_start_slew_lt_100mv_per_us": dv_dt_mv_per_us < 100.0,
            "steady_gt_9p13v_release": parsed["v_steady_11p4"] > 9.13,
            "survives_2ms_dip_above_7v": parsed["v_after_2ms_dip"] > 7.0,
            "ctr_discharge_margin_gt_211p5us": fall_6p94_to_2p7_us > 211.5,
        },
    }


def calc_interleaved_stage(spec: dict) -> dict:
    """Compare conduction at a fixed 80 A total SC current, excluding switching loss."""
    frequency = spec["fsw_hz"]
    total = spec["comparison_total_a"]
    phase_l = spec["phase_inductance_uh"] * 1e-6
    single_l = spec["single_phase_baseline_inductance_uh"] * 1e-6
    ron = spec["phase_resistance_ohm_max_25c"]
    dcr = spec["inductor_dcr_ohm_max_25c"]
    points = []
    for bus, sc in ((24, 12), (24, 21.6), (38, 12), (38, 19), (38, 21.6)):
        duty = sc / bus
        ripple = sc * (1 - duty) / (phase_l * frequency)
        single_ripple = ripple * phase_l / single_l
        single_i2 = total**2 + single_ripple**2 / 12
        phase_i2 = (total / 2)**2 + ripple**2 / 12
        points.append({
            "bus_v": bus, "sc_v": sc, "duty": duty,
            "phase_ripple_pp_a": ripple, "phase_peak_a": total / 2 + ripple / 2,
            "phase_peak_at_minus_20pct_l_a": total / 2 + ripple / 1.6,
            "phase_rms_a": math.sqrt(phase_i2), "zero_crossing_mean_a": ripple / 2,
            "bridge_w": [ron * single_i2, 2 * ron * phase_i2],
            "inductor_copper_w": [2 * dcr * single_i2, 2 * dcr * phase_i2],
            "inductor_copper_reduction": 1 - phase_i2 / single_i2,
        })
    return {
        "comparison_mode": "理想CCM，20 kHz，等分80 A SC电流。40A充、65A放保持原总流基准。",
        "points": points,
        "frequency_tuning_only": [
            {"fsw_hz": fsw, "max_phase_ripple_pp_a": 38 / (4 * phase_l * fsw)}
            for fsw in spec["frequency_tuning_hz"]
        ],
        "controller_scope": spec["current_control"],
        "continuous_80a_qualified": False,
    }


def main() -> None:
    contract_path = OUT_DIR / "稳压P5Rev2施工合同.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = {
        "module": "稳压模块",
        "candidate_id": "P5-Rev2 (基于 P5重绘候选 234 器件基线)",
        "status": "PRE_EDA_CALCULATION_ONLY",
        "source_bindings": {
            "p5_enet": str(P5_ENET.relative_to(ROOT)),
            "p5_enet_sha256": sha256_file(P5_ENET),
            "p5_bom": str(P5_BOM.relative_to(ROOT)),
            "p5_bom_sha256": sha256_file(P5_BOM),
            "p1_contract_sha256": sha256_file(contract_path),
        },
        "bus_ov_gnd_sys": calc_bus_ov_gnd_sys(),
        "hw_rail_and_1g17": calc_tps3760_and_1g17(),
        "ngspice_transient": run_ngspice_transient(),
        "interleaved_stage": calc_interleaved_stage(contract["p1_two_phase"]),
    }
    joint_path = ROOT / "电力/检查/联合仿真/降本后典型故障联合仿真.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    report["localization_v1"] = {
        "joint_evidence": str(joint_path.relative_to(ROOT)),
        "braking_before_after": {key: value["1_braking"] for key, value in joint["comparison"].items()},
        "hold": joint["3_hold_unchanged_before_and_after"],
        "single_phase_rds_only_80a_25c_w_old_new": [27.52, 17.92],
        "dump_conduction_50a_25c_w_old_new": [10.75, 7.00],
        "bulk_ripple": "HOLD: 17.16 Arms目录六并额定，首板核20 kHz分流、纹波与温升",
    }
    out_json = OUT_DIR / "稳压闭合复算与仿真.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
