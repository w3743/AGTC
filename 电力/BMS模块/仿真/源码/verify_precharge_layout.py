#!/usr/bin/env python3
"""Reproduce paper calculations for the BMS precharge Layout closure."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "BMS模块" / "仿真" / "结果" / "预充Layout闭合.json"


def tpl_tip_from_resistance_ohm(resistance_ohm: float) -> float:
    """TPL5111 datasheet fit for 1 < T <= 5 s; R is converted to 100-ohm units."""
    rd = resistance_ohm / 100.0
    return (0.2253 * rd**2 - 20.7654 * rd + 570.5679) / 100.0


def main() -> None:
    # Timer set resistor: 7.32 kΩ, ±1%, ±100 ppm/°C, rated at 25°C.
    r_set_nom = 7320.0
    r_set_max_60c = r_set_nom * 1.01 * (1 + 100e-6 * (60 - 25))
    r_set_min_0c = r_set_nom * 0.99 * (1 - 100e-6 * (25 - 0))

    # Datasheet errors are conservatively compounded as independent worst signs.
    # Reference: VDD=2.5 V, TA=25°C. Circuit supply range is 3.0–3.6 V.
    dv = 3.6 - 2.5
    timer_high_factor = (
        (1 + 0.006)  # tIP setting accuracy, excluding Rext precision
        * (1 + 25e-6 * dv)  # tIP setting supply sensitivity
        * (1 + 0.005)  # oscillator accuracy at TA=25°C, VDD=2.5 V
        * (1 + 400e-6 * (60 - 25))  # maximum oscillator temperature drift
        * (1 + 0.004 * dv)  # oscillator supply sensitivity
        * (1 + 0.0024)  # lifetime change
    )
    dv_low = 3.0 - 2.5
    timer_low_factor = (
        (1 - 0.006)
        * (1 - 25e-6 * dv_low)
        * (1 - 0.005)
        * (1 - 400e-6 * (60 - 25))
        * (1 - 0.004 * dv_low)
        * (1 - 0.0024)
    )
    tip_nom = tpl_tip_from_resistance_ohm(r_set_nom)
    tip_high = tpl_tip_from_resistance_ohm(r_set_max_60c) * timer_high_factor
    tip_low = tpl_tip_from_resistance_ohm(r_set_min_0c) * timer_low_factor

    # TPL5111: DRVn pulse is tIP - 50 ms; first-start Rext delay is 100–120 ms.
    drv_high = tip_high - 0.050
    drv_low = tip_low - 0.050
    active_window_high = drv_high + 0.120
    active_window_low = drv_low + 0.100

    # External TE CJP300J8R2J resistor. Bound TCR to the datasheet's 350°C
    # maximum CJP300 surface temperature at rated free-air power, not only ambient.
    r_pre_min_25c = 8.2 * 0.95
    r_pre_min_60c_ambient_only = r_pre_min_25c * (1 - 200e-6 * (60 - 25))
    r_pre_min_at_max_surface = r_pre_min_25c * (1 - 200e-6 * (350 - 25))
    i_pre_25c = 37.8 / r_pre_min_25c  # ignores MOSFET drops: conservative current
    i_pre_60c_ambient_only = 37.8 / r_pre_min_60c_ambient_only
    i_pre_60c = 37.8 / r_pre_min_at_max_surface
    p_pre_25c = 37.8**2 / r_pre_min_25c
    p_pre_60c_ambient_only = 37.8**2 / r_pre_min_60c_ambient_only
    p_pre_60c = 37.8**2 / r_pre_min_at_max_surface
    c_bus = 60e-3
    e_cap = 0.5 * c_bus * 37.8**2
    derated_power_60c_linear = 300.0 * (250.0 - 60.0) / (250.0 - 25.0)
    fault_energy_2p97_s_60c = p_pre_60c * 2.970  # worst-TCR constant-power scenario only

    # Two common-gate P-MOSFETs. RDS(on) is guaranteed at the stated 25°C tests;
    # Qg/Ciss and resulting RC are typical-only models and are never treated as maxima.
    qg_each_typ_nc = 136.0
    ciss_each_typ_pf = 3810.0
    rds_each_max_25c_ohm = 0.058
    rds_each_max_25c_minus4p5_ohm = 0.065
    i_mos_current_bound = i_pre_60c
    vds_each_on_25c_bound = i_mos_current_bound * rds_each_max_25c_ohm
    p_mos_each_25c_bound = i_mos_current_bound**2 * rds_each_max_25c_ohm
    i_on_at_rds_max_minus10_25c = 37.8 / (r_pre_min_25c + 2 * rds_each_max_25c_ohm)
    i_on_at_rds_max_minus4p5_25c = 37.8 / (
        r_pre_min_25c + 2 * rds_each_max_25c_minus4p5_ohm
    )
    vds_at_rds_max_minus10_25c = i_on_at_rds_max_minus10_25c * rds_each_max_25c_ohm
    p_at_rds_max_minus10_25c = i_on_at_rds_max_minus10_25c**2 * rds_each_max_25c_ohm
    vds_at_rds_max_minus4p5_25c = i_on_at_rds_max_minus4p5_25c * rds_each_max_25c_minus4p5_ohm
    p_at_rds_max_minus4p5_25c = i_on_at_rds_max_minus4p5_25c**2 * rds_each_max_25c_minus4p5_ohm
    # Selected BMS/FOC-shared 4.7 kΩ ±1% 0603 (RC0603FR-074K7L).
    # QPRE collector limiter remains 4.7 kΩ ±1% 1206.
    gate_r_pullup_nom = 4_700.0
    gate_tau_typ_s = gate_r_pullup_nom * 2 * ciss_each_typ_pf * 1e-12
    gate_three_tau_typ_s = 3 * gate_tau_typ_s
    gate_charge_typ_total_c = 2 * qg_each_typ_nc * 1e-9

    # Static zener-gate-drive boundary. Switching parasitics and low-current
    # VZ minimum are not guaranteed by the cited datasheet conditions.
    zener_vz_min_at_20ma = 11.4
    zener_vz_max_at_20ma = 12.6
    zener_iz_test_ma = 20.0
    zener_izk_ma = 0.25
    gate_r_pullup_min = 4_700.0 * 0.99
    gate_r_pullup_max = 4_700.0 * 1.01
    gate_r_limit_min = 4_700.0 * 0.99
    gate_r_limit_max = 4_700.0 * 1.01
    qpre_npn_vce_sat_assumption_v = 0.2
    gate_vgs_unclamped_37p8_worst = (
        37.8 * gate_r_pullup_max / (gate_r_pullup_max + gate_r_limit_min)
    )
    gate_vgs_unclamped_22p5_min = (
        (22.5 - qpre_npn_vce_sat_assumption_v)
        * gate_r_pullup_min
        / (gate_r_pullup_min + gate_r_limit_max)
    )
    gate_vgs_unclamped_22p5_max = (
        (22.5 - qpre_npn_vce_sat_assumption_v)
        * gate_r_pullup_max
        / (gate_r_pullup_max + gate_r_limit_min)
    )
    qpre_sink_max_a = (
        37.8 - zener_vz_min_at_20ma - qpre_npn_vce_sat_assumption_v
    ) / gate_r_limit_min
    qpre_limit_power_max_w = qpre_sink_max_a**2 * gate_r_limit_min
    qpre_pullup_power_at_vzmin_w = zener_vz_min_at_20ma**2 / gate_r_pullup_min
    qpre_zener_current_at_vzmin_a = qpre_sink_max_a - (
        zener_vz_min_at_20ma / gate_r_pullup_min
    )
    qpre_zener_power_at_vzmin_w = (
        zener_vz_min_at_20ma * qpre_zener_current_at_vzmin_a
    )
    base_driver_voh_min_assumed_v = 2.4
    qpre_npn_vbe_max_assumed_v = 1.0
    qpre_base_resistor_max_ohm = 1_000.0 * 1.01
    qpre_be_pull_min_ohm = 100_000.0 * 0.99
    qpre_base_available_min_a = (
        (base_driver_voh_min_assumed_v - qpre_npn_vbe_max_assumed_v)
        / qpre_base_resistor_max_ohm
        - qpre_npn_vbe_max_assumed_v / qpre_be_pull_min_ohm
    )
    qpre_forced_beta_at_max_sink = qpre_sink_max_a / qpre_base_available_min_a
    qpre_base_drive_logic_load_3v3_typ_a = (3.3 - 0.7) / 1_000.0

    results = {
        "scope": "paper/Layout closure; physical board validation is not claimed; 2.970 s is a legacy comparison scenario, not a confirmed product cutoff requirement",
        "input_corner": {
            "pack_voltage_v": 37.8,
            "ambient_c": [0, 60],
            "bus_capacitance_f": c_bus,
            "timer_supply_v": [3.0, 3.6],
        },
        "timer": {
            "r_set_nominal_ohm": r_set_nom,
            "r_set_min_0c_ohm": r_set_min_0c,
            "r_set_max_60c_ohm": r_set_max_60c,
            "tip_nominal_s": tip_nom,
            "tip_min_conservative_s": tip_low,
            "tip_max_conservative_s": tip_high,
            "drv_pulse_min_s": drv_low,
            "drv_pulse_max_s": drv_high,
            "first_start_delay_min_s": 0.100,
            "first_start_delay_max_s": 0.120,
            "ff_b_power_to_drv_low_min_s": active_window_low,
            "ff_b_power_to_drv_low_max_s": active_window_high,
            "margin_to_2p970_s": 2.970 - active_window_high,
            "reference_deadline_s": 2.970,
            "reference_deadline_status": "legacy comparison scenario; not a confirmed physical cutoff requirement",
            "max_factor": timer_high_factor,
            "min_factor": timer_low_factor,
            "polynomial_coefficients": {"a": 0.2253, "b": -20.7654, "c": 570.5679},
            "status": "PASS for TMR_ACTIVE/logic-window deassertion only; excludes BJT and MOSFET physical release",
        },
        "precharge_resistor": {
            "mpn": "TE CJP300J8R2J",
            "nominal_ohm": 8.2,
            "tolerance_fraction": 0.05,
            "tcr_ppm_per_c_max": 200,
            "r_min_25c_ohm": r_pre_min_25c,
            "r_min_60c_ambient_only_ohm": r_pre_min_60c_ambient_only,
            "r_min_at_max_surface_350c_ohm": r_pre_min_at_max_surface,
            "initial_current_upper_ignore_mos_drops_25c_a": i_pre_25c,
            "initial_current_upper_ignore_mos_drops_60c_ambient_only_a": i_pre_60c_ambient_only,
            "initial_current_upper_ignore_mos_drops_worst_tcr_350c_a": i_pre_60c,
            "initial_power_upper_ignore_mos_drops_25c_w": p_pre_25c,
            "initial_power_upper_ignore_mos_drops_60c_ambient_only_w": p_pre_60c_ambient_only,
            "initial_power_upper_ignore_mos_drops_worst_tcr_350c_w": p_pre_60c,
            "bus_capacitor_energy_j": e_cap,
            "60c_derated_power_linear_envelope_w": derated_power_60c_linear,
            "continuous_power_fraction_at_60c": p_pre_60c / derated_power_60c_linear,
            "constant_worst_tcr_power_fault_scenario_2p970s_j": fault_energy_2p97_s_60c,
            "status": "PASS when external, unobstructed/free-air mounted per TE rating; no added heatsink indicated",
        },
        "qpre_mosfets": {
            "mpn": "NCE01P30K x2, P-channel common-source, common-gate",
            "topology": "fully-enhanced series path during precharge; external resistor dominates voltage drop",
            "rds_on_max_25c_ohm_per_fet_at_vgs_minus10v": rds_each_max_25c_ohm,
            "rds_on_max_25c_ohm_per_fet_at_vgs_minus4p5v": rds_each_max_25c_minus4p5_ohm,
            "on_state_current_upper_at_rpre_min_25c_and_minus10v_a": i_on_at_rds_max_minus10_25c,
            "on_state_current_upper_at_rpre_min_25c_and_minus4p5v_a": i_on_at_rds_max_minus4p5_25c,
            "on_state_vds_per_fet_at_rpre_min_25c_and_minus10v_v": vds_at_rds_max_minus10_25c,
            "on_state_power_per_fet_at_rpre_min_25c_and_minus10v_w": p_at_rds_max_minus10_25c,
            "on_state_vds_per_fet_at_rpre_min_25c_and_minus4p5v_v": vds_at_rds_max_minus4p5_25c,
            "on_state_power_per_fet_at_rpre_min_25c_and_minus4p5v_w": p_at_rds_max_minus4p5_25c,
            "on_state_vds_per_fet_upper_at_minus10v_25c_v": vds_each_on_25c_bound,
            "on_state_power_per_fet_upper_at_minus10v_25c_w": p_mos_each_25c_bound,
            "gate_pullup_mpn": "Yageo RC0603FR-074K7L; LCSC C99782; shared project MPN",
            "gate_pullup_ohm_nominal": gate_r_pullup_nom,
            "gate_charge_typ_nc_per_fet": qg_each_typ_nc,
            "gate_charge_typ_pair_nc": gate_charge_typ_total_c * 1e9,
            "input_capacitance_typ_pf_per_fet": ciss_each_typ_pf,
            "simple_rc_tau_typ_s_pair": gate_tau_typ_s,
            "simple_rc_3tau_typ_s_pair": gate_three_tau_typ_s,
            "physical_turnoff_max_guaranteed": False,
            "status": "no multi-second linear-SOA premise; transient SOA and actual turn-off still require controlled device limits or board qualification",
        },
        "gate_clamp": {
            "network": "4.7 kOhm ±1% QPRE_G-QPRE_SRC pullup; 4.7 kOhm ±1% QPRE_G to NPN collector; MMSZ5242BT1G reverse clamp",
            "r_pullup_min_ohm": gate_r_pullup_min,
            "r_pullup_max_ohm": gate_r_pullup_max,
            "r_limit_min_ohm": gate_r_limit_min,
            "r_limit_max_ohm": gate_r_limit_max,
            "npn_vce_sat_assumption_v": qpre_npn_vce_sat_assumption_v,
            "zener_vz_min_at_20ma_v": zener_vz_min_at_20ma,
            "zener_vz_max_at_20ma_v": zener_vz_max_at_20ma,
            "zener_iz_test_ma": zener_iz_test_ma,
            "zener_izk_ma": zener_izk_ma,
            "unclamped_abs_vgs_at_37p8v_worst_resistor_ratio_v": gate_vgs_unclamped_37p8_worst,
            "unclamped_abs_vgs_at_22p5v_min_v": gate_vgs_unclamped_22p5_min,
            "unclamped_abs_vgs_at_22p5v_max_v": gate_vgs_unclamped_22p5_max,
            "max_npn_collector_sink_at_37p8_vz_min_a": qpre_sink_max_a,
            "r_limit_max_dissipation_at_37p8_vz_min_w": qpre_limit_power_max_w,
            "r_pullup_dissipation_at_vz_min_w": qpre_pullup_power_at_vzmin_w,
            "zener_current_at_vz_min_a": qpre_zener_current_at_vzmin_a,
            "zener_dissipation_at_vz_min_w": qpre_zener_power_at_vzmin_w,
            "steady_abs_vgs_upper_if_zener_regulates_v": zener_vz_max_at_20ma,
            "npn_base_drive_check": {
                "base_driver_voh_min_assumption_v": base_driver_voh_min_assumed_v,
                "npn_vbe_max_assumption_v": qpre_npn_vbe_max_assumed_v,
                "base_available_min_after_100k_be_pull_a": qpre_base_available_min_a,
                "npn_forced_beta_at_max_sink": qpre_forced_beta_at_max_sink,
                "precharge_window_load_typ_at_3p3v_a": qpre_base_drive_logic_load_3v3_typ_a,
                "status": "paper check only; confirm exact U_GATE1 VOH at this load and MMBT5551 VCEsat/current/temperature on first board",
            },
            "off_state_static_current": "4.7k pullup has zero ideal DC current when QPRE_G=QPRE_SRC; MMBT5551 datasheet gives ICBO only at a different test condition, not a guaranteed circuit ICEO bound. No off-state leakage maximum is claimed.",
            "status": "unclamped DC divider is approximately 11.04–11.26 V at 22.5 V and 19.09 V worst at 37.8 V; static bias/load powers are bounded by stated assumptions, but Zener low-current knee, logic VOH, switching transients and physical cutoff remain measurement items",
        },
        "evidence_and_limits": {
            "official_vendor_sources": [
                "https://www.ti.com/lit/ds/symlink/tpl5111.pdf",
                "https://www.te.com/en/product-1-2176508-5.html",
                "https://www.te.com/commerce/DocumentDelivery/DDEController?Action=srchrtrv&DocFormat=pdf&DocLang=English&DocNm=1773309-8&DocType=Data+Sheet&PartCntxt=1-2176508-5",
                "https://www.onsemi.com/pdf/datasheet/mmsz5221et1-d.pdf",
            ],
            "nce_local_pdf": "BMS模块/厂商资料/NCE01P30K_V2_Elecfans镜像.pdf (mirror copy; controlled manufacturer version not obtained)",
            "mmbt_local_pdf": "BMS模块/厂商资料/MDD_MMBT5551_C408396.pdf (no guaranteed storage/turn-off time found)",
            "pre_ok_used_as_cutoff": False,
            "full_chain_physical_max_time_s": None,
            "layout_verdict": "FAIL to guarantee physical current extinction within the legacy 2.970 s scenario; candidate Layout may proceed under the constraints in the companion Markdown",
        },
    }

    assert abs(tip_nom - 2.57752092) < 1e-8
    assert active_window_high < 2.970
    assert p_pre_60c < derated_power_60c_linear
    assert gate_vgs_unclamped_22p5_min >= 10.0
    assert gate_vgs_unclamped_37p8_worst < 20.0
    assert qpre_limit_power_max_w < 0.25
    assert qpre_base_available_min_a > 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps({
        "tip_nominal_s": tip_nom,
        "logic_window_max_s": active_window_high,
        "margin_to_2p970_s": 2.970 - active_window_high,
        "precharge_power_60c_upper_w": p_pre_60c,
        "resistor_derated_power_60c_w": derated_power_60c_linear,
        "qpre_gate_pullup_ohm": gate_r_pullup_nom,
        "qpre_turnoff_rc_tau_typ_us": gate_tau_typ_s * 1e6,
        "qpre_turnoff_rc_3tau_typ_us": gate_three_tau_typ_s * 1e6,
        "qpre_vgs_unclamped_low_pack_min_v": gate_vgs_unclamped_22p5_min,
        "qpre_vgs_unclamped_high_pack_max_v": gate_vgs_unclamped_37p8_worst,
        "qpre_collector_sink_max_ma": qpre_sink_max_a * 1e3,
        "qpre_limit_resistor_worst_power_w": qpre_limit_power_max_w,
        "qpre_npn_forced_beta_paper": qpre_forced_beta_at_max_sink,
        "mos_turnoff_max_guaranteed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
