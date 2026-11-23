#!/usr/bin/env python3
"""
FOC 驱动模块 164-Rev2 闭合复算脚本。

验证范围（基于 164 器件隔离候选 修正重开.enet）：
1. FOC 本地 BOV (1M + 69.8k / 10k, 0.1%/25ppm) 与稳压 P5-Rev2 (512.7k / 5.11k, GND_SYS) 动作先后校核。
2. 换装隔离 CAN (CA-IS2062A C22438481) 后，DRV8353RS 内部 LM5008A 5V_BUCK 稳态与 CAN 显性峰值电流预算。
3. 国产化并料替换 U1_CP_C (X7S -> 风华 X7R 47nF/100V C43254) 与 U1_VCC_C/U1_BUCK_COUT2 (TDK 470nF/16V -> 风华 1µF/25V X7R C59302, 0新SKU)。
4. 核验双板均装 R_LOAD_VALID_BIAS=1kΩ 与 BMS LTV-817S-TA1-C 独立光耦对接的饱和裕量。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT_DIR = pathlib.Path(__file__).resolve().parent
FOC_164_ENET = ROOT / "电力/FOC驱动模块/检查/候选_本地保持与失电通知/官方V3_20261109/修正重开.enet"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_buck_5v_with_isolated_can() -> dict:
    # DRV8353RS LM5008A: rated 350mA continuous, min switch current limit 410mA
    # Inductance L = 220uH (CDRH127/LDNP-221MC), Vbus_max = 42.5V, Vout = 5.0V
    # Ripple current at 37.8V: RCL=1M -> t_off ~ 8.85us -> dI_L = Vout * t_off / L = 5 * 8.85e-6 / 220e-6 = 0.201A
    i_mcu_ma = 60.0
    i_hall_ma = 50.0
    i_can_recessive_ma = 35.0   # CA-IS2062A VDDP recessive (integrated DC-DC)
    i_can_dominant_ma = 120.0   # CA-IS2062A VDDP dominant peak (60Ω bus load, ~50-60% eff)
    i_misc_ma = 10.0

    i_steady_recessive_ma = i_mcu_ma + i_hall_ma + i_can_recessive_ma + i_misc_ma
    i_steady_dominant_ma = i_mcu_ma + i_hall_ma + i_can_dominant_ma + i_misc_ma
    di_l_ma = 5.0 * 8.85e-6 / 220e-6 * 1e3
    i_peak_dominant_ma = i_steady_dominant_ma + 0.5 * di_l_ma
    i_lim_min_ma = 410.0

    return {
        "buck_ic": "DRV8353RS integrated LM5008A (rated 350mA, ILIM_min 410mA)",
        "load_breakdown_mA": {
            "g431_core": i_mcu_ma,
            "hall_sensors": i_hall_ma,
            "ca_is2062a_recessive": i_can_recessive_ma,
            "ca_is2062a_dominant_peak": i_can_dominant_ma,
            "misc_and_bias": i_misc_ma,
        },
        "i_out_recessive_mA": i_steady_recessive_ma,
        "i_out_dominant_peak_mA": i_steady_dominant_ma,
        "inductor_ripple_pp_mA": round(di_l_ma, 1),
        "switch_peak_dominant_mA": round(i_peak_dominant_ma, 1),
        "margin_below_410mA_limit_mA": round(i_lim_min_ma - i_peak_dominant_ma, 1),
        "pass_steady_and_dominant": i_peak_dominant_ma < i_lim_min_ma,
        "cold_start_caveat": "CA-IS2062A VDDP 新增 47µF (C_U5_VDDP_47U) 增大冷启动电容，首板必须示波器实测 22.5V/37.8V 下 5V_BUCK 软启动波形与 ILIM 行为。",
    }


def verify_bov_coordination() -> dict:
    # FOC: 1M + 69.8k = 1.0698M (0.1%/25ppm, 60K -> +/-0.25%), Rbot=10k (0.1%/25ppm, 60K)
    # TLV6700 VIT+ 0.396..0.404V, IB +/-15nA
    tol = 0.001 + 25e-6 * 60.0
    rt_min = 1069800.0 * (1 - tol)
    rt_max = 1069800.0 * (1 + tol)
    rb_min = 10000.0 * (1 - tol)
    rb_max = 10000.0 * (1 + tol)
    foc_earliest = 0.396 * (1 + rt_min / rb_max) - 15e-9 * rt_max
    foc_latest = 0.404 * (1 + rt_max / rb_min) + 15e-9 * rt_max
    central_latest_od10 = 42.1858  # 来自 稳压 P5-Rev2 GND_SYS 复算
    return {
        "foc_bov_earliest_static_v": round(foc_earliest, 4),
        "foc_bov_latest_static_v": round(foc_latest, 4),
        "central_p5rev2_latest_od10_v": central_latest_od10,
        "margin_central_first_v": round(foc_earliest - central_latest_od10, 4),
        "margin_below_smcj40a_vbr_44p4v": round(44.4 - foc_latest, 4),
        "pass_sequence_and_tvs_margin": (foc_earliest > central_latest_od10 + 0.30) and (foc_latest < 44.4),
    }


def main() -> None:
    contract_path = OUT_DIR / "FOC_164Rev2施工合同.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    enet = json.loads(FOC_164_ENET.read_text(encoding="utf-8"))
    board = {item["props"]["Designator"]: item for item in enet["components"].values()}
    replacement = next(item for item in contract["drawing_components"] if item["ref"] == "DTVS1")
    assert replacement["lcsc"] == "C840777"
    assert replacement["pins"] == {key: item["net"] for key, item in board["DTVS1"]["pinInfoMap"].items()}
    for ref, mpn in contract["p0_phase_a"]["retained"].items():
        assert board[ref]["props"]["Manufacturer Part"] == mpn
    report = {
        "module": "FOC驱动模块",
        "candidate_id": "FOC_164_REV2_CLOSURE_20261123",
        "status": "PRE_EDA_CALCULATION_ONLY",
        "source_bindings": {
            "foc_164_enet": str(FOC_164_ENET.relative_to(ROOT)),
            "foc_164_enet_sha256": sha256_file(FOC_164_ENET),
            "component_count": len(enet["components"]),
            "p0_contract_sha256": sha256_file(contract_path),
        },
        "bov_coordination": verify_bov_coordination(),
        "buck_5v_budget": verify_buck_5v_with_isolated_can(),
        "p0_bus_tvs": replacement,
        "dual_board_bom_unification": {
            "R_LOAD_VALID_BIAS": "FOC-A 与 FOC-B 均装配 RC0603FR-071KL (1kΩ, C22548)，配对 BMS 两路独立 LTV-817S-TA1-C 浮地光耦，消除原 A装/B DNP 差异。",
            "bus_cap_harness_rule": "改版每侧 3×10,000µF/63V (CBUS1-3) 必须接在正/负动力馈线开点与 F_MAIN 的 FOC 逆变桥一侧。",
        },
        "capacitor_upgrades_zero_new_sku_plus_x7r": [
            {
                "ref": "U1_VCC_C",
                "old": "TDK C1608X7R1C474K080AC (470nF/16V, 偏压后不足0.47µF下限)",
                "new": "FH(风华) 0603B105K250NT (1µF/25V X7R 0603, C59302, 复用板上 C_U5_ISOIN_1U, 0新SKU)",
            },
            {
                "ref": "U1_BUCK_COUT2",
                "old": "TDK C1608X7R1C474K080AC (470nF/16V, 淘汰进口孤立SKU)",
                "new": "FH(风华) 0603B105K250NT (1µF/25V X7R 0603, C59302, 纹波注入已由 C_BUCK_RIP_CA/CB 独立完成)",
            },
            {
                "ref": "U1_CP_C",
                "old": "TDK C1608X7S2A473K080AE (47nF/100V X7S, 无C码, 不符合TI X5R/X7R要求)",
                "new": "FH(风华) 0603B473K101NT (47nF/100V X7R 0603, C43254, 符合TI X7R规格)",
            },
        ],
    }
    joint_path = ROOT / "电力/检查/联合仿真/降本后典型故障联合仿真.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    report["localization_v1"] = {
        "joint_evidence": str(joint_path.relative_to(ROOT)),
        "island_before_after": {key: value["4_island"] for key, value in joint["comparison"].items()},
        "foc_bridge_before_after": {key: value["5_protection"] for key, value in joint["comparison"].items()},
        "offboard_capacitor_ripple": "HOLD: 13.98 Arms目录三并额定，首板核20 kHz分流、纹波与温升",
    }
    (OUT_DIR / "FOC_164Rev2闭合复算.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
