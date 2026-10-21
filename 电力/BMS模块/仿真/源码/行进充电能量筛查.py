#!/usr/bin/env python3
"""First-order BUS capacitor and charge-budget screens; not circuit validation."""
import argparse
import json
import math
from pathlib import Path


def rounded(value):
    return round(value, 6)


def build_report(args):
    c_f = args.bus_capacitance_mf / 1000.0
    v0 = args.initial_voltage_v
    vlim = args.limit_voltage_v
    delta_energy_j = 0.5 * c_f * (vlim * vlim - v0 * v0)
    bus_rows = []
    for p_w in args.power_w:
        t_limit_s = delta_energy_j / p_w
        v_duration = math.sqrt(v0 * v0 + 2.0 * p_w * args.duration_s / c_f)
        bus_rows.append({
            "remaining_injected_power_W": p_w,
            "energy_to_limit_J": rounded(delta_energy_j),
            "time_to_limit_ms": rounded(t_limit_s * 1000.0),
            "ideal_voltage_at_duration_V": rounded(v_duration),
            "crosses_limit_before_duration": t_limit_s < args.duration_s
        })

    solar_rows = []
    concurrent_rows = []
    for eta in args.pv_to_pack_efficiencies:
        pv_pack_power_w = args.pv_nameplate_w * eta
        for voltage_v in args.battery_voltages_v:
            pv_current_a = pv_pack_power_w / voltage_v
            solar_rows.append({
                "battery_voltage_V": voltage_v,
                "pv_to_pack_efficiency": eta,
                "pv_nameplate_power_W": args.pv_nameplate_w,
                "estimated_pack_charge_power_W": rounded(pv_pack_power_w),
                "estimated_pack_charge_current_A": rounded(pv_current_a),
                "remaining_16A_design_input_headroom_A": rounded(args.max_ccl_a - pv_current_a),
                "remaining_16A_design_input_headroom_W": rounded(args.max_ccl_a * voltage_v - pv_pack_power_w),
                "exceeds_6A_daily_design_input": pv_current_a > args.daily_ccl_a,
                "within_16A_max_net_design_input": pv_current_a <= args.max_ccl_a
            })
            for regen_power_w in args.power_w:
                # Regen is treated as electrical power already present at the common BMS-connected BUS.
                # The BUS-to-pack path is idealized; losses and control response are not characterized.
                total_pack_power_w = pv_pack_power_w + regen_power_w
                total_charge_current_a = total_pack_power_w / voltage_v
                concurrent_rows.append({
                    "battery_voltage_V": voltage_v,
                    "pv_to_pack_efficiency": eta,
                    "pv_nameplate_power_W": args.pv_nameplate_w,
                    "estimated_pv_pack_power_W": rounded(pv_pack_power_w),
                    "regen_electrical_power_at_common_BUS_W": regen_power_w,
                    "idealized_total_pack_charge_power_W": rounded(total_pack_power_w),
                    "idealized_net_pack_charge_current_A": rounded(total_charge_current_a),
                    "remaining_16A_design_input_headroom_A": rounded(args.max_ccl_a - total_charge_current_a),
                    "remaining_16A_design_input_headroom_W": rounded(args.max_ccl_a * voltage_v - total_pack_power_w),
                    "exceeds_6A_daily_design_input": total_charge_current_a > args.daily_ccl_a,
                    "exceeds_16A_max_net_design_input": total_charge_current_a > args.max_ccl_a,
                    "within_16A_max_net_design_input": total_charge_current_a <= args.max_ccl_a
                })

    net_current_a = args.solar_current_a + args.regen_current_a - args.load_current_a
    return {
        "title": "行进充电与回收：BUS电容恒功率筛查及30W光伏并发预算",
        "model": "两类独立的一阶筛查：BUS失去吸收后的恒功率电容能量；以及30W光伏和回馈同时接入公共BUS时对电池净充电电流的理想化预算。不是电路/热/闭环验证。",
        "inputs": {
            "known_lower_bound_bus_capacitance_F": c_f,
            "initial_bus_voltage_V": v0,
            "screening_voltage_V": vlim,
            "comparison_duration_s": args.duration_s,
            "remaining_bus_injection_power_cases_W": args.power_w,
            "solar_pv_nameplate_power_W": args.pv_nameplate_w,
            "battery_voltage_cases_V": args.battery_voltages_v,
            "pv_to_pack_efficiency_cases": args.pv_to_pack_efficiencies,
            "daily_charge_design_input_A": args.daily_ccl_a,
            "maximum_net_charge_design_input_A": args.max_ccl_a
        },
        "equations": {
            "capacitor_energy_change_J": "0.5*C*(V_limit^2 - V_initial^2)",
            "time_to_voltage_limit_s": "0.5*C*(V_limit^2 - V_initial^2)/P_remaining",
            "voltage_at_time_s": "sqrt(V_initial^2 + 2*P_remaining*t/C)",
            "pv_pack_power_W": "PV_nameplate_power * eta_PV_to_pack",
            "pv_pack_current_A": "PV_pack_power / V_pack",
            "idealized_concurrent_pack_current_A": "(PV_pack_power + P_regen_at_common_BUS) / V_pack"
        },
        "results": bus_rows,
        "solar_30W_nameplate_budget": {
            "status": "screening_input_only_not_a_selected_panel_string_or_validated_MPPT",
            "nameplate_power_W": args.pv_nameplate_w,
            "panel_count_series_parallel": "未假设，尚未冻结",
            "pv_to_pack_efficiency_definition": "从30W组件名牌输出到电池端净充电功率的合并筛查效率；0.8/0.9/1.0是敏感性边界，不代表特定MPPT实测效率。",
            "battery_voltage_cases_V": args.battery_voltages_v,
            "efficiency_cases": args.pv_to_pack_efficiencies,
            "battery_side_cases": solar_rows,
            "charge_current_limits": {
                "daily_design_input_A": args.daily_ccl_a,
                "maximum_net_charge_design_input_A": args.max_ccl_a,
                "limit_status": "16A仅为设计输入算术上限，不是已验证的BMS板级限流能力；OCC/FET过流保护不能替代CC限流。"
            },
            "power_path_boundary": "架构边界：独立MPPT接公共BUS；电池侧净充电必须经BMS主CHG FET及电流分流器进入电池，禁止接B0或其他方式绕过CHG FET/分流器。此筛查不表示MPPT硬件或端到端闭环已验收。"
        },
        "solar_regeneration_concurrency": {
            "regen_power_semantics": "500/1000/2000W参数表示在公共BMS连接BUS上的回馈电功率；直接作为电池侧预算功率，未建模BUS到电池路径压降/损耗。结果是输入组合的理想化净充电请求，不是实际可吸收能力。",
            "single_shared_16A_budget": "太阳能与回馈共同占用电池净充电能力；不得由各来源分别重复取得完整CCL。实际应由单一协调策略按最新本地CCL分配。",
            "cases": concurrent_rows,
            "full_battery_or_charge_prohibited": "当电池满电、低温/故障或CHG FET断开而不允许充电时，MPPT/其他源必须停止向BUS注入；剩余回馈能量须由已协调且经过验证的超容/本地母线吸能或机械制动处理。超容满、通信过期或吸收能力未知时不得继续把其计入额度。"
        },
        "same_bus_generic_stress_example": {
            "solar_current_A": args.solar_current_a,
            "regen_current_A": args.regen_current_a,
            "bus_load_current_A": args.load_current_a,
            "net_current_into_battery_A": net_current_a,
            "comparison": {
                "daily_CCL_candidate_A": args.daily_ccl_a,
                "max_net_CCL_design_input_A": args.max_ccl_a,
                "exceeds_daily_CCL": net_current_a > args.daily_ccl_a,
                "below_max_design_input": net_current_a <= args.max_ccl_a
            },
            "classification": "仅为通用6A光伏电流压力示例，不代表30W太阳能板；实际30W名牌输入已在solar_30W_nameplate_budget中单独计算。",
            "assumption": "电流已折算到相同电池BUS。真实混合电压侧功率需使用实测电压、方向和转换效率。"
        },
        "interpretation_limits": [
            "60mF是已记录的两块FOC BUS电容下限，不含其他母线电容。",
            "BUS电容能量模型假设电池/超容/DC-DC/dump均不吸收、无负载无损耗且剩余注入功率恒定；仅用于说明失去吸收后的时间尺度。",
            "300ms理想电压是跨越筛查电压之后的数学外推，不是实际工作电压预测。",
            "30W是太阳能名牌输入，不代表任意光照温度下的可用功率；组件数量、串并联和工作/开路电压未假设。",
            "16A是净充电设计输入算术上限，非已验证的限流、全温能力或安全保证；日常设计输入仍为6A。",
            "回馈与光伏并发预算不模拟实时CCL、MPPT动态、FOC体二极管、母线压降、过压钳位、超容SOC/功率或机械制动。",
            "这不是SPICE、开关、控制环路、热或实物验证。"
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus-capacitance-mf", type=float, default=60.0)
    parser.add_argument("--initial-voltage-v", type=float, default=37.8)
    parser.add_argument("--limit-voltage-v", type=float, default=45.0)
    parser.add_argument("--duration-s", type=float, default=0.300)
    parser.add_argument("--power-w", type=float, nargs="+", default=[500.0, 1000.0, 2000.0])
    parser.add_argument("--pv-nameplate-w", type=float, default=30.0)
    parser.add_argument("--battery-voltages-v", type=float, nargs="+", default=[22.5, 30.0, 37.8])
    parser.add_argument("--pv-to-pack-efficiencies", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    parser.add_argument("--solar-current-a", type=float, default=6.0,
                        help="generic stress illustration only; it is not the 30W PV model")
    parser.add_argument("--regen-current-a", type=float, default=10.0)
    parser.add_argument("--load-current-a", type=float, default=4.0)
    parser.add_argument("--daily-ccl-a", type=float, default=6.0)
    parser.add_argument("--max-ccl-a", type=float, default=16.0)
    parser.add_argument("--output", type=Path, default=Path("BMS模块/检查/行进充电与回收能量筛查.json"))
    args = parser.parse_args()
    if args.bus_capacitance_mf <= 0 or args.duration_s <= 0 or args.pv_nameplate_w < 0:
        parser.error("电容与持续时间必须为正，光伏名牌功率不得为负")
    if any(v <= 0 for v in args.battery_voltages_v) or any(not 0 <= e <= 1 for e in args.pv_to_pack_efficiencies):
        parser.error("电池电压必须为正，效率必须在0到1之间")
    if any(p <= 0 for p in args.power_w):
        parser.error("功率案例必须为正")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
