"""从现行BOM及旧AFE候选推导精简清单；只生成设计候选，不修改CAD/BOM。"""
from pathlib import Path
import csv
import hashlib
import json

MODULE = Path(__file__).resolve().parents[1]
OLD = MODULE / "工具/待审AFE方案.json"
BOM = MODULE / "原理图准备/BOM.csv"
ENET = MODULE.parent / "检查/统一审查/稳压当前源.enet"
OUTPUT = MODULE / "原理图准备/精简替代方案.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = json.loads(OLD.read_text(encoding="utf-8-sig"))
    with BOM.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    current = {r["Reference"]: r for r in rows}
    enet = json.loads(ENET.read_text(encoding="utf-8-sig"))
    native = {v["props"]["Designator"] for v in enet["components"].values()}
    additions = []
    for group in old["expected_additions"]:
        if "ref" in group:
            refs = [group["ref"]]
        elif group["ref_pattern"].startswith("R_VC0,"):
            refs = ["R_VC0"] + [f"R_VC{i}" for i in range(1, 8)] + ["R_VC16"]
        elif group["ref_pattern"].startswith("C_VC0,"):
            refs = ["C_VC0"] + [f"C_VC{i}" for i in range(1, 8)] + ["C_VC16"]
        else:
            refs = group["ref_pattern"].split(",")
        assert len(refs) == group["quantity"], group
        additions += [{"ref": ref, "mpn": group["mpn"]} for ref in refs]
    assert len(additions) == 45

    excluded = {"U_ADC_ISO", "C_MUX_VDD", "R_MUX_BUS_PD", "R_MUX_SUM_PD",
                "R_MUX_IMON_PD", "R_IMON_PD", "C_AFE_CP1"}
    additions = [a for a in additions if a["ref"] not in excluded]
    assert len(additions) == 38
    reuse = {
        "Q_AFE_PRE": "Q_PMOS_CTRL",  # 同厂同料候选；不能拿Diodes规格替代JKSEMI保证
        "R_AFE_PACK": "R_DRV_FAULT_PD",
        **{f"R_VC{i}": "R_PWM_H_SER" for i in [0, 1, 2, 3, 4, 5, 6, 7, 16]},
        "R_AFE_BAT_SER": "R_PWM_H_SER",
        **{f"R_AFE_{name}_PD": "R_PWM_H_PD" for name in ["DCHG", "DDSG", "PERMIT"]},
    }
    for a in additions:
        if a["ref"] in reuse:
            ref = reuse[a["ref"]]
            a["previous_candidate_mpn"] = a["mpn"]
            a["mpn"] = current[ref]["Manufacturer part"]
            a["manufacturer"] = current[ref]["Manufacturer"]
            a["supplier_part"] = current[ref]["Supplier part"]
            a["reuse_part_identity_from"] = ref
    additions.append({"ref": "U_DRV_FAULT_FB", "mpn": "SN74AUP1T14DCKR",
                      "manufacturer": "Texas Instruments", "supplier_part": None,
                      "purpose": "CORE供电的低门限施密特反相回读；Y高表示共享许可/故障线低",
                      "pins": {"1": "NC", "2": "DRV_FAULT_N", "3": "GND_SYS",
                               "4": "FAULT_ANY_CORE_PA10", "5": "3V3_CORE"},
                      "temperature_contract_C": [-40, 85], "quote_status": "UNQUOTED"})
    additions.append({"ref": "C_DRV_FAULT_FB", "mpn": current["C_DRV_CSAREF"]["Manufacturer part"],
                      "manufacturer": current["C_DRV_CSAREF"]["Manufacturer"],
                      "supplier_part": current["C_DRV_CSAREF"]["Supplier part"],
                      "purpose": "新增100nF局部旁路；不得挪用REG1上AND门的去耦实例",
                      "pins": {"1": "3V3_CORE", "2": "GND_SYS"}})
    additions.append({"ref": "U_PERMIT_POR", "mpn": "TPS3808G30DBVR",
                      "manufacturer": "Texas Instruments", "supplier_part": None,
                      "quote_status": "UNQUOTED", "purpose": "CORE欠压/上电时硬件钳低许可，恢复后固定延时",
                      "pins": {"1": "MCU_PERMIT_AFE", "2": "SC_MINUS", "3": "3V3_CORE",
                               "4": "NC_CT_FIXED_20ms", "5": "3V3_CORE", "6": "3V3_CORE"}})
    existing_edits = [
        {"ref": "R_PB5_PERMIT", "new_mpn": "RC0603FR-074K7L", "new_value": "4.7kΩ",
         "pins": {"1": "PERMIT", "2": "MCU_PERMIT_AFE"}},
        {"ref": "U_SC_OV", "pins": {"1": "NC", "2": "SC_MINUS", "3": "SC_MINUS",
                                          "4": "SC_OV_DIV", "5": "3V3_CORE", "6": "MCU_PERMIT_AFE"}},
        {"ref": "C_SC_OV_VDD", "pins": {"1": "3V3_CORE", "2": "SC_MINUS"},
         "note": "仅在TLV与POR紧邻且同一电源参考时共用此既有100nF；否则另加本地旁路并重算+1件"},
    ]
    edit_mpn = {e["ref"]: e["new_mpn"] for e in existing_edits if "new_mpn" in e}
    assert all(e["ref"] in current and e["ref"] in native for e in existing_edits)
    assert len(additions) == 41
    assert len({a["ref"] for a in additions}) == len(additions)
    assert not ({a["ref"] for a in additions} & native)
    old_sampling = old["delete_components_exact_allowlist"]
    balance = [f"{prefix}{i}" for prefix in ["U_BAL", "R_BAL"] for i in range(1, 9)]
    bypass = [f"C_BAL{i}" for i in range(1, 9)]
    for ref in old_sampling + balance + bypass:
        assert ref in current and ref in native, ref
    stages = []
    for name, removals in [("AFE替代采样，暂保留BW", old_sampling),
                           ("合并均衡，保留单体端100nF", old_sampling + balance),
                           ("再取消仅服务BW的8只旁路", old_sampling + balance + bypass)]:
        remaining = [r for r in rows if r["Reference"] not in removals]
        before_mpn = {r["Manufacturer part"] for r in rows if r["Manufacturer part"]}
        after_mpn = {edit_mpn.get(r["Reference"], r["Manufacturer part"])
                     for r in remaining if r["Manufacturer part"]}
        after_mpn |= {a["mpn"] for a in additions}
        stages.append({"name": name, "status": "CONDITIONAL_NOT_IMPLEMENTED",
                       "removed_refs": removals, "removed_instances": len(removals),
                       "added_instances": len(additions),
                       "net_instance_delta": len(additions) - len(removals),
                       "new_mpn_count_vs_module": len(after_mpn - before_mpn),
                       "removed_mpn_count_vs_module": len(before_mpn - after_mpn),
                       "net_mpn_delta": len(after_mpn) - len(before_mpn)})
    result = {
        "date": "2026-10-28", "status": "DESIGN_SIMPLIFICATION_NOT_IMPLEMENTED_OR_QUALIFIED",
        "scope": "正常工作逐节监测、硬许可与内部均衡候选；不含0V初充实现或新增功率隔离MOS",
        "native_or_procurement_bom_modified": False,
        "source_binding": {"bom_sha256": digest(BOM), "enet_sha256": digest(ENET),
                           "old_candidate_sha256": digest(OLD)},
        "baseline_bom_rows": len(rows), "baseline_native_objects": len(native),
        "excluded_old_additions": sorted(excluded), "proposed_additions": additions,
        "existing_component_edits": existing_edits,
        "stages": stages,
        "counts_are_incomplete_budget_targets": True,
        "unpriced_unincluded_hardware": "已计入施密特故障回读2件和POR监督器1件；共用旁路的布局条件若不成立另+1件；功率隔离/快断/钳位未计，当前净数量不是最终落图结果",
        "pin_contract": {"fault_feedback": "DRV_FAULT_N直入SN74AUP1T14DCKR.A；VCC=3V3_CORE；Y接PA10/P1-12，FAULT高有效；新加100nF，不装旧1k回读电阻",
                         "i2c_candidate": "PB10/P2-5=SCL、PB11/P2-6=SDA；TEST网络删除后才可改用途",
                         "main_ntc_relocation": "NTC_MAIN由PB11/P2-6迁往已释放PA0/P2-20，保留原上拉/滤波/探头",
                         "permit": "PB5串阻换4.7k进入三输入AND许可节点；CORE供电的TLV6700和TPS3808开漏钳低同一节点；AND输出经680R至DRV_FAULT_N",
                         "cp1": "仅CPEN=0且CHG/DSG不用时，CP1直连BAT；不装C_AFE_CP1"},
        "hardware_only_review": {
            "firmware_implementation_out_of_scope": True,
            "gate_vcc_contract": "AND拟由AFE REG1供电，外部47k输入下拉与DRV原10k下拉保留；未获资格",
            "feedback_digital_high": {
                "status": "DIRECT_GPIO_REJECTED_BUFFER_SELECTED",
                "gate_voh_assumption_V": 2.4,
                "series_ohm_max": 680 * 1.02,
                "pulldown_ohm_min": 10000 * .98,
                "node_high_bound_V_ignoring_leakage": 2.4 * 9800 / (9800 + 680 * 1.02),
                "drv_vih_V": 2.2, "mcu_vih_at_3p3_V": .7 * 3.3,
                "mcu_vih_at_3p6_V": .7 * 3.6,
                "note": "串联1k不作电平转换；这个结论针对新AND许可拓扑，不等同于原PB5直驱拓扑的高电平预算"},
            "selected_feedback_buffer": {
                "status": "STATIC_INTERFACE_BOUNDS_PASS_WITH_STATED_RAIL_AND_TEMPERATURE_CONTRACT",
                "mpn": "SN74AUP1T14DCKR", "supply_V": [3.0, 3.6], "ambient_C": [-40, 85],
                "positive_threshold_max_V": 1.19, "negative_threshold_min_V": .5,
                "input_leakage_max_A": .5e-6, "poweroff_leakage_max_A": .5e-6,
                "input_high_lower_screen_V": 2.4 * 9800 / (9800 + 680 * 1.02) - .5e-6 * 680 * 1.02,
                "input_low_upper_screen_V": .4,
                "output_test_load_A": 20e-6, "mcu_input_leakage_bound_A": 4.5e-6,
                "output_high_min": "VCC-0.1V at <=20uA", "output_low_max_V": .1,
                "core_poweroff": "VCC=0时A/Y容许0..3.6V、Ioff<=0.5uA；不代表电源斜坡中的逻辑有效",
                "scope_limit": "只闭合读回接口；不把共享线失电电平、故障分类或MOS短路断流判为PASS；不得驱动额外负载后仍沿用20uA保证点"},
            "analog_feedback_option": {
                "status": "OPEN_NOT_SELECTED", "pin": "PA1/P2-17 ADC1_IN2",
                "low_example_V": .4,
                "note": "PA1是TT_a/3.6V容限，不能未经核实断言必反灌；仍需确认VDDA=0时模拟开关/输入边界。不能直接以该选项关闭失电问题"},
            "drv_fault_sink_screen": {
                "status": "STATIC_SCREEN_ONLY",
                "source_current_at_0p4V_A": (3.6 - .4) / (680 * .98),
                "drv_vol_test_current_A": .005,
                "note": "近5mA测试点，未覆盖瞬态、实际电源上限和其他源电流"},
            "total_ov_power_valid": {
                "status": "POWERUP_AND_CAPTURED_UV_TIMING_BOUNDS_PASS_FAST_DIPS_OPEN",
                "selected_supervisor": "TPS3808G30DBVR", "tlv_supply_new": "3V3_CORE relative to SC_MINUS",
                "nominal_negative_threshold_V": 2.79, "threshold_accuracy_fraction": .015,
                "maximum_hysteresis_fraction": .025,
                "maximum_release_threshold_screen_V": 2.79 * 1.015 * 1.025,
                "lowest_supervised_rail_example_V": 3.0 - .04,
                "release_delay_min_s": .012, "release_delay_typ_s": .020,
                "release_delay_max_s": .028, "tlv_start_max_s": .000450,
                "reset_low_screen_V_to_GND_SYS": .4 + .04,
                "mcu_sink_current_upper_screen_A": 3.6 / (4700 * .98),
                "shared_bypass_existing_ref": "C_SC_OV_VDD",
                "note": "CT悬空，不接地；不依赖G431仅典型250us的复位等待。CORE有效且12V_DRV跌落时TLV继续工作。SENSE短脉冲捕获仅有典型数据，不能宣称任意极短跌落必被捕获；该项仍需电源瞬态合同/实物验证"},
            "power_path_short_isolation": {
                "status": "FAIL_UNRESOLVED_IN_CURRENT_HARDWARE",
                "note": "QH D-S短路后撤销DRV许可不阻断电流；当前P-MOS体二极管允许BUS到桥侧，保险/串联隔离/电感断流能量必须联合收敛"},
            "conditional_power_replacement_not_counted": {
                "status": "TOPOLOGY_ONLY_CONDITIONAL_NOT_A_COMPLETE_PROTECTION_BOM",
                "placement": "F_SC后、SC_FUSED前的超容正端共漏双N；源极朝两端，漏极共接",
                "preferred_common_mpn": "HYG022N10NS1TA", "manufacturer": "HUAYI",
                "remove_old_group_if_entire_replacement_qualified": ["Q_BUS_P", "Q_PMOS_CTRL",
                    "R_PB6_BASE", "R_QP_BE", "R_QP_COL", "R_QP_GS", "D_QP_GS_ZENER"],
                "minimum_identified_positions_not_final": {"power_nmos": 2, "gate_100R": 2,
                    "gate_source_47k": 2, "cp1_470nF": 1,
                    "sr_input_100R": 2, "sr_differential_100nF": 1},
                "minimum_identified_positions_total": 10,
                "incomplete_group_delta_vs_7_old_positions": 3,
                "gate_resistors_return_to": "每只FET自己的源极；禁止回GND",
                "bq_current_candidate": "SRP由SC_MINUS、SRN由GND_SYS分别Kelvin引入；VSS仍GND_SYS；不得短接shunt",
                "no_fast_charge_trip_claim": "BQ SCD为放电保护，OCC最短10ms；正负两方向的快速判决/关断及源侧持续注能处理尚未选定",
                "no_self_locking_permission": "不得用跟随FET禁用状态的DCHG/DDSG回驱自身BOTHOFF而造成启动死锁；数字许可配置不可直接迁用",
                "fast_current_output_candidates": [
                    {"signal": "COMP1_OUT", "mcu_pin": "PA6", "header": "P2-14", "af": 8,
                     "released_by": "delete old T7", "polarity": "inverted; over-upper=low", "output": "open-drain"},
                    {"signal": "COMP2_OUT", "mcu_pin": "PA2", "header": "P2-18", "af": 8,
                     "released_by": "delete old T3", "polarity": "non-inverted; under-lower=low", "output": "open-drain"}],
                "conditional_latch_not_counted": {
                    "status": "INTERFACE_DIRECTION_ONLY_NOT_IMPLEMENTED_OR_TIMING_QUALIFIED",
                    "chain": "PB6 RUN via4.7k to CLR_REQ/47kPD; TLV/POR/COMP1/2 OD pull CLR_REQ low; 1G17->1G175.CLR",
                    "mpns": ["SN74LVC1G17DCKR", "SN74LVC1G175DCKR"],
                    "arm": "PB5 via1k to CLK with47kPD; D=CORE; Q has10kPD; all ICs CORE powered",
                    "output": "Q to AND permit and configured active-low BQ BOTHOFF; DCHG/DDSG only feed AND",
                    "no_auto_restart": "CLR fault clears Q; release alone holds Q=0 until new CLK rising edge",
                    "scope_limit": "不把1G175慢CLR输入、复位高阻COMP脚或31ns指定过驱指标当已完成全链路快断；双IC/局部去耦及电阻变更未计数"},
                "inductor_energy_screen_J": .5 * 20e-6 * 80 ** 2,
                "isolated_local_cap_min_F_example": 160e-6,
                "isolated_local_voltage_if_all_L_energy_enters_cap_V": (21.6 ** 2 + 20e-6 * 80 ** 2 / 160e-6) ** .5,
                "remaining": ["双方向快断的最大时延和峰值", "栅压/钳位与SOA",
                              "隔离后本地35V电容和双向续流/钳位", "源侧持续故障能量及保险协调"],
                "primary_sources": ["https://www.ti.com/lit/ds/symlink/bq76952.pdf",
                    "https://www.ti.com/lit/ug/sluuby2b/sluuby2b.pdf",
                    "https://atta.szlcsc.com/upload/public/pdf/source/20211129/85A27B49583C97E874E50AC12D8AF79A.pdf"]},
            "primary_sources": ["https://www.st.com/resource/en/datasheet/stm32g431cb.pdf",
                                "https://www.ti.com/lit/ds/symlink/sn74lvc1g11.pdf",
                                "https://www.ti.com/lit/ds/symlink/drv8161.pdf",
                                "https://www.ti.com/lit/ds/symlink/tlv6700.pdf",
                                "https://www.ti.com/lit/ds/symlink/sn74aup1t14.pdf",
                                "https://www.ti.com/lit/ds/symlink/tps3808.pdf"]},
        "balanced_current_illustration": {
            "not_a_guaranteed_bound": True, "cell_V": 2.45, "input_resistor_each_ohm": 100,
            "illustrative_internal_switch_ohm": 28,
            "on_current_mA": 2.45 / (200 + 28) * 1000,
            "old_120ohm_branch_ideal_mA": 2.45 / 120 * 1000,
            "additional_limits": "按真实物理单体避免相邻同时均衡，尤其物理第7/8节映射为通道7/16；计入测量暂停及温控占空损失"},
        "npn_power_screen": {
            "not_manufacturer_qualification": True, "stack_max_V": 21.6, "series_min_ohm": 990,
            "normal_load_A": 0.006, "regin_example_V": 5.5,
            "normal_q_power_W_ignoring_diode_drop": (21.6 - 6 - 5.5) * 0.006,
            "q_power_bound_W_if_regin_short_ideal_model": 21.6 ** 2 / (4 * 990),
            "resistor_short_power_bound_W_ideal_model": 21.6 ** 2 / 990,
            "exact_manufacturer_pdf": "../../元件封装库/厂商资料/JKSEMI_MMBT5551_C2972766.pdf",
            "exact_pdf_sha256": "e78e2d88fd77a3bae7f4ff91a55aefba1a48337ace9d46ef2bf71eba9537778c",
            "known_exact_part_ratings": {"VCEO_V": 160, "IC_A": .6, "PC_at_25C_W": .3,
                                         "RthetaJA_C_per_W": 416, "Tj_max_C": 150,
                                         "hFE_min_at_1mA_5V_25C": 80,
                                         "hFE_min_at_10mA_5V_25C": 100},
            "qualification_limit": "已取得JKSEMI自身手册，仍未给全温hFE和SOA；61/118mW只是电热筛选，不能批准替代"},
        "release_gates": ["逐节测量/开线/保护输出及失电默认低全部有效，方可删除旧采样和BW",
                          "内部均衡必须实际启用并核对低压范围、物理邻接、均衡速度及停机态；旧配置禁用，不能直接复用",
                          "删C_BAL前确认它们仅作已删除BW的旁路，没有额外线束EMI功能",
                          "MMBT5551按JKSEMI精确料号核定温度、SOA、hFE、启动/短路和SOT23脚序，禁止照搬FCX495脚号",
                          "I2C任一侧掉电反灌及REGIN<=BAT仍需闭合；禁止把外5V直接接REGIN覆盖0V",
                          "0V初充保持未实现/未获资格，普通PWM不能以精简为由越过逐节许可",
                          "总压后备、驱动保护、电流快关断、保险、磁件/电容与有效泄放路径不在本轮删减清单"],
        "cost_status": "完整新旧BOM含MOQ成本未核定，不将删件数量换算为已实现降本"}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "stages": [{k: v for k, v in s.items() if k != "removed_refs"} for s in stages]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
