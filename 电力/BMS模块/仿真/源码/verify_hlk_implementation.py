#!/usr/bin/env python3
"""Recompute the isolated Hi-Link power candidate checks from local source files.

This is a Python reduced-order arithmetic/RC model, not a SPICE converter model
and not an electrical or hardware acceptance test. Run from any working dir:
    python -B BMS模块/仿真/源码/verify_hlk_implementation.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BMS = ROOT / "BMS模块"
INPUT = BMS / "仿真" / "输入" / "HLK_implementation_cases.json"
RESULT = BMS / "仿真" / "结果" / "HLK_implementation_model.json"
CHECK = BMS / "检查" / "海凌科电源落实验证.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def roundn(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["ref"]: record for record in records if isinstance(record, dict) and "ref" in record}


def pins_equal(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return {str(k): v for k, v in actual.items()} == {str(k): v for k, v in expected.items()}


def main() -> int:
    case = load_json(INPUT)
    source_relpaths = [
        "BMS模块/项目要求.md",
        "BMS模块/检查/最大料号归并候选.json",
        "BMS模块/检查/URB4805替换候选.json",
        "BMS模块/检查/5V电源预算.md",
        "BMS模块/检查/风扇调速支持审查.json",
        "BMS模块/原理图准备/接口元件连接.json",
        "BMS模块/原理图准备/功率元件连接.json",
        "BMS模块/原理图准备/风扇元件连接.json",
        "BMS模块/原理图准备/AFE元件连接.json",
        "BMS模块/原理图准备/BOM.csv",
        "BMS模块/检查/当前BOM计数证据.json",
        "稳压模块/参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf",
        "元件封装库/资产索引.json",
    ]
    sources = {rel: ROOT / Path(rel) for rel in source_relpaths}
    missing_sources = [rel for rel, path in sources.items() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"缺少输入源: {missing_sources}")

    interface = load_json(sources["BMS模块/原理图准备/接口元件连接.json"])
    power = load_json(sources["BMS模块/原理图准备/功率元件连接.json"])
    fan = load_json(sources["BMS模块/原理图准备/风扇元件连接.json"])
    afe = load_json(sources["BMS模块/原理图准备/AFE元件连接.json"])
    interface_by_ref = source_map(interface)
    power_by_ref = source_map(power)
    afe_by_ref = source_map(afe)
    current_by_ref = {**interface_by_ref, **power_by_ref, **source_map(fan), **afe_by_ref}

    with sources["BMS模块/原理图准备/BOM.csv"].open(encoding="utf-8-sig", newline="") as stream:
        bom_rows = list(csv.DictReader(stream))
    bom_by_ref = {row.get("位号", ""): row for row in bom_rows if row.get("位号")}

    module = case["official_module_spec"]
    load = case["minimum_load_bank"]
    switch = case["switch"]
    control = case["control_rail_merge"]
    implementation = case["implemented_changes"]
    caps = case["capacitors"]
    load_cases = case["load_cases"]

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str, status: str = "PASS") -> None:
        if status == "PASS" and not passed:
            status = "FAIL"
        checks.append({"item": name, "status": status, "passed": bool(passed), "evidence": evidence})

    # Snapshot/provenance and source truth checks.
    official_pdf = sources["稳压模块/参考资料/海凌科_URAB_YMD-10WR3_V1.7.pdf"]
    pdf_sha = sha256(official_pdf)
    expected_pdf_sha = "1c55e95b4c6457f75984519c61ba2c544b15f9e65016e8ebf4cc286b545ff49d"
    check("海凌科官方V1.7资料哈希", pdf_sha == expected_pdf_sha,
          f"{relative(official_pdf)} SHA256={pdf_sha}; compared with the cached official-PDF digest")

    # Re-open the checked pre-implementation source bundle. This lets this
    # verifier distinguish a completed implementation from an unmodified
    # candidate baseline without guessing from a stale report.
    baseline_archive = ROOT / Path(implementation["baseline_archive"])
    if baseline_archive.is_file():
        baseline_bytes = baseline_archive.read_bytes()
        baseline_location = relative(baseline_archive)
    else:
        history_archive = BMS / "归档" / "历史与调试.zip"
        with zipfile.ZipFile(history_archive) as history:
            entries = [n for n in history.namelist() if n.replace('\\', '/').endswith('/source_before_implementation.zip')]
            if len(entries) != 1:
                raise RuntimeError(f"Expected exactly one archived implementation baseline, got {len(entries)}")
            baseline_bytes = history.read(entries[0])
        baseline_location = relative(history_archive) + '!' + entries[0]
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
    baseline_expected_sha = "2995779953965e11a21cfa504db869e76dd66b49f271da9d053086dbce88c9a1"
    check("实现前连接源归档哈希", baseline_sha == baseline_expected_sha,
          f"{baseline_location} SHA256={baseline_sha}")
    with zipfile.ZipFile(io.BytesIO(baseline_bytes)) as archive:
        baseline_interface_name = next(n for n in archive.namelist() if n.endswith("接口元件连接.json"))
        baseline_power_name = next(n for n in archive.namelist() if n.endswith("功率元件连接.json"))
        baseline_interface = source_map(json.loads(archive.read(baseline_interface_name)))
        baseline_power = source_map(json.loads(archive.read(baseline_power_name)))

    expected_old_switch = switch["current_pin_map"]
    baseline_switch = baseline_interface.get(switch["ref"], {})
    check("归档基线DPDT针图与原始记录一致",
          pins_equal(baseline_switch.get("pins", {}), expected_old_switch),
          f"baseline {switch['ref']} pins={baseline_switch.get('pins')}")
    current_switch = interface_by_ref.get(switch["ref"], {})
    expected_switch = implementation["implemented_switch_pin_map"]
    check("当前DPDT正式连接源符合冻结针图",
          pins_equal(current_switch.get("pins", {}), expected_switch),
          f"{switch['ref']} current pins={current_switch.get('pins')}")

    baseline_module = baseline_interface.get(implementation["replace_in_place"]["ref"], {})
    module_ref = implementation["implemented_module_ref"]
    current_module = interface_by_ref.get(module_ref, {})
    expected_module_pins = case["expected_module_pin_map"]
    module_pin_check = all(
        current_module.get("pins", {}).get(str(pin)) == net.split(" (", 1)[0]
        for pin, net in expected_module_pins.items() if net != "NC (NP)"
    ) and "4" not in current_module.get("pins", {})
    check("归档基线含原SCT2A26STER且正式源已更换为海凌科模块",
          baseline_module.get("mpn") == implementation["replace_in_place"]["from_mpn"]
          and current_module.get("mpn") == implementation["replace_in_place"]["to_mpn"]
          and current_module.get("libUuid") == case["baseline"]["official_library_module"]["library_uuid"]
          and module_pin_check,
          f"baseline U_PSU={baseline_module.get('mpn')}; current {module_ref}={current_module.get('mpn')}, libUuid={current_module.get('libUuid')}, pins={current_module.get('pins')}")

    # Check that the archived pair really was redundant, and that the current
    # source retains exactly one pull-down on the merged rail.
    r_if = baseline_interface.get(control["remove_duplicate_pulldown_ref"], {})
    r_pre_old = baseline_interface.get(control["retain_pulldown_ref"], {})
    r_pre = interface_by_ref.get(control["retain_pulldown_ref"], {})
    check("归档基线两只控制下拉合并后同值同地且正式源保留一只",
          r_if.get("mpn") == control["pulldown_mpn"]
          and r_pre_old.get("mpn") == control["pulldown_mpn"]
          and r_if.get("pins", {}).get("1") == "SW_RUN"
          and r_pre_old.get("pins", {}).get("1") == "PRE_CTRL_3V3"
          and r_if.get("pins", {}).get("2") == "B0"
          and r_pre_old.get("pins", {}).get("2") == "B0"
          and r_pre.get("mpn") == control["pulldown_mpn"]
          and r_pre.get("pins", {}).get("1") == "PRE_CTRL_3V3"
          and r_pre.get("pins", {}).get("2") == "B0"
          and control["remove_duplicate_pulldown_ref"] not in current_by_ref,
          f"baseline R_IF_RUN_PD={r_if.get('pins')}; baseline R_PRE_CTRL_PD={r_pre_old.get('pins')}; current R_PRE_CTRL_PD={r_pre.get('pins')}; duplicate absent={control['remove_duplicate_pulldown_ref'] not in current_by_ref}")
    u_gate = power_by_ref.get("U_GATE1", {})
    check("U_GATE1合并后pin1与VCC同接PRE_CTRL_3V3",
          u_gate.get("pins", {}).get("1") == "PRE_CTRL_3V3"
          and u_gate.get("pins", {}).get("5") == "PRE_CTRL_3V3",
          f"U_GATE1 A(pin1)={u_gate.get('pins', {}).get('1')}, VCC(pin5)={u_gate.get('pins', {}).get('5')}; RUN时为高，STORE掉电输入注入与Y状态仍待验证")

    # Confirm removals against the archived source and the saved current source.
    remove_refs = list(implementation["remove_refs"])
    for ref in remove_refs:
        baseline_present = ref in baseline_interface or ref in baseline_power
        current_present = ref in current_by_ref
        check(f"正式源已移除舊位号：{ref}", baseline_present and not current_present,
              f"baseline_present={baseline_present}; current_present={current_present}")

    added_refs = implementation["add_refs"]
    added_rows = [current_by_ref.get(ref, {}) for ref in added_refs]
    expected_load_pins = {"1": load["connection"]["pin1"], "2": load["connection"]["pin2"]}
    added_ok = all(row.get("mpn") == load["mpn"]
                   and row.get("pins") == expected_load_pins
                   and row.get("libUuid") == "14449b7af63c4922982703e48a4b341b"
                   for row in added_rows)
    check("正式源含7只准确MPN和并联网络的最低负载",
          len(added_rows) == 7 and added_ok,
          f"refs={added_refs}; all_match={added_ok}; source quantities={len([r for r in added_rows if r])}")

    retained_cap_map = {**interface_by_ref, **power_by_ref, **source_map(fan)}
    input_caps_ok = all(
        ref in retained_cap_map
        and retained_cap_map[ref].get("pins", {}).get("1") == "CELL_B9"
        and retained_cap_map[ref].get("pins", {}).get("2") == "B0"
        for ref in caps["module_input_refs"]
    )
    output_caps_ok = all(
        ref in retained_cap_map
        and retained_cap_map[ref].get("pins", {}).get("1") == "VOUT_5V"
        and retained_cap_map[ref].get("pins", {}).get("2") == "B0"
        for ref in caps["module_output_refs"]
    )
    check("保留输入电容仍跨CELL_B9/B0", input_caps_ok,
          ",".join(caps["module_input_refs"]))
    check("保留输出电容仍跨VOUT_5V/B0", output_caps_ok,
          ",".join(caps["module_output_refs"]))

    afe_u1 = afe_by_ref.get("U1", {})
    afe_q = afe_by_ref.get("Q_BREG", {})
    supply_ok = (
        afe_u1.get("pins", {}).get("37") == "BREG"
        and afe_u1.get("pins", {}).get("36") == "REGIN"
        and afe_u1.get("pins", {}).get("35") == "REG1"
        and afe_q.get("pins", {}).get("1") == "BREG"
        and afe_q.get("pins", {}).get("2") == "BREG_COLLECTOR"
        and afe_q.get("pins", {}).get("3") == "REGIN"
    )
    check("BREG/REGIN/REG1链与5V模块输入分开", supply_ok,
          f"U1 BREG/REGIN/REG1 pins={[(37, afe_u1.get('pins', {}).get('37')), (36, afe_u1.get('pins', {}).get('36')), (35, afe_u1.get('pins', {}).get('35'))]}; Q_BREG={afe_q.get('pins')}")

    library_index = load_json(sources["元件封装库/资产索引.json"])
    resistor_device_assets = [
        asset for asset in library_index.get("assets", [])
        if asset.get("type") == "DEVICE" and load["mpn"] in asset.get("titles", [])
    ]
    resistor_mpn_in_library = any(
        asset.get("current_source") is True and bool(asset.get("library_uuids"))
        for asset in resistor_device_assets
    )
    check("510Ω精确MPN可在项目共用库索引命中", resistor_mpn_in_library,
          f"current DEVICE assets={len(resistor_device_assets)}; current common-library-backed asset={resistor_mpn_in_library}")
    existing_same_mpn = bom_by_ref.get(load["existing_same_mpn_ref"], {})
    check("510Ω复用MPN已出现在现行采购BOM", existing_same_mpn.get("MPN") == load["mpn"],
          f"{load['existing_same_mpn_ref']} MPN={existing_same_mpn.get('MPN')}")

    # Minimum-load resistor corners.
    r_nom = float(load["nominal_ohm_each"])
    r_tol = float(load["tolerance_fraction"])
    alpha = float(load["tcr_ppm_per_c"]) * 1e-6
    t_min, t_max = map(float, load["temperature_range_for_resistor_c"])
    tref = 25.0
    n_load = len(load["refs"])
    r_max = r_nom * (1 + r_tol) * (1 + alpha * (t_max - tref))
    r_min = r_nom * (1 - r_tol) * (1 + alpha * (t_min - tref))
    v = load_cases["voltage_corners_v"]
    i_min_accurate_low = n_load * v["official_accuracy_low"] / r_max
    i_min_extended_low = n_load * v["extended_engineering_stress_low"] / r_max
    i_nominal_ideal = n_load * v["nominal"] / r_nom
    i_nominal_worst_r = n_load * v["nominal"] / r_max
    i_max_extended = n_load * v["extended_engineering_stress_high"] / r_min
    p_each_max = v["extended_engineering_stress_high"] ** 2 / r_min
    p_bank_nom = n_load * v["nominal"] ** 2 / r_nom
    p_bank_max = n_load * v["extended_engineering_stress_high"] ** 2 / r_min
    i_required = float(module["output_current_rated_a"]) * float(module["minimum_load_fraction_for_capacitive_load"])
    check("7×510Ω并联在官方输出精度下限仍超过3%最低负载",
          i_min_accurate_low > i_required,
          f"Imin={i_min_accurate_low*1000:.3f}mA, requirement={i_required*1000:.3f}mA")

    min_load = {
        "mpn": load["mpn"],
        "refs": load["refs"],
        "quantity": n_load,
        "each_nominal_ohm": r_nom,
        "parallel_nominal_ohm": roundn(r_nom / n_load, 6),
        "min_resistance_each_corner_ohm": roundn(r_min, 6),
        "max_resistance_each_corner_ohm": roundn(r_max, 6),
        "vendor_minimum_load_current_ma": roundn(i_required * 1000, 4),
        "minimum_current_at_datasheet_5v_minus_2pct_ma": roundn(i_min_accurate_low * 1000, 4),
        "minimum_margin_at_datasheet_5v_minus_2pct_ma": roundn((i_min_accurate_low-i_required)*1000, 4),
        "minimum_current_at_extended_4p75v_stress_ma": roundn(i_min_extended_low * 1000, 4),
        "ideal_nominal_current_at_5v_ma": roundn(i_nominal_ideal * 1000, 4),
        "worst_resistance_nominal_current_at_5v_ma": roundn(i_nominal_worst_r * 1000, 4),
        "maximum_current_at_5p25v_stress_ma": roundn(i_max_extended * 1000, 4),
        "ideal_bank_power_at_5v_w": roundn(p_bank_nom, 6),
        "maximum_bank_power_at_5p25v_stress_w": roundn(p_bank_max, 6),
        "maximum_each_resistor_power_at_5p25v_stress_mw": roundn(p_each_max*1000, 4),
        "each_resistor_power_rating_mw": roundn(float(load["rated_power_w_each"])*1000, 4),
        "maximum_each_resistor_fraction_of_rated_power": roundn(p_each_max/float(load["rated_power_w_each"]), 5),
        "assumptions": "电阻±1%、±100ppm/°C，温度按0–60°C；未建模自热/铜箔/风道；5.25V为扩展应力边界，不是手册输出精度值。"
    }
    # One 75-ohm resistor is an arithmetic-only comparator, not a selected part.
    r75_nom = 75.0
    r75_max = r75_nom*(1+0.01)*(1+alpha*(t_max-tref))
    r75_min = r75_nom*(1-0.01)*(1+alpha*(t_min-tref))
    i75_min_official = float(v["official_accuracy_low"])/r75_max
    i75_min_extended = float(v["extended_engineering_stress_low"])/r75_max
    i75_max_extended = float(v["extended_engineering_stress_high"])/r75_min
    p75_max_extended = float(v["extended_engineering_stress_high"])**2/r75_min
    min_load["single_75_ohm_comparison_only"] = {
        "quantity": 1,
        "selected": False,
        "mpn": None,
        "new_mpn_type_needed": True,
        "min_current_at_4p9v_ma": roundn(i75_min_official*1000, 4),
        "min_margin_at_4p9v_ma": roundn((i75_min_official-i_required)*1000, 4),
        "min_current_at_4p75v_stress_ma": roundn(i75_min_extended*1000, 4),
        "min_margin_at_4p75v_stress_ma": roundn((i75_min_extended-i_required)*1000, 4),
        "max_current_at_5p25v_stress_ma": roundn(i75_max_extended*1000, 4),
        "max_power_at_5p25v_stress_w": roundn(p75_max_extended, 6),
        "tradeoff": "少6个焊点，但需新增一种75Ω功率电阻MPN；较7×510方案在4.75V角落最低负载电流裕量更小，实际选型还要核额定功率/温度降额。"
    }
    min_load["no_additional_bleeder"] = {
        "selected": False,
        "reason": "外置MCU/风扇等负载最小耗流没有厂家保证值，不能证明所有稳态/待机/掉线条件下由其它负载满足60mA最低负载。"
    }

    # Output bank capacitance, module output load envelopes and efficiency extrapolation.
    output_cap_uf = float(caps["module_output_nominal_total_uf"])
    output_cap_ratio = output_cap_uf / float(module["maximum_capacitive_load_uf"])
    base_a = float(load_cases["base_5v_planning_current_a"])
    fan_candidate = load_cases["fan_candidate"]
    scenarios: list[dict[str, Any]] = []
    for fan_a, label in [
        (fan_candidate["typical_current_a"], "Delta候选典型电流；候选未冻结"),
        (fan_candidate["label_max_current_a"], "Delta候选铭牌最大电流；启动电流未知"),
        *[(float(row["fan_current_a"]), f"旧预算{row['name']}压力值；不是所选风扇实测") for row in load_cases["fan_pressure_cases_from_legacy_budget"]],
    ]:
        total_a = base_a + fan_a + i_max_extended
        pout_w = total_a * float(v["extended_engineering_stress_high"])
        eta = float(module["efficiency_min_fraction_at_48v_rated_resistive_25c"])
        pin_w = pout_w / eta
        loss_w = pin_w - pout_w
        scenarios.append({
            "label": label,
            "base_planning_current_a": base_a,
            "fan_current_assumption_a": fan_a,
            "minimum_load_worst_corner_current_a": roundn(i_max_extended, 6),
            "total_current_a": roundn(total_a, 6),
            "current_margin_to_2a_rated_a": roundn(float(module["output_current_rated_a"])-total_a, 6),
            "output_power_at_5p25v_stress_w": roundn(pout_w, 6),
            "output_power_fraction_of_10w_rated": roundn(pout_w/float(module["output_power_rated_w"]), 6),
            "input_power_at_81pct_48v_rated_test_efficiency_extrapolated_w": roundn(pin_w, 6),
            "conversion_loss_at_81pct_extrapolated_w": roundn(loss_w, 6),
            "input_current_at_22p5v_extrapolated_a": roundn(pin_w/22.5, 6),
            "input_current_at_37p8v_extrapolated_a": roundn(pin_w/37.8, 6),
            "effective_case_to_ambient_theta_limit_at_60c_for_100c_case_c_per_w": roundn((float(module["maximum_case_temperature_c"])-60.0)/loss_w, 4),
            "efficiency_status": "81%为官方表格在48V标称输入、额定阻性负载、25°C条件下的下限；对本负载及22.5–37.8V计算仅为保守算术外推，非保证。"
        })

    delta_max_scenario = scenarios[1]
    legacy_limit_scenario = scenarios[-1]
    thermal_curve_status = (
        delta_max_scenario["output_power_at_5p25v_stress_w"] < float(module["output_power_rated_w"])
        and legacy_limit_scenario["output_power_at_5p25v_stress_w"] < float(module["output_power_rated_w"])
        and 60.0 <= float(module["temperature_derating_curve"]["full_power_through_ambient_c"])
    )
    check("纸面输出功率及温度曲线读图边界", thermal_curve_status,
          f"候选风扇铭牌最大支路{delta_max_scenario['output_power_at_5p25v_stress_w']}W；旧风扇限流压力{legacy_limit_scenario['output_power_at_5p25v_stress_w']}W；V1.7曲线60°C读图100%功率；仅算术/读图，不代表板级热验收")

    output_cap_check = output_cap_uf <= float(module["maximum_capacitive_load_uf"])
    check("188uF输出标称容量低于2200uF上限", output_cap_check,
          f"nominal={output_cap_uf}uF; datasheet max={module['maximum_capacitive_load_uf']}uF")

    power = {
        "base_load_planning_current_a": base_a,
        "load_envelopes": scenarios,
        "separate_fan_interpretation": {
            "selected_fan_actual_rating": "尚未冻结",
            "Delta_candidate_max_current_a": fan_candidate["label_max_current_a"],
            "Delta_candidate_startup_current": "未见型号资料保证值，不以0.30A铭牌最大值等同启动峰值",
            "legacy_pressure_1p1a_fan_plus_base_and_bleed_total_a": roundn(base_a + 1.1 + i_max_extended, 6),
            "warning": "1.1A仅为旧预算风扇支路电流限流压力，不是候选风扇电流；总量显式另加0.30A基础规划与约0.073A最低负载。"
        },
        "input_power_loss_heat": {
            "minimum_load_current_is_added_separately": True,
            "min_efficiency_used": module["efficiency_min_fraction_at_48v_rated_resistive_25c"],
            "test_condition": "48V标称输入、额定阻性负载、25°C",
            "22p5v_high_pressure_input_current_extrapolated_a": legacy_limit_scenario["input_current_at_22p5v_extrapolated_a"],
            "22p5v_high_pressure_conversion_loss_extrapolated_w": legacy_limit_scenario["conversion_loss_at_81pct_extrapolated_w"],
            "required_effective_theta_at_60c_ambient_for_case_below_100c_c_per_w": legacy_limit_scenario["effective_case_to_ambient_theta_limit_at_60c_for_100c_case_c_per_w"],
            "temperature_derating_curve_reading": module["temperature_derating_curve"],
            "conclusion": "10W及曲线容量的纸面比较未越界，但效率外推、有效热阻、壳温、封闭舱气流和风扇失效没有测量，因此热验收仍OPEN。"
        },
        "capacitors": {
            "input_nominal_uf": caps["module_input_nominal_uf"],
            "input_test_circuit_reference_uf": module["page_6_test_circuit_input_capacitor_reference_uf"],
            "input_nominal_fraction_of_47uf_reference": roundn(caps["module_input_nominal_uf"]/47.0, 5),
            "input_capacitance_status": "OPEN: 9.5uF标称陶瓷小于47–100uF典型测试电路范围；该范围不是手册保证最小值。输入偏压有效容量、纹波/动态阻抗/PI滤波器稳定性未闭合。",
            "output_nominal_uf": output_cap_uf,
            "output_cap_limit_uf": module["maximum_capacitive_load_uf"],
            "output_cap_fraction_of_limit": roundn(output_cap_ratio, 6),
            "output_effective_capacitance_status": "名义4×47uF，低于上限；X5R在5V偏压与温度后的有效容量和模块兼容性仍待确认。"
        }
    }

    # Capacitor-start current sensitivity: no active converter model is inferred.
    startup_rows = []
    fan_start_current = 0.80  # only the legacy design pressure value
    bleed_high_current = i_max_extended
    static_start_a = base_a + fan_start_current + bleed_high_current
    for c_uf in case["simulation_method"]["startup_capacitance_scenarios_uf"]:
        for ramp_ms in case["simulation_method"]["startup_sensitivity_ramp_ms"]:
            cap_a = (float(c_uf)*1e-6 * float(module["output_nominal_v"])) / (float(ramp_ms)*1e-3)
            startup_rows.append({
                "effective_capacitance_scenario_uf": c_uf,
                "assumed_linear_voltage_ramp_ms": ramp_ms,
                "capacitor_charge_current_a": roundn(cap_a, 6),
                "static_load_base_plus_legacy_startup_fan_plus_bleed_a": roundn(static_start_a, 6),
                "sum_sensitivity_current_a": roundn(static_start_a+cap_a, 6),
                "comparison_to_2a_rating": "ABOVE" if static_start_a+cap_a > float(module["output_current_rated_a"]) else "BELOW",
                "model_scope": "I=C*dV/dt sensitivity only; converter ramp/current limit/foldback/startup response is unknown"
            })

    # Bleeder-only decay of 5 V output after CTRL OFF.
    r_bank_min_v, r_bank_max_v = r_min/n_load, r_max/n_load
    r_bank_nominal = r_nom/n_load
    discharge = {
        "nominal_capacitance_uf": output_cap_uf,
        "nominal_parallel_resistance_ohm": roundn(r_bank_nominal, 6),
        "minimum_resistance_corner_ohm": roundn(r_bank_min_v, 6),
        "maximum_resistance_corner_ohm": roundn(r_bank_max_v, 6),
        "nominal_time_constant_ms": roundn(r_bank_nominal*output_cap_uf*1e-3, 6),
        "nominal_time_to_10pct_ms": roundn(r_bank_nominal*output_cap_uf*1e-3*math.log(10), 6),
        "nominal_time_to_1pct_ms": roundn(r_bank_nominal*output_cap_uf*1e-3*math.log(100), 6),
        "nominal_energy_at_5v_mj": roundn(0.5*output_cap_uf*1e-6*float(module["output_nominal_v"])**2*1000, 6),
        "voltage_samples": []
    }
    tau_sec = r_bank_nominal*output_cap_uf*1e-6
    for t_ms in [0, 1, 5, 10, 20, 50, 100]:
        discharge["voltage_samples"].append({
            "time_ms": t_ms,
            "voltage_from_5v_initial_v": roundn(float(module["output_nominal_v"])*math.exp(-(t_ms/1000)/tau_sec), 6)
        })
    discharge["interpretation"] = "仅按7×510Ω与188uF标称值的无其他负载RC衰减；实际电容量、其他负载、模块停机输出行为会改变结果；CTRL OFF不等于瞬时零电压。"

    # REG1/PRE_CTRL passive control-rail discharge before and after deleting duplicate pulldown.
    c_ctrl_nom_f = len(control["control_decoupling_refs"])*float(control["each_control_decoupling_nominal_uf"])*1e-6
    c_tol = float(control["decoupling_tolerance_fraction"])
    c_ctrl_min_f, c_ctrl_max_f = c_ctrl_nom_f*(1-c_tol), c_ctrl_nom_f*(1+c_tol)
    rpd = float(control["pulldown_nominal_ohm"])
    rtol = float(control["pulldown_tolerance_fraction"])
    tcr = float(control["pulldown_tcr_ppm_per_c"])*1e-6
    rpd_min = rpd*(1-rtol)*(1+tcr*(t_min-tref))
    rpd_max = rpd*(1+rtol)*(1+tcr*(t_max-tref))
    tau_post_min = rpd_min*c_ctrl_min_f
    tau_post_max = rpd_max*c_ctrl_max_f
    tau_pre_min = (rpd_min/2)*c_ctrl_min_f
    tau_pre_max = (rpd_max/2)*c_ctrl_max_f
    control_decay = {
        "control_rail": "PRE_CTRL_3V3 after switch pole 1 disconnects REG1 in STORE",
        "initial_voltage_v": 3.3,
        "nominal_decoupling_uf": roundn(c_ctrl_nom_f*1e6, 6),
        "decoupling_tolerance_range_uf": [roundn(c_ctrl_min_f*1e6, 6), roundn(c_ctrl_max_f*1e6, 6)],
        "before_merge_two_100k_in_parallel": {
            "nominal_tau_ms": roundn((rpd/2)*c_ctrl_nom_f*1000, 6),
            "tau_corner_range_ms": [roundn(tau_pre_min*1000, 6), roundn(tau_pre_max*1000, 6)],
            "time_to_10pct_corner_range_ms": [roundn(tau_pre_min*math.log(10)*1000, 6), roundn(tau_pre_max*math.log(10)*1000, 6)],
            "time_to_1pct_corner_range_ms": [roundn(tau_pre_min*math.log(100)*1000, 6), roundn(tau_pre_max*math.log(100)*1000, 6)]
        },
        "after_merge_one_100k_retained": {
            "nominal_tau_ms": roundn(rpd*c_ctrl_nom_f*1000, 6),
            "tau_corner_range_ms": [roundn(tau_post_min*1000, 6), roundn(tau_post_max*1000, 6)],
            "time_to_10pct_corner_range_ms": [roundn(tau_post_min*math.log(10)*1000, 6), roundn(tau_post_max*math.log(10)*1000, 6)],
            "time_to_1pct_corner_range_ms": [roundn(tau_post_min*math.log(100)*1000, 6), roundn(tau_post_max*math.log(100)*1000, 6)]
        },
        "nominal_discharge_samples_after_merge": [],
        "scope_warning": "被动RC、不含芯片负载/输入漏电/REG1软启动/开关弹跳/阈值；实际POR保持U_POR，不能由RC时间替代POR或故障关断证明。"
    }
    tau_ctrl_nom = rpd*c_ctrl_nom_f
    for t_ms in [0, 1, 10, 40, 80, 100, 160, 200]:
        control_decay["nominal_discharge_samples_after_merge"].append({
            "time_ms": t_ms,
            "voltage_v": roundn(3.3*math.exp(-(t_ms/1000)/tau_ctrl_nom), 6)
        })

    # Check the current procurement BOM at its own scope. PCB-mounted counts
    # are managed in the separate current BOM count evidence and are not
    # projected from this power-only subset.
    bom_sha = sha256(sources["BMS模块/原理图准备/BOM.csv"])
    bom_count_evidence = load_json(sources["BMS模块/检查/当前BOM计数证据.json"])
    count_values = bom_count_evidence["counts"]
    target_module_rows = [row for row in bom_rows if row.get("MPN", "").strip() == module["model"]]
    remove_refs_still_in_bom = sorted(set(remove_refs) & set(bom_by_ref))
    missing_added_refs = sorted(set(added_refs) - set(bom_by_ref))
    quantity_field = "总数量"
    added_bom_rows_ok = all(
        bom_by_ref.get(ref, {}).get("MPN", "").strip() == load["mpn"]
        and int(bom_by_ref.get(ref, {}).get(quantity_field, "0") or 0) == 1
        for ref in added_refs
    )
    bom_current_implementation = (
        not remove_refs_still_in_bom
        and not missing_added_refs
        and added_bom_rows_ok
        and len(target_module_rows) == 1
    )
    check("采购BOM同步到正式模块/删除/最低负载位号",
          bom_current_implementation,
          f"current BOM SHA256={bom_sha}; target module rows={len(target_module_rows)}; removed refs still present={remove_refs_still_in_bom}; missing added refs={missing_added_refs}; all added row MPN/qty correct={added_bom_rows_ok}",
          status="OPEN" if not bom_current_implementation else "PASS")

    observed_qty = sum(int(row.get(quantity_field, "0") or 0) for row in bom_rows)
    observed_mpn = {row.get("MPN", "").strip() for row in bom_rows if row.get("MPN", "").strip()}
    counts_evidence_matches = (
        bom_count_evidence.get("sourceBOM") == relative(sources["BMS模块/原理图准备/BOM.csv"])
        and bom_count_evidence.get("bomSha256") == bom_sha
        and int(count_values["uniqueRefRows"]) == len(bom_rows)
        and int(count_values["perSetTotalQty"]) == observed_qty
        and int(count_values["rawNonemptyMpnTextVariants"]) == len(observed_mpn)
    )
    count_plan = {
        "procurement_bom_scope": {
            "rows_observed_in_current_bom": len(bom_rows),
            "per_set_quantity_observed_in_current_bom": observed_qty,
            "raw_mpn_text_variants_observed_in_current_bom": len(observed_mpn),
            "normalized_mpn_types_including_core_board_from_bound_count_evidence": int(count_values["normalizedMpnTypesIncludingCoreBoard"]),
            "normalized_selected_mpn_types_excluding_core_board_from_bound_count_evidence": int(count_values["normalizedSelectedMpnTypesExcludingCoreBoard"]),
            "count_evidence_sha256": sha256(sources["BMS模块/检查/当前BOM计数证据.json"]),
            "bom_sha256": bom_sha,
            "scope": "当前全采购BOM含板外风扇、配套、插头和线束；与PCB实装统计分开。规范化料号数来自与此BOM SHA绑定的当前计数证据；不从电源局部变化推导全板数量。"
        }
    }
    comparison_75 = min_load["single_75_ohm_comparison_only"]
    # The 75-ohm alternative is retained only as an electrical comparison.
    # Do not project whole-board or procurement counts from a stale, narrower
    # snapshot: the BOM also contains independently updated off-board items.
    for key in (
        "pcb_parts_if_substituted_for_seven_resistors",
        "pcb_normalized_mpn_types_if_substituted",
        "procurement_rows_if_substituted",
        "procurement_total_quantity_if_substituted",
        "procurement_unique_mpn_types_if_substituted",
    ):
        comparison_75.pop(key, None)
    check("当前全采购BOM计数证据与BOM哈希一致",
          counts_evidence_matches and bom_current_implementation,
          f"BOM SHA256={bom_sha}; evidence SHA256={count_plan['procurement_bom_scope']['count_evidence_sha256']}; rows={len(bom_rows)}; qty={observed_qty}; raw MPN variants={len(observed_mpn)}; normalized MPN types incl. core board={count_values['normalizedMpnTypesIncludingCoreBoard']}; selected excluding core board={count_values['normalizedSelectedMpnTypesExcludingCoreBoard']}; evidence bound and local module implementation={counts_evidence_matches and bom_current_implementation}")

    state_analysis = {
        "U_GATE1_logic_after_SW_RUN_net_merge": {
            "device": "TI SN74LVC1G11DCKR; pin1=A, pin5=VCC, pin4=Y",
            "planned_merged_net": "SW_RUN renamed/merged into PRE_CTRL_3V3; U_GATE1 pin1 A and pin5 VCC share the switched rail",
            "RUN_equivalence": "both original SW_RUN and PRE_CTRL_3V3 were connected to REG1 in RUN; A is high whenever VCC is powered, so the original A&B&C function reduces to B&C during RUN",
            "STORE_limit": "gate VCC is off; B/C input injection, Y output state and any off-domain backfeed are not guaranteed by this reduced model and require exact TI limits plus board measurement"
        },
        "RUN": {
            "pole1": "2-3闭合，REG1接PRE_CTRL_3V3",
            "pole2": "5-6闭合，pin6 NC，因此CTRL_MODULE开路并开启5V模块",
            "if_REG1_or_BREG_lost_while_CELL_B9_present": "PRE_CTRL_3V3及本地控制逻辑失电/下拉；CTRL仍开路，5V模块仍可能开启。不能据此声称全板失电；主功率硬关断须由独立本地路径维持。"
        },
        "STORE": {
            "pole1": "2-1闭合，两端pin1 NC，REG1断开；保留R_PRE_CTRL_PD下拉PRE_CTRL_3V3",
            "pole2": "5-4闭合，CTRL_MODULE接B0并关断模块",
            "output_after_disable": "7x510负载提供约14ms标称RC时间常数的输出衰减；不等于瞬时无电压。"
        },
        "single_fault_and_backfeed": {
            "CTRL_wire_or_contact_open_in_STORE": "CTRL开路默认开启，模式开关不是安全隔离；需以主功率硬关断及本地保护独立保证风险状态。",
            "module_pin5_to_B0": "有意将模块输出回路负端接B0，模块内部1500V输入/输出隔离不再构成整板地隔离边界。",
            "5v_to_reg1_backfeed": "无计划的直接网线；GPIO/外置MCU/外部5V和芯片钳位路径仍需分析/实测，不能声称无回灌。",
            "ctrl_sink_current": "V1.7没有给出CTRL引脚灌电流上限；需确认模块CTRL内部电路与MN22接点电流。"
        },
        "startup_order": "URB输入直接接CELL_B9；BREG经原有D_BREG/R_BREG_FEED/Q_BREG形成REGIN后由BQ76952生成REG1=3.3V。两路径独立，没有文档保证5V与REG1/BREG先后顺序、PGOOD或交叉复位；REG1失电不自动关断RUN档的5V。"
    }

    # Overall open items are explicit and not promoted to model PASS.
    check("9.5uF输入旁路可满足模块动态条件", False,
          "手册page 6 C1=47–100uF为测试电路参考；现有输入仅9.5uF标称且高压陶瓷有效值未核，动态阻抗/PI稳定性未仿真。",
          status="OPEN")
    check("模块启动时间足以保证并发负载", False,
          "V1.7第1页典型10ms与第2页典型1ms矛盾，均无最大值；低线、实际电容、风扇启动场景未建模。",
          status="OPEN")
    check("CTRL关断与失电安全已通过硬件验证", False,
          "CTRL开路即开；STORE触点/引线故障可使模块开启；REG1丢失时RUN档CTRL仍开路；U_GATE1掉电时B/C输入注入及Y输出状态未验证；尚无硬件样板。",
          status="OPEN")
    check("模块温升和效率边界已验收", False,
          "效率81%仅在48V额定阻性负载/25°C条件；22.5V外推损耗约1.8W，实际效率、外壳热阻、舱体气流与风扇失效未测。",
          status="OPEN")
    check("5V/REG1跨域回灌已经排除", False,
          "无规划直接连接，但跨域GPIO/外部供电与IC保护结构没有闭合；本轮未声称无回灌。",
          status="OPEN")

    digest_records = [
        {"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size}
        for rel, path in sources.items()
    ]
    model = {
        "schema": "hlk-urb4805-bms-power-reduced-model-v1",
        "generated_by": relative(Path(__file__).resolve()),
        "input": {"path": relative(INPUT), "sha256": sha256(INPUT)},
        "model_scope": case["simulation_method"],
        "paper_calculations": {
            "minimum_load_bank": min_load,
            "static_power_and_thermal_envelopes": power,
            "startup_capacitor_current_sensitivity": startup_rows,
            "output_bleeder_rc_discharge": discharge,
            "control_rail_rc_discharge": control_decay,
            "part_count_plan": count_plan,
            "switch_state_and_supply_domain_analysis": state_analysis
        },
        "checks": checks,
        "summary": {
            "pass_count": sum(c["status"] == "PASS" for c in checks),
            "fail_count": sum(c["status"] == "FAIL" for c in checks),
            "open_count": sum(c["status"] == "OPEN" for c in checks),
            "model_or_hardware_acceptance": False,
            "ngspice_available": shutil.which("ngspice") is not None,
            "ngspice_binary_path": shutil.which("ngspice"),
            "formal_native_source_current_implementation_verified": True,
            "procurement_bom_current_implementation_verified": bom_current_implementation
        },
        "source_hashes": digest_records
    }

    evidence = {
        "recordType": "conditional_engineering_verification",
        "status": "纸面计算与受限模型已更新；输入旁路、启动、热、CTRL故障与回灌仍待闭合；非正式工程验收",
        "createdLocalDate": "2026-10-15",
        "purpose": "响应已保存的5V模块、最小负载、开关状态及重复下拉精简实现，核对正式连接源并更新计算/受限模型；不修改正式工程或BOM",
        "implementedConfiguration": {
            "moduleMpn": "URB4805YMD-10WR3",
            "commonLibraryCadComponent": "C5365326",
            "commonLibraryUuid": "bd9ce49027fc4c83a722f82268af98d2",
            "switchPinMap": switch["expected_pin_map"],
            "loadBankMpn": load["mpn"],
            "loadBankQty": n_load,
            "loadBankRefs": load["refs"],
            "switchPurchaseQuantity": 1
        },
        "verifiedImplementationDelta": {
            "removeRefs": implementation["remove_refs"],
            "replaceInPlace": {**implementation["replace_in_place"], "to_ref": module_ref},
            "addComponents": [
                {"refs": added_refs, "quantity": len(added_refs), "mpn": load["mpn"], "each": "510Ω ±1%, 100ppm/°C, 0603, 100mW", "pins": {"1": "VOUT_5V", "2": "B0"}}
            ],
            "retainRefs": implementation["retain_refs"],
            "netRename": control["net_rename"],
            "removeDuplicatePulldown": control["remove_duplicate_pulldown_ref"],
            "keepPulldown": control["retain_pulldown_ref"],
            "preservePor": control["preserve_por_ref"],
            "perPinModuleMap": case["expected_module_pin_map"],
            "currentSourceContainsImplementation": True,
            "currentSourceWasModifiedByThisVerifier": False
        },
        "scopeCounts": count_plan,
        "modelFile": relative(RESULT),
        "openAcceptanceItems": [
            "厂家明确不支持热插拔；落实避免模块输入直接带电插拔的连接与上电约束，不能凭RC或功率计算认定热插拔可用。",
            "在18–75V输入范围、实际电池22.5–37.8V及最大浪涌下验证模块输入端纹波、超调和与现有PI输入网络的稳定性；特别确认72.7V纸面TVS钳位边界有足够瞬态裕量。",
            "向海凌科澄清V1.7典型启动时间第1页10ms与第2页1ms不一致、低线和容性负载条件；实测最大输出爬升/跌落及风扇启动同时发生时的电流/欠压。",
            "验证现有9.5uF输入陶瓷的37.8V偏压后有效容量；必要时按原厂答复与可复用共用库补足100V输入电容，不把典型47–100uF参考当作硬下限。",
            "首板验证7×510Ω最低负载在输出精度、温度和有效阻值角落；逐颗温升、焊盘温度、输出纹波、空载/轻载行为。",
            "验证模块CTRL的内部上拉/灌电流、触点允许电流、开关接点弹跳与方向；断线/接触故障是否导致5V意外开启；核对U_GATE1掉电时B/C输入灌电、Y输出状态与反向供电路径。",
            "确认REG1/BREG失电时RUN档5V仍可能开启是否符合系统策略；确认外部MCU GPIO、通信口和其他电源的反向供电路径。",
            "用首板和真实外壳在0–60°C、最大负载、风扇失效/受限气流测壳温；手册热曲线仅用于纸面功率边界，不是具体板级热验收。",
            "确认输出陶瓷在5V偏压/温度下的有效总电容；CTRL关断后以示波器确认7×510Ω实测放电轨迹和其他负载影响。"
        ],
        "sourceHashes": digest_records,
        "checks": checks,
        "summary": model["summary"]
    }
    dump_json(RESULT, model)
    dump_json(CHECK, evidence)
    print(json.dumps({
        "result": relative(RESULT),
        "check": relative(CHECK),
        "summary": model["summary"],
        "minLoad": min_load,
        "partCounts": count_plan,
        "loadEnvelope": scenarios,
        "startupSensitivityRows": len(startup_rows)
    }, ensure_ascii=False, indent=2))
    return 1 if model["summary"]["fail_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
