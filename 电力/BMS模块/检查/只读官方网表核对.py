#!/usr/bin/env python3
"""Compare a newly exported official ENET against the BMS construction manifest.

This is a file-only audit. It never opens or controls EasyEDA. The netlist path is
required so a stale snapshot cannot accidentally be treated as fresh evidence.

Example:
  py BMS模块/检查/只读官方网表核对.py "<new official netlist JSON or .enet>"

When explicitly run, it updates:
  检查/官方网表连接核对.json
  检查/官方网表连接核对.md
  检查/BMS-9S4P.enet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "当前施工输入.json"
CLEANUP_MANIFEST = HERE / "代理采购身份清理清单.json"
OUT_JSON = HERE / "官方网表连接核对.json"
OUT_MD = HERE / "官方网表连接核对.md"
OUT_ENET = HERE / "BMS-9S4P.enet"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_pin(value: Any) -> str:
    value = str(value or "").strip()
    if value.isdigit():
        return str(int(value))
    return value.upper()


def canonical_net(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("name", value.get("net", ""))
    return str(value or "").strip().upper()


def canonical_mpn(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value or "").strip().lower()
    if value in {"yes", "true", "1", "on", "y"}:
        return True
    if value in {"no", "false", "0", "off", "n"}:
        return False
    return None


def read_netlist(path: Path) -> tuple[dict[str, Any], str, bytes]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        enet_text = parsed["text"]
        enet = json.loads(enet_text)
    elif isinstance(parsed, dict) and isinstance(parsed.get("components"), dict):
        enet = parsed
        enet_text = text
    else:
        raise ValueError("输入既不是官方网表包装 JSON（text 字段），也不是 ENET JSON。")
    if not isinstance(enet.get("components"), dict):
        raise ValueError("ENET 缺少 components 对象。")
    return enet, enet_text, raw


def collect_expected(items: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not isinstance(items, list):
        raise ValueError("施工输入必须是位号记录数组。")
    expected: dict[str, dict[str, Any]] = {}
    duplicate_refs: list[str] = []
    for item in items:
        ref = str(item.get("ref", "")).strip()
        if not ref:
            continue
        if ref not in expected:
            expected[ref] = {
                "mpn": item.get("mpn") or "",
                "pins": {},
                "addIntoPcb": item.get("addIntoPcb"),
                "records": 1,
            }
        else:
            duplicate_refs.append(ref)
            expected[ref]["records"] += 1
            # Merge units belonging to one designator, including multi-unit switches.
            # If values disagree, preserve the first and report a conflict below.
            if not expected[ref].get("mpn") and item.get("mpn"):
                expected[ref]["mpn"] = item.get("mpn")
            if expected[ref].get("addIntoPcb") is None:
                expected[ref]["addIntoPcb"] = item.get("addIntoPcb")
        pins = item.get("pins") or {}
        if not isinstance(pins, dict):
            continue
        for number, net in pins.items():
            key = canonical_pin(number)
            value = "" if net is None else str(net).strip()
            if key in expected[ref]["pins"] and canonical_net(expected[ref]["pins"][key]) != canonical_net(value):
                expected[ref].setdefault("pinConflicts", []).append({
                    "pin": key,
                    "first": expected[ref]["pins"][key],
                    "later": value,
                })
            else:
                expected[ref]["pins"][key] = value
    return expected, duplicate_refs


def collect_actual_pins(component: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    source = component.get("pinInfoMap") or {}
    if not isinstance(source, dict):
        return {}, []
    pins: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for key, info in source.items():
        if not isinstance(info, dict):
            continue
        number = canonical_pin(info.get("number") or key)
        net_raw = info.get("net", "")
        net = "" if net_raw is None else str(net_raw).strip()
        if number in pins and canonical_net(pins[number]) != canonical_net(net):
            conflicts.append({"pin": number, "nets": [pins[number], net]})
        else:
            pins[number] = net
    return pins, conflicts


def compare(expected: dict[str, dict[str, Any]], enet: dict[str, Any], duplicate_refs: list[str]) -> dict[str, Any]:
    actual = enet["components"]
    named_nets: set[str] = set()
    pin_info_record_count = 0
    blank_net_record_count = 0
    for component in actual.values():
        component = component if isinstance(component, dict) else {}
        pin_info_map = component.get("pinInfoMap") or {}
        if not isinstance(pin_info_map, dict):
            continue
        for info in pin_info_map.values():
            if not isinstance(info, dict):
                continue
            pin_info_record_count += 1
            net = canonical_net(info.get("net"))
            if net:
                named_nets.add(net)
            else:
                blank_net_record_count += 1
    expected_refs, actual_refs = set(expected), set(actual)
    missing_refs = sorted(expected_refs - actual_refs)
    unexpected_refs = sorted(actual_refs - expected_refs)
    pin_mismatches: list[dict[str, Any]] = []
    pin_presence_mismatches: list[dict[str, Any]] = []
    pin_conflicts: list[dict[str, Any]] = []
    mpn_mismatches: list[dict[str, Any]] = []
    pcb_mismatches: list[dict[str, Any]] = []
    transformer_audit: list[dict[str, Any]] = []
    compared_pin_count = 0

    for ref in sorted(expected_refs & actual_refs):
        exp = expected[ref]
        component = actual[ref] if isinstance(actual[ref], dict) else {}
        props = component.get("props") or {}
        exp_pins = exp["pins"]
        act_pins, act_conflicts = collect_actual_pins(component)
        pin_conflicts.extend({"ref": ref, **item} for item in act_conflicts)
        pin_conflicts.extend({"ref": ref, **item} for item in exp.get("pinConflicts", []))
        absent_expected_pins = [
            {"pin": pin, "expectedNet": net}
            for pin, net in exp_pins.items()
            if pin not in act_pins
        ]
        extra_actual_pins = [
            {"pin": pin, "actualNet": net}
            for pin, net in act_pins.items()
            if pin not in exp_pins
        ]
        if absent_expected_pins or extra_actual_pins:
            pin_presence_mismatches.append({
                "ref": ref,
                "expectedPinsAbsentFromNetlist": absent_expected_pins,
                "actualPinsAbsentFromInput": extra_actual_pins,
            })
        pin_diffs = []
        for pin in sorted(set(exp_pins) | set(act_pins), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            expected_net = exp_pins.get(pin)
            actual_net = act_pins.get(pin)
            compared_pin_count += 1
            if canonical_net(expected_net) != canonical_net(actual_net):
                pin_diffs.append({"pin": pin, "expectedNet": expected_net, "actualNet": actual_net})
        if pin_diffs:
            pin_mismatches.append({"ref": ref, "differences": pin_diffs})

        expected_mpn = str(exp.get("mpn") or "").strip()
        actual_mpn = str(props.get("Manufacturer Part") or "").strip()
        if canonical_mpn(expected_mpn) != canonical_mpn(actual_mpn):
            mpn_mismatches.append({"ref": ref, "expectedSourceMpn": expected_mpn, "actualManufacturerPart": actual_mpn})

        expected_pcb = parse_bool(exp.get("addIntoPcb"))
        actual_pcb = parse_bool(props.get("Convert to PCB"))
        if expected_pcb is not None and actual_pcb != expected_pcb:
            pcb_mismatches.append({"ref": ref, "expectedAddIntoPcb": expected_pcb, "actualConvertToPCB": props.get("Convert to PCB")})

    for index in range(1, 10):
        ref = f"T_BAL{index}"
        if ref not in expected and ref not in actual:
            continue
        exp = expected.get(ref, {})
        component = actual.get(ref, {})
        props = component.get("props") or {} if isinstance(component, dict) else {}
        source_mpn = str(exp.get("mpn") or "").strip()
        manufacturer = str(props.get("Manufacturer") or "").strip()
        supplier = str(props.get("Supplier") or "").strip()
        supplier_part = str(props.get("Supplier Part") or "").strip()
        device_name = str(props.get("DeviceName") or "").strip()
        source_is_coilcraft = source_mpn.casefold().startswith("coilcraft ")
        manufacturer_is_wurth = "wurth" in manufacturer.casefold()
        neutral_bom_identity_present = (
            supplier == "BOM_ONLY"
            and supplier_part == "SEE_BOM_CSV"
            and canonical_mpn(device_name) == canonical_mpn(source_mpn)
        )
        transformer_audit.append({
            "ref": ref,
            "sourceMpn": source_mpn,
            "officialManufacturer": manufacturer,
            "officialManufacturerPart": props.get("Manufacturer Part", ""),
            "officialSupplier": supplier,
            "officialSupplierPart": supplier_part,
            "officialDeviceName": device_name,
            "sourceStartsWithCoilcraft": source_is_coilcraft,
            "WurthManufacturerOnCoilcraftSource": source_is_coilcraft and manufacturer_is_wurth,
            "neutralBOMIdentityPresent": neutral_bom_identity_present,
            "supplierIdentityStillPresent": not neutral_bom_identity_present,
            "officialComponentPresent": ref in actual,
        })

    pin_count_expected = sum(len(record["pins"]) for record in expected.values())
    pin_count_actual = sum(len(collect_actual_pins(value if isinstance(value, dict) else {})[0]) for value in actual.values())
    return {
        "netlistStructure": {
            "formatVersion": enet.get("version"),
            "topLevelKeys": sorted(enet.keys()),
            "componentCount": len(actual),
            "pinInfoMapRecordCount": pin_info_record_count,
            "pinInfoMapBlankNetRecordCount": blank_net_record_count,
            "distinctNamedNetCount": len(named_nets),
        },
        "summary": {
            "expectedRefCount": len(expected_refs),
            "actualRefCount": len(actual_refs),
            "refSetMatch": not missing_refs and not unexpected_refs,
            "missingRefCount": len(missing_refs),
            "unexpectedRefCount": len(unexpected_refs),
            "expectedPinCount": pin_count_expected,
            "actualPinCount": pin_count_actual,
            "comparedPinCount": compared_pin_count,
            "pinMismatchRefCount": len(pin_mismatches),
            "pinDifferenceCount": sum(len(x["differences"]) for x in pin_mismatches),
            "pinPresenceMismatchRefCount": len(pin_presence_mismatches),
            "expectedPinsAbsentFromNetlistCount": sum(len(x["expectedPinsAbsentFromNetlist"]) for x in pin_presence_mismatches),
            "actualPinsAbsentFromInputCount": sum(len(x["actualPinsAbsentFromInput"]) for x in pin_presence_mismatches),
            "mpnMismatchCount": len(mpn_mismatches),
            "pcbFlagMismatchCount": len(pcb_mismatches),
            "transformerCount": len(transformer_audit),
            "transformerWurthOnCoilcraftCount": sum(item["WurthManufacturerOnCoilcraftSource"] for item in transformer_audit),
            "transformerNeutralBOMIdentityMismatchCount": sum(not item["neutralBOMIdentityPresent"] for item in transformer_audit),
            "transformerSupplierIdentityRemainingCount": sum(item["supplierIdentityStillPresent"] for item in transformer_audit),
            "duplicateInputRefCount": len(duplicate_refs),
            "pinConflictCount": len(pin_conflicts),
        },
        "missingRefs": missing_refs,
        "unexpectedRefs": unexpected_refs,
        "duplicateInputRefs": sorted(set(duplicate_refs)),
        "pinMismatches": pin_mismatches,
        "pinPresenceMismatches": pin_presence_mismatches,
        "pinConflicts": pin_conflicts,
        "mpnMismatches": mpn_mismatches,
        "pcbFlagMismatches": pcb_mismatches,
        "transformerProxyAudit": transformer_audit,
    }


def audit_proxy_cleanup(expected: dict[str, dict[str, Any]], enet: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify only the 56 approved attribute-cleanup refs; exclude 18 SiR reviews."""
    rows = manifest.get("clearOrCorrect", [])
    actions = {
        str(row.get("ref", "")).strip(): row
        for row in rows
        if row.get("ref")
        and row.get("action") != "manual_source_order_code_review_before_clear"
    }
    transformer_refs = {f"T_BAL{index}" for index in range(1, 10)}
    target_refs = set(actions) | transformer_refs
    neutral_supplier_refs = {
        ref for ref, row in actions.items()
        if "Supplier" in (row.get("clearFields") or [])
    } | transformer_refs
    manual_refs = set(manifest.get("reviewBeforeClearing", {}).get("refs", []))
    carrier_refs = {
        str(row.get("ref", "")).strip()
        for row in manifest.get("carrierOnlyKeep", [])
        if row.get("ref")
    }
    actual = enet.get("components", {})
    checks = []
    for ref in sorted(target_refs):
        expected_mpn = str(expected.get(ref, {}).get("mpn") or "").strip()
        component = actual.get(ref)
        props = component.get("props") or {} if isinstance(component, dict) else {}
        actual_device_name = str(props.get("DeviceName") or "").strip()
        device_name_match = bool(expected_mpn) and actual_device_name == expected_mpn
        expected_manufacturer = "Coilcraft" if ref in transformer_refs else None
        actual_manufacturer = str(props.get("Manufacturer") or "").strip()
        manufacturer_match = (
            actual_manufacturer == expected_manufacturer
            if expected_manufacturer is not None else None
        )
        expects_neutral_supplier = ref in neutral_supplier_refs
        actual_supplier = str(props.get("Supplier") or "").strip()
        actual_supplier_part = str(props.get("Supplier Part") or "").strip()
        supplier_match = (
            actual_supplier == "BOM_ONLY" and actual_supplier_part == "SEE_BOM_CSV"
            if expects_neutral_supplier else None
        )
        checks.append({
            "ref": ref,
            "sourceMpn": expected_mpn,
            "componentPresent": isinstance(component, dict),
            "actualDeviceName": props.get("DeviceName", ""),
            "deviceNameMatchesSourceMpn": device_name_match,
            "expectedManufacturer": expected_manufacturer,
            "actualManufacturer": props.get("Manufacturer", ""),
            "manufacturerMatchesExpected": manufacturer_match,
            "expectsNeutralSupplierIdentity": expects_neutral_supplier,
            "actualSupplier": props.get("Supplier", ""),
            "actualSupplierPart": props.get("Supplier Part", ""),
            "neutralSupplierIdentityMatches": supplier_match,
        })

    missing_source_mpn_refs = [x["ref"] for x in checks if not x["sourceMpn"]]
    missing_component_refs = [x["ref"] for x in checks if not x["componentPresent"]]
    device_name_mismatches = [x for x in checks if not x["deviceNameMatchesSourceMpn"]]
    manufacturer_mismatches = [x for x in checks if x["manufacturerMatchesExpected"] is False]
    supplier_mismatches = [
        x for x in checks
        if x["expectsNeutralSupplierIdentity"] and not x["neutralSupplierIdentityMatches"]
    ]
    scope_errors = {
        "expectedAutomaticActionCountIs48": len(actions) == 48,
        "expectedManualSourceOrderReviewCountIs18": len(manual_refs) == 18,
        "expectedUniqueRefCountIs56": len(target_refs) == 56,
        "expectedNeutralSupplierRefCountIs44": len(neutral_supplier_refs) == 44,
        "expectedDeviceNameOnlyRefCountIs12": len(target_refs - neutral_supplier_refs) == 12,
        "expectedCarrierOnlyKeepRefCountIs6": len(carrier_refs) == 6,
        "manualSourceOrderReviewOverlap": sorted(target_refs & manual_refs),
        "carrierOnlyKeepOverlap": sorted(target_refs & carrier_refs),
    }
    scope_valid = (
        scope_errors["expectedAutomaticActionCountIs48"]
        and scope_errors["expectedManualSourceOrderReviewCountIs18"]
        and scope_errors["expectedUniqueRefCountIs56"]
        and scope_errors["expectedNeutralSupplierRefCountIs44"]
        and scope_errors["expectedDeviceNameOnlyRefCountIs12"]
        and scope_errors["expectedCarrierOnlyKeepRefCountIs6"]
        and not scope_errors["manualSourceOrderReviewOverlap"]
        and not scope_errors["carrierOnlyKeepOverlap"]
    )
    scope_errors["scopeValid"] = scope_valid
    passed = (
        scope_valid and not missing_source_mpn_refs and not missing_component_refs
        and not device_name_mismatches and not manufacturer_mismatches
        and not supplier_mismatches
    )
    return {
        "scope": "仅核对清理清单中48个自动处理位号与T_BAL1–9的并集；18个Q_P/Q_S订货号人工复核项及carrierOnlyKeep器件排除。",
        "targetRefs": sorted(target_refs),
        "neutralSupplierRefs": sorted(neutral_supplier_refs),
        "deviceNameOnlyRefs": sorted(target_refs - neutral_supplier_refs),
        "excludedManualSourceOrderCodeRefs": sorted(manual_refs),
        "excludedCarrierOnlyKeepRefs": sorted(carrier_refs),
        "scopeChecks": scope_errors,
        "summary": {
            "targetRefCount": len(target_refs),
            "neutralSupplierTargetCount": len(neutral_supplier_refs),
            "deviceNameOnlyTargetCount": len(target_refs - neutral_supplier_refs),
            "manualSourceOrderCodeRefsExcludedCount": len(manual_refs),
            "carrierOnlyKeepRefsExcludedCount": len(carrier_refs),
            "missingSourceMpnCount": len(missing_source_mpn_refs),
            "missingComponentCount": len(missing_component_refs),
            "deviceNameMismatchCount": len(device_name_mismatches),
            "manufacturerMismatchCount": len(manufacturer_mismatches),
            "neutralSupplierMismatchCount": len(supplier_mismatches),
            "passed": passed,
        },
        "missingSourceMpnRefs": missing_source_mpn_refs,
        "missingComponentRefs": missing_component_refs,
        "deviceNameMismatches": device_name_mismatches,
        "manufacturerMismatches": manufacturer_mismatches,
        "neutralSupplierMismatches": supplier_mismatches,
        "checks": checks,
    }


def markdown_report(result: dict[str, Any], evidence: dict[str, Any]) -> str:
    s = result["summary"]
    lines = [
        "# BMS 官方网表与施工输入核对",
        "",
        f"- 网表证据：`{evidence['netlistPath']}`",
        f"- 网表 SHA256：`{evidence['netlistSha256']}`",
        f"- 检查时间：{evidence['checkedAtUtc']}",
        f"- 输入位号数：{s['expectedRefCount']}；官方网表位号数：{s['actualRefCount']}；位号集合：{'PASS' if s['refSetMatch'] else 'FAIL'}",
        f"- 网表结构：ENET {result['netlistStructure']['formatVersion']}；顶层字段 {', '.join(result['netlistStructure']['topLevelKeys'])}；components {result['netlistStructure']['componentCount']}；pinInfoMap 记录 {result['netlistStructure']['pinInfoMapRecordCount']}（空网 {result['netlistStructure']['pinInfoMapBlankNetRecordCount']}）；不同网络名 {result['netlistStructure']['distinctNamedNetCount']}",
        f"- 引脚：输入 {s['expectedPinCount']}，官方网表 {s['actualPinCount']}，比较 {s['comparedPinCount']}；差异位号 {s['pinMismatchRefCount']}、差异引脚 {s['pinDifferenceCount']}、重复编号/网络冲突 {s['pinConflictCount']}",
        f"- pinInfoMap 编号出现情况：施工输入引脚未出现在官方网表 {s['expectedPinsAbsentFromNetlistCount']}，官方网表多出引脚 {s['actualPinsAbsentFromInputCount']}",
        f"- 源 MPN 差异：{s['mpnMismatchCount']}；Convert to PCB 标记差异：{s['pcbFlagMismatchCount']}",
        f"- T_BAL1–9 代理身份核查：覆盖 {s['transformerCount']} 个；Coilcraft 源仍标 Wurth：{s['transformerWurthOnCoilcraftCount']}；中性 Supplier/Supplier Part 与源 DeviceName 不匹配：{s['transformerNeutralBOMIdentityMismatchCount']}",
        "",
        "本报告只核对官方网表导出与施工输入的位号、引脚网络、Manufacturer Part 和 Convert to PCB 标记；不代表 ERC、封装/焊盘、热设计、装配或制造验收通过。",
        "",
    ]
    sections = [
        ("位号缺失", result["missingRefs"]),
        ("位号多出", result["unexpectedRefs"]),
        ("MPN 差异", result["mpnMismatches"]),
        ("PCB 标记差异", result["pcbFlagMismatches"]),
        ("引脚网络差异", result["pinMismatches"]),
        ("引脚编号出现情况差异", result["pinPresenceMismatches"]),
        ("输入重复位号", result["duplicateInputRefs"]),
        ("引脚编号或网络冲突", result["pinConflicts"]),
        ("T_BAL1–9 代理身份明细", result["transformerProxyAudit"]),
    ]
    for title, values in sections:
        lines.extend([f"## {title}", ""])
        if not values:
            lines.extend(["无。", ""])
            continue
        if title in {"位号缺失", "位号多出", "输入重复位号"}:
            lines.extend([", ".join(values), ""])
        elif title == "MPN 差异":
            lines.extend(["| 位号 | 输入源 MPN | 网表 Manufacturer Part |", "|---|---|---|"])
            lines.extend([f"| {x['ref']} | {x['expectedSourceMpn']} | {x['actualManufacturerPart']} |" for x in values])
            lines.append("")
        elif title == "PCB 标记差异":
            lines.extend(["| 位号 | 输入 addIntoPcb | 网表 Convert to PCB |", "|---|---:|---|"])
            lines.extend([f"| {x['ref']} | {x['expectedAddIntoPcb']} | {x['actualConvertToPCB']} |" for x in values])
            lines.append("")
        elif title == "引脚网络差异":
            for item in values:
                diffs = "; ".join(f"pin {d['pin']}: 输入 `{d['expectedNet'] or 'NC'}` / 网表 `{d['actualNet'] or 'NC'}`" for d in item["differences"])
                lines.append(f"- **{item['ref']}**：{diffs}")
            lines.append("")
        elif title == "引脚编号出现情况差异":
            for item in values:
                absent = ", ".join(
                    f"pin {x['pin']} (输入网 `{x['expectedNet'] or 'NC'}`)"
                    for x in item["expectedPinsAbsentFromNetlist"]
                ) or "无"
                extra = ", ".join(
                    f"pin {x['pin']} (网表网 `{x['actualNet'] or 'NC'}`)"
                    for x in item["actualPinsAbsentFromInput"]
                ) or "无"
                lines.append(f"- **{item['ref']}**：输入有、网表 pinInfoMap 未列出：{absent}；网表有、输入未列出：{extra}")
            lines.append("")
        elif title == "T_BAL1–9 代理身份明细":
            lines.extend(["| 位号 | 源 MPN | Manufacturer | Manufacturer Part | Supplier | Supplier Part | DeviceName |", "|---|---|---|---|---|---|---|"])
            lines.extend([
                f"| {x['ref']} | {x['sourceMpn']} | {x['officialManufacturer']} | {x['officialManufacturerPart']} | {x['officialSupplier']} | {x['officialSupplierPart']} | {x['officialDeviceName']} |"
                for x in values
            ])
            lines.append("")
        else:
            lines.extend(["```json", json.dumps(values, ensure_ascii=False, indent=2), "```", ""])
    cleanup = result.get("proxyCleanupAudit")
    if cleanup:
        c = cleanup["summary"]
        lines.extend([
            "## 代理采购属性清理复核（限定范围）",
            "",
            f"- 范围：{cleanup['scope']}",
            f"- 目标位号：{c['targetRefCount']}；中性 Supplier/Supplier Part 目标：{c['neutralSupplierTargetCount']}；仅校正 DeviceName：{c['deviceNameOnlyTargetCount']}",
            f"- 排除：18个SiR订货号人工复核项、{c['carrierOnlyKeepRefsExcludedCount']}个carrierOnlyKeep项；范围检查：{'PASS' if cleanup['scopeChecks']['scopeValid'] else 'FAIL'}",
            f"- DeviceName/源MPN 不匹配：{c['deviceNameMismatchCount']}；T_BAL Manufacturer 非 Coilcraft：{c['manufacturerMismatchCount']}；中性采购字段不匹配：{c['neutralSupplierMismatchCount']}；缺少组件：{c['missingComponentCount']}；清理属性复核：{'PASS' if c['passed'] else 'FAIL'}",
            "",
        ])
        for title, key in (("DeviceName 不匹配", "deviceNameMismatches"), ("Manufacturer 不匹配", "manufacturerMismatches"), ("中性采购字段不匹配", "neutralSupplierMismatches")):
            values = cleanup[key]
            lines.extend([f"### {title}", ""])
            if values:
                lines.extend(["| 位号 | 源 MPN | 网表 DeviceName | Supplier | Supplier Part |", "|---|---|---|---|---|"])
                lines.extend([
                    f"| {x['ref']} | {x['sourceMpn']} | {x['actualDeviceName']} | {x['actualSupplier']} | {x['actualSupplierPart']} |"
                    for x in values
                ])
                lines.append("")
            else:
                lines.extend(["无。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("netlist", type=Path, help="新导出的官方网表 JSON 包装文件或纯 .enet 文件")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="施工输入 JSON；默认检查/当前施工输入.json")
    args = parser.parse_args()

    netlist_path = args.netlist.resolve()
    input_path = args.input.resolve()
    if not netlist_path.is_file():
        parser.error(f"网表文件不存在：{netlist_path}")
    if not input_path.is_file():
        parser.error(f"施工输入文件不存在：{input_path}")

    try:
        input_raw = input_path.read_bytes()
        items = json.loads(input_raw.decode("utf-8-sig"))
        enet, enet_text, netlist_raw = read_netlist(netlist_path)
        expected, duplicate_refs = collect_expected(items)
        result = compare(expected, enet, duplicate_refs)
        cleanup_manifest = json.loads(CLEANUP_MANIFEST.read_text(encoding="utf-8"))
        result["proxyCleanupAudit"] = audit_proxy_cleanup(expected, enet, cleanup_manifest)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"读取或核对失败：{exc}", file=sys.stderr)
        return 2

    enet_bytes = enet_text.encode("utf-8")
    evidence = {
        "netlistPath": str(netlist_path),
        "netlistFileSha256": sha256(netlist_raw),
        "netlistSha256": sha256(enet_bytes),
        "inputPath": str(input_path),
        "inputSha256": sha256(input_raw),
        "checkedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "officialNetlistName": enet.get("name") or netlist_path.name,
        "readOnlyFileAudit": True,
    }
    report = {
        "title": "BMS 官方网表与施工输入核对",
        "evidence": evidence,
        **result,
        "scopeNote": "只读文件核对，不代表 ERC、封装/焊盘、热设计、装配或制造验收通过。",
    }

    # These files are updated only after the user supplies an explicit netlist path.
    OUT_ENET.write_bytes(enet_bytes)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown_report(result, evidence), encoding="utf-8")
    print(json.dumps({"outputs": [str(OUT_JSON), str(OUT_MD), str(OUT_ENET)], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
