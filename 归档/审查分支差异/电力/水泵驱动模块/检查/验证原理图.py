#!/usr/bin/env python3
"""只读核对水泵驱动导出网表与现行连接契约和 BOM。

运行：python 验证原理图.py
脚本只读取连接契约、BOM 和当前交付 ENET；仅写入本目录的网表逐脚核对.json。
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parents[1]
CHECK_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = MODULE_DIR / "原理图准备" / "原理图连接契约.json"
BOM_PATH = MODULE_DIR / "原理图准备" / "BOM.csv"
NETLIST_PATH = MODULE_DIR / "eda工程文件" / "导出" / "当前交付" / "水泵驱动.enet"
OUTPUT_PATH = CHECK_DIR / "网表逐脚核对.json"

EXPECTED_BOARD_REFS = 49
EXPECTED_BOM_ROWS = 37
EXPECTED_TOTAL_BOM_REFS = 58
EXPECTED_EXTERNAL_REFS = 9


def nonempty(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def normalized_net(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() if text else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_result(passed: bool, **fields: Any) -> dict[str, Any]:
    return {"status": "PASS" if passed else "FAIL", **fields}


def expand_bom(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str], list[dict[str, Any]]]:
    by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    board_refs: list[str] = []
    external_refs: list[str] = []
    issues: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=2):
        raw_refs = (row.get("Reference") or "").strip()
        scope = (row.get("Scope") or "").strip().upper()
        mpn = (row.get("Manufacturer part") or "").strip()
        refs = [ref.strip() for ref in raw_refs.split(",") if ref.strip()]
        if not refs:
            issues.append({"row": row_number, "issue": "BOM行没有Reference"})
            continue
        if scope not in {"ON_BOARD", "HARNESS", "DAUGHTERBOARD", "OFF_BOARD"}:
            issues.append({"row": row_number, "issue": "未知Scope", "scope": scope})

        for ref in refs:
            entry = {
                "row": row_number,
                "scope": scope,
                "mpn": mpn,
                "package": (row.get("Package") or "").strip(),
            }
            by_ref[ref].append(entry)
            if scope == "ON_BOARD":
                board_refs.append(ref)
            else:
                external_refs.append(ref)

    return by_ref, board_refs, external_refs, issues


def main() -> int:
    report: dict[str, Any] = {
        "title": "水泵驱动原生网表逐脚核对",
        "scope": "只读比对当前导出 ENET、原理图连接契约与 BOM；不执行 EDA/DRC，不验证 PCB、实物或整机验收。",
        "generated_at": None,
        "inputs": {},
        "checks": {},
        "references": [],
        "pins": [],
        "fatal_error": None,
    }

    try:
        contract_bytes = CONTRACT_PATH.read_bytes()
        bom_bytes = BOM_PATH.read_bytes()
        netlist_bytes = NETLIST_PATH.read_bytes()
        report["inputs"] = {
            "contract": str(CONTRACT_PATH),
            "bom": str(BOM_PATH),
            "netlist": str(NETLIST_PATH),
            "netlist_sha256": hashlib.sha256(netlist_bytes).hexdigest(),
            "netlist_bytes": len(netlist_bytes),
        }

        contract = json.loads(contract_bytes.decode("utf-8-sig"))
        with BOM_PATH.open("r", encoding="utf-8-sig", newline="") as bom_file:
            bom_reader = csv.DictReader(bom_file)
            bom_rows = list(bom_reader)
            bom_columns = bom_reader.fieldnames or []
        netlist = json.loads(netlist_bytes.decode("utf-8-sig"))

        contract_parts = contract.get("parts", []) if isinstance(contract, dict) else []
        contract_refs = [str(part.get("ref", "")).strip() for part in contract_parts]
        contract_duplicates = sorted(ref for ref, count in Counter(contract_refs).items() if ref and count > 1)
        contract_empty_refs = [index for index, ref in enumerate(contract_refs) if not ref]
        contract_by_ref = {
            str(part.get("ref", "")).strip(): part
            for part in contract_parts
            if str(part.get("ref", "")).strip()
        }

        bom_by_ref, bom_board_refs, bom_external_refs, bom_issues = expand_bom(bom_rows)
        duplicate_bom_refs = sorted(ref for ref, entries in bom_by_ref.items() if len(entries) > 1)

        enet_components = netlist.get("components", {}) if isinstance(netlist, dict) else {}
        if not isinstance(enet_components, dict):
            enet_components = {}
        actual_components_by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
        no_designator_components: list[str] = []
        for unique_id, component in enet_components.items():
            props = component.get("props", {}) if isinstance(component, dict) else {}
            ref = str(props.get("Designator", "")).strip()
            if ref:
                actual_components_by_ref[ref].append(component)
            else:
                no_designator_components.append(str(unique_id))

        actual_refs = sorted(actual_components_by_ref)
        actual_duplicates = sorted(ref for ref, comps in actual_components_by_ref.items() if len(comps) > 1)
        contract_ref_set = set(contract_refs) - {""}
        bom_board_ref_set = set(bom_board_refs)
        bom_external_ref_set = set(bom_external_refs)
        actual_ref_set = set(actual_refs)

        missing_actual_refs = sorted(contract_ref_set - actual_ref_set)
        unexpected_actual_refs = sorted(actual_ref_set - contract_ref_set)
        external_refs_in_netlist = sorted(actual_ref_set & bom_external_ref_set)
        bom_contract_board_delta = {
            "in_contract_not_bom_on_board": sorted(contract_ref_set - bom_board_ref_set),
            "in_bom_on_board_not_contract": sorted(bom_board_ref_set - contract_ref_set),
        }

        report["checks"]["source_inventory"] = check_result(
            len(bom_rows) == EXPECTED_BOM_ROWS
            and len({ref for ref in bom_by_ref}) == EXPECTED_TOTAL_BOM_REFS
            and len(bom_board_refs) == EXPECTED_BOARD_REFS
            and len(bom_external_refs) == EXPECTED_EXTERNAL_REFS
            and not duplicate_bom_refs
            and not bom_issues
            and set(bom_columns) >= {"Reference", "Manufacturer part", "Scope", "Package"},
            expected={
                "bom_rows": EXPECTED_BOM_ROWS,
                "total_unique_refs": EXPECTED_TOTAL_BOM_REFS,
                "on_board_refs": EXPECTED_BOARD_REFS,
                "external_refs": EXPECTED_EXTERNAL_REFS,
            },
            actual={
                "bom_rows": len(bom_rows),
                "total_unique_refs": len(bom_by_ref),
                "on_board_refs": len(bom_board_refs),
                "external_refs": len(bom_external_refs),
            },
            duplicate_refs=duplicate_bom_refs,
            issues=bom_issues,
            missing_required_columns=sorted({"Reference", "Manufacturer part", "Scope", "Package"} - set(bom_columns)),
        )

        contract_inventory_ok = (
            len(contract_parts) == EXPECTED_BOARD_REFS
            and len(contract_ref_set) == EXPECTED_BOARD_REFS
            and not contract_duplicates
            and not contract_empty_refs
            and not any(bom_contract_board_delta.values())
        )
        report["checks"]["contract_bom_inventory"] = check_result(
            contract_inventory_ok,
            contract_parts=len(contract_parts),
            unique_contract_refs=len(contract_ref_set),
            bom_on_board_refs=len(bom_board_ref_set),
            duplicate_contract_refs=contract_duplicates,
            empty_contract_ref_indices=contract_empty_refs,
            contract_vs_bom_delta=bom_contract_board_delta,
        )

        netlist_ref_ok = (
            len(enet_components) == EXPECTED_BOARD_REFS
            and len(actual_refs) == EXPECTED_BOARD_REFS
            and actual_ref_set == contract_ref_set
            and not actual_duplicates
            and not no_designator_components
            and not external_refs_in_netlist
        )
        report["checks"]["netlist_reference_set"] = check_result(
            netlist_ref_ok,
            expected_board_components=EXPECTED_BOARD_REFS,
            enet_component_records=len(enet_components),
            unique_designators=len(actual_refs),
            missing_board_refs=missing_actual_refs,
            unexpected_refs=unexpected_actual_refs,
            duplicate_refs=actual_duplicates,
            components_without_designator=no_designator_components,
            external_bom_refs_in_netlist=external_refs_in_netlist,
        )

        reference_audits: list[dict[str, Any]] = []
        mpn_issues: list[dict[str, Any]] = []
        footprint_issues: list[dict[str, Any]] = []
        conversion_issues: list[dict[str, Any]] = []
        pin_coverage_issues: list[dict[str, Any]] = []
        pin_net_issues: list[dict[str, Any]] = []
        pin_details: list[dict[str, Any]] = []
        expected_pin_total = 0
        pin_records_found = 0

        for ref in sorted(contract_ref_set):
            contract_part = contract_by_ref.get(ref, {})
            bom_entries = bom_by_ref.get(ref, [])
            component_candidates = actual_components_by_ref.get(ref, [])
            component = component_candidates[0] if component_candidates else {}
            props = component.get("props", {}) if isinstance(component, dict) else {}

            bom_mpns = sorted({entry["mpn"] for entry in bom_entries})
            bom_mpn = bom_mpns[0] if len(bom_mpns) == 1 else None
            contract_mpn = contract_part.get("mpn")
            actual_mpn = props.get("Manufacturer Part")
            actual_footprint_id = props.get("Footprint")
            actual_footprint_name = props.get("FootprintName")
            convert_to_pcb = str(props.get("Convert to PCB", "")).strip().casefold()

            if len(bom_entries) != 1 or len(bom_mpns) != 1 or not nonempty(bom_mpn):
                mpn_issues.append({"ref": ref, "issue": "BOM中MPN缺失或不唯一", "bom_mpn_values": bom_mpns})
            if contract_mpn != bom_mpn:
                mpn_issues.append({"ref": ref, "issue": "连接契约MPN与BOM不精确相同", "contract_mpn": contract_mpn, "bom_mpn": bom_mpn})
            if actual_mpn != bom_mpn:
                mpn_issues.append({"ref": ref, "issue": "ENET Manufacturer Part与BOM不精确相同", "enet_mpn": actual_mpn, "bom_mpn": bom_mpn})

            if not nonempty(actual_footprint_id) or not nonempty(actual_footprint_name):
                footprint_issues.append({
                    "ref": ref,
                    "footprint_id": actual_footprint_id,
                    "footprint_name": actual_footprint_name,
                })
            if convert_to_pcb not in {"yes", "true", "1"}:
                conversion_issues.append({"ref": ref, "convert_to_pcb": props.get("Convert to PCB")})

            reference_audits.append({
                "ref": ref,
                "contract_mpn": contract_mpn,
                "bom_mpn": bom_mpn,
                "enet_manufacturer_part": actual_mpn,
                "mpn_exact_match": contract_mpn == bom_mpn == actual_mpn,
                "footprint_id": actual_footprint_id,
                "footprint_name": actual_footprint_name,
                "footprint_nonempty": nonempty(actual_footprint_id) and nonempty(actual_footprint_name),
                "convert_to_pcb": props.get("Convert to PCB"),
            })

            expected_pins = contract_part.get("pins", {})
            expected_pins = expected_pins if isinstance(expected_pins, dict) else {}
            expected_pin_total += len(expected_pins)
            pin_info = component.get("pinInfoMap", {}) if isinstance(component, dict) else {}
            if not isinstance(pin_info, dict):
                pin_info = {}

            actual_pin_map: dict[str, dict[str, Any]] = {}
            actual_pin_duplicates: list[str] = []
            invalid_pin_records: list[str] = []
            for key, value in pin_info.items():
                if not isinstance(value, dict):
                    invalid_pin_records.append(str(key))
                    continue
                number = str(value.get("number", key)).strip()
                if not number:
                    invalid_pin_records.append(str(key))
                    continue
                if number in actual_pin_map:
                    actual_pin_duplicates.append(number)
                else:
                    actual_pin_map[number] = value

            expected_pin_numbers = {str(pin).strip() for pin in expected_pins}
            actual_pin_numbers = set(actual_pin_map)
            missing_pins = sorted(expected_pin_numbers - actual_pin_numbers, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))
            extra_pins = sorted(actual_pin_numbers - expected_pin_numbers, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))
            if missing_pins or extra_pins or actual_pin_duplicates or invalid_pin_records:
                pin_coverage_issues.append({
                    "ref": ref,
                    "missing_pins": missing_pins,
                    "extra_pins": extra_pins,
                    "duplicate_actual_pin_numbers": sorted(set(actual_pin_duplicates)),
                    "invalid_pin_records": invalid_pin_records,
                })

            for pin_number in sorted(expected_pins, key=lambda item: (not str(item).isdigit(), int(item) if str(item).isdigit() else str(item))):
                expected_net = expected_pins[pin_number]
                actual_pin = actual_pin_map.get(str(pin_number).strip())
                if actual_pin is not None:
                    pin_records_found += 1
                actual_net_present = actual_pin is not None and "net" in actual_pin
                actual_net = actual_pin.get("net") if actual_net_present else None
                expected_upper = normalized_net(expected_net)
                actual_upper = normalized_net(actual_net)

                if actual_pin is None:
                    pin_status = "FAIL"
                    issue = "缺少网表引脚记录"
                elif expected_upper is None:
                    # 契约NC只接受网表net字段为空或空字符串，不能接到命名网络。
                    is_empty_nc = actual_net_present and (
                        actual_net is None or (isinstance(actual_net, str) and not actual_net.strip())
                    )
                    pin_status = "PASS" if is_empty_nc else "FAIL"
                    issue = None if pin_status == "PASS" else "契约为NC但网表net字段缺失或非空"
                else:
                    pin_status = "PASS" if expected_upper == actual_upper else "FAIL"
                    issue = None if pin_status == "PASS" else "网名不匹配"

                detail = {
                    "ref": ref,
                    "pin": str(pin_number),
                    "expected_net": expected_net,
                    "enet_net": actual_net,
                    "expected_upper": expected_upper,
                    "enet_upper": actual_upper,
                    "enet_net_field_present": actual_net_present,
                    "status": pin_status,
                }
                if issue:
                    detail["issue"] = issue
                    pin_net_issues.append(detail.copy())
                pin_details.append(detail)

        report["references"] = reference_audits
        report["pins"] = pin_details
        report["checks"]["mpn_exact"] = check_result(
            not mpn_issues,
            checked_board_refs=len(contract_ref_set),
            issues=mpn_issues,
        )
        report["checks"]["footprints_present"] = check_result(
            not footprint_issues,
            checked_board_refs=len(contract_ref_set),
            fields_checked=["Footprint", "FootprintName"],
            missing_footprints=footprint_issues,
        )
        report["checks"]["board_convert_to_pcb"] = check_result(
            not conversion_issues,
            checked_board_refs=len(contract_ref_set),
            refs_not_convertible=conversion_issues,
        )
        report["checks"]["external_parts_excluded"] = check_result(
            not external_refs_in_netlist,
            external_bom_ref_count=len(bom_external_ref_set),
            external_refs_in_enet=external_refs_in_netlist,
        )
        report["checks"]["pin_coverage"] = check_result(
            not pin_coverage_issues
            and expected_pin_total == len(pin_details)
            and pin_records_found == expected_pin_total,
            expected_contract_pins=expected_pin_total,
            enet_pin_records_found=pin_records_found,
            missing_extra_or_duplicate_pin_records=pin_coverage_issues,
        )
        report["checks"]["pin_nets_and_nc"] = check_result(
            not pin_net_issues,
            pins_compared=len(pin_details),
            comparison="网名去除首尾空白后按upper比较；契约null只接受ENET net缺失/null/空字符串。",
            mismatches=pin_net_issues,
        )

        report["summary"] = {
            "contract_board_refs": len(contract_ref_set),
            "bom_rows": len(bom_rows),
            "bom_total_refs": len(bom_by_ref),
            "bom_board_refs": len(bom_board_ref_set),
            "bom_external_refs": len(bom_external_ref_set),
            "enet_components": len(enet_components),
            "enet_unique_refs": len(actual_refs),
            "expected_pins": expected_pin_total,
            "pins_compared": len(pin_details),
            "netlist_sha256": report["inputs"]["netlist_sha256"],
        }

    except Exception as exc:  # 输入不存在、编码/JSON/CSV损坏时仍写出失败证据。
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        report["checks"]["input_read_and_parse"] = check_result(False, error=report["fatal_error"])

    report["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["status"] = "PASS" if report["checks"] and all(item.get("status") == "PASS" for item in report["checks"].values()) else "FAIL"
    report["conclusion"] = (
        "当前ENET与连接契约/BOM的网表逐脚核对通过；此结果不代表DRC、PCB、实物或整机验收。"
        if report["status"] == "PASS"
        else "当前ENET逐脚核对存在失败项；请查看checks、references及pins中的差异。"
    )

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {OUTPUT_PATH}")
    print(f"ENET SHA256: {report.get('inputs', {}).get('netlist_sha256', 'unavailable')}")
    for check_id, result in report["checks"].items():
        print(f"{result.get('status', 'FAIL')} {check_id}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
