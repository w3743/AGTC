#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原生器件实例属性覆盖规则的只读回归检查。"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = ROOT / "元件封装库" / "工具" / "更新元件库.py"
API_EVIDENCE_PATH = ROOT / "FOC驱动模块" / "检查" / "三板共用修复" / "修改前API器件.json"
REPORT_PATH = Path(__file__).resolve().with_name("解析属性覆盖回归.json")
STRUCTURE_FIELDS = {"Symbol", "Footprint"}
PROCUREMENT_FIELDS = {"Manufacturer", "Manufacturer Part", "Supplier", "Supplier Part", "Value"}


def load_generator():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("update_component_library", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入生成器：{GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_synthetic_rules(generator: Any) -> list[str]:
    carrier = {
        "Manufacturer": "Carrier Maker",
        "Manufacturer Part": "CARRIER-123",
        "Supplier": "LCSC",
        "Supplier Part": "C123456",
        "Value": "carrier value",
        "Other": "carrier-only",
        "Symbol": "symbol-default",
        "Footprint": "footprint-default",
    }

    # 键缺失时继承；空采购/实例数据显式清除；非空实例值覆盖载体默认。
    inherited = generator.effective_attributes(carrier, {"OtherInstance": "instance-only"})
    assert inherited["Manufacturer Part"] == "CARRIER-123"
    assert inherited["Other"] == "carrier-only"

    cleared = generator.effective_attributes(
        carrier,
        {
            "Manufacturer": "",
            "Manufacturer Part": None,
            "Supplier": "  ",
            "Supplier Part": [],
            "Value": "",
            "Symbol": "",
            "Footprint": None,
        },
    )
    for key in ("Manufacturer", "Manufacturer Part", "Supplier", "Supplier Part", "Value"):
        assert key in cleared and not generator.is_nonempty(cleared[key]), (key, cleared.get(key))
    assert cleared["Symbol"] == "symbol-default"
    assert cleared["Footprint"] == "footprint-default"

    overridden = generator.effective_attributes(
        carrier,
        {
            "Manufacturer": "Actual Maker",
            "Manufacturer Part": "ACTUAL-456",
            "Supplier Part": "C654321",
            "Value": "actual value",
            "Symbol": "symbol-instance",
            "Footprint": "footprint-instance",
        },
    )
    assert overridden["Manufacturer"] == "Actual Maker"
    assert overridden["Manufacturer Part"] == "ACTUAL-456"
    assert overridden["Supplier Part"] == "C654321"
    assert overridden["Value"] == "actual value"
    assert overridden["Symbol"] == "symbol-instance"
    assert overridden["Footprint"] == "footprint-instance"

    differences = generator.attribute_differences(carrier, cleared)
    for key in ("Manufacturer", "Manufacturer Part", "Supplier", "Supplier Part", "Value"):
        assert key in differences and not generator.is_nonempty(differences[key]["actual"])

    inherited_value = generator.effective_attributes(carrier, {})
    assert generator.mpn_changed_with_value_inherited(True, carrier, {}, inherited_value)
    assert not generator.mpn_changed_with_value_inherited(True, carrier, {"Value": ""}, cleared)
    assert not generator.mpn_changed_with_value_inherited(True, carrier, {"Value": None}, cleared)
    assert not generator.mpn_changed_with_value_inherited(False, carrier, {}, inherited_value)

    component = {"attrs": {"DeviceName": '{"uuid":"carrier-device"}'}}
    assert generator.device_identifier({}, component) == "carrier-device"
    assert generator.device_identifier({"Device": ""}, component) is None
    return [
        "缺失的实例键继承DEVICE默认值",
        "Manufacturer/Manufacturer Part/Supplier/Supplier Part/Value的显式空值清除载体值",
        "非空实例值覆盖载体值",
        "空Symbol/Footprint继续回退到DEVICE结构绑定",
        "Device键缺失回退DeviceName，显式空Device不回退",
        "显式清除值出现在属性差异记录中",
        "MPN变化提示仅对实例缺少Value键且有效Value继承载体的情形成立",
    ]


def check_live_sources(generator: Any) -> dict[str, Any]:
    sources, config_hash, config_description = generator.source_config()
    parsed_sources = []
    source_evidence = []
    for item in sources:
        source_path = generator.resolve_source(item["path"])
        parsed = generator.parse_source(source_path, item["module"], item["path"])
        assert parsed["sha256_before"] == parsed["sha256_after_parse"], item["path"]
        parsed_sources.append(parsed)
        source_evidence.append(
            {
                "module": item["module"],
                "path": item["path"],
                "sha256_before": parsed["sha256_before"],
                "sha256_after_parse": parsed["sha256_after_parse"],
                "unchanged_during_parse": True,
            }
        )

    module_counts: dict[str, Any] = {}
    explicit_empty_impact: dict[str, Any] = {}
    all_instances = []
    for parsed in parsed_sources:
        instances, counts = generator.get_components(parsed, {})
        module_counts[parsed["module"]] = counts
        all_instances.extend(instances)

        impacted_refs: set[str] = set()
        cleared_keys: Counter[str] = Counter()
        procurement_cleared: Counter[str] = Counter()
        for instance in instances:
            instance_attrs = instance["attrs"]["instance"]
            device_attrs = instance["attrs"]["device"]
            for key, actual in instance_attrs.items():
                if key in STRUCTURE_FIELDS or key not in device_attrs:
                    continue
                if not generator.is_nonempty(actual) and generator.is_nonempty(device_attrs[key]):
                    impacted_refs.add(instance["key"])
                    cleared_keys[key] += 1
                    if key in PROCUREMENT_FIELDS:
                        procurement_cleared[key] += 1
        explicit_empty_impact[parsed["module"]] = {
            "affected_instance_count": len(impacted_refs),
            "cleared_attribute_count": sum(cleared_keys.values()),
            "cleared_attribute_keys": dict(sorted(cleared_keys.items())),
            "procurement_field_cleared_counts": dict(sorted(procurement_cleared.items())),
        }

    assert len(sources) == 3, f"来源数量变更：{len(sources)}"
    assert len(all_instances) == sum(row["bom_instance_count"] for row in module_counts.values())
    for module, counts in module_counts.items():
        assert counts["missing_device_binding_count"] == 0, (module, counts)
        assert counts["missing_symbol_binding_count"] == 0, (module, counts)
        assert counts["missing_footprint_binding_count"] == 0, (module, counts)

    foc_rows = [row for row in all_instances if row["module"] == "FOC驱动模块" and row["ref"] == "J_BRAKE_R"]
    assert len(foc_rows) == 1, f"J_BRAKE_R实例数量异常：{len(foc_rows)}"
    j_brake = foc_rows[0]
    api_doc = json.loads(API_EVIDENCE_PATH.read_text(encoding="utf-8-sig"))
    api_rows = [row for row in api_doc.get("rows", []) if row.get("ref") == "J_BRAKE_R"]
    assert len(api_rows) == 1, f"API快照中J_BRAKE_R数量异常：{len(api_rows)}"
    api_row = api_rows[0]

    expected_api = {
        "mpn": api_row.get("mpn") or None,
        "manufacturer": api_row.get("manufacturer"),
        "supplier": api_row.get("supplier") or None,
        "supplier_part": api_row.get("supplierId") or None,
        "pcb": "yes" if api_row.get("pcb") is True else "no" if api_row.get("pcb") is False else "unknown",
        "symbol_id": (api_row.get("symbol") or {}).get("uuid"),
        "footprint_id": (api_row.get("footprint") or {}).get("uuid"),
    }
    actual_api = {
        "mpn": j_brake.get("mpn"),
        "manufacturer": j_brake.get("manufacturer"),
        "supplier": j_brake["attrs"]["effective"].get("Supplier") or None,
        "supplier_part": j_brake.get("supplier_part"),
        "pcb": j_brake.get("pcb"),
        "symbol_id": j_brake.get("symbol_id"),
        "footprint_id": j_brake.get("footprint_id"),
    }
    assert actual_api == expected_api, {"expected": expected_api, "actual": actual_api}
    assert j_brake["mpn_diff"] is True, "载体型号与J_BRAKE_R实际空MPN应被标为差异"

    return {
        "source_config": {
            "description": config_description,
            "sha256": config_hash,
            "source_count": len(sources),
        },
        "sources": source_evidence,
        "module_counts": module_counts,
        "total_bom_instances": len(all_instances),
        "explicit_empty_overrides": explicit_empty_impact,
        "j_brake_r": {
            "actual_parsed": actual_api,
            "official_api_snapshot_expected": expected_api,
            "api_evidence_path": str(API_EVIDENCE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "mpn_diff_from_carrier": j_brake["mpn_diff"],
            "carrier_mpn": j_brake.get("carrier_mpn"),
            "assembly_parts": j_brake["attrs"]["effective"].get("Assembly Parts"),
            "matches_api_snapshot": True,
        },
    }


def main() -> int:
    compile(GENERATOR_PATH.read_text(encoding="utf-8-sig"), str(GENERATOR_PATH), "exec")
    generator = load_generator()
    synthetic_checks = check_synthetic_rules(generator)
    source_checks = check_live_sources(generator)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "只读解析回归；不运行同步入口、不写索引/资产或工程源文件",
        "generator_path": str(GENERATOR_PATH.relative_to(ROOT)).replace("\\", "/"),
        "generator_sha256": generator.sha256_file(GENERATOR_PATH),
        "synthetic_regression_checks": synthetic_checks,
        **source_checks,
        "limitations": [
            "验证属性覆盖、绑定保留与源文件解析一致性，不等同于电气、机械、PCB或实物验收。",
            "解析回归仅写入本目录的本报告，不执行正式索引同步。",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(REPORT_PATH), "total_bom_instances": report["total_bom_instances"], "module_counts": report["module_counts"], "explicit_empty_overrides": report["explicit_empty_overrides"], "j_brake_r": report["j_brake_r"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
