#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从显式配置的现行原理图生成原生器件清单与可复用原生片段。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LIB_ROOT = ROOT / "元件封装库"
CONFIG_PATH = LIB_ROOT / "来源配置.json"
INDEX_PATH = LIB_ROOT / "索引" / "元件总表.json"
ASSET_INDEX_PATH = LIB_ROOT / "资产索引.json"
EVIDENCE_PATH = LIB_ROOT / "检查" / "同步证据.json"

# 配置文件缺失时的显式安全来源；不递归搜索工程、归档或临时目录。
DEFAULT_SOURCES = [
    {
        "module": "BMS模块",
        "path": "BMS模块/sch/Schematic1/P1.esch2"
    },
    {
        "module": "FOC驱动模块",
        "path": "FOC驱动模块/eda工程文件/FOC_G431_Driver/sch/Schematic1/P1.esch2"
    },
    {
        "module": "稳压模块",
        "path": "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/01 主功率与超容.esch2"
    },
    {
        "module": "稳压模块",
        "path": "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/02 门极驱动与硬保护.esch2"
    },
    {
        "module": "稳压模块",
        "path": "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/03 模拟采样.esch2"
    },
    {
        "module": "稳压模块",
        "path": "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/04 控制通信与辅助供电.esch2"
    },
    {
        "module": "水泵驱动模块",
        "path": "水泵驱动模块/eda工程文件/水泵驱动/sch/Schematic1/P1.esch2"
    }
]

ASSET_TYPES = ("SYMBOL", "FOOTPRINT", "DEVICE")
# EasyEDA's ESCH symbol document-type IDs (also ELIB_SymbolType values).
# Net flags, net ports, and off-page connectors are schematic connectivity
# primitives, not purchasable component instances, even if copied ATTRs say yes.
NON_COMPONENT_SYMBOL_DOC_TYPES = frozenset({18, 19, 25})
SOURCE_COUNT_FIELDS = (
    "component_count",
    "bom_instance_count",
    "excluded_non_bom_count",
    "missing_device_binding_count",
    "missing_symbol_binding_count",
    "missing_footprint_binding_count",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        if not value:
            return ""
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalized_key(value: Any) -> str:
    """只做Unicode兼容、大小写和空白规整；保留型号中的标点与字符。"""
    if not is_nonempty(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def source_config() -> tuple[list[dict[str, str]], str | None, str]:
    if not CONFIG_PATH.exists():
        return copy.deepcopy(DEFAULT_SOURCES), None, "内置显式来源列表（配置文件缺失）"
    config_hash = sha256_file(CONFIG_PATH)
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"来源配置无法解析：{CONFIG_PATH}：{exc}") from exc
    sources = config.get("native_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("来源配置 native_sources 必须是非空列表")
    clean: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in sources:
        if not isinstance(item, dict) or not item.get("module") or not item.get("path"):
            raise ValueError("native_sources 每项必须包含 module 与 path")
        module, relative = str(item["module"]).strip(), str(item["path"]).strip()
        path_obj = Path(relative)
        if path_obj.is_absolute() or re.match(r"^[A-Za-z]:", relative):
            raise ValueError(f"来源必须使用工程根目录相对路径：{relative}")
        path_parts = [part.casefold() for part in path_obj.parts]
        if any(part in {"归档", "archive", "archives", "tmp", "temp", "temporary"} for part in path_parts):
            raise ValueError(f"禁止将归档或临时文件列为现行来源：{relative}")
        key = (module, relative.replace("\\", "/"))
        if key in seen:
            raise ValueError(f"来源重复：{module} / {relative}")
        seen.add(key)
        clean.append({"module": module, "path": key[1]})
    return clean, config_hash, "元件封装库/来源配置.json"


def resolve_source(relative: str) -> Path:
    target = (ROOT / Path(relative)).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"来源路径超出电力工程目录：{relative}") from exc
    if not target.is_file():
        raise FileNotFoundError(f"现行原理图不存在：{relative}")
    return target


def parse_record(line: str, location: str) -> tuple[dict[str, Any], dict[str, Any]]:
    header_text, separator, payload = line.partition("||")
    if not separator:
        raise ValueError(f"记录缺少 || 分隔符：{location}")
    if payload.endswith("|"):
        payload = payload[:-1]
    try:
        header = json.loads(header_text)
        value = json.loads(payload) if payload.strip() else {}
    except Exception as exc:
        raise ValueError(f"原理图记录JSON无效：{location}：{exc}") from exc
    if not isinstance(header, dict) or not isinstance(value, dict):
        raise ValueError(f"原理图记录必须是JSON对象：{location}")
    return header, value


def parse_source(path: Path, module: str, relative: str) -> dict[str, Any]:
    digest_before = sha256_file(path)
    text = path.read_text(encoding="utf-8-sig")
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        header, value = parse_record(line, f"{relative}:{line_no}")
        if header.get("type") == "DOCHEAD":
            if current is not None:
                sections.append(current)
            current = {"dochead": value, "meta": {}, "records": [], "raw_lines": [line]}
            continue
        if current is None:
            raise ValueError(f"DOCHEAD之前出现原理图记录：{relative}:{line_no}")
        current["records"].append({"header": header, "value": value, "raw": line})
        current["raw_lines"].append(line)
        if header.get("type") == "META":
            current["meta"] = value
    if current is not None:
        sections.append(current)
    digest_after = sha256_file(path)
    if digest_before != digest_after:
        raise RuntimeError(f"解析期间原理图SHA256变化：{relative}")
    for section in sections:
        section["doc_type"] = section.get("dochead", {}).get("docType", "")
        section["local_uuid"] = section.get("dochead", {}).get("uuid")
        section["title"] = section.get("meta", {}).get("title", "")
        source = section.get("meta", {}).get("source", "")
        if isinstance(source, str) and source:
            pieces = source.split("|")
            section["official_uuid"] = pieces[0] or None
            section["library_uuid"] = pieces[1] if len(pieces) > 1 and pieces[1] else None
        else:
            section["official_uuid"] = None
            section["library_uuid"] = None
        section["content_bytes"] = ("\n".join(section["raw_lines"]) + "\n").encode("utf-8")
        section["content_sha256"] = sha256_bytes(section["content_bytes"])
    pages = [section for section in sections if section["doc_type"] == "SCH_PAGE"]
    if not pages:
        raise ValueError(f"来源未找到 SCH_PAGE：{relative}")
    return {
        "module": module,
        "path": relative,
        "path_obj": path,
        "sha256_before": digest_before,
        "sha256_after_parse": digest_after,
        "sections": sections,
        "pages": pages,
    }


def section_map(sections: list[dict[str, Any]], doc_type: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        if section["doc_type"] == doc_type and is_nonempty(section.get("local_uuid")):
            result[str(section["local_uuid"])].append(section)
    return result


def lookup_asset(
    doc_type: str,
    identifier: Any,
    by_local_uuid: dict[str, list[dict[str, Any]]],
    by_official_uuid: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if not is_nonempty(identifier):
        return {"id": identifier, "type": doc_type, "status": "id_missing", "sha256": None, "path": None}
    key = str(identifier)
    matches = by_local_uuid.get(key) or by_official_uuid.get(key) or []
    if not matches:
        return {"id": identifier, "type": doc_type, "status": "missing", "sha256": None, "path": None}
    hashes = sorted({item["content_sha256"] for item in matches})
    metadata = {
        "id": identifier,
        "type": doc_type,
        "title": matches[0].get("title", "") if len({item.get("title", "") for item in matches}) <= 1 else None,
        "titles": sorted({str(item.get("title", "")) for item in matches if item.get("title")}),
        "source_uuid": matches[0].get("official_uuid") if len({item.get("official_uuid") for item in matches}) <= 1 else None,
        "library_uuid": matches[0].get("library_uuid") if len({item.get("library_uuid") for item in matches}) <= 1 else None,
        "sha256": hashes[0] if len(hashes) == 1 else None,
        "path": f"元件封装库/原生片段/{doc_type}/{hashes[0]}.jsonl" if len(hashes) == 1 else None,
        "status": "present" if len(hashes) == 1 else "ambiguous_variant",
        "candidate_sha256": hashes,
        "candidate_local_uuids": sorted({str(item.get("local_uuid", "")) for item in matches}),
    }
    return metadata


def is_non_component_symbol(
    identifier: Any,
    by_local_uuid: dict[str, list[dict[str, Any]]],
    by_official_uuid: dict[str, list[dict[str, Any]]],
) -> bool:
    """Identify standard net-flag/net-port symbols from their native symbol type.

    EasyEDA stores these as COMPONENT instances in ESCH, so their instance BOM
    attributes alone are not enough to decide whether they represent a physical
    part. The referenced SYMBOL document's META.docType is the authoritative
    ELIB_SymbolType ID (18=NET_FLAG, 19=NET_PORT, 25=OFF_PAGE_CONNECTOR).
    """
    if not is_nonempty(identifier):
        return False
    sections = by_local_uuid.get(str(identifier)) or by_official_uuid.get(str(identifier)) or []
    if not sections:
        return False
    doc_types = [section.get("meta", {}).get("docType") for section in sections]
    return bool(doc_types) and all(doc_type in NON_COMPONENT_SYMBOL_DOC_TYPES for doc_type in doc_types)


def extract_instance_attributes(records: list[dict[str, Any]], component_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_records: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {}
    for record in records:
        if record["header"].get("type") != "ATTR":
            continue
        value = record["value"]
        if str(value.get("parentId", "")) != str(component_id):
            continue
        raw_records.append({"header": record["header"], "value": value})
        key = value.get("key")
        if is_nonempty(key):
            attrs[str(key)] = value.get("value")
    return attrs, raw_records


def first_nonempty(attrs: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if is_nonempty(attrs.get(key)):
            return attrs[key]
    return None


def device_identifier(instance_attrs: dict[str, Any], component: dict[str, Any]) -> Any:
    # ATTR中存在Device键即视为显式指定；空值不再回退到组件载体DeviceName。
    if "Device" in instance_attrs:
        return first_nonempty(instance_attrs, "Device")
    component_attrs = component.get("attrs", {})
    device_name = component_attrs.get("DeviceName") if isinstance(component_attrs, dict) else None
    if isinstance(device_name, str):
        try:
            parsed = json.loads(device_name)
            if is_nonempty(parsed.get("uuid")):
                return parsed["uuid"]
        except Exception:
            pass
    return None


def effective_attributes(device_attrs: dict[str, Any], instance_attrs: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(device_attrs)
    for key, value in instance_attrs.items():
        # Symbol与Footprint是CAD结构绑定字段；空值表示沿用DEVICE默认绑定。
        if key in {"Symbol", "Footprint"} and not is_nonempty(value):
            continue
        # 其他实例ATTR只要显式存在该键就覆盖DEVICE META；空值用于清除载体默认。
        effective[key] = value
    return effective


def attribute_differences(device_attrs: dict[str, Any], effective: dict[str, Any]) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for key in sorted(set(device_attrs) | set(effective)):
        actual = effective.get(key)
        carrier = device_attrs.get(key)
        if actual != carrier:
            differences[key] = {"carrier": carrier, "actual": actual}
    return differences


def mpn_changed_with_value_inherited(
    mpn_diff: bool, device_attrs: dict[str, Any], instance_attrs: dict[str, Any], effective: dict[str, Any]
) -> bool:
    """Keep the legacy metric: the effective DEVICE.Value was directly inherited."""
    if not mpn_diff or "Value" in instance_attrs:
        return False
    carrier_value = first_nonempty(device_attrs, "Value")
    actual_value = first_nonempty(effective, "Value")
    return is_nonempty(carrier_value) and actual_value == carrier_value


def explicit_instance_value(instance_attrs: dict[str, Any]) -> tuple[Any, str | None]:
    """Return a BOM-confirmed instance value, or a concrete instance Name label.

    Formula-valued Name strings such as ``={Manufacturer Part}`` are not values.
    """
    value = first_nonempty(instance_attrs, "Value")
    name = first_nonempty(instance_attrs, "Name")
    literal_name = isinstance(name, str) and name.strip() and not name.strip().startswith("=")
    actual_mpn = first_nonempty(instance_attrs, "Manufacturer Part")
    if is_nonempty(value):
        # Some BOM imports place the MPN in Value while Name retains the real
        # component label (for example, 100nF/100V). Prefer that explicit label.
        if literal_name and normalized_key(value) == normalized_key(actual_mpn) and normalized_key(name) != normalized_key(actual_mpn):
            return name, "instance.Name (Value duplicates MPN)"
        return value, "instance.Value"
    if literal_name:
        return name, "instance.Name"
    return None, None


def resolve_actual_value(
    mpn_diff: bool, device_attrs: dict[str, Any], instance_attrs: dict[str, Any]
) -> tuple[Any, str | None]:
    """Use explicit BOM instance data; inherit DEVICE.Value only for an unchanged MPN."""
    value, source = explicit_instance_value(instance_attrs)
    if is_nonempty(value):
        return value, source
    carrier_value = first_nonempty(device_attrs, "Value")
    if not mpn_diff and is_nonempty(carrier_value):
        return carrier_value, "DEVICE.Value (MPN unchanged)"
    return None, None


def mpn_changed_without_actual_value(
    mpn_diff: bool, device_attrs: dict[str, Any], instance_attrs: dict[str, Any]
) -> bool:
    """Flag changed MPN when only a carrier Value exists and no explicit part value does."""
    return bool(
        mpn_diff
        and is_nonempty(first_nonempty(device_attrs, "Value"))
        and not is_nonempty(explicit_instance_value(instance_attrs)[0])
    )


def part_key_for(module: str, designator: str, manufacturer: Any, mpn: Any, instance_key: str) -> tuple[str, str, str]:
    norm_mpn = normalized_key(mpn)
    norm_manufacturer = normalized_key(manufacturer)
    if norm_mpn and norm_manufacturer:
        raw = f"MODEL\0{norm_manufacturer}\0{norm_mpn}"
        return "MODEL-" + sha256_bytes(raw.encode("utf-8"))[:16], norm_manufacturer, norm_mpn
    # 无MPN或无制造商时保留为板级独立项，避免未知来源被自动合并。
    raw = f"INSTANCE\0{module}\0{designator}\0{instance_key}"
    return "INSTANCE-" + sha256_bytes(raw.encode("utf-8"))[:16], norm_manufacturer, norm_mpn


def get_components(parsed: dict[str, Any], device_map: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sections = parsed["sections"]
    by_symbol = section_map(sections, "SYMBOL")
    by_footprint = section_map(sections, "FOOTPRINT")
    by_device = section_map(sections, "DEVICE")
    by_symbol_official: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_footprint_official: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_device_official: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        if section["doc_type"] == "SYMBOL" and section.get("official_uuid"):
            by_symbol_official[section["official_uuid"]].append(section)
        elif section["doc_type"] == "FOOTPRINT" and section.get("official_uuid"):
            by_footprint_official[section["official_uuid"]].append(section)
        elif section["doc_type"] == "DEVICE" and section.get("official_uuid"):
            by_device_official[section["official_uuid"]].append(section)

    instances: list[dict[str, Any]] = []
    excluded = 0
    missing_device = missing_symbol = missing_footprint = 0
    component_count = 0
    for page in parsed["pages"]:
        for record in page["records"]:
            if record["header"].get("type") != "COMPONENT":
                continue
            component_count += 1
            component_id = str(record["header"].get("id", ""))
            component = record["value"]
            instance_attrs, attr_records = extract_instance_attributes(page["records"], component_id)
            device_id = device_identifier(instance_attrs, component)
            device_candidates = by_device.get(str(device_id), []) if is_nonempty(device_id) else []
            device_section = device_candidates[0] if device_candidates else None
            device_attrs = device_section.get("meta", {}).get("attributes", {}) if device_section else {}
            if not isinstance(device_attrs, dict):
                device_attrs = {}
            effective = effective_attributes(device_attrs, instance_attrs)
            symbol_id = first_nonempty(effective, "Symbol")
            if not is_nonempty(symbol_id) and device_section:
                symbol_id = first_nonempty(device_attrs, "Symbol")
            if is_non_component_symbol(symbol_id, by_symbol, by_symbol_official):
                excluded += 1
                continue
            bom_flag = display(effective.get("Add into BOM")).strip().casefold()
            if bom_flag != "yes":
                excluded += 1
                continue

            designator = display(first_nonempty(effective, "Designator") or component.get("partId", "")).strip()
            part_id = display(component.get("partId", ""))
            ref_key = f"{parsed['module']}::{designator or part_id or component_id}"
            actual_mpn = first_nonempty(effective, "Manufacturer Part")
            actual_manufacturer = first_nonempty(effective, "Manufacturer")
            carrier_mpn = first_nonempty(device_attrs, "Manufacturer Part")
            carrier_manufacturer = first_nonempty(device_attrs, "Manufacturer")
            mpn_diff = normalized_key(actual_mpn) != normalized_key(carrier_mpn)
            if not device_section:
                missing_device += 1

            footprint_id = first_nonempty(instance_attrs, "Footprint")
            if not is_nonempty(footprint_id):
                footprint_id = first_nonempty(device_attrs, "Footprint")
            if not is_nonempty(footprint_id):
                component_attrs = component.get("attrs", {})
                fp_name = component_attrs.get("FootprintName") if isinstance(component_attrs, dict) else None
                if isinstance(fp_name, str):
                    try:
                        footprint_id = json.loads(fp_name).get("uuid")
                    except Exception:
                        footprint_id = None
            symbol_asset = lookup_asset("SYMBOL", symbol_id, by_symbol, by_symbol_official)
            device_asset = lookup_asset("DEVICE", device_id, by_device, by_device_official)
            footprint_asset = lookup_asset("FOOTPRINT", footprint_id, by_footprint, by_footprint_official)
            missing_symbol += symbol_asset["status"] in {"missing", "id_missing"}
            missing_footprint += footprint_asset["status"] in {"missing", "id_missing"}
            page_uuid = page.get("local_uuid")
            page_title = page.get("title", "")
            instance_key = ref_key
            part_key, manufacturer_norm, mpn_norm = part_key_for(
                parsed["module"], designator, actual_manufacturer, actual_mpn, instance_key
            )
            carrier_value = first_nonempty(device_attrs, "Value")
            effective_value_raw = first_nonempty(effective, "Value")
            actual_value, actual_value_source = resolve_actual_value(
                mpn_diff, device_attrs, instance_attrs
            )
            instance_value_raw = instance_attrs.get("Value")
            legacy_value_inherited = mpn_changed_with_value_inherited(
                mpn_diff, device_attrs, instance_attrs, effective
            )
            missing_actual_value = mpn_changed_without_actual_value(mpn_diff, device_attrs, instance_attrs)
            differences = attribute_differences(device_attrs, effective)
            instances.append(
                {
                    "key": instance_key,
                    "part_key": part_key,
                    "module": parsed["module"],
                    "ref": designator,
                    "designator": designator,
                    "source_ref": parsed["path"],
                    "page_title": page_title,
                    "page_uuid": page_uuid,
                    "component_id": component_id,
                    "part_id": part_id,
                    "bom": "yes",
                    "pcb": (display(effective.get("Convert to PCB")).strip().casefold() or "unknown"),
                    "manufacturer": actual_manufacturer,
                    "manufacturer_normalized": manufacturer_norm or None,
                    "mpn": actual_mpn,
                    "mpn_normalized": mpn_norm or None,
                    "carrier_manufacturer": carrier_manufacturer,
                    "carrier_mpn": carrier_mpn,
                    "mpn_diff": mpn_diff,
                    "supplier_part": first_nonempty(effective, "Supplier Part"),
                    "device_value": actual_value,
                    "actual_value": actual_value,
                    "actual_value_source": actual_value_source,
                    "carrier_value": carrier_value,
                    "effective_value_raw": effective_value_raw,
                    "instance_value_raw": instance_value_raw,
                    "mpn_changed_with_value_inherited": legacy_value_inherited,
                    "mpn_changed_without_actual_value": missing_actual_value,
                    "device_attribute_differences": differences,
                    "device_id": device_id,
                    "symbol_id": symbol_id,
                    "footprint_id": footprint_id,
                    "device_asset": device_asset,
                    "symbol_asset": symbol_asset,
                    "footprint_asset": footprint_asset,
                    "device_local_uuid": device_section.get("local_uuid") if device_section else None,
                    "device_source_uuid": device_section.get("official_uuid") if device_section else None,
                    "device_library_uuid": device_section.get("library_uuid") if device_section else None,
                    "attrs": {
                        "component": copy.deepcopy(component.get("attrs", {})),
                        "device": copy.deepcopy(device_attrs),
                        "instance": copy.deepcopy(instance_attrs),
                        "effective": copy.deepcopy(effective),
                        "attribute_records": attr_records,
                    },
                    "component_native": {"header": record["header"], "value": component},
                    "references": {
                        "source": parsed["path"],
                        "page": {"title": page_title, "uuid": page_uuid},
                        "component_id": component_id,
                        "part_id": part_id,
                    },
                }
            )
    counts = {
        "component_count": component_count,
        "bom_instance_count": len(instances),
        "excluded_non_bom_count": excluded,
        "missing_device_binding_count": int(missing_device),
        "missing_symbol_binding_count": int(missing_symbol),
        "missing_footprint_binding_count": int(missing_footprint),
    }
    return instances, counts


def unique_module_order(sources: list[dict[str, str]]) -> list[str]:
    """Return module names once each, preserving the configured source order."""
    return list(dict.fromkeys(item["module"] for item in sources))


def aggregate_module_stats(
    source_stats: dict[tuple[str, str], dict[str, Any]], module_order: list[str]
) -> list[dict[str, Any]]:
    """Sum page/source counters into one stable summary row per module."""
    result: list[dict[str, Any]] = []
    for module in module_order:
        counts = [
            values
            for (source_module, _source_path), values in source_stats.items()
            if source_module == module
        ]
        result.append(
            {
                "module": module,
                **{
                    field: sum(int(values.get(field, 0) or 0) for values in counts)
                    for field in SOURCE_COUNT_FIELDS
                },
            }
        )
    return result


def build_source_evidence_rows(
    parsed_sources: list[dict[str, Any]],
    final_source_hashes: list[str],
    source_stats: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build evidence per native source file, not from its module aggregate."""
    if len(parsed_sources) != len(final_source_hashes):
        raise ValueError("source与最终SHA256数量不一致")
    result: list[dict[str, Any]] = []
    for parsed, final_sha in zip(parsed_sources, final_source_hashes):
        key = (parsed["module"], parsed["path"])
        counts = source_stats[key]
        result.append(
            {
                "module": parsed["module"],
                "path": parsed["path"],
                "sha256": parsed["sha256_before"],
                "sha256_after_parse": parsed["sha256_after_parse"],
                "sha256_after_generation": final_sha,
                "unchanged": parsed["sha256_before"] == final_sha,
                "page_count": len(parsed["pages"]),
                "component_count": counts["component_count"],
                "bom_instance_count": counts["bom_instance_count"],
            }
        )
    return result


def build_parts(instances: list[dict[str, Any]], module_order: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        groups[instance["part_key"]].append(instance)
    module_rank = {name: index for index, name in enumerate(module_order)}
    parts: list[dict[str, Any]] = []
    for key, rows in groups.items():
        modules = sorted({row["module"] for row in rows}, key=lambda name: module_rank.get(name, 999))
        refs = [f"{row['module']}:{row['ref']}" for row in sorted(rows, key=lambda row: (module_rank.get(row["module"], 999), row["ref"].casefold()))]
        mfr_values = sorted({display(row.get("manufacturer")) for row in rows if is_nonempty(row.get("manufacturer"))})
        mpn_values = sorted({display(row.get("mpn")) for row in rows if is_nonempty(row.get("mpn"))})
        supplier_parts = sorted({display(row.get("supplier_part")) for row in rows if is_nonempty(row.get("supplier_part"))})
        footprint_ids = sorted({display(row.get("footprint_id")) for row in rows if is_nonempty(row.get("footprint_id"))})
        device_ids = sorted({display(row.get("device_id")) for row in rows if is_nonempty(row.get("device_id"))})
        pcb_counts = Counter(row.get("pcb", "unknown") for row in rows)
        asset_statuses = {
            name: sorted({row[name].get("status", "missing") for row in rows})
            for name in ("device_asset", "symbol_asset", "footprint_asset")
        }
        parts.append(
            {
                "key": key,
                "manufacturer": mfr_values[0] if len(mfr_values) == 1 else mfr_values,
                "mpn": mpn_values[0] if len(mpn_values) == 1 else mpn_values,
                "supplier_parts": supplier_parts,
                "modules": modules,
                "designators": refs,
                "bom_instance_count": len(rows),
                "board_count": len(modules),
                "shared_across_boards": len(modules) > 1,
                "pcb_counts": {key: pcb_counts.get(key, 0) for key in ("yes", "no", "unknown")},
                "device_ids": device_ids,
                "footprint_ids": footprint_ids,
                "asset_statuses": asset_statuses,
                "missing_mpn": not any(is_nonempty(row.get("mpn")) for row in rows),
                "missing_manufacturer": not any(is_nonempty(row.get("manufacturer")) for row in rows),
                "instance_keys": [row["key"] for row in rows],
            }
        )
    return sorted(parts, key=lambda part: (str(part.get("manufacturer", "")).casefold(), str(part.get("mpn", "")).casefold(), part["key"]))


def build_mpn_only_candidates(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """找出跨板MPN相同但制造商字段不一致或缺失的候选，绝不并入严格型号组。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        normalized_mpn = instance.get("mpn_normalized") or normalized_key(instance.get("mpn"))
        if normalized_mpn:
            groups[normalized_mpn].append(instance)

    candidates: list[dict[str, Any]] = []
    for normalized_mpn, rows in groups.items():
        modules = sorted({row["module"] for row in rows})
        manufacturers = {row.get("manufacturer_normalized") or "" for row in rows}
        if len(modules) < 2 or (len(manufacturers) == 1 and "" not in manufacturers):
            continue

        manufacturer_entries: list[dict[str, Any]] = []
        entry_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            raw_manufacturer = display(row.get("manufacturer"))
            manufacturer_norm = row.get("manufacturer_normalized") or ""
            entry_groups[(row["module"], manufacturer_norm, raw_manufacturer)].append(row["ref"])
        for (module, _manufacturer_norm, raw_manufacturer), refs in sorted(
            entry_groups.items(), key=lambda item: (item[0][0].casefold(), item[0][1], item[0][2].casefold())
        ):
            manufacturer_entries.append(
                {
                    "module": module,
                    "manufacturer": raw_manufacturer or None,
                    "designators": sorted(refs, key=str.casefold),
                }
            )
        candidates.append(
            {
                "mpn": display(rows[0].get("mpn")),
                "mpn_normalized": normalized_mpn,
                "modules": modules,
                "board_count": len(modules),
                "bom_instance_count": len(rows),
                "manufacturer_fields": manufacturer_entries,
                "strict_part_keys": sorted({row["part_key"] for row in rows}),
                "instance_keys": sorted(row["key"] for row in rows),
                "note": "仅实际MPN字段跨板相同，制造商字段不一致或缺失；保留各严格型号组，不自动合并，也不代表电气/机械验收。",
            }
        )
    return sorted(candidates, key=lambda item: (item["mpn_normalized"], item["mpn"].casefold()))


def collect_assets(parsed_sources: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    current: dict[tuple[str, str], dict[str, Any]] = {}
    official_variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for parsed in parsed_sources:
        for section in parsed["sections"]:
            doc_type = section["doc_type"]
            if doc_type not in ASSET_TYPES:
                continue
            digest = section["content_sha256"]
            key = (doc_type, digest)
            origin = {
                "module": parsed["module"],
                "source_ref": parsed["path"],
                "doc_uuid": section.get("local_uuid"),
                "source_uuid": section.get("official_uuid"),
                "library_uuid": section.get("library_uuid"),
                "title": section.get("title", ""),
            }
            entry = current.setdefault(
                key,
                {
                    "type": doc_type,
                    "sha256": digest,
                    "path": f"元件封装库/原生片段/{doc_type}/{digest}.jsonl",
                    "titles": [],
                    "source_uuids": [],
                    "library_uuids": [],
                    "origins": [],
                    "current_sources": [],
                    "current_source": True,
                },
            )
            for field, value in (("titles", origin["title"]), ("source_uuids", origin["source_uuid"]), ("library_uuids", origin["library_uuid"])):
                if is_nonempty(value) and value not in entry[field]:
                    entry[field].append(value)
            if origin not in entry["origins"]:
                entry["origins"].append(origin)
            if origin not in entry["current_sources"]:
                entry["current_sources"].append(origin)
            if origin["source_uuid"]:
                official_variants[(doc_type, origin["source_uuid"])].add(digest)
    variants = []
    for (doc_type, official_uuid), hashes in sorted(official_variants.items()):
        if len(hashes) > 1:
            variants.append(
                {
                    "type": doc_type,
                    "source_uuid": official_uuid,
                    "sha256_variants": sorted(hashes),
                    "library_uuids": sorted(
                        {
                            origin["library_uuid"]
                            for (kind, digest), entry in current.items()
                            if kind == doc_type and digest in hashes
                            for origin in entry["origins"]
                            if origin.get("source_uuid") == official_uuid and origin.get("library_uuid")
                        }
                    ),
                }
            )
    return current, variants


def merge_asset_index(current: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    old_assets: dict[tuple[str, str], dict[str, Any]] = {}
    if ASSET_INDEX_PATH.exists():
        try:
            old = json.loads(ASSET_INDEX_PATH.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise ValueError(f"既有资产索引无法解析，停止以免丢失保留资产：{ASSET_INDEX_PATH}：{exc}") from exc
        for entry in old.get("assets", []) if isinstance(old, dict) else []:
            if isinstance(entry, dict) and entry.get("type") and entry.get("sha256"):
                saved = copy.deepcopy(entry)
                saved["current_source"] = False
                saved["current_sources"] = []
                old_assets[(str(saved["type"]), str(saved["sha256"]))] = saved
    for key, entry in current.items():
        if key not in old_assets:
            old_assets[key] = copy.deepcopy(entry)
            continue
        saved = old_assets[key]
        saved["current_source"] = True
        saved["current_sources"] = copy.deepcopy(entry["current_sources"])
        for field in ("titles", "source_uuids", "library_uuids", "origins"):
            existing = saved.setdefault(field, [])
            for value in entry.get(field, []):
                if value not in existing:
                    existing.append(copy.deepcopy(value))
        saved["path"] = entry["path"]
    assets = sorted(old_assets.values(), key=lambda entry: (entry.get("type", ""), entry.get("sha256", "")))
    return {
        "schema_version": 1,
        "description": "按原生块内容SHA256寻址；用于工程追溯与复用，文件不是EasyEDA可直接导入包。",
        "asset_count": len(assets),
        "current_source_asset_count": sum(bool(asset.get("current_source")) for asset in assets),
        "assets": assets,
    }


def write_fragments(parsed_sources: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    current, variants = collect_assets(parsed_sources)
    for (doc_type, digest), entry in current.items():
        section = next(
            section
            for parsed in parsed_sources
            for section in parsed["sections"]
            if section["doc_type"] == doc_type and section["content_sha256"] == digest
        )
        target = LIB_ROOT / "原生片段" / doc_type / f"{digest}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        data = section["content_bytes"]
        if target.exists():
            if sha256_file(target) != digest:
                raise RuntimeError(f"已有原生片段与其内容SHA256不符，拒绝覆盖：{target}")
        else:
            target.write_bytes(data)
        if sha256_file(target) != digest:
            raise RuntimeError(f"原生片段写入后SHA256校验失败：{target}")
    return current, variants


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: display(row.get(field)) for field in fieldnames})


def csv_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "part_key": part["key"],
            "manufacturer": part["manufacturer"],
            "mpn": part["mpn"],
            "bom_instance_count": part["bom_instance_count"],
            "board_count": part["board_count"],
            "modules": "; ".join(part["modules"]),
            "designators": "; ".join(part["designators"]),
            "supplier_parts": "; ".join(part["supplier_parts"]),
            "device_ids": "; ".join(part["device_ids"]),
            "footprint_ids": "; ".join(part["footprint_ids"]),
            "missing_mpn": "yes" if part["missing_mpn"] else "no",
            "missing_manufacturer": "yes" if part["missing_manufacturer"] else "no",
        }
        for part in parts
    ]


def csv_board_use(instances: list[dict[str, Any]], module_order: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[(instance["module"], instance["part_key"])].append(instance)
    rank = {module: index for index, module in enumerate(module_order)}
    rows = []
    for (module, part_key), items in sorted(grouped.items(), key=lambda item: (rank.get(item[0][0], 999), item[0][1])):
        rows.append(
            {
                "module": module,
                "part_key": part_key,
                "quantity": len(items),
                "manufacturer": items[0].get("manufacturer"),
                "mpn": items[0].get("mpn"),
                "designators": "; ".join(item["ref"] for item in items),
                "supplier_parts": "; ".join(sorted({display(item.get("supplier_part")) for item in items if is_nonempty(item.get("supplier_part"))})),
                "pcb_yes": sum(item.get("pcb") == "yes" for item in items),
                "pcb_no": sum(item.get("pcb") == "no" for item in items),
                "pcb_unknown": sum(item.get("pcb") not in {"yes", "no"} for item in items),
                "source_refs": "; ".join(sorted({item["source_ref"] for item in items})),
            }
        )
    return rows


def render_markdown(
    parts: list[dict[str, Any]], instances: list[dict[str, Any]], module_stats: list[dict[str, Any]], variants: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    mpn_only_candidates = build_mpn_only_candidates(instances)
    lines = [
        "# 复用与待核对",
        "",
        "本表仅反映所列现行原生原理图中的型号、绑定和数量。跨板相同型号是现有库存候选，不代表电气或机械验收；缺少型号或绑定的实例仍保留在总表。FOC按单板当前原理图计数，未乘以整车两板。",
        "",
        "## 同步统计",
        "",
        f"- BOM实例：{summary['bom_instance_count']}；严格型号分组：{summary['part_count']}；跨板同制造商+MPN严格组：{summary['shared_model_count']}；另有MPN相同但制造商字段不一致/缺失的跨板候选：{len(mpn_only_candidates)}组。",
        f"- 非BOM组件排除：{summary['excluded_non_bom_count']}；缺MPN：{summary['missing_mpn_count']}；缺制造商：{summary['missing_manufacturer_count']}。",
        f"- 缺失DEVICE/SYMBOL/FOOTPRINT绑定：{summary['missing_device_binding_count']}/{summary['missing_symbol_binding_count']}/{summary['missing_footprint_binding_count']}。",
        "",
        "| 模块 | BOM实例 | 非BOM组件排除 | 缺DEVICE | 缺SYMBOL | 缺FOOTPRINT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in module_stats:
        lines.append(
            f"| {item['module']} | {item['bom_instance_count']} | {item['excluded_non_bom_count']} | {item['missing_device_binding_count']} | {item['missing_symbol_binding_count']} | {item['missing_footprint_binding_count']} |"
        )

    shared = [part for part in parts if part["shared_across_boards"]]
    lines.extend(["", "## 已存在的跨板同型号候选", ""])
    if shared:
        lines.extend(["| 制造商 | 实际MPN | 模块 | 数量 | 位号 |", "|---|---|---|---:|---|"])
        for part in shared:
            refs_by_module: dict[str, list[str]] = defaultdict(list)
            for instance in instances:
                if instance["part_key"] == part["key"]:
                    refs_by_module[instance["module"]].append(instance["ref"])
            rows = []
            for module in part["modules"]:
                refs = refs_by_module[module]
                rows.append(f"{module} ×{len(refs)} ({', '.join(refs)})")
            lines.append(
                f"| {display(part['manufacturer']) or '（制造商缺失）'} | {display(part['mpn']) or '（MPN缺失）'} | {'；'.join(rows)} | {part['bom_instance_count']} | {'；'.join(part['designators'])} |"
            )
    else:
        lines.append("当前三份原理图中没有已确认的跨板同型号分组。")

    lines.extend(["", "## MPN相同但制造商字段不一致或缺失的跨板候选", ""])
    lines.append("下表仅按实际MPN字段寻找跨板重合；制造商字段差异可能是命名写法差异，也可能指向不同来源。严格厂家+MPN型号组保持独立，不自动合并，也不表示电气或机械验收。")
    if mpn_only_candidates:
        lines.extend(["", "| 实际MPN | 模块及制造商字段 / 位号 | 严格型号组数 | BOM实例 |", "|---|---|---:|---:|"])
        for candidate in mpn_only_candidates:
            module_entries: dict[str, list[str]] = defaultdict(list)
            for entry in candidate["manufacturer_fields"]:
                manufacturer = entry["manufacturer"] or "（制造商缺失）"
                module_entries[entry["module"]].append(f"{manufacturer}：{', '.join(entry['designators'])}")
            module_summary = "；".join(
                f"{module}（{' / '.join(module_entries[module])}）" for module in candidate["modules"]
            )
            lines.append(
                f"| {candidate['mpn']} | {module_summary} | {len(candidate['strict_part_keys'])} | {candidate['bom_instance_count']} |"
            )
    else:
        lines.append("未发现MPN相同但制造商字段不一致或缺失的跨板BOM实例。")

    mpn_diffs = [item for item in instances if item["mpn_diff"]]
    lines.extend(["", "## 实际MPN与DEVICE载体差异", ""])
    if mpn_diffs:
        lines.extend(["| 模块/位号 | 实际制造商 / MPN | DEVICE载体制造商 / MPN | 实际值（BOM/MPN） | 载体Value | 实际值来源 | 核对提示 |", "|---|---|---|---|---|---|---|"])
        for item in mpn_diffs:
            hint = "MPN已覆盖；缺明确实料值，载体值不作实际值" if item["mpn_changed_without_actual_value"] else "已记录实际值"
            lines.append(
                f"| {item['module']} / {item['ref']} | {display(item['manufacturer']) or '（缺失）'} / {display(item['mpn']) or '（缺失）'} | {display(item['carrier_manufacturer']) or '（缺失）'} / {display(item['carrier_mpn']) or '（缺失）'} | {display(item['actual_value']) or '（未确认）'} | {display(item['carrier_value']) or '（无）'} | {display(item['actual_value_source']) or '未确认'} | {hint} |"
            )
            special = {
                key: values
                for key, values in item["device_attribute_differences"].items()
                if "ctr" in key.casefold()
                or "transfer ratio" in key.casefold()
                or "grade" in key.casefold()
                or "等级" in key
            }
            if special:
                details = "; ".join(f"{key}: {display(value['carrier']) or '（缺失）'} → {display(value['actual'])}" for key, value in special.items())
                lines.append(f"| ↳ CTR/等级字段 |  |  |  | {details} |")
    else:
        lines.append("没有发现实际 Manufacturer Part 与 DEVICE META 载体 Manufacturer Part 不同的BOM实例。")
    lines.extend(
        [
            "",
            "## 缺失型号或原生绑定",
            "",
            "| 模块/位号 | Manufacturer | MPN | DEVICE | SYMBOL | FOOTPRINT |",
            "|---|---|---|---|---|---|",
        ]
    )
    pending = [
        item
        for item in instances
        if not is_nonempty(item.get("mpn"))
        or not is_nonempty(item.get("manufacturer"))
        or item["device_asset"]["status"] != "present"
        or item["symbol_asset"]["status"] != "present"
        or item["footprint_asset"]["status"] != "present"
    ]
    if not pending:
        lines.append("当前BOM实例均有制造商、MPN及可解析的原生绑定。")
    else:
        for item in pending:
            lines.append(
                f"| {item['module']} / {item['ref']} | {display(item['manufacturer']) or '缺失'} | {display(item['mpn']) or '缺失'} | {item['device_asset']['status']} | {item['symbol_asset']['status']} | {item['footprint_asset']['status']} |"
            )
    if variants:
        lines.extend(["", "## 同官方UUID的内容变体", "", "| 类型 | 官方UUID | 内容SHA256变体数 | Library UUID |", "|---|---|---:|---|"])
        for variant in variants:
            lines.append(
                f"| {variant['type']} | {variant['source_uuid']} | {len(variant['sha256_variants'])} | {', '.join(variant['library_uuids'])} |"
            )
    lines.extend(["", "原生片段按内容SHA256保存，供追溯和复用。片段索引不表示其可直接导入EasyEDA，也不构成电气、封装或实物验收。", ""])
    return "\n".join(lines)


def check_only() -> int:
    errors: list[str] = []
    if not EVIDENCE_PATH.is_file():
        print(f"FAIL 缺少同步证据：{EVIDENCE_PATH}")
        return 1
    try:
        evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8-sig"))
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8-sig"))
        asset_index = json.loads(ASSET_INDEX_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"FAIL 索引或证据无法读取：{exc}")
        return 1

    config_sha = sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None
    if config_sha != evidence.get("config_sha256"):
        errors.append("来源配置SHA256与同步证据不符")
    try:
        sources, _, _ = source_config()
    except Exception as exc:
        errors.append(str(exc))
        sources = []
    evidence_sources = {(entry.get("module"), entry.get("path")): entry for entry in evidence.get("sources", [])}
    for source in sources:
        key = (source["module"], source["path"])
        item = evidence_sources.get(key)
        if item is None:
            errors.append(f"同步证据缺少来源：{source['module']} / {source['path']}")
            continue
        try:
            path = resolve_source(source["path"])
            current_sha = sha256_file(path)
            if current_sha != item.get("sha256"):
                errors.append(f"现行源SHA256已变化：{source['module']} / {source['path']}")
        except Exception as exc:
            errors.append(str(exc))
    if len(evidence_sources) != len(sources):
        errors.append("同步证据来源数量与来源配置不符")

    listed_assets = {(item.get("type"), item.get("sha256")): item for item in asset_index.get("assets", []) if isinstance(item, dict)}
    referenced: set[tuple[str, str]] = set()
    for instance in index.get("instances", []):
        for field in ("device_asset", "symbol_asset", "footprint_asset"):
            ref = instance.get(field, {})
            digest = ref.get("sha256")
            if digest:
                key = (ref.get("type"), digest)
                referenced.add(key)
                if key not in listed_assets:
                    errors.append(f"实例引用未登记的原生资产：{instance.get('key')} / {key}")
    for key, asset in listed_assets.items():
        relative = asset.get("path") or f"元件封装库/原生片段/{key[0]}/{key[1]}.jsonl"
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"资产索引路径越界：{relative}")
            continue
        if not path.is_file():
            errors.append(f"缺少原生片段：{relative}")
        elif sha256_file(path) != key[1]:
            errors.append(f"原生片段SHA256不符：{relative}")
    expected_count = evidence.get("summary", {}).get("bom_instance_count")
    actual_count = len(index.get("instances", []))
    if expected_count != actual_count or index.get("summary", {}).get("bom_instance_count") != actual_count:
        errors.append(f"BOM覆盖数量不符：证据={expected_count}，索引={actual_count}")
    if evidence.get("asset_reference_count") != len(referenced):
        errors.append("同步证据资产引用覆盖数与总表不符")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print(
        "PASS "
        f"来源={len(sources)}，BOM实例={actual_count}，资产引用={len(referenced)}，"
        f"资产文件={len(listed_assets)}；配置、源文件和资产SHA256一致。"
    )
    return 0


def generate() -> int:
    sources, config_hash, config_label = source_config()
    parsed_sources = []
    for item in sources:
        path = resolve_source(item["path"])
        parsed_sources.append(parse_source(path, item["module"], item["path"]))

    module_order = unique_module_order(sources)
    all_instances: list[dict[str, Any]] = []
    source_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for parsed in parsed_sources:
        instances, counts = get_components(parsed, {})
        all_instances.extend(instances)
        source_stats[(parsed["module"], parsed["path"])] = counts
    module_stats = aggregate_module_stats(source_stats, module_order)
    # 保证稳定顺序；位号缺失时已回退到原生partId或component id。
    module_rank = {module: index for index, module in enumerate(module_order)}
    all_instances.sort(key=lambda item: (module_rank.get(item["module"], 999), item["ref"].casefold(), item["component_id"]))
    parts = build_parts(all_instances, module_order)
    mpn_only_candidates = build_mpn_only_candidates(all_instances)
    shared_model_count = sum(part["shared_across_boards"] for part in parts)
    missing_mpn_count = sum(not is_nonempty(item.get("mpn")) for item in all_instances)
    missing_manufacturer_count = sum(not is_nonempty(item.get("manufacturer")) for item in all_instances)
    summary = {
        "module_count": len(module_order),
        "source_file_count": len(parsed_sources),
        "bom_instance_count": len(all_instances),
        "part_count": len(parts),
        "shared_model_count": int(shared_model_count),
        "mpn_same_cross_board_manufacturer_mismatch_count": len(mpn_only_candidates),
        "excluded_non_bom_count": sum(item["excluded_non_bom_count"] for item in module_stats),
        "missing_mpn_count": int(missing_mpn_count),
        "missing_manufacturer_count": int(missing_manufacturer_count),
        "mpn_diff_count": sum(item["mpn_diff"] for item in all_instances),
        "mpn_changed_with_value_inherited_count": sum(item["mpn_changed_with_value_inherited"] for item in all_instances),
        "mpn_changed_without_actual_value_count": sum(item["mpn_changed_without_actual_value"] for item in all_instances),
        "missing_device_binding_count": sum(item["missing_device_binding_count"] for item in module_stats),
        "missing_symbol_binding_count": sum(item["missing_symbol_binding_count"] for item in module_stats),
        "missing_footprint_binding_count": sum(item["missing_footprint_binding_count"] for item in module_stats),
    }
    current_assets, variants = write_fragments(parsed_sources)
    asset_index = merge_asset_index(current_assets)
    asset_index["current_official_uuid_variants"] = variants
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index = {
        "schema_version": 2,
        "generated_at_utc": generated_at,
        "quantity_basis": "按各模块当前原理图BOM实例计数；FOC为单板用量，未乘整车两板。",
        "identity_rule": "实际 Manufacturer Part 与 Manufacturer 经NFKC、大小写及空白规整后精确匹配；缺MPN或缺制造商时按模块+位号独立记录。",
        "value_semantics": {
            "device_value": "实际器件值：优先使用明确实例Value或非公式实例Name；仅当MPN与DEVICE载体MPN一致时才回退DEVICE.Value；MPN改变且无明确值时为null。",
            "actual_value": "与device_value同义，保留显式命名以便下游兼容。",
            "carrier_value": "DEVICE载体原始Value，只描述CAD几何/库载体，不自动代表实际采购器件。",
            "effective_value_raw": "原理图属性合成后的Value原始值，供追溯，不作为跨MPN实际值的默认来源。"
        },
        "sources": [
            {
                "module": parsed["module"],
                "path": parsed["path"],
                "sha256": parsed["sha256_before"],
                "sha256_after_parse": parsed["sha256_after_parse"],
                "page_count": len(parsed["pages"]),
                "page_titles": [page.get("title", "") for page in parsed["pages"]],
                "page_uuids": [page.get("local_uuid") for page in parsed["pages"]],
            }
            for parsed in parsed_sources
        ],
        "summary": summary,
        "module_stats": module_stats,
        "parts": parts,
        "mpn_same_cross_board_manufacturer_mismatch_candidates": mpn_only_candidates,
        "instances": all_instances,
    }

    write_json(INDEX_PATH, index)
    write_csv(
        LIB_ROOT / "索引" / "元件总表.csv",
        ["part_key", "manufacturer", "mpn", "bom_instance_count", "board_count", "modules", "designators", "supplier_parts", "device_ids", "footprint_ids", "missing_mpn", "missing_manufacturer"],
        csv_parts(parts),
    )
    write_csv(
        LIB_ROOT / "索引" / "板级用量.csv",
        ["module", "part_key", "quantity", "manufacturer", "mpn", "designators", "supplier_parts", "pcb_yes", "pcb_no", "pcb_unknown", "source_refs"],
        csv_board_use(all_instances, module_order),
    )
    write_json(ASSET_INDEX_PATH, asset_index)
    (LIB_ROOT / "索引" / "复用与待核对.md").parent.mkdir(parents=True, exist_ok=True)
    (LIB_ROOT / "索引" / "复用与待核对.md").write_text(
        render_markdown(parts, all_instances, module_stats, variants, summary), encoding="utf-8"
    )

    config_after = sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None
    final_source_hashes = []
    for parsed in parsed_sources:
        final_source_hashes.append(sha256_file(parsed["path_obj"]))
    expected_hashes = [parsed["sha256_before"] for parsed in parsed_sources]
    if config_after != config_hash:
        raise RuntimeError("生成期间来源配置SHA256变化；同步证据未更新")
    if final_source_hashes != expected_hashes:
        raise RuntimeError("生成期间现行原理图SHA256变化；同步证据未更新")

    asset_refs = {
        (field["type"], field["sha256"])
        for instance in all_instances
        for field in (instance["device_asset"], instance["symbol_asset"], instance["footprint_asset"])
        if field.get("sha256")
    }
    evidence = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "config": config_label,
        "config_sha256": config_hash,
        "sources": build_source_evidence_rows(parsed_sources, final_source_hashes, source_stats),
        "summary": summary,
        "asset_reference_count": len(asset_refs),
        "asset_file_count": asset_index["asset_count"],
        "retained_asset_file_count": asset_index["asset_count"] - asset_index["current_source_asset_count"],
        "content_variants": variants,
        "coverage": {
            "bom_instances_in_index": len(all_instances),
            "module_stats_sum": sum(item["bom_instance_count"] for item in module_stats),
            "excluded_non_bom_components": summary["excluded_non_bom_count"],
            "source_sha256_verified_before_after": all(item["sha256_before"] == final_sha for item, final_sha in zip(parsed_sources, final_source_hashes)),
        },
        "scope_note": "只解析来源配置 native_sources 中的原生SCH_PAGE；不读取 bom_sources、归档或临时材料。原生片段索引不表示可直接导入，也不表示设计验收。",
    }
    write_json(EVIDENCE_PATH, evidence)
    print(
        f"已生成：BOM实例 {summary['bom_instance_count']}，型号分组 {summary['part_count']}，"
        f"跨板同型号候选 {summary['shared_model_count']}，排除非BOM组件 {summary['excluded_non_bom_count']}。"
    )
    print(
        "缺失绑定 DEVICE/SYMBOL/FOOTPRINT："
        f"{summary['missing_device_binding_count']}/{summary['missing_symbol_binding_count']}/{summary['missing_footprint_binding_count']}；"
        f"MPN与载体差异 {summary['mpn_diff_count']}；旧指标直接载体Value继承 {summary['mpn_changed_with_value_inherited_count']}，"
        f"缺明确实际值提示 {summary['mpn_changed_without_actual_value_count']}。"
    )
    print("原理图与来源配置SHA256前后核对一致；同步证据已写入 检查/同步证据.json。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="更新现行原理图元件总表与原生片段索引")
    parser.add_argument("--check", action="store_true", help="只读校验来源SHA、资产引用与BOM覆盖数量")
    args = parser.parse_args()
    try:
        return check_only() if args.check else generate()
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
