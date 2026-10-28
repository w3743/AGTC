"""Build and verify the current one-page BMS EPRJ3 schematic through the official API.

--prepare is offline. --build requires a matching count closure, the expected open blank
EPRJ3 project, and exact live API readback. It creates no PCB and no firmware.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
     if (parent / "原理图准备").is_dir() and (parent / "检查").is_dir()),
    SCRIPT_DIR.parent / "BMS模块"
    if (SCRIPT_DIR.parent / "BMS模块" / "原理图准备").is_dir()
    and (SCRIPT_DIR.parent / "BMS模块" / "检查").is_dir()
    else None,
)
if MODULE is None:
    raise RuntimeError("无法从脚本位置定位BMS模块目录")
CHECK = MODULE / "检查"
PREP = MODULE / "原理图准备"
BOM_PATH = PREP / "BOM.csv"
SOURCE_FILES = [
    "AFE元件连接.json",
    "功率元件连接.json",
    "接口元件连接.json",
    "共享电感矩阵连接.json",
    "共享电感功率级连接.json",
    "共享电感控制保护连接.json",
    "共享电感泄放连接.json",
    "风扇元件连接.json",
    "维护线束元件连接.json",
]
NO_MATERIAL_ANCHORS = {
    "J_BAL_SERVICE", "J_BAT_HV", "J_BUS_OUTPUT", "J_CELL_SERVICE", "J_PACK_HV"
}
MAP_PATH = CHECK / "Layout器件映射.json"
LAYOUT_SEARCH_PATH = CHECK / "Layout库检索.json"
BINDING_CANDIDATES = [
    CHECK / "本轮新增选型库绑定.json",
    PREP / "本轮新增选型库绑定.json",
]
COUNT_EVIDENCE_PATH = CHECK / "当前BOM计数证据.json"
COUNT_CLOSURE_PATH = CHECK / "单页eprj3计数闭合.json"
DEFAULT_LIBRARY_UUID = "0819f05c4eef4c71ace90d822a990e87"
CELL_COLUMNS = 4
CELL_X = 1150
CELL_Y = 900
GROUP_GAP_X = 950
GROUP_GAP_Y = 900
GROUP_COLUMNS = 6
EPRJ3_WORK = CHECK / "tmp" / "bms_eprj3_work" / "BMS-9S4P.eprj3"
EXPECTED_PROJECT_UUID = "4937470dd1eb648e6c8c7bd1ea94829fb5683ef1413c780ea7a2d442ce2ec3d4"
EXPECTED_SCHEMATIC_UUID = "1c3e06662fdc8e9a"
EXPECTED_PAGE_UUID = "350a4904c4ec87b3"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def as_uuid(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("uuid") or value.get("libUuid") or value.get("deviceUuid")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_mapping() -> dict[str, dict[str, Any]]:
    if not MAP_PATH.exists():
        return {}
    raw = read_json(MAP_PATH)
    rows = raw if isinstance(raw, list) else raw.get("components", [])
    return {str(row["ref"]): row for row in rows if isinstance(row, dict) and row.get("ref")}


def exact_old_mapping_device(old: dict[str, Any], mpn: str | None) -> dict[str, Any] | None:
    """旧施工映射中candidate/footprintCarrier含真实device UUID；旧libUuid可能只是symbol UUID。"""
    expected_mpn = as_text(mpn)
    for field in ("candidate", "footprintCarrier"):
        device = old.get(field)
        if not isinstance(device, dict):
            continue
        device_mpn = as_text(device.get("mpn")) or as_text(device.get("manufacturerId")) or as_text(device.get("name"))
        if not expected_mpn or device_mpn != expected_mpn:
            continue
        lib_uuid = as_uuid(device.get("uuid"))
        symbol_uuid = as_uuid(device.get("symbolUuid"))
        footprint_uuid = as_uuid(device.get("footprintUuid"))
        if lib_uuid and symbol_uuid and footprint_uuid:
            return {
                "libUuid": lib_uuid,
                "libraryUuid": as_uuid(device.get("libraryUuid")) or as_uuid(old.get("libraryUuid")) or DEFAULT_LIBRARY_UUID,
                "libType": as_text(old.get("libType")) or "2",
                "symbolUuid": symbol_uuid,
                "footprintUuid": footprint_uuid,
                "footprintName": as_text(device.get("footprintName")),
                "mpn": device_mpn,
                "sourceField": field,
            }
    return None


def exact_layout_search_binding(library_key: str | None, mpn: str | None) -> dict[str, Any] | None:
    """从已保存的官方库搜索结果读取唯一精确MPN器件，避免用错封装载体。"""
    key = as_text(library_key)
    part = as_text(mpn)
    if not key or not part or key != part or not LAYOUT_SEARCH_PATH.exists():
        return None
    raw = read_json(LAYOUT_SEARCH_PATH)
    records = raw if isinstance(raw, list) else raw.get("items", []) if isinstance(raw, dict) else []
    exact: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or as_text(record.get("key")) != key:
            continue
        for item in record.get("items", []):
            if not isinstance(item, dict):
                continue
            item_mpn = as_text(item.get("manufacturerId")) or as_text(item.get("name"))
            if item_mpn == part and as_uuid(item.get("uuid")) and as_uuid(item.get("symbolUuid")) and as_uuid(item.get("footprintUuid")):
                exact.append(item)
    # 同一器件完全重复可折叠；同MPN若多库器件/封装则视为歧义，不自动挑选。
    unique = {(as_uuid(item.get("uuid")), as_uuid(item.get("libraryUuid")), as_uuid(item.get("footprintUuid"))): item for item in exact}
    if len(unique) != 1:
        return None
    item = next(iter(unique.values()))
    return {
        "libUuid": as_uuid(item.get("uuid")),
        "libraryUuid": as_uuid(item.get("libraryUuid")) or DEFAULT_LIBRARY_UUID,
        "libType": "2",
        "symbolUuid": as_uuid(item.get("symbolUuid")),
        "footprintUuid": as_uuid(item.get("footprintUuid")),
        "footprintName": as_text(item.get("footprintName")),
        "libraryKey": key,
        "mpn": part,
        "_bindingSource": "saved official exact-MPN library search",
    }


def binding_records(node: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """接受新增绑定文件常见的 parts/items/byRef 结构，提取 ref 与 libraryKey 映射。"""
    by_ref: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    system_library_uuid = None
    if isinstance(node, dict) and isinstance(node.get("source"), dict):
        system_library_uuid = as_uuid(node["source"].get("systemLibraryUuid"))

    def visit(value: Any, inherited: dict[str, str] | None = None, allow_binding: bool = True) -> None:
        inherited = dict(inherited or {})
        if isinstance(value, list):
            for child in value:
                visit(child, inherited, allow_binding)
            return
        if not isinstance(value, dict):
            return
        context = dict(inherited)
        for field, target in (("ref", "ref"), ("reference", "ref"), ("designator", "ref"), ("key", "key"),
                              ("libraryKey", "key"), ("mpn", "mpn")):
            if value.get(field) is not None:
                context[target] = str(value[field]).strip()
        descriptor = value.get("device") if isinstance(value.get("device"), dict) else value
        lib_uuid = as_uuid(descriptor.get("libUuid") or descriptor.get("uuid") or descriptor.get("deviceUuid"))
        library_uuid = as_uuid(descriptor.get("libraryUuid") or value.get("libraryUuid")) or system_library_uuid
        if lib_uuid and allow_binding:
            row = {
                "libUuid": lib_uuid,
                "libraryUuid": library_uuid,
                "libType": as_text(descriptor.get("libType") or value.get("libType")),
                "symbolUuid": as_uuid(descriptor.get("symbolUuid") or value.get("symbolUuid")),
                "footprintUuid": as_uuid(descriptor.get("footprintUuid") or value.get("footprintUuid")),
                "footprintName": as_text(descriptor.get("footprintName") or value.get("footprintName")),
                "libraryKey": context.get("key"),
                "mpn": context.get("mpn"),
                "bindingStatus": as_text(descriptor.get("bindingStatus") or value.get("bindingStatus")),
                "carrierForMpn": as_text(descriptor.get("carrierForMpn") or value.get("carrierForMpn")),
                "bindingNote": as_text(descriptor.get("bindingNote") or value.get("bindingNote")),
            }
            if context.get("ref"):
                by_ref[context["ref"]] = row
            if context.get("key"):
                by_key[context["key"]] = row
            if context.get("mpn"):
                by_key[context["mpn"]] = row
        for key, child in value.items():
            if key in {"device", "devices"} and isinstance(child, (dict, list)):
                visit(child, context, allow_binding)
            elif isinstance(child, (dict, list)):
                child_context = dict(context)
                # 支持 {"R1": {"libUuid": ..., "libraryUuid": ...}}。
                if isinstance(child, dict) and not any(k in child for k in ("ref", "reference", "designator", "key", "libraryKey", "mpn")):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]*\d+[A-Z0-9_]*", str(key), re.I):
                        child_context["ref"] = str(key)
                # pendingDevices 仅是未确认候选，不允许进入可施工绑定清单。
                visit(child, child_context, allow_binding and key != "pendingDevices")

    visit(node)
    return by_ref, by_key


def source_rows() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    count_source = current_count_snapshot()
    if not count_source.get("available"):
        raise RuntimeError(f"缺少当前主板BOM计数证据：{COUNT_EVIDENCE_PATH}")
    board_refs = count_source["mainPcbRefs"]
    board_ref_set = set(board_refs)
    bom_rows_by_ref = load_current_board_bom_rows(board_refs)

    mapping = load_mapping()
    bind_ref: dict[str, dict[str, Any]] = {}
    bind_key: dict[str, dict[str, Any]] = {}
    binding_file = next((p for p in BINDING_CANDIDATES if p.exists()), None)
    if binding_file:
        bind_ref, bind_key = binding_records(read_json(binding_file))

    grouped: dict[str, dict[str, Any]] = {}
    non_electrical: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    source_row_count = 0
    board_source_row_count = 0
    excluded_source_refs: set[str] = set()
    for filename in SOURCE_FILES:
        path = PREP / filename
        if not path.exists():
            issues.append(f"缺少源文件：{path}")
            continue
        data = read_json(path)
        if not isinstance(data, list):
            issues.append(f"源文件顶层不是数组：{path}")
            continue
        for raw in data:
            source_row_count += 1
            if not isinstance(raw, dict) or not raw.get("ref"):
                issues.append(f"{filename} 含无 ref 记录：{raw!r}")
                continue
            ref = str(raw["ref"]).strip()
            if ref not in board_ref_set:
                excluded_source_refs.add(ref)
                continue
            if ref in NO_MATERIAL_ANCHORS:
                issues.append(f"计数证据将非器件连接边界列入主板refs：{ref}")
                continue
            board_source_row_count += 1
            bom_row = bom_rows_by_ref[ref]
            bom_mpn = as_text(bom_row.get("MPN"))
            source_mpn = as_text(raw.get("mpn"))
            if not bom_mpn:
                issues.append(f"当前主板BOM器件缺少MPN：{ref}")
                continue
            if source_mpn != bom_mpn:
                issues.append(f"当前BOM与连接JSON的MPN不一致：{ref}: BOM={bom_mpn!r}, JSON={source_mpn!r}")
            pins = raw.get("pins")
            if isinstance(pins, dict) and not pins and filename == "维护线束元件连接.json" and raw.get("mpn"):
                issues.append(f"主板BOM位号出现无引脚维护件记录：{ref}（{filename}）")
                continue
            if not isinstance(pins, dict) or not pins:
                issues.append(f"{ref} 缺少 pins 映射（{filename}）")
                continue
            pins = {str(k): (None if v is None else str(v)) for k, v in pins.items()}
            source_pin_role_map = None
            old = grouped.get(ref)
            if old:
                if old["pins"] != pins:
                    issues.append(f"重复 ref 引脚/网络冲突：{ref}（{', '.join(old['_sourceFiles'] + [filename])}）")
                    continue
                old_mpn = as_text(old.get("mpn"))
                new_mpn = bom_mpn
                if old_mpn and new_mpn and old_mpn != new_mpn:
                    issues.append(f"重复 ref MPN 冲突：{ref}: {old_mpn} / {new_mpn}")
                    continue
                if not old_mpn and new_mpn:
                    old["mpn"] = new_mpn
                for field in ("libUuid", "libraryUuid", "libType", "value", "pageGroup", "addIntoPcb", "addIntoBom"):
                    if not old.get(field) and raw.get(field) is not None:
                        old[field] = raw.get(field)
                note = as_text(raw.get("note"))
                if note and len(note) > len(old.get("note") or ""):
                    old["note"] = note
                old["_sourceFiles"].append(filename)
                continue
            grouped[ref] = {
                **raw,
                "ref": ref,
                "mpn": bom_mpn,
                "_sourceMpn": source_mpn,
                "_mpnMatchesBom": source_mpn == bom_mpn,
                "_bomValue": as_text(bom_row.get("MPN编码标称值")),
                "pins": pins,
                "_sourcePinRoleMap": source_pin_role_map,
                "_sourceFiles": [filename],
            }

    missing_board_refs = sorted(board_ref_set - set(grouped))
    for ref in missing_board_refs:
        if ref in non_electrical:
            issues.append(f"当前主板ref只有无引脚记录，不能进入单板器件清单：{ref}")
        else:
            issues.append(f"当前主板BOM ref缺少现行连接JSON记录：{ref}")

    entries: list[dict[str, Any]] = []
    for ref, raw in grouped.items():
        old = mapping.get(ref, {})
        mpn = as_text(raw.get("mpn"))
        source_file = raw["_sourceFiles"][0]
        key = as_text(raw.get("libraryKey")) or as_text(raw.get("libraryKeyName"))
        key_bind = bind_key.get(key) if key else None
        if key_bind and as_text(key_bind.get("mpn")) != mpn:
            key_bind = None
        if not key_bind and mpn:
            key_bind = bind_key.get(mpn)
        if key_bind and as_text(key_bind.get("mpn")) != mpn:
            key_bind = None
        saved_search_bind = exact_layout_search_binding(key, mpn) if not key_bind else None
        if not key_bind and saved_search_bind:
            key_bind = saved_search_bind
        ref_bind = bind_ref.get(ref)
        new_bind = ref_bind if ref_bind and as_text(ref_bind.get("mpn")) == mpn else key_bind
        # 旧映射只允许从带完整器件UUID的candidate/footprintCarrier取值，且MPN必须逐字匹配。
        old_device = exact_old_mapping_device(old, mpn)
        source_mpn_matches = raw.get("_mpnMatchesBom") is True
        direct = {
            "libUuid": as_uuid(raw.get("libUuid")) if source_mpn_matches else None,
            "libraryUuid": as_uuid(raw.get("libraryUuid")) if source_mpn_matches else None,
            "libType": as_text(raw.get("libType")) if source_mpn_matches else None,
        }
        resolved = {}
        for field in ("libUuid", "libraryUuid", "libType"):
            old_value = (old_device or {}).get(field)
            if field in {"libUuid", "libraryUuid"}:
                old_value = as_uuid(old_value)
            else:
                old_value = as_text(old_value)
            resolved[field] = (new_bind or {}).get(field) or direct.get(field) or old_value
        # 主板清单由权威计数refs限定；不得继承历史的板外/不入BOM标志。
        add_pcb = True
        add_bom = True
        page_group = as_text(raw.get("pageGroup")) or as_text(old.get("pageGroup"))
        if not page_group:
            page_group = Path(source_file).stem.replace("元件连接", "")
        entry = {
            "ref": ref,
            "mpn": mpn,
            "value": as_text(raw.get("value")) or as_text(raw.get("_bomValue")) or mpn or ref,
            "pins": raw["pins"],
            "_sourcePinRoleMap": raw.get("_sourcePinRoleMap"),
            "note": as_text(raw.get("note")) or as_text(old.get("note")) or "",
            "pageGroup": page_group,
            "sourceFiles": sorted(set(raw["_sourceFiles"])),
            "libraryKey": key,
            "libUuid": resolved["libUuid"],
            "libraryUuid": resolved["libraryUuid"] or DEFAULT_LIBRARY_UUID,
            "libType": resolved["libType"],
            "symbolUuid": (new_bind or {}).get("symbolUuid") or (as_uuid(raw.get("symbolUuid")) if source_mpn_matches else None) or (old_device or {}).get("symbolUuid"),
            "footprintUuid": (new_bind or {}).get("footprintUuid") or (as_uuid(raw.get("footprintUuid")) if source_mpn_matches else None) or (old_device or {}).get("footprintUuid"),
            "addIntoPcb": bool(add_pcb),
            "addIntoBom": bool(add_bom),
            "expectedFootprintName": (raw.get("expectedFootprintName") if source_mpn_matches else None) or (new_bind or {}).get("footprintName") or (old_device or {}).get("footprintName"),
            "_bindingSource": (new_bind or {}).get("_bindingSource") or ("exact-MPN new binding" if new_bind else "source" if direct["libUuid"] else f"exact-MPN old map {old_device['sourceField']}" if old_device else "missing"),
            "bindingStatus": (new_bind or {}).get("bindingStatus"),
            "carrierForMpn": (new_bind or {}).get("carrierForMpn"),
            "bindingNote": (new_bind or {}).get("bindingNote"),
        }
        entries.append(entry)

    entries.sort(key=lambda e: e["ref"])
    # 同一精确MPN且同端号时，复用现存同源器件；从多数载体足迹唯一的MPN/封装组合中选择。
    # 只复制库绑定字段，源MPN、值、引脚网络与BOM标志保持不变。
    for entry in entries:
        if entry.get("libUuid") or not entry.get("mpn") or not entry["ref"].startswith(("R_", "C_")):
            continue
        candidates = [
            candidate for candidate in entries
            if candidate is not entry
            and candidate.get("libUuid")
            and candidate.get("mpn") == entry.get("mpn")
            and sorted(candidate["pins"]) == sorted(entry["pins"])
            and candidate.get("libraryUuid") == entry.get("libraryUuid")
        ]
        groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
        for candidate in candidates:
            key = (candidate["libUuid"], candidate.get("expectedFootprintName"))
            groups.setdefault(key, []).append(candidate)
        if groups:
            max_count = max(len(rows) for rows in groups.values())
            winners = [(key, rows) for key, rows in groups.items() if len(rows) == max_count]
            if len(winners) == 1:
                _, winner_rows = winners[0]
                carrier = sorted(winner_rows, key=lambda row: row["ref"])[0]
                for field in ("libUuid", "libraryUuid", "libType", "symbolUuid", "footprintUuid", "expectedFootprintName"):
                    if carrier.get(field):
                        entry[field] = carrier[field]
                entry["_bindingSource"] = f"exact-MPN carrier:{carrier['ref']}"
                entry["_carrierRef"] = carrier["ref"]
                entry["_carrierReason"] = "Exact source MPN and pin numbers match an existing current device; source value and net map remain unchanged."

    return entries, issues, {
        "sourceRowCount": source_row_count,
        "boardOnlySourceRowCount": board_source_row_count,
        "sourceRefCount": len(grouped) + len(non_electrical),
        "electricalSymbolRefCount": len(grouped),
        "nonElectricalItemCount": len(non_electrical),
        "nonElectricalItems": [non_electrical[ref] for ref in sorted(non_electrical)],
        "boardOnlyRefSource": count_source,
        "boardOnlyRefs": board_refs,
        "boardOnlyRefCount": len(board_refs),
        "excludedOutOfBoardSourceRefs": sorted(excluded_source_refs),
        "excludedAnchorRefs": sorted(excluded_source_refs & NO_MATERIAL_ANCHORS),
        "missingBoardRefs": missing_board_refs,
        "bindingFile": str(binding_file) if binding_file else None,
        "oldMappingRows": len(mapping),
    }


def input_signature(
    entries: list[dict[str, Any]],
    board_ref_source: dict[str, Any],
    non_electrical_items: list[dict[str, Any]] | None = None,
) -> str:
    normalized = [{k: e.get(k) for k in (
        "ref", "mpn", "value", "pins", "pageGroup", "sourceFiles", "libraryKey",
        "libUuid", "libraryUuid", "libType", "symbolUuid", "footprintUuid",
        "expectedFootprintName", "_carrierRef", "_carrierReason", "_carrierPinNumbers", "_sourcePinRoleMap", "note", "addIntoPcb", "addIntoBom",
        "bindingStatus", "carrierForMpn", "bindingNote",
    )} for e in entries]
    non_electrical = [{k: e.get(k) for k in (
        "ref", "mpn", "value", "note", "pageGroup", "sourceFiles", "addIntoPcb", "addIntoBom", "representation", "bomOnly",
    )} for e in (non_electrical_items or [])]
    return canonical_sha({
        "components": normalized,
        "nonElectricalItems": non_electrical,
        "boardOnlyRefSource": {
            "evidencePath": board_ref_source.get("evidencePath"),
            "sourceGeneratorPath": board_ref_source.get("sourceGeneratorPath"),
            "sourceBOMPath": board_ref_source.get("sourceBOMPath"),
            "bomSha256": board_ref_source.get("bomSha256"),
            "mainPcbRefs": board_ref_source.get("mainPcbRefs"),
        },
    })


def current_count_snapshot() -> dict[str, Any]:
    if not COUNT_EVIDENCE_PATH.exists():
        return {"available": False}
    data = read_json(COUNT_EVIDENCE_PATH)
    boundary = data.get("boardMountBoundary") if isinstance(data, dict) else None
    refs_raw = boundary.get("mainPcbRefs") if isinstance(boundary, dict) else None
    if not isinstance(refs_raw, list) or not refs_raw:
        raise RuntimeError(f"当前BOM计数证据缺少boardMountBoundary.mainPcbRefs：{COUNT_EVIDENCE_PATH}")
    refs = [as_text(ref) for ref in refs_raw]
    if any(ref is None for ref in refs):
        raise RuntimeError("当前主板BOM计数证据含空ref")
    normalized_refs = [str(ref) for ref in refs]
    if len(set(normalized_refs)) != len(normalized_refs):
        raise RuntimeError("当前主板BOM计数证据含重复ref")
    declared_unique_refs = boundary.get("mainPcbUniqueRefs")
    if declared_unique_refs is not None and declared_unique_refs != len(normalized_refs):
        raise RuntimeError(
            f"当前主板BOM计数证据ref数不一致：列表={len(normalized_refs)}，声明={declared_unique_refs}"
        )
    forbidden_refs = sorted(set(normalized_refs) & NO_MATERIAL_ANCHORS)
    if forbidden_refs:
        raise RuntimeError("主板ref清单含非器件连接边界：" + ", ".join(forbidden_refs))
    if not BOM_PATH.exists():
        raise RuntimeError(f"缺少当前BOM：{BOM_PATH}")
    current_bom_sha256 = hashlib.sha256(BOM_PATH.read_bytes()).hexdigest()
    if data.get("bomSha256") != current_bom_sha256:
        raise RuntimeError(
            f"当前BOM计数证据已过期：证据BOM SHA256={data.get('bomSha256')}，"
            f"当前BOM SHA256={current_bom_sha256}；先刷新当前计数证据。"
        )
    return {
        "available": True,
        "bomSha256": data.get("bomSha256"),
        "countEvidenceSha256": hashlib.sha256(COUNT_EVIDENCE_PATH.read_bytes()).hexdigest(),
        "evidencePath": str(COUNT_EVIDENCE_PATH),
        "sourceGeneratorPath": str(CHECK / "更新当前BOM计数.py"),
        "sourceBOMPath": str(BOM_PATH),
        "sourceBOM": data.get("sourceBOM"),
        "updatedAt": data.get("updatedAt"),
        "mainPcbMountedQty": boundary.get("mainPcbMountedQty"),
        "mainPcbUniqueRefs": len(normalized_refs),
        "mainPcbNormalizedMpnTypes": boundary.get("mainPcbNormalizedMpnTypes"),
        "mainPcbRefs": normalized_refs,
    }


def load_current_board_bom_rows(refs: list[str]) -> dict[str, dict[str, str]]:
    if not BOM_PATH.exists():
        raise RuntimeError(f"缺少当前BOM：{BOM_PATH}")
    with BOM_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_ref: dict[str, dict[str, str]] = {}
    wanted = set(refs)
    for row in rows:
        ref = as_text(row.get("位号"))
        if not ref or ref not in wanted:
            continue
        if ref in by_ref:
            raise RuntimeError(f"当前BOM含重复主板ref：{ref}")
        by_ref[ref] = row
    missing = sorted(wanted - set(by_ref))
    if missing:
        raise RuntimeError("计数证据中的主板ref不在当前BOM：" + ", ".join(missing))
    return by_ref


def extract_cli_path() -> Path:
    source = CHECK / "原生更新.py"
    if not source.exists():
        raise RuntimeError(f"找不到 CLI 配置来源：{source}")
    match = re.search(r"^\s*CLI\s*=\s*r?['\"]([^'\"]+)['\"]", source.read_text(encoding="utf-8-sig"), re.M)
    if not match:
        raise RuntimeError(f"无法从 {source.name} 解析 CLI 路径")
    cli = Path(match.group(1))
    if not cli.exists():
        raise RuntimeError(f"现有 CLI 路径不可读：{cli}")
    return cli


def invoke(cli: Path, session: str, code: str, timeout_ms: int = 60000) -> Any:
    # 当前 CLI 的 --timeout 上限为 60000 ms；所有调用统一夹紧到官方范围。
    timeout_ms = min(max(int(timeout_ms), 1), 60000)
    proc = subprocess.run(
        [str(cli), "invoke", "--session", session, "--ext-uuid", "eda", "--timeout", str(timeout_ms), "--code", code],
        capture_output=True,
        timeout=timeout_ms / 1000 + 10,
    )
    raw = proc.stdout.decode("utf-8-sig", errors="replace").strip()
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CLI 返回无法解析 JSON，exit={proc.returncode}: {raw[:800]}") from exc
    if proc.returncode != 0 or not response.get("ok"):
        raise RuntimeError(f"CLI invoke 失败：{response}")
    return response.get("value")


def upload_json(cli: Path, session: str, global_name: str, data: Any) -> None:
    encoded = base64.b64encode(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).decode("ascii")
    invoke(cli, session, f"globalThis.{global_name}='';return true;")
    for start in range(0, len(encoded), 18000):
        piece = json.dumps(encoded[start:start + 18000])
        invoke(cli, session, f"globalThis.{global_name}+={piece};return true;")
    invoke(cli, session, f"globalThis.{global_name}=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(globalThis.{global_name}),c=>c.charCodeAt(0))));return true;")


def api_readback(cli: Path, session: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [{"ref": e["ref"], "libUuid": e.get("libUuid"), "libraryUuid": e.get("libraryUuid") or DEFAULT_LIBRARY_UUID, "libType": e.get("libType"), "symbolUuid": e.get("symbolUuid"), "footprintUuid": e.get("footprintUuid")} for e in entries]
    missing_static = [e["ref"] for e in requests if not e["libUuid"]]
    valid = [e for e in requests if e["libUuid"]]
    if not valid:
        return {"checked": 0, "readableRefs": [], "missingBindingRefs": missing_static, "unreadableRefs": []}
    upload_json(cli, session, "BMS_SHEET_BINDINGS", valid)
    invoke(cli, session, "globalThis.BMS_SHEET_DEVICE_CACHE=globalThis.BMS_SHEET_DEVICE_CACHE||{};return true;")
    results: list[dict[str, Any]] = []
    for start in range(0, len(valid), 48):
        js = r"""
const batch=globalThis.BMS_SHEET_BINDINGS.slice(START,END),out=[];
for(const e of batch){
 const k=String(e.libraryUuid)+'/'+String(e.libUuid);
 try{
  const d=e.libType==='2'?await eda.lib_Device.get(e.libUuid):await eda.lib_Device.get(e.libUuid,e.libraryUuid);
  if(!d){out.push({ref:e.ref,ok:false,reason:'lib_Device.get returned empty'});continue;}
  const symbolUuid=d.association?.symbolUuid||null,footprintUuid=d.association?.footprintUuid||null;
  const mismatches=[];
  if(String(d.uuid)!==String(e.libUuid))mismatches.push('device UUID');
  if(String(d.libraryUuid)!==String(e.libraryUuid))mismatches.push('library UUID');
  if(String(symbolUuid)!==String(e.symbolUuid))mismatches.push('symbol UUID');
  if(String(footprintUuid)!==String(e.footprintUuid))mismatches.push('footprint UUID');
  if(mismatches.length)out.push({ref:e.ref,ok:false,reason:'association mismatch: '+mismatches.join(', '),actual:{uuid:d.uuid,libraryUuid:d.libraryUuid,symbolUuid,footprintUuid}});
  else{globalThis.BMS_SHEET_DEVICE_CACHE[k]=d;out.push({ref:e.ref,ok:true,actual:{uuid:d.uuid,libraryUuid:d.libraryUuid,symbolUuid,footprintUuid,libraryType:d.libraryType,name:d.name}});}
 }catch(err){out.push({ref:e.ref,ok:false,reason:String(err)});}
}
return out;
""".replace("START", str(start)).replace("END", str(min(start + 48, len(valid))))
        value = invoke(cli, session, js, timeout_ms=60000)
        if not isinstance(value, list):
            raise RuntimeError(f"器件库读取返回非数组：{value!r}")
        results.extend(value)
    readable = [x["ref"] for x in results if x.get("ok")]
    unreadable = [{"ref": x.get("ref"), "reason": x.get("reason"), "actual": x.get("actual")} for x in results if not x.get("ok")]
    return {"checked": len(valid), "readableRefs": readable, "missingBindingRefs": missing_static, "unreadableRefs": unreadable, "deviceReadback": [x for x in results if x.get("ok")]}

def make_layout(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["pageGroup"], []).append(entry)
    group_names = sorted(groups)
    group_rows: list[list[tuple[str, int]]] = []
    for i in range(0, len(group_names), GROUP_COLUMNS):
        group_rows.append([(name, math.ceil(len(groups[name]) / CELL_COLUMNS)) for name in group_names[i:i + GROUP_COLUMNS]])
    row_y = 1800
    positions: dict[str, tuple[int, int]] = {}
    group_width = CELL_COLUMNS * CELL_X + GROUP_GAP_X
    for row in group_rows:
        row_height = max((nrows * CELL_Y + GROUP_GAP_Y for _, nrows in row), default=GROUP_GAP_Y)
        for col, (name, _) in enumerate(row):
            x0 = 1800 + col * group_width
            for index, entry in enumerate(groups[name]):
                x = x0 + (index % CELL_COLUMNS) * CELL_X
                y = row_y + (index // CELL_COLUMNS) * CELL_Y
                positions[entry["ref"]] = (x, y)
        row_y += row_height
    output = []
    for entry in entries:
        x, y = positions[entry["ref"]]
        output.append({**entry, "x": x, "y": y})
    return output


def make_annotations(layout: list[dict[str, Any]], non_electrical_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """准备页内功能组标题和说明；板外采购件不进入主板施工清单。"""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for entry in layout:
        by_group.setdefault(entry["pageGroup"], []).append(entry)
    annotations: list[dict[str, Any]] = []
    for group, items in sorted(by_group.items()):
        x = min(int(item["x"]) for item in items)
        y = min(int(item["y"]) for item in items)
        annotations.append({
            "kind": "functional_group_title",
            "x": x,
            "y": max(0, y - 560),
            "content": group,
            "fontSize": 14,
            "bold": True,
            "addIntoBom": False,
        })
        annotations.append({
            "kind": "functional_group_description",
            "x": x,
            "y": max(0, y - 360),
            "content": "按现行连接JSON映射器件端号与网络；端口为跨组连接标记。",
            "fontSize": 9,
            "bold": False,
            "addIntoBom": False,
        })
    if non_electrical_items:
        note_x = 1800
        note_y = max((int(item["y"]) for item in layout), default=1800) + GROUP_GAP_Y
        annotations.append({
            "kind": "non_electrical_heading",
            "x": note_x,
            "y": note_y,
            "content": "外置维护件与装配采购注记（无电气引脚）",
            "fontSize": 14,
            "bold": True,
            "addIntoBom": False,
        })
        for index, item in enumerate(non_electrical_items, 1):
            mpn = item.get("mpn") or "MPN待定"
            value = item.get("value") or ""
            note = item.get("note") or ""
            content = (
                f"{item['ref']} | {mpn} | {value} | BOM={'YES' if item.get('addIntoBom') else 'NO'}"
                f" | PCB={'YES' if item.get('addIntoPcb') else 'NO'}"
            )
            if note:
                content += f" | {note}"
            annotations.append({
                "kind": "non_electrical_item",
                "ref": item["ref"],
                "x": note_x,
                "y": note_y + index * 120,
                "content": content,
                "fontSize": 8,
                "bold": False,
                "addIntoBom": bool(item.get("addIntoBom")),
            })
    return annotations


def all_stub_intersections(pin_rows: list[dict[str, Any]], max_report: int = 50) -> list[dict[str, Any]]:
    segments = []
    ports: dict[tuple[float, float], dict[str, Any]] = {}
    collisions = []
    for p in pin_rows:
        if p.get("net") is None:
            continue
        x, y = float(p["x"]), float(p["y"])
        rot = int(round(float(p.get("rotation") or 0))) % 360
        if rot == 0:
            q = (x + 35, y)
        elif rot == 180:
            q = (x - 35, y)
        elif rot == 90:
            q = (x, y + 35)
        elif rot == 270:
            q = (x, y - 35)
        else:
            collisions.append({"reason": "non-orthogonal pin orientation", "ref": p["ref"], "pin": p["pin"], "rotation": rot})
            continue
        prior = ports.get(q)
        if prior:
            collisions.append({"reason": "duplicate net-port coordinate", "a": prior, "b": {"ref": p["ref"], "pin": p["pin"], "net": p["net"]}, "xy": q})
        else:
            ports[q] = {"ref": p["ref"], "pin": p["pin"], "net": p["net"]}
        segments.append({"ref": p["ref"], "pin": p["pin"], "net": p["net"], "a": (x, y), "b": q})

    def intersection(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ax1, ay1 = a["a"]
        ax2, ay2 = a["b"]
        bx1, by1 = b["a"]
        bx2, by2 = b["b"]
        ah = ay1 == ay2
        bh = by1 == by2
        if ah and bh:
            return ay1 == by1 and max(min(ax1, ax2), min(bx1, bx2)) <= min(max(ax1, ax2), max(bx1, bx2))
        if not ah and not bh:
            return ax1 == bx1 and max(min(ay1, ay2), min(by1, by2)) <= min(max(ay1, ay2), max(by1, by2))
        h, v = (a, b) if ah else (b, a)
        hx1, hy = h["a"]
        hx2, _ = h["b"]
        vx, vy1 = v["a"]
        _, vy2 = v["b"]
        return min(hx1, hx2) <= vx <= max(hx1, hx2) and min(vy1, vy2) <= hy <= max(vy1, vy2)

    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            if a["ref"] == b["ref"] and a["pin"] == b["pin"]:
                continue
            if intersection(a, b):
                collisions.append({
                    "reason": "overlapping 35-unit pin stubs",
                    "a": {"ref": a["ref"], "pin": a["pin"], "net": a["net"]},
                    "b": {"ref": b["ref"], "pin": b["pin"], "net": b["net"]},
                })
                if len(collisions) >= max_report:
                    return collisions
    return collisions


def project_snapshot(roots: list[str]) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*.eprj3"):
            try:
                stat = path.stat()
                digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
                snapshot[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns, digest)
            except (OSError, PermissionError):
                continue
    return snapshot


def require_count_closure(signature: str, count_snapshot: dict[str, Any], source_count: int) -> dict[str, Any]:
    if not COUNT_CLOSURE_PATH.exists():
        raise RuntimeError(
            f"未找到 {COUNT_CLOSURE_PATH.name}；请先完成本轮BOM/源ref计数闭合。" 
            f"准备输入摘要 sourceInputSha256={signature}，源器件数={source_count}。"
        )
    closure = read_json(COUNT_CLOSURE_PATH)
    if closure.get("closed") is not True:
        raise RuntimeError(f"{COUNT_CLOSURE_PATH.name} 未标记 closed=true")
    if closure.get("sourceInputSha256") != signature:
        raise RuntimeError(f"计数闭合源摘要过期：当前 {signature}，闭合文件 {closure.get('sourceInputSha256')}")
    bom_hash = count_snapshot.get("bomSha256")
    if bom_hash and closure.get("bomSha256") != bom_hash:
        raise RuntimeError(f"BOM计数证据摘要不一致：当前 {bom_hash}，闭合文件 {closure.get('bomSha256')}")
    if closure.get("sourceRefCount") != source_count:
        raise RuntimeError(f"闭合文件 sourceRefCount={closure.get('sourceRefCount')}，当前={source_count}")
    return closure


def prepare(args: argparse.Namespace) -> int:
    entries, issues, stats = source_rows()
    count_source = stats["boardOnlyRefSource"]
    signature = input_signature(entries, count_source, stats.get("nonElectricalItems"))
    missing = [e["ref"] for e in entries if not e.get("libUuid")]
    report: dict[str, Any] = {
        "mode": "prepare",
        "status": "source_issues" if issues else "missing_bindings" if missing else "ready_for_offline_review",
        "sourceInputSha256": signature,
        **stats,
        "electricalComponentCount": len(entries),
        "componentCount": len(entries),
        "boardOnlyFilter": {
            "sourceEvidencePath": count_source["evidencePath"],
            "sourceGeneratorPath": count_source["sourceGeneratorPath"],
            "sourceBOMPath": count_source["sourceBOMPath"],
            "sourceBOMSha256": count_source["bomSha256"],
            "includedRefCount": len(stats["boardOnlyRefs"]),
            "includedRefs": stats["boardOnlyRefs"],
            "excludedOutOfBoardSourceRefCount": len(stats["excludedOutOfBoardSourceRefs"]),
        },
        "missingBindingRefs": missing,
        "sourceIssues": issues,
        "currentBOMCountEvidence": count_source,
        "nonElectricalItems": stats.get("nonElectricalItems", []),
        "components": [{"ref": e["ref"], "mpn": e["mpn"], "pageGroup": e["pageGroup"], "libUuid": e["libUuid"], "symbolUuid": e["symbolUuid"], "footprintUuid": e["footprintUuid"], "expectedFootprintName": e["expectedFootprintName"], "bindingSource": e["_bindingSource"], "bindingStatus": e.get("bindingStatus"), "carrierForMpn": e.get("carrierForMpn"), "bindingNote": e.get("bindingNote"), "carrierRef": e.get("_carrierRef"), "carrierReason": e.get("_carrierReason"), "sourcePinRoleMap": e.get("_sourcePinRoleMap"), "addIntoPcb": e["addIntoPcb"], "addIntoBom": e["addIntoBom"], "pinCount": len(e["pins"]), "pins": e["pins"]} for e in entries],
    }
    report["layout"] = make_layout(entries)
    report["schematicAnnotations"] = make_annotations(report["layout"], stats.get("nonElectricalItems", []))
    report_path = CHECK / "单页eprj3施工清单.json"
    report["reportPath"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": report["status"],
        "sourceInputSha256": signature,
        "sourceRowCount": stats["sourceRowCount"],
        "boardOnlySourceRowCount": stats["boardOnlySourceRowCount"],
        "sourceRefCount": stats["sourceRefCount"],
        "electricalComponentCount": len(entries),
        "nonElectricalItemCount": stats["nonElectricalItemCount"],
        "mainPcbUniqueRefs": count_source["mainPcbUniqueRefs"],
        "mainPcbNormalizedMpnTypes": count_source["mainPcbNormalizedMpnTypes"],
        "apiCalls": False,
        "buildPerformed": False,
        "missingBindingRefs": missing,
        "reportPath": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not issues else 2


def create_skeleton(cli: Path, session: str, slug: str) -> dict[str, Any]:
    friendly = f"BMS Single Sheet {slug[-6:]}"
    code = r"""
const projectUuid=await eda.dmt_Project.createProject(
  FRIENDLY,SLUG,undefined,undefined,
  'Single-sheet BMS schematic from current nine connection JSON sources',undefined,
  EDMT_ProjectFileFormat.EPRJ3
);
if(!projectUuid)throw Error('createProject EPRJ3 returned empty UUID');
if(await eda.dmt_Project.openProject(projectUuid)!==true)throw Error('openProject failed');
await new Promise(r=>setTimeout(r,2000));
const schematics=await eda.dmt_Schematic.getAllSchematicsInfo();
const pages=await eda.dmt_Schematic.getAllSchematicPagesInfo();
if(!Array.isArray(schematics)||schematics.length!==1)throw Error('expected one schematic; got '+(schematics||[]).length);
if(!Array.isArray(pages)||pages.length!==1)throw Error('expected one page; got '+(pages||[]).length);
const pcbs=await eda.dmt_Pcb.getAllPcbsInfo();
for(const p of pcbs||[])if(await eda.dmt_Pcb.deletePcb(p.uuid)!==true)throw Error('deletePcb failed: '+p.uuid);
const left=await eda.dmt_Pcb.getAllPcbsInfo();
if((left||[]).length!==0)throw Error('PCB remains after deletion');
return {projectUuid,schematicUuid:schematics[0].uuid,pageUuid:pages[0].uuid,pageCount:pages.length,pcbCount:(left||[]).length};
""".replace("FRIENDLY", json.dumps(friendly)).replace("SLUG", json.dumps(slug))
    value = invoke(cli, session, code, timeout_ms=120000)
    if not isinstance(value, dict) or value.get("pageCount") != 1 or value.get("pcbCount") != 0:
        raise RuntimeError(f"EPRJ3单页骨架回读不合格：{value!r}")
    return value


def require_board_budget(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate full BOM coverage and report price closure without enforcing a hard cap."""
    path = CHECK / "当前主板成本.csv"
    if not path.exists():
        raise RuntimeError("缺少现行整板成本表，未建工程")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cost_refs: set[str] = set()
    entry_by_ref = {str(entry["ref"]): entry for entry in entries}
    total = Decimal(0)
    priced_parts = 0
    priced_types = 0
    missing: list[str] = []
    for row in rows:
        # Formal cost rows use semicolon-separated current designators.
        refs = {ref.strip() for ref in row.get("位号", "").split(";") if ref.strip()}
        if not refs or cost_refs.intersection(refs):
            raise RuntimeError("成本表位号缺失或重复，未建工程")
        raw_mpn = row.get("BOM原文MPN", "").strip().casefold()
        if any(ref not in entry_by_ref or str(entry_by_ref[ref].get("mpn") or "").strip().casefold() != raw_mpn for ref in refs):
            raise RuntimeError("成本表料号与本轮连接源不一致，未建工程")
        cost_refs.update(refs)
        if row.get("是否计入已证实小计") != "是":
            missing.append(row.get("规范MPN", "未知料号"))
            continue
        try:
            amount = Decimal(row["本板需求成本CNY（仅完整证据）"])
        except (KeyError, InvalidOperation) as exc:
            raise RuntimeError("成本金额字段无效，未建工程") from exc
        if not amount.is_finite() or amount < 0:
            raise RuntimeError("成本金额不是有限非负数，未建工程")
        total += amount
        priced_parts += int(row.get("板上数量", "0"))
        priced_types += 1
    if cost_refs != {str(entry["ref"]) for entry in entries}:
        raise RuntimeError("成本表未覆盖本轮全部板上位号，未建工程")
    return {
        "verifiedSmallBatchTaxIncludedTypes": priced_types,
        "verifiedSmallBatchTaxIncludedBoardParts": priced_parts,
        "verifiedConsumptionSubtotalCNY": str(total),
        "unpricedOrEstimateOnlyTypes": len(missing),
        "unpricedOrEstimateOnlyMpn": missing,
        "estimatedWholeBoardRangeCNY": "208.44281-261.099183",
        "softCostTargetCNY": "200",
        "targetMayReasonablyFloat": True,
        "quoteComplete": not missing,
    }


def build(args: argparse.Namespace) -> int:
    entries, issues, stats = source_rows()
    if issues:
        raise RuntimeError("source JSON validation failed:\n- " + "\n- ".join(issues))
    missing = [e["ref"] for e in entries if not e.get("libUuid") or not e.get("symbolUuid") or not e.get("footprintUuid")]
    if missing:
        raise RuntimeError("missing exact device/symbol/footprint bindings: " + ", ".join(missing))
    non_electrical_items = stats.get("nonElectricalItems", [])
    signature = input_signature(entries, stats["boardOnlyRefSource"], non_electrical_items)
    count_snapshot = current_count_snapshot()
    source_count = stats.get("sourceRefCount", len(entries))
    closure = require_count_closure(signature, count_snapshot, source_count)
    cost_status = require_board_budget(entries)
    cli = extract_cli_path()

    project = invoke(cli, args.session, "return await eda.dmt_Project.getCurrentProjectInfo();")
    if not isinstance(project, dict) or project.get("uuid") != EXPECTED_PROJECT_UUID:
        raise RuntimeError(f"wrong active EPRJ3 project; expected {EXPECTED_PROJECT_UUID}, got {project!r}")
    board_rows = project.get("data") or []
    if len(board_rows) != 1 or (board_rows[0].get("schematic") or {}).get("uuid") != EXPECTED_SCHEMATIC_UUID:
        raise RuntimeError("active project does not match the expected one-board/one-schematic EPRJ3 skeleton")
    pages = (board_rows[0].get("schematic") or {}).get("page") or []
    if len(pages) != 1 or pages[0].get("uuid") != EXPECTED_PAGE_UUID or board_rows[0].get("pcb") is not None:
        raise RuntimeError("active EPRJ3 project must contain exactly the expected one schematic page and zero PCB documents")
    page_uuid = EXPECTED_PAGE_UUID
    open_result = invoke(cli, args.session, "return await eda.dmt_EditorControl.openDocument(" + json.dumps(page_uuid) + ");")
    if not (
        open_result is True
        or (
            isinstance(open_result, str)
            and open_result.startswith(f"{page_uuid}@{EXPECTED_PROJECT_UUID}")
        )
    ):
        raise RuntimeError(f"failed to open the expected schematic page: {open_result!r}")
    blank = invoke(cli, args.session, r"""
const all=await eda.sch_PrimitiveComponent.getAll();
const parts=(all||[]).filter(c=>c.getState_ComponentType()==='part');
const ports=(all||[]).filter(c=>c.getState_ComponentType()==='netport');
const wires=await eda.sch_PrimitiveWire.getAll();
const texts=await eda.sch_PrimitiveText.getAll();
return {parts:parts.length,ports:ports.length,wires:(wires||[]).length,texts:(texts||[]).length};
""")
    if blank != {"parts": 0, "ports": 0, "wires": 0, "texts": 0}:
        raise RuntimeError(f"expected a blank page before construction; refusing to overwrite: {blank!r}")

    readback = api_readback(cli, args.session, entries)
    if readback["missingBindingRefs"] or readback["unreadableRefs"]:
        refs = readback["missingBindingRefs"] + [x["ref"] for x in readback["unreadableRefs"]]
        raise RuntimeError("live device association readback failed for: " + ", ".join(refs))

    layout = make_layout(entries)
    annotations = make_annotations(layout, non_electrical_items)
    upload_json(cli, args.session, "BMS_SHEET_BUILD_ENTRIES", layout)
    upload_json(cli, args.session, "BMS_SHEET_BUILD_ANNOTATIONS", annotations)
    invoke(cli, args.session, "globalThis.BMS_SHEET_CREATED={ids:[],parts:[],textIds:[],wireIds:[],portIds:[]};return true;")
    pin_rows: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []

    def cleanup_page() -> None:
        invoke(cli, args.session, r"""
const s=globalThis.BMS_SHEET_CREATED||{};
if(s.wireIds?.length)await eda.sch_PrimitiveWire.delete(s.wireIds);
const ids=[...(s.ids||[]),...(s.portIds||[])];
if(ids.length)await eda.sch_PrimitiveComponent.delete(ids);
if(s.textIds?.length)await eda.sch_PrimitiveText.delete(s.textIds);
return true;
""")

    try:
        for start in range(0, len(layout), 24):
            end = min(start + 24, len(layout))
            code = r"""
const entries=globalThis.BMS_SHEET_BUILD_ENTRIES.slice(START,END),out=[];
await eda.dmt_EditorControl.openDocument(PAGE);
for(const e of entries){
 const key=String(e.libraryUuid)+'/'+String(e.libUuid);
 const d=globalThis.BMS_SHEET_DEVICE_CACHE?.[key]||(e.libType==='2'?await eda.lib_Device.get(e.libUuid):await eda.lib_Device.get(e.libUuid,e.libraryUuid));
 if(!d)throw Error('device binding became unreadable: '+e.ref);
 const c=await eda.sch_PrimitiveComponent.create(d,e.x+20000,e.y+20000,undefined,0,false,e.addIntoBom,e.addIntoPcb);
 if(!c)throw Error('create failed: '+e.ref);
 const id=c.getState_PrimitiveId();
 globalThis.BMS_SHEET_CREATED.ids.push(id);
 const modified=await eda.sch_PrimitiveComponent.modify(id,{x:e.x,y:e.y,designator:e.ref,name:e.value,manufacturerId:e.mpn||null,uniqueId:e.ref,addIntoBom:e.addIntoBom,addIntoPcb:e.addIntoPcb,otherProperty:{'Electrical manifest':e.ref,'Design source':e.pageGroup,'Design note':e.bindingNote||e.note||''}});
 if(modified===false)throw Error('component property update failed: '+e.ref);
 const footprint=c.getState_Footprint();
 if(e.addIntoPcb&&!footprint)throw Error('PCB-bound device has no actual library footprint: '+e.ref);
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
 const actual=(pins||[]).map(p=>String(p.pinNumber));
 const expected=Object.keys(e.pins);
 const missing=expected.filter(k=>!actual.includes(k)),extra=actual.filter(k=>!Object.hasOwn(e.pins,k));
 if(missing.length||extra.length)throw Error('pin mismatch '+e.ref+' missing='+missing.join(',')+' extra='+extra.join(','));
 const pr=[];
 for(const p of pins||[]){const n=String(p.pinNumber),net=e.pins[n];pr.push({ref:e.ref,pin:n,net:net===null?null:net,x:p.x,y:p.y,rotation:p.rotation});}
 const row={ref:e.ref,id,footprint:footprint||null,pins:pr};
 globalThis.BMS_SHEET_CREATED.parts.push(row);out.push(row);
}
return out;
""".replace("START", str(start)).replace("END", str(end)).replace("PAGE", json.dumps(page_uuid))
            value = invoke(cli, args.session, code, timeout_ms=60000)
            if not isinstance(value, list):
                raise RuntimeError(f"component construction returned non-array: {value!r}")
            placed.extend(value)
            for part in value:
                pin_rows.extend(part.get("pins", []))
        collisions = all_stub_intersections(pin_rows)
        if collisions:
            raise RuntimeError("overlapping pin stubs: " + json.dumps(collisions[:12], ensure_ascii=False))

        upload_json(cli, args.session, "BMS_SHEET_PIN_ROWS", [{"ref": p["ref"], "pin": p["pin"], "net": p["net"], "x": p["x"], "y": p["y"], "rotation": p["rotation"]} for p in pin_rows])
        for start in range(0, len(pin_rows), 100):
            end = min(start + 100, len(pin_rows))
            code = r"""
const rows=globalThis.BMS_SHEET_PIN_ROWS.slice(START,END),out=[];
for(const p of rows){
 if(p.net===null){
  const c=(globalThis.BMS_SHEET_CREATED.parts||[]).find(x=>x.ref===p.ref);
  const pin=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.id)).find(x=>String(x.pinNumber)===p.pin);
  if(!pin)throw Error('NC pin missing '+p.ref+'.'+p.pin);
  pin.noConnected=true;await pin.done();continue;
 }
 const rot=((Number(p.rotation)||0)%360+360)%360,d=35;
 const xx=Number(p.x)+(rot===0?d:rot===180?-d:0);
 const yy=Number(p.y)+(rot===90?d:rot===270?-d:0);
 const wire=await eda.sch_PrimitiveWire.create([Number(p.x),Number(p.y),xx,yy],p.net);
 if(!wire)throw Error('wire creation failed '+p.ref+'.'+p.pin);
 globalThis.BMS_SHEET_CREATED.wireIds.push(wire.getState_PrimitiveId());
 const port=await eda.sch_PrimitiveComponent.createNetPort('BI',p.net,xx,yy,rot===180?0:rot===0?180:rot===90?270:90);
 if(!port)throw Error('net port creation failed '+p.ref+'.'+p.pin);
 globalThis.BMS_SHEET_CREATED.portIds.push(port.getState_PrimitiveId());
}
return {processed:rows.length};
""".replace("START", str(start)).replace("END", str(end))
            invoke(cli, args.session, code, timeout_ms=60000)

        for start in range(0, len(annotations), 80):
            end = min(start + 80, len(annotations))
            code = r"""
const rows=globalThis.BMS_SHEET_BUILD_ANNOTATIONS.slice(START,END),out=[];
for(const a of rows){
 const t=await eda.sch_PrimitiveText.create(a.x,a.y,a.content,0,undefined,undefined,a.fontSize,a.bold,false,false,1);
 if(!t)throw Error('schematic annotation creation failed: '+a.kind+' '+a.content);
 const id=t.getState_PrimitiveId();globalThis.BMS_SHEET_CREATED.textIds.push(id);out.push(id);
}
return {created:out.length};
""".replace("START", str(start)).replace("END", str(end))
            invoke(cli, args.session, code, timeout_ms=60000)

        final = invoke(cli, args.session, r"""
const saved=await eda.sch_Document.save();
const all=await eda.sch_PrimitiveComponent.getAll();
const parts=(all||[]).filter(c=>c.getState_ComponentType()==='part');
const netports=(all||[]).filter(c=>c.getState_ComponentType()==='netport');
const otherPrimitives=(all||[]).filter(c=>!['part','netport'].includes(c.getState_ComponentType()));
const expected=globalThis.BMS_SHEET_BUILD_ENTRIES.map(e=>e.ref).sort();
const actual=parts.map(c=>c.getState_Designator()).sort();
const duplicateDesignators=actual.filter((x,i)=>i>0&&x===actual[i-1]);
const missingDesignators=expected.filter(x=>!actual.includes(x));
const extraDesignators=actual.filter(x=>!expected.includes(x));
const texts=await eda.sch_PrimitiveText.getAll();
const textIds=new Set(globalThis.BMS_SHEET_CREATED.textIds||[]);
const createdTexts=(texts||[]).filter(t=>textIds.has(t.getState_PrimitiveId()));
const expectedText=globalThis.BMS_SHEET_BUILD_ANNOTATIONS.map(a=>a.content).sort();
const actualText=createdTexts.map(t=>t.getState_Content()).sort();
const missingText=expectedText.filter(x=>!actualText.includes(x));
const pages=await eda.dmt_Schematic.getAllSchematicPagesInfo();
const pcbs=await eda.dmt_Pcb.getAllPcbsInfo();
const project=await eda.dmt_Project.getCurrentProjectInfo();
return {saved,componentCount:parts.length,netPortCount:netports.length,otherPrimitiveComponentCount:otherPrimitives.length,duplicateDesignators,missingDesignators,extraDesignators,annotationCount:createdTexts.length,expectedAnnotationCount:expectedText.length,missingAnnotationCount:missingText.length,pageCount:(pages||[]).length,pcbCount:(pcbs||[]).length,projectUuid:project?.uuid};
""", timeout_ms=60000)
        expected_netport_count = sum(1 for e in entries for net in e["pins"].values() if net is not None)
        if (
            not isinstance(final, dict)
            or final.get("saved") is not True
            or final.get("projectUuid") != EXPECTED_PROJECT_UUID
            or final.get("componentCount") != len(entries)
            or final.get("netPortCount") != expected_netport_count
            or final.get("duplicateDesignators")
            or final.get("missingDesignators")
            or final.get("extraDesignators")
            or final.get("annotationCount") != final.get("expectedAnnotationCount")
            or final.get("missingAnnotationCount")
            or final.get("pageCount") != 1
            or final.get("pcbCount") != 0
        ):
            raise RuntimeError(f"schematic structural/save readback failed: {final!r}")
    except Exception:
        try:
            cleanup_page()
            invoke(cli, args.session, "await eda.sch_Document.save();return true;")
        except Exception:
            pass
        raise

    # Reopen and save before netlist export so the native connectivity cache is refreshed.
    refreshed = invoke(cli, args.session, "const opened=await eda.dmt_EditorControl.openDocument(" + json.dumps(page_uuid) + ");const saved=await eda.sch_Document.save();return {opened,saved};")
    if not isinstance(refreshed, dict) or not refreshed.get("saved") or not str(refreshed.get("opened", "")).startswith(f"{page_uuid}@{EXPECTED_PROJECT_UUID}"):
        raise RuntimeError(f"schematic reopen/save refresh failed before official netlist export: {refreshed!r}")

    # Export the official schematic netlist and compare every source pin-to-net assignment.
    netlist_init = invoke(cli, args.session, r"""
const file=await eda.sch_ManufactureData.getNetlistFile('BMS-9S4P.enet','EasyEDA');
if(!file)throw Error('getNetlistFile returned empty');
const text=await file.text();globalThis.BMS_SHEET_NETLIST_TEXT=text;
return {length:text.length};
""", timeout_ms=60000)
    if not isinstance(netlist_init, dict) or not isinstance(netlist_init.get("length"), int):
        raise RuntimeError(f"official netlist export failed: {netlist_init!r}")
    netlist_text = []
    for offset in range(0, netlist_init["length"], 24000):
        chunk = invoke(cli, args.session, f"return globalThis.BMS_SHEET_NETLIST_TEXT.slice({offset},{min(offset+24000,netlist_init['length'])});", timeout_ms=60000)
        if not isinstance(chunk, str):
            raise RuntimeError(f"official netlist chunk readback failed at {offset}")
        netlist_text.append(chunk)
    text = "".join(netlist_text)
    netlist = json.loads(text)
    actual_components = netlist.get("components", {})
    expected_refs = {e["ref"] for e in entries}
    actual_refs = set(actual_components)
    net_issues: list[dict[str, Any]] = []
    for ref in sorted(expected_refs - actual_refs):
        net_issues.append({"ref": ref, "issue": "component missing from exported netlist"})
    for ref in sorted(actual_refs - expected_refs):
        net_issues.append({"ref": ref, "issue": "unexpected component in exported netlist"})
    for entry in entries:
        comp = actual_components.get(entry["ref"])
        if not comp:
            continue
        pin_map = comp.get("pinInfoMap") or {}
        pins_by_number = {str(row.get("number")): row for row in pin_map.values()}
        props = comp.get("props") or {}
        actual_mpn = props.get("Manufacturer Part")
        if actual_mpn != entry["mpn"]:
            net_issues.append({"ref": entry["ref"], "issue": "BOM MPN property mismatch", "expected": entry["mpn"], "actual": actual_mpn})
        for pin_number, expected_net in entry["pins"].items():
            pin = pins_by_number.get(str(pin_number))
            if pin is None:
                net_issues.append({"ref": entry["ref"], "pin": str(pin_number), "issue": "pin missing from exported netlist"})
                continue
            actual_net = pin.get("net") or None
            if actual_net != expected_net:
                net_issues.append({"ref": entry["ref"], "pin": str(pin_number), "issue": "pin-to-net mismatch", "expected": expected_net, "actual": actual_net})
    netlist_path = CHECK / "当前BMS9S4P官方网表.enet"
    netlist_path.write_text(text, encoding="utf-8")
    netlist_check = {
        "status": "PASS" if not net_issues else "FAIL",
        "componentCount": len(actual_components),
        "expectedComponentCount": len(entries),
        "expectedPinAssignments": sum(len(e["pins"]) for e in entries),
        "checkedPinAssignments": sum(len(e["pins"]) for e in entries if e["ref"] in actual_components),
        "netIssueCount": len(net_issues),
        "issues": net_issues[:100],
        "netlistPath": str(netlist_path.resolve()),
    }
    (CHECK / "当前BMS9S4P网表连接核对.json").write_text(json.dumps(netlist_check, ensure_ascii=False, indent=2), encoding="utf-8")
    if net_issues:
        raise RuntimeError(f"official netlist comparison found {len(net_issues)} issues; see {netlist_path.name}")

    source_page = EPRJ3_WORK.parent / "sch" / "Schematic1" / "P1.esch2"
    if not EPRJ3_WORK.is_file() or not source_page.is_file() or source_page.stat().st_size <= 296:
        raise RuntimeError("saved EPRJ3 source files are missing or the schematic page did not persist to disk")
    destination = MODULE / "eda工程文件" / "BMS-9S4P.eprj3"
    if destination.exists() or any((destination.parent / name).exists() for name in ("sch", "pcb", "panel")):
        raise RuntimeError("EPRJ3 delivery destination already exists; refusing to overwrite existing project files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EPRJ3_WORK, destination)
    for folder in ("sch", "pcb", "panel"):
        source_folder = EPRJ3_WORK.parent / folder
        if source_folder.exists():
            shutil.copytree(source_folder, destination.parent / folder)
    # Keep the delivered manifest name and default sheet aligned with the API-created page.
    manifest = json.loads(destination.read_text(encoding="utf-8-sig"))
    manifest["name"] = "BMS-9S4P"
    manifest["default_sheet"] = page_uuid
    manifest.setdefault("config", {})["defaultSheet"] = page_uuid
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    layout_bounds = {"minX": min(x["x"] for x in layout), "minY": min(x["y"] for x in layout), "maxX": max(x["x"] for x in layout), "maxY": max(x["y"] for x in layout), "unitsPerInch": 1000}
    record = {
        "status": "created_and_netlist_verified",
        "projectUuid": EXPECTED_PROJECT_UUID,
        "projectFile": str(destination.resolve()),
        "sourceProjectFile": str(EPRJ3_WORK.resolve()),
        "pageUuid": page_uuid,
        "sourceInputSha256": signature,
        "sourceRefCount": source_count,
        "electricalComponentCount": len(entries),
        "bomCounts": count_snapshot,
        "countClosure": closure,
        "libraryReadback": {"checked": readback["checked"], "readableCount": len(readback["readableRefs"]), "deviceReadback": readback["deviceReadback"]},
        "nativeReadback": final,
        "officialNetlistCheck": netlist_check,
        "costStatus": cost_status,
        "pinCount": len(pin_rows),
        "stubCollisionCount": 0,
        "layoutBounds": layout_bounds,
        "components": [{"ref": p["ref"], "footprint": p["footprint"]} for p in placed],
        "bindingCaveats": [{"ref": e["ref"], "mpn": e["mpn"], "bindingStatus": e.get("bindingStatus"), "carrierForMpn": e.get("carrierForMpn"), "bindingNote": e.get("bindingNote")} for e in entries if e.get("bindingStatus")],
    }
    (CHECK / "单页eprj3施工记录.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="BMS主板单页 EPRJ3（离线准备/后续构建）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="只生成离线主板清单；不连接EasyEDA、不构建")
    mode.add_argument("--build", action="store_true", help="在现存 EasyEDA session 中建立单页 EPRJ3；需先完成本轮计数闭合")
    parser.add_argument("--session", help="现存 EasyEDA CLI session；仅--build必填")
    args = parser.parse_args()
    if args.prepare and args.session:
        parser.error("--prepare 是离线清单模式，不接受 --session")
    if args.build and not args.session:
        parser.error("--build 必须显式指定现存 --session")
    if args.prepare:
        return prepare(args)
    return build(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
