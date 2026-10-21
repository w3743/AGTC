"""由现行 BMS 连接 JSON 准备单页 EPRJ3 施工清单。

默认无动作；必须显式传 --prepare 或 --build。--prepare 只读源 JSON、旧映射和可选
库绑定，可选用现存 EasyEDA CLI session 只读回读库器件。--build 仅供后续执行，要求
存在与本轮 sourceInputSha256 匹配的“单页eprj3计数闭合.json”；本轮没有执行 build。

不运行 ERC、DRC、仿真或测试。项目骨架由官方 API 的 7 参数 createProject 建立，
API 自动创建的空 PCB 随后删除；仅保留一个原理图图页。
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
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


MODULE = Path(__file__).resolve().parents[2]
CHECK = MODULE / "检查"
PREP = MODULE / "原理图准备"
SOURCE_FILES = [
    "AFE元件连接.json",
    "功率元件连接.json",
    "接口元件连接.json",
    "风扇元件连接.json",
    "维护线束元件连接.json",
]
NO_MATERIAL_ANCHORS = {
    "J_BAL_SERVICE", "J_BAT_HV", "J_BUS_OUTPUT", "J_CELL_SERVICE", "J_PACK_HV"
}
BOM_ONLY_NOTE_REFS = {"CORE_BOARD", "FAN_EXT", "FH_FAN", "J_FAN_MATE"}
MAP_PATH = CHECK / "Layout器件映射.json"
LAYOUT_SEARCH_PATH = CHECK / "Layout库检索.json"
BINDING_CANDIDATES = [
    CHECK / "本轮新增选型库绑定.json",
    PREP / "本轮新增选型库绑定.json",
]
COUNT_EVIDENCE_PATH = CHECK / "当前BOM计数证据.json"
COUNT_CLOSURE_PATH = CHECK / "单页eprj3计数闭合.json"
DEFAULT_LIBRARY_UUID = "0819f05c4eef4c71ace90d822a990e87"
CELL_COLUMNS = 12
CELL_X = 1400
CELL_Y = 1050
GROUP_GAP_X = 2400
GROUP_GAP_Y = 2200
GROUP_COLUMNS = 3

# 对外置线束和维护件只复用已读到的通用原理图图形载体；源MPN、网络和端号不改。
# 元件均 addIntoPcb=false，footprint 仅随 library device 保留，不参加本板PCB。
GENERIC_DEVICE_CARRIERS = {
    "fuse2": {
        "ref": "F_FAN",
        "pinNumbers": ["1", "2"],
        "reason": "外置保险的两端原理图符号载体；复用F_FAN端号1/2，保持源MPN。此载体的封装不代表外置保险的实物封装，PCB关闭。",
    },
    "resistor2_external": {
        "ref": "R_FAN_EN_PD",
        "pinNumbers": ["1", "2"],
        "reason": "外置功率电阻的两端原理图符号载体；复用R_FAN_EN_PD端号1/2，保持源MPN与阻值。此载体封装不代表外置300W电阻的实物封装，PCB关闭。",
    },
    "connector2": {
        "ref": "J_FAN",
        "pinNumbers": ["1", "2"],
        "reason": "外置两端接插件的通用连接符号；J_FAN库器件端号1/2，保持源MPN。",
    },
    "switch6": {
        "device": {
            "libUuid": "b36fcd32dfc3463a95b635aab6de124d",
            "libraryUuid": DEFAULT_LIBRARY_UUID,
            "libType": "3",
            "symbolUuid": "4f2278df1a614559a17a0ab1eaa7b020",
            "footprintUuid": "5af22b9a4963478c9ecc88b361054d3e",
            "footprintName": "KEY-TH_SW-2P2T_L2580W0740",
        },
        "pinNumbers": ["1", "2", "3", "4", "5", "6"],
        "reason": "SW_RUN_STORE为外置6端开关，复用已检查的6端DPDT符号；端号按1..6映射，PCB关闭。",
    },
    "connector10": {
        "device": {
            "libUuid": "1a030d409aca4e499d67ad8cbc06ab1e",
            "libraryUuid": DEFAULT_LIBRARY_UUID,
            "libType": "3",
            "symbolUuid": "33f29460a7a74813b7b9611cef180646",
            "footprintUuid": "da133cd002b749f4bb5a48d47cbb8ff1",
            "footprintName": "HDR-TH_10P-P2.54-V-F-S2.54",
        },
        "pinNumbers": [str(i) for i in range(1, 11)],
        "reason": "外置10端维护连接件，复用已检查的10端通用连接符号；端号按1..10映射，PCB关闭。",
    },
    "connector12": {
        "device": {
            "libUuid": "12bda752438e4c7d9e83453b3b4fee45",
            "libraryUuid": DEFAULT_LIBRARY_UUID,
            "libType": "3",
            "symbolUuid": "f0847e66cc8b43f191d58727fa05ba9c",
            "footprintUuid": "ec98ae1f370a433299dd896c2bca292e",
            "footprintName": "KG-4.0-12PIN",
        },
        "pinNumbers": [str(i) for i in range(1, 13)],
        "reason": "外置12端维护连接件，复用已检查的12端通用连接符号；端号按1..12映射，PCB关闭。",
    },
}


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
        device_mpn = as_text(device.get("manufacturerId")) or as_text(device.get("name"))
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


def add_bom_only_note_items(
    grouped: dict[str, dict[str, Any]],
    non_electrical: dict[str, dict[str, Any]],
    issues: list[str],
) -> list[str]:
    """把当前BOM中的四个无引脚采购项作为原理图文字/BOM注记，不造假符号端子。"""
    path = PREP / "BOM.csv"
    if not path.exists():
        issues.append(f"缺少当前BOM：{path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {str(row.get("位号", "")).strip(): row for row in csv.DictReader(stream)}
    added: list[str] = []
    for ref in sorted(BOM_ONLY_NOTE_REFS):
        row = rows.get(ref)
        if not row:
            issues.append(f"当前BOM缺少应标注的外置/装配项：{ref}")
            continue
        if ref in grouped:
            # 若后续源JSON已经给出有引脚的实际电气连接，以源JSON为准，避免重复位号。
            continue
        mpn = as_text(row.get("MPN"))
        value = as_text(row.get("MPN编码标称值")) or mpn or ref
        note = as_text(row.get("备注")) or ""
        source_name = as_text(row.get("来源JSON")) or "BOM.csv"
        item = non_electrical.get(ref)
        if item:
            if item.get("mpn") and mpn and item["mpn"] != mpn:
                issues.append(f"BOM与维护注记MPN不一致：{ref}: {item['mpn']} / {mpn}")
            item["bomOnly"] = False
            if note and len(note) > len(item.get("note") or ""):
                item["note"] = note
        else:
            non_electrical[ref] = {
                "ref": ref,
                "mpn": mpn,
                "value": value,
                "note": note,
                "pageGroup": "EXTERNAL_ASSEMBLY",
                "sourceFiles": [source_name],
                "addIntoPcb": False,
                "addIntoBom": True,
                "bomOnly": True,
                "representation": "BOM-only assembly note; no invented electrical pins or library symbol",
            }
            added.append(ref)
    return added


def source_rows() -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
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
    all_source_row_count = 0
    skipped_anchors: set[str] = set()
    count_evidence = read_json(COUNT_EVIDENCE_PATH) if COUNT_EVIDENCE_PATH.exists() else {}
    board_boundary = count_evidence.get("boardMountBoundary", {}) if isinstance(count_evidence, dict) else {}
    board_refs = set(board_boundary.get("mainPcbRefs", [])) if isinstance(board_boundary, dict) else set()
    if not board_refs:
        issues.append("当前BOM计数证据缺少 boardMountBoundary.mainPcbRefs；拒绝按板上范围生成清单")
    excluded_outside_board_refs: set[str] = set()
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
            all_source_row_count += 1
            if not isinstance(raw, dict) or not raw.get("ref"):
                issues.append(f"{filename} 含无 ref 记录：{raw!r}")
                continue
            ref = str(raw["ref"]).strip()
            if ref not in board_refs:
                excluded_outside_board_refs.add(ref)
                continue
            source_row_count += 1
            if ref in NO_MATERIAL_ANCHORS:
                skipped_anchors.add(ref)
                continue
            pins = raw.get("pins")
            if isinstance(pins, dict) and not pins and filename == "维护线束元件连接.json" and raw.get("mpn"):
                item = non_electrical.get(ref)
                mpn = as_text(raw.get("mpn"))
                if item and item["mpn"] != mpn:
                    issues.append(f"非电气维护件重复 ref 的 MPN 冲突：{ref}: {item['mpn']} / {mpn}")
                elif item:
                    item["sourceFiles"].append(filename)
                else:
                    non_electrical[ref] = {
                        "ref": ref,
                        "mpn": mpn,
                        "value": as_text(raw.get("value")) or mpn,
                        "note": as_text(raw.get("note")) or "",
                        "pageGroup": as_text(raw.get("pageGroup")) or "MAINT_MECHANICAL",
                        "sourceFiles": [filename],
                        "addIntoPcb": False,
                        "addIntoBom": True,
                        "representation": "non-electrical assembly note; no invented pins or library symbol",
                    }
                continue
            if not isinstance(pins, dict) or not pins:
                issues.append(f"{ref} 缺少 pins 映射（{filename}）")
                continue
            pins = {str(k): (None if v is None else str(v)) for k, v in pins.items()}
            source_pin_role_map = None
            # F1 的A/B是源端子角色，不是库符号端号；原理图用F_FAN已读回的1/2pin保险符号。
            if ref == "F1" and set(pins) == {"A", "B"}:
                source_pin_role_map = {"A": "1", "B": "2"}
                pins = {source_pin_role_map[role]: pins[role] for role in ("A", "B")}
            old = grouped.get(ref)
            if old:
                if old["pins"] != pins:
                    issues.append(f"重复 ref 引脚/网络冲突：{ref}（{', '.join(old['_sourceFiles'] + [filename])}）")
                    continue
                old_mpn = as_text(old.get("mpn"))
                new_mpn = as_text(raw.get("mpn"))
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
                "mpn": as_text(raw.get("mpn")),
                "pins": pins,
                "_sourcePinRoleMap": source_pin_role_map,
                "_sourceFiles": [filename],
            }

    entries: list[dict[str, Any]] = []
    for ref, raw in grouped.items():
        old_raw = mapping.get(ref, {})
        source_mpn = as_text(raw.get("mpn"))
        mapped_mpn = as_text(old_raw.get("mpn"))
        # Tmp-only safety: never inherit a stale device UUID when the same ref changed MPN.
        old = old_raw if ((source_mpn and source_mpn == mapped_mpn) or (not source_mpn and not mapped_mpn)) else {}
        source_file = raw["_sourceFiles"][0]
        key = as_text(raw.get("libraryKey")) or as_text(raw.get("libraryKeyName"))
        key_bind = bind_key.get(key) if key else None
        if not key_bind and raw.get("mpn"):
            key_bind = bind_key.get(str(raw["mpn"]))
        saved_search_bind = exact_layout_search_binding(key, as_text(raw.get("mpn"))) if not key_bind else None
        if not key_bind and saved_search_bind:
            key_bind = saved_search_bind
        new_bind = bind_ref.get(ref) or key_bind
        # 本轮显式绑定优先，其次使用源字段，再次复用“同一现行 ref”的旧映射。
        old_device = exact_old_mapping_device(old, raw.get("mpn"))
        direct = {
            "libUuid": as_uuid(raw.get("libUuid")),
            "libraryUuid": as_uuid(raw.get("libraryUuid")),
            "libType": as_text(raw.get("libType")),
        }
        resolved = {}
        for field in ("libUuid", "libraryUuid", "libType"):
            old_value = (old_device or {}).get(field) or old.get(field)
            if field in {"libUuid", "libraryUuid"}:
                old_value = as_uuid(old_value)
            else:
                old_value = as_text(old_value)
            resolved[field] = (new_bind or {}).get(field) or direct.get(field) or old_value
        # Old mapping is supplemental only when its ref remains in current sources.
        add_pcb = raw.get("addIntoPcb")
        if add_pcb is None:
            add_pcb = old.get("addIntoPcb")
        if add_pcb is None:
            add_pcb = source_file != "维护线束元件连接.json"
        add_bom = raw.get("addIntoBom")
        if add_bom is None:
            add_bom = old.get("addIntoBom")
        if add_bom is None:
            add_bom = bool(raw.get("mpn"))
        page_group = as_text(raw.get("pageGroup")) or as_text(old.get("pageGroup"))
        if not page_group:
            if source_file == "维护线束元件连接.json":
                page_group = "MAINT_" + ref.split("_", 2)[1] if "_" in ref else "MAINT_HARNESS"
            else:
                page_group = Path(source_file).stem.replace("元件连接", "")
        entry = {
            "ref": ref,
            "mpn": raw.get("mpn"),
            "value": as_text(raw.get("value")) or as_text(old.get("value")) or raw.get("mpn") or ref,
            "pins": raw["pins"],
            "_sourcePinRoleMap": raw.get("_sourcePinRoleMap"),
            "note": as_text(raw.get("note")) or as_text(old.get("note")) or "",
            "pageGroup": page_group,
            "sourceFiles": sorted(set(raw["_sourceFiles"])),
            "libraryKey": key,
            "libUuid": resolved["libUuid"],
            "libraryUuid": resolved["libraryUuid"] or DEFAULT_LIBRARY_UUID,
            "libType": resolved["libType"],
            "symbolUuid": (new_bind or {}).get("symbolUuid") or as_uuid(raw.get("symbolUuid")) or (old_device or {}).get("symbolUuid") or as_uuid(old.get("symbolUuid")),
            "footprintUuid": (new_bind or {}).get("footprintUuid") or as_uuid(raw.get("footprintUuid")) or (old_device or {}).get("footprintUuid") or as_uuid(old.get("footprintUuid")),
            "addIntoPcb": bool(add_pcb),
            "addIntoBom": bool(add_bom),
            "expectedFootprintName": raw.get("expectedFootprintName") or (new_bind or {}).get("footprintName") or (old_device or {}).get("footprintName") or old.get("expectedFootprintName") or as_text((old.get("nativeFootprint") or {}).get("name")) or as_text(old.get("footprintName")),
            "_bindingSource": (new_bind or {}).get("_bindingSource") or ("new binding" if new_bind else "source" if direct["libUuid"] else f"same-ref old map {old_device['sourceField']}" if old_device else "same-ref old map" if as_uuid(old.get("libUuid")) else "missing"),
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

    # 未绑定的普通电阻/电容可复用同封装、同引脚编号的现行库载体；源MPN和值保持不变。
    # 载体只用于原理图符号与焊盘几何，后续仍须对照正式源MPN审核焊盘。
    for entry in entries:
        if entry.get("libUuid") or not entry["ref"].startswith(("R_", "C_")):
            continue
        footprint_name = entry.get("expectedFootprintName")
        if not footprint_name:
            continue
        candidates = [
            candidate for candidate in entries
            if candidate is not entry
            and candidate["ref"].startswith(entry["ref"][:2])
            and candidate.get("libUuid")
            and candidate.get("expectedFootprintName") == footprint_name
            and sorted(candidate["pins"]) == sorted(entry["pins"])
            and candidate.get("libraryUuid") == entry.get("libraryUuid")
        ]
        candidates.sort(key=lambda candidate: (candidate["pageGroup"] != entry["pageGroup"], candidate["ref"]))
        if candidates:
            carrier = candidates[0]
            for field in ("libUuid", "libraryUuid", "libType", "symbolUuid", "footprintUuid"):
                if carrier.get(field):
                    entry[field] = carrier[field]
            entry["_bindingSource"] = f"same-footprint carrier:{carrier['ref']}"
            entry["_carrierRef"] = carrier["ref"]
            entry["_carrierReason"] = (
                f"Reuse existing {footprint_name} symbol/footprint with identical pin numbering; "
                f"keep selected MPN {entry.get('mpn')} and source value unchanged."
            )

    by_ref = {entry["ref"]: entry for entry in entries}
    generic_bindings: list[dict[str, Any]] = []
    for entry in entries:
        # F1/R_PRE旧映射指向未核验的符号/器件UUID；强制改用已回读的两端通用符号载体。
        if entry.get("libUuid") and entry["ref"] not in {"F1", "R_PRE"}:
            continue
        ref = entry["ref"]
        pin_numbers = sorted(entry["pins"], key=lambda value: (len(value), value))
        carrier_key = None
        carrier_ref = None
        if (ref == "F1" or ref.startswith(("F_", "FH_"))) and pin_numbers == ["1", "2"]:
            carrier_key, carrier_ref = "fuse2", GENERIC_DEVICE_CARRIERS["fuse2"]["ref"]
        elif ref == "R_PRE" and pin_numbers == ["1", "2"]:
            carrier_key, carrier_ref = "resistor2_external", GENERIC_DEVICE_CARRIERS["resistor2_external"]["ref"]
        elif ref in {"J_SB175_BMS", "J_SB175_CELL", "K_SB175_CONTACTS_BMS", "K_SB175_CONTACTS_CELL"} and pin_numbers == ["1", "2"]:
            carrier_key, carrier_ref = "connector2", GENERIC_DEVICE_CARRIERS["connector2"]["ref"]
        elif ref in {"J_BAL_BATTERY", "J_BAL_BMS", "J_CELL_BATTERY", "J_CELL_BMS", "K_BAL_BATTERY", "K_BAL_BMS"} and pin_numbers == GENERIC_DEVICE_CARRIERS["connector12"]["pinNumbers"]:
            carrier_key = "connector12"
        elif ref in {"K_CELL_BATTERY", "K_CELL_BMS"} and pin_numbers == GENERIC_DEVICE_CARRIERS["connector10"]["pinNumbers"]:
            carrier_key = "connector10"
        elif ref == "SW_RUN_STORE" and pin_numbers == GENERIC_DEVICE_CARRIERS["switch6"]["pinNumbers"]:
            carrier_key = "switch6"

        if not carrier_key:
            continue
        spec = GENERIC_DEVICE_CARRIERS[carrier_key]
        carrier = by_ref.get(carrier_ref) if carrier_ref else None
        device = spec.get("device") or carrier or {}
        if not device.get("libUuid"):
            issues.append(f"通用符号载体缺少库绑定：{ref} -> {carrier_ref or carrier_key}")
            continue
        expected = spec["pinNumbers"]
        if pin_numbers != expected:
            issues.append(f"通用符号载体端号不匹配：{ref} source={pin_numbers}, carrier={expected}")
            continue
        for field in ("libUuid", "libraryUuid", "libType", "symbolUuid", "footprintUuid"):
            if device.get(field):
                entry[field] = device[field]
        # 这些外置部件只借用库符号；不把符号载体的贴片封装误报为实物封装。
        entry["expectedFootprintName"] = None
        entry["addIntoPcb"] = False
        entry["_bindingSource"] = f"generic external carrier:{carrier_ref or carrier_key}"
        entry["_carrierRef"] = carrier_ref or carrier_key
        entry["_carrierReason"] = spec["reason"]
        entry["_carrierPinNumbers"] = expected
        generic_bindings.append({"ref": ref, "carrier": carrier_ref or carrier_key, "pinNumbers": expected, "addIntoPcb": False, "reason": spec["reason"]})

    bom_only_added = add_bom_only_note_items(grouped, non_electrical, issues)
    return entries, issues, {
        "sourceRowCount": source_row_count,
        "allSourceRowCount": all_source_row_count,
        "sourceRefCount": len(grouped) + len(non_electrical),
        "electricalSymbolRefCount": len(grouped),
        "nonElectricalItemCount": len(non_electrical),
        "nonElectricalItems": [non_electrical[ref] for ref in sorted(non_electrical)],
        "bomOnlyAddedRefs": bom_only_added,
        "genericCarrierBindings": generic_bindings,
        "excludedAnchorRefs": sorted(skipped_anchors),
        "boardMountBoundarySource": "检查/当前BOM计数证据.json: boardMountBoundary.mainPcbRefs",
        "boardMountedRefCount": len(board_refs),
        "excludedOutsideMainPcbRefs": sorted(excluded_outside_board_refs),
        "missingMainPcbSourceRefs": sorted(board_refs - set(grouped) - set(non_electrical)),
        "bindingFile": str(binding_file) if binding_file else None,
        "oldMappingRows": len(mapping),
    }


def input_signature(entries: list[dict[str, Any]], non_electrical_items: list[dict[str, Any]] | None = None) -> str:
    normalized = [{k: e.get(k) for k in (
        "ref", "mpn", "value", "pins", "pageGroup", "sourceFiles", "libraryKey",
        "libUuid", "libraryUuid", "libType", "symbolUuid", "footprintUuid",
        "expectedFootprintName", "_carrierRef", "_carrierReason", "_carrierPinNumbers", "_sourcePinRoleMap", "note", "addIntoPcb", "addIntoBom",
    )} for e in entries]
    non_electrical = [{k: e.get(k) for k in (
        "ref", "mpn", "value", "note", "pageGroup", "sourceFiles", "addIntoPcb", "addIntoBom", "representation", "bomOnly",
    )} for e in (non_electrical_items or [])]
    return canonical_sha({"components": normalized, "nonElectricalItems": non_electrical})


def current_count_snapshot() -> dict[str, Any]:
    if not COUNT_EVIDENCE_PATH.exists():
        return {"available": False}
    data = read_json(COUNT_EVIDENCE_PATH)
    return {
        "available": True,
        "bomSha256": data.get("bomSha256"),
        "counts": data.get("counts", {}),
        "updatedAt": data.get("updatedAt"),
    }


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
    requests = [{"ref": e["ref"], "libUuid": e.get("libUuid"), "libraryUuid": e.get("libraryUuid") or DEFAULT_LIBRARY_UUID, "libType": e.get("libType")} for e in entries]
    missing_static = [e["ref"] for e in requests if not e["libUuid"]]
    valid = [e for e in requests if e["libUuid"]]
    if not valid:
        return {"checked": 0, "readableRefs": [], "missingBindingRefs": missing_static, "unreadableRefs": []}
    upload_json(cli, session, "BMS_SHEET_BINDINGS", valid)
    invoke(cli, session, "globalThis.BMS_SHEET_DEVICE_CACHE=globalThis.BMS_SHEET_DEVICE_CACHE||{};return true;")
    results: list[dict[str, Any]] = []
    for start in range(0, len(valid), 80):
        js = r"""
const batch=globalThis.BMS_SHEET_BINDINGS.slice(START,END),out=[];
for(const e of batch){
 const k=String(e.libraryUuid)+'/'+String(e.libUuid);
 try{
  // 系统库（libType 2）按deviceUuid读取；用户库（libType 3）需同时传libraryUuid。
  const d=e.libType==='2'?await eda.lib_Device.get(e.libUuid):await eda.lib_Device.get(e.libUuid,e.libraryUuid);
  if(d){globalThis.BMS_SHEET_DEVICE_CACHE[k]=d;out.push({ref:e.ref,ok:true});}
  else out.push({ref:e.ref,ok:false,reason:'lib_Device.get returned empty'});
 }catch(err){out.push({ref:e.ref,ok:false,reason:String(err)});}
}
return out;
""".replace("START", str(start)).replace("END", str(min(start + 80, len(valid))))
        value = invoke(cli, session, js, timeout_ms=120000)
        if not isinstance(value, list):
            raise RuntimeError(f"器件库回读返回非数组：{value!r}")
        results.extend(value)
    readable = [x["ref"] for x in results if x.get("ok")]
    unreadable = [{"ref": x.get("ref"), "reason": x.get("reason")} for x in results if not x.get("ok")]
    return {
        "checked": len(valid),
        "readableRefs": readable,
        "missingBindingRefs": missing_static,
        "unreadableRefs": unreadable,
    }


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
    """准备页内功能组标题/说明和无引脚采购件文字注记。"""
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
    signature = input_signature(entries, stats.get("nonElectricalItems"))
    missing = [e["ref"] for e in entries if not e.get("libUuid")]
    report: dict[str, Any] = {
        "mode": "prepare",
        "status": "source_issues" if issues else "missing_bindings" if missing else "ready_for_api_readback",
        "sourceInputSha256": signature,
        **stats,
        "electricalComponentCount": len(entries),
        "componentCount": len(entries),
        "sourceRefCountIncludesNonElectricalItems": stats.get("sourceRefCount"),
        "missingBindingRefs": missing,
        "sourceIssues": issues,
        "currentBOMCountEvidence": current_count_snapshot(),
        "nonElectricalItems": stats.get("nonElectricalItems", []),
        "components": [{"ref": e["ref"], "mpn": e["mpn"], "pageGroup": e["pageGroup"], "libUuid": e["libUuid"], "symbolUuid": e["symbolUuid"], "footprintUuid": e["footprintUuid"], "expectedFootprintName": e["expectedFootprintName"], "bindingSource": e["_bindingSource"], "carrierRef": e.get("_carrierRef"), "carrierReason": e.get("_carrierReason"), "sourcePinRoleMap": e.get("_sourcePinRoleMap"), "addIntoPcb": e["addIntoPcb"], "addIntoBom": e["addIntoBom"], "pinCount": len(e["pins"]), "pins": e["pins"]} for e in entries],
    }
    if args.session:
        cli = extract_cli_path()
        roots = invoke(cli, args.session, "return await eda.sys_FileSystem.getProjectsPaths();")
        report["existingSessionRoots"] = roots if isinstance(roots, list) else []
        report["libraryReadback"] = api_readback(cli, args.session, entries)
        if issues:
            report["status"] = "source_issues"
        elif report["libraryReadback"]["missingBindingRefs"] or report["libraryReadback"]["unreadableRefs"]:
            report["status"] = "library_binding_readback_incomplete"
        else:
            report["status"] = "ready_for_count_closure"
    report["layout"] = make_layout(entries)
    report["schematicAnnotations"] = make_annotations(report["layout"], stats.get("nonElectricalItems", []))
    report_path = Path(__file__).resolve().parent / "单页eprj3施工清单.json"
    report["reportPath"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readback = report.get("libraryReadback") or {}
    summary = {
        "status": report["status"],
        "sourceInputSha256": signature,
        "sourceRowCount": stats["sourceRowCount"],
        "sourceRefCount": stats["sourceRefCount"],
        "electricalComponentCount": len(entries),
        "nonElectricalItemCount": stats["nonElectricalItemCount"],
        "missingBindingRefs": missing,
        "libraryReadback": {
            "checked": readback.get("checked", 0),
            "readableCount": len(readback.get("readableRefs", [])),
            "unreadableCount": len(readback.get("unreadableRefs", [])),
            "unreadableRefs": [row.get("ref") for row in readback.get("unreadableRefs", [])],
        },
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


def build(args: argparse.Namespace) -> int:
    entries, issues, stats = source_rows()
    if issues:
        raise RuntimeError("源JSON检查不通过：\n- " + "\n- ".join(issues))
    missing = [e["ref"] for e in entries if not e.get("libUuid")]
    if missing:
        raise RuntimeError("缺少准确库绑定，未建工程；missing refs: " + ", ".join(missing))
    non_electrical_items = stats.get("nonElectricalItems", [])
    signature = input_signature(entries, non_electrical_items)
    count_snapshot = current_count_snapshot()
    source_count = stats.get("sourceRefCount", len(entries))
    closure = require_count_closure(signature, count_snapshot, source_count)
    cli = extract_cli_path()
    roots = invoke(cli, args.session, "return await eda.sys_FileSystem.getProjectsPaths();")
    if not isinstance(roots, list) or not roots:
        raise RuntimeError(f"现存 session 未返回有效工程根目录：{roots!r}")
    readback = api_readback(cli, args.session, entries)
    if readback["missingBindingRefs"] or readback["unreadableRefs"]:
        refs = readback["missingBindingRefs"] + [x["ref"] for x in readback["unreadableRefs"]]
        raise RuntimeError("库器件 API 回读不完整，未建工程；refs: " + ", ".join(refs))

    before = project_snapshot(roots)
    slug = "bms-single-sheet-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    skeleton = create_skeleton(cli, args.session, slug)
    page_uuid = skeleton["pageUuid"]
    layout = make_layout(entries)
    annotations = make_annotations(layout, non_electrical_items)
    upload_json(cli, args.session, "BMS_SHEET_BUILD_ENTRIES", layout)
    upload_json(cli, args.session, "BMS_SHEET_BUILD_ANNOTATIONS", annotations)
    invoke(cli, args.session, "globalThis.BMS_SHEET_CREATED={ids:[],parts:[],textIds:[]};return true;")

    pin_rows: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []
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
 await eda.sch_PrimitiveComponent.modify(id,{x:e.x,y:e.y,designator:e.ref,name:e.value,manufacturerId:e.mpn||null,uniqueId:e.ref,addIntoBom:e.addIntoBom,addIntoPcb:e.addIntoPcb,otherProperty:{'Electrical manifest':e.ref,'Design source':e.pageGroup,'Design note':e.note||''}});
 const footprint=c.getState_Footprint();
 if(e.addIntoPcb&&!footprint)throw Error('PCB-bound device has no actual library footprint: '+e.ref);
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(id);
 const actual=(pins||[]).map(p=>String(p.pinNumber));
 const expected=Object.keys(e.pins);
 const missing=expected.filter(k=>!actual.includes(k)),extra=actual.filter(k=>!Object.hasOwn(e.pins,k));
 if(missing.length||extra.length)throw Error('pin mismatch '+e.ref+' missing='+missing.join(',')+' extra='+extra.join(','));
 const pr=[];
 for(const p of pins||[]){
  const n=String(p.pinNumber),net=e.pins[n];
  pr.push({ref:e.ref,pin:n,net:net===null?null:net,x:p.x,y:p.y,rotation:p.rotation});
 }
 const row={ref:e.ref,id,footprint:footprint||null,pins:pr};
 globalThis.BMS_SHEET_CREATED.parts.push(row);out.push(row);
}
return out;
""".replace("START", str(start)).replace("END", str(end)).replace("PAGE", json.dumps(page_uuid))
            value = invoke(cli, args.session, code, timeout_ms=180000)
            if not isinstance(value, list):
                raise RuntimeError(f"器件放置返回非数组：{value!r}")
            placed.extend(value)
            for part in value:
                pin_rows.extend(part.get("pins", []))
    except Exception:
        # 发生引脚/封装问题时，先清理此空白新页已放置的器件，避免留下误导性半成品。
        try:
            invoke(cli, args.session, "const ids=globalThis.BMS_SHEET_CREATED?.ids||[];if(ids.length)await eda.sch_PrimitiveComponent.delete(ids);return ids.length;")
        except Exception:
            pass
        raise

    collisions = all_stub_intersections(pin_rows)
    if collisions:
        invoke(cli, args.session, "const ids=globalThis.BMS_SHEET_CREATED?.ids||[];if(ids.length)await eda.sch_PrimitiveComponent.delete(ids);return ids.length;")
        raise RuntimeError("35单位引脚 stub 存在几何重叠，已删除空页内器件、未接线；冲突示例：" + json.dumps(collisions[:12], ensure_ascii=False))

    invoke(cli, args.session, "await eda.dmt_EditorControl.openDocument(" + json.dumps(page_uuid) + ");return true;")
    pin_payload = [{"ref": p["ref"], "pin": p["pin"], "net": p["net"], "x": p["x"], "y": p["y"], "rotation": p["rotation"]} for p in pin_rows]
    upload_json(cli, args.session, "BMS_SHEET_PIN_ROWS", pin_payload)
    for start in range(0, len(pin_payload), 100):
        end = min(start + 100, len(pin_payload))
        code = r"""
const rows=globalThis.BMS_SHEET_PIN_ROWS.slice(START,END),out=[];
for(const p of rows){
 if(p.net===null){
  const c=(globalThis.BMS_SHEET_CREATED.parts||[]).find(x=>x.ref===p.ref);
  const pin=(await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.id)).find(x=>String(x.pinNumber)===p.pin);
  if(!pin)throw Error('NC pin missing '+p.ref+'.'+p.pin);pin.noConnected=true;await pin.done();continue;
 }
 const rot=((Number(p.rotation)||0)%360+360)%360,d=35;
 const xx=Number(p.x)+(rot===0?d:rot===180?-d:0);
 const yy=Number(p.y)+(rot===90?d:rot===270?-d:0);
 await eda.sch_PrimitiveWire.create([Number(p.x),Number(p.y),xx,yy],p.net);
 const port=await eda.sch_PrimitiveComponent.createNetPort('BI',p.net,xx,yy,rot===180?0:rot===0?180:rot===90?270:90);
 if(!port)throw Error('net port creation failed '+p.ref+'.'+p.pin);
}
return {processed:rows.length};
""".replace("START", str(start)).replace("END", str(end))
        invoke(cli, args.session, code, timeout_ms=180000)

    # 功能组标题、说明与全部无引脚外置/装配采购注记均写入单页原理图。
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
        invoke(cli, args.session, code, timeout_ms=180000)

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
return {saved,componentCount:parts.length,netPortCount:netports.length,otherPrimitiveComponentCount:otherPrimitives.length,duplicateDesignators,missingDesignators,extraDesignators,annotationCount:createdTexts.length,expectedAnnotationCount:expectedText.length,missingAnnotationCount:missingText.length,pageCount:(pages||[]).length,pcbCount:(pcbs||[]).length};
""", timeout_ms=180000)
    expected_netport_count = sum(1 for e in entries for net in e["pins"].values() if net is not None)
    if (
        not isinstance(final, dict)
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
        raise RuntimeError(f"施工后原生结构回读不合格：{final!r}")
    time.sleep(1)
    after = project_snapshot(roots)
    changed = [path for path, state in after.items() if before.get(path) != state]
    if len(changed) != 1:
        raise RuntimeError(f"无法唯一确定新建 EPRJ3 工程文件：changed={changed}")
    record = {
        "status": "created",
        "projectUuid": skeleton["projectUuid"],
        "projectFile": changed[0],
        "pageUuid": page_uuid,
        "sourceInputSha256": signature,
        "sourceRefCount": source_count,
        "electricalComponentCount": len(entries),
        "bomCounts": count_snapshot,
        "countClosure": closure,
        "libraryReadback": {"checked": readback["checked"], "readableCount": len(readback["readableRefs"])},
        "nativeReadback": final,
        "pinCount": len(pin_rows),
        "stubCollisionCount": 0,
        "components": [{"ref": p["ref"], "footprint": p["footprint"]} for p in placed],
    }
    (CHECK / "单页eprj3施工记录.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BMS 单页 EPRJ3 施工（准备/后续构建）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="只读解析五份现行源 JSON，输出动态绑定/计数清单")
    mode.add_argument("--build", action="store_true", help="在现存 EasyEDA session 中建立单页 EPRJ3；需先完成本轮计数闭合")
    parser.add_argument("--session", help="现存 EasyEDA CLI session；prepare 可选，build 必须显式给出")
    args = parser.parse_args()
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
