"""Merge the current BMS connection sources into BOM.csv and 连接表.csv.

This script is intentionally conservative: it validates every source and all
reference collisions before staging either output. It must be run only after
the current source JSON set is complete and internally consistent.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "BMS模块"
SCHEMATIC = MODULE / "原理图准备"
CHECKS = MODULE / "检查"
BOM_PATH = SCHEMATIC / "BOM.csv"
CONNECTION_PATH = SCHEMATIC / "连接表.csv"
EXIT_PATH = CHECKS / "共享电感退出器件.json"
GENERAL_EXIT_PATH = CHECKS / "大量精简退出器件.json"
CONSTRUCTION_INPUT_PATH = CHECKS / "共享电感施工输入.json"
LIBRARY_EVIDENCE_PATHS = [
    CHECKS / "共享电感补充引脚.json",
    CHECKS / "共享电感选型引脚.json",
    CHECKS / "Layout新增库检索.json",
]

SNAPSHOT_DATE = datetime.date.today().isoformat()
SOURCES = OrderedDict(
    [
        ("AFE元件连接.json", "AFE/主动均衡"),
        ("功率元件连接.json", "功率/预充"),
        ("接口元件连接.json", "MCU/接口/测温/CAN"),
        ("维护线束元件连接.json", "维护线束"),
        ("共享电感矩阵连接.json", "共享电感/矩阵"),
        ("共享电感功率级连接.json", "共享电感/功率级"),
        ("共享电感控制保护连接.json", "共享电感/控制保护"),
        ("共享电感泄放连接.json", "共享电感/泄放"),
        # U21 retired the BMS fan branch; the legacy file is retained for history only.
        ("采购配套清单.json", "板外/装配配套"),
    ]
)

BOM_EXTRA_COLUMNS = ["单价（人民币）", "单套成本（人民币）", "价格依据"]
GENERIC_CARRIER_KEYS = {"R0603", "C0603", "Q_SOT23", "D_SOD123", "LED_0603"}
# These are the library device UUIDs for the generic EasyEDA carrier symbols,
# confirmed by the root agent. They identify footprint/symbol carriers only;
# they are not procurement part numbers.
GENERIC_CARRIER_UUIDS = {
    "R0603": "7696e5ef97934e7985ae08c81ed833f2",
    "C0603": "582e6743b52f4d549473833313fcf026",
}
CURRENT_DATE_FORMAT = "%Y-%m-%d"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def load_exit_refs() -> set[str]:
    refs: set[str] = set()
    for path in (EXIT_PATH, GENERAL_EXIT_PATH):
        data = read_json(path)
        if not isinstance(data, list) or any(not isinstance(ref, str) for ref in data):
            raise ValueError(f"退出器件清单应为位号字符串数组：{path}")
        refs.update(data)
    return refs


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV没有表头：{path}")
        return list(reader.fieldnames), list(reader)


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def load_library_evidence() -> dict[str, str]:
    """Load exact key -> device UUID mappings from checked library readbacks."""
    by_key: dict[str, str] = {}
    for path in LIBRARY_EVIDENCE_PATHS:
        if not path.is_file():
            continue
        data = read_json(path)
        if isinstance(data, dict) and isinstance(data.get("parts"), list):
            for entry in data["parts"]:
                if not isinstance(entry, dict):
                    continue
                key = text(entry.get("key")).strip()
                device = entry.get("device") if isinstance(entry.get("device"), dict) else {}
                uuid = text(device.get("uuid")).strip()
                if key and uuid:
                    by_key[key] = uuid
        if isinstance(data, dict) and isinstance(data.get("groups"), list):
            for group in data["groups"]:
                if not isinstance(group, dict):
                    continue
                key = text(group.get("key")).strip()
                for entry in group.get("items", []):
                    if not isinstance(entry, dict):
                        continue
                    uuid = text(entry.get("uuid")).strip()
                    if key and uuid:
                        by_key[key] = uuid
    return by_key


def load_sources() -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, dict[str, Any]], dict[str, str]]:
    items: list[tuple[str, str, dict[str, Any]]] = []
    for filename, category in SOURCES.items():
        path = SCHEMATIC / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少当前连接源：{path}")
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"连接源必须为JSON数组：{path}")
        seen_in_file: set[str] = set()
        for index, part in enumerate(data):
            # Explicitly disabled historical candidates remain traceable in source JSON but are never merged.
            if isinstance(part, dict) and part.get("active") is False:
                continue
            if not isinstance(part, dict) or not text(part.get("ref")):
                raise ValueError(f"源文件第{index + 1}项缺少位号：{path}")
            ref = text(part["ref"])
            if ref in seen_in_file:
                raise ValueError(f"同一源文件重复位号 {ref}：{path}")
            seen_in_file.add(ref)
            if not isinstance(part.get("pins"), dict):
                raise ValueError(f"位号 {ref} 没有pins映射：{path}")
            items.append((filename, category, part))

    exit_refs = load_exit_refs()

    construction_by_ref: dict[str, dict[str, Any]] = {}
    if CONSTRUCTION_INPUT_PATH.is_file():
        construction_data = read_json(CONSTRUCTION_INPUT_PATH)
        if isinstance(construction_data, list):
            construction_by_ref = {
                text(entry.get("ref")): entry
                for entry in construction_data
                if isinstance(entry, dict) and text(entry.get("ref"))
            }

    # Reject stale/active source collisions before touching either CSV. Identical
    # connection anchors may occur in two source documents; keep the last source
    # (the maintenance-harness source is later and has the assembly detail).
    by_ref: OrderedDict[str, tuple[str, str, dict[str, Any]]] = OrderedDict()
    for filename, category, part in items:
        ref = text(part["ref"])
        previous = by_ref.get(ref)
        if previous:
            old_file, _old_category, old_part = previous
            old_identity = (
                old_part.get("pins"),
                old_part.get("mpn"),
                old_part.get("value"),
                old_part.get("libraryKey"),
            )
            new_identity = (
                part.get("pins"),
                part.get("mpn"),
                part.get("value"),
                part.get("libraryKey"),
            )
            if old_identity != new_identity:
                raise ValueError(
                    "不同源文件的同位号定义冲突："
                    f"{ref}；{old_file} 与 {filename}。"
                    "请先给其中一个对象改唯一位号或明确废止，再重新合并。"
                )
        by_ref[ref] = (filename, category, part)

    if exit_refs & set(by_ref):
        overlap = ", ".join(sorted(exit_refs & set(by_ref)))
        raise ValueError(f"当前源仍包含退出器件位号：{overlap}")

    return list(by_ref.values()), construction_by_ref, load_library_evidence()


def source_library_uuid(
    part: dict[str, Any],
    construction_by_ref: dict[str, dict[str, Any]],
    library_by_key: dict[str, str],
) -> str:
    value = part.get("libUuid")
    if value:
        return text(value)
    key = text(part.get("libraryKey")).strip()
    if key in GENERIC_CARRIER_UUIDS:
        return GENERIC_CARRIER_UUIDS[key]
    if key in library_by_key:
        return library_by_key[key]

    # Construction inputs are ref-based and may lag behind source changes.
    # Reuse a UUID only when the stated procurement identity still matches;
    # this prevents old 0402 selections from being attached to current 0603s.
    constructed = construction_by_ref.get(text(part.get("ref")), {})
    current_mpn = text(part.get("mpn")).strip().casefold()
    constructed_mpn = text(constructed.get("mpn")).strip().casefold()
    if current_mpn and current_mpn == constructed_mpn:
        return text(constructed.get("libUuid"))
    return ""


def procurement_mpn(part: dict[str, Any]) -> str:
    """Return only a stated MPN, or an exact non-carrier library part key.

    Generic footprint keys (R0603/C0603) are never reported as procurement
    numbers. The source JSON remains authoritative for the part's actual MPN.
    """
    mpn = text(part.get("mpn")).strip()
    if mpn:
        return mpn
    key = text(part.get("libraryKey")).strip()
    if key and key not in GENERIC_CARRIER_KEYS:
        return key
    return ""


def include_in_bom(part: dict[str, Any]) -> bool:
    if part.get("addIntoBom") is False:
        return False
    if part.get("addIntoBom") is True:
        return True
    # Pad groups and abstract boundaries without an MPN, value, or selected
    # library item are connection anchors, not purchasable BOM line items.
    return bool(
        text(part.get("mpn")).strip()
        or text(part.get("value")).strip()
        or text(part.get("libraryKey")).strip()
    )


def bom_row(
    filename: str,
    category: str,
    part: dict[str, Any],
    old_row: dict[str, str] | None,
    bom_fields: list[str],
    construction_by_ref: dict[str, dict[str, Any]],
    library_by_key: dict[str, str],
) -> dict[str, str]:
    old_row = old_row or {}
    mpn = procurement_mpn(part)
    library_key = text(part.get("libraryKey"))
    generic_carrier = library_key in GENERIC_CARRIER_KEYS
    value = text(part.get("value"))

    row = {field: text(old_row.get(field)) for field in bom_fields}
    row.update(
        {
            "快照日期": SNAPSHOT_DATE,
            "类别": category,
            "位号": text(part.get("ref")),
            "MPN": mpn,
            "每套用量": text(old_row.get("每套用量")) or "1",
            "设计套数": text(old_row.get("设计套数")) or "1",
            "总数量": text(old_row.get("总数量")) or "1",
            "采购状态": "待询价；未采购/未验收",
            "料号状态": (
                "候选料号待核价"
                if mpn
                else "仅封装载体；实际料号待选型"
                if generic_carrier
                else "未定型；料号待选"
            ),
            "LibUUID": source_library_uuid(part, construction_by_ref, library_by_key),
            "来源JSON": filename,
            "备注": text(part.get("note")),
            "MPN编码标称值": value or text(old_row.get("MPN编码标称值")),
            "编码与备注核对": "待复核",
            "单价（人民币）": "待询价",
            "单套成本（人民币）": "待汇总",
            "价格依据": "未取得可核验的中国大陆单套器件报价",
        }
    )
    if not mpn and not generic_carrier and not value:
        row["料号状态"] = "未定型；未取得可采购料号"
    # A rerun must not erase an independently obtained quote for an unchanged
    # purchase identity. Changed or still unspecified identities keep unknowns.
    if mpn and mpn == text(old_row.get("MPN")):
        for field in BOM_EXTRA_COLUMNS + ["采购状态"]:
            if old_row.get(field):
                row[field] = text(old_row[field])
    return row


def connection_rows(
    filename: str,
    part: dict[str, Any],
    construction_by_ref: dict[str, dict[str, Any]],
    library_by_key: dict[str, str],
    fields: list[str],
) -> list[dict[str, str]]:
    common = {
        "快照日期": SNAPSHOT_DATE,
        "来源JSON": filename,
        "位号": text(part.get("ref")),
        "MPN": procurement_mpn(part),
        "LibUUID": source_library_uuid(part, construction_by_ref, library_by_key),
        "备注": text(part.get("note")),
    }
    result = []
    for pin, net in part["pins"].items():
        row = {field: "" for field in fields}
        row.update(common)
        row["引脚"] = text(pin)
        row["网络"] = text(net)
        result.append(row)
    return result


def write_staged_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    handle, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    with temp_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return temp_path


def main() -> None:
    current_parts, construction_by_ref, library_by_key = load_sources()
    bom_fields, old_bom_rows = read_csv(BOM_PATH)
    connection_fields, old_connection_rows = read_csv(CONNECTION_PATH)
    exit_refs = load_exit_refs()

    required_bom = [
        "快照日期", "类别", "位号", "MPN", "每套用量", "设计套数", "总数量",
        "采购状态", "料号状态", "LibUUID", "来源JSON", "备注",
        "MPN编码标称值", "编码与备注核对",
    ]
    required_connections = ["快照日期", "来源JSON", "位号", "MPN", "引脚", "网络", "LibUUID", "备注"]
    missing_bom = [name for name in required_bom if name not in bom_fields]
    missing_connections = [name for name in required_connections if name not in connection_fields]
    if missing_bom or missing_connections:
        raise ValueError(
            f"现有CSV表头与合并器不兼容；BOM缺列={missing_bom}，连接表缺列={missing_connections}"
        )
    for name in BOM_EXTRA_COLUMNS:
        if name not in bom_fields:
            bom_fields.append(name)

    old_bom_by_ref: dict[str, dict[str, str]] = {}
    for row in old_bom_rows:
        ref = text(row.get("位号"))
        if not ref:
            raise ValueError("现有BOM含空位号行")
        if ref in old_bom_by_ref:
            raise ValueError(f"现有BOM位号重复：{ref}")
        old_bom_by_ref[ref] = row

    current_by_ref = {text(part["ref"]): (filename, category, part) for filename, category, part in current_parts}
    current_bom_refs = {ref for ref, (_filename, _category, part) in current_by_ref.items() if include_in_bom(part)}

    merged_bom: list[dict[str, str]] = []
    emitted_bom_refs: set[str] = set()
    for old_row in old_bom_rows:
        ref = text(old_row.get("位号"))
        if ref in exit_refs:
            continue
        current = current_by_ref.get(ref)
        if current:
            if ref in current_bom_refs:
                filename, category, part = current
                merged_bom.append(
                    bom_row(filename, category, part, old_row, bom_fields, construction_by_ref, library_by_key)
                )
                emitted_bom_refs.add(ref)
            continue
        row = {field: text(old_row.get(field)) for field in bom_fields}
        row["快照日期"] = SNAPSHOT_DATE
        row.setdefault("单价（人民币）", "待询价")
        row.setdefault("单套成本（人民币）", "待汇总")
        if not row.get("单价（人民币）"):
            row["单价（人民币）"] = "待询价"
        if not row.get("单套成本（人民币）"):
            row["单套成本（人民币）"] = "待汇总"
        if not row.get("价格依据"):
            row["价格依据"] = "未取得可核验的中国大陆单套器件报价"
        merged_bom.append(row)
        emitted_bom_refs.add(ref)

    for filename, category, part in current_parts:
        ref = text(part["ref"])
        if not include_in_bom(part) or ref in emitted_bom_refs:
            continue
        merged_bom.append(
            bom_row(filename, category, part, old_bom_by_ref.get(ref), bom_fields, construction_by_ref, library_by_key)
        )
        emitted_bom_refs.add(ref)

    if len(emitted_bom_refs) != len(merged_bom):
        raise ValueError("合并后的BOM仍存在重复位号")

    current_refs = set(current_by_ref)
    merged_connections: list[dict[str, str]] = []
    for old_row in old_connection_rows:
        ref = text(old_row.get("位号"))
        # Drop stale rows from the retired fan source; the historical JSON remains
        # on disk but is not part of the current BMS connection table.
        if text(old_row.get("来源JSON")) == "风扇元件连接.json":
            continue
        if ref in exit_refs or ref in current_refs:
            continue
        row = {field: text(old_row.get(field)) for field in connection_fields}
        row["快照日期"] = SNAPSHOT_DATE
        merged_connections.append(row)
    for filename, _category, part in current_parts:
        merged_connections.extend(
            connection_rows(filename, part, construction_by_ref, library_by_key, connection_fields)
        )

    staged_bom = write_staged_csv(BOM_PATH, bom_fields, merged_bom)
    staged_connections = write_staged_csv(CONNECTION_PATH, connection_fields, merged_connections)
    try:
        os.replace(staged_bom, BOM_PATH)
        os.replace(staged_connections, CONNECTION_PATH)
    finally:
        staged_bom.unlink(missing_ok=True)
        staged_connections.unlink(missing_ok=True)

    print(
        "合并完成："
        f"BOM {len(merged_bom)}行/{len(emitted_bom_refs)}个唯一位号；"
        f"连接表 {len(merged_connections)}行；"
        f"当前源 {len(current_parts)}个唯一位号，退出器件 {len(exit_refs)}个。"
    )
    print("同料号既有报价保留；新料号未核价仍标待询价，没有据此声明成本达标。")


if __name__ == "__main__":
    main()
