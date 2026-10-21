#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成整车器件主档；只读取原生索引、采购补充、显式补充与商城报价。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LIB = Path(__file__).resolve().parents[1]
POWER = LIB.parent
PROJECT = POWER.parent
NATIVE_PATH = LIB / "索引" / "元件总表.json"
PROCUREMENT_PATH = LIB / "索引" / "采购补充.json"
SUPPLEMENT_PATH = LIB / "整车器件补充.json"
QUOTE_PATH = LIB / "商城报价.json"
OUTPUT_PATH = LIB / "索引" / "项目器件主档.json"
OUTPUT_MD_PATH = LIB / "索引" / "项目器件主档.md"
ALLOWED_STATUSES = {"selected", "current", "candidate", "DNP", "obsolete", "unresolved"}
GENERIC_MAKERS = {
    "", "unknown", "generic", "various", "tbd", "n/a", "na", "待确认", "未知", "通用", "未指定"
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return value


def norm(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def canonical_maker(value: Any, aliases: dict[str, str]) -> str | None:
    key = norm(value)
    if not key or key in GENERIC_MAKERS:
        return None
    return aliases.get(key, key)


def parse_mpn_brand_prefix(value: Any, prefixes: list[dict[str, Any]], aliases: dict[str, str]) -> tuple[str, str, str] | None:
    """Split only explicitly maintained `brand + MPN` source strings."""
    original = str(value or "").strip()
    normalized = norm(original)
    candidates = sorted(
        (entry for entry in prefixes if norm(entry.get("prefix"))),
        key=lambda entry: len(norm(entry.get("prefix"))),
        reverse=True,
    )
    for entry in candidates:
        prefix = str(entry["prefix"]).strip()
        prefix_key = norm(prefix)
        if normalized.startswith(prefix_key + " "):
            suffix = original[len(prefix):].strip()
            manufacturer = str(entry.get("canonical") or "").strip()
            maker_key = canonical_maker(manufacturer, aliases)
            if maker_key and suffix:
                return manufacturer, maker_key, suffix
    return None


def source_path(path: str, base: Path) -> Path:
    rel = Path(path)
    if rel.is_absolute() or re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"来源必须使用相对路径：{path}")
    resolved = (base / rel).resolve()
    try:
        resolved.relative_to(PROJECT.resolve())
    except ValueError as exc:
        raise ValueError(f"来源路径越出项目根目录：{path}") from exc
    return resolved


def verify_embedded_sources(rows: list[dict[str, Any]], base: Path, label: str) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        path_text = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        if not path_text or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"{label}包含不完整的来源绑定：{row!r}")
        if path_text in seen:
            raise ValueError(f"{label}来源重复：{path_text}")
        seen.add(path_text)
        path = source_path(path_text, base)
        if not path.is_file():
            raise FileNotFoundError(f"{label}来源不存在：{path_text}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label}来源SHA不匹配：{path_text} expected={expected} actual={actual}")
        checked.append({"path": path_text.replace("\\", "/"), "sha256": actual})
    return checked


def identity_status(manufacturer: Any, mpn: Any) -> str:
    maker_ok = canonical_maker(manufacturer, {}) is not None
    mpn_ok = bool(norm(mpn))
    if maker_ok and mpn_ok:
        return "manufacturer_and_mpn_present"
    if maker_ok:
        return "mpn_missing"
    if mpn_ok:
        return "manufacturer_missing"
    return "manufacturer_and_mpn_missing"


def status_from_source(source: Any, fallback: str) -> str:
    text = norm(source)
    if any(token in text for token in ("obsolete", "eol", "停产", "废弃", "作废")):
        return "obsolete"
    if any(token in text for token in ("dnp", "do not populate", "不装", "不贴装", "不焊接")):
        return "DNP"
    if any(token in text for token in ("selected", "selected part", "已选定", "已确认", "已采用", "确定采用")):
        return "selected"
    if any(token in text for token in ("candidate", "候选", "待选", "待核", "待验证")):
        return "candidate"
    if any(token in text for token in ("unresolved", "未选型", "待选型", "待确认", "未知")):
        return "unresolved"
    return fallback


def quote_supplier_part(observation: dict[str, Any]) -> str:
    return norm(observation.get("supplier_part"))


def is_usable_quote(observation: dict[str, Any]) -> bool:
    url = str(observation.get("source_url") or "")
    return (
        observation.get("identity_verified") is True
        and observation.get("supplier") == "LCSC_CN"
        and observation.get("currency") == "CNY"
        and observation.get("acquisition") == "browser_visible_DOM"
        and bool(re.fullmatch(r"https?://item\.szlcsc\.com/\d+\.html", url))
        and bool(norm(observation.get("mpn")))
        and bool(canonical_maker(observation.get("manufacturer"), {}))
        and bool(quote_supplier_part(observation))
        and isinstance(observation.get("price_tiers"), list)
        and all(
            isinstance(tier, dict)
            and isinstance(tier.get("min_quantity"), (int, float))
            and isinstance(tier.get("unit_price"), (int, float))
            and tier["min_quantity"] > 0
            and tier["unit_price"] >= 0
            for tier in observation.get("price_tiers", [])
        )
    )


def status_value(raw: Any) -> str:
    value = norm(raw)
    mapping = {"yes": "yes", "no": "no", "true": "yes", "false": "no", "1": "yes", "0": "no"}
    return mapping.get(value, "unknown")


def line_supplier_part(row: dict[str, Any]) -> str | None:
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    for key, value in raw.items():
        key_norm = norm(key).replace(" ", "")
        if key_norm in {"supplierpart", "supplierpartnumber", "lcscpart", "lcscpartnumber", "立创料号", "供应商料号"}:
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def build() -> dict[str, Any]:
    # Verify snapshots and all evidence hashes before building the derived file.
    file_inputs = [NATIVE_PATH, PROCUREMENT_PATH, SUPPLEMENT_PATH]
    before = {path: sha256_file(path) for path in file_inputs}
    native = read_json(NATIVE_PATH)
    procurement = read_json(PROCUREMENT_PATH)
    supplement = read_json(SUPPLEMENT_PATH)

    native_sources = verify_embedded_sources(native.get("sources", []), POWER, "native index")
    procurement_sources = verify_embedded_sources(procurement.get("sources", []), POWER, "procurement index")
    manual_sources = verify_embedded_sources(supplement.get("sources", []), PROJECT, "manual supplement")

    aliases: dict[str, str] = {}
    for entry in supplement.get("manufacturer_aliases", []):
        canonical = norm(entry.get("canonical"))
        if not canonical:
            raise ValueError(f"制造商别名缺canonical：{entry!r}")
        for alias in entry.get("aliases", []):
            key = norm(alias)
            if key in aliases and aliases[key] != canonical:
                raise ValueError(f"制造商别名冲突：{alias}")
            aliases[key] = canonical

    mpn_prefixes = supplement.get("manufacturer_mpn_prefixes", [])
    prefix_keys: set[str] = set()
    for entry in mpn_prefixes:
        prefix_key = norm(entry.get("prefix"))
        canonical = entry.get("canonical")
        if not prefix_key or not canonical or not canonical_maker(canonical, aliases):
            raise ValueError(f"品牌MPN前缀记录不完整：{entry!r}")
        if prefix_key in prefix_keys:
            raise ValueError(f"品牌MPN前缀重复：{entry.get('prefix')}")
        prefix_keys.add(prefix_key)

    native_parts = native.get("parts")
    native_instances = native.get("instances")
    purchase_rows = procurement.get("records")
    if not all(isinstance(value, list) for value in (native_parts, native_instances, purchase_rows)):
        raise ValueError("原生索引或采购补充中缺少数组")
    part_by_key: dict[str, dict[str, Any]] = {}
    for part in native_parts:
        key = part.get("key")
        if not key or key in part_by_key:
            raise ValueError(f"原生part key缺失或重复：{key!r}")
        part_by_key[str(key)] = part

    instance_by_key: dict[str, dict[str, Any]] = {}
    instance_counts: Counter[str] = Counter()
    native_part_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    native_catalog_parts: list[dict[str, Any]] = []
    for part in native_parts:
        part_id = f"native:{part['key']}"
        raw_makers = part.get("manufacturer")
        if isinstance(raw_makers, str):
            raw_makers = [raw_makers]
        elif not isinstance(raw_makers, list):
            raw_makers = []
        makers = [canonical_maker(value, aliases) for value in raw_makers]
        makers = sorted({value for value in makers if value})
        mpn_key = norm(part.get("mpn"))
        exact_identity = (makers[0], mpn_key) if len(makers) == 1 and mpn_key else None
        if exact_identity:
            native_part_identity[exact_identity].append(part_id)
        if len(makers) > 1:
            native_identity_status = "manufacturer_conflict"
        elif len(makers) == 1 and mpn_key:
            native_identity_status = "manufacturer_and_mpn_present"
        elif len(makers) == 1:
            native_identity_status = "mpn_missing"
        elif mpn_key:
            native_identity_status = "manufacturer_missing"
        else:
            native_identity_status = "manufacturer_and_mpn_missing"
        native_catalog_parts.append({
            "part_id": part_id,
            "origin": "native_index",
            "status": "current",
            "manufacturer": part.get("manufacturer") or [],
            "manufacturer_identity_keys": makers,
            "manufacturer_part": part.get("mpn") or None,
            "manufacturer_part_identity_key": mpn_key or None,
            "supplier_parts": part.get("supplier_parts") or [],
            "supplier_parts_source": part.get("supplier_parts") or [],
            "native_part_key": part["key"],
            "native_instance_keys": list(part.get("instance_keys") or []),
            "modules": list(part.get("modules") or []),
            "designators": list(part.get("designators") or []),
            "bom_instance_count": part.get("bom_instance_count"),
            "identity_status": native_identity_status,
            "quote_ids": [],
            "quote_observations": [],
        })
    catalog_part_by_id = {item["part_id"]: item for item in native_catalog_parts}
    part_id_by_native_key = {key: f"native:{key}" for key in part_by_key}
    for instance in native_instances:
        key = instance.get("key")
        part_key = instance.get("part_key")
        if not key or key in instance_by_key:
            raise ValueError(f"原生instance key缺失或重复：{key!r}")
        if part_key not in part_by_key:
            raise ValueError(f"native instance引用不存在part_key：{key} -> {part_key}")
        part_instance_keys = part_by_key[part_key].get("instance_keys") or []
        if key not in part_instance_keys:
            raise ValueError(f"native part的instance_keys没有回链：{key} -> {part_key}")
        instance_by_key[str(key)] = instance
        instance_counts[str(part_key)] += 1
    for key, part in part_by_key.items():
        declared = set(part.get("instance_keys") or [])
        actual = {inst_key for inst_key, inst in instance_by_key.items() if inst.get("part_key") == key}
        if declared != actual:
            raise ValueError(f"part实例覆盖不一致：{key}")

    # Strict identity map. Never merge on MPN alone or on CAD carrier data.
    catalog_by_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identity, part_ids in native_part_identity.items():
        catalog_by_identity[identity].extend(part_ids)

    procurement_lines: list[dict[str, Any]] = []
    procurement_only_parts: dict[str, dict[str, Any]] = {}
    procurement_line_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    manufacturer_resolution_counts: Counter[str] = Counter()

    for record_index, record in enumerate(purchase_rows, start=1):
        source = str(record.get("source", ""))
        row_number = record.get("source_row")
        # Some grouped BOM rows are deliberately expanded to one-reference records;
        # preserve source_row and disambiguate them with their stable record ordinal.
        line_id = f"procurement:{source}#row={row_number}#record={record_index}"
        if not source or not isinstance(row_number, int) or line_id in procurement_line_ids:
            raise ValueError(f"采购行source/source_row缺失或重复：{line_id}")
        procurement_line_ids.add(line_id)
        qty = record.get("quantity_source")
        if not isinstance(qty, str):
            raise ValueError(f"采购行数量必须保留原始字符串：{line_id}")
        native_keys = list(record.get("native_keys") or [])
        refs = list(record.get("references") or [])
        if len(refs) == 0:
            raise ValueError(f"采购行缺位号引用：{line_id}")
        linked = []
        for native_key in native_keys:
            if native_key not in instance_by_key:
                raise ValueError(f"采购行引用不存在native instance：{line_id} -> {native_key}")
            linked.append(instance_by_key[native_key])

        maker_raw = record.get("manufacturer_source")
        mpn_raw = record.get("mpn_source")
        maker_source_key = canonical_maker(maker_raw, aliases)
        source_prefix = parse_mpn_brand_prefix(mpn_raw, mpn_prefixes, aliases)
        maker_effective = maker_raw
        maker_key = maker_source_key
        mpn_effective = mpn_raw
        mpn_key = norm(mpn_raw)
        manufacturer_resolution_basis = "source_manufacturer_field" if maker_source_key else "unresolved"
        mpn_identity_basis = "source_mpn_field"
        if source_prefix:
            prefix_maker, prefix_maker_key, prefix_mpn = source_prefix
            mpn_identity_basis = "explicit_manufacturer_mpn_prefix"
            if maker_source_key and maker_source_key != prefix_maker_key:
                manufacturer_resolution_basis = "source_manufacturer_prefix_conflict"
            elif not maker_source_key:
                maker_effective = prefix_maker
                maker_key = prefix_maker_key
                mpn_effective = prefix_mpn
                mpn_key = norm(prefix_mpn)
                manufacturer_resolution_basis = "manufacturer_resolved_from_explicit_mpn_prefix"
        supplier_part = line_supplier_part(record)
        linked_native_part_ids = sorted({part_id_by_native_key[str(item["part_key"])] for item in linked})
        part_id: str | None = None
        identity_result = identity_status(maker_effective, mpn_effective)
        identity_conflicts: list[str] = []
        association = "none"
        lifecycle = status_from_source(record.get("status_source"), "candidate")
        if lifecycle not in ALLOWED_STATUSES:
            raise ValueError(f"采购行状态分类错误：{line_id} => {lifecycle}")

        # Resolve an omitted manufacturer only through an exact native actual identity,
        # or through a manually maintained, explicit `brand + MPN` prefix. Never inspect CAD carriers.
        if not maker_key and linked and mpn_key:
            native_manufacturer_keys: set[str] = set()
            native_manufacturer_display: set[str] = set()
            native_mpn_keys: set[str] = set()
            for item in linked:
                native_mpn = item.get("mpn")
                native_prefix = parse_mpn_brand_prefix(native_mpn, mpn_prefixes, aliases)
                native_mpn_keys.add(norm(native_prefix[2]) if native_prefix else norm(native_mpn))
                native_maker = item.get("manufacturer")
                native_maker_key = canonical_maker(native_maker, aliases)
                if native_maker_key:
                    native_manufacturer_keys.add(native_maker_key)
                    native_manufacturer_display.add(str(native_maker))
                elif native_prefix:
                    native_manufacturer_keys.add(native_prefix[1])
                    native_manufacturer_display.add(native_prefix[0])
            if native_mpn_keys == {mpn_key} and len(native_manufacturer_keys) == 1:
                maker_key = next(iter(native_manufacturer_keys))
                maker_effective = sorted(native_manufacturer_display, key=norm)[0]
                manufacturer_resolution_basis = (
                    "manufacturer_resolved_from_exact_native_reference"
                    if not source_prefix
                    else "manufacturer_resolved_from_exact_native_reference_and_explicit_prefix"
                )

        if linked:
            association = "native_reference"
            linked_matches: list[bool] = []
            for item in linked:
                native_mpn_raw = item.get("mpn")
                native_prefix = parse_mpn_brand_prefix(native_mpn_raw, mpn_prefixes, aliases)
                native_maker = canonical_maker(item.get("manufacturer"), aliases)
                if not native_maker and native_prefix:
                    native_maker = native_prefix[1]
                native_mpn = norm(native_prefix[2]) if native_prefix else norm(native_mpn_raw)
                if not maker_key:
                    linked_matches.append(False)
                    identity_conflicts.append("manufacturer_missing_in_procurement")
                elif not mpn_key:
                    linked_matches.append(False)
                    identity_conflicts.append("mpn_missing_in_procurement")
                elif source_prefix and maker_source_key and maker_source_key != source_prefix[1]:
                    linked_matches.append(False)
                    identity_conflicts.append("manufacturer_prefix_conflict")
                elif maker_key != native_maker:
                    linked_matches.append(False)
                    identity_conflicts.append("manufacturer_conflict")
                elif mpn_key != native_mpn:
                    linked_matches.append(False)
                    identity_conflicts.append("mpn_conflict")
                else:
                    linked_matches.append(True)
            if linked_matches and all(linked_matches) and len(linked_native_part_ids) == 1:
                part_id = linked_native_part_ids[0]
                identity_result = "exact_manufacturer_mpn_match"
                lifecycle = "current"
            else:
                identity_result = "identity_incomplete_or_conflicting"
                lifecycle = "unresolved"
        elif maker_key and mpn_key:
            identity_key = (maker_key, mpn_key)
            matches = catalog_by_identity.get(identity_key, [])
            if len(matches) == 1:
                part_id = matches[0]
                association = "exact_identity_without_native_reference"
                identity_result = "exact_identity_match_without_reference_link"
            elif len(matches) > 1:
                identity_result = "ambiguous_exact_identity"
                identity_conflicts.append("multiple_part_ids_for_exact_identity")
                lifecycle = "unresolved"
            else:
                candidate_id = "procurement:" + sha256_bytes(
                    (maker_key + "\0" + mpn_key).encode("utf-8")
                )[:20]
                part_id = candidate_id
                association = "procurement_candidate_identity"
                identity_result = "exact_manufacturer_mpn_candidate"
                if candidate_id not in procurement_only_parts:
                    procurement_only_parts[candidate_id] = {
                        "part_id": candidate_id,
                        "origin": "procurement_supplement",
                        "status": lifecycle,
                        "manufacturer": maker_effective,
                        "manufacturer_identity_key": maker_key,
                        "manufacturer_part": mpn_effective,
                        "manufacturer_part_identity_key": mpn_key,
                        "supplier_parts": [],
                        "supplier_parts_source": [],
                        "procurement_line_ids": [],
                        "identity_status": "manufacturer_and_mpn_present",
                        "quote_ids": [],
                        "quote_observations": [],
                    }
                    catalog_by_identity[identity_key].append(candidate_id)
                candidate = procurement_only_parts[candidate_id]
                candidate["procurement_line_ids"].append(line_id)
                if supplier_part and supplier_part not in candidate["supplier_parts"]:
                    candidate["supplier_parts"].append(supplier_part)
                    candidate["supplier_parts_source"].append(supplier_part)
                if lifecycle == "selected":
                    candidate["status"] = "selected"
                elif candidate["status"] not in {"selected", "current"} and lifecycle in {"candidate", "unresolved", "obsolete", "DNP"}:
                    candidate["status"] = lifecycle
        else:
            identity_result = identity_status(maker_effective, mpn_effective)
            if identity_result == "manufacturer_and_mpn_present":
                raise AssertionError("identity classifier inconsistency")
            lifecycle = "unresolved" if not maker_key or not mpn_key else lifecycle

        status_counts[lifecycle] += 1
        manufacturer_resolution_counts[manufacturer_resolution_basis] += 1
        procurement_lines.append({
            "line_id": line_id,
            "part_id": part_id,
            "lifecycle_status": lifecycle,
            "identity_status": identity_result,
            "identity_conflicts": sorted(set(identity_conflicts)),
            "association_basis": association,
            "module": record.get("module"),
            "references": refs,
            "native_instance_keys": sorted(set(native_keys)),
            "quantity_source": qty,
            "unit_source": record.get("unit_source"),
            "scope_source": record.get("scope_source"),
            "status_source": record.get("status_source"),
            "manufacturer_source": maker_raw,
            "mpn_source": mpn_raw,
            "manufacturer_effective": maker_effective,
            "mpn_effective_for_identity": mpn_effective,
            "manufacturer_resolution_basis": manufacturer_resolution_basis,
            "mpn_identity_basis": mpn_identity_basis,
            "supplier_part_source": supplier_part,
            "relation_source": record.get("relation"),
            "source": source,
            "source_row": row_number,
            "raw": record.get("raw"),
            "notes_source": record.get("notes_source"),
            "counts_as_native_instance": False,
            "quantity_aggregation": "not_aggregated",
        })

    # Vehicle-level products with explicit manufacturer + MPN get catalog identities;
    # unknown SKU/model entries remain separate annotations with part_id=null.
    external_items: list[dict[str, Any]] = []
    supplemental_parts: dict[str, dict[str, Any]] = {}
    seen_item_ids: set[str] = set()
    source_ids = {s.get("source_id") for s in supplement.get("sources", [])}
    for item in supplement.get("items", []):
        item_id = item.get("item_id")
        status = item.get("status")
        if not item_id or item_id in seen_item_ids:
            raise ValueError(f"整车补充item_id缺失或重复：{item_id!r}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"整车补充状态非法：{item_id} => {status}")
        item_source_ids = list(item.get("source_ids") or [])
        missing_sources = [value for value in item_source_ids if value not in source_ids]
        if missing_sources:
            raise ValueError(f"整车补充引用未知source_id：{item_id} => {missing_sources}")
        quantity = item.get("quantity_per_vehicle")
        quantity_status = item.get("quantity_status")
        if quantity is None and not str(quantity_status).startswith("needs_"):
            raise ValueError(f"数量null需显式标记待核：{item_id}")
        allocations = item.get("demand_allocations", [])
        allocation_ids: set[str] = set()
        allocation_total = 0
        for allocation in allocations:
            allocation_id = allocation.get("allocation_id")
            allocation_qty = allocation.get("quantity")
            if not allocation_id or allocation_id in allocation_ids:
                raise ValueError(f"整车需求分配ID缺失或重复：{item_id} => {allocation_id!r}")
            if allocation.get("source_id") not in source_ids:
                raise ValueError(f"整车需求分配引用未知来源：{item_id} => {allocation!r}")
            if not isinstance(allocation_qty, int) or allocation_qty <= 0:
                raise ValueError(f"整车需求分配数量必须为正整数：{item_id} => {allocation!r}")
            if not allocation.get("consumer_module") or not allocation.get("consumer_instance"):
                raise ValueError(f"整车需求分配缺少消费者标识：{item_id} => {allocation!r}")
            allocation_ids.add(allocation_id)
            allocation_total += allocation_qty
        if allocations and allocation_total != quantity:
            raise ValueError(f"整车需求分配不能与唯一整车数量对齐：{item_id} allocated={allocation_total} quantity={quantity}")
        seen_item_ids.add(item_id)
        maker_key = canonical_maker(item.get("manufacturer"), aliases)
        mpn_key = norm(item.get("manufacturer_part"))
        part_id: str | None = None
        if maker_key and mpn_key:
            identity_key = (maker_key, mpn_key)
            matches = catalog_by_identity.get(identity_key, [])
            if len(matches) == 1:
                part_id = matches[0]
            elif not matches:
                part_id = "vehicle:" + sha256_bytes((maker_key + "\0" + mpn_key).encode("utf-8"))[:20]
                catalog_by_identity[identity_key].append(part_id)
                supplemental_parts[part_id] = {
                    "part_id": part_id,
                    "origin": "vehicle_supplement",
                    "status": status,
                    "manufacturer": item.get("manufacturer"),
                    "manufacturer_identity_key": maker_key,
                    "manufacturer_part": item.get("manufacturer_part"),
                    "manufacturer_part_identity_key": mpn_key,
                    "supplier_parts": [item["supplier_part"]] if item.get("supplier_part") else [],
                    "supplier_parts_source": [item["supplier_part"]] if item.get("supplier_part") else [],
                    "external_item_ids": [],
                    "identity_status": "manufacturer_and_mpn_present",
                    "quote_ids": [],
                    "quote_observations": [],
                }
            # Alias-based duplicate native identities are kept separate; do not guess which record to merge.
        if part_id and part_id in supplemental_parts:
            supplemental_parts[part_id]["external_item_ids"].append(item_id)
        external_items.append({
            **item,
            "part_id": part_id,
            "identity_status": identity_status(item.get("manufacturer"), item.get("manufacturer_part")),
            "counted_with_native_bom": False,
            "demand_allocation_total": allocation_total if allocations else None,
            "demand_allocation_semantics": (
                "allocation_only; included within quantity_per_vehicle, never add allocations again"
                if allocations else None
            ),
            "quote_ids": [],
            "current_unit_price": None,
        })

    # Attach verified quote observations by exact maker+MPN. A source SKU must match when
    # present; if absent, record the quote SKU as enrichment without filling source fields.
    quote_before = sha256_file(QUOTE_PATH) if QUOTE_PATH.exists() else None
    if QUOTE_PATH.exists():
        quote_catalog = read_json(QUOTE_PATH)
        if quote_catalog.get("schema_version") != 1 or not isinstance(quote_catalog.get("observations"), list):
            raise ValueError("商城报价.json schema不符合已约定版本")
        observations = quote_catalog["observations"]
    else:
        quote_catalog = {"schema_version": 1, "observations": []}
        observations = []

    all_parts = native_catalog_parts + list(procurement_only_parts.values()) + list(supplemental_parts.values())
    part_lookup = {part["part_id"]: part for part in all_parts}
    quote_match_audit: list[dict[str, Any]] = []
    for observation in observations:
        quote_id = observation.get("quote_id")
        if not quote_id:
            raise ValueError("商城报价 observation 缺 quote_id")
        if not is_usable_quote(observation):
            quote_match_audit.append({
                "quote_id": quote_id,
                "match_status": "excluded_unverified_or_invalid",
                "matched_part_ids": [],
                "reason": "requires verified LCSC CNY item URL, visible-DOM evidence, exact identity fields, and valid price tiers",
                "source_path": "商城报价.json",
            })
            continue
        q_maker = canonical_maker(observation.get("manufacturer"), aliases)
        q_mpn = norm(observation.get("mpn"))
        q_sku = quote_supplier_part(observation)
        identity_candidates: list[dict[str, Any]] = []
        for part in all_parts:
            maker_values = part.get("manufacturer_identity_keys")
            if maker_values is None:
                maker_values = [part.get("manufacturer_identity_key")]
            maker_match = q_maker in {value for value in maker_values if value}
            mpn_match = q_mpn == norm(part.get("manufacturer_part"))
            if maker_match and mpn_match:
                identity_candidates.append(part)
        matches: list[tuple[str, str]] = []
        conflicting_part_ids: list[str] = []
        for part in identity_candidates:
            source_skus = part.get("supplier_parts_source", part.get("supplier_parts", []))
            skus = {norm(value) for value in source_skus if value is not None}
            if q_sku in skus:
                matches.append((part["part_id"], "exact_source_supplier_part"))
            elif not skus:
                # Quote page independently verified the LCSC SKU and exact MPN+maker.
                # Keep the project source SKU null; this is an enrichment, not overwrite.
                matches.append((part["part_id"], "supplier_code_verified_from_quote"))
            else:
                conflicting_part_ids.append(part["part_id"])
        for matching_part_id, supplier_code_binding in matches:
            part = part_lookup[matching_part_id]
            part["quote_ids"].append(quote_id)
            part["quote_observations"].append({
                "quote_id": quote_id,
                "supplier": observation.get("supplier"),
                "supplier_part": observation.get("supplier_part"),
                "manufacturer": observation.get("manufacturer"),
                "manufacturer_part": observation.get("mpn"),
                "source_url": observation.get("source_url"),
                "captured_at_utc": observation.get("captured_at_utc"),
                "currency": observation.get("currency"),
                "price_tiers": observation.get("price_tiers"),
                "moq": observation.get("moq"),
                "order_multiple": observation.get("order_multiple"),
                "stock_quantity": observation.get("stock_quantity"),
                "availability": observation.get("availability"),
                "tax_status": observation.get("tax_status"),
                "shipping_included": observation.get("shipping_included"),
                "identity_verified": True,
                "supplier_code_binding_status": supplier_code_binding,
            })
        if matches:
            bindings = {binding for _, binding in matches}
            if bindings == {"exact_source_supplier_part"}:
                match_status = "exact_manufacturer_mpn_supplier_part_match"
                reason = None
            elif bindings == {"supplier_code_verified_from_quote"}:
                match_status = "exact_manufacturer_mpn_supplier_enrichment"
                reason = "manufacturer+MPN match; source supplier part remains null and the verified quote SKU is recorded as enrichment"
            else:
                match_status = "exact_manufacturer_mpn_with_mixed_supplier_binding"
                reason = "exact source-SKU match and supplier-code enrichment are separated per part; see matched_part_bindings"
                if conflicting_part_ids:
                    reason += "; other identity candidates with different recorded SKUs were rejected"
        elif conflicting_part_ids:
            match_status = "unmatched_supplier_part_conflict"
            reason = "manufacturer and MPN match, but the quote supplier part differs from recorded project supplier part(s)"
        else:
            match_status = "unmatched_no_exact_catalog_identity"
            reason = "no exact manufacturer+MPN catalog identity"
        quote_match_audit.append({
            "quote_id": quote_id,
            "match_status": match_status,
            "matched_part_ids": sorted({part_id for part_id, _ in matches}),
            "rejected_conflict_part_ids": sorted(set(conflicting_part_ids)),
            "matched_part_bindings": [
                {"part_id": part_id, "supplier_code_binding_status": binding}
                for part_id, binding in sorted(set(matches))
            ],
            "reason": reason,
            "manufacturer": observation.get("manufacturer"),
            "manufacturer_part": observation.get("mpn"),
            "supplier_part": observation.get("supplier_part"),
            "source_url": observation.get("source_url"),
            "captured_at_utc": observation.get("captured_at_utc"),
            "source_path": "商城报价.json",
        })
    for part in all_parts:
        part["quote_ids"] = sorted(set(part.get("quote_ids", [])))
        part["quote_observations"].sort(key=lambda item: (item.get("captured_at_utc") or "", item["quote_id"]))
        part["current_unit_price"] = None
        part["price_semantics"] = "null unless a user-selected purchase quantity and exact applicable quote are resolved; quote tiers remain observations"
    if len({obs.get("quote_id") for obs in observations}) != len(observations):
        raise ValueError("商城报价中quote_id重复")

    for item in external_items:
        if item["part_id"] in part_lookup:
            item["quote_ids"] = part_lookup[item["part_id"]]["quote_ids"]

    requirement_slots = []
    seen_slot_ids: set[str] = set()
    for slot in supplement.get("requirement_slots", []):
        slot_id = slot.get("slot_id")
        if not slot_id or slot_id in seen_slot_ids or slot.get("status") != "unresolved":
            raise ValueError(f"需求槽位ID重复/状态无效：{slot!r}")
        if slot.get("quantity_per_vehicle") is not None:
            raise ValueError(f"未冻结需求槽位数量应为null：{slot_id}")
        seen_slot_ids.add(slot_id)
        requirement_slots.append({**slot, "part_id": None, "counted_as_selected_part": False, "current_unit_price": None})

    project_constraints = supplement.get("project_constraints", {})
    if not isinstance(project_constraints, dict):
        raise ValueError("project_constraints必须是对象")
    for constraint_name, constraint in project_constraints.items():
        if not isinstance(constraint, dict):
            raise ValueError(f"项目约束必须是对象：{constraint_name}")
        unknown_sources = [sid for sid in constraint.get("source_ids", []) if sid not in source_ids]
        if unknown_sources:
            raise ValueError(f"项目约束引用未知来源：{constraint_name} => {unknown_sources}")
    supercap = project_constraints.get("regulator_supercapacitor")
    if supercap:
        current_count = supercap.get("native_current_count")
        candidate_total = supercap.get("candidate_total_count")
        candidate_added = supercap.get("candidate_added_count")
        max_count = supercap.get("total_count_max")
        if not all(isinstance(value, int) for value in (current_count, candidate_total, candidate_added, max_count)):
            raise ValueError("稳压超容数量约束必须使用整数")
        if candidate_total - current_count != candidate_added or candidate_total > max_count:
            raise ValueError("稳压超容候选总量/增量/总上限关系不一致")

    # Expand per-instance references to strict identity prices and separate CAD carriers.
    native_instance_records: list[dict[str, Any]] = []
    for instance in native_instances:
        part_id = part_id_by_native_key[str(instance["part_key"])]
        lifecycle = "current" if status_value(instance.get("bom")) == "yes" else (
            "DNP" if status_value(instance.get("bom")) == "no" else "unresolved"
        )
        native_instance_records.append({
            "instance_key": instance["key"],
            "part_id": part_id,
            "module": instance.get("module"),
            "reference": instance.get("ref"),
            "native_source": instance.get("source_ref"),
            "page_uuid": instance.get("page_uuid"),
            "component_id": instance.get("component_id"),
            "lifecycle_status": lifecycle,
            "bom_flag_source": instance.get("bom"),
            "pcb_flag_source": instance.get("pcb"),
            "identity": {
                "manufacturer": instance.get("manufacturer"),
                "manufacturer_part": instance.get("mpn"),
                "supplier_part_source": instance.get("supplier_part"),
                "supplier_parts_source": [instance.get("supplier_part")] if instance.get("supplier_part") else [],
                "device_value": instance.get("device_value"),
                "actual_value": instance.get("actual_value"),
                "actual_value_source": instance.get("actual_value_source"),
                "identity_status": identity_status(instance.get("manufacturer"), instance.get("mpn")),
            },
            "cad_assets": {
                "carrier_manufacturer": instance.get("carrier_manufacturer"),
                "carrier_mpn": instance.get("carrier_mpn"),
                "carrier_value": instance.get("carrier_value"),
                "effective_value_raw": instance.get("effective_value_raw"),
                "device_id": instance.get("device_id"),
                "symbol_id": instance.get("symbol_id"),
                "footprint_id": instance.get("footprint_id"),
                "device_asset": instance.get("device_asset"),
                "symbol_asset": instance.get("symbol_asset"),
                "footprint_asset": instance.get("footprint_asset"),
            },
            "quote_ids": catalog_part_by_id[part_id]["quote_ids"],
            "current_unit_price": None,
        })

    if len(native_instance_records) != len(native_instances):
        raise ValueError("native instance输出覆盖不完整")
    if len(procurement_lines) != len(purchase_rows) or len(procurement_line_ids) != len(purchase_rows):
        raise ValueError("采购补充行输出覆盖不完整或重复")
    native_summary = native.get("summary", {})
    procurement_summary = procurement.get("summary", {})
    if native_summary.get("part_count") != len(native_parts):
        raise ValueError("元件总表summary part_count与parts长度不符")
    if native_summary.get("bom_instance_count") != len(native_instances):
        raise ValueError("元件总表summary bom_instance_count与instances长度不符")
    if procurement_summary.get("bom_rows") != len(purchase_rows):
        raise ValueError("采购补充summary bom_rows与records长度不符")

    quote_after = sha256_file(QUOTE_PATH) if QUOTE_PATH.exists() else None
    if quote_after != quote_before:
        raise RuntimeError("生成期间商城报价.json变化，请重试")
    for path, digest in before.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"生成期间输入快照变化：{path}")
    for bindings, base, label in (
        (native_sources, POWER, "native index source"),
        (procurement_sources, POWER, "procurement source"),
        (manual_sources, PROJECT, "manual supplement source"),
    ):
        for binding in bindings:
            current = sha256_file(source_path(binding["path"], base))
            if current != binding["sha256"]:
                raise RuntimeError(f"生成期间{label}变化：{binding['path']}")

    sources = {
        "native_index": {"path": "索引/元件总表.json", "sha256": before[NATIVE_PATH]},
        "procurement_index": {"path": "索引/采购补充.json", "sha256": before[PROCUREMENT_PATH]},
        "manual_supplement": {"path": "整车器件补充.json", "sha256": before[SUPPLEMENT_PATH]},
        "quote_catalog": {"path": "商城报价.json", "sha256": quote_before},
        "native_source_files": native_sources,
        "procurement_source_files": procurement_sources,
        "manual_evidence_files": manual_sources,
    }
    summary = {
        "native_part_count": len(native_catalog_parts),
        "native_instance_count": len(native_instance_records),
        "procurement_record_count": len(procurement_lines),
        "procurement_native_reference_count": sum(bool(row["native_instance_keys"]) for row in procurement_lines),
        "procurement_bom_only_count": sum(not row["native_instance_keys"] for row in procurement_lines),
        "procurement_identity_unresolved_count": sum(row["identity_status"] in {
            "identity_incomplete_or_conflicting", "manufacturer_missing", "mpn_missing",
            "manufacturer_and_mpn_missing", "ambiguous_exact_identity"
        } for row in procurement_lines),
        "procurement_only_identity_count": len(procurement_only_parts),
        "supplemental_part_identity_count": len(supplemental_parts),
        "external_item_count": len(external_items),
        "requirement_slot_count": len(requirement_slots),
        "quote_observation_count": len(observations),
        "usable_quote_observation_count": sum(is_usable_quote(item) for item in observations),
        "matched_quote_observation_count": sum(item["match_status"].startswith("exact_manufacturer_mpn_") for item in quote_match_audit),
        "unmatched_quote_observation_count": sum(item["match_status"].startswith("unmatched_") for item in quote_match_audit),
        "part_ids_with_verified_quote_count": sum(bool(part["quote_ids"]) for part in all_parts),
        "verified_quote_part_link_count": sum(len(part["quote_ids"]) for part in all_parts),
        "lifecycle_counts_procurement_rows": dict(sorted(status_counts.items())),
        "manufacturer_resolution_counts_procurement_rows": dict(sorted(manufacturer_resolution_counts.items())),
        "quantity_policy": "原生BOM按当前实例计数；采购数量只保留原始quantity_source字符串，不汇总进原生用量。",
        "price_policy": "没有按数量选择阶梯价；当前单价始终null，商城阶梯仅保留为可追溯观测。",
    }
    payload = {
        "schema_version": 1,
        "identity_policy": {
            "native_instance_part_id": "native:<native part key>",
            "exact_merge": "只有manufacturer、MPN及supplier part严格匹配并且报价身份已验证时才关联报价；采购行复用part身份要求manufacturer和MPN明确且精确一致。",
            "no_merge_cases": ["manufacturer missing/generic", "MPN missing", "MPN conflict", "CAD carrier identity", "supplier part mismatch"],
            "cad_separation": "实际采购manufacturer/MPN与carrier、device、symbol、footprint资产完全分栏；载体不参与采购身份或价格匹配。",
        },
        "status_vocabulary": sorted(ALLOWED_STATUSES),
        "sources": sources,
        "summary": summary,
        "native_parts": native_catalog_parts,
        "procurement_only_parts": list(procurement_only_parts.values()),
        "supplemental_parts": list(supplemental_parts.values()),
        "native_instances": native_instance_records,
        "procurement_records": procurement_lines,
        "external_items": external_items,
        "requirement_slots": requirement_slots,
        "project_constraints": project_constraints,
        "quote_match_audit": quote_match_audit,
    }
    stable_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result = {
        **payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": sha256_bytes(stable_payload.encode("utf-8")),
    }
    return result


def payload_for_compare(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "generated_at_utc"}


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只读检查主档是否与当前来源一致")
    args = parser.parse_args()
    try:
        expected = build()
        if args.check:
            if not OUTPUT_PATH.is_file():
                print(json.dumps({"pass": False, "error": "主档文件缺失"}, ensure_ascii=True))
                return 1
            current = read_json(OUTPUT_PATH)
            matches = payload_for_compare(current) == payload_for_compare(expected)
            print(json.dumps({
                "pass": matches,
                "content_sha256": expected.get("content_sha256"),
                "native_parts": expected["summary"]["native_part_count"],
                "native_instances": expected["summary"]["native_instance_count"],
                "procurement_rows": expected["summary"]["procurement_record_count"],
                "quote_observations": expected["summary"]["quote_observation_count"],
                "readonly": True,
            }, ensure_ascii=True))
            return 0 if matches else 1
        write_atomic(OUTPUT_PATH, expected)
        print(json.dumps({
            "generated": True,
            "path": str(O