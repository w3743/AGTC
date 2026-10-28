#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成整车器件主档；只读取原生索引、采购补充、显式补充与商城报价。"""
from __future__ import annotations

import argparse
import csv
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


def procurement_lifecycle(record: dict[str, Any], has_native_reference: bool) -> str:
    """Keep design selection state separate from whether purchasing is complete."""
    status = norm(record.get("status_source"))
    scope = norm(record.get("scope_source"))
    relation = norm(record.get("relation"))
    fallback = "current" if has_native_reference else "unresolved"
    lifecycle = status_from_source(status, fallback)
    if lifecycle in {"obsolete", "DNP", "selected"}:
        return lifecycle

    # A candidate supplier SKU awaiting quote/validation is a procurement state,
    # not a candidate design item. Preserve a candidate lifecycle only when the
    # source explicitly labels the component/assembly itself as candidate.
    quote_only_candidate = any(token in status for token in (
        "候选料号待核价", "candidate mpn", "library_candidate", "前轮报价待刷新"
    ))
    status_candidate = any(token in status for token in (
        "candidate_not_qualified", "candidate_or_placeholder", "候选：", "候选方案"
    )) or status in {"candidate", "候选", "conditional", "条件装配"}
    scope_candidate = any(token in scope for token in (
        "conditional", "conditional assembly", "candidate assembly", "候选装配", "条件装配"
    )) or scope in {"candidate", "候选"}
    if scope_candidate:
        return "candidate"
    if status_candidate and not quote_only_candidate:
        return "candidate"
    if "bom独有条目" in relation:
        # A BOM-only accessory is a current requirement unless its source explicitly
        # marks it candidate/conditional; “待询价/未采购” describes procurement progress.
        return "current"
    return fallback


def quote_supplier_part(observation: dict[str, Any]) -> str:
    return norm(observation.get("supplier_part"))


def selected_quote_is_bound_to_current_bom(observation: dict[str, Any]) -> bool:
    """Validate an explicit current-BOM selection binding against source bytes and row identity."""
    if observation.get("selection_scope") != "selected_current_bom":
        return False
    binding = observation.get("selected_bom_binding")
    if not isinstance(binding, dict):
        return False
    relative_path = binding.get("source_path")
    expected_sha = norm(binding.get("source_sha256"))
    reference = str(binding.get("reference") or "").strip()
    if not relative_path or not expected_sha or not reference:
        return False
    source_path = (PROJECT / str(relative_path)).resolve()
    project_root = PROJECT.resolve()
    if project_root not in source_path.parents or not source_path.is_file():
        return False
    try:
        if sha256_file(source_path) != expected_sha:
            return False
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except (OSError, csv.Error, UnicodeError):
        return False
    matches = [row for row in rows if str(row.get("Designator") or "").strip() == reference]
    if len(matches) != 1:
        return False
    row = matches[0]
    primary_match = (
        norm(row.get("Manufacturer Part")) == norm(observation.get("mpn")) == norm(binding.get("manufacturer_part"))
        and norm(row.get("Manufacturer")) == norm(binding.get("manufacturer_as_listed"))
        and quote_supplier_part(observation) == norm(row.get("Supplier Part")) == norm(binding.get("supplier_part"))
        and str(row.get("Quantity") or "").strip() == str(binding.get("quantity_per_board") or "").strip()
    )
    if not primary_match:
        return False

    # A selected cross-board SKU may be installed in several current BOMs. Keep the
    # legacy primary FOC binding above for backwards compatibility, while validating
    # every additional source with its explicit column map and all covered references.
    extra_bindings = observation.get("selected_bom_bindings", [])
    if not isinstance(extra_bindings, list):
        return False
    for extra in extra_bindings:
        if not isinstance(extra, dict):
            return False
        rel = extra.get("source_path")
        expected_sha = norm(extra.get("source_sha256"))
        ref_field = str(extra.get("reference_field") or "")
        refs = {str(x).strip() for x in extra.get("references", []) if str(x).strip()}
        mpn_field = str(extra.get("manufacturer_part_field") or "")
        maker_field = str(extra.get("manufacturer_field") or "")
        qty_field = str(extra.get("quantity_field") or "")
        code_field = extra.get("supplier_part_field")
        if not rel or not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or not refs or not all((ref_field, mpn_field, maker_field, qty_field)):
            return False
        path = (PROJECT / str(rel)).resolve()
        if PROJECT.resolve() not in path.parents or not path.is_file():
            return False
        try:
            if sha256_file(path) != expected_sha:
                return False
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=str(extra.get("delimiter", ","))))
        except (OSError, csv.Error, UnicodeError):
            return False
        covered: set[str] = set()
        matched_rows: list[dict[str, str]] = []
        quantity_total = 0
        for candidate in rows:
            row_refs = {x.strip() for x in str(candidate.get(ref_field) or "").split(str(extra.get("reference_separator", ","))) if x.strip()}
            overlap = row_refs & refs
            if not overlap:
                continue
            if not row_refs.issubset(refs) or covered & overlap:
                return False
            if norm(candidate.get(mpn_field)) != norm(observation.get("mpn")):
                return False
            if norm(candidate.get(maker_field)) != norm(observation.get("manufacturer")):
                return False
            if code_field and quote_supplier_part(observation) != norm(candidate.get(str(code_field))):
                return False
            try:
                quantity_total += int(float(str(candidate.get(qty_field) or "").strip()))
            except (TypeError, ValueError):
                return False
            covered.update(overlap)
            matched_rows.append(candidate)
        if covered != refs or not matched_rows or quantity_total != int(extra.get("quantity_total", -1)):
            return False
    return True


def is_usable_quote(observation: dict[str, Any]) -> bool:
    url = str(observation.get("source_url") or "")
    acquisition = observation.get("acquisition")
    selection_scope = norm(observation.get("selection_scope"))
    if selection_scope in {"candidate_not_selected", "unselected", "not_selected"}:
        return False
    if selection_scope == "selected_current_bom" and not selected_quote_is_bound_to_current_bom(observation):
        return False
    capture_verified = acquisition == "browser_visible_DOM" or (
        acquisition == "compact_batch_import"
        and observation.get("evidence_kind") == "compact_browser_transcription"
        and str(observation.get("raw_evidence") or "").startswith("compact_browser_transcription;")
    ) or (
        acquisition == "user_supplied_screenshot"
        and observation.get("evidence_kind") in {"candidate_page_screenshot", "product_page_screenshot"}
        and bool(str(observation.get("raw_evidence") or "").strip())
    )
    return (
        observation.get("identity_verified") is True
        and observation.get("supplier") == "LCSC_CN"
        and observation.get("currency") == "CNY"
        and capture_verified
        and bool(re.fullmatch(r"https?://item\.szlcsc\.com/\d+\.html", url))
        and bool(norm(observation.get("mpn")))
        and bool(canonical_maker(observation.get("manufacturer"), {}))
        and bool(quote_supplier_part(observation))
        and isinstance(observation.get("price_tiers"), list)
        and bool(observation.get("price_tiers"))
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
        lifecycle = procurement_lifecycle(record, bool(linked))
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
            "selection_status": lifecycle,
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
            "procurement_status_source": record.get("status_source"),
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
        unresolved_quote_rows = quote_catalog.get("unresolved", [])
        if not isinstance(unresolved_quote_rows, list):
            raise ValueError("商城报价未闭合记录必须为数组")
    else:
        quote_catalog = {"schema_version": 1, "observations": []}
        observations = []
        unresolved_quote_rows = []

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
                "reason": "requires verified LCSC CNY item URL, visible-DOM, compact transcription, or trusted user-screenshot evidence, exact identity fields, and valid price tiers; explicitly unselected candidates are excluded and current-BOM selections require an exact source-hash/row binding",
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
        candidate_status = supercap.get("candidate_status")
        if not all(isinstance(value, int) for value in (current_count, candidate_added, max_count)):
            raise ValueError("稳压超容当前数量、增量和总上限必须使用整数")
        if current_count > max_count:
            raise ValueError("稳压超容现行数量超过有效总量上限")
        if candidate_status == "cancelled_obsolete":
            if candidate_total is not None or candidate_added != 0 or supercap.get("candidate_is_current") is not False:
                raise ValueError("已作废的稳压扩容不得作为有效候选或计入增量")
            if not isinstance(supercap.get("historical_cancelled_candidate_total_count"), int):
                raise ValueError("已作废扩容需单独标明历史候选总量")
        else:
            if not isinstance(candidate_total, int):
                raise ValueError("有效稳压候选总量必须使用整数")
            if candidate_total - current_count != candidate_added or candidate_total > max_count:
                raise ValueError("稳压超容候选总量/增量/总上限关系不一致")
        if supercap.get("use_case") == "startup_transient_power_assist":
            if supercap.get("sustained_1s_requirement") is not False or supercap.get("duration_s") is not None:
                raise ValueError("启动瞬态辅助不得保留固定持续1秒要求")

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
        "unresolved_quote_input_count": len(unresolved_quote_rows),
        "matched_quote_observation_count": sum(item["match_status"].startswith("exact_manufacturer_mpn_") for item in quote_match_audit),
        "unmatched_quote_observation_count": sum(item["match_status"].startswith("unmatched_") for item in quote_match_audit),
        "part_ids_with_verified_quote_count": sum(bool(part["quote_ids"]) for part in all_parts),
        "verified_quote_part_link_count": sum(len(part["quote_ids"]) for part in all_parts),
        "lifecycle_counts_procurement_rows": dict(sorted(status_counts.items())),
        "manufacturer_resolution_counts_procurement_rows": dict(sorted(manufacturer_resolution_counts.items())),
        "quantity_policy": "原生BOM按当前实例计数；采购数量只保留原始quantity_source字符串，不汇总进原生用量。",
        "price_policy": "没有按数量选择阶梯价；当前单价始终null，商城阶梯仅保留为可追溯观测。",
    }
    unresolved_quote_records = [
        {
            "unresolved_id": item.get("unresolved_id"),
            "quote_id": item.get("quote_id"),
            "manufacturer": item.get("manufacturer"),
            "manufacturer_part": item.get("mpn"),
            "supplier_part": item.get("supplier_part"),
            "source_url": item.get("source_url"),
            "reason": item.get("reason"),
        }
        for item in unresolved_quote_rows
    ]
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
        "manual_source_catalog": {
            row["source_id"]: {
                "path": row["path"].replace("\\", "/"),
                "locator": row.get("locator"),
                "sha256": next((item["sha256"] for item in manual_sources if item["path"] == row["path"].replace("\\", "/")), None),
            }
            for row in supplement.get("sources", [])
        },
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
        "unresolved_quote_records": unresolved_quote_records,
    }
    stable_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result = {
        **payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_sha256": sha256_bytes(stable_payload.encode("utf-8")),
    }
    return result


def md_cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if item not in (None, "")) or "—"
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("|", "\\|").replace("\n", "<br>")


def manufacturer_display(part: dict[str, Any]) -> str:
    raw = part.get("manufacturer")
    values = raw if isinstance(raw, list) else [raw]
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        key = norm(value)
        if key not in seen:
            seen.add(key)
            result.append(str(value))
    return " / ".join(result) or "未确认"


def carrier_difference(instances: list[dict[str, Any]]) -> str:
    mismatches: list[str] = []
    comparable = 0
    for instance in instances:
        identity = instance.get("identity") or {}
        cad = instance.get("cad_assets") or {}
        ref = str(instance.get("reference") or instance.get("instance_key"))
        actual_mpn = identity.get("manufacturer_part")
        carrier_mpn = cad.get("carrier_mpn")
        if actual_mpn and carrier_mpn:
            comparable += 1
            if norm(actual_mpn) != norm(carrier_mpn):
                mismatches.append(f"{ref}:MPN {carrier_mpn}≠{actual_mpn}")
        actual_value = identity.get("actual_value")
        carrier_value = cad.get("carrier_value") or cad.get("effective_value_raw")
        if actual_value and carrier_value:
            comparable += 1
            if norm(actual_value) != norm(carrier_value):
                mismatches.append(f"{ref}:Value {carrier_value}≠{actual_value}")
    if mismatches:
        head = mismatches[:4]
        suffix = f"等{len(mismatches)}处" if len(mismatches) > len(head) else ""
        return f"{'; '.join(head)}{suffix}"
    if comparable:
        return "可比MPN/Value字段未见差异"
    return "载体字段不足，未作差异判断"


def render_markdown(data: dict[str, Any]) -> str:
    summary = data["summary"]
    constraints = data.get("project_constraints", {})
    supercap = constraints.get("regulator_supercapacitor", {})
    bms_budget = constraints.get("bms_cost_target", {})
    post_schematic = constraints.get("post_schematic_human_scope", {})
    status_labels = {
        "selected": "已选型", "current": "现行", "candidate": "候选",
        "DNP": "不装", "obsolete": "作废", "unresolved": "未闭合",
    }
    status_summary = "、".join(
        f"{status_labels.get(key, key)} {value} 行"
        for key, value in sorted(summary["lifecycle_counts_procurement_rows"].items())
    ) or "无采购行"
    lines = [
        "# 项目器件主档（整车范围）",
        "",
        "本主档覆盖整车现行原生电路、采购补充及已记录的板外部件和未冻结需求。它是统一身份与来源索引，不把采购行重复加到原生实例，也不把商城阶梯价自动转成当前单价或整车总价。",
        "",
        "## 快照统计",
        "",
        f"- 原生器件：{summary['native_part_count']} 个身份、{summary['native_instance_count']} 个位号实例。",
        f"- 采购补充：{summary['procurement_record_count']} 行；{summary['procurement_native_reference_count']} 行关联原生实例，{summary['procurement_bom_only_count']} 行为板外/BOM独有条目；采购数量保留源字符串，不再乘算进原生用量。",
        f"- 整车补充：{summary['external_item_count']} 项；另有 {summary['requirement_slot_count']} 个未冻结需求槽位。",
        f"- 商城价格输入：{summary['quote_observation_count']} 条观测，{summary['usable_quote_observation_count']} 条符合当前精确身份与商品页证据规则；{summary['unresolved_quote_input_count']} 条仍未闭合。价格阶梯是观测，按本主档政策当前单价保持 null。",
        f"- 采购行选型状态：{status_summary}。采购进度原文单独保留，不把待询价或未采购自动解释成候选。",
        "",
        "## 全车约束与状态口径",
        "",
    ]
    if supercap:
        lines.append(
            f"- 稳压超容：固定 {md_cell(supercap.get('manufacturer'))} {md_cell(supercap.get('fixed_manufacturer_part'))}；当前 {md_cell(supercap.get('native_current_count'))} 颗，有效总量上限 {md_cell(supercap.get('total_count_max'))} 颗，活动增量 {md_cell(supercap.get('candidate_added_count'))}。16 颗扩容仅保留为已取消/作废历史方案，不是有效候选。用途为启动瞬态功率辅助；持续 1 秒要求为否，持续时间待实际启动波形和功率缺口核定。800–1000 W 仅为未验证参考值。旧稳压 500 元上限已取消。"
        )
    if bms_budget:
        lines.append(
            f"- BMS 板内成本目标：{md_cell(bms_budget.get('target_cny'))} 元，仅板上器件；外部 MCU 与板外配套不在该范围内，状态为 {md_cell(bms_budget.get('status'))}。"
        )
    if post_schematic:
        scope_items = "、".join(post_schematic.get("items") or []) or "各项原理图后工作"
        lines.append(
            f"- 原理图后的工作由人类负责：{scope_items}。本主档和原理图审查不代表这些后续工作已完成。"
        )
    lines.extend([
        "- 全项目仍需优先复用满足要求的器件、减少型号并控制采购总成本；缺少精确报价或订单数量时不填价格、不声称降本。",
        "- `selected/current/candidate/DNP/obsolete/unresolved` 表示设计选型或装配生命周期；采购进度保留在对应 `procurement_status_source` 原文中，两者分开。",
        "- 每个原生 instance 有唯一 `part_id`。下表的原生用量按各源板单板列出，不乘以整车板数；实际采购制造商/MPN、supplier part 与 CAD 载体/符号/封装资产分列，载体不用于推断采购身份。",
        "",
        "## 整车板外器件与未冻结槽位",
        "",
        "需求分配只说明消费者归属，包含在该物件的整车总量内，不重复计数。未知数量、制造商或准确 SKU 保留空值。",
        "",
        "| 类别 | 项目 | 制造商 / MPN | 整车数量 | 生命周期 | 需求分配 / 当前价格状态 | 来源 |",
        "|---|---|---|---:|---|---|---|",
    ])
    for item in data["external_items"]:
        maker_mpn = " / ".join(str(v) for v in (item.get("manufacturer"), item.get("manufacturer_part")) if v) or "未冻结"
        quantity = "待核" if item.get("quantity_per_vehicle") is None else f"{item['quantity_per_vehicle']} {item.get('quantity_unit') or ''}".strip()
        allocations = item.get("demand_allocations") or []
        allocation_text = "; ".join(
            f"{a.get('consumer_instance')} ×{a.get('quantity')}" for a in allocations
        )
        if allocations:
            allocation_text += "（分配量包含在整车总量内）"
        if item.get("current_unit_price") is None:
            allocation_text = (allocation_text + "；" if allocation_text else "") + "当前单价未确定"
        sources_text = ", ".join(item.get("source_ids") or [])
        lines.append(
            "| 整车部件 | " + " | ".join(md_cell(value) for value in (
                item.get("product_name"), maker_mpn, quantity,
                status_labels.get(item.get("status"), item.get("status")), allocation_text, sources_text,
            )) + " |"
        )
    for slot in data["requirement_slots"]:
        maker_mpn = " / ".join(str(v) for v in (slot.get("manufacturer"), slot.get("manufacturer_part")) if v) or "未冻结"
        slot_status = f"未冻结：{slot.get('reason') or ''}"
        lines.append(
            "| 需求槽位 | " + " | ".join(md_cell(value) for value in (
                slot.get("name"), maker_mpn, "待定", "未闭合", slot_status,
                ", ".join(slot.get("source_ids") or []),
            )) + " |"
        )

    lines.extend([
        "",
        "## 原生器件目录（单板用量）",
        "",
        "模块用量只统计原生 instance；`DNP` 列显示 BOM 标记为不装的位号。载体差异根据实例已有的载体 MPN/Value 与实际 MPN/Value 比较，缺失字段不会被当成一致。精确商城匹配提供商品页链接；未匹配时不显示价格。",
        "",
        "| part_id | 实际制造商 | 实际 MPN | 每板现行用量 | DNP位号数 | 实例状态 | CAD载体差异 | C号（源） | 商城精确关联 |",
        "|---|---|---|---|---:|---|---|---|---|",
    ])
    instances_by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in data["native_instances"]:
        instances_by_part[instance["part_id"]].append(instance)
    procurement_by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in data["procurement_records"]:
        if record.get("part_id"):
            procurement_by_part[record["part_id"]].append(record)

    for part in data["native_parts"]:
        part_instances = instances_by_part.get(part["part_id"], [])
        module_current: Counter[str] = Counter()
        dnp_count = 0
        unresolved_count = 0
        for instance in part_instances:
            lifecycle = instance.get("lifecycle_status")
            if lifecycle == "current":
                module_current[str(instance.get("module") or "未标模块")] += 1
            elif lifecycle == "DNP":
                dnp_count += 1
            else:
                unresolved_count += 1
        module_text = "; ".join(f"{module} ×{count}" for module, count in sorted(module_current.items())) or "无现行实例"
        lifecycle_text = "现行" if module_current else "不装" if dnp_count else "未闭合"
        if unresolved_count:
            lifecycle_text += f"；未闭合 {unresolved_count}"
        purchase_rows = procurement_by_part.get(part["part_id"], [])
        selection_states = sorted({row.get("selection_status") for row in purchase_rows if row.get("selection_status")})
        if selection_states:
            lifecycle_text += "；采购选型 " + ", ".join(status_labels.get(value, str(value)) for value in selection_states)
        quote_links = []
        for observation in part.get("quote_observations", []):
            qid = observation.get("quote_id")
            url = observation.get("source_url")
            if qid and url:
                quote_links.append(f"[{qid}]({url})")
        quote_text = ", ".join(sorted(set(quote_links))) if quote_links else (
            "身份未闭合" if part.get("identity_status") != "manufacturer_and_mpn_present" else "待精确商品页报价"
        )
        supplier_parts = part.get("supplier_parts_source") or part.get("supplier_parts") or []
        lines.append(
            "| " + " | ".join(md_cell(value) for value in (
                part.get("part_id"), manufacturer_display(part), part.get("manufacturer_part"),
                module_text, dnp_count, lifecycle_text, carrier_difference(part_instances),
                ", ".join(str(value) for value in supplier_parts), quote_text,
            )) + " |"
        )

    lines.extend([
        "",
        "## 商城输入未闭合项",
        "",
        "以下是主档输入中的未闭合报价候选；没有阶梯价或身份不匹配时保持 null，不映射相似型号或不同厂家。",
        "",
        "| 标识 | 制造商 / MPN | C号 | 商品页 | 未闭合原因 |",
        "|---|---|---|---|---|",
    ])
    for item in data.get("unresolved_quote_records", []):
        maker_mpn = " / ".join(str(v) for v in (item.get("manufacturer"), item.get("manufacturer_part")) if v) or "未确认"
        page = f"[官方页]({item['source_url']})" if item.get("source_url") else "未取得精确商品页"
        lines.append(
            "| " + " | ".join(md_cell(value) for value in (
                item.get("quote_id") or item.get("unresolved_id"), maker_mpn,
                item.get("supplier_part"), page, item.get("reason"),
            )) + " |"
        )

    source_rows: list[tuple[str, str, str]] = []
    for key in ("native_index", "procurement_index", "manual_supplement", "quote_catalog"):
        value = data["sources"].get(key) or {}
        source_rows.append((key, str(value.get("path") or ""), str(value.get("sha256") or "")))
    for key in ("native_source_files", "procurement_source_files"):
        for value in data["sources"].get(key, []):
            source_rows.append((key, value["path"], value["sha256"]))
    for source_id, value in sorted(data.get("manual_source_catalog", {}).items()):
        source_rows.append((source_id, value["path"], str(value.get("sha256") or "")))
    lines.extend([
        "",
        "## 输入来源与SHA-256",
        "",
        "| 来源 | 相对路径 | SHA-256 |",
        "|---|---|---|",
    ])
    for label, path, digest in source_rows:
        lines.append(f"| {md_cell(label)} | `{md_cell(path)}` | `{md_cell(digest)}` |")
    lines.append("")
    return "\n".join(lines)


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


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp = Path(stream.name)
            stream.write(value)
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
        expected_md = render_markdown(expected)
        expected_md_sha256 = sha256_bytes(expected_md.encode("utf-8"))
        if args.check:
            if not OUTPUT_PATH.is_file() or not OUTPUT_MD_PATH.is_file():
                print(json.dumps({"pass": False, "error": "JSON或Markdown主档缺失"}, ensure_ascii=True))
                return 1
            current = read_json(OUTPUT_PATH)
            json_matches = payload_for_compare(current) == payload_for_compare(expected)
            md_matches = OUTPUT_MD_PATH.read_text(encoding="utf-8-sig") == expected_md
            matches = json_matches and md_matches
            print(json.dumps({
                "pass": matches,
                "json_pass": json_matches,
                "markdown_pass": md_matches,
                "content_sha256": expected.get("content_sha256"),
                "markdown_sha256": expected_md_sha256,
                "native_parts": expected["summary"]["native_part_count"],
                "native_instances": expected["summary"]["native_instance_count"],
                "procurement_rows": expected["summary"]["procurement_record_count"],
                "quote_observations": expected["summary"]["quote_observation_count"],
                "usable_quote_observations": expected["summary"]["usable_quote_observation_count"],
                "readonly": True,
            }, ensure_ascii=True))
            return 0 if matches else 1
        write_atomic(OUTPUT_PATH, expected)
        write_text_atomic(OUTPUT_MD_PATH, expected_md)
        print(json.dumps({
            "generated": True,
            "path": str(OUTPUT_PATH),
            "markdown_path": str(OUTPUT_MD_PATH),
            "content_sha256": expected.get("content_sha256"),
            "markdown_sha256": expected_md_sha256,
            **expected["summary"],
        }, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
