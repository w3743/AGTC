#!/usr/bin/env python3
"""Build deterministic purchase-cost scenarios from current module BOMs and verified SZLCSC quotes.

Default: write 索引/商城成本.json and 索引/当前商城成本.md.
--check: read-only byte comparison against those generated files (exit 0 when current).

This tool never reads old BOM prices as quotes, never treats a CAD carrier as a purchase item,
and only joins a price when the verified quote's exact manufacturer part, manufacturer, and
LCSC code match the current procurement identity. MOQ is taken only from an explicitly observed
minimum-order field; it is never inferred from the first price tier or a reel/package quantity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LIB = ROOT / "电力" / "元件封装库"
PRICE_TABLE = LIB / "商城报价.json"
CATALOG = LIB / "索引" / "元件总表.json"
PURCHASE_SUPPLEMENT = LIB / "索引" / "采购补充.json"
PRICE_INPUT_DIR = LIB / "检查" / "商城核价"
OUTPUT_JSON = LIB / "索引" / "商城成本.json"
OUTPUT_MD = LIB / "索引" / "当前商城成本.md"

MODULES = {
    "BMS": {
        "label": "BMS",
        "input": PRICE_INPUT_DIR / "BMS.json",
        "bom": ROOT / "电力" / "BMS模块" / "原理图准备" / "BOM.csv",
        "kind": "bms",
    },
    "FOC": {
        "label": "FOC",
        "input": PRICE_INPUT_DIR / "FOC.json",
        "bom": ROOT / "电力" / "FOC驱动模块" / "eda工程文件" / "导出" / "当前交付" / "FOC_G431_BOM.csv",
        "supplement": ROOT / "电力" / "FOC驱动模块" / "eda工程文件" / "导出" / "当前交付" / "FOC_G431_双板装配补充.csv",
        "kind": "foc",
    },
    "REGULATOR": {
        "label": "稳压模块",
        "input": PRICE_INPUT_DIR / "稳压.json",
        "bom": ROOT / "电力" / "稳压模块" / "原理图准备" / "BOM.csv",
        "kind": "csv",
    },
    "PUMP": {
        "label": "水泵驱动模块",
        "input": PRICE_INPUT_DIR / "水泵.json",
        "bom": ROOT / "电力" / "水泵驱动模块" / "原理图准备" / "BOM.csv",
        "kind": "csv",
    },
}

SCENARIOS = {
    "four_modules_one_set": {
        "label": "四模块各一套：BMS×1 + FOC-A×1 + 稳压×1 + 水泵×1",
        "modules": {"BMS": 1, "FOC-A": 1, "REGULATOR": 1, "PUMP": 1},
    },
    "actual_vehicle": {
        "label": "整车电力模块组合：BMS×1 + FOC-A×1 + FOC-B×1 + 稳压×1 + 水泵×1（不含电芯、2台电机等整车其他物料）",
        "modules": {"BMS": 1, "FOC-A": 1, "FOC-B": 1, "REGULATOR": 1, "PUMP": 1},
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def as_int(value: Any, default: int = 0) -> int:
    try:
        d = Decimal(str(value).strip())
        if d != d.to_integral_value():
            raise ValueError(f"non-integral piece quantity: {value}")
        return int(d)
    except (InvalidOperation, ValueError, TypeError):
        if default is not None:
            return default
        raise


def clean(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.casefold() in {"none", "null", "unverified", "n/a", "-"}:
        return None
    return s


def norm_mpn(value: Any) -> str:
    return re.sub(r"\s+", "", (clean(value) or "").casefold())


def norm_code(value: Any) -> str | None:
    s = clean(value)
    if not s:
        return None
    s = s.upper().strip()
    if s in {"UNVERIFIED", "UNKNOWN", "TBD"}:
        return None
    return s


def norm_manufacturer(value: Any) -> str | None:
    s = clean(value)
    if not s:
        return None
    # Parenthetical text here is routinely a Chinese localized brand name.
    s = re.split(r"[（(]", s, maxsplit=1)[0].strip().casefold()
    # Preserve Chinese-only manufacturer names instead of erasing them to an empty key.
    s = re.sub(r"[^a-z0-9\u0080-\uffff]+", "", s)
    aliases = {
        "ti": "texasinstruments",
        "texasinstruments": "texasinstruments",
        "murata": "murata",
        "murata mfg": "murata",
        "muratamanufacturingco": "murata",
        "yageo": "yageo",
        "degson": "degson",
        "infineon": "infineon",
        "kemet": "kemet",
        "tdk": "tdk",
        "jst": "jst",
        "nichicon": "nichicon",
        "semitec": "semitec",
        "eaton": "eaton",
        "littelfuse": "littelfuse",
        "vishay": "vishay",
        "vishaysfernice": "vishaysfernice",
        "weact": "weact",
        "littelfuseixys": "littelfuse",
    }
    return aliases.get(s, s) or None


def split_refs(value: Any) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


FOC_EXTERNAL_REFS = {
    "CBUS1", "CBUS2", "CBUS3", "R_DUMP_A", "R_DUMP_B", "F_MAIN", "F_BRAKE",
    "J_BRAKE_R", "J_LOAD_POWER_VALID配对线端",
}


def scope_bucket_for(module: str, scope: str, refs: list[str]) -> str:
    """Classify each source demand before SKU merging, retaining its exact source scope too."""
    raw = (scope or "unspecified").strip().casefold().replace("-", "_").replace(" ", "_")
    if module == "BMS":
        if raw == "on_board":
            return "on_board"
        if raw == "core_board_module":
            return "core_board_module"
        return "external_accessory"
    if module in {"FOC-A", "FOC-B", "FOC"}:
        return "external_accessory" if set(refs) & FOC_EXTERNAL_REFS or "external" in raw or "harness" in raw else "on_board"
    if raw in {"daughterboard", "core_board_module", "core_board"} or "核心板" in scope:
        return "core_board_module"
    if raw in {"off_board", "off_board_external", "harness", "board_interface_package_pending", "external_accessory"} or "external" in raw or "harness" in raw:
        return "external_accessory"
    return "on_board"


def module_item_meta(input_doc: dict[str, Any], module: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in input_doc.get("items", []):
        refs = item.get("designators") if module == "BMS" else item.get("references")
        if module == "FOC":
            refs = item.get("designators", [])
        for ref in refs or []:
            out[str(ref)].append(item)
    return out


def register_binding(bindings: dict[str, dict[str, Any]], label: str, path: Path, role: str) -> None:
    rel = path.relative_to(ROOT).as_posix()
    bindings[label] = {"path": rel, "sha256": sha256(path), "role": role}


def make_demand(
    module: str,
    scope: str,
    refs: list[str],
    mpn: Any,
    manufacturer: Any,
    supplier_part: Any,
    quantity: int,
    source_path: Path,
    source_row: int | None,
    identity_notes: list[str] | None = None,
    population: str | None = None,
    source_supplier_part: Any = ...,
    supplier_part_resolution_source: str | None = None,
    scope_bucket: str | None = None,
) -> dict[str, Any]:
    m = clean(mpn)
    maker = clean(manufacturer)
    code = norm_code(supplier_part)
    source_code = code if source_supplier_part is ... else norm_code(source_supplier_part)
    issues = list(identity_notes or [])
    if not m:
        issues.append("missing_exact_mpn")
    if not maker:
        issues.append("manufacturer_unresolved")
    if not code:
        issues.append("no_verified_lcsc_code")
    return {
        "module": module,
        "scope": scope or "unspecified",
        "references": sorted(set(refs)),
        "manufacturer_part": m,
        "manufacturer": maker,
        "supplier": "LCSC_CN" if code else None,
        "supplier_part": code,
        "source_supplier_part": source_code,
        "supplier_part_resolution_source": supplier_part_resolution_source or ("source_record" if code else None),
        "supplier_code_verified_from_quote": False,
        "quantity": quantity,
        "population": population,
        "scope_bucket": scope_bucket or scope_bucket_for(module, scope, refs),
        "references_by_scope_bucket": {scope_bucket or scope_bucket_for(module, scope, refs): sorted(set(refs))},
        "source": source_path.relative_to(ROOT).as_posix(),
        "source_row": source_row,
        "identity_notes": sorted(set(issues)),
    }


def demand_key(d: dict[str, Any]) -> tuple[str, str, str, str]:
    # Blank manufacturer/code are deliberately distinct from verified identities.
    fallback = "|".join(d["references"]) if not d.get("manufacturer_part") else ""
    return (
        norm_manufacturer(d.get("manufacturer")) or "!manufacturer-unknown",
        norm_mpn(d.get("manufacturer_part")) or "!mpn-unknown:" + fallback,
        norm_code(d.get("supplier_part")) or "!supplier-part-unknown",
        str(d.get("supplier") or ""),
    )


def merge_demands(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for d in records:
        key = demand_key(d)
        if key not in merged:
            x = dict(d)
            x["references"] = list(d.get("references", []))
            x["identity_notes"] = list(d.get("identity_notes", []))
            x["sources"] = [{"path": d["source"], "row": d.get("source_row")}]
            x["scope_set"] = {d.get("scope", "unspecified")}
            x["population_set"] = {d.get("population")} if d.get("population") else set()
            x["quantity_by_scope"] = {d.get("scope", "unspecified"): d["quantity"]}
            x["quantity_by_scope_bucket"] = {d.get("scope_bucket", "unspecified"): d["quantity"]}
            x["references_by_scope_bucket"] = {k: set(v) for k, v in d.get("references_by_scope_bucket", {}).items()}
            x["source_supplier_parts"] = [d.get("source_supplier_part")]
            x["supplier_part_resolution_sources"] = [d.get("supplier_part_resolution_source")] if d.get("supplier_part_resolution_source") else []
            x["supplier_code_verified_from_quote"] = bool(d.get("supplier_code_verified_from_quote"))
            merged[key] = x
        else:
            x = merged[key]
            x["quantity"] += d["quantity"]
            x["references"] = sorted(set(x["references"] + d.get("references", [])))
            x["identity_notes"] = sorted(set(x["identity_notes"] + d.get("identity_notes", [])))
            x["sources"].append({"path": d["source"], "row": d.get("source_row")})
            x["scope_set"].add(d.get("scope", "unspecified"))
            if d.get("population"):
                x["population_set"].add(d["population"])
            raw_scope = d.get("scope", "unspecified")
            x["quantity_by_scope"][raw_scope] = x["quantity_by_scope"].get(raw_scope, 0) + d["quantity"]
            bucket = d.get("scope_bucket", "unspecified")
            x["quantity_by_scope_bucket"][bucket] = x["quantity_by_scope_bucket"].get(bucket, 0) + d["quantity"]
            for scope_bucket, refs in d.get("references_by_scope_bucket", {}).items():
                x["references_by_scope_bucket"].setdefault(scope_bucket, set()).update(refs)
            x["source_supplier_parts"].append(d.get("source_supplier_part"))
            if d.get("supplier_part_resolution_source"):
                x["supplier_part_resolution_sources"].append(d["supplier_part_resolution_source"])
            x["supplier_code_verified_from_quote"] = x["supplier_code_verified_from_quote"] or bool(d.get("supplier_code_verified_from_quote"))
    result = []
    for _, x in sorted(merged.items(), key=lambda kv: kv[0]):
        x["scopes"] = sorted(x.pop("scope_set"))
        x["populations"] = sorted(x.pop("population_set"))
        x["supplier_part_resolution_sources"] = sorted(set(x["supplier_part_resolution_sources"]))
        x["source_supplier_parts"] = sorted(set(x["source_supplier_parts"]), key=lambda v: v or "")
        x["references_by_scope_bucket"] = {k: sorted(v) for k, v in sorted(x["references_by_scope_bucket"].items())}
        x.pop("scope", None)
        x.pop("population", None)
        x["sources"] = sorted(x["sources"], key=lambda z: (z["path"], z.get("row") or 0))
        result.append(x)
    return result


def catalog_indexes(catalog: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[str, dict[str, Any]] = {}
    parts_by_mpn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in catalog.get("instances", []):
        by_key[str(x.get("key"))] = x
    for x in catalog.get("parts", []):
        parts_by_mpn[norm_mpn(x.get("mpn"))].append(x)
    return by_key, parts_by_mpn


def bms_demands(
    input_doc: dict[str, Any],
    bom_rows: list[dict[str, str]],
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], int]:
    issues: list[str] = []
    by_ref, parts_by_mpn = catalog_indexes(catalog)
    bom_by_ref = {clean(r.get("位号")): r for r in bom_rows if clean(r.get("位号"))}
    input_items = input_doc.get("items", [])
    item_by_ref: dict[str, dict[str, Any]] = {}
    for item in input_items:
        for ref in item.get("designators", []):
            if ref in item_by_ref:
                issues.append(f"BMS核价输入重复映射位号:{ref}")
            item_by_ref[ref] = item

    records: list[dict[str, Any]] = []
    for item in input_items:
        scope = str(item.get("scope") or "unspecified")
        refs = list(item.get("designators", []))
        q = as_int(item.get("qty_per_bms_set", 0))
        if scope == "on_board":
            resolved = []
            for ref in refs:
                row = bom_by_ref.get(ref)
                ci = by_ref.get(f"BMS模块::{ref}")
                local_issues: list[str] = []
                if row is None:
                    local_issues.append("designator_missing_from_current_BOM")
                if ci is None:
                    local_issues.append("designator_missing_from_current_catalog")
                bom_mpn = clean(row.get("MPN")) if row else None
                cat_mpn = clean(ci.get("mpn")) if ci else None
                item_mpn = clean(item.get("mpn"))
                known_variants = {norm_mpn(item_mpn), *[norm_mpn(x) for x in item.get("source_mpn_variants", [])]}
                if bom_mpn and item_mpn and norm_mpn(bom_mpn) not in known_variants:
                    local_issues.append("audit_input_mpn_differs_from_current_BOM")
                if bom_mpn and cat_mpn and norm_mpn(bom_mpn) != norm_mpn(cat_mpn) and norm_mpn(cat_mpn) not in known_variants:
                    local_issues.append("central_catalog_mpn_differs_from_current_BOM")
                maker = clean(ci.get("manufacturer")) if ci else None
                code = norm_code(ci.get("supplier_part")) if ci else None
                if item.get("lcsc_codes") and code and code not in {norm_code(x) for x in item.get("lcsc_codes", [])}:
                    local_issues.append("catalog_supplier_part_not_in_price_input_crosscheck")
                resolved.append({
                    "ref": ref,
                    # The audit input's normalized exact MPN resolves harmless manufacturer
                    # prefixes recorded in source BOM/native attributes (e.g. "TI SN..." vs "SN...").
                    "mpn": item_mpn if item_mpn and ((bom_mpn and norm_mpn(bom_mpn) in known_variants) or (cat_mpn and norm_mpn(cat_mpn) in known_variants)) else (bom_mpn or cat_mpn or item_mpn),
                    "manufacturer": maker,
                    "supplier_part": code,
                    "issues": local_issues,
                    "row": row,
                })
            signatures = {(norm_manufacturer(x["manufacturer"]), norm_mpn(x["mpn"]), norm_code(x["supplier_part"])) for x in resolved}
            if len(signatures) <= 1:
                entry = resolved[0] if resolved else {"mpn": item.get("mpn"), "manufacturer": None, "supplier_part": None, "issues": ["no_designators"]}
                all_issues = [p for x in resolved for p in x["issues"]]
                if q != len(refs):
                    all_issues.append(f"quantity_{q}_differs_from_designator_count_{len(refs)}")
                records.append(make_demand(
                    "BMS", scope, refs, entry["mpn"], entry["manufacturer"], entry["supplier_part"], q,
                    MODULES["BMS"]["bom"], None, all_issues, "included_current_BMS_input",
                    source_supplier_part=None,
                    supplier_part_resolution_source="current_central_index" if entry["supplier_part"] else None,
                ))
            elif q == len(resolved):
                for x in resolved:
                    records.append(make_demand(
                        "BMS", scope, [x["ref"]], x["mpn"], x["manufacturer"], x["supplier_part"], 1,
                        MODULES["BMS"]["bom"], None, x["issues"], "included_current_BMS_input",
                        source_supplier_part=None,
                        supplier_part_resolution_source="current_central_index" if x["supplier_part"] else None,
                    ))
            else:
                records.append(make_demand(
                    "BMS", scope, refs, item.get("mpn"), None, None, q,
                    MODULES["BMS"]["bom"], None,
                    ["central_catalog identities differ within an input group and quantity cannot be split safely"],
                    "included_current_BMS_input",
                    source_supplier_part=None,
                ))
        else:
            # The audit input joins the current BOM and the purchase supplement; keep every current
            # off-board accessory and the explicitly required core-board module as purchase demand.
            maker_candidates = [clean(x.get("name")) for x in item.get("manufacturer_candidates", []) if clean(x.get("name"))]
            makers = {norm_manufacturer(x): x for x in maker_candidates}
            maker = next(iter(makers.values())) if len(makers) == 1 else None
            code_list = sorted({norm_code(x) for x in item.get("lcsc_codes", []) if norm_code(x)})
            mpn = clean(item.get("mpn"))
            # Exact catalog part identity may disambiguate alternate codes for external items.
            matching_parts = [x for x in parts_by_mpn.get(norm_mpn(mpn), []) if clean(x.get("mpn")) and norm_mpn(x.get("mpn")) == norm_mpn(mpn)]
            matching_codes = {norm_code(c) for p in matching_parts for c in p.get("supplier_parts", []) if norm_code(c)}
            if matching_codes:
                exact_codes = sorted(set(code_list) & matching_codes)
                if len(exact_codes) == 1:
                    code_list = exact_codes
                elif not code_list and len(matching_codes) == 1:
                    code_list = sorted(matching_codes)
            code = code_list[0] if len(code_list) == 1 else None
            local_issues = []
            if len(makers) > 1:
                local_issues.append("ambiguous_manufacturer_candidates")
            if len(code_list) > 1:
                local_issues.append("ambiguous_supplier_part_candidates")
            if not refs or any(ref not in bom_by_ref for ref in refs):
                local_issues.append("reference_not_found_in_current_BMS_BOM")
            records.append(make_demand(
                "BMS", scope, refs, mpn, maker, code, q, MODULES["BMS"]["bom"], None,
                local_issues, "included_current_BMS_input" if scope == "core_board_module" else "current_external_accessory",
                source_supplier_part=code,
                supplier_part_resolution_source="BMS_price_audit_input" if code else None,
            ))

    total_from_items = sum(as_int(x.get("qty_per_bms_set", 0)) for x in input_items)
    total_from_bom = sum(as_int(r.get("每套用量", 0)) for r in bom_rows)
    if total_from_items != total_from_bom:
        issues.append(f"BMS输入数量{total_from_items}与当前BOM数量{total_from_bom}不一致")
    if len(bom_rows) != input_doc.get("summary", {}).get("bom_designator_rows"):
        issues.append("BMS输入声明的BOM行数与当前BOM不一致")
    return records, issues, total_from_items


def ref_meta_map(input_doc: dict[str, Any], module: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = list(input_doc.get("items", [])) + list(input_doc.get("priority_no_lcsc_code_items", []))
    for item in candidates:
        refs = item.get("designators", []) if module == "FOC" else item.get("references", [])
        for ref in refs:
            out[str(ref)].append(item)
    return out


def verified_code_from_meta(meta: dict[str, Any], module: str) -> str | None:
    if module == "FOC":
        # supplierPart is copied from the current official BOM; candidates are not treated as proof.
        return norm_code(meta.get("supplierPart"))
    code = norm_code(meta.get("lcsc_part"))
    status = str(meta.get("item_identity_status") or "")
    if not code and "official_page_found_supplier_code_absent" in status:
        code = norm_code(meta.get("official_page_supplier_part"))
    if code and "official_szlcsc_page_matches_mpn_and_part_number" in status:
        return code
    if code and "official_page_found_supplier_code_absent" in status:
        return code
    return None


def generic_module_demands(
    module_key: str,
    input_doc: dict[str, Any],
    bom_rows: list[dict[str, str]],
    catalog: dict[str, Any],
    candidate_exclusions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_ref, parts_by_mpn = catalog_indexes(catalog)
    meta_by_ref = ref_meta_map(input_doc, "FOC" if module_key == "FOC" else module_key)
    records: list[dict[str, Any]] = []
    for row_i, row in enumerate(bom_rows, start=2):
        if module_key == "FOC":
            refs = split_refs(row.get("Designator"))
            mpn = clean(row.get("Manufacturer Part"))
            maker = clean(row.get("Manufacturer"))
            bom_code = norm_code(row.get("Supplier Part"))
            scope = "FOC_BOM"
            quantity = as_int(row.get("Quantity", 0))
        else:
            refs = split_refs(row.get("Reference"))
            mpn = clean(row.get("Manufacturer part"))
            maker = clean(row.get("Manufacturer"))
            bom_code = norm_code(row.get("Supplier part"))
            scope = clean(row.get("Scope")) or "unspecified"
            quantity = as_int(row.get("Quantity", 0))
            status = str(row.get("Procurement status") or "")
            # Explicit non-current candidates/placeholders are listed separately, never priced as current.
            is_candidate = (
                "CONDITIONAL" in scope.upper()
                or status.strip().upper() == "CANDIDATE_NOT_QUALIFIED"
                or status.strip().lower() == "candidate_or_placeholder"
                or "DNP" in status.upper()
                or scope.upper() == "DNP"
            )
            if is_candidate:
                candidate_exclusions.append({
                    "module": MODULES[module_key]["label"],
                    "references": refs,
                    "manufacturer_part": mpn,
                    "manufacturer": maker,
                    "supplier_part": bom_code,
                    "quantity": quantity,
                    "scope": scope,
                    "status": status,
                    "source": MODULES[module_key]["bom"].relative_to(ROOT).as_posix(),
                    "source_row": row_i,
                    "reason": "明示的候选/条件项，不计入当前采购场景",
                })
                continue

        source_bom_code = bom_code
        code_resolution_source = "current_BOM" if bom_code else None

        local_issues: list[str] = []
        codes_from_catalog: set[str] = set()
        mfrs_from_catalog: dict[str, str] = {}
        catalog_mpns: set[str] = set()
        for ref in refs:
            ci = by_ref.get(f"{MODULES[module_key]['label']}::{ref}")
            # Catalog module labels are source-native Chinese for these projects.
            if module_key == "FOC":
                ci = by_ref.get(f"FOC驱动模块::{ref}")
            elif module_key == "REGULATOR":
                ci = by_ref.get(f"稳压模块::{ref}")
            elif module_key == "PUMP":
                ci = by_ref.get(f"水泵驱动模块::{ref}")
            if ci:
                if norm_code(ci.get("supplier_part")):
                    codes_from_catalog.add(norm_code(ci.get("supplier_part")) or "")
                if clean(ci.get("manufacturer")):
                    mfrs_from_catalog[norm_manufacturer(ci.get("manufacturer")) or ""] = clean(ci.get("manufacturer")) or ""
                if clean(ci.get("mpn")):
                    catalog_mpns.add(norm_mpn(ci.get("mpn")))
        if len(codes_from_catalog) > 1:
            local_issues.append("current_catalog_has_multiple_supplier_codes_for_group")
        if bom_code and codes_from_catalog and bom_code not in codes_from_catalog:
            local_issues.append("current_BOM_supplier_code_differs_from_central_catalog")
        if not bom_code and len(codes_from_catalog) == 1:
            bom_code = next(iter(codes_from_catalog))
            code_resolution_source = "current_central_index"
            local_issues.append("supplier_code_filled_from_current_central_index")
        if not bom_code and mpn:
            exact_parts = [
                x for x in parts_by_mpn.get(norm_mpn(mpn), [])
                if norm_mpn(x.get("mpn")) == norm_mpn(mpn)
                and norm_manufacturer(x.get("manufacturer")) == norm_manufacturer(maker)
            ]
            exact_part_codes = {norm_code(c) for p in exact_parts for c in p.get("supplier_parts", []) if norm_code(c)}
            if len(exact_part_codes) == 1:
                bom_code = next(iter(exact_part_codes))
                code_resolution_source = "current_central_exact_MPN_manufacturer_index"
                local_issues.append("supplier_code_filled_from_exact_central_part_identity")
            elif len(exact_part_codes) > 1:
                local_issues.append("multiple_central_codes_for_exact_MPN_manufacturer")
        if not maker and len(mfrs_from_catalog) == 1:
            maker = next(iter(mfrs_from_catalog.values()))
            local_issues.append("manufacturer_filled_from_current_central_index")
        if maker and mfrs_from_catalog and norm_manufacturer(maker) not in mfrs_from_catalog:
            local_issues.append("manufacturer_differs_from_current_central_index")
        if mpn and catalog_mpns and norm_mpn(mpn) not in catalog_mpns:
            local_issues.append("manufacturer_part_differs_from_current_central_index")

        meta_items = [x for ref in refs for x in meta_by_ref.get(ref, [])]
        for meta in meta_items:
            meta_mpn = meta.get("manufacturerPart") if module_key == "FOC" else meta.get("manufacturer_part")
            if meta_mpn and mpn and norm_mpn(meta_mpn) != norm_mpn(mpn):
                continue
            if not maker:
                maker = clean(meta.get("manufacturer"))
            if not bom_code:
                meta_code = verified_code_from_meta(meta, module_key)
                if meta_code:
                    bom_code = meta_code
                    code_resolution_source = "module_price_input_exact_item_identity"
                    local_issues.append("supplier_code_filled_from_exact_item_identity_input")
        if module_key == "FOC" and not maker:
            # Input groups may carry one exact manufacturer even when a BOM export field is blank.
            candidates = {norm_manufacturer(x.get("manufacturer")): clean(x.get("manufacturer")) for x in meta_items if clean(x.get("manufacturer"))}
            if len(candidates) == 1:
                maker = next(iter(candidates.values()))
        if module_key != "FOC" and not bom_code and meta_items:
            codes = {verified_code_from_meta(x, module_key) for x in meta_items}
            codes.discard(None)
            if len(codes) == 1:
                bom_code = next(iter(codes))
                code_resolution_source = "module_price_input_exact_item_identity"
                local_issues.append("supplier_code_filled_from_exact_item_identity_input")

        if module_key != "FOC":
            # The BOM scope is authoritative. `library_candidate` means a CAD library binding is
            # pending, not that the exact on-board target disappears from purchase demand.
            identity_notes = [f"procurement_status={row.get('Procurement status','')}"] if row.get("Procurement status") else []
            local_issues.extend(identity_notes)
        if len(refs) > 1 and quantity != len(refs):
            # A grouped BOM line can legitimately have more than one unit per designator; preserve
            # its declared quantity, but report the distinction for review.
            local_issues.append(f"quantity_{quantity}_vs_reference_count_{len(refs)}")
        records.append(make_demand(
            module_key, scope, refs, mpn, maker, bom_code, quantity,
            MODULES[module_key]["bom"], row_i, local_issues,
            "current_BOM_target; CAD carrier excluded" if mpn else "current_BOM_identity_incomplete",
            source_supplier_part=source_bom_code,
            supplier_part_resolution_source=code_resolution_source,
        ))
    return records


def foc_demands(input_doc: dict[str, Any], bom_rows: list[dict[str, str]], catalog: dict[str, Any], candidate_exclusions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Current BOM is one board. Its shared bias is fitted only on A, DNP on B.
    by_ref, _ = catalog_indexes(catalog)
    meta_by_ref = ref_meta_map(input_doc, "FOC")
    a_records: list[dict[str, Any]] = []
    b_records: list[dict[str, Any]] = []
    ext_csv = read_csv(MODULES["FOC"]["supplement"])
    bias_rows = [r for r in ext_csv if "R_LOAD_VALID_BIAS" in split_refs(r.get("references"))]
    if len(bias_rows) != 1 or "人口数量覆盖" not in str(bias_rows[0].get("relation", "")):
        candidate_exclusions.append({
            "module": "FOC", "references": ["R_LOAD_VALID_BIAS"], "quantity": 0,
            "scope": "装配人口覆盖", "status": "supplement_contract_unresolved",
            "source": MODULES["FOC"]["supplement"].relative_to(ROOT).as_posix(),
            "source_row": None, "reason": "当前双板装配补充中未找到唯一且明确的不重复bias人口覆盖行",
        })
        bias_contract_valid = False
    else:
        br = bias_rows[0]
        bias_contract_valid = all(token in str(br.get("notes_source", "")) for token in ("FOC-A安装", "FOC-B DNP"))
        if not bias_contract_valid:
            candidate_exclusions.append({
                "module": "FOC", "references": ["R_LOAD_VALID_BIAS"], "quantity": 0,
                "scope": "装配人口覆盖", "status": "supplement_contract_unresolved",
                "source": MODULES["FOC"]["supplement"].relative_to(ROOT).as_posix(),
                "source_row": as_int(br.get("source_row", 0)), "reason": "bias覆盖行未同时明确FOC-A装/FOC-B DNP",
            })
    for row_i, row in enumerate(bom_rows, start=2):
        refs = split_refs(row.get("Designator"))
        q = as_int(row.get("Quantity", 0))
        is_bias = "R_LOAD_VALID_BIAS" in refs
        if is_bias and not bias_contract_valid:
            qa, qb = q, q
        else:
            qa = 1 if is_bias else q
            qb = 0 if is_bias else q
        if is_bias and qb == 0:
            candidate_exclusions.append({
                "module": "FOC-B", "references": ["R_LOAD_VALID_BIAS"],
                "manufacturer_part": clean(row.get("Manufacturer Part")),
                "manufacturer": clean(row.get("Manufacturer")),
                "supplier_part": norm_code(row.get("Supplier Part")), "quantity": 1,
                "scope": "DNP population override", "status": "FOC-B DNP",
                "source": MODULES["FOC"]["supplement"].relative_to(ROOT).as_posix(),
                "source_row": as_int(bias_rows[0].get("source_row", 0)) if bias_rows else None,
                "reason": "双板装配覆盖：该共享10k偏置仅FOC-A安装一次，FOC-B不装",
            })

        for module_label, target_q, target in (("FOC-A", qa, a_records), ("FOC-B", qb, b_records)):
            if target_q == 0:
                continue
            maker = clean(row.get("Manufacturer"))
            mpn = clean(row.get("Manufacturer Part"))
            code = norm_code(row.get("Supplier Part"))
            source_bom_code = code
            code_resolution_source = "current_BOM" if code else None
            local_issues: list[str] = []
            catalog_codes: set[str] = set()
            catalog_makers: dict[str, str] = {}
            catalog_mpns: set[str] = set()
            for ref in refs:
                ci = by_ref.get(f"FOC驱动模块::{ref}")
                if ci:
                    if norm_code(ci.get("supplier_part")):
                        catalog_codes.add(norm_code(ci.get("supplier_part")) or "")
                    if clean(ci.get("manufacturer")):
                        catalog_makers[norm_manufacturer(ci.get("manufacturer")) or ""] = clean(ci.get("manufacturer")) or ""
                    if clean(ci.get("mpn")):
                        catalog_mpns.add(norm_mpn(ci.get("mpn")))
            if len(catalog_codes) > 1:
                local_issues.append("current_catalog_has_multiple_supplier_codes_for_group")
            if code and catalog_codes and code not in catalog_codes:
                local_issues.append("current_BOM_supplier_code_differs_from_central_catalog")
            if not code and len(catalog_codes) == 1:
                code = next(iter(catalog_codes))
                code_resolution_source = "current_central_index"
                local_issues.append("supplier_code_filled_from_current_central_index")
            if not maker and len(catalog_makers) == 1:
                maker = next(iter(catalog_makers.values()))
                local_issues.append("manufacturer_filled_from_current_central_index")
            if maker and catalog_makers and norm_manufacturer(maker) not in catalog_makers:
                local_issues.append("manufacturer_differs_from_current_central_index")
            if mpn and catalog_mpns and norm_mpn(mpn) not in catalog_mpns:
                local_issues.append("manufacturer_part_differs_from_current_central_index")

            for ref in refs:
                for meta in meta_by_ref.get(ref, []):
                    if clean(meta.get("manufacturerPart")) and norm_mpn(meta.get("manufacturerPart")) == norm_mpn(mpn):
                        if not maker:
                            maker = clean(meta.get("manufacturer"))
                        if not code:
                            code = norm_code(meta.get("supplierPart"))
                            if code:
                                code_resolution_source = "FOC_price_input_exact_item_identity"
            if not maker and len(catalog_makers) == 1:
                maker = next(iter(catalog_makers.values()))
            d = make_demand(
                module_label, "FOC_BOM", refs, mpn, maker, code, target_q,
                MODULES["FOC"]["bom"], row_i, local_issues,
                "FOC-A fitted" if is_bias and module_label == "FOC-A" else ("FOC-B DNP override" if is_bias else "current FOC BOM; off-board items kept"),
                source_supplier_part=source_bom_code,
                supplier_part_resolution_source=code_resolution_source,
            )
            target.append(d)

    # The paired two-position wire-side connector is additional to the PCB-side BOM connector.
    ext_items = input_doc.get("externalSupplementalItems", [])
    ext_rows = [r for r in ext_csv if "配对线端" in str(r.get("references", "")) and "额外采购" in str(r.get("relation", ""))]
    for row in ext_rows:
        total_q = as_int(row.get("quantity_source", 0))
        if total_q <= 0 or total_q % 2:
            split_issue = [f"two_board_supplement_total_quantity_{total_q}_cannot_be_evenly_split"]
            per_board = 0
        else:
            split_issue = []
            per_board = total_q // 2
        ref = str(row.get("references") or "J_LOAD_POWER_VALID配对线端")
        mpn = clean(row.get("mpn_source"))
        maker = clean(row.get("manufacturer_source"))
        matched_meta = [x for x in ext_items if norm_mpn(x.get("manufacturerPart")) == norm_mpn(mpn) and norm_manufacturer(x.get("manufacturer")) == norm_manufacturer(maker)]
        code_set = {norm_code(x.get("supplierPart")) for x in matched_meta if norm_code(x.get("supplierPart"))}
        code = next(iter(code_set)) if len(code_set) == 1 else None
        if not matched_meta:
            split_issue.append("external_connector_not_crosschecked_by_FOC_price_input")
        if len(code_set) > 1:
            split_issue.append("external_connector_has_conflicting_input_C_codes")
        for module_label, target in (("FOC-A", a_records), ("FOC-B", b_records)):
            if per_board:
                target.append(make_demand(
                    module_label, clean(row.get("scope_source")) or "board-external/cable harness", [ref],
                    mpn, maker, code, per_board, MODULES["FOC"]["supplement"], as_int(row.get("source_row", 0)),
                    split_issue, "extra wire-side mate; not duplicate of PCB-side connector",
                    source_supplier_part=code,
                    supplier_part_resolution_source="FOC_assembly_supplement_and_price_input" if code else None,
                ))
    # Validate the explicit population override and expected module piece counts.
    q_a = sum(x["quantity"] for x in a_records)
    q_b = sum(x["quantity"] for x in b_records)
    expected_a = input_doc.get("quantityBasis", {}).get("FOC_A_total", 156)
    expected_b = input_doc.get("quantityBasis", {}).get("FOC_B_total", 155)
    if q_a != expected_a or q_b != expected_b:
        # Quantity basis in earlier inputs may be free text; verification is also performed in build_output.
        pass
    return a_records, b_records


def make_quote_index(quote_doc: dict[str, Any]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    exact: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    code_mpn: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in quote_doc.get("observations", []):
        mpn = norm_mpn(o.get("mpn"))
        maker = norm_manufacturer(o.get("manufacturer"))
        code = norm_code(o.get("supplier_part"))
        if not mpn or not maker or not code:
            continue
        key = (maker, mpn, code)
        exact[key].append(o)
        code_mpn[(code, mpn)].append(o)
    chosen = {}
    for key, observations in exact.items():
        chosen[key] = sorted(observations, key=lambda x: (str(x.get("captured_at_utc") or ""), str(x.get("quote_id") or "")))[-1]
    return chosen, code_mpn


def apply_quote_resolution(demands: list[dict[str, Any]], quote_doc: dict[str, Any]) -> None:
    _, by_code_mpn = make_quote_index(quote_doc)
    # Propagate a single, exact known manufacturer among repeated exact MPN + C-code records.
    makers: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for d in demands:
        if d.get("manufacturer_part") and d.get("supplier_part") and d.get("manufacturer"):
            k = (norm_code(d["supplier_part"]) or "", norm_mpn(d["manufacturer_part"]))
            makers[k][norm_manufacturer(d["manufacturer"]) or ""] = d["manufacturer"]
    quote_makers: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    verified_codes_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for obs in quote_doc.get("observations", []):
        if obs.get("identity_verified") is True and norm_code(obs.get("supplier_part")) and norm_mpn(obs.get("mpn")) and clean(obs.get("manufacturer")):
            k = (norm_code(obs.get("supplier_part")) or "", norm_mpn(obs.get("mpn")))
            quote_makers[k][norm_manufacturer(obs.get("manufacturer")) or ""] = clean(obs.get("manufacturer")) or ""
            identity = (norm_manufacturer(obs.get("manufacturer")) or "", norm_mpn(obs.get("mpn")))
            verified_codes_by_identity[identity].add(norm_code(obs.get("supplier_part")) or "")
    for d in demands:
        if d.get("manufacturer") or not d.get("manufacturer_part") or not d.get("supplier_part"):
            continue
        if "ambiguous_manufacturer_candidates" in d.get("identity_notes", []):
            continue
        k = (norm_code(d["supplier_part"]) or "", norm_mpn(d["manufacturer_part"]))
        known = makers.get(k, {})
        if len(known) == 1:
            d["manufacturer"] = next(iter(known.values()))
            d["identity_notes"] = sorted(set(d.get("identity_notes", []) + ["manufacturer_resolved_from_same_exact_MPN_and_C_code_elsewhere_in_current_demand"]))
        elif len(quote_makers.get(k, {})) == 1:
            qmaker_norm, qmaker = next(iter(quote_makers[k].items()))
            candidates = {norm_manufacturer(x) for x in d.get("manufacturer_candidates", []) if x}
            if not candidates or qmaker_norm in candidates:
                d["manufacturer"] = qmaker
                d["identity_notes"] = sorted(set(d.get("identity_notes", []) + ["manufacturer_resolved_by_verified_exact_quote_identity"]))
        if d.get("manufacturer"):
            d["identity_notes"] = [x for x in d.get("identity_notes", []) if x != "manufacturer_unresolved"]

    # Fill a missing native/procurement C-code only from a verified exact MPN+manufacturer quote.
    # Preserve the source value (None) so derived mapping remains auditable.
    for d in demands:
        mpn = norm_mpn(d.get("manufacturer_part"))
        maker = norm_manufacturer(d.get("manufacturer"))
        if not mpn or not maker:
            continue
        quote_codes = verified_codes_by_identity.get((maker, mpn), set())
        existing_code = norm_code(d.get("supplier_part"))
        source_code = norm_code(d.get("source_supplier_part"))
        if existing_code and quote_codes and existing_code not in quote_codes:
            d["identity_notes"] = sorted(set(d.get("identity_notes", []) + ["supplier_code_conflict_with_verified_quote"]))
            continue
        if source_code is None and len(quote_codes) == 1:
            quoted_code = next(iter(quote_codes))
            if existing_code is None:
                d["supplier_part"] = quoted_code
            if norm_code(d.get("supplier_part")) == quoted_code:
                d["supplier_code_verified_from_quote"] = True
                d["supplier_part_resolution_source"] = "verified_quote_identity"
                d["identity_notes"] = sorted(set(d.get("identity_notes", []) + ["supplier_code_verified_from_quote"]))
                d["identity_notes"] = [x for x in d["identity_notes"] if x != "no_verified_lcsc_code"]
        elif source_code is None and not existing_code and len(quote_codes) > 1:
            d["identity_notes"] = sorted(set(d.get("identity_notes", []) + ["ambiguous_verified_quote_C_codes_for_exact_identity"]))


def quote_match(d: dict[str, Any], quote_doc: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    code = norm_code(d.get("supplier_part"))
    mpn = norm_mpn(d.get("manufacturer_part"))
    maker = norm_manufacturer(d.get("manufacturer"))
    if "supplier_code_conflict_with_verified_quote" in d.get("identity_notes", []):
        return None, "supplier_code_conflict_with_verified_quote"
    if "ambiguous_verified_quote_C_codes_for_exact_identity" in d.get("identity_notes", []):
        return None, "ambiguous_verified_quote_C_codes_for_exact_identity"
    if not mpn:
        return None, "missing_exact_mpn"
    if not code:
        return None, "no_verified_lcsc_code"
    if not maker:
        return None, "manufacturer_unresolved"
    exact, by_code_mpn = make_quote_index(quote_doc)
    candidates = by_code_mpn.get((code, mpn), [])
    if not candidates:
        return None, "no_current_public_quote_for_exact_C_code_and_MPN"
    identity_matches = [x for x in candidates if norm_manufacturer(x.get("manufacturer")) == maker]
    identity_verified = [x for x in identity_matches if x.get("identity_verified") is True]
    if not identity_verified:
        if any(x.get("identity_verified") is True for x in candidates):
            return None, "quote_manufacturer_does_not_match_current_identity"
        return None, "quote_identity_not_verified"
    obs = exact.get((maker, mpn, code))
    if not obs:
        return None, "quote_identity_not_verified"
    if str(obs.get("currency")) != "CNY":
        return None, "quote_currency_not_CNY"
    if not isinstance(obs.get("price_tiers"), list) or not obs.get("price_tiers"):
        return None, "quote_has_no_public_price_tiers"
    return obs, "verified_exact_quote"


def decimal_or_none(value: Any) -> Decimal | None:
    try:
        if value is None or value == "":
            return None
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def round_up_multiple(q: int, multiple: int | None) -> int:
    if not multiple or multiple <= 1:
        return q
    return int((Decimal(q) / Decimal(multiple)).to_integral_value(rounding=ROUND_CEILING)) * multiple


def tier_price(obs: dict[str, Any], quantity: int) -> Decimal | None:
    tiers = []
    for t in obs.get("price_tiers", []):
        q = as_int(t.get("min_quantity", 0))
        p = decimal_or_none(t.get("unit_price"))
        if q > 0 and p is not None and p >= 0:
            tiers.append((q, p))
    eligible = [x for x in tiers if x[0] <= quantity]
    if not eligible:
        return None
    return max(eligible, key=lambda x: x[0])[1]


def choose_purchase_lot(obs: dict[str, Any], required_qty: int) -> dict[str, Any]:
    moq = as_int(obs.get("moq"), default=0) if obs.get("moq") is not None else None
    multiple = as_int(obs.get("order_multiple"), default=0) if obs.get("order_multiple") is not None else None
    if moq is not None and moq < 1:
        moq = None
    if multiple is not None and multiple < 1:
        multiple = None
    moq_basis = obs.get("moq_basis") or ("recorded_quote_field" if moq is not None else "unknown_not_inferred_from_tiers_or_packaging")
    floor = max(required_qty, moq or 1)
    purchase_qty = round_up_multiple(floor, multiple)
    unit = tier_price(obs, purchase_qty)
    if unit is None:
        # If no published tier applies at the actual demand/MOQ lot, buy at the first
        # published threshold. Do not jump to a large bulk tier just to lower one-set cost.
        later_tiers = sorted(
            as_int(t.get("min_quantity", 0))
            for t in obs.get("price_tiers", [])
            if as_int(t.get("min_quantity", 0)) > purchase_qty
        )
        if later_tiers:
            purchase_qty = round_up_multiple(later_tiers[0], multiple)
            unit = tier_price(obs, purchase_qty)
    if unit is None:
        return {
            "cash_cost_at_required_order_lot_cny": None,
            "purchase_quantity": None,
            "unit_price_cny": None,
            "tier_min_quantity": None,
            "overbuy_quantity": None,
            "status": "no_price_tier_covers_required_or_MOQ_quantity",
            "moq": moq,
            "moq_basis": moq_basis,
            "moq_known": moq is not None,
            "order_multiple_known": multiple is not None,
        }
    q = purchase_qty
    cash = unit * q
    tier_min = max(
        as_int(t.get("min_quantity", 0))
        for t in obs.get("price_tiers", [])
        if decimal_or_none(t.get("unit_price")) == unit and as_int(t.get("min_quantity", 0)) <= q
    )
    stock = as_int(obs.get("stock_quantity"), default=-1) if obs.get("stock_quantity") is not None else None
    availability = str(obs.get("availability") or "unknown").strip().casefold()
    if availability == "zero_snapshot" or stock == 0:
        stock_status = "zero_snapshot_needs_review"
    elif availability in {"unknown", "", "out_of_stock", "unavailable"}:
        stock_status = "availability_needs_review" if availability in {"out_of_stock", "unavailable"} else "unknown"
    elif availability in {"in_stock", "in_stock_snapshot"} and stock is not None and stock > 0:
        stock_status = "shortage" if stock < q else "sufficient_at_observation"
    else:
        stock_status = "unknown"
    if moq is None and multiple is None:
        lot_status = "MOQ_and_order_multiple_unknown_cost_assumes_required_quantity"
    elif moq is None:
        lot_status = "MOQ_unknown_cost_assumes_required_quantity"
    elif multiple is None:
        lot_status = "order_multiple_unknown_assumed_piece_quantity"
    else:
        lot_status = "ok"
    return {
        "cash_cost_at_required_order_lot_cny": cash,
        "purchase_quantity": q,
        "unit_price_cny": unit,
        "tier_min_quantity": tier_min,
        "overbuy_quantity": q - required_qty,
        "status": lot_status,
        "order_multiple_known": multiple is not None,
        "moq": moq,
        "moq_basis": moq_basis,
        "moq_known": moq is not None,
        "order_multiple": multiple,
        "stock_status": stock_status,
        "stock_quantity": stock,
    }


def decimal_json(d: Any) -> Any:
    if isinstance(d, Decimal):
        return float(d.quantize(Decimal("0.00000001")))
    if isinstance(d, dict):
        return {k: decimal_json(v) for k, v in d.items()}
    if isinstance(d, list):
        return [decimal_json(x) for x in d]
    return d


def scenario_result(
    key: str,
    spec: dict[str, Any],
    module_demands: dict[str, list[dict[str, Any]]],
    quote_doc: dict[str, Any],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    module_rows: dict[str, list[dict[str, Any]]] = {}
    for m, factor in spec["modules"].items():
        if factor <= 0:
            continue
        rows = []
        for d in module_demands.get(m, []):
            if d.get("quantity", 0) <= 0:
                continue
            x = dict(d)
            x["scenario_quantity"] = d["quantity"] * factor
            x["scenario_quantity_by_scope"] = {k: v * factor for k, v in d.get("quantity_by_scope", {}).items()}
            x["scenario_quantity_by_scope_bucket"] = {k: v * factor for k, v in d.get("quantity_by_scope_bucket", {}).items()}
            rows.append(x)
            selected.append(x)
        module_rows[m] = rows

    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for d in selected:
        k = demand_key(d)
        if k not in groups:
            groups[k] = {
                "manufacturer_part": d.get("manufacturer_part"),
                "manufacturer": d.get("manufacturer"),
                "supplier_part": d.get("supplier_part"),
                "supplier": d.get("supplier"),
                "quantity": 0,
                "quantity_by_module": defaultdict(int),
                "references_by_module": defaultdict(set),
                "scopes": set(),
                "identity_notes": set(),
                "source_records": [],
                "source_supplier_parts": set(),
                "supplier_part_resolution_sources": set(),
                "supplier_code_verified_from_quote": False,
                "quantity_by_scope_by_module": defaultdict(lambda: defaultdict(int)),
                "quantity_by_scope_bucket_by_module": defaultdict(lambda: defaultdict(int)),
                "references_by_scope_bucket_by_module": defaultdict(lambda: defaultdict(set)),
            }
        g = groups[k]
        g["quantity"] += d["scenario_quantity"]
        g["quantity_by_module"][d["module"]] += d["scenario_quantity"]
        g["references_by_module"][d["module"]].update(d.get("references", []))
        g["scopes"].update(d.get("scopes", [d.get("scope", "unspecified")]))
        g["identity_notes"].update(d.get("identity_notes", []))
        g["source_supplier_parts"].update(d.get("source_supplier_parts", [d.get("source_supplier_part")]))
        g["supplier_part_resolution_sources"].update(d.get("supplier_part_resolution_sources", []))
        g["supplier_code_verified_from_quote"] = g["supplier_code_verified_from_quote"] or bool(d.get("supplier_code_verified_from_quote"))
        for scope, qscope in d.get("scenario_quantity_by_scope", {}).items():
            g["quantity_by_scope_by_module"][d["module"]][scope] += qscope
        for bucket, qbucket in d.get("scenario_quantity_by_scope_bucket", {}).items():
            g["quantity_by_scope_bucket_by_module"][d["module"]][bucket] += qbucket
            if qbucket:
                g["references_by_scope_bucket_by_module"][d["module"]][bucket].update(
                    d.get("references_by_scope_bucket", {}).get(bucket, d.get("references", []))
                )
        g["source_records"].append({"module": d["module"], "source": d["source"], "row": d.get("source_row")})

    sku_rows = []
    for _, g in sorted(groups.items(), key=lambda kv: kv[0]):
        probe = {
            "manufacturer_part": g["manufacturer_part"],
            "manufacturer": g["manufacturer"],
            "supplier_part": g["supplier_part"],
        }
        obs, match_status = quote_match(probe, quote_doc)
        item = {
            "manufacturer_part": g["manufacturer_part"],
            "manufacturer": g["manufacturer"],
            "supplier_part": g["supplier_part"],
            "supplier": g["supplier"],
            "source_supplier_part": next(iter(g["source_supplier_parts"])) if len(g["source_supplier_parts"]) == 1 else None,
            "source_supplier_parts": sorted(g["source_supplier_parts"], key=lambda v: v or ""),
            "supplier_part_resolution_sources": sorted(g["supplier_part_resolution_sources"]),
            "supplier_code_verified_from_quote": g["supplier_code_verified_from_quote"],
            "quantity_required": g["quantity"],
            "quantity_by_module": dict(sorted(g["quantity_by_module"].items())),
            "references_by_module": {m: sorted(v) for m, v in sorted(g["references_by_module"].items())},
            "quantity_by_scope_by_module": {
                m: dict(sorted(scopes.items())) for m, scopes in sorted(g["quantity_by_scope_by_module"].items())
            },
            "quantity_by_scope_bucket_by_module": {
                m: dict(sorted(scopes.items())) for m, scopes in sorted(g["quantity_by_scope_bucket_by_module"].items())
            },
            "references_by_scope_bucket_by_module": {
                m: {bucket: sorted(refs) for bucket, refs in sorted(buckets.items())}
                for m, buckets in sorted(g["references_by_scope_bucket_by_module"].items())
            },
            "scopes": sorted(g["scopes"]),
            "identity_notes": sorted(g["identity_notes"]),
            "quote_match_status": match_status,
            "quote": None,
            "installed_value_cny": None,
            "purchase": None,
            "stock_status": "not_queried",
            "stock_shortage": False,
            "unpriced_quantity": 0 if obs else g["quantity"],
        }
        if obs:
            lot = choose_purchase_lot(obs, g["quantity"])
            unit = lot.get("unit_price_cny")
            installed = unit * g["quantity"] if unit is not None else None
            stock = lot.get("stock_quantity")
            shortage = lot.get("stock_status") == "shortage"
            item.update({
                "quote": {
                    "quote_id": obs.get("quote_id"),
                    "source_url": obs.get("source_url"),
                    "captured_at_utc": obs.get("captured_at_utc"),
                    "currency": obs.get("currency"),
                    "price_tiers": obs.get("price_tiers"),
                    "moq": obs.get("moq"),
                    "order_multiple": obs.get("order_multiple"),
                    "stock_quantity": obs.get("stock_quantity"),
                    "availability": obs.get("availability"),
                    "tax_status": obs.get("tax_status"),
                    "shipping_included": obs.get("shipping_included"),
                    "identity_evidence": obs.get("identity_evidence"),
                    "raw_evidence": obs.get("raw_evidence"),
                },
                "installed_value_cny": installed,
                "purchase": lot,
                "stock_status": lot.get("stock_status", "unknown"),
                "stock_shortage": shortage,
                "unpriced_quantity": 0,
            })
        sku_rows.append(item)

    known_installed = sum((decimal_or_none(x.get("installed_value_cny")) or Decimal(0)) for x in sku_rows if x.get("installed_value_cny") is not None)
    known_cash = sum(
        (decimal_or_none((x.get("purchase") or {}).get("cash_cost_at_required_order_lot_cny")) or Decimal(0))
        for x in sku_rows
        if x.get("purchase") and x["purchase"].get("cash_cost_at_required_order_lot_cny") is not None
    )
    all_priced = all(x["quote_match_status"] == "verified_exact_quote" for x in sku_rows)
    all_quantity_identity_known = all(
        x["manufacturer_part"] and x["manufacturer"] and x["supplier_part"] and x["quantity_required"] > 0
        for x in sku_rows
    )
    all_cash_constraints_known = all((x.get("purchase") or {}).get("order_multiple_known", False) and (x.get("purchase") or {}).get("moq_known", False) for x in sku_rows if x["quote_match_status"] == "verified_exact_quote")
    quoted_overbuy_qty = sum(as_int((x.get("purchase") or {}).get("overbuy_quantity", 0)) for x in sku_rows if x["quote_match_status"] == "verified_exact_quote")
    complete_total = known_installed if all_priced and all_quantity_identity_known else None
    complete_cash = known_cash if all_priced and all_quantity_identity_known and all_cash_constraints_known and not any(x["stock_shortage"] for x in sku_rows) else None

    per_module = {}
    for m, rows in module_rows.items():
        qtotal = sum(x["scenario_quantity"] for x in rows)
        related_skus = [x for x in sku_rows if x.get("quantity_by_module", {}).get(m, 0)]
        priced_skus = [x for x in related_skus if x["quote_match_status"] == "verified_exact_quote"]
        partial = Decimal(0)
        for sku in priced_skus:
            qmod = sku["quantity_by_module"][m]
            unit = decimal_or_none((sku.get("purchase") or {}).get("unit_price_cny"))
            if unit is not None:
                partial += unit * qmod
        unpriced = [x for x in related_skus if x["quote_match_status"] != "verified_exact_quote"]
        query_failures = [x for x in unpriced if x["quote_match_status"] == "no_current_public_quote_for_exact_C_code_and_MPN"]
        unresolved_identity = [x for x in related_skus if x["quote_match_status"] in {"missing_exact_mpn", "no_verified_lcsc_code", "manufacturer_unresolved", "quote_manufacturer_does_not_match_current_identity", "supplier_code_conflict_with_verified_quote", "ambiguous_verified_quote_C_codes_for_exact_identity"}]
        shortages = [x for x in related_skus if x["stock_shortage"]]
        stock_review = [x for x in related_skus if x["stock_status"] in {"unknown", "zero_snapshot_needs_review", "availability_needs_review"}]
        scope_breakdown = {}
        for bucket in ("on_board", "core_board_module", "external_accessory"):
            bucket_skus = [x for x in related_skus if x.get("quantity_by_scope_bucket_by_module", {}).get(m, {}).get(bucket, 0)]
            bucket_priced = [x for x in bucket_skus if x["quote_match_status"] == "verified_exact_quote"]
            bucket_unpriced = [x for x in bucket_skus if x["quote_match_status"] != "verified_exact_quote"]
            bucket_partial = Decimal(0)
            for sku in bucket_priced:
                quantity = sku["quantity_by_scope_bucket_by_module"][m][bucket]
                unit = decimal_or_none((sku.get("purchase") or {}).get("unit_price_cny"))
                if unit is not None:
                    bucket_partial += unit * quantity
            scope_breakdown[bucket] = {
                "demand_quantity": sum(x["quantity_by_scope_bucket_by_module"][m][bucket] for x in bucket_skus),
                "verified_quoted_sku_count": len(bucket_priced),
                "verified_installed_subtotal_cny": bucket_partial,
                "unpriced_sku_count": len(bucket_unpriced),
                "unpriced_quantity": sum(x["quantity_by_scope_bucket_by_module"][m][bucket] for x in bucket_unpriced),
                "complete_scope_total_cny": bucket_partial if not bucket_unpriced else None,
                "unpriced_items": [
                    {
                        "manufacturer_part": x["manufacturer_part"],
                        "manufacturer": x["manufacturer"],
                        "supplier_part": x["supplier_part"],
                        "quantity": x["quantity_by_scope_bucket_by_module"][m][bucket],
                        "references": x.get("references_by_scope_bucket_by_module", {}).get(m, {}).get(bucket, []),
                        "reason": x["quote_match_status"],
                    } for x in bucket_unpriced
                ],
            }
        per_module[m] = {
            "demand_quantity": qtotal,
            "sku_count": len(related_skus),
            "verified_quoted_sku_count": len(priced_skus),
            "verified_installed_subtotal_cny": partial,
            "complete_module_total_cny": partial if not unpriced and all(
                x.get("manufacturer_part") and x.get("manufacturer") and x.get("supplier_part") and x.get("quantity_required", 0) > 0
                for x in related_skus
            ) else None,
            "unpriced_sku_count": len(unpriced),
            "unpriced_quantity": sum(x["quantity_by_module"][m] for x in unpriced),
            "unpriced_items": [
                {
                    "manufacturer_part": x["manufacturer_part"],
                    "manufacturer": x["manufacturer"],
                    "supplier_part": x["supplier_part"],
                    "quantity": x["quantity_by_module"][m],
                    "references": x["references_by_module"][m],
                    "reason": x["quote_match_status"],
                } for x in unpriced
            ],
            "query_failure_sku_count": len(query_failures),
            "query_failure_quantity": sum(x["quantity_by_module"][m] for x in query_failures),
            "identity_unresolved_sku_count": len(unresolved_identity),
            "stock_shortage_sku_count": len(shortages),
            "stock_unknown_sku_count": len(stock_review),
            "scope_breakdown": scope_breakdown,
        }

    cash_risk_examples = []
    for mpn in ("BZT52H-C3V6,115", "GRM32ER61C476KE15L"):
        matches = [x for x in sku_rows if norm_mpn(x.get("manufacturer_part")) == norm_mpn(mpn)]
        if matches:
            x = matches[0]
            purchase = x.get("purchase") or {}
            cash_risk_examples.append({
                "manufacturer_part": x.get("manufacturer_part"),
                "manufacturer": x.get("manufacturer"),
                "supplier_part": x.get("supplier_part"),
                "quantity_required": x.get("quantity_required"),
                "moq": (x.get("quote") or {}).get("moq"),
                "order_multiple": (x.get("quote") or {}).get("order_multiple"),
                "purchase_quantity": purchase.get("purchase_quantity"),
                "unit_price_cny": purchase.get("unit_price_cny"),
                "cash_cost_cny": purchase.get("cash_cost_at_required_order_lot_cny"),
                "overbuy_quantity": purchase.get("overbuy_quantity"),
                "quote_match_status": x.get("quote_match_status"),
                "note": "首轮采购现金风险示例；按合并需求与已声明MOQ/阶梯估算，不代表建议替换或最低系统成本",
            })
        else:
            cash_risk_examples.append({"manufacturer_part": mpn, "status": "not_present_in_this_scenario"})

    return decimal_json({
        "scenario": key,
        "label": spec["label"],
        "module_multipliers": spec["modules"],
        "modules": per_module,
        "sku_demands": sku_rows,
        "cash_purchase_risk_examples": cash_risk_examples,
        "verified_installed_subtotal_cny": known_installed,
        "complete_installed_total_cny": complete_total,
        "known_sku_cash_estimate_assuming_declared_MOQ_and_piece_step_cny": known_cash,
        "known_sku_overbuy_quantity": quoted_overbuy_qty,
        "known_sku_cash_overbuy_vs_installed_value_cny": known_cash - known_installed,
        "complete_minimum_cash_purchase_cny": complete_cash,
        "all_current_skus_have_verified_quotes": all_priced,
        "all_required_quantities_and_identities_known": all_quantity_identity_known,
        "all_moq_and_order_multiple_known": all_cash_constraints_known,
        "stock_shortage_sku_count": sum(1 for x in sku_rows if x["stock_shortage"]),
        "stock_unknown_sku_count": sum(1 for x in sku_rows if x["stock_status"] in {"unknown", "zero_snapshot_needs_review", "availability_needs_review"}),
        "stock_review_sku_count": sum(1 for x in sku_rows if x["stock_status"] in {"unknown", "zero_snapshot_needs_review", "availability_needs_review"}),
        "unpriced_sku_count": sum(1 for x in sku_rows if x["quote_match_status"] != "verified_exact_quote"),
        "unpriced_quantity": sum(x["unpriced_quantity"] for x in sku_rows),
        "cost_scope": "当前BOM采购件+板外配套；不含PCB、SMT、运输；税费仅按报价页明确字段记录，不推算",
        "partial_cost_semantics": "已核价SKU的实装折算小计/最低现金估算；未核价SKU数量保留，完整总价为null",
    })


def json_safe_text(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_outputs() -> tuple[dict[str, Any], str]:
    price_doc = read_json(PRICE_TABLE)
    catalog = read_json(CATALOG)
    inputs = {k: read_json(v["input"]) for k, v in MODULES.items()}
    bms_rows = read_csv(MODULES["BMS"]["bom"])
    foc_rows = read_csv(MODULES["FOC"]["bom"], "\t")
    regulator_rows = read_csv(MODULES["REGULATOR"]["bom"])
    pump_rows = read_csv(MODULES["PUMP"]["bom"])

    bindings: dict[str, dict[str, Any]] = {}
    register_binding(bindings, "商城报价表", PRICE_TABLE, "current public SZLCSC CNY quote observations")
    register_binding(bindings, "中央元件索引", CATALOG, "read-only exact per-designator MPN/manufacturer/C-code cross-check")
    register_binding(bindings, "BMS采购补充", PURCHASE_SUPPLEMENT, "BMS external accessory source; already represented in BMS BOM/input, no duplicate add")
    for key, spec in MODULES.items():
        register_binding(bindings, f"{key}核价输入", spec["input"], "module normalized procurement identity and price-audit input")
        register_binding(bindings, f"{key}当前BOM", spec["bom"], "current source of purchase quantities")
        if spec.get("supplement"):
            register_binding(bindings, f"{key}装配补充", spec["supplement"], "population overrides and additive off-board purchase lines")

    input_issues: list[str] = []
    # Verify each audit input is bound to the BOM bytes actually being priced.
    for key, spec in MODULES.items():
        input_doc = inputs[key]
        actual_sha = sha256(spec["bom"])
        declared_sha = None
        if key == "BMS":
            declared_sha = input_doc.get("source_binding", {}).get("bom_sha256")
        elif key == "FOC":
            declared_sha = next((x.get("sha256") for x in input_doc.get("sourceFiles", []) if x.get("path", "").endswith("FOC_G431_BOM.csv")), None)
        else:
            declared_sha = input_doc.get("source_bom", {}).get("sha256")
        if declared_sha and declared_sha.casefold() != actual_sha.casefold():
            input_issues.append(f"{spec['label']}核价输入绑定BOM SHA与当前BOM不一致")

    candidate_exclusions: list[dict[str, Any]] = []
    bms_records, bms_issues, bms_total = bms_demands(inputs["BMS"], bms_rows, catalog)
    input_issues.extend(bms_issues)
    foc_a, foc_b = foc_demands(inputs["FOC"], foc_rows, catalog, candidate_exclusions)
    regulator_records = generic_module_demands("REGULATOR", inputs["REGULATOR"], regulator_rows, catalog, candidate_exclusions)
    pump_records = generic_module_demands("PUMP", inputs["PUMP"], pump_rows, catalog, candidate_exclusions)

    # The user decision is to keep only the current eight supercapacitors; no 16-piece alternative
    # remains an active candidate or procurement recommendation.
    supercap_refs = [r for r in regulator_rows if str(r.get("Reference", "")).strip().startswith("SC") and norm_mpn(r.get("Manufacturer part")) == norm_mpn("SC0350-270-RSS")]
    supercap_current = sum(as_int(r.get("Quantity", 0)) for r in supercap_refs)
    supercap_sensitivity = {
        "manufacturer": "LICAP Technologies",
        "manufacturer_part": "SC0350-270-RSS",
        "current_bom_quantity": supercap_current,
        "candidate_total_system_quantity": supercap_current,
        "candidate_incremental_quantity": 0,
        "supplier_part": None,
        "price_cny": None,
        "included_in_current_scenario": True,
        "additional_quantity_included": 0,
        "configuration_status": "single_current_configuration_only; no active 16-piece alternative",
        "source": "电力/稳压模块/原理图准备/BOM.csv",
        "note": "当前设计目标使用现有8只，增量0；超容首要承接制动回生，有余电时辅助启动。先前16只方案已撤回；如重新采购须另行核价。",
    }

    # Add source binding checks and input quantities as audit facts, not as current timestamps.
    # Keep raw per-reference records through identity propagation; merge once afterwards so
    # scope/placement metadata is retained and identical SKUs can be resolved cross-module.
    module_demands = {
        "BMS": bms_records,
        "FOC-A": foc_a,
        "FOC-B": foc_b,
        "REGULATOR": regulator_records,
        "PUMP": pump_records,
    }
    all_demands = [d for values in module_demands.values() for d in values]
    apply_quote_resolution(all_demands, price_doc)
    # Rebuild after manufacturer propagation so repeated supplier items merge into one exact SKU.
    module_demands = {m: merge_demands(ds) for m, ds in module_demands.items()}

    quantity_a = sum(x["quantity"] for x in module_demands["FOC-A"])
    quantity_b = sum(x["quantity"] for x in module_demands["FOC-B"])
    foc_basis = inputs["FOC"].get("quantityBasis", {})
    foc_native_a = as_int(foc_basis.get("FOC_AActual", 156))
    foc_native_b = as_int(foc_basis.get("FOC_BActual", 155))
    foc_mates = as_int(foc_basis.get("extraExternalMatePlugUnits", 2))
    foc_mates_a = foc_mates // 2
    foc_mates_b = foc_mates - foc_mates_a
    expected_a = foc_native_a + foc_mates_a
    expected_b = foc_native_b + foc_mates_b
    if quantity_a != expected_a:
        input_issues.append(
            f"FOC-A当前采购数为{quantity_a}，预期原生实装{foc_native_a}+线端配套{foc_mates_a}={expected_a}"
        )
    if quantity_b != expected_b:
        input_issues.append(
            f"FOC-B当前采购数为{quantity_b}，预期原生实装{foc_native_b}+线端配套{foc_mates_b}={expected_b}"
        )
    if bms_total != 271:
        input_issues.append(f"BMS输入当前采购数为{bms_total}，既有核对基线为271")
    if supercap_current != 8:
        input_issues.append(f"当前SC0350-270-RSS数量为{supercap_current}，既有当前BOM配置预期8")

    scenarios = {
        key: scenario_result(key, spec, module_demands, price_doc)
        for key, spec in SCENARIOS.items()
    }

    # External heavy/large items are separated for easy system-cost review; these remain in BOM totals once.
    large_external_refs = {
        "FOC": {"CBUS1", "CBUS2", "CBUS3", "R_DUMP_A", "R_DUMP_B", "F_MAIN", "F_BRAKE", "J_BRAKE_R", "J_LOAD_POWER_VALID配对线端"},
        "REGULATOR": {"C_BUS_BULK1", "C_BUS_BULK2", "C_BUS_BULK3", "C_BUS_BULK4", "C_BUS_BULK5", "C_BUS_BULK6", "R_DUMP_EXT", "F_BUS", "F_SC"},
        "PUMP": {"F_BUS", "H_FUSE", "P_BUS", "P_MOTOR"},
    }

    def summarize_large(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for d in items:
            refs = set(d.get("references", []))
            module_key = "FOC" if d.get("module", "").startswith("FOC-") else d.get("module")
            scopes = " ".join(d.get("scopes", []))
            external_scope = any(token in scopes.casefold() for token in (
                "external_accessory", "off_board", "off-board", "harness",
                "board_interface_package_pending", "board-external", "daughterboard",
                "core_board_module", "on_board_module_assembly",
            ))
            large_ref = bool(refs & large_external_refs.get(module_key, set()))
            if refs and module_key == "REGULATOR" and any(r.startswith("SC") for r in refs):
                large_ref = True
            if external_scope or large_ref:
                note = "已包含在上方当前BOM数量中，仅作大型/板外件分列，不重复加到合计"
                if any(x.startswith("C_BUS_BULK") for x in refs):
                    note = "当前BOM Scope=ON_BOARD、网表接BUS_EXT；按母线大电容独立列示且只计一次"
                elif any(x.startswith("SC") for x in refs) and module_key == "REGULATOR":
                    note = "当前稳压BOM内8只LICAP超级电容；单独展示且已包含在当前场景，不重复加件"
                out.append({
                    "module": d["module"], "references": d["references"],
                    "manufacturer": d.get("manufacturer"), "manufacturer_part": d.get("manufacturer_part"),
                    "supplier_part": d.get("supplier_part"), "quantity": d.get("quantity"),
                    "scope": d.get("scopes", []),
                    "note": note,
                })
        return out

    # FOC input quantity basis and exact special population are cross-checked from the live BOM.
    foc_native_a = sum(as_int(r.get("Quantity", 0)) for r in foc_rows)
    foc_bias = [r for r in foc_rows if "R_LOAD_VALID_BIAS" in split_refs(r.get("Designator"))]
    foc_supp_rows = read_csv(MODULES["FOC"]["supplement"])
    bms_scopes = defaultdict(int)
    for d in module_demands["BMS"]:
        bms_scopes[d.get("scopes", ["unspecified"])[0]] += d["quantity"]
    evidence = {
        "module_input_summaries": {
            "BMS": {"items": len(inputs["BMS"].get("items", [])), "total_qty": bms_total, "expected_scope_qty": inputs["BMS"].get("summary", {}).get("items_by_scope", [])},
            "FOC": {"input_items": len(inputs["FOC"].get("items", [])), "current_bom_rows": len(foc_rows), "native_bom_qty_per_board": foc_native_a, "current_a_total_including_external_mate": quantity_a, "current_b_total_including_external_mate": quantity_b, "bias_rows": len(foc_bias), "supplement_rows": len(foc_supp_rows)},
            "REGULATOR": {"input_items": len(inputs["REGULATOR"].get("items", [])), "current_bom_rows": len(regulator_rows), "current_bom_qty_including_candidate_lines": sum(as_int(r.get("Quantity", 0)) for r in regulator_rows), "candidate_excluded_qty": sum(x["quantity"] for x in candidate_exclusions if x["module"] == MODULES["REGULATOR"]["label"])},
            "PUMP": {"input_items": len(inputs["PUMP"].get("items", [])), "current_bom_rows": len(pump_rows), "current_bom_qty": sum(as_int(r.get("Quantity", 0)) for r in pump_rows)},
        },
        "input_issues": sorted(set(input_issues)),
        "candidate_exclusions": sorted(candidate_exclusions, key=lambda x: (x["module"], x["references"])),
    }
    data = decimal_json({
        "schema_version": 1,
        "currency": "CNY",
        "marketplace": "LCSC China public item pages; current visible RMB price observations only",
        "price_policy": {
            "exact_identity_join": "exact manufacturer part + exact manufacturer + verified supplier part C-code; no CAD carrier substitution",
            "excluded_price_sources": ["historical BOM estimates", "USD list prices", "similar MPN or different manufacturer", "unverified codes", "large-order tiers below actual procurement quantity"],
            "candidate_and_dnp": "not included in current scenario quantities; listed separately",
            "price_scope": "excludes PCB, SMT, shipping; tax status only recorded when explicitly stated by the quote source",
            "full_total_null_rule": "any current SKU without an exact verified current quote or with unresolved required identity/quantity makes the complete scenario total null; verified subtotal remains labeled partial",
            "minimum_cash_rule": "combine same exact SKU across all modules in a scenario before selecting the price tier at the required demand quantity rounded to stated MOQ/order multiple; only move to the next published tier if no tier covers that lot, never borrow a bulk tier for a small order; if MOQ/order multiple missing, report the assumption and leave a fully constrained cash total null",
            "stock_rule": "zero_snapshot/unknown/out_of_stock inventory is review-needed, not confirmed shortage; only positive known quantity with in_stock or in_stock_snapshot below the chosen lot is counted as stock_shortage",
            "scope_rule": "scope quantities are carried from each original source demand before identical SKUs are merged; costs are apportioned by those recorded quantities, not by the merged SKU's scope set",
        },
        "source_bindings": bindings,
        "input_validation": evidence,
        "scenarios": scenarios,
        "supercapacitor_candidate_sensitivity": supercap_sensitivity,
        "large_or_off_board_items_included_once": summarize_large([d for rows in module_demands.values() for d in rows]),
        "current_module_skus": {m: ds for m, ds in module_demands.items()},
        "quote_observation_count": len(price_doc.get("observations", [])),
        "quote_table_unresolved_count": len(price_doc.get("unresolved", [])),
    })
    md = render_markdown(data)
    return data, md


def fmt_money(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"¥{Decimal(str(value)):,.4f}"
    except InvalidOperation:
        return "—"


def render_markdown(data: dict[str, Any]) -> str:
    summaries = data["input_validation"]["module_input_summaries"]
    bms_scopes = {
        x.get("scope"): as_int(x.get("qty_per_bms_set", 0))
        for x in summaries["BMS"].get("expected_scope_qty", [])
    }
    bms_scope_text = " + ".join(
        f"{bms_scopes.get(scope, 0)}{label}"
        for scope, label in (("on_board", "板上"), ("external_accessory", "外配"), ("core_board_module", "核心板"))
    )
    vehicle_modules = data["scenarios"]["actual_vehicle"]["modules"]
    foc_a_qty = vehicle_modules.get("FOC-A", {}).get("demand_quantity", 0)
    foc_b_qty = vehicle_modules.get("FOC-B", {}).get("demand_quantity", 0)
    reg_qty = vehicle_modules.get("REGULATOR", {}).get("demand_quantity", 0)
    pump_qty = vehicle_modules.get("PUMP", {}).get("demand_quantity", 0)
    current_bom_summary = summaries
    lines = [
        "# 当前商城成本",
        "",
        "本页由 `工具/商城成本.py` 从下方列出的原生BOM、模块核价输入和统一商城报价表确定性生成。只将精确MPN、厂家、已验证LCSC C码三者一致且有记录人民币价梯的SKU计入已核价小计。",
        "",
        "**使用边界：稳压核价输入仍绑定原 `原理图准备/BOM.csv`，未覆盖新版 `选型BOM.csv`；BMS原生亦未完成9S5P升级。报价日期以各条记录为准，不代表今天可成交价格。本页不能作为现行整车总价或新版稳压总价。** 本次新旧口径核对见 [电力器件成本复核](../../检查/电力器件成本复核_20261030.json)。",
        "",
        "完整总价在存在未核价SKU、身份/数量不确定或现金约束缺失时保持 `null`。本页不把部分小计称为总价，也不包括PCB、SMT、运费；没有报价明确说明的税费不推算。",
        "库存快照中的 `zero_snapshot`、`unknown`、`out_of_stock` 仅列待复核；只有报价标为在售且有正的已知库存数、数量低于采购批量时才计为确证库存不足。",
        "",
        f"报价表中当前观察数：{data['quote_observation_count']}；本轮中央报价文件SHA绑定见下表。输入生成时间不作为本报告更新时间，重复运行不会仅因时钟变化改文件。",
        "",
        "## 来源绑定",
        "",
        "|来源|SHA-256|",
        "|---|---|",
    ]
    for k, v in data["source_bindings"].items():
        lines.append(f"|{k} (`{v['path']}`)|`{v['sha256']}`|")

    if data["input_validation"]["input_issues"]:
        lines.extend(["", "## 输入校验问题", ""])
        for issue in data["input_validation"]["input_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.extend(["", "## 输入校验", "", "核价输入声明的原生BOM绑定、数量基线与对应源文件检查一致；不代表已覆盖新版选型BOM。"])

    lines.extend(["", "## 场景汇总", ""])
    for key, scenario in data["scenarios"].items():
        lines.extend([
            f"### {scenario['label']}",
            "",
            "|模块|当前需求数量|已核价SKU|已核价实装小计（部分）|未核价SKU / 数量|查询失败SKU / 数量|确证库存不足 / 库存待复核|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for m, result in scenario["modules"].items():
            label = {"BMS": "BMS", "FOC-A": "FOC-A", "FOC-B": "FOC-B", "REGULATOR": "稳压模块", "PUMP": "水泵驱动模块"}.get(m, m)
            lines.append(
                f"|{label}|{result['demand_quantity']}|{result['verified_quoted_sku_count']}/{result['sku_count']}|"
                f"{fmt_money(result['verified_installed_subtotal_cny'])}|{result['unpriced_sku_count']} / {result['unpriced_quantity']}|"
                f"{result['query_failure_sku_count']} / {result['query_failure_quantity']}|"
                f"{result['stock_shortage_sku_count']} / {result['stock_unknown_sku_count']}|"
            )
        lines.extend([
            "",
            "按原始需求scope分摊（每个模块单独核算；同SKU跨scope时按需求数量拆分）：",
            "",
            "|模块|板内：已核价小计 / 未核价SKU·数量|核心板/板级模块：已核价小计 / 未核价SKU·数量|板外附件：已核价小计 / 未核价SKU·数量|",
            "|---|---:|---:|---:|",
        ])
        for m, result in scenario["modules"].items():
            label = {"BMS": "BMS", "FOC-A": "FOC-A", "FOC-B": "FOC-B", "REGULATOR": "稳压模块", "PUMP": "水泵驱动模块"}.get(m, m)
            cells = []
            for bucket in ("on_board", "core_board_module", "external_accessory"):
                b = result["scope_breakdown"][bucket]
                cells.append(f"{fmt_money(b['verified_installed_subtotal_cny'])} / {b['unpriced_sku_count']}·{b['unpriced_quantity']}")
            lines.append(f"|{label}|{cells[0]}|{cells[1]}|{cells[2]}|")
        lines.extend([
            "",
            f"已核价SKU的实装折算小计：**{fmt_money(scenario['verified_installed_subtotal_cny'])}**（部分金额）。",
            f"完整实装总价：**{fmt_money(scenario['complete_installed_total_cny'])}**。",
            f"已核价SKU现金采购估算：**{fmt_money(scenario['known_sku_cash_estimate_assuming_declared_MOQ_and_piece_step_cny'])}**（部分；按已声明MOQ、假设按单只/单件下单；订单步长缺失时不称精确最低现金额）。",
            f"已核价SKU MOQ余料：**{scenario['known_sku_overbuy_quantity']}件**；对应现金估算与已核实装折算差额：**{fmt_money(scenario['known_sku_cash_overbuy_vs_installed_value_cny'])}**。",
            f"完整最低采购现金额：**{fmt_money(scenario['complete_minimum_cash_purchase_cny'])}**。",
            f"当前未核价：{scenario['unpriced_sku_count']} 个SKU，共 {scenario['unpriced_quantity']} 件；确证库存不足SKU {scenario['stock_shortage_sku_count']} 个，库存待复核SKU {scenario['stock_unknown_sku_count']} 个。",
            "",
        ])

        if scenario.get("cash_purchase_risk_examples"):
            lines.extend([
                "首轮大MOQ现金风险示例（仅展示当前合并需求，不据此自动换型）：",
                "",
                "|MPN|需求量|MOQ|需采购量|阶梯单价|预计现金|余料|",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for x in scenario["cash_purchase_risk_examples"]:
                if x.get("status") == "not_present_in_this_scenario":
                    continue
                lines.append(
                    f"|{x.get('manufacturer_part')}|{x.get('quantity_required')}|{x.get('moq') if x.get('moq') is not None else '未知'}|"
                    f"{x.get('purchase_quantity') if x.get('purchase_quantity') is not None else '未知'}|{fmt_money(x.get('unit_price_cny'))}|"
                    f"{fmt_money(x.get('cash_cost_cny'))}|{x.get('overbuy_quantity') if x.get('overbuy_quantity') is not None else '未知'}|"
                )
            lines.extend([""])

    lines.extend([
        "## 当前采购范围与边界",
        "",
        "|模块|单套/当前装配用量|边界说明|",
        "|---|---:|---|",
        f"|BMS|{summaries['BMS']['total_qty']}件（{bms_scope_text}）|核心板身份/SKU未冻结，作为当前BMS采购输入中的指定需求列入且保持未核价；采购补充已包含在BMS BOM与输入，不重复加件。|",
        f"|FOC-A|{foc_a_qty} BOM与板外件|`R_LOAD_VALID_BIAS`仅A安装。板外母线电容、保险、dump电阻与线端配对插头已计数并单独标注。|",
        f"|FOC-B|{foc_b_qty} BOM与板外件|共享1k bias DNP；不把A的1只扩成双装。|",
        f"|稳压模块|当前BOM {current_bom_summary['REGULATOR']['current_bom_rows']}行；明示候选/条件项{current_bom_summary['REGULATOR']['candidate_excluded_qty']}件剔除后，当前采购量{reg_qty}件|当前SC0350-270-RSS为{data['supercapacitor_candidate_sensitivity']['current_bom_quantity']}只；大型BUS_EXT电容与外置dump/保险按BOM计一次。|",
        f"|水泵驱动模块|当前BOM {current_bom_summary['PUMP']['current_bom_rows']}行、当前采购量{pump_qty}件|板端/线端连接器、保险、线束、核心板等按scope分别保留。|",
        "",
        "### 超级电容配置",
        "",
        f"当前BOM记录 {data['supercapacitor_candidate_sensitivity']['current_bom_quantity']} 只，设计目标沿用现有8只，增量0；主要用于承接制动回生，有余电时辅助启动。此前16只方案已撤回。超容已列入数量统计，但无已核实单价时不会计入已核价小计；如重新采购应另行询价。",
        "",
        "## 已验证报价与逐SKU明细",
        "",
        "逐SKU精确MPN、厂家、C码、两个场景合并采购数量、阶梯价、MOQ、步长、库存、现金估算与未核价原因均在 `商城成本.json` 的 `scenarios.*.sku_demands` 和 `current_module_skus` 中。大型/板外件见 `large_or_off_board_items_included_once`；该表仅拆分展示，不再加到总量。",
        "",
    ])
    # List every exact quote currently used; no stale or CAD-carrier price is displayed.
    used = {}
    for scenario in data["scenarios"].values():
        for sku in scenario["sku_demands"]:
            if sku.get("quote_match_status") == "verified_exact_quote":
                used[(sku.get("manufacturer_part"), sku.get("manufacturer"), sku.get("supplier_part"))] = sku
    if used:
        lines.extend(["|MPN|厂家|C码|C码来源|场景需求|采用单价（元/只）|MOQ|步长|库存|报价来源/时间|", "|---|---|---|---|---:|---:|---:|---:|---:|---|"])
        for _, sku in sorted(used.items(), key=lambda kv: (norm_mpn(kv[0][0]), kv[0][2] or "")):
            q = sku["quote"]
            p = sku.get("purchase") or {}
            lines.append(
                f"|{sku.get('manufacturer_part') or '—'}|{sku.get('manufacturer') or '—'}|{sku.get('supplier_part') or '—'}|"
                f"{'已验证报价补映射' if sku.get('supplier_code_verified_from_quote') else '来源记录/索引'}|{sku['quantity_required']}|{fmt_money(p.get('unit_price_cny'))}|{q.get('moq') if q.get('moq') is not None else '未知'}|"
                f"{q.get('order_multiple') if q.get('order_multiple') is not None else '未知'}|{q.get('stock_quantity') if q.get('stock_quantity') is not None else '未知'}|"
                f"[{q.get('quote_id')}]({q.get('source_url')}) / {q.get('captured_at_utc')}|")
    else:
        lines.append("当前没有匹配到任何完整的MPN/厂家/C码三元组报价。")
    lines.extend(["", "## 排除项", ""])
    excluded = data["input_validation"]["candidate_exclusions"]
    if excluded:
        lines.extend(["|模块|位号|MPN|数量|排除原因|", "|---|---|---|---:|---|"])
        for x in excluded:
            lines.append(f"|{x['module']}|{', '.join(x['references'])}|{x.get('manufacturer_part') or '—'}|{x['quantity']}|{x['reason']}|")
    else:
        lines.append("当前输入中未发现明确标为DNP/条件候选的BOM行。")
    lines.extend([
        "",
        "本报告是当前公开价快照下的采购需求与成本覆盖情况，不表示器件电气、机械或整车资格。价格可用性仅代表报价记录采集时刻。",
        "",
    ])
    return "\n".join(lines)


def write_or_check(check: bool) -> int:
    data, md = build_outputs()
    json_text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if check:
        failures = []
        for path, expected in ((OUTPUT_JSON, json_text), (OUTPUT_MD, md)):
            if not path.exists():
                failures.append(f"missing generated output: {path.relative_to(ROOT)}")
            elif path.read_text(encoding="utf-8-sig") != expected:
                failures.append(f"stale generated output: {path.relative_to(ROOT)}")
        if failures:
            for f in failures:
                print(f, file=sys.stderr)
            return 1
        print("PASS: 当前商城成本 JSON/Markdown 与当前BOM、核价输入和商城报价表一致。")
        return 0
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json_text, encoding="utf-8", newline="\n")
    OUTPUT_MD.write_text(md, encoding="utf-8", newline="\n")
    print(f"WROTE {OUTPUT_JSON.relative_to(ROOT)} sha256={sha256(OUTPUT_JSON)}")
    print(f"WROTE {OUTPUT_MD.relative_to(ROOT)} sha256={sha256(OUTPUT_MD)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只读校验生成文件是否与当前输入一致")
    args = parser.parse_args()
    return write_or_check(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
