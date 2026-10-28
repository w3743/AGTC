#!/usr/bin/env python3
"""Refresh mechanical source bindings for the four active power schematics.

Read-only by default. Pass --review-report MODULE=PATH for each final module
review and --write only after those reports have been finalized. This script
does not run library synchronization, edit native EDA files, or recalculate prices.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


POWER = Path(__file__).resolve().parents[2]
WORKSPACE = POWER.parent
COVERAGE_PATH = POWER / "检查" / "统一审查" / "全部原理图检查覆盖.json"
COST_PATH = POWER / "元件封装库" / "检查" / "四板复用与成本核对.json"
SUPPLEMENT_PATH = POWER / "元件封装库" / "整车器件补充.json"

MODULES: list[dict[str, Any]] = [
    {
        "module": "BMS模块",
        "project": "BMS模块/BMS-9S4P.eprj3",
        "page": "BMS模块/sch/Schematic1/P1.esch2",
        "sheet_id": "350a4904c4ec87b3",
    },
    {
        "module": "FOC驱动模块",
        "project": "FOC驱动模块/eda工程文件/FOC_G431_Driver/FOC_G431_Driver.eprj3",
        "page": "FOC驱动模块/eda工程文件/FOC_G431_Driver/sch/Schematic1/P1.esch2",
        "sheet_id": "90aa6c46493624cf",
    },
    {
        "module": "稳压模块",
        "project": "稳压模块/eda工程文件/sc-regulator/sc-regulator.eprj3",
        "page": "稳压模块/eda工程文件/sc-regulator/sch/sc-regulator/01 稳压模块全部电路.esch2",
        "sheet_id": "9546dca0bc8bce7f",
    },
    {
        "module": "水泵驱动模块",
        "project": "水泵驱动模块/eda工程文件/水泵驱动/水泵驱动.eprj3",
        "page": "水泵驱动模块/eda工程文件/水泵驱动/sch/Schematic1/P1.esch2",
        "sheet_id": "bafe8a48e4625018",
    },
]

BOM_SOURCES: list[dict[str, Any]] = [
    {"module": "BMS模块", "path": "BMS模块/原理图准备/BOM.csv", "board_count": 1},
    {"module": "FOC驱动模块", "path": "FOC驱动模块/eda工程文件/导出/当前交付/FOC_G431_BOM.csv", "board_count": 2},
    {"module": "FOC驱动模块", "path": "FOC驱动模块/原理图准备/装配采购补充.csv", "board_count": 2},
    {"module": "稳压模块", "path": "稳压模块/原理图准备/BOM.csv", "board_count": 1},
    {"module": "水泵驱动模块", "path": "水泵驱动模块/原理图准备/BOM.csv", "board_count": 1},
]

DEFAULT_REVIEWS = {
    "BMS模块": None,
    "FOC驱动模块": "FOC驱动模块/检查/当前原理图证据.json",
    "稳压模块": "稳压模块/检查/全功能与逐脚复核.json",
    "水泵驱动模块": "水泵驱动模块/检查/原理图全功能与逐脚复核.json",
}


def rel_path(path: Path) -> str:
    return path.resolve().relative_to(POWER.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    os.replace(temp, path)


def parse_bom(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"BOM缺少表头：{path}")
        field_by_fold = {name.strip().casefold(): name for name in reader.fieldnames if name}
        ref_field = next(
            (field_by_fold[name] for name in ("designator", "reference", "位号") if name in field_by_fold),
            None,
        )
        if ref_field is None:
            raise ValueError(f"找不到位号列：{path}; headers={reader.fieldnames}")
        rows = list(reader)
    references: list[str] = []
    for row in rows:
        raw = (row.get(ref_field) or "").strip()
        references.extend(token.strip() for token in re.split(r"[,;]", raw) if token.strip())
    counts = Counter(references)
    return {
        "bom_rows": len(rows),
        "bom_reference_occurrences": len(references),
        "unique_bom_references": len(counts),
        "duplicate_references": sorted(ref for ref, count in counts.items() if count > 1),
        "bom_sha256": sha256(path),
        "delimiter": dialect.delimiter,
        "reference_column": ref_field,
    }


def normalized_index_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("../"):
        normalized = normalized[3:]
    if normalized.startswith("电力/"):
        normalized = normalized[len("电力/"):]
    return normalized.casefold()


def matching_source_entry(entries: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    wanted = normalized_index_path(path)
    matches = [
        entry for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        and normalized_index_path(entry["path"]) == wanted
    ]
    if len(matches) > 1:
        raise ValueError(f"中央索引对来源路径有重复记录：{path}")
    return matches[0] if matches else None


def rebuild_incomplete_identity(index: dict[str, Any], procurement: dict[str, Any]) -> dict[str, Any]:
    records_by_native: dict[str, list[dict[str, Any]]] = {}
    for record in procurement.get("records", []):
        for key in record.get("native_keys", []) or []:
            records_by_native.setdefault(key, []).append(record)
    missing = []
    for instance in index.get("instances", []):
        mpn = instance.get("mpn")
        manufacturer = instance.get("manufacturer")
        missing_fields = []
        if mpn is None or not str(mpn).strip():
            missing_fields.append("mpn")
        if manufacturer is None or not str(manufacturer).strip():
            missing_fields.append("manufacturer")
        if not missing_fields:
            continue
        key = instance.get("key") or f"{instance.get('module')}::{instance.get('designator') or instance.get('ref')}"
        candidates = []
        for record in records_by_native.get(key, []):
            candidates.append({
                "mpn": record.get("mpn_source"),
                "manufacturer": record.get("manufacturer_source"),
                "source": record.get("source"),
                "row": record.get("source_row"),
                "notes": record.get("notes_source"),
            })
        missing.append({
            "module": instance.get("module"),
            "designator": instance.get("designator") or instance.get("ref"),
            "mpn": mpn,
            "manufacturer": manufacturer,
            "pcb": instance.get("pcb"),
            "missing_fields": missing_fields,
            "reason": "当前中央索引中的原生有效属性为空；不自动推断厂家或替代采购目标。",
            "procurement_candidates": candidates,
        })
    summary = index.get("summary", {})
    return {
        "counts": {
            "missing_mpn_count": summary.get("missing_mpn_count"),
            "missing_manufacturer_count": summary.get("missing_manufacturer_count"),
            "actual_rows_rebuilt": len(missing),
        },
        "instances": missing,
        "scope": "由当前元件总表.instances直接重建；空字段仍是采购身份缺口，不代表结构一致性检查失败。",
    }


def rebuild_board_scope(index: dict[str, Any], procurement: dict[str, Any], fresh_boms: list[dict[str, Any]]) -> dict[str, Any]:
    module_stats = {entry.get("module"): entry for entry in index.get("module_stats", [])}
    modules = [entry["module"] for entry in MODULES]
    board_counts = {entry["module"]: 1 for entry in MODULES}
    board_counts["FOC驱动模块"] = 2
    scope = {}
    for module in modules:
        stat = module_stats.get(module, {})
        module_instances = [entry for entry in index.get("instances", []) if entry.get("module") == module]
        pcb_counts = Counter((entry.get("pcb") or "unknown").lower() for entry in module_instances)
        module_procurement = [entry for entry in procurement.get("records", []) if entry.get("module") == module]
        procurement_only = sum(1 for entry in module_procurement if entry.get("relation") == "BOM独有条目，未在原生BOM中登记")
        scope[module] = {
            "vehicle_multiplicity": board_counts[module],
            "native_component_primitives": stat.get("component_count"),
            "native_bom_instances": stat.get("bom_instance_count"),
            "excluded_non_bom_instances": stat.get("excluded_non_bom_count"),
            "pcb_counts": {
                "yes": pcb_counts.get("yes", 0),
                "no": pcb_counts.get("no", 0),
                "unknown": pcb_counts.get("unknown", 0),
            },
            "procurement_records": len(module_procurement),
            "procurement_only_records": procurement_only,
            "procurement_sources": [entry for entry in fresh_boms if entry["module"] == module],
        }
    return scope


def report_page_binding(report: Any, page_rel: str) -> str | None:
    """Return an explicitly recorded SHA for the page, if the report exposes one."""
    normalized_page = page_rel.replace("\\", "/")
    candidates = {normalized_page.casefold(), ("电力/" + normalized_page).casefold()}
    if "/" in normalized_page:
        candidates.add(normalized_page.split("/", 1)[1].casefold())

    def walk(value: Any, parent_key: str = ""):
        if isinstance(value, dict):
            path_value = (
                value.get("schematic") or value.get("path") or value.get("sourcePath")
                or value.get("nativePath") or value.get("source")
            )
            hash_value = (value.get("schematic_sha256") or value.get("sha256")
                          or value.get("sourceSha256") or value.get("nativeSha256"))
            if isinstance(path_value, str) and isinstance(hash_value, str):
                normalized = path_value.replace("\\", "/").casefold()
                if any(normalized.endswith(candidate) for candidate in candidates):
                    yield hash_value.lower()
            page_sha = value.get("page_sha256") or value.get("p1_sha256")
            if isinstance(page_sha, str) and parent_key.casefold() in {
                "native_source", "current_native_source", "source_binding", "schematic_source"
            }:
                yield page_sha.lower()
            for key, child in value.items():
                yield from walk(child, key)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child, parent_key)

    # Final reports retain before/after proofs as historical evidence. Prefer
    # explicit current-source sections; still reject conflicts between them.
    current_sections = (
        "sourceFiles", "sourceBinding", "source_binding", "native_source",
        "current_native_source", "schematic_source", "source",
    )
    hashes = set()
    if isinstance(report, dict):
        for key in current_sections:
            if isinstance(report.get(key), (dict, list)):
                hashes.update(walk(report[key], key))
    if not hashes:
        hashes = set(walk(report))
    if len(hashes) > 1:
        raise ValueError(f"报告对同一P1记录多个不同SHA：{page_rel}: {sorted(hashes)}")
    return next(iter(hashes), None)


def review_record(module: str, report_path: Path, page_rel: str, page_sha: str) -> dict[str, Any]:
    data = read_json(report_path)
    if not isinstance(data, dict):
        raise ValueError(f"审查报告顶层必须是JSON对象：{report_path}")
    recorded_sha = report_page_binding(data, page_rel)
    binding_status = "NOT_EXPOSED_IN_REPORT"
    if recorded_sha is not None:
        binding_status = "MATCH" if recorded_sha == page_sha.lower() else "MISMATCH"
    summary = data.get("summary")
    if isinstance(summary, dict):
        summary_out = summary
    elif isinstance(summary, (str, int, float, bool)) or summary is None:
        summary_out = summary
    else:
        summary_out = str(summary)
    return {
        "report_path": rel_path(report_path),
        "report_sha256": sha256(report_path),
        "report_title": data.get("title"),
        "review_date": data.get("review_date") or data.get("asOf") or data.get("date"),
        "review_complete": data.get("reviewComplete", data.get("review_complete")),
        "design_release": data.get("designRelease") or data.get("design_release"),
        "status": data.get("status"),
        "summary": summary_out,
        "page_binding_status": binding_status,
        "report_recorded_page_sha256": recorded_sha,
        "full_schematic_review_present": isinstance(data.get("full_schematic_review"), dict),
        "review_findings_count": len(data.get("findings", [])) if isinstance(data.get("findings"), list) else None,
    }


def load_review_args(items: list[str]) -> dict[str, str]:
    reviews = dict(DEFAULT_REVIEWS)
    for value in items:
        if "=" not in value:
            raise ValueError(f"--review-report格式应为 模块=相对电力目录路径：{value}")
        module, path = value.split("=", 1)
        if module not in {item["module"] for item in MODULES}:
            raise ValueError(f"未知模块：{module}")
        reviews[module] = path.replace("\\", "/")
    return reviews


def build_coverage(reviews: dict[str, str]) -> dict[str, Any]:
    entries = []
    missing_reviews = []
    mismatches = []
    for item in MODULES:
        project_path = POWER / item["project"]
        page_path = POWER / item["page"]
        for source in (project_path, page_path):
            if not source.is_file():
                raise FileNotFoundError(source)
        page_sha = sha256(page_path)
        review_rel = reviews.get(item["module"])
        review = None
        if not review_rel:
            missing_reviews.append(item["module"])
        else:
            review_path = POWER / review_rel
            if not review_path.is_file():
                missing_reviews.append(item["module"])
            else:
                review = review_record(item["module"], review_path, item["page"], page_sha)
                if review["page_binding_status"] == "MISMATCH":
                    mismatches.append(item["module"])
        entries.append({
            "module": item["module"],
            "project_path": item["project"],
            "project_sha256": sha256(project_path),
            "page_path": item["page"],
            "page_sha256": page_sha,
            "sheet_id": item["sheet_id"],
            "review": review,
        })
    return {
        "schema_version": 1,
        "title": "四个当前电力原理图审查覆盖",
        "as_of": date.today().isoformat(),
        "scope": "四个当前EPRJ3工程及各自唯一现行原理图页；不含检查备份、归档、开源参考和仿真示例。",
        "module_count": len(entries),
        "active_project_count": len(entries),
        "current_page_count": len(entries),
        "review_evidence_complete": not missing_reviews,
        "all_review_page_bindings_match": not missing_reviews and all(
            entry["review"] is not None and entry["review"]["page_binding_status"] == "MATCH"
            for entry in entries
        ),
        "overall_release": "本覆盖索引不授予设计放行；各模块按其审查结论与未闭合事项执行。",
        "missing_or_unresolved_review_modules": missing_reviews,
        "source_binding_mismatches": mismatches,
        "modules": entries,
    }


def refresh_cost(coverage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cost = read_json(COST_PATH)
    if not isinstance(cost, dict):
        raise ValueError("成本JSON顶层必须是对象")
    supplement = read_json(SUPPLEMENT_PATH)
    project_constraints = supplement.get("project_constraints", {})
    sc_constraints = project_constraints.get("regulator_supercapacitor", {})
    if not isinstance(sc_constraints, dict) or sc_constraints.get("fixed_manufacturer_part") != "SC0350-270-RSS":
        raise ValueError("整车器件补充缺少已确认的LICAP超容约束，拒绝生成成本快照")
    if sc_constraints.get("total_count_max") != 8 or sc_constraints.get("native_current_count") != 8:
        raise ValueError("整车器件补充超容数量与当前用户固定8颗约束不符")
    optimization = cost.setdefault("global_optimization_rule", {})
    supercapacitor = optimization.setdefault("supercapacitor", {})
    supercapacitor.update({
        "required_mpn": sc_constraints["fixed_manufacturer_part"],
        "max_cells": sc_constraints["total_count_max"],
        "current_regulator_bom_cells": sc_constraints["native_current_count"],
        "candidate_total_count": sc_constraints.get("candidate_total_count"),
        "candidate_added_count": sc_constraints.get("candidate_added_count"),
        "candidate_is_current": sc_constraints.get("candidate_is_current"),
        "candidate_status": sc_constraints.get("candidate_status"),
        "historical_cancelled_candidate_total_count": sc_constraints.get("historical_cancelled_candidate_total_count"),
        "historical_cancelled_candidate_added_count": sc_constraints.get("historical_cancelled_candidate_added_count"),
        "quantity_rule": sc_constraints.get("quantity_rule"),
        "use_case": sc_constraints.get("use_case"),
        "sustained_1s_requirement": sc_constraints.get("sustained_1s_requirement"),
        "duration_s": sc_constraints.get("duration_s"),
        "duration_status": sc_constraints.get("duration_status"),
        "reference_power_w": sc_constraints.get("reference_power_w"),
        "screened_model_candidates": {
            "8S2P_low_voltage_window_800W_bidirectional": "cancelled_obsolete_historical",
            "1000W_directional": "cancelled_obsolete_historical",
            "active_candidate": False,
            "transfer_to_current_bom_quantity": False,
            "note": "用户最新决定固定现有8颗，仅作启动瞬态功率辅助；16颗扩容及其模型候选已取消，不进入当前用量/成本。历史仿真结果仍留在原仿真位置。",
        },
    })
    index_path = POWER / "元件封装库" / "索引" / "元件总表.json"
    procurement_path = POWER / "元件封装库" / "索引" / "采购补充.json"
    source_config_path = POWER / "元件封装库" / "来源配置.json"
    sync_report_path = POWER / "检查" / "统一审查" / "全库同步与只读校验.json"
    sync_evidence_path = POWER / "元件封装库" / "检查" / "同步证据.json"
    index = read_json(index_path)
    procurement = read_json(procurement_path)
    sync_report = read_json(sync_report_path) if sync_report_path.is_file() else {}
    sync_evidence = read_json(sync_evidence_path) if sync_evidence_path.is_file() else {}
    index_sha = sha256(index_path)
    procurement_sha = sha256(procurement_path)
    source_config_sha = sha256(source_config_path)

    previous_snapshot = cost.get("snapshot", {})
    old_boms = {entry.get("bom_path"): entry for entry in previous_snapshot.get("module_boms", [])}
    fresh_boms = []
    local_bom_sources = []
    for source in BOM_SOURCES:
        path = POWER / source["path"]
        stats = parse_bom(path)
        old = old_boms.get(source["path"], {})
        indexed_source = matching_source_entry(procurement.get("sources", []), source["path"])
        indexed_sha = indexed_source.get("sha256") if indexed_source else None
        in_sync_index = indexed_sha == stats["bom_sha256"]
        local_bom_sources.append({
            "module": source["module"],
            "path": source["path"],
            "current_sha256": stats["bom_sha256"],
            "procurement_index_sha256": indexed_sha,
            "rows": stats["bom_rows"],
            "matches_procurement_index": in_sync_index,
        })
        if source["path"].endswith("原理图准备/装配采购补充.csv"):
            quantity_basis = "单板来源Quantity=1；FOC车辆两板，整车共2只板外配对插头；不得再乘其它倍数。"
        else:
            quantity_basis = old.get("quantity_basis", "按单板来源记录；车辆用量按模块板数乘算，板外/DNP应依来源说明区分。")
        record = {
            "module": source["module"],
            "vehicle_board_multiplicity": source["board_count"],
            "bom_path": source["path"],
            "format": old.get("format", "procurement_source"),
            **{key: stats[key] for key in ("bom_rows", "bom_reference_occurrences", "unique_bom_references", "bom_sha256")},
            "vehicle_reference_occurrences": stats["bom_reference_occurrences"] * source["board_count"],
            "quantity_basis": quantity_basis,
            "bom_basis": old.get("bom_basis", "现行来源；不代表现价、采购或实物资格"),
            "matches_current_procurement_index": in_sync_index,
        }
        record["duplicate_references"] = stats["duplicate_references"]
        fresh_boms.append(record)

    index_sources = index.get("sources", [])
    current_pages = []
    page_differences = []
    for item in MODULES:
        page_path = POWER / item["page"]
        page_sha = sha256(page_path)
        indexed_source = next((entry for entry in index_sources if entry.get("module") == item["module"]), None)
        indexed_sha = indexed_source.get("sha256") if indexed_source else None
        matches_index = indexed_sha == page_sha
        if not matches_index:
            page_differences.append(item["module"])
        current_pages.append({
            "module": item["module"],
            "path": item["page"],
            "sha256": page_sha,
            "index_sha256": indexed_sha,
            "matches": matches_index,
            "index_status": "MATCHES_CURRENT_CENTRAL_INDEX" if matches_index else "STALE_PENDING_CENTRAL_SYNC",
        })

    procurement_differences = [entry["path"] for entry in local_bom_sources if not entry["matches_procurement_index"]]
    sync_evidence_sources = sync_evidence.get("sources", []) if isinstance(sync_evidence, dict) else []
    sync_source_checks = []
    for item in MODULES:
        current_page_sha = sha256(POWER / item["page"])
        source_entry = matching_source_entry(sync_evidence_sources, item["page"])
        sync_source_checks.append(bool(
            source_entry
            and source_entry.get("sha256") == current_page_sha
            and source_entry.get("sha256_after_parse") == current_page_sha
            and source_entry.get("sha256_after_generation") == current_page_sha
            and source_entry.get("unchanged") is True
        ))
    required_runs = {"最终同步", "最终只读校验"}
    run_records = sync_report.get("executedRuns", []) if isinstance(sync_report, dict) else []
    run_success = {
        entry.get("label") for entry in run_records
        if isinstance(entry, dict) and entry.get("exitCode") == 0
    }
    check_report_matches = (
        normalized_index_path(str(sync_report.get("indexPath", "")))
        == normalized_index_path("元件封装库/索引/元件总表.json")
        and sync_report.get("indexSha256") == index_sha
        and normalized_index_path(str(sync_report.get("syncEvidencePath", "")))
        == normalized_index_path("元件封装库/检查/同步证据.json")
        and sync_report.get("syncEvidenceSha256") == (
            sha256(sync_evidence_path) if sync_evidence_path.is_file() else None
        )
    )
    sync_evidence_matches = (
        sync_evidence.get("config_sha256") == source_config_sha
        and len(sync_source_checks) == len(MODULES)
        and all(sync_source_checks)
    )
    sync_proof_matches = bool(
        check_report_matches
        and sync_evidence_matches
        and required_runs.issubset(run_success)
        and str(sync_report.get("verification", sync_report.get("status", ""))).startswith("PASS")
    )
    index_sources_match = not page_differences and not procurement_differences
    central_status = (
        "CURRENT_INDEX_AND_SYNC_PROOF_MATCH_CURRENT_FILES"
        if sync_proof_matches and index_sources_match
        else "INDEX_SOURCE_HASHES_STALE_VS_CURRENT_FILES"
        if not index_sources_match
        else "INDEX_FILES_MATCH_BUT_SYNC_PROOF_STALE_OR_UNVERIFIED"
    )

    freshness = []
    for source in current_pages:
        freshness.append({
            "module": source["module"],
            "path": source["path"],
            "index_sha256": source["index_sha256"],
            "current_sha256": source["sha256"],
            "matches": source["matches"],
            "index_status": source["index_status"],
        })

    board_scope = rebuild_board_scope(index, procurement, fresh_boms)
    cost["board_scope"] = board_scope
    cost["board_scope_snapshot_status"] = central_status
    cost["incomplete_native_procurement_identity"] = rebuild_incomplete_identity(index, procurement)
    cost["exact_shared_model_groups_from_index_snapshot"] = [
        {
            "mpn": part.get("mpn"),
            "manufacturer": part.get("manufacturer"),
            "modules": part.get("modules", []),
            "indexed_instance_count": part.get("bom_instance_count"),
            "device_asset_variants": len(part.get("device_ids", [])),
            "footprint_asset_variants": len(part.get("footprint_ids", [])),
            "carrier_mpn_diff_instances": sum(
                1 for instance in index.get("instances", [])
                if instance.get("part_key") == part.get("key") and instance.get("mpn_diff")
            ),
        }
        for part in index.get("parts", [])
        if part.get("shared_across_boards")
    ]
    cost["same_mpn_manufacturer_mismatch_groups_from_index_snapshot"] = index.get(
        "mpn_same_cross_board_manufacturer_mismatch_candidates", []
    )
    cost["carrier_identity"] = {
        "mpn_diff_instances": index.get("summary", {}).get("mpn_diff_count"),
        "rule": "采购目标与CAD载体分开；不据载体下单或自动合并。",
    }
    cost["snapshot"]["central_native_index"] = {
        "path": "元件封装库/索引/元件总表.json",
        "sha256": index_sha,
        "generated_at_utc": index.get("generated_at_utc"),
        "summary": index.get("summary", {}),
    }
    cost["snapshot"]["vehicle_supplement"] = {
        "path": "元件封装库/整车器件补充.json",
        "sha256": sha256(SUPPLEMENT_PATH),
        "project_constraints": project_constraints,
    }
    cost["snapshot"]["source_configuration"] = {
        "path": "元件封装库/来源配置.json",
        "sha256": source_config_sha,
    }
    cost["snapshot"]["procurement_index"] = {
        "path": "元件封装库/索引/采购补充.json",
        "sha256": procurement_sha,
        "generated_at": procurement.get("generated_at"),
        "summary": procurement.get("summary", {}),
    }
    cost["snapshot"]["final_library_check"] = {
        "path": "检查/统一审查/全库同步与只读校验.json",
        "sha256": sha256(sync_report_path) if sync_report_path.is_file() else None,
        "status": sync_report.get("verification", sync_report.get("status")) if sync_proof_matches else "STALE_OR_UNVERIFIED",
        "reported_status": sync_report.get("verification", sync_report.get("status")) if isinstance(sync_report, dict) else None,
        "matches_current_index_artifacts": sync_proof_matches,
    }
    numeric = {entry["module"]: entry for entry in cost.get("cost_boundary", {}).get("numeric_records_by_module", [])}
    for module, values in board_scope.items():
        prior = numeric.get(module, {})
        numeric[module] = {
            **prior,
            "module": module,
            "procurement_record_count": values["procurement_records"],
        }
    cost["cost_boundary"]["numeric_records_by_module"] = [numeric[module] for module in board_scope]

    cost["generated_date"] = date.today().isoformat()
    cost["status"] = (
        "CURRENT_INDEX_AND_SYNC_PROOF_MATCH_CURRENT_FILES_ELECTRICAL_AND_FULL_COST_OPEN"
        if sync_proof_matches else "LOCAL_SOURCE_SNAPSHOT_INDEX_OR_SYNC_PROOF_STALE_ELECTRICAL_AND_FULL_COST_OPEN"
    )
    cost["snapshot"]["module_boms"] = fresh_boms
    cost["snapshot"]["native_schematic_hash_freshness"] = freshness
    cost["snapshot"]["local_source_refresh"] = {
        "as_of": date.today().isoformat(),
        "coverage_path": "检查/统一审查/全部原理图检查覆盖.json",
        "coverage_sha256": hashlib.sha256((json.dumps(coverage, ensure_ascii=False, indent=2) + "\n").encode("utf-8")).hexdigest(),
        "central_sync_run_by_this_script": False,
        "central_index_status": central_status,
        "index_path": "元件封装库/索引/元件总表.json",
        "index_sha256": index_sha,
        "procurement_index_path": "元件封装库/索引/采购补充.json",
        "procurement_index_sha256": procurement_sha,
        "sync_proof_sha256": sha256(sync_report_path) if sync_report_path.is_file() else None,
        "pages_different_from_current_index": page_differences,
        "bom_sources_different_from_current_procurement_index": procurement_differences,
        "sync_proof_matches_current_index_artifacts": sync_proof_matches,
        "current_index_summary": index.get("summary", {}),
        "prices_recalculated": False,
        "note": "本次仅读取当前源及真实中央索引/同步证明重算路径、SHA、BOM和索引派生统计；未运行中央同步或刷新价格。",
    }
    cost["snapshot"]["final_library_check_status_note"] = (
        "final_library_check仅在同步证明索引SHA、采购索引SHA、来源配置SHA与现存文件一致且源SHA相符时视为当前PASS；本脚本不运行同步或--check。"
    )
    warning = "本次机械快照与当前中央索引/同步证明绑定；索引缺失、源SHA不匹配或同步证明过期时明确标为STALE；价格未刷新。"
    native_instance_count = index.get("summary", {}).get("bom_instance_count")
    native_module_count = index.get("summary", {}).get("module_count")
    cost["snapshot"]["note"] = (
        f"{native_module_count}类当前原生源、{len(fresh_boms)}个采购CSV来源；"
        f"{native_instance_count}是按各模块单板/单套口径统计的原生BOM实例总数，非全车实装量。"
        "采购行关联不重复加入原生实例；FOC装配采购补充及整车数量口径分别见snapshot.module_boms与board_scope。 "
        f"{warning}"
    )
    cost.setdefault("nonclaims", [])
    statement = "本轮文件哈希/行数机械快照不代表中央元件库同步、价格核验或设计放行。"
    if statement not in cost["nonclaims"]:
        cost["nonclaims"].append(statement)
    return cost, {
        "coverage_modules": len(coverage["modules"]),
        "review_evidence_complete": coverage["review_evidence_complete"],
        "source_binding_mismatches": coverage["source_binding_mismatches"],
        "local_bom_sources": len(fresh_boms),
        "bom_sources_different_from_current_procurement_index": procurement_differences,
        "pages_different_from_current_index": page_differences,
        "central_index_status": central_status,
        "sync_proof_matches_current_index_artifacts": sync_proof_matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-report", action="append", default=[], metavar="MODULE=PATH",
                        help="Final module audit JSON; repeat once for BMS/FOC and to override defaults.")
    parser.add_argument("--write", action="store_true", help="Write coverage and mechanical cost snapshot; default is preview only.")
    args = parser.parse_args()
    reviews = load_review_args(args.review_report)
    coverage = build_coverage(reviews)
    cost, summary = refresh_cost(coverage)
    if args.write:
        if not coverage["review_evidence_complete"]:
            raise ValueError(f"缺少最终模块审查JSON，拒绝落盘：{coverage['missing_or_unresolved_review_modules']}")
        nonmatching = [
            entry["module"] for entry in coverage["modules"]
            if not entry["review"] or entry["review"]["page_binding_status"] != "MATCH"
        ]
        if nonmatching:
            raise ValueError(f"每份报告均须明确绑定当前P1 SHA并匹配；缺失或不匹配：{nonmatching}")
        write_json_atomic(COVERAGE_PATH, coverage)
        write_json_atomic(COST_PATH, cost)
    print(json.dumps({
        "write_mode": args.write,
        "coverage_output": rel_path(COVERAGE_PATH),
        "cost_output": rel_path(COST_PATH),
        "missing_or_unresolved_review_modules": coverage["missing_or_unresolved_review_modules"],
        "source_binding_mismatches": coverage["source_binding_mismatches"],
        "all_review_page_bindings_match": coverage["all_review_page_bindings_match"],
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
