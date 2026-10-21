#!/usr/bin/env python3
"""Preflight or rebind BMS calculation evidence after local drawing publication.

Default mode is read-only and prints the verified execution plan. ``--apply``
updates only P1_SHA_EXPECTED in the protection audit source, then runs the PSU
evidence rebind, 16 A static-P1 replay, and protection audit in that order.
The script does not call EasyEDA and never changes native CAD or BOM sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "BMS模块"
LAYOUT = MODULE / "检查" / "Layout验证"
P1 = MODULE / "sch" / "Schematic1" / "P1.esch2"
BOM = MODULE / "原理图准备" / "BOM.csv"
SNAPSHOT = LAYOUT / "current-snapshot.json"
PORTS = LAYOUT / "ports-with-cleanup-whitelist.json"
CLEANUP = LAYOUT / "cleanup-manifest.json"
MANIFEST = MODULE / "检查" / "当前施工输入.json"
GEOMETRY = LAYOUT / "final-geometry.json"
REOPEN = LAYOUT / "root-after-cleanup-reopen.json"
CURRENT_PUBLICATION = LAYOUT / "当前发布.json"
READBACK_SCOPE = MODULE / "检查" / "局部连线" / "readback-scope.json"
DRAWING_EVIDENCE = MODULE / "检查" / "局部连线"
DRC_FIX = MODULE / "检查" / "DRC修复"
DRC_CLEANUP_PLAN = DRC_FIX / "重复网标清理分析.json"
DRC_BASELINE_SNAPSHOT = DRC_FIX / "latest-drc-before.json"
DRC_BASELINE_SEMANTIC = DRC_FIX / "latest-drc-before-semantic.json"
PROTECTION_SOURCE = MODULE / "仿真" / "源码" / "verify_protection_reduction.py"
PSU_REBINDER = MODULE / "检查" / "DRC修复" / "rebind_psu_evidence.py"
CHARGE_REPLAY = MODULE / "仿真" / "源码" / "replay_16a_with_static_p1.py"
PROTECTION_AUDIT = PROTECTION_SOURCE

EXPECTED_COMPONENTS = 208
EXPECTED_PINS = 700
EXPECTED_PORTS = 285
EXPECTED_WIRES = 331
EXPECTED_TEXTS = 237
EXPECTED_REMOVED_NET_ATTRIBUTES = 29
EXPECTED_DUPLICATE_NET_WIRE_GROUPS = 11
REQUIRED_COMPARISONS = {"rows", "ports", "wires", "texts"}
NATIVE = [
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"缺少发布后证据：{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(source_map: Any, basename: str, label: str) -> str:
    require(isinstance(source_map, dict), f"{label} 缺少 sourceSHA256")
    matches = [
        str(value).lower()
        for key, value in source_map.items()
        if isinstance(key, str)
        and key.replace("\\", "/").rsplit("/", 1)[-1].casefold() == basename.casefold()
    ]
    require(len(matches) == 1, f"{label} 对 {basename} 的 SHA256 缺失或不唯一")
    require(re.fullmatch(r"[0-9a-f]{64}", matches[0]) is not None,
            f"{label} 中 {basename} 的 SHA256 格式无效")
    return matches[0]


def normalize_drc_kind(value: Any) -> str | None:
    compact = re.sub(r"[\s_-]+", "", str(value).strip().casefold())
    if compact in {"fatal", "fatals", "fatalerror", "fatalerrors"}:
        return "fatal"
    if compact in {"error", "errors", "err"}:
        return "error"
    if compact in {"warning", "warnings", "warn"}:
        return "warning"
    return None


def drc_counts(raw: Any, label: str) -> dict[str, int]:
    counts = {"fatal": 0, "error": 0, "warning": 0}
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = [{"type": name, "count": count} for name, count in raw.items()]
    else:
        raise RuntimeError(f"{label} DRC计数格式无效")
    for row in rows:
        require(isinstance(row, dict), f"{label} DRC包含非对象行")
        kind = normalize_drc_kind(row.get("type", ""))
        if kind is None:
            continue
        value = row.get("count")
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{label} DRC {kind}计数无效")
        counts[kind] += value
    return counts


def validate_identical_copy_scope(
    scope: Any, before: Any, formal_hashes: dict[str, str]
) -> dict[str, str]:
    require(isinstance(scope, dict), "readback-scope.json格式无效")
    require(scope.get("method") == "OFFICIAL_IDENTICAL_COPY_READBACK"
            and scope.get("root_path_readback") is False
            and scope.get("root_path_status") == "HOLD_OPEN_DOCUMENT_TIMEOUT"
            and scope.get("source_hashes_same") is True,
            "同字节副本readback-scope字段与根路径超时状态不一致")
    project_value = scope.get("project")
    require(isinstance(project_value, str) and bool(project_value.strip()),
            "readback-scope缺少隔离副本project路径")
    project = Path(project_value).resolve(strict=True)
    formal_project = (MODULE / NATIVE[0]).resolve(strict=True)
    require(project.name.casefold() == Path(NATIVE[0]).name.casefold()
            and project != formal_project,
            "readback-scope project必须是不同于正式入口的同名隔离工程")
    declared = scope.get("copy_source_sha256")
    require(isinstance(declared, dict) and set(declared) == set(NATIVE),
            "readback-scope必须列出全部3个native SHA256")
    copy_hashes: dict[str, str] = {}
    for rel in NATIVE:
        copy_hash = sha256(project.parent / rel)
        formal_hash = str(formal_hashes.get(rel, "")).lower()
        prior_hash = str(before.get(rel, "")).lower() if isinstance(before, dict) else ""
        declared_hash = str(declared.get(rel, "")).lower()
        require(bool(formal_hash and prior_hash and declared_hash)
                and len({copy_hash, formal_hash, prior_hash, declared_hash}) == 1,
                f"隔离副本/正式源/before-reopen/scope的SHA不一致：{rel}")
        copy_hashes[rel] = copy_hash
    return copy_hashes


def validate_ready() -> dict[str, Any]:
    for path in (
        P1, BOM, SNAPSHOT, PORTS, CLEANUP, MANIFEST, GEOMETRY, REOPEN,
        CURRENT_PUBLICATION, PROTECTION_SOURCE, PSU_REBINDER, CHARGE_REPLAY,
        DRC_CLEANUP_PLAN, DRC_BASELINE_SNAPSHOT, DRC_BASELINE_SEMANTIC,
        DRAWING_EVIDENCE / "saved-snapshot.json",
        DRAWING_EVIDENCE / "reopened-snapshot.json",
        DRAWING_EVIDENCE / "final-semantic.json",
    ):
        require(path.is_file(), f"必需文件不存在：{path}")

    p1_sha = sha256(P1)
    bom_sha = sha256(BOM)
    formal_native_hashes = {rel: sha256(MODULE / rel) for rel in NATIVE}
    geometry = read_json(GEOMETRY)
    reopen = read_json(REOPEN)
    publication = read_json(CURRENT_PUBLICATION)
    readback_scope = read_json(READBACK_SCOPE) if READBACK_SCOPE.is_file() else None
    identical_copy = readback_scope is not None

    require(geometry.get("status") == "PASS"
            and geometry.get("electrical_geometry_status") == "PASS"
            and geometry.get("strict_semantic_geometry") is True,
            "当前final-geometry未通过严格语义几何核验")
    counts = geometry.get("counts", {})
    require(counts.get("components", counts.get("snapshotComponents")) == EXPECTED_COMPONENTS,
            "当前几何证据的元件数不是208")
    require(counts.get("pins", counts.get("pinsChecked")) == EXPECTED_PINS,
            "当前几何证据的针数不是700")

    semantic = geometry.get("construction_input_net_validation", {})
    require(isinstance(semantic, dict) and semantic.get("status") == "PASS"
            and semantic.get("checked_pins") == EXPECTED_PINS,
            "construction-input严格语义验证未通过700针检查")
    summary = geometry.get("drawing_summary")
    require(isinstance(summary, dict)
            and summary.get("user_requirement") == "U18"
            and summary.get("status") == (
                "PASS_ELECTRICAL_IDENTICAL_COPY_READBACK_ROOT_HOLD"
                if identical_copy else "PASS_ELECTRICAL_READBACK"
            )
            and summary.get("drc_warning_cleared") is False,
            "U18保存/重开摘要缺失或错误标记DRC警告状态")
    expected_readback_statement = (
        "同字节副本重开PASS；根路径图页打开超时待用户核对"
        if identical_copy else "正式根路径官方重开PASS"
    )
    require(summary.get("readback_statement") == expected_readback_statement,
            "U18摘要未准确区分副本读回与正式根路径状态")
    verification = summary.get("verification", {})
    require(verification.get("saved_reopened_match") is True
            and verification.get("saved_reopened_match_basis") == (
                "Rows/ports/wires/texts compare by ID; all fields exact except pin x/y absolute tolerance 1e-9."
                " Snapshot file byte identity is reported separately using SHA256."
            )
            and verification.get("strict_semantic_status") == "PASS"
            and (verification.get("components_checked"), verification.get("pins_checked"))
            == (EXPECTED_COMPONENTS, EXPECTED_PINS),
            "U18摘要未证明208位号/700针保存重开语义PASS")
    if identical_copy:
        require(verification.get("readback_method") == "OFFICIAL_IDENTICAL_COPY_READBACK"
                and verification.get("root_path_readback") is False
                and verification.get("root_path_status") == "HOLD_OPEN_DOCUMENT_TIMEOUT"
                and reopen.get("method") == "OFFICIAL_IDENTICAL_COPY_READBACK",
                "U18/重开证据没有明确记录同字节副本及正式根路径HOLD")
    comparison = verification.get("record_comparison")
    require(isinstance(comparison, dict) and set(comparison) == REQUIRED_COMPARISONS
            and all(isinstance(comparison[key], dict)
                    and comparison[key].get("equal") is True
                    for key in REQUIRED_COMPARISONS),
            "U18摘要未证明rows/ports/wires/texts按ID匹配")
    row_comparison = comparison["rows"]
    coordinate_tolerance = row_comparison.get("pin_coordinate_tolerance_absolute")
    maximum_coordinate_difference = row_comparison.get("maximum_pin_coordinate_difference")
    require(row_comparison.get("comparison_mode") == "ALL_FIELDS_EXACT_EXCEPT_PIN_XY_ABSOLUTE_TOLERANCE"
            and isinstance(coordinate_tolerance, (int, float))
            and not isinstance(coordinate_tolerance, bool)
            and coordinate_tolerance == 1e-9
            and isinstance(maximum_coordinate_difference, (int, float))
            and not isinstance(maximum_coordinate_difference, bool)
            and math.isfinite(maximum_coordinate_difference)
            and 0 <= maximum_coordinate_difference <= coordinate_tolerance
            and row_comparison.get("pin_coordinate_errors") == [],
            "U18针脚坐标容差、最大实际偏差或其他字段精确比较证据无效")
    saved_snapshot_sha = sha256(DRAWING_EVIDENCE / "saved-snapshot.json")
    reopened_snapshot_sha = sha256(DRAWING_EVIDENCE / "reopened-snapshot.json")
    snapshot_files_byte_identical = saved_snapshot_sha == reopened_snapshot_sha
    require(verification.get("snapshot_files_byte_identical") is snapshot_files_byte_identical
            and verification.get("saved_snapshot_sha256") == saved_snapshot_sha
            and verification.get("reopened_snapshot_sha256") == reopened_snapshot_sha,
            "U18摘要快照字节相同性或SHA与本地saved/reopened文件不一致")
    locks = verification.get("procurement_nc_bom_pcb_locks")
    require(isinstance(locks, dict) and locks.get("status") == "PASS"
            and locks.get("differences") == [],
            "采购身份/NC/BOM/PCB锁定未通过")

    root_status = reopen.get("status")
    if identical_copy:
        copy_hashes = validate_identical_copy_scope(
            readback_scope, reopen.get("source_sha256_before"), formal_native_hashes
        )
        require(reopen.get("status") in {
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD",
            "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD",
        }, f"同字节副本读回状态不可用于重绑定：{root_status!r}")
        scope_project = str(Path(str(readback_scope["project"])).resolve(strict=True))
        require(reopen.get("project") == scope_project
                and reopen.get("root_path_readback") is False
                and reopen.get("root_path_status") == "HOLD_OPEN_DOCUMENT_TIMEOUT"
                and reopen.get("readback_status") == "PASS_IDENTICAL_COPY_READBACK"
                and reopen.get("source_hashes_same") is True
                and reopen.get("copy_source_sha256") == copy_hashes,
                "同字节副本读回证据与readback-scope/实际native SHA不一致")
    else:
        require(root_status in {
            "PASS_READBACK",
            "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING",
        }, f"根路径重开状态不可用于重绑定：{root_status!r}")
    require(reopen.get("source_unchanged") is True
            and reopen.get("snapshot_match") is True
            and reopen.get("snapshot_match_basis") == (
                "Records match by ID; all fields exact except pin x/y absolute tolerance 1e-9."
            )
            and reopen.get("snapshot_files_byte_identical") is snapshot_files_byte_identical
            and reopen.get("saved_snapshot_sha256") == saved_snapshot_sha
            and reopen.get("reopened_snapshot_sha256") == reopened_snapshot_sha
            and reopen.get("saved_snapshot_sha256") == verification.get("saved_snapshot_sha256")
            and reopen.get("reopened_snapshot_sha256") == verification.get("reopened_snapshot_sha256")
            and reopen.get("maximum_pin_coordinate_difference") == maximum_coordinate_difference
            and reopen.get("pin_coordinate_tolerance_absolute") == coordinate_tolerance
            and reopen.get("snapshot_differences", []) in ([], 0),
            "重开证据缺少源文件不变/坐标容差匹配/快照字节差异记录")
    before = reopen.get("source_sha256_before", {})
    after = reopen.get("source_sha256_after", {})
    require(isinstance(before, dict) and before == after,
            "读回源前后SHA映射不一致")
    require(str(before.get("sch/Schematic1/P1.esch2", "")).lower() == p1_sha,
            "读回证据未绑定当前正式P1 SHA256")
    require(all(str(before.get(rel, "")).lower() == formal_native_hashes[rel]
                for rel in NATIVE),
            "读回证据三份native SHA未绑定正式入口")

    actual_drc = drc_counts(reopen.get("drc"), "根路径重开")
    review = reopen.get("drc_log_review", {})
    require(actual_drc["fatal"] == 0 and actual_drc["error"] == 0,
            "模型重绑定要求根路径重开DRC fatal/error均为0")
    require(isinstance(review, dict)
            and review.get("aggregate_warning_groups") == actual_drc["warning"]
            and review.get("errors") == actual_drc["error"]
            and review.get("fatal_errors") == actual_drc["fatal"],
            "DRC复核计数与根路径重开计数不一致")
    if actual_drc["warning"]:
        expected_warning_status = (
            "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD"
            if identical_copy else "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING"
        )
        require(root_status == expected_warning_status
                and reopen.get("warning_cleared") is False
                and reopen.get("drcReview") == "HOLD"
                and reopen.get("layout_release_status") == "HOLD"
                and reopen.get("warning_details_reviewed") is False
                and review.get("drcReview") == "HOLD"
                and review.get("warning_details_reviewed") is False,
                "存在DRC警告但根重开状态没有明确保留未解决标记")
        require(geometry.get("drcReview") == "HOLD"
                and geometry.get("layout_release_status") == "HOLD"
                and geometry.get("warning_details_reviewed") is False
                and summary.get("drcReview") == "HOLD"
                and summary.get("layout_release_status") == "HOLD"
                and summary.get("warning_details_reviewed") is False,
                "几何证据未保留DRC review/Layout release HOLD")
    else:
        expected_zero_status = (
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD"
            if identical_copy else "PASS_READBACK"
        )
        require(root_status == expected_zero_status
                and reopen.get("drcReview") == "NO_WARNINGS_IN_READBACK"
                and reopen.get("layout_release_status") == "HOLD"
                and reopen.get("warning_details_reviewed") is True
                and review.get("drcReview") == "NO_WARNINGS_IN_READBACK"
                and review.get("warning_details_reviewed") is True,
                "零警告重开缺少明确DRC review或仍标记警告未解决")
        require(geometry.get("drcReview") == "NO_WARNINGS_IN_READBACK"
                and geometry.get("layout_release_status") == "HOLD"
                and geometry.get("warning_details_reviewed") is True
                and summary.get("drcReview") == "NO_WARNINGS_IN_READBACK"
                and summary.get("layout_release_status") == "HOLD"
                and summary.get("warning_details_reviewed") is True,
                "零警告几何/U18摘要缺少明确DRC review或Layout HOLD")

    repair_gate = geometry.get("drc_repair_validation")
    native_cleanup = geometry.get("native_net_attribute_cleanup")
    pin_equivalence = geometry.get("pin_network_equivalence")
    require(isinstance(repair_gate, dict) and repair_gate.get("status") == "PASS",
            "当前几何证据缺少本轮DRC修复PASS门禁")
    require(isinstance(native_cleanup, dict)
            and native_cleanup.get("status") == "PASS"
            and repair_gate.get("native_net_attribute_cleanup") == native_cleanup,
            "native NET ATTR清理证明缺失或不一致")
    require(isinstance(pin_equivalence, dict)
            and pin_equivalence.get("status") == "PASS"
            and repair_gate.get("pin_network_equivalence") == pin_equivalence
            and pin_equivalence.get("checked_pins") == EXPECTED_PINS
            and pin_equivalence.get("baseline_pin_identities") == EXPECTED_PINS
            and pin_equivalence.get("current_pin_identities") == EXPECTED_PINS
            and pin_equivalence.get("all_pin_network_results_unchanged") is True
            and pin_equivalence.get("differences") == [],
            "删除NET ATTR后未证明完整700针网络结果与基线逐针完全一致")

    cleanup_plan = read_json(DRC_CLEANUP_PLAN)
    plan_sources = cleanup_plan.get("sources", {})
    require(isinstance(plan_sources, dict)
            and plan_sources.get("snapshot", {}).get("sha256") == sha256(DRC_BASELINE_SNAPSHOT)
            and plan_sources.get("strictSemanticBaseline", {}).get("sha256") == sha256(DRC_BASELINE_SEMANTIC),
            "NET ATTR清理计划的基线快照/语义SHA与当前文件不一致")
    plan_removed = cleanup_plan.get("removeNetAttributes", [])
    plan_kept = cleanup_plan.get("keepNetAttributes", [])
    require(isinstance(plan_removed, list) and len(plan_removed) == len(set(plan_removed))
            and len(plan_removed) == EXPECTED_REMOVED_NET_ATTRIBUTES
            and isinstance(plan_kept, list) and len(plan_kept) == len(set(plan_kept))
            and len(plan_kept) == EXPECTED_DUPLICATE_NET_WIRE_GROUPS,
            "NET ATTR计划没有唯一的29个删除ID和11个保留ID")
    require(native_cleanup.get("cleanup_plan_sha256") == sha256(DRC_CLEANUP_PLAN)
            and native_cleanup.get("baseline_p1_sha256")
            == str(plan_sources.get("nativeSchematic", {}).get("sha256", "")).lower()
            and native_cleanup.get("current_p1_sha256") == p1_sha,
            "native NET ATTR证明没有绑定计划、归档基线和当前正式P1")
    removed_ids = native_cleanup.get("removed_attribute_ids")
    retained_ids = native_cleanup.get("retained_attribute_ids")
    config_change = native_cleanup.get("config_metadata_change")
    require(isinstance(config_change, dict)
            and config_change.get("record") == "DOCHEAD"
            and config_change.get("changed_fields") == ["client"]
            and isinstance(config_change.get("before_client"), str)
            and bool(config_change.get("before_client"))
            and isinstance(config_change.get("after_client"), str)
            and bool(config_change.get("after_client"))
            and config_change.get("before_client") != config_change.get("after_client")
            and native_cleanup.get("project_sha256_unchanged") is True
            and native_cleanup.get("config_sha256_unchanged") is False,
            "原生配置改动必须严格限于DOCHEAD.client，且工程文件SHA不变")
    require(isinstance(removed_ids, list) and len(removed_ids) == EXPECTED_REMOVED_NET_ATTRIBUTES
            and len(removed_ids) == len(set(removed_ids)) and set(removed_ids) == set(plan_removed)
            and isinstance(retained_ids, list) and len(retained_ids) == EXPECTED_DUPLICATE_NET_WIRE_GROUPS
            and len(retained_ids) == len(set(retained_ids)) and set(retained_ids) == set(plan_kept)
            and native_cleanup.get("removed_attribute_count") == EXPECTED_REMOVED_NET_ATTRIBUTES
            and native_cleanup.get("duplicate_wire_group_count") == EXPECTED_DUPLICATE_NET_WIRE_GROUPS
            and native_cleanup.get("baseline_active_wire_count") == EXPECTED_WIRES
            and native_cleanup.get("current_active_wire_count") == EXPECTED_WIRES
            and native_cleanup.get("other_native_record_changes") == []
            and repair_gate.get("native_record_change_allowlist", {}).get("status") == "PASS"
            and repair_gate.get("native_record_change_allowlist", {}).get("config_metadata_change") == config_change,
            "native diff超出29个NET ATTR删除/11个keep ID及唯一DOCHEAD.client白名单")
    group_proofs = native_cleanup.get("duplicate_wire_groups")
    plan_groups = cleanup_plan.get("groups", [])
    require(isinstance(group_proofs, list) and len(group_proofs) == EXPECTED_DUPLICATE_NET_WIRE_GROUPS
            and isinstance(plan_groups, list) and len(plan_groups) == EXPECTED_DUPLICATE_NET_WIRE_GROUPS,
            "native NET ATTR proof does not include all 11 audited wire groups")
    plan_by_wire = {group.get("nativeWireId"): group for group in plan_groups if isinstance(group, dict)}
    require(len(plan_by_wire) == EXPECTED_DUPLICATE_NET_WIRE_GROUPS,
            "NET ATTR plan has missing/duplicate native wire IDs")
    removed_group_ids = set()
    for group_proof in group_proofs:
        wire_id = group_proof.get("wire_id")
        expected_group = plan_by_wire.get(wire_id)
        require(expected_group is not None
                and group_proof.get("network_name") == expected_group.get("net")
                and group_proof.get("retained_attribute_id") == expected_group.get("keepNetAttributeId"),
                f"native NET ATTR retained ID/network differs from plan for WIRE {wire_id}")
        expected_removed_group = {
            row.get("id") for row in expected_group.get("attributes", [])
            if isinstance(row, dict) and row.get("action") != "keep"
        }
        actual_removed_group = group_proof.get("removed_attribute_ids")
        require(isinstance(actual_removed_group, list)
                and len(actual_removed_group) == len(set(actual_removed_group))
                and set(actual_removed_group) == expected_removed_group,
                f"native NET ATTR removed IDs differ from plan for WIRE {wire_id}")
        removed_group_ids.update(actual_removed_group)
    require(removed_group_ids == set(plan_removed),
            "native NET ATTR per-wire removals do not sum to the exact allowlist")

    snapshot_proof = repair_gate.get("snapshot_record_comparison")
    require(isinstance(snapshot_proof, dict) and snapshot_proof.get("status") == "PASS"
            and snapshot_proof.get("rows_ports_wires_texts_unchanged") is True,
            "保存/重开快照未证明本轮只改NET ATTR原生记录")
    expected_counts = {
        "components": EXPECTED_COMPONENTS,
        "pins": EXPECTED_PINS,
        "ports": EXPECTED_PORTS,
        "wires": EXPECTED_WIRES,
        "texts": EXPECTED_TEXTS,
    }
    snapshot_counts = snapshot_proof.get("counts", {})
    require(isinstance(snapshot_counts, dict)
            and all(snapshot_counts.get(label) == expected_counts for label in ("baseline", "saved", "reopened")),
            "DRC修复前/保存/重开记录的208元件、700针、285端口、331线、237文本计数不一致")
    record_pairs = (
        "baseline_to_saved", "baseline_to_reopened", "saved_to_reopened"
    )
    for pair in record_pairs:
        comparison_pair = snapshot_proof.get(pair)
        require(isinstance(comparison_pair, dict)
                and set(comparison_pair) == REQUIRED_COMPARISONS
                and all(isinstance(comparison_pair[key], dict)
                        and comparison_pair[key].get("equal") is True
                        for key in REQUIRED_COMPARISONS),
                f"{pair}未证明rows/ports/wires/texts全部相同")
    require(repair_gate.get("baseline_snapshot_sha256") == plan_sources["snapshot"]["sha256"]
            and repair_gate.get("baseline_semantic_sha256")
            == plan_sources["strictSemanticBaseline"]["sha256"]
            and repair_gate.get("current_semantic_sha256") == sha256(DRAWING_EVIDENCE / "final-semantic.json")
            and repair_gate.get("saved_snapshot_sha256") == saved_snapshot_sha
            and repair_gate.get("reopened_snapshot_sha256") == reopened_snapshot_sha,
            "DRC修复证明与当前快照/基线语义文件SHA不匹配")
    identity_proof = repair_gate.get("procurement_nc_bom_pcb_identity")
    require(identity_proof == locks and identity_proof.get("status") == "PASS"
            and identity_proof.get("differences") == [],
            "采购MPN/厂家/供应商/身份/NC/BOM/PCB字段未通过本轮修复前后锁定")
    baseline_drc_proof = repair_gate.get("baseline_drc_log_review", {})
    require(baseline_drc_proof.get("scope") == "PRE_REPAIR_BASELINE_ONLY"
            and baseline_drc_proof.get("reviewed_by_user") is True
            and baseline_drc_proof.get("applies_to_current_reopened_snapshot") is False
            and baseline_drc_proof.get("warning_count") == 12,
            "本轮12条用户已审基线DRC日志没有与当前未审DRC分离记录")
    current_drc_gate = repair_gate.get("current_reopened_drc", {})
    require(current_drc_gate.get("aggregate_warning_groups") == actual_drc["warning"]
            and current_drc_gate.get("fatal_errors") == actual_drc["fatal"]
            and current_drc_gate.get("errors") == actual_drc["error"]
            and current_drc_gate.get("warning_details_reviewed") is False
            and current_drc_gate.get("drcReview") == "HOLD"
            and current_drc_gate.get("layout_release_status") == "HOLD",
            "当前DRC汇总或待新日志确认的HOLD语义不一致")
    require(native_cleanup.get("current_p1_sha256") == p1_sha
            and sha256(DRC_CLEANUP_PLAN) == native_cleanup.get("cleanup_plan_sha256"),
            "重绑门禁遇到P1或清理计划更新，需重新发布/校验")

    current_sources = publication.get("sources", {})
    p1_record = current_sources.get("sch/Schematic1/P1.esch2", {})
    bom_record = current_sources.get("原理图准备/BOM.csv", {})
    require(p1_record.get("sha256") == p1_sha,
            "当前发布清单未绑定实际P1")
    require(bom_record.get("sha256") == bom_sha
            and bom_record.get("bytes") == BOM.stat().st_size,
            "BOM与当前发布清单不一致")

    source_map = geometry.get("sourceSHA256")
    for path in (P1, SNAPSHOT, PORTS, CLEANUP, MANIFEST):
        require(source_hash(source_map, path.name, "final-geometry") == sha256(path),
                f"final-geometry源SHA不匹配：{path.name}")

    source_text = PROTECTION_SOURCE.read_text(encoding="utf-8")
    expected_matches = re.findall(
        r"(?m)^P1_SHA_EXPECTED\s*=\s*(['\"])([0-9a-fA-F]{64})\1\s*$",
        source_text,
    )
    require(len(expected_matches) == 1,
            "保护模型源码中P1_SHA_EXPECTED声明缺失或不唯一")
    return {
        "status": (
            (
                "READY_FOR_MODEL_REBIND_FROM_IDENTICAL_COPY_WITH_ROOT_HOLD_AND_DRC_HOLD"
                if actual_drc["warning"]
                else "READY_FOR_MODEL_REBIND_FROM_IDENTICAL_COPY_WITH_ROOT_HOLD"
            )
            if identical_copy else (
                "READY_FOR_MODEL_REBIND_WITH_DRC_HOLD"
                if actual_drc["warning"]
                else "PASS_READY_FOR_MODEL_REBIND"
            )
        ),
        "p1_sha256": p1_sha,
        "bom_sha256": bom_sha,
        "readback_status": reopen.get("readback_status", root_status),
        "readback_method": reopen.get("method", "OFFICIAL_ROOT_PATH_READBACK"),
        "readback_project": reopen.get("project"),
        "root_path_readback": reopen.get("root_path_readback", True),
        "root_path_status": reopen.get("root_path_status", root_status),
        "source_hashes_same": reopen.get("source_hashes_same", True),
        "drc": actual_drc,
        "drc_repair_validation": repair_gate,
        "native_net_attribute_cleanup": native_cleanup,
        "pin_network_equivalence": pin_equivalence,
        "procurement_nc_bom_pcb_identity": identity_proof,
        "drcReview": reopen.get("drcReview"),
        "layout_release_status": reopen.get("layout_release_status", "HOLD"),
        "warning_details_reviewed": reopen.get("warning_details_reviewed", False),
        "warning_cleared": reopen.get("warning_cleared", False),
        "drawing_summary": (
            f"U18; 208 components/700 pins; rows/ports/wires/texts match by ID; all fields exact except pin x/y abs tolerance 1e-9 (max delta={maximum_coordinate_difference:.17g}); snapshot files byte-identical={snapshot_files_byte_identical} (verified by SHA256); locked fields PASS; "
            "same-byte copy reopen PASS; formal root page open timed out, user verification pending"
            if identical_copy else
            f"U18; 208 components/700 pins; rows/ports/wires/texts match by ID; all fields exact except pin x/y abs tolerance 1e-9 (max delta={maximum_coordinate_difference:.17g}); snapshot files byte-identical={snapshot_files_byte_identical} (verified by SHA256); locked fields PASS; official root reopen PASS"
        ),
        "protection_source_expected_p1_sha256": expected_matches[0][1],
        "stages": [
            "Update only verify_protection_reduction.py:P1_SHA_EXPECTED",
            "Run rebind_psu_evidence.py",
            "Run replay_16a_with_static_p1.py",
            "Run verify_protection_reduction.py",
        ],
    }


def atomic_replace(path: Path, text: str) -> None:
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def update_protection_p1_hash(p1_sha: str) -> None:
    raw = PROTECTION_SOURCE.read_bytes()
    text = raw.decode("utf-8")
    pattern = re.compile(
        r"(?m)^(P1_SHA_EXPECTED\s*=\s*['\"])[0-9a-fA-F]{64}(['\"]\s*)$"
    )
    updated, replacements = pattern.subn(lambda match: match.group(1) + p1_sha + match.group(2), text)
    require(replacements == 1,
            f"保护模型源码P1_SHA_EXPECTED替换次数应为1，实际{replacements}")
    atomic_replace(PROTECTION_SOURCE, updated)


def run_stage(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", str(path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "script": str(path.relative_to(ROOT)),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="更新保护模型P1 SHA并执行3项模型重绑定/复算；默认仅做只读预检",
    )
    args = parser.parse_args()

    plan = validate_ready()
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("只读预检完成；尚未修改模型源码或模型结果。需要执行时使用 --apply。")
        return 0

    protected_hashes = {
        path: sha256(path)
        for path in (P1, BOM, SNAPSHOT, PORTS, CLEANUP, MANIFEST, GEOMETRY, REOPEN, CURRENT_PUBLICATION)
    }
    update_protection_p1_hash(plan["p1_sha256"])
    stages = [
        run_stage(PSU_REBINDER),
        run_stage(CHARGE_REPLAY),
        run_stage(PROTECTION_AUDIT),
    ]
    changed = [
        str(path.relative_to(ROOT))
        for path, before_sha in protected_hashes.items()
        if sha256(path) != before_sha
    ]
    require(not changed, f"模型重绑定期间非目标源/布局证据发生变化：{changed}")
    final_status = (
        "MODEL_REBIND_AND_REPLAY_COMPLETE_DRC_HOLD"
        if plan["drc"]["warning"]
        else "PASS_MODEL_REBIND_AND_REPLAY"
    )
    print(json.dumps({
        **plan,
        "status": final_status,
        "stages": stages,
        "protected_native_and_layout_inputs_unchanged": True,
        "warning_cleared": False if plan["drc"]["warning"] else plan["warning_cleared"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
