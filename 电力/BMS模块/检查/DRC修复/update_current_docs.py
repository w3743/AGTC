#!/usr/bin/env python3
"""Refresh current BMS documentation from the latest readback evidence.

By default this script prints a diff only. Pass --apply to write the seven
allowlisted Markdown files. It reads the current publication, snapshot,
readback, geometry, and model reports. It never edits CAD, ZIP, simulation
source/results, or JSON evidence.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "BMS模块"
EVIDENCE_DIR = MODULE / "检查" / "Layout验证"
DRAWING_EVIDENCE_DIR = MODULE / "检查" / "局部连线"
CONCLUSION_PATH = MODULE / "检查" / "DRC修复结论.md"
PSU_DOC_PATH = MODULE / "检查" / "PSU支路闭合.md"

INPUTS = {
    "publication": EVIDENCE_DIR / "当前发布.json",
    "root_reopen": EVIDENCE_DIR / "root-after-cleanup-reopen.json",
    "snapshot": EVIDENCE_DIR / "current-snapshot.json",
    "geometry": EVIDENCE_DIR / "final-geometry.json",
    "charge_result": MODULE / "仿真" / "结果" / "16A净充电上限验证_result.json",
    "charge_check": MODULE / "检查" / "16A净充电上限验证.json",
    "protection": MODULE / "仿真" / "结果" / "保护精简验证.json",
    "precharge": MODULE / "仿真" / "结果" / "预充Layout闭合.json",
    "readback_scope": MODULE / "检查" / "局部连线" / "readback-scope.json",
}
NATIVE_PATHS = (
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
)

def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少证据文件：{path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"证据文件顶层应为JSON对象：{path}")
    return value


def number(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"证据字段 {label} 缺失或不是数字：{value!r}")
    return int(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_p1_sha(geometry: dict[str, Any]) -> str:
    source_shas = geometry.get("sourceSHA256", {})
    if not isinstance(source_shas, dict):
        raise ValueError("final-geometry.json 缺少 sourceSHA256 映射")
    matches = [
        str(value)
        for path, value in source_shas.items()
        if str(path).replace("\\", "/").endswith("/P1.esch2")
    ]
    if len(matches) != 1:
        raise ValueError(f"final-geometry.json 中P1.esch2指纹应唯一，实际 {len(matches)} 项")
    return matches[0]


def snapshot_sha(geometry: dict[str, Any]) -> str:
    source_shas = geometry.get("sourceSHA256", {})
    matches = [
        str(value)
        for path, value in source_shas.items()
        if str(path).replace("\\", "/").endswith("/current-snapshot.json")
    ]
    if len(matches) != 1:
        raise ValueError(f"final-geometry.json 中current-snapshot指纹应唯一，实际 {len(matches)} 项")
    return matches[0]


def validate_drawing_summary(
    geometry: dict[str, Any], *, identical_copy: bool = False
) -> dict[str, Any]:
    summary = geometry.get("drawing_summary")
    if not isinstance(summary, dict) or summary.get("user_requirement") != "U18":
        raise ValueError("final-geometry.json 缺少本轮U18 drawing_summary")
    expected_status = (
        "PASS_ELECTRICAL_IDENTICAL_COPY_READBACK_ROOT_HOLD"
        if identical_copy else "PASS_ELECTRICAL_READBACK"
    )
    if summary.get("status") != expected_status:
        raise ValueError("本轮U18 drawing_summary未通过电气读回核验")
    expected_statement = (
        "同字节副本重开PASS；根路径图页打开超时待用户核对"
        if identical_copy else "正式根路径官方重开PASS"
    )
    if summary.get("readback_statement") != expected_statement:
        raise ValueError("U18摘要未准确区分副本重开与正式根路径状态")
    if summary.get("drcReview") not in {"HOLD", "NO_WARNINGS_IN_READBACK"}:
        raise ValueError("U18 drawing_summary缺少独立DRC review状态")
    if summary.get("layout_release_status") != "HOLD":
        raise ValueError("U18绘图读回不能被当作Layout正式放行")
    if summary.get("drcReview") == "HOLD" and summary.get("warning_details_reviewed") is not False:
        raise ValueError("U18未审查警告工作态必须标记warning_details_reviewed=false")
    if summary.get("drcReview") == "NO_WARNINGS_IN_READBACK" and summary.get("warning_details_reviewed") is not True:
        raise ValueError("U18零警告读回必须记录warning_details_reviewed=true")
    verification = summary.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("U18 drawing_summary 缺少 verification")
    if verification.get("saved_reopened_match") is not True:
        raise ValueError("U18 drawing_summary未证明保存/重开记录匹配")
    if verification.get("saved_reopened_match_basis") != (
        "Rows/ports/wires/texts compare by ID; all fields exact except pin x/y absolute tolerance 1e-9."
        " Snapshot file byte identity is reported separately using SHA256."
    ):
        raise ValueError("U18快照比较必须明确针脚坐标容差并将字节身份单独按SHA报告")
    if identical_copy and (
        verification.get("readback_method") != "OFFICIAL_IDENTICAL_COPY_READBACK"
        or verification.get("root_path_readback") is not False
        or verification.get("root_path_status") != "HOLD_OPEN_DOCUMENT_TIMEOUT"
    ):
        raise ValueError("U18摘要没有明确记录同字节副本读回和正式根路径HOLD")
    if verification.get("strict_semantic_status") != "PASS":
        raise ValueError("U18 drawing_summary严格语义状态不是PASS")
    if (verification.get("components_checked"), verification.get("pins_checked")) != (208, 700):
        raise ValueError("U18 drawing_summary应核验208位号/700针")
    comparisons = verification.get("record_comparison")
    expected_groups = {"rows", "ports", "wires", "texts"}
    if not isinstance(comparisons, dict) or set(comparisons) != expected_groups:
        raise ValueError("U18 drawing_summary缺少rows/ports/wires/texts完整比较")
    if any(not isinstance(item, dict) or item.get("equal") is not True for item in comparisons.values()):
        raise ValueError("U18 drawing_summary保存/重开记录未按ID匹配")
    rows = comparisons["rows"]
    tolerance = rows.get("pin_coordinate_tolerance_absolute")
    maximum_difference = rows.get("maximum_pin_coordinate_difference")
    if (
        rows.get("comparison_mode") != "ALL_FIELDS_EXACT_EXCEPT_PIN_XY_ABSOLUTE_TOLERANCE"
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance != 1e-9
        or isinstance(maximum_difference, bool)
        or not isinstance(maximum_difference, (int, float))
        or not math.isfinite(maximum_difference)
        or maximum_difference < 0
        or maximum_difference > tolerance
        or rows.get("pin_coordinate_errors") != []
    ):
        raise ValueError("U18针脚坐标绝对容差、最大实际偏差或其他字段严格比较无效")
    saved_snapshot_sha = file_sha256(DRAWING_EVIDENCE_DIR / "saved-snapshot.json")
    reopened_snapshot_sha = file_sha256(DRAWING_EVIDENCE_DIR / "reopened-snapshot.json")
    snapshot_files_byte_identical = saved_snapshot_sha == reopened_snapshot_sha
    if (
        verification.get("snapshot_files_byte_identical") is not snapshot_files_byte_identical
        or verification.get("saved_snapshot_sha256") != saved_snapshot_sha
        or verification.get("reopened_snapshot_sha256") != reopened_snapshot_sha
    ):
        raise ValueError("U18快照字节相同性标记或SHA与本地saved/reopened文件不一致")
    locks = verification.get("procurement_nc_bom_pcb_locks")
    if not isinstance(locks, dict) or locks.get("status") != "PASS" or locks.get("differences") != []:
        raise ValueError("U18 drawing_summary采购身份/NC/BOM/PCB锁定未通过")
    if summary.get("drc_warning_cleared") is not False:
        raise ValueError("U18 drawing_summary不得把当前DRC警告标记为已清除")
    return summary


def validate_readback_scope(
    scope: dict[str, Any] | None, root_reopen: dict[str, Any]
) -> dict[str, Any]:
    if scope is None:
        return {
            "identical_copy": False,
            "method": root_reopen.get("method", "OFFICIAL_ROOT_PATH_READBACK"),
            "project": root_reopen.get("project"),
            "root_path_readback": True,
            "root_path_status": root_reopen.get("status", "UNKNOWN"),
            "statement": "正式根路径官方重开PASS",
        }
    expected_fields = {
        "method": "OFFICIAL_IDENTICAL_COPY_READBACK",
        "root_path_readback": False,
        "root_path_status": "HOLD_OPEN_DOCUMENT_TIMEOUT",
        "source_hashes_same": True,
    }
    for key, expected in expected_fields.items():
        actual = scope.get(key)
        matches = type(actual) is bool and actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            raise ValueError(f"readback-scope.json {key} 不符合预期：{actual!r}")
    project_value = scope.get("project")
    if not isinstance(project_value, str) or not project_value.strip():
        raise ValueError("readback-scope.json缺少隔离副本project路径")
    project = Path(project_value).resolve(strict=True)
    formal_project = (MODULE / NATIVE_PATHS[0]).resolve(strict=True)
    if project.name.casefold() != Path(NATIVE_PATHS[0]).name.casefold() or project == formal_project:
        raise ValueError("readback-scope project必须是不同于正式入口的同名隔离工程")
    declared = scope.get("copy_source_sha256")
    before = root_reopen.get("source_sha256_before")
    after = root_reopen.get("source_sha256_after")
    if not isinstance(declared, dict) or set(declared) != set(NATIVE_PATHS):
        raise ValueError("readback-scope未列出全部3个native SHA256")
    if not isinstance(before, dict) or not isinstance(after, dict) or before != after:
        raise ValueError("副本读回证据缺少一致的源前后SHA映射")
    saved_snapshot_sha = file_sha256(DRAWING_EVIDENCE_DIR / "saved-snapshot.json")
    reopened_snapshot_sha = file_sha256(DRAWING_EVIDENCE_DIR / "reopened-snapshot.json")
    snapshot_files_byte_identical = saved_snapshot_sha == reopened_snapshot_sha
    actual_copy_hashes: dict[str, str] = {}
    for rel in NATIVE_PATHS:
        copy_sha = file_sha256(project.parent / rel)
        formal_sha = file_sha256(MODULE / rel)
        prior_sha = str(before.get(rel, "")).lower()
        after_sha = str(after.get(rel, "")).lower()
        scope_sha = str(declared.get(rel, "")).lower()
        if len({copy_sha, formal_sha, prior_sha, after_sha, scope_sha}) != 1:
            raise ValueError(f"隔离副本/正式源/readback/scope的native SHA不一致：{rel}")
        actual_copy_hashes[rel] = copy_sha

    if (
        root_reopen.get("method") != expected_fields["method"]
        or root_reopen.get("readback_status") != "PASS_IDENTICAL_COPY_READBACK"
        or root_reopen.get("project") != str(project)
        or root_reopen.get("root_path_readback") is not False
        or root_reopen.get("root_path_status") != expected_fields["root_path_status"]
        or root_reopen.get("source_hashes_same") is not True
        or root_reopen.get("copy_source_sha256") != actual_copy_hashes
    ):
        raise ValueError("root-after-cleanup-reopen.json与readback-scope或实际native SHA不一致")
    if (
        root_reopen.get("snapshot_match") is not True
        or root_reopen.get("snapshot_match_basis") != (
            "Records match by ID; all fields exact except pin x/y absolute tolerance 1e-9."
        )
        or root_reopen.get("snapshot_files_byte_identical") is not snapshot_files_byte_identical
        or root_reopen.get("saved_snapshot_sha256") != saved_snapshot_sha
        or root_reopen.get("reopened_snapshot_sha256") != reopened_snapshot_sha
    ):
        raise ValueError("root-after-cleanup-reopen.json未准确记录坐标容差匹配及快照文件SHA身份")
    if root_reopen.get("status") not in {
        "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD",
        "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD",
    }:
        raise ValueError("副本重开状态缺失正式根路径HOLD标记")
    return {
        "identical_copy": True,
        "method": expected_fields["method"],
        "project": str(project),
        "root_path_readback": False,
        "root_path_status": expected_fields["root_path_status"],
        "statement": "同字节副本重开PASS；根路径图页打开超时待用户核对",
        "copy_source_sha256": actual_copy_hashes,
    }


def normalize_drc_kind(value: Any) -> str | None:
    compact = re.sub(r"[\s_-]+", "", str(value).strip().casefold())
    if compact in {"fatal", "fatals", "fatalerror", "fatalerrors"}:
        return "fatal"
    if compact in {"error", "errors", "err"}:
        return "error"
    if compact in {"warning", "warnings", "warn"}:
        return "warning"
    return None


def collect_drc(root_reopen: dict[str, Any], publication: dict[str, Any]) -> dict[str, int]:
    """Read the current root readback DRC counts, with publication fallback."""
    raw = root_reopen.get("drc")
    if raw is None:
        raw = publication.get("known_root_path_runtime", {}).get("drc")

    counts = {"fatal": 0, "error": 0, "warning": 0}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = str(item.get("type", ""))
            normalized = normalize_drc_kind(label)
            if normalized:
                counts[normalized] += number(item.get("count", 0), f"drc.{label}.count")
    elif isinstance(raw, dict):
        for name, value in raw.items():
            normalized = normalize_drc_kind(name)
            if normalized:
                counts[normalized] += number(value, f"drc.{name}")
    else:
        raise ValueError("当前根路径重开JSON中没有可读取的DRC计数")
    return counts


def collect_state() -> dict[str, Any]:
    publication = load_json(INPUTS["publication"])
    root_reopen = load_json(INPUTS["root_reopen"])
    scope = load_json(INPUTS["readback_scope"]) if INPUTS["readback_scope"].is_file() else None
    readback = validate_readback_scope(scope, root_reopen)
    identical_copy = readback["identical_copy"]
    snapshot = load_json(INPUTS["snapshot"])
    geometry = load_json(INPUTS["geometry"])
    charge_result = load_json(INPUTS["charge_result"])
    charge_check = load_json(INPUTS["charge_check"])
    protection = load_json(INPUTS["protection"])
    precharge = load_json(INPUTS["precharge"])

    published_counts = publication.get("counts", {})
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("current-snapshot.json 缺少非空rows数组")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("current-snapshot.json rows包含非对象项")

    refs = [str(row.get("ref", "")).strip() for row in rows]
    if any(not ref for ref in refs) or len(refs) != len(set(refs)):
        raise ValueError("current-snapshot.json 位号为空或重复")
    board_rows = [row for row in rows if row.get("pcb") is True]
    placed_rows = [row for row in rows if row.get("pcb") is True and row.get("bom") is True]
    pin_count = sum(
        len(row.get("pins", []))
        for row in rows
        if isinstance(row.get("pins", []), list)
    )
    raw_mpns = {
        str(row.get("mpn", "")).strip()
        for row in placed_rows
        if str(row.get("mpn", "")).strip()
    }
    normalized_mpns = {
        "SN74LVC1G11DCKR" if mpn == "TI SN74LVC1G11DCKR" else mpn
        for mpn in raw_mpns
    }

    components = number(root_reopen.get("components"), "root_reopen.components")
    pins = number(root_reopen.get("pins"), "root_reopen.pins")
    board_components = number(published_counts.get("pcb_components"), "counts.pcb_components")
    cad_components = number(published_counts.get("cad_components"), "counts.cad_components")
    published_pins = number(published_counts.get("pins"), "counts.pins")
    if len(rows) != cad_components or components != cad_components:
        raise ValueError(
            f"snapshot/root/publication CAD位号不一致：snapshot={len(rows)}，"
            f"root={components}，publication={cad_components}。"
        )
    if board_components != len(board_rows):
        raise ValueError(
            f"发布板上元件数 {board_components} 与snapshot PCB载体数 {len(board_rows)} 不一致"
        )
    if len(placed_rows) != board_components:
        raise ValueError("snapshot中BOM/PCB双标记的板上元件数与发布清单不一致")
    if pins != pin_count or published_pins != pin_count:
        raise ValueError(
            f"root/publication/snapshot针数不一致：root={pins}，publication={published_pins}，"
            f"snapshot={pin_count}。"
        )
    geometry_counts = geometry.get("counts", {})
    geometry_pins = number(geometry_counts.get("pins"), "geometry.counts.pins")
    if pins != geometry_pins:
        raise ValueError("final-geometry.json 与根路径重开针数不一致，拒绝发布矛盾计数")
    drawing_summary = validate_drawing_summary(geometry, identical_copy=identical_copy)
    rows_comparison = drawing_summary["verification"]["record_comparison"]["rows"]
    if (
        root_reopen.get("maximum_pin_coordinate_difference")
        != rows_comparison.get("maximum_pin_coordinate_difference")
        or root_reopen.get("pin_coordinate_tolerance_absolute")
        != rows_comparison.get("pin_coordinate_tolerance_absolute")
        or root_reopen.get("saved_snapshot_sha256")
        != drawing_summary["verification"].get("saved_snapshot_sha256")
        or root_reopen.get("reopened_snapshot_sha256")
        != drawing_summary["verification"].get("reopened_snapshot_sha256")
    ):
        raise ValueError("重开证据与drawing_summary针脚坐标最大偏差或快照SHA不一致")
    if number(published_counts.get("raw_mpn_strings"), "counts.raw_mpn_strings") != len(raw_mpns):
        raise ValueError("发布清单的原始采购MPN数与current-snapshot板上器件不一致")
    if number(published_counts.get("normalized_mpn"), "counts.normalized_mpn") != len(normalized_mpns):
        raise ValueError("发布清单的归一化采购MPN数与current-snapshot板上器件不一致")
    if number(published_counts.get("netports"), "counts.netports") != number(
        geometry_counts.get("labels"), "geometry.counts.labels"
    ):
        raise ValueError("发布网标数与final-geometry网标数不一致")

    p1_sha = str(
        publication.get("sources", {})
        .get("sch/Schematic1/P1.esch2", {})
        .get("sha256", "")
    )
    geometry_p1_sha = expected_p1_sha(geometry)
    root_before_sha = root_reopen.get("source_sha256_before", {}).get("sch/Schematic1/P1.esch2")
    root_after_sha = root_reopen.get("source_sha256_after", {}).get("sch/Schematic1/P1.esch2")
    if not p1_sha or {p1_sha, geometry_p1_sha, root_before_sha, root_after_sha} != {p1_sha}:
        raise ValueError("publication、根路径重开与final-geometry的P1 SHA不一致")
    if file_sha256(INPUTS["snapshot"]) != snapshot_sha(geometry):
        raise ValueError("current-snapshot.json SHA与final-geometry引用不一致")
    accepted_readback_statuses = (
        {
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD",
            "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD",
        }
        if identical_copy else {
            "PASS_READBACK",
            "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING",
        }
    )
    if root_reopen.get("status") not in accepted_readback_statuses or root_reopen.get("snapshot_match") is not True:
        raise ValueError("当前读回尚未通过快照一致性检查")
    if geometry.get("status") != "PASS" or geometry.get("strict_semantic_geometry") is not True:
        raise ValueError("final-geometry没有严格语义几何PASS，不更新当前验证文档")

    pin_results = geometry.get("pinResults", [])
    pin_by_ref = {(str(pin.get("ref")), str(pin.get("pin"))): pin for pin in pin_results}
    required_precharge_pins = {
        ("R_PRE", "1"): "QPRE_SW",
        ("R_PRE", "2"): "PACK+",
        ("Q_PRE2", "2"): "QPRE_SW",
    }
    for key, expected_net in required_precharge_pins.items():
        result = pin_by_ref.get(key)
        if (
            result is None
            or result.get("status") != "PASS"
            or result.get("expected") != expected_net
            or expected_net not in result.get("actualNames", [])
        ):
            raise ValueError(f"預充器件针语义未通过：{key[0]}.{key[1]} 应连接 {expected_net}")
    rpre_rows = [row for row in rows if row.get("ref") == "R_PRE"]
    if len(rpre_rows) != 1:
        raise ValueError("current-snapshot中未找到唯一R_PRE载体")
    rpre = rpre_rows[0]

    actual_lines = number(geometry_counts.get("activeNativeLines"), "geometry.counts.activeNativeLines")
    wire_count = number(geometry_counts.get("wires"), "geometry.counts.wires")
    wire_groups = number(geometry_counts.get("segments"), "geometry.counts.segments")
    if wire_count != wire_groups or actual_lines < wire_groups:
        raise ValueError(
            "几何证据应分别记录活动原生LINE和wire group；wire/segment组数须相等且LINE数不得少于组数："
            f"LINE={actual_lines}, wires={wire_count}, groups={wire_groups}"
        )

    geom_status = str(geometry.get("status", "UNKNOWN")).upper()
    drc = collect_drc(root_reopen, publication)
    drc_review = root_reopen.get("drc_log_review")
    if not isinstance(drc_review, dict):
        raise ValueError("根路径重开缺少本轮DRC计数/审查状态对象")
    if (
        drc_review.get("source") != "reopened-snapshot.json"
        or drc_review.get("scope") != "CURRENT_REOPENED_SNAPSHOT_COUNTS_ONLY"
    ):
        raise ValueError("当前DRC review必须来自本轮重开快照计数，不能套用历史DRC日志")
    warning_details = (
        drc_review.get("warning", {})
        if isinstance(drc_review, dict) else {}
    )
    reviewed_count = number(
        drc_review.get("aggregate_warning_groups"),
        "root_reopen.drc_log_review.aggregate_warning_groups",
    )
    if reviewed_count != drc["warning"]:
        raise ValueError(
            "本轮DRC计数与官方根重开计数不一致："
            f"review={reviewed_count}，root={drc['warning']}"
        )
    warning_reviewed = root_reopen.get("warning_details_reviewed")
    if warning_reviewed is None:
        warning_reviewed = False
    drc_review_state = root_reopen.get("drcReview", drc_review.get("drcReview"))
    if drc_review.get("drcReview") != drc_review_state:
        raise ValueError("根路径重开与DRC review对象的drcReview状态不一致")
    layout_release_status = root_reopen.get("layout_release_status")
    if layout_release_status != "HOLD":
        raise ValueError("当前绘图工作态必须保留layout_release_status=HOLD")
    if drc["fatal"] or drc["error"]:
        raise ValueError("当前DRC包含fatal/error，不能刷新当前验证文档")
    if drc["warning"]:
        if drc_review_state != "HOLD":
            raise ValueError("当前仍有DRC警告时必须标记drcReview=HOLD")
        if warning_reviewed is not False or drc_review.get("warning_details_reviewed") is not False:
            raise ValueError("未审查警告工作态必须显式标记warning_details_reviewed=false")
        warning = drc_review.get("warning", {})
        if warning.get("reference_count") is not None or warning.get("references") != []:
            raise ValueError("仅有重开快照计数时不得从历史日志带入警告位号明细")
    elif drc_review_state != "NO_WARNINGS_IN_READBACK":
        raise ValueError("零警告重开必须明确记录DRC review结果")
    if drawing_summary.get("drcReview") != drc_review_state:
        raise ValueError("U18摘要与根重开DRC review状态不一致")

    prior_independent = publication.get("official_independent_copy_reopen", {})
    prior_independent_warnings = prior_independent.get("warnings")

    charge_result_sha = file_sha256(INPUTS["charge_result"])
    if charge_check.get("result_sha256") != charge_result_sha:
        raise ValueError("16A检查报告引用的结果SHA与当前结果文件不匹配")
    charge_checks = charge_check.get("checks", {})
    if not isinstance(charge_checks, dict) or not charge_checks:
        raise ValueError("16A检查报告缺少检查结果映射")
    charge_faults = charge_check.get("fault_injection_results", {})
    if not isinstance(charge_faults, dict):
        raise ValueError("16A检查报告fault_injection_results格式错误")
    closed_faults = sum(
        1
        for fault in charge_faults.values()
        if isinstance(fault, dict)
        and fault.get("status") == "FAIL_CLOSED_CCL_ZERO"
        and fault.get("command_CCL_A") == 0
    )
    charge_scenarios = number(
        charge_result.get("bus_energy", {}).get("scenario_count"),
        "charge_result.bus_energy.scenario_count",
    )
    protection_checks = protection.get("checks", [])
    if not isinstance(protection_checks, list):
        raise ValueError("保护模型检查列表格式错误")
    protection_passed = sum(
        1 for item in protection_checks if isinstance(item, dict) and item.get("status") == "PASS"
    )
    protection_open_items = protection.get("open_items", [])
    if not isinstance(protection_open_items, list):
        raise ValueError("保护模型open_items格式错误")
    precharge_timer = precharge.get("timer", {})
    precharge_qpre = precharge.get("qpre_mosfets", {})
    precharge_limits = precharge.get("evidence_and_limits", {})

    return {
        "date": str(publication.get("updated", "日期待核")),
        "board_components": board_components,
        "cad_components": components,
        "offboard_components": max(0, components - board_components),
        "pins": pins,
        "raw_mpn": len(raw_mpns),
        "normalized_mpn": len(normalized_mpns),
        "netports": number(
            published_counts.get("netports", geometry.get("counts", {}).get("labels")),
            "counts.netports/counts.labels",
        ),
        "p1_sha": p1_sha,
        "geometry_status": geom_status,
        "geometry": geometry,
        "actual_lines": actual_lines,
        "wire_groups": wire_groups,
        "root_status": str(root_reopen.get("status", "UNKNOWN")),
        "identical_copy": identical_copy,
        "readback": readback,
        "readback_statement": readback["statement"],
        "readback_project": readback.get("project"),
        "snapshot_match": bool(root_reopen.get("snapshot_match", False)),
        "snapshot_files_byte_identical": root_reopen.get("snapshot_files_byte_identical"),
        "snapshot_sha256": file_sha256(INPUTS["snapshot"]),
        "geometry_sha256": file_sha256(INPUTS["geometry"]),
        "reopen_sha256": file_sha256(INPUTS["root_reopen"]),
        "historical_independent_reopen_sha256": file_sha256(
            EVIDENCE_DIR / "final-independent-reopen.json"
        ),
        "geometry_defects_zero": all(
            value == 0 for value in geometry.get("defectCounts", {}).values()
        ),
        "maximum_pin_coordinate_difference": geometry["drawing_summary"]["verification"][
            "record_comparison"]["rows"]["maximum_pin_coordinate_difference"
        ],
        "pin_coordinate_tolerance_absolute": geometry["drawing_summary"]["verification"][
            "record_comparison"]["rows"]["pin_coordinate_tolerance_absolute"
        ],
        "drc": drc,
        "warning_details_reviewed": bool(warning_reviewed),
        "drc_review_state": str(drc_review_state),
        "layout_release_status": str(layout_release_status),
        "warning_reference_count": warning_details.get("reference_count"),
        "warning_classification": warning_details.get("classification"),
        "drawing_summary": drawing_summary,
        "historical_independent_warnings": prior_independent_warnings,
        "rpre": rpre,
        "rpre_semantic_closed": True,
        "charge_result": charge_result,
        "charge_check": charge_check,
        "charge_check_count": len(charge_checks),
        "charge_check_passed": sum(1 for result in charge_checks.values() if result is True),
        "charge_fault_count": len(charge_faults),
        "charge_faults_closed": closed_faults,
        "charge_model_status": str(charge_result.get("model_status", "UNKNOWN")),
        "charge_release_status": str(charge_result.get("release_status", "UNKNOWN")),
        "charge_scenarios": charge_scenarios,
        "protection": protection,
        "protection_status": str(protection.get("status", "UNKNOWN")),
        "protection_check_count": len(protection_checks),
        "protection_check_passed": protection_passed,
        "protection_open_count": len(protection_open_items),
        "precharge_timer": precharge_timer,
        "precharge_qpre": precharge_qpre,
        "precharge_full_chain_max": precharge_limits.get("full_chain_physical_max_time_s"),
        "publication": publication,
        "root_reopen": root_reopen,
    }


def drc_text(state: dict[str, Any]) -> str:
    drc = state["drc"]
    return f"{drc['fatal']}致命/{drc['error']}错误/{drc['warning']}警告"


def warning_text(state: dict[str, Any]) -> str:
    warning_count = state["drc"]["warning"]
    readback_label = "同字节副本重开" if state.get("identical_copy") else "正式根路径重开"
    if warning_count:
        reference_count = state.get("warning_reference_count")
        if state["warning_details_reviewed"]:
            suffix = (
                f"，涉及{reference_count}个位号"
                if isinstance(reference_count, int) and not isinstance(reference_count, bool)
                else ""
            )
            return (
                f"当前仍有{warning_count}条DRC警告{suffix}；明细已审阅但未清除，"
                "drcReview=HOLD，Layout正式放行=HOLD"
            )
        return (
            f"当前{readback_label}计数为{warning_count}条DRC警告；规则/位号明细尚未审阅，"
            "drcReview=HOLD，Layout正式放行=HOLD"
        )
    return f"当前{readback_label}未报告DRC警告；Layout正式放行仍为HOLD"


def drawing_summary_text(state: dict[str, Any]) -> str:
    summary = state["drawing_summary"]
    verification = summary["verification"]
    groups = verification["record_comparison"]
    row_comparison = groups["rows"]
    max_pin_delta = row_comparison["maximum_pin_coordinate_difference"]
    tolerance = row_comparison["pin_coordinate_tolerance_absolute"]
    group_text = "/".join(
        f"{name}按ID匹配"
        for name in ("rows", "ports", "wires", "texts")
        if groups[name]["equal"] is True
    )
    reopen_source = "同字节隔离副本官方保存/重开" if state.get("identical_copy") else "正式根路径官方保存/重开"
    return (
        f"本轮按{summary['user_requirement']}完成{summary['scope']}读回核验："
        f"{reopen_source}{group_text}；除针脚x/y外字段精确，针脚x/y绝对容差{tolerance:g}，"
        f"最大实际偏差{max_pin_delta:.17g}；快照文件字节相同={state['snapshot_files_byte_identical']}。"
        f"208位号/700针严格语义PASS，"
        "采购身份、NC、BOM及PCB锁定PASS；"
        f"{state['readback_statement']}；drcReview={summary['drcReview']}，"
        f"Layout正式放行={summary['layout_release_status']}。"
    )


def geometry_evidence(state: dict[str, Any]) -> str:
    geometry = state["geometry"]
    counts = geometry.get("counts", {})
    failures = sum(
        len(geometry.get(key, []))
        for key in ("nonOrthogonal", "missingPins", "foreignNetAtPins", "foreignNetAtWireEndpoints", "danglingLabels")
    )
    return (
        f"原理图连线几何 {state['geometry_status']}；{state['pins']}针、"
        f"{state['actual_lines']}条活动原生LINE、{state['wire_groups']}个wire group、"
        f"{counts.get('labels', state['netports'])}个网标，"
        f"列出的几何异常项 {failures} 项。该检查不替代ERC/DRC、PCB布局或实物验证。"
    )


def model_evidence(state: dict[str, Any]) -> str:
    timer = state["precharge_timer"]
    qpre = state["precharge_qpre"]
    physical_guaranteed = bool(qpre.get("physical_turnoff_max_guaranteed", False))
    physical_max = state["precharge_full_chain_max"]
    physical_max_text = "未形成" if physical_max is None else f"{physical_max} s"
    return (
        f"16A净充电参考算法：{state['charge_model_status']}；"
        f"检查 {state['charge_check_passed']}/{state['charge_check_count']} 项通过，"
        f"故障注入 {state['charge_faults_closed']}/{state['charge_fault_count']} 项收敛至CCL=0，"
        f"母线能量参数情景 {state['charge_scenarios']} 组；发布门禁={state['charge_release_status']}。"
        f"保护模型：{state['protection_status']}，"
        f"检查 {state['protection_check_passed']}/{state['protection_check_count']} 项通过，"
        f"仍有{state['protection_open_count']}项开放事项。"
        f"预充定时模型：{timer.get('status', '状态未记录')}；"
        f"FF-B供电至DRV低电平上界 {timer.get('ff_b_power_to_drv_low_max_s', '未记录')} s；"
        f"MOSFET物理关断最大时间有保证={physical_guaranteed}，整链物理最大时间={physical_max_text}。"
        "这些是参考算法/纸面模型结果，不构成固件、硬件、台架或整车验收。"
    )


def update_line(
    content: str,
    description: str,
    predicate: Callable[[str], bool],
    replacement: str,
) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.replace("\r\n", "\n").split("\n")
    matches = [index for index, line in enumerate(lines) if predicate(line)]
    if len(matches) != 1:
        raise ValueError(f"{description} 应恰好匹配一行，实际 {len(matches)} 行")
    lines[matches[0]] = replacement
    return newline.join(lines)


def update_table_row(content: str, key: str, replacement: str) -> str:
    return update_line(
        content,
        f"表格行 {key}",
        lambda line: line.startswith(f"| {key} |"),
        replacement,
    )


def upsert_todo_drawing(content: str, replacement: str) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.replace("\r\n", "\n").split("\n")
    historical = [i for i, line in enumerate(lines) if line.startswith("- [x] 本轮DRC定点修复：")]
    if len(historical) > 1:
        raise ValueError("TODO中存在多个旧本轮DRC定点修复条目")
    if historical:
        lines[historical[0]] = (
            "- [x] 历史DRC/原理图定点修复记录：既往修复细节仅作历史；"
            "当前局部连线与DRC状态以本轮发布证据为准。"
        )
    matches = [i for i, line in enumerate(lines) if line.startswith("- [x] 本轮U18局部连线重画：")]
    if len(matches) > 1:
        raise ValueError("TODO中存在多个本轮U18局部连线重画条目")
    if matches:
        lines[matches[0]] = replacement
    else:
        reopen = [
            i for i, line in enumerate(lines)
            if line.startswith(("- [x] 官方重开正式根EPRJ3", "- [ ] 官方重开正式根EPRJ3"))
        ]
        if len(reopen) != 1:
            raise ValueError("找不到唯一的正式根EPRJ3重开待办，无法插入U18摘要")
        lines.insert(reopen[0] + 1, replacement)
    return newline.join(lines)


def render_documents(state: dict[str, Any]) -> dict[Path, str]:
    counts = (
        f"{state['board_components']}板上元件、{state['normalized_mpn']}种归一化采购MPN "
        f"（{state['raw_mpn']}个原始MPN字符串）；另含{state['offboard_components']}个板外CAD载体，"
        f"共{state['cad_components']}个CAD位号、{state['pins']}针。"
    )
    drc = drc_text(state)
    current_warning_text = warning_text(state)
    current_drawing_text = drawing_summary_text(state)
    snapshot_match_text = (
        f"快照记录ID比较匹配={state['snapshot_match']}（针脚x/y绝对容差"
        f"{state['pin_coordinate_tolerance_absolute']:g}，最大实际偏差"
        f"{state['maximum_pin_coordinate_difference']:.17g}）；"
        f"快照文件字节相同={state['snapshot_files_byte_identical']}"
    )
    if state.get("identical_copy"):
        readback_label = "同字节副本重开"
        root_line = (
            f"{state['readback_statement']}；副本项目={state['readback_project']}，"
            f"{state['cad_components']} CAD位号/{state['pins']}针，DRC {drc}；"
            f"P1 SHA={state['p1_sha']}，{snapshot_match_text}。"
            f"正式根路径图页状态={state['readback']['root_path_status']}，待用户核对。"
            f"{current_warning_text}。"
        )
        root_todo_line = (
            "- [ ] 官方重开正式根EPRJ3：图页打开超时，待用户核对；"
            f"同字节副本重开PASS，{state['cad_components']}CAD位号/{state['pins']}针，{drc}。"
        )
        root_table_line = (
            f"| 根路径客户端重开 | {state['readback']['root_path_status']} | "
            f"同字节副本官方重开PASS，项目={state['readback_project']}；正式根路径图页打开超时待用户核对；"
            f"{state['cad_components']}位号/{state['pins']}针，DRC {drc}；{current_warning_text} |"
        )
        current_drc_source = "同字节副本重开记录"
        requirements_status_line = (
            f"{state['readback_statement']}；本次读取项目={state['readback_project']}；"
            f"正式根路径状态={state['readback']['root_path_status']}。"
            f"{state['cad_components']}位号/{state['pins']}针，DRC {drc}；P1 SHA={state['p1_sha']}，"
            f"{snapshot_match_text}。"
        )
    else:
        readback_label = "正式根路径重开"
        root_line = (
            f"当前正式根路径重开：{state['root_status']}，{state['cad_components']} CAD位号/"
            f"{state['pins']}针，DRC {drc}；P1 SHA={state['p1_sha']}，"
            f"{snapshot_match_text}。{current_warning_text}。"
        )
        root_todo_line = (
            f"- [x] 官方重开正式根EPRJ3，{state['cad_components']}CAD位号/"
            f"{state['pins']}针，{drc}；本轮源文件读回与计数见Layout验证证据。"
        )
        root_table_line = (
            f"| 根路径客户端重开 | {state['root_status']} | {state['cad_components']}位号/"
            f"{state['pins']}针，DRC {drc}；P1 SHA={state['p1_sha']}，"
            f"{snapshot_match_text}；{current_warning_text} |"
        )
        current_drc_source = "正式根路径重开记录"
        requirements_status_line = (
            f"正式根入口当前官方重开：{state['cad_components']}位号/{state['pins']}针，DRC {drc}；"
            f"P1 SHA={state['p1_sha']}，{snapshot_match_text}。"
        )
    historical_warning_text = state["historical_independent_warnings"]
    if historical_warning_text is None:
        historical_warning_text = "未记录"
    rpre = state["rpre"]
    rpre_summary = (
        f"R_PRE在正式快照中存在（MPN {rpre.get('mpn', '未记录')}；"
        f"BOM标记={rpre.get('bom')}、PCB标记={rpre.get('pcb')}表示该板外件不重复计入）；"
        "R_PRE.1=QPRE_SW、R_PRE.2=PACK+、Q_PRE2.2=QPRE_SW均逐针PASS，QPRE_SW已闭合。"
    )

    readme_path = MODULE / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = update_line(
        readme,
        "README current counts",
        lambda line: line.startswith("当前工程："),
        f"当前工程：[BMS-9S4P.eprj3](BMS-9S4P.eprj3)，必须与同目录 `sch/` 一起保留。{counts}",
    )
    readme = update_line(
        readme,
        "README current validation paragraph",
        lambda line: line.startswith("已整理到人工Layout交接"),
        (
            f"已整理到人工Layout交接，可按约束开展可回退初摆；冻结布线和投板仍为HOLD。{root_line} "
            f"{current_drawing_text} {geometry_evidence(state)} 旧独立路径记录中的{historical_warning_text}条警告明确属于历史，"
            "不作为本轮独立路径通过证据。"
            "详见[DRC修复结论](检查/DRC修复结论.md)。"
        ),
    )

    todo_path = MODULE / "TODO.md"
    todo = todo_path.read_text(encoding="utf-8")
    todo = update_line(
        todo,
        "TODO current stage",
        lambda line: line.startswith("当前阶段为"),
        "当前阶段为人工Layout初摆交接；布线冻结与投板HOLD。原生文件、BOM、布局约束入口见[README](README.md)。",
    )
    todo = update_line(
        todo,
        "TODO current root readback",
        lambda line: line.startswith(("- [x] 官方重开正式根EPRJ3", "- [ ] 官方重开正式根EPRJ3")),
        root_todo_line,
    )
    todo_drawing_details = current_drawing_text.partition("：")[2]
    if not todo_drawing_details:
        raise ValueError("U18摘要缺少待办显示所需的概览正文")
    todo = upsert_todo_drawing(todo, f"- [x] 本轮U18局部连线重画：{todo_drawing_details}")
    todo = update_line(
        todo,
        "TODO DRC export gate",
        lambda line: line.startswith("- [ ]") and "DRC" in line and ("警告" in line or "明细" in line),
        f"- [ ] 定位并逐项审查{current_warning_text}；未关闭前布线冻结和投板保持HOLD。"
        "不能把既往DRC日志或旧修复计数当成本轮结论；详见[DRC结论](检查/DRC修复结论.md)。",
    )

    current_path = MODULE / "检查" / "当前设计状态.md"
    current = current_path.read_text(encoding="utf-8")
    source_input_status = (
        "PASS_IDENTICAL_COPY_SHA_ROOT_PATH_HOLD"
        if state.get("identical_copy") else state["root_status"]
    )
    source_input_detail = (
        "同字节副本与正式native SHA一致；正式根路径图页打开超时待用户核对；"
        if state.get("identical_copy") else ""
    )
    current = update_table_row(
        current,
        "原生文件/采购输入",
        f"| 原生文件/采购输入 | {source_input_status} | {source_input_detail}"
        f"{state['board_components']}板上元件，{state['cad_components']}CAD位号、"
        f"{state['pins']}针；板外载体不计入PCB数量 |",
    )
    current = update_table_row(
        current,
        "料号计数",
        f"| 料号计数 | 已核对 | {state['raw_mpn']}个原始MPN字符串、"
        f"{state['normalized_mpn']}种归一化采购MPN；采购身份不按载体名覆盖 |",
    )
    current = update_table_row(
        current,
        "官方独立路径重开",
        f"| 官方独立路径重开 | 历史记录 | 历史独立路径记录为{historical_warning_text}条警告；"
        "未作为本轮独立路径PASS证据，当前独立路径仍需单独复核 |",
    )
    current = update_table_row(
        current,
        "根路径客户端重开",
        root_table_line,
    )
    geometry = state["geometry"]
    geom_counts = geometry.get("counts", {})
    geom_errors = sum(
        len(geometry.get(key, []))
        for key in ("nonOrthogonal", "missingPins", "foreignNetAtPins", "foreignNetAtWireEndpoints", "danglingLabels")
    )
    current = update_table_row(
        current,
        "局部连线几何",
        f"| 局部连线几何 | {state['geometry_status']} | {state['pins']}针、"
        f"{state['actual_lines']}条活动原生LINE、{state['wire_groups']}个wire group、"
        f"{geom_counts.get('labels', state['netports'])}个网标；"
        f"列出的几何异常{geom_errors}项；不代替ERC/DRC或PCB/实物验证 |",
    )
    current = upsert_table_row(
        current,
        "DRC定点修复",
        f"| DRC定点修复 | 历史记录 | 既往修复信息不代表本轮DRC结果；当前状态见{current_drc_source}证据：{current_warning_text}；"
        "见[DRC修复结论](DRC修复结论.md) |",
        "旧注释清理",
    )
    current = upsert_table_row(
        current,
        "U18局部连线重画",
        f"| U18局部连线重画 | {state['drawing_summary']['status']} | {current_drawing_text} |",
        "局部连线几何",
    )
    current = upsert_table_row(
        current,
        "板外R_PRE与QPRE_SW语义",
        f"| 板外R_PRE与QPRE_SW语义 | PASS | {rpre_summary} |",
        "根路径客户端重开",
    )
    precharge_timer = state["precharge_timer"]
    current = update_table_row(
        current,
        "预充纸面模型",
        f"| 预充纸面模型 | 逻辑窗口PASS/物理HOLD | FF-B供电至DRV低电平上界 "
        f"{precharge_timer.get('ff_b_power_to_drv_low_max_s', '未记录')}s；"
        "不含BJT/MOSFET物理释放，整链最大关断时间未形成保证；旧2.970s仅为历史对照 |",
    )
    current = update_table_row(
        current,
        "电池16A/回馈",
        f"| 电池16A/回馈 | 参考算法模型 | {state['charge_model_status']}；"
        f"检查{state['charge_check_passed']}/{state['charge_check_count']}，"
        f"故障注入{state['charge_faults_closed']}/{state['charge_fault_count']}项CCL=0，"
        f"{state['charge_scenarios']}组能量情景；发布门禁={state['charge_release_status']} |",
    )
    current = upsert_table_row(
        current,
        "保护模型复算",
        f"| 保护模型复算 | {state['protection_status']} | "
        f"{state['protection_check_passed']}/{state['protection_check_count']}项检查通过；"
        f"开放事项{state['protection_open_count']}项；不代表硬件/台架验收 |",
        "电池16A/回馈",
    )

    requirements_path = MODULE / "项目要求.md"
    requirements = requirements_path.read_text(encoding="utf-8")
    requirements = update_line(
        requirements,
        "项目要求 §5 current counts",
        lambda line: line.startswith("20") and "模块根目录EPRJ3及sch" in line,
        f"{state['date']}模块根目录EPRJ3及sch当前读回：{counts} 已完成97项重复NET清理、7处off-grid修正、"
        "补齐板外R_PRE原理图表达、145个空Value及光耦错误C号映射修正；采购MPN保持原身份。",
    )
    requirements = update_line(
        requirements,
        "项目要求 §5 current DRC/status",
        lambda line: line.startswith(("正式根入口与同字节独立路径", "正式根入口当前官方重开", "同字节副本当前重开")),
        f"{('同字节副本当前重开（正式根路径待核）' if state.get('identical_copy') else '正式根入口当前官方重开')}："
        f"{requirements_status_line}"
        f"{current_drawing_text} {geometry_evidence(state)} 独立路径旧记录的{historical_warning_text}条警告属于历史，"
        f"本轮未据此宣称独立路径通过。{current_warning_text}；审查完成前布线冻结和投板均HOLD。"
        f"{rpre_summary} {model_evidence(state)} "
        "PCB布局布线仍由人类完成，代理未创建PCB。详见[DRC修复结论](检查/DRC修复结论.md)。",
    )

    layout_path = MODULE / "原理图准备" / "Layout交接.md"
    layout = layout_path.read_text(encoding="utf-8")
    layout = update_line(
        layout,
        "Layout handoff counts",
        lambda line: line.startswith("20") and "当前输入为" in line,
        f"{state['date']}。当前输入为[根目录EPRJ3](../BMS-9S4P.eprj3)及其 `sch/`、"
        f"[BOM](BOM.csv)、[器件映射](../检查/Layout器件映射.json)。{state['board_components']}板上元件、"
        f"{state['normalized_mpn']}种归一化采购MPN；{state['cad_components']}CAD位号/{state['pins']}针，"
        f"其中{state['offboard_components']}个板外载体不计入PCB。",
    )
    layout = update_line(
        layout,
        "Layout handoff validation paragraph",
        lambda line: line.startswith("**可开展可回退初摆"),
        (
            f"**可开展可回退初摆；冻结布线和投板HOLD。** {root_line} "
            f"{current_drawing_text} {geometry_evidence(state)} 旧独立路径的{historical_warning_text}条警告属于历史，"
            "不能证明本轮独立路径已通过；明细核实前不冻结布线。"
            "详见[当前证据](../检查/Layout验证/当前发布.json)和[DRC修复结论](../检查/DRC修复结论.md)。"
        ),
    )
    layout = update_line(
        layout,
        "Layout current DRC gate",
        lambda line: line.startswith("1. ") and "当前" in line and "警告" in line,
        f"1. 定位并逐项审查{current_warning_text}；审查完成前冻结原理图和布线仍为HOLD。"
        f"旧独立路径{historical_warning_text}警告只是历史记录，不能替代本轮独立路径复核。",
    )

    conclusion_path = CONCLUSION_PATH
    conclusion = f"""# BMS 当前DRC修复与Layout证据结论

日期：{state['date']}。当前输入：[发布清单](Layout验证/当前发布.json)、[{('同字节副本重开读回' if state.get('identical_copy') else '正式根路径重开读回')}](Layout验证/root-after-cleanup-reopen.json)、[几何核验](Layout验证/final-geometry.json)。采购映射细节见[采购身份核查](DRC修复/采购身份核查.md)。

## 当前计数与验证状态

| 项目 | 当前读回 |
| --- | --- |
| 板上元件 / 归一化采购MPN | {state['board_components']} / {state['normalized_mpn']}（原始MPN字符串 {state['raw_mpn']}） |
| CAD位号 / 针 | {state['cad_components']} / {state['pins']}（含{state['offboard_components']}个板外载体） |
| {('同字节副本重开 / 正式根路径状态' if state.get('identical_copy') else '根路径官方重开')} | {state['readback_statement']}；{('副本项目=' + str(state['readback_project']) + '；根路径=' + state['readback']['root_path_status']) if state.get('identical_copy') else state['root_status']}；P1 SHA={state['p1_sha']}；记录ID比较匹配={state['snapshot_match']}，针脚x/y绝对容差={state['pin_coordinate_tolerance_absolute']:g}，最大实际偏差={state['maximum_pin_coordinate_difference']:.17g}；快照文件字节相同={state['snapshot_files_byte_identical']} |
| {('同字节副本当前DRC' if state.get('identical_copy') else '根路径当前DRC')} | {drc} |
| DRC review / Layout正式放行 | {state['drc_review_state']} / {state['layout_release_status']} |
| 警告明细 | {current_warning_text} |
| U18局部连线重画 | {current_drawing_text} |
| 原理图连线几何 | {geometry_evidence(state)} |
| 独立路径 | 旧记录中的{historical_warning_text}条警告只作历史，不能作为本轮独立路径通过证据 |
| 板外R_PRE/QPRE_SW | {rpre_summary} |

## 当前模型状态

{model_evidence(state)}

## 既往原理图修复记录

- 清理97项重复NET。
- 修正7处off-grid坐标。
- 补齐1处板外R_PRE原理图表达；当前快照已含R_PRE，R_PRE两针与Q_PRE2.2逐针语义PASS，QPRE_SW已闭合。
- 补齐145个空Value字段。
- 纠正光耦错误C号映射；采购MPN `VO617A-2X007T` 身份保持不变，不用不同等级载体号覆盖采购身份。

供应商属性规范化及其历史DRC计数属于既往记录，不作为本轮告警清除依据。当前告警状态以本轮{readback_label}读回为准：{current_warning_text}。不能把旧的{historical_warning_text}警告记录当成本轮结果。

当前原理图图面按U18完成VC/SRP局部连线重画的保存/重开和逐针语义核验；这不替代DRC告警关闭、人工PCB布局布线或实物验证。**冻结布线和投板继续HOLD**，直至当前告警完成定位、审阅并按项目门槛闭合。以上模型状态只引用现有结果文件，不替代固件、硬件、台架或整车验收；无样板，热、保护、EMC及整车验证仍未完成。PCB布局布线由人类完成，代理未创建PCB。

文档入口：[模块README](../README.md) · [当前设计状态](当前设计状态.md) · [Layout交接](../原理图准备/Layout交接.md) · [项目要求](../项目要求.md)。
"""

    psu_doc = PSU_DOC_PATH.read_text(encoding="utf-8")
    psu_status = (
        "状态：条件纸面方案，仅供 Layout 初始摆放；PSU 输入已接到 TAP_B9；"
        f"{state['readback_statement']}（208器件/700引脚）。"
        f"当前读回DRC为{state['drc']['fatal']} fatal/{state['drc']['error']} error/"
        f"{state['drc']['warning']} warning；警告规则/位号明细尚未审阅，drcReview=HOLD，"
        "Layout正式放行=HOLD。辅助正极线支路故障/短路分断验收仍 OPEN：电芯 DC 最大预期短路电流、"
        "熔断总清除 I²t 配合及首板试验未闭合；未冻结或放行投板。"
    )
    psu_doc = update_line(
        psu_doc,
        "PSU status and readback scope",
        lambda line: line.startswith("状态：条件纸面方案"),
        psu_status,
    )
    psu_p1_line = (
        "当前 P1 已将 `PSU5_EXT.2`、`C_PSU_IN1.1`、`C_PSU_IN2.1`、`C_PSU_IN3.1` 接到 `TAP_B9`，"
        "三只旁路电容另一端保持 `B0`；板端复用 `J_CELL_SERVICE.10=TAP_B9`，不增加第二只保险。"
        f"当前 P1 SHA256 为 `{state['p1_sha']}`。严格语义几何逐针证明四个输入正端均实际连接到唯一 `TAP_B9`、"
        "非 NC 且有物理线段/netport 证据；同字节隔离副本官方只读重开PASS并绑定相同的三份正式原生文件SHA，"
        "正式根路径图页打开超时、待用户核对。该核对证明当前原理图连接和记录几何，不是 PCB DRC、"
        "布线/载流验收或实物检查。"
    )
    psu_doc = update_line(
        psu_doc,
        "PSU current schematic evidence summary",
        lambda line: line.startswith("当前 final P1 已将 `PSU5_EXT.2`"),
        psu_p1_line,
    )
    comparisons = state["drawing_summary"]["verification"]["record_comparison"]
    if any(comparisons.get(name, {}).get("equal") is not True for name in ("rows", "ports", "wires", "texts")):
        raise ValueError("PSU文档更新要求rows/ports/wires/texts重开比较全部相同")
    psu_current_evidence_line = (
        f"- 当前保存快照 `BMS模块/检查/Layout验证/current-snapshot.json` SHA256 `{state['snapshot_sha256']}`："
        f"{state['cad_components']} 个器件、{state['pins']} 个 pin。当前严格语义几何记录 "
        f"`BMS模块/检查/Layout验证/final-geometry.json` SHA256 `{state['geometry_sha256']}`："
        f"{state['cad_components']} 个器件、{state['pins']} 个 pin、{state['actual_lines']} 条原生 LINE、"
        f"{state['netports']} 个 netport；语义状态 {state['geometry_status']}，"
        f"defectCounts 全零={state['geometry_defects_zero']}。同字节隔离副本官方重开记录 "
        f"`BMS模块/检查/Layout验证/root-after-cleanup-reopen.json` SHA256 `{state['reopen_sha256']}`："
        f"208/700，rows/ports/wires/texts按ID比较全部一致，快照文件字节相同="
        f"{state['snapshot_files_byte_identical']}；当前读回DRC {drc}，警告明细尚未审阅，"
        "drcReview=HOLD；正式根路径图页打开超时、待用户核对。旧独立重开记录 "
        f"`BMS模块/检查/Layout验证/final-independent-reopen.json` SHA256 "
        f"`{state['historical_independent_reopen_sha256']}` "
        "属于修复前历史证据，不代表当前重开。以上均为原理图记录/重开证据，不是 PCB 布线、铜箔载流、温升或支路分断验收。"
    )
    psu_doc = update_line(
        psu_doc,
        "PSU current snapshot, geometry and readback evidence",
        lambda line: line.startswith("- 当前官方快照 `BMS模块/检查/Layout验证/current-snapshot.json`"),
        psu_current_evidence_line,
    )
    psu_current_p1_line = (
        f"- 当前原理图：`BMS模块/sch/Schematic1/P1.esch2`，SHA256 `{state['p1_sha']}`。"
        f"同字节隔离副本官方只读重开PASS（208器件/700引脚、记录ID比较匹配={state['snapshot_match']}）；"
        "正式根路径图页打开超时，待用户核对。当前读回DRC为"
        f"{state['drc']['fatal']} fatal/{state['drc']['error']} error/{state['drc']['warning']} warning，"
        "警告明细尚未审阅，drcReview=HOLD，Layout正式放行=HOLD。"
    )
    psu_doc = update_line(
        psu_doc,
        "PSU current P1 SHA and root-path status",
        lambda line: line.startswith("- 当前原理图：`BMS模块/sch/Schematic1/P1.esch2`"),
        psu_current_p1_line,
    )

    return {
        readme_path: readme,
        todo_path: todo,
        current_path: current,
        requirements_path: requirements,
        layout_path: layout,
        conclusion_path: conclusion,
        PSU_DOC_PATH: psu_doc,
    }


def upsert_table_row(content: str, key: str, new_row: str, anchor_key: str) -> str:
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.replace("\r\n", "\n").split("\n")
    existing = [i for i, line in enumerate(lines) if line.startswith(f"| {key} |")]
    if len(existing) > 1:
        raise ValueError(f"当前设计状态中存在多条表格行：{key}")
    if existing:
        lines[existing[0]] = new_row
        return newline.join(lines)
    anchors = [i for i, line in enumerate(lines) if line.startswith(f"| {anchor_key} |")]
    if len(anchors) != 1:
        raise ValueError(f"无法在当前设计状态表格中找到唯一锚点：{anchor_key}")
    lines.insert(anchors[0] + 1, new_row)
    return newline.join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="写入允许更新的Markdown文件；默认只显示diff")
    args = parser.parse_args()

    state = collect_state()
    rendered = render_documents(state)
    changes: list[tuple[Path, str, str]] = []
    for path, new_text in rendered.items():
        old_text = path.read_text(encoding="utf-8") if path.exists() else ""
        if old_text != new_text:
            changes.append((path, old_text, new_text))

    if not changes:
        print("文档已与输入证据一致；没有改动。")
        return 0

    if not args.apply:
        print("只读预览。复核以下diff后，使用 --apply 才会写入文档。")
        for path, old_text, new_text in changes:
            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(path.relative_to(ROOT)),
                tofile=str(path.relative_to(ROOT)),
            )
            print("".join(diff), end="")
        return 0

    allowlist = {
        MODULE / "README.md",
        MODULE / "TODO.md",
        MODULE / "项目要求.md",
        MODULE / "检查" / "当前设计状态.md",
        MODULE / "原理图准备" / "Layout交接.md",
        CONCLUSION_PATH,
        PSU_DOC_PATH,
    }
    unexpected = {path for path, _, _ in changes} - allowlist
    if unexpected:
        raise RuntimeError(f"脚本目标超出文档白名单：{unexpected}")

    for path, _, new_text in changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(new_text)
    print("已更新以下文档：")
    for path, _, _ in changes:
        print(f"- {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
