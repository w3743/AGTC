#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在最终导出完成后刷新水泵驱动交付哈希、网表和共用库证据。

运行位置：电力工程根目录
  py -3 -B .\\水泵驱动模块\\EDA自动化\\刷新交付证据.py

本脚本不会启动或控制 EDA。它会调用离线网表核对脚本、共享库同步及其 --check，
并更新当前交付清单、最终库同步核对和当前状态。它不会运行或改写 DRC 明细；
当前状态中的旧 DRC 结论会标成历史参考/待本轮复核。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "水泵驱动模块"
CHECK = MODULE / "检查"
EDA = MODULE / "eda工程文件"
DELIVERY = EDA / "导出" / "当前交付"

MANIFEST_PATH = CHECK / "当前交付清单.json"
SYNC_REPORT_PATH = CHECK / "最终库同步核对.json"
STATUS_PATH = CHECK / "当前状态.md"
DRC_PATH = CHECK / "DRC明细与处理.json"
NETLIST_CHECK_PATH = CHECK / "网表逐脚核对.json"
BASELINE_PATH = CHECK / "源文件基线.json"

NETLIST_VERIFIER = CHECK / "验证原理图.py"
LIBRARY_SYNC = ROOT / "元件封装库" / "工具" / "同步全部.py"
LIBRARY_EVIDENCE = ROOT / "元件封装库" / "检查" / "同步证据.json"

PUMP_SCHEMATIC_REL = "水泵驱动模块/eda工程文件/水泵驱动/sch/Schematic1/P1.esch2"
NETLIST_REL = "水泵驱动模块/eda工程文件/导出/当前交付/水泵驱动.enet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temp_name = stream.name
            stream.write(content)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def normalized_relative(value: str) -> str:
    relative = str(value).replace("\\", "/")
    while relative.startswith("./"):
        relative = relative[2:]
    return relative


def resolve_module_file(relative: str) -> Path:
    candidate = (MODULE / Path(relative.replace("/", os.sep))).resolve()
    try:
        candidate.relative_to(MODULE.resolve())
    except ValueError as exc:
        raise ValueError(f"清单路径超出水泵驱动模块：{relative}") from exc
    return candidate


def run_python(script: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
    relative_script = script.relative_to(ROOT).as_posix()
    args = [sys.executable, "-B", relative_script]
    if extra_args:
        args.extend(extra_args)
    completed = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": subprocess.list2cmdline(args),
        "command_args": args,
        "cwd": ".",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def snapshot_baselines(source_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("源文件基线.json 缺少 sources 列表")

    output: list[dict[str, Any]] = []
    for item in sources:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("源文件基线.json 存在无效 sources 项")
        relative = normalized_relative(item["path"])
        if relative.startswith("水泵驱动模块/"):
            continue
        path = (ROOT / Path(relative.replace("/", os.sep))).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"源文件基线路径超出工程根目录：{relative}") from exc
        current = sha256_file(path) if path.is_file() else None
        baseline = item.get("sha256")
        output.append({
            "kind": item.get("kind"),
            "path": relative,
            "baseline_sha256": baseline,
            "current_sha256_before": current,
            "matches_baseline_before": bool(current and current == baseline),
        })
    return output, sha256_file(BASELINE_PATH)


def finalize_baselines(entries: list[dict[str, Any]]) -> dict[str, Any]:
    for item in entries:
        path = ROOT / Path(item["path"].replace("/", os.sep))
        current = sha256_file(path) if path.is_file() else None
        item["current_sha256_after"] = current
        item["matches_baseline_after"] = bool(current and current == item["baseline_sha256"])
        item["unchanged_during_refresh"] = current == item["current_sha256_before"]
        item["unchanged"] = bool(
            item["matches_baseline_before"]
            and item["matches_baseline_after"]
            and item["unchanged_during_refresh"]
        )
    return {
        "manifest_path": "水泵驱动模块/检查/源文件基线.json",
        "manifest_sha256": sha256_file(BASELINE_PATH),
        "source_count": len(entries),
        "all_unchanged": len(entries) == 6 and all(item["unchanged"] for item in entries),
        "sources": entries,
    }


def refresh_delivery_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("当前交付清单.json 缺少 files 列表")
    for item in files:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("当前交付清单.json 存在无效 files 项")
        path = resolve_module_file(normalized_relative(item["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"交付文件不存在：{item['path']}")
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
    return files


def update_status_markdown(
    text: str,
    date_text: str,
    netlist_sha: str,
    netlist_status: str,
    component_count: Any,
    pin_count: Any,
    library_status: str,
    old_drc_sha: Any,
    current_schematic_sha: str,
) -> str:
    text, date_count = re.subn(r"(?m)^更新：\d{4}-\d{2}-\d{2}。", f"更新：{date_text}。", text, count=1)
    if date_count != 1:
        raise ValueError("当前状态.md 未找到唯一的更新日期行")

    netlist_row = (
        f"| ENET逐脚核对 | {netlist_status} | [网表逐脚核对.json](网表逐脚核对.json)"
        f"以当前ENET、BOM及连接契约为输入，核对{component_count}个器件、{pin_count}个引脚"
        f"（含明确NC项），状态{netlist_status}。当前ENET SHA256为`{netlist_sha}`。"
        "该结果证明导出网表与契约/BOM一致，不证明DRC、PCB或实物正确。 |"
    )
    text, net_count = re.subn(r"(?m)^\| ENET逐脚核对 \|.*$", lambda _: netlist_row, text, count=1)
    if net_count != 1:
        raise ValueError("当前状态.md 未找到唯一的 ENET逐脚核对 行")

    drc_sha_text = old_drc_sha if isinstance(old_drc_sha, str) and old_drc_sha else "未记录"
    source_match_text = "与当前原理图源 SHA256 一致" if drc_sha_text == current_schematic_sha else "与当前原理图源 SHA256 不一致"
    drc_row = (
        "| 原生DRC | 待本轮复核 | [DRC明细与处理.json](DRC明细与处理.json)仅保留既有记录，"
        f"其记录的原理图源 SHA256 为`{drc_sha_text}`，{source_match_text}（当前为`{current_schematic_sha}`）。"
        "本刷新脚本未运行EDA/DRC；该记录不作为本轮DRC结论。 |"
    )
    text, drc_count = re.subn(r"(?m)^\| 原生DRC \|.*$", lambda _: drc_row, text, count=1)
    if drc_count != 1:
        raise ValueError("当前状态.md 未找到唯一的 原生DRC 行")

    library_row = (
        f"| 共用库同步 | {library_status} | [最终库同步核对.json](最终库同步核对.json)："
        "本轮同步及只读校验结果、6份其它源文件SHA256基线核对见记录；该证据不替代PCB或实物验收。 |"
    )
    text, library_count = re.subn(r"(?m)^\| 共用库同步 \|.*$", lambda _: library_row, text, count=1)
    if library_count != 1:
        raise ValueError("当前状态.md 未找到唯一的 共用库同步 行")

    # 旧状态页在表格后有一段基于旧 DRC 的详细解释。保留 DRC JSON 原件，
    # 但停止把旧警告列表和数量呈现为刷新后的当前结论。
    marker = "告警涉及位号："
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + (
            "\n\n本轮刷新脚本没有运行原生DRC。上方DRC条目只记录既有明细的来源SHA与当前原理图SHA关系；"
            "完成本轮GUI DRC检查后，再由负责人更新当前告警分类、数量和处理说明。"
        )
    return text.rstrip() + "\n"


def main() -> int:
    for path in (MANIFEST_PATH, SYNC_REPORT_PATH, STATUS_PATH, DRC_PATH, NETLIST_CHECK_PATH,
                 BASELINE_PATH, NETLIST_VERIFIER, LIBRARY_SYNC):
        if not path.is_file():
            raise FileNotFoundError(f"缺少刷新所需的现有文件：{path.relative_to(ROOT).as_posix()}")

    # 读取现有证据作为输入上下文；旧 DRC 只记入历史来源关系，不继承其数量/结论。
    manifest = read_json(MANIFEST_PATH)
    previous_sync_report = read_json(SYNC_REPORT_PATH)
    old_status = STATUS_PATH.read_text(encoding="utf-8-sig")
    old_drc = read_json(DRC_PATH)
    previous_netlist_check = read_json(NETLIST_CHECK_PATH)
    baseline_manifest = read_json(BASELINE_PATH)

    baseline_entries, _ = snapshot_baselines(baseline_manifest)
    old_drc_record_sha = sha256_file(DRC_PATH)
    current_schematic_path = ROOT / Path(PUMP_SCHEMATIC_REL.replace("/", os.sep))
    current_schematic_sha_before = sha256_file(current_schematic_path)
    netlist_path = ROOT / Path(NETLIST_REL.replace("/", os.sep))
    if not netlist_path.is_file():
        raise FileNotFoundError(f"当前交付 ENET 不存在：{NETLIST_REL}")

    verifier_run = run_python(NETLIST_VERIFIER)
    verifier_report = read_json(NETLIST_CHECK_PATH)
    current_netlist_sha = sha256_file(netlist_path)
    verifier_sha = verifier_report.get("inputs", {}).get("netlist_sha256")
    verifier_current = (
        verifier_run["exit_code"] == 0
        and verifier_report.get("status") == "PASS"
        and verifier_sha == current_netlist_sha
    )

    sync_run = run_python(LIBRARY_SYNC)
    check_run = run_python(LIBRARY_SYNC, ["--check"])
    current_schematic_sha_after = sha256_file(current_schematic_path)
    sync_evidence: dict[str, Any] | None = None
    sync_evidence_error = None
    try:
        sync_evidence = read_json(LIBRARY_EVIDENCE)
    except Exception as exc:  # 把缺失/损坏作为本轮证据失败记录下来。
        sync_evidence_error = f"{type(exc).__name__}: {exc}"

    sync_pump_record = None
    if sync_evidence:
        for source in sync_evidence.get("sources", []):
            if normalized_relative(source.get("path", "")) == PUMP_SCHEMATIC_REL:
                sync_pump_record = source
                break
    sync_source_matches = bool(
        sync_pump_record
        and sync_pump_record.get("sha256") == current_schematic_sha_after
        and sync_pump_record.get("sha256_after_parse") == current_schematic_sha_after
        and sync_pump_record.get("sha256_after_generation") == current_schematic_sha_after
        and sync_pump_record.get("unchanged") is True
    )
    source_hash_stable = current_schematic_sha_before == current_schematic_sha_after
    finalized_baselines = finalize_baselines(baseline_entries)
    library_ok = bool(
        sync_run["exit_code"] == 0
        and check_run["exit_code"] == 0
        and sync_source_matches
        and source_hash_stable
        and finalized_baselines["all_unchanged"]
        and sync_evidence
        and sync_evidence.get("coverage", {}).get("source_sha256_verified_before_after") is True
    )

    now = datetime.now().astimezone()
    now_iso = now.isoformat(timespec="seconds")
    sync_evidence_hash = sha256_file(LIBRARY_EVIDENCE) if LIBRARY_EVIDENCE.is_file() else None
    netlist_summary = verifier_report.get("summary", {})
    component_count = netlist_summary.get("enet_components", 0)
    pin_count = netlist_summary.get("pins_compared", 0)

    refreshed_files = refresh_delivery_files(manifest)
    manifest["date"] = now.date().isoformat()
    manifest["native_components"] = component_count
    manifest["pins_verified"] = pin_count
    manifest["netlist_check"] = verifier_report.get("status", "FAIL") if verifier_current else "FAIL"
    manifest["library_sync_check"] = "PASS" if library_ok else "FAIL"
    manifest["drc"] = {
        "status": "PENDING_CURRENT_ROUND_DRC",
        "fatal": None,
        "error": None,
        "warn": None,
        "disposition": "本刷新脚本未运行EDA/DRC；既有DRC明细仅作为历史来源记录，不作为本轮结论。完成本轮DRC后更新此字段。",
        "previous_record": {
            "path": "水泵驱动模块/检查/DRC明细与处理.json",
            "record_sha256": old_drc_record_sha,
            "recorded_source_sha256": old_drc.get("source_sha256"),
            "current_schematic_sha256": current_schematic_sha_after,
            "source_sha_matches": old_drc.get("source_sha256") == current_schematic_sha_after,
            "treated_as_current_round": False,
        },
    }

    prior_root_readme = previous_sync_report.get("root_readme", {
        "path": "README.md",
        "pump_entry_updated_to_current_native_project_pdf_enet_and_checks": None,
    })
    sync_report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if library_ok else "FAIL",
        "executions": [
            {
                **sync_run,
                "summary": sync_evidence.get("summary") if sync_evidence else None,
                "source_sha256_verified_before_after": bool(
                    sync_evidence and sync_evidence.get("coverage", {}).get("source_sha256_verified_before_after")
                ),
                "sync_evidence_path": "元件封装库/检查/同步证据.json",
                "sync_evidence_sha256": sync_evidence_hash,
                "sync_evidence_error": sync_evidence_error,
            },
            {
                **check_run,
                "result": check_run["stdout"].strip() if check_run["stdout"].strip() else None,
            },
        ],
        "netlist_verification": {
            **verifier_run,
            "path": "水泵驱动模块/检查/网表逐脚核对.json",
            "status": verifier_report.get("status"),
            "current_enet_sha256": current_netlist_sha,
            "verified_enet_sha256": verifier_sha,
            "matches_current_enet": verifier_sha == current_netlist_sha,
            "components": component_count,
            "pins_compared": pin_count,
        },
        "native_netlist": {
            "path": NETLIST_REL,
            "expected_sha256": verifier_sha,
            "current_sha256": current_netlist_sha,
            "matches_expected": verifier_sha == current_netlist_sha,
            "pin_check_path": "水泵驱动模块/检查/网表逐脚核对.json",
            "pin_check_status": verifier_report.get("status"),
            "components": component_count,
            "pins_compared": pin_count,
            "bytes": netlist_path.stat().st_size,
        },
        "pump_schematic": {
            "path": PUMP_SCHEMATIC_REL,
            "sha256_before_sync": current_schematic_sha_before,
            "sha256_after_sync": current_schematic_sha_after,
            "unchanged_during_sync": source_hash_stable,
            "library_sync_source_record_matches": sync_source_matches,
        },
        "other_source_baseline": finalized_baselines,
        "drc_context": {
            "path": "水泵驱动模块/检查/DRC明细与处理.json",
            "record_sha256": old_drc_record_sha,
            "recorded_source_sha256": old_drc.get("source_sha256"),
            "current_schematic_sha256": current_schematic_sha_after,
            "source_sha_matches": old_drc.get("source_sha256") == current_schematic_sha_after,
            "record_treated_as_current_round": False,
            "current_round_drc_run": False,
        },
        "refreshed_delivery_files": refreshed_files,
        "root_readme": prior_root_readme,
        "scope_note": "记录本轮共享库同步、--check、当前ENET逐脚核对和六份其它源文件哈希基线；本脚本不运行EDA/DRC，不构成PCB、实物或整车验收。",
    }

    status_markdown = update_status_markdown(
        old_status,
        now.date().isoformat(),
        current_netlist_sha,
        verifier_report.get("status", "FAIL") if verifier_current else "FAIL",
        component_count,
        pin_count,
        "PASS" if library_ok else "FAIL",
        old_drc.get("source_sha256"),
        current_schematic_sha_after,
    )

    # 所有输入与计算完成后再落盘。DRC JSON 不在写入目标中。
    write_json_atomic(SYNC_REPORT_PATH, sync_report)
    write_json_atomic(MANIFEST_PATH, manifest)
    write_text_atomic(STATUS_PATH, status_markdown)

    print(f"{'PASS' if library_ok else 'FAIL'}: shared library sync and --check")
    print(f"{'PASS' if verifier_current else 'FAIL'}: current ENET verification ({current_netlist_sha})")
    print(f"{'PASS' if finalized_baselines['all_unchanged'] else 'FAIL'}: six other source baselines")
    print(f"Updated: {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    print(f"Updated: {SYNC_REPORT_PATH.relative_to(ROOT).as_posix()}")
    print(f"Updated: {STATUS_PATH.relative_to(ROOT).as_posix()}")
    return 0 if library_ok and verifier_current else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
