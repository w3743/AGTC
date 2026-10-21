#!/usr/bin/env python3
"""Generate/check the layout map from the current official snapshot only.
Read-only check is the default. --write requires matching P1 and entry evidence.
This script never creates or edits a PCB.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

MAP_FIELDS = ("ref", "id", "x", "y", "rotation", "mirror", "name", "mpn", "manufacturer", "component", "symbol", "footprint", "bom", "pcb", "net", "pins")
MACHINE_FIELDS = ("native_sha256", "counts", "source", "pcb_created", "pcb_components", "external_cad_carriers", "consistency")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_mpn(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    for prefix in ("TI ", "TE Connectivity "):
        if value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def recorded_p1_hashes(entry: dict[str, Any]) -> list[str]:
    hashes = []
    for section_name in ("sourceSHA256Before", "sourceSHA256After", "sourceSHA256AfterSessionClose"):
        section = entry.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"入口证据缺少 {section_name}")
        found = [v for k, v in section.items() if str(k).replace("\\", "/").casefold().endswith("/sch/schematic1/p1.esch2")]
        if len(found) != 1 or not isinstance(found[0], str):
            raise ValueError(f"入口证据 {section_name} 未唯一记录 P1.esch2 SHA256")
        hashes.append(found[0].lower())
    if entry.get("sourceUnchanged") is not True or entry.get("sourceUnchangedAfterSessionClose") is not True:
        raise ValueError("入口证据未证明P1在会话前后保持不变")
    return hashes


def load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    module = root / "BMS模块"
    check = module / "检查"
    snapshot_path = check / "Layout验证" / "current-snapshot.json"
    entry = read_json(check / "Layout验证" / "layout-entry-check.json")
    snapshot = read_json(snapshot_path)
    p1_sha = sha256(module / "sch" / "Schematic1" / "P1.esch2")
    recorded = recorded_p1_hashes(entry)
    if any(value != p1_sha for value in recorded):
        raise ValueError(f"拒绝更新：当前P1 SHA256与入口证据不一致；当前={p1_sha}，记录={recorded}")
    snapshot_sha = sha256(snapshot_path)
    if entry.get("comparisonBasisSHA256", "").lower() != snapshot_sha:
        raise ValueError(f"拒绝更新：current-snapshot SHA256与入口证据不一致；当前={snapshot_sha}")
    comparison = entry.get("snapshotComparison") or {}
    if comparison.get("status") != "PASS" or comparison.get("differences") != []:
        raise ValueError("拒绝更新：入口证据未证明快照与原理图读回一致")
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("pcb"), bool) for row in rows):
        raise ValueError("current-snapshot.json 的 rows/pcb 字段格式错误")
    pins = sum(len(row.get("pins") or []) for row in rows)
    expected = {"rows": len(rows), "ports": len(snapshot.get("ports", [])), "wires": len(snapshot.get("wires", [])), "texts": len(snapshot.get("texts", []))}
    declared = entry.get("counts") or {}
    if any(declared.get(k) != v for k, v in expected.items()) or entry.get("pins") != pins or entry.get("boardParts") != sum(row["pcb"] for row in rows):
        raise ValueError("拒绝更新：入口证据计数与当前快照不一致")
    return snapshot, entry, snapshot_sha, p1_sha


def project_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in MAP_FIELDS if key not in row]
    if missing:
        raise ValueError(f"快照元件 {row.get('ref')} 缺少字段：{missing}")
    # Exact projection: including native name expressions; never fall back to historical labels.
    return {key: row[key] for key in MAP_FIELDS}


def build_expected(existing: dict[str, Any], snapshot: dict[str, Any], entry: dict[str, Any], snapshot_sha: str, p1_sha: str) -> dict[str, Any]:
    rows = snapshot["rows"]
    board_rows = [r for r in rows if r["pcb"] is True]
    off_rows = [r for r in rows if r["pcb"] is False]
    board = [project_row(r) for r in board_rows]
    offboard = [project_row(r) for r in off_rows]
    pins = sum(len(r.get("pins") or []) for r in rows)
    mounted = [r for r in rows if r.get("pcb") is True and r.get("bom") is True]
    raw_mpn = {r.get("mpn") for r in mounted if r.get("mpn")}
    normalized_mpn = {norm_mpn(r.get("mpn")) for r in mounted if norm_mpn(r.get("mpn"))}
    board_refs = [r["ref"] for r in board_rows]
    off_refs = [r["ref"] for r in off_rows]
    missing_fp = sorted(r["ref"] for r in board if not (r.get("footprint") or {}).get("uuid") or not (r.get("footprint") or {}).get("name"))
    carrier_status = (entry.get("carrierPinPadReview") or {}).get("status", "NOT_RECORDED")

    out = copy.deepcopy(existing)
    out.update({
        "native_sha256": p1_sha,
        "source": f"Layout验证/current-snapshot.json official reopened snapshot; SHA256={snapshot_sha}",
        "counts": {**(existing.get("counts") or {}), "pcb_components": len(board), "cad_components": len(rows), "pins": pins, "raw_mpn_strings": len(raw_mpn), "normalized_mpn": len(normalized_mpn), "texts": len(snapshot.get("texts", [])), "netports": len(snapshot.get("ports", []))},
        "pcb_created": existing.get("pcb_created", False),
        "pcb_components": board,
        "external_cad_carriers": offboard,
    })
    consistency = copy.deepcopy(existing.get("consistency") or {})
    consistency.update({
        "source_snapshot": "Layout验证/current-snapshot.json",
        "source_snapshot_sha256": snapshot_sha,
        "native_p1_sha256": p1_sha,
        "snapshot_counts": {"rows": len(rows), "board_components": len(board), "offboard_components": len(offboard), "pins": pins, "netports": len(snapshot.get("ports", [])), "wires": len(snapshot.get("wires", [])), "texts": len(snapshot.get("texts", []))},
        "pcb_ref_set_matches_snapshot": len(board_refs) == len(set(board_refs)),
        "offboard_ref_set_matches_snapshot": len(off_refs) == len(set(off_refs)),
        "offboard_refs": off_refs,
        "duplicate_board_refs": sorted(k for k, n in Counter(board_refs).items() if n > 1),
        "duplicate_offboard_refs": sorted(k for k, n in Counter(off_refs).items() if n > 1),
        "board_components_with_footprint_uuid_and_name": len(board) - len(missing_fp),
        "board_components_missing_footprint_binding": missing_fp,
        "layout_entry_status": entry.get("layoutEntry"),
        "carrier_pin_pad_review_status": carrier_status,
        "manufacturing_release": entry.get("manufacturingRelease", "NOT_RECORDED"),
        "footprint_binding_scope": ("Current snapshot footprint UUID/name presence only; not full purchased-part package or pin-to-pad signoff. "
            f"U_LOAD_POWER_VALID carrier/pad review is {carrier_status} for layout geometry only; CTR/receiver compatibility, "
            "exact procurement identity, purchase readiness and manufacturing release remain separate open/HOLD gates."),
        "pcb_created": out["pcb_created"],
    })
    out["consistency"] = consistency
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="只读检查（默认）")
    group.add_argument("--write", action="store_true", help="证据一致后更新映射")
    parser.add_argument("--root", type=Path, help="项目根目录；默认从脚本推导，兼容原参数")
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[2]
    output_path = root / "BMS模块" / "检查" / "Layout器件映射.json"
    try:
        snapshot, entry, snapshot_sha, p1_sha = load_inputs(root)
        if args.write:
            if entry.get("layoutEntry") != "READY_FOR_HUMAN_LAYOUT":
                raise ValueError("拒绝更新：layoutEntry 尚未 READY_FOR_HUMAN_LAYOUT")
            if (entry.get("carrierPinPadReview") or {}).get("status") != "PASS_FOR_LAYOUT_GEOMETRY_ONLY":
                raise ValueError("拒绝更新：光耦载体/针盘几何审查未通过")
        current = read_json(output_path)
        expected = build_expected(current, snapshot, entry, snapshot_sha, p1_sha)
        drift = [key for key in MACHINE_FIELDS if current.get(key) != expected.get(key)]
        if args.write:
            expected["updated"] = date.today().isoformat()
            output_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"已更新 {output_path}；板上{len(expected['pcb_components'])}件、板外{len(expected['external_cad_carriers'])}件；P1 SHA256={p1_sha}；snapshot SHA256={snapshot_sha}")
            return 0
        if drift:
            print(f"FAIL：映射漂移字段：{', '.join(drift)}；默认只读，审核后显式使用 --write。")
            return 1
        print(f"PASS：映射与当前快照一致；板上{len(expected['pcb_components'])}件、板外{len(expected['external_cad_carriers'])}件；未写入。")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())