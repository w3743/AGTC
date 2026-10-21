"""Stage or apply the approved resistor-chain source-data update.

This is a non-native source-data tool. --prepare writes only a review plan under
检查/tmp. --apply requires unchanged input hashes, archives and verifies the
preimage, updates BOM/source/connection/cost data, and refreshes count evidence.
It never opens or edits EasyEDA or an EPRJ3 file.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
PREP = MODULE / "原理图准备"
TMP = HERE / "tmp" / "approved_resistor_source_update"
ARCHIVE = MODULE / "归档" / "历史与调试.zip"
BOM = PREP / "BOM.csv"
AFE = PREP / "AFE元件连接.json"
CONNECTIONS = PREP / "连接表.csv"
COST = HERE / "当前主板成本.csv"
COUNT_EVIDENCE = HERE / "当前BOM计数证据.json"
COUNT_CLOSURE = HERE / "单页eprj3计数闭合.json"
COUNT_REFRESHER = HERE / "更新当前BOM计数.py"
BUILDER = HERE / "单页eprj3施工.py"
PLAN = TMP / "plan.json"
PLAN_MD = TMP / "plan.md"

OLD_REFS = ["R_BAL_MAINCLR_BASE", "R_BAL_ENABLE_BASE", "R_LD", "R_PACK"]
OLD_MPN_33K = "RMC06033.3K1%N"
OLD_MPN_51K = "RC0603FR-075K1L"
MPN_1K = "RMC06031K1%N"
MPN_100R = "ABD03K1000FT"
CSV_PATHS = {
    "BOM.csv": BOM,
    "AFE元件连接.json": AFE,
    "连接表.csv": CONNECTIONS,
    "当前主板成本.csv": COST,
}
EVIDENCE_PATHS = [COUNT_EVIDENCE, COUNT_CLOSURE]
EXTERNAL_REFS = {"F1", "R_PRE", "SW_RUN_STORE", "NTC_CELL1", "NTC_CELL3", "NTC_CELL5"}
BOARD_EXCLUDED_CATEGORIES = {"维护线束", "外部控制板"}
PREFIX = re.compile(
    r"^(?:ADI|Anderson|Eaton|Infineon|KOA Speer|Littelfuse|MDD|Murata|onsemi|"
    r"Songle|TE Connectivity|TE|TI|WeAct Normal|Yageo)\s+", re.I
)

VALUE_BY_MPN = {
    MPN_1K: "1kΩ ±1%, 0603, 100mW, 50V",
    MPN_100R: "100Ω ±1%, 0603, 0.1W, 75V",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def normalize(mpn: str) -> str:
    value = PREFIX.sub("", mpn.strip())
    value = re.sub(r"\s*\(LCSC\s+[^)]*\)", "", value, flags=re.I)
    match = re.search(r"Phoenix\s+MSTBA.*\((\d+)\)", value, re.I)
    if match:
        value = match.group(1)
    if re.fullmatch(r"WSF[ ._-]*60[ ._-]*N[ ._-]*06", value, re.I):
        value = "WSF60N06"
    return value.lower()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise ValueError(f"CSV缺少表头：{path}")
    return list(reader.fieldnames), list(reader), raw


def write_csv_bytes(path: Path, fields: list[str], rows: list[dict[str, Any]], original: bytes) -> bytes:
    newline = "\r\n" if b"\r\n" in original else "\n"
    quote = csv.QUOTE_ALL if original.lstrip(codecs.BOM_UTF8).startswith(b'"') else csv.QUOTE_MINIMAL
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator=newline, quoting=quote)
    writer.writeheader()
    writer.writerows(rows)
    encoded = stream.getvalue().encode("utf-8")
    if original.startswith(codecs.BOM_UTF8):
        encoded = codecs.BOM_UTF8 + encoded
    return encoded


def write_json_bytes(value: Any, original: bytes) -> bytes:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if original.startswith(codecs.BOM_UTF8):
        encoded = codecs.BOM_UTF8 + encoded
    return encoded


def item_qty(row: dict[str, str]) -> int:
    return int(float(row.get("每套用量") or "0")) * int(float(row.get("设计套数") or "1"))


def inventory(rows: list[dict[str, str]]) -> dict[str, Any]:
    all_refs = [row["位号"].strip() for row in rows]
    all_mpns = {normalize(row["MPN"]) for row in rows if row.get("MPN", "").strip()}
    board_rows = [
        row for row in rows
        if row.get("类别") not in BOARD_EXCLUDED_CATEGORIES
        and (row.get("类别") != "板外/装配配套" or row.get("位号") == "FH_FAN")
        and row.get("位号") not in EXTERNAL_REFS
    ]
    board_refs = [row["位号"].strip() for row in board_rows]
    board_mpns = {normalize(row["MPN"]) for row in board_rows if row.get("MPN", "").strip()}
    if len(all_refs) != len(set(all_refs)) or len(board_refs) != len(set(board_refs)):
        raise ValueError("BOM位号不唯一，拒绝生成变更计划")
    return {
        "expandedProcurement": {
            "bomRows": len(rows),
            "uniqueRefs": len(set(all_refs)),
            "qty": sum(item_qty(row) for row in rows),
            "normalizedMpnTypes": len(all_mpns),
        },
        "mainBoard": {
            "uniqueRefs": len(set(board_refs)),
            "qty": sum(item_qty(row) for row in board_rows),
            "normalizedMpnTypes": len(board_mpns),
            "refs": sorted(board_refs),
        },
        "boardRows": board_rows,
    }


def role_note(old_row: dict[str, Any]) -> str:
    note = str(old_row.get("note") or old_row.get("备注") or "").strip()
    # Remove the obsolete single-part MPN statement while retaining the circuit role.
    note = re.split(r"\s+普通电阻跨板复用：", note, maxsplit=1)[0].strip()
    return note.rstrip("；;。 ")


def make_chain(ref: str, endpoint1: str, endpoint2: str, total_ohm: int, role: str) -> list[dict[str, Any]]:
    # Parent-approved physical order: all 1k parts, then the 100R parts.
    one_k_count = 3 if total_ohm == 3300 else 5
    segment_mpns = [MPN_1K] * one_k_count + [MPN_100R] * (6 - one_k_count)
    records: list[dict[str, Any]] = []
    for index, mpn in enumerate(segment_mpns, start=1):
        rref = f"{ref}_{index}"
        net1 = endpoint1 if index == 1 else f"{ref}_SER{index-1}"
        net2 = endpoint2 if index == 6 else f"{ref}_SER{index}"
        records.append({
            "ref": rref,
            "mpn": mpn,
            "pins": {"1": net1, "2": net2},
            "value": VALUE_BY_MPN[mpn],
            "note": (
                f"{role} 串联拆分链路：总标称{total_ohm}Ω；本件为第{index}/6段，"
                f"由{net1}接至{net2}。位号按原位号后缀编号；"
                f"个体参数：{VALUE_BY_MPN[mpn]}。逐器件电气/布局验收以本轮验证为准。"
            ),
        })
    return records


def staged_inputs() -> dict[str, bytes]:
    for path in (*CSV_PATHS.values(), *EVIDENCE_PATHS, ARCHIVE, COUNT_REFRESHER, BUILDER):
        if not path.is_file():
            raise FileNotFoundError(path)
    bom_fields, bom_rows, bom_raw = read_csv(BOM)
    afe_raw = AFE.read_bytes()
    afe_rows = json.loads(afe_raw.decode("utf-8-sig"))
    connection_fields, connection_rows, connection_raw = read_csv(CONNECTIONS)
    cost_fields, cost_rows, cost_raw = read_csv(COST)

    before = inventory(bom_rows)
    if before["mainBoard"]["uniqueRefs"] != 213 or before["mainBoard"]["normalizedMpnTypes"] != 60:
        raise ValueError(
            "当前正式BOM已偏离批准方案基线213件/60种；先复核当前源，不继续套用此计划。"
        )
    bom_by_ref = {row["位号"].strip(): row for row in bom_rows}
    afe_by_ref = {str(row.get("ref", "")).strip(): row for row in afe_rows if isinstance(row, dict)}
    if set(OLD_REFS) - set(bom_by_ref) or set(OLD_REFS) - set(afe_by_ref):
        raise ValueError("目标旧位号在当前BOM/AFE连接矩阵中不完整")
    if len(afe_by_ref) != len([row for row in afe_rows if isinstance(row, dict)]):
        raise ValueError("AFE连接矩阵存在重复ref")

    template_bom_by_mpn: dict[str, dict[str, str]] = {}
    for row in bom_rows:
        if row.get("MPN") in {MPN_1K, MPN_100R}:
            template_bom_by_mpn.setdefault(row["MPN"], row)
    template_afe_by_mpn: dict[str, dict[str, Any]] = {}
    for row in afe_rows:
        if isinstance(row, dict) and row.get("mpn") in {MPN_1K, MPN_100R}:
            template_afe_by_mpn.setdefault(row["mpn"], row)
    if set(template_bom_by_mpn) != {MPN_1K, MPN_100R} or set(template_afe_by_mpn) != {MPN_1K, MPN_100R}:
        raise ValueError("现有BOM/AFE中缺少可复用的1k或100Ω现行器件模板")

    new_bom_rows: list[dict[str, str]] = []
    new_afe_rows: list[dict[str, Any]] = []
    old_to_new: dict[str, list[dict[str, Any]]] = {}
    for old_ref in OLD_REFS:
        old_bom = bom_by_ref[old_ref]
        old_afe = afe_by_ref[old_ref]
        pins = old_afe.get("pins")
        if not isinstance(pins, dict) or set(pins) != {"1", "2"}:
            raise ValueError(f"{old_ref}不是预期的1/2脚串联电阻")
        expected_mpn = OLD_MPN_33K if old_ref.startswith("R_BAL_") else OLD_MPN_51K
        if old_bom.get("MPN") != expected_mpn or old_afe.get("mpn") != expected_mpn:
            raise ValueError(f"{old_ref}的MPN与预期基线不符")
        total_ohm = 3300 if expected_mpn == OLD_MPN_33K else 5100
        role = role_note({"note": old_afe.get("note"), "备注": old_bom.get("备注")})
        chain = make_chain(old_ref, str(pins["1"]), str(pins["2"]), total_ohm, role)
        old_to_new[old_ref] = chain
        for connection in chain:
            mpn = connection["mpn"]
            bom = dict(template_bom_by_mpn[mpn])
            bom.update({
                "快照日期": dt.date.today().isoformat(),
                "类别": old_bom["类别"],
                "位号": connection["ref"],
                "MPN": mpn,
                "每套用量": "1",
                "设计套数": "1",
                "总数量": "1",
                "来源JSON": "AFE元件连接.json",
                "备注": connection["note"],
                "MPN编码标称值": VALUE_BY_MPN[mpn],
            })
            new_bom_rows.append(bom)
            template_afe = dict(template_afe_by_mpn[mpn])
            # Keep only source/template metadata that belongs to this exact existing MPN.
            template_afe.update({
                "ref": connection["ref"],
                "mpn": mpn,
                "pins": connection["pins"],
                "value": VALUE_BY_MPN[mpn],
                "note": connection["note"],
            })
            new_afe_rows.append(template_afe)

    # Replace in-place to keep each BOM and connection source's broad ordering stable.
    transformed_bom: list[dict[str, str]] = []
    for row in bom_rows:
        ref = row["位号"].strip()
        if ref in old_to_new:
            transformed_bom.extend(r for r in new_bom_rows if r["位号"].startswith(ref + "_"))
        else:
            transformed_bom.append(dict(row))
    transformed_afe: list[dict[str, Any]] = []
    for row in afe_rows:
        ref = str(row.get("ref", "")).strip()
        if ref in old_to_new:
            transformed_afe.extend(r for r in new_afe_rows if r["ref"].startswith(ref + "_"))
        else:
            transformed_afe.append(row)

    # Replace just the old 8 pin rows in the consolidated connection table.
    old_connection_refs = set(OLD_REFS)
    connection_indices = [i for i, row in enumerate(connection_rows) if row.get("位号", "").strip() in old_connection_refs]
    if len(connection_indices) != 8:
        raise ValueError(f"连接表中4个旧电阻应恰有8条引脚记录，实际{len(connection_indices)}")
    first_index = min(connection_indices)
    unchanged_connections = [row for row in connection_rows if row.get("位号", "").strip() not in old_connection_refs]
    new_connection_rows: list[dict[str, str]] = []
    for ref in OLD_REFS:
        template = next(row for row in connection_rows if row.get("位号", "").strip() == ref)
        for connection in old_to_new[ref]:
            for pin, net in connection["pins"].items():
                row = dict(template)
                row.update({
                    "快照日期": dt.date.today().isoformat(),
                    "来源JSON": "AFE元件连接.json",
                    "位号": connection["ref"],
                    "MPN": connection["mpn"],
                    "引脚": pin,
                    "网络": net,
                    "备注": connection["note"],
                })
                new_connection_rows.append(row)
    # Maintain the relative order of all untouched rows; put replacement records at the earliest target row.
    connection_rows_out: list[dict[str, str]] = []
    insert_count = 0
    for i, row in enumerate(connection_rows):
        if i == first_index:
            connection_rows_out.extend(new_connection_rows)
            insert_count += len(new_connection_rows)
        if row.get("位号", "").strip() not in old_connection_refs:
            connection_rows_out.append(dict(row))
    if insert_count != 48:
        raise AssertionError("连接表replacement引脚数错误")

    after = inventory(transformed_bom)
    if after["mainBoard"]["uniqueRefs"] != 233 or after["mainBoard"]["normalizedMpnTypes"] != 58:
        raise ValueError("变换后的主板计数不是批准预期233件/58种")
    old_ids_left = [row["位号"] for row in transformed_bom if row["位号"] in OLD_REFS]
    if old_ids_left:
        raise AssertionError(f"旧位号未移除：{old_ids_left}")
    for old_ref, chain in old_to_new.items():
        refs = {row["ref"] for row in chain}
        if len(refs) != 6 or sum(1000 if row["mpn"] == MPN_1K else 100 for row in chain) != (3300 if old_ref.startswith("R_BAL_") else 5100):
            raise AssertionError(f"{old_ref}链路计数或标称等效值错误")
        for idx, row in enumerate(chain, 1):
            expect1 = old_to_new[old_ref][idx-2]["pins"]["2"] if idx > 1 else afe_by_ref[old_ref]["pins"]["1"]
            expect2 = old_to_new[old_ref][idx]["pins"]["1"] if idx < 6 else afe_by_ref[old_ref]["pins"]["2"]
            if row["pins"] != {"1": expect1, "2": expect2}:
                raise AssertionError(f"{old_ref}内部net顺序错误：{row['ref']}")

    # Refresh board quantities/ref lists in the existing cost table without inventing prices.
    cost_by_norm: dict[str, dict[str, str]] = {}
    for row in cost_rows:
        key = normalize(row["规范MPN"])
        if key in cost_by_norm:
            raise ValueError(f"成本表规范MPN重复：{row['规范MPN']}")
        cost_by_norm[key] = row
    after_board_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in after["boardRows"]:
        if row.get("MPN", "").strip():
            after_board_groups[normalize(row["MPN"])].append(row)
    missing = sorted(set(after_board_groups) - set(cost_by_norm))
    extra = sorted(set(cost_by_norm) - set(after_board_groups))
    expected_retired_cost_rows = {normalize(OLD_MPN_33K), normalize(OLD_MPN_51K)}
    if missing or set(extra) != expected_retired_cost_rows:
        raise ValueError(f"成本表MPN种类与主板BOM不一致：缺={missing}, 多={extra}")
    refreshed_cost: list[dict[str, str]] = []
    for row in cost_rows:
        key = normalize(row["规范MPN"])
        group = after_board_groups.get(key)
        if not group:
            continue
        out = dict(row)
        out["板上数量"] = str(sum(item_qty(item) for item in group))
        out["位号"] = "; ".join(sorted(item["位号"] for item in group))
        refreshed_cost.append(out)
    for index, row in enumerate(refreshed_cost, start=1):
        row["范围序号"] = str(index)

    output_bytes = {
        "BOM.csv": write_csv_bytes(BOM, bom_fields, transformed_bom, bom_raw),
        "AFE元件连接.json": write_json_bytes(transformed_afe, afe_raw),
        "连接表.csv": write_csv_bytes(CONNECTIONS, connection_fields, connection_rows_out, connection_raw),
        "当前主板成本.csv": write_csv_bytes(COST, cost_fields, refreshed_cost, cost_raw),
    }
    original_bytes = {
        "BOM.csv": bom_raw,
        "AFE元件连接.json": afe_raw,
        "连接表.csv": connection_raw,
        "当前主板成本.csv": cost_raw,
        "当前BOM计数证据.json": COUNT_EVIDENCE.read_bytes(),
        "单页eprj3计数闭合.json": COUNT_CLOSURE.read_bytes(),
    }
    input_paths = {name: path for name, path in CSV_PATHS.items()}
    input_paths.update({path.name: path for path in EVIDENCE_PATHS})
    input_hashes = {name: sha(path.read_bytes()) for name, path in input_paths.items()}
    evidence_before = json.loads(COUNT_EVIDENCE.read_text(encoding="utf-8-sig"))
    baseline_checks = {
        "countEvidenceBoard": evidence_before.get("boardMountBoundary", {}),
        "countEvidenceFull": evidence_before.get("counts", {}),
    }

    # A known-cost subtotal decrease here means only the old priced part was removed;
    # replacement 1k/100R resistors remain unpriced, so true BOM cost delta is unknown.
    def known_subtotal(rows: list[dict[str, str]]) -> float:
        total = 0.0
        for row in rows:
            raw = row.get("本板需求成本CNY（仅完整证据）", "").strip()
            if raw:
                try:
                    total += float(raw)
                except ValueError:
                    raise ValueError(f"成本表金额不可解析：{raw}")
        return total

    cost_meta = {
        "knownBoardSubtotalBeforeCNY": round(known_subtotal(cost_rows), 5),
        "knownBoardSubtotalAfterCNY": round(known_subtotal(refreshed_cost), 5),
        "knownSubtotalDeltaCNY_notActualCostDelta": round(known_subtotal(refreshed_cost) - known_subtotal(cost_rows), 5),
        "actualCostDelta": "未知：移除的Yageo 5.1kΩ原小批报价为0.82716元/颗，但替代RMC 1kΩ与ABD 100Ω均无可核验成本；不得把已证实小计减少写成成本节省。",
        "newPartPricing": {MPN_1K: "待询价", MPN_100R: "待询价"},
    }
    return {
        "payloads": output_bytes,
        "originals": original_bytes,
        "inputHashes": input_hashes,
        "before": before,
        "after": after,
        "baselineChecks": baseline_checks,
        "cost": cost_meta,
        "chains": old_to_new,
    }


def make_plan() -> dict[str, Any]:
    staged = staged_inputs()
    TMP.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, payload in staged["payloads"].items():
        stage_path = TMP / ("staged_" + name)
        stage_path.write_bytes(payload)
        if sha(stage_path.read_bytes()) != sha(payload):
            raise IOError(f"暂存读回SHA不一致：{stage_path}")
        output_hashes[name] = sha(payload)
    plan: dict[str, Any] = {
        "status": "prepared_only_no_formal_files_changed",
        "createdAt": dt.datetime.now().astimezone().isoformat(),
        "scope": "仅BOM.csv、AFE元件连接.json、连接表.csv、当前主板成本.csv及计数证据刷新；不修改原生EPRJ3、原理图工程或PCB。",
        "inputSha256": staged["inputHashes"],
        "stagedSha256": output_hashes,
        "before": staged["before"]["expandedProcurement"],
        "after": staged["after"]["expandedProcurement"],
        "boardBefore": {k: v for k, v in staged["before"]["mainBoard"].items() if k != "refs"},
        "boardAfter": {k: v for k, v in staged["after"]["mainBoard"].items() if k != "refs"},
        "boardMpnListAfter": sorted({row["MPN"].strip() for row in staged["after"]["boardRows"] if row.get("MPN", "").strip()}, key=str.casefold),
        "targetRefsRemoved": OLD_REFS,
        "targetRefsAdded": [row["ref"] for chain in staged["chains"].values() for row in chain],
        "chains": staged["chains"],
        "cost": staged["cost"],
        "baselineCountEvidence": staged["baselineChecks"],
        "archiveOnApply": str(ARCHIVE),
        "archiveMemberPrefixOnApply": "电阻网络源数据更新_时间戳/",
        "postApply": [
            "归档BOM/AFE JSON/连接表/成本表及旧计数证据、旧计数闭合并SHA读回校验。",
            "原子更新4份非原生源文件，运行更新当前BOM计数.py刷新整包与板上计数证据。",
            "旧单页eprj3计数闭合将因源摘要变化而自然失效；须在正式API更新和全量网络读回验证后重闭合。",
        ],
    }
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 已批准电阻网络源更新计划",
        "",
        "状态：`--prepare` 只生成了本目录内的暂存文件；正式BOM/连接源/CAD没有修改。",
        "",
        "| 口径 | 当前 | 计划 | 变化 |",
        "|---|---:|---:|---:|",
        f"| 全采购BOM行/位号 | {staged['before']['expandedProcurement']['bomRows']} | {staged['after']['expandedProcurement']['bomRows']} | +{staged['after']['expandedProcurement']['bomRows']-staged['before']['expandedProcurement']['bomRows']} |",
        f"| 全采购数量 | {staged['before']['expandedProcurement']['qty']} | {staged['after']['expandedProcurement']['qty']} | +{staged['after']['expandedProcurement']['qty']-staged['before']['expandedProcurement']['qty']} |",
        f"| 全采购MPN种类 | {staged['before']['expandedProcurement']['normalizedMpnTypes']} | {staged['after']['expandedProcurement']['normalizedMpnTypes']} | {staged['after']['expandedProcurement']['normalizedMpnTypes']-staged['before']['expandedProcurement']['normalizedMpnTypes']:+d} |",
        f"| 主板贴装位号/器件 | {staged['before']['mainBoard']['uniqueRefs']} | {staged['after']['mainBoard']['uniqueRefs']} | +{staged['after']['mainBoard']['uniqueRefs']-staged['before']['mainBoard']['uniqueRefs']} |",
        f"| 主板MPN种类 | {staged['before']['mainBoard']['normalizedMpnTypes']} | {staged['after']['mainBoard']['normalizedMpnTypes']} | {staged['after']['mainBoard']['normalizedMpnTypes']-staged['before']['mainBoard']['normalizedMpnTypes']:+d} |",
        "",
        "## 计划后的58种主板MPN",
        "",
        *(f"- `{mpn}`" for mpn in sorted({row["MPN"].strip() for row in staged["after"]["boardRows"] if row.get("MPN", "").strip()}, key=str.casefold)),
        "",
        "两处3.3kΩ拆成3×1kΩ+3×100Ω；R_LD、R_PACK各拆成5×1kΩ+1×100Ω。每条链按 `_1…_6` 编号，内部网络按 `_SER1…_SER5`。总计移除4个旧ref、新增24个ref，净增20个板上器件；新增MPN均已在当前BOM出现，旧3.3kΩ和Yageo 5.1kΩ料号因此消失。",
        "",
        f"成本：已证实板上小计从 {staged['cost']['knownBoardSubtotalBeforeCNY']:.5f} 元变为 {staged['cost']['knownBoardSubtotalAfterCNY']:.5f} 元，账面差 {staged['cost']['knownSubtotalDeltaCNY_notActualCostDelta']:+.5f} 元仅来自原Yageo 5.1kΩ报价项退出；新增1kΩ/100Ω无可核价，实际成本变化未知，不代表省钱。",
        "",
        "应用后会刷新BOM计数证据，但旧计数闭合摘要将过期，需在正式API更新和网络逐项读回验证后重新闭合。此脚本不触碰原生工程、PCB或EDA。",
        "",
        f"暂存SHA-256详见 [{PLAN.name}]({PLAN.name})。",
    ]
    PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".source-update.tmp")
    tmp.write_bytes(data)
    if sha(tmp.read_bytes()) != sha(data):
        raise IOError(f"临时文件读回SHA不一致：{tmp}")
    os.replace(tmp, path)


def archive_preimage(originals: dict[str, bytes]) -> str:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"历史ZIP不存在，拒绝创建另一个归档：{ARCHIVE}")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"电阻网络源数据更新_{stamp}"
    archive_items: dict[str, bytes] = {}
    for name, data in originals.items():
        archive_items[f"{prefix}/{name}"] = data
    with zipfile.ZipFile(ARCHIVE, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        existing = set(zf.namelist())
        if existing.intersection(archive_items):
            raise FileExistsError(f"归档前缀冲突：{prefix}")
        for member, data in archive_items.items():
            zf.writestr(member, data)
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        for member, data in archive_items.items():
            if sha(zf.read(member)) != sha(data):
                raise IOError(f"归档读回SHA不一致：{member}")
    return prefix


def apply_plan() -> None:
    if not PLAN.is_file():
        raise FileNotFoundError(f"未找到准备计划：{PLAN}；先执行 --prepare")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("status") != "prepared_only_no_formal_files_changed":
        raise ValueError("计划状态不符合预期")
    current_paths = dict(CSV_PATHS)
    current_paths.update({path.name: path for path in EVIDENCE_PATHS})
    current_hashes = {name: sha(path.read_bytes()) for name, path in current_paths.items()}
    if current_hashes != plan["inputSha256"]:
        changed = [name for name in current_hashes if current_hashes[name] != plan["inputSha256"].get(name)]
        raise RuntimeError(f"准备后源文件发生变化，拒绝应用；变化项={changed}。请重新 --prepare")
    payloads = {}
    for name, path in CSV_PATHS.items():
        staged = TMP / ("staged_" + name)
        if not staged.is_file():
            raise FileNotFoundError(staged)
        payload = staged.read_bytes()
        if sha(payload) != plan["stagedSha256"].get(name):
            raise RuntimeError(f"暂存内容哈希不匹配：{staged}")
        payloads[name] = payload

    originals = {name: path.read_bytes() for name, path in current_paths.items()}
    archive_prefix = archive_preimage(originals)
    try:
        for name, path in CSV_PATHS.items():
            atomic_write(path, payloads[name])
        count_result = subprocess.run(
            [sys.executable, str(COUNT_REFRESHER)],
            cwd=str(MODULE.parent), capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        if count_result.returncode != 0:
            raise RuntimeError(f"计数刷新失败：{count_result.stdout}\n{count_result.stderr}")
        refreshed = json.loads(COUNT_EVIDENCE.read_text(encoding="utf-8-sig"))
        expected = plan["boardAfter"]
        board = refreshed["boardMountBoundary"]
        if (board.get("mainPcbMountedQty"), board.get("mainPcbUniqueRefs"), board.get("mainPcbNormalizedMpnTypes")) != (
            expected["qty"], expected["uniqueRefs"], expected["normalizedMpnTypes"]
        ):
            raise RuntimeError(f"刷新后主板计数与计划不一致：{board}")
        # This invokes only the offline source parser; no EDA process/API is used.
        spec = importlib.util.spec_from_file_location("bms_builder_source_verify", BUILDER)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载单页施工输入解析器")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        entries, issues, stats = builder.source_rows()
        if issues:
            raise RuntimeError("更新源无法通过离线施工输入一致性检查：" + "；".join(issues))
        if len(entries) != expected["uniqueRefs"] or stats.get("sourceRefCount") != expected["uniqueRefs"]:
            raise RuntimeError(f"更新源ref计数与计划不一致：entries={len(entries)}, stats={stats.get('sourceRefCount')}")
    except Exception:
        # Keep the archive as recovery evidence; restore every file this operation could change.
        for name, path in current_paths.items():
            atomic_write(path, originals[name])
        raise

    print(json.dumps({
        "status": "applied_non_native_source_update",
        "archive": str(ARCHIVE),
        "archivePrefix": archive_prefix,
        "boardAfter": plan["boardAfter"],
        "expandedProcurementAfter": plan["after"],
        "countEvidence": str(COUNT_EVIDENCE),
        "offlineSourceRows": len(entries),
        "nativeProjectTouched": False,
        "note": "原计数闭合因源摘要变化已过期，需在原生API更新/全量网络读回后重闭合。",
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="只在检查/tmp生成暂存计划，不改正式源")
    mode.add_argument("--apply", action="store_true", help="应用已准备计划；校验哈希、归档并刷新计数")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(make_plan(), ensure_ascii=False, indent=2))
    else:
        apply_plan()


if __name__ == "__main__":
    main()
