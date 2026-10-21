"""Publish the saved four-resistor FAN_ILIM revision after strict read-only gates.

This script is intentionally opt-in: importing it has no side effects. Run it only
after the isolated EasyEDA export and reopened snapshot are final and coherent.
"""
from __future__ import annotations

if __name__ == '__main__':
    raise SystemExit('已撤回：用户要求精简芯片及高价值器件，不采用四串电阻方案。禁止发布219/52候选。')

import copy
import csv
import datetime as dt
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any


P = Path(__file__).resolve().parent                 # BMS模块/检查
M = P.parent                                        # BMS模块
O = P / "必要保护精简"
W = P / "tmp" / "necessary_protection_work"
N = M / "eda工程文件"
ARCHIVE = M / "归档" / "历史与调试.zip"

INITIAL_HASHES = O / "初始原生SHA256.json"
REOPENED_NET = O / "final_reopened.enet"
REOPENED_SNAPSHOT = O / "保存重开器件核对.json"
EXPORT_HASHES = O / "官方导出SHA256.json"
CANDIDATE_EPRO2 = O / "BMS-9S4P.epro2"
CANDIDATE_P1 = W / "sch" / "Schematic1" / "P1.esch2"
CANDIDATE_RAW_PDF = O / "BMS_current_raw.pdf"

CURRENT_NET = P / "BMS-9S4P.enet"
CURRENT_INPUT = P / "当前施工输入.json"
LAYOUT_MAP = P / "Layout器件映射.json"
FAN_SOURCE = M / "原理图准备" / "风扇元件连接.json"
BOM = M / "原理图准备" / "BOM.csv"

OLD_MPN = "RC0603FR-076K8L"
CHAIN = [
    ("R_FAN_ILIM", "RMC06033.3K1%N", "FAN_ILIM", "FAN_ILIM_MID", "3.3kΩ 1% 0603"),
    ("R_FAN_ILIM_B", "RMC06033.3K1%N", "FAN_ILIM_MID", "FAN_ILIM_MID2", "3.3kΩ 1% 0603"),
    ("R_FAN_ILIM_C", "ABD03K1000FT", "FAN_ILIM_MID2", "FAN_ILIM_MID3", "100Ω 1% 0603"),
    ("R_FAN_ILIM_D", "ABD03K1000FT", "FAN_ILIM_MID3", "B0", "100Ω 1% 0603"),
]
ADDED_REFS = {"R_FAN_ILIM_B", "R_FAN_ILIM_C", "R_FAN_ILIM_D"}
PREFIX = re.compile(
    r"^(?:ADI|Anderson|Eaton|Infineon|KOA Speer|Littelfuse|MDD|Murata|onsemi|"
    r"Songle|TE Connectivity|TE|TI|WeAct Normal|Yageo)\s+",
    re.I,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), raw


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".necessary_protection.tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_bytes(data)
    tmp.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_bytes_atomic(path, data)


def native_file_hashes() -> dict[str, str]:
    return {
        path.relative_to(N).as_posix(): sha(path.read_bytes())
        for path in N.rglob("*")
        if path.is_file()
    }


def as_native_path(relative: str) -> Path:
    return N.joinpath(*Path(relative.replace("/", "\\")).parts)


def component_index(netlist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for component in netlist["components"].values():
        ref = component.get("props", {}).get("Designator", "").strip()
        if not ref:
            continue
        assert ref not in result, f"duplicate designator in ENET: {ref}"
        result[ref] = component
    return result


def pinmap(netlist: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for ref, component in component_index(netlist).items():
        for key, pin in component.get("pinInfoMap", {}).items():
            number = str(pin.get("number", key))
            identity = (ref, number)
            assert identity not in result, f"duplicate pin in ENET: {identity}"
            result[identity] = pin.get("net") or ""
    return result


def ref_pinmap(netlist: dict[str, Any], ref: str) -> dict[str, str]:
    component = component_index(netlist)[ref]
    return {
        str(pin.get("number", key)): pin.get("net") or ""
        for key, pin in component.get("pinInfoMap", {}).items()
    }


def normalize_mpn(value: str) -> str:
    value = PREFIX.sub("", value.strip())
    value = re.sub(r"\s*\(LCSC\s+[^)]*\)", "", value, flags=re.I)
    phoenix = re.search(r"Phoenix\s+MSTBA.*\((\d+)\)", value, re.I)
    if phoenix:
        value = phoenix.group(1)
    if re.fullmatch(r"WSF[ ._-]*60[ ._-]*N[ ._-]*06", value, re.I):
        value = "WSF60N06"
    return value.lower()


def assert_baseline_unchanged(initial: dict[str, str]) -> None:
    current = native_file_hashes()
    assert set(current) == set(initial), "formal native file set changed since isolation"
    for relative, expected in initial.items():
        assert current[relative] == expected, f"formal native changed before publication: {relative}"


def make_pin_expectations(old_net: dict[str, Any], new_net: dict[str, Any]) -> list[dict[str, str]]:
    old_components = component_index(old_net)
    new_components = component_index(new_net)
    assert set(new_components) == set(old_components) | ADDED_REFS, "unexpected ENET component refs"
    assert old_components["R_FAN_ILIM"]["props"].get("Manufacturer Part") == OLD_MPN

    old_pins = pinmap(old_net)
    new_pins = pinmap(new_net)
    expected_keys = set(old_pins)
    expected_keys.update((ref, pin) for ref in ADDED_REFS for pin in ("1", "2"))
    assert set(new_pins) == expected_keys, "unexpected ENET pin addition/removal"

    expected_change = {("R_FAN_ILIM", "2"): ("B0", "FAN_ILIM_MID")}
    changes: list[dict[str, str]] = []
    for key, old_value in old_pins.items():
        new_value = new_pins[key]
        if old_value != new_value:
            changes.append({"ref": key[0], "pin": key[1], "from": old_value, "to": new_value})
    assert {
        (row["ref"], row["pin"]): (row["from"], row["to"]) for row in changes
    } == expected_change, f"unexpected ENET pin changes: {changes}"

    expected_by_ref = {
        ref: {"1": net_a, "2": net_b}
        for ref, _mpn, net_a, net_b, _value in CHAIN
    }
    for ref, expected in expected_by_ref.items():
        assert ref_pinmap(new_net, ref) == expected, f"unexpected series-chain nets for {ref}"
    assert ref_pinmap(old_net, "C_BAT2") == {"1": "BAT", "2": "B0"}
    assert ref_pinmap(new_net, "C_BAT2") == ref_pinmap(old_net, "C_BAT2"), "C_BAT2 nets changed"

    expected_mpns = {ref: mpn for ref, mpn, *_ in CHAIN}
    for ref, expected_mpn in expected_mpns.items():
        actual = new_components[ref]["props"].get("Manufacturer Part", "")
        assert actual == expected_mpn, f"wrong official MPN for {ref}: {actual}"
    return changes


def validate_snapshot(snapshot: dict[str, Any], new_net: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = snapshot["rows"]
    by_ref = {row["ref"]: row for row in rows}
    assert len(by_ref) == len(rows) == 220, f"expected 220 unique snapshot rows, got {len(rows)}"
    assert set(by_ref) == set(component_index(new_net)), "snapshot/official ENET refs differ"
    assert sum(len(row.get("pins", [])) for row in rows) == 729, "snapshot pin count is not 729"
    assert sum(bool(row.get("pcb")) for row in rows) == 219, "expected 219 PCB components"
    mpn_types = {
        normalize_mpn(row.get("mpn", ""))
        for row in rows
        if row.get("pcb") and row.get("mpn", "").strip() and row["ref"] != "CORE_BOARD"
    }
    assert len(mpn_types) == 52, f"expected 52 normalized MPN types, got {len(mpn_types)}"
    assert snapshot.get("drc") == [{"type": "warn", "count": 8}], f"unexpected DRC result: {snapshot.get('drc')}"

    components = component_index(new_net)
    for ref, snap in by_ref.items():
        if ref in {row[0] for row in CHAIN} | {"C_BAT2"}:
            props = components[ref]["props"]
            assert snap.get("mpn", "") == props.get("Manufacturer Part", ""), f"snapshot MPN mismatch: {ref}"
            assert snap.get("component", {}).get("uuid") == props.get("Device"), f"snapshot Device mismatch: {ref}"
    return by_ref


def source_note(mpn: str, ref: str, start: str, end: str) -> str:
    series = "R_FAN_ILIM → R_FAN_ILIM_B → R_FAN_ILIM_C → R_FAN_ILIM_D"
    if mpn == "RMC06033.3K1%N":
        vendor = "Tyohm幸亚 RMC06033.3K1%N，3.3kΩ ±1%、0603、0.1W、50V、独立两脚，非排阻。"
    else:
        vendor = "风华 ABD03K1000FT（立创C20616261，100Ω ±1%、100mW、75V）。"
    return (
        f"{vendor}用于SGM2593 FAN_ILIM电流限制串联链，四件标称总阻值6.8kΩ；"
        f"{series}。本段 {ref} 接 {start} 至 {end}。"
    )


def create_source_rows(
    old_row: dict[str, Any], snapshots: dict[str, dict[str, Any]], net: dict[str, Any]
) -> list[dict[str, Any]]:
    components = component_index(net)
    output = []
    base_x = old_row.get("x", old_row.get("xy", {}).get("x", 1250))
    base_y = old_row.get("y", old_row.get("xy", {}).get("y", 1100))
    for index, (ref, mpn, net_a, net_b, value) in enumerate(CHAIN):
        snap = snapshots[ref]
        source = copy.deepcopy(old_row)
        note = source_note(mpn, ref, net_a, net_b)
        xy = {"x": base_x, "y": base_y + index * 100}
        source.update(
            {
                "ref": ref,
                "mpn": mpn,
                "libUuid": snap["component"]["uuid"],
                "libraryUuid": snap["component"]["libraryUuid"],
                "pins": ref_pinmap(net, ref),
                "value": value,
                "note": note,
                "sourceFile": "风扇元件连接.json",
                "x": xy["x"],
                "y": xy["y"],
                "xy": xy,
                "addIntoPcb": bool(snap["pcb"]),
                "addIntoBom": bool(snap["bom"]),
                "keepExisting": False,
                "unresolved": [],
                "expectedFootprintName": snap["footprint"]["name"],
                "libraryKey": mpn,
            }
        )
        output.append(source)
    return output


def update_component_source(
    rows: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]], net: dict[str, Any]
) -> list[dict[str, Any]]:
    matches = [i for i, row in enumerate(rows) if row.get("ref") == "R_FAN_ILIM"]
    assert len(matches) == 1, "fan source must contain exactly one original R_FAN_ILIM row"
    old = rows[matches[0]]
    assert old.get("mpn") == OLD_MPN and old.get("pins") == {"1": "FAN_ILIM", "2": "B0"}
    assert not ({row.get("ref") for row in rows} & ADDED_REFS), "fan source already contains new refs"
    replacements = create_source_rows(old, snapshots, net)
    out = copy.deepcopy(rows[: matches[0]]) + replacements + copy.deepcopy(rows[matches[0] + 1 :])
    assert len({row.get("ref") for row in out}) == len(out)
    return out


def update_expected_input(
    rows: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]], net: dict[str, Any]
) -> list[dict[str, Any]]:
    matches = [i for i, row in enumerate(rows) if row.get("ref") == "R_FAN_ILIM"]
    assert len(matches) == 1, "current construction input must contain one R_FAN_ILIM row"
    old = rows[matches[0]]
    assert old.get("mpn") == OLD_MPN and old.get("pins") == {"1": "FAN_ILIM", "2": "B0"}
    assert not ({row.get("ref") for row in rows} & ADDED_REFS), "construction input already contains new refs"
    replacements = []
    for ref, mpn, _net_a, _net_b, _value in CHAIN:
        replacements.append(
            {
                "ref": ref,
                "mpn": mpn,
                "pins": ref_pinmap(net, ref),
                "addIntoPcb": bool(snapshots[ref]["pcb"]),
            }
        )
    out = copy.deepcopy(rows[: matches[0]]) + replacements + copy.deepcopy(rows[matches[0] + 1 :])
    assert len({row.get("ref") for row in out}) == len(out)
    return out


def fresh_map_row(
    base: dict[str, Any], ref: str, mpn: str, note: str,
    snapshot: dict[str, Any], component: dict[str, Any], net: dict[str, Any],
) -> dict[str, Any]:
    row = copy.deepcopy(base)
    row.update(
        {
            "ref": ref,
            "mpn": mpn,
            "pins": ref_pinmap(net, ref),
            "addIntoPcb": bool(snapshot["pcb"]),
            "primitiveId": snapshot["id"],
            "projectLocalDeviceId": component["props"]["Device"],
            "symbol": copy.deepcopy(snapshot["symbol"]),
            "footprint": copy.deepcopy(snapshot["footprint"]),
            "xy": {"x": snapshot["x"], "y": snapshot["y"]},
            "addIntoBom": bool(snapshot["bom"]),
            "carrierNote": note,
        }
    )
    return row


def update_layout_map(
    data: dict[str, Any], snapshots: dict[str, dict[str, Any]], net: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = data["components"]
    matches = [i for i, row in enumerate(rows) if row.get("ref") == "R_FAN_ILIM"]
    assert len(matches) == 1, "layout map must contain exactly one original R_FAN_ILIM row"
    assert "C_BAT2" in {row.get("ref") for row in rows}, "C_BAT2 mapping is missing"
    assert not ({row.get("ref") for row in rows} & ADDED_REFS), "layout map already contains new refs"
    old_fan = rows[matches[0]]
    assert old_fan.get("mpn") == OLD_MPN and old_fan.get("pins") == {"1": "FAN_ILIM", "2": "B0"}
    old_bat2 = next(row for row in rows if row.get("ref") == "C_BAT2")
    assert old_bat2.get("pins") == {"1": "BAT", "2": "B0"}, "C_BAT2 mapping changed before publication"

    components = component_index(net)
    source_by_ref = {row["ref"]: row for row in source_rows}
    chain_rows = [
        fresh_map_row(
            old_fan, ref, mpn, source_by_ref[ref]["note"], snapshots[ref], components[ref], net
        )
        for ref, mpn, *_ in CHAIN
    ]

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        ref = row.get("ref")
        if ref == "R_FAN_ILIM":
            out_rows.extend(chain_rows)
        else:
            refreshed = copy.deepcopy(row)
            assert ref in snapshots, f"layout ref missing from snapshot: {ref}"
            # Primitive IDs are session-local; refresh them from this exact reopened readback.
            refreshed["primitiveId"] = snapshots[ref]["id"]
            if ref == "C_BAT2":
                assert ref_pinmap(net, ref) == {"1": "BAT", "2": "B0"}
                refreshed["mpn"] = snapshots[ref]["mpn"]
                refreshed["projectLocalDeviceId"] = components[ref]["props"]["Device"]
                refreshed["symbol"] = copy.deepcopy(snapshots[ref]["symbol"])
                refreshed["footprint"] = copy.deepcopy(snapshots[ref]["footprint"])
                refreshed["xy"] = {"x": snapshots[ref]["x"], "y": snapshots[ref]["y"]}
                refreshed["pins"] = ref_pinmap(net, ref)
                refreshed["addIntoPcb"] = bool(snapshots[ref]["pcb"])
                refreshed["addIntoBom"] = bool(snapshots[ref]["bom"])
            out_rows.append(refreshed)

    out = copy.deepcopy(data)
    out["components"] = out_rows
    out.setdefault("inputs", {})["independentExpected"] = "当前施工输入.json"
    out["inputs"]["officialReopenedEnet"] = "必要保护精简/final_reopened.enet"
    assert len({row.get("ref") for row in out_rows}) == len(out_rows)
    assert set(row.get("ref") for row in out_rows) == set(snapshots)
    return out


def bom_mpn_description(template: dict[str, str], mpn: str) -> str:
    note = template["备注"]
    if mpn == "RMC06033.3K1%N":
        start = note.find("普通电阻跨板复用：")
        end = note.find("沿用原阻值", start)
        assert start >= 0 and end > start, "3.3k BOM template lacks manufacturer/spec metadata"
        return note[start + len("普通电阻跨板复用：") : end].strip()
    # R_BAT's first sentence carries Fenghua's exact model and supplier catalog number.
    end = note.find("。")
    assert "ABD03K1000FT" in note and end > 0, "100-ohm BOM template lacks exact part metadata"
    return note[: end + 1]


def update_bom(text: str, has_utf8_bom: bool) -> tuple[bytes, list[dict[str, str]]]:
    bom_rows = list(csv.DictReader(io.StringIO(text, newline="")))
    assert bom_rows and "位号" in bom_rows[0] and "MPN" in bom_rows[0]
    headers = list(bom_rows[0])
    old_matches = [row for row in bom_rows if row["位号"] == "R_FAN_ILIM"]
    assert len(old_matches) == 1 and old_matches[0]["MPN"] == OLD_MPN
    assert not ({row["位号"] for row in bom_rows} & ADDED_REFS), "BOM already contains new fan refs"
    old_fan = old_matches[0]
    templates: dict[str, dict[str, str]] = {}
    for mpn, template_ref in (("RMC06033.3K1%N", "R_BAL_ENABLE_BASE"), ("ABD03K1000FT", "R_BAT")):
        found = [row for row in bom_rows if row["位号"] == template_ref and row["MPN"] == mpn]
        assert len(found) == 1, f"missing exact-MPN BOM template {template_ref}/{mpn}"
        templates[mpn] = found[0]

    fan_refs = {row[0] for row in CHAIN}
    result: list[dict[str, str]] = []
    new_rows: list[dict[str, str]] = []
    for ref, mpn, _net_a, _net_b, _value in CHAIN:
        source = templates[mpn]
        row = copy.deepcopy(source)
        row["类别"] = old_fan["类别"]
        row["位号"] = ref
        row["MPN"] = mpn
        row["每套用量"] = "1"
        row["设计套数"] = "1"
        row["总数量"] = "1"
        row["来源JSON"] = "风扇元件连接.json"
        vendor_spec = bom_mpn_description(source, mpn)
        row["备注"] = (
            f"{vendor_spec}用于SGM2593 FAN_ILIM串联限流链；四件标称合计6.8kΩ，"
            "分别串联于 FAN_ILIM → FAN_ILIM_MID → FAN_ILIM_MID2 → FAN_ILIM_MID3 → B0。"
        )
        new_rows.append(row)

    inserted = False
    for row in bom_rows:
        if row["位号"] in fan_refs:
            if row["位号"] == "R_FAN_ILIM":
                result.extend(new_rows)
                inserted = True
            continue
        result.append(row)
    assert inserted
    assert len({row["位号"] for row in result}) == len(result)
    assert len(result) == len(bom_rows) + 3
    normalized_types = {
        normalize_mpn(row["MPN"])
        for row in result
        if row["MPN"].strip() and row["位号"] != "CORE_BOARD"
    }
    assert len(normalized_types) == 52, f"post-change BOM has {len(normalized_types)} normalized MPN types"

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="raise", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(result)
    prefix = "\ufeff" if has_utf8_bom else ""
    payload = (prefix + buffer.getvalue()).encode("utf-8")
    return payload, new_rows


def archive_targets() -> list[Path]:
    paths: set[Path] = {path for path in N.rglob("*") if path.is_file()}
    paths.update(
        {
            CURRENT_INPUT, LAYOUT_MAP, CURRENT_NET, FAN_SOURCE, BOM,
            M / "原理图准备" / "连接表.csv",
            M / "README.md", M / "TODO.md", M / "项目要求.md",
            P / "当前设计状态.md", P / "保护精简落实审查.json",
            P / "本轮落实与验证.json", P / "本轮落实与验证.md",
            P / "本轮直接减件落实.json", P / "当前BOM边界清单.md",
            P / "当前原理图导航审阅.json", P / "当前原理图导航审阅.pdf",
            P / "当前BMS9S4P官方网表.enet", P / "当前官方网表.json",
            P / "当前BMS9S4P网表连接核对.json", P / "当前网表API.json",
            P / "当前原生器件清单.json", P / "当前BOM计数证据.json",
            P / "当前主板成本.csv", P / "当前主板成本.md",
            P / "官方网表连接核对.json", P / "官方网表连接核对.md",
            P / "当前BMS9S4P原生工程保存证据.json",
            P / "DRC逐项核对.json", P / "DRC逐项核对.md",
            O / "初始原生SHA256.json", O / "官方导出SHA256.json",
            O / "保存重开器件核对.json", O / "final_reopened.enet",
            O / "BMS-9S4P.epro2", O / "BMS_current_raw.pdf",
            W / "sch" / "Schematic1" / "P1.esch2",
            M / "仿真" / "输入" / "保护精简审查输入.json",
            M / "仿真" / "结果" / "保护精简复算.json",
            M / "仿真" / "输入" / "fan_urb_startup_20261015.json",
            M / "仿真" / "源码" / "verify_protection_reduction.py",
            M / "仿真" / "源码" / "verify_fan_urb_startup_20261015.py",
            M / "仿真" / "结果" / "fan_urb_startup_20261015_model.json",
        }
    )
    # Include all current evidence variants, especially counters, netlists and PDFs.
    paths.update(path for path in P.glob("当前*") if path.is_file())
    paths.update(path for path in (P / "必要保护精简").glob("*.json") if path.is_file())
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    initial, _ = read_json(INITIAL_HASHES)
    assert_baseline_unchanged(initial)

    old_net, old_net_raw = read_json(CURRENT_NET)
    new_net, new_net_raw = read_json(REOPENED_NET)
    snapshot, snapshot_raw = read_json(REOPENED_SNAPSHOT)
    export_hashes, export_hashes_raw = read_json(EXPORT_HASHES)
    assert make_pin_expectations(old_net, new_net) == [
        {"ref": "R_FAN_ILIM", "pin": "2", "from": "B0", "to": "FAN_ILIM_MID"}
    ]
    snapshots = validate_snapshot(snapshot, new_net)

    exported_epro2 = next((row for row in export_hashes if row.get("path") == CANDIDATE_EPRO2.name), None)
    exported_pdf = next((row for row in export_hashes if row.get("path") == CANDIDATE_RAW_PDF.name), None)
    assert exported_epro2 and exported_pdf, "official export manifest lacks epro2/PDF entries"
    epro2_raw = CANDIDATE_EPRO2.read_bytes()
    pdf_raw = CANDIDATE_RAW_PDF.read_bytes()
    p1_raw = CANDIDATE_P1.read_bytes()
    assert sha(epro2_raw) == exported_epro2["sha256"], "candidate epro2 differs from official export hash"
    assert sha(pdf_raw) == exported_pdf["sha256"], "candidate PDF differs from official export hash"
    assert CANDIDATE_P1.is_file() and CANDIDATE_EPRO2.is_file(), "official native candidates are missing"
    assert p1_raw, "candidate P1 is empty"

    fan_rows, fan_raw = read_json(FAN_SOURCE)
    input_rows, input_raw = read_json(CURRENT_INPUT)
    layout, layout_raw = read_json(LAYOUT_MAP)
    bom_raw = BOM.read_bytes()
    bom_text = bom_raw.decode("utf-8-sig")
    fan_updated = update_component_source(fan_rows, snapshots, new_net)
    input_updated = update_expected_input(input_rows, snapshots, new_net)
    layout_updated = update_layout_map(layout, snapshots, new_net, fan_updated)
    bom_updated, bom_rows = update_bom(bom_text, bom_raw.startswith(b"\xef\xbb\xbf"))

    # Check all candidate inputs again immediately before creating the recovery archive.
    candidate_digests = {
        REOPENED_NET: sha(new_net_raw),
        REOPENED_SNAPSHOT: sha(snapshot_raw),
        EXPORT_HASHES: sha(export_hashes_raw),
        CANDIDATE_EPRO2: sha(epro2_raw),
        CANDIDATE_RAW_PDF: sha(pdf_raw),
        CANDIDATE_P1: sha(p1_raw),
    }
    for path, expected in candidate_digests.items():
        assert sha(path.read_bytes()) == expected, f"candidate export changed during preflight: {path.name}"
    assert_baseline_unchanged(initial)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_必要保护精简发布前"
    records: list[dict[str, str]] = []
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    files = archive_targets()
    with zipfile.ZipFile(ARCHIVE, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing_names = set(archive.namelist())
        for path in files:
            entry = stamp + "/" + path.relative_to(M).as_posix()
            assert entry not in existing_names, f"archive entry already exists: {entry}"
            raw = path.read_bytes()
            archive.writestr(entry, raw)
            records.append({"path": path.relative_to(M).as_posix(), "entry": entry, "sha256": sha(raw)})
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        for row in records:
            archived = archive.read(row["entry"])
            source = M / Path(row["path"].replace("/", "\\"))
            assert sha(archived) == row["sha256"] == sha(source.read_bytes()), row["path"]

    # All validation and archive readback succeeded; now update the small source set and native files.
    assert sha(FAN_SOURCE.read_bytes()) == sha(fan_raw), "fan source changed after preflight"
    assert sha(CURRENT_INPUT.read_bytes()) == sha(input_raw), "construction input changed after preflight"
    assert sha(LAYOUT_MAP.read_bytes()) == sha(layout_raw), "layout map changed after preflight"
    assert sha(BOM.read_bytes()) == sha(bom_raw), "BOM changed after preflight"
    for path, expected in candidate_digests.items():
        assert sha(path.read_bytes()) == expected, f"candidate export changed before publish: {path.name}"
    assert_baseline_unchanged(initial)
    write_json_atomic(FAN_SOURCE, fan_updated)
    write_json_atomic(CURRENT_INPUT, input_updated)
    write_json_atomic(LAYOUT_MAP, layout_updated)
    write_bytes_atomic(BOM, bom_updated)
    write_bytes_atomic(CURRENT_NET, new_net_raw)
    write_bytes_atomic(as_native_path("sch/Schematic1/P1.esch2"), p1_raw)
    write_bytes_atomic(N / "BMS-9S4P.epro2", epro2_raw)

    assert sha((N / "sch" / "Schematic1" / "P1.esch2").read_bytes()) == sha(p1_raw)
    assert sha((N / "BMS-9S4P.epro2").read_bytes()) == sha(epro2_raw)
    assert_baseline_unchanged({
        key: value for key, value in initial.items()
        if key not in {"sch/Schematic1/P1.esch2", "BMS-9S4P.epro2"}
    } | {
        key: sha((N / key.replace("/", "\\")).read_bytes())
        for key in ("sch/Schematic1/P1.esch2", "BMS-9S4P.epro2")
    })

    published = [
        {"file": path.relative_to(N).as_posix(), "sha256": sha(path.read_bytes())}
        for path in N.rglob("*") if path.is_file()
    ]
    write_json_atomic(O / "原生发布校验.json", published)
    write_json_atomic(
        O / "发布归档记录.json",
        {
            "archive": "归档/历史与调试.zip",
            "archiveReadbackSHA256Verified": True,
            "archiveEntries": records,
            "pinChanges": [{"ref": "R_FAN_ILIM", "pin": "2", "from": "B0", "to": "FAN_ILIM_MID"}],
            "addedRefs": sorted(ADDED_REFS),
            "C_BAT2PinsPreserved": True,
            "BOMRows": [row["位号"] for row in bom_rows],
            "publishedNativeSHA256": published,
            "scope": "仅FAN_ILIM四电阻串联实现、对应输入/映射/BOM及官方保存重开原生文件",
        },
    )
    print(json.dumps({
        "archiveReadback": "PASS",
        "addedRefs": sorted(ADDED_REFS),
        "pinChange": "R_FAN_ILIM.2: B0 -> FAN_ILIM_MID",
        "snapshot": {"components": 220, "pins": 729, "pcb": 219, "normalizedMpnTypes": 52, "drcWarnings": 8},
        "updatedFiles": [str(FAN_SOURCE), str(CURRENT_INPUT), str(LAYOUT_MAP), str(BOM), str(CURRENT_NET), str(N)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
