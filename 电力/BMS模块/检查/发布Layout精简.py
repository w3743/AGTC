"""Publish the saved Layout fan-rail simplification after strict read-only gates.

This is an opt-in release script. Importing it does not publish anything.
It refuses to run until the official EPRO2 export and its SHA256 manifest
have been placed in Layout收尾. It archives and verifies the pre-publication
state before updating source inputs, BOM, official ENET, and the two authorized
native files (EPRO2 and P1). It deliberately does not generate or replace PDF.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


P = Path(__file__).resolve().parent                  # BMS模块/检查
M = P.parent                                         # BMS模块
O = P / "Layout收尾"
W = P / "tmp" / "layout_handoff_work"
N = M / "eda工程文件"
ARCHIVE = M / "归档" / "历史与调试.zip"

INITIAL_HASHES = O / "初始原生SHA256.json"
REOPENED_NET = O / "final_reopened.enet"
REOPENED_SNAPSHOT = O / "保存重开器件核对.json"
EXPORT_HASHES = O / "官方导出SHA256.json"
CANDIDATE_EPRO2 = O / "BMS-9S4P.epro2"
CANDIDATE_P1 = W / "sch" / "Schematic1" / "P1.esch2"

CURRENT_NET = P / "BMS-9S4P.enet"
CURRENT_INPUT = P / "当前施工输入.json"
LAYOUT_MAP = P / "Layout器件映射.json"
FAN_SOURCE = M / "原理图准备" / "风扇元件连接.json"
POWER_SOURCE = M / "原理图准备" / "功率元件连接.json"
BOM = M / "原理图准备" / "BOM.csv"
CURRENT_PDF = P / "当前原理图导航审阅.pdf"

REMOVED_REFS = {"U_FAN_SW", "R_FAN_ILIM", "C_FAN_IN1", "C_FAN_OUT1"}
ALL_FAN_REFS = REMOVED_REFS | {"J_FAN"}
EXPECTED_MPN = {
    "U_FAN_SW": "SGM2593XN6G/TR",
    "R_FAN_ILIM": "RC0603FR-076K8L",
    "C_FAN_IN1": "GRM32ER61C476KE15L",
    "C_FAN_OUT1": "GRM32ER61C476KE15L",
    "J_FAN": "2EDGVC-5.08-02P-14-100A(H)",
}
EXPECTED_OLD_FAN_PINS = {"1": "FAN_5V", "2": "B0"}
EXPECTED_NEW_FAN_PINS = {"1": "VOUT_5V", "2": "B0"}
EXPECTED_DRC = [{"type": "warn", "count": 8}]
EXPECTED_COUNTS = {"cad": 213, "pcb": 212, "mpnTypes": 51, "pins": 711}
REVIEWED_ENET_HASHES = {
    "baseline": "065d9137fad8a4235e32cf2870c0e04a5dc9b46e2cefe5a249f8286585c6b65c",
    "candidate": "01761c55a731d71857dcc2b77643c4d5d34bb956ecc57378a190ed55d4cb65ec",
}
EXPECTED_CHANNEL_ID_CHANGES = 178
EXPECTED_CHANNEL_ID_DELTA_HISTOGRAM = {2: 92, 3: 60, 10: 17, 16: 9}

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
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".layout-release.tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_bytes_atomic(path, payload)


def relative_to_module(path: Path) -> str:
    return path.relative_to(M).as_posix()


def native_file_hashes() -> dict[str, str]:
    return {
        path.relative_to(N).as_posix(): sha(path.read_bytes())
        for path in N.rglob("*")
        if path.is_file()
    }


def assert_native_baseline(initial: dict[str, str]) -> None:
    current = native_file_hashes()
    assert set(current) == set(initial), "formal native file set changed since Layout isolation"
    for relative, expected in initial.items():
        assert current[relative] == expected, f"formal native changed before publication: {relative}"


def component_index(netlist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for component in netlist["components"].values():
        ref = component.get("props", {}).get("Designator", "").strip()
        if not ref:
            continue
        assert ref not in result, f"duplicate ENET designator: {ref}"
        result[ref] = component
    return result


def pinmap(netlist: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for ref, component in component_index(netlist).items():
        for key, pin in component.get("pinInfoMap", {}).items():
            number = str(pin.get("number", key))
            identity = (ref, number)
            assert identity not in result, f"duplicate ENET pin: {identity}"
            result[identity] = pin.get("net") or ""
    return result


def normalize_mpn(value: str) -> str:
    value = PREFIX.sub("", value.strip())
    value = re.sub(r"\s*\(LCSC\s+[^)]*\)", "", value, flags=re.I)
    phoenix = re.search(r"Phoenix\s+MSTBA.*\((\d+)\)", value, re.I)
    if phoenix:
        value = phoenix.group(1)
    if re.fullmatch(r"WSF[ ._-]*60[ ._-]*N[ ._-]*06", value, re.I):
        value = "WSF60N06"
    return value.lower()


def normalized_pins(pins: dict[str, Any]) -> dict[str, str]:
    return {str(number): value or "" for number, value in pins.items()}


def component_refs(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        ref = row.get("ref")
        assert isinstance(ref, str) and ref, f"{label} has a row without ref"
        assert ref not in result, f"duplicate {label} ref: {ref}"
        result[ref] = row
    return result


def candidate_rows(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    # Layout收尾's official API capture stores rows/drc at the root. Accept the
    # older wrapped API shape only because baseline captures used that format.
    payload = snapshot.get("value", snapshot)
    rows = payload.get("rows")
    assert isinstance(rows, list), "saved/reopened snapshot lacks a rows array"
    return rows, payload.get("drc")


def assert_net_delta(
    old_net: dict[str, Any], new_net: dict[str, Any]
) -> dict[str, list[dict[str, str]]]:
    assert set(old_net) == set(new_net), "official ENET top-level schema changed"
    assert isinstance(old_net.get("projectId"), str) and old_net["projectId"]
    assert isinstance(new_net.get("projectId"), str) and new_net["projectId"]
    # The official reopened export regenerates projectId. Apart from that
    # export identity, preserve every top-level rule/group object except the
    # two stale rules for nets made unused by the authorized fan removals.
    for key in set(old_net) - {"components", "projectId", "designRule"}:
        assert old_net[key] == new_net[key], f"unexpected ENET metadata change: {key}"
    old_rules = old_net.get("designRule", {})
    new_rules = new_net.get("designRule", {})
    assert set(old_rules) == set(new_rules), "ENET design-rule schema changed"
    for key in set(old_rules) - {"netRule"}:
        assert old_rules[key] == new_rules[key], f"unexpected ENET design-rule change: {key}"
    old_net_rules = old_rules.get("netRule", {})
    new_net_rules = new_rules.get("netRule", {})
    assert set(old_net_rules) - set(new_net_rules) == {"FAN_5V", "FAN_ILIM"}, (
        "official ENET removed unexpected net rules or retained stale fan net rules"
    )
    assert not (set(new_net_rules) - set(old_net_rules)), "official ENET added net rules"
    for name in new_net_rules:
        assert old_net_rules[name] == new_net_rules[name], f"unexpected ENET net-rule change: {name}"

    old_components = component_index(old_net)
    new_components = component_index(new_net)
    assert set(old_net["components"]) - REMOVED_REFS == set(new_net["components"]), (
        "official ENET component storage keys differ from the exact authorized deletions"
    )
    assert set(new_components) == set(old_components) - REMOVED_REFS, (
        "official ENET component refs differ from the exact four authorized deletions"
    )
    old_pins = pinmap(old_net)
    new_pins = pinmap(new_net)
    assert len(old_components) == 217 and len(old_pins) == 723, (
        f"unexpected formal ENET baseline: {len(old_components)} components/{len(old_pins)} pins"
    )
    removed_pin_keys = {key for key in old_pins if key[0] in REMOVED_REFS}
    assert set(old_pins) - set(new_pins) == removed_pin_keys, "unexpected ENET pin additions/removals"
    assert len(new_components) == EXPECTED_COUNTS["cad"]
    assert len(new_pins) == EXPECTED_COUNTS["pins"]

    changes: list[dict[str, str]] = []
    channel_id_changes: list[dict[str, str]] = []
    for key, old_net_name in old_pins.items():
        if key[0] in REMOVED_REFS:
            continue
        assert key in new_pins, f"retained ENET pin disappeared: {key}"
        new_net_name = new_pins[key]
        if old_net_name != new_net_name:
            changes.append({"ref": key[0], "pin": key[1], "from": old_net_name, "to": new_net_name})
    expected_changes = [{"ref": "J_FAN", "pin": "1", "from": "FAN_5V", "to": "VOUT_5V"}]
    assert changes == expected_changes, f"unexpected official ENET net changes: {changes}"

    # A strict ENET comparison allows only the net labels to differ on surviving
    # pins. Component properties, pin identities, and all other pin metadata stay fixed.
    for ref in new_components:
        before = old_components[ref]
        after = new_components[ref]
        before_props = copy.deepcopy(before.get("props", {}))
        after_props = copy.deepcopy(after.get("props", {}))
        old_channel_id = before_props.pop("Channel ID", None)
        new_channel_id = after_props.pop("Channel ID", None)
        if old_channel_id != new_channel_id:
            assert isinstance(old_channel_id, str) and re.fullmatch(r"\$1I\d+", old_channel_id), (
                f"unexpected old Channel ID format on {ref}: {old_channel_id!r}"
            )
            assert isinstance(new_channel_id, str) and re.fullmatch(r"\$1I\d+", new_channel_id), (
                f"unexpected new Channel ID format on {ref}: {new_channel_id!r}"
            )
            channel_id_changes.append({
                "ref": ref, "from": old_channel_id, "to": new_channel_id
            })
        assert before_props == after_props, f"unexpected non-identity ENET property change on {ref}"
        assert {k: v for k, v in before.items() if k not in {"pinInfoMap", "props"}} == {
            k: v for k, v in after.items() if k not in {"pinInfoMap", "props"}
        }, f"unexpected non-pin ENET change on {ref}"
        before_pins = before.get("pinInfoMap", {})
        after_pins = after.get("pinInfoMap", {})
        assert set(before_pins) == set(after_pins), f"pin identities changed on {ref}"
        for pin_id in before_pins:
            assert {
                k: v for k, v in before_pins[pin_id].items() if k != "net"
            } == {
                k: v for k, v in after_pins[pin_id].items() if k != "net"
            }, f"non-net pin data changed on {ref}.{pin_id}"

    assert pinmap(new_net).get(("J_FAN", "1")) == "VOUT_5V"
    assert pinmap(new_net).get(("J_FAN", "2")) == "B0"
    assert len(channel_id_changes) == EXPECTED_CHANNEL_ID_CHANGES, (
        f"unexpected Channel ID renumbering count: {len(channel_id_changes)}"
    )
    channel_id_delta_histogram = Counter(
        int(row["from"][3:]) - int(row["to"][3:]) for row in channel_id_changes
    )
    assert channel_id_delta_histogram == EXPECTED_CHANNEL_ID_DELTA_HISTOGRAM, (
        f"unexpected Channel ID renumbering pattern: {channel_id_delta_histogram}"
    )
    return {"pinChanges": changes, "channelIdChanges": channel_id_changes}


def assert_snapshot(snapshot: dict[str, Any], new_net: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows, drc = candidate_rows(snapshot)
    by_ref = component_refs(rows, "official reopened snapshot")
    enet_refs = set(component_index(new_net))
    assert set(by_ref) == enet_refs, "saved/reopened snapshot refs do not match official ENET"
    pins = sum(len(row.get("pins", [])) for row in rows)
    pcb_rows = [row for row in rows if row.get("pcb")]
    mpn_types = {
        normalize_mpn(row.get("mpn", ""))
        for row in pcb_rows
        if row.get("mpn", "").strip() and row.get("ref") != "CORE_BOARD"
    }
    actual = {
        "cad": len(rows),
        "pcb": len(pcb_rows),
        "mpnTypes": len(mpn_types),
        "pins": pins,
    }
    assert actual == EXPECTED_COUNTS, f"official reopened counts differ: {actual}"
    assert drc == EXPECTED_DRC, f"unexpected official reopened DRC summary: {drc}"
    return by_ref


def assert_source_alignment(
    input_rows: list[dict[str, Any]],
    fan_rows: list[dict[str, Any]],
    layout: dict[str, Any],
    net: dict[str, Any],
    snapshot_rows: dict[str, dict[str, Any]] | None = None,
) -> None:
    net_components = component_index(net)
    expected_refs = set(snapshot_rows) if snapshot_rows is not None else set(net_components)
    input_by_ref = component_refs(input_rows, "current construction input")
    layout_by_ref = component_refs(layout["components"], "layout map")
    assert set(input_by_ref) == set(layout_by_ref) == expected_refs == set(net_components), (
        "current input/layout refs do not match the selected ENET/snapshot"
    )
    pins = pinmap(net)
    for ref, row in input_by_ref.items():
        snapshot = snapshot_rows.get(ref) if snapshot_rows is not None else None
        expected_mpn = (
            snapshot.get("mpn", "") if snapshot is not None
            else net_components[ref].get("props", {}).get("Manufacturer Part", "")
        )
        assert row.get("mpn") == expected_mpn, f"input MPN mismatch: {ref}"
        expected_pinmap = {
            pin: net_value
            for (component_ref, pin), net_value in pins.items()
            if component_ref == ref
        }
        assert normalized_pins(row.get("pins", {})) == expected_pinmap, f"input pin map mismatch: {ref}"
        mapped = layout_by_ref[ref]
        assert mapped.get("mpn") == expected_mpn, f"layout MPN mismatch: {ref}"
        assert normalized_pins(mapped.get("pins", {})) == expected_pinmap, f"layout pin map mismatch: {ref}"
        if snapshot is not None:
            assert bool(row.get("addIntoPcb")) == bool(snapshot.get("pcb")), f"input PCB flag mismatch: {ref}"
            assert mapped.get("primitiveId") == snapshot.get("id"), f"layout primitive ID mismatch: {ref}"
            assert bool(mapped.get("addIntoPcb")) == bool(snapshot.get("pcb")), f"layout PCB flag mismatch: {ref}"
            assert bool(mapped.get("addIntoBom")) == bool(snapshot.get("bom")), f"layout BOM flag mismatch: {ref}"

    fan_by_ref = component_refs(fan_rows, "fan source")
    for ref in ALL_FAN_REFS - REMOVED_REFS:
        assert ref in fan_by_ref, f"fan source missing retained fan ref: {ref}"
        expected_mpn = (
            snapshot_rows[ref].get("mpn", "") if snapshot_rows is not None
            else net_components[ref].get("props", {}).get("Manufacturer Part", "")
        )
        assert fan_by_ref[ref].get("mpn") == expected_mpn, f"fan source MPN mismatch: {ref}"
        expected_pinmap = {
            pin: net_value
            for (component_ref, pin), net_value in pins.items()
            if component_ref == ref
        }
        assert normalized_pins(fan_by_ref[ref].get("pins", {})) == expected_pinmap, f"fan source pin map mismatch: {ref}"


def update_json_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    indexed = component_refs(rows, label)
    assert set(indexed) >= ALL_FAN_REFS, f"{label} lacks one or more expected fan refs"
    for ref in ALL_FAN_REFS:
        assert indexed[ref].get("mpn") == EXPECTED_MPN[ref], f"unexpected {label} MPN for {ref}"
    assert normalized_pins(indexed["J_FAN"].get("pins", {})) == EXPECTED_OLD_FAN_PINS, (
        f"unexpected pre-publication J_FAN wiring in {label}"
    )
    output = [copy.deepcopy(row) for row in rows if row.get("ref") not in REMOVED_REFS]
    new_index = component_refs(output, label + " candidate")
    new_index["J_FAN"]["pins"]["1"] = "VOUT_5V"
    if label == "fan source":
        note = new_index["J_FAN"].get("note", "")
        assert note.count("pin 1=FAN_5V") == 1, "fan-source J_FAN note is not the expected old pin/net text"
        new_index["J_FAN"]["note"] = note.replace("pin 1=FAN_5V", "pin 1=VOUT_5V")
    return output


def update_layout_map(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    rows = result["components"]
    indexed = component_refs(rows, "layout map")
    assert set(indexed) >= ALL_FAN_REFS, "layout map lacks expected fan refs"
    for ref in ALL_FAN_REFS:
        assert indexed[ref].get("mpn") == EXPECTED_MPN[ref], f"unexpected layout map MPN for {ref}"
    assert normalized_pins(indexed["J_FAN"].get("pins", {})) == EXPECTED_OLD_FAN_PINS
    result["components"] = [row for row in rows if row.get("ref") not in REMOVED_REFS]
    result_by_ref = component_refs(result["components"], "layout map candidate")
    result_by_ref["J_FAN"]["pins"]["1"] = "VOUT_5V"
    policy = result.get("policy", [])
    old_policy = "保留VO617A-2/VO617A-4、SGM2593/SGM2593A载体差异；本映射不构成器件验收"
    assert old_policy in policy, "layout-map policy text changed; review before replacing stale SGM2593 scope"
    result["policy"] = [
        "保留VO617A-2/VO617A-4载体差异；采购MPN与CAD载体分开记录；本映射不构成器件验收"
        if line == old_policy else line
        for line in policy
    ]
    result.setdefault("inputs", {})["independentExpected"] = "当前施工输入.json"
    result["inputs"]["officialReopenedEnet"] = "Layout收尾/final_reopened.enet"
    return result


def update_bom(raw: bytes) -> tuple[bytes, int]:
    has_utf8_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    assert rows and "位号" in rows[0] and "MPN" in rows[0] and "备注" in rows[0]
    headers = list(rows[0])
    by_ref: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_ref.setdefault(row["位号"], []).append(row)
    for ref in ALL_FAN_REFS:
        matches = by_ref.get(ref, [])
        assert len(matches) == 1, f"BOM must have exactly one {ref} row"
        assert matches[0]["MPN"] == EXPECTED_MPN[ref], f"unexpected BOM MPN for {ref}"
    assert "Pin 1=FAN_5V" in by_ref["J_FAN"][0]["备注"], "BOM J_FAN note lacks expected old pin/net text"

    result: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        if row["位号"] in REMOVED_REFS:
            removed += 1
            continue
        candidate = copy.deepcopy(row)
        if candidate["位号"] == "J_FAN":
            note = candidate["备注"]
            assert note.count("Pin 1=FAN_5V") == 1
            candidate["备注"] = note.replace("Pin 1=FAN_5V", "Pin 1=VOUT_5V")
            assert "FAN_5V" not in candidate["备注"]
        result.append(candidate)
    assert removed == 4 and len(result) == len(rows) - 4, "BOM deletion count is not four"
    assert {row["位号"] for row in result if row["位号"] in ALL_FAN_REFS} == {"J_FAN"}

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="raise", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(result)
    prefix = "\ufeff" if has_utf8_bom else ""
    return (prefix + buffer.getvalue()).encode("utf-8"), len(rows)


def manifest_digest(manifest: Any, filename: str) -> str:
    assert isinstance(manifest, list), "official export SHA256 manifest must be a list"
    matches = [
        row for row in manifest
        if Path(str(row.get("path", row.get("name", "")))).name == filename
    ]
    assert len(matches) == 1, f"official export manifest must contain exactly one {filename} entry"
    digest = matches[0].get("sha256")
    assert isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest), (
        f"official export manifest has no valid SHA256 for {filename}"
    )
    return digest.lower()


def archive_targets() -> list[Path]:
    paths: set[Path] = {path for path in N.rglob("*") if path.is_file()}
    paths.update(
        {
            CURRENT_INPUT, FAN_SOURCE, POWER_SOURCE, LAYOUT_MAP, BOM,
            CURRENT_NET, CURRENT_PDF,
            INITIAL_HASHES, O / "baseline.json", O / "candidate_mutation.json",
            O / "fan_local_wires.json", O / "session.json",
            REOPENED_NET, REOPENED_SNAPSHOT, EXPORT_HASHES, CANDIDATE_EPRO2,
            CANDIDATE_P1,
            O / "原生发布校验.json", O / "发布归档记录.json",
        }
    )
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    required = [
        INITIAL_HASHES, REOPENED_NET, REOPENED_SNAPSHOT, EXPORT_HASHES,
        CANDIDATE_EPRO2, CANDIDATE_P1, CURRENT_INPUT, FAN_SOURCE, POWER_SOURCE,
        LAYOUT_MAP, BOM, CURRENT_NET, CURRENT_PDF, ARCHIVE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, "Layout release preflight files are missing:\n" + "\n".join(missing)
    assert zipfile.is_zipfile(ARCHIVE), f"project recovery archive is not a readable ZIP: {ARCHIVE}"

    initial, initial_raw = read_json(INITIAL_HASHES)
    assert isinstance(initial, dict) and initial, "initial native SHA256 manifest is empty or malformed"
    assert_native_baseline(initial)
    assert initial.get("BMS-9S4P.eprj2") and initial.get("BMS-9S4P.eprj3"), (
        "initial native manifest must include both eprj files"
    )

    old_net, old_net_raw = read_json(CURRENT_NET)
    new_net, new_net_raw = read_json(REOPENED_NET)
    snapshot, snapshot_raw = read_json(REOPENED_SNAPSHOT)
    export_hashes, export_hashes_raw = read_json(EXPORT_HASHES)
    assert sha(old_net_raw) == REVIEWED_ENET_HASHES["baseline"], (
        "formal ENET differs from the exact reviewed Layout baseline"
    )
    assert sha(new_net_raw) == REVIEWED_ENET_HASHES["candidate"], (
        "official reopened ENET differs from the exact reviewed candidate"
    )
    enet_deltas = assert_net_delta(old_net, new_net)
    pin_changes = enet_deltas["pinChanges"]
    channel_id_changes = enet_deltas["channelIdChanges"]
    snapshot_by_ref = assert_snapshot(snapshot, new_net)

    export_digest = manifest_digest(export_hashes, CANDIDATE_EPRO2.name)
    epro2_raw = CANDIDATE_EPRO2.read_bytes()
    p1_raw = CANDIDATE_P1.read_bytes()
    assert sha(epro2_raw) == export_digest, "candidate EPRO2 does not match official export SHA256"
    assert epro2_raw, "official candidate EPRO2 is empty"
    assert p1_raw, "official saved P1.esch2 is empty"
    assert sha(epro2_raw) != initial["BMS-9S4P.epro2"], (
        "official EPRO2 is byte-identical to the initial file; refusing to claim it was replaced"
    )
    assert sha(p1_raw) != initial["sch/Schematic1/P1.esch2"], (
        "WORK P1.esch2 is byte-identical to the initial page; refusing to claim it was refreshed"
    )

    input_rows, input_raw = read_json(CURRENT_INPUT)
    fan_rows, fan_raw = read_json(FAN_SOURCE)
    power_rows, power_raw = read_json(POWER_SOURCE)
    layout, layout_raw = read_json(LAYOUT_MAP)
    bom_raw = BOM.read_bytes()
    assert isinstance(input_rows, list) and isinstance(fan_rows, list) and isinstance(power_rows, list)
    power_refs = {row.get("ref") for row in power_rows}
    assert not (power_refs & ALL_FAN_REFS), (
        "four fan refs are absent from 功率元件连接.json; source ownership changed, review manually"
    )
    assert_source_alignment(input_rows, fan_rows, layout, old_net)

    input_updated = update_json_rows(input_rows, "current construction input")
    fan_updated = update_json_rows(fan_rows, "fan source")
    layout_updated = update_layout_map(layout)
    bom_updated, old_bom_rows = update_bom(bom_raw)

    # Validate all generated source views against the exact official ENET before archiving.
    assert_source_alignment(input_updated, fan_updated, layout_updated, new_net, snapshot_by_ref)
    bom_candidate_rows = list(csv.DictReader(io.StringIO(bom_updated.decode("utf-8-sig"), newline="")))
    assert len(bom_candidate_rows) == old_bom_rows - 4
    assert [row["MPN"] for row in bom_candidate_rows if row["位号"] == "J_FAN"] == [EXPECTED_MPN["J_FAN"]]
    assert [row["备注"] for row in bom_candidate_rows if row["位号"] == "J_FAN"][0].count("Pin 1=VOUT_5V") == 1

    # The official worktree export must leave formal eprj2/eprj3 untouched; they
    # remain protected by the complete initial native SHA256 baseline.
    candidate_digests = {
        CURRENT_INPUT: sha(input_raw), FAN_SOURCE: sha(fan_raw), POWER_SOURCE: sha(power_raw),
        LAYOUT_MAP: sha(layout_raw), BOM: sha(bom_raw), CURRENT_NET: sha(old_net_raw),
        INITIAL_HASHES: sha(initial_raw), REOPENED_NET: sha(new_net_raw),
        REOPENED_SNAPSHOT: sha(snapshot_raw), EXPORT_HASHES: sha(export_hashes_raw),
        CANDIDATE_EPRO2: sha(epro2_raw), CANDIDATE_P1: sha(p1_raw),
    }
    assert_native_baseline(initial)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_Layout精简发布前"
    records: list[dict[str, str]] = []
    files = archive_targets()
    assert all(path.is_relative_to(M) for path in files), "archive target escaped BMS module root"
    with zipfile.ZipFile(ARCHIVE, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing_names = set(archive.namelist())
        for path in files:
            entry = stamp + "/" + relative_to_module(path)
            assert entry not in existing_names, f"archive entry already exists: {entry}"
            raw = path.read_bytes()
            archive.writestr(entry, raw)
            records.append({"path": relative_to_module(path), "entry": entry, "sha256": sha(raw)})
    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        for row in records:
            archived = archive.read(row["entry"])
            source = M / Path(row["path"])
            assert sha(archived) == row["sha256"] == sha(source.read_bytes()), row["path"]

    # Recheck every dependency immediately before the first write.
    for path, expected in candidate_digests.items():
        assert sha(path.read_bytes()) == expected, f"release input changed after preflight: {path.name}"
    assert_native_baseline(initial)

    write_json_atomic(CURRENT_INPUT, input_updated)
    write_json_atomic(FAN_SOURCE, fan_updated)
    write_json_atomic(LAYOUT_MAP, layout_updated)
    write_bytes_atomic(BOM, bom_updated)
    write_bytes_atomic(CURRENT_NET, new_net_raw)
    write_bytes_atomic(N / "BMS-9S4P.epro2", epro2_raw)
    write_bytes_atomic(N / "sch" / "Schematic1" / "P1.esch2", p1_raw)

    published_hashes = native_file_hashes()
    changed_native = {"BMS-9S4P.epro2", "sch/Schematic1/P1.esch2"}
    assert set(published_hashes) == set(initial), "native file set changed during Layout release"
    for relative, expected in initial.items():
        if relative not in changed_native:
            assert published_hashes[relative] == expected, f"unexpected native change: {relative}"
    assert published_hashes["BMS-9S4P.epro2"] == sha(epro2_raw)
    assert published_hashes["sch/Schematic1/P1.esch2"] == sha(p1_raw)
    assert published_hashes["BMS-9S4P.eprj2"] == initial["BMS-9S4P.eprj2"]
    assert published_hashes["BMS-9S4P.eprj3"] == initial["BMS-9S4P.eprj3"]
    assert component_refs(read_json(CURRENT_INPUT)[0], "published construction input").keys() == snapshot_by_ref.keys()
    assert pinmap(read_json(CURRENT_NET)[0]) == pinmap(new_net)

    write_json_atomic(O / "原生发布校验.json", [
        {"file": relative, "sha256": digest}
        for relative, digest in sorted(published_hashes.items())
    ])
    write_json_atomic(
        O / "发布归档记录.json",
        {
            "archive": "归档/历史与调试.zip",
            "archiveReadbackSHA256Verified": True,
            "archiveEntries": records,
            "candidate": {
                "officialReopenedEnetSHA256": sha(new_net_raw),
                "savedReopenedSnapshotSHA256": sha(snapshot_raw),
                "officialEpro2SHA256": sha(epro2_raw),
                "officialP1SHA256": sha(p1_raw),
            },
            "enetMetadataDeltas": {
                "projectId": {"before": old_net.get("projectId"), "after": new_net.get("projectId")},
                "removedStaleNetRules": ["FAN_5V", "FAN_ILIM"],
                "regeneratedChannelIds": channel_id_changes,
            },
            "pinChanges": pin_changes,
            "removedRefs": sorted(REMOVED_REFS),
            "counts": {
                "before": {"cad": 217, "pcb": 216, "mpnTypes": 53, "pins": 723, "bomRows": old_bom_rows},
                "after": {**EXPECTED_COUNTS, "bomRows": old_bom_rows - 4},
                "drc": EXPECTED_DRC,
            },
            "nativeFilesUpdated": ["BMS-9S4P.epro2", "sch/Schematic1/P1.esch2"],
            "nativeFilesPreserved": ["BMS-9S4P.eprj2", "BMS-9S4P.eprj3"],
            "pdf": "当前PDF已进入发布前归档；本脚本不生成或替换PDF，待root另行生成并核对。",
            "scope": "删除风扇高边限流开关及其ILIM电阻、输入/输出大电容；J_FAN pin1由FAN_5V改接VOUT_5V。采购MPN与CAD载体身份字段保持分离。",
        },
    )
    print(json.dumps({
        "archiveReadback": "PASS",
        "removedRefs": sorted(REMOVED_REFS),
        "pinChanges": pin_changes,
        "regeneratedChannelIdCount": len(channel_id_changes),
        "counts": {**EXPECTED_COUNTS, "drcWarnings": 8, "bomRows": old_bom_rows - 4},
        "nativeUpdated": sorted(changed_native),
        "eprj2eprj3": "SHA256 unchanged",
        "pdf": "left for root to regenerate",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
