"""Publish a locally redrawn BMS schematic after its close/reopen evidence is ready.

This publisher never calls EasyEDA and never edits native CAD or BOM sources.
It validates the captured snapshots first, then refreshes Layout evidence in one
pass. Run ``audit-drc`` to print a read-only JSON review of the supplied DRC log.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import math
import json
import re
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]
LAYOUT = MODULE / "检查" / "Layout验证"
DRC_FIX = MODULE / "检查" / "DRC修复"
READBACK_SCOPE = HERE / "readback-scope.json"
DRC_LOG = Path(r"C:\Users\wangj\Downloads\schDrcLog_2026-10-16.txt")
DRC_BACKUP_MANIFEST = DRC_FIX / "latest-drc-backup.json"
DRC_BASELINE_SNAPSHOT = DRC_FIX / "latest-drc-before.json"
DRC_BASELINE_SEMANTIC = DRC_FIX / "latest-drc-before-semantic.json"
DRC_CLEANUP_PLAN = DRC_FIX / "重复网标清理分析.json"

NATIVE = [
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
]
P1 = "sch/Schematic1/P1.esch2"
EXPECTED_COMPONENTS = 208
EXPECTED_PINS = 700
EXPECTED_PORTS = 285
EXPECTED_WIRES = 331
EXPECTED_TEXTS = 237
EXPECTED_BASELINE_WARNINGS = 12
EXPECTED_CURRENT_WARNINGS = 1
EXPECTED_REMOVED_NET_ATTRIBUTES = 29
EXPECTED_DUPLICATE_NET_WIRE_GROUPS = 11
RECORD_KEYS = {
    "rows": "id",
    "ports": "id",
    "wires": "primitiveId",
    "texts": "primitiveId",
}
PIN_COORDINATE_TOLERANCE = 1e-9
LOCKED_COMPONENT_FIELDS = (
    "mpn",
    "manufacturer",
    "supplier",
    "supplierId",
    "component",
    "symbol",
    "footprint",
    "bom",
    "pcb",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.write_bytes(json_payload(value))


def json_payload(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_readback_scope(before, current_hashes):
    """Verify an optional identical-copy readback against actual native bytes."""
    if not READBACK_SCOPE.is_file():
        return None
    scope = read_json(READBACK_SCOPE)
    if not isinstance(scope, dict):
        raise AssertionError("readback-scope.json must be an object")
    required = {
        "method": "OFFICIAL_IDENTICAL_COPY_READBACK",
        "root_path_readback": False,
        "root_path_status": "HOLD_OPEN_DOCUMENT_TIMEOUT",
        "source_hashes_same": True,
    }
    for key, expected in required.items():
        actual = scope.get(key)
        if isinstance(expected, bool):
            matches = type(actual) is bool and actual is expected
        else:
            matches = actual == expected
        if not matches:
            raise AssertionError(f"readback-scope.json {key} must be {expected!r}")

    project_value = scope.get("project")
    if not isinstance(project_value, str) or not project_value.strip():
        raise AssertionError("readback-scope.json project path is missing")
    project = Path(project_value).resolve(strict=True)
    formal_project = (MODULE / NATIVE[0]).resolve(strict=True)
    if project.name.casefold() != Path(NATIVE[0]).name.casefold() or project == formal_project:
        raise AssertionError("readback-scope.json must name a separate native project copy")
    copy_root = project.parent
    declared = scope.get("copy_source_sha256")
    if not isinstance(declared, dict) or set(declared) != set(NATIVE):
        raise AssertionError("readback-scope.json must list SHA256 for all three native files")

    actual_copy_hashes = {}
    for rel in NATIVE:
        copy_path = (copy_root / rel).resolve(strict=True)
        actual_copy_hashes[rel] = sha256(copy_path)
        expected = str(current_hashes.get(rel, "")).lower()
        prior = str(before.get(rel, "")).lower()
        declared_hash = str(declared.get(rel, "")).lower()
        if not expected or not prior or not declared_hash:
            raise AssertionError(f"native SHA256 missing for identical-copy check: {rel}")
        if len({actual_copy_hashes[rel], expected, prior, declared_hash}) != 1:
            raise AssertionError(
                f"isolated-copy/formal/before-reopen SHA256 mismatch for {rel}: "
                f"copy={actual_copy_hashes[rel]}, formal={expected}, before={prior}, "
                f"scope={declared_hash}"
            )
    return {
        "method": required["method"],
        "root_path_readback": False,
        "root_path_status": required["root_path_status"],
        "project": str(project),
        "source_hashes_same": True,
        "copy_source_sha256": actual_copy_hashes,
    }


def read_drc_log(path: Path = DRC_LOG):
    if not path.is_file():
        raise FileNotFoundError(f"DRC log not found: {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    warning_lines = [line for line in lines if "[警告]" in line]
    summary_lines = [line for line in lines if "完成设计规则检查" in line]
    if len(summary_lines) != 1:
        raise AssertionError(
            f"expected one DRC summary; got "
            f"{len(summary_lines)} summary lines"
        )
    summary = re.search(
        r"致命错误[：:]\s*(\d+)\s*[,，]\s*错误[：:]\s*(\d+)"
        r"\s*[,，]\s*警告[：:]\s*(\d+)\s*[,，]\s*信息[：:]\s*(\d+)",
        summary_lines[0],
    )
    if not summary:
        raise AssertionError("could not parse fatal/error/warning/info totals")
    fatal, errors, warnings, infos = map(int, summary.groups())
    if len(warning_lines) != warnings:
        raise AssertionError(
            f"warning-line count {len(warning_lines)} differs from DRC summary {warnings}"
        )

    parsed_warnings = []
    all_references = []
    for warning_line in warning_lines:
        body = warning_line.split("[警告]", 1)[1]
        body = body.split(":", 1)[-1].strip(" ：,:，。")
        if "建议使用器件标准化" in warning_line:
            refs_text = warning_line.split("建议使用器件标准化", 1)[1]
            pairs = re.findall(r"([A-Za-z0-9_]+)\((\$1I\d+)\)", refs_text)
            references = [
                {"ref": ref, "component_id": component_id}
                for ref, component_id in pairs
            ]
            all_references.extend(references)
            classification = "component_property_supplier_number_mismatch"
        elif "有多个网络名" in body:
            wire_match = re.search(r"导线\s+(\S+?)\s+有多个网络名", body)
            names_text = body.split("有多个网络名", 1)[1].strip(" ：,:，。")
            classification = "duplicate_wire_net_attributes"
            references = []
            if wire_match:
                references.append({"wire": wire_match.group(1)})
            names = [name.strip() for name in re.split(r"[、,，]", names_text) if name.strip()]
            parsed_warnings.append({
                "timestamp": warning_line.split("[警告]", 1)[0].strip(),
                "classification": classification,
                "description": body,
                "wire": wire_match.group(1) if wire_match else None,
                "network_names": names,
                "references": references,
            })
            continue
        else:
            classification = "REVIEWED_WARNING_DETAILS"
            references = []
        parsed_warnings.append({
            "timestamp": warning_line.split("[警告]", 1)[0].strip(),
            "classification": classification,
            "description": body,
            "references": references,
        })

    return {
        "status": "PASS_REVIEWED_BASELINE_DRC_LOG",
        "source": str(path),
        "timestamp": (
            warning_lines[0].split("[警告]", 1)[0].strip()
            if warning_lines else summary_lines[0].split("完成设计规则检查", 1)[0].strip()
        ),
        "fatal_errors": fatal,
        "errors": errors,
        "aggregate_warning_groups": warnings,
        "warning_line_count": len(warning_lines),
        "information_messages": infos,
        "warning_details_reviewed": True,
        "warning_cleared": False,
        "warning_group_is_device_count": False,
        "warnings": parsed_warnings,
        "warning": {
            "classification": "MULTIPLE_REVIEWED_BASELINE_WARNING_TYPES",
            "description": "All warning lines in the supplied baseline DRC log were captured for review.",
            "reference_count": len(all_references),
            "references": all_references,
            "interpretation": (
                "The log summary warning count is independent of device/reference counts."
            ),
            "resolution": "BASELINE_ONLY; CURRENT_REOPENED_WARNING_DETAILS_REQUIRE_A_NEW_DRC_LOG",
        },
    }


def snapshot_counts(snapshot, label):
    for key in ("rows", "ports", "wires", "texts", "drc"):
        if key not in snapshot:
            raise AssertionError(f"{label} missing {key}")
    refs = [row.get("ref") for row in snapshot["rows"]]
    if any(ref is None for ref in refs) or len(refs) != len(set(refs)):
        raise AssertionError(f"{label} has missing or duplicate component refs")
    pins = sum(len(row.get("pins", [])) for row in snapshot["rows"])
    if len(refs) != EXPECTED_COMPONENTS or pins != EXPECTED_PINS:
        raise AssertionError(
            f"{label} expected {EXPECTED_COMPONENTS} components/"
            f"{EXPECTED_PINS} pins, got {len(refs)}/{pins}"
        )
    return {
        "components": len(refs),
        "pins": pins,
        "ports": len(snapshot["ports"]),
        "wires": len(snapshot["wires"]),
        "texts": len(snapshot["texts"]),
    }


def canonical_records(snapshot, group):
    identity = RECORD_KEYS[group]
    records = snapshot[group]
    ids = [str(record.get(identity, "")) for record in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise AssertionError(f"{group} has missing or duplicate {identity} values")
    ordered = sorted(records, key=lambda record: str(record[identity]))
    # API readback may order a component's pins differently. Keep every pin
    # field while sorting pins by number so the comparison tests the records.
    if group == "rows":
        ordered = copy.deepcopy(ordered)
        for row in ordered:
            pins = row.get("pins", [])
            if not isinstance(pins, list):
                raise AssertionError(f"rows contains a malformed pins list: {row.get('ref')}")
            pin_ids = [str(pin.get("number", "")) for pin in pins]
            if any(not pin_id for pin_id in pin_ids) or len(pin_ids) != len(set(pin_ids)):
                raise AssertionError(f"rows contains missing or duplicate pin numbers: {row.get('ref')}")
            row["pins"] = sorted(pins, key=lambda pin: str(pin.get("number", "")))
    return ordered


def exact_json_equal(left, right):
    """Compare JSON values with type equality; pin coordinates are handled separately."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def rows_equal_with_pin_coordinate_tolerance(saved, reopened):
    left = copy.deepcopy(saved)
    right = copy.deepcopy(reopened)
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False, [], []
    left_pins = left.get("pins")
    right_pins = right.get("pins")
    if not isinstance(left_pins, list) or not isinstance(right_pins, list):
        return False, [], []
    if len(left_pins) != len(right_pins):
        return False, [], []

    tolerated = []
    coordinate_errors = []
    for left_pin, right_pin in zip(left_pins, right_pins):
        if not isinstance(left_pin, dict) or not isinstance(right_pin, dict):
            return False, tolerated, coordinate_errors + [{
                "reason": "pin record is not an object",
            }]
        pin_number = str(left_pin.get("number", ""))
        for coordinate in ("x", "y"):
            left_has = coordinate in left_pin
            right_has = coordinate in right_pin
            if left_has != right_has:
                coordinate_errors.append({
                    "pin": pin_number, "coordinate": coordinate,
                    "saved": left_pin.get(coordinate), "reopened": right_pin.get(coordinate),
                    "reason": "coordinate field presence differs",
                })
                continue
            if not left_has or exact_json_equal(left_pin[coordinate], right_pin[coordinate]):
                continue
            a, b = left_pin[coordinate], right_pin[coordinate]
            numeric = (
                isinstance(a, (int, float)) and not isinstance(a, bool)
                and isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(float(a)) and math.isfinite(float(b))
            )
            if not numeric:
                coordinate_errors.append({
                    "pin": pin_number, "coordinate": coordinate,
                    "saved": a, "reopened": b,
                    "reason": "different nonnumeric or nonfinite coordinate",
                })
                continue
            delta = abs(float(a) - float(b))
            if delta <= PIN_COORDINATE_TOLERANCE:
                if delta > 0:
                    tolerated.append({
                        "pin": pin_number,
                        "coordinate": coordinate,
                        "saved": a,
                        "reopened": b,
                        "absolute_difference": delta,
                    })
            else:
                coordinate_errors.append({
                    "pin": pin_number,
                    "coordinate": coordinate,
                    "saved": a,
                    "reopened": b,
                    "absolute_difference": delta,
                    "reason": "difference exceeds absolute tolerance",
                })

    # Ignore only x/y inside corresponding pins. Component row coordinates,
    # all pin identity/network/NC fields, and every other field stay exact.
    for row in (left, right):
        for pin in row["pins"]:
            pin.pop("x", None)
            pin.pop("y", None)
    return exact_json_equal(left, right) and not coordinate_errors, tolerated, coordinate_errors


def compare_saved_reopened(saved, reopened):
    report = {}
    for group in RECORD_KEYS:
        left = canonical_records(saved, group)
        right = canonical_records(reopened, group)
        key = RECORD_KEYS[group]
        left_by_id = {str(item[key]): item for item in left}
        right_by_id = {str(item[key]): item for item in right}
        all_ids = sorted(set(left_by_id) | set(right_by_id))
        differences = []
        tolerated_pin_coordinates = []
        pin_coordinate_errors = []
        for item_id in all_ids:
            left_item = left_by_id.get(item_id)
            right_item = right_by_id.get(item_id)
            if left_item is None or right_item is None:
                differences.append(item_id)
                continue
            if group == "rows":
                equal, tolerated, coordinate_errors = rows_equal_with_pin_coordinate_tolerance(
                    left_item, right_item
                )
                tolerated_pin_coordinates.extend(
                    {"ref": item_id, **entry} for entry in tolerated
                )
                pin_coordinate_errors.extend(
                    {"ref": item_id, **entry} for entry in coordinate_errors
                )
            else:
                equal = exact_json_equal(left_item, right_item)
            if not equal:
                differences.append(item_id)
        report[group] = {
            "identity_key": key,
            "saved_count": len(left),
            "reopened_count": len(right),
            "equal": not differences,
            "different_ids": differences,
        }
        if group == "rows":
            report[group].update({
                "comparison_mode": "ALL_FIELDS_EXACT_EXCEPT_PIN_XY_ABSOLUTE_TOLERANCE",
                "pin_coordinate_tolerance_absolute": PIN_COORDINATE_TOLERANCE,
                "maximum_pin_coordinate_difference": max(
                    (entry["absolute_difference"] for entry in tolerated_pin_coordinates + pin_coordinate_errors
                     if "absolute_difference" in entry),
                    default=0.0,
                ),
                "tolerated_pin_coordinate_differences": tolerated_pin_coordinates,
                "pin_coordinate_errors": pin_coordinate_errors,
            })
    if not all(item["equal"] for item in report.values()):
        raise AssertionError(f"saved/reopened snapshot mismatch: {report}")
    return report


def validate_component_locks(baseline, saved):
    baseline_rows = {row["ref"]: row for row in baseline["rows"]}
    saved_rows = {row["ref"]: row for row in saved["rows"]}
    if set(baseline_rows) != set(saved_rows):
        raise AssertionError("component refs changed from the drawing baseline")

    differences = []
    for ref in sorted(baseline_rows):
        before = baseline_rows[ref]
        after = saved_rows[ref]
        changed = [
            field
            for field in LOCKED_COMPONENT_FIELDS
            if before.get(field) != after.get(field)
        ]
        before_nc = {
            str(pin.get("number")): pin.get("nc")
            for pin in before.get("pins", [])
        }
        after_nc = {
            str(pin.get("number")): pin.get("nc")
            for pin in after.get("pins", [])
        }
        if before.get("id") != after.get("id"):
            changed.append("id")
        if before_nc != after_nc:
            changed.append("pin_nc")
        if changed:
            differences.append({"ref": ref, "fields": changed})
    if differences:
        raise AssertionError(
            "procurement identity, NC, BOM, or PCB fields changed from "
            f"baseline: {differences}"
        )
    return {
        "baseline_components": len(baseline_rows),
        "saved_components": len(saved_rows),
        "locked_fields": list(LOCKED_COMPONENT_FIELDS) + ["id", "pin_nc"],
        "differences": [],
        "status": "PASS",
    }


def validate_semantic(semantic, current_p1_hash):
    if semantic.get("status") != "PASS":
        raise AssertionError("final-semantic.json is not PASS")
    counts = semantic.get("counts", {})
    required_counts = {
        "snapshotComponents": EXPECTED_COMPONENTS,
        "manifestComponents": EXPECTED_COMPONENTS,
        "snapshotPins": EXPECTED_PINS,
        "manifestPins": EXPECTED_PINS,
        "pinsChecked": EXPECTED_PINS,
    }
    mismatches = {
        key: {"expected": expected, "actual": counts.get(key)}
        for key, expected in required_counts.items()
        if counts.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"semantic construction-input counts differ: {mismatches}")
    if any(value for value in semantic.get("defectCounts", {}).values()):
        raise AssertionError("final-semantic.json contains nonzero defectCounts")
    if any(
        items
        for items in semantic.get("defects", {}).values()
        if isinstance(items, list)
    ):
        raise AssertionError("final-semantic.json contains defect records")
    p1_hashes = {
        value
        for path, value in semantic.get("sourceSHA256", {}).items()
        if path.replace("\\", "/").rsplit("/", 1)[-1].lower() == "p1.esch2"
    }
    if current_p1_hash not in p1_hashes:
        raise AssertionError(
            "final-semantic.json is not bound to the current P1.esch2 bytes"
        )
    return {
        "status": "PASS",
        "checked_components": counts["manifestComponents"],
        "checked_pins": counts["pinsChecked"],
        "construction_input_net_check": "PASS_FROM_FINAL_SEMANTIC",
        "source_p1_sha256": current_p1_hash,
        "defectCounts": semantic.get("defectCounts", {}),
    }


def source_hash_matches_basename(source_map, basenames, expected_hash, label):
    if not isinstance(source_map, dict):
        raise AssertionError(f"{label} sourceSHA256 must be an object")
    allowed = {name.casefold() for name in basenames}
    matches = [
        str(value).lower()
        for path, value in source_map.items()
        if Path(str(path).replace("\\", "/")).name.casefold() in allowed
    ]
    if str(expected_hash).lower() not in matches:
        raise AssertionError(
            f"{label} is not SHA-bound to expected source {sorted(allowed)}: "
            f"expected {expected_hash}, found {matches}"
        )


def validate_semantic_pin_networks(baseline, current):
    def index(semantic, label):
        pins = semantic.get("pinResults")
        if not isinstance(pins, list) or len(pins) != EXPECTED_PINS:
            raise AssertionError(f"{label} must contain exactly {EXPECTED_PINS} pinResults")
        result = {}
        for item in pins:
            if not isinstance(item, dict):
                raise AssertionError(f"{label} contains a non-object pinResult")
            key = (str(item.get("ref", "")), str(item.get("pin", "")))
            if not all(key) or key in result:
                raise AssertionError(f"{label} has missing or duplicate pin identity {key}")
            result[key] = item
        if len(result) != EXPECTED_PINS:
            raise AssertionError(f"{label} does not uniquely identify all 700 pins")
        return result

    before = index(baseline, "latest-drc-before-semantic.json")
    after = index(current, "final-semantic.json")
    if set(before) != set(after):
        raise AssertionError("the baseline/current semantic pin identity set changed")
    differences = [
        {"ref": ref, "pin": pin}
        for ref, pin in sorted(before)
        if not exact_json_equal(before[(ref, pin)], after[(ref, pin)])
    ]
    if differences:
        raise AssertionError(
            "700-pin network results differ from the pre-repair baseline: "
            f"{differences[:20]}"
        )
    return {
        "status": "PASS",
        "checked_pins": len(after),
        "baseline_pin_identities": len(before),
        "current_pin_identities": len(after),
        "all_pin_network_results_unchanged": True,
        "compared_fields": "complete pinResults record by ref/pin identity",
        "differences": [],
    }


def fold_native_records(payload: bytes, label: str):
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError(f"{label} is not UTF-8: {exc}") from exc
    latest = {}
    raw_count = 0
    malformed = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        raw_count += 1
        try:
            header_text, payload_text = raw_line.split("||", 1)
            header = json.loads(header_text)
            payload_text = payload_text.rstrip("|")
            record = json.loads(payload_text) if payload_text.strip() else None
            kind = header.get("type")
            record_id = header.get("id")
            if kind == "DOCHEAD" and record_id is None:
                record_id = "__DOCHEAD__"
            if not isinstance(kind, str) or not kind or not isinstance(record_id, str) or not record_id:
                raise ValueError("record type/id missing")
        except (ValueError, json.JSONDecodeError, AttributeError) as exc:
            malformed.append({"line": line_number, "error": str(exc)})
            continue
        latest[(kind, record_id)] = {
            "header": header,
            "payload": record,
        }
    if malformed:
        raise AssertionError(f"{label} contains malformed native records: {malformed[:10]}")
    return latest, raw_count


def validate_cleanup_plan_sources(baseline_p1_hash):
    if not DRC_CLEANUP_PLAN.is_file():
        raise FileNotFoundError(f"NET attribute cleanup plan is missing: {DRC_CLEANUP_PLAN}")
    plan = read_json(DRC_CLEANUP_PLAN)
    if not isinstance(plan, dict) or plan.get("schemaVersion") != 1:
        raise AssertionError("duplicate-net cleanup plan has an unsupported schema")
    sources = plan.get("sources")
    if not isinstance(sources, dict):
        raise AssertionError("duplicate-net cleanup plan is missing its source hashes")
    expected_sources = {
        "nativeSchematic": P1,
        "snapshot": "检查/DRC修复/latest-drc-before.json",
        "strictSemanticBaseline": "检查/DRC修复/latest-drc-before-semantic.json",
    }
    for name, expected_path in expected_sources.items():
        source = sources.get(name)
        if not isinstance(source, dict):
            raise AssertionError(f"cleanup plan source entry is missing: {name}")
        actual_path = str(source.get("path", "")).replace("\\", "/")
        expected_hash = str(source.get("sha256", "")).lower()
        if actual_path != expected_path or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise AssertionError(f"cleanup plan source path/hash is invalid: {name}")
        if name == "nativeSchematic":
            # The plan is bound to the archived pre-repair P1 bytes. The
            # formal source is expected to differ after its allowlisted ATTR
            # removals, so compare this declaration with the extracted
            # baseline archive hash below rather than the current P1 file.
            if expected_hash != baseline_p1_hash:
                raise AssertionError(
                    "cleanup plan nativeSchematic hash differs from archived baseline P1"
                )
            continue
        source_file = MODULE / expected_path
        if not source_file.is_file() or sha256(source_file) != expected_hash:
            raise AssertionError(f"cleanup plan no longer matches its declared source: {name}")
    if str(sources["nativeSchematic"]["sha256"]).lower() != baseline_p1_hash:
        raise AssertionError("cleanup plan baseline P1 hash differs from archived baseline bytes")
    baseline_semantic = read_json(DRC_BASELINE_SEMANTIC)
    source_hash_matches_basename(
        baseline_semantic.get("sourceSHA256"),
        {"P1.esch2"},
        baseline_p1_hash,
        "latest-drc-before-semantic.json",
    )
    if plan.get("baseline", {}).get("strictStatus") != "PASS":
        raise AssertionError("cleanup plan baseline strict semantic result is not PASS")

    groups = plan.get("groups")
    if not isinstance(groups, list) or len(groups) != EXPECTED_DUPLICATE_NET_WIRE_GROUPS:
        raise AssertionError(
            f"cleanup plan must contain exactly {EXPECTED_DUPLICATE_NET_WIRE_GROUPS} wire groups"
        )
    expected_removed = set()
    expected_kept = set()
    plan_groups_by_wire = {}
    for group in groups:
        if not isinstance(group, dict):
            raise AssertionError("cleanup plan contains a malformed wire group")
        wire_id = group.get("nativeWireId")
        wire_ref = group.get("drcWireRef")
        net_name = group.get("net")
        attrs = group.get("attributes")
        if (not isinstance(wire_id, str) or not wire_id
                or not isinstance(wire_ref, str) or not wire_ref
                or not isinstance(net_name, str) or not net_name
                or not isinstance(attrs, list) or len(attrs) < 2
                or wire_id in plan_groups_by_wire):
            raise AssertionError("cleanup plan wire-group identity/attributes are invalid")
        keep_ids = []
        remove_ids = []
        group_attribute_ids = set()
        for attr in attrs:
            if not isinstance(attr, dict):
                raise AssertionError(f"cleanup plan has a malformed ATTR in wire {wire_id}")
            attr_id = attr.get("id")
            if (not isinstance(attr_id, str) or not attr_id or attr_id in group_attribute_ids
                    or attr.get("parentWireId") != wire_id
                    or attr.get("key") != "NET" or attr.get("value") != net_name):
                raise AssertionError(f"cleanup plan ATTR identity/contents are invalid: {attr_id!r}")
            group_attribute_ids.add(attr_id)
            if attr.get("action") == "keep":
                keep_ids.append(attr_id)
            elif attr.get("action") == "candidate-remove-attribute-record":
                remove_ids.append(attr_id)
            else:
                raise AssertionError(
                    f"cleanup plan ATTR action is unsupported: {attr.get('action')!r}"
                )
        if (len(keep_ids) != 1
                or keep_ids[0] != group.get("keepNetAttributeId")
                or set(remove_ids) != set(group.get("candidateRemoveAttributeIds", []))):
            raise AssertionError(f"cleanup plan keep/remove IDs are inconsistent for wire {wire_id}")
        expected_removed.update(remove_ids)
        expected_kept.update(keep_ids)
        plan_groups_by_wire[wire_id] = {
            "drc_wire_ref": wire_ref,
            "network_name": net_name,
            "kept_attribute_id": keep_ids[0],
            "removed_attribute_ids": sorted(remove_ids),
        }

    top_removed = plan.get("removeNetAttributes")
    top_kept = plan.get("keepNetAttributes")
    if (not isinstance(top_removed, list) or len(top_removed) != len(set(top_removed))
            or set(top_removed) != expected_removed
            or not isinstance(top_kept, list) or len(top_kept) != len(set(top_kept))
            or set(top_kept) != expected_kept):
        raise AssertionError("cleanup plan top-level keep/remove ID lists do not match its groups")
    if (len(expected_removed) != EXPECTED_REMOVED_NET_ATTRIBUTES
            or len(expected_kept) != EXPECTED_DUPLICATE_NET_WIRE_GROUPS):
        raise AssertionError("cleanup plan has an unexpected number of NET ATTR IDs")
    return plan, expected_removed, expected_kept, plan_groups_by_wire


def validate_baseline_drc_log(plan):
    warning_source = plan.get("sources", {}).get("warningLog", {})
    if not isinstance(warning_source, dict):
        raise AssertionError("cleanup plan does not identify the reviewed baseline DRC log")
    warning_path = Path(str(warning_source.get("path", "")))
    expected_hash = str(warning_source.get("sha256", "")).lower()
    if (not warning_path.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or sha256(warning_path) != expected_hash):
        raise AssertionError("reviewed baseline DRC log is missing or its SHA256 changed")
    review = read_drc_log(warning_path)
    if (review["fatal_errors"], review["errors"], review["aggregate_warning_groups"]
            ) != (0, 0, EXPECTED_BASELINE_WARNINGS):
        raise AssertionError(f"unexpected reviewed pre-repair DRC totals: {review}")
    warnings = review["warnings"]
    duplicate_warnings = [
        item for item in warnings if item["classification"] == "duplicate_wire_net_attributes"
    ]
    supplier_warnings = [
        item for item in warnings
        if item["classification"] == "component_property_supplier_number_mismatch"
    ]
    other_warnings = [
        item for item in warnings if item["classification"] not in {
            "duplicate_wire_net_attributes",
            "component_property_supplier_number_mismatch",
        }
    ]
    if (len(duplicate_warnings) != EXPECTED_DUPLICATE_NET_WIRE_GROUPS
            or len(supplier_warnings) != 1 or other_warnings):
        raise AssertionError(
            "baseline DRC log warning categories do not match the reviewed 11 duplicate-net "
            f"plus 1 supplier mismatch baseline: duplicate={len(duplicate_warnings)}, "
            f"supplier={len(supplier_warnings)}, other={len(other_warnings)}"
        )
    plan_groups = {
        group["drcWireRef"]: group for group in plan["groups"]
    }
    if len(plan_groups) != EXPECTED_DUPLICATE_NET_WIRE_GROUPS:
        raise AssertionError("cleanup plan has missing/duplicate DRC wire references")
    log_groups = {}
    for warning in duplicate_warnings:
        wire_ref = warning.get("wire")
        if not wire_ref or wire_ref in log_groups:
            raise AssertionError(f"baseline DRC duplicate-net warning has missing/duplicate wire {wire_ref!r}")
        log_groups[wire_ref] = warning
    if set(log_groups) != set(plan_groups):
        raise AssertionError("baseline DRC duplicate-wire references differ from the cleanup plan")
    for wire_ref, group in plan_groups.items():
        warning = log_groups[wire_ref]
        names = warning.get("network_names", [])
        if (len(names) != group.get("duplicateNETAttributeCount")
                or any(name != group.get("net") for name in names)):
            raise AssertionError(
                f"baseline DRC warning details differ from audited NET ATTR group {wire_ref}"
            )
    return {
        **review,
        "scope": "PRE_REPAIR_BASELINE_ONLY",
        "applies_to_current_reopened_snapshot": False,
        "reviewed_by_user": True,
        "baseline_log_sha256": expected_hash,
        "review_basis": (
            "User-reviewed pre-repair log; publisher rechecks its pinned SHA, complete warning "
            "line count, 11 duplicate-wire details, and the one supplier-mismatch warning."
        ),
        "current_reopened_warning_details_reviewed": False,
        "current_reopened_log_required": True,
    }


def validate_native_net_attribute_cleanup():
    for path in (
        DRC_BACKUP_MANIFEST, DRC_BASELINE_SNAPSHOT, DRC_BASELINE_SEMANTIC, DRC_CLEANUP_PLAN
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required DRC repair baseline is missing: {path}")
    backup = read_json(DRC_BACKUP_MANIFEST)
    if not isinstance(backup, dict) or backup.get("sha256_readback") is not True:
        raise AssertionError("latest-drc-backup.json must record sha256_readback=true")
    archive_rel = backup.get("archive")
    prefix = backup.get("prefix")
    files = backup.get("files")
    if not isinstance(archive_rel, str) or not isinstance(prefix, str) or not isinstance(files, dict):
        raise AssertionError("latest-drc-backup.json lacks archive/prefix/files")
    archive_relative = Path(archive_rel)
    if archive_relative.is_absolute() or ".." in archive_relative.parts:
        raise AssertionError("backup archive path must be module-relative and stay in the module")
    archive_path = (MODULE / archive_relative).resolve(strict=True)
    module_root = MODULE.resolve(strict=True)
    if not archive_path.is_relative_to(module_root):
        raise AssertionError("backup archive resolves outside the BMS module")
    prefix_path = Path(prefix.replace("\\", "/"))
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise AssertionError("backup archive prefix is unsafe")
    archived_p1_rel = P1
    archived_native_bytes = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        member_names = [item.filename for item in archive.infolist()]
        for rel in NATIVE:
            member = f"{prefix_path.as_posix()}/{rel}"
            if member_names.count(member) != 1:
                raise AssertionError(f"backup archive must contain exactly one {member}")
            archived_native_bytes[rel] = archive.read(member)
    archived_member = f"{prefix_path.as_posix()}/{archived_p1_rel}"
    baseline_p1_bytes = archived_native_bytes[archived_p1_rel]
    baseline_p1_hash = sha256_bytes(baseline_p1_bytes)
    expected_baseline_p1_hash = str(files.get(P1, "")).lower()
    if not expected_baseline_p1_hash or baseline_p1_hash != expected_baseline_p1_hash:
        raise AssertionError("archived baseline P1 SHA256 does not match latest-drc-backup.json")
    plan, plan_removed_ids, plan_kept_ids, plan_groups_by_wire = (
        validate_cleanup_plan_sources(baseline_p1_hash)
    )

    baseline_state, baseline_raw_count = fold_native_records(
        baseline_p1_bytes, "archived pre-repair P1.esch2"
    )
    current_p1_path = MODULE / P1
    current_p1_bytes = current_p1_path.read_bytes()
    current_p1_hash = sha256_bytes(current_p1_bytes)
    current_state, current_raw_count = fold_native_records(current_p1_bytes, "current P1.esch2")

    def active_net_attrs(state, label):
        result = {}
        for (kind, record_id), native_record in state.items():
            payload = native_record["payload"]
            if kind != "ATTR" or payload is None:
                continue
            if not isinstance(payload, dict):
                raise AssertionError(f"{label} has non-object ATTR payload {record_id}")
            if payload.get("key") == "NET":
                parent_id = payload.get("parentId")
                value = payload.get("value")
                if not isinstance(parent_id, str) or not parent_id or not isinstance(value, str):
                    raise AssertionError(f"{label} has malformed NET attribute {record_id}")
                result[record_id] = payload
        return result

    baseline_attrs = active_net_attrs(baseline_state, "baseline")
    current_attrs = active_net_attrs(current_state, "current")
    removed_ids = sorted(set(baseline_attrs) - set(current_attrs))
    added_ids = sorted(set(current_attrs) - set(baseline_attrs))
    changed_ids = sorted(
        record_id
        for record_id in set(baseline_attrs) & set(current_attrs)
        if not exact_json_equal(baseline_attrs[record_id], current_attrs[record_id])
    )
    if added_ids or changed_ids:
        raise AssertionError(
            f"NET attributes were added/rewritten instead of only removed: "
            f"added={added_ids}, changed={changed_ids}"
        )
    if len(removed_ids) != EXPECTED_REMOVED_NET_ATTRIBUTES:
        raise AssertionError(
            f"expected {EXPECTED_REMOVED_NET_ATTRIBUTES} removed NET attributes, got {len(removed_ids)}"
        )
    if set(removed_ids) != plan_removed_ids:
        raise AssertionError(
            "removed NET attribute IDs differ from the exact non-keep IDs in the audit plan: "
            f"expected={sorted(plan_removed_ids)}, actual={removed_ids}"
        )

    baseline_groups = {}
    current_groups = {}
    for record_id, payload in baseline_attrs.items():
        if payload["value"].strip():
            baseline_groups.setdefault(payload["parentId"], []).append((record_id, payload["value"]))
    for record_id, payload in current_attrs.items():
        if payload["value"].strip():
            current_groups.setdefault(payload["parentId"], []).append((record_id, payload["value"]))
    duplicate_groups = {
        parent_id: records
        for parent_id, records in baseline_groups.items()
        if len(records) > 1
    }
    if len(duplicate_groups) != EXPECTED_DUPLICATE_NET_WIRE_GROUPS:
        raise AssertionError(
            f"expected {EXPECTED_DUPLICATE_NET_WIRE_GROUPS} duplicate wire groups, "
            f"got {len(duplicate_groups)}"
        )
    redundant_total = sum(len(records) - 1 for records in duplicate_groups.values())
    if redundant_total != EXPECTED_REMOVED_NET_ATTRIBUTES:
        raise AssertionError(
            f"baseline duplicates imply {redundant_total} removals, not {EXPECTED_REMOVED_NET_ATTRIBUTES}"
        )

    baseline_wire_ids = {
        record_id for (kind, record_id), native_record in baseline_state.items()
        if kind == "WIRE" and native_record["payload"] is not None
    }
    current_wire_ids = {
        record_id for (kind, record_id), native_record in current_state.items()
        if kind == "WIRE" and native_record["payload"] is not None
    }
    group_proofs = []
    expected_removed_ids = set()
    for parent_id, before_records in sorted(duplicate_groups.items()):
        values = {value for _, value in before_records}
        if len(values) != 1:
            raise AssertionError(f"duplicate wire group {parent_id} carries different network names: {sorted(values)}")
        if parent_id not in baseline_wire_ids or parent_id not in current_wire_ids:
            raise AssertionError(f"NET attributes do not remain attached to a WIRE record: {parent_id}")
        planned_group = plan_groups_by_wire.get(parent_id)
        if planned_group is None or set(record_id for record_id, _ in before_records) != (
            set(planned_group["removed_attribute_ids"]) | {planned_group["kept_attribute_id"]}
        ):
            raise AssertionError(f"baseline ATTR IDs differ from the audited group for WIRE {parent_id}")
        after_records = current_groups.get(parent_id, [])
        if (len(after_records) != 1
                or after_records[0][1] not in values
                or after_records[0][0] != planned_group["kept_attribute_id"]):
            raise AssertionError(
                f"wire {parent_id} must retain exactly its audited keep ATTR ID; "
                f"found {after_records}"
            )
        group_removed = sorted(set(record_id for record_id, _ in before_records) -
                               set(record_id for record_id, _ in after_records))
        if len(group_removed) != len(before_records) - 1:
            raise AssertionError(f"wire {parent_id} removed the wrong number of NET attributes")
        expected_removed_ids.update(group_removed)
        group_proofs.append({
            "wire_id": parent_id,
            "network_name": next(iter(values)),
            "before_attribute_ids": sorted(record_id for record_id, _ in before_records),
            "retained_attribute_id": after_records[0][0],
            "removed_attribute_ids": group_removed,
        })
    if expected_removed_ids != set(removed_ids):
        raise AssertionError("removed NET attribute IDs do not exactly match the 11 duplicate wire groups")
    for parent_id, before_records in baseline_groups.items():
        if parent_id in duplicate_groups:
            continue
        if sorted(before_records) != sorted(current_groups.get(parent_id, [])):
            raise AssertionError(f"nonduplicate NET attribute group changed on wire {parent_id}")

    removed_keys = {("ATTR", record_id) for record_id in removed_ids}
    changed_native_keys = []
    all_keys = set(baseline_state) | set(current_state)
    for key in all_keys - removed_keys:
        if key not in baseline_state or key not in current_state:
            changed_native_keys.append(key)
        elif not exact_json_equal(baseline_state[key], current_state[key]):
            changed_native_keys.append(key)
    if changed_native_keys:
        raise AssertionError(
            "current P1 contains native record changes outside the 29 removed NET attributes: "
            f"{sorted(changed_native_keys)[:20]}"
        )
    for record_id in removed_ids:
        state_record = current_state.get(("ATTR", record_id), "<ABSENT>")
        state_value = (
            state_record["payload"]
            if isinstance(state_record, dict) and "payload" in state_record
            else state_record
        )
        if state_value not in (None, "<ABSENT>"):
            raise AssertionError(f"removed NET attribute {record_id} remains active in the native source")

    project_rel = NATIVE[0]
    archived_project_hash = str(files.get(project_rel, "")).lower()
    current_project_hash = sha256(MODULE / project_rel)
    if not archived_project_hash or archived_project_hash != current_project_hash:
        raise AssertionError(f"native project file changed: {project_rel}")

    config_rel = NATIVE[2]
    archived_config_bytes = archived_native_bytes[config_rel]
    archived_config_hash = str(files.get(config_rel, "")).lower()
    if not archived_config_hash or sha256_bytes(archived_config_bytes) != archived_config_hash:
        raise AssertionError(f"archived config bytes do not match backup manifest: {config_rel}")
    current_config_bytes = (MODULE / config_rel).read_bytes()
    current_config_hash = sha256_bytes(current_config_bytes)
    archived_config_state, archived_config_raw_count = fold_native_records(
        archived_config_bytes, "archived Schematic1.ecfg"
    )
    current_config_state, current_config_raw_count = fold_native_records(
        current_config_bytes, "current Schematic1.ecfg"
    )
    if (archived_config_raw_count != current_config_raw_count
            or set(archived_config_state) != set(current_config_state)):
        raise AssertionError("Schematic1.ecfg native record count/identity changed")
    config_header_differences = []
    config_payload_differences = []
    client_metadata_change = None
    client_key = ("DOCHEAD", "__DOCHEAD__")
    for key in sorted(archived_config_state):
        before_record = archived_config_state[key]
        after_record = current_config_state[key]
        if not exact_json_equal(before_record["header"], after_record["header"]):
            config_header_differences.append(key)
        before_payload = before_record["payload"]
        after_payload = after_record["payload"]
        if key != client_key:
            if not exact_json_equal(before_payload, after_payload):
                config_payload_differences.append(key)
            continue
        if (not isinstance(before_payload, dict) or not isinstance(after_payload, dict)
                or set(before_payload) != set(after_payload)):
            config_payload_differences.append(key)
            continue
        changed_fields = sorted(
            field for field in before_payload
            if not exact_json_equal(before_payload[field], after_payload[field])
        )
        if (changed_fields != ["client"]
                or not isinstance(before_payload.get("client"), str)
                or not before_payload["client"]
                or not isinstance(after_payload.get("client"), str)
                or not after_payload["client"]):
            config_payload_differences.append(key)
        else:
            client_metadata_change = {
                "record": "DOCHEAD",
                "changed_fields": changed_fields,
                "before_client": before_payload["client"],
                "after_client": after_payload["client"],
            }
    if config_header_differences or config_payload_differences or client_metadata_change is None:
        raise AssertionError(
            "Schematic1.ecfg changes exceed the exact DOCHEAD.client metadata allowance: "
            f"header={config_header_differences}, payload={config_payload_differences}"
        )

    return {
        "status": "PASS",
        "method": "folded-native-record-allowlist; exact 29 duplicate WIRE ATTR key=NET IDs may be absent/tombstoned; config permits only DOCHEAD.client",
        "backup_manifest": str(DRC_BACKUP_MANIFEST.relative_to(MODULE)).replace("\\", "/"),
        "backup_manifest_sha256": sha256(DRC_BACKUP_MANIFEST),
        "cleanup_plan": str(DRC_CLEANUP_PLAN.relative_to(MODULE)).replace("\\", "/"),
        "cleanup_plan_sha256": sha256(DRC_CLEANUP_PLAN),
        "cleanup_plan_status": plan.get("status"),
        "archive": str(archive_path.relative_to(MODULE)).replace("\\", "/"),
        "archive_sha256": sha256(archive_path),
        "archive_member": archived_member,
        "baseline_p1_sha256": baseline_p1_hash,
        "current_p1_sha256": current_p1_hash,
        "baseline_raw_record_count": baseline_raw_count,
        "current_raw_record_count": current_raw_count,
        "baseline_active_wire_count": len(baseline_wire_ids),
        "current_active_wire_count": len(current_wire_ids),
        "baseline_active_net_attribute_count": len(baseline_attrs),
        "current_active_net_attribute_count": len(current_attrs),
        "removed_attribute_count": len(removed_ids),
        "removed_attribute_ids": removed_ids,
        "retained_attribute_ids": sorted(plan_kept_ids),
        "added_attribute_ids": [],
        "changed_attribute_ids": [],
        "duplicate_wire_group_count": len(group_proofs),
        "duplicate_wire_groups": group_proofs,
        "other_native_record_changes": [],
        "project_sha256_unchanged": True,
        "config_sha256_unchanged": archived_config_hash == current_config_hash,
        "config_metadata_change": client_metadata_change,
        "project_and_config_sha256_unchanged": False,
    }


def validate_drc_repair_snapshots(baseline, saved, reopened):
    counts = {
        "baseline": snapshot_counts(baseline, "latest-drc-before"),
        "saved": snapshot_counts(saved, "saved-snapshot"),
        "reopened": snapshot_counts(reopened, "reopened-snapshot"),
    }
    expected_geometry = {
        "components": EXPECTED_COMPONENTS,
        "pins": EXPECTED_PINS,
        "ports": EXPECTED_PORTS,
        "wires": EXPECTED_WIRES,
        "texts": EXPECTED_TEXTS,
    }
    for label, actual in counts.items():
        if actual != expected_geometry:
            raise AssertionError(f"{label} snapshot counts differ from the DRC repair baseline: {actual}")
    baseline_drc = aggregate_warning_groups(baseline)
    saved_drc = aggregate_warning_groups(saved)
    reopened_drc = aggregate_warning_groups(reopened)
    if (baseline_drc["fatal"], baseline_drc["errors"], baseline_drc["warnings"]) != (0, 0, EXPECTED_BASELINE_WARNINGS):
        raise AssertionError(f"unexpected pre-repair DRC totals: {baseline_drc}")
    for label, drc in (("saved", saved_drc), ("reopened", reopened_drc)):
        if (drc["fatal"], drc["errors"], drc["warnings"]) != (0, 0, EXPECTED_CURRENT_WARNINGS):
            raise AssertionError(f"unexpected post-repair {label} DRC totals: {drc}")
    baseline_to_saved = compare_saved_reopened(baseline, saved)
    baseline_to_reopened = compare_saved_reopened(baseline, reopened)
    saved_to_reopened = compare_saved_reopened(saved, reopened)
    return {
        "status": "PASS",
        "counts": counts,
        "baseline_drc": baseline_drc,
        "saved_drc": saved_drc,
        "reopened_drc": reopened_drc,
        "baseline_to_saved": baseline_to_saved,
        "baseline_to_reopened": baseline_to_reopened,
        "saved_to_reopened": saved_to_reopened,
        "rows_ports_wires_texts_unchanged": True,
        "expected_post_repair_warnings": EXPECTED_CURRENT_WARNINGS,
        "warning_details_reviewed_for_current_snapshot": False,
    }


def normalize_drc_kind(value):
    """Normalize EasyEDA DRC labels such as ``fatalError`` and ``fatal error``."""
    compact = re.sub(r"[\s_-]+", "", str(value).strip().casefold())
    if compact in {"fatal", "fatals", "fatalerror", "fatalerrors"}:
        return "fatal"
    if compact in {"error", "errors", "err"}:
        return "error"
    if compact in {"warning", "warnings", "warn"}:
        return "warning"
    return None


def aggregate_warning_groups(snapshot):
    raw = snapshot["drc"]
    entries = raw if isinstance(raw, list) else [raw]
    totals = {"warning": 0, "error": 0, "fatal": 0}
    for item in entries:
        if not isinstance(item, dict):
            raise AssertionError(f"unrecognized DRC entry: {item!r}")
        label = str(item.get("type", ""))
        kind = normalize_drc_kind(label)
        count = item.get("count", 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise AssertionError(f"invalid DRC count for {label!r}: {count!r}")
        if kind is not None:
            totals[kind] += count
    return {
        "warnings": totals["warning"],
        "errors": totals["error"],
        "fatal": totals["fatal"],
        "raw": raw,
    }


def readback_drc_review(drc, *, allow_unreviewed_warnings):
    if drc["fatal"] or drc["errors"]:
        raise AssertionError(
            "drawing workstate requires zero fatal/errors in the reopened snapshot: "
            f"{drc}"
        )
    warning_count = drc["warnings"]
    if warning_count and not allow_unreviewed_warnings:
        raise AssertionError(
            f"current reopened snapshot has {warning_count} unreviewed DRC warnings; "
            "refusing publication by default. Use --allow-unreviewed-warnings only "
            "to publish a drawing workstate explicitly marked DRC HOLD."
        )
    if warning_count:
        status = "HOLD_UNREVIEWED_WARNINGS"
        drc_review_state = "HOLD"
        warning_details_reviewed = False
        resolution = "HOLD_UNTIL_CURRENT_DRC_LOG_IS_REVIEWED"
        description = (
            "The reopened snapshot supplies only aggregate warning counts. "
            "No current DRC log was reviewed, so warning rules and references are unknown."
        )
    else:
        status = "READBACK_ZERO_WARNINGS_NOT_FULL_LAYOUT_RELEASE"
        drc_review_state = "NO_WARNINGS_IN_READBACK"
        warning_details_reviewed = True
        resolution = "NO_WARNINGS_REPORTED_IN_REOPENED_SNAPSHOT"
        description = "The reopened snapshot reports no DRC warnings."
    return {
        "status": status,
        "drcReview": drc_review_state,
        "source": "reopened-snapshot.json",
        "scope": "CURRENT_REOPENED_SNAPSHOT_COUNTS_ONLY",
        "warning_count_basis": "warning entries/counts in reopened snapshot",
        "fatal_errors": drc["fatal"],
        "errors": drc["errors"],
        "aggregate_warning_groups": warning_count,
        "information_messages": None,
        "warning_details_reviewed": warning_details_reviewed,
        "warning_cleared": False,
        "warning_group_is_device_count": None,
        "warning": {
            "classification": "UNREVIEWED_CURRENT_WARNING_TOTAL" if warning_count else "NONE_REPORTED",
            "description": description,
            "reference_count": None,
            "references": [],
            "interpretation": (
                "The snapshot total is a warning count only; no rule or component references are inferred."
                if warning_count else "No warning references are present in the reopened snapshot."
            ),
            "resolution": resolution,
        },
    }


def prepare_publication(*, allow_unreviewed_warnings=False):
    saved = read_json(HERE / "saved-snapshot.json")
    reopened = read_json(HERE / "reopened-snapshot.json")
    baseline = read_json(DRC_BASELINE_SNAPSHOT)
    before = read_json(HERE / "before-reopen-hashes.json")
    semantic = read_json(HERE / "final-semantic.json")
    baseline_semantic = read_json(DRC_BASELINE_SEMANTIC)
    backup_manifest = read_json(DRC_BACKUP_MANIFEST)

    saved_counts = snapshot_counts(saved, "saved-snapshot")
    reopened_counts = snapshot_counts(reopened, "reopened-snapshot")
    baseline_counts = snapshot_counts(baseline, "latest-drc-before")
    baseline_snapshot_path = DRC_BASELINE_SNAPSHOT
    current_snapshot_path = LAYOUT / "current-snapshot.json"
    current_snapshot_sha = sha256(current_snapshot_path)
    baseline_snapshot_sha = sha256(baseline_snapshot_path)
    expected_published_snapshot_sha = sha256_bytes(json_payload(reopened))
    archived_current_snapshot_sha = str(
        backup_manifest.get("files", {}).get("检查/Layout验证/current-snapshot.json", "")
    ).lower()
    if current_snapshot_sha not in {archived_current_snapshot_sha, expected_published_snapshot_sha}:
        raise AssertionError(
            "Layout验证/current-snapshot.json changed after the DRC backup, and does not "
            "match an idempotently published reopened snapshot"
        )
    snapshot_comparison = compare_saved_reopened(saved, reopened)
    saved_snapshot_sha = sha256(HERE / "saved-snapshot.json")
    reopened_snapshot_sha = sha256(HERE / "reopened-snapshot.json")
    snapshot_files_byte_identical = saved_snapshot_sha == reopened_snapshot_sha
    component_locks = validate_component_locks(baseline, saved)
    cleanup_proof = validate_native_net_attribute_cleanup()
    cleanup_plan = read_json(DRC_CLEANUP_PLAN)
    if sha256(DRC_CLEANUP_PLAN) != cleanup_proof["cleanup_plan_sha256"]:
        raise AssertionError("NET attribute cleanup plan changed during publication preflight")
    repair_snapshot_proof = validate_drc_repair_snapshots(baseline, saved, reopened)
    pin_network_equivalence = validate_semantic_pin_networks(baseline_semantic, semantic)
    baseline_log_review = validate_baseline_drc_log(cleanup_plan)

    # NATIVE and bom_rel are module-relative paths. Keep them module-relative
    # in publication/source maps, while resolving the bytes from MODULE here.
    current_hashes = {path: sha256(MODULE / path) for path in NATIVE}
    if any(path not in before for path in NATIVE):
        raise AssertionError("before-reopen-hashes.json is missing native files")
    readback_scope = validate_readback_scope(before, current_hashes)
    source_comparison = {
        path: {
            "before_reopen_sha256": before[path],
            "after_reopen_sha256": current_hashes[path],
            **({"identical_copy_sha256": readback_scope["copy_source_sha256"][path]}
               if readback_scope else {}),
            "unchanged": before[path] == current_hashes[path],
        }
        for path in NATIVE
    }
    changed_sources = [
        path for path, record in source_comparison.items() if not record["unchanged"]
    ]
    if changed_sources:
        raise AssertionError(f"native sources changed after reopen: {changed_sources}")
    semantic_summary = validate_semantic(semantic, current_hashes[P1])

    readback_method = (
        readback_scope["method"] if readback_scope else "OFFICIAL_ROOT_PATH_READBACK"
    )
    readback_project = (
        readback_scope["project"] if readback_scope else str(MODULE / NATIVE[0])
    )
    root_path_readback = readback_scope is None
    root_path_status = (
        readback_scope["root_path_status"] if readback_scope else "PASS_ROOT_PATH_READBACK"
    )
    readback_status = (
        "PASS_IDENTICAL_COPY_READBACK" if readback_scope else "PASS_ROOT_PATH_READBACK"
    )

    drc = aggregate_warning_groups(reopened)
    drc_review = readback_drc_review(
        drc, allow_unreviewed_warnings=allow_unreviewed_warnings
    )
    drc_review["pre_repair_baseline_log_review"] = baseline_log_review
    drc_repair_validation = {
        "status": "PASS",
        "baseline_snapshot": str(DRC_BASELINE_SNAPSHOT.relative_to(MODULE)).replace("\\", "/"),
        "baseline_snapshot_sha256": baseline_snapshot_sha,
        "baseline_semantic": str(DRC_BASELINE_SEMANTIC.relative_to(MODULE)).replace("\\", "/"),
        "baseline_semantic_sha256": sha256(DRC_BASELINE_SEMANTIC),
        "current_semantic_sha256": sha256(HERE / "final-semantic.json"),
        "saved_snapshot_sha256": saved_snapshot_sha,
        "reopened_snapshot_sha256": reopened_snapshot_sha,
        "native_net_attribute_cleanup": cleanup_proof,
        "snapshot_record_comparison": repair_snapshot_proof,
        "pin_network_equivalence": pin_network_equivalence,
        "baseline_drc_log_review": {
            "scope": baseline_log_review["scope"],
            "status": baseline_log_review["status"],
            "sha256": baseline_log_review["baseline_log_sha256"],
            "reviewed_by_user": baseline_log_review["reviewed_by_user"],
            "fatal_errors": baseline_log_review["fatal_errors"],
            "errors": baseline_log_review["errors"],
            "warning_count": baseline_log_review["aggregate_warning_groups"],
            "warning_line_count": baseline_log_review["warning_line_count"],
            "applies_to_current_reopened_snapshot": False,
        },
        "current_reopened_drc": {
            "fatal_errors": drc_review["fatal_errors"],
            "errors": drc_review["errors"],
            "aggregate_warning_groups": drc_review["aggregate_warning_groups"],
            "warning_details_reviewed": False,
            "drcReview": "HOLD",
            "layout_release_status": "HOLD",
            "required_next": "Review a newly captured current DRC log before interpreting or closing the remaining warning.",
        },
        "procurement_nc_bom_pcb_identity": component_locks,
        "p1_only_allowlisted_net_attribute_ids_changed": True,
        "native_record_change_allowlist": {
            "status": "PASS",
            "p1_removed_net_attribute_count": cleanup_proof["removed_attribute_count"],
            "p1_removed_net_attribute_ids": cleanup_proof["removed_attribute_ids"],
            "project_sha256_unchanged": cleanup_proof["project_sha256_unchanged"],
            "config_metadata_change": cleanup_proof["config_metadata_change"],
        },
        "warning_cleared": False,
        "layout_release_status": "HOLD",
    }

    current_pub = read_json(LAYOUT / "当前发布.json")
    bom_rel = "原理图准备/BOM.csv"
    bom_file = MODULE / bom_rel
    recorded_bom = current_pub.get("sources", {}).get(bom_rel, {}).get("sha256")
    current_bom = sha256(bom_file)
    if not recorded_bom or recorded_bom != current_bom:
        raise AssertionError(
            "BOM.csv differs from the current publication baseline; refusing "
            "to publish a drawing update over it"
        )

    baseline_hash = sha256(baseline_snapshot_path)
    current_snapshot = copy.deepcopy(reopened)
    whitelist_path = LAYOUT / "ports-with-cleanup-whitelist.json"
    whitelist = read_json(whitelist_path)
    whitelist.update(
        {
            "wireCount": len(reopened["wires"]),
            "wires": reopened["wires"],
            "ports": reopened["ports"],
        }
    )

    final_geometry = copy.deepcopy(semantic)
    final_geometry.update(
        {
            "electrical_geometry_status": semantic["status"],
            "strict_current_geometry": True,
            "strict_semantic_geometry": True,
            "method": (
                "Official local drawing publication adapter of "
                "verify_semantic_geometry.py; all strict semantic findings retained. "
                + str(semantic.get("method", ""))
            ),
            "drawing_reopen_comparison": snapshot_comparison,
            "drc_repair_validation": drc_repair_validation,
            "native_net_attribute_cleanup": cleanup_proof,
            "pin_network_equivalence": pin_network_equivalence,
            "baseline_component_locks": component_locks,
            "construction_input_net_validation": semantic_summary,
            "drc_log_review": "检查/Layout验证/drc-log-review.json",
            "drcReview": drc_review["drcReview"],
            "layout_release_status": "HOLD",
            "warning_details_reviewed": drc_review["warning_details_reviewed"],
            "drawing_summary": {
                "status": (
                    "PASS_ELECTRICAL_IDENTICAL_COPY_READBACK_ROOT_HOLD"
                    if readback_scope else "PASS_ELECTRICAL_READBACK"
                ),
                "drcReview": drc_review["drcReview"],
                "layout_release_status": "HOLD",
                "warning_details_reviewed": drc_review["warning_details_reviewed"],
                "user_requirement": "U18",
                "scope": "VC/SRP局部连线重画",
                "readback_statement": (
                    "同字节副本重开PASS；根路径图页打开超时待用户核对"
                    if readback_scope else "正式根路径官方重开PASS"
                ),
                "readback_scope": readback_scope,
                "user_intent": (
                    "按功能及功率/信号流组织原理图；局部电路优先使用真实导线，"
                    "避免只排列器件并依赖网名拼接。"
                ),
                "verification": {
                    "source": (
                        "Parent-captured official identical-copy save/reopen snapshots plus strict semantic verifier"
                        if readback_scope else
                        "Parent-captured official root-path save/reopen snapshots plus strict semantic verifier"
                    ),
                    "saved_reopened_match": True,
                    "saved_reopened_match_basis": (
                        "Rows/ports/wires/texts compare by ID; all fields exact except pin x/y absolute tolerance 1e-9."
                        " Snapshot file byte identity is reported separately using SHA256."
                    ),
                    "snapshot_files_byte_identical": snapshot_files_byte_identical,
                    "saved_snapshot_sha256": saved_snapshot_sha,
                    "reopened_snapshot_sha256": reopened_snapshot_sha,
                    "record_comparison": snapshot_comparison,
                    "strict_semantic_status": semantic_summary["status"],
                    "components_checked": semantic_summary["checked_components"],
                    "pins_checked": semantic_summary["checked_pins"],
                    "procurement_nc_bom_pcb_locks": component_locks,
                    "drc_repair_validation": drc_repair_validation,
                    "pin_network_equivalence": pin_network_equivalence,
                    "readback_method": readback_method,
                    "readback_project": readback_project,
                    "root_path_readback": root_path_readback,
                    "root_path_status": root_path_status,
                    "source_hashes_same": readback_scope["source_hashes_same"] if readback_scope else True,
                },
                "drc_warning_cleared": False,
            },
        }
    )
    source_map = final_geometry.get("sourceSHA256", {})
    if not isinstance(source_map, dict):
        raise AssertionError("final-semantic.json sourceSHA256 must be an object")
    published_source_targets = {
        "sch/Schematic1/P1.esch2": sha256(MODULE / P1),
        "检查/Layout验证/current-snapshot.json": sha256_bytes(json_payload(current_snapshot)),
        "检查/Layout验证/ports-with-cleanup-whitelist.json": sha256_bytes(json_payload(whitelist)),
        "检查/Layout验证/cleanup-manifest.json": sha256(LAYOUT / "cleanup-manifest.json"),
        "检查/当前施工输入.json": sha256(MODULE / "检查" / "当前施工输入.json"),
    }
    target_basenames = {Path(path).name.casefold() for path in published_source_targets}
    final_geometry["sourceSHA256"] = {
        key: value
        for key, value in source_map.items()
        if Path(str(key).replace("\\", "/")).name.casefold() not in target_basenames
    }
    final_geometry["sourceSHA256"].update(published_source_targets)
    final_geometry.setdefault("counts", {}).update(
        {
            "components": reopened_counts["components"],
            "pins": reopened_counts["pins"],
            "ports": reopened_counts["ports"],
            "labels": reopened_counts["ports"],
            "wires": reopened_counts["wires"],
            "segments": reopened_counts["wires"],
            "texts": reopened_counts["texts"],
        }
    )
    for key in ("missingPins", "foreignNetAtPins", "foreignNetAtWireEndpoints", "danglingLabels"):
        final_geometry.setdefault(key, [])

    reopen_evidence = {
        "checked_at": dt.datetime.now().astimezone().isoformat(),
        "method": readback_method,
        "readback_status": readback_status,
        "project": readback_project,
        "root_path_readback": root_path_readback,
        "root_path_status": root_path_status,
        "source_hashes_same": readback_scope["source_hashes_same"] if readback_scope else True,
        "copy_source_sha256": (
            readback_scope["copy_source_sha256"] if readback_scope else None
        ),
        "readback_scope": readback_scope,
        "publisher_note": (
            "Snapshots were captured by the parent official identical-copy workflow; "
            "this publisher performs no EDA call or export. The formal root project "
            "page did not open and remains HOLD for user verification."
            if readback_scope else
            "Snapshots were captured by the parent official root-path workflow; "
            "this publisher performs no EDA call or export."
        ),
        "source_sha256_before": before,
        "source_sha256_after": current_hashes,
        "source_file_comparison": source_comparison,
        "source_unchanged": True,
        "components": reopened_counts["components"],
        "pins": reopened_counts["pins"],
        "snapshot_comparison": snapshot_comparison,
        "snapshot_match": True,
        "snapshot_match_basis": (
            "Records match by ID; all fields exact except pin x/y absolute tolerance 1e-9."
        ),
        "snapshot_files_byte_identical": snapshot_files_byte_identical,
        "saved_snapshot_sha256": saved_snapshot_sha,
        "reopened_snapshot_sha256": reopened_snapshot_sha,
        "maximum_pin_coordinate_difference": snapshot_comparison["rows"]["maximum_pin_coordinate_difference"],
        "pin_coordinate_tolerance_absolute": PIN_COORDINATE_TOLERANCE,
        "baseline_snapshot_sha256": baseline_hash,
        "baseline_counts": baseline_counts,
        "baseline_component_locks": component_locks,
        "drc_repair_validation": drc_repair_validation,
        "native_net_attribute_cleanup": cleanup_proof,
        "pin_network_equivalence": pin_network_equivalence,
        "construction_input_net_validation": semantic_summary,
        "drc": reopened["drc"],
        "drc_log_review": drc_review,
        "drcReview": drc_review["drcReview"],
        "layout_release_status": "HOLD",
        "aggregate_warning_groups": drc_review["aggregate_warning_groups"],
        "warning_details_reviewed": drc_review["warning_details_reviewed"],
        "warning_reference_count": drc_review["warning"]["reference_count"],
        "warning_cleared": False,
        "status": (
            (
                "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD"
                if readback_scope else "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING"
            )
            if drc_review["aggregate_warning_groups"]
            else (
                "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD"
                if readback_scope else "PASS_READBACK"
            )
        ),
    }

    current_pub["updated"] = dt.date.today().isoformat()
    current_pub["sources"] = {
        path: {
            "sha256": sha256(MODULE / path),
            "bytes": (MODULE / path).stat().st_size,
        }
        for path in NATIVE + [bom_rel]
    }
    current_pub.setdefault("counts", {}).update(
        {
            "cad_components": reopened_counts["components"],
            "pins": reopened_counts["pins"],
            "texts": reopened_counts["texts"],
            "netports": reopened_counts["ports"],
        }
    )
    current_pub["known_root_path_runtime"] = {
        "status": root_path_status,
        "readback_status": readback_status,
        "readback_method": readback_method,
        "readback_project": readback_project,
        "root_path_readback": root_path_readback,
        "root_path_status": root_path_status,
        "source_hashes_same": readback_scope["source_hashes_same"] if readback_scope else True,
        "observed_components": reopened_counts["components"],
        "observed_pins": reopened_counts["pins"],
        "fatal": drc_review["fatal_errors"],
        "errors": drc_review["errors"],
        "warnings": drc_review["aggregate_warning_groups"],
        "drcReview": drc_review["drcReview"],
        "layout_release_status": "HOLD",
        "warning_details_reviewed": drc_review["warning_details_reviewed"],
        "warning_group_is_device_count": drc_review["warning_group_is_device_count"],
        "warning_reference_count": drc_review["warning"]["reference_count"],
        "warning_references": [item["ref"] for item in drc_review["warning"]["references"]],
        "warning_cleared": False,
        "source_unchanged": True,
        "snapshot_match": True,
        "snapshot_match_basis": (
            "Records match by ID; all fields exact except pin x/y absolute tolerance 1e-9."
        ),
        "snapshot_files_byte_identical": snapshot_files_byte_identical,
        "maximum_pin_coordinate_difference": snapshot_comparison["rows"]["maximum_pin_coordinate_difference"],
        "pin_coordinate_tolerance_absolute": PIN_COORDINATE_TOLERANCE,
        "evidence": "root-after-cleanup-reopen.json",
        "drc_log_review": "drc-log-review.json",
    }
    if drc_review["aggregate_warning_groups"]:
        current_pub["required_next"] = (
            (
                "Same-byte copy readback passed; the formal root schematic page open timed out and awaits user verification. "
                if readback_scope else ""
            )
            + f"The current reopened snapshot reports {drc_review['aggregate_warning_groups']} DRC warning entries, "
            "but no current DRC log has been reviewed. Keep drcReview=HOLD and layout_release_status=HOLD; "
            "do not infer warning rules/references or claim DRC closure. Obtain and review the current DRC log. "
            "Layout routing, fabrication, and physical qualification remain gated."
        )
    else:
        current_pub["required_next"] = (
            (
                "Same-byte copy readback passed; the formal root schematic page open timed out and awaits user verification. "
                if readback_scope else ""
            )
            + "The reopened snapshot reports zero DRC warnings. This drawing readback does not release Layout; "
            "layout_release_status remains HOLD pending the project's separate Layout, fabrication, model, "
            "and physical qualification gates."
        )
    current_pub["drcReview"] = drc_review["drcReview"]
    current_pub["drc_repair_validation"] = drc_repair_validation
    current_pub["drc_reviewed_baseline_log"] = {
        "scope": baseline_log_review["scope"],
        "sha256": baseline_log_review["baseline_log_sha256"],
        "reviewed_by_user": True,
        "applies_to_current_reopened_snapshot": False,
    }
    current_pub["layout_release_status"] = "HOLD"
    current_pub["drc_log_review"] = "检查/Layout验证/drc-log-review.json"

    publication_path = LAYOUT / "publication.json"
    publication = read_json(publication_path)
    publication["files"] = {
        path: current_pub["sources"][path] for path in NATIVE
    }
    publication["verifiedFrom"] = (
        (
            "Official identical-copy readback of the current BMS native bytes; formal root schematic page open timed out, "
            if readback_scope else "Current BMS native project after official root-path save/reopen readback; "
        )
        + "component procurement, NC, BOM, and PCB flags match drawing baseline"
    )
    publication["reopenEvidence"] = "检查/Layout验证/root-after-cleanup-reopen.json"
    publication["geometryEvidence"] = "检查/Layout验证/final-geometry.json"
    publication["drcLogReview"] = "检查/Layout验证/drc-log-review.json"
    publication["drcReview"] = drc_review["drcReview"]
    publication["drcRepairValidation"] = "检查/Layout验证/final-geometry.json:drc_repair_validation"
    publication["layout_release_status"] = "HOLD"

    return {
        LAYOUT / "current-snapshot.json": current_snapshot,
        whitelist_path: whitelist,
        LAYOUT / "root-after-cleanup-reopen.json": reopen_evidence,
        LAYOUT / "final-geometry.json": final_geometry,
        LAYOUT / "drc-log-review.json": drc_review,
        LAYOUT / "publication.json": publication,
        LAYOUT / "当前发布.json": current_pub,
    }, {
        "status": (
            (
                "DRAWING_IDENTICAL_COPY_READBACK_WITH_ROOT_HOLD_AND_DRC_HOLD"
                if drc_review["drcReview"] == "HOLD"
                else "DRAWING_IDENTICAL_COPY_READBACK_WITH_ROOT_HOLD"
            )
            if readback_scope else (
                "DRAWING_PUBLICATION_WITH_DRC_HOLD"
                if drc_review["drcReview"] == "HOLD"
                else "DRAWING_READBACK_ONLY_LAYOUT_HOLD"
            )
        ),
        "components": reopened_counts["components"],
        "pins": reopened_counts["pins"],
        "ports": reopened_counts["ports"],
        "wires": reopened_counts["wires"],
        "texts": reopened_counts["texts"],
        "snapshot_match": True,
        "readback_method": readback_method,
        "readback_status": readback_status,
        "readback_project": readback_project,
        "root_path_readback": root_path_readback,
        "root_path_status": root_path_status,
        "source_hashes_same": readback_scope["source_hashes_same"] if readback_scope else True,
        "native_sources_unchanged": True,
        "drc_warning_groups": drc_review["aggregate_warning_groups"],
        "drc_warning_references": drc_review["warning"]["reference_count"],
        "drcReview": drc_review["drcReview"],
        "layout_release_status": "HOLD",
        "warning_details_reviewed": drc_review["warning_details_reviewed"],
        "warning_cleared": False,
        "baseline_snapshot_sha256": baseline_hash,
        "baseline_warning_log_sha256": baseline_log_review["baseline_log_sha256"],
        "drc_repair_validation": drc_repair_validation,
        "native_net_attribute_cleanup": cleanup_proof,
        "pin_network_equivalence": pin_network_equivalence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit-drc", "publish"),
        help="audit-drc rechecks the pinned pre-repair baseline log; publish refreshes drawing evidence",
    )
    parser.add_argument(
        "--allow-unreviewed-warnings",
        action="store_true",
        help=(
            "allow publishing a drawing workstate with current reopened warning counts; "
            "the output remains drcReview=HOLD and layout_release_status=HOLD"
        ),
    )
    args = parser.parse_args()

    if args.command == "audit-drc":
        baseline_review = validate_baseline_drc_log(read_json(DRC_CLEANUP_PLAN))
        print(json.dumps(baseline_review, ensure_ascii=False, indent=2))
        return

    outputs, result = prepare_publication(
        allow_unreviewed_warnings=args.allow_unreviewed_warnings
    )
    for path, value in outputs.items():
        write_json(path, value)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
