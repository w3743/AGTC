"""Publish current DRC standardization readback into Layout evidence.

This publisher consumes snapshots and strict semantic evidence already captured
by the parent workflow. It does not call EasyEDA, export DRC, or edit native CAD.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys


M = Path(__file__).resolve().parents[2]
E = M / "检查" / "DRC标准化"
L = M / "检查" / "Layout验证"
NATIVE = [
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
]
P1 = NATIVE[1]
EXPECTED_COMPONENTS = 208
EXPECTED_PINS = 700


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def snapshot_counts(snapshot, label):
    for key in ("rows", "ports", "wires", "texts", "drc"):
        if key not in snapshot:
            raise AssertionError(f"{label} missing {key}")
    rows = snapshot["rows"]
    refs = [row.get("ref") for row in rows]
    if any(ref is None for ref in refs) or len(refs) != len(set(refs)):
        raise AssertionError(f"{label} has missing or duplicate component refs")
    pins = sum(len(row.get("pins", [])) for row in rows)
    if len(rows) != EXPECTED_COMPONENTS or pins != EXPECTED_PINS:
        raise AssertionError(
            f"{label} expected {EXPECTED_COMPONENTS} components/"
            f"{EXPECTED_PINS} pins, got {len(rows)}/{pins}"
        )
    return {
        "components": len(rows),
        "pins": pins,
        "ports": len(snapshot["ports"]),
        "wires": len(snapshot["wires"]),
        "texts": len(snapshot["texts"]),
    }


def compare_saved_to_final(saved, final):
    saved_refs = [row["ref"] for row in saved["rows"]]
    final_refs = [row["ref"] for row in final["rows"]]
    saved_rows = sorted(saved["rows"], key=lambda row: str(row["ref"]))
    final_rows = sorted(final["rows"], key=lambda row: str(row["ref"]))
    row_differences = [
        saved_row["ref"]
        for saved_row, final_row in zip(saved_rows, final_rows)
        if saved_row != final_row
    ]
    if len(saved_rows) != len(final_rows):
        row_differences.extend(
            sorted(set(saved_refs) ^ set(final_refs), key=str)
        )
    rows_equal = len(saved_rows) == len(final_rows) and saved_rows == final_rows
    comparison = {
        "rows_sorted_by_ref_equal": rows_equal,
        "saved_row_count": len(saved_rows),
        "final_row_count": len(final_rows),
        "row_differences": sorted(set(row_differences), key=str),
        "ports_equal": saved["ports"] == final["ports"],
        "wires_equal": saved["wires"] == final["wires"],
        "texts_equal": saved["texts"] == final["texts"],
        "saved_counts": snapshot_counts(saved, "saved-snapshot"),
        "final_counts": snapshot_counts(final, "final-snapshot"),
    }
    if not (
        comparison["rows_sorted_by_ref_equal"]
        and comparison["ports_equal"]
        and comparison["wires_equal"]
        and comparison["texts_equal"]
    ):
        raise AssertionError(f"saved/final snapshot mismatch: {comparison}")
    return comparison


def drc_summary(snapshot):
    raw = snapshot["drc"]
    entries = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(item, dict) for item in entries):
        raise AssertionError(f"unrecognized DRC summary: {raw!r}")
    counts = {}
    for item in entries:
        kind = str(item.get("type", "")).lower()
        counts[kind] = counts.get(kind, 0) + int(item.get("count", 0))
    warning_groups = counts.get("warn", counts.get("warning", 0))
    errors = counts.get("error", counts.get("err", 0))
    fatals = counts.get("fatal", 0)
    if warning_groups != 1 or errors or fatals:
        raise AssertionError(
            f"expected one aggregate warning group and no errors/fatals; "
            f"got {counts}"
        )
    return {
        "raw": raw,
        "aggregate_warning_groups": warning_groups,
        "errors": errors,
        "fatal": fatals,
        "warning_details_reviewed": False,
        "warning_group_is_device_count": False,
        "pre_fix_exported_warning_groups_reviewed": 7,
        "remaining_details": "PENDING_USER_EXPORTED_DRC_LOG",
    }


def validate_electrical_equivalence(evidence):
    if (
        evidence.get("status") != "PASS"
        or evidence.get("components") != EXPECTED_COMPONENTS
        or evidence.get("pins") != EXPECTED_PINS
        or evidence.get("ports_wires_texts_unchanged") is not True
    ):
        raise AssertionError("electrical-equivalence evidence is not a PASS")
    allowed = set(evidence.get("allowed_changed_fields", []))
    if not allowed.issubset({"name", "other"}):
        raise AssertionError(f"unexpected changed fields allowed: {sorted(allowed)}")
    rows = evidence.get("rows", [])
    if len(rows) != EXPECTED_COMPONENTS:
        raise AssertionError(
            f"electrical-equivalence has {len(rows)} rows, expected "
            f"{EXPECTED_COMPONENTS}"
        )
    changed_outside_metadata = [
        {"ref": row.get("ref"), "fields": sorted(set(row.get("fields", [])) - allowed)}
        for row in rows
        if set(row.get("fields", [])) - allowed
    ]
    if changed_outside_metadata:
        raise AssertionError(
            f"electrical-equivalence has non-metadata changes: "
            f"{changed_outside_metadata}"
        )
    return {
        "status": evidence["status"],
        "components": evidence["components"],
        "pins": evidence["pins"],
        "allowed_changed_fields": sorted(allowed),
        "changed_component_rows": sum(bool(row.get("fields")) for row in rows),
        "ports_wires_texts_unchanged": True,
    }


def native_line_geometry(p1_path):
    latest = {}
    for raw in p1_path.read_text(encoding="utf-8").splitlines():
        header_text, payload_text = raw.split("||", 1)
        header = json.loads(header_text)
        payload_text = payload_text.rstrip("|")
        payload = json.loads(payload_text) if payload_text.strip() else None
        if header.get("type") in {"LINE", "TEXT", "COMPONENT"}:
            key = (header["type"], header["id"])
            if payload is None:
                latest.pop(key, None)
            else:
                latest[key] = payload
    non_orthogonal = [
        payload
        for (kind, _), payload in latest.items()
        if kind == "LINE"
        and payload["startX"] != payload["endX"]
        and payload["startY"] != payload["endY"]
    ]
    return latest, non_orthogonal


def publish_inputs():
    snapshot = read(E / "final-snapshot.json")
    snapshot_counts(snapshot, "final-snapshot")
    write(L / "current-snapshot.json", snapshot)
    whitelist = read(L / "ports-with-cleanup-whitelist.json")
    whitelist.update(
        {
            "page_id": "350a4904c4ec87b3",
            "wireCount": len(snapshot["wires"]),
            "wires": snapshot["wires"],
            "ports": snapshot["ports"],
        }
    )
    write(L / "ports-with-cleanup-whitelist.json", whitelist)
    print(
        json.dumps(
            {
                "status": "PASS_INPUTS",
                "components": len(snapshot["rows"]),
                "pins": sum(len(row["pins"]) for row in snapshot["rows"]),
            },
            ensure_ascii=False,
        )
    )


def publish():
    final_snapshot = read(E / "final-snapshot.json")
    current_snapshot = read(L / "current-snapshot.json")
    saved_snapshot = read(E / "saved-snapshot.json")
    semantic = read(E / "final-semantic.json")
    equivalence = read(E / "electrical-equivalence.json")

    final_counts = snapshot_counts(final_snapshot, "final-snapshot")
    if current_snapshot != final_snapshot:
        raise AssertionError(
            "Layout current-snapshot is not the same final-snapshot used "
            "for final reopen readback"
        )
    snapshot_comparison = compare_saved_to_final(saved_snapshot, final_snapshot)
    equivalence_summary = validate_electrical_equivalence(equivalence)
    drc = drc_summary(final_snapshot)

    if semantic.get("status") != "PASS" or any(
        semantic.get("defectCounts", {}).values()
    ):
        raise AssertionError("current strict semantic result is not a clean PASS")
    if any(
        any(items)
        for items in semantic.get("defects", {}).values()
        if isinstance(items, list)
    ):
        raise AssertionError("current strict semantic result contains defects")

    before = read(E / "after-close-hashes.json")
    missing_hashes = [path for path in NATIVE if path not in before]
    if missing_hashes:
        raise AssertionError(f"after-close-hashes missing {missing_hashes}")
    after = {path: sha(M / path) for path in NATIVE}
    source_file_comparison = {
        path: {
            "before_sha256": before[path],
            "after_sha256": after[path],
            "unchanged": before[path] == after[path],
        }
        for path in NATIVE
    }
    p1_stable = source_file_comparison[P1]["unchanged"]
    if not p1_stable:
        raise AssertionError("P1.esch2 changed after the close baseline")
    source_unchanged = all(
        source_file_comparison[path]["unchanged"] for path in NATIVE
    )

    # Bind this strict semantic result to the same P1 bytes read after reopen.
    semantic_source_hashes = semantic.get("sourceSHA256", {})
    semantic_p1_hashes = [
        value
        for path, value in semantic_source_hashes.items()
        if Path(path.replace("\\", "/")).name.lower() == "p1.esch2"
    ]
    if not semantic_p1_hashes or after[P1] not in semantic_p1_hashes:
        raise AssertionError(
            "final-semantic.json does not carry the current P1.esch2 SHA256"
        )

    reopen = {
        "checked_at": datetime.datetime.now().isoformat(),
        "method": (
            "Published from DRC标准化/final-snapshot.json captured by the "
            "parent reopen workflow; this publisher performs no EDA call or export"
        ),
        "project": str(M / NATIVE[0]),
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_file_comparison": source_file_comparison,
        "p1_stable": p1_stable,
        "source_unchanged": source_unchanged,
        "components": final_counts["components"],
        "pins": final_counts["pins"],
        "snapshot_comparison": snapshot_comparison,
        "snapshot_differences": [],
        "snapshot_match": True,
        "snapshot_pin_and_mpn_differences": 0,
        "drc": final_snapshot["drc"],
        "aggregate_warning_groups": drc["aggregate_warning_groups"],
        "pre_fix_exported_warning_groups_reviewed": 7,
        "warning_details_reviewed": False,
        "warning_group_is_device_count": False,
        "warning_details_status": "PENDING_USER_EXPORTED_DRC_LOG",
        "status": "PASS_READBACK",
    }
    write(L / "root-after-cleanup-reopen.json", reopen)

    # Retain the complete current strict semantic result and add the legacy
    # Layout compatibility fields expected by existing consumers.
    geometry = semantic
    geometry.update(
        {
            "electrical_geometry_status": geometry["status"],
            "strict_current_geometry": True,
        }
    )
    geometry["method"] = (
        "Compatibility adapter of verify_semantic_geometry.py; all semantic "
        "findings retained. " + geometry["method"]
    )
    geometry["counts"]["pins"] = geometry["counts"]["pinsChecked"]
    geometry["counts"]["wires"] = geometry["counts"]["activeNativeLines"]
    geometry["counts"]["labels"] = len(final_snapshot["ports"])
    geometry["counts"]["segments"] = len(final_snapshot["wires"])
    _, non_orthogonal = native_line_geometry(M / P1)
    geometry["nonOrthogonal"] = non_orthogonal
    if non_orthogonal:
        raise AssertionError(f"non-orthogonal native lines: {len(non_orthogonal)}")
    for key in (
        "missingPins",
        "foreignNetAtPins",
        "foreignNetAtWireEndpoints",
        "danglingLabels",
    ):
        geometry[key] = []

    cleanup = read(L / "cleanup-manifest.json")
    latest, _ = native_line_geometry(M / P1)
    active_text_ids = {
        item_id for kind, item_id in latest if kind == "TEXT"
    }
    active_removed_text_ids = [
        item["id"]
        for item in cleanup["removed"]
        if item["id"] in active_text_ids
    ]
    if active_removed_text_ids:
        raise AssertionError(
            f"removed cleanup text remains active: {active_removed_text_ids}"
        )
    geometry["cleanup_text_check"] = {
        "expected_removed": len(cleanup["removed"]),
        "active_removed_text_ids": [],
    }
    active_fan_refs = [
        row["ref"] for row in final_snapshot["rows"] if "FAN" in row["ref"]
    ]
    if active_fan_refs:
        raise AssertionError(f"fan refs remain in final snapshot: {active_fan_refs}")
    geometry["active_fan_refs_in_final_snapshot"] = active_fan_refs
    geometry["standardization_snapshot_comparison"] = snapshot_comparison
    geometry["standardization_electrical_equivalence"] = equivalence_summary
    geometry["standardization_readback"] = {
        "final_snapshot": "检查/DRC标准化/final-snapshot.json",
        "saved_snapshot": "检查/DRC标准化/saved-snapshot.json",
        "strict_semantic_result": "检查/DRC标准化/final-semantic.json",
        "readback_evidence": "检查/Layout验证/root-after-cleanup-reopen.json",
        "all_rows_sorted_by_ref_equal": True,
        "ports_wires_texts_equal": True,
    }
    for path in (
        L / "cleanup-manifest.json",
        M / "检查" / "当前施工输入.json",
        E / "final-snapshot.json",
        E / "saved-snapshot.json",
        E / "electrical-equivalence.json",
        E / "after-close-hashes.json",
        L / "current-snapshot.json",
    ):
        geometry.setdefault("sourceSHA256", {})[str(path)] = sha(path)
    write(L / "final-geometry.json", geometry)

    publication = read(L / "当前发布.json")
    publication["sources"] = {
        path: {"sha256": sha(M / path), "bytes": (M / path).stat().st_size}
        for path in NATIVE + ["原理图准备/BOM.csv"]
    }
    publication["counts"].update(
        cad_components=final_counts["components"],
        pins=final_counts["pins"],
        texts=final_counts["texts"],
        netports=final_counts["ports"],
    )
    runtime = publication.get("known_root_path_runtime", {})
    runtime.update(
        {
            "status": "PASS_READBACK",
            "observed_components": final_counts["components"],
            "observed_pins": final_counts["pins"],
            "fatal": drc["fatal"],
            "errors": drc["errors"],
            "warnings": drc["aggregate_warning_groups"],
            "source_sha_unchanged": source_unchanged,
            "source_unchanged": source_unchanged,
            "p1_stable": p1_stable,
            "source_file_comparison": source_file_comparison,
            "snapshot_pin_and_mpn_differences": 0,
            "warning_details_reviewed": False,
            "pre_fix_exported_warning_groups_reviewed": 7,
            "warning_group_is_device_count": False,
            "warning_details_status": "PENDING_USER_EXPORTED_DRC_LOG",
            "evidence": "root-after-cleanup-reopen.json",
            "note": (
                "The current API aggregate contains one warning group. Seven "
                "groups in the earlier exported DRC log were reviewed; details "
                "for the remaining post-standardization group await a fresh "
                "user export. A group count does not identify a device/ref. "
                "source_unchanged is true only when all three native files "
                "match the after-close hashes; P1 stability is reported separately."
            ),
        }
    )
    publication["official_independent_copy_reopen"]["current"] = False
    publication["official_independent_copy_reopen"]["note"] = (
        "Historical baseline; current evidence is "
        "root-after-cleanup-reopen.json"
    )
    publication["required_next"] = (
        "Obtain a fresh user-exported post-standardization DRC log and identify "
        "the rule and references for the remaining aggregate warning group. "
        "The current API reports one group only; do not infer that it equals "
        "one device. Retain model, physical, and mechanical qualification gates."
    )
    write(L / "当前发布.json", publication)

    manifest = read(L / "publication.json")
    manifest["files"] = {
        path: publication["sources"][path] for path in NATIVE
    }
    manifest["verifiedFrom"] = (
        "Current BMS native project after metadata standardization, with final "
        "reopen snapshot and current strict semantic evidence"
    )
    manifest["reopenEvidence"] = (
        "检查/Layout验证/root-after-cleanup-reopen.json"
    )
    manifest["geometryEvidence"] = "检查/Layout验证/final-geometry.json"
    write(L / "publication.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "warnings": drc["aggregate_warning_groups"],
                "components": final_counts["components"],
                "pins": final_counts["pins"],
                "p1_sha256": after[P1],
                "source_unchanged": source_unchanged,
            },
            ensure_ascii=False,
        )
    )


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"inputs", "publish"}:
        raise SystemExit("usage: publish_evidence.py inputs|publish")
    if sys.argv[1] == "inputs":
        publish_inputs()
    else:
        publish()


if __name__ == "__main__":
    main()
