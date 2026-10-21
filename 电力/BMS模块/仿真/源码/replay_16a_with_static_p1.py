#!/usr/bin/env python3
"""Replay the 16 A reference model against the saved P1 without exporting ENET.

Run from the project root with:
    py -3 "BMS模块/仿真/源码/replay_16a_with_static_p1.py"

This adapter verifies the saved P1 RSHUNT pin-to-net joins from strict semantic
geometry evidence and the current BOM, then calls the unchanged calculation
functions in verify_charge_limit_16a.py.
It is a schematic/model evidence check, not electrical-function acceptance.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENTRY = Path(__file__).resolve()
MODEL_PATH = ROOT / "BMS模块" / "仿真" / "源码" / "verify_charge_limit_16a.py"
INPUT_PATH = ROOT / "BMS模块" / "仿真" / "输入" / "16A净充电上限验证_input.json"
P1_PATH = ROOT / "BMS模块" / "sch" / "Schematic1" / "P1.esch2"
BOM_PATH = ROOT / "BMS模块" / "原理图准备" / "BOM.csv"
REVIEW_PATH = ROOT / "BMS模块" / "检查" / "行进充电与回收支持审查.json"
LAYOUT_DIR = ROOT / "BMS模块" / "检查" / "Layout验证"
SNAPSHOT_PATH = LAYOUT_DIR / "current-snapshot.json"
PORTS_PATH = LAYOUT_DIR / "ports-with-cleanup-whitelist.json"
GEOMETRY_PATH = LAYOUT_DIR / "final-geometry.json"
REOPEN_PATH = LAYOUT_DIR / "root-after-cleanup-reopen.json"
READBACK_SCOPE_PATH = ROOT / "BMS模块" / "检查" / "局部连线" / "readback-scope.json"
MANIFEST_PATH = ROOT / "BMS模块" / "检查" / "当前施工输入.json"
NATIVE_PATHS = (
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize_drc_kind(value) -> str | None:
    compact = "".join(str(value).strip().casefold().split()).replace("_", "").replace("-", "")
    if compact in {"fatal", "fatals", "fatalerror", "fatalerrors"}:
        return "fatal"
    if compact in {"error", "errors", "err"}:
        return "error"
    if compact in {"warn", "warning", "warnings"}:
        return "warning"
    return None


def validate_identical_copy_readback(reopen: dict, scope: dict, p1_hash: str) -> dict:
    require(scope.get("method") == "OFFICIAL_IDENTICAL_COPY_READBACK"
            and scope.get("root_path_readback") is False
            and scope.get("root_path_status") == "HOLD_OPEN_DOCUMENT_TIMEOUT"
            and scope.get("source_hashes_same") is True,
            "readback-scope does not preserve identical-copy PASS/root-path HOLD")
    project_value = scope.get("project")
    require(isinstance(project_value, str) and project_value.strip(),
            "readback-scope is missing isolated copy project path")
    project = Path(project_value).resolve(strict=True)
    formal_project = (ROOT / "BMS模块" / NATIVE_PATHS[0]).resolve(strict=True)
    require(project.name.casefold() == Path(NATIVE_PATHS[0]).name.casefold()
            and project != formal_project,
            "readback-scope project must be a separate same-name project copy")
    declared = scope.get("copy_source_sha256")
    before = reopen.get("source_sha256_before")
    after = reopen.get("source_sha256_after")
    reopened = reopen.get("copy_source_sha256")
    require(isinstance(declared, dict) and set(declared) == set(NATIVE_PATHS)
            and isinstance(before, dict) and before == after
            and isinstance(reopened, dict) and set(reopened) == set(NATIVE_PATHS),
            "readback requires complete copy and before/after SHA maps for all native files")
    actual = {}
    module = ROOT / "BMS模块"
    for rel in NATIVE_PATHS:
        copy_sha = sha256(project.parent / rel)
        formal_sha = sha256(module / rel)
        values = {
            copy_sha,
            formal_sha,
            str(before.get(rel, "")).lower(),
            str(after.get(rel, "")).lower(),
            str(declared.get(rel, "")).lower(),
            str(reopened.get(rel, "")).lower(),
        }
        require(all(len(value) == 64 for value in values) and len(values) == 1,
                f"formal/before/after/scope/readback/actual copy native SHA mismatch: {rel}")
        actual[rel] = copy_sha
    require(actual["sch/Schematic1/P1.esch2"] == p1_hash,
            "identical-copy readback P1 SHA does not match current formal P1")
    require(reopen.get("method") == "OFFICIAL_IDENTICAL_COPY_READBACK"
            and reopen.get("readback_method", "OFFICIAL_IDENTICAL_COPY_READBACK")
            == "OFFICIAL_IDENTICAL_COPY_READBACK"
            and reopen.get("readback_status") == "PASS_IDENTICAL_COPY_READBACK"
            and reopen.get("project") == str(project)
            and reopen.get("root_path_readback") is False
            and reopen.get("root_path_status") == "HOLD_OPEN_DOCUMENT_TIMEOUT"
            and reopen.get("source_hashes_same") is True,
            "reopen evidence must identify copy PASS while retaining root-path HOLD")
    return {
        "method": "OFFICIAL_IDENTICAL_COPY_READBACK",
        "project": str(project),
        "root_path_readback": False,
        "root_path_status": "HOLD_OPEN_DOCUMENT_TIMEOUT",
        "source_hashes_same": True,
        "copy_source_sha256": actual,
    }


def point(x, y):
    return round(float(x), 7), round(float(y), 7)


def report_hash_for(path: Path, source_hashes: dict) -> str:
    basename = path.name.casefold()
    matches = [
        value for key, value in source_hashes.items()
        if isinstance(key, str) and key.replace("\\", "/").rsplit("/", 1)[-1].casefold() == basename
    ]
    require(len(matches) == 1, f"formal semantic report must bind exactly one {path.name}")
    return matches[0]


def check_strict_semantic_geometry(
    geometry: dict,
    p1_hash: str,
    expected_component_count: int,
    expected_pin_count: int,
    manifest_row: dict,
    snapshot_row: dict,
) -> dict:
    require(geometry.get("status") == "PASS"
            and geometry.get("electrical_geometry_status") == "PASS",
            "formal geometry report is not PASS")
    require(geometry.get("strict_semantic_geometry") is True,
            "formal geometry report is missing strict semantic verification")
    method = geometry.get("method")
    require(isinstance(method, str)
            and "semantic" in method.casefold()
            and ("adapter" in method.casefold() or "适配器" in method),
            "formal geometry report does not identify the semantic-verifier adapter")

    source_hashes = geometry.get("sourceSHA256")
    require(isinstance(source_hashes, dict), "formal geometry report sourceSHA256 is malformed")
    source_paths = (
        P1_PATH,
        SNAPSHOT_PATH,
        PORTS_PATH,
        MANIFEST_PATH,
        LAYOUT_DIR / "cleanup-manifest.json",
    )
    for path in source_paths:
        actual = sha256(path)
        reported = report_hash_for(path, source_hashes)
        require(reported == actual, f"formal geometry report input hash mismatch: {path.name}")
        if path == P1_PATH:
            require(reported == p1_hash, "formal geometry report is not bound to current P1")

    counts = geometry.get("counts")
    require(isinstance(counts, dict), "formal semantic geometry counts are malformed")
    expected_counts = {
        "pins": expected_pin_count,
        "pinsChecked": expected_pin_count,
        "snapshotPins": expected_pin_count,
        "manifestPins": expected_pin_count,
        "snapshotComponents": expected_component_count,
        "manifestComponents": expected_component_count,
    }
    for key, expected in expected_counts.items():
        require(counts.get(key) == expected,
                f"formal semantic geometry count {key}={counts.get(key)} does not match manifest {expected}")

    expected_defect_keys = {
        "malformedNativeRecords", "missingNativeComponents", "unrepresentedNativeComponents",
        "stalePortRows", "missingPortRows", "portsMissingNET", "malformedPins", "duplicateRefs",
        "duplicatePins", "missingExpectedComponents", "unmanifestedSnapshotComponents",
        "missingExpectedPins", "unexpectedSnapshotPins", "snapshotNCFlagMismatches",
        "floatingExpectedPins", "pinNetMismatches", "ncContact", "lineGroupNameConflicts",
        "clusterNameConflicts", "danglingNetports",
    }
    defect_counts = geometry.get("defectCounts")
    defects = geometry.get("defects")
    require(isinstance(defect_counts, dict) and isinstance(defects, dict),
            "formal semantic geometry defects are missing")
    require(expected_defect_keys <= set(defect_counts) and expected_defect_keys <= set(defects),
            "formal semantic geometry report omits required defect categories")
    require(all(isinstance(defect_counts[key], int) and defect_counts[key] == 0
                for key in expected_defect_keys),
            "formal semantic geometry report contains nonzero defect counts")
    require(all(isinstance(defects[key], list) and not defects[key] for key in expected_defect_keys),
            "formal semantic geometry report contains defect details")

    manifest_pins = manifest_row.get("pins", {})
    snapshot_pins = {}
    for pin in snapshot_row.get("pins", []):
        number = str(pin.get("number"))
        require(number not in snapshot_pins, f"duplicate RSHUNT snapshot pin {number}")
        snapshot_pins[number] = pin
    require(set(snapshot_pins) == set(manifest_pins) == {"1", "2"},
            f"RSHUNT pin set changed: snapshot={sorted(snapshot_pins)}, manifest={sorted(manifest_pins)}")
    require(manifest_pins == {"1": "B0", "2": "PACK-"},
            f"RSHUNT manifest pin mapping changed: {manifest_pins}")

    pin_results = geometry.get("pinResults")
    require(isinstance(pin_results, list), "formal semantic geometry pinResults are missing")
    rshunt_results = [row for row in pin_results if isinstance(row, dict) and row.get("ref") == "RSHUNT"]
    require(len(rshunt_results) == 2, f"semantic report has {len(rshunt_results)} RSHUNT pin results, expected 2")
    results_by_pin = {}
    for row in rshunt_results:
        number = str(row.get("pin"))
        require(number not in results_by_pin, f"duplicate semantic RSHUNT pin result {number}")
        results_by_pin[number] = row
    require(set(results_by_pin) == {"1", "2"},
            f"semantic RSHUNT pin results changed: {sorted(results_by_pin)}")

    joined = {}
    for number, result in results_by_pin.items():
        expected_net = manifest_pins[number]
        actual_names = result.get("actualNames")
        require(result.get("expected") == expected_net,
                f"RSHUNT pin {number} semantic expected net differs from manifest")
        require(actual_names == [expected_net],
                f"RSHUNT pin {number} semantic actualNames mismatch: {actual_names}")
        require(result.get("status") == "PASS",
                f"RSHUNT pin {number} semantic status is {result.get('status')}")
        require(result.get("snapshotNC") is False,
                f"RSHUNT pin {number} is marked NC in semantic snapshot results")
        require(isinstance(result.get("cluster"), str) and result["cluster"],
                f"RSHUNT pin {number} has no physical cluster identity")
        cluster_pins = result.get("clusterPins")
        require(isinstance(cluster_pins, list) and f"RSHUNT.{number}" in cluster_pins,
                f"RSHUNT pin {number} is absent from its semantic physical cluster")
        line_segments = result.get("lineSegments")
        require(isinstance(line_segments, int) and line_segments > 0,
                f"RSHUNT pin {number} has no connected native line segment")
        label_sources = result.get("labelSources")
        require(isinstance(label_sources, list)
                and any(source.get("source") in {"native_LINE_NET_ATTR", "netport_NET"}
                        and source.get("name") == expected_net
                        for source in label_sources if isinstance(source, dict)),
                f"RSHUNT pin {number} has no matching native/netport semantic name source")
        pin_point = point(snapshot_pins[number]["x"], snapshot_pins[number]["y"])
        require(result.get("point") == list(pin_point),
                f"RSHUNT pin {number} semantic point does not match official snapshot coordinates")
        joined[number] = {
            "coordinate": list(pin_point),
            "net": actual_names[0],
            "joined": True,
            "status": result["status"],
            "cluster": result["cluster"],
            "line_segments": line_segments,
            "label_sources": label_sources,
        }
    return joined


def static_rshunt_join(model_input: dict) -> dict:
    p1_hash = sha256(P1_PATH)
    snapshot = read_json(SNAPSHOT_PATH)
    manifest = read_json(MANIFEST_PATH)
    geometry = read_json(GEOMETRY_PATH)
    reopen = read_json(REOPEN_PATH)
    scope = read_json(READBACK_SCOPE_PATH) if READBACK_SCOPE_PATH.is_file() else None

    require(isinstance(manifest, list), "current construction manifest must be a row array")
    manifest_by_ref = {}
    expected_pin_count = 0
    for row in manifest:
        ref = row.get("ref")
        pins = row.get("pins")
        require(isinstance(ref, str) and ref, "construction manifest row is missing a reference")
        require(ref not in manifest_by_ref, f"duplicate reference in construction manifest: {ref}")
        require(isinstance(pins, dict), f"construction manifest pins are malformed: {ref}")
        expected_pin_count += len(pins)
        manifest_by_ref[ref] = row
    expected_component_count = len(manifest_by_ref)

    snapshot_rows = snapshot.get("rows", [])
    require(isinstance(snapshot_rows, list), "official snapshot rows must be an array")
    snapshot_by_ref = {}
    snapshot_pin_count = 0
    for row in snapshot_rows:
        ref = row.get("ref")
        require(isinstance(ref, str) and ref, "snapshot row is missing a reference")
        require(ref not in snapshot_by_ref, f"duplicate reference in current snapshot: {ref}")
        pins = row.get("pins", [])
        require(isinstance(pins, list), f"snapshot pin list is malformed: {ref}")
        snapshot_pin_count += len(pins)
        snapshot_by_ref[ref] = row
    require(set(snapshot_by_ref) == set(manifest_by_ref),
            "current snapshot references do not exactly match the construction manifest")
    require(snapshot_pin_count == expected_pin_count,
            f"snapshot pin count {snapshot_pin_count} does not match manifest {expected_pin_count}")

    reopen_status = reopen.get("status")
    if scope is not None:
        readback_context = validate_identical_copy_readback(reopen, scope, p1_hash)
        require(reopen_status in {
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD",
            "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD",
        }, "identical-copy readback status must retain formal root-path HOLD")
    else:
        readback_context = {
            "method": "OFFICIAL_ROOT_PATH_READBACK",
            "project": reopen.get("project"),
            "root_path_readback": True,
            "root_path_status": reopen_status,
            "source_hashes_same": True,
            "copy_source_sha256": None,
        }
        require(reopen_status in {
            "PASS_READBACK",
            "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING",
        }, "official root-project reopen evidence is not a recognized readback status")
    require(reopen.get("source_unchanged") is True,
            "official readback evidence does not prove native source remained unchanged")
    drc_rows = reopen.get("drc", [])
    require(isinstance(drc_rows, list), "official root-project reopen DRC rows are malformed")
    drc_counts = {"fatal": 0, "error": 0, "warning": 0}
    for row in drc_rows:
        require(isinstance(row, dict), "reopen DRC rows contain a non-object entry")
        kind = normalize_drc_kind(row.get("type", ""))
        if kind is not None:
            count = row.get("count")
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    f"reopen DRC {kind} count is invalid")
            drc_counts[kind] += count
    warning_count = drc_counts["warning"]
    require(drc_counts["fatal"] == 0 and drc_counts["error"] == 0,
            f"reopen DRC fatal/error must be zero: {drc_counts}")
    warning_status = (
        "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD"
        if scope is not None else "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING"
    )
    if warning_count:
        require(warning_count > 0 and reopen.get("warning_cleared") is False,
                "unresolved-warning readback status is inconsistent with DRC evidence")
        require(reopen_status == warning_status
                and reopen.get("drcReview") == "HOLD"
                and reopen.get("layout_release_status") == "HOLD"
                and reopen.get("warning_details_reviewed") is False,
                "unreviewed DRC warnings must remain explicitly HOLD")
    else:
        expected_zero_status = (
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD"
            if scope is not None else "PASS_READBACK"
        )
        require(reopen_status == expected_zero_status,
                "zero-warning readback status conflicts with DRC counts or root-path scope")
    require(reopen.get("source_sha256_before", {}).get("sch/Schematic1/P1.esch2") == p1_hash
            and reopen.get("source_sha256_after", {}).get("sch/Schematic1/P1.esch2") == p1_hash,
            "readback evidence is not bound to current P1")
    require(reopen.get("components") == expected_component_count
            and reopen.get("pins") == expected_pin_count
            and reopen.get("snapshot_match") is True and not reopen.get("snapshot_differences"),
            "root-project reopen readback counts or snapshot match changed")

    manifest_row = [row for row in manifest if row.get("ref") == "RSHUNT"]
    snapshot_row = [row for row in snapshot.get("rows", []) if row.get("ref") == "RSHUNT"]
    require(len(manifest_row) == len(snapshot_row) == 1, "RSHUNT missing or duplicated in current manifest/snapshot")
    manifest_row, snapshot_row = manifest_row[0], snapshot_row[0]
    expected_mpn = model_input["shunt"]["manufacturer_part"]
    require(manifest_row.get("mpn") == expected_mpn == snapshot_row.get("mpn"), "P1 snapshot/manifest RSHUNT MPN mismatch")
    joined = check_strict_semantic_geometry(
        geometry,
        p1_hash,
        expected_component_count,
        expected_pin_count,
        manifest_row,
        snapshot_row,
    )
    require(joined["1"]["net"] == "B0" and joined["2"]["net"] == "PACK-", "RSHUNT pin polarity/net mapping mismatch")

    with BOM_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bom_rows = [row for row in rows if row.get("位号") == "RSHUNT"]
    bom_ok = (len(bom_rows) == 1 and bom_rows[0].get("MPN") == expected_mpn)
    require(bom_ok, "current BOM RSHUNT row does not match the model MPN")

    review = read_json(REVIEW_PATH)
    old_wording = review.get("battery_boundary", {}).get("bms_oc_current_protection", "")
    review_text = REVIEW_PATH.read_text(encoding="utf-8")
    typo_patterns = ("3mV阈值名义12A", "3 mV阈值名义12A", "3mV阈值，名义12A", "3 mV阈值，名义12A")
    typo_detected = any(pattern in review_text for pattern in typo_patterns)
    corrected = ("OCC寄存器 code 3（实际写入/读回未验证）= 6mV" in old_wording
                 and "不是恒流调节或16A硬上限" in old_wording
                 and "早动作角约7.92A" in old_wording)
    semantic_rshunt_results = {
        str(result["pin"]): result
        for result in geometry["pinResults"]
        if isinstance(result, dict) and result.get("ref") == "RSHUNT"
    }
    geometry_status = {
        "path": str(GEOMETRY_PATH.relative_to(ROOT)),
        "sha256": sha256(GEOMETRY_PATH),
        "status": geometry.get("status"),
        "pins": geometry.get("counts", {}).get("pins"),
        "manifest_components": expected_component_count,
        "manifest_pins": expected_pin_count,
        "strict_semantic_geometry": geometry.get("strict_semantic_geometry"),
        "method": geometry.get("method"),
        "pin_results": semantic_rshunt_results,
        "source_matches_p1": True,
        "source_hashes_match_all_inputs": True,
    }
    reopen_status = {
        "path": str(REOPEN_PATH.relative_to(ROOT)),
        "sha256": sha256(REOPEN_PATH),
        "ok": True,
        "readback_method": readback_context["method"],
        "readback_project": readback_context["project"],
        "readback_status": reopen.get("readback_status", reopen_status),
        "root_path_readback": readback_context["root_path_readback"],
        "root_path_status": readback_context["root_path_status"],
        "source_hashes_same": readback_context["source_hashes_same"],
        "copy_source_sha256": readback_context["copy_source_sha256"],
        "component_rows": reopen.get("components"),
        "pins": reopen.get("pins"),
        "snapshot_differences": len(reopen.get("snapshot_differences", [])),
        "drc_warnings": warning_count,
        "warning_details_reviewed": reopen.get("warning_details_reviewed", False),
        "warning_cleared": reopen.get("warning_cleared", False),
        "root_reopen_status": reopen_status,
        "warning_details_reviewed": reopen.get("warning_details_reviewed"),
    }
    return {
        "formal_netlist": {
            "path": str(P1_PATH.relative_to(ROOT)),
            "sha256": p1_hash,
            "mpn": expected_mpn,
            "supplier_part": model_input["shunt"]["supplier_part"],
            "pin1_net": joined["1"]["net"],
            "pin2_net": joined["2"]["net"],
            "matches_model_and_expected_nets": True,
        },
        "bom": {
            "path": str(BOM_PATH.relative_to(ROOT)),
            "sha256": sha256(BOM_PATH),
            "rshunt_rows": len(bom_rows),
            "mpn": bom_rows[0].get("MPN"),
            "purchase_status": bom_rows[0].get("采购状态"),
            "matches_model": bom_ok,
        },
        "old_active_review": {
            "path": str(REVIEW_PATH.relative_to(ROOT)),
            "sha256": sha256(REVIEW_PATH),
            "contains_threshold_typo": typo_detected,
            "corrected_wording_present": corrected,
            "current_text": old_wording,
            "recommended_correction": (
                "当前设计候选OCC寄存器 code 3（实际写入/读回未验证）= 6mV，名义12A；"
                "小阈值负误差−2mV与0.505mΩ分流器高角组合时早动作角约7.92A；"
                "小阈值正误差+2.65mV与0.495mΩ分流器低角组合时晚动作角约17.5A。"
                "OCC仅作延时过流切断，不是恒流调节或16A硬上限。"
            ),
            "correction_applied_before_model_run": corrected,
        },
        "native_schematic_static_join": {
            "evidence_kind": "current_native_P1_strict_semantic_pin_result_binding",
            "path": str(P1_PATH.relative_to(ROOT)),
            "sha256": p1_hash,
            "rshunt_pin1": joined["1"],
            "rshunt_pin2": joined["2"],
            "static_geometry_report": geometry_status,
            "independent_reopen": reopen_status,
            "matches_model_and_expected_nets": True,
            "enet_export_checked": False,
        },
    }


def load_model_module():
    spec = importlib.util.spec_from_file_location("charge_limit_16a_model", MODEL_PATH)
    require(spec is not None and spec.loader is not None, "cannot load the unchanged reference model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def postprocess_outputs(artifacts: dict) -> None:
    result_path = ROOT / "BMS模块" / "仿真" / "结果" / "16A净充电上限验证_result.json"
    check_path = ROOT / "BMS模块" / "检查" / "16A净充电上限验证.json"
    result = read_json(result_path)
    check = read_json(check_path)
    model_hash = sha256(MODEL_PATH)
    entry_hash = sha256(ENTRY)
    input_hash = sha256(INPUT_PATH)
    binding = artifacts["native_schematic_static_join"]
    bom = artifacts["bom"]
    provenance = result["provenance"]
    reopen_binding = binding["independent_reopen"]
    reopen_scope_note = (
        "the official identical-copy reopen record, with all three copy native-file SHA256 values verified against the formal project; the formal root schematic page remains HOLD for user verification"
        if reopen_binding.get("root_path_readback") is False else
        "the official root-project reopen record"
    )
    provenance["artifacts"] = {
        "bom": artifacts["bom"],
        "old_active_review": artifacts["old_active_review"],
        "native_schematic_static_join": binding,
    }
    provenance["native_design_binding_note"] = (
        "The saved P1 RSHUNT pins were checked against strict semantic pinResults from physically connected schematic "
        "clusters: pin 1=B0, pin 2=PACK-. Each result was checked against the manifest net and official snapshot "
        "coordinate, including actualNames, cluster membership, native/netport name sources, and NC state. The "
        "semantic report is SHA-bound to the current P1, snapshot, ports, construction manifest, and cleanup manifest; "
        f"{reopen_scope_note} and unique current BOM RSHUNT MPN were also checked. No formal .enet "
        "export was available or checked; this is not firmware, hardware, bench, or vehicle acceptance."
    )
    provenance["native_design_static_binding"] = {
        "schematic_path": str(P1_PATH.relative_to(ROOT)),
        "schematic_sha256": sha256(P1_PATH),
        "bom_path": str(BOM_PATH.relative_to(ROOT)),
        "bom_sha256": sha256(BOM_PATH),
        "static_geometry_path": str(GEOMETRY_PATH.relative_to(ROOT)),
        "static_geometry_sha256": sha256(GEOMETRY_PATH),
        "independent_reopen_path": str(REOPEN_PATH.relative_to(ROOT)),
        "independent_reopen_sha256": sha256(REOPEN_PATH),
        "independent_reopen_method": reopen_binding["readback_method"],
        "independent_reopen_project": reopen_binding["readback_project"],
        "root_path_readback": reopen_binding["root_path_readback"],
        "root_path_status": reopen_binding["root_path_status"],
        "copy_source_sha256": reopen_binding["copy_source_sha256"],
        "enet_export_checked": False,
    }
    provenance["actual_run_entry"] = {
        "path": str(ENTRY.relative_to(ROOT)),
        "sha256": entry_hash,
        "command_from_project_root": 'py -3 "BMS模块/仿真/源码/replay_16a_with_static_p1.py"',
        "model_source_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_source_sha256": model_hash,
        "input_path": str(INPUT_PATH.relative_to(ROOT)),
        "input_sha256": input_hash,
    }
    provenance["native_design_binding_note"] += (
        f" Actual adapter entry SHA256={entry_hash}; unchanged model source SHA256={model_hash}; "
        f"current input SHA256={input_hash}."
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    check.update({
        "actual_run_entry": provenance["actual_run_entry"],
        "native_schematic_static_binding": provenance["native_design_static_binding"],
        "native_design_changed": False,
        "bom_changed": False,
    })
    check["limitations"] = [
        "计算模型与参数化能量筛查，不是固件、硬件、台架或整车验收。",
        (
            "当前设计绑定为严格语义几何报告中的 RSHUNT 逐针结果（含物理簇、来源网名和NC状态）、五项源SHA、三份native实际字节核对后的同字节副本官方重开记录（正式根路径仍待用户核对）及当前 BOM；本轮未检查 .enet 导出。"
            if reopen_binding.get("root_path_readback") is False else
            "当前设计绑定为严格语义几何报告中的 RSHUNT 逐针结果（含物理簇、来源网名和NC状态）、五项源SHA、官方根路径重开记录及当前 BOM；本轮未检查 .enet 导出。"
        ),
        "1A源超调和20uΩ采样路径误差是设计接受预算，不是实测或当前硬件保证。",
        "alpha<=1/3仍是未验证假设；参考算法在验证前按alpha=1.0回退，但只有在无支路反向/环流资格成立时才允许该回退，否则CCL=0。",
        "EVE 4A/cell最大测试电流只在25±2°C模型窗口使用；必须以覆盖全部36颗电芯的保守温度下界和上界确认每颗均在23..27°C。窗口外缺温度曲线所以CCL=0。",
        "OCC是过流延时切断，不是恒流调节；不增加独立比较器。",
        "本轮未修改 CAD、BOM、项目要求或其他模块。",
    ]
    check["result_sha256"] = sha256(result_path)
    check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    model = load_model_module()
    params = model.load_json(model.INPUT)
    artifacts = static_rshunt_join(params)
    model.verify_artifacts = lambda _params: artifacts
    model.main()
    postprocess_outputs(artifacts)
    print(json.dumps({
        "status": "PASS_MODEL_REPLAY_STATIC_P1_BOUND",
        "model_status": "reference algorithm and deterministic fail-safe tests only",
        "release_status": "BLOCKED; no firmware/hardware/vehicle acceptance",
        "actual_run_entry_sha256": sha256(ENTRY),
        "model_source_sha256": sha256(MODEL_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "p1_sha256": sha256(P1_PATH),
        "bom_sha256": sha256(BOM_PATH),
        "rshunt_pin1_net": artifacts["native_schematic_static_join"]["rshunt_pin1"]["net"],
        "rshunt_pin2_net": artifacts["native_schematic_static_join"]["rshunt_pin2"]["net"],
        "enet_export_checked": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
