"""Rebind the PSU branch calculation to current schematic evidence.

This script updates currentP1Evidence and its top-level current-status summary
in the existing calculation JSON.
It refuses to write if any source hashes, official reopen counts, strict
semantic findings, or the four TAP_B9 input-positive pin results disagree.
It does not run EDA, alter electrical calculations, or create exports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


MODULE = Path(__file__).resolve().parents[2]
PROJECT = MODULE.parent
LAYOUT = MODULE / "检查" / "Layout验证"
P1 = MODULE / "sch" / "Schematic1" / "P1.esch2"
SNAPSHOT = LAYOUT / "current-snapshot.json"
PORTS = LAYOUT / "ports-with-cleanup-whitelist.json"
CLEANUP = LAYOUT / "cleanup-manifest.json"
MANIFEST = MODULE / "检查" / "当前施工输入.json"
GEOMETRY = LAYOUT / "final-geometry.json"
REOPEN = LAYOUT / "root-after-cleanup-reopen.json"
READBACK_SCOPE = MODULE / "检查" / "局部连线" / "readback-scope.json"
RESULT = MODULE / "仿真" / "结果" / "PSU支路闭合计算.json"
NATIVE = (
    "BMS-9S4P.eprj3",
    "sch/Schematic1/P1.esch2",
    "sch/Schematic1/Schematic1.ecfg",
)

BOUND_INPUTS = {
    "sch/Schematic1/P1.esch2": P1,
    "检查/Layout验证/current-snapshot.json": SNAPSHOT,
    "检查/Layout验证/ports-with-cleanup-whitelist.json": PORTS,
    "检查/Layout验证/cleanup-manifest.json": CLEANUP,
    "检查/当前施工输入.json": MANIFEST,
}

POSITIVE_INPUT_PINS = {
    ("PSU5_EXT", "2"): "TAP_B9",
    ("C_PSU_IN1", "1"): "TAP_B9",
    ("C_PSU_IN2", "1"): "TAP_B9",
    ("C_PSU_IN3", "1"): "TAP_B9",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"无法读取 JSON {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_suffix_matches(key: str, relative_path: str) -> bool:
    normalized_key = str(key).replace("\\", "/").rstrip("/").casefold()
    normalized_suffix = relative_path.replace("\\", "/").strip("/").casefold()
    return normalized_key == normalized_suffix or normalized_key.endswith("/" + normalized_suffix)


def bound_hash(source_map: Any, relative_path: str, label: str) -> str:
    require(isinstance(source_map, dict), f"{label} 缺少 sourceSHA256 映射")
    matches = [
        str(value).lower()
        for key, value in source_map.items()
        if path_suffix_matches(str(key), relative_path)
    ]
    require(len(matches) == 1, f"{label} 对 {relative_path} 的源哈希缺失或不唯一")
    require(len(matches[0]) == 64, f"{label} 中 {relative_path} 的 SHA256 格式错误")
    return matches[0]


def field_count(counts: dict[str, Any], names: tuple[str, ...], label: str) -> int:
    for name in names:
        value = counts.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            require(value >= 0, f"{label} 不能为负数")
            return value
    raise RuntimeError(f"{label} 在正式几何报告中缺失")


def get_manifest_expected(manifest: Any) -> tuple[dict[tuple[str, str], str], int, int]:
    require(isinstance(manifest, list), "当前施工输入必须是器件清单数组")
    result: dict[tuple[str, str], str] = {}
    refs: set[str] = set()
    pin_count = 0
    for row in manifest:
        if not isinstance(row, dict) or not isinstance(row.get("pins"), dict):
            continue
        ref = str(row.get("ref", ""))
        require(bool(ref), "当前施工输入存在缺少 ref 的器件行")
        require(ref not in refs, f"当前施工输入存在重复 ref: {ref}")
        refs.add(ref)
        pin_count += len(row["pins"])
        for pin, net in row["pins"].items():
            key = (ref, str(pin))
            require(key not in result, f"当前施工输入存在重复 pin: {ref}.{pin}")
            result[key] = str(net)
    return result, len(refs), pin_count


def validate_geometry(
    geometry: Any,
    snapshot: Any,
    manifest: Any,
    current_hashes: dict[str, str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    require(isinstance(geometry, dict), "正式 final-geometry.json 必须是对象")
    require(geometry.get("strict_semantic_geometry") is True, "几何报告未声明 strict semantic geometry")
    semantic_status = geometry.get("electrical_geometry_status", geometry.get("status"))
    require(semantic_status == "PASS", f"几何语义状态不是 PASS: {semantic_status!r}")
    require("semantic" in str(geometry.get("method", "")).casefold(), "几何报告方法未声明语义连接验证")

    defect_counts = geometry.get("defectCounts")
    require(isinstance(defect_counts, dict) and bool(defect_counts), "几何报告缺少完整 defectCounts")
    invalid_defect_counts = {
        name: value for name, value in defect_counts.items()
        if not isinstance(value, int) or isinstance(value, bool) or value < 0
    }
    require(not invalid_defect_counts,
            f"语义缺陷计数格式无效: {invalid_defect_counts}")
    nonzero_defects = {name: value for name, value in defect_counts.items() if value != 0}
    require(not nonzero_defects, f"语义几何仍有缺陷: {nonzero_defects}")
    defects = geometry.get("defects")
    if isinstance(defects, list):
        require(not defects, "语义几何 defects 非空")
    elif isinstance(defects, dict):
        require(set(defects) == set(defect_counts),
                "语义几何 defects 与 defectCounts 分类不一致")
        nonempty_defects = {
            name: value for name, value in defects.items()
            if not (
                isinstance(value, list) and not value
                or isinstance(value, int) and not isinstance(value, bool) and value == 0
            )
        }
        require(not nonempty_defects,
                f"语义几何存在非空或格式异常的 defects: {nonempty_defects}")
    else:
        raise RuntimeError("语义几何 defects 缺失或格式异常")

    counts = geometry.get("counts")
    require(isinstance(counts, dict), "几何报告缺少 counts")
    rows = snapshot.get("rows") if isinstance(snapshot, dict) else None
    require(isinstance(rows, list), "官方 current-snapshot 缺少 rows")
    refs = [str(row.get("ref", "")) for row in rows if isinstance(row, dict)]
    require(len(refs) == len(rows) and all(refs), "官方 snapshot 行缺少 ref")
    require(len(refs) == len(set(refs)), "官方 snapshot 有重复 ref")
    pins = 0
    for row in rows:
        row_pins = row.get("pins") if isinstance(row, dict) else None
        require(isinstance(row_pins, list), f"snapshot {row.get('ref')} 的 pins 不是数组")
        pins += len(row_pins)

    components = field_count(counts, ("components", "snapshotComponents", "manifestComponents"), "几何组件数")
    pin_count = field_count(counts, ("pins", "pinsChecked", "snapshotPins", "manifestPins"), "几何 pin 数")
    wires = field_count(counts, ("wires", "activeNativeLines", "segments"), "几何线段数")
    labels = field_count(counts, ("labels", "activeNetportRows", "portRowsSource"), "几何端口数")
    require(components == len(rows), f"几何组件数 {components} 与官方 snapshot {len(rows)} 不同")
    require(pin_count == pins, f"几何 pin 数 {pin_count} 与官方 snapshot {pins} 不同")

    if "snapshotComponents" in counts:
        require(counts["snapshotComponents"] == len(rows), "语义 snapshotComponents 与官方 snapshot 不同")
    if "snapshotPins" in counts:
        require(counts["snapshotPins"] == pins, "语义 snapshotPins 与官方 snapshot 不同")
    if "pinsChecked" in counts:
        require(counts["pinsChecked"] == pins, "语义 pinsChecked 与官方 snapshot 不同")

    source_map = geometry.get("sourceSHA256")
    for relative_path, path in BOUND_INPUTS.items():
        expected = current_hashes[relative_path]
        observed = bound_hash(source_map, relative_path, "正式几何报告")
        require(observed == expected, f"正式几何报告源哈希不匹配: {relative_path}")

    manifest_nets, manifest_components, manifest_pin_count = get_manifest_expected(manifest)
    require(manifest_components == len(rows),
            f"施工输入组件数 {manifest_components} 与官方 snapshot {len(rows)} 不同")
    require(manifest_pin_count == pins,
            f"施工输入 pin 数 {manifest_pin_count} 与官方 snapshot {pins} 不同")
    require(counts.get("manifestComponents") == manifest_components,
            "semantic manifestComponents 与当前施工输入不同")
    require(counts.get("manifestPins") == manifest_pin_count,
            "semantic manifestPins 与当前施工输入不同")
    for pin, expected in POSITIVE_INPUT_PINS.items():
        require(manifest_nets.get(pin) == expected, f"施工输入对 {pin[0]}.{pin[1]} 的期望网不是 {expected}")

    pin_results = geometry.get("pinResults")
    require(isinstance(pin_results, list), "正式几何报告缺少完整 semantic pinResults")
    validated: list[dict[str, Any]] = []
    for (ref, number), expected in POSITIVE_INPUT_PINS.items():
        matches = [
            row for row in pin_results
            if isinstance(row, dict) and str(row.get("ref")) == ref and str(row.get("pin")) == number
        ]
        require(len(matches) == 1, f"semantic pinResults 中 {ref}.{number} 缺失或重复")
        result = matches[0]
        actual_names = result.get("actualNames")
        require(result.get("expected") == expected, f"{ref}.{number} 的 semantic expected 不匹配")
        require(isinstance(actual_names, list) and set(map(str, actual_names)) == {expected},
                f"{ref}.{number} 实际网络不是唯一 {expected}: {actual_names!r}")
        require(result.get("status") == "PASS", f"{ref}.{number} semantic pin status 不是 PASS")
        require(result.get("snapshotNC") is False, f"{ref}.{number} 是 NC 或缺少 NC=false 证据")
        require(isinstance(result.get("lineSegments"), int) and result["lineSegments"] > 0,
                f"{ref}.{number} 缺少物理线段连接证据")
        netports = result.get("clusterNetports")
        require(isinstance(netports, list) and any(str(item).split("[", 1)[0] == expected for item in netports),
                f"{ref}.{number} 的物理簇缺少 {expected} netport 证据")
        validated.append(copy.deepcopy(result))

    return {"components": components, "pins": pin_count, "wires": wires, "labels": labels}, validated


def reopen_pin_sha(source_map: Any) -> str:
    return bound_hash(source_map, "sch/Schematic1/P1.esch2", "官方重开记录")


def parse_drc(reopen: dict[str, Any]) -> tuple[dict[str, int], bool]:
    drc = reopen.get("drc")
    counts: dict[str, int] = {}
    if isinstance(drc, list):
        for row in drc:
            require(isinstance(row, dict) and "type" in row and "count" in row, "重开 DRC 列表行格式错误")
            name, count = str(row["type"]), row["count"]
            require(name not in counts, f"重开 DRC 类别重复: {name}")
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    f"重开 DRC {name} 数量无效")
            counts[name] = count
    elif isinstance(drc, dict):
        for name, count in drc.items():
            if isinstance(count, int) and not isinstance(count, bool):
                require(count >= 0, f"重开 DRC {name} 数量无效")
                counts[str(name)] = count
    else:
        raise RuntimeError("官方根重开记录缺少可解析的 DRC 数量")
    reviewed = reopen.get("warning_details_reviewed", reopen.get("warningDetailsReviewed", False))
    if isinstance(drc, dict):
        reviewed = drc.get("warningDetailsReviewed", reviewed)
    return counts, reviewed is True


def normalize_drc_kind(value: Any) -> str | None:
    compact = "".join(str(value).strip().casefold().split()).replace("_", "").replace("-", "")
    if compact in {"fatal", "fatals", "fatalerror", "fatalerrors"}:
        return "fatal"
    if compact in {"error", "errors", "err"}:
        return "error"
    if compact in {"warn", "warning", "warnings"}:
        return "warning"
    return None


def validate_identical_copy_readback(
    reopen: dict[str, Any],
    scope: Any,
    source_before: Any,
    source_after: Any,
    current_native_hashes: dict[str, str],
) -> dict[str, Any]:
    require(isinstance(scope, dict), "readback-scope.json格式无效")
    expected = {
        "method": "OFFICIAL_IDENTICAL_COPY_READBACK",
        "root_path_readback": False,
        "root_path_status": "HOLD_OPEN_DOCUMENT_TIMEOUT",
        "source_hashes_same": True,
    }
    for key, value in expected.items():
        actual = scope.get(key)
        matches = type(actual) is bool and actual is value if isinstance(value, bool) else actual == value
        require(matches, f"readback-scope.json {key}与同字节副本/根路径HOLD约定不符")
    require(isinstance(source_before, dict) and source_before == source_after,
            "副本重开缺少一致的三份源文件前后SHA映射")
    project_value = scope.get("project")
    require(isinstance(project_value, str) and project_value.strip(),
            "readback-scope.json缺少隔离副本project路径")
    project = Path(project_value).resolve(strict=True)
    formal_project = (MODULE / NATIVE[0]).resolve(strict=True)
    require(project.name.casefold() == Path(NATIVE[0]).name.casefold()
            and project != formal_project,
            "readback-scope.project必须指向不同于正式入口的同名副本工程")
    declared = scope.get("copy_source_sha256")
    require(isinstance(declared, dict) and set(declared) == set(NATIVE),
            "readback-scope必须声明三份native的SHA256")
    reopen_copy_hashes = reopen.get("copy_source_sha256")
    require(isinstance(reopen_copy_hashes, dict) and set(reopen_copy_hashes) == set(NATIVE),
            "重开证据没有列出三份副本native SHA256")

    actual_copy_hashes: dict[str, str] = {}
    for relative in NATIVE:
        formal_sha = str(current_native_hashes.get(relative, "")).lower()
        before_sha = str(source_before.get(relative, "")).lower()
        after_sha = str(source_after.get(relative, "")).lower()
        declared_sha = str(declared.get(relative, "")).lower()
        recorded_sha = str(reopen_copy_hashes.get(relative, "")).lower()
        copy_sha = sha256(project.parent / relative)
        require(all(len(value) == 64 for value in (
            formal_sha, before_sha, after_sha, declared_sha, recorded_sha, copy_sha
        )), f"{relative}的native SHA256缺失或格式错误")
        require(len({formal_sha, before_sha, after_sha, declared_sha, recorded_sha, copy_sha}) == 1,
                f"正式入口/before-after/scope/重开记录/实际副本SHA不一致：{relative}")
        actual_copy_hashes[relative] = copy_sha

    require(reopen.get("method") == expected["method"]
            and reopen.get("readback_method", expected["method"]) == expected["method"]
            and reopen.get("project") == str(project)
            and reopen.get("readback_status") == "PASS_IDENTICAL_COPY_READBACK"
            and reopen.get("root_path_readback") is False
            and reopen.get("root_path_status") == expected["root_path_status"]
            and reopen.get("source_hashes_same") is True,
            "root-after-cleanup-reopen.json没有保持隔离副本PASS/正式根路径HOLD")
    return {
        "method": expected["method"],
        "project": str(project),
        "root_path_readback": False,
        "root_path_status": expected["root_path_status"],
        "source_hashes_same": True,
        "copy_source_sha256": actual_copy_hashes,
    }


def validate_reopen(
    reopen: Any,
    counts: dict[str, int],
    current_p1_sha: str,
    current_native_hashes: dict[str, str],
    scope: Any = None,
) -> tuple[dict[str, int], dict[str, int], bool, dict[str, Any]]:
    require(isinstance(reopen, dict), "官方根重开记录必须是对象")
    status = reopen.get("status")
    identical_copy = scope is not None
    if identical_copy:
        require(status in {
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD",
            "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD",
        }, f"同字节副本重开状态不可用于重绑: {status!r}")
    else:
        require(status in {
            "PASS_READBACK",
            "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING",
        }, f"正式根路径重开状态不可用于重绑: {status!r}")
    require(reopen.get("source_unchanged") is True, "官方根重开未证明源前后不变")
    require(reopen.get("snapshot_match") is True, "官方根重开未证明 snapshot 匹配")
    differences = reopen.get("snapshot_differences", reopen.get("snapshotDifferences"))
    require(differences in ([], 0, None), f"官方根重开存在 snapshot 差异: {differences!r}")

    before = reopen.get("source_sha256_before", reopen.get("sourceSHA256Before"))
    after = reopen.get("source_sha256_after", reopen.get("sourceSHA256After"))
    require(isinstance(before, dict) and isinstance(after, dict), "官方根重开缺少源前后 SHA256 映射")
    require(before == after, "官方根重开源文件前后哈希不同")
    require(reopen_pin_sha(after) == current_p1_sha, "官方根重开中的 P1 SHA 与当前 P1 不同")
    if identical_copy:
        readback_scope = validate_identical_copy_readback(
            reopen, scope, before, after, current_native_hashes
        )
    else:
        require(all(str(before.get(relative, "")).lower() == current_native_hashes[relative]
                    for relative in NATIVE), "正式根重开三份native SHA未绑定当前入口")
        readback_scope = {
            "method": "OFFICIAL_ROOT_PATH_READBACK",
            "project": reopen.get("project"),
            "root_path_readback": True,
            "root_path_status": status,
            "source_hashes_same": True,
            "copy_source_sha256": None,
        }

    reopened_components = reopen.get("components")
    reopened_pins = reopen.get("pins")
    require(reopened_components == counts["components"], "官方根重开组件数与正式几何不同")
    require(reopened_pins == counts["pins"], "官方根重开 pin 数与正式几何不同")
    drc_counts, warnings_reviewed = parse_drc(reopen)
    warning_count = sum(
        count for name, count in drc_counts.items()
        if normalize_drc_kind(name) == "warning"
    )
    fatal_count = sum(count for name, count in drc_counts.items() if normalize_drc_kind(name) == "fatal")
    error_count = sum(count for name, count in drc_counts.items() if normalize_drc_kind(name) == "error")
    require(fatal_count == 0 and error_count == 0,
            f"重开DRC存在fatal/error，拒绝模型重绑：fatal={fatal_count} error={error_count}")
    warning_status = (
        "PASS_IDENTICAL_COPY_READBACK_WITH_UNRESOLVED_DRC_WARNING_ROOT_PATH_HOLD"
        if identical_copy else "PASS_READBACK_WITH_UNRESOLVED_DRC_WARNING"
    )
    if warning_count:
        require(status == warning_status,
                "当前重开有DRC警告，但状态未明确保留警告/HOLD")
        require(warning_count > 0, "未解决DRC警告状态没有对应的warning计数")
        require(reopen.get("warning_cleared") is False,
                "重开状态声称警告未解决，但warning_cleared不是false")
        require(reopen.get("drcReview") == "HOLD"
                and reopen.get("layout_release_status") == "HOLD"
                and reopen.get("warning_details_reviewed") is False,
                "未审查DRC警告必须保持DRC/Layout HOLD且warning_details_reviewed=false")
    else:
        expected_zero = (
            "PASS_IDENTICAL_COPY_READBACK_ROOT_PATH_HOLD" if identical_copy else "PASS_READBACK"
        )
        require(status == expected_zero, "零警告重开状态与DRC计数不一致")
    return ({"components": reopened_components, "pins": reopened_pins}, drc_counts, warnings_reviewed, readback_scope)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    for path in [*BOUND_INPUTS.values(), GEOMETRY, REOPEN, RESULT]:
        require(path.is_file(), f"必需证据文件不存在: {path}")

    initial_hashes = {relative: sha256(path) for relative, path in BOUND_INPUTS.items()}
    current_native_hashes = {relative: sha256(MODULE / relative) for relative in NATIVE}
    geometry_sha = sha256(GEOMETRY)
    reopen_sha = sha256(REOPEN)
    result_sha_before = sha256(RESULT)
    current_p1_sha = initial_hashes["sch/Schematic1/P1.esch2"]

    snapshot = read_json(SNAPSHOT)
    ports = read_json(PORTS)
    cleanup = read_json(CLEANUP)
    manifest = read_json(MANIFEST)
    geometry = read_json(GEOMETRY)
    reopen = read_json(REOPEN)
    scope = read_json(READBACK_SCOPE) if READBACK_SCOPE.is_file() else None
    result_doc = read_json(RESULT)
    require(isinstance(result_doc, dict) and isinstance(result_doc.get("currentP1Evidence"), dict),
            "PSU 计算 JSON 缺少 currentP1Evidence")

    # Keep these reads in the binding contract even though semantic pinResults
    # carry the electrical interpretation of their contents.
    require(isinstance(ports, dict) and isinstance(ports.get("ports"), list),
            "正式端口源格式异常")
    require(isinstance(cleanup, dict), "正式 cleanup manifest 格式异常")

    geometry_counts, positive_pin_results = validate_geometry(
        geometry, snapshot, manifest, initial_hashes
    )
    reopen_counts, drc_counts, warnings_reviewed, readback_scope = validate_reopen(
        reopen, geometry_counts, current_p1_sha, current_native_hashes, scope
    )

    previous = result_doc["currentP1Evidence"]
    evidence = copy.deepcopy(previous)
    evidence["path"] = "BMS模块/sch/Schematic1/P1.esch2"
    evidence["sha256"] = current_p1_sha
    evidence["staticGeometry"] = {
        "path": "BMS模块/检查/Layout验证/final-geometry.json",
        "sha256": geometry_sha,
        "status": geometry.get("status"),
        "semanticStatus": geometry.get("electrical_geometry_status", geometry.get("status")),
        "strictSemanticGeometry": True,
        "components": geometry_counts["components"],
        "pins": geometry_counts["pins"],
        "wires": geometry_counts["wires"],
        "labels": geometry_counts["labels"],
        "defectCounts": copy.deepcopy(geometry["defectCounts"]),
        "sourceSHA256": copy.deepcopy(geometry["sourceSHA256"]),
    }

    independent = evidence.get("independentReopen")
    if isinstance(independent, dict):
        independent["current"] = False
        independent["historical"] = True
        independent["note"] = "Historical independent reopen before current repair; current source is bound to rootCleanupReopen."

    evidence["rootCleanupReopen"] = {
        "path": "BMS模块/检查/Layout验证/root-after-cleanup-reopen.json",
        "sha256": reopen_sha,
        "status": reopen["status"],
        "readbackMethod": readback_scope["method"],
        "readbackProject": readback_scope["project"],
        "readbackStatus": reopen.get("readback_status", reopen["status"]),
        "rootPathReadback": readback_scope["root_path_readback"],
        "rootPathStatus": readback_scope["root_path_status"],
        "sourceHashesSame": readback_scope["source_hashes_same"],
        "copySourceSHA256": readback_scope["copy_source_sha256"],
        "components": reopen_counts["components"],
        "pins": reopen_counts["pins"],
        "sourceUnchanged": True,
        "p1Sha256": current_p1_sha,
        "snapshotMatch": True,
        "snapshotDifferences": 0,
        "drc": {**drc_counts, "warningDetailsReviewed": warnings_reviewed},
        "warningCleared": reopen.get("warning_cleared", False),
        "scope": (
            "Official identical-copy read-only reopen; all three native file bytes were rehashed and match the formal project, before/after maps, and readback-scope. The formal root project page remains HOLD after open timeout. Any DRC warning remains unresolved; this is not PCB routing, thermal, or fault-interruption acceptance."
            if scope is not None else
            "Official root-project read-only reopen, source SHA and pin/snapshot comparison. Any DRC warning remains unresolved and is recorded separately; this is not PCB routing, thermal, or fault-interruption acceptance."
        ),
    }
    evidence["positiveInputSemanticPins"] = {
        "reportPath": "BMS模块/检查/Layout验证/final-geometry.json",
        "reportSHA256": geometry_sha,
        "allExpectedAndActualNames": "TAP_B9",
        "pins": positive_pin_results,
    }
    evidence["boundSourceSHA256"] = {
        relative: initial_hashes[relative] for relative in BOUND_INPUTS
    }
    evidence["scope"] = (
        (
            "Current schematic semantic geometry and official identical-copy readback; the formal root page remains HOLD; "
            if scope is not None else "Current schematic semantic geometry and official root-project reopen only; "
        )
        + "not PCB DRC, board layout, thermal, wire/connector capacity, or fault-interruption acceptance."
    )

    # Detect evidence replacement or edits that happen while this script is
    # preparing the write. Any disagreement leaves the calculation untouched.
    require(sha256(RESULT) == result_sha_before, "PSU 计算 JSON 在重绑期间发生变化")
    require(sha256(GEOMETRY) == geometry_sha, "final-geometry.json 在重绑期间发生变化")
    require(sha256(REOPEN) == reopen_sha, "root reopen 证据在重绑期间发生变化")
    for relative, path in BOUND_INPUTS.items():
        require(sha256(path) == initial_hashes[relative], f"源文件在重绑期间发生变化: {relative}")

    fatal_count = sum(
        count for name, count in drc_counts.items()
        if normalize_drc_kind(name) == "fatal"
    )
    error_count = sum(
        count for name, count in drc_counts.items()
        if normalize_drc_kind(name) == "error"
    )
    warning_count = sum(
        count for name, count in drc_counts.items()
        if normalize_drc_kind(name) == "warning"
    )
    if readback_scope["method"] == "OFFICIAL_IDENTICAL_COPY_READBACK":
        readback_status = (
            "同字节副本官方只读重开PASS（208器件/700引脚）；"
            "正式根路径图页打开超时，待用户核对。"
        )
    else:
        readback_status = "正式根路径官方只读重开PASS（208器件/700引脚）。"
    warning_status = (
        f"当前读回DRC为{fatal_count} fatal/{error_count} error/{warning_count} warning；"
        "警告规则/位号明细尚未审阅，drcReview=HOLD，Layout正式放行=HOLD。"
        if warning_count else
        f"当前读回DRC为{fatal_count} fatal/{error_count} error/0 warning；Layout正式放行仍为HOLD。"
    )
    result_doc["status"] = (
        "条件纸面方案，仅供 Layout 初始摆放；PSU 输入已接 TAP_B9；"
        f"{readback_status}{warning_status}"
        "辅助正极线支路故障/短路分断、实际电池 DC 预期短路电流及熔断总清除 I²t 配合均 OPEN；"
        "未冻结、未放行投板。"
    )
    result_doc["currentP1Evidence"] = evidence
    atomic_write_json(RESULT, result_doc)


if __name__ == "__main__":
    main()
