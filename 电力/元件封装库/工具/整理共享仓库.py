#!/usr/bin/env python3
"""Filter the legacy shared component library by exact current MPN evidence.

The script reads a pinned Git tree without checking it out, writes only into an
independent publish directory, and never changes the source repository.  It is
stdlib-only and intentionally keeps the generated current-model/native-assets
files owned by the separate exporter untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import posixpath
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any


BASE_COMMIT = "1a204748f491d6e71beb4cd08fda1af0521bcc01"
DEFAULT_REPO = "tmp/vehicle-power-hardware-common-sync"
DEFAULT_OUTPUT = "tmp/vehicle-power-hardware-common-publish"
DEFAULT_INDEX = "元件封装库/索引/元件总表.json"
DEFAULT_PROCUREMENT = "元件封装库/索引/采购补充.json"
DEFAULT_REPORT = "元件封装库/检查/共享仓库清理.json"
CATALOG_JSON = "库/元器件目录.json"
CATALOG_CSV = "库/元器件目录.csv"
ASSOCIATION_JSON = "库/型号封装关联.json"
SUPPLEMENT_JSON = "库/封装补充依据.json"
DATASHEET_INDEX = "数据手册/索引.json"
IMPORT_ZIP = "库/可导入封装库.zip"
SUPPLIER_ZIP = "库/嘉立创库数据.zip"
PROTECTED_PREFIXES = ("库/原生片段/",)
PROTECTED_FILES = ("库/当前型号与原生资产.json",)
ALLOWED_PREFIXES = ("TI", "Murata", "Littelfuse", "onsemi")


class AuditError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def run_git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise AuditError(f"git {' '.join(args)} 失败: {err}")
    return proc.stdout


def git_blob(repo: Path, revision: str, relpath: str) -> bytes:
    return run_git(repo, "show", f"{revision}:{relpath}")


def tree_blobs(repo: Path, revision: str) -> dict[str, str]:
    raw = run_git(repo, "ls-tree", "-r", "-z", "--full-tree", revision)
    result: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid = meta.decode("ascii").split(" ", 2)
        if kind == "blob":
            result[path.decode("utf-8")] = oid
    return result


def load_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8-sig"))
    except Exception as exc:  # noqa: BLE001 - produce a source-specific diagnostic
        raise AuditError(f"无法解析 {label}: {exc}") from exc


def active_model_set(index: dict[str, Any], procurement: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    models: dict[str, dict[str, Any]] = {}
    parts = index.get("parts", [])
    if not isinstance(parts, list):
        raise AuditError("元件总表 JSON 的 parts 不是列表")
    for row in parts:
        model = str(row.get("mpn") or "").strip()
        if not model:
            continue
        key = norm(model)
        item = models.setdefault(key, {"model": model, "origins": []})
        if "原生总表MPN" not in item["origins"]:
            item["origins"].append("原生总表MPN")

    records = procurement.get("records", [])
    if not isinstance(records, list):
        raise AuditError("采购补充 JSON 的 records 不是列表")
    unique_rows = 0
    for row in records:
        if not str(row.get("relation") or "").startswith("BOM独有"):
            continue
        unique_rows += 1
        model = str(row.get("mpn_source") or "").strip()
        if not model:
            continue
        key = norm(model)
        item = models.setdefault(key, {"model": model, "origins": []})
        if "采购独有MPN" not in item["origins"]:
            item["origins"].append("采购独有MPN")
    return models, [{"count": str(unique_rows), "basis": "采购补充中 relation 以 BOM独有 开头的行"}]


def verified_aliases(index: dict[str, Any], active: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    aliases: dict[str, dict[str, str]] = {}
    for row in index.get("instances", []):
        mpn = str(row.get("mpn") or "").strip()
        carrier = str(row.get("carrier_mpn") or "").strip()
        if not mpn or not carrier or norm(mpn) == norm(carrier):
            continue
        if norm(mpn) not in active:
            continue
        for prefix in ALLOWED_PREFIXES:
            expected = f"{prefix} {carrier}"
            if norm(mpn) == norm(expected):
                aliases.setdefault(norm(carrier), {"library_model": carrier, "current_mpn": mpn, "prefix": prefix})
                break
    return aliases


def resolve_model(value: Any, active: dict[str, dict[str, Any]], aliases: dict[str, dict[str, str]]) -> dict[str, str] | None:
    model = str(value or "").strip()
    if not model:
        return None
    key = norm(model)
    if key in active:
        return {"model": model, "basis": "精确MPN命中当前原生总表或采购独有库存"}
    if key in aliases:
        alias = aliases[key]
        return {
            "model": model,
            "basis": "已核对原生carrier_mpn，实际MPN精确等于允许厂商前缀 + carrier完整料号",
            "current_mpn": alias["current_mpn"],
            "prefix": alias["prefix"],
            "carrier_mpn": alias["library_model"],
        }
    return None


def model_key(value: Any) -> str:
    return norm(value)


def source_decision(
    path: str,
    blobs: dict[str, str],
    action: str,
    reason: str,
    output_path: str | None = None,
    source_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "source_path": path,
        "source_blob_oid": blobs.get(path),
        "source_sha256": source_sha,
        "action": action,
        "reason": reason,
        "output_path": output_path,
    }


def validate_local_ref(base_path: str, value: str) -> str:
    value = value.replace("\\", "/").strip()
    if not value or "://" in value or value.startswith("//"):
        raise AuditError(f"不是本地相对路径依赖: {value!r}")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(base_path), value))
    if joined == ".." or joined.startswith("../") or joined.startswith("/"):
        raise AuditError(f"路径依赖逃出仓库: {base_path} -> {value}")
    return joined


def walk_explicit_pdf_dependencies(obj: Any, source_path: str) -> list[tuple[str, str | None, str]]:
    found: list[tuple[str, str | None, str]] = []

    def walk(value: Any, location: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if isinstance(child, str) and key_text.lower() in {"manufacturer_pdf", "file", "pdf_file"} and child.lower().endswith(".pdf"):
                    expected = value.get(f"{key_text}_sha256")
                    if not expected and key_text == "file":
                        expected = value.get("sha256")
                    found.append((validate_local_ref(source_path, child), str(expected) if expected else None, f"{location}.{key_text}"))
                else:
                    walk(child, f"{location}.{key_text}")
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{location}[{idx}]")

    walk(obj)
    return found


def split_member_ref(ref: Any) -> tuple[str, str] | None:
    if not isinstance(ref, str) or "::" not in ref:
        return None
    archive, member = ref.split("::", 1)
    archive = archive.replace("\\", "/").lstrip("./")
    member = member.replace("\\", "/").lstrip("/")
    return archive, member


def archive_ref_matches(archive_ref: str, expected_archive: str) -> bool:
    ref_name = PurePosixPath(archive_ref).name.casefold()
    expected_name = PurePosixPath(expected_archive).name.casefold()
    return ref_name == expected_name


def filter_catalog(source: dict[str, Any], active: dict[str, dict[str, Any]], aliases: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    output_models: set[str] = set()
    source_rows = source.get("components", [])
    for row in source_rows:
        src_model = str(row.get("model") or "").strip()
        primary_match = resolve_model(src_model, active, aliases)
        active_alts = [alt for alt in row.get("alternatives", []) if resolve_model(alt.get("model"), active, aliases)]
        active_matches = [match for match in row.get("matches", []) if resolve_model(match.get("model"), active, aliases)]
        selected = row.get("selected_catalog_match")
        selected_active = selected if isinstance(selected, dict) and resolve_model(selected.get("model"), active, aliases) else None
        if primary_match:
            new_row = copy.deepcopy(row)
            new_row["matches"] = copy.deepcopy(active_matches)
            new_row["alternatives"] = copy.deepcopy(active_alts)
            if selected_active is not None:
                new_row["selected_catalog_match"] = copy.deepcopy(selected_active)
            else:
                new_row.pop("selected_catalog_match", None)
            kept.append(new_row)
            output_models.add(norm(src_model))
            decisions.append({
                "source_model": src_model,
                "action": "保留精确在用型号目录项",
                "match": primary_match,
                "retained_exact_matches": [x.get("model") for x in active_matches],
                "retained_active_alternatives": [x.get("model") for x in active_alts],
                "dropped_unused_alternative_count": len(row.get("alternatives", [])) - len(active_alts),
            })
        else:
            decisions.append({
                "source_model": src_model,
                "action": "主型号未命中；仅在有精确在用alternatives时提取其原始条目",
                "match": None,
                "retained_exact_matches": [],
                "retained_active_alternatives": [x.get("model") for x in active_alts],
                "dropped_unused_alternative_count": len(row.get("alternatives", [])) - len(active_alts),
            })
        for alt in active_alts:
            alt_model = str(alt.get("model") or "").strip()
            key = norm(alt_model)
            if key in output_models:
                continue
            alt_match = resolve_model(alt_model, active, aliases)
            promoted = {
                "model": alt_model,
                "match_status": "源目录alternatives中的精确在用型号；保留为来源候选，不表示选用",
                "manufacturer": alt.get("manufacturer") or row.get("manufacturer"),
                "matches": [copy.deepcopy(alt)],
                "alternatives": [],
                "source_component_model": src_model,
            }
            kept.append(promoted)
            output_models.add(key)
            decisions.append({
                "source_model": alt_model,
                "source_parent_model": src_model,
                "action": "从原始alternatives提取精确在用型号原始记录",
                "match": alt_match,
                "retained_exact_matches": [alt_model],
                "retained_active_alternatives": [],
                "dropped_unused_alternative_count": 0,
            })
    filtered = {
        "description": "Pinned legacy catalog filtered by exact current MPN evidence. Retention records source availability only; it does not imply present design adoption, pin validation, or mechanical approval.",
        "components": kept,
    }
    return filtered, decisions


def catalog_csv_bytes(catalog: dict[str, Any]) -> bytes:
    fields = [
        "model", "match_status", "manufacturer", "supplier_id", "catalog_model",
        "catalog_package", "datasheet", "product_url", "alternative_models", "alternative_supplier_ids",
    ]
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for component in catalog["components"]:
        candidates = component.get("matches") or []
        selected = component.get("selected_catalog_match")
        if not isinstance(selected, dict) or norm(selected.get("model")) != norm(component.get("model")):
            selected = next((x for x in candidates if norm(x.get("model")) == norm(component.get("model"))), None)
        selected = selected or (candidates[0] if candidates else {})
        alternatives = component.get("alternatives") or []
        writer.writerow({
            "model": component.get("model", ""),
            "match_status": component.get("match_status", ""),
            "manufacturer": component.get("manufacturer", ""),
            "supplier_id": selected.get("supplier_id", ""),
            "catalog_model": selected.get("model", ""),
            "catalog_package": selected.get("catalog_package", ""),
            "datasheet": selected.get("datasheet", ""),
            "product_url": selected.get("product_url", ""),
            "alternative_models": "; ".join(str(x.get("model", "")) for x in alternatives),
            "alternative_supplier_ids": "; ".join(str(x.get("supplier_id", "")) for x in alternatives),
        })
    return out.getvalue().encode("utf-8")


def filter_associations(source: dict[str, Any], active: dict[str, dict[str, Any]], aliases: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, list[dict[str, str]]]]]:
    kept: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    archive_refs: dict[str, dict[str, list[dict[str, str]]]] = {IMPORT_ZIP: {}, SUPPLIER_ZIP: {}}
    for row in source.get("components", []):
        model = str(row.get("model") or "").strip()
        match = resolve_model(model, active, aliases)
        if not match:
            decisions.append({"source_model": model, "action": "删除未匹配型号关联", "match": None, "retained_candidates": 0})
            continue
        new_row = copy.deepcopy(row)
        candidates = new_row.get("footprint_candidates", [])
        for candidate in candidates:
            for archive, ref_key, sha_key in (
                (IMPORT_ZIP, "current_archive_member", "current_archive_sha256"),
                (SUPPLIER_ZIP, "supplier_archive_member", "supplier_archive_sha256"),
            ):
                member_ref = split_member_ref(candidate.get(ref_key))
                if not member_ref:
                    continue
                archive_ref, member = member_ref
                if not archive_ref_matches(archive_ref, archive):
                    raise AuditError(f"型号 {model} 的 {ref_key} 指向非预期ZIP: {candidate.get(ref_key)}")
                expected = str(candidate.get(sha_key) or "").strip().lower()
                archive_refs[archive].setdefault(member, []).append({
                    "model": model,
                    "basis": f"{ref_key}由精确型号关联显式引用",
                    "expected_sha256": expected,
                })
        decisions.append({
            "source_model": model,
            "action": "保留精确在用型号的来源封装候选关联",
            "match": match,
            "retained_candidates": len(candidates),
            "candidate_only": all(bool(x.get("candidate_only")) for x in candidates) if candidates else True,
        })
        kept.append(new_row)
    return {
        "description": "Model-to-footprint source candidates retained only for exact current MPNs or explicitly verified carrier aliases.",
        "interpretation": "Footprints remain candidates only. Their presence does not prove current project selection, pin mapping approval, or mechanical fit.",
        "components": kept,
    }, decisions, archive_refs


def filter_supplement(source: dict[str, Any], active: dict[str, dict[str, Any]], aliases: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, str | None, str]]]:
    filtered = copy.deepcopy(source)
    decisions: list[dict[str, Any]] = []
    deps: list[tuple[str, str | None, str]] = []
    map_keys = (
        "raw_pin_number_aliases",
        "approved_geometry_aliases",
        "exact_model_footprints",
        "manufacturer_model_footprints",
        "shared_model_candidates",
    )
    for map_name in map_keys:
        old_map = source.get(map_name, {})
        new_map: dict[str, Any] = {}
        if isinstance(old_map, dict):
            for model, value in old_map.items():
                match = resolve_model(model, active, aliases)
                if match:
                    new_map[model] = copy.deepcopy(value)
                    decisions.append({"dataset": map_name, "source_model": model, "action": "保留精确在用型号补充记录", "match": match})
                    deps.extend(walk_explicit_pdf_dependencies(value, SUPPLEMENT_JSON))
                else:
                    decisions.append({"dataset": map_name, "source_model": model, "action": "删除未匹配型号补充记录", "match": None})
        filtered[map_name] = new_map
    generic = source.get("reference_land_patterns", [])
    filtered["reference_land_patterns"] = []
    for row in generic:
        decisions.append({
            "dataset": "reference_land_patterns",
            "source_model": None,
            "action": "删除无精确MPN归属的通用参考图样索引",
            "match": None,
            "record_name": row.get("name"),
        })
    filtered["description"] = "Source-backed footprint geometry filtered to exact current MPN identities. Retention is archival evidence only and is not a pin, mechanical, or manufacturing approval."
    return filtered, decisions, deps


def filter_datasheet_index(source: dict[str, Any], active: dict[str, dict[str, Any]], aliases: dict[str, dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, str | None, str]]]:
    models_out = []
    docs_out = []
    no_file_out = []
    decisions: list[dict[str, Any]] = []
    deps: list[tuple[str, str | None, str]] = []
    for row in source.get("models", []):
        model = str(row.get("model") or "").strip()
        match = resolve_model(model, active, aliases)
        if match:
            models_out.append(copy.deepcopy(row))
            decisions.append({"dataset": "models", "source_model": model, "action": "保留精确在用型号手册索引", "match": match})
            file = row.get("file")
            if isinstance(file, str) and file.lower().endswith(".pdf"):
                deps.append((validate_local_ref(DATASHEET_INDEX, file), row.get("sha256"), f"models[{model}].file"))
        else:
            decisions.append({"dataset": "models", "source_model": model, "action": "删除未匹配型号手册索引", "match": None})
    for row in source.get("documents", []):
        source_models = row.get("models", [])
        kept_models = [m for m in source_models if resolve_model(m, active, aliases)]
        if kept_models:
            new_row = copy.deepcopy(row)
            new_row["models"] = kept_models
            docs_out.append(new_row)
            decisions.append({"dataset": "documents", "source_models": source_models, "retained_models": kept_models, "action": "保留有精确在用型号依赖的PDF索引"})
            file = row.get("file")
            if isinstance(file, str) and file.lower().endswith(".pdf"):
                deps.append((validate_local_ref(DATASHEET_INDEX, file), row.get("sha256"), f"documents[{file}].file"))
        else:
            decisions.append({"dataset": "documents", "source_models": source_models, "retained_models": [], "action": "删除无在用型号依赖的PDF索引"})
    for row in source.get("component_sources_without_archived_file", []):
        source_models = row.get("models", [])
        kept_models = [m for m in source_models if resolve_model(m, active, aliases)]
        if kept_models:
            new_row = copy.deepcopy(row)
            new_row["models"] = kept_models
            no_file_out.append(new_row)
            decisions.append({"dataset": "component_sources_without_archived_file", "source_models": source_models, "retained_models": kept_models, "action": "保留精确在用型号的未归档外部来源索引"})
        else:
            decisions.append({"dataset": "component_sources_without_archived_file", "source_models": source_models, "retained_models": [], "action": "删除未匹配型号外部来源索引"})
    filtered = {
        "description": "Model-level datasheet index filtered by exact current MPN identities and verified carrier aliases; only explicit local document dependencies are copied.",
        "models": models_out,
        "documents": docs_out,
        "component_sources_without_archived_file": no_file_out,
    }
    return filtered, decisions, deps


def supplier_member_models(data: bytes) -> list[str]:
    try:
        root = json.loads(data.decode("utf-8-sig"))
    except Exception:  # noqa: BLE001 - a content scan is supplemental to explicit association links
        return []
    result = root.get("result", {}) if isinstance(root, dict) else {}
    if not isinstance(result, dict):
        return []
    found: list[str] = []
    title = result.get("title")
    if isinstance(title, str) and title.strip():
        found.append(title.strip())
    data_str = result.get("dataStr")
    if isinstance(data_str, dict):
        head = data_str.get("head")
        c_para = head.get("c_para") if isinstance(head, dict) else None
        if isinstance(c_para, dict):
            for key in ("Manufacturer Part", "MPN", "manufacturerPart", "partNumber"):
                value = c_para.get(key)
                if isinstance(value, str) and value.strip():
                    found.append(value.strip())
    return found


def filter_zip(
    source_bytes: bytes,
    archive_path: str,
    explicit_refs: dict[str, list[dict[str, str]]],
    active: dict[str, dict[str, Any]],
    aliases: dict[str, dict[str, str]],
    is_supplier_zip: bool,
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    try:
        zin = zipfile.ZipFile(io.BytesIO(source_bytes), "r")
    except Exception as exc:  # noqa: BLE001
        raise AuditError(f"无法打开旧ZIP {archive_path}: {exc}") from exc
    bad_member = zin.testzip()
    if bad_member:
        raise AuditError(f"旧ZIP CRC失败: {archive_path}::{bad_member}")
    infos = zin.infolist()
    member_names = [info.filename for info in infos]
    duplicate_names = sorted({name for name in member_names if member_names.count(name) > 1})
    if duplicate_names:
        raise AuditError(f"旧ZIP有重名成员，无法按路径无歧义过滤: {archive_path}: {duplicate_names[:10]}")
    missing_refs = sorted(set(explicit_refs) - set(member_names))
    if missing_refs:
        raise AuditError(f"型号关联显式引用了ZIP中不存在的成员: {archive_path}: {missing_refs[:10]}")
    reasons: dict[str, list[dict[str, str]]] = {}
    for member, refs in explicit_refs.items():
        reasons.setdefault(member, []).extend(refs)
    content_models: dict[str, list[str]] = {}
    if is_supplier_zip:
        for info in infos:
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            models_here = supplier_member_models(zin.read(info))
            matching = [m for m in models_here if resolve_model(m, active, aliases)]
            if matching:
                content_models[info.filename] = matching
                for model in matching:
                    reasons.setdefault(info.filename, []).append({
                        "model": model,
                        "basis": "ZIP JSON result.title 或明确Manufacturer Part字段精确命中当前MPN/已验证carrier别名",
                        "expected_sha256": "",
                    })
    if archive_path == IMPORT_ZIP and "使用说明.md" in {x.filename for x in infos} and any(not x.is_dir() and x.filename.lower().endswith(".json") for x in infos if x.filename in reasons):
        reasons.setdefault("使用说明.md", []).append({"model": "", "basis": "保留封装导入包的配套使用说明", "expected_sha256": ""})
    out_io = io.BytesIO()
    member_decisions: list[dict[str, Any]] = []
    retained: list[zipfile.ZipInfo] = []
    for info in infos:
        member = info.filename
        data = zin.read(info)
        digest = sha256(data)
        expected = [x["expected_sha256"].lower() for x in reasons.get(member, []) if x.get("expected_sha256")]
        if expected and any(value != digest for value in expected):
            raise AuditError(f"ZIP成员SHA不符: {archive_path}::{member}; actual={digest}, expected={expected}")
        keep = member in reasons
        model_reasons = reasons.get(member, [])
        member_decisions.append({
            "member": member,
            "action": "保留" if keep else "删除",
            "reason": [x["basis"] for x in model_reasons] if keep else ["没有精确型号关联、成员内容型号命中或明确配套依赖"],
            "matched_models": sorted({x["model"] for x in model_reasons if x.get("model")}),
            "source_sha256": digest,
            "source_crc32": f"{info.CRC:08x}",
            "output_sha256": digest if keep else None,
        })
        if keep:
            retained.append(info)
    with zipfile.ZipFile(out_io, "w") as zout:
        zout.comment = zin.comment
        for info in retained:
            # ZipFile.writestr mutates header_offset on its ZipInfo argument;
            # preserve the source record so it remains usable for SHA checks.
            zout.writestr(copy.copy(info), zin.read(info))
    result = out_io.getvalue()
    with zipfile.ZipFile(io.BytesIO(result), "r") as zout:
        bad = zout.testzip()
        if bad:
            raise AuditError(f"输出ZIP CRC失败: {archive_path}::{bad}")
        output_infos = zout.infolist()
        if len(output_infos) != len(retained):
            raise AuditError(f"输出ZIP成员数变化: {archive_path}")
        for source_info, output_info in zip(retained, output_infos):
            if source_info.filename != output_info.filename or sha256(zout.read(output_info)) != sha256(zin.read(source_info)):
                raise AuditError(f"ZIP成员复制后字节SHA变化: {archive_path}::{source_info.filename}")
    summary = {
        "source_zip_sha256": sha256(source_bytes),
        "output_zip_sha256": sha256(result),
        "source_member_count": len(infos),
        "retained_member_count": len(retained),
        "removed_member_count": len(infos) - len(retained),
        "source_crc_test": "PASS",
        "output_crc_test": "PASS",
    }
    return result, member_decisions, summary


def is_protected_output(relpath: str) -> bool:
    return relpath in PROTECTED_FILES or any(relpath.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def safe_publish_path(output: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise AuditError(f"非法输出路径: {relpath}")
    path = output.joinpath(*rel.parts)
    if not path.resolve(strict=False).is_relative_to(output.resolve()):
        raise AuditError(f"输出路径越界: {relpath}")
    return path


def add_dependency(
    deps: dict[str, dict[str, Any]],
    source_path: str,
    expected_sha: str | None,
    reason: str,
) -> None:
    entry = deps.setdefault(source_path, {"expected_sha256": set(), "reasons": []})
    if expected_sha:
        entry["expected_sha256"].add(str(expected_sha).strip().lower())
    entry["reasons"].append(reason)


def check_mode(args: argparse.Namespace, root: Path, repo: Path, output: Path, report_path: Path, index_path: Path, procurement_path: Path) -> int:
    errors: list[str] = []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_revision = report.get("source", {}).get("revision")
    actual_revision = run_git(repo, "rev-parse", "--verify", f"{expected_revision}^{{commit}}").decode().strip()
    if actual_revision != expected_revision:
        errors.append(f"源revision不可复现: expected {expected_revision}, actual {actual_revision}")
    actual_tree = run_git(repo, "rev-parse", "--verify", f"{expected_revision}^{{tree}}").decode().strip()
    expected_tree = report.get("source", {}).get("tree_oid")
    if actual_tree != expected_tree:
        errors.append(f"固定源tree指纹变化: expected {expected_tree}, actual {actual_tree}")
    current_inputs = {"current_index": index_path, "procurement": procurement_path}
    for key, path in current_inputs.items():
        actual = sha256(path.read_bytes()) if path.is_file() else None
        expected = report.get("inputs", {}).get(key, {}).get("sha256")
        if actual != expected:
            errors.append(f"输入变化: {path} expected={expected} actual={actual}")
    for relpath, expected in report.get("generated_outputs", {}).items():
        target = safe_publish_path(output, relpath)
        actual = sha256(target.read_bytes()) if target.is_file() else None
        if actual != expected:
            errors.append(f"输出缺失或SHA变化: {relpath} expected={expected} actual={actual}")
    blobs = tree_blobs(repo, expected_revision)
    expected_file_count = report.get("source", {}).get("tracked_file_count")
    if len(blobs) != expected_file_count:
        errors.append(f"固定源tree文件数变化: expected {expected_file_count}, actual {len(blobs)}")
    for entry in report.get("source_paths", []):
        relpath = entry.get("source_path")
        expected_oid = entry.get("source_blob_oid")
        if relpath not in blobs or blobs.get(relpath) != expected_oid:
            errors.append(f"固定源tree blob变化或缺失: {relpath} expected={expected_oid} actual={blobs.get(relpath)}")
    for relpath, zip_record in report.get("zip_archives", {}).items():
        target = safe_publish_path(output, relpath)
        if not target.is_file():
            continue
        try:
            with zipfile.ZipFile(target) as archive:
                bad = archive.testzip()
                if bad:
                    errors.append(f"输出ZIP CRC失败: {relpath}::{bad}")
                    continue
                indexed = {x["member"]: x for x in zip_record.get("members", []) if x.get("action") == "保留"}
                for info in archive.infolist():
                    data = archive.read(info)
                    rec = indexed.get(info.filename)
                    if rec is None:
                        errors.append(f"输出ZIP含未索引成员: {relpath}::{info.filename}")
                    elif sha256(data) != rec.get("source_sha256") or f"{info.CRC:08x}" != rec.get("source_crc32"):
                        errors.append(f"输出ZIP成员SHA/CRC变化: {relpath}::{info.filename}")
                if set(indexed) != set(archive.namelist()):
                    errors.append(f"输出ZIP成员集合与清理报告不同: {relpath}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"输出ZIP无法读取 {relpath}: {exc}")
    mode = "仅读取固定Git tree/blob，不比较工作区checkout状态"
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print(f"FAIL: --check 检出 {len(errors)} 项问题；源工作树模式={mode}")
        return 1
    print(f"PASS: --check；固定源revision {expected_revision}；输出文件 {len(report.get('generated_outputs', {}))} 个；ZIP CRC/SHA通过；源工作树={mode}")
    return 0


def generate(args: argparse.Namespace, root: Path, repo: Path, output: Path, report_path: Path, index_path: Path, procurement_path: Path) -> int:
    revision = run_git(repo, "rev-parse", "--verify", f"{args.revision}^{{commit}}").decode("ascii").strip()
    blobs = tree_blobs(repo, revision)
    needed = {CATALOG_JSON, CATALOG_CSV, ASSOCIATION_JSON, SUPPLEMENT_JSON, DATASHEET_INDEX, IMPORT_ZIP, SUPPLIER_ZIP, "README.md", ".gitattributes", ".gitignore"}
    missing = sorted(needed - set(blobs))
    if missing:
        raise AuditError(f"固定源tree缺少所需文件: {missing}")
    index_bytes = index_path.read_bytes()
    procurement_bytes = procurement_path.read_bytes()
    index = load_json(index_bytes, str(index_path))
    procurement = load_json(procurement_bytes, str(procurement_path))
    active, procurement_basis = active_model_set(index, procurement)
    aliases = verified_aliases(index, active)

    source_data: dict[str, bytes] = {}
    for relpath in (CATALOG_JSON, CATALOG_CSV, ASSOCIATION_JSON, SUPPLEMENT_JSON, DATASHEET_INDEX, IMPORT_ZIP, SUPPLIER_ZIP, "README.md", ".gitattributes", ".gitignore"):
        source_data[relpath] = git_blob(repo, revision, relpath)
    catalog_src = load_json(source_data[CATALOG_JSON], CATALOG_JSON)
    association_src = load_json(source_data[ASSOCIATION_JSON], ASSOCIATION_JSON)
    supplement_src = load_json(source_data[SUPPLEMENT_JSON], SUPPLEMENT_JSON)
    datasheet_src = load_json(source_data[DATASHEET_INDEX], DATASHEET_INDEX)

    catalog, catalog_decisions = filter_catalog(catalog_src, active, aliases)
    association, association_decisions, archive_refs = filter_associations(association_src, active, aliases)
    supplement, supplement_decisions, supplement_deps = filter_supplement(supplement_src, active, aliases)
    datasheet, datasheet_decisions, datasheet_deps = filter_datasheet_index(datasheet_src, active, aliases)

    manual_deps: dict[str, dict[str, Any]] = {}
    for path, expected, reason in supplement_deps + datasheet_deps:
        add_dependency(manual_deps, path, expected, reason)

    outputs: dict[str, bytes] = {
        CATALOG_JSON: json_bytes(catalog),
        CATALOG_CSV: catalog_csv_bytes(catalog),
        ASSOCIATION_JSON: json_bytes(association),
        SUPPLEMENT_JSON: json_bytes(supplement),
        DATASHEET_INDEX: json_bytes(datasheet),
    }

    zip_reports: dict[str, dict[str, Any]] = {}
    zip_member_decisions: dict[str, list[dict[str, Any]]] = {}
    for archive_path, supplier in ((IMPORT_ZIP, False), (SUPPLIER_ZIP, True)):
        filtered_zip, member_decisions, summary = filter_zip(
            source_data[archive_path], archive_path, archive_refs[archive_path], active, aliases, supplier,
        )
        outputs[archive_path] = filtered_zip
        zip_reports[archive_path] = {**summary, "members": member_decisions}
        zip_member_decisions[archive_path] = member_decisions

    for path, info in manual_deps.items():
        if path not in blobs:
            raise AuditError(f"保留型号依赖的手册不在固定源tree: {path} ({info['reasons']})")
        data = git_blob(repo, revision, path)
        actual_sha = sha256(data)
        expected_hashes = sorted(info["expected_sha256"])
        if expected_hashes and any(value != actual_sha for value in expected_hashes):
            raise AuditError(f"保留型号手册SHA不匹配: {path} actual={actual_sha} expected={expected_hashes}")
        outputs[path] = data

    previous = None
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None
    previous_managed = set(previous.get("generated_outputs", {})) if isinstance(previous, dict) else set()
    for relpath in (".gitattributes", ".gitignore"):
        target = safe_publish_path(output, relpath)
        if not target.exists() or relpath in previous_managed:
            outputs[relpath] = source_data[relpath]

    # Validate collisions before writing. Only files in the previous manifest
    # may be replaced; native assets and other concurrently generated content
    # are preserved and never included in this script's output map.
    for relpath in outputs:
        if is_protected_output(relpath):
            raise AuditError(f"脚本输出触及受保护的当前公共资产: {relpath}")
        target = safe_publish_path(output, relpath)
        if target.exists() and relpath not in previous_managed:
            raise AuditError(f"发布目录已有非本脚本管理的文件，拒绝覆盖: {relpath}")

    source_paths: list[dict[str, Any]] = []
    file_roles = {
        CATALOG_JSON: ("filtered", "目录项和内部matches/alternatives仅保留精确在用MPN；未使用alternatives剔除"),
        CATALOG_CSV: ("regenerated", "根据精确MPN过滤后的目录JSON重建CSV，去掉未使用alternatives"),
        ASSOCIATION_JSON: ("filtered", "只保留精确在用MPN的来源封装候选及其显式ZIP依赖；仍是候选"),
        SUPPLEMENT_JSON: ("filtered", "只保留精确在用型号关联的补充记录；无型号绑定通用参考land pattern删除"),
        DATASHEET_INDEX: ("filtered", "只保留精确在用型号手册记录及显式文件依赖"),
        IMPORT_ZIP: ("filtered_members", "仅保留精确型号关联引用且SHA核验通过的封装成员与使用说明"),
        SUPPLIER_ZIP: ("filtered_members", "仅保留型号关联引用或ZIP JSON明确MPN字段精确命中的成员"),
        ".gitattributes": ("kept_metadata", "保留仓库Git属性配置"),
        ".gitignore": ("kept_metadata", "保留仓库忽略规则"),
    }
    for relpath, oid in sorted(blobs.items()):
        if relpath in outputs:
            payload = outputs[relpath]
            action, reason = file_roles.get(relpath, ("kept_dependency", "显式依赖"))
            source_sha = sha256(source_data[relpath]) if relpath in source_data else None
            source_paths.append(source_decision(relpath, blobs, action, reason, relpath, source_sha))
        elif relpath == "README.md":
            source_paths.append(source_decision(relpath, blobs, "not_copied_owner_writes_current_readme", "不复制旧README；由主代理按筛选后发布树编写当前说明", None, sha256(source_data[relpath])))
        elif relpath in {"库/封装核对图.png", "库/封装焊盘尺寸.csv"}:
            source_bytes = git_blob(repo, revision, relpath)
            source_paths.append(source_decision(relpath, blobs, "removed_static_derivative", "没有可追溯的当前精确型号引用的旧静态图/焊盘尺寸派生物；按要求从发布树移除", None, sha256(source_bytes)))
        elif relpath.startswith("数据手册/") and relpath != DATASHEET_INDEX:
            if relpath in outputs:
                source_paths.append(source_decision(relpath, blobs, "kept_explicit_model_dependency", "; ".join(manual_deps[relpath]["reasons"]), relpath, sha256(outputs[relpath])))
            else:
                source_paths.append(source_decision(relpath, blobs, "removed_unreferenced_manual", "在保留的精确型号索引/封装补充记录中无显式文件依赖", None))
        elif relpath in {"库/封装核对图.png", "库/封装焊盘尺寸.csv"}:
            continue
        else:
            source_paths.append(source_decision(relpath, blobs, "removed_unreferenced_legacy", "不属于精确在用型号目录、封装关系、补充记录或手册依赖；当前公共型号/原生片段由独立导出器维护", None))

    # Model-level decisions are explicit and contain no designators or quantities.
    output_payloads: dict[str, str] = {}
    for relpath, payload in outputs.items():
        target = safe_publish_path(output, relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if sha256(target.read_bytes()) != sha256(payload):
            raise AuditError(f"输出回读SHA校验失败: {relpath}")
        output_payloads[relpath] = sha256(payload)

    # Remove only stale files listed by the previous manifest and owned by this script.
    stale = previous_managed - set(outputs)
    for relpath in sorted(stale):
        if is_protected_output(relpath):
            continue
        target = safe_publish_path(output, relpath)
        if target.is_file():
            target.unlink()

    existing_protected = []
    for relpath in PROTECTED_FILES:
        target = safe_publish_path(output, relpath)
        if target.is_file():
            existing_protected.append({"path": relpath, "sha256": sha256(target.read_bytes()), "reason": "由导出当前公共型号/原生资产的主代理生成；本工具只读保护"})
    for prefix in PROTECTED_PREFIXES:
        folder = safe_publish_path(output, prefix.rstrip("/"))
        if folder.is_dir():
            for target in sorted(folder.rglob("*")):
                if target.is_file():
                    rel = target.relative_to(output).as_posix()
                    existing_protected.append({"path": rel, "sha256": sha256(target.read_bytes()), "reason": "由主代理导出；本工具不修改或覆盖"})

    # Mark source paths that are manual files retained by exact dependency rather
    # than accidental same-name matches.
    for entry in source_paths:
        if entry["source_path"] in manual_deps:
            entry.update({
                "action": "kept_explicit_model_dependency",
                "reason": "; ".join(manual_deps[entry["source_path"]]["reasons"]),
                "output_path": entry["source_path"],
                "source_sha256": output_payloads[entry["source_path"]],
            })

    manual_records = []
    for path, info in sorted(manual_deps.items()):
        manual_records.append({
            "source_path": path,
            "action": "保留并核验SHA256",
            "sha256": output_payloads[path],
            "expected_sha256": sorted(info["expected_sha256"]),
            "model_link_reasons": sorted(set(info["reasons"])),
        })

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "按当前精确MPN筛选历史共享器件库来源；完整记录逐路径/ZIP成员处置。此报告本地保存，不公开板级位号。",
        "source": {
            "repository": str(repo),
            "revision": revision,
            "tree_oid": run_git(repo, "rev-parse", "--verify", f"{revision}^{{tree}}").decode("ascii").strip(),
            "pinned_default_revision": BASE_COMMIT,
            "read_mode": "git show + git ls-tree；不checkout、不修改、不删除、不提交",
            "tracked_file_count": len(blobs),
        },
        "inputs": {
            "current_index": {"path": str(index_path), "sha256": sha256(index_bytes), "part_rows": len(index.get("parts", [])), "unique_mpn_count": len({norm(x.get("mpn")) for x in index.get("parts", []) if x.get("mpn")})},
            "procurement": {"path": str(procurement_path), "sha256": sha256(procurement_bytes), "unique_rows": int(procurement_basis[0]["count"]), "unique_mpn_count": sum(1 for x in active.values() if "采购独有MPN" in x["origins"])},
        },
        "identity_policy": {
            "direct_active_mpn_count": len(active),
            "direct_active_models": [{"model": row["model"], "origins": row["origins"]} for _, row in sorted(active.items())],
            "verified_carrier_aliases": [dict(x) for _, x in sorted(aliases.items())],
            "alias_rule": "只允许原生instances中mpn/carrier_mpn绑定存在、mpn本身在当前总表，且MPN精确为允许厂家前缀 + carrier_mpn完整料号；不对相似/不同封装型号作别名推断。",
            "evidence_limit": "所有保留库材料仅为来源库存档；不表示当前工程已采用，不证明引脚、机械或铜盘通过。",
        },
        "counts": {
            "catalog_source_models": len(catalog_src.get("components", [])),
            "catalog_retained_models": len(catalog.get("components", [])),
            "association_source_models": len(association_src.get("components", [])),
            "association_retained_models": len(association.get("components", [])),
            "supplement_map_entries_retained": sum(len(supplement.get(x, {})) for x in ("raw_pin_number_aliases", "approved_geometry_aliases", "exact_model_footprints", "manufacturer_model_footprints", "shared_model_candidates")),
            "datasheet_source_models": len(datasheet_src.get("models", [])),
            "datasheet_retained_models": len(datasheet.get("models", [])),
            "manual_files_retained": len(manual_deps),
        },
        "source_paths": source_paths,
        "catalog_model_decisions": catalog_decisions,
        "association_model_decisions": association_decisions,
        "supplement_record_decisions": supplement_decisions,
        "datasheet_record_decisions": datasheet_decisions,
        "manual_dependencies": manual_records,
        "zip_archives": zip_reports,
        "generated_outputs": output_payloads,
        "preserved_existing_current_exports": existing_protected,
        "stale_managed_outputs_removed": sorted(stale),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(json_bytes(report))
    print(f"PASS: 固定源tree={revision}")
    print(f"PASS: 活跃MPN={len(active)}，carrier精确别名={len(aliases)}；采购独有行={procurement_basis[0]['count']}")
    print(f"PASS: 目录 {len(catalog_src.get('components', []))}->{len(catalog['components'])}；关联 {len(association_src.get('components', []))}->{len(association['components'])}；手册 {len(datasheet_src.get('models', []))}->{len(datasheet['models'])}，依赖PDF={len(manual_deps)}")
    for path, item in zip_reports.items():
        print(f"PASS: {path} 成员 {item['source_member_count']}->{item['retained_member_count']}，源/输出CRC PASS，保留payload SHA不变")
    print(f"PASS: 写入 {len(output_payloads)} 个受管理文件；保留主代理现有当前公共索引/原生片段 {len(existing_protected)} 个文件")
    print(f"报告: {report_path}")
    return 0


def main() -> int:
    script = Path(__file__).resolve()
    root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root, help="电力项目根目录")
    parser.add_argument("--repo", type=Path, default=None, help="旧共享仓库克隆目录；默认使用tmp partial clone")
    parser.add_argument("--output", type=Path, default=None, help="独立发布目录")
    parser.add_argument("--revision", default=BASE_COMMIT, help="固定旧Git commit；默认使用本工具记录的基线")
    parser.add_argument("--current-index", type=Path, default=None, help="当前元件总表JSON")
    parser.add_argument("--procurement", type=Path, default=None, help="采购补充JSON")
    parser.add_argument("--report", type=Path, default=None, help="本地清理报告")
    parser.add_argument("--check", action="store_true", help="只读核验输入、固定源tree、输出SHA及ZIP CRC/member SHA")
    args = parser.parse_args()
    root = args.root.resolve()
    canonical_repo = root / "元件封装库" / "共享仓库"
    default_repo = canonical_repo if canonical_repo.is_dir() else root / DEFAULT_REPO
    repo = (args.repo or default_repo).resolve()
    output = (args.output or (root / DEFAULT_OUTPUT)).resolve()
    index_path = (args.current_index or (root / DEFAULT_INDEX)).resolve()
    procurement_path = (args.procurement or (root / DEFAULT_PROCUREMENT)).resolve()
    report_path = (args.report or (root / DEFAULT_REPORT)).resolve()
    for path in (repo, index_path, procurement_path):
        if not path.exists():
            print(f"FAIL: 输入路径不存在: {path}", file=sys.stderr)
            return 2
    try:
        if args.check:
            return check_mode(args, root, repo, output, report_path, index_path, procurement_path)
        output.mkdir(parents=True, exist_ok=True)
        return generate(args, root, repo, output, report_path, index_path, procurement_path)
    except (AuditError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
