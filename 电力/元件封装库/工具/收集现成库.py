#!/usr/bin/env python3
"""Collect explicitly listed EasyEDA library archives without editing their contents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCES = [
    "BMS模块/检查/当前精确器件库.elibz",
    "BMS模块/检查/低价候选精确器件库.elibz",
    "BMS模块/检查/C915627官方器件库.elibz",
    "BMS模块/检查/AO3400A精确器件库补充.elibz",
    "BMS模块/检查/CAN复用47uF官方库.elibz",
    "BMS模块/检查/PSU100nF当前精确器件库补充.elibz",
    "BMS模块/检查/泄放继电器官方库.elibz",
    "稳压模块/参考资料/超级电容.elibz2",
]

ARCHIVE_DISCLAIMER = "来源库存档；不表示现行工程已采用，也不表示引脚与机械尺寸通过。"
SUPERCAP_NOTE = (
    "本文件是原始参考库；工程现行铜盘版本为5 mm，原库未修正（4 mm→5 mm），"
    "不得用此参考包覆盖当前工程。"
)
CSV_FIELDS = [
    "source_path",
    "source_status",
    "source_sha256",
    "source_size_bytes",
    "central_path",
    "zip_crc_check",
    "zip_member_count",
    "device_json_path",
    "device_uuid",
    "manufacturer",
    "manufacturer_mpn",
    "supplier_part",
    "symbol_ref",
    "footprint_ref",
    "archive_note",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def make_central_name(source: Path, digest: str, central_dir: Path) -> Path:
    candidate = central_dir / source.name
    if not candidate.exists():
        return candidate
    if sha256_file(candidate) == digest:
        return candidate

    suffix = source.suffix
    stem = source.name[: -len(suffix)] if suffix else source.name
    for length in range(8, len(digest) + 1, 4):
        candidate = central_dir / f"{stem}-{digest[:length]}{suffix}"
        if not candidate.exists() or sha256_file(candidate) == digest:
            return candidate
    raise RuntimeError(f"无法为同名不同内容的包生成唯一文件名：{source.name}")


def archive_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "zip_crc_check": "not_applicable",
        "zip_crc_error": None,
        "zip_members": [],
        "device_json_path": None,
        "devices": [],
        "archive_note": ARCHIVE_DISCLAIMER,
    }
    if not path.is_file():
        return result
    if path.suffix.lower() not in {".elibz", ".elibz2"}:
        return result

    try:
        with zipfile.ZipFile(path, "r") as archive:
            result["zip_members"] = [
                {
                    "path": info.filename,
                    "size_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                }
                for info in archive.infolist()
            ]
            bad_member = archive.testzip()
            if bad_member is None:
                result["zip_crc_check"] = "pass"
            else:
                result["zip_crc_check"] = "fail"
                result["zip_crc_error"] = f"CRC不匹配：{bad_member}"

            json_names = {info.filename for info in archive.infolist()}
            device_name = next(
                (name for name in ("device.json", "device2.json") if name in json_names),
                None,
            )
            if device_name is not None:
                result["device_json_path"] = device_name
                document = json.loads(archive.read(device_name).decode("utf-8-sig"))
                device_map = document.get("devices", {}) if isinstance(document, dict) else {}
                if isinstance(device_map, dict):
                    for map_uuid, device in device_map.items():
                        if not isinstance(device, dict):
                            continue
                        attributes = device.get("attributes", {})
                        if not isinstance(attributes, dict):
                            attributes = {}
                        result["devices"].append(
                            {
                                "uuid": device.get("uuid") or map_uuid,
                                "title": device.get("display_title") or device.get("title"),
                                "manufacturer": attributes.get("Manufacturer"),
                                "manufacturer_mpn": attributes.get("Manufacturer Part"),
                                "supplier_part": attributes.get("Supplier Part"),
                                "symbol_ref": attributes.get("Symbol"),
                                "footprint_ref": attributes.get("Footprint"),
                            }
                        )
    except (OSError, zipfile.BadZipFile, RuntimeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        result["zip_crc_check"] = "fail"
        result["zip_crc_error"] = f"{type(exc).__name__}: {exc}"

    if path.name == "超级电容.elibz2":
        result["archive_note"] = f"{ARCHIVE_DISCLAIMER}{SUPERCAP_NOTE}"
    return result


def source_record(source_rel: str, source: Path, root: Path, central_dir: Path) -> dict[str, Any]:
    if not source.is_file():
        record = {
            "source_path": source_rel,
            "source_status": "missing",
            "source_sha256": None,
            "source_size_bytes": None,
            "central_path": None,
            "archive_note": ARCHIVE_DISCLAIMER,
        }
        record.update(archive_metadata(source))
        return record

    digest = sha256_file(source)
    same_bytes = None
    if central_dir.is_dir():
        same_bytes = next(
            (candidate for candidate in central_dir.iterdir() if candidate.is_file() and sha256_file(candidate) == digest),
            None,
        )
    central = same_bytes or make_central_name(source, digest, central_dir)
    if central.exists():
        if sha256_file(central) != digest:
            raise RuntimeError(f"集中副本路径已存在且内容不同：{relative(central, root)}")
        status = "deduplicated" if same_bytes is not None or central.name != source.name else "already_present"
    else:
        central_dir.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".collect-", dir=central_dir, delete=False) as tmp:
                temp_name = tmp.name
            temp_path = Path(temp_name)
            shutil.copyfile(source, temp_path)
            if sha256_file(source) != digest or sha256_file(temp_path) != digest:
                raise RuntimeError(f"复制期间源文件发生变化或副本哈希不符：{source_rel}")
            os.replace(temp_path, central)
        finally:
            if temp_name and Path(temp_name).exists():
                Path(temp_name).unlink()
        if sha256_file(central) != digest:
            raise RuntimeError(f"写入后副本哈希不符：{relative(central, root)}")
        status = "copied"

    record = {
        "source_path": source_rel,
        "source_status": status,
        "source_sha256": digest,
        "source_size_bytes": source.stat().st_size,
        "central_path": relative(central, root),
    }
    record.update(archive_metadata(source))
    return record


def write_indexes(records: list[dict[str, Any]], index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "按显式清单集中保存来源库原包；仅作来源库存档，不代表工程采用或电气/机械验收。",
        "archive_disclaimer": ARCHIVE_DISCLAIMER,
        "supercapacitor_reference_note": SUPERCAP_NOTE,
        "sources": records,
    }
    (index_dir / "包索引.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        devices = record.get("devices") or [None]
        for device in devices:
            device = device or {}
            rows.append(
                {
                    "source_path": record.get("source_path"),
                    "source_status": record.get("source_status"),
                    "source_sha256": record.get("source_sha256"),
                    "source_size_bytes": record.get("source_size_bytes"),
                    "central_path": record.get("central_path"),
                    "zip_crc_check": record.get("zip_crc_check"),
                    "zip_member_count": len(record.get("zip_members", [])),
                    "device_json_path": record.get("device_json_path"),
                    "device_uuid": device.get("uuid"),
                    "manufacturer": device.get("manufacturer"),
                    "manufacturer_mpn": device.get("manufacturer_mpn"),
                    "supplier_part": device.get("supplier_part"),
                    "symbol_ref": device.get("symbol_ref"),
                    "footprint_ref": device.get("footprint_ref"),
                    "archive_note": record.get("archive_note", ARCHIVE_DISCLAIMER),
                }
            )
    with (index_dir / "包索引.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_collect(root: Path, central_dir: Path, index_dir: Path) -> int:
    records: list[dict[str, Any]] = []
    failures = 0
    for source_rel in SOURCES:
        source = root / Path(source_rel)
        try:
            record = source_record(source_rel, source, root, central_dir)
        except Exception as exc:  # Keep every requested source represented in the index.
            failures += 1
            record = {
                "source_path": source_rel,
                "source_status": "error",
                "source_sha256": sha256_file(source) if source.is_file() else None,
                "source_size_bytes": source.stat().st_size if source.is_file() else None,
                "central_path": None,
                "zip_crc_check": "not_checked",
                "zip_crc_error": f"{type(exc).__name__}: {exc}",
                "zip_members": [],
                "device_json_path": None,
                "devices": [],
                "archive_note": ARCHIVE_DISCLAIMER,
            }
        records.append(record)
        print(f"{record['source_status']}: {source_rel} -> {record.get('central_path') or '-'}")
        if record.get("zip_crc_check") == "fail":
            failures += 1

    write_indexes(records, index_dir)
    print(f"索引已写入：{relative(index_dir / '包索引.json', root)}")
    return 1 if failures else 0


def run_check(root: Path, central_dir: Path, index_dir: Path) -> int:
    index_path = index_dir / "包索引.json"
    if not index_path.is_file():
        print(f"FAIL: 缺少索引 {relative(index_path, root)}", file=sys.stderr)
        return 1
    try:
        manifest = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: 无法读取索引：{exc}", file=sys.stderr)
        return 1

    records = {item.get("source_path"): item for item in manifest.get("sources", [])}
    failures = 0
    for source_rel in SOURCES:
        record = records.get(source_rel)
        if record is None:
            print(f"FAIL: 索引缺少来源记录：{source_rel}")
            failures += 1
            continue

        source = root / Path(source_rel)
        recorded_sha = record.get("source_sha256")
        if recorded_sha is None:
            if source.exists():
                print(f"FAIL: 原记录缺失但当前来源已出现，需重新收集：{source_rel}")
                failures += 1
            else:
                print(f"PASS (记录为缺失): {source_rel}")
            continue

        if not source.is_file():
            print(f"FAIL: 来源文件缺失：{source_rel}")
            failures += 1
            continue
        source_sha = sha256_file(source)
        if source_sha != recorded_sha:
            print(f"FAIL: 来源SHA256变化：{source_rel} expected={recorded_sha} actual={source_sha}")
            failures += 1
            continue

        central_rel = record.get("central_path")
        if not central_rel:
            print(f"FAIL: 索引没有中央副本路径：{source_rel}")
            failures += 1
            continue
        central = root / Path(central_rel)
        if not central.is_file():
            print(f"FAIL: 中央副本缺失：{central_rel}")
            failures += 1
            continue
        central_sha = sha256_file(central)
        if central_sha != recorded_sha:
            print(f"FAIL: 中央副本SHA256不符：{central_rel} expected={recorded_sha} actual={central_sha}")
            failures += 1
            continue

        source_crc = archive_metadata(source)
        central_crc = archive_metadata(central)
        if source_crc.get("zip_crc_check") == "fail" or central_crc.get("zip_crc_check") == "fail":
            print(
                f"FAIL: ZIP CRC检查失败：{source_rel}; "
                f"source={source_crc.get('zip_crc_error')}; copy={central_crc.get('zip_crc_error')}"
            )
            failures += 1
            continue
        print(f"PASS: SHA256与ZIP CRC：{source_rel} -> {central_rel}")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只读检查索引记录的源/副本SHA256与ZIP CRC，不写入文件",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    central_dir = root / "元件封装库" / "嘉立创EDA库"
    index_dir = root / "元件封装库" / "检查"
    if args.check:
        return run_check(root, central_dir, index_dir)
    return run_collect(root, central_dir, index_dir)


if __name__ == "__main__":
    raise SystemExit(main())
