#!/usr/bin/env python3
"""Preview or apply an exact, allowlisted BMS DRC-closeout documentation patch.

The patch plan is prepared only after final readback facts and hashes are known.
Default behavior is a diff preview. Writing requires --apply.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ALLOWLIST = {
    "BMS模块/README.md",
    "BMS模块/项目要求.md",
    "BMS模块/TODO.md",
    "BMS模块/检查/当前设计状态.md",
    "BMS模块/检查/DRC修复结论.md",
    "BMS模块/检查/局部连线/结论.md",
    "BMS模块/检查/PSU支路闭合.md",
    "BMS模块/原理图准备/Layout交接.md",
    "BMS模块/仿真/结果/README.md",
    "元件封装库/README.md",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("patch plan must be a version 1 JSON object")
    patches = plan.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("patch plan must contain a non-empty patches array")
    return plan


def checked_path(relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if normalized not in ALLOWLIST:
        raise ValueError(f"path is outside the documentation allowlist: {relative}")
    path = (ROOT / Path(*normalized.split("/"))).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError(f"path escapes project root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_input_hashes(plan: dict[str, Any]) -> None:
    inputs = plan.get("input_hashes", [])
    if not isinstance(inputs, list):
        raise ValueError("input_hashes must be an array")
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("each input_hashes item needs path and sha256")
        relative = item["path"].replace("\\", "/")
        candidate = (ROOT / Path(*relative.split("/"))).resolve()
        if ROOT.resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError(f"invalid evidence path: {relative}")
        expected = str(item.get("sha256", "")).lower()
        actual = sha256(candidate.read_bytes())
        if len(expected) != 64 or actual != expected:
            raise ValueError(f"evidence SHA256 mismatch: {relative}")


def render(plan: dict[str, Any]) -> dict[Path, tuple[bytes, bytes]]:
    verify_input_hashes(plan)
    original_by_path: dict[Path, bytes] = {}
    text_by_path: dict[Path, str] = {}
    newline_by_path: dict[Path, str] = {}
    bom_by_path: dict[Path, bool] = {}
    expected_hash_by_path: dict[Path, str] = {}

    for patch in plan["patches"]:
        if not isinstance(patch, dict):
            raise ValueError("each patch must be an object")
        relative = patch.get("file")
        before = patch.get("before")
        after = patch.get("after")
        expected_hash = patch.get("expected_file_sha256")
        if not all(isinstance(value, str) for value in (relative, before, after)):
            raise ValueError("each patch needs string file, before, and after fields")
        if not before:
            raise ValueError("empty before text is forbidden")
        path = checked_path(relative)
        if path not in original_by_path:
            raw = path.read_bytes()
            actual_hash = sha256(raw)
            if expected_hash is None or str(expected_hash).lower() != actual_hash:
                raise ValueError(f"document SHA256 changed since patch preparation: {relative}")
            bom = raw.startswith(b"\xef\xbb\xbf")
            decoded = raw.decode("utf-8-sig")
            newline = "\r\n" if "\r\n" in decoded else "\n"
            normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
            original_by_path[path] = raw
            text_by_path[path] = normalized
            newline_by_path[path] = newline
            bom_by_path[path] = bom
            expected_hash_by_path[path] = actual_hash
        elif str(expected_hash).lower() != expected_hash_by_path[path]:
            raise ValueError(f"inconsistent expected document SHA256 for {relative}")

        before_n = before.replace("\r\n", "\n").replace("\r", "\n")
        after_n = after.replace("\r\n", "\n").replace("\r", "\n")
        text = text_by_path[path]
        occurrences = text.count(before_n)
        if occurrences != 1:
            raise ValueError(
                f"{relative}: exact before block must occur once, found {occurrences}"
            )
        text_by_path[path] = text.replace(before_n, after_n, 1)

    rendered: dict[Path, tuple[bytes, bytes]] = {}
    for path, normalized in text_by_path.items():
        restored = normalized.replace("\n", newline_by_path[path])
        encoded = restored.encode("utf-8")
        if bom_by_path[path]:
            encoded = b"\xef\xbb\xbf" + encoded
        rendered[path] = (original_by_path[path], encoded)
    return rendered


def atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="JSON patch plan with final evidence hashes")
    parser.add_argument("--apply", action="store_true", help="write changes; default only previews")
    args = parser.parse_args()

    plan = read_plan(args.plan)
    rendered = render(plan)
    changes = [(path, before, after) for path, (before, after) in rendered.items() if before != after]
    if not changes:
        print("No document changes.")
        return 0

    if not args.apply:
        print("Preview only. Review this diff; pass --apply to write.")
        for path, before, after in changes:
            old = before.decode("utf-8-sig").splitlines(keepends=True)
            new = after.decode("utf-8-sig").splitlines(keepends=True)
            relative = path.relative_to(ROOT).as_posix()
            print("".join(difflib.unified_diff(old, new, relative, relative)), end="")
        return 0

    for path, _, after in changes:
        atomic_write(path, after)
    print("Updated:")
    for path, _, _ in changes:
        print(f"- {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
