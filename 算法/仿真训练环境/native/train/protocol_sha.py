"""Compute/verify/embed the frozen native protocol SHA.

Canonical form: the line `> protocol_sha256: ...` is normalised to `> protocol_sha256:`
before hashing, so embedding the value does not change the hash. Any other edit does.

By default the newest protocol (V1.1 if present, else V1) is used; pass `--protocol` to
target a specific file. The SHA companion file lives next to the protocol:
`PROTOCOL_NATIVE_V1.1.sha256` / `PROTOCOL_NATIVE_V1.sha256` (old V1 record is kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE / "PROTOCOL_NATIVE_V1.1.md", HERE / "PROTOCOL_NATIVE_V1.md"]
LINE_RE = re.compile(r"(?m)^> protocol_sha256:.*$")


def current_protocol() -> Path:
    for path in CANDIDATES:
        if path.exists():
            return path
    raise SystemExit("no protocol file found")


def sha_file_for(path: Path) -> Path:
    return path.with_suffix(".sha256")


def canonical_text(text: str) -> str:
    if not LINE_RE.search(text):
        raise SystemExit(f"{text[:40]!r}: missing the '> protocol_sha256:' line")
    return LINE_RE.sub("> protocol_sha256:", text)


def compute(path: Path) -> str:
    return hashlib.sha256(canonical_text(path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()


def embedded(path: Path) -> str | None:
    match = re.search(r"(?m)^> protocol_sha256:\s*([0-9a-f]{64})\s*$", path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def embed(sha: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(LINE_RE.sub(f"> protocol_sha256: {sha}", text, count=1), encoding="utf-8")


def verify(path: Path) -> bool:
    return embedded(path) == compute(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the embedded SHA instead of writing it")
    parser.add_argument("--protocol", type=str, default="", help="protocol markdown path (default: newest)")
    args = parser.parse_args()
    path = Path(args.protocol) if args.protocol else current_protocol()
    if not path.is_absolute():
        path = HERE / path
    sha = compute(path)
    if args.check:
        ok = verify(path)
        print(json.dumps({"protocol": path.name, "protocol_sha256": sha,
                          "embedded": embedded(path), "match": ok}))
        raise SystemExit(0 if ok else 1)
    embed(sha, path)
    sha_file = sha_file_for(path)
    sha_file.write_text(
        json.dumps({
            "file": path.name,
            "protocol_sha256": sha,
            "algorithm": "sha256 over UTF-8 bytes with the '> protocol_sha256:' line normalised to empty",
            "frozen": "2026-10-25",
            "supersedes": "PROTOCOL_NATIVE_V1.md" if "V1.1" in path.name else None,
            "note": "any change to the protocol body changes this hash; create a new version instead of editing",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol": path.name, "protocol_sha256": sha,
                      "embedded": embedded(path), "sha_file": sha_file.name}))


if __name__ == "__main__":
    main()
