#!/usr/bin/env python3
"""AGTC repository self-check.

Deterministic, offline checks used for periodic project hygiene review. It mirrors
the checks that the PR workflow enforces and adds project-specific consistency
checks (stale machine paths, broken Markdown links, syntax validity).

Usage (repository root):

    python .github/scripts/repository-self-check.py
    python .github/scripts/repository-self-check.py --range origin/main..HEAD
    python .github/scripts/repository-self-check.py --skip-native --verbose

Exit code 0 when all hard checks pass; 1 when a hard check fails. Warnings do not
change the exit code. The script never writes to the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
MAX_ORDINARY_BLOB = 10 * 1024 * 1024
WHITESPACE = "blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol"
NATIVE_TOOLS = "算法/仿真训练环境/native/tools"

STALE_PATH_PATTERNS = (
    r"C:\\pioneer\\",  # other machine's spelling
    r"C:/pioneer/",
    r"C:\\pinoeer\\rnn-research-",
    r"C:/pinoeer/rnn-research-",
    r"C:\\pinoeer\\agtc-brake-review-",
    r"C:/pinoeer/agtc-brake-review-",
)

TEXT_SUFFIXES = {
    ".md", ".py", ".ps1", ".js", ".cjs", ".mjs", ".json", ".jsonl", ".csv",
    ".txt", ".yml", ".yaml", ".toml", ".sh", ".ini", ".cfg", ".html", ".svg",
}

AUDIT_DIRS = ("归档/", "算法/rnn研究/runs/")


class Result:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))

    @property
    def failures(self) -> int:
        return sum(1 for status, _, _ in self.lines if status == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for status, _, _ in self.lines if status == "WARN")


def git(*args: str, check: bool = True) -> bytes:
    """Run git and return stdout bytes."""
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout


def git_text(*args: str, check: bool = True) -> str:
    return git(*args, check=check).decode("utf-8", "replace")


def resolve_range(explicit: str | None) -> str:
    if explicit:
        return explicit
    if git("rev-parse", "--verify", "--quiet", "origin/main", check=False).strip():
        return "origin/main..HEAD"
    if git("rev-parse", "--verify", "--quiet", "HEAD~1", check=False).strip():
        return "HEAD~1..HEAD"
    return "HEAD"


def check_whitespace(result: Result, diff_range: str) -> None:
    proc = subprocess.run(
        ["git", "-c", f"core.whitespace={WHITESPACE}", "diff", "--check", diff_range],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    output = proc.stdout.decode("utf-8", "replace").strip()
    if proc.returncode == 0:
        result.add("PASS", "空白检查", diff_range)
    else:
        offenders = sorted({
            line.split(":", 1)[0]
            for line in output.splitlines()
            if re.search(r":\d+: ", line)
        })
        result.add("FAIL", "空白检查", f"{len(offenders)} 个文件，例如 {offenders[:3]}")


def lfs_attribute(head: str, path: str) -> bool:
    output = git_text("check-attr", f"--source={head}", "filter", "--", path)
    return output.rstrip("\n").endswith(": lfs")


def is_canonical_lfs_pointer(content: bytes) -> bool:
    proc = subprocess.run(
        ["git", "lfs", "pointer", "--check", "--strict", "--stdin"],
        input=content,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        check=False,
    )
    return proc.returncode == 0


def check_blob_policy(result: Result, diff_range: str) -> None:
    head = git_text("rev-parse", "HEAD").strip()
    raw = git("diff", "--raw", "--no-abbrev", "--no-renames", "-z",
              "--diff-filter=ACMT", diff_range)
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) % 2:
        result.add("FAIL", "Git/LFS 大文件策略", "无法解析 NUL 分隔的 diff 记录")
        return
    checked = 0
    problems: list[str] = []
    for index in range(0, len(records), 2):
        metadata = records[index].decode("ascii", "replace")
        path = records[index + 1].decode("utf-8", "replace")
        fields = metadata.lstrip(":").split()
        if len(fields) != 5:
            problems.append(f"无法解析 diff 记录: {metadata}")
            continue
        new_mode, new_oid = fields[1], fields[3]
        if new_mode in ("000000", "160000"):
            continue
        size = int(git_text("cat-file", "-s", new_oid).strip())
        checked += 1
        if lfs_attribute(head, path):
            if size >= 1024:
                problems.append(f"LFS 文件不是指针（{size} B）: {path}")
                continue
            if not is_canonical_lfs_pointer(git("cat-file", "blob", new_oid)):
                problems.append(f"LFS 指针格式不规范: {path}")
        elif size > MAX_ORDINARY_BLOB:
            problems.append(f"普通 Git blob 超过 10 MiB（{size} B）: {path}")
    if problems:
        result.add("FAIL", "Git/LFS 大文件策略",
                   f"{len(problems)} 项，例如 {problems[:3]}")
    else:
        result.add("PASS", "Git/LFS 大文件策略", f"{checked} 个变更 blob")


def iter_tracked_text_files() -> Iterable[tuple[str, Path]]:
    for record in git("ls-files", "-z").split(b"\0"):
        if not record:
            continue
        path = record.decode("utf-8", "replace")
        if Path(path).suffix.lower() not in TEXT_SUFFIXES:
            continue
        full = ROOT / path
        if not full.is_file():
            continue
        try:
            if full.stat().st_size > 2 * 1024 * 1024:
                continue
        except OSError:
            continue
        yield path, full


def check_stale_paths(result: Result) -> None:
    patterns = [re.compile(p) for p in STALE_PATH_PATTERNS]
    hits: list[str] = []
    for path, full in iter_tracked_text_files():
        if path.startswith(AUDIT_DIRS):
            continue
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                hits.append(f"{path}:{line_number}")
    if hits:
        result.add("WARN", "失效机器路径", f"{len(hits)} 处，例如 {hits[:3]}")
    else:
        result.add("PASS", "失效机器路径", "无")


LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def check_markdown_links(result: Result) -> None:
    broken: list[str] = []
    scanned = 0
    for path, full in iter_tracked_text_files():
        if not path.endswith(".md") or path.startswith("归档/"):
            continue
        scanned += 1
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        base = full.parent
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            if target.startswith("<"):
                target = target.lstrip("<").rstrip(">")
            fragment_free = target.split("#", 1)[0]
            if not fragment_free:
                continue
            resolved = (base / fragment_free).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue  # outside repository: reported by stale path check if relevant
            if not resolved.exists():
                broken.append(f"{path} -> {target}")
    if broken:
        result.add("WARN", "Markdown 相对链接", f"{len(broken)} 处，例如 {broken[:3]}")
    else:
        result.add("PASS", "Markdown 相对链接", f"{scanned} 个文件")


def iter_changed_files(diff_range: str, statuses: str) -> list[str]:
    output = git_text("diff", "--name-only", "--no-renames", f"--diff-filter={statuses}",
                      "-z", diff_range)
    return [item for item in output.split("\0") if item]


def check_syntax(result: Result, diff_range: str) -> None:
    checked = 0
    problems: list[str] = []
    for path in iter_changed_files(diff_range, "AM"):
        suffix = Path(path).suffix.lower()
        full = ROOT / path
        if not full.is_file():
            continue
        if suffix == ".py":
            try:
                source = full.read_text(encoding="utf-8")
                compile(source, path, "exec")
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                problems.append(f"{path}: {exc}")
            checked += 1
        elif suffix == ".json":
            try:
                json.loads(full.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{path}: {exc}")
            checked += 1
    if problems:
        result.add("FAIL", "语法检查", f"{len(problems)} 项，例如 {problems[:3]}")
    else:
        result.add("PASS", "语法检查", f"{checked} 个 Python/JSON 文件")


def check_native_tests(result: Result, python: str) -> None:
    target = ROOT / NATIVE_TOOLS
    if not target.is_dir():
        result.add("WARN", "原生训练 CPU 回归", f"目录不存在: {NATIVE_TOOLS}")
        return
    proc = subprocess.run(
        [python, "-B", "-m", "unittest", "discover", "-s", str(target),
         "-p", "test_*.py"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = proc.stdout.decode("utf-8", "replace")
    match = re.search(r"^Ran (\d+) tests? in", output, re.MULTILINE)
    ran = match.group(1) if match else "?"
    if proc.returncode == 0:
        result.add("PASS", "原生训练 CPU 回归", f"{ran} 项")
    else:
        tail = next(
            (line for line in reversed(output.splitlines())
             if line.startswith(("FAILED", "ERROR"))),
            f"exit {proc.returncode}",
        )
        result.add("FAIL", "原生训练 CPU 回归", f"{ran} 项: {tail}")


def check_worktree(result: Result) -> None:
    status = git_text("status", "--porcelain", "-uno").strip()
    if status:
        result.add("WARN", "工作区状态", f"{len(status.splitlines())} 个已跟踪文件有未提交改动")
    else:
        result.add("PASS", "工作区状态", "已跟踪文件无未提交改动")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="diff_range",
                        help="Git 比较范围，默认 origin/main..HEAD")
    parser.add_argument("--python", default=sys.executable,
                        help="运行原生回归测试的 Python（默认当前解释器）")
    parser.add_argument("--skip-native", action="store_true", help="跳过原生训练 CPU 回归")
    parser.add_argument("--skip-links", action="store_true", help="跳过 Markdown 链接检查")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if subprocess.run(["git", "--version"], cwd=ROOT, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, check=False).returncode != 0:
        print("错误：找不到 git，请先把 git 加入 PATH 或使用包装脚本。")
        return 2

    diff_range = resolve_range(args.diff_range)
    result = Result()
    print(f"AGTC 仓库自检 · 范围 {diff_range} · 根目录 {ROOT}\n")

    check_worktree(result)
    check_whitespace(result, diff_range)
    check_blob_policy(result, diff_range)
    check_syntax(result, diff_range)
    check_stale_paths(result)
    if not args.skip_links:
        check_markdown_links(result)
    if not args.skip_native:
        check_native_tests(result, args.python)

    for status, name, detail in result.lines:
        if status == "PASS" and not args.verbose and detail in ("无", ""):
            print(f"[PASS] {name}")
        else:
            print(f"[{status}] {name} · {detail}")
    print(f"\n合计：{len(result.lines)} 项检查，{result.failures} 项失败，"
          f"{result.warnings} 项警告")
    return 1 if result.failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except RuntimeError as exc:
        print(f"错误：{exc}")
        sys.exit(1)
