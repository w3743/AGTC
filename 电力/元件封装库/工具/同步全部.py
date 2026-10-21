#!/usr/bin/env python3
"""统一更新或只读核验中央库，不改各板工程。"""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
folder = Path(__file__).resolve().parent
for name in [
    "更新元件库.py",
    "收集现成库.py",
    "更新采购补充.py",
    "更新项目主档.py",
    "商城成本.py",
]:
    result = subprocess.run([sys.executable, "-B", str(folder / name)] + (["--check"] if args.check else []))
    if result.returncode:
        raise SystemExit(result.returncode)
