"""核对每板两份现行设计文档；旧批量README/TODO生成行为已按用户新规则撤销。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("BMS模块", "FOC驱动模块", "稳压模块", "水泵驱动模块")
ALLOWED = {"设计需求.md", "设计细节.md"}

def evidence_attachment(path):
    rel = path.relative_to(ROOT).as_posix()
    return "/source_snapshot/" in rel or "/仿真/tools/ngspice-47-portable/" in rel

def main():
    result = {}
    for name in MODULES:
        folder = ROOT / name
        docs = [p for p in folder.rglob("*.md") if not evidence_attachment(p)]
        expected = {folder / x for x in ALLOWED}
        result[name] = {
            "currentDocuments": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in docs),
            "missing": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in expected if not p.is_file()),
            "extra": sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in docs if p not in expected),
            "frozenOrThirdPartyAttachments": sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                for p in folder.rglob("*.md") if evidence_attachment(p)),
        }
    passed = all(not x["missing"] and not x["extra"] for x in result.values())
    print(json.dumps({"status": "PASS" if passed else "FAIL", "boards": result}, ensure_ascii=False, indent=2))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
