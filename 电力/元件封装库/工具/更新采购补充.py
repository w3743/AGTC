#!/usr/bin/env python3
"""把现行采购 BOM 与原生器件按模块和位号关联，保留板外件及差异。"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
ROOT = LIB.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def save(name, value):
    path = LIB / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_cell(value):
    """CSV空值写为空单元格；JSON主数据仍保留None/null。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join("" if item is None else str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv_atomic(path, fields, records):
    """先在目标目录写完整临时CSV，再原子替换当前表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({key: csv_cell(record.get(key)) for key in fields})
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def collect():
    config = read(LIB / "来源配置.json")
    native_file = LIB / "索引/元件总表.json"
    native = read(native_file)
    actual = {}
    for inst in native["instances"]:
        key = (inst["module"], inst["designator"])
        actual.setdefault(key, []).append(inst)
    sources = [{"path": "元件封装库/来源配置.json", "sha256": sha(LIB / "来源配置.json")},
               {"path": "元件封装库/索引/元件总表.json", "sha256": sha(native_file)}]
    records = []
    for spec in config["bom_sources"]:
        path = ROOT / spec["path"]
        before = sha(path)
        text = path.read_text(encoding="utf-8-sig")
        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
        rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
        for number, row in enumerate(rows, 2):
            fmt = spec["format"]
            if fmt == "bms":
                refs, mpn, maker = [row["位号"]], row["MPN"], row.get("Manufacturer", "")
                qty, unit = row["每套用量"], "pcs"
                notes, scope = row["备注"], row["类别"]
                status = row["采购状态"] + "；" + row["料号状态"]
            elif fmt == "regulator":
                refs = [r.strip() for r in re.split(r"[,，]", row["Reference"]) if r.strip()]
                mpn, maker = row["Manufacturer part"], row["Manufacturer"]
                qty, unit = row["Quantity"], row["Unit"]
                notes, scope, status = row["Requirements and notes"], row["Scope"], row["Procurement status"]
                if not refs:
                    raise ValueError(f"{spec['path']} 第{number}行没有有效Reference")
                try:
                    row_quantity = Decimal(str(qty).strip())
                except InvalidOperation as exc:
                    raise ValueError(f"{spec['path']} 第{number}行Quantity无效: {qty!r}") from exc
                if row_quantity != len(refs):
                    raise ValueError(
                        f"{spec['path']} 第{number}行Reference数量({len(refs)}) "
                        f"与Quantity({qty})不一致；请先核对BOM计量口径"
                    )
                # 稳压/水泵采购BOM以每个器件位号为一件。展开分组行后逐器件关联，
                # 避免将整组数量重复挂到每个原生器件上。
                row_records = [([ref], "1") for ref in refs]
            elif fmt == "foc":
                refs = [r.strip() for r in re.split(r"[,，]", row["Designator"]) if r.strip()]
                mpn, maker = row["Manufacturer Part"], row["Manufacturer"]
                qty, unit = row["Quantity"], "pcs"
                notes, scope, status = row["Comment"], "交付BOM", "交付BOM原记录"
            else:
                raise ValueError(f"未知BOM格式: {fmt}")
            if fmt != "regulator":
                row_records = [(refs, qty)]
            for row_refs, row_qty in row_records:
                linked = [i for ref in row_refs for i in actual.get((spec["module"], ref), [])]
                missing = [ref for ref in row_refs if (spec["module"], ref) not in actual]
                # 原生MPN允许为null；避免None与字符串混排时排序报错，并保留JSON null。
                actual_mpns = sorted(
                    {i.get("mpn") for i in linked},
                    key=lambda value: (value is not None, "" if value is None else str(value).casefold(),
                                       "" if value is None else str(value)),
                )
                # BMS旧式字段常有厂家前缀；仅在确切原生制造商/MPN组合匹配时判一致。
                matched = bool(linked) and all(
                    norm(mpn) in {norm(i["mpn"]), norm((i["manufacturer"] or "") + " " + (i["mpn"] or ""))}
                    for i in linked)
                if not linked:
                    relation = "BOM独有条目，未在原生BOM中登记"
                elif missing:
                    relation = "部分位号匹配，需核对"
                elif matched:
                    relation = "原生已有，采购行关联，不重复计件"
                else:
                    relation = "原生已有，采购字段差异待核对，不重复计件"
                records.append({
                    "module": spec["module"], "references": row_refs,
                    "mpn_source": mpn, "manufacturer_source": maker,
                    "quantity_source": row_qty, "unit_source": unit, "scope_source": scope,
                    "status_source": status, "notes_source": notes,
                    "relation": relation, "native_keys": sorted({i["key"] for i in linked}),
                    "native_mpns": actual_mpns, "unmatched_references": missing,
                    "source": spec["path"], "source_row": number, "raw": row,
                })
        if sha(path) != before:
            raise RuntimeError(f"读取时源已改变: {path}")
        sources.append({"path": spec["path"], "sha256": before, "rows": len(rows)})
    summary = {"bom_rows": len(records), "relations": dict(Counter(r["relation"] for r in records))}
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "quantity_basis": config["quantity_basis"], "sources": sources,
            "summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = LIB / "索引/采购补充.json"
    if args.check:
        current = read(output)
        errors = [s["path"] for s in current["sources"]
                  if not (ROOT / s["path"]).is_file() or sha(ROOT / s["path"]) != s["sha256"]]
        print(json.dumps({"pass": not errors, "changed_or_missing_sources": errors}, ensure_ascii=False))
        return 1 if errors else 0
    result = collect()
    save("索引/采购补充.json", result)
    path = LIB / "索引/采购补充.csv"
    fields = ["module", "references", "mpn_source", "manufacturer_source", "quantity_source",
              "unit_source", "scope_source", "relation", "native_mpns", "status_source",
              "notes_source", "source", "source_row"]
    write_csv_atomic(path, fields, result["records"])
    lines = ["# 全项目元件目录", "", "现行原生器件与采购补充统一从此查找。原生已有的采购行仅作关联，不重复计件。", "",
             "- [原生元件总表](元件总表.csv) · [完整原生属性与库绑定](元件总表.json)",
             "- [板级用量](板级用量.csv) · [现有跨板复用与待核对项](复用与待核对.md)",
             "- [全部采购BOM关联](采购补充.csv) · [完整采购来源记录](采购补充.json)",
             "- [嘉立创EDA现成库包](../检查/包索引.json)", "",
             "## 原生BOM之外的采购条目", "",
             "以下是BOM独有记录。保留原始型号、数量及条件，不能据此认定均为已选定的板外器件。", "",
             "| 模块 | 位号 | BOM原始型号 | 数量 | 单位 |", "|---|---|---|---:|---|"]
    for r in result["records"]:
        if not r["native_keys"]:
            values = [r["module"], ", ".join(r["references"]), r["mpn_source"] or "待选型",
                      r["quantity_source"], r["unit_source"]]
            lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |")
    (LIB / "索引/全项目元件目录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
