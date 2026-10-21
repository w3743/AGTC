#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用当前真实索引和BOM回归采购补充CSV的空MPN处理。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = Path(__file__).resolve().with_name("采购补充空MPN回归.json")
PURCHASE_PATH = ROOT / "元件封装库" / "工具" / "更新采购补充.py"


def load_module(name: str, path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    purchase = load_module("update_purchase_supplement", PURCHASE_PATH)

    # collect只读当前索引与来源BOM，不调用同步入口或写正式输出。
    result = purchase.collect()
    null_mpn_records = [record for record in result["records"] if None in record["native_mpns"]]
    assert null_mpn_records, "当前来源未覆盖任何native_mpns=[None]真实记录"
    only_null_records = [record for record in null_mpn_records if record["native_mpns"] == [None]]
    assert only_null_records, "未找到native_mpns恰为[None]的真实关联记录"
    assert all(None in record["native_mpns"] for record in null_mpn_records)

    fields = [
        "module", "references", "mpn_source", "manufacturer_source", "quantity_source",
        "unit_source", "scope_source", "relation", "native_mpns", "status_source",
        "notes_source", "source", "source_row",
    ]
    with tempfile.TemporaryDirectory(prefix="native-empty-mpn-") as temporary_dir:
        output = Path(temporary_dir) / "采购补充.csv"
        purchase.write_csv_atomic(output, fields, result["records"])
        csv_bytes = output.read_bytes()
        assert csv_bytes.startswith(b"\xef\xbb\xbf"), "CSV没有UTF-8 BOM"
        with output.open("r", encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == len(result["records"])
        null_rows = [row for row in csv_rows if row["native_mpns"] == ""]
        assert len(null_rows) >= len(only_null_records), "native_mpns=[None]未序列化为空CSV单元格"
        assert b"None" not in csv_bytes, "CSV中出现了Python None字面量"

        # 模拟替换失败：旧的当前表必须保持完整，临时文件应清理。
        previous_bytes = output.read_bytes()
        try:
            with patch.object(purchase.os, "replace", side_effect=OSError("模拟替换失败")):
                purchase.write_csv_atomic(output, fields, result["records"])
        except OSError:
            pass
        else:
            raise AssertionError("模拟os.replace失败未抛出异常")
        assert output.read_bytes() == previous_bytes, "替换失败破坏了既有CSV"
        leftovers = [path.name for path in Path(temporary_dir).iterdir() if path != output]
        assert not leftovers, f"写失败留下临时文件：{leftovers}"

    # JSON主数据中保持真正的null，不把它转换成字符串"None"或空文字。
    round_trip = json.loads(json.dumps(null_mpn_records, ensure_ascii=False))
    assert any(record["native_mpns"] == [None] for record in round_trip)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "只读collect真实采购/BOM记录；CSV写入限于临时目录；不调用正式同步入口",
        "purchase_generator": str(PURCHASE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "bom_rows_checked": len(result["records"]),
        "native_mpns_contains_null_record_count": len(null_mpn_records),
        "native_mpns_exactly_null_record_count": len(only_null_records),
        "native_mpns_exactly_null_examples": [
            {
                "module": record["module"],
                "references": record["references"],
                "source": record["source"],
                "source_row": record["source_row"],
                "native_mpns": record["native_mpns"],
            }
            for record in only_null_records[:20]
        ],
        "checks": [
            "真实native_mpns=[None]记录可被collect读取",
            "JSON对象往返仍为null",
            "CSV中null列表输出为空单元格且不出现None字面量",
            "CSV采用同目录临时文件后替换",
            "模拟替换异常时原CSV保持完整且临时文件清理",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
