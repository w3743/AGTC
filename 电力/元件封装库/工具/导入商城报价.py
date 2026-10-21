#!/usr/bin/env python3
"""Validate compact browser quote batches and merge them into 商城报价.json.

Batch rows:
  [Ccode, mpn, manufacturer, productId, captured_at_utc, moq, stock,
   [[min_quantity, unit_price], ...], identity_verified, optional_raw_evidence]

Identity is checked against the current canonical product mapping before price
data is admitted. Existing unresolved entries are always retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


LIBRARY_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = LIBRARY_ROOT / "检查" / "商城核价" / "已核价批次"
MAPPING_PATH = LIBRARY_ROOT / "索引" / "商城商品映射.json"
MASTER_PATH = LIBRARY_ROOT / "商城报价.json"
CODE_RE = re.compile(r"^C\d{3,}$")
PRODUCT_ID_RE = re.compile(r"^\d+$")
MAX_REASONABLE_UNIT_PRICE_CNY = Decimal("1000000")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def normalize_identity_text(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.split()).casefold()
    return text or None


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("captured_at_utc must be an ISO-8601 timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("captured_at_utc has no timezone")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def quote_id_for(code: str, captured_at: datetime) -> str:
    return f"LCSC_CN-{code}-{captured_at:%Y%m%d}"


def as_optional_positive_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer or null")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a non-negative integer or null") from exc
    if str(value).strip() not in (str(integer), f"{integer}.0") and not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    if integer < 0 or (field == "moq" and integer == 0):
        raise ValueError(f"{field} must be {'positive' if field == 'moq' else 'non-negative'}")
    return integer


def parse_price_ladder(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("price tiers are missing; price not imported")
    tiers: list[dict[str, Any]] = []
    previous_min: int | None = None
    previous_price: Decimal | None = None
    for index, row in enumerate(value, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"price tier {index} must be [min_quantity, unit_price]")
        min_quantity = as_optional_positive_integer(row[0], f"tier {index} min_quantity")
        if min_quantity is None or min_quantity == 0:
            raise ValueError(f"tier {index} min_quantity must be positive")
        try:
            unit_price = Decimal(str(row[1]))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"tier {index} unit_price is not numeric") from exc
        if not unit_price.is_finite() or unit_price <= 0:
            raise ValueError(f"tier {index} unit_price must be finite and positive")
        if unit_price > MAX_REASONABLE_UNIT_PRICE_CNY:
            raise ValueError(
                f"tier {index} unit_price exceeds sanity limit {MAX_REASONABLE_UNIT_PRICE_CNY} CNY"
            )
        if previous_min is not None and min_quantity <= previous_min:
            raise ValueError("price tier minimum quantities must be strictly increasing")
        if previous_price is not None and unit_price > previous_price:
            raise ValueError("price tiers increase with quantity; review capture")
        tiers.append(
            {
                "min_quantity": min_quantity,
                "unit_price": float(unit_price),
            }
        )
        previous_min = min_quantity
        previous_price = unit_price
    return tiers


def parse_batch_file(path: Path) -> list[Any]:
    content = read_json(path)
    if isinstance(content, list):
        return content
    if isinstance(content, dict) and isinstance(content.get("rows"), list):
        return content["rows"]
    raise ValueError("batch JSON must be an array of compact rows or an object with a rows array")


def index_canonical_pages(mapping: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for page in mapping.get("canonical_product_pages", []):
        product_id = str(page.get("product_id") or "").strip()
        if product_id:
            by_id.setdefault(product_id, []).append(page)
    return by_id


def canonical_identity(row: list[Any], pages_by_id: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    code, mpn, manufacturer, product_id = (str(row[index]).strip() if row[index] is not None else "" for index in range(4))
    if not CODE_RE.fullmatch(code):
        raise ValueError("C-code format is invalid")
    if not mpn:
        raise ValueError("MPN is empty")
    if not manufacturer:
        raise ValueError("manufacturer is empty")
    if not PRODUCT_ID_RE.fullmatch(product_id):
        raise ValueError("productId must be numeric")

    pages = pages_by_id.get(product_id, [])
    exact = [
        page
        for page in pages
        if page.get("product_model") == mpn
        and page.get("product_code") == code
        and normalize_identity_text(page.get("manufacturer_name"))
        == normalize_identity_text(manufacturer)
    ]
    if len(exact) != 1:
        same_id = pages[0] if pages else None
        if not pages:
            raise ValueError("productId is absent from canonical mapping")
        mismatches = []
        if same_id.get("product_model") != mpn:
            mismatches.append(f"MPN mismatch: page={same_id.get('product_model')!r}")
        if same_id.get("product_code") != code:
            mismatches.append(f"C-code mismatch: page={same_id.get('product_code')!r}")
        if normalize_identity_text(same_id.get("manufacturer_name")) != normalize_identity_text(manufacturer):
            mismatches.append(f"manufacturer mismatch: page={same_id.get('manufacturer_name')!r}")
        if len(pages) > 1:
            mismatches.append("productId is ambiguous in canonical mapping")
        raise ValueError("; ".join(mismatches) or "canonical identity is ambiguous")
    return exact[0]


def unresolved_record(
    batch_path: Path,
    row_index: int,
    row: Any,
    reason: str,
    quote_id: str | None,
    imported_at: str,
) -> dict[str, Any]:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    stable_id = hashlib.sha256(
        (batch_path.name + "\0" + str(row_index) + "\0" + encoded).encode("utf-8")
    ).hexdigest()
    if isinstance(row, list):
        code = row[0] if len(row) > 0 else None
        mpn = row[1] if len(row) > 1 else None
        manufacturer = row[2] if len(row) > 2 else None
    else:
        code = mpn = manufacturer = None
    return {
        "unresolved_id": stable_id,
        "source_batch": batch_path.name,
        "source_row": row_index,
        "quote_id": quote_id,
        "supplier_part": code,
        "mpn": mpn,
        "manufacturer": manufacturer,
        "reason": reason,
        "received_row": row,
        "recorded_at_utc": imported_at,
    }


def build_observation(row: Any, batch_path: Path, row_index: int, pages_by_id: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not isinstance(row, list) or len(row) not in (9, 10):
        raise ValueError("row must contain 9 fields plus optional raw_evidence")

    code, mpn, manufacturer, product_id, captured_text = row[:5]
    code = str(code).strip() if code is not None else ""
    parsed_time = parse_utc(captured_text)
    quote_id = quote_id_for(code, parsed_time) if CODE_RE.fullmatch(code) else None

    if row[8] is not True:
        raise ValueError("identity_verified is not boolean true")
    page = canonical_identity(row, pages_by_id)
    tiers = parse_price_ladder(row[7])
    moq = as_optional_positive_integer(row[5], "moq")
    stock = as_optional_positive_integer(row[6], "stock")
    if moq is not None and tiers[0]["min_quantity"] < moq:
        raise ValueError("lowest price tier starts below the reported MOQ")

    provided_evidence = row[9] if len(row) == 10 else None
    if provided_evidence is None:
        tier_text = "; ".join(
            f"{tier['min_quantity']}+ CNY {tier['unit_price']}"
            for tier in tiers
        )
        evidence = (
            "compact_browser_transcription; "
            f"tiers: {tier_text}; MOQ: {moq if moq is not None else 'null'}; "
            f"stock: {stock if stock is not None else 'null'}"
        )
        evidence_kind = "compact_browser_transcription"
    elif isinstance(provided_evidence, str):
        evidence = provided_evidence
        evidence_kind = "provided_raw_evidence"
    else:
        evidence = json.dumps(provided_evidence, ensure_ascii=False, sort_keys=True)
        evidence_kind = "provided_raw_evidence_structured"

    availability = "unknown" if stock is None else ("zero_snapshot" if stock == 0 else "in_stock_snapshot")
    return {
        "quote_id": quote_id,
        "mpn": str(mpn).strip(),
        "manufacturer": str(manufacturer).strip(),
        "supplier": "LCSC_CN",
        "supplier_part": code,
        "source_url": page["canonical_item_url"],
        "captured_at_utc": utc_text(parsed_time),
        "acquisition": "compact_batch_import",
        "evidence_kind": evidence_kind,
        "currency": "CNY",
        "identity_verified": True,
        "identity_evidence": (
            "Compact-row identity_verified=true and exact productId, MPN, C-code, "
            "and manufacturer match with the current canonical product mapping."
        ),
        "price_tiers": tiers,
        "moq": moq,
        "order_multiple": None,
        "stock_quantity": stock,
        "stock_unit": None,
        "availability": availability,
        "tax_status": "not_supplied_in_compact_batch",
        "shipping_included": None,
        "raw_evidence": evidence,
    }


def observation_time(item: dict[str, Any]) -> datetime:
    try:
        return parse_utc(item.get("captured_at_utc"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def latest_by_quote_id(observations: list[Any]) -> list[Any]:
    chosen: dict[str, tuple[int, Any]] = {}
    no_id: list[Any] = []
    for position, item in enumerate(observations):
        if not isinstance(item, dict) or not item.get("quote_id"):
            no_id.append(item)
            continue
        key = str(item["quote_id"])
        current = chosen.get(key)
        timestamp = observation_time(item)
        if current is None or timestamp >= observation_time(current[1]):
            chosen[key] = (position, item)
    deduped = no_id + [pair[1] for pair in chosen.values()]
    return sorted(
        deduped,
        key=lambda item: (
            str(item.get("quote_id", "")) if isinstance(item, dict) else "",
            observation_time(item) if isinstance(item, dict) else datetime.min.replace(tzinfo=timezone.utc),
        ),
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.stem + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="校验并导入立创商城紧凑核价批次。")
    parser.add_argument("--dry-run", action="store_true", help="验证并预览变化，不写商城报价.json。")
    args = parser.parse_args()

    if not MAPPING_PATH.is_file():
        parser.error(f"找不到商品身份映射：{MAPPING_PATH}")
    if not MASTER_PATH.is_file():
        parser.error(f"找不到主商城报价文件：{MASTER_PATH}")
    if not BATCH_DIR.is_dir():
        print(f"没有批次目录，未修改主报价文件：{BATCH_DIR}")
        return 0

    mapping = read_json(MAPPING_PATH)
    pages_by_id = index_canonical_pages(mapping)
    master = read_json(MASTER_PATH)
    if not isinstance(master, dict):
        parser.error("商城报价.json 顶层必须为对象")
    existing_observations = master.get("observations")
    existing_unresolved = master.get("unresolved")
    if not isinstance(existing_observations, list) or not isinstance(existing_unresolved, list):
        parser.error("商城报价.json 的 observations/unresolved 必须为数组")

    batches = sorted(BATCH_DIR.glob("*.json"))
    imported: list[dict[str, Any]] = []
    new_unresolved: list[dict[str, Any]] = []
    read_rows = 0
    now_text = utc_text(datetime.now(timezone.utc))
    for batch_path in batches:
        try:
            rows = parse_batch_file(batch_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            new_unresolved.append(
                unresolved_record(batch_path, 0, None, f"batch parse failure: {exc}", None, now_text)
            )
            continue
        for row_index, row in enumerate(rows, start=1):
            read_rows += 1
            quote_id: str | None = None
            try:
                if isinstance(row, list) and len(row) >= 5:
                    code = str(row[0]).strip() if row[0] is not None else ""
                    quote_id = quote_id_for(code, parse_utc(row[4])) if CODE_RE.fullmatch(code) else None
                imported.append(build_observation(row, batch_path, row_index, pages_by_id))
            except (ValueError, TypeError, IndexError, OverflowError) as exc:
                new_unresolved.append(
                    unresolved_record(batch_path, row_index, row, str(exc), quote_id, now_text)
                )

    combined_observations = latest_by_quote_id(existing_observations + imported)
    unresolved_by_id = {
        str(item.get("unresolved_id")): item
        for item in existing_unresolved
        if isinstance(item, dict) and item.get("unresolved_id")
    }
    preserved_unresolved = list(existing_unresolved)
    for item in new_unresolved:
        unresolved_id = item["unresolved_id"]
        if unresolved_id not in unresolved_by_id:
            preserved_unresolved.append(item)
            unresolved_by_id[unresolved_id] = item

    updated = dict(master)
    updated["observations"] = combined_observations
    updated["unresolved"] = preserved_unresolved

    summary = {
        "batch_files": len(batches),
        "rows_read": read_rows,
        "valid_observations": len(imported),
        "new_unresolved": len(new_unresolved),
        "observations_before": len(existing_observations),
        "observations_after": len(combined_observations),
        "unresolved_before": len(existing_unresolved),
        "unresolved_after": len(preserved_unresolved),
        "dry_run": args.dry_run,
        "unresolved_preview": [
            {
                "batch": item.get("source_batch"),
                "row": item.get("source_row"),
                "mpn": item.get("mpn"),
                "reason": item.get("reason"),
            }
            for item in new_unresolved
        ],
    }
    if not args.dry_run:
        write_json_atomic(MASTER_PATH, updated)
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
