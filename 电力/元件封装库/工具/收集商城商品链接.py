#!/usr/bin/env python3
"""Collect and identity-check literal SZLCSC item links from the current catalog.

Only public HTML GET requests are made. The script extracts product identity
from Next.js server-rendered JSON and never reads price or stock fields.
"""

from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit("缺少 requests；请在项目Python环境安装后重跑。") from exc


LIBRARY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = LIBRARY_ROOT / "索引" / "元件总表.json"
QUOTE_ROOT = LIBRARY_ROOT / "检查" / "商城核价"
OUTPUT_PATH = LIBRARY_ROOT / "索引" / "商城商品映射.json"
QUOTE_FILES = {
    "BMS模块": QUOTE_ROOT / "BMS.json",
    "FOC驱动模块": QUOTE_ROOT / "FOC.json",
    "稳压模块": QUOTE_ROOT / "稳压.json",
    "水泵驱动模块": QUOTE_ROOT / "水泵.json",
}

ITEM_URL_RE = re.compile(r"https?://item\.szlcsc\.com/[^\s<>\"']+", re.IGNORECASE)
LCSC_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(C\d{3,})(?!\d)")
NEXT_DATA_RE = re.compile(
    r"<script\b[^>]*\bid\s*=\s*([\"'])__NEXT_DATA__\1[^>]*>(.*?)</script\s*>",
    re.IGNORECASE | re.DOTALL,
)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}，。；：！？）】》"
CAPTCHA_MARKERS = ("验证码", "安全验证", "人机验证", "访问受限", "请完成验证")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def exact_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def strings_in(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings_in(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from strings_in(child, path + (str(index),))


def find_item_urls(value: Any) -> list[tuple[str, tuple[str, ...]]]:
    found: list[tuple[str, tuple[str, ...]]] = []
    for path, text in strings_in(value):
        decoded = html.unescape(text)
        for match in ITEM_URL_RE.findall(decoded):
            candidate = match.rstrip(TRAILING_URL_PUNCTUATION)
            parsed = urlparse(candidate)
            if parsed.scheme.lower() == "https" and parsed.netloc.lower() == "item.szlcsc.com":
                found.append((candidate, path))
    return found


def lcsc_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for _, text in strings_in(value):
        codes.extend(LCSC_CODE_RE.findall(text))
    return list(dict.fromkeys(codes))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def quote_item_fields(module: str, item: dict[str, Any]) -> dict[str, Any]:
    if module == "BMS模块":
        mpn = exact_text(item.get("mpn"))
        makers = [
            exact_text(entry.get("name"))
            for entry in as_list(item.get("manufacturer_candidates"))
            if isinstance(entry, dict)
        ]
        codes = lcsc_codes(item.get("lcsc_codes"))
        designators = as_list(item.get("designators"))
        quantities = {"qty_per_bms_set": item.get("qty_per_bms_set")}
        url = item.get("official_szlcsc_item_url")
    elif module == "FOC驱动模块":
        mpn = exact_text(item.get("manufacturerPart"))
        makers = [exact_text(item.get("manufacturer"))]
        makers.extend(exact_text(v) for v in as_list(item.get("manufacturerNamesAsListed")))
        codes = lcsc_codes([item.get("supplierPart"), item.get("supplierPartCandidates")])
        designators = as_list(item.get("designators"))
        quantities = {
            key: item.get(key)
            for key in (
                "purchaseQuantitySingleBoardFOCA",
                "purchaseQuantitySingleBoardFOCB",
                "purchaseQuantityTwoBoards",
            )
            if item.get(key) is not None
        }
        if item.get("quantities") is not None:
            quantities["quantities"] = item.get("quantities")
        url = item.get("officialSZLCSCItemUrl")
    else:
        mpn = exact_text(item.get("manufacturer_part"))
        makers = [exact_text(item.get("manufacturer"))]
        codes = lcsc_codes(item.get("lcsc_part"))
        designators = as_list(item.get("references"))
        quantities = {"quantity_per_module": item.get("quantity_per_module")}
        url = item.get("official_szlcsc_item_url")

    makers = list(dict.fromkeys(v for v in makers if v))
    designators = list(dict.fromkeys(str(v).strip() for v in designators if v))
    return {
        "module": module,
        "manufacturer_part": mpn,
        "manufacturer_names": makers,
        "supplier_codes": codes,
        "designators": designators,
        "quantity_context": quantities,
        "source_url": exact_text(url),
        "source_kind": "module_price_input",
        "procurement_status": item.get("procurement_status")
        or item.get("item_identity_status")
        or item.get("priceSnapshotStatus")
        or item.get("price_status"),
    }


def quote_records(quote_data: dict[str, Any], module: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in quote_data.get("items", []):
        if not isinstance(item, dict):
            continue
        record = quote_item_fields(module, item)
        if record["manufacturer_part"]:
            records.append(record)

    supplemental_key = "external_items" if module == "BMS模块" else "externalSupplementalItems"
    for item in quote_data.get(supplemental_key, []):
        if not isinstance(item, dict):
            continue
        if module == "BMS模块":
            record = quote_item_fields(module, item)
            record["source_kind"] = "module_price_input_external"
            if record["manufacturer_part"]:
                records.append(record)
        elif module == "FOC驱动模块":
            mpn = exact_text(item.get("manufacturerPart"))
            if not mpn:
                continue
            maker = exact_text(item.get("manufacturer"))
            records.append(
                {
                    "module": module,
                    "manufacturer_part": mpn,
                    "manufacturer_names": [maker] if maker else [],
                    "supplier_codes": lcsc_codes(item.get("supplierPart")),
                    "designators": as_list(item.get("designators")),
                    "quantity_context": {
                        key: item.get(key)
                        for key in ("quantity", "purchaseQuantityTwoBoards")
                        if item.get(key) is not None
                    },
                    "source_url": exact_text(item.get("officialSZLCSCItemUrl")),
                    "source_kind": "module_price_input_external",
                    "procurement_status": item.get("itemUrlStatus"),
                }
            )
    return records


def manufacturer_status(expected_names: list[str], page_name: str | None) -> str:
    if not expected_names or not page_name:
        return "unknown"
    normalized = {name.casefold().strip() for name in expected_names if name}
    if page_name.casefold().strip() in normalized:
        return "exact_text_match"
    return "name_difference_review"


def parse_public_product_page(body: str) -> dict[str, Any]:
    match = NEXT_DATA_RE.search(body)
    if not match:
        raise ValueError("missing __NEXT_DATA__")
    data = json.loads(match.group(2))
    record = (
        data.get("props", {})
        .get("pageProps", {})
        .get("webData", {})
        .get("productRecord")
    )
    if not isinstance(record, dict):
        raise ValueError("missing props.pageProps.webData.productRecord")
    return {
        "product_model": exact_text(record.get("productModel")),
        "product_code": exact_text(record.get("productCode")),
        "product_id": exact_text(record.get("productId")),
        "manufacturer_name": exact_text(record.get("productGradePlateName")),
    }


def fetch_one(url: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_url": url,
        "http_status": None,
        "final_url": None,
        "fetch_status": "failed",
        "product": None,
        "failure_reason": None,
    }
    try:
        response = requests.get(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "Mozilla/5.0 (compatible; PowerPartsLinkAudit/1.0)",
            },
            timeout=(8, 20),
            allow_redirects=True,
        )
        result["http_status"] = response.status_code
        result["final_url"] = response.url
        if urlparse(response.url).netloc.lower() != "item.szlcsc.com":
            result["fetch_status"] = "blocked_redirect"
            result["failure_reason"] = "redirect left official item domain"
            return result
        body = response.text
        if response.status_code in (203, 403, 429):
            result["fetch_status"] = "blocked_or_verification"
            result["failure_reason"] = f"HTTP {response.status_code}; no retry or bypass attempted"
            return result
        if any(marker in body for marker in CAPTCHA_MARKERS):
            result["fetch_status"] = "blocked_or_verification"
            result["failure_reason"] = "verification marker found; no retry or bypass attempted"
            return result
        if response.status_code != 200:
            result["fetch_status"] = "http_error"
            result["failure_reason"] = f"HTTP {response.status_code}"
            return result
        product = parse_public_product_page(body)
        result["product"] = product
        if not product["product_model"] or not product["product_id"]:
            result["fetch_status"] = "identity_incomplete"
            result["failure_reason"] = "productModel or productId missing from public productRecord"
        else:
            result["fetch_status"] = "identity_read"
    except requests.RequestException as exc:
        result["failure_reason"] = f"request error: {type(exc).__name__}: {exc}"
    except (ValueError, json.JSONDecodeError) as exc:
        result["fetch_status"] = "identity_parse_failed"
        result["failure_reason"] = str(exc)
    return result


def catalog_link_contexts(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = {}
    for instance in catalog.get("instances", []):
        attrs = instance.get("attrs") or {}
        actual_attrs = attrs.get("instance") or {}
        device_attrs = attrs.get("device") or {}
        actual_mpn = exact_text(actual_attrs.get("Manufacturer Part")) or exact_text(instance.get("mpn"))
        actual_maker = exact_text(actual_attrs.get("Manufacturer")) or exact_text(instance.get("manufacturer"))
        actual_code = exact_text(actual_attrs.get("Supplier Part")) or exact_text(instance.get("supplier_part"))
        carrier_mpn = exact_text(device_attrs.get("Manufacturer Part")) or exact_text(instance.get("carrier_mpn"))
        for source_scope, values in (("attrs.instance", actual_attrs), ("attrs.device", device_attrs)):
            for url, value_path in find_item_urls(values):
                contexts.setdefault(url, []).append(
                    {
                        "source_kind": "catalog_instance_literal",
                        "module": instance.get("module"),
                        "designator": instance.get("ref"),
                        "bom": instance.get("bom"),
                        "source_scope": source_scope,
                        "attribute_path": list(value_path),
                        "actual_manufacturer_part": actual_mpn,
                        "actual_manufacturer": actual_maker,
                        "supplier_code": actual_code,
                        "carrier_manufacturer_part": carrier_mpn,
                        "quantity_context": {"catalog_bom_instance": 1 if instance.get("bom") == "yes" else 0},
                    }
                )
    return contexts


def dedupe_dicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="收集并核对立创商城商品页公开身份字段。")
    parser.add_argument(
        "--reuse-current-observations",
        action="store_true",
        help="仅当中央目录、四个核价输入和直链集合与现有映射完全一致时复用已抓取HTTP响应，不重新请求。",
    )
    args = parser.parse_args()

    if not CATALOG_PATH.is_file():
        print(f"找不到中央元件总表：{CATALOG_PATH}", file=sys.stderr)
        return 2
    missing_quotes = [str(path) for path in QUOTE_FILES.values() if not path.is_file()]
    if missing_quotes:
        print("缺少核价输入：" + ", ".join(missing_quotes), file=sys.stderr)
        return 2

    catalog = read_json(CATALOG_PATH)
    quote_data = {module: read_json(path) for module, path in QUOTE_FILES.items()}
    url_contexts = catalog_link_contexts(catalog)
    purchases: list[dict[str, Any]] = []
    for module, data in quote_data.items():
        for record in quote_records(data, module):
            purchases.append(record)
            url = record.get("source_url")
            if url:
                parsed = urlparse(url)
                if parsed.scheme.lower() == "https" and parsed.netloc.lower() == "item.szlcsc.com":
                    url_contexts.setdefault(url, []).append(record)

    urls = sorted(url_contexts)
    current_source_bindings = {
        "catalog_path": str(CATALOG_PATH.relative_to(LIBRARY_ROOT)),
        "catalog_sha256": sha256_file(CATALOG_PATH),
        "quote_inputs": {
            module: {
                "path": str(path.relative_to(LIBRARY_ROOT)),
                "sha256": sha256_file(path),
            }
            for module, path in QUOTE_FILES.items()
        },
    }
    fetched: dict[str, dict[str, Any]] = {}
    if args.reuse_current_observations:
        if not OUTPUT_PATH.is_file():
            print("没有现有映射可复用；请在受控环境中正常运行一次联网收集。", file=sys.stderr)
            return 2
        previous = read_json(OUTPUT_PATH)
        if previous.get("source_bindings") != current_source_bindings:
            print("源文件SHA已变化；拒绝复用旧HTTP观察，请重新进行正常公开GET。", file=sys.stderr)
            return 2
        previous_observations = previous.get("http_observations", [])
        fetched = {
            item["source_url"]: item
            for item in previous_observations
            if isinstance(item, dict) and item.get("source_url")
        }
        if set(fetched) != set(urls):
            print("候选直链集合已变化；拒绝复用不完整HTTP观察。", file=sys.stderr)
            return 2
    else:
        # At most two workers; each unique source URL is requested once, without retries.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_to_url = {pool.submit(fetch_one, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                fetched[url] = future.result()

    pages: dict[str, dict[str, Any]] = {}
    for url, result in fetched.items():
        product = result.get("product")
        if not product or not product.get("product_id"):
            continue
        canonical = f"https://item.szlcsc.com/{product['product_id']}.html"
        page = pages.setdefault(
            canonical,
            {
                "canonical_item_url": canonical,
                "product_id": product["product_id"],
                "product_model": product["product_model"],
                "product_code": product["product_code"],
                "manufacturer_name": product["manufacturer_name"],
                "source_urls": [],
                "source_link_contexts": [],
                "exact_procurement_matches": [],
                "rejected_or_flagged_matches": [],
            },
        )
        if url not in page["source_urls"]:
            page["source_urls"].append(url)
        page["source_link_contexts"].extend(url_contexts[url])

        for context in url_contexts[url]:
            is_quote = context.get("source_kind", "").startswith("module_price_input")
            if is_quote:
                expected_mpn = context.get("manufacturer_part")
                expected_makers = context.get("manufacturer_names") or []
                expected_codes = context.get("supplier_codes") or []
            else:
                expected_mpn = context.get("actual_manufacturer_part")
                expected_makers = [context.get("actual_manufacturer")] if context.get("actual_manufacturer") else []
                expected_codes = lcsc_codes(context.get("supplier_code"))

            if expected_mpn == product["product_model"]:
                maker_status = manufacturer_status(expected_makers, product["manufacturer_name"])
                code_conflict = bool(expected_codes and product["product_code"] not in expected_codes)
                match_record = {
                    "source_kind": context.get("source_kind"),
                    "module": context.get("module"),
                    "manufacturer_part": expected_mpn,
                    "manufacturer_names": expected_makers,
                    "supplier_codes": expected_codes,
                    "product_code": product["product_code"],
                    "product_manufacturer_name": product["manufacturer_name"],
                    "manufacturer_match_status": maker_status,
                    "supplier_code_conflict": code_conflict,
                    "designators": context.get("designators")
                    or ([context.get("designator")] if context.get("designator") else []),
                    "quantity_context": context.get("quantity_context", {}),
                }
                flags = []
                if code_conflict:
                    flags.append("supplier_code_conflict")
                if maker_status == "name_difference_review":
                    flags.append("manufacturer_name_difference")
                if flags:
                    match_record["review_flags"] = flags
                    page["rejected_or_flagged_matches"].append(match_record)
                else:
                    page["exact_procurement_matches"].append(match_record)
            elif (
                not is_quote
                and context.get("carrier_manufacturer_part") == product["product_model"]
                and context.get("carrier_manufacturer_part") != expected_mpn
            ):
                page["rejected_or_flagged_matches"].append(
                    {
                        "source_kind": context.get("source_kind"),
                        "module": context.get("module"),
                        "designators": [context.get("designator")],
                        "actual_manufacturer_part": expected_mpn,
                        "carrier_manufacturer_part": context.get("carrier_manufacturer_part"),
                        "observed_product_model": product["product_model"],
                        "review_flags": ["carrier_url_does_not_match_procurement_mpn"],
                    }
                )

    # Also bind quote identities to catalog literal URLs when their live public
    # productModel exactly matches the actual purchase MPN.
    for purchase in purchases:
        expected_mpn = purchase.get("manufacturer_part")
        expected_codes = purchase.get("supplier_codes") or []
        expected_makers = purchase.get("manufacturer_names") or []
        for page in pages.values():
            if page.get("product_model") != expected_mpn:
                continue
            quote_direct = any(context == purchase for context in page["source_link_contexts"])
            catalog_actual = any(
                context.get("source_kind") == "catalog_instance_literal"
                and context.get("actual_manufacturer_part") == expected_mpn
                and context.get("module") == purchase.get("module")
                for context in page["source_link_contexts"]
            )
            if not quote_direct and not catalog_actual:
                continue
            code_conflict = bool(expected_codes and page.get("product_code") not in expected_codes)
            maker_status = manufacturer_status(expected_makers, page.get("manufacturer_name"))
            evidence = "quote_input_literal_url" if quote_direct else "catalog_actual_mpn_literal_url"
            identity_key = (
                purchase.get("module"),
                expected_mpn,
                tuple(expected_codes),
                tuple(purchase.get("designators") or []),
            )
            all_existing = page["exact_procurement_matches"] + page["rejected_or_flagged_matches"]
            if any(
                (
                    existing.get("module"),
                    existing.get("manufacturer_part"),
                    tuple(existing.get("supplier_codes") or []),
                    tuple(existing.get("designators") or []),
                )
                == identity_key
                for existing in all_existing
                if existing.get("source_kind", "").startswith("module_price_input")
            ):
                continue
            match_record = {
                "source_kind": purchase.get("source_kind"),
                "link_evidence": evidence,
                "module": purchase.get("module"),
                "manufacturer_part": expected_mpn,
                "manufacturer_names": expected_makers,
                "supplier_codes": expected_codes,
                "product_code": page.get("product_code"),
                "product_manufacturer_name": page.get("manufacturer_name"),
                "manufacturer_match_status": maker_status,
                "supplier_code_conflict": code_conflict,
                "designators": purchase.get("designators", []),
                "quantity_context": purchase.get("quantity_context", {}),
                "source_urls": page.get("source_urls", []),
            }
            flags = []
            if code_conflict:
                flags.append("supplier_code_conflict")
            if maker_status == "name_difference_review":
                flags.append("manufacturer_name_difference")
            if flags:
                match_record["review_flags"] = flags
                page["rejected_or_flagged_matches"].append(match_record)
            else:
                page["exact_procurement_matches"].append(match_record)

    unresolved: list[dict[str, Any]] = []
    for purchase in purchases:
        expected_mpn = purchase.get("manufacturer_part")
        expected_codes = purchase.get("supplier_codes") or []
        attempted_urls = [
            url
            for url, contexts in url_contexts.items()
            if any(context == purchase for context in contexts)
        ]
        matched_pages: list[str] = []
        flagged_pages: list[dict[str, Any]] = []
        catalog_candidate_urls: list[str] = []
        for page in pages.values():
            if page.get("product_model") != expected_mpn:
                continue
            evidence_contexts = [
                context
                for context in page["source_link_contexts"]
                if (
                    context.get("source_kind", "").startswith("module_price_input")
                    and context == purchase
                )
                or (
                    context.get("source_kind") == "catalog_instance_literal"
                    and context.get("actual_manufacturer_part") == expected_mpn
                    and context.get("module") == purchase.get("module")
                )
            ]
            if not evidence_contexts:
                continue
            canonical = page.get("canonical_item_url")
            catalog_candidate_urls.extend(page.get("source_urls", []))
            code_conflict = bool(expected_codes and page.get("product_code") not in expected_codes)
            maker_conflict = manufacturer_status(
                purchase.get("manufacturer_names") or [],
                page.get("manufacturer_name"),
            ) == "name_difference_review"
            if code_conflict or maker_conflict:
                flagged_pages.append(
                    {
                        "canonical_item_url": canonical,
                        "supplier_code_conflict": code_conflict,
                        "manufacturer_name_difference": maker_conflict,
                    }
                )
            elif canonical:
                matched_pages.append(canonical)
        if not matched_pages:
            candidate_urls = sorted(set(attempted_urls + catalog_candidate_urls))
            reason = "no literal official item URL in current inputs"
            if candidate_urls:
                statuses = {fetched[url]["fetch_status"] for url in candidate_urls if url in fetched}
                if any(status != "identity_read" for status in statuses):
                    reason = "candidate page blocked or could not provide public product identity"
                else:
                    reason = "no candidate page productModel exactly matched procurement MPN"
            if flagged_pages:
                reason = "exact productModel found but supplier/manufacturer identity conflicts; review required"
            unresolved.append(
                {
                    "module": purchase["module"],
                    "manufacturer_part": expected_mpn,
                    "manufacturer_names": purchase.get("manufacturer_names", []),
                    "supplier_codes": expected_codes,
                    "designators": purchase.get("designators", []),
                    "quantity_context": purchase.get("quantity_context", {}),
                    "candidate_source_urls": candidate_urls,
                    "flagged_candidate_pages": flagged_pages,
                    "reason": reason,
                }
            )

    for page in pages.values():
        page["source_urls"].sort()
        page["exact_procurement_matches"] = dedupe_dicts(page["exact_procurement_matches"])
        page["rejected_or_flagged_matches"] = dedupe_dicts(page["rejected_or_flagged_matches"])
        page["source_link_contexts"] = dedupe_dicts(page["source_link_contexts"])
    unresolved = dedupe_dicts(unresolved)

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "中央商品链接身份映射；仅验证公开商品身份，不包含价格或库存结论。",
        "source_bindings": current_source_bindings,
        "collection_method": {
            "source_links": "中央元件总表 instances.attrs.instance/device 的 literal HTTPS item.szlcsc.com URL；并合并四个核价输入中的精确商品页直链。",
            "http": "普通公开HTML GET，最多2并发，每个URL一次请求；遇403/203/429/验证码不重试、不绕过。",
            "identity_fields_read": [
                "props.pageProps.webData.productRecord.productModel",
                "props.pageProps.webData.productRecord.productCode",
                "props.pageProps.webData.productRecord.productId",
                "props.pageProps.webData.productRecord.productGradePlateName",
            ],
            "price_or_stock_read": False,
            "match_rule": "仅当公开页 productModel 与采购 MPN 字符串精确相等才形成商品映射；设备载体MPN匹配但采购MPN不匹配的候选拒绝映射。制造商名称差异或LCSC代码冲突保留复核标记。",
        },
        "summary": {
            "catalog_instances_scanned": len(catalog.get("instances", [])),
            "catalog_literal_item_url_contexts": sum(
                1
                for contexts in url_contexts.values()
                for context in contexts
                if context.get("source_kind") == "catalog_instance_literal"
            ),
            "unique_candidate_source_urls": len(urls),
            "http_identity_read_urls": sum(1 for value in fetched.values() if value["fetch_status"] == "identity_read"),
            "canonical_product_pages": len(pages),
            "procurement_identity_records": len(purchases),
            "procurement_exact_mappings": sum(
                1
                for page in pages.values()
                for match in page["exact_procurement_matches"]
                if match.get("source_kind", "").startswith("module_price_input")
            ),
            "procurement_identity_conflicts": sum(
                1
                for page in pages.values()
                for match in page["rejected_or_flagged_matches"]
                if match.get("source_kind", "").startswith("module_price_input")
            ),
            "unmatched_or_conflicted_procurement_records": len(unresolved),
            "unmatched_without_candidate_page": sum(
                1 for item in unresolved if not item.get("candidate_source_urls")
            ),
        },
        "http_observations": [fetched[url] for url in urls],
        "canonical_product_pages": sorted(
            pages.values(),
            key=lambda page: (page["product_model"] or "", page["canonical_item_url"]),
        ),
        "unresolved_procurement_identities": unresolved,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "output_sha256": sha256_file(OUTPUT_PATH),
                "summary": result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
