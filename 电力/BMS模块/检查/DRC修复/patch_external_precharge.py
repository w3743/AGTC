#!/usr/bin/env python3
"""Append an off-board R_PRE symbol and two short nets to the current native sheet.

This script does not call EasyEDA. By default it writes a separate candidate file.
Use --in-place only after reviewing the generated change and closing EasyEDA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path


EXPECTED_SHA256 = "A07DB6C1A04A9F9DED55ACC70D6F14748715F7522522E3BF438F2E26ADE027D1"
SOURCE_REL = Path("BMS模块/sch/Schematic1/P1.esch2")
RESISTOR_TEMPLATE_ID = "26e3049ba6b5564b"
PORT_TEMPLATE_ID = "223f232ff73759f6"
WIRE_TEMPLATE_ID = "2a37512986ff86a4"
RESISTOR_PART = "RC0603FR-07510RL.1"
PORT_PART = "pid8a0e77bacb214e"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def parse_native(data: bytes):
    records = []
    for number, raw in enumerate(data.splitlines(), start=1):
        if not raw:
            continue
        if b"||" not in raw:
            raise ValueError(f"第{number}行不是native记录，拒绝修改")
        head_raw, body_raw = raw.split(b"||", 1)
        if body_raw.endswith(b"|"):
            body_raw = body_raw[:-1]
        try:
            head = json.loads(head_raw.decode("utf-8"))
            body = json.loads(body_raw.decode("utf-8")) if body_raw.strip() else None
        except Exception as exc:
            raise ValueError(f"第{number}行JSON解析失败：{exc}") from exc
        if not isinstance(head, dict):
            raise ValueError(f"第{number}行header不是对象，拒绝修改")
        if body is not None and not isinstance(body, dict):
            body = None
        records.append((head, body, raw))
    return records


def latest_live_records(records):
    """Fold native history by (type,id); empty/null payloads are tombstones."""
    latest = {}
    for row in records:
        head, body, _ = row
        kind, object_id = head.get("type"), head.get("id")
        if kind is None or object_id is None:
            continue
        key = (kind, object_id)
        latest[key] = row if body is not None else None
    return [row for row in latest.values() if row is not None]


def native_line(head: dict, body: dict) -> bytes:
    return (
        json.dumps(head, ensure_ascii=False, separators=(",", ":"))
        + "||"
        + json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        + "|"
    ).encode("utf-8")


def find_record(records, kind: str, object_id: str):
    found = [(h, d) for h, d, _ in records if h.get("type") == kind and h.get("id") == object_id]
    if len(found) != 1:
        raise ValueError(f"模板{kind}/{object_id}数量应为1，实际{len(found)}")
    return found[0]


def attr_rows(records, parent_id: str):
    return [(h, d) for h, d, _ in records if h.get("type") == "ATTR" and d.get("parentId") == parent_id]


def allocate_id(used_ids: set[str]) -> str:
    while True:
        candidate = uuid.uuid4().hex[:16]
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate


def make_patch(records, all_records, data: bytes):
    resistor_head, resistor_data = find_record(records, "COMPONENT", RESISTOR_TEMPLATE_ID)
    if resistor_data.get("partId") != RESISTOR_PART:
        raise ValueError("R_PSU_MINLOAD1模板partId与预期不符")
    port_head, port_data = find_record(records, "COMPONENT", PORT_TEMPLATE_ID)
    if port_data.get("partId") != PORT_PART:
        raise ValueError("Netport-BI模板partId与预期不符")
    wire_head, wire_data = find_record(records, "WIRE", WIRE_TEMPLATE_ID)

    resistor_attrs = attr_rows(records, RESISTOR_TEMPLATE_ID)
    resistor_by_key = {d.get("key"): (h, d) for h, d in resistor_attrs}
    for required in ("Unique ID", "Designator", "Symbol", "Device", "Add into BOM", "Convert to PCB", "Footprint", "Value", "Supplier", "Supplier Part"):
        if required not in resistor_by_key:
            raise ValueError(f"R_PSU_MINLOAD1缺少必需属性：{required}")
    if len(resistor_by_key) != len(resistor_attrs):
        raise ValueError("R_PSU_MINLOAD1存在重名ATTR，不能安全克隆")
    carrier_device_id = resistor_by_key["Device"][1].get("value")
    carrier_device_name = resistor_data.get("attrs", {}).get("DeviceName")
    if not isinstance(carrier_device_id, str) or not carrier_device_id:
        raise ValueError("R_PSU_MINLOAD1的Device属性为空，不能作为symbol carrier")
    if not isinstance(carrier_device_name, dict) or carrier_device_name.get("uuid") != carrier_device_id:
        raise ValueError("R_PSU_MINLOAD1的Device属性与组件DeviceName不一致")

    # Do not create a second R_PRE or silently apply on a changed sheet.
    for head, body, _ in records:
        if head.get("type") == "ATTR" and body.get("key") in ("Unique ID", "Designator") and body.get("value") == "R_PRE":
            raise ValueError("当前原理图已存在R_PRE，脚本停止")

    source_parts = [(h, d) for h, d, _ in records if h.get("type") == "PART" and h.get("id") == RESISTOR_PART]
    if not source_parts:
        raise ValueError("原理图中找不到0603符号定义")
    bbox = source_parts[0][1].get("BBOX")
    if bbox != [-10, -4, 10, 4]:
        raise ValueError(f"0603符号BBOX与已核对模板不符：{bbox!r}")

    port_attrs = attr_rows(records, PORT_TEMPLATE_ID)
    port_by_key = {d.get("key"): (h, d) for h, d in port_attrs}
    for required in ("Symbol", "Device", "Name", "Description"):
        if required not in port_by_key:
            raise ValueError(f"Netport-BI缺少必需属性：{required}")

    line_rows = [(h, d) for h, d, _ in records if h.get("type") == "LINE" and d.get("lineGroup") == WIRE_TEMPLATE_ID]
    wire_attrs = attr_rows(records, WIRE_TEMPLATE_ID)
    wire_attr_by_key = {d.get("key"): (h, d) for h, d in wire_attrs}
    if len(line_rows) != 1 or not all(k in wire_attr_by_key for k in ("NET", "Relevance")):
        raise ValueError("QPRE模板导线结构不符合预期")

    used_ids = {h.get("id") for h, _, _ in all_records if isinstance(h.get("id"), str)}
    # Also guard UUID-like values stored outside record headers.
    used_ids.update(x.decode("ascii").lower() for x in re.findall(rb'"(?:id|uuid)"\s*:\s*"([0-9a-fA-F]{16,32})"', data))
    max_ticket = max((h.get("ticket", 0) for h, _, _ in all_records if isinstance(h.get("ticket"), int)), default=0)
    next_ticket = max_ticket + 1
    additions: list[tuple[dict, dict]] = []

    def add(kind: str, body: dict, object_id: str | None = None, z_head: dict | None = None):
        nonlocal next_ticket
        head = {"type": kind, "ticket": next_ticket, "id": object_id or allocate_id(used_ids)}
        next_ticket += 1
        if z_head:
            for key in ("zIndex",):
                if key in z_head:
                    body.setdefault(key, z_head[key])
        additions.append((head, body))
        return head["id"]

    def clone_attrs(source_rows, new_parent: str, dx: int, dy: int, overrides=None, skip=()):
        overrides = overrides or {}
        for _, original in source_rows:
            key = original.get("key")
            if key in skip:
                continue
            body = copy.deepcopy(original)
            body["parentId"] = new_parent
            for coord, delta in (("x", dx), ("y", dy)):
                if isinstance(body.get(coord), (int, float)) and not isinstance(body.get(coord), bool):
                    body[coord] += delta
            if key in overrides:
                body["value"] = overrides[key]
            add("ATTR", body)

    # R_PRE is an off-board BOM item using the existing resistor symbol/device as a graphic carrier.
    r_id = allocate_id(used_ids)
    dx, dy = 780 - resistor_data["x"], 1660 - resistor_data["y"]
    r_body = copy.deepcopy(resistor_data)
    r_body.update(x=780, y=1660)
    r_body["attrs"] = copy.deepcopy(resistor_data.get("attrs", {}))
    add("COMPONENT", r_body, object_id=r_id, z_head=resistor_head)
    # Null/blank catalog fields must not fall back to the 510-ohm carrier's values.
    r_overrides = {
        original.get("key"): "OFFBOARD"
        for _, original in resistor_attrs
        if original.get("key") and (original.get("value") is None or original.get("value") == "")
    }
    r_overrides.update({
        "Unique ID": "R_PRE",
        "Designator": "R_PRE",
        "Name": "8.2R 300W OFFBOARD",
        "Add into BOM": "no",
        "Convert to PCB": "no",
        "Footprint": "OFFBOARD",
        "Manufacturer": "TE Connectivity",
        "Manufacturer Part": "CJP300J8R2J",
        "Supplier": "BOM_ONLY",
        "Supplier Part": "SEE_BOM_CSV",
        "LCSC Part Name": "OFFBOARD",
        "Supplier Footprint": "OFFBOARD",
        "JLCPCB Part Class": "OFFBOARD",
        "Datasheet": "https://www.te.com/en/product-CJP300J8R2J.html",
        "3D Model": "OFFBOARD",
        "3D Model Title": "OFFBOARD",
        "3D Model Transform": "OFFBOARD",
        "Type": "Power resistor, external/off-board",
        "Value": "8.2Ω ±5% 300W",
        "Tolerance": "±5%",
        "Power(Watts)": "300W",
        "Electrical manifest": "R_PRE",
        "Description": "External precharge resistor; symbol-only, not placed on PCB.",
        "Design note": "Off-board TE Connectivity CJP300J8R2J; 8.2Ω ±5%, 300W; see BOM CSV.",
    })
    clone_attrs(resistor_attrs, r_id, dx, dy, r_overrides)

    # Netport BI has a pin at the component origin in the currently inspected library.
    for port_x, net, label_side in ((720, "QPRE_SW", "left"), (840, "PACK+", "right")):
        p_id = allocate_id(used_ids)
        pdx, pdy = port_x - port_data["x"], 1660 - port_data["y"]
        p_body = copy.deepcopy(port_data)
        p_body.update(x=port_x, y=1660)
        add("COMPONENT", p_body, object_id=p_id, z_head=port_head)
        for _, original in port_attrs:
            body = copy.deepcopy(original)
            body["parentId"] = p_id
            for coord, delta in (("x", pdx), ("y", pdy)):
                if isinstance(body.get(coord), (int, float)) and not isinstance(body.get(coord), bool):
                    body[coord] += delta
            if body.get("key") == "Name":
                body["value"] = net
                body["x"] = port_x - 40 if label_side == "left" else port_x + 45
                body["y"] = 1660
                body["align"] = "RIGHT_MIDDLE" if label_side == "left" else "LEFT_MIDDLE"
            add("ATTR", body)

    # Pin geometry already checked against the present templates: resistor pins x=760/800;
    # Netport-BI pins at component origins x=720/840. Each stub is one native LINE.
    for start_x, end_x, net in ((720, 760, "QPRE_SW"), (800, 840, "PACK+")):
        w_id = allocate_id(used_ids)
        w_body = copy.deepcopy(wire_data)
        add("WIRE", w_body, object_id=w_id, z_head=wire_head)
        line_head, source_line = line_rows[0]
        line_body = copy.deepcopy(source_line)
        line_body.update(startX=start_x, startY=1660, endX=end_x, endY=1660, lineGroup=w_id, yAxisDirection="up")
        add("LINE", line_body, z_head=line_head)
        for key in ("NET", "Relevance"):
            _, original = wire_attr_by_key[key]
            body = copy.deepcopy(original)
            body["parentId"] = w_id
            if key == "NET":
                body.update(x=(start_x + end_x) / 2, y=1660, value=net, keyVisible=False, valueVisible=True)
            add("ATTR", body)

    added_ids = [h["id"] for h, _ in additions]
    if len(added_ids) != len(set(added_ids)) or set(added_ids) & {h.get("id") for h, _, _ in records}:
        raise ValueError("新对象ID发生冲突")
    if [h["ticket"] for h, _ in additions] != list(range(max_ticket + 1, max_ticket + 1 + len(additions))):
        raise ValueError("新增ticket不连续")
    if any(h.get("type") == "JUNCTION" for h, _ in additions):
        raise ValueError("补丁不应生成JUNCTION")

    newline = b"\r\n" if b"\r\n" in data else b"\n"
    if data.endswith((b"\n", b"\r")):
        prefix = data
        separator = b""
    else:
        # The source EOF may omit the record's terminal pipe. Close that record
        # before adding the newline that separates the newly appended records.
        final_record = data.rsplit(b"\n", 1)[-1].rstrip(b"\r")
        prefix = data + (b"" if final_record.endswith(b"|") else b"|")
        separator = newline
    appended = separator + newline.join(native_line(h, d) for h, d in additions) + newline
    result = prefix + appended
    if not result.startswith(data):
        raise AssertionError("原记录前缀被改写")

    summary = {
        "source": str(SOURCE_REL),
        "source_sha256": sha256(data),
        "result_sha256": sha256(result),
        "preserved_original_bytes": result.startswith(data),
        "added_record_count": len(additions),
        "added_type_counts": dict(Counter(h["type"] for h, _ in additions)),
        "ticket_range": [max_ticket + 1, next_ticket - 1],
        "R_PRE": {
            "component_id": r_id,
            "partId": RESISTOR_PART,
            "center": [780, 1660],
            "pin1": [760, 1660],
            "pin2": [800, 1660],
            "mpn": "TE Connectivity CJP300J8R2J",
            "value": "8.2Ω ±5% 300W",
            "BOM": "BOM_ONLY / SEE_BOM_CSV; Add into BOM=no; Convert to PCB=no",
            "carrier_device_retained": {
                "Device_ATTR": carrier_device_id,
                "DeviceName": carrier_device_name.get("name"),
                "symbol_only": True,
            },
            "empty_or_null_template_attrs_marked_OFFBOARD": True,
        },
        "ports": [
            {"component_id": p_id, "net": net, "pin": [x, 1660]}
            for p_id, net, x in ((next(h["id"] for h, d in additions if h["type"] == "COMPONENT" and d.get("partId") == PORT_PART and d.get("x") == 720), "QPRE_SW", 720),
                                 (next(h["id"] for h, d in additions if h["type"] == "COMPONENT" and d.get("partId") == PORT_PART and d.get("x") == 840), "PACK+", 840))
        ],
        "wires": [
            {"net": net, "line": [[sx, 1660], [ex, 1660]], "wire_id": wid}
            for (sx, ex, net), wid in zip(((720, 760, "QPRE_SW"), (800, 840, "PACK+")),
                                          [h["id"] for h, d in additions if h["type"] == "WIRE"])
        ],
        "assertions": {
            "R_PRE_names_unique": True,
            "each_wire_group_has_one_NET_attr": True,
            "new_ids_unique": True,
            "new_tickets_contiguous": True,
            "no_junction_or_old_NET_cloned": True,
            "all_preexisting_bytes_unchanged": True,
        },
        "appended_native_rows": [native_line(h, d).decode("utf-8") for h, d in additions],
    }
    return result, summary


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-place", action="store_true", help="原位追加。默认只写独立候选文件")
    parser.add_argument("--output", type=Path, help="候选文件路径，仅适用于默认候选模式")
    args = parser.parse_args()
    if args.in_place and args.output:
        parser.error("--in-place和--output不能同时使用")

    source = project_root / SOURCE_REL
    original = source.read_bytes()
    actual_sha = sha256(original)
    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(f"源文件SHA不匹配，停止：expected={EXPECTED_SHA256} actual={actual_sha}")
    all_records = parse_native(original)
    records = latest_live_records(all_records)
    result, manifest = make_patch(records, all_records, original)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.in_place:
        output = source
        manifest_path = script_dir / f"native-precharge-patch-{stamp}.json"
    else:
        output = args.output or (script_dir / "P1_with_external_R_PRE_candidate.esch2")
        if not output.is_absolute():
            output = project_root / output
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and not args.in_place:
        raise SystemExit(f"候选文件已存在，拒绝覆盖：{output}")
    if manifest_path.exists():
        raise SystemExit("manifest目标已存在，拒绝覆盖")

    manifest.update({"output": str(output), "backup": None})
    if args.in_place:
        if sha256(source.read_bytes()) != EXPECTED_SHA256:
            raise SystemExit("源文件在准备期间发生变化，拒绝覆盖")
        fd, temp_name = tempfile.mkstemp(prefix="P1-R_PRE-", suffix=".tmp", dir=source.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(result)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, source)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    else:
        with output.open("xb") as stream:
            stream.write(result)
            stream.flush()
            os.fsync(stream.fileno())

    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(json.dumps({k: v for k, v in manifest.items() if k != "appended_native_rows"}, ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"patch aborted: {exc}", file=sys.stderr)
        raise
