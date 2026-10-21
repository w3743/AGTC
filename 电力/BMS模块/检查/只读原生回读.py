#!/usr/bin/env python3
"""Read-only EasyEDA schematic readback for the current BMS build inputs.

This script opens each mapped page and reads its placed parts, pin wires,
netports, NC markers, MPN, BOM/PCB flags, and footprint IDs. It never modifies,
deletes, or saves project data. Expected entries are staged in batches of at
most three through globalThis to keep each CLI invocation below command-line
length limits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLI = Path(r"D:\lceda-pro\lceda-pro.exe")
DEFAULT_INPUT = Path("当前施工输入.json")
DEFAULT_PAGE_MAP = Path("原理图页面映射.json")
DEFAULT_OUTPUT = Path("原理图只读回读.json")
STAGE_SIZE = 10


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields needed for comparison and handoff evidence."""
    result: dict[str, Any] = {
        "ref": entry.get("ref"),
        "mpn": entry.get("mpn"),
        "value": entry.get("value"),
        "pins": entry.get("pins") or {},
    }
    for key in ("addIntoPcb", "addIntoBom"):
        if key in entry:
            result[key] = entry[key]

    candidate = entry.get("candidate") or {}
    carrier = entry.get("footprintCarrier") or {}
    footprint_uuid = candidate.get("footprintUuid") or carrier.get("footprintUuid")
    if footprint_uuid:
        result["expectedFootprint"] = {
            "libraryUuid": candidate.get("footprintLibraryUuid")
            or carrier.get("footprintLibraryUuid")
            or candidate.get("libraryUuid")
            or carrier.get("libraryUuid"),
            "uuid": footprint_uuid,
            "name": candidate.get("footprintName") or carrier.get("footprintName"),
        }
    return result


def run_cli(cli: Path, session: str, timeout_ms: int, code: str) -> Any:
    command = [
        str(cli),
        "invoke",
        "--session",
        session,
        "--ext-uuid",
        "eda",
        "--timeout",
        str(timeout_ms),
        "--code",
        code,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-3000:]
        raise RuntimeError(f"EasyEDA CLI exited {completed.returncode}: {detail}")
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stdout.strip()[-3000:]
        raise RuntimeError(f"EasyEDA CLI returned non-JSON output: {detail}") from exc
    if not isinstance(response, dict) or not response.get("ok"):
        raise RuntimeError(f"EasyEDA invocation failed: {response!r}")
    return response.get("value")


READ_PAGE_JS = r"""
const entries = globalThis.__bms_ro_entries || [];
await eda.dmt_EditorControl.openDocument(__PAGE_ID__);
const components = await eda.sch_PrimitiveComponent.getAll();
const wires = (await eda.sch_PrimitiveWire.getAll()).map(w => ({
  net: w.getState_Net(), line: w.getState_Line()
}));
const netports = [];
for (const c of components.filter(c => c.getState_ComponentType() === 'netport')) {
  const pins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
  for (const p of (pins || [])) {
    netports.push({name: c.getState_Name(), x: p.x, y: p.y});
  }
}
const errors = [];
const parts = [];
const same = (a, b) => Math.abs(a - b) < 0.001;
const normalize = v => v === undefined || v === null || v === '' ? null : String(v);
const partComponents = components.filter(c => c.getState_ComponentType() === 'part');
for (const c of partComponents) {
  const ref = c.getState_Designator();
  const e = entries.find(item => item.ref === ref);
  if (!e) errors.push({ref, kind: 'unexpected part'});
  const primitiveId = c.getState_PrimitiveId();
  const componentPins = await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(primitiveId);
  const pins = [];
  for (const p of (componentPins || [])) {
    const pinKey = String(p.pinNumber);
    const hasExpected = !!e && Object.prototype.hasOwnProperty.call(e.pins || {}, pinKey);
    const expected = hasExpected ? e.pins[pinKey] : undefined;
    const at = wires.filter(w => Array.isArray(w.line) && w.line.length === 4 && (
      (same(w.line[0], p.x) && same(w.line[1], p.y)) ||
      (same(w.line[2], p.x) && same(w.line[3], p.y))
    ));
    const connected = at.map(w => {
      const startsAtPin = same(w.line[0], p.x) && same(w.line[1], p.y);
      const far = startsAtPin ? [w.line[2], w.line[3]] : [w.line[0], w.line[1]];
      return {
        net: w.net,
        ports: netports.filter(port => same(port.x, far[0]) && same(port.y, far[1])).map(port => port.name)
      };
    });
    const noConnected = p.getState_NoConnected();
    if (!hasExpected) {
      errors.push({ref, pin: pinKey, kind: 'unexpected pin'});
    } else if (expected === null) {
      if (!noConnected || at.length) {
        errors.push({ref, pin: pinKey, kind: 'NC mismatch', noConnected, connected});
      }
    } else if (noConnected || connected.length !== 1 || connected[0].net !== expected ||
               !connected[0].ports.includes(expected)) {
      errors.push({ref, pin: pinKey, kind: 'wire/port mismatch', expected, noConnected, connected});
    }
    pins.push({
      pin: pinKey, name: p.pinName, x: p.x, y: p.y, noConnected,
      expected: hasExpected ? expected : undefined, connected
    });
  }

  const expectedPins = e?.pins || {};
  const absentPins = Object.entries(expectedPins).filter(([pin, net]) =>
    net !== null && !(componentPins || []).some(p => String(p.pinNumber) === pin)
  );
  if (absentPins.length) errors.push({ref, kind: 'missing live pins', absentPins});

  const expectedName = e?.value ?? e?.mpn ?? null;
  const actualName = c.getState_Name();
  const actualMpn = c.getState_ManufacturerId();
  if (e && expectedName !== null && normalize(actualName) !== normalize(expectedName)) {
    errors.push({ref, kind: 'value mismatch', expected: expectedName, actual: actualName});
  }
  if (e && normalize(actualMpn) !== normalize(e.mpn)) {
    errors.push({ref, kind: 'MPN mismatch', expected: e.mpn, actual: actualMpn});
  }

  const expectedFootprint = e?.expectedFootprint || null;
  const actualFootprint = c.getState_Footprint() || null;
  if (expectedFootprint && normalize(actualFootprint?.uuid) !== normalize(expectedFootprint.uuid)) {
    errors.push({
      ref, kind: 'footprint UUID mismatch', expected: expectedFootprint,
      actual: actualFootprint
    });
  }
  const actualAddIntoPcb = c.getState_AddIntoPcb();
  const actualAddIntoBom = c.getState_AddIntoBom();
  if (e && Object.prototype.hasOwnProperty.call(e, 'addIntoPcb') &&
      actualAddIntoPcb !== e.addIntoPcb) {
    errors.push({ref, kind: 'addIntoPcb mismatch', expected: e.addIntoPcb, actual: actualAddIntoPcb});
  }
  if (e && Object.prototype.hasOwnProperty.call(e, 'addIntoBom') &&
      actualAddIntoBom !== e.addIntoBom) {
    errors.push({ref, kind: 'addIntoBom mismatch', expected: e.addIntoBom, actual: actualAddIntoBom});
  }

  parts.push({
    ref, value: actualName, mpn: actualMpn,
    addIntoPcb: actualAddIntoPcb, addIntoBom: actualAddIntoBom,
    footprint: actualFootprint, expectedFootprint, pins
  });
}
for (const e of entries) {
  const count = parts.filter(p => p.ref === e.ref).length;
  if (count !== 1) errors.push({ref: e.ref, kind: 'component count', count});
}
return {
  page: __PAGE_ID__, group: __PAGE_GROUP__, expectedPartCount: entries.length,
  partCount: parts.length, wireCount: wires.length, netportCount: netports.length,
  errors, parts, wires, netports
};
"""


def make_read_page_code(page_id: str, group: str) -> str:
    return (
        READ_PAGE_JS.replace("__PAGE_ID__", json.dumps(page_id, ensure_ascii=False))
        .replace("__PAGE_GROUP__", json.dumps(group, ensure_ascii=False))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="EasyEDA CLI session ID")
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="Path to lceda-pro.exe")
    parser.add_argument("--input", type=Path, default=None, help="施工输入 JSON (default: script folder)")
    parser.add_argument("--page-map", type=Path, default=None, help="page mapping JSON (default: script folder)")
    parser.add_argument("--output", type=Path, default=None, help="readback JSON output path")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="per invocation timeout")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = args.input or (script_dir / DEFAULT_INPUT)
    page_map_path = args.page_map or (script_dir / DEFAULT_PAGE_MAP)
    output_path = args.output or (script_dir / DEFAULT_OUTPUT)
    if not args.cli.is_file():
        raise FileNotFoundError(f"EasyEDA CLI not found: {args.cli}")

    entries = read_json(input_path)
    page_map = read_json(page_map_path)
    if not isinstance(entries, list) or not isinstance(page_map, dict):
        raise ValueError("Input must be an array and page map must be a JSON object")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        group = entry.get("pageGroup")
        if not group:
            raise ValueError(f"Entry has no pageGroup: {entry.get('ref')}")
        grouped[group].append(compact_entry(entry))
    missing_pages = sorted(set(grouped) - set(page_map))
    empty_pages = sorted(set(page_map) - set(grouped))
    if missing_pages or empty_pages:
        raise ValueError(f"Page map/input mismatch; missing page IDs={missing_pages}; empty page groups={empty_pages}")

    reports: list[dict[str, Any]] = []
    total_calls = sum((len(grouped[group]) + STAGE_SIZE - 1) // STAGE_SIZE + 2 for group in page_map)
    call_index = 0
    for group, page_id in page_map.items():
        group_entries = grouped[group]
        call_index += 1
        run_cli(args.cli, args.session, args.timeout_ms, "globalThis.__bms_ro_entries = []; return 0;")
        for start in range(0, len(group_entries), STAGE_SIZE):
            batch = group_entries[start : start + STAGE_SIZE]
            batch_json = json.dumps(batch, ensure_ascii=False, separators=(",", ":"))
            code = (
                "globalThis.__bms_ro_entries.push(..."
                + batch_json
                + "); return globalThis.__bms_ro_entries.length;"
            )
            call_index += 1
            run_cli(args.cli, args.session, args.timeout_ms, code)

        call_index += 1
        report = run_cli(args.cli, args.session, args.timeout_ms, make_read_page_code(page_id, group))
        if not isinstance(report, dict):
            raise RuntimeError(f"Unexpected page response for {group}: {report!r}")
        reports.append(report)
        print(
            f"[{call_index}/{total_calls}] {group}: "
            f"parts={report.get('partCount')} wires={report.get('wireCount')} "
            f"ports={report.get('netportCount')} errors={len(report.get('errors') or [])}",
            flush=True,
        )

    result = {
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "mode": "read-only; no modify/delete/save calls",
        "sourceInput": str(input_path),
        "pageMap": str(page_map_path),
        "expectedPageCount": len(page_map),
        "expectedPartCount": len(entries),
        "actualPageCount": len(reports),
        "errorCount": sum(len(page.get("errors") or []) for page in reports),
        "pages": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved read-only report: {output_path}")
    return 0 if result["errorCount"] == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # CLI diagnostics for the operator
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
