"""Independently check exported electrical topology and the published schematic/BOM."""

import csv
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree

from reference import ROOT, read_json, source_record


def verify_circuit(reference_path, directory, count):
    reference = read_json(reference_path)
    circuit_path = directory / "circuit.json"
    bom_path = directory / "bom.csv"
    svg_path = directory / "schematic.svg"
    elements = read_json(circuit_path)
    reference_parts = {part["ref"]: part for part in reference["components"]}
    findings = []

    def check(condition, category, subject, expected=None, actual=None):
        if not condition:
            findings.append({"category": category, "subject": subject, "expected": expected, "actual": actual})
        return condition

    for role, upstream in reference["inputs"].items():
        path = ROOT / upstream["path"]
        check(path.is_file() and source_record(path) == upstream, "provenance", role, upstream["sha256"])
    check(len(reference_parts) == count and len(reference["components"]) == count,
          "reference", "reference scope", count, len(reference_parts))

    by_type = defaultdict(list)
    for element in elements:
        by_type[element["type"]].append(element)

    def index(type_name, field):
        rows = by_type[type_name]
        indexed = {row[field]: row for row in rows}
        check(len(indexed) == len(rows), "duplicate", f"{type_name} {field}")
        return indexed

    parts = index("source_component", "name")
    parts_by_id = index("source_component", "source_component_id")
    missing = sorted(reference_parts.keys() - parts.keys())
    added = sorted(parts.keys() - reference_parts.keys())
    check(len(parts) == count and not missing and not added, "scope", "component reference inventory",
          count, {"count": len(parts), "missing": missing, "extra": added})
    suppliers_matched = 0
    uncoded = []
    for ref, expected in reference_parts.items():
        if expected["lcsc"] is None:
            uncoded.append(ref)
        if ref not in parts:
            continue
        actual = parts[ref]
        for field, json_field in [("mpn", "manufacturer_part_number"), ("manufacturer", "manufacturer"),
                                  ("footprint", "footprint_name"), ("dnp", "do_not_place"),
                                  ("placement", "placement"), ("in_bom", "in_bom")]:
            wanted = expected.get(field, field == "in_bom")
            check(wanted == actual.get(json_field), "attributes", f"{ref}.{field}", wanted, actual.get(json_field))
        codes = actual.get("supplier_part_numbers", {}).get("jlcpcb", [])
        exact = [expected["lcsc"]] if expected["lcsc"] else []
        if check(exact == codes, "supplier", ref, exact, codes) and exact:
            suppliers_matched += 1

    ports = index("source_port", "source_port_id")
    nets = index("source_net", "source_net_id")
    by_pin = defaultdict(dict)
    for port_id, port in ports.items():
        owner = parts_by_id.get(port["source_component_id"])
        number = port["contract_pin_number"]
        if not check(owner is not None, "pin", f"{port_id} owner"):
            continue
        pin_map = by_pin[owner["name"]]
        check(number not in pin_map, "pin", f"{owner['name']}.{number} duplicate")
        pin_map[number] = port_id
        if number.isdecimal():
            check(port["pin_number"] == int(number), "pin", f"{owner['name']}.{number} physical number",
                  int(number), port["pin_number"])

    expected_nets = {net for part in reference_parts.values() for net in part["pins"].values() if net is not None}
    actual_nets = {row["name"] for row in nets.values()}
    check(actual_nets == expected_nets and len(nets) == len(actual_nets), "net", "named nets",
          sorted(expected_nets), sorted(row["name"] for row in nets.values()))

    source_traces = index("source_trace", "source_trace_id")
    sch_ports = index("schematic_port", "source_port_id")
    sch_wires = index("schematic_trace", "source_trace_id")
    sch_labels = index("schematic_net_label", "source_trace_id")
    check(set(sch_ports) == set(ports), "drawing", "schematic/source port IDs")
    check(set(sch_wires) == set(sch_labels) == set(source_traces), "drawing", "schematic/source trace IDs")
    check(not by_type["source_component_internal_connection"], "drawing", "no undeclared internal tie")

    pin_connections = defaultdict(list)
    for trace_id, trace in source_traces.items():
        port_ids, net_ids = trace.get("connected_source_port_ids", []), trace.get("connected_source_net_ids", [])
        if not check(len(port_ids) == len(net_ids) == 1 and port_ids[0] in ports and net_ids[0] in nets,
                     "trace", trace_id, "one valid pin and one valid net", [port_ids, net_ids]):
            continue
        port_id, net_id = port_ids[0], net_ids[0]
        net = nets[net_id]["name"]
        pin_connections[port_id].append(net)
        port = sch_ports.get(port_id, {})
        wire = sch_wires.get(trace_id, {}).get("edges", [])
        label = sch_labels.get(trace_id, {})
        check(len(wire) == 1 and wire[0]["from"] == port.get("center") and
              wire[0]["to"] == label.get("anchor_position") and wire[0]["from"] != wire[0]["to"] and
              label.get("source_net_id") == net_id and label.get("text") == net,
              "drawing", trace_id, f"pin anchored to {net} via a wire and label")

    matched = total = 0
    for ref, component in reference_parts.items():
        expected_pins = component["pins"]
        exported_pins = by_pin[ref]
        check(set(expected_pins) == set(exported_pins), "pin", f"{ref} pin inventory",
              sorted(expected_pins), sorted(exported_pins))
        for number, net in expected_pins.items():
            total += 1
            port_id = exported_pins.get(number)
            actual = pin_connections[port_id]
            wanted = [] if net is None else [net]
            if check(port_id is not None and actual == wanted, "pin_net", f"{ref}.{number}", wanted, actual):
                matched += 1

    with bom_path.open(encoding="utf-8-sig", newline="") as handle:
        bom_rows = list(csv.DictReader(handle))
    bom = {row["Reference"]: row for row in bom_rows}
    check(len(bom_rows) == count and bom.keys() == reference_parts.keys(), "bom", "component inventory")
    for ref, component in reference_parts.items():
        expected = {
            "MPN": component["mpn"] or "", "LCSC": component["lcsc"] or "",
            "Manufacturer": component["manufacturer"] or "", "Footprint": component["footprint"] or "",
            "DNP": "yes" if component["dnp"] else "no", "Placement": component["placement"],
            "In BOM": "yes" if component.get("in_bom", True) else "no", "Value": component["value"],
        }
        row = bom.get(ref, {})
        actual = {field: row.get(field) for field in expected}
        check(expected == actual, "bom", ref, expected, actual)
    errors = [kind for kind in by_type if "error" in kind]
    check(not errors, "render", "tscircuit render errors", [], errors)

    svg = svg_path.read_text(encoding="utf-8")
    document = ElementTree.fromstring(svg)
    check(not re.search(r"\bNaN", svg), "render", "invalid SVG number or passive value")
    xy = document.attrib["data-real-to-screen-transform"].removeprefix("matrix(").removesuffix(")")
    sx, skew_y, skew_x, sy, dx, dy = map(float, xy.split(","))
    width, height = (float(document.attrib[key]) for key in ("width", "height"))
    bodies = by_type["schematic_component"]
    check(len(bodies) == count and len({row["source_component_id"] for row in bodies}) == count,
          "geometry", "schematic symbol count", count, len(bodies))
    for body in bodies:
        x, y = body["center"]["x"], body["center"]["y"]
        check(0 < sx * x + skew_x * y + dx < width and 0 < skew_y * x + sy * y + dy < height,
              "geometry", f"{body['schematic_component_id']} lies inside canvas")
    for a, b in combinations(bodies, 2):
        if all(abs(a["center"][axis] - b["center"][axis]) <
               (a["size"][dimension] + b["size"][dimension]) / 2
               for axis, dimension in (("x", "width"), ("y", "height"))):
            check(False, "geometry", f"{a['source_component_id']} overlaps {b['source_component_id']}")

    from verify_kicad import verify_kicad
    kicad = verify_kicad(directory / "kicad", reference_parts, check)
    kicad["files"] = [source_record(path) for path in sorted((directory / "kicad").iterdir())
                      if path.suffix in (".kicad_pro", ".kicad_sch")]
    return {
        "status": "FAIL" if findings else "PASS",
        "scope": "Frozen contract schematic objects, physical/logical pins, net labels, BOM and KiCad schematic readback",
        "contract_id": reference["contract_id"],
        "inputs": reference["inputs"],
        "artifacts": {key: source_record(path) for key, path in {
            "reference": reference_path, "circuit": circuit_path, "svg": svg_path, "bom": bom_path,
        }.items()},
        "components": {
            "expected": count, "observed": len(parts), "missing": missing, "extra": added,
            "by_placement": dict(sorted(Counter(part["placement"] for part in reference_parts.values()).items())),
            "specified_lcsc_count": count - len(uncoded), "specified_lcsc_matching": suppliers_matched,
            "refs_without_lcsc": uncoded,
        },
        "pin_net_comparison": {"equal": matched, "total": total,
                               "distinct_named_nets": len(actual_nets), "percent": round(100 * matched / total, 3)},
        "kicad": kicad,
        "render_warnings": {kind: len(rows) for kind, rows in sorted(by_type.items()) if "warning" in kind},
        "footprints": "Matches contractual footprint identities. No PCB, physical footprint library or land-pattern qualification is generated for these three boards.",
        "findings": findings,
    }
