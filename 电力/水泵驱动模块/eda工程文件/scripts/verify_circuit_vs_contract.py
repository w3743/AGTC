"""Read back the tscircuit topology, schematic stubs and BOM against the contract."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_reference_netlist import (
    PROJECT, REFERENCE, build_reference, read_json, source_record, write_json,
)

EXPORTS = PROJECT / "exports"
sys.path.insert(0, str(PROJECT.parents[1] / "检查/联合仿真/code_as_eda"))
from verify_kicad import verify_kicad

NET_ALIASES = {"V5": "+5V", "V3V3": "+3V3"}
CRITICAL = {
    "D_BUS": ("SMCJ40A", "C840777"),
    "D_5V": ("SS310-SMC", "C2848691"),
    "U_CAN": ("CA-IS2062A", "C22438481"),
    "D_CAN": ("PESD2CANFD24V-TR", "C552486"),
    "J_CAN": ("B3B-XH-A(LF)(SN)", "C144394"),
    "U_PWM": ("NSG1416", "C53436462"),
    "Q_PWM": ("NCE0117K", "C108638"),
    "Q_BRAKE": ("E100P045CL1", "C29781153"),
    "R_PUMP_SHUNT": ("HoJLR2512-2W-20mR-1%-75ppm", "C2924538"),
    "J_MCU_A": ("JXT-PM2.54-2X12P-A9", "C54803369"),
    "J_MCU_B": ("JXT-PM2.54-2X12P-A9", "C54803369"),
}


def verify(reference_path, circuit_path, bom_path):
    reference = read_json(reference_path)
    findings = []

    def check(ok, category, subject, expected=True, observed=None):
        if not ok:
            findings.append({"category": category, "subject": subject,
                             "expected": expected, "observed": observed})
        return ok

    check(reference == build_reference(), "provenance", "reference matches current frozen inputs")
    expected = {part["ref"]: part for part in reference["components"]}
    by_type = defaultdict(list)
    for element in read_json(circuit_path):
        by_type[element["type"]].append(element)
    errors = {kind: len(items) for kind, items in by_type.items() if "error" in kind}
    check(not errors, "render", "Circuit JSON errors", {}, errors)

    def unique_index(kind, key):
        result = {}
        for row in by_type[kind]:
            value = row[key]
            check(value not in result, "duplicate", f"{kind}.{key}={value}")
            result[value] = row
        return result

    components = unique_index("source_component", "name")
    ids = unique_index("source_component", "source_component_id")
    ports = unique_index("source_port", "source_port_id")
    nets = unique_index("source_net", "source_net_id")
    traces = unique_index("source_trace", "source_trace_id")
    drawn_ports = unique_index("schematic_port", "source_port_id")
    drawn_traces = unique_index("schematic_trace", "source_trace_id")
    labels = unique_index("schematic_net_label", "source_trace_id")
    missing = sorted(expected.keys() - components.keys())
    extra = sorted(components.keys() - expected.keys())
    check(not missing and not extra and len(components) == 52, "component_count", "references",
          {"count": 52}, {"count": len(components), "missing": missing, "extra": extra})

    foot_pins = defaultdict(dict)
    for port_id, port in ports.items():
        part = ids.get(port.get("source_component_id"))
        if not check(part is not None, "pin", f"{port_id}: owner"):
            continue
        pin = str(port.get("pin_number"))
        pins = foot_pins[part["name"]]
        check(pin not in pins, "duplicate", f"{part['name']}.{pin}")
        pins[pin] = port_id

    # Union-find over rendered source ports, source net IDs and internal ties.
    groups = {key: key for key in ports.keys() | nets.keys()}

    def representative(node):
        while groups[node] != node:
            groups[node] = groups[groups[node]]
            node = groups[node]
        return node

    def connect(nodes, subject):
        if not check(bool(nodes) and all(node in groups for node in nodes),
                     "connection", subject, "valid source IDs", nodes):
            return
        first = representative(nodes[0])
        for node in nodes[1:]:
            groups[representative(node)] = first

    connected_traces_by_port = Counter()
    for trace_id, trace in traces.items():
        port_ids = trace.get("connected_source_port_ids", [])
        net_ids = trace.get("connected_source_net_ids", [])
        connect(port_ids + net_ids, trace_id)
        if not check(len(port_ids) == len(net_ids) == 1, "schematic", trace_id,
                     "one pin and one named net", [port_ids, net_ids]):
            continue
        port = drawn_ports.get(port_ids[0], {})
        connected_traces_by_port[port_ids[0]] += 1
        label = labels.get(trace_id, {})
        wire = drawn_traces.get(trace_id, {})
        edges = wire.get("edges", [])
        net = nets.get(net_ids[0], {}).get("name")
        expected_label = NET_ALIASES.get(net, net)
        valid = (len(edges) == 1 and edges[0]["from"] == port.get("center")
                 and edges[0]["to"] == label.get("anchor_position")
                 and edges[0]["to"] != edges[0]["from"]
                 and label.get("source_net_id") == net_ids[0]
                 and label.get("text") == expected_label)
        check(valid, "schematic", trace_id, f"real wire from pin to label {expected_label!r}")

    for tie in by_type["source_component_internal_connection"]:
        connect(tie["source_port_ids"], tie["source_component_internal_connection_id"])
    members = Counter(representative(port_id) for port_id in ports)
    group_nets = defaultdict(set)
    actual_names = []
    for net_id, net in nets.items():
        name = NET_ALIASES.get(net["name"], net["name"])
        actual_names.append(name)
        group_nets[representative(net_id)].add(name)
    expected_names = {net for part in expected.values() for net in part["pins"].values() if net}
    check(Counter(actual_names) == Counter(expected_names), "net_names", "board",
          sorted(expected_names), sorted(actual_names))
    check(set(traces) == set(drawn_traces) == set(labels), "schematic", "wire/label coverage",
          len(traces), [len(drawn_traces), len(labels)])

    with bom_path.open(encoding="utf-8-sig", newline="") as handle:
        bom_rows = list(csv.DictReader(handle))
    bom = {row["Reference"]: row for row in bom_rows}
    check(len(bom_rows) == len(bom) == 52 and bom.keys() == expected.keys(),
          "bom", "references", sorted(expected), sorted(bom))

    pin_total = pin_matched = nc_count = supplier_matched = footprint_matched = 0
    components_with_codes = 0
    missing_catalog_codes = []
    critical_results = {}
    net_pin_counts = Counter()
    for ref, part in expected.items():
        actual = components.get(ref, {})
        expected_codes = [part["lcsc"]] if part["lcsc"] else []
        actual_codes = actual.get("supplier_part_numbers", {}).get("jlcpcb", [])
        components_with_codes += bool(expected_codes)
        if not expected_codes:
            missing_catalog_codes.append({"ref": ref, "mpn": part["mpn"],
                                          "reason": "Exact MPN has no confirmed LCSC code"})
        supplier_matched += check(expected_codes == actual_codes, "lcsc", ref,
                                  expected_codes, actual_codes)
        check(actual.get("manufacturer_part_number") == part["mpn"], "mpn", ref,
              part["mpn"], actual.get("manufacturer_part_number"))

        actual_pins = foot_pins[ref]
        check(actual_pins.keys() == part["pins"].keys(), "pin_set", ref,
              sorted(part["pins"]), sorted(actual_pins))
        pad_numbers = actual.get("footprint_pad_numbers", [])
        footprint_ok = (actual.get("footprint_name") == part["footprint"]
                        and Counter(map(str, pad_numbers)) == Counter(part["pins"].keys())
                        and bool(actual.get("footprint_binding")))
        footprint_matched += check(footprint_ok, "footprint", ref,
                                   [part["footprint"], sorted(part["pins"])],
                                   [actual.get("footprint_name"), pad_numbers])
        check(actual.get("do_not_place") == part["dnp"], "population", ref,
              part["dnp"], actual.get("do_not_place"))

        row = bom.get(ref, {})
        bom_identity = [part["mpn"], part["lcsc"] or "", part["footprint"],
                        "yes" if part["dnp"] else "no"]
        bom_actual = [row.get(key) for key in ("MPN", "LCSC", "Footprint", "DNP")]
        check(bom_actual == bom_identity, "bom", ref, bom_identity, bom_actual)
        if ref in CRITICAL:
            expected_mpn, expected_code = CRITICAL[ref]
            observed = (actual.get("manufacturer_part_number"), actual_codes)
            ok = check(observed == (expected_mpn, [expected_code]), "critical_identity", ref,
                       [expected_mpn, expected_code], observed)
            critical_results[ref] = {"status": "PASS" if ok else "FAIL",
                                     "mpn": expected_mpn, "lcsc": expected_code}

        for pin, net in part["pins"].items():
            pin_total += 1
            port_id = actual_pins.get(pin)
            if net is None:
                nc_count += 1
            else:
                net_pin_counts[net] += 1
            if not check(port_id is not None, "pin", f"{ref}.{pin}", net, "missing"):
                continue
            group = representative(port_id)
            named_nets = sorted(group_nets[group])
            expected_set = [net] if net else []
            pin_matched += check(named_nets == expected_set, "pin_net", f"{ref}.{pin}",
                                 expected_set, named_nets)
            if net is None:
                check(members[group] == 1 and ports[port_id].get("do_not_connect") is True,
                      "nc", f"{ref}.{pin}", "independent pin with explicit NC")
            check(connected_traces_by_port[port_id] == int(net is not None),
                  "connection", f"{ref}.{pin}: trace count",
                  int(net is not None), connected_traces_by_port[port_id])
            check(port_id in drawn_ports, "schematic", f"{ref}.{pin}")

    shunt = components.get("R_PUMP_SHUNT", {})
    check(shunt.get("resistance") == 0.02 and "2W" in bom.get("R_PUMP_SHUNT", {}).get("Value", ""),
          "shunt", "R_PUMP_SHUNT", "20 mΩ / 2 W")
    check(len(ports) == pin_total, "pin_count", "board", pin_total, len(ports))
    svg_text = (EXPORTS / "schematic.svg").read_text(encoding="utf-8")
    check('class="schematic-sheet"' not in svg_text and "schematic_sheet_0_outer" not in svg_text
          and "schematic_sheet_0_inner" not in svg_text,
          "geometry", "borderless SVG without schematic_sheet border")
    kicad_parts = {ref: {**part, "placement": "board", "in_bom": True}
                   for ref, part in expected.items()}
    kicad = verify_kicad(EXPORTS / "kicad", kicad_parts, check)
    return {
        "status": "PASS" if not findings else "FAIL",
        "scope": "Electrical topology, parts and schematic pin-to-label geometry",
        "contract_id": reference["contract_id"],
        "artifacts": {key: source_record(path) for key, path in
                      (("reference", reference_path), ("circuit", circuit_path), ("bom", bom_path))},
        "inputs": reference["inputs"],
        "components": {"expected": 52, "observed": len(components), "missing": missing, "extra": extra,
                       "supplier_fields_matching": supplier_matched,
                       "with_confirmed_lcsc_code": components_with_codes,
                       "unresolved_lcsc_codes": missing_catalog_codes,
                       "footprint_names_and_pad_sets_matching": footprint_matched},
        "pin_net_comparison": {"total": pin_total, "equal": pin_matched,
                               "percent": round(100 * pin_matched / pin_total, 6),
                               "connected": pin_total - nc_count, "nc": nc_count,
                               "distinct_named_nets": len(expected_names),
                               "net_pin_counts": dict(sorted(net_pin_counts.items())),
                               "normalized_circuit_names": NET_ALIASES},
        "critical_parts": critical_results,
        "kicad_readback": kicad,
        "can_harness_pinout": reference["can_harness_pinout"],
        "mcu_mapping": reference["mcu_mapping"],
        "prototype_scope": {"underlying_hardware": reference["hardware_scope"],
                            "pcb_layout_and_manufacturing_drc": "not evaluated"},
        "render_warnings": dict(sorted((kind, len(items)) for kind, items in by_type.items()
                                        if "warning" in kind)),
        "findings": findings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--circuit", type=Path, default=EXPORTS / "circuit.json")
    parser.add_argument("--bom", type=Path, default=EXPORTS / "bom.csv")
    parser.add_argument("--report", type=Path, default=EXPORTS / "netlist_verify_report.json")
    args = parser.parse_args()
    result = verify(args.reference.resolve(), args.circuit.resolve(), args.bom.resolve())
    write_json(args.report, result)
    pins = result["pin_net_comparison"]
    print(f"{result['status']}: {result['components']['observed']} components, "
          f"{pins['equal']}/{pins['total']} pins, {pins['distinct_named_nets']} nets, "
          f"{len(result['findings'])} discrepancies.")
    raise SystemExit(result["status"] != "PASS")
