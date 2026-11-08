"""Redraw four dense pin fans on an isolated copy and stage official readback."""
import argparse
import base64
import csv
import importlib.util
import io
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location("verify_foc", HERE / "Verify-SavedFoc.py")
eda = importlib.util.module_from_spec(loader)
loader.loader.exec_module(eda)

TARGETS = {"U1": (570, -145), "U3": (1525, -90), "J_MCU_L": (2300, -115), "J_MCU_R": (3100, -115)}
SUPPORT_GROUPS = (
    (-460, ("U1_CP_C", "U1_VCP_C", "U1_VGLS_C", "U1_DVDD_C", "U1_VREF_C")),
    (-570, ("U1_VM_C1", "U1_VM_C2", "U1_VM_C3", "U1_BUCK_CIN1", "U1_BUCK_CIN2")),
    (-680, ("U1_BUCK_CBST", "U1_VCC_C", "U1_BUCK_RT", "U1_BUCK_RCL", "U1_BUCK_D", "U1_BUCK_L")),
    (-790, ("U1_BUCK_COUT1", "U1_BUCK_COUT2", "U1_BUCK_RRIP", "U1_BUCK_RFB1", "U1_BUCK_RFB2")),
)
SUPPORT = {ref: (320 + 585*i, y) for y, refs in SUPPORT_GROUPS for i, ref in enumerate(refs)}
SUPPORT.update({"U3_C1": (1230, -340), "U3_C2": (1830, -340)})
NATIVE_ALLOWED = {eda.NATIVE.name + ".eprj3", "sch/Schematic1/P1.esch2"}


def point(item):
    return round(item["x"], 6), round(item["y"], 6)


def segments(wire):
    line = wire["line"]
    rows = line if line and isinstance(line[0], list) else [line[i:i+4] for i in range(0, len(line), 4)]
    return [(tuple(round(v, 6) for v in row[:2]), tuple(round(v, 6) for v in row[2:])) for row in rows]


def on(point, segment):
    x, y = point
    (a, b), (c, d) = segment
    return (min(a, c) <= x <= max(a, c) and min(b, d) <= y <= max(b, d)
            and (x - a)*(d - b) == (y - b)*(c - a))


def prune(lines, terminals, old_pins):
    vertices = {p for edge in lines for p in edge} | terminals | old_pins
    graph = defaultdict(set)
    for edge in lines:
        along = sorted(p for p in vertices if on(p, edge))
        for a, b in zip(along, along[1:]):
            graph[a].add(b)
            graph[b].add(a)
    if len(terminals) == 1:
        pin = next(iter(terminals))
        nearby = min(graph[pin], key=lambda q: abs(q[0]-pin[0])+abs(q[1]-pin[1]))
        return [list(pin + nearby)]
    leaves = [p for p, neighbors in graph.items() if len(neighbors) <= 1 and p not in terminals]
    while leaves:
        p = leaves.pop()
        for q in graph.pop(p, set()):
            graph[q].remove(p)
            if len(graph[q]) <= 1 and q not in terminals:
                leaves.append(q)
    return [list(p+q) for p in sorted(graph) for q in sorted(graph[p]) if p < q]


def get_plan(geometry, expected, support):
    targets = TARGETS | SUPPORT if support else TARGETS
    components = {part["ref"]: part for part in geometry["components"]}
    pin_table = [(ref, point(pin), expected[ref][pin["pin"]])
                 for ref, part in components.items() for pin in part["pins"]]
    all_lines = {wire["id"]: segments(wire) for wire in geometry["wires"]}
    touched = {}
    for wire, lines in all_lines.items():
        attached = [pin for pin in pin_table if any(on(pin[1], edge) for edge in lines)]
        moving = {p for ref, p, net in attached if ref in targets}
        if moving:
            nets = {net for ref, p, net in attached}
            if len(nets) != 1 or not all(nets):
                raise ValueError((wire, attached))
            remaining = {p for ref, p, net in attached if ref not in targets}
            touched[wire] = (next(iter(nets)), remaining, moving)
    removed_ports = []
    keep_ports = defaultdict(set)
    for port in geometry["ports"]:
        attached = {wire for wire, lines in all_lines.items()
                    if any(on(point(pin), edge) for pin in port["pins"] for edge in lines)}
        if attached and attached <= touched.keys():
            surviving = {wire for wire in attached if touched[wire][1]}
            if surviving:
                for wire in surviving:
                    keep_ports[wire].update(point(pin) for pin in port["pins"])
            else:
                removed_ports.append(port["id"])
    kept_wires, removed_wires = [], []
    for wire, (net, remain, moving) in touched.items():
        if remain:
            terminals = remain | keep_ports[wire]
            line = prune(all_lines[wire], terminals, moving)
            port = None
            if not keep_ports[wire]:
                # A single surviving pin gets a local port at its existing stub end.
                # For a group, retain one old branch leading away from the local pins.
                if len(remain) > 1:
                    retained = {tuple(segment[:2]) for segment in line} | {tuple(segment[2:]) for segment in line}
                    branch = next((list(a+b) for a, b in all_lines[wire] if a in retained and b not in retained), None)
                    branch = branch or next(list(b+a) for a, b in all_lines[wire] if b in retained and a not in retained)
                    line.append(branch)
                a, b, x, y = line[-1]
                angle = (180 if x > a else 0) if y == b else (270 if y > b else 90)
                port = [x, y, angle]
            kept_wires.append([wire, net, line, port])
        else:
            removed_wires.append(wire)
    moves = [[ref, components[ref]["id"], *location, expected[ref],
              ref in SUPPORT and ref != "U1_VM_C3"]
             for ref, location in targets.items()]
    return {"keepWires": kept_wires, "removeWires": removed_wires,
            "removePorts": removed_ports, "moves": moves, "support": support}


def electrical_netlist(net):
    net = json.loads(json.dumps(net))
    net.pop("projectId", None)
    for component in net["components"].values():
        component["props"].pop("Channel ID", None)
    return net


def bom_rows(value):
    value = value.lstrip("\ufeff")
    separator = "\t" if "\t" in value.splitlines()[0] else ","
    return list(csv.reader(io.StringIO(value), delimiter=separator))


def warning_primitives(drc):
    return Counter((row["type"], row["rule"], row.get("net"), primitive["sheet"], primitive["primitiveId"])
                   for row in drc for primitive in row.get("primitives", []))


def nc_markers(geometry):
    return sorted(f"{c['ref']}.{p['pin']}" for c in geometry["components"]
                  for p in c["pins"] if p["noConnect"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", type=Path)
    parser.add_argument("--support", action="store_true", help="Also space the nearby passive networks")
    args = parser.parse_args()
    before = eda.hashes(eda.NATIVE)
    delivery = eda.DELIVERY
    contract = json.loads((eda.ROOT / "原理图准备/当前原理图连接.json").read_text(encoding="utf-8-sig"))
    expected = {c["ref"]: c["pins"] for c in contract["components"]}
    current = json.loads((delivery / "FOC_G431_P1.enet").read_text(encoding="utf-8-sig"))
    read_geometry = (HERE / "Read-SingleSheetActualGeometry.js").read_text(encoding="utf-8-sig")
    with tempfile.TemporaryDirectory(prefix="foc_reflow_") as tmp:
        copy = Path(tmp) / eda.NATIVE.name
        shutil.copytree(eda.NATIVE, copy)
        assert before == eda.hashes(copy)
        project = copy / (copy.name + ".eprj3")
        with eda.headless(project) as (invoke, _):
            baseline = invoke(read_geometry)
            baseline_drc = invoke("return await eda.sch_Drc.check(true, false, true);")
            plan = get_plan(baseline, expected, args.support)
            action = (HERE / "Reflow-DenseP1.js").read_text(encoding="utf-8-sig")
            program = ("const plan = " + json.dumps(plan, ensure_ascii=False, separators=(",", ":")) +
                       ";\nlet stage;\ntry {\n" + action +
                       "\n} catch (error) { return {stage, error: String(error)}; }")
            result = invoke(program)
            if not result or result.get("error"):
                raise RuntimeError(result)
        with eda.headless(project) as (invoke, _):
            drc, net_text, bom = eda.exports(invoke)
            reopened = invoke(read_geometry)
            pdf = invoke((HERE / "Export-CurrentPdf.js").read_text(encoding="utf-8-sig"))
        after = eda.hashes(copy)
        altered = {name: (copy/name).read_bytes() for name, sha in after.items() if before.get(name) != sha}
    staged = args.staging
    staged.mkdir(parents=True, exist_ok=True)
    net = json.loads(net_text.lstrip("\ufeff"))
    report = {
        "beforeSha256": before, "reopenedSha256": after, "edits": result,
        "workspaceSourceUnchanged": before == eda.hashes(eda.NATIVE),
        "pinDifferences": eda.compare(eda.pins(current), eda.pins(net)),
        "electricalNetlistMatches": electrical_netlist(current) == electrical_netlist(net),
        "bomRowsMatch": bom_rows((delivery / "FOC_G431_BOM.csv").read_text(encoding="utf-8-sig")) == bom_rows(bom),
        "drcPrimitivesMatch": warning_primitives(baseline_drc) == warning_primitives(drc),
        "ncMarkersBefore": nc_markers(baseline), "ncMarkersAfter": nc_markers(reopened),
        "drcCounts": dict(Counter(row["type"] for row in drc)), "drc": drc,
    }
    (staged / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print({key: report[key] for key in ("edits", "pinDifferences", "electricalNetlistMatches",
                                       "bomRowsMatch", "drcPrimitivesMatch", "drcCounts")})
    if (before.keys() != after.keys() or not altered.keys() <= NATIVE_ALLOWED or
            not report["workspaceSourceUnchanged"] or report["pinDifferences"] or
            not report["electricalNetlistMatches"] or not report["bomRowsMatch"] or not report["drcPrimitivesMatch"] or
            report["ncMarkersBefore"] != report["ncMarkersAfter"] or report["drcCounts"].get("error")):
        raise SystemExit(1)
    for name, data in altered.items():
        file = staged / "native" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    pdf_bytes = base64.b64decode(pdf["base64"], validate=True)
    assert pdf_bytes.startswith(b"%PDF-") and len(pdf_bytes) == pdf["bytes"]
    (staged / "FOC_G431_单页原理图.pdf").write_bytes(pdf_bytes)
    (staged / "FOC_G431_P1.enet").write_bytes(net_text.encode("utf-8"))
    (staged / "FOC_G431_BOM.csv").write_bytes(bom.encode("utf-8"))


if __name__ == "__main__":
    main()
