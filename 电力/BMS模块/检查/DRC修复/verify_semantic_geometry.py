"""Infer per-pin schematic nets from native LINE geometry and netports.

This read-only verifier deliberately does not use API wire.getState_Net().  It
folds the current .esch2 append log, builds physical connected clusters from
the active LINE records, and labels those clusters from native NET attributes
and active netport pin coordinates/properties.

Example:
  python verify_semantic_geometry.py --snapshot current-snapshot.json \
      --ports ports.json --manifest ../当前施工输入.json --output result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE = MODULE_ROOT / "sch" / "Schematic1" / "P1.esch2"
EPS = 1e-7
NC_TOKENS = {"", "NC", "N/C", "NONE", "NO_CONNECT"}


def point(x, y):
    return (round(float(x), 7), round(float(y), 7))


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def point_on_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    px, py = p[0] - a[0], p[1] - a[1]
    if abs(dx) <= EPS and abs(dy) <= EPS:
        return abs(px) <= EPS and abs(py) <= EPS
    if abs(cross((dx, dy), (px, py))) > EPS * max(1.0, math.hypot(dx, dy)):
        return False
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def endpoint_contacts(a, b, c, d):
    """Return contacts where a segment endpoint lands on the other segment.

    Interior/interior crossings are not conductive unless represented by an
    explicit native JUNCTION record.
    """
    contacts = set()
    for candidate in (a, b):
        if point_on_segment(candidate, c, d):
            contacts.add(candidate)
    for candidate in (c, d):
        if point_on_segment(candidate, a, b):
            contacts.add(candidate)
    return contacts


def strict_interior_intersection(a, b, c, d):
    """Return a crossing point only when it lies inside both segments."""
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    qmp = (c[0] - a[0], c[1] - a[1])
    denom = cross(r, s)
    if abs(denom) <= EPS:
        return None
    t = cross(qmp, s) / denom
    u = cross(qmp, r) / denom
    if EPS < t < 1 - EPS and EPS < u < 1 - EPS:
        return point(a[0] + t * r[0], a[1] + t * r[1])
    return None


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            prior = self.parent[item]
            self.parent[item] = root
            item = prior
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            # Deterministic roots make the output stable across runs.
            if a <= b:
                self.parent[b] = a
            else:
                self.parent[a] = b


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rows(document, key, source_name):
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get(key), list):
        return document[key]
    raise ValueError(f"{source_name} must be a list or an object containing a {key!r} list")


def fold_native(path):
    latest = {}
    malformed = []
    junction_points = []
    raw_records = 0
    tombstones = 0
    replacements = 0

    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        raw_records += 1
        try:
            header_text, payload_text = raw_line.split("||", 1)
            header = json.loads(header_text)
            payload_text = payload_text.rstrip("|")
            payload = json.loads(payload_text) if payload_text.strip() else None
        except (ValueError, json.JSONDecodeError) as exc:
            malformed.append({"line": line_number, "error": str(exc)})
            continue

        kind, record_id = header.get("type"), header.get("id")
        if kind not in {"LINE", "COMPONENT", "ATTR", "JUNCTION"}:
            continue
        if kind == "ATTR" and payload is not None and payload.get("key") != "NET":
            continue
        if not record_id:
            malformed.append({"line": line_number, "error": "relevant record missing id", "type": kind})
            continue

        key = (kind, str(record_id))
        if payload is None:
            tombstones += 1
            latest.pop(key, None)
        else:
            if key in latest:
                replacements += 1
            latest[key] = {"payload": payload, "line": line_number}

    lines = []
    components = {}
    group_names = defaultdict(set)
    for (kind, record_id), row in latest.items():
        if kind != "JUNCTION":
            continue
        payload = row["payload"]
        try:
            if payload.get("x") is not None and payload.get("y") is not None:
                marker_point = point(payload["x"], payload["y"])
            elif payload.get("centerX") is not None and payload.get("centerY") is not None:
                marker_point = point(payload["centerX"], payload["centerY"])
            else:
                raise ValueError("JUNCTION missing x/y or centerX/centerY")
            junction_points.append({"id": record_id, "point": marker_point, "sourceLine": row["line"]})
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            malformed.append({"line": row["line"], "id": record_id, "type": "JUNCTION", "error": str(exc)})

    for (kind, record_id), row in latest.items():
        payload = row["payload"]
        if kind == "LINE":
            required = ("startX", "startY", "endX", "endY")
            try:
                if any(payload.get(field) is None for field in required):
                    raise ValueError("LINE missing endpoint coordinate")
                start = point(payload["startX"], payload["startY"])
                end = point(payload["endX"], payload["endY"])
                lines.append({
                    "id": record_id,
                    "start": start,
                    "end": end,
                    "lineGroup": str(payload.get("lineGroup") or ""),
                    "sourceLine": row["line"],
                })
            except (TypeError, ValueError, OverflowError) as exc:
                malformed.append({"line": row["line"], "id": record_id, "error": str(exc)})
        elif kind == "COMPONENT":
            components[record_id] = payload
        elif kind == "ATTR":
            group = payload.get("parentId")
            name = payload.get("value")
            if group and isinstance(name, str) and name.strip():
                group_names[str(group)].add(name.strip())

    return lines, components, group_names, {
        "raw_records": raw_records,
        "folded_relevant_records": len(latest),
        "active_LINE": len(lines),
        "active_COMPONENT": len(components),
        "active_JUNCTION": len(junction_points),
        "net_ATTR_groups": len(group_names),
        "relevant_tombstones": tombstones,
        "relevant_replacements": replacements,
        "malformed": malformed,
    }, junction_points


def normalize_expected_net(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in NC_TOKENS:
        return None
    return text


def get_pin_number(pin):
    number = pin.get("number")
    if number is None:
        raise ValueError("pin missing number")
    return str(number)


def collect_semantics(snapshot, ports_doc, manifest, native_path, source_paths):
    native_lines, native_components, line_group_names, fold_info, junction_points = fold_native(native_path)
    snapshot_rows = load_rows(snapshot, "rows", "snapshot")
    port_rows = load_rows(ports_doc, "ports", "ports source")
    manifest_rows = load_rows(manifest, "rows", "manifest")

    defects = {
        "malformedNativeRecords": list(fold_info["malformed"]),
        "missingNativeComponents": [],
        "unrepresentedNativeComponents": [],
        "stalePortRows": [],
        "missingPortRows": [],
        "portsMissingNET": [],
        "malformedPins": [],
        "duplicateRefs": [],
        "duplicatePins": [],
        "missingExpectedComponents": [],
        "unmanifestedSnapshotComponents": [],
        "missingExpectedPins": [],
        "unexpectedSnapshotPins": [],
        "snapshotNCFlagMismatches": [],
        "floatingExpectedPins": [],
        "pinNetMismatches": [],
        "ncContact": [],
        "lineGroupNameConflicts": [],
        "clusterNameConflicts": [],
        "danglingNetports": [],
    }

    rows_by_ref = {}
    row_ids = set()
    component_pin_records = []
    for row in snapshot_rows:
        ref = str(row.get("ref") or "")
        if not ref:
            defects["malformedPins"].append({"source": "snapshot", "error": "row missing ref"})
            continue
        if ref in rows_by_ref:
            defects["duplicateRefs"].append({"source": "snapshot", "ref": ref})
        rows_by_ref[ref] = row
        if row.get("id"):
            row_ids.add(str(row["id"]))
            if str(row["id"]) not in native_components:
                defects["missingNativeComponents"].append({"ref": ref, "id": str(row["id"])})

        seen_numbers = set()
        for pin in row.get("pins", []):
            try:
                number = get_pin_number(pin)
                p = point(pin["x"], pin["y"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                defects["malformedPins"].append({"source": "snapshot", "ref": ref, "error": str(exc)})
                continue
            if number in seen_numbers:
                defects["duplicatePins"].append({"source": "snapshot", "ref": ref, "pin": number})
            seen_numbers.add(number)
            component_pin_records.append({
                "ref": ref,
                "pin": number,
                "point": p,
                "snapshotNC": bool(pin.get("nc", False)),
            })

    manifest_by_ref = {}
    for row in manifest_rows:
        ref = str(row.get("ref") or "")
        if not ref:
            defects["malformedPins"].append({"source": "manifest", "error": "row missing ref"})
            continue
        if ref in manifest_by_ref:
            defects["duplicateRefs"].append({"source": "manifest", "ref": ref})
        manifest_by_ref[ref] = row

    for ref in sorted(set(manifest_by_ref) - set(rows_by_ref)):
        defects["missingExpectedComponents"].append({"ref": ref})
    for ref in sorted(set(rows_by_ref) - set(manifest_by_ref)):
        defects["unmanifestedSnapshotComponents"].append({"ref": ref})

    expected = {}
    actual_pin_keys = {(r["ref"], r["pin"]) for r in component_pin_records}
    for ref, manifest_row in manifest_by_ref.items():
        manifest_pins = manifest_row.get("pins")
        if not isinstance(manifest_pins, dict):
            defects["malformedPins"].append({"source": "manifest", "ref": ref, "error": "pins is not an object"})
            continue
        for number_value, raw_net in manifest_pins.items():
            number = str(number_value)
            expected[(ref, number)] = normalize_expected_net(raw_net)
            if (ref, number) not in actual_pin_keys:
                defects["missingExpectedPins"].append({"ref": ref, "pin": number, "expected": normalize_expected_net(raw_net)})
        for record in component_pin_records:
            if record["ref"] == ref and record["pin"] not in {str(n) for n in manifest_pins}:
                defects["unexpectedSnapshotPins"].append({"ref": ref, "pin": record["pin"]})

    active_port_rows = []
    port_ids = set()
    port_pin_records = []
    for port in port_rows:
        port_id = str(port.get("id") or "")
        if not port_id:
            defects["malformedPins"].append({"source": "ports", "error": "port missing id"})
            continue
        if port_id in port_ids:
            defects["duplicateRefs"].append({"source": "ports", "id": port_id})
        port_ids.add(port_id)
        if port_id not in native_components:
            defects["stalePortRows"].append({"id": port_id, "name": port.get("name"), "net": port.get("net")})
            continue
        active_port_rows.append(port)
        net = port.get("net")
        net = str(net).strip() if net is not None else ""
        if not net:
            defects["portsMissingNET"].append({"id": port_id, "name": port.get("name")})
        pins = port.get("pins")
        if not isinstance(pins, list) or not pins:
            defects["malformedPins"].append({"source": "ports", "id": port_id, "error": "port has no pin coordinates"})
            continue
        for pin in pins:
            try:
                number = get_pin_number(pin)
                p = point(pin["x"], pin["y"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                defects["malformedPins"].append({"source": "ports", "id": port_id, "error": str(exc)})
                continue
            port_pin_records.append({
                "id": port_id,
                "name": port.get("name"),
                "net": net,
                "pin": number,
                "point": p,
            })

    active_port_ids = {str(p["id"]) for p in active_port_rows}
    for component_id in sorted(set(native_components) - row_ids - active_port_ids):
        defects["unrepresentedNativeComponents"].append({"id": component_id})
    for component_id in sorted(set(native_components) - row_ids - port_ids):
        defects["missingPortRows"].append({"id": component_id})

    # Connect native segments only at actual topology points: endpoints,
    # endpoint-to-segment T contacts, pin/netport coordinates, or explicit
    # JUNCTION markers. Pure interior/interior crossings remain separate.
    segment_points = [{line["start"], line["end"]} for line in native_lines]
    actual_vertices = {r["point"] for r in component_pin_records + port_pin_records}
    actual_vertices.update(marker["point"] for marker in junction_points)
    for index, line in enumerate(native_lines):
        for feature in component_pin_records:
            if point_on_segment(feature["point"], line["start"], line["end"]):
                segment_points[index].add(feature["point"])
        for feature in port_pin_records:
            if point_on_segment(feature["point"], line["start"], line["end"]):
                segment_points[index].add(feature["point"])
        for marker in junction_points:
            if point_on_segment(marker["point"], line["start"], line["end"]):
                segment_points[index].add(marker["point"])

    ignored_interior_crossings = []
    for left_index, left in enumerate(native_lines):
        for right_index in range(left_index + 1, len(native_lines)):
            right = native_lines[right_index]
            contacts = endpoint_contacts(left["start"], left["end"], right["start"], right["end"])
            for shared in contacts:
                segment_points[left_index].add(shared)
                segment_points[right_index].add(shared)
            if not contacts:
                crossing = strict_interior_intersection(
                    left["start"], left["end"], right["start"], right["end"]
                )
                if crossing is not None and crossing not in actual_vertices:
                    ignored_interior_crossings.append({
                        "point": crossing,
                        "leftLine": left["id"],
                        "leftWireGroup": left["lineGroup"],
                        "leftNames": sorted(line_group_names.get(left["lineGroup"], set())),
                        "rightLine": right["id"],
                        "rightWireGroup": right["lineGroup"],
                        "rightNames": sorted(line_group_names.get(right["lineGroup"], set())),
                        "conductive": False,
                        "reason": "interior/interior crossing with no endpoint, pin, port, or JUNCTION at the crossing",
                    })

    dsu = DisjointSet()
    for line in native_lines:
        dsu.add(line["start"])
        dsu.add(line["end"])
    for record in component_pin_records + port_pin_records:
        dsu.add(record["point"])
    for line, points_on_line in zip(native_lines, segment_points):
        ordered = sorted(
            points_on_line,
            key=lambda p: (p[0] - line["start"][0]) * (line["end"][0] - line["start"][0])
            + (p[1] - line["start"][1]) * (line["end"][1] - line["start"][1]),
        )
        if ordered:
            for shared in ordered[1:]:
                dsu.union(ordered[0], shared)

    cluster_labels = defaultdict(set)
    cluster_label_sources = defaultdict(list)
    cluster_segments = defaultdict(set)
    cluster_pins = defaultdict(list)
    cluster_ports = defaultdict(list)
    line_group_conflicts = []

    for group, names in sorted(line_group_names.items()):
        if len(names) > 1:
            line_group_conflicts.append({"lineGroup": group, "names": sorted(names)})
    defects["lineGroupNameConflicts"].extend(line_group_conflicts)

    for index, (line, points_on_line) in enumerate(zip(native_lines, segment_points)):
        if not points_on_line:
            continue
        root = dsu.find(next(iter(points_on_line)))
        cluster_segments[root].add(index)
        for name in line_group_names.get(line["lineGroup"], set()):
            cluster_labels[root].add(name)
            cluster_label_sources[root].append({"source": "native_LINE_NET_ATTR", "name": name, "lineGroup": line["lineGroup"]})

    for record in component_pin_records:
        root = dsu.find(record["point"])
        cluster_pins[root].append(record)
    for record in port_pin_records:
        root = dsu.find(record["point"])
        cluster_ports[root].append(record)
        if record["net"]:
            cluster_labels[root].add(record["net"])
            cluster_label_sources[root].append({
                "source": "netport_NET",
                "name": record["net"],
                "portId": record["id"],
                "pin": record["pin"],
            })

    roots = set(cluster_labels) | set(cluster_segments) | set(cluster_pins) | set(cluster_ports)
    root_ids = {root: f"C{index:05d}" for index, root in enumerate(sorted(roots), 1)}
    for root in sorted(roots):
        names = sorted(cluster_labels.get(root, set()))
        if len(names) > 1:
            defects["clusterNameConflicts"].append({
                "cluster": root_ids[root],
                "point": root,
                "names": names,
                "componentPins": [f"{p['ref']}.{p['pin']}" for p in cluster_pins.get(root, [])],
                "netports": sorted({f"{p['name']}[{p['id']}]" for p in cluster_ports.get(root, [])}),
            })
        ports_here = cluster_ports.get(root, [])
        if ports_here and not cluster_segments.get(root) and not cluster_pins.get(root):
            defects["danglingNetports"].append({
                "cluster": root_ids[root],
                "point": root,
                "netports": [{"id": p["id"], "name": p["name"], "net": p["net"]} for p in ports_here],
            })

    pin_results = []
    for record in component_pin_records:
        key = (record["ref"], record["pin"])
        if key not in expected:
            continue
        root = dsu.find(record["point"])
        expected_net = expected[key]
        labels = sorted(cluster_labels.get(root, set()))
        all_cluster_pins = cluster_pins.get(root, [])
        all_cluster_ports = cluster_ports.get(root, [])
        segment_count = len(cluster_segments.get(root, set()))
        touched = bool(segment_count or len(all_cluster_pins) > 1 or all_cluster_ports)
        result = {
            "ref": record["ref"],
            "pin": record["pin"],
            "point": record["point"],
            "expected": expected_net,
            "actualNames": labels,
            "cluster": root_ids[root],
            "lineSegments": segment_count,
            "clusterPins": sorted({f"{p['ref']}.{p['pin']}" for p in all_cluster_pins}),
            "clusterNetports": sorted({f"{p['name']}[{p['id']}]" for p in all_cluster_ports}),
            "labelSources": cluster_label_sources.get(root, []),
            "snapshotNC": record["snapshotNC"],
        }

        if (expected_net is None) != record["snapshotNC"]:
            defects["snapshotNCFlagMismatches"].append({
                "ref": record["ref"], "pin": record["pin"],
                "expectedNC": expected_net is None, "snapshotNC": record["snapshotNC"],
            })
        if expected_net is None:
            if touched or labels:
                defects["ncContact"].append({**result, "reason": "NC pin shares a physical cluster with wiring, a pin, or a netport"})
            result["status"] = "PASS" if not touched and not labels else "FAIL"
        else:
            if not labels:
                defects["floatingExpectedPins"].append({**result, "reason": "no physical-cluster NET name"})
            if labels and expected_net not in labels:
                defects["pinNetMismatches"].append({**result, "reason": "expected network absent from physical-cluster labels"})
            result["status"] = "PASS" if labels == [expected_net] else "FAIL"
        pin_results.append(result)

    defect_names = [name for name, entries in defects.items() if entries]
    status = "FAIL" if defect_names else "PASS"
    source_paths = [Path(native_path), *(Path(p) for p in source_paths)]
    hashes = {}
    for source in source_paths:
        try:
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            hashes[str(source)] = None

    # Compact cluster inventory helps explain the evidence without emitting all
    # intermediate graph nodes.
    clusters = []
    for root in sorted(roots):
        clusters.append({
            "cluster": root_ids[root],
            "representativePoint": root,
            "names": sorted(cluster_labels.get(root, set())),
            "lineSegments": len(cluster_segments.get(root, set())),
            "componentPins": sorted({f"{p['ref']}.{p['pin']}" for p in cluster_pins.get(root, [])}),
            "netports": sorted({f"{p['name']}[{p['id']}]" for p in cluster_ports.get(root, [])}),
        })

    result = {
        "status": status,
        "method": "Fold the latest native LINE/COMPONENT/ATTR/JUNCTION records by (type,id). Connect shared endpoints, endpoints landing on a segment, pin/netport coordinates on segments, and explicit JUNCTION marker coordinates. Pure interior/interior crossings without a JUNCTION remain disconnected and are listed as ignoredInteriorCrossings. Infer cluster names from native ATTR key=NET and active netport net properties; API wire.getState_Net is intentionally not used. Repeated identical names in one cluster are allowed; multiple distinct names fail.",
        "strict_semantic_geometry": True,
        "sourceSHA256": hashes,
        "counts": {
            "snapshotComponents": len(snapshot_rows),
            "manifestComponents": len(manifest_by_ref),
            "snapshotPins": len(component_pin_records),
            "manifestPins": len(expected),
            "activeNativeLines": len(native_lines),
            "activeNativeJunctionMarkers": len(junction_points),
            "ignoredInteriorCrossings": len(ignored_interior_crossings),
            "portRowsSource": len(port_rows),
            "activeNetportRows": len(active_port_rows),
            "netportPins": len(port_pin_records),
            "physicalClusters": len(roots),
            "pinsChecked": len(pin_results),
        },
        "nativeRecordSummary": fold_info,
        "pinResults": pin_results,
        "clusters": clusters,
        "ignoredInteriorCrossings": ignored_interior_crossings,
        "defectCounts": {name: len(entries) for name, entries in defects.items()},
        "defects": defects,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Official reopened schematic snapshot JSON")
    parser.add_argument("--ports", required=True, help="Official port snapshot JSON or JSON file containing a ports array")
    parser.add_argument("--manifest", required=True, help="Current expected pin-net manifest JSON")
    parser.add_argument("--output", required=True, help="Only result file written by this script")
    parser.add_argument("--native", default=str(DEFAULT_NATIVE), help="Current native .esch2 path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = {
        "snapshot": Path(args.snapshot),
        "ports": Path(args.ports),
        "manifest": Path(args.manifest),
    }
    try:
        for name, path in inputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
        native_path = Path(args.native)
        if not native_path.is_file():
            raise FileNotFoundError(f"native schematic not found: {native_path}")

        snapshot = read_json(inputs["snapshot"])
        ports_doc = read_json(inputs["ports"])
        manifest = read_json(inputs["manifest"])
        result = collect_semantics(
            snapshot,
            ports_doc,
            manifest,
            native_path,
            [path.resolve() for path in inputs.values()],
        )
    except Exception as exc:  # A failed read must never look like a PASS.
        result = {
            "status": "FAIL",
            "strict_semantic_geometry": True,
            "fatalErrors": [f"{type(exc).__name__}: {exc}"],
            "sourceSHA256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [*inputs.values(), Path(args.native)]
                if path.is_file()
            },
        }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": result.get("status"),
        "counts": result.get("counts"),
        "defectCounts": result.get("defectCounts"),
        "fatalErrors": result.get("fatalErrors"),
        "output": str(out_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
