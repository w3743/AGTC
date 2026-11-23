"""Independently read back KiCad symbol pins and their named-net stubs."""

import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\(|\)|[^\s()]+')
TOLERANCE_MM = 0.03


def parse_sexpr(path):
    tokens = TOKEN.findall(Path(path).read_text(encoding="utf-8"))
    stack = []
    root = None
    for token in tokens:
        if token == "(":
            node = []
            if stack:
                stack[-1].append(node)
            else:
                if root is not None:
                    raise ValueError(f"{path}: multiple roots")
                root = node
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError(f"{path}: unexpected closing parenthesis")
            stack.pop()
        else:
            if not stack:
                raise ValueError(f"{path}: token outside root")
            stack[-1].append(json.loads(token) if token.startswith('"') else token)
    if stack or root is None:
        raise ValueError(f"{path}: incomplete expression")
    return root


def children(node, kind):
    return [item for item in node if isinstance(item, list) and item and item[0] == kind]


def one(node, kind):
    matches = children(node, kind)
    if len(matches) != 1:
        raise ValueError(f"expected one {kind}, got {len(matches)}")
    return matches[0]


def point(node):
    return float(node[1]), float(node[2])


def close(left, right):
    return math.dist(left, right) <= TOLERANCE_MM


def on_segment(point_value, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return close(point_value, start)
    fraction = ((point_value[0] - start[0]) * dx + (point_value[1] - start[1]) * dy) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    nearest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return close(point_value, nearest)


def property_value(node, name):
    matches = [prop[2] for prop in children(node, "property") if prop[1] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} property, got {len(matches)}")
    return matches[0]


def symbol_pin_points(symbol, library):
    location = one(symbol, "at")
    origin_x, origin_y = point(location)
    angle = math.radians(float(location[3]) if len(location) > 3 else 0.0)
    if children(symbol, "mirror"):
        raise ValueError("mirrored symbol mapping is not supported")
    result = {}
    for unit in [library, *children(library, "symbol")]:
        for pin in children(unit, "pin"):
            number = one(pin, "number")[1]
            x, y = point(one(pin, "at"))
            x_rotated = x * math.cos(angle) - y * math.sin(angle)
            y_rotated = x * math.sin(angle) + y * math.cos(angle)
            location = (origin_x + x_rotated, origin_y - y_rotated)
            if number in result:
                raise ValueError(f"duplicate library pin {number}")
            result[number] = location
    return result


def connected_labels(pin, wires, labels):
    reachable = [pin]
    reached_segments = []
    changed = True
    while changed:
        changed = False
        for a, b in wires:
            if (not any(on_segment(existing, a, b) for existing in reachable)
                    and not any(on_segment(a, start, end) or on_segment(b, start, end)
                                for start, end in reached_segments)):
                continue
            if (a, b) not in reached_segments:
                reached_segments.append((a, b))
            for endpoint in (a, b):
                if not any(close(endpoint, existing) for existing in reachable):
                    reachable.append(endpoint)
                    changed = True
    names = {name for name, location in labels
             if any(close(location, node) for node in reachable)
             or any(on_segment(location, a, b) for a, b in reached_segments)}
    return names, bool(reached_segments)


def verify_kicad(directory: Path, parts: dict, check: callable) -> dict:
    """Check project links, instance and library pins, NCs, and each pin's electrical labels."""
    directory = Path(directory)
    projects = list(directory.glob("*.kicad_pro"))
    check(len(projects) == 1, "kicad_project", "project_count", 1, len(projects))
    if len(projects) != 1:
        return {"sheets": 0, "instances": 0, "pins_checked": 0}
    root_path = projects[0].with_suffix(".kicad_sch")
    check(root_path.is_file(), "kicad_project", "root_schematic", str(root_path), root_path.is_file())
    if not root_path.is_file():
        return {"sheets": 0, "instances": 0, "pins_checked": 0}
    try:
        json.loads(projects[0].read_text(encoding="utf-8"))
        root = parse_sexpr(root_path)
        project_name = projects[0].stem
        root_uuid = one(root, "uuid")[1]
        pending = [root_path]
        parsed = {root_path: root}
        instance_paths = {root_path: "/" + root_uuid}
        while pending:
            path = pending.pop()
            for sheet in children(parsed[path], "sheet"):
                child_name = property_value(sheet, "Sheetfile")
                project = one(one(sheet, "instances"), "project")
                sheet_path = one(project, "path")[1]
                check(project[1] == project_name and sheet_path == instance_paths[path],
                      "kicad_sheet_instance", child_name,
                      {"project": project_name, "path": instance_paths[path]},
                      {"project": project[1], "path": sheet_path})
                child_path = (path.parent / child_name).resolve()
                check(child_path.is_relative_to(directory.resolve()) and child_path.is_file(),
                      "kicad_project", f"{path.name}:{child_name}", "existing local schematic", str(child_path))
                if child_path.is_relative_to(directory.resolve()) and child_path.is_file():
                    if child_path in parsed:
                        check(False, "kicad_project", f"duplicate_sheet:{child_name}")
                    else:
                        parsed[child_path] = parse_sexpr(child_path)
                        instance_paths[child_path] = instance_paths[path] + "/" + one(sheet, "uuid")[1]
                        pending.append(child_path)
        seen = Counter()
        pin_count = 0
        for path, sheet in parsed.items():
            check(sheet[0] == "kicad_sch", "kicad_project", f"format:{path.name}", "kicad_sch", sheet[0])
            libraries = {item[1]: item for item in children(one(sheet, "lib_symbols"), "symbol")}
            wires = []
            for wire in children(sheet, "wire"):
                points = [point(node) for node in children(one(wire, "pts"), "xy")]
                wires.extend(zip(points, points[1:]))
            labels = [(node[1], point(one(node, "at"))) for kind in ("global_label", "label")
                      for node in children(sheet, kind)]
            no_connects = [point(one(node, "at")) for node in children(sheet, "no_connect")]
            for instance in children(sheet, "symbol"):
                ref = property_value(instance, "Reference")
                seen[ref] += 1
                check(ref in parts, "kicad_component", ref, "contract reference", "unknown" if ref not in parts else ref)
                if ref not in parts:
                    continue
                part = parts[ref]
                expected_flags = {
                    "on_board": part["placement"] == "board",
                    "dnp": bool(part["dnp"]),
                    "in_bom": part.get("in_bom") is not False and part["placement"] == "board",
                }
                for key, expected_flag in expected_flags.items():
                    actual_flag = one(instance, key)[1]
                    check(actual_flag == ("yes" if expected_flag else "no"),
                          "kicad_assembly", f"{ref}.{key}", expected_flag, actual_flag)
                check(property_value(instance, "Footprint") == (part.get("footprint") or ""),
                      "kicad_footprint", ref, part.get("footprint") or "",
                      property_value(instance, "Footprint"))
                project = one(one(instance, "instances"), "project")
                symbol_path = one(project, "path")
                check(project[1] == project_name and symbol_path[1] == instance_paths[path]
                      and one(symbol_path, "reference")[1] == ref,
                      "kicad_symbol_instance", ref,
                      {"project": project_name, "path": instance_paths[path], "reference": ref},
                      {"project": project[1], "path": symbol_path[1],
                       "reference": one(symbol_path, "reference")[1]})
                library_name = one(instance, "lib_id")[1]
                check(library_name in libraries, "kicad_library", ref, "embedded library symbol", library_name)
                if library_name not in libraries:
                    continue
                expected = parts[ref]["pins"]
                try:
                    actual_pins = symbol_pin_points(instance, libraries[library_name])
                except ValueError as exc:
                    check(False, "kicad_pin", ref, "unique mapped library pins", str(exc))
                    continue
                instance_pins = [pin[1] for pin in children(instance, "pin")]
                check(Counter(instance_pins) == Counter(expected.keys()), "kicad_instance_pins", ref,
                      sorted(expected), sorted(instance_pins))
                check(set(actual_pins) == set(expected), "kicad_library_pins", ref,
                      sorted(expected), sorted(actual_pins))
                for pin, expected_net in expected.items():
                    if pin not in actual_pins:
                        continue
                    pin_count += 1
                    position = actual_pins[pin]
                    nets, wired = connected_labels(position, wires, labels)
                    marked_nc = any(close(position, nc) for nc in no_connects)
                    if expected_net is None:
                        check(marked_nc and not nets and not wired, "kicad_pin_net", f"{ref}.{pin}", "NC",
                              {"net_labels": sorted(nets), "no_connect": marked_nc, "wired": wired})
                    else:
                        check(nets == {expected_net} and not marked_nc and wired,
                              "kicad_pin_net", f"{ref}.{pin}", expected_net,
                              {"net_labels": sorted(nets), "no_connect": marked_nc, "wired": wired})
        check(set(seen) == set(parts) and all(count == 1 for count in seen.values()),
              "kicad_components", "references",
              {"count": len(parts)}, {"count": len(seen), "missing": sorted(set(parts) - set(seen)),
                                      "extra": sorted(set(seen) - set(parts)),
                                      "duplicates": sorted(ref for ref, count in seen.items() if count != 1)})
        return {"sheets": len(parsed), "instances": sum(seen.values()), "pins_checked": pin_count}
    except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
        check(False, "kicad_parse", str(directory), "valid project and schematics", str(exc))
        return {"sheets": 0, "instances": 0, "pins_checked": 0}
