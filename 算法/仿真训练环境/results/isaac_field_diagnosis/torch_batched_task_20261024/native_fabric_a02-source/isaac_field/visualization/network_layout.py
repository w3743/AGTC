"""Stable, topology-only layout and route diagnostics for the 2D policy view.

The rendered cards are HTML/SVG.  Graphviz supplies only their coordinates and
the cubic routes between explicitly assigned side ports; frame values never
participate in layout.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
from typing import Any, Callable, Iterable, Mapping, Sequence


CARD_WIDTH = 170.0
CARD_HEIGHT = 110.0
GRAPH_PADDING = 42.0
_GRAPHVIZ_TIMEOUT_SECONDS = 8.0
_CACHE_SIZE = 4

_SENSOR_LABELS = {
    "lidar": "雷达输入支路",
    "depth": "深度输入支路",
    "detections": "检测输入支路",
    "state": "状态输入支路",
    "feedback": "惯性与驱动反馈支路",
}


def _finite_column(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _normalized_nodes(nodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in nodes:
        if not isinstance(raw, Mapping) or raw.get("key") is None:
            continue
        key = str(raw["key"])
        if key in seen:
            raise ValueError(f"duplicate node key in network layout: {key}")
        seen.add(key)
        parents_raw = raw.get("parents", ())
        # Preserve inspected parent order: it is useful for assigning stable,
        # semantically meaningful input ports (notably the LSTM gates).
        parents = list(dict.fromkeys(str(parent) for parent in parents_raw or ()))
        result.append({
            "key": key,
            "branch": str(raw.get("branch") or ""),
            "column": _finite_column(raw.get("column")),
            "parents": parents,
        })
    return result


def topology_signature(nodes: Iterable[Mapping[str, Any]]) -> str:
    """Return a stable key for all data that can move nodes or routes."""
    normalized = _normalized_nodes(nodes)
    encoded = json.dumps(
        {
            "layout_version": 2,
            "card": [CARD_WIDTH, CARD_HEIGHT],
            "padding": GRAPH_PADDING,
            "nodes": normalized,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _partition(branch: str) -> tuple[str, str]:
    if branch in _SENSOR_LABELS:
        return "sensor-" + branch, _SENSOR_LABELS[branch]
    if branch == "fusion":
        return "fusion", "传感融合"
    if branch in {"actor", "actor_memory", "output"}:
        return "actor", "策略与动作"
    if branch in {"critic", "critic_memory", "value"}:
        return "critic", "独立价值诊断"
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in branch) or "other"
    return "branch-" + safe, branch or "其他支路"


def _quote_dot(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _port_label(incoming: Sequence[str], outgoing: Sequence[str]) -> str:
    rows = max(1, len(incoming), len(outgoing))
    row_height = CARD_HEIGHT / rows
    output: list[str] = [
        '<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="0" CELLSPACING="0" '
        f'FIXEDSIZE="TRUE" WIDTH="{int(CARD_WIDTH)}" HEIGHT="{int(CARD_HEIGHT)}">'
    ]
    for index in range(rows):
        input_cell = (
            f'<TD PORT="in{index}" WIDTH="1" HEIGHT="{row_height:.4f}"></TD>'
            if index < len(incoming)
            else f'<TD WIDTH="1" HEIGHT="{row_height:.4f}"></TD>'
        )
        output_cell = (
            f'<TD PORT="out{index}" WIDTH="1"></TD>'
            if index < len(outgoing)
            else '<TD WIDTH="1"></TD>'
        )
        output.append(f"<TR>{input_cell}<TD WIDTH=\"168\"></TD>{output_cell}</TR>")
    output.append("</TABLE>")
    return "".join(output)


def _dot_source(nodes: list[dict[str, Any]], edges: list[tuple[str, str]],
                port_orders: Mapping[str, Mapping[str, Sequence[str]]] | None = None
                ) -> tuple[str, dict[str, str], dict[tuple[str, str], tuple[str, str]]]:
    node_order = {node["key"]: index for index, node in enumerate(nodes)}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    requested_incoming = {} if port_orders is None else port_orders.get("incoming", {})
    requested_outgoing = {} if port_orders is None else port_orders.get("outgoing", {})
    node_names = {node["key"]: f"n{index}" for index, node in enumerate(nodes)}
    ports: dict[tuple[str, str], tuple[str, str]] = {}

    for node in nodes:
        key = node["key"]
        incoming[key] = list(requested_incoming.get(key, ())) or incoming[key]
        outgoing[key] = list(requested_outgoing.get(key, ())) or outgoing[key]
        if key not in requested_incoming:
            incoming[key].sort(key=lambda source: node_order[source])
        if key not in requested_outgoing:
            outgoing[key].sort(key=lambda target: node_order[target])
        for index, source in enumerate(incoming[key]):
            ports[(source, key)] = (f"out{outgoing[source].index(key)}", f"in{index}")

    lines = [
        "digraph Network {",
        "graph [rankdir=LR, splines=true, concentrate=false, overlap=false, newrank=true, "
        "compound=true, outputorder=edgesfirst, ordering=out, nodesep=0.48, ranksep=1.18, "
        "margin=0.0, pad=0.0, bgcolor=\"transparent\"] ;",
        "node [shape=plain, margin=0, labeljust=c, labelloc=c];",
        "edge [arrowhead=normal, arrowsize=0.72, minlen=1];",
    ]

    grouped: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, str] = {}
    for node in nodes:
        partition, label = _partition(node["branch"])
        grouped[partition].append(node["key"])
        labels[partition] = label

    # Graphviz only treats names with the ``cluster_`` prefix as clusters.
    # Keep each sensor branch in its own band as well as splitting actor and
    # critic, so their partition captions cannot drift across unrelated routes.
    for partition in sorted(grouped):
        group_name = "cluster_" + "".join(char if char.isalnum() or char == "_" else "_" for char in partition)
        lines.append(f"subgraph {group_name} {{")
        lines.append("graph [style=invis, color=\"transparent\", margin=4];")
        for key in grouped[partition]:
            name = node_names[key]
            lines.append(f"{name} [label=<{_port_label(incoming[key], outgoing[key])}>];")
        lines.append("}")

    # Metadata columns describe the actual inspected network; equal columns
    # remain aligned while unknown/missing columns are left to the DAG ranker.
    ranks: dict[float, list[str]] = defaultdict(list)
    for node in nodes:
        if node["column"] is not None:
            ranks[node["column"]].append(node_names[node["key"]])
    for rank, names in sorted(ranks.items()):
        if len(names) > 1:
            lines.append("{ rank=same; " + "; ".join(names) + "; }")

    for source, target in edges:
        tail_port, head_port = ports[(source, target)]
        lines.append(
            f"{node_names[source]}:{tail_port}:e -> {node_names[target]}:{head_port}:w;"
        )
    lines.append("}")
    return "\n".join(lines), node_names, ports


def _parse_pos(value: Any) -> tuple[float, float]:
    parts = str(value).split(",")
    if len(parts) < 2:
        raise ValueError("Graphviz node has no x,y position")
    return float(parts[0]), float(parts[1])


def _cubic_sample(curves: Sequence[Sequence[float]], steps_per_segment: int = 16) -> list[tuple[float, float]]:
    if len(curves) < 4 or (len(curves) - 1) % 3:
        raise ValueError("Graphviz spline is not a sequence of cubic Bezier segments")
    points: list[tuple[float, float]] = []
    for offset in range(0, len(curves) - 1, 3):
        p0, p1, p2, p3 = curves[offset:offset + 4]
        for step in range(steps_per_segment + 1):
            if offset and step == 0:
                continue
            t = step / steps_per_segment
            one = 1.0 - t
            x = one**3 * p0[0] + 3 * one**2 * t * p1[0] + 3 * one * t**2 * p2[0] + t**3 * p3[0]
            y = one**3 * p0[1] + 3 * one**2 * t * p1[1] + 3 * one * t**2 * p2[1] + t**3 * p3[1]
            points.append((x, y))
    return points


def _svg_path(points: Sequence[Sequence[float]]) -> str:
    if len(points) < 4 or (len(points) - 1) % 3:
        raise ValueError("route points cannot be encoded as cubic SVG segments")
    values = [f"M {points[0][0]:.3f},{points[0][1]:.3f}"]
    for offset in range(1, len(points), 3):
        first, second, third = points[offset:offset + 3]
        values.append(
            f"C {first[0]:.3f},{first[1]:.3f} {second[0]:.3f},{second[1]:.3f} {third[0]:.3f},{third[1]:.3f}"
        )
    return " ".join(values)


def _rounded_orthogonal_curve(points: Sequence[Sequence[float]], radius: float = 32.0) -> list[tuple[float, float]]:
    """Convert an orthogonal route to joined cubic Beziers with rounded bends.

    Graphviz's ortho router avoids unrelated cards, but its raw route is a set
    of right-angle segments. Rounded corners keep the viewer's connections
    visually continuous while preserving the router's obstacle clearance.
    The point evaluation below follows the cubic interpolation in
    ``external/llm-viz/src/utils/bezier.ts`` (MIT; see THIRD_PARTY_NOTICES.md).
    """
    raw_route: list[tuple[float, float]] = []
    for raw in points:
        point = (float(raw[0]), float(raw[1]))
        if not raw_route or math.dist(point, raw_route[-1]) > 0.1:
            raw_route.append(point)
    route: list[tuple[float, float]] = []
    for point in raw_route:
        while len(route) >= 2:
            a, b = route[-2], route[-1]
            cross = (b[0] - a[0]) * (point[1] - b[1]) - (b[1] - a[1]) * (point[0] - b[0])
            if abs(cross) > 0.1:
                break
            route.pop()
        route.append(point)
    if len(route) < 2:
        raise ValueError("Graphviz orthogonal route has fewer than two distinct points")

    segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    corners: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], float]] = []
    for index in range(1, len(route) - 1):
        previous, corner, following = route[index - 1:index + 2]
        incoming_length = math.dist(previous, corner)
        outgoing_length = math.dist(corner, following)
        if incoming_length < 1e-3 or outgoing_length < 1e-3:
            continue
        # Keep enough straight length between adjacent rounded corners that
        # both cubic handles fit without independently scaling either side.
        # Matching the handles is what makes each join C1, not only tangent-
        # collinear (G1).
        turn_radius = min(radius, incoming_length * 0.24, outgoing_length * 0.24)
        incoming = ((corner[0] - previous[0]) / incoming_length,
                    (corner[1] - previous[1]) / incoming_length)
        outgoing = ((following[0] - corner[0]) / outgoing_length,
                    (following[1] - corner[1]) / outgoing_length)
        start = (corner[0] - incoming[0] * turn_radius, corner[1] - incoming[1] * turn_radius)
        end = (corner[0] + outgoing[0] * turn_radius, corner[1] + outgoing[1] * turn_radius)
        corners.append((start, end, incoming, outgoing, turn_radius))

    current = route[0]
    current_tangent_handle: float | None = None
    kappa = 0.5522847498307936
    for corner_start, corner_end, incoming, outgoing, turn_radius in corners:
        length = math.dist(current, corner_start)
        if length > 0.1:
            direction = ((corner_start[0] - current[0]) / length,
                         (corner_start[1] - current[1]) / length)
            start_handle = current_tangent_handle if current_tangent_handle is not None else min(42.0, length / 3.0)
            end_handle = kappa * turn_radius
            if start_handle + end_handle > length * 0.82:
                raise ValueError("orthogonal straight is too short for a C1 cubic join")
            segments.append((current,
                             (current[0] + direction[0] * start_handle,
                              current[1] + direction[1] * start_handle),
                             (corner_start[0] - direction[0] * end_handle,
                              corner_start[1] - direction[1] * end_handle),
                             corner_start))
        segments.append((corner_start,
                         (corner_start[0] + incoming[0] * kappa * turn_radius,
                          corner_start[1] + incoming[1] * kappa * turn_radius),
                         (corner_end[0] - outgoing[0] * kappa * turn_radius,
                          corner_end[1] - outgoing[1] * kappa * turn_radius),
                         corner_end))
        current = corner_end
        current_tangent_handle = kappa * turn_radius

    end = route[-1]
    length = math.dist(current, end)
    if length > 0.1:
        direction = ((end[0] - current[0]) / length, (end[1] - current[1]) / length)
        start_handle = current_tangent_handle if current_tangent_handle is not None else min(42.0, length / 3.0)
        end_handle = min(42.0, length / 3.0)
        if start_handle + end_handle > length * 0.82:
            raise ValueError("orthogonal exit is too short for a C1 cubic join")
        segments.append((current,
                         (current[0] + direction[0] * start_handle,
                          current[1] + direction[1] * start_handle),
                         (end[0] - direction[0] * end_handle,
                          end[1] - direction[1] * end_handle),
                         end))
    if not segments:
        raise ValueError("Graphviz orthogonal route could not be converted to cubics")

    # The SVG helper accepts p0,c1,c2,p3 followed by three points per segment.
    controls = [segments[0][0]]
    for p0, c1, c2, p3 in segments:
        if math.dist(controls[-1], p0) > 0.1:
            raise ValueError("rounded orthogonal curve contains a discontinuity")
        controls.extend((c1, c2, p3))
    return controls


def _rectangles(positions: Mapping[str, Mapping[str, float]]) -> dict[str, tuple[float, float, float, float]]:
    return {
        key: (float(value["x"]), float(value["y"]),
              float(value["x"]) + CARD_WIDTH, float(value["y"]) + CARD_HEIGHT)
        for key, value in positions.items()
    }


def _segment_intersection(a: tuple[float, float], b: tuple[float, float],
                          c: tuple[float, float], d: tuple[float, float]) -> tuple[float, float] | None:
    abx, aby = b[0] - a[0], b[1] - a[1]
    cdx, cdy = d[0] - c[0], d[1] - c[1]
    denominator = abx * cdy - aby * cdx
    if abs(denominator) < 1e-8:
        return None
    acx, acy = c[0] - a[0], c[1] - a[1]
    t = (acx * cdy - acy * cdx) / denominator
    u = (acx * aby - acy * abx) / denominator
    if 1e-5 < t < 1.0 - 1e-5 and 1e-5 < u < 1.0 - 1e-5:
        return a[0] + t * abx, a[1] + t * aby
    return None


def _inside(rect: tuple[float, float, float, float], point: tuple[float, float], pad: float = 1.0) -> bool:
    return rect[0] + pad < point[0] < rect[2] - pad and rect[1] + pad < point[1] < rect[3] - pad


def _diagnostics(positions: Mapping[str, Mapping[str, float]], edge_routes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rectangles = _rectangles(positions)
    keys = sorted(rectangles)
    card_overlaps: list[list[str]] = []
    for index, left_key in enumerate(keys):
        ax0, ay0, ax1, ay1 = rectangles[left_key]
        for right_key in keys[index + 1:]:
            bx0, by0, bx1, by1 = rectangles[right_key]
            if min(ax1, bx1) - max(ax0, bx0) > 0.5 and min(ay1, by1) - max(ay0, by0) > 0.5:
                card_overlaps.append([left_key, right_key])

    curve_card_hits: set[tuple[str, str]] = set()
    for route in edge_routes:
        source, target = route["source"], route["target"]
        sampled = route["samples"]
        for card_key, rect in rectangles.items():
            if card_key in (source, target):
                continue
            min_x = min(point[0] for point in sampled)
            max_x = max(point[0] for point in sampled)
            min_y = min(point[1] for point in sampled)
            max_y = max(point[1] for point in sampled)
            if max_x < rect[0] or min_x > rect[2] or max_y < rect[1] or min_y > rect[3]:
                continue
            if any(_inside(rect, point) for point in sampled[1:-1]):
                curve_card_hits.add((source + "→" + target, card_key))

    crossing_pairs: set[tuple[str, str]] = set()
    crossing_points: list[dict[str, Any]] = []
    route_boxes: list[tuple[float, float, float, float]] = []
    for route in edge_routes:
        xs = [point[0] for point in route["samples"]]
        ys = [point[1] for point in route["samples"]]
        route_boxes.append((min(xs), min(ys), max(xs), max(ys)))
    for index, first in enumerate(edge_routes):
        first_box = route_boxes[index]
        first_samples = first["samples"]
        for other_index in range(index + 1, len(edge_routes)):
            second = edge_routes[other_index]
            if not _boxes_overlap(first_box, route_boxes[other_index]):
                continue
            shared_nodes = {first["source"], first["target"]} & {second["source"], second["target"]}
            pair_key = tuple(sorted((first["source"] + "→" + first["target"],
                                     second["source"] + "→" + second["target"])))
            found: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
            second_samples = second["samples"]
            # Use local segment boxes first; most candidate pairs are rejected
            # cheaply because routes occupy disjoint rank corridors.
            for a, b in zip(first_samples, first_samples[1:]):
                seg_a = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
                for c, d in zip(second_samples, second_samples[1:]):
                    seg_b = (min(c[0], d[0]), min(c[1], d[1]), max(c[0], d[0]), max(c[1], d[1]))
                    if not _boxes_overlap(seg_a, seg_b):
                        continue
                    point = _segment_intersection(a, b, c, d)
                    if point is None:
                        continue
                    if any(_inside(rectangles[key], point, pad=3.0) for key in shared_nodes):
                        continue
                    if not any(math.dist(point, old[0]) < 5.0 for old in found):
                        first_length = math.dist(a, b) or 1.0
                        second_length = math.dist(c, d) or 1.0
                        found.append((point,
                                      ((b[0] - a[0]) / first_length, (b[1] - a[1]) / first_length),
                                      ((d[0] - c[0]) / second_length, (d[1] - c[1]) / second_length)))
            if found:
                crossing_pairs.add(pair_key)
                first_id = first["source"] + "→" + first["target"]
                second_id = second["source"] + "→" + second["target"]
                under_id, over_id = sorted((first_id, second_id))
                for point, first_tangent, second_tangent in found:
                    under_tangent, over_tangent = (first_tangent, second_tangent) if under_id == first_id else (second_tangent, first_tangent)
                    crossing_points.append({
                        "x": round(point[0], 3), "y": round(point[1], 3),
                        "first": first_id, "second": second_id,
                        "under": under_id, "over": over_id,
                        "under_tangent": [round(under_tangent[0], 6), round(under_tangent[1], 6)],
                        "over_tangent": [round(over_tangent[0], 6), round(over_tangent[1], 6)],
                    })

    shared_path_pairs: list[list[str]] = []
    grid = 8.0
    cell_sets: list[set[tuple[int, int]]] = []
    for route in edge_routes:
        samples = route["samples"][2:-2]
        cell_sets.append({(round(point[0] / grid), round(point[1] / grid)) for point in samples})
    for index, first in enumerate(edge_routes):
        for other_index in range(index + 1, len(edge_routes)):
            if not _boxes_overlap(route_boxes[index], route_boxes[other_index]):
                continue
            common_cells = len(cell_sets[index] & cell_sets[other_index])
            if common_cells >= 4:
                shared_path_pairs.append(sorted((first["source"] + "→" + first["target"],
                                                  edge_routes[other_index]["source"] + "→" + edge_routes[other_index]["target"])))

    return {
        "card_overlap_pairs": card_overlaps,
        "curve_card_intersections": [list(item) for item in sorted(curve_card_hits)],
        "shared_path_pairs": sorted(shared_path_pairs),
        "crossing_pairs": sorted([list(pair) for pair in crossing_pairs]),
        "crossing_points": crossing_points,
        "counts": {
            "card_overlap_pairs": len(card_overlaps),
            "curve_card_intersections": len(curve_card_hits),
            "shared_path_pairs": len(shared_path_pairs),
            "crossing_pairs": len(crossing_pairs),
            "crossing_points": len(crossing_points),
        },
    }


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _regions(nodes: Sequence[dict[str, Any]], positions: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, str] = {}
    for node in nodes:
        name, label = _partition(node["branch"])
        groups[name].append(node["key"])
        labels[name] = label
    result = []
    for group, keys in groups.items():
        located = [positions[key] for key in keys if key in positions]
        if not located:
            continue
        left = min(item["x"] for item in located) - 18.0
        # Reserve a dedicated caption strip above the first card so branch
        # labels cannot sit on top of input ports or incoming routes.
        top = min(item["y"] for item in located) - 50.0
        right = max(item["x"] + CARD_WIDTH for item in located) + 18.0
        bottom = max(item["y"] + CARD_HEIGHT for item in located) + 18.0
        result.append({
            "key": group, "label": labels[group],
            "x": round(left, 3), "y": round(top, 3),
            "width": round(right - left, 3), "height": round(bottom - top, 3),
        })
    return sorted(result, key=lambda item: (item["y"], item["x"], item["key"]))


def _fallback_layout(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], signature: str, reason: str) -> dict[str, Any]:
    by_key = {node["key"]: node for node in nodes}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        group, _label = _partition(node["branch"])
        groups[group].append(node)
    group_order = sorted(groups, key=lambda group: (
        0 if group.startswith("sensor-") else 1 if group == "fusion" else 2 if group == "actor" else 3 if group == "critic" else 4,
        _partition(groups[group][0]["branch"])[1], group,
    ))
    y_cursor = GRAPH_PADDING
    positions: dict[str, dict[str, float]] = {}
    group_top: dict[str, float] = {}
    for group in group_order:
        group_top[group] = y_cursor
        by_column: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for node in groups[group]:
            by_column[node["column"] if node["column"] is not None else 0.0].append(node)
        max_count = 1
        for column in sorted(by_column):
            column_nodes = sorted(by_column[column], key=lambda item: item["key"])
            max_count = max(max_count, len(column_nodes))
            for row, node in enumerate(column_nodes):
                positions[node["key"]] = {
                    "x": GRAPH_PADDING + float(column) * (CARD_WIDTH + 96.0),
                    "y": y_cursor + row * (CARD_HEIGHT + 34.0),
                }
        y_cursor += max(CARD_HEIGHT + 36.0, max_count * (CARD_HEIGHT + 34.0)) + 54.0

    edge_routes: list[dict[str, Any]] = []
    port_order: dict[str, dict[str, int]] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    for node in nodes:
        outgoing[node["key"]].sort(key=lambda target: (positions[target]["y"], target))
        incoming[node["key"]].sort(key=lambda source: (positions[source]["y"], source))
    for source, target in edges:
        source_pos, target_pos = positions[source], positions[target]
        source_idx = outgoing[source].index(target)
        target_idx = incoming[target].index(source)
        source_y = source_pos["y"] + CARD_HEIGHT * (source_idx + 1) / (len(outgoing[source]) + 1)
        target_y = target_pos["y"] + CARD_HEIGHT * (target_idx + 1) / (len(incoming[target]) + 1)
        sx, ex = source_pos["x"] + CARD_WIDTH, target_pos["x"]
        gap = ex - sx
        bend = max(28.0, gap * 0.42)
        route_points = [(sx, source_y), (sx + bend, source_y), (ex - bend, target_y), (ex, target_y)]
        edge_routes.append({"source": source, "target": target, "path": _svg_path(route_points), "samples": _cubic_sample(route_points, 24)})
        port_order[source + "\0" + target] = {"source": source_idx, "target": target_idx}

    all_points = list(positions.values())
    max_x = max((item["x"] + CARD_WIDTH for item in all_points), default=1.0) + GRAPH_PADDING
    max_y = max((item["y"] + CARD_HEIGHT for item in all_points), default=1.0) + GRAPH_PADDING
    diag = _diagnostics(positions, edge_routes)
    diagnostics_public = {key: value for key, value in diag.items() if key != "crossing_points"}
    diagnostics_public["missing_parent_edges"] = sorted({
        parent for node in nodes for parent in node["parents"] if parent not in by_key
    })
    return {
        "topology": signature,
        "engine": "fallback · 按列与支路固定布局",
        "status": "fallback",
        "reason": reason,
        "positions": positions,
        "edges": [{key: route[key] for key in ("source", "target", "path")} for route in edge_routes],
        "regions": _regions(nodes, positions),
        "bounds": {"minX": 0.0, "minY": 0.0, "maxX": round(max_x, 3), "maxY": round(max_y, 3)},
        "diagnostics": diagnostics_public,
        "crossings": diag["crossing_points"],
    }


def _graphviz_layout(nodes: list[dict[str, Any]], edges: list[tuple[str, str]], signature: str,
                     dot_command: Sequence[str], runner: Callable[..., Any]) -> dict[str, Any]:
    node_names = {node["key"]: f"n{index}" for index, node in enumerate(nodes)}
    by_key = {node["key"] for node in nodes}
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        incoming[target].append(source)
        outgoing[source].append(target)
    port_orders: dict[str, dict[str, list[str]]] | None = None
    best: tuple[tuple[int, int, int, int], dict[str, Any]] | None = None

    # First let dot find the layer/partition arrangement, then assign each
    # explicit side port in the same top-to-bottom order as its neighbor. A
    # few deterministic passes remove avoidable fan-in/fan-out crossings.
    for _iteration in range(4):
        dot, node_names, ports = _dot_source(nodes, edges, port_orders)
        # The orthogonal router respects the HTML table's per-edge side ports
        # and routes around unrelated nodes. Its bends are rounded below and
        # emitted as cubic SVG paths, so the visible geometry remains smooth.
        dot = dot.replace("splines=true", "splines=ortho")
        completed = runner(
            [*dot_command, "-Tjson"], input=dot, text=True, encoding="utf-8",
            errors="replace", capture_output=True, check=True, timeout=_GRAPHVIZ_TIMEOUT_SECONDS,
        )
        document = json.loads(completed.stdout)
        bb_parts = [float(part) for part in str(document.get("bb", "0,0,0,0")).split(",")]
        if len(bb_parts) != 4:
            raise ValueError("Graphviz did not return a valid bounding box")
        min_x, min_y, max_x, max_y = bb_parts
        graph_width = max_x - min_x
        graph_height = max_y - min_y
        if graph_width <= 0 or graph_height <= 0:
            raise ValueError("Graphviz returned an empty bounding box")
        pos_by_name: dict[str, tuple[float, float]] = {}
        for item in document.get("objects", []):
            name = str(item.get("name", ""))
            if name in node_names.values():
                pos_by_name[name] = _parse_pos(item.get("pos"))
        positions: dict[str, dict[str, float]] = {}
        for key, name in node_names.items():
            if name not in pos_by_name:
                raise ValueError(f"Graphviz omitted position for node {key}")
            center_x, center_y = pos_by_name[name]
            positions[key] = {
                "x": round(center_x - CARD_WIDTH / 2.0 - min_x + GRAPH_PADDING, 3),
                "y": round(max_y - center_y - CARD_HEIGHT / 2.0 + GRAPH_PADDING, 3),
            }

        route_by_edge: dict[tuple[str, str], dict[str, Any]] = {}
        edge_lookup = {name: key for key, name in node_names.items()}
        object_name_by_id = {
            int(item["_gvid"]): str(item["name"])
            for item in document.get("objects", [])
            if "_gvid" in item and "name" in item
        }
        for item in document.get("edges", []):
            try:
                source_name = object_name_by_id[int(item["tail"])]
                target_name = object_name_by_id[int(item["head"])]
                source = edge_lookup[source_name]
                target = edge_lookup[target_name]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError("Graphviz edge endpoints could not be mapped") from exc
            route_points: list[tuple[float, float]] = []
            for drawing in item.get("_draw_", []):
                if drawing.get("op") == "b":
                    route_points.extend((float(point[0]), float(point[1])) for point in drawing.get("points", []))
            if len(route_points) < 2:
                raise ValueError(f"Graphviz omitted orthogonal route for {source} -> {target}")
            transformed = [
                (x - min_x + GRAPH_PADDING, max_y - y + GRAPH_PADDING)
                for x, y in route_points
            ]
            controls = _rounded_orthogonal_curve(transformed)
            samples = _cubic_sample(controls, 24)
            route_by_edge[(source, target)] = {
                "source": source,
                "target": target,
                "path": _svg_path(controls),
                "samples": samples,
            }
        if len(route_by_edge) != len(edges):
            raise ValueError(f"Graphviz returned {len(route_by_edge)} of {len(edges)} routes")
        routes = [route_by_edge[(source, target)] for source, target in edges]
        diag = _diagnostics(positions, routes)
        score = (
            diag["counts"]["card_overlap_pairs"],
            diag["counts"]["curve_card_intersections"],
            diag["counts"]["shared_path_pairs"],
            diag["counts"]["crossing_pairs"],
        )
        diagnostics_public = {key: value for key, value in diag.items() if key != "crossing_points"}
        diagnostics_public["missing_parent_edges"] = sorted({
            parent for node in nodes for parent in node["parents"] if parent not in by_key
        })
        candidate = {
            "topology": signature,
            "engine": "Graphviz dot · 分层避障圆角贝塞尔",
            "status": "graphviz",
            "reason": "",
            "positions": positions,
            "edges": [{key: route[key] for key in ("source", "target", "path")} for route in routes],
            "regions": _regions(nodes, positions),
            "bounds": {
                "minX": 0.0, "minY": 0.0,
                "maxX": round(graph_width + 2 * GRAPH_PADDING, 3),
                "maxY": round(graph_height + 2 * GRAPH_PADDING, 3),
            },
            "diagnostics": diagnostics_public,
            "crossings": diag["crossing_points"],
        }
        if best is None or score < best[0]:
            best = (score, candidate)

        next_orders = _neighbor_ordered_ports(positions, routes, ports, incoming, outgoing)
        if next_orders == port_orders:
            break
        port_orders = next_orders

    if best is None:
        raise ValueError("Graphviz did not produce a layout")
    return best[1]


def _neighbor_ordered_ports(positions: Mapping[str, Mapping[str, float]], routes: Sequence[dict[str, Any]],
                            ports: Mapping[tuple[str, str], tuple[str, str]],
                            incoming: Mapping[str, Sequence[str]], outgoing: Mapping[str, Sequence[str]]) -> dict[str, dict[str, list[str]]]:
    input_slots: dict[str, dict[int, float]] = defaultdict(dict)
    output_slots: dict[str, dict[int, float]] = defaultdict(dict)
    for route in routes:
        source, target = route["source"], route["target"]
        output_name, input_name = ports[(source, target)]
        output_slots[source][int(output_name.removeprefix("out"))] = route["samples"][0][1]
        input_slots[target][int(input_name.removeprefix("in"))] = route["samples"][-1][1]

    incoming_order: dict[str, list[str]] = {}
    for target, sources in incoming.items():
        if not sources:
            continue
        desired = sorted(sources, key=lambda key: (positions[key]["y"], key))
        slots = [slot for slot, _y in sorted(input_slots[target].items(), key=lambda item: (item[1], item[0]))]
        order = [""] * len(sources)
        for source, slot in zip(desired, slots):
            order[slot] = source
        incoming_order[target] = order

    outgoing_order: dict[str, list[str]] = {}
    for source, targets in outgoing.items():
        if not targets:
            continue
        desired = sorted(targets, key=lambda key: (positions[key]["y"], key))
        slots = [slot for slot, _y in sorted(output_slots[source].items(), key=lambda item: (item[1], item[0]))]
        order = [""] * len(targets)
        for target, slot in zip(desired, slots):
            order[slot] = target
        outgoing_order[source] = order
    return {"incoming": incoming_order, "outgoing": outgoing_order}


def _dot_command(dot_executable: str | Sequence[str] | None) -> list[str] | None:
    if isinstance(dot_executable, str):
        return [dot_executable]
    if dot_executable is not None:
        return [str(part) for part in dot_executable]
    executable = shutil.which("dot")
    if executable:
        return [executable]
    # The project viewer runs as a Windows process while Graphviz is installed
    # in its Ubuntu WSL distro. Keep the bridge optional so non-WSL installs
    # transparently use the documented curve fallback.
    wsl = shutil.which("wsl.exe") if os.name == "nt" else None
    if wsl:
        return [wsl, "-d", "Ubuntu-24.04", "--exec", "dot"]
    return None


def build_layout(nodes: Iterable[Mapping[str, Any]], *, dot_executable: str | Sequence[str] | None = None,
                 runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Build a fixed layout from actual node/parent metadata, with a curve fallback."""
    normalized = _normalized_nodes(nodes)
    signature = topology_signature(normalized)
    keys = {node["key"] for node in normalized}
    edges = list(dict.fromkeys(
        (parent, node["key"])
        for node in normalized for parent in node["parents"] if parent in keys
    ))
    command = _dot_command(dot_executable)
    if command:
        try:
            return _graphviz_layout(normalized, edges, signature, command, runner)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            reason = f"Graphviz不可用或布局失败：{type(exc).__name__}"
    else:
        reason = "未找到Graphviz dot；使用固定列/支路曲线布局"
    return _fallback_layout(normalized, edges, signature, reason)


class NetworkLayoutCache:
    """Thread-safe small LRU keyed only by graph topology, never frame data."""

    def __init__(self, *, dot_executable: str | Sequence[str] | None = None,
                 runner: Callable[..., Any] = subprocess.run) -> None:
        self._dot_executable = dot_executable
        self._runner = runner
        self._lock = threading.Lock()
        self._layouts: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, nodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        normalized = _normalized_nodes(nodes)
        signature = topology_signature(normalized)
        with self._lock:
            cached = self._layouts.get(signature)
            if cached is not None:
                self._layouts.move_to_end(signature)
                return cached
            layout = build_layout(normalized, dot_executable=self._dot_executable, runner=self._runner)
            self._layouts[signature] = layout
            self._layouts.move_to_end(signature)
            while len(self._layouts) > _CACHE_SIZE:
                self._layouts.popitem(last=False)
            return layout


__all__ = ["NetworkLayoutCache", "build_layout", "topology_signature"]
