#!/usr/bin/env python3
"""Static layout and provenance checks for the expanded live network preview."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "isaac_field" / "visualization" / "network_2d.html"
FRAME = ROOT / "results" / "isaac_field_viewer" / "direct_tracks_20261020" / "frame_after.json"
HTML = HERE / "index.html"
OUTPUT = HERE / "static_validation.json"

CARD_W, CARD_H = 170, 110
CARD_GAP_X = max(90, round(CARD_W * 0.53))
COL_GAP = CARD_W + CARD_GAP_X
GRAPH_LEFT = round(CARD_W * 1.2)
SENSOR_ROW_STEP = 2 * CARD_H + 20
MEMORY_GATE_STEP = CARD_H + 50
MEMORY_INPUT_OFFSET = CARD_H + 50
MEMORY_C0_OFFSET = 4 * MEMORY_GATE_STEP
MEMORY_C1_OFFSET = round(1.5 * MEMORY_GATE_STEP)
MEMORY_H1_OFFSET = round(2.25 * MEMORY_GATE_STEP)
MEMORY_SECTION_GAP = round(CARD_H * 1.73)
ACTOR_GATE_TOP = 0
CRITIC_GATE_TOP = ACTOR_GATE_TOP + MEMORY_C0_OFFSET + CARD_H + MEMORY_SECTION_GAP + MEMORY_INPUT_OFFSET
SENSOR_MAX_Y = 3 * SENSOR_ROW_STEP
INPUT_TO_MASK_ROUTE_Y = SENSOR_MAX_Y + CARD_H + 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_column(node: dict) -> float:
    try:
        return float(node.get("column", 0))
    except (TypeError, ValueError):
        return 0.0


def layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    sorted_nodes = sorted(nodes, key=lambda n: (node_column(n), str(n["key"])))
    positions: dict[str, tuple[float, float]] = {}
    sensor_rows = {"lidar": 0, "depth": 1, "detections": 2, "state": 3}
    gate_rows = {"f": 0, "i": 1, "g": 2, "o": 3}
    actor_y = ACTOR_GATE_TOP + MEMORY_H1_OFFSET
    critic_y = CRITIC_GATE_TOP + MEMORY_H1_OFFSET
    distribution_y = ACTOR_GATE_TOP + MEMORY_C0_OFFSET
    fallback_rows: dict[str, int] = {}

    def x_at(column: float) -> float:
        return GRAPH_LEFT + column * COL_GAP

    for node in sorted_nodes:
        key, branch = str(node["key"]), str(node.get("branch", ""))
        col = node_column(node)
        position = None
        if branch in sensor_rows:
            y = sensor_rows[branch] * SENSOR_ROW_STEP
            position = (x_at(col), y + (CARD_H + 10 if key == "detections.winner" else 0))
        elif key == "features_extractor":
            position = (x_at(9), 1.5 * SENSOR_ROW_STEP)
        elif key.startswith("lstm_actor.") or key.startswith("lstm_critic."):
            is_critic = key.startswith("lstm_critic.")
            top = CRITIC_GATE_TOP if is_critic else ACTOR_GATE_TOP
            prefix = "lstm_critic" if is_critic else "lstm_actor"
            suffix = key[len(prefix) + 1:]
            states = {
                "h0": (10, top - MEMORY_INPUT_OFFSET),
                "c0": (11, top + MEMORY_C0_OFFSET),
                "c": (12, top + MEMORY_C1_OFFSET),
                "h": (13, top + MEMORY_H1_OFFSET),
            }
            if suffix in states:
                column, y = states[suffix]
                position = (x_at(column), y)
            elif suffix in gate_rows:
                position = (x_at(11), top + gate_rows[suffix] * MEMORY_GATE_STEP)
        elif key == "lstm_actor":
            position = (x_at(14), actor_y)
        elif key == "mlp_extractor.policy_net.0":
            position = (x_at(15), actor_y)
        elif key == "mlp_extractor.policy_net.1":
            position = (x_at(16), actor_y)
        elif key == "action_net":
            position = (x_at(17), actor_y)
        elif key == "action.log_std":
            position = (x_at(17), distribution_y)
        elif key == "action.std":
            position = (x_at(18), distribution_y)
        elif key == "action.applied":
            position = (x_at(19), actor_y)
        elif key == "lstm_critic":
            position = (x_at(14), critic_y)
        elif key == "mlp_extractor.value_net.0":
            position = (x_at(15), critic_y)
        elif key == "mlp_extractor.value_net.1":
            position = (x_at(16), critic_y)
        elif key == "value_net":
            position = (x_at(17), critic_y)
        if position is None:
            row = fallback_rows.get(branch, 0)
            fallback_rows[branch] = row + 1
            position = (
                x_at(col),
                CRITIC_GATE_TOP + MEMORY_C0_OFFSET + 2 * CARD_H + MEMORY_SECTION_GAP + row * (CARD_H + 30),
            )
        positions[key] = position
    return positions


def rects_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) < min(a[0] + CARD_W, b[0] + CARD_W) and max(a[1], b[1]) < min(a[1] + CARD_H, b[1] + CARD_H)


def segment_hits_card(a, b, pos, excluded: set[str], key: str) -> bool:
    if key in excluded:
        return False
    x, y = pos
    if abs(a[1] - b[1]) < 1e-7:
        sy = a[1]
        return y < sy < y + CARD_H and max(min(a[0], b[0]), x) < min(max(a[0], b[0]), x + CARD_W)
    if abs(a[0] - b[0]) < 1e-7:
        sx = a[0]
        return x < sx < x + CARD_W and max(min(a[1], b[1]), y) < min(max(a[1], b[1]), y + CARD_H)
    raise AssertionError("Only orthogonal edge segments are expected")


def validate_edge_routes(nodes: list[dict], positions: dict[str, tuple[float, float]], edges: list[list[str]]) -> list[dict]:
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for source, target in edges:
        outgoing.setdefault(source, []).append(target)
        incoming.setdefault(target, []).append(source)
    center_y = lambda key: positions[key][1] + CARD_H / 2
    for source, targets in outgoing.items():
        targets.sort(key=lambda key: (center_y(key), key))
    for target, sources in incoming.items():
        sources.sort(key=lambda key: (center_y(key), key))

    violations = []
    for source, target in edges:
        from_x, from_y = positions[source]
        to_x, to_y = positions[target]
        out_targets, in_sources = outgoing[source], incoming[target]
        out_index, in_index = out_targets.index(target), in_sources.index(source)
        sy = from_y + CARD_H * (out_index + 1) / (len(out_targets) + 1)
        ey = to_y + CARD_H * (in_index + 1) / (len(in_sources) + 1)
        sx, ex = from_x + CARD_W, to_x
        if source == "input.detections" and target == "detections.masked":
            sx2, ex2 = sx + CARD_GAP_X / 2, ex - CARD_GAP_X / 2
            points = [(sx, sy), (sx2, sy), (sx2, INPUT_TO_MASK_ROUTE_Y), (ex2, INPUT_TO_MASK_ROUTE_Y), (ex2, ey), (ex, ey)]
        elif abs(ey - sy) < 0.5:
            points = [(sx, sy), (ex, ey)]
        else:
            out_rank = out_index - (len(out_targets) - 1) / 2
            in_rank = in_index - (len(in_sources) - 1) / 2
            fan = lambda rank: rank * min(4, CARD_GAP_X / 18)
            sx2 = sx + CARD_GAP_X / 2 + fan(out_rank)
            ex2 = ex - CARD_GAP_X / 2 + fan(in_rank)
            candidates = {sy, ey, (sy + ey) / 2}
            min_y, max_y = float("inf"), float("-inf")
            for px, py in positions.values():
                min_y, max_y = min(min_y, py), max(max_y, py + CARD_H)
                candidates.update((py - 5, py + CARD_H + 5))
            ignored = {source, target}
            def clear(y):
                return all(
                    key in ignored or not (
                        max(min(sx2, ex2), px) < min(max(sx2, ex2), px + CARD_W)
                        and y > py - 1 and y < py + CARD_H + 1
                    )
                    for key, (px, py) in positions.items()
                )
            safe = sorted((y for y in candidates if clear(y)), key=lambda y: (abs(y - sy) + abs(y - ey), y))
            route_y = safe[0] if safe else min_y - CARD_H
            points = [(sx, sy), (sx2, sy), (sx2, route_y), (ex2, route_y), (ex2, ey), (ex, ey)]
        ignored = {source, target}
        for index, (a, b) in enumerate(zip(points, points[1:])):
            for key, pos in positions.items():
                if segment_hits_card(a, b, pos, ignored, key):
                    violations.append({"edge": [source, target], "segment": index, "intersects": key})
    return violations


def main() -> None:
    data = json.loads(FRAME.read_text(encoding="utf-8"))
    nodes = data["nodes"]
    keys = [str(node["key"]) for node in nodes]
    key_set = set(keys)
    positions = layout(nodes)
    missing_parents = sorted({parent for node in nodes for parent in node.get("parents", []) if parent not in key_set})
    edges = [[parent, str(node["key"])] for node in nodes for parent in node.get("parents", []) if parent in key_set]
    overlaps = []
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            if rects_overlap(positions[first], positions[second]):
                overlaps.append([first, second])
    route_violations = validate_edge_routes(nodes, positions, edges)
    source_hash, frame_hash = sha(SOURCE), sha(FRAME)
    html_text = HTML.read_text(encoding="utf-8")
    checks = {
        "source_snapshot_hash_unchanged": source_hash == "149797ABFE2519E9058D4CF6F580727B6AC4E8FB6FF0B47881C7CE4CED12D5A9".lower(),
        "all_nodes_expanded": len(nodes) == 57 and len(positions) == len(nodes),
        "all_parents_present": not missing_parents,
        "all_node_cards_non_overlapping": not overlaps,
        "all_edges_route_clear_of_non_endpoint_cards": not route_violations,
        "no_overview_or_mode_switch": not any(token in html_text for token in ("overviewGraph", "detailMode", "overviewMode", "mode-overview", "mode-detail")),
        "api_always_requests_detail": 'detail: "1"' in html_text and 'detail: "0"' not in html_text,
    }
    result = {
        "frame_sequence": data.get("sequence"),
        "frame_sim_time": data.get("sim_time"),
        "frame_output_time": data.get("output_time"),
        "source_network_2d_sha256": source_hash,
        "source_snapshot_hash_expected": "149797abfe2519e9058d4cf6f580727b6ac4e8fb6ff0b47881c7ce4ced12d5a9",
        "frame_after_sha256": frame_hash,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "positions": [{"key": key, "x": positions[key][0], "y": positions[key][1]} for key in keys],
        "edges": edges,
        "missing_parents": missing_parents,
        "overlaps": overlaps,
        "edge_route_violations": route_violations,
        "checks": checks,
        "passed": all(checks.values()),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "node_count": len(nodes), "edge_count": len(edges), "overlap_count": len(overlaps), "route_violation_count": len(route_violations), "checks": checks}, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
