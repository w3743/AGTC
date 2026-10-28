"""Topology-only layout checks for the complete 2D policy graph."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

import pytest

import isaac_field.visualization.network_layout as network_layout


FIXTURE_57 = Path(__file__).parent / "fixtures" / "network_layout_57.json"


def _nodes_57() -> list[dict]:
    return json.loads(FIXTURE_57.read_text(encoding="utf-8"))


def _nodes_60_feedback() -> list[dict]:
    nodes = _nodes_57()
    by_key = {node["key"]: node for node in nodes}
    by_key["features_extractor"]["parents"].append("features_extractor.feedback.1")
    nodes.extend([
        {"key": "input.feedback", "branch": "feedback", "column": 0.0, "parents": []},
        {"key": "features_extractor.feedback.0", "branch": "feedback", "column": 1.0,
         "parents": ["input.feedback"]},
        {"key": "features_extractor.feedback.1", "branch": "feedback", "column": 2.0,
         "parents": ["features_extractor.feedback.0"]},
    ])
    return nodes


def _real_edges(nodes: list[dict]) -> set[tuple[str, str]]:
    keys = {node["key"] for node in nodes}
    return {(str(parent), str(node["key"]))
            for node in nodes for parent in node.get("parents", ()) if parent in keys}


def _svg_controls(path: str) -> list[tuple[float, float]]:
    numbers = [float(value) for value in re.findall(r"-?(?:\d+\.?\d*|\.\d+)", path)]
    return list(zip(numbers[0::2], numbers[1::2]))


def _assert_c1_path(path: str) -> None:
    controls = _svg_controls(path)
    assert len(controls) >= 4 and (len(controls) - 1) % 3 == 0
    for endpoint_index in range(3, len(controls) - 1, 3):
        incoming = (controls[endpoint_index][0] - controls[endpoint_index - 1][0],
                    controls[endpoint_index][1] - controls[endpoint_index - 1][1])
        outgoing = (controls[endpoint_index + 1][0] - controls[endpoint_index][0],
                    controls[endpoint_index + 1][1] - controls[endpoint_index][1])
        assert incoming == pytest.approx(outgoing, abs=0.002)


def test_fixed_cubic_segments_have_c1_join_tangents():
    controls = network_layout._rounded_orthogonal_curve(
        [(0.0, 0.0), (160.0, 0.0), (160.0, 150.0), (310.0, 150.0)]
    )
    assert len(controls) >= 7 and (len(controls) - 1) % 3 == 0
    for endpoint_index in range(3, len(controls) - 1, 3):
        previous_tangent = (
            controls[endpoint_index][0] - controls[endpoint_index - 1][0],
            controls[endpoint_index][1] - controls[endpoint_index - 1][1],
        )
        next_tangent = (
            controls[endpoint_index + 1][0] - controls[endpoint_index][0],
            controls[endpoint_index + 1][1] - controls[endpoint_index][1],
        )
        assert previous_tangent == pytest.approx(next_tangent, abs=1e-5)
    path = network_layout._svg_path(controls)
    assert path.startswith("M ") and path.count(" C ") == (len(controls) - 1) // 3


@pytest.mark.parametrize("route", [
    [(0.0, 0.0), (34.0, 0.0), (34.0, 23.0), (58.0, 23.0)],
    [(0.0, 0.0), (44.0, 0.0), (44.0, 26.0), (76.0, 26.0), (76.0, 62.0)],
    [(0.0, 0.0), (160.0, 0.0), (160.0, 28.0), (240.0, 28.0), (240.0, 80.0)],
])
def test_short_rounded_turns_keep_equal_derivatives_at_every_join(route):
    controls = network_layout._rounded_orthogonal_curve(route)
    assert (len(controls) - 1) % 3 == 0
    for endpoint_index in range(3, len(controls) - 1, 3):
        incoming = (
            controls[endpoint_index][0] - controls[endpoint_index - 1][0],
            controls[endpoint_index][1] - controls[endpoint_index - 1][1],
        )
        outgoing = (
            controls[endpoint_index + 1][0] - controls[endpoint_index][0],
            controls[endpoint_index + 1][1] - controls[endpoint_index][1],
        )
        assert incoming == pytest.approx(outgoing, abs=1e-5)


def test_fallback_is_explicit_and_keeps_only_real_edges():
    nodes = [
        {"key": "input.a", "branch": "state", "column": 0.0, "parents": []},
        {"key": "layer.b", "branch": "fusion", "column": 1.0, "parents": ["input.a", "missing.parent"]},
    ]
    layout = network_layout.build_layout(nodes, dot_executable=[])

    assert layout["status"] == "fallback"
    assert "未找到Graphviz" in layout["reason"]
    assert {(edge["source"], edge["target"]) for edge in layout["edges"]} == {("input.a", "layer.b")}
    assert layout["diagnostics"]["missing_parent_edges"] == ["missing.parent"]
    assert all(" C " in edge["path"] for edge in layout["edges"])


def test_layout_cache_ignores_frame_values_and_titles_but_tracks_topology(monkeypatch):
    calls = []
    monkeypatch.setattr(network_layout, "build_layout", lambda nodes, **kwargs: (
        calls.append(network_layout.topology_signature(nodes)) or {"topology": network_layout.topology_signature(nodes)}
    ))
    cache = network_layout.NetworkLayoutCache(dot_executable=[])
    nodes = _nodes_57()[:2]
    first = cache.get(nodes)
    changed_frame = [{**node, "title": "new title", "values": [math.pi]} for node in nodes]
    second = cache.get(changed_frame)
    changed_topology = [{**node, "parents": list(node["parents"])} for node in nodes]
    changed_topology[1]["parents"].append("unresolved.new.edge")
    third = cache.get(changed_topology)

    assert first is second
    assert first is not third
    assert len(calls) == 2


def test_viewer_uses_an_expanded_dark_canvas_and_real_branch_colored_cubic_edges():
    page = (Path(network_layout.__file__).with_name("network_2d.html")
            .read_text(encoding="utf-8"))
    assert '<meta name="color-scheme" content="dark">' in page
    assert 'id="network-layout-redesign"' in page
    assert "grid-template-rows: 24px 32px 18px 18px" in page
    assert "height: 111px" in page
    assert "nav-open" not in page
    assert "nav-drawer-toggle" not in page
    assert 'id="region-sidebar"' not in page
    assert "markerUnits=\"userSpaceOnUse\"" in page
    assert 'class: `flow-edge branch-${edge.branch}`' in page
    assert "unitsPerPixel" in page and "crossing-bridge" in page and "crossing-gap" in page
    assert "pointerenter" in page and "updateSelectionStyle" in page
    for branch in ("lidar", "depth", "detections", "state", "feedback", "actor", "critic", "output"):
        assert f"--branch-{branch}:" in page
        assert f"arrow-{branch}" in page
    assert "overview-toggle" not in page


@pytest.mark.parametrize("nodes_factory, expected_nodes, expected_edges", [
    (_nodes_57, 57, 68),
    (_nodes_60_feedback, 60, 71),
])
def test_graphviz_full_network_has_no_card_hits_or_shared_routes(nodes_factory, expected_nodes, expected_edges):
    if not network_layout._dot_command(None):
        pytest.skip("Graphviz is unavailable; the explicit fallback is tested separately")
    nodes = nodes_factory()
    layout = network_layout.build_layout(nodes)

    assert layout["status"] == "graphviz"
    assert len(layout["positions"]) == expected_nodes
    assert len(layout["edges"]) == expected_edges
    assert {(edge["source"], edge["target"]) for edge in layout["edges"]} == _real_edges(nodes)
    assert all(edge["path"].startswith("M ") and " C " in edge["path"] for edge in layout["edges"])
    for edge in layout["edges"]:
        _assert_c1_path(edge["path"])
    assert layout["diagnostics"]["missing_parent_edges"] == []
    counts = layout["diagnostics"]["counts"]
    assert counts["card_overlap_pairs"] == 0
    assert counts["curve_card_intersections"] == 0
    assert counts["shared_path_pairs"] == 0
    # Dense actor/critic LSTM inputs contain real, unavoidable crossings. Keep
    # the measured count visible instead of suppressing those edges/diagnostics.
    assert 0 < counts["crossing_points"] <= 20
    edge_ids = {edge["source"] + "→" + edge["target"] for edge in layout["edges"]}
    assert all(item["under"] in edge_ids and item["over"] in edge_ids for item in layout["crossings"])
    assert all(item["under_tangent"] != [0.0, 0.0] and item["over_tangent"] != [0.0, 0.0]
               for item in layout["crossings"])
