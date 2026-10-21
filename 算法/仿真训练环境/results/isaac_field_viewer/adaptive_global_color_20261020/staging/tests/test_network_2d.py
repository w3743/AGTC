"""Local HTTP contract tests for the independent 2D network view."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

import isaac_field.visualization.network_2d as network_2d
from isaac_field.visualization.activation_maps import STATE_LABELS
from isaac_field.visualization.network_2d import Network2DServer


@pytest.fixture
def http_server(monkeypatch, tmp_path):
    html_path = tmp_path / "network_2d.html"
    html_path.write_text("<a href=\"__VIEWER_URL__\">3D viewer</a>", encoding="utf-8")
    monkeypatch.setattr(network_2d, "_HTML_PATH", html_path)
    commands = []
    server = Network2DServer(0, commands.append, "http://127.0.0.1:8769/")
    try:
        yield server, commands
    finally:
        server.close()


def _get(server, path="/api/frame"):
    with urlopen(server.url.rstrip("/") + path, timeout=3.0) as response:
        return response.status, response.headers, response.read()


def _get_json(server, path="/api/frame"):
    status, headers, body = _get(server, path)
    assert status == 200
    assert headers.get_content_type() == "application/json"
    return json.loads(body)


def _image_pixels(data_url):
    image_bytes = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def _layer(key, values, *, title=None, summary=False, parents=()):
    return {
        "key": key,
        "title": title or key,
        "description": f"description for {key}",
        "branch": "fusion",
        "column": 2.0,
        "parents": parents,
        "summary": summary,
        "source": "actual forward",
        "values": values,
    }


def _post(server, payload, *, origin=None, path="/api/control"):
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    request = Request(
        server.url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    return urlopen(request, timeout=3.0)


def test_page_replaces_viewer_url_and_get_marks_a_recent_client(http_server):
    server, _commands = http_server

    assert server.has_clients() is False
    with urlopen(server.url, timeout=3.0) as response:
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert response.headers.get_content_type() == "text/html"
    assert "href=\"http://127.0.0.1:8769/\"" in page
    assert "__VIEWER_URL__" not in page
    assert server.has_clients() is True

    with server._lock:
        server._last_get = time.monotonic() - 3.1
    assert server.has_clients() is False


def test_fusion_dimension_is_carried_from_the_live_frame_and_html_label_is_dynamic(http_server):
    server, _commands = http_server
    server.publish(
        {"sequence": 4},
        [_layer("features_extractor", np.zeros(160, dtype=np.float32),
                title="四路特征拼接 · 160维", summary=True)],
    )

    payload = _get_json(server)
    fusion = next(node for node in payload["nodes"] if node["key"] == "features_extractor")
    assert fusion["shape"] == [160]
    assert fusion["title"] == "四路特征拼接 · 160维"

    html_source = Path(network_2d.__file__).with_name("network_2d.html").read_text(encoding="utf-8")
    assert "融合 · 128维" not in html_source
    assert "function fusionLabel(node)" in html_source


def test_frame_is_detached_consistent_and_contains_exact_values_and_png(http_server):
    server, _commands = http_server
    values = np.asarray([1.0, -2.0, 7.0, np.nan, np.inf], dtype=np.float32)
    snapshot = {
        "sequence": 12,
        "sim_time": 1.25,
        "output_time": 1.30,
        "mode": "actual policy forward",
        "critic": "diagnostic branch",
        "errors": (),
    }
    state = {"running": True, "time": 1.25, "robot": {"x": 0.3}}
    layers = [
        _layer("input.state", np.asarray([0.1, 0.2], dtype=np.float32)),
        _layer("features_extractor", values, summary=True, parents=("input.state",)),
        _layer("features_extractor.hidden", np.asarray([4.0]), parents=("features_extractor",)),
    ]
    server.publish(snapshot, layers, state)

    # Later simulator writes must not mutate the HTTP-visible frame.
    values[:] = 99.0
    snapshot["sequence"] = 99
    state["running"] = False
    state["robot"]["x"] = 42.0
    layers[1]["title"] = "mutated title"

    payload = _get_json(
        server,
        "/api/frame?detail=0&selected=features_extractor&tensor=%E6%BF%80%E6%B4%BB%E5%80%BC&index=2&scale=auto",
    )
    assert payload["sequence"] == 12
    assert payload["sim_time"] == 1.25
    assert payload["output_time"] == 1.3
    assert payload["legend"]["name"] == "turbo"
    assert payload["legend"]["minimum"] == -7.0
    assert payload["legend"]["zero"] == 0.0
    assert payload["legend"]["maximum"] == 7.0
    assert payload["legend"]["mode"] == "adaptive-global"
    assert payload["legend"]["scale"] == 7.0
    assert payload["legend"]["observed_max_abs"] == 7.0
    assert len(payload["legend"]["colors"]) == 9
    assert payload["legend"]["colors"][0] == [48, 18, 59]
    assert payload["state"] == {"running": True, "time": 1.25, "robot": {"x": 0.3}}
    assert [node["key"] for node in payload["nodes"]] == ["input.state", "features_extractor"]
    selected = payload["detail"]
    assert selected["title"] == "features_extractor"
    assert selected["shape"] == [5]
    assert selected["size"] == 5
    assert selected["min"] == -2.0
    assert selected["max"] == 7.0
    assert selected["mean"] == 2.0
    assert selected["nonfinite"] == 2
    assert selected["index"] == 2
    assert selected["coordinate"] == [2]
    assert selected["value"] == 7.0
    assert selected["description"] == "description for features_extractor"
    assert selected["source"] == "actual forward"
    assert selected["parameters"] == []
    assert selected["image"].startswith("data:image/png;base64,")
    image_bytes = base64.b64decode(selected["image"].split(",", 1)[1])
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert image.format == "PNG"
        assert image.size == (3, 2)
    assert payload["nodes"][0]["shape"] == [2]
    assert payload["nodes"][0]["image"].startswith("data:image/png;base64,")

    nonfinite = _get_json(
        server,
        "/api/frame?detail=1&selected=features_extractor&index=3&scale=auto",
    )["detail"]
    assert nonfinite["value"] == "NaN"
    assert [node["key"] for node in _get_json(server, "/api/frame?detail=1")["nodes"]] == [
        "input.state",
        "features_extractor",
        "features_extractor.hidden",
    ]


def test_sensor_input_shapes_and_state_field_labels_match_simplified_contract(http_server):
    server, _commands = http_server
    depth = np.zeros((1, 160, 88), dtype=np.float32)
    state = np.arange(43, dtype=np.float32)
    server.publish(
        {"sequence": 1},
        [
            _layer("input.depth", depth, title="深度图输入"),
            _layer("input.state", state, title="车辆与目标状态输入"),
        ],
    )

    depth_detail = _get_json(server, "/api/frame?selected=input.depth&detail=1")["detail"]
    assert depth_detail["shape"] == [1, 160, 88]
    assert depth_detail["image"].startswith("data:image/png;base64,")
    depth_pixels = base64.b64decode(depth_detail["image"].split(",", 1)[1])
    with Image.open(io.BytesIO(depth_pixels)) as image:
        assert image.size == (88, 160)

    state_detail = _get_json(server, "/api/frame?selected=input.state&detail=1")["detail"]
    assert state_detail["shape"] == [43]
    assert state_detail["fields"] == list(STATE_LABELS)
    assert len(state_detail["fields"]) == 43
    assert state_detail["fields"][41:] == ["车体横滚角 / π", "车体俯仰角 / π"]
    assert all("噪声配置" not in label for label in state_detail["fields"])

    with pytest.raises(HTTPError) as removed_scaled_node:
        urlopen(server.url.rstrip("/") + "/api/frame?selected=state.scaled", timeout=3.0)
    assert removed_scaled_node.value.code == 400


def test_uncollected_layer_has_null_shape_and_image(http_server):
    server, _commands = http_server
    server.publish({"sequence": 1}, [_layer("input.state", None)])

    payload = _get_json(server, "/api/frame?selected=input.state")
    assert payload["nodes"] == [
        {
            "key": "input.state",
            "title": "input.state",
            "description": "description for input.state",
            "branch": "fusion",
            "column": 2.0,
            "parents": [],
            "summary": False,
            "source": "actual forward",
            "shape": None,
            "image": None,
        }
    ]
    assert payload["detail"]["shape"] is None
    assert payload["detail"]["size"] is None
    assert payload["detail"]["index"] is None
    assert payload["detail"]["coordinate"] is None
    assert payload["detail"]["value"] is None
    assert payload["detail"]["image"] is None


def test_bind_copies_parameters_and_cache_tracks_revision_when_sequence_repeats(http_server):
    server, _commands = http_server
    weight = np.asarray([[1.0, 2.0], [7.0, 4.0]], dtype=np.float32)

    class Inspector:
        layers = {"dense": {}}

        def parameters(self, key):
            assert key == "dense"
            return {"weight": weight, "bias": np.asarray([0.25, -0.5], dtype=np.float32)}

    server.bind(Inspector())
    weight[:] = -100.0
    server.publish({"sequence": 3}, [_layer("dense", np.asarray([1.0]))])
    detail = _get_json(server, "/api/frame?selected=dense&tensor=weight&index=2")["detail"]
    assert detail["parameters"] == ["bias", "weight"]
    assert detail["shape"] == [2, 2]
    assert detail["coordinate"] == [1, 0]
    assert detail["value"] == 7.0

    first_image = _get_json(server, "/api/frame?selected=dense")["detail"]["image"]
    server.publish({"sequence": 3}, [_layer("dense", np.asarray([-1.0]))])
    second_image = _get_json(server, "/api/frame?selected=dense")["detail"]["image"]
    assert first_image != second_image
    assert _get_json(server)["legend"]["observed_max_abs"] == 7.0


def test_global_range_includes_hidden_layers_all_parameters_and_is_selection_independent(http_server):
    server, _commands = http_server
    selected_weight = np.asarray([[15.0, -15.0]], dtype=np.float32)
    unused_weight = np.asarray([240.0], dtype=np.float32)

    class Inspector:
        layers = {"dense": {}, "unused": {}}

        def parameters(self, key):
            return {"weight": selected_weight if key == "dense" else unused_weight}

    server.bind(Inspector())
    selected_weight[:] = 999.0
    unused_weight[:] = -999.0
    server.publish(
        {"sequence": 1, "sim_time": 0.1},
        [
            _layer("small-vector", np.asarray([15.0], dtype=np.float32), summary=True),
            _layer("small-matrix", np.asarray([[15.0]], dtype=np.float32), summary=True),
            _layer("dense", np.asarray([-10.0], dtype=np.float32), summary=True),
            _layer("unused", np.asarray([1.0], dtype=np.float32)),
            _layer("dense.hidden", np.asarray([100.0], dtype=np.float32)),
            _layer("feedback.physical", np.asarray([-180.0], dtype=np.float32)),
        ],
    )

    vector = _get_json(server, "/api/frame?detail=1&selected=small-vector")
    matrix = _get_json(server, "/api/frame?detail=1&selected=small-matrix")
    parameter = _get_json(server, "/api/frame?detail=1&selected=dense&tensor=weight")
    unused_parameter = _get_json(server, "/api/frame?detail=1&selected=unused&tensor=weight")
    for payload in (vector, matrix, parameter, unused_parameter):
        assert payload["legend"]["maximum"] == 240.0
        assert payload["legend"]["observed_max_abs"] == 240.0
        assert payload["legend"]["mode"] == "adaptive-global"

    # Equal values receive equal Turbo colors across different shapes and tensor kinds.
    vector_pixel = _image_pixels(vector["detail"]["image"])[0, 0]
    matrix_pixel = _image_pixels(matrix["detail"]["image"])[0, 0]
    parameter_pixel = _image_pixels(parameter["detail"]["image"])[0, 0]
    np.testing.assert_array_equal(vector_pixel, matrix_pixel)
    np.testing.assert_array_equal(vector_pixel, parameter_pixel)
    assert parameter["detail"]["value"] == 15.0  # bind kept a detached copy
    assert unused_parameter["detail"]["value"] == 240.0

    # A hidden activation still expands the global range even while detail=0 hides it.
    server.publish(
        {"sequence": 2, "sim_time": 0.2},
        [
            _layer("small-vector", np.asarray([15.0], dtype=np.float32), summary=True),
            _layer("dense.hidden", np.asarray([300.0], dtype=np.float32)),
        ],
    )
    expanded = _get_json(server, "/api/frame?detail=0&selected=small-vector")
    assert expanded["legend"]["maximum"] == 300.0
    assert expanded["legend"]["observed_max_abs"] == 300.0
    assert "dense.hidden" not in {node["key"] for node in expanded["nodes"]}

    server.publish(
        {"sequence": 3, "sim_time": 0.3},
        [
            _layer("small-vector", np.asarray([15.0], dtype=np.float32), summary=True),
            _layer("critic.diagnostic", np.asarray([350.0], dtype=np.float32)),
        ],
    )
    diagnostic = _get_json(server, "/api/frame?detail=0&selected=small-vector")
    assert diagnostic["legend"]["maximum"] == 350.0
    assert diagnostic["legend"]["observed_max_abs"] == 350.0

    server.publish(
        {"sequence": 4, "sim_time": 0.4},
        [
            _layer("small-vector", np.asarray([15.0], dtype=np.float32), summary=True),
            _layer("feedback.physical", np.asarray([-400.0], dtype=np.float32)),
        ],
    )
    feedback = _get_json(server, "/api/frame?detail=0&selected=small-vector")
    assert feedback["legend"]["maximum"] == 400.0
    assert feedback["legend"]["observed_max_abs"] == 400.0


def test_adaptive_range_expands_immediately_and_only_shrinks_on_new_frame_token(http_server):
    server, _commands = http_server
    server.publish({"sequence": 1, "sim_time": 1.0}, [_layer("dense", np.asarray([100.0]))])
    assert _get_json(server)["legend"]["maximum"] == 100.0

    server.publish({"sequence": 1, "sim_time": 1.0}, [_layer("dense", np.asarray([1.0]))])
    paused_first = _get_json(server, "/api/frame?selected=dense")["legend"]["maximum"]
    server.publish({"sequence": 1, "sim_time": 1.0}, [_layer("dense", np.asarray([1.0]))])
    paused_second = _get_json(server, "/api/frame?selected=dense&detail=1")["legend"]["maximum"]
    assert paused_first == paused_second == 100.0

    server.publish({"sequence": 2, "sim_time": 1.1}, [_layer("dense", np.asarray([1.0]))])
    assert _get_json(server)["legend"]["maximum"] == pytest.approx(95.0)
    server.publish({"sequence": 2, "sim_time": 1.1}, [_layer("dense", np.asarray([1.0]))])
    assert _get_json(server)["legend"]["maximum"] == pytest.approx(95.0)

    server.publish({"sequence": 2, "sim_time": 1.1}, [_layer("dense", np.asarray([120.0]))])
    assert _get_json(server)["legend"]["maximum"] == 120.0
    server.publish({"sequence": 3, "sim_time": 1.2}, [_layer("dense", np.asarray([1.0]))])
    assert _get_json(server)["legend"]["maximum"] == pytest.approx(114.0)


def test_round_rollback_snapshot_none_and_rebind_reset_the_range(http_server):
    server, _commands = http_server
    server.publish({"sequence": 10, "sim_time": 10.0}, [_layer("dense", np.asarray([100.0]))])
    server.publish({"sequence": 11, "sim_time": 11.0}, [_layer("dense", np.asarray([30.0]))])
    assert _get_json(server)["legend"]["maximum"] == pytest.approx(95.0)

    server.publish({"sequence": 0, "sim_time": 0.0}, [_layer("dense", np.asarray([20.0]))])
    assert _get_json(server)["legend"]["maximum"] == 20.0
    server.publish(None, [_layer("dense", np.asarray([2.0]))])
    assert _get_json(server)["legend"]["maximum"] == 2.0
    server.publish({"sequence": 1, "sim_time": 0.1}, [_layer("dense", np.asarray([1.0]))])
    assert _get_json(server)["legend"]["maximum"] == 1.0

    class Inspector:
        layers = {"dense": {}}

        def __init__(self, weight):
            self.weight = np.asarray([weight], dtype=np.float32)

        def parameters(self, _key):
            return {"weight": self.weight}

    server.bind(Inspector(8.0))
    bound = _get_json(server)
    assert bound["legend"]["maximum"] == 8.0
    assert bound["legend"]["observed_max_abs"] == 8.0
    server.publish({"sequence": 50, "sim_time": 5.0}, [_layer("dense", np.asarray([4.0]))])
    assert _get_json(server)["legend"]["maximum"] == 8.0

    server.bind(Inspector(3.0))
    rebound = _get_json(server)
    assert rebound["legend"]["maximum"] == 3.0
    assert rebound["legend"]["observed_max_abs"] == 3.0


def test_extreme_finite_values_and_all_nonfinite_frames_have_safe_ranges(http_server):
    server, _commands = http_server
    extreme = float(np.finfo(np.float32).max)
    server.publish(
        {"sequence": 1, "sim_time": 0.0},
        [_layer("dense", np.asarray([extreme, -extreme, np.nan, np.inf, -np.inf], dtype=np.float32))],
    )
    payload = _get_json(server, "/api/frame?selected=dense&detail=1")
    assert payload["legend"]["maximum"] == extreme
    assert payload["legend"]["observed_max_abs"] == extreme
    pixels = _image_pixels(payload["detail"]["image"])
    endpoint = network_2d.activation_image(np.asarray([1.0]), scale=1.0)[0][0, 0]
    assert tuple(pixels.reshape(-1, 3)[0]) == tuple(endpoint)
    assert tuple(pixels.reshape(-1, 3)[2]) == (255, 0, 255)

    server.bind(None)
    server.publish(
        {"sequence": 1, "sim_time": 0.0},
        [
            _layer("zeros", np.zeros(3, dtype=np.float32)),
            _layer("nonfinite", np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)),
        ],
    )
    safe = _get_json(server, "/api/frame?selected=nonfinite")
    assert safe["legend"]["maximum"] == 1.0
    assert safe["legend"]["observed_max_abs"] == 0.0
    assert safe["detail"]["value"] == "NaN"


def test_png_cache_reuses_one_encoding_until_a_new_frame(monkeypatch, http_server):
    server, _commands = http_server
    original = network_2d.activation_image
    calls = []

    def counted(values, *, scale=None):
        calls.append((np.asarray(values).shape, scale))
        return original(values, scale=scale)

    monkeypatch.setattr(network_2d, "activation_image", counted)
    server.publish({"sequence": 4}, [_layer("dense", np.asarray([0.25, -0.5]), summary=True)])

    _get_json(server, "/api/frame?selected=dense&scale=auto")
    _get_json(server, "/api/frame?selected=dense&scale=auto")
    assert len(calls) == 1  # node and detail share the same cached image
    assert calls[0][1] == 0.5

    server.publish({"sequence": 4}, [_layer("dense", np.asarray([0.5, -0.5]), summary=True)])
    _get_json(server, "/api/frame?selected=dense&scale=auto")
    assert len(calls) == 2


def test_control_mapping_same_origin_and_input_errors(http_server):
    server, commands = http_server
    origin = f"http://{urlsplit(server.url).netloc}"
    for control, queued in (("run", "network_run"), ("pause", "network_pause"),
                            ("reset", "reset"), ("preview", "network_preview")):
        with _post(server, {"command": control}, origin=origin) as response:
            assert response.status == 200
            assert json.loads(response.read())["command"] == control
        assert commands[-1] == queued

    with _post(server, {"command": "run"}) as response:
        assert response.status == 200  # missing Origin is permitted for local clients/tests
    assert commands[-1] == "network_run"

    with pytest.raises(HTTPError) as cross_origin:
        _post(server, {"command": "run"}, origin="http://example.com")
    assert cross_origin.value.code == 400
    with pytest.raises(HTTPError) as unsupported:
        _post(server, {"command": "teleport"})
    assert unsupported.value.code == 400
    with pytest.raises(HTTPError) as missing_get:
        urlopen(server.url.rstrip("/") + "/api/missing", timeout=3.0)
    assert missing_get.value.code == 404
    with pytest.raises(HTTPError) as missing:
        _post(server, {"command": "run"}, path="/api/other")
    assert missing.value.code == 404
    with pytest.raises(HTTPError) as bad_scale:
        urlopen(server.url.rstrip("/") + "/api/frame?scale=nan", timeout=3.0)
    assert bad_scale.value.code == 400
    with pytest.raises(HTTPError) as manual_scale:
        urlopen(server.url.rstrip("/") + "/api/frame?scale=10", timeout=3.0)
    assert manual_scale.value.code == 400
    with pytest.raises(HTTPError) as bad_index:
        urlopen(server.url.rstrip("/") + "/api/frame?index=-1", timeout=3.0)
    assert bad_index.value.code == 400
