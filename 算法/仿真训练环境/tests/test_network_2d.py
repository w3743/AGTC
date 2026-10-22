"""Local HTTP contract tests for the independent 2D network view."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

from isaac_field.action_contract import ACTION_LABELS
from isaac_field.feedback import FEEDBACK_LABELS
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


def _layer(key, values, *, title=None, summary=False, parents=(), branch="fusion"):
    return {
        "key": key,
        "title": title or key,
        "description": f"description for {key}",
        "branch": branch,
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


def test_rsl_graph_uses_actual_server_default_and_rejects_sb3_alias(http_server):
    server, _commands = http_server
    server.publish(
        {"sequence": 1, "mode": "RSL actor forward"},
        [
            _layer("rsl.actor.input", np.zeros(48, dtype=np.float32), branch="input"),
            _layer("rsl.actor.encoder", np.zeros(192, dtype=np.float32), summary=True),
            _layer("rsl.actor.lstm", np.zeros((1, 512), dtype=np.float32)),
        ],
    )

    # The first request intentionally omits `selected`; the server chooses a
    # real key from this frame instead of relying on an SB3-only default.
    payload = _get_json(server, "/api/frame?detail=1")
    actual_keys = {node["key"] for node in payload["nodes"]}
    assert actual_keys == {"rsl.actor.input", "rsl.actor.encoder", "rsl.actor.lstm"}
    assert payload["detail"]["key"] in actual_keys

    with pytest.raises(HTTPError) as unknown_sb3_layer:
        urlopen(server.url.rstrip("/") + "/api/frame?selected=features_extractor", timeout=3.0)
    assert unknown_sb3_layer.value.code == 400


def test_network_html_adopts_frame_selection_and_bounds_unknown_key_retry():
    html_source = Path(network_2d.__file__).with_name("network_2d.html").read_text(encoding="utf-8")

    assert 'let selectedKey = null;' in html_source
    assert 'if (includeSelected && selectedKey !== null) query.set("selected", selectedKey);' in html_source
    assert 'if (response.status === 400 && requestedKey !== null)' in html_source
    assert 'errorMessage === "selected layer is unknown"' in html_source
    assert 'response = await fetch(queryUrl({ includeSelected: false }), requestOptions);' in html_source
    assert 'const detailKey = frame.detail && frame.detail.key;' in html_source
    assert 'frameKeys.has(detailKey) ? detailKey' in html_source


def test_network_html_400_handling_only_retries_unknown_selection_and_preserves_other_inputs():
    html_path = Path(network_2d.__file__).with_name("network_2d.html")
    node = shutil.which("node") or next(
        (candidate for candidate in (
            Path("/mnt/c/Program Files/nodejs/node.exe"),
            Path(r"C:\Program Files\nodejs\node.exe"),
        ) if candidate.is_file()),
        None,
    )
    if node is None:
        pytest.skip("Node.js is required for the network fetch behavior harness")

    harness = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const html = fs.readFileSync(process.argv[1], "utf8");
function extract(name, nextName) {
  const start = html.indexOf("    " + name);
  const end = html.indexOf("    " + nextName, start + 1);
  if (start < 0 || end < 0) throw new Error("Could not find " + name);
  return html.slice(start, end).trim();
}
const queryUrlSource = extract("function queryUrl(", "async function responseErrorMessage(");
const responseSource = extract("async function responseErrorMessage(", "async function pollOnce(");
  const responseFns = responseSource;
if (!responseFns.includes("async function fetchFrameResponse(")) {
  throw new Error("fetchFrameResponse was not extracted");
}

function makeResponse(status, body) {
  return { status, ok: status >= 200 && status < 300, json: async () => body };
}
async function runCase(name, statuses, expected) {
  const calls = [];
  const context = {
    URLSearchParams,
    els: { detailLocalScale: { checked: false } },
    fetch: async url => { calls.push(url); return statuses.shift(); }
  };
  vm.runInNewContext(`
    let selectedKey = "features_extractor";
    let tensorName = "weight";
    let elementIndex = 19;
    let resets = 0;
    function resetDetailCursor() { resets += 1; tensorName = "激活值"; elementIndex = 0; }
    ${queryUrlSource}
    ${responseFns}
    globalThis.invoke = () => fetchFrameResponse({}, selectedKey);
    globalThis.snapshot = () => ({ selectedKey, tensorName, elementIndex, resets });
  `, context);
  let error = "";
  try { await context.invoke(); } catch (caught) { error = String(caught.message || caught); }
  const state = context.snapshot();
  if (calls.length !== expected.calls) throw new Error(name + ": unexpected fetch count " + calls.length);
  if (state.resets !== expected.resets) throw new Error(name + ": unexpected reset count " + state.resets);
  if (error !== expected.error) throw new Error(name + ": unexpected error " + JSON.stringify(error));
  if (state.tensorName !== expected.tensorName || state.elementIndex !== expected.elementIndex) {
    throw new Error(name + ": detail cursor changed unexpectedly");
  }
  if (expected.fallback && !new URL(calls[0], "http://local").searchParams.has("selected")) {
    throw new Error(name + ": initial request omitted its selected key");
  }
  if (expected.fallback && new URL(calls[1], "http://local").searchParams.has("selected")) {
    throw new Error(name + ": fallback still includes selected");
  }
}

(async () => {
  await runCase("unknown selected", [
    makeResponse(400, { error: "selected layer is unknown" }),
    makeResponse(200, { nodes: [], detail: null })
  ], { calls: 2, resets: 1, error: "", tensorName: "激活值", elementIndex: 0, fallback: true });
  await runCase("invalid index", [
    makeResponse(400, { error: "index is outside the selected tensor" })
  ], { calls: 1, resets: 0, error: "index is outside the selected tensor", tensorName: "weight", elementIndex: 19 });
  await runCase("invalid tensor", [
    makeResponse(400, { error: "tensor parameter is unavailable for selected layer" })
  ], { calls: 1, resets: 0, error: "tensor parameter is unavailable for selected layer", tensorName: "weight", elementIndex: 19 });
  console.log("network 2D fetch behavior: unknown selection retries once; invalid index/tensor preserve cursor and server error");
})().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
"""
    node_html_path = str(html_path)
    if str(node).lower().endswith(".exe") and html_path.as_posix().startswith("/mnt/"):
        drive, relative = html_path.as_posix()[5:].split("/", 1)
        node_html_path = drive.upper() + ":\\" + relative.replace("/", "\\")
    completed = subprocess.run(
        [str(node), "-e", harness, node_html_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "unknown selection retries once" in completed.stdout


def test_feedback_layer_exposes_the_complete_37_field_contract(http_server):
    server, _commands = http_server
    server.publish(
        {"sequence": 9},
        [_layer("input.feedback", np.arange(37, dtype=np.float32),
                title="云台IMU与驱动反馈 · 37维", branch="feedback")],
    )

    payload = _get_json(server, "/api/frame?detail=1&selected=input.feedback")
    feedback = next(node for node in payload["nodes"] if node["key"] == "input.feedback")
    assert feedback["shape"] == [37]
    assert payload["detail"]["fields"] == list(FEEDBACK_LABELS)
    assert len(payload["detail"]["fields"]) == 37


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


def test_local_detail_scale_uses_selected_activation_and_keeps_global_nodes_unchanged(http_server):
    server, _commands = http_server

    class Inspector:
        layers = {"dense": {}}

        def parameters(self, _key):
            return {"weight": np.asarray([64.0], dtype=np.float32)}

    server.bind(Inspector())
    values = np.asarray([1.0, -2.0, np.nan, np.inf], dtype=np.float32)
    server.publish({"sequence": 1}, [_layer("dense", values, summary=True)])

    global_payload = _get_json(server, "/api/frame?detail=1&selected=dense")
    local_payload = _get_json(server, "/api/frame?detail=1&selected=dense&detail_scale=local")
    global_detail = global_payload["detail"]
    local_detail = local_payload["detail"]

    assert global_detail["tensor"] == local_detail["tensor"] == "激活值"
    assert global_detail["legend"]["mode"] == "adaptive-global"
    assert global_detail["legend"]["maximum"] == 64.0
    assert local_detail["legend"]["mode"] == "adaptive-local"
    assert local_detail["legend"]["minimum"] == -2.0
    assert local_detail["legend"]["zero"] == 0.0
    assert local_detail["legend"]["maximum"] == local_detail["legend"]["scale"] == 2.0
    assert local_detail["legend"]["observed_max_abs"] == 2.0
    assert local_detail["legend"]["name"] == global_payload["legend"]["name"] == "turbo"
    assert local_detail["legend"]["colors"] == global_payload["legend"]["colors"]
    assert local_payload["legend"] == global_payload["legend"]
    assert local_payload["nodes"] == global_payload["nodes"]
    assert global_detail["image"] == global_payload["nodes"][0]["image"]
    expected, _indices, _scale = network_2d.activation_image(values, scale=2.0)
    np.testing.assert_array_equal(_image_pixels(local_detail["image"]), expected)


def test_local_detail_scale_tracks_selected_weight_and_bias(http_server):
    server, _commands = http_server
    tensors = {
        "weight": np.asarray([[4.0, -2.0]], dtype=np.float32),
        "bias": np.asarray([0.25, -0.5], dtype=np.float32),
    }

    class Inspector:
        layers = {"dense": {}, "wide": {}}

        def parameters(self, key):
            return tensors if key == "dense" else {"weight": np.asarray([100.0], dtype=np.float32)}

    server.bind(Inspector())
    server.publish(
        {"sequence": 1},
        [_layer("dense", np.asarray([8.0], dtype=np.float32), summary=True),
         _layer("wide", np.asarray([40.0], dtype=np.float32))],
    )

    for tensor, values, expected_scale in (
        ("weight", tensors["weight"], 4.0),
        ("bias", tensors["bias"], 0.5),
    ):
        payload = _get_json(
            server,
            f"/api/frame?detail=1&selected=dense&tensor={tensor}&detail_scale=local",
        )
        detail = payload["detail"]
        assert detail["tensor"] == tensor
        assert detail["legend"]["mode"] == "adaptive-local"
        assert detail["legend"]["scale"] == expected_scale
        assert detail["legend"]["maximum"] == expected_scale
        assert detail["legend"]["observed_max_abs"] == expected_scale
        assert payload["legend"]["maximum"] == 100.0
        expected, _indices, _scale = network_2d.activation_image(values, scale=expected_scale)
        np.testing.assert_array_equal(_image_pixels(detail["image"]), expected)


def test_local_detail_scale_uses_safe_fallback_for_zero_and_nonfinite_tensors(http_server):
    server, _commands = http_server
    zero_values = np.zeros(3, dtype=np.float32)
    nonfinite_values = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
    server.publish(
        {"sequence": 1},
        [_layer("zero", zero_values), _layer("nonfinite", nonfinite_values)],
    )

    for key, values in (("zero", zero_values), ("nonfinite", nonfinite_values)):
        detail = _get_json(
            server,
            f"/api/frame?selected={key}&detail_scale=local",
        )["detail"]
        assert detail["legend"]["mode"] == "adaptive-local"
        assert detail["legend"]["observed_max_abs"] == 0.0
        assert detail["legend"]["minimum"] == -1.0
        assert detail["legend"]["maximum"] == detail["legend"]["scale"] == 1.0
        expected, _indices, _scale = network_2d.activation_image(values, scale=1.0)
        np.testing.assert_array_equal(_image_pixels(detail["image"]), expected)


def test_sensor_input_shapes_and_state_field_labels_match_simplified_contract(http_server):
    server, _commands = http_server
    depth = np.zeros((1, 160, 88), dtype=np.float32)
    state = np.arange(33, dtype=np.float32)
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
    assert state_detail["shape"] == [33]
    assert state_detail["fields"] == list(STATE_LABELS)
    assert len(state_detail["fields"]) == 33
    assert not any("车体横滚角" in field or "车体俯仰角" in field for field in state_detail["fields"])
    assert all("噪声配置" not in label for label in state_detail["fields"])

    with pytest.raises(HTTPError) as removed_scaled_node:
        urlopen(server.url.rstrip("/") + "/api/frame?selected=state.scaled", timeout=3.0)
    assert removed_scaled_node.value.code == 400


def test_action_node_exposes_direct_track_component_order_as_output_endpoint(http_server):
    server, _commands = http_server
    description = (
        "动作分量依次为" + "、".join(ACTION_LABELS)
        + "；前两维分别乘最大履带速度，形成左右独立履带速度指令。"
    )
    action = _layer("action.applied", np.asarray([0.2, -0.3, 0.1, -0.2]), title="四维输出动作", summary=True)
    action["description"] = description
    server.publish({"sequence": 1}, [action])

    detail = _get_json(server, "/api/frame?selected=action.applied&detail=1")["detail"]
    assert detail["fields"] == list(ACTION_LABELS)
    assert detail["description"] == description
    nodes = _get_json(server, "/api/frame?detail=1")["nodes"]
    assert {node["key"] for node in nodes} == {"action.applied"}


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
    server.publish(
        {"sequence": 4},
        [_layer("dense", np.asarray([0.25, -0.5]), summary=True),
         _layer("wide", np.asarray([10.0]))],
    )

    _get_json(server, "/api/frame?selected=dense&scale=auto")
    _get_json(server, "/api/frame?selected=dense&scale=auto")
    assert len(calls) == 1  # node and detail share the same cached image
    assert calls[0][1] == 10.0

    _get_json(server, "/api/frame?selected=dense&detail_scale=local")
    _get_json(server, "/api/frame?selected=dense&detail_scale=local")
    assert len(calls) == 2  # local detail image is distinct, then cached
    assert calls[1][1] == 0.5

    server.publish(
        {"sequence": 4},
        [_layer("dense", np.asarray([0.5, -0.5]), summary=True),
         _layer("wide", np.asarray([10.0]))],
    )
    _get_json(server, "/api/frame?selected=dense&scale=auto")
    assert len(calls) == 3


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
    with pytest.raises(HTTPError) as bad_detail_scale:
        urlopen(server.url.rstrip("/") + "/api/frame?detail_scale=automatic", timeout=3.0)
    assert bad_detail_scale.value.code == 400
    with pytest.raises(HTTPError) as bad_index:
        urlopen(server.url.rstrip("/") + "/api/frame?index=-1", timeout=3.0)
    assert bad_index.value.code == 400
