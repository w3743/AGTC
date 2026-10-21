"""Capture every raw activation node from a paused live Isaac Field frame.

Only GET requests are issued. The selected scalar for each actual node, its
raw stats, node metadata and the shared auto-colour legend are saved as JSON;
render-only PNG data is stripped before writing evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch(base_url: str, query: dict[str, str]) -> dict:
    url = f"{base_url.rstrip('/')}/api/frame?{urlencode(query)}"
    with urlopen(url, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def strip_png(value):
    """Remove data-URI PNG fields recursively while preserving other JSON."""
    if isinstance(value, dict):
        return {
            key: strip_png(item)
            for key, item in value.items()
            if key != "image"
        }
    if isinstance(value, list):
        return [strip_png(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def same_shared_legend(reference: dict, candidate: dict) -> bool:
    keys = ("name", "minimum", "zero", "maximum", "mode", "scale", "observed_max_abs")
    return all(reference.get(key) == candidate.get(key) for key in keys)


def require_paused_frame(frame: dict, sequence, sim_time) -> None:
    if frame.get("state", {}).get("running") is not False:
        raise RuntimeError("refusing capture: viewer is not paused")
    if frame.get("sequence") != sequence or frame.get("sim_time") != sim_time:
        raise RuntimeError("frame advanced during read-only capture; discard and retry after pause")


def capture(base_url: str, out_dir: Path, prefix: str) -> dict:
    # Keep the same designated selection used by the previous approved capture.
    first = fetch(base_url, {"detail": "1", "selected": "action.applied", "scale": "auto"})
    if first.get("state", {}).get("running") is not False:
        raise RuntimeError("refusing capture: viewer is not paused")
    if first.get("detail", {}).get("key") != "action.applied":
        raise RuntimeError("designated frame did not select action.applied")

    nodes = first.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("frame has no actual node list")
    keys = [node.get("key") for node in nodes]
    if any(not isinstance(key, str) or not key for key in keys):
        raise RuntimeError("actual node list contains a missing/invalid key")
    if len(keys) != len(set(keys)):
        raise RuntimeError("actual node list contains duplicate keys")

    sequence = first.get("sequence")
    sim_time = first.get("sim_time")
    legend = first.get("legend")
    if not isinstance(legend, dict):
        raise RuntimeError("frame did not expose shared legend metadata")

    # Select the 41-value feedback input from actual branch/shape metadata;
    # do not depend on a fixed list of network layer names for the node sweep.
    feedback_candidates = [
        node for node in nodes
        if node.get("branch") == "feedback"
        and node.get("shape") == [41]
    ]
    if len(feedback_candidates) != 1:
        raise RuntimeError(
            f"expected one actual feedback input node with shape [41], found {len(feedback_candidates)}"
        )
    feedback_node = feedback_candidates[0]

    tensors: dict[str, dict] = {}
    observed_legends = []
    detail_frames = {}

    # Query every node returned by frame.nodes, in order, using its own key.
    for node in nodes:
        key = node["key"]
        detail_frame = fetch(
            base_url,
            {"detail": "1", "selected": key, "index": "0", "scale": "auto"},
        )
        require_paused_frame(detail_frame, sequence, sim_time)
        detail = detail_frame.get("detail") or {}
        if detail.get("key") != key:
            raise RuntimeError(f"detail response key mismatch: requested {key!r}, got {detail.get('key')!r}")
        call_legend = detail_frame.get("legend")
        if not isinstance(call_legend, dict) or not same_shared_legend(legend, call_legend):
            raise RuntimeError(f"shared auto colour scale changed during paused capture at {key}")
        observed_legends.append(call_legend)
        detail_frames[key] = detail_frame
        tensors[key] = {
            "node": strip_png(deepcopy(node)),
            "query": {"selected": key, "index": 0, "scale": "auto"},
            "detail": strip_png(detail),
        }

    # Preserve exact values for all 41 feedback fields, especially the eight
    # validity flags and eight ages, by querying each selected scalar. Index 0
    # was already fetched in the all-node sweep.
    feedback_key = feedback_node["key"]
    feedback_first = tensors[feedback_key]["detail"]
    fields = feedback_first.get("fields")
    if not isinstance(fields, list) or len(fields) != 41:
        raise RuntimeError("feedback detail did not expose all 41 field labels")
    feedback_scalars = []
    for index in range(41):
        if index == 0:
            detail = feedback_first
        else:
            detail_frame = fetch(
                base_url,
                {
                    "detail": "1",
                    "selected": feedback_key,
                    "index": str(index),
                    "scale": "auto",
                },
            )
            require_paused_frame(detail_frame, sequence, sim_time)
            detail = detail_frame.get("detail") or {}
            if detail.get("key") != feedback_key or detail.get("index") != index:
                raise RuntimeError(f"feedback scalar index mismatch at {index}")
            call_legend = detail_frame.get("legend")
            if not isinstance(call_legend, dict) or not same_shared_legend(legend, call_legend):
                raise RuntimeError(f"shared auto colour scale changed during feedback scalar capture at {index}")
            observed_legends.append(call_legend)
        value = detail.get("value")
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"feedback scalar {index} is not numeric")
        feedback_scalars.append({
            "index": index,
            "field": fields[index],
            "value": value,
        })

    legends_equal = all(same_shared_legend(legend, item) for item in observed_legends)
    if not legends_equal:
        raise RuntimeError("not all detail GETs used the same shared auto colour scale")
    if not all(math.isfinite(float(legend.get(k))) for k in ("minimum", "zero", "maximum", "scale")):
        raise RuntimeError("shared legend contains a nonfinite numeric bound")
    if legend.get("zero") != 0 or not math.isclose(
        float(legend["minimum"]), -float(legend["maximum"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("shared auto colour scale is not finite, symmetric and zero-centred")

    # Keep the complete node/parent metadata and frame state, but remove every
    # render-only image field from the evidence JSON.
    frame = strip_png(deepcopy(first))
    frame["capture_metadata"] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "endpoint": "/api/frame",
        "method": "GET only; no vehicle control or state mutation",
        "paused": True,
        "sequence": sequence,
        "sim_time": sim_time,
        "node_keys_source": "live frame.nodes; every returned key queried dynamically",
        "node_count": len(nodes),
        "detail_get_count": len(nodes) + 40,
        "all_detail_reads_same_sequence_and_time": True,
        "feedback_key": feedback_key,
        "feedback_scalar_indices_captured": [0, 40],
        "feedback_validity_indices": [25, 32],
        "feedback_age_indices": [33, 40],
        "png_fields_removed": True,
        "shared_auto_legend": legend,
        "all_detail_gets_share_same_legend": True,
    }
    tensor_payload = {
        "capture": frame["capture_metadata"],
        "tensors": tensors,
        "feedback_scalar_values": feedback_scalars,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_path = out_dir / f"frame_{prefix}.json"
    tensors_path = out_dir / f"tensors_{prefix}.json"
    write_json(frame_path, frame)
    write_json(tensors_path, tensor_payload)
    return {
        "passed": True,
        "sequence": sequence,
        "sim_time": sim_time,
        "node_count": len(nodes),
        "captured_tensor_count": len(tensors),
        "detail_get_count": len(nodes) + 40,
        "feedback_key": feedback_key,
        "shared_legend": legend,
        "png_fields_removed": True,
        "files": [str(frame_path), str(tensors_path)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="prototype_before")
    args = parser.parse_args()
    print(json.dumps(capture(args.base_url, args.out_dir, args.prefix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
