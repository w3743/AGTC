"""Read-only capture of the live Isaac Field 2D network viewer HTTP frame.

The viewer remains in its current run/pause state.  This script performs only
GET requests to the public /api/frame endpoint and chooses activation keys from
the returned node metadata (it does not assume layer names for the requested
detail calls).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def fetch(base_url: str, query: dict[str, str]) -> dict:
    url = f"{base_url.rstrip('/')}/api/frame?{urlencode(query)}"
    with urlopen(url, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def one(nodes: list[dict], label: str, predicate) -> dict:
    matches = [node for node in nodes if predicate(node)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one node for {label}, found {len(matches)}")
    return matches[0]


def field_node(nodes: list[dict], branch: str, title_fragment: str, label: str) -> dict:
    return one(
        nodes,
        label,
        lambda node: node.get("branch") == branch
        and title_fragment in str(node.get("title", "")),
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="before")
    args = parser.parse_args()

    # The requested designated frame is preserved verbatim, including the
    # explicit action.applied detail selection.
    frame = fetch(
        args.base_url,
        {"detail": "1", "selected": "action.applied", "scale": "auto"},
    )
    if frame.get("state", {}).get("running") is not False:
        raise RuntimeError("refusing capture: viewer is not paused")
    if frame.get("detail", {}).get("key") != "action.applied":
        raise RuntimeError("designated frame did not select action.applied")
    nodes = frame.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("frame has no actual node list")

    lidar_convs = sorted(
        [
            node
            for node in nodes
            if node.get("branch") == "lidar"
            and "一维卷积" in str(node.get("title", ""))
        ],
        key=lambda node: float(node["column"]),
    )
    depth_convs = sorted(
        [
            node
            for node in nodes
            if node.get("branch") == "depth"
            and "二维卷积" in str(node.get("title", ""))
        ],
        key=lambda node: float(node["column"]),
    )
    if len(lidar_convs) < 2 or len(depth_convs) < 2:
        raise RuntimeError("actual node list did not expose two conv layers per sensor")

    selected_nodes = [
        field_node(nodes, "lidar", "输入", "lidar input"),
        lidar_convs[0],
        lidar_convs[1],
        field_node(nodes, "lidar", "编码输出", "lidar encoder output"),
        field_node(nodes, "depth", "输入", "depth input"),
        depth_convs[0],
        depth_convs[1],
        field_node(nodes, "depth", "编码输出", "depth encoder output"),
        field_node(nodes, "state", "状态输入", "state input"),
        one(nodes, "fusion", lambda node: node.get("branch") == "fusion"),
        field_node(nodes, "actor", "驾驶记忆 LSTM", "actor LSTM"),
        one(
            nodes,
            "action output",
            lambda node: node.get("key") == frame["detail"]["key"],
        ),
    ]

    base_sequence = frame.get("sequence")
    base_sim_time = frame.get("sim_time")
    tensors: dict = {
        "capture": {
            "base_url": args.base_url,
            "endpoint": "/api/frame",
            "method": "GET",
            "query_template": "detail=1&selected=<key-from-frame.nodes>&index=0&scale=auto",
            "sequence": base_sequence,
            "sim_time": base_sim_time,
            "paused": True,
            "run_state_changed": False,
            "node_keys_source": "frame_before.json nodes; selected keys resolved from actual branch/title/column metadata",
        },
        "tensors": {},
    }

    # Use only keys observed in the frame's node list for selected detail GETs.
    for node in selected_nodes:
        key = str(node["key"])
        detail_frame = fetch(
            args.base_url,
            {
                "detail": "1",
                "selected": key,
                "index": "0",
                "scale": "auto",
            },
        )
        if (
            detail_frame.get("sequence") != base_sequence
            or detail_frame.get("sim_time") != base_sim_time
        ):
            raise RuntimeError(
                "frame advanced during read-only capture; discard partial capture and retry after pause"
            )
        detail = detail_frame.get("detail") or {}
        if detail.get("key") != key:
            raise RuntimeError(f"detail response key mismatch for {key}")
        # Exclude the render-only PNG; preserve all raw JSON stats and the exact
        # API-selected scalar element/coordinate (default linear index 0).
        detail = {name: value for name, value in detail.items() if name != "image"}
        tensors["tensors"][key] = {
            "node": {
                "title": node.get("title"),
                "branch": node.get("branch"),
                "column": node.get("column"),
                "shape_from_nodes": node.get("shape"),
                "source": node.get("source"),
            },
            "query_key": key,
            "detail": detail,
        }

    output_dir = args.out_dir
    save_json(output_dir / f"frame_{args.prefix}.json", frame)
    save_json(output_dir / f"tensors_{args.prefix}.json", tensors)
    print(
        json.dumps(
            {
                "passed": True,
                "sequence": base_sequence,
                "sim_time": base_sim_time,
                "node_count": len(nodes),
                "captured_tensor_count": len(selected_nodes),
                "keys": [node["key"] for node in selected_nodes],
                "files": [
                    str(output_dir / f"frame_{args.prefix}.json"),
                    str(output_dir / f"tensors_{args.prefix}.json"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
