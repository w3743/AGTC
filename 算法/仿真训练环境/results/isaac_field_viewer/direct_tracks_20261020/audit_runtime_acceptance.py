"""Build a read-only before/after acceptance record for the live 2D viewer."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8770"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_frame() -> dict:
    query = urlencode({"detail": "1", "selected": "action.applied", "scale": "auto"})
    with urlopen(f"{BASE_URL}/api/frame?{query}", timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status: {response.status}")
        return json.loads(response.read().decode("utf-8"))


def numeric_changed(before, after) -> bool:
    return before != after


def main() -> None:
    frame_before = read_json(HERE / "frame_before.json")
    tensors_before = read_json(HERE / "tensors_before.json")["tensors"]
    frame_after = read_json(HERE / "frame_after.json")
    tensors_after = read_json(HERE / "tensors_after.json")["tensors"]

    repeated = [get_frame(), get_frame()]
    signatures = [
        {
            "sequence": frame.get("sequence"),
            "sim_time": frame.get("sim_time"),
            "output_time": frame.get("output_time"),
            "running": frame.get("state", {}).get("running"),
            "scale": frame.get("legend", {}).get("scale"),
            "errors": frame.get("errors"),
            "node_count": len(frame.get("nodes", [])),
        }
        for frame in repeated
    ]
    stable_fields = ("sequence", "sim_time", "output_time", "running", "scale", "errors", "node_count")
    repeat_stable = all(
        signatures[0].get(field) == signatures[1].get(field) for field in stable_fields
    )

    frame_keys = {str(node["key"]) for node in frame_after.get("nodes", [])}
    mechanical_nodes = sorted(
        key for key in frame_keys if key.startswith("actuator.") or key == "motion.actual"
    )
    deltas = []
    for key in tensors_before.keys() & tensors_after.keys():
        before_detail = tensors_before[key]["detail"]
        after_detail = tensors_after[key]["detail"]
        stat_fields = ("min", "max", "mean", "std", "nonfinite")
        changed_stats = [
            field
            for field in stat_fields
            if numeric_changed(before_detail.get(field), after_detail.get(field))
        ]
        deltas.append(
            {
                "key": key,
                "title": tensors_before[key]["node"].get("title"),
                "shape_before": before_detail.get("shape"),
                "shape_after": after_detail.get("shape"),
                "changed_stats": changed_stats,
                "value_index_before": before_detail.get("index"),
                "value_index_after": after_detail.get("index"),
                "element_before": before_detail.get("value"),
                "element_after": after_detail.get("value"),
                "element_changed": numeric_changed(
                    before_detail.get("value"), after_detail.get("value")
                ),
            }
        )

    errors = frame_after.get("errors") or []
    result = {
        "audit": "direct tracks live 2D viewer runtime acceptance",
        "date": "2026-10-20",
        "endpoint": f"{BASE_URL}/api/frame",
        "method": "GET only; no control requests or UI interactions",
        "before": {
            "sequence": frame_before.get("sequence"),
            "sim_time": frame_before.get("sim_time"),
            "output_time": frame_before.get("output_time"),
            "running": frame_before.get("state", {}).get("running"),
            "mode": frame_before.get("mode"),
            "scale": frame_before.get("legend", {}).get("scale"),
            "node_count": len(frame_before.get("nodes", [])),
        },
        "after": {
            "sequence": frame_after.get("sequence"),
            "sim_time": frame_after.get("sim_time"),
            "output_time": frame_after.get("output_time"),
            "running": frame_after.get("state", {}).get("running"),
            "mode": frame_after.get("mode"),
            "scale": frame_after.get("legend", {}).get("scale"),
            "node_count": len(frame_after.get("nodes", [])),
            "errors": errors,
            "mechanical_or_motion_nodes": mechanical_nodes,
        },
        "paused_repeat_gets": {
            "snapshots": signatures,
            "sequence_and_scale_and_status_stable": repeat_stable,
        },
        "raw_tensor_comparison": {
            "captured_count_before": len(tensors_before),
            "captured_count_after": len(tensors_after),
            "compared_count": len(deltas),
            "stat_changed_count": sum(bool(item["changed_stats"]) for item in deltas),
            "element_changed_count": sum(item["element_changed"] for item in deltas),
            "layers": deltas,
        },
        "checks": {
            "before_was_paused": frame_before.get("state", {}).get("running") is False,
            "after_was_paused": frame_after.get("state", {}).get("running") is False,
            "after_has_57_nodes": len(frame_after.get("nodes", [])) == 57,
            "after_errors_empty": not errors,
            "after_has_no_mechanical_or_motion_nodes": not mechanical_nodes,
            "repeat_gets_stable": repeat_stable,
            "before_after_same_tensor_keys": tensors_before.keys() == tensors_after.keys(),
            "at_least_one_raw_stat_changed": any(item["changed_stats"] for item in deltas),
            "at_least_one_selected_element_changed": any(
                item["element_changed"] for item in deltas
            ),
        },
    }
    result["passed"] = all(result["checks"].values())
    (HERE / "runtime_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "checks": result["checks"], "files": [
        str(HERE / "runtime_acceptance.json")
    ]}, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
