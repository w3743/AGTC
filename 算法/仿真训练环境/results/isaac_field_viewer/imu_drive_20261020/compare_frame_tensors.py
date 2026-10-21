"""Compare paused before/after captures using raw detail JSON, not PNGs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STAT_FIELDS = ("min", "max", "mean", "std", "nonfinite", "value")
FEEDBACK_KEY = "input.feedback"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def contains_image_key(value) -> bool:
    if isinstance(value, dict):
        return any(key == "image" or contains_image_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_image_key(item) for item in value)
    return False


def finite_summary(tensors: dict) -> dict:
    bad = []
    checked_values = 0
    for key, payload in tensors.items():
        detail = payload.get("detail", {})
        for field in STAT_FIELDS:
            value = detail.get(field)
            if isinstance(value, (int, float)):
                checked_values += 1
                if not math.isfinite(float(value)):
                    bad.append({"key": key, "field": field, "value": repr(value)})
        count = detail.get("nonfinite")
        if isinstance(count, (int, float)) and count != 0:
            bad.append({"key": key, "field": "nonfinite_count", "value": count})
    return {"finite": not bad, "checked_numeric_stats": checked_values, "bad_values": bad}


def capture_info(folder: Path, prefix: str) -> dict:
    frame_path = folder / f"frame_{prefix}.json"
    tensors_path = folder / f"tensors_{prefix}.json"
    frame = read_json(frame_path)
    payload = read_json(tensors_path)
    metadata = payload.get("capture") or frame.get("capture_metadata") or {}
    nodes = frame.get("nodes")
    tensors = payload.get("tensors")
    if not isinstance(nodes, list) or not isinstance(tensors, dict):
        raise ValueError(f"{prefix}: missing nodes or tensor details")
    if frame.get("state", {}).get("running") is not False or metadata.get("paused") is not True:
        raise ValueError(f"{prefix}: capture was not confirmed paused")
    if metadata.get("all_detail_reads_same_sequence_and_time") is not True:
        raise ValueError(f"{prefix}: detail reads were not verified on one sequence/time")
    if metadata.get("png_fields_removed") is not True:
        raise ValueError(f"{prefix}: capture does not certify PNG removal")
    node_map = {node.get("key"): node for node in nodes if isinstance(node, dict)}
    if len(node_map) != len(nodes) or any(not isinstance(key, str) for key in node_map):
        raise ValueError(f"{prefix}: node metadata keys are invalid or duplicated")
    node_keys = set(node_map)
    tensor_keys = set(tensors)
    if node_keys != tensor_keys:
        raise ValueError(
            f"{prefix}: capture does not include exactly every actual node "
            f"(missing={sorted(node_keys - tensor_keys)}, extra={sorted(tensor_keys - node_keys)})"
        )

    missing_parents = []
    duplicate_parents = []
    for key, node in node_map.items():
        parents = node.get("parents", [])
        if not isinstance(parents, list):
            missing_parents.append({"node": key, "parent_value": repr(parents)})
            continue
        if len(parents) != len(set(parents)):
            duplicate_parents.append(key)
        for parent in parents:
            if parent not in node_keys:
                missing_parents.append({"node": key, "missing_parent": parent})
        detail_node = tensors[key].get("node", {})
        if detail_node.get("parents") != parents:
            raise ValueError(f"{prefix}: per-tensor parent metadata differs from frame.nodes for {key}")

    if contains_image_key(frame) or contains_image_key(payload):
        raise ValueError(f"{prefix}: PNG image fields were not removed")

    legend = metadata.get("shared_auto_legend")
    if not isinstance(legend, dict) or metadata.get("all_detail_gets_share_same_legend") is not True:
        raise ValueError(f"{prefix}: missing shared automatic colour-scale evidence")
    legend_values = [legend.get(name) for name in ("minimum", "zero", "maximum", "scale")]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in legend_values):
        legend_valid = False
    else:
        legend_valid = (
            float(legend["zero"]) == 0.0
            and math.isclose(float(legend["minimum"]), -float(legend["maximum"]), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(legend["scale"]), float(legend["maximum"]), rel_tol=0.0, abs_tol=1e-12)
        )

    feedback_values = payload.get("feedback_scalar_values")
    if not isinstance(feedback_values, list) or len(feedback_values) != 41:
        raise ValueError(f"{prefix}: missing 41 feedback scalar values")
    feedback_by_index = {item.get("index"): item for item in feedback_values}
    if set(feedback_by_index) != set(range(41)):
        raise ValueError(f"{prefix}: feedback scalar indices are incomplete")
    if FEEDBACK_KEY not in tensors:
        raise ValueError(f"{prefix}: expected feedback input detail {FEEDBACK_KEY!r}")
    feedback_validity = []
    feedback_age = []
    for index, item in sorted(feedback_by_index.items()):
        field = str(item.get("field", ""))
        value = item.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{prefix}: feedback scalar {index} is nonfinite or not numeric")
        record = {"index": index, "field": field, "value": value}
        if "有效标记" in field or "valid" in field.lower():
            feedback_validity.append(record)
        if "数据年龄" in field or "age" in field.lower():
            feedback_age.append(record)
    if len(feedback_validity) != 8 or len(feedback_age) != 8:
        raise ValueError(
            f"{prefix}: expected 8 validity and 8 age feedback fields; "
            f"found {len(feedback_validity)} and {len(feedback_age)}"
        )

    return {
        "prefix": prefix,
        "frame_path": str(frame_path),
        "tensors_path": str(tensors_path),
        "sequence": frame.get("sequence"),
        "sim_time": frame.get("sim_time"),
        "node_count": len(nodes),
        "node_map": node_map,
        "tensors": tensors,
        "parents_complete": not missing_parents,
        "missing_parents": missing_parents,
        "duplicate_parent_references": duplicate_parents,
        "legend": legend,
        "legend_valid": legend_valid,
        "finite": finite_summary(tensors),
        "feedback_validity": feedback_validity,
        "feedback_age": feedback_age,
        "feedback_all_fields_finite": True,
    }


def raw_detail_change(before: dict, after: dict) -> dict:
    before_detail = before.get("detail", {})
    after_detail = after.get("detail", {})
    fields = {}
    changed = False
    for field in STAT_FIELDS:
        old, new = before_detail.get(field), after_detail.get(field)
        equal = old == new
        fields[field] = {"before": old, "after": new, "changed": not equal}
        changed |= not equal
    return {"changed": changed, "fields": fields}


def compare(before: dict, after: dict) -> dict:
    before_keys = set(before["node_map"])
    after_keys = set(after["node_map"])
    common = sorted(before_keys & after_keys)
    per_node = {}
    changed_keys = []
    for key in common:
        old_node, new_node = before["node_map"][key], after["node_map"][key]
        old_shape, new_shape = old_node.get("shape"), new_node.get("shape")
        old_parents, new_parents = old_node.get("parents"), new_node.get("parents")
        change = raw_detail_change(before["tensors"][key], after["tensors"][key])
        metadata_changed = old_shape != new_shape or old_parents != new_parents
        per_node[key] = {
            "title_before": old_node.get("title"),
            "title_after": new_node.get("title"),
            "branch_before": old_node.get("branch"),
            "branch_after": new_node.get("branch"),
            "shape_before": old_shape,
            "shape_after": new_shape,
            "parents_before": old_parents,
            "parents_after": new_parents,
            "metadata_changed": metadata_changed,
            "raw_stats_and_selected_scalar": change,
        }
        if change["changed"]:
            changed_keys.append(key)

    def indexed_diffs(left: list[dict], right: list[dict]) -> list[dict]:
        right_by_index = {item["index"]: item for item in right}
        result = []
        for item in left:
            other = right_by_index[item["index"]]
            result.append({
                "index": item["index"],
                "field_before": item["field"],
                "field_after": other["field"],
                "before": item["value"],
                "after": other["value"],
                "changed": item["value"] != other["value"],
            })
        return result

    validity = indexed_diffs(before["feedback_validity"], after["feedback_validity"])
    ages = indexed_diffs(before["feedback_age"], after["feedback_age"])
    before_legend = before["legend"]
    after_legend = after["legend"]
    color_range_equal = all(
        before_legend.get(key) == after_legend.get(key)
        for key in ("minimum", "zero", "maximum", "scale", "mode", "name")
    )
    return {
        "schema": "paused-network-raw-tensor-compare-v1",
        "raw_png_not_used": True,
        "before": {key: before[key] for key in (
            "prefix", "frame_path", "tensors_path", "sequence", "sim_time", "node_count"
        )},
        "after": {key: after[key] for key in (
            "prefix", "frame_path", "tensors_path", "sequence", "sim_time", "node_count"
        )},
        "topology": {
            "keys_match": before_keys == after_keys,
            "added_keys": sorted(after_keys - before_keys),
            "removed_keys": sorted(before_keys - after_keys),
            "before_parents_complete": before["parents_complete"],
            "after_parents_complete": after["parents_complete"],
            "before_missing_parents": before["missing_parents"],
            "after_missing_parents": after["missing_parents"],
            "before_duplicate_parent_references": before["duplicate_parent_references"],
            "after_duplicate_parent_references": after["duplicate_parent_references"],
        },
        "finite_values": {
            "before": before["finite"],
            "after": after["finite"],
            "feedback_fields_finite": {
                "before": before["feedback_all_fields_finite"],
                "after": after["feedback_all_fields_finite"],
            },
        },
        "shared_auto_colour_scale": {
            "before": before_legend,
            "after": after_legend,
            "before_valid": before["legend_valid"],
            "after_valid": after["legend_valid"],
            "within_each_capture_all_detail_gets_use_one_scale": True,
            "same_scale_across_captures": color_range_equal,
        },
        "feedback": {
            "validity_fields": validity,
            "age_fields_normalized_by_2s": ages,
            "validity_changed_count": sum(item["changed"] for item in validity),
            "age_changed_count": sum(item["changed"] for item in ages),
        },
        "raw_tensor_change": {
            "compared_common_node_count": len(common),
            "changed_node_count": len(changed_keys),
            "changed_node_keys": changed_keys,
            "per_node": per_node,
            "constant_layers_are_allowed": True,
        },
        "passed": (
            before["parents_complete"]
            and after["parents_complete"]
            and before["legend_valid"]
            and after["legend_valid"]
            and before["finite"]["finite"]
            and after["finite"]["finite"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--before-prefix", default="prototype_before")
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--after-prefix", default="prototype_after")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = capture_info(args.before_dir, args.before_prefix)
    after = capture_info(args.after_dir, args.after_prefix)
    report = compare(before, after)
    write_json(args.output, report)
    print(json.dumps({
        "passed": report["passed"],
        "output": str(args.output),
        "before_nodes": before["node_count"],
        "after_nodes": after["node_count"],
        "compared_common_nodes": report["raw_tensor_change"]["compared_common_node_count"],
        "raw_changed_nodes": report["raw_tensor_change"]["changed_node_count"],
        "validity_changed_fields": report["feedback"]["validity_changed_count"],
        "age_changed_fields": report["feedback"]["age_changed_count"],
        "parents_complete": report["topology"]["before_parents_complete"] and report["topology"]["after_parents_complete"],
        "finite": report["finite_values"]["before"]["finite"] and report["finite_values"]["after"]["finite"],
        "same_auto_scale": report["shared_auto_colour_scale"]["same_scale_across_captures"],
        "constant_layers_allowed": True,
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
