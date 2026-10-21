"""Compare the two copied raw network exports on CPU; does not load a simulator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
TRIAL = ROOT / "results/isaac_field_viewer/coupled_tracks_20261020/runtime_8192"
SOURCE = ROOT / "results/isaac_field_viewer/network_exports"
BEFORE_STEM = "frame_20261020_115035_283818"
AFTER_STEM = "frame_20261020_115246_663781"
BASELINE_STEM = "frame_20261020_114747_131505"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frame(stem: str) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads((TRIAL / f"{stem}.json").read_text(encoding="utf-8"))
    with np.load(TRIAL / f"{stem}.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays


def frame_summary(stem: str, metadata: dict) -> dict:
    return {
        "stem": stem,
        "sequence": metadata.get("sequence"),
        "sim_time": metadata.get("sim_time"),
        "output_time": metadata.get("output_time"),
        "mode": metadata.get("mode"),
        "checkpoint_label": metadata.get("checkpoint_label"),
        "action_config": metadata.get("action_config"),
        "errors": metadata.get("errors"),
        "layer_count": len(metadata.get("layers", [])),
    }


def compare() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    before_meta, before_arrays = read_frame(BEFORE_STEM)
    after_meta, after_arrays = read_frame(AFTER_STEM)

    before_activations = {key.removeprefix("activation/"): value
                          for key, value in before_arrays.items() if key.startswith("activation/")}
    after_activations = {key.removeprefix("activation/"): value
                         for key, value in after_arrays.items() if key.startswith("activation/")}
    before_weights = {key.removeprefix("weight/"): value
                      for key, value in before_arrays.items() if key.startswith("weight/")}
    after_weights = {key.removeprefix("weight/"): value
                     for key, value in after_arrays.items() if key.startswith("weight/")}

    layers = before_meta.get("layers", [])
    layer_results = []
    activation_names = set(before_activations)
    shape_ok = set(after_activations) == activation_names
    finite_activations = True
    for layer in layers:
        key = layer.get("key")
        before = before_activations.get(key)
        after = after_activations.get(key)
        present = before is not None and after is not None
        expected_shape = tuple(layer.get("shape", []))
        shapes_match = present and tuple(before.shape) == tuple(after.shape) == expected_shape
        finite = present and bool(np.isfinite(before).all()) and bool(np.isfinite(after).all())
        finite_activations = finite_activations and finite
        changed = False
        changed_elements = 0
        max_abs_delta = 0.0
        if present and before.shape == after.shape:
            different = before != after
            changed_elements = int(np.count_nonzero(different))
            changed = changed_elements > 0
            if changed and np.issubdtype(before.dtype, np.number) and np.issubdtype(after.dtype, np.number):
                max_abs_delta = float(np.max(np.abs(after.astype(np.float64) - before.astype(np.float64))))

        parents = layer.get("parents", [])
        parent_changes = [parent for parent in parents
                          if parent in before_activations and parent in after_activations
                          and not np.array_equal(before_activations[parent], after_activations[parent])]
        if changed:
            unchanged_reason = None
        elif not present:
            unchanged_reason = "activation array missing from one or both NPZ exports"
        elif layer.get("source") == "实际前向" and not parents:
            unchanged_reason = "input layer raw array is identical in both exports"
        elif key == "action.log_std":
            unchanged_reason = (
                "actual policy distribution parameter is identical in both frames; "
                "all exported checkpoint weight arrays also match"
            )
        elif not parent_changes:
            unchanged_reason = "all declared parent activations are unchanged; no upstream raw change is observed"
        elif parents:
            unchanged_reason = (
                "declared parent activation(s) changed but this layer stayed bit-identical; "
                "cause cannot be established from two captures"
            )
        else:
            unchanged_reason = "layer has no declared parents; cause is not inferable from these captures"

        input_stats = None
        if layer.get("source") == "实际前向" and not parents and present and np.issubdtype(before.dtype, np.number):
            input_stats = {
                "min": float(np.min(before)),
                "max": float(np.max(before)),
                "nonzero_count": int(np.count_nonzero(before)),
                "unique_value_count": int(np.unique(before).size),
            }

        layer_results.append({
            "key": key,
            "title": layer.get("title"),
            "branch": layer.get("branch"),
            "shape": list(before.shape) if present else None,
            "metadata_shape": list(expected_shape),
            "present": present,
            "shape_matches_metadata": shapes_match,
            "finite": finite,
            "changed": changed,
            "changed_elements": changed_elements,
            "max_abs_delta": max_abs_delta,
            "changed_parent_keys": parent_changes,
            "unchanged_reason": unchanged_reason,
            "identical_input_stats": input_stats,
        })

    weight_names_match = set(before_weights) == set(after_weights)
    weights_finite = True
    weights_equal = weight_names_match
    weight_results = []
    for name in sorted(set(before_weights) | set(after_weights)):
        before = before_weights.get(name)
        after = after_weights.get(name)
        present = before is not None and after is not None
        same_shape = present and before.shape == after.shape
        finite = present and bool(np.isfinite(before).all()) and bool(np.isfinite(after).all())
        equal = present and same_shape and before.dtype == after.dtype and np.array_equal(before, after)
        weights_finite = weights_finite and finite
        weights_equal = weights_equal and equal
        weight_results.append({
            "key": name,
            "present": present,
            "shape": list(before.shape) if present else None,
            "finite": finite,
            "identical": equal,
        })

    source_copy_checks = {}
    for stem in (BEFORE_STEM, AFTER_STEM):
        for suffix in (".npz", ".json"):
            source = SOURCE / f"{stem}{suffix}"
            copied = TRIAL / f"{stem}{suffix}"
            source_copy_checks[f"{stem}{suffix}"] = {
                "source_sha256": sha256(source),
                "copied_sha256": sha256(copied),
                "identical_copy": sha256(source) == sha256(copied),
            }

    checks = {
        "before_is_requested_paused_loaded_checkpoint_frame": (
            before_meta.get("sequence") == 2
            and before_meta.get("sim_time") == 0.0
            and "checkpoint loaded" in str(before_meta.get("checkpoint_label", ""))
        ),
        "after_is_requested_same_loaded_checkpoint_runtime_frame": (
            after_meta.get("sequence") == 349
            and abs(float(after_meta.get("sim_time", -1.0)) - 17.3) < 1e-8
            and abs(float(after_meta.get("output_time", -1.0)) - 17.35) < 1e-8
            and "checkpoint loaded" in str(after_meta.get("checkpoint_label", ""))
        ),
        "action_and_checkpoint_labels_match": (
            before_meta.get("action_config") == after_meta.get("action_config")
            and before_meta.get("checkpoint_label") == after_meta.get("checkpoint_label")
        ),
        "60_activation_arrays_match_metadata_and_are_finite": (
            len(layers) == 60
            and len(before_activations) == len(after_activations) == 60
            and shape_ok
            and all(row["shape_matches_metadata"] and row["finite"] for row in layer_results)
        ),
        "all_weight_arrays_are_finite_and_identical": (
            len(before_weights) == len(after_weights) == 77
            and weights_finite and weights_equal
        ),
        "raw_exports_were_copied_byte_for_byte": all(row["identical_copy"] for row in source_copy_checks.values()),
        "zero_step_baseline_frame_not_mixed_into_pair": (
            (SOURCE / f"{BASELINE_STEM}.npz").is_file()
            and (SOURCE / f"{BASELINE_STEM}.json").is_file()
            and not (TRIAL / f"{BASELINE_STEM}.npz").exists()
            and not (TRIAL / f"{BASELINE_STEM}.json").exists()
        ),
        "frame_error_arrays_empty": not before_meta.get("errors") and not after_meta.get("errors"),
    }
    return {
        "schema": "coupled-tracks-runtime-8192-raw-activation-comparison-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "read_only_cpu": True,
        "image_or_color_changes_used_as_evidence": False,
        "source_frames": {
            "before": frame_summary(BEFORE_STEM, before_meta),
            "after": frame_summary(AFTER_STEM, after_meta),
            "sequence_delta": int(after_meta["sequence"]) - int(before_meta["sequence"]),
            "sim_time_delta_s": float(after_meta["sim_time"]) - float(before_meta["sim_time"]),
            "output_time_after_s": after_meta.get("output_time"),
            "baseline_frame_excluded": True,
        },
        "copy_sha256_checks": source_copy_checks,
        "activation_counts": {
            "layers_in_metadata": len(layers),
            "before_arrays": len(before_activations),
            "after_arrays": len(after_activations),
            "finite_shape_correct": sum(row["finite"] and row["shape_matches_metadata"] for row in layer_results),
            "changed_layers": [row["key"] for row in layer_results if row["changed"]],
            "unchanged_layers": [row for row in layer_results if not row["changed"]],
        },
        "weight_counts": {
            "before": len(before_weights),
            "after": len(after_weights),
            "identical": sum(row["identical"] for row in weight_results),
        },
        "checks": checks,
        "layers": layer_results,
        "weights": weight_results,
    }


def write_readme(report: dict) -> None:
    frames = report["source_frames"]
    counts = report["activation_counts"]
    lines = [
        "# 8192 步运行前后原始激活对照",
        "",
        "本对照直接读取复制后的 JSON 与 NPZ 数组，CPU-only；没有用 PNG、色标或截图变化代替 raw tensor 对比。原始导出未移动或覆盖，零步 baseline 帧留在 `network_exports`，未混入这组配对。",
        "",
        "## 帧与元数据",
        "",
        f"- 运行前：`{frames['before']['stem']}`，seq={frames['before']['sequence']}，sim_time={frames['before']['sim_time']} s，{frames['before']['mode']}。",
        f"- 运行后：`{frames['after']['stem']}`，seq={frames['after']['sequence']}，sim_time={frames['after']['sim_time']} s，output_time={frames['after']['output_time']} s，{frames['after']['mode']}。",
        f"- 两帧动作合同和 checkpoint label 相同：`{frames['before']['checkpoint_label']}`。运行后 seq 增加 {frames['sequence_delta']}，sim_time 增加 {frames['sim_time_delta_s']:.6f} s。",
        "- 两帧的 NPZ/JSON 都是从 `results/isaac_field_viewer/network_exports/` 原样复制；逐文件 SHA-256 相同，详见 `raw_comparison.json`。",
        "",
        "## Raw 数组结果",
        "",
        f"- 60/60 activation 数组与 metadata shape 一致且全部有限；其中 {len(counts['changed_layers'])} 层 raw 数组改变，{len(counts['unchanged_layers'])} 层逐元素相同。",
        f"- 77/77 权重数组有限且运行前后完全相同；本组变化来自运行时输入/状态，而非检查点权重切换。",
        "",
        "### 改变的 activation 层",
        "",
    ]
    lines.extend(f"- `{key}`" for key in counts["changed_layers"])
    lines += ["", "### 保持不变的层与可由 raw 证据支持的原因", "", "| 层 | 观测到的原因 |", "|---|---|"]
    for row in counts["unchanged_layers"]:
        reason = row["unchanged_reason"]
        stats = row.get("identical_input_stats")
        if stats is not None:
            reason += (
                f"；输入数组 min/max={stats['min']:.6g}/{stats['max']:.6g}，"
                f"非零元素 {stats['nonzero_count']}/{int(np.prod(row['shape']))}"
            )
        lines.append(f"| `{row['key']}` | {reason} |")
    lines += [
        "",
        "检测支路输入及其后续编码保持相同；raw 导出显示检测输入未变化。对父激活未变的层，仅报告其上游输入未变化，不推断更深原因。策略 `log_std` 参数也保持不变，且所有 77 个权重数组完全相同。",
        "",
        "## 验收",
        "",
        f"- CPU 比较结果：`{report['status']}`；所有 metadata、60 层 finite/shape、权重一致、原始文件哈希及 baseline 排除门禁通过。",
        "- 机器可读逐层记录：`raw_comparison.json`。实际 CPU 执行退出码见 `raw_comparison_cpu.exit.json`。",
        "- 这仅证明同一个已加载 8192-step checkpoint 在暂停与实际运行帧之间产生了原始 activation 变化；不代表任务能力已学会或导航效果已提升。",
        "",
    ]
    (TRIAL / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    try:
        report = compare()
    except Exception as exc:
        report = {
            "schema": "coupled-tracks-runtime-8192-raw-activation-comparison-v1",
            "status": "failed",
            "read_only_cpu": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    report["exit_code"] = 0 if report["status"] == "passed" else 1
    output = TRIAL / "raw_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["status"] == "passed":
        write_readme(report)
    print(json.dumps({
        "status": report["status"],
        "exit_code": report["exit_code"],
        "checks": report.get("checks", {}),
        "changed_layers": report.get("activation_counts", {}).get("changed_layers", []),
        "unchanged_layers": len(report.get("activation_counts", {}).get("unchanged_layers", [])),
        "identical_weights": report.get("weight_counts", {}).get("identical"),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
