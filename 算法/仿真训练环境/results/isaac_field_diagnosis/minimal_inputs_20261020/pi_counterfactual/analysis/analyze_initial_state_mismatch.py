#!/usr/bin/env python3
"""Read completed capped-P and PI summaries; quantify why strict pairing failed."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual"
OUT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_differences(left: Any, right: Any, path: str = "") -> tuple[list[dict], list[dict]]:
    numeric: list[dict] = []
    other: list[dict] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                other.append({"field": child, "p_present": key in left, "pi_present": key in right})
            else:
                a, b = flatten_differences(left[key], right[key], child)
                numeric.extend(a)
                other.extend(b)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            other.append({"field": path, "p_length": len(left), "pi_length": len(right)})
        for index, (a_value, b_value) in enumerate(zip(left, right)):
            a, b = flatten_differences(a_value, b_value, f"{path}[{index}]")
            numeric.extend(a)
            other.extend(b)
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        delta = abs(float(left) - float(right))
        if delta != 0.0:
            numeric.append({"field": path, "p": left, "pi": right, "abs_difference": delta})
    elif left != right:
        other.append({"field": path, "p": left, "pi": right})
    return numeric, other


def pick_case(summary: dict, name: str) -> dict:
    matches = [row for row in summary["cases"] if row["case"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} case in {summary['run_label']}")
    return matches[0]


def drive_metrics(row: dict) -> dict:
    drive = row["drive"]
    return {
        "net_yaw_delta_rad": drive["net_yaw_delta_unwrapped_rad"],
        "mean_signed_yaw_rate_last_1s_rad_s": drive["mean_signed_actual_yaw_rate_last_1s_rad_s"],
        "mean_abs_yaw_rate_last_1s_rad_s": drive["mean_abs_actual_yaw_rate_last_1s_rad_s"],
        "last_1s_mean_abs_qd_tracking_error_rad_s": drive["last_1s_mean_abs_qd_tracking_error_rad_s"],
        "last_1s_actual_v_w_mps_rad_s": drive["mean_actual_v_w"],
        "max_abs_applied_effort_nm_per_proxy_wheel": drive["max_abs_applied_effort_nm"],
        "max_abs_side_integral_feedforward_nm_per_proxy_wheel": drive[
            "max_abs_side_integral_feedforward_nm_per_proxy_wheel"
        ],
    }


def brake_metrics(row: dict) -> dict:
    brake = row["brake"]
    return {
        "start_actual_v_w_mps_rad_s": brake["start_actual_v_w"],
        "end_actual_v_w_mps_rad_s": brake["end_actual_v_w"],
        "last_0_3s_mean_actual_v_w_mps_rad_s": brake["last_0_3s_mean_actual_v_w"],
        "last_0_3s_max_abs_v_mps": brake["last_0_3s_max_abs_actual_v_mps"],
        "last_0_3s_max_abs_yaw_rate_rad_s": brake["last_0_3s_max_abs_actual_yaw_rate_rad_s"],
        "last_0_3s_low_speed_and_yaw_thresholds_met": brake["last_0_3s_meets_both_low_speed_thresholds"],
        "end_track_command_mps": brake["end_track_command_mps"],
        "end_side_integral_ff_nm_per_proxy_wheel": brake[
            "last_0_3s_side_integral_feedforward_nm_per_proxy_wheel"
        ]["end_per_side"],
        "max_abs_applied_effort_nm_per_proxy_wheel": brake["max_abs_applied_effort_nm"],
    }


def main() -> None:
    outputs = (OUT / "initial_state_differences.json", OUT / "analysis.md")
    if any(path.exists() for path in outputs):
        raise SystemExit("Refusing to overwrite existing analysis output")

    p = read_json(BASE / "capped_p_reference/counterfactual_summary.json")
    pi = read_json(BASE / "pi_candidate/counterfactual_summary.json")
    p_exit = read_json(BASE / "capped_p_reference_exit.json")
    pi_exit = read_json(BASE / "pi_candidate_exit.json")
    pair_exit = read_json(BASE / "paired_counterfactual_attempt1_exit.json")
    comparison_path = BASE / "pair_comparison.json"

    p_cases = {row["case"]: row for row in p["cases"]}
    pi_cases = {row["case"]: row for row in pi["cases"]}
    if set(p_cases) != set(pi_cases) or len(p_cases) != 10:
        raise ValueError("the completed runs do not contain the same ten unique cases")

    case_differences = []
    field_max: dict[str, float] = {}
    for case_name in p_cases:
        p_row, pi_row = p_cases[case_name], pi_cases[case_name]
        numeric, other = flatten_differences(
            p_row["initial_state_after_settle"], pi_row["initial_state_after_settle"]
        )
        for item in numeric:
            field_max[item["field"]] = max(field_max.get(item["field"], 0.0), item["abs_difference"])
        case_differences.append(
            {
                "case": case_name,
                "action_exact_match": p_row["policy_normalized_action"] == pi_row["policy_normalized_action"],
                "initial_state_exact_match": not numeric and not other,
                "max_abs_numeric_difference": max(
                    (item["abs_difference"] for item in numeric), default=0.0
                ),
                "numeric_field_differences": numeric,
                "non_numeric_or_shape_differences": other,
            }
        )

    source_hashes_equal = p["source_sha256"] == pi["source_sha256"]
    source_hash_differences = []
    for path in sorted(set(p["source_sha256"]) | set(pi["source_sha256"])):
        p_hash, pi_hash = p["source_sha256"].get(path), pi["source_sha256"].get(path)
        if p_hash != pi_hash:
            source_hash_differences.append({"path": path, "p_sha256": p_hash, "pi_sha256": pi_hash})

    metadata = {
        "schema_equal": p.get("schema") == pi.get("schema"),
        "source_sha256_equal": source_hashes_equal,
        "source_sha256_mismatches": source_hash_differences,
        "environment_and_reset_config_equal": p.get("environment") == pi.get("environment"),
        "base_drive_config_equal": p.get("base_drive_config") == pi.get("base_drive_config"),
        "case_set_equal": set(p_cases) == set(pi_cases),
        "all_actions_equal": all(item["action_exact_match"] for item in case_differences),
        "all_settle_states_exactly_equal": all(item["initial_state_exact_match"] for item in case_differences),
        "settle_control_strategy_equal": all(
            p_cases[name]["settle_control_strategy"] == pi_cases[name]["settle_control_strategy"]
            for name in p_cases
        ),
    }

    small_response = {}
    for name in ("diff_left_plus_0.05", "diff_left_plus_0.10"):
        p_metrics, pi_metrics = drive_metrics(p_cases[name]), drive_metrics(pi_cases[name])
        p_yaw = abs(p_metrics["mean_abs_yaw_rate_last_1s_rad_s"])
        pi_yaw = abs(pi_metrics["mean_abs_yaw_rate_last_1s_rad_s"])
        small_response[name] = {
            "capped_p": p_metrics,
            "pi_ki_0_1": pi_metrics,
            "descriptive_yaw_rate_magnitude_ratio_pi_over_p": pi_yaw / p_yaw if p_yaw else None,
            "strict_pairing": False,
        }

    brake_results = {}
    for name in ("brake_straight_0.20_to_zero", "brake_diff_left_plus_0.80_to_zero"):
        brake_results[name] = {
            "capped_p": brake_metrics(p_cases[name]),
            "pi_ki_0_1": brake_metrics(pi_cases[name]),
            "strict_pairing": False,
        }

    raw_paths = {
        "capped_p_policy_samples": BASE / "capped_p_reference/policy_step_samples.jsonl",
        "pi_policy_samples": BASE / "pi_candidate/policy_step_samples.jsonl",
        "capped_p_substep_effort": BASE / "capped_p_reference/physics_substep_effort.jsonl",
        "pi_substep_effort": BASE / "pi_candidate/physics_substep_effort.jsonl",
    }
    result = {
        "schema": "isaac-field-pi-initial-state-mismatch-audit-v1",
        "source_runs": {
            "capped_p_summary": "capped_p_reference/counterfactual_summary.json",
            "pi_summary": "pi_candidate/counterfactual_summary.json",
            "pair_comparison_exists": comparison_path.exists(),
            "capped_p_exit": p_exit,
            "pi_exit": pi_exit,
            "pair_attempt_exit": pair_exit,
        },
        "pairing_metadata": metadata,
        "configuration_values": {
            "environment_and_reset": p["environment"],
            "base_drive_config": p["base_drive_config"],
            "controller_ki_p": p["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"],
            "controller_ki_pi": pi["counterfactual_controller"]["ki_nm_per_proxy_wheel_per_rad_s_side_mean_s"],
        },
        "initial_state_audit": {
            "case_count": len(case_differences),
            "exact_state_match_count": sum(item["initial_state_exact_match"] for item in case_differences),
            "max_abs_numeric_difference_across_cases": max(field_max.values(), default=0.0),
            "max_abs_difference_by_field": dict(sorted(field_max.items())),
            "cases": case_differences,
        },
        "descriptive_response_not_strictly_paired": {
            "low_differential_drive": small_response,
            "braking": brake_results,
        },
        "raw_input_sha256": {name: sha256(path) for name, path in raw_paths.items()},
        "interpretation_limits": [
            "The comparator's exact settle-state equality failed, so the P and PI values below are descriptive side-by-side results, not a strict paired causal estimate.",
            "The two summaries report identical source-hash maps, environment/reset settings, v5 base drive configuration, actions, and disabled-integral settle strategy.",
            "Ki=0.1 is an uncalibrated simulator hypothesis; applied effort is proxy-wheel simulator effort, not measured motor torque.",
            "For differential braking, the 2 s zero normalized action does not produce a zero track-speed command by the end: the acceleration limiter leaves approximately +/-0.333 m/s. The low-speed threshold therefore cannot establish stopping from a zero command or absence of integral residue.",
            "No strict comparison plot is produced while exact pairing remains false.",
        ],
    }
    outputs[0].write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    initial = result["initial_state_audit"]
    top_fields = sorted(field_max.items(), key=lambda item: item[1], reverse=True)[:8]
    lines = [
        "# P / PI 初始状态与响应核查",
        "",
        "P(Ki=0) 与 PI(Ki=0.1) 两次原生运行本身均真实 exit 0，但比较器 exit 1，未生成 `pair_comparison.json`。比较器要求每个 case 的 settle 后状态逐字段完全一致；本核查保留该严格门禁，不做容差放宽，也不生成严格配对图。",
        "",
        f"运行配置匹配：源文件 SHA 映射 `{metadata['source_sha256_equal']}`；环境/reset 配置 `{metadata['environment_and_reset_config_equal']}`；v5 drive 配置 `{metadata['base_drive_config_equal']}`；10 个动作 `{metadata['all_actions_equal']}`；settle 控制策略 `{metadata['settle_control_strategy_equal']}`。settle 后状态精确匹配 `{initial['exact_state_match_count']}/{initial['case_count']}`。",
        "",
        f"10 个 case 中 settle 状态最大数值差为 **{initial['max_abs_numeric_difference_across_cases']:.9g}**（单位随字段）。差值最大的字段如下：",
        "",
        "| 字段 | 最大绝对差 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{field}` | {delta:.9g} |" for field, delta in top_fields)
    lines += [
        "",
        "位置与零目标命令可保持精确相同；姿态、底盘速度、root 高度和轮速存在小量非位级差异。因此比较器失败是精确值门禁所致，差异量级很小，但不能仅凭现有记录断言它们全是浮点舍入。两组 source hash、仿真/reset配置、动作和 settle 控制策略均匹配。",
        "",
        "## 低差速响应（描述性并列，不是严格配对）",
        "",
        "| 动作 | 控制器 | 8 s净 yaw (rad) | 末1 s平均 |yaw rate| (rad/s) | 末1 s左右轮速误差 (rad/s) | 最大单轮努力 (Nm) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case_name, label in (("diff_left_plus_0.05", ".05"), ("diff_left_plus_0.10", ".10")):
        metrics_pair = small_response[case_name]
        for controller_name, metrics in (("P / Ki=0", metrics_pair["capped_p"]), ("PI / Ki=.1", metrics_pair["pi_ki_0_1"])):
            errors = metrics["last_1s_mean_abs_qd_tracking_error_rad_s"]
            lines.append(
                f"| {label} | {controller_name} | {metrics['net_yaw_delta_rad']:.4f} | "
                f"{metrics['mean_abs_yaw_rate_last_1s_rad_s']:.4f} | "
                f"L {errors['left']:.3f}, R {errors['right']:.3f} | "
                f"{metrics['max_abs_applied_effort_nm_per_proxy_wheel']:.3f} |"
            )
    lines += [
        "",
        "PI 的末1秒 yaw-rate 响应在 .05/.10 case 分别约为 0.0416/0.0715 rad/s；P 为约 0.00144/0.00364 rad/s。两者仍远低于名义命令 1.016/2.033 rad/s。由于 settle 状态不完全相同，这只能作为本次运行的描述性观察。",
        "",
        "## 制动响应（描述性并列，不是严格配对）",
        "",
        "| case | 控制器 | 末0.3 s均值 v (m/s) | 末0.3 s均值 w (rad/s) | 末track command (m/s) | 末0.3 s阈值 |",
        "|---|---|---:|---:|---|---|",
    ]
    for case_name, label in (("brake_straight_0.20_to_zero", "直行"), ("brake_diff_left_plus_0.80_to_zero", "差速")):
        for controller_name, metrics in (("P / Ki=0", brake_results[case_name]["capped_p"]), ("PI / Ki=.1", brake_results[case_name]["pi_ki_0_1"])):
            v, w = metrics["last_0_3s_mean_actual_v_w_mps_rad_s"]
            cmd_l, cmd_r = metrics["end_track_command_mps"]
            lines.append(
                f"| {label} | {controller_name} | {v:.4f} | {w:.4f} | "
                f"{cmd_l:+.3f}, {cmd_r:+.3f} | {metrics['last_0_3s_low_speed_and_yaw_thresholds_met']} |"
            )
    lines += [
        "",
        "直行制动末端命令已为 0；差速制动的 2 s 零 normalized action 后，命令仍约为 ±0.333 m/s，因为 1.5 m/s² 的加速度限幅还未降至 0。故差速制动末0.3 s低速阈值不等于从零命令停车，也不能据此排除积分残留。",
        "",
        "Ki=.1 是未标定的仿真假设；这些 proxy wheel effort 数值不代表实测电机扭矩。本结果不支持生产采用、导航成功、避障或作物安全结论。逐 case 完整差值及输入 JSONL 哈希见 `initial_state_differences.json`。",
        "",
    ]
    outputs[1].write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {outputs[0]}")
    print(f"WROTE {outputs[1]}")


if __name__ == "__main__":
    main()
