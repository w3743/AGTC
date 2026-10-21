#!/usr/bin/env python3
"""Plot the protocol-passed four-case v5 P/PI confirmation from raw JSONL."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
BASE = ROOT / "results/isaac_field_diagnosis/minimal_inputs_20261020/pi_counterfactual"
OUT = Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def assert_finite(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{context}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value at {context}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            assert_finite(row, f"{path.name}:{line_number}")
            rows.append(row)
    return rows


def phase_rows(rows: list[dict], case: str, phase: str) -> list[dict]:
    selected = sorted(
        (row for row in rows if row.get("case") == case and row.get("phase") == phase),
        key=lambda row: row["step"],
    )
    if not selected:
        raise ValueError(f"missing raw samples for {case}/{phase}")
    return selected


def relative_times(rows: list[dict]) -> list[float]:
    start = float(rows[0]["task_time_s"])
    return [float(row["task_time_s"]) - start for row in rows]


def command_zero_time(rows: list[dict]) -> float | None:
    commands = [row["track_command_mps"] for row in rows]
    target = [0.0, 0.0]
    for index, command in enumerate(commands):
        if command == target and all(rest == target for rest in commands[index:]):
            return float(rows[index]["task_time_s"]) - float(rows[0]["task_time_s"])
    return None


def validate_inputs() -> tuple[dict, dict, dict, dict, dict, dict, list, list, dict]:
    protocol_path = BASE / "confirmation_protocol.json"
    pair_path = BASE / "comparison_repair/confirmation_pair_summary_repaired.json"
    pair_exit_path = BASE / "comparison_repair/comparison_repair_exit.json"
    original_pair_exit_path = BASE / "confirmation_pair_exit.json"
    p_exit_path = BASE / "confirmation_p_exit.json"
    pi_exit_path = BASE / "confirmation_pi_exit.json"
    p_summary_path = BASE / "confirmation_p/counterfactual_summary.json"
    pi_summary_path = BASE / "confirmation_pi/counterfactual_summary.json"
    protocol = read_json(protocol_path)
    pair = read_json(pair_path)
    pair_exit, original_pair_exit, p_exit, pi_exit = map(
        read_json, (pair_exit_path, original_pair_exit_path, p_exit_path, pi_exit_path)
    )
    p_summary, pi_summary = read_json(p_summary_path), read_json(pi_summary_path)

    if pair.get("schema") != "isaac-field-pi-counterfactual-confirmation-result-v1":
        raise ValueError("unexpected confirmation comparison schema")
    if pair.get("status") != "passed" or pair_exit.get("actual_exit_code") != 0:
        raise ValueError(
            f"confirmation gate did not pass (status={pair.get('status')}, exit={pair_exit.get('actual_exit_code')}); no plot/report written"
        )
    if p_exit.get("actual_exit_code") != 0 or pi_exit.get("actual_exit_code") != 0:
        raise ValueError("one of the confirmation runs did not exit successfully")
    if original_pair_exit.get("actual_exit_code") != 1:
        raise ValueError("the original comparator exit-1 record is missing or was changed")
    if pair.get("protocol_sha256") != digest(protocol_path):
        raise ValueError("comparison did not use the current confirmation protocol")

    required_gates = (
        "same_source_hashes",
        "source_freeze_matches_current",
        "same_v5_drive_config",
        "same_environment",
        "four_case_actions_match",
        "initial_state_within_protocol_tolerance",
        "finite_policy_and_substep_logs",
        "all_substeps_within_2nm_plus_1e_5",
        "braking_stop_gate_all_cases",
    )
    gates = pair.get("gates", {})
    if any(gates.get(key) is not True for key in required_gates):
        raise ValueError("not all prescribed confirmation gates passed; no plot/report written")
    if pair.get("initial_state_pairing", {}).get("passed") is not True:
        raise ValueError("predeclared initial-state tolerance gate failed; no plot/report written")

    p_rows = read_jsonl(BASE / "confirmation_p/policy_step_samples.jsonl")
    pi_rows = read_jsonl(BASE / "confirmation_pi/policy_step_samples.jsonl")
    expected_names = set(protocol["case_names"])
    if {row.get("case") for row in p_rows} != expected_names:
        raise ValueError("P raw JSONL case coverage does not match the confirmation protocol")
    if {row.get("case") for row in pi_rows} != expected_names:
        raise ValueError("PI raw JSONL case coverage does not match the confirmation protocol")
    if len(p_rows) != pair["policy_sample_count_by_run"]["confirmation_p"]:
        raise ValueError("P raw JSONL sample count disagrees with pair summary")
    if len(pi_rows) != pair["policy_sample_count_by_run"]["confirmation_pi"]:
        raise ValueError("PI raw JSONL sample count disagrees with pair summary")

    input_paths = {
        "protocol": protocol_path,
        "pair_summary": pair_path,
        "comparison_repair_exit": pair_exit_path,
        "original_comparison_exit": original_pair_exit_path,
        "P_summary": p_summary_path,
        "PI_summary": pi_summary_path,
        "P_raw_policy_samples": BASE / "confirmation_p/policy_step_samples.jsonl",
        "PI_raw_policy_samples": BASE / "confirmation_pi/policy_step_samples.jsonl",
        "P_raw_substep_effort": BASE / "confirmation_p/physics_substep_effort.jsonl",
        "PI_raw_substep_effort": BASE / "confirmation_pi/physics_substep_effort.jsonl",
    }
    return protocol, pair, pair_exit, original_pair_exit, p_summary, pi_summary, p_exit, pi_exit, p_rows, pi_rows, {
        name: digest(path) for name, path in input_paths.items()
    }


def plot_confirmation(p_rows: list[dict], pi_rows: list[dict], output_path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"P": "#56B4E9", "PI": "#E69F00", "left": "#56B4E9", "right": "#E69F00"}
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    fig.suptitle(
        "Newton v5 P / PI 原生物理确认：4 个 case（植物 0）\n"
        "预定容差内配对，非逐数值相同；物理响应不是导航验收",
        fontsize=15,
    )
    data = {"P": p_rows, "PI": pi_rows}

    for column, case in enumerate(("diff_left_plus_0.05", "diff_left_plus_0.10")):
        ax_yaw = axes[0, column]
        ax_wheels = axes[1, column]
        for controller, rows in data.items():
            drive = phase_rows(rows, case, "drive")
            time_s = relative_times(drive)
            yaw0 = float(drive[0]["yaw_unwrapped_rad"])
            yaw_delta = [float(row["yaw_unwrapped_rad"]) - yaw0 for row in drive]
            ax_yaw.plot(time_s, yaw_delta, color=colors[controller], label=controller, linewidth=1.6)

            side_fields = (
                ("left", "left_target_mean_rad_s", "left_actual_qd_mean_rad_s"),
                ("right", "right_target_mean_rad_s", "right_actual_qd_mean_rad_s"),
            )
            style = "-" if controller == "P" else "--"
            for side, target_key, actual_key in side_fields:
                target_color = colors[side]
                ax_wheels.plot(
                    time_s,
                    [float(row[actual_key]) for row in drive],
                    linestyle=style,
                    color=target_color,
                    linewidth=1.4,
                    label=f"{controller} {side} 实际",
                )
                if controller == "P":
                    ax_wheels.plot(
                        time_s,
                        [float(row[target_key]) for row in drive],
                        linestyle=":",
                        color=target_color,
                        alpha=0.75,
                        linewidth=1.2,
                        label=f"{side}目标",
                    )

        action_level = "0.05" if case.endswith("0.05") else "0.10"
        ax_yaw.set_title(f"差速 {action_level}：实际偏航")
        ax_yaw.set_xlabel("drive time (s)")
        ax_yaw.set_ylabel("Δyaw (rad)")
        ax_yaw.axhline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
        ax_yaw.grid(True, alpha=0.22)
        ax_yaw.legend(fontsize=8)

        ax_wheels.set_title(f"差速 {action_level}：七轮均值轮速跟踪")
        ax_wheels.set_xlabel("drive time (s)")
        ax_wheels.set_ylabel("wheel speed (rad/s)")
        ax_wheels.axhline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
        ax_wheels.grid(True, alpha=0.22)
        ax_wheels.legend(fontsize=7, ncol=2)

    brake_specs = (
        ("brake_straight_0.20_to_zero", "直行制动"),
        ("brake_diff_left_plus_0.80_to_zero", "差速制动"),
    )
    zero_times = {}
    for row, (case, label) in enumerate(brake_specs, start=2):
        for controller, samples in data.items():
            brake = phase_rows(samples, case, "brake")
            time_s = relative_times(brake)
            zero_t = command_zero_time(brake)
            zero_times[f"{controller}/{case}"] = zero_t
            for column, component, unit, axis_label in (
                (0, 0, "m/s", "v"),
                (1, 1, "rad/s", "w"),
            ):
                ax = axes[row, column]
                ax.plot(
                    time_s,
                    [float(sample["actual_v_w"][component]) for sample in brake],
                    color=colors[controller],
                    linewidth=1.5,
                    label=controller,
                )
                if zero_t is not None:
                    ax.axvline(
                        zero_t,
                        color=colors[controller],
                        linestyle=":",
                        alpha=0.7,
                        linewidth=1.0,
                        label=f"{controller} cmd=0 @ {zero_t:.2f}s",
                    )
                ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.6)
                ax.set_title(f"{label}：{axis_label}（零命令时刻标线）")
                ax.set_xlabel("brake time (s)")
                ax.set_ylabel(f"{axis_label} ({unit})")
                ax.grid(True, alpha=0.22)
                ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return zero_times


def main() -> None:
    outputs = (OUT / "confirmation_comparison_cjk.png", OUT / "confirmation_analysis_cjk.md")
    if any(path.exists() for path in outputs):
        raise SystemExit("Refusing to overwrite existing confirmation analysis")
    protocol, pair, pair_exit, original_pair_exit, p, pi, p_exit, pi_exit, p_rows, pi_rows, input_hashes = validate_inputs()
    output_png, output_md = outputs
    zero_times = plot_confirmation(p_rows, pi_rows, output_png)

    p_cases = {row["case"]: row for row in p["cases"]}
    pi_cases = {row["case"]: row for row in pi["cases"]}
    lines = [
        "# P / PI 原生物理确认",
        "",
        "**本轮在预定容差内配对，非逐数值相同；这是物理响应检查，不是导航验收。**",
        "",
        f"P(Ki=0) exit={p_exit['actual_exit_code']}，PI(Ki=0.1) exit={pi_exit['actual_exit_code']}；修复后的 comparison exit={pair_exit['actual_exit_code']}、status=`{pair['status']}`。原 comparison exit={original_pair_exit['actual_exit_code']} 保留在原文件；失败原因是比较器读取 `expected_timing.drive_s`，而协议键名为 `drive`。四个 case 按固定 protocol 门禁通过。",
        "",
        "## 预定初态容差",
        "",
        "| 字段 | 允许最大差 | 实际最大差 |",
        "|---|---:|---:|",
    ]
    tolerances = pair["initial_state_pairing"]["protocol_tolerances"]
    observed = pair["initial_state_pairing"]["max_abs_difference_by_field_across_cases"]
    for field, tolerance in tolerances.items():
        lines.append(f"| `{field}` | {float(tolerance):.9g} | {float(observed[field]):.9g} |")

    lines += [
        "",
        "## 低差速响应",
        "",
        "| 动作 | 控制器 | 8 s净 yaw (rad) | 末1 s平均 abs(yaw rate) (rad/s) | 左/右轮速误差 (rad/s) | 最大单轮 effort (Nm) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case, level in (("diff_left_plus_0.05", ".05"), ("diff_left_plus_0.10", ".10")):
        for label, row in (("P / Ki=0", p_cases[case]), ("PI / Ki=.1", pi_cases[case])):
            drive = row["drive"]
            err = drive["last_1s_mean_abs_qd_tracking_error_rad_s"]
            lines.append(
                f"| {level} | {label} | {drive['net_yaw_delta_unwrapped_rad']:.4f} | "
                f"{drive['mean_abs_actual_yaw_rate_last_1s_rad_s']:.4f} | "
                f"L {err['left']:.3f}, R {err['right']:.3f} | {drive['max_abs_applied_effort_nm']:.3f} |"
            )

    lines += [
        "",
        "## 制动与命令归零",
        "",
        "| 工况 | 控制器 | 两侧 cmd 持续归零 (s) | 末0.3 s最大 abs(v) (m/s) | 末0.3 s最大 abs(w) (rad/s) | 门禁 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for case, label in (("brake_straight_0.20_to_zero", "直行"), ("brake_diff_left_plus_0.80_to_zero", "差速")):
        for run_label, summary_key in (("P / Ki=0", "confirmation_p"), ("PI / Ki=.1", "confirmation_pi")):
            gate = pair["braking_stop_gate"][f"{summary_key}/{case}"]
            zero_t = zero_times[f"{'P' if summary_key == 'confirmation_p' else 'PI'}/{case}"]
            zero_text = "未见持续归零" if zero_t is None else f"{zero_t:.2f}"
            lines.append(
                f"| {label} | {run_label} | {zero_text} | {gate['max_abs_actual_v_mps']:.4f} | "
                f"{gate['max_abs_actual_w_rad_s']:.4f} | {gate['passed']} |"
            )

    effort = pair["physics_effort_gate"]["max_abs_effort_nm_by_run"]
    lines += [
        "",
        f"每组共有 920 个策略采样和 9,200 个 Newton 物理子步记录。Newton 子步采样的最大单轮 effort：P `{effort['confirmation_p']:.6f} Nm`，PI `{effort['confirmation_pi']:.6f} Nm`；protocol 上限为 2 Nm + 1e-5 Nm。图中制动曲线的竖线标记两侧履带速度命令持续归零，不表示车体在该时刻瞬间停止。",
        "",
        "PI 的 Ki=0.1 是未标定的仿真假设；proxy-wheel effort 不是实测电机扭矩。结果只支持这个固定空场物理工况下的速度/偏航/制动响应描述，不证明训练效果、导航成功、绕障或作物安全。",
        "",
        "Raw inputs SHA-256：",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in input_hashes.items())
    lines += ["", "图文件：`confirmation_comparison_cjk.png`。", ""]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {output_png}")
    print(f"WROTE {output_md}")
    print(json.dumps({"status": pair["status"], "cmd_zero_times_s": zero_times}, indent=2))


if __name__ == "__main__":
    main()
