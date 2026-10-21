#!/usr/bin/env python3
"""Summarize already completed, paired minimal-input evaluations (read-only)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "results/isaac_field_improvement/fresh_minimal_inputs_20261020"
HERE = Path(__file__).resolve().parent
STAGE_DIR = HERE / "stage_pairing"
SEEDS = (87001, 87002, 87003, 87004)
STAGES = (
    ("stage01_near", "evaluation_stage01_near_free_87001_87004.json"),
    ("stage02_steer30", "evaluation_stage02_steer30_free_87001_87004.json"),
    ("stage03_steer60", "evaluation_stage03_steer60_free_87001_87004.json"),
    ("stage04_turn90_final", "evaluation_free_final_87001_87004.json"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def summarize_row(row: dict) -> dict:
    traj = row["trajectory"]
    if not traj or len(traj) != int(row["steps"]):
        raise ValueError(f"seed {row['seed']}: missing or inconsistent trajectory")
    distances = [float(sample["after"]["goal_distance_m"]) for sample in traj]
    min_i = min(range(len(traj)), key=distances.__getitem__)
    min_sample = traj[min_i]
    final = traj[-1]["after"]
    cmds = [sample["command_v_w"] for sample in traj]
    actual = [sample["after"]["actual_v_w"] for sample in traj]
    tracks = [sample["track_command_lr_mps"] for sample in traj]
    actions = [sample["action_4"] for sample in traj]
    near = [d <= 0.4 for d in distances]
    max_near_run = run = 0
    for flag in near:
        run = run + 1 if flag else 0
        max_near_run = max(max_near_run, run)
    initial_yaw = float(traj[0]["before"]["pose_xy_yaw"][2])
    final_yaw = float(final["pose_xy_yaw"][2])
    mean_abs_cmd_w = sum(abs(float(x[1])) for x in cmds) / len(cmds)
    mean_abs_actual_w = sum(abs(float(x[1])) for x in actual) / len(actual)
    path_len = sum(
        math.hypot(
            float(sample["after"]["pose_xy_yaw"][0]) - float(sample["before"]["pose_xy_yaw"][0]),
            float(sample["after"]["pose_xy_yaw"][1]) - float(sample["before"]["pose_xy_yaw"][1]),
        ) for sample in traj
    )
    flat_commands = [float(x[0]) for x in cmds]
    return {
        "seed": int(row["seed"]),
        "success": bool(row["success"]),
        "reason": row["reason"],
        "timed_out": bool(row["timed_out"]),
        "steps": int(row["steps"]),
        "raw_return": float(row["raw_return"]),
        "damage": float(row["damage"]),
        "collisions": int(row["collisions"]),
        "start_distance_m": distances[0] if len(traj) == 1 else float(traj[0]["before"]["goal_distance_m"]),
        "min_distance_m": distances[min_i],
        "min_distance_step": int(min_sample["step"]),
        "min_distance_hold_s": float(min_sample["after"].get("hold_s", 0.0)),
        "min_distance_actual_v_w": [float(x) for x in min_sample["after"]["actual_v_w"]],
        "min_distance_command_v_w": [float(x) for x in min_sample["command_v_w"]],
        "min_distance_track_commands_lr_mps": [float(x) for x in min_sample["track_command_lr_mps"]],
        "final_distance_m": float(row["final_distance"]),
        "final_goal_frame_xy_m": [float(x) for x in final["goal_robot_frame_m"]],
        "final_goal_bearing_rad": float(final["goal_bearing_rad"]),
        "final_actual_v_w": [float(x) for x in final["actual_v_w"]],
        "final_command_v_w": [float(x) for x in traj[-1]["command_v_w"]],
        "final_track_commands_lr_mps": [float(x) for x in traj[-1]["track_command_lr_mps"]],
        "max_abs_action_component": max(abs(float(v)) for values in actions for v in values),
        "mean_command_v_mps": sum(float(x[0]) for x in cmds) / len(cmds),
        "mean_actual_v_mps": sum(float(x[0]) for x in actual) / len(actual),
        "reverse_command_fraction": sum(v < 0 for v in flat_commands) / len(flat_commands),
        "forward_command_fraction": sum(v > 0 for v in flat_commands) / len(flat_commands),
        "mean_abs_command_yaw_rate_rad_s": mean_abs_cmd_w,
        "mean_abs_actual_yaw_rate_rad_s": mean_abs_actual_w,
        "abs_command_actual_yaw_ratio": mean_abs_cmd_w / mean_abs_actual_w if mean_abs_actual_w > 1e-12 else None,
        "mean_left_track_command_mps": sum(float(x[0]) for x in tracks) / len(tracks),
        "mean_right_track_command_mps": sum(float(x[1]) for x in tracks) / len(tracks),
        "path_length_m": path_len,
        "net_body_yaw_change_rad": wrap_angle(final_yaw - initial_yaw),
        "steps_within_0p4m": sum(near),
        "max_consecutive_steps_within_0p4m": max_near_run,
        "max_hold_s": max(float(sample["after"].get("hold_s", 0.0)) for sample in traj),
    }


def main() -> None:
    pair = read(STAGE_DIR / "stage_pairing_summary.json")
    pairing = pair["pairing"]
    if not pairing["all_same_reset_and_goal_for_each_seed"] or not pairing["all_trajectories_start_at_recorded_reset"]:
        raise ValueError("paired initial-condition verification failed")
    if set(pair["evaluation_config"]["evaluation_seeds"]) != set(SEEDS):
        raise ValueError("unexpected evaluation seed set")
    reports = {}
    report_hashes = {}
    for name, filename in STAGES:
        path = STAGE_DIR / filename if name != "stage04_turn90_final" else TRIAL / filename
        report = read(path)
        if set(report["seeds"]) != set(SEEDS):
            raise ValueError(f"{name}: unexpected seed set {report['seeds']}")
        report_hashes[name] = sha256(path)
        reports[name] = {int(row["seed"]): summarize_row(row) for row in report["rows"]}
    stages_by_name = {item["stage"]: item for item in pair["stages"]}
    result = {
        "schema": "minimal-inputs-stage-pairing-analysis-v1",
        "scope": "Only the authorized free-space development seeds 87001-87004; no holdout seeds.",
        "evaluation_execution_status": {
            "evaluation_subprocess_exit_code_recorded_in_runner_log": 0,
            "outer_wsl_runner_exit_code": 1,
            "outer_runner_failure": "PowerShell-created CRLF made bash pass a trailing carriage return to exit; the runner script has since been normalized to LF.",
            "all_three_evaluation_reports_and_pairing_summary_written": True,
            "evaluation_repeated_after_wrapper_error": False,
        },
        "paired_protocol": pair["evaluation_config"],
        "pairing_checks": pairing,
        "stage_checkpoint_records": pair["stages"],
        "evaluation_report_sha256": report_hashes,
        "aggregate_by_stage": {},
        "episodes_by_seed": {},
    }
    for name, _ in STAGES:
        rows = list(reports[name].values())
        result["aggregate_by_stage"][name] = {
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "clean_successes": sum(row["success"] and row["damage"] == 0 and row["collisions"] == 0 for row in rows),
            "mean_final_distance_m": sum(row["final_distance_m"] for row in rows) / len(rows),
            "mean_raw_return": sum(row["raw_return"] for row in rows) / len(rows),
            "damaged_episodes": sum(row["damage"] > 0 for row in rows),
            "total_collisions": sum(row["collisions"] for row in rows),
            "timed_out_episodes": sum(row["timed_out"] for row in rows),
            "mean_abs_command_yaw_rate_rad_s": sum(row["mean_abs_command_yaw_rate_rad_s"] for row in rows) / len(rows),
            "mean_abs_actual_yaw_rate_rad_s": sum(row["mean_abs_actual_yaw_rate_rad_s"] for row in rows) / len(rows),
            "max_action_component": max(row["max_abs_action_component"] for row in rows),
        }
    for seed in SEEDS:
        result["episodes_by_seed"][str(seed)] = {name: reports[name][seed] for name, _ in STAGES}
    out_json = HERE / "stage_pairing_analysis.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    names = [name for name, _ in STAGES]
    lines = [
        "# 同一 free 初态的阶段检查点配对结果",
        "",
        "本文件由已完成的 evaluation JSON 只读汇总生成；没有重跑评估。四个阶段仅使用 87001–87004，goal distance 1–3 m、goal angle ±90°、plants=0、flat terrain、environment seed 80003、deterministic、distance_cost=0。",
        "",
        "配对检查确认每个 seed 的 reset 初态、goal、plant layout hash、轨迹起点完全一致，且 action/drive/damage/observation 合同与原 free baseline 相同。阶段 checkpoint SHA 与报告文件 SHA 见同目录 `stage_pairing_analysis.json`。",
        "",
        "执行记录：评测子命令实际 exit 0（runner log 的 `ACTUAL_EXIT_CODE=0`），三份 evaluation JSON 与 pairing summary 均已完整写出；外层 WSL runner 最终 exit 1，原因是其 PowerShell 写入的 CRLF 让 bash 在 `exit` 参数带入尾随回车。runner 已修成 LF；为避免重复 12 个 GPU 回合，没有重跑。两个退出状态均保留，不把外层 runner 标为通过。",
        "",
        "| 检查点 | clean success | 平均最终距离 | 平均 raw return | 损伤回合 | 碰撞 | 超时回合 | 平均 abs 命令 yaw | 平均 abs 实际 yaw | 最大 action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "stage01_near": "stage01 near",
        "stage02_steer30": "stage02 steer30",
        "stage03_steer60": "stage03 steer60",
        "stage04_turn90_final": "stage04 final",
    }
    for name in names:
        agg = result["aggregate_by_stage"][name]
        lines.append(
            f"| {labels[name]} | {agg['clean_successes']}/4 | {agg['mean_final_distance_m']:.3f} m | "
            f"{agg['mean_raw_return']:.3f} | {agg['damaged_episodes']} | {agg['total_collisions']} | "
            f"{agg['timed_out_episodes']}/4 | {agg['mean_abs_command_yaw_rate_rad_s']:.3f} rad/s | "
            f"{agg['mean_abs_actual_yaw_rate_rad_s']:.5f} rad/s | {agg['max_action_component']:.3f} |"
        )
    lines += [
        "",
        "每格为 `clean success; 最近距离 → 最终距离`（m）。`*` 表示曾进入 0.4 m，但该回合没有 clean success。",
        "",
        "| Seed | stage01 | stage02 | stage03 | stage04 final |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        cells = []
        for name in names:
            row = reports[name][seed]
            marker = "✓" if row["success"] and row["damage"] == 0 and row["collisions"] == 0 else "✗"
            star = "*" if row["min_distance_m"] <= 0.4 and not row["success"] else ""
            cells.append(f"{marker}{star} {row['min_distance_m']:.2f} → {row['final_distance_m']:.2f}")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 轨迹中可区分的变化",
        "",
        "stage01 与 stage02 都在 87002、87004 成功（2/4），stage03 在四个 seed 上均失败（0/4），stage04 仍为 0/4。stage01→02 的均值变化不一致：成功数相同，平均 raw return 从 4.596 降到 3.365、平均末距从 0.917 增到 1.340 m；stage02→03 则降到 raw return −5.175、末距 2.980 m。",
        "",
        "两个此前成功的初态 87002 和 87004 在 stage03 分别最近到 0.10 m（step "
        f"{reports['stage03_steer60'][87002]['min_distance_step']}）与 0.16 m（step {reports['stage03_steer60'][87004]['min_distance_step']}），但最终退到 "
        f"{reports['stage03_steer60'][87002]['final_distance_m']:.2f} m 与 {reports['stage03_steer60'][87004]['final_distance_m']:.2f} m；两者均不是成功回合，且均跑满 400 步。轨迹中最大连续 0.4 m 内步数分别为 "
        f"{reports['stage03_steer60'][87002]['max_consecutive_steps_within_0p4m']} 与 {reports['stage03_steer60'][87004]['max_consecutive_steps_within_0p4m']}，最大 hold timer 分别为 "
        f"{reports['stage03_steer60'][87002]['max_hold_s']:.3f} s 与 {reports['stage03_steer60'][87004]['max_hold_s']:.3f} s。该证据支持接近后未保持稳定，而非简单“完全不会向目标移动”。",
        "",
        "转向迹象在 stage01–04 都存在：大幅 yaw command 与微小 actual yaw 同时出现；所以该差异不是 stage03 才开始。stage03 的轨迹同时有接近后距离明显反弹，且 87001/87003 的最近距离也分别只有 1.39/1.24 m。这里仅描述命令和记录响应的相关性，不据此确定物理/控制根因。最大 action 分量低于 1，未饱和。",
        "",
        "## 分seed关键轨迹数据",
        "",
        "`actual v/w` 与 `command v/w` 的顺序均为线速度 m/s、角速度 rad/s；履带命令按 left/right m/s。完整逐步轨迹仍在每个 `evaluation_*.json`。",
        "",
        "| Seed / stage | min step, distance | min 时 actual v/w | min 时 command v/w | min 时履带命令 L/R | final distance | 最终 actual v/w | 最终 command v/w |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for name in ("stage02_steer30", "stage03_steer60", "stage04_turn90_final"):
            row = reports[name][seed]
            def pair(values):
                return f"[{float(values[0]):.3f}, {float(values[1]):.4f}]"
            label = labels[name]
            lines.append(
                f"| {seed} / {label} | {row['min_distance_step']}, {row['min_distance_m']:.3f} m | "
                f"{pair(row['min_distance_actual_v_w'])} | {pair(row['min_distance_command_v_w'])} | "
                f"[{row['min_distance_track_commands_lr_mps'][0]:.3f}, {row['min_distance_track_commands_lr_mps'][1]:.3f}] | "
                f"{row['final_distance_m']:.3f} m | {pair(row['final_actual_v_w'])} | {pair(row['final_command_v_w'])} |"
            )
    lines += [
        "",
        "本结果只代表四个开发初态，不能外推到未观测种子或田间导航。stage03 的最近距离进入 0.4 m，但任务未成功；应把它和训练近期 success 在 steer60/turn90 阶段骤降、以及所有阶段 command/actual yaw 差异一起看，先区分课程遗忘/hold 减速与动作到实际转向响应，再决定下一轮训练课程。",
    ]
    out_md = HERE / "stage_pairing_findings.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
