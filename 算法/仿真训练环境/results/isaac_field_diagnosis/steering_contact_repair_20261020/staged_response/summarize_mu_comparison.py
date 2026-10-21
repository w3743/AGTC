#!/usr/bin/env python3
"""Compare the frozen mu=0.8 baseline with the two staged friction runs."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response"
RUNS = (("0.8", "baseline_5ms_pyramidal_1_kf1000"),
        ("0.4", "mu_0_4_5ms_pyramidal_1_kf1000"),
        ("1.2", "mu_1_2_5ms_pyramidal_1_kf1000"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    json_path, md_path = HERE / "mu_comparison.json", HERE / "mu_comparison.md"
    if json_path.exists() or md_path.exists():
        raise SystemExit("refusing to overwrite mu comparison output")
    runs, summaries = {}, []
    for mu_text, name in RUNS:
        folder = HERE / "runs" / name
        summary = load(folder / "staged_motion_response_summary.json")
        exit_record = load(HERE / "runs" / f"{name}.exit.json")
        assert exit_record["actual_exit_code"] == 0 and summary["status"] == "complete"
        labels = summary["runtime_physics"]["newton_runtime_shape_labels"]
        arrays = summary["runtime_physics"]["newton_runtime_shape_material_arrays_in_label_order"]
        contact = [i for i, label in enumerate(labels) if label.endswith("/Field/Ground") or
                   ("_wheel_link/" in label and label.endswith("/wheel_collision_visual"))]
        assert len(contact) == 15
        mu_values = [float(arrays["shape_material_mu"][i]) for i in contact]
        assert all(math.isclose(v, float(mu_text), rel_tol=0.0, abs_tol=1e-6) for v in mu_values)
        material = {key: sorted(set(float(arrays[f"shape_material_{key}"][i]) for i in contact))
                    for key in ("kf", "ke", "kd")}
        assert material == {"kf": [1000.0], "ke": [2500.0], "kd": [100.0]}
        trajectory_path, effort_path = folder / "policy_trajectories.jsonl", folder / "native_physics_efforts.jsonl"
        trajectories, efforts = jsonl(trajectory_path), jsonl(effort_path)
        effort_max = max(abs(float(t)) for row in efforts for t in row["native_applied_effort_nm_by_wheel_order"])
        assert len(trajectories) == 1280 and len(efforts) == 12800
        assert math.isclose(effort_max, summary["native_effort_max_abs_nm_all_cases"], rel_tol=0.0, abs_tol=1e-6)
        runs[mu_text] = {"name": name, "summary": summary, "exit": exit_record,
                         "mu_values": mu_values, "contact_material_kf_ke_kd": material,
                         "trajectory_rows": len(trajectories), "effort_rows": len(efforts),
                         "raw_effort_max_nm": effort_max,
                         "summary_path": str((folder / "staged_motion_response_summary.json").relative_to(ROOT)),
                         "trajectory_path": str(trajectory_path.relative_to(ROOT)),
                         "effort_path": str(effort_path.relative_to(ROOT))}
        summaries.append(summary)

    first = summaries[0]
    shared = ("seed", "start_xy_yaw", "goal_xy", "case_count", "cases", "physics_dt_s", "policy_dt_s", "decimation")
    for summary in summaries[1:]:
        assert all(summary[k] == first[k] for k in shared)
        assert summary["solver_treatment"] == first["solver_treatment"]
        assert summary["drive_config"] == first["drive_config"]
        assert summary["initial_state_hashes_by_case"] == first["initial_state_hashes_by_case"]
        assert summary["training_source_sha256"] == first["training_source_sha256"]
        assert summary["diagnostic_source_sha256"] == first["diagnostic_source_sha256"]
        assert summary["runtime_physics"]["solver_config"] == first["runtime_physics"]["solver_config"]
        assert {k: v for k, v in summary["runtime_physics"]["shape_material_config"].items() if k != "mu"} == \
               {k: v for k, v in first["runtime_physics"]["shape_material_config"].items() if k != "mu"}
        assert summary["training_and_diagnostic_hashes_unchanged_after_run"] is True

    cases = []
    for index, case_name in enumerate(first["cases"]):
        records = {mu: runs[mu]["summary"]["cases_summary"][index] for mu, _ in RUNS}
        actions = [records[mu]["action"] for mu, _ in RUNS]
        assert actions[0] == actions[1] == actions[2]
        by_mu = {}
        for mu, record in records.items():
            failed = [key for group in (record["reference_checks"], record["informational_checks"])
                      for key, value in group.items() if value is False]
            by_mu[mu] = {key: record.get(key) for key in (
                "drive_distance_projected_on_start_heading_m", "drive_yaw_delta_unwrapped_rad",
                "drive_mean_v_last_1s_m_s", "drive_mean_w_last_1s_rad_s",
                "brake_last_0_3s_max_abs_v_m_s", "brake_last_0_3s_max_abs_w_rad_s",
                "brake_final_v_m_s", "brake_final_w_rad_s", "opposite_translation_speed_integral_m",
                "opposite_yaw_rate_integral_rad", "native_effort_max_abs_nm", "native_effort_cap_pass",
                "max_damage_count", "max_collision_count", "max_abs_roll_rad", "max_abs_pitch_rad")}
            by_mu[mu]["failed_checks"] = failed
        cases.append({"case": case_name, "action": actions[0], "response_by_mu": by_mu})

    result = {"schema": "staged-mu-response-comparison-v1", "status": "complete",
              "comparison": "same reset, seed, action matrix, dt, solver treatment, drive config, and source hashes; only default shape mu varies",
              "calibration_limit": "These simulation responses do not calibrate real-world friction and do not select a production coefficient.",
              "same_initial_hashes_across_all_runs": True,
              "initial_state_hashes_by_case": first["initial_state_hashes_by_case"],
              "runs": {mu: {k: v for k, v in run.items() if k != "summary"} |
                       {"summary_path": str((HERE / "runs" / run["name"] / "staged_motion_response_summary.json").relative_to(ROOT)),
                        "status": run["summary"]["status"], "initial_state_hashes_match": run["summary"]["initial_state_hashes_match_across_all_cases"],
                        "effort_cap_pass": run["summary"]["native_effort_cap_pass"],
                        "runtime_solver_treatment": run["summary"]["solver_treatment"]}
                       for mu, run in runs.items()},
              "fixed_conditions": {key: first[key] for key in shared} | {"solver_treatment": first["solver_treatment"],
                                  "drive_config": first["drive_config"], "runtime_solver_config": first["runtime_physics"]["solver_config"]},
              "cases": cases}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# 摩擦系数响应对照（5 ms，pyramidal，impratio=1）", "",
             "三组采用相同 seed、八动作、驱动配置及逐case初态哈希。Ground 与 14 个轮碰撞形状的运行时 μ 均核验通过；KF/KE/KD 均为 1000/2500/100。每组策略轨迹 1280 行、物理步力矩 12800 行；原始文件路径列在文末。", "",
             "本对照只反映当前仿真材料设置下的响应，不用于真实摩擦标定，也不据此选择生产系数。", "",
             "| μ | exit | Ground+14轮实测 μ | 最大力矩 Nm | 策略轨迹 / effort 行数 | summary |", "|---:|---:|---:|---:|---:|---|"]
    for mu, _ in RUNS:
        run = runs[mu]
        lines.append(f"| {mu} | {run['exit']['actual_exit_code']} | {min(run['mu_values']):.6f}–{max(run['mu_values']):.6f} | "
                     f"{run['raw_effort_max_nm']:.6f} | {run['trajectory_rows']} / {run['effort_rows']} | `{run['summary_path']}` |")
    lines += ["", f"共同初态哈希：`{first['initial_state_hashes_by_case'][first['cases'][0]]}`。", "",
              "每格顺序为 drive distance / yaw delta / last-1s yaw rate / 最大轮力矩；之后列制动末 0.3 s 最大 |v|、|w| 及反向积分。"]
    lines += ["", "| 动作 | μ=0.8 | μ=0.4 | μ=1.2 |"]
    lines += ["|---|---|---|---|"]
    for case in cases:
        cells = []
        for mu, _ in RUNS:
            m = case["response_by_mu"][mu]
            rebound = (f"反向位移 {m['opposite_translation_speed_integral_m']:.4f} m" if m["opposite_translation_speed_integral_m"] is not None
                       else f"反向偏航 {m['opposite_yaw_rate_integral_rad']:.4f} rad")
            cells.append(f"d={m['drive_distance_projected_on_start_heading_m']:.3f} m; Δyaw={m['drive_yaw_delta_unwrapped_rad']:.3f} rad; "
                         f"w₁={m['drive_mean_w_last_1s_rad_s']:.3f}; τmax={m['native_effort_max_abs_nm']:.3f} Nm; "
                         f"brake max|v/w|={m['brake_last_0_3s_max_abs_v_m_s']:.5f}/{m['brake_last_0_3s_max_abs_w_rad_s']:.5f}; {rebound}")
        lines.append(f"| {case['case']} {case['action'][:2]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "完整逐策略步轨迹和逐物理步力矩见各 run 目录中的 `policy_trajectories.jsonl` 与 `native_physics_efforts.jsonl`。"]
    lines += ["", "原始数据文件："]
    for mu, _ in RUNS:
        run = runs[mu]
        lines.append(f"- μ={mu}：轨迹 `{run['trajectory_path']}`；力矩 `{run['effort_path']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
