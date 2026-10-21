#!/usr/bin/env python3
"""CPU-only paired summary for the frozen 1 ms baseline/mimic response runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUN_NAMES = {
    "baseline": "baseline_1ms_elliptic_1_mu0_8_kf1000",
    "mimic": "mimic_v2_1ms_elliptic_1_mu0_8_kf1000",
}
PROTOCOL = ROOT.parent.parent / "same_side_mimic_v2" / "protocol.json"
CASE_ORDER = [
    "forward", "reverse", "pivot_left_low", "pivot_right_low",
    "arc_left", "arc_right", "pivot_right_high", "pivot_left_high",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def effort_stats(path: Path) -> dict:
    values: dict[str, list[float]] = {case: [] for case in CASE_ORDER}
    rows: dict[str, int] = {case: 0 for case in CASE_ORDER}
    finite = True
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case = row["case"]
            efforts = [float(x) for x in row["native_applied_effort_nm_by_wheel_order"]]
            finite = finite and bool(row.get("finite")) and all(math.isfinite(x) for x in efforts)
            values[case].extend(abs(x) for x in efforts)
            rows[case] += 1
    return {
        case: {
            "physics_rows": rows[case],
            "applied_effort_value_count": len(values[case]),
            "mean_abs_effort_nm": statistics.fmean(values[case]) if values[case] else None,
            "p95_abs_effort_nm": p95(values[case]),
            "max_abs_effort_nm": max(values[case]) if values[case] else None,
            "finite": finite,
        }
        for case in CASE_ORDER
    }


def load_run(mode: str) -> dict:
    name = RUN_NAMES[mode]
    path = RUNS / name
    summary_path = path / "staged_motion_response_summary.json"
    exit_path = RUNS / f"{name}.exit.json"
    summary = read_json(summary_path)
    exit_record = read_json(exit_path)
    return {
        "name": name,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
        "exit_record_path": str(exit_path.relative_to(ROOT)),
        "exit_record": exit_record,
        "summary": summary,
        "effort_by_case": effort_stats(path / "native_physics_efforts.jsonl"),
    }


def case_dict(summary: dict) -> dict[str, dict]:
    return {row["case"]: row for row in summary["cases_summary"]}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def main() -> int:
    baseline = load_run("baseline")
    mimic = load_run("mimic")
    base = baseline["summary"]
    cand = mimic["summary"]
    protocol = read_json(PROTOCOL)
    base_cases = case_dict(base)
    cand_cases = case_dict(cand)
    thresholds = protocol["acceptance"]

    paired = {}
    for case in CASE_ORDER:
        b = base_cases[case]
        m = cand_cases[case]
        br = base.get("same_side_pair_residual_acceptance", {}).get("per_case", {}).get(case, {})
        mr = cand.get("same_side_pair_residual_acceptance", {}).get("per_case", {}).get(case, {})
        be = baseline["effort_by_case"][case]
        me = mimic["effort_by_case"][case]
        metrics = {
            "action": m["action"],
            "initial_state_sha256": m["initial_state_canonical_sha256"],
            "baseline": {
                "yaw_delta_rad": b["drive_yaw_delta_unwrapped_rad"],
                "yaw_rate_last_1s_rad_s": b["drive_mean_w_last_1s_rad_s"],
                "distance_m": b["drive_distance_projected_on_start_heading_m"],
                "drive_mean_v_last_1s_m_s": b["drive_mean_v_last_1s_m_s"],
                "brake_last_0_3s_mean_v_m_s": b["brake_last_0_3s_mean_v_m_s"],
                "brake_last_0_3s_mean_w_rad_s": b["brake_last_0_3s_mean_w_rad_s"],
                "brake_last_0_3s_max_abs_v_m_s": b["brake_last_0_3s_max_abs_v_m_s"],
                "brake_last_0_3s_max_abs_w_rad_s": b["brake_last_0_3s_max_abs_w_rad_s"],
                "brake_final_v_m_s": b["brake_final_v_m_s"],
                "brake_final_w_rad_s": b["brake_final_w_rad_s"],
                "opposite_translation_speed_integral_m": b.get("opposite_translation_speed_integral_m"),
                "opposite_yaw_rate_integral_rad": b.get("opposite_yaw_rate_integral_rad"),
                "effort": be,
                "native_qd_residual": br.get("observed"),
                "native_qd_residual_gates": br.get("criteria"),
                "reference_checks": b["reference_checks"],
                "informational_checks": b["informational_checks"],
            },
            "mimic": {
                "yaw_delta_rad": m["drive_yaw_delta_unwrapped_rad"],
                "yaw_rate_last_1s_rad_s": m["drive_mean_w_last_1s_rad_s"],
                "distance_m": m["drive_distance_projected_on_start_heading_m"],
                "drive_mean_v_last_1s_m_s": m["drive_mean_v_last_1s_m_s"],
                "brake_last_0_3s_mean_v_m_s": m["brake_last_0_3s_mean_v_m_s"],
                "brake_last_0_3s_mean_w_rad_s": m["brake_last_0_3s_mean_w_rad_s"],
                "brake_last_0_3s_max_abs_v_m_s": m["brake_last_0_3s_max_abs_v_m_s"],
                "brake_last_0_3s_max_abs_w_rad_s": m["brake_last_0_3s_max_abs_w_rad_s"],
                "brake_final_v_m_s": m["brake_final_v_m_s"],
                "brake_final_w_rad_s": m["brake_final_w_rad_s"],
                "opposite_translation_speed_integral_m": m.get("opposite_translation_speed_integral_m"),
                "opposite_yaw_rate_integral_rad": m.get("opposite_yaw_rate_integral_rad"),
                "effort": me,
                "native_qd_residual": mr.get("observed"),
                "native_qd_residual_gates": mr.get("criteria"),
                "native_qd_residual_pass": mr.get("passed"),
                "reference_checks": m["reference_checks"],
                "informational_checks": m["informational_checks"],
            },
        }
        keys = [
            "yaw_delta_rad", "yaw_rate_last_1s_rad_s", "distance_m",
            "drive_mean_v_last_1s_m_s", "brake_last_0_3s_mean_v_m_s",
            "brake_last_0_3s_mean_w_rad_s", "brake_last_0_3s_max_abs_v_m_s",
            "brake_last_0_3s_max_abs_w_rad_s", "brake_final_v_m_s",
            "brake_final_w_rad_s", "opposite_translation_speed_integral_m",
            "opposite_yaw_rate_integral_rad",
        ]
        metrics["delta_mimic_minus_baseline"] = {
            key: (metrics["mimic"][key] - metrics["baseline"][key])
            if metrics["mimic"][key] is not None and metrics["baseline"][key] is not None
            else None
            for key in keys
        }
        paired[case] = metrics

    state_hashes_equal = base["initial_state_hashes_by_case"] == cand["initial_state_hashes_by_case"]
    source_hashes_equal = (
        base["training_source_sha256"] == cand["training_source_sha256"]
        and base["diagnostic_source_sha256"] == cand["diagnostic_source_sha256"]
    )
    runtime_physics_equal = base["runtime_physics"] == cand["runtime_physics"]
    drive_config_equal = base["drive_config"] == cand["drive_config"]
    candidate_eq = cand["same_side_treatment"]["solver_equality_audit"]
    candidate_eq_counts = {
        "newton_model_mimic_constraint_count": candidate_eq.get("newton_model_mimic_count"),
        "newton_model_equality_constraint_count": candidate_eq.get("newton_model_equality_constraint_count"),
        "solver_host_neq": candidate_eq.get("solver_host_equality", {}).get("neq"),
        "solver_mjwarp_neq": candidate_eq.get("solver_mjwarp_equality", {}).get("neq"),
        "all_12_expected_joint_pairs_match_once": candidate_eq.get("all_expected_mjEQ_JOINT_pairs_present_once"),
    }

    reference_gate_counts = {}
    informational_failures = {}
    for mode, run in (("baseline", base), ("mimic", cand)):
        rows = case_dict(run)
        gate_names = sorted({name for row in rows.values() for name in row["reference_checks"]})
        reference_gate_counts[mode] = {
            gate: sum(bool(rows[case]["reference_checks"].get(gate)) for case in CASE_ORDER)
            for gate in gate_names
        }
        informational_failures[mode] = {
            case: {key: value for key, value in rows[case]["informational_checks"].items() if not value}
            for case in CASE_ORDER
            if any(not value for value in rows[case]["informational_checks"].values())
        }

    runtime_gate = {}
    for mode, summary in (("baseline", base), ("mimic", cand)):
        phy = summary["runtime_physics"]
        runtime_gate[mode] = {
            "ground_and_14_wheel_mu_status": phy["ground_wheel_mu_kf_runtime_validation"].get("status"),
            "observed_ground_and_14_wheel_mu": phy["ground_wheel_mu_kf_runtime_validation"].get("observed_mu"),
            "observed_ground_and_14_wheel_kf": phy["ground_wheel_mu_kf_runtime_validation"].get("observed_kf"),
            "drive_fixed_config_status": summary["runtime_drive_fixed_validation"].get("status"),
            "native_wheel_joint_count": summary["runtime_drive_fixed_validation"].get("native_wheel_joint_count"),
            "effort_cap_pass": summary.get("native_effort_cap_pass"),
            "effort_cap_nm": summary.get("native_effort_cap_nm"),
            "all_case_reference_checks_pass": all(
                all(row["reference_checks"].values()) for row in summary["cases_summary"]
            ),
            "all_hashes_unchanged": summary.get("training_and_diagnostic_hashes_unchanged_after_run"),
        }

    result = {
        "schema": "isaac-field-paired-motion-response-comparison-v1",
        "status": "complete",
        "purpose": "CPU-only readout of the frozen 1 ms same-reset baseline and same-side mimic runs; no repair verdict.",
        "sources": {
            "protocol_path": str(PROTOCOL.relative_to(ROOT.parent.parent)),
            "protocol_sha256": sha256(PROTOCOL),
            "protocol_acceptance_thresholds": thresholds,
            "baseline": {key: baseline[key] for key in ("name", "summary_path", "summary_sha256", "exit_record_path", "exit_record")},
            "mimic": {key: mimic[key] for key in ("name", "summary_path", "summary_sha256", "exit_record_path", "exit_record")},
            "raw_effort_jsonl": {
                mode: base["output_paths"]["native_physics_efforts"] if mode == "baseline" else cand["output_paths"]["native_physics_efforts"]
                for mode in ("baseline", "mimic")
            },
        },
        "fixed_contract": {
            "seed": base["seed"],
            "case_count": base["case_count"],
            "case_order": CASE_ORDER,
            "settle_drive_brake_s": base["settle_s_drive_s_brake_s"],
            "physics_dt_s": base["physics_dt_s"],
            "decimation": base["decimation"],
            "policy_dt_s": base["policy_dt_s"],
            "solver_treatment": base["solver_treatment"],
            "shape_mu_treatment": base["shape_mu_treatment"],
            "current_drive_interface": base["drive_config"]["interface"],
            "drive_config_identical": drive_config_equal,
            "runtime_physics_identical": runtime_physics_equal,
            "source_hash_maps_identical": source_hashes_equal,
            "source_hashes_unchanged_after_both_runs": all(runtime_gate[m]["all_hashes_unchanged"] for m in runtime_gate),
            "initial_state_hash_maps_identical": state_hashes_equal,
            "initial_state_hashes_match_across_cases_both_runs": (
                base.get("initial_state_hashes_match_across_all_cases")
                and cand.get("initial_state_hashes_match_across_all_cases")
            ),
            "initial_state_hash_per_case": base["initial_state_hashes_by_case"],
            "only_treatment_difference": "same_side_mimic_v2 hook off (baseline) vs on (candidate)",
            "baseline_hook_enabled": base["same_side_treatment"].get("hook_enabled"),
            "mimic_hook_enabled": cand["same_side_treatment"].get("hook_enabled"),
            "runtime_gate": runtime_gate,
            "candidate_mimic_equality_counts": candidate_eq_counts,
            "mimic_residual_run_gate": cand["same_side_pair_residual_acceptance"].get("all_cases_passed_residual_and_effort_gates"),
            "mimic_residual_thresholds": cand["same_side_pair_residual_acceptance"].get("thresholds_from_same_side_mimic_v2_protocol"),
            "baseline_residual_run_gate": base["same_side_pair_residual_acceptance"].get("all_cases_passed_residual_and_effort_gates"),
            "reference_gate_pass_counts_of_8": reference_gate_counts,
            "informational_gate_failures": informational_failures,
        },
        "paired_cases": paired,
        "interpretation": [
            "All observed yaw, speed, distance, effort, brake, and residual values are taken from the original summaries/raw effort JSONL; original run outputs were not edited.",
            "The candidate passes the frozen v2 coupling-residual and effort gates in all eight response actions; this verifies the mimic mechanism in this fixed simulation protocol.",
            "Yaw remains a measured response rather than a target. This comparison does not calibrate friction or establish a production repair; any production decision still requires the prescribed same-state old/new validation.",
        ],
    }
    result = finite_json(result)
    (ROOT / "paired_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(result)
    return 0


def write_markdown(data: dict) -> None:
    limits = data["sources"]["protocol_acceptance_thresholds"]
    lines = [
        "# Same-side mimic 配对运动响应（1 ms）",
        "",
        "本报告由 `summarize_paired_comparison.py` 直接读取两组原始 summary、effort JSONL、冻结 v2 coupling protocol 与 runner exit 记录生成；未改写原始结果。该八动作响应组使用 1 s zero-action settle、2 s drive、5 s brake。",
        "",
        "## 配对与门禁",
        "",
        f"- baseline / mimic 实际退出码：{data['sources']['baseline']['exit_record'].get('actual_exit_code')} / {data['sources']['mimic']['exit_record'].get('actual_exit_code')}。",
        f"- 固定物理步长 {data['fixed_contract']['physics_dt_s']} s，decimation {data['fixed_contract']['decimation']}，策略步长 {data['fixed_contract']['policy_dt_s']} s；cone `{data['fixed_contract']['solver_treatment']['cone']}`、impratio {data['fixed_contract']['solver_treatment']['impratio']}、shape μ {data['fixed_contract']['runtime_gate']['baseline']['observed_ground_and_14_wheel_mu'][0]}、ground/wheel kf {data['fixed_contract']['runtime_gate']['baseline']['observed_ground_and_14_wheel_kf'][0]}。",
        f"- 两次真实 reset 的逐动作 hash map 一致：{data['fixed_contract']['initial_state_hash_maps_identical']}；八动作均为同一 reset hash：{data['fixed_contract']['initial_state_hashes_match_across_cases_both_runs']}。hash：`{next(iter(data['fixed_contract']['initial_state_hash_per_case'].values()))}`。",
        f"- 两组 drive/runtime physics/source-hash map 相同：drive={data['fixed_contract']['drive_config_identical']}，runtime physics={data['fixed_contract']['runtime_physics_identical']}，source maps={data['fixed_contract']['source_hash_maps_identical']}；两组运行后冻结源均未变：{data['fixed_contract']['source_hashes_unchanged_after_both_runs']}。仅 hook 从 baseline={data['fixed_contract']['baseline_hook_enabled']} 切到 mimic={data['fixed_contract']['mimic_hook_enabled']}。",
        f"- Ground+14轮 μ/kf 和 fixed drive runtime gates 两组均通过；14轮实际 effort cap ≤2 Nm 两组均通过。候选 Newton model/host/MJWarp equality 计数：`{json.dumps(data['fixed_contract']['candidate_mimic_equality_counts'], ensure_ascii=False)}`。",
        f"- Frozen v2 coupling gates：native qd-pair abs p95 ≤{limits['last_1s_native_physics_qd_pair_abs_p95_max_rad_s']:.5f} rad/s，abs max ≤{limits['last_1s_native_physics_qd_pair_abs_max_rad_s']:.5f} rad/s，policy q-pair abs max ≤{limits['last_1s_policy_q_pair_abs_max_rad']:.3f} rad；baseline all-case={data['fixed_contract']['baseline_residual_run_gate']}，mimic all-case={data['fixed_contract']['mimic_residual_run_gate']}。",
        "",
        "### 八动作并列",
        "",
        "| 动作 | 实际Δyaw rad baseline→mimic | Δyaw | 末1 s yaw rate rad/s baseline→mimic | 距离 m baseline→mimic | 制动末0.3 s max |v| m/s baseline→mimic | max |w| rad/s baseline→mimic | 回摆积分 m 或 rad/s·s baseline→mimic | max / mean abs effort Nm baseline→mimic | mimic qd p95 / max / policy q max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, row in data["paired_cases"].items():
        b, m, d = row["baseline"], row["mimic"], row["delta_mimic_minus_baseline"]
        residual = m["native_qd_residual"] or {}
        rebound_key = "opposite_yaw_rate_integral_rad" if case.startswith("pivot_") else "opposite_translation_speed_integral_m"
        rebound_unit = "rad/s·s" if case.startswith("pivot_") else "m"
        rebound_b = "—" if b[rebound_key] is None else f"{b[rebound_key]:.4f} {rebound_unit}"
        rebound_m = "—" if m[rebound_key] is None else f"{m[rebound_key]:.4f} {rebound_unit}"
        lines.append(
            f"| {case} `{row['action']}` | {b['yaw_delta_rad']:.4f} → {m['yaw_delta_rad']:.4f} | {d['yaw_delta_rad']:+.4f} | {b['yaw_rate_last_1s_rad_s']:.4f} → {m['yaw_rate_last_1s_rad_s']:.4f} | {b['distance_m']:.4f} → {m['distance_m']:.4f} | {b['brake_last_0_3s_max_abs_v_m_s']:.5f} → {m['brake_last_0_3s_max_abs_v_m_s']:.5f} | {b['brake_last_0_3s_max_abs_w_rad_s']:.5f} → {m['brake_last_0_3s_max_abs_w_rad_s']:.5f} | {rebound_b} → {rebound_m} | {b['effort']['max_abs_effort_nm']:.4f} / {b['effort']['mean_abs_effort_nm']:.4f} → {m['effort']['max_abs_effort_nm']:.4f} / {m['effort']['mean_abs_effort_nm']:.4f} | {residual.get('native_physics_qd_pair_abs_p95_rad_s', float('nan')):.4f} / {residual.get('native_physics_qd_pair_abs_max_rad_s', float('nan')):.4f} / {residual.get('policy_q_pair_abs_max_rad', float('nan')):.6f} ({m['native_qd_residual_pass']}) |"
        )
    lines += [
        "",
        "各组共同运动 reference gates 每动作全部通过（有限值、effort cap、无损伤/碰撞/episode done、姿态与制动速度门槛、对应方向/位移或 pivot/arc 检查）；记录的 informational gate 单独保留，baseline `pivot_right_low` 的 `signed_pivot_yaw_delta_ge_0_1rad` 为 false，不影响 reference-gate 通过状态。各组 8 个动作的全 reference gate 计数见 `paired_comparison.json`。",
        "",
        "## 结论范围",
        "",
        "mimic 候选在此固定仿真协议下通过八动作耦合残差门槛，pivot 与弧线的 yaw 响应幅度普遍高于 unlinked baseline；直行/倒车 yaw 仍接近零。制动尾部的 |v|/|w| 数值符合本组原门槛，但回摆积分并非各动作全面改善。该结果验证隔离机制响应，不是摩擦标定或生产修复结论；production 仍需经同初态、旧/新物理的既定验收。",
        "",
        f"Frozen coupling protocol: `{data['sources']['protocol_path']}` (SHA-256 `{data['sources']['protocol_sha256']}`). Runner exits and source summaries are enumerated in `paired_comparison.json`.",
        "",
    ]
    (ROOT / "paired_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
