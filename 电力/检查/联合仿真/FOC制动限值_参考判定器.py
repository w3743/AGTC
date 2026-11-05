"""Fail-closed reference adjudicator for the FOC braking contract.

Input is an evidence/configuration JSON, not a firmware configuration. Missing or
unqualified measurements always yield HOLD. No motor or hardware limits are invented.
Run: python FOC制动限值_参考判定器.py evidence.json
"""
import json
import math
import sys
from pathlib import Path

MOTORS = ("left", "right")
REQUIRED = (
    "rs_phase_ohm", "ld_h", "lq_h", "psi_f_wb", "pole_pairs",
    "current_peak_limit_a", "current_rms_limit_a", "current_pulse_limit_s",
    "current_sense_error_a", "current_sense_delay_s", "iq_slew_a_per_s",
    "vbus_uv_v", "vbus_ov_v", "vbus_hard_max_v", "voltage_vector_limit_v",
    "hall_delay_s", "hall_speed_error_mps", "direction_verified",
    "temperature_limit_c", "motor_soak_test_passed",
)
ALLOWED_TRANSITIONS = {
    "DISABLED": {"DRIVE", "FAULT_LATCHED"},
    "DRIVE": {"REGEN", "COAST/SHORT_TRANSITION", "FAULT_LATCHED"},
    "REGEN": {"LOW_SPEED_HOLD", "COAST/SHORT_TRANSITION", "STOPPED", "FAULT_LATCHED"},
    "LOW_SPEED_HOLD": {"COAST/SHORT_TRANSITION", "STOPPED", "FAULT_LATCHED"},
    "COAST/SHORT_TRANSITION": {"STOPPED", "FAULT_LATCHED"},
    "STOPPED": {"DRIVE", "FAULT_LATCHED"},
    "FAULT_LATCHED": {"DISABLED"},
}


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def adjudicate(data):
    blockers, findings = [], []
    if data.get("weakening_enabled") is not False:
        blockers.append("弱磁必须显式关闭；需另行授权并完成退磁/热/故障退出验收")
    short_enabled = data.get("active_short_enabled")
    if type(short_enabled) is not bool:
        blockers.append("须明确三低侧短接是否允许")
    motors = data.get("motors", {})
    for name in MOTORS:
        m = motors.get(name, {})
        missing = [k for k in REQUIRED if k not in m or m[k] is None]
        if missing:
            blockers.append(f"{name}: 缺参数/证据 {', '.join(missing)}")
            continue
        numerical = set(REQUIRED) - {"direction_verified", "motor_soak_test_passed"}
        if any(not finite_number(m[k]) for k in numerical):
            blockers.append(f"{name}: 限值与电机参数须为有限数值")
            continue
        if not all(m[k] > 0 for k in ("rs_phase_ohm", "ld_h", "lq_h", "psi_f_wb",
                                      "voltage_vector_limit_v")) or m["pole_pairs"] != 7:
            blockers.append(f"{name}: 电机参数或极对数不符合当前电机边界")
        if m.get("qualified") is not True:
            blockers.append(f"{name}: 参数未标记为有签署的实测/定级证据")
        if m["direction_verified"] is not True or m["motor_soak_test_passed"] is not True:
            blockers.append(f"{name}: 方向或热浸验收未通过")
        if not (0 < m["current_rms_limit_a"] <= m["current_peak_limit_a"]
                and 0 < m["current_pulse_limit_s"] and m["current_sense_error_a"] >= 0
                and m["current_sense_delay_s"] >= 0 and m["iq_slew_a_per_s"] > 0):
            blockers.append(f"{name}: 电流/斜率限值非法")
        if not (0 < m["vbus_uv_v"] < m["vbus_ov_v"] < m["vbus_hard_max_v"]):
            blockers.append(f"{name}: UV < OV < 硬件最高母线值关系不成立")
        if m["hall_delay_s"] <= 0 or m["hall_speed_error_mps"] < 0:
            blockers.append(f"{name}: Hall 延迟/速度误差无效")
        if short_enabled:
            peak = m.get("short_circuit_peak_a")
            limit = m.get("short_circuit_peak_limit_a")
            if not (finite_number(peak) and finite_number(limit)
                    and 0 < peak + m["current_sense_error_a"] <= limit
                    and peak >= 0 and m.get("short_transient_measured") is True):
                blockers.append(f"{name}: 缺有效短接瞬态或含误差峰值超过脉冲限值")
    if set(motors) != set(MOTORS):
        blockers.append("必须且仅能提供 left/right 两台电机数据")

    acceptance = data.get("acceptance", {})
    required_acceptance = (
        "dual_motor_imbalance_test", "low_speed_transition_test", "reverse_inhibit_test",
        "uv_fault_test", "ov_fault_test", "sink_rejection_test", "independent_stop_measurement",
        "fault_latch_test", "signed_evidence_bundle",
    )
    for k in required_acceptance:
        if acceptance.get(k) is not True:
            blockers.append(f"验收缺失/未通过: {k}")
    if acceptance.get("weakening_test") is True and data.get("weakening_enabled") is not False:
        blockers.append("弱磁无有效授权路径，本判定器拒绝放行")
    if data.get("authorized_boundary") != {"vbus_v": [22.5, 37.8], "speed_kmh_max": 15,
                                             "flat_ground_only": True}:
        blockers.append("工况超出/未明确限定于当前待验证边界 22.5–37.8 V、≤15 km/h、平地")

    transition = data.get("transition", {})
    src, dst = transition.get("from"), transition.get("to")
    if src not in ALLOWED_TRANSITIONS or dst not in ALLOWED_TRANSITIONS.get(src, set()):
        blockers.append(f"非法状态迁移: {src} -> {dst}")
    if transition.get("reverse_command", False) or transition.get("speed_sign_reversed", False):
        blockers.append("反向命令/速度翻转：锁存故障并禁止驱动")
    if transition.get("active_short_requested", False):
        if short_enabled is not True or dst != "COAST/SHORT_TRANSITION":
            blockers.append("主动短接未获授权或切换状态错误")
        if not all(transition.get(k) is True for k in ("entry_current_window_confirmed",
                                                       "dead_time_confirmed", "speed_window_confirmed")):
            blockers.append("主动短接切换前须确认入场电流、死区和速度窗口")
    if dst == "STOPPED" and transition.get("independent_zero_speed_confirmed") is not True:
        blockers.append("STOPPED 必须由独立地面停稳测量确认")

    if blockers:
        verdict = "HOLD"
    else:
        verdict = "HUMAN_REVIEW_REQUIRED"
        findings.append("判定器仅确认合同输入齐备；仍须负责人签署特定工况批准。")
    return {"verdict": verdict, "blockers": blockers, "findings": findings,
            "warning": "非固件、非安全认证、非车辆放行；不得将 HOLD 当作制动能力下界。"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python FOC制动限值_参考判定器.py evidence.json")
        raise SystemExit(2)
    p = Path(sys.argv[1])
    try:
        result = adjudicate(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError, TypeError, OverflowError) as exc:
        result = {"verdict": "HOLD", "blockers": [f"无法读取有效证据 JSON: {exc}"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["verdict"] == "HUMAN_REVIEW_REQUIRED" else 1)


if __name__ == "__main__":
    main()
