"""水泵本地卸载纸面预算；不读取或修改EDA运行时。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\pinoeer\AGTC\电力\水泵驱动模块")
# 2026-11-07：审查工作树（agtc-brake-review-20261105）已并入 AGTC 并删除；
# 审查侧原始字节按原相对路径归档于此，供本脚本历史复跑。
WORKTREE = Path(r"C:\pinoeer\AGTC\归档\审查分支差异\电力\水泵驱动模块")
OUT = HERE / "水泵快速卸载预算.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bus_voltage(t_s: float, initial_v: float, total_power_w: float, capacitance_f: float) -> float:
    v2 = initial_v * initial_v - 2.0 * total_power_w * t_s / capacitance_f
    return math.sqrt(max(0.0, v2))


def filter_crossing(
    *,
    threshold_v: float,
    gain: float,
    tau_s: float,
    initial_filter_v: float,
    initial_bus_v: float,
    full_power_w: float,
    capacitance_f: float,
    baseline_w: float,
    max_time_s: float = 0.050,
    dt_s: float = 10e-6,
) -> dict[str, float | None]:
    """First-order RC response while the pump remains at full input power."""
    total_power = full_power_w + baseline_w
    filt = initial_filter_v
    steps = math.ceil(max_time_s / dt_s)
    for n in range(steps):
        t0 = n * dt_s
        vmid = bus_voltage(t0 + 0.5 * dt_s, initial_bus_v, total_power, capacitance_f)
        measured = gain * vmid
        filt += (measured - filt) * (1.0 - math.exp(-dt_s / tau_s))
        if filt <= threshold_v:
            t_cross = (n + 1) * dt_s
            return {
                "continuous_crossing_s": t_cross,
                "bus_v_at_crossing": bus_voltage(t_cross, initial_bus_v, total_power, capacitance_f),
                "filtered_v_at_crossing": filt,
            }
    return {"continuous_crossing_s": None, "bus_v_at_crossing": None, "filtered_v_at_crossing": None}


def voltage_at_50ms(
    *,
    shed_deadline_s: float,
    initial_v: float,
    capacitance_f: float,
    baseline_w: float,
    pump_full_w: float,
    pump_shed_w: float,
    window_s: float,
) -> float:
    """Conservative tail model: full pump draw persists until deadline, then steps down."""
    t_high = min(max(shed_deadline_s, 0.0), window_s)
    energy_j = (
        baseline_w * window_s
        + pump_full_w * t_high
        + pump_shed_w * (window_s - t_high)
    )
    v2 = initial_v * initial_v - 2.0 * energy_j / capacitance_f
    return math.sqrt(max(0.0, v2))


def candidate(
    *,
    name: str,
    threshold_v: float,
    gain_min: float,
    gain_max: float,
    tau_s: float,
    initial_filter_v: float,
    sample_s: float,
    decision_s: float,
    gpio_s: float,
    power_tail_s: float,
    normal_bus_v: float,
    initial_bus_v: float,
    capacitance_f: float,
    baseline_w: float,
    pump_full_w: float,
    pump_shed_w: float,
    window_s: float,
) -> dict:
    # 高增益使同一BUS读数偏高，成为低压门限的最晚触发角。
    crossing = filter_crossing(
        threshold_v=threshold_v,
        gain=gain_max,
        tau_s=tau_s,
        initial_filter_v=initial_filter_v,
        initial_bus_v=initial_bus_v,
        full_power_w=pump_full_w,
        capacitance_f=capacitance_f,
        baseline_w=baseline_w,
    )
    t_cross = crossing["continuous_crossing_s"]
    if t_cross is None:
        control_s = None
        shed_s = None
        v50 = None
    else:
        control_s = float(t_cross) + sample_s + decision_s + gpio_s
        shed_s = control_s + power_tail_s
        v50 = voltage_at_50ms(
            shed_deadline_s=shed_s,
            initial_v=initial_bus_v,
            capacitance_f=capacitance_f,
            baseline_w=baseline_w,
            pump_full_w=pump_full_w,
            pump_shed_w=pump_shed_w,
            window_s=window_s,
        )
    actual_trip_min = threshold_v / gain_max
    actual_trip_max = threshold_v / gain_min
    normal_meas_min = normal_bus_v * gain_min
    normal_meas_max = normal_bus_v * gain_max
    return {
        "name": name,
        "threshold_code_bus_equivalent_v": threshold_v,
        "full_chain_gain_range": [gain_min, gain_max],
        "static_actual_trip_range_v": [actual_trip_min, actual_trip_max],
        "normal_22_5v_reported_range_v": [normal_meas_min, normal_meas_max],
        "normal_22_5v_false_shed_possible": normal_meas_min <= threshold_v,
        "tau_s": tau_s,
        "initial_filter_state_v_bus_equivalent": initial_filter_v,
        "worst_late_continuous_crossing": crossing,
        "post_cross_sample_s": sample_s,
        "decision_s": decision_s,
        "gpio_s": gpio_s,
        "power_tail_s": power_tail_s,
        "command_time_ms": None if control_s is None else 1000.0 * control_s,
        "power_at_or_below_5w_deadline_ms": None if shed_s is None else 1000.0 * shed_s,
        "v_bus_at_50ms_with_full_power_until_deadline_v": v50,
    }


def build() -> dict:
    initial_v = 22.5
    c_f = 0.05286
    pump_full_w = 70.0
    pump_shed_w = 5.0
    foc_logic_w_each = 1.2
    independent_source_a = 0.0193
    baseline_w = 2.0 * foc_logic_w_each + initial_v * independent_source_a
    window_s = 0.050
    foc_vmin_v = 21.10762

    # ADC_BUS现行名义网络：110k+110k上臂、10k下臂，PA前串1k、并100nF。
    rtop_nom = 220_000.0
    rbottom_nom = 10_000.0
    rseries_nom = 1_000.0
    c_nom = 100e-9
    rth_nom = (rtop_nom * rbottom_nom / (rtop_nom + rbottom_nom)) + rseries_nom
    tau_nom = rth_nom * c_nom

    # BOM精确料号的温漂边界作为资格预算：上臂50 ppm/C、下臂25 ppm/C，
    # 1k串阻按100 ppm/C筛查；25C到60C最大偏差35C；100nF X7R
    # 以初始+10%与X7R温度类别+15%独立叠加，给出保守C上限126.5nF。
    rtop_max = rtop_nom * (1.0 + 0.001 + 50e-6 * 35.0)
    rbottom_max = rbottom_nom * (1.0 + 0.001 + 25e-6 * 35.0)
    rseries_max = rseries_nom * (1.0 + 0.01 + 100e-6 * 35.0)
    rth_max = (rtop_max * rbottom_max / (rtop_max + rbottom_max)) + rseries_max
    cmax_temp_f = c_nom * 1.10 * 1.15
    tau_component_bound_s = rth_max * cmax_temp_f
    tau_limit_s = 1.5e-3
    rtotal_max_at_cmax = tau_limit_s / cmax_temp_f
    cmax_at_rmax = tau_limit_s / rth_max

    # 22.5 V到18 V，是旧比较下限，不是新的整车停车判据。
    t_to_18_s = c_f * (initial_v**2 - 18.0**2) / (2.0 * (baseline_w + pump_full_w))
    # 用户给定的20ms固定卸载复核点。
    v50_fixed20 = voltage_at_50ms(
        shed_deadline_s=0.020,
        initial_v=initial_v,
        capacitance_f=c_f,
        baseline_w=baseline_w,
        pump_full_w=pump_full_w,
        pump_shed_w=pump_shed_w,
        window_s=window_s,
    )
    e_allow_j = 0.5 * c_f * (initial_v**2 - foc_vmin_v**2)
    t_full_max_s = (e_allow_j - baseline_w * window_s - pump_shed_w * window_s) / (
        pump_full_w - pump_shed_w
    )
    v50_at15 = voltage_at_50ms(
        shed_deadline_s=0.015,
        initial_v=initial_v,
        capacitance_f=c_f,
        baseline_w=baseline_w,
        pump_full_w=pump_full_w,
        pump_shed_w=pump_shed_w,
        window_s=window_s,
    )

    cases = [
        candidate(
            name="22.4V,±2% full-chain, tau=2ms, existing maximum 1ms sample/decision/GPIO, 5ms tail",
            threshold_v=22.4,
            gain_min=0.98,
            gain_max=1.02,
            tau_s=2e-3,
            initial_filter_v=39.009,
            sample_s=1e-3,
            decision_s=1e-3,
            gpio_s=1e-3,
            power_tail_s=5e-3,
            normal_bus_v=initial_v,
            initial_bus_v=initial_v,
            capacitance_f=c_f,
            baseline_w=baseline_w,
            pump_full_w=pump_full_w,
            pump_shed_w=pump_shed_w,
            window_s=window_s,
        ),
        candidate(
            name="23.0V,±2% full-chain, tau=2ms, 0.5ms sample, 0.5ms decision/GPIO each, 5ms tail",
            threshold_v=23.0,
            gain_min=0.98,
            gain_max=1.02,
            tau_s=2e-3,
            initial_filter_v=39.009,
            sample_s=0.5e-3,
            decision_s=0.5e-3,
            gpio_s=0.5e-3,
            power_tail_s=5e-3,
            normal_bus_v=initial_v,
            initial_bus_v=initial_v,
            capacitance_f=c_f,
            baseline_w=baseline_w,
            pump_full_w=pump_full_w,
            pump_shed_w=pump_shed_w,
            window_s=window_s,
        ),
        candidate(
            name="22.2V,逐板校准后±1%资格预算, tau=1.5ms, 0.5ms sample/decision/GPIO, 5ms tail",
            threshold_v=22.2,
            gain_min=0.99,
            gain_max=1.01,
            tau_s=1.5e-3,
            initial_filter_v=39.009,
            sample_s=0.5e-3,
            decision_s=0.5e-3,
            gpio_s=0.5e-3,
            power_tail_s=5e-3,
            normal_bus_v=initial_v,
            initial_bus_v=initial_v,
            capacitance_f=c_f,
            baseline_w=baseline_w,
            pump_full_w=pump_full_w,
            pump_shed_w=pump_shed_w,
            window_s=window_s,
        ),
        candidate(
            name="22.2V,±2%未校准对照, tau=2ms, 0.5ms sample/decision/GPIO, 5ms tail",
            threshold_v=22.2,
            gain_min=0.98,
            gain_max=1.02,
            tau_s=2e-3,
            initial_filter_v=39.009,
            sample_s=0.5e-3,
            decision_s=0.5e-3,
            gpio_s=0.5e-3,
            power_tail_s=5e-3,
            normal_bus_v=initial_v,
            initial_bus_v=initial_v,
            capacitance_f=c_f,
            baseline_w=baseline_w,
            pump_full_w=pump_full_w,
            pump_shed_w=pump_shed_w,
            window_s=window_s,
        ),
    ]

    source_files = {
        "project": SOURCE / "eda工程文件" / "水泵驱动" / "水泵驱动.eprj3",
        "native_sheet_p1": SOURCE / "eda工程文件" / "水泵驱动" / "sch" / "Schematic1" / "P1.esch2",
        "current_delivery_enet": SOURCE / "eda工程文件" / "导出" / "当前交付" / "水泵驱动.enet",
        "bundled_readonly_enet_snapshot": SOURCE / "检查" / "当前源" / "跨板并料_当前.enet",
        "source_design_details": SOURCE / "设计细节.md",
        "source_purchase_bom": SOURCE / "原理图准备" / "BOM.csv",
        "source_connection_contract": SOURCE / "原理图准备" / "原理图连接契约.json",
    }
    worktree_files = {
        "worktree_native_sheet_p1": WORKTREE / "eda工程文件" / "水泵驱动" / "sch" / "Schematic1" / "P1.esch2",
        "worktree_current_delivery_enet": WORKTREE / "eda工程文件" / "导出" / "当前交付" / "水泵驱动.enet",
    }

    payload = {
        "title": "AGTC 水泵快速卸载零增件预算",
        "date": "2026-11-05",
        "status": "CONDITIONAL_PAPER_BUDGET_NOT_BENCH_OR_VEHICLE_PASS",
        "scope": {
            "worktree": str(WORKTREE),
            "only_outputs": ["水泵快速卸载预算.py", "水泵快速卸载预算.json", "水泵驱动模块/设计细节.md append-only section"],
            "eda_operated": False,
            "pcb_or_firmware_modified": False,
            "new_hardware_added": False,
            "upstream_switch": "Not present in the native pump schematic; TPS26631 is an unadopted candidate and not selected here.",
            "pump_controller_brownout_state": "nSLEEP defaults low and the H-bridge becomes high impedance; do not claim H/H braking after loss of the pump MCU/local supply.",
        },
        "native_contract": {
            "J_BUS": "Direct bus connection in current native pump design; no board-level hard total-power switch.",
            "J_MCU_A_17": "PA1 / ADC_BUS (the authoritative pin mapping for this appended contract).",
            "pump_control": {
                "DRV_IN2_PWM": "PA8 / TIM1_CH1 / J_MCU_A.10",
                "DRV_IN1": "PB5 / J_MCU_A.20",
                "DRV_nSLEEP": "PB7 / J_MCU_A.22, 10k pulldown; reset/brownout defaults low",
                "active_brake": "If MCU and local supply remain valid: keep nSLEEP=1 and command IN1=IN2=1 (H/H).",
                "lost_supply": "nSLEEP=0; high impedance coast. Signed winding current may flow through body diodes back to BUS_M.",
            },
            "adc_bus_network": {
                "divider": "110k + 110k series top, 10k bottom, nominal ratio 1:23",
                "series": "1k ±1%",
                "capacitor": "100nF ±10% X7R",
                "tau_nominal_ms": tau_nom * 1e3,
                "tau_component_boundary_ms": tau_component_bound_s * 1e3,
                "qualification_limit_ms": tau_limit_s * 1e3,
                "qualification_boundary": {
                    "max_total_thevenin_resistance_at_Cmax_126p5nF_kohm": rtotal_max_at_cmax / 1000.0,
                    "max_capacitance_at_Rmax_nF": cmax_at_rmax * 1e9,
                    "computed_Rtotal_max_kohm": rth_max / 1000.0,
                    "computed_Cmax_nF": cmax_temp_f * 1e9,
                    "notes": "The 1.5ms full-temperature estimate assumes top TC 50ppm/C, bottom 25ppm/C, 1k series 100ppm/C, +35C from 25C, and X7R initial +10% independently combined with class temperature +15%; it is a computed BOM boundary, not a measured guarantee. The accepted contract is to qualify the assembled RC ≤1.5ms over the actual temperature window.",
                },
            },
        },
        "calculation_inputs": {
            "starting_bus_v": initial_v,
            "common_effective_capacitance_f": c_f,
            "zero_regen_first_50ms": True,
            "pump_bus_power_before_shed_w": pump_full_w,
            "pump_bus_power_after_shed_w": pump_shed_w,
            "foc_logic_w_each": foc_logic_w_each,
            "independent_source_current_a_at_22_5v": independent_source_a,
            "other_baseline_bus_power_w": baseline_w,
            "bus_power_before_shed_total_w": pump_full_w + baseline_w,
            "bus_power_after_shed_total_w": pump_shed_w + baseline_w,
            "foc_5deg_dq_required_v_at_50ms": foc_vmin_v,
            "filter_initial_state_max_v_bus_equivalent": 39.009,
            "tail_energy_model": "Conservative step-at-deadline: do not credit a gradual decline during the allocated power tail; pump remains at 70W until the ≤5W deadline, then steps to 5W.",
        },
        "adc_threshold_comparison": cases,
        "recommended_zero_add_candidate": {
            "trip_threshold_reported_bus_v": 22.2,
            "trip_gain_qualification": "Per-board full-chain ±1% residual budget required; not yet measured/guaranteed by native parts alone.",
            "recovery_threshold_reported_bus_v": 23.0,
            "recovery_static_actual_range_v_if_qualified": [23.0 / 1.01, 23.0 / 0.99],
            "normal_22_5_margin_v_at_earliest_trip": initial_v - 22.2 / 0.99,
            "normal_operation_cost": "The calibrated ±1% 22.2V threshold leaves only about 75.8mV to steady 22.5V BUS; ripple, harness/near-bridge drop and residual noise must fit within this margin or nuisance shedding may occur.",
            "recovery_policy": "Require BUS ≥23.0V continuously for 100ms, valid fresh CAN/heartbeat, healthy ADC, then a new-session explicit ARM. No auto-resume on CAN return, reset, or restored 12V/CORE.",
            "can_fast_precursor_and_loss_lease": {
                "reuse_existing_command_or_heartbeat": True,
                "cadence_target_ms": 2.0,
                "lease_expiry_target_ms": 8.0,
                "post_expiry_decision_ms": 0.5,
                "gpio_ms": 0.5,
                "power_tail_ms": 5.0,
                "worst_loss_to_le_5w_ms": 14.0,
                "message_identifier_or_new_DBC_allocated": False,
                "local_latch_overrides_all_future_CAN_frames": True,
                "main_disconnect_with_CAN_alive": "Local ADC remains independent; a continuously repeated stale/incorrect permission cannot clear or bypass a local latched trip.",
                "interface_status": "Candidate timing contract only. Current pump design details state CAN IDs, period and DBC are not frozen; use an existing assigned control/heartbeat frame only if it can support freshness. Do not count an unassigned CAN frame as implemented.",
            },
        },
        "system_energy_budget": {
            "fixed_20ms_shed_v_at_50ms_v": v50_fixed20,
            "no_shed_time_to_18v_ms": t_to_18_s * 1e3,
            "maximum_full_70w_duration_ms_to_keep_50ms_v_at_or_above_dq_limit": t_full_max_s * 1e3,
            "selected_15ms_power_deadline_v_at_50ms": v50_at15,
            "selected_15ms_margin_over_dq_limit_v": v50_at15 - foc_vmin_v,
            "18ms_deadline_v_at_50ms": voltage_at_50ms(
                shed_deadline_s=0.018,
                initial_v=initial_v,
                capacitance_f=c_f,
                baseline_w=baseline_w,
                pump_full_w=pump_full_w,
                pump_shed_w=pump_shed_w,
                window_s=window_s,
            ),
            "margin_to_18ms_deadline_v": voltage_at_50ms(
                shed_deadline_s=0.018,
                initial_v=initial_v,
                capacitance_f=c_f,
                baseline_w=baseline_w,
                pump_full_w=pump_full_w,
                pump_shed_w=pump_shed_w,
                window_s=window_s,
            ) - foc_vmin_v,
            "interpretation": "The 21.10762V value is a local dq comparison requested by the parent, not a vehicle hard limit. It sets 18.6627ms maximum full-70W duration under this constant-power, zero-regen 50ms budget. A 15ms ≤5W deadline leaves more margin than the old 20ms step.",
        },
        "calibration_method_candidate": {
            "hardware_added": False,
            "existing_signals": ["ADC_BUS at J_MCU_A.17 / PA1", "MCU internal VREFINT and its factory calibration word"],
            "procedure": "Use VREFINT_CAL to correct VDDA; at each board, calibrate the BUS divider ratio against a traceable external voltage reference at a known BUS point; verify at a second point and across 0–60C. Store the software gain/offset; accept only if the complete reported BUS error stays within ±1% and residual noise/near-bridge offset stays within the separate 75.8mV normal-operation margin.",
            "status": "Qualification target, no calibration or firmware implementation performed.",
        },
        "paper_calculation_boundaries": [
            "Constant power is integrated as V(t)^2 = V0^2 - 2*P*t/C.",
            "The filter starts at 39.009V in the worst stale-high case, not at 22.5V.",
            "Threshold-delay cases assume the pump remains at 70W until local ADC threshold crossing.",
            "The 5ms tail is conservatively modeled as 70W until its final deadline and then 5W; no linear-ramp energy credit is taken.",
            "The 0.05286F effective common capacitance and zero-recovery energy over 50ms are supplied scenario inputs, not re-measured here.",
            "No signed pump return current, H/H energy, winding tail, motor inductance, local BUS peak, TVS/SOA, or near-bridge wiring drop is validated by this script.",
            "The former 18V FOC comparison is only a screening reference; the 21.10762V dq comparison is supplied by the parent and is not a vehicle PASS boundary.",
        ],
        "minimum_bench_acceptance": [
            "Per-board ADC calibration/verification: VREFINT correction, BUS ratio ±1%, threshold margin, PA1 RC full-temperature tau ≤1.5ms, and 0.5ms sampling/decision/GPIO timing; test both 22.5V settled input and stale filter initialized to 39.009V.",
            "Double-refusal plus BMS main-path open with CAN alive, including repeated stale/incorrect permission frames; separately remove old 12V/CORE/CAN. Capture J_BUS voltage and signed current, ADC_BUS/filter voltage, message freshness, MCU state, IN1/IN2/nSLEEP, and measured average J_BUS power on a common timebase.",
            "Real pump H/H and high-impedance cases over initial current/speed: signed input current/energy, current decay/crossing, local BUS_M peak and return-to-BUS energy; confirm actual J_BUS power reaches ≤5W within the assigned 5ms tail without claiming all H/H braking is non-returning.",
            "Verify local pump 5V/3V3/MCU hold-up and reset boundary. If the MCU/local rail fails, record nSLEEP=0/high-Z behavior and diode return; do not claim active H/H or ≤5W from software after brownout.",
            "Re-arm: keep latched through recovery below 23.0V, CAN return, and reset; accept only after ≥23.0V for 100ms plus healthy fresh CAN and a new explicit ARM session.",
        ],
        "manufacturer_status_as_of_2026_11_05": {
            "real_pump_manufacturer_and_mpn": "Unknown; no exact pump source/nameplate/datasheet was found in the source design. Cannot verify pump manufacturer lifecycle or ratings.",
            "DRV8262DDWR": {"status": "TI product page lists ACTIVE", "url": "https://www.ti.com/product/DRV8262/part-details/DRV8262DDWR"},
            "TCAN1042HGVDRQ1": {"status": "TI orderable part table lists ACTIVE/Production", "url": "https://www.ti.com/lit/gpn/TCAN1042H-Q1"},
            "TPS26631PWPT": {"status": "TI orderable part table lists ACTIVE; status only and not adopted in native pump topology", "url": "https://www.ti.com/product/TPS2663/part-details/TPS26631PWPT"},
            "divider_top_tempco": {"status": "UNI-ROYAL TC03 ordering code supports 50ppm/C; exact source BOM MPN is TC0350B1103T5F", "url": "https://www.uni-royal.cn/images/userfile/file/UniroyalCatalog2023-2024_Full-C%26E%281%29%20-%E5%85%A8%E7%9B%AE%E5%BD%952023.11.6.pdf"},
            "VREFINT": {"status": "ST documents per-part VREFINT factory calibration data in system memory", "url": "https://www.st.com/resource/en/datasheet/stm32g431kb.pdf"},
        },
        "source_sha256": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in source_files.items()
        },
        "worktree_sha256_before_append": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in worktree_files.items()
        },
        "source_identity_note": "原始只读源的P1.esch2、当前交付ENET与检查/当前源里的只读ENET快照均单独列哈希。已有导出/快照不一致，因此本纸面预算不宣称本轮EDA/ENET一致性复验；来源的最新P1页hash是计算绑定身份。",
    }
    payload["calculation_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}
    return payload


if __name__ == "__main__":
    result = build()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
