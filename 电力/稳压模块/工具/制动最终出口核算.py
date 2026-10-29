"""单只1Ω制动电阻作为唯一耗能出口的确定性纸面核算。"""
from __future__ import annotations

import json
import math
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1]
OUT = MODULE / "检查" / "制动最终出口核算.json"

# 用户给定的纸面参照条件；减速度是比较输入，不是绝对停车指标。
MASS_KG = 30.0
SPEED_KMH = 15.0
REFERENCE_DECEL_M_S2 = 3.9624
GRAVITY_M_S2 = 9.80665
BUS_POINTS_V = [22.5, 37.8, 42.0]

# 原厂HEWR500W标准单只路线：1Ω ±5%。
R_NOMINAL_OHM = 1.0
R_TOLERANCE = 0.05
R_TCR_PPM_PER_C = 400.0
TCR_REFERENCE_C = 25.0
TCR_SCREEN_C = 125.0
AMBIENT_MAX_C = 60.0
DERATE_ZERO_AMBIENT_C = 275.0
NAMEPLATE_POWER_25C_W = 500.0

# 60°C连续降额约430W；正常电阻平均功率限400W，留30W热功率余量。
NORMAL_RESISTOR_POWER_LIMIT_W = 400.0
SENSITIVITY_FACTORS = [1.0, 1.5, 2.0]


def r_range_at_temp(temp_c: float) -> tuple[float, float]:
    """以标称值、公差及±TCR作线性阻值包络。"""
    delta = abs(temp_c - TCR_REFERENCE_C) * R_TCR_PPM_PER_C * 1e-6
    return (
        R_NOMINAL_OHM * (1.0 - R_TOLERANCE) * (1.0 - delta),
        R_NOMINAL_OHM * (1.0 + R_TOLERANCE) * (1.0 + delta),
    )


def rnd(x: float, digits: int = 6) -> float:
    return round(x, digits)


def capped_power_stop(mass_kg: float, inertia_factor: float) -> dict[str, float]:
    """平地上按P=min(Pcap, m_eff*a_ref*v)积分停车时间和距离。"""
    v0 = SPEED_KMH / 3.6
    effective_mass = mass_kg * inertia_factor
    a_ref = REFERENCE_DECEL_M_S2
    pcap = NORMAL_RESISTOR_POWER_LIMIT_W
    v_cross = pcap / (effective_mass * a_ref)

    if v0 <= v_cross:
        # 从起始速度起即未触及400W限幅。
        stop_t = v0 / a_ref
        stop_x = v0**2 / (2.0 * a_ref)
        cap_phase_t = 0.0
        cap_phase_x = 0.0
        reference_decel_phase_t = stop_t
        reference_decel_phase_x = stop_x
    else:
        # P=400W时，a=P/(m_eff*v)，由v0降到交点速度。
        cap_phase_t = effective_mass * (v0**2 - v_cross**2) / (2.0 * pcap)
        cap_phase_x = effective_mass * (v0**3 - v_cross**3) / (3.0 * pcap)
        reference_decel_phase_t = v_cross / a_ref
        reference_decel_phase_x = v_cross**2 / (2.0 * a_ref)
        stop_t = cap_phase_t + reference_decel_phase_t
        stop_x = cap_phase_x + reference_decel_phase_x

    energy_j = 0.5 * effective_mass * v0**2
    average_p = energy_j / stop_t
    return {
        "effective_inertial_mass_kg": effective_mass,
        "power_crossover_speed_m_s": v_cross,
        "power_crossover_speed_kmh": v_cross * 3.6,
        "constant_400W_phase_time_s": cap_phase_t,
        "constant_400W_phase_distance_m": cap_phase_x,
        "reference_deceleration_phase_time_s": reference_decel_phase_t,
        "reference_deceleration_phase_distance_m": reference_decel_phase_x,
        "stop_time_s": stop_t,
        "stop_distance_m": stop_x,
        "energy_to_resistor_J": energy_j,
        "mean_resistor_power_over_stop_W": average_p,
        "start_deceleration_m_s2_on_level_ground": pcap / (effective_mass * v0),
        "peak_resistor_average_power_W": min(pcap, effective_mass * a_ref * v0),
        "power_profile_integral_check_J": pcap * cap_phase_t
        + 0.5 * pcap * reference_decel_phase_t,
    }


speed_m_s = SPEED_KMH / 3.6
reference_force_n = MASS_KG * REFERENCE_DECEL_M_S2
reference_energy_j = 0.5 * MASS_KG * speed_m_s**2
reference_stop_time_s = speed_m_s / REFERENCE_DECEL_M_S2
reference_stop_distance_m = speed_m_s**2 / (2.0 * REFERENCE_DECEL_M_S2)
reference_peak_power_w = reference_force_n * speed_m_s
reference_mean_power_w = reference_energy_j / reference_stop_time_s

derate_at_60c = 1.0 - (AMBIENT_MAX_C - 25.0) / (DERATE_ZERO_AMBIENT_C - 25.0)
continuous_rating_60c_w = NAMEPLATE_POWER_25C_W * derate_at_60c
normal_power_margin_to_60c_rating_w = continuous_rating_60c_w - NORMAL_RESISTOR_POWER_LIMIT_W
r25_min, r25_max = r_range_at_temp(25.0)
r60_proxy_min, r60_proxy_max = r_range_at_temp(60.0)
r125_min, r125_max = r_range_at_temp(TCR_SCREEN_C)

profile_at_400w = capped_power_stop(MASS_KG, 1.0)

bus_screen = []
for bus_v in BUS_POINTS_V:
    i_on_min = bus_v / r125_max
    i_on_max = bus_v / r125_min
    p_on_min = bus_v**2 / r125_max
    p_on_max = bus_v**2 / r125_min
    current_for_400w = NORMAL_RESISTOR_POWER_LIMIT_W / bus_v
    bus_screen.append(
        {
            "bus_V": bus_v,
            "meaning": {
                22.5: "9S低端正常功率能力核算",
                37.8: "9S理论上限参考",
                42.0: "PWM导通瞬时电压/电流/功率筛查，不是连续全开工况",
            }[bus_v],
            "normal_sink_power_command_peak_W": NORMAL_RESISTOR_POWER_LIMIT_W,
            "equivalent_average_sink_current_at_400W_A": rnd(current_for_400w),
            "resistor_full_on_current_A_at_125C_TCR_screen": {
                "min": rnd(i_on_min),
                "max": rnd(i_on_max),
            },
            "resistor_full_on_power_W_at_125C_TCR_screen": {
                "min": rnd(p_on_min),
                "max": rnd(p_on_max),
            },
            "ideal_PWM_duty_range_for_400W_average": {
                "min": rnd(NORMAL_RESISTOR_POWER_LIMIT_W / p_on_max),
                "max": rnd(NORMAL_RESISTOR_POWER_LIMIT_W / p_on_min),
            },
            "power_at_Rmax_margin_over_400W_W": rnd(p_on_min - NORMAL_RESISTOR_POWER_LIMIT_W),
            "note": (
                "BUS决定PWM导通瞬时应力；母线闭环和制动功率曲线决定电阻平均热功率。"
                "42V没有被当成连续满导通负载。"
            ),
        }
    )

low_bus = BUS_POINTS_V[0]
low_bus_power_at_rmax_w = low_bus**2 / r125_max
low_bus_min_voltage_for_400w_v = math.sqrt(NORMAL_RESISTOR_POWER_LIMIT_W * r125_max)
largest_r_for_400w_ohm = low_bus**2 / NORMAL_RESISTOR_POWER_LIMIT_W
bus42_power_min = BUS_POINTS_V[-1] ** 2 / r125_max
bus42_power_max = BUS_POINTS_V[-1] ** 2 / r125_min
bus42_current_max = BUS_POINTS_V[-1] / r125_min
rcwv_v = math.sqrt(NAMEPLATE_POWER_25C_W * R_NOMINAL_OHM)
overload_voltage_formula_v = 2.5 * rcwv_v
overload_power_test_w = 6.25 * NAMEPLATE_POWER_25C_W

# 重复停车的热平均边界；同一停止波形功率在任何时刻都被限至400W。
repeat_cases = []
for factor in SENSITIVITY_FACTORS:
    stop = capped_power_stop(MASS_KG, factor)
    interval_for_400w_avg_s = stop["energy_to_resistor_J"] / NORMAL_RESISTOR_POWER_LIMIT_W
    interval_for_430w_avg_s = stop["energy_to_resistor_J"] / continuous_rating_60c_w
    repeat_cases.append(
        {
            "inertia_factor": factor,
            "status": "translation-only paper baseline" if factor == 1.0 else "sensitivity only, not a user requirement",
            **{k: rnd(v) for k, v in stop.items()},
            "start_to_start_period_lower_bound_from_400W_mean_only_s": rnd(
                max(stop["stop_time_s"], interval_for_400w_avg_s)
            ),
            "additional_dwell_after_stop_from_400W_mean_only_s": rnd(
                max(0.0, interval_for_400w_avg_s - stop["stop_time_s"])
            ),
            "energy_over_430W_continuous_rating_s": rnd(interval_for_430w_avg_s),
            "400W_peak_below_430W_60C_continuous_rating": stop["peak_resistor_average_power_W"]
            <= continuous_rating_60c_w,
            "repeat_boundary": (
                "Average energy divided by power is only a thermal mean condition. With the assumed control law, "
                "the requested resistor average power never exceeds 400W; repeated pulse qualification still depends "
                "on manufacturer overload limits and actual thermal installation."
            ),
        }
    )

# 斜坡参数表：不挑一个坡度作为额外需求。
slope_sweep = []
for theta_deg in [0.0, 5.0, 10.0, 15.0, 18.0, 19.0, 20.0, 25.0, 30.0, 45.0]:
    sin_theta = math.sin(math.radians(theta_deg))
    gravity_power = MASS_KG * GRAVITY_M_S2 * speed_m_s * sin_theta
    reference_brake_peak = MASS_KG * (REFERENCE_DECEL_M_S2 + GRAVITY_M_S2 * sin_theta) * speed_m_s
    start_decel_with_400w = NORMAL_RESISTOR_POWER_LIMIT_W / (MASS_KG * speed_m_s) - GRAVITY_M_S2 * sin_theta
    reference_stop_energy = reference_energy_j + MASS_KG * GRAVITY_M_S2 * reference_stop_distance_m * sin_theta
    slope_sweep.append(
        {
            "downhill_angle_deg_parameter_only": theta_deg,
            "constant_speed_gravity_power_W": rnd(gravity_power),
            "power_peak_to_hold_reference_deceleration_W": rnd(reference_brake_peak),
            "energy_for_reference_deceleration_stop_J": rnd(reference_stop_energy),
            "initial_net_deceleration_if_resistor_is_limited_to_400W_m_s2": rnd(start_decel_with_400w),
            "400W_sink_can_balance_gravity_at_15kmh": gravity_power <= NORMAL_RESISTOR_POWER_LIMIT_W,
            "note": "parameter capability curve only; does not assign a user-required grade",
        }
    )

constant_speed_slope_limit_deg = math.degrees(
    math.asin(
        min(
            1.0,
            NORMAL_RESISTOR_POWER_LIMIT_W
            / (MASS_KG * GRAVITY_M_S2 * speed_m_s),
        )
    )
)

result = {
    "status": "DETERMINISTIC_PAPER_SCREEN_NOT_PURCHASE_OR_VEHICLE_VALIDATION",
    "scope": {
        "sole_energy_outlet": "one central resistor",
        "battery_absorption_credit_W": 0.0,
        "supercapacitor_absorption_credit_W": 0.0,
        "FOC_or_other_resistor_takeover_credit": False,
        "other_absorber_or_energy_sink_credit": False,
        "thermal_fuse_or_BMS_interlock_added": False,
        "existing_FOC_devices_modified_or_removed": False,
        "rotating_inertia_requirement_assumed": False,
        "stop_target_status": "3.9624m/s^2 is a paper comparison reference, not a user hard requirement",
        "energy_policy": "all translational kinetic energy and any modeled downhill gravitational energy is assigned to resistor heat",
    },
    "input_and_part": {
        "mass_kg": MASS_KG,
        "speed_kmh": SPEED_KMH,
        "reference_deceleration_m_s2": REFERENCE_DECEL_M_S2,
        "BUS_points_V": BUS_POINTS_V,
        "temperature_environment_C": [0.0, AMBIENT_MAX_C],
        "candidate_family": "Uni-Royal HEWR500W",
        "single_standard_order_code_from_published_coding_rule": "HEWRA0J0100500",
        "code_decode": "HEWR series / A0 terminal / J ±5% / 0100 = 1Ω / 500W",
        "standard_resistance_range_ohm": [0.5, 12000.0],
        "published_package_dimensions_mm": {"length": 335, "width": 60, "height": 30},
        "actual_stock_quote_lead_time": "not publicly verified; confirm with manufacturer or distributor before purchase",
        "manufacturer_sources": [
            {
                "title": "Uni-Royal DIP-SP-049 V.6 official HEWR/HBWR datasheet, 2024-05-06",
                "url": "https://www.uni-royal.cn/en/images/userfile/file/1717729679cac485d4bbc6a42a.pdf",
                "pages": [2, 3, 5, 6],
                "claims": [
                    "coding: HEWR+A0+J+0100+500 gives 1Ω ±5%, 500W terminal type",
                    "HEWR500W standard resistance range 0.5Ω–12kΩ; package 335×60×30mm",
                    "continuous power at ambient through 25°C; linear derating reaches zero at 275°C",
                    "for R below 20Ω, temperature coefficient ±400ppm/°C",
                    "HEWR short-time overload test is 6.25× rated power or applicable maximum overload voltage, whichever is lower, for 5s",
                    "RCWV=sqrt(P*R); overload voltage is limited by the lower of 2.5×RCWV and applicable maximum overload voltage",
                ],
            },
            {
                "title": "Uni-Royal official HEWR product page",
                "url": "https://www.uni-royal.cn/en/product.php?id=76",
                "claims": ["HEWR series published power options, resistance ranges and TCR ranges"],
            },
        ],
    },
    "resistor_resistance_and_thermal": {
        "nominal_resistance_ohm": R_NOMINAL_OHM,
        "tolerance_fraction": R_TOLERANCE,
        "TCR_ppm_per_C": R_TCR_PPM_PER_C,
        "TCR_reference_C": TCR_REFERENCE_C,
        "25C_resistance_range_ohm": {"min": rnd(r25_min), "max": rnd(r25_max)},
        "60C_ambient_proxy_range_ohm_not_hot_element_bound": {
            "min": rnd(r60_proxy_min),
            "max": rnd(r60_proxy_max),
        },
        "125C_TCR_screen_range_ohm": {"min": rnd(r125_min), "max": rnd(r125_max)},
        "125C_screen_note": (
            "Uses the datasheet ±400ppm/°C across its room-temperature/125°C test span. "
            "The resistor element temperature under braking is not predicted."
        ),
        "ambient_derating_fraction_at_60C": rnd(derate_at_60c),
        "nameplate_continuous_power_25C_W": NAMEPLATE_POWER_25C_W,
        "manufacturer_continuous_power_at_60C_W": rnd(continuous_rating_60c_w),
        "normal_average_resistor_power_limit_W": NORMAL_RESISTOR_POWER_LIMIT_W,
        "thermal_margin_at_60C_W": rnd(normal_power_margin_to_60c_rating_w),
        "thermal_margin_fraction_of_60C_rating": rnd(normal_power_margin_to_60c_rating_w / continuous_rating_60c_w),
        "thermal_management_policy": (
            "Keep commanded average resistor power at or below 400W under the specified 0–60°C ambient and mounting conditions. "
            "Do not interrupt the sole braking resistor because a temperature sensor trips during braking; no fuse or BMS interlock is included."
        ),
        "mounting_airflow_resistor_thermal_impedance_verified": False,
    },
    "reference_braking_for_comparison_only": {
        "mass_kg": MASS_KG,
        "speed_kmh": SPEED_KMH,
        "speed_m_s": rnd(speed_m_s),
        "deceleration_reference_m_s2": REFERENCE_DECEL_M_S2,
        "brake_force_N": rnd(reference_force_n),
        "uncapped_reference_stop_time_s": rnd(reference_stop_time_s),
        "uncapped_reference_stop_distance_m": rnd(reference_stop_distance_m),
        "translation_energy_to_resistor_J_battery_and_supercap_absorption_zero": rnd(reference_energy_j),
        "uncapped_reference_power_profile_W": {
            "equation": "P_ref(t)=m*a_ref*(v0-a_ref*t)",
            "start_peak": rnd(reference_peak_power_w),
            "end": 0.0,
            "mean": rnd(reference_mean_power_w),
        },
        "normal_400W_limited_control_law": "P(t)=min(400W, m*a_ref*v(t)); all brake energy is dissipated in the resistor",
        "crossover_speed_for_400W_limit_m_s": rnd(
            NORMAL_RESISTOR_POWER_LIMIT_W / (MASS_KG * REFERENCE_DECEL_M_S2)
        ),
        "capped_stop_result": {k: rnd(v) for k, v in profile_at_400w.items()},
        "interpretation": (
            "400W cap reduces initial level-ground deceleration from 3.9624m/s² to the power-limited value; "
            "once speed falls to the crossover, the reference deceleration is restored. The 260.4J kinetic energy still all goes to the resistor."
        ),
    },
    "low_BUS_22_5V_margin": {
        "worst_Rmax_125C_TCR_screen_ohm": rnd(r125_max),
        "full_on_resistor_power_at_Rmax_W": rnd(low_bus_power_at_rmax_w),
        "normal_requested_peak_average_power_W": NORMAL_RESISTOR_POWER_LIMIT_W,
        "power_headroom_W": rnd(low_bus_power_at_rmax_w - NORMAL_RESISTOR_POWER_LIMIT_W),
        "power_headroom_fraction_of_400W": rnd(
            (low_bus_power_at_rmax_w - NORMAL_RESISTOR_POWER_LIMIT_W) / NORMAL_RESISTOR_POWER_LIMIT_W
        ),
        "largest_R_for_400W_at_22_5V_ohm": rnd(largest_r_for_400w_ohm),
        "minimum_BUS_voltage_for_400W_at_Rmax_V": rnd(low_bus_min_voltage_for_400w_v),
        "minimum_full_on_resistor_current_at_Rmax_A": rnd(low_bus / r125_max),
        "peak_equivalent_current_for_400W_A": rnd(NORMAL_RESISTOR_POWER_LIMIT_W / low_bus),
        "screen": "PASS for ideal static resistor power at 22.5V, including ±5% and ±400ppm/°C to 125°C screen",
        "qualification_limit": "does not cover wiring drop, switch loss, BUS dynamic excursion, or actual element temperature",
    },
    "BUS_PWM_instantaneous_stress": {
        "rows": bus_screen,
        "42V_screen": {
            "resistor_terminal_voltage_V": 42.0,
            "maximum_full_on_current_A": rnd(bus42_current_max),
            "full_on_power_range_W": {"min": rnd(bus42_power_min), "max": rnd(bus42_power_max)},
            "RCWV_for_500W_1ohm_V": rnd(rcwv_v),
            "2_5x_RCWV_V": rnd(overload_voltage_formula_v),
            "42V_below_2_5x_RCWV_formula": 42.0 < overload_voltage_formula_v,
            "6_25x_nameplate_overload_test_power_W": rnd(overload_power_test_w),
            "peak_power_below_6_25x_test_power_number": bus42_power_max <= overload_power_test_w,
            "qualification_limit": (
                "The series datasheet further limits overload voltage by the lesser applicable maximum overload voltage; "
                "that exact maximum is not stated in the retrieved sheet. The 6.25×/5s test screens peak voltage/power only; "
                "it does not establish arbitrary repetitive PWM or repeated-stop qualification."
            ),
        },
    },
    "repeated_stop_heat": {
        "normal_resistor_power_limit_W": NORMAL_RESISTOR_POWER_LIMIT_W,
        "manufacturer_60C_continuous_rating_W": rnd(continuous_rating_60c_w),
        "cases": repeat_cases,
        "boundary": (
            "For the capped power law each stop averages less than 400W and instantaneous average resistor command is never above 400W. "
            "E/P gives a mean-energy interval only; PWM peak repetition and installation thermal time constants still require manufacturer limits or qualification."
        ),
    },
    "slope_parameter_capability": {
        "constant_speed_downhill_gravity_power_equation": "P=m*g*v*sin(theta)",
        "reference_deceleration_power_equation": "P_ref_start=m*(a_ref+g*sin(theta))*v0",
        "400W_power_limited_initial_net_deceleration_equation": "a_net_start=400/(m*v0)-g*sin(theta)",
        "sweep": slope_sweep,
        "constant_speed_slope_at_15kmh_for_400W_W_only_deg": rnd(constant_speed_slope_limit_deg),
        "interpretation": (
            "At slopes where m*g*v*sin(theta)>400W, the single outlet cannot hold 15km/h at steady descent under this average power ceiling. "
            "This is a parameter capability boundary, not an assumed slope demand or proven vehicle limit."
        ),
    },
    "selection_conclusion_for_document": {
        "candidate": "one Uni-Royal HEWR500W 1Ω ±5% terminal-type resistor, code HEWRA0J0100500",
        "continuous_case": (
            "At 60°C the manufacturer derating curve gives about 430W continuous. Set the normal average resistor command to 400W, "
            "leaving 30W margin. Under all battery/supercapacitor absorption set to zero, the 30kg/15km/h translation-only energy is 260.4J; "
            "the 400W-capped power-law stop is about 1.076s and 2.285m in the paper model. This is near the 3.9624m/s² comparison "
            "reference (uncapped 1.052s/2.191m) while keeping the resistor's requested average heat under its 60°C continuous rating."
        ),
        "low_bus_case": "At 22.5V, Rmax=1.092Ω in the 125°C TCR screen still permits about 464W full-on, above the 400W command; ideal low-BUS power margin is about 63.6W.",
        "high_bus_PWM_case": (
            "At 42V the worst screened full-on pulse is about 1.93kW/46.1A, below the 6.25×500W test-power number of 3.125kW and below 2.5×RCWV=55.9V. "
            "The separate maximum-overload-voltage limit and any repetitive PWM envelope still need factory confirmation; 42V is not treated as continuous full-on."
        ),
        "temperature_policy": "No thermal fuse, BMS interlock, or overtemperature cutoff may interrupt the only braking energy outlet; manage the 0–60°C installation and 400W operating envelope by resistor rating and thermal design.",
        "purchase_status": "Code/range/form factor are supported by the published factory coding and datasheet; inventory, lead time, price, exact maximum overload voltage, and mounting/thermal conditions are not confirmed. Do not freeze purchase before quote and installation review.",
        "status": "paper selection candidate, not schematic/hardware/vehicle qualification",
    },
    "not_included": [
        "Rotating inertia, payload variation, drivetrain conversion loss, wheel torque, traction, zero-speed brake torque, or parking hold.",
        "A specific downhill grade, repeated-stop requirement, or vehicle stopping acceptance limit.",
        "Any energy acceptance by battery, supercapacitor, FOC resistor, or other device.",
        "A thermal fuse, BMS interlock, or switching shutdown that would remove the sole resistor path during braking.",
        "Chopper topology, MOSFET SOA, current waveform, wire drop, BUS overshoot, controller response, or physical thermal test.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(
    f"E={reference_energy_j:.3f}J; reference={reference_stop_time_s:.6f}s/{reference_stop_distance_m:.6f}m; "
    f"400W-limited={profile_at_400w['stop_time_s']:.6f}s/{profile_at_400w['stop_distance_m']:.6f}m; "
    f"R125-screen={r125_min:.6f}..{r125_max:.6f}ohm; lowBUS={low_bus_power_at_rmax_w:.3f}W"
)
