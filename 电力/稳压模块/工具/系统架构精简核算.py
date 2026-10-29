"""Architecture screening only; not switching, protection, thermal or hardware qualification."""
from pathlib import Path
import hashlib
import json
import math

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parent
OUT = MODULE / "检查" / "系统架构筛选.json"


def discharge(ns, np, power, efficiency=0.95, esr_factor=1.0, dt=0.0005):
    # Independent capacitance and ESR EOL limits combined as a screening corner.
    capacitance = 280.0 * np / ns
    resistance = 0.0056 * ns / np * esr_factor
    voltage = ns * 2.45
    low = ns * 2.2
    initial = voltage
    elapsed = 0.0
    loss = 0.0
    peak = 0.0
    cause = "comparison_voltage_window"
    # P_bus/eta = I*(Voc-I*R). Use the low-current stable root.
    while voltage > low and elapsed < 10.0:
        discriminant = voltage * voltage - 4.0 * resistance * power / efficiency
        if discriminant <= 0:
            cause = "ESR_power_limit"
            break
        current = 2.0 * power / efficiency / (voltage + math.sqrt(discriminant))
        peak = max(peak, current)
        step = min(dt, (voltage - low) * capacitance / current)
        if step <= 1e-12:
            break
        loss += current * current * resistance * step
        voltage -= current * step / capacitance
        elapsed += step
    energy_drop = 0.5 * capacitance * (initial * initial - voltage * voltage)
    accounted = power * elapsed / efficiency + loss
    return {
        "series": ns, "parallel": np, "capacitors": ns * np,
        "voltage_window_ocv_V": [low, initial],
        "C_EOL_F": capacitance, "R_EOL_ohm": resistance,
        "bus_power_W": power, "non_capacitor_efficiency_assumption": efficiency,
        "esr_stress_factor_assumption": esr_factor,
        "modeled_duration_s": elapsed, "stop_reason": cause,
        "peak_group_current_A": peak, "peak_cell_current_A": peak / np,
        "cap_esr_loss_J": loss,
        "bus_energy_J": power * elapsed,
        "energy_accounting_relative_error": abs(energy_drop-accounted)/max(energy_drop, 1),
        "half_bridge_voltage_ordering_screen": initial < 22.5,
        "scope": "constant bus power, fixed voltage comparison window; no qualification",
    }


def main():
    cases = []
    for ns, np in [(8,1), (4,2), (6,2), (8,2), (9,1), (10,1), (12,1), (16,1)]:
        for power in [1350.0, 1675.0]:
            for eta, stress in [(0.95,1.0), (0.90,1.5)]:
                fine = discharge(ns, np, power, eta, stress)
                coarse = discharge(ns, np, power, eta, stress, 0.001)
                fine["time_step_convergence_delta_s"] = abs(
                    fine["modeled_duration_s"]-coarse["modeled_duration_s"])
                assert fine["energy_accounting_relative_error"] < 0.001
                assert fine["time_step_convergence_delta_s"] < 0.003
                cases.append(fine)
    battery_upgrade = []
    for parallel, current in [(4,100.0), (5,120.0), (5,125.0), (5,130.0), (6,135.0)]:
        gap = 3700.0 - 22.5 * current
        nominal = discharge(8,1,gap)
        stressed = discharge(8,1,gap,0.90,1.5)
        for result in [nominal, stressed]:
            assert result["energy_accounting_relative_error"] < 0.001
        battery_upgrade.append({
            "battery_series":9, "battery_parallel":parallel,
            "cells":9*parallel, "added_cells_vs_4P":9*(parallel-4),
            "battery_current_target_A":current,
            "ideal_average_cell_current_A":current/parallel,
            "nominal_capacity_Ah":3.0*parallel,
            "nominal_energy_Wh":9*3.6*3*parallel,
            "added_cell_mass_upper_kg":9*(parallel-4)*0.048,
            "bus_V_under_load_assumption":22.5,
            "motor_plus_aux_power_assumption_W":3700,
            "cap_bus_power_gap_W":gap,
            "cap_nominal_screen":nominal, "cap_stress_screen":stressed,
            "existing_shunt_power_W":current**2*0.0005,
            "existing_MOS_25C_conduction_screen_W":current**2*(2*0.0028/3),
            "battery_net_charge_limit_kept_A":16,
            "scope":"ideal current sharing; requires upgraded and qualified BMS/power path",
        })
    source_paths = [MODULE / "原理图准备/BOM.csv", ROOT / "BMS模块/设计需求.md",
                    ROOT / "FOC驱动模块/设计需求.md",
                    ROOT / "BMS模块/设计细节.md",
                    ROOT / "参考资料/电池/11-_EVE_INR18650-30P_SPEC._PBRI-INR1865030P-D06-01-B-2023.8.4.pdf"]
    data = {
        "date": "2026-10-29",
        "status": "ARCHITECTURE_SCREEN_ONLY_NOT_HARDWARE_QUALIFIED",
        "user_requirement": "two motors simultaneously at 1.8kW input peak each",
        "selected_configuration": "9S5P battery and existing 8S1P capacitor bank",
        "primary_supercap_function": "electrical braking energy acceptance; no mechanical brake",
        "empty_supercap_startup": "limited drive power and slow acceleration allowed",
        "zero_volt_charge_time_requirement": "tens of seconds; exact voltage and deadline not fixed",
        "discharge_model_initial_2p45V_is_not_required_standby_or_start_condition": True,
        "standalone_regulator_required": True,
        "share_FOC_controller_allowed": False,
        "peak_duration_requirement_s": None,
        "one_second": "comparison scenario from existing pulse sizing, not a new user requirement",
        "baseline_bus_range_V": [22.5,37.8],
        "baseline_battery_current_design_inputs_A": {"continuous":60,"pulse_1s":100},
        "battery_power_screens": [
            {"bus_V":v, "motor_input_W":3600, "bus_current_A":3600/v,
             "cap_gap_at_100A_W":max(0,3600-v*100),
             "cap_gap_at_90A_plus_100W_aux_W":max(0,3700-v*90)}
            for v in [22.5,24.0,27.0,32.4,36.0,37.8]],
        "cell": {"MPN":"SC0350-270-RSS", "C_nominal_F":350,
                 "C_EOL_screen_F":280,"ESR_max_initial_ohm":0.0028,
                 "ESR_EOL_screen_ohm":0.0056,
                 "initial_voltage_assumption_V":2.45,"final_voltage_assumption_V":2.2},
        "direct_parallel": {
            "series_at_2p45V_per_cell": math.ceil(37.8/2.45),
            "precharge_energy_new_J":0.5*(350/16)*37.8**2,
            "cannot_guarantee_battery_16A_split":True},
        "source_sha256": {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "cases":cases,
        "battery_upgrade_with_existing_8_caps":battery_upgrade,
        "unmodeled": ["cell imbalance", "temperature-dependent ESR", "pulse current duration rating",
                      "MOSFET and magnetics thermal limits", "interconnect resistance",
                      "switching and minimum pulse behavior", "0V charging", "fault isolation",
                      "battery pulse qualification", "actual startup and braking duration"],
    }
    OUT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for row in cases:
        if row["bus_power_W"] == 1675 and row["esr_stress_factor_assumption"] == 1.0:
            print(f'{row["series"]}S{row["parallel"]}P: {row["modeled_duration_s"]:.3f}s, '
                  f'{row["peak_group_current_A"]:.1f}A group, '
                  f'{row["peak_cell_current_A"]:.1f}A/cell, {row["stop_reason"]}')
    print("Energy accounting and numerical convergence checks passed; hardware qualification remains open.")
    for row in battery_upgrade:
        print(f'9S{row["battery_parallel"]}P at {row["battery_current_target_A"]:g}A: '
              f'cap gap {row["cap_bus_power_gap_W"]:.1f}W, '
              f'base {row["cap_nominal_screen"]["modeled_duration_s"]:.3f}s, '
              f'stress {row["cap_stress_screen"]["modeled_duration_s"]:.3f}s')


if __name__ == "__main__":
    main()
