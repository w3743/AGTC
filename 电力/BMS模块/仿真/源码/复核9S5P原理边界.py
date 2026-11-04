#!/usr/bin/env python3
"""Recheck the saved 9S4P source and run bounded 9S5P design comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import verify_charge_limit_16a as charge


BOARD = Path(__file__).resolve().parents[2]
OUT = BOARD / "仿真/结果/9S5P原理边界复核.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_source() -> dict:
    coverage = charge.load_json(BOARD.parent / "检查/统一审查/BMS/current-pin-net-coverage.json")
    project = coverage["project"]
    render_readback = charge.load_json(BOARD / "检查/当前源/本轮图面与官方复核.json")
    assert render_readback["source_bytes_unchanged"] and not render_readback["own_session_alive_after_close"]
    assert not render_readback["netlist"]["identity_or_pin_net_differences"]
    netlist_name = "检查/当前源/BMS-9S4P_P1_图面检查.enet"
    paths = {
        "BMS-9S4P.eprj3": project["eprj3Sha256"],
        "sch/Schematic1/P1.esch2": render_readback["source_sha256_after"]["sch/Schematic1/P1.esch2"],
        "原理图准备/BOM.csv": render_readback["source_sha256_after"]["原理图准备/BOM.csv"],
        netlist_name: render_readback["netlist"]["new_export_sha256"],
    }
    for name, expected in paths.items():
        if digest(BOARD / name) != expected:
            raise ValueError(f"Source drift: {name}")

    netlist = charge.load_json(BOARD / netlist_name)
    assert netlist["projectId"] == render_readback["scope"]["uuid"]
    components = {c["props"]["Designator"]: c for c in netlist["components"].values()}
    pins = {
        ref: {number: pin["net"] for number, pin in c["pinInfoMap"].items()}
        for ref, c in components.items()
    }
    removed = {"R_PSU_MINLOAD6", "R_PSU_MINLOAD7"}
    assert set(render_readback["netlist"]["expected_removed_refs"]) == removed
    assert len(pins) == render_readback["netlist"]["refs"] == 224
    assert sum(map(len, pins.values())) == render_readback["netlist"]["pins"] == 736
    assert pins == {
        ref: {number: pin["net"] for number, pin in c["pinInfoMap"].items()}
        for ref, c in coverage["components"].items()
        if ref not in removed
    }

    with (BOARD / "原理图准备/BOM.csv").open(encoding="utf-8-sig", newline="") as file:
        bom = {row["位号"]: row["MPN"] for row in csv.DictReader(file)}
    pcb_bom_refs = {
        ref for ref, component in components.items()
        if component["props"]["Add into BOM"] == "yes" and component["props"]["Convert to PCB"] == "yes"
    }
    assert len(pcb_bom_refs) == 204 and pcb_bom_refs <= bom.keys()
    assert not removed & bom.keys()
    assert not any(ref.startswith(("J_FAN", "Q_FAN", "R_FAN")) for ref in components)
    required = {
        "RSHUNT": {"1": "B0", "2": "PACK-"},
        "R_SRP": {"1": "B0", "2": "SRP"},
        "R_SRN": {"1": "PACK-", "2": "SRN"},
        "U1": {"18": "SRP", "20": "SRN", "22": "NTC_BOARD_NODE", "30": "DFETOFF", "43": "DSG", "45": "CHG"},
        "F1": {"1": "CELL_B9", "2": "BATT+_FUSED"},
        "F_CELL_B9": {"1": "CELL_B9", "2": "TAP_B9"},
        "PSU5_EXT": {"1": "B0", "2": "TAP_B9", "3": "VOUT_5V", "5": "B0", "6": "CTRL_MODULE"},
        "Q_PRE1": {"1": "QPRE_G", "2": "BATT+_FUSED", "3": "QPRE_SRC"},
        "Q_PRE2": {"1": "QPRE_G", "2": "QPRE_SW", "3": "QPRE_SRC"},
        "R_PRE": {"1": "QPRE_SW", "2": "PACK+"},
        "R_QPRE_PU": {"1": "QPRE_SRC", "2": "QPRE_G"},
        "R_TIMER_SET": {"1": "TMR_SET", "2": "B0"},
        "U_TIMER": {"1": "TMR_VDD", "2": "B0", "3": "TMR_SET", "4": "B0", "5": "TMR_ACTIVE", "6": "B0"},
        "R_DFETOFF_PU": {"1": "REG1", "2": "DFETOFF"},
        "Q_DFETOFF": {"1": "DFETOFF_G", "2": "B0", "3": "DFETOFF"},
        "U_SEQ": {"1": "MAIN_CLR_N", "3": "MAIN_REQ", "5": "FET_RELEASE", "13": "POR_OK_N"},
        "U_AND2": {"4": "MAIN_CLR_N", "5": "PRE_CTRL_3V3", "6": "PRE_CTRL_3V3"},
    }
    preload_refs = {f"R_PSU_MINLOAD{i}" for i in range(1, 6)}
    assert set(render_readback["netlist"]["expected_identity_replacements"]) == preload_refs
    for ref in preload_refs:
        assert bom[ref] == components[ref]["props"]["Manufacturer Part"] == "1206W4F2700T5E"
        required[ref] = {"1": "VOUT_5V", "2": "B0"}
    for direction, source in (("CHG", "BATT+_FUSED"), ("DSG", "PACK+")):
        for i in range(1, 4):
            ref = f"Q_{direction}{i}"
            gate = f"{direction}_G{i}"
            required[ref] = {"1": gate, **{str(p): source for p in range(2, 9)}, "9": "FET_MID"}
            assert bom[ref] == "HYG022N10NS1TA"
            assert set(pins[f"R_G_{direction}{i}"].values()) == {direction, gate}
            assert set(pins[f"R_GS_{direction}{i}"].values()) == {source, gate}
    for i in range(1, 9):
        lower = "B0" if i == 1 else f"TAP_B{i-1}"
        required[f"U_ETA_BAL{i}"] = {
            "1": f"TAP_B{i}", "2": f"TAP_B{i+1}", "3": f"BAL_ETA_SW_{i}",
            "4": lower, "6": "", "8": f"BAL_ETA_EN_{i}", "9": lower,
        }
        assert set(pins[f"L_ETA_BAL{i}"].values()) == {f"BAL_ETA_SW_{i}", f"TAP_B{i}"}
    for ref, expected in required.items():
        assert all(pins[ref][number] == net for number, net in expected.items()), ref
    prefixes = {"R_PRE": "TE Connectivity ", "F1": "Littelfuse ", "F_CELL_B9": "Eaton "}
    for ref in ("RSHUNT", "PSU5_EXT", "Q_PRE1", "Q_PRE2", "R_PRE", "U1", "F1", "F_CELL_B9"):
        assert bom[ref].removeprefix(prefixes.get(ref, "")) == components[ref]["props"]["Manufacturer Part"], ref

    return {
        "source_sha256": paths, "checked_refs": len(pins), "checked_pins": sum(map(len, pins.values())),
        "cad_board_purchase_refs_all_in_bom": len(pcb_bom_refs),
        "targeted_pin_checks": required, "current_drc": render_readback["drc"],
        "official_drc_rerun": True,
    }


def discharge() -> dict:
    # Each bank has n ideal equal shares. The only guaranteed RDS(on) used here is at 25 °C.
    cases = []
    for bank_size in (3, 4):
        for current in (60, 100, 130):
            cases.append({
                "fets_per_bank": bank_size, "pack_A": current, "per_fet_A": current / bank_size,
                "per_cell_A_ideal_5p": current / 5,
                "both_fet_banks_W_25C_10V": current**2 * 2 * 0.0028 / bank_size,
                "shunt_W_nominal": current**2 * 0.0005,
                "main_path_drop_V_no_harness": current * (0.0056 / bank_size + 0.0005),
            })

    low = 0.5 * (1 - 0.01 - 50e-6 * 50) - 0.010  # mΩ incl. path sensitivity
    high = 0.5 * (1 + 0.01 + 50e-6 * 50) + 0.010
    thresholds = {
        label: {"early_A": earliest / high, "late_A": latest / low}
        for label, (earliest, latest) in {
            "OCD1_35": (65, 75), "OCD1_40": (75, 85), "SCD_5": (85, 115), "OCC_3": (4, 8.65),
        }.items()
    }
    assert thresholds["OCD1_35"]["early_A"] < 130 < thresholds["OCD1_40"]["early_A"]
    assert thresholds["OCC_3"]["late_A"] > 16
    return {
        "cases": cases, "thresholds_with_10uohm_path_sensitivity": thresholds,
        "max_130A_sense_mV": 130 * high,
        "maximum_cell_share_for_130A_at_30A_per_cell": 30 / 130,
        "loaded_bus_V_for_130A_and_3600W_plus_100W_aux": 3700 / 130,
        "dc_dc_assist_W_at_22p5V_130A_plus_100W_aux": 3700 - 22.5 * 130,
    }


def charge_cases() -> dict:
    params = charge.load_json(BOARD / "仿真/输入/16A净充电上限验证_input.json")
    params["pack"]["parallel_cells"] = 5
    occ = charge.occ_bounds(params)
    # Apply the same 20 µΩ path-error magnitude in the overread direction
    # for the OCC early corner, as well as the underread direction in CC2.
    path = params["shunt"]["reference_algorithm_sense_path_underread_uohm_limit"] * 1e-6
    occ_early = occ["threshold_early_corner_mV"] * 1e-3 / (occ["shunt_high_corner_ohm_including_TCR"] + path)
    occ_late = occ["threshold_late_corner_mV"] * 1e-3 / (occ["shunt_low_corner_ohm_including_TCR"] - path)
    margin = params["occ_protection"]["normal_operation_peak_planning_margin_A"]
    params["occ_protection"]["normal_operation_peak_planning_margin_A"] += occ["early_trip_current_A"] - occ_early
    good = charge.default_algorithm_state(params, alpha_validated=True)
    evaluate = charge.reference_ccl_algorithm
    cases = {
        f"qualified_alpha_{alpha:g}": evaluate(params, 16.0, dict(good, alpha_max_branch_share=alpha))
        for alpha in (0.2, 0.25, 1 / 3, 0.6)
    }
    for key, value in {
        "cell_temperature_bounds_qualified_all_36_cells": False,
        "cell_temperature_lower_bound_C": 22.9,
        "cell_temperature_upper_bound_C": 27.1,
        "cell_voltage_headroom_valid": False,
        "bq_current_calibration_readback_valid": False,
        "occ_code3_readback_valid": False,
        "source_capability_fresh": False,
        "source_local_overshoot_bound_qualified": False,
        "source_current_overshoot_bound_A": 1.01,
        "shunt_temperature_upper_bound_C": 75.1,
        "sense_path_underread_bound_uohm": 20.01,
        "alpha_max_branch_share": 0.19,
    }.items():
        case = evaluate(params, 16.0, dict(good, **{key: value}))
        assert case["command_CCL_A"] == 0.0, (key, case)
        cases[f"reject_{key.replace('all_36_cells', 'all_45_cells')}"] = case
    fallback = charge.default_algorithm_state(params, alpha_validated=False)
    cases["fallback_qualified_no_counterflow"] = evaluate(params, 16.0, fallback)
    cases["reject_unknown_counterflow"] = evaluate(params, 16.0, dict(fallback, no_counterflow_current_qualified=False))
    assert cases["reject_unknown_counterflow"]["command_CCL_A"] == 0.0
    for result in cases.values():
        calc = result.get("calculation")
        if calc:
            assert calc["model_predicted_pack_peak_A"] <= 16
            assert calc["model_predicted_pack_peak_A"] <= occ_early - margin
            assert calc["predicted_max_branch_within_cell_rate_limit"]

    # The legacy model's state field refers to 36 cells. Here its qualification
    # means all 45 target cells; result metadata reflects the target scope.
    for result in cases.values():
        if "temperature_bounds_cover_all_36_cells" in result:
            result["temperature_bounds_cover_all_45_cells_hypothetical"] = result.pop("temperature_bounds_cover_all_36_cells")
        result["reason_codes"] = [reason.replace("ALL_36_CELLS", "ALL_45_CELLS") for reason in result["reason_codes"]]
    current = evaluate(params, 16.0, {})
    assert current["command_CCL_A"] == 0.0
    return {
        "current_unqualified_inputs_model_CCL_A": current["command_CCL_A"], "hypothetical_cases": cases,
        "occ_with_20uohm_path_early_A": occ_early, "occ_with_20uohm_path_late_A": occ_late,
        "counterexample_to_crediting_removable_load_A": {"sources": 20, "initial_load": 4, "initial_net": 16, "net_after_load_shed": 20},
        "source_reallocation_check": {"old_source_A": 6, "new_source_A": 6, "overlap_A": 12, "shared_budget_A": 6},
    }


def precharge(spice: Path) -> dict:
    # Model the verified precharge power path as a 25 °C on-resistance and an
    # ideal timed switch. This is an RC and load envelope, not a gate model.
    cases = {}
    for name, capacitance, resistance, load, short in (
        ("nominal_unloaded", 0.060, 8.2, 0, False),
        ("slow_unloaded", 0.072, 8.61, 0, False),
        ("slow_50mA", 0.072, 8.61, 0.05, False),
        ("slow_100mA", 0.072, 8.61, 0.10, False),
        ("bus_short_hot_resistor_corner", 0.060, 8.2 * 0.95 * 0.935, 0, True),
    ):
        on_time = 2.3459738837 if not short else 2.7377405271
        netlist = f"""BMS precharge RC envelope
Vpack src 0 37.8
Vtim ctl 0 PULSE(1 0 {on_time} 1u 1u 20 40)
Spre src n1 ctl 0 gate
.model gate SW(Ron=0.116 Roff=1e12 Vt=0.5 Vh=0.1)
Rpre n1 bus {resistance}
Cbus bus 0 {capacitance} IC=0
{'Rshort bus 0 1m' if short else f'Bload bus 0 I={load}*tanh(v(bus)/0.05)'}
.tran 0.1m 3.2 uic
.meas tran dv_end find par('v(src)-v(bus)') at={on_time - 0.001}
.meas tran initial_current find par('-i(Vpack)') at=0.0001
.meas tran resistor_joules integ par('(v(n1)-v(bus))^2/{resistance}') from=0 to={on_time}
.end
"""
        with TemporaryDirectory(prefix="bms-precharge-") as folder:
            circuit = Path(folder) / "precharge.cir"
            circuit.write_text(netlist, encoding="ascii")
            run = subprocess.run([str(spice), "-b", circuit.name], cwd=folder, capture_output=True, text=True, check=True)
        measures = {key: float(value) for key, value in re.findall(
            r"(?m)^(dv_end|initial_current|resistor_joules)\s*=\s*([-+0-9.eE]+)", run.stdout
        )}
        assert len(measures) == 3, (name, run.stdout, run.stderr)
        r_total = resistance + 0.116
        asymptotic_gap = load * r_total
        time_to_0p5 = None
        if not short and asymptotic_gap < 0.5:
            time_to_0p5 = r_total * capacitance * math.log((37.8 - asymptotic_gap) / (0.5 - asymptotic_gap))
        cases[name] = {"capacitance_F": capacitance, "resistor_ohm": resistance,
                       "standby_load_A": load, "ideal_switch_on_s": on_time,
                       "analytical_time_to_0p5V_s": time_to_0p5, **measures}
    assert cases["slow_unloaded"]["dv_end"] < 1.0
    assert cases["slow_unloaded"]["dv_end"] > 0.5
    assert cases["slow_50mA"]["dv_end"] > cases["slow_unloaded"]["dv_end"]
    assert math.isclose(cases["nominal_unloaded"]["dv_end"], 37.8 * math.exp(-2.34497 / (8.316 * 0.06)), rel_tol=0.005)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngspice", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "scope": "Same-source readback, analytical comparison, reference CCL and reduced RC model only",
        "baseline": "saved 9S4P, 3+3 MOS", "target": "9S5P, 45 cells; no new MOS adopted",
        "source": saved_source(), "discharge": discharge(), "charge": charge_cases(),
        "precharge": precharge(args.ngspice.resolve()),
        "model_sha256": {"runner": digest(Path(__file__)), "legacy_ccl": digest(Path(charge.__file__)),
                         "ccl_input": digest(BOARD / "仿真/输入/16A净充电上限验证_input.json")},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
