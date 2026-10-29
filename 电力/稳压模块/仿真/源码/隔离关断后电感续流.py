"""Run bounded post-isolation inductor-clamp cases with ngspice, headless.

This evaluates the already-open isolation switch and bridge channels-off state.
It does not simulate the gate-off transition, switch parasitics, or vehicle-level
regeneration. The two STPS30100ST devices use ST's unmodified PSpice library.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess


MODULE = Path(__file__).resolve().parents[2]
POWER = MODULE.parent
EXE = (
    POWER
    / "FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe"
)
OUT = MODULE / "仿真/结果/隔离关断续流"
MODEL = MODULE / "仿真/输入/厂家模型/ST_Schottky_PSpice_v6/ST_POWER_SCHOTTKY_V10.LIB"
MODEL_ZIP = MODULE / "仿真/输入/厂家模型/ST_Schottky_PSpice_Models_v6.zip"

L_CASES = ((12.8e-6, "12p8uH"), (24e-6, "24uH"))
I_CASES = ((80.0, "Ip80A"), (-80.0, "In80A"))
VBUS0 = 37.8
VSC0 = 17.6  # 8 x 2.20 V/cell upper normal operating threshold
CBUS = 5.4e-3  # six 900 uF capacitors, nominal total
CSCF = 3.2e-6  # 2.2 uF film + 1.0 uF HF local to SC_FUSED
RL = 2.23e-3  # two inductor windings, estimated total DCR at 100 C
R_SW_REF = 1e6  # numeric DC reference only; negligible during the pulse
MAX_STEP = 5e-9
TSTOP = 150e-6

NETLIST_TEMPLATE = r"""Post-isolation freewheel clamp; {case_id}
* Current reference: positive i(Lmain_a) flows SW -> SC_FUSED.
* Isolation MOS channels are already off; only QH/QL body diodes remain.
* The SC storage stack is disconnected. No pack source: 5.4 mF alone absorbs L energy.
.include "ST_POWER_SCHOTTKY_V10.LIB"
.temp 25
.param L_TOTAL={l_value}
.param I0={i_value}
.param SW0={sw0_value}
.options method=gear reltol=1e-5 abstol=1e-10 vntol=1e-8
.ic V(N_SC_LO)=17.6 V(N_SC_HI)=37.8 V(N_QH_BODY)=37.8 V(N_QL_BODY)={{SW0}}

C_BUS BUS 0 5.4m IC=37.8
C_SC_LOCAL SC_FUSED 0 3.2u IC=17.6
R_L_DCR LDCR SC_FUSED 2.23m
L_MAIN_A SW LMID {{L_TOTAL/2}} IC={{I0}}
L_MAIN_B LMID LDCR {{L_TOTAL/2}} IC={{I0}}

* STPS30100ST vendor subcircuit pins: 1=A, 2=K, 3=A.
* V_SC_LO and V_SC_HI are zero-volt current probes in the diode branches.
X_D_SC_LO 0 N_SC_LO 0 STPS30100ST
V_SC_LO N_SC_LO SC_FUSED 0
X_D_SC_HI SC_FUSED N_SC_HI SC_FUSED STPS30100ST
V_SC_HI N_SC_HI BUS 0

* Explicit generic D-only body-diode equivalents for the off IRLB4030PBF QH/QL.
* They intentionally contain no MOS channel, Coss, Qrr, package or thermal model.
D_QH_BODY SW N_QH_BODY D_MOS_BODY
V_QH_SENSE N_QH_BODY BUS 0
D_QL_BODY 0 N_QL_BODY D_MOS_BODY
V_QL_SENSE N_QL_BODY SW 0
R_SW_NUMERIC_REFERENCE SW 0 1Meg
.model D_MOS_BODY D(IS=1e-8 N=1.5 RS=3m BV=100 IBV=1m)

.control
set noaskquit
set wr_singlescale
tran 5n 150u 1n 5n uic
wrdata {wave_name} v(BUS) v(SC_FUSED) v(SW) i(L_MAIN_A) i(V_SC_LO) i(V_SC_HI) i(V_QH_SENSE) i(V_QL_SENSE) v(N_SC_LO) v(N_SC_HI) v(N_QH_BODY) v(N_QL_BODY)
quit
.endc
.end
"""


def parse_wave(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 13:
            continue
        try:
            rows.append([float(item) for item in fields])
        except ValueError:
            continue
    return rows


def extrema(rows: list[list[float]], column: int) -> dict[str, float]:
    values = [row[column] for row in rows]
    return {
        "initial": values[0],
        "min": min(values),
        "max": max(values),
        "final": values[-1],
    }


def integrate(rows: list[list[float]], power_at) -> float:
    total = 0.0
    for left, right in zip(rows, rows[1:]):
        dt = right[0] - left[0]
        total += 0.5 * (power_at(left) + power_at(right)) * dt
    return total


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if not EXE.is_file():
        raise FileNotFoundError(f"ngspice console executable not found: {EXE}")
    if not MODEL.is_file() or not MODEL_ZIP.is_file():
        raise FileNotFoundError("ST's official Schottky model input is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep a byte-identical model beside the generated decks so they can be
    # run directly from this result directory without Unicode include paths.
    shutil.copyfile(MODEL, OUT / MODEL.name)

    cases: list[dict[str, object]] = []
    for l_total, l_name in L_CASES:
        for current, i_name in I_CASES:
            case_id = f"{l_name}_{i_name}"
            wave_name = f"{case_id}.txt"
            cir_path = OUT / f"{case_id}.cir"
            log_path = OUT / f"{case_id}.log"
            wave_path = OUT / wave_name
            cir_path.write_text(
                NETLIST_TEMPLATE.format(
                    case_id=case_id,
                    l_value=f"{l_total:.10g}",
                    i_value=f"{current:.10g}",
                    sw0_value="-1.1187" if current > 0 else "38.9187",
                    wave_name=wave_name,
                ),
                encoding="ascii",
            )
            wave_path.unlink(missing_ok=True)
            run = subprocess.run(
                [str(EXE.resolve()), "-b", cir_path.name],
                cwd=OUT,
                capture_output=True,
                timeout=90,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            log = run.stdout.decode("utf-8", "replace") + run.stderr.decode(
                "utf-8", "replace"
            )
            log_path.write_text(log, encoding="utf-8")
            if run.returncode or "Error:" in log or not wave_path.is_file():
                raise RuntimeError(f"{case_id} ngspice failure:\n{log[-3000:]}")

            rows = parse_wave(wave_path)
            if len(rows) < 10 or rows[-1][0] < TSTOP * 0.999:
                raise RuntimeError(
                    f"{case_id} waveform incomplete: rows={len(rows)}, "
                    f"last_t={rows[-1][0] if rows else None}"
                )

            # Data columns: t, Vbus, Vsc, Vsw, IL, IDlo, IDhi, IQH, IQL,
            # V(cathode-low), V(cathode-high), V(cathode-QH), V(cathode-QL).
            i_l = extrema(rows, 4)
            v_bus = extrema(rows, 1)
            v_sc = extrema(rows, 2)
            v_sw = extrema(rows, 3)
            path_peaks = {
                "D_SC_LO_A_positive": max(row[5] for row in rows),
                "D_SC_HI_A_positive": max(row[6] for row in rows),
                "QH_body_A_positive": max(row[7] for row in rows),
                "QL_body_A_positive": max(row[8] for row in rows),
            }

            # Use the measured state at the first recorded transient point.
            i_start = i_l["initial"]
            v_bus_start = v_bus["initial"]
            v_sc_start = v_sc["initial"]
            e_l_initial = 0.5 * l_total * i_start**2
            e_l_final = 0.5 * l_total * i_l["final"] ** 2
            e_bus_initial = 0.5 * CBUS * v_bus_start**2
            e_bus_final = 0.5 * CBUS * v_bus["final"] ** 2
            e_sc_initial = 0.5 * CSCF * v_sc_start**2
            e_sc_final = 0.5 * CSCF * v_sc["final"] ** 2
            e_dcr = integrate(rows, lambda row: RL * row[4] ** 2)
            e_sw_ref = integrate(rows, lambda row: row[3] ** 2 / R_SW_REF)
            e_st_lo = integrate(rows, lambda row: (0.0 - row[9]) * row[5])
            e_st_hi = integrate(rows, lambda row: (row[2] - row[10]) * row[6])
            e_qh = integrate(rows, lambda row: (row[3] - row[11]) * row[7])
            e_ql = integrate(rows, lambda row: (0.0 - row[12]) * row[8])
            e_initial = e_l_initial + e_bus_initial + e_sc_initial
            e_final = e_l_final + e_bus_final + e_sc_final
            e_diode = e_st_lo + e_st_hi + e_qh + e_ql
            e_residual = e_initial - e_final - e_dcr - e_diode - e_sw_ref

            expected = (
                ("D_SC_HI", "Q_LOW_BODY")
                if current > 0
                else ("D_SC_LO", "Q_HIGH_BODY")
            )
            energy_path = (
                "SW -> L_MAIN_A/B -> SC_FUSED -> D_SC_HI -> BUS_EXT -> C_BUS -> GND_SYS -> Q_MAIN_L body diode -> SW"
                if current > 0
                else "GND_SYS -> D_SC_LO -> SC_FUSED -> L_MAIN_B/A -> SW -> Q_MAIN_H body diode -> BUS_EXT -> C_BUS -> GND_SYS"
            )
            active_indices = (6, 8) if current > 0 else (5, 7)
            inactive_indices = (5, 7) if current > 0 else (6, 8)
            active_peak = min(max(row[col] for row in rows) for col in active_indices)
            inactive_abs_peak = max(
                max(abs(row[col]) for row in rows) for col in inactive_indices
            )
            extinguish_index = None
            if abs(rows[-1][4]) < 0.1:
                extinguish_index = len(rows) - 1
                for idx in range(len(rows) - 2, -1, -1):
                    if abs(rows[idx][4]) < 0.1:
                        extinguish_index = idx
                    else:
                        break
            extinguish_s = rows[extinguish_index][0] if extinguish_index is not None else None
            closure_pct = 100.0 * e_residual / e_initial if e_initial else math.nan
            energy_balance_check = abs(closure_pct) < 0.01
            bus_peak_check = v_bus["max"] < 40.3
            clamp_check = (
                v_sc["min"] > -2.0
                and v_sc["max"] - v_bus["max"] < 2.0
                and v_sw["min"] > -2.0
                and v_sw["max"] - v_bus["max"] < 2.0
            )

            cases.append(
                {
                    "case": case_id,
                    "current_reference": (
                        "positive SW_to_SC_FUSED"
                        if current > 0
                        else "negative SC_FUSED_to_SW"
                    ),
                    "L_total_H": l_total,
                    "initial_inductor_current_A": current,
                    "measured_window_start_current_A": i_start,
                    "initial_magnetic_energy_J": e_l_initial,
                    "sample_count": len(rows),
                    "analysis_window_start_s": rows[0][0],
                    "last_time_s": rows[-1][0],
                    "current_extinguished_below_0p1A_s": extinguish_s,
                    "node_voltage_V": {
                        "BUS": v_bus,
                        "SC_FUSED": v_sc,
                        "SW": v_sw,
                    },
                    "inductor_current_A": i_l,
                    "expected_conduction_path": list(expected),
                    "energy_path_loop": energy_path,
                    "branch_current_peak_A": path_peaks,
                    "expected_path_lower_peak_A": active_peak,
                    "opposite_path_absolute_peak_A": inactive_abs_peak,
                    "energy_J": {
                        "initial_BUS_cap": e_bus_initial,
                        "final_BUS_cap": e_bus_final,
                        "initial_SC_FUSED_cap": e_sc_initial,
                        "final_SC_FUSED_cap": e_sc_final,
                        "final_inductor": e_l_final,
                        "inductor_DCR_loss": e_dcr,
                        "SW_numeric_reference_loss": e_sw_ref,
                        "ST_diode_conduction_loss": e_st_lo + e_st_hi,
                        "generic_body_diode_conduction_loss": e_qh + e_ql,
                        "balance_residual": e_residual,
                        "balance_residual_percent_of_initial_total": closure_pct,
                    },
                    "bus_rise_V": v_bus["final"] - v_bus_start,
                    "energy_balance_check": energy_balance_check,
                    "BUS_below_40p3V_reference_check": bus_peak_check,
                    "SC_FUSED_and_SW_diode_clamp_check": clamp_check,
                    "energy_path_check": (
                        active_peak > 10.0
                        and inactive_abs_peak < 1.0
                        and extinguish_s is not None
                        and e_bus_final > e_bus_initial
                        and energy_balance_check
                        and bus_peak_check
                        and clamp_check
                    ),
                    "source_netlist": cir_path.name,
                    "source_netlist_sha256": sha256(cir_path),
                    "waveform": wave_path.name,
                    "waveform_sha256": sha256(wave_path),
                    "log": log_path.name,
                    "log_sha256": sha256(log_path),
                }
            )

    report = {
        "status": "BOUNDED_POST_SHUTDOWN_PATH_SIMULATED_NOT_REAL_SHUTDOWN",
        "date": "2026-10-30",
        "simulator": "ngspice 47 portable console; batch mode; no GUI",
        "official_ST_model": {
            "source_url": "https://www.st.com/resource/en/spice_model/st_power_schottky_diodes_pspice_models_v6.zip",
            "product_url": "https://www.st.com/en/diodes-and-rectifiers/stps30100st.html",
            "resource": "Power Schottky diode PSpice models v10.0 (2023-05-05)",
            "subcircuit": "STPS30100ST",
            "pin_order": ["1=A", "2=K", "3=A"],
            "zip_sha256": sha256(MODEL_ZIP),
            "library_sha256": sha256(MODEL),
            "unmodified": True,
            "ngspice_compatibility": "This exact ST PSpice subcircuit ran successfully in the supplied ngspice 47; ST does not claim ngspice validation.",
            "limitations": [
                "No thermal node or self-heating.",
                "TT=0 in the ST model; reverse-recovery behavior is not represented.",
            ],
        },
        "electrical_boundary": {
            "D_SC_LO": "anode=GND_SYS, cathode=SC_FUSED",
            "D_SC_HI": "anode=SC_FUSED, cathode=BUS_EXT",
            "inductor_current_positive_reference": "SW -> SC_FUSED",
            "inductors": "Two series IHXL2000VZEB100M3A represented by fixed total L; both gate channels are already off and isolation FETs are open at t=0.",
            "L_total_H_cases": [value for value, _ in L_CASES],
            "initial_current_A_cases": [value for value, _ in I_CASES],
            "initial_SC_FUSED_V": VSC0,
            "initial_BUS_V": VBUS0,
            "C_BUS_F": CBUS,
            "C_SC_FUSED_F": CSCF,
            "inductor_DCR_ohm": RL,
            "inductor_DCR_source": "Design detail estimate for two inductors at 100 C; fixed resistor.",
            "SW_numerical_reference_ohm": R_SW_REF,
            "SW_numerical_reference_note": "Only resolves the otherwise floating switch node after current extinguishes; its pulse energy is integrated in the report.",
            "bus_storage_boundary": "Only the six 900 uF/80 V bus capacitors are modeled; battery/source absorption is excluded, so this is a capacitor-only bus-voltage rise estimate.",
            "SC_FUSED_capacitance_boundary": "2.2 uF film + 1.0 uF local HF capacitance; ideal C, initialized to the 8 x 2.20 V/cell upper normal threshold.",
            "QH_QL_off_state": "Generic D-only body-diode equivalents, no MOS channel, Coss, Qrr, package, wiring inductance, or thermal model.",
            "max_timestep_s": MAX_STEP,
            "simulation_stop_s": TSTOP,
            "fixed_time_step_method": "Gear, maximum step 5 ns; transient uses UIC to preserve assigned L/C initial conditions.",
            "reported_window_start": "1 ns, after the ST PSpice junction-cap initialization impulse; energy balance uses the measured state at this first recorded point. The UIC assigned initial state is +/-80 A at t=0.",
        },
        "cases": cases,
        "checks": {
            "all_cases_completed": len(cases) == 4,
            "all_expected_diode_paths_conduct": all(c["energy_path_check"] for c in cases),
            "all_energy_balances_within_0p01_percent": all(c["energy_balance_check"] for c in cases),
            "all_BUS_peaks_below_40p3V_reference": all(c["BUS_below_40p3V_reference_check"] for c in cases),
            "all_nodes_stay_within_diode_clamp_envelope": all(c["SC_FUSED_and_SW_diode_clamp_check"] for c in cases),
            "runner_sha256": sha256(Path(__file__)),
            "source_files": [c["source_netlist"] for c in cases],
        },
        "important_limits": [
            "This starts after isolation and gate shutdown are complete; it does not simulate gate-driver delay, MOS turn-off, isolation opening, or fault detection.",
            "The 80 A initial current is the design's provisional screening point, not an established continuous-current rating or validated trip threshold.",
            "Fixed 12.8 uH and 24 uH cases bracket the documented high-current screening value and air-inductance tolerance; nonlinear saturation is not modeled.",
            "Capacitors are ideal. No PCB/lead ESL, diode package inductance, ringing, or semiconductor thermal rise is modeled; reported node maxima are not physical switching-spike limits.",
            "The 1 MOhm SW reference is a numerical DC path, not a board component; its simulated pulse energy is included in the balance.",
            "The 24 uH/80 A fixed-L case stores 76.8 mJ; it is an energy screen, not a claim that the physical inductors retain 24 uH at 80 A.",
            "This does not verify current rating, repetitive pulse thermal limits, shutdown safety, BUS battery charging acceptance, or vehicle braking performance.",
        ],
    }
    result_path = OUT / "隔离关断续流_汇总.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], ensure_ascii=False))
    for case in cases:
        print(
            json.dumps(
                {
                    "case": case["case"],
                    "path": case["expected_conduction_path"],
                    "bus_rise_V": case["bus_rise_V"],
                    "V_SC_FUSED": case["node_voltage_V"]["SC_FUSED"],
                    "I_L_final_A": case["inductor_current_A"]["final"],
                    "energy_residual_percent": case["energy_J"][
                        "balance_residual_percent_of_initial_total"
                    ],
                    "energy_path_check": case["energy_path_check"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
