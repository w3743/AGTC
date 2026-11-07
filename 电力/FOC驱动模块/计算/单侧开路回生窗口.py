"""Compute conditional FOC feeder-open energy and passive commutation windows."""
import json
import math
from itertools import product
from pathlib import Path


def magnetic_energy(inductance_uh, phase_peak_a):
    return 0.75 * inductance_uh * 1e-6 * phase_peak_a**2


def capacitor_cases():
    initial_v = 37.8
    cases = []
    for capacitance_uf, limit_v in product((5, 10, 14.4, 58.9, 24_000, 30_000, 36_000),
                                          (39.209, 40.3, 41.724)):
        c = capacitance_uf * 1e-6
        headroom = 0.5 * c * (limit_v**2 - initial_v**2)
        for inductance_uh, phase_peak_a in ((0, 0), *product((20, 44, 80), (100, 120))):
            energy = magnetic_energy(inductance_uh, phase_peak_a)
            cases.append({
                "capacitance_uF": capacitance_uf, "initial_bus_V": initial_v, "limit_V": limit_v,
                "phase_effective_L_uH": inductance_uh, "phase_peak_A": phase_peak_a,
                "capacitor_headroom_J": headroom, "magnetic_energy_J": energy,
                "ideal_voltage_if_magnetic_energy_reaches_C_V": math.sqrt(initial_v**2 + 2*energy/c),
                "headroom_after_magnetic_energy_J": headroom - energy,
                "time_after_release_ms_by_residual_net_W": {
                    str(power): 1000 * max(0.0, headroom-energy) / power
                    for power in (250, 500, 1000, 1800)
                },
            })
    return cases


def passive_commutation_cases():
    v_bus, v_diode, i_initial, r_loop = 37.8, 0.7, 120.0, 0.02
    cases = []
    for loop_uh, emf_v in product((40, 88), (0, 20, 35, 39, 40)):
        drive = v_bus + 2*v_diode - emf_v
        duration_ms = 1e-3 * loop_uh / r_loop * math.log1p(r_loop*i_initial/drive) if drive > 0 else None
        cases.append({
            "loop_L_uH": loop_uh, "line_to_line_emf_V": emf_v, "bus_V": v_bus,
            "each_diode_V": v_diode, "loop_R_ohm": r_loop, "initial_loop_A": i_initial,
            "bus_plus_diodes_minus_emf_V": drive, "ideal_decay_to_zero_ms": duration_ms,
        })
    return cases


def main():
    report = {
        "status": "CONDITIONAL_ONLY",
        "scope": [
            "24/30/36 mF is three remaining local board-external capacitors at initial tolerance.",
            "14.4 uF is nominal near-bridge capacitance, 5/10 uF are sensitivities, 58.9 uF is nominal whole-board small capacitance. None guarantees effective capacitance.",
            "Magnetic energy is deposited before a separate residual net power term. Large calculated voltages only flag a crossed limit and do not predict a physical clamp or MOSFET survival.",
            "Commutation is a fixed two-phase loop with constant BUS, EMF and Vf. Null means no finite zero-current time in the model.",
        ],
        "capacitor_cases": capacitor_cases(),
        "passive_commutation_cases": passive_commutation_cases(),
        "separate_parasitic_step_V_at_50_A_per_us": {str(l): l * 0.05 for l in (50, 150, 300)},
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
