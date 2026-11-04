"""Three-node differential BUS model for the regulator/FOC interface."""
import cmath
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEAR = (10e-6, 3e-3, 3e-9)
BULK = (10e-3, 12e-3, 35e-9)
BULK_MOUNT = (1e-3, 30e-9)
CENTRAL = (5.4e-3, 4e-3, 20e-9)
SOURCE = (20e-3, 500e-9)


def z_rl(w, resistance, inductance):
    return complex(resistance, w * inductance)


def z_cap(w, cap, esr, esl):
    return z_rl(w, esr, esl) + 1 / (1j * w * cap)


def solve(frequency, pair_nh, bulk_count, currents, source_connected):
    w = math.tau * frequency
    y_wire = 1 / z_rl(w, 2e-3, pair_nh * 1e-9)
    y_near = 1 / z_cap(w, *NEAR)
    y_bulk = 1 / (z_cap(w, *BULK) / bulk_count + z_rl(w, *BULK_MOUNT)) if bulk_count else 0
    y_central = 1 / z_cap(w, *CENTRAL)
    y_source = 1 / z_rl(w, *SOURCE) if source_connected else 0
    y_local = y_near + y_bulk
    # Schur complement. All branch impedances include forward and return paths.
    local_sum = y_local + y_wire
    central_v = (currents[0] + y_wire / local_sum * sum(currents[1:])) / (
        y_central + y_source + 2 * y_wire * y_local / local_sum
    )
    local_v = [(current + y_wire * central_v) / local_sum for current in currents[1:]]
    wire_i = [(v - central_v) * y_wire for v in local_v]
    assert abs(currents[0] + sum(wire_i) - central_v * (y_central + y_source)) < 1e-9
    for current, v, wire in zip(currents[1:], local_v, wire_i):
        assert abs(current - wire - v * y_local) < 1e-9
    def polar(value):
        return {"magnitude": round(abs(value), 8), "phase_deg": round(math.degrees(cmath.phase(value)), 5)}
    return {
        "voltage_V": {"regulator": polar(central_v), "foc_A": polar(local_v[0]), "foc_B": polar(local_v[1])},
        "current_A": {
            "regulator_bank": polar(central_v * y_central),
            "source": polar(central_v * y_source),
            "FOC_to_regulator": [polar(i) for i in wire_i],
            "FOC_near_bridge_bank": [polar(v * y_near) for v in local_v],
            "FOC_external_bulk_bank": [polar(v * y_bulk) for v in local_v],
        },
    }


def storage_screen(bulk_count):
    def window(cap):
        return cap * (40.3**2 - 37.8**2) / 2
    cap_all = 5.4e-3 + bulk_count * 20e-3
    cap_isolated = bulk_count * 10e-3
    return {
        "capacitors_per_FOC": bulk_count, "whole_bus_nominal_F": cap_all,
        "common_window_J": window(cap_all), "common_window_ms_at_1kW": window(cap_all),
        "isolated_FOC_external_window_J": window(cap_isolated),
        "isolated_FOC_external_window_ms_at_500W": 2 * window(cap_isolated),
    }


def main():
    nominal_banks = (0, 2, 3)
    excitations = {
        "both_FOC_same_phase": (0, 1, 1),
        "both_FOC_opposed": (0, 1, -1),
        "FOC_A_only": (0, 1, 0),
        "regulator_only": (1, 0, 0),
    }
    rows = [
        {
            "frequency_Hz": f, "wire_pair_nH": wire, "capacitors_per_FOC": bank,
            "source_connected": connected, "excitation_1_Arms": name,
            **solve(f, wire, bank, currents, connected),
        }
        for f in (1000, 10000, 20000, 40000, 100000, 1000000)
        for wire in (50, 150, 300)
        for bank in nominal_banks
        for connected in (False, True)
        for name, currents in excitations.items()
    ]
    powers = []
    for bus in (22.5, 24, 27.6923076923, 32.4, 37.8):
        powers.append({
            "V": bus, "FOC_input_A_each": 1800 / bus, "dual_FOC_input_A": 3600 / bus,
            "DC_DC_min_bus_A_if_pack_supplies_130A": max(0, 3600 / bus - 130),
            "DC_DC_min_bus_W_if_pack_supplies_130A": max(0, 3600 - 130 * bus),
            "ideal_min_phase_peak_A_at_unity_pf_and_linear_SVPWM": 1800 / (math.sqrt(3) / 2 * bus),
        })
    source = ROOT / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
    result = {
        "status": "PARAMETER_SENSITIVITY_NOT_RATING",
        "focOfficialEnetSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "notes": [
            "1 A RMS sinusoidal injections are transfer functions, not measured PWM spectra. Preserve complex phase when superposing left FOC, right FOC and regulator sources.",
            "Wire 50/150/300 nH and 2 mOhm represent the full positive and return loop including joints; a <=10 cm one-way length does not establish these values.",
            "All ESR/ESL values, mounting loop, source impedance and effective near capacitance are illustrative. The 5.4 mF and individual 10 mF are nominal.",
            "Near-bridge bank lumps six ceramics per FOC. This model does not resolve individual commutation cells, MOS switching edges, motor regen or chopper response.",
            "Source disconnected means battery-branch AC admittance zero. Both FOC feeders are still connected. A feeder-open fault has no central sink for that FOC, modeled only by its ideal local storage comparison.",
            "Storage window 37.8 to 40.3 V neglects ESR/ESL and assumes specified net injection and no other sinks. 40.3 V is a comparison point, not an allowed voltage.",
            "Dual-peak source budget assumes BUS equals pack voltage and no auxiliary loads, harness drops or conversion loss. 130 A is a candidate, not a BMS rating.",
        ],
        "assumptions": {
            "near_FOC_each_F_ohm_H": NEAR, "external_cap_each_F_ohm_H": BULK,
            "external_bank_mount_ohm_H": BULK_MOUNT, "regulator_bank_F_ohm_H": CENTRAL,
            "source_ohm_H_when_connected": SOURCE,
        },
        "power_balance": powers,
        "storage": [storage_screen(n) for n in nominal_banks],
        "phasor_cases": rows,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} cases; nodal KCL checked in all cases")


if __name__ == "__main__":
    main()
