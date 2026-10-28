#!/usr/bin/env python3
"""Average-value 1 s DC-link / brake-chopper screening model.

This intentionally complements the ngspice PWM model: it solves capacitor
energy and a one-way battery source over 1 s, not switching waveforms.  It is
not motor/firmware or thermal qualification.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "结果" / "dc_link_1s"

NOM = {
    "r_bus_top_ohm": 162_000.0,
    "r_bus_bottom_ohm": 10_000.0,
    "r_ref_top_ohm": 10_200.0,
    "r_ref_bottom_ohm": 10_000.0,
    "r_hyst_ohm": 200_000.0,
    "r_pullup_ohm": 4_700.0,
    "r_dump_ohm": 1.0,
    "dc_link_esr_ohm": 6.333e-3,
    "v_safe_v": 4.9764817276,
    "v_ol_v": 0.2,
    "tlv_offset_v": 0.0,
    "r_tolerance": 0.01,
    "r_tolerance_bus_divider_ref": 0.001,
    "v_safe_tolerance": 0.025,
    "tlv_offset_max_v": 0.0055,
}


def ref_node(vsafe: float, vout: float, rtop: float, rbot: float, rhyst: float) -> float:
    return (vsafe / rtop + vout / rhyst) / (1 / rtop + 1 / rbot + 1 / rhyst)


def output_high_and_ref(vsafe: float, rref_top: float, rref_bottom: float,
                        rhyst: float, rpull: float) -> tuple[float, float]:
    # Solve the two-node passive high-state network: BUSOV_N pulled up to
    # SAFE_5V and coupled to V_OV_REF through Rhyst.
    a = 1 / rpull + 1 / rhyst
    b = -1 / rhyst
    c = -1 / rhyst
    d = 1 / rref_top + 1 / rref_bottom + 1 / rhyst
    y1 = vsafe / rpull
    y2 = vsafe / rref_top
    det = a * d - b * c
    vout = (y1 * d - b * y2) / det
    vref = (a * y2 - c * y1) / det
    return vout, vref


def thresholds(values: dict) -> dict:
    ratio = values["r_bus_bottom_ohm"] / (values["r_bus_top_ohm"] + values["r_bus_bottom_ohm"])
    vout_high, vref_high = output_high_and_ref(
        values["v_safe_v"], values["r_ref_top_ohm"], values["r_ref_bottom_ohm"],
        values["r_hyst_ohm"], values["r_pullup_ohm"])
    vref_low = ref_node(values["v_safe_v"], values["v_ol_v"],
                        values["r_ref_top_ohm"], values["r_ref_bottom_ohm"],
                        values["r_hyst_ohm"])
    offset = values["tlv_offset_v"]
    return {
        "vout_high_v": vout_high,
        "vref_high_v": vref_high,
        "vref_low_v": vref_low,
        "bus_on_v": (vref_high + offset) / ratio,
        "bus_off_v": (vref_low + offset) / ratio,
    }


def ovp_corner_sweep() -> dict:
    r_names = ["r_bus_top_ohm", "r_bus_bottom_ohm", "r_ref_top_ohm",
               "r_ref_bottom_ohm", "r_hyst_ohm", "r_pullup_ohm"]
    all_pairs = []
    # Include the full-temperature TLV1704 output-low bound as well as the
    # input-offset bound. VOL=0.9 V is the datasheet maximum at the stated
    # sink-current condition; 0 V is the opposite endpoint.
    for signs in itertools.product((-1, 1), repeat=len(r_names) + 3):
        c = dict(NOM)
        for i, (name, sign) in enumerate(zip(r_names, signs[:len(r_names)])):
            tol = NOM["r_tolerance_bus_divider_ref"] if i < 4 else NOM["r_tolerance"]
            c[name] *= 1 + sign * tol
        c["v_safe_v"] *= 1 + signs[-3] * NOM["v_safe_tolerance"]
        c["tlv_offset_v"] = signs[-2] * NOM["tlv_offset_max_v"]
        c["v_ol_v"] = 0.9 if signs[-1] > 0 else 0.0
        all_pairs.append(thresholds(c))
    ons = [x["bus_on_v"] for x in all_pairs]
    offs = [x["bus_off_v"] for x in all_pairs]
    nominal = thresholds(dict(NOM))
    return {
        "nominal": nominal,
        "corner_method": "all 512 simultaneous endpoint combinations: four divider resistors ±0.1%, 200k feedback and 4.7k pull-up ±1%, SAFE_5V ±2.5%, TLV1704 input offset ±5.5mV full-temp and VOL 0..0.9V endpoints; TLV VOL max applies only at stated sink-current condition",
        "bus_on_min_v": min(ons), "bus_on_max_v": max(ons),
        "bus_off_min_v": min(offs), "bus_off_max_v": max(offs),
        "corner_count": len(all_pairs),
    }


def profile(name: str, vdc: float, c_f: float, pac_w: float,
            bridge_loss_w: float, safe5_v: float = NOM["v_safe_v"],
            core5_v: float = 5.0, dump_mode: str = "hysteretic",
            safe5_loss_s: float | None = None, thermal_open_s: float | None = None,
            regen_sink_limit_w: float = 0.0, regen_sink_quota_j: float | None = 0.0,
            duration_s: float = 1.0, dt_s: float = 10e-6,
            profile_csv: Path | None = None) -> dict:
    """Pac>0 motor drive; Pac<0 motor regenerative power at motor terminals."""
    ov = thresholds(dict(NOM))
    v_on = ov["bus_on_v"]
    v_off = ov["bus_off_v"]
    # Positive power is requested from the DC link. For regeneration, a
    # positive bridge loss reduces bus-returned power: Pbus = Pac + Ploss.
    pinv = pac_w + (bridge_loss_w if pac_w != 0 else 0.0)
    e_bus = 0.5 * c_f * vdc * vdc
    e_initial = e_bus
    dt = dt_s
    steps = int(math.ceil(duration_s / dt))
    source_r = 0.010  # explicit Thevenin path assumption, not a battery spec
    source_i_limit = 100.0
    dump_resistor_r = 1.0
    # IRLB4030 at SAFE_5V≈4.98V: 4.5mΩ max at VGS=4.5V, hot factor 2.5.
    q7_ron = 11.25e-3
    is_on = dump_mode == "stuck_on"
    enabled = dump_mode in ("hysteretic", "stuck_on")
    transitions = []
    first_on_s = None
    t_100 = None
    t_63 = None
    t_75 = None
    t_43 = None
    v_min = float("inf")
    v_max = 0.0
    i_source_max = 0.0
    i_inv_abs_max = 0.0
    i_dump_max = 0.0
    i_cap_rms_acc = 0.0
    esr_j = 0.0
    dump_j = 0.0
    dump_resistor_j = 0.0
    q7_j = 0.0
    source_j = 0.0
    inv_j = 0.0
    regen_sink_j = 0.0
    on_s = 0.0
    pre_failure = True
    failure_reason = None
    samples = []
    sample_stride = max(1, round(100e-6 / dt))  # 100 us display data
    for k in range(steps + 1):
        t = min(k * dt, duration_s)
        v_cap = math.sqrt(max(0.0, 2 * e_bus / c_f))
        v = v_cap
        safe_live = safe5_v > 4.5 and (safe5_loss_s is None or t < safe5_loss_s)
        thermal_ok = thermal_open_s is None or t < thermal_open_s
        dump_allowed = enabled and safe_live and thermal_ok
        if dump_mode == "stuck_on":
            # Q7 D-S short bypasses gate, SAFE_5V and thermal-control logic;
            # only a physical upstream disconnect can remove this path.
            new_on = True
        elif dump_mode == "hysteretic" and dump_allowed:
            new_on = (v >= v_on) if not is_on else (v > v_off)
        else:
            new_on = False
        if new_on != is_on:
            transitions.append({"time_s": t, "dump_on": new_on, "bus_v": v,
                                "reason": "BUS_OV hysteresis" if dump_allowed else "SAFE/thermal unavailable"})
            if new_on and first_on_s is None:
                first_on_s = t
        is_on = new_on

        # Solve the local series ESR drop approximately by fixed-point
        # iteration. Positive Icap charges the ideal capacitor and raises the
        # bus-side terminal above its internal voltage by ESR*Icap.
        for _ in range(4):
            isrc = min(source_i_limit, max(0.0, (vdc - v) / source_r))
            dump_r_total = dump_resistor_r + (q7_ron if is_on and dump_mode != "stuck_on" else 0.0)
            idump = v / dump_r_total if is_on else 0.0
            iinv = pinv / max(v, 1.0)
            remaining_sink_j = (float("inf") if regen_sink_quota_j is None
                                else max(0.0, regen_sink_quota_j - regen_sink_j))
            p_sink = min(max(0.0, -pinv), max(0.0, regen_sink_limit_w),
                         remaining_sink_j / dt if dt > 0 else 0.0)
            i_cap = isrc - iinv - idump - p_sink / max(v, 1.0)
            v = v_cap + NOM["dc_link_esr_ohm"] * i_cap
        # One-way source: it can supply the inverter but cannot accept reverse
        # current. During regen pinv<0 and inverter current injects into BUS.
        p_source = v * isrc
        p_dump_resistor = idump * idump * dump_resistor_r
        p_q7 = idump * idump * q7_ron if is_on and dump_mode != "stuck_on" else 0.0
        p_dump = p_dump_resistor + p_q7
        p_inv = pinv
        p_net_cap = p_source - p_inv - p_dump - p_sink
        p_esr = i_cap * i_cap * NOM["dc_link_esr_ohm"]
        v_min = min(v_min, v)
        v_max = max(v_max, v)
        i_source_max = max(i_source_max, isrc)
        i_inv_abs_max = max(i_inv_abs_max, abs(iinv))
        i_dump_max = max(i_dump_max, idump)
        if k < steps:
            i_cap_rms_acc += i_cap * i_cap * dt
            esr_j += p_esr * dt
            dump_j += p_dump * dt
            dump_resistor_j += p_dump_resistor * dt
            q7_j += p_q7 * dt
            source_j += p_source * dt
            inv_j += p_inv * dt
            regen_sink_j += p_sink * dt
            on_s += dt if is_on else 0.0

        if t_43 is None and v >= 43.0:
            t_43 = t
        if t_63 is None and v >= 63.0:
            t_63 = t
        if t_75 is None and v >= 75.0:
            t_75 = t
        if t_100 is None and v >= 100.0:
            t_100 = t
        if profile_csv is not None and (k % sample_stride == 0 or k == steps):
            samples.append((t, v, isrc, iinv, idump, p_sink, i_cap, safe_live, core5_v > 4.5))

        if v >= 100.0:
            pre_failure = False
            failure_reason = "Vbus reached 100 V rating boundary (switching model beyond this point invalid)"
            break
        if v <= 10.0 and pac_w > 0:
            pre_failure = False
            failure_reason = "Vbus fell below 10 V screening stop; requested constant power is no longer valid"
            break
        if k == steps:
            break
        e_bus = max(0.0, e_bus + (p_net_cap - p_esr) * dt)

    v_end = v
    elapsed_s = min(duration_s, k * dt)
    measured_duration = max(dt, elapsed_s)
    if profile_csv is not None:
        with profile_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "vbus_v", "isource_a", "iinverter_bus_a_positive_draw_negative_regen",
                        "idump_a", "i_sink_a_positive_absorb", "icap_slow_model_a", "safe5_live", "core5_live"])
            w.writerows(samples)
    return {
        "name": name, "vdc_initial_v": vdc, "c_link_f": c_f,
        "pac_motor_terminal_w": pac_w, "bridge_loss_assumed_w": bridge_loss_w,
        "p_inverter_from_bus_w_positive_draw_negative_regen": pinv,
        "safe5_initial_v": safe5_v, "core5_initial_v": core5_v,
        "dump_mode": dump_mode, "safe5_loss_s": safe5_loss_s,
        "thermal_open_s": thermal_open_s, "duration_requested_s": duration_s,
        "regen_sink_limit_w": regen_sink_limit_w, "regen_sink_quota_j": regen_sink_quota_j,
        "duration_simulated_s": elapsed_s, "dt_s": dt,
        "bus_on_nom_v": v_on, "bus_off_nom_v": v_off,
        "vbus_min_v": v_min, "vbus_max_v": v_max, "vbus_end_v": v_end,
        "first_dump_on_s": first_on_s, "first_43v_s": t_43,
        "first_63v_s": t_63, "first_75v_s": t_75, "first_100v_s": t_100,
        "source_current_peak_a": i_source_max, "inverter_bus_current_abs_peak_a": i_inv_abs_max,
        "dump_current_peak_a": i_dump_max,
        "dump_energy_j_total_including_Q7_Ron": dump_j,
        "dump_resistor_energy_j_total": dump_resistor_j,
        "dump_resistor_energy_j_each_of_two_equal_resistors": dump_resistor_j / 2,
        "Q7_channel_energy_j": q7_j,
        "source_energy_j": source_j, "inverter_energy_j_signed_draw_positive": inv_j,
        "regen_sink_energy_j_accepted": regen_sink_j,
        "signed_external_source_energy_j_positive_to_bus_negative_absorbed": source_j - regen_sink_j,
        "dc_link_esr_energy_j_low_frequency_only_and_subtracted": esr_j,
        "dc_link_current_rms_a_low_frequency_only": math.sqrt(i_cap_rms_acc / measured_duration),
        "energy_balance_error_j": (e_bus - e_initial) - (source_j - inv_j - dump_j - regen_sink_j - esr_j),
        "dump_on_time_s": on_s, "dump_on_duty_fraction": on_s / measured_duration,
        "dump_transitions": transitions,
        "two_board_energy_j": {"motor_terminal_abs": 2 * abs(pac_w) * elapsed_s,
                               "inverter_signed_draw": 2 * inv_j, "dump_total": 2 * dump_j},
        "reached_1s_without_rating_stop": pre_failure and k >= steps,
        "failure_reason": failure_reason,
        "assumptions": [
            "bulk bank initial voltage is already precharged; precharge/inrush is outside this averaged profile",
            "battery source is one-way and represented by nominal voltage, 10mOhm series resistance and 100A source current limit; no reverse energy acceptance",
            "regen_sink_limit_w/quota_j is an explicit hypothetical external absorber envelope, not an asserted battery/DC-DC/supercapacitor capability",
            "constant motor-terminal power for the entire profile is intentionally conservative for dump thermal sizing; BUSOV/BREAK_N may stop commanded PWM, but motor inductive current decay and diode regeneration are not modeled",
            "healthy chopper uses 11.25mOhm Q7 on-resistance (4.5mOhm max at VGS=4.5V times hot factor 2.5) in series with the 1Ohm resistor bank; Q7 D-S stuck-on failure is modeled as zero channel resistance while resistor bank remains connected",
            "bridge loss is a constant input bracket, not measured switching loss; pac and bus power remain separately reported",
            "PWM ripple, bulk/ceramic ESL/frequency spectrum, MOSFET switching, TVS clamp, gate-driver/comparator propagation and resistor thermal impedance are not included; low-frequency capacitor ESR loss is subtracted, high-frequency ESR heating is not",
            "C=24mF is a -20% capacitance corner; nominal C=30mF. Threshold uses the candidate resistor network at nominal component values",
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vth = ovp_corner_sweep()
    cases = []
    for v in (22.5, 32.4, 37.8):
        for loss in (0.0, 100.0, 200.0):
            # Pac=-1.8kW at motor terminals. Positive losses reduce bus return.
            name = f"regen_{v:g}V_loss{loss:g}W_dump30mF"
            path = OUT / f"{name}.csv" if name == "regen_32.4V_loss0W_dump30mF" else None
            cases.append(profile(name, v, 30e-3, -1800.0, loss, profile_csv=path))
        name = f"regen_{v:g}V_loss0W_nodump30mF"
        path = OUT / "regen32p4_1800W_nodump_nominal.csv" if v == 32.4 else None
        cases.append(profile(name, v, 30e-3, -1800.0, 0.0, safe5_v=0,
                             dump_mode="off", profile_csv=path))
    # Timing and capacity sensitivities at the 32.4 V nominal bus.
    cases.append(profile("regen32p4_1800W_dump_C24mF", 32.4, 24e-3, -1800.0, 0.0))
    cases.append(profile("regen32p4_1800W_dump_C30mF", 32.4, 30e-3, -1800.0, 0.0))
    cases.append(profile("regen32p4_1800W_core5off_SAFE5on", 32.4, 30e-3, -1800.0,
                         0.0, core5_v=0.0))
    pth = OUT / "regen32p4_1800W_safe_loss_100ms.csv"
    cases.append(profile("regen32p4_1800W_safe_loss_100ms", 32.4, 30e-3, -1800.0,
                         0.0, safe5_loss_s=0.1, profile_csv=pth))
    cases.append(profile("regen32p4_1800W_thermal_open_100ms", 32.4, 30e-3, -1800.0,
                         0.0, thermal_open_s=0.1))
    cases.append(profile("regen32p4_1800W_nodump_C24mF", 32.4, 24e-3, -1800.0,
                         0.0, safe5_v=0.0, dump_mode="off"))
    cases.append(profile("regen32p4_absorb1800W_quota1800J", 32.4, 30e-3, -1800.0,
                         0.0, regen_sink_limit_w=1800.0, regen_sink_quota_j=1800.0))
    cases.append(profile("regen32p4_absorb600W_quota300J_then_dump", 32.4, 30e-3,
                         -1800.0, 0.0, regen_sink_limit_w=600.0,
                         regen_sink_quota_j=300.0))
    cases.append(profile("regen32p4_absorb600W_no_quota_then_dump", 32.4, 30e-3,
                         -1800.0, 0.0, regen_sink_limit_w=600.0,
                         regen_sink_quota_j=None))
    # Drive: 1.8kW phase input plus 0/100/200W constant loss bracket.
    for v in (22.5, 32.4, 37.8):
        for loss in (0.0, 100.0, 200.0):
            cases.append(profile(f"drive_{v:g}V_Pac1800_loss{loss:g}W", v, 30e-3,
                                 1800.0, loss, dump_mode="off"))
    # Failure case: drain bank stuck on while a stiff source remains connected.
    for v in (22.5, 32.4, 37.8):
        cases.append(profile(f"dump_stuck_on_source_{v:g}V", v, 30e-3, 0.0,
                             0.0, dump_mode="stuck_on"))
    cases.append(profile("dump_stuck_on_SAFE5off_32p4V", 32.4, 30e-3, 0.0,
                         0.0, safe5_v=0.0, dump_mode="stuck_on"))
    cases.append(profile("dump_stuck_on_thermal_open_32p4V", 32.4, 30e-3, 0.0,
                         0.0, thermal_open_s=0.1, dump_mode="stuck_on"))
    results = {
        "title": "1 s average-value DC-link/regen/chopper screening model",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ngspice_complement": "Python energy-state model; no switching waveforms",
        "ovp_resistor_network": {**NOM, **vth},
        "global_parameters": {
            "dc_link_nominal_f": 30e-3, "dc_link_minus20pct_f": 24e-3,
            "dc_link_esr_nom_ohm": 6.333e-3, "dc_link_esl_initial_assumption_h": 15e-9,
            "dump_resistance_total_ohm": 1.0, "dump_resistors": "2 x LPS300 2ohm in parallel",
            "dump_Ron_Q7_ohm": 11.25e-3,
            "bus_source_reverse_acceptance": "blocked; battery source cannot absorb regeneration",
            "profile_step_s": 10e-6,
            "two_board_scope": "two_board_energy_j is a simple 2x per-board scaling only; shared-bus impedance/current-sharing interaction is not simulated",
        },
        "cases": cases,
    }
    (OUT / "dc_link_1s_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"results": str(OUT / 'dc_link_1s_results.json'),
                      "thresholds": vth, "case_count": len(cases)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
