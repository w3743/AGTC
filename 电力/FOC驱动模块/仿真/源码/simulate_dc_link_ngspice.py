#!/usr/bin/env python3
"""Run a 1 s ngspice DC-link/chopper network cross-check.

This isolates the average bus from the fast motor bridge so that the long
energy event can be simulated and audited without claiming FOC validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from simulate_dc_link_1s import HERE, NOM, output_high_and_ref, ref_node

NGSPICE = HERE / "tools" / "ngspice-47-portable" / "Spice64" / "bin" / "ngspice_con.exe"
OUT = HERE / "结果" / "dc_link_1s" / "ngspice"


def n(x: float) -> str:
    return f"{x:.12g}"


def make_case(name: str, vdc: float = 32.4, safe5: float = NOM["v_safe_v"],
              core5: float = 5.0, cov_sense: float = 1e-9,
              safe5_loss_s: float | None = None, regen_sink_w: float = 0.0,
              sink_quota_j: float = 0.0, vol_v: float = 0.9,
              dump_enabled: bool = True) -> tuple[str, dict]:
    rbt = NOM["r_bus_top_ohm"]
    rbb = NOM["r_bus_bottom_ohm"]
    rrt = NOM["r_ref_top_ohm"]
    rrb = NOM["r_ref_bottom_ohm"]
    rh = NOM["r_hyst_ohm"]
    rp = NOM["r_pullup_ohm"]
    if safe5_loss_s is None:
        safe_src = f"Vsafe safe5 0 {{{n(safe5)}}}"
    else:
        # Smoothstep loss avoids an ideal-source slope discontinuity at the
        # exact outage instant, which otherwise drives Gear to a vanishing
        # timestep in ngspice. This is still a forced SAFE_5V brownout, not a
        # regulator transient model.
        t0 = max(0.0, safe5_loss_s - 100e-6)
        t1 = safe5_loss_s + 900e-6
        safe_src = (f"Bsafe safe5 0 V={{ VSAFE*(time<{n(t0)} ? 1 : "
                    f"(time<{n(t1)} ? 1-3*pow((time-{n(t0)})/{n(t1-t0)},2)+"
                    f"2*pow((time-{n(t0)})/{n(t1-t0)},3) : 0)) }}")
    if safe5 > 4.5:
        busov_hi, ref_hi = output_high_and_ref(safe5, rrt, rrb, rh, rp)
    else:
        busov_hi, ref_hi = 0.0, 0.0
    sense_init = vdc * rbb / (rbt + rbb)
    # Account for the 1 ms linear power ramp when expiring a finite absorber
    # quota. Before source power reaches PABS its energy is triangular; after
    # the crossing it is constant PABS. This avoids a false abrupt turn-off at
    # t=1.0 s for an exact 1.8 kJ quota.
    t_ramp = 1e-3
    p_gen = 1800.0
    if regen_sink_w > 0 and sink_quota_j > 0:
        p_abs = min(regen_sink_w, p_gen)
        t_cross = p_abs * t_ramp / p_gen
        e_before_cross = 0.5 * p_abs * t_cross
        if sink_quota_j <= e_before_cross:
            t_abs = math.sqrt(2.0 * sink_quota_j * t_ramp / p_gen)
        else:
            t_abs = t_cross + (sink_quota_j - e_before_cross) / p_abs
    else:
        t_abs = 0.0
    extra_measures = ""
    if safe5 > 4.5 and regen_sink_w < 1800.0:
        extra_measures += "meas tran t_busov_1v WHEN v(busov_n)=1 FALL=1\n"
        if dump_enabled:
            extra_measures += "meas tran t_gate_2p5 WHEN v(brkgate)=2.5 RISE=1\n"
    if not dump_enabled or safe5 <= 4.5 or safe5_loss_s is not None:
        extra_measures += """meas tran t75 WHEN v(bus)=75 RISE=1
meas tran t100 WHEN v(bus)=100 RISE=1
"""
    net = f"""* Independent 1 s DC-link energy/crowbar cross-check: {name}
* Average source/injection model; no phase bridge or vendor MOS macro.
.title {name}
.options method=gear reltol=0.001 abstol=1u vntol=1u maxord=2
.param VDC={n(vdc)} VSAFE={n(safe5)} VOL={n(vol_v)} PGEN=1800 COV={n(cov_sense)} PABS={n(regen_sink_w)} TABS={n(t_abs)} TRAMP=1m DUMPEN={1 if dump_enabled else 0}

* Precharged bus. Battery source has a one-way diode and 10mohm/1uH path;
* it may source motoring current but cannot absorb regenerative current.
Vbattery batt 0 {{VDC}}
Rsource batt src1 10m
Lsource src1 src2 1u IC=0
Dsrc src2 bus D_SOURCE
.model D_SOURCE D(IS=1e-9 N=1.2 RS=2m CJO=500p TT=200n)

* Local bulk C, ESR and ESL, isolated from the source/bridge by the modeled
* connection inductance. No remote DC/DC or distant capacitance is included.
Llink bus cap_esr 15n IC=0
Resr cap_esr cap_i 6.333m
Cbulk cap_i 0 30m IC={{VDC}}

* Motor port is an explicit 1.8kW constant-power injection into the BUS.
* It intentionally continues for the full second as chopper thermal upper
* bound; actual BREAK_N current decay and diode regeneration are not modeled.
Bregen 0 bus I={{ ((time<TRAMP) ? PGEN*time/TRAMP : PGEN)/max(v(bus),1) }}
Bregen_power regen_power 0 V={{ ((time<TRAMP) ? PGEN*time/TRAMP : PGEN) }}
* Optional external absorption envelope (not present in the no-sink baseline).
* It is power-limited and expires after the requested finite energy quota.
Babs bus 0 I={{ (time<TABS) ? min(PABS, ((time<TRAMP) ? PGEN*time/TRAMP : PGEN))/max(v(bus),1) : 0 }}
Babs_power abs_power 0 V={{ (time<TABS) ? min(PABS, ((time<TRAMP) ? PGEN*time/TRAMP : PGEN)) : 0 }}

* Candidate TLV1704 resistor network. The COV option is the proposed 1nF
* VBUS_OV_SENSE filter; compare with 1fF for the unfiltered sensitivity.
Rovtop bus ov_sense 162k
Rovbot ov_sense 0 10k
Covsense ov_sense 0 {{COV}} IC={n(sense_init)}
Rref_top safe5 ov_ref 10.2k
Rref_bot ov_ref 0 10k
Rhyst busov_n ov_ref 200k
Rov_pullup safe5 busov_n 4.7k
* Open-collector comparator behavior at the full-temperature VOL=0.9V
* datasheet boundary; no official TLV1704 macromodel or worst-case delay.
Boc busov_n 0 I={{ (v(ov_sense)>v(ov_ref)) ? (v(busov_n)-VOL)/5 : 0 }}
Ccmp busov_n 0 100p IC={n(busov_hi)}

* SAFE_5V independent from controller 5V; UCC gate RC and IRLB4030 dump
* switch represented behaviorally. The two 2ohm resistors make 1ohm total.
{safe_src}
Vcore5 core5 0 {{{n(core5)}}}
Bdrv drv_cmd 0 V={{ ((DUMPEN>0.5) && (v(safe5)>4.5) && (v(busov_n)<1.0)) ? 5 : 0 }}
Rgate drv_cmd brkgate 10
Cgate brkgate 0 10n IC=0
Rgate_pd brkgate 0 10k
RdumpA bus dump_bus 2
RdumpB bus dump_bus 2
Ldump dump_bus dump_sw 10n
Sbrake dump_sw 0 brkgate 0 SW_DUMP
.model SW_DUMP SW(Ron=11.25m Roff=1G Vt=2.5 Vh=0.1)

.ic V(bus)={{VDC}} V(cap_i)={{VDC}} V(ov_sense)={n(sense_init)} V(ov_ref)={n(ref_hi)} V(busov_n)={n(busov_hi)} V(brkgate)=0
.save V(bus) V(batt) V(cap_i) V(ov_sense) V(ov_ref) V(busov_n) V(brkgate) V(dump_bus) V(regen_power) V(abs_power) I(Llink) I(Ldump) I(Vbattery)

.control
set noaskquit
set numdgt=10
tran 2u 1s 0 2u uic
let preg = v(regen_power)
let idump = i(Ldump)
let pdump_r = v(bus,dump_bus)*idump
let pq7 = idump*idump*11.25m
let pdump = pdump_r+pq7
let pabs = v(abs_power)
let isource = -i(Vbattery)
let psource = v(batt)*isource
meas tran t_end max time from=0 to=1
meas tran vbus_min min v(bus) from=0 to=1
meas tran vbus_max max v(bus) from=0 to=1
meas tran vbus_end find v(bus) at=1
meas tran regen_energy integ preg from=0 to=1
meas tran dump_energy integ pdump from=0 to=1
meas tran dump_resistor_energy integ pdump_r from=0 to=1
meas tran q7_channel_energy integ pq7 from=0 to=1
meas tran absorb_energy integ pabs from=0 to=1
meas tran battery_energy integ psource from=0 to=1
meas tran source_current_max max isource from=0 to=1
meas tran source_current_min min isource from=0 to=1
{extra_measures}write {name}.raw V(bus) V(cap_i) V(ov_sense) V(ov_ref) V(busov_n) V(brkgate) V(dump_bus) V(regen_power) V(abs_power) I(Llink) I(Ldump) I(Vbattery)
quit
.endc
.end
"""
    metadata = {
        "name": name, "vdc_v": vdc, "safe5_v": safe5, "core5_v": core5,
        "dump_enabled": dump_enabled,
        "c_ov_sense_f": cov_sense, "safe5_loss_s": safe5_loss_s,
        "pac_motor_port_w_injected": 1800.0,
        "source": "ideal 1800W regenerative injection; full 1s even after OVP/BREAK_N as a conservative dump-energy screen",
        "external_absorption_limit_w": regen_sink_w,
        "external_absorption_quota_j": sink_quota_j,
        "external_absorption_duration_s": t_abs,
        "battery_source_model": "one-way diode, 10mohm + 1uH, cannot absorb regen",
        "bulk": {"C_F": 30e-3, "ESR_ohm": 6.333e-3, "ESL_H": 15e-9, "precharged": True},
        "chopper": "2x2ohm parallel, 11.25mohm Q7 channel (4.5mOhm max at VGS=4.5V times hot factor 2.5) with separate resistor-bank and Q7 energy integrals, 10ohm/10nF gate, hysteretic comparator behavior model",
        "regen_ramp": "linear 0-to-1800W over 1ms avoids an unphysical step through bulk ESL; full requested energy still approximately 1.8kJ over 1s",
        "comparator": f"TLV1704 threshold network; VOL={vol_v:g}V behavior boundary; no official macromodel or worst-case delay; measured crossing times are from behavior-level comparator, without datasheet propagation delay",
        "netlist_scope": "independent average 1s DC-link network, not a full inverter/FOC test",
    }
    return net, metadata


def parse(log: str) -> dict:
    out = {}
    for key, value in re.findall(r"(?im)^\s*(\w+)\s*=\s*([-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?)", log):
        out[key.lower()] = float(value)
    return out


def run(name: str, keep_raw: bool = True, **args) -> dict:
    if not NGSPICE.exists():
        raise FileNotFoundError(NGSPICE)
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cir_text, meta = make_case(name, **args)
    cir = run_dir / "model.cir"
    cir.write_text(cir_text, encoding="ascii")
    (run_dir / "parameters.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log_file = run_dir / "ngspice.log"
    proc = subprocess.run([str(NGSPICE), "-b", "-o", str(log_file), str(cir)],
                          cwd=run_dir, capture_output=True, text=True, timeout=2400)
    log = log_file.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0 or "ngspice-47 done" not in log or re.search(r"(?im)^\s*Error:", log):
        raise RuntimeError(f"ngspice failure {name}: {proc.returncode}\n{log[-4000:]}\n{proc.stderr[-2000:]}")
    measures = parse(log)
    if abs(measures.get("t_end", 0) - 1.0) > 2e-6:
        raise RuntimeError(f"{name} did not reach 1s: {measures}\n{log[-2000:]}")
    # Positive pabs is energy removed from the bus by a hypothetical external
    # absorber; use the bus-interface sign convention (negative when the bus
    # exports energy into that sink). Regen is positive into the modeled bus.
    measures["external_absorber_signed_energy_j"] = -measures.get("absorb_energy", 0.0)
    measures["regen_source_injected_energy_j"] = measures.get("regen_energy", 0.0)
    if not keep_raw:
        raw = run_dir / f"{name}.raw"
        if raw.exists():
            raw.unlink()
    return {"parameters": meta, "ngspice_version": "47", "return_code": proc.returncode,
            "measures": measures, "netlist_sha256": hashlib.sha256(cir_text.encode("ascii")).hexdigest(),
            "log_file": str(log_file.relative_to(HERE)), "raw_file": str((run_dir / f"{name}.raw").relative_to(HERE))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        run("healthy_32p4_SAFEon_COV1n", cov_sense=1e-9),
        run("healthy_32p4_SAFEon_noCOV", keep_raw=False, cov_sense=1e-15),
        run("dumpdisabled_32p4_SAFEon_COV1n", safe5=NOM["v_safe_v"], dump_enabled=False, cov_sense=1e-9),
        run("nodump_32p4_SAFEoff_COV1n", safe5=0.0, cov_sense=1e-9),
        run("core5off_32p4_SAFEon_COV1n", core5=0.0, cov_sense=1e-9),
        run("safe5loss_100ms_32p4_COV1n", keep_raw=False, cov_sense=1e-9, safe5_loss_s=0.1),
        run("regen_sink_1800W_1800J_quota", keep_raw=False, regen_sink_w=1800.0, sink_quota_j=1800.0),
        run("regen_sink_600W_300J_quota_then_dump", regen_sink_w=600.0, sink_quota_j=300.0),
    ]
    out = {"title": "ngspice-47 1s local DC-link/chopper behavior simulation",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "cases": cases}
    (OUT / "ngspice_dc_link_1s_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"result": str(OUT / "ngspice_dc_link_1s_results.json"),
                      "cases": [{"name": c["parameters"]["name"], **c["measures"]} for c in cases]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
