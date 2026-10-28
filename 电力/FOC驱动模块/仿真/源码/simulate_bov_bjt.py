#!/usr/bin/env python3
"""ngspice check of the BUSOV_N -> BC857C/BC847C -> BREAK_N path.

The two small-signal BJTs use standard Gummel-Poon models with BF set to the
datasheet C-bin minimum. Nexperia's device-specific SPICE downloads are not
embedded; this verifies topology and static logic behavior, not corners.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from simulate_dc_link_1s import HERE, NOM, output_high_and_ref

NGSPICE = HERE / "tools" / "ngspice-47-portable" / "Spice64" / "bin" / "ngspice_con.exe"
OUT = HERE / "结果" / "bov_bjt"


def n(x: float) -> str:
    return f"{x:.12g}"


def make_case(name: str, *, core5_v: float = 5.0, logic3v3_v: float = 3.3,
              bus_profile: str = "normal", negative_break: bool = False,
              v_safe: float = NOM["v_safe_v"], vol_v: float = 0.9,
              cov_sense_f: float = 1e-9) -> tuple[str, dict]:
    vbus = "PWL(0 32.4 2m 32.4 2.01m 45 20m 45)" if bus_profile == "overvoltage" else "32.4"
    rbt, rbb = NOM["r_bus_top_ohm"], NOM["r_bus_bottom_ohm"]
    rrt, rrb = NOM["r_ref_top_ohm"], NOM["r_ref_bottom_ohm"]
    rh, rp = NOM["r_hyst_ohm"], NOM["r_pullup_ohm"]
    out_hi, ref_hi = output_high_and_ref(v_safe, rrt, rrb, rh, rp)
    sense_init = 32.4 * rbb / (rbt + rbb)
    neg_control = "PULSE(0 3.3 1m 1n 1n 3m 20m)" if negative_break else "0"
    logic3v3 = logic3v3_v if core5_v > 4.5 else 0.0
    event_meas = ""
    if bus_profile == "overvoltage":
        event_meas = "meas tran t_busov_1v WHEN v(busov_n)=1 FALL=1\n"
        if logic3v3 > 1.0:
            event_meas += "meas tran t_break_1v WHEN v(break_n)=1 FALL=1\n"
        event_meas += "meas tran t_gate_2p5 WHEN v(brkgate)=2.5 RISE=1\n"
    elif negative_break:
        event_meas = "meas tran t_break_1v WHEN v(break_n)=1 FALL=1\n"
    net = f"""* Discrete BJT BUSOV level shift to BREAK_N: {name}
* Q_BOV_PNP=BC857C; Q_BOV_NPN=BC847C. Generic Gummel-Poon models use
* datasheet hFE bins as typical parameters; not Nexperia production corners.
.title {name}
.options method=gear reltol=0.001 abstol=1n vntol=1u maxord=2
.param VSAFE={n(v_safe)} VOL={n(vol_v)} COV={n(cov_sense_f)}

* SAFE rail current is sensed at the source; the Bdrv behavioral source is
* ideal and its UCC gate-drive current is intentionally not included.
Vbus bus 0 {vbus}
Vsafe_src safe5_src 0 {{VSAFE}}
Vsafe_sense safe5_src safe5 0
Vcore5 core5 0 {n(core5_v)}
Vlogic mcu3v3 0 {n(logic3v3)}

* Candidate resistor network and sense filter, with comparator hysteresis
* represented by an open-collector behavior source.
Rovtop bus ov_sense 162k
Rovbot ov_sense 0 10k
Covsense ov_sense 0 {{COV}}
Rref_top safe5 ov_ref 10.2k
Rref_bot ov_ref 0 10k
Rhyst busov_n ov_ref 200k
Rov_pullup safe5 busov_n 4.7k
Boc busov_n 0 I={{ (v(ov_sense)>v(ov_ref)) ? (v(busov_n)-VOL)/5 : 0 }}
Ccmp busov_n 0 100p

* Actual candidate level shift topology: 10k from BUSOV_N to PNP base;
* PNP emitter at SAFE_5V, collector feeds NPN base through 4.7k; 10k
* pull-down; NPN
* collector pulls BREAK_N low. Other open-collector faults can also pull
* BREAK_N down but do not connect to the UCC27511 input.
V_pnp_base_sense bov_pnp_base bov_base_r 0
R_bov_pnp_b bov_base_r busov_n 10k
Q_BOV_PNP bov_bk_src bov_pnp_base safe5 QBC857C
V_pnp_col_sense bov_bk_src bov_col_probe 0
R_BOV_DRIVE bov_col_probe bov_bk_drive 4.7k
R_bov_npn_pd bov_bk_drive 0 10k
V_npn_base_sense bov_bk_drive bov_npn_base 0
Q_BOV_NPN break_n bov_npn_base 0 QBC847C
Rbreak_pull mcu3v3 break_n 2.2k
Cbreak break_n 0 50p

* Independent open-collector current-fault comparator input for the
* negativeBreak-alone case. It has no route to the chopper enable.
Vneg_ctrl neg_ctrl 0 {neg_control}
Snegative_break break_n 0 neg_ctrl 0 SW_OC
.model SW_OC SW(Ron=10 Roff=1G Vt=1.5 Vh=0.1)

* SAFE-powered UCC input/gate and 1-ohm dump bank. BREAK_N is logged but is
* intentionally not the dump command. Q7 is a simple Ron switch model here.
Bdrv drv_cmd 0 V={{ ((v(safe5)>4.5) && (v(busov_n)<1.0)) ? 5 : 0 }}
Rgate drv_cmd brkgate 10
Cgate brkgate 0 10n IC=0
Rgate_pd brkgate 0 10k
RdumpA bus dump_bus 2
RdumpB bus dump_bus 2
Ldump dump_bus dump_sw 10n
Sbrake dump_sw 0 brkgate 0 SW_Q7
.model SW_Q7 SW(Ron=11.25m Roff=1G Vt=2.5 Vh=0.1)

.model QBC857C PNP(IS=1e-14 BF=420 VAF=70)
.model QBC847C NPN(IS=1e-14 BF=420 VAF=80)

.save V(bus) V(safe5) V(core5) V(mcu3v3) V(ov_sense) V(ov_ref) V(busov_n) V(bov_pnp_base) V(bov_bk_src) V(bov_bk_drive) V(break_n) V(neg_ctrl) V(drv_cmd) V(brkgate) V(dump_bus) I(Ldump) I(V_pnp_col_sense) I(V_pnp_base_sense) I(V_npn_base_sense) I(Vsafe_sense)

.control
set noaskquit
set numdgt=10
tran 20n 4m 0 20n
let ipnp_c = i(V_pnp_col_sense)
let ipnp_b = i(V_pnp_base_sense)
let inpn_b = i(V_npn_base_sense)
let isafe = i(Vsafe_sense)
let abs_ipnp_c = abs(ipnp_c)
let abs_ipnp_b = abs(ipnp_b)
let abs_inpn_b = abs(inpn_b)
let abs_isafe = abs(isafe)
let pnp_power = (v(safe5)-v(bov_bk_src))*ipnp_c + (v(safe5)-v(bov_pnp_base))*ipnp_b
let safe_power = v(safe5)*isafe
let idump = i(Ldump)
let pdump = v(bus,dump_bus)*idump
let abs_idump = abs(idump)
meas tran t_end max time from=0 to=4m
meas tran busov_n_min min v(busov_n) from=0 to=4m
meas tran busov_n_max max v(busov_n) from=0 to=4m
meas tran break_n_min min v(break_n) from=0 to=4m
meas tran break_n_max max v(break_n) from=0 to=4m
meas tran bov_drive_max max v(bov_bk_drive) from=0 to=4m
meas tran brkgate_max max v(brkgate) from=0 to=4m
meas tran dump_current_max max abs_idump from=0 to=4m
meas tran dump_resistor_energy integ pdump from=0 to=4m
meas tran pnp_col_current_peak max abs_ipnp_c from=0 to=4m
meas tran pnp_col_current_avg avg abs_ipnp_c from=0 to=4m
meas tran pnp_base_current_peak max abs_ipnp_b from=0 to=4m
meas tran pnp_base_current_avg avg abs_ipnp_b from=0 to=4m
meas tran pnp_power_peak max pnp_power from=0 to=4m
meas tran pnp_power_avg avg pnp_power from=0 to=4m
meas tran pnp_energy integ pnp_power from=0 to=4m
meas tran npn_base_current_peak max abs_inpn_b from=0 to=4m
meas tran npn_base_current_avg avg abs_inpn_b from=0 to=4m
meas tran safe_current_peak max abs_isafe from=0 to=4m
meas tran safe_current_avg avg abs_isafe from=0 to=4m
meas tran safe_power_peak max safe_power from=0 to=4m
meas tran safe_energy integ safe_power from=0 to=4m
meas tran bov_drive_at3p5ms find v(bov_bk_drive) at=3.5m
meas tran busov_n_at3p5ms find v(busov_n) at=3.5m
meas tran break_n_at3p5ms find v(break_n) at=3.5m
meas tran brkgate_at3p5ms find v(brkgate) at=3.5m
{event_meas}write {name}.raw V(bus) V(ov_sense) V(ov_ref) V(busov_n) V(bov_pnp_base) V(bov_bk_drive) V(break_n) V(neg_ctrl) V(brkgate) I(Ldump)
quit
.endc
.end
"""
    metadata = {
        "name": name,
        "case": {"core5_v": core5_v, "logic3v3_v": logic3v3,
                 "bus_profile": bus_profile, "negative_break_fault": negative_break,
                 "safe5_v": v_safe, "vol_max_v": vol_v, "c_ov_sense_f": cov_sense_f},
        "candidate_topology": "BC857C PNP E=SAFE_5V, B through 10k from BUSOV_N, C drives BC847C base through R_BOV_DRIVE=4.7k; BC847C NPN E=GND, C=BREAK_N; 10k base pulldown; 2.2k BREAK_N pullup to MCU_3V3",
        "bjt_model": "standard Gummel-Poon models with BF=420 (datasheet BC847C/BC857C C-group minimum hFE at the specified 2mA test point); not manufacturer device-specific SPICE files, and low-current/temperature behavior is not guaranteed",
        "comparator_model": "actual resistor ratios/hysteresis and C_OV_SENSE, behavioral open-collector comparator with VOL corner; no TLV1704 macromodel, propagation delay/noise/overdrive omitted",
        "chopper_gate_logic": "BUSOV_N < 1.0V and SAFE_5V present; BREAK_N is separately logged and does not enable Q7",
        "q7": "SW Ron=11.25mOhm behavior model from 4.5mOhm max at VGS=4.5V times 2.5 hot factor; no IRLB manufacturer macro",
        "supply_measurement_scope": "SAFE rail current includes modeled OVP reference/comparator and BJT path; Bdrv is an ideal behavioral voltage source, so UCC27511 gate-charge/quiescent current and its 10k gate pulldown are not included",
        "scope": "topology/polarity and BJT current-limiting check only; 4ms short transient; not 1s thermal or board validation",
    }
    return net, metadata


def parse(log: str) -> dict:
    out = {}
    for key, value in re.findall(r"(?im)^\s*(\w+)\s*=\s*([-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?)", log):
        out[key.lower()] = float(value)
    return out


def run(name: str, **kwargs) -> dict:
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cir_text, meta = make_case(name, **kwargs)
    cir = run_dir / "model.cir"
    cir.write_text(cir_text, encoding="ascii")
    (run_dir / "parameters.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_file = run_dir / "ngspice.log"
    proc = subprocess.run([str(NGSPICE), "-b", "-o", str(log_file), str(cir)],
                          cwd=run_dir, capture_output=True, text=True, timeout=300)
    log = log_file.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0 or "ngspice-47 done" not in log or re.search(r"(?im)^\s*Error:", log):
        raise RuntimeError(f"ngspice failure {name}: {proc.returncode}\n{log[-4000:]}\n{proc.stderr[-2000:]}")
    measures = parse(log)
    if abs(measures.get("t_end", 0) - 4e-3) > 2e-8:
        raise RuntimeError(f"{name} did not reach 4ms: {measures}\n{log[-2000:]}")
    return {"parameters": meta, "measures": measures,
            "ngspice_version": "47",
            "netlist_sha256": hashlib.sha256(cir_text.encode("ascii")).hexdigest(),
            "netlist_file": str(cir.relative_to(HERE)),
            "log_file": str(log_file.relative_to(HERE)),
            "raw_file": str((run_dir / f"{name}.raw").relative_to(HERE))}


def run_prefixed_direct_collector_op() -> dict:
    """One static pre-fix OP showing why the PNP collector needed a resistor."""
    name = "pre_fix_direct_collector_45V_op"
    run_dir = OUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cir_text = f"""* Pre-fix diagnostic only: PNP collector directly drives NPN base.
* SAFE current and BJT branch current are measured; this is not the candidate.
.title {name}
.options method=gear reltol=0.001 abstol=1n vntol=1u
.param VSAFE={n(NOM['v_safe_v'])}
Vbus bus 0 45
Vsafe_src safe5_src 0 {{VSAFE}}
Vsafe_sense safe5_src safe5 0
Vlogic mcu3v3 0 3.3
Vbusov busov_n 0 0.9
Rov_pullup safe5 busov_n 4.7k
Rref_top safe5 ov_ref 10.2k
Rref_bot ov_ref 0 10k
V_pnp_base_sense bov_pnp_base bov_base_r 0
R_bov_pnp_b bov_base_r busov_n 10k
Q_BOV_PNP bov_bk_src bov_pnp_base safe5 QBC857C
V_pnp_col_sense bov_bk_src bov_bk_drive 0
R_bov_npn_pd bov_bk_drive 0 10k
V_npn_base_sense bov_bk_drive bov_npn_base 0
Q_BOV_NPN break_n bov_npn_base 0 QBC847C
Rbreak_pull mcu3v3 break_n 2.2k
.model QBC857C PNP(IS=1e-14 BF=420 VAF=70)
.model QBC847C NPN(IS=1e-14 BF=420 VAF=80)
.control
set noaskquit
set numdgt=12
op
let ipnp_c = i(V_pnp_col_sense)
let ipnp_b = i(V_pnp_base_sense)
let inpn_b = i(V_npn_base_sense)
let isafe = i(Vsafe_sense)
let pnp_power = (v(safe5)-v(bov_bk_src))*ipnp_c + (v(safe5)-v(bov_pnp_base))*ipnp_b
let safe_power = v(safe5)*isafe
print v(busov_n) v(bov_pnp_base) v(bov_bk_drive) v(break_n) ipnp_c ipnp_b inpn_b isafe pnp_power safe_power
quit
.endc
.end
"""
    cir = run_dir / "model.cir"
    cir.write_text(cir_text, encoding="ascii")
    log_file = run_dir / "ngspice.log"
    proc = subprocess.run([str(NGSPICE), "-b", "-o", str(log_file), str(cir)],
                          cwd=run_dir, capture_output=True, text=True, timeout=60)
    log = log_file.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0 or "ngspice-47 done" not in log or re.search(r"(?im)^\s*Error:", log):
        raise RuntimeError(f"ngspice pre-fix OP failed: {proc.returncode}\n{log[-4000:]}\n{proc.stderr[-2000:]}")
    measures = parse(log)
    required = ("ipnp_c", "ipnp_b", "inpn_b", "isafe", "pnp_power", "safe_power")
    if any(k not in measures for k in required):
        raise RuntimeError(f"Could not parse pre-fix OP currents/power: {measures}\n{log[-3000:]}")
    return {"topology": "pre-fix: PNP collector directly tied to NPN base; no R_BOV_DRIVE",
            "model_limit": "one .op with generic BF=420 Gummel-Poon devices; diagnostic only, not all-corner device evidence",
            "measures": measures,
            "netlist_sha256": hashlib.sha256(cir_text.encode("ascii")).hexdigest(),
            "netlist_file": str(cir.relative_to(HERE)),
            "log_file": str(log_file.relative_to(HERE))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pre_fix = run_prefixed_direct_collector_op()
    cases = [
        run("normal_32p4V_SAFEon"),
        run("overvoltage_45V_SAFEon_VOL0p9_COV1n", bus_profile="overvoltage", vol_v=0.9),
        run("overvoltage_45V_SAFEon_VOL0_COV1n", bus_profile="overvoltage", vol_v=0.0),
        run("overvoltage_45V_SAFEon_noCOV", bus_profile="overvoltage", vol_v=0.9, cov_sense_f=1e-15),
        run("overvoltage_45V_core5off_SAFEon", bus_profile="overvoltage", core5_v=0.0, logic3v3_v=0.0),
        run("negativeBreak_only_32p4V", negative_break=True),
    ]
    result = {"title": "Discrete BJT OVP-to-BREAK_N path: 4.7k current-limited topology check",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "pre_fix_direct_collector_op": pre_fix,
              "cases": cases}
    (OUT / "bov_bjt_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(OUT / "bov_bjt_results.json"),
                      "pre_fix_direct_collector_op": pre_fix,
                      "cases": [{"name": c["parameters"]["name"], **c["measures"]} for c in cases]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
