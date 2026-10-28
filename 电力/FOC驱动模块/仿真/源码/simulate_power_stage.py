#!/usr/bin/env python3
"""Generate and run the first ngspice FOC power-stage behavior model.

This is a transparent switch-level screening model, not an IRLB/IPT vendor
MOSFET macro model and not a closed-loop FOC firmware validation.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import re
import subprocess
from pathlib import Path
from simulate_dc_link_1s import NOM as DC_NOM, output_high_and_ref


HERE = Path(__file__).resolve().parents[1]
NGSPICE = HERE / "tools" / "ngspice-47-portable" / "Spice64" / "bin" / "ngspice_con.exe"
RESULTS = HERE / "结果"


def current_for_ac_power(p_w: float, e_phase_pk_v: float, r_phase_ohm: float,
                         limit_a: float) -> tuple[float, float | None]:
    """Solve P=1.5*(Epk*Ipk + R*Ipk^2), return signed Ipk and exact solution."""
    a = r_phase_ohm
    b = e_phase_pk_v
    c = -(2.0 / 3.0) * p_w
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return math.copysign(limit_a, p_w), None
    roots = ((-b + math.sqrt(disc)) / (2.0 * a), (-b - math.sqrt(disc)) / (2.0 * a))
    candidates = [x for x in roots if math.copysign(1.0, x or 1.0) == math.copysign(1.0, p_w)]
    if not candidates:
        return math.copysign(limit_a, p_w), None
    exact = min(candidates, key=abs)
    return max(-limit_a, min(limit_a, exact)), exact


def ng_num(value: float) -> str:
    return f"{value:.12g}"


def build_netlist(case: dict) -> tuple[str, dict]:
    vdc = float(case["vdc_v"])
    fsw = float(case["fsw_hz"])
    rpm = float(case["rpm"])
    pole_pairs = int(case["pole_pairs"])
    kv = float(case["kv_rpm_per_v"])
    ke_scale = float(case["ke_scale"])
    rph = float(case["r_phase_ohm"])
    lph = float(case["l_phase_h"])
    target_pac = float(case["pac_request_w"])
    i_limit = float(case["phase_command_limit_a"])
    e_pk = ke_scale * rpm / (math.sqrt(3.0) * kv)
    f_e = pole_pairs * rpm / 60.0
    w_e = 2.0 * math.pi * f_e
    e_init = (e_pk * math.sin(0.0), e_pk * math.sin(-2.0 * math.pi / 3.0),
              e_pk * math.sin(2.0 * math.pi / 3.0))
    # Center open-circuit phase nodes in the rails at t=0 so the UIC state
    # agrees with body-diode polarity and zero phase-inductor current.
    vcm_init = vdc / 2.0 - (max(e_init) + min(e_init)) / 2.0
    vphase_init = tuple(vcm_init + x for x in e_init)
    i_cmd, i_exact = current_for_ac_power(target_pac, e_pk, rph, i_limit)
    deadtime = float(case["deadtime_s"])
    edge_sum = float(case["tr_tf_s"])
    # A PWL conductance ramp is applied over tr_tf. The complementary edge
    # centers are separated by tr_tf + deadtime, leaving a full requested
    # deadtime after the outgoing channel ramp has completed.
    wedge = fsw * edge_sum
    dt_duty = 2.0 * fsw * (deadtime + edge_sum)
    pmax = abs(target_pac)
    maxstep = float(case["max_step_s"])
    tstop = float(case["tstop_s"])
    tstart = float(case["load_start_s"])
    tload_end = tstart + float(case["load_duration_s"])
    dump_active = bool(case["dump_enabled"])
    vcore = float(case["core5_v"])
    vsafe = float(case["safe5_v"])
    arm_ramp = float(case.get("arm_ramp_s", 2e-6))
    cbulk = float(case["dc_link_c_f"])
    esr = float(case["dc_link_esr_ohm"])
    esl = float(case["dc_link_esl_h"])
    c_hf = float(case["bridge_bypass_c_eff_f"])
    r_hf = float(case["bridge_bypass_esr_ohm"])
    l_hf = float(case["bridge_bypass_esl_h"])
    rsrc = float(case["source_r_ohm"])
    lsrc = float(case["source_l_h"])
    rds = float(case["mos_rds_hot_ohm"])
    coss = float(case["mos_coss_f"])
    dump_gate_c = float(case.get("dump_gate_c_f", 10e-9))
    shunt = float(case["shunt_ohm"])
    ov_sense_c = float(case.get("ov_sense_c_f", 1e-9))
    bus_ff_tau = float(case.get("vbus_feedforward_tau_s", 0.0))
    bus_ff_c = float(case.get("vbus_feedforward_c_f", 1e-6))
    if bus_ff_tau > 0.0:
        bus_ff_r = bus_ff_tau / bus_ff_c
        pwm_bus_node = "bus_ff"
        bus_ff_net = (f"Rbus_ff bus bus_ff {ng_num(bus_ff_r)}\n"
                      f"Cbus_ff bus_ff 0 {ng_num(bus_ff_c)} IC={{VDC}}\n")
        bus_ff_ic = " V(bus_ff)={VDC}"
    else:
        bus_ff_r = None
        pwm_bus_node = "bus"
        bus_ff_net = ""
        bus_ff_ic = ""
    tvs_enabled = bool(case.get("tvs_enabled", False))
    compact_output = bool(case.get("compact_output", False))
    if compact_output:
        save_stmt = (".save V(bus) V(bus_inv) V(batt) V(cap_i) V(hfA_cap) V(dump_bus) V(bus_tvs) V(nstar) "
                     "V(busov_n) V(ov_sense) V(ov_ref) V(brkgate) V(fault) V(swA) V(swB) V(swC) "
                     "V(nsA) V(nsB) V(nsC) V(ghA) V(glA) V(ghB) V(glB) V(ghC) V(glC) "
                     "V(shA_f) V(shB_f) V(shC_f) I(Vsense) I(Vtvs_sense) I(VsenseA) I(VsenseB) I(VsenseC) "
                     "I(Vbus_sense) I(Llink) I(LhfA)")
        raw_vectors = ("V(bus) V(cap_i) V(hfA_cap) V(busov_n) V(ov_sense) V(ov_ref) V(brkgate) V(fault) "
                       "V(swA) V(swB) V(swC) V(ghA) V(glA) V(ghB) V(glB) V(ghC) V(glC) "
                       "V(shA_f) V(shB_f) V(shC_f) I(VsenseA) I(VsenseB) I(VsenseC) I(Llink) I(LhfA) I(Vtvs_sense)")
        filetype_stmt = "set filetype=binary\n"
        trace_stmt = ""
    else:
        save_stmt = (".save V(bus) V(bus_inv) V(batt) V(cap_i) V(hfA_cap) V(dump_bus) V(bus_tvs) V(brkgate) V(fault) V(busov_n) "
                     "V(ov_sense) V(ov_ref) V(shA_f) V(shB_f) V(shC_f) V(swA) V(swB) V(swC) V(nstar) "
                     "V(nsA) V(nsB) V(nsC) V(glA) V(glB) V(glC) V(ghA) V(ghB) V(ghC) V(arm) "
                     "I(VsenseA) I(VsenseB) I(VsenseC) I(Vbus_sense) I(Vsense) I(Vtvs_sense) I(Llink) I(LhfA)")
        raw_vectors = "V(bus) V(fault) I(VsenseA) I(VsenseB) I(VsenseC) I(Vbus_sense)"
        filetype_stmt = ""
        trace_stmt = (f"wrdata {case['name']}_trace.dat time v(arm) v(bus) v(cap_i) v(hfA_cap) i(Llink) i(LhfA) "
                      "v(swA) v(swB) v(swC) v(ghA) v(glA) v(ghB) v(glB) v(ghC) v(glC) "
                      "i(VsenseA) i(VsenseB) i(VsenseC) v(busov_n) v(ov_sense) v(ov_ref) v(brkgate) "
                      "v(fault) v(shA_f) v(shB_f) v(shC_f) v(bus_inv,swA) v(swA,nsA) "
                      "v(bus_inv,swB) v(swB,nsB) v(bus_inv,swC) v(swC,nsC)\n")
    if vsafe > 4.5:
        busov_high, ov_ref_high = output_high_and_ref(
            vsafe, DC_NOM["r_ref_top_ohm"], DC_NOM["r_ref_bottom_ohm"],
            DC_NOM["r_hyst_ohm"], DC_NOM["r_pullup_ohm"])
    else:
        busov_high, ov_ref_high = 0.0, 0.0
    ov_sense_init = vdc * DC_NOM["r_bus_bottom_ohm"] / (
        DC_NOM["r_bus_top_ohm"] + DC_NOM["r_bus_bottom_ohm"])
    tedge_log = ng_num(wedge)
    edge_head = f"\n* PWM comparator edge smoothing uses datasheet tr+tf = {edge_sum*1e9:.3g} ns.\n"
    net = f"""* FOC power stage behavior model: {case['name']}
* Generated by simulate_power_stage.py. No manufacturer MOS macro is used.
* Six switch states, 3 low-side shunts, DC-link C/ESR/ESL, source R/L,
* parameterized sinusoidal motor back-EMF and phase R/L, local OVP/OCP latch,
* and a hysteretic local brake chopper. Fixed-speed EMF is a mechanical energy
* reservoir; the electrical source is one-way and cannot silently absorb regen.
.title {case['name']}
.options method=gear reltol={ng_num(float(case.get('reltol', 0.001)))} abstol=1u vntol=1u maxord=2
.param PI=3.14159265358979 VDC={ng_num(vdc)} FSW={ng_num(fsw)} FE={ng_num(f_e)}
.param EPK={ng_num(e_pk)} IPK={ng_num(i_cmd)} RPH={ng_num(rph)} LPH={ng_num(lph)}
.param RDSON={ng_num(rds)} RSH={ng_num(shunt)} TD={ng_num(deadtime)}
.param WEDGE={tedge_log} DTDU={ng_num(dt_duty)}
.param CLOCAL={ng_num(cbulk)} ESRLOCAL={ng_num(esr)} ESLLOCAL={ng_num(esl)}
.param RSRC={ng_num(rsrc)} LSRC={ng_num(lsrc)}
.param TSTART={ng_num(tstart)} TEND={ng_num(tload_end)}
.param TARM={ng_num(arm_ramp)}
.param RAMP={ng_num(float(case['load_ramp_s']))}
.param VCORE={ng_num(vcore)} VSAFE={ng_num(vsafe)}
.param DUMPEN={1 if dump_active else 0}
.param TVSEN={1 if tvs_enabled else 0} TVS_INTERCEPT=49.1 TVSRD=0.644
.param RDUMP=2

* Battery/source Thevenin R-L. Dsrc blocks reverse/regen acceptance in this
* first baseline profile. Source current is explicitly measured at Vsense.
Vbattery batt 0 {{VDC}}
Vsense batt src1 0
Rsource src1 src2 {{RSRC}}
Lsource src2 src3 {{LSRC}} IC=0
Dsrc src3 bus D_SOURCE
.model D_SOURCE D(IS=1e-9 N=1.2 RS=2m CJO=500p TT=200n)

* Local DC-link bank, not remote module capacitance. ESR/ESL are lumped.
Llink bus cap_esr {{ESLLOCAL}} IC=0
Resr cap_esr cap_i {{ESRLOCAL}}
Cbulk cap_i 0 {{CLOCAL}} IC={{VDC}}
* Three phase-leg near-bridge capacitors: each branch models one biased
* 4.7uF (2.35uF effective) plus one 100nF ceramic, including ESR/ESL.
LhfA bus hfA_esr {ng_num(l_hf)} IC=0
RhfA hfA_esr hfA_cap {ng_num(r_hf)}
ChfA hfA_cap 0 {ng_num(c_hf)} IC={{VDC}}
LhfB bus hfB_esr {ng_num(l_hf)} IC=0
RhfB hfB_esr hfB_cap {ng_num(r_hf)}
ChfB hfB_cap 0 {ng_num(c_hf)} IC={{VDC}}
LhfC bus hfC_esr {ng_num(l_hf)} IC=0
RhfC hfC_esr hfC_cap {ng_num(r_hf)}
ChfC hfC_cap 0 {ng_num(c_hf)} IC={{VDC}}
{bus_ff_net}
* Optional linear high-current SMCJ40A clamp proxy from ST DS1284 Rev 11 Table 2:
* TVS_INTERCEPT=49.1 V is the upward-rounded 49.044 V = 64.5 V - 24 A * 0.644 ohm; it is NOT VBR(max).
* Datasheet VBR(max)=46.7 V at 25 C, IT=1 mA; low-current and temperature behavior are not modeled.
Vtvs_sense bus bus_tvs 0
Btvs bus_tvs 0 I={{ (TVSEN>0.5 && v(bus_tvs)>TVS_INTERCEPT) ? (v(bus_tvs)-TVS_INTERCEPT)/TVSRD : 0 }}

* 0-V measurement source separates inverter DC power from source/cable power.
Vbus_sense bus bus_inv 0

* Local brake chopper: resistor-programmed open-collector comparator, an
* independent SAFE_5V rail, gate driver, Q7 and 2x 2-ohm pulse resistors.
Vcore5 core5 0 {{VCORE}}
Vsafe5 safe5 0 {{VSAFE}}
Rovtop bus ov_sense 162k
Rovbot ov_sense 0 10k
Rref_top safe5 ov_ref 10.2k
Rref_bot ov_ref 0 10k
Rhyst busov_n ov_ref 200k
Rov_pullup safe5 busov_n 4.7k
* Full-temperature VOL boundary; behavior-level comparator has no propagation delay.
Boc busov_n 0 I={{ (v(ov_sense)>v(ov_ref)) ? (v(busov_n)-0.9)/5 : 0 }}
Ccmp busov_n 0 100p IC={ng_num(busov_high)}
Covsense ov_sense 0 {ng_num(ov_sense_c)}
Bdrv drv_cmd 0 V={{ ((DUMPEN>0.5) && (v(safe5)>4.5) && (v(busov_n)<1.0)) ? 5 : 0 }}
Rgate drv_cmd brkgate 10
Cgate brkgate 0 {ng_num(dump_gate_c)}
Rgate_pd brkgate 0 10k
RdumpA bus dump_bus {{RDUMP}}
RdumpB bus dump_bus {{RDUMP}}
Ldump dump_bus dump_sw 10n
Sbrake dump_sw 0 brkgate 0 SW_DUMP
.model SW_DUMP SW(Ron=11.25m Roff=1G Vt=2.5 Vh=0.1)

* PWM carrier and sinusoidal references. Min-max zero-sequence injection is
* space-vector modulation in the linear range; no rotor estimator or firmware.
Btri tri 0 V={{0.5+asin(sin(2*pi*FSW*time))/pi}}
Bamp amp 0 V={{ ((time<TSTART) || (time>TEND)) ? 0 : (((time<(TSTART+RAMP)) ? IPK*(time-TSTART)/RAMP : ((time>(TEND-RAMP)) ? IPK*(TEND-time)/RAMP : IPK))) }}
Btheta theta 0 V={{2*pi*FE*time}}
BvA vA 0 V={{EPK*sin(v(theta)) + RPH*v(amp)*sin(v(theta)) + LPH*(2*pi*FE)*v(amp)*cos(v(theta))}}
BvB vB 0 V={{EPK*sin(v(theta)-2*pi/3) + RPH*v(amp)*sin(v(theta)-2*pi/3) + LPH*(2*pi*FE)*v(amp)*cos(v(theta)-2*pi/3)}}
BvC vC 0 V={{EPK*sin(v(theta)+2*pi/3) + RPH*v(amp)*sin(v(theta)+2*pi/3) + LPH*(2*pi*FE)*v(amp)*cos(v(theta)+2*pi/3)}}
BvMax vMax 0 V={{ (v(vA)>v(vB)) ? ((v(vA)>v(vC)) ? v(vA) : v(vC)) : ((v(vB)>v(vC)) ? v(vB) : v(vC)) }}
BvMin vMin 0 V={{ (v(vA)<v(vB)) ? ((v(vA)<v(vC)) ? v(vA) : v(vC)) : ((v(vB)<v(vC)) ? v(vB) : v(vC)) }}
BvCM vCM 0 V={{-0.5*(v(vMax)+v(vMin))}}
Barm arm 0 V={{ (time<TSTART) ? 0 : ((time>TSTART+TARM) ? 1 : (time-TSTART)/TARM) }}
BrawDA rawDA 0 V={{0.5+(v(vA)+v(vCM))/((v({pwm_bus_node})>5)?v({pwm_bus_node}):5)}}
BrawDB rawDB 0 V={{0.5+(v(vB)+v(vCM))/((v({pwm_bus_node})>5)?v({pwm_bus_node}):5)}}
BrawDC rawDC 0 V={{0.5+(v(vC)+v(vCM))/((v({pwm_bus_node})>5)?v({pwm_bus_node}):5)}}
BdutyA dutyA 0 V={{(v(rawDA)<0.02)?0.02:((v(rawDA)>0.98)?0.98:v(rawDA))}}
BdutyB dutyB 0 V={{(v(rawDB)<0.02)?0.02:((v(rawDB)>0.98)?0.98:v(rawDB))}}
BdutyC dutyC 0 V={{(v(rawDC)<0.02)?0.02:((v(rawDC)>0.98)?0.98:v(rawDC))}}

* Candidate low-side shunt trip +/-75mV (= +/-150A at 0.5mohm), after a
* simplified 0.5us first-order CSA/filter response. Shunt voltage is only
* present when current physically flows through the shunt, including diode.
BshA shA_src 0 V={{v(nsA)}}
RshfA shA_src shA_f 10k
CshfA shA_f 0 50p IC=0
BshB shB_src 0 V={{v(nsB)}}
RshfB shB_src shB_f 10k
CshfB shB_f 0 50p IC=0
BshC shC_src 0 V={{v(nsC)}}
RshfC shC_src shC_f 10k
CshfC shC_f 0 50p IC=0
Bfault_set fault 0 I={{ ((abs(v(shA_f))>75m) || (abs(v(shB_f))>75m) || (abs(v(shC_f))>75m)) ? -1m : 0 }}
Bfault_ov fault 0 I={{ ((v(safe5)>4.5) && (v(busov_n)<1.0)) ? -1m : 0 }}
Cfault fault 0 1n IC=0
Rfault fault 0 100G
Bfault_clamp fault 0 I={{ (v(fault)>3.3) ? (v(fault)-3.3)/1 : 0 }}

* Each MOSFET is a datasheet-parameterized behavioral channel (hot RDS(on)),
* plus a built-in diode and fixed Coss; no vendor MOS macro is used.
* PWL body diodes (0.7V + 20mohm) avoid an unqualified vendor diode macro.

* Phase A half bridge and 0.5-milliohm low-side shunt.
BghA ghA 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyA)-WEDGE)?1:((v(tri)>v(dutyA)+WEDGE)?0:(v(dutyA)+WEDGE-v(tri))/(2*WEDGE))) : 0) }}
BglA glA 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyA)+DTDU-WEDGE)?0:((v(tri)>v(dutyA)+DTDU+WEDGE)?1:(v(tri)-v(dutyA)-DTDU+WEDGE)/(2*WEDGE))) : 0) }}
BchHA bus_inv swA I={{v(bus_inv,swA)*((((v(ghA)<0)?0:((v(ghA)>10)?10:v(ghA)))/10)/RDSON+(1-((v(ghA)<0)?0:((v(ghA)>10)?10:v(ghA)))/10)/1Meg)}}
BchLA swA nsA I={{v(swA,nsA)*((((v(glA)<0)?0:((v(glA)>10)?10:v(glA)))/10)/RDSON+(1-((v(glA)<0)?0:((v(glA)>10)?10:v(glA)))/10)/1Meg)}}
Bdha swA bus_inv I={{(v(swA,bus_inv)>0.7)?(v(swA,bus_inv)-0.7)/20m:0}}
Bdla nsA swA I={{(v(nsA,swA)>0.7)?(v(nsA,swA)-0.7)/20m:0}}
CossHA bus_inv swA {ng_num(coss)} IC={ng_num(vdc-vphase_init[0])}
CossLA swA nsA {ng_num(coss)} IC={ng_num(vphase_init[0])}
RshA nsA 0 {{RSH}}

* Phase B half bridge and low-side shunt.
BghB ghB 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyB)-WEDGE)?1:((v(tri)>v(dutyB)+WEDGE)?0:(v(dutyB)+WEDGE-v(tri))/(2*WEDGE))) : 0) }}
BglB glB 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyB)+DTDU-WEDGE)?0:((v(tri)>v(dutyB)+DTDU+WEDGE)?1:(v(tri)-v(dutyB)-DTDU+WEDGE)/(2*WEDGE))) : 0) }}
BchHB bus_inv swB I={{v(bus_inv,swB)*((((v(ghB)<0)?0:((v(ghB)>10)?10:v(ghB)))/10)/RDSON+(1-((v(ghB)<0)?0:((v(ghB)>10)?10:v(ghB)))/10)/1Meg)}}
BchLB swB nsB I={{v(swB,nsB)*((((v(glB)<0)?0:((v(glB)>10)?10:v(glB)))/10)/RDSON+(1-((v(glB)<0)?0:((v(glB)>10)?10:v(glB)))/10)/1Meg)}}
BdhB swB bus_inv I={{(v(swB,bus_inv)>0.7)?(v(swB,bus_inv)-0.7)/20m:0}}
BdlB nsB swB I={{(v(nsB,swB)>0.7)?(v(nsB,swB)-0.7)/20m:0}}
CossHB bus_inv swB {ng_num(coss)} IC={ng_num(vdc-vphase_init[1])}
CossLB swB nsB {ng_num(coss)} IC={ng_num(vphase_init[1])}
RshB nsB 0 {{RSH}}

* Phase C half bridge and low-side shunt.
BghC ghC 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyC)-WEDGE)?1:((v(tri)>v(dutyC)+WEDGE)?0:(v(dutyC)+WEDGE-v(tri))/(2*WEDGE))) : 0) }}
BglC glC 0 V={{ ((v(fault)<1.5 && v(core5)>4.5) ? 10*v(arm)*((v(tri)<v(dutyC)+DTDU-WEDGE)?0:((v(tri)>v(dutyC)+DTDU+WEDGE)?1:(v(tri)-v(dutyC)-DTDU+WEDGE)/(2*WEDGE))) : 0) }}
BchHC bus_inv swC I={{v(bus_inv,swC)*((((v(ghC)<0)?0:((v(ghC)>10)?10:v(ghC)))/10)/RDSON+(1-((v(ghC)<0)?0:((v(ghC)>10)?10:v(ghC)))/10)/1Meg)}}
BchLC swC nsC I={{v(swC,nsC)*((((v(glC)<0)?0:((v(glC)>10)?10:v(glC)))/10)/RDSON+(1-((v(glC)<0)?0:((v(glC)>10)?10:v(glC)))/10)/1Meg)}}
BdhC swC bus_inv I={{(v(swC,bus_inv)>0.7)?(v(swC,bus_inv)-0.7)/20m:0}}
BdlC nsC swC I={{(v(nsC,swC)>0.7)?(v(nsC,swC)-0.7)/20m:0}}
CossHC bus_inv swC {ng_num(coss)} IC={ng_num(vdc-vphase_init[2])}
CossLC swC nsC {ng_num(coss)} IC={ng_num(vphase_init[2])}
RshC nsC 0 {{RSH}}

* Three-phase star-connected motor model with sinusoidal back-EMF.
VsenseA swA phA 0
RphA phA phaL {{RPH}}
LphA phaL emA {{LPH}}
BemfA emA nstar V={{EPK*sin(v(theta))}}
VsenseB swB phB 0
RphB phB phbL {{RPH}}
LphB phbL emB {{LPH}}
BemfB emB nstar V={{EPK*sin(v(theta)-2*pi/3)}}
VsenseC swC phC 0
RphC phC phcL {{RPH}}
LphC phcL emC {{LPH}}
BemfC emC nstar V={{EPK*sin(v(theta)+2*pi/3)}}

* Use explicit transient initial conditions so the bus starts at VDC even
* though it is separated from Cbulk by the modeled link ESL.
.ic V(bus)={{VDC}} V(cap_i)={{VDC}} V(hfA_cap)={{VDC}} V(hfB_cap)={{VDC}} V(hfC_cap)={{VDC}} V(fault)=0 V(busov_n)={ng_num(busov_high)} V(ov_ref)={ng_num(ov_ref_high)} V(ov_sense)={ng_num(ov_sense_init)} V(shA_f)=0 V(shB_f)=0 V(shC_f)=0 V(nstar)={ng_num(vcm_init)} V(swA)={ng_num(vphase_init[0])} V(swB)={ng_num(vphase_init[1])} V(swC)={ng_num(vphase_init[2])} V(phA)={ng_num(vphase_init[0])} V(phaL)={ng_num(vphase_init[0])} V(emA)={ng_num(vphase_init[0])} V(phB)={ng_num(vphase_init[1])} V(phbL)={ng_num(vphase_init[1])} V(emB)={ng_num(vphase_init[1])} V(phC)={ng_num(vphase_init[2])} V(phcL)={ng_num(vphase_init[2])} V(emC)={ng_num(vphase_init[2])}{bus_ff_ic}
{save_stmt}

.control
set noaskquit
set numdgt=8
{filetype_stmt} 
tran {ng_num(float(case['print_step_s']))} {ng_num(tstop)} 0 {ng_num(maxstep)} uic
let pac = v(swA,nstar)*i(VsenseA) + v(swB,nstar)*i(VsenseB) + v(swC,nstar)*i(VsenseC)
let pbus = v(bus)*i(Vbus_sense)
let psource = v(batt)*i(Vsense)
let iphase_abs = max(abs(i(VsenseA)),max(abs(i(VsenseB)),abs(i(VsenseC))))
let pdump = 2*v(bus,dump_bus)*v(bus,dump_bus)/2
let ptvs = v(bus)*i(Vtvs_sense)
let pmoss = (v(bus_inv,swA)*v(bus_inv,swA))*(v(ghA)/10/{ng_num(rds)}+(1-v(ghA)/10)/1Meg)+(v(swA,nsA)*v(swA,nsA))*(v(glA)/10/{ng_num(rds)}+(1-v(glA)/10)/1Meg)+(v(bus_inv,swB)*v(bus_inv,swB))*(v(ghB)/10/{ng_num(rds)}+(1-v(ghB)/10)/1Meg)+(v(swB,nsB)*v(swB,nsB))*(v(glB)/10/{ng_num(rds)}+(1-v(glB)/10)/1Meg)+(v(bus_inv,swC)*v(bus_inv,swC))*(v(ghC)/10/{ng_num(rds)}+(1-v(ghC)/10)/1Meg)+(v(swC,nsC)*v(swC,nsC))*(v(glC)/10/{ng_num(rds)}+(1-v(glC)/10)/1Meg)
meas tran pac_avg avg pac from={ng_num(tstart+float(case['load_ramp_s']))} to={ng_num(tload_end-float(case['load_ramp_s']))}
meas tran pac_energy integ pac from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran pbus_avg avg pbus from={ng_num(tstart+float(case['load_ramp_s']))} to={ng_num(tload_end-float(case['load_ramp_s']))}
meas tran pbus_energy integ pbus from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran psource_energy integ psource from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran dump_energy integ pdump from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran pmoss_avg avg pmoss from={ng_num(tstart+float(case['load_ramp_s']))} to={ng_num(tload_end-float(case['load_ramp_s']))}
meas tran pmoss_energy integ pmoss from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran tvs_current_peak max i(Vtvs_sense) from=0 to={ng_num(tstop)}
meas tran tvs_power_peak max ptvs from=0 to={ng_num(tstop)}
meas tran tvs_energy integ ptvs from=0 to={ng_num(tstop)}
meas tran vbus_max max v(bus) from=0 to={ng_num(tstop)}
meas tran vbus_min min v(bus) from=0 to={ng_num(tstop)}
meas tran busov_n_min min v(busov_n) from=0 to={ng_num(tstop)}
meas tran dump_gate_max max v(brkgate) from=0 to={ng_num(tstop)}
meas tran t_end max time from=0 to={ng_num(tstop)}
meas tran ia_max max i(VsenseA) from=0 to={ng_num(tstop)}
meas tran ia_min min i(VsenseA) from=0 to={ng_num(tstop)}
meas tran ib_max max i(VsenseB) from=0 to={ng_num(tstop)}
meas tran ib_min min i(VsenseB) from=0 to={ng_num(tstop)}
meas tran ic_max max i(VsenseC) from=0 to={ng_num(tstop)}
meas tran ic_min min i(VsenseC) from=0 to={ng_num(tstop)}
meas tran isource_max max i(Vsense) from=0 to={ng_num(tstop)}
meas tran isource_min min i(Vsense) from=0 to={ng_num(tstop)}
meas tran ibus_max max i(Vbus_sense) from=0 to={ng_num(tstop)}
meas tran ibus_min min i(Vbus_sense) from=0 to={ng_num(tstop)}
meas tran icap_rms rms i(Llink) from={ng_num(tstart)} to={ng_num(tload_end)}
meas tran icap_max max i(Llink) from=0 to={ng_num(tstop)}
meas tran icap_min min i(Llink) from=0 to={ng_num(tstop)}
meas tran fault_latched max v(fault) from=0 to={ng_num(tstop)}
let vds_hi_max = max(v(bus_inv,swA),max(v(bus_inv,swB),v(bus_inv,swC)))
let vds_low_max = max(abs(v(swA,nsA)),max(abs(v(swB,nsB)),abs(v(swC,nsC))))
let vds_any_max = max(abs(v(bus_inv,swA)),max(abs(v(bus_inv,swB)),max(abs(v(bus_inv,swC)),vds_low_max)))
meas tran vds_hi_peak max vds_hi_max from=0 to={ng_num(tstop)}
meas tran vds_any_peak max vds_any_max from=0 to={ng_num(tstop)}
meas tran vds_startup_peak max vds_any_max from=0 to={ng_num(tstart+arm_ramp)}
meas tran vds_postarm_peak max vds_any_max from={ng_num(tstart+arm_ramp)} to={ng_num(tstop)}
meas tran vbus_startup_max max v(bus) from=0 to={ng_num(tstart+arm_ramp)}
meas tran vbus_postarm_max max v(bus) from={ng_num(tstart+arm_ramp)} to={ng_num(tstop)}
meas tran i_phase_startup_peak max iphase_abs from=0 to={ng_num(tstart+arm_ramp)}
meas tran i_phase_postarm_peak max iphase_abs from={ng_num(tstart+arm_ramp)} to={ng_num(tstop)}
write {case['name']}_summary.raw {raw_vectors}
{trace_stmt.rstrip()}
quit
.endc
.end
"""
    metadata = {
        "name": case["name"],
        "vdc_v": vdc,
        "pac_request_w": target_pac,
        "duration_s": float(case["load_duration_s"]),
        "arm_ramp_s": arm_ramp,
        "rpm": rpm,
        "pole_pairs": pole_pairs,
        "electrical_frequency_hz": f_e,
        "ke_conversion": "E_phase_peak = ke_scale * rpm/(sqrt(3)*KV); convention uncertainty is a sweep input",
        "e_phase_peak_v": e_pk,
        "r_phase_ohm_assumed": rph,
        "l_phase_h_assumed": lph,
        "command_current_peak_a": i_cmd,
        "unclamped_current_solution_a": i_exact,
        "command_limit_a": i_limit,
        "svm_v_phase_peak_limit_v": vdc / math.sqrt(3.0),
        "svm_duty_margin_unverified": True,
        "dump_enabled": dump_active,
        "core5_v": vcore,
        "safe5_v": vsafe,
        "source_reverse_acceptance": "blocked by Dsrc in this initial no-absorption baseline",
        "arming": f"global gate-command ramp to requested PWM pattern over {arm_ramp:g} s; transient is reported separately from post-arm PWM interval",
        "mosfet_behavior_model": f"passive behavioral B-channel conductance capped to 0..10V normalized command, Ron={rds} ohm selected hot screen; no vendor macro model; fixed Coss={coss} F/device; behavioral body diode; gate transition timing from smoothed PWM comparator",
        "bridge_bypass_model": f"3 phase-leg branches, each one 4.7uF ceramic assumed 2.35uF at bias plus 0.1uF, Ceff={c_hf} F/leg; ESR={r_hf} ohm, ESL={l_hf} H; bulk bank is separate",
        "bus_voltage_feedforward": (f"RC-filtered bus sample, tau={bus_ff_tau:g} s, R={bus_ff_r:g} ohm, C={bus_ff_c:g} F"
                                     if bus_ff_tau > 0 else "direct instantaneous BUS node in PWM denominator; retained only for the pre-fix comparison"),
        "tvs_enabled": tvs_enabled,
        "tvs_behavior_model": "ST SMCJ40A linear high-current clamp proxy: TVS_INTERCEPT=49.1 V (up-rounded from 64.5 V - 24 A*0.644 ohm = 49.044 V; NOT VBR(max)); RD(max)=0.644 ohm; no guaranteed low-current I-V, RD temperature coefficient, device capacitance, package inductance, thermal, or repetitive-pulse model" if tvs_enabled else "disabled in this run",
        "ov_sense_filter_c_f": ov_sense_c,
        "ov_nominal_initial_busov_n_v": busov_high,
        "ov_nominal_initial_reference_v": ov_ref_high,
        "ov_comparator_vol_behavior_bound_v": 0.9,
        "ov_comparator_timing": "behavior-level threshold only; no official TLV1704 macro or worst-case propagation delay; independent BJT/OVP network checked separately",
        "shunt_trip_threshold_v": 75e-3,
        "nominal_shunt_current_trip_a": 75e-3 / shunt,
        "compact_output": compact_output,
        "data_scope": "ngspice measurements plus selected raw vectors; switch-level behavior screening, not closed-loop FOC qualification",
    }
    return net, metadata


def parse_measures(log_text: str) -> dict:
    measured = {}
    for name, value in re.findall(r"(?im)^\s*(\w+)\s*=\s*([-+]?\d+(?:\.\d*)?(?:e[-+]?\d+)?)", log_text):
        measured[name.lower()] = float(value)
    return measured


def run_case(case: dict, keep_raw: bool = True) -> dict:
    if not NGSPICE.exists():
        raise FileNotFoundError(f"ngspice console not found: {NGSPICE}")
    run_dir = RESULTS / case["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    net, metadata = build_netlist(case)
    cir = run_dir / "model.cir"
    cir.write_text(net, encoding="ascii")
    (run_dir / "parameters.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    log = run_dir / "ngspice.log"
    result_path = run_dir / "result.json"
    if result_path.exists():
        result_path.unlink()
    proc = subprocess.run([str(NGSPICE), "-b", "-o", str(log), str(cir)], cwd=run_dir,
                          text=True, capture_output=True, timeout=2400)
    if proc.returncode != 0:
        raise RuntimeError(f"ngspice failed ({proc.returncode}): {proc.stderr}\n{log.read_text(encoding='utf-8', errors='replace')[-6000:]}")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if re.search(r"(?im)^\s*Error:", log_text) or "ngspice-47 done" not in log_text:
        raise RuntimeError(f"ngspice reported model/measure errors despite zero exit code:\n{log_text[-6000:]}")
    measures = parse_measures(log_text)
    if "t_end" not in measures or abs(measures["t_end"] - float(case["tstop_s"])) > max(2e-6, float(case["max_step_s"]) * 2):
        raise RuntimeError(f"transient did not demonstrably reach tstop={case['tstop_s']}: {measures.get('t_end')}\n{log_text[-3000:]}")
    result = {
        "parameters": metadata,
        "ngspice_return_code": proc.returncode,
        "ngspice_version": "47",
        "measures": measures,
        "netlist_sha256": hashlib.sha256(net.encode("ascii")).hexdigest(),
        "log": str(log.relative_to(HERE)),
        "netlist": str(cir.relative_to(HERE)),
    }
    if not keep_raw:
        for raw in run_dir.glob("*_summary.raw"):
            raw.unlink()
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return result


def base_case(name: str, vdc: float, dump_enabled: bool, core5_v: float,
              safe5_v: float, tstop_s: float = 1.02, duration_s: float = 1.0,
              print_step_s: float = 10e-6, max_step_s: float = 1e-6,
              mos_type: str = "ipt") -> dict:
    mos = {
        "ipt": (5.0e-3, 2300e-12),
        "ipt_3p75m": (3.75e-3, 2300e-12),
        "irlb": (10.75e-3, 760e-12),
    }
    if mos_type not in mos:
        raise ValueError(f"unsupported MOS candidate: {mos_type}")
    rds_hot, coss_fixed = mos[mos_type]
    # The chopper comparator/LDO network uses its resistor-derived nominal
    # SAFE_5V value, not an ideal 5.000V rail.
    safe5_v = 4.9764817276 if safe5_v > 4.5 else 0.0
    return {
        "name": name,
        "vdc_v": vdc,
        "fsw_hz": 20_000.0,
        "rpm": 6_000.0,
        "pole_pairs": 7,
        "kv_rpm_per_v": 270.0,
        "ke_scale": 1.0,
        "r_phase_ohm": 0.017,
        "l_phase_h": 20e-6,
        "pac_request_w": -1800.0,
        "phase_command_limit_a": 110.0,
        "deadtime_s": 500e-9,
        "tr_tf_s": 500e-9,
        "max_step_s": max_step_s,
        "print_step_s": print_step_s,
        "tstop_s": tstop_s,
        "load_start_s": 0.012,
        "load_duration_s": duration_s,
        "load_ramp_s": 0.002,
        "arm_ramp_s": 2e-6,
        "dump_enabled": dump_enabled,
        "core5_v": core5_v,
        "safe5_v": safe5_v,
        "dc_link_c_f": 30e-3,
        "dc_link_esr_ohm": 6.333e-3,
        "dc_link_esl_h": 15e-9,
        "bridge_bypass_c_eff_f": 2.45e-6,
        "bridge_bypass_esr_ohm": 5e-3,
        "bridge_bypass_esl_h": 1e-9,
        "source_r_ohm": 10e-3,
        "source_l_h": 1e-6,
        "mos_type": mos_type,
        "mos_rds_hot_ohm": rds_hot,
        "mos_coss_f": coss_fixed,
        "shunt_ohm": 0.5e-3,
        "ov_sense_c_f": 1e-9,
        "vbus_feedforward_tau_s": 100e-6,
        "vbus_feedforward_c_f": 1e-6,
        "tvs_enabled": False,
        "compact_output": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["smoke", "switch_profile", "switch_filtered_nodump",
                                           "switch_filtered_tvs", "switch_positive_filtered", "baseline"], default="smoke")
    parser.add_argument("--voltage", type=float, default=32.4)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--core5", type=float, default=5.0)
    parser.add_argument("--safe5", type=float, default=None)
    args = parser.parse_args()
    if args.safe5 is None:
        args.safe5 = 5.0
    if args.case == "smoke":
        c = base_case(f"smoke_regen_{args.voltage:g}V_{'dump' if args.dump else 'nodump'}",
                      args.voltage, args.dump, args.core5, args.safe5,
                      tstop_s=0.02, duration_s=0.01, print_step_s=2e-6, max_step_s=200e-9)
        result = run_case(c, keep_raw=True)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.case == "switch_profile":
        # 5 ms detailed window at 20 ns maximum internal step. The 1 s heat
        # and energy cases are evaluated separately with the averaged model.
        c = base_case("switch32p4_regen1800W_5ms_max20ns", 32.4, True, 5.0,
                      5.0, tstop_s=6e-3, duration_s=4e-3,
                      print_step_s=20e-9, max_step_s=20e-9, mos_type="ipt")
        c["load_start_s"] = 1e-3
        c["load_ramp_s"] = 200e-6
        c["reltol"] = 1e-3
        result = run_case(c, keep_raw=True)
        out = {"model_scope": "ngspice-47 switch-level behavior screening; 5 ms detailed window only, not a 1 s motor/FOC qualification",
               "case": result}
        target = RESULTS / "switch_profile_results.json"
        target.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(json.dumps({"result": str(target), "measures": result["measures"],
                          "netlist_sha256": result["netlist_sha256"]}, indent=2, ensure_ascii=False))
    elif args.case in {"switch_filtered_nodump", "switch_filtered_tvs", "switch_positive_filtered"}:
        target_pac = 1800.0 if args.case == "switch_positive_filtered" else -1800.0
        case = base_case(args.case, 32.4, True, 5.0, 5.0,
                         tstop_s=2.2e-3, duration_s=1.8e-3,
                         print_step_s=20e-9, max_step_s=20e-9, mos_type="ipt")
        case.update({
            "pac_request_w": target_pac,
            "load_start_s": 0.2e-3,
            "load_ramp_s": 150e-6,
            "vbus_feedforward_tau_s": 100e-6,
            "vbus_feedforward_c_f": 1e-6,
            "tvs_enabled": args.case == "switch_filtered_tvs",
            "compact_output": True,
            "reltol": 1e-3,
        })
        print(f"Running {case['name']} with 100 us filtered VBUS feedforward; TVS={case['tvs_enabled']}...")
        result = run_case(case, keep_raw=True)
        target = RESULTS / f"{args.case}_results.json"
        target.write_text(json.dumps({"model_scope": "ngspice-47 2.2 ms switch-level behavior sensitivity; not FOC qualification",
                                      "case": result}, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(json.dumps({"result": str(target), "measures": result["measures"],
                          "netlist_sha256": result["netlist_sha256"]}, indent=2, ensure_ascii=False))
    else:
        cases = []
        for v in (22.5, 32.4, 37.8):
            # Keep independent SAFE_5V alive while the resistor/Q7 branch is
            # open; a separate SAFE5off case below exercises loss of the rail.
            cases.append(base_case(f"regen_{v:g}V_nodump_SAFE5on", v, False, 5.0, 5.0))
            cases.append(base_case(f"regen_{v:g}V_dump_SAFE5", v, True, 5.0, 5.0))
        cases.extend([
            base_case("regen_37p8V_core5off_SAFE5on", 37.8, True, 0.0, 5.0),
            base_case("regen_37p8V_SAFE5off", 37.8, True, 5.0, 0.0),
        ])
        all_results = []
        for case in cases:
            print(f"Running {case['name']}...")
            result = run_case(case, keep_raw=False)
            all_results.append(result)
            print(json.dumps(result["measures"], ensure_ascii=False))
        summary = {"ngspice_version": "47", "cases": all_results}
        (RESULTS / "baseline_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(f"Wrote {RESULTS / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
