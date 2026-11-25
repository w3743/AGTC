"""Run reproducible ngspice boundary comparisons for the independent charger board."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
SPICE = POWER / "FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe"
OUT = HERE / "充电MPPT与自持仿真核算.json"


def simulate(work, name, schematic, step, stop, measures):
    deck = work / f"{name}.cir"
    commands = "\n".join(f"meas tran {key} {expr}" for key, expr in measures.items())
    deck.write_text(
        f"{name}\n{schematic}\n.options method=gear reltol=1e-5 abstol=1e-11\n"
        f".tran {step} {stop} 0 {step} uic\n.control\nrun\n"
        f"{commands}\nquit\n.endc\n.end\n", encoding="ascii"
    )
    run = subprocess.run([SPICE, "-b", deck.name], cwd=work, capture_output=True,
                         encoding="utf-8", errors="replace", timeout=60, check=True)
    raw = dict(re.findall(r"^([a-z][a-z0-9_]*)\s*=\s*([-+\d.eE]+)", run.stdout, re.M))
    if not all(key in raw and math.isfinite(float(raw[key])) for key in measures):
        raise RuntimeError(f"{name}: Missing ngspice measurements\n{run.stdout[-2500:]}\n{run.stderr[-2500:]}")
    return {
        "measurements": {key: float(raw[key]) for key in measures},
        "deck_sha256": sha256(deck.read_bytes()).hexdigest(),
    }


PV = (
    {"vmp_v": 12.0, "voc_v": 15.0, "rated_w": 20.0, "after_shade": 1.0},
    {"vmp_v": 18.0, "voc_v": 22.0, "rated_w": 60.0, "after_shade": 0.28},
    {"vmp_v": 16.0, "voc_v": 20.0, "rated_w": 40.0, "after_shade": 1.0},
)


def pv_power_stage_screen():
    vin, vout, average_a = 12, 37.6, 5
    fs, minimum_l = 100_000, 22e-6 * .8
    duty = 1 - vin / vout
    ripple = vin * duty / (fs * minimum_l)
    peak = average_a + ripple / 2
    mean_squared = average_a**2 + ripple**2 / 12
    return {
        "scope": "Single-channel 60 W analytical screen, not a controller, peak-current protection or thermal test. "
                 "100 kHz is a comparison point, and 10 V VGS is an assumption, not a demonstrated gate voltage.",
        "input_v": vin, "bus_proxy_v": vout, "average_a": average_a,
        "frequency_hz": fs, "minimum_l_h": minimum_l, "ideal_duty": duty,
        "ripple_pp_a": ripple, "peak_a": peak, "rms_a": math.sqrt(mean_squared),
        "screen_isat_a": 9.5, "screen_irms_a": 8.0,
        "mosfet_pair_conduction_at_10v_25c_w": mean_squared * .0053,
        "mosfet_pair_conduction_at_assumed_1p6_r_multiplier_w": mean_squared * .0053 * 1.6,
        "inductor_copper_25c_w": mean_squared * .0178,
        "hard_switch_overlap_at_assumed_100ns_w": .5 * vout * average_a * 100e-9 * fs,
        "gate_charge_at_10v_typ_w": 2 * 73e-9 * 10 * fs,
    }


def pv_model(work):
    """Averaged current sink/controller comparison; one shaded input is later unplugged."""
    lines = ["Vbus bus 0 36"]
    measures = {}
    # A 1-D perturb-and-observe loop is simulated on independent panel curves.
    # Its PV current command is read back in ngspice with the electrical port capacitors.
    for index, p in enumerate(PV, 1):
        vmp, voc, power = p["vmp_v"], p["voc_v"], p["rated_w"]
        imp = power / vmp
        isc = 1.1 * imp

        def curve(voltage, gain):
            if voltage <= vmp:
                return gain * (isc - (isc - imp) * voltage / vmp)
            return gain * max(0.0, imp * (voc - voltage) / (voc - vmp))

        setpoints = []
        voltage, direction, previous = voc * 0.70, 1, None
        for sequence in range(100):
            gain = 1 if sequence < 58 else p["after_shade"]
            if sequence == 58:
                previous = None
            sampled = voltage * curve(voltage, gain)
            setpoints.append((sequence * .01, voltage))
            if previous is not None and sampled < previous:
                direction = -direction
            previous = sampled
            voltage = min(voc * .99, max(2.0, voltage + direction * .18))
        def transition_rows(intervals):
            values = [f"0 {intervals[0][1]:.8g}"]
            for (time, value), (_, old) in zip(intervals[1:], intervals[:-1]):
                values.append(f"{time-1e-6:.8f} {old:.8g} {time:.8f} {value:.8g}")
            return " ".join(values)

        gain_pre = p["after_shade"]
        gain_end = 0 if index == 2 else gain_pre
        lines += [
            f"Vref{index} r{index} 0 PWL({transition_rows(setpoints)})",
            f"Vgain{index} g{index} 0 PWL(0 1 0.579999 1 0.58 {gain_pre} 0.879999 {gain_pre} 0.88 {gain_end})",
            f"Bpv{index} 0 pv{index} I=V(g{index})*max(0,({isc:.9g}-{(isc-imp)/vmp:.9g}*V(pv{index})))"
            f"*u({vmp}-V(pv{index}))+V(g{index})*max(0,{imp/(voc-vmp):.9g}*({voc}-V(pv{index})))"
            f"*u(V(pv{index})-{vmp})",
            f"Cpv{index} pv{index} 0 22u ic={setpoints[0][1]}",
            f"Rin{index} pv{index} 0 100meg",
            f"Bdraw{index} pv{index} inv{index} I=min(5,max(0,30*(V(pv{index})-V(r{index}))))",
            f"Vin{index} inv{index} 0 0",
            f"Bboost{index} 0 out{index} I=0.92*max(0,V(pv{index})*I(Vin{index}))/max(V(bus),1)",
            f"Vsense{index} out{index} bus 0",
            f"Bp{index} p{index} 0 V=V(pv{index})*I(Vin{index})",
        ]
        measures.update({
            f"p{index}_unshaded_w": f"avg v(p{index}) from=0.40 to=0.57",
            f"p{index}_shaded_w": f"avg v(p{index}) from=0.72 to=0.86",
            f"p{index}_unplug_w": f"avg v(p{index}) from=0.93 to=0.99",
            f"i{index}_min_a": f"min i(Vsense{index}) from=0.1 to=0.99",
        })
    result = simulate(work, "three_pv", "\n".join(lines), "25u", "1.0", measures)
    m = result["measurements"]
    passed = all(abs(m[f"p{i}_unshaded_w"] - panel["rated_w"]) <= 0.05 * panel["rated_w"] and
                 abs(m[f"p{i}_shaded_w"] - panel["rated_w"] * panel["after_shade"]) <= 0.05 * panel["rated_w"] and
                 m[f"i{i}_min_a"] >= -1e-7 for i, panel in enumerate(PV, 1))
    result.update(status="PASS_MODEL" if passed and abs(m["p2_unplug_w"]) < .01 else "FAIL_MODEL",
                  irradiance_inputs=PV, power_stage_screen=pv_power_stage_screen(),
                  scope="P&O on idealized single-knee I–V curves, averaged power stage, "
                  "ideal unidirectional output. No measured PV curves, device-specific reverse current or local-shading global maximum proof.")
    return result


def cold_start(work):
    # Both BMS FETs are open. A tiny effective series leakage is used only to set a
    # numerical upper comparison. LD voltage is a separate observation, not permission.
    result = simulate(work, "cold_start_partition", """
Vpva a 0 PWL(0 0 1m 0 12m 12)
Vpvb b 0 0
Vpvc c 0 0
Da a aux dsel
Db b aux dsel
Dc c aux dsel
.model dsel D(Is=1n N=1.08 Rs=.12)
Raux aux 0 4.7k
Bwake dv 0 V=3.3*u(V(aux)-6)
Rw dv wake 150
Cw wake 0 10u ic=0
Ilogic wake 0 2m
Dclamp 0 wake dsel
Vpack pack 0 22.5
Roff pack bus 1g
Vtest src 0 PWL(0 0 50m 0 51m 28 90m 28)
Sbias src bus gate 0 sdefault
Vgate gate 0 0
.model sdefault SW(Ron=.02 Roff=1e12 Vt=1 Vh=.1)
Rload bus 0 150
Cload bus 0 60m ic=0
""", "20u", "100m", {
        "local_3v3_v": "find v(wake) at=45m", "bus_v": "find v(bus) at=95m",
        "batt_current_a": "find i(Vpack) at=95m",
        "wake_crossing_s": "when v(wake)=2.7 rise=1",
    })
    m = result["measurements"]
    result["local_bias"] = "PASS_MODEL" if m["local_3v3_v"] > 2.7 and m["bus_v"] < .001 else "FAIL_MODEL"
    result["pack_charge"] = "BLOCKED_BY_FROZEN_INTERFACE"
    result["scope"] = ("Ideal powered bias with example 12 V source after diode OR. Both BMS FETs and charger "
                       "output isolation off. Nanoamp leakage is model-only. AFE wake and charge are separate.")
    result["boost_body_diode_negative_control"] = simulate(work, "boost_body_diode", """
Vpanel pv 0 PWL(0 0 1m 0 2m 12)
Rpanel pv pin 2.5
Lbare pin sw_bare 22u
Dbare sw_bare raw_bare dbody
Lprotected pin sw_protected 22u
Dprotected sw_protected raw_protected dbody
.model dbody D(Is=50n N=1.2 Rs=.012)
Vgate gate 0 0
Siso raw_protected bus_protected gate 0 iso
.model iso SW(Ron=.01 Roff=1g Vt=1 Vh=.1)
Rinternal raw_protected 0 1meg
Cbare raw_bare 0 1m ic=0
Cprotected bus_protected 0 1m ic=0
Rbare raw_bare 0 47
Rprotected bus_protected 0 47
""", "10u", "40m", {
        "bare_bus_v": "find v(raw_bare) at=39m",
        "protected_bus_v": "find v(bus_protected) at=39m",
    })
    body = result["boost_body_diode_negative_control"]["measurements"]
    result["isolation_negative_control"] = "PASS_MODEL" if (
        body["bare_bus_v"] > 5 and body["protected_bus_v"] < .001) else "FAIL_MODEL"
    return result


def ld_wake_probe(work):
    """Screen the newly requested charge-authorized interface over existing PACK+/- only."""
    result = {}
    for load_ma in (10, 60):
        load = load_ma / 1000
        case = simulate(work, f"ld_probe_{load_ma}ma", f"""
Venable gate 0 PULSE(0 1 .05 1u 1u 5.5 25)
Bprobe 0 source I=V(gate)*min(.05,max(0,.8*(3.3-V(bus))))
Vsupply source bus 0
Cbus bus 0 79.784m ic=0
Bvehicle bus 0 I={load}*min(1,max(0,V(bus)/.05))
Rld bus ld 5.1k
Rinternal ld 0 80k
Vbattery pack 0 22.5
Ropen pack bus 1g
Bpower en 0 V=V(bus)*I(Vsupply)
""", "200u", "6", {
            "v_ld_end_v": "find v(ld) at=5.5",
            "v_bus_end_v": "find v(bus) at=5.5",
            "i_pack_at_end_a": "find i(Vbattery) at=5.5",
            "energy_injected_j": "integ v(en) from=.05 to=5.551",
            "source_current_after_off_a": "find i(Vsupply) at=5.7",
        })
        case["vld_threshold_crossing_supported_in_model"] = case["measurements"]["v_ld_end_v"] >= 2.25
        result[str(load_ma)] = case
    return {
        "status": "PROPOSED_INTERFACE_NOT_QUALIFIED", "cases_by_assumed_load_ma": result,
        "model_limits": {"max_v": 3.3, "max_a": .05, "duration_s": 5.5, "energy_upper_j": .9075,
                         "bus_cap_f": .079784, "ld_resistor_ohm": 5100, "ld_typical_shutdown_pulldown_ohm": 80000,
                         "ld_wake_threshold_max_v": 2.25},
        "scope": "Ideal only. This probe may raise LD, but cannot imply MCU boot, CCL, CHG conduction or charge permission. "
                 "Not approved as an external precharge stimulus. Active prebias must end and positively disconnect before "
                 "any traditional BMS precharge. 60 mA comparison deliberately tests failure to reach threshold.",
    }


def charge_window_model(work):
    """Run positive and negative permission cases without assuming real BMS timing."""
    result = {}
    for label, permit in (("verified_charge_path_and_fresh_lease", 1), ("no_lease", 0)):
        case = simulate(work, f"charge_window_{permit}", f"""
Vlease lease 0 PWL(0 0 .009999 0 .01 {permit} .499999 {permit} .5 0)
Vnotprecharge np 0 PWL(0 0 .019999 0 .02 1)
Bqualified g 0 V=V(lease)*V(np)
Bccl ccl 0 V=.4*V(g)
Bsource 0 output I=V(ccl)
Vsource output bus 0
Cdownstream bus 0 60m ic=30.2
Rdownstream bus 0 10k
Vcell stack 0 29.5
Dcharge bus inter diode
.model diode D(IS=10n N=1.15 Rs=.02)
Sverified inter stack g 0 verified
.model verified SW(Ron=.006 Roff=1g Vt=.5 Vh=.1)
""", "20u", ".58", {
            "source_before_lease_a": "find i(Vsource) at=.005",
            "source_during_precharge_a": "find i(Vsource) at=.015",
            "pack_during_precharge_a": "find i(Vcell) at=.015",
            "source_with_lease_a": "find i(Vsource) at=.4",
            "pack_at_400ms_a": "find i(Vcell) at=.4",
            "source_after_revoke_a": "find i(Vsource) at=.51",
        })
        m = case["measurements"]
        case["status"] = "PASS_MODEL" if (
            all(abs(m[key]) < 1e-6 for key in ("source_before_lease_a", "source_during_precharge_a", "pack_during_precharge_a",
                                               "source_after_revoke_a")) and
            abs(m["source_with_lease_a"] - .4 * permit) < 1e-6 and
            abs(m["pack_at_400ms_a"] - .4 * permit) < .02
        ) else "FAIL_MODEL"
        result[label] = case
    return {
        "status": "PASS_MODEL" if all(case["status"] == "PASS_MODEL" for case in result.values()) else "FAIL_MODEL",
        "scope": "Fault-absent interface logic and ideal switches with a prebiased test bus. Real BMS lease, "
                 "charge-gate state, diode behaviour, frozen precharge interlock and isolation delay unverified.",
        "cases": result,
    }


def regen_window(work):
    divider = .400 / 37.8
    results = {}
    for label, release_v, assert_us, release_us, pv_decay_us, bq_decay_us in (
        ("target_window", 37.2, 29, 18, 5, 30),
        ("native_divider_typ", .3945 / divider, 29, 18, 5, 30),
        ("delayed_response_sensitivity", 37.2, 60, 36, 20, 100),
    ):
        release_threshold = release_v * divider
        hyst = (.400 - release_threshold) / 2
        case = simulate(work, f"regen_{label}", f"""
Vbus bus 0 PWL(0 36.6 100u 36.6 148u 39 600u 39 630u 37.5 700u 37.5 750u 36.5 800u 36.5)
Bdiv div 0 V=V(bus)*{divider:.12g}
Vlogic one 0 1
Sov one raw div 0 threshold off
Rraw raw 0 1k
.model threshold SW(Ron=1m Roff=1g Vt={.400-hyst:.12g} Vh={hyst:.12g})
Estart delayin 0 raw 0 1
Tassert delayin 0 assert 0 Z0=50 TD={assert_us}u
Rassert assert 0 50
Trelease delayin 0 release 0 Z0=50 TD={release_us}u
Rrelease release 0 50
Bfault fault 0 V=u(V(assert)-.5)*u(V(release)-.5)
Bce ce 0 V=3.3*V(fault)
Bbkin bkin 0 V=3.3*(1-V(fault))
Renable bkin en 1k
Cenable en 0 100p ic=3.3
Bpv_req pv_req 0 V=2*u(V(en)-1.65)
Bbq_req bq_req 0 V=2*u(1.65-V(ce))
Rpv pv_req pv 1k
Cpv pv 0 {pv_decay_us}n ic=2
Rbq bq_req bq 1k
Cbq bq 0 {bq_decay_us}n ic=2
Bbranch 0 sum I=V(pv)+V(bq)
Vinject sum bus 0
""", ".05u", "800u", {
            "t_trip_s": "when v(raw)=0.5 rise=1",
            "t_release_s": "when v(raw)=0.5 fall=1",
            "trip_v": "find v(bus) when v(raw)=0.5 rise=1",
            "release_v": "find v(bus) when v(raw)=0.5 fall=1",
            "t_ce_disable_s": "when v(ce)=1.65 rise=1",
            "t_bkin_assert_s": "when v(bkin)=1.65 fall=1",
            "t_ucc_disable_s": "when v(en)=1.65 fall=1",
            "t_ce_release_s": "when v(ce)=1.65 fall=1",
            "t_below_100ma_s": "when i(Vinject)=0.1 fall=1",
            "i_hysteresis_hold_a": "find i(Vinject) at=680u",
        })
        m = case["measurements"]
        for event in ("ce_disable", "bkin_assert", "ucc_disable", "below_100ma"):
            m[f"trip_to_{event}_us"] = (m[f"t_{event}_s"] - m["t_trip_s"]) * 1e6
        m["release_to_ce_release_us"] = (m["t_ce_release_s"] - m["t_release_s"]) * 1e6
        passed = (abs(m["trip_v"] - 37.8) < .005 and abs(m["release_v"] - release_v) < .005
                  and abs(m["trip_to_ce_disable_us"] - assert_us) < .2
                  and abs(m["trip_to_bkin_assert_us"] - assert_us) < .2
                  and abs(m["trip_to_ucc_disable_us"] - assert_us) < .3
                  and abs(m["release_to_ce_release_us"] - release_us) < .2
                  and m["trip_to_below_100ma_us"] > assert_us and abs(m["i_hysteresis_hold_a"]) < .03)
        results[label] = {"status": "PASS_MODEL" if passed else "FAIL_MODEL", **case,
                          "assumptions": {"falling_threshold_mv": 1000 * release_threshold,
                                          "tlv_assert_us": assert_us, "tlv_release_us": release_us,
                                          "pv_decay_tau_us": pv_decay_us, "bq_decay_tau_us": bq_decay_us}}
    return {
        "status": "PASS_MODEL" if all(case["status"] == "PASS_MODEL" for case in results.values()) else "FAIL_MODEL",
        "cases": results, "tlv6700_source": "TI SNVSAV2B sections 7.5–7.7, Figure 1, INB− to OUTB",
        "plain_divider_threshold_v_without_resistor_tolerance": {
            "trip_min": .396 / divider, "trip_max": .404 / divider,
            "release_min": .387 / divider, "release_max": .400 / divider,
        },
        "scope": "Behavioral TLV6700 hysteresis and asymmetric propagation (29/18 us nominal at 5 V, 10 mV overdrive), "
                 "CE high, BKIN and UCC EN low. The 37.2 V case assumes added threshold shaping; a plain divider does not produce it. "
                 "PV/BQ current decay constants and UCC EN load are assumptions, not silicon data or guaranteed cutoff times. "
                 "The ramp through 37.8 V does not immediately deliver the datasheet's 10 mV overdrive. "
                 "No CV, bus dynamics, fault reset, body-diode isolation or output disconnect timing is modeled.",
    }


def b3_epr_power_loss(work):
    # 47 ohm (C114623) series limiter + TPS3809K33 (2.93 V threshold) cuts NCE01P30K + fires self-powered MMBT5551 / 100 ohm active bleeder across 47 nF 100 V (C43254).
    return simulate(work, "b3_epr_power_loss_screen", """
Vraw raw 0 48
Rlim raw sw_in 47
Vvpwr vpwr 0 PWL(0 3.25 1m 3.25 1.01m 0)
Benable enable 0 V={v(vpwr)>2.93 ? 1 : 0}
Sdisconnect sw_in prot enable 0 ideal_switch
Bbleed_en bleed_en 0 V={1-v(enable)}
Sbleed prot bleed_node bleed_en 0 bleed_switch
Rbleed bleed_node 0 100
Cprot prot 0 47n IC=48
Rillustrative_load prot 0 48k
Dclamp 0 prot z56
.model ideal_switch SW(Ron=.065 Roff=1e12 Vt=.5)
.model bleed_switch SW(Ron=.25 Roff=1e12 Vt=.5)
.model z56 D(Bv=56 Ibv=2m Is=1n)
""", "50n", "2m", {
        "powered_v": "find v(prot) at=.9m",
        "vpwr_zero_v": "find v(prot) at=1.01m",
        "unpowered_peak_v": "max v(prot) from=1.01m to=2m",
    })


def b3_epr_gate_rc_screen(work):
    # The ideal fast_pnp switch represents the intended MMBT5401 (C8326, -150 V) turn-off path, not its device model.
    # 47 ohm (C114623) series limiter + 100 ohm gate pull-up + 47 nF 100 V (C43254) + 100 ohm active MMBT5551 bleeder + 56 V clamp.
    results = {}
    for name, initial, peak, cutoff in (("unpowered_hotplug_from_zero", 0, 48, 20.6),
                                        ("unpowered_step_from_spr", 5, 48, 20.6),
                                        ("powered_surge_from_epr", 48, 82.4, 56)):
        initial_vsg = min(initial / 2, 12)
        results[name] = simulate(work, f"b3_epr_{name}", f"""
Vsource source 0 PWL(0 {initial} 1m {initial} 1.001m {peak} 2m {peak})
Rsource source src_pin 2
Rlim src_pin raw 47
Rpullup_hi raw gate_pnp_c 100
Cgs raw gate 3.81n IC={initial_vsg}
Dgate gate raw z12
Rpulldown gate sink 100k
Bpermission permission 0 V={{v(src_pin)<{cutoff} ? 1 : 0}}
Binhibit inhibit 0 V={{1-v(permission)}}
Sdriver_on sink 0 permission 0 ideal_switch
Sdriver_off gate_pnp_c gate inhibit 0 fast_pnp
Sfet raw prot raw gate p_switch
Coss raw prot 93p
Cprot prot 0 47n IC={initial}
Sbleed prot bleed_node inhibit 0 bleed_sw
Rbleed bleed_node 0 100
Rload prot 0 48k
Bclamp prot 0 I={{max(0, (v(prot)-58.53)/135)}}
.model z12 D(Bv=12 Ibv=1m Is=1n)
.model ideal_switch SW(Ron=1 Roff=1e12 Vt=.5)
.model fast_pnp SW(Ron=10 Roff=1e12 Vt=.5)
.model bleed_sw SW(Ron=.25 Roff=1e12 Vt=.5)
.model p_switch SW(Ron=.065 Roff=1e12 Vt=1.5 Vh=.1)
""", "50n", "2m", {"pre_step_v": "find v(prot) at=.99m", "peak_v": "max v(prot) from=1m to=2m"})
    return results


def b3_pv_or_screen(work):
    # Each PV input independently feeds AUX_HV_IN via SS310-SMC AND triggers AUX_WAKE_OR -> PRE_3V6 -> 5V_SYS with BUS=0 V.
    results = {}
    for source_index, pv in enumerate(PV, 1):
        inputs = "\n".join(
            f"Vpv{index} pv{index} 0 PWL(0 0 1m {pv['vmp_v'] if index == source_index else 0})\n"
            f"Dpv{index} pv{index} aux ss310"
            for index in range(1, 4))
        results[f"pv{source_index}"] = simulate(work, f"b3_pv{source_index}_aux_or", f"""
{inputs}
Vbus bus 0 0
Dbus bus aux ss310
Cinput aux 0 4.7u
Iload aux 0 PWL(0 0 5m 0 5.1m .1)
Bwake en 0 V={{3.3*u(v(aux)-6.0)}}
Bpre pre_tgt 0 V={{3.60*u(v(aux)-4.5)*u(v(en)-1.24)}}
Rpre pre_tgt pre 20
Cpre pre 0 10u ic=0
Bsys sys_tgt 0 V={{5.10*u(v(pre)-2.5)}}
Rsys sys_tgt sys 20
Csys sys 0 10u ic=0
.model ss310 D(Is=1u N=1.4 Rs=.14)
""", "10u", "15m", {
            "aux_input_v": "find v(aux) at=15m",
            "pre_3v6_v": "find v(pre) at=15m",
            "sys_5v_v": "find v(sys) at=15m",
            "bus_feed_a": "find i(Vbus) at=15m",
        })
    return results


def b3_bbm_rc_screen(work):
    # 100 kOhm (C25803) + 220 nF (C83626, X7R low-corner 0.765 * 220n = 168.3n) + fast BAT46W discharge path.
    results = {}
    for name, resistance, capacitance in (("nominal_rc", 100e3, 220e-9),
                                         ("low_rc_x7r_screen", 99e3, 220e-9 * .765)):
        case = simulate(work, f"b3_bbm_{name}", f"""
Vold old 0 PWL(0 3 50m 3 50.001m 0)
Vnew new 0 PWL(0 0 50m 0 50.001m 3)
Rold old old_rc {resistance}
Rnew new new_rc {resistance}
Cold old_rc 0 {capacitance}
Cnew new_rc 0 {capacitance}
Doff old_rc dump schottky
Roff dump old 100
.model schottky D(Is=1u N=1 Rs=.05)
""", "2u", "75m", {
            "old_low_s": "when v(old_rc)=.89 fall=1",
            "new_high_s": "when v(new_rc)=1.48 rise=1",
        })
        measurements = case["measurements"]
        results[name] = {**case, "command_to_off_s": measurements["old_low_s"] - .05,
                         "remaining_deadtime_with_500us_total_off_s": measurements["new_high_s"] - .0505}
    return results


def b3_boost_discharge_screen(work):
    # Upstream DMP3098L-7 opens 5V_DRV_SW at t=0.1m; 100 nF HF + 1 kOhm PD on 5V_DRV_SW and 4.7 uF + active AO3400A/100 ohm bleeder on 10V_DRV discharge < 5 ms.
    return simulate(work, "b3_boost_disabled_after_run", """
Vsys sys 0 5.1
Vgate drv_en 0 PWL(0 3.3 .1m 3.3 .101m 0)
Ssupply sys drv_sw drv_en 0 pmos_sw
Cin drv_sw 0 100n IC=5.1
Rdrv_pd drv_sw 0 1k
Lboost drv_sw sw_l 10u
Rl sw_l sw .15
Csw sw 0 50p
Dboost sw drv schottky
Cout drv 0 4.7u IC=11.406
Rpassive drv 0 10k
Rfeedback drv 0 190.1k
Bbleed_ctl bleed_ctl 0 V={3.3*(1-u(v(drv_en)-1.5))}
Sbleed drv bleed_node bleed_ctl 0 bleed_sw
Rbleed bleed_node 0 100
.model pmos_sw SW(Ron=.06 Roff=1e8 Vt=1.5 Vh=.2)
.model bleed_sw SW(Ron=.05 Roff=1e8 Vt=1.5 Vh=.2)
.model schottky D(Is=1u N=1 Rs=.05)
""", "1u", "20m", {
        "drv_sw_at_5ms_v": "find v(drv_sw) at=5m",
        "drv_at_5ms_v": "find v(drv) at=5m",
        "drv_below_50mv_s": "when v(drv)=.05 fall=1",
    })


def b2_lock_review(work):
    tolerance = .002
    corners = {
        "low": (1.15e6 * (1 - tolerance), 12e3 * (1 + tolerance), -15e-9, .396, .387),
        "nominal": (1.15e6, 12e3, 0, .400, .3945),
        "high": (1.15e6 * (1 + tolerance), 12e3 * (1 - tolerance), 15e-9, .404, .400),
    }
    ov = {}
    for label, (top, bottom, bias, rising, falling) in corners.items():
        case = simulate(work, f"b3_ov_{label}", f"""
Vbus bus 0 PWL(0 35 1m 40 2m 35)
Rtop bus sense {top}
Rbottom sense 0 {bottom}
Ibias sense 0 DC {bias}
""", "1u", "2m", {
            "trip_v": f"find v(bus) when v(sense)={rising} rise=1",
            "release_v": f"find v(bus) when v(sense)={falling} fall=1",
        })
        predicted = {key: threshold * (1 + top / bottom) + bias * top
                     for key, threshold in (("trip_v", rising), ("release_v", falling))}
        ov[label] = {**case, "analytic_v": predicted}
    boost_off = simulate(work, "b2_boost_disabled_diode_path", """
Vsys sys 0 PWL(0 0 1m 5.1 4m 5.1)
Lboost sys sw 10u
Dboost sw drv schottky
Cout drv 0 4.7u
Rload drv 0 1k
.model schottky D(Is=1u N=1 Rs=.05)
""", "1u", "4m", {"drv_v": "find v(drv) at=4m"})
    epr_power_loss = b3_epr_power_loss(work)
    epr_gate_rc = b3_epr_gate_rc_screen(work)
    bbm = b3_bbm_rc_screen(work)
    boost_discharge = b3_boost_discharge_screen(work)
    pv_or = b3_pv_or_screen(work)
    pv_startup_ok = all(
        case["measurements"]["aux_input_v"] > 4.5
        and case["measurements"]["pre_3v6_v"] > 3.45
        and case["measurements"]["sys_5v_v"] > 4.95
        and abs(case["measurements"]["bus_feed_a"]) < 1e-4
        for case in pv_or.values()
    )
    boost_disconn_ok = (
        boost_off["measurements"]["drv_v"] > 4
        and boost_discharge["measurements"]["drv_sw_at_5ms_v"] < .05
        and boost_discharge["measurements"]["drv_at_5ms_v"] < .05
        and boost_discharge["measurements"]["drv_below_50mv_s"] < .005
    )
    model_pass = (
        all(abs(case["measurements"][key] - value) < .001
            for case in ov.values() for key, value in case["analytic_v"].items())
        and pv_startup_ok
        and boost_disconn_ok
    )
    margin = ov["low"]["measurements"]["trip_v"] - 37.80
    bov_margin = 40.30 - ov["high"]["measurements"]["trip_v"]
    # 230.0 kOhm = 220 kOhm (C861231) + 10 kOhm (C5123581) over 10 kOhm (C5123581)
    bq_nominal = 1.536 * (1 + 230e3 / (10e3 + 33))
    bq_upper_no_fbg_min = 1.536 * 1.007 * (1 + 230e3 * 1.001 / (10e3 * .999))
    boost_low = .588 * (1 + 18.01 * .998)
    boost_high = .612 * (1 + 18.01 * 1.002)
    gate_min = boost_low - .45 - .2 - .15
    criteria = {
        "ov_trip_guard_at_least_200mv": margin >= .2 and bov_margin >= 1,
        "epr_tvs_clamp_below_powered_protector_abs_max": epr_gate_rc["powered_surge_from_epr"]["measurements"]["peak_v"] <= 59,
        "epr_voltage_below_unpowered_protector_abs_max": all(
            epr_gate_rc[name]["measurements"]["peak_v"] <= 22
            for name in ("unpowered_hotplug_from_zero", "unpowered_step_from_spr")),
        "epr_power_loss_below_unpowered_protector_abs_max": epr_power_loss["measurements"]["unpowered_peak_v"] <= 22,
        "b2_pv_only_has_aux_power_path": pv_startup_ok,
        "disabled_boost_output_disconnected": boost_disconn_ok,
        "illustrative_low_corner_gate_at_least_10v": gate_min >= 10.35 and boost_high < 12,
        "bq_default_cv_upper_below_37p45v": bq_upper_no_fbg_min < 37.45,
        "bbm_deadtime_at_least_5ms": all(case["remaining_deadtime_with_500us_total_off_s"] > .005 for case in bbm.values()),
    }
    return {
        "status": "PASS_MODEL" if model_pass else "FAIL_MODEL",
        "design_status": "PASS" if all(criteria.values()) else "BLOCKED",
        "criteria": criteria,
        "regen_window": {
            "bq_divider_only_nominal_v": 1.536 * (1 + 230e3 / 10e3),
            "bq_nominal_v_with_33ohm_fbg": bq_nominal,
            "bq_upper_v_without_guaranteed_fbg_min": bq_upper_no_fbg_min,
            "bq_register_output_step_v_with_33ohm_fbg": .002 * (1 + 230e3 / (10e3 + 33)),
            "running_hardware_default_nominal_v": bq_nominal, "parked_calibrated_target_range_v": [37.65, 37.70],
            "parked_hard_ceiling_v": 37.80, "min_trip_margin_above_parked_ceiling_v": margin,
            "min_frozen_bov_margin_v": bov_margin,
            "ov_divider_power_mw_at_36v": 1000 * 36**2 / 1162e3, "ov_corners": ov,
        },
        "aux": {"nominal_pre_5v_and_drv_v": [3.60, 5.10, 11.406],
                "ideal_buck_duty_at_4p65v": 3.60 / 4.65,
                "gate_rail_corners_v": [boost_low, boost_high], "illustrative_low_corner_vgs_v": gate_min,
                "boost_disabled_negative_control": boost_off, "boost_post_disable_discharge": boost_discharge,
                "pv_or_front_end_screen": pv_or},
        "epr_power_loss_screen": epr_power_loss,
        "epr_gate_rc_screen": epr_gate_rc,
        "bbm_rc_screen": bbm,
        "scope": "B4a circuit-level screens using locked B4 parameters (10 ohm series limiter, active PNP fast turn-off, "
                 "10 nF prot cap with self-powered 100 ohm active bleeder, 230k/10k BQ default divider, 220 nF X7R BBM, "
                 "4.7 uF + active 100 ohm 10V_DRV bleeder, SS310 PV-to-AUX_HV_IN behavioral startup). "
                 "Behavioral switches/diodes do not replace physical bench thermal/hot-plug qualification.",
    }


def pd_modes(work):
    results = []
    fet_r_25c = .0053
    assumed_hot_multiplier = 1.6
    dcr_25c = .0101
    assumed_hot_dcr = .018
    shunt_r = .002
    saturation_a, rms_rating_a = 21, 11
    overlaps_s, gate_charge_c = 100e-9, 73e-9
    l_min = 8e-6
    corners = [(volts, 3 if volts == 5 else 5, bus, 250_000)
               for volts in (5, 9, 12, 15, 20, 28, 36, 48) for bus in (22.5, 37.6)]
    corners.append((48, 5, 22.5, 212_000))
    for vin, imax, vout, fs in corners:
        delivered = vin * imax * .90
        inductor_avg = imax if vin < vout else delivered / vout
        seconds = 1 / fs
        buck = vin > vout
        duty = vout / vin if buck else 1 - vin / vout
        hi, lo = ((vin - vout), -vout) if buck else (vin, vin-vout)
        ripple = hi * duty * seconds / l_min
        peak = inductor_avg + ripple / 2
        case = simulate(work, f"pd_{vin}_{int(vout * 10)}_{fs}", f"""
Vg gate 0 PULSE(1 0 {duty*seconds:.9g} .2n .2n {(1-duty)*seconds-4e-10:.9g} {seconds:.9g})
Bstress sw 0 V={lo:.9g}+({hi:.9g}-({lo:.9g}))*V(gate)
L current 0 {l_min:.9g} ic={inductor_avg-ripple/2:.9g}
Vsense sw current 0
Bsq sar 0 V=I(Vsense)^2
""", "10n", f"{seconds*8:.9g}", {
            "inductor_peak_a": f"max i(Vsense) from={seconds*5:.9g} to={seconds*7:.9g}",
            "inductor_min_a": f"min i(Vsense) from={seconds*5:.9g} to={seconds*7:.9g}",
            "mean_squared_a2": f"avg v(sar) from={seconds*5:.9g} to={seconds*7:.9g}",
        })
        m = case["measurements"]
        m["inductor_rms_a"] = math.sqrt(m["mean_squared_a2"])
        m["isat_headroom_a"] = saturation_a - m["inductor_peak_a"]
        m["irms_headroom_a"] = rms_rating_a - m["inductor_rms_a"]
        loss = {
            "two_fets_25c_w": m["mean_squared_a2"] * 2 * fet_r_25c,
            "two_fets_assumed_hot_w": m["mean_squared_a2"] * 2 * fet_r_25c * assumed_hot_multiplier,
            "inductor_copper_25c_w": m["mean_squared_a2"] * dcr_25c,
            "inductor_copper_assumed_hot_w": m["mean_squared_a2"] * assumed_hot_dcr,
            "input_and_output_shunts_w": shunt_r * (imax**2 + (delivered / vout)**2),
            "hard_switch_overlap_screen_w": .5 * max(vin, vout) * inductor_avg * overlaps_s * fs,
            "two_switching_gates_at_10v_w": 2 * 10 * gate_charge_c * fs,
        }
        passed = abs(m["inductor_peak_a"] - peak) < .03 and min(m["isat_headroom_a"], m["irms_headroom_a"]) > 0
        results.append({
            "input_v": vin, "pdo_current_a": imax, "bus_v": vout, "output_power_w_at_90pct": delivered,
            "loss_budget_at_90pct_w": vin * imax - delivered, "frequency_hz": fs, "loss_screen": loss,
            "mode": "buck" if buck else "boost", "theoretical_peak_a": peak,
            "status": "PASS_MODEL" if passed else "FAIL_MODEL", **case,
        })
    return {"status": "PASS_MODEL" if all(case["status"] == "PASS_MODEL" for case in results) else "FAIL_MODEL",
            "cases": results,
            "aux_5v_buck_negative_control": {"usb_nominal_v": 5.0, "assumed_schottky_drop_v": .35,
                                            "aux_after_diode_v": 4.65, "required_buck_output_v": 5.0,
                                            "required_ideal_duty": 5 / 4.65,
                                            "status": "CANNOT_REGULATE_5V_FROM_5V_ONLY"},
            "assumptions": {"inductor": "CODACA CSBX1275-100M", "isat_a": saturation_a, "irms_a": rms_rating_a,
                            "inductance_worst_h": l_min, "efficiency_for_budget": .90,
                            "mosfet": "HUAYI HYG053N10NS1C2", "vgs_v": 10, "fet_r_25c_max_ohm": fet_r_25c,
                            "assumed_hot_r_multiplier": assumed_hot_multiplier, "gate_charge_typ_c": gate_charge_c,
                            "rise_plus_fall_typ_s": overlaps_s, "inductor_dcr_max_25c_ohm": dcr_25c,
                            "assumed_hot_inductor_dcr_ohm": assumed_hot_dcr, "each_shunt_assumed_ohm": shunt_r},
            "scope": "Ideal buck/boost switching-cell stress screen with 90% output power assumption, not BQ control or thermal validation. "
                     "Input cases are illustrative, not evidence of PDO/cable support or 5 V cold start. Isat is a datasheet L-drop point, "
                     "Irms a temperature-rise point, neither is a hard board limit. The switching screen omits Coss, reverse recovery, "
                     "deadtime, overlap-mode operation, core loss, transient loading and junction rise."}


def energy_budget():
    pack_wh = 9 * 3.6 * 15
    survival_reserve_wh = pack_wh * (0.25 - 0.10)
    cold_sensitivity_wh = survival_reserve_wh * .65
    baseline_mw = (80, 130)
    planning_mw = 50
    solar = {str(hours): 120 * hours * .65 for hours in (.0, .5, 1., 4., 5.)}
    heater_wh = 40 * .5
    sleep_wh = planning_mw * 24 / 1000
    daily_logic = .75 * 4
    return {
        "status": "CONDITIONAL_BUDGET_ONLY", "nominal_pack_wh": pack_wh,
        "survival_soc_fraction_start_end": [.25, .10], "usable_reserve_wh_nominal": survival_reserve_wh,
        "usable_reserve_wh_with_65pct_capacity_sensitivity": cold_sensitivity_wh,
        "tier0_board6_only": {"target_exclusive_mw": 5, "status": "DESIGN_TARGET_NOT_MEASURED",
                              "scope": "Includes dark-state bias, sensing, watchdog and all board leakage."},
        "tier0_vehicle_frozen_bms_run": {"estimate_mw": baseline_mw, "status": "ESTIMATE_NOT_MEASURED",
                                        "bms_can": "CA-IS2062A remains on VOUT_5V",
                                        "wh_per_day": [mw * 24 / 1000 for mw in baseline_mw],
                                        "cold_zero_sun_days": [cold_sensitivity_wh / (mw * .024)
                                                               for mw in reversed(baseline_mw)]},
        "tier0_vehicle_bms_rev2": {"target_exclusive_mw": planning_mw, "status": "PLAN_NOT_IMPLEMENTED",
                                  "prerequisite": "Switch off BMS CAN rail; no modification to frozen BMS."},
        "tier0_consumption_wh_per_day_at_50mw": sleep_wh,
        "solar_net_harvest_wh_by_equivalent_sun_hours": solar,
        "example_heater_w": 40, "example_heater_operating_h": .5, "example_heater_wh": heater_wh,
        "daily_tier1_controller_wh_if_4h_irradiated": daily_logic,
        "sunny_day_remaining_wh_at_50mw_planning_after_heater_sleep_and_tier1": [
            solar[str(hours)] - heater_wh - sleep_wh - daily_logic for hours in (4., 5.)
        ],
        "zero_sun_days_at_50mw_nominal_and_65pct": [
            survival_reserve_wh / sleep_wh, cold_sensitivity_wh / sleep_wh
        ],
        "zero_sun_days_at_100mw_and_65pct": cold_sensitivity_wh / 2.4,
        "break_even_sun_hours_daily_with_heater_and_tier1": (heater_wh + sleep_wh + daily_logic) / (120 * .65),
        "limitations": "No Tier 2 work, measured leakage, pack self-discharge or measured temperature-capacity curve included. "
                       "No solar means heater stays off. The 13.9-day lunar-night reference is metaphorical; local inputs govern. "
                       "Rated heater power requires enough instantaneous solar, compatible voltage and measured thermal coupling.",
    }


def one_shot_risk_screens(work):
    cold = simulate(work, "one_shot_cold_assist", """
.model QNPN NPN(IS=1e-14 BF=150 VAF=100)
.model QPNP PNP(IS=1e-14 BF=200 VAF=100)
.model DSS310 D(IS=2e-6 N=1.05 RS=0.05)
.model DBAT46 D(IS=5e-7 N=1.05 RS=0.4)
.subckt TLV431 ref cat ano
Bik cat ano I=5e-7 + 0.02*(1/(1+exp(-(V(ref,ano)-1.240)/0.002)))*u(V(cat,ano)-1.24)
Cka cat ano 10p
.ends TLV431
Vusb pd_raw 0 PWL(0 0 0.5m 4.50 15m 4.50 15.5m 4.75 30m 4.75 30.5m 5.25 45m 5.25 45.5m 9.0 60m 9.0 60.5m 48.0 75m 48.0 75.5m 82.0 85m 82.0)
Daux pd_raw aux_hv DSS310
Caux aux_hv 0 4.7u
Vpass_s aux_hv pass_e 0
Rbe aux_hv pass_b 10k
Qpass pre3v6 pass_b pass_e QPNP
Rb pass_b drv_c 1k
Qdrv drv_c drv_b 0 QNPN
Rstart aux_hv drv_b 100k
Rov_top aux_hv ov_b 100k
Rov_bot ov_b 0 10k
Qov drv_b ov_b 0 QNPN
Rfb_top pre3v6 ref_n 20k
Rfb_bot1 ref_n fb_mid 10k
Rfb_bot2 fb_mid 0 1k
Xref ref_n tlv_k 0 TLV431
Rerr_be pre3v6 tlv_k 2k
Qerr err_c tlv_k pre3v6 QPNP
Rerr_c err_c shunt_b 10k
Qshunt drv_b shunt_b 0 QNPN
Ccomp tlv_k ref_n 100n
Bbuck 0 pre3v6 I=max(0,(3.600-V(pre3v6))*2.0)*u(V(aux_hv)-4.7)
Cpre pre3v6 0 47u
Rload pre3v6 0 40
Dvpwr pre3v6 vpwr DBAT46
Cvpwr vpwr 0 10u
Rvpwr vpwr 0 1.6k
Bp_cold pcold 0 V=abs((V(aux_hv)-V(pre3v6))*I(Vpass_s))
""", "20u", "85m", {
        "vpre_4v5": "avg v(pre3v6) from=10m to=14.5m",
        "vvpwr_4v5": "avg v(vpwr) from=10m to=14.5m",
        "vpre_4v75": "avg v(pre3v6) from=25m to=29.5m",
        "vpre_9v": "avg v(pre3v6) from=55m to=59.5m",
        "vtlv_max": "max v(tlv_k) from=0 to=85m",
        "vdrvb_48v": "avg v(drv_b) from=65m to=74m",
        "vdrvb_82v": "avg v(drv_b) from=78m to=84m",
        "pqpass_4v75": "avg v(pcold) from=25m to=29.5m",
    })
    m = cold["measurements"]
    cold["status"] = "PASS_MODEL" if (m["vpre_4v5"] >= 3.40 and m["vvpwr_4v5"] >= 3.15
        and m["vtlv_max"] <= 4.0 and m["vdrvb_48v"] <= .1 and m["vdrvb_82v"] <= .1
        and m["pqpass_4v75"] <= .15) else "FAIL_MODEL"

    acuv_nom = 1.15 * (1 + 100e3 / 40.1e3)
    acuv_high = 1.20 * (1 + 100e3 * 1.001 / (40.1e3 * .999))
    fb = 1.566 * .993
    top = 230e3 * .999
    bottom = 10e3 * 1.001 + 55
    trimmed_bottom = 1 / (1 / (10e3 * 1.001) + 1 / (1e6 * 1.001)) + 55
    cv_untrimmed = fb * (1 + top / bottom)
    cv_trimmed = fb * (1 + top / trimmed_bottom)
    dac_for_37p2 = 37.2 / (1 + top / trimmed_bottom)
    max_code_cv = 1.566 * (1 + top / trimmed_bottom)
    corners = {"acuv_nominal_v": acuv_nom, "acuv_upper_v": acuv_high,
               "usb_4v75_margin_v": 4.75 - acuv_high,
               "cv_untrimmed_v": cv_untrimmed, "cv_trimmed_v": cv_trimmed,
               "dac_vfb_for_37p2_v": dac_for_37p2, "max_code_cv_at_corner_v": max_code_cv, "vfb_max_v": 1.566,
               "assumptions": "CV low corner: 230k top -0.1%, 10k bottom and 1M trim +0.1%, 55 ohm series FBG, VFB at 1.566 V minus 0.7%; DAC voltage is calibrated at this corner."}
    corners["status"] = "PASS_MODEL" if (acuv_high <= 4.21 and 4.75 - acuv_high > .54
        and 37.04 < cv_untrimmed < 37.07 and cv_trimmed > 37.20
        and cv_trimmed <= 37.80 and dac_for_37p2 <= 1.566) else "FAIL_MODEL"

    switches = {}
    for label, voltage in (("pd_5v", 5), ("xt60_48v", 48)):
        switches[label] = simulate(work, f"one_shot_vom_{label}", f"""
Vsrc src 0 {voltage}
Rwire src src_w 0.05
Ivom vs_n gate PWL(0 0 0.49m 0 0.50m 15u 15m 15u)
Bclamp gate vs_n I=max(0,(V(gate,vs_n)-8.4)*0.01)
Rgs_lkg gate vs_n 100meg
Rg1 gate g1 22
Rg2 gate g2 22
M1 src_w g1 vs_n vs_n QMOS
D1 vs_n src_w DMOD
M2 out g2 vs_n vs_n QMOS
D2 vs_n out DMOD
Cbulk out bulk_esr 470u ic=0
Resr bulk_esr 0 0.04
Cmlcc out 0 14u ic=0
Rload out 0 1k
Bpm pm 0 V=abs(V(src_w,out)*(-I(Vsrc)))
Bi_probe iprobe 0 V=-I(Vsrc)
.model QMOS NMOS(LEVEL=1 VTO=2.0 KP=80 RD=0.004 RS=0.003 CGSO=1.8e-9 CGDO=1.2e-10)
.model DMOD D(IS=1e-9 N=1.1 RS=0.01)
""", "20u", "15m", {"i_peak_a": "max v(iprobe) from=0.5m to=15m",
        "p_peak_w": "max v(pm) from=0.5m to=15m",
        "e_total_j": "integ v(pm) from=0.5m to=15m",
        "t_90_s": f"when v(out)={voltage * .9} rise=1"})
    switches["pd_5v"]["status"] = "PASS_MODEL" if switches["pd_5v"]["measurements"]["e_total_j"] < .015 else "FAIL_MODEL"
    switches["xt60_48v"]["operational_rule"] = "Connect XT60 with source disabled or pre-ramped, or verify 48 V hot-plug SOA on bench."
    switches["status"] = switches["pd_5v"]["status"]
    switches["scope"] = "Simplified MOSFET model and lumped energy are not a device SOA or hot-plug qualification."

    rms2 = pv_power_stage_screen()["rms_a"] ** 2
    sku = {"rms_a_squared": rms2, "hyg080n10ls1c2_w_per_fet": rms2 * .0069,
           "hyg053n10ns1c2_w_per_fet": rms2 * .0053,
           "delta_w_per_fet": rms2 * (.0069 - .0053),
           "ina293a1_common_mode_v": 110, "comparison_common_mode_v": 26,
           "bat46w_vrrm_v": 100, "comparison_vrrm_v": 70,
           "scope": "60 W, 12 V to 37.6 V, 100 kHz; full inductor RMS through each FET is a conservative comparison at 10 V gate and typical resistance, not thermal validation."}
    sku["status"] = "PASS_MODEL" if sku["delta_w_per_fet"] < .06 else "FAIL_MODEL"
    checks = {"usb_5v_cold_start_assist": cold, "bq_acuv_and_cv_corners": corners,
              "vom1271_switch_inrush_and_soa": switches, "pv_mosfet_merge_and_sku_summary": sku}
    return {"status": "PASS_MODEL" if all(x["status"] == "PASS_MODEL" for x in checks.values()) else "FAIL_MODEL", **checks}


def main():
    version = subprocess.run([SPICE, "--version"], capture_output=True, check=True, text=True).stdout
    if "ngspice-47" not in version:
        raise RuntimeError(f"Expected ngspice-47. {version}")
    tasks = {"solar_3_channel": pv_model, "dark_to_sun_bus_zero": cold_start,
             "recovery_interface_ld_probe": ld_wake_probe,
             "recovery_interface_charge_window": charge_window_model,
             "regen_cutback": regen_window, "usb_pd_boundaries": pd_modes,
             "b2_lock_review": b2_lock_review, "one_shot_risk_screens": one_shot_risk_screens}
    with TemporaryDirectory(prefix="ev_mppt_spice_") as temp, ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        runs = {key: pool.submit(fn, Path(temp)) for key, fn in tasks.items()}
        scenarios = {key: call.result() for key, call in runs.items()}
    report = {
        "status": "BOUNDARY_SIMULATION_COMPLETE_FINAL_SCHEMATIC_PENDING",
        "engine": {"version": "ngspice-47", "binary": SPICE.relative_to(POWER.parent).as_posix(),
                   "sha256": sha256(SPICE.read_bytes()).hexdigest()},
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": scenarios, "energy": energy_budget(),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    summary = {key: value.get("status", value.get("pack_charge")) for key, value in scenarios.items()}
    summary["b2_design"] = scenarios["b2_lock_review"]["design_status"]
    print(json.dumps(summary, ensure_ascii=False))
    probe = scenarios["recovery_interface_ld_probe"]["cases_by_assumed_load_ma"]
    return int(summary["b2_design"] != "PASS" or
               scenarios["one_shot_risk_screens"]["status"] != "PASS_MODEL" or
               scenarios["solar_3_channel"]["status"] != "PASS_MODEL" or
               scenarios["dark_to_sun_bus_zero"]["local_bias"] != "PASS_MODEL" or
               scenarios["dark_to_sun_bus_zero"]["isolation_negative_control"] != "PASS_MODEL" or
               not probe["10"]["vld_threshold_crossing_supported_in_model"] or
               probe["60"]["vld_threshold_crossing_supported_in_model"] or
               any(scenarios[name]["status"] != "PASS_MODEL" for name in (
                   "recovery_interface_charge_window", "regen_cutback", "usb_pd_boundaries", "b2_lock_review")))


if __name__ == "__main__":
    raise SystemExit(main())
