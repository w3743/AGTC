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
                  irradiance_inputs=PV, scope="P&O on idealized single-knee I–V curves, averaged power stage, "
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
    results = {}
    for tau_us in (5, 30, 200):
        tau = tau_us * 1e-6
        case = simulate(work, f"regen_tau_{tau_us}", f"""
Vbus bus 0 PWL(0 37.55 100u 37.55 103u 37.8 120u 39 350u 39)
Bhard in 0 V=4*u(37.6-V(bus))
Rresponse in resp 1k
Cresponse resp 0 {tau/1000:.9g} ic=4
Bout 0 sense I=max(0,V(resp))
Vinject sense bus 0
""", ".1u", "350u", {
            "t_cv_s": "when v(bus)=37.6 rise=1",
            "t_below_100ma_s": f"when i(Vinject)=0.1 fall=1 td=100u",
            "i_at_39v_a": "find i(Vinject) at=120u",
            "tail_charge_c": "integ i(Vinject) from=100.6u to=350u",
        } if tau_us != 200 else {
            "t_cv_s": "when v(bus)=37.6 rise=1",
            "i_at_39v_a": "find i(Vinject) at=120u",
            "i_at_350us_a": "find i(Vinject) at=350u",
            "tail_charge_c": "integ i(Vinject) from=100.6u to=350u",
        })
        m = case["measurements"]
        if "t_below_100ma_s" in m:
            m["time_to_below_100ma_us"] = 1e6 * (m["t_below_100ma_s"] - m["t_cv_s"])
        results[str(tau_us)] = case
    return {
        "status": "TRANSFER_TIME_UNVERIFIED", "cases_by_assumed_time_constant_us": results,
        "scope": "4 A combined charger action, ideal 37.6 V threshold, injected positive current is ramped down "
                 "through an assumed lag. No BQ25756 CV loop or output disconnect delay is embedded. Effective zero = <0.1 A. "
                 "37.8 V mixed-mode starting bus already exceeds the nominal 37.6 V running setpoint.",
    }


def pd_modes(work):
    results = []
    loss_r = 2 * .006 + .018 + .002
    fs = 250_000
    l_min = 8e-6
    for vin, imax, vout in [(5, 3, 37.6), (28, 5, 22.5), (28, 5, 37.6),
                           (48, 5, 22.5), (48, 5, 37.6)]:
        delivered = vin * imax * .90
        inductor_avg = imax if vin < vout else delivered / vout
        seconds = 1 / fs
        buck = vin > vout
        duty = vout / vin if buck else 1 - vin / vout
        hi, lo = ((vin - vout), -vout) if buck else (vin, vin-vout)
        ripple = hi * duty * seconds / l_min
        peak = inductor_avg + ripple / 2
        case = simulate(work, f"pd_{vin}_{str(vout).replace('.', '_')}", f"""
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
        m["conduction_only_loss_w"] = m["mean_squared_a2"] * loss_r
        results.append({
            "input_v": vin, "pdo_current_a": imax, "bus_v": vout, "output_power_w_at_90pct": delivered,
            "loss_budget_at_90pct_w": vin * imax - delivered,
            "mode": "buck" if buck else "boost", "theoretical_peak_a": peak,
            "nominal_10a_as_peak": "FAIL" if m["inductor_peak_a"] > 10 else "PASS_MODEL", **case,
        })
    return {"status": "THERMAL_AND_SWITCHING_UNVERIFIED", "cases": results,
            "assumptions": {"frequency_hz": fs, "inductance_worst_h": l_min, "efficiency_for_budget": .90,
                            "two_hot_fets_ohm_each": .006, "inductor_dcr_ohm": .018, "shunt_ohm": .002},
            "scope": "Ideal inductor voltage stress and conduction-loss screen, not a BQ25756 power/thermal model. "
                     "5 V 3 A is conditional on negotiated/advertised 3 A. Check 4.35 V VINDPM before enabling full current. "
                     "Switching, deadtime, AC/core loss and junction rise require exact parts and layout."}


def energy_budget():
    pack_wh = 9 * 3.6 * 15
    survival_reserve_wh = pack_wh * (0.25 - 0.10)
    cold_sensitivity_wh = survival_reserve_wh * .65
    sleep = {"AFE_and_RTC": 2, "BMS_quiescent_regulator_and_sleep_controller": 10,
             "MPPT_source_detection_and_sleep": 5, "hardware_watchdog_and_LP_relay_interfaces": 3,
             "pack_sense_leakage_and_other_ALWAYS_ON": 20, "margin": 10}
    solar = {str(hours): 120 * hours * .65 for hours in (.0, .5, 1., 4., 5.)}
    heater_wh = 40 * .5
    sleep_wh = sum(sleep.values()) * 24 / 1000
    daily_logic = .75 * 4
    return {
        "status": "CONDITIONAL_BUDGET_ONLY", "nominal_pack_wh": pack_wh,
        "survival_soc_fraction_start_end": [.25, .10], "usable_reserve_wh_nominal": survival_reserve_wh,
        "usable_reserve_wh_with_65pct_capacity_sensitivity": cold_sensitivity_wh,
        "tier0_allocations_mw_not_measured": sleep, "tier0_consumption_wh_per_day_at_50mw": sleep_wh,
        "solar_net_harvest_wh_by_equivalent_sun_hours": solar,
        "example_heater_w": 40, "example_heater_operating_h": .5, "example_heater_wh": heater_wh,
        "daily_tier1_controller_wh_if_4h_irradiated": daily_logic,
        "sunny_day_remaining_wh_after_heater_sleep_and_tier1": [
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


def main():
    version = subprocess.run([SPICE, "--version"], capture_output=True, check=True, text=True).stdout
    if "ngspice-47" not in version:
        raise RuntimeError(f"Expected ngspice-47. {version}")
    tasks = {"solar_3_channel": pv_model, "dark_to_sun_bus_zero": cold_start,
             "recovery_interface_ld_probe": ld_wake_probe,
             "recovery_interface_charge_window": charge_window_model,
             "regen_cutback": regen_window, "usb_pd_boundaries": pd_modes}
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
    print(json.dumps({key: value.get("status", value.get("pack_charge"))
                      for key, value in scenarios.items()}, ensure_ascii=False))
    return int(scenarios["solar_3_channel"]["status"] != "PASS_MODEL" or
               scenarios["dark_to_sun_bus_zero"]["local_bias"] != "PASS_MODEL" or
               scenarios["dark_to_sun_bus_zero"]["isolation_negative_control"] != "PASS_MODEL" or
               scenarios["recovery_interface_charge_window"]["status"] != "PASS_MODEL")


if __name__ == "__main__":
    raise SystemExit(main())
