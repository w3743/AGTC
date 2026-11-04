"""Periodic ripple reconstruction and ideal energy/RC interface checks."""
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from 双FOC母线分流筛查 import BULK, BULK_MOUNT, CENTRAL, NEAR, ROOT, SOURCE, solve

PWM = 20_000
FUNDAMENTAL = 500
SAMPLES_PER_PWM = 256
DT = 1 / (PWM * SAMPLES_PER_PWM)
SAMPLE = np.arange(PWM // FUNDAMENTAL * SAMPLES_PER_PWM)
TIME = SAMPLE * DT
FREQ = np.fft.rfftfreq(len(TIME), DT)
W = 2 * np.pi * FREQ[1:]


def foc_ripple(volts, shift):
    carrier_index = SAMPLE + round(shift * SAMPLES_PER_PWM)
    phase = 2 * np.pi * FUNDAMENTAL / PWM * (carrier_index // SAMPLES_PER_PWM + .5 - shift)
    angles = phase + 2 * np.pi / 3 * np.arange(3)[:, None]
    sinusoid = np.sin(angles)
    carrier = 2 * np.abs(carrier_index % SAMPLES_PER_PWM / SAMPLES_PER_PWM - .5)
    duty = .5 + .5 * (1800 / (.75 * volts * 120)) * sinusoid
    return np.sum(120 * sinusoid * (carrier < duty), axis=0)


def branch_admittance(capacitor, count=1, mount=(0, 0)):
    c, r, l = capacitor
    return 1 / ((r + 1j * (W * l - 1 / (W * c))) / count + mount[0] + 1j * W * mount[1])


def branch_metrics(spectrum, esr=None, count=1):
    waveform = np.fft.irfft(np.r_[0, spectrum], n=len(TIME))
    rms = float(np.sqrt(np.mean(waveform**2)))
    power_per_bin = 2 * abs(spectrum / len(TIME))**2
    power_per_bin[-1] *= .5
    rms_bands = [float(np.sqrt(sum(power_per_bin[mask]))) for mask in (
        FREQ[1:] < 10_000, (FREQ[1:] >= 10_000) & (FREQ[1:] < 100_000), FREQ[1:] >= 100_000)]
    assert math.isclose(sum(x*x for x in rms_bands), rms*rms, rel_tol=1e-6, abs_tol=1e-12)
    result = {"rms": rms, "p2p": float(np.ptp(waveform)), "absPeak": float(max(abs(waveform))),
              "bandRms": rms_bands}
    if esr:
        result.update({"eachRmsAtEqualShare_A": rms / count,
                       "eachLossAtModelEsr_W": (rms / count)**2 * esr})
    return result


def ripple(volts, pair_nh, count, carrier_shift, regulator_pwm, battery, external_esr):
    cap = (BULK[0], external_esr, BULK[2])
    yn = branch_admittance(NEAR)
    yb = branch_admittance(cap, count, BULK_MOUNT)
    yc = branch_admittance(CENTRAL)
    yw = 1 / (2e-3 + 1j * W * pair_nh * 1e-9)
    ys = 1 / (SOURCE[0] + 1j * W * SOURCE[1]) if battery else 0
    local = yn + yb + yw
    # Periodic forced response. Strip the mean since DC supply/regen is checked separately.
    reg = 80 * ((SAMPLE * regulator_pwm) % (PWM * SAMPLES_PER_PWM) < PWM * SAMPLES_PER_PWM / 2) \
        if regulator_pwm else np.zeros_like(TIME)
    injections = np.stack((reg, foc_ripple(volts, 0), foc_ripple(volts, carrier_shift)))
    center, left, right = np.fft.rfft(injections)[:, 1:]
    vc = (center + yw / local * (left + right)) / (yc + ys + 2 * yw * (yn + yb) / local)
    va, vb = (left + yw * vc) / local, (right + yw * vc) / local
    wa, wb = (va - vc) * yw, (vb - vc) * yw
    np.testing.assert_allclose(center + wa + wb, (yc + ys) * vc, atol=1e-9)
    np.testing.assert_allclose(left, (yb + yn) * va + wa, atol=1e-9)
    np.testing.assert_allclose(right, (yb + yn) * vb + wb, atol=1e-9)
    # Discrete sampling leaves an unpaired Nyquist coefficient. Omit that single bin.
    def metrics(spectrum, esr=None, qty=1):
        spectrum = spectrum.copy()
        spectrum[-1] = 0
        return branch_metrics(spectrum, esr, qty)
    if external_esr == BULK[1]:
        harmonic = int(PWM / FUNDAMENTAL) - 1
        ref = solve(PWM, pair_nh, count, (1, 0, 0), battery)
        normalized = 1 / (yc + ys + 2 * yw * (yn + yb) / local)
        assert abs(abs((normalized * yc)[harmonic]) -
                   ref["current_A"]["regulator_bank"]["magnitude"]) < 1e-7
    return {
        "case": {"volts": volts, "pair_nH": pair_nh, "capCountEachFOC": count,
                 "rightCarrierOffsetPeriods": carrier_shift, "regulatorHz": regulator_pwm,
                 "batterySmallSignalBranch": battery, "eachBulkEsr_ohm": external_esr},
        "switchingInputArms": [metrics(x)["rms"] for x in (center, left, right)],
        "bankCurrent_A": {"central": metrics(vc*yc, CENTRAL[1]*6, 6),
                          "bulkA": metrics(va*yb, external_esr, count),
                          "bulkB": metrics(vb*yb, external_esr, count),
                          "nearA": metrics(va*yn), "nearB": metrics(vb*yn),
                          "wireA": metrics(wa), "wireB": metrics(wb), "battery": metrics(vc*ys)},
        "busRipple_V": {"central": metrics(vc), "FOC_A": metrics(va), "FOC_B": metrics(vb)},
        "localBovSenseRipple_V": metrics(va / (101 * (1 + 1j * W * 1e-9 / (1/1e6 + 1/10e3)))),
    }


def precharge(nominal_vehicle, upper_vehicle):
    # 72 mF is FOC-only; the all-board high case also includes the upstream pump.
    rows = []
    corners = ((8.726, 1.512), (8.726, .5), (8.86, .9))
    for (resistance, limit), capacitance, pack, silent in product(
        corners, (.060, .0655334, .072, nominal_vehicle, upper_vehicle),
        (22.5, 37.8), (0, .002, .020, .025, .050, .075, .100),
    ):
        drop = resistance * silent
        tau = resistance * capacitance
        time = tau * math.log((pack-drop)/(limit-drop)) if drop < limit else None
        rows.append({"C_F": capacitance, "Vpack": pack, "R_ohm": resistance, "Iquiet_A": silent,
                     "finalPackMinusBus_V": drop, "deltaTarget_V": limit, "tToDelta_s": time,
                     "deltaAt2p346s_V": drop + (pack-drop)*math.exp(-2.346/tau),
                     "deltaAt2p738s_V": drop + (pack-drop)*math.exp(-2.738/tau),
                     "E_0_to_pack_J": .5 * capacitance * pack**2})
    return rows


def fault_storage():
    # Compare both feeders still connected and an isolated FOC; open circuits use local caps.
    return [{"case": path, "C_F": cap, "Pnet_W": power, "I_initial_A": power / 37.8,
             "time37p8_to_40p3_ms": 1000*.5*cap*(40.3**2 - 37.8**2) / power,
             "v_after_5ms": math.sqrt(37.8**2 + 2*power*.005/cap)}
            for path, cap in (("connected nominal", .0654), ("connected capacitance low corner", .05286),
                              ("isolated FOC nominal", .030), ("isolated FOC -20%", .024))
            for power in (250, 500, 1000)]


def braking_examples():
    mass = 30
    speed = 15 / 3.6
    deceleration = 3.9624
    time = speed / deceleration
    distance = .5 * speed * time
    # Gravity acts along the path. These examples omit rotation, drag and losses.
    examples = []
    for descent in (0, 10, 20):
        gravity = 9.80665 * math.sin(math.radians(descent))
        peak = mass * (deceleration + gravity) * speed
        energy = .5 * mass * speed**2 + mass * gravity * distance
        examples.append({"downSlope_deg": descent, "peakPower_W": peak, "stopEnergy_J": energy,
                         "peakBusCurrent_A_if_unity_efficiency": [peak / 37.8, peak / 22.5],
                         "gravityPowerAtConstantSpeed_W": mass * gravity * speed})
    return {"mass_kg": mass, "speed_m_s": speed, "deceleration_m_s2": deceleration,
            "duration_s": time, "distance_m": distance, "examples": examples,
            "batteryMaxIfFullCCL_W_atPack22p5_37p8": [22.5*16, 37.8*16],
            "fullPackOrCCLUnavailable_W": 0,
            "centralDumpStaticAt40p3V": {
                "current_A": [40.3/(.82*1.05), 40.3/(.82*.95)],
                "power_W": [40.3**2/(.82*1.05), 40.3**2/(.82*.95)],
            }}


def main():
    cases = [ripple(33.3333333333, wire, count, shift, 0, battery, esr)
             for wire in (50, 150, 300) for count in (2, 3) for shift in (0, .5)
             for battery in (False, True) for esr in (.012, .023)]
    cases += [ripple(v, 150, 3, shift, regulator, False, .012)
              for v in (22.5, 33.3333333333, 37.8) for shift in (0, .5) for regulator in (20_000, 100_000)]
    source = ROOT / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
    bms_input = ROOT.parent / "BMS模块/仿真/结果/跨板母线预充.json"
    regulator_input = ROOT.parent / "稳压模块/检查/本轮磁盘读回.enet"
    bms = json.loads(bms_input.read_text(encoding="utf-8-sig"))
    small_per_foc_uf = sum((14.4, 2.2, .1, .1, 4.7, 33, 2.2, 2.2))
    foc_population = next(row for row in bms["populations_uF"] if row["board"] == "FOC-A and B")
    bms_small_uf = foc_population["direct_small_nominal_uF"]
    correction_f = (2*small_per_foc_uf - bms_small_uf)*1e-6
    nominal_vehicle = bms["nominal_F"] + correction_f
    upper_vehicle = bms["upper_initial_tolerance_F"] + 1.2*correction_f
    result = {"status": "MODEL_AND_CONDITIONAL_SCREENS_ONLY",
              "FOC_ENET_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "crossBoardInputsSha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (bms_input, regulator_input)},
              "FOCSmallCapsNominal_uF_each": small_per_foc_uf,
              "BMSCrossCheck": {"originalSmallCaps_uF_dualFOC": bms_small_uf,
                                "correctionNominal_F": correction_f,
                                "vehicleNominal_F_afterCorrection": nominal_vehicle,
                                "vehicleInitialToleranceScreen_F_afterCorrection": upper_vehicle},
              "method": "Reconstructed periodic time-domain waveforms using FFT of three ideal switching sources and complex three-node RLC response. KCL and earlier 20 kHz transfer function checked.",
              "conditions": {"FOC_PWM_Hz": PWM, "electrical_Hz": FUNDAMENTAL, "eachPhasePeak_A": 120,
                             "dt_s": DT, "simulatedWindow_s": len(TIME)*DT,
                             "rmsBands_Hz": [[0, 10000], [10000, 100000], [100000, 1/(2*DT)]]},
              "limits": ["Ideal centered SPWM and in-phase sinusoidal phase current with frozen carrier-period angle; no measured modulation or switching-edge spectrum.",
                         "Regulator rectangular 0/80 A 50% duty injection is a sensitivity input with idealized timing, not a qualified regulator waveform.",
                         "ESR/ESL and wire impedances are assumed. Bulk uses 12 or 23 mOhm flat ESR as sensitivity, not a validated frequency/temperature limit. Thermal resistance and unequal sharing excluded.",
                         "Reconstructed voltage is AC ripple only. No DC/DC/BMS controls, MOS turn-off, TVS, mechanical brake or switching overshoot.",
                         "Ideal precharge uses constant current and inherited capacitance/timing assumptions. Additional MPPT and internal module input capacitance are excluded; comparison windows are not adopted requirements.",
                         "Fault storage uses net input power and ideal capacitors. The 40.3 V comparison point is not an allowed limit."],
              "rippleCases": cases, "prechargeCases": precharge(nominal_vehicle, upper_vehicle),
              "faultStorageCases": fault_storage(), "brakingExamples": braking_examples()}
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(cases)} periodic ripple cases, {len(result['prechargeCases'])} precharge cases, {len(result['faultStorageCases'])} fault storage cases")


if __name__ == "__main__":
    main()
