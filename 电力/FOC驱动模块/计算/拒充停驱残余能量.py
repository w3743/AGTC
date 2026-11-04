"""Bound active tail and passive bridge energy without crediting central dump."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENET = ROOT / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
V_MIN = 37.8
V_LIMIT = 39.009
C_F = 0.05286
HEADROOM = 0.5*C_F*(V_LIMIT**2-V_MIN**2)
RPM = (15/3.6)/(math.pi*0.06)*60*4
KV_SCALE_V = RPM/270
POWER = 915.0


def passive_case(inductance_uh, current_a, emf_v):
    magnetic = 2*0.75*inductance_uh*1e-6*current_a**2
    multiplier = 1/(1-emf_v/V_MIN) if emf_v < V_MIN else None
    passive = magnetic*multiplier if multiplier is not None else None
    remaining = HEADROOM-passive if passive is not None else None
    return {"L_eff_each_uH":inductance_uh, "phase_peak_each_A":current_a, "emf_LL_peak_cap_V":emf_v,
            "both_initial_magnetic_J":magnetic, "passive_bus_energy_cap_J":passive,
            "remaining_J_for_active_tail_and_other_sources":remaining,
            "both_active_max_ms_at_915W":1000*remaining/POWER if remaining is not None and remaining >= 0 else None}


def main():
    net = json.loads(ENET.read_text(encoding="utf-8-sig"))
    refs = {part["props"]["Designator"]: {pin:row.get("net") for pin,row in part["pinInfoMap"].items()}
            for part in net["components"].values()}
    assert refs["U1"]["28"] == refs["J_MCU_L"]["5"] == "BREAK_N"
    assert refs["U1"]["33"] == refs["J_MCU_R"]["6"] == "DRV_ENABLE"
    assert [refs["U1"][str(i)] for i in range(34,40)] == ["PWM_UH","PWM_UL","PWM_VH","PWM_VL","PWM_WH","PWM_WL"]
    for phase, high, low in zip("UVW",("Q1","Q3","Q5"),("Q2","Q4","Q6")):
        assert (refs[high]["2"],refs[high]["3"],refs[low]["3"]) == (f"PH_{phase}","BUS",f"PH_{phase}")
    conversions = {"line_peak":1,"line_rms_sine":math.sqrt(2),"phase_peak_sine":math.sqrt(3),
                   "phase_rms_sine":math.sqrt(6)}
    cases = [passive_case(l,i,e) for l in (20,44,80) for i in (100,120)
             for e in (0,KV_SCALE_V,KV_SCALE_V*math.sqrt(2),KV_SCALE_V*math.sqrt(6))]
    example = passive_case(44,120,KV_SCALE_V)
    remain = example["remaining_J_for_active_tail_and_other_sources"]
    result = {
        "status":"CONDITIONAL_ONLY_NO_QUALIFIED_MOTOR_OR_EXIT_TIME_BOUND",
        "FOC_ENET_SHA256":hashlib.sha256(ENET.read_bytes()).hexdigest(),
        "connected_bus":{"initial_V":V_MIN,"limit_V":V_LIMIT,"initial_tolerance_capacitance_F":C_F,
                         "available_J":HEADROOM,"central_early_sink_W":0},
        "mechanical_input":{"speed_km_h":15,"effective_wheel_diameter_m":0.06,"ratio":4,"KV_label_rpm_per_V":270,
                            "motor_rpm":RPM,"electrical_Hz":RPM*7/60,
                            "hypothetical_LL_peak_V":{k:KV_SCALE_V*v for k,v in conversions.items()},
                            "line_peak_coefficient_limit_mV_per_rpm_at_37p8V":1000*V_MIN/RPM},
        "both_boards_passive_cases":cases,
        "fixed_two_phase_loop_examples_us_without_R_or_diode_drop":{
            str(l):l*120/(V_MIN-KV_SCALE_V) for l in (40,88,160)},
        "asymmetry_example":{"both_active_peak_W":POWER,"late_board_peak_W_assumed_half":POWER/2,
                             "W0_and_passive_J":example,"common_active_ms":0.25,
                             "remaining_late_board_skew_ms":1000*max(0,remain-POWER*0.00025)/(POWER/2),
                             "max_phase_peak_A_each_at_44uH_and_2ms_both_active":
                                 math.sqrt(max(0,HEADROOM-POWER*0.002)*(1-KV_SCALE_V/V_MIN)/(1.5*44e-6))},
        "conditions":[
            "Count latency from the loss of the physical sink to actual gating, separately for each side.",
            "At full passive gating with current-continuous three-phase commutation, EMF power is bounded by line voltage span times positive rectifier current. The energy multiplier assumes no alternative drive or supply, positive dissipative losses, bus floor verified throughout, and an effective EMF ceiling that includes any loaded magnetic conversion excluded from W0.",
            "Loop times apply only to a fixed conducting phase pair. The inductances, phase currents, EMF conversions and 915 W active power are screening inputs.",
            "Use the magnetic-plus-EMF model only for the post-gate interval; if actual signed BUS integral spans that interval, do not add W0 or mechanical contribution again.",
            "Unverified capacitor effectiveness, node differences, and other in-flight sources must be included in the admission margin. Do not allocate the 0.20 V voltage reserve twice.",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
