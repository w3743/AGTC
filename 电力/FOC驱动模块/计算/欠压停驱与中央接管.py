"""Independent electrical corner screens for the isolated DUMP_5V monitor."""
import hashlib
import itertools
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENET = HERE.parent / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
REGULATOR = HERE.parents[1] / "稳压模块/检查/拒充边界与减件筛查.json"
REGULATOR_CANDIDATE = HERE.parents[1] / "稳压模块/工具/隔离候选独立门限.js"
VIT_NOM = 0.505
R_TOP = 180_000
R_BOTTOM = 26_700


def divider_corners():
    # 1% initial tolerance plus independent 100 ppm/K over a 100 K excursion.
    cases = []
    for a, b, threshold, hysteresis, ibias in itertools.product(
        (0.98, 1.02), (0.98, 1.02), (0.975, 1.025), (0.03, 0.08), (-50e-9, 50e-9)
    ):
        top, bottom = R_TOP*a, R_BOTTOM*b
        sense = VIT_NOM*threshold
        bias_drop = ibias*top
        ratio = 1+top/bottom
        cases.append({"fall_V": sense*ratio+bias_drop,
                      "rise_V": sense*(1+hysteresis)*ratio+bias_drop,
                      "step_10pct_below_trip_equivalent_V": sense*0.9*ratio+bias_drop,
                      "source_R_ohm": top*bottom/(top+bottom)})
    return cases


def central_windows():
    cap_f = 0.05286
    vlimit = 39.209-0.2
    power_w = 915
    return {
        "assumptions": {"C_F":cap_f,"limit_V":vlimit,"net_power_W":power_w},
        "time_ms_by_true_trigger_V": {str(v):1000*0.5*cap_f*(vlimit**2-v**2)/power_w
                                      for v in (38.0,38.25,38.5)},
        "latest_true_trigger_V_by_response_ms": {str(t):math.sqrt(vlimit**2-2*power_w*t/1000/cap_f)
                                                 for t in (0.5,1,1.5,2,2.25)},
        "TLV6700_independent_divider_k_must_exceed":0.404/39.209,
        "TLV6700_independent_divider_k_must_be_below":0.387/37.8,
        "has_static_intersection":0.404/39.209 < 0.387/37.8,
    }


def regulator_candidate():
    # Thresholds use the 40x range and the actual isolated candidate's two taps.
    # Apply the monitor's ±0.4 V input-referred error once, after REF and divider tolerances.
    def thresholds(ref, r1, r2, r3):
        total = r1+r2+r3
        return 40*ref*(r2+r3)/total, 40*ref*r3/total

    corners = [thresholds(ref, 150_000*a, 1000*b, 100_000*c)
               for ref,a,b,c in itertools.product((2.378,2.426),(0.999,1.001),(0.99,1.01),(0.99,1.01))]
    return {"Rs": "REF / 40x", "PS": "REF / inverting", "VIN": "BUS_EXT",
            "output": "DUMP_OFF", "R_QP_COL_ohm":30_000,
            "nominal_rise_fall_V":thresholds(2.402,150_000,1000,100_000),
            "initial_rise_V": [min(p[0] for p in corners)-0.4,max(p[0] for p in corners)+0.4],
            "initial_fall_V": [min(p[1] for p in corners)-0.4,max(p[1] for p in corners)+0.4],
            "qualification": "Conservative independent corner screen excludes resistor TCR, ground differences, RC delay, OUT startup and current establishment. Datasheet propagation maximum is for 10x and 10% overdrive, not this near-threshold 40x use."}


def main():
    divider = divider_corners()
    ranges = {k:[min(row[k] for row in divider),max(row[k] for row in divider)] for k in divider[0]}
    r_pull_min = 2200*0.99
    report = {
        "status":"ISOLATED_CANDIDATE_NOT_ADOPTED",
        "sourcesSha256": {str(f.relative_to(HERE.parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in (ENET,REGULATOR,REGULATOR_CANDIDATE)},
        "central": central_windows(),
        "regulator_isolated_candidate": regulator_candidate(),
        "divider_range":ranges, "divider_cases":divider,
        "bench_conditions": {
            "BREAK_N_sink_at_3p3V_pullup_and_0p3V_mA":1000*(3.3-0.3)/r_pull_min,
            "BREAK_N_sink_at_3p6V_pullup_and_0p3V_mA":1000*(3.6-0.3)/r_pull_min,
            "datasheet_VOL_max_V_at_VDD_3p3V_and_2mA":0.3,
            "divider_max_feed_uA_if_SENSE_short_at_DUMP_5p3V":1e6*5.3/(R_TOP*0.98),
            "DUMP_U3_min_operating_V":2.2,
            "step_screen_example_delay_us":50,
            "RC_us_for_sensitivity_pF":{str(c):ranges["source_R_ohm"][1]*c*1e-6 for c in (5,30,100)},
        },
        "bounds_and_gaps": [
            "Divider thresholds use the published adjustment accuracy and hysteresis plus 50 nA bias sensitivity. The bias specification test point is VDD=VSENSE=6 V, not a separate VDD=0 guarantee.",
            "Open-drain VOL is supported at the specified VDD=3.3 V test point. Pull-up corner arithmetic is not proof of VOL at 3.0/3.6 V or during MCU rail ramp-down.",
            "The 50 us detect time belongs to the stated step test, not all rail slew rates or the full BREAK_N-PWM-six VGS path.",
            "DUMP_5V and MCU_3V3 sequencing, off-state backfeed, fastest DUMP_5V collapse and local residual current must be measured. No new central or FOC overvoltage setpoint is introduced.",
        ],
    }
    HERE.joinpath("欠压停驱与中央接管.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
