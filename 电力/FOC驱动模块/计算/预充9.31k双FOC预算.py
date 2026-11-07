"""FOC contribution to the BMS 9.31 kΩ precharge comparison, with no assumed load ceiling."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENET = ROOT / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
ASSEMBLY = ENET.with_name("FOC_G431_双板装配补充.csv")
BMS = ROOT.parent / "BMS模块/设计细节.md"
R_PATH = 9.41965
C_BUS = 0.079784
T = 5.593 - 0.326
TAU = R_PATH * C_BUS
DV_TARGET = 0.1108
V_PACK = 37.8


def rc_result():
    retained = math.exp(-T / TAU)
    dc_gain = R_PATH * (1 - retained)
    load_max = (DV_TARGET - V_PACK * retained) / dc_gain
    return retained, dc_gain, load_max


def terminal_pulse_error(current_a, start_s, end_s):
    return R_PATH * current_a * (math.exp(-(T-end_s)/TAU) - math.exp(-(T-start_s)/TAU))


def main():
    raw = json.loads(ENET.read_text(encoding="utf-8-sig"))
    parts = {c["props"]["Designator"]: c for c in raw["components"].values()}
    refs = ("U1", "U2", "U6", "U3", "U7", "U2_CIN", "U6_CIN", "U2_COUT", "U6_COUT",
            "U3_C1", "U3_C2", "C_U7", "CBUS1", "CBUS2", "CBUS3", "DTVS1", "Q1", "Q2",
            "R_LOAD_VALID_BIAS", "R_BUCK_VALID_PD", "R_BUCK_VALID_BASE", "R_BUCK_CLAMP_BUS",
            "Q_BUCK_CLAMP", "Q_BUCK_RELEASE", "U1_BUCK_RT", "J_LOAD_POWER_VALID", "R_OV_PU",
            "R_BOV_PNP_B", "R_BOV_DRIVE", "R_SAFEADC_TOP", "R_BUSADC_TOP", "R_OVTOP", "R_DRV_EN_PD")
    observed = {ref: {"MPN": parts[ref]["props"]["Manufacturer Part"],
                      "pins": {n: p.get("net") or "" for n,p in parts[ref]["pinInfoMap"].items()}}
                for ref in refs}
    for ref,rail in (("U2","SAFE_5V"),("U6","DUMP_5V")):
        assert {n:observed[ref]["pins"][n] for n in ("1","5","8")} == {"1":rail,"5":"BUS","8":"BUS"}
    assert {n:observed["U1"]["pins"][n] for n in ("5","6","33","43","47")} == {
        "5":"BUS","6":"BUS","33":"DRV_ENABLE","43":"BUS","47":"BUCK_RT_SD"}
    assert observed["Q_BUCK_CLAMP"]["pins"] == {"1":"BUCK_CLAMP_BASE","2":"GND_SYS","3":"BUCK_RT_SD"}
    assert observed["Q_BUCK_RELEASE"]["pins"] == {"1":"BUCK_RELEASE_BASE","2":"GND_SYS","3":"BUCK_CLAMP_BASE"}
    assert observed["R_LOAD_VALID_BIAS"]["pins"] == {"1":"LOAD_EN_BIAS_EXT","2":"SAFE_5V"}
    assert "FOC-A安装、FOC-B DNP" in ASSEMBLY.read_text(encoding="utf-8-sig")
    assert all(set(observed[ref]["pins"].values()) == {"BUS","GND_SYS"}
               for ref in ("CBUS1","CBUS2","CBUS3","U2_CIN","U6_CIN"))
    _, dc_gain, load_max = rc_result()
    # Conservative passive sub-envelope: each BUS branch sees all VBUS and each LDO rail is <=5.25 V.
    nominal_bus_resistors = (100_000, 280_000, 180_000, 1_000_000)
    resistance_factor = 1 - 0.01 - 100e-6 * 35
    passive_mA = 2000 * (V_PACK*sum(1/r for r in nominal_bus_resistors)
                         + 2*5.25/127_700 + 5.25/27_400) / resistance_factor
    # The output capacitors are not in C_BUS. Bound charge for a single monotonic startup only.
    output_charge = 2*(21.2e-6 * 1.2)*5.25
    allowance_rows = []
    for latest_s in (0.5,1.0,2.0,3.0,4.0,T):
        drop = output_charge / C_BUS * math.exp(-(T-latest_s)/TAU)
        allowance_rows.append({"complete_by_s": latest_s, "charge_only_terminal_penalty_mV": 1000*drop,
                               "charge_only_equivalent_plateau_mA": 1000*drop/dc_gain})
    result = {
        "status":"NO_GUARANTEED_FOC_CURRENT_CEILING",
        "sourceSha256":{str(f.relative_to(ROOT.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in (ENET,ASSEMBLY,BMS)},
        "savedPinout":observed,
        "BMSComparison": {"Vpack_V":V_PACK,"R_path_ohm":R_PATH,"C_BUS_F":C_BUS,"T_charge_s":T,
                          "target_delta_V":DV_TARGET,"total_constant_sink_max_mA":1000*load_max,
                          "steady_state_ceiling_mA":1000*DV_TARGET/R_PATH,
                          "other_after_2p04mA_nominal_mA":1000*load_max-2.04,
                          "existing_passive_branch_corner_mA":passive_mA,
                          "other_after_passive_corner_mA":1000*load_max-passive_mA},
        "directCapacitance": {"bulk_dual_mF_nominal_initial_low_high":[60,48,72],
                              "small_dual_uF_nominal":117.8,"small_dual_uF_model_plus20":141.36,
                              "included_in_BMS_BUS_model":True},
        "downstreamLDOChargeOnly": {"Cout_dual_uF_nominal":42.4,"Cout_model_factor":1.2,
                                     "output_rail_comparison_V":5.25,"Q_dual_uC":output_charge*1e6,
                                     "pulse_weighting":allowance_rows},
        "exampleAllocations_mA": {str(total):1000*load_max-total for total in (3,4,5,6,7,8)},
        "exampleEarlyBuck": {"input_assumed_W":2.4,"input_mA_ideal_at_37p8":2400/V_PACK,
                             "terminal_delta_mV_for_last_50ms":1000*terminal_pulse_error(2.4/V_PACK,T-0.05,T)},
        "limits": [
            "The BMS constant-load limit is a system-model condition, not the FOC guaranteed current.",
            "The bounded charge example assumes a single monotonic rail start, no thermal cycling or restart, fixed ideal equivalent C, and does not cap LDO instantaneous current.",
            "The passive-resistor envelope replaces the already booked 2.04 mA nominal line. It must not be added to that line.",
            "No guaranteed worst-case leakage for the six PEH200 capacitors is specified for this sub-rated-voltage, short-ramp and temperature profile.",
            "A proven main-path-low-impedance indication must precede enabling the Buck; gate enabling is independent of energized BUS-derived U6 protection.",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
