"""Check the FOC share of the BMS 4.25 mA precharge budget."""
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENET = HERE.parent / "eda工程文件/导出/当前交付/FOC_G431_P1.enet"
BMS = HERE.parents[1] / "BMS模块/设计细节.md"
PREVIOUS = HERE / "预充9.31k双FOC预算.json"


def main():
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source = ENET.relative_to(HERE.parents[1])
    assert previous["sourceSha256"][str(source)] == digest(ENET)
    old = previous["BMSComparison"]
    voltage, r, c, t = (old[k] for k in ("Vpack_V", "R_path_ohm", "C_BUS_F", "T_charge_s"))
    system_target_ma, side_target_ma, delta_v = 4.25, 1.50, 0.0760
    tau = r*c
    retained = math.exp(-t/tau)
    steady = system_target_ma/1000*r
    terminal = steady+(voltage-steady)*retained
    budget_uc = 1e6*c*(delta_v-terminal)
    two_side_output_uc = previous["downstreamLDOChargeOnly"]["Q_dual_uC"]
    passive_ma = old["existing_passive_branch_corner_mA"]/2
    protection_ma = 4*0.100+0.013
    cutoff_buck_ma = 0.176
    assess = []
    for each_cap_ua in (0, 10, 20, 50):
        cap_ma = 3*each_cap_ua/1000
        assess.append({"each_PEH200_leakage_target_uA": each_cap_ua,
                       "remaining_electronics_total_mA":side_target_ma-cap_ma,
                       "remaining_after_passive_and_comparator_maxima_mA":side_target_ma-cap_ma-passive_ma-protection_ma,
                       "remaining_if_also_reserving_Buck_testpoint_mA":side_target_ma-cap_ma-passive_ma-protection_ma-cutoff_buck_ma})
    result = {
        "status":"HOLD_LEAKAGE_DERATING_ALONE_INSUFFICIENT_FOR_GUARANTEE",
        "source_sha256":{source.as_posix():digest(ENET), "BMS模块/设计细节.md":digest(BMS), PREVIOUS.name:digest(PREVIOUS)},
        "saved_pinout":{k:previous["savedPinout"][k] for k in
                        ("U1","U2","U6","U3","U7","Q_BUCK_CLAMP","Q_BUCK_RELEASE","R_LOAD_VALID_BIAS","J_LOAD_POWER_VALID")},
        "per_side_mA":{"assignment":side_target_ma,"passive_corner":passive_ma,
                       "available_after_passive":side_target_ma-passive_ma,
                       "U3_and_U7_datasheet_conditions":protection_ma,
                       "remaining_after_comparators":side_target_ma-passive_ma-protection_ma,
                       "Buck_shutdown_at_RT_SD_0V":cutoff_buck_ma},
        "leakage_derating_sensitivity": assess,
        "dynamic_comparison":{"R_ohm":r,"C_F":c,"T_s":t,"V_pack_V":voltage,"delta_limit_V":delta_v,
                              "baseline_current_mA":system_target_ma,"terminal_delta_V":terminal,
                              "unallocated_terminal_weighted_charge_uC":budget_uc,
                              "two_side_output_cap_charge_comparison_uC":two_side_output_uc,
                              "if_complete_at_1s_weighted_uC":two_side_output_uc*math.exp(-(t-1)/tau),
                              "if_at_window_end_delta_increment_mV":two_side_output_uc/(1000*c),
                              "latest_common_completion_s_if_all_spare_for_this_charge":t+tau*math.log(budget_uc/two_side_output_uc),
                              "comparison_bus_ramps_V":{str(v):{str(s):(v-steady)*(1-math.exp(-s/tau))
                                                               for s in (0.1,0.25,0.5,1.0,t)}
                                                         for v in (22.5,37.8)}},
        "boundaries":["Published current limits are separate test points, not measurements or a simultaneous board maximum.",
                      "Leakage-only screening cannot close the 1.50 mA worst-case allocation even with zero electrolytic leakage in the component-bound budget.",
                      "Both rails remain active; early output charge is separate from already-booked input capacitance. Weighted spare is system-wide and cannot also be spent on effective capacitance or other modules.",
                      "Keep Buck and gate drive off until main path is physically proven. No proposed pin change or procurement adoption."],
    }
    HERE.joinpath("预充4.25mA关态复核.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
