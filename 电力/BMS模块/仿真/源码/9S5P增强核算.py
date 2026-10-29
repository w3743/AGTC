"""9S5P/130A design and incremental cost screen; does not edit CAD or current BOM."""
from pathlib import Path
from decimal import Decimal as D
from collections import Counter
import copy
import csv
import hashlib
import json

POWER = Path(__file__).resolve().parents[3]
BMS = POWER / "BMS模块"
PLAN = BMS / "原理图准备"


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    bom_path = PLAN / "BOM.csv"
    baseline = list(csv.DictReader(bom_path.open(encoding="utf-8-sig")))
    by_ref = {r["位号"]:r for r in baseline}
    parts = json.loads((PLAN / "功率元件连接.json").read_text(encoding="utf-8-sig"))
    wiring = {r["ref"]:r for r in parts}
    quote_path = PLAN / "9S5P增量报价.json"
    quote = json.loads(quote_path.read_text(encoding="utf-8"))
    prices = {r["mpn"]:r for r in quote["observations"]}
    additions = []
    for ref3 in ["Q_CHG3","Q_DSG3","R_G_CHG3","R_G_DSG3","R_GS_CHG3","R_GS_DSG3"]:
        row = copy.deepcopy(wiring[ref3])
        assert row["mpn"] == by_ref[ref3]["MPN"]
        row["ref"] = ref3[:-1]+"4"
        assert row["ref"] not in by_ref
        row["pins"] = {p:n.replace("_G3","_G4") for p,n in row["pins"].items()}
        row["note"] = "新增第4并联支路；仅预期针网，尚未写入原生工程。每颗MOS保留独立门阻和GS泄放。"
        row["quantity"] = 1
        row["baseline_copy_ref"] = ref3
        row["supplier_part"] = prices[row["mpn"]]["supplier_part"]
        additions.append(row)
    demand = Counter(r["mpn"] for r in additions)
    assert demand == {"HYG022N10NS1TA":2,"ABD03K1000FT":2,"RS-03L1005FT":2}
    assert set(demand) <= {r["MPN"] for r in baseline}
    cost_rows = []
    installed = D(0)
    separate_cash = D(0)
    merged_delta = D(0)
    for mpn,quantity in demand.items():
        observation = prices[mpn]
        price = D(observation["unit_price"])
        old_quantity = sum(int(r["每套用量"]) for r in baseline if r["MPN"]==mpn)
        # Other BMS circuits may reuse this exact resistor; merge all existing demand.
        assert old_quantity >= 6
        subtotal = quantity*price
        cash = max(quantity,observation["moq"])*price
        old_cash = max(old_quantity,observation["moq"])*price
        new_cash = max(old_quantity+quantity,observation["moq"])*price
        installed += subtotal
        separate_cash += cash
        merged_delta += new_cash-old_cash
        cost_rows.append({"mpn":mpn,"supplier_part":observation["supplier_part"],
                          "old_quantity":old_quantity,"new_quantity":old_quantity+quantity,
                          "added_quantity":quantity,"unit_price_CNY":str(price),
                          "installed_delta_CNY":str(subtotal),"moq":observation["moq"],
                          "separate_purchase_cash_estimate_CNY":str(cash),
                          "merged_existing_BMS_order_cash_delta_estimate_CNY":str(new_cash-old_cash),
                          "source_url":observation["source_url"]})
    cells = 9*D(prices["INR18650-30P"]["unit_price"])
    assert installed==D("14.9792") and separate_cash==D("19.8400")
    assert merged_delta==D("14.88") and cells==D("117.00")
    source_files = [bom_path,PLAN/"功率元件连接.json",quote_path,
                    BMS/"设计需求.md",BMS/"厂商资料/HYG022N10NS1TA.pdf",
                    BMS/"厂商资料/HoLRS2512-6W-0.5mR-1%_HoS20260522-54_A0.pdf"]
    binding = {str(p.relative_to(POWER)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    change = {
        "date":"2026-10-29","status":"DESIGN_CONTRACT_NOT_NATIVE_IMPLEMENTED_OR_QUALIFIED",
        "source_sha256":binding,"battery":{"old":"9S4P","selected":"9S5P","old_cells":36,
            "new_cells":45,"added_cells":9,"nominal_capacity_Ah":15,"nominal_energy_Wh":486,
            "new_capacitor_quantity":8,"new_IC_quantity":0},
        "standalone_regulator_required":True,
        "native_and_current_procurement_bom_modified":False,
        "main_scheme":{"FET_parallel_per_bank":4,"shunt_ohm":0.0005,"shunt_quantity":1,
            "continuous_current_target_kept_A":60,"peak_current_engineering_target_A":130,
            "peak_duration_user_requirement_s":None,"pulse_comparison_s":1},
        "proposed_added_components":additions,"removed_components":[],
        "count":{"BMS_board_added":6,"new_board_MPNs":0,"battery_added":9,
                 "known_discrete_added_total":15,"new_ICs":0,"added_capacitors":0,
                 "added_cell_fusible_connection_sites":9,"connection_sites_are_not_counted_as_nine_discrete_fuses":True},
        "cost_rows":cost_rows,"cell_user_unit_price_CNY":"13.00",
        "cost":{"board_installed_delta_CNY":str(installed),"cells_delta_CNY":str(cells),
                "known_installed_delta_CNY":str(cells+installed),
                "separate_additions_order_estimate_CNY":str(cells+separate_cash),
                "merged_original_BMS_order_delta_estimate_CNY":str(cells+merged_delta),
                "complete_incremental_total_CNY":None,"excluded":quote["exclusions"],
                "cash_note":quote["cash_note"]},
        "protection_register_contract":{"OCD1_threshold":{"address":"0x9282","old":35,"new_candidate":40,"nominal_mV":80},
            "OCD1_delay":{"address":"0x9283","candidate":1,"nominal_ms":9.9},
            "SCD_threshold":{"address":"0x9286","kept":5,"nominal_mV":100,"accuracy_fraction":0.15},
            "SCD_delay":{"address":"0x9287","kept":1,"nominal_us":15},
            "OCC_threshold":{"address":"0x9280","kept":3,"nominal_mV":6,"not_16A_regulator":True},
            "battery_net_charge_upper_limit_A":16,"all_settings_require_write_readback_and_qualification":True},
        "not_credited_simplification":{"DFETOFF_active_low_candidate_delete_count":2,
            "status":"separate review candidate; not mixed into +6 count or cost"}}
    dump(PLAN/"9S5P增量清单.json",change)

    rmin = 0.0005*(1-0.01)*(1-50e-6*50)
    rmax = 0.0005*(1+0.01)*(1+50e-6*50)
    parasitic = 10e-6  # symmetric sensitivity budget; not a measured value
    ocd_low = (0.080-0.005)/(rmax+parasitic)
    old_ocd_low = (0.070-0.005)/(rmax+parasitic)
    scd_low = (0.100*(1-0.15))/(rmax+parasitic)
    assert old_ocd_low<130<ocd_low<scd_low
    assert 130/4 <=100/3
    screen = {
        "date":"2026-10-29","status":"ARITHMETIC_AND_CONTRACT_CHECKS_ONLY",
        "source_sha256":binding,"checks_passed":["six unique new refs", "existing exact MPN reuse",
            "incremental money arithmetic", "per-FET current not above old comparison point",
            "OCD1 no-false-trip static screen with declared parasitic budget"],
        "shunt":{"nominal_ohm":0.0005,"signal_at_130A_mV":65,
                 "power_at_130A_W":130**2*0.0005,"pulse_1s_energy_J":130**2*0.0005,
                 "manufacturer_short_overload_test":"15W for 5s; resistance change test, not a board thermal rating"},
        "MOS_conduction":[{"current_A":i,"parallel_per_bank":n,
                           "path_R_25C_max_ohm":2*0.0028/n,
                           "path_loss_25C_W":i*i*2*0.0028/n,
                           "per_MOS_current_A":i/n,"heat_factor2_screen_W":2*i*i*2*0.0028/n}
                          for i,n in [(60,3),(60,4),(100,3),(130,3),(130,4)]],
        "gate":{"Qg_typ_nC_per_FET":160.5,"Qg_typ_nC_per_bank":642,
                "twice_typ_engineering_screen_nC_per_bank":1284,
                "Ciss_typ_nF_per_bank":4*9.026,
                "BQ20us_turnoff_test_load_nF":20,"maximum_bank_turnoff_time_proved":False},
        "thresholds":{"shunt_corner_temperature_C":75,"calibration_temperature_C":25,
            "Rmin_ohm":rmin,"Rmax_ohm":rmax,"sense_error_budget_abs_ohm":parasitic,
            "old_OCD1_earliest_A":old_ocd_low,"new_OCD1_earliest_A":ocd_low,
            "new_OCD1_latest_A":(0.080+0.005)/(rmin-parasitic),
            "SCD_earliest_A":scd_low,"SCD_latest_A":0.115/(rmin-parasitic),
            "OCC_latest_A":0.00865/(rmin-parasitic),
            "OCC_does_not_guarantee_16A":True},
        "peak_power":{"motor_input_W":3600,"auxiliary_assumption_W":100,
            "loaded_BUS_assumption_V":22.5,"battery_target_A":130,"battery_power_W":2925,
            "capacitor_gap_W":775,"no_loss_of_regen_priority":True},
        "limits":["no 130A real hardware rating", "no temperature/repetition qualified",
                  "fault energy and fuse coordination pending", "no switch gate timing extrapolation",
                  "startup limited when capacitor energy unavailable", "no mechanical brake fallback",
                  "0V startup and braking envelope remain separate open items"]}
    dump(BMS/"检查/9S5P增强核算.json",screen)
    print(json.dumps({"board_added":6,"cells_added":9,"new_MPNs":0,
        "installed_CNY":str(cells+installed),"separate_order_CNY":str(cells+separate_cash),
        "merged_delta_CNY":str(cells+merged_delta),"OCD_earliest_A":ocd_low,
        "SCD_earliest_A":scd_low,"status":"SCREEN_ONLY"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
