"""Check the sixth-board deliverables and prove the previously frozen files stayed intact."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import runpy

HERE = Path(__file__).resolve().parent
POWER = HERE.parents[1]
ROOT = POWER.parent
OUT = HERE / "充电MPPT交付校验.json"
CAN_PINOUT = {"1": "GND_CAN", "2": "CANH", "3": "CANL"}
PLACEHOLDER = re.compile(r"UNVERIFIED|SYMBOL_ONLY|待核|待定|TBD|TO_VERIFY", re.I)
PASSIVE_UNITS = {"R": r"(?:Ω|ohms?)", "C": "F", "L": "H"}
RETIRED_LCSC = {"C5156605", "C22471967", "C3215600"}
REQUIRED_B3_CRITERIA = {
    "ov_trip_guard_at_least_200mv", "bq_default_cv_upper_below_37p45v",
    "epr_tvs_clamp_below_powered_protector_abs_max", "epr_voltage_below_unpowered_protector_abs_max",
    "epr_power_loss_below_unpowered_protector_abs_max", "b2_pv_only_has_aux_power_path",
    "disabled_boost_output_disconnected", "illustrative_low_corner_gate_at_least_10v",
    "bbm_deadtime_at_least_5ms",
}
CRITICAL_ICS = {
    "TPS26750SRSMR": (1, "C42166327", {"1": "LDO_3V3", "20": "POWER_PATH_EN"}),
    "TPD4S480RUKR": (1, "C43131250", {"10": "VPWR", "16": "EPR_EN", "19": "VBUS_LV", "20": "VBUS"}),
    "BQ25756RRVR": (1, "C19272232", {"11": "FBG", "12": "FB", "13": "SRN", "14": "SRP",
                                      "23": "DRV_SUP", "24": "REGN"}),
    "SCT2A23ASTER": (1, "C2901894", {"2": "VIN", "3": "EN", "5": "FB"}),
    "TPS7A1633DGNR": (1, "C181239", {"1": "OUT", "3": "PG", "4": "GND", "5": "EN", "7": "DELAY", "8": "IN", "9": "EP"}),
    "TMUX1511PWR": (2, "C2866750", dict(zip(map(str, range(1, 15)), (
        "SEL1", "S1", "D1", "SEL2", "S2", "D2", "GND", "D3", "S3", "SEL3", "D4", "S4", "SEL4", "VDD")))),
    "TPS3435CAIEGDDFR": (1, "C22427898", dict(zip(map(str, range(1, 9)), (
        "SET0", "MR", "WDI", "GND", "SET1", "WD-EN", "WDO", "VDD")))),
}
ADC_FUNCTIONS = {
    "PA0": "ADC1_IN1", "PA1": "ADC1_IN2", "PA2": "ADC1_IN3", "PA3": "ADC1_IN4",
    "PA4": "ADC2_IN17", "PA5": "ADC2_IN13", "PA6": "ADC2_IN3", "PA7": "ADC2_IN4",
    "PB0": "ADC1_IN15", "PB1": "ADC1_IN12", "PC4": "ADC2_IN5",
}
PERIPHERALS = {
    "PB12": ("TIM1_BKIN", 6, "MPPT_BKIN_N"),
    "PB13": ("TIM1_CH1N", 6, "PV1_PWM_LS"), "PB14": ("TIM1_CH2N", 6, "PV2_PWM_LS"),
    "PB15": ("TIM1_CH3N", 4, "PV3_PWM_LS"),
    "PA8": ("TIM1_CH1", 6, "PV1_PWM_HS"), "PA9": ("TIM1_CH2", 6, "PV2_PWM_HS"),
    "PA10": ("TIM1_CH3", 6, "PV3_PWM_HS"),
    "PA11": ("FDCAN1_RX", 9, "CAN_RX"), "PA12": ("FDCAN1_TX", 9, "CAN_TX"),
    "PA15": ("I2C1_SCL", 4, "CHG_I2C_SCL"), "PB7": ("I2C1_SDA", 4, "CHG_I2C_SDA"),
}
ADC_SIGNALS = {f"{kind}_PV{index}_ADC" for kind in ("V", "I") for index in range(1, 4)} | {
    "V_BUS_ADC", "I_OUT_ADC", "NTC_BOARD_ADC", "NTC_HEATER_ADC",
}
TMUX_CHANNEL_PINS = (("2", "3", "1"), ("5", "6", "4"), ("9", "8", "10"), ("12", "11", "13"))
ISOLATION_CHANNELS = {
    "U_ISO_I2C_PD": (
        ("CHG_I2C_SCL", "PD_I2CT_SCL", "PD_I2C_EN"),
        ("CHG_I2C_SDA", "PD_I2CT_SDA", "PD_I2C_EN"),
        ("TPS26750_IRQ_N", "PD_IRQ_DEV_N", "PD_I2C_EN"),
        ("BQ_INT_N", "BQ_INT_DEV_N", "BQ_I2C_EN"),
    ),
    "U_ISO_I2C_BQ_CAN": (
        ("CHG_I2C_SCL", "BQ_I2C_SCL", "BQ_I2C_EN"),
        ("CHG_I2C_SDA", "BQ_I2C_SDA", "BQ_I2C_EN"),
        ("CAN_TX", "CAN_TX_DEV", "CAN_RAILS_VALID"),
        ("CAN_RX", "CAN_RX_DEV", "CAN_RAILS_VALID"),
    ),
}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_documents():
    # Reuse the exact repository rule in-process, adding the sixth module to
    # this invocation while leaving the existing command and all four boards intact.
    namespace = runpy.run_path(str(POWER / ".agents/tools/统一目录说明.py"))
    main = namespace["main"]
    main.__globals__["MODULES"] = (*namespace["MODULES"], "充电MPPT模块")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = main()
    return status, json.loads(buf.getvalue())


def text_field(value):
    return isinstance(value, str) and bool(value.strip()) and not PLACEHOLDER.search(value)


def check_drawing(contract):
    parts, nets, errors, test_nets = {}, {}, [], set()
    rows = contract.get("drawing_components")
    if not isinstance(rows, list) or not rows:
        return parts, ["drawing_components is not a nonempty array"], {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"invalid row {index}")
            continue
        ref = row.get("ref")
        if not text_field(ref) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", ref) or ref in parts:
            errors.append(f"invalid or duplicate ref at row {index}: {ref}")
            continue
        parts[ref] = row
        for field in ("mpn", "manufacturer", "lcsc", "footprint", "value"):
            if not text_field(row.get(field)):
                errors.append(f"{ref}.{field}: missing or placeholder")
        if not re.fullmatch(r"C[1-9][0-9]*", str(row.get("lcsc"))):
            errors.append(f"{ref}: invalid LCSC code")
        if row.get("lcsc") in RETIRED_LCSC:
            errors.append(f"{ref}: retired LCSC code")
        if row.get("placement") != "board" or any(type(row.get(key)) is not bool for key in ("dnp", "in_bom")):
            errors.append(f"{ref}: invalid placement or BOM flags")
        pins, roles = row.get("pins"), row.get("pin_roles")
        if not isinstance(pins, dict) or not pins or not isinstance(roles, dict):
            errors.append(f"{ref}: missing pin maps")
            continue
        if (set(roles) - set(pins) or any(not text_field(role) for role in roles.values())
                or (ref.startswith(("J", "Q", "U", "D")) and set(roles) != set(pins))):
            errors.append(f"{ref}: invalid or incomplete pin roles")
        if ref[0] in PASSIVE_UNITS and not ref.startswith(("RN", "RL", "RY", "CN")):
            unit = PASSIVE_UNITS[ref[0]]
            if set(pins) != {"1", "2"} or not re.match(
                    rf"^\d+(?:\.\d+)?[pnumkKMµμ]?{unit}(?=\s|$)", str(row.get("value"))):
                errors.append(f"{ref}: invalid passive pins or renderer value")
        for pin, net in pins.items():
            if not text_field(pin) or (net is not None and (not text_field(net) or net in ("NC", "N/C"))):
                errors.append(f"{ref}.{pin}: invalid pin or net")
            elif net is not None:
                endpoints = nets.setdefault(net, [])
                if row.get("dnp") is False:
                    endpoints.append(f"{ref}.{pin}")
                    if ref.startswith("TP") and len(pins) == 1:
                        test_nets.add(net)
    single_pin_nets = {net: endpoints for net, endpoints in sorted(nets.items())
                       if len(endpoints) < 2 and net not in test_nets}
    return parts, errors, single_pin_nets


def check_critical_ics(parts):
    errors = []
    for mpn, (minimum, lcsc, expected_roles) in CRITICAL_ICS.items():
        matches = [row for row in parts.values() if row.get("mpn") == mpn and row.get("dnp") is False]
        if len(matches) < minimum:
            errors.append(f"{mpn}: expected at least {minimum}, found {len(matches)}")
        for row in matches:
            roles, pins = row.get("pin_roles"), row.get("pins")
            if (row.get("lcsc") != lcsc or not isinstance(roles, dict) or not isinstance(pins, dict)
                    or any(roles.get(pin) != expected for pin, expected in expected_roles.items())
                    or any(not text_field(pins.get(pin)) for pin in expected_roles
                           if mpn != "TPS7A1633DGNR" or pin not in {"3", "7"})):
                errors.append(f"{row['ref']}: {mpn} LCSC, physical pin function or required connection mismatch")
            elif mpn == "TPS7A1633DGNR" and (
                set(pins) != {str(pin) for pin in range(1, 10)} or roles.get("2") != "DNC" or roles.get("6") != "NC"
                or pins["2"] is not None or pins["6"] is not None or pins["1"] != "3V3_STBY"
                or pins["4"] != "GND_SYS" or pins["9"] != "GND_SYS"
                or pins["8"] in {"PD_VBUS_RAW", "XT60_IN", "AUX_HV_IN"}
            ):
                errors.append(f"{row['ref']}: standby rail, ground, reserved pins or input domain mismatch")
    fast_pnp = [row for row in parts.values() if row.get("lcsc") == "C8326" and row.get("dnp") is False
                and row.get("mpn") == "MMBT5401(RANGE:200-300)"]
    if not any(row.get("pin_roles") == {"1": "B", "2": "E", "3": "C"}
               and isinstance(row.get("pins"), dict) and set(row["pins"]) == {"1", "2", "3"}
               and all(text_field(net) for net in row["pins"].values()) for row in fast_pnp):
        errors.append("EPR fast turn-off PNP: missing populated C8326 with B/E/C on pins 1/2/3")
    if any(str(row.get("mpn", "")).upper().startswith("BC857C") for row in parts.values()):
        errors.append("BC857C is prohibited in this drawing contract")
    return errors


def check_isolation(parts):
    errors = []
    for ref, channels in ISOLATION_CHANNELS.items():
        row = parts.get(ref, {})
        expected = {"7": "GND_SYS", "14": "MCU_3V3"}
        for physical, logical in zip(TMUX_CHANNEL_PINS, channels):
            expected.update(zip(physical, logical))
        if (row.get("mpn") != "TMUX1511PWR" or row.get("lcsc") != "C2866750"
                or row.get("dnp") is not False or row.get("pins") != expected):
            errors.append(f"{ref}: eight-channel allocation, supplies or populated device does not match")
    return errors


def check_lock_review(review):
    if not isinstance(review, dict) or review.get("design_status") != "PASS":
        return False
    criteria = review.get("criteria")
    return (isinstance(criteria, dict) and REQUIRED_B3_CRITERIA <= criteria.keys()
            and all(result is True for result in criteria.values()))


def check_mcu(contract, parts):
    frozen = read(POWER / "BMS模块/检查/主控头引脚核对.json")["pin_map"]
    mcu = contract.get("mcu_pin_map", {})
    errors, assigned, adc = [], {}, {}
    for header in ("P1", "P2"):
        ref = f"J_CORE_{header}"
        rows = mcu.get(ref, [])
        pins = {str(row["pin"]): row for row in rows}
        drawing = parts.get(ref, {})
        expected_pins = {str(pin) for pin in range(1, 25)}
        if len(rows) != 24 or set(pins) != expected_pins or set(drawing.get("pins", {})) != expected_pins:
            errors.append(f"{ref}: expected 24 distinct physical pins")
            continue
        for known in frozen[header]:
            pin = str(known["pin"])
            pad, gpio = known["board_net"], known["mcu_gpio"]
            row = pins[pin]
            function, signal = row.get("function"), row.get("signal")
            if (row.get("pad") != pad or pin not in drawing.get("pins", {})
                    or drawing["pins"][pin] != signal or drawing.get("pin_roles", {}).get(pin) != pad
                    or ((function == "NC") != (signal is None))):
                errors.append(f"{ref}.{pin}: pad, NC or drawing mismatch")
            if gpio in {"PB2", "PB4", "PB6", "PB8", "PC13", "PC14", "PC15"} or pad in {"VREF+", "NRST", "VBAT/VB"}:
                if signal is not None:
                    errors.append(f"{ref}.{pin}: {pad} must be NC")
            if gpio in PERIPHERALS and (function, row.get("af"), signal) != PERIPHERALS[gpio]:
                errors.append(f"{gpio}: function, AF or signal mismatch")
            if gpio == "PB3" and (function, row.get("af"), signal) != ("GPIO_OUT", None, "LD_PROBE_EN"):
                errors.append("PB3: expected LD_PROBE_EN as a GPIO output")
            if isinstance(function, str) and function.startswith("ADC"):
                if function != ADC_FUNCTIONS.get(gpio) or function in adc or row.get("af") is not None:
                    errors.append(f"{gpio}: invalid or duplicate ADC channel")
                adc[function] = signal
            if gpio is not None and signal is not None:
                if signal in assigned or gpio in assigned.values():
                    errors.append(f"{gpio}: duplicate GPIO or signal assignment")
                assigned[signal] = gpio
    if len(adc) != 10 or set(adc.values()) != ADC_SIGNALS or adc.get("ADC2_IN5") != "I_OUT_ADC":
        errors.append("expected ten unique ADC signals, including the ADC2_IN5 output-current path")
    return {"status": "PASS_PIN_MAP" if not errors else "FAIL",
            "gpio_assigned": len(assigned), "adc_channels": len(adc), "errors": errors}


def main():
    expected = read(HERE / "冻结四板SHA256基线.json")
    changed = [name for name, old in expected.items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != old]
    projects = {ROOT / name.split("/eda工程文件/")[0] / "eda工程文件"
                for name in expected if "/eda工程文件/" in name}
    exempt = {"node_modules", "dist", ".cache", "__pycache__"}
    added = sorted(
        p.relative_to(ROOT).as_posix()
        for project in projects for p in project.rglob("*") if p.is_file()
        and not exempt.intersection(p.relative_to(project).parts)
        and p.relative_to(ROOT).as_posix() not in expected
    )
    docs_code, docs = check_documents()
    contract = read(HERE / "充电MPPT施工合同.json")
    sim = read(HERE / "充电MPPT与自持仿真核算.json")
    can = read(HERE / contract["known_schematic_scope"]["deliverable"])
    can_contract = read(HERE / contract["known_schematic_scope"]["contract"])
    pins = can["pin_net_comparison"]
    cases = sim["conditions"]
    check_frozen = not changed and not added and len(projects) == 4 and len(expected) == 91
    parts, drawing_errors, single_pin_nets = check_drawing(contract)
    mcu = check_mcu(contract, parts)
    check_can = (can["status"] == "PASS" and can["components"]["observed"] == 10 and pins["equal"] == pins["total"] == 36
                 and can_contract["connector_pinout"] == CAN_PINOUT and parts.get("J_CAN", {}).get("pins") == CAN_PINOUT)
    probe = cases["recovery_interface_ld_probe"]["cases_by_assumed_load_ma"]
    b2 = cases.get("b2_lock_review", {})
    check_sim = (
        check_lock_review(b2)
        and sim["engine"]["version"] == "ngspice-47"
        and sim["engine"]["sha256"] == digest(ROOT / sim["engine"]["binary"])
        and sim["script_sha256"] == digest(HERE / "充电MPPT与自持仿真核算.py")
        and all(cases[case]["status"] == "PASS_MODEL" for case in (
            "solar_3_channel", "recovery_interface_charge_window", "regen_cutback", "usb_pd_boundaries",
            "b2_lock_review"))
        and cases["dark_to_sun_bus_zero"]["local_bias"] == "PASS_MODEL"
        and cases["dark_to_sun_bus_zero"]["isolation_negative_control"] == "PASS_MODEL"
        and probe["10"]["vld_threshold_crossing_supported_in_model"] is True
        and probe["60"]["vld_threshold_crossing_supported_in_model"] is False
    )
    target = contract["target_component_count"]
    eda = HERE.parent / "eda工程文件"
    try:
        report = read(eda / "exports/netlist_verify_report.json")
        reference = read(eda / f"reference_{target}_netlist.json")
        comparison = report["pin_net_comparison"]
        check_eda = (report["status"] == "PASS" and report["components"]["observed"] == target
                     and comparison["equal"] == comparison["total"] and comparison["total"] > 0
                     and report["kicad"]["sheets"] == 1 and not report["findings"]
                     and reference["target_schematic_count"] == target
                     and len(reference["components"]) == target
                     and reference["inputs"]["contract"]["sha256"] == digest(HERE / "充电MPPT施工合同.json")
                     and reference["inputs"]["simulation"]["sha256"] == digest(HERE / "充电MPPT与自持仿真核算.json")
                     and report["inputs"] == reference["inputs"])
    except (OSError, ValueError, KeyError, TypeError):
        check_eda = False
    safe_scope = (
        contract["release_for_power_connection"] is False
        and contract["release_for_purchase"] is False
        and len(contract.get("offboard_and_bench_holds", [])) == 6
        and not (HERE / "CodeAsEDA_tscircuit/exports/netlist_verify_report.json").exists()
        and check_eda
    )
    critical_ic_errors = check_critical_ics(parts)
    isolation_errors = check_isolation(parts)
    onboard_blockers = [item["id"] for item in contract["onboard_blockers"]]
    populated = [row for row in parts.values() if row.get("in_bom") and not row.get("dnp")]
    sku = sorted({row.get("lcsc") for row in populated if isinstance(row.get("lcsc"), str)})
    procurement = contract.get("procurement_summary", {})
    check_procurement = (procurement.get("listed_board_items") == len(populated)
                         and procurement.get("distinct_lcsc") == len(sku)
                         and procurement.get("lcsc_codes") == sku)
    drawing_complete = (contract["status"] == "CONTRACT_COMPLETE_EDA_CAPTURE_PENDING"
                        and type(target) is int and target == len(parts) and not onboard_blockers
                        and not drawing_errors and not critical_ic_errors and not isolation_errors
                        and not single_pin_nets and check_procurement)
    passed = (not docs_code and check_frozen and check_can and check_sim and safe_scope
              and drawing_complete and mcu["status"] == "PASS_PIN_MAP")
    data = {
        "status": "PASS_EDA_COMPLETE" if passed else "FAIL",
        "eda_capture": {"status": "PASS" if check_eda else "FAIL"},
        "full_board_construction": contract["status"],
        "drawing_contract": {"status": "PASS" if drawing_complete else "FAIL", "listed": len(parts),
                             "target": target, "metadata_errors": drawing_errors,
                             "critical_ic_errors": critical_ic_errors,
                             "isolation_errors": isolation_errors,
                             "single_pin_nets": single_pin_nets, "onboard_blockers": onboard_blockers},
        "mcu_pin_map": mcu,
        "procurement_count_check": {"status": "PASS" if check_procurement else "FAIL", "distinct_lcsc": sku},
        "frozen_assets": {"status": "PASS" if check_frozen else "FAIL", "checks_sha256": len(expected),
                          "changed_or_missing": changed, "added_in_frozen_projects": added},
        "document_gate": docs,
        "can_interface_only": {"status": "PASS" if check_can else "FAIL",
                               "objects": can["components"]["observed"], "pins": pins["equal"]},
        "simulation_evidence": {"status": "PASS" if check_sim else "FAIL",
                                "b2_design_status": b2.get("design_status"), "b2_criteria": b2.get("criteria"),
                                "scope": "source and binary hashes and model checks only; no bench qualification"},
        "scope_enforced": safe_scope,
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(data["status"], "drawing", data["drawing_contract"]["status"], "MCU", mcu["status"],
          "frozen", data["frozen_assets"]["status"], "docs", docs["status"],
          "CAN", data["can_interface_only"]["status"], "simulation", data["simulation_evidence"]["status"],
          "B2 design", b2.get("design_status"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
