"""Reconstruct FOC source objects from the frozen 164-object ENET and contract."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
POWER = PROJECT.parents[2]
ROOT = POWER.parent
sys.path.insert(0, str(POWER / "检查/联合仿真/code_as_eda"))
from reference import read_json, require, source_record, load_baseline, apply_drawing_rows, write_reference

CONTRACT = PROJECT.parent / "闭合候选_164Rev2/FOC_164Rev2施工合同.json"


def build():
    contract = read_json(CONTRACT)
    baseline = ROOT / contract["baseline"]["candidate_164_enet"]
    parts = load_baseline(baseline)
    require(len(parts) == contract["baseline"]["candidate_component_count"] == 164,
            "Frozen FOC candidate must contain 164 schematic objects")
    baseline_pins = {ref: part["pins"].copy() for ref, part in parts.items()}
    old_hall = parts["J_HALL"]["pins"].copy()
    require(old_hall["1"] == "5V_BUCK", "Unexpected frozen Hall supply")

    drawing = contract["drawing_components"]
    drawing_by_ref = {entry["ref"]: entry for entry in drawing}
    require(len(drawing_by_ref) == len(drawing), "Duplicate contract drawing ref")
    require(set(drawing_by_ref) - parts.keys() == {"F_HALL_5V"}, "Unexpected object additions")
    apply_drawing_rows(parts, drawing)
    for part in parts.values():
        if part["placement"] == "on-board":
            part["placement"] = "board"
    expected_rewired = {"J_HALL", "J_BUS_P", "J_BUS_N", "J_MOTOR_U", "J_MOTOR_V", "J_MOTOR_W"}
    expected_rewired.update(f"Q{i}" for i in range(1, 7))
    actual_rewired = {ref for ref, pins in baseline_pins.items() if parts[ref]["pins"] != pins}
    require(actual_rewired == expected_rewired, f"Unexpected FOC rewiring: {actual_rewired ^ expected_rewired}")
    for ref in ("J_BUS_P", "J_BUS_N", "J_MOTOR_U", "J_MOTOR_V", "J_MOTOR_W"):
        old_nets = set(baseline_pins[ref].values())
        require(len(old_nets) == 1 and set(parts[ref]["pins"].values()) == old_nets,
                f"M4 replacement must preserve the original net: {ref}")
    require(parts["J_HALL"]["pins"] == {**old_hall, "1": "5V_HALL_PROT"}, "Hall pins differ")
    require(parts["F_HALL_5V"]["pins"] == {"1": "5V_BUCK", "2": "5V_HALL_PROT"},
            "Hall PTC wiring differs")

    # The baseline Name field can describe a superseded component or value.
    # These overrides describe the corresponding explicit contract replacements.
    current_values = {
        "F_MAIN": "100A / 70V EXT",
        "J_LOAD_POWER_VALID": "A2501WV-2P",
        "U1_BUCK_COUT2": "1uF / 25V",
        "U1_VCC_C": "1uF / 25V",
        **{f"Q{i}": "HYG022N10NS1TA / 100V" for i in range(1, 7)},
        **{ref: "MMBT5551" for ref in ("Q_HOLD", "Q_BOV_NPN", "Q_BUCK_CLAMP", "Q_BUCK_RELEASE")},
    }
    for ref, value in current_values.items():
        parts[ref]["value"] = value

    for row in contract["localization_v1"]["main_mos"]["reference_pins"]:
        part = parts[row["ref"]]
        require(part["pins"] == row["pins"], "MOS physical pad mapping differs for " + row["ref"])
        require(part["lcsc"] == "C2925050", "MOS stock code differs")

    bulk = contract["localization_v1"]["offboard_bulk"]
    require(len(bulk["refs"]) == bulk["per_side"] == 3, "FOC bank count differs")
    for ref in bulk["refs"]:
        part = parts[ref]
        require(part["pins"] == bulk["pins"], ref + " capacitor nets differ")
        for field in ("mpn", "manufacturer", "lcsc"):
            part[field] = bulk[field]
        part["value"] = "10000uF 63V off-board"
        part["footprint"] = bulk["physical_package"]
        part["pin_roles"] = bulk["pin_roles"]
        part["placement"] = "off-board"
        part["source_layers"].append("localization_v1.offboard_bulk")

    require(parts["F_MAIN"]["placement"] == "off-board", "Main fuse is off board")
    require(contract["target_component_count"] == len(parts) == 165, "FOC object count differs")
    require(sum(p["placement"] == "off-board" for p in parts.values()) == 4,
            "FOC has four off-board schematic objects")
    write_reference(PROJECT, 165, CONTRACT, baseline, parts, notes=[
        "165 schematic objects include 161 on-board objects and four off-board electrical symbols per FOC board.",
        "CBUS1–3 are an off-board bank; F_MAIN is an off-board bolt fuse.",
        "Q1–Q6 use the frozen contract's physical TOLL pins 1–9.",
        "HALL supply short-circuit ride-through and bank thermal qualification remain subject to bench evidence.",
    ])


if __name__ == "__main__":
    build()
