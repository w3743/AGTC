"""Reconstruct the 268-component regulator candidate from the frozen P5 and drawing contract."""
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[2] / "检查/联合仿真/code_as_eda"))
from reference import read_json, require, load_baseline, apply_drawing_rows, write_reference

ROOT = PROJECT.parents[3]
CONTRACT = PROJECT.parent / "闭合候选_P5Rev2/稳压P5Rev2施工合同.json"


def build_reference():
    contract = read_json(CONTRACT)
    enet = ROOT / contract["baseline"]["enet"]
    parts = load_baseline(enet)
    require(len(parts) == contract["baseline"]["component_count"] == 234, "P5 baseline count differs")
    rows = contract["drawing_components"]
    require(len({row["ref"] for row in rows}) == len(rows), "Duplicate regulator drawing refs")
    substitutions = {row["ref"]: row["replaces_ref"] for row in rows if row.get("replaces_ref")}
    require(substitutions == {"J_L_SW": "L_MAIN_A", "J_L_SC": "L_MAIN_B"},
            "Unexpected board connector substitutions")
    by_mpn = {}
    for entry in read_json(enet)["components"].values():
        props = entry["props"]
        if props.get("Manufacturer Part") and props.get("Value"):
            by_mpn.setdefault(props["Manufacturer Part"], set()).add(props["Value"])
    for ref in substitutions.values():
        require(ref in parts, f"Missing baseline substitution {ref}")
        del parts[ref]
    baseline_mpns = {ref: part["mpn"] for ref, part in parts.items()}
    apply_drawing_rows(parts, rows)
    for row in rows:
        ref = row["ref"]
        if ref in baseline_mpns and row.get("mpn") != baseline_mpns[ref] and not row.get("value"):
            parts[ref]["value"] = row.get("mpn") or ref
    # Reuse nominal values only where an unchanged MPN has one unambiguous
    # value in the frozen native P5.  Never infer them from a similar part.
    explicit_values = {row["ref"] for row in rows if row.get("value")}
    for ref, part in parts.items():
        values = by_mpn.get(part["mpn"], set())
        if ref not in explicit_values and ref.startswith(("R", "C", "L")) and len(values) == 1:
            part["value"] = next(iter(values))
            part["value_source"] = "identical MPN in frozen P5 ENET Value field"
    for index in range(1, 7):
        bulk = parts[f"C_BUS_BULK{index}"]
        require(bulk["mpn"] == "LKMI2001J102MF", "Bulk capacitor selection differs")
        bulk["value"] = "1000uF"
        bulk["value_source"] = "稳压模块/设计细节.md, LKMI2001J102MF"
    # Give the primitive parser a single, unambiguous value.  In particular,
    # µF followed by voltage/tolerance prose is otherwise read as farads.
    for ref, part in parts.items():
        if ref.startswith(("R", "C", "L")) and set(part["pins"]) == {"1", "2"}:
            match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([pnumkMµμ]?)(Ω|ohm|F|H)", part["value"])
            require(match is not None, f"{ref} lacks a traceable nominal value: {part['value']}")
            value, multiplier, unit = match.groups()
            part["value_full"] = part["value"]
            part["value"] = value + multiplier.replace("µ", "u").replace("μ", "u") + unit
    require(len(parts) == contract["target_component_count"] == 268, "Regulator count differs")
    require(all(row["ref"] in parts and
                parts[row["ref"]]["pins"] == {
                    str(pin): None if net in ("", "NC", None) else net
                    for pin, net in row["pins"].items()
                } for row in rows), "Drawing overlay pins differ")
    for ref in ("Q_MAIN_H", "Q_MAIN_L", "Q_MAIN_BH", "Q_MAIN_BL", "Q_DUMP"):
        pins = parts[ref]["pins"]
        require(set(pins) == set(map(str, range(1, 10))), f"{ref} must have all TOLL physical pins")
        require(len({pins[str(pin)] for pin in range(2, 9)}) == 1, f"{ref} source pads differ")
    expected_can = {"1": "GND_CAN", "2": "CANH", "3": "CANL"}
    require(parts["J_CAN"]["pins"] == expected_can, "J_CAN pin order differs")
    require(parts["U_AUX1"]["mpn"] == "URB4812YMD-10WR3", "Isolation supply differs")
    require(all(parts[f"C_BUS_FILM{i}"]["lcsc"] == "C5121905" for i in range(1, 4)),
            "Film capacitor freeze differs")
    write_reference(
        PROJECT, 268, CONTRACT, enet, parts,
        notes=["P5 234, plus 36 new refs and minus two replaced inductor carrier refs.",
               "Eight tap harness fuses are board-external and excluded from 268 schematic objects.",
               "TOLL MOSFETs use the contract physical pad pin numbers."],
    )


if __name__ == "__main__":
    build_reference()
