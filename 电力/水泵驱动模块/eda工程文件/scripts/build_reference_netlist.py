"""Reconstruct the candidate netlist from read-only EasyEDA and contract inputs."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[2]
CONTRACT = PROJECT.parent / "检查/闭合候选_隔离CAN/水泵闭合施工合同.json"
JOINT = ROOT / "电力/检查/联合仿真/画图前合同完整性校验.json"
REFERENCE = PROJECT / "reference_52_netlist.json"

# Exact MPN matches from the distributor catalogue for blank legacy fields.
# These annotations do not replace any stock code supplied by the contract.
CATALOG_CODES = {
    "UHW2A471MHD": ("C1585392", "https://item.szlcsc.com/1676207.html"),
    "UHE1C331MPD": ("C445436", "https://www.lcsc.com/product-detail/C445436.html"),
    "C2012X7R2A104K125AA": ("C2167715", "https://item.szlcsc.com/2259767.html"),
    "SWPA8065S101MT": ("C334254", "https://item.szlcsc.com/312813.html"),
    "GCJ188R71H333KA12D": ("C2173059", "https://item.szlcsc.com/2265125.html"),
    "MF52A103F3950(A1)": ("C13879", "https://item.szlcsc.com/14534.html"),
    "C1608X7R1H103K080AA": ("C2180830", "https://www.lcsc.com/product-detail/C2180830.html"),
    "C1608X7R1H104K080AA": ("C72453", "https://www.lcsc.com/product-detail/C72453.html"),
}


def read_json(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def source_record(path):
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": display_path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pin_map(pins):
    return {str(pin): None if net in ("", "NC", None) else net
            for pin, net in sorted(pins.items(), key=lambda item: int(item[0]))}


def load_baseline(enet, bom, expected_count):
    parts = {}
    for entry in read_json(enet)["components"].values():
        props = entry["props"]
        ref = props["Designator"]
        require(ref not in parts, f"Duplicate baseline ref {ref}")
        ports = entry["pinInfoMap"]
        require(all(pin == port["number"] for pin, port in ports.items()), ref)
        parts[ref] = {
            "ref": ref, "mpn": props["Manufacturer Part"],
            "lcsc": props["Supplier Part"] or None,
            "lcsc_source": "published_enet" if props["Supplier Part"] else None,
            "manufacturer": props["Manufacturer"],
            "footprint": props["FootprintName"],
            "value": props["Name"] or props.get("Value", ""),
            "dnp": "DNP" in props["Name"],
            "pins": pin_map({pin: port["net"] for pin, port in ports.items()}),
            "pin_roles": {pin: port["name"] for pin, port in ports.items()
                          if pin != port["name"]},
            "source_layers": ["published_enet"],
        }
    require(len(parts) == expected_count == 49, "Expected the published 49-part enet")
    with bom.open(encoding="utf-8-sig", newline="") as handle:
        bom_refs = []
        for row in csv.DictReader(handle, delimiter="\t"):
            refs = row["Designator"].split(",")
            require(len(refs) == int(row["Quantity"]), f"BOM quantity for {refs}")
            bom_refs.extend(refs)
            for ref in refs:
                expected = (row["Manufacturer Part"], row["Footprint"], row["Supplier Part"])
                actual = (parts[ref]["mpn"], parts[ref]["footprint"], parts[ref]["lcsc"] or "")
                require(actual == expected, f"Published enet and BOM differ for {ref}")
    require(Counter(bom_refs) == Counter(parts.keys()), "Published enet/BOM refs differ")
    return parts


def overlay(parts, ref, entry, layer, *, added=False, changed_part=False):
    require((ref not in parts) == added, f"Unexpected {ref} during {layer}")
    part = parts.setdefault(ref, {"ref": ref, "dnp": False, "source_layers": []})
    for key in ("manufacturer", "footprint", "value"):
        if key in entry:
            part[key] = entry[key]
    if "new_mpn" in entry or "mpn" in entry:
        part["mpn"] = entry.get("new_mpn", entry.get("mpn"))
    if "lcsc" in entry:
        part.update(lcsc=entry["lcsc"], lcsc_source=layer)
    if "pins" in entry:
        part["pins"] = pin_map(entry["pins"])
    if added or changed_part or "pin_roles" in entry:
        part["pin_roles"] = entry.get("pin_roles", {})
    part["source_layers"].append(layer)


def build_reference():
    contract = read_json(CONTRACT)
    joint = read_json(JOINT)
    expected_count = contract["target_component_count"]
    baseline = contract["baseline"]
    enet = ROOT / baseline["published_enet"]
    bom = enet.with_name(f"{enet.stem}_BOM.csv")
    parts = load_baseline(enet, bom, baseline["published_component_count"])
    require(joint["status"] == joint["boards"]["水泵"]["status"] == "PASS", "Joint check failed")
    joint_hash = joint["contract_sha256"][CONTRACT.relative_to(ROOT / "电力").as_posix()]
    require(source_record(CONTRACT)["sha256"] == joint_hash, "Joint report is for another contract")

    changes = {entry["id"]: entry for entry in contract["changes"]}
    can = changes["CHG-PUMP-01"]
    for entry in can["removed_components"]:
        del parts[entry["ref"]]
    for entry in can["modified_components"]:
        overlay(parts, entry["ref"], entry, "CHG-PUMP-01.modified", changed_part=True)
    for entry in can["added_components"]:
        overlay(parts, entry["ref"], entry, "CHG-PUMP-01.added", added=True)
    for pin_ref, transition in can["pin_updates"].items():
        ref, pin = pin_ref.split()[0].split(".")
        old, new = transition.split(" -> ")
        require(parts[ref]["pins"][pin] == old, f"Unexpected previous net for {pin_ref}")
        parts[ref]["pins"][pin] = new.split()[0]
        parts[ref]["source_layers"].append("CHG-PUMP-01.pin_updates")

    localization = contract["localization_v1"]
    require(len(parts) == localization["board_count_before"], "CAN delta count differs")
    for ref in localization["remove"]:
        del parts[ref]
    for ref, entry in localization["add"].items():
        overlay(parts, ref, entry, "localization_v1.add", added=True)
    for ref, entry in localization["reuse_pins"].items():
        updated = {pin: net.split(" (")[0] for pin, net in entry.items() if pin.isdecimal()}
        if "pins" in entry:
            require(entry["pins"] == updated, f"Conflicting localization pins for {ref}")
        pins = parts[ref]["pins"] | pin_map(updated)
        overlay(parts, ref, entry | {"pins": pins}, "localization_v1.reuse_pins")
    require(len(parts) == localization["board_count_after"], "Localization count differs")

    for entry in contract["p0_phase_a"]["replaced_components"]:
        overlay(parts, entry["ref"], entry, "p0_phase_a", changed_part=True)

    drawing = contract["drawing_components"]
    drawing_by_ref = {entry["ref"]: entry for entry in drawing}
    for ref, change in contract["passive_freeze"]["refs"].items():
        entry = drawing_by_ref[ref]
        prior_mpn = parts[ref]["mpn"].removeprefix("风华 ")
        require(prior_mpn == change["before"]["mpn"], f"Unexpected prior part for {ref}")
        require(parts[ref]["pins"] == pin_map(change["pins_before"]) == pin_map(entry["pins"]),
                f"Passive freeze must preserve pin nets for {ref}")
        require(entry["lcsc"] == change["selected_lcsc"], f"Passive freeze C-code conflict for {ref}")
        overlay(parts, ref, entry, "passive_freeze")

    # Drawing entries form a partial bill. Compare all their pins to
    # the delta-derived map, then take their complete identity annotations.
    require(len(drawing) == len({entry["ref"] for entry in drawing})
            == joint["boards"]["水泵"]["drawing_entries"], "Drawing entry count differs")
    for entry in drawing:
        ref = entry["ref"]
        require(pin_map(entry["pins"]) == parts[ref]["pins"], f"Contract layer pin conflict for {ref}")
        if parts[ref]["lcsc"]:
            require(parts[ref]["lcsc"] == entry["lcsc"], f"Contract layer C-code conflict for {ref}")
        overlay(parts, ref, entry, "drawing_components")

    for part in parts.values():
        if not part["lcsc"] and part["mpn"] in CATALOG_CODES:
            part["lcsc"], part["lcsc_source"] = CATALOG_CODES[part["mpn"]]
        require(part["mpn"] and part["footprint"] and part["pins"], f"Incomplete {part['ref']}")
        require(part["lcsc"] is None or re.fullmatch(r"C\d+", part["lcsc"]), part["ref"])
    require(len(parts) == expected_count == 52, "Unexpected final part count")

    can_order = joint["harness"]["can_signal_to_connector_pin"]["水泵"]
    require(parts["J_CAN"]["pins"] == {pin: net for net, pin in can_order.items()}, "J_CAN mismatch")
    require(joint["pump_pinmux"]["status"] == "PASS", "Joint MCU pinmux check failed")
    controller = contract["controller_pin_verification"]
    for net, item in controller["signals"].items():
        ref, pin = item["board_connector"].split(".")
        require(parts[ref]["pins"][pin] == net, f"MCU mapping mismatch for {net}")

    return {
        "contract_id": contract["contract_id"],
        "inputs": {key: source_record(path) for key, path in
                   (("enet", enet), ("bom", bom), ("contract", CONTRACT), ("joint", JOINT))},
        "counts": {"published": 49, "after_can": localization["board_count_before"],
                   "final": len(parts), "drawing_entries": len(drawing)},
        "can_harness_pinout": parts["J_CAN"]["pins"],
        "mcu_mapping": controller["signals"],
        "hardware_scope": contract["verification_notes"]["remaining_hardware_blocker"],
        "components": [parts[ref] for ref in sorted(parts)],
    }


if __name__ == "__main__":
    reference = build_reference()
    write_json(REFERENCE, reference)
    print(f"Wrote {len(reference['components'])} reference components to {REFERENCE.name}")
