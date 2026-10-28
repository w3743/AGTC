"""Check official API readback against the maintained pin contract (no EDA writes)."""
import argparse
import collections
import json
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return data.get("value", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True)
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--expected", default=str(Path(__file__).parents[1] / "原理图准备/当前原理图连接.json"))
    args = parser.parse_args()
    actual = load(args.actual)
    netlist = load(args.netlist)
    if "content" in netlist:
        netlist = json.loads(netlist["content"])
    expected = load(args.expected)
    net_components = list(netlist["components"].values())
    refs = [c["props"].get("Designator") for c in net_components]
    lookup = {c["props"].get("Designator"): c for c in net_components}
    geometry = {c["ref"]: c for c in actual["components"]}
    wanted = {c["ref"]: c for c in expected["components"]}
    failures = []
    if len(refs) != len(set(refs)):
        failures.append({"reason": "duplicate designators"})
    if set(lookup) != set(wanted) or set(geometry) != set(wanted):
        failures.append({"reason": "component set mismatch", "missing": sorted(set(wanted)-set(lookup)), "extra": sorted(set(lookup)-set(wanted))})
    pins_checked = nc_count = 0
    nets = collections.defaultdict(list)
    for ref, target in wanted.items():
        if ref not in lookup or ref not in geometry:
            continue
        component = lookup[ref]
        actual_pins = component["pinInfoMap"]
        geometric_pins = {p["pin"]: p for p in geometry[ref]["pins"]}
        if set(target["pins"]) != set(actual_pins) or set(target["pins"]) != set(geometric_pins):
            failures.append({"ref": ref, "reason": "pin set mismatch"})
        for pin, net in target["pins"].items():
            pins_checked += 1
            found = actual_pins.get(pin, {}).get("net") or ""
            if (net or "") != found:
                failures.append({"ref": ref, "pin": pin, "expected": net, "actual": found})
            marked_nc = bool(geometric_pins.get(pin, {}).get("noConnect"))
            if bool(net) == marked_nc:
                failures.append({"ref": ref, "pin": pin, "reason": "net / NC marker mismatch"})
            nc_count += marked_nc
            if found:
                nets[found].append(f"{ref}.{pin}")
        for target_key, actual_key in [("manufacturerId", "Manufacturer Part"), ("addIntoBom", "Add into BOM"), ("addIntoPcb", "Convert to PCB")]:
            if target_key not in target:
                continue
            value = component["props"].get(actual_key)
            if isinstance(target[target_key], bool):
                value = str(value).lower() == "yes"
            if value != target[target_key]:
                failures.append({"ref": ref, "reason": target_key, "expected": target[target_key], "actual": value})
        if target.get("addIntoPcb") and not component["props"].get("Footprint"):
            failures.append({"ref": ref, "reason": "PCB component missing footprint"})
    single_nets = {net: endpoints for net, endpoints in nets.items() if len(endpoints) == 1}
    # PACK+ is the source side of the external main fuse, not an extra PCB terminal.
    unexpected_single = {net: pins for net, pins in single_nets.items() if not (net == "PACK+" and pins == ["F_MAIN.1"])}
    if unexpected_single:
        failures.append({"reason": "unexpected one-pin nets", "nets": unexpected_single})
    report = {"status": "PASS_STATIC_PIN_CONTRACT" if not failures else "FAIL", "scope": "official netlist + API pin / population readback; excludes DRC warning details, dynamic behavior, firmware and hardware acceptance", "components": len(lookup), "pinsChecked": pins_checked, "ncPins": nc_count, "connectedPins": sum(map(len, nets.values())), "nets": len(nets), "onePinNets": single_nets, "failures": failures}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
