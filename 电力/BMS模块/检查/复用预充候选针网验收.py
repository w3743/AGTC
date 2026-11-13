"""Read-only comparison of an official BMS candidate ENET to its contract.

Reads the frozen baseline ENET, contract, candidate ENET and optional paired
PCB files. Writes only the requested JSON evidence report; it never edits EDA
projects or board data.
"""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BMS = HERE.parent
BASELINE = HERE / "预充合闸候选基线" / "精简候选_P1.enet"
CONTRACT = BMS / "原理图准备" / "复用预充合并施工合同.json"
DEFAULT_REPORT = HERE / "复用预充候选针网验收.json"

# These two contract strings combine a manufacturer/catalogue description
# with a supplier SKU. Keep the split explicit; never strip prefixes globally.
IDENTITY_NORMALIZATIONS = {
    "Q_PROBE": {
        "contract_mpn": "MMBT5551 MDD C408396",
        "manufacturer_part": "MMBT5551",
        "manufacturer": "MDD",
        "supplier_part": "C408396",
        "reason": "Contract text combines MPN, manufacturer and supplier SKU.",
    },
    "R_PROBE": {
        "contract_mpn": "RN 3WS 820R F T/B A1 C433538",
        "manufacturer_part": "RN 3WS 820R F T/B A1",
        "manufacturer": None,
        "supplier_part": "C433538",
        "reason": "Contract text combines catalogue description and supplier SKU.",
    },
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_text(value):
    """Identity fields stay exact; only missing/null is represented as empty."""
    return "" if value is None else str(value)


def pin_net(value):
    if isinstance(value, dict):
        value = value.get("net")
    return "" if value is None or value == "" else str(value)


def enet_components(path):
    data = read_json(path)
    components = data.get("components")
    if not isinstance(components, dict):
        raise ValueError("ENET has no components object: " + str(path))

    records = {}
    duplicates = []
    missing_designators = []
    for item in components.values():
        props = item.get("props") or {}
        ref = exact_text(props.get("Designator"))
        if not ref:
            missing_designators.append(item.get("uuid") or item.get("id") or "<unknown>")
            continue
        pins = {
            str(pin): pin_net(value)
            for pin, value in (item.get("pinInfoMap") or {}).items()
        }
        record = {
            "pins": pins,
            "manufacturer_part": exact_text(props.get("Manufacturer Part")),
            "supplier_part": exact_text(props.get("Supplier Part")),
            "manufacturer": exact_text(props.get("Manufacturer")),
        }
        if ref in records:
            duplicates.append(ref)
        else:
            records[ref] = record
    return records, sorted(set(duplicates)), missing_designators


def normalized_contract_identity(ref, part):
    raw = exact_text(part.get("mpn"))
    override = IDENTITY_NORMALIZATIONS.get(ref)
    if override is not None:
        if raw != override["contract_mpn"]:
            return None, {
                "code": "identity_normalization_contract_drift",
                "ref": ref,
                "expected_contract_mpn": override["contract_mpn"],
                "actual_contract_mpn": raw,
            }
        return {
            "contract_mpn_raw": raw,
            "manufacturer_part": override["manufacturer_part"],
            "supplier_part": override["supplier_part"],
            "manufacturer": override["manufacturer"],
            "normalization": override["reason"],
        }, None
    return {
        "contract_mpn_raw": raw,
        "manufacturer_part": raw,
        "supplier_part": exact_text(part.get("supplier")) or None,
        "manufacturer": exact_text(part.get("manufacturer")) or None,
        "normalization": None,
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_enet", type=Path, help="official candidate ENET export")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON evidence output")
    parser.add_argument("--source-pcb", type=Path)
    parser.add_argument("--candidate-pcb", type=Path)
    args = parser.parse_args()

    protected_paths = [BASELINE, CONTRACT, args.candidate_enet]
    if args.source_pcb is not None:
        protected_paths.append(args.source_pcb)
    if args.candidate_pcb is not None:
        protected_paths.append(args.candidate_pcb)
    report_path = args.report.resolve()
    if report_path in {path.resolve() for path in protected_paths}:
        parser.error("--report must not overwrite an input artifact")

    issues = []
    identity_gaps = []
    identity_audit = []
    pcb_comparison = None
    contract = read_json(CONTRACT)
    baseline_hash = sha256(BASELINE)
    candidate_hash = sha256(args.candidate_enet)
    expected_hash = exact_text(contract.get("baselineSha256")).lower()
    if expected_hash != baseline_hash.lower():
        issues.append({
            "code": "baseline_hash_mismatch",
            "contract_sha256": expected_hash,
            "actual_baseline_sha256": baseline_hash,
        })

    old, old_duplicates, old_missing = enet_components(BASELINE)
    new, new_duplicates, new_missing = enet_components(args.candidate_enet)
    if old_duplicates:
        issues.append({"code": "duplicate_baseline_designator", "refs": old_duplicates})
    if old_missing:
        issues.append({"code": "baseline_missing_designator", "instances": old_missing})
    if new_duplicates:
        issues.append({"code": "duplicate_candidate_designator", "refs": new_duplicates})
    if new_missing:
        issues.append({"code": "candidate_missing_designator", "instances": new_missing})

    merged = contract.get("mergedPinNets")
    if not isinstance(merged, dict):
        raise ValueError("contract has no mergedPinNets object")
    expected_refs = set(merged)
    actual_refs = set(new)
    if expected_refs - actual_refs:
        issues.append({"code": "missing_instances", "refs": sorted(expected_refs - actual_refs)})
    if actual_refs - expected_refs:
        issues.append({"code": "unexpected_instances", "refs": sorted(actual_refs - expected_refs)})

    added_list = contract.get("added") or []
    added = {}
    added_duplicates = []
    for item in added_list:
        ref = exact_text(item.get("ref"))
        if ref in added:
            added_duplicates.append(ref)
        else:
            added[ref] = item
    if added_duplicates:
        issues.append({"code": "duplicate_contract_added_ref", "refs": sorted(set(added_duplicates))})
    expected_added = expected_refs - set(old)
    if set(added) != expected_added:
        issues.append({
            "code": "contract_added_list_mismatch",
            "expected_added": sorted(expected_added),
            "contract_added": sorted(added),
        })

    replacement_refs = set()
    for item in contract.get("replaced") or []:
        ref = exact_text(item).split(maxsplit=1)[0] if exact_text(item).strip() else ""
        if ref:
            replacement_refs.add(ref)
    invalid_replacements = sorted(replacement_refs - (set(old) & expected_refs))
    if invalid_replacements:
        issues.append({"code": "replacement_ref_not_in_baseline_and_contract", "refs": invalid_replacements})

    for ref in sorted(expected_refs & actual_refs):
        wanted = merged[ref]
        got = new[ref]
        expected_pins = {str(k): pin_net(v) for k, v in (wanted.get("pins") or {}).items()}
        actual_pins = got["pins"]
        if set(expected_pins) != set(actual_pins):
            issues.append({
                "code": "pin_number_set_mismatch",
                "ref": ref,
                "expected": sorted(expected_pins),
                "actual": sorted(actual_pins),
            })
        pin_diffs = [
            {"pin": pin, "expected": expected_pins.get(pin, "<missing>"),
             "actual": actual_pins.get(pin, "<missing>")}
            for pin in sorted(set(expected_pins) | set(actual_pins), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
            if expected_pins.get(pin, "<missing>") != actual_pins.get(pin, "<missing>")
        ]
        if pin_diffs:
            issues.append({"code": "pin_net_mismatch", "ref": ref, "pins": pin_diffs})

        identity, identity_error = normalized_contract_identity(ref, wanted)
        if identity_error:
            issues.append(identity_error)
            identity = {
                "contract_mpn_raw": exact_text(wanted.get("mpn")),
                "manufacturer_part": exact_text(wanted.get("mpn")),
                "supplier_part": None,
                "manufacturer": None,
                "normalization": None,
            }

        is_old = ref in old
        is_replacement = ref in replacement_refs
        before = old.get(ref)
        added_spec = added.get(ref)
        if not is_old and added_spec:
            added_mpn = exact_text(added_spec.get("mpn"))
            if added_mpn and added_mpn != exact_text(wanted.get("mpn")):
                # The Q/R descriptions have deliberate canonical splits.
                override = IDENTITY_NORMALIZATIONS.get(ref)
                if not override or added_mpn != override["contract_mpn"]:
                    issues.append({"code": "added_mpn_contract_inconsistency", "ref": ref,
                                   "merged_mpn": exact_text(wanted.get("mpn")), "added_mpn": added_mpn})

        expected_mpn = identity["manufacturer_part"]
        if not expected_mpn:
            identity_gaps.append({"ref": ref, "field": "Manufacturer Part", "reason": "contract has no effective MPN"})
        elif got["manufacturer_part"] != expected_mpn:
            issues.append({"code": "manufacturer_part_mismatch", "ref": ref,
                           "expected": expected_mpn, "actual": got["manufacturer_part"]})

        # Unchanged baseline items must preserve their exact stored identity.
        # No manufacturer-prefix stripping or case folding is applied.
        if is_old and not is_replacement:
            if exact_text(wanted.get("mpn")) != before["manufacturer_part"]:
                issues.append({"code": "unapproved_contract_mpn_change", "ref": ref,
                               "baseline": before["manufacturer_part"],
                               "contract": exact_text(wanted.get("mpn"))})
            for field in ("manufacturer_part", "supplier_part", "manufacturer"):
                if got[field] != before[field]:
                    issues.append({"code": "unchanged_identity_changed", "ref": ref, "field": field,
                                   "baseline": before[field], "candidate": got[field]})

        # Supplier Part is procurement identity and is audited separately from
        # the CAD Manufacturer Part. Added parts use only an explicit supplier
        # code; replacements without one are surfaced as unresolved.
        expected_supplier = identity.get("supplier_part")
        supplier_source = "explicit normalization" if ref in IDENTITY_NORMALIZATIONS else None
        if not supplier_source and added_spec and exact_text(added_spec.get("supplier")):
            expected_supplier = exact_text(added_spec.get("supplier"))
            supplier_source = "contract.added.supplier"
        if not supplier_source and exact_text(wanted.get("supplier")):
            expected_supplier = exact_text(wanted.get("supplier"))
            supplier_source = "contract.mergedPinNets.supplier"
        if is_old and not is_replacement:
            expected_supplier = before["supplier_part"]
            supplier_source = "frozen baseline; unchanged ref"
        if supplier_source:
            if got["supplier_part"] != expected_supplier:
                issues.append({"code": "supplier_part_mismatch", "ref": ref,
                               "source": supplier_source, "expected": expected_supplier,
                               "actual": got["supplier_part"]})
        else:
            identity_gaps.append({
                "ref": ref,
                "field": "Supplier Part",
                "reason": "contract does not specify a supplier SKU",
                "baseline": before["supplier_part"] if before else None,
                "candidate": got["supplier_part"],
            })

        expected_manufacturer = identity.get("manufacturer")
        manufacturer_source = "explicit normalization" if ref in IDENTITY_NORMALIZATIONS and expected_manufacturer else None
        if not manufacturer_source and added_spec and added_spec.get("manufacturer"):
            expected_manufacturer = exact_text(added_spec.get("manufacturer"))
            manufacturer_source = "contract.added.manufacturer"
        if is_old and not is_replacement:
            expected_manufacturer = before["manufacturer"]
            manufacturer_source = "frozen baseline; unchanged ref"
        if manufacturer_source and got["manufacturer"] != expected_manufacturer:
            issues.append({"code": "manufacturer_metadata_mismatch", "ref": ref,
                           "source": manufacturer_source, "expected": expected_manufacturer,
                           "actual": got["manufacturer"]})

        identity_audit.append({
            "ref": ref,
            "origin": "baseline" if is_old else "added",
            "replacement_allowed": bool(is_replacement),
            "cad_carrier_identity": {
                "contract_mpn_raw": identity["contract_mpn_raw"],
                "expected_manufacturer_part": expected_mpn,
                "candidate_manufacturer_part": got["manufacturer_part"],
                "baseline_manufacturer_part": before["manufacturer_part"] if before else None,
                "normalization": identity["normalization"],
            },
            "purchase_identity": {
                "expected_supplier_part": expected_supplier,
                "supplier_source": supplier_source,
                "candidate_supplier_part": got["supplier_part"],
                "baseline_supplier_part": before["supplier_part"] if before else None,
            },
            "manufacturer_metadata": {
                "expected": expected_manufacturer,
                "source": manufacturer_source,
                "candidate": got["manufacturer"],
                "baseline": before["manufacturer"] if before else None,
            },
        })

    if (args.source_pcb is None) != (args.candidate_pcb is None):
        issues.append({"code": "pcb_pair_incomplete", "message": "supply both PCB paths or neither"})
    elif args.source_pcb is not None:
        source_pcb_hash = sha256(args.source_pcb)
        candidate_pcb_hash = sha256(args.candidate_pcb)
        pcb_comparison = {
            "source_path": str(args.source_pcb.resolve()),
            "candidate_path": str(args.candidate_pcb.resolve()),
            "source_sha256": source_pcb_hash,
            "candidate_sha256": candidate_pcb_hash,
            "identical": source_pcb_hash == candidate_pcb_hash,
        }
        if source_pcb_hash != candidate_pcb_hash:
            issues.append({"code": "pcb_hash_mismatch", **pcb_comparison})

    if issues:
        status = "FAIL"
    elif identity_gaps:
        status = "PASS_STATIC_ENET_WITH_IDENTITY_GAPS"
    else:
        status = "PASS_STATIC_ENET_ONLY"
    report = {
        "status": status,
        "contract_path": str(CONTRACT),
        "contract_file_sha256": sha256(CONTRACT),
        "baseline_path": str(BASELINE),
        "contract_baseline_sha256": expected_hash,
        "actual_baseline_sha256": baseline_hash,
        "candidate_path": str(args.candidate_enet.resolve()),
        "candidate_sha256": candidate_hash,
        "expected_instances": len(expected_refs),
        "candidate_instances": len(new),
        "replacement_refs": sorted(replacement_refs),
        "optional_pcb_comparison": pcb_comparison,
        "identity_normalizations": IDENTITY_NORMALIZATIONS,
        "identity_audit": identity_audit,
        "identity_gaps": identity_gaps,
        "issues": issues,
        "scope": "Static ENET pins and stated identity fields, plus optional paired PCB SHA256. "
                 "Does not prove EDA source save/reopen, DRC, layout, manufacturing, or hardware acceptance.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(status, len(issues), "issues;", len(identity_gaps), "identity gaps")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
