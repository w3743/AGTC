"""Repair semantic fields in a frozen input-memory history catalog.

The original catalog is read-only. This writes a separate normalized JSON next
to it, reconstructing model rows from original summaries and aggregate files,
then checks cited artifacts against their run-level historical hash manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def task_family(stage: str, task: str) -> tuple[str, str]:
    if stage in {"decimal_guess", "linear_readout"}:
        return "decimal", "decimal_separate"
    if stage == "retention_no_decay_legacy":
        return "legacy", "legacy_different_tasks"
    if task in {"alternating", "triples", "period16"}:
        return "binary", "primary_binary_three_tasks"
    return "other", "historical_other_task"


def model_name(stage: str, summary: dict, folder: str) -> str:
    mode = summary.get("mode")
    if stage == "retention_weight_decay" and mode in {"with_decay", "without_decay"}:
        return mode
    if stage == "per_neuron" and mode in {"shared", "per_neuron"}:
        return mode
    if summary.get("kind"):
        return str(summary["kind"])
    # Some legacy summary files only encode the architecture in their directory.
    if stage == "per_neuron":
        for m in ("per_neuron", "shared"):
            if folder.endswith("_" + m):
                return m
    return folder.rsplit("_seed", 1)[0]


def run_root_for(record: dict, runs_root: Path) -> Path | None:
    ref = record.get("aggregate_ref") or record.get("summary_ref") or record.get("protocol_ref")
    if not ref:
        return None
    p = Path(ref)
    if not p.is_absolute():
        p = runs_root / p
    if p.name == "summary.json":
        return p.parent.parent
    return p.parent


def normalized_metrics(s: dict) -> dict:
    keys = ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll", "count", "hits")
    out = {k: s[k] for k in keys if k in s}
    if "last_window_accuracy" in s:
        out["last_window_accuracy"] = s["last_window_accuracy"]
    if "last_50_accuracy" in s:
        out["last_window_accuracy"] = s["last_50_accuracy"]
    if "last_50_predictions" in s:
        out["last_window_count"] = s["last_50_predictions"]
    return out


def manifest_checks(records: list[dict], runs_root: Path, extra_refs: list[dict]) -> tuple[list[dict], list[dict]]:
    # Only check immutable historical run artifacts, never live workspace definitions.
    seen: dict[str, dict] = {}
    for r in records:
        for field, role in (("summary_ref", "summary"), ("aggregate_ref", "aggregate"), ("protocol_ref", "protocol"), ("input_refs", "input")):
            refs = r.get(field) or []
            if isinstance(refs, str):
                refs = [refs]
            for ref in refs:
                p = Path(ref)
                if not p.is_absolute():
                    p = runs_root / p
                # A checksum manifest cannot contain its own stable hash. It is
                # loaded as the verifier, but excluded from the set it verifies.
                if p.is_file() and p.name.lower() not in {"checksums.json", "analysis_checksums.json"}:
                    seen[str(p.resolve())] = {"path": str(p.resolve()), "role": role, "captured_sha256": sha256(p)}
    for entry in extra_refs:
        p = Path(str(entry.get("path", "")))
        if not p.is_absolute():
            p = runs_root / p
        if p.is_file():
            seen[str(p.resolve())] = {"path": str(p.resolve()), "role": "source_snapshot", "captured_sha256": sha256(p)}
    checks = []
    for key, item in seen.items():
        p = Path(key)
        run_root = None
        for anc in (p.parent, *p.parents):
            if (anc / "checksums.json").is_file() or (anc / "analysis_checksums.json").is_file():
                run_root = anc
                break
            if anc == runs_root:
                break
        if run_root is None:
            item.update({"status": "legacy_observed_hash_only" if "retention_no_decay" in key else "no_original_manifest", "manifest_sha256": None})
            checks.append(item)
            continue
        relative = p.relative_to(run_root).as_posix()
        relative_win = relative.replace("/", "\\")
        manifest_hash = None
        manifest_file = None
        for mf in (run_root / "checksums.json", run_root / "analysis_checksums.json"):
            doc = read_json(mf)
            if not isinstance(doc, dict):
                continue
            for k, v in doc.items():
                if str(k).replace("\\", "/").casefold() == relative.casefold():
                    manifest_hash, manifest_file = str(v), mf
                    break
            if manifest_hash:
                break
        if manifest_hash is None:
            status = "not_listed_in_original_manifest"
        else:
            status = "verified_match" if manifest_hash == item["captured_sha256"] else "manifest_mismatch"
        item.update({"status": status, "manifest_sha256": manifest_hash, "manifest_path": str(manifest_file.resolve()) if manifest_file else None})
        checks.append(item)
    counts: dict[str, int] = {}
    for x in checks:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    return checks, [{"status": k, "count": v} for k, v in sorted(counts.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog", type=Path, help="Frozen history_catalog.json")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    source = args.catalog.resolve()
    raw = read_json(source)
    if not isinstance(raw, dict) or not isinstance(raw.get("records"), list):
        raise ValueError("Invalid history catalog")
    runs_root = Path(raw["reference_root"])
    outpath = (args.output or source.with_name("history_catalog_normalized.json")).resolve()

    normalized: list[dict] = []
    grouped_summary_keys: set[tuple] = set()
    for old in raw["records"]:
        stage = str(old.get("stage", "unknown"))
        # Preserve legacy sessions as collected: source catalog's last-50 values are not last-100.
        if stage == "retention_no_decay_legacy":
            row = dict(old)
            row["task_family"], row["comparison_scope"] = task_family(stage, str(row.get("task", "")))
            row["metrics"] = dict(row.get("metrics") or {})
            row["parameters"] = row.get("parameters") or row.get("training", {}).get("parameter_count")
            normalized.append(row)
            continue
        summary_ref = old.get("summary_ref")
        if not summary_ref:
            continue
        sp = Path(summary_ref)
        if not sp.is_absolute():
            sp = runs_root / sp
        s = read_json(sp)
        if not isinstance(s, dict):
            continue
        task = str(s.get("task", old.get("task", "unknown")))
        model = model_name(stage, s, sp.parent.name)
        family, scope = task_family(stage, task)
        record = dict(old)
        record.update({
            "stage": stage, "model": model, "task": task, "task_family": family,
            "comparison_scope": scope, "seed": s.get("seed", old.get("seed")),
            "parameters": s.get("parameters", s.get("parameter_count", old.get("parameters"))),
            "updates": s.get("optimizer_updates", old.get("updates")),
            "metrics": normalized_metrics(s),
            "origin": s.get("origin", old.get("origin")),
            "source_origin": s.get("source_run") or old.get("source_origin"),
            "summary_ref": str(sp.resolve()),
        })
        if not record["source_origin"]:
            rr = run_root_for(old, runs_root)
            record["source_origin"] = str(rr.resolve()) if rr else old.get("run_id")
        normalized.append(record)
        grouped_summary_keys.add((stage, old.get("run_id"), model, task))

    # Rebuild the aggregate rows from source JSON to recover nested model schemas.
    aggregate_paths = set()
    for r in raw["records"]:
        if r.get("aggregate_ref"):
            p = Path(r["aggregate_ref"])
            if not p.is_absolute():
                p = runs_root / p
            aggregate_paths.add(p.resolve())
    for agg_path in sorted(aggregate_paths):
        arr = read_json(agg_path)
        if not isinstance(arr, list):
            continue
        old_meta = next((r for r in raw["records"] if r.get("aggregate_ref") and Path(r["aggregate_ref"]).resolve() == agg_path), None)
        if old_meta is None:
            continue
        stage, run_id = str(old_meta.get("stage", "unknown")), old_meta.get("run_id")
        for row in arr:
            if not isinstance(row, dict):
                continue
            task = str(row.get("task", "unknown"))
            if stage == "retention_weight_decay":
                variants = [(k, v) for k, v in row.items() if k in {"with_decay", "without_decay"} and isinstance(v, dict)]
            elif stage == "per_neuron":
                variants = [(k, v) for k, v in row.items() if k in {"shared", "per_neuron"} and isinstance(v, dict)]
            else:
                variants = [(str(row.get("kind", "unknown")), row)]
            for model, values in variants:
                if any(x["stage"] == stage and x.get("run_id") == run_id and x.get("model") == model and x.get("task") == task for x in normalized):
                    continue
                family, scope = task_family(stage, task)
                normalized.append({
                    "stage": stage, "run_id": run_id, "model": model, "task": task,
                    "task_family": family, "comparison_scope": scope, "seed": None,
                    "parameters": None, "updates": None,
                    "metrics": normalized_metrics(values), "origin": row.get("origin"),
                    "source_origin": str(agg_path.parent), "duplicate_source_of": None,
                    "summary_ref": None, "aggregate_ref": str(agg_path),
                    "protocol_ref": str(agg_path.parent / "protocol.json") if (agg_path.parent / "protocol.json").is_file() else None,
                    "input_refs": [], "training": {}, "aggregate_only": True,
                })

    # Mark a repeated source as dependent evidence only when the same origin,
    # model, task and seed occur under a different historical stage/run.
    prior_sources: dict[tuple, tuple[str, str]] = {}
    for record in normalized:
        identity = (record.get("source_origin"), record.get("model"), record.get("task"), record.get("seed"))
        group = (str(record.get("stage")), str(record.get("run_id")))
        prior = prior_sources.get(identity)
        if identity[0] and prior and prior != group:
            record["duplicate_source_of"] = f"{prior[0]}:{prior[1]}"
        elif identity[0] and not prior:
            prior_sources[identity] = group

    snapshot_refs = [x for x in raw.get("checksums", []) if "source_snapshot" in (x.get("roles") or [])]
    checks, check_counts = manifest_checks(normalized, runs_root, snapshot_refs)
    original_records = raw["records"]
    payload = {
        "schema_version": 1, "normalization": "input_memory_history_v1",
        "source_catalog": str(source), "source_catalog_sha256": sha256(source),
        "reference_root": str(runs_root), "original_record_count": len(original_records),
        "record_count": len(normalized), "records": normalized,
        "original_records": original_records,
        "checksum_verification": {"counts": check_counts, "files": checks,
            "live_experiment_definitions_excluded": True,
            "legacy_note": "retention_no_decay_legacy has no original run checksum manifest; captured SHA-256 values are collection-time hashes only."},
    }
    outpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(outpath)
    print(f"original_records={len(original_records)} normalized_records={len(normalized)} checksum_files={len(checks)} statuses={check_counts}")


if __name__ == "__main__":
    main()
