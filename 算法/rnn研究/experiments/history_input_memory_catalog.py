#!/usr/bin/env python3
"""Build a checksum-backed catalog of completed historical RNN experiments.

Only reads formal run artifacts (aggregate.json, protocol.json, summary.json,
and small input/config metadata). Checkpoints and event streams are not hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

STAGES = {
    "retention_weight_decay", "per_neuron", "representative_compute",
    "decimal_guess", "linear_readout", "binary_readout", "streaming_rtrl",
    "activity_retention",
}
RUNS_DEFAULT = Path(r"C:\pioneer\rnn-research-runs")
EXP_DEFAULT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def add_file(path: Path, root: Path, registry: dict[str, dict[str, Any]], role: str) -> str:
    key = str(path.resolve())
    if key not in registry:
        registry[key] = {"path": rel(path, root), "sha256": digest(path), "roles": []}
    if role not in registry[key]["roles"]:
        registry[key]["roles"].append(role)
    return key


def seed_from(name: str, data: dict[str, Any]) -> Any:
    if data.get("seed") is not None:
        return data["seed"]
    m = re.search(r"seed[_-]?(\d+)", name, re.I)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUNS_DEFAULT)
    ap.add_argument("--experiments-root", type=Path, default=EXP_DEFAULT)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    runs_root = args.runs_root.resolve()
    output = (args.output or runs_root / "input_memory_20261031" / "history_catalog.json").resolve()
    files: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    stage_roots: list[Path] = []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        prefix = p.name.rsplit("_2026", 1)[0]
        if prefix in STAGES or p.name in {"retention_no_decay_20261028", "per_neuron_no_decay_20261028"}:
            stage_roots.append(p)

    seen_origin: dict[str, tuple[str, str]] = {}
    for stage_root in stage_roots:
        stage = stage_root.name.rsplit("_2026", 1)[0]
        if stage == "per_neuron_no_decay":
            stage = "per_neuron"
        for aggregate in sorted(stage_root.rglob("aggregate.json")):
            run_root = aggregate.parent
            protocol = run_root / "protocol.json"
            completed = run_root / "completed.json"
            if not completed.is_file():
                # Legacy outputs sometimes record completion inside protocol metadata.
                ptmp = read_json(protocol) if protocol.is_file() else {}
                if not (isinstance(ptmp, dict) and ptmp.get("completed") is True):
                    continue
            agg = read_json(aggregate)
            if not isinstance(agg, list):
                continue
            agg_key = add_file(aggregate, runs_root, files, "aggregate")
            protocol_key = add_file(protocol, runs_root, files, "protocol") if protocol.is_file() else None
            completion_key = add_file(completed, runs_root, files, "completion_marker") if completed.is_file() else None
            # Small, explicit input/provenance documents only; never checkpoint/event files.
            input_refs = []
            for candidate in sorted(run_root.glob("*.json")):
                if candidate in (aggregate, protocol) or candidate.name.endswith("summary.json"):
                    continue
                if any(token in candidate.name.lower() for token in ("input", "manifest", "config", "budget", "matched", "mechanism", "compute", "checksum", "provenance", "validation")):
                    input_refs.append(add_file(candidate, runs_root, files, "input_or_compute_metadata"))
            pdoc = read_json(protocol) if protocol.is_file() else {}
            summaries = sorted(run_root.glob("*/summary.json"))
            summary_pairs: set[tuple[str, str]] = set()
            by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for sp in summaries:
                s = read_json(sp)
                if not isinstance(s, dict):
                    continue
                model = str(s.get("kind") or sp.parent.name.split("_")[0])
                task = str(s.get("task") or "unknown")
                skey = add_file(sp, runs_root, files, "summary")
                origin = s.get("source_run")
                origin_key = str(Path(origin).resolve()) if origin else str(run_root.resolve())
                source_group = (stage, str(run_root.resolve()))
                prior_group = seen_origin.get(origin_key)
                duplicate_of = f"{prior_group[0]}:{Path(prior_group[1]).name}" if prior_group and prior_group != source_group else None
                if not prior_group:
                    seen_origin[origin_key] = source_group
                item = {
                    "stage": stage, "run_id": run_root.name,
                    "model": model, "task": task,
                    "seed": seed_from(sp.parent.name, s),
                    "parameters": s.get("parameters"), "updates": s.get("optimizer_updates"),
                    "metrics": {k: s.get(k) for k in ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll", "count", "hits") if k in s},
                    "origin": s.get("origin"), "source_origin": origin or str(run_root.resolve()),
                    "duplicate_source_of": duplicate_of,
                    "comparison_scope": "decimal_separate" if stage == "decimal_guess" else ("primary_binary_three_tasks" if task in {"alternating", "triples", "period16"} else "historical_other_task"),
                    "summary_ref": skey, "aggregate_ref": agg_key, "protocol_ref": protocol_key,
                    "completion_ref": completion_key,
                    "input_refs": input_refs,
                    "training": {k: pdoc.get(k) for k in ("optimizer", "length", "neurons", "checkpoint_interval", "comparison", "compute", "compute_budget", "training_order") if k in pdoc},
                }
                records.append(item)
                summary_pairs.add((model, task))
                by_key.setdefault((model, task), []).append(item)
            # Include aggregate task rows even where per-seed summaries were not retained.
            for row in agg:
                if not isinstance(row, dict):
                    continue
                pair = (str(row.get("kind", "unknown")), str(row.get("task", "unknown")))
                if pair in summary_pairs:
                    continue
                records.append({"stage": stage, "run_id": run_root.name, "model": pair[0], "task": pair[1], "seed": None, "parameters": None, "updates": None,
                    "metrics": {k: row.get(k) for k in ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll", "count", "hits") if k in row},
                    "origin": row.get("origin"), "source_origin": str(run_root.resolve()), "duplicate_source_of": None,
                    "comparison_scope": "decimal_separate" if stage == "decimal_guess" else ("primary_binary_three_tasks" if pair[1] in {"alternating", "triples", "period16"} else "historical_other_task"),
                    "summary_ref": None, "aggregate_ref": agg_key, "protocol_ref": protocol_key, "completion_ref": completion_key, "input_refs": input_refs,
                    "training": {k: pdoc.get(k) for k in ("optimizer", "length", "neurons", "checkpoint_interval", "comparison", "compute", "compute_budget", "training_order") if k in pdoc}})

    # Older retention baseline used a server/session format and has no aggregate.json.
    legacy = runs_root / "retention_no_decay_20261028"
    model_summary = legacy / "model_summary.json"
    if model_summary.is_file():
        doc = read_json(model_summary) or {}
        sm_key = add_file(model_summary, runs_root, files, "legacy_summary")
        manifest = next(iter(sorted(legacy.rglob("run_manifest.json"))), None)
        manifest_key = add_file(manifest, runs_root, files, "legacy_protocol_manifest") if manifest else None
        for s in doc.get("summaries", []):
            records.append({
                "stage": "retention_no_decay_legacy", "run_id": (manifest.parent.name if manifest else legacy.name),
                "model": doc.get("model", {}).get("class", "unknown"), "task": s.get("sequence_kind", "unknown"),
                "seed": s.get("seed"), "parameters": doc.get("model", {}).get("parameter_count"),
                "updates": s.get("optimizer_updates"),
                "metrics": {"accuracy": s.get("total_accuracy"), "last_window_accuracy": s.get("last_50_accuracy"), "last_window_count": s.get("last_50_predictions"), "count": s.get("sequence_length"), "elapsed_seconds": s.get("elapsed_seconds")},
                "origin": "legacy_server_session", "source_origin": s.get("session_id"), "duplicate_source_of": None,
                "comparison_scope": "legacy_different_tasks", "summary_ref": sm_key, "aggregate_ref": None,
                "protocol_ref": manifest_key, "completion_ref": None, "input_refs": [],
                "training": {**doc.get("model", {}), **doc.get("checks", {})},
            })

    # Add source snapshots and experiment definitions as hashed provenance inputs where available.
    for p in sorted(args.experiments_root.iterdir()):
        if p.is_file() and p.suffix.lower() in {".py", ".md"}:
            add_file(p, args.experiments_root, files, "experiment_definition")
    for root in stage_roots:
        for p in root.rglob("source_snapshot/*.py"):
            add_file(p, runs_root, files, "source_snapshot")

    # Validate every referenced artifact is present in the checksum registry and unchanged.
    refs = [r for x in records for r in [x.get("summary_ref"), x.get("aggregate_ref"), x.get("protocol_ref"), x.get("completion_ref"), *(x.get("input_refs") or [])] if r]
    checked = []
    for key in refs:
        entry = files[key]
        disk = (runs_root / entry["path"]) if not Path(entry["path"]).is_absolute() else Path(entry["path"])
        actual = digest(disk)
        checked.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual, "ok": actual == entry["sha256"]})
    payload = {
        "schema_version": 1,
        "reference_root": str(runs_root),
        "experiments_root": str(args.experiments_root.resolve()),
        "comparison_policy": {"primary_binary_tasks": ["alternating", "triples", "period16"], "decimal_guess": "separate task family; excluded from binary rankings", "deduplication": "source_run path; duplicate rows marked and not treated as independent source runs"},
        "stage_roots": [str(p) for p in stage_roots],
        "run_count": len({(r["stage"], r["run_id"]) for r in records}),
        "record_count": len(records), "records": records,
        "checksums": list(files.values()),
        "reference_integrity": {"checked_reference_count": len(checked), "all_ok": all(c["ok"] for c in checked), "failures": [c for c in checked if not c["ok"]]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(f"records={len(records)} runs={payload['run_count']} checksums={len(files)} references_ok={payload['reference_integrity']['all_ok']}")


if __name__ == "__main__":
    main()
