"""Independent audit for an input-memory RTRL benchmark run.

Run with the experiment Python:
    python audit_input_memory.py RUN_DIRECTORY

The audit validates raw run artifacts, source and history-catalog hashes,
metrics, checkpoints, frozen-source replay, old binary baseline parity, and
recomputed operation counters. It writes audit.json only after the audit ends.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

KINDS = ("baseline_rtrl20", "fixed_input_memory20", "learned_input_memory20")
TASKS = (
    "binary_alternating", "binary_triples", "binary_period16",
    "decimal_cycle", "decimal_triples", "decimal_zigzag",
    "decimal_fibonacci_mod10", "decimal_switching", "decimal_iid_random",
)
SEEDS = (2, 20261027, 20261028)
PREVIOUS = Path(r"C:\pioneer\rnn-research-runs\streaming_rtrl_20261028\run_20261028T142841_111140Z")
CATALOG_DEFAULT = Path(r"C:\pioneer\rnn-research-runs\input_memory_20261031\history_catalog.json")
SOURCE_COUNT = 17
EXPERIMENT_DIR = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(a, b, *, atol=1e-10, rtol=1e-10):
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(x) for x in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(message)


def verify_checksums(root: Path, manifest_path: Path, *, allowed_missing=()):
    manifest = read_json(manifest_path)
    checked, missing = 0, []
    for rel, expected in manifest.items():
        path = root / Path(rel)
        if not path.is_file():
            missing.append(rel)
            continue
        if digest(path) != expected:
            raise AssertionError(f"checksum mismatch: {path}")
        checked += 1
    unexpected = sorted(set(missing) - set(allowed_missing))
    if unexpected:
        raise AssertionError(f"checksum manifest references missing files: {unexpected[:5]}")
    return checked


def run_frozen(snapshot: Path, action: str, request=None) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), action, str(snapshot)],
        input=None if request is None else json.dumps(request, separators=(",", ":")),
        text=True, capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    if proc.returncode:
        raise RuntimeError(f"{action} subprocess failed: {proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


def _build(kind: str, seed: int):
    from input_memory_rtrl import InputMemoryRTRL
    from streaming_rtrl import StreamingRTRL
    if kind == "baseline_rtrl20":
        return StreamingRTRL(seed=seed, credit="rtrl")
    return InputMemoryRTRL(seed=seed, mode="fixed" if kind == "fixed_input_memory20" else "learned")


def _frozen_replay_main(snapshot: Path):
    sys.path.insert(0, str(snapshot))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from input_memory_benchmark import sequences

    requests = json.load(sys.stdin)
    result = []
    for request in requests:
        kind, seed = request["kind"], int(request["seed"])
        sequence = request["sequence"]
        model = _build(kind, seed)
        initial_parameters = {name: value.detach().reshape(-1).tolist()
                              for name, value in model.network.state_dict().items()}
        rows = []
        for i, target in enumerate(sequence):
            probabilities = model.predict().detach().tolist()
            before = (model.input_retention.detach().reshape(-1).tolist()
                      if hasattr(model, "input_retention") else [])
            r_before = model.network.trace_decay.detach().reshape(-1).tolist()
            stats = model.learn_target(int(target)) if i else None
            after = (model.input_retention.detach().reshape(-1).tolist()
                     if hasattr(model, "input_retention") else [])
            model.advance(int(target))
            rows.append({
                "probabilities": probabilities,
                "loss": None if stats is None else stats["loss"],
                "grad_norm": None if stats is None else stats["grad_norm"],
                "retention_grad_norm": None if stats is None else stats["retention_grad_norm"],
                "retention_grad": [] if stats is None else stats["retention_grad"],
                "input_retention_grad_norm": (None if stats is None else
                                               stats.get("input_retention_grad_norm", 0.0)),
                "input_retention_grad": (
                    [] if stats is None else
                    stats.get("input_retention_grad", [0.0] * 30)),
                "input_retention_before": before, "input_retention_after": after,
                "r_before": r_before,
                "r_after": model.network.trace_decay.detach().reshape(-1).tolist(),
                "memory_norm": float(model.memory.norm()) if hasattr(model, "memory") else 0.0,
                "memory_grad_norm": None if stats is None else stats.get("memory_grad_norm", 0.0),
                "em_norm": float(model.em.norm()) if hasattr(model, "em") else 0.0,
                "eh_norm": float(model.eh.norm()), "ef_norm": float(model.ef.norm()),
                "steps": model.steps, "updates_after": model.updates,
            })
        result.append({"kind": kind, "seed": seed, "rows": rows,
                       "initial_parameters": initial_parameters,
                       "generated_sequences": sequences()})
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


def _counter_main(snapshot: Path):
    sys.path.insert(0, str(snapshot))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from input_memory_benchmark import sequences
    from streaming_rtrl_compute import counted, process

    seq = sequences()["binary_period16"]
    result = {}
    for kind in KINDS:
        profiles = {}
        for age in (0, 1, 2, 32):
            model = _build(kind, 73)
            for digit in seq[:age]:
                process(model, digit)
            profiles[str(age)] = counted(lambda: process(model, seq[age]))
        result[kind] = {
            "profiles": profiles,
            "full_run_proxy": (profiles["0"]["total_proxy"] + profiles["1"]["total_proxy"]
                               + 598 * profiles["32"]["total_proxy"]),
        }
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


def verify_catalog(root: Path, catalog_path: Path) -> int:
    catalog = read_json(catalog_path)
    if catalog.get("schema_version") != 1:
        raise AssertionError("history catalog schema_version is not 1")
    if catalog.get("normalization") == "input_memory_history_v1":
        source_catalog = Path(catalog["source_catalog"])
        expected_source_hash = catalog.get("source_catalog_sha256")
        if not source_catalog.is_file() or digest(source_catalog) != expected_source_hash:
            raise AssertionError("normalized catalog source hash mismatch")
        run_copy = root / "history_catalog.json"
        if run_copy.is_file() and digest(run_copy) != expected_source_hash:
            raise AssertionError("normalized catalog does not match run's raw catalog copy")
        raw_catalog = read_json(source_catalog)
        raw_root = Path(raw_catalog["reference_root"])
        verification = catalog.get("checksum_verification", {})
        items = verification.get("files", [])
        counts = {item["status"]: int(item["count"]) for item in verification.get("counts", [])}
        assert_equal(sum(counts.values()), len(items), "normalized catalog status counts differ")
        allowed_statuses = {"verified_match", "legacy_observed_hash_only",
                            "not_listed_in_original_manifest", "no_original_manifest",
                            "manifest_document_hash_captured"}
        if set(counts) - allowed_statuses:
            raise AssertionError(f"unexpected normalized checksum status: {set(counts)-allowed_statuses}")
        checked_paths = {}
        for item in items:
            path = Path(item["path"])
            if not path.is_file() or digest(path) != item.get("captured_sha256"):
                raise AssertionError(f"normalized historical hash mismatch: {path}")
            status = item["status"]
            manifest_path = item.get("manifest_path")
            if status == "verified_match":
                if not manifest_path or not Path(manifest_path).is_file():
                    raise AssertionError(f"missing owning checksum manifest: {manifest_path}")
                manifest = read_json(Path(manifest_path))
                manifest_hash = item.get("manifest_sha256")
                if manifest_hash != item["captured_sha256"]:
                    raise AssertionError(f"normalized manifest hash differs: {path}")
                try:
                    rel = path.resolve().relative_to(Path(manifest_path).parent.resolve()).as_posix()
                except ValueError as exc:
                    raise AssertionError(f"historical file is outside owning run: {path}") from exc
                normalized_manifest = {key.replace("\\", "/"): value
                                       for key, value in manifest.items()}
                if normalized_manifest.get(rel) != manifest_hash:
                    raise AssertionError(f"owning run checksum manifest mismatch: {path}")
            elif status == "not_listed_in_original_manifest":
                if item.get("manifest_sha256") is not None:
                    raise AssertionError(f"unlisted historical file unexpectedly has manifest hash: {path}")
            elif status == "legacy_observed_hash_only":
                # A few early records have no checksums.json; the normalized
                # catalog preserves their observed hash and the live hash above.
                if item.get("manifest_sha256") is not None:
                    raise AssertionError(f"legacy hash-only item has a manifest hash: {path}")
            elif status == "no_original_manifest":
                if item.get("manifest_sha256") is not None or item.get("manifest_path") is not None:
                    raise AssertionError(f"no-manifest history file unexpectedly names a manifest: {path}")
            elif status == "manifest_document_hash_captured":
                if path.name.lower() not in {"checksums.json", "analysis_checksums.json"}:
                    raise AssertionError(f"manifest-captured status is not a manifest file: {path}")
                if item.get("manifest_sha256") is not None:
                    raise AssertionError(f"manifest document cannot contain its own hash: {path}")
            checked_paths[str(path.resolve()).casefold()] = item["captured_sha256"]
        # The normalizer verifies immutable references used in its repaired
        # rows; the raw catalog also references completion markers that its
        # record builder intentionally omits. Verify every non-live raw
        # catalog artifact against its captured hash, including those markers.
        raw_historical = 0
        for entry in raw_catalog.get("checksums", []):
            roles = set(entry.get("roles", []))
            if roles == {"experiment_definition"}:
                continue
            raw_path = Path(entry["path"])
            if not raw_path.is_absolute():
                raw_path = raw_root / raw_path
            if raw_path.name.lower() in {"checksums.json", "analysis_checksums.json"}:
                continue
            if not raw_path.is_file() or digest(raw_path) != entry.get("sha256"):
                raise AssertionError(f"raw history catalog hash mismatch: {raw_path}")
            raw_historical += 1
            path_key = str(raw_path.resolve()).casefold()
            checked_paths[path_key] = entry["sha256"]
            owner = None
            for ancestor in (raw_path.parent, *raw_path.parents):
                candidate = ancestor / "checksums.json"
                if candidate.is_file():
                    owner = candidate
                    break
                if ancestor == raw_root:
                    break
            if owner is not None:
                relative = raw_path.resolve().relative_to(owner.parent.resolve()).as_posix()
                manifest = read_json(owner)
                normalized_manifest = {key.replace("\\", "/"): value
                                       for key, value in manifest.items()}
                manifest_value = normalized_manifest.get(relative)
                if manifest_value is not None and manifest_value != entry["sha256"]:
                    raise AssertionError(f"raw history run manifest mismatch: {raw_path}")
        assert_equal(len(catalog.get("records", [])), catalog.get("record_count"),
                     "normalized history record count differs")
        refs_checked = 0
        for record in catalog.get("records", []):
            refs = [record.get(key) for key in ("summary_ref", "aggregate_ref", "protocol_ref", "completion_ref")]
            refs.extend(record.get("input_refs") or [])
            for ref in filter(None, refs):
                ref_path = Path(ref)
                if ref_path.name.lower() in {"checksums.json", "analysis_checksums.json"}:
                    continue
                if str(ref_path.resolve()).casefold() not in checked_paths:
                    raise AssertionError(f"normalized history record lacks checked hash: {ref}")
                refs_checked += 1
        return {"files": raw_historical, "normalized_files": len(items),
                "references": refs_checked, "statuses": counts}

    entries = catalog.get("checksums", [])
    assert_equal(len(entries), 632, "history catalog should reference 632 files")
    if not catalog.get("reference_integrity", {}).get("all_ok"):
        raise AssertionError("history catalog records failed reference integrity")
    reference_root = Path(catalog["reference_root"])
    seen = set()
    checked = 0
    for item in entries:
        rel = item.get("path")
        roles = set(item.get("roles", []))
        # Experiment definitions are live workspace files, not immutable
        # historical run artifacts. Their recorded hash can change after the
        # catalog was produced. A file also classified as historical (for
        # example a frozen source snapshot) is still checked below.
        if roles == {"experiment_definition"}:
            continue
        if not rel or Path(rel).is_absolute():
            raise AssertionError(f"history catalog path should be relative: {rel}")
        lowered = rel.lower().replace("\\", "/")
        if "checkpoint" in lowered or "events.jsonl" in lowered:
            raise AssertionError(f"history catalog unexpectedly references raw checkpoint/event data: {rel}")
        if rel in seen:
            raise AssertionError(f"duplicate history catalog path: {rel}")
        seen.add(rel)
        path = reference_root / Path(rel)
        if not path.is_file() or digest(path) != item.get("sha256"):
            raise AssertionError(f"history catalog hash mismatch/missing: {path}")
        checked += 1
    by_hash_path = {item["path"]: item for item in entries}
    checked_refs = 0
    for record in catalog.get("records", []):
        refs = [record.get(key) for key in ("summary_ref", "aggregate_ref", "protocol_ref", "completion_ref")]
        refs.extend(record.get("input_refs") or [])
        for ref in filter(None, refs):
            item = by_hash_path.get(ref)
            if item is None:
                raise AssertionError(f"history record references unhashed artifact: {ref}")
            path = reference_root / Path(ref)
            if digest(path) != item["sha256"]:
                raise AssertionError(f"history reference hash mismatch: {path}")
            checked_refs += 1
    expected_ref_count = catalog.get("reference_integrity", {}).get("checked_reference_count")
    assert_equal(checked_refs, expected_ref_count, "history catalog checked reference count differs")
    return {"files": checked, "references": checked_refs, "statuses": {"catalog_hash": checked}}


def audit(root: Path, catalog_override: Path | None = None) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = root.resolve()
    protocol = read_json(root / "protocol.json")
    sequences_doc = read_json(root / "input_sequences.json")
    completed = read_json(root / "completed.json")
    assert_equal(completed.get("success"), True, "completed.success is not true")
    assert_equal(completed.get("new_runs"), 81, "completed run count differs")
    assert_equal(completed.get("new_predictions"), 48600, "completed prediction count differs")
    assert_equal(protocol.get("runs"), 81, "protocol run count differs")
    assert_equal(protocol.get("predictions"), 48600, "protocol prediction count differs")
    assert_equal(protocol.get("length"), 600, "protocol length differs")
    assert_equal(set(protocol.get("kinds", [])), set(KINDS), "protocol kinds differ")
    assert_equal(set(protocol.get("tasks", [])), set(TASKS), "protocol tasks differ")
    assert_equal(protocol.get("input_retention_initial"), 0.8, "input retention initial value differs")
    assert_equal(protocol.get("input_memory_sources"),
                 "10 external onehot channels + 20 previous neuron outputs; one state per source shared across receivers",
                 "input memory source definition differs")
    assert_equal(protocol.get("history_buffer"), False, "protocol unexpectedly enables history buffer")

    new_checked = verify_checksums(root, root / "checksums.json",
                                   allowed_missing=("checksums.json", "audit.json"))
    checksum_keys = set(read_json(root / "checksums.json"))
    required_raw = [p for p in root.glob("*_seed*/*") if p.is_file()
                    and (p.name in ("events.jsonl", "summary.json") or p.name.startswith("checkpoint_"))]
    omitted = [str(p.relative_to(root)) for p in required_raw
               if str(p.relative_to(root)) not in checksum_keys]
    if omitted:
        raise AssertionError(f"raw run files missing from checksums: {omitted[:5]}")

    source_hashes = protocol.get("source_sha256", {})
    assert_equal(len(source_hashes), SOURCE_COUNT, "expected 17 frozen source hashes")
    snapshot = root / "source_snapshot"
    snapshot_files = {p.name for p in snapshot.iterdir() if p.is_file()}
    assert_equal(set(source_hashes), snapshot_files, "source snapshot file set differs")
    for name, expected in source_hashes.items():
        if digest(snapshot / name) != expected:
            raise AssertionError(f"source snapshot hash mismatch: {name}")
    required_sources = {
        "input_memory_rtrl.py", "input_memory_benchmark.py", "test_input_memory_rtrl.py",
        "streaming_rtrl.py", "streaming_rtrl_benchmark.py", "streaming_rtrl_compute.py",
        "linear_readout.py", "retention_weight_decay.py", "representative_benchmark.py",
        "per_neuron_retention.py", "representative_recurrent.py", "representative_sequence.py",
        "compute_budget.py", "decimal_benchmark.py", "decimal_tasks.py", "model.py", "requirements.txt",
    }
    assert_equal(set(source_hashes), required_sources, "frozen source list differs")

    normalized_catalog = root / "history_catalog_normalized.json"
    catalog_path = catalog_override or (normalized_catalog if normalized_catalog.is_file()
                                        else root / "history_catalog.json")
    if not catalog_path.is_file():
        catalog_path = CATALOG_DEFAULT
    catalog_count = verify_catalog(root, catalog_path)
    if catalog_path.name == "history_catalog.json" and catalog_path.resolve() != CATALOG_DEFAULT.resolve():
        if digest(root / "history_catalog.json") != digest(catalog_path):
            raise AssertionError("run history_catalog copy differs from verified catalog")

    old_protocol = read_json(PREVIOUS / "protocol.json")
    old_checked = verify_checksums(PREVIOUS, PREVIOUS / "checksums.json")
    if not old_protocol.get("all_old_checksums_verified"):
        raise AssertionError("previous baseline protocol does not certify old checksums")
    if not (root / "mechanism_validation.json").is_file():
        raise AssertionError("mechanism_validation.json missing")
    mechanism = read_json(root / "mechanism_validation.json")
    assert_equal(mechanism.get("passed"), True, "frozen unittest mechanism validation failed")

    summaries = read_json(root / "runs_summary.json")
    aggregate = read_json(root / "aggregate.json")
    assert_equal(len(summaries), 81, "runs_summary must contain 81 runs")
    assert_equal(len(aggregate), 27, "aggregate must contain 27 kind/task rows")
    assert_equal(set(sequences_doc), set(TASKS), "input sequence task set differs")
    for task in TASKS:
        assert_equal(len(sequences_doc[task]), 600, f"input sequence length differs: {task}")

    summary_map, events_map, initial_states = {}, {}, {}
    checkpoint_count = 0
    expected_keys = {(kind, task, seed) for kind in KINDS for task in TASKS for seed in SEEDS}
    for row in summaries:
        key = (row["kind"], row["task"], int(row["seed"]))
        if key in summary_map:
            raise AssertionError(f"duplicate summary row {key}")
        summary_map[key] = row
    assert_equal(set(summary_map), expected_keys, "run key set differs")

    for key in sorted(expected_keys):
        kind, task, seed = key
        summary = summary_map[key]
        run_dir = root / summary["run_directory"]
        assert_equal(run_dir.name, f"{kind}_{task}_seed{seed}", f"run directory mismatch: {key}")
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert_equal(len(events), 600, f"event count differs: {key}")
        events_map[key] = events
        hits, nlls = 0, []
        for i, event in enumerate(events):
            assert_equal(event.get("step"), i + 1, f"step index differs: {key}/{i}")
            target = int(sequences_doc[task][i])
            assert_equal(event.get("target"), target, f"target differs: {key}/{i}")
            probs = event.get("probabilities")
            assert_equal(len(probs), 10, f"probability width differs: {key}/{i}")
            if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in probs):
                raise AssertionError(f"invalid probability {key}/{i}")
            if not close(sum(probs), 1.0, atol=1e-6, rtol=0):
                raise AssertionError(f"probability sum differs {key}/{i}")
            pred = max(range(10), key=lambda j: (probs[j], -j))
            assert_equal(event.get("prediction"), pred, f"prediction differs: {key}/{i}")
            nll = -math.log(max(float(probs[target]), 1e-300))
            if not close(event.get("nll"), nll, atol=2e-12, rtol=2e-12):
                raise AssertionError(f"NLL differs from probabilities: {key}/{i}")
            assert_equal(bool(event.get("hit")), pred == target, f"hit flag differs: {key}/{i}")
            if i == 0:
                assert_equal(event.get("loss"), None, f"first loss should be null: {key}")
                assert_equal(event.get("updates_after"), 0, f"first update count differs: {key}")
            else:
                if not close(event.get("loss"), nll, atol=1e-5, rtol=0):
                    raise AssertionError(f"loss differs from prediction NLL: {key}/{i}")
                assert_equal(event.get("updates_after"), i, f"update count differs: {key}/{i}")
            assert_equal(event.get("steps"), i + 1, f"step count differs: {key}/{i}")
            if kind == "baseline_rtrl20":
                assert_equal(event.get("input_retention_before"), [], f"baseline rho before should be empty: {key}/{i}")
                assert_equal(event.get("input_retention_after"), [], f"baseline rho after should be empty: {key}/{i}")
            else:
                assert_equal(len(event.get("input_retention_before", [])), 30, f"rho width differs: {key}/{i}")
                assert_equal(len(event.get("input_retention_after", [])), 30, f"rho width differs: {key}/{i}")
            if not finite_tree(event):
                raise AssertionError(f"nonfinite event value: {key}/{i}")
            hits += int(pred == target)
            nlls.append(nll)

        expected_summary = {
            "count": 600, "hits": hits, "accuracy": hits / 600,
            "first100_accuracy": sum(e["hit"] for e in events[:100]) / 100,
            "last100_accuracy": sum(e["hit"] for e in events[-100:]) / 100,
            "mean_nll": sum(nlls) / 600, "optimizer_updates": 599,
        }
        for name, expected in expected_summary.items():
            actual = summary.get(name)
            if isinstance(expected, float):
                if not close(actual, expected, atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"summary {name} mismatch: {key}")
            else:
                assert_equal(actual, expected, f"summary {name} mismatch: {key}")

        params = {"baseline_rtrl20": 851, "fixed_input_memory20": 1451,
                  "learned_input_memory20": 1481}[kind]
        core = {"baseline_rtrl20": 641, "fixed_input_memory20": 1241,
                "learned_input_memory20": 1271}[kind]
        assert_equal(summary.get("parameters"), params, f"summary parameter count differs: {key}")
        assert_equal(summary.get("core_size"), core, f"summary core size differs: {key}")

        initial_state = None
        for step in (0, 100, 200, 300, 400, 500, 600):
            path = run_dir / f"checkpoint_{step:03d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not finite_tree(payload):
                raise AssertionError(f"nonfinite checkpoint: {path}")
            checkpoint_count += 1
            config = payload.get("config", {})
            assert_equal(config.get("core_size"), core, f"checkpoint core size differs: {path}")
            assert_equal(config.get("parameters"), params, f"checkpoint parameter count differs: {path}")
            assert_equal(config.get("kind"), kind, f"checkpoint kind differs: {path}")
            assert_equal(payload.get("steps"), step, f"checkpoint step differs: {path}")
            assert_equal(payload.get("updates"), max(step - 1, 0), f"checkpoint updates differ: {path}")
            assert_equal(tuple(payload["eh"].shape), (20, core), f"eh shape differs: {path}")
            assert_equal(tuple(payload["ef"].shape), (20, 20, core), f"ef shape differs: {path}")
            if kind == "baseline_rtrl20":
                if "memory" in payload or "em" in payload:
                    raise AssertionError(f"baseline unexpectedly has memory state: {path}")
            else:
                assert_equal(tuple(payload["memory"].shape), (30,), f"memory shape differs: {path}")
                assert_equal(tuple(payload["em"].shape), (30, core), f"em shape differs: {path}")
                assert_equal(tuple(payload["model"]["memory_weights"].shape), (20, 30), f"V shape differs: {path}")
                if kind == "learned_input_memory20":
                    assert_equal(tuple(payload["model"]["input_retention_logit"].shape), (30,), f"rho parameter shape differs: {path}")
            if step == 0:
                initial_state = payload
                initial_states[key] = payload["model"]
            event = events[step - 1] if step else events[0]
            rf, af = (("r_after", "input_retention_after") if step else
                      ("r_before", "input_retention_before"))
            r = torch.sigmoid(payload["model"]["decay_logit"]).reshape(-1).tolist()
            if any(not close(x, y, atol=1e-7, rtol=0) for x, y in zip(r, event[rf])):
                raise AssertionError(f"trace retention checkpoint/event mismatch: {path}")
            if kind == "baseline_rtrl20":
                if event[af] != []:
                    raise AssertionError(f"baseline input retention should be empty: {path}")
            else:
                rho = ([0.8] * 30 if kind == "fixed_input_memory20" else
                       torch.sigmoid(payload["model"]["input_retention_logit"]).reshape(-1).tolist())
                if any(not close(x, y, atol=1e-7, rtol=0) for x, y in zip(rho, event[af])):
                    raise AssertionError(f"input retention checkpoint/event mismatch: {path}")
        base = initial_states[key]
        for shared in ("input_weights", "recurrent_weights", "bias", "eta_raw", "decay_logit", "W_out", "b_out"):
            if shared not in base:
                raise AssertionError(f"missing common initial parameter {shared}: {key}")

    # Identical old 851-parameter initial state across all kinds for each seed.
    for task in TASKS:
        for seed in SEEDS:
            base = initial_states[("baseline_rtrl20", task, seed)]
            for kind in KINDS[1:]:
                other = initial_states[(kind, task, seed)]
                for name, value in base.items():
                    if name not in other or not torch.equal(value, other[name]):
                        raise AssertionError(f"common initial parameter mismatch: {task}/{seed}/{kind}/{name}")
                if not torch.equal(other["memory_weights"], torch.zeros_like(other["memory_weights"])):
                    raise AssertionError(f"initial V is not zero: {task}/{seed}/{kind}")
                if kind == "learned_input_memory20":
                    rho = torch.sigmoid(other["input_retention_logit"])
                    if not torch.equal(rho, torch.full_like(rho, 0.8)):
                        raise AssertionError(f"initial learned rho is not exactly .8: {task}/{seed}")

    learned_grad_runs = fixed_drift_runs = learned_change_runs = 0
    for key, events in events_map.items():
        kind, task, seed = key
        if kind == "fixed_input_memory20":
            fixed_values = events[0]["input_retention_before"]
            if any(not close(v, 0.8, atol=1e-7, rtol=0) for v in fixed_values):
                raise AssertionError(f"fixed input retention initialization differs from .8: {key}")
            if any(v != fixed_values[j] for event in events
                   for field in ("input_retention_before", "input_retention_after")
                   for j, v in enumerate(event[field])):
                raise AssertionError(f"fixed input retention changed: {key}")
            fixed_drift_runs += 1
        elif kind == "learned_input_memory20":
            if not any(float(e.get("input_retention_grad_norm") or 0) > 0 for e in events[1:]):
                raise AssertionError(f"learned rho gradient is always zero: {key}")
            learned_grad_runs += 1
            if not any(a != b for e in events for a, b in zip(e["input_retention_before"], e["input_retention_after"])):
                raise AssertionError(f"learned rho never changes: {key}")
            learned_change_runs += 1

    for ag in aggregate:
        kind, task = ag["kind"], ag["task"]
        rows = [summary_map[(kind, task, seed)] for seed in SEEDS]
        assert_equal(ag.get("count"), 1800, f"aggregate count differs: {kind}/{task}")
        assert_equal(ag.get("hits"), sum(row["hits"] for row in rows), f"aggregate hits differ: {kind}/{task}")
        for name in ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll"):
            if not close(ag[name], sum(row[name] for row in rows) / 3, atol=1e-12, rtol=1e-12):
                raise AssertionError(f"aggregate {name} differs: {kind}/{task}")
        assert_equal(ag.get("seed_accuracy_min"), min(row["accuracy"] for row in rows), f"aggregate min differs: {kind}/{task}")
        assert_equal(ag.get("seed_accuracy_max"), max(row["accuracy"] for row in rows), f"aggregate max differs: {kind}/{task}")

    # Verify generated input streams from the frozen benchmark. A single frozen
    # child process handles all 27 kind/task seed-2 replays.
    replay_requests = []
    for task in TASKS:
        replay_requests.extend({"kind": kind, "seed": 2, "task": task,
                                "sequence": sequences_doc[task][:20]}
                               for kind in KINDS)
    replay_results = run_frozen(snapshot, "--frozen-replay", replay_requests)
    assert_equal(len(replay_results), 27, "expected 27 batched frozen replay records")
    frozen_events = 0
    for result, request in zip(replay_results, replay_requests):
        kind, task = request["kind"], request["task"]
        key = (kind, task, 2)
        if result["generated_sequences"] != sequences_doc:
            raise AssertionError("input sequences differ from frozen benchmark generator")
        initial = torch.load(root / summary_map[key]["run_directory"] / "checkpoint_000.pt",
                             map_location="cpu", weights_only=False)
        for name, tensor in initial["model"].items():
            if result["initial_parameters"].get(name) != tensor.reshape(-1).tolist():
                raise AssertionError(f"frozen initial parameter mismatch: {key}/{name}")
        for fresh, event in zip(result["rows"], events_map[key][:20]):
            if fresh["probabilities"] != event["probabilities"]:
                raise AssertionError(f"frozen probability mismatch: {key}/{event['step']}")
            for field in ("loss", "grad_norm", "retention_grad_norm", "input_retention_grad_norm", "memory_grad_norm"):
                a, b = fresh[field], event.get(field)
                if a is None or b is None:
                    assert_equal(a, b, f"frozen {field} mismatch: {key}/{event['step']}")
                elif not close(a, b, atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"frozen {field} mismatch: {key}/{event['step']}")
            for field in ("retention_grad", "input_retention_grad", "input_retention_before",
                          "input_retention_after", "r_before", "r_after"):
                if fresh[field] != event[field]:
                    raise AssertionError(f"frozen {field} mismatch: {key}/{event['step']}")
            for field in ("memory_norm", "em_norm", "eh_norm", "ef_norm"):
                if not close(fresh[field], event[field], atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"frozen {field} mismatch: {key}/{event['step']}")
            assert_equal(fresh["steps"], event["steps"], f"frozen step mismatch: {key}")
            assert_equal(fresh["updates_after"], event["updates_after"], f"frozen updates mismatch: {key}")
            if not finite_tree(fresh):
                raise AssertionError(f"nonfinite frozen replay: {key}")
            frozen_events += 1

    # Exact prior streaming baseline parity for the nine binary runs only.
    old_events_count = 0
    for task in ("alternating", "triples", "period16"):
        for seed in SEEDS:
            old_dir = PREVIOUS / f"streaming_rtrl20_{task}_seed{seed}"
            old_events = [json.loads(line) for line in (old_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            new_events = events_map[("baseline_rtrl20", f"binary_{task}", seed)]
            assert_equal(len(old_events), 600, f"previous baseline event count differs: {task}/{seed}")
            for i, (old, new) in enumerate(zip(old_events, new_events)):
                if old["probabilities"] != new["probabilities"]:
                    raise AssertionError(f"binary baseline probability mismatch: {task}/{seed}/{i}")
                if not close(old.get("loss") or 0.0, new.get("loss") or 0.0, atol=0, rtol=0):
                    raise AssertionError(f"binary baseline loss mismatch: {task}/{seed}/{i}")
                assert_equal(old["target"], new["target"], f"old/new binary input differs: {task}/{seed}/{i}")
            old_events_count += len(old_events)

    counters = run_frozen(snapshot, "--counter")
    profile = read_json(root / "compute_profile.json")
    for kind in KINDS:
        for age in (0, 1, 32):
            assert_equal(counters[kind]["profiles"][str(age)]["total_proxy"],
                         profile[kind]["profiles"][str(age)]["total_proxy"],
                         f"operation counter differs {kind}/age{age}")
        assert_equal(counters[kind]["profiles"]["32"]["total_proxy"],
                     profile[kind]["profiles"]["2"]["total_proxy"],
                     f"steady counter age 2/32 differs: {kind}")
        assert_equal(counters[kind]["full_run_proxy"], profile[kind]["full_run_proxy"],
                     f"full-run proxy differs: {kind}")

    # Execute the six unit tests from the hashed source snapshot, independently
    # of the mutable working tree files.
    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "test_input_memory_rtrl", "-q"],
        cwd=snapshot, text=True, capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}, check=False)
    if unit.returncode:
        raise RuntimeError(f"frozen input-memory unittest failed: {unit.stderr[-4000:]}")
    unit_output = unit.stderr + unit.stdout
    if "Ran 6 tests" not in unit_output:
        raise AssertionError(f"expected six frozen unit tests, got output: {unit_output[-1000:]}")

    return {
        "passed": True,
        "counts": {
            "new_checksums_verified": new_checked,
            "frozen_source_files_verified": len(source_hashes),
            "history_catalog_files_verified": catalog_count["files"],
            "history_catalog_normalized_files_manifest_checked": catalog_count.get("normalized_files", catalog_count["files"]),
            "history_catalog_references_verified": catalog_count["references"],
            "history_catalog_statuses": catalog_count["statuses"],
            "previous_checksums_verified": old_checked,
            "runs": len(summaries), "predictions": sum(len(events) for events in events_map.values()),
            "checkpoints_loaded_and_finite": checkpoint_count,
            "aggregates_recomputed": len(aggregate),
            "previous_binary_baseline_events_compared": old_events_count,
            "frozen_source_replay_records": len(replay_results),
            "frozen_source_events_replayed": frozen_events,
            "frozen_unittests_run": 6,
            "fixed_rho_runs_checked": fixed_drift_runs,
            "learned_rho_gradient_runs_checked": learned_grad_runs,
            "learned_rho_change_runs_checked": learned_change_runs,
            "counter_variants": len(KINDS), "counter_ages_per_variant": 3,
        },
        "checks": ["run checksums", "17 frozen source hashes", "history catalog metadata hashes",
                   "81 run event streams, probability hits and NLL", "567 finite checkpoints and config/state shapes",
                   "common 851-parameter initial state, V=0 and rho=.8", "fixed/learned input-retention behavior",
                   "summary and aggregate recomputation", "exact nine-run previous binary baseline parity",
                   "frozen source seed-2 replay across 27 kind/task pairs (540 events)",
                   "six frozen-source unit tests", "frozen-source age 0/1/32/full-run compute counters"],
    }


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--frozen-replay":
        _frozen_replay_main(Path(sys.argv[2])); return
    if len(sys.argv) >= 2 and sys.argv[1] == "--counter":
        _counter_main(Path(sys.argv[2])); return
    if len(sys.argv) not in (2, 4):
        raise SystemExit("usage: audit_input_memory.py RUN_DIRECTORY [--catalog CATALOG_JSON]")
    run_dir = Path(sys.argv[1]).resolve()
    catalog = Path(sys.argv[3]).resolve() if len(sys.argv) == 4 and sys.argv[2] == "--catalog" else None
    if len(sys.argv) == 4 and catalog is None:
        raise SystemExit("usage: audit_input_memory.py RUN_DIRECTORY [--catalog CATALOG_JSON]")
    try:
        result = audit(run_dir, catalog)
    except Exception as exc:
        failure = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (run_dir / "audit.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    (run_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
