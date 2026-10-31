"""Independent audit for an activity-retention benchmark run directory.

Run with the experiment Python:
    python audit_activity_retention.py RUN_DIRECTORY

The script validates checksums, raw event streams, summaries, checkpoints,
frozen-source replay, the previous baseline, and recomputed compute counters.
It writes audit.json with passed=true only after every check succeeds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

KINDS = ("baseline_rtrl20", "fixed_activity20", "learned_activity20")
TASKS = ("alternating", "triples", "period16")
SEEDS = (2, 20261027, 20261028)
PREVIOUS = Path(r"C:\pioneer\rnn-research-runs\streaming_rtrl_20261028\run_20261028T142841_111140Z")
EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_COUNT = 17


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def assert_equal(a, b, message):
    if a != b:
        raise AssertionError(message)


def verify_checksums(root: Path, manifest_path: Path, *, allowed_missing=()):
    manifest = read_json(manifest_path)
    checked = 0
    missing = []
    for rel, expected in manifest.items():
        path = root / Path(rel)
        if not path.is_file():
            missing.append(rel)
            continue
        actual = digest(path)
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {path}")
        checked += 1
    unexpected = sorted(set(missing) - set(allowed_missing))
    if unexpected:
        raise AssertionError(f"checksum manifest references missing files: {unexpected[:5]}")
    return checked


def frozen_replay(snapshot: Path, request: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--frozen-replay", str(snapshot)],
        input=json.dumps(request), text=True, capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    if proc.returncode:
        raise RuntimeError(f"frozen replay failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def counter_replay(snapshot: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--counter", str(snapshot)],
        text=True, capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    if proc.returncode:
        raise RuntimeError(f"frozen counter replay failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def _frozen_replay_main(snapshot: Path):
    sys.path.insert(0, str(snapshot))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from streaming_rtrl import StreamingRTRL
    from activity_retention import ActivityRetentionRTRL

    request = json.load(sys.stdin)
    kind, seed, sequence = request["kind"], int(request["seed"]), request["sequence"]
    if kind == "baseline_rtrl20":
        model = StreamingRTRL(seed=seed, credit="rtrl")
    else:
        model = ActivityRetentionRTRL(seed=seed,
                                      mode="fixed" if kind == "fixed_activity20" else "learned")
    initial_parameters = {name: value.detach().reshape(-1).tolist()
                          for name, value in model.network.state_dict().items()}
    initial_dynamic = {name: getattr(model, name).detach().reshape(-1).tolist()
                       for name in ("h", "fast_edges", "eh", "ef")}
    rows = []
    for i, target in enumerate(sequence):
        probabilities = model.predict().detach().tolist()
        a_before = (model.activity_retention.detach().reshape(-1).tolist()
                    if hasattr(model, "activity_retention") else [0.0] * 20)
        r_before = model.network.trace_decay.detach().reshape(-1).tolist()
        stats = model.learn_target(int(target)) if i else None
        a_after = (model.activity_retention.detach().reshape(-1).tolist()
                   if hasattr(model, "activity_retention") else [0.0] * 20)
        model.advance(int(target))
        rows.append({
            "probabilities": probabilities,
            "loss": None if stats is None else stats["loss"],
            "grad_norm": None if stats is None else stats["grad_norm"],
            "retention_grad_norm": None if stats is None else stats["retention_grad_norm"],
            "retention_grad": [0.0] * 20 if stats is None else stats["retention_grad"],
            "activity_grad_norm": None if stats is None else stats.get("activity_grad_norm", 0.0),
            "activity_grad": [0.0] * 20 if stats is None else stats.get("activity_grad", [0.0] * 20),
            "a_before": a_before, "a_after": a_after,
            "r_before": r_before,
            "r_after": model.network.trace_decay.detach().reshape(-1).tolist(),
            "eh_norm": float(model.eh.norm()), "ef_norm": float(model.ef.norm()),
            "h": model.h.detach().tolist(), "fast_edges": model.fast_edges.detach().tolist(),
            "steps": model.steps, "updates_after": model.updates,
        })
    print(json.dumps({"rows": rows, "initial_parameters": initial_parameters,
                      "initial_dynamic": initial_dynamic}, allow_nan=False, separators=(",", ":")))


def _counter_main(snapshot: Path):
    sys.path.insert(0, str(snapshot))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from streaming_rtrl import StreamingRTRL
    from activity_retention import ActivityRetentionRTRL
    from streaming_rtrl_compute import counted, process
    from streaming_rtrl_benchmark import streams

    seq = streams()["period16"]
    result = {}
    for kind in KINDS:
        def build():
            if kind == "baseline_rtrl20":
                return StreamingRTRL(seed=73, credit="rtrl")
            return ActivityRetentionRTRL(seed=73,
                                         mode="fixed" if kind == "fixed_activity20" else "learned")
        p = {}
        for age in (0, 1, 32):
            model = build()
            for digit in seq[:age]:
                process(model, digit)
            p[str(age)] = counted(lambda: process(model, seq[age]))
        result[kind] = {"profiles": p,
                        "full_run_proxy": p["0"]["total_proxy"] + p["1"]["total_proxy"]
                        + 598 * p["32"]["total_proxy"]}
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


def audit(root: Path) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = root.resolve()
    protocol = read_json(root / "protocol.json")
    sequences = read_json(root / "input_sequences.json")
    completed = read_json(root / "completed.json")
    assert_equal(completed.get("success"), True, "completed.success is not true")
    assert_equal(completed.get("new_runs"), 27, "completed run count differs")
    assert_equal(completed.get("new_predictions"), 16200, "completed prediction count differs")
    assert_equal(protocol.get("runs"), 27, "protocol run count differs")
    assert_equal(protocol.get("predictions"), 16200, "protocol prediction count differs")
    assert_equal(protocol.get("length"), 600, "protocol length differs")
    assert_equal(set(protocol.get("kinds", [])), set(KINDS), "protocol kinds differ")
    assert_equal(set(protocol.get("tasks", [])), set(TASKS), "protocol tasks differ")
    assert_equal(set(sequences), set(TASKS), "input task set differs")
    for task in TASKS:
        assert_equal(len(sequences[task]), 600, f"input sequence length differs: {task}")

    new_checked = verify_checksums(root, root / "checksums.json", allowed_missing=("checksums.json", "audit.json"))
    checksum_keys = set(read_json(root / "checksums.json"))
    required_raw = [p for p in root.glob("*_seed*/*") if p.is_file()
                    and (p.name in ("events.jsonl", "summary.json") or p.name.startswith("checkpoint_"))]
    omitted_raw = [str(p.relative_to(root)) for p in required_raw
                   if str(p.relative_to(root)) not in checksum_keys]
    if omitted_raw:
        raise AssertionError(f"raw run files missing from checksum manifest: {omitted_raw[:5]}")
    source_hashes = protocol.get("source_sha256", {})
    assert_equal(len(source_hashes), SOURCE_COUNT, "expected 17 frozen source hashes")
    snapshot = root / "source_snapshot"
    assert_equal(set(source_hashes), {p.name for p in snapshot.iterdir() if p.is_file()},
                 "snapshot source file set differs from source hashes")
    for name, expected in source_hashes.items():
        assert_equal(digest(snapshot / name), expected, f"source snapshot hash mismatch: {name}")

    old_protocol = read_json(PREVIOUS / "protocol.json")
    old_checked = verify_checksums(PREVIOUS, PREVIOUS / "checksums.json")
    if not old_protocol.get("all_old_checksums_verified"):
        raise AssertionError("previous baseline protocol does not certify old checksums")

    summaries = read_json(root / "runs_summary.json")
    aggregate = read_json(root / "aggregate.json")
    assert_equal(len(summaries), 27, "runs_summary must contain 27 runs")
    assert_equal(len(aggregate), 9, "aggregate must contain nine kind/task rows")
    summary_map = {}
    events_map = {}
    checkpoint_count = 0
    expected_keys = {(k, t, s) for k in KINDS for t in TASKS for s in SEEDS}
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
        expected_dir = f"{kind}_{task}_seed{seed}"
        assert_equal(run_dir.name, expected_dir, f"run directory mismatch: {key}")
        event_path = run_dir / "events.jsonl"
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert_equal(len(events), 600, f"event count differs: {key}")
        events_map[key] = events
        hits, nlls = 0, []
        for i, event in enumerate(events):
            assert_equal(event.get("step"), i + 1, f"step index differs: {key}/{i}")
            target = int(sequences[task][i])
            assert_equal(event.get("target"), target, f"target differs: {key}/{i}")
            probs = event.get("probabilities")
            assert_equal(len(probs), 10, f"probability width differs: {key}/{i}")
            if any(not math.isfinite(float(x)) or float(x) < 0 or float(x) > 1 for x in probs):
                raise AssertionError(f"invalid probability: {key}/{i}")
            if not close(sum(probs), 1.0, atol=1e-6, rtol=0):
                raise AssertionError(f"probabilities do not sum to one: {key}/{i}")
            pred = max(range(10), key=lambda j: (probs[j], -j))
            assert_equal(event.get("prediction"), pred, f"argmax differs: {key}/{i}")
            nll = -math.log(max(float(probs[target]), 1e-300))
            if not close(event.get("nll"), nll, atol=2e-12, rtol=2e-12):
                raise AssertionError(f"NLL differs from probabilities: {key}/{i}")
            if bool(event.get("hit")) != (pred == target):
                raise AssertionError(f"hit flag differs: {key}/{i}")
            expected_loss = None if i == 0 else nll
            if expected_loss is None:
                assert_equal(event.get("loss"), None, f"first event loss should be null: {key}")
                assert_equal(event.get("updates_after"), 0, f"first event update count: {key}")
            else:
                if not close(event.get("loss"), expected_loss, atol=1e-5, rtol=0):
                    raise AssertionError(f"training loss differs from prediction NLL: {key}/{i}")
                assert_equal(event.get("updates_after"), i, f"update count differs: {key}/{i}")
            assert_equal(event.get("steps"), i + 1, f"step count differs: {key}/{i}")
            hits += int(pred == target)
            nlls.append(nll)
        expected_summary = {
            "count": 600, "hits": hits,
            "accuracy": hits / 600,
            "first100_accuracy": sum(e["hit"] for e in events[:100]) / 100,
            "last100_accuracy": sum(e["hit"] for e in events[-100:]) / 100,
            "mean_nll": sum(nlls) / 600, "optimizer_updates": 599,
        }
        for name, expected in expected_summary.items():
            actual = summary.get(name)
            if isinstance(expected, float):
                if not close(actual, expected, atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"summary {name} mismatch {key}")
            else:
                assert_equal(actual, expected, f"summary {name} mismatch {key}")

        cp_states = []
        for step in (0, 100, 200, 300, 400, 500, 600):
            path = run_dir / f"checkpoint_{step:03d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not finite_tree(payload):
                raise AssertionError(f"nonfinite checkpoint content: {path}")
            checkpoint_count += 1
            cp_states.append((step, payload))
            model = payload["model"]
            config = payload.get("config", {})
            wanted_core = 661 if kind == "learned_activity20" else 641
            wanted_params = 871 if kind == "learned_activity20" else 851
            assert_equal(config.get("core_size"), wanted_core, f"core size mismatch: {path}")
            assert_equal(config.get("parameters"), wanted_params, f"parameter count mismatch: {path}")
            assert_equal(payload.get("steps"), step, f"checkpoint step mismatch: {path}")
            assert_equal(payload.get("updates"), max(step - 1, 0), f"checkpoint updates mismatch: {path}")
            assert_equal(tuple(payload["eh"].shape), (20, wanted_core), f"eh shape mismatch: {path}")
            assert_equal(tuple(payload["ef"].shape), (20, 20, wanted_core), f"ef shape mismatch: {path}")
            event = events[step - 1] if step else events[0]
            rf, af = ("r_after", "a_after") if step else ("r_before", "a_before")
            r = torch.sigmoid(model["decay_logit"]).reshape(-1).tolist()
            if any(not close(x, y, atol=1e-7, rtol=0) for x, y in zip(r, event[rf])):
                raise AssertionError(f"checkpoint r differs from event {step}: {path}")
            if kind == "baseline_rtrl20":
                a = [0.0] * 20
            elif kind == "fixed_activity20":
                a = [0.5] * 20
            else:
                a = torch.sigmoid(model["activity_logit"]).reshape(-1).tolist()
            if any(not close(x, y, atol=1e-7, rtol=0) for x, y in zip(a, event[af])):
                raise AssertionError(f"checkpoint a differs from event {step}: {path}")
        # Load each frozen initial parameter set to verify paired common starts.
        cp_states_by_key = getattr(audit, "_initial_states", None)
        if cp_states_by_key is None:
            audit._initial_states = {}
            cp_states_by_key = audit._initial_states
        cp_states_by_key[key] = cp_states[0][1]["model"]

    # Every variant starts from exactly the same common 851 parameters per task/seed.
    for task in TASKS:
        for seed in SEEDS:
            base = audit._initial_states[("baseline_rtrl20", task, seed)]
            for kind in KINDS[1:]:
                other = audit._initial_states[(kind, task, seed)]
                for name, value in base.items():
                    if name not in other or not torch.equal(value, other[name]):
                        raise AssertionError(f"initial common parameter mismatch: {task}/{seed}/{kind}/{name}")

    # Retention values are checked at the initial checkpoint and each saved step.
    for key, events in events_map.items():
        kind, task, seed = key
        run_dir = root / summary_map[key]["run_directory"]
        initial = torch.load(run_dir / "checkpoint_000.pt", map_location="cpu", weights_only=False)
        if kind == "baseline_rtrl20":
            initial_a = [0.0] * 20
        elif kind == "fixed_activity20":
            initial_a = [0.5] * 20
        else:
            initial_a = torch.sigmoid(initial["model"]["activity_logit"]).reshape(-1).tolist()
        initial_r = torch.sigmoid(initial["model"]["decay_logit"]).reshape(-1).tolist()
        for field, expected in (("a_before", initial_a), ("r_before", initial_r)):
            if any(not close(x, y, atol=1e-7, rtol=0) for x, y in zip(events[0][field], expected)):
                raise AssertionError(f"initial checkpoint {field} mismatch: {key}")
        if kind == "fixed_activity20":
            if any(any(v != 0.5 for v in event["a_before"] + event["a_after"]) for event in events):
                raise AssertionError(f"fixed activity retention changed: {key}")
        if kind == "learned_activity20":
            norms = [float(e.get("activity_grad_norm") or 0) for e in events[1:]]
            if not any(x > 0 for x in norms):
                raise AssertionError(f"learned activity gradients are all zero: {key}")
            if not any(a != b for e in events for a, b in zip(e["a_before"], e["a_after"])):
                raise AssertionError(f"learned activity retention never changes: {key}")

    # Recompute aggregate means from the 27 raw runs.
    for ag in aggregate:
        kind, task = ag["kind"], ag["task"]
        rows = [summary_map[(kind, task, seed)] for seed in SEEDS]
        assert_equal(ag.get("count"), 1800, f"aggregate count differs: {kind}/{task}")
        assert_equal(ag.get("hits"), sum(r["hits"] for r in rows), f"aggregate hits differ: {kind}/{task}")
        for name in ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll"):
            expected = sum(r[name] for r in rows) / 3
            if not close(ag[name], expected, atol=1e-12, rtol=1e-12):
                raise AssertionError(f"aggregate {name} differs: {kind}/{task}")
        assert_equal(ag.get("seed_accuracy_min"), min(r["accuracy"] for r in rows), f"aggregate min differs {kind}/{task}")
        assert_equal(ag.get("seed_accuracy_max"), max(r["accuracy"] for r in rows), f"aggregate max differs {kind}/{task}")

    # Exact baseline trajectories are paired against all nine previous RTRL runs.
    old_events_count = 0
    for task in TASKS:
        for seed in SEEDS:
            old_dir = PREVIOUS / f"streaming_rtrl20_{task}_seed{seed}"
            old_events = [json.loads(line) for line in (old_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            new_events = events_map[("baseline_rtrl20", task, seed)]
            assert_equal(len(old_events), 600, f"previous event count differs: {task}/{seed}")
            for i, (old, new) in enumerate(zip(old_events, new_events)):
                if old["probabilities"] != new["probabilities"]:
                    raise AssertionError(f"baseline probabilities not exact vs previous: {task}/{seed}/{i}")
                if not close(old["loss"] or 0.0, new["loss"] or 0.0, atol=0, rtol=0):
                    raise AssertionError(f"baseline loss not exact vs previous: {task}/{seed}/{i}")
            old_events_count += len(old_events)

    # Replay the first 20 events with frozen source files, not the working tree.
    replay_count = 0
    for key in sorted(expected_keys):
        kind, task, seed = key
        request = {"kind": kind, "seed": seed, "sequence": sequences[task][:20]}
        replay_result = frozen_replay(snapshot, request)
        replayed = replay_result["rows"]
        source_events = events_map[key][:20]
        assert_equal(len(replayed), 20, f"frozen replay length differs: {key}")
        for i, (fresh, event) in enumerate(zip(replayed, source_events)):
            if fresh["probabilities"] != event["probabilities"]:
                raise AssertionError(f"frozen-source replay probability mismatch: {key}/{i}")
            for name in ("loss", "grad_norm", "retention_grad_norm", "activity_grad_norm"):
                actual = fresh[name]
                expected = event.get(name)
                if actual is None or expected is None:
                    assert_equal(actual, expected, f"frozen replay {name} mismatch: {key}/{i}")
                elif not close(actual, expected, atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"frozen replay {name} mismatch: {key}/{i}")
            for name in ("retention_grad", "activity_grad", "a_before", "a_after", "r_before", "r_after"):
                if fresh[name] != event[name]:
                    raise AssertionError(f"frozen replay {name} mismatch: {key}/{i}")
            for name in ("eh_norm", "ef_norm"):
                if not close(fresh[name], event[name], atol=1e-12, rtol=1e-12):
                    raise AssertionError(f"frozen replay {name} mismatch: {key}/{i}")
            if not finite_tree((fresh["h"], fresh["fast_edges"])):
                raise AssertionError(f"frozen replay produced nonfinite state: {key}/{i}")
        # Check the frozen source starts from the checkpoint's common initialization.
        initial = torch.load(root / summary_map[key]["run_directory"] / "checkpoint_000.pt",
                             map_location="cpu", weights_only=False)
        for name, tensor in initial["model"].items():
            actual = replay_result["initial_parameters"].get(name)
            if actual != tensor.reshape(-1).tolist():
                raise AssertionError(f"frozen initial parameter differs from checkpoint_000: {key}/{name}")
        for name, tensor in (("h", initial["h"]), ("fast_edges", initial["fast_edges"]),
                             ("eh", initial["eh"]), ("ef", initial["ef"])):
            if replay_result["initial_dynamic"][name] != tensor.reshape(-1).tolist():
                raise AssertionError(f"frozen initial dynamic state differs from checkpoint_000: {key}/{name}")
        if replayed[0]["probabilities"] != source_events[0]["probabilities"]:
            raise AssertionError(f"initial replay does not match checkpoint-start run: {key}")
        replay_count += 20

    # Recompute operation-counter controls with frozen source and age 0/1/32.
    counters = counter_replay(snapshot)
    profile = read_json(root / "compute_profile.json")
    for kind in KINDS:
        for age in (0, 1, 32):
            got = counters[kind]["profiles"][str(age)]["total_proxy"]
            saved = profile[kind]["profiles"][str(age)]["total_proxy"]
            assert_equal(got, saved, f"operation counter age {age} differs: {kind}")
        if counters[kind]["profiles"]["32"]["total_proxy"] != profile[kind]["profiles"]["2"]["total_proxy"]:
            raise AssertionError(f"steady proxy age 2/32 differs: {kind}")
        assert_equal(counters[kind]["full_run_proxy"], profile[kind]["full_run_proxy"],
                     f"full-run counter differs: {kind}")

    result = {
        "passed": True,
        "counts": {
            "new_checksums_verified": new_checked,
            "frozen_source_files_verified": len(source_hashes),
            "previous_checksums_verified": old_checked,
            "previous_baseline_source_checksums_verified": len(old_protocol.get("source_sha256", {})),
            "runs": len(summaries), "predictions": sum(len(v) for v in events_map.values()),
            "checkpoints_loaded_and_finite": checkpoint_count,
            "aggregates_recomputed": len(aggregate), "previous_baseline_events_compared": old_events_count,
            "frozen_source_events_replayed": replay_count,
            "counter_variants": len(KINDS), "counter_ages_per_variant": 3,
        },
        "checks": ["checksums", "17-source snapshot hashes", "run/event schema and metrics",
                   "checkpoint finite/load/dimensions", "common initialization", "retention checkpoint-event alignment",
                   "fixed and learned retention behavior", "summary and aggregate recomputation",
                   "exact prior baseline probabilities/losses", "frozen-source first-20 replay",
                   "frozen-source age 0/1/32 operation counters and full-run proxy"],
    }
    return result


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--frozen-replay":
        _frozen_replay_main(Path(sys.argv[2]))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--counter":
        _counter_main(Path(sys.argv[2]))
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_activity_retention.py RUN_DIRECTORY")
    root = Path(sys.argv[1]).resolve()
    try:
        result = audit(root)
    except Exception as exc:
        failure = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (root / "audit.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    (root / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
