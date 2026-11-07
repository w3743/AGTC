"""Guess the next 0..9 digit online, using the previous compute-matched models."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch

import representative_benchmark as base
from decimal_tasks import TASKS, SEEDS, make_sequence, validate

KINDS = base.KINDS
PREVIOUS = Path("C:/pioneer/rnn-research-runs/representative_compute_20261028/run_20261028T032905_545175Z")


def segment_scores(rows):
    return {"accuracy": statistics.mean(r["hit"] for r in rows),
            "mean_nll": statistics.mean(r["nll"] for r in rows), "count": len(rows)}


def switching_metrics(rows):
    result = []
    for phase in range(3):
        part = rows[phase*200:(phase+1)*200]
        result.append({"phase": phase+1, "start": phase*200+1, "end": (phase+1)*200,
                       **segment_scores(part), "first30_accuracy": statistics.mean(r["hit"] for r in part[:30]),
                       "last50_accuracy": statistics.mean(r["hit"] for r in part[-50:])})
    return result


def run_one(root_string, kind, width, task, sequence, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    directory = root / f"{kind}_{task}_seed{seed}"
    directory.mkdir()
    model = base.OnlineModel(kind, width, seed)
    history, rows = [], []
    base.checkpoint(model, directory / "checkpoint_000.pt", 0)
    start = time.perf_counter()
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(600):
            # sequence[i] is read only after probabilities have been frozen.
            probabilities = model.predict(history)
            assert len(probabilities) == 10 and abs(sum(probabilities)-1) < 1e-6
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            prediction = max(range(10), key=lambda j: (probabilities[j], -j))
            target = sequence[i]
            row = {"step": i+1, "target": target, "prediction": prediction,
                   "probabilities": probabilities, "hit": prediction == target,
                   "nll": -math.log(max(probabilities[target], 1e-300)), "updates_before": model.updates}
            history.append(target)
            row["training_loss"] = model.train(history)
            row["updates_after"] = model.updates
            assert all(torch.isfinite(p).all() for p in model.network.parameters())
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if (i+1) % 100 == 0:
                handle.flush()
                base.checkpoint(model, directory / f"checkpoint_{i+1:03d}.pt", i+1)
    assert model.updates == 1797
    summary = {"kind": kind, "width": width, "task": task, "seed": seed, "count": 600,
               "hits": sum(r["hit"] for r in rows), **segment_scores(rows),
               "first100_accuracy": statistics.mean(r["hit"] for r in rows[:100]),
               "last100_accuracy": statistics.mean(r["hit"] for r in rows[-100:]),
               "optimizer_updates": model.updates, "parameters": model.config["parameters"],
               "run_directory": directory.name, "phase_metrics": switching_metrics(rows) if task == "switching" else [],
               "concurrent_elapsed_seconds": time.perf_counter()-start}
    base.common.write_json(directory / "summary.json", summary)
    return summary


def aggregate(summaries):
    result = []
    for kind in KINDS:
        for task in TASKS:
            selected = [s for s in summaries if s["kind"] == kind and s["task"] == task]
            assert len(selected) == 3
            result.append({"kind": kind, "task": task, "count": 1800,
                           "hits": sum(s["hits"] for s in selected),
                           **{m: statistics.mean(s[m] for s in selected) for m in
                              ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll")},
                           "seed_accuracy_min": min(s["accuracy"] for s in selected),
                           "seed_accuracy_max": max(s["accuracy"] for s in selected)})
    return result


def aggregate_phases(summaries):
    result = []
    for kind in KINDS:
        selected = [s for s in summaries if s["kind"] == kind and s["task"] == "switching"]
        for phase in range(3):
            values = [s["phase_metrics"][phase] for s in selected]
            result.append({"kind": kind, "phase": phase+1, "start": phase*200+1, "end": (phase+1)*200,
                           "count": 600, **{m: statistics.mean(v[m] for v in values)
                              for m in ("accuracy", "first30_accuracy", "last50_accuracy", "mean_nll")}})
    return result


def reference_scores(sequences):
    """Simple causal references, not part of the neural compute-matched ranking."""
    result = []
    for task, sequence in sequences.items():
        for kind in ("copy_last", "window_majority"):
            hits = []
            for i, target in enumerate(sequence):
                history = sequence[max(0, i-32):i]
                if not history:
                    prediction = 0
                elif kind == "copy_last":
                    prediction = history[-1]
                else:
                    prediction = max(range(10), key=lambda d: (history.count(d), -d))
                hits.append(prediction == target)
            result.append({"task": task, "kind": kind, "count": 600, "hits": sum(hits),
                           "accuracy": statistics.mean(hits), "last100_accuracy": statistics.mean(hits[-100:])})
    return result


def verify_budget_reference(root, previous, profiles, sequences):
    """Reuse fixed model sizes and check counting is unchanged for 0..9 inputs."""
    old_protocol = json.loads((previous / "protocol.json").read_text())
    for name, expected in old_protocol["source_sha256"].items():
        assert base.common.digest(root / "source_snapshot" / name) == expected, f"Source changed: {name}"
    records = []
    for kind, info in profiles.items():
        width = info["config"]["width"]
        for task, sequence in sequences.items():
            model = base.OnlineModel(kind, width, seed=73)
            model.train(sequence[:32])
            prediction = base.counted(lambda: model.predict(sequence[:32]))
            training = base.counted(lambda: model.train(sequence[:32]))
            assert prediction["total_proxy"] == info["prediction"]["total_proxy"]
            assert training["total_proxy"] == info["training_3_updates"]["total_proxy"]
            records.append({"kind": kind, "task": task, "prediction_proxy": prediction["total_proxy"],
                            "training_proxy": training["total_proxy"], "unsupported": {}})
    result = {"passed": True, "reference_run": str(previous), "source_files_matched": len(old_protocol["source_sha256"]),
              "full_run_budget_max_deviation": max(abs(r["ratio_to_custom"]-1) for r in
                   json.loads((previous / "full_run_budgets.json").read_text()).values()),
              "decimal_profiles_verified": len(records), "profiles": records,
              "cpu_latency_note": "Copied previous serial microbenchmark for the same unchanged implementations; not remeasured on these digit tasks."}
    assert result["full_run_budget_max_deviation"] <= 0.05
    base.common.write_json(root / "budget_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("C:/pioneer/rnn-research-runs/decimal_guess_20261028"))
    parser.add_argument("--budget-reference", type=Path, default=PREVIOUS)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output_parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    (root / "source_snapshot").mkdir(parents=True)
    old_protocol = json.loads((args.budget_reference / "protocol.json").read_text())
    for name in old_protocol["source_sha256"]:
        path = base.common.MODEL_ROOT / name if name in ("model.py", "requirements.txt") else Path(__file__).with_name(name)
        shutil.copyfile(path, root / "source_snapshot" / name)
    for name in ("decimal_tasks.py", "decimal_benchmark.py"):
        shutil.copyfile(Path(__file__).with_name(name), root / "source_snapshot" / name)
    for name in ("matched_budgets.json", "full_run_budgets.json", "cpu_latency.json"):
        shutil.copyfile(args.budget_reference / name, root / name)
    sequences = {task: make_sequence(task) for task in TASKS}
    base.common.write_json(root / "task_validation.json", validate())
    base.common.write_json(root / "input_sequences.json", sequences)
    base.common.write_json(root / "baseline_scores.json", reference_scores(sequences))
    profiles = json.loads((root / "matched_budgets.json").read_text())
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "tasks": TASKS, "kinds": KINDS,
                "seeds": SEEDS, "length": 600, "classes": 10, "context": 32,
                "updates_per_press": 3, "learning_rate": 0.015, "weight_decay": 1e-4,
                "custom_retention_weight_decay": 0.0, "custom_input_prox": 0.02, "baseline_input_prox": 0.0,
                "clip_norm": 1.0, "uniform_random_expected_accuracy": 0.1, "uniform_random_expected_nll": math.log(10),
                "data_policy": "Same frozen sequence per task for all architectures and model seeds; sequence not supplied as model metadata",
                "training_order": "predict(history), freeze and score, append target, train; cold-start uniform argmax0 included",
                "switching_reset": False, "switching_phase_lengths": [200,200,200],
                "budget_reference": str(args.budget_reference), "compute_tolerance": 0.05,
                "compute_metric": "auditable scalar-op proxy, including forward/backward/Adam; not hardware exact FLOPs",
                "cpu_latency_remeasured": False, "device": "cpu", "dtype": "float32", "torch_threads": 1,
                "torch": str(torch.__version__), "python": sys.version,
                "source_sha256": {p.name: base.common.digest(p) for p in (root / "source_snapshot").iterdir()}}
    base.common.write_json(root / "protocol.json", protocol)
    print(f"OUTPUT_ROOT={root}", flush=True)
    budget = verify_budget_reference(root, args.budget_reference, profiles, sequences)
    print("BUDGET_VALIDATION=" + json.dumps({k:v for k,v in budget.items() if k != "profiles"}), flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, str(root), kind, profiles[kind]["config"]["width"], task, sequence, seed)
                for kind in KINDS for task, sequence in sequences.items() for seed in SEEDS]
        for job in as_completed(jobs):
            result = job.result()
            summaries.append(result)
            print("RUN=" + json.dumps({k:v for k,v in result.items() if k != "phase_metrics"}), flush=True)
    assert len(summaries) == 126 and sum(s["count"] for s in summaries) == 75600
    summaries.sort(key=lambda s: (s["kind"], s["task"], s["seed"]))
    base.common.write_json(root / "runs_summary.json", summaries)
    base.common.write_json(root / "aggregate.json", aggregate(summaries))
    base.common.write_json(root / "switching_phases.json", aggregate_phases(summaries))
    with (root / "runs_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in summaries[0] if k != "phase_metrics"])
        writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k != "phase_metrics"} for row in summaries)
    base.common.write_json(root / "checksums.json", {str(p.relative_to(root)): base.common.digest(p)
                         for p in root.rglob("*") if p.is_file()})
    base.common.write_json(root / "completed.json", {"success": True, "runs": 126, "predictions": 75600})
    print(f"COMPLETE={root}", flush=True)


if __name__ == "__main__":
    main()
