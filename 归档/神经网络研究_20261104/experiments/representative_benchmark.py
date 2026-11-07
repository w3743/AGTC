"""Online architecture comparison at matched measured scalar-operation budgets."""
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
from torch.nn import functional as F

import retention_weight_decay as common
from per_neuron_retention import PerNeuronRetentionRNN
from representative_recurrent import SequenceRNN, validate as validate_recurrent
from representative_sequence import make_model, validate as validate_sequence
from compute_budget import ArithmeticCounter

KINDS = ["custom", "rnn", "gru", "lstm", "mlp", "tcn", "transformer"]
SEEDS, PATTERNS = common.SEEDS, common.PATTERNS


class OnlineModel:
    def __init__(self, kind, width, seed):
        self.kind, self.width = kind, width
        if kind == "custom":
            self.network = PerNeuronRetentionRNN(seed=seed)
            self.optimizer = self.network.optimizer
        else:
            self.network = (SequenceRNN(kind, width, seed) if kind in ("rnn", "gru", "lstm")
                            else make_model(kind, width, seed))
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.015, weight_decay=1e-4)
        self.updates = 0
        self.config = {"kind": kind, "width": width,
                       "parameters": sum(p.numel() for p in self.network.parameters()),
                       "architecture": getattr(self.network, "config", {"neurons": 20, "connection_traces": 400,
                           "retention_parameters": 20, "input_sparsity": 0.02, "retention_weight_decay": 0.0})}

    def predict(self, history):
        context = history[-32:]
        if self.kind == "custom":
            return self.network.predict_next(context)[0]
        if context:
            with torch.no_grad():
                logits = self.network(context)[-1]
        if len(context) < 2 or self.updates == 0:
            return [0.1] * 10
        return torch.softmax(logits, dim=0).tolist()

    def train(self, history):
        context = history[-32:]
        if self.kind == "custom":
            loss = self.network.train_on_history(context)
            self.updates = self.network.optimizer_updates
            return loss
        if len(context) < 2:
            return None
        target = torch.tensor(context[1:], dtype=torch.long)
        value = None
        for _ in range(3):
            logits = self.network(context[:-1])
            loss = F.cross_entropy(logits, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
            value = float(loss.detach())
        return value


def counted(call):
    counter = ArithmeticCounter()
    with counter:
        call()
    result = counter.stats()
    if result["unsupported"]:
        raise RuntimeError("UNSUPPORTED=" + json.dumps(result["unsupported"]))
    return result


def profile(kind, width, length=32):
    model = OnlineModel(kind, width, seed=73)
    warm = [int("0000100110101111"[i % 16]) for i in range(32)]
    model.train(warm)  # measure the initialized Adam path, not allocation.
    history = warm[:length]
    prediction = counted(lambda: model.predict(history))
    training = counted(lambda: model.train(history))
    return {"config": model.config, "prediction": prediction, "training_3_updates": training,
            "total_proxy": prediction["total_proxy"] + training["total_proxy"]}


def choose_widths(root):
    reference = profile("custom", 20)
    target = reference["total_proxy"]
    profiles = {"custom": reference}
    candidates = {"custom": {20: reference}}
    print(f"TARGET_PROXY={target}", flush=True)
    for kind in KINDS[1:]:
        cache = {}
        def probe(width):
            if width not in cache:
                cache[width] = profile(kind, width)
            return cache[width]["total_proxy"]
        low, high = (2 if kind == "transformer" else 1), 128
        while low < high:
            middle = (low + high) // 2
            if probe(middle) < target:
                low = middle + 1
            else:
                high = middle
        for width in range(max(2 if kind == "transformer" else 1, low-3), low+4):
            probe(width)
        width = min(cache, key=lambda w: abs(cache[w]["total_proxy"] / target - 1))
        chosen = cache[width]
        chosen["ratio_to_custom"] = chosen["total_proxy"] / target
        profiles[kind] = chosen
        candidates[kind] = cache
        print("MATCH=" + json.dumps({"kind": kind, "width": width,
              "parameters": chosen["config"]["parameters"], "ratio": chosen["ratio_to_custom"]}), flush=True)
    profiles["custom"]["ratio_to_custom"] = 1.0
    common.write_json(root / "budget_search.json", candidates)
    common.write_json(root / "matched_budgets.json", profiles)
    # A one-integer-width architecture may not land exactly on the target.
    # Reject a mismatch over 5%; never insert dummy operations to pad the total.
    assert all(abs(p["ratio_to_custom"] - 1) <= 0.05 for p in profiles.values()), "Budget mismatch exceeds 5%"
    return profiles


def full_run_budgets(root, profiles):
    """Account for all shorter startup windows, not only the steady 32-bit case."""
    results = {}
    for kind, selected in profiles.items():
        width = selected["config"]["width"]
        lengths = {32: selected}
        for length in range(33):
            if length != 32:
                lengths[length] = profile(kind, width, length)
        prediction = sum(lengths[min(step-1, 32)]["prediction"]["total_proxy"] for step in range(1, 601))
        training = sum(lengths[min(step, 32)]["training_3_updates"]["total_proxy"] for step in range(1, 601))
        results[kind] = {"prediction_proxy": prediction, "training_proxy": training,
                         "total_proxy": prediction + training,
                         "per_length": {length: {"prediction": info["prediction"]["total_proxy"],
                                                  "training": info["training_3_updates"]["total_proxy"]}
                                        for length, info in lengths.items()}}
    target = results["custom"]["total_proxy"]
    for row in results.values():
        row["ratio_to_custom"] = row["total_proxy"] / target
    common.write_json(root / "full_run_budgets.json", results)
    assert all(abs(row["ratio_to_custom"] - 1) <= 0.05 for row in results.values()), "Full-run mismatch exceeds 5%"
    return results


def microbenchmark(root, profiles):
    rows = {}
    history = [int("0000100110101111"[i % 16]) for i in range(32)]
    for kind, info in profiles.items():
        model = OnlineModel(kind, info["config"]["width"], seed=73)
        for _ in range(5):
            model.predict(history)
            model.train(history)
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            model.predict(history)
            model.train(history)
            timings.append(time.perf_counter()-start)
        rows[kind] = {"serial_single_thread_mean_ms": 1000*statistics.mean(timings),
                      "serial_single_thread_median_ms": 1000*statistics.median(timings),
                      "repeats": 20, "warmup": 5}
    common.write_json(root / "cpu_latency.json", rows)


def checkpoint(model, path, step):
    torch.save({"step": step, "updates": model.updates, "config": model.config,
                "model": model.network.state_dict(), "optimizer": model.optimizer.state_dict()}, path)


def run_one(root_string, kind, width, task, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    path = root / f"{kind}_{task}_seed{seed}"
    path.mkdir()
    model = OnlineModel(kind, width, seed)
    history, events = [], []
    pattern = PATTERNS[task]
    start = time.perf_counter()
    checkpoint(model, path / "checkpoint_000.pt", 0)
    with (path / "events.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(600):
            probabilities = model.predict(history)
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            assert abs(sum(probabilities)-1) < 1e-6
            prediction = max(range(10), key=lambda j: (probabilities[j], -j))
            target = int(pattern[i % len(pattern)])
            row = {"step": i+1, "target": target, "prediction": prediction,
                   "probabilities": probabilities, "hit": prediction == target,
                   "nll": -math.log(max(probabilities[target], 1e-300)), "updates_before": model.updates}
            history.append(target)
            row["training_loss"] = model.train(history)
            row["updates_after"] = model.updates
            assert all(torch.isfinite(p).all() for p in model.network.parameters())
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            events.append(row)
            if (i+1) % 100 == 0:
                handle.flush()
                checkpoint(model, path / f"checkpoint_{i+1:03d}.pt", i+1)
    assert model.updates == 1797
    result = {"kind": kind, "width": width, "task": task, "seed": seed, "count": 600,
              "hits": sum(r["hit"] for r in events), "accuracy": statistics.mean(r["hit"] for r in events),
              "last100_accuracy": statistics.mean(r["hit"] for r in events[-100:]),
              "mean_nll": statistics.mean(r["nll"] for r in events), "optimizer_updates": model.updates,
              "parameters": model.config["parameters"], "run_directory": path.name,
              "concurrent_elapsed_seconds": time.perf_counter()-start}
    common.write_json(path / "summary.json", result)
    return result


def aggregate(summaries):
    rows = []
    for kind in KINDS:
        for task in PATTERNS:
            selected = [r for r in summaries if r["kind"] == kind and r["task"] == task]
            assert len(selected) == 3
            rows.append({"kind": kind, "task": task, "count": 1800,
                         "hits": sum(r["hits"] for r in selected),
                         **{metric: statistics.mean(r[metric] for r in selected)
                            for metric in ("accuracy", "last100_accuracy", "mean_nll")},
                         "seed_accuracy_min": min(r["accuracy"] for r in selected),
                         "seed_accuracy_max": max(r["accuracy"] for r in selected)})
    return rows


def check_custom_reproduction(root, summaries, previous):
    previous_summaries = json.loads((previous / "runs_summary.json").read_text(encoding="utf-8"))
    count = 0
    for summary in summaries:
        if summary["kind"] != "custom":
            continue
        before = next(s for s in previous_summaries if s["mode"] == "per_neuron"
                      and s["task"] == summary["task"] and s["seed"] == summary["seed"])
        old_rows = [json.loads(s) for s in (previous / before["run_directory"] / "events.jsonl").read_text().splitlines()]
        new_rows = [json.loads(s) for s in (root / summary["run_directory"] / "events.jsonl").read_text().splitlines()]
        assert len(old_rows) == len(new_rows) == 600
        for a, b in zip(old_rows, new_rows):
            assert a["probabilities"] == b["probabilities"]
            count += 1
    common.write_json(root / "custom_reproduction.json", {"passed": True, "bitwise_predictions": count,
                      "previous_run": str(previous)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("C:/pioneer/rnn-research-runs/representative_compute_20261028"))
    parser.add_argument("--previous-run", type=Path, default=Path("C:/pioneer/rnn-research-runs/per_neuron_no_decay_20261028/run_20261028T031322_535786Z"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output_parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    (root / "source_snapshot").mkdir(parents=True)
    source_names = ["representative_benchmark.py", "representative_recurrent.py", "representative_sequence.py",
                    "compute_budget.py", "per_neuron_retention.py", "retention_weight_decay.py"]
    for source in [common.MODEL_ROOT / "model.py", common.MODEL_ROOT / "requirements.txt",
                   *[Path(__file__).with_name(n) for n in source_names]]:
        shutil.copyfile(source, root / "source_snapshot" / source.name)
    print(f"OUTPUT_ROOT={root}", flush=True)
    common.write_json(root / "mechanism_validation.json", {"recurrent": validate_recurrent(), "sequence": validate_sequence()})
    profiles = choose_widths(root)
    budgets = full_run_budgets(root, profiles)
    microbenchmark(root, profiles)
    protocol = {"kinds": KINDS, "seeds": SEEDS, "patterns": PATTERNS, "length": 600,
                "compute_matching": "measured arithmetic scalar-op proxy, including prediction + three forward/backward Adam updates",
                "tolerance": 0.05, "nonlinear_unit_cost": 1,
                "excluded_from_proxy": "memory/index/shape/copy; CPU time separately measured",
                "context": 32, "updates_per_press": 3, "learning_rate": 0.015,
                "weight_decay": 1e-4, "custom_retention_weight_decay": 0.0,
                "custom_input_prox": 0.02, "baseline_input_prox": 0.0,
                "clip_norm": 1.0, "classes": 10, "cold_start": "uniform probabilities; smallest-index argmax, included",
                "seed_policy": "same seed set, architecture-native initializers; no seed or hyperparameter selection by task accuracy",
                "learning_rate_search": False, "device": "cpu", "dtype": "float32", "torch_threads": 1,
                "torch": str(torch.__version__), "python": sys.version, "previous_run": str(args.previous_run),
                "source_sha256": {p.name: common.digest(p) for p in (root / "source_snapshot").iterdir()}}
    common.write_json(root / "protocol.json", protocol)
    if args.profile_only:
        return
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, str(root), kind, profiles[kind]["config"]["width"], task, seed)
                for kind in KINDS for task in PATTERNS for seed in SEEDS]
        for job in as_completed(jobs):
            summary = job.result()
            summaries.append(summary)
            print("RUN=" + json.dumps(summary), flush=True)
    assert len(summaries) == 63 and sum(s["count"] for s in summaries) == 37800
    summaries.sort(key=lambda s: (s["kind"], s["task"], s["seed"]))
    common.write_json(root / "runs_summary.json", summaries)
    common.write_json(root / "aggregate.json", aggregate(summaries))
    check_custom_reproduction(root, summaries, args.previous_run)
    with (root / "runs_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    common.write_json(root / "checksums.json", {str(p.relative_to(root)): common.digest(p) for p in root.rglob("*") if p.is_file()})
    common.write_json(root / "completed.json", {"success": True, "runs": 63, "predictions": 37800})
    print(f"COMPLETE={root}", flush=True)


if __name__ == "__main__":
    main()
