"""Compare shared and per-receiver retention; neither applies decay to logits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

import torch

import retention_weight_decay as common
from per_neuron_retention import PerNeuronRetentionRNN, validate_mechanism

SEEDS, PATTERNS = common.SEEDS, common.PATTERNS
MODES = {"shared": 623, "per_neuron": 642}
MODEL_ROOT = common.MODEL_ROOT


def vector(tensor):
    return tensor.detach().reshape(-1).tolist()


def build_model(seed, mode):
    model = (common.PlasticRNN if mode == "shared" else PerNeuronRetentionRNN)(seed=seed)
    assert sum(p.numel() for p in model.parameters()) == MODES[mode]
    assert model.decay_logit.numel() == (1 if mode == "shared" else 20)
    assert model.optimizer.param_groups[1]["weight_decay"] == 0.0
    assert model.optimizer.param_groups[1]["params"][0] is model.decay_logit
    assert model.optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert all(g["lr"] == 0.015 for g in model.optimizer.param_groups)
    return model


def run_pair(root_string, task, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    pattern = PATTERNS[task]
    models = {mode: build_model(seed, mode) for mode in MODES}
    shared, independent = models.values()
    for name, value in shared.state_dict().items():
        other = independent.state_dict()[name]
        if name == "decay_logit":
            assert torch.equal(value.expand_as(other), other)
        else:
            assert torch.equal(value, other), name
    assert shared.predict_next([0, 1, 0, 0, 1]) == independent.predict_next([0, 1, 0, 0, 1])
    histories = []
    events = {mode: [] for mode in MODES}
    paths, handles = {}, {}
    start = time.perf_counter()
    try:
        for mode, model in models.items():
            paths[mode] = root / f"{task}_seed{seed}_{mode}"
            paths[mode].mkdir()
            handles[mode] = (paths[mode] / "events.jsonl").open("w", encoding="utf-8")
            common.checkpoint(model, paths[mode] / "checkpoint_000.pt", 0)
        for index in range(600):
            target = int(pattern[index % len(pattern)])
            for mode, model in models.items():
                probabilities, _, _ = model.predict_next(histories)
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                assert abs(sum(probabilities) - 1) < 1e-6
                prediction = max(range(10), key=lambda i: (probabilities[i], -i))
                row = {"step": index + 1, "target": target, "probabilities": probabilities,
                       "prediction": prediction, "hit": prediction == target,
                       "nll": -math.log(max(probabilities[target], 1e-300)),
                       "retention_before": vector(model.trace_decay),
                       "optimizer_updates_before": model.optimizer_updates}
                model.train_on_history(histories + [target])
                row.update(retention_after=vector(model.trace_decay), raw_logit_after=vector(model.decay_logit),
                           last_update_task_gradient=(vector(model.decay_logit.grad)
                               if model.decay_logit.grad is not None else None),
                           eta_after=float(model.eta.detach()), optimizer_updates_after=model.optimizer_updates)
                assert all(torch.isfinite(p).all() for p in model.parameters())
                handles[mode].write(json.dumps(row, allow_nan=False) + "\n")
                events[mode].append(row)
                if (index + 1) % 100 == 0:
                    handles[mode].flush()
                    common.checkpoint(model, paths[mode] / f"checkpoint_{index+1:03d}.pt", index + 1)
            histories.append(target)
        summaries = []
        for mode, model in models.items():
            rows = events[mode]
            assert len(rows) == 600 and model.optimizer_updates == 1797
            retention = rows[-1]["retention_after"]
            summary = {"task": task, "pattern": pattern, "seed": seed, "mode": mode,
                       "parameter_count": MODES[mode], "retention_weight_decay": 0.0,
                       "count": len(rows), "hits": sum(r["hit"] for r in rows),
                       "accuracy": statistics.mean(r["hit"] for r in rows),
                       "last100_accuracy": statistics.mean(r["hit"] for r in rows[-100:]),
                       "mean_nll": statistics.mean(r["nll"] for r in rows),
                       "initial_retention": rows[0]["retention_before"],
                       "final_retention": retention, "final_retention_mean": statistics.mean(retention),
                       "final_retention_min": min(retention), "final_retention_max": max(retention),
                       "final_retention_std": statistics.pstdev(retention),
                       "final_eta": rows[-1]["eta_after"], "optimizer_updates": model.optimizer_updates,
                       "run_directory": paths[mode].name}
            common.write_json(paths[mode] / "summary.json", summary)
            summaries.append(summary)
        return {"task": task, "seed": seed, "seconds": time.perf_counter() - start, "runs": summaries}
    finally:
        for handle in handles.values():
            handle.close()


def aggregate(summaries):
    results = []
    for task in PATTERNS:
        row = {"task": task, "pattern": PATTERNS[task]}
        for mode in MODES:
            selected = [s for s in summaries if s["task"] == task and s["mode"] == mode]
            assert len(selected) == 3
            retention = [v for s in selected for v in s["final_retention"]]
            row[mode] = {"hits": sum(s["hits"] for s in selected), "count": 1800,
                         **{metric: statistics.mean(s[metric] for s in selected)
                            for metric in ["accuracy", "last100_accuracy", "mean_nll", "final_retention_mean", "final_retention_std"]},
                         "retention_min": min(retention), "retention_max": max(retention)}
        row["accuracy_delta_pp"] = 100 * (row["per_neuron"]["accuracy"] - row["shared"]["accuracy"])
        results.append(row)
    return results


def compare_previous_baseline(root, summaries, previous):
    """Recomputed shared controls must exactly reproduce the previous no-WD runs."""
    previous_summaries = common.json.loads((previous / "runs_summary.json").read_text(encoding="utf-8"))
    checked = 0
    for summary in summaries:
        if summary["mode"] != "shared":
            continue
        previous_summary = next(s for s in previous_summaries if s["task"] == summary["task"]
                                and s["seed"] == summary["seed"] and s["mode"] == "without_decay")
        before = [json.loads(line) for line in (previous / previous_summary["run_directory"] / "events.jsonl").read_text().splitlines()]
        current = [json.loads(line) for line in (root / summary["run_directory"] / "events.jsonl").read_text().splitlines()]
        assert len(before) == len(current) == 600
        for a, b in zip(before, current):
            assert a["probabilities"] == b["probabilities"]
            assert a["retention_after"] == b["retention_after"][0]
            assert a["eta_after"] == b["eta_after"]
            checked += 1
    common.write_json(root / "previous_baseline_reproduction.json", {"passed": True,
                      "predictions_bitwise_reproduced": checked, "previous_run": str(previous)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("C:/pioneer/rnn-research-runs/per_neuron_no_decay_20261028"))
    parser.add_argument("--previous-run", type=Path, default=Path("C:/pioneer/rnn-research-runs/retention_weight_decay_20261028/run_20261028T030338_968933Z"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output_parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    (root / "source_snapshot").mkdir(parents=True)
    for source in [MODEL_ROOT / "model.py", MODEL_ROOT / "requirements.txt", Path(common.__file__),
                   Path(__file__), Path(__file__).with_name("per_neuron_retention.py"),
                   Path(__file__).with_name("test_per_neuron_retention.py")]:
        shutil.copyfile(source, root / "source_snapshot" / source.name)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": SEEDS,
                "patterns": PATTERNS, "length": 600, "modes": MODES,
                "unique_changed_factor": "shared scalar retention logit -> 20 independently trained receiver-row logits",
                "existing_design": "400 independent connection traces retained in both conditions",
                "retention_weight_decay": 0.0, "other_weight_decay": 1e-4,
                "context_length": 32, "updates_per_press": 3, "learning_rate": 0.015,
                "input_sparsity": 0.02, "gradient_clip_norm": 1.0,
                "input_count": 10, "neuron_count": 20, "initial_retention": 0.8,
                "score": "10-class argmax lowest-index tie; predict before training; uniform cold start included",
                "trace_update": "H_new[i,j] = r_i*H_old[i,j] + (1-r_i)*h_new[i]*h_old[j]",
                "receiver_index": "row i, zero-based in tensors; neuron labels 1..20 in reports",
                "state": "replay latest 32 digits from zero; no persistent state between replays",
                "device": "cpu", "dtype": "float32", "torch_threads_per_worker": 1,
                "deterministic_algorithms": True, "workers": args.workers,
                "python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
                "platform": platform.platform(), "previous_run": str(args.previous_run),
                "source_sha256": {p.name: common.digest(p) for p in (root / "source_snapshot").iterdir()}}
    common.write_json(root / "protocol.json", protocol)
    print(f"OUTPUT_ROOT={root}", flush=True)
    mechanism = validate_mechanism()
    common.write_json(root / "mechanism_validation.json", mechanism)
    print(f"MECHANISM={json.dumps(mechanism)}", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_pair, str(root), task, seed) for task in PATTERNS for seed in SEEDS]
        for job in as_completed(jobs):
            pair = job.result()
            summaries.extend(pair["runs"])
            print(json.dumps(pair), flush=True)
    summaries.sort(key=lambda s: (s["task"], s["seed"], s["mode"]))
    assert len(summaries) == 18 and sum(s["count"] for s in summaries) == 10800
    common.write_json(root / "runs_summary.json", summaries)
    common.write_json(root / "aggregate.json", aggregate(summaries))
    compare_previous_baseline(root, summaries, args.previous_run)
    with (root / "runs_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (root / "final_retention.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", "mode", "neuron", "retention"])
        writer.writeheader()
        for summary in summaries:
            for i, value in enumerate(summary["final_retention"]):
                writer.writerow({"task": summary["task"], "seed": summary["seed"], "mode": summary["mode"],
                                 "neuron": "shared" if summary["mode"] == "shared" else i+1, "retention": value})
    common.write_json(root / "checksums.json", {str(p.relative_to(root)): common.digest(p)
                      for p in root.rglob("*") if p.is_file()})
    common.write_json(root / "completed.json", {"success": True, "runs": 18, "predictions": 10800})
    print(f"COMPLETE={root}", flush=True)


if __name__ == "__main__":
    main()
