"""Paired CPU ablation: only decay_logit's Adam weight_decay changes.

Run with the project's Python environment; raw outputs stay outside Git.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))
from model import PlasticRNN

SEEDS = [2, 20261027, 20261028]
PATTERNS = {"alternating": "01", "triples": "000111", "period16": "0000100110101111"}
MODES = {"with_decay": 1e-4, "without_decay": 0.0}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(seed, mode):
    model = PlasticRNN(seed=seed)
    model.optimizer.param_groups[1]["weight_decay"] = MODES[mode]
    assert model.optimizer.param_groups[1]["params"][0] is model.decay_logit
    assert sum(p.numel() for p in model.parameters()) == 623
    for i, group in enumerate(model.optimizer.param_groups):
        assert group["lr"] == 0.015
        assert group["weight_decay"] == (1e-4 if i == 0 else MODES[mode])
    return model


def checkpoint(model, path, step):
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "optimizer_updates": model.optimizer_updates}, path)


def preflight(root):
    """Check split-group control against the original single-group Adam model."""
    repo = MODEL_ROOT.parents[1]
    # The original directory is being moved; HEAD will no longer contain it
    # after the move is committed. Pin the actual historical control source.
    old_source = subprocess.check_output(
        ["git", "show", "1fd8209e4d136d62ab805040c5098f9bd6945f52:算法/按键预测20神经元/model.py"], cwd=repo)
    old_path = root / "source_snapshot" / "legacy_model.py"
    old_path.write_bytes(old_source)
    spec = importlib.util.spec_from_file_location("legacy_retention_model", old_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    comparisons = 0
    for pattern in PATTERNS.values():
        legacy = module.PlasticRNN(seed=2)
        control = build_model(2, "with_decay")
        history = []
        for step in range(40):
            assert legacy.predict_next(history) == control.predict_next(history)
            history.append(int(pattern[step % len(pattern)]))
            assert legacy.train_on_history(history) == control.train_on_history(history)
            for name, value in legacy.state_dict().items():
                assert torch.equal(value, control.state_dict()[name]), name
            comparisons += 1
    # Isolate the regularization effect with fresh optimizers and no task gradient.
    drift = {}
    for mode in MODES:
        model = build_model(2, mode)
        initial = float(model.trace_decay.detach())
        for _ in range(10):
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            model.optimizer.step()
        drift[mode] = {"initial": initial, "after_10_zero_gradient_updates": float(model.trace_decay.detach())}
    assert drift["without_decay"]["initial"] == drift["without_decay"]["after_10_zero_gradient_updates"]
    assert drift["with_decay"]["after_10_zero_gradient_updates"] < drift["with_decay"]["initial"]
    result = {"passed": True, "legacy_control_bitwise_equal_steps": comparisons,
              "legacy_model_sha256": digest(old_path), "zero_task_gradient": drift}
    write_json(root / "preflight.json", result)
    return result


def run_pair(root_string, task, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    pattern = PATTERNS[task]
    sequence = [int(pattern[i % len(pattern)]) for i in range(600)]
    models = {mode: build_model(seed, mode) for mode in MODES}
    for name, value in models["with_decay"].state_dict().items():
        assert torch.equal(value, models["without_decay"].state_dict()[name]), name
    history = []
    events = {mode: [] for mode in MODES}
    paths = {}
    handles = {}
    started = time.perf_counter()
    try:
        for mode, model in models.items():
            paths[mode] = root / f"{task}_seed{seed}_{mode}"
            paths[mode].mkdir()
            handles[mode] = (paths[mode] / "events.jsonl").open("w", encoding="utf-8")
            checkpoint(model, paths[mode] / "checkpoint_000.pt", 0)
        for index, target in enumerate(sequence):
            for mode, model in models.items():
                probabilities, activity, edges = model.predict_next(history)
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                assert abs(sum(probabilities) - 1) < 1e-6
                prediction = max(range(10), key=lambda i: (probabilities[i], -i))
                row = {"step": index + 1, "target": target, "prediction": prediction,
                       "probabilities": probabilities, "hit": prediction == target,
                       "nll": -math.log(max(probabilities[target], 1e-300)),
                       "retention_before": float(model.trace_decay.detach()),
                       "optimizer_updates_before": model.optimizer_updates}
                model.train_on_history(history + [target])
                row.update(retention_after=float(model.trace_decay.detach()),
                           raw_logit_after=float(model.decay_logit.detach()),
                           eta_after=float(model.eta.detach()),
                           optimizer_updates_after=model.optimizer_updates,
                           last_update_task_gradient=(float(model.decay_logit.grad)
                               if model.decay_logit.grad is not None else None))
                assert all(torch.isfinite(p).all() for p in model.parameters())
                handles[mode].write(json.dumps(row, allow_nan=False) + "\n")
                events[mode].append(row)
                if (index + 1) % 100 == 0:
                    handles[mode].flush()
                    checkpoint(model, paths[mode] / f"checkpoint_{index+1:03d}.pt", index + 1)
            history.append(target)
        summaries = []
        for mode, model in models.items():
            rows = events[mode]
            assert model.optimizer_updates == 1797
            summary = {"task": task, "pattern": pattern, "seed": seed, "mode": mode,
                       "retention_weight_decay": MODES[mode], "count": len(rows),
                       "hits": sum(r["hit"] for r in rows),
                       "accuracy": statistics.mean(r["hit"] for r in rows),
                       "last100_accuracy": statistics.mean(r["hit"] for r in rows[-100:]),
                       "mean_nll": statistics.mean(r["nll"] for r in rows),
                       "initial_retention": rows[0]["retention_before"],
                       "final_retention": rows[-1]["retention_after"],
                       "final_eta": rows[-1]["eta_after"],
                       "optimizer_updates": model.optimizer_updates,
                       "run_directory": paths[mode].name}
            write_json(paths[mode] / "summary.json", summary)
            summaries.append(summary)
        return {"task": task, "seed": seed, "seconds": time.perf_counter() - started, "runs": summaries}
    finally:
        for handle in handles.values():
            handle.close()


def aggregate(summaries):
    rows = []
    for task in PATTERNS:
        row = {"task": task, "pattern": PATTERNS[task]}
        for mode in MODES:
            selected = [s for s in summaries if s["task"] == task and s["mode"] == mode]
            assert len(selected) == 3
            row[mode] = {"hits": sum(s["hits"] for s in selected), "count": 1800,
                         **{metric: statistics.mean(s[metric] for s in selected)
                            for metric in ["accuracy", "last100_accuracy", "mean_nll", "final_retention"]},
                         "retention_min": min(s["final_retention"] for s in selected),
                         "retention_max": max(s["final_retention"] for s in selected)}
        row["accuracy_delta_pp"] = 100 * (row["without_decay"]["accuracy"] - row["with_decay"]["accuracy"])
        row["retention_delta"] = row["without_decay"]["final_retention"] - row["with_decay"]["final_retention"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("C:/pioneer/rnn-research-runs/retention_weight_decay_20261028"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output_parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    (root / "source_snapshot").mkdir(parents=True)
    for source in [MODEL_ROOT / "model.py", MODEL_ROOT / "requirements.txt", Path(__file__).resolve()]:
        shutil.copyfile(source, root / "source_snapshot" / source.name)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seeds": SEEDS,
                "patterns": PATTERNS, "length": 600, "modes": MODES,
                "unique_changed_factor": "Adam weight_decay for shared decay_logit only",
                "context_length": 32, "updates_per_press": 3, "learning_rate": 0.015,
                "other_weight_decay": 1e-4, "input_sparsity": 0.02, "gradient_clip_norm": 1.0,
                "input_count": 10, "neuron_count": 20, "parameter_count": 623,
                "score": "10-class argmax, lowest-index tie; predict before observing target; includes cold start",
                "state": "replay latest 32 digits from zero for prediction and training",
                "device": "cpu", "dtype": "float32", "torch_threads_per_worker": 1,
                "deterministic_algorithms": True, "workers": args.workers,
                "python": sys.version, "python_executable": sys.executable,
                "torch": str(torch.__version__), "platform": platform.platform(),
                "source_sha256": {p.name: digest(p) for p in (root / "source_snapshot").iterdir()}}
    write_json(root / "protocol.json", protocol)
    print(f"OUTPUT_ROOT={root}", flush=True)
    result = preflight(root)
    print(f"PREFLIGHT={json.dumps(result)}", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_pair, str(root), task, seed) for task in PATTERNS for seed in SEEDS]
        for job in as_completed(jobs):
            pair = job.result()
            summaries.extend(pair["runs"])
            print(json.dumps(pair, ensure_ascii=False), flush=True)
    summaries.sort(key=lambda s: (s["task"], s["seed"], s["mode"]))
    write_json(root / "runs_summary.json", summaries)
    write_json(root / "aggregate.json", aggregate(summaries))
    with (root / "runs_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(root / "checksums.json", {str(p.relative_to(root)): digest(p)
               for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"})
    write_json(root / "completed.json", {"runs": len(summaries), "predictions": sum(s["count"] for s in summaries), "success": True})
    print("COMPLETE=" + str(root), flush=True)


if __name__ == "__main__":
    main()
