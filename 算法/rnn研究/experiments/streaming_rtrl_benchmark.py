"""Benchmark bounded streaming RTRL against a replay-last-32 control."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
import representative_benchmark as base
from decimal_benchmark import segment_scores
from decimal_tasks import SEEDS
from linear_readout import LinearReadoutRNN
from streaming_rtrl import StreamingRTRL

TASKS = ["alternating", "period16", "triples"]
KINDS = ["custom", "custom_linear20", "rnn", "gru", "lstm", "mlp", "tcn", "transformer",
         "streaming_rtrl20", "streaming_one_step20", "replay_lastpair20"]
NEW_KINDS = KINDS[-3:]
PREVIOUS = Path("C:/pioneer/rnn-research-runs/binary_readout_20261028/run_20261028T072957_438720Z")
OUTPUT_PARENT = Path("C:/pioneer/rnn-research-runs/streaming_rtrl_20261028")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def streams():
    patterns = base.common.PATTERNS
    return {task: [int(pattern[i % len(pattern)]) for i in range(600)] for task, pattern in patterns.items()}


def checkpoint(model, path: Path, step: int) -> None:
    if isinstance(model, StreamingRTRL):
        payload = model.state_payload()
        payload["step"] = step
    elif isinstance(model, ReplayLastPair):
        payload = {"model": model.network.state_dict(), "optimizer": model.network.optimizer.state_dict(),
                   "history": model.history.copy(), "steps": model.steps, "updates": model.updates, "step": step}
    else:
        raise TypeError(type(model))
    payload['config'] = {'width':20, 'parameters':851,
                         'credit':getattr(model,'credit','lastpair_bptt'),
                         'persistent_state':isinstance(model,StreamingRTRL)}
    torch.save(payload, path)


class ReplayLastPair:
    """Bounded last-32 replay with one update per newly observed target."""
    def __init__(self, seed: int):
        self.network = LinearReadoutRNN(seed=seed, width=20)
        self.optimizer = self.network.optimizer
        self.history: list[int] = []
        self.steps = self.updates = 0

    def predict(self):
        if not self.history:
            return torch.full((10,), .1)
        with torch.no_grad():
            logits, _ = self.network.replay(self.history[-32:])
            return torch.softmax(logits, dim=0)

    def learn_target(self, target: int):
        if not self.history:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        logits, _ = self.network.replay(self.history[-32:])
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
        loss.backward()
        retention_grad = (self.network.decay_logit.grad.detach().clone()
                          if self.network.decay_logit.grad is not None
                          else torch.zeros_like(self.network.decay_logit))
        preclip = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in self.network.parameters()
                                                        if p.grad is not None]))
        retention_norm = retention_grad.norm()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            self.network.input_weights.copy_(self.network.input_weights.sign() *
                (self.network.input_weights.abs() - lr * .02).clamp_min(0))
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        self.network.optimizer_updates = self.updates
        return {"loss": float(loss.detach()), "grad_norm": float(preclip),
                "retention_grad_norm": float(retention_norm), "retention_grad": retention_grad.reshape(-1).tolist()}

    def advance(self, digit: int):
        self.history.append(digit)
        if len(self.history) > 32:
            del self.history[:-32]
        self.steps += 1


def _factory(kind: str, seed: int):
    return ReplayLastPair(seed) if kind == "replay_lastpair20" else StreamingRTRL(
        seed=seed, learn=True, credit="rtrl" if kind == "streaming_rtrl20" else "one_step")


def build_model(kind: str, seed: int):
    """Factory entry point shared with the profiling helper."""
    if kind in NEW_KINDS:
        return _factory(kind, seed)
    raise ValueError(f"unsupported new model kind: {kind}")


def run_one(root_string: str, kind: str, task: str, sequence: list[int], seed: int):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root_string)
    directory = root / f"{kind}_{task}_seed{seed}"
    directory.mkdir()
    model = _factory(kind, seed)
    rows = []
    checkpoint(model, directory / "checkpoint_000.pt", 0)
    start = time.perf_counter()
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(len(sequence)):
            # Freeze the prediction before reading/training on this target.
            probabilities = model.predict()
            probs = probabilities.detach().cpu().tolist()
            assert len(probs) == 10 and all(math.isfinite(x) and 0 <= x <= 1 for x in probs)
            assert abs(sum(probs) - 1) < 1e-6
            prediction = max(range(10), key=lambda j: (probs[j], -j))
            with torch.no_grad():
                r_before = model.network.trace_decay.reshape(-1).detach().cpu().tolist()
            before = model.updates
            target = sequence[i]
            stats = model.learn_target(target) if i else None
            if stats is None:
                loss = grad_norm = ret_norm = None
                ret_grad = [0.0] * 20
            else:
                loss, grad_norm = stats["loss"], stats["grad_norm"]
                ret_norm, ret_grad = stats["retention_grad_norm"], stats["retention_grad"]
            model.advance(target)
            n = model.network
            with torch.no_grad():
                r_after = n.trace_decay.reshape(-1).detach().cpu().tolist()
                eta = float(n.eta)
            if isinstance(model, StreamingRTRL):
                eh_norm, ef_norm = float(model.eh.norm()), float(model.ef.norm())
                state_elements, elig_elements = 420, 269220
                finite_tensors = (model.h, model.fast_edges, model.eh, model.ef, *n.parameters())
                assert not hasattr(model, "history")
            else:
                eh_norm = ef_norm = None
                state_elements, elig_elements = 420, 0
                finite_tensors = tuple(n.parameters())
            assert all(torch.isfinite(t).all() for t in finite_tensors), f"nonfinite state {kind}/{task}/{seed}/{i+1}"
            if stats is not None:
                assert math.isfinite(loss) and math.isfinite(grad_norm) and math.isfinite(ret_norm)
                assert all(math.isfinite(x) for x in ret_grad)
            row = {"step": i+1, "target": target, "prediction": prediction, "probabilities": probs,
                   "hit": prediction == target, "nll": -math.log(max(probs[target], 1e-300)),
                   "loss": loss, "grad_norm": grad_norm, "retention_grad_norm": ret_norm,
                   "retention_grad": ret_grad, "r_before": r_before, "r_after": r_after,
                   "eta": eta, "eh_norm": eh_norm, "ef_norm": ef_norm,
                   "state_elements": state_elements, "elig_elements": elig_elements,
                   "steps": model.steps, "updates_before": before, "updates_after": model.updates}
            if stats is not None:
                assert abs(row["nll"] - loss) < 1e-5, (kind, task, i, row["nll"], loss)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if model.steps % 100 == 0:
                handle.flush()
                checkpoint(model, directory / f"checkpoint_{model.steps:03d}.pt", model.steps)
    expected_steps = len(sequence)
    assert model.steps == expected_steps and model.updates == expected_steps - 1
    summary = {"kind": kind, "width": 20, "task": task, "seed": seed, "count": len(rows),
               "hits": sum(r["hit"] for r in rows), **segment_scores(rows),
               "first100_accuracy": statistics.mean(r["hit"] for r in rows[:100]),
               "last100_accuracy": statistics.mean(r["hit"] for r in rows[-100:]),
               "optimizer_updates": model.updates, "parameters": 851,
               "run_directory": directory.name, "source_run": str(root), "origin": "new",
               "concurrent_elapsed_seconds": time.perf_counter()-start}
    write_json(directory / "summary.json", summary)
    return summary


def merge_old(previous: Path):
    summaries = json.loads((previous / "runs_summary.json").read_text(encoding="utf-8"))
    assert len(summaries) == 72
    checks = json.loads((previous / "checksums.json").read_text(encoding="utf-8"))
    for rel, expected in checks.items():
        path = previous / rel
        assert path.is_file() and digest(path) == expected, f"old checksum mismatch: {rel}"
    for row in summaries:
        row["source_run"], row["origin"] = str(previous), "reused"
    return summaries, checks


def aggregate(summaries):
    rows = []
    for kind in KINDS:
        for task in TASKS:
            selected = [s for s in summaries if s["kind"] == kind and s["task"] == task]
            assert len(selected) == 3, (kind, task, len(selected))
            rows.append({"kind": kind, "task": task, "count": 1800,
                         "hits": sum(s["hits"] for s in selected),
                         **{m: statistics.mean(s[m] for s in selected) for m in
                            ("accuracy", "first100_accuracy", "last100_accuracy", "mean_nll")},
                         "seed_accuracy_min": min(s["accuracy"] for s in selected),
                         "seed_accuracy_max": max(s["accuracy"] for s in selected),
                         "origin": "new" if kind in NEW_KINDS else "reused"})
    return rows


def snapshot(root: Path):
    names = ["model.py", "requirements.txt", "linear_readout.py", "streaming_rtrl.py",
             "streaming_rtrl_benchmark.py", "retention_weight_decay.py", "test_streaming_rtrl.py",
             "representative_benchmark.py", "per_neuron_retention.py", "representative_recurrent.py",
             "representative_sequence.py", "compute_budget.py", "streaming_rtrl_compute.py",
             "decimal_benchmark.py", "decimal_tasks.py"]
    exp = Path(__file__).parent
    model_root = base.common.MODEL_ROOT
    for name in names:
        source = model_root / name if name in ("model.py", "requirements.txt") else exp / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, root / "source_snapshot" / name)
    return {p.name: digest(p) for p in (root / "source_snapshot").iterdir()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--reference", type=Path, default=PREVIOUS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true", help="run each new model on three 5-step sequences")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output_parent / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    (root / "source_snapshot").mkdir(parents=True)
    print(f"OUTPUT_ROOT={root}", flush=True)
    source_hashes = snapshot(root)
    old_protocol = json.loads((args.reference/'protocol.json').read_text(encoding='utf-8'))
    for name in ('model.py','linear_readout.py','compute_budget.py','retention_weight_decay.py'):
        assert source_hashes[name] == old_protocol['source_sha256'][name], name
    if not args.smoke:
        import unittest
        import test_streaming_rtrl
        test_result = unittest.TextTestRunner(verbosity=1).run(
            unittest.defaultTestLoader.loadTestsFromModule(test_streaming_rtrl))
        assert test_result.wasSuccessful()
        write_json(root/'mechanism_validation.json',{'passed':True,'tests_run':test_result.testsRun,
            'checks':['fixed-parameter full-sequence autograd vs RTRL gradients',
                      'three random projections of activity and fast-edge sensitivities in float64',
                      '45-step retained state differs from last32 zero-state replay',
                      'live retention task gradients vs zero in one_step control',
                      '600 observations advance600 times and learn599 times',
                      'fixed eligibility allocation and optimizer groups'],
            'state_elements':420,'eligibility_elements':269220,'eligibility_bytes_float32':1076880})
    if not args.smoke:
        from streaming_rtrl_compute import measure_models
        write_json(root / "compute_profile.json", measure_models(build_model, NEW_KINDS, args.reference))
    sequences = streams()
    write_json(root / "input_sequences.json", sequences)
    old, old_checks = merge_old(args.reference)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "kinds": KINDS,
                "new_kinds": NEW_KINDS, "tasks": TASKS, "seeds": SEEDS,
                "length": 5 if args.smoke else 600, "classes": 10, "context": None,
                "context_by_new_kind":{'streaming_rtrl20':None,'streaming_one_step20':None,'replay_lastpair20':32},
                "updates_per_press": 1, "new_runs": 9 if args.smoke else 27,
                "new_predictions": 45 if args.smoke else 16200,
                "old_runs_reused": 72, "all_old_checksums_verified": True,
                "training_order": "predict before target; learn_target if prior step exists; advance target once",
                "optimizer": "Adam lr=.015 WD=1e-4, retention WD=0, clip=1, input prox=.02",
                "exactness": "RTRL differentiates fixed parameters exactly online; no derivative through optimizer history",
                "credit_control": "one_step zeros prior eligibility while carrying the plastic state",
                "replay_control": "replay history[-32:] for current prediction/training; append only after scoring/training",
                "causal_limit": "one update per press versus three in prior binary benchmark; not a memory-only causal contrast",
                "checkpoint_interval": 100, "workers": args.workers,
                "reference_run": str(args.reference), "reference_checksums": old_checks,
                "source_sha256": source_hashes, "python": sys.version, "torch": str(torch.__version__),
                "git_branch_at_start":subprocess.check_output(['git','branch','--show-current'],text=True).strip()}
    write_json(root / "protocol.json", protocol)
    jobs = []
    if args.smoke:
        for kind in NEW_KINDS:
            for task in TASKS:
                jobs.append((kind, task, sequences[task][:5], SEEDS[0]))
    else:
        jobs = [(kind, task, seq, seed) for kind in NEW_KINDS for task, seq in sequences.items() for seed in SEEDS]
    fresh = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, str(root), kind, task, seq, seed) for kind, task, seq, seed in jobs]
        for future in as_completed(futures):
            item = future.result(); fresh.append(item); print("RUN=" + json.dumps(item), flush=True)
    if args.smoke:
        write_json(root / "smoke_summary.json", fresh)
        write_json(root / "completed.json", {"success": True, "smoke": True, "runs": len(fresh)})
        write_json(root / "checksums.json", {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()})
        print(f"COMPLETE={root}", flush=True)
        return
    summaries = sorted(old + fresh, key=lambda s: (s["kind"], s["task"], s["seed"]))
    assert len(fresh) == 27 and sum(s["count"] for s in fresh) == 16200
    write_json(root / "runs_summary.json", summaries)
    write_json(root / "aggregate.json", aggregate(summaries))
    fields=['kind','task','seed','count','hits','accuracy','first100_accuracy','last100_accuracy',
            'mean_nll','optimizer_updates','origin','source_run','run_directory']
    with (root/'runs_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(summaries)
    write_json(root / "completed.json", {"success": True, "new_runs": 27, "new_predictions": 16200,
              "total_source_runs": 99, "total_predictions": 59400, "old_results_reused": True})
    write_json(root / "checksums.json", {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()})
    print(f"COMPLETE={root}", flush=True)


if __name__ == "__main__":
    main()
