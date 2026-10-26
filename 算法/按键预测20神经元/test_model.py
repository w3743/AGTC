"""Contract and synthetic-sequence checks for the 20-neuron key predictor.

Run with KEYPRESS20_QA_OUTPUT pointing at an evidence directory to save the
raw prequential predictions and summary outside the repository.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import unittest
from datetime import datetime

import torch
import torch.nn.functional as F

from model import INPUT_COUNT, NEURON_COUNT, PlasticRNN


SEED = 20261026
OUTPUT_DIR = os.environ.get("KEYPRESS20_QA_OUTPUT")


def _argmax(probabilities: list[float]) -> int:
    return max(range(INPUT_COUNT), key=lambda digit: (probabilities[digit], -digit))


def _validate_prediction(
    test: unittest.TestCase,
    probabilities: list[float],
    activity: list[float],
    edges: list[list[float]],
) -> None:
    test.assertEqual(len(probabilities), INPUT_COUNT)
    test.assertEqual(len(activity), NEURON_COUNT)
    test.assertEqual(len(edges), NEURON_COUNT)
    test.assertTrue(all(len(row) == NEURON_COUNT for row in edges))
    test.assertTrue(all(math.isfinite(value) for value in probabilities))
    test.assertTrue(all(math.isfinite(value) for value in activity))
    test.assertTrue(all(math.isfinite(value) for row in edges for value in row))
    test.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities))
    test.assertAlmostEqual(sum(probabilities), 1.0, places=6)


def _run_prequential(
    name: str,
    sequence: list[int],
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Predict from observed prefix, freeze probabilities, then train on label."""

    model = PlasticRNN(seed=seed)
    observed: list[int] = []
    records: list[dict[str, object]] = []
    per_key_seconds: list[float] = []

    for index, actual in enumerate(sequence):
        started = time.perf_counter()
        probabilities, activity, edges = model.predict_next(observed)
        _validate_prediction(unittest.TestCase(), probabilities, activity, edges)
        prediction = _argmax(probabilities)
        updates_before_feedback = model.optimizer_updates

        # Preserve this exact vector before the target is visible to training.
        frozen_probabilities = [float(value) for value in probabilities]
        record: dict[str, object] = {
            "session_id": name,
            "prediction_id": f"{name}-{index:04d}",
            "event_index": index,
            "history_before": list(observed),
            "history_length": len(observed),
            "probabilities": frozen_probabilities,
            "prediction": prediction,
            "actual_digit": actual,
            "hit": prediction == actual,
            "optimizer_updates_at_prediction": updates_before_feedback,
            "prediction_source": (
                "uniform_cold_start"
                if updates_before_feedback == 0 or len(observed) < 2
                else "plastic_rnn"
            ),
        }

        observed.append(actual)
        model.train_on_history(observed)
        record["optimizer_updates_after_feedback"] = model.optimizer_updates
        per_key_seconds.append(time.perf_counter() - started)
        records.append(record)

    last_probabilities, activity, edges = model.predict_next(observed)
    _validate_prediction(unittest.TestCase(), last_probabilities, activity, edges)
    hits = [bool(record["hit"]) for record in records]
    last_50 = hits[-50:]
    summary: dict[str, object] = {
        "session_id": name,
        "seed": seed,
        "sequence_length": len(sequence),
        "sequence_kind": name.split("_seed", 1)[0],
        "total_accuracy": sum(hits) / len(hits) if hits else None,
        "last_50_accuracy": sum(last_50) / len(last_50) if last_50 else None,
        "last_50_predictions": len(last_50),
        "elapsed_seconds": sum(per_key_seconds),
        "mean_seconds_per_key": statistics.mean(per_key_seconds),
        "p50_seconds_per_key": statistics.median(per_key_seconds),
        "p95_seconds_per_key": sorted(per_key_seconds)[
            min(len(per_key_seconds) - 1, math.ceil(len(per_key_seconds) * 0.95) - 1)
        ],
        "optimizer_updates": model.optimizer_updates,
        "last_training_pairs": model.last_training_pairs,
        "eta_after_training": float(model.eta.detach().item()),
        "activity_count": len(activity),
        "effective_edge_shape": [len(edges), len(edges[0])],
        "final_probabilities": last_probabilities,
    }
    return records, summary


def _write_artifacts(records: list[dict[str, object]], summaries: list[dict[str, object]], extra: dict[str, object]) -> None:
    if not OUTPUT_DIR:
        return
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "model": {
            "class": "PlasticRNN",
            "neuron_count": NEURON_COUNT,
            "input_count": INPUT_COUNT,
            "context_length": 32,
            "updates_per_press": 3,
            "seed_policy": "fixed seed; local RNG fork",
        },
        "summaries": summaries,
        "checks": extra,
        "random_sequence_note": (
            "Accuracy is descriptive only. This run checks that each prediction "
            "uses only the observed prefix; it does not claim human random behavior is predictable."
        ),
    }
    (output / "model_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


class PlasticRNNContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_fixed_topology_cold_start_and_strict_digit_types(self) -> None:
        model = PlasticRNN(seed=SEED)
        self.assertEqual(tuple(model.input_weights.shape), (20, 10))
        self.assertEqual(tuple(model.recurrent_weights.shape), (20, 20))
        state = model.initial_state()
        self.assertEqual(tuple(state.activity.shape), (20,))
        self.assertEqual(tuple(state.fast_edges.shape), (20, 20))

        for history in ([], [7]):
            probabilities, activity, edges = model.predict_next(history)
            _validate_prediction(self, probabilities, activity, edges)
            self.assertEqual(probabilities, [0.1] * 10)

        for invalid in (True, False, 2.0, 2.5, "2", None, -1, 10):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    model.step(invalid, model.initial_state())  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    model.predict_next([invalid])  # type: ignore[list-item]

        model.train_on_history([0, 1, 2, 3])
        probabilities, activity, edges = model.predict_next([0, 1, 2, 3])
        _validate_prediction(self, probabilities, activity, edges)
        self.assertEqual(tuple(model.recurrent_weights.shape), (20, 20))
        self.assertEqual(tuple(model.initial_state().activity.shape), (20,))

    def test_seed_is_reproducible_without_changing_global_rng(self) -> None:
        torch.manual_seed(9001)
        rng_before = torch.random.get_rng_state().clone()
        first = PlasticRNN(seed=SEED)
        rng_after = torch.random.get_rng_state()
        second = PlasticRNN(seed=SEED)
        self.assertTrue(torch.equal(rng_before, rng_after))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_shared_fast_edge_gain_has_gradient_and_changes(self) -> None:
        model = PlasticRNN(seed=SEED)
        history = [index % 10 for index in range(32)]
        eta_before = float(model.eta.detach().item())
        model.optimizer.zero_grad(set_to_none=True)
        state = model.initial_state()
        losses = []
        for source, target in zip(history[:-1], history[1:]):
            logits, state = model.step(source, state)
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])))
        torch.stack(losses).mean().backward()
        gradient = float(model.eta_raw.grad.item()) if model.eta_raw.grad is not None else 0.0
        self.assertTrue(math.isfinite(gradient))
        self.assertGreater(abs(gradient), 1e-12)

        model.train_on_history(history)
        eta_after = float(model.eta.detach().item())
        self.assertTrue(math.isfinite(eta_after))
        self.assertGreater(abs(eta_after - eta_before), 1e-7)
        self.assertEqual(model.optimizer_updates, model.updates_per_press)
        if OUTPUT_DIR:
            output = Path(OUTPUT_DIR)
            output.mkdir(parents=True, exist_ok=True)
            (output / "fast_edge_gradient.json").write_text(
                json.dumps(
                    {
                        "seed": SEED,
                        "history_length": len(history),
                        "eta_before": eta_before,
                        "eta_raw_gradient_before_update": gradient,
                        "eta_after_three_updates": eta_after,
                        "optimizer_updates": model.optimizer_updates,
                        "gradient_finite_nonzero": math.isfinite(gradient) and abs(gradient) > 1e-12,
                        "eta_changed": abs(eta_after - eta_before) > 1e-7,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_prequential_sequences_learn_and_repeat_exactly(self) -> None:
        cycle = [index % 10 for index in range(160)]
        repeated = [index % 10 for index in range(160)]
        two_ones = [value for _ in range(40) for value in (0, 0, 1, 1)]
        rng = random.Random(SEED)
        random_digits = [rng.randrange(10) for _ in range(120)]

        all_records: list[dict[str, object]] = []
        summaries: list[dict[str, object]] = []
        for name, sequence in (
            ("cycle_seed20261026_run1", cycle),
            ("cycle_seed20261026_run2", repeated),
            ("0011_seed20261026", two_ones),
            ("random_seed20261026", random_digits),
        ):
            records, summary = _run_prequential(name, sequence, SEED)
            all_records.extend(records)
            summaries.append(summary)

        run1 = [record for record in all_records if record["session_id"] == "cycle_seed20261026_run1"]
        run2 = [record for record in all_records if record["session_id"] == "cycle_seed20261026_run2"]
        self.assertEqual(len(run1), len(run2))
        for first, second in zip(run1, run2):
            self.assertEqual(first["prediction"], second["prediction"])
            self.assertEqual(first["probabilities"], second["probabilities"])
        for summary in summaries[:3]:
            self.assertGreaterEqual(float(summary["last_50_accuracy"]), 0.80)
        # The random run deliberately has no target accuracy threshold.
        self.assertEqual(len([r for r in all_records if r["session_id"] == "random_seed20261026"]), 120)

        cycle_summary = summaries[0]
        self.assertEqual(cycle_summary["optimizer_updates"], (len(cycle) - 1) * 3)
        self.assertEqual(cycle_summary["activity_count"], 20)
        self.assertEqual(cycle_summary["effective_edge_shape"], [20, 20])
        _write_artifacts(
            all_records,
            summaries,
            {
                "fixed_seed_probability_trajectories_identical": True,
                "prequential_order": "predict and freeze on observed prefix, then append label and train",
                "first_prediction_uniform": all(
                    record["probabilities"] == [0.1] * 10
                    for record in run1[:2]
                ),
                "cycle_last_50_accuracy_threshold": 0.80,
                "random_accuracy_threshold": None,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
