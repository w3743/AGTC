"""Contract checks for the persistent rank-r UORO model.

Run with KEYPRESS20_QA_OUTPUT set to retain deterministic test evidence.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from online_model import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANK,
    DEFAULT_SPARSITY_STRENGTH,
    INPUT_COUNT,
    NEURON_COUNT,
    PARAMETER_SIZE,
    STATE_SIZE,
    OnlinePlasticRNN,
    functional_logits,
    functional_state_step,
)


SEED = 20261027
OUTPUT_DIR = os.environ.get("KEYPRESS20_QA_OUTPUT")


def _assert_same_tree(test: unittest.TestCase, actual: object, expected: object, path: str = "root") -> None:
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        test.assertIsInstance(actual, torch.Tensor, path)
        test.assertIsInstance(expected, torch.Tensor, path)
        test.assertTrue(torch.equal(actual, expected), path)
    elif isinstance(actual, dict) or isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        test.assertIsInstance(expected, dict, path)
        test.assertEqual(set(actual), set(expected), path)
        for key in actual:
            _assert_same_tree(test, actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected), path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_same_tree(test, left, right, f"{path}[{index}]")
    else:
        test.assertEqual(actual, expected, path)


class OnlinePlasticRNNContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_defaults_shapes_cold_start_and_read_only_prediction(self) -> None:
        model = OnlinePlasticRNN(seed=SEED)
        config = model.config()
        diag = model.diagnostics()
        self.assertEqual(config["mode"], "uoro")
        self.assertEqual(config["uoro_rank"], DEFAULT_RANK)
        self.assertTrue(config["exact_readout_step"])
        self.assertEqual(config["learning_rate"], DEFAULT_LEARNING_RATE)
        self.assertEqual(config["sparsity_strength"], DEFAULT_SPARSITY_STRENGTH)
        self.assertEqual(tuple(model.parameter_vector().shape), (PARAMETER_SIZE,))
        self.assertEqual(tuple(model.state_vector().shape), (STATE_SIZE,))
        self.assertEqual(len(diag["input_weights"]), NEURON_COUNT)
        self.assertTrue(all(len(row) == INPUT_COUNT for row in diag["input_weights"]))
        self.assertEqual(len(diag["input_nonzero_by_digit"]), INPUT_COUNT)
        self.assertEqual(diag["input_nonzero_total"], sum(diag["input_nonzero_by_digit"]))
        self.assertEqual(diag["factor_bytes"], DEFAULT_RANK * (STATE_SIZE + PARAMETER_SIZE) * 4)
        self.assertTrue(diag["finite"])

        before = model.checkpoint_state()
        probabilities, activity, edges = model.predict_next()
        again = model.predict_next([9] * 100)
        self.assertEqual(probabilities, [0.1] * INPUT_COUNT)
        self.assertEqual((probabilities, activity, edges), again)
        _assert_same_tree(self, model.checkpoint_state(), before, "predict_next must be read-only")

    def test_observe_consumes_once_and_first_digit_has_no_transition_update(self) -> None:
        torch.manual_seed(9001)
        global_rng_before = torch.random.get_rng_state().clone()
        model = OnlinePlasticRNN(seed=SEED)
        self.assertTrue(torch.equal(global_rng_before, torch.random.get_rng_state()))

        initial = model.checkpoint_state()
        for invalid in (True, False, 2.0, 2.5, "2", None, -1, 10):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    model.observe(invalid)  # type: ignore[arg-type]
        _assert_same_tree(self, model.checkpoint_state(), initial, "invalid digits must be side-effect free")

        first_loss = model.observe(7)
        self.assertIsNone(first_loss)
        self.assertEqual(model.diagnostics()["observed_steps"], 1)
        self.assertEqual(model.diagnostics()["optimizer_updates"], 0)
        self.assertEqual(model.diagnostics()["last_training_pairs"], 0)
        self.assertTrue(model.diagnostics()["exact_readout_step"])
        self.assertEqual(
            model.diagnostics()["factor_bytes"],
            2 * DEFAULT_RANK * (STATE_SIZE + PARAMETER_SIZE) * 4,
        )
        self.assertGreater(model.diagnostics()["pending_exact_cache_bytes"], 0)
        self.assertEqual(model.predict_next()[0], [0.1] * INPUT_COUNT)

        second_loss = model.observe(3)
        self.assertIsInstance(second_loss, float)
        self.assertTrue(math.isfinite(float(second_loss)))
        self.assertEqual(model.diagnostics()["observed_steps"], 2)
        self.assertEqual(model.diagnostics()["optimizer_updates"], 1)
        self.assertEqual(model.diagnostics()["last_training_pairs"], 1)

        model.observe(3)
        self.assertEqual(model.diagnostics()["observed_steps"], 3)
        self.assertEqual(model.diagnostics()["optimizer_updates"], 2)
        self.assertTrue(torch.equal(global_rng_before, torch.random.get_rng_state()))

    def test_persistent_state_after_32_differs_from_replaying_only_the_tail(self) -> None:
        def canary_model() -> OnlinePlasticRNN:
            model = OnlinePlasticRNN(seed=SEED, rank=2, sparsity_strength=0.0)
            with torch.no_grad():
                model.input_weights.zero_()
                model.input_weights[0, 6] = 1.0
                model.recurrent_weights.zero_()
                model.recurrent_weights[0, 0] = 1.1
                model.bias.zero_()
                model.eta_raw.zero_()
            return model

        # Fixed-parameter canary isolates recurrent persistence from training:
        # cue 6 activates one self-connected neuron, while all zero inputs leave
        # it alone. Replaying only 32 zeros from a zero state cannot recreate it.
        full_canary = canary_model()
        full_canary.compute_estimate(6)
        for _ in range(40):
            full_canary.compute_estimate(0)
        tail_canary = canary_model()
        for _ in range(32):
            tail_canary.compute_estimate(0)
        self.assertEqual(full_canary.diagnostics()["optimizer_updates"], 0)
        self.assertEqual(full_canary.diagnostics()["observed_steps"], 41)
        self.assertGreater(float(full_canary.h[0].item()), 0.1)
        self.assertEqual(float(tail_canary.h[0].item()), 0.0)

        digits = [((index * 5) + 3) % INPUT_COUNT for index in range(48)]
        full = OnlinePlasticRNN(seed=SEED, rank=2, sparsity_strength=0.0)
        for digit in digits:
            full.observe(digit)
        full_checkpoint = full.checkpoint_state()

        replay = OnlinePlasticRNN(seed=SEED, rank=2, sparsity_strength=0.0)
        for digit in digits[-32:]:
            replay.observe(digit)
        replay_checkpoint = replay.checkpoint_state()

        self.assertEqual(full_checkpoint["runtime"]["observed_steps"], 48)
        self.assertEqual(replay_checkpoint["runtime"]["observed_steps"], 32)
        self.assertFalse(torch.equal(full.state_vector(), replay.state_vector()))
        self.assertFalse(torch.equal(full_checkpoint["runtime"]["uoro_a"], replay_checkpoint["runtime"]["uoro_a"]))

    def test_proximal_zero_weights_can_regrow_without_a_mask(self) -> None:
        same_penalty = OnlinePlasticRNN(seed=SEED, sparsity_strength=DEFAULT_SPARSITY_STRENGTH)
        with torch.no_grad():
            same_penalty.input_weights.zero_()
        same_penalty.observe(0)
        self.assertEqual(same_penalty.diagnostics()["input_nonzero_total"], 0)
        same_penalty.observe(1)
        same_penalty_diag = same_penalty.diagnostics()
        self.assertEqual(same_penalty_diag["sparsity_strength"], DEFAULT_SPARSITY_STRENGTH)
        self.assertGreater(same_penalty_diag["input_nonzero_total"], 0)

        model = OnlinePlasticRNN(seed=SEED, sparsity_strength=1000.0)
        for digit in (0, 1, 2):
            model.observe(digit)
        zero_diag = model.diagnostics()
        self.assertEqual(zero_diag["input_nonzero_total"], 0)
        self.assertTrue(all(value == 0 for row in zero_diag["input_weights"] for value in row))

        # The candidate remains trainable after exact-zero proximal pruning.
        model.sparsity_strength = 0.0
        for digit in (3, 4, 5, 6, 7, 0, 1, 2):
            model.observe(digit)
        regrown_diag = model.diagnostics()
        self.assertGreater(regrown_diag["input_nonzero_total"], 0)
        self.assertTrue(regrown_diag["finite"])

    def test_checkpoint_restores_optimizer_runtime_rng_and_continuation(self) -> None:
        uninterrupted = OnlinePlasticRNN(
            seed=SEED,
            rank=3,
            learning_rate=0.003,
            sparsity_strength=0.02,
        )
        for digit in [4, 1, 8, 2, 6, 3, 9, 0, 7, 5, 4, 1, 8]:
            uninterrupted.observe(digit)
        checkpoint = uninterrupted.checkpoint_state()
        prediction_before = uninterrupted.predict_next()
        prediction_after = uninterrupted.predict_next([0] * 40)
        self.assertEqual(prediction_before, prediction_after)
        _assert_same_tree(
            self,
            uninterrupted.checkpoint_state(),
            checkpoint,
            "prediction must not mutate the checkpointed stream state",
        )

        resumed = OnlinePlasticRNN(seed=999, rank=1, sparsity_strength=0.0)
        resumed.load_checkpoint_state(checkpoint)
        _assert_same_tree(self, resumed.checkpoint_state(), checkpoint, "restored checkpoint")
        self.assertEqual(uninterrupted.predict_next(), resumed.predict_next())

        for digit in (6, 2, 9, 1, 8, 3):
            left_loss = uninterrupted.observe(digit)
            right_loss = resumed.observe(digit)
            if left_loss is None:
                self.assertIsNone(right_loss)
            else:
                self.assertAlmostEqual(float(left_loss), float(right_loss), places=12)
            _assert_same_tree(
                self,
                uninterrupted.checkpoint_state(),
                resumed.checkpoint_state(),
                "optimizer/UORO/RNG continuation",
            )

    def test_uoro_monte_carlo_gradient_mean_tracks_exact_autograd(self) -> None:
        """Compare injected-random UORO estimates with an exact short BPTT gradient."""

        sequence = [2, 7, 1, 4, 3]
        rank = 4
        trials = 256
        base = OnlinePlasticRNN(seed=SEED, rank=rank, sparsity_strength=0.0)
        theta = base.parameter_vector().detach().clone().requires_grad_(True)
        state = torch.zeros(STATE_SIZE, dtype=theta.dtype, device=theta.device)
        exact_losses = []
        for source, target in zip(sequence[:-1], sequence[1:]):
            state = functional_state_step(source, state, theta)
            logits = functional_logits(state, theta)
            exact_losses.append(
                F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([target], dtype=torch.long, device=logits.device),
                )
            )
        exact_gradient = torch.autograd.grad(torch.stack(exact_losses).sum(), theta)[0].detach()

        gradient_sum = torch.zeros_like(exact_gradient)
        for trial in range(trials):
            estimator = OnlinePlasticRNN(seed=SEED, rank=rank, sparsity_strength=0.0)
            trial_rng = torch.Generator(device="cpu").manual_seed(740000 + trial)
            trial_gradient = torch.zeros_like(exact_gradient)
            for digit in sequence:
                draws = torch.randint(0, 2, (rank, STATE_SIZE), generator=trial_rng)
                signs = draws.mul(2).sub(1).to(dtype=exact_gradient.dtype)
                _, estimate = estimator.compute_estimate(digit, random_signs=signs)
                if estimate is not None:
                    trial_gradient.add_(estimate)
            gradient_sum.add_(trial_gradient)

        mean_gradient = gradient_sum / trials
        relative_error = torch.linalg.vector_norm(mean_gradient - exact_gradient) / (
            torch.linalg.vector_norm(exact_gradient) + 1e-12
        )
        cosine = F.cosine_similarity(mean_gradient.unsqueeze(0), exact_gradient.unsqueeze(0)).item()
        self.assertEqual(tuple(mean_gradient.shape), (PARAMETER_SIZE,))
        self.assertTrue(bool(torch.isfinite(mean_gradient).all().item()))
        self.assertTrue(bool(torch.isfinite(exact_gradient).all().item()))
        self.assertLess(float(relative_error.item()), 0.65)
        self.assertGreater(float(cosine), 0.55)

        if OUTPUT_DIR:
            output = Path(OUTPUT_DIR)
            output.mkdir(parents=True, exist_ok=True)
            (output / "uoro_gradient_mc.json").write_text(
                json.dumps(
                    {
                        "seed": SEED,
                        "sequence": sequence,
                        "uoro_rank": rank,
                        "trials": trials,
                        "exact_gradient_norm": float(torch.linalg.vector_norm(exact_gradient).item()),
                        "mean_estimate_norm": float(torch.linalg.vector_norm(mean_gradient).item()),
                        "relative_error": float(relative_error.item()),
                        "cosine_similarity": float(cosine),
                        "fixed_seed_sign_generation": "torch.Generator seed 740000 + trial; ±1 rank-by-420 per observed digit",
                        "exact_gradient": exact_gradient.cpu().tolist(),
                        "mean_uoro_gradient": mean_gradient.cpu().tolist(),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
