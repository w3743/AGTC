"""Checks for the exact-online RTRL ablation reference.

Set KEYPRESS20_REFERENCE_OUTPUT to retain compact JSON evidence from a run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from ablation_reference import ExactOnlineRNN, TruncatedOnlineRNN
from online_model import (
    INPUT_COUNT,
    PARAMETER_SIZE,
    STATE_SIZE,
    OnlinePlasticRNN,
    functional_logits,
    functional_state_step,
)


SEED = 20261027
OUTPUT_DIR = os.environ.get("KEYPRESS20_REFERENCE_OUTPUT")


class ExactOnlineRNNTests(unittest.TestCase):
    evidence: dict[str, object] = {"seed": SEED, "fixed_theta_checks": []}

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        if OUTPUT_DIR:
            output = Path(OUTPUT_DIR)
            output.mkdir(parents=True, exist_ok=True)
            (output / "reference_checks.json").write_text(
                json.dumps(cls.evidence, ensure_ascii=False, allow_nan=False, indent=2)
                + "\n",
                encoding="utf-8",
            )

    def test_fixed_theta_state_sensitivity_and_loss_gradient_match_jacrev(self) -> None:
        sequence = [2, 7, 1, 4, 3]
        targets = {2: 8, 3: 6, 5: 9}

        for length in (2, 3, 5):
            with self.subTest(length=length):
                model = ExactOnlineRNN(
                    seed=SEED,
                    rank=2,
                    learning_rate=0.003,
                    sparsity_strength=0.0,
                    weight_decay=0.0,
                )
                with torch.no_grad():
                    model.eta_raw.fill_(0.31)
                    model.decay_logit.fill_(0.62)
                    model.scale_raw.fill_(0.87)
                    model.h.copy_(torch.linspace(-0.27, 0.33, 20))
                    model.fast_edges.copy_(
                        torch.linspace(-0.09, 0.11, 400).view(20, 20)
                    )

                initial_state = model.state_vector()
                theta = model.parameter_vector().requires_grad_(True)
                selected_sequence = sequence[:length]

                def unroll(parameters: torch.Tensor) -> torch.Tensor:
                    state = initial_state
                    for digit in selected_sequence:
                        state = functional_state_step(digit, state, parameters)
                    return state

                expected_state = unroll(theta).detach()
                expected_sensitivity = torch.func.jacrev(unroll)(theta).detach()

                for digit in selected_sequence:
                    model.compute_estimate(digit)

                actual_state = model.state_vector()
                actual_sensitivity = model.state_sensitivity.detach()
                torch.testing.assert_close(
                    actual_state, expected_state, rtol=1e-6, atol=2e-7
                )
                torch.testing.assert_close(
                    actual_sensitivity,
                    expected_sensitivity,
                    rtol=8e-5,
                    atol=3e-6,
                )

                target = targets[length]

                def loss_after_unroll(parameters: torch.Tensor) -> torch.Tensor:
                    logits = functional_logits(unroll(parameters), parameters)
                    return F.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([target], dtype=torch.long, device=logits.device),
                    )

                expected_gradient = torch.func.grad(loss_after_unroll)(theta).detach()
                loss_value, actual_gradient = model._estimate_current_loss_gradient(target)
                torch.testing.assert_close(
                    actual_gradient,
                    expected_gradient,
                    rtol=8e-5,
                    atol=3e-6,
                )
                self.assertTrue(math.isfinite(loss_value))
                self.assertEqual(tuple(actual_sensitivity.shape), (STATE_SIZE, PARAMETER_SIZE))

                self.evidence["fixed_theta_checks"].append(  # type: ignore[union-attr]
                    {
                        "steps": length,
                        "sequence": selected_sequence,
                        "eta_raw": 0.31,
                        "decay_logit": 0.62,
                        "scale_raw": 0.87,
                        "initial_activity_norm": float(torch.linalg.vector_norm(initial_state[:20]).item()),
                        "initial_fast_edge_norm": float(torch.linalg.vector_norm(initial_state[20:]).item()),
                        "state_sensitivity_max_abs_error": float(
                            (actual_sensitivity - expected_sensitivity).abs().max().item()
                        ),
                        "loss_gradient_max_abs_error": float(
                            (actual_gradient - expected_gradient).abs().max().item()
                        ),
                        "loss": loss_value,
                    }
                )

    def test_same_initialization_first_key_and_online_finiteness(self) -> None:
        exact = ExactOnlineRNN(
            seed=SEED,
            rank=3,
            learning_rate=0.003,
            sparsity_strength=0.02,
            weight_decay=0.001,
        )
        uoro = OnlinePlasticRNN(
            seed=SEED,
            rank=3,
            learning_rate=0.003,
            sparsity_strength=0.02,
        )
        self.assertTrue(torch.equal(exact.parameter_vector(), uoro.parameter_vector()))
        self.assertEqual(exact.optimizer.param_groups[0]["weight_decay"], 0.001)
        self.assertEqual(exact.config()["mode"], "exact_rtrl")
        self.assertFalse(exact.config()["rank_used"])
        self.assertEqual(exact.diagnostics()["optimizer_updates"], 0)

        first_loss = exact.observe(7)
        self.assertIsNone(first_loss)
        self.assertEqual(exact.diagnostics()["observed_steps"], 1)
        self.assertEqual(exact.diagnostics()["optimizer_updates"], 0)
        self.assertEqual(exact.predict_next()[0], [1.0 / INPUT_COUNT] * INPUT_COUNT)

        sequence = [3, 8, 1, 4, 9, 2, 6, 0, 5, 7, 4, 1, 8, 3, 9, 2]
        losses: list[float] = []
        for digit in sequence:
            loss = exact.observe(digit)
            if loss is not None:
                self.assertTrue(math.isfinite(float(loss)))
                losses.append(float(loss))
            diagnostics = exact.diagnostics()
            self.assertTrue(diagnostics["finite"])
            self.assertTrue(math.isfinite(diagnostics["exact_state_sensitivity_norm"]))
            self.assertEqual(diagnostics["observed_steps"], len(losses) + 1)

        self.assertEqual(exact.diagnostics()["optimizer_updates"], len(sequence))
        self.assertEqual(len(losses), len(sequence))
        self.assertTrue(exact.diagnostics()["finite"])
        self.assertEqual(
            exact.diagnostics()["sensitivity_bytes"],
            STATE_SIZE * PARAMETER_SIZE * torch.tensor([], dtype=torch.float32).element_size(),
        )
        self.evidence["online_fixed_seed"] = {
            "sequence": [7, *sequence],
            "loss_count": len(losses),
            "final_loss": losses[-1],
            "optimizer_updates": exact.diagnostics()["optimizer_updates"],
            "sensitivity_norm": exact.diagnostics()["exact_state_sensitivity_norm"],
            "finite": exact.diagnostics()["finite"],
            "same_initialization_with_production_uoro": True,
            "first_key_updates": 0,
            "weight_decay": exact.weight_decay,
        }

    def test_full_rtrl_checkpoint_continues_with_sensitivity_intact(self) -> None:
        uninterrupted = ExactOnlineRNN(
            seed=SEED,
            rank=2,
            learning_rate=0.003,
            sparsity_strength=0.02,
            weight_decay=0.001,
        )
        for digit in (4, 1, 8, 2, 6, 3, 9):
            uninterrupted.observe(digit)
        checkpoint = uninterrupted.checkpoint_state()

        resumed = ExactOnlineRNN(seed=999, rank=1, sparsity_strength=0.0)
        resumed.load_checkpoint_state(checkpoint)
        torch.testing.assert_close(resumed.state_sensitivity, uninterrupted.state_sensitivity, rtol=0, atol=0)
        torch.testing.assert_close(resumed.state_vector(), uninterrupted.state_vector(), rtol=0, atol=0)
        torch.testing.assert_close(resumed.parameter_vector(), uninterrupted.parameter_vector(), rtol=0, atol=0)
        self.assertEqual(resumed.weight_decay, 0.001)

        continuation = (5, 0, 7, 2, 9)
        for digit in continuation:
            left_loss = uninterrupted.observe(digit)
            right_loss = resumed.observe(digit)
            self.assertAlmostEqual(float(left_loss), float(right_loss), places=7)
            torch.testing.assert_close(resumed.state_sensitivity, uninterrupted.state_sensitivity, rtol=0, atol=0)
            torch.testing.assert_close(resumed.state_vector(), uninterrupted.state_vector(), rtol=0, atol=0)
            torch.testing.assert_close(resumed.parameter_vector(), uninterrupted.parameter_vector(), rtol=0, atol=0)
        self.assertTrue(resumed.diagnostics()["finite"])
        self.evidence["full_checkpoint"] = {
            "saved_after_steps": 7,
            "continuation_steps": len(continuation),
            "sensitivity_restored": True,
            "optimizer_updates": resumed.optimizer_updates,
            "finite": resumed.diagnostics()["finite"],
        }

    def test_truncated_horizon_matches_full_gradient_and_restores_queue(self) -> None:
        sequence = [2, 7, 1, 4, 3]
        model = TruncatedOnlineRNN(
            seed=SEED,
            rank=2,
            learning_rate=0.003,
            sparsity_strength=0.0,
            weight_decay=0.001,
            credit_horizon=32,
        )
        with torch.no_grad():
            model.eta_raw.fill_(0.29)
            model.decay_logit.fill_(0.54)
            model.scale_raw.fill_(0.91)
            model.h.copy_(torch.linspace(-0.22, 0.29, 20))
            model.fast_edges.copy_(torch.linspace(-0.08, 0.1, 400).view(20, 20))
        initial_state = model.state_vector()
        theta = model.parameter_vector().requires_grad_(True)

        def unroll(parameters: torch.Tensor) -> torch.Tensor:
            state = initial_state
            for digit in sequence:
                state = functional_state_step(digit, state, parameters)
            return state

        expected_state = unroll(theta).detach()
        expected_sensitivity = torch.func.jacrev(unroll)(theta).detach()
        for digit in sequence:
            model.compute_estimate(digit)
        torch.testing.assert_close(model.state_vector(), expected_state, rtol=1e-6, atol=2e-7)
        fixed_state_error = float(
            (model.state_vector() - expected_state).abs().max().item()
        )
        target = 8

        def loss_after_unroll(parameters: torch.Tensor) -> torch.Tensor:
            logits = functional_logits(unroll(parameters), parameters)
            return F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target], dtype=torch.long, device=logits.device),
            )

        expected_gradient = torch.func.grad(loss_after_unroll)(theta).detach()
        _, actual_gradient = model._estimate_current_loss_gradient(target)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=8e-5, atol=3e-6)
        self.assertEqual(model.diagnostics()["transition_history_length"], len(sequence))

        checkpoint = model.checkpoint_state()
        resumed = TruncatedOnlineRNN(seed=123, rank=1, credit_horizon=1)
        resumed.load_checkpoint_state(checkpoint)
        self.assertEqual(resumed.credit_horizon, 32)
        self.assertEqual(resumed.diagnostics()["transition_history_length"], len(sequence))
        torch.testing.assert_close(resumed.state_vector(), model.state_vector(), rtol=0, atol=0)
        torch.testing.assert_close(resumed.parameter_vector(), model.parameter_vector(), rtol=0, atol=0)
        next_target = 6
        _, left_gradient = model._estimate_current_loss_gradient(next_target)
        _, right_gradient = resumed._estimate_current_loss_gradient(next_target)
        torch.testing.assert_close(right_gradient, left_gradient, rtol=0, atol=0)

        # The forward state remains continuous as the bounded queue rolls over.
        for index in range(35):
            model.compute_estimate(index % INPUT_COUNT)
        self.assertEqual(model.diagnostics()["transition_history_length"], 32)
        self.assertEqual(model.diagnostics()["observed_steps"], len(sequence) + 35)
        self.assertTrue(model.diagnostics()["finite"])
        self.evidence["truncated_horizon"] = {
            "horizon": 32,
            "fixed_theta_steps": len(sequence),
            "forward_state_max_abs_error": fixed_state_error,
            "full_jacrev_sensitivity_norm": float(
                torch.linalg.vector_norm(expected_sensitivity).item()
            ),
            "loss_gradient_max_abs_error": float(
                (actual_gradient - expected_gradient).abs().max().item()
            ),
            "checkpoint_queue_length": len(sequence),
            "post_rollover_queue_length": model.diagnostics()["transition_history_length"],
            "continuous_state_steps": 35,
            "finite": model.diagnostics()["finite"],
        }

    def test_full_and_truncated_match_during_online_parameter_updates(self) -> None:
        full = ExactOnlineRNN(
            seed=SEED,
            rank=2,
            learning_rate=0.003,
            sparsity_strength=0.02,
            weight_decay=0.001,
        )
        truncated = TruncatedOnlineRNN(
            seed=SEED,
            rank=2,
            learning_rate=0.003,
            sparsity_strength=0.02,
            weight_decay=0.001,
            credit_horizon=32,
        )
        with torch.no_grad():
            full.eta_raw.fill_(0.28)
            truncated.eta_raw.fill_(0.28)
            full.decay_logit.fill_(0.59)
            truncated.decay_logit.fill_(0.59)
        sequence = [7, 3, 8, 1, 4, 9, 2, 6, 0, 5, 7, 4, 1, 8, 3, 9, 2, 6, 5, 0]
        maximum_gradient_error = 0.0
        for digit in sequence:
            if full.observed_steps:
                _, full_gradient = full._estimate_current_loss_gradient(digit)
                _, truncated_gradient = truncated._estimate_current_loss_gradient(digit)
                torch.testing.assert_close(
                    truncated_gradient, full_gradient, rtol=2e-4, atol=2e-6
                )
                maximum_gradient_error = max(
                    maximum_gradient_error,
                    float((truncated_gradient - full_gradient).abs().max().item()),
                )
            full_loss = full.observe(digit)
            truncated_loss = truncated.observe(digit)
            if full_loss is not None:
                self.assertAlmostEqual(float(full_loss), float(truncated_loss), places=6)
            torch.testing.assert_close(
                full.state_vector(),
                truncated.state_vector(),
                rtol=1e-5,
                atol=5e-8,
            )
            torch.testing.assert_close(
                full.parameter_vector(),
                truncated.parameter_vector(),
                rtol=1e-5,
                atol=5e-8,
            )

        self.assertEqual(full.optimizer_updates, len(sequence) - 1)
        self.assertEqual(truncated.optimizer_updates, len(sequence) - 1)
        self.assertTrue(full.diagnostics()["finite"])
        self.assertTrue(truncated.diagnostics()["finite"])
        self.evidence["updated_full_vs_truncated"] = {
            "sequence_length": len(sequence),
            "updates": full.optimizer_updates,
            "credit_horizon": truncated.credit_horizon,
            "nonzero_eta": float(full.eta.detach().item()),
            "max_gradient_abs_error": maximum_gradient_error,
            "state_and_parameters_identical": True,
            "finite": True,
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
