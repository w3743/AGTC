"""Mechanism tests for the experimental per-neuron retention model."""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(MODEL_ROOT))
from per_neuron_retention import PerNeuronRetentionRNN, validate_mechanism
from model import PlasticRNN


class PerNeuronRetentionTests(unittest.TestCase):
    def test_mechanism_precheck(self):
        result = validate_mechanism()
        self.assertTrue(result["passed"])
        self.assertEqual(result["parameter_count"], 642)

    def test_initial_output_and_state_match_shared_baseline_bitwise(self):
        shared = PlasticRNN(seed=91)
        vector = PerNeuronRetentionRNN(seed=91)
        for history in ([0], [0, 1, 0, 1], [4, 4, 4, 2, 3]):
            expected_logits, expected_state = shared.replay(history)
            actual_logits, actual_state = vector.replay(history)
            self.assertTrue(torch.equal(expected_logits, actual_logits))
            self.assertTrue(torch.equal(expected_state.activity, actual_state.activity))
            self.assertTrue(torch.equal(expected_state.fast_edges, actual_state.fast_edges))

    def test_one_logit_changes_only_its_receiving_trace_row(self):
        left = PerNeuronRetentionRNN(seed=92)
        right = PerNeuronRetentionRNN(seed=92)
        _, state_left = left.replay([0, 1, 0, 1, 1])
        _, state_right = right.replay([0, 1, 0, 1, 1])
        with torch.no_grad():
            right.decay_logit[13, 0] += 0.5
        _, next_left = left.step(0, state_left)
        _, next_right = right.step(0, state_right)
        changed = (next_left.fast_edges != next_right.fast_edges).any(dim=1)
        self.assertEqual(changed.nonzero().flatten().tolist(), [13])

    def test_task_gradient_is_zero_when_eta_is_zero(self):
        model = PerNeuronRetentionRNN(seed=93)
        with torch.no_grad():
            model.eta_raw.zero_()
        logits, _ = model.replay([0, 1, 0, 1, 1, 0])
        torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([1])).backward()
        self.assertIsNotNone(model.decay_logit.grad)
        self.assertTrue(torch.equal(model.decay_logit.grad, torch.zeros_like(model.decay_logit.grad)))

    def test_active_task_gradient_and_shared_gradient_sum(self):
        history = [0, 1, 0, 1, 1, 0]
        vector = PerNeuronRetentionRNN(seed=94)
        shared = PlasticRNN(seed=94)
        with torch.no_grad():
            vector.eta_raw.fill_(0.8)
            shared.eta_raw.fill_(0.8)
        logits, _ = vector.replay(history)
        torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([1])).backward()
        shared_logits, _ = shared.replay(history)
        torch.nn.functional.cross_entropy(shared_logits.unsqueeze(0), torch.tensor([1])).backward()
        grad = vector.decay_logit.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.all(grad.abs().sum(dim=1) > 0))
        self.assertTrue(torch.allclose(grad.sum(), shared.decay_logit.grad, atol=1e-6, rtol=0))

    def test_fresh_adam_zero_gradient_does_not_drift_and_groups_cover_parameters(self):
        model = PerNeuronRetentionRNN(seed=95)
        before = model.decay_logit.detach().clone()
        for _ in range(10):
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            model.optimizer.step()
        self.assertTrue(torch.equal(before, model.decay_logit.detach()))
        groups = model.optimizer.param_groups
        optimized = [p for group in groups for p in group["params"]]
        self.assertEqual(len(optimized), len({id(p) for p in optimized}))
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in model.parameters()})
        self.assertEqual(groups[0]["weight_decay"], 1e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[1]["params"], [model.decay_logit])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 642)


if __name__ == "__main__":
    unittest.main()
