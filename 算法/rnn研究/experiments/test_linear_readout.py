"""Validation for the independent per-neuron-retention linear-readout model."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from linear_readout import LinearReadoutRNN, validate_mechanism

PREVIOUS = Path("C:/pioneer/rnn-research-runs/decimal_guess_20261028/run_20261028T034514_978315Z")


class LinearReadoutTests(unittest.TestCase):
    def test_all_custom_checkpoints(self):
        result = validate_mechanism(PREVIOUS)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checkpoints_checked"], 18)
        self.assertEqual(result["parameter_count"], 851)

    def test_rng_gradient_optimizer_and_width(self):
        torch.manual_seed(771)
        before = torch.random.get_rng_state().clone()
        model = LinearReadoutRNN(seed=18)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(model.decay_logit.shape, (20, 1))
        self.assertEqual(model.trace_decay.shape, (20, 1))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 851)
        logits, _ = model.replay([0, 1, 0, 1, 1, 0])
        F.cross_entropy(logits.unsqueeze(0), torch.tensor([1])).backward()
        self.assertIsNotNone(model.W_out.grad)
        self.assertGreater(float(model.W_out.grad[:, 10:].abs().sum()), 0.0)
        groups = model.optimizer.param_groups
        params = [p for group in groups for p in group["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters()})
        self.assertEqual(groups[0]["weight_decay"], 1e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[1]["params"], [model.decay_logit])
        narrow = LinearReadoutRNN(seed=19, width=18)
        self.assertEqual(narrow.decay_logit.shape, (18, 1))
        self.assertEqual(narrow.train_on_history([0, 1, 0, 1]), narrow.last_training_loss)
        self.assertEqual(narrow.optimizer_updates, 3)


if __name__ == "__main__":
    unittest.main()
