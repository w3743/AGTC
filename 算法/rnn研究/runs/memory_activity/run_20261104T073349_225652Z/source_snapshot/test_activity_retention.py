"""Independent checks for shared hidden-activity retention in streaming RTRL."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from activity_retention import ActivityRetentionRTRL
from streaming_rtrl import StreamingRTRL


HISTORY = [0, 1, 1, 0, 1]


def reference_unroll(network, history, retention):
    h = torch.zeros(network.width, dtype=network.input_weights.dtype)
    fast = torch.zeros(network.width, network.width, dtype=h.dtype)
    for digit in history:
        previous = h
        inp = F.one_hot(torch.tensor(digit), num_classes=10).to(dtype=h.dtype)
        candidate = torch.tanh(network.input_weights @ inp
                               + (network.recurrent_weights + network.eta * fast) @ previous
                               + network.bias)
        h = retention * previous + (1.0 - retention) * candidate
        fast = (network.trace_decay * fast
                + (1.0 - network.trace_decay) * torch.outer(h, previous))
    return network.W_out @ h + network.b_out, h, fast


class ActivityRetentionTests(unittest.TestCase):
    def test_full_state_rtrl_projections_match_autograd_for_fixed_and_learned(self):
        for mode, seed in (("fixed", 201), ("learned", 202)):
            with self.subTest(mode=mode):
                model = ActivityRetentionRTRL(seed=seed, mode=mode)
                model.network.double()
                model.reset_state()
                with torch.no_grad():
                    model.network.eta_raw.fill_(0.3)
                    if mode == "learned":
                        model.network.activity_logit.fill_(0.2)
                for digit in HISTORY:
                    model.advance(digit)

                reference = ActivityRetentionRTRL(seed=seed, mode=mode).network.double()
                with torch.no_grad():
                    reference.eta_raw.fill_(0.3)
                    if mode == "learned":
                        reference.activity_logit.fill_(0.2)
                retention = (torch.full((20,), 0.5, dtype=torch.float64)
                             if mode == "fixed" else torch.sigmoid(reference.activity_logit))
                _, h, fast = reference_unroll(reference, HISTORY, retention)
                core = [reference.input_weights, reference.recurrent_weights,
                        reference.bias, reference.eta_raw, reference.decay_logit]
                if mode == "learned":
                    core.append(reference.activity_logit)
                generator = torch.Generator().manual_seed(771)
                for _ in range(3):
                    vh = torch.randn(20, generator=generator, dtype=torch.float64)
                    vf = torch.randn(20, 20, generator=generator, dtype=torch.float64)
                    projection = (h * vh).sum() + (fast * vf).sum()
                    grads = torch.autograd.grad(projection, core, retain_graph=True)
                    expected = torch.cat([g.reshape(-1) for g in grads])
                    actual = (model.eh.T @ vh
                              + model.ef.reshape(400, model.core_size).T @ vf.reshape(-1))
                    self.assertTrue(torch.allclose(actual, expected, atol=2e-8, rtol=2e-7),
                                    float((actual - expected).abs().max()))
                self.assertEqual(model.core_size, 641 if mode == "fixed" else 661)

    def test_clipped_online_gradient_matches_full_sequence_autograd(self):
        mode, seed, target = "learned", 203, 2
        model = ActivityRetentionRTRL(seed=seed, mode=mode)
        reference = ActivityRetentionRTRL(seed=seed, mode=mode).network
        with torch.no_grad():
            model.network.eta_raw.fill_(0.3)
            reference.eta_raw.fill_(0.3)
        for digit in HISTORY:
            model.advance(digit)
        retention = torch.sigmoid(reference.activity_logit)
        logits, _, _ = reference_unroll(reference, HISTORY, retention)
        F.cross_entropy(logits.unsqueeze(0), torch.tensor([target])).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        expected = {name: p.grad.detach().clone() for name, p in reference.named_parameters()}
        captured = {}
        opt = model.network.optimizer
        original_step = opt.step

        def capture(*args, **kwargs):
            captured.update({name: p.grad.detach().clone()
                             for name, p in model.network.named_parameters()})
            return original_step(*args, **kwargs)

        opt.step = capture
        stats = model.learn_target(target)
        self.assertIsNotNone(stats)
        for name, expected_grad in expected.items():
            self.assertTrue(torch.allclose(captured[name], expected_grad, atol=2e-5, rtol=2e-5), name)

    def test_seed_compatibility_initial_state_and_fixed_retention(self):
        for seed in (204, 205):
            baseline = StreamingRTRL(seed=seed)
            fixed = ActivityRetentionRTRL(seed=seed, mode="fixed")
            learned = ActivityRetentionRTRL(seed=seed, mode="learned")
            for name, value in baseline.network.state_dict().items():
                self.assertTrue(torch.equal(value, fixed.network.state_dict()[name]), name)
                self.assertTrue(torch.equal(value, learned.network.state_dict()[name]), name)
            self.assertTrue(torch.equal(fixed.activity_retention, torch.full((20,), 0.5)))
            for digit in HISTORY:
                baseline.advance(digit)
                fixed.advance(digit)
                learned.advance(digit)
            for model in (fixed, learned):
                self.assertTrue(torch.equal(fixed.h, model.h))
                self.assertTrue(torch.equal(fixed.fast_edges, model.fast_edges))
                self.assertTrue(torch.equal(fixed.predict(), model.predict()))
            self.assertTrue(torch.equal(baseline.network.W_out, fixed.network.W_out))
            # With a=0 the new recurrence exactly recovers the baseline.
            zero = ActivityRetentionRTRL(seed=seed, mode="fixed")
            with torch.no_grad():
                zero._fixed_activity.zero_()
            for digit in HISTORY:
                zero.advance(digit)
            self.assertTrue(torch.equal(baseline.h, zero.h))
            self.assertTrue(torch.equal(baseline.fast_edges, zero.fast_edges))
            self.assertTrue(torch.equal(baseline.predict(), zero.predict()))

    def test_fixed_retention_is_constant_and_learned_retention_updates(self):
        fixed = ActivityRetentionRTRL(seed=206, mode="fixed")
        fixed_before = fixed.activity_retention.clone()
        learned = ActivityRetentionRTRL(seed=206, mode="learned")
        learned_before = learned.network.activity_logit.detach().clone()
        for digit in HISTORY:
            fixed.advance(digit)
            learned.advance(digit)
        fixed.learn_target(2)
        stats = learned.learn_target(2)
        self.assertTrue(torch.equal(fixed_before, fixed.activity_retention))
        self.assertGreater(stats["activity_grad_norm"], 0.0)
        self.assertTrue(any(abs(x) > 0 for x in stats["activity_grad"]))
        self.assertFalse(torch.equal(learned_before, learned.network.activity_logit.detach()))

    def test_long_stream_is_bounded_and_finite(self):
        for mode, core in (("fixed", 641), ("learned", 661)):
            with self.subTest(mode=mode):
                model = ActivityRetentionRTRL(seed=207, mode=mode)
                for i in range(600):
                    model.observe(i % 10)
                self.assertEqual((model.steps, model.updates), (600, 599))
                self.assertEqual(model.eh.shape, (20, core))
                self.assertEqual(model.ef.shape, (20, 20, core))
                for value in (model.h, model.fast_edges, model.eh, model.ef):
                    self.assertTrue(torch.isfinite(value).all())
                    self.assertLess(value.numel(), 300_000)
                self.assertFalse({"history", "histories", "graph", "graphs", "computation_graph"}
                                 .intersection(vars(model)))

    def test_optimizer_groups_and_checkpoint_roundtrip(self):
        for mode, expected in (("fixed", 851), ("learned", 871)):
            model = ActivityRetentionRTRL(seed=208, mode=mode)
            self.assertEqual(sum(p.numel() for p in model.network.parameters()), expected)
            groups = model.network.optimizer.param_groups
            parameters = [p for group in groups for p in group["params"]]
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual({id(p) for p in parameters},
                             {id(p) for p in model.network.parameters()})
            self.assertEqual(groups[0]["weight_decay"], 1e-4)
            self.assertEqual(groups[1]["weight_decay"], 0.0)
            self.assertEqual(len(model.network.optimizer.state), 0)
            if mode == "learned":
                self.assertEqual(set(map(id, groups[1]["params"])),
                                 {id(model.network.decay_logit), id(model.network.activity_logit)})
            else:
                self.assertEqual(groups[1]["params"], [model.network.decay_logit])
            for digit in HISTORY:
                model.observe(digit)
            self.assertTrue(model.network.optimizer.state)
            payload = model.state_payload()
            restored = ActivityRetentionRTRL(seed=999, mode=mode)
            restored.load_state_payload(payload)
            self.assertTrue(torch.equal(model.h, restored.h))
            self.assertTrue(torch.equal(model.fast_edges, restored.fast_edges))
            self.assertTrue(torch.equal(model.eh, restored.eh))
            self.assertTrue(torch.equal(model.ef, restored.ef))
            self.assertTrue(torch.equal(model.predict(), restored.predict()))
            self.assertEqual(len(model.network.optimizer.state), len(restored.network.optimizer.state))
            if mode == "learned":
                self.assertTrue(torch.equal(model.activity_retention, restored.activity_retention))


if __name__ == "__main__":
    unittest.main()
