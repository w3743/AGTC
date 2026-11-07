"""Independent behavioral and gradient checks for streaming RTRL."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from streaming_rtrl import StreamingRTRL
from linear_readout import LinearReadoutRNN as LinearReadoutRNN20


HISTORY = [0, 1, 1, 0, 1]


class StreamingRTRLTests(unittest.TestCase):
    def test_initial_prediction_is_uniform_and_does_not_advance(self):
        model = StreamingRTRL(seed=101)
        before = model.h.clone()
        probs = model.predict()
        self.assertEqual(model.steps, 0)
        self.assertTrue(torch.equal(model.h, before))
        self.assertTrue(torch.allclose(probs, torch.full((10,), 0.1, dtype=probs.dtype), atol=1e-7))

    def test_streaming_accounting_and_observe_order(self):
        model = StreamingRTRL(seed=102)
        for i in range(600):
            probs = model.observe(i % 10)
            self.assertEqual(probs.shape, (10,))
        self.assertEqual(model.steps, 600)
        self.assertEqual(model.updates, 599)

    def test_advance_and_learn_target_have_separate_state_and_update_effects(self):
        model = StreamingRTRL(seed=103)
        logits = model.advance(3)
        self.assertEqual(logits.shape, (10,))
        self.assertEqual(model.steps, 1)
        before = model.h.clone()
        stats = model.learn_target(4)
        if stats is not None:
            self.assertTrue(torch.isfinite(torch.as_tensor(stats["loss"])))
        self.assertEqual(model.steps, 1)
        self.assertTrue(torch.equal(model.h, before))
        self.assertEqual(model.updates, 1)

    def test_public_trace_storage_is_bounded_and_finite(self):
        model = StreamingRTRL(seed=104)
        self.assertEqual(model.fast_edges.numel(), 400)
        self.assertEqual(model.eh.numel(), 20 * 641)
        self.assertEqual(model.ef.numel(), 20 * 20 * 641)
        for digit in HISTORY * 4:
            model.advance(digit)
        for value in (model.h, model.fast_edges, model.eh, model.ef):
            self.assertTrue(torch.isfinite(value).all())
            self.assertLess(value.numel(), 300_000)
        forbidden = {"history", "histories", "graph", "graphs", "computation_graph"}
        self.assertFalse(forbidden.intersection(vars(model)))

    def test_rtrl_traces_match_independent_forward_sensitivities(self):
        model = StreamingRTRL(seed=105)
        model.network.double()
        model.reset_state()
        with torch.no_grad():
            model.network.eta_raw.fill_(0.3)
        for digit in HISTORY:
            model.advance(digit)
        # Project the full (h, F) state Jacobian onto three random cotangents.
        # This checks all 420 state coordinates without materializing a 420x641
        # autograd Jacobian and independently exercises the entire RTRL trace.
        reference = LinearReadoutRNN20(seed=105).double()
        with torch.no_grad():
            reference.eta_raw.fill_(0.3)
        core = [reference.input_weights, reference.recurrent_weights,
                reference.bias, reference.eta_raw, reference.decay_logit]
        generator = torch.Generator().manual_seed(771)
        for _ in range(3):
            vh = torch.randn(20, generator=generator, dtype=torch.float64)
            vf = torch.randn(20, 20, generator=generator, dtype=torch.float64)
            _, state = reference.replay(HISTORY)
            projected = (state.activity * vh).sum() + (state.fast_edges * vf).sum()
            grads = torch.autograd.grad(projected, core)
            expected = torch.cat([g.reshape(-1) for g in grads])
            actual = model.eh.T @ vh + model.ef.reshape(400, 641).T @ vf.reshape(-1)
            self.assertTrue(torch.allclose(actual, expected, atol=2e-8, rtol=2e-7),
                            float((actual - expected).abs().max()))
        self.assertEqual(model.eh.shape, (20, 641))
        self.assertEqual(model.ef.shape, (20, 20, 641))

    def test_task_gradient_is_live_for_rtrl_and_zero_for_one_step(self):
        rtrl = StreamingRTRL(seed=106, credit="rtrl")
        one = StreamingRTRL(seed=106, credit="one_step")
        with torch.no_grad():
            rtrl.network.eta_raw.fill_(0.3)
            one.network.eta_raw.fill_(0.3)
        for digit in HISTORY:
            rtrl.advance(digit)
            one.advance(digit)
        captured = {}
        for label, model in (("rtrl", rtrl), ("one", one)):
            opt = model.network.optimizer
            step = opt.step
            def capture(*args, _label=label, _step=step, _model=model, **kwargs):
                captured[_label] = _model.network.decay_logit.grad.detach().clone()
                return _step(*args, **kwargs)
            opt.step = capture
            model.learn_target(2)
        self.assertTrue(torch.all(captured["rtrl"].reshape(20, -1).abs().sum(1) > 0))
        self.assertTrue(torch.equal(captured["one"], torch.zeros_like(captured["one"])))

    def test_online_rtrl_gradient_matches_full_sequence_autograd(self):
        model = StreamingRTRL(seed=109)
        reference = LinearReadoutRNN20(seed=109)
        with torch.no_grad():
            model.network.eta_raw.fill_(0.3)
            reference.eta_raw.fill_(0.3)
        for digit in HISTORY:
            model.advance(digit)
        ref_logits, _ = reference.replay(HISTORY)
        F.cross_entropy(ref_logits.unsqueeze(0), torch.tensor([2])).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        expected = {name: p.grad.detach().clone() for name, p in reference.named_parameters()}
        captured = {}
        opt = model.network.optimizer
        original_step = opt.step
        def capture(*args, **kwargs):
            captured.update({name: p.grad.detach().clone() for name, p in model.network.named_parameters()})
            return original_step(*args, **kwargs)
        opt.step = capture
        model.learn_target(2)
        for name, expected_grad in expected.items():
            self.assertTrue(torch.allclose(captured[name], expected_grad, atol=2e-5, rtol=2e-5), name)

    def test_forward_matches_untruncated_reference_and_short_reset_is_distinct(self):
        model = StreamingRTRL(seed=107)
        reference = LinearReadoutRNN20(seed=107)
        history = [(i * 7 + i * i) % 10 for i in range(45)]
        with torch.no_grad():
            logits = None
            for digit in history:
                logits = model.advance(digit)
            expected, expected_state = reference.replay(history)
        self.assertTrue(torch.allclose(logits, expected, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(model.h, expected_state.activity, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(model.fast_edges, expected_state.fast_edges, atol=2e-5, rtol=2e-5))
        # Force persistent activity in coordinate 0. A 32-step reset drops the
        # initial 1, while streaming all 45 symbols retains its fast trace.
        with torch.no_grad():
            for p in reference.parameters():
                p.zero_()
            reference.recurrent_weights.copy_(torch.eye(20, dtype=torch.float32) * 0.99)
            reference.input_weights[0, 0] = 0.1
            reference.eta_raw.zero_()
        long_history = [0] + [1] * 44
        _, continuous = reference.replay(long_history)
        _, reset_state = reference.replay(long_history[-32:])
        self.assertGreater(float(continuous.activity[0].detach()), 0.01)
        self.assertEqual(float(reset_state.activity[0].detach()), 0.0)

    def test_optimizer_is_fresh_and_retention_has_no_weight_decay(self):
        model = StreamingRTRL(seed=108)
        self.assertEqual(sum(p.numel() for p in model.network.parameters()), 851)
        groups = model.network.optimizer.param_groups
        params = [p for group in groups for p in group["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.network.parameters()})
        self.assertEqual(groups[0]["weight_decay"], 1e-4)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[1]["params"], [model.network.decay_logit])
        self.assertEqual(len(model.network.optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
