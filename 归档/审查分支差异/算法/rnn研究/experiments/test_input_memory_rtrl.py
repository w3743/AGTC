"""Independent checks for input-conditioned memory RTRL."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from input_memory_rtrl import InputMemoryRTRL
from linear_readout import LinearReadoutRNN

HISTORY = [0, 1, 1, 0, 1]


class InputMemoryRTRLTests(unittest.TestCase):
    def test_seeded_legacy_prefix_initialization_and_parameter_layout(self):
        for mode, expected_core, expected_total in (("fixed", 1241, 1451), ("learned", 1271, 1481)):
            model = InputMemoryRTRL(seed=201, mode=mode)
            self.assertEqual(model.core_size, expected_core)
            self.assertEqual(sum(p.numel() for p in model.network.parameters()), expected_total)
            old = LinearReadoutRNN(seed=201)
            for name in ("input_weights", "recurrent_weights", "bias", "eta_raw", "decay_logit", "W_out", "b_out"):
                self.assertTrue(torch.equal(getattr(model.network, name), getattr(old, name)), name)
            self.assertEqual(model.network.memory_weights.shape, (20, 30))
            self.assertTrue(torch.equal(model.network.memory_weights, torch.zeros_like(model.network.memory_weights)))
            self.assertEqual(model.input_retention.shape, (30,))
            self.assertTrue(torch.allclose(model.input_retention, torch.full((30,), .8)))

    def test_zero_memory_weights_reproduce_baseline_forward_exactly(self):
        model = InputMemoryRTRL(seed=202)
        old = LinearReadoutRNN(seed=202)
        for digit in HISTORY * 3:
            model.advance(digit)
        with torch.no_grad():
            expected_logits, expected_state = old.replay(HISTORY * 3)
        self.assertTrue(torch.equal(model.h, expected_state.activity))
        self.assertTrue(torch.equal(model.fast_edges, expected_state.fast_edges))
        self.assertTrue(torch.equal(model._logits, expected_logits))

    @staticmethod
    def _reference(model, sequence):
        n = model.network
        h = torch.zeros(20, dtype=n.input_weights.dtype)
        f = torch.zeros(20, 20, dtype=h.dtype)
        m = torch.zeros(30, dtype=h.dtype)
        for digit in sequence:
            x = F.one_hot(torch.tensor(digit), 10).to(h.dtype)
            # The memory read is from m_prev; its write occurs after h_new.
            h_new = torch.tanh(n.input_weights @ x + (n.recurrent_weights + n.eta * f) @ h + n.memory_weights @ m + n.bias)
            decay = n.trace_decay
            f = decay * f + (1 - decay) * torch.outer(h_new, h)
            retention = model.input_retention
            m = retention * m + (1 - retention) * torch.cat((x, h))
            h = h_new
        return n.W_out @ h + n.b_out, h, f, m

    def test_projected_full_autograd_sensitivities_match_rtrl(self):
        model = InputMemoryRTRL(seed=203, mode="learned")
        model.network.double()
        model.reset_state()
        with torch.no_grad():
            model.network.eta_raw.fill_(.3)
            model.network.memory_weights.normal_(0, .04)
        for digit in HISTORY:
            model.advance(digit)
        n = model.network
        core = [n.input_weights, n.recurrent_weights, n.bias, n.eta_raw,
                n.decay_logit, n.memory_weights, n.input_retention_logit]
        gen = torch.Generator().manual_seed(204)
        for _ in range(3):
            vh = torch.randn(20, generator=gen, dtype=torch.float64)
            vf = torch.randn(20, 20, generator=gen, dtype=torch.float64)
            vm = torch.randn(30, generator=gen, dtype=torch.float64)
            _, h, f, m = self._reference(model, HISTORY)
            expected = torch.autograd.grad(h @ vh + (f * vf).sum() + m @ vm, core)
            expected = torch.cat([g.reshape(-1) for g in expected])
            actual = model.eh.T @ vh + model.ef.reshape(400, -1).T @ vf.reshape(-1) + model.em.T @ vm
            self.assertTrue(torch.allclose(actual, expected, atol=3e-8, rtol=3e-7), float((actual-expected).abs().max()))

    def test_online_gradient_matches_unrolled_autograd_after_clipping(self):
        model = InputMemoryRTRL(seed=205, mode="learned")
        ref = InputMemoryRTRL(seed=205, mode="learned")
        for m in (model, ref):
            with torch.no_grad():
                m.network.eta_raw.fill_(.3)
        with torch.no_grad():
            model.network.memory_weights.normal_(0, .04)
            ref.network.memory_weights.copy_(model.network.memory_weights)
        for digit in HISTORY:
            model.advance(digit)
        logits, *_ = self._reference(ref, HISTORY)
        F.cross_entropy(logits[None, :], torch.tensor([2])).backward()
        torch.nn.utils.clip_grad_norm_(ref.network.parameters(), 1.0)
        expected = {name: p.grad.detach().clone() for name, p in ref.network.named_parameters()}
        captured = {}
        opt = model.network.optimizer
        original = opt.step
        def capture(*args, **kwargs):
            captured.update({name: p.grad.detach().clone() for name, p in model.network.named_parameters()})
            return original(*args, **kwargs)
        opt.step = capture
        model.learn_target(2)
        for name, grad in expected.items():
            self.assertTrue(torch.allclose(captured[name], grad, atol=3e-5, rtol=3e-5), name)

    def test_optimizer_layout_fresh_state_checkpoint_and_bounded_stream(self):
        fixed = InputMemoryRTRL(seed=206, mode="fixed")
        learned = InputMemoryRTRL(seed=206, mode="learned")
        self.assertEqual(sum(p.numel() for p in LinearReadoutRNN(seed=206).parameters()), 851)
        for model, size in ((fixed, 1241), (learned, 1271)):
            n = model.network
            groups = n.optimizer.param_groups
            params = [p for group in groups for p in group["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params}, {id(p) for p in n.parameters()})
            self.assertEqual(groups[0]["weight_decay"], 1e-4)
            self.assertEqual(groups[1]["weight_decay"], 0.0)
            self.assertEqual(len(n.optimizer.state), 0)
            self.assertIs(n.optimizer.param_groups[0]["params"][-1], n.memory_weights)
            self.assertEqual(model.core_size, size)
        for i in range(600):
            fixed.advance(i % 10)
        self.assertEqual(fixed.steps, 600)
        self.assertEqual(fixed.updates, 0)
        for value in (fixed.h, fixed.fast_edges, fixed.memory, fixed.eh, fixed.ef, fixed.em):
            self.assertTrue(torch.isfinite(value).all())
        self.assertLess(sum(v.numel() for v in (fixed.h, fixed.fast_edges, fixed.memory, fixed.eh, fixed.ef, fixed.em)), 600_000)
        source = InputMemoryRTRL(seed=207, mode="learned")
        for d in HISTORY:
            source.observe(d)
        payload = source.state_payload()
        restored = InputMemoryRTRL(seed=999, mode="learned")
        restored.load_state_payload(payload)
        for key in ("h", "fast_edges", "memory", "em"):
            self.assertTrue(torch.equal(getattr(source, key), getattr(restored, key)))
        self.assertEqual(source.steps, restored.steps)
        self.assertEqual(source.updates, restored.updates)
        self.assertEqual(source.network.optimizer.state_dict()["state"], restored.network.optimizer.state_dict()["state"])

    def test_memory_write_uses_previous_activity_and_rho_can_learn_after_v(self):
        model = InputMemoryRTRL(seed=208, mode="learned")
        before = model.memory.clone()
        model.advance(4)
        expected_write = torch.cat((F.one_hot(torch.tensor(4), 10).float(), torch.zeros(20)))
        expected = model.input_retention.detach() * before + (1-model.input_retention.detach()) * expected_write
        self.assertTrue(torch.allclose(model.memory, expected))
        # With V=0 the rho path has no first-step effect on h; after a nonzero
        # learned V has been installed, a multi-step target has rho credit.
        self.assertTrue(torch.equal(model.eh[:, -30:], torch.zeros_like(model.eh[:, -30:])))
        with torch.no_grad():
            model.network.memory_weights.normal_(0, .1)
        model.reset_state()
        for digit in HISTORY:
            model.advance(digit)
        self.assertGreater(float(model.em[:, -30:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
