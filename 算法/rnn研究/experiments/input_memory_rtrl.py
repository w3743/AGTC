"""Streaming RTRL with a shared exponential memory per input source."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from streaming_rtrl import StreamingRTRL


class InputMemoryRTRL(StreamingRTRL):
    """Add source-wise input/history memory to the 20-unit streaming core.

    ``scope='all'`` filters the ten external input channels and twenty previous
    hidden activities. ``scope='external'`` filters only the ten input channels.
    Each source has one scalar memory shared by all receiving units.
    """

    def __init__(self, seed: int = 0, learn: bool = True,
                 credit: str = "rtrl", width: int = 20,
                 scope: str = "all", mode: str = "learned") -> None:
        if scope not in ("all", "external"):
            raise ValueError("scope must be 'all' or 'external'")
        if mode not in ("fixed", "learned"):
            raise ValueError("mode must be 'fixed' or 'learned'")
        super().__init__(seed=seed, learn=learn, credit=credit, width=width)
        self.scope, self.mode = scope, mode
        self.memory_width = 30 if scope == "all" else 10
        self.core_size = 641 + 20 * self.memory_width + (self.memory_width if mode == "learned" else 0)
        # Constant initialization consumes no RNG and preserves every legacy
        # network parameter and its initialization sequence.
        self.network.register_parameter(
            "memory_weights", nn.Parameter(torch.zeros(width, self.memory_width)))
        if mode == "learned":
            rho0 = math.log(0.8 / 0.2)
            self.network.register_parameter(
                "input_retention_logit", nn.Parameter(torch.full((self.memory_width,), rho0)))
        else:
            self._fixed_input_retention = torch.full((self.memory_width,), 0.8,
                                                      dtype=self.h.dtype)
        self._make_fresh_optimizer()
        self.reset_state()

    def _make_fresh_optimizer(self) -> None:
        n = self.network
        ordinary = [p for name, p in n.named_parameters()
                    if name not in ("decay_logit", "input_retention_logit")]
        retention = [n.decay_logit]
        if self.mode == "learned":
            retention.append(n.input_retention_logit)
        n.optimizer = torch.optim.Adam(
            [{"params": ordinary}, {"params": retention, "weight_decay": 0.0}],
            lr=0.015, weight_decay=1e-4)
        n.optimizer_updates = 0

    @property
    def input_retention(self) -> Tensor:
        if self.mode == "learned":
            return torch.sigmoid(self.network.input_retention_logit)
        return self._fixed_input_retention

    @property
    def sensitivities(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.eh, self.ef, self.em

    def reset_state(self) -> None:
        super().reset_state()
        p = self.network.input_weights
        self.memory = torch.zeros(self.memory_width, dtype=p.dtype, device=p.device)
        self.em = torch.zeros(self.memory_width, self.core_size, dtype=p.dtype, device=p.device)
        if self.mode == "fixed":
            self._fixed_input_retention = torch.full(
                (self.memory_width,), 0.8, dtype=p.dtype, device=p.device)
        self._refresh_indices()

    def _refresh_indices(self) -> None:
        p, width = self.network.input_weights, self.network.width
        device = p.device
        rows = torch.arange(width, device=device)
        self._rows = rows
        self._input_indices = rows * 10
        self._recurrent_indices = 200 + torch.arange(width * width, device=device).reshape(width, width)
        self._decay_basis = torch.zeros((width, self.core_size), dtype=p.dtype, device=device)
        self._decay_basis[rows, 621 + rows] = 1.0
        self._memory_start = 641
        self._retention_start = self._memory_start + width * self.memory_width
        self._memory_rows = torch.arange(self.memory_width, device=device)
        self._v_indices = (self._memory_start + rows[:, None] * self.memory_width
                           + self._memory_rows[None, :])
        self._retention_cols = self._retention_start + self._memory_rows

    def _advance(self, digit: int) -> Tensor:
        n = self.network
        digit = n._validate_digit(digit)
        dtype, device = n.input_weights.dtype, n.input_weights.device
        with torch.no_grad():
            h0, f0, m0 = self.h, self.fast_edges, self.memory
            eh0, ef0, em0 = self.eh, self.ef, self.em
            eta = n.eta
            decay = n.trace_decay.reshape(-1)
            r = decay[:, None]
            retention = self.input_retention
            effective = n.recurrent_weights + eta * f0
            inp = F.one_hot(torch.tensor(digit, dtype=torch.long, device=device),
                            num_classes=10).to(dtype)
            sources = torch.cat((inp, h0)) if self.scope == "all" else inp
            candidate = torch.tanh(n.input_weights @ inp + effective @ h0
                                   + n.memory_weights @ m0 + n.bias)
            h = candidate
            f = decay[:, None] * f0 + (1.0 - decay[:, None]) * torch.outer(h, h0)

            # Pre-activation sensitivities, including both recurrent paths and
            # the current read from the previous memory state.
            du = torch.zeros_like(eh0) if self.credit == "one_step" else effective @ eh0
            if self.credit != "one_step":
                du = du + eta * torch.einsum("ijp,j->ip", ef0, h0)
                du = du + n.memory_weights @ em0
            du[:, 620] += (0.5 * (1.0 - torch.tanh(n.eta_raw).square())) * (f0 @ h0)
            row = self._rows
            du[row, self._input_indices + digit] += 1.0
            du[row[:, None], self._recurrent_indices] += h0[None, :]
            du[row, 600 + row] += 1.0
            # Direct derivative of V @ m_prev for each receiver/source pair.
            du[row[:, None], self._v_indices] += m0[None, :]
            dh = (1.0 - h.square())[:, None] * du

            dr = decay * (1.0 - decay)
            if self.credit == "one_step":
                ef = torch.zeros_like(ef0)
                em = torch.zeros_like(em0)
            else:
                ef = (r[:, :, None] * ef0
                      + (1.0 - r[:, :, None])
                      * (dh[:, None, :] * h0[None, :, None]
                         + h[:, None, None] * eh0[None, :, :]))
                decay_basis = self._decay_basis * dr[:, None]
                ef += (f0 - torch.outer(h, h0))[:, :, None] * decay_basis[:, None, :]

                # m_new = rho*m_prev + (1-rho)*sources. In all-scope, the
                # recurrent sources are h_prev and carry its RTRL derivative.
                em = retention[:, None] * em0
                if self.scope == "all":
                    em[10:30] += (1.0 - retention[10:30])[:, None] * eh0
                if self.mode == "learned":
                    d_rho = retention * (1.0 - retention)
                    em[self._memory_rows, self._retention_cols] += (
                        d_rho * (m0 - sources))

            m = retention * m0 + (1.0 - retention) * sources
            self.h, self.fast_edges, self.memory = h, f, m
            self.eh, self.ef, self.em = dh, ef, em
            self._logits = n.W_out @ h + n.b_out
            self.steps += 1
            return self._logits

    def learn_target(self, target: int) -> dict[str, Any] | None:
        if type(target) is not int or not 0 <= target < 10:
            raise ValueError("target must be an integer from 0 to 9")
        if self.steps == 0 or not self.learn:
            return None
        n = self.network
        with torch.no_grad():
            probabilities = torch.softmax(n.W_out @ self.h + n.b_out, dim=0)
            g = probabilities.clone()
            g[target] -= 1
            n.W_out.grad = torch.outer(g, self.h)
            n.b_out.grad = g
            core_grad = self.eh.T @ (n.W_out.T @ g)
            grads = {"input_weights": core_grad[:200].reshape_as(n.input_weights),
                     "recurrent_weights": core_grad[200:600].reshape_as(n.recurrent_weights),
                     "bias": core_grad[600:620], "eta_raw": core_grad[620],
                     "decay_logit": core_grad[621:641].reshape_as(n.decay_logit),
                     "memory_weights": core_grad[self._memory_start:self._retention_start].reshape_as(
                         n.memory_weights)}
            if self.mode == "learned":
                grads["input_retention_logit"] = core_grad[
                    self._retention_start:self.core_size]
            for name, parameter in n.named_parameters():
                parameter.grad = grads.get(name, parameter.grad)
            trace_grad = n.decay_logit.grad.detach().clone()
            input_grad = (n.input_retention_logit.grad.detach().clone()
                          if self.mode == "learned" else torch.zeros_like(self.input_retention))
            memory_grad = n.memory_weights.grad.detach().clone()
            loss = -torch.log(probabilities[target].clamp_min(torch.finfo(probabilities.dtype).tiny))
            retention_values = trace_grad.reshape(-1).tolist()
            input_retention_values = input_grad.reshape(-1).tolist()
            grad_norm = torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0, error_if_nonfinite=True)
            n.optimizer.step()
            lr = n.optimizer.param_groups[0]["lr"]
            n.input_weights.copy_(torch.sign(n.input_weights) * torch.clamp(
                n.input_weights.abs() - lr * 0.02, min=0.0))
            n.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            n.optimizer_updates = self.updates
            return {"loss": float(loss.item()), "grad_norm": float(grad_norm.item()),
                    "retention_grad_norm": float(trace_grad.norm().item()),
                    "retention_grad": retention_values,
                    "memory_grad_norm": float(memory_grad.norm().item()),
                    "input_retention_grad_norm": float(input_grad.norm().item()),
                    "input_retention_grad": input_retention_values}

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload.update({"memory": self.memory.clone(), "em": self.em.clone(),
                        "scope": self.scope, "mode": self.mode})
        return payload

    def load_state_payload(self, payload: dict[str, Any]) -> None:
        saved_scope = str(payload.get("scope", "all"))
        saved_mode = str(payload.get("mode", "learned"))
        if (saved_scope, saved_mode) != (self.scope, self.mode):
            raise ValueError("payload scope/mode does not match this instance")
        self.network.load_state_dict(payload["model"])
        self.network.optimizer.load_state_dict(payload["optimizer"])
        for key in ("h", "fast_edges", "eh", "ef", "memory", "em"):
            setattr(self, key, payload[key].to(dtype=self.h.dtype, device=self.h.device))
        expected = (self.core_size, self.memory_width)
        if self.eh.shape != (20, expected[0]) or self.ef.shape != (20, 20, expected[0]):
            raise ValueError("payload RTRL sensitivity dimensions do not match scope/mode")
        if self.em.shape != (expected[1], expected[0]) or self.memory.shape != (expected[1],):
            raise ValueError("payload memory dimensions do not match scope/mode")
        self.steps, self.updates = int(payload["steps"]), int(payload["updates"])
        self.learn, self.credit = bool(payload["learn"]), str(payload["credit"])
        self.network.optimizer_updates = self.updates
        self._refresh_indices()
