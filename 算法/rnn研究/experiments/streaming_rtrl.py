"""Constant-memory online RTRL for the linear-readout plastic RNN."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from linear_readout import LinearReadoutRNN


class StreamingRTRL:
    """Exact fixed-parameter RTRL sensitivities with bounded state.

    After an optimizer update, the carried sensitivities follow conventional
    online RTRL; gradients do not differentiate through the optimizer history.
    """

    core_size = 641

    def __init__(self, seed: int = 0, learn: bool = True,
                 credit: str = "rtrl", width: int = 20) -> None:
        if credit not in ("rtrl", "one_step"):
            raise ValueError("credit must be 'rtrl' or 'one_step'")
        if width != 20:
            raise ValueError("streaming core currently requires width=20")
        self.network = LinearReadoutRNN(seed=seed, width=width)
        self.learn = bool(learn)
        self.credit = credit
        self.h = torch.zeros(width, dtype=self.network.input_weights.dtype)
        self.fast_edges = torch.zeros(width, width, dtype=self.h.dtype)
        self.eh = torch.zeros(width, self.core_size, dtype=self.h.dtype)
        self.ef = torch.zeros(width, width, self.core_size, dtype=self.h.dtype)
        self.steps = 0
        self.updates = 0
        self._logits = self.network.b_out.detach().clone()
        rows = torch.arange(width)
        self._rows = rows
        self._input_indices = rows * 10
        self._recurrent_indices = 200 + torch.arange(width * width).reshape(width, width)
        self._decay_basis = torch.zeros((width, self.core_size), dtype=self.h.dtype)
        self._decay_basis[rows, 621 + rows] = 1.0

    @property
    def sensitivities(self) -> tuple[Tensor, Tensor]:
        return self.eh, self.ef

    def reset_state(self) -> None:
        """Clear dynamic state and rebuild cached indices for current parameter dtype/device."""
        p = self.network.input_weights
        width = self.network.width
        self.h = torch.zeros(width, dtype=p.dtype, device=p.device)
        self.fast_edges = torch.zeros(width, width, dtype=p.dtype, device=p.device)
        self.eh = torch.zeros(width, self.core_size, dtype=p.dtype, device=p.device)
        self.ef = torch.zeros(width, width, self.core_size, dtype=p.dtype, device=p.device)
        self.steps = self.updates = 0
        self._logits = self.network.b_out.detach().clone()
        rows = torch.arange(width, device=p.device)
        self._rows = rows
        self._input_indices = rows * 10
        self._recurrent_indices = 200 + torch.arange(width * width, device=p.device).reshape(width, width)
        self._decay_basis = torch.zeros((width, self.core_size), dtype=p.dtype, device=p.device)
        self._decay_basis[rows, 621 + rows] = 1.0

    def _advance(self, digit: int) -> Tensor:
        n = self.network
        digit = n._validate_digit(digit)
        dtype = n.input_weights.dtype
        with torch.no_grad():
            h0, f0 = self.h, self.fast_edges
            eh0, ef0 = self.eh, self.ef
            eta = n.eta
            decay = n.trace_decay.reshape(-1)
            r = decay[:, None]
            effective = n.recurrent_weights + eta * f0
            inp = F.one_hot(torch.tensor(digit, dtype=torch.long), num_classes=10).to(dtype)
            h = torch.tanh(n.input_weights @ inp + effective @ h0 + n.bias)
            f = decay[:, None] * f0 + (1.0 - decay[:, None]) * torch.outer(h, h0)

            # Differentiate pre-activation with respect to the 621 non-decay core parameters.
            if self.credit == "one_step":
                # Diagnostic control: carry numeric state but deliberately
                # remove all historical credit; no zero-valued RTRL products.
                du = torch.zeros_like(eh0)
            else:
                du = effective @ eh0
                du = du + eta * torch.einsum("ijp,j->ip", ef0, h0)
            # d(eta * F h)/d eta_raw
            du[:, 620] += (0.5 * (1.0 - torch.tanh(n.eta_raw).square())) * (f0 @ h0)
            row = self._rows
            du[row, self._input_indices + digit] += 1.0
            du[row[:, None], self._recurrent_indices] += h0[None, :]
            du[row, 600 + row] += 1.0
            dh = (1.0 - h.square())[:, None] * du
            dr = decay * (1.0 - decay)
            if self.credit == "one_step":
                ef = torch.zeros_like(ef0)
            else:
                ef = (r[:, :, None] * ef0
                      + (1.0 - r[:, :, None]) * (dh[:, None, :] * h0[None, :, None]
                                                   + h[:, None, None] * eh0[None, :, :]))
                decay_basis = self._decay_basis * dr[:, None]
                ef += (f0 - torch.outer(h, h0))[:, :, None] * decay_basis[:, None, :]
            self.h, self.fast_edges, self.eh, self.ef = h, f, dh, ef
            self._logits = n.W_out @ h + n.b_out
            self.steps += 1
            return self._logits

    def advance(self, digit: int) -> Tensor:
        return self._advance(digit)

    def predict(self) -> Tensor:
        with torch.no_grad():
            if self.steps == 0:
                return torch.full((10,), 0.1, dtype=self.h.dtype)
            return torch.softmax(self.network.W_out @ self.h + self.network.b_out, dim=0)

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
                     "decay_logit": core_grad[621:641].reshape_as(n.decay_logit)}
            for name, parameter in n.named_parameters():
                parameter.grad = grads.get(name, parameter.grad)
            retention_grad = n.decay_logit.grad.detach().clone()
            retention_grad_norm = retention_grad.norm()
            loss = -torch.log(probabilities[target].clamp_min(torch.finfo(probabilities.dtype).tiny))
            retention_values = retention_grad.reshape(-1).tolist()
            grad_norm = torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0, error_if_nonfinite=True)
            n.optimizer.step()
            lr = n.optimizer.param_groups[0]["lr"]
            n.input_weights.copy_(torch.sign(n.input_weights) * torch.clamp(
                n.input_weights.abs() - lr * 0.02, min=0.0))
            n.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            n.optimizer_updates = self.updates
            return {"loss": float(loss.item()), "grad_norm": float(grad_norm.item()),
                    "retention_grad_norm": float(retention_grad_norm.item()),
                    "retention_grad": retention_values}

    def observe(self, digit: int) -> Tensor:
        if self.steps:
            self.learn_target(digit)
        self.advance(digit)
        return self.predict()

    def state_payload(self) -> dict[str, Any]:
        return {"model": self.network.state_dict(), "optimizer": self.network.optimizer.state_dict(),
                "h": self.h.clone(), "fast_edges": self.fast_edges.clone(),
                "eh": self.eh.clone(), "ef": self.ef.clone(), "steps": self.steps,
                "updates": self.updates, "learn": self.learn, "credit": self.credit,
                "optimizer_updates": self.network.optimizer_updates}

    def load_state_payload(self, payload: dict[str, Any]) -> None:
        self.network.load_state_dict(payload["model"])
        self.network.optimizer.load_state_dict(payload["optimizer"])
        for key in ("h", "fast_edges", "eh", "ef"):
            setattr(self, key, payload[key].to(dtype=self.h.dtype))
        self.steps, self.updates = int(payload["steps"]), int(payload["updates"])
        self.learn, self.credit = bool(payload["learn"]), str(payload["credit"])
        self.network.optimizer_updates = self.updates
