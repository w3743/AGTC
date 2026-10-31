"""Online RTRL with a shared activity-retention coefficient per unit."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from streaming_rtrl import StreamingRTRL


class ActivityRetentionRTRL(StreamingRTRL):
    """Streaming RTRL where hidden activity blends its predecessor and candidate.

    The persistent fast-edge state and its trace retention follow the existing
    ``StreamingRTRL`` rule. Only the activity update gains the coefficient
    ``a``. The learned variant appends its 20 coefficients to the existing
    641-coordinate core without changing any legacy initialization.
    """

    def __init__(self, seed: int = 0, learn: bool = True,
                 credit: str = "rtrl", width: int = 20,
                 mode: str = "fixed") -> None:
        if mode not in ("fixed", "learned"):
            raise ValueError("mode must be 'fixed' or 'learned'")
        super().__init__(seed=seed, learn=learn, credit=credit, width=width)
        self.mode = mode
        if mode == "learned":
            # Constant initialization consumes no RNG and leaves all legacy
            # parameter values untouched.
            self.network.register_parameter("activity_logit", nn.Parameter(torch.zeros(width)))
            self.core_size = 661
            self._make_fresh_optimizer()
        else:
            self.core_size = 641
            self._fixed_activity = torch.full((width,), 0.5, dtype=self.h.dtype)
        self.reset_state()

    def _make_fresh_optimizer(self) -> None:
        n = self.network
        ordinary = [p for name, p in n.named_parameters()
                    if name not in ("decay_logit", "activity_logit")]
        n.optimizer = torch.optim.Adam(
            [{"params": ordinary},
             {"params": [n.decay_logit, n.activity_logit], "weight_decay": 0.0}],
            lr=0.015, weight_decay=1e-4)
        n.optimizer_updates = 0

    @property
    def activity_retention(self) -> Tensor:
        if self.mode == "learned":
            return torch.sigmoid(self.network.activity_logit)
        return self._fixed_activity

    def reset_state(self) -> None:
        super().reset_state()
        if self.mode == "fixed":
            self._fixed_activity = torch.full(
                (self.network.width,), 0.5,
                dtype=self.network.input_weights.dtype,
                device=self.network.input_weights.device)

    def _advance(self, digit: int) -> Tensor:
        n = self.network
        digit = n._validate_digit(digit)
        dtype, device = n.input_weights.dtype, n.input_weights.device
        with torch.no_grad():
            h0, f0, eh0, ef0 = self.h, self.fast_edges, self.eh, self.ef
            eta = n.eta
            decay = n.trace_decay.reshape(-1)
            r = decay[:, None]
            a = self.activity_retention
            effective = n.recurrent_weights + eta * f0
            inp = torch.nn.functional.one_hot(
                torch.tensor(digit, dtype=torch.long, device=device), num_classes=10).to(dtype)
            candidate = torch.tanh(n.input_weights @ inp + effective @ h0 + n.bias)
            h = a * h0 + (1.0 - a) * candidate
            f = decay[:, None] * f0 + (1.0 - decay[:, None]) * torch.outer(h, h0)

            du = torch.zeros_like(eh0) if self.credit == "one_step" else effective @ eh0
            if self.credit != "one_step":
                du = du + eta * torch.einsum("ijp,j->ip", ef0, h0)
            du[:, 620] += (0.5 * (1.0 - torch.tanh(n.eta_raw).square())) * (f0 @ h0)
            row = self._rows
            du[row, self._input_indices + digit] += 1.0
            du[row[:, None], self._recurrent_indices] += h0[None, :]
            du[row, 600 + row] += 1.0
            dcandidate = (1.0 - candidate.square())[:, None] * du

            # dh = a*dh_prev + (1-a)*dcandidate + (h_prev-candidate)*da.
            if self.credit == "one_step":
                dh = (1.0 - a)[:, None] * dcandidate
            else:
                dh = a[:, None] * eh0 + (1.0 - a)[:, None] * dcandidate
            if self.mode == "learned":
                da = a * (1.0 - a)
                dh[row, 641 + row] += (h0 - candidate) * da

            dr = decay * (1.0 - decay)
            if self.credit == "one_step":
                ef = torch.zeros_like(ef0)
            else:
                ef = (r[:, :, None] * ef0
                      + (1.0 - r[:, :, None])
                      * (dh[:, None, :] * h0[None, :, None]
                         + h[:, None, None] * eh0[None, :, :]))
                decay_basis = self._decay_basis * dr[:, None]
                ef += (f0 - torch.outer(h, h0))[:, :, None] * decay_basis[:, None, :]
            self.h, self.fast_edges, self.eh, self.ef = h, f, dh, ef
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
                     "decay_logit": core_grad[621:641].reshape_as(n.decay_logit)}
            if self.mode == "learned":
                grads["activity_logit"] = core_grad[641:661]
            for name, parameter in n.named_parameters():
                parameter.grad = grads.get(name, parameter.grad)
            retention_grad = n.decay_logit.grad.detach().clone()
            activity_grad = (n.activity_logit.grad.detach().clone()
                             if self.mode == "learned" else torch.zeros_like(self.activity_retention))
            loss = -torch.log(probabilities[target].clamp_min(torch.finfo(probabilities.dtype).tiny))
            grad_norm = torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0, error_if_nonfinite=True)
            n.optimizer.step()
            lr = n.optimizer.param_groups[0]["lr"]
            n.input_weights.copy_(torch.sign(n.input_weights) * torch.clamp(
                n.input_weights.abs() - lr * 0.02, min=0.0))
            n.optimizer.zero_grad(set_to_none=True)
            self.updates += 1
            n.optimizer_updates = self.updates
            values = activity_grad.reshape(-1).tolist()
            return {"loss": float(loss.item()), "grad_norm": float(grad_norm.item()),
                    "retention_grad_norm": float(retention_grad.norm().item()),
                    "retention_grad": retention_grad.reshape(-1).tolist(),
                    "activity_grad_norm": float(activity_grad.norm().item()),
                    "activity_grad": values}

    def state_payload(self) -> dict[str, Any]:
        payload = super().state_payload()
        payload["mode"] = self.mode
        return payload

    def load_state_payload(self, payload: dict[str, Any]) -> None:
        saved_mode = str(payload.get("mode", "fixed"))
        if saved_mode != self.mode:
            raise ValueError(f"payload mode {saved_mode!r} does not match instance mode {self.mode!r}")
        self.network.load_state_dict(payload["model"])
        self.network.optimizer.load_state_dict(payload["optimizer"])
        width = self.network.width
        expected_core = 661 if self.mode == "learned" else 641
        for key in ("h", "fast_edges", "eh", "ef"):
            value = payload[key].to(dtype=self.h.dtype, device=self.h.device)
            setattr(self, key, value)
        if self.eh.shape != (width, expected_core) or self.ef.shape != (width, width, expected_core):
            raise ValueError("payload sensitivity dimensions do not match activity-retention mode")
        self.core_size = expected_core
        self.steps, self.updates = int(payload["steps"]), int(payload["updates"])
        self.learn, self.credit = bool(payload["learn"]), str(payload["credit"])
        self.network.optimizer_updates = self.updates
        if self.mode == "fixed":
            self._fixed_activity = torch.full_like(self.h, 0.5)
        # Rebuild parameter index/cache tensors for the restored device.
        self.reset_state_indices()

    def reset_state_indices(self) -> None:
        """Refresh static index caches without clearing restored dynamic state."""
        p, width = self.network.input_weights, self.network.width
        rows = torch.arange(width, device=p.device)
        self._rows = rows
        self._input_indices = rows * 10
        self._recurrent_indices = 200 + torch.arange(width * width, device=p.device).reshape(width, width)
        self._decay_basis = torch.zeros((width, self.core_size), dtype=p.dtype, device=p.device)
        self._decay_basis[rows, 621 + rows] = 1.0
        self._logits = self.network.b_out.detach().clone()
