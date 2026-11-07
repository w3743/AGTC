"""Block-sparse, differentiable input-memory recurrence."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseInputMemory(nn.Module):
    width = 1024
    indegree = 32
    groups = width // indegree

    def __init__(self):
        super().__init__()
        # Each 32-unit receiving block reads four distinct 8-unit senders
        # from its own block and three other blocks. Offsets include 1 so
        # information can travel around the ring.
        receiver = torch.arange(self.groups)[:, None]
        offset = torch.tensor([0, 1, 3, 9])
        source = ((receiver + offset) % self.groups)[:, :, None] * self.indegree
        source = (source + torch.arange(4)[None, :, None]*8 + torch.arange(8)).flatten(1)
        self.register_buffer('sources', source)
        self.win = nn.Parameter(torch.empty(self.width, 10))
        self.recurrent = nn.Parameter(torch.empty(self.groups, self.indegree, self.indegree))
        self.memory_recurrent = nn.Parameter(torch.zeros_like(self.recurrent))
        self.memory_external = nn.Parameter(torch.zeros(self.width, 10))
        self.bias = nn.Parameter(torch.zeros(self.width))
        self.eta_raw = nn.Parameter(torch.zeros(()))
        self.trace_logit = nn.Parameter(torch.full((self.groups, self.indegree, 1), 1.38629436))
        self.memory_logit = nn.Parameter(torch.full((10 + self.width,), 1.38629436))
        self.head = nn.Linear(self.width, 10)
        nn.init.xavier_uniform_(self.win, gain=.35)
        nn.init.uniform_(self.recurrent, -.35 / self.indegree**.5, .35 / self.indegree**.5)

    @property
    def edge_count(self):
        return self.width * self.indegree

    def initial_state(self, batch):
        return (self.win.new_zeros(batch, self.width),
                self.win.new_zeros(batch, 10+self.width),
                self.win.new_zeros(batch, self.groups, self.indegree, self.indegree))

    def step(self, x, state, rho, retention, eta):
        h0, m0, trace = state
        send = h0[:, self.sources]
        remembered = m0[:, 10:][:, self.sources]
        recurrent = ((self.recurrent + eta*trace) @ send.unsqueeze(-1)).squeeze(-1)
        memory_read = (self.memory_recurrent @ remembered.unsqueeze(-1)).squeeze(-1)
        h = torch.tanh(x @ self.win.T + (recurrent + memory_read).flatten(1)
                       + m0[:, :10] @ self.memory_external.T + self.bias)
        outer = h.reshape(-1, self.groups, self.indegree, 1) * send.unsqueeze(-2)
        trace = retention*trace + (1-retention)*outer
        memory = rho*m0 + (1-rho)*torch.cat((x, h0), dim=-1)
        return h, memory, trace

    def forward(self, digits):
        state = self.initial_state(digits.shape[0])
        rho = self.memory_logit.sigmoid()
        retention = self.trace_logit.sigmoid()
        eta = .5*self.eta_raw.tanh()
        for x in F.one_hot(digits, 10).to(self.win.dtype).transpose(0, 1):
            state = self.step(x, state, rho, retention, eta)
        return self.head(state[0])
