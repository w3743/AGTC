"""A query-conditioned graph over historical token events."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ReadHop(nn.Module):
    def __init__(self, width=64, heads=4, degree=4):
        super().__init__()
        self.heads, self.degree = heads, degree
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.merge = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.ff_norm = nn.LayerNorm(width)

    def forward(self, query, nodes, mode, forced=None):
        batch, length, width = nodes.shape
        q = self.query(query).reshape(batch, self.heads, -1)
        k = self.key(nodes).reshape(batch, length, self.heads, -1).transpose(1, 2)
        v = self.value(nodes).reshape(batch, length, self.heads, -1).transpose(1, 2)
        scores = (q.unsqueeze(-2) @ k.transpose(-1, -2)).squeeze(-2) / q.shape[-1]**.5
        selected = None
        if mode == 'dense':
            weights = scores.softmax(-1)
        else:
            degree = min(self.degree, length)
            if forced is not None:
                selected = forced
            elif mode == 'adaptive':
                selected = scores.topk(degree, dim=-1).indices
            else:
                anchors = torch.linspace(0, length-1, degree, device=nodes.device).long()
                shifts = torch.arange(self.heads, device=nodes.device)*length//self.heads
                selected = ((anchors + shifts[:, None]) % length).expand(batch, -1, -1)
            sparse = scores.gather(-1, selected).softmax(-1)
            weights = torch.zeros_like(scores).scatter(-1, selected, sparse)
            if self.training and mode == 'adaptive' and forced is None:
                # Sparse graph in the forward pass; a differentiable soft score
                # provides a training signal to routes that were not selected.
                soft = scores.softmax(-1)
                weights = weights.detach() + (soft - soft.detach())
        read = (weights.unsqueeze(-2) @ v).reshape(batch, width)
        query = self.norm(query + self.merge(read))
        return self.ff_norm(query + self.ff(query)), selected


class AdaptiveMemoryGraph(nn.Module):
    """Two sparse reads with identical-width fixed and dense controls.

    Nodes receive a shared encoding of the current and previous token. No
    task-specific markers, relevance masks or labels enter the encoder.
    """

    def __init__(self, mode='adaptive', width=64):
        super().__init__()
        if mode not in ('adaptive', 'fixed', 'dense'):
            raise ValueError(f'Unknown graph mode {mode}')
        self.mode = mode
        self.token = nn.Embedding(11, width)
        self.context = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.hops = nn.ModuleList(ReadHop(width) for _ in range(2))
        self.output = nn.Linear(width, 10)

    def forward(self, digits, *, forced=None, return_edges=False):
        history = digits[:, :-1]
        prev = F.pad(history[:, :-1], (1,0), value=10)
        nodes = self.context(torch.cat((self.token(prev), self.token(history)), dim=-1))
        query = self.token(digits[:, -1])
        edges = []
        for i, hop in enumerate(self.hops):
            query, picked = hop(query, nodes, self.mode, None if forced is None else forced[i])
            edges.append(picked)
        logits = self.output(query)
        return (logits, edges) if return_edges else logits
