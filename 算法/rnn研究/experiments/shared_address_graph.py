"""Full-history diagnostic for query-dependent sparse edges."""

import torch
from torch import nn
from torch.nn import functional as F


class SharedAddressGraph(nn.Module):
    """Give one attention head a trainable address shared with history tokens.

    Aligned and rotated initializations differ only in the query projection.
    Both can learn the projection, its strength and all four content scores.
    """

    def __init__(self, aligned=True):
        super().__init__()
        width = 112
        address_width = 64
        self.token = nn.Embedding(64, address_width)
        self.address = nn.Linear(address_width, address_width, bias=False)
        self.address_gain = nn.Parameter(torch.tensor(8.0))
        self.context = nn.Sequential(nn.Linear(3 * address_width, width), nn.GELU(), nn.LayerNorm(width))
        self.query = nn.Linear(2 * address_width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4)
        )
        with torch.no_grad():
            self.token.weight.copy_(torch.eye(address_width))
            if aligned:
                self.address.weight.copy_(torch.eye(address_width))
            else:
                rng = torch.Generator().manual_seed(67123)
                order = torch.randperm(address_width, generator=rng)
                self.address.weight.copy_(torch.eye(address_width)[order])

    def encode_context(self, two_back, previous, current):
        return self.context(torch.cat((two_back, previous, current), dim=-1))

    def read(self, x, forced=None):
        tokens = self.token(x)
        current = tokens[:, :-2]
        previous = F.pad(current[:, :-1], (0, 0, 1, 0))
        two_back = F.pad(current[:, :-2], (0, 0, 2, 0))
        nodes = self.encode_context(two_back, previous, current)
        query = self.query(tokens[:, -2:].flatten(1))
        head_width = query.shape[-1] // 4
        keys = self.key(nodes).unflatten(-1, (4, head_width)).transpose(1, 2)
        scores = (keys * query.unflatten(-1, (4, head_width)).unsqueeze(2)).sum(-1) / head_width**0.5
        address = F.normalize(self.address(tokens[:, -1]), dim=-1)
        similarity = F.normalize(previous, dim=-1) @ address.unsqueeze(-1)
        scores = scores + F.pad(self.address_gain * similarity.transpose(1, 2), (0, 0, 0, 3))
        edges = scores.topk(min(4, nodes.shape[1]), dim=-1).indices if forced is None else forced[0]

        # At inference, only selected node/head values are projected. During
        # training a dense surrogate gives the unselected scores gradients.
        heads = torch.arange(4, device=x.device)[None, :, None]
        batches = torch.arange(x.shape[0], device=x.device)[:, None, None]
        chosen = nodes[batches, edges]
        weights = self.value.weight.view(4, head_width, -1)[heads]
        selected_values = (weights @ chosen.unsqueeze(-1)).squeeze(-1)
        hard = (scores.gather(-1, edges).softmax(-1).unsqueeze(-1) * selected_values).sum(2)
        if self.training:
            values = self.value(nodes).unflatten(-1, (4, head_width)).transpose(1, 2).detach()
            soft = (scores.softmax(-1).unsqueeze(-1) * values).sum(2)
            hard = hard + (soft - soft.detach())
        return query, hard.flatten(1), edges

    def forward(self, x, return_edges=False, forced=None):
        query, history, edges = self.read(x, forced)
        logits = self.head(torch.cat((query, history), dim=-1))
        return (logits, [edges]) if return_edges else logits
