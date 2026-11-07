"""Instruction-conditioned interaction with the output of the shared-address graph."""

import torch
from torch.nn import functional as F

from shared_address_graph import SharedAddressGraph


class ConditionalReadout(SharedAddressGraph):
    def __init__(self, mode, neutral_routing=False, binding=False, relation_bank=False):
        super().__init__()
        self.mode = mode
        self.neutral_routing = neutral_routing
        self.binding = binding
        self.relation_bank = relation_bank

    def encode_context(self, two_back, previous, current):
        if not self.binding:
            return super().encode_context(two_back, previous, current)
        linear, activation, normalize = self.context
        older, left, right = linear.weight.split(current.shape[-1], dim=1)
        address = F.linear(previous, left)
        value = F.linear(current, right)
        context = F.linear(two_back, older, linear.bias) + address + value + 8*address*value
        return normalize(activation(context))

    def forward(self, x, return_edges=False, forced=None, readout_query=None, evidence='all'):
        query, history, edges = self.composition_inputs(x, forced, evidence)
        if readout_query is not None:
            query = self.query(self.token(readout_query).flatten(1))
        logits = self.combine(query, history)
        return (logits, [edges]) if return_edges else logits

    def composition_inputs(self, x, forced=None, evidence='all'):
        if self.neutral_routing:
            # The boundary token queries history. The actual suffix conditions
            # the combination of the retrieved evidence.
            routing_input = torch.cat((x[:, :-1], x[:, -2:-1]), dim=1)
            _, history, edges = self.read(routing_input, forced)
            query = self.query(self.token(x[:, -2:]).flatten(1))
        else:
            query, history, edges = self.read(x, forced)
        if self.relation_bank:
            history = .25*history if evidence!='relations' else torch.zeros_like(history)
            if evidence!='graph':
                history = history + self.bind_pairs(x[:, :-2])
        return query, history, edges

    def bind_pairs(self, prefix):
        """Pool observed successor distributions by preceding symbol identity."""
        size = self.token.num_embeddings
        adjacency = prefix[:, :-1]*size+prefix[:, 1:]
        counts = self.value.weight.new_zeros((len(prefix),size*size))
        counts.scatter_add_(1,adjacency,torch.ones_like(adjacency,dtype=counts.dtype))
        counts = counts.unflatten(1,(size,size))
        distribution = counts/counts.sum(-1,keepdim=True).clamp_min(1)
        values = distribution @ self.value.weight[:4,:size].T
        projection = torch.cat((self.context[0].weight,self.query.weight[:,:size]),dim=1)
        relations = F.linear(values.flatten(1),projection)
        return F.layer_norm(relations,relations.shape[-1:])

    def combine(self, query, history):
        if self.mode == 'shared':
            return self.head(torch.cat((query, history), dim=-1))
        width = query.shape[-1]
        linear = self.head[0]
        instruction = F.linear(query, linear.weight[:, :width], linear.bias)
        evidence = F.linear(history, linear.weight[:, width:])
        gate = instruction if self.mode == 'conditioned' else evidence
        hidden = instruction + evidence * (1 + gate.tanh())
        for layer in list(self.head)[1:]:
            hidden = layer(hidden)
        return hidden
