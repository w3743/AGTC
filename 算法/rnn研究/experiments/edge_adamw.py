"""AdamW with a fresh bias-correction clock for renewed recurrent edges."""

from math import log

import torch


class EdgeAdamW(torch.optim.AdamW):
    def __init__(self, parameters, recurrent, *, lr):
        super().__init__(parameters,lr=lr)
        self.recurrent = recurrent

    @torch.no_grad()
    def reset_edges(self, asleep, awake):
        state = self.state[self.recurrent]
        age = state.get('edge_age')
        if age is None:
            if not awake.any():
                return
            age = torch.full_like(self.recurrent,-1.)
            state['edge_age'] = age
        age[asleep] = -1.
        age[awake] = 0.

    @torch.no_grad()
    def step(self, closure=None):
        recurrent = self.recurrent
        state = self.state.get(recurrent,{})
        age = state.get('edge_age')
        if age is None:
            return super().step(closure)
        # Compute the coordinates once. Repeated boolean indexing rescans the
        # whole recurrent matrix even when only a few edges were renewed.
        renewed = torch.where(age>=0)
        if not renewed[0].numel():
            return super().step(closure)
        before = recurrent[renewed]
        result = super().step(closure)
        if recurrent.grad is None:
            return result
        age[renewed] += 1
        # Keep the stock update for all other edges. Its moments are already
        # correct here; only the bias corrections need an edge-local clock.
        group = self.param_groups[0]
        lr,eps,decay = (group[key] for key in ('lr','eps','weight_decay'))
        beta1,beta2 = group['betas']
        steps = age[renewed]
        m = state['exp_avg'][renewed]/(-torch.expm1(log(beta1)*steps))
        v = state['exp_avg_sq'][renewed]/(-torch.expm1(log(beta2)*steps))
        recurrent[renewed] = before*(1-lr*decay)-lr*m/(v.sqrt()+eps)
        return result


def make_optimizer(model, independent_age):
    if independent_age:
        return EdgeAdamW(model.parameters(),model.core.weight_hh_l0,lr=.001)
    return torch.optim.AdamW(model.parameters(),lr=.001)
