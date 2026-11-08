"""A dense-start RNN with cost-driven, reversible recurrent connections."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call

from edge_adamw import EdgeAdamW


@dataclass(frozen=True)
class Survival:
    cost: float = .43
    competition: float = .15
    rate: float = .18
    dormant_credit: float = .82
    wake: float = .30
    cooldown: float = .30
    traces: tuple[float,...] = ()
    dynamics: float = 0.
    independent_age: bool = False
    conductance_floor: float = .5


class EvolvingRNN(nn.Module):
    def __init__(self, width=512, mode='evolve', config=Survival(), fixed_edges=None, vocab=12,
                 near_identity=False, fixed_mask=None):
        super().__init__()
        if mode not in ('dense','fixed','decay','evolve'):
            raise ValueError(mode)
        self.mode = mode
        self.config = config
        self.tokens = nn.Embedding(vocab,24)
        self.core = nn.RNN(24,width,batch_first=True,nonlinearity='tanh')
        self.head = nn.Linear(width,4)
        with torch.no_grad():
            nn.init.orthogonal_(self.core.weight_hh_l0)
            if near_identity:
                self.core.weight_hh_l0.mul_(.02).diagonal().add_(.98)
            self.core.weight_ih_l0.mul_(.35)
            self.core.bias_ih_l0.zero_()
            self.core.bias_hh_l0.zero_()
        self.register_buffer('life',torch.ones(width,width))
        self.register_buffer('active',torch.ones(width,width,dtype=torch.bool))
        if mode=='fixed':
            if fixed_edges is None or not 0<fixed_edges<=width*width:
                raise ValueError('A fixed sparse control requires a valid matched edge count.')
            if fixed_mask is None:
                rng = torch.Generator().manual_seed(torch.initial_seed()+32027)
                inactive = torch.randperm(width*width,generator=rng)[fixed_edges:]
                self.active.flatten()[inactive] = False
            else:
                if (fixed_mask.shape!=self.active.shape or fixed_mask.dtype!=torch.bool
                        or int(fixed_mask.sum())!=fixed_edges):
                    raise ValueError('The supplied fixed mask must match the declared shape and edge count.')
                self.active.copy_(fixed_mask)
            with torch.no_grad():
                self.core.weight_hh_l0.mul_(self.active).mul_(self.active.float().mean().rsqrt())
        self.probe = None
        self.probe_states = None
        self.probe_outputs = None
        self.noise = torch.Generator().manual_seed(torch.initial_seed()+81031)

    def effective_recurrent(self, cut=None):
        scale = 1
        if self.mode in ('decay','evolve') and self.config.conductance_floor < 1:
            scale = (self.life+self.config.conductance_floor).clamp(0,1)
        live = self.active if cut is None else self.active & ~cut
        return self.core.weight_hh_l0*live*scale

    def forward(self, x, probe=False, cut=None, prefix_index=None):
        if self.mode!='dense' or cut is not None or probe:
            effective = self.effective_recurrent(cut)
            if probe:
                effective.retain_grad()
                self.probe = effective
            states,_ = functional_call(self.core,{'weight_hh_l0':effective},(self.tokens(x),))
        else:
            states,_ = self.core(self.tokens(x))
        logits = self.head(states[:,-1])
        if probe and (self.config.traces or self.config.dynamics):
            self.probe_states = states.detach().transpose(0,1)
        if probe and self.config.traces:
            states.retain_grad()
            self.probe_outputs = states
        return logits if prefix_index is None else (logits,states[:,prefix_index])

    @torch.no_grad()
    def temporal_gradients(self, horizons):
        """Separate recurrent gradients by how far a transition is from the answer."""
        states = self.probe_states
        recurrent = self.probe.detach()
        direct = self.probe_outputs.grad.transpose(0,1)
        adjoint = torch.zeros_like(states[0])
        backward = []
        for state,supervised in zip(reversed(states.unbind()),reversed(direct.unbind()),strict=True):
            delta = (adjoint+supervised)*(1-state.square())
            backward.append(delta)
            adjoint = delta @ recurrent
        delta = torch.stack(list(reversed(backward)))
        earlier = states[:-1].flatten(0,1)
        lag = torch.arange(len(states)-2,-1,-1,device=states.device)
        result = []
        for horizon in horizons:
            weight = (-lag/horizon).exp()
            weight = weight/weight.mean()
            tagged = (delta[1:]*weight[:,None,None]).flatten(0,1)
            result.append(tagged.T @ earlier)
        return result

    @torch.no_grad()
    def perturbation_credit(self):
        """Measure late energy flow after a small hidden-state perturbation."""
        states = self.probe_states
        recurrent = self.probe.detach()
        anchor = min(1,len(states)-2)
        signal = torch.randint(0,2,states[anchor].shape,generator=self.noise).to(states.dtype)
        signal = (signal*2-1)/states.shape[-1]**.5
        late = (len(states)+anchor)//2
        source,sensitivity = [],[]
        for t in range(anchor+1,len(states)):
            derivative = 1-states[t].square()
            if t>=late:
                source.append(signal.square())
                sensitivity.append(derivative.square())
            signal = (signal @ recurrent.T)*derivative
            # Keep a growing perturbation from dominating the credit.
            signal.div_(signal.norm(dim=-1,keepdim=True).clamp_min(1.))
        energy = torch.cat(sensitivity).T @ torch.cat(source)
        return recurrent.square()*energy,float(signal.norm(dim=-1).mean())

    @torch.no_grad()
    def evolve(self, optimizer, sleep_queue=None, sleep_window=1, sleep_filter=None):
        if self.mode not in ('decay','evolve') or self.probe is None or self.probe.grad is None:
            raise RuntimeError('Structural updates require a recurrent gradient probe.')
        cfg = self.config
        grad = self.probe.grad
        gradients = self.temporal_gradients(cfg.traces) if cfg.traces else [grad]
        live_utility = torch.stack([(-self.probe.detach()*g).clamp_min(0) for g in gradients]).mean(0)
        scale = live_utility.sum()/self.active.sum().clamp_min(1)
        live_utility = live_utility/(live_utility+scale.clamp_min(1e-10))
        perturbation = None
        if cfg.dynamics:
            dynamic,perturbation = self.perturbation_credit()
            scale = dynamic.sum()/self.active.sum().clamp_min(1)
            dynamic = dynamic/(dynamic+scale.clamp_min(1e-10))
            live_utility.lerp_(dynamic,cfg.dynamics)
        demand = torch.stack([g.abs() for g in gradients]).mean(0)
        scale = demand.mean().clamp_min(1e-10)
        demand = cfg.dormant_credit*demand/(demand+scale)
        utility = torch.where(self.active,live_utility,demand if self.mode=='evolve' else 0.)
        cost = cfg.cost+cfg.competition*self.active.float().mean(1,keepdim=True)
        self.life.add_(cfg.rate*(utility-cost)).clamp_(-1,1)
        asleep = self.active & (self.life<=0)
        awake = (~self.active) & (self.life>=cfg.wake) if self.mode=='evolve' else torch.zeros_like(asleep)
        eligible = int(asleep.sum())
        if sleep_filter is not None and eligible:
            if cfg.conductance_floor!=1 or sleep_window!=1:
                raise ValueError('Sleep filtering requires immediate full-strength connections.')
            approved = sleep_filter(asleep)
            if approved.shape!=asleep.shape or approved.dtype!=torch.bool or (approved & ~asleep).any():
                raise ValueError('Sleep filter must return a subset of eligible edges.')
            asleep = approved
        if sleep_window > 1:
            if sleep_queue is None or sleep_queue or cfg.conductance_floor != 1:
                raise ValueError('Paced sleep needs an empty queue and full-strength live edges.')
            candidates = torch.where(asleep.flatten())[0]
            order = self.life.flatten()[candidates].argsort(stable=True)
            first, *later = candidates[order].tensor_split(sleep_window)
            sleep_queue.extend(reversed(later))
            asleep.zero_().flatten()[first] = True
        slept,woke = self._change_edges(optimizer,asleep,awake,grad)
        self.probe = None
        self.probe_states = self.probe_outputs = None
        return dict(active=int(self.active.sum()),slept=slept,woke=woke,eligible=eligible,
                    mean_life=float(self.life.mean()),perturbation_gain=perturbation)

    @torch.no_grad()
    def finish_sleep(self, optimizer, indices):
        asleep = torch.zeros_like(self.active)
        asleep.flatten()[indices] = True
        if not self.active.flatten()[indices].all():
            raise RuntimeError('A queued edge is already asleep.')
        slept,woke = self._change_edges(optimizer,asleep,torch.zeros_like(asleep),None)
        return dict(active=int(self.active.sum()),slept=slept,woke=woke)

    def _change_edges(self,optimizer,asleep,awake,grad):
        cfg = self.config
        weight = self.core.weight_hh_l0
        self.active.logical_xor_(asleep | awake)
        self.life[asleep] = -cfg.cooldown
        self.life[awake] = cfg.wake
        weight[asleep] = 0.
        # A tiny step in the descent direction gives each renewed edge a signal.
        if grad is not None:
            weight[awake] = -.01*grad[awake].sign()
        for key in ('exp_avg','exp_avg_sq'):
            optimizer.state[weight][key][asleep | awake] = 0.
        if isinstance(optimizer,EdgeAdamW):
            optimizer.reset_edges(asleep,awake)
        return int(asleep.sum()),int(awake.sum())
