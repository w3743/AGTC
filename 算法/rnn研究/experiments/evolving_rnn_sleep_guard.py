"""Test groups of eligible edges before committing a topology change."""

import torch
from torch.nn import functional as F

from evolving_rnn_eval import INTERVAL, TRAIN_GAPS, questions


GROUPS = 8
PREFIX_DIVISORS = (2,4,8,16)
CALIBRATION_SIZE = 12
TOLERANCE = (.006,.012)  # Recall and lookup, measured as average answer cross-entropy.


class TrainingCalibration:
    """Retain only the most recent weight-training batch per task and delay."""

    def __init__(self,groups_per_cell=None):
        self.batches = {}
        self.groups_per_cell = groups_per_cell

    def add(self,lookup,x,y):
        gap = x.shape[-1]-(11 if lookup else 3)
        group = 8 if lookup else 2
        self.batches[lookup,gap] = (x.reshape(-1,group,x.shape[-1]),y.reshape(-1,group))

    def data(self,rotation=0):
        data = [[self.batches[lookup,gap] for gap in TRAIN_GAPS] for lookup in (False,True)]
        if self.groups_per_cell is None:
            return data
        return [[self._groups(x,y,rotation) for x,y in task] for task in data]

    def _groups(self,x,y,rotation):
        count = min(len(y),self.groups_per_cell)
        indices = (torch.arange(count)+rotation)%len(y)
        return x[indices],y[indices]


def samples(seed,step,rule,n,audit):
    namespace = 9172723 if audit else 6811949
    rng = torch.Generator().manual_seed(namespace+1000003*seed+13*step+101*rule)
    return [[questions(task,gap,n,rng,rule) for gap in TRAIN_GAPS] for task in ('recall','lookup')]


@torch.inference_mode()
def scores(model,data,cut=None):
    losses,joint = [],[]
    for task in data:
        total,count,hits,groups = 0.,0,0,0
        for x,y in task:
            logits = model(x.flatten(0,1),cut=cut)
            total += float(F.cross_entropy(logits,y.flatten(),reduction='sum'))
            count += y.numel()
            hits += int((logits.argmax(-1).reshape_as(y)==y).all(1).sum())
            groups += len(y)
        losses.append(total/count)
        joint.append(hits/groups)
    return dict(loss=losses,joint=joint)


class SleepGuard:
    def __init__(self,model,seed,step,rule,prefix=False,training_calibration=None):
        self.model = model
        self.seed = seed
        self.step = step
        self.rule = rule
        self.prefix = prefix
        self.training_calibration = training_calibration
        self.log = {}

    def __call__(self,eligible):
        model = self.model
        candidates = torch.where(eligible.flatten())[0]
        ranked = candidates[model.life.flatten()[candidates].argsort(stable=True)]
        data = (samples(self.seed,self.step,self.rule,CALIBRATION_SIZE,audit=False)
                if self.training_calibration is None else self.training_calibration.data(self.step//INTERVAL))
        baseline = scores(model,data)
        limits = [x+d for x,d in zip(baseline['loss'],TOLERANCE,strict=True)]

        def acceptable(measured):
            return all(a<=b for a,b in zip(measured['loss'],limits,strict=True))

        proposed = scores(model,data,eligible)
        checks = 1
        if acceptable(proposed):
            accepted,after = eligible,proposed
        elif self.prefix:
            accepted,after = torch.zeros_like(eligible),baseline
            # Each tested mask is an initial segment of the same survival order.
            sizes = tuple(dict.fromkeys(len(ranked)//d for d in PREFIX_DIVISORS))
            trial = torch.zeros_like(eligible)
            for count in sizes:
                if not count:
                    break
                trial.zero_().flatten()[ranked[:count]] = True
                measured = scores(model,data,trial)
                checks += 1
                if acceptable(measured):
                    accepted,after = trial,measured
                    break
        else:
            accepted,after = torch.zeros_like(eligible),baseline
            flat = accepted.flatten()
            for subset in ranked.tensor_split(GROUPS):
                if not subset.numel():
                    continue
                flat[subset] = True
                measured = scores(model,data,accepted)
                checks += 1
                if acceptable(measured):
                    after = measured
                else:
                    flat[subset] = False
        self.log = dict(checks=checks,before=baseline,after=after,all_candidates=proposed,
                        sequences_per_check=sum(y.numel() for task in data for _,y in task))
        count = int(accepted.sum())
        if not self.prefix and self.step%100==0 and 0<count<len(ranked):
            audit = samples(self.seed,self.step,self.rule,48,audit=True)
            ordered = torch.zeros_like(eligible)
            ordered.flatten()[ranked[:count]] = True
            self.log['equal_count'] = dict(count=count,same_edges=bool(torch.equal(accepted,ordered)),
                                           before=scores(model,audit),guarded=scores(model,audit,accepted),
                                           low_survival=scores(model,audit,ordered))
        return accepted
