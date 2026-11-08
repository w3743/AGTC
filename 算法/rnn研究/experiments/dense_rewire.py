"""Dense-to-sparse experiment on queried two-source reasoning."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from frontier_eval import SEEDS, setup, write


KEYS = 4
MODES = ('dense','random','adaptive')
QUESTIONS = ('A','B','both')
WIDTH = 96
BATCH = 64
STEPS = 2000
WARMUP = 400
TARGET_STEP = 1000
LAST_REWIRE = 1650
INTERVAL = 50
TARGET_DENSITY = .30
TURNOVER = .08
INPUTS = 2*KEYS*(KEYS+1)+KEYS+len(QUESTIONS)
CHECKPOINTS = (0,WARMUP,700,TARGET_STEP,1300,LAST_REWIRE,STEPS)
GROUPS = 512


class DataStream:
    """Split on the pair of key permutations so all counterfactuals stay together."""

    def __init__(self, seed, partition):
        self.rng = torch.Generator().manual_seed(seed)
        self.partition = partition

    def draw(self, size, question=None):
        layouts = []
        remaining = size
        while remaining:
            factor = 2 if self.partition=='train' else 14
            order = torch.rand((max(32,factor*remaining),2,KEYS),generator=self.rng).argsort(-1)
            code = (order.flatten(1)*KEYS**torch.arange(2*KEYS)).sum(1)
            bucket = ((code*0x45D9F3B) ^ (code>>5)) % 11
            keep = bucket>=3 if self.partition=='train' else bucket==self.partition
            selected = order[keep][:remaining]
            layouts.append(selected)
            remaining -= len(selected)
        order = torch.cat(layouts)
        bits = torch.randint(2,(size,2,KEYS),generator=self.rng)
        key = torch.randint(KEYS,(size,),generator=self.rng)
        if question is None:
            q = torch.randint(4,(size,),generator=self.rng).clamp_max(2)
        else:
            q = torch.full((size,),question,dtype=torch.long)
        return order,bits,key,q


def encode(order,bits,key,q):
    values = bits.gather(2,order)
    records = torch.cat((F.one_hot(order,KEYS),values.unsqueeze(-1)),dim=-1)
    inputs = torch.cat((records.flatten(1),F.one_hot(key,KEYS),F.one_hot(q,3)),dim=-1)
    target = bits.gather(2,key[:,None,None].expand(-1,2,1)).squeeze(2)
    a,b = target.unbind(1)
    labels = torch.where(q==0,a,torch.where(q==1,b,(a==b).long()))
    return inputs.float(),labels


def paired_questions(partition, size):
    stream = DataStream(78190+partition,partition)
    result = {}
    for question,name in enumerate(QUESTIONS):
        order,bits,key,q = stream.draw(size,question)
        a,b,other = (bits.clone() for _ in range(3))
        rows = torch.arange(size)
        a[rows,0,key] ^= 1
        b[rows,1,key] ^= 1
        other[rows,0,(key+1)%KEYS] ^= 1
        variants = [encode(order,v,key,q) for v in (bits,a,b,other)]
        x,y = (torch.stack(v,dim=1) for v in zip(*variants,strict=True))
        assert torch.equal(y[:,0]!=y[:,1],q!=1)
        assert torch.equal(y[:,0]!=y[:,2],q!=0)
        assert torch.equal(y[:,0],y[:,3])
        assert torch.all((x[:,0,None]!=x[:,1:]).sum(-1)==1)
        result[name] = x,y
    return result


class RewirableLinear(nn.Linear):
    def __init__(self, n_in, n_out):
        super().__init__(n_in,n_out)
        self.register_buffer('mask',torch.ones_like(self.weight,dtype=torch.bool))
        self.probe = None

    def forward(self,x,capture=False,cut=None):
        mask = self.mask if cut is None else self.mask & ~cut
        effective = self.weight*mask
        if capture:
            effective.retain_grad()
            self.probe = effective
        return F.linear(x,effective,self.bias)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        dims = (INPUTS,WIDTH,WIDTH,WIDTH,2)
        self.layers = nn.ModuleList(RewirableLinear(a,b) for a,b in zip(dims[:-1],dims[1:],strict=True))

    def forward(self,x,capture=False,cuts=None):
        for i,layer in enumerate(self.layers):
            x = layer(x,capture,None if cuts is None else cuts[i])
            if i<len(self.layers)-1:
                x = F.relu(x)
        return x


def target_count(total,step):
    progress = min(1,(step-WARMUP)/(TARGET_STEP-WARMUP))
    return round(total*(1-(1-TARGET_DENSITY)*progress))


@torch.no_grad()
def update_connections(layer,optimizer,mode,goal,rng,turnover,protected=None):
    mask = layer.mask.flatten()
    weight = layer.weight.flatten()
    active = torch.where(mask)[0]
    inactive = torch.where(~mask)[0]
    grow = min(len(inactive),round(turnover*goal))
    drop = len(active)-goal+grow
    if mode=='adaptive':
        ranked = active[weight[active].abs().argsort(stable=True)]
        if protected is not None:
            # Fall back to the protected candidates only if the density budget requires it.
            ranked = ranked[protected[ranked].to(torch.uint8).argsort(stable=True)]
        dead = ranked[:drop]
        if grow:
            gradient = layer.probe.grad.flatten()
            born = inactive[gradient[inactive].abs().argsort(descending=True,stable=True)[:grow]]
        else:
            born = inactive[:0]
    else:
        dead = active[torch.randperm(len(active),generator=rng)[:drop]]
        born = inactive[torch.randperm(len(inactive),generator=rng)[:grow]]
    mask[dead] = False
    mask[born] = True
    changed = torch.cat((dead,born))
    weight[changed] = 0
    for key in ('exp_avg','exp_avg_sq'):
        optimizer.state[layer.weight][key].flatten()[changed] = 0
    assert int(mask.sum())==goal
    layer.probe = None
    return dead,born


def rewire(model,optimizer,mode,step,rng,turnover=TURNOVER):
    changes = []
    for layer in model.layers:
        goal = target_count(layer.mask.numel(),step)
        dead,born = update_connections(layer,optimizer,mode,goal,rng,turnover)
        changes.append(dict(dropped=dead.tolist(),regrown=born.tolist(),active=goal))
    return changes


@torch.no_grad()
def predict(model,cases,cuts=None):
    return {name:model(x.flatten(0,1),cuts=cuts).softmax(-1).reshape(*y.shape,2)
            for name,(x,y) in cases.items()}


def scores(probs,cases):
    result = {}
    for name,p in probs.items():
        y = cases[name][1]
        hits = p.argmax(-1)==y
        result[name] = dict(single=float(hits[:,0].float().mean()),joint=float(hits.all(1).float().mean()),
                            per_variant=hits.float().mean(0).tolist())
    return result


def importance(model,cases):
    """Rank hidden-to-hidden edges using answer margins on development inputs."""
    values = {}
    for question in ('A','B'):
        x,y = cases[question]
        estimates = [torch.zeros_like(l.weight) for l in model.layers]
        for start in range(0,len(x),32):
            model.zero_grad(set_to_none=True)
            xx = x[start:start+32].flatten(0,1)
            yy = y[start:start+32].flatten()
            logits = model(xx,capture=True)
            margin = logits.gather(1,yy[:,None])-logits.gather(1,(1-yy)[:,None])
            margin.mean().backward()
            for estimate,layer in zip(estimates,model.layers,strict=True):
                estimate.add_((layer.weight.detach()*layer.probe.grad).abs())
                layer.probe = None
        values[question] = estimates
    return values


def cuts_from_indices(model,indices):
    cuts = [torch.zeros_like(l.mask) for l in model.layers]
    for index,ids in indices.items():
        cuts[int(index)].flatten()[ids] = True
    return cuts


def select_cut(model,dev,values,target,budget):
    other = 'B' if target=='A' else 'A'
    coordinates = []
    for i in (1,2):
        ids = torch.where(model.layers[i].mask.flatten())[0]
        score = values[target][i].flatten()-values[other][i].flatten()
        coordinates.extend((float(score[j]),i,int(j)) for j in ids)
    ranked = sorted(coordinates,reverse=True)
    baseline = scores(predict(model,dev),dev)
    best = None
    # Select a cut size on development data. Both sources use the same candidates.
    for count in (budget//4,budget//2,budget):
        ids = {i:[j for _,layer,j in ranked[:count] if layer==i] for i in (1,2)}
        s = scores(predict(model,dev,cuts_from_indices(model,ids)),dev)
        drop = baseline[target]['joint']-s[target]['joint']
        side = baseline[other]['joint']-s[other]['joint']
        criterion = drop-side
        if best is None or criterion>best[0]:
            best = criterion,ids
    return best[1]


def intervention(model,dev,test,seed):
    values = importance(model,dev)
    rng = torch.Generator().manual_seed(seed+7521)
    output,raw = {},{}
    for target in ('A','B'):
        ids = select_cut(model,dev,values,target,budget=192)
        cut = predict(model,test,cuts_from_indices(model,ids))
        output[target] = dict(edges=ids,scores=scores(cut,test),random=[])
        raw[target] = {name:p.tolist() for name,p in cut.items()}
        for _ in range(12):
            random_ids = {}
            for i,selected in ids.items():
                active = torch.where(model.layers[i].mask.flatten())[0]
                random_ids[i] = active[torch.randperm(len(active),generator=rng)[:len(selected)]].tolist()
            sampled = predict(model,test,cuts_from_indices(model,random_ids))
            output[target]['random'].append(dict(edges=random_ids,scores=scores(sampled,test)))
    return output,raw


def run(root,mode,seed,extra_warmup):
    setup(seed)
    model = DenseNet()
    rng = torch.Generator().manual_seed(seed+91310)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    data = DataStream(seed+31806,'train')
    dev = paired_questions(0,GROUPS)
    test = paired_questions(1,GROUPS)
    folder = root/f'{mode}-{seed}'
    folder.mkdir()
    timeline,history = [],[]
    data_hash = sha256()
    start_hash = sha256()
    for t in model.state_dict().values():
        start_hash.update(t.numpy().tobytes())

    def checkpoint(step):
        torch.save(model.state_dict(),folder/f'weights-{step}.pt')
        accuracy = scores(predict(model,dev),dev)
        timeline.append(dict(step=step,development=accuracy,active=[int(l.mask.sum()) for l in model.layers]))
        print(mode,seed,step,[round(v['joint'],3) for v in accuracy.values()],flush=True)

    checkpoint_steps = {WARMUP,*(t+extra_warmup for t in CHECKPOINTS if t>=WARMUP)}
    checkpoint(0)
    for step in range(1,STEPS+extra_warmup+1):
        x,y = encode(*data.draw(BATCH))
        data_hash.update(x.numpy().tobytes())
        data_hash.update(y.numpy().tobytes())
        relative = step-extra_warmup
        change = mode!='dense' and WARMUP<relative<=LAST_REWIRE and relative%INTERVAL==0
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x,capture=change and mode=='adaptive'),y)
        loss.backward()
        optimizer.step()
        if change:
            history.append(dict(step=step,layers=rewire(model,optimizer,mode,relative,rng)))
        if step in checkpoint_steps:
            checkpoint(step)
    p = predict(model,test)
    cuts,cut_probs = intervention(model,dev,test,seed)
    write(folder/'predictions.json',dict(
        targets={name:y.tolist() for name,(_,y) in test.items()},
        baseline={name:v.tolist() for name,v in p.items()},cuts=cut_probs))
    write(folder/'rewiring.json',history)
    return dict(mode=mode,seed=seed,timeline=timeline,test=scores(p,test),interventions=cuts,
                total=[l.mask.numel() for l in model.layers],
                parameters=sum(p.numel() for p in model.parameters()),
                init_sha256=start_hash.hexdigest(),train_sha256=data_hash.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--extra-warmup',type=int,default=0)
    args = parser.parse_args()
    if args.extra_warmup<0:
        parser.error('--extra-warmup must be nonnegative')
    shift = args.extra_warmup
    args.output.mkdir(parents=True)
    write(args.output/'protocol.json',dict(seeds=SEEDS,modes=MODES,steps=STEPS+shift,batch=BATCH,groups=GROUPS,
          warmup=WARMUP+shift,target_step=TARGET_STEP+shift,last_rewire=LAST_REWIRE+shift,interval=INTERVAL,
          density=TARGET_DENSITY,turnover=TURNOVER,learning_rate=.003,inputs=INPUTS,width=WIDTH))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run,args.output,m,s,shift) for m,s in product(MODES,SEEDS)]
        for job in as_completed(jobs):
            results.append(job.result())
            write(args.output/'results.json',sorted(results,key=lambda r:(r['mode'],r['seed'])))


if __name__=='__main__':
    main()
