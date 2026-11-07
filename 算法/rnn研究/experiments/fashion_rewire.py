"""Fashion-MNIST benchmark for topology learned from a plain dense network."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from hashlib import md5, sha256
from itertools import product
import math
from pathlib import Path
import struct
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from dense_rewire import RewirableLinear, update_connections
from frontier_eval import SEEDS, digest, read, setup, write


FILES = {
    'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310',
}
MODES = ('dense','random','prune','rewire','protected')
EPOCHS = 22
BATCH = 256
WARMUP = 5
TARGET = 12
FREEZE = 17
INTERVAL = 128
DENSITY = .30
TURNOVER = .05
PROTECTION = 2
WIDTHS = (784,160,96,10)
SPLIT_SEED = 365773


def check_dataset(root):
    hashes = {}
    for name,expected in FILES.items():
        data = (root/name).read_bytes()
        if md5(data).hexdigest()!=expected:
            raise ValueError(f'Official Fashion-MNIST checksum mismatch: {name}')
        hashes[name] = sha256(data).hexdigest()
    return hashes


def load(root, prefix, n):
    pictures = gzip.decompress((root/f'{prefix}-images-idx3-ubyte.gz').read_bytes())
    classes = gzip.decompress((root/f'{prefix}-labels-idx1-ubyte.gz').read_bytes())
    assert struct.unpack('>4I',pictures[:16])==(2051,n,28,28)
    assert struct.unpack('>2I',classes[:8])==(2049,n)
    x = torch.frombuffer(bytearray(pictures[16:]),dtype=torch.uint8).reshape(n,784)
    y = torch.frombuffer(bytearray(classes[8:]),dtype=torch.uint8).long()
    assert y.shape==(n,) and torch.bincount(y).tolist()==[n//10]*10
    return x,y


def split(y):
    rng = torch.Generator().manual_seed(SPLIT_SEED)
    train,val = [],[]
    for label in range(10):
        ids = torch.where(y==label)[0]
        ids = ids[torch.randperm(len(ids),generator=rng)]
        train.append(ids[500:])
        val.append(ids[:500])
    return torch.cat(train),torch.cat(val)


class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(RewirableLinear(a,b) for a,b in zip(WIDTHS[:-1],WIDTHS[1:],strict=True))

    def forward(self,x,capture=False,cuts=None):
        for i,layer in enumerate(self.layers):
            x = layer(x,capture,None if cuts is None else cuts[i])
            if i<len(self.layers)-1:
                x = x.relu()
        return x


@torch.no_grad()
def predictions(model,x,cuts=None):
    return torch.cat([model(batch,cuts=cuts).argmax(-1) for batch in x.split(512)])


def count(layer, fraction):
    return round(layer.mask.numel()*fraction)


def fit(root,data,mode,seed):
    setup(seed)
    start = perf_counter()
    x,y = load(data,'train',60000)
    train,val = split(y)
    xt = x[train].float()
    center,scale = float(xt.mean()),float(xt.std())
    xt.sub_(center).div_(scale)
    xv = (x[val].float()-center)/scale
    yt,yv = y[train],y[val]
    model = ImageMLP()
    init_hash = sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    order_rng = torch.Generator().manual_seed(seed+47300)
    mask_rng = torch.Generator().manual_seed(seed+93107)
    if mode=='random':
        for layer in model.layers:
            ids = torch.randperm(layer.mask.numel(),generator=mask_rng)[count(layer,DENSITY):]
            layer.mask.flatten()[ids] = False
            with torch.no_grad():
                layer.weight.flatten()[ids] = 0
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    steps = math.ceil(len(train)/BATCH)
    warm,target,freeze = (e*steps for e in (WARMUP,TARGET,FREEZE))
    graph_changes = set(range(warm+INTERVAL,freeze+1,INTERVAL)) | {target,freeze}
    if mode=='prune':
        graph_changes = {s for s in graph_changes if s<=target}
    elif mode not in ('rewire','protected'):
        graph_changes = set()
    ages = [torch.full((l.mask.numel(),),-1,dtype=torch.long) for l in model.layers]
    folder = root/f'{mode}-{seed}'
    folder.mkdir()
    timeline,changes = [],[]
    best = None
    order_hash = sha256()
    step = event = 0
    for epoch in range(1,EPOCHS+1):
        model.train()
        order = torch.randperm(len(train),generator=order_rng)
        order_hash.update(order.numpy().tobytes())
        loss_sum = 0.
        for ids in order.split(BATCH):
            step += 1
            rate = .0002+.0018*(1+math.cos(math.pi*(step-1)/(EPOCHS*steps)))/2
            optimizer.param_groups[0]['lr'] = rate
            update = step in graph_changes
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[ids],capture=update and mode in ('rewire','protected'))
            loss = F.cross_entropy(logits,yt[ids])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(ids)
            if update:
                event += 1
                fraction = 1-(1-DENSITY)*min(1,(step-warm)/(target-warm))
                layers = []
                for layer,age in zip(model.layers,ages,strict=True):
                    recent = (age>=0)&(event-age<=PROTECTION) if mode=='protected' else None
                    dead,born = update_connections(layer,optimizer,'adaptive',count(layer,fraction),mask_rng,
                                                   0 if mode=='prune' else TURNOVER,recent)
                    layers.append(dict(dropped=dead.tolist(),regrown=born.tolist(),active=int(layer.mask.sum()),
                                       young_removed=int(((age[dead]>=0)&(event-age[dead]<=PROTECTION)).sum())))
                    age[dead] = -1
                    age[born] = event
                changes.append(dict(step=step,epoch=epoch,event=event,layers=layers))
        model.eval()
        correct = int((predictions(model,xv)==yv).sum())
        timeline.append(dict(epoch=epoch,validation_correct=correct,train_loss=loss_sum/len(train),
                             active=[int(l.mask.sum()) for l in model.layers]))
        if epoch in (1,WARMUP,TARGET,FREEZE,EPOCHS):
            torch.save(model.state_dict(),folder/f'epoch-{epoch}.pt')
            print(mode,seed,epoch,round(correct/len(val),4),flush=True)
        if epoch>=FREEZE and (best is None or correct>best[0]):
            best = correct,epoch
            torch.save(model.state_dict(),folder/'selected.pt')
    write(folder/'rewiring.json',changes)
    # Store origins aligned with the selected mask. Topology is fixed from FREEZE.
    torch.save(ages,folder/'birth-events.pt')
    return dict(mode=mode,seed=seed,timeline=timeline,validation_size=len(val),selected_epoch=best[1],
                selected_sha256=digest(folder/'selected.pt'),init_sha256=init_hash,
                order_sha256=order_hash.hexdigest(),train_index_sha256=sha256(train.numpy().tobytes()).hexdigest(),
                center=center,scale=scale,seconds=perf_counter()-start)


def layer_cut_study(model,x,y,ages,seed):
    layer = model.layers[1]
    active = layer.mask.flatten()
    newborn = torch.where(active & (ages[1]>=0))[0]
    retained = torch.where(active & (ages[1]<0))[0]
    size = min(round(.10*int(active.sum())),len(newborn),len(retained))
    if not size:
        return None
    rng = torch.Generator().manual_seed(seed+9801)
    result = {'edges_per_cut':size,'active_regrown':len(newborn),'active_retained':len(retained),
              'regrown_correct':[],'retained_correct':[]}
    for _ in range(8):
        for label,source in (('regrown',newborn),('retained',retained)):
            ids = source[torch.randperm(len(source),generator=rng)[:size]]
            cuts = [torch.zeros_like(l.mask) for l in model.layers]
            cuts[1].flatten()[ids] = True
            result[f'{label}_correct'].append(int((predictions(model,x,cuts)==y).sum()))
    return result


def test(root,data,r):
    setup(r['seed'])
    x,y = load(data,'t10k',10000)
    x = (x.float()-r['center'])/r['scale']
    folder = root/f'{r["mode"]}-{r["seed"]}'
    assert digest(folder/'selected.pt')==r['selected_sha256']
    model = ImageMLP().eval()
    model.load_state_dict(torch.load(folder/'selected.pt',weights_only=True))
    p = predictions(model,x)
    write(folder/'test-predictions.json',p.tolist())
    cuts = None
    if r['mode'] in ('rewire','protected'):
        ages = torch.load(folder/'birth-events.pt',weights_only=True)
        cuts = layer_cut_study(model,x,y,ages,r['seed'])
    return r | dict(correct=int((p==y).sum()),n=len(y),confusion=torch.bincount(10*y+p,minlength=100).reshape(10,10).tolist(),
                    active=[int(l.mask.sum()) for l in model.layers],capacity=[l.mask.numel() for l in model.layers],cuts=cuts)


def ordered(results):
    return sorted(results,key=lambda r:(r['mode'],r['seed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','test'))
    parser.add_argument('output',type=Path)
    parser.add_argument('data',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    hashes = check_dataset(args.data)
    if args.phase=='train':
        args.output.mkdir(parents=True)
        write(args.output/'protocol.json',dict(dataset='Fashion-MNIST',sha256=hashes,modes=MODES,seeds=SEEDS,
              epochs=EPOCHS,batch=BATCH,train_size=55000,val_size=5000,test_size=10000,
              split_seed=SPLIT_SEED,warmup=WARMUP,target=TARGET,freeze=FREEZE,interval=INTERVAL,
              density=DENSITY,turnover=TURNOVER,protection=PROTECTION,widths=WIDTHS,torch=torch.__version__))
        jobs = [(fit,(args.output,args.data,m,s)) for m,s in product(MODES,SEEDS)]
        name = 'validation.json'
    else:
        protocol = read(args.output/'protocol.json')
        assert protocol['sha256']==hashes
        validation = read(args.output/'validation.json')
        assert {(r['mode'],r['seed']) for r in validation}==set(product(MODES,SEEDS)) and len(validation)==15
        for seed in SEEDS:
            runs = [r for r in validation if r['seed']==seed]
            for key in ('init_sha256','order_sha256','train_index_sha256','center','scale'):
                assert len({r[key] for r in runs})==1
        jobs = [(test,(args.output,args.data,r)) for r in validation]
        name = 'results.json'
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fn,*inputs) for fn,inputs in jobs]
        for task in as_completed(futures):
            results.append(task.result())
            write(args.output/name,ordered(results))


if __name__=='__main__':
    main()
