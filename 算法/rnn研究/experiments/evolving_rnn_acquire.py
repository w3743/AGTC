"""Test whether renewed edges help acquire a memory that was previously irrelevant."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch
from torch.nn import functional as F

from edge_adamw import make_optimizer
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_eval import BATCH, GROUPS, INTERVAL, TRAIN_GAPS, WARMUP, WIDTH, evaluate, launch
from frontier_eval import SEEDS, digest, read, setup, write


PRETRAIN = 1200
ADAPT = 480
LONG = (64,128)
EXTRAPOLATE = (256,512)
CONFIG = Survival(dormant_credit=1.30,independent_age=True)


def paired(gap,n,rng,channel):
    cue = torch.randint(4,(n,2),generator=rng)
    distractors = torch.randint(4,6,(n,gap),generator=rng)
    marker = torch.full((n,1),6)
    suffix = torch.tensor([7,8+channel]).expand(n,2)
    x = torch.cat((marker,cue[:,:1],marker,cue[:,1:],distractors,suffix),dim=1)
    alternatives = [x]
    changed_values = (cue+torch.randint(1,4,(n,2),generator=rng))%4
    for source in (channel,1-channel):
        altered = x.clone()
        altered[:,1+2*source] = changed_values[:,source]
        alternatives.append(altered)
    return (torch.stack(alternatives,dim=1),
            torch.stack((cue[:,channel],changed_values[:,channel],cue[:,channel]),dim=1))


def sets(channel,gaps,namespace):
    for gap in gaps:
        rng = torch.Generator().manual_seed(namespace+gap)
        yield f'cue-{channel+1}',gap,*paired(gap,GROUPS,rng,channel)


def scores(model,namespace):
    return dict(old=evaluate(model,list(sets(0,TRAIN_GAPS,namespace))),
                novel_short=evaluate(model,list(sets(1,TRAIN_GAPS,namespace+16000))),
                novel=evaluate(model,list(sets(1,(*LONG,*EXTRAPOLATE),namespace+8000))))


def update(model,optimizer,seed,step,rng,phase,freeze,staged=False):
    channel = int(phase=='adapt' and step%3!=0)
    gaps = LONG if channel else (0,4) if phase=='pretrain' and step<=160 else TRAIN_GAPS
    if staged and channel and step<=240:
        gaps = (0,4) if step<=120 else (16,32)
    gap = gaps[(step+seed)%len(gaps)]
    x,y = paired(gap,BATCH//2,rng,channel)
    variants = (0,1+step%2)
    x,y = x[:,variants].flatten(0,1),y[:,variants].flatten()
    structural = not freeze and step%INTERVAL==0 and (phase=='adapt' or step>WARMUP)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(x,probe=structural),y).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    event = model.evolve(optimizer) if structural else None
    return x,y,event


def pretrain(root,seed):
    setup(seed)
    model = EvolvingRNN(WIDTH,'evolve',CONFIG)
    optimizer = make_optimizer(model,CONFIG.independent_age)
    rng = torch.Generator().manual_seed(seed+99413)
    path = root/f'seed-{seed}'
    path.mkdir()
    topology = []
    for step in range(1,PRETRAIN+1):
        _,_,event = update(model,optimizer,seed,step,rng,'pretrain',False)
        if event is not None:
            topology.append(dict(step=step,**event))
        if step in (400,800,PRETRAIN):
            progress = evaluate(model,list(sets(0,TRAIN_GAPS,36419)))
            print(seed,step,[c['joint'] for c in progress],int(model.active.sum()),flush=True)
    weights = path/'weights.pt'
    torch.save(model.state_dict(),weights)
    write(path/'topology.json',topology)
    result = dict(mode='evolve',seed=seed,active=int(model.active.sum()),weights_sha256=digest(weights),
                  scores=scores(model,36419))
    write(path/'result.json',result)
    return result


def adapt(root,result,freeze,staged):
    seed = result['seed']
    setup(seed)
    path = root/f'seed-{seed}'/'weights.pt'
    assert digest(path)==result['weights_sha256']
    model = EvolvingRNN(WIDTH,'evolve',CONFIG)
    model.load_state_dict(torch.load(path,weights_only=True))
    optimizer = make_optimizer(model,CONFIG.independent_age)
    rng = torch.Generator().manual_seed(seed+77322)
    points,topology = [],[]
    batches = sha256()
    for step in range(ADAPT+1):
        if step:
            x,y,event = update(model,optimizer,seed,step,rng,'adapt',freeze,staged)
            batches.update(x.numpy().tobytes())
            batches.update(y.numpy().tobytes())
            if event is not None:
                topology.append(dict(step=step,**event))
        if step in (0,80,160,320,ADAPT):
            point = scores(model,871091)
            points.append(dict(step=step,active=int(model.active.sum()),**point))
            print(seed,'frozen' if freeze else 'evolve',step,[c['joint'] for c in point['novel']],flush=True)
    return dict(mode='frozen' if freeze else 'evolve',seed=seed,staged=staged,source_sha256=result['weights_sha256'],
                train_sha256=batches.hexdigest(),points=points,topology=topology)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('pretrain','adapt'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--staged',action='store_true')
    args = parser.parse_args()
    root = args.root
    if args.phase=='pretrain':
        root.mkdir(parents=True)
        write(root/'protocol.json',dict(seeds=SEEDS,pretrain=PRETRAIN,adapt=ADAPT,old_gaps=TRAIN_GAPS,
              new_train_gaps=LONG,new_extra_gaps=EXTRAPOLATE,batch=BATCH,groups=GROUPS))
        launch(root,[(pretrain,(root,seed)) for seed in SEEDS],args.workers,'pretrain.json')
    else:
        rows = read(root/'pretrain.json')
        assert {r['seed'] for r in rows}==set(SEEDS)
        if not all(min(c['joint'] for c in r['scores']['old'])>=.8 for r in rows):
            raise SystemExit('Pretraining has not established old-cue recall across all training gaps.')
        destination = 'adaptation-staged.json' if args.staged else 'adaptation.json'
        launch(root,[(adapt,(root,row,freeze,args.staged)) for row in rows for freeze in (False,True)],
               args.workers,destination)
        measured = read(root/destination)
        for seed in SEEDS:
            same = [r for r in measured if r['seed']==seed]
            assert len({r['train_sha256'] for r in same})==1
            assert len({r['source_sha256'] for r in same})==1
            assert same[0]['points'][0]==same[1]['points'][0]


if __name__=='__main__':
    main()
