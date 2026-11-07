"""Measure recovery when the queried address changes, from frozen checkpoints."""

import argparse
from hashlib import sha256
from pathlib import Path

import torch
from torch.nn import functional as F

from edge_adamw import make_optimizer
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_eval import BATCH, INTERVAL, TRAIN_GAPS, WIDTH, batch, cells, evaluate, launch, questions
from frontier_eval import digest, read, setup, write


ADAPT_STEPS = 320
LONG_GAPS = (64,128)


@torch.inference_mode()
def calibration_metrics(model,sets):
    metrics = []
    for _,gap,x,y in sets:
        flat = torch.cat([model(part) for part in x.flatten(0,1).split(24)])
        loss = float(F.cross_entropy(flat,y.flatten()))
        hits = (flat.argmax(-1).reshape_as(y)==y).all(1)
        metrics.append((gap,float(hits.float().mean()),loss))
    return metrics


def guarded_update(model,optimizer,sets):
    weight = model.core.weight_hh_l0
    state = optimizer.state[weight]
    tensors = (weight,model.active,model.life,state['exp_avg'],state['exp_avg_sq'])
    saved = [t.detach().clone() for t in tensors]
    age = state.get('edge_age')
    saved_age = age.clone() if age is not None else None
    noise_state = model.noise.get_state()
    before = calibration_metrics(model,sets)
    proposal = model.evolve(optimizer)
    after = calibration_metrics(model,sets)
    accept = all(a[1]>=b[1]-.04 and a[2]<=b[2]+.05 for b,a in zip(before,after,strict=True))
    if not accept:
        with torch.no_grad():
            for target,original in zip(tensors,saved,strict=True):
                target.copy_(original)
            if saved_age is None:
                state.pop('edge_age',None)
            else:
                state['edge_age'].copy_(saved_age)
            model.noise.set_state(noise_state)
    return dict(active=int(model.active.sum()),slept=proposal['slept'] if accept else 0,
                woke=proposal['woke'] if accept else 0,accepted=accept,
                proposed_sleep=proposal['slept'],proposed_wake=proposal['woke'],before=before,after=after)


def adapt(root,result,config,enable_changes,challenge,guard=False):
    seed,mode = result['seed'],result['mode']
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,config,result['active'] if mode=='fixed' else None)
    checkpoint = root/f'{mode}-{seed}'/'weights.pt'
    assert digest(checkpoint)==result['weights_sha256']
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    destination = 'long-transfer-guarded' if guard else challenge
    folder = root/destination/f'{mode}-{seed}{"-frozen" if not enable_changes else ""}'
    folder.mkdir()
    optimizer = make_optimizer(model,config.independent_age)
    rng = torch.Generator().manual_seed(79523+seed)
    if challenge=='transfer':
        new_rule = [c for c in cells('adaptation',(*TRAIN_GAPS,128),rule=1) if c[0]=='lookup']
        old_rule = [c for c in cells('adaptation',(*TRAIN_GAPS,128),rule=0) if c[0]=='lookup']
    else:
        new_rule = [c for c in cells('adaptation',(*LONG_GAPS,256,512)) if c[0]=='recall']
        old_rule = list(cells('adaptation',TRAIN_GAPS))
    calibration = [c for c in cells('calibration',(4,32,*LONG_GAPS),n=24) if c[0]=='recall'] if guard else None
    points,structural = [],[]
    hashes = sha256()
    for step in range(ADAPT_STEPS+1):
        if step:
            if challenge=='transfer':
                x,y = batch(seed,step,rng,rule=1)
            elif step%3:
                gap = LONG_GAPS[(step+seed)%len(LONG_GAPS)]
                x,y = questions('recall',gap,BATCH//2,rng)
                x,y = x.flatten(0,1),y.flatten()
            else:
                x,y = batch(seed,step,rng)
            hashes.update(x.numpy().tobytes())
            hashes.update(y.numpy().tobytes())
            update = enable_changes and mode in ('decay','evolve') and step%INTERVAL==0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x,probe=update),y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            if update:
                event = guarded_update(model,optimizer,calibration) if guard else model.evolve(optimizer)
                structural.append(dict(step=step,**event))
        if step in (0,80,160,ADAPT_STEPS):
            points.append(dict(step=step,old_rule=evaluate(model,old_rule),new_rule=evaluate(model,new_rule),
                               active=int(model.active.sum())))
    write(folder/'trajectory.json',points)
    write(folder/'topology.json',structural)
    return dict(mode='evolve_guarded' if guard else mode if enable_changes else 'evolve_frozen',seed=seed,
                adapt_sha256=hashes.hexdigest(),
                points=points,structural=structural)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--challenge',choices=('transfer','long-transfer'),default='transfer')
    parser.add_argument('--guard',action='store_true')
    args = parser.parse_args()
    if args.guard and args.challenge!='long-transfer':
        parser.error('--guard requires the long-transfer challenge')
    root = args.root
    results = read(root/'development.json')+read(root/'matched.json')
    config = Survival(**read(root/'protocol.json')['survival'])
    dest = 'long-transfer-guarded' if args.guard else args.challenge
    (root/dest).mkdir()
    if args.guard:
        results = [r for r in results if r['mode']=='evolve']
    jobs = [(adapt,(root,r,config,enable,args.challenge,args.guard)) for r in results
            for enable in ((True,) if args.guard else ((True,False) if r['mode']=='evolve' else (True,)))]
    destination = f'{dest}/results.json'
    launch(root,jobs,args.workers,destination)
    loaded = read(root/destination)
    for seed in {r['seed'] for r in loaded}:
        assert len({r['adapt_sha256'] for r in loaded if r['seed']==seed})==1


if __name__=='__main__':
    main()
