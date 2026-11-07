"""Measure retention of two independently randomized, addressable values."""

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from edge_adamw import make_optimizer
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_audit import profile_model, sensitivity
from evolving_rnn_eval import INTERVAL, MODES, WIDTH, launch
from frontier_eval import SEEDS, digest, read, setup, write


RECIPES = ('answers','joint-state','compute')
CONFIG = Survival(cost=.32,dormant_credit=1.30,independent_age=True)
STEPS = 2400
WARMUP = 600
TRAIN_GAPS = (0,4,16,32)
DEV_GAPS = (*TRAIN_GAPS,64)
TEST_GAPS = (*DEV_GAPS,128,256,512)
BATCH = 24
MIN_JOINT = .80
ADAPT_STEPS = 360


def cue_groups(gap,n,rng):
    values = torch.randint(4,(n,2),generator=rng)
    offsets = torch.randint(1,4,(n,2),generator=rng)
    order = torch.rand((n,2),generator=rng).argsort(1)
    noise = torch.randint(4,6,(n,gap),generator=rng)
    variants = [values]
    for channel in range(2):
        changed = values.clone()
        changed[:,channel] = (changed[:,channel]+offsets[:,channel])%4
        variants.append(changed)
    targets = torch.stack(variants,dim=1)
    recorded = targets.gather(2,order[:,None].expand(-1,3,-1))
    records = torch.stack((10+order[:,None].expand(-1,3,-1),recorded),dim=-1).flatten(2)
    prefix = torch.cat((torch.full((n,3,1),6),records,noise[:,None].expand(-1,3,-1),
                        torch.full((n,3,1),7)),dim=-1)
    query = torch.tensor([8,9]).view(1,1,2,1).expand(n,3,-1,-1)
    x = torch.cat((prefix[:,:,None].expand(-1,-1,2,-1),query),dim=-1)
    return x,targets


def encode(x,recipe):
    if recipe=='compute':
        return torch.cat((x,x[...,-1:].expand(*x.shape[:-1],3)),dim=-1)
    return x


def data_sets(partition,gaps,n):
    base = {'dev':108513,'test':769831,'diagnosis':257657}[partition]
    for gap in gaps:
        generator = torch.Generator().manual_seed(base+gap)
        yield gap,*cue_groups(gap,n,generator)


@torch.inference_mode()
def evaluate(model,sets,recipe,auxiliary=None,save_probabilities=False,cut=None):
    model.eval()
    results = []
    for gap,x,y in sets:
        guesses,reconstructed = [],[]
        for part in x.flatten(0,2).split(24):
            if auxiliary is None:
                logits = model(encode(part,recipe),cut=cut)
            else:
                logits,state = model(encode(part,recipe),cut=cut,prefix_index=-2)
                reconstructed.append(auxiliary(state[::2]).reshape(-1,2,4).argmax(-1))
            guesses.append(logits.softmax(-1))
        probs = torch.cat(guesses).reshape(*y.shape,4)
        hits = probs.argmax(-1)==y
        row = dict(gap=gap,joint=float(hits.flatten(1).all(1).float().mean()),
                   both_original=float(hits[:,0].all(1).float().mean()),
                   per_cue=hits.all(1).float().mean(0).tolist())
        if reconstructed:
            decoded = torch.cat(reconstructed).reshape_as(y)
            row['prefix_both'] = float((decoded==y).all(-1).float().mean())
        if save_probabilities:
            row.update(targets=y.tolist(),probabilities=probs.tolist())
        results.append(row)
    return results


def fit(root,recipe,mode,seed,steps,matched=None):
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,CONFIG,matched)
    optimizer = make_optimizer(model,CONFIG.independent_age)
    auxiliary = nn.Linear(WIDTH,8) if recipe=='joint-state' else None
    aux_optimizer = torch.optim.AdamW(auxiliary.parameters(),lr=.001) if auxiliary is not None else None
    rng = torch.Generator().manual_seed(seed+49719)
    folder = root/f'{recipe}-{mode}-{seed}'
    folder.mkdir()
    history,updates = [],[]
    data_hash = sha256()
    dev = list(data_sets('dev',DEV_GAPS,64))
    for step in range(1,steps+1):
        gaps = (0,) if step<=300 else (0,4) if step<=900 else TRAIN_GAPS
        gap = gaps[(step+seed)%len(gaps)]
        x,y = cue_groups(gap,BATCH//6,rng)
        x,y = x.flatten(0,2),y.flatten()
        data_hash.update(x.numpy().tobytes())
        data_hash.update(y.numpy().tobytes())
        structural = step>WARMUP and step%INTERVAL==0 and mode in ('evolve','decay')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if auxiliary is None:
            loss = F.cross_entropy(model(encode(x,recipe),probe=structural),y)
        else:
            aux_optimizer.zero_grad(set_to_none=True)
            logits,states = model(x,probe=structural,prefix_index=[2,4,-2])
            out = auxiliary(states[::2]).reshape(-1,3,2,4)
            first = F.one_hot(x[::2,1]-10,2).bool()
            mask = torch.cat((first[:,None],torch.ones_like(first[:,None]).expand(-1,2,-1)),dim=1)
            target = y.reshape(-1,1,2).expand(-1,3,-1)
            loss = F.cross_entropy(logits,y)+.5*F.cross_entropy(out[mask],target[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        if structural:
            updates.append(dict(step=step,**model.evolve(optimizer)))
        if step in (300,600,900,1400,1900) or step==steps:
            scores = evaluate(model,dev,recipe,auxiliary)
            history.append(dict(step=step,scores=scores,loss=loss.item(),active=int(model.active.sum())))
            write(folder/'development.json',history)
            print(recipe,mode,seed,step,[round(c['joint'],3) for c in scores],
                  [round(c.get('prefix_both',0.),3) for c in scores],flush=True)
    weights = folder/'weights.pt'
    torch.save(model.state_dict(),weights)
    write(folder/'topology.json',updates)
    return dict(recipe=recipe,mode=mode,seed=seed,steps=steps,history=history,active=int(model.active.sum()),
                train_sha256=data_hash.hexdigest(),weights_sha256=digest(weights))


def restore(root,row):
    recipe,mode,seed = (row[k] for k in ('recipe','mode','seed'))
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,CONFIG,row['active'] if mode=='fixed' else None)
    folder = root/f'{recipe}-{mode}-{seed}'
    weights = folder/'weights.pt'
    assert digest(weights)==row['weights_sha256']
    model.load_state_dict(torch.load(weights,weights_only=True))
    return model.eval(),folder


def test(root,row):
    model,folder = restore(root,row)
    recipe = row['recipe']
    scores = evaluate(model,list(data_sets('test',TEST_GAPS,128)),recipe,save_probabilities=True)
    write(folder/'test.json',scores)
    return row | {'test':[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in scores]}


def interventions(root,row):
    model,_ = restore(root,row)
    recipe = row['recipe']
    gap,x,y = next(data_sets('dev',(32,),48))
    importance = [sensitivity(model,('cue',gap,encode(x[:,:,cue],recipe),y[:,:,cue]))
                  for cue in range(2)]
    active = torch.where(model.active.flatten())[0]
    count = max(1,round(.01*len(active)))
    rng = torch.Generator().manual_seed(row['seed']+847207)
    cases = list(data_sets('test',(4,32,128,256),96))
    base = evaluate(model,cases,recipe)
    effects = []
    for cue in range(2):
        contrast = importance[cue]-importance[1-cue]
        selected = active[contrast.flatten()[active].argsort(descending=True,stable=True)[:count]]
        cut = torch.zeros_like(model.active)
        cut.flatten()[selected] = True
        treated = evaluate(model,cases,recipe,cut=cut)
        controls = []
        rows = selected//WIDTH
        for _ in range(3):
            random_cut = torch.zeros_like(model.active)
            for i in rows.unique().tolist():
                sources = torch.where(model.active[i])[0]
                k = int((rows==i).sum())
                random_cut[i,sources[torch.randperm(len(sources),generator=rng)[:k]]] = True
            controls.append(evaluate(model,cases,recipe,cut=random_cut))
        effects.append(dict(cue=cue,indices=selected.tolist(),target=treated,random=controls))
    return dict(mode=row['mode'],seed=row['seed'],base=base,effects=effects)


def resource(root,row):
    model,_ = restore(root,row)
    _,x,_ = next(data_sets('test',(512,),2))
    return profile_model(model,encode(x[0,0,:1],row['recipe']),row)


def adapt(root,row,frozen=False):
    recipe,mode,seed = (row[k] for k in ('recipe','mode','seed'))
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,CONFIG,row['active'] if mode=='fixed' else None)
    weights = root/f'{recipe}-{mode}-{seed}'/'weights.pt'
    assert digest(weights)==row['weights_sha256']
    model.load_state_dict(torch.load(weights,weights_only=True))
    optimizer = make_optimizer(model,CONFIG.independent_age)
    rng = torch.Generator().manual_seed(seed+603017)
    queries = [(gap,x,y.flip(-1)) for gap,x,y in data_sets('diagnosis',TEST_GAPS,96)]
    points,updates = [],[]
    data_hash = sha256()
    for step in range(ADAPT_STEPS+1):
        if step:
            x,y = cue_groups(TRAIN_GAPS[(step+seed)%len(TRAIN_GAPS)],BATCH//6,rng)
            x,y = x.flatten(0,2),y.flip(-1).flatten()
            data_hash.update(x.numpy().tobytes())
            data_hash.update(y.numpy().tobytes())
            structural = step%INTERVAL==0 and mode in ('evolve','decay') and not frozen
            model.train()
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(encode(x,recipe),probe=structural),y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if structural:
                updates.append(dict(step=step,**model.evolve(optimizer)))
        if step in (0,40,100,200,ADAPT_STEPS):
            points.append(dict(step=step,active=int(model.active.sum()),scores=evaluate(model,queries,recipe)))
    return dict(mode='evolve-frozen' if frozen else mode,seed=seed,recipe=recipe,points=points,updates=updates,
                train_sha256=data_hash.hexdigest(),source_sha256=row['weights_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('probe','topology','matched','test','adapt','cuts','resources'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--recipes',choices=RECIPES,nargs='+',default=RECIPES)
    parser.add_argument('--mode',choices=MODES)
    args = parser.parse_args()
    root = args.root
    if args.phase=='probe':
        root.mkdir(parents=True)
        write(root/'protocol.json',dict(recipes=args.recipes,seeds=SEEDS,steps=args.steps,batch=BATCH,
                                       warmup=WARMUP,train_gaps=TRAIN_GAPS,gate=MIN_JOINT,config=asdict(CONFIG)))
        jobs = [(fit,(root,r,'dense',s,args.steps)) for r,s in product(args.recipes,SEEDS)]
        dest = 'development.json'
    else:
        protocol = read(root/'protocol.json')
        trials = read(root/'development.json')
        assert {(r['recipe'],r['seed']) for r in trials}==set(product(protocol['recipes'],SEEDS))
        if args.phase=='topology':
            scores = {recipe:min(c['joint'] for r in trials if r['recipe']==recipe
                                 for c in r['history'][-1]['scores'] if c['gap'] in (0,4))
                      for recipe in protocol['recipes']}
            recipe = max(scores,key=scores.get)
            if scores[recipe]<MIN_JOINT:
                raise SystemExit('Short-gap two-cue development gate not reached.')
            write(root/'selection.json',dict(recipe=recipe,short_gap_minimum=scores))
            jobs = [(fit,(root,recipe,mode,s,protocol['steps'])) for mode,s in product(('evolve','decay'),SEEDS)]
            dest = 'topology.json'
        else:
            recipe = read(root/'selection.json')['recipe']
            adaptive = read(root/'topology.json')
            assert len(adaptive)==6
            if args.phase=='matched':
                jobs = [(fit,(root,recipe,'fixed',r['seed'],protocol['steps'],r['active']))
                        for r in adaptive if r['mode']=='evolve']
                dest = 'matched.json'
            else:
                rows = [r for r in trials if r['recipe']==recipe]+adaptive+read(root/'matched.json')
                assert {(r['mode'],r['seed']) for r in rows}==set(product(MODES,SEEDS)) and len(rows)==12
                for seed in SEEDS:
                    assert len({r['train_sha256'] for r in rows if r['seed']==seed})==1
                if args.phase=='test':
                    jobs = [(test,(root,row)) for row in rows]
                    dest = 'results.json'
                elif args.phase=='adapt':
                    jobs = [(adapt,(root,row,frozen)) for row in rows
                            for frozen in ((False,True) if row['mode']=='evolve' else (False,))]
                    dest = 'adaptation.json'
                elif args.phase=='cuts':
                    jobs = [(interventions,(root,row)) for row in rows if row['mode'] in ('dense','evolve')]
                    dest = 'cuts.json'
                else:
                    if args.mode is None:
                        parser.error('resources requires --mode in an isolated process')
                    row = next(r for r in rows if r['mode']==args.mode and r['seed']==2)
                    write(root/f'profile-{args.mode}.json',resource(root,row))
                    return
    launch(root,jobs,args.workers,dest)


if __name__=='__main__':
    main()
