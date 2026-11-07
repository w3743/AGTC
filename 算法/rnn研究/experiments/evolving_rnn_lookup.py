"""Train a plain 512-unit RNN to answer every key from a shared history."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import mean
from time import process_time

import torch
from torch import nn
from torch.nn import functional as F

from edge_adamw import make_optimizer
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_audit import profile_model
from evolving_rnn_eval import BATCH, INTERVAL, MODES, TRAIN_GAPS, WIDTH, cells, evaluate, launch, questions
from frontier_eval import SEEDS, digest, read, setup, write


STEPS = 3200
WARMUP = 1600
CHECKPOINTS = (400,800,1200,WARMUP,2000,2400,2800)
DEV_GAPS = (*TRAIN_GAPS,64)
PROBES = ('queries','prefix-readout','ranked','query-ticks','write-readout','packed-records',
          'retention-readout','packed-retention','identity-retention','capacity-curriculum')
PACKED_PROBES = ('packed-records','packed-retention')
RETENTION_PROBES = ('retention-readout','packed-retention','identity-retention')
AUXILIARY_PROBES = ('prefix-readout','write-readout','packed-records',*RETENTION_PROBES,'capacity-curriculum')
CONFIG = Survival(cost=.32,dormant_credit=1.30,independent_age=True)
ADAPT_STEPS = 480


def lookup_queries(gap,n,rng,rule=0):
    """Four questions about each history, plus the same questions after one source edit."""
    order = torch.rand((n,4),generator=rng).argsort(dim=1)
    values = torch.randint(4,(n,4),generator=rng)
    source = torch.randint(4,(n,),generator=rng)
    changed = values.clone()
    rows = torch.arange(n)
    changed[rows,source] = (values[rows,source]+torch.randint(1,4,(n,),generator=rng))%4
    values = torch.stack((values,changed),dim=1).flatten(0,1)
    order = order.repeat_interleave(2,dim=0)
    noise = torch.randint(4,6,(n,gap),generator=rng).repeat_interleave(2,dim=0)
    stem = torch.stack((8+order,values.gather(1,order)),dim=-1).flatten(1)
    n = len(stem)
    prefix = torch.cat((torch.full((n,1),6),stem,noise,torch.full((n,1),7)),dim=1)
    keys = torch.arange(4)
    x = torch.cat((prefix[:,None].expand(-1,4,-1),(8+keys).view(1,4,1).expand(n,-1,-1)),dim=-1)
    y = values[:,(keys+rule)%4]
    return x.flatten(0,1),y.flatten()


def capacity_batch(seed,step,rng):
    """Learn record load before adding temporal distractors."""
    count = 2 if step<=600 else 3 if step<=1200 else 4
    gaps = (0,) if step<=1800 else (0,4) if step<=2600 else (
        (0,4,16) if step<=3400 else TRAIN_GAPS)
    gap = gaps[(step+seed)%len(gaps)]
    if step>2600 and step%5==0:
        x,y = questions('recall',gap,12,rng)
        return False,x.flatten(0,1),y.flatten(),count
    if count==4:
        return True,*lookup_queries(gap,3,rng),count
    groups = 24//(2*count)
    keys = torch.rand((groups,4),generator=rng).argsort(1)[:,:count]
    values = torch.randint(4,(groups,count),generator=rng)
    altered = values.clone()
    target = torch.randint(count,(groups,),generator=rng)
    ids = torch.arange(groups)
    altered[ids,target] = (values[ids,target]+torch.randint(1,4,(groups,),generator=rng))%4
    y = torch.stack((values,altered),dim=1).flatten(0,1)
    keys = keys.repeat_interleave(2,dim=0)
    noise = torch.randint(4,6,(groups,gap),generator=rng).repeat_interleave(2,dim=0)
    records = torch.stack((keys+8,y),dim=-1).flatten(1)
    n = len(records)
    stem = torch.cat((torch.full((n,1),6),records,noise,torch.full((n,1),7)),dim=-1)
    x = torch.cat((stem[:,None].expand(-1,count,-1),(keys+8).unsqueeze(-1)),dim=-1)
    return True,x.flatten(0,1),y.flatten(),count


def batch(seed,step,rng,rule=0,lookup_only=False):
    if lookup_only or step<=WARMUP or (step+seed)%2:
        gap = (0 if step<=800 else (0,4)[step%2]) if step<=WARMUP and not rule else (
            TRAIN_GAPS[(step+seed)%4] if lookup_only else TRAIN_GAPS[(step//2+seed)%4])
        return True,*lookup_queries(gap,BATCH//8,rng,rule)
    gap = TRAIN_GAPS[(step//2+seed)%4]
    x,y = questions('recall',gap,BATCH//2,rng)
    return False,x.flatten(0,1),y.flatten()


def query_ticks(x):
    return torch.cat((x,x[...,-1:].expand(*x.shape[:-1],3)),dim=-1)


def pack_records(x):
    # A fixed input code for the already-delimited key/value records.
    # It leaves the addressed key and the post-record delay unchanged.
    records = 12+4*(x[...,1:9:2]-8)+x[...,2:9:2]
    return torch.cat((x[...,:1],records,x[...,9:]),dim=-1)


def model_for(method,mode,matched=None,config=CONFIG,fixed_mask=None):
    network_mode = 'fixed' if mode=='fixed_learned' else mode
    return EvolvingRNN(WIDTH,network_mode,config,matched,vocab=28 if method in PACKED_PROBES else 12,
                       near_identity=method=='identity-retention',fixed_mask=fixed_mask)


def prepare_cells(probe,sets):
    transform = pack_records if probe in PACKED_PROBES else query_ticks if probe=='query-ticks' else None
    return [(task,gap,transform(x) if task=='lookup' and transform is not None else x,y)
            for task,gap,x,y in sets]


def loss_for(model,readout,method,lookup,x,y,probe=False,record_count=4):
    if not lookup:
        return F.cross_entropy(model(x,probe=probe),y)
    if method=='query-ticks':
        return F.cross_entropy(model(query_ticks(x),probe=probe),y)
    if method=='ranked':
        logits = model(x,probe=probe)
        scores = logits.reshape(-1,4,4)
        answers = y.reshape(-1,4)
        correct = scores.gather(-1,answers.unsqueeze(-1))
        competing = scores.gather(-1,answers.unsqueeze(1).expand(-1,4,-1))
        different = answers.unsqueeze(-1)!=answers.unsqueeze(1)
        ranking = (F.softplus(competing-correct)*different).sum()/different.sum().clamp_min(1)
        return F.cross_entropy(logits,y)+2*ranking
    if readout is None:
        return F.cross_entropy(model(x,probe=probe),y)
    if method!='prefix-readout':
        packed = method in PACKED_PROBES
        encoded = pack_records(x) if packed else x
        positions = list(range(1,record_count+1)) if packed else list(range(2,2*record_count+1,2))
        if method in RETENTION_PROBES or method=='capacity-curriculum':
            positions.extend(range(positions[-1]+4,encoded.shape[-1]-2,4))
        positions.append(-2)
        logits,stems = model(encoded,probe=probe,prefix_index=positions)
        order = x[::record_count,1:2*record_count+1:2]-8
        written = F.one_hot(order,4).cumsum(1).bool()
        observed = written[:,-1:].expand(-1,len(positions)-record_count,-1)
        written = torch.cat((written,observed),dim=1)
        reconstruction = readout(stems[::record_count]).reshape(len(order),len(positions),4,4)
        addressed = x[:,-1].reshape(-1,record_count)-8
        targets = torch.zeros((len(order),4),dtype=y.dtype,device=y.device)
        targets.scatter_(1,addressed,y.reshape(-1,record_count))
        targets = targets[:,None,:].expand(-1,len(positions),-1)
        memory_loss = F.cross_entropy(reconstruction[written],targets[written])
    else:
        logits,stem = model(x,probe=probe,prefix_index=-2)
        reconstruction = readout(stem[::4]).reshape(-1,4)
        memory_loss = F.cross_entropy(reconstruction,y)
    return F.cross_entropy(logits,y)+.5*memory_loss


def fit(root,probe,mode,seed,steps,matched=None,topology_warmup=WARMUP,config=CONFIG,fixed_mask=None,
        reset_after=None,save_training_state=False):
    setup(seed)
    model = model_for(probe,mode,matched,config,fixed_mask)
    optimizer = make_optimizer(model,config.independent_age)
    readout = nn.Linear(WIDTH,16) if probe in AUXILIARY_PROBES else None
    auxiliary = torch.optim.AdamW(readout.parameters(),lr=.001) if readout is not None else None
    rng = torch.Generator().manual_seed(245287+seed)
    folder = root/f'{probe}-{mode}-{seed}'
    folder.mkdir()
    development = prepare_cells(probe,cells('development',DEV_GAPS,n=64))
    topology,history = [],[]
    train_hash = sha256()
    losses = []
    training_cpu_seconds = 0.
    for step in range(1,steps+1):
        if reset_after is not None and step==reset_after+1:
            optimizer = make_optimizer(model,config.independent_age)
            if auxiliary is not None:
                auxiliary = torch.optim.AdamW(readout.parameters(),lr=.001)
        if probe=='capacity-curriculum':
            lookup,x,y,count = capacity_batch(seed,step,rng)
        else:
            lookup,x,y = batch(seed,step,rng)
            count = 4
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
        structural = step>topology_warmup and step%INTERVAL==0 and mode in ('evolve','decay')
        timed = save_training_state and (reset_after is None or step>reset_after)
        if timed:
            started = process_time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if auxiliary is not None:
            auxiliary.zero_grad(set_to_none=True)
        loss = loss_for(model,readout,probe,lookup,x,y,structural,count)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if auxiliary is not None:
            auxiliary.step()
        if structural:
            topology.append(dict(step=step,**model.evolve(optimizer)))
        losses.append(loss.item())
        if timed:
            training_cpu_seconds += process_time()-started
        if step in CHECKPOINTS or step==steps:
            scores = evaluate(model,development)
            history.append(dict(step=step,loss=mean(losses),active=int(model.active.sum()),scores=scores))
            write(folder/'development.json',history)
            print(probe,mode,seed,step,round(mean(losses),3),
                  [round(c['joint'],3) for c in scores],int(model.active.sum()),flush=True)
            losses.clear()
    torch.save(model.state_dict(),folder/'weights.pt')
    if readout is not None:
        torch.save(readout.state_dict(),folder/'auxiliary.pt')
    write(folder/'topology.json',topology)
    result = dict(probe=probe,mode=mode,seed=seed,steps=steps,active=int(model.active.sum()),
                  weights_sha256=digest(folder/'weights.pt'),train_sha256=train_hash.hexdigest(),history=history)
    if save_training_state:
        auxiliary_state = auxiliary.state_dict() if auxiliary is not None else None
        torch.save((optimizer.state_dict(),auxiliary_state),folder/'optimizers.pt')
        result['training_cpu_seconds'] = training_cpu_seconds
    write(folder/'result.json',result)
    return result


def read_selected(root):
    protocol = read(root/'protocol.json')
    selection = read(root/'selection.json')
    return protocol,selection['probe'],Survival(**selection.get('survival',asdict(CONFIG)))


def test(root,row):
    seed,mode,probe = (row[k] for k in ('seed','mode','probe'))
    setup(seed)
    model = model_for(probe,mode,row['active'] if mode in ('fixed','fixed_learned') else None)
    folder = root/f'{probe}-{mode}-{seed}'
    checkpoint = folder/'weights.pt'
    assert digest(checkpoint)==row['weights_sha256']
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    split = 'capacity-holdout' if probe=='capacity-curriculum' else 'holdout'
    results = evaluate(model,prepare_cells(probe,cells(split,(*TRAIN_GAPS,64,128,256,512))),
                       save_predictions=True)
    write(folder/'holdout.json',results)
    return row | {'holdout':[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in results]}


@torch.inference_mode()
def diagnose(root,include_topology=False):
    diagnostics = []
    rows = read(root/'probes.json')
    if include_topology:
        rows += read(root/'topology-controls.json')
        if (root/'matched.json').exists():
            rows += read(root/'matched.json')
    for row in rows:
        method,mode,seed = (row[k] for k in ('probe','mode','seed'))
        setup(seed)
        folder = root/f'{method}-{mode}-{seed}'
        model = model_for(method,mode,row['active'] if mode=='fixed' else None).eval()
        assert digest(folder/'weights.pt')==row['weights_sha256']
        model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
        readout = None
        if method in AUXILIARY_PROBES:
            readout = nn.Linear(WIDTH,16)
            readout.load_state_dict(torch.load(folder/'auxiliary.pt',weights_only=True))
        by_gap = []
        for gap in (0,4,16,32):
            rng = torch.Generator().manual_seed(784271+gap)
            x,y = lookup_queries(gap,128,rng)
            guesses,memory_guesses = [],[]
            for part in x.split(32):
                inputs = prepare_cells(method,[('lookup',gap,part,None)])[0][2]
                logits,stem = model(inputs,prefix_index=-2)
                guesses.append(logits.argmax(-1))
                if readout is not None:
                    memory_guesses.append(readout(stem[::4]).reshape(-1,4).argmax(-1))
            target = y.reshape(-1,4)
            predictions = torch.cat(guesses).reshape(-1,4)
            differing = target[:,:,None]!=target[:,None,:]
            observed = predictions[:,:,None]!=predictions[:,None,:]
            acc = predictions==target
            by_gap.append(dict(gap=gap,single=float(acc.float().mean()),
                               all_queries=float(acc.all(1).float().mean()),
                               prediction_changes=float(observed[differing].float().mean()),
                               auxiliary_single=(float((torch.cat(memory_guesses)==y).float().mean())
                                                 if memory_guesses else None)))
        diagnostics.append(dict(probe=method,mode=mode,seed=seed,by_gap=by_gap))
    name = 'binding-diagnosis-topology.json' if include_topology else 'binding-diagnosis.json'
    write(root/name,diagnostics)


def adapt(root,row,freeze):
    seed,mode,probe = (row[k] for k in ('seed','mode','probe'))
    setup(seed)
    _,_,config = read_selected(root)
    model = model_for(probe,mode,row['active'] if mode=='fixed' else None,config)
    folder = root/f'{probe}-{mode}-{seed}'
    checkpoint = folder/'weights.pt'
    assert digest(checkpoint)==row['weights_sha256']
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    optimizer = make_optimizer(model,config.independent_age)
    readout = nn.Linear(WIDTH,16) if probe in AUXILIARY_PROBES else None
    auxiliary = None
    if readout is not None:
        readout.load_state_dict(torch.load(folder/'auxiliary.pt',weights_only=True))
        auxiliary = torch.optim.AdamW(readout.parameters(),lr=.001)
    rng = torch.Generator().manual_seed(806131+seed)
    split = 'capacity-adaptation' if probe=='capacity-curriculum' else 'adaptation'
    new_tasks = prepare_cells(probe,(c for c in cells(split,DEV_GAPS,n=96,rule=1) if c[0]=='lookup'))
    recall = [c for c in cells(split,DEV_GAPS,n=96) if c[0]=='recall']
    history,updates = [],[]
    data_hash = sha256()
    for step in range(ADAPT_STEPS+1):
        if step:
            lookup,x,y = batch(seed,step,rng,rule=1,lookup_only=True)
            data_hash.update(x.numpy().tobytes())
            data_hash.update(y.numpy().tobytes())
            structural = not freeze and step%INTERVAL==0 and mode in ('evolve','decay')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if auxiliary is not None:
                auxiliary.zero_grad(set_to_none=True)
            loss_for(model,readout,probe,lookup,x,y,structural).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if auxiliary is not None:
                auxiliary.step()
            if structural:
                updates.append(dict(step=step,**model.evolve(optimizer)))
        if step in (0,80,160,320,ADAPT_STEPS):
            history.append(dict(step=step,lookup=evaluate(model,new_tasks),recall=evaluate(model,recall),
                                active=int(model.active.sum())))
    return dict(mode='evolve_frozen' if freeze else mode,seed=seed,history=history,updates=updates,
                train_sha256=data_hash.hexdigest(),source_sha256=row['weights_sha256'])


def resource(root,row):
    probe,mode,seed = (row[k] for k in ('probe','mode','seed'))
    setup(seed)
    model = model_for(probe,mode,row['active'] if mode=='fixed' else None).eval()
    folder = root/f'{probe}-{mode}-{seed}'
    assert digest(folder/'weights.pt')==row['weights_sha256']
    model.load_state_dict(torch.load(folder/'weights.pt',weights_only=True))
    case = next(c for c in cells('capacity-holdout',(512,),n=2) if c[0]=='lookup')
    _,_,x,_ = prepare_cells(probe,[case])[0]
    return profile_model(model,x[0,:1],row)


def retrain_mask(root,probe,seed,steps,config,source):
    original = root/f'{probe}-evolve-{seed}'/'weights.pt'
    assert digest(original)==source['weights_sha256']
    mask = torch.load(original,weights_only=True)['active']
    trained = fit(root,probe,'fixed_learned',seed,steps,source['active'],WARMUP,config,mask)
    checkpoint = root/f'{probe}-fixed_learned-{seed}'/'weights.pt'
    assert torch.equal(torch.load(checkpoint,weights_only=True)['active'],mask)
    return test(root,trained) | {'mask_source_sha256':source['weights_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('probe','evolve','matched','test','adapt','diagnose','resources','retrain-mask'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--probes',choices=PROBES,nargs='+',default=PROBES[:2])
    parser.add_argument('--topology-warmup',type=int,default=WARMUP)
    parser.add_argument('--survival-rate',type=float,default=CONFIG.rate)
    parser.add_argument('--traces',type=float,nargs='+',default=())
    parser.add_argument('--dynamics',type=float,default=0.)
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--include-topology',action='store_true')
    args = parser.parse_args()
    root = args.root
    if 'capacity-curriculum' in args.probes and len(args.probes)!=1:
        parser.error('Capacity curriculum uses a different training protocol. Run it in its own directory.')
    if args.phase=='diagnose':
        diagnose(root,args.include_topology)
        return
    if args.phase=='probe':
        root.mkdir(parents=True)
        write(root/'protocol.json',dict(steps=args.steps,probes=args.probes,seeds=SEEDS,
                                       batch=24 if 'capacity-curriculum' in args.probes else BATCH,
                                       warmup=WARMUP,interval=INTERVAL,dev_gaps=DEV_GAPS))
        # Both probes see exactly the same four questions and source edits.
        jobs = [(fit,(root,p,'dense',s,args.steps)) for p,s in product(args.probes,SEEDS)]
        output = 'probes.json'
    elif args.phase=='evolve':
        protocol = read(root/'protocol.json')
        rows = read(root/'probes.json')
        probes = protocol['probes']
        assert {(r['probe'],r['seed']) for r in rows}==set(product(probes,SEEDS))
        for seed in SEEDS:
            assert len({r['train_sha256'] for r in rows if r['seed']==seed})==1
        def priority(probe):
            return mean(c['joint'] for r in rows if r['probe']==probe for c in r['history'][-1]['scores']
                        if c['task']=='lookup' and c['gap'] in TRAIN_GAPS)
        priority_by_probe = {p:priority(p) for p in probes}
        probe = max(probes,key=priority_by_probe.get)
        if priority_by_probe[probe]<.60:
            raise SystemExit('Need >= 60% development lookup joint accuracy before starting topology comparisons.')
        if not 0<=args.topology_warmup<protocol['steps']:
            parser.error('Topology warmup must be shorter than training.')
        if not 0<args.survival_rate<=1:
            parser.error('Survival rate must be in (0,1].')
        if any(h<=0 for h in args.traces) or not 0<=args.dynamics<=1:
            parser.error('Trace horizons must be positive and dynamics must be in [0,1].')
        config = replace(CONFIG,rate=args.survival_rate,traces=args.traces,dynamics=args.dynamics)
        write(root/'selection.json',dict(probe=probe,development_lookup_joint=priority_by_probe,
                                        topology_warmup=args.topology_warmup,survival=asdict(config)))
        jobs = [(fit,(root,probe,m,s,protocol['steps'],None,args.topology_warmup,config))
                for m,s in product(('evolve','decay'),SEEDS)]
        output = 'topology-controls.json'
    elif args.phase=='matched':
        protocol,probe,config = read_selected(root)
        rows = read(root/'topology-controls.json')
        assert len(rows)==6
        jobs = [(fit,(root,probe,'fixed',r['seed'],protocol['steps'],r['active'],WARMUP,config))
                for r in rows if r['mode']=='evolve']
        output = 'matched.json'
    elif args.phase=='retrain-mask':
        protocol,probe,config = read_selected(root)
        source = [r for r in read(root/'topology-controls.json') if r['mode']=='evolve']
        assert {r['seed'] for r in source}==set(SEEDS)
        jobs = [(retrain_mask,(root,probe,r['seed'],protocol['steps'],config,r)) for r in source]
        output = 'retrained-masks.json'
    else:
        _,probe,_ = read_selected(root)
        rows = [r for r in read(root/'probes.json') if r['probe']==probe]
        rows += read(root/'topology-controls.json')+read(root/'matched.json')
        assert {(r['mode'],r['seed']) for r in rows}==set(product(MODES,SEEDS)) and len(rows)==12
        for seed in SEEDS:
            assert len({r['train_sha256'] for r in rows if r['seed']==seed})==1
        if args.phase=='test':
            jobs = [(test,(root,r)) for r in rows]
            output = 'results.json'
        elif args.phase=='adapt':
            jobs = [(adapt,(root,r,freeze)) for r in rows
                    for freeze in ((False,True) if r['mode']=='evolve' else (False,))]
            output = 'adaptation.json'
        else:
            if args.mode is None:
                parser.error('resources requires --mode in an isolated process')
            row = next(r for r in rows if r['mode']==args.mode and r['seed']==2)
            write(root/f'profile-{args.mode}.json',resource(root,row))
            return
    launch(root,jobs,args.workers,output)


if __name__=='__main__':
    main()
