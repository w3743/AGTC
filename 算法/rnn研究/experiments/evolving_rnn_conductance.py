"""Separate gradual attenuation of live edges from discrete sleep and renewal."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import process_time

import torch
from torch import nn

from edge_adamw import make_optimizer
from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_audit import profile_model, sensitivity
from evolving_rnn_eval import INTERVAL, TEST_GAPS, TRAIN_GAPS, WIDTH, cells, evaluate, launch
from evolving_rnn_lookup import CONFIG, capacity_batch, fit, loss_for, lookup_queries
from evolving_rnn_sleep_guard import (CALIBRATION_SIZE, GROUPS as GUARD_GROUPS, PREFIX_DIVISORS,
                                      TOLERANCE, SleepGuard, TrainingCalibration)
from frontier_eval import SEEDS, digest, read, setup, write


PROBE = 'capacity-curriculum'
ARMS = ('dense','gradual','binary','binary-decay')
PACED_ARMS = ('dense','binary','binary-decay','paced','paced-decay')
GUARDED_ARMS = ('dense','binary','binary-decay','guarded','guarded-decay')
PREFIX_ARMS = ('dense','binary','binary-decay','guarded','prefix','prefix-decay')
CREDIT_VARIANTS = {'prefix-traces':{'traces':(2.,16.,64.)},'prefix-dynamics':{'dynamics':.15}}
CONTINUE = 1200
ADAPT = 480
SNAPSHOTS = (0,200,400,600,800,1000,CONTINUE)


def network(arm,config,active=None):
    mode = {'dense':'dense','gradual':'evolve','binary':'evolve',
            'binary-decay':'decay','paced':'evolve','paced-decay':'decay',
            'guarded':'evolve','guarded-decay':'decay','prefix':'evolve','prefix-decay':'decay',
            'prefix-traces':'evolve','prefix-dynamics':'evolve','prefix-indexed':'evolve','prefix-no-dynamics':'evolve',
            'blind':'evolve','fixed':'fixed'}[arm]
    return EvolvingRNN(WIDTH,mode,config,active if mode=='fixed' else None)


def config_for(arm,protocol):
    config = Survival(**(protocol['survival'] | protocol.get('credit_variants',{}).get(arm,{})))
    return replace(config,conductance_floor=1.) if arm not in ('dense','gradual','fixed') else config


def structural_update(model,optimizer,arm,seed,step,queue,window,rule=0,blind_count=None,calibration=None):
    gate = (SleepGuard(model,seed,step,rule,prefix=arm.startswith('prefix'),training_calibration=calibration)
            if arm.startswith(('guarded','prefix')) else None)
    def blind(eligible):
        candidates = torch.where(eligible.flatten())[0]
        ranked = model.life.flatten()[candidates].argsort(stable=True)
        cut = torch.zeros_like(eligible)
        cut.flatten()[candidates[ranked[:blind_count]]] = True
        return cut

    event = model.evolve(optimizer,queue,window,sleep_filter=blind if blind_count is not None else gate)
    if blind_count is not None:
        event['requested_sleep'] = blind_count
    if gate is not None and gate.log:
        event['guard'] = gate.log
    return dict(step=step,**event)


def update_weights(model,optimizer,auxiliary,aux_optimizer,batch,probe=False):
    lookup,x,y,count = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    loss = loss_for(model,auxiliary,PROBE,lookup,x,y,probe,count)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    aux_optimizer.step()
    return loss.item()


def replay_event(model,optimizer,auxiliary,aux_optimizer,dev,batches,seed,step):
    arms = []
    for mode in ('binary','paced','frozen'):
        candidate = network('binary',model.config)
        candidate.load_state_dict(model.state_dict())
        candidate.probe = model.probe.detach().clone().requires_grad_()
        candidate.probe.grad = model.probe.grad.detach().clone()
        opt = make_optimizer(candidate,candidate.config.independent_age)
        opt.load_state_dict(deepcopy(optimizer.state_dict()))
        aux = deepcopy(auxiliary)
        aux_opt = torch.optim.AdamW(aux.parameters(),lr=.001)
        aux_opt.load_state_dict(deepcopy(aux_optimizer.state_dict()))
        queue = []
        events = [] if mode=='frozen' else [candidate.evolve(opt,queue,INTERVAL if mode=='paced' else 1)]
        points = [dict(offset=0,active=int(candidate.active.sum()),scores=evaluate(candidate,dev))]
        for offset,batch in enumerate(batches,1):
            update_weights(candidate,opt,aux,aux_opt,batch)
            if queue:
                indices = queue.pop()
                if indices.numel():
                    events.append(candidate.finish_sleep(opt,indices))
            if offset in (4,9,INTERVAL-1,len(batches)):
                points.append(dict(offset=offset,active=int(candidate.active.sum()),scores=evaluate(candidate,dev)))
        assert not queue
        arms.append(dict(mode=mode,points=points,slept=sum(e['slept'] for e in events),
                         woke=sum(e['woke'] for e in events),mask_sha256=sha256(candidate.active.numpy().tobytes()).hexdigest()))
    immediate,paced,_ = arms
    assert immediate['mask_sha256']==paced['mask_sha256'] and immediate['slept']==paced['slept']
    return dict(mode='replay',seed=seed,step=step,before=evaluate(model,dev),arms=arms)


def continuation(root,arm,seed,protocol,source_result,replay=False,blind_counts=None):
    setup(seed)
    config = config_for(arm,protocol)
    model = network(arm,config)
    folder = root/f'{arm}-{seed}'
    if not replay:
        folder.mkdir()
    source = Path(protocol['source'])/f'{PROBE}-dense-{seed}'
    weights = source/'weights.pt'
    assert digest(weights)==source_result['weights_sha256']
    model.load_state_dict(torch.load(weights,weights_only=True))
    auxiliary = nn.Linear(WIDTH,16)
    auxiliary.load_state_dict(torch.load(source/'auxiliary.pt',weights_only=True))
    optimizer = make_optimizer(model,config.independent_age)
    aux_optimizer = torch.optim.AdamW(auxiliary.parameters(),lr=.001)
    rng = torch.Generator().manual_seed(245287+seed)
    train_hash = sha256()
    for step in range(1,protocol['pretrain_steps']+1):
        _,x,y,_ = capacity_batch(seed,step,rng)
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
    assert train_hash.hexdigest()==source_result['train_sha256']
    dev = list(cells('development',(*TRAIN_GAPS,64),n=64))
    history = [dict(step=0,active=int(model.active.sum()),scores=evaluate(model,dev))]
    updates = []
    queue = []
    calibration = (TrainingCalibration(protocol.get('calibration_groups_per_cell'))
                   if protocol.get('reuse_calibration') and arm.startswith('prefix') else None)
    weight_cpu_seconds = structure_cpu_seconds = 0.
    sleep_window = protocol['sleep_window'] if arm.startswith('paced') else 1
    final_structure_step = protocol.get('final_structure_step',CONTINUE-INTERVAL if 'sleep_window' in protocol else CONTINUE)
    if arm.startswith('prefix'):
        final_structure_step = protocol.get('consolidate_after',final_structure_step)
    delayed_observations = {}
    for step in range(1,CONTINUE+1):
        batch = capacity_batch(seed,protocol['pretrain_steps']+step,rng)
        _,x,y,_ = batch
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
        structural = arm!='dense' and step%INTERVAL==0 and step<=final_structure_step
        started = process_time()
        loss = update_weights(model,optimizer,auxiliary,aux_optimizer,batch,structural)
        if calibration is not None:
            calibration.add(batch[0],x,y)
        weight_cpu_seconds += process_time()-started
        if replay and step==protocol['replay_steps'][str(seed)]:
            batches = [capacity_batch(seed,protocol['pretrain_steps']+step+offset,rng)
                       for offset in range(1,2*INTERVAL)]
            return replay_event(model,optimizer,auxiliary,aux_optimizer,dev,batches,seed,step)
        if structural:
            if step in SNAPSHOTS:
                pre = evaluate(model,dev)
            count = None if blind_counts is None else blind_counts[step]
            started = process_time()
            updates.append(structural_update(model,optimizer,arm,seed,step,queue,sleep_window,
                                             blind_count=count,calibration=calibration))
            structure_cpu_seconds += process_time()-started
        elif queue:
            indices = queue.pop()
            if indices.numel():
                updates.append(dict(step=step,**model.finish_sleep(optimizer,indices)))
        if step in SNAPSHOTS:
            scores = evaluate(model,dev)
            snapshot = dict(step=step,active=int(model.active.sum()),scores=scores,loss=loss)
            if structural:
                snapshot['before_structure'] = pre
                if 'sleep_window' in protocol:
                    delayed_observations[step+INTERVAL-1] = snapshot
            history.append(snapshot)
            write(folder/'development.json',history)
            print(arm,seed,step,snapshot['active'],[round(c['joint'],3) for c in scores],flush=True)
        if step in delayed_observations:
            snapshot = delayed_observations.pop(step)
            snapshot['window_end'] = dict(step=step,active=int(model.active.sum()),scores=evaluate(model,dev))
    assert not queue and not delayed_observations
    assert not replay
    torch.save(model.state_dict(),folder/'weights.pt')
    torch.save(auxiliary.state_dict(),folder/'auxiliary.pt')
    if protocol.get('reuse_calibration'):
        if arm=='dense':
            torch.save((optimizer.state_dict(),aux_optimizer.state_dict()),folder/'optimizers.pt')
        if calibration is not None:
            torch.save(calibration.batches,folder/'calibration.pt')
    write(folder/'topology.json',updates)
    return dict(mode=arm,seed=seed,active=int(model.active.sum()),source_sha256=digest(weights),
                train_sha256=train_hash.hexdigest(),history=history,
                weights_sha256=digest(folder/'weights.pt'),training_cpu_seconds=weight_cpu_seconds+structure_cpu_seconds,
                weight_cpu_seconds=weight_cpu_seconds,structure_cpu_seconds=structure_cpu_seconds)


def fit_fixed(root,row,protocol):
    steps = protocol['pretrain_steps']+CONTINUE
    result = fit(root,PROBE,'fixed',row['seed'],steps,row['active'],
                 config=config_for('fixed',protocol),reset_after=protocol['pretrain_steps'],
                 save_training_state=protocol.get('reuse_calibration',False))
    folder = root/f'{PROBE}-fixed-{row["seed"]}'
    folder.rename(root/f'fixed-{row["seed"]}')
    return result


def restore(root,row,protocol):
    seed,arm = row['seed'],row['mode']
    setup(seed)
    model = network(arm,config_for(arm,protocol),row['active'])
    folder = root/f'{arm}-{seed}'
    weights = folder/'weights.pt'
    assert digest(weights)==row['weights_sha256']
    model.load_state_dict(torch.load(weights,weights_only=True))
    return model.eval(),folder


def frozen_test(root,row,protocol):
    model,folder = restore(root,row,protocol)
    partition = protocol.get('holdout_partition','conductance-holdout')
    measured = evaluate(model,list(cells(partition,TEST_GAPS)),save_predictions=True)
    write(folder/'holdout.json',measured)
    return row | {'holdout':[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in measured]}


def previous_test(root,source,row,reference_protocol,partition):
    model,_ = restore(source,row,reference_protocol)
    folder = root/f'previous-{row["seed"]}'
    folder.mkdir()
    measured = evaluate(model,list(cells(partition,TEST_GAPS)),save_predictions=True)
    write(folder/'holdout.json',measured)
    return row | {'mode':'previous','checkpoint_root':str(source.resolve()),
                  'holdout':[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in measured]}


def compute_matched(root,row,reference,protocol,tag='cpu'):
    """Spend the remaining CPU budget on BPTT over the same observed batches."""
    model,source = restore(root,row,protocol)
    optimizer = make_optimizer(model,model.config.independent_age)
    auxiliary = nn.Linear(WIDTH,16)
    auxiliary.load_state_dict(torch.load(source/'auxiliary.pt',weights_only=True))
    aux_optimizer = torch.optim.AdamW(auxiliary.parameters(),lr=.001)
    optimizer_state,aux_state = torch.load(source/'optimizers.pt',weights_only=True)
    optimizer.load_state_dict(optimizer_state)
    aux_optimizer.load_state_dict(aux_state)
    seed = row['seed']
    rng = torch.Generator().manual_seed(245287+seed)
    train_hash = sha256()
    replay = []
    for step in range(1,protocol['pretrain_steps']+CONTINUE+1):
        batch = capacity_batch(seed,step,rng)
        _,x,y,_ = batch
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
        if step>protocol['pretrain_steps']:
            replay.append(batch)
    assert train_hash.hexdigest()==row['train_sha256']==reference['train_sha256']
    initial = row['training_cpu_seconds']
    target = reference['training_cpu_seconds']
    steps = 0
    started = process_time()
    while initial+process_time()-started<target:
        update_weights(model,optimizer,auxiliary,aux_optimizer,replay[steps%len(replay)])
        steps += 1
    elapsed = process_time()-started
    mode = f'{row["mode"]}-{tag}'
    folder = root/f'{mode}-{seed}'
    folder.mkdir()
    torch.save(model.state_dict(),folder/'weights.pt')
    measured = evaluate(model,list(cells(protocol['holdout_partition'],TEST_GAPS)),save_predictions=True)
    write(folder/'holdout.json',measured)
    return dict(mode=mode,seed=seed,active=int(model.active.sum()),train_sha256=train_hash.hexdigest(),
                source_sha256=row['weights_sha256'],weights_sha256=digest(folder/'weights.pt'),
                base_cpu_seconds=initial,extra_cpu_seconds=elapsed,target_cpu_seconds=target,extra_steps=steps,
                holdout=[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in measured])


def cut_lookup(root,row,protocol):
    model,_ = restore(root,row,protocol)
    development = {c[0]:c for c in cells('development',(16,),n=64)}
    score = sensitivity(model,development['lookup'])-sensitivity(model,development['recall'])
    active = torch.where(model.active.flatten())[0]
    chosen = active[score.flatten()[active].argsort(descending=True,stable=True)[:round(.01*len(active))]]
    cut = torch.zeros_like(model.active)
    cut.flatten()[chosen] = True
    partition = protocol.get('holdout_partition','conductance-holdout')
    data = list(cells(partition,(4,16,32,128),n=96))
    rng = torch.Generator().manual_seed(row['seed']+760889)
    random = []
    for _ in range(5):
        control = torch.zeros_like(cut)
        for i in torch.where(cut.any(1))[0].tolist():
            available = torch.where(model.active[i])[0]
            indices = torch.randperm(len(available),generator=rng)[:int(cut[i].sum())]
            control[i,available[indices]] = True
        random.append(evaluate(model,data,cut=control))
    return dict(mode=row['mode'],seed=row['seed'],indices=chosen.tolist(),base=evaluate(model,data),
                targeted=evaluate(model,data,cut=cut),random=random)


def cut_time_scales(root,row,protocol):
    model,_ = restore(root,row,protocol)
    short,long = (c for c in cells('prefix-cut-probe',(4,128),n=48) if c[0]=='recall')
    difference = sensitivity(model,long)-sensitivity(model,short)
    quota = (.01*model.active.sum(1)).round().long()

    def choose(score,allowed):
        order = score.masked_fill(~allowed,-torch.inf).argsort(dim=1,descending=True,stable=True)
        selected = torch.arange(order.shape[1])[None,:]<quota[:,None]
        return torch.zeros_like(model.active).scatter_(1,order,selected)

    long_cut = choose(difference,model.active)
    short_cut = choose(-difference,model.active & ~long_cut)
    assert not (long_cut & short_cut).any()
    rng = torch.Generator().manual_seed(row['seed']+1428653)
    random = [choose(torch.rand(difference.shape,generator=rng),model.active) for _ in range(5)]
    data = [c for c in cells(protocol['holdout_partition'],(4,32,128,256,512),n=128) if c[0]=='recall']
    masks = [long_cut,short_cut,*random]
    for mask in masks:
        assert torch.equal(mask.sum(1),quota) and not (mask & ~model.active).any()
    return dict(mode=row['mode'],seed=row['seed'],count=int(quota.sum()),base=evaluate(model,data),
                long=evaluate(model,data,cut=long_cut),short=evaluate(model,data,cut=short_cut),
                random=[evaluate(model,data,cut=cut) for cut in random],
                indices=[torch.where(mask.flatten())[0].tolist() for mask in masks])


def audit_modules(root,row,protocol):
    from evolving_rnn_modules import audit_modules as measure

    model,_ = restore(root,row,protocol)
    return dict(mode=row['mode'],seed=row['seed'],**measure(model,row['seed'],protocol['holdout_partition']))


def adapt(root,row,protocol,freeze,disable_wake=False,save_final=False,save_state=False):
    model,folder = restore(root,row,protocol)
    if disable_wake:
        model.mode = 'decay'
    optimizer = make_optimizer(model,model.config.independent_age)
    auxiliary = nn.Linear(WIDTH,16)
    auxiliary.load_state_dict(torch.load(folder/'auxiliary.pt',weights_only=True))
    aux_optimizer = torch.optim.AdamW(auxiliary.parameters(),lr=.001)
    seed,arm = row['seed'],row['mode']
    suffix = '-frozen' if freeze else '-no-wake' if disable_wake else ''
    cpu_seconds = 0.
    rng = torch.Generator().manual_seed(seed+850071)
    partition = protocol.get('adaptation_partition',
                             'pacing-adaptation' if 'sleep_window' in protocol else 'conductance-adaptation')
    shifted = [c for c in cells(partition,(*TRAIN_GAPS,64),n=96,rule=1) if c[0]=='lookup']
    recall = [c for c in cells(partition,TRAIN_GAPS,n=96) if c[0]=='recall']
    points,updates = [],[]
    queue = []
    calibration = None
    if protocol.get('reuse_calibration') and arm.startswith('prefix') and not freeze:
        calibration = TrainingCalibration(protocol.get('calibration_groups_per_cell'))
        calibration.batches = torch.load(folder/'calibration.pt',weights_only=True)
    window = protocol['sleep_window'] if arm.startswith('paced') else 1
    last = ADAPT-INTERVAL if 'sleep_window' in protocol or 'sleep_guard' in protocol else ADAPT
    data_hash = sha256()
    for step in range(ADAPT+1):
        if step:
            x,y = lookup_queries(TRAIN_GAPS[(step+seed)%4],3,rng,rule=1)
            data_hash.update(x.numpy().tobytes())
            data_hash.update(y.numpy().tobytes())
            structural = not freeze and arm not in ('dense','fixed') and step%INTERVAL==0 and step<=last
            started = process_time()
            update_weights(model,optimizer,auxiliary,aux_optimizer,(True,x,y,4),structural)
            if calibration is not None:
                calibration.add(True,x,y)
            if structural:
                updates.append(structural_update(model,optimizer,arm,seed,step,queue,window,rule=1,
                                                 calibration=calibration))
            elif queue:
                indices = queue.pop()
                if indices.numel():
                    updates.append(dict(step=step,**model.finish_sleep(optimizer,indices)))
            cpu_seconds += process_time()-started
        if step in (0,80,160,320,ADAPT):
            lookup = evaluate(model,shifted,save_predictions=save_final and step==ADAPT)
            if save_final and step==ADAPT:
                write(root/f'adapt-final-{arm+suffix}-{seed}.json',lookup)
                lookup = [{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in lookup]
            points.append(dict(step=step,lookup=lookup,recall=evaluate(model,recall),
                               active=int(model.active.sum())))
    assert not queue
    result = dict(mode=arm+suffix,seed=seed,train_sha256=data_hash.hexdigest(),
                  source_sha256=row['weights_sha256'],points=points,updates=updates)
    if save_state:
        path = root/f'adapt-state-{arm+suffix}-{seed}.pt'
        torch.save(dict(model=model.state_dict(),auxiliary=auxiliary.state_dict(),optimizer=optimizer.state_dict(),
                        auxiliary_optimizer=aux_optimizer.state_dict(),noise=model.noise.get_state(),
                        calibration=None if calibration is None else calibration.batches),path)
        result.update(cpu_seconds=cpu_seconds,state_sha256=digest(path))
    return result


def compute_adaptation(root,row,protocol,target_cpu):
    """Continue the three adaptation policies over the same already seen questions."""
    mode,seed = row['mode'],row['seed']
    setup(seed)
    model = network('prefix',config_for('prefix',protocol))
    if mode=='prefix-no-wake':
        model.mode = 'decay'
    state_path = root/f'adapt-state-{mode}-{seed}.pt'
    assert digest(state_path)==row['state_sha256']
    state = torch.load(state_path,weights_only=True)
    model.load_state_dict(state['model'])
    model.noise.set_state(state['noise'])
    auxiliary = nn.Linear(WIDTH,16)
    auxiliary.load_state_dict(state['auxiliary'])
    optimizer = make_optimizer(model,model.config.independent_age)
    optimizer.load_state_dict(state['optimizer'])
    aux_optimizer = torch.optim.AdamW(auxiliary.parameters(),lr=.001)
    aux_optimizer.load_state_dict(state['auxiliary_optimizer'])
    calibration = None
    if state['calibration'] is not None:
        calibration = TrainingCalibration(protocol.get('calibration_groups_per_cell'))
        calibration.batches = state['calibration']
    rng = torch.Generator().manual_seed(seed+850071)
    replay = [lookup_queries(TRAIN_GAPS[(step+seed)%4],3,rng,rule=1) for step in range(1,ADAPT+1)]
    data_hash = sha256()
    for x,y in replay:
        data_hash.update(x.numpy().tobytes())
        data_hash.update(y.numpy().tobytes())
    assert data_hash.hexdigest()==row['train_sha256']
    extra_steps = 0
    events = []
    initial = row['cpu_seconds']
    started = process_time()
    while initial+process_time()-started<target_cpu:
        x,y = replay[extra_steps%len(replay)]
        extra_steps += 1
        step = ADAPT+extra_steps
        structural = mode!='prefix-frozen' and step%INTERVAL==0
        update_weights(model,optimizer,auxiliary,aux_optimizer,(True,x,y,4),structural)
        if calibration is not None:
            calibration.add(True,x,y)
        if structural:
            events.append(structural_update(model,optimizer,'prefix',seed,step,[],1,rule=1,calibration=calibration))
    elapsed = process_time()-started
    partition = protocol['adaptation_partition']
    shifted = [c for c in cells(partition,(*TRAIN_GAPS,64),n=96,rule=1) if c[0]=='lookup']
    recall = [c for c in cells(partition,TRAIN_GAPS,n=96) if c[0]=='recall']
    lookup = evaluate(model,shifted,save_predictions=True)
    write(root/f'adapt-cpu-final-{mode}-{seed}.json',lookup)
    return dict(mode=mode,seed=seed,train_sha256=row['train_sha256'],state_sha256=row['state_sha256'],
                base_cpu_seconds=initial,extra_cpu_seconds=elapsed,target_cpu_seconds=target_cpu,extra_steps=extra_steps,
                lookup=[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in lookup],
                recall=evaluate(model,recall),active=int(model.active.sum()),updates=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('branch','replay','matched','test','adapt','adapt-audit','adapt-compute','cuts','time-cuts','time-cuts-decay',
                                         'resources','blind','blind-test','credit','credit-test','compute','modules',
                                         'previous','indexed','compute-indexed'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--mode',choices=tuple(dict.fromkeys((*ARMS,*PACED_ARMS,*GUARDED_ARMS,*PREFIX_ARMS,'fixed'))))
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--sleep-window',type=int,choices=(INTERVAL,))
    experiment.add_argument('--sleep-guard',action='store_true')
    experiment.add_argument('--sleep-prefix',action='store_true')
    experiment.add_argument('--reuse-calibration',action='store_true')
    experiment.add_argument('--thin-calibration',action='store_true')
    experiment.add_argument('--thin-dynamics',action='store_true')
    parser.add_argument('--consolidate-after',type=int,choices=range(INTERVAL,CONTINUE,INTERVAL),metavar='STEP')
    args = parser.parse_args()
    root = args.root
    if args.phase=='branch':
        if args.source is None:
            parser.error('branch requires --source')
        if args.consolidate_after is not None and not args.thin_dynamics:
            parser.error('consolidation currently requires --thin-dynamics')
        source = read(args.source/'probes.json')
        assert len(source)==3 and {(r['probe'],r['mode'],r['seed']) for r in source}=={
            (PROBE,'dense',s) for s in SEEDS}
        assert len({r['steps'] for r in source})==1
        root.mkdir(parents=True)
        thin = args.thin_calibration or args.thin_dynamics
        reuse = args.reuse_calibration or thin
        prefix = args.sleep_prefix or reuse
        arms = PREFIX_ARMS if prefix else GUARDED_ARMS if args.sleep_guard else PACED_ARMS if args.sleep_window else ARMS
        protocol = dict(source=str(args.source.resolve()),pretrain_steps=source[0]['steps'],continuation_steps=CONTINUE,
                        survival=asdict(replace(CONFIG,rate=.08)),seeds=SEEDS,arms=arms,interval=INTERVAL)
        if args.sleep_window:
            protocol.update(sleep_window=args.sleep_window,holdout_partition='pacing-holdout')
        if args.sleep_guard or prefix:
            protocol.update(sleep_guard=dict(groups=GUARD_GROUPS,samples_per_gap=CALIBRATION_SIZE,tolerance=TOLERANCE),
                            final_structure_step=CONTINUE-INTERVAL,holdout_partition='guard-holdout',
                            adaptation_partition='guard-adaptation',selected='guarded')
        if prefix:
            protocol['sleep_guard']['prefix_divisors'] = PREFIX_DIVISORS
            protocol.update(holdout_partition='prefix-holdout',adaptation_partition='prefix-adaptation',selected='prefix')
        if reuse:
            protocol.update(reuse_calibration=True,holdout_partition='reuse-holdout',adaptation_partition='reuse-adaptation')
        if thin:
            protocol.update(calibration_groups_per_cell=1,holdout_partition='thin-holdout',
                            adaptation_partition='thin-adaptation')
        if args.thin_dynamics:
            protocol.update(credit_variants={arm:CREDIT_VARIANTS['prefix-dynamics'] for arm in ('prefix','prefix-decay')},
                            holdout_partition='thin-dynamics-holdout',adaptation_partition='thin-dynamics-adaptation')
        if args.consolidate_after is not None:
            protocol.update(consolidate_after=args.consolidate_after,holdout_partition='consolidation-holdout',
                            adaptation_partition='consolidation-adaptation')
        write(root/'protocol.json',protocol)
        jobs = [(continuation,(root,arm,r['seed'],protocol,r)) for arm,r in product(arms,source)]
        destination = 'development.json'
    else:
        protocol = read(root/'protocol.json')
        rows = read(root/'development.json')
        assert {(r['mode'],r['seed']) for r in rows}==set(product(protocol['arms'],SEEDS))
        selected = protocol.get('selected','paced' if 'sleep_window' in protocol else 'binary')
        if args.phase=='replay':
            assert 'sleep_window' in protocol
            sources = read(Path(protocol['source'])/'probes.json')
            protocol['replay_steps'] = {}
            for seed in SEEDS:
                events = read(root/f'binary-{seed}'/'topology.json')
                protocol['replay_steps'][str(seed)] = next(e['step'] for e in events if e['slept']>=5000)
            jobs = [(continuation,(root,'binary',r['seed'],protocol,r,True)) for r in sources]
            destination = 'replay.json'
        elif args.phase=='blind':
            assert 'sleep_guard' in protocol
            sources = read(Path(protocol['source'])/'probes.json')
            jobs = []
            for row in sources:
                seed = row['seed']
                counts = {e['step']:e['slept'] for e in read(root/f'guarded-{seed}'/'topology.json')}
                jobs.append((continuation,(root,'blind',seed,protocol,row,False,counts)))
            destination = 'blind-development.json'
        elif args.phase=='blind-test':
            blind = read(root/'blind-development.json')
            assert {r['seed'] for r in blind}==set(SEEDS)
            by_seed = {r['seed']:r for r in rows if r['mode']=='guarded'}
            for row in blind:
                for key in ('source_sha256','train_sha256'):
                    assert row[key]==by_seed[row['seed']][key]
            jobs = [(frozen_test,(root,row,protocol)) for row in blind]
            destination = 'blind-results.json'
        elif args.phase in ('credit','credit-test'):
            assert selected=='prefix'
            path = root/'credit-protocol.json'
            if args.phase=='credit':
                if protocol.get('consolidate_after'):
                    dynamics = protocol['credit_variants'][selected]['dynamics']
                    protocol['credit_variants'] = {
                        'prefix-no-dynamics':{'dynamics':0.},
                        'prefix-traces':{'dynamics':dynamics,**CREDIT_VARIANTS['prefix-traces']},
                    }
                else:
                    protocol['credit_variants'] = CREDIT_VARIANTS
                write(path,protocol)
                sources = read(Path(protocol['source'])/'probes.json')
                jobs = [(continuation,(root,mode,r['seed'],protocol,r))
                        for mode,r in product(protocol['credit_variants'],sources)]
                destination = 'credit-development.json'
            else:
                protocol = read(path)
                variants = read(root/'credit-development.json')
                assert {(r['mode'],r['seed']) for r in variants}==set(product(protocol['credit_variants'],SEEDS))
                by_seed = {r['seed']:r for r in rows if r['mode']==selected}
                for row in variants:
                    for key in ('source_sha256','train_sha256'):
                        assert row[key]==by_seed[row['seed']][key]
                jobs = [(frozen_test,(root,row,protocol)) for row in variants]
                destination = 'credit-results.json'
        elif args.phase=='indexed':
            assert protocol.get('calibration_groups_per_cell')==1
            sources = read(Path(protocol['source'])/'probes.json')
            jobs = [(continuation,(root,'prefix-indexed',r['seed'],protocol,r)) for r in sources]
            destination = 'indexed-results.json'
        elif args.phase=='previous':
            assert protocol.get('calibration_groups_per_cell')==1 and args.source is not None
            reference_protocol = read(args.source/'protocol.json')
            assert reference_protocol.get('reuse_calibration')
            dynamics = protocol.get('credit_variants',{}).get(selected,{}).get('dynamics')
            consolidation = protocol.get('consolidate_after')
            if consolidation:
                assert not reference_protocol.get('consolidate_after')
                assert reference_protocol['credit_variants']==protocol['credit_variants']
                assert reference_protocol.get('calibration_groups_per_cell')==1
            elif dynamics:
                assert reference_protocol.get('calibration_groups_per_cell')==1
                assert not reference_protocol.get('credit_variants',{}).get(selected)
            else:
                assert not reference_protocol.get('calibration_groups_per_cell')
            assert all(reference_protocol[k]==protocol[k] for k in ('survival','pretrain_steps','continuation_steps','interval'))
            reference = [r for r in read(args.source/'development.json') if r['mode']==selected]
            if dynamics and not consolidation:
                indexed = {r['seed']:r for r in read(args.source/'indexed-results.json')}
                assert set(indexed)==set(SEEDS)
                for r in reference:
                    equivalent = indexed[r['seed']]
                    assert all(r[k]==equivalent[k] for k in ('source_sha256','train_sha256','weights_sha256','history'))
                    r['training_cpu_seconds'] = equivalent['training_cpu_seconds']
                    r['cpu_reference'] = equivalent['mode']
            by_seed = {r['seed']:r for r in rows if r['mode']==selected}
            assert {r['seed'] for r in reference}==set(SEEDS)
            for row in reference:
                current = by_seed[row['seed']]
                assert all(row[k]==current[k] for k in ('source_sha256','train_sha256'))
                if consolidation:
                    old = read(args.source/f'{selected}-{row["seed"]}'/'topology.json')
                    new = read(root/f'{selected}-{row["seed"]}'/'topology.json')
                    assert new==[e for e in old if e['step']<=consolidation]
            jobs = [(previous_test,(root,args.source,row,reference_protocol,protocol['holdout_partition'])) for row in reference]
            destination = 'previous-results.json'
        elif args.phase=='matched':
            jobs = [(fit_fixed,(root,r,protocol)) for r in rows if r['mode']==selected]
            destination = 'matched.json'
        else:
            rows += read(root/'matched.json')
            for seed in SEEDS:
                assert len({r['train_sha256'] for r in rows if r['seed']==seed})==1
            if args.phase=='test':
                jobs = [(frozen_test,(root,row,protocol)) for row in rows]
                destination = 'results.json'
            elif args.phase in ('compute','compute-indexed'):
                assert protocol.get('reuse_calibration')
                reference = {r['seed']:r for r in rows if r['mode']==selected}
                tag = 'cpu'
                if args.phase=='compute-indexed':
                    indexed = read(root/'indexed-results.json')
                    assert {r['seed'] for r in indexed}==set(SEEDS)
                    for r in indexed:
                        original = reference[r['seed']]
                        assert all(r[k]==original[k] for k in ('train_sha256','weights_sha256','history'))
                    reference = {r['seed']:r for r in indexed}
                    tag = 'indexed-cpu'
                jobs = [(compute_matched,(root,r,reference[r['seed']],protocol,tag)) for r in rows
                        if r['mode'] in ('dense','fixed')]
                destination = args.phase+'-results.json'
            elif args.phase=='adapt':
                jobs = []
                for row in rows:
                    primary = row['mode']==selected
                    save = primary and selected=='prefix'
                    checkpoint = save and bool(protocol.get('consolidate_after'))
                    jobs.extend((adapt,(root,row,protocol,freeze,False,save,checkpoint))
                                for freeze in ((False,True) if primary else (False,)))
                    if save:
                        jobs.append((adapt,(root,row,protocol,False,True,True,checkpoint)))
                destination = 'adaptation.json'
            elif args.phase=='adapt-audit':
                assert selected=='prefix'
                jobs = [(adapt,(root,row,protocol,freeze,no_wake,True)) for row in rows if row['mode']==selected
                        for freeze,no_wake in ((False,False),(True,False),(False,True))]
                destination = 'adaptation-audit.json'
            elif args.phase=='adapt-compute':
                assert protocol.get('consolidate_after')
                modes = ('prefix','prefix-no-wake','prefix-frozen')
                adaptation = [r for r in read(root/'adaptation.json') if r['mode'] in modes]
                assert {(r['mode'],r['seed']) for r in adaptation}==set(product(modes,SEEDS))
                budgets = {seed:max(r['cpu_seconds'] for r in adaptation if r['seed']==seed) for seed in SEEDS}
                jobs = [(compute_adaptation,(root,r,protocol,budgets[r['seed']])) for r in adaptation]
                destination = 'adapt-compute-results.json'
            elif args.phase=='cuts':
                jobs = [(cut_lookup,(root,row,protocol)) for row in rows if row['mode'] in ('dense','fixed',selected)]
                destination = 'cuts.json'
            elif args.phase=='modules':
                jobs = [(audit_modules,(root,row,protocol)) for row in rows if row['mode'] in ('dense','fixed',selected)]
                destination = 'modules.json'
            elif args.phase in ('time-cuts','time-cuts-decay'):
                assert selected=='prefix'
                modes = ('dense','fixed',selected) if args.phase=='time-cuts' else ('prefix-decay',)
                jobs = [(cut_time_scales,(root,row,protocol)) for row in rows if row['mode'] in modes]
                destination = args.phase+'.json'
            else:
                if args.mode is None:
                    parser.error('resources requires --mode')
                row = next(r for r in rows if r['seed']==2 and r['mode']==args.mode)
                model,_ = restore(root,row,protocol)
                partition = protocol.get('holdout_partition','conductance-holdout')
                x = next(x[0,:1] for task,_,x,_ in cells(partition,(512,),n=2) if task=='lookup')
                write(root/f'profile-{args.mode}.json',profile_model(model,x,row))
                return
    launch(root,jobs,args.workers,destination)


if __name__=='__main__':
    main()
