"""Paired, matched-budget memory evaluation for a dense-start evolving RNN."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from edge_adamw import make_optimizer

from evolving_rnn import EvolvingRNN, Survival
from frontier_eval import SEEDS, digest, read, setup, write


TASKS = ('recall','lookup')
MODES = ('dense','decay','evolve','fixed')
WIDTH = 512
BATCH = 16
WARMUP = 240
INTERVAL = 20
STEPS = 1000
TRAIN_GAPS = (0,4,16,32)
TEST_GAPS = (*TRAIN_GAPS,64,128,256,512)
GROUPS = 128


def questions(task,gap,n,rng,rule=0,record_count=4):
    """Return original, changed cue/query and source-edit variants."""
    noise = torch.randint(4,6,(n,gap),generator=rng)
    if task=='recall':
        values = torch.randint(4,(n,),generator=rng)
        other = (values+torch.randint(1,4,(n,),generator=rng))%4
        x = torch.cat((torch.full((n,1),6),values[:,None],noise,torch.full((n,1),7)),dim=1)
        changed = x.clone()
        changed[:,1] = other
        return torch.stack((x,changed),dim=1),torch.stack((values,other),dim=1)
    # Randomize the record order and ensure the two queried values differ.
    order = torch.rand((n,4),generator=rng).argsort(dim=1)
    query = torch.randint(4,(n,),generator=rng)
    swap = (query+torch.randint(1,4,(n,),generator=rng))%4
    source,alt = (query+rule)%4,(swap+rule)%4
    if record_count<4:
        if record_count<2:
            raise ValueError('Lookup needs at least two records for the query counterfactual.')
        required = (order==source[:,None]) | (order==alt[:,None])
        chosen = required.long().argsort(dim=1,descending=True,stable=True)[:,:record_count].sort(dim=1).values
        order = order.gather(1,chosen)
    row = torch.arange(n)
    values = torch.randint(4,(n,4),generator=rng)
    v = values[row,source].clone()
    v2 = (v+torch.randint(1,4,(n,),generator=rng))%4
    values[row,alt] = v2
    records = torch.stack((8+order,values.gather(1,order)),dim=-1).flatten(1)
    x = torch.cat((torch.full((n,1),6),records,noise,torch.full((n,1),7),8+query[:,None]),dim=1)
    changed = x.clone()
    changed[:,-1] = 8+swap
    edited = x.clone()
    edited_value = (v+torch.randint(1,4,(n,),generator=rng))%4
    at = (order==source[:,None]).long().argmax(1)
    edited[row,2+2*at] = edited_value
    return torch.stack((x,changed,edited),dim=1),torch.stack((v,v2,edited_value),dim=1)


def cells(partition,gaps,n=GROUPS,rule=0):
    namespace = {'development':47281,'holdout':98137,'adaptation':27942,'calibration':73961,
                 'capacity-holdout':604781,'capacity-adaptation':698517,
                 'conductance-holdout':743971,'conductance-adaptation':849041,
                 'pacing-holdout':962821,'pacing-adaptation':1024097,
                 'guard-holdout':1107187,'guard-adaptation':1160989,
                 'prefix-holdout':1236811,'prefix-adaptation':1291949,'prefix-cut-probe':1361731,
                 'reuse-holdout':1430179,'reuse-adaptation':1496317,
                 'thin-holdout':1568791,'thin-adaptation':1632079,
                 'thin-dynamics-holdout':1702487,'thin-dynamics-adaptation':1761961,
                 'consolidation-holdout':1833917,'consolidation-adaptation':1897303}[partition]
    for task_index,task in enumerate(TASKS):
        for gap in gaps:
            rng = torch.Generator().manual_seed(namespace+10000*task_index+gap)
            yield task,gap,*questions(task,gap,n,rng,rule)


@torch.inference_mode()
def evaluate(model,sets,cut=None,save_predictions=False):
    model.eval()
    result = []
    for task,gap,x,y in sets:
        logits = torch.cat([model(part,cut=cut) for part in x.flatten(0,1).split(24)])
        probs = logits.softmax(-1).reshape(*y.shape,4)
        hits = probs.argmax(-1)==y
        result.append(dict(task=task,gap=gap,joint=float(hits.all(1).float().mean()),
                           single=float(hits[:,0].float().mean())))
        if save_predictions:
            result[-1].update(targets=y.tolist(),probabilities=probs.tolist())
    return result


def batch(seed,step,rng,rule=0,curriculum=False,preview=False):
    task = TASKS[(step+seed)%2] if not rule else 'lookup'
    gaps = (0,4) if step<=160 and not rule else TRAIN_GAPS
    gap = gaps[((step-1)//2+seed)%len(gaps)]
    records = 4
    if curriculum and step<=1200:
        task = 'lookup'
        records = 2 if step<=600 else 3 if step<=900 else 4
        gap = 0 if step<=600 else (0,4)[step%2]
    x,y = questions(task,gap,BATCH//2,rng,rule,records)
    if preview and step<=1200 and (step<=480 or step%3<(2 if step<=900 else 1)):
        x = torch.cat((x[:,:,:1],x[:,:,-1:],x[:,:,1:]),dim=-1)
    # Every update includes one counterfactual with the same distractors.
    alt = 1+((torch.arange(len(x))+step)%(x.shape[1]-1))
    pick = torch.stack((torch.zeros_like(alt),alt),dim=1)
    idx = torch.arange(len(x))[:,None]
    return x[idx,pick].flatten(0,1),y[idx,pick].flatten()


def fit(root,mode,seed,steps,config,matched=None,curriculum=False,preview=False):
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,config,matched)
    optimizer = make_optimizer(model,config.independent_age)
    rng = torch.Generator().manual_seed(38041+seed)
    history,updates = [],[]
    train_hash = sha256()
    run = root/f'{mode}-{seed}'
    run.mkdir()
    dev = list(cells('development',TRAIN_GAPS,n=48))
    loss_sum = loss_count = 0
    for step in range(1,steps+1):
        x,y = batch(seed,step,rng,curriculum=curriculum,preview=preview)
        train_hash.update(x.numpy().tobytes())
        train_hash.update(y.numpy().tobytes())
        model.train()
        structural = step>WARMUP and step%INTERVAL==0 and mode in ('decay','evolve')
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x,probe=structural),y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if structural:
            updates.append(dict(step=step,**model.evolve(optimizer)))
        loss_sum += loss.item()
        loss_count += 1
        if step in (160,WARMUP,400,600,800,1200,1800) or step==steps:
            dev_scores = evaluate(model,dev)
            point = dict(step=step,loss=loss_sum/loss_count,active=int(model.active.sum()),scores=dev_scores)
            history.append(point)
            write(run/'development.json',history)
            print(mode,seed,step,round(point['loss'],3),point['active'],
                  [round(c['joint'],2) for c in dev_scores],flush=True)
            loss_sum = loss_count = 0
    weights = run/'weights.pt'
    torch.save(model.state_dict(),weights)
    write(run/'topology.json',updates)
    result = dict(mode=mode,seed=seed,steps=steps,active=int(model.active.sum()),weights_sha256=digest(weights),
                  train_sha256=train_hash.hexdigest(),history=history)
    write(run/'result.json',result)
    return result


def evaluate_frozen(root,result,config):
    seed,mode = result['seed'],result['mode']
    setup(seed)
    model = EvolvingRNN(WIDTH,mode,config,result['active'] if mode=='fixed' else None)
    run = root/f'{mode}-{seed}'
    weights = run/'weights.pt'
    assert digest(weights)==result['weights_sha256']
    model.load_state_dict(torch.load(weights,weights_only=True))
    scored = evaluate(model,list(cells('holdout',TEST_GAPS)),save_predictions=True)
    write(run/'holdout.json',scored)
    return result | {'holdout':[{k:v for k,v in c.items() if k not in ('targets','probabilities')} for c in scored]}


def launch(root,jobs,workers,destination):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(function,*args) for function,args in jobs]
        for task in as_completed(tasks):
            results.append(task.result())
            write(root/destination,sorted(results,key=lambda r:(r['mode'],r['seed'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('train','matched','test'))
    parser.add_argument('output',type=Path)
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--dormant-credit',type=float,default=Survival.dormant_credit)
    parser.add_argument('--cost',type=float,default=Survival.cost)
    parser.add_argument('--traces',type=float,nargs='+',default=())
    parser.add_argument('--dynamics',type=float,default=0.)
    parser.add_argument('--independent-age',action='store_true')
    parser.add_argument('--lookup-curriculum',action='store_true')
    parser.add_argument('--lookup-preview',action='store_true')
    parser.add_argument('--modes',choices=MODES[:-1],nargs='+',default=MODES[:-1])
    args = parser.parse_args()
    if args.lookup_preview and not args.lookup_curriculum:
        parser.error('--lookup-preview requires --lookup-curriculum')
    root = args.output
    if args.phase=='train':
        root.mkdir(parents=True)
        config = Survival(cost=args.cost,dormant_credit=args.dormant_credit,traces=args.traces,dynamics=args.dynamics,
                          independent_age=args.independent_age)
        write(root/'protocol.json',dict(seeds=SEEDS,width=WIDTH,batch=BATCH,steps=args.steps,
              warmup=WARMUP,interval=INTERVAL,train_gaps=TRAIN_GAPS,test_gaps=TEST_GAPS,groups=GROUPS,
              survival=asdict(config),lookup_curriculum=args.lookup_curriculum,lookup_preview=args.lookup_preview,
              torch=torch.__version__))
        jobs = [(fit,(root,mode,seed,args.steps,config,None,args.lookup_curriculum,args.lookup_preview))
                for mode,seed in product(args.modes,SEEDS)]
        destination = 'development.json'
    else:
        protocol = read(root/'protocol.json')
        config = Survival(**protocol['survival'])
        development = read(root/'development.json')
        assert {(r['mode'],r['seed']) for r in development}==set(product(MODES[:-1],SEEDS))
        if args.phase=='matched':
            jobs = [(fit,(root,'fixed',r['seed'],protocol['steps'],config,r['active'],protocol.get('lookup_curriculum',False),
                          protocol.get('lookup_preview',False)))
                    for r in development if r['mode']=='evolve']
            destination = 'matched.json'
        else:
            all_runs = development+read(root/'matched.json')
            assert {(r['mode'],r['seed']) for r in all_runs}==set(product(MODES,SEEDS))
            for seed in SEEDS:
                assert len({r['train_sha256'] for r in all_runs if r['seed']==seed})==1
            jobs = [(evaluate_frozen,(root,r,config)) for r in all_runs]
            destination = 'results.json'
    launch(root,jobs,args.workers,destination)


if __name__=='__main__':
    main()
