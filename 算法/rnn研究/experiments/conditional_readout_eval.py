"""Matched training, development and held-out arithmetic composition audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
from pathlib import Path

import torch
from torch.nn import functional as F

from conditional_readout import ConditionalReadout
from frontier_data import episode, oracle
from frontier_eval import CHECKPOINTS, SEEDS, digest, read, setup, write
from shared_address_eval import BATCH, LAYOUTS, RATE, TEST_GAPS, TRAIN_GAPS, aggregate, prefix


MODES = ('conditioned','history_gate','shared')
TASK = 'arithmetic'


def cases(partition, gaps, count):
    for layout, gap, n in itertools.product(LAYOUTS, gaps, (2,3)):
        for i in range(count):
            identity = f'conditional-readout/v1/{partition}/{layout}/{gap}/{n}/{i}'
            yield episode(TASK, layout, gap, n, identity)


def training_batch(seed, step, arities):
    cells = list(itertools.product(LAYOUTS, TRAIN_GAPS, arities))
    layout,gap,n = cells[(step-1) % len(cells)]
    for i in range(BATCH//2):
        row = episode(TASK,layout,gap,n,f'conditional-readout/v1/train/{seed}/{step}/{i}')
        # Each operation and its counterfactual are in the same update.
        # Rotate through the source edits, including even final operands.
        edits = (0,*range(2,len(row['variants'])))
        evidence = row['variants'][edits[(step//len(cells)+i) % len(edits)]][:-1]
        for variant in row['variants'][:2]:
            seq = evidence+[variant[-1]]
            yield row, seq, oracle(seq)


@torch.no_grad()
def predict(model, rows, ablate_evidence=False):
    model.eval()
    result = []
    for _, group in itertools.groupby(rows, key=lambda r:(r['layout'],r['gap'],r['difficulty'])):
        group = list(group)
        for start in range(0,len(group),BATCH):
            batch = group[start:start+BATCH]
            width = len(batch[0]['variants'])
            x = torch.tensor([v for r in batch for v in r['variants']])
            logits, edges = model(x, return_edges=True)
            base, swapped = x[::width], x[1::width]
            # Hold retrieval input constant. Replace only the instruction embedding
            # at the readout, or restore the old embedding after normal retrieval.
            readout_only = model(base, readout_query=swapped[:,-2:]).softmax(-1).tolist()
            old_readout = model(swapped, readout_query=base[:,-2:]).softmax(-1).tolist()
            probabilities = logits.softmax(-1).reshape(len(batch),width,4).tolist()
            routed = edges[0].reshape(len(batch),width,4,-1)[:,:2].tolist()
            sources = {
                name:model(x,evidence=name).softmax(-1).reshape(len(batch),width,4).tolist()
                for name in ('relations','graph')
            } if ablate_evidence else {}
            for i,r in enumerate(batch):
                record = dict(id=r['id'], probabilities=probabilities[i], edges=[routed[i]],
                              readout_only=readout_only[i], old_readout=old_readout[i])
                if sources:
                    record['evidence_only'] = {name:values[i] for name,values in sources.items()}
                result.append(record)
    return result


def pair_flip(tensor):
    return tensor.unflatten(0, (-1,2)).flip(1).flatten(0,1)


def training_loss(model, x, y, contrastive, ranking):
    query, history, _ = model.composition_inputs(x)
    logits = model.combine(query,history)
    loss = F.cross_entropy(logits,y)
    if contrastive:
        counterfactual = model.combine(pair_flip(query),history)
        if ranking:
            alternate = pair_flip(y)
            different = (y!=alternate).to(logits.dtype)
            actual_margin = logits.gather(1,y[:,None])-logits.gather(1,alternate[:,None])
            swapped_margin = counterfactual.gather(1,alternate[:,None])-counterfactual.gather(1,y[:,None])
            penalties = F.softplus(-actual_margin[:,0])+F.softplus(-swapped_margin[:,0])
            loss = loss + 2*(penalties*different).sum()/different.sum().clamp_min(1)
        else:
            labels = F.one_hot(y,4).to(logits.dtype)
            target = 2*(labels-pair_flip(labels))
            loss = loss + F.mse_loss(logits-counterfactual,target)
    return loss


def train(root, mode, seed, steps, dev, heldout_prefixes, protocol):
    setup(seed)
    model = build(mode,protocol)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE)
    isolated = {prefix(r) for r in dev} | heldout_prefixes
    run = root/f'{mode}-{seed}'
    run.mkdir()
    training_hash = hashlib.sha256()
    checkpoints = []
    losses = []
    for step in range(1,steps+1):
        rows, inputs, targets = zip(*training_batch(seed,step,protocol['train_arities']),strict=True)
        assert all(prefix(row) not in isolated for row in rows)
        x,y = torch.tensor(inputs),torch.tensor(targets)
        training_hash.update(x.numpy().tobytes())
        training_hash.update(y.numpy().tobytes())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model,x,y,protocol['contrastive'],protocol.get('ranking',False))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step in CHECKPOINTS or step==steps:
            weight_path = run/f'weights-{step}.pt'
            torch.save(model.state_dict(),weight_path)
            points = aggregate(dev,predict(model,dev))
            checkpoint = dict(questions=BATCH*step, loss=sum(losses)/len(losses), development=points,
                              sha256=digest(weight_path))
            checkpoints.append(checkpoint)
            losses.clear()
            print(mode,seed,step,round(checkpoint['loss'],3),[round(c['joint'],3) for c in points],flush=True)
    result = dict(mode=mode, seed=seed, steps=steps, parameters=sum(p.numel() for p in model.parameters()),
                  train_sha256=training_hash.hexdigest(),checkpoints=checkpoints)
    write(run/'development.json', result)
    return result


def build(mode, protocol):
    if mode=='original':
        return ConditionalReadout('shared')
    return ConditionalReadout(mode,protocol.get('neutral_routing',False),protocol.get('binding',False),
                              protocol.get('relation_bank',False))


def test(root, result, rows, protocol):
    mode,seed,steps = (result[k] for k in ('mode','seed','steps'))
    setup(seed)
    run = root/f'{mode}-{seed}'
    weight_path = run/f'weights-{steps}.pt'
    assert digest(weight_path)==result['checkpoints'][-1]['sha256']
    model = build(mode,protocol)
    model.load_state_dict(torch.load(weight_path,weights_only=True))
    predictions = predict(model,rows,ablate_evidence=mode=='conditioned' and model.relation_bank)
    write(run/'predictions.json',predictions)
    return result | {'holdout': aggregate(rows,predictions)}


def run_train(args):
    args.output.mkdir(parents=True)
    dev = list(cases('development',(0,384,768),64))
    heldout_prefixes = {prefix(r) for r in cases('holdout',TEST_GAPS,128)}
    assert len(heldout_prefixes)==128*len(LAYOUTS)*len(TEST_GAPS)*2
    assert len({prefix(r) for r in dev}|heldout_prefixes)==len(dev)+len(heldout_prefixes)
    protocol = dict(
        steps=args.steps, modes=args.modes, seeds=args.seeds, batch=BATCH, learning_rate=RATE,
        optimizer='AdamW', train_gaps=TRAIN_GAPS, holdout_gaps=TEST_GAPS, groups_per_holdout_cell=128,
        contrastive=args.contrastive, train_arities=args.train_arities, neutral_routing=args.neutral_routing,
        binding=args.binding, ranking=args.ranking, relation_bank=args.relation_bank,
        torch=torch.__version__,
    )
    write(args.output/'protocol.json',protocol)
    write(args.output/'development-cases.json',dev)
    jobs = list(itertools.product(args.modes,args.seeds))
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        futures = [pool.submit(train,args.output,*job,args.steps,dev,heldout_prefixes,protocol) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            write(args.output/'development.json',ordered(results))


def ordered(results):
    return sorted(results,key=lambda r:(r['mode'],r['seed']))


def run_test(args):
    protocol = read(args.output/'protocol.json')
    development = read(args.output/'development.json')
    assert set(MODES)<=set(protocol['modes']) and set(protocol['seeds'])==set(SEEDS)
    expected = set(itertools.product(protocol['modes'],SEEDS))
    assert {(r['mode'],r['seed']) for r in development}==expected and len(development)==len(expected)
    assert protocol['steps']==1000
    if not all(next(c['joint'] for c in r['checkpoints'][-1]['development']
                    if (c['layout'],c['gap'],c['difficulty'])==('scattered',0,2))>=.85
               for r in development if r['mode']=='conditioned'):
        raise SystemExit('The conditioned model must reach 85% on zero-gap scattered development in every seed.')
    for seed in SEEDS:
        assert len({r['train_sha256'] for r in development if r['seed']==seed})==1
    rows = list(cases('holdout',TEST_GAPS,128))
    write(args.output/'holdout-cases.json',rows)
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers,len(development))) as pool:
        futures = [pool.submit(test,args.output,result,rows,protocol) for result in development]
        for future in as_completed(futures):
            results.append(future.result())
            write(args.output/'results.json',ordered(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('train','test'))
    parser.add_argument('output',type=Path)
    parser.add_argument('--steps',type=int,default=1000,choices=range(1,1001),metavar='1..1000')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--contrastive',action='store_true')
    parser.add_argument('--train-arities',nargs='+',type=int,default=(2,3),choices=(2,3))
    parser.add_argument('--neutral-routing',action='store_true')
    parser.add_argument('--binding',action='store_true')
    parser.add_argument('--relation-bank',action='store_true')
    parser.add_argument('--ranking',action='store_true',help='Use paired answer ranking instead of squared differences.')
    parser.add_argument('--modes',nargs='+',default=MODES,choices=(*MODES,'original'))
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS,choices=SEEDS)
    args = parser.parse_args()
    if args.ranking and not args.contrastive:
        parser.error('--ranking requires --contrastive')
    (run_train if args.stage=='train' else run_test)(args)
