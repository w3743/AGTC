"""Isolate the role of regrowth and test whether the learned mask can be reused."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import product
from pathlib import Path

import torch
from torch.nn import functional as F

from dense_rewire import BATCH, TARGET_STEP, WARMUP, DataStream, DenseNet, encode, paired_questions, predict, rewire, scores
from frontier_eval import SEEDS, read, setup, write


MODES = ('prune_only','reset_adaptive','reset_random')


def fit(root,mode,seed,protocol):
    reset = mode!='prune_only'
    setup(seed+37471 if reset else seed)
    model = DenseNet()
    if reset:
        original = mode.removeprefix('reset_')
        checkpoint = root/f'{original}-{seed}/weights-{protocol["steps"]}.pt'
        saved = torch.load(checkpoint,weights_only=True)
        with torch.no_grad():
            for i,l in enumerate(model.layers):
                l.mask.copy_(saved[f'layers.{i}.mask'])
    data = DataStream(seed+31806,'train')
    optimizer = torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'])
    digest = sha256()
    shift = protocol['warmup']-WARMUP
    for step in range(1,protocol['steps']+1):
        x,y = encode(*data.draw(BATCH))
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x),y)
        loss.backward()
        optimizer.step()
        relative = step-shift
        if not reset and WARMUP<relative<=TARGET_STEP and relative%protocol['interval']==0:
            rewire(model,optimizer,'adaptive',relative,None,turnover=0)
    folder = root/'controls'/f'{mode}-{seed}'
    folder.mkdir()
    torch.save(model.state_dict(),folder/'weights.pt')
    data = paired_questions(2 if reset else 1,protocol['groups'])
    probs = predict(model,data)
    write(folder/'predictions.json',dict(targets={k:y.tolist() for k,(_,y) in data.items()},
                                        baseline={k:p.tolist() for k,p in probs.items()}))
    result = dict(mode=mode,seed=seed,test=scores(probs,data),train_sha256=digest.hexdigest(),
                  active=[int(l.mask.sum()) for l in model.layers])
    print(mode,seed,result['test']['both']['joint'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    protocol = read(args.output/'protocol.json')
    (args.output/'controls').mkdir()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fit,args.output,m,s,protocol) for m,s in product(MODES,SEEDS)]
        for job in as_completed(jobs):
            results.append(job.result())
            write(args.output/'controls/results.json',sorted(results,key=lambda r:(r['mode'],r['seed'])))


if __name__=='__main__':
    main()
