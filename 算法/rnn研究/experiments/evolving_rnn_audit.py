"""Causal memory-edge cuts and actual single-thread inference costs."""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import torch
from torch.nn import functional as F

from evolving_rnn import EvolvingRNN, Survival
from evolving_rnn_eval import WIDTH, cells, evaluate, launch
from frontier_eval import digest, read, setup, write
from frontier_profile import rss


def restore(root,result):
    seed,mode = result['seed'],result['mode']
    setup(seed)
    config = Survival(**read(root/'protocol.json')['survival'])
    model = EvolvingRNN(WIDTH,mode,config,result['active'] if mode=='fixed' else None)
    path = root/f'{mode}-{seed}'/'weights.pt'
    assert digest(path)==result['weights_sha256']
    model.load_state_dict(torch.load(path,weights_only=True))
    return model.eval()


def sensitivity(model,cell):
    _,_,x,y = cell
    strength = torch.zeros_like(model.life)
    for xb,yb in zip(x.flatten(0,1).split(16),y.flatten().split(16),strict=True):
        model.zero_grad(set_to_none=True)
        logits = model(xb,probe=True)
        F.cross_entropy(logits,yb).backward()
        strength.add_((model.probe.detach()*model.probe.grad).abs())
        model.probe = model.probe_states = model.probe_outputs = None
    return strength/strength.sum().clamp_min(1e-10)


def cut_trial(root,result):
    model = restore(root,result)
    short,long = [c for c in cells('development',(4,32),n=64) if c[0]=='recall']
    score = sensitivity(model,long)-sensitivity(model,short)
    active = torch.where(model.active.flatten())[0]
    count = max(1,round(.01*len(active)))
    chosen = active[score.flatten()[active].argsort(descending=True,stable=True)[:count]]
    cut = torch.zeros_like(model.active)
    cut.flatten()[chosen] = True
    evaluation = [c for c in cells('holdout',(4,32,128,256),n=96) if c[0]=='recall']
    base = evaluate(model,evaluation)
    targeted = evaluate(model,evaluation,cut=cut)
    rng = torch.Generator().manual_seed(result['seed']+7941)
    random_controls = []
    rows = chosen//WIDTH
    for _ in range(5):
        other = torch.zeros_like(model.active)
        for i in rows.unique().tolist():
            candidates = torch.where(model.active[i])[0]
            length = int((rows==i).sum())
            other[i,candidates[torch.randperm(len(candidates),generator=rng)[:length]]] = True
        random_controls.append(evaluate(model,evaluation,cut=other))
    return dict(mode=result['mode'],seed=result['seed'],indices=chosen.tolist(),base=base,targeted=targeted,
                random=random_controls)


@torch.inference_mode()
def profile(root,result):
    model = restore(root,result)
    x = next(c[2][0,0:1] for c in cells('holdout',(512,),n=2) if c[0]=='recall')
    return profile_model(model,x,result)


@torch.inference_mode()
def profile_model(model,x,result):
    before,old_peak = rss()
    reference = model(x)
    after,new_peak = rss()
    peak = max(after,new_peak if new_peak>old_peak else 0)

    def measure(call):
        for _ in range(3):
            call()
        durations = []
        cpu_started = process_time()
        for _ in range(15):
            t = perf_counter()
            call()
            durations.append((perf_counter()-t)*1000)
        return median(durations),1000*(process_time()-cpu_started)/len(durations)

    baseline,baseline_cpu = measure(lambda:model(x))
    csr = None
    try:
        # Time steady-state sparse inference with the packed recurrent matrix reused.
        prepare = perf_counter()
        packed = model.effective_recurrent().to_sparse_csr()
        build_ms = (perf_counter()-prepare)*1000

        def sparse():
            h = model.head.weight.new_zeros((1,model.core.hidden_size))
            inputs = F.linear(model.tokens(x),model.core.weight_ih_l0,model.core.bias_ih_l0)
            for value in inputs.unbind(1):
                h = (value+torch.sparse.mm(packed,h.T).T+model.core.bias_hh_l0).tanh()
            return model.head(h)

        candidate = sparse()
        error = float((candidate-reference).abs().max())
        sparse_ms,sparse_cpu = measure(sparse)
        csr = dict(ms=sparse_ms,cpu_ms=sparse_cpu,build_ms=build_ms,max_logit_difference=error,
                   matches=bool(torch.allclose(candidate,reference,atol=.001,rtol=.001)),
                   nonzero=packed.values().numel())
    except (NotImplementedError,RuntimeError) as error:
        csr = dict(error=str(error).splitlines()[0])
    return dict(mode=result['mode'],seed=result['seed'],active=result['active'],tokens=x.shape[1],
                masked_ms=baseline,masked_cpu_ms=baseline_cpu,masked_peak_increment_mib=(peak-before)/2**20,csr=csr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('cuts','resources'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--mode',choices=('dense','fixed','decay','evolve'))
    args = parser.parse_args()
    results = read(args.root/'development.json')+read(args.root/'matched.json')
    if args.phase=='cuts':
        jobs = [(cut_trial,(args.root,r)) for r in results]
        launch(args.root,jobs,args.workers,'cuts.json')
    else:
        if args.mode is None:
            parser.error('resources requires --mode, and should run in an isolated process')
        for r in results:
            if r['seed']==2 and r['mode']==args.mode:
                write(args.root/f'profile-{r["mode"]}.json',profile(args.root,r))


if __name__=='__main__':
    main()
