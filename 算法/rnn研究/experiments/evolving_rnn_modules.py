"""Discover communities without task labels, then test activity and lesion selectivity."""

from statistics import mean

import networkx as nx
import torch
from torch.func import functional_call

from evolving_rnn_eval import cells, evaluate


def communities(matrix,seed):
    symmetric = matrix.abs()+matrix.abs().T
    symmetric.fill_diagonal_(0.)
    graph = nx.from_numpy_array(symmetric.numpy())
    partition = nx.community.louvain_communities(graph,seed=seed)
    return [torch.tensor(sorted(group)) for group in partition],nx.community.modularity(graph,partition)


@torch.no_grad()
def lesion(model,indices,sets):
    core = model.core
    tensors = (core.weight_ih_l0,core.weight_hh_l0,core.bias_ih_l0,core.bias_hh_l0)
    original = [parameter.clone() for parameter in tensors]
    try:
        for parameter in tensors:
            parameter[indices] = 0.
        core.weight_hh_l0[:,indices] = 0.
        return evaluate(model,sets)
    finally:
        for parameter,value in zip(tensors,original,strict=True):
            parameter.copy_(value)


def drops(before,after):
    return {task:mean(b['joint']-a['joint'] for b,a in zip(before,after,strict=True) if b['task']==task)
            for task in ('recall','lookup')}


@torch.inference_mode()
def query_response(model,sets):
    delta = []
    for task,_,x,_ in sets:
        if task!='lookup':
            continue
        assert torch.equal(x[:,0,:-1],x[:,1,:-1])
        paired = x[:,:2].flatten(0,1)
        effective = model.effective_recurrent()
        for batch in paired.split(24):
            states,_ = functional_call(model.core,{'weight_hh_l0':effective},(model.tokens(batch),))
            a,b = states[:,-1].reshape(-1,2,states.shape[-1]).unbind(1)
            delta.append((a-b).abs())
    return torch.cat(delta).mean(0)


def contrast(response,groups):
    return float(torch.stack([response[group].mean() for group in groups]).std(unbiased=False))


def audit_modules(model,seed,partition):
    matrix = model.effective_recurrent().detach()
    width = matrix.shape[0]
    rng = torch.Generator().manual_seed(seed+622593)
    groups,modularity = communities(matrix,seed)
    null = []
    for _ in range(2):
        permutation = torch.rand(matrix.shape,generator=rng).argsort(dim=1)
        _,score = communities(matrix.gather(1,permutation),seed)
        null.append(score)
    development = list(cells('development',(4,16),n=64))
    holdout = list(cells(partition,(4,16),n=128))
    base_dev = evaluate(model,development)
    base_test = evaluate(model,holdout)
    lesion_dev = [drops(base_dev,lesion(model,group,development)) for group in groups]
    chosen = max(range(len(groups)),key=lambda i:abs(lesion_dev[i]['lookup']-lesion_dev[i]['recall']))
    target = drops(base_test,lesion(model,groups[chosen],holdout))
    response = query_response(model,holdout)
    random_lesions,random_contrast = [],[]
    sizes = [len(group) for group in groups]
    for _ in range(8):
        random = torch.randperm(width,generator=rng).split(sizes)
        random_lesions.append(drops(base_test,lesion(model,random[chosen],holdout)))
        random_contrast.append(contrast(response,random))
    return dict(modularity=modularity,null_modularity=null,groups=[g.tolist() for g in groups],selected=chosen,
                selection=lesion_dev[chosen],targeted=target,random=random_lesions,
                query_contrast=contrast(response,groups),random_query_contrast=random_contrast)
