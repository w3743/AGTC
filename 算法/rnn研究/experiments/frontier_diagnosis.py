"""Separate clue order from spacing in the frozen event graph."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import torch

from adaptive_graph_retest import load_model, read


def layout(compact, scattered, reorder, spread):
    spans = scattered['spans']
    order = [i for i, _, _ in spans] if reorder else list(range(len(spans)))
    template = scattered['variants'][0][0]
    start = 0
    pads = []
    for _, k, v in spans:
        pads.append(template[start:k] if spread else [])
        start = v+1
    variants, sources = [], []
    for x,y in compact['variants']:
        prefix, positions = [], {}
        for idx, noise in zip(order,pads,strict=True):
            prefix.extend(noise)
            positions[idx] = len(prefix)+1
            prefix.extend(x[2*idx:2*idx+2])
        variants.append((prefix+x[2*len(order):],y))
        sources.append(list(positions.values()))
    return variants,sources


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    protocol=read(args.retest/'protocol.json')
    output=[]
    for task,seed in itertools.product(protocol['tasks'],protocol['seeds']):
        data=read(args.retest/f'{task}-fresh.json')
        model,path=load_model('adaptive',task,seed,Path(protocol['reference']),Path(protocol['graph']))
        for gap,reorder,spread in itertools.product(('32','256','1024'),(False,True),(False,True)):
            cases=[layout(a,b,reorder,spread) for a,b in zip(data['compact'][gap],data['scattered'][gap],strict=True)]
            rows=[]
            for i in range(0,len(cases),8):
                batch=cases[i:i+8]; size=len(batch[0][0]); x=torch.tensor([s for vs,_ in batch for s,_ in vs])
                y=torch.tensor([y for vs,_ in batch for _,y in vs]); logits,edges=model(x,return_edges=True)
                flat=torch.stack(edges,dim=1).flatten(1)
                correct=logits.argmax(-1)==y
                for j,(variants,positions) in enumerate(batch):
                    hits=correct[j*size:(j+1)*size].tolist(); covered=[]
                    for k,(seq,_) in enumerate(variants):
                        relevant=positions[k]
                        if task=='binding':
                            relevant=[pos for pos in relevant if seq[pos-1]==seq[-1]]
                        actual=set(flat[j*size+k].tolist())
                        covered.append(all(pos in actual or pos+1 in actual for pos in relevant))
                    a,b=(e[j*size:j*size+2] for e in edges)
                    changed=mean(1-len(set(u.tolist())&set(v.tolist()))/len(u) for hop in (a,b) for u,v in zip(hop[0],hop[1]))
                    rows.append((all(hits),mean(covered),mean(c and not h for c,h in zip(covered,hits)),changed))
            output.append(dict(task=task,seed=seed,gap=int(gap),reorder=reorder,spread=spread,
                joint=mean(r[0] for r in rows),coverage=mean(r[1] for r in rows),
                covered_but_wrong=mean(r[2] for r in rows),rewire=mean(r[3] for r in rows),checkpoint=str(path)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(f'Diagnosed {len(output)} frozen model/condition rows.')


if __name__=='__main__':
    main()
