"""Replay the depth-probe weights on independently generated cyclic graphs."""

import argparse
from hashlib import blake2b, sha256
from pathlib import Path
import random

import torch

from frontier_eval import SEEDS, read, setup, write
from structure_depth import COMMANDS, KEYS, MARKERS, VALUES, SymbolWalk, evaluate, resolve


DEPTHS = (8,16,32,64)
GAP = 1024
COUNT = 128


def walk(graph,start,program):
    route = []
    for step in program:
        edge = start,step
        route.append(edge)
        start = graph[edge]
    return route


def cycle_episode(depth, index):
    identity = f'structure-depth/v1/cycles/{depth}/{index}'
    rng = random.Random(int.from_bytes(blake2b(identity.encode(),digest_size=16).digest(),'little'))
    nonce = blake2b(('prefix/'+identity).encode(),digest_size=8).digest()
    prefix = [1+((byte>>bit)&1) for byte in nonce for bit in range(8)]
    for _ in range(4096):
        graph = {(src,j):dst for src in KEYS for j,dst in enumerate(rng.sample(KEYS,2))}
        start = rng.choice(KEYS)
        program = [rng.randrange(2) for _ in range(depth)]
        original = walk(graph,start,program)
        if len({src for src,_ in original})==depth or original[-1] in original[:-1]:
            continue
        branches = list(range(depth))
        rng.shuffle(branches)
        for k in branches:
            new_program = program.copy()
            new_program[k] ^= 1
            changed = walk(graph,start,new_program)
            terminal_edges = original[-1],changed[-1]
            earlier = set(original[:-1]) | set(changed[:-1])
            touched = original[k]
            if (terminal_edges[0]==terminal_edges[1] or any(e in earlier for e in terminal_edges)
                    or touched in changed or touched in original[:k]
                    or len({src for src,_ in changed})==depth):
                continue
            graph[terminal_edges[0]],graph[terminal_edges[1]] = rng.sample(VALUES,2)
            replacement = graph[changed[k]]
            records = [(src,j,dst) for (src,j),dst in graph.items()]
            rng.shuffle(records)
            break
        else:
            continue
        break
    else:
        raise RuntimeError(f'No valid cyclic graph for {identity}')
    spacers = [rng.choices((1,2),k=3) for _ in records]
    tail = rng.choices((1,2),k=GAP)

    def encode(directions,edit=False):
        history = prefix.copy()
        for (src,j,dst),spacer in zip(records,spacers,strict=True):
            history.extend(spacer)
            history.extend((MARKERS[j],src,replacement if edit and (src,j)==touched else dst))
        return history+tail+[6,start]+[COMMANDS[j] for j in directions]

    variants = [encode(program),encode(new_program),encode(program,True)]
    labels,paths = zip(*(resolve(v,depth) for v in variants),strict=True)
    assert labels[0]!=labels[1] and labels[1]==labels[2]
    assert all(len(set(path))<len(path) for path in paths[:2])
    return dict(id=identity,width=len(KEYS),depth=depth,gap=GAP,program_size=depth,variants=variants,
                targets=labels,paths=paths,records=records,branch=k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='Existing depth experiment directory.')
    args = parser.parse_args()
    output = args.output/'cycles'
    output.mkdir()
    cases = [cycle_episode(depth,i) for depth in DEPTHS for i in range(COUNT)]
    combined = cases+read(args.output/'cases.json')
    assert len({tuple(r['variants'][0][:64]) for r in combined}) == len(combined)
    write(output/'cases.json',cases)
    results = []
    for seed in SEEDS:
        setup(seed)
        model = SymbolWalk(adaptive=True)
        checkpoint = args.output/f'adaptive-{seed}/weights.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        cells,predictions = evaluate(model,'adaptive',cases,COUNT)
        results.append(dict(mode='adaptive',seed=seed,checkpoint_sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                            cells=cells))
        write(output/f'predictions-{seed}.json',predictions)
        write(output/'results.json',results)
        print(seed,[c['joint'] for c in cells],[c['trace_exact'] for c in cells],flush=True)


if __name__=='__main__':
    main()
