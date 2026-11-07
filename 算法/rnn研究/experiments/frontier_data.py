"""Deterministic paired episodes for memory retention and composition."""
from __future__ import annotations

import hashlib
import itertools
import random


TASKS=('lookup','latest','pointer','arithmetic')
LAYOUTS=('compact','scattered')
LENGTHS={'train':(16,64,128,256),'validation':(16,64,128,256),
         'development':(384,768),'sealed':(512,1024,2048)}
RECORDS={3,4,5}
QUERY=6
KEYS=range(10,42)
VALUES=range(42,46)
OPS={50:(1,),51:(-1,),52:(1,1),53:(1,-1),54:(-1,1),55:(-1,-1)}


def oracle(digits):
    assert digits[-2]==QUERY
    graph,operands={},{}
    i=0
    while i<len(digits)-2:
        op=digits[i]
        if op in RECORDS:
            a,b=digits[i+1:i+3]
            (operands if op==5 else graph)[a]=b
            i+=3
        else:
            assert op in (1,2)
            i+=1
    q=digits[-1]
    if q in OPS:
        return (operands[46]-42+sum(sign*(operands[47+j]-42) for j,sign in enumerate(OPS[q])))%4
    visited=set()
    while q not in VALUES:
        assert q not in visited and q in graph
        visited.add(q)
        q=graph[q]
    return q-42


def episode(task,layout,gap,difficulty,identity):
    rng=random.Random(int.from_bytes(hashlib.blake2b(identity.encode(),digest_size=16).digest(),'little'))
    keys=rng.sample(KEYS,16)
    a,b=rng.sample(list(VALUES),2)
    if task in ('lookup','latest'):
        target,other=(keys[0],keys[1])
        records=[[3,k,v] for k,v in zip(keys[:6],[a,b,*rng.choices(VALUES,k=4)])]
        edits=[(0,2,42+(a-41)%4)]
        if task=='latest':
            x,y=rng.choice([(x,y) for x,y in itertools.permutations(VALUES,2) if x!=a and y!=b])
            records.extend([[3,target,x],[3,other,y]])
            edits=[(6,2,42+(records[6][2]-41)%4),(0,2,42+(a-41)%4)]
        query,swap=target,other
    elif task=='pointer':
        first,second=keys[:difficulty+1],keys[4:5+difficulty]
        records=[[4,u,v] for u,v in zip(first,first[1:])]
        records+=[[3,first[-1],a]]
        records+=[[4,u,v] for u,v in zip(second,second[1:])]+[[3,second[-1],b]]
        records.extend([[3,k,rng.choice(VALUES)] for k in keys[8:12]])
        edits=[(i,2,second[i+1]) for i in range(difficulty)]+[(difficulty,2,42+(a-41)%4)]
        query,swap=first[0],second[0]
    else:
        values=[rng.choice(VALUES) for _ in range(difficulty-1)]+[rng.choice((43,45))]
        records=[[5,46+i,v] for i,v in enumerate(values)]
        query=rng.choice((50,51) if difficulty==2 else (52,53,54,55))
        swap=query^1 if query>=52 else 101-query
        edits=[(i,2,42+(v-41)%4) for i,v in enumerate(values)]
    order=list(range(len(records)))
    if layout=='scattered':
        rng.shuffle(order)
        if task=='latest':
            for old,new in ((0,6),(1,7)):
                i,j=order.index(old),order.index(new)
                if i>j:
                    order[i],order[j]=order[j],order[i]
    # This 64-bit nonce is task-irrelevant noise and stays fixed across variants.
    nonce=hashlib.blake2b(('prefix:'+identity).encode(),digest_size=8).digest()
    prefix=[1+((byte>>bit)&1) for byte in nonce for bit in range(8)]
    widths=[6 if layout=='scattered' else 0 for _ in order]
    if layout=='scattered':
        for _ in order:
            i,j=rng.sample(range(len(order)),2)
            if widths[i]>2 and widths[j]<10:
                widths[i]-=1
                widths[j]+=1
    pads=[rng.choices((1,2),k=w) for w in widths]
    tail=rng.choices((1,2),k=gap)
    def encode(q,edit=None):
        x=prefix.copy()
        for j,pad in zip(order,pads,strict=True):
            rec=records[j].copy()
            if edit and j==edit[0]:
                rec[edit[1]]=edit[2]
            x.extend(pad)
            x.extend(rec)
        return x+tail+[QUERY,q]
    seqs=[encode(query),encode(swap),*[encode(query,e) for e in edits]]
    targets=[oracle(x) for x in seqs]
    assert targets[0]!=targets[1]
    assert all(y!=targets[0] for y in targets[2:-1] if task=='latest')
    if task=='latest':
        assert targets[-1]==targets[0]
    else:
        assert all(y!=targets[0] for y in targets[2:])
    return dict(id=identity,task=task,layout=layout,gap=gap,difficulty=difficulty,variants=seqs,targets=targets)


def cells(split):
    for task,layout,gap in itertools.product(TASKS,LAYOUTS,LENGTHS[split]):
        levels=(1,2,3) if task=='pointer' and split=='sealed' else (1,2) if task=='pointer' else (2,3) if task=='arithmetic' else (1,)
        for level in levels:
            yield task,layout,gap,level


def split_episodes(split,round_id=1,per_cell=128):
    for cell in cells(split):
        for i in range(per_cell):
            yield episode(*cell,f'{split}/{round_id}/{cell}/{i}')


def batch_episodes(seed,step):
    rng=random.Random(208313+seed+step*173)
    train_cells=list(cells('train'))
    # Cycle every cell before repeating; variable-length scattered sequences are padded.
    cell=train_cells[(step+seed)%len(train_cells)]
    for j in range(16):
        row=episode(*cell,f'train/{seed}/{step}/{j}/{cell}')
        variant=rng.randrange(len(row['variants']))
        yield row['variants'][variant],row['targets'][variant]
