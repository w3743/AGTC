"""Symbolic IMMW diagnostic with independently controlled retention and interruptions."""

from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations, product
import random


SKILLS = ('checkpoint','operation','live','rule')
TRAIN_MASKS = tuple(sum(1<<i for i in combo) for size in (1,2)
                    for combo in combinations(range(4),size))
NOVEL_MASKS = tuple(sum(1<<i for i in combo) for size in (3,4)
                    for combo in combinations(range(4),size))
GAPS = (64,256,1024,2048,8192)
RATES = (4,16,64)  # Actual interruptions requested per 256 steps in the retention phase.
CAPACITIES = (2,4,8)
SWITCH = 1
START = (10,14,18,22)
NOISE = range(24,28)
BIT = 28
ANSWER = 30
VOCAB = 31


@dataclass(frozen=True)
class Factors:
    gap: int
    rate: int
    capacity: int

    def __post_init__(self):
        if self.gap<16 or not 0<=self.rate<=128 or not 2<=self.capacity<=8:
            raise ValueError(f'Invalid IMMW condition {self}')


@dataclass(frozen=True)
class World:
    history: tuple
    target: int
    sources: tuple
    counterfactuals: tuple
    gap_start: int
    gap_end: int
    interruptions: int
    operation_switches: tuple


def random_stream(identity):
    return random.Random(int.from_bytes(blake2b(identity.encode(),digest_size=16).digest(),'little'))


def create_world(factors,identity):
    rng = random_stream(identity)
    tasks = rng.sample(range(8),factors.capacity)
    target = rng.choice(tasks)
    # Randomize creation order independently of which task will be resumed.
    rng.shuffle(tasks)
    history = []
    latest = {}

    def write(task,kind,value):
        latest[task,kind] = len(history)
        history.append(START[kind]+value)

    for active in tasks:
        history.append(SWITCH+active)
        for kind in range(4):
            write(active,kind,rng.randrange(2 if kind==3 else 4))
    gap_start = len(history)
    interruptions = round(factors.gap*factors.rate/256)
    planned = set(rng.sample(range(factors.gap),interruptions))
    available = sorted(set(range(factors.gap))-planned)
    event_steps = rng.sample(available, max(1,factors.gap//96)+factors.gap//512)
    operations = set(event_steps[:max(1,factors.gap//96)])
    rules = set(event_steps[max(1,factors.gap//96):])
    switched = None
    operation_switches = set()
    for step in range(factors.gap):
        if step in planned:
            active = rng.choice([task for task in tasks if task!=active])
            switched = len(history)
            history.append(SWITCH+active)
        elif step in operations:
            write(active,1,rng.randrange(1,4))
            if switched is not None:
                operation_switches.add(switched)
        elif step in rules:
            write(active,3,rng.randrange(2))
        elif rng.random()<.10:
            write(active,2,rng.randrange(4))
        else:
            history.append(rng.choice(NOISE))
    gap_end = len(history)
    # Refresh every context, in random order. Only the live channel changes here.
    rng.shuffle(tasks)
    for active in tasks:
        history.append(SWITCH+active)
        write(active,2,rng.randrange(4))
    other = rng.choice([task for task in tasks if task!=target])
    history.append(SWITCH+target)
    return World(tuple(history),target,tuple(latest[target,i] for i in range(4)),
                 tuple(latest[other,i] for i in range(4)),gap_start,gap_end,interruptions,
                 tuple(sorted(operation_switches)))


def solve(tokens):
    """Oracle receives observable events and reconstructs the addressed task."""
    if len(tokens)<6 or tokens[-1]!=ANSWER or any(t not in (BIT,BIT+1) for t in tokens[-5:-1]):
        raise ValueError('Malformed IMMW question.')
    slots = {}
    current = None
    for token in tokens[:-5]:
        if SWITCH<=token<SWITCH+8:
            current = token-SWITCH
        elif 10<=token<=23:
            kind = min((token-10)//4,3)
            value = token-START[kind]
            if kind==1:
                value = (slots.get((current,kind),0)+value)%4
            slots[current,kind] = value
        elif token not in NOISE:
            raise ValueError(f'Invalid IMMW event {token}')
    flags = [t-BIT for t in tokens[-5:-1]]
    result = sum(slots[current,k] for k in range(3) if flags[k])%4
    if flags[3] and slots[current,3]:
        result = (1-result)%4
    return result


def ask(world,mask):
    kinds = [i for i in range(4) if mask&(1<<i)]
    if not kinds or mask>=1<<4:
        raise ValueError('An IMMW query must select one or more valid factors.')
    suffix = tuple(BIT+int(i in kinds) for i in range(4))+(ANSWER,)
    base = world.history+suffix
    variants = [base]
    edits,kinds_edited = [],[]
    # Edit the most recent source for each required kind, plus a source in another task.
    for pos,kind in [(world.sources[i],i) for i in kinds]+[(world.counterfactuals[kinds[0]],kinds[0])]:
        old = base[pos]-START[kind]
        token = START[kind]+(old+1)%(2 if kind==3 else 4)
        variants.append(base[:pos]+(token,)+base[pos+1:])
        edits.append(pos)
        kinds_edited.append(SKILLS[kind])
    kinds_edited[-1] = 'irrelevant'
    base_target = solve(base)
    if 1 in kinds:
        for pos in reversed(world.operation_switches):
            prior = base[pos]-SWITCH
            other = next(t-SWITCH for t in world.history[:world.gap_start] if SWITCH<=t<SWITCH+8
                         and t-SWITCH!=world.target)
            new_task = world.target if prior!=world.target else other
            trial = base[:pos]+(SWITCH+new_task,)+base[pos+1:]
            if solve(trial)!=base_target:
                variants.insert(-1,trial)
                edits.insert(-1,pos)
                kinds_edited.insert(-1,'interruption')
                break
    targets = tuple(solve(v) for v in variants)
    assert all(t!=targets[0] for t in targets[1:-1]) and targets[-1]==targets[0]
    return dict(variants=variants,targets=targets,edits=edits,edit_kinds=kinds_edited,mask=mask,
                ages=[len(base)-1-pos for pos in world.sources])


def cases(partition, gaps=GAPS, rates=RATES, capacities=CAPACITIES, count=32, novel=False):
    if partition not in ('train','development','holdout'):
        raise ValueError(partition)
    if partition=='train' and novel:
        raise ValueError('Novel combinations are reserved for evaluation.')
    masks = NOVEL_MASKS if novel else TRAIN_MASKS
    for gap,rate,capacity in product(gaps,rates,capacities):
        factors = Factors(gap,rate,capacity)
        for i in range(count):
            identity = f'immw/v1/{partition}/{gap}/{rate}/{capacity}/{i}'
            world = create_world(factors,identity)
            offset = (i+gap+rate+capacity)%len(masks)
            yield factors,identity,world,ask(world,masks[offset])


def suffix_ceiling(episode, window):
    """An exact group-wise impossibility certificate for any deterministic suffix model."""
    seen = {}
    for x,y in zip(episode['variants'],episode['targets'],strict=True):
        suffix = x[-window:] if window else ()
        if suffix in seen and seen[suffix]!=y:
            return False
        seen[suffix] = y
    return True
