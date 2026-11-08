"""Arithmetic and serial latency measurements for the streaming experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from compute_budget import ArithmeticCounter, _base_name

class StreamingCounter(ArithmeticCounter):
    """Extend the old arithmetic proxy for non-accumulating indexed writes."""
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        name=_base_name(func)
        if name == 'aten::full':
            out=func(*args,**kwargs)
            self.zero_cost[name]+=1
            return out
        if name in ('aten::index_put_','aten::index_put'):
            accumulate=kwargs.get('accumulate',args[3] if len(args)>3 else False)
            if not accumulate:
                out=func(*args,**kwargs)
                self.zero_cost[name]+=1
                return out
        return super().__torch_dispatch__(func,types,args,kwargs)

def counted(call):
    counter=StreamingCounter()
    with counter: call()
    result=counter.stats()
    result['additional_memory_only_allowlist']=['aten::full constant allocation',
        'aten::index_put_ / index_put with accumulate=False']
    if result['unsupported']:
        raise RuntimeError('Unsupported arithmetic operations: '+json.dumps(result['unsupported']))
    return result

def process(model,digit):
    model.predict()
    model.learn_target(digit)
    model.advance(digit)

def measure_models(factory,kinds,previous):
    torch.set_num_threads(1)
    history=[int('0000100110101111'[i%16]) for i in range(64)]
    result={}
    for kind in kinds:
        lengths=[0,1,32] if kind!='replay_lastpair20' else list(range(33))
        profiles={}
        for length in lengths:
            m=factory(kind,73)
            for digit in history[:length]: process(m,digit)
            profiles[length]=counted(lambda:process(m,history[length]))
        cost=lambda n:profiles[n if kind=='replay_lastpair20' or n<2 else 32]['total_proxy']
        full=sum(cost(min(step-1,32)) for step in range(1,601))
        m=factory(kind,73)
        for digit in history[:10]: process(m,digit)
        timings=[]
        for digit in history[10:30]:
            start=time.perf_counter();process(m,digit);timings.append((time.perf_counter()-start)*1000)
        state_elements=420 if kind!='replay_lastpair20' else 0
        eligible=269220 if kind!='replay_lastpair20' else 0
        result[kind]={'origin':'new','parameters':sum(p.numel() for p in m.network.parameters()),
            'width':20,'steady_proxy':profiles[32]['total_proxy'],'full_run_proxy':full,
            'serial_single_thread_mean_ms':statistics.mean(timings),
            'serial_single_thread_median_ms':statistics.median(timings),'warmup':10,'repeats':20,
            'state_elements':state_elements,'eligibility_elements':eligible,
            'active_eligibility_elements':12820 if kind=='streaming_one_step20' else eligible,
            'eligibility_bytes_float32':eligible*4,
            'profiles':{n:p for n,p in profiles.items()},
            'full_run_weighting':'event1 age0; event2 age1; event3..600 growing to age32; streaming age>=2 same-shaped path',
            'included':'predict, learn_target (including clip/Adam/prox), advance; excludes event logging, checkpoint I/O, finite assertions',
            'metric':'old documented scalar-op proxy plus indexed non-accumulating writes excluded as memory movement'}
        print('COMPUTE='+json.dumps({k:v for k,v in result[kind].items() if k!='profiles'}),flush=True)
    previous=Path(previous)
    full=json.loads((previous/'full_run_budgets.json').read_text())
    matched=json.loads((previous/'matched_budgets.json').read_text())
    latency=json.loads((previous/'cpu_latency.json').read_text())
    target=full['custom_linear20']['total_proxy']
    for kind in result:
        result[kind]['ratio_to_previous_replay_linear20']=result[kind]['full_run_proxy']/target
    for kind in full:
        result[kind]={'origin':'reused','parameters':matched[kind]['config']['parameters'],
            'width':matched[kind]['config']['width'],'steady_proxy':matched[kind]['total_proxy'],
            'full_run_proxy':full[kind]['total_proxy'],**latency[kind],
            'ratio_to_previous_replay_linear20':full[kind]['total_proxy']/target,
            'note':'Previous 3 updates on all31 adjacent pairs; old timings reused, not remeasured.'}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    from streaming_rtrl_benchmark import build_model, NEW_KINDS, PREVIOUS
    result=measure_models(build_model,NEW_KINDS,PREVIOUS)
    (args.root/'compute_profile.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(args.root/'compute_profile.json')

if __name__=='__main__':main()
