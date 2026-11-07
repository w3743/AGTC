"""Single-process, single-thread CPU inference resource probe."""
import argparse
import ctypes
import json
from pathlib import Path
from statistics import median
import sys
from threading import Event, Thread
from time import perf_counter

import torch

from frontier_data import episode
from frontier_eval import setup
from frontier_models import build


def rss():
    if sys.platform=='win32':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[
                (name,ctypes.c_size_t) for name in ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','page','peak_page')]
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        query=ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes=(ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong)
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        if not query(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError('GetProcessMemoryInfo failed')
        return counters.working,counters.peak
    if sys.platform.startswith('linux'):
        lines=Path('/proc/self/status').read_text().splitlines()
        return tuple(int(next(line for line in lines if line.startswith(k)).split()[1])*1024 for k in ('VmRSS:','VmHWM:'))
    raise RuntimeError('Peak RSS probe supports Windows and Linux')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name');p.add_argument('checkpoint',type=Path);p.add_argument('gap',type=int)
    args=p.parse_args()
    setup(2)
    m=build(args.name).eval()
    m.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    x=torch.tensor([episode('arithmetic','scattered',args.gap,3,'resource-probe')['variants'][0]])
    before,old_peak=rss(); observed=[before]; done=Event()
    def sample():
        while not done.wait(.001):
            observed.append(rss()[0])
    watcher=Thread(target=sample,daemon=True)
    watcher.start()
    try:
        m(x)
    finally:
        done.set();watcher.join()
    current,new_peak=rss()
    peak=max(*observed,current,new_peak if new_peak>old_peak else 0)
    for _ in range(2): m(x)
    times=[]
    for _ in range(9):
        t=perf_counter();m(x);times.append(1000*(perf_counter()-t))
    latency=median(times)
    edges={'adaptive':64,'slot_fixed':64,'slot_dense':512,'event_dense':8*(x.shape[1]-1)}.get(args.name)
    print(json.dumps(dict(name=args.name,gap=args.gap,tokens=x.shape[1],latency_ms=latency,
        ms_per_token=latency/x.shape[1],peak_increment_mib=(peak-before)/2**20,active_read_edges=edges,samples_ms=times)))


if __name__=='__main__':
    main()
