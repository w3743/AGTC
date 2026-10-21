import sys,os,time,json
sys.path.insert(0,'/opt/rk-lightstereo/app')
import numpy as np
import cv2
from depth import MetricDepth as BaseDepth
from lightstereo_depth import MetricDepth
cv2.setNumThreads(4)
m=MetricDepth.__new__(MetricDepth)
BaseDepth.__init__(m,'/opt/rk-lightstereo/app/calibration.npz')
m.width-=8;m.masks=[x[:,4:-4].copy() for x in m.masks];m.yy,m.xx=np.indices((m.height,m.width))
dl=np.load('/opt/lightstereo/portrait-npu-forward.npz')['disparity'];dr=-np.load('/opt/lightstereo/portrait-npu-reverse.npz')['disparity'][:,::-1]
left=np.zeros(dl.shape,np.uint8);results={}
for name,cpus in [('all',set(range(8))),('big',set(range(4,8)))]:
 os.sched_setaffinity(0,cpus);times=[]
 for i in range(12):
  t=time.perf_counter();d,v=m.metric_values(dl,dr,left);q=np.percentile(d[v],[0,5,50,95,100]);c=m.colorize(d,[q[1],q[-2]]);times.append((time.perf_counter()-t)*1000)
 results[name]={'median_ms':float(np.median(times[2:])),'valid':int(v.sum())}
print(json.dumps(results))
