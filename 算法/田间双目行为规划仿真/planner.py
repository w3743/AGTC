"""Build a visual height map and use a learned score to route over it."""
from collections import deque
import heapq
import math
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch
from torch import nn

from terrain import DT, XS, YS, VehicleConfig, terrain_reward, wrap
from road_memory import AisleExplorer, FieldLayout, MapMemory

CELL = .4
CHECKPOINT = Path(__file__).with_name("双目地形模型.pt")


class TerrainNet(nn.Module):
    """Predict probability of passage and expected terrain reward."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,24),nn.SiLU(),nn.Linear(24,2))

    def forward(self,x):
        return self.layers(x)


def load_model(path=CHECKPOINT):
    net=TerrainNet()
    net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["weights"])
    return net.eval()


def indexes(xy,xs=XS,ys=YS):
    return np.rint((np.asarray(xy)+[xs,ys])/CELL).astype(int)


def mapped(cells,xs=XS,ys=YS):
    return np.asarray(cells)*CELL-[xs,ys]


class VisualMap:
    def __init__(self,xs=XS,ys=YS,*,layout=None,memory=None):
        self.xs,self.ys=xs,ys
        self.nx,self.ny=round(2*xs/CELL)+1,round(2*ys/CELL)+1
        shape=self.ny,self.nx
        self.z=np.zeros(shape,dtype=np.float32)
        self.hit=np.zeros(shape,dtype=np.float32)
        self.rocks=np.zeros(shape,dtype=np.uint8)
        self.rock_z=np.zeros(shape,dtype=np.float32)
        self.risk=np.zeros(shape,dtype=np.float32)
        self.reward=np.zeros(shape,dtype=np.float32)
        self.blocked=np.zeros(shape,dtype=bool)
        self.live_blocked=np.zeros(shape,dtype=bool)
        self.rocks_blocked=np.zeros(shape,dtype=bool)
        self.traps=np.zeros(shape,dtype=bool)
        self.steep_seen=np.zeros(shape,dtype=np.int8)
        self.grade=np.zeros(shape,dtype=np.float32)
        self.features=np.zeros((*shape,5),dtype=np.float32)
        self.layout=layout
        self.memory=memory
        self.scope=None
        self.travel_cost=np.full(shape,1.06,dtype=np.float32)
        self.refresh_constraints()

    def refresh_constraints(self):
        self.blocked[:]=self.live_blocked | self.traps
        if self.memory is not None:
            self.blocked |= self.memory.occupied & (self.hit<4)
        if self.layout is not None:
            self.blocked |= self.layout.ridges
        if self.scope is not None:
            self.blocked |= ~self.scope
        self.blocked[[0,-1],:]=True
        self.blocked[:,[0,-1]]=True

    def indexes(self,xy):
        return indexes(xy,self.xs,self.ys)

    def mapped(self,cells):
        return mapped(cells,self.xs,self.ys)

    def update(self,frame):
        if not len(frame.cloud):
            return
        ix,iy=self.indexes(frame.cloud[:,:2]).T
        cell=iy*self.nx+ix
        for kind in (False,True):
            selected=np.flatnonzero(frame.rock==kind)
            if not len(selected):
                continue
            sorted_ids=selected[np.argsort(cell[selected])]
            boundaries=np.r_[0,1+np.flatnonzero(np.diff(cell[sorted_ids])),len(sorted_ids)]
            for a,b in zip(boundaries[:-1],boundaries[1:]):
                if b-a<2:
                    continue
                ids=sorted_ids[a:b]
                y,x=iy[ids[0]],ix[ids[0]]
                if kind:
                    levels=frame.cloud[ids,2]
                    if np.ptp(levels)>.58:
                        continue
                    z=np.median(levels)
                    old=min(int(self.rocks[y,x]),8)
                    self.rock_z[y,x]=(old*self.rock_z[y,x]+z)/(old+1)
                    self.rocks[y,x]=min(255,int(self.rocks[y,x])+1)
                else:
                    z=np.median(frame.cloud[ids,2])
                    weight=min(8.,self.hit[y,x])
                    self.z[y,x]=(weight*self.z[y,x]+z)/(weight+1)
                    self.hit[y,x]=min(25., self.hit[y,x]+1)

    def assess(self,net,max_slope=30.):
        known=self.hit>0
        if self.layout is not None:
            # Crop tops and both sides of each forbidden ridge are kept in
            # the reference layer. Interpolate driving grades from the aisle.
            known &= ~self.layout.crop_margin
        if not np.any(known):
            return
        nearest=ndimage.distance_transform_edt(~known,return_distances=False,return_indices=True)
        fill=self.z[tuple(nearest)]
        blur=ndimage.gaussian_filter(fill,.8)
        gy,gx=np.gradient(blur,CELL)
        slope=np.hypot(gx,gy)
        rise=ndimage.maximum_filter(blur,3)-ndimage.minimum_filter(blur,3)
        rugged=np.abs(fill-blur)
        support=ndimage.uniform_filter(known.astype(np.float32),size=3)
        above_floor=self.rock_z-fill
        rock=((self.rocks>=3)&(above_floor>.19))|((self.rocks>=8)&(above_floor>.095))
        rock_near=ndimage.maximum_filter(rock,size=3)
        self.rocks_blocked[:]=rock
        self.features=np.stack((slope,rise,rugged,support,rock_near),axis=-1).astype(np.float32)
        with torch.inference_mode():
            y=net(torch.from_numpy(self.features))
            self.risk[:] = 1-y[:,:,0].sigmoid().numpy()
            self.reward[:] = y[:,:,1].numpy()
        self.grade[:]=slope
        confident=(self.hit>=2) & (support>.40)
        physical_limit=slope>math.tan(math.radians(max_slope))
        steep=confident & (physical_limit|((self.risk>.69)&((slope>.37)|(rise>.48))))
        if self.layout is not None and self.layout.enabled:
            coherent=ndimage.uniform_filter(steep.astype(np.float32),size=3)>=.40
            self.steep_seen[:]=np.clip(self.steep_seen+np.where(steep & coherent,1,-1),0,4)
            steep=self.steep_seen>=3
        self.live_blocked[:]=rock | ndimage.maximum_filter(steep,size=3)
        remembered=self.memory.free if self.memory is not None else np.zeros_like(known)
        unknown_cost=np.where(remembered,1.13,1.35)
        estimated_speed=np.clip(1.18*(1.-1.15*slope),.40,1.18)
        pace=.65+.28/estimated_speed
        self.travel_cost[:]=np.where(known,pace+.16*self.risk,unknown_cost)
        self.travel_cost[rock_near & ~rock]+=.22
        self.refresh_constraints()

    def fields(self,xy):
        x,y=self.indexes(xy)
        x,y=np.clip(x,0,self.nx-1),np.clip(y,0,self.ny-1)
        return (self.hit[y,x]>0,bool(self.blocked[y,x]),float(self.risk[y,x]),
                float(self.grade[y,x]),float(self.reward[y,x]))

    def local_patch(self,xy,heading):
        forward=np.array([math.cos(heading),math.sin(heading)])
        lateral=np.array([-forward[1],forward[0]])
        rows,cols=np.mgrid[3:-4:-1,-3:4]
        patch=np.asarray(xy)+CELL*(rows[:,:,None]*forward+cols[:,:,None]*lateral)
        x,y=self.indexes(patch).transpose(2,0,1)
        x,y=np.clip(x,0,self.nx-1),np.clip(y,0,self.ny-1)
        return np.where(self.blocked[y,x],1.,np.where(self.hit[y,x]>0,self.risk[y,x],.34)).astype(np.float32)

    def segment(self,a,b):
        length=float(np.linalg.norm(np.subtract(a,b)))
        x,y=self.indexes(np.linspace(a,b,max(2,math.ceil(length/(CELL*.43))+1))).T
        if np.any((x<1)|(x>=self.nx-1)|(y<1)|(y>=self.ny-1)):
            return False
        return not np.any(self.blocked[y,x])

    def segment_price(self,a,b):
        length=float(np.linalg.norm(np.subtract(a,b)))
        x,y=self.indexes(np.linspace(a,b,max(2,math.ceil(length/(CELL*.43))+1))).T
        return length*float(np.mean(self.travel_cost[y,x]))

    def mark_trap(self,point):
        x,y=self.indexes(point)
        if 0<x<self.nx-1 and 0<y<self.ny-1:
            self.traps[y-1:y+2,x-1:x+2]=True
            self.blocked |= self.traps

    def route(self,start,goal):
        s=tuple(map(int,self.indexes(start)))
        g=tuple(map(int,self.indexes(goal)))
        if not (0<g[0]<self.nx-1 and 0<g[1]<self.ny-1 and 0<s[0]<self.nx-1 and 0<s[1]<self.ny-1):
            return []
        if self.blocked[g[1],g[0]]:
            return []
        direct=self.segment(start,goal)
        direct_price=self.segment_price(start,goal) if direct else math.inf
        if direct and (self.layout is None or not self.layout.enabled
                       or direct_price<=np.linalg.norm(np.subtract(start,goal))*1.11):
            return [tuple(start),tuple(goal)]
        queue=[(0.,0.,s)]
        prev={}
        cost={s:0.}
        moves=[(i,j,math.hypot(i,j)) for i in (-1,0,1) for j in (-1,0,1) if i or j]
        while queue:
            _,spent,p=heapq.heappop(queue)
            if spent>cost[p]:
                continue
            if p==g:
                break
            x,y=p
            for i,j,d in moves:
                u,v=x+i,y+j
                if not (0<u<self.nx-1 and 0<v<self.ny-1) or self.blocked[v,u]:
                    continue
                if i and j and (self.blocked[y,u] or self.blocked[v,x]):
                    continue
                price=spent+d*float(self.travel_cost[v,u])
                new=(u,v)
                if price<cost.get(new,math.inf):
                    cost[new],prev[new]=price,p
                    heapq.heappush(queue,(price+.88*math.hypot(u-g[0],v-g[1]),price,new))
        if g not in cost:
            return []
        points=[tuple(goal)]
        cur=g
        while cur!=s:
            points.append(tuple(self.mapped(cur)))
            cur=prev[cur]
        points.append(tuple(start))
        points.reverse()
        distances=np.r_[0,np.cumsum([self.segment_price(a,b) for a,b in zip(points,points[1:])])]
        if direct_price<=distances[-1]*1.05:
            return [tuple(start),tuple(goal)]
        # Smooth stair steps while preserving the weighted route choice.
        result=[points[0]]
        i=0
        while i<len(points)-1:
            j=next((k for k in range(len(points)-1,i,-1)
                    if self.segment(points[i],points[k])
                    and self.segment_price(points[i],points[k])<=(distances[k]-distances[i])*1.06),i+1)
            if np.linalg.norm(np.subtract(points[j],result[-1]))>.025:
                result.append(points[j])
            i=j
        return result


BACKTRACK="自救后退"
MANUAL="手动记忆"
ACTIVE=("驾驶中","减速停稳","视觉暂失","寻找绕行",BACKTRACK)
DONE="到达并停稳"
BACKTRACK_DISTANCE=1.0
BACKTRACK_MINIMUM=.65
BACKTRACK_LIMIT=round(8./DT)


class Runner:
    def __init__(self,world,net,research=None,memory=None):
        self.world,self.net=world,net
        if memory is None:
            reference=VisualMap(world.xs,world.ys)
            memory=MapMemory(reference,FieldLayout(reference,world.kind))
        self.memory=memory
        self.map=self.task_map()
        self.explorer=None
        self.motion_anchor=None
        self.manual_cmd=(0.,0.)
        self.manual_until=0.
        self.goal=None
        self.phase="等待目标"
        self.reason="在二维道路地图设定目标"
        self.route,self.trace=[],[]
        self.ticks,self.hold,self.latency,self.no_route=0,0,0.,0
        self.cmd=(0.,0.)
        self.distance=0.
        self.last_truth=world.data.qpos[:2].copy()
        self.ground_z=float(world.data.qpos[2])
        self.crossed=self.detoured=self.stuck=False
        self.relief=0.
        self.climb_start=None
        self.history=deque(maxlen=85)
        self.breadcrumbs=deque(maxlen=48)
        self.backtrack=deque()
        self.backtrack_started=0
        self.backtrack_origin=None
        self.backtrack_distance=0.
        self.backtrack_blocked=None
        self.trap=None
        self.recovery_count=0
        self.alert=None
        self.look=None
        self.visual=None
        self.vision_misses=0
        self.recoveries=0
        self.target_revision=0
        self.scan_origin=world.camera_yaw
        self.scan_step=0
        self.recenter_at=0
        self.research_net=research
        self.research=None
        if research is not None:
            from research import ResearchState
            self.research=ResearchState(research)
        self.research_tick=-1
        self.measure()

    def task_map(self):
        return VisualMap(self.world.xs,self.world.ys,layout=self.memory.layout,memory=self.memory)

    def measure(self):
        started=time.perf_counter()
        angle=self.world.camera_yaw
        if self.phase in (*ACTIVE,MANUAL):
            if self.vision_misses:
                self.scan_step+=1
                angle=self.scan_origin+math.radians(45)*((self.scan_step+1)//2)*(-1 if self.scan_step%2==0 else 1)
            elif self.phase==BACKTRACK:
                angle=math.pi
            elif self.ticks>=self.recenter_at:
                angle+=float(np.clip(wrap(-angle),-.23,.23))
        self.frame=self.world.observe(angle)
        if self.frame.pose is not None:
            if self.vision_misses:
                self.recoveries+=1
                self.recenter_at=self.ticks+12
                self.route=[]
            self.vision_misses=0
            self.scan_step=0
            self.visual=self.frame.pose.copy()
            if self.phase!=BACKTRACK and not self.world.belly and not self.world.collision and not self.world.flipped:
                xy=self.visual[:2]
                if not self.breadcrumbs or np.linalg.norm(xy-self.breadcrumbs[-1])>=.13:
                    self.breadcrumbs.append(xy.copy())
        else:
            if not self.vision_misses:
                self.scan_origin=angle
            self.vision_misses+=1
        if self.phase in (*ACTIVE,MANUAL):
            self.map.update(self.frame)
            self.map.assess(self.net,self.world.config.max_slope)
        if self.memory.recording is not None:
            e=self.explorer
            scope=self.memory.layout.scope(e.only) if e and e.mode=="semi" else None
            repeat=bool(e and e.mode=="auto" and e.stage=="scan" and self.phase=="驾驶中"
                        and abs(self.visual[0])<20.4 and abs(self.visual[1]-e.layout.aisles[e.number])<.72)
            self.memory.observe(self.frame,self.map,scope,repeat)
        self.vision_ms=1000*(time.perf_counter()-started)

    def can_configure(self):
        return (self.phase not in (*ACTIVE,MANUAL,"停车中")
                and np.linalg.norm(self.world.data.qvel[:2])<.05
                and np.linalg.norm(self.world.data.qvel[3:6])<.08)

    def configure(self,config: VehicleConfig):
        if not self.can_configure():
            return False,"请先停车并等待任务停止，再修改车辆参数"
        try:
            self.world.configure(config)
        except ValueError as exc:
            return False,str(exc)
        self.goal=None
        self.route=[]
        self.look=None
        self.breadcrumbs.clear()
        if self.research is not None:
            self.research=type(self.research)(self.research_net)
            self.research_tick=-1
        self.phase,self.reason="等待目标","车辆参数已应用，在二维道路地图设定目标"
        self.vision_misses=self.scan_step=0
        self.measure()
        return True,"车辆参数已应用"

    def start_memory(self,mode="manual",number=None):
        if mode not in ("manual","auto","semi"):
            return False,"建图模式无效"
        if not self.can_configure():
            return False,"车辆停稳后才可开始记忆"
        if self.world.collision or self.world.flipped or self.world.belly or self.stuck:
            return False,"请先重置小车"
        if self.frame.pose is None:
            self.measure()
        if self.frame.pose is None:
            return False,"未取得双目定位，暂无法设置建图起点"
        layout=self.memory.layout
        if mode!="manual" and not layout.enabled:
            return False,"此模式需要田垄场景"
        if mode=="semi":
            number=layout.closest(self.visual[1]) if number is None else number
            if type(number) is not int or number not in layout.aisles:
                return False,"请在有效的田垄间通道内建图"
        if mode=="auto" and len(self.memory.swept)==len(layout.aisles):
            return True,"已建图的通道将重复，全自动建图保持停车"
        self.memory.start(mode,self.visual)
        if mode=="manual":
            self.phase,self.reason=MANUAL,"按住方向键或遥控按钮采集地图"
            self.map=self.task_map()
            self.goal=self.look=None
            self.route,self.trace=[],[]
            self.ticks,self.distance=0,0.
            self.last_truth=self.world.data.qpos[:2].copy()
            self.manual_cmd=(0.,0.)
            self.manual_until=0.
            self.measure()
            return True,"开始手动记忆地图"
        self.map=self.task_map()
        self.map.update(self.frame)
        self.map.assess(self.net,self.world.config.max_slope)
        self.explorer=AisleExplorer(self.memory,mode,number)
        return self.explore_goal(self.explorer.next_target(self.visual))

    def explore_goal(self,targets):
        explorer=self.explorer
        message="没有可安全进入的建图通道"
        while targets:
            scope=self.memory.layout.scope(explorer.number) if explorer.stage=="scan" else None
            for target in targets:
                ok,message=self.goal_at(target,exploring=True,scope=scope)
                if ok:
                    return True,"已启动地图采集"
            targets=explorer.advance(self.visual,False)
        self.explorer=None
        self.memory.finish(message)
        return False,message

    def set_origin(self,point):
        if (len(point)!=2 or not np.isfinite(point).all()
                or abs(point[0])>self.world.xs or abs(point[1])>self.world.ys):
            return False,"原点须在场地内"
        self.memory.set_origin(point)
        return True,"已设置原点，已记地图位置保持原位"

    def manual_input(self,throttle,steering):
        if self.phase!=MANUAL:
            return False,"先按开始记忆"
        if not np.isfinite([throttle,steering]).all() or max(abs(throttle),abs(steering))>1.:
            return False,"遥控输入超出范围"
        self.manual_cmd=(.60*throttle,1.05*steering)
        self.manual_until=time.perf_counter()+.34
        return True,""

    def step_manual(self):
        self.ticks+=1
        allowed=time.perf_counter()<self.manual_until and self.frame.pose is not None
        self.cmd=self.manual_cmd if allowed else (0.,0.)
        self.reason="手动遥控 · 记忆双目地图" if self.frame.pose is not None else "等待双目恢复定位，保持停车"
        if self.cmd[0] and not self.safe_motion(*self.cmd):
            self.cmd=(0.,self.cmd[1])
            self.reason="田垄或风险格限制通行，请改变朝向"
        self.world.drive(*self.cmd)
        current=self.world.data.qpos[:2].copy()
        self.distance+=float(np.linalg.norm(current-self.last_truth))
        self.last_truth=current
        if self.world.collision or self.world.flipped or self.world.belly:
            self.stop()
            self.phase,self.reason="手动终止","检测到碰撞、倾覆或托底，已停止记忆"
        elif self.ticks%3==0:
            self.measure()

    def stop(self,reason="已结束建图"):
        self.cmd=self.manual_cmd=(0.,0.)
        self.manual_until=0.
        if self.memory.recording is not None:
            ok,message=self.memory.finish(reason)
        else:
            ok,message=True,"手动停车"
        self.explorer=None
        self.phase,self.reason="停车中",message
        return ok,message

    def goal_at(self,xy,*,exploring=False,scope=None):
        if self.memory.recording is not None and not exploring:
            return False,"请先结束记忆"
        if (len(xy)!=2 or not np.isfinite(xy).all()
                or not (-self.world.xs+CELL/2<xy[0]<self.world.xs-CELL/2
                        and -self.world.ys+CELL/2<xy[1]<self.world.ys-CELL/2)):
            return False,"目标超出场地或坐标无效"
        xy=tuple(float(v) for v in xy)
        x,y=self.map.indexes(xy)
        if self.memory.layout.ridges[y,x]:
            return False,"目标在禁行田垄上，请选择通道或地头内的点"
        if self.phase==BACKTRACK:
            self.goal=xy
            self.target_revision+=1
            return True,"新目标已接收，退出受困区域后重规划"
        if self.world.collision or self.world.flipped or self.world.belly or self.stuck:
            return False,"先重置小车"
        changing=self.phase in ACTIVE and self.goal is not None
        candidate=self.map if changing or exploring else self.task_map()
        if not changing and not exploring:
            candidate.update(self.frame)
            candidate.assess(self.net,self.world.config.max_slope)
        candidate.scope=scope
        candidate.refresh_constraints()
        if candidate.blocked[y,x]:
            return False,"目标位置有障碍或地形风险，请选择附近可通行位置"
        route=[] if self.frame.pose is None else candidate.route(self.visual[:2],xy)
        if self.frame.pose is not None and len(route)<2 and np.linalg.norm(np.subtract(self.visual[:2],xy))>=.4:
            return False,"当前地图没有到该坐标的安全通路"
        self.map,self.route=candidate,route
        self.goal=xy
        self.target_revision+=1
        self.ticks,self.hold,self.no_route,self.distance=0,0,0,0.
        self.motion_anchor=None
        self.last_truth=self.world.data.qpos[:2].copy()
        self.ground_z=float(self.world.data.qpos[2])
        self.history.clear()
        self.trace=[self.last_truth.tolist()]
        self.crossed=self.detoured=self.stuck=False
        self.relief=0.
        self.climb_start=None
        self.look=None
        self.research_tick=-1
        self.recenter_at=0
        if self.frame.pose is None:
            if self.vision_misses>=60:
                self.vision_misses,self.scan_step=1,0
                self.scan_origin=self.world.camera_yaw
            self.phase,self.reason="视觉暂失","新目标已接收，等待双目云台恢复定位"
        else:
            self.phase,self.reason="驾驶中","正在根据当前车位重规划" if changing else "正在评估前方地形"
        if not changing:
            self.measure()
        return True,("新目标已接收，正在重规划" if changing else "目标已接收")

    def safe_motion(self,speed,rate):
        heading=self.visual[3]
        duration=.45
        turned=heading+rate*duration/2
        offset=np.array([math.cos(turned),math.sin(turned)])*math.copysign(max(.28,abs(speed)*duration),speed)
        xy=self.visual[:2]
        x,y=self.map.indexes(np.linspace(xy,xy+offset,5)).T
        if np.any((x<1)|(x>=self.map.nx-1)|(y<1)|(y>=self.map.ny-1)):
            return False
        hard=self.memory.layout.ridges[y,x]|self.map.rocks_blocked[y,x]
        leaving=(x!=x[0])|(y!=y[0])
        # A new terrain estimate can mark the cell under the tracks. The
        # planned egress may leave it at low speed, with hard barriers intact.
        return not np.any(hard| (self.map.blocked[y,x]&leaving))

    def begin_backtrack(self,cause,mark_trap=True):
        self.phase,self.reason=BACKTRACK,cause+"，云台转向车后并准备沿视觉轨迹后退"
        self.backtrack_started=self.ticks
        self.backtrack.clear()
        self.backtrack_origin=self.visual[:2].copy() if self.visual is not None else None
        self.backtrack_distance=0.
        self.backtrack_blocked=None
        self.trap=None
        self.route=[]
        self.cmd=(0.,0.)
        self.history.clear()
        if self.backtrack_origin is None:
            return
        xy=self.backtrack_origin
        heading=self.visual[3]
        forward=np.array([math.cos(heading),math.sin(heading)])
        if mark_trap:
            self.trap=xy+.55*forward
        for point in reversed(self.breadcrumbs):
            delta=point-xy
            length=float(np.linalg.norm(delta))
            if length<.08:
                continue
            # A path beginning in front of the chassis cannot be retraced in reverse.
            if not self.backtrack and np.dot(delta,forward)>-.04:
                break
            remaining=BACKTRACK_DISTANCE-self.backtrack_distance
            self.backtrack.append(xy+delta*min(1.,remaining/length))
            self.backtrack_distance+=min(remaining,length)
            if self.backtrack_distance>=BACKTRACK_DISTANCE:
                break
            xy=point
        self.measure()

    def fail_backtrack(self,reason):
        self.stuck=True
        self.phase,self.reason="自救失败",reason
        self.alert={"message":reason,"seconds":round(self.ticks*DT,2)}
        self.cmd=(0.,0.)

    def reverse_command(self):
        if self.backtrack_distance<BACKTRACK_MINIMUM:
            self.backtrack_blocked="已行驶的视觉轨迹不足，无法安全后退"
            return 0.,0.
        if self.frame.pose is None or abs(wrap(self.world.camera_yaw-math.pi))>math.pi/3:
            self.reason="停车，等待车后双目定位"
            return 0.,0.
        xy=self.visual[:2]
        while self.backtrack and np.linalg.norm(self.backtrack[0]-xy)<.13:
            self.backtrack.popleft()
        self.route=[xy.tolist(),*(point.tolist() for point in self.backtrack)]
        if not self.backtrack:
            return 0.,0.
        delta=self.backtrack[0]-xy
        distance=np.linalg.norm(delta)
        behind=xy+delta/distance*max(distance,.30)
        x,y=self.map.indexes(behind)
        if (not (0<x<self.map.nx-1 and 0<y<self.map.ny-1)
                or self.map.rocks_blocked[y,x] or self.memory.layout.ridges[y,x]):
            self.backtrack_blocked="后退路线检测到障碍或场地边界"
            return 0.,0.
        angle=float(wrap(math.atan2(delta[1],delta[0])+math.pi-self.visual[3]))
        self.reason=f"沿已行驶轨迹后退，计划 {self.backtrack_distance:.2f} m"
        speed=-.32*max(0.,1-abs(angle)/.8)
        return speed,float(np.clip(2.6*angle,-.8,.8))

    def finish_backtrack(self):
        if self.trap is not None:
            self.map.mark_trap(self.trap)
        self.recovery_count+=1
        self.breadcrumbs.clear()
        self.breadcrumbs.append(self.visual[:2].copy())
        self.history.clear()
        self.route=[]
        self.no_route=self.hold=0
        self.climb_start=None
        self.detoured=True
        self.phase,self.reason="驾驶中","自救成功，重新规划路线"

    def decide(self):
        if self.frame.pose is None:
            self.phase,self.reason="视觉暂失","底盘停车，双目云台旋转搜索视觉地标"
            if self.research is not None and self.research_tick!=self.ticks//3:
                self.research.assess(np.full((7,7),.34,np.float32),.34,0.,False,False)
                self.research_tick=self.ticks//3
            return (0.,0.)
        pose=self.visual
        xy=pose[:2]
        aim=self.goal
        remaining=np.linalg.norm(np.subtract(aim,xy))
        if remaining<.40:
            self.phase,self.reason="减速停稳","已进入目标停稳范围"
            return 0.,0.
        if self.ticks%8==1 or (len(self.route)<2 and (self.no_route==0 or self.no_route%4==0)):
            self.route=self.map.route(xy,aim)
        if len(self.route)<2:
            self.no_route+=1
            self.phase,self.reason="寻找绕行","当前没有可行绕行路线，停止推进"
            return 0.,0.
        self.no_route=0
        self.phase="驾驶中"
        blocked_direct=not self.map.segment(xy,aim)
        self.detoured |= blocked_direct
        points=self.route[1:]
        while (len(points)>1 and np.linalg.norm(np.subtract(points[0],xy))<.43
               and self.map.segment(xy,points[1])):
            points=points[1:]
        point=points[0]
        delta=np.subtract(point,xy)
        look=xy+delta/max(np.linalg.norm(delta),.001)*min(np.linalg.norm(delta),.9)
        known,blocked,risk,grade,score=self.map.fields(look)
        self.look={"point":look.tolist(),"known":bool(known),"blocked":blocked,
                   "pass_probability":1-risk if known else None,"grade":grade if known else None,
                   "network_reward":score if known else None,"detour":blocked_direct}
        crossing=known and grade>.12 and not blocked
        if crossing and self.climb_start is None:
            self.climb_start=self.distance
        if self.climb_start is not None and self.distance-self.climb_start>1.1:
            if not crossing and self.relief>.15:
                self.crossed=True
                self.climb_start=None
        if blocked_direct:
            self.reason="按可通行区域绕过田垄或实时发现的障碍"
        elif crossing:
            self.reason="网络判定可通过，减速跨越起伏"
        else:
            self.reason="双目地图持续更新，跟随当前路线"
        if self.explorer is not None:
            self.reason=f"通道 {self.explorer.number} 建图 · {self.reason}"
        heading=math.atan2(delta[1],delta[0])
        angle=float(wrap(heading-pose[3]))
        rate=float(np.clip(2.8*angle,-1.15,1.15))
        speed=min(1.18,max(.16,(remaining-.25)*1.35))*max(0,1-abs(angle)/1.05)
        if crossing:
            speed*=max(.46,1-1.15*grade)
        if not known:
            speed=min(speed,.70)
        if max(abs(pose[4:]))>.46:
            speed=min(speed,.25)
        if abs(self.world.camera_yaw)>.82:
            speed=min(speed,.32)
        if self.research is not None:
            if self.research_tick!=self.ticks//3:
                self.research.assess(self.map.local_patch(look,heading),risk if known else .34,
                                     grade if known else 0.,known,True)
                self.research_tick=self.ticks//3
            evidence=max(self.research.values.get("local_risk",0.),self.research.values.get("memory_risk",0.))
            speed*=1-.20*max(0.,(evidence-.45)/.55)
        if self.memory.layout.enabled:
            x,y=self.map.indexes(xy)
            if self.map.blocked[y,x]:
                speed=min(speed,.28)
            if speed and not self.safe_motion(speed,rate):
                speed=0.
                self.route=[]
        return float(speed),rate

    def step(self):
        if self.phase not in ACTIVE:
            return
        t=time.perf_counter()
        self.ticks+=1
        reversing=self.phase==BACKTRACK
        self.cmd=self.reverse_command() if reversing else self.decide()
        if reversing:
            self.world.drive(*self.cmd,recovering=True)
        else:
            self.world.drive(*self.cmd)
        truth=self.world.data.qpos[:3]
        travelled=float(np.linalg.norm(truth[:2]-self.last_truth))
        self.distance+=travelled
        self.relief=max(self.relief,abs(float(truth[2])-self.ground_z))
        self.last_truth=truth[:2].copy()
        if np.linalg.norm(np.subtract(self.trace[-1],truth[:2]))>.06:
            self.trace.append(truth[:2].tolist())
        self.history.append((travelled,self.cmd[0]))
        if reversing:
            if self.world.collision or self.world.flipped:
                self.fail_backtrack("自救时检测到碰撞或倾覆，已停车")
            elif self.backtrack_blocked:
                self.fail_backtrack(self.backtrack_blocked+"，已停车")
            elif not self.backtrack and self.world.belly_steps==0:
                self.finish_backtrack()
            elif self.ticks-self.backtrack_started>=BACKTRACK_LIMIT:
                self.fail_backtrack("8 秒内未能沿轨迹退出受困区域，已停车")
        elif self.world.collision:
            self.phase,self.reason="碰撞终止","检测到实体岩石接触"
        elif self.world.flipped:
            self.phase,self.reason="翻车终止","车体倾斜超限"
        elif self.world.belly or (len(self.history)==self.history.maxlen
                                   and sum(v for v,_ in self.history)<.12
                                   and sum(v>.25 for _,v in self.history)>65):
            self.begin_backtrack("底盘托底或持续驱动没有有效位移")
        elif self.no_route>=55:
            if self.explorer:
                self.phase,self.reason="没有绕行路线","当前通道视野内暂无安全通路"
            else:
                self.begin_backtrack("连续规划未找到通路",mark_trap=False)
        elif self.phase=="减速停稳":
            self.hold = self.hold+1 if np.linalg.norm(self.world.data.qvel[:2])<.13 else 0
            if self.hold>=10:
                self.phase,self.reason=DONE,"到点停稳，结算路线奖励"
        else:
            self.hold=0
        if self.vision_misses>=60 and self.phase=="视觉暂失":
            self.phase,self.reason="定位未恢复","已扫描环视，保持停车，可重新设置目标或重置"
        elif self.ticks>=2000 and self.phase in ACTIVE and self.phase!=BACKTRACK:
            self.phase,self.reason="超时结束","未在本回合时间内到达"
        if self.ticks%3==0:
            self.measure()
            self.check_motion_progress()
        if self.explorer is not None:
            self.advance_exploration()
        self.latency=1000*(time.perf_counter()-t)

    def check_motion_progress(self):
        if self.phase!="驾驶中" or self.frame.pose is None or not self.memory.layout.enabled:
            self.motion_anchor=None
            return
        pose=self.visual
        anchor=self.motion_anchor
        if (anchor is None or np.linalg.norm(pose[:2]-anchor[1][:2])>.25
                or abs(wrap(pose[3]-anchor[1][3]))>.23):
            self.motion_anchor=(self.ticks,pose.copy())
        elif self.ticks-anchor[0]>=120:
            if self.explorer:
                self.phase,self.reason="没有绕行路线","车辆未能推进，跳过受阻区域"
            else:
                self.begin_backtrack("持续驱动或转向没有有效进展")

    def advance_exploration(self):
        if self.memory.repeat_distance>=3.0:
            self.stop("已确认重复经过且无新增区域，全自动建图结束")
        elif self.phase not in ACTIVE:
            explorer=self.explorer
            if self.phase==DONE or self.phase=="没有绕行路线":
                reached=self.phase==DONE
                target=explorer.advance(self.visual,reached)
                if target is not None:
                    self.explore_goal(target)
                elif explorer.mode=="semi":
                    self.stop("已到通道尽头，半自动建图完成" if reached else self.reason)
                else:
                    reason=("可行通道已扫过，继续将重复" if not explorer.failed
                            else f"已完成可达通道建图，{len(explorer.failed)} 条通道未完成")
                    self.stop(reason)
            else:
                self.explorer=None
                self.memory.finish(self.reason)

    @property
    def reward(self):
        return terrain_reward(self.distance,self.ticks*DT,self.phase==DONE,self.world.collision,
                              self.world.flipped,self.stuck,self.crossed,self.detoured and not self.memory.layout.enabled)
