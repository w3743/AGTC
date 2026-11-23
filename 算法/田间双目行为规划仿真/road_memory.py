"""Remember observed map cells and label inter-row corridors by field geometry."""
from dataclasses import replace
import numpy as np
from scipy import ndimage

from farm import ROW_SPACING, ROWS


class FieldLayout:
    def __init__(self,grid,kind):
        self.enabled=kind=="farm"
        self.ridge_ys=ROWS if self.enabled else ()
        self.aisles={i+1:(a+b)/2 for i,(a,b) in enumerate(zip(self.ridge_ys,self.ridge_ys[1:]))}
        self.end=21.2
        iy,ix=np.indices(grid.hit.shape)
        self.x,self.y=grid.mapped(np.stack((ix,iy),axis=-1)).transpose(2,0,1)
        self.ridges=np.zeros_like(grid.hit,dtype=bool)
        self.crop_margin=self.ridges.copy()
        for y in self.ridge_ys:
            self.ridges |= self.interior & (np.abs(self.y-y)<.66)
            self.crop_margin |= self.interior & (np.abs(self.y-y)<.86)

    @property
    def interior(self):
        return np.abs(self.x)<=20.4

    def closest(self,y):
        return min(self.aisles,key=lambda number:abs(self.aisles[number]-y))

    def scope(self,number):
        return (np.abs(self.y-self.aisles[number])<=ROW_SPACING/2+.40)&(np.abs(self.x)<=self.end+.25)


class MapMemory:
    def __init__(self,reference,layout):
        self.reference=reference
        self.layout=layout
        self.known=np.zeros_like(reference.hit,dtype=bool)
        self.occupied=self.known.copy()
        self.free=self.known.copy()
        self.driven=self.known.copy()
        self.origin=None
        self.recording=None
        self.revision=0
        self.swept=set()
        self.result=""
        self.previous_drive=self.driven.copy()
        self.visit_order=np.zeros_like(reference.hit,dtype=np.int32)
        self.motion_count=0
        self.last_position=None
        self.repeat_distance=0.

    def set_origin(self,point):
        self.origin=tuple(float(v) for v in point)
        self.revision+=1

    def restore_reference(self,payload):
        points=np.asarray(payload["reference"],dtype=float)
        if points.size==0:
            points=points.reshape(0,4)
        if (points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all()
                or not np.isin(points[:,3],[0,1]).all()):
            raise ValueError("旧版三维参考格式无效")
        grid=self.reference
        if np.any(np.abs(points[:,:2])>[grid.xs,grid.ys]):
            raise ValueError("旧版三维参考超出新场地")
        x,y=grid.indexes(points[:,:2]).T
        if np.any((x<0)|(x>=grid.nx)|(y<0)|(y>=grid.ny)):
            raise ValueError("旧版三维参考超出新场地")
        cells=np.asarray(payload.get("cells",[]),dtype=float)
        if cells.size==0:
            cells=cells.reshape(0,3)
        if (cells.ndim!=2 or cells.shape[1]!=3 or not np.isfinite(cells).all()
                or not np.isin(cells[:,2],[1,2,3,4]).all()):
            raise ValueError("二维地图格式无效")
        if np.any(np.abs(cells[:,:2])>[grid.xs,grid.ys]):
            raise ValueError("二维地图超出场地")
        cx,cy=grid.indexes(cells[:,:2]).T
        if np.any((cx<0)|(cx>=grid.nx)|(cy<0)|(cy>=grid.ny)):
            raise ValueError("二维地图超出场地")
        origin=payload.get("origin")
        if origin is not None:
            origin=np.asarray(origin,dtype=float)
            if origin.shape!=(2,) or not np.isfinite(origin).all() or np.any(abs(origin)>[grid.xs,grid.ys]):
                raise ValueError("地图原点无效")
        corridors=payload.get("corridors",[])
        if not isinstance(corridors,list):
            raise ValueError("通道状态格式无效")
        ground=points[:,3]==0
        gx,gy=x[ground],y[ground]
        rx,ry=x[~ground],y[~ground]
        grid.hit[gy,gx]=1.
        grid.z[gy,gx]=points[ground,2]
        grid.rocks[ry,rx]=1
        grid.rock_z[ry,rx]=points[~ground,2]
        self.known[y,x]=True
        self.known[cy,cx]=True
        self.free[cy,cx]=cells[:,2]==1
        self.occupied[cy,cx]=cells[:,2]==2
        if not payload.get("rock_footprint_only"):
            rock=(grid.rocks>0)&self.occupied
            self.occupied[ndimage.maximum_filter(rock,size=3)&~rock]=False
        for lane in corridors:
            if isinstance(lane,dict) and type(lane.get("number")) is int and lane.get("scanned") is True:
                if lane["number"] in self.layout.aisles:
                    self.swept.add(lane["number"])
        if origin is not None:
            self.set_origin(origin)
        self.result=("已恢复地图，旧岩石缓冲格待重新确认" if len(cells)
                     else "已迁移旧版双目测点，待确认通行状态")
        self.revision+=1

    def start(self,mode,pose):
        if self.origin is None:
            self.set_origin(pose[:2])
        self.recording=mode
        self.previous_drive=self.driven.copy()
        self.visit_order.fill(0)
        self.motion_count=0
        self.last_position=None
        self.repeat_distance=0.
        self.result=""
        self.revision+=1

    def observe(self,frame,live,scope=None,check_repeat=False):
        if frame.pose is None:
            return
        eligible=np.ones_like(self.known) if scope is None else scope
        if scope is not None:
            ix,iy=live.indexes(frame.cloud[:,:2]).T
            keep=scope[iy,ix]
            frame=replace(frame,cloud=frame.cloud[keep],rock=frame.rock[keep])
        self.reference.update(frame)
        seen=(self.reference.hit>0)|(self.reference.rocks>0)
        new=int(np.count_nonzero(eligible & seen & ~self.known))
        self.known |= eligible & seen
        reliable=eligible & (live.hit>=4) & (self.reference.hit>0)
        self.occupied[reliable]=live.live_blocked[reliable]
        self.occupied |= eligible & live.rocks_blocked & seen
        self.free[reliable]=~live.live_blocked[reliable]
        self.free &= ~self.occupied & ~self.layout.ridges
        xy=frame.pose[:2]
        x,y=live.indexes(xy)
        if 0<x<live.nx-1 and 0<y<live.ny-1 and eligible[y,x]:
            if self.last_position is None:
                self.last_position=xy.copy()
            distance=float(np.linalg.norm(xy-self.last_position))
            if distance>=.16:
                history=self.visit_order[y-1:y+2,x-1:x+2]
                repeated=(np.any(self.previous_drive[y-1:y+2,x-1:x+2])
                          or np.any((history>0)&(self.motion_count-history>=24)))
                self.repeat_distance=self.repeat_distance+distance if check_repeat and repeated and new<4 else 0.
                self.driven[y,x]=True
                self.motion_count+=1
                self.visit_order[y,x]=self.motion_count
                self.last_position=xy.copy()
        self.revision+=1

    def finish(self,reason):
        self.recording=None
        self.result=reason
        self.revision+=1
        return True,f"{reason} · {int(self.known.sum())} 个已记格"

    def corridor_coverage(self,number):
        middle=self.layout.aisles[number]
        lane=self.layout.interior & (np.abs(self.layout.y-middle)<.64)
        columns=np.any(lane,axis=0)
        observed=np.any(self.known & lane,axis=0)
        return float(np.mean(observed[columns]))

    def payload(self):
        grid=self.reference
        types=np.where(self.layout.ridges,3,np.where(self.occupied,2,np.where(self.free,1,4)))
        cells=np.column_stack((self.layout.x[self.known],self.layout.y[self.known],types[self.known]))
        ground=grid.hit>0
        rocks=grid.rocks>0
        points=np.column_stack((self.layout.x[ground],self.layout.y[ground],grid.z[ground],np.zeros(ground.sum())))
        obstacles=np.column_stack((self.layout.x[rocks],self.layout.y[rocks],grid.rock_z[rocks],np.ones(rocks.sum())))
        corridors=[]
        for number,y in self.layout.aisles.items():
            visible=self.known & self.layout.interior & (np.abs(self.layout.y-y)<.47)
            if np.any(visible):
                x=float(np.median(self.layout.x[visible]))
                corridors.append({"number":number,"x":round(x,2),"y":round(y,2),"scanned":number in self.swept})
        ridges=[]
        for i,y in enumerate(self.layout.ridge_ys,1):
            visible=self.known & self.layout.interior & (np.abs(self.layout.y-y)<.66)
            if np.any(visible):
                ridges.append({"number":i,"x":round(float(np.median(self.layout.x[visible])),2),"y":round(y,2)})
        return {"revision":self.revision,"origin":self.origin,"cells":np.round(cells,2).tolist(),
                "rock_footprint_only":True,
                "corridors":corridors,"ridges":ridges,"recording":self.recording,
                "result":self.result,"reference":np.round(np.vstack((points,obstacles)),3).tolist()}


class AisleExplorer:
    def __init__(self,memory,mode,lane=None):
        self.memory=memory
        self.layout=memory.layout
        self.mode=mode
        self.only=lane
        self.number=None
        self.stage=None
        self.exit=None
        self.failed=set()
        self.backside=set()

    def scan_targets(self,pose):
        x=float(pose[0])
        end,y=self.exit
        sign=1. if end>x else -1.
        relative=abs(end-x)
        if relative<=1.0:
            return [self.exit]
        targets=[]
        for span in (6.,4.,2.,1.1):
            target=(x+sign*min(relative,span),y)
            if target not in targets:
                targets.append(target)
        return targets

    def next_target(self,pose):
        x,y,_,heading=pose[:4]
        options=[self.only] if self.mode=="semi" else [
            n for n in self.layout.aisles if n not in self.memory.swept and n not in self.failed]
        if not options:
            return None
        self.number=min(options,key=lambda n: (abs(self.layout.aisles[n]-y),self.memory.corridor_coverage(n),n))
        cy=self.layout.aisles[self.number]
        end=self.layout.end
        if self.mode=="semi" and abs(y-cy)<.72 and abs(x)<end-.65:
            self.stage="scan"
            self.exit=(end if np.cos(heading)>=0 else -end,cy)
            return self.scan_targets(pose)
        entrance=(-end if x<0 else end,cy)
        self.exit=(-entrance[0],cy)
        self.stage="join"
        return [entrance]

    def advance(self,pose,success):
        if success and self.stage=="join":
            self.stage="scan"
            return self.scan_targets(pose)
        if success and abs(self.exit[0]-pose[0])>.55:
            return self.scan_targets(pose)
        if not success and self.stage=="scan" and self.mode=="auto" and self.number not in self.backside:
            self.backside.add(self.number)
            entrance=self.exit
            self.exit=(-entrance[0],entrance[1])
            self.stage="join"
            return [entrance]
        if success and self.mode=="auto":
            if self.memory.corridor_coverage(self.number)>=.70:
                self.memory.swept.add(self.number)
            else:
                self.failed.add(self.number)
            self.memory.revision+=1
        elif not success:
            self.failed.add(self.number)
        if self.mode=="semi":
            return None
        return self.next_target(pose)
