"""AGTC rough-terrain physics, stereo reconstruction and vision-only state."""
from dataclasses import dataclass
import colorsys
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
import cv2
from farm import FARM_START, FARM_XS, FARM_YS, heights as farm_heights, plant_mesh, scattered_rocks, vegetation


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def episode_reward(distance, seconds, success, collision, flipped):
    arrived=success and not (collision or flipped)
    return 18*arrived - .65*distance - .28*seconds - 32*(collision or flipped)


cv2.setNumThreads(4)
XS, YS = 12., 10.
N = 129
X, Y = np.meshgrid(np.linspace(-XS, XS, N), np.linspace(-YS, YS, N))
CLEARANCE, HULL = .50, .075
BASE_Z = CLEARANCE + HULL
W, H, FOV, BASELINE = 320, 208, 67., .096286563
FOCAL = H / (2 * math.tan(math.radians(FOV / 2)))
CX, CY = (W - 1) / 2, (H - 1) / 2
MOUNT = np.array([.34, 0., .70-BASE_Z])
DOWN = .13
CAM_ROT = Rotation.from_euler("Y", DOWN).as_matrix()
HUES = np.linspace(.14, .94, 48)
PHI = np.arange(len(HUES)) * 2 * math.pi / len(HUES)
RADII = np.resize([8.3, 13.9, 17.], len(PHI))
TAGS = np.column_stack((np.cos(PHI)*RADII, np.sin(PHI)*RADII,
                       1.35 + np.resize([0., .6, 1.10, .30], len(PHI)) + (RADII<10)*1.2))
FIELD_BEACONS = np.resize([1.,.94,1.07],len(PHI))
FARM_TAGS = np.column_stack((
    36.*np.cos(PHI)*FIELD_BEACONS,27.*np.sin(PHI)*FIELD_BEACONS,
    np.resize([1.15,1.80,2.35,1.50,2.05],len(PHI))))
DT = .05
START = (-9., 0., 0.)
GOAL = (8.5, 0.)
ROCKS = [(-4.8, 2.15, .70, .50, .85, -.22), (-.25, -2.5, .88, .64, .88, .45),
         (4.55, -1.65, .67, .61, .69, .22), (5.5, 3.10, .94, .62, .92, -.8),
         (-6.5, -3.7, .68, .46, .74, .3)]


@dataclass(frozen=True)
class CameraProfile:
    capture_size: tuple[int,int]
    match_size: tuple[int,int]
    fps: int | None = None

    @property
    def capture_focal(self):
        return self.capture_size[1] / (2*math.tan(math.radians(FOV/2)))

    @property
    def focal(self):
        return self.capture_focal*self.match_size[1]/self.capture_size[1]

    @property
    def principal(self):
        return tuple((n-1)/2 for n in self.match_size)


STEREO_SIZE=(1024,576)
PREVIEW_CAMERA=CameraProfile(STEREO_SIZE,STEREO_SIZE)
TRAINING_CAMERA=CameraProfile((3840,2160),STEREO_SIZE,30)


class StereoClock:
    """Schedule paired exposures on the simulation clock."""
    def __init__(self,fps):
        self.fps=fps
        self.next_frame=0

    def poll(self,seconds):
        if seconds+1.e-9 < (self.next_frame+1)/self.fps:
            return None
        frame=self.next_frame
        self.next_frame+=1
        return frame


@dataclass(frozen=True)
class VehicleConfig:
    height: float = .80
    camera_height: float = .70
    max_slope: float = 30.
    camera_yaw: float = 0.

    def validated(self):
        values=(self.height,self.camera_height,self.max_slope,self.camera_yaw)
        if not np.isfinite(values).all():
            raise ValueError("车辆参数必须是有限数值")
        if not .68<=self.height<=1.40 or not .60<=self.camera_height<=self.height:
            raise ValueError("车高 68–140 cm，相机高至少 60 cm 且不得超过车高")
        if not 5<=self.max_slope<=30:
            raise ValueError("最大爬坡角须在 5°–30°")
        return self


def make_heights(kind="mixed"):
    z = .61*np.exp(-((X+5)/3.05)**2-(Y/2.95)**2)
    if kind != "cross":
        z += .90*np.exp(-((X-1.45)/.62)**2-(Y/1.6)**4)
    z += .88*np.exp(-((X+3.5)/2.8)**2-((Y-6.1)/2.8)**2)
    z -= .93*np.exp(-((X-5.4)/2.6)**2-((Y+5.35)/2.9)**2)
    z += .035*np.sin(.82*X + .55*Y)*np.cos(.62*Y)
    return 2*(z-z.min())/(z.max()-z.min()) - 1


def terrain_reward(distance, seconds, success, collision, flipped, stuck, crossed, detoured):
    clean = success and not (collision or flipped or stuck)
    return episode_reward(distance, seconds, clean, collision, flipped) + (3.0*crossed + 2.0*detoured)*clean - 18.0*stuck


def floor_texture():
    rng = np.random.default_rng(6915)
    tex = np.zeros((1024, 1024), np.float32)
    for size, gain in ((320, .27), (105, .38), (39, .35)):
        tex += gain * cv2.resize(rng.random((size, size), dtype=np.float32), tex.shape, interpolation=cv2.INTER_LINEAR)
    colours = (np.array([42, 60, 66]) + tex[:,:,None]*np.array([126,144,156])).astype(np.uint8)
    return cv2.imencode(".png", colours)[1].tobytes()


def surface(z, x, y, xs=XS, ys=YS):
    rows,cols=z.shape
    u = float(np.clip((x+xs)*(cols-1)/(2*xs),0,cols-1.00001))
    v = float(np.clip((y+ys)*(rows-1)/(2*ys),0,rows-1.00001))
    i, j = int(u), int(v)
    a, b = u-i, v-j
    return float((1-b)*((1-a)*z[j,i]+a*z[j,i+1])+b*((1-a)*z[j+1,i]+a*z[j+1,i+1]))


def scene(z, rocks,camera=PREVIEW_CAMERA,*,xs=XS,ys=YS,tags=TAGS,plants=()):
    xml = ET.parse(ROOT / "chassis.xml").getroot()
    if camera.capture_size!=(W,H):
        w,h=camera.capture_size
        ET.SubElement(xml.find("visual"),"global",offwidth=str(w),offheight=str(h))
    xml.find("option").set("gravity","0 0 -9.81")
    assets, world = xml.find("asset"), xml.find("worldbody")
    assets.remove(assets.find("texture[@name='tiles']"))
    assets.insert(0, ET.Element("texture", name="tiles", type="2d", file="earth.png"))
    assets.find("material[@name='ground_tiles']").attrib.update(texrepeat="1 1", emission=".9")
    ET.SubElement(assets, "hfield", name="relief", size=f"{xs} {ys} 2 .10",
                  nrow=str(z.shape[0]), ncol=str(z.shape[1]),
                  elevation=" ".join(f"{v:.7f}" for v in ((z[::-1]+1)/2).ravel()))
    ground = world.find("geom[@name='ground']")
    ground.attrib.pop("size")
    ground.attrib.update(type="hfield", hfield="relief", pos="0 0 -1")
    base = world.find("body[@name='base']")
    for child in base.findall("body"):
        if "wheel" in child.get("name", "") or "guide" in child.get("name", ""):
            p = child.get("pos").split()
            p[2] = str(float(p[2]) - (BASE_Z-.15))
            child.set("pos"," ".join(p))
            child.find("geom").set("friction", ".025 .001 .0001")
    base.insert(0, ET.Element("inertial", pos="0 0 -.26", mass="26.4", diaginertia=".85 2.0 2.15"))
    for cam, side in zip(base.findall("camera"), (1,-1)):
        cam.set("pos",f"{MOUNT[0]} {side*BASELINE/2} {MOUNT[2]}")
        cam.set("xyaxes",f"0 -1 0 {math.sin(DOWN)} 0 {math.cos(DOWN)}")
        cam.set("fovy",str(FOV))
    for tag in list(world.findall("geom")):
        if tag.get("name","").startswith("tag"):
            world.remove(tag)
    for i, ((x,y,h), hue) in enumerate(zip(tags,HUES)):
        name=f"beacon{i}"
        colour=colorsys.hsv_to_rgb(float(hue), .97, .84)
        ET.SubElement(assets,"material", name=name, emission=".85",specular="0")
        ET.SubElement(world,"geom", name=name,type="sphere",size=".31",
                      pos=f"{x} {y} {h}",material=name,
                      rgba=" ".join(map(str,(*colour,1))),contype="0",conaffinity="0")
    for i,(x,y,rx,ry,h,yaw) in enumerate(rocks):
        rng=np.random.default_rng(105+i)
        phi=np.arange(7)*2*math.pi/7
        r=rng.uniform(.73,1.05,len(phi))
        lower=np.column_stack((r*rx*np.cos(phi),r*ry*np.sin(phi),np.full(len(phi),-.15)))
        upper=lower.copy()
        upper[:,:2] *= rng.uniform(.36,.74,(len(phi),1))
        upper[:,2] = h * rng.uniform(.52,1.05,len(phi))
        vertices=np.vstack((lower,upper))
        name=f"rock{i}"
        ET.SubElement(assets,"mesh",name=name,vertex=" ".join(f"{p:.4f}" for p in vertices.ravel()))
        z0=surface(z,x,y,xs,ys)
        ET.SubElement(world,"geom", name=name,type="mesh",mesh=name,
                      pos=f"{x} {y} {z0}",quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}",
                      rgba="1 .29 .06 1",material="obstacle")
    if plants:
        for kind,(name,colour) in enumerate((("wheat",".75 .56 .26 1"),("weeds",".28 .44 .17 1"))):
            vertices,faces=plant_mesh(plants,kind)
            ET.SubElement(assets,"mesh",name=name,vertex=" ".join(f"{p:.4f}" for p in vertices.ravel()),
                          face=" ".join(str(i) for i in faces.ravel()),maxhullvert="12")
            ET.SubElement(world,"geom",name=name,type="mesh",mesh=name,rgba=colour,contype="0",conaffinity="0")
    return ET.tostring(xml,encoding="unicode")


def markers(img):
    h,w=img.shape[:2]
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV_FULL)
    ys,xs=np.nonzero(cv2.inRange(hsv,(30,97,97),(246,255,255)))
    samples=hsv[ys,xs]
    chroma=np.multiply(samples[:,1],samples[:,2],dtype=np.float32)>.38*255**2
    samples,xs,ys=samples[chroma],xs[chroma],ys[chroma]
    ids=np.rint((samples[:,0]/256.-HUES[0])/(HUES[1]-HUES[0])).astype(int)
    valid=(ids>=0)&(ids<len(HUES))
    ids,xs,ys=ids[valid],xs[valid],ys[valid]
    n=np.bincount(ids,minlength=len(HUES))
    a=np.bincount(ids,weights=xs,minlength=len(HUES))
    b=np.bincount(ids,weights=ys,minlength=len(HUES))
    x0,y0,x1,y1=np.full(len(HUES),w),np.full(len(HUES),h),np.zeros(len(HUES)),np.zeros(len(HUES))
    np.minimum.at(x0,ids,xs)
    np.minimum.at(y0,ids,ys)
    np.maximum.at(x1,ids,xs)
    np.maximum.at(y1,ids,ys)
    return {k: (a[k]/n[k], b[k]/n[k]) for k in np.flatnonzero(n>=8)
            if 1 < x0[k] and x1[k]<w-2 and 1<y0[k] and y1[k]<h-2
            and .63<(x1[k]-x0[k]+1)/(y1[k]-y0[k]+1)<1.6}


def locate(left,right,prior,mount=MOUNT,camera_rotation=CAM_ROT,camera=PREVIEW_CAMERA,tags=TAGS):
    focal=camera.focal
    cx,cy=camera.principal
    ma,mb=markers(left),markers(right)
    ids=sorted(ma.keys() & mb.keys())
    if len(ids)<3:
        return None,len(ids),None
    targets=(np.array([cx,cy])-np.array([[ma[k] for k in ids],[mb[k] for k in ids]]))/focal
    points=tags[ids]

    def projections(q,which):
        rot=Rotation.from_euler("ZYX",q[3:]).as_matrix()
        cam=((points[which]-q[:3])@rot-mount)@camera_rotation
        pair=np.stack((cam-[0,BASELINE/2,0],cam+[0,BASELINE/2,0]))
        return pair[:,:,1:]/np.maximum(pair[:,:,:1],.35)-targets[:,which]

    use=np.max(np.abs(projections(prior,np.ones(len(ids),bool))),axis=(0,2))<.24
    def global_fix():
        if len(ids)<4:
            return None,len(ids),None
        intrinsic=np.array([[focal,0,cx],[0,focal,cy],[0,0,1]],dtype=np.float64)
        chosen=np.array([ma[k] for k in ids],dtype=np.float64)
        okay,rvec,tvec,inliers=cv2.solvePnPRansac(
            points,chosen,intrinsic,None,iterationsCount=75,
            reprojectionError=2.4*focal/FOCAL,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        if not okay or inliers is None or len(inliers)<4:
            return None,len(ids),None
        cam_to_world=cv2.Rodrigues(rvec)[0].T
        cv_axes=camera_rotation[:,[1,2,0]]*[-1,-1,1]
        body_to_world=cam_to_world@cv_axes.T
        left_origin=mount+camera_rotation[:,1]*BASELINE/2
        pos=-cam_to_world@tvec.ravel()-body_to_world@left_origin
        euler=Rotation.from_matrix(body_to_world).as_euler("ZYX")
        hint=np.r_[pos,euler]
        active=np.zeros(len(ids),dtype=bool)
        active[inliers.ravel()]=True
        q=solve(hint,active)
        quality=float(np.median(abs(projections(q,active))))
        if quality>.014 or np.linalg.norm(q[:3]-prior[:3])>1.25 or abs(wrap(q[3]-prior[3]))>.65:
            return None,len(ids),quality
        return q,int(np.count_nonzero(active)),quality

    def solve(init,active):
        def error(q):
            return np.r_[projections(q,active).ravel(), (q[:3]-prior[:3])*[.07,.07,.11],
                         wrap(q[3:]-prior[3:])*[.10,.04,.04]]
        return least_squares(error,init,max_nfev=14,loss="soft_l1",f_scale=.003).x
    if np.count_nonzero(use)<3:
        return global_fix()
    fit=solve(prior,use)
    worse=np.max(np.abs(projections(fit,use)),axis=(0,2))>.018
    if np.any(worse) and np.count_nonzero(use)-np.count_nonzero(worse)>=3:
        use[np.flatnonzero(use)[worse]]=False
        fit=solve(fit,use)
    quality=float(np.median(np.abs(projections(fit,use))))
    if quality>.010 or np.linalg.norm(fit[:3]-prior[:3])>.55:
        return global_fix()
    return fit,int(np.count_nonzero(use)),quality


def orange_components(image):
    height,width=image.shape[:2]
    scale=height/H
    r,g,b=np.moveaxis(image.astype(np.float32),-1,0)
    mask=(r>145)&(g>30)&(r>2.6*g)&(r<4.15*g)&(b<.41*g)
    n,index,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return index, [(i,centers[i],stats[i,2:4],stats[i,4]) for i in range(1,n)
                   if stats[i,4]>=90*scale**2 and stats[i,2]>8*scale and stats[i,3]>7*scale
                   and stats[i,0]>2 and stats[i,0]+stats[i,2]<width-3]


def rock_pixels(left,right,disparity):
    scale=left.shape[0]/H
    left_index, left_objects=orange_components(left)
    _, right_objects=orange_components(right)
    rgb_rock=np.zeros_like(disparity,dtype=bool)
    dm=left_index[::3,::3]
    for i,(lx,ly),(lw,lh),la in left_objects:
        matches=[]
        for _,(rx,ry),(rw,rh),ra in right_objects:
            d=lx-rx
            if abs(ly-ry)<2.9*scale and .68<la/ra<1.46 and 2.8*scale<d<46*scale and .28<lh*BASELINE/d<1.55:
                matches.append((abs(ly-ry)+abs(math.log(la/ra))*6,d))
        if matches:
            _,estimate=min(matches)
            rgb_rock |= (dm==i)&(abs(disparity-estimate)<max(.85*scale,.19*estimate))
    return rgb_rock


@dataclass
class Frame:
    pose: np.ndarray | None
    cloud: np.ndarray
    rock: np.ndarray
    left: np.ndarray
    right: np.ndarray
    markers: int
    error: float | None
    camera_yaw: float = 0.
    sim_time: float = 0.


class World:
    def __init__(self, kind="mixed", start=None, rocks=None, heights=None,config=None,
                 camera=PREVIEW_CAMERA):
        self.camera=camera
        self.kind=kind
        self.xs,self.ys=(FARM_XS,FARM_YS) if kind=="farm" else (XS,YS)
        self.start=tuple(start) if start is not None else (FARM_START if kind=="farm" else START)
        self.heights = heights if heights is not None else (farm_heights() if kind=="farm" else make_heights(kind))
        self.plants=vegetation(self.height_at) if kind=="farm" else []
        self.tags=FARM_TAGS if kind=="farm" else TAGS
        if rocks is None:
            rocks=scattered_rocks() if kind=="farm" else ROCKS
        xml = scene(self.heights,rocks,camera,xs=self.xs,ys=self.ys,tags=self.tags,plants=self.plants)
        self.model=mujoco.MjModel.from_xml_string(xml,{"earth.png":floor_texture()})
        self.data=mujoco.MjData(self.model)
        self.base=self.model.body("base").id
        self.gimbal=self.model.body("gimbal_yaw_link").id
        self.hull=self.model.geom("chassis").id
        self.ground=self.model.geom("ground").id
        self.rock_ids={self.model.geom(f"rock{i}").id for i in range(len(rocks))}
        width,height=camera.capture_size
        self.reader=mujoco.Renderer(self.model,height,width)
        num_disparities=16*math.ceil(48*camera.focal/FOCAL/16)
        self.stereo=cv2.StereoSGBM_create(minDisparity=0,numDisparities=num_disparities,blockSize=5,
                                         P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=11,
                                         speckleWindowSize=22,speckleRange=2,
                                         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.config=(config or VehicleConfig()).validated()
        self.mount=MOUNT.copy()
        self.camera_rotation=CAM_ROT.copy()
        self.camera_yaw=0.
        self.reset(self.start)
        self.rock_render=[]
        for gid in sorted(self.rock_ids):
            mesh=self.model.geom_dataid[gid]
            m=self.model
            v,f=m.mesh_vertadr[mesh],m.mesh_faceadr[mesh]
            vertices=m.mesh_vert[v:v+m.mesh_vertnum[mesh]]
            vertices=vertices@self.data.geom_xmat[gid].reshape((3,3)).T+self.data.geom_xpos[gid]
            self.rock_render.append({"vertices":vertices.tolist(),
                                     "faces":m.mesh_face[f:f+m.mesh_facenum[mesh]].tolist()})

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.reader.close()

    def height_at(self,x,y):
        return surface(self.heights,x,y,self.xs,self.ys)

    def aim(self,yaw):
        self.camera_yaw=float(wrap(yaw))
        self.camera_rotation=Rotation.from_euler("Z",self.camera_yaw).as_matrix()@CAM_ROT
        axes=self.camera_rotation[:,[1,2,0]]*[-1,1,-1]
        q=Rotation.from_matrix(axes).as_quat()
        for i,side in enumerate((1,-1)):
            self.model.cam_pos[i]=self.mount+side*BASELINE/2*self.camera_rotation[:,1]
            self.model.cam_quat[i]=q[[3,0,1,2]]
        mujoco.mj_camlight(self.model,self.data)

    def configure(self,config):
        self.config=config.validated()
        self.mount[2]=config.camera_height-BASE_Z
        # The upper gimbal cover's highest point defines the vehicle top.
        self.model.body_pos[self.gimbal,2]=config.height-BASE_Z-.02
        mujoco.mj_forward(self.model,self.data)
        self.aim(math.radians(config.camera_yaw))

    def reset(self,start):
        x,y,yaw=start
        c,s=math.cos(yaw),math.sin(yaw)
        slope_x=(self.height_at(x+.16,y)-self.height_at(x-.16,y))/.32
        slope_y=(self.height_at(x,y+.16)-self.height_at(x,y-.16))/.32
        up=np.array([-slope_x,-slope_y,1.])
        up/=np.linalg.norm(up)
        forward=np.array([c,s,c*slope_x+s*slope_y])
        forward/=np.linalg.norm(forward)
        rot=Rotation.from_matrix(np.column_stack((forward,np.cross(up,forward),up)))
        # Align the contact plane before the first solver step, so an uphill
        # spawn cannot begin with a wheel penetrating the heightfield.
        z=self.height_at(x,y)+BASE_Z/up[2]
        q=rot.as_quat()
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:7]=[x,y,z,q[3],q[0],q[1],q[2]]
        mujoco.mj_forward(self.model,self.data)
        self.prior=np.r_[x,y,z,rot.as_euler("ZYX")]
        self.collision=self.flipped=self.belly=False
        self.belly_steps=0
        self.configure(self.config)

    def bottom_clearance(self):
        p=self.data.qpos[:3]
        return p[2]-HULL-self.height_at(*p[:2])

    def observe(self,yaw=None):
        if yaw is not None:
            self.aim(yaw)
        camera=self.camera
        sim_time=float(self.data.time)
        images=self.capture()
        pose,n,error=locate(*images,self.prior,self.mount,self.camera_rotation,camera,self.tags)
        cloud=np.empty((0,3))
        rock=np.empty(0,dtype=bool)
        if pose is not None:
            self.prior=pose
            grey=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in images]
            disp=self.stereo.compute(*grey)[::3,::3].astype(np.float32)/16
            vv,uu=np.indices(disp.shape)*3
            colours=images[0][::3,::3]
            rgb_rock=rock_pixels(*images,disp)
            floor=(colours[:,:,1]>24)&(np.max(colours,axis=2).astype(int)-np.min(colours,axis=2)<57)
            focal=camera.focal
            cx,cy=camera.principal
            scale=focal/FOCAL
            valid=(disp>=2.0*scale)&(disp<43*scale)&(floor|rgb_rock)&(vv>camera.match_size[1]*.22)
            depth=focal*BASELINE/disp[valid]
            points=np.column_stack((depth,(cx-uu[valid])*depth/focal,
                                    (cy-vv[valid])*depth/focal))
            rot=Rotation.from_euler("ZYX",pose[3:]).as_matrix()
            origin=self.mount+self.camera_rotation[:,1]*BASELINE/2
            cloud=((points@self.camera_rotation.T+origin)@rot.T)+pose[:3]
            good=(abs(cloud[:,0])<self.xs)&(abs(cloud[:,1])<self.ys)&(cloud[:,2]>-1.5)&(cloud[:,2]<2.4)
            cloud=cloud[good]
            rock=rgb_rock[valid][good]
        return Frame(pose,cloud,rock,*images,n,error,self.camera_yaw,sim_time)

    def capture(self):
        """Render a synchronized RGB pair, independently of depth estimation."""
        camera=self.camera
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_camlight(self.model,self.data)
        images=[]
        for cam in ("left","right"):
            self.reader.update_scene(self.data,camera=cam)
            rgb=self.reader.render()
            width,height=camera.capture_size
            if rgb.shape!=(height,width,3):
                raise RuntimeError(f"虚拟相机输出规格异常：{rgb.shape}")
            images.append(cv2.resize(rgb,camera.match_size,interpolation=cv2.INTER_AREA)
                          if camera.capture_size!=camera.match_size else rgb)
        return images

    def drive(self,speed,turn,after_substep=None,*,recovering=False):
        d=self.data
        forward=d.xmat[self.base,[0,3]]
        forward=forward / max(np.linalg.norm(forward),.01)
        # Force and yaw torque model track traction; all vertical dynamics and
        # obstacle/ground contacts remain native MuJoCo.
        force=235*(speed*forward-d.qvel[:2])
        # Track holding/braking compensates the horizontal component needed
        # to hold on a contacted incline. This is inside the physics actuator.
        normal=d.xmat[self.base,[2,5,8]]
        ground_contact=any(self.ground in (int(c.geom1),int(c.geom2)) for c in d.contact)
        if ground_contact and normal[2]>.72:
            slope=-normal[:2]/normal[2]
            slope_limit=math.tan(math.radians(self.config.max_slope))
            slope*=min(1,slope_limit/max(np.linalg.norm(slope),1.e-6))
            force+=float(self.model.body_subtreemass[self.base])*9.81*slope
        length=np.linalg.norm(force)
        force *= min(1,210/max(length,1.e-6))
        # At a standstill the two tracks counter-rotate against static
        # friction. Keep the lower steering limit when the vehicle is moving.
        pivot=abs(speed)<.10 and abs(turn)>.24
        gain,limit=(50,48) if pivot else (25,18)
        torque=np.clip(gain*(turn-d.qvel[5]),-limit,limit)
        d.xfrc_applied[self.base,[0,1,5]]=(*force,torque)
        for _ in range(50):
            mujoco.mj_step(self.model,d)
            hull_ground=False
            for c in d.contact:
                pair=(int(c.geom1),int(c.geom2))
                if pair[0] in self.rock_ids or pair[1] in self.rock_ids:
                    self.collision=True
                hull_ground |= self.hull in pair and self.ground in pair
            self.belly_steps=(self.belly_steps+1) if hull_ground else 0
            if recovering:
                self.belly=False
            self.belly |= self.belly_steps>=160
            self.flipped |= bool(d.xmat[self.base,8]<math.cos(1.2))
            if self.collision or self.flipped or (self.belly and not recovering):
                break
            if after_substep is not None:
                after_substep(float(d.time))
