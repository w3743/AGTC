"""Local interactive 3D terrain simulator."""
import argparse
import base64
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import numpy as np
import torch

from terrain import BASE_Z, CLEARANCE, DT, GOAL, Frame, VehicleConfig, World
import cv2
from farm import FARM_GOAL, ROW_SPACING
from planner import ACTIVE, CELL, MANUAL, Runner, load_model
from research import load_research

HERE=Path(__file__).resolve().parent
STATE=None
SCENE=b"{}"
REQUESTS=Queue(maxsize=16)
ASSETS={"/":("terrain.html","text/html"),"/think":("terrain.html","text/html"),
        "/view.js":("view.js","application/javascript"),"/field_view.js":("field_view.js","application/javascript"),
        "/road_view.js":("road_view.js","application/javascript"),
        "/terrain.css":("terrain.css","text/css")}


class Handler(BaseHTTPRequestHandler):
    def respond(self, body, mime, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
            return

    def log_message(self,*_):
        pass
    def do_GET(self):
        path,_,query=self.path.partition("?")
        published=STATE
        if path=="/api/scene":
            self.respond(SCENE,"application/json; charset=utf-8")
        elif path in ("/api/live","/api/map","/api/state") and published is not None:
            data=published.live if path=="/api/live" else published.vision.grid if path=="/api/map" else published.full()
            self.respond(data,"application/json; charset=utf-8")
        elif path=="/api/memory" and published is not None:
            self.respond(published.memory,"application/json; charset=utf-8")
        elif path in ("/api/camera/left.jpg","/api/camera/right.jpg") and published is not None:
            images=next((pair for pair in published.photos if not query or query==f"v={pair.revision}"),None)
            if images is None:
                self.respond(b"","text/plain",404)
            else:
                self.respond(images.left if path.endswith("left.jpg") else images.right,"image/jpeg")
        elif path in ASSETS:
            name,kind=ASSETS[path]
            self.respond((HERE/name).read_bytes(),kind+"; charset=utf-8")
        else:
            self.respond(b"","text/plain",404)

    def do_POST(self):
        if self.path not in ("/api/config","/api/memory/start","/api/memory/finish","/api/memory/origin",
                             "/api/memory/goal","/api/memory/restore","/api/manual",
                             "/api/goal","/api/stop","/api/reset"):
            self.respond(b"{}", "application/json", 404)
            return
        try:
            length=int(self.headers.get("Content-Length","0"))
            size_limit=2_000_000 if self.path=="/api/memory/restore" else 512
            if not 0<=length<=size_limit:
                raise ValueError("参数数据无效")
            v=json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(v,dict):
                raise ValueError("参数数据无效")
            if self.path=="/api/config":
                args=VehicleConfig(*(float(v[key]) for key in ("height","camera_height","max_slope","camera_yaw")))
                args.validated()
            elif self.path=="/api/memory/start":
                args=(v.get("mode","manual"),v.get("number"))
            elif self.path=="/api/memory/restore":
                args=v
            elif self.path in ("/api/memory/finish","/api/stop","/api/reset"):
                args=None
            elif self.path=="/api/manual":
                args=(float(v["throttle"]),float(v["steering"]))
            else:
                args=(float(v["x"]),float(v["y"]))
            output=Queue(maxsize=1)
            REQUESTS.put((self.path,args,output),timeout=.5)
            okay,message=output.get(timeout=3)
        except (ValueError,KeyError,TypeError,Full,Empty) as exc:
            okay,message=False,str(exc) or "参数无效或仿真未响应"
        self.respond(encode({"ok":okay,"message":message}),"application/json; charset=utf-8",200 if okay else 400)


def encode(v):
    return json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()


@dataclass(frozen=True)
class VisionPublication:
    frame: Frame
    revision: int
    grid: bytes
    observed: int
    hazards: int


@dataclass(frozen=True)
class StereoPreview:
    revision: int
    left: bytes
    right: bytes


@dataclass(frozen=True)
class Publication:
    live: bytes
    vision: VisionPublication
    photos: tuple[StereoPreview,...]
    memory_revision: int
    memory: bytes

    def full(self):
        payload=json.loads(self.live)
        payload.update(json.loads(self.vision.grid))
        payload["left"]="data:image/jpeg;base64,"+base64.b64encode(self.photos[0].left).decode()
        payload["right"]="data:image/jpeg;base64,"+base64.b64encode(self.photos[0].right).decode()
        return encode(payload)


def jpeg(image):
    preview=cv2.resize(image,(384,216),interpolation=cv2.INTER_AREA)
    bgr=cv2.cvtColor(preview,cv2.COLOR_RGB2BGR)
    return cv2.imencode(".jpg",bgr,[cv2.IMWRITE_JPEG_QUALITY,70])[1].tobytes()


def publish_vision(r,revision):
    known=r.map.hit>0
    xys=r.map.mapped(np.argwhere(known)[:,::-1])
    grid=np.round(np.column_stack((xys,r.map.z[known],r.map.risk[known])),3).tolist()
    hazards=r.map.blocked & ~r.memory.layout.ridges
    if r.map.scope is not None:
        hazards &= r.map.scope
    blocked=r.map.mapped(np.argwhere(hazards)[:,::-1])
    blocked=blocked[(abs(blocked[:,0])<r.world.xs-.2)&(abs(blocked[:,1])<r.world.ys-.2)]
    data=encode({"revision":revision,"map":grid,"blocked":np.round(blocked,3).tolist()})
    return VisionPublication(r.frame,revision,data,len(grid),len(blocked))


def publish(r,previous=None,preview=None):
    vision=previous.vision if previous else None
    if vision is None or vision.frame is not r.frame:
        vision=publish_vision(r,(vision.revision+1) if vision else 1)
        preview=(r.frame.left,r.frame.right)
    photos=previous.photos if previous else ()
    if preview is not None:
        pair=StereoPreview(photos[0].revision+1 if photos else 1,*(jpeg(image) for image in preview))
        photos=(pair,*photos[:3])
    revision=r.memory.revision
    memory=previous.memory if previous is not None and previous.memory_revision==revision else encode(r.memory.payload())
    w=r.world
    q=w.data.qpos[:7].tolist()
    live=encode({
        "phase":r.phase,"reason":r.reason,"seconds":r.ticks*DT,"distance":r.distance,
        "sim_time":float(w.data.time),"vision_revision":vision.revision,"preview_revision":photos[0].revision,
        "memory_revision":revision,"recording":r.memory.recording,
        "mapping":None if r.explorer is None else {"number":r.explorer.number,"stage":r.explorer.stage,
                                                    "complete":len(r.memory.swept)},
        "goal":r.goal,"vehicle":q,"visual":None if r.frame.pose is None else np.round(r.visual,4).tolist(),
        "route":r.route,"trace":r.trace,"reward":r.reward,"look":r.look,
        "observed_cells":vision.observed,"hazard_cells":vision.hazards,"cloud":len(r.frame.cloud),
        "markers":r.frame.markers,"vision_error":r.frame.error,
        "camera_yaw":float(np.degrees(w.camera_yaw)%360),"config":asdict(w.config),
        "can_configure":bool(r.can_configure()),"recoveries":r.recoveries,
        "scan_attempts":r.vision_misses,"target_revision":r.target_revision,
        "stereo_size":w.camera.match_size,"backtrack_count":r.recovery_count,"alert":r.alert,
        "research":r.research.values if r.research is not None else None,
        "passed":bool(r.crossed),"detour":bool(r.detoured),"stuck":bool(r.stuck),
        "collision":bool(w.collision),"flipped":bool(w.flipped),"latency":r.latency,"vision_ms":r.vision_ms,
        "speed":float(np.linalg.norm(w.data.qvel[:2])),"cmd":r.cmd,
    })
    return Publication(live,vision,photos,revision,memory)


def snapshot(r):
    return publish(r).full()


def main(port=8922,terrain_checkpoint=None,research_checkpoint=None,kind="farm",memory_reference=None):
    global STATE,SCENE
    torch.set_num_threads(1)
    net=load_model(terrain_checkpoint) if terrain_checkpoint else load_model()
    research=load_research(research_checkpoint) if research_checkpoint else load_research()
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    with World(kind=kind) as w,server:
        runner=Runner(w,net,research)
        if memory_reference is not None:
            runner.memory.restore_reference(json.loads(memory_reference.read_text(encoding="utf-8-sig")))
        stride=2 if kind=="farm" else 4
        SCENE=encode({"heights":w.heights[::stride,::stride].round(3).tolist(),"x":w.xs,"y":w.ys,"cell":CELL,
                      "memory_schema":2,"aisles":list(runner.memory.layout.aisles),
                      "kind":kind,"row_spacing":ROW_SPACING if kind=="farm" else None,
                      "goal":FARM_GOAL if kind=="farm" else GOAL,"clearance":CLEARANCE,"carZ":BASE_Z,
                      "plants":np.round(w.plants,3).tolist(),"rocks":w.rock_render})
        STATE=publish(runner)
        last_preview=time.perf_counter()
        threading.Thread(target=server.serve_forever,daemon=True).start()
        print(f"http://127.0.0.1:{port}/  http://127.0.0.1:{port}/think",flush=True)
        while True:
            tick=time.perf_counter()
            touched=False
            try:
                verb,goal,result=REQUESTS.get_nowait()
            except Empty:
                pass
            else:
                if verb=="/api/goal":
                    reply=runner.goal_at(goal)
                elif verb=="/api/config":
                    reply=runner.configure(goal)
                elif verb=="/api/memory/start":
                    reply=runner.start_memory(*goal)
                elif verb=="/api/memory/finish":
                    reply=runner.stop() if runner.memory.recording is not None else (False,"当前没有正在建图")
                elif verb=="/api/memory/origin":
                    reply=runner.set_origin(goal)
                elif verb=="/api/memory/restore":
                    if not runner.can_configure() or runner.memory.revision:
                        reply=False,"只能恢复到空闲的空白地图"
                    else:
                        try:
                            runner.memory.restore_reference(goal)
                            runner.map.refresh_constraints()
                            reply=True,runner.memory.result
                        except (ValueError,KeyError,TypeError) as exc:
                            reply=False,str(exc)
                elif verb=="/api/memory/goal":
                    reply=runner.goal_at(goal)
                elif verb=="/api/manual":
                    reply=runner.manual_input(*goal)
                elif verb=="/api/reset":
                    if runner.memory.recording is not None:
                        runner.stop()
                    w.reset(w.start)
                    runner=Runner(w,net,research,memory=runner.memory)
                    reply=True,runner.phase
                else:
                    reply=runner.stop()
                touched=True
                STATE=publish(runner,STATE)
                result.put(reply)
            if runner.phase in ACTIVE:
                runner.step()
                if runner.alert:
                    print("ALERT "+json.dumps(runner.alert,ensure_ascii=False),flush=True)
                touched=True
            elif runner.phase==MANUAL:
                runner.step_manual()
                touched=True
            elif runner.phase=="停车中":
                w.drive(0.,0.)
                runner.cmd=(0.,0.)
                if np.linalg.norm(w.data.qvel[:2])<.07:
                    runner.measure()
                    runner.phase="已停车"
                touched=True
            if touched:
                preview=None
                if STATE.vision.frame is not runner.frame:
                    last_preview=tick
                elif runner.phase in (*ACTIVE,MANUAL) and tick-last_preview>=.065:
                    preview=w.capture()
                    last_preview=tick
                STATE=publish(runner,STATE,preview)
            time.sleep(max(0.,DT-(time.perf_counter()-tick)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8922)
    parser.add_argument("--terrain-checkpoint",type=Path)
    parser.add_argument("--research-checkpoint",type=Path)
    parser.add_argument("--scene",choices=("farm","mixed"),default="farm")
    parser.add_argument("--memory-reference",type=Path)
    args=parser.parse_args()
    main(args.port,args.terrain_checkpoint,args.research_checkpoint,args.scene,args.memory_reference)
