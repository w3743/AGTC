"""Fit the terrain risk/reward network and evaluate independent physical routes."""
import argparse
import base64
import io
import json
import math
import sys

import numpy as np
import torch
from torch.nn import functional as F

from terrain import DT, GOAL, ROCKS, START, StereoClock, TRAINING_CAMERA, World, X, Y, terrain_reward
from planner import ACTIVE, DONE, Runner, TerrainNet, VisualMap, load_model


def reference(features):
    slope,rise,rough,support,rock=features.T
    passes=(slope<.41)&(rise<.48)&(rough<.17)&(rock<.5)
    expected=np.where(passes,1.5*slope+support*.18-.65*rise-.65*rough,-2.5)
    return passes.astype(np.float32),expected.astype(np.float32)


def generate(rng,count):
    grades=rng.uniform(0,.9,count)
    rises=grades*rng.uniform(.45,1.45,count)
    rough=rng.beta(.8,4,count)*.35
    support=rng.uniform(.25,1.,count)
    rocks=(rng.random(count)<.13)
    features=np.column_stack((grades,rises,rough,support,rocks)).astype(np.float32)
    return features,*reference(features)


def camera_report(pairs=0,localized=0,intervals=()):
    return {"capture_size":TRAINING_CAMERA.capture_size,"match_size":TRAINING_CAMERA.match_size,
            "fps_per_eye":TRAINING_CAMERA.fps,"clock":"simulation","stereo_pairs":pairs,
            "localized_pairs":localized,"interval_count":len(intervals),
            "interval_min_ms":round(min(intervals),3) if len(intervals) else None,
            "interval_max_ms":round(max(intervals),3) if len(intervals) else None}


def physics_samples():
    features, distances, results=[],[],[]
    pairs,localized=0,0
    intervals=[]
    dummy=TerrainNet()
    for i,(peak,width) in enumerate(zip(np.linspace(.22,1.,12),np.linspace(1.24,.28,12))):
        z=peak*np.exp(-(X/width)**2)*np.exp(-(Y/2.4)**4)
        # Preserve metric height while MuJoCo normalizes every hfield asset.
        z[0,0],z[0,1]=-1,1
        with World(heights=z,rocks=[],start=(-2.4,0.,0.),camera=TRAINING_CAMERA) as w:
            begin=w.data.qpos[:2].copy()
            visual=VisualMap()
            valid=[]
            clock=StereoClock(TRAINING_CAMERA.fps)
            previous_id,previous_time=None,None
            profile_pairs=0

            def capture(seconds):
                nonlocal pairs,localized,profile_pairs,previous_id,previous_time
                frame_id=clock.poll(seconds)
                if frame_id is None or frame_id>=60 or frame_id%12>=6 or w.data.qpos[0]>=-.48:
                    return
                frame=w.observe()
                pairs+=1
                profile_pairs+=1
                if previous_id is not None and frame_id==previous_id+1:
                    intervals.append((frame.sim_time-previous_time)*1000)
                previous_id,previous_time=frame_id,frame.sim_time
                if frame.pose is None:
                    return
                localized+=1
                visual.update(frame)
                # Keep two feature snapshots per burst to limit the weight of
                # near-identical frames.
                if frame_id%12 not in (2,5):
                    return
                visual.assess(dummy)
                xs=np.arange(24,33)
                xs=xs[visual.hit[25,xs]>1]
                if len(xs):
                    valid.extend(visual.features[25,xs].copy())

            for tick in range(180):
                w.drive(.60,0.,after_substep=capture)
                if w.belly or w.flipped or w.collision or w.data.qpos[0]>1.1:
                    break
            metres=float(np.linalg.norm(w.data.qpos[:2]-begin))
            success=bool(w.data.qpos[0]>1.1 and not (w.belly or w.flipped or w.collision))
            score=terrain_reward(metres,float(w.data.time),success,w.collision,w.flipped,w.belly,success,False)
            expected_pass=success and (peak / width)<.57
            for feat in valid:
                if feat[0]>.17 or not expected_pass:
                    features.append(feat)
                    distances.append(float(expected_pass))
                    results.append(float(np.clip(score*.15,-3.,2.)))
            print(f"PROFILE {i+1}/12 4K stereo pairs={profile_pairs} pass={success} belly={w.belly} flip={w.flipped} samples={len(valid)}",flush=True)
    if not len(features) or not intervals:
        raise RuntimeError("没有通过验证的 4K 双目坡道训练样本")
    dataset=(np.asarray(features,dtype=np.float32).reshape(-1,5),np.asarray(distances,np.float32),
             np.asarray(results,np.float32))
    return dataset,camera_report(pairs,localized,intervals)


def train():
    rng=np.random.default_rng(6917)
    train_data=generate(rng,10240)
    validation=generate(rng,2048)
    physical,camera=physics_samples()
    x,p,r=[torch.from_numpy(np.concatenate((a,b))) for a,b in zip(train_data,physical)]
    xv,pv,rv=[torch.from_numpy(k) for k in validation]
    net=TerrainNet()
    opt=torch.optim.AdamW(net.parameters(),lr=.0025)
    for _ in range(48):
        for i in torch.randperm(len(x)).split(256):
            y=net(x[i])
            loss=F.binary_cross_entropy_with_logits(y[:,0],p[i])+.28*F.smooth_l1_loss(y[:,1],r[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.inference_mode():
        y=net(xv)
        metrics={"epochs":48,"geometry_training":len(train_data[0]),"physics_stereo_samples":len(physical[0]),
                 "physics_profiles":12,"geometry_validation":len(xv),"camera":camera,
                 "validation_safety_label_match":float(((y[:,0]>0)==(pv>0)).float().mean()),
                 "validation_reward_mae":float(torch.mean(abs(y[:,1]-rv)))}
    out=io.BytesIO()
    torch.save({"metrics":metrics,"weights":net.state_dict()},out)
    print("METRICS "+json.dumps(metrics),flush=True)
    print("CHECKPOINT "+base64.b64encode(out.getvalue()).decode(),flush=True)


def evaluate():
    net=load_model()
    rows=[]
    for name,kind,start,goal,rocks in [
        ("起伏直穿","cross",START,(-1.6,0.),ROCKS),
        ("跨坡与绕行","mixed",START,GOAL,ROCKS),
        ("斜向绕岩","mixed",(-7.,-2.8,.27),(6.6,2.4),ROCKS)
    ]:
        with World(kind,start,rocks) as w:
            player=Runner(w,net)
            okay,why=player.goal_at(goal)
            latency=[]
            for step in range(2000 if okay else 0):
                player.step()
                latency.append(player.latency)
                if step%140==0:
                    print(f"STEP {name} {step} {player.phase} {np.round(w.data.qpos[:3],2)} {player.cmd} {player.reason}",file=sys.stderr,flush=True)
                if player.phase not in ACTIVE:
                    break
            goal_error=float(np.linalg.norm(w.data.qpos[:2]-goal))
            row={"scene":name,"state":player.phase,"success":player.phase==DONE and goal_error<.49,
                 "reason":player.reason if okay else why,"goal_error_m":goal_error,"seconds":player.ticks*DT,
                 "path_m":player.distance,"climbed":player.crossed,"detoured":player.detoured,
                 "vertical_range_m":player.relief,"collision":w.collision,"flipped":w.flipped,
                 "stuck":player.stuck,"reward":player.reward,
                 "p95_ms":float(np.quantile(latency,.95)) if latency else None}
            rows.append(row)
            print("CASE "+json.dumps(row,ensure_ascii=False),flush=True)
    print("EVAL "+json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evaluate",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(6917)
    evaluate() if args.evaluate else train()
