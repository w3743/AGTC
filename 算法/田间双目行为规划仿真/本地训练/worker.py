"""Fine-tune the AGTC terrain, local-neuron and temporal models."""
import argparse
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from planner import load_model
from research import load_research, samples, sequences
from train_terrain import camera_report, generate, physics_samples

ORDER=("NPU","GPU","CPU")


def choose_backend(probe, report=lambda _:None, order=ORDER):
    """Select only a device on which the full training probe completed."""
    for kind in order:
        try:
            backend=probe(kind)
        except (ImportError,RuntimeError,ValueError,NotImplementedError,OSError) as exc:
            report({"kind":kind,"ok":False,"detail":str(exc)})
        else:
            report({"kind":kind,"ok":True,**backend})
            return backend
    raise RuntimeError("没有通过训练探针的设备")


def emit(event,**fields):
    print("@AGTC "+json.dumps({"event":event,**fields},ensure_ascii=False,allow_nan=False),flush=True)


def verify_gradients(device):
    base=load_model().to(device)
    research=load_research().to(device)
    features=torch.full((4,5),.4,device=device)
    local=torch.full((4,49),.6,device=device)
    opt=torch.optim.SGD([*base.parameters(),*research.parameters()],lr=.05)
    weight=next(base.parameters())
    before=weight.detach().clone()
    pred_a=base(features)
    pred_b=research.local(local)
    pred_c,_=research.temporal(features,research.temporal.initial(len(features),features))
    loss=(pred_a.square().mean()+pred_b.square().mean()+pred_c.square().mean()+pred_a.mean())
    if pred_a.device.type!=features.device.type or pred_b.device.type!=features.device.type or pred_c.device.type!=features.device.type:
        raise RuntimeError("模型运算没有保留在请求设备")
    loss.backward()
    if weight.grad is None or not bool(torch.isfinite(weight.grad).all().item()):
        raise RuntimeError("反向传播没有返回有限梯度")
    opt.step()
    if torch.equal(weight.detach(),before):
        raise RuntimeError("优化器未更新参数")
    if torch.device(device).type=="cuda":
        torch.cuda.synchronize()


def probe_device(kind):
    if kind=="NPU":
        npu_name=os.environ.get("AGTC_NPU_NAME","未报告名称的 NPU")
        # torch_npu is a native training backend when paired with its supported
        # hardware. An installed inference plugin is not an autograd backend.
        if not hasattr(torch,"npu") and importlib.util.find_spec("torch_npu") is not None:
            import torch_npu
        if not hasattr(torch,"npu") or not torch.npu.is_available():
            raise RuntimeError(f"{npu_name}：当前 Python 环境没有可用的 NPU 反向传播后端")
        device,name="npu:0",torch.npu.get_device_name(0)
    elif kind=="GPU":
        if torch.cuda.is_available():
            device,name="cuda:0",torch.cuda.get_device_name(0)
        elif hasattr(torch,"xpu") and torch.xpu.is_available():
            device,name="xpu:0",torch.xpu.get_device_name(0)
        else:
            raise RuntimeError("当前 PyTorch 未发现可用于训练的 CUDA/XPU 设备")
    else:
        device,name="cpu","本机 CPU"
    verify_gradients(device)
    info={"kind":kind,"device":device,"name":name,"detail":"三个模型均通过前向、反向与参数更新探针"}
    if device.startswith("cuda"):
        info["memory_mib"]=round(torch.cuda.get_device_properties(0).total_memory/(1024**2))
    return info


def terrain_loss(model,x,p,r):
    out=model(x)
    loss=F.binary_cross_entropy_with_logits(out[:,0],p)+.28*F.smooth_l1_loss(out[:,1],r)
    return loss,(out[:,0]>0)==(p>.5),abs(out[:,1]-r).mean()


def local_loss(model,x,y):
    out=model(x)
    return F.binary_cross_entropy_with_logits(out,y),(out>0)==(y>.5),None


def memory_loss(model,x,y):
    state=model.initial(len(x),x)
    output=[]
    for time_index in range(x.shape[1]):
        value,state=model(x[:,time_index],state)
        output.append(value)
    out=torch.stack(output,dim=1)
    return F.binary_cross_entropy_with_logits(out,y),(out>0)==(y>.5),None


def make_data(seed,physical=True):
    rng=np.random.default_rng(seed)
    emit("stage",phase="生成数据",message="坡道仿真按每眼 3840×2160、30 Hz 采集双目图")
    train,val=generate(rng,10240),generate(rng,2048)
    camera=camera_report()
    if physical:
        collected,camera=physics_samples()
        train=tuple(np.concatenate((a,b)) for a,b in zip(train,collected))
    emit("data",terrain=len(train[0]),terrain_validation=len(val[0]),
         physical_stereo=len(train[0])-10240,local=4608,local_validation=1024,
         memory=2400,memory_validation=500,camera=camera)
    datasets=((tuple(torch.from_numpy(t) for t in train),tuple(torch.from_numpy(t) for t in val)),
              (samples(rng,4608),samples(rng,1024)),(sequences(rng,2400),sequences(rng,500)))
    return datasets,camera


def metrics(model,val,objective):
    model.eval()
    with torch.inference_mode():
        loss,match,mae=objective(model,*val)
        return {"val_loss":float(loss.item()),"label_match":float(match.float().mean().item()),
                **({"reward_mae":float(mae.item())} if mae is not None else {})}


def fit(label,model,train,val,objective,epochs,backend,lr):
    device=backend["device"]
    model.to(device)
    train=tuple(t.to(device) for t in train)
    val=tuple(t.to(device) for t in val)
    count=len(train[0])
    best=metrics(model,val,objective)
    best_epoch=0
    best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    emit("epoch",model=label,epoch=0,epochs=epochs,device=device,**best)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    for epoch in range(1,epochs+1):
        started=time.perf_counter()
        model.train()
        losses=[]
        for index in torch.randperm(count,device=device).split(256):
            batch=tuple(t[index] for t in train)
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=objective(model,*batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        record=metrics(model,val,objective)
        train_loss=float(torch.stack(losses).mean().item())
        if not np.isfinite([train_loss,*record.values()]).all():
            raise RuntimeError(f"{label}第 {epoch} 轮出现非有限值")
        if record["val_loss"]<best["val_loss"]:
            best=record
            best_epoch=epoch
            best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        emit("epoch",model=label,epoch=epoch,epochs=epochs,device=device,
             train_loss=train_loss,best_epoch=best_epoch,
             seconds=time.perf_counter()-started,**record)
    model.load_state_dict(best_state)
    model.cpu().eval()
    return {"selected_epoch":best_epoch,**best}


def checkpoint(name,model,report):
    out=io.BytesIO()
    torch.save({"weights":model.state_dict(),"metrics":report},out)
    emit("checkpoint",name=name,bytes_b64=base64.b64encode(out.getvalue()).decode("ascii"))


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    def report(row):
        emit("probe",**row)
    backend=choose_backend(probe_device,report)
    emit("backend",**backend)
    datasets,camera=make_data(args.seed,physical=not args.skip_physics)
    while True:
        try:
            terrain=load_model()
            research=load_research()
            configs=(("地形网络",terrain,terrain_loss,args.terrain,.0012),
                     ("局部神经元",research.local,local_loss,args.local,.0015),
                     ("RNN 记忆",research.temporal,memory_loss,args.memory,.0025))
            selected={}
            for (label,model,objective,epochs,lr),(tr,va) in zip(configs,datasets):
                emit("stage",phase=label,message=f"{backend['name']} 正在续训 {label}")
                selected[label]=fit(label,model,tr,va,objective,epochs,backend,lr)
            break
        except (RuntimeError,NotImplementedError) as exc:
            remaining=ORDER[ORDER.index(backend["kind"])+1:]
            if not remaining:
                raise
            emit("fallback",device=backend["device"],detail=str(exc))
            backend=choose_backend(probe_device,report,remaining)
            emit("backend",**backend)
    final={"device":backend,"seed":args.seed,"epochs":args.terrain+args.local+args.memory,"camera":camera,
           "selected":selected,"validation_scope":"独立特征与序列验证集，未包含闭环到达率"}
    checkpoint("terrain",terrain,final)
    checkpoint("research",research,final)
    emit("complete",report=final)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key,default in (("terrain",12),("local",8),("memory",8),("threads",4),("seed",112121)):
        parser.add_argument("--"+key,type=int,default=default)
    parser.add_argument("--skip-physics",action="store_true",help="仅用于快速工作台测试")
    args=parser.parse_args()
    run(args)
