"""Adapt AGTC's local-neuron graph and all-input RNN memory to terrain."""
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from local_neuron import LocalNeuronClassifier

ROOT=Path(__file__).resolve().parent
CHECKPOINT=ROOT/"局部神经元与地形记忆.pt"


class LocalRelief(LocalNeuronClassifier):
    """Use the source graph/edge/cell rules with one terrain sample per port."""
    def __init__(self):
        super().__init__(seed=1121,n_hidden=12,input_degree=12,recurrent_degree=3,
                         steps=3,memory_dim=3,edge_hidden=8,cell_hidden=12)
        self.terrain_readout=nn.Linear(self.n_classes,1)

    def _sensor_values(self,patches):
        if patches.ndim!=2 or patches.shape[1]!=49:
            raise ValueError("expected 7×7 terrain ports")
        return patches

    def forward(self,patches):
        return self.terrain_readout(super().forward(patches)).squeeze(-1)


class SourceMemory(nn.Module):
    """All-input-source retention over visual cues and prior hidden activity."""
    def __init__(self):
        super().__init__()
        self.retention_logit=nn.Parameter(torch.full((11,),1.38))
        self.current=nn.Linear(5,6)
        self.recur=nn.Linear(6,6,bias=False)
        self.memory=nn.Linear(11,6,bias=False)
        self.readout=nn.Linear(6,1)

    def initial(self,batch,tensor):
        return tensor.new_zeros((batch,6)),tensor.new_zeros((batch,11))

    def forward(self,x,state):
        h,mem=state
        rho=self.retention_logit.sigmoid()
        next_h=torch.tanh(self.current(x)+self.recur(h)+self.memory(mem))
        next_mem=rho*mem+(1-rho)*torch.cat((x,h),dim=-1)
        return self.readout(next_h).squeeze(-1),(next_h,next_mem)


class Research(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=LocalRelief()
        self.temporal=SourceMemory()


def load_research(path=CHECKPOINT):
    net=Research()
    net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["weights"])
    return net.eval()


class ResearchState:
    def __init__(self,network):
        self.net=network
        self.state=network.temporal.initial(1,next(network.parameters()))
        self.values={}

    @torch.inference_mode()
    def assess(self,patch,risk,grade,known,fresh):
        local=float(self.net.local(torch.from_numpy(patch.reshape(1,49))).sigmoid()[0])
        inputs=torch.tensor([[risk,local,min(2.,grade*2),float(known),float(fresh)]],dtype=torch.float32)
        out,self.state=self.net.temporal(inputs,self.state)
        retained=float(self.net.temporal.retention_logit.sigmoid().mean())
        self.values={"local_risk":local,"memory_risk":float(out.sigmoid()[0]),
                     "retention":retained,"ports":49}
        return self.values


def samples(rng,n):
    yy,xx=np.mgrid[-3:4,-3:4]
    center=rng.uniform(-2.8,2.8,(n,1,2))
    dist=(np.column_stack((xx.ravel(),yy.ravel()))-center)**2
    strength=rng.uniform(.07,1.,(n,1))
    cloud=strength*np.exp(-dist.sum(axis=-1)*rng.uniform(.10,.60,(n,1)))
    clean=np.clip(cloud+rng.uniform(0,.13,(n,49)),0,1)
    corridor=(np.abs(xx.ravel())<2)&(yy.ravel()<2)&(yy.ravel()>-3)
    label=np.max(clean[:,corridor],axis=1)>.56
    known=rng.random((n,49))>.11
    inputs=np.where(known,clean,.34)
    return torch.from_numpy(inputs.astype("float32")),torch.from_numpy(label.astype("float32"))


def sequences(rng,n,ticks=12):
    hazard=(rng.random((n,1))>.53)
    hazard=np.broadcast_to(hazard,(n,ticks)).copy()
    hazard[:,6:]=rng.random((n,1))>.53
    cues=np.clip(hazard[:,:,None]+rng.normal(0,.26,(n,ticks,3)),0,1)
    fresh=rng.random((n,ticks))>.23
    cues=np.where(fresh[:,:,None],cues,.37)
    return (torch.from_numpy(np.concatenate((cues,fresh[:,:,None],fresh[:,:,None]),axis=-1).astype("float32")),
            torch.from_numpy(hazard.astype("float32")))


def train():
    torch.set_num_threads(1)
    torch.manual_seed(1121)
    rng=np.random.default_rng(1121)
    model=Research()
    train_x,train_y=samples(rng,4608)
    valid_x,valid_y=samples(rng,1024)
    sequence_x,sequence_y=sequences(rng,2400)
    seq_x,seq_y=sequences(rng,500)
    opt=torch.optim.AdamW(model.local.parameters(),lr=.004)
    for _ in range(32):
        for idx in torch.randperm(len(train_x)).split(128):
            loss=F.binary_cross_entropy_with_logits(model.local(train_x[idx]),train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    def run(seq):
        state=model.temporal.initial(len(seq),seq)
        outputs=[]
        for t in range(seq.shape[1]):
            pred,state=model.temporal(seq[:,t],state)
            outputs.append(pred)
        return torch.stack(outputs,dim=1)
    opt=torch.optim.AdamW(model.temporal.parameters(),lr=.006)
    for _ in range(30):
        for idx in torch.randperm(len(sequence_x)).split(128):
            loss=F.binary_cross_entropy_with_logits(run(sequence_x[idx]),sequence_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.inference_mode():
        metrics={"local_epochs":32,"local_patches":len(train_x),"local_validation_match":
                 float(((model.local(valid_x)>0)==(valid_y>.5)).float().mean()),
                 "rnn_epochs":30,"rnn_sequences":len(sequence_x),"rnn_validation_match":
                 float(((run(seq_x)>0)==(seq_y>.5)).float().mean()),
                 "scope":"synthetic local terrain patches and visual source dropout, no MNIST digit weights"}
    out=io.BytesIO()
    torch.save({"weights":model.state_dict(),"metrics":metrics},out)
    print("METRICS "+json.dumps(metrics),flush=True)
    print("CHECKPOINT "+base64.b64encode(out.getvalue()).decode(),flush=True)


if __name__=="__main__":
    train()
