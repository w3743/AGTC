"""Standard and topology controls under the same token interface."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from adaptive_memory_graph import ReadHop
from slot_memory_graph import SlotMemoryGraph, VOCAB


NAMES=('adaptive','GRU','LSTM','TCN','Transformer','event_dense','slot_dense','slot_fixed')
STANDARDS=NAMES[1:6]


class SequenceControl(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.name=name
        width={'GRU':164,'LSTM':140,'TCN':64,'Transformer':64,'event_dense':72}[name]
        self.token=nn.Embedding(VOCAB,32 if name in ('GRU','LSTM') else width)
        if name in ('GRU','LSTM'):
            self.core=getattr(nn,name)(32,width,batch_first=True)
        elif name=='TCN':
            self.core=nn.ModuleList(nn.Conv1d(width,width,2,dilation=2**i) for i in range(12))
        elif name=='Transformer':
            self.core=nn.ModuleList(nn.TransformerEncoderLayer(width,4,2*width,dropout=0,batch_first=True) for _ in range(3))
            angle=torch.arange(4096)[:,None]*torch.exp(-math.log(10000)*torch.arange(0,width,2)/width)
            self.register_buffer('positions',torch.stack((angle.sin(),angle.cos()),-1).flatten(1),persistent=False)
        else:
            self.context=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.LayerNorm(width))
            self.core=nn.ModuleList(ReadHop(width) for _ in range(2))
        self.head=nn.Linear(width,4)

    def forward(self,x):
        h=self.token(x)
        if self.name in ('GRU','LSTM'):
            h=self.core(h)[0][:,-1]
        elif self.name=='TCN':
            h=h.transpose(1,2)
            for layer in self.core:
                h=h+layer(F.pad(h,(layer.dilation[0],0))).relu()
            h=h[:,:,-1]
        elif self.name=='Transformer':
            h=h+self.positions[:h.shape[1]]
            for layer in self.core:
                h=layer(h)
            h=h[:,-1]
        else:
            prefix=h[:,:-1]
            nodes=self.context(torch.cat((F.pad(prefix[:,:-1],(0,0,1,0)),prefix),-1))
            h=h[:,-1]
            for layer in self.core:
                h,_=layer(h,nodes,'dense')
        return self.head(h)


def build(name):
    slots={'adaptive':'adaptive','slot_dense':'dense','slot_fixed':'fixed'}
    model=SlotMemoryGraph(slots[name]) if name in slots else SequenceControl(name)
    count=sum(p.numel() for p in model.parameters())
    assert 90_000 <= count <= 110_000, (name,count)
    return model
