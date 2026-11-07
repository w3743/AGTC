"""Fixed-capacity streaming memory with learned sparse writes and reads."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


VOCAB = 64
CHUNK = 64
SLOTS = 32
WIDTH = 32


class Stream(NamedTuple):
    slots: torch.Tensor
    previous: torch.Tensor
    pending: torch.Tensor


class SlotRead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(WIDTH,WIDTH)
        self.k = nn.Linear(WIDTH,WIDTH)
        self.v = nn.Linear(WIDTH,WIDTH)
        self.out = nn.Linear(WIDTH,WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.ff = nn.Sequential(nn.Linear(WIDTH,1024), nn.GELU(), nn.Linear(1024,WIDTH))
        self.final = nn.LayerNorm(WIDTH)

    def forward(self, query, slots, mode, hop, forced=None):
        batch = slots.shape[0]
        q = self.q(query).reshape(batch,4,8)
        k = self.k(slots).reshape(batch,SLOTS,4,8).transpose(1,2)
        scores = (q.unsqueeze(2)*k).sum(-1)/math.sqrt(8)
        if mode == 'dense':
            v = self.v(slots).reshape(batch,SLOTS,4,8).transpose(1,2)
            message = (scores.softmax(-1).unsqueeze(-1)*v).sum(2)
            edges = None
        else:
            if forced is not None:
                edges = forced
            elif mode == 'fixed':
                edges = ((torch.arange(16,device=slots.device).reshape(4,4)+hop*8)%SLOTS).expand(batch,-1,-1)
            else:
                edges = scores.topk(4,dim=-1).indices
            # Apply only the relevant head's value projection to its four sources.
            chosen = slots[torch.arange(batch,device=slots.device)[:,None,None],edges]
            w = self.v.weight.reshape(4,8,WIDTH)
            b = self.v.bias.reshape(4,8)
            v = torch.einsum('bhkd,hod->bhko',chosen,w)+b[None,:,None,:]
            message = (scores.gather(-1,edges).softmax(-1).unsqueeze(-1)*v).sum(2)
            if self.training and mode == 'adaptive' and forced is None:
                soft = scores.softmax(-1)
                proxy_values = self.v(slots).detach().reshape(batch,SLOTS,4,8).transpose(1,2)
                message = message + ((soft-soft.detach()).unsqueeze(-1)*proxy_values).sum(2)
        query = self.norm(query+self.out(message.reshape(batch,WIDTH)))
        return self.final(query+self.ff(query)), edges


class SlotMemoryGraph(nn.Module):
    def __init__(self, mode='adaptive'):
        super().__init__()
        if mode not in ('adaptive','fixed','dense'):
            raise ValueError(mode)
        self.mode = mode
        self.token = nn.Embedding(VOCAB,WIDTH,padding_idx=0)
        self.address = nn.Parameter(torch.empty(SLOTS,WIDTH))
        nn.init.normal_(self.address,std=.15)
        self.encoder = nn.Sequential(nn.Linear(WIDTH*3,WIDTH),nn.GELU(),nn.LayerNorm(WIDTH))
        self.write_q = nn.Linear(WIDTH,WIDTH)
        self.write_k = nn.Linear(WIDTH,WIDTH)
        self.update = nn.Sequential(nn.Linear(WIDTH,256),nn.GELU(),nn.Linear(256,WIDTH))
        self.gate = nn.Linear(WIDTH*2,WIDTH)
        self.query = nn.Linear(WIDTH*2,WIDTH)
        self.read = SlotRead()
        self.head = nn.Linear(WIDTH,4)

    def initial(self,batch,device):
        empty=torch.zeros(batch,0,dtype=torch.long,device=device)
        return Stream(self.address.expand(batch,-1,-1), torch.zeros(batch,2,dtype=torch.long,device=device),empty)

    def write_block(self,slots,previous,block):
        joined=torch.cat((previous,block),dim=1)
        encoded=self.encoder(torch.cat([self.token(joined[:,i:i+block.shape[1]]) for i in range(3)],dim=-1))
        keys=self.write_k(encoded)
        scores=self.write_q(slots+self.address) @ keys.transpose(1,2)/math.sqrt(WIDTH)
        valid=block.ne(0)
        scores=scores.masked_fill(~valid[:,None,:],-1e4)
        edge=scores.argmax(-1)
        selected=encoded.gather(1,edge.unsqueeze(-1).expand(-1,-1,WIDTH))
        if self.training:
            soft=scores.softmax(-1)
            selected=selected+(soft-soft.detach())@encoded.detach()
        gate=self.gate(torch.cat((slots,selected),-1)).sigmoid()
        gate=gate*valid.any(-1)[:,None,None]
        return slots+gate*(self.update(selected)-slots),joined[:,-2:],edge

    def push(self,state,tokens):
        slots,previous,pending=state
        # Keep at most 63 unprocessed tokens in a returned streaming state.
        joined=torch.cat((pending,tokens),dim=1)
        full=joined.shape[1]//CHUNK*CHUNK
        for start in range(0,full,CHUNK):
            slots,previous,_=self.write_block(slots,previous,joined[:,start:start+CHUNK])
        return Stream(slots,previous,joined[:,full:].clone())

    def answer(self,state,query,forced=None,return_edges=False):
        slots,previous,pending=state
        if pending.shape[1]:
            slots,_,_=self.write_block(slots,previous,pending)
        h=self.query(self.token(query).flatten(1))
        edges=[]
        for hop in range(4):
            h,edge=self.read(h,slots,self.mode,hop,None if forced is None else forced[hop])
            edges.append(edge)
        logits=self.head(h)
        return (logits,edges) if return_edges else logits

    def forward(self,digits,*,forced=None,return_edges=False):
        state=self.push(self.initial(len(digits),digits.device),digits[:,:-2])
        return self.answer(state,digits[:,-2:],forced,return_edges)
