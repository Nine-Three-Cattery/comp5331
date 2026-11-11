"""Independent deterministic social-grid recurrent predictor for causal_v2.

Joint closed-loop rollout of observation-complete actors; only target labels
supervise loss. Gaussian sampling and teacher forcing are absent.
"""
import torch
from torch import nn

class DeterministicSocialLSTM(nn.Module):
    def __init__(self, hidden=64, embedding=32, grid=4, neighborhood=4.):
        super().__init__()
        self.hidden,self.grid,self.neighborhood=hidden,grid,neighborhood
        self.motion=nn.Linear(2,embedding)
        self.social=nn.Linear(grid*grid*hidden,embedding)
        self.cell=nn.LSTMCell(2*embedding,hidden)
        self.head=nn.Linear(hidden,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def pool(self,p,h,valid):
        b,n,_=p.shape;g=self.grid
        rel=p[:,None,:,:]-p[:,:,None,:]
        cells=torch.floor((rel+self.neighborhood/2)*g/self.neighborhood).long()
        active=((cells>=0)&(cells<g)).all(-1)&valid[:,:,None]&valid[:,None,:]
        active=active&~torch.eye(n,dtype=torch.bool,device=p.device)[None]
        index=(cells[...,0]+g*cells[...,1]).clamp(0,g*g-1)
        pool=h.new_zeros((b,n,g*g,self.hidden))
        pool.scatter_add_(2,index[...,None].expand(-1,-1,-1,self.hidden),
                          h[:,None].expand(-1,n,-1,-1)*active[...,None])
        return pool.flatten(2)

    def step(self,p,d,h,c,valid):
        x=torch.cat((torch.relu(self.motion(d)),torch.relu(self.social(self.pool(p,h,valid)))),dim=-1)
        shape=h.shape
        h,c=self.cell(x.flatten(0,1),(h.flatten(0,1),c.flatten(0,1)))
        return h.reshape(shape)*valid[...,None],c.reshape(shape)*valid[...,None]

    def forward(self, observed, valid):
        # observed [B,P,8,2] contains no labels. Actor 0 is the target.
        origin=observed[:,0,-1,:].clone()
        obs=observed-origin[:,None,None,:]
        h=obs.new_zeros((*obs.shape[:2],self.hidden));c=torch.zeros_like(h)
        for t in range(8):
            delta=obs[:,:,t]-obs[:,:,t-1] if t else torch.zeros_like(obs[:,:,0])
            h,c=self.step(obs[:,:,t],delta,h,c,valid)
        # Fixed history-only drift; never accumulate innovations into this anchor.
        base=(obs[:,:,-1]-obs[:,:,-4])/3
        p=obs[:,:,-1];pred=[]
        for t in range(12):
            delta=base+self.head(h)
            p=p+delta;pred.append(p[:,0]+origin)
            if t<11:h,c=self.step(p,delta,h,c,valid)
        return torch.stack(pred,1)[:,None] # B,K=1,T,xy
