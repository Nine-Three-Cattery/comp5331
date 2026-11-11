"""Fresh implementation of Alahi et al. (CVPR 2016) core Social-LSTM.

Metric displacements replace image-normalized absolute positions (documented departure).
All simultaneous agents update synchronously from previous hidden states.
"""
import math
import torch
from torch import nn

class SocialLSTM(nn.Module):
    def __init__(self,hidden=128,embedding=64,grid=8,neighborhood=4.,dropout=.1):
        super().__init__();self.hidden=hidden;self.grid=grid;self.neighborhood=neighborhood
        self.position_embedding=nn.Linear(2,embedding)
        self.social_embedding=nn.Linear(grid*grid*hidden,embedding)
        self.cell=nn.LSTMCell(embedding*2,hidden)
        self.head=nn.Linear(hidden,5);self.dropout=nn.Dropout(dropout)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        # Initial per-step Gaussian standard deviation is 0.2 m.
        with torch.no_grad():self.head.bias[2:4]=math.log(.2)

    def social_tensor(self,positions,hidden,groups,replicates=1):
        n=len(positions)//replicates;g=self.grid
        positions=positions.reshape(replicates,n,2);hidden=hidden.reshape(replicates,n,self.hidden)
        groups=groups.reshape(replicates,n)
        rel=positions[:,None,:,:]-positions[:,:,None,:] # recipient i, neighbor j
        cells=torch.floor((rel+self.neighborhood/2)/self.neighborhood*g).long()
        valid=((cells>=0)&(cells<g)).all(-1)&(groups[:,:,None]==groups[:,None,:])
        valid=valid&~torch.eye(n,dtype=torch.bool,device=positions.device)
        idx=(cells[...,0]+g*cells[...,1]).clamp(0,g*g-1)
        # Scatter sums h_j into every recipient/grid cell; self is excluded.
        pooled=hidden.new_zeros((replicates,n,g*g,self.hidden))
        pooled.scatter_add_(2,idx[:,:,:,None].expand(-1,-1,-1,self.hidden),
                            hidden[:,None,:,:].expand(-1,n,-1,-1)*valid[:,:,:,None])
        return pooled.reshape(replicates*n,g*g,self.hidden)

    def step(self,position,displacement,h,c,groups,replicates=1):
        social=self.social_tensor(position,h,groups,replicates).flatten(1)
        x=self.dropout(torch.relu(self.position_embedding(displacement)))
        s=self.dropout(torch.relu(self.social_embedding(social)))
        h,c=self.cell(torch.cat([x,s],-1),(h,c))
        raw=self.head(h)
        # Residual velocity parameterization improves translation invariance.
        mu=displacement+raw[:,:2]
        log_sigma=raw[:,2:4].clamp(-5,2)
        rho=.99*torch.tanh(raw[:,4])
        return (mu,log_sigma,rho),h,c

    def initial(self,obs):return obs.new_zeros((obs.shape[1],self.hidden)),obs.new_zeros((obs.shape[1],self.hidden))

    def teacher_nll(self,xy,mask,groups):
        h,c=self.initial(xy);loss=xy.new_zeros(());count=0
        # Future-incomplete context agents are observed for 8 steps, then removed
        # from future teacher forcing. They are still rolled out at inference.
        for t in range(19):
            active=torch.ones_like(mask) if t<8 else mask
            p=xy[t];d=xy[t]-xy[t-1] if t else torch.zeros_like(p)
            # Isolate unavailable future-context states so zeros cannot pool.
            grp=groups.clone();grp[~active]=-torch.arange(1,(~active).sum()+1,device=xy.device)
            params,h,c=self.step(p,d,h*active[:,None],c*active[:,None],grp)
            eligible=mask if t>=7 else torch.ones_like(mask)
            if t>=7:
                loss+=gaussian_nll(params,xy[t+1]-xy[t])[eligible].sum();count+=int(eligible.sum())
        return loss/count

    @torch.no_grad()
    def predict(self,obs,groups,samples=20,generator=None):
        # This signature accepts only observation coordinates, never future targets.
        h,c=self.initial(obs)
        for t in range(obs.shape[0]):
            p=obs[t];d=obs[t]-obs[t-1] if t else torch.zeros_like(p)
            params,h,c=self.step(p,d,h,c,groups)
        n=obs.shape[1];h=h.repeat(samples,1);c=c.repeat(samples,1)
        groups=groups.repeat(samples)+torch.arange(samples,device=obs.device).repeat_interleave(n)*(int(groups.max())+1)
        p=obs[-1].repeat(samples,1);mu,ls,rho=[v.repeat((samples,)+(1,)*(v.ndim-1)) for v in params]
        paths=[]
        for t in range(12):
            eps=torch.randn((samples*n,2),device=obs.device,generator=generator)
            sig=ls.exp();delta=mu+torch.stack([sig[:,0]*eps[:,0],sig[:,1]*(rho*eps[:,0]+torch.sqrt(1-rho.square())*eps[:,1])],-1)
            p=p+delta;paths.append(p.reshape(samples,n,2))
            if t<11:(mu,ls,rho),h,c=self.step(p,delta,h,c,groups,samples)
        return torch.stack(paths,2) # samples, agents, time, xy

def gaussian_nll(params,target):
    mu,log_sigma,rho=params;z=(target-mu)*torch.exp(-log_sigma)
    one=1-rho.square()
    q=(z.square().sum(-1)-2*rho*z[:,0]*z[:,1])/one
    return math.log(2*math.pi)+log_sigma.sum(-1)+.5*torch.log(one)+.5*q
