"""Our deterministic baseline design, not a claimed paper-exact reproduction."""
import torch
from torch import nn

class IndividualLSTM(nn.Module):
    def __init__(self,embedding=16,hidden=32,pred_len=12):
        super().__init__()
        self.pred_len=pred_len
        self.embedding=nn.Sequential(nn.Linear(2,embedding),nn.ReLU())
        self.encoder=nn.LSTM(embedding,hidden,batch_first=True)
        self.decoder=nn.LSTMCell(embedding,hidden)
        self.head=nn.Linear(hidden,2)

    def forward(self,obs):
        # Difference input is translation invariant; units remain meters/step.
        delta=obs[:,1:]-obs[:,:-1]
        _,(h,c)=self.encoder(self.embedding(delta))
        h,c=h[0],c[0]
        step=delta[:,-1]
        position=torch.zeros_like(step)
        positions=[]
        for _ in range(self.pred_len):
            h,c=self.decoder(self.embedding(step),(h,c))
            step=self.head(h)
            position=position+step
            positions.append(position)
        return obs[:,-1,None,:]+torch.stack(positions,dim=1)

def constant_velocity(obs,pred_len=12):
    delta=obs[:,-1]-obs[:,-2]
    t=torch.arange(1,pred_len+1,device=obs.device,dtype=obs.dtype)
    return obs[:,-1,None,:]+t[None,:,None]*delta[:,None,:]
