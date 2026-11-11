"""Synthetic training/inference smoke; does not claim shared benchmark acceptance."""
import hashlib,json,torch
from pathlib import Path
from model import SocialLSTM

def main():
    torch.set_num_threads(1);torch.manual_seed(7)
    model=SocialLSTM(hidden=16,embedding=8,grid=4,dropout=0.)
    xy=torch.zeros(20,3,2);xy[:,:,0]=torch.arange(20)[:,None]*.1
    xy[:,1,1]=.5;xy[:,2,1]=2.;groups=torch.tensor([0,0,0]);mask=torch.ones(3,dtype=torch.bool)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);loss=model.teacher_nll(xy,mask,groups)
    before=model.head.weight.detach().clone();loss.backward();optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before,model.head.weight)
    model.eval();pred=model.predict(xy[:8],groups,20,torch.Generator().manual_seed(7))
    assert pred.shape==(20,3,12,2) and torch.isfinite(pred).all()
    path=Path(__file__).with_name('model.py')
    print(json.dumps({'status':'LEGACY_SMOKE_OK_NOT_SHARED_AUDIT','loss':float(loss.detach()),
        'prediction_shape':list(pred.shape),'source_model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
