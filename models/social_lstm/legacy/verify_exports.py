"""Fresh independent metric recomputation from saved arrays (no model import)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def verify(path):
    path=Path(path);z=np.load(path);p=z['pred'].astype(np.float64);y=z['target'].astype(np.float64)
    k,n,t,d=p.shape;assert k in [1,20] and (t,d)==(12,2);assert y.shape==(n,t,d)
    assert z['observed'].shape==(n,8,2) and len(z['ids'])==n
    assert int(z['offsets'][-1])==n and len(z['window_ids'])+1==len(z['offsets'])
    uid=hashlib.sha256(('\n'.join(z['ids'])+'\n').encode()).hexdigest();assert str(z['test_set_id'])==uid
    e=np.sqrt(((p-y[None])**2).sum(-1));ade=e.mean(-1);fde=e[:,:,-1]
    # Broadcasting all K pairs independently checks streaming pairwise implementation.
    sequence=p.transpose(1,0,2,3).reshape(n,k,-1)
    pair_mean=[];pair_diversity=[]
    for start in range(0,n,256):
        block=sequence[start:start+256]
        pair=np.sqrt(((block[:,:,None]-block[:,None,:])**2).sum(-1))
        pair_mean.extend(pair.mean((1,2)))
        pair_diversity.extend(pair.sum((1,2))/(max(k*(k-1),1)*np.sqrt(t)))
    energy=np.sqrt(((p-y[None])**2).sum((2,3))).mean(0)-.5*np.array(pair_mean)
    diversity=np.array(pair_diversity)
    values={'mean_ADE':ade.mean(),'mean_FDE':fde.mean(),'min_ADE':ade.min(0).mean(),
            'min_FDE':fde.min(0).mean(),'energy_score':energy.mean(),'diversity_RMS':diversity.mean()}
    reported=json.loads(path.with_name(path.name.replace('_predictions.npz','_metrics.json')).read_text())
    assert reported['test_set_id']==uid and reported['n_trajectories']==n
    for key,val in values.items():np.testing.assert_allclose(val,reported[key],rtol=2e-6,atol=2e-6,err_msg=key)
    return {'file':str(path),'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'samples':k,'targets':n,'max_error':max(abs(float(v)-reported[key]) for key,v in values.items()),'status':'PASS'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+');args=parser.parse_args()
    for path in args.paths:print(json.dumps(verify(path)))
