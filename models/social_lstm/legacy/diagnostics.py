"""Offline diagnostics with thresholds derived exclusively from each fold's training data.

Existing prior-based diagnostics remain available; no model is retrained or changed.
"""
import json
import numpy as np
from data import ROOT,SCENES,load_fold,write_json,protocol_fingerprint,digest
from metrics import motion_arrays

def fit(train):
    angles=[];distances=[]
    for w in train:
        x=w['xy'][:,w['mask']].transpose(1,0,2)
        vel=np.diff(x,axis=1)/.4;speed=np.linalg.norm(vel,axis=-1)
        cosine=(vel[:,1:]*vel[:,:-1]).sum(-1)/(speed[:,1:]*speed[:,:-1]+1e-9)
        moving=(speed[:,1:]>.1)&(speed[:,:-1]>.1)
        angles.extend(np.arccos(np.clip(cosine,-1,1))[moving])
        if len(x)>1:
            i,j=np.triu_indices(len(x),1)
            distances.extend(np.linalg.norm(x[i,8:]-x[j,8:],axis=-1).ravel())
    return {'heading_99_radians':float(np.quantile(angles,.99)),
            'proximity_01_meters':float(np.quantile(distances,.01)),
            'heading_values':len(angles),'pair_time_values':len(distances),
            'heading_definition':'99th percentile of consecutive training-target velocity heading changes; both speeds > .1 m/s; all 20 training positions',
            'proximity_definition':'1st percentile of unordered synchronous training-target pair distances over the 12 future steps; weighted by pair-time',
            'interpretation':'distribution-tail diagnostics, not collision or safety thresholds',
            'fit_split':'training only; no held-out scene or validation data',
            'data_protocol_fingerprint':protocol_fingerprint()}

def evaluate(path,thresholds):
    z=np.load(path);p=z['pred'];obs=z['observed'];offsets=z['offsets']
    _,_,angle,moving=motion_arrays(p,obs[:,-1],(obs[:,-1]-obs[:,-2])/.4)
    heading_count=int(((angle>thresholds['heading_99_radians'])&moving).sum());moving_count=int(moving.sum())
    close=0;pairs=0
    for start,end in zip(offsets[:-1],offsets[1:]):
        n=int(end-start)
        if n<2:continue
        i,j=np.triu_indices(n,1);d=np.linalg.norm(p[:,start+i]-p[:,start+j],axis=-1)
        close+=int((d<thresholds['proximity_01_meters']).sum());pairs+=d.size
    return {'abrupt_turn_train99_rate':heading_count/max(1,moving_count),
            'close_approach_train01_pair_rate':close/max(1,pairs),
            'heading_exceed_count':heading_count,'moving_turn_count':moving_count,
            'close_count':close,'pair_time_count':pairs,'thresholds':thresholds,
            'source_prediction_sha256':digest(path),'test_set_id':str(z['test_set_id'])}

def main():
    all_thresholds={}
    for scene in SCENES:
        train,_,_=load_fold(scene);th=fit(train);all_thresholds[scene]=th
        for run in sorted((ROOT/'runs').glob(scene+'_seed[0-9][0-9]')):
            for path in sorted((run/'evaluation').glob('*_predictions.npz')):
                if path.name.startswith('cv_'):continue
                result=evaluate(path,th);out=path.with_name(path.name.replace('_predictions.npz','_training_diagnostics.json'));write_json(out,result)
    write_json(ROOT/'results/training_diagnostic_thresholds.json',all_thresholds)
    print(json.dumps({k:{'heading_99_degrees':v['heading_99_radians']*180/np.pi,'proximity_01_meters':v['proximity_01_meters']} for k,v in all_thresholds.items()},indent=2))

if __name__=='__main__':main()
