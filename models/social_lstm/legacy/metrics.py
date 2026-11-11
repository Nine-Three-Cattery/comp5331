"""Metrics accept [K,N,T,2] metric-coordinate samples and [N,T,2] targets."""
import numpy as np

def per_trajectory(pred,target):
    pred=np.asarray(pred,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    k,n,t,d=pred.shape
    assert target.shape==(n,t,d)
    dist=np.linalg.norm(pred-target[None],axis=-1)
    ade=dist.mean(-1);fde=dist[:,:,-1]
    # Euclidean norm of the concatenated 24-D future-coordinate vector.
    first=np.linalg.norm((pred-target[None]).reshape(k,n,-1),axis=-1).mean(0)
    pair_sum=np.zeros(n);pair_rms=np.zeros(n)
    for i in range(k):
        for j in range(i):
            norm=np.linalg.norm((pred[i]-pred[j]).reshape(n,-1),axis=-1)
            pair_sum+=norm;pair_rms+=norm/np.sqrt(t)
    return {'mean_ADE':ade.mean(0),'mean_FDE':fde.mean(0),
            'min_ADE':ade.min(0),'min_FDE':fde.min(0),
            'energy_score':first-pair_sum/(k*k),
            'diversity_RMS':pair_rms/max(1,k*(k-1)/2)}

def motion_arrays(pred,last_obs,last_velocity,dt=.4):
    vel=np.diff(np.concatenate([np.broadcast_to(last_obs,(pred.shape[0],)+last_obs.shape)[:,:,None],pred],axis=2),axis=2)/dt
    acc=np.diff(np.concatenate([np.broadcast_to(last_velocity,(pred.shape[0],)+last_velocity.shape)[:,:,None],vel],axis=2),axis=2)/dt
    speed=np.linalg.norm(vel,axis=-1);accel=np.linalg.norm(acc,axis=-1)
    cos=(vel[:,:,1:]*vel[:,:,:-1]).sum(-1)/(np.linalg.norm(vel[:,:,1:],axis=-1)*np.linalg.norm(vel[:,:,:-1],axis=-1)+1e-9)
    angle=np.arccos(np.clip(cos,-1,1))
    moving=(speed[:,:,1:]>.1)&(speed[:,:,:-1]>.1)
    return speed,accel,angle,moving

def aggregate(values):return {k:float(np.concatenate(v).mean()) for k,v in values.items()}

def close_approaches(pred,radius=.2):
    # Count unordered pedestrian pairs at each synchronous predicted timestep.
    k,n,t,_=pred.shape
    if n<2:return 0,0
    ii,jj=np.triu_indices(n,1)
    d=np.linalg.norm(pred[:,ii]-pred[:,jj],axis=-1)
    return int((d<radius).sum()),int(d.size)
