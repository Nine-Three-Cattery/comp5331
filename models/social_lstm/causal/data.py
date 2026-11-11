"""Owner feature builder over the unchanged common neutral loader.

No stored derivative columns or future coordinates enter the model features.
Neighbor membership uses only its eight available history coordinates.
"""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'li_junhao_trajectronpp'))
from independent_tpp.data import load_split
from independent_tpp.protocol import perturb_history,corruption_seed

class Windows:
    def __init__(self,store):
        self.store=store;self.histories=[];self.future=[];self.context=[]
        cache={}
        for rec in store.manifests:
            key=(rec.scene_key,rec.anchor)
            if key not in cache:
                tracks=store.tracks[rec.scene_key];frames=range(rec.anchor-7,rec.anchor+1)
                ids=[i for i in sorted(tracks) if all(t in tracks[i] for t in frames)]
                histories=np.stack([[tracks[i][t][:2] for t in frames] for i in ids])
                cache[key]=(ids,histories)
            ids,hist=cache[key]
            target=ids.index(rec.target_id)
            self.histories.append(hist[target].copy())
            self.context.append(hist[np.arange(len(ids))!=target])
            track=store.tracks[rec.scene_key][rec.target_id]
            self.future.append([track[t][:2] for t in range(rec.anchor+1,rec.anchor+13)])
        self.histories=np.asarray(self.histories,np.float64)
        self.future=np.asarray(self.future,np.float32)
    def __len__(self):return len(self.histories)
    def features(self,index,seed=0,condition='clean',level=0):
        target,mask=perturb_history(self.histories[index],condition,level,corruption_seed(seed,index))
        neighbors=self.context[index]
        # Retain any neighbor within 6m at any observed time after target corruption.
        # This broad, causal admission radius is distinct from the 4m pooling square.
        near=(np.linalg.norm(neighbors-target[None],axis=-1)<=6).any(-1)
        obs=np.concatenate([target[None],neighbors[near]],axis=0).astype(np.float32)
        return obs,mask
    def batch(self,indices,device='cpu',seed=0,condition='clean',level=0):
        items=[self.features(int(i),seed,condition,level) for i in indices]
        n=max(len(x[0]) for x in items)
        obs=np.zeros((len(items),n,8,2),np.float32);valid=np.zeros((len(items),n),bool)
        masks=[]
        for j,(x,mask) in enumerate(items):obs[j,:len(x)]=x;valid[j,:len(x)]=True;masks.append(mask)
        return torch.from_numpy(obs).to(device),torch.from_numpy(valid).to(device),np.asarray(masks)

def load(fold,split,data_root=None):
    return Windows(load_split(fold,split,str(data_root or ROOT/'data/causal_v2')))
