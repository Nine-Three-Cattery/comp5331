"""Group-7 causal_v2 adapter for the owner's deterministic position LSTM."""
import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from model import IndividualLSTM, constant_velocity

ROOT = Path(__file__).resolve().parents[2]
OWNER = ROOT / 'li_junhao_trajectronpp'
sys.path[:0] = [str(OWNER), str(OWNER/'scripts')]
from independent_tpp.data import load_split
from independent_tpp.protocol import (FOLDS, SEEDS, CONDITIONS, corruption_seed,
                                     perturb_history, sha256, digest_json, write_json)
from analyze_sample_budget import metrics_for_budget_batch

OUTPUT = ROOT/'artifacts/individual_lstm'
DATA = ROOT/'data/causal_v2'
CONFIG = dict(embedding=16, hidden=32, lr=0.001, batch_size=512,
              patience=8, min_delta=1e-5, grad_clip=1.0, epochs=40,
              sample_count=1, cpu_threads=2, model='individual_lstm',
              criterion='mean clean validation ADE; improvement > 1e-5 m; retain earlier incumbent')

def history(store, record):
    """Read strictly the target's visible positions, never precomputed derivatives."""
    track = store.tracks[record.scene_key][record.target_id]
    return np.stack([track[t][:2] for t in range(record.anchor-7,record.anchor+1)])

def materialize(store):
    obs=np.stack([history(store,r) for r in store.manifests])
    y=np.array([[store.state(r.scene_key,r.target_id,r.anchor+t)[:2] for t in range(1,13)]
                for r in store.manifests],dtype=np.float32)
    return obs,y

@torch.inference_mode()
def predict(model,x):
    return torch.cat([model(batch) for batch in x.split(1024)]).numpy()

def fit(run,fold,seed,config,train,val):
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'config.json',config)
    torch.manual_seed(seed)
    x,y=[torch.as_tensor(a,dtype=torch.float32) for a in train]
    v=torch.as_tensor(val[0],dtype=torch.float32)
    model=IndividualLSTM(config['embedding'],config['hidden'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    best=float('inf'); candidates=[]
    t0=time.perf_counter()
    for ep in range(1,config['epochs']+1):
        model.train()
        for ids in torch.randperm(len(x)).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x[ids])-y[ids]).square().sum(-1).mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'])
            optimizer.step()
        model.eval()
        ade=float(np.linalg.norm(predict(model,v).astype(float)-val[1],axis=-1).mean())
        candidates.append({'epoch':ep,'val_mean_ade_m':ade})
        if ade<best-config['min_delta']:
            best=ade; selected=ep
            torch.save(model.state_dict(),run/'best.pt')
        if ep-selected >= config['patience']: break
    model.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu'))
    model.eval()
    assert np.isclose(np.linalg.norm(predict(model,v).astype(float)-val[1],axis=-1).mean(),best)
    write_json(run/'checkpoint_selection.json',dict(selection_split='val',test_used_for_selection=False,
               rule=config['criterion'],selected={'checkpoint':'best.pt','epoch':selected,'ade_m':best},
               candidates=candidates,elapsed_seconds=time.perf_counter()-t0))
    return model

def export(run,fold,seed,model,config,store,test):
    history64,truth=test
    cv_rows=[]
    for tag,kind,level in CONDITIONS:
        items=[perturb_history(h,kind,level,corruption_seed(seed,i)) for i,h in enumerate(history64)]
        obs=np.asarray([a for a,_ in items],np.float32); mask=np.stack([m for _,m in items])
        x=torch.from_numpy(obs)
        predictions=predict(model,x)[:,None]
        # CV has no learned or sampled weights; seed pairs its corruption with the LSTM's.
        cv=predict(constant_velocity,x)[:,None]
        scores=metrics_for_budget_batch(cv.astype(float),truth.astype(float),1)
        cv_rows.append(dict(fold=fold,seed=seed,condition=tag,n=len(cv),k=1,
                            manifest_hash=store.identity['manifest_hash'],
                            **{k:float(v.mean()) for k,v in scores.items()}))
        records=[json.dumps(dict(r.values, target_id=r.target_id, fold=fold,split='test',
                    seed=seed,sample_count=1,sampling_seed=seed,dt=0.4,coordinate_unit='meter',
                    condition=kind,sigma=level if kind=='noise' else 0,missing=level if kind=='missing' else 0),
                    separators=(',',':')) for r in store.manifests]
        meta=dict(store.identity,seed=seed,sample_count=1,condition=kind,level=level,
                  model_name='individual_lstm',config_hash=digest_json(config),
                  checkpoint='best.pt',checkpoint_sha256=sha256(run/'best.pt'))
        dest=run/'evaluation'/tag
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/f'predictions_test_{tag}.npz',samples=predictions,ground_truth=truth,
                            observed=obs,observed_mask=mask,records=records,metadata=json.dumps(meta))
    write_json(run/'cv_supplement.json',cv_rows)

@torch.inference_mode()
def runtime(run,model,store):
    fold=store.identity['fold']
    ids=json.loads((ROOT/'benchmark/fixtures/runtime_windows.json').read_text())[fold]
    lookup={r.window_id:r for r in store.manifests}
    records=[lookup[i] for i in ids]
    assert len(records)==128
    rows=[]
    for i in range(13):
        t=time.perf_counter()
        batches=[torch.from_numpy(np.stack([history(store,r) for r in records[j:j+32]]).astype(np.float32))
                 for j in range(0,128,32)]
        t1=time.perf_counter(); out=[model(x) for x in batches]; t2=time.perf_counter()
        cpu=[x.cpu() for x in out]; t3=time.perf_counter()
        if i>=3: rows.append(dict(preprocessing_ms=(t1-t)*1000,prediction_sampling_ms=(t2-t1)*1000,
                                  end_to_end_ms=(t3-t)*1000))
    write_json(run/'runtime.json',dict(window_ids=ids,repetitions=rows,
                medians={k:statistics.median(r[k] for r in rows) for k in rows[0]},
                platform=platform.platform(),hardware=platform.machine(),cpu_threads=2,
                python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='CPU',
                warmups=3,batch_size=32,windows=128,k=1))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['pilot','formal'])
    p.add_argument('--folds',nargs='+',choices=FOLDS)
    a=p.parse_args()
    pilot=a.stage=='pilot'
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    t=time.perf_counter()
    for fold in (('eth',) if pilot else a.folds or FOLDS):
        stores={s:load_split(fold,s,str(DATA)) for s in ('train','val','test')}
        data={s:materialize(v) for s,v in stores.items()}
        for seed in ((0,) if pilot else SEEDS):
            cfg=dict(CONFIG,fold=fold,seed=seed,stage=a.stage,epochs=2 if pilot else CONFIG['epochs'],
                     protocol='causal_v2',train_identity=stores['train'].identity,val_identity=stores['val'].identity)
            run=OUTPUT/a.stage/'runs'/fold/f'seed_{seed}'
            model=fit(run,fold,seed,cfg,data['train'],data['val'])
            runtime(run,model,stores['test'])
            export(run,fold,seed,model,cfg,stores['test'],data['test'])
            print('OWNER_RUN_OK',a.stage,fold,seed,round(time.perf_counter()-t,2),flush=True)
    write_json(OUTPUT/a.stage/('elapsed_'+('_'.join(a.folds) if a.folds else 'all')+'.json'),
               dict(seconds=time.perf_counter()-t,peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    peak_rss_unit='byte' if sys.platform=='darwin' else 'KiB'))

if __name__=='__main__': main()
