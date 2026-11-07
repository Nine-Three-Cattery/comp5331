"""Pinned ETH/UCY preparation. No dataset download occurs implicitly."""
from pathlib import Path
import hashlib, json, os, time
import numpy as np
import torch

CODE_ROOT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('SOCIAL_LSTM_LEGACY_WORKDIR',str(CODE_ROOT/'artifacts'/'legacy'))).expanduser().resolve()
SCENES={'eth':['biwi_eth'],'hotel':['biwi_hotel'],'univ':['students001','students003'],
        'zara1':['crowds_zara01'],'zara2':['crowds_zara02']}
COMMIT='333d3a57b4d2705e129b21aefefa09c79b2b9ae1'
OBS,PRED,DT=8,12,.4

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protocol_fingerprint(manifest=None):
    if manifest is None:manifest=json.loads((ROOT/'data/processed/manifest.json').read_text())
    stable={k:v for k,v in manifest.items() if k not in ['preprocessing_seconds','protocol_fingerprint']}
    return hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def current_provenance():
    """Preserve an append-only mapping from exact manifests to stable data content."""
    path=ROOT/'data/processed/manifest.json';sha=digest(path);fingerprint=protocol_fingerprint()
    directory=ROOT/'data/processed/provenance';directory.mkdir(exist_ok=True)
    snapshot=directory/f'{sha}.json'
    if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
    assert digest(snapshot)==sha
    return fingerprint,sha

def saved_data_fingerprint(config):
    if config.get('data_protocol_fingerprint'):return config['data_protocol_fingerprint']
    # Compatibility for already-started runs: verify the retained exact manifest,
    # then derive its stable content hash. Historical configs remain untouched.
    sha=config['data_manifest_sha256'];snapshot=ROOT/'data/processed/provenance'/f'{sha}.json'
    if not snapshot.exists() or digest(snapshot)!=sha:return None
    return protocol_fingerprint(json.loads(snapshot.read_text()))
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');temp.replace(path)

def make_windows(a,recording,split,stride):
    byframe={int(f):rows for f in np.unique(a[:,0]) for rows in [a[a[:,0]==f]]}
    out=[]
    if len(a)==0:return out
    for start in range(int(a[:,0].min()),int(a[:,0].max())-190+1,10*stride):
        frames=np.arange(start,start+200,10)
        if any(int(f) not in byframe for f in frames[:OBS]):continue
        ids=sorted(set.intersection(*(set(byframe[int(f)][:,1].astype(int)) for f in frames[:OBS])))
        if not ids:continue
        maps=[{int(r[1]):r[2:] for r in byframe.get(int(f),[])} for f in frames]
        target=np.array([all(p in m for m in maps) for p in ids])
        if not target.any():continue
        xy=np.zeros((20,len(ids),2),np.float32)
        for t,m in enumerate(maps):
            for j,p in enumerate(ids):
                if p in m:xy[t,j]=m[p]
        wid=f'{recording}:{start}'
        out.append({'xy':xy,'mask':target,'pids':np.array(ids),'id':wid,'recording':recording,
                    'split':split,'start':start})
    return out

def prepare():
    t0=time.perf_counter(); datasets={}; sources=[]; splits={}
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    locked={x['recording']:x['sha256'] for x in json.loads((CODE_ROOT/'configs/raw_release.json').read_text())['files']}
    for scene,names in SCENES.items():
        datasets[scene]={'train':[],'val':[],'test':[]}
        for name in names:
            path=ROOT/'data/raw'/f'{name}.txt'
            if digest(path)!=locked[name]:raise ValueError(f'Raw SHA256 mismatch: {name}')
            a=np.loadtxt(path)
            assert a.shape[1]==4 and len(np.unique(a[:,:2],axis=0))==len(a)
            assert np.all(a[:,0]%10==a[0,0]%10),name
            lo,hi=int(a[:,0].min()),int(a[:,0].max())
            boundary=lo+int(((hi-lo)/10)*.8)*10
            val_start=boundary+200
            # A track crossing the boundary is purged, so even person IDs do not leak.
            spans={int(p):(rows[:,0].min(),rows[:,0].max()) for p in np.unique(a[:,1])
                   for rows in [a[a[:,1]==p]]}
            crossing={p for p,(l,h) in spans.items() if l<val_start and h>=boundary}
            clean=~np.isin(a[:,1],list(crossing))
            tr=a[(a[:,0]<boundary)&clean];va=a[(a[:,0]>=val_start)&clean]
            datasets[scene]['train']+=make_windows(tr,name,'train',4)
            datasets[scene]['val']+=make_windows(va,name,'val',1)
            datasets[scene]['test']+=make_windows(a,name,'test',1)
            sources.append({'scene':scene,'recording':name,'path':str(path.relative_to(ROOT)),
                'sha256':digest(path),'rows':len(a),'frame_min':lo,'frame_max':hi,
                'url':f'https://raw.githubusercontent.com/abduallahmohamed/Social-STGCNN/{COMMIT}/datasets/raw/all_data/{name}.txt'})
            splits[name]={'train_end_exclusive':boundary,'val_start_inclusive':val_start,
                'purged_crossing_pids':sorted(crossing),'train_pids':sorted(set(tr[:,1].astype(int))),
                'val_pids':sorted(set(va[:,1].astype(int)))}
            assert set(tr[:,1]).isdisjoint(set(va[:,1]))
    # Portable numpy data objects are internal trusted artifacts, not input from users.
    temporary=ROOT/'data/processed'/f'windows.{os.getpid()}.pt.tmp'
    torch.save(datasets,temporary);temporary.replace(ROOT/'data/processed/windows.pt')
    counts={s:{k:{'windows':len(v),'target_trajectories':sum(int(w['mask'].sum()) for w in v),
                    'context_trajectories':sum(len(w['pids']) for w in v)} for k,v in d.items()} for s,d in datasets.items()}
    ids={s:{k:[f"{w['id']}:{p}" for w in v for p,ok in zip(w['pids'],w['mask']) if ok]
            for k,v in d.items()} for s,d in datasets.items()}
    write_json(ROOT/'data/processed/eligible_ids.json',ids)
    manifest={'protocol_status':'LEGACY EXPLORATORY; not causal_v2; shared contribution pending','release_repository':
        'https://github.com/abduallahmohamed/Social-STGCNN','release_commit':COMMIT,
        'sources':sources,'splits':splits,'counts':counts,'obs':OBS,'pred':PRED,'dt_seconds':DT,
        'train_window_stride':4,'validation_test_window_stride':1,
        'excluded_mirror_recordings':['crowds_zara03','uni_examples'],
        'eligible_ids_sha256':digest(ROOT/'data/processed/eligible_ids.json'),
        'preprocessing_seconds':time.perf_counter()-t0}
    write_json(ROOT/'data/processed/manifest.json',manifest)
    current_provenance()
    print(json.dumps(manifest['counts'],indent=2));return manifest

def load_fold(scene):
    d=torch.load(ROOT/'data/processed/windows.pt',weights_only=False)
    return (sum((v['train'] for s,v in d.items() if s!=scene),[]),
            sum((v['val'] for s,v in d.items() if s!=scene),[]),d[scene]['test'])

def collate(windows,device='cpu'):
    # Packed groups never pool across unrelated scene windows.
    xy=torch.as_tensor(np.concatenate([w['xy'] for w in windows],axis=1),device=device)
    mask=torch.as_tensor(np.concatenate([w['mask'] for w in windows]),device=device)
    groups=torch.cat([torch.full((len(w['pids']),),i,device=device,dtype=torch.long) for i,w in enumerate(windows)])
    return xy,mask,groups

def batches(windows,max_people=96,order=None):
    buf=[]; n=0
    for i in (range(len(windows)) if order is None else order):
        w=windows[i]
        if buf and n+len(w['pids'])>max_people:yield buf;buf=[];n=0
        buf.append(w);n+=len(w['pids'])
    if buf:yield buf

def visible_history(xy):
    """Actual legacy inference boundary: future coordinates never become features."""
    if len(xy)<OBS:raise ValueError('Eight observed steps required')
    return np.array(xy[:OBS],dtype=np.float32,copy=True)

def perturb(obs,window,condition,base_seed=20261105):
    x=np.array(obs,copy=True)
    # Stable per-window RNG, independent of batching and process/hash randomization.
    seed=int.from_bytes(hashlib.sha256(f"{base_seed}|{window['id']}|{condition}".encode()).digest()[:8],'little')
    if condition.startswith('noise_'):
        sigma=float(condition.split('_')[1]);x+=np.random.Generator(np.random.PCG64(seed)).normal(0,sigma,x.shape)
    elif condition.startswith('missing_'):
        k=int(condition.split('_')[1]);n=OBS-k;v=x[n-1]-x[n-2]
        for t in range(n,OBS):x[t]=x[n-1]+(t-n+1)*v
    elif condition!='clean':raise ValueError(condition)
    return x

if __name__=='__main__':prepare()
