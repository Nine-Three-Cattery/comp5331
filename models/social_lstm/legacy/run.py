"""Training, auditable prediction export, perturbations, and runtime CLI."""
import argparse, copy, csv, hashlib, json, os, platform, random, sys, time
from pathlib import Path
import numpy as np
import torch
from data import CODE_ROOT,ROOT,SCENES,load_fold,collate,batches,write_json,digest,perturb,visible_history,current_provenance,saved_data_fingerprint,protocol_fingerprint
from model import SocialLSTM
from metrics import per_trajectory,motion_arrays,close_approaches

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def sync(device):
    if str(device).startswith('mps'):torch.mps.synchronize()
def architecture(cfg):return {k:cfg[k] for k in ['hidden','embedding','grid','neighborhood','dropout']}
def thresholds(train):
    v=[];a=[]
    for w in train:
        x=w['xy'][:,w['mask']];vel=np.diff(x,axis=0)/.4
        v.extend(np.linalg.norm(vel,axis=-1).ravel());a.extend(np.linalg.norm(np.diff(vel,axis=0)/.4,axis=-1).ravel())
    return {'speed_99_mps':float(np.quantile(v,.99)),'acceleration_99_mps2':float(np.quantile(a,.99)),
            'turn_angle_radians':float(np.pi/2),'close_approach_radius_m':.2,
            'fit_split':'speed/acceleration: training only; angle/radius: fixed a-priori diagnostic choices'}

def train_one(scene,seed,cfg,device,epochs=None,tag=''):
    seed_all(seed);train,val,_=load_fold(scene);run=ROOT/'runs'/f'{scene}_seed{seed}{tag}';run.mkdir(parents=True,exist_ok=True)
    fingerprint,manifest_sha=current_provenance()
    config={**cfg,'fold':scene,'seed':seed,'device':device,'epochs':epochs or cfg['epochs'],
            'architecture':architecture(cfg),'data_manifest_sha256':digest(ROOT/'data/processed/manifest.json'),
            'data_protocol_fingerprint':fingerprint,
            'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
            'train_windows':len(train),'val_windows':len(val),'thresholds':thresholds(train)}
    write_json(run/'config.json',config)
    model=SocialLSTM(**architecture(cfg)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=float('inf');log=[];total=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        t0=time.perf_counter();model.train();s=0.;n=0;maxnorm=0.
        for ws in batches(train,cfg['max_people'],np.random.permutation(len(train))):
            xy,mask,g=collate(ws,device);opt.zero_grad(set_to_none=True);loss=model.teacher_nll(xy,mask,g)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);maxnorm=max(maxnorm,float(norm))
            opt.step();n+=int(mask.sum());s+=float(loss.detach())*int(mask.sum())
        model.eval();vs=0.;vn=0
        with torch.no_grad():
            for ws in batches(val,cfg['max_people']):
                xy,mask,g=collate(ws,device);loss=model.teacher_nll(xy,mask,g);vs+=float(loss)*int(mask.sum());vn+=int(mask.sum())
        sync(device);row={'epoch':epoch,'train_nll':s/n,'val_nll':vs/vn,'seconds':time.perf_counter()-t0,'max_gradient_norm':maxnorm}
        log.append(row)
        if row['val_nll']<best:
            best=row['val_nll'];torch.save({'state_dict':model.state_dict(),'config':config,'epoch':epoch,'val_nll':best},run/'best.pt')
        write_json(run/'training_log.json',log);print(json.dumps({'fold':scene,'seed':seed,**row}),flush=True)
    write_json(run/'training_summary.json',{'total_seconds':time.perf_counter()-total,'selected_val_nll':best,'epochs_completed':len(log)})
    return run

def prediction_generator(window,seed,base,device):
    # Same random draws across corruption conditions; different across windows/training seeds.
    n=int.from_bytes(hashlib.sha256(f"{base}|{seed}|{window['id']}".encode()).digest()[:8],'little')%(2**63-1)
    return torch.Generator(device=device).manual_seed(n)

def evaluate(run,conditions=None,device='cpu',export=True,limit=None,baseline=False):
    run=Path(run);ckpt=torch.load(run/'best.pt',map_location=device,weights_only=False);cfg=ckpt['config']
    scene,seed=cfg['fold'],cfg['seed'];_,_,test=load_fold(scene)
    if limit:test=test[:limit]
    model=SocialLSTM(**cfg['architecture']).to(device);model.load_state_dict(ckpt['state_dict']);model.eval()
    out=run/'evaluation';out.mkdir(exist_ok=True);rows=[]
    for condition in conditions or cfg['conditions']:
        sums={};records=[];predictions=[];targets=[];histories=[];ids=[];window_ids=[];offsets=[0]
        count=0;motion={'speed_sum':0.,'speed_count':0,'acceleration_sum':0.,'acceleration_count':0,
            'speed_exceed':0,'acceleration_exceed':0,'abrupt_turn':0,'moving_turn_count':0,'close_count':0,'pair_count':0}
        infer=0.;metric=0.;t0=time.perf_counter()
        for w in test:
            obs=perturb(visible_history(w['xy']),w,condition,cfg['corruption_seed'])
            x=torch.as_tensor(obs,device=device);g=torch.zeros(x.shape[1],dtype=torch.long,device=device)
            sync(device);t=time.perf_counter()
            if baseline:
                p=(x[-1][None,None]+torch.arange(1,13,device=device)[None,:,None,None]*(x[-1]-x[-2])[None,None]).permute(0,2,1,3)
            else:p=model.predict(x,g,cfg['samples'],prediction_generator(w,seed,cfg['prediction_seed'],device))
            sync(device);infer+=time.perf_counter()-t;t=time.perf_counter()
            p=p.cpu().numpy()[:,w['mask']];y=w['xy'][8:,w['mask']].transpose(1,0,2);ob=obs[:,w['mask']].transpose(1,0,2)
            met=per_trajectory(p,y)
            for k,v in met.items():sums.setdefault(k,[]).append(v)
            speed,accel,angle,moving=motion_arrays(p,ob[:,-1],(ob[:,-1]-ob[:,-2])/.4)
            motion['speed_sum']+=float(speed.sum());motion['speed_count']+=speed.size;motion['acceleration_sum']+=float(accel.sum());motion['acceleration_count']+=accel.size
            motion['speed_exceed']+=int((speed>cfg['thresholds']['speed_99_mps']).sum());motion['acceleration_exceed']+=int((accel>cfg['thresholds']['acceleration_99_mps2']).sum())
            motion['abrupt_turn']+=int(((angle>np.pi/2)&moving).sum());motion['moving_turn_count']+=int(moving.sum())
            close,pairs=close_approaches(p);motion['close_count']+=close;motion['pair_count']+=pairs
            useids=[f"{w['id']}:{pid}" for pid,m in zip(w['pids'],w['mask']) if m]
            for j,uid in enumerate(useids):records.append({'id':uid,'window':w['id'],**{k:float(v[j]) for k,v in met.items()}})
            ids.extend(useids);window_ids.append(w['id']);count+=len(y);offsets.append(count)
            if export:predictions.append(p.astype(np.float32));targets.append(y);histories.append(ob)
            metric+=time.perf_counter()-t
        allmetrics={k:float(np.concatenate(v).mean()) for k,v in sums.items()}
        row={'model':'ConstantVelocity' if baseline else 'Social-LSTM','scene':scene,'seed':seed,'condition':condition,
             'K':1 if baseline else cfg['samples'],'n_trajectories':count,'n_windows':len(test),
             'test_set_id':hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
             'eligible_manifest_sha256':digest(ROOT/'data/processed/eligible_ids.json'),
             'is_partial':bool(limit),'checkpoint_epoch':ckpt['epoch'],
             'checkpoint_sha256':digest(run/'best.pt'),'protocol_status':'provisional',**allmetrics,
             'mean_speed_mps':motion['speed_sum']/motion['speed_count'],'mean_acceleration_mps2':motion['acceleration_sum']/motion['acceleration_count'],
             'speed_exceed_rate':motion['speed_exceed']/motion['speed_count'],'acceleration_exceed_rate':motion['acceleration_exceed']/motion['acceleration_count'],
             'abrupt_turn_rate':motion['abrupt_turn']/max(1,motion['moving_turn_count']),
             'close_approach_pair_rate':motion['close_count']/max(1,motion['pair_count']),
             'inference_seconds':infer,'metric_seconds':metric,'wall_seconds':time.perf_counter()-t0,'device':device}
        prefix=('cv_' if baseline else '')+condition
        write_json(out/f'{prefix}_metrics.json',row);write_json(out/f'{prefix}_per_trajectory.json',records)
        if export:
            np.savez_compressed(out/f'{prefix}_predictions.npz',pred=np.concatenate(predictions,axis=1),
                target=np.concatenate(targets),observed=np.concatenate(histories),ids=np.array(ids),
                window_ids=np.array(window_ids),offsets=np.array(offsets),condition=condition,
                checkpoint_sha256=row['checkpoint_sha256'],test_set_id=row['test_set_id'],
                is_partial=row['is_partial'],eligible_manifest_sha256=row['eligible_manifest_sha256'])
        rows.append(row);print(json.dumps(row),flush=True)
    return rows

def benchmark(cfg,device,samples=20):
    seed_all(53031);model=SocialLSTM(**architecture(cfg)).to(device).eval();rows=[]
    for n in [8,16,32]:
        obs=torch.cumsum(torch.randn(8,n,2,device=device)*.05,0);g=torch.zeros(n,dtype=torch.long,device=device)
        for _ in range(3):model.predict(obs,g,samples)
        sync(device);times=[]
        for _ in range(10):
            sync(device);t=time.perf_counter();model.predict(obs,g,samples);sync(device);times.append(time.perf_counter()-t)
        rows.append({'people':n,'K':samples,'median_seconds':float(np.median(times)),'p90_seconds':float(np.quantile(times,.9)),
                     'repeats':10,'warmup':3,'boundary':f'observed tensor on device -> {samples} joint future tensor on device; allocation/RNG included; data loading/metrics/export excluded'})
    result={'device':device,'hardware':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'threads':torch.get_num_threads(),'measurements':rows}
    filename=f'runtime_{device}.json' if samples==20 else f'runtime_k{samples}_{device}.json'
    write_json(ROOT/'results'/filename,result);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','benchmark','all']);p.add_argument('--scene',choices=list(SCENES),default='eth');p.add_argument('--seed',type=int,default=11)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int);p.add_argument('--tag',default='');p.add_argument('--run');p.add_argument('--conditions',nargs='+');p.add_argument('--limit',type=int);p.add_argument('--baseline',action='store_true');p.add_argument('--no-export',action='store_true');p.add_argument('--samples',type=int,default=20,help='sampling budget for benchmark only')
    args=p.parse_args();cfg=json.loads((CODE_ROOT/'configs/legacy.json').read_text());torch.set_num_threads(cfg['cpu_threads'])
    if args.command=='benchmark':benchmark(cfg,args.device,args.samples)
    elif args.command=='train':train_one(args.scene,args.seed,cfg,args.device,args.epochs,args.tag)
    elif args.command=='evaluate':evaluate(args.run or ROOT/'runs'/f'{args.scene}_seed{args.seed}{args.tag}',args.conditions,args.device,not args.no_export,args.limit,args.baseline)
    else:
        for scene in SCENES:
            for seed in cfg['seeds']:
                run=ROOT/'runs'/f'{scene}_seed{seed}'
                expected_epochs=args.epochs or cfg['epochs']
                complete=False
                if (run/'training_summary.json').exists():
                    summary=json.loads((run/'training_summary.json').read_text());saved=json.loads((run/'config.json').read_text())
                    complete=(summary['epochs_completed']==expected_epochs and
                              all(saved.get(k)==v for k,v in cfg.items() if k!='epochs') and
                              saved['epochs']==expected_epochs and
                              saved['seed']==seed and saved['fold']==scene and
                              saved_data_fingerprint(saved)==protocol_fingerprint())
                if not complete:train_one(scene,seed,cfg,args.device,args.epochs)
                evaluate(run,device=args.device,export=not args.no_export)
                if seed==cfg['seeds'][0]:evaluate(run,['clean'],args.device,not args.no_export,baseline=True)

if __name__=='__main__':main()
