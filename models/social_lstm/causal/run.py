"""Train, export and time the deterministic owner model using the shared contract.

Defaults frozen before the pilot: 12 epochs, Adam .001, batch32, hidden64,
embedding32, grid4, clean validation ADE with earlier-epoch tie-break.
"""
import argparse,copy,json,platform,random,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'li_junhao_trajectronpp'))
from causal.data import load
from causal.model import DeterministicSocialLSTM
from independent_tpp.protocol import CONDITIONS,FOLDS,SEEDS,VERSION,corruption_seed,digest_json,sha256,write_json

MODEL_CONFIG={'hidden':64,'embedding':32,'grid':4,'neighborhood':4.}

def source_identity():
    return digest_json({p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})

def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    elif str(device).startswith('mps'):torch.mps.synchronize()

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def environment(device):
    chip=platform.processor() or None
    if platform.system()=='Darwin':
        try:chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):pass
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'platform':platform.platform(),'device':str(device),'cpu_threads':torch.get_num_threads(),
        'hardware_chip':chip,'os':platform.system(),'os_release':platform.release(),
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        'reference_environment_match':platform.system()=='Linux' and platform.python_version().startswith('3.12.') and torch.__version__.split('+')[0]=='2.14.0'}

def chunks(indices,size):
    for start in range(0,len(indices),size):yield indices[start:start+size]

def training_batches(actor_counts,batch_size=32,bucket_size=256):
    """Seeded actor-count bucketing; every manifest index occurs exactly once."""
    ordered=np.argsort(actor_counts,kind='stable');batches=[]
    for bucket in chunks(ordered,bucket_size):
        shuffled=np.random.permutation(bucket)
        batches.extend(chunks(shuffled,batch_size))
    np.random.shuffle(batches)
    return batches

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

@torch.no_grad()
def validation(model,dataset,device,batch_size):
    model.eval();error=0.;final=0.;n=0
    for indices in chunks(np.arange(len(dataset)),batch_size):
        obs,valid,_=dataset.batch(indices,device)
        target=torch.as_tensor(dataset.future[indices],device=device)
        distances=torch.linalg.norm(model(obs,valid)[:,0]-target,dim=-1)
        error+=float(distances.sum());final+=float(distances[:,-1].sum());n+=len(indices)
    return {'mean_ade':error/(n*12),'mean_fde':final/n,'windows':n}

def train(fold,seed,runs,epochs=12,device='cpu',data_root=None,batch_size=32,stage='formal'):
    seed_everything(seed);run=Path(runs)/fold/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
    train_data=load(fold,'train',data_root);val_data=load(fold,'val',data_root)
    config={'protocol':VERSION,'model_name':'Social-LSTM-deterministic-causal-v2','fold':fold,'seed':int(seed),
        'sample_count':1,'model_config':MODEL_CONFIG,'epochs':int(epochs),'batch_size':int(batch_size),
        'optimizer':'Adam','learning_rate':.001,'weight_decay':.0001,'gradient_clip':10.,
        'loss':'mean squared metric future position, target only, full closed-loop rollout',
        'batching':'all indices once per epoch; sorted actor-count buckets of256, seeded shuffle within buckets and across batches',
        'selection':{'criterion':'clean validation mean ADE','tie_break':'earlier epoch','test_used_for_selection':False},
        'train_identity':dict(train_data.store.identity),'val_identity':dict(val_data.store.identity),
        'source_fingerprint':source_identity(),'stage':stage,'environment':environment(device),
        'baseline':'fixed mean of last three observed displacement intervals; no sampled residual feedback',
        'neighbors':'observation-complete; within6m at any observed time after target perturbation; future availability unused'}
    config_hash=digest_json(config)
    if (run/'checkpoint_selection.json').exists():
        prior=json.loads((run/'config.json').read_text());selection=json.loads((run/'checkpoint_selection.json').read_text())
        if prior!=config:raise ValueError(f'Refusing to overwrite a different completed configuration: {run}')
        checkpoint=run/selection['selected']['checkpoint']
        if not checkpoint.exists():raise ValueError('Completed run checkpoint is absent')
        print(f'CAUSAL_TRAIN_REUSED {fold} seed={seed}',flush=True);return run
    write_json(run/'config.json',config)
    model=DeterministicSocialLSTM(**MODEL_CONFIG).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    actor_counts=np.array([len(train_data.features(i)[0]) for i in range(len(train_data))])
    log=[];best=float('inf');selected=None;t0=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();start=time.perf_counter();sum_loss=0.;n=0
        for indices in training_batches(actor_counts,batch_size):
            obs,valid,_=train_data.batch(indices,device)
            truth=torch.as_tensor(train_data.future[indices],device=device)
            optimizer.zero_grad(set_to_none=True);prediction=model(obs,valid)[:,0]
            loss=(prediction-truth).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            sum_loss+=float(loss.detach())*len(indices);n+=len(indices)
        score=validation(model,val_data,device,batch_size);synchronize(device)
        row={'epoch':epoch,'train_mse':sum_loss/n,**score,'epoch_seconds':time.perf_counter()-start}
        log.append(row);write_json(run/'training_log.json',log)
        if score['mean_ade']<best:
            best=score['mean_ade'];selected={'checkpoint':'best.pt','epoch':epoch,**score}
            torch.save({'model_state':cpu_state(model),'model_config':MODEL_CONFIG,'protocol_version':VERSION,
                'fold':fold,'seed':seed,'epoch':epoch,'config_hash':config_hash,
                'source_fingerprint':source_identity()},run/'best.pt')
        print(json.dumps({'fold':fold,'seed':seed,**row}),flush=True)
    write_json(run/'checkpoint_selection.json',{'selection_split':'val','test_used_for_selection':False,
        'criterion':'minimum clean validation mean ADE','tie_break':'earlier epoch','selected':selected,'candidates':log})
    write_json(run/'training_summary.json',{'completed_epochs':epochs,'wall_seconds':time.perf_counter()-t0,
        'selected_epoch':selected['epoch'],'selected_val_ade':selected['mean_ade'],'environment':environment(device),
        'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024),
        'memory_boundary':'process-lifetime ru_maxrss, Darwin bytes / Linux KiB normalized to MiB; not GPU peak',
        'train_actor_count':{'median':float(np.median(actor_counts)),'p95':float(np.quantile(actor_counts,.95)),'max':int(actor_counts.max())}})
    return run

def restore(run,dataset,device):
    run=Path(run);config=json.loads((run/'config.json').read_text());selection=json.loads((run/'checkpoint_selection.json').read_text())
    if selection.get('selection_split')!='val' or selection.get('test_used_for_selection') is not False:raise ValueError('Checkpoint was not validation-selected')
    filename=selection['selected']['checkpoint']
    if Path(filename).name!=filename:raise ValueError('Selected checkpoint must be a local filename')
    checkpoint=torch.load(run/filename,map_location='cpu',weights_only=True)
    if checkpoint['protocol_version']!=VERSION or checkpoint['config_hash']!=digest_json(config):raise ValueError('Checkpoint config/protocol mismatch')
    if (checkpoint['fold'],checkpoint['seed'])!=(config['fold'],config['seed']):raise ValueError('Checkpoint run identity mismatch')
    if dataset.store.identity['fold']!=config['fold']:raise ValueError('Wrong holdout')
    split=dataset.store.identity['split']
    if split in ('train','val') and dataset.store.identity!=config[split+'_identity']:raise ValueError('Input identity changed')
    model=DeterministicSocialLSTM(**checkpoint['model_config']).to(device);model.load_state_dict(checkpoint['model_state']);model.eval()
    identity={'config_hash':digest_json(config),'checkpoint':filename,'checkpoint_sha256':sha256(run/filename)}
    return model,config,identity

@torch.no_grad()
def predict(run,data_root=None,device='cpu',batch_size=32,conditions=CONDITIONS,split='test'):
    run=Path(run);config=json.loads((run/'config.json').read_text());fold,seed=config['fold'],config['seed']
    dataset=load(fold,split,data_root);model,config,identity=restore(run,dataset,device)
    for tag,condition,level in conditions:
        samples=[];observed=[];masks=[];records=[];started=time.perf_counter()
        for indices in chunks(np.arange(len(dataset)),batch_size):
            obs,valid,mask=dataset.batch(indices,device,seed,condition,level)
            samples.append(model(obs,valid).cpu().numpy().astype(np.float32))
            observed.append(obs[:,0].cpu().numpy().astype(np.float32));masks.append(mask.astype(bool))
            for index in indices:
                row=copy.deepcopy(dict(dataset.store.manifests[int(index)].values))
                row.update({'target_id':dataset.store.manifests[int(index)].target_id,'fold':fold,'split':split,
                    'seed':int(seed),'sample_count':1,'sampling_seed':int(seed),'dt':.4,'coordinate_unit':'meter',
                    'condition':condition,'sigma':float(level) if condition=='noise' else 0.,
                    'missing':int(level) if condition=='missing' else 0,'model_name':'Social-LSTM-deterministic-causal-v2'})
                records.append(json.dumps(row,sort_keys=True))
        metadata={**dataset.store.identity,**identity,'seed':int(seed),'sample_count':1,
            'condition':condition,'level':level,'model_name':'Social-LSTM-deterministic-causal-v2',
            'source_fingerprint':config['source_fingerprint'],'deterministic':True}
        directory=run/'evaluation'/tag;directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'predictions_{split}_{tag}.npz'
        np.savez_compressed(path,samples=np.concatenate(samples),ground_truth=dataset.future.astype(np.float32),
            observed=np.concatenate(observed),observed_mask=np.concatenate(masks).astype(bool),
            records=np.array(records),metadata=np.array(json.dumps(metadata,sort_keys=True)))
        print(json.dumps({'status':'CAUSAL_PREDICTION_WRITTEN','fold':fold,'seed':seed,'condition':tag,
            'windows':len(dataset),'sample_count':1,'seconds':time.perf_counter()-started,'sha256':sha256(path)}),flush=True)

def runtime(run,data_root=None,device='cpu',batch_size=32):
    run=Path(run);config=json.loads((run/'config.json').read_text());fold=config['fold']
    dataset=load(fold,'test',data_root);model,_,identity=restore(run,dataset,device)
    ids=json.loads((ROOT/'benchmark/fixtures/runtime_windows.json').read_text())[fold]
    lookup={r.window_id:i for i,r in enumerate(dataset.store.manifests)}
    indices=np.array([lookup[x] for x in ids]);assert len(indices)==128 and batch_size==32
    measurements=[]
    for repetition in range(13):
        synchronize(device);total_start=time.perf_counter();preprocess=0.;inference=0.
        for batch_indices in chunks(indices,32):
            start=time.perf_counter();obs,valid,_=dataset.batch(batch_indices,'cpu');obs=obs.to(device);valid=valid.to(device);synchronize(device)
            preprocess+=time.perf_counter()-start;start=time.perf_counter()
            with torch.no_grad():pred=model(obs,valid)
            synchronize(device);inference+=time.perf_counter()-start;pred.cpu().numpy();synchronize(device)
        elapsed=time.perf_counter()-total_start
        if repetition>=3:measurements.append({'repetition':repetition-3,'preprocessing_seconds':preprocess,
            'prediction_including_sampling_seconds':inference,'end_to_end_seconds':elapsed})
    result={'fold':fold,'seed':config['seed'],'sample_count':1,'targets':128,'batch_size':32,'warmups':3,'measurements':10,
        'ordered_window_ids':ids,'environment':environment(device),**identity,
        'boundaries':'128 ordered neutral targets, four batches: history extraction/padding/transfers; deterministic joint rollout; metric CPU array output. Disk loading and scoring excluded.',
        'sampling_timing':'deterministic K1; sampling absent, no fabricated separate cost','repetitions':measurements}
    write_json(run/'runtime.json',result)
    print(json.dumps({'status':'CAUSAL_RUNTIME_WRITTEN','fold':fold,'seed':config['seed'],
        'median_end_to_end_seconds':float(np.median([x['end_to_end_seconds'] for x in measurements]))}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','predict','runtime','pilot','formal'])
    parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--fold',choices=FOLDS,default='eth');parser.add_argument('--seed',type=int,choices=SEEDS,default=0)
    parser.add_argument('--epochs',type=int);parser.add_argument('--device',default='cpu');parser.add_argument('--data-root',type=Path,default=ROOT/'data/causal_v2');parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if args.command in ('train','predict','runtime'):
        run=args.runs/args.fold/f'seed_{args.seed}'
        if args.command=='train':train(args.fold,args.seed,args.runs,args.epochs or 12,args.device,args.data_root)
        elif args.command=='predict':predict(run,args.data_root,args.device)
        else:runtime(run,args.data_root,args.device)
    else:
        epochs=args.epochs or (1 if args.command=='pilot' else 12)
        for fold in (('eth',) if args.command=='pilot' else FOLDS):
            for seed in ((0,) if args.command=='pilot' else SEEDS):
                run=train(fold,seed,args.runs,epochs,args.device,args.data_root,stage=args.command)
                predict(run,args.data_root,args.device)

if __name__=='__main__':main()
