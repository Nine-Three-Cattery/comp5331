from pathlib import Path
import json,csv,sys,hashlib,statistics
import numpy as np
root=Path(__file__).resolve().parents[3];p=root/'submission/contributions/social_lstm';runs=root/'artifacts/social_lstm/formal/runs'
rows=list(csv.DictReader((p/'results.csv').open()))
folds=['eth','hotel','univ','zara1','zara2'];conditions=['clean_0','noise_005','noise_010','noise_020','missing_1','missing_2'];metrics=['mean_ade_mean','mean_fde_mean','energy_score_mean']
summary=[]
for tag in conditions:
 r={'condition':tag}
 for metric in metrics:
  seedmacros=[np.mean([float(x[metric]) for x in rows if x['condition']==tag and int(x['seed'])==seed]) for seed in range(3)]
  r[metric]=float(np.mean(seedmacros));r[metric+'_seed_sd']=float(np.std(seedmacros,ddof=1))
 summary.append(r)
(p/'evidence').mkdir(exist_ok=True)
with (p/'evidence/causal_macro_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
sys.path.insert(0,str(root/'li_junhao_trajectronpp'));sys.path.insert(0,str(root/'li_junhao_trajectronpp/scripts'))
from independent_tpp.data import load_split
from analyze_sample_budget import metrics_for_budget_batch
cv=[]
for fold in folds:
 store=load_split(fold,'test',str(root/'data/causal_v2'))
 obs=np.array([[store.state(r.scene_key,r.target_id,r.anchor+t)[:2] for t in range(-7,1)] for r in store.manifests],np.float32)
 truth=np.array([[store.state(r.scene_key,r.target_id,r.anchor+t)[:2] for t in range(1,13)] for r in store.manifests],np.float32)
 pred=obs[:,-1,None,:]+np.arange(1,13)[None,:,None]*(obs[:,-1]-obs[:,-2])[:,None,:]
 scores=metrics_for_budget_batch(pred[:,None].astype(np.float64),truth.astype(np.float64),1)
 cv.append({'fold':fold,'manifest_hash':store.identity['manifest_hash'],'window_count':len(obs),**{k:float(v.mean()) for k,v in scores.items()}})
(p/'evidence/causal_constant_velocity.json').write_text(json.dumps({'definition':'Last observed two-position displacement extrapolation; shared clean ordered test manifests; K1; common headline metric function','folds':cv,'macro':{k:float(np.mean([x[k] for x in cv])) for k in scores}},indent=2)+'\n')
runtime=[];training=[]
for fold in folds:
 for seed in range(3):
  r=runs/fold/f'seed_{seed}'
  runtime.append(json.loads((r/'runtime.json').read_text()));training.append({'fold':fold,'seed':seed,**json.loads((r/'training_summary.json').read_text())})
(p/'evidence/causal_runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
(p/'evidence/causal_training_summary.json').write_text(json.dumps(training,indent=2)+'\n')
print(json.dumps({'macros':summary,'cv_macro':json.loads((p/'evidence/causal_constant_velocity.json').read_text())['macro'],'training_seconds':sum(x['wall_seconds'] for x in training)},indent=2))
