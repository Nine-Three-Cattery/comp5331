from pathlib import Path
import json,csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3];p=root/'submission/contributions/social_lstm';ev=p/'evidence';runs=root/'artifacts/social_lstm/formal/runs'
sys.path.insert(0,str(root/'models/social_lstm/causal'));from data import load
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
ink='#172D3A';teal='#007465';rust='#AD482B';muted='#405864'
rows=list(csv.DictReader((p/'results.csv').open()));folds=['eth','hotel','univ','zara1','zara2']
cv=json.loads((ev/'causal_constant_velocity.json').read_text())
x=np.arange(5);means=[np.mean([float(r['mean_ade_mean']) for r in rows if r['fold']==f and r['condition']=='clean_0']) for f in folds]
sd=[np.std([float(r['mean_ade_mean']) for r in rows if r['fold']==f and r['condition']=='clean_0'],ddof=1) for f in folds]
fig,ax=plt.subplots(figsize=(7.3,3));ax.bar(x-.18,[r['mean_ade'] for r in cv['folds']],.36,label='Constant velocity (K=1)',color=muted);ax.bar(x+.18,means,.36,yerr=sd,label='Social-LSTM (K=1)',color=teal,capsize=3);ax.set_xticks(x,folds);ax.set_ylabel('Clean ADE (m)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(ev/'causal_clean_accuracy.pdf');plt.close(fig)
summary=list(csv.DictReader((ev/'causal_macro_summary.csv').open()));fig,ax=plt.subplots(figsize=(7.3,3));ax.bar(np.arange(6),[float(r['mean_ade_mean']) for r in summary],color=teal,yerr=[float(r['mean_ade_mean_seed_sd']) for r in summary],capsize=3);ax.set_xticks(np.arange(6),['Clean','Noise .05','Noise .10','Noise .20','Missing 1','Missing 2']);ax.set_ylabel('Macro ADE (m)');fig.tight_layout();fig.savefig(ev/'causal_robustness.pdf');plt.close(fig)
# Select from fixed runtime examples using truth-defined motion only, never prediction error.
d=load('hotel','test');ids=json.loads((root/'benchmark/fixtures/runtime_windows.json').read_text())['hotel'];lookup={r.window_id:i for i,r in enumerate(d.store.manifests)}
def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-12),-1,1)))
h=d.histories;future=d.future;old=(h[:,-1]-h[:,-4])/3;new=(future[:,-1]-future[:,-4])/3;angles=angle(old,new);eligible=[lookup[w] for w in ids if np.linalg.norm(old[lookup[w]])/.4>=.3 and np.linalg.norm(new[lookup[w]])/.4>=.3]
i=max(eligible,key=lambda i:angles[i]) if eligible else lookup[ids[0]]
fig,axs=plt.subplots(1,3,figsize=(9.8,3.9),sharex=True,sharey=True); points=[]
for ax,tag in zip(axs,['clean_0','noise_020','missing_2']):
 z=np.load(runs/'hotel/seed_0/evaluation'/tag/f'predictions_test_{tag}.npz');obs=z['observed'][i];truth=z['ground_truth'][i];pred=z['samples'][i,0];points.extend([obs,truth,pred])
 ax.plot(obs[:,0],obs[:,1],'-o',color=ink,ms=3,label='Observed input');ax.plot(np.r_[h[i,-1,0],truth[:,0]],np.r_[h[i,-1,1],truth[:,1]],'-o',color=teal,ms=3,label='True future');ax.plot(np.r_[obs[-1,0],pred[:,0]],np.r_[obs[-1,1],pred[:,1]],'-o',color=rust,ms=3,label='Prediction (K=1)');ax.set_title({'clean_0':'Clean','noise_020':'Noise 0.20 m','missing_2':'Missing 2'}[tag]);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x (m)');ax.grid(alpha=.2)
allp=np.concatenate(points); ymin,ymax=allp[:,1].min()-.2,allp[:,1].max()+.2; xc=(allp[:,0].min()+allp[:,0].max())/2; span=max(float(np.ptp(allp[:,0]))+.5,(ymax-ymin)*.7)
for ax in axs:ax.set_xlim(xc-span/2,xc+span/2);ax.set_ylim(ymin,ymax)
axs[0].set_ylabel('y (m)');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=10,frameon=False,bbox_to_anchor=(.5,.01));fig.suptitle('Hotel, seed 0: selected turning example',fontsize=11);fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(ev/'causal_turning.png',dpi=180);fig.savefig(ev/'causal_turning.pdf');plt.close(fig)
(ev/'causal_qualitative_selection.json').write_text(json.dumps({'fold':'hotel','seed':0,'window_id':d.store.manifests[i].window_id,'manifest_index':i,'turn_degrees':float(angles[i]),'rule':('Largest observed-to-late-future direction change among fixed runtime128 IDs with both speeds >= .3m/s.' if eligible else 'No speed-eligible case; first fixed runtime ID.')+' Uses truth only for descriptive example selection, never checkpoint selection. Not representative.'},indent=2)+'\n')
strata=[]
for f in folds:
 d=load(f,'test');obs=d.histories;old=(obs[:,-1]-obs[:,-4])/3;new=(d.future[:,-1]-d.future[:,-4])/3;turn=(angle(old,new)>=30)&(np.linalg.norm(old,axis=-1)/.4>=.3)&(np.linalg.norm(new,axis=-1)/.4>=.3);crowd=np.array([len(d.features(i)[0])-1 for i in range(len(d))])>=5
 for seed in range(3):
  z=np.load(runs/f/f'seed_{seed}/evaluation/clean_0/predictions_test_clean_0.npz');ade=np.linalg.norm(z['samples'][:,0]-z['ground_truth'],axis=-1).mean(1)
  for label,mask in [('turning',turn),('other_motion',~turn),('five_plus_neighbors',crowd),('fewer_neighbors',~crowd)]:strata.append({'fold':f,'seed':seed,'group':label,'count':int(mask.sum()),'ade_mean':float(ade[mask].mean()) if mask.any() else None})
(ev/'causal_motion_strata.json').write_text(json.dumps({'definitions':{'turning':'Observed last3-interval mean and future final3-interval mean directions differ >=30deg, both speeds >=.3m/s','five_plus_neighbors':'>=5 observation-complete actors within6m at any history step; group membership history-only','interpretation':'Descriptive unpaired subsets, different scene and trajectory composition; not causal attribution of interaction or turning effects.'},'rows':strata},indent=2)+'\n')
print('Figures and crowd/turn diagnostics generated')
