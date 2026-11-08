#!/usr/bin/env python3
"""Produce fold/seed summaries and auditable fixed/failure trajectory figures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'li_junhao_trajectronpp'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from independent_tpp.protocol import sha256, write_json

FOLDS = ['eth','hotel','univ','zara1','zara2']
CONDITIONS = ['clean_0','noise_005','noise_010','noise_020','missing_1','missing_2']
METRICS = ['min_ade_mean','min_fde_mean','mean_ade_mean','mean_fde_mean','energy_score_mean','sample_diversity_mean']


def bundle(runs, fold, tag):
    return runs/fold/'seed_0'/'evaluation'/tag/f'predictions_test_{tag}.npz'


def trajectory(ax, data, index, title):
    observed, future, samples = (data[k][index] for k in ['observed','ground_truth','samples'])
    ax.plot(observed[:,0], observed[:,1], 'o-', color='#27485e',ms=3,label='Observed')
    truth = np.vstack((observed[-1],future))
    ax.plot(truth[:,0],truth[:,1],color='#158572',lw=2,label='Future')
    for k, sample in enumerate(samples):
        path = np.vstack((observed[-1],sample))
        ax.plot(path[:,0],path[:,1],color='#c58a33',alpha=0.18,lw=0.8,label='20 samples' if k==0 else None)
    mean = samples.mean(0)
    ax.plot(mean[:,0],mean[:,1],color='#c05b36',lw=1.5,label='Sample mean')
    ax.set_title(title,fontsize=10)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal',adjustable='datalim')
    ax.grid(alpha=0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'submission/contributions/dspigcn')
    args = parser.parse_args()
    root = args.output
    evidence, assets = root/'evidence',root/'assets'
    evidence.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    results = pd.read_csv(root/'results.csv')
    expected = {(f,s,c) for f in FOLDS for s in range(3) for c in CONDITIONS}
    if len(results)!=90 or set(zip(results.fold,results.seed,results.condition))!=expected:
        raise ValueError('Analysis requires the complete 90-row matrix')
    by_fold = results.groupby(['condition','fold'])[METRICS].agg(['mean','std'])
    by_fold.to_csv(evidence/'per_fold_seed_mean_std.csv')
    # Equal holdout weights per seed, then seed mean/sample SD. Not window pooling.
    per_seed = results.groupby(['condition','seed'])[METRICS].mean()
    macro = per_seed.groupby('condition').agg(['mean','std'])
    macro.to_csv(evidence/'macro_mean_std.csv')
    base = results[results.condition=='clean_0'].set_index(['fold','seed'])
    paired = []
    for tag in CONDITIONS[1:]:
        condition = results[results.condition==tag].set_index(['fold','seed'])
        delta = condition[METRICS]-base[METRICS]
        delta['condition'] = tag
        paired.append(delta.reset_index())
    pd.concat(paired).to_csv(evidence/'paired_differences.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,metric,title in zip(axes,['mean_ade_mean','energy_score_mean'],['Typical-sample error','Ensemble Energy Score']):
        means = macro.loc[CONDITIONS,(metric,'mean')].to_numpy()
        sd = macro.loc[CONDITIONS,(metric,'std')].to_numpy()
        ax.bar(range(6),means,color=['#27485e']+['#3e8990']*3+['#c58a33']*2)
        ax.errorbar(range(6),means,yerr=sd,fmt='none',color='#222222',capsize=3)
        ax.set_xticks(range(6),['Clean','0.05 m','0.10 m','0.20 m','Miss 1','Miss 2'],rotation=25)
        ax.set_ylabel('meters');ax.set_title(title+' (K=20)')
    fig.savefig(assets/'robustness.png',dpi=180);plt.close(fig)
    fixed = json.loads((ROOT/'benchmark/fixtures/runtime_windows.json').read_text())
    examples = []
    for fold in FOLDS:
        with np.load(bundle(args.runs,fold,'clean_0'),allow_pickle=False) as clean:
            records = [json.loads(str(x)) for x in clean['records']]
            indices = {r['window_id']:i for i,r in enumerate(records)}
            fixed_index = indices[fixed[fold][0]]
            errors = np.linalg.norm(clean['samples']-clean['ground_truth'][:,None],axis=-1).mean((1,2))
            worst = int(errors.argmax())
            fig,axes = plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
            for ax,tag in zip(axes,['clean_0','noise_020','missing_2']):
                with np.load(bundle(args.runs,fold,tag),allow_pickle=False) as data:
                    trajectory(ax,data,fixed_index,fold.upper()+' / '+tag.replace('_',' '))
            axes[0].legend(fontsize=7,loc='best')
            fig.savefig(assets/f'fixed_{fold}.png',dpi=180);plt.close(fig)
            fig,ax = plt.subplots(figsize=(5,4),layout='constrained')
            trajectory(ax,clean,worst,f'{fold.upper()} largest meanADE: {errors[worst]:.3f} m')
            ax.legend(fontsize=8)
            fig.savefig(assets/f'failure_{fold}.png',dpi=180);plt.close(fig)
            for role,index in [('fixed',fixed_index),('largest_clean_mean_ade',worst)]:
                examples.append(dict(fold=fold,seed=0,role=role,window_id=records[index]['window_id'],
                                     mean_ade=float(errors[index]),prediction_sha256=sha256(bundle(args.runs,fold,'clean_0'))))
    write_json(evidence/'examples.json',dict(selection='Fixed: first locked runtime ID. Failure: argmax clean meanADE, seed 0, all samples shown.',examples=examples))
    # Motion metrics have training-only thresholds; retain each run/condition.
    diagnostics = []
    metric_root = ROOT/'artifacts/dspigcn/metrics'
    for row in results.itertuples(index=False):
        path = metric_root/row.fold/f'seed_{row.seed}'/row.condition/'summary.json'
        if not path.exists():
            raise FileNotFoundError(f'Run make score first: {path}')
        values = json.loads(path.read_text())['overall']
        diagnostics.append(dict(fold=row.fold,seed=row.seed,condition=row.condition,**values))
    pd.DataFrame(diagnostics).to_csv(evidence/'diagnostics.csv',index=False)
    print('ANALYSIS_OK',flush=True)


if __name__=='__main__':
    main()
