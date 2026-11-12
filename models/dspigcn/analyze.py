#!/usr/bin/env python3
"""Produce fold/seed summaries and auditable fixed/failure trajectory figures."""
import argparse
import json
import os
from operator import itemgetter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'li_junhao_trajectronpp'), str(ROOT/'li_junhao_trajectronpp'/'scripts')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'artifacts/dspigcn/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from independent_tpp.protocol import sha256, write_json
from independent_tpp.data import load_split
from dspigcn.data import GraphDataset
from analyze_independent_diagnostics import heading_rows, training_thresholds

FOLDS = ['eth','hotel','univ','zara1','zara2']
CONDITIONS = ['clean_0','noise_005','noise_010','noise_020','missing_1','missing_2']
METRICS = ['min_ade_mean','min_fde_mean','mean_ade_mean','mean_fde_mean','energy_score_mean','sample_diversity_mean']


def bundle(runs, fold, tag, seed=0):
    return runs/fold/f'seed_{seed}'/'evaluation'/tag/f'predictions_test_{tag}.npz'


def displacement_scores(samples, truth):
    """Select ADE and FDE winners per target, then average over targets."""
    offsets = np.subtract(samples, truth[:, None], dtype=np.float64)
    distance = np.hypot(offsets[..., 0], offsets[..., 1])
    ade = distance.mean(axis=2)
    fde = distance[..., -1]
    return dict(min_ade_mean=ade.min(axis=1).mean(), min_fde_mean=fde.min(axis=1).mean(),
                mean_ade_mean=ade.mean(axis=1).mean(), mean_fde_mean=fde.mean(axis=1).mean())


def trajectory(ax, data, index, title):
    observed, future, samples = (data[k][index] for k in ['observed','ground_truth','samples'])
    mask = data['observed_mask'][index]
    ax.plot(observed[mask,0], observed[mask,1], 'o-', color='#27485e',ms=3,label='Observed')
    if not mask.all():
        suffix = np.vstack((observed[mask][-1],observed[~mask]))
        ax.plot(suffix[:,0],suffix[:,1],'o--',color='#27485e',mfc='white',ms=4,label='Causal fill')
    truth = np.vstack((observed[-1],future))
    ax.plot(truth[:,0],truth[:,1],color='#158572',lw=2,label='Future')
    for k, sample in enumerate(samples):
        path = np.vstack((observed[-1],sample))
        ax.plot(path[:,0],path[:,1],color='#c58a33',alpha=0.18,lw=0.8,label='20 samples' if k==0 else None)
    mean = samples.mean(0)
    ax.plot(mean[:,0],mean[:,1],color='#c05b36',lw=1.5,label='Sample mean')
    ax.set_title(title,fontsize=10)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=0.15)
    return samples, future


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
    examples, fixed_conditions = [], []
    for fold in FOLDS:
        context = GraphDataset(load_split(fold,'test',str(ROOT/'data/causal_v2')))
        with np.load(bundle(args.runs,fold,'clean_0'),allow_pickle=False) as clean:
            records = [json.loads(str(x)) for x in clean['records']]
            indices = {r['window_id']:i for i,r in enumerate(records)}
            fixed_index = indices[fixed[fold][0]]
            errors = np.linalg.norm(clean['samples']-clean['ground_truth'][:,None],axis=-1).mean((1,2))
            worst = int(errors.argmax())
            fig,axes = plt.subplots(1,3,figsize=(10,3.6),layout='constrained',sharex=True,sharey=True)
            for ax,tag in zip(axes,['clean_0','noise_020','missing_2']):
                with np.load(bundle(args.runs,fold,tag),allow_pickle=False) as data:
                    paths, target = trajectory(ax, data, fixed_index, fold.upper()+' / '+tag.replace('_',' '))
                    fixed_conditions.append(dict(fold=fold, seed=0, condition=tag,
                                                 window_id=records[fixed_index]['window_id'],
                                                 **displacement_scores(paths[None], target[None])))
            legend = {}
            for ax in axes:
                handles,labels = ax.get_legend_handles_labels()
                legend.update(zip(labels,handles))
            fig.legend(legend.values(),legend.keys(),loc='outside lower center',ncol=5,fontsize=8)
            xlim,ylim = axes[0].get_xlim(),axes[0].get_ylim()
            height = np.clip(2.9*(ylim[1]-ylim[0])/max(xlim[1]-xlim[0],1e-6)+1.,2.3,4.8)
            fig.set_size_inches(10,float(height))
            fig.savefig(assets/f'fixed_{fold}.png',dpi=180,bbox_inches='tight',pad_inches=0.04);plt.close(fig)
            fig,ax = plt.subplots(figsize=(5,4),layout='constrained')
            trajectory(ax,clean,worst,f'{fold.upper()} largest meanADE: {errors[worst]:.3f} m')
            ax.legend(fontsize=8)
            fig.savefig(assets/f'failure_{fold}.png',dpi=180);plt.close(fig)
            for role,index in [('fixed',fixed_index),('largest_clean_mean_ade',worst)]:
                truth = np.vstack((clean['observed'][index,-1],clean['ground_truth'][index]))
                steps = np.diff(truth,axis=0)
                lengths = np.linalg.norm(steps,axis=-1)
                moving = (lengths[:-1]>1e-6)&(lengths[1:]>1e-6)
                cosine = (steps[:-1]*steps[1:]).sum(-1)/np.maximum(lengths[:-1]*lengths[1:],1e-12)
                turns = np.arccos(np.clip(cosine[moving],-1,1))
                examples.append(dict(fold=fold,seed=0,role=role,window_id=records[index]['window_id'],
                                     mean_ade=float(errors[index]),truth_travel_m=float(lengths.sum()),
                                     max_truth_heading_change_rad=float(turns.max()) if len(turns) else 0.,
                                     visible_neighbor_count=context.inputs(index)['positions'].shape[1]-1,
                                     prediction_sha256=sha256(bundle(args.runs,fold,'clean_0'))))
        context.context.cache_clear()
    write_json(evidence/'examples.json',dict(selection='Fixed: first locked runtime ID. Failure: argmax clean meanADE, seed 0, all samples shown.',examples=examples))
    # Motion metrics have training-only thresholds; retain each run/condition.
    pd.DataFrame(fixed_conditions).to_csv(evidence/'fixed_condition_scores.csv', index=False)
    diagnostics, spatial_diagnostics, review = [], [], []
    turns = {fold: training_thresholds(load_split(fold, 'train', str(ROOT/'data/causal_v2')))['turn_threshold_rad']
             for fold in FOLDS}
    metric_root = ROOT/'artifacts/dspigcn/metrics'
    for row in results.itertuples(index=False):
        with np.load(bundle(args.runs, row.fold, row.condition, row.seed), allow_pickle=False) as data:
            samples = data['samples']
            scores = displacement_scores(samples, data['ground_truth'])
            expected_scores = [getattr(row, key) for key in scores]
            actual_scores = np.array(list(scores.values()))
            np.testing.assert_allclose(actual_scores, expected_scores, rtol=0, atol=1e-7,
                                       err_msg=f'{row.fold}/{row.seed}/{row.condition}')
            heading = heading_rows({'samples': samples, 'observed': data['observed']}, None,
                                   turns[row.fold], row.fold, row.seed, row.condition)
            review.append(dict(fold=row.fold, seed=row.seed, condition=row.condition, windows=len(samples),
                               **scores, max_score_difference_m=np.abs(actual_scores - expected_scores).max(),
                               training_turn_p95_rad=turns[row.fold],
                               mean_abs_heading_change_rad=heading['turn_rate_mean_mean'],
                               windows_reaching_turn_threshold_fraction=heading['turn_violation_rate_mean']))
        path = metric_root/row.fold/f'seed_{row.seed}'/row.condition/'summary.json'
        if not path.exists():
            raise FileNotFoundError(f'Run make score first: {path}')
        values = json.loads(path.read_text())['overall']
        diagnostics.append(dict(fold=row.fold,seed=row.seed,condition=row.condition,**values))
        spatial_path = path.with_name('spatial_consistency.csv')
        try:
            spatial = pd.read_csv(spatial_path)
        except pd.errors.EmptyDataError:
            spatial = pd.DataFrame()
        spatial_diagnostics.append(dict(fold=row.fold,seed=row.seed,condition=row.condition,
            simultaneous_anchor_count=len(spatial),
            **{key:float(spatial[key].mean()) if len(spatial) else None for key in
               ['min_distance_mean','close_approach_rate']}))
    pd.DataFrame(diagnostics).to_csv(evidence/'diagnostics.csv',index=False)
    pd.DataFrame(spatial_diagnostics).to_csv(evidence/'spatial_diagnostics.csv',index=False)
    pd.DataFrame(review).to_csv(evidence/'review_metrics.csv', index=False)
    selections = []
    ranking = itemgetter('energy_score', 'mean_ade', 'epoch')
    for fold in FOLDS:
        for seed in range(3):
            run = args.runs/fold/f'seed_{seed}'
            selection = json.loads((run/'checkpoint_selection.json').read_text())
            history = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
            if [r['epoch'] for r in history]!=list(range(1,201)):
                raise ValueError(f'Incomplete or duplicate training history: {fold}/{seed}')
            selected = selection['selected']
            candidates = [r for r in history if 'energy_score' in r]
            if [r['epoch'] for r in candidates] != list(range(10, 201, 10)):
                raise ValueError(f'Unexpected clean-validation schedule: {fold}/{seed}')
            if ([ranking(r) for r in candidates] != [ranking(r) for r in selection['candidates']]
                    or ranking(selected) != min(map(ranking, candidates))):
                raise ValueError(f'Checkpoint selection differs from the training log: {fold}/{seed}')
            selections.append(dict(fold=fold,seed=seed,trained_epochs=len(history),
                selected_epoch=selected['epoch'],validation_energy_score=selected['energy_score'],
                validation_mean_ade=selected['mean_ade'],validation_candidates=len(selection['candidates']),
                epoch_seconds_total=sum(r['epoch_seconds'] for r in history),
                peak_cuda_bytes=max(r['peak_cuda_bytes'] or 0 for r in history),
                peak_host_rss_bytes=max(r['peak_host_rss_bytes'] for r in history),
                checkpoint_sha256=sha256(run/selected['checkpoint'])))
    pd.DataFrame(selections).to_csv(evidence/'checkpoint_selection.csv',index=False)
    print('ANALYSIS_OK',flush=True)


if __name__=='__main__':
    main()
