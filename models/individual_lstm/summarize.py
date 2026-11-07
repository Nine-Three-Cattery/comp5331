"""Produce owner evidence and tables after the independent group audit."""
from pathlib import Path
import json, shutil, hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'artifacts/individual_lstm'
C=ROOT/'submission/contributions/individual_lstm'
E=C/'evidence'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())

def main():
    receipt=read(A/'benchmark_audit.json')
    assert receipt['status']=='passed' and receipt['prediction_bundles']==90
    assert receipt['results_sha256']==sha(C/'results.csv')
    E.mkdir(exist_ok=True)
    shutil.copyfile(A/'benchmark_audit.json',C/'benchmark_audit.json')
    r=pd.read_csv(C/'results.csv')
    tpp_path=C.parent/'trajectronpp/results.csv'
    tpp_receipt=read(tpp_path.parent/'benchmark_audit.json')
    assert tpp_receipt['contract_sha256']==receipt['contract_sha256'] and tpp_receipt['results_sha256']==sha(tpp_path)
    t=pd.read_csv(tpp_path)
    keys=['fold','seed','condition','manifest_hash','window_count','protocol']
    assert len(r.merge(t,on=keys,validate='one_to_one'))==90
    cv=[]; timing=[]
    for p in sorted((A/'formal/runs').glob('*/seed_*')):
        cfg=read(p/'config.json'); cv+=read(p/'cv_supplement.json')
        timing.append(dict(fold=cfg['fold'],seed=cfg['seed'],**read(p/'runtime.json')['medians']))
    cv=pd.DataFrame(cv).rename(columns={'mean_ade':'mean_ade_mean','mean_fde':'mean_fde_mean',
        'min_ade':'min_ade_mean','min_fde':'min_fde_mean','energy_score':'energy_score_mean','sample_diversity':'sample_diversity_mean'})
    cv.to_csv(E/'cv_supplement.csv',index=False)
    pd.DataFrame(timing).to_csv(E/'runtime_cpu.csv',index=False)
    cols=['mean_ade_mean','mean_fde_mean','min_ade_mean','min_fde_mean','energy_score_mean','sample_diversity_mean']
    allrows=pd.concat([r.assign(model='Individual LSTM',k=1),cv.assign(model='CV',k=1),
                         t.assign(model='Trajectron++',k=20)],ignore_index=True)
    assert not allrows.duplicated(['model','fold','seed','condition']).any()
    per_fold=allrows.groupby(['model','k','condition','fold'],as_index=False)[cols].mean()
    per_fold.to_csv(E/'fold_means.csv',index=False)
    macro=per_fold.groupby(['model','k','condition'],as_index=False)[cols].mean()
    # Sample SD of the three per-seed, equal-fold macros, not independent-window uncertainty.
    sd=allrows.groupby(['model','condition','seed'])[cols].mean().groupby(['model','condition']).std().add_suffix('_seed_sd')
    macro=macro.join(sd,on=['model','condition'])
    macro.to_csv(E/'macro_comparison.csv',index=False)
    print(macro[['model','condition','mean_ade_mean','mean_fde_mean','min_ade_mean','energy_score_mean']].to_string(index=False))

if __name__=='__main__': main()
