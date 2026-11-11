"""Exact device policy used in this study: CPU training, then MPS evaluation.

CPU-only fallback: python src/reproduce.py --evaluation-device cpu
"""
import argparse,json,torch
from data import CODE_ROOT,ROOT,SCENES,current_provenance,saved_data_fingerprint,protocol_fingerprint
from run import train_one,evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-device',choices=['cpu','mps'],default='mps');p.add_argument('--skip-evaluation',action='store_true');args=p.parse_args()
    cfg=json.loads((CODE_ROOT/'configs/legacy.json').read_text());torch.set_num_threads(cfg['cpu_threads']);current_provenance()
    for scene in SCENES:
        for seed in cfg['seeds']:
            run=ROOT/'runs'/f'{scene}_seed{seed}';complete=False
            if (run/'training_summary.json').exists():
                saved=json.loads((run/'config.json').read_text());summary=json.loads((run/'training_summary.json').read_text())
                complete=(summary['epochs_completed']==cfg['epochs'] and all(saved.get(k)==v for k,v in cfg.items())
                          and saved['fold']==scene and saved['seed']==seed and saved_data_fingerprint(saved)==protocol_fingerprint())
            if not complete:train_one(scene,seed,cfg,'cpu')
            if not args.skip_evaluation:
                evaluate(run,device=args.evaluation_device)
                if seed==cfg['seeds'][0]:evaluate(run,['clean'],'cpu',baseline=True)

if __name__=='__main__':main()
