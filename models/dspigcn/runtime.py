#!/usr/bin/env python3
"""Locked 128-target runtime measurement, independent of training batches."""
import argparse
import csv
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'li_junhao_trajectronpp'))
import torch
from independent_tpp.data import load_split
from independent_tpp.protocol import sha256, write_json
from dspigcn.data import GraphDataset, collate
from dspigcn.experiment import environment, load_model, sampling_seed
from dspigcn.model import anchor_sample


def synchronize(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


@torch.no_grad()
def measure(store, model, radius, seed, device, ids):
    dataset = GraphDataset(store, radius=radius)
    index_by_id = {r.window_id:i for i,r in enumerate(store.manifests)}
    indices = [index_by_id[x] for x in ids]
    if len(indices) != 128:
        raise ValueError('Runtime fixture must contain exactly 128 targets')
    measurements = []
    for repetition in range(13):
        # Include history context gathering every repetition, not a warmed cache.
        dataset.context.cache_clear()
        synchronize(device)
        start = time.perf_counter()
        prepared = []
        for offset in range(0,128,32):
            batch = collate([dataset.inputs(i) for i in indices[offset:offset+32]])
            prepared.append((batch['positions'].to(device),batch['mask'].to(device),
                             batch['origin'].to(device),batch['indices']))
        synchronize(device)
        ready = time.perf_counter()
        results = []
        for positions, mask, origin, batch_indices in prepared:
            result = anchor_sample(model(positions,mask),[sampling_seed(seed,i) for i in batch_indices])
            results.append(result+origin[:,None,None])
        synchronize(device)
        predicted = time.perf_counter()
        cpu = [result.cpu() for result in results]
        synchronize(device)
        finished = time.perf_counter()
        assert all(torch.isfinite(x).all() for x in cpu)
        if repetition >= 3:
            measurements.append(dict(repetition=repetition-3,preprocessing_ms=1000*(ready-start),
                                     prediction_sampling_ms=1000*(predicted-ready),
                                     end_to_end_ms=1000*(finished-start)))
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/causal_v2')
    parser.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    parser.add_argument('--folds',nargs='+',default=['eth','hotel','univ','zara1','zara2'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    fixture = ROOT/'benchmark/fixtures/runtime_windows.json'
    ids = json.loads(fixture.read_text())
    rows, raw = [], []
    for fold in args.folds:
        store = load_split(fold,'test',str(args.data_root))
        for seed in args.seeds:
            model, config, checkpoint = load_model(args.runs/fold/f'seed_{seed}',args.device)
            times = measure(store,model,config['radius'],seed,args.device,ids[fold])
            row = dict(fold=fold,seed=seed,checkpoint_sha256=sha256(checkpoint),windows=128,
                       batch_size=32,sample_count=20,warmups=3,repetitions=10)
            row.update({k:statistics.median(x[k] for x in times) for k in
                        ['preprocessing_ms','prediction_sampling_ms','end_to_end_ms']})
            rows.append(row)
            raw.append(dict(fold=fold,seed=seed,repetitions=times))
            print(row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    metadata = environment(args.device)
    if args.device == 'cuda':
        metadata['hardware'] = torch.cuda.get_device_name()
        metadata['driver'] = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
    else:
        metadata['hardware'] = platform.processor() or platform.machine()
    write_json(args.output.with_suffix('.json'),dict(environment=metadata,
        fixture_sha256=sha256(fixture),window_ids_by_fold={f:ids[f] for f in args.folds},
        boundary='In-memory neutral histories through graph construction/device transfer, coupled network+anchor sampling, metric CPU output. No disk/model loading. Cache cleared per repetition.',
        measurements=raw))


if __name__ == '__main__':
    main()
