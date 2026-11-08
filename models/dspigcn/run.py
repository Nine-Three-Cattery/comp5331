#!/usr/bin/env python3
"""Run from the repository root; independent owner code + shared benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'li_junhao_trajectronpp'))

import torch
from dspigcn.experiment import predict, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict', 'matrix'])
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/causal_v2')
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/formal.json')
    parser.add_argument('--folds', nargs='+', default=['eth','hotel','univ','zara1','zara2'],
                        choices=['eth','hotel','univ','zara1','zara2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    for fold in args.folds:
        for seed in args.seeds:
            run = args.runs/fold/f'seed_{seed}'
            if args.command in ('train', 'matrix'):
                train(args.data_root, run, fold, seed, config, args.device)
            if args.command in ('predict', 'matrix'):
                predict(args.data_root, run, args.device)


if __name__ == '__main__':
    main()
