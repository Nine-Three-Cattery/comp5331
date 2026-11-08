#!/usr/bin/env python3
"""Execute in an extracted archive, not over the development checkout."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-python',default=sys.executable,
                        help='Interpreter matching the saved training/inference environment')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    ledger = json.loads((ROOT/'release_manifest.json').read_text())
    for name,expected in ledger.items():
        path = ROOT/name
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Invalid ledger path')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream,'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Hash mismatch: {name}')
    commands = [
        [sys.executable,'-m','pytest','models/dspigcn/own_unit_tests','-q'],
        [sys.executable,'benchmark/setup.py'],
        [sys.executable,'benchmark/validate.py','dspigcn','--runs','artifacts/dspigcn/formal/runs'],
    ]
    for command in commands:
        subprocess.run(command,cwd=ROOT,check=True)
    # Replay every archived checkpoint into a separate output tree. Validate all
    # 90 bundles, not just a pilot or a convenient holdout. Originals stay intact.
    reference = ROOT/'artifacts/dspigcn/formal/runs'
    with tempfile.TemporaryDirectory(prefix='dspigcn-replay-') as scratch:
        replay_root = Path(scratch)/'runs'
        for config in reference.glob('*/seed_*/config.json'):
            source = config.parent
            destination = replay_root/source.relative_to(reference)
            destination.mkdir(parents=True)
            selected = json.loads((source/'checkpoint_selection.json').read_text())['selected']['checkpoint']
            for name in ['config.json','checkpoint_selection.json',selected]:
                shutil.copyfile(source/name,destination/name)
        subprocess.run([args.prediction_python,str(ROOT/'models/dspigcn/run.py'),'predict',
                        '--runs',str(replay_root),'--device',args.device],cwd=ROOT,check=True)
        originals = sorted(reference.glob('*/seed_*/evaluation/*/*.npz'))
        if len(originals)!=90:
            raise ValueError('Expected all ninety original prediction bundles')
        for original in originals:
            replay = replay_root/original.relative_to(reference)
            with np.load(original,allow_pickle=False) as expected, np.load(replay,allow_pickle=False) as actual:
                np.testing.assert_allclose(actual['samples'],expected['samples'],rtol=1e-5,atol=1e-5)
                for key in ['ground_truth','observed','observed_mask','records','metadata']:
                    np.testing.assert_array_equal(actual[key],expected[key])
        print('CHECKPOINT_REPLAY_OK checkpoints=15 prediction_bundles=90')
    print('EXTRACTED_RELEASE_VERIFIED',len(ledger),'files')


if __name__=='__main__':
    main()
