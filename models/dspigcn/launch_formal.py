#!/usr/bin/env python3
"""Run one process per seed after the Linux pilot gate, with explicit status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmark'),str(ROOT/'li_junhao_trajectronpp')]
from contract import fingerprint
from independent_tpp.protocol import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,default=ROOT/'artifacts/dspigcn/formal/runs')
    parser.add_argument('--pilot-receipt',type=Path,default=ROOT/'artifacts/dspigcn/pilot_audit.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = parser.parse_args()
    receipt = json.loads(args.pilot_receipt.read_text())
    if (receipt.get('model')!='dspigcn' or receipt.get('status')!='passed'
            or receipt.get('prediction_bundles')!=6 or receipt.get('contract_sha256')!=fingerprint()):
        raise ValueError('Complete the current shared pilot gate before formal training')
    output = args.runs.parent
    output.mkdir(parents=True,exist_ok=True)
    children,logs = {},[]
    try:
        for seed in range(3):
            log = (output/f'seed_{seed}.log').open('a')
            logs.append(log)
            command = [sys.executable,'-u',str(Path(__file__).with_name('run.py')),'matrix',
                       '--runs',str(args.runs),'--seeds',str(seed),'--device',args.device]
            children[seed] = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while True:
            status = {str(seed):dict(pid=p.pid,returncode=p.poll()) for seed,p in children.items()}
            write_json(output/'process_status.json',dict(updated_unix=time.time(),processes=status))
            if all(p.poll() is not None for p in children.values()):
                break
            time.sleep(10)
        if any(p.returncode for p in children.values()):
            raise SystemExit('One or more formal jobs failed; inspect seed logs and resume the exact command')
        print('FORMAL_TRAIN_PREDICT_COMPLETE',flush=True)
    finally:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()


if __name__=='__main__':
    main()
