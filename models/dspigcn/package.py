#!/usr/bin/env python3
"""Package only audited, completed runs; record hashes for independent replay."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'li_junhao_trajectronpp'))
sys.path.insert(0,str(ROOT/'benchmark'))
from contract import fingerprint
from independent_tpp.protocol import sha256, write_json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,default=ROOT/'artifacts/dspigcn/formal/runs')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/dspigcn/dspigcn-causal-v2.tar.gz')
    args = parser.parse_args()
    contribution = ROOT/'submission/contributions/dspigcn'
    audit = json.loads((ROOT/'artifacts/dspigcn/benchmark_audit.json').read_text())
    if (audit['model']!='dspigcn' or audit['protocol']!='causal_v2' or audit['status']!='passed'
            or audit['prediction_bundles']!=90 or audit['contract_sha256']!=fingerprint()
            or audit['results_sha256']!=sha256(contribution/'results.csv')):
        raise ValueError('Run the formal shared audit before packaging')
    registered = json.loads((ROOT/'models/dspigcn/configs/formal.json').read_text())
    for config_path in args.runs.glob('*/seed_*/config.json'):
        config = json.loads(config_path.read_text())
        if any(config[key]!=value for key,value in registered.items()):
            raise ValueError(f'Run differs from the registered formal config: {config_path}')
        for name,digest in config['implementation_hashes'].items():
            if sha256(ROOT/'models/dspigcn/dspigcn'/name)!=digest:
                raise ValueError(f'Implementation changed after training: {name}')
        last = config_path.parent/'training.jsonl'
        final = json.loads(last.read_text().splitlines()[-1])
        if final['epoch'] != config['epochs'] or config['epochs'] != 200:
            raise ValueError('Formal runs must complete their registered 200 epochs')
    if len(list(args.runs.glob('*/seed_*/config.json')))!=15:
        raise ValueError('Expected fifteen formal runs')
    write_json(contribution/'benchmark_audit.json',audit)
    source_dirs = ['models/dspigcn','benchmark','li_junhao_trajectronpp/independent_tpp',
                   'li_junhao_trajectronpp/scripts','li_junhao_trajectronpp/configs',
                   'submission','data/causal_v2','data/eth_ucy_raw','data/benchmark_thresholds']
    files = []
    for directory in source_dirs:
        for path in (ROOT/directory).rglob('*'):
            if (not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts
                    or 'build' in path.parts or path.name.startswith('._')
                    or path==contribution/'provenance.json'):
                continue
            files.append((path,path.relative_to(ROOT).as_posix()))
    for path in args.runs.rglob('*'):
        if path.is_file() and path.name!='last.pt' and not path.name.startswith('._'):
            files.append((path,'artifacts/dspigcn/formal/runs/'+path.relative_to(args.runs).as_posix()))
    for name in ['runtime.csv','runtime.json']:
        path = args.runs.parent/name
        files.append((path,'artifacts/dspigcn/formal/'+name))
    for name in ['Makefile','GROUP7_COMMON_DATA_EVALUATION_SPEC.md']:
        files.append((ROOT/name,name))
    ledger = {name:sha256(path) for path,name in files}
    ledger_path = args.runs.parent/'release_manifest.json'
    write_json(ledger_path,ledger)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'w:gz') as archive:
        for path,name in sorted(files,key=lambda pair:pair[1]):
            archive.add(path,arcname=name,recursive=False)
        archive.add(ledger_path,arcname='release_manifest.json')
    runtime = pd.read_csv(args.runs.parent/'runtime.csv')
    if len(runtime)!=15:
        raise ValueError('Expected runtime measurements for all fifteen runs')
    write_json(contribution/'provenance.json',dict(protocol='causal_v2',runs=15,prediction_bundles=90,
        artifact=dict(name=args.output.name,sha256=sha256(args.output)),
        evidence={name:sha256(contribution/name) for name in ['results.csv','benchmark_audit.json']},
        runtime_ms={key:float(runtime[key].median()) for key in ['preprocessing_ms','prediction_sampling_ms','end_to_end_ms']}))
    print('PACKAGE_OK',args.output.name,sha256(args.output),flush=True)


if __name__=='__main__':
    main()
