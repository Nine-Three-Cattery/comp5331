#!/usr/bin/env python3
"""Execute in an extracted archive, not over the development checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
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
    print('EXTRACTED_RELEASE_VERIFIED',len(ledger),'files')


if __name__=='__main__':
    main()
