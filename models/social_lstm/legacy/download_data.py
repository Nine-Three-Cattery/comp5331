"""Fetch only the locked legacy coordinate files; never download model code."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from data import CODE_ROOT,ROOT

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    release=json.loads((CODE_ROOT/'configs/raw_release.json').read_text());directory=ROOT/'data/raw'
    directory.mkdir(parents=True,exist_ok=True)
    for item in release['files']:
        path=directory/(item['recording']+'.txt')
        if not path.exists() and not args.verify_only:
            temporary=path.with_suffix('.download')
            subprocess.run(['curl','--fail','--location','--retry','2',item['url'],'--output',str(temporary)],check=True)
            if digest(temporary)!=item['sha256']:
                temporary.unlink();raise RuntimeError('Downloaded raw SHA256 mismatch: '+item['recording'])
            temporary.replace(path)
        if not path.exists() or digest(path)!=item['sha256']:
            raise RuntimeError('Missing or altered pinned raw file: '+item['recording'])
        print(item['recording']+' RAW_SHA256_OK')
    print('LEGACY_RAW_VERIFIED_NOT_CAUSAL_V2')
if __name__=='__main__':main()
