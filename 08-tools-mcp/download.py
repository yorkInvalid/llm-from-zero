"""Fetch only the seven files used by Transformers. Verify the published digests."""
import hashlib
from pathlib import Path
import subprocess
import json
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent / 'weights'
FILES = ('config.json','generation_config.json','merges.txt','model.safetensors',
         'tokenizer.json','tokenizer_config.json','vocab.json')
MODEL = 'Qwen/Qwen3-0.6B'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ROOT.mkdir(exist_ok=True)
    with urlopen(f'https://modelscope.cn/api/v1/models/{MODEL}/repo/files?Revision=master&Recursive=true', timeout=30) as res:
        files = {v['Path']:v for v in json.load(res)['Data']['Files']}
    for name in FILES:
        meta = files[name]
        target = ROOT/name
        if target.exists() and digest(target) == meta['Sha256']:
            continue
        tmp = target.with_suffix('.partial')
        subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','20','-o',str(tmp),
            f'https://modelscope.cn/models/{MODEL}/resolve/{meta["Revision"]}/{name}'], check=True)
        if digest(tmp) != meta['Sha256']:
            raise ValueError(f'{name}: checksum mismatch')
        tmp.replace(target)
        print(f'已校验 {name}', flush=True)

if __name__ == '__main__':
    main()
