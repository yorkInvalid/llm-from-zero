"""Verify ModelScope's checksum and copy only expected experiment files into the workspace."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / 'Downloads'
LOCAL = ROOT / 'download-hashes.json'

def sha256(path):
    with path.open('rb') as infile:
        return hashlib.file_digest(infile, 'sha256').hexdigest()

def collect(filename):
    if not re.fullmatch(r'(?:control-measurements|pair-measurements|paired-generations-[246]|(?:paired|control)-epoch-(?:0\.25|0\.5|0\.75|1|1\.25|1\.5|1\.75|2))\.tar\.gz',filename):
        raise ValueError(filename)
    source=DOWNLOADS/filename
    sidecar=DOWNLOADS/(filename+'.sha256')
    expected=sidecar.read_text().split()[0]
    if not re.fullmatch('[0-9a-f]{64}',expected) or expected!=sha256(source):
        raise ValueError('checksum mismatch for '+filename)
    archive=tarfile.open(source,'r:gz')
    for member in archive.getmembers():
        path=PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError(member.name)
        if path.parts[0]=='run' and len(path.parts)>=2 and path.parts[1] in ('control','paired'):
            continue
        if path.as_posix() not in ('experiment.py','schedule.json','prompts.jsonl','run-pair.sh','pack.py'):
            raise ValueError(member.name)
    archive.extractall(ROOT,filter='data')
    archive.close()
    if '-epoch-' in filename:
        arm,stage=filename[:-7].split('-',1)
        checkpoint=ROOT/'run'/arm/stage
        state=json.loads((checkpoint/'state.json').read_text())
        if state['adapter_sha256']!=sha256(checkpoint/'adapter_model.safetensors'):
            raise ValueError('unpacked checkpoint mismatch')
        with zipfile.ZipFile(checkpoint/'training_state.pt') as z:
            if z.testzip() is not None:
                raise ValueError('invalid saved optimizer')
    seen=json.loads(LOCAL.read_text()) if LOCAL.exists() else {}
    seen[filename]={'sha256':expected,'bytes':source.stat().st_size,'verified':True}
    LOCAL.write_text(json.dumps(seen,indent=2,sort_keys=True)+'\n')
    print(filename,expected,source.stat().st_size,flush=True)

if __name__=='__main__':
    for filename in sys.argv[1:]:
        collect(filename)
