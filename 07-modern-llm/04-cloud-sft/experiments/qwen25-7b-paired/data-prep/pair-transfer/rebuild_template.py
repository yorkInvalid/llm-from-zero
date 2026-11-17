"""Rebuild the exact 9k treatment dataset from the frozen 9k control; Python standard library."""
import argparse
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import sys
import tempfile
import zipfile

BASE='e314fe11f0a578009e67809ec7d67e3c72484756a25e6df50825eba8f41a279e'
TARGET='911843fd787d5366efc63fd73af290ca40f37cc12da6365ccca898b34afc6965'

def patch_bytes():
    with zipfile.ZipFile(sys.argv[0]) as bundle:
        return bundle.read('delta.xz')

def restore(value,refs):
    if isinstance(value,int):
        return refs[value]
    return ''.join(x if isinstance(x,str) else refs[x[0]][x[1]:x[1]+x[2]] for x in value)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control',type=Path,help='original train.jsonl with SHA-256 '+BASE)
    parser.add_argument('output',type=Path,help='write reconstructed paired train JSONL to this path')
    args=parser.parse_args()
    raw=args.control.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASE:
        parser.error('input does not match the frozen control SHA-256')
    if args.output.exists():
        if args.output.is_file() and hashlib.sha256(args.output.read_bytes()).hexdigest()==TARGET:
            print(TARGET,args.output,'(already verified)');return
        parser.error('output already exists with different contents; choose a new filename')
    lines=raw.splitlines(keepends=True)
    if len(lines)!=9000:raise ValueError('unexpected baseline row count')
    suffixes,records=json.loads(lzma.decompress(PATCH_DATA))
    for si,di,core_recipe,sa,sb,br,ai,bi,is_reason in records:
        source,donor=json.loads(lines[si]),json.loads(lines[di])
        refs=(source['prompt'],source['response'])
        core=restore(core_recipe,refs)
        other=restore(br,refs+(core,))
        for slot,original,suffix,answer,n,reason in [(si,source,sa,source['response'],ai,is_reason),(di,donor,sb,other,bi,not is_reason)]:
            row=dict(source)
            row.update(id=original['id'],prompt=core+suffixes[suffix],response=answer,input_tokens=n,
                       supervised_tokens=original['supervised_tokens'],total_tokens=n+original['supervised_tokens'],
                       baseline_id=original['id'],source_id=source['id'],pair_mode='reason' if reason else 'short')
            lines[slot]=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
    data=b''.join(lines)
    got=hashlib.sha256(data).hexdigest()
    if got!=TARGET:raise ValueError('reconstructed SHA-256 mismatch: '+got)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    part=None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent,prefix='.paired-incomplete-',delete=False) as f:
            part=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
        # Publish with a same-filesystem hard link, avoiding overwrites even on a concurrent retry.
        try: os.link(part,args.output)
        except FileExistsError:
            if hashlib.sha256(args.output.read_bytes()).hexdigest()!=TARGET:raise
    finally:
        if part is not None:part.unlink(missing_ok=True)
    print(got,args.output,len(lines),'rows',len(data),'bytes')

if __name__=='__main__':main()
