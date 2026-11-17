"""Reconstruct and SHA-verify the frozen pair data using a compact, lossless JSON edit stream."""
import hashlib
import json
from pathlib import Path
import sys
import lzma

CONTROL_SHA='e314fe11f0a578009e67809ec7d67e3c72484756a25e6df50825eba8f41a279e'
PAIRED_SHA='911843fd787d5366efc63fd73af290ca40f37cc12da6365ccca898b34afc6965'
def h(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def restore(control,patch,target):
    assert h(control)==CONTROL_SHA
    with control.open() as f:
        records=[json.loads(l) for l in f]
    out=[r.copy() for r in records]
    changes=json.loads(lzma.decompress(patch.read_bytes()))
    for i,source,items in changes:
        before=records[source]
        for key,operation,content in items:
            if operation==1:
                out[i][key]=before[content]
            elif operation==2:
                k,offset,size,head,tail=content
                out[i][key]=head+before[k][offset:offset+size]+tail
            else:
                out[i][key]=content
    temp=target.with_suffix('.reconstructing')
    with temp.open('w',encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    assert h(temp)==PAIRED_SHA
    temp.replace(target)
    print('VERIFIED',target,PAIRED_SHA,len(out))
if __name__=='__main__':
    restore(*(Path(s) for s in sys.argv[1:]))
