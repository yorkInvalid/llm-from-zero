"""Package the already-frozen treatment data as a portable delta against the control."""
import base64
from difflib import SequenceMatcher
import hashlib
import json
import lzma
from pathlib import Path
import zipfile


HERE=Path(__file__).resolve().parent
WORK=HERE.parent
def rows(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')]

def encode(value,refs):
    for i,source in enumerate(refs):
        if value==source:return i
    def pieces(chunk):
        if not chunk:return []
        for i,source in enumerate(refs):
            pos=source.find(chunk)
            if pos>=0 and len(chunk)>=4:return [[i,pos,len(chunk)]]
        i,m=max(((i,SequenceMatcher(None,chunk,source,autojunk=False).find_longest_match()) for i,source in enumerate(refs)),key=lambda x:x[1].size)
        if m.size<8:return [chunk]
        return pieces(chunk[:m.a])+[[i,m.b,m.size]]+pieces(chunk[m.a+m.size:])
    return pieces(value)


def main():
    baseline=WORK/'nine-data/train.jsonl'; final=WORK/'pair-data/paired.jsonl'
    base=rows(baseline); paired=rows(final)
    idx={r['id']:i for i,r in enumerate(base)}
    assert [r['id'] for r in base]==[r['id'] for r in paired]
    suffixes=[];records=[]
    for pair in rows(WORK/'pair-data/pairs.jsonl'):
        si=idx[pair['source_id']];di=idx[pair['replaced_id']]
        source=base[si];a,b=paired[si],paired[di]
        core=pair['core']; refs=(source['prompt'],source['response'])
        def suff(r):
            assert r['prompt'].startswith(core)
            suffix=r['prompt'][len(core):]
            if suffix not in suffixes:suffixes.append(suffix)
            return suffixes.index(suffix)
        assert a['response']==source['response']
        records.append([si,di,encode(core,refs),suff(a),suff(b),encode(b['response'],refs+(core,)),a['input_tokens'],b['input_tokens'],int(a['pair_mode']=='reason')])
    encoded=json.dumps([suffixes,records],ensure_ascii=False,separators=(',',':')).encode()
    compressed=lzma.compress(encoded,preset=9)
    code=(HERE/'rebuild_template.py').read_text()
    archive=HERE/'paired-patch.pyz'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('__main__.py',code.replace('PATCH_DATA','patch_bytes()'),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        z.writestr('delta.xz',compressed,compress_type=zipfile.ZIP_STORED)
    inline=HERE/'rebuild_paired.py'
    printable=base64.b85encode(compressed)
    body='base64.b85decode(\n'+'\n'.join('    '+repr(printable[i:i+100]) for i in range(0,len(printable),100))+'\n)'
    inline.write_text(code.replace('PATCH_DATA',body))
    print(archive,len(archive.read_bytes()),inline,len(inline.read_bytes()),'xz patch',len(compressed))

if __name__=='__main__':main()
