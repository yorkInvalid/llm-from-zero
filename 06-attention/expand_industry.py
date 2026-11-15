"""Expand our corpus across four Chinese domains, keeping the tokenizer frozen."""
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from tokenizers import Tokenizer

ROOT=Path(__file__).parent/'artifacts/book-lm'
REVISION='249ea52f1ee412303e9741ae69fdd2109c0f8e1c'
SHARDS={
 'ai':'artificial_intelligence_machine_learning/chinese/high/rank_00012.parquet',
 'education':'subject_education_education/chinese/high/rank_00122.parquet',
 'science':'technology_scientific_research/chinese/high/rank_00127.parquet',
 'news':'news_media/chinese/high/rank_00082.parquet',
}

def canonical(text):
    return re.sub(r'[^\w\u4e00-\u9fff]','',text).lower()


def clean(text):
    if not isinstance(text,str):return None
    text=' '.join(text.split())
    if not 200<=len(text)<=20000 or '\ufffd' in text:return None
    if sum('\u4e00'<=c<='\u9fff' for c in text)/len(text)<.4:return None
    # Reject extreme sentence repetition, while retaining punctuation and document boundaries.
    sentences=[s for s in re.split(r'[。.!?！？]',text) if len(s)>15]
    if len(sentences)>8 and len(set(sentences))/len(sentences)<.6:return None
    return text


def main():
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    old=ROOT/'industry-data';out=ROOT/'industry-expanded-v2'
    if out.exists():raise ValueError('Use a new version; refusing to overwrite existing data')
    old_splits={n:json.loads((old/f'{n}.json').read_text()) for n in ('train','validation','test')}
    seen={canonical(d) for docs in old_splits.values() for d in docs}
    # Conservatively exclude any candidate sharing a 100-character passage with old holdouts.
    protected=set()
    for d in old_splits['validation']+old_splits['test']:
        c=canonical(d)
        protected.update(c[i:i+100] for i in range(max(0,len(c)-99)))
    train=list(old_splits['train']);validation=list(old_splits['validation'])
    manifest=[];fs=HfFileSystem();rng=random.Random(20261115)
    for domain,shard in SHARDS.items():
        print('Reading',domain,shard,flush=True)
        with fs.open(f'datasets/BAAI/IndustryCorpus2@{REVISION}/{shard}','rb') as stream:
            rows=pq.ParquetFile(stream).read_row_group(0,columns=['text']).to_pylist()
        rng.shuffle(rows)
        size=0;count=0;val_count=0;skipped=0
        for row in rows:
            d=clean(row['text'])
            if d is None:skipped+=1;continue
            c=canonical(d)
            if c in seen or any(c[i:i+100] in protected for i in range(0,len(c)-99,50)):
                skipped+=1;continue
            seen.add(c)
            if val_count<40:
                validation.append(d);val_count+=1
                protected.update(c[i:i+100] for i in range(len(c)-99))
                continue
            train.append(d);size+=len(d);count+=1
            if size>=2500000:break
        if size<2500000:raise ValueError(f'Insufficient filtered {domain} text: {size}')
        manifest.append(dict(domain=domain,shard=shard,row_group=0,train_documents=count,
                             train_characters=size,new_validation_documents=val_count,skipped=skipped))
        print(manifest[-1],flush=True)
    # Keep validation/test out of training, including conservative passage-overlap exclusion
    # against validation from domains visited later in the loop.
    filtered=[]
    for d in train:
        c=canonical(d)
        if not any(c[i:i+100] in protected for i in range(0,len(c)-99,50)):filtered.append(d)
    train=filtered;rng.shuffle(train)
    tokenizer=Tokenizer.from_file(str(old/'tokenizer.json'))
    splits={'train':train,'validation':validation,'test':old_splits['test'],
            'validation_legacy':old_splits['validation']}
    out.mkdir(parents=True)
    shutil.copyfile(old/'tokenizer.json',out/'tokenizer.json')
    stats={};hashes={}
    for name,docs in splits.items():
        p=out/f'{name}.json';p.write_text(json.dumps(docs,ensure_ascii=False))
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        stats[name]=dict(documents=len(docs),characters=sum(map(len,docs)),
                         target_tokens=sum(len(e.ids)+1 for e in tokenizer.encode_batch(docs)))
    meta=dict(version='industry-expanded-v2',dataset='BAAI/IndustryCorpus2',revision=REVISION,seed=20261115,
              sources=manifest,stats=stats,split_sha256=hashes,
              tokenizer_sha256=hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(),
              tokenizer='frozen original 8192 BPE, no refit',
              sampling='one row group per chosen high-quality Chinese shard; not representative of full corpus',
              cleaning='whitespace collapse, 200..20000 chars, >=40% CJK, reject replacement chars/extreme sentence repetition; canonical exact dedup; conservative 100-char passage exclusion against holdouts',
              validation='40 legacy AI docs + 40 newly reserved docs per domain; compare legacy validation separately',
              test='legacy holdout retained, never used for training or tuning')
    (out/'source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
