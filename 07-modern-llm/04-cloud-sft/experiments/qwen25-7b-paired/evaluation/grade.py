"""Bind frozen prompts to real generations, assist blind judgment, and return paired scores."""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EPOCHS = [0.25,0.5,0.75,1,1.25,1.5,1.75,2]
MATRIX = [('base','base',0)] + [(arm,'epoch-'+str(epoch),epoch) for arm in ('control','paired') for epoch in EPOCHS]
FIELDS = {
    'S': ['result','format','overexplains'],
    'R': ['result','present','sufficient']
}


def lines(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_lines(path, data):
    with path.open('w',encoding='utf-8') as f:
        for obj in data: f.write(json.dumps(obj,ensure_ascii=False)+'\n')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(pid, raw):
    return pid + '_' + hashlib.sha256((pid+'\0'+raw.strip()).encode()).hexdigest()[:16]


def frozen():
    manifest=json.loads((ROOT/'freeze.json').read_text())
    for f,h in manifest['sha256'].items():
        if file_hash(ROOT/f)!=h:
            raise ValueError(('frozen file has changed',f))
    items={r['id']:r for r in lines(ROOT/'items.jsonl')}
    prompts={r['id']:r['prompt'] for r in lines(ROOT/'prompts.jsonl')}
    return items,prompts


def read_generations(run, prompts, partial):
    stages={}
    for arm,stage,epoch in MATRIX:
        path=run/('control' if arm=='base' else arm)/'generations'/(stage+'.jsonl')
        if partial and not path.is_file(): continue
        data={}
        for r in lines(path):
            pid=r['id']
            if pid not in prompts or pid in data or r['prompt']!=prompts[pid]:
                raise ValueError(('wrong, repeated or unknown prompt',str(path),pid))
            if not isinstance(r['text'],str):
                raise ValueError(('missing decoded output',pid))
            if r['generated_tokens']!=len(r['token_ids']):
                raise ValueError(('length mismatch',pid))
            data[pid]=r
        if data.keys()!=prompts.keys():
            raise ValueError(('incomplete stage',str(path)))
        stages[arm,stage,epoch]=data
    return stages


# Valid short Chinese answers that have unequivocal form and meaning.
# Other text goes to blind adjudication, not automatic rejection.
NAMES = {
    'P15': (['周三','星期三','礼拜三'], [p+d for p in ['周','星期','礼拜'] for d in '一二三四五六日天']),
    'P18': (['青'],list('青杏墨石')),
    'P25': (['宽盘'],['宽盘','窄杯']),
    'P28': (['小面'],['大面','小面']),
    'P30': (['不变'],['不变','变']),
    'P31': (['甲','甲抽屉'],['甲','乙','丙','甲抽屉','乙抽屉','丙抽屉']),
    'P34': (['丙','丙房'],['甲','乙','丙','甲房','乙房','丙房']),
    'P35': (['丙','丙项','任务丙'],['甲','乙','丙','甲项','乙项','丙项','任务甲','任务乙','任务丙']),
    'P38': (['谷安'],['谷安','林拓','辛宁']),
    'P39': (['灰线'],['灰线','白线','黄线']),
    'P41': (['摆渡','摆渡一次'],['摆渡','摆渡一次','讲解','讲解一次']),
    'P42': (['打开','开启','开着','开'],['打开','开启','开着','开','关闭','关着','关']),
    'P43': (['通常'],['通常','仅今天']),
    'P46': (['北厅'],[a+'厅' for a in ['东','西','南','北']]),
    'P48': (['符合'],['符合','不符合']),
    'P60': (['金','金色'],['金','金色','蓝','蓝色']),
}


def direct(item, text):
    """Return (answer correct, short form correct) for fully closed, unambiguous shapes."""
    t=text.strip()
    pid=item['id']
    if not t: return False,False
    if item['shape']=='choice':
        if re.fullmatch('[ABCD]',t):
            return t==item['correct_letter'], True
        for letter,option in item['options'].items():
            if t.rstrip('。')==option:
                return letter==item['correct_letter'],False
        return None
    if pid in NAMES:
        correct,valid=NAMES[pid]
        word=t[:-1] if t.endswith('。') else t
        if word in valid: return word in correct,True
        return None
    shape=item['short_shape']
    ref=item['reference']
    if shape=='一个阿拉伯整数':
        if re.fullmatch(r'[0-9]+',t): return int(t)==int(ref),True
    elif shape=='一个阿拉伯数值':
        if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?',t): return Decimal(t)==Decimal(ref),True
    elif shape=='一个百分数':
        if re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?[％%]',t):
            return Decimal(t[:-1])==Decimal(ref[:-1]),True
    elif 'HH:MM' in shape:
        if re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]',t): return t==ref,True
    elif pid=='P23':
        if re.fullmatch(r'[0-9]+\s*,\s*[0-9]+',t): return [int(n) for n in t.split(',')]==[2,5],True
    elif pid=='P49':
        mat=re.fullmatch(r'\(\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*\)',t)
        if mat:return tuple(map(int,mat.groups()))==(2,2),True
    elif pid=='P19':
        if re.fullmatch('[a-z]+',t): return t==ref,True
    elif pid=='P45':
        if t.lower() in {'borrow','lend'}:return t.lower()=='lend',True
    elif pid=='P24':
        if re.fullmatch('[0-9]+',t): return t==ref,True
    elif pid=='P56':
        if re.fullmatch('[PQR]',t): return t==ref,True
    elif pid=='P57':
        if re.fullmatch('[A-Z0-9]+',t):return t==ref,True
    elif pid=='P58':
        if re.fullmatch('[青白][0-9]',t):return t==ref,True
    return None


def auto(item, mode, text):
    if not text.strip():
        return [False,False,False]
    obj=direct(item,text)
    if obj is not None:
        correct,fmt=obj
        return [correct,fmt,False] if mode=='S' else [correct,False,False]
    # A bare prefix, a final punctuation mark or surrounding quotes on a single closed answer
    # contain no evidence-based explanation. A short format requiring a word accepts normal
    # Chinese sentence periods via direct() above; other added wrappers fail format.
    simple=re.fullmatch(r'(?:答案(?:是|为|[：:])|答[：:]|选择(?:是|为|[：:])?)?\s*[\"“‘\x27]?([^\r\n。！？.!?：:，,；;“”‘’\x27\"]{1,18})[\"”’\x27]?\s*[。.!！]?',text.strip())
    if simple:
        cleaned=simple.group(1).strip()
        v=direct(item,cleaned)
        if v is not None and cleaned!=text.strip():
            return [v[0],False,False]
    return [None,None,None]


def prepare(run, partial):
    items,prompts=frozen()
    models=read_generations(run,prompts,partial)
    records={}
    for data in models.values():
        for pid,row in data.items():
            mode=pid[-1]; item=items[pid[:-1]]; raw=row['text'].strip()
            bid=key(pid,raw)
            if bid in records:
                if records[bid]['text']!=raw: raise ValueError('digest collision')
                continue
            automatic=auto(item,mode,raw)
            records[bid]=dict(key=bid,id=pid,mode=mode,text=raw,auto=automatic)
    deck=sorted(records.values(),key=lambda v:v['key'])
    dest=ROOT/'review'
    dest.mkdir(exist_ok=True)
    write_lines(dest/'deck.jsonl',deck)
    pending=[r for r in deck if None in r['auto']]
    write_lines(dest/'pending.jsonl',pending)
    print(f'{len(models)} stages; {len(deck)} unique responses; {len(pending)} need blinded reading.')


def review_slice(start,end,mode,pending_path,decisions_dir):
    refs,_=frozen()
    by_id=defaultdict(list)
    deck=lines(pending_path.parent/'deck.jsonl')
    already=decisions(deck,decisions_dir) if decisions_dir.exists() else {}
    for rec in lines(pending_path):
        if rec['key'] in already: continue
        if start<=int(rec['id'][1:3])<=end and (mode=='both' or rec['mode']==mode):
            by_id[rec['id'][:3]].append(rec)
    for pid,records in sorted(by_id.items()):
        r=refs[pid]
        print('\n',pid,r['task_stem'], ' | ',r['short_shape'],r.get('correct_letter') or '',r['reference'], ' | ',r['rationale'])
        for rec in records:
            print(rec['key'][:4]+'_'+rec['key'].split('_')[1][:7],json.dumps(rec['text'],ensure_ascii=False))


def decisions(deck, decisions_dir):
    shortkey={r['key'].split('_')[0]+'_'+r['key'].split('_')[1][:7]:r['key'] for r in deck}
    if len(shortkey)!=len(deck): raise ValueError('Short key collision')
    verdict={}
    for path in sorted(decisions_dir.glob('*.tsv')):
        with path.open(encoding='utf-8') as f:
            for l in f:
                if not l.strip() or l.startswith('#'): continue
                k,bits,note=l.rstrip('\r\n').split('\t')
                bid=shortkey[k]
                if bid in verdict or not re.fullmatch('[01]{3}',bits) or not note:
                    raise ValueError((k,bits,'bad or duplicated judgment'))
                if k[3]=='R' and bits[2]=='1' and bits[1]=='0':
                    raise ValueError((k,'sufficient reason absent'))
                verdict[bid]=dict(bits=list(map(lambda c:c=='1',bits)),note=note)
    return verdict


def score(run, decisions_dir, partial):
    items,prompts=frozen()
    models=read_generations(run,prompts,partial)
    deck=lines(ROOT/'review/deck.jsonl')
    verdict=decisions(deck,decisions_dir)
    scored={}
    for rec in deck:
        bid=rec['key']
        if None in rec['auto']:
            if bid not in verdict: raise ValueError((bid,'missing blind judgment'))
            bits=verdict[bid]['bits']; note=verdict[bid]['note']; source='blind'
        else:
            bits=rec['auto']; note='无歧义的纯结果或带简单前后缀的结果'; source='automatic'
        scored[bid]=dict(rec,bits=bits,note=note,source=source)
    write_lines(ROOT/'review/audited.jsonl',list(scored.values()))
    output=[]
    for (arm,stage,epoch),records in models.items():
        for item in items.values():
            s=records[item['short_prompt_id']]; r=records[item['why_prompt_id']]
            sb=key(s['id'],s['text']); rb=key(r['id'],r['text'])
            sr,sf,soe=scored[sb]['bits'];wr,wp,ws=scored[rb]['bits']
            output.append(dict(arm=arm,stage=stage,epoch=epoch,split=item['split'],item_id=item['id'],domain=item['domain'],shape=item['shape'],choice=item['shape']=='choice',
                               short_result_correct=sr,short_format_ok=sf,short_overexplains=soe,why_result_correct=wr,why_reason_present=wp,reason_sufficient=ws,why_format_ok=bool(r['text'].strip()),
                               pair_pass=bool(sr and sf and wr and ws),short_tokens=s['generated_tokens']-int(s['ended_any_stop']),why_tokens=r['generated_tokens']-int(r['ended_any_stop']),
                               short_stopped=s['ended_any_stop'],why_stopped=r['ended_any_stop'],short_eos=s['ended_eos'],why_eos=r['ended_eos'],short_prompt_id=s['id'],why_prompt_id=r['id'],
                               short_text=s['text'],why_text=r['text'],short_blind=sb,why_blind=rb))
    target=ROOT/('paired-scores.partial.jsonl' if partial else 'paired-scores.jsonl')
    write_lines(target,output);print('Wrote',len(output),'pairs to',target)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','show','score'])
    p.add_argument('--run',type=Path,default=ROOT.parent/'pair-train/run')
    p.add_argument('--partial',action='store_true')
    p.add_argument('--decisions',type=Path,default=ROOT/'review/decisions')
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--end',type=int,default=60)
    p.add_argument('--mode',choices=['both','S','R'],default='both')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.run,a.partial)
    elif a.action=='score':score(a.run,a.decisions,a.partial)
    else:review_slice(a.start,a.end,a.mode,ROOT/'review/pending.jsonl',a.decisions)
