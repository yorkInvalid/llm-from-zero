"""Independent checks of both real token budgets, pair semantics, and benchmark overlap."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
from difflib import SequenceMatcher
import unicodedata
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.open()]
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'[\W_]+','',unicodedata.normalize('NFKC',s).casefold())
def grams(s,n):return {s[i:i+n] for i in range(len(s)-n+1)}


def audit():
    control_path=WORK/'nine-data/train.jsonl'; treatment_path=ROOT/'paired.jsonl'
    control=lines(control_path); treatment=lines(treatment_path)
    validation=lines(WORK/'nine-data/validation.jsonl')
    tok=Tokenizer.from_file(str(WORK/'seven-weights/metadata/tokenizer.json'))
    def encode(r):
        i=len(tok.encode('<|im_start|>user\n'+r['prompt']+'<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False).ids)
        s=len(tok.encode(r['response'],add_special_tokens=False).ids)+1
        assert [i,s,i+s]==[r[k] for k in ['input_tokens','supervised_tokens','total_tokens']]
        assert i+s <=384
        return i,s
    assert tok.token_to_id('<|endoftext|>')==151643
    by={r['id']:r for r in treatment};cb={r['id']:r for r in control}
    assert len(cb)==len(by)==9000 and list(cb)==list(by)
    for a,b in zip(control,treatment):assert encode(a)[1]==encode(b)[1]
    for r in validation:encode(r)
    pairs=lines(ROOT/'pairs.jsonl'); changed={q for x in pairs for q in [x['source_id'],x['replaced_id']]}
    assert len(changed)==2*len(pairs)
    assert len({norm(p['core']) for p in pairs})==len(pairs)
    assert len({norm(r['prompt']) for r in treatment})==len(treatment)
    assert {r['id'] for r in treatment if r!=cb[r['id']]}==changed
    nmode=Counter(); family=Counter(); answers=[]
    for pair in pairs:
        a,b=(by[pair[t]] for t in ['source_id','replaced_id'])
        old=cb[a['id']]
        assert a['response']==old['response']
        assert a['source_id']==b['source_id']==pair['source_id']
        assert a['source']==b['source']==old['source']==pair['source']
        assert a['source_file']==b['source_file']==old['source_file']
        assert a['source_line']==b['source_line']==old['source_line']
        assert {a['pair_mode'],b['pair_mode']}=={'short','reason'}
        assert all(r['prompt'].startswith(pair['core']+'\n\n') for r in (a,b))
        assert a['prompt']!=b['prompt']
        s,r=(a,b) if a['pair_mode']=='short' else (b,a)
        assert len(s['response'])<len(r['response'])
        if pair['family']=='来源单选解析':
            gold=pair['gold']; assert re.search(r'(?<![A-Z])[A-D](?:项|选项)?[。．.!！\s]*$',r['response'])
            g=list(re.finditer(r'(?<![A-Za-z0-9])([A-D])[.．、:：]\s*',pair['core']))
            assert [m[1] for m in g]==list('ABCD')
            oi=ord(gold)-65
            text=pair['core'][g[oi].end():(g[oi+1].start() if oi<3 else None)].strip()
            assert s['response'] in {gold,text,gold+'. '+text}
        elif pair['family']!='四则运算核算':
            assert r['response'].startswith(s['response']+'\n') and r['response'].endswith(pair['evidence'])
        family[pair['family']]+=1
        nmode['short_targets']+=s['supervised_tokens'];nmode['reason_targets']+=r['supervised_tokens']
        answers.append(s['supervised_tokens']-1)
    schedule=read(WORK/'pair-train/schedule.json')
    quarters=[]
    for i,q in enumerate(schedule['quarters']):
        before=[cb[t] for t in q['ids']];after=[by[t] for t in q['ids']]
        assert len(after)==len({r['id'] for r in after})==2250
        totals=lambda rs:[sum(x[k] for x in rs) for k in ['input_tokens','supervised_tokens','total_tokens']]
        assert totals(before)==totals(after)
        quarters.append(dict(quarter=i+1,both_arms_input_supervised_total=totals(after)))
    eq=WORK/'pair-eval/items.jsonl'; cases=lines(eq)
    assert len({r['id'] for r in cases})==len(cases)==60
    queries={r['id']:norm(r['stem']) for r in cases}; index=defaultdict(list)
    qg={uid:grams(q,4) for uid,q in queries.items()}
    for uid,gs in qg.items():
        for g in gs:index[g].append(uid)
    hits=[];maxcov=0;seen=set()
    for r in control+treatment+validation:
        for field in ['prompt','response']:
            s=norm(r[field])
            if s in seen:continue
            seen.add(s)
            shared=Counter(uid for gram in grams(s,4) for uid in index.get(gram,()))
            for uid,n in shared.items():
                q=queries[uid];cov=n/len(qg[uid]);maxcov=max(cov,maxcov)
                if n<4 and s!=q:continue
                block=SequenceMatcher(None,q,s,autojunk=False).find_longest_match()
                if s==q or block.size>=16 or cov>=.60 or (len(q)>=8 and q in s):
                    hits.append(dict(eval_id=uid,row_id=r['id'],field=field,overlap=q[block.a:block.a+block.size],coverage=cov))
    assert not hits,hits[:5]
    answers.sort()
    result=dict(control_sha256=hashfile(control_path),treatment_sha256=hashfile(treatment_path),validation_sha256=hashfile(WORK/'nine-data/validation.jsonl'),
                pair_eval_stems_sha256=hashfile(eq),pair_eval_prompts_sha256=hashfile(WORK/'pair-eval/prompts.jsonl'),
                validation_rows=120,pairs=len(pairs),rows_changed=len(changed),rows_unchanged=9000-len(changed),
                removed_content_ids=len(pairs),remaining_unique_original_examples=9000-len(pairs),full_treatment_prompts=len(set(norm(r['prompt']) for r in treatment)),
                families=dict(family),eight_quarters=quarters,paired_supervision=dict(nmode),short_answer_body_token_median=answers[len(answers)//2],short_answer_body_token_max=max(answers),
                short_answer_body_le16=sum(x<=16 for x in answers),overlap=dict(evaluation_cores=60,train_and_validation_distinct_normalized_strings=len(seen),flagged=len(hits),max_4gram_coverage=maxcov,
                          rule='NFKC/casefold/alphanumeric. Training and validation prompts and responses vs fresh evaluation stems. Flag identical, full stem >=8 characters, contiguous span >=16, or >=60% unique 4-gram coverage with >=4 matches.'))
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(result['pairs'],result['eight_quarters'],result['overlap'])


if __name__=='__main__':audit()
