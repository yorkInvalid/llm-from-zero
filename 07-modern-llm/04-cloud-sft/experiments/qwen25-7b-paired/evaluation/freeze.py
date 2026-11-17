import csv
import hashlib
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
# Each entry supplies three distractors, distinct from the answer in core.tsv.
CHOICES={
    3:['11','12','16'], 6:['4','8','10'],
    9:['1/4','1/2','1'], 12:['灰组','相同','缺两组总投票数，无法比较'],
    14:['14:45','15:25','15:45'], 17:['09:37','10:05','10:17'],
    20:['3','5','19'], 22:['on','mon','nome'],
    27:['小船较大','实心球较大','须给出排水量才能决定'], 29:['第一秒','一样','必须给出物体质量才能比较'],
    32:['能，领队肯定未签字','能，规定表明签字必然导致离场','能，留下意味着签字被撤回'], 36:['8','12','这三个都不能构成反例'],
    37:['能确定达标','能确定未达标','能确定本次验收已被取消'], 40:['拆老窗','挂画','三件同一时间完成'],
    44:['能，成年人都实际签过','能，条款要求所有成年人都签','能，因为小遥是这张表的签署人'], 47:['错过了','改乘了早一班船','取消了行程'],
    50:['北','南','东'], 54:['2','4','5'],
    55:['L','M','L和N'], 59:['亮灯','先亮后灭','先灭后亮'],
}


def dump_lines(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def main():
    with (ROOT/'core.tsv').open(encoding='utf-8') as f:
        core=list(csv.DictReader(f,delimiter='\t'))
    assert len(core)==60 and sorted(CHOICES)==list(CHOICES)
    rng=random.Random(318265)
    letters=list('ABCD')*5
    rng.shuffle(letters)
    records=[]; prompts=[]
    for group in range(10):
        subset=core[group*6:group*6+6]
        nums=[int(r['id'][1:]) for r in subset]
        choice=[n for n in nums if n in CHOICES]
        open_q=[n for n in nums if n not in CHOICES]
        assert len(choice)==2 and len(open_q)==4
        rng.shuffle(choice);rng.shuffle(open_q)
        dev={choice[0],*open_q[:2]}
        for n,item in zip(nums,subset):
            stem=item['stem']
            typ='open'; correct=None; option_map={}
            if n in CHOICES:
                typ='choice'; correct=letters.pop()
                incorrect=CHOICES[n].copy();rng.shuffle(incorrect)
                for label in 'ABCD':
                    option_map[label] = item['reference'] if label==correct else incorrect.pop()
                stem+='\n'+'；'.join(f'{key}. {value}' for key,value in option_map.items())+'。'
            short = '只给最后的答案，写'+(item['short_shape'] if typ=='open' else '一个大写选项字母')+'，不要说明理由。'
            why = '给出你的'+('答案' if typ=='open' else '选择')+'，并解释依据。说明为什么在这道题的具体条件下会得到这个结果。'
            sid=item['id']+'S'; rid=item['id']+'R'
            prompts.extend([{'id':sid,'prompt':stem+'\n'+short},{'id':rid,'prompt':stem+'\n'+why}])
            records.append(dict(item, task_stem=stem,shape=typ,options=option_map,correct_letter=correct,
                              split='dev' if n in dev else 'confirm',short_prompt_id=sid,why_prompt_id=rid))
    for split_name in ['dev','confirm']:
        subset=[r for r in records if r['split']==split_name]
        assert len(subset)==30 and sum(r['shape']=='choice' for r in subset)==10
    dump_lines(ROOT/'items.jsonl',records)
    # Deliberately mix S/R order; all stages receive identical order and prompt strings.
    rng.shuffle(prompts)
    dump_lines(ROOT/'prompts.jsonl',prompts)
    m={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['freeze.py','core.tsv','items.jsonl','prompts.jsonl','SCORING.md']}
    (ROOT/'freeze.json').write_text(json.dumps(dict(count_core=60,count_prompts=120,sha256=m),indent=2)+'\n')


if __name__=='__main__':
    main()
