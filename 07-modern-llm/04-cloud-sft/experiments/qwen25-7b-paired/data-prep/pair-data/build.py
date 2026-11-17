"""Build counterfactual instruction pairs anchored to the 9k COIG-CQIA slice."""
import argparse
import sys
import collections
import datetime
import difflib
import hashlib
import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PARENT = WORK / "nine-data"
TOKEN = WORK / "seven-weights/metadata/tokenizer.json"
T = None
GROUPS = ("chinese_traditional/trad-multi-choice","exam/coig_exam","exam/kaoyan","exam/law_gee")
OPTS = re.compile(r"(?<![A-Za-z0-9])([A-D])[.．、:：]\s*")
FINAL = re.compile(r"(?:故(?:本题)?(?:正确)?(?:答案)?(?:应)?(?:选(?:择)?|为|是)|(?:因此|所以)?(?:本题)?(?:正确)?答案(?:为|是|选)|因此(?:本题)?(?:应)?选(?:择)?|所以(?:本题)?(?:应)?选(?:择)?|本题(?:应)?选(?:择)?|故选项)(?:项)?\s*[:：]?\s*([A-D])(?:项|选项)?[。．.!！\s]*$")
UNSAFE = re.compile(r'爆炸|炸弹|弹药|枪械|枪支|军火|恐怖袭击|杀人|强奸|自杀|投毒|下毒|炸药|毒剂|病毒|病原|细菌|疫苗|生物武器|基因改造|核武|原子弹|火箭弹|武器|入侵|漏洞|绕过|恶意软件|木马|黑客|反侦查|图中|下图|上图|图示|如图|漫画|图像|图甲|图乙|表中|如下图|如下表|图\s*[1-9一二三四]|见图|见表|附图|图所示|读图|读表|下表|上表|[\ufffd]|＜图')
TAIL = re.compile(r'^(?:[（(]?[0-9]+[)）][ .、])?\s*')
CONFLICT = re.compile(r'(?:只|仅|直接)(?:需要?|须|要)?(?:写|答|填|给|输|选|返回)|无需解释|不必解释|不[要用]解释|答案[填写]在|请说明理由|画出|材料[一二三四][:：]|第[一二三四]问')
NOOPTIONS = re.compile(r'以上(?:都|均|说法)|上述(?:都|均)|(?:A|B|C|D)[和与、](?:A|B|C|D)|^以上$')
BLANK = re.compile(r'__+|[（(]\s*[)）]')


def sha(b): return hashlib.sha256(b).hexdigest()
def rank(s): return sha(("pair-20261116-v1/"+s).encode())
def rows(p): return [json.loads(l) for l in p.open() if l.strip()]
def read(p): return json.loads(p.read_text())
def dump(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
def writelines(p,obj): p.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in obj))
def norm(s): return re.sub(r"[\W_]+","",s).casefold()
def il(s): return len(T.encode("<|im_start|>user\n"+s+"<|im_end|>\n<|im_start|>assistant\n",add_special_tokens=False).ids)
def sl(s): return len(T.encode(s,add_special_tokens=False).ids)+1
def prompt(core,mode): return core+"\n\n"+mode


@dataclass
class Variant:
    core: str
    source_id: str
    family: str
    key: str
    anchor_mode: str
    anchor_prompt: str
    anchor_answer: str
    other_prompt: str
    other_answer: str
    evidence: str
    def metrics(self):
        self.ai=il(self.anchor_prompt); self.asup=sl(self.anchor_answer)
        self.oi=il(self.other_prompt); self.osup=sl(self.other_answer)
        return self


def mc(r):
    if not r["source_file"].startswith(GROUPS):
        return []
    p,answer = r['prompt'],r['response']
    if "\n\n" in p:
        lead, body=p.split("\n\n",1)
        if (len(lead)<115 and not OPTS.search(lead) and
            re.match(r'^(?:请(?:从|选|根据|回答|阅读)|在每小题|[一二三]、|[每下]列各题|本题|在下列|选择题|单项选择题|以下各题|下列各题|单选题|对于下述|从下列|请你)',lead) and
            not re.search(r'[？?]|[《「“]',lead) and len(body)>24):
            p=body.strip()
    if UNSAFE.search(p+' '+answer) or CONFLICT.search(p) or ('多项选择' in p) or '多选题' in p: return []
    match=FINAL.search(answer)
    if not match: return []
    key=match[1]; rationale=answer[:match.start()].strip()
    if len(norm(rationale))<12 or not re.search('[\u4e00-\u9fff]',rationale): return []
    markers=list(OPTS.finditer(p))
    if [m[1] for m in markers]!=list("ABCD"):return []
    head=p[:markers[0].start()].strip()
    if len(norm(head))<6: return []
    if re.search(r'字形演变|演变过程|观察图|如[左右上下]图|据图|此图|看图|看漫画|图片中|所示|如右|如左|如上图|阴影部|四幅|图画|图内|示意图|加点|画线|划线|下划线|文中|原文|图示|根据图|图片|此图|该图|材料|文章中|前文|本文|篇章|选文|第[①②③④⑤⑥⑦⑧]段|选段|患者|处方|冠心病|心脏病|剪切线',head):return []
    if any(m[1]!=key for m in re.finditer(r'(?:故本题选|故选|因此答案为)[:： ]*([A-D])(?![A-D、])',rationale)):return []
    options={m[1]:p[m.end():(markers[i+1].start() if i<3 else len(p))].strip()
             for i,m in enumerate(markers)}
    if any(not s or '\n\n' in s or len(s)>180 for s in options.values()) or len(set(map(norm,options.values())))!=4:return []
    if any(c not in head for c in re.findall(r"[①②③④⑤⑥⑦⑧]",options[key])):return []
    if any(any(t in opt for t in ['答案：','答案:','解析：','请回答','故选','参考答案']) for opt in options.values()): return []
    lp=prompt(p,"给出所选答案，并说明判断依据。")
    result=[]
    for style,short,req in [("letter",key,"只答所选项的字母。"),
                            ("text",options[key],"只写所选项的原文，不加字母或解释。"),
                            ("labeled",key+". "+options[key],"只给所选项的字母和原文，以“字母. 原文”作答。")]:
        if len(short)>=len(answer) or sl(short)+3>=sl(answer):continue
        if style!="letter" and (NOOPTIONS.search(options[key]) or len(options[key])>118):continue
        result.append(Variant(p,r['id'],'来源单选解析',key,"reason",lp,answer,prompt(p,req),short,"来源答案的明确末尾单选标记；四个有序选项，理由原文保留。").metrics())
    return result


def algorithm(r):
    if r['source_file']!="coig_pc/coig_pc_core_sample.jsonl":return []
    inst,sep,inp=r['prompt'].partition('\n\n');answer=r['response']
    if not sep or not inp or UNSAFE.search(r['prompt']+answer):return []
    fam,why,core,reqs="","","",None
    minor=r['task_type']['minor']
    if minor==['文本分类','知识'] and re.fullmatch(r'\d{4}-\d\d-\d\d',inp) and re.fullmatch('星期[一二三四五六日天]',answer):
        d=datetime.date.fromisoformat(inp);mon=d-datetime.timedelta(days=d.weekday())
        if answer.replace('天','日')!='星期'+'一二三四五六日'[d.weekday()]:return []
        fam="日期换算";core=f"{inp}是星期几？"
        anchor=datetime.date(d.year,1,1); offset=(d-anchor).days
        refdow="星期"+"一二三四五六日"[anchor.weekday()]
        why=f"{d.year}年1月1日是{refdow}，到这一天相隔{offset}天，除以7余{offset%7}，因此是{answer}。"
        reqs=("只写星期几。","给出星期几，并说明按七天一周如何推得。")
    elif minor==['错别字']:
        if len(inp)!=len(answer):return []
        changes=[i for i,(a,b) in enumerate(zip(inp,answer)) if a!=b]
        if len(changes)!=1:return []
        i=changes[0];left=max(0,i-2);right=min(len(inp),i+3)
        original=inp[left:right];correct=answer[left:right]
        core="改正下面句子中的错别字。\n"+inp; fam="错字校对"
        edits={"coig_pc/coig_pc_core_sample-L02399":("渲下铜牌","拿下铜牌","获奖应说“拿下铜牌”，“渲”不能表达取得奖牌的意思。"),
               "coig_pc/coig_pc_core_sample-L02413":("菖养研究所","营养研究所","研究饮食的机构名称用“营养”，“菖”应改为“营”。"),
               "coig_pc/coig_pc_core_sample-L02412":("很他老人","很多老人","这里用“很多”表示数量，所以把“他”改为“多”。"),
               "coig_pc/coig_pc_core_sample-L02402":("世霁报业协会","世界报业协会","机构名称中应为“世界”，“世霁”的“霁”是错字。")}
        if r['id'] not in edits:return []
        original,correct,why=edits[r['id']]
        assert original in inp and correct in answer
        reqs=("只给改好的句子。","给出改好的句子，并指出哪处错字作了什么更改。")
    elif minor==['词分割'] and '分词' in inst and len(inp)<=110:
        items=answer.split()
        if len(items)<3 or len(items)>20 or ''.join(items)!=''.join(inp.split()):return []
        if not all(re.search(r'[\w\u4e00-\u9fff]',t) for t in items[:2]):return []
        core="用空格将下面的句子分词。\n"+inp;fam="分词与边界核对"
        why=f"第一处分隔位于“{items[0]}”和“{items[1]}”之间；包括单独分开的标点在内，共有{len(items)}个片段。"
        reqs=("只交付分词结果。","给出分词结果，并用首处分隔和片段数说明你的切分；单独分开的标点算一个片段。")
    elif minor==['命名实体识别'] and '{' in answer and len(inp)<160:
        try: entities=json.loads(answer)
        except ValueError:return []
        if not isinstance(entities,dict) or not entities:return []
        evid=[]
        for typ,vs in entities.items():
            if not isinstance(vs,dict) or not vs:return []
            for word,locs in vs.items():
                if not isinstance(locs,list) or not locs:return []
                for pair in locs:
                    if not isinstance(pair,list) or len(pair)!=2 or not all(type(n)==int for n in pair):return []
                    a,b=pair
                    if not(0<=a<=b<len(inp)) or inp[a:b+1]!=word:return []
                    if typ=="职位" and re.search("党员|团员|民族",word):return []
                    evid.append((a,b,word,typ))
        a,b,word,typ=sorted(evid)[0]
        core=f"从以下文字提取实体，按原任务的类别及从0计数的首尾下标返回 JSON 对象。\n类别与示例：{inst}\n原文：{inp}"
        if CONFLICT.search(inst) or len(core)>550:return []
        fam="实体位置核验";why=f"例如从0计数的第{a}至{b}个字符连起来是“{word}”，对应“{typ}”类。"
        reqs=("只返回结果对象。","给出结果对象，并用原文位置核对其中一个实体。")
    else:return []
    return [Variant(core,r['id'],fam,answer,"short",prompt(core,reqs[0]),answer,prompt(core,reqs[1]),answer+"\n"+why,why).metrics()]

MATCHES = [
    ("陛下","您","对君主的尊称"),("翌日","第二天","时间"),("次日","第二天","时间"),("明日","第二天","时间"),
    ("翌年","第二年","年份"),("次年","第二年","年份"),("未几","不久","时间"),("未几","没过多久","时间"),
    ("少顷","一会儿","时间"),("俄顷","一会儿","时间"),("不复","不再","否定"),("不得已","不得不","情态"),
    ("去岁","去年","年份"),("是岁","这一年","年份"),("是月","这个月","时间"),("是日","这一天","时间"),
    ("是时","这时","时间"),("此时","这时","时间"),("不敢","不敢","语气"),("不肯","不肯","意愿"),
    ("不知","不知道","认知"),("不欲","不愿","意愿"),("不许","不允许","许可"),("莫不","无不","范围"),
    ("以为","认为","理解"),("曰","说道","言说"),("曰","说：","言说"),("谓","对","言说"),
    ("皆","全都","范围"),("皆","都是","范围"),("悉","全部","范围"),("乃","于是","承接"),("遂","于是","承接"),
    ("不得","不能","情态"),("欲","想要","意愿"),("何以","凭什么","疑问"),("何故","为什么","疑问"),
    ("未尝","从未","否定"),("不以为然","不赞成","态度"),("所至","所到之处","处所"),("何如","怎么样","疑问"),
    ("未能","没能","结果"),("其后","此后","时间"),("及至","等到","时间"),("人皆","人都","范围")
]
# The items below have an unambiguous string correspondence in the particular source and reference.
MATCHES=[z for z in MATCHES if z[0]!=z[1] and z[0] not in {"谓","以为","陛下","不得"}]
PROPER = ["孔子","孟子","庄子","老子","陶渊明","王安石","司马迁","司马光","诸葛亮","李白","杜甫","苏轼","李世民","汉武帝","唐太宗","汉高祖","汉文帝","秦始皇","洛阳","长安","黄河","泰山","洞庭湖","庐山","东海","骊山"]
def translation(r):
    if "translate_classical_chinese" not in r["source_file"] or UNSAFE.search(r['prompt']+r['response']):return []
    p=r['prompt'];ans=r['response']
    classical="文言文" in p and "现代文" not in p
    modern="现代文" in p and "文言文" not in p
    if not (classical or modern):return []
    core=p.strip()
    core=re.sub(r'\n答案[:：]\s*$','',core)
    normal=p if classical else ans;old=ans if classical else p
    anchors=[(a,b,t) for a,b,t in MATCHES if old.count(a)==1 and normal.count(b)==1 and abs(old.index(a)/len(old)-normal.index(b)/len(normal))<=.10]
    for name in PROPER:
        if old.count(name)==1 and normal.count(name)==1: anchors.insert(0,(name,name,'专名'))
    if not anchors:return []
    a,b,t=anchors[0]
    if t=='专名':
        why=f"原文中的专名“{a}”在译文里保留为“{a}”，没有改写成普通词。"
    else:
        before,after=(b,a) if classical else (a,b)
        why=f"措辞取舍方面，原文用“{before}”表达{t}，译文用“{after}”传达这一层意思。"
    return [Variant(core,r['id'],'古今互译用词核对',ans,'short',prompt(core,'只给译文。'),ans,
                    prompt(core,'给出译文，并用一处原文和译文的用词对应解释你的处理。'),ans+'\n'+why,why).metrics()]


def arithmetic(r):
    import ast,operator
    from fractions import Fraction
    if r['source_file']!="coig_pc/coig_pc_core_sample.jsonl" or r['task_type']['minor']!=['问答','计算']:return []
    try: obj=ast.literal_eval(r['response'])
    except (ValueError,SyntaxError): return []
    if not isinstance(obj,dict) or not all(k in obj for k in ["公式","答案","题目（分词后）"]):return []
    p=r['prompt'].split("\n\n")[-1]
    def calc(s):
        s=re.sub(r'(\d+(?:\.\d+)?)%',r'(\1/100)',s)
        tree=ast.parse(s,mode='eval')
        def visit(node):
            if isinstance(node,ast.Constant) and type(node.value) in [int,float]:return Fraction(str(node.value))
            if isinstance(node,ast.BinOp) and type(node.op) in {ast.Add,ast.Sub,ast.Mult,ast.Div}:
                return {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}[type(node.op)](visit(node.left),visit(node.right))
            raise ValueError
        return visit(tree.body)
    try: val=calc(obj['公式'].removeprefix('x=')); ref=calc(obj['答案'])
    except (ValueError,SyntaxError,ZeroDivisionError):return []
    if val!=ref or '化简分数' in p or re.search('^52与48的和4与',p):return []
    return [Variant(p,r['id'],'四则运算核算',str(val),'reason',prompt(p,'给出题目分词、列式和计算结果。'),r['response'],
                    prompt(p,'只给结果，分数用最简分数表示。'),str(val),'原列式用精确有理数独立复算，与原答案相等。').metrics()]


def candidates(rs):
    out={}
    for r in rs:
        ans=mc(r) or algorithm(r) or arithmetic(r) or translation(r)
        keep=[v for v in ans if v.ai+v.asup<=384 and v.oi+v.osup<=384 and v.asup==r['supervised_tokens']]
        if keep:out[r['id']]=keep
    return out


def catalog(qi, ids, record, vs, second):
    randomgen=random.Random(32001+qi)
    dby=collections.defaultdict(list)
    for id_ in ids:
        if id_ not in vs:dby[record[id_]['supervised_tokens']].append(id_)
    core={k:vs[k] for k in ids if k in vs}; edges=[]; variables=[]
    for id_,alts in core.items():
        a=record[id_]
        for j,v in enumerate(alts):
            ref=v.ai+v.oi-a['input_tokens']
            ds=dby[v.osup]
            if not ds:continue
            ranked=sorted(ds,key=lambda b:record[b]['input_tokens'])
            near=sorted(ds,key=lambda b:abs(record[b]['input_tokens']-ref))
            same_src=sorted([b for b in ds if record[b]['source']==a['source']],key=lambda b:abs(record[b]['input_tokens']-ref))
            donors=list(dict.fromkeys(ranked[:3]+ranked[-4:]+near[:6]+same_src[:5]+randomgen.sample(ds,min(12,len(ds)))))
            for b in donors:
                br=record[b]; delta=ref-br['input_tokens']
                # Prefer exact same sources, then naturally aligned prompt lengths (within the same pair count).
                quality=(.12 if br['source']==a['source'] else 0)+(.07 if v.family!='来源单选解析' else 0)-min(abs(delta),400)/3500.
                if v.family=='来源单选解析' and v.other_answer==v.key:quality+=.025
                second_delta=[0]*4
                second_delta[second[id_]] += v.ai-a['input_tokens']
                second_delta[second[b]] += v.oi-br['input_tokens']
                assert sum(second_delta)==delta
                edges.append([id_,b,second_delta,quality])
                variables.append((id_,j,b))
    return edges,variables


def balance_second(selected, vs, record, second):
    """Exchange matching-length donor slots. Solve remaining second-quarter input equalities exactly."""
    import itertools
    # Each item stays in its first-quarter group and its donor retains exactly its old target length.
    items=list(selected)
    group=collections.defaultdict(list)
    balance=[0]*4
    for idx,(qi,a,j,b) in enumerate(items):
        var=vs[a][j]
        group[(qi,var.osup)].append(idx)
        balance[second[a]]+=var.ai-record[a]['input_tokens']
        balance[second[b]]+=var.oi-record[b]['input_tokens']
    assert sum(balance)==0
    if not any(balance):return items
    all_swaps=collections.defaultdict(list)
    for ids in group.values():
        for i,j in itertools.combinations(ids,2):
            _,a,av,b=items[i];_,x,xv,y=items[j]
            q,p=second[b],second[y]
            if p==q:continue
            delta=vs[x][xv].oi-vs[a][av].oi
            if not delta:continue
            all_swaps[q,p].append((i,j,delta))
            all_swaps[p,q].append((i,j,-delta))
    for attempt in range(140):
        rng=random.Random(624001+attempt); pivot=attempt%4
        occupied=set();picks=[]
        need=[q for q in range(4) if q!=pivot and balance[q]]
        need.sort(key=lambda q:-abs(balance[q]))
        for q in need:
            target=-balance[q]; magnitude=abs(target)
            candidates=[(i,j,abs(v)) for i,j,v in all_swaps[q,pivot]
                        if i not in occupied and j not in occupied and v*target>0 and abs(v)<=magnitude]
            rng.shuffle(candidates)
            # Build a matching first so its knapsack subsets are also disjoint.
            disjoint=[];taken=set()
            for i,j,v in candidates:
                if i not in taken and j not in taken:
                    disjoint.append((i,j,v)); taken.update([i,j])
            # Nonnegative integer subset sum; keep links only to earlier indices.
            prev={0:None}
            for k,(i,j,v) in enumerate(disjoint):
                for s in list(prev):
                    if s+v<=magnitude and s+v not in prev:prev[s+v]=(s,k)
                if magnitude in prev:break
            if magnitude not in prev:break
            ss=magnitude
            while ss:
                ss,k=prev[ss]; i,j,_=disjoint[k];picks.append((i,j));occupied.update((i,j))
        else:
            for i,j in picks:
                qi,a,av,b=items[i];qj,x,xv,y=items[j]
                assert qi==qj and record[b]['supervised_tokens']==record[y]['supervised_tokens']
                items[i]=(qi,a,av,y);items[j]=(qj,x,xv,b)
            print('secondary schedule balanced with',len(picks),'disjoint donor swaps',flush=True)
            return items
    raise RuntimeError(f'Could not solve second-quarter differences {balance}')

def main():
    global T
    import subprocess,tempfile,concurrent.futures
    p=argparse.ArgumentParser()
    p.add_argument('--solver-python',default=sys.executable,help='Python with scipy.optimize.milp (unused with --selection)')
    p.add_argument('--selection',type=Path,help='Rebuild the exact frozen sample using pairs.jsonl; bypass solver.')
    p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    T=Tokenizer.from_file(str(TOKEN))
    rs=rows(PARENT/'train.jsonl'); record={r['id']:r for r in rs}
    order=read(WORK/'pair-train/order.json')
    schedule=read(WORK/'pair-train/schedule.json')
    assert [q['ids'] for q in schedule['quarters'][:4]]==order['quarters']
    second={idx:q for q,b in enumerate(schedule['quarters'][4:]) for idx in b['ids']}
    assert len(second)==9000
    vs=candidates(rs)
    print('candidate cores:',len(vs),dict(collections.Counter(obs[0].family for obs in vs.values())),flush=True)
    if args.selection:
        selected=[(r['quarter']-1,r['source_id'],r['variant_index'],r['replaced_id']) for r in rows(args.selection)]
        slot_order={id_:i for q in order['quarters'] for i,id_ in enumerate(q)}
        selected.sort(key=lambda item:(item[0],slot_order[item[1]]))
    else:
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            def solve(qi):
                edges,index=catalog(qi,order['quarters'][qi],record,vs,second)
                src=Path(d)/f'{qi}.json'; src.write_text(json.dumps(edges))
                dst=Path(d)/f'{qi}-solution.json'
                cmd=[args.solver_python,str(HERE/'match.py'),str(src),str(dst)]
                subprocess.run(cmd,check=True)
                return [(qi,*index[j]) for j in json.loads(dst.read_text())]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                selected=[entry for picks in pool.map(solve,range(4)) for entry in picks]
        selected=balance_second(selected,vs,record,second)
    result={r['id']:r.copy() for r in rs}
    pairs=[]
    for qi,anchor_id,j,donor_id in selected:
        v=vs[anchor_id][j]
        a=record[anchor_id];d=record[donor_id]
        for base,newp,newa,mode in [(a,v.anchor_prompt,v.anchor_answer,v.anchor_mode),(d,v.other_prompt,v.other_answer,'reason' if v.anchor_mode=='short' else 'short')]:
            new={**a,'id':base['id'],'prompt':newp,'response':newa,'baseline_id':base['id'],'source_id':anchor_id,'pair_mode':mode,
                 'input_tokens':il(newp),'supervised_tokens':sl(newa),'total_tokens':il(newp)+sl(newa)}
            assert new['supervised_tokens']==base['supervised_tokens'] and new['total_tokens']<=384
            result[base['id']]=new
        pairs.append(dict(quarter=qi+1,variant_index=j,source_id=anchor_id,replaced_id=donor_id,family=v.family,source=a['source'],
                          removed_source=d['source'],core=v.core,gold=v.key,anchor_mode=v.anchor_mode,evidence=v.evidence))
    final=[result[r['id']] for r in rs]
    assert len(final)==len(rs)==len(set(result))==9000
    assert all(d['supervised_tokens']==c['supervised_tokens'] for c,d in zip(rs,final))
    counts=[]
    for obj in schedule['quarters']:
        quarter=obj['ids']
        before=[record[k] for k in quarter];after=[result[k] for k in quarter]
        stat=lambda q:[sum(r[f] for r in q) for f in ['input_tokens','supervised_tokens','total_tokens']]
        assert stat(before)==stat(after),(stat(before),stat(after))
        counts.append(dict(rows=len(after),input_tokens=stat(after)[0],supervised_tokens=stat(after)[1],total_tokens=stat(after)[2]))
    writelines(out/'paired.jsonl',final)
    writelines(out/'pairs.jsonl',sorted(pairs,key=lambda r:(r['quarter'],r['source_id'])))
    hashes={f:sha((out/f).read_bytes()) for f in ['paired.jsonl','pairs.jsonl']}
    manifest=dict(baseline_sha256=sha((PARENT/'train.jsonl').read_bytes()),derived_sha256=hashes,order_sha256=sha((WORK/'pair-train/order.json').read_bytes()),schedule_sha256=sha((WORK/'pair-train/schedule.json').read_bytes()),
                  pairs=len(pairs),by_family=dict(collections.Counter(r['family'] for r in pairs)),by_quarter=dict(collections.Counter(r['quarter'] for r in pairs)),
                  by_source=dict(collections.Counter(r['source'] for r in pairs)),removed_sources=dict(collections.Counter(r['removed_source'] for r in pairs)),
                  same_source_count=sum(r['source']==r['removed_source'] for r in pairs),unchanged_rows=9000-2*len(pairs),
                  retained_unique_base_items=9000-len(pairs),sources_considered=len(vs),quarter_totals=counts)
    dump(out/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
