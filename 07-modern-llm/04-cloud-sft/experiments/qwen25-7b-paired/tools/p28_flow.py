from html import escape
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
def record(stage):
    rows=json.loads((root/'evidence/P28.json').read_text())
    arm='control' if stage=='base' else 'paired'
    return next(v for v in rows if v['id']=='P28S' and v['checkpoint']==stage and v['arm']==arm)
r=record('base'); s=record('epoch-2')
assert r['token_ids'][:5]==[30709,27091,271,100021,107826] and not r['ended_any_stop'] and r['generated_tokens']==256
assert s['token_ids']==[30709,27091,151643]
out=root/'figures/P28-纯底座生成流程.svg'
a=['''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="870" viewBox="0 0 1120 870" role="img" aria-labelledby="title desc">
<title id="title">Qwen2.5-7B 纯底座如何生成“小面”</title><desc id="desc">上方是从题目到 generate 的调用；中间是下一 token 的循环；下方是真实的前三步与停止情况。</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#43776d"/></marker></defs>
<style>text{font-family:"Hiragino Sans GB","PingFang SC","Microsoft YaHei",sans-serif}.num{font-variant-numeric:tabular-nums}</style>
<rect width="1120" height="870" fill="#f6f6f0"/>''']
def rect(x,y,w,h,fill,stroke='none',rx=12):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
def txt(x,y,t,size=18,color='#153d37',bold=False):
    a.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{650 if bold else 400}" fill="{color}">{escape(t)}</text>')
def arrow(d):
    a.append(f'<path d="{d}" stroke="#43776d" stroke-width="2.1" stroke-linejoin="round" fill="none" marker-end="url(#arrow)"/>')
txt(42,39,'P28S   /   木块压强题   /   Qwen2.5-7B Base',15,'#476b63',True)
txt(42,95,'纯预训练底座怎样生成「小面」？',38,'#153d37',True)
txt(43,133,'这一题要求只写“大面”或“小面”。下面从评测脚本的一次调用，追到每一步实际生成的 token。',18)
txt(42,190,'01   评测脚本把什么交给模型？',23,'#153d37',True)
for x,w,title,line1,line2 in [
    (42,224,'读取底座','from_pretrained(…)','model = base，未挂 LoRA'),
    (291,234,'包装题目','prompt(题目 + 短答指令)','结尾留出 assistant 的位置'),
    (550,222,'转成张量','tok(…) → input_ids','attention_mask 标出有效位置'),
    (797,281,'开始生成','model.generate(**inputs, …)','贪心解码，上限 256 个新 token')]:
    rect(x,210,w,112,'#ffffff','#c9d8d1')
    txt(x+13,240,title,20,'#153d37',True);txt(x+13,273,line1,16);txt(x+13,300,line2,14,'#466a62')
for x in (266,525,772): arrow(f'M{x+2} 265 h20')
txt(42,375,'02   generate 内部：每次选一个 token',23,'#153d37',True)
for x,w,title,detail in [(42,174,'Embedding','token ID → 向量'),(244,266,'28 层 Transformer','每层：因果自注意力 + MLP'),(538,284,'RMSNorm → LM head','末位置向量 → 152,064 个分数'),(850,228,'argmax','选分数最高的 token ID')]:
    rect(x,397,w,88,'#e2efea','#afcbbf');txt(x+13,433,title,20,'#153d37',True);txt(x+13,463,detail,15)
for x in (216,510,822): arrow(f'M{x+2} 441 h23')
arrow('M964 486 V512 H128 V488')
txt(247,545,'没遇到结束符且未到上限：把刚选的 token 接到上下文，复用 KV 缓存，再走一遍。',15,'#466a62')
txt(42,604,'03   实测：为什么没有停在「小面」？',23,'#153d37',True)
for x,w,head,outcome,ids in [(42,236,'第 1 步 · 读完整提示','小','ID 30709'),(303,236,'第 2 步 · 已生成「小」','面','ID 27091'),(564,236,'第 3 步 · 已生成「小面」','两个换行','ID 271'),(825,253,'第 4 步起 · 继续生成','水平、放置、…','开始重复题目')]:
    rect(x,625,w,113,'#153d37' if x<564 else '#f7ecd7')
    c='#ffffff' if x<564 else '#553e22'
    txt(x+13,652,head,14,c);txt(x+13,693,outcome,26,c,True);txt(x+13,720,ids,15,c)
for x in (278,539,800):arrow(f'M{x+2} 682 h20')
txt(43,782,'纯底座：生成满 256 个 token 才停。双问 SFT 两轮后：同题前两步不变，第三步选中结束符 151643。',17,'#153d37',True)
txt(43,829,'来源：experiment.py 的 generate()；P28S 的 base.jsonl 和 epoch-2.jsonl。日志记录了选中的 ID，没有保存各候选的概率。',14,'#466a62')
a.append('</svg>')
out.write_text('\n'.join(a))
