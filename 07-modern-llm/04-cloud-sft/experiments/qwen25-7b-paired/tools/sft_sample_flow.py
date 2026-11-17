"""Rebuild the offline lesson with tokenizers; use the repository's Python environment."""
from pathlib import Path
import tarfile,json,html
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[1]
def member(path,n):
    with tarfile.open(root/path) as t:return t.extractfile(n).read()
def rows(b):return [json.loads(l) for l in b.splitlines()]
def frozen(n):return rows(member('inputs/frozen-inputs.tar.gz','data/'+n+'/train.jsonl'))
srcid='chinese_traditional/trad-multi-choice-100-2-L00013'
raw=next(r for r in frozen('control') if r['id']==srcid)
data={r['id']:r for r in frozen('paired')}
r=next(r for r in data.values() if r.get('source_id')==srcid and r.get('pair_mode')=='short')
u=next(v for v in rows(member('evidence/pair-measurements.tar.gz','run/paired/updates.jsonl')) if r['id'] in v['ids'])
tok=Tokenizer.from_str(member('inputs/base-metadata.tar.gz','tokenizer.json').decode())
template='<|im_start|>user\n'+r['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
pre=tok.encode(template,add_special_tokens=False).ids
ans=tok.encode(r['response'],add_special_tokens=False).ids+[151643]
micro=[data[k] for k in u['ids'][2:4]]
assert (u['step'],u['ids'][2],len(pre),ans,u['targets'])==(3,r['id'],41,[34,151643],468)
width=max(x['total_tokens'] for x in micro); count=sum(x['supervised_tokens'] for x in micro)
esc=html.escape
page=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>一次 SFT 更新：从 C 到梯度</title><style>
*{box-sizing:border-box}body{max-width:1020px;margin:auto;padding:32px 16px 60px;background:#f4f5ef;color:#163a36;font:17px/1.65 system-ui,sans-serif}h1{font-size:clamp(29px,5vw,46px);line-height:1.2}h2{margin-top:44px}h3{margin:.1em 0}.card,details{background:white;border:1px solid #c6d5cd;border-radius:12px;padding:13px 17px}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.flow b,summary{color:#12645b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}summary{cursor:pointer;font-weight:650}details{margin:6px 0}.cell{display:inline-block;padding:6px;margin:2px;border-radius:4px;background:#e5ebe8;font:14px monospace}.gold{background:#ffdf97}.small{font-size:14px}a{color:#14685f}.arrow{color:#638279;text-align:center} @media(max-width:680px){.flow{grid-template-columns:1fr}} </style>
<small>QWEN2.5-7B　/　魔搭 A10　/　双问组 · 第 3 个优化步</small><h1>一道选择题，怎样参与一次权重更新？</h1><a href="#code">跳到图 02：同一步的代码调用 ↓</a><h2>图 01　沿着样本走</h2><section class="flow">''']
def card(a,b):return '<div class="card"><b>'+a+'</b><br>'+b+'</div>'
page += [card('① 原题与来源答案',esc(raw['prompt']).replace('\n\n','<br>')+'<br>来源解析末句：'+esc(raw['response'].splitlines()[-1])),
card('② 清洗、核对、改写','规范换行、去首尾空白，核对四个选项和来源末句的选项字母。给原题加上“只答所选项的字母。”，把短答目标写为 <b>C</b>。'),
card('③ 模板 → token ID','加 user / assistant 标记：提示共 <b>41</b> 个 token。训练时拼入正确回答 <b>C = 34</b>，再加原生结束符 <b>151643</b>。合计 <b>43</b> 个。'),'</section>']
page+=['<details><summary>查看这条真实样本的完整输入、位置和 ID</summary><pre>'+esc(template+'C<|endoftext|>')+'</pre><p>位置:ID（从 0 开始）；灰色是条件，黄色是计入 loss 的目标。</p>'+''.join(f'<span class="cell {"gold" if i>=41 else ""}">{i}:{n}</span>' for i,n in enumerate(pre+ans))+'</details>']
page+=[f'''<div class="arrow">↓ 与同一微批的另一条样本一起右侧补齐 ↓</div><section class="flow">''',
card('④ 一次前向，预测两个位置',f'输入形状 [2, {width}]。28 层因果 Transformer 输出 [2, {width}, 3584]。位置 40 看完提示，预测位置 41 的 C；位置 41 看到了正确的 C，预测位置 42 的结束符。因果遮罩挡住未来的答案。'),
card('⑤ 只给回答打分',f'提示和补齐位置不直接计分。这个微批取 {count} 个预测位置，经 lm_head 得到 [{count}, 152064] 个分数。本题贡献两项：−log P(C｜提示) 与 −log P(结束｜提示,C)。'),
card('⑥ 反传并更新 LoRA','4 条样本分成 2 个微批。每个微批的交叉熵之和除以整步 468 个监督 token，再反传、累积梯度。AdamW 更新 LoRA 的 A、B 矩阵；原始底座权重冻结。'),'</section>']
page+=[f'''<p class="card"><b>云端第 3 步实测</b>　{u['nll_sum']:.4f} ÷ {u['targets']} = <b>{u['loss']:.5f}</b>（4 条合计的平均 loss）。学习率 {u['lr']:.4g}，裁剪前梯度范数 {u['gradient_norm']:.5f}，低于阈值 1。共有 20,185,088 个可训练参数。日志没有保存这条样本单独的概率、loss 或权重改变量。</p>
<h2 id="code">图 02　沿着实际代码走</h2><p>按顺序展开即可读项目源代码。训练前向直接用已知答案监督，不调用推理时的 model.generate()。</p>''']
def snippet(title,p,start,end):
    lines=(root/p).read_text().splitlines()
    return f'<details><summary>{title}　·　{p.split("/")[-1]}:{start}</summary><pre>'+esc('\n'.join(f'{n:3}  {lines[n-1]}' for n in range(start,end+1)))+'</pre></details>'
p='cloud/experiment.py'
for a in [('离线准备　从来源解析提取选项，生成短答','data-prep/pair-data/build.py',61,96),('入口　启动本组 train','cloud/run-pair.sh',15,22),('01　train：校验底座，读 JSONL，调用 encode',p,154,165),('02　prompt → encode：拼提示、答案、EOS，记下 split',p,65,78),('03　挂 LoRA，挑出可训练参数，创建 AdamW',p,180,190),('04　batches → 取 4 条，求整步分母并设置学习率',p,235,251),('05　active_logits：右补齐、因果前向、错位取监督位置',p,93,104),('06　交叉熵 → backward → 裁剪梯度 → optimizer step',p,250,266),('07　写更新日志，到四分之一轮再验证、保存',p,269,283)]:
    page.append(snippet(*a))
page+=['<p class="small">来源：冻结训练集、官方分词器、云端 run/paired/updates.jsonl 和 cloud/experiment.py。张量形状由真实批次长度与代码推得。</p></html>']
(root/'figures/SFT-单样本与训练代码.html').write_text('\n'.join(page))
