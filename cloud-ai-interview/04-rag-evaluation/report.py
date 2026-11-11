"""静态报告：默认展示结论，细节折叠；原文全部转义，不执行其中 HTML。"""
from html import escape
from pathlib import Path
import json


def esc(value): return escape(str(value),quote=True)


def page(title,content,path):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+esc(title)+'''</title><style>
body{font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif;color:#26354b;background:#f3f6fa;margin:0}main{max-width:1080px;margin:36px auto;padding:0 24px 60px}h1{font-size:30px}h2{font-size:21px}h3{font-size:17px}section,.banner{background:white;border:1px solid #dce4ee;border-radius:14px;padding:22px;margin:18px 0}.banner{background:#e7effe}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #e4e9ef}th{background:#edf2f9}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:16px;border-radius:9px;font:14px/1.65 monospace}summary{cursor:pointer;color:#315ca1;margin:8px 0}.good{color:#176b4d}.bad{color:#a75120}.muted{color:#607086}a{color:#315ca1}.answer{font-size:18px;border-left:4px solid #4c77bc;padding:15px;background:#f5f8ff}.chips{display:flex;gap:15px;flex-wrap:wrap}.chips div{background:white;border:1px solid #cfdbed;padding:12px 20px;border-radius:10px}code{background:#edf1f7;padding:2px 5px} @media(max-width:700px){main{padding:0 12px}td,th{padding:7px;font-size:13px}}
</style><main><h1>'''+esc(title)+'</h1>'+content+'</main></html>'
    Path(path).write_text(html,encoding='utf-8')
    return Path(path)


def status(row):
    n=row['score']['complete']
    return '不适用：资料无答案' if n is None else ('标注依据已齐' if n else '标注依据未齐')


def detail(row):
    c=row['case']; s=row['score']
    out='<section id="'+esc(c['id'])+'"><h2>'+esc(c['id']+' · '+c['question'])+'</h2>'
    out+='<p><b>'+esc(status(row))+'</b> · 已覆盖 '+str(s['covered'])+'/'+str(s['required'])+' 处标注依据 · 上下文 '+str(row['context_characters'])+' 字符</p>'
    out+='<p class="muted">此处只评价依据覆盖，不评价回答正确性。</p>'
    if c.get('note'): out+='<p>'+esc(c['note'])+'</p>'
    out+='<details><summary>先看人工标注：为什么认为资料有答案？</summary><p>'+esc(c['expected'])+'</p>'
    for ev,flag in zip(row['evidence'],s['flags']):
        out+='<p>'+('✓ 覆盖' if flag else '△ 未覆盖')+' · '+esc(ev['source'])+' · '+str(ev['start'])+':'+str(ev['end'])+'</p><pre>'+esc(ev['quote'])+'</pre>'
    out+='</details><details><summary>'+('再看人工选定的真实原文' if row.get('manual_evidence') else '再看本次实际检索到的片段')+'</summary>'
    for h in row['hits']:
        out+='<h3>'+esc(h['id']+' · '+h['source'])+'</h3><p>字符 '+str(h['start'])+':'+str(h['end'])+' · 距离 '+(format(h['cosine_distance'],'.4f') if h['cosine_distance'] is not None else '人工选取，不计算距离')+'</p><pre>'+esc(h['text'])+'</pre>'
    return out+'</details></section>'


def render_run(run,path):
    s=run['summary']
    out='<div class="banner"><b>这次只检查检索，没有调用回答 API。</b><p>绿色表示人工标注的依据范围已被候选片段覆盖，不等于答案正确。无答案问题单列，不混入覆盖率。</p><div class="chips"><div>取回 '+str(run['k'])+' 段 / 问题</div><div>依据找齐 '+str(s['complete_count'])+' / '+str(s['answerable_count'])+' 题</div><div>平均上下文 '+str(s['mean_context_characters'])+' 字符</div></div></div>'
    out+='<table><tr><th>问题</th><th>依据</th><th>带入文字</th></tr>'
    for r in run['rows']:
        out+='<tr><td><a href="#'+esc(r['case']['id'])+'">'+esc(r['case']['id']+' '+r['case']['question'])+'</a></td><td>'+'<span class="'+('good' if r['score']['complete'] is True else 'bad' if r['score']['complete'] is False else 'muted')+'">'+esc(status(r))+'</span></td><td>'+str(r['context_characters'])+' 字符</td></tr>'
    out+='</table>'+''.join(detail(r) for r in run['rows'])
    return page('第四课 · 这次 RAG 找齐依据了吗？',out,path)


def render_compare(before,after,rows,path):
    out='<div class="banner">同一资料、切分、向量模型和问题集，只比较 k='+str(before['k'])+' 与 k='+str(after['k'])+'。<br>改善和退步仅指“标注依据是否找齐”；没有测试生成答案。</div><table><tr><th>题目</th><th>变化</th><th>覆盖处数：前 → 后</th><th>上下文字符：前 → 后</th></tr>'
    for row in rows:
        a,b=row['before'],row['after']
        out+='<tr><td>'+esc(row['id'])+'</td><td>'+esc(row['change'])+'</td><td>'+str(a['score']['covered'])+' → '+str(b['score']['covered'])+'</td><td>'+str(a['context_characters'])+' → '+str(b['context_characters'])+'</td></tr>'
    out+='</table><section><h2>是否保留改动？</h2><p>先看哪道失败题改善了，再检查原来成功的题有没有退步，以及上下文增加了多少。如果没有改善，不必为了优化而改配置。</p><p>字符数不是 token 数；这里没有用它直接估计费用。8 题是课堂开发集，不是生产正确率。</p></section>'
    return page('第四课 · 一次只改变返回片段数量',out,path)


def render_answer(record,path):
    row=record['row']; result=record.get('response'); error=record.get('error')
    if result:
        top='<div class="answer">'+esc(result['answer']).replace('\n','<br>')+'</div><p>生成状态：'+esc(result['finish_reason'])+' · 用量：'+esc(json.dumps(result['usage'],ensure_ascii=False))+'</p>'
    elif error: top='<div class="answer">调用失败：'+esc(error['code'])+' · '+esc(error['message'])+'</div>'
    else: top='<div class="answer">仅预览，还没有调用回答模型。</div>'
    top+='<p>本次使用 '+esc({'retrieved':'自动检索所得','manual':'人工选定的真实原文（仅用于诊断）'}[record['evidence_mode']])+' 片段。人工预期答案和证据标签没有发给模型。</p>'
    top+='<section><h2>人工核对三件事</h2><p>① 回答是否解决问题？② 引用原文是否支持结论？③ 有没有不该编造的内容？</p><p>超时算通信失败，不能直接算模型答错；引用编号存在也不代表事实正确。</p></section>'
    top+='<details><summary>展开实际发送的模型消息</summary><pre>'+esc(json.dumps(record['messages'],ensure_ascii=False,indent=2))+'</pre></details>'
    top+=detail(row)
    return page('第四课 · '+row['case']['id']+' 的回答检查',top,path)
