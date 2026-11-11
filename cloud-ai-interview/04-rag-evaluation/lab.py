"""命令入口。先看报告，只有讲义指定时才读对应函数。"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from core import ROOT, PREVIOUS, cases, prepare, evaluate, compare, valid_name, read_json, write_json, score_evidence
from report import render_run, render_compare, render_answer


def run_path(name):
    return ROOT/'artifacts/runs'/(valid_name(name)+'.json')


def show(path):
    print('报告已保存：',path)
    print('在 VS Code 中找到这个 HTML 文件，右键在 Finder 显示，再双击用浏览器打开。')


def selected_row(run, case_id, mode):
    rows=[r for r in run['rows'] if r['case']['id']==case_id]
    if not rows: raise ValueError('问题编号不存在，请看 cases.json。')
    row=deepcopy(rows[0])
    if mode=='manual':
        if not row['evidence']: raise ValueError('无答案问题没有人工证据，不能用 manual 模式。')
        # 诊断对照：只选择真实原文，不把 expected（人工预期答案）放进上下文。
        docs=read_json(ROOT/'artifacts/indexes'/valid_name(run['index'])/'documents.json')
        from core import digest
        if digest(docs)!=run['index_manifest']['documents_hash']:
            raise ValueError('原文快照已变化，不能进行人工证据对照。')
        texts={d['source']:d['text'] for d in docs}
        row['hits']=[]
        for i,ev in enumerate(row['evidence'],1):
            text=texts[ev['source']]
            start=max(0,ev['start']-100);end=min(len(text),ev['end']+100)
            row['hits'].append({'id':f'chunk-manual-{i:04d}','source':ev['source'],'start':start,'end':end,
                                'text':text[start:end],'cosine_distance':None})
        row['score']=score_evidence(row['evidence'],row['hits'])
        from knowledge import context_text
        row['context_characters']=len(context_text(row['hits']))
        row['manual_evidence']=True
    return row


async def answer(args):
    from answer import Settings, RagGateway, generate, ServiceError
    from knowledge import context_text
    run=read_json(run_path(args.run))
    row=selected_row(run,args.case,args.evidence)
    settings=Settings.from_env()
    messages=RagGateway(None,settings).payload(row['case']['question'],context_text(row['hits']),512,False)['messages']
    record={'kind':'answer_review','parent_run':args.run,'row':row,'evidence_mode':args.evidence,
            'messages':messages,'live_requested':args.live,'call_state':'not_started',
            'model':settings.model,'read_timeout':settings.read_timeout,'deadline':settings.deadline}
    suffix=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path=ROOT/'artifacts/answers'/f'{args.case}-{suffix}.json'
    write_json(path,record)
    if args.live:
        print('本次只请求一次真实回答 API，可能计费；不自动重试。',flush=True)
        record['call_state']='attempting'
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        try:
            record['response']=await generate(row['case']['question'],row['hits'],settings)
            record['call_state']='completed'
        except ServiceError as exc:
            record['error']={'code':exc.code,'message':exc.message}
            record['call_state']='failed_or_unknown_upstream_completion'
        except ValueError:
            record['error']={'code':'configuration_error','message':'请检查第二课模型配置，未使用人工回答兜底。'}
            record['call_state']='configuration_failed'
        finally:
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    show(render_answer(record,path.with_suffix('.html')))
    if record.get('error'): return 1
    return 0


def main():
    parser=argparse.ArgumentParser(description='第四课：固定问题、先查证据、一次改变一个参数。')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare',help='固定三份资料并建立本课索引，不修改第三课')
    p.add_argument('--name',default='study')
    p=commands.add_parser('evaluate',help='仅评估检索，无回答 API 调用')
    p.add_argument('--index',default='study');p.add_argument('--k',type=int,default=3);p.add_argument('--name',required=True)
    p=commands.add_parser('compare',help='比较两个检索记录')
    p.add_argument('before');p.add_argument('after');p.add_argument('--name',default='comparison')
    p=commands.add_parser('answer',help='只预览一个问题；--live 才调用模型')
    p.add_argument('--run',required=True);p.add_argument('--case',required=True)
    p.add_argument('--evidence',choices=['retrieved','manual'],default='retrieved')
    p.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        print('本课索引已保存：',prepare(args.name))
    elif args.command=='evaluate':
        path=run_path(args.name)
        if path.exists(): raise ValueError('实验名已存在；换名保留记录。')
        result=evaluate(args.index,args.k)
        write_json(path,result)
        s=result['summary']
        print(f"标注依据找齐：{s['complete_count']}/{s['answerable_count']} 题；另外 {s['unanswerable_count']} 道无答案题单独检查。")
        print(f"平均上下文：{s['mean_context_characters']} 字符。未调用回答 API。")
        show(render_run(result,path.with_suffix('.html')))
    elif args.command=='compare':
        a,b=read_json(run_path(args.before)),read_json(run_path(args.after))
        result=compare(a,b)
        path=ROOT/'artifacts/comparisons'/(valid_name(args.name)+'.json')
        write_json(path,{'before':args.before,'after':args.after,'rows':result})
        show(render_compare(a,b,result,path.with_suffix('.html')))
    else:
        return asyncio.run(answer(args))
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        print('未完成：',str(exc),file=sys.stderr)
        sys.exit(1)
