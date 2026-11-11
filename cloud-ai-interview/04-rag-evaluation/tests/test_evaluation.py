import asyncio
from argparse import Namespace
from copy import deepcopy
import json
import pytest
import core
import lab
from report import render_run, render_answer


def case(id_='q01',answerable=True):
    return {'id':id_,'question':'测试问题','answerable':answerable,'expected':'DO_NOT_SEND_LABEL',
            'evidence':[{'source':'a.md','quote':'真实原文'}] if answerable else []}


def row(answerable=True,complete=True):
    return {'case':case(answerable=answerable),'score':{'required':int(answerable),'covered':int(answerable and complete),
            'complete':complete if answerable else None,'flags':[complete] if answerable else []},
            'hits':[{'id':'chunk-0001','source':'a.md','start':0,'end':4,'text':'真实原文','cosine_distance':.1}],
            'evidence':[{'source':'a.md','quote':'真实原文','start':0,'end':4}] if answerable else [],'context_characters':50}


def run():
    rows=[row(),row(False)]
    rows[1]['case']['id']='q02'
    return {'index':'fixture','index_manifest':{'documents_hash':'hash','model':'fixture'},'suite_hash':'suite',
            'k':3,'rows':rows,'summary':core.summary(rows)}


def test_union_covers_evidence():
    span={'source':'a','start':3,'end':12}
    assert core.span_covered(span,[{'source':'a','start':0,'end':7},{'source':'a','start':7,'end':14}])


@pytest.mark.parametrize('hits',[
    [{'source':'a','start':0,'end':7},{'source':'a','start':8,'end':14}],
    [{'source':'b','start':0,'end':30}],
    [{'source':'a','start':4,'end':30}],[]])
def test_gap_or_wrong_file_not_covered(hits):
    assert not core.span_covered({'source':'a','start':3,'end':12},hits)


def test_empty_evidence_not_automatic_success():
    assert core.score_evidence([],[])['complete'] is None


def test_summary_denominator_and_cost():
    s=core.summary([row(),row(True,False),row(False)])
    assert s['complete_count']==1 and s['answerable_count']==2
    assert s['coverage_rate']==.5 and s['unanswerable_count']==1
    assert s['mean_context_characters']==50


def test_all_unanswerable_has_no_rate():
    assert core.summary([row(False)])['coverage_rate'] is None


def test_locate_requires_unique_anchor():
    assert core.locate_evidence(case(),[{'source':'a.md','text':'xx真实原文'}])[0]['start']==2
    with pytest.raises(ValueError):core.locate_evidence(case(),[{'source':'a.md','text':'真实原文真实原文'}])


@pytest.mark.parametrize('field',['suite_hash','index_manifest'])
def test_comparison_rejects_other_changes(field):
    a=run();b=deepcopy(a);b[field]='changed'
    with pytest.raises(ValueError):core.compare(a,b)


def test_comparison_tracks_regression():
    a=run();b=deepcopy(a);b['k']=5;b['rows'][0]['score']['complete']=False
    changes=core.compare(a,b)
    assert changes[0]['change']=='退步' and changes[1]['change']=='不计分'


def test_output_refuses_overwrite(tmp_path):
    p=tmp_path/'record.json';core.write_json(p,{'a':1})
    with pytest.raises(FileExistsError):core.write_json(p,{'a':2})
    assert core.read_json(p)=={'a':1}


@pytest.mark.parametrize('name',['../x','a/b','name with spaces',''])
def test_names_reject_traversal(name):
    with pytest.raises(ValueError):core.valid_name(name)


def test_html_escapes_document_code(tmp_path):
    d=run();d['rows'][0]['hits'][0]['text']='<script>alert(1)</script>'
    out=render_run(d,tmp_path/'report.html').read_text()
    assert '<script>alert(1)</script>' not in out
    assert '&lt;script&gt;' in out
    assert '不适用：资料无答案' in out


def test_manual_evidence_is_original_not_label(tmp_path,monkeypatch):
    monkeypatch.setattr(lab,'ROOT',tmp_path)
    docs=[{'source':'a.md','text':'真实原文，后文信息'}]
    core.write_json(tmp_path/'artifacts/indexes/fixture/documents.json',docs)
    d=run();d['index_manifest']['documents_hash']=core.digest(docs)
    r=lab.selected_row(d,'q01','manual')
    assert r['hits'][0]['text']==docs[0]['text']
    assert r['hits'][0]['cosine_distance'] is None
    assert 'DO_NOT_SEND_LABEL' not in r['hits'][0]['text']
    assert r['score']['complete'] is True
    with pytest.raises(ValueError):lab.selected_row(d,'q02','manual')


def test_preview_never_calls_generation(tmp_path,monkeypatch):
    import answer
    monkeypatch.setattr(lab,'ROOT',tmp_path)
    core.write_json(lab.run_path('fixture'),run())
    async def fail(*args,**kwargs):pytest.fail('preview must not call model')
    monkeypatch.setattr(answer,'generate',fail)
    rc=asyncio.run(lab.answer(Namespace(run='fixture',case='q01',evidence='retrieved',live=False)))
    result=core.read_json(next((tmp_path/'artifacts/answers').glob('*.json')))
    assert rc==0 and result['call_state']=='not_started'
    assert 'DO_NOT_SEND_LABEL' not in json.dumps(result['messages'])


def test_live_timeout_preserves_failure_and_exits_nonzero(tmp_path,monkeypatch):
    import answer
    monkeypatch.setattr(lab,'ROOT',tmp_path)
    core.write_json(lab.run_path('fixture'),run())
    calls=[]
    async def timeout(*args,**kwargs):
        calls.append(1)
        raise answer.ServiceError(504,'upstream_timeout','人工超时测试')
    monkeypatch.setattr(answer,'generate',timeout)
    rc=asyncio.run(lab.answer(Namespace(run='fixture',case='q01',evidence='retrieved',live=True)))
    result=core.read_json(next((tmp_path/'artifacts/answers').glob('*.json')))
    assert rc==1 and len(calls)==1
    assert result['error']['code']=='upstream_timeout'
    assert 'response' not in result


def test_answer_report_does_not_call_preview_live(tmp_path):
    record={'row':row(),'evidence_mode':'retrieved','messages':[],
            'response':{'answer':'人工测试回答','finish_reason':'stop','usage':None}}
    text=render_answer(record,tmp_path/'answer.html').read_text()
    assert '人工测试回答' in text
    assert '未调用回答模型' not in text
