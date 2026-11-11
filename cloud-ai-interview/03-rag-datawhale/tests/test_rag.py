import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient
import knowledge
from knowledge import split_documents, fingerprint, search, context_text
from answer import Settings, ServiceError, RagGateway, generate, citation_check
from server import create_app

HITS = [{"id": "chunk-0001", "text": "人工测试材料：分隔符用于区分输入。", "source": "fixture.md", "start": 0, "end": 23, "cosine_distance": .2}]
SETTINGS = Settings("deepseek", "test-model", "https://fixture.invalid/chat/completions", "app-test", "model-test")


def test_split_roundtrip():
    text = '段落一：这是有关资料检索的一段话。\n\n' * 60
    chunks = split_documents([{"source": "a.md", "text": text}], 100, 20)
    assert len(chunks) > 1
    assert len({x['id'] for x in chunks}) == len(chunks)
    for c in chunks:
        assert 0 <= c['start'] < c['end'] <= len(text)
        assert text[c['start']:c['end']] == c['text']
        assert len(c['text']) <= 100


@pytest.mark.parametrize('size,overlap', [(0,0),(300,300),(500,20),(100,-1)])
def test_invalid_split(size,overlap):
    with pytest.raises(ValueError): split_documents([],size,overlap)


def test_stale_index_blocks_before_model(tmp_path, monkeypatch):
    docs = [{"source":"a.md","text":"old"}]
    (tmp_path/'manifest.json').write_text(json.dumps({"model":knowledge.MODEL,"query_prefix":knowledge.QUERY_PREFIX,"documents_sha256":fingerprint(docs)}))
    monkeypatch.setattr(knowledge,'documents',lambda p:[{"source":"a.md","text":"changed"}])
    monkeypatch.setattr(knowledge,'embed',lambda *a,**k:pytest.fail('must not embed stale data'))
    with pytest.raises(ValueError,match='原始资料'): search('问题',tmp_path)


def test_chroma_real_storage_without_download(tmp_path, monkeypatch):
    import chromadb
    docs=[{"source":"fixture.md","text":"test"}]
    (tmp_path/'manifest.json').write_text(json.dumps({"model":knowledge.MODEL,"query_prefix":knowledge.QUERY_PREFIX,"documents_sha256":fingerprint(docs)}))
    monkeypatch.setattr(knowledge,'documents',lambda p:docs)
    monkeypatch.setattr(knowledge,'embed',lambda *a,**k:[[1.,0.,0.]])
    db=chromadb.PersistentClient(path=str(tmp_path), settings=chromadb.config.Settings(anonymized_telemetry=False))
    c=db.create_collection('learning_materials',embedding_function=None,metadata={'hnsw:space':'cosine'})
    c.add(ids=['a','b'],documents=['人工测试A','人工测试B'],embeddings=[[1.,0.,0.],[0.,1.,0.]],metadatas=[{'source':'fixture.md','start':0,'end':5}]*2)
    result=search('测试',tmp_path,3)
    assert [r['id'] for r in result]==['a','b']
    assert result[0]['cosine_distance'] < result[1]['cosine_distance']


def test_model_receives_text_not_vectors():
    def handler(request):
        body=json.loads(request.content)
        assert request.headers['authorization']=='Bearer model-test'
        assert '人工测试材料' in body['messages'][1]['content']
        assert 'chunk-0001' in body['messages'][1]['content']
        assert 'app-test' not in request.content.decode()
        return httpx.Response(200,json={'model':'fixture','choices':[{'message':{'content':'人工测试回答 [chunk-0001]'},'finish_reason':'stop'}]})
    result=asyncio.run(generate('测试',HITS,SETTINGS,transport=httpx.MockTransport(handler)))
    assert result['usage'] is None
    assert result['citation_check']['unknown_ids']==[]


@pytest.mark.parametrize('status,code',[(401,'upstream_auth'),(429,'upstream_rate_limited')])
def test_upstream_error(status,code):
    with pytest.raises(ServiceError) as caught:
        asyncio.run(generate('测试',HITS,SETTINGS,transport=httpx.MockTransport(lambda r:httpx.Response(status))))
    assert caught.value.code==code


def test_truncated_answer_not_success():
    def handler(r): return httpx.Response(200,json={'model':'fixture','choices':[{'message':{'content':'不完整'},'finish_reason':'length'}]})
    with pytest.raises(ServiceError): asyncio.run(generate('测试',HITS,SETTINGS,transport=httpx.MockTransport(handler)))


def test_citation_not_semantic_guarantee():
    check=citation_check('错误结论 [chunk-0001] [invented]',HITS)
    assert check['cited_ids']==['chunk-0001']
    assert check['unknown_ids']==['invented']


def test_http_boundary():
    calls=[]
    def retrieve(q,index,k):
        calls.append(q)
        return HITS
    async def respond(q,hits,s,max_tokens): return {'answer':'人工测试回答','source':'test_fixture'}
    with TestClient(create_app(SETTINGS,retrieve,respond)) as client:
        assert client.get('/health').status_code==200
        assert calls==[]
        assert client.post('/ask',json={'question':'测试'}).status_code==401
        headers={'Authorization':'Bearer app-test'}
        assert client.post('/ask',headers=headers,json={'question':'测试','unexpected':1}).status_code==422
        assert calls==[]
        r=client.post('/ask',headers=headers,json={'question':'测试'})
        assert r.status_code==200
        assert r.json()['retrieved_chunks']==HITS
        assert r.headers['x-request-id']==r.json()['request_id']
        assert calls==['测试']


def test_http_stale_index():
    def retrieve(*args): raise ValueError('stale')
    async def respond(*args): pytest.fail('must not call model')
    with TestClient(create_app(SETTINGS,retrieve,respond)) as client:
        r=client.post('/ask',headers={'Authorization':'Bearer app-test'},json={'question':'测试'})
        assert r.status_code==503
        assert r.json()['error']['code']=='index_not_ready'


def test_http_long_query_does_not_call_answer():
    def retrieve(*args): raise knowledge.EmbeddingInputTooLong('too long')
    async def respond(*args): pytest.fail('must not call model')
    with TestClient(create_app(SETTINGS,retrieve,respond)) as client:
        r=client.post('/ask',headers={'Authorization':'Bearer app-test'},json={'question':'测试'})
        assert r.status_code==422
        assert r.json()['error']['code']=='embedding_input_too_long'
