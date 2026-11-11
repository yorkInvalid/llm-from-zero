import asyncio
from dataclasses import replace
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from service.main import create_app
from service.provider import ModelGateway
from service.settings import Settings

SETTINGS = Settings("lab", "teaching-fixture", "http://127.0.0.1:9001/v1/chat/completions", "test-app-key", "test-llm-key")
HEADERS = {"Authorization": "Bearer test-app-key"}
BODY = {"question": "借几天？"}


def answer(reason="stop", text="人工测试响应", usage=None):
    return {"model": "teaching-fixture", "choices": [{"message": {"content": text}, "finish_reason": reason}], "usage": usage}


def make_client(handler, settings=SETTINGS):
    return TestClient(create_app(settings, transport=httpx.MockTransport(handler)))


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content, self.closed = content, False
    async def __aiter__(self):
        # 任意网络分块：包括拆在中文 UTF-8 字节中间，不等于一个 SSE 事件。
        for offset in range(0, len(self.content), 7):
            yield self.content[offset:offset+7]
    async def aclose(self):
        self.closed = True


def frames(*, finish="stop", done=True, usage_tail=False):
    values = [{"model":"fixture", "choices":[{"delta":{"content":"人工中文片段"},"finish_reason":None}]},
              {"model":"fixture", "choices":[{"delta":{},"finish_reason":finish}]}]
    if usage_tail:
        values.append({"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}})
    text = ": heartbeat\n\n" + "".join("data: " + json.dumps(v,ensure_ascii=False)+"\n\n" for v in values)
    return (text + ("data: [DONE]\n\n" if done else "")).encode()


def test_health_and_config_checks_do_not_call_model():
    def forbidden(request):
        pytest.fail("健康/配置检查不能发模型请求")
    with make_client(forbidden,replace(SETTINGS,llm_key="")) as client:
        assert client.get("/health").json()["model_checked"] is False
        assert client.get("/ready").status_code==503
        assert client.post("/ask",json=BODY,headers=HEADERS).status_code==503


@pytest.mark.parametrize("path",["/ask","/ask/stream"])
def test_auth_rejected_before_model(path):
    with make_client(lambda r:pytest.fail("未授权请求不应花模型费用")) as client:
        response=client.post(path,json=BODY)
        assert response.status_code==401
        assert response.json()["request_id"]==response.headers["x-request-id"]


@pytest.mark.parametrize("body",[{"question":" "},{"question":"x"*2001},{"question":"x","max_tokens":0},
                                 {"question":"x","max_tokens":1025},{"question":"x","extra":1},
                                 {"question":123},{"question":"x","max_tokens":True}])
def test_invalid_inputs_do_not_call_model(body):
    with make_client(lambda r:pytest.fail("非法参数不应调用模型")) as client:
        response=client.post("/ask",json=body,headers=HEADERS)
        assert response.status_code==422
        assert "input" not in response.json()["error"]


def test_json_contract_and_server_owned_credentials(caplog):
    def handler(request):
        assert request.headers["authorization"]=="Bearer test-llm-key"
        body=json.loads(request.content)
        assert "DEV-01" in body["messages"][-1]["content"]
        assert body["stream"] is False
        return httpx.Response(200,json=answer(usage={"prompt_tokens":4,"completion_tokens":2}))
    with make_client(handler) as client:
        response=client.post("/ask",json=BODY,headers=HEADERS)
        assert response.status_code==200
        assert response.json()["source"]=="local_fixture"
        assert response.json()["usage"]["prompt_tokens"]==4
        assert response.headers["x-request-id"]==response.json()["request_id"]
    assert "test-llm-key" not in caplog.text and "test-app-key" not in caplog.text


@pytest.mark.parametrize("upstream,downstream,code",[(401,502,"upstream_auth"),(403,502,"upstream_auth"),
    (402,503,"upstream_unavailable"),(429,503,"upstream_rate_limited"),(500,502,"upstream_http_error")])
def test_status_translation_and_no_retry(upstream,downstream,code):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(upstream,text="DO NOT ECHO test-llm-key")
    with make_client(handler) as client:
        response=client.post("/ask",json=BODY,headers=HEADERS)
        assert response.status_code==downstream and response.json()["error"]["code"]==code
        assert "test-llm-key" not in response.text and len(calls)==1


@pytest.mark.parametrize("body",[answer("length"),answer(text=" "),{},[],{"choices":[]}])
def test_bad_or_incomplete_json_is_not_success(body):
    with make_client(lambda r:httpx.Response(200,json=body)) as client:
        assert client.post("/ask",json=BODY,headers=HEADERS).status_code==502


def test_missing_usage_is_unknown():
    with make_client(lambda r:httpx.Response(200,json=answer())) as client:
        assert client.post("/ask",json=BODY,headers=HEADERS).json()["usage"] is None


@pytest.mark.parametrize("exception,status",[(httpx.ReadTimeout("fixture"),504),(httpx.ConnectError("fixture"),502)])
def test_network_failures(exception,status):
    def handler(request): raise exception
    with make_client(handler) as client:
        assert client.post("/ask",json=BODY,headers=HEADERS).status_code==status


def test_stream_fragmentation_usage_and_close():
    stream=ByteStream(frames(usage_tail=True))
    with make_client(lambda r:httpx.Response(200,stream=stream)) as client:
        response=client.post("/ask/stream",json=BODY,headers=HEADERS)
        assert response.status_code==200
        assert "人工中文片段" in response.text
        assert "event: done" in response.text and "event: error" not in response.text
        assert '"prompt_tokens": 4' in response.text
    assert stream.closed


@pytest.mark.parametrize("content",[frames(done=False),frames(finish="length"),b'data: broken-json\n\n'])
def test_stream_failure_after_headers_is_error_event(content):
    stream=ByteStream(content)
    with make_client(lambda r:httpx.Response(200,stream=stream)) as client:
        response=client.post("/ask/stream",json=BODY,headers=HEADERS)
        assert response.status_code==200
        assert "event: error" in response.text and "event: done" not in response.text
    assert stream.closed


def test_stream_upstream_429_is_not_a_successful_answer():
    with make_client(lambda r:httpx.Response(429)) as client:
        response=client.post("/ask/stream",json=BODY,headers=HEADERS)
        assert response.status_code==200
        assert "upstream_rate_limited" in response.text and "event: done" not in response.text


def test_total_deadline_and_explicit_stream_close():
    async def run():
        async def slow(request):
            await asyncio.sleep(.1)
            return httpx.Response(200,json=answer())
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            gateway=ModelGateway(client,replace(SETTINGS,deadline=.02))
            with pytest.raises(Exception) as exc:
                await gateway.complete("q","rules",10)
            assert exc.value.code=="upstream_timeout"
        stream=ByteStream(frames())
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=stream))) as client:
            generator=ModelGateway(client,SETTINGS).stream("q","rules",10)
            assert (await anext(generator))["event"]=="delta"
            await generator.aclose()
        assert stream.closed
    asyncio.run(run())


def test_response_contract_is_in_openapi():
    with make_client(lambda r:pytest.fail("读取接口定义不应调用模型")) as client:
        schema=client.get("/openapi.json").json()
        assert "AskResponse" in schema["components"]["schemas"]
        assert "HTTPBearer" in schema["components"]["securitySchemes"]


def test_lab_requires_explicit_opt_in_and_only_uses_loopback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER","lab")
    monkeypatch.delenv("ALLOW_LAB_PROVIDER",raising=False)
    with pytest.raises(ValueError): Settings.from_env()
    monkeypatch.setenv("ALLOW_LAB_PROVIDER","1")
    monkeypatch.setenv("LAB_PROVIDER_PORT","19001")
    assert Settings.from_env().endpoint=="http://127.0.0.1:19001/v1/chat/completions"
    monkeypatch.setenv("LAB_PROVIDER_PORT","70000")
    with pytest.raises(ValueError): Settings.from_env()
