import asyncio
from contextlib import aclosing, asynccontextmanager
import hmac
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx

from .contracts import AskRequest, AskResponse, ServiceError
from .provider import ModelGateway
from .settings import Settings

logger = logging.getLogger("uvicorn.error")
bearer = HTTPBearer(auto_error=False)
RULES_PATH = Path(__file__).resolve().parents[1] / "data" / "device-loan.md"


def log_event(request_id, event, **fields):
    # 只写排错必要字段；不写 key、Authorization、问题全文或模型正文。
    logger.info(json.dumps({"request_id": request_id, "event": event, **fields}, ensure_ascii=False))


def frame(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings=None, *, transport=None):
    settings = settings or Settings.from_env()
    rules = RULES_PATH.read_text(encoding="utf-8")

    @asynccontextmanager
    async def lifespan(app):
        # 进程启动时创建连接池，所有请求复用；退出时关闭。
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(settings.read_timeout, connect=5, pool=5),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=False, transport=transport,
        ) as client:
            app.state.gateway = ModelGateway(client, settings)
            yield

    app = FastAPI(title="团队资料助手 · 课程02", lifespan=lifespan)

    @app.middleware("http")
    async def identify(request, call_next):
        # 自己生成编号，避免把未经检查的外来请求头写入日志。
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Answer-Source"] = settings.source
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        rid = request.state.request_id
        log_event(rid, "request_error", code=exc.code, status=exc.status)
        return JSONResponse(status_code=exc.status, content={
            "request_id": rid, "source": settings.source,
            "error": {"code": exc.code, "message": exc.message},
        })

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        # 不直接回显包含原始输入的整份校验异常。
        return JSONResponse(status_code=422, content={
            "request_id": request.state.request_id, "source": settings.source,
            "error": {"code": "invalid_request", "message": "请检查问题文字、长度和输出上限。",
                      "fields": [list(e["loc"]) for e in exc.errors()]},
        })

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not settings.app_key:
            raise ServiceError(503, "service_not_configured", "服务端尚未配置应用访问密钥。")
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), settings.app_key.encode()):
            raise ServiceError(401, "unauthorized", "缺少或使用了错误的应用访问密钥。")
        if not settings.configured:
            raise ServiceError(503, "model_not_configured", "服务端尚未配置模型访问参数。")

    @app.get("/health")
    async def health():
        return {"status": "alive", "source": settings.source, "model_checked": False}

    @app.get("/ready")
    async def ready():
        # 检查本地配置，不消耗模型请求，也不承诺远端模型现在可用。
        return JSONResponse(status_code=200 if settings.configured else 503, content={
            "configured": settings.configured, "source": settings.source, "model_checked": False,
        })

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(authorize)])
    async def ask(body: AskRequest, request: Request):
        rid, started = request.state.request_id, perf_counter()
        log_event(rid, "model_start", mode="json", source=settings.source)
        result = await app.state.gateway.complete(body.question, rules, body.max_tokens)
        elapsed_ms = round((perf_counter() - started) * 1000, 1)
        log_event(rid, "answer_complete", mode="json", elapsed_ms=elapsed_ms)
        return {"request_id": rid, "source": settings.source, "elapsed_ms": elapsed_ms, **result}

    @app.post("/ask/stream", dependencies=[Depends(authorize)])
    async def ask_stream(body: AskRequest, request: Request):
        rid = request.state.request_id

        async def events():
            started = perf_counter()
            log_event(rid, "model_start", mode="stream", source=settings.source)
            yield frame("meta", {"request_id": rid, "source": settings.source})
            try:
                async with aclosing(app.state.gateway.stream(body.question, rules, body.max_tokens)) as upstream:
                    async for item in upstream:
                        data = {"request_id": rid, **item["data"]}
                        if item["event"] == "done":
                            data["elapsed_ms"] = round((perf_counter() - started) * 1000, 1)
                            log_event(rid, "answer_complete", mode="stream", elapsed_ms=data["elapsed_ms"])
                        yield frame(item["event"], data)
            except ServiceError as exc:
                log_event(rid, "stream_error", code=exc.code)
                # 响应头已发送，不能改 HTTP 状态。客户端必须检查终止事件。
                yield frame("error", {"request_id": rid, "code": exc.code, "message": exc.message})
            except asyncio.CancelledError:
                # 不吞掉客户端断开引发的取消；上游 stream 上下文负责关闭连接。
                log_event(rid, "client_disconnected")
                raise

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    return app


app = create_app()
