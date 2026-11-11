"""最后一段才打开：把检索插入已经熟悉的 POST /ask。"""
import asyncio
import hmac
import os
from time import perf_counter
from uuid import uuid4
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from answer import Settings, ServiceError, generate
from service.contracts import AskRequest
from knowledge import ROOT, search, EmbeddingInputTooLong


def create_app(settings=None, retrieve=search, respond=generate):
    settings = settings or Settings.from_env()
    app = FastAPI(title="课程03 · Datawhale 资料问答")
    bearer = HTTPBearer(auto_error=False)
    index = ROOT / "artifacts" / os.getenv("RAG_INDEX", "first")
    # 课堂服务一次执行一条 RAG，避免把 CPU 检索并发误当成模型并发。
    gate = asyncio.Semaphore(1)

    @app.middleware("http")
    async def identify(request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ServiceError)
    async def provider_error(request, exc):
        return JSONResponse(status_code=exc.status, content={"request_id": request.state.request_id,
            "error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"error": "请检查问题类型、长度、max_tokens 和未知字段。"})

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not settings.configured or settings.provider not in {"deepseek", "tokenhub"}:
            raise ServiceError(503, "not_configured", "请检查应用与真实模型配置。")
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), settings.app_key.encode()):
            raise ServiceError(401, "unauthorized", "应用访问密钥不正确。")

    @app.get("/health")
    async def health():
        return {"status": "alive", "index_manifest_exists": (index / "manifest.json").is_file(),
                "model_checked": False}

    @app.post("/ask", dependencies=[Depends(authorize)])
    async def ask(body: AskRequest, request: Request):
        started = perf_counter()
        async with gate:
            try:
                # ← 相比第二课新增：先找片段，再把它们交给回答模型。
                hits = await asyncio.to_thread(retrieve, body.question, index, 3)
            except EmbeddingInputTooLong:
                raise ServiceError(422, "embedding_input_too_long", "问题超过向量模型的输入上限，请缩短问题。") from None
            except (FileNotFoundError, ValueError):
                raise ServiceError(503, "index_not_ready", "索引缺失或资料已变化，请检查并重新入库。") from None
            result = await respond(body.question, hits, settings, body.max_tokens)
        elapsed = round((perf_counter() - started) * 1000, 1)
        print(f"request_id={request.state.request_id} elapsed_ms={elapsed} hits={len(hits)}", flush=True)
        return {"request_id": request.state.request_id, "elapsed_ms": elapsed, **result, "retrieved_chunks": hits}

    return app


app = create_app()
