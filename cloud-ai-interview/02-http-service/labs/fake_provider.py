"""本机模型服务替身，不是模型。让 HTTP 失败和流中断可以稳定复现。"""
import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="教学模型替身 · 绝非真实模型")
TEXT = ["【人工响应，用于通信实验】", "便携显示器最多借7个自然日。", "续借需要申请，", "提交不等于批准。[DEV-01]"]
USAGE = {"prompt_tokens": 180, "completion_tokens": 40, "total_tokens": 220}


def data_frame(data):
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


@app.get("/health")
async def health():
    return {"source": "local_fixture"}


@app.post("/v1/chat/completions")
async def complete(request: Request):
    body = await request.json()
    question = body["messages"][-1]["content"]
    if "[LAB:429]" in question:
        return JSONResponse(status_code=429, content={"error": "teaching rate limit"})
    if "[LAB:401]" in question:
        return JSONResponse(status_code=401, content={"error": "teaching upstream auth error"})
    if "[LAB:timeout]" in question:
        await asyncio.sleep(2)  # 比 .env.lab.example 中 1 秒的 read timeout 更长。
    truncated = "[LAB:length]" in question
    if not body.get("stream"):
        return {"model": "teaching-fixture", "usage": USAGE,
                "choices": [{"message": {"content": "".join(TEXT)}, "finish_reason": "length" if truncated else "stop"}]}

    async def generate():
        for index, text in enumerate(TEXT):
            await asyncio.sleep(0.18)
            yield data_frame({"model": "teaching-fixture", "choices": [{"delta": {"content": text}, "finish_reason": None}]})
            if index == 0 and "[LAB:disconnect]" in question:
                return  # 真实地关闭 HTTP 流，不发 finish_reason 或 [DONE]。
        yield data_frame({"model": "teaching-fixture", "usage": USAGE,
                          "choices": [{"delta": {}, "finish_reason": "length" if truncated else "stop"}]})
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
