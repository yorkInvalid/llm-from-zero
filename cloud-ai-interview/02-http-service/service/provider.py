"""只负责模型通信和协议。不读取用户身份，不自动重试，不降级为假回答。"""
import asyncio
from contextlib import asynccontextmanager
import json

import httpx

from .contracts import ServiceError, error_from_status, incomplete, invalid_response


def token_usage(value):
    if not isinstance(value, dict):
        return None
    keys = {"prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
    return {k: v for k, v in value.items() if k in keys and type(v) is int and v >= 0} or None


async def sse_data(lines):
    """按空行分事件；一个网络块、一个 SSE 事件和一个 token 不是同一概念。"""
    data = []
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        # SSE 注释心跳、其他字段不属于 data，不能当成 JSON 解析。
    if data:
        yield "\n".join(data)


@asynccontextmanager
async def network_errors():
    try:
        yield
    except (httpx.TimeoutException, TimeoutError):
        raise ServiceError(504, "upstream_timeout", "等待模型超时；不能据此认定服务商未处理或未计费。") from None
    except httpx.RequestError:
        raise ServiceError(502, "upstream_connection", "无法完成到模型服务的网络通信。") from None


class ModelGateway:
    def __init__(self, client, settings):
        self.client, self.settings = client, settings

    def payload(self, question, rules, max_tokens, stream):
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": "按提供的团队规则回答并注明来源。资料没有说明时不要猜测。请在150字以内回答。"},
                {"role": "user", "content": f"参考材料：\n{rules}\n\n问题：{question}"},
            ],
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": max_tokens, "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def headers(self):
        return {"Authorization": f"Bearer {self.settings.llm_key}"}

    async def complete(self, question, rules, max_tokens):
        async with network_errors(), asyncio.timeout(self.settings.deadline):
            response = await self.client.post(
                self.settings.endpoint, headers=self.headers(),
                json=self.payload(question, rules, max_tokens, False),
            )
        if response.status_code != 200:
            raise error_from_status(response.status_code)
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
            reason = choice["finish_reason"]
            model = body["model"]
            if not isinstance(text, str) or not isinstance(model, str):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            raise invalid_response() from None
        if reason != "stop" or not text.strip():
            raise incomplete()
        return {"answer": text, "model": model, "finish_reason": reason,
                "usage": token_usage(body.get("usage"))}

    async def stream(self, question, rules, max_tokens):
        finish, usage, has_text, model = None, None, False, self.settings.model
        async with network_errors(), asyncio.timeout(self.settings.deadline):
            async with self.client.stream(
                "POST", self.settings.endpoint, headers=self.headers(),
                json=self.payload(question, rules, max_tokens, True),
            ) as response:
                if response.status_code != 200:
                    raise error_from_status(response.status_code)
                async for data in sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        if finish != "stop" or not has_text:
                            raise incomplete()
                        yield {"event": "done", "data": {"model": model, "finish_reason": finish, "usage": usage}}
                        return
                    try:
                        chunk = json.loads(data)
                        if not isinstance(chunk, dict) or "error" in chunk:
                            raise ValueError()
                        if isinstance(chunk.get("model"), str):
                            model = chunk["model"]
                        if isinstance(chunk.get("usage"), dict):
                            usage = token_usage(chunk["usage"])
                        choices = chunk.get("choices")
                        if not isinstance(choices, list):
                            raise ValueError()
                        if not choices:  # 部分兼容服务有单独 usage 事件。
                            continue
                        choice = choices[0]
                        reason = choice.get("finish_reason")
                        delta = choice.get("delta", {})
                        text = delta.get("content")
                        if text is not None and not isinstance(text, str):
                            raise ValueError()
                    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                        raise invalid_response() from None
                    if reason is not None:
                        finish = reason
                        if finish != "stop":
                            raise incomplete()
                    if text:
                        has_text = has_text or bool(text.strip())
                        yield {"event": "delta", "data": {"text": text}}
        # 即使已见到 finish_reason，也要求完整协议终止标记，不能把断线算成功。
        raise incomplete()
