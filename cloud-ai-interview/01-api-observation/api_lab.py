"""一次真实 HTTP 请求的观察工具。导入与配置不请求；不自动重试或替换回答。

接口核对日期：2026-11-03。支持 DeepSeek，以及 TokenHub 的 DeepSeek 系列。
课堂只需看 build_payload 和 request_once 中发送请求的几行，其余负责记录。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
from html import escape
import json
import os
import socket
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
import warnings


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    endpoint: str
    api_key: str = field(repr=False)


def configure(provider="deepseek", model=None, region="guangzhou"):
    """只读取配置。隐藏输入密钥或从指定环境变量读取，不读取磁盘上的密钥文件。"""
    if provider not in {"deepseek", "tokenhub"}:
        raise ValueError("provider 请填 deepseek 或 tokenhub。")
    prefix = "DEEPSEEK" if provider == "deepseek" else "TOKENHUB"
    model = model or os.environ.get(prefix + "_MODEL", "").strip()
    if provider == "deepseek":
        model = model or "deepseek-flash"
        endpoint = "https://api.deepseek.com/chat/completions"
    else:
        endpoints = {
            "guangzhou": "https://tokenhub.tencentmaas.com/v1/chat/completions",
            "singapore": "https://tokenhub-intl.tencentmaas.com/v1/chat/completions",
        }
        if region not in endpoints:
            raise ValueError("region 请填 guangzhou 或 singapore，并匹配密钥所属站点。")
        endpoint = endpoints[region]
        if not isinstance(model, str) or not model.startswith("deepseek"):
            raise ValueError("请在 MODEL 填控制台已开通的 DeepSeek 系列模型 ID。")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("模型 ID 不能为空。")
    key = os.environ.get(prefix + "_API_KEY", "").strip()
    if not key:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass(f"{prefix} API key（隐藏输入）：").strip()
        except (getpass.GetPassWarning, EOFError, OSError):
            raise RuntimeError("无法隐藏输入；请在启动内核的环境设置对应 API_KEY。") from None
    if not key or not key.isascii() or any(c.isspace() for c in key):
        raise ValueError("密钥为空或含空白/非 ASCII 字符；尚未发请求。")
    return Config(provider, model.strip(), endpoint, key)


def build_payload(messages, model, max_tokens=512):
    """构造待发送 JSON。两次对照使用同一个模型和参数，只改变是否提供材料。"""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 2048:
        raise ValueError("本实验 max_tokens 请设为 1–2048 的整数。")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 应为非空列表。")
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("每条消息需要合法 role。")
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise ValueError("content 应为非空文字。")
    return {
        "model": model,
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def request_once(config, messages, *, max_tokens=512, timeout=30):
    """发送一次真实请求。返回记录而不隐藏失败；HTTP 200 不自动评为事实正确。

    timeout 是底层网络操作的等待参数，不是严格的端到端总时限。
    elapsed_seconds 是从开始发送到本次读完/出错的客户端耗时，不是首 token 时间。
    """
    if not isinstance(config, Config):
        raise TypeError("请先 configure()。")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout 必须大于 0。")
    payload = build_payload(messages, config.model, max_tokens)
    record = {
        "source": "live_api", "local_request_id": uuid4().hex[:12],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": config.provider, "requested_model": config.model,
        "http_status": None, "response_id": None, "returned_model": None,
        "finish_reason": None, "content": None, "usage": None,
        "error_kind": None, "elapsed_seconds": None,
    }
    request = Request(
        config.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"},
        method="POST",
    )
    start = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            raw = json.load(response)
        if not isinstance(raw, dict):
            raise ValueError("response must be an object")
        record["response_id"] = raw.get("id") if isinstance(raw.get("id"), str) else None
        record["returned_model"] = raw.get("model") if isinstance(raw.get("model"), str) else None
        usage = raw.get("usage")
        if isinstance(usage, dict):
            # 缺失仍为 None，不能把缺失统计当成 0。缓存字段保留，算费时再用。
            record["usage"] = {
                k: v for k, v in usage.items()
                if k in {"prompt_tokens", "completion_tokens", "total_tokens",
                         "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
                and isinstance(v, int) and not isinstance(v, bool) and v >= 0
            } or None
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("missing choices")
        choice = choices[0]
        reason = choice.get("finish_reason")
        record["finish_reason"] = reason if isinstance(reason, str) else None
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("missing message")
        content = message.get("content")
        record["content"] = content if isinstance(content, str) else None
    except HTTPError as exc:
        record["http_status"] = exc.code
        record["error_kind"] = "http_error"
        exc.close()
    except (TimeoutError, socket.timeout):
        record["error_kind"] = "timeout"
    except URLError as exc:
        record["error_kind"] = "timeout" if isinstance(exc.reason, TimeoutError) else "connection_error"
    except OSError:
        record["error_kind"] = "connection_error"
    except (ValueError, UnicodeError):
        record["error_kind"] = "invalid_response"
    finally:
        record["elapsed_seconds"] = round(perf_counter() - start, 3)
    # 不输出原始异常或错误响应正文；即使服务回显密钥，也从记录中移除。
    for key, value in record.items():
        if isinstance(value, str):
            record[key] = value.replace(config.api_key, "[REDACTED]")
    return record


def show_record(record, title="本次请求"):
    """展示真实记录，不展示请求头、不对事实正确性自动打分。"""
    from IPython.display import HTML, display
    fields = [
        ("来源", record.get("source")), ("HTTP 状态", record.get("http_status")),
        ("错误类型", record.get("error_kind")), ("结束原因", record.get("finish_reason")),
        ("返回模型", record.get("returned_model")), ("客户端总耗时 / 秒", record.get("elapsed_seconds")),
        ("token 统计", record.get("usage")), ("本地记录 ID", record.get("local_request_id")),
        ("服务商响应 ID", record.get("response_id")),
    ]
    def fmt(value):
        return "未提供 / 未取得" if value is None else str(value)
    rows = "".join(f"<tr><th style='text-align:left;padding:5px 14px 5px 0'>{escape(k)}</th>"
                   f"<td>{escape(fmt(v))}</td></tr>" for k, v in fields)
    display(HTML(f"<div style='border-left:4px solid #468a91;padding:12px 20px;max-width:850px'>"
                 f"<b>{escape(title)}</b><table>{rows}</table>"
                 f"<p style='white-space:pre-wrap;overflow-wrap:anywhere'>{escape(fmt(record.get('content')))}</p>"
                 "<small>正文是否符合材料，需要另外核对。</small></div>"))


def teaching_cost(record, input_per_million=3, output_per_million=9):
    """仅教学假设费率。不处理真实缓存计费，不返回缺失用量的伪零费用。"""
    usage = record.get("usage") or {}
    values = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
        return None
    return (values[0] * input_per_million + values[1] * output_per_million) / 1_000_000
