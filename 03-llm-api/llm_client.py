"""Small teaching scaffold for one non-streaming DeepSeek Chat API request.

No API request happens at import time. Credentials stay in environment variables.
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_example_response():
    """Read a handwritten fixture. This does not generate an answer."""
    path = Path(__file__).with_name("example_response.json")
    return json.loads(path.read_text())


def call_model(messages):
    """Send exactly one paid-capable request; no automatic retries.

    This adapter targets DeepSeek's documented Chat Completions endpoint.
    Other providers need their own documented parameters, not just a new URL.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError(
            "真 API 模式需要 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL。"
            "先按 README 设置环境；不要把 key 写进 notebook。"
        )
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.load(response)
    except HTTPError as exc:
        # Do not print headers or the raw response body: they may contain secrets.
        raise RuntimeError(
            f"API 返回 HTTP {exc.code}。检查账号、模型权限、额度和请求格式；"
            "本课不自动重试。"
        ) from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError(
            "API 网络连接失败或超时。本次不自动重试；可先使用离线样例继续学习。"
        ) from None
    except (ValueError, UnicodeError):
        raise RuntimeError("API 响应无法解析为 JSON；本次未自动重试。") from None
    if not isinstance(result, dict):
        raise RuntimeError("API 返回结构不符合预期：响应应为对象。")
    return result


def validate_ticket(ticket):
    """Check the small classroom contract, not semantic correctness."""
    if not isinstance(ticket, dict):
        return ["工单草稿必须是一个 JSON 对象"]
    required = {"category", "summary", "missing_fields"}
    issues = []
    if set(ticket) != required:
        issues.append("字段应恰好包含 category、summary、missing_fields")
    category = ticket.get("category")
    if not isinstance(category, str) or category not in {"network", "account", "other"}:
        issues.append("category 必须是 network / account / other 之一")
    summary = ticket.get("summary")
    if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 80:
        issues.append("summary 必须是 1–80 字符的非空字符串")
    missing = ticket.get("missing_fields")
    if not isinstance(missing, list) or any(
        not isinstance(item, str) or not item.strip() for item in missing
    ):
        issues.append("missing_fields 必须是非空字符串组成的列表，可为空列表")
    return issues
