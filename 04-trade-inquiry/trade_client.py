"""课程 4 的真实 API 请求；导入、配置都不会发送请求。

官方接口于 2026-10-31 核实：
https://api-docs.deepseek.com/quick_start
https://api-docs.deepseek.com/api/create-chat-completion/
https://cloud.tencent.com/document/product/1823/130079
https://cloud.tencent.com/document/product/1823/132248

TokenHub 在本课只使用其 DeepSeek 系列，以便明确关闭思考模式。
本文件不读取人工回答、不自动重试、不将密钥或请求写入文件。
"""

from dataclasses import dataclass, field
import getpass
import json
import os
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import warnings


@dataclass(frozen=True)
class ModelConfig:
    """配置可直接显示；api_key 不参与 repr，只保存在当前内核内存中。"""

    provider: str
    model: str
    endpoint: str
    api_key: str = field(repr=False)
    max_tokens: int = 2048
    timeout: int = 45


def configure(provider="deepseek", model=None):
    """选择服务并读取密钥，不发请求；缺少密钥时在隐藏输入框中询问。

    DeepSeek: DEEPSEEK_API_KEY，DEEPSEEK_MODEL（默认 deepseek-flash）。
    TokenHub: TOKENHUB_API_KEY，TOKENHUB_MODEL；模型未填时询问。
    TOKENHUB_REGION 可设 guangzhou（默认）或 singapore，需匹配密钥地域。
    本课的 TokenHub 配置对应按量 API，不对应 Token Plan 编程套餐。
    """
    if not isinstance(provider, str) or provider.lower().strip() not in {
        "deepseek", "tokenhub"
    }:
        raise ValueError("provider 请填 'deepseek' 或 'tokenhub'。")
    provider = provider.lower().strip()
    prefix = "DEEPSEEK" if provider == "deepseek" else "TOKENHUB"

    if model is None:
        model = os.environ.get(f"{prefix}_MODEL", "").strip()
    if not model:
        if provider == "deepseek":
            model = "deepseek-flash"
        else:
            try:
                model = input(
                    "填写 TokenHub 控制台中已开通的 DeepSeek 模型 ID\n"
                    "（官方示例：deepseek/deepseek-flash；这不是密钥）："
                ).strip()
            except (EOFError, OSError):
                raise RuntimeError(
                    "当前环境不能交互输入模型名；请在 MODEL 中填写控制台模型 ID，"
                    "或设置 TOKENHUB_MODEL 后重跑配置单元。"
                ) from None
    if not isinstance(model, str) or not model.strip():
        raise ValueError("模型名不能为空，请填写你账号可用的模型 ID。")
    model = model.strip()
    if provider == "tokenhub" and not model.startswith("deepseek"):
        raise ValueError(
            "本课只适配 TokenHub 的 DeepSeek 系列，以明确关闭思考模式。"
            "请从控制台选择已开通的 DeepSeek 模型 ID。"
        )

    if provider == "deepseek":
        endpoint = "https://api.deepseek.com/chat/completions"
    else:
        region = os.environ.get("TOKENHUB_REGION", "guangzhou").strip().lower()
        endpoints = {
            "guangzhou": "https://tokenhub.tencentmaas.com/v1/chat/completions",
            "singapore": "https://tokenhub-intl.tencentmaas.com/v1/chat/completions",
        }
        if region not in endpoints:
            raise ValueError("TOKENHUB_REGION 请填 guangzhou 或 singapore。")
        endpoint = endpoints[region]

    api_key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    if not api_key:
        try:
            # 无法隐藏输入时直接停止，防止 getpass 回退为明文输入。
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                api_key = getpass.getpass(
                    f"输入 {prefix} API key（隐藏输入，仅保留在本次内核内存）："
                ).strip()
        except (getpass.GetPassWarning, EOFError, OSError):
            raise RuntimeError(
                f"当前环境无法隐藏输入。请为启动内核的环境设置 {prefix}_API_KEY，"
                "再重跑配置单元；不要将密钥写进 notebook 代码。"
            ) from None
    if not api_key:
        raise ValueError("没有输入 API key；尚未发送任何请求。")
    if not api_key.isascii() or any(character.isspace() for character in api_key):
        raise ValueError("API key 中出现了空白或非 ASCII 字符，请重新输入完整密钥。")
    return ModelConfig(provider, model, endpoint, api_key)


def call_model(messages, config):
    """发送一次真实请求，返回正文、用量、模型名、结束原因和耗时。

    本课仅支持包含 role/content 的文字消息。每次调用都可能计费；
    网络失败、回答为空或截断时直接报错，由学习者决定下一步。
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("config 应来自 configure()，请先运行配置单元。")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 应是非空消息列表。")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError(
                "每条消息需要 role（system/user/assistant）和非空文字 content。"
                "先完成 TODO，再运行调用单元。"
            )

    payload = {
        "model": config.model,
        "messages": [
            {"role": item["role"], "content": item["content"]} for item in messages
        ],
        "stream": False,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": config.max_tokens,
    }
    request = Request(
        config.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    start = perf_counter()
    try:
        # urlopen 是测试可替换的通信入口；课堂运行时使用真实 HTTPS。
        with urlopen(request, timeout=config.timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        hints = {
            400: "检查模型 ID 以及该模型支持的请求参数。",
            401: "检查密钥及其所属服务和地域。",
            402: "检查额度，以及是否已在控制台开通该模型。",
            403: "检查密钥权限及模型访问权限。",
            404: "检查模型 ID 和接口地址。",
            429: "服务限流，稍后再手动尝试。",
        }
        hint = hints.get(exc.code, "检查服务商状态和账号配置。")
        # 不显示响应正文、请求头和原异常，以免其中带有密钥。
        raise RuntimeError(f"API 返回 HTTP {exc.code}。{hint}本次未自动重试。") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError(
            "API 连接失败或等待超时。本次未自动重试，"
            "请检查网络后决定是否重新运行；超时不代表服务端一定未计费。"
        ) from None
    except (ValueError, UnicodeError):
        raise RuntimeError("API 返回的内容不是有效 JSON；本次未自动重试。") from None
    elapsed = perf_counter() - start

    if not isinstance(result, dict):
        raise RuntimeError("API 响应结构异常：最外层应是对象。")
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("API 没有返回可读取的 choices；请检查服务商响应。")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError(
            "回答因输出长度上限而截断，本次不能作为完整结果。"
            "请先缩短要求的回答篇幅，再决定是否重新调用；本次未自动重试。"
        )
    if finish_reason != "stop":
        raise RuntimeError(
            "模型没有正常完成文字回答（可能被过滤或返回了本课未处理的工具调用）。"
            "本次未自动重试。"
        )
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型没有返回非空正文；本次未自动重试。")
    model = result.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError("API 未返回有效的模型名，无法记录本次结果。")
    raw_usage = result.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        for name in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
        ):
            value = raw_usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                usage[name] = value
    return {
        "text": content,
        "usage": usage,
        "model": model,
        "finish_reason": finish_reason,
        "elapsed_seconds": round(elapsed, 3),
    }
