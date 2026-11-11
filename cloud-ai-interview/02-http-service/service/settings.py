from dataclasses import dataclass, field
import math
import os


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    endpoint: str
    app_key: str = field(repr=False)
    llm_key: str = field(repr=False)
    read_timeout: float = 20
    deadline: float = 45

    @property
    def source(self):
        return "local_fixture" if self.provider == "lab" else "live_api"

    @property
    def configured(self):
        return bool(self.app_key and self.llm_key and self.model)

    @classmethod
    def from_env(cls):
        provider = os.getenv("LLM_PROVIDER", "deepseek")
        model = os.getenv("LLM_MODEL", "")
        if provider == "deepseek":
            endpoint = "https://api.deepseek.com/chat/completions"
            model = model or "deepseek-flash"
        elif provider == "tokenhub":
            region = os.getenv("TOKENHUB_REGION", "guangzhou")
            endpoints = {
                "guangzhou": "https://tokenhub.tencentmaas.com/v1/chat/completions",
                "singapore": "https://tokenhub-intl.tencentmaas.com/v1/chat/completions",
            }
            if region not in endpoints or not model.startswith("deepseek"):
                raise ValueError("TokenHub 请配置正确地域和已开通的 DeepSeek 系列模型 ID。")
            endpoint = endpoints[region]
        elif provider == "lab" and os.getenv("ALLOW_LAB_PROVIDER") == "1":
            port = int(os.getenv("LAB_PROVIDER_PORT", "9001"))
            if not 1 <= port <= 65535:
                raise ValueError("LAB_PROVIDER_PORT 应在 1–65535 之间。")
            endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
            model = "teaching-fixture"
        else:
            raise ValueError("LLM_PROVIDER 无效，或未明确启用本机教学服务。")
        read_timeout = float(os.getenv("LLM_READ_TIMEOUT", "20"))
        deadline = float(os.getenv("REQUEST_DEADLINE", "45"))
        if not all(math.isfinite(x) and x > 0 for x in (read_timeout, deadline)):
            raise ValueError("超时时间必须是有限的正数。")
        return cls(provider, model, endpoint, os.getenv("APP_API_KEY", ""),
                   os.getenv("LLM_API_KEY", ""), read_timeout, deadline)
