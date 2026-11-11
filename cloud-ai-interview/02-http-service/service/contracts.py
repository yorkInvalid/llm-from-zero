from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class AskRequest(BaseModel):
    # 服务拒绝空问题、超长问题和拼错的字段；不是直接透传模型商所有参数。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    question: str = Field(min_length=1, max_length=2000)
    max_tokens: int = Field(default=512, ge=1, le=1024)


class AskResponse(BaseModel):
    request_id: str
    source: Literal["live_api", "local_fixture"]
    answer: str
    model: str
    finish_reason: Literal["stop"]
    usage: dict[str, int] | None
    elapsed_ms: float


class ServiceError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def error_from_status(status):
    """模型商的状态，不直接当成调用我们服务的用户犯的错误。"""
    if status in (401, 403):
        return ServiceError(502, "upstream_auth", "模型服务认证失败，请检查服务端配置。")
    if status == 402:
        return ServiceError(503, "upstream_unavailable", "模型未开通或额度不足，请检查服务端账号。")
    if status == 429:
        return ServiceError(503, "upstream_rate_limited", "模型服务限制了请求；本次没有自动重试。")
    return ServiceError(502, "upstream_http_error", "模型服务返回异常状态，请用请求编号定位。")


def incomplete():
    return ServiceError(502, "incomplete_answer", "回答未正常完成，已返回的片段不能作为完整答案。")


def invalid_response():
    return ServiceError(502, "upstream_invalid_response", "模型服务的响应不符合预期协议。")
