"""运行时由学习者输入自己的密钥；不会覆盖已有 .env，不发模型请求。"""
import getpass
import os
from pathlib import Path
import secrets
import warnings


def main():
    target = Path(__file__).resolve().parent / ".env"
    if target.exists():
        raise SystemExit(".env 已存在；请自行编辑，不自动覆盖。")
    provider = input("模型服务 deepseek / tokenhub [deepseek]：").strip() or "deepseek"
    if provider not in {"deepseek", "tokenhub"}:
        raise SystemExit("只支持 deepseek 或 tokenhub。")
    model = "deepseek-flash"
    region = "guangzhou"
    if provider == "tokenhub":
        model = input("TokenHub 已开通的 DeepSeek 模型 ID：").strip()
        region = input("站点 guangzhou / singapore [guangzhou]：").strip() or region
        if not model.startswith("deepseek") or region not in {"guangzhou", "singapore"}:
            raise SystemExit("请核对模型 ID 和站点。")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        key = getpass.getpass("模型 API key（隐藏输入）：").strip()
    if not key or not key.isascii() or any(c.isspace() or c in "'\\" for c in key):
        raise SystemExit("密钥为空或含本配置工具不接受的空白/引号/反斜杠。")
    values = {
        "APP_API_KEY": secrets.token_urlsafe(24), "LLM_PROVIDER": provider,
        "LLM_MODEL": model, "LLM_API_KEY": key, "TOKENHUB_REGION": region,
        "LLM_READ_TIMEOUT": "20", "REQUEST_DEADLINE": "45",
    }
    if any("'" in v or "\\" in v or "\n" in v or "\r" in v for v in values.values()):
        raise SystemExit("配置中含不支持的特殊字符。")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("# 仅用于本机开发，不提交或分享。\n")
        stream.writelines(f"{name}='{value}'\n" for name, value in values.items())
    print("已保存本课 .env，权限仅当前用户读写。没有发请求；APP_API_KEY 已单独随机生成。")


if __name__ == "__main__":
    main()
