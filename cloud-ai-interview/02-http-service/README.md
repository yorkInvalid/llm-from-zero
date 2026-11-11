# 第 02 课 把模型接成一个可调用的服务

这一课使用普通 Python 项目：**FastAPI + HTTPX + 终端客户端 + pytest**。打开 [课程讲解与操作](LESSON.md) 开始，不需要 Jupyter。

我们把第一课的“团队资料助手”接成 HTTP 服务，让别的程序只提交问题就能得到回答。重点是接口边界、两套认证、异步等待、流式结束条件和错误定位。完整代码能先运行，课堂再增加一个功能并用测试验证。

## 文件怎么读

| 文件 | 什么时候看 | 负责什么 |
|---|---|---|
| [LESSON.md](LESSON.md) | 首先打开 | 按同一条请求组织的讲解、启动和观察步骤 |
| [service/main.py](service/main.py) | 先看 `/ask` 那十来行，再看流式接口 | 接收请求、检查身份、调用模型、返回响应 |
| [service/contracts.py](service/contracts.py) | 发送空问题或故障实验时 | 输入约束和错误类型 |
| [service/provider.py](service/provider.py) | 看真实请求和流式解析时 | 构造模型请求、读取模型响应、处理连接和完成状态 |
| [service/settings.py](service/settings.py) | 配置服务时 | 从环境读取配置，区分真实模型和教学服务 |
| [client.py](client.py) | 发请求、看时间时 | 独立 HTTP 客户端，不直接连接模型商 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 主链路之后 | 本地故障实验、课堂功能改动和面试追问 |
| [tests/test_service.py](tests/test_service.py) | 改代码前后 | 自动验证接口约定，测试不调用付费模型 |

## 快速启动

在 VS Code 中打开本目录的终端，或从原项目根目录进入：

```bash
cd cloud-ai-interview/02-http-service
uv sync --locked
uv run python configure.py
uv run --env-file .env uvicorn service.main:app --host 127.0.0.1 --port 8000
```

配置脚本让你选择已有 DeepSeek / TokenHub，并隐藏输入模型密钥，保存到本课 `.env`。它另外随机生成一把访问我们服务的 `APP_API_KEY`；这两把密钥用途不同。不会覆盖已有 `.env`，也不发模型请求。`.env` 仅供本机开发，分享项目时不要带上；代码仓库忽略它。

在**第二个同目录终端**中：

```bash
curl -i http://127.0.0.1:8000/health
uv run --env-file .env python client.py
uv run --env-file .env python client.py --stream
```

后两个命令各发起一次真实模型请求，重跑会再次请求；不自动重试。第一条健康检查不调用模型。没准备真实账号时，可先看 LESSON，再按 EXPERIMENTS 使用明确标注的本机教学服务，不会静默切换。

查看接口文档：[本机 Swagger](http://127.0.0.1:8000/docs)。它由 FastAPI 生成；流式效果主要用命令行客户端观察。

若 8000 已被占用，启动参数换成 `--port 18000`，客户端同时加 `--url http://127.0.0.1:18000`，健康检查地址也随之改变。不要为腾出端口结束来源不明的进程。

## 为什么本课有单独的环境

本课有自己的 `pyproject.toml`、`uv.lock` 和 `.venv`。它只安装 HTTP 服务依赖，不需要把原项目的 PyTorch、绘图和 Notebook 依赖一起带上云。这里没有“选内核”步骤；`uv run` 会按当前项目选择环境。第三课直接部署这份服务项目。

## 已实现与适用范围

- `/health` 检查进程能否响应；`/ready` 检查本地必需配置，两者都不验证真实模型可用性。
- `/ask` 返回完整 JSON；`/ask/stream` 返回 `meta / delta / done / error` 事件。
- 应用访问校验、输入限制、连接复用、网络等待与整体时间上限、请求编号、错误分类、日志脱敏和连接清理。
- 默认只监听本机。当前没有多租户权限、数据库、限流或生产网关；第三、四课逐步做部署和访问边界。这个项目采用服务开发方式，不宣称已经满足生产上线的全部条件。

## 本课检查

```bash
uv run --locked pytest -q
```

测试替换模型通信，明确使用人工响应；通过代表程序行为符合测试条件，不代表真实账号可用、模型答对或你已经独立掌握。交付时没有使用你的真实密钥或发起付费请求。

2026-11-03 验证：32 项自动检查通过；另启动两个临时本机服务，验证实际 HTTP、分段到达、上游 401/429、读取超时、流中断、输出截断、客户端断开清理，以及命令行退出状态。验证服务已停止，课件图示已检查，未生成包含真实密钥的 `.env`。

官方参考核对日期 2026-11-03：[FastAPI 流式响应](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)、[HTTPX 异步客户端](https://www.python-httpx.org/async/)、[HTTPX 超时](https://www.python-httpx.org/advanced/timeouts/)、[DeepSeek 参数和响应](https://api-docs.deepseek.com/api/create-chat-completion/)、[TokenHub 入口](https://cloud.tencent.com/document/product/1823/130079)。
