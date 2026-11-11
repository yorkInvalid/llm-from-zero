> **已暂停的外贸课程草稿。** 用户已改为通用技术面试路线；当前课程 4 是 [真实模型调用与对话历史](../04-model-api/README.md)。本目录保留，不继续教学，也未进行真实模型付费测试。

# 课程 4 · 让真实模型处理第一封外贸询盘

你为虚构的“山行杯业”开发外贸接单助手。它有 8 款保温杯、库存与定制规则，收到海外买家的询价。今天希望助手交付：有依据的初步接单建议、待确认事项和英文回复草稿。

**业务资料与询盘是模拟的，回答由真实模型生成。** 本课没有人工回答模式，也未连接国际站后台。完整路线是：只给询盘 → 加入工厂资料 → 改成 10 天到货，观察模型如何调整判断。

## 从这里开始

打开 [lesson.ipynb](lesson.ipynb)，选择界面上的 `llm-from-zero` 内核，从第一格开始。第一格会打印实际 Python 路径，内核名称不必叫 `.venv`。可以从项目根目录或课程目录运行。

主实验有 3 次真实请求，每格调用一次，不自动重试。阅读和核对不产生请求；重新运行调用格会再次请求。课堂有 2 处代码填空、1 处人工核对表；约 20–30 分钟完成代码实验，讲解与讨论另计。

已有 `ipykernel` 环境足够，发送请求的代码使用 Python 标准库，无需安装新的模型 SDK。

## 选择你已有的服务

笔记本配置格如下：

```python
PROVIDER = "deepseek"  # 或 "tokenhub"
MODEL = None
config = configure(provider=PROVIDER, model=MODEL)
print(config)
```

`configure()` 不发请求。未配置环境变量时，它会隐藏输入 API key；按回车后仅保留在当前 Python 内核内存中，不写入文件。`print(config)` 不包含密钥。不要把 key 写进代码、输出或聊天。

| 服务 | 模型名 | 密钥与地域 |
|---|---|---|
| DeepSeek | `MODEL` 可指定；否则读取 `DEEPSEEK_MODEL`，未设置则用官方当前示例 `deepseek-flash` | `DEEPSEEK_API_KEY`；没有时隐藏输入 |
| 腾讯云 TokenHub | 从控制台复制已开通的 DeepSeek 系列模型 ID，填入 `MODEL`；也可设置 `TOKENHUB_MODEL`，都未提供时会提示输入。官方示例为 `deepseek/deepseek-flash` | `TOKENHUB_API_KEY`；没有时隐藏输入。默认广州；新加坡设 `TOKENHUB_REGION=singapore`，密钥需要匹配地域 |

本课的 TokenHub 接入对应**按量 API**，只适配已核实非思考参数的 DeepSeek 系列，不对应 Token Plan 编程套餐，也不保证其他系列模型接受相同参数。模型 ID 以你的账号实际可用项为准。

若使用新加坡地域，可以在配置格前加入下面两行，再运行配置格。这里只有地域，没有密钥：

```python
import os
os.environ["TOKENHUB_REGION"] = "singapore"
```

官方信息核实日期：**2026-10-31**。DeepSeek 入口是 `https://api.deepseek.com/chat/completions`；TokenHub 广州入口是 `https://tokenhub.tencentmaas.com/v1/chat/completions`，新加坡入口是 `https://tokenhub-intl.tencentmaas.com/v1/chat/completions`。工具根据服务与地域选择入口，无需手动拼接。

两家均使用非流式、非思考请求。每次输出上限为 2048 tokens、等待上限 45 秒。输出上限不等于实际用量，也不是总费用预算；计费以所选服务的模型规则为准。

## 连接遇到问题

- **隐藏输入框**：点输入框、粘贴 key、回车；没有字符显示是正常的。若运行环境无法隐藏输入，程序会停止，避免明文回显。可改用能交互输入的 Jupyter/VS Code，或让启动内核的进程继承对应环境变量。
- **HTTP 401 / 403**：检查 key 是否属于所选服务、地域，以及模型权限。
- **HTTP 400 / 404**：检查模型 ID 和所选模型支持的参数。
- **额度不足 / 限流 / 超时**：先看服务商控制台和网络，再决定是否手动重跑。超时不意味着服务端一定没处理或没计费。
- **回答被截断**：本次结果不能当完整回复；先缩短提示词要求的回答篇幅，再手动请求。工具不会静默重试。

在已经打开的 VS Code 子终端里 `export`，不会自动改变已经运行的内核。隐藏输入通常最省事。若选择环境变量，请在设置变量的终端中启动 Jupyter，例如 `uv run --with jupyterlab jupyter lab`，再选择项目内核。

## 文件各做什么

- [lesson.ipynb](lesson.ipynb)：故事、三次真实请求、观察与填空。
- [homework.ipynb](homework.ipynb)：1 简单、2 中等、1 思考题，共 100 分。复用课堂回答，并增加 1 次真实请求。
- [data/factory_brief.md](data/factory_brief.md)：8 款商品、价格、库存、定制和交期规则；所有数字均是课程假设。
- [data/inquiries.json](data/inquiries.json)：初始、加急和作业询盘，没有预制模型回答。
- [trade_client.py](trade_client.py)：配置账号、发送一次请求、取出正文和用量。返回的 `text` 是工具整理出的字段，原始 API 正文仍在 `choices[0].message.content`。

本课按同一份库存快照开展实验，不会扣库存、发邮件或创建订单。模拟数据能用于核对功能，尚不能证明真实商家需求或商业收益。

## 怎么判断学会了

你能实际获得三次模型回答，指出：第一次缺哪些工厂信息；第二次哪些结论有资料支持；第三次是否识别 10 天到货与加工、运输规则的冲突。看到错误时，能引用原句和资料解释错误，而不是仅凭文风好坏评价。

制作时已验证笔记本格式、代码和模拟传输下的完整运行。**当前环境没有 API 凭证，尚未实际调用两家模型，回答质量待你首次运行时观察。** 验证用的人工响应仅放在临时检查脚本里，没有写进课堂作为可选答案。

## 官方资料

- [DeepSeek 首次调用与模型名](https://api-docs.deepseek.com/quick_start)
- [DeepSeek 请求与响应](https://api-docs.deepseek.com/api/create-chat-completion/)
- [TokenHub API 使用说明](https://cloud.tencent.com/document/product/1823/130078)
- [TokenHub 语言模型调用与地域入口](https://cloud.tencent.com/document/product/1823/130079)
- [TokenHub DeepSeek 模型指南](https://cloud.tencent.com/document/product/1823/132248)

下一课：商品和资料变多以后，怎样先找到回答当前问题真正需要的那几条资料。
