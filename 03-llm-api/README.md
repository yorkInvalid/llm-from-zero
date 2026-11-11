# 课程 3 · 帮 IT 同事整理一句报修消息

假设你帮助公司 IT 同事登记员工的报修消息。手里有一条员工原话、能运行 Python 的电脑，以及本课自己规定的三项登记要求：问题类别、问题简介、还要追问什么。课程最后把这三项显示出来，供人核对。

这一课还没有连接公司工单系统。先学怎样把一句话整理成记录，后面的课再加保存、查询和处理动作。

## 从哪里开始

打开 [lesson.ipynb](lesson.ipynb)，从最上方的故事与“我们手里有什么”开始。选你界面上的 **`llm-from-zero` 内核**；第一格的 `sys.executable` 可以核对它使用的 Python。项目环境在 `llm-from-zero/.venv/bin/python`，内核显示名称不必叫 `.venv`。

新版按这个顺序推进：

1. 把员工原话存下来，先看人工登记会填成什么。
2. 试一下“看到 VPN 就归网络类”，观察两种不同说法让它判断错。
3. 说明上一课的小模型缺什么，再引入已经训练好的模型。
4. 为了让程序分别拿到三项信息，约定 JSON 格式，再写任务说明。
5. 把任务说明和员工原话发出去，分清真实回答与人工样例的来历。
6. 从返回结果拿出正文，读成 Python 字典。
7. 检查填写内容，打印下一步还需要问什么。
8. 加做：读取用量，算一次费用。

课堂有 **3 处必做填空、1 处费用加做填空**。可以分两次学：先到“把消息送给模型”，再从“拿出回答正文”继续。实际代码运行很快，阅读和讨论可以按你的节奏来。

## 文件各做什么

- [lesson.ipynb](lesson.ipynb)：讲解、小实验与填空。
- [homework.ipynb](homework.ipynb)：1 简单 + 2 中等 + 1 思考题，共 100 分。
- [llm_client.py](llm_client.py)：已经写好的发送请求、读取人工样例、检查登记内容的函数，课上会说明各自输入和输出。
- [example_response.json](example_response.json)：给这条 VPN 消息写好的人工返回样例，内容与用量数字都不是模型实测。

`USE_LIVE_API = False` 时读取人工样例，离线可以运行。修改问题不会改变样例，它只用于练习后半段代码。配置账号后改成 `True`，才会将问题真正发给模型；每运行调用那格一次，就请求一次。

## 可选：接入真实 API

本课提供 DeepSeek 的一个具体实现，便于看清请求和返回；这不是模型排名或采购推荐。其他厂商需要按其文档调整参数，不能保证只改网址就通用。

如要接入，在 VS Code 的终端设置环境，再从同一终端启动 Jupyter 或重新启动 VS Code，使 notebook 内核继承变量。**不要把 key 写进 notebook 或聊天**。macOS / zsh 可用隐藏输入：

```zsh
read -rs 'DEEPSEEK_API_KEY?输入 API key（不回显）：'
export DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_MODEL='deepseek-flash'
```

`deepseek-flash` 来自 2026-10-30 核实的[官方入门文档](https://api-docs.deepseek.com/quick_start)，模型名和可用性以后可能变化。执行时以你账号可用的模型和文档为准。本课使用非思考模式、JSON 输出和非流式响应；模型需支持这些设置。

如果 VS Code 已经打开，在它的子终端 `export` 不会反向修改已启动的内核环境。最直接的路径是在该终端运行 `uv run --with jupyterlab jupyter lab`，在打开的浏览器页面选择项目 Python 内核；这只临时补运行工具，不改项目依赖。

本课 `max_tokens=512` 只限制单次输出，不是完整人民币预算。先看服务价格和自己的额度，选择要使用的服务后再切换真 API。交付时已验证离线全流程与发送请求函数在模拟响应下的处理；没有调用付费服务。

## 学完能说清什么

- 能说清开始时有什么、登记三项信息是为了什么。
- 能说明 messages 是准备发出的消息，response 是完整返回结果，正文只是其中一部分。
- 能把 JSON 正文变成 Python 字典，按约定检查字段。
- 能区分“合法 JSON”“填写符合要求”“内容正确”“实际执行成功”。
- 能读取 usage，并按假设单价计算一次调用费用。
- 能解释：生成草稿只是流程的一步，后续还需工具才能执行动作。

## 官方资料

- [请求、响应与参数](https://api-docs.deepseek.com/api/create-chat-completion/)
- [JSON Output：仍要在提示中要求 JSON，检查空结果与截断](https://api-docs.deepseek.com/guides/json_mode/)
- [首次 API 调用](https://api-docs.deepseek.com/quick_start)

下一课：准备几种不同的报修说法，逐条记录模型哪里分对、哪里分错；之后再接上查询、保存记录的功能。
