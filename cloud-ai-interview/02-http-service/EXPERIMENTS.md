# 第二课的故障实验与课堂改动

主课已有真实模型路径。这里把**模型商这一段**换成本机教学服务，实际发送 HTTP 请求、产生等待、逐段传输与断流。输出文字由代码写定，明确标记 `local_fixture`；不能据此评测模型回答质量或宣称触发过服务商限流。

## 一 启动可控的本机模型服务

终端 A，在本课目录：

```bash
uv run uvicorn labs.fake_provider:app --host 127.0.0.1 --port 9001
```

终端 B，在同一目录：

```bash
uv run --env-file .env.lab.example uvicorn service.main:app --host 127.0.0.1 --port 8001
```

这个练习入口用 8001，和真实模型服务的 8000 区分。配置明确写了 `LLM_PROVIDER=lab` 和 `ALLOW_LAB_PROVIDER=1`；真实路径失败时没有自动切换。若你的 shell 已导出同名环境变量，它们可能优先于 env 文件，请先核对 `/health` 的来源为 `local_fixture`。

若提示 `Address already in use`，说明端口已有进程监听。可把应用端口 8001 换成 18001，并把客户端 `--url` 同步改掉。教学模型端口也被占用时，把终端 A 的端口改为 19001，在终端 B 的命令前加 `LAB_PROVIDER_PORT=19001`。这两个端口分别对应两段连接，不要只改其中一端。

终端 C，先观察成功流：

```bash
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --stream --raw
```

你应看到 `meta → 多个 delta → done`。要看原始 SSE 文本而不是解析后的显示，可以用公开的本机练习密钥：

```bash
curl -N -i http://127.0.0.1:8001/ask/stream \
  -H 'Authorization: Bearer local-lab-only' \
  -H 'Content-Type: application/json' \
  -d '{"question":"便携显示器借多久？"}'
```

`-N` 让 curl 不把收到的输出积攒起来再显示；它不改变服务器的生成速度。这个固定密钥只适用于本机实验，不能拿去保护公网服务。

## 二 每次只制造一种故障

先预测，再选其中三条运行。不要为凑记录重复调用真实模型。

```bash
# 上游限流：普通接口如何返回？
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --question '[LAB:429]'

# 相同上游限流：流式接口如何返回？
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --question '[LAB:429]' --stream --raw

# 先给一段正文，再关闭上游流。
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --question '[LAB:disconnect]' --stream --raw

# 上游等 2 秒，本实验读取等待上限为 1 秒。
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --question '[LAB:timeout]'

# 模型商认证失败：与调用者的应用认证失败比较。
uv run --env-file .env.lab.example python client.py --url http://127.0.0.1:8001 --question '[LAB:401]'
```

这些命令预期可能以退出码 1 结束，这是客户端识别失败的结果，不代表终端工具坏了。停止实验时，在两个服务终端分别按 `Ctrl+C`。

记录不用写成长文：

| 现象 | 我观察到的 HTTP 状态 | 最后事件/错误码 | 请求编号对应的日志 | 下一步检查 |
|---|---|---|---|---|
| 上游限流，普通接口 | 待观察 | 待观察 | 待观察 | 待解释 |
| 上游限流，流式接口 | 待观察 | 待观察 | 待观察 | 待解释 |
| 流中途结束 | 待观察 | 待观察 | 待观察 | 待解释 |

<details><summary>观察后再核对接口约定</summary>

- 普通接口的上游 429，被本服务翻译为 503 + `upstream_rate_limited`。调用者通过了我们的认证，只是依赖暂时无法服务。
- 流式接口已经发送 HTTP 200，则在正文中出现 `error`，没有 `done`。只统计 HTTP 成功率会漏掉这类失败。
- 上游流中断，即使已经有 `delta`，仍应得到 `incomplete_answer`，不能冒充完成。
- 等待超时是 504 + `upstream_timeout`；错误说明不声称模型商没执行。
- 上游 401 是 502 + `upstream_auth`；应用访问凭据错误才是我们的 401。不同的密钥由不同角色负责修复。

</details>

## 三 做一个真正穿过服务各层的改动

新增 `answer_style`：调用者可以选 `brief`（简短结论）或 `steps`（按步骤解释），默认 `brief`。普通与流式接口都应支持；模型只通过提示收到这个偏好，不保证每次严格照做。

这项改动是本课的核心动手部分。起始服务可直接运行，不会因为练习留空而无法体验；现在请你在现有代码上改，而不是另写一份不相连的演示函数。

验收条件：

1. 不传字段，旧客户端仍正常工作。
2. 传 `steps`，发给模型的消息中出现步骤风格要求。
3. 传 `whatever`，两个接口都在调用模型前返回 422。
4. 普通与流式使用同一份风格映射，避免两边行为漂移。
5. 为第 2、3 条新增测试，使用现有 `httpx.MockTransport` 截获请求，不靠额外付费调用检查消息是否拼对。

API 写法直接给你，不需要从语法猜：

```python
from typing import Literal

# 放进 AskRequest 类：只有这两个值合法，省略时使用 brief。
answer_style: Literal["brief", "steps"] = "brief"
```

接下来沿 `AskRequest → main 中两个接口 → provider 的请求构造` 传递这个值。改变输入对象并不会让下游函数自动知道新字段。

<details><summary>提示 1 先确定每层职责</summary>

输入模型负责合法取值；接口把合法值交给模型通信层；公共的 payload 函数把它映射成提示。客户端不能直接替换完整 system 消息。别在普通和流式方法里分别复制一份风格逻辑。

</details>

<details><summary>提示 2 怎样验证没有偷偷发模型请求</summary>

参考 `tests/test_service.py` 中 `test_invalid_inputs_do_not_call_model`：通信函数只要被调用就令测试失败。检查模型消息时，参考 `test_json_contract_and_server_owned_credentials`：从 `request.content` 解析出 JSON，再检查 messages；这是程序结构检查，不是模型效果评测。

</details>

改动前后都运行：

```bash
uv run --locked pytest -q
```

### 面试追问

任选两个，先用实际结果回答，再讨论边界：

- 给路由加 `async`，为什么仍可能把所有请求卡住？
- 只看 `/health` 返回 200，可以承诺问答服务可用吗？
- 流式请求收到了 80% 的文字，为什么不能标记为任务成功？
- 如果多实例部署，你的应用密钥、资料版本和日志编号需要怎样管理？哪些本课已做到，哪些只是下一步设计？

不用补交单独作业。记录本次改动、一次失败与原因，作为之后模拟面试的材料。
