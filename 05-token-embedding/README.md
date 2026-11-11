# 课程 5 · 文本进入模型：token 账与 embedding

从 [lesson.ipynb](lesson.ipynb) 开始。2026-10-31 按“解决方案岗位面试”的目标改版：保留分词与计数，embedding 讲到能说清机制为止，补上 token 密度、消息格式、合同估算、embedding 占比，以及“模型内部 embedding 表”和“RAG 文本向量”的区别。改版前的完整版本备份在项目 `.backups/2026-10-31-*-before-solution-redesign/`。

你已完成的任务 1（分段/整串计数函数）和第 3 节的记录原样保留，本地进度记录未重置。

## 三站

| 部分 | 要解决的问题 | 预计时间 |
|---|---|---|
| A：分词与 token 账 | 字符、token、ID 的区别；分段计数为什么不能直接相加；中英文、代码、JSON 的 token 密度；消息格式多出的 token；一份合同的 token、窗口和费用估算 | 约 40 分钟（前半已完成） |
| B：编号进入模型 | 查表在做什么；改一行为什么影响多个位置；真实模型里 embedding 参数占多大 | 约 15 分钟 |
| C：两种 embedding | 模型内部的输入表和 RAG 用的文本向量模型有什么不同 | 约 15 分钟 |

必做代码两项（任务 1 计数对照、任务 2 密度函数），两道估算，一次一分钟面试解释。

[optional-deep-dive.ipynb](optional-deep-dive.ipynb) 收着原来偏模型实现的内容：shape 推导、one-hot 与查表等价的代码证明、改表行定位、显式 one-hot 内存账、补齐后的平均。选做，主课和作业都不依赖。

[homework.ipynb](homework.ipynb) 为 10+30+30+30 分：分清对象与形状、上下文预算函数（含输出预留）、新材料的密度与两种方案费用估算、embedding 占比迁移、面试追问。

## 网页试玩

笔记本里的组件能看切分，版式改不大。另有一页本机网页：贴任何一段文字，token 块马上出现，5 题游戏也在这一页。查表、估算和写函数仍在 `lesson.ipynb`。

在项目根目录运行 `uv run python 05-token-embedding/web/server.py`，打开 http://127.0.0.1:8765 。只监听本机，用同一份分词器，最好成绩写入本课 `.learning-progress.json`。不调用付费接口。

## 环境

- 使用已有的 `llm-from-zero` 内核；第一格打印 `sys.executable`。
- 依赖 `tokenizers==0.22.2`、`ipywidgets` 已在项目 uv 环境中。换机器时在项目根目录运行 `uv sync`。
- 分词器丢失时重建：`uv run python 05-token-embedding/tokenizer_tools.py --download`。只下载固定版本的公开分词器与许可证，不下载权重。
- 全课离线，不需要 API key，没有付费调用。

## 真实资料与边界

- 分词器：`Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，11,422,654 字节，SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`，Apache-2.0（[许可证](assets/LICENSE-Qwen.txt)、[来源记录](assets/source.json)）。选它是因为可复现，不声称是最新模型。
- 模型配置（2026-10-31 核对）：Qwen3-0.6B 的 V=151,936、D=1,024、输入表与输出层共用，模型卡总参数 0.6B、不含 embedding 0.44B、原生窗口 32,768；Qwen3-8B 的 D=4,096、不共用，模型卡 8.2B / 6.95B。0.6B 取自固定 revision，8B 取自 `main` 分支。
- 消息格式实验是按 ChatML 公开格式手写的近似，不是官方模板的完整输出；别家服务的格式不同且在服务端完成。
- 本课所有 token 数都是 Qwen 分词器的结果，不是任何厂商的账单；价格都是标明的教学假设。正式报价以目标服务返回的 usage 和计费规则为准。
- 小 embedding 表是随机教学表，没有训练，不代表 Qwen 的向量。

## 交互实验与积分

| 实验 | 操作 | 之后要完成的学习任务 |
|---|---|---|
| 小游戏「猜 token 数」 | 每轮 5 题，从 10 段方案常见材料（合同、客服 FAQ、英文手册、工具返回的 JSON、代码等）随机抽；揭晓后看 token 块、尺子和“1 万字符约多少 token”，一轮结束看各类材料的比例图 | 任务 2 密度函数；第 6 节合同估算 |
| 查表与改行 | 选输入位置看取哪一行；先预测改行影响再揭晓 | 用自己的话解释静态查表与后续上下文 |

2026-10-31 试玩后调整：原来的「切一刀」和「猜 token 数」练的是同一件事（看分词边界），合并为一个游戏，每轮从 10 题减到 5 题；切分边界的结论由任务 1 的计数对照负责。组件和示意图改为跟随编辑器主题的浅色样式，去掉深色“控制室”画面。

积分：分词观察 5、猜数打完一轮 10、查表预测 10、两项必做代码各 20，必做共 65；选做 Notebook 四项各 20。旧版两关（找反例 15、切一刀 10）已有的记录照常保留并计入总分，不再新增。积分不是成绩；代码检查只说明这些用例成立，提示使用情况为自报，估算和解释待讨论。进度保存在本课 `.learning-progress.json`，只在本地；游戏最好成绩也记在这里，旧版 10 题的成绩单独保留，不和 5 题的新成绩混比。

组件显示异常（看到 `VBox(...)` 或空白）时，先确认内核选对并重启后重跑准备格；普通代码与练习仍可继续。

## 文件

- `lesson.ipynb`：主课。
- `optional-deep-dive.ipynb`：选做的底层实现。
- `homework.ipynb`：作业，没有答案。
- `token_games.py`：小游戏；`learning_lab.py`、`lab_visuals.py`：显示与检查；`theme.py`：共用样式；`tests/`：组件行为检查。
- `tokenizer_tools.py`：本地校验与加载。
- `assets/`：分词器、许可证、来源记录与示意图（主课用 `lesson5-axes.svg`、`lesson5-two-embeddings.svg`，选做用 `lesson5-batch.svg`）。

验证记录（2026-10-31 合并游戏后）：组件测试 18 项通过；主课、选做、作业的临时填答副本均执行通过，副本里模拟打完一轮游戏；进度写入临时文件，个人记录未改动。

## 官方参考

- [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[固定版本 config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)
- [Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)、[config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Qwen 字节级 BPE 概念](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#byte-level-byte-pair-encoding)
- [Tokenizers 0.22.2](https://pypi.org/project/tokenizers/0.22.2/#files)、[Tokenizers API 参考](https://huggingface.co/docs/tokenizers/v0.20.3/api/tokenizer)
- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding)
- [Jupyter Widgets](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Basics.html)
