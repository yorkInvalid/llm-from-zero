# 10 · 一句话的奖励，怎样更新每个 token？

打开 [主课 lesson.ipynb](lesson.ipynb)，先看第 1～3 步。沿用上一课的采样、奖励、优势和 SGD，扩展成逐 token 生成的序列。只有普通顺序代码与两个小函数。

## 本地运行

主课只需要现有项目 .venv 的 PyTorch、Matplotlib，在 CPU 上运行。选择项目 Python 内核，从头逐格执行。首次获取仓库可以在根目录运行 `uv sync --extra inference`。

[Qwen token 对照 qwen_trace.ipynb](qwen_trace.ipynb) 使用本机已缓存的 Qwen3-0.6B，优先使用 Mac MPS，也可用 CUDA 或 CPU。代码设置 local_files_only，不会自动下载。未缓存时可以先在根目录执行 `uv run --extra inference python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B")'`。需要约 1.5 GB 权重空间。

## 学什么

- 条件概率连乘，log 概率相加。
- 整句只打一次分，优势如何进入每个实际生成 token 的 loss。
- 结束符为什么参与求导，为什么不能计算 padding 或结束后的 token。
- 用手算核对逐 token 梯度，再对多个回答平均并更新一次。
- Transformer 每一步共用权重，梯度如何累加。
- 整句评分的归因局限，以及截断输出与完整回答的区别。

## 验证结果与范围

主课是只有 12 个参数、只看上一个 token 的教学模型，不是 Transformer。固定题目“2+2”没有作为文字输入。奖励要求输出恰好是 4、EOS，不能据此衡量新题泛化。

本机真实执行：手算梯度与 backward 一致，SGD 更新与公式一致。每次重新采样 32 个完整回答，120 次更新后，P(4, EOS) 从 0.1111 变为约 0.9459，两个条件概率约为 0.9725、0.9726。实际图表与执行输出保存在 Notebook。

Qwen 对照只执行前向、采样和评分，没有更新 Qwen 权重；逐 token 概率来自实际前向，不是预写结果。本机实测回答“2”并结束，得到 0 分；保留错误输出，展示高生成概率不等于正确。没有启用 top-k/top-p 或温度缩放。最长生成 8 个 token，不保证任何机器上逐 token 一致。

这节是带固定基准的基础 REINFORCE；PPO、GRPO、KL 和价值网络留到后续。参考：[策略梯度推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。
