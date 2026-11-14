# 03 · 预训练与继续预训练

打开 [lesson.ipynb](lesson.ipynb)：先看流程图，再看真实文本、labels、梯度、参数变化和训练曲线。已保存教师运行输出，不需要先执行就能阅读。

本课使用 **125,312 参数的随机初始化 Qwen3 小配置**，不是官方 Qwen3-0.6B 权重。先在书籍文本 A 上做 200 步全参数预训练，再加载模型 A，在另一批文本 B 上做 100 步继续预训练。默认 CPU、float32、2 线程，无下载、GPU 租赁或 API 费用。

## 运行

在仓库根目录执行：

```bash
uv sync --extra inference
```

选择项目 `.venv/bin/python` 对应的 Notebook 内核，从头运行。分词器、固定语料片段已保存在 `assets/`。训练输出保存到本地 `artifacts/`，不提交权重和 Adam 状态。

## 重点观察

1. 没有人工标注，原文的下一个 token 就是监督目标。
2. Hugging Face 的 CausalLM 内部会错开 labels；不要再手动错开一次。
3. padding 的 `attention_mask` 与 labels 中的 `-100` 分别控制注意力和 loss。
4. `backward()` 产生梯度，`optimizer.step()` 才更新参数；Embedding、Wq、RMSNorm 都能看到变化。
5. 继续预训练用已有权重学习新文本；断点恢复要恢复同一次训练的 Adam、进度和随机状态。
6. 验证 loss、PPL 与续写分别说明什么；测试集只在末尾评估，不用于选择训练步数。

小型实验的验证 loss 下降不代表拥有通用语言能力。实际续写中仍有重复和不通顺，课件保留原始结果。本课 A/B 是同一本书的不同段落，不作为跨领域迁移或遗忘评测。

## 数据与实现

- `assets/corpus.json`：Project Gutenberg《Alice’s Adventures in Wonderland》英语原文的固定片段，段落集合互不重叠。
- `assets/source.json`：来源、原文 SHA256、选择规则、语料和分词器 SHA256。
- `assets/tokenizer.json`：只用 A 训练的 512 词表 ByteLevel BPE，后续冻结。
- `training_lab.py`：切窗口、组 batch、训练、评估、保存与恢复。
- `tests/test_training.py`：标签错位、padding、因果性、数据隔离、梯度更新与恢复一致性测试。
- `assets/execution-report.json`：真实验证记录，不代表用户完成本课。
- `build_lesson.py`：重新生成课件源文件与示意图，会清空运行输出，日常学习不必执行。
- `execute_lesson.py`：在项目内核执行课件，保存输出。

```bash
.venv/bin/python -m unittest discover -s 07-modern-llm/03-pretraining/tests -v
uv run --extra inference --with nbclient --with nbformat python 07-modern-llm/03-pretraining/execute_lesson.py
```

本实验的精确恢复限定 CPU、单进程、固定语料、无 dropout 和常数学习率。实际多卡、混合精度、scheduler、DataLoader worker 等场景还需相应状态管理。
