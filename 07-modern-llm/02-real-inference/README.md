# 02 · 真实 Qwen 推理：看输入、看张量、逐 token 生成

打开 [lesson.ipynb](lesson.ipynb)。课件包含两张内嵌图和教师在 Mac MPS 上实际执行的输出，不需要先运行就能阅读。第一课只读配置，本课真正加载固定版本的 Qwen3-0.6B 权重。

## 运行

仓库根目录执行：

```bash
uv sync --extra inference
```

在 Notebook 选择项目 `.venv/bin/python` 对应的 Python 内核，从头运行。已安装后若仍提示找不到 transformers，请重启内核并检查第一格的 Python 路径。

首次下载约 1.2 GB 权重到 Hugging Face 缓存，之后复用。没有 API 调用、无需租 GPU。默认使用 MPS（否则 CUDA/CPU）、float32；参数数据约 2.22 GiB，实际进程占用更大。输入和生成长度默认很短。完整下载后可用 `HF_HUB_OFFLINE=1` 离线运行。

## 看什么

1. 先看模型实际回答，再拆 messages → chat template → Token IDs。
2. 观察首层 Embedding、Q/K/V、SwiGLU、LM head 的真实张量形状。
3. 从最后一个位置的 logits 看下一 token 候选。
4. 不用 `generate`，手动运行 5 步，检查与贪心生成一致。
5. 比较贪心与 temperature/top-k 采样。
6. 观察每层 KV cache 的尺寸与增长，比较完整前缀与缓存路径的 logits。
7. 固定生成步数、预热、设备同步、重复测量，查看本机速度和内存快照。

生成成功不等于回答正确，课件对真实回答中笼统的表述另作解释。推理工具仅支持单条无 padding 输入；批处理不在这一课范围。后续再进入预训练、SFT、RL。

## 文件与验证

- `lesson.ipynb`：主课，实际输出已保存。
- `inference_utils.py`：完整生成循环、采样、缓存一致性检查和计时。
- `assets/execution-report.json`：模型 revision、软件版本、设备、回答、形状和实测结果；是教师验证记录，不是用户学习成绩。
- `tests/test_inference.py`：小随机 Qwen 的算法测试，不冒充真实语言生成。
- `build_lesson.py`：生成课件与示意图，会清空已执行输出，一般学习无需运行。
- `execute_lesson.py`：用项目 `.venv` 内核从头执行并保存输出。

```bash
.venv/bin/python -m unittest discover -s 07-modern-llm/02-real-inference/tests -v
uv run --extra inference --with nbclient --with nbformat python 07-modern-llm/02-real-inference/execute_lesson.py
```

每次执行课件会更新运行记录。测速包含 prefill 与 decode，不包含模型加载和分词；固定 12 步，忽略 EOS 以比较相同工作量，3 次取中位数。MPS 分配快照不是进程 RSS，也不是峰值。短输入的结果不能直接外推到长上下文或云 GPU。
