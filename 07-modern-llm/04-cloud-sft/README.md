# 04 · 云 GPU 上的第一次 SFT（AutoDL）

先打开 [lesson.ipynb](lesson.ipynb) 看输入、labels 和训练目标；准备上云时按 [CLOUD_RUN.md](CLOUD_RUN.md) 操作。用户已有 AutoDL 账户，本课不自动创建付费实例。

任务是从合成采购文本抽取四个字段并输出严格 JSON。使用真实 Qwen3-4B，提供全参数 SFT 与 LoRA SFT 配置；先两步试跑，再根据实测选择正式实验。面向单张 A800 80GB：LoRA 基座 BF16，全参数权重/梯度/Adam 状态 FP32，前向使用 BF16 混合精度；均开启梯度检查点，不使用量化。

## 本地准备

仓库根目录：

```bash
uv sync --extra sft
.venv/bin/python 07-modern-llm/04-cloud-sft/run.py inspect
.venv/bin/python -m unittest discover -s 07-modern-llm/04-cloud-sft/tests -v
.venv/bin/python 07-modern-llm/04-cloud-sft/package_lesson.py
```

生成 `dist/autodl-sft-qwen3-4b-a800.zip`，上传到 AutoDL `/root/autodl-tmp/`，解压后目录为 `cloud-sft-a800/`。包里有代码、数据、配置、课件和指南，不含权重、训练结果、环境或账号信息。云端请使用 `requirements-cloud.txt`，不要直接同步仓库的 Mac 环境。

非交互 SSH 的 PATH 可能没有 `python`；本次镜像解释器为 `/root/miniconda3/bin/python`。先按 [环境准备步骤](CLOUD_RUN.md#3-创建环境和模型缓存) 检查解释器，再创建 `--system-site-packages` venv、安装依赖并运行 `pip check` 和 doctor。本次实例有 30 GB 系统盘、50 GB 数据盘；课件、环境、模型缓存和结果放在 `/root/autodl-tmp` 数据盘，分别用 `df -h / /root/autodl-tmp` 检查剩余空间。

## 课程与代码

| 文件 | 用途 |
|---|---|
| `lesson.ipynb` | 图示、真实模板、loss mask、参数更新方式、严格评分、本地验证输出 |
| `CLOUD_RUN.md` | 选卡、环境、试跑、正式训练、TensorBoard、恢复、下载与关机 |
| `make_data.py`、`data/` | 可重复生成的合成样本及来源/哈希清单 |
| `sft_data.py` | 模板边界、回答 mask、padding、严格 JSON/字段评分 |
| `run.py` | doctor / inspect / train / eval / compare 命令 |
| `configs/lora.json` | rank 16 / alpha 32，七类投影层，60 步，微批量 4，累积 4，学习率 1e-4 |
| `configs/full.json` | 全参数，60 步，微批量 1，累积 16，学习率 1e-5 |
| `export_run.py` | 打包指定运行、checkpoint、日志、报告及复现脚本 |
| `assets/local-verification.json` | 当前 4B 分词器与算法检查，明确区分 CUDA 未执行 |

两份配置学习率不同，不构成严格方法对比。每步按有效回答 token 数加权累积梯度，避免长度不同造成错误权重。训练 checkpoint 每 60 步及最后一步保存，含 Adam、随机状态、进度与数据哈希。推理重载时必须使用 checkpoint；LoRA 的 adapter 需要搭配固定版本基座。

## 数据与评估范围

160 条训练、32 条验证、32 条测试，全部明确标注为合成教学数据。先分场景组再改写，同组不跨集合；验证/测试句式另外保留。来源和生成种子见 `data/manifest.json`。样本实际长度以 inspect 输出为准，max_length=512 作为检查上限，batch 实际按最长样本动态 padding。没有调用外部模型生成数据或使用真实客户信息。

监督目标为 JSON 正文与 EOS，system/user/assistant 前缀和 PAD 对应的 labels 为 -100。HF 内部自动错位，不能再手动错位；prompt 仍作为上下文参与计算。过长样本直接报错。

评分同时报告 JSON 合法率、schema 合规率、字段准确率和整条准确率；不自动修复输出。测试集不参与本地集成检查或参数选择。

## 已验证与待验证

2026-11-14 已在单张 A800-SXM4-80GB（Python 3.12.3 / PyTorch 2.8.0+cu128）完成：9 项测试、2 步 LoRA 试跑、checkpoint 重载、正式 60 步训练与独立测试集前后评测。训练循环约 105.6 秒，PyTorch allocated 峰值 9.32 GiB、reserved 峰值 11.78 GiB。

32 条合成测试样本（16 个场景组）的整条正确率 87.5% → 100%，字段准确率 96.875% → 100%；不能外推为通用能力提升。完整汇总见 [A800 实测](assets/a800-verification.json)，[训练曲线](assets/a800-training-loss.png)，逐条预测见 `assets/a800-before-test.json` 与 `assets/a800-after-test.json`。

- 全参数训练尚未在 A800 执行。实际数据盘 50 GB，本次 LoRA 完成后约占 8.5 GiB。
- 旧版 0.6B 的 MPS 实验保留在 `assets/history-qwen3-0.6b/`，与本次结果独立。
- `tokenizer_for()` 固定 revision 下载分词器文件后从快照目录加载，修复 Transformers 4.57.3 离线模式下的额外联网查询；回归测试覆盖此问题。

课件源文件由 `build_lesson.py` 生成，重建会清空运行输出。教师验证：

```bash
uv run --extra sft --with nbclient --with nbformat python 07-modern-llm/04-cloud-sft/execute_lesson.py
```

这条命令只执行课件里的本地检查，不自动训练或租 GPU。
