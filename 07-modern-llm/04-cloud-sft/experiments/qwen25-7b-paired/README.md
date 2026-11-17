# 魔搭 A10 上的 Qwen2.5-7B：同一道题，要求短答或解释

同一套纯预训练底座，一组用原始 9,000 条，另一组把其中一部分改成“只给结果 / 解释理由”配对样本。两组都训练两轮，学习率和监督 token 预算一致。这里保存的是这次实验的代码、冻结输入和云端实测输出。

**从代码看模型如何回答：** [训练与推理原件](cloud/experiment.py) → [P28 真实输出](evidence/P28.json) → [逐 token 调用图](figures/P28-纯底座生成流程.svg)。在云端脚本中找 `prompt`（第 65 行）、`model_and_tokenizer`（132）、`train`（154）、`generate`（292）；第 328 行的 `model.generate` 就是生成入口。文件开头的“四个快照”是原件沿用的旧注释；实际的 `STAGES` 定义和回传记录均为每组 **8 个**检查点。

[一次 SFT 更新：单条真实样本与训练代码流程图](figures/SFT-单样本与训练代码.html)（离线可展开源码）。

[用一道短答题逐行读 train(args)：154–288 行](figures/train-逐行讲解.md)。

[Qwen、DeepSeek、MiMo 的 SFT 配方与权重实测](research/sft-recipes/README.md)：区分公开训练流程、SFT 学生与旗舰，以及实际发生变化的原参数。

## 先看结果：不用 GPU

以下命令均在本目录运行。前两条只使用 Python 标准库，不需要下载模型。

```bash
python3 replay.py --check-only
python3 tools/show_p28.py --mode both
```

第二条直接从云端回传包读出同一题的六段原始输出（纯底座、原数据 SFT、双问 SFT，各有短答和解释）。短答时，底座生成 `30709 小 → 27091 面 → 271 双换行`，继续复述直到 256 token；双问组两轮后生成 `30709 小 → 27091 面 → 151643 结束`。在解释问法下，双问组却把“一大一小”答成了“面积相等”。[离线实验报告](figures/同题双问-魔搭实测.html)可查看各题、各检查点；[评分复核说明](evaluation/README.md)说明如何从 2,040 条回答重算当时的评分。

## 哪些文件真正在云上运行

| 位置 | 用途与来源 |
|---|---|
| [cloud/experiment.py](cloud/experiment.py)、[run-pair.sh](cloud/run-pair.sh)、[pack.py](cloud/pack.py) | 上次从魔搭下载的原始测量包内的文件，逐字节提取。分别负责训练与生成、启动两组实验、打包检查点和测量结果。 |
| [cloud/schedule.json](cloud/schedule.json)、[prompts.jsonl](cloud/prompts.jsonl) | 同一回传包里的 8 段训练顺序和 120 条生成提示；生成提示不含参考答案。 |
| [cloud/status.py](cloud/status.py)、[early-control.sh](cloud/early-control.sh) | 从当时的上传包提取的进度与提前评测辅助脚本；它们没有随最终测量包回传。原 `.sh` 保留旧实例的挂载路径、标志文件和等待逻辑。 |
| [evidence/](evidence/) | [回传原包](evidence/pair-measurements.tar.gz)、[训练日志](evidence/pair-experiment.log)、原校验文件、[成员来源与 SHA-256](evidence/source-index.json)，以及两组云端配置。包内 `run/{control,paired}/` 包含生成 ID、训练和验证记录。 |
| [inputs/](inputs/)、[data-prep/](data-prep/README.md) | 可直接复跑的冻结输入、模型清单，以及魔搭 COIG-CQIA 取样、配对、审计和传输脚本。共用 120 条验证集；60 个新评测题每题两种问法。 |
| [evaluation/](evaluation/README.md)、[tools/](tools/) | 本地评分与盲评记录、云端结果验收、单题读取与制图工具。 |
| [prepare_base.py](prepare_base.py)、[replay.py](replay.py) | 为项目新加的便携入口。前者准备固定纯底座；后者核验输入后直接调用上面的云端原版 Python，不依赖旧实例目录。 |

可从本目录执行 `(cd evidence && shasum -a 256 -c pair-measurements.tar.gz.sha256)` 核对云端原包；Linux 也可用 `sha256sum -c`。仓库没有放入 15.2 GB 的底座分片、16 份 LoRA 权重或优化器状态；本目录的原始回答和报告不依赖它们。

## 在新的 CUDA 实例上复跑

历史实测环境是 **单张 NVIDIA A10**，CUDA 版 PyTorch `2.13.0+cu130`、Transformers `4.57.3`、PEFT `0.17.1`；直接沿用云镜像的 CUDA PyTorch。Python 3.11+。从参考权重、冻结数据到每次生成的 token 都有记录；不同硬件或软件版本不保证逐 token 完全一样。

1. 在有 CUDA PyTorch 的 Linux 环境建隔离环境，并准备纯预训练底座：

   ```bash
   python3 -m venv --system-site-packages .venv
   source .venv/bin/activate
   python -m pip install -r requirements-gpu.txt
   python prepare_base.py --modelscope --output verified-base
   ```

   已有这四个模型分片时，上一条可换为 `python prepare_base.py --shards-dir /path/to/shards --output verified-base`。脚本核验模型文件；不使用 Instruct。参考版本及所有哈希见 [base-manifest.json](inputs/base-manifest.json)。

2. 选一种用途。输出目录需尚不存在，避免混用旧权重和缓存：

   ```bash
   # 只跑纯底座：两种问法共 120 条，不训练
   python replay.py --generate-base --base verified-base --output runs/base-only

   # 跑完整实验：两个训练组、纯底座及 16 个检查点的全部生成
   python replay.py --run --base verified-base --output runs/full
   ```

   结果在输出目录的 `run/control/` 和 `run/paired/`。历史两组各训练 4,504 次更新，纯训练分别约 52 分钟和 56 分钟，只计优化步骤，不含权重加载、检查点验证与保存、生成和打包。每轮每组 9,000 条、1,353,439 个监督 token，每 2,250 条保存一次；原生训练结束符是 151643。单卡资源很紧，实际耗时与可用显存取决于镜像和卡型。脚本退出不会自动停云实例，结果落盘后需在服务商面板停止 GPU。
