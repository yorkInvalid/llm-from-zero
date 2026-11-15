# AutoDL：第一次真实 Qwen3-4B SFT

本次目标是跑通训练、保存、重载与评测。先用同一任务学习 SFT，再切换全参数或 LoRA；LoRA 不属于独立训练阶段。所有命令均在课件目录执行。

## 1. 创建什么实例

使用你已确认的 **A800-80GB-NVLink × 1 卡**，本次实例实际识别为 A800-SXM4-80GB。单卡不使用 NVLink 通信。模型选 Qwen3-4B，兼顾 LoRA 与全参数 SFT；仍需两步试跑确认实际峰值显存。

镜像选择 Python 3.10–3.12、PyTorch 2.6 或以上、CUDA 可用的常规 PyTorch 镜像。我们本地验证用 Python 3.12 / PyTorch 2.14.1 / MPS；本次云端镜像为 Python 3.12.3 / PyTorch 2.8.0+cu128，已通过 SSH 完成依赖安装、doctor 和 pip check。本次 4B LoRA 的两步试跑及 60 步训练已通过；换环境或配置后仍建议先做两步试跑。

下单前记录页面小时单价、显存、数据盘容量与是否有付费扩容，并设置合理的定时关机。当前代码是单卡程序，不使用 torchrun 或多卡通信。

预算估算：GPU 开机小时数 × 页面单价 + 另行计费的存储等。训练脚本耗时不等于全部计费时长，安装、下载、评测和空闲也要考虑。官方按量实例计费按开机到关机时长计算；付费扩容数据盘可能在关机后继续计费。参考 [计费文档](https://api.autodl.com/docs/price/) 和 [本地数据盘说明](https://www.autodl.com/docs/local_disk/)，核对日期 2026-11-14。

## 2. 上传课件包

本机运行 `python package_lesson.py` 可以生成 `dist/autodl-sft-qwen3-4b-a800.zip`。它包含当前代码、数据和课件，不含密钥、模型权重、训练结果或环境。

用实例 JupyterLab 文件上传，把 ZIP 放到 `/root/autodl-tmp/`。打开实例终端：

```bash
cd /root/autodl-tmp
unzip autodl-sft-qwen3-4b-a800.zip
cd cloud-sft-a800
```

也可以使用控制台提供的 SSH/SCP 连接信息上传。密码和 token 不要写进代码、报告或聊天。无需把 AutoDL 账号密码交给助手。

## 3. 创建环境和模型缓存

非交互 SSH 的 PATH 可能没有 `python`。本次镜像解释器位于 `/root/miniconda3/bin/python`；下面优先使用该路径，其他镜像回退到 PATH 中的 Python。先确认输出的解释器、PyTorch 版本和 CUDA 可用性；如果检查失败，先找到镜像自带且能导入 CUDA PyTorch 的解释器，再创建环境。

```bash
if [ -x /root/miniconda3/bin/python ]; then
    SFT_PYTHON=/root/miniconda3/bin/python
else
    SFT_PYTHON="$(command -v python || command -v python3)"
fi
"$SFT_PYTHON" -c 'import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__, torch.cuda.is_available())'
"$SFT_PYTHON" -m venv --system-site-packages /root/autodl-tmp/sft-venv
source /root/autodl-tmp/sft-venv/bin/activate
python -m pip install -r requirements-cloud.txt
python -m pip check
export HF_HOME=/root/autodl-tmp/hf-cache
python run.py doctor --device cuda
df -h / /root/autodl-tmp
```

`--system-site-packages` 复用镜像的 CUDA PyTorch；不要在云端照搬本仓库的 Mac 环境或整个根目录 uv.lock。依赖固定在本课的 requirements 中，PyTorch 与驱动由镜像提供。重新打开终端时需再次激活环境、设置 HF_HOME。

本次实例的 **30 GB 系统盘**与挂载在 **`/root/autodl-tmp` 的 50 GB 数据盘**是不同文件系统，不能把系统盘容量当成数据盘可用空间。课件、venv、模型缓存和训练结果均按上面的路径放在数据盘；容量与剩余空间以 `df` 输出为准。50 GB 数据盘先用于 LoRA 流程，全参数实验需按下面的 checkpoint 需求另行准备空间。

doctor 应输出 CUDA 设备名、显存、torch/transformers/peft 版本。若 CUDA 不可用，先修复镜像或实例配置，不要直接把云端训练偷偷切到 CPU。

LoRA 以 BF16 加载冻结基座，adapter 使用 PEFT 默认 FP32；全参数训练保留 FP32 权重、梯度与 Adam 状态。两者前向均启用 BF16 autocast 和梯度检查点，Adam/梯度裁剪关闭 foreach 以控制临时显存。BF16 不需要 GradScaler。统一用 BF16 加载做训练前后生成评测，使用 PyTorch SDPA。

全参数约 4B × 16 字节 ≈ 60 GiB 用于权重、梯度及 Adam，此外还有激活和运行开销。单个包含优化器的 checkpoint 约 45 GiB。默认只保存末尾（60 步，试跑则 2 步）；保留试跑、正式结果和导出 ZIP 时建议约 150–200 GB 数据盘。脚本检查输出盘空间，但 HF_HOME 若放另一块盘还需单独检查。空间不足时先做 LoRA；不要把免费小数据盘容量当作足够。

## 4. 下载一次模型，检查所有样本

```bash
python run.py inspect
python -m unittest discover -s tests -v
```

第一次 inspect 下载 tokenizer；首次 eval/train 会下载约 8 GB 基座权重。固定模型为 Qwen/Qwen3-4B，revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。不要把不存在的镜像版本或其它仓库当作同一权重。

inspect 会打印训练/验证/测试数量、最大 token 长度，以及一条样本的生成前缀和真正参与 loss 的回答。超出 max_length 会报错，不静默截断。若 Hugging Face 连接失败，按 AutoDL 官方 [学术资源加速](https://www.autodl.com/docs/network_turbo/) / [Hugging Face 使用说明](https://www.autodl.com/docs/huggingface/) 操作后重试；网络连通性需在你的区域实测。

## 5. 先保存训练前表现，用验证集试跑

```bash
python run.py eval --device cuda --split validation --limit 8 --out runs/before-validation8.json
python run.py train --device cuda --config configs/lora.json --steps 2 --out runs/lora-trial
python run.py eval --device cuda --split validation --limit 8 --checkpoint runs/lora-trial/checkpoint-000002 --out runs/after-trial-validation8.json
python run.py compare --before runs/before-validation8.json --after runs/after-trial-validation8.json
```

这 2 步验证 CUDA、反向传播、保存和重载，不证明模型质量提升。脚本输出有效回答 token 数、更新耗时、梯度范数和 CUDA 峰值显存；`summary.json` 记录完整环境。`peak_allocated_gib` 是 PyTorch 分配峰值，不是整张卡的全部使用量，同时查看 `nvidia-smi`。

若要试全参数配置，使用另一个新目录：

```bash
python run.py train --device cuda --config configs/full.json --steps 2 --out runs/full-trial
```

全参数 trial 尚未在实际 CUDA 上验证。如显存不足，先用 LoRA 完成流程；若修改长度、微批量或 checkpointing，保留修改后的配置再试。不要减少 max_length 后把异常截断当成功。

## 6. 正式短训练：先固定配置，再保留测试集对照

根据验证集和试跑资源选择一次正式配置。下面沿用 LoRA，共 60 次 optimizer 更新，微批量 4、梯度累积 4，有效 batch 为 16 条样本（单卡）。学习率固定；60 步是流程实验预算，不是最优训练量。

```bash
python run.py eval --device cuda --split test --out runs/before-test.json
python run.py train --device cuda --config configs/lora.json --out runs/lora-60
python run.py eval --device cuda --split test --checkpoint runs/lora-60/checkpoint-000060 --out runs/after-test.json
python run.py compare --before runs/before-test.json --after runs/after-test.json --out runs/test-comparison.json
```

看到测试结果后不要反复针对这 32 题改配置；需要调参就回到验证集，保留最终独立评估。原模型可能已经做得很好；没有提升、甚至下降也要如实记录。

想做全参数正式训练，把 `configs/lora.json` 换成 `configs/full.json`，输出目录和评测文件换新名字。两份配置学习率不同，第一轮不据此宣称哪种方法更优。测试分数只代表这套合成数据，不代表真实业务能力。

## 7. 实时看训练

每一步会打印指标并写 `runs/lora-60/metrics.jsonl`，同时写 TensorBoard 事件：

```bash
mkdir -p /root/tf-logs
ln -s /root/autodl-tmp/cloud-sft-a800/runs/lora-60/tensorboard /root/tf-logs/lesson04-lora60
```

首次创建该链接即可；如果同名链接已经存在，检查其目标，不要覆盖其他实验。按 AutoDL 控制台的 AutoPanel/TensorBoard 入口查看，官方默认读取 `/root/tf-logs/`。[官方 TensorBoard 文档](https://www.autodl.com/docs/tensorboard/)

重点看 `loss`、`validation_loss`、`gradient_norm_before_clip`、`clip_scale`、`update_seconds` 和 `peak_allocated_gib`。小 batch 的训练 loss 会波动，不要求每步下降。

需要关掉浏览器仍继续训练时，可在镜像提供的 tmux 会话里运行命令：`tmux new -s sft`，按 `Ctrl-b` 后再按 `d` 分离，`tmux attach -t sft` 返回。断开 SSH 不等于关机。

## 8. 中断后恢复

默认每 60 步及最后一步保存 checkpoint；中断在首次保存前无法恢复。需要更频繁保存时，开跑前调整 save_every，并预留多个大 checkpoint 的空间。正常完成的 checkpoint 有 `run_metadata.json`。例如恢复到总共 60 步：

```bash
python run.py train --device cuda --config configs/lora.json --resume runs/lora-trial/checkpoint-000002 --out runs/lora-resumed
```

保存内容：模型或 adapter、tokenizer、Adam 状态、抽样器和 CPU/CUDA 随机状态、步数、配置、数据哈希。steps 表示总目标步数，不是再增加多少步。除总步数外的配置或数据变动会被拒绝；固定学习率，无 scheduler/AMP scaler。跨设备、软件版本、多卡不能保证逐位相同。新输出目录防止覆盖旧记录。

## 9. 下载并释放实例

```bash
python export_run.py --run runs/lora-60 --reports runs/before-test.json runs/after-test.json runs/test-comparison.json --out /root/autodl-tmp/sft-results.zip
```

恢复训练使用 runs/lora-resumed 时，把 --run 改为对应目录。导出会包含该实验的 checkpoint、训练状态、日志、配置、评分报告，以及本课数据/脚本以便复现。LoRA checkpoint 不包含全部基座权重，离线使用时另需固定版本的基座缓存。

通过 JupyterLab 或 SCP 下载 ZIP 到本机，确认可打开，且包含 `adapter_model.safetensors`（LoRA）或 `model-*.safetensors` 和索引（全参数可能分片）、tokenizer、配置和报告。云端已经通过重新加载 checkpoint 完成过推理验证后，再在控制台关机；确认实例状态和账单。准备释放实例前确保结果已备份。关机不是释放，付费存储等资源需按实际使用情况另外处理。

## 本课验证边界

2026-11-14 已在当前 A800 实例完成 9 项自动测试、真实 Qwen3-4B 的 2 步 LoRA 试跑、60 步正式训练、保存重载及 32 条独立测试样本的前后评估。实测汇总见 `assets/a800-verification.json`；逐条输出和训练曲线保存在 assets。全参数配置尚未执行，未把它记录为成功。

本次运行目录为 `runs/a800-lora-20261114/`，其中 `train60/checkpoint-000060/` 是正式 checkpoint。TensorBoard run 名为 `lesson04-a800-lora60`，读取 `/root/tf-logs`。模型缓存下载并校验完成后，可以设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 运行；首次使用仍需先取得固定版本模型文件。分词器 helper 从本地快照目录加载，避免当前 Transformers 版本的额外联网查询。

旧版 0.6B 的 MPS 结果独立归档，不用于替代本次 A800 实测；计费金额仍需在 AutoDL 账单核对。
