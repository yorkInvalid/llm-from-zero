# Qwen、DeepSeek、小米 MiMo 的 SFT 到底怎么做？

核对日期：2026-11-16。比较通用语言／多模态模型的**开放权重**，而非三家的全部 API 或专用模型。区分三种证据：作者披露的训练过程、公开代码实际指定的参数，以及本次直接比较权重得到的观测。一个团队不同代、不同尺寸、老师与学生的配方不能互换。

## 先分清三件事

- **训练目标**：普通 SFT 给出一条示范，在指定的回答位置提高示范 token 的概率。示范可以是一个字母、带解释的答案或多轮工具交互；“有标准答案”不等于“只有最后一个 token 受监督”。
- **哪些参数可动**：全参数 SFT 可以直接改原权重；LoRA SFT 冻结原权重，只改指定分支；也能选择只冻结某个模块，其余照常训练。单看交叉熵、epoch 或学习率，不能反推出 trainable 参数名单。
- **哪个阶段**：生成并筛选教师答案，再用固定答案教学生，属于可以用 SFT 实现的离线蒸馏。学生自己生成、教师按学生所处的上下文给反馈，是在线策略蒸馏（OPD）。强化学习（RL）又以任务反馈作为优化信号。这三种步骤可以出现在同一个发布模型的训练历史里。

## 近期开放权重：哪些信息确实公开了

| 公开型号 | 官方材料能支持的结论 | 不能移植过来的数字／结论 |
|---|---|---|
| Qwen3.8-2.4T-A95B、Qwen3.8-Flash-Next | 发布的是后训练权重；Flash-Next 的报告重点是架构、稀疏注意力和优化稳定性。[2.4T 模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[Flash-Next 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[论文](https://arxiv.org/abs/2608.30320) | 我查到的材料未给出可核对的旗舰 SFT 可训练参数名单。架构／预训练的优化器配方不是 SFT 的冻结清单。11 月 16 日发布的 Qwen4-Max 是服务；阿里说较小的 Qwen4 开放权重将另行发布。[阿里公告](https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen4-max) |
| DeepSeek-V4.1-Flash；1017 更新 | 家族报告明确为 SFT → RL → OPD；重点改在构造可验证任务、互动环境、去重与难度筛选。1017 是后续更新，官方强调用自反馈 RL 改进项目任务。[V4.1 §5](https://arxiv.org/html/2609.19969v1#S5)、[1017 发布](https://api-docs.deepseek.com/news/news261017/)、[1017 权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 可验证环境包含用于 RL 的数据，不能统称为 SFT 示范。更早的 R1 约 80 万条也不是 V4.1 的 SFT 条数。原文没有给出能复建 1017 SFT 参数组的配置。 |
| MiMo-V2.6-Pro／Flash | 作者明确写了先做一段短 SFT，然后扩大混合 RL；混合 RL 后还通过多教师在线蒸馏补能力。难以写出可靠奖励的领域会用合成示范训练教师。甚至可以只取旧 SFT 轨迹的前缀，由学生自己续写，让教师指导新的决策。[V2.6 §4](https://arxiv.org/html/2610.11959v1#S4)、[§5.6](https://arxiv.org/html/2610.11959v1#S5.SS6)、[旗舰发布权重](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | “短”没有给出旗舰的 SFT 轮数或墙钟时长。著名的巨额算力数字写的是 RL，而同文的 774 亿 token 写的是另一个 9B 学生；都不能当旗舰 SFT 预算。 |

## 哪些实验披露到能看懂样本、loss 或超参

**Qwen2.5：与本项目同家族的对照。** 通用模型报告的 SFT 数据超过 100 万条，包含代码、数学、指令遵循、长文、结构化数据、多语言等，不是将一个简单问答集重复很多轮。以指令遵循为例，先生成要求、验证程序和测试，再用执行结果淘汰违背要求的候选回答。报告给出 2 轮、训练序列长度 32,768、学习率 7e-6 降到 7e-7。后面还有偏好训练，不能从 Base 与最终 Instruct 的差值单独恢复 SFT。数字是该系列报告的口径，不是 3.8 配方，也不是公开的 7B 专属 GPU 时长。[Qwen2.5 报告 §4.1–4.3](https://arxiv.org/html/2412.15115v2#S4.SS1)

**Qwen3：为什么“短答”和“解释”可以同时学？** 旗舰先用可验证的长推理示范冷启动，再做推理 RL，之后有**第二次 SFT 做模式融合**。推理侧让上一阶段的模型对题目生成候选、拒绝不合格回答；非推理侧另收指令、问答、创作等数据。模板用 /think 和 /no_think 区分模式；非思考示范保留空思考块，之后依然要输出最终回答。多轮示范还教模型服从最近的模式指令。这和本项目通过提示词控制行为的思路相通，但“是否输出思考块”不等同于“最终回答多少字”，也不能断言官方每道题都配了长短两个版本。[Qwen3 §4.1、§4.3 与表 9](https://arxiv.org/html/2505.09388v1#S4.SS3) 参数冻结也未必是全有或全无：同团队另外的 **Qwen2.5-VL** 明确在 SFT、DPO 冻结视觉编码器；这不能推出文字版也冻了同样的层。[Qwen2.5-VL §2.3](https://arxiv.org/html/2502.13923v1#S2.SS3)

**DeepSeek-V3：按任务决定谁能验答案。** 总计约 150 万条。推理题借助领域模型生成多个候选，经筛选保留正确、清楚的示范；创作、角色扮演、简单问答等由 V2.5 生成、人工核验。训练两轮，学习率按余弦从 5e-6 降到 1e-6。多条样本可以拼进一个序列以减少浪费，但注意力掩码让它们互不可见。这个“样本隔离”与“回答中哪些 token 计 loss”是两回事。[DeepSeek-V3 §5.1](https://arxiv.org/html/2412.19437v2#S5.SS1)

**DeepSeek-R1-Distill-Qwen-7B：最接近我们 7B 实验的官方学生。** 起点是 Qwen2.5-**Math**-7B，不是我们的通用 Qwen2.5-7B。老师对推理题生成多个解，规则或评判模型筛掉错误候选，再混入写作、翻译、事实问答等；简单问候无需硬写推理。报告概括为约 60 万条推理 + 20 万条非推理；细表按领域统计为 804,745 条，平均每条 5,355.3 token。学生阶段只有 SFT，没有再做 RL。修订报告给出蒸馏模型训练 2–3 轮、batch 64、最长 32,768；7B 那行的初始学习率 8e-5，余弦降到十分之一。未细分每个尺寸究竟跑 2 还是 3 轮。报告里 5,000 H800 GPU 小时是**构造 SFT 数据**的成本，不是这台 7B 学生的训练时间。[R1 修订版，表 5](https://arxiv.org/html/2501.12948v2#A2.SS3.SSS3)、[表 6 与表 7](https://arxiv.org/html/2501.12948v2#A2.SS4.SSS3)、[学生模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

**小米 MiMo：同家公司、两种可核查的 SFT。** 第一代 MiMo-7B-SFT 将公开和自有蒸馏示范合并，去掉与评测有 16-gram 重叠的问题、语言混乱或不完整的回答，每题最多留 8 个回答；初版清洗后约 50 万条，batch 128、恒定学习率 3e-5、样本拼接到最长 32,768。该报告还做了一个对本项目有参考价值的实验：只教答案格式的轻量 SFT 在后续 RL 起初占优，但约 500 步后落在直接从 Base 开始的曲线后面。这个结论限于那次实验，不是说格式训练无用。[MiMo-7B §3.1](https://arxiv.org/html/2505.07608v1#S3.SS1)、[§3.6 的消融](https://arxiv.org/html/2505.07608v1#S3.SS6)

新一代小米的 **MiMo-V2.6-Distill-Qwen-9B** 是另一条路径：从已经发布的 Qwen3.5-9B 出发，模仿 MiMo 生成的代码、通用任务、视觉和网络安全任务轨迹。作者的加权统计是总计 **77.4B（774 亿）token，其中 27.2B（272 亿）计入 SFT loss，约 35.1%**。论文没有把所有其余 token 按角色逐项披露，不能擅自全算为用户提问。作为可观察的结果，官方报告同一 SWE-bench Pro 指标从 Qwen3.5-9B 的 32.0 到 SFT 学生的 44.6；这个对比不含随后单独开展的 RL。[MiMo-V2.6 §7.1 与表 4](https://arxiv.org/html/2610.11959v1#S7.SS1)、[9B 学生模型卡](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)

## 回查小米原文：哪个 SFT 明确放开了原模型？（2026-11-17）

**MiMo-VL-7B 有直接答案。** 小米把四段放在论文的预训练章节介绍，但表 1 给第四段专门命名为 Long-context SFT。四段可训练组件依次为：①只训投影层；②视觉编码器 + 投影层（语言模型仍冻结）；③所有组件；④所有组件。正文说明第三段已放开全部参数，第四段产出 MiMo-VL-7B-SFT，序列长从 8K 提到 32K；表中第四段的学习率为 2.5e-5，训练量为 550B token。这是模块级参数范围的一手证据，比从权重差分猜测更直接。它不是纯文本 MiMo-7B，也不是 V2.6-Pro。[小米 MiMo-VL 报告，表 1](https://arxiv.org/html/2506.03569v1#S2.T1)

**V2.6 旗舰的已知流程。** 先做文本预训练，再接入视听编码器联合训练；中期训练混入智能体完成任务的交互轨迹，先用 256K 上下文，最后拉到 1M。该阶段原网络的隐藏权重矩阵用 Muown，词嵌入、输出头和路由器用 AdamW。随后是短 SFT → 混合强化学习 → 多教师在线蒸馏。进入 RL 时从 SFT 检查点继承 FP32 主权重与 Muown 的行状态（优化器保存的训练状态）；**冻结路由器是 RL 阶段明确的决定**，不能倒推成旗舰 SFT 的完整冻结清单。[预训练与中期训练](https://arxiv.org/html/2610.11959v1#S3)、[短 SFT 与 RL](https://arxiv.org/html/2610.11959v1#S4)、[优化器与冻结细节](https://arxiv.org/html/2610.11959v1#S5.SS1)

后两段可以直接对照官方的 [Pro-RL 检查点](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) 与 [Pro-MOPD 检查点](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)。[公开的训练看板](https://mimo.xiaomi.com/rl/mimo-v26/)展示 RL；它不等于旗舰 SFT 的逐步训练日志。遇到模型反复调用工具而没有推进任务，小米还单独记录了如何训练专项教师，再将行为蒸馏回主模型。[专项训练复盘](https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition)

## 独立抽查：原模型到底有没有权重变过？

本项目的 [LoRA 配置](../../cloud/experiment.py) 只动注意力和 MLP 的七类投影；即便把 LoRA 合并，**RMSNorm 的 weight 也应原样保留**。据此选定第 0 个语言层的两个小向量（input_layernorm、post_attention_layernorm），逐元素比较官方 checkpoint。每对张量均同名、同形状、同为 BF16。全部在 CPU 上完成，仅读分片的头和对应字节范围。

| 官方公开起点 → 终点 | input_layernorm 不同元素 / 总数 | post_attention_layernorm 不同元素 / 总数 | 能归因到哪一步 |
|---|---:|---:|---|
| [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) → [7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 4 / 3,584 | 92 / 3,584 | 起点到最终指令模型之间还有偏好训练，不能归因给 SFT 单一步骤 |
| [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) → [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 3,584 / 3,584 | 3,181 / 3,584 | 作者标明的父模型与仅 SFT 蒸馏的学生 |
| [MiMo-7B-Base](https://huggingface.co/XiaomiMiMo/MiMo-7B-Base) → [7B-SFT](https://huggingface.co/XiaomiMiMo/MiMo-7B-SFT) | 3,939 / 4,096 | 3,875 / 4,096 | 初版 Base 与 SFT 发布点 |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) → [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) | 3,787 / 4,096 | 3,426 / 4,096 | 作者标明的起点与 SFT 学生，不是 V2.6-Pro 旗舰 |

**证据边界：** 这些公开终点不是在相应公开起点上“只加本项目七投影 LoRA、其余张量和导出都不动”就能得到的。抽两个向量**不能**判断所有投影是否全参更新、排除 LoRA 加额外解冻、区分内部起点／导出转换，或还原优化器。不要把“原参数的发布值变了”说成“已拿到官方梯度记录”。[逐张量 SHA-256、提交版本、字节范围与计数](norm-diff.json)；[复核脚本](probe_norms.py)。

若想看一个**直接从训练代码判定**的全参实例：AI2 的 Tülu 3 8B [官方 SFT 配置](https://github.com/allenai/open-instruct/blob/3dd7a07da07e78550e7be48a7d317ec3d9c9cd6d/configs/train_configs/tulu3/tulu3_sft.yaml) 从 Llama-3.1-8B 出发，没开 LoRA；[同版本训练器](https://github.com/allenai/open-instruct/blob/3dd7a07da07e78550e7be48a7d317ec3d9c9cd6d/open_instruct/finetune.py#L240) 默认 use_lora=False，第 706 行只在开关打开时创建适配器。这个公开配方可以复查，不能把它当另三家旗舰的参数配置。

复核权重抽查（仅需 Python 标准库和网络，不启动 GPU）：

~~~bash
python3 probe_norms.py > /tmp/norm-diff-recheck.json
diff -u norm-diff.json /tmp/norm-diff-recheck.json
~~~

## 回到我们的单卡实验

我们每轮 9,000 条、1,353,439 个监督 token；两轮实际训练预算为 2,706,878 个监督 token。我们那道短答中，43 个有效输入 token 只有 C 和 EOS 两个参与监督。大厂也是在决定哪些行为值得模仿，但训练规模、任务构成、答案长度和筛选强度都不同。“同样训练两轮”绝不等于训练预算相同。[本次云端配置](../../evidence/paired-config.json)

针对本项目，可借鉴的是两条：**先验内容，再验格式**（解释不能与题设冲突，也不能和短答结论矛盾）；**分别评分**（选对、按要求停止、理由正确、同题两问都通过）。可以加一组打乱选项顺序的新题，检查模型是否随答案内容而不是原字母作答。这是依据本次实验与公开配方提出的评测建议，不是三家已经做过完全相同的实验。
