# 多层 Encoder 学习实验

当前实现多层 Post-LN Encoder，每层为：多头注意力 → Add & Norm → FFN → Add & Norm。支持单序列、Batch、Padding Mask 和可配置 Dropout。

所有 token 按列排列。默认 3 层，每层 2 个头、特征维度 10、FFN 中间维度 40。输入输出均为 `(10, 3)`；单层 1290 个参数，3 层共 3870 个参数，各层参数独立。

## 阅读顺序

1. `attention.py`：逐头计算 Q、K、V、分数、softmax 和加权求和，再拼接并通过 W_o 投影。权重的行是 key，列是 query，每列之和为 1。
2. `position_encoding.py`：固定 sin/cos 编码和 RoPE。每头 5 维时，前 4 维参与 RoPE，最后一维不变。
3. `attention_sublayer.py`：`Z = LayerNorm1(X + Attention(X))`。LayerNorm 沿每个 token 的特征归一化；代码通过转置适配 PyTorch 最后一维的接口。
4. `feed_forward.py`：`Linear(10, 40) → ReLU → Linear(40, 10)`，所有 token 共享参数，但独立计算。
5. `encoder_layer.py`：`output = LayerNorm2(Z + FFN(Z))`。第二次残差加回 Z；两个 LayerNorm 的参数独立。
6. `encoder.py`：用 `nn.ModuleList` 注册多个独立创建的 EncoderLayer，依次传递上一层的输出。返回最终输出和逐层注意力权重列表。
7. `train.py`：固定输入求平均实验，使用最终 Encoder 输出计算 MSE，由 Adam 更新全部层的参数。
8. `visualization.py`：`plot_encoder_training()` 按层选择记录，复用原有的 `plot_training()`，每层生成曲线和热力图。

导入模块不会自动训练或绘图。sin/cos 只在进入第一层前加一次；RoPE 在每层 Attention 中旋转该层的 Q、K。

## 运行

在项目根目录执行：

```bash
.venv/bin/python 06-attention/train.py --position-encoding sinusoidal --num-layers 3
.venv/bin/python 06-attention/train.py --position-encoding rope --num-layers 3
.venv/bin/python 06-attention/train.py --position-encoding none --num-layers 1
```

默认 sinusoidal、3 层、300 次参数更新。层数必须为正整数。图片保存到 `artifacts/`，文件名包含编码模式、总层数和当前层号，例如 `attention-training-rope-3layers-layer1.png`，避免覆盖单层或其他层的结果。

在本目录下也可分开调用训练和绘图：

```python
from train import train
from visualization import plot_encoder_training

model, losses, history = train("rope", num_steps=300, num_layers=3)
paths = plot_encoder_training(losses, history, "artifacts", name="rope")

# X 已准备好；sin/cos 模式需要先在模型外加位置编码。
output, layer_weights = model(X)
# layer_weights[0]：第一层权重，形状 (head, key, query)
```

`history` 包含初始状态和每次更新后的 eval 状态，共 301 条；每条是已 detach 的 `(layer, head, key, query)` 张量，默认 `(3, 2, 3, 3)`。全部记录堆叠后为 `(301, 3, 2, 3, 3)`。即使只配置一层，记录也保留 layer 维度。`plot_training()` 仍接受不含 layer 维度的单层记录，可使用 `[step[0] for step in history]` 取出第一层。

原有 `MultiHeadAttention`、`AttentionSublayer` 和 `EncoderLayer` 仍可单独使用。`forward_with_residual(attention, X)` 在 `attention_sublayer.py` 中，train.py 保留导入兼容；只应传入原始注意力模块，避免重复添加残差。

## 测试与实验边界

```bash
.venv/bin/python -m unittest discover -s 06-attention/tests -v
```

- `test_residual.py`：残差手算、恒等路径、梯度；通过一层 Encoder 对照手算的一步 Adam 更新；三种编码下默认三层训练 300 步，检查 loss 至少下降 99%、301 条记录及每层注意力归一化。
- `test_layernorm.py`：逐 token 公式、仿射参数、常量输入与梯度。
- `test_feed_forward.py`：FC/ReLU 手算、token 独立性、参数共享与更新。
- `test_encoder_layer.py`：第二次残差加回 Z、两个 LayerNorm 独立及梯度。
- `test_encoder.py`：层数校验、独立参数与注册、手动逐层计算对照、不同序列长度、所有层梯度和更新、最终记录与模型输出对照、逐层绘图分发。

目标始终是原始输入 I 的平均值。求平均不依赖顺序，本实验只拟合一组固定输入，不能比较位置编码优劣，也不能证明泛化能力。多层模型也不保证比单层更适合这个简单任务。

artifacts 中不含层号的图片是早期单层实验记录；含层数和层号的图片对应多层训练。

## 可训练 Token Embedding 实验

新增 `token_encoder.py`：

- `TokenEncoder` 包含随机初始化的 `nn.Embedding`、位置编码和多层 Encoder。输入为非空的 `(n,)` 或 `(batch,n)` 整数 token ID；输出为 `(...,d_model,n)`，同时返回逐层注意力。
- 每次 forward 都重新查表：`X = self.embedding(token_ids).transpose(-2, -1)`。Embedding 在构造时初始化一次，不能每步重建或将查表结果 detach 后缓存。
- sin/cos 在第一层之前加一次；RoPE 在每层 Q/K 内部生效；none 不注入位置信息。这些位置编码没有可训练参数。这里沿用现有直接相加的教学约定，没有添加原论文的 embedding 缩放。
- `TokenClassifier` 组合 TokenEncoder 和逐 token 线性分类头，返回 `(n, num_classes)` logits。交叉熵直接接收 logits，无需提前 softmax。

新增 `train_tokens.py`，与原 `train.py` 求平均实验独立：

```bash
.venv/bin/python 06-attention/train_tokens.py
.venv/bin/python 06-attention/train_tokens.py --position-encoding rope --num-layers 3 --num-steps 200
.venv/bin/python 06-attention/train_tokens.py --position-encoding none --num-steps 1
```

教学词表为 `猫、吃、鱼、狗、喝、水`，ID 为 0–5，人工固定标签为 `动物、动作、食物/饮品`。词表映射不变，Embedding 表中的 60 个数会学习。默认模型包括 3870 个 Encoder 参数、60 个 Embedding 参数、33 个分类头参数，共 3963 个参数。这里直接输入已分好的 token，没有接入通用文本 Tokenizer。

参数更新在训练循环中明确展示：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
logits, _ = model(token_ids)
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()   # 计算 embedding.weight.grad，权重本身尚未改变
optimizer.step()  # 同时更新 Embedding、Encoder 和分类头
```

程序打印固定 ID、猫的初始/最终向量、第一次反向传播的 Embedding 梯度量、第一次 Adam 更新的变化量及最终预测。`train_tokens()` 返回 `(model, losses, snapshots)`；snapshots 保存更新前、第一次 backward 后、第一次 step 后和最终的表，以及第一次梯度（num_steps=0 时没有首次更新的快照）。所有快照都 detach 并 clone，不影响训练。

两个训练函数返回的 model 均处于 eval 模式，losses 是关闭 Dropout 后在相同训练样本上的评估 loss，不是独立验证集结果。继续训练前需调用 `model.train()`。

`tests/test_token_embedding.py` 验证查表数值与重复 ID、三种位置编码的注入位置、整数 ID 和范围、参数注册及梯度、backward 与 step 的区别、未使用行在首次无 weight decay 的 Adam 更新中不变、零步不更新，以及三种模式能拟合固定标签。

这个任务只用于展示 Embedding 与后续网络一起训练。标签可由 token 身份直接确定，没有独立验证集；准确率 100% 不代表理解语言、使用了上下文或具有泛化能力。后续再接入真实 Tokenizer 和依赖上下文的任务。

## Batch 与 Padding Mask

原来的单序列接口和默认训练实验保持可用。运行不同长度的 Batch 示例：

```bash
.venv/bin/python 06-attention/train_tokens.py --batch
.venv/bin/python 06-attention/train_tokens.py --batch --position-encoding rope
```

`make_padded_batch()` 构造两条序列 `[猫, 吃, 鱼]`、`[狗, 喝, PAD]`。PAD 的 ID 为 6，补在右侧；第二条序列的最后一个标签为 -100。Batch 模式下词表有 7 行，总参数量为 3973。

| 数据 | 单序列 | Batch |
|---|---|---|
| token ID / mask | `(n,)` | `(batch,n)` |
| Encoder 特征 | `(d_model,n)` | `(batch,d_model,n)` |
| 每层注意力 | `(head,key,query)` | `(batch,head,key,query)` |
| 分类 logits | `(n,classes)` | `(batch,n,classes)` |

`TokenEncoder(..., padding_idx=6)` 根据 ID 自动生成 bool mask，True 表示 PAD；`Encoder` 将 mask 传给每一层。直接使用特征向量时，可调用 `encoder(X, padding_mask=mask)`。

Q/K/V 都正常计算，mask 只替换 PAD key 对应的分数行：

```python
A = K.transpose(-2, -1) @ Q / (self.d_k ** 0.5)
A = A.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
weights = torch.softmax(A, dim=-2)  # 沿 key 归一化
B = V @ weights
```

PAD key 的权重为 0，其 V 不贡献输出；PAD query 仍有输出，不强行清零。FFN 和 LayerNorm 只处理各 token 的特征，不混合样本或位置。`nn.Embedding(..., padding_idx=6)` 使 PAD 行初始化为零且不通过查表获得梯度，但这不能代替 attention mask。

`token_classification_loss()` 将 logits/labels 的 batch 和序列维度展平，通过 `ignore_index=-100` 忽略 PAD 标签，对所有有效 token 求平均；准确率也只统计有效标签。PAD token ID 6 和忽略标签 -100 是两个不同概念，调用方需正确设置标签。

每条序列至少需要一个非 PAD token，全 PAD 样本明确报错，避免 softmax 全为负无穷产生 NaN；全部标签都忽略时也明确报错。位置编码仍按槽位 0…n-1 计算，当前示例采用右侧 padding，不保证左侧 padding 后的位置表示与未补齐时相同。

`tests/test_padding_mask.py` 覆盖手算 mask 方向、所有层 PAD key 权重为零、三种位置编码下 Batch 与单条未补齐结果一致、跨样本隔离、修改 PAD 向量不影响有效输出、有效输出对 PAD 输入梯度为零、loss 忽略与梯度、全 PAD 校验、Batch 与逐条梯度对照和实际训练。绘图接口仍针对单条序列；如自行绘制 Batch 的某个样本，先选取该样本再传入原绘图函数。

## Dropout 的配置与启停

所有模型构造函数及两个训练入口都接受 `dropout`，默认 `0.0` 保持原有实验行为；显式设置 `0.1` 才会在训练时丢弃约 10% 的元素：

```bash
.venv/bin/python 06-attention/train_tokens.py --batch --dropout 0.1
.venv/bin/python 06-attention/train.py --num-layers 3 --dropout 0.1
```

本次设置四个位置，使用同一个概率，但每次调用分别采样 mask：

1. `token_encoder.py`：Embedding 加 sin/cos 后的输入（none/RoPE 模式直接作用于 embedding 输出）。
2. `attention.py`：softmax 后的注意力权重，随后与 V 相乘。
3. `attention_sublayer.py`：Attention 经 W_o 投影后的分支输出，在加回 X 之前。
4. `encoder_layer.py`：FFN 第二层 FC 后的分支输出，在加回 Z 之前。

```python
X = input_dropout(embedding_plus_position)
B = attention(X)  # 内部包含 attention weight dropout
Z = norm1(X + output_dropout(B))
output = norm2(Z + ffn_dropout(ffn(Z)))
```

内部代码仍通过 `transpose(-2, -1)` 对齐特征维度。Dropout 不修改参数量、特征形状或 PAD 规则；FFN 内部 ReLU 后没有额外增加 Dropout。求平均入口 `train.py` 直接输入固定向量，没有 TokenEncoder，因此只有每层内部的三处 Dropout。

```python
model = TokenClassifier(vocab_size=7, padding_idx=6, dropout=0.1)
model.train()  # 递归开启子模块 Dropout（新建模块默认也是 train 模式）
logits, _ = model(token_ids)
# loss.backward() / optimizer.step()

model.eval()   # 递归关闭 Dropout
with torch.no_grad():  # 另一个开关：不记录梯度，不能替代 eval()
    logits, _ = model(token_ids)
```

训练循环显式交替使用 `eval()+no_grad()` 记录固定样本的 loss/预测，以及 `train()` 计算实际反向传播的 loss。训练中保留的元素乘以 `1/(1-p)`，推理时不丢弃、不缩放；p=1 是全部置零的边界情况，仅用于测试，不推荐用于训练。

Attention 返回/绘图展示的是 **Dropout 前的 softmax 概率**，每个 query 的 key 权重之和仍为 1。实际参与 V 加权的 Dropout 后权重不再保证和为 1，也可能大于 1，但 PAD 的零权重仍为零。开启随机 Dropout 后，Batch 与逐条输出不要求相同；在 eval 模式比较才有意义。

`tests/test_dropout.py` 验证训练随机性、eval 确定性、no_grad 不关闭 Dropout、p=0 兼容、p=1 时残差保留、输入 Dropout 的位置、注意力缩放与 PAD 保持、递归开关和真实训练模式切换。图像文件名包含 dropout 概率，以免覆盖其他实验。

## 上下文任务：判断 token 是否重复

新增 `context_task.py` 和 `train_context.py`，复用现有 TokenClassifier，不替换前两个教学实验。每个位置的标签取决于该 token 在当前序列中是否出现至少两次：

```text
输入 [0, 1, 0] → 标签 [1, 0, 1]
输入 [0, 1, 2] → 标签 [0, 0, 0]
```

相同的 token 0 在这两条序列中标签不同，只看 token 本身无法总是答对。

```bash
.venv/bin/python 06-attention/train_context.py
# 快速检查流程，不代表充分训练：
.venv/bin/python 06-attention/train_context.py --epochs 2 --train-size 64 --validation-size 32
```

默认使用 seed=42、8 个有效 token（ID 0–7）、PAD ID=8、长度 3–8、1024 条训练序列和 256 条验证序列。按排序后的 token 列表去重并划分，保证两个集合既没有相同序列，也没有 token 数量完全相同的重排序列。标签只根据每条原始序列生成，右侧 padding 的标签为 -100。

本任务对排列顺序不敏感，因此两个模型都不加位置编码，避免引入无关因素：

- Encoder：32 维 Embedding、3 层 Encoder、2 类分类头，共 12474 个参数。沿用每层 2 个头、每头 Q/K/V 5 维、FFN 中间维度 40。
- 逐 token 基线：32 维 Embedding、3 层逐 token FFN/残差/LayerNorm、2 类分类头，共 8442 个参数；没有 Attention 或其他跨位置聚合。
- 两者使用同一数据划分、每轮相同 shuffle 顺序、batch=64、Adam lr=0.003、dropout=0.1，固定训练 30 轮，不用验证集进行参数更新或挑选最佳轮次。参数量并不完全相等，因此这是教学对照，不是严格等参数消融。

每轮分别记录训练集和验证集 loss、token 准确率、整条序列完全正确率、两类召回率、平衡准确率和标签比例。评估使用 eval/no_grad，对所有有效 token 汇总，不把 PAD 或 batch 大小差异算进结果；训练使用 train 模式。

默认配置的一次本地运行结果（第 30 轮）：

| 模型 | 训练 token 准确率 | 验证 token 准确率 | 验证整序列准确率 | 验证平衡准确率 |
|---|---:|---:|---:|---:|
| Encoder | 94.2% | 92.4% | 80.1% | 92.3% |
| 无 Attention 基线 | 51.7% | 51.7% | 4.3% | 49.7% |

验证集中重复标签比例为 52.6%，因此全部预测为“重复”也能得到约 52.6% token 准确率；基线没有明显超过这个水平。平衡准确率是两类召回率的平均值，避免单看总体准确率误判。结果支持 Encoder 利用了上下文，但不代表掌握自然语言或能泛化到未测过的长度、词表及分布，也未做多随机种子统计。

完整逐轮记录保存在 `artifacts/context-task-report.json`，可用 `--output` 更改路径。再次运行会覆盖指定报告；当前不保存模型 checkpoint。`tests/test_context_task.py` 检查标签、无泄漏划分、PAD 补齐、评估统计权重、评估不更新参数、基线缺少上下文能力及训练流程。

## Decoder 第一步：可选 Causal Mask

`MultiHeadAttention.forward(X, padding_mask=None, causal=False)` 新增独立的因果开关。当前 Encoder 不传 causal，仍使用双向注意力；本步尚未添加 Decoder 或 Cross-Attention。

```python
attention = MultiHeadAttention()
output, weights = attention(X, padding_mask=padding_mask)               # 双向
output, weights = attention(X, padding_mask=padding_mask, causal=True)  # 因果
```

本代码的行是 key、列是 query，`positions[:, None] > positions[None, :]` 标记未来 key（严格下三角），对角线仍可读。先分别构造 `padding_blocked` 和 `future_blocked`，最后 OR 成 `blocked`：任意一个禁止，该 key-query 对就不能读取。每个头复用此 mask，支持 `(d_model,n)` 和 `(batch,d_model,n)`，不跨 batch 混合。

计算顺序为 `KᵀQ / √d_k → mask → softmax → attention dropout → V 加权 → 多头拼接与 W_o`。返回的权重仍为 Dropout 前的概率。

全 PAD 样本仍然报错。合并 mask 后，左侧 PAD 的早期 query 可能没有任何可读 key：代码将这些分数列暂时置零以安全执行 softmax，再把被屏蔽的权重置零，避免前向和反向出现 NaN。这样的 query 权重和为 0，注意力模块输出为 0；有可读 key 的 query 权重和为 1。此规则不表示残差之后的 PAD 输出也为零，任务 loss 仍需忽略 PAD 标签。

`tests/test_causal_mask.py` 验证手算前缀平均值、未来位置零权重、未来修改不影响此前输出、对应未来输入梯度为零、Batch/单序列与 RoPE、左/右/中间 PAD 合并、全屏蔽 query 的有限梯度、单 token 及全 PAD 拒绝。比较输出时使用 eval 模式，避免 Dropout 随机性影响因果性检查。

## Decoder 第二步：Cross-Attention

`MultiHeadAttention.forward(X, padding_mask=None, causal=False, memory=None)` 支持独立的 K/V 来源。`memory=None` 时仍为原来的自注意力；传入 Encoder 输出 memory 时：

```python
Q = W_q @ X       # Decoder 当前状态，目标长度 T
K = W_k @ memory  # Encoder 最终输出，源长度 S
V = W_v @ memory
```

创建两个不同实例，避免自注意力与交叉注意力意外共享参数：

```python
self_attention = MultiHeadAttention()
cross_attention = MultiHeadAttention()
# H 应为 Decoder 因果自注意力及第一次 Add & Norm 后的状态。
B, weights = cross_attention(H, memory=encoder_output,
                             padding_mask=source_padding_mask)
```

Cross-Attention 没有 causal mask。源序列没有 PAD 时无需传 mask；有 PAD 时，`padding_mask` 对应 **源 K/V 序列**，不是目标 query 序列。目标 PAD 的预测继续由任务 loss 忽略。

单序列 X 为 `(d_model,T)`，memory 为 `(d_model,S)`；Batch 时前面都加 batch 维。权重形状为 `(...,heads,S,T)`，输出为 `(...,d_model,T)`。源/目标长度允许不同，特征维度、batch 大小、dtype 和设备必须一致，不隐式广播 memory 到多个样本。所有目标 query 都可以读取全部有效源位置。

第一版明确拒绝 `memory` 与 `causal=True` 或 `use_rope=True` 同用，避免把自注意力的时间轴规则误用于两条独立序列。自注意力 RoPE 功能不受影响；未来 Decoder 如使用 RoPE，应为交叉注意力建立 `use_rope=False` 的独立实例。源长度为零或源序列全 PAD 时也明确报错。

Dropout、W_o 投影和返回 Dropout 前概率的约定保持不变。`tests/test_cross_attention.py` 验证矩形矩阵手算、全源可读、Batch 与逐条对照、源 PAD 隔离、有效源变化可影响输出、梯度到达两侧输入和所有投影参数、self/cross 同输入一致性，以及无效形状和 mask 校验。本步仅实现交叉注意力计算，尚未组装 Decoder Block 的三次 Add & Norm。

## Decoder 第三步：完整 Decoder Block

新增 `decoder_layer.py` 的 `DecoderLayer`，将前两个步骤的组件组装起来：

```text
X → 因果自注意力 → Dropout → 加回 X → LayerNorm1 → decoder_hidden
decoder_hidden → Cross-Attention(memory) → Dropout → 加回 decoder_hidden → LayerNorm2 → H
H → FFN → Dropout → 加回 H → LayerNorm3 → output
```

两种 Attention 各有独立投影参数，三个 LayerNorm 的 gamma/beta 也独立。默认单个 Block 为 1710 个参数。`use_rope=True` 只用于目标自注意力，交叉注意力仍不使用 RoPE 或 causal mask。

调用示例（在本目录内）：

```python
import torch
from encoder import Encoder
from decoder_layer import DecoderLayer

encoder = Encoder(num_layers=3)
decoder = DecoderLayer()
source = torch.randn(2, 10, 4)  # 实际应用中来自源 Embedding + 位置编码
target = torch.randn(2, 10, 3)  # 来自目标 Embedding + 位置编码
source_pad = torch.tensor([[False, False, False, True], [False, False, False, False]])
target_pad = torch.tensor([[False, False, True], [False, False, False]])
memory, _ = encoder(source, padding_mask=source_pad)
output, weights = decoder(target, memory,
                          target_padding_mask=target_pad,
                          source_padding_mask=source_pad)
```

`decoder_hidden` 现在由 `DecoderLayer.forward()` 实际计算，而不再只是示意变量。memory 由外部 Encoder 生成并传入，不 detach，梯度可以回传。`target_padding_mask` 只传给因果自注意力；`source_padding_mask` 只传给交叉注意力。没有 PAD 时对应参数可省略，memory 不可为 None。

输出为 `(...,d_model,T)`；返回的 `weights['self']` 为 `(...,heads,T,T)`，`weights['cross']` 为 `(...,heads,S,T)`，都是 Dropout 前的概率。源/目标长度允许不同。PAD query 的最终输出不保证为零，后续任务 loss 仍需忽略目标 PAD。

`tests/test_decoder_layer.py` 验证三条分支与手算归一化一致、独立参数、Dropout 不丢弃残差直通路径、完整 Block 的未来隔离、两个 mask 正确路由、Batch/未补齐单条对照、左 PAD 数值稳定性，以及梯度能够从 Decoder 回传到真实 Encoder。

## Decoder 第四步：多层堆叠

新增 `decoder.py` 的 `Decoder`，通过 `nn.ModuleList` 注册多个独立的 `DecoderLayer`。默认 3 层，共 5130 个参数。各层结构相同、参数独立，上一层的输出作为下一层的目标输入；所有层都读取同一份 Encoder 最终输出 memory。

沿用上面示例的 source、target 和两个 mask，将单层 Decoder 替换为：

```python
from decoder import Decoder

encoder = Encoder(num_layers=2)
decoder = Decoder(num_layers=3)
memory, _ = encoder(source, padding_mask=source_pad)
output, layer_weights = decoder(
    target, memory,
    target_padding_mask=target_pad,
    source_padding_mask=source_pad,
)
```

Encoder 和 Decoder 的层数可以不同。输出仍为 `(...,d_model,T)`；`layer_weights` 是按层排列的列表，每项包含 `self` 和 `cross` 两种注意力权重，形状分别为 `(...,heads,T,T)` 和 `(...,heads,S,T)`。

每层都接收目标 padding mask 和源 padding mask，分别用于自注意力与交叉注意力；因果 mask 由每层自注意力内部应用。memory 不 detach，因此 Decoder 的损失可以训练 Encoder。`dropout` 配置传入每个 Block，调用 `decoder.eval()` 会关闭所有层的 Dropout。

`tests/test_decoder.py` 验证层数校验、参数独立、与手动逐层调用一致、同一 memory 传递、完整堆叠的因果性、两种 PAD 隔离、左 PAD 数值稳定性，以及优化器实际更新每层 Encoder/Decoder 参数。

当前完成了 Decoder 多层堆叠；尚未加入目标 Token Embedding 封装、词表预测头、teacher forcing 训练和自回归生成。

## Decoder 第五步：目标 Token Embedding 与词表预测层

`token_decoder.py` 提供两层封装：`TokenDecoder` 输出隐藏向量；`TokenDecoderLM` 在其后增加 `nn.Linear(d_model, vocab_size)`，输出目标词表分数。

具体流程（B 为 batch，T 为目标长度，S 为源长度，V 为目标词表大小）：

1. 目标 ID `(B,T)` 通过 `nn.Embedding(V,d_model)` 查表，得到 `(B,T,d_model)`。表随机初始化一次，训练更新表中的向量，ID 本身不更新。源/目标 Embedding 独立，词表大小可以不同。
2. 转置成 `(B,d_model,T)`。默认加固定 sin/cos 位置编码；`none` 不加，`rope` 在每层目标自注意力的 Q/K 内旋转。与现有 TokenEncoder 一致，暂不乘 `sqrt(d_model)`。两侧位置各自从 0 开始，固定位置编码不参与优化。
3. 经过输入 Dropout。`train()` 时启用，`eval()` 时关闭；默认 dropout=0。
4. 将向量传入堆叠 Decoder，每层读取同一份 `(B,d_model,S)` memory。目标 PAD 根据 `padding_idx` 自动识别；源 PAD 必须通过 `source_padding_mask` 传入，不能从 memory 数值推断。每层自注意力仍使用 causal mask。
5. Decoder 输出转回 `(B,T,d_model)`，再经过共享于各位置的 FC 层，得到 `(B,T,V)` logits。不池化，不先做 softmax；交叉熵直接接收 logits。需要查看概率时才使用 `logits.softmax(dim=-1)`。本版本预测头与 Embedding 不共享权重。
6. `loss.backward()` 计算梯度，`optimizer.step()` 更新注册参数，包括源/目标 Embedding、两侧堆叠层和预测头。memory 不 detach，梯度能回传到 Encoder。

下面是在本目录内可运行的一次训练更新示例。PAD=0、BOS=1、EOS=2 均由调用方定义；模型不会自动插入这些 ID，也不会自动右移输入。

```python
import torch
from torch import nn
from token_encoder import TokenEncoder
from token_decoder import TokenDecoderLM

encoder = TokenEncoder(vocab_size=9, num_layers=2, padding_idx=0)
decoder = TokenDecoderLM(vocab_size=7, num_layers=3, padding_idx=0)
source_ids = torch.tensor([[3, 4, 5, 0], [4, 5, 0, 0]])
# 目标完整序列为 [BOS, 3, 4, EOS] 和 [BOS, 5, EOS, PAD]。
full_target = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
decoder_input_ids = full_target[:, :-1]  # [BOS,3,4] / [BOS,5,EOS]
labels = full_target[:, 1:]             # [3,4,EOS] / [5,EOS,PAD]
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
encoder.train()
decoder.train()
optimizer.zero_grad()
memory, _ = encoder(source_ids)
logits, layer_weights = decoder(
    decoder_input_ids, memory, source_padding_mask=source_ids == 0)
loss = nn.CrossEntropyLoss(ignore_index=0)(
    logits.reshape(-1, 7), labels.reshape(-1))
loss.backward()
optimizer.step()
```

输入的 BOS 位置预测第一个目标 token；后续位置预测下一个 token，最后有效位置预测 EOS。输入 PAD mask 与标签忽略是两件事：mask 屏蔽作为 K/V 的 PAD，`ignore_index` 则让 PAD 标签不产生损失。PAD query 的 logits 不保证为零。

单条序列也可直接传 `(T,)` ID 和 `(d_model,S)` memory，返回 `(T,V)` logits。`tests/test_token_decoder.py` 覆盖三种位置模式、逐位置 FC 手算、Batch 与未补齐单条一致性、双 mask、因果性、Dropout、输入校验，以及一次真实交叉熵更新中的梯度和 PAD 保护。

本步完成 ID 到词表 logits 的连接，并以测试和上面的单步示例验证训练链路；完整序列任务的 teacher forcing 训练脚本、验证指标和逐 token 自回归生成留待下一步实现。

## Decoder 第六步：自回归生成

`generation.py` 提供 `greedy_generate(encoder, decoder, source_ids, bos_id, eos_id, max_new_tokens=50)`。第一版支持单条源序列，采用贪心解码，每步重新计算完整目标前缀，尚未使用 KV cache。

生成过程：

1. 根据 Encoder 的 `padding_idx` 构造源 PAD mask，并且只运行一次 Encoder 得到 memory。
2. 用目标 BOS ID 初始化目标序列，例如 `[1]`。
3. 将当前完整前缀与同一份 memory 送入 Decoder，取最后位置的 logits。
4. 屏蔽 BOS 和目标 PAD 的候选分数，再用 argmax 选择分数最高的 token。EOS 仍是可选候选。这是对合法候选的贪心解码，无需先做 softmax。
5. 将预测 ID 接到前缀末尾。如果它是 EOS 则结束，否则继续。即使没有预测 EOS，达到 `max_new_tokens` 也会停止，不强行追加 EOS。

返回的一维张量包含初始 BOS，以及实际选中的 EOS（如果有）。长度上限只计算新生成 token，不包含 BOS；上限为 0 时直接返回 `[BOS]`。函数自动关闭梯度与 Dropout，并在结束或异常后恢复原来的训练状态，不更新模型参数。

沿用上一步定义的 encoder、decoder、source_ids，可运行：

```python
from generation import greedy_generate

generated = greedy_generate(
    encoder, decoder, source_ids[0], bos_id=1, eos_id=2, max_new_tokens=10)
print(generated.tolist())
```

模型和输入应在同一设备上。这个例子展示调用流程；随机初始化或只训练一步的模型尚不具备有意义的生成能力。

EOS 是预先加入目标词表的特殊 token，有固定 ID。例如目标词表可以为 `{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "我": 3, "爱": 4, "你": 5}`，此时 `vocab_size=6`。训练时将 `[BOS, 我, 爱, 你]` 作为输入，`[我, 爱, 你, EOS]` 作为标签，最后一个位置的交叉熵教模型预测 EOS。EOS 对应词表预测头的一个输出类别，和普通 token 一样由模型预测；生成循环只负责识别这个 ID 并停止。

本仓库的模型接收 ID，不自动维护词表；词表映射、BOS/EOS ID 和训练标签必须由调用方保持一致，BOS、EOS、目标 PAD 要使用不同 ID。当前使用独立输出头，因此 EOS 的输出权重通过交叉熵学习；EOS 的 Embedding 行只有在它作为输入时才可能获得查表梯度。

`tests/test_generation.py` 验证前缀逐步增长、只取最后位置、Encoder 只计算一次、memory 与源 mask 复用、EOS 提前停止、长度限制、BOS/PAD 候选过滤、真实模型推理、参数不变及正常/异常情况下的模式恢复。完整序列任务的训练脚本和生成质量评估仍留待下一步。

## 完整序列任务：训练模型生成倒序序列

在仓库根目录运行（CPU 即可）：

```bash
.venv/bin/python 06-attention/train_seq2seq.py
```

`sequence_task.py` 固定 PAD=0、BOS=1、EOS=2，普通 token 为 3～12。随机生成长度 2～6 的源序列，允许重复 token；所有源序列去重后划分为 1024 条训练数据和 128 条验证数据，两者没有相同源序列。相同种子可复现划分。

数据处理以 `[3,5,7]` 为例：

```text
source_ids:        [3,5,7]
decoder_input_ids: [BOS,7,5,3]
labels:            [7,5,3,EOS]
```

每条样本先独立构造输入和标签，再分别右侧补 PAD。这样输入无需包含 EOS；在较短样本尾部，输入和标签同时为 PAD。它与先补完整目标再切片的方案在有效标签位置上等价。

`train_seq2seq.py` 的训练流程：

1. 创建独立 TokenEncoder 与 TokenDecoderLM：各 2 层，d_model=32，2 个注意力头，每头 Q/K/V 维度为 16，FFN 隐层 128，固定 sin/cos 位置编码。为便于观察这个小任务的学习过程，dropout=0。
2. 每个 batch 重新计算源 memory，使用真实目标前缀进行 teacher forcing，预测各位置的下一个 token。
3. 对非 PAD 标签计算交叉熵；Adam 同时更新两侧 Embedding、Encoder、Decoder、预测头。学习率 0.001，梯度范数裁剪上限为 1，默认训练 40 轮。
4. 每轮在 eval/no_grad 下评估训练和验证损失、token 准确率。损失按有效 token 数加权，包含 EOS，排除 PAD。
5. 训练结束后，对整个验证集逐条从 BOS 开始生成，最多生成 8 个新 token。整句准确率要求内容、长度和 EOS 全部匹配；EOS 率单独表示是否生成了结束符。生成时不提供真实目标前缀。
6. 保存 `artifacts/seq2seq/report.json`（配置、历史指标、生成示例）和 `artifacts/seq2seq/model.pt`（模型结构配置、两侧权重和特殊 token ID）。权重文件被 gitignore 忽略；此文件用于推理，不包含恢复训练所需的 Adam 状态。

可选参数包括 `--epochs`、`--train-size`、`--validation-size`、`--batch-size`、`--seed`、`--output-dir`。

默认种子 42 的一次 CPU 实测：验证损失从 2.7146 降至 0.0049，teacher forcing token 准确率约 99.9%；自回归整句准确率为 127/128（99.2%），EOS 率为 100%。这仅验证长度 2～6、同分布未见序列上的倒序能力，不代表自然语言能力或更长序列上的泛化。

在 `06-attention` 目录中加载保存的模型进行预测：

```python
import torch
from train_seq2seq import load_model

model = load_model('artifacts/seq2seq/model.pt')  # 兼容新旧权重格式。
result = model.generate(torch.tensor([3, 5, 7]), bos_id=1, eos_id=2, max_new_tokens=8)
print(result.tolist())  # 包含 BOS 和模型实际生成的 EOS。
```

`tests/test_sequence_task.py` 验证数据划分无重复、目标右移和 EOS/PAD 位置、验证损失的 token 加权、生成指标、评估状态恢复，并训练一个小样本任务达到完整生成正确，再检查权重保存/加载后的结果一致。

## 统一的 Transformer 模型接口

`transformer.py` 的 `Transformer(nn.Module)` 将现有 TokenEncoder、TokenDecoderLM 注册为 `model.encoder` 和 `model.decoder`。这是接口整理，内部 Attention、FFN、位置编码和生成算法不变。

```python
from transformer import Transformer

model = Transformer(
    source_vocab_size=13, target_vocab_size=13,
    source_padding_idx=0, target_padding_idx=0,
)
source_ids = torch.tensor([[3, 5, 7]])
target_input_ids = torch.tensor([[1, 7, 5, 3]])
logits = model(source_ids, target_input_ids)  # (1,4,13)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

`forward` 内部完成源查表与 Encoder、源 PAD mask 构造、目标查表与 Decoder、词表投影。源/目标词表大小、PAD ID 和堆叠层数可以分别配置。输入支持单序列或 batch；训练代码仍负责目标右移、标签、交叉熵和优化器更新，模型不会自动改变目标输入。

需要观察注意力时调用 `logits, weights = model(source_ids, target_input_ids, return_attention=True)`；`weights['encoder']` 为逐层 Encoder 权重，`weights['decoder']` 为逐层 self/cross 权重。

生成使用 `model.generate(source_ids[0], bos_id=1, eos_id=2, max_new_tokens=8)`，沿用单序列贪心生成。`model.train()`、`model.eval()` 和 `model.to(device)` 会递归作用于两侧组件，`model.parameters()` 包含两侧所有可训练参数。

倒序脚本已改用 `build_model()`、`train_model(model, ...)` 和 `evaluate(model, ...)`。新 checkpoint 保存 `model.state_dict()` 与完整 `model.config`，`load_model()` 同时支持旧的 encoder/decoder 分开存储格式；之前的已训练权重无需重训。新的完整配置也可以通过 `Transformer(**model.config)` 重建模型。

`tests/test_transformer.py` 验证封装与手动调用组件的结果一致、不同源/目标 PAD ID 正确传递、统一优化器更新两侧参数、生成一致性以及旧 checkpoint 加载；倒序任务测试继续覆盖新 checkpoint 保存与加载。

## 连续文本预训练：从书籍中自动构造标签

这是另一条教学路线：`causal_lm.py` 实现 Decoder-only 模型，每层只有因果自注意力、FFN 及两次残差/LayerNorm，没有 Encoder 和 Cross-Attention。现有翻译 Transformer 保留。默认模型约 16.6 万参数，2 层、d_model=64、4 头、FFN=256、Dropout=0.1、上下文 64 token，适合先在本机观察。

默认数据来自 [Project Gutenberg 的 Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11)，页面标注美国公版。下载时保留带来源与许可的原文，同时记录 URL、字节数和 SHA256；处理后的文本仅用于本地实验。也可通过 `--text-file path/to/book.txt` 使用自己的 UTF-8 文本（空行分段）。数据和模型保存在被 gitignore 忽略的 `artifacts/book-lm/`，不会随代码上传。

以下命令都在仓库根目录运行。

### 1. 准备数据：看原文如何变成训练材料

```bash
.venv/bin/python 06-attention/train_lm.py prepare
```

`text_pretraining.py` 去掉 Gutenberg 外壳与示例书目录，合并段落内换行，过滤不足 40 字符的段落并去重。按原书顺序划分 80%/10%/10% 段落为训练/验证/测试集，之后才切窗口。512 词表的 ByteLevel BPE 只在训练段落上学习；完整字节 alphabet 可表示未见字符，不需要拿测试集扩词表。

查看 `artifacts/book-lm/data/`：`original.txt` 是保留的原文，`train.json`、`validation.json`、`test.json` 是可直接阅读的段落，`tokenizer.json` 为实际分词器，`source.json` 为来源与划分记录。这只是基础清洗，不保证彻底排除书内近重复内容。

每个段落作为一份独立短文档，编码为 `[BOS, 文本tokens..., EOS]`。EOS 在此表示段落结束。长段落切成窗口，仅原段落末尾有 EOS；窗口边界不伪造 EOS。窗口之间不保留模型状态，不学习跨段落长程关系。

### 2. 检查输入与标签：先不训练

```bash
.venv/bin/python 06-attention/train_lm.py inspect
```

终端与 `artifacts/book-lm/run/inspect.json` 显示：

- `first_original_paragraph`：对应原文。
- `input_text`：第一个窗口还原出的文本。
- `input_ids` / `labels`：一个 batch 的实际 ID 矩阵。
- `rows`：逐位置的输入 token、输入 ID、标签 token、标签 ID。

核心操作在 `make_windows()`：`input=piece[:-1]`，`label=piece[1:]`。例如 `[BOS,A,B,C,EOS]` 变为输入 `[BOS,A,B,C]`、标签 `[A,B,C,EOS]`。BPE 的一个 token 不一定是一个完整单词，显示中的 `Ġ` 是 ByteLevel 空格标记。PAD 输入 ID=0；标签补齐为 -100，由交叉熵忽略。

### 3. 训练：看前向输出、梯度和参数变化

```bash
.venv/bin/python 06-attention/train_lm.py train --epochs 5
```

模型数据流为 `ID (B,T) → Embedding → 位置编码 → 因果 Blocks → FC → logits (B,T,512)`。例子中 batch=16、T=64，因此 logits 是 `(16,64,512)`；每个位置都有 512 个候选 token 分数。

`FIRST STEP` 日志打印实际输入/输出形状、loss、Embedding 梯度范数与 optimizer.step 后的权重变化范数。训练顺序为清梯度、前向、交叉熵、backward、梯度裁剪、Adam.step；标签来自原文，无需人工问答标注。

每轮日志及 `report.json` 包含：

- `loss`：有效位置的平均交叉熵，越低表示给真实下一个 token 的概率越高。
- `perplexity`：`exp(loss)`，越低越好；只在相同 tokenizer/数据处理下比较。不是生成准确率。
- `token_accuracy`：给定真实前缀时，最高分 token 命中标签的比例，不代表独立续写质量。
- train 与 validation 的差距：训练损失下降而验证损失上升，可能是过拟合。

评估使用 eval/no_grad；训练只更新训练数据上的参数。仅根据 validation loss 选择 `best.pt`，里面包含模型配置、权重和 tokenizer，不包含完整断点续训状态。`predictions.json` 展示最佳模型在验证样本每个位置上的真实标签和 top-3 候选及概率。

本机已经验证 MPS 前向/反向可运行；可使用 `--device mps` 训练。默认 CPU，短序列小模型上 MPS 不保证更快。保持数据和参数一致，实测后再选择。

### 4. 测试：评估未参与选模型的留出集

```bash
.venv/bin/python 06-attention/train_lm.py test
```

该命令加载 best.pt，单独读取 test.json，输出 loss、perplexity、token_accuracy 和有效 token 数，并保存 run/test.json。测试过程不训练。不要反复根据测试结果调整配置；调参只看验证集。这是同一本书后段的留出评估，不代表其他书籍或通用知识能力。

### 5. 续写：看自由生成与给定真实前缀评估的区别

```bash
.venv/bin/python 06-attention/train_lm.py generate --prompt "Alice was" --max-new-tokens 32
```

输出 prompt、prompt_ids、output_ids 和解码后的 text。程序将模型自己预测的 token 追加到前缀，继续预测，而不是提供原文答案。当前使用贪心解码，遇 EOS 停止；输出不包含参考续文的比较分数。

超过上下文长度时只看最后 64 个 token，并重新从位置 0 编码，这是教学滑动窗口实现；尚无 KV cache。小数据、短时间训练下出现重复、拼写错误或不通顺是可预期的，不应把损失下降当作已获得成熟语言能力。

命令支持 `--data-dir`、`--run-dir` 分离不同实验。重新 prepare 会改变数据与词表，因此旧 checkpoint 应配套使用原数据目录；test 会校验 tokenizer 哈希。新实验使用新 run-dir，避免覆盖之前模型。

测试文件 `tests/test_text_pretraining.py` 检查数据划分、字节分词还原、下一 token 错位、窗口标签覆盖、PAD 忽略、未来隔离、真实小样本学习、EOS 生成、checkpoint 还原和验证指标加权。

### 6. 逐轮观察报告：模型到底学到了什么

训练命令现在自动记录第 0 轮（随机初始化）和每个训练轮次。示例：

```bash
.venv/bin/python 06-attention/train_lm.py train --epochs 5 --run-dir 06-attention/artifacts/book-lm/observation-run
open 06-attention/artifacts/book-lm/observation-run/training_observation.html
```

这是本地离线 HTML 报告，无需启动服务。报告引用同目录的 `curves.png`，分享时应一起保留。内容包括：

- 训练/验证 loss、perplexity、token accuracy 曲线，标出最佳验证轮次。困惑度使用对数纵轴。
- 固定提示词 `Alice was`、`The rabbit`、`She said` 在每一轮的完整续写。所有轮次均使用贪心解码、最多 32 个新 token，并关闭 Dropout。
- 同一个验证窗口的输入文本、每个位置的真实下一 token、真实标签概率、前三候选及命中标记。PAD 标签不计入命中率。

点击轮次展开对照。真实标签概率变高、进入 top-3 的位置变多，表示这个窗口上的预测在改善，但不保证每个位置单调改善；整体判断仍看完整验证集。自由续写使用模型自己的输出，可能累积错误，因此验证 loss 下降不意味着续写一定流畅。

`observations.json` 保存逐轮原始记录与指标；`report.json` 仍保存训练汇总；`predictions.json` 仍是最佳 checkpoint 的验证预测。逐轮报告不会把所有轮次替换成最佳模型，也不读取测试集。

默认配置实测：固定验证窗口 top-3 命中从第 0 轮 0/40 增加到第 5 轮 20/40；自由续写仍会重复与拼写出错。观察函数在 no_grad/eval 下执行，恢复各模块原先状态，测试确认参数和训练随机数状态不变；同种子实验的训练指标与加入观察前完全一致。

### 7. 现代结构与实时 TensorBoard

上面第 6 节的数值为旧版 Post-LN/ReLU/sin-cos/恒定学习率实验。现在新建 CausalLM 的默认配置升级为以下五项；KV cache 见下一节：

| 项目 | 当前行为 | 切回旧设置 |
|---|---|---|
| Pre-Norm | 子层先 norm，再 Attention/FFN，最后残差相加；全部 Blocks 后还有 final norm | `--norm-style post` |
| RMSNorm | 每 token 除以均方根，使用可学习 gamma，无 bias；半精度用 float32 统计 | `--norm-type layer` |
| RoPE | 每个头的 Q/K 内旋转，不再在 Embedding 处加 sin/cos | `--position-encoding sinusoidal` |
| SwiGLU | `down(silu(gate(x)) * up(x))`，三投影无 bias | `--ffn-type relu` |
| Warmup + cosine | 前 10% 优化步骤线性升至峰值，之后余弦降到峰值的 10% | `--lr-schedule constant` |

默认 d_model=64，SwiGLU 隐层为 176（约 8/3*d，向上对齐 8），参数量与旧 4*d ReLU FFN 接近但不完全相同。RoPE 复用已有实现；生成仍按窗口重算。现有翻译 Encoder–Decoder 不受此次修改影响。

学习率是按 optimizer 更新步骤计算，第一步不跳过 warmup；`--learning-rate` 设置峰值，`--warmup-ratio` 设置 warmup 比例，`--min-lr-ratio` 设置最终比例。新 checkpoint 保存完整模型配置与训练配置；旧 checkpoint 缺少结构字段时，加载器按 Post-LN/LayerNorm/sin-cos/ReLU 重建。仍只提供推理 checkpoint，不承诺断点续训。

现成的实时可视化已接入 **TensorBoard**。依赖已写入 pyproject.toml / uv.lock，重建环境运行 `uv sync`。不需要安装 TensorFlow，不需要外部账号。开两个终端：

```bash
# 终端 1：本地仪表盘，浏览器访问 http://127.0.0.1:6006/
.venv/bin/tensorboard --logdir 06-attention/artifacts/book-lm --host 127.0.0.1 --port 6006 --reload_interval 5
```

```bash
# 终端 2：新版训练，使用独立目录，避免覆盖旧模型
.venv/bin/python 06-attention/train_lm.py train --epochs 5 --run-dir 06-attention/artifacts/book-lm/modern-run
```

训练日志保存到每个 run 的 `tensorboard/`。训练中每 10 步（可用 `--log-every` 调整）写入并刷新：

- `optimization/loss`：该记录步骤的 minibatch loss，不是整轮平均。
- `optimization/learning_rate`：该次参数更新实际使用的学习率。
- `optimization/gradient_norm`：裁剪前的全模型梯度范数，裁剪阈值仍为 1。
- `optimization/tokens_per_second`：日志区间内有效 token 吞吐量，不含每轮验证时间，不代表纯 GPU 内核速度。

每轮增加完整 train/validation 的 loss、perplexity、token accuracy；Text 页显示配置、固定提示词续写和验证窗口 top-3 候选。横轴为 optimizer global step（不是 epoch）；文本中标注 epoch。`test` 命令独立记录 `test/*`，不会为了刷新网页而反复评估测试集：

```bash
.venv/bin/python 06-attention/train_lm.py test --run-dir 06-attention/artifacts/book-lm/modern-run
```

使用以下命令跑旧配置对照，网页中勾选两个 run 即可比较。不同实验使用新的 run-dir，避免把多个训练的 event 文件混到同一个 run。相同种子不意味着不同结构具有逐参数相同的初始化，且一次改变五项不能归因到单独组件。

```bash
.venv/bin/python 06-attention/train_lm.py train --epochs 5 --norm-style post --norm-type layer --position-encoding sinusoidal --ffn-type relu --lr-schedule constant --run-dir 06-attention/artifacts/book-lm/baseline-run
```

实测新版 5 轮验证 loss=4.0215，旧版为 3.9132；小数据短训练没有显示新版优于旧版，不应把结构升级当成效果保证。新版 RMSNorm/SwiGLU/RoPE 的 MPS 前向反向、学习率边界、新旧 checkpoint、实时日志可读性均有验证。

TensorBoard 支持长期记录和多实验对照，但它不会自动让数据加载变成流式：当前书籍脚本仍将分词后的窗口放在内存中，并每轮完整验证。扩大到大量书籍前还需要分片/流式 Dataset、可配置验证频率和完整训练恢复。云端租 GPU 时，可用 SSH 端口转发查看远端的本地 TensorBoard，不必公开仪表盘端口。

### 8. KV cache：生成时复用各层历史 K/V

`CausalLM.generate()` 现在默认 `use_cache=True`，旧模型无需重训。训练 `model(ids)` 仍返回 logits，仍对整个序列做前向与反向，不启用缓存。

生成分两阶段：第一次 prefill 完整提示词，在每层/每头保存 K、V 和历史 PAD mask；随后 decode 只输入新增 token，在每层计算它的 Q/K/V，再将新的 K/V 追加到旧缓存，得到当前位置的输出。历史 Q 不保存。RoPE 缓存的是已经旋转的 K，V 不旋转，新增 Q/K 使用从缓存长度开始的位置。旧版 sin/cos 模式也使用相同的位置偏移。

```bash
# 默认开启
.venv/bin/python 06-attention/train_lm.py generate --run-dir 06-attention/artifacts/book-lm/modern-run --prompt "Alice was" --max-new-tokens 32
# 完整重算，对照输出
.venv/bin/python 06-attention/train_lm.py generate --run-dir 06-attention/artifacts/book-lm/modern-run --prompt "Alice was" --max-new-tokens 32 --no-kv-cache
```

低层接口供观察缓存用，传入 ids 仅包含新增部分：

```python
model.eval()
with torch.no_grad():
    logits, cache = model(prompt_ids, use_cache=True)
    # next_ids 可以是一枚 token，也可以是连续的一小段新 token。
    logits, cache = model(next_ids, use_cache=True, kv_cache=cache)
```

`cache` 是按层排列的列表，每层包含 `k`、`v`（各头张量列表）和累计 `padding_mask`。单条序列每头 K 形状为 `(d_k, 已缓存长度)`，V 为 `(d_v, 已缓存长度)`；batch 在前面多一维。模型低层支持 batch，但 generate 仍只支持单序列。缓存不得跨不同模型参数或不同请求复用；generate 使用局部缓存，不污染下一次调用、不保存到 checkpoint。缓存路径要求 eval + no_grad，避免误当作训练优化。

**上下文窗口的限制：**原来的生成算法在超过 context_length 时截掉最早 token，并把窗口位置重新从零编码。保留语义需要重新计算窗口内各层表示，不能只删掉缓存里最早的 K/V。因此当前在窗口满后每一步重建缓存；未满窗口时才走增量加速。低层 forward 遇到缓存加新输入超长会明确报错，generate 自动重建。进一步做持续滑动缓存需要单独设计模型上下文语义。

测试覆盖新旧结构的 prefill、逐 token 和多 token 解码、PAD 与 batch、RoPE/sin-cos 位置偏移、窗口重建、EOS 停止、模式恢复、缓存隔离及正常训练梯度。以 3-token 提示生成 4 个新 token 为例，未满窗口时实际前向输入长度是 `[3,1,1,1]`，完整重算则为 `[3,4,5,6]`。浮点运算顺序可能产生微小差异，测试比较 logits 的数值容差，并对真实 checkpoint 检查生成 ID 一致性。

## IndustryCorpus2 中文小样本预训练

复用本目录的 `CausalLM`，不加载已有模型权重。需要先在 Hugging Face
的 `BAAI/IndustryCorpus2` 页面获得访问权限，并在本机 `hf auth login`。

```bash
uv sync --extra pretraining
.venv/bin/python 06-attention/prepare_industry.py
.venv/bin/python 06-attention/train_lm.py train \
  --data-dir 06-attention/artifacts/book-lm/industry-data \
  --run-dir 06-attention/artifacts/book-lm/industry-33m-run \
  --epochs 1 --d-model 512 --num-layers 8 --num-heads 8 \
  --context-length 256 --batch-size 2 --device mps --learning-rate 0.0003 \
  --observation-prompts 人工智能 机器学习的主要方法 数据
.venv/bin/python 06-attention/train_lm.py test \
  --data-dir 06-attention/artifacts/book-lm/industry-data \
  --run-dir 06-attention/artifacts/book-lm/industry-33m-run --device mps --batch-size 2
.venv/bin/python 06-attention/train_lm.py generate \
  --run-dir 06-attention/artifacts/book-lm/industry-33m-run \
  --device mps --prompt 人工智能 --max-new-tokens 64
```

抽样器固定版本和中文 high/人工智能分片，读取其第一个 Parquet row group 的
text 列；该分片只有一个 row group，所以网络读取规模仍接近一个分片（约 280MB），
不是只下载最终样本的 2.6MB。清理空白、完整文档去重后，保留 200–20000 字符的文档，
固定种子打乱，选取累计约 100 万字符。它是单行业试跑样本，不代表整个数据集。

按文档 80/10/10 划分，训练集单独训练 8192 词表 ByteLevel BPE。
现有窗口逻辑在文档内切分，每篇文档末尾添加 EOS；不将同一文档的不同窗口分到不同集合。
`source.json` 保存来源版本、抽样规则、原文哈希和文档哈希。
数据目录已存在时抽样器拒绝覆盖；训练重复实验也应换新的 run-dir。

本次配置为 33,604,096 参数。训练沿用 Adam、梯度裁剪、warmup/cosine、TensorBoard
和逐轮 HTML 报告。默认模型仍为两层、64 维，旧书籍实验命令保持有效。
`--observation-prompts` 可指定中文观察提示词。`best.pt` 是推理权重，尚不支持完整断点续训。
一遍小样本用于验证训练链路和损失变化，不足以训练出通用中文语言模型。

本地训练时同时启动 TensorBoard 并打开浏览器，方便实时观察（用户偏好）：

```bash
.venv/bin/tensorboard --logdir 06-attention/artifacts/book-lm --host 127.0.0.1 --port 6006 --reload_interval 5
# 另一个终端（macOS）
open http://127.0.0.1:6006/
```

### 判断是否超出词频记忆

```bash
.venv/bin/python 06-attention/unigram_baseline.py --data-dir 06-attention/artifacts/book-lm/industry-data
```

基线只在 train 标签上计数、add-one 平滑，不读取测试集。该样本验证 loss 为
7.6474，第一轮模型为 7.5759，仅略优于不使用上下文的词频预测。

继续训练可使用 `--init-checkpoint <旧run>/best.pt`，保存到新的 run-dir。
它加载旧模型的结构和权重，要求 tokenizer 哈希和上下文长度一致；优化器和学习率计划
重新初始化，因此不是精确断点恢复。新 run 的 epoch=0 表示加载后的模型，
并将其纳入最佳验证 checkpoint 的选择，防止后续训练变差时覆盖更好的起点。
这次继续训练采用 3 epochs、batch=8、peak lr=0.0003，其余配置与首次一致。
对比使用验证集和固定生成提示词，不再次使用测试集调参。

追加训练实测（`industry-33m-continue`，同一批文档）：

| 追加轮次 | train loss | validation loss |
|---|---:|---:|
| 0（原权重） | 7.5470 | 7.5759 |
| 1 | 7.4668 | 7.5358 |
| 2（最佳） | 7.3835 | 7.5097 |
| 3 | 7.3492 | 7.5121 |

词频基线 validation loss=7.6474。第三遍训练损失继续下降但验证略回升，
最佳权重选追加第 2 遍；固定提示词续写仍有严重重复，不能把这次损失改善视为
已经获得流畅中文能力。本次未读取测试集。后续应扩大并清理语料后再评估，
不优先扩大模型或无限重复这批文档。TensorBoard 对比面板可用根目录 logdir，
其中 `industry-unigram` 为不使用上下文的水平参考线。

### 进度、耗时与诊断（2026-11-15）

`train_lm.py train` 现在显示 tqdm 总更新步数、epoch、当前 batch loss 和 ETA。
TensorBoard `time/*` 包含 elapsed_seconds、update_seconds、evaluation_seconds、
eta_updates_seconds、eta_total_seconds、initial_evaluation_seconds、total_wall_seconds。
首次完成整轮评估前，ETA 不含评估；之后按已完成轮次的平均评估耗时估计。
ETA 不保证准确，最终模型恢复与报告生成也会产生额外耗时。
`time/total_wall_seconds` 是结束时含初始评估的实测总耗时。
Text 中的 `generation/*`、`current_model/*` 展示生成，`validation/fixed_predictions`
展示给定真实前缀的预测；它们与自由生成是不同任务。

```bash
# 查看完整对比面板（6007 未占用时）
.venv/bin/tensorboard --logdir 06-attention/artifacts/book-lm --host 127.0.0.1 --port 6007
open http://127.0.0.1:6007/
# 重现诊断：两组各 160 步，从零初始化；不读取 test，不覆盖原 checkpoint
.venv/bin/python 06-attention/audit_industry.py --steps 160
# 导出当前最佳 checkpoint 的真实张量与逐步交互展示
.venv/bin/python 06-attention/build_industry_flow.py
```

可视化输出为 `output/industry-flow/flow.html`；模板位于 `assets/industry-flow.template.html`。
依赖预先生成的 industry-data 与 industry-33m-continue/best.pt。
首次重建若未保存原文样本，会读取固定源分片并匹配验证集第一篇文档。
显示原文、清理、字节/BPE、完整序列、256-token 窗口、标签右移、padding、
Embedding 数值、8 层输出、因果遮罩、Top-5 概率和逐位置 loss。
反向传播演示只更新内存副本，绝不覆盖 checkpoint。

**诊断证据与局限：**

- train/validation 全量分词还原一致、标签右移和窗口目标覆盖检查通过；无完全重复文档。
  这不代表已排除近重复、模板文本或语料质量问题。
- 独立 SDPA 对照验证了注意力前向与输入梯度；修改未来输入时，过去输出差异为 0。
- 旧注意力所有投影固定 std=0.1。扩到 512 维时，第一层 score std=4.918，
  注意力分支 RMS=3.799，而残差输入 RMS=0.989。8 层后初始化输出 RMS=11.312。
- 仅将注意力投影改为 std=1/sqrt(d_model)，其他随机参数、训练批次和 dropout 种子一致，
  score std=0.960，第一层分支 RMS=0.241，8 层输出 RMS=1.793。
  160 步后，同一固定 32 窗口验证探针（7703 个目标 token），旧方案 loss=7.6853，
  新方案=7.4233。仅单种子短实验，不能当作完整验证集或多种子结论。
- 新增 `--attention-init fan_in`，仅用于新训练。新训练默认 fan_in；显式 legacy 可复现历史行为。
  checkpoint 保存配置；旧 checkpoint 无该字段仍按 legacy 加载。
  加载旧权重时此选项不会重置已训练参数，验证新初始化需另开从零训练实验。
- 当前模型的上下文探针（107 个验证位置）完整前缀 loss=7.5549，仅当前 token 为 7.7422。
  表明模型使用了一些上下文，但样本有限，且自由续写仍严重重复。
- 训练目标中逗号占约 7.4%，因此只猜逗号也有接近模型的 token accuracy。
  抽查源文档本身即使用英文逗号，不是本地分词把中文标点损坏。
- 320 篇、46 万目标 token、单一行业且仅 256 上下文，仍不足以支撑该模型的通用中文能力。
  当前空白折叠会丢失段落格式；过滤只覆盖长度与精确重复，尚无近重复或模板清理。

先前将主要原因归于数据量不足的解释不完整；初始化随扩模失衡已有直接对照证据。
建议先以 fan_in 做同数据完整训练对照，再扩大和清理语料，不通过改变解码规则掩盖重复。

### 修正初始化后的完整试跑

新建 `CausalLM` 和 CLI 新训练现在均默认 `attention_init=fan_in`，注意力投影
标准差为 `1/sqrt(d_model)`。底层教学用 `MultiHeadAttention` 保留原行为。
加载历史 checkpoint 时，缺失的 `attention_init` 明确补为 `legacy`，随后加载全部权重；
不会重初始化已训练的参数。需要复现旧实验时显式传 `--attention-init legacy`。

```bash
.venv/bin/python -u 06-attention/train_lm.py train \
  --data-dir 06-attention/artifacts/book-lm/industry-data \
  --run-dir 06-attention/artifacts/book-lm/industry-33m-fanin-run \
  --epochs 4 --d-model 512 --num-layers 8 --num-heads 8 \
  --attention-init fan_in --context-length 256 --batch-size 8 \
  --device mps --learning-rate 0.0003 \
  --observation-prompts 人工智能 机器学习的主要方法 数据
```

这次从零训练，保持语料、词表、模型规模不变，完整验证集每轮评估，测试集不参与。
与历史实验的对比不是严格的单变量消融：历史记录为 batch=2 的一遍训练，
加上重置优化器后 batch=8 的三遍训练；本次为连续四遍 batch=8、一个学习率计划。
初始化的单变量证据来自前面的 160 步双组对照，而本次用于验证完整训练效果。

完整试跑结果（fan_in，4 遍）：

| 轮次 | train loss | validation loss | validation token accuracy |
|---|---:|---:|---:|
| 1 | 7.3337 | 7.4224 | 7.92% |
| 2 | 6.6060 | 6.9207 | 9.10% |
| 3 | 6.0679 | 6.6772 | 9.93% |
| 4（最佳） | 5.8505 | 6.6021 | 10.41% |

历史最佳 validation loss=7.5097、PPL=1825.7；新版为 6.6021、PPL=736.6。
训练与轮末评估/报告约 642.6 秒，不含初始评估。没有重新使用测试集。
自由生成仍重复：例如“机器学习的主要方法,在深度学习中,数据,数据,…”；
损失改善不等于能流畅续写。训练和验证差距已扩大，下一步优先改善数据而非盲目加轮数。

`industry-33m-fanin-run/best.pt` 保存第 4 遍权重；同目录有 `report.json`、
`observations.json`、`training_observation.html`、`train.log` 和 TensorBoard 日志。

```bash
.venv/bin/python 06-attention/compare_industry_runs.py
```

对照摘要位于 `artifacts/book-lm/industry-comparison/summary.json`，TensorBoard
`industry-comparison/*` 中使用 `*_by_tokens_seen` 标签，横轴为累计训练目标 token 数。
历史 batch 与优化器计划不同的限制仍然适用。

### V100 云端扩展语料训练（自有模型）

`expand_industry.py` 固定 IndustryCorpus2 revision，选 AI、教育、科技、新闻四个中文
high 分片中的首个 row group，按每域约 250 万字符抽样。保留原 8192 词表不重新训练，
继续使用自己的 8 层、512 维、33,604,096 参数模型，不下载第三方模型权重。

v2 训练集 4,656 篇 / 7,291,108 个目标 token（约原数据的 15.7 倍）；混合验证集
200 篇 / 313,654 个目标 token；`validation_legacy.json` 保留原 40 篇用于同口径对照。
原 test 保留且不参与调参。过滤包含长度、中文占比、替换字符、严重长句重复、
忽略空白/标点/大小写后的精确去重，以及对留出集的 100 字符长片段保守排除。
这仍不等于全面语义去重或全库代表性采样，数据清理继续折叠空白。

每次训练使用新 run-dir。`--run-label`、`--run-note` 写入 TensorBoard Text 的
`run/description`；configuration 和 run.json 保存数据版本/哈希、分词器哈希、
初始权重哈希、代码哈希、训练起点、模型配置、GPU、精度与时间。
`run/status` 标记 running/completed/failed，run/result 给出最佳验证结果。
普通异常会记录 failed；进程被系统强杀时状态可能停在 running，需结合进程/日志判断。
历史 Mac 实验已补写明确标记（`label_industry_history.py`），原曲线不变。

云端已验证 Python 3.12 / torch 2.8.0+cu128 / V100-SXM2-32GB。
使用镜像自带 CUDA PyTorch 与 `--system-site-packages` venv，不同步 Mac 的 uv.lock。
依赖见 requirements-cloud.txt。新增 `--device cuda --precision fp16`；训练前向混合精度、
FP32 参数与 Adam 状态、GradScaler、先 unscale 再裁剪梯度，非有限梯度降 scale 重试，
最多 8 次；实际更新后才推进进度。验证和生成仍为 FP32，保证前后评估口径一致。

隔离测速（随机 token，同架构/256 上下文/Adam/梯度裁剪，预热 3 步后测 4 步）：

| 设备 / 精度 | batch | token/s | 峰值 allocated 显存 GiB |
|---|---:|---:|---:|
| M1 Pro / FP32 | 8 | 4,410 | — |
| V100 / FP32 | 8 | 9,931 | 1.61 |
| V100 / FP32 | 32 | 29,413 | 5.25 |
| V100 / FP32 | 64 | 34,842 | 10.09 |
| V100 / FP32 | 96 | 36,926 | 14.93 |
| V100 / FP32 | 128 | 37,518 | 19.79 |
| V100 / FP16 | 32 | 32,148 | 5.27 |
| V100 / FP16 | 64 | 53,469 | 10.06 |
| V100 / FP16 | 96 | 58,464 | 14.84 |
| V100 / FP16 | 128 | 61,228 | 19.63 |

同条件 FP32 batch8 约 2.25 倍；选择 FP16 batch128 后相对 Mac 原配置探针吞吐
约 13.9 倍，不是端到端训练加速倍数。首轮受并行 CPU 单测干扰的测速已弃用。
实际选择以吞吐、稳定性和余量为准，非占满显存。batch 变化同时改变每遍更新次数，
不能把硬件吞吐提升当作同等模型质量提升。

正式命令保存在 cloud_run_v100.sh：从 Mac fan_in 最佳权重继续 1 遍，batch128，
FP16，学习率 1e-4；优化器重新初始化。云端工作目录 `/root/autodl-tmp/ownlm-industry-v2`。

当前 SSH（实例更换后需更新 host/port）：

```bash
ssh -i ~/.ssh/llm_from_zero_autodl_20261114 -p 57901 root@region-41.seetacloud.com
# 查看云端训练日志
ssh -i ~/.ssh/llm_from_zero_autodl_20261114 -p 57901 root@region-41.seetacloud.com \
  'tail -f /root/autodl-tmp/ownlm-industry-v2/train-v2.log'
# 如现有隧道断开，在 Mac 重建（本地 6008 须空闲）
ssh -N -i ~/.ssh/llm_from_zero_autodl_20261114 -p 57901 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:6008:127.0.0.1:6006 root@region-41.seetacloud.com
```

浏览器访问 http://127.0.0.1:6008/ 。云端 TensorBoard 用
`/root/autodl-tmp/ownlm-venv/bin/python -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006`。
面板含历史 Mac 记录、性能探针、真实语料短跑及正式训练的独立标记。

V100 首轮正式结果：FP16 batch128，241 次更新，约 130 秒完成参数更新；
含初始全量评估、训练和末尾评估/报告共 302.3 秒。真实训练约 5.6 万有效 token/s，
观察到 GPU 利用率约 100%、显存占用约 22GB；AMP 溢出重试总计 0。

| 评估集合（均 FP32） | 训练前 loss | 训练后 loss | 训练后 token accuracy |
|---|---:|---:|---:|
| v2 四领域验证（200 篇） | 6.7398 | 6.2262 | 12.64% |
| 原 AI 验证（40 篇） | 6.6021 | 6.3283 | 11.72% |

未使用测试集调参。贪心续写仍重复“人工智能”“数据”等词语，不能把 loss 下降
等同于流畅生成。新旧数据验证均改善，比单看混合验证集数字更有解释力。
下载记录保存在 `artifacts/book-lm/industry-v100-results/04-v100-v2-continue-fanin-fp16-b128/`，
包括 best.pt、run.json、report.json、observations.json、legacy-validation-comparison.json 和日志。
Mac 的 `~/.ssh/config` 已添加 `ownlm-v100` 别名，可用 `ssh ownlm-v100` 连接。
SSH key/账户信息没有打进训练包；云实例训练完成后仍保持开机供查看 TensorBoard。

### 每 50 步验证与完整断点续训

追加实验 `05-v100-v2-continue-3ep-eval50` 的命令见 `cloud_continue_v100.sh`。
保持自有模型、数据和词表，从 04 最佳权重追加 3 遍（723 次更新）。04 仅保存了权重，
因此本轮明确重置 Adam 和学习率计划；05 开始保存完整训练状态。

TensorBoard 选择 `05-v100-v2-continue-3ep-eval50/tensorboard`：

| 标签 | 频率 | 含义 |
|---|---|---|
| `optimization/loss` | 每 10 步 | 当前随机 batch 的训练损失，Dropout 开启，会波动 |
| `train_probe/loss` | 每 50 步及轮末 | 固定 512 个训练窗口，关闭 Dropout，用于与验证趋势对照 |
| `validation/loss` | 每 50 步及轮末 | 完整混合验证集，最佳模型选择依据 |
| `train/loss` | 每遍结束 | 完整训练集评估，仍然是稀疏的点 |
| `time/eta_total_seconds` | 每 10 步 | 估算剩余更新、周期检查和轮末评估时间；先测到相应开销才计入 |
| `time/total_wall_seconds` | 结束时 | 本次运行耗时，包含初始评估、训练、检查、保存及报告 |

Text 中 `generation/prompt_*` 每次检查保留相同提示词的贪心续写。
`validation_checks.json` 保存密集评估点，`observations.json` 和 HTML 报告保留整轮快照。
之前一遍训练的 `train/loss` 只有起点和终点，因此连起来是直线；调低 TensorBoard
平滑只能显示原始记录值，不能补出未记录的中间点。

`best.pt` 用于推理，按完整验证集最低 loss 选取，可能在轮内。
`last.pt` 每次检查原子替换，包含当前权重、Adam、GradScaler、固定学习率计划、
随机数状态、下一批索引、历史指标及最佳权重。训练批次顺序由 seed 和 epoch 确定；
评估前后保存并恢复随机状态，避免检查频率扰动 Dropout 序列。

如果 05 **尚未完成就被中断**，在云端使用新的空目录恢复，保持原来的 3 遍总计划：

```bash
cd /root/autodl-tmp/ownlm-industry-v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/ownlm-venv/bin/python -u train_lm.py train \
  --data-dir data --run-dir runs/05-resumed \
  --resume-checkpoint runs/05-v100-v2-continue-3ep-eval50/last.pt \
  --context-length 256 --device cuda --precision fp16 --batch-size 128 \
  --epochs 3 --learning-rate 0.0001 --eval-every 50 --train-probe-windows 512 \
  --run-label '05 · 中断后完整恢复' \
  --observation-prompts 人工智能 机器学习的主要方法 数据 科学研究的意义
```

恢复校验数据/词表哈希、总更新次数、学习率计划、batch、设备和精度等配置。
已完成计划后要新增训练，使用 `--init-checkpoint best.pt` 开始新计划，会重置优化器。
`--stop-after-step N` 供主动保存中途断点；该运行标为 paused。
同一环境的小模型端到端测试验证：连续训练与中断恢复的参数、Adam、GradScaler、
随机状态和验证结果一致；CPU 和 V100 FP16 均通过。跨硬件或库版本不承诺逐位一致。

05 实验完成：18 个验证检查点，最佳为 step 723（第 3 遍末尾）。

| 指标 | 追加训练前 | 追加训练后 |
|---|---:|---:|
| 全量 train loss | 6.1414 | 5.5345 |
| v2 混合 validation loss | 6.2262 | 5.7104 |
| v2 validation perplexity | 505.84 | 302.00 |
| v2 validation token accuracy | 12.64% | 15.36% |
| 原 AI validation loss（同一 40 篇） | 6.3283 | 5.9051 |
| 原 AI validation token accuracy | 11.72% | 13.86% |

三遍末尾的混合 validation loss 分别为 5.9255、5.7569、5.7104，未观察到反弹，
但每遍收益减小，且后期学习率也在降低。固定提示词贪心生成仍严重重复，例如
“人工智能(AI)人工智能(AI)…”和“机器学习的主要方法,是机器学习,深度学习,算法,算法,…”。
这说明真实前缀下的预测改善，但自由生成尚未过关，不能仅凭这些结果判定瓶颈只有数据量。
本次没有使用测试集。

参数更新约 392.3 秒；初始全量评估约 86.2 秒，周期检查和保存约 188.6 秒，
轮末全量评估和保存约 262.8 秒，总计训练/评估/报告约 931.9 秒（15.5 分钟）。
更新阶段约 5.6 万有效 token/s，AMP 溢出重试为 0。
频繁评估用于观察这次短实验，其开销不能算成 GPU 更新吞吐下降。

本轮实际代码保存在结果目录 `code/`，对应 run.json 中的训练源文件哈希。
运行后改进了下一次运行的 ETA：用初始全量评估估算首遍后的评估开销，
并隐藏 tqdm 自带的另一套剩余时间，进度条统一看 `ETA=`。
本地 120 项测试通过，V100 FP16 完整恢复测试通过。

结果目录：`artifacts/book-lm/industry-v100-results/05-v100-v2-continue-3ep-eval50/`。
包含最佳权重、完整断点、全部评估点、前后续写对照、TensorBoard、日志和代码快照。
原 AI 验证与续写对照可由 `evaluate_industry_continuation.py` 重现。

### 1 亿 token 新语料实验（v3 / run 06）

`prepare_industry_100m.py` 在固定 IndustryCorpus2 revision 上构建新的训练样本，
AI、教育、科技、新闻各 25,000,000 个目标 token，共 62,533 篇文档、421,560 个窗口。
相对 v2 排除旧训练文档的规范化精确重复；保留 v2 的 tokenizer、混合验证、原 AI
验证和 test 原始文件及哈希。此过滤不等于语义近重复去重。

数据构建命令（输出目录必须是新目录）：

```bash
.venv/bin/python 06-attention/prepare_industry_100m.py \
  --previous-data 06-attention/artifacts/book-lm/industry-expanded-v2 \
  --output 06-attention/artifacts/book-lm/industry-expanded-v3-100m
```

保留原质量过滤，并对留出集建立所有 50 字符规范化片段：训练候选每隔 50 字符
取片段检查，因此任意共有的 100 字符连续段都会被排除，也可能保守排除更短的模板句。
每个领域最后一篇只取满足 token 预算的真实前缀，不人为补 EOS。训练目标总数恰好
100,000,000，包含自然文档结束的 EOS，不包含 padding。

`indexed_tokens.py` 将 token 存为 uint32 文件，窗口位置和长度存为 NumPy 索引，
训练时通过 mmap 按需转换当前 batch。保持原来的文档边界、BOS/EOS 和向右错位标签；
不会把整份数据展开成大量常驻 PyTorch 张量。原始入选文本、来源位置和实际使用长度
保存在 `training_documents.jsonl.gz`；`source.json` 记录采样/过滤/哈希，`audit.json`
记录核对结果。257 篇抽查文档重新分词与二进制内容一致；全部留出文件/词表哈希一致。

云端启动命令见 `cloud_100m_v100.sh`。继续使用 8 层、512 维、33,604,096 参数的
自有模型，V100 FP16 / batch128 / context256，训练一遍、3294 次更新。
从 05 的 last.pt 开始（已确认它也对应最佳 step723），保留 Adam 和 GradScaler；
为新增预算设置新的学习率计划，峰值 1e-4、warmup 3%、余弦下降到 1e-5。

`--init-checkpoint ... --keep-optimizer-state` 表示保留权重与优化器，开始新计划；
`--resume-checkpoint ...` 表示恢复同一计划，要求数据与预算等配置一致。
`--no-full-train-eval` 关闭全量训练集评估，避免扩大语料后反复扫描整个训练集。
本次训练指标为固定 1024 个窗口的 `train_probe/loss`，没有全量 `train/loss`；
每 250 步及轮末评估完整混合验证集、生成固定提示词结果并原子保存完整断点。

TensorBoard 选择 `06-v100-v3-100m-1ep-fp16-b128/tensorboard`，关注：

- `validation/loss`、`train_probe/loss`：模型质量趋势。
- `progress/train_tokens_seen`、`progress/token_fraction`：成功更新实际使用的 token 数与预算比例。
- `optimization/tokens_per_second`：更新阶段吞吐；`time/eta_total_seconds`：估计剩余时间。
- Text 的 `run/description`、`configuration`、`generation/prompt_*`：实验条件与续写。

```bash
ssh ownlm-v100 'tail -f /root/autodl-tmp/ownlm-industry-v2/train-v4-100m.log'
```

如果这次实验被中断，用新的空 run-dir 恢复，保留同一计划参数：

```bash
cd /root/autodl-tmp/ownlm-industry-v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/ownlm-venv/bin/python -u train_lm.py train \
  --data-dir data-v3-100m --run-dir runs/06-resumed \
  --resume-checkpoint runs/06-v100-v3-100m-1ep-fp16-b128/last.pt \
  --context-length 256 --device cuda --precision fp16 --batch-size 128 \
  --epochs 1 --learning-rate 0.0001 --warmup-ratio 0.03 \
  --eval-every 250 --train-probe-windows 1024 --no-full-train-eval \
  --run-label '06 · 1亿token实验 · 完整恢复' \
  --observation-prompts 人工智能 机器学习的主要方法 数据 科学研究的意义
```

06 已完成：实际成功更新使用 **100,000,000 个目标 token**，3294 次更新，
最佳验证为最后一步。训练、评估、保存及报告共约 **2030 秒（33.8 分钟）**，
不含语料构建、上传和结果下载。参数更新阶段约 1812 秒，吞吐约 5.52 万有效 token/s；
AMP 溢出重试为 0。Adam 步数从 723 接续至 4017，完整断点保存了优化器、
GradScaler、随机状态及下一个数据位置。

| 指标（固定词表与评估数据） | 训练前 | 训练后 |
|---|---:|---:|
| 混合 validation loss | 5.7104 | 4.8828 |
| 混合 validation perplexity | 302.00 | 132.00 |
| 混合 validation token accuracy | 15.36% | 21.58% |
| 固定训练 probe loss（1024 窗口） | 5.6993 | 4.8194 |
| 原 AI validation loss（40 篇） | 5.9051 | 5.1179 |
| 原 AI validation token accuracy | 13.86% | 18.98% |

15 个原始检查点见结果目录的 `validation_progress.png`，可用
`plot_training_checks.py <run-dir>` 重绘；没有平滑。验证与固定训练 probe 同步下降，
这轮没有明显的验证反弹。曲线后期趋缓，同时学习率也在降低，不能仅据此判定模型
已到容量上限，也不能把小 probe 与验证的差距视为全量训练/验证差距。测试集未用于评估。

自由生成仍未过关。`generation-modes-comparison.json` 对照 05/06 最佳模型，在
同样 6 个提示词上分别进行贪心和固定种子采样（temperature=0.8、top-p=0.9、
seed=42、最多新增 64 token，不加重复惩罚）。贪心续写仍严重循环；采样有较完整的
局部句子，但仍存在语义、事实及重复问题。例如训练后贪心会输出
“人工智能(AI)是AI(AI)的\"AI+AI+AI+…”；采样也会输出“学生是学习的一个班级”。
这组小样例不能代表全面能力，低重复率也不等于流畅或正确。

因此，本轮支持“增加新数据能改善未见文本预测”，尚不足以认定具备可靠问答能力。
可以把当前权重作为后续小规模 SFT 对照实验的起点，但本轮没有启动 SFT 或追加训练。
TensorBoard 的 Text → `run/conclusion` 已记录结果与限制；
`generation_comparison/prompt_*` 展示 05/06 的贪心与采样续写对照。

结果目录：`artifacts/book-lm/industry-v100-results/06-v100-v3-100m-1ep-fp16-b128/`。
模型、完整断点、数据审计、TensorBoard、生成对照和日志均已保留；本机两个权重文件
与云端 SHA-256 一致，训练核心代码快照哈希一致，详见 `local-verification.json`。
本地 124 项测试通过；V100 上 3 项完整恢复测试通过，包含索引数据与优化器接续。
云端训练进程已结束，实例保留开机供查看 TensorBoard。

### 学习率对照与新增 2 亿 token（runs 07–10）

用户授权在离开的一小时多内持续实验。`cloud_loss_experiments.py` 在云端依次执行
三档学习率对照，再接续新语料训练。每档从 06 的同一完整断点开始，保留 Adam 和
GradScaler，使用相同随机种子和 400 个 batch，各处理 12,113,685 个目标 token。
短测用恒定学习率，每 200 步验证，按第 400 步完整验证 loss 选取配置。

| Run | 学习率 | 第 400 步验证 loss |
|---|---:|---:|
| 07 | 5e-5 | 4.848635 |
| 08 | 1e-4 | 4.827792 |
| 09 | 2e-4 | 4.795015 |

三档的起点验证 loss 均为 4.882830。07–09 的 paused 状态表示按计划在 400 步
结束短测，完整训练计划在元数据中保留，便于恢复。三个分支共消耗 36,341,055 个
训练目标 token，主模型接续其中 09 分支。

10 使用 `industry-expanded-v4-200m`，四个领域各 5000 万目标 token，共 125,547
篇文档、843,391 个窗口。构建时额外用 `--exclude-data industry-expanded-v3-100m`
排除 v3 文档精确重复，继续使用 v2 词表与留出集。完整审计核对目标计数、文档哈希、
旧训练集排除、文件哈希和 token ID 范围，另抽查 257 篇重新分词及留出集片段过滤。
上传后云端再次核对数据哈希，写入 READY.json 后进入训练。

10 从 09 的 last.pt 继续，保留优化器状态，峰值学习率 2e-4，3% warmup，
一遍 cosine 下降至 2e-5。每 500 步评估完整验证集和固定 1024 个训练窗口。
TensorBoard 选择 `10-v100-fresh200m-selected-lr/tensorboard`，Text 中
`run/learning_rate_comparison` 记录选择依据。云端 `loss-experiments-state.json`
记录当前运行和对照结果，每轮拥有独立日志与完整断点。

10 已完成 6589 次更新，实际处理 200,000,000 个目标 token。主训练与检查共
3839 秒（64.0 分钟），最佳为最后一步。相对 06，混合验证 loss 从 4.882830
降至 4.160423，PPL 从 132.00 降至 64.10，下一 token 准确率从 21.58% 升至
29.34%。相对本轮实际起点 09，验证 loss 从 4.795015 降至 4.160423。
训练 probe 从 4.758376 降至 4.089226。AMP 重试累计为 0。

| 固定验证领域 | 06 loss | 10 loss |
|---|---:|---:|
| AI（原有与新增验证共 80 篇） | 5.058748 | 4.331885 |
| 教育（40 篇） | 4.747239 | 4.036237 |
| 科技（40 篇） | 4.843018 | 4.086245 |
| 新闻（40 篇） | 4.705481 | 4.011490 |

固定生成样例保存在 `generation-modes-comparison.json`。部分句子变得更完整，
贪心与采样续写仍出现重复和语义问题。验证指标改善与自由生成能力需分别观察。
本机 best.pt、last.pt 的 SHA-256 与云端一致，43 个训练源文件快照哈希核对通过，
完整断点中的 Adam 步数为 11006。`loss_and_learning_rate.png` 展示原始验证检查点
和实际学习率计划，虚线标记后续对照的第 5500 步起点。

用户询问后期曲线变缓是否值得停训，因此保留 5500 步完整断点并完成原计划。
当时学习率已从峰值 2e-4 降至约 3.3e-5。4000–5000 步验证 loss 从 4.229034
降至 4.188843，固定训练 probe 也同步改善。当前记录支持继续获得小幅收益，
学习率变化对下降速度的具体贡献需要对照确认。

相关研究通过实验讨论 cosine 周期长度及最终衰减阶段对模型质量的影响。
其结果用于设计后续对照，具体收益由本项目实测决定。
[Hägele 等，NeurIPS 2024](https://arxiv.org/html/2405.18392v3)
[Wen 等，ICLR 2025](https://arxiv.org/html/2410.05192v3)

`compare_plateau_schedules.py` 已完成两个受控分支。它们从第 5500 步的同一完整断点
恢复权重、Adam、GradScaler 和随机状态，使用相同的后续 500 批数据，各处理
15,216,560 个目标 token。11 复现原 cosine 计划。12 使用 1e-4 学习率训练 400 步，
再用 100 步 cosine 衰减到与 11 相同的结束学习率 2.37448e-5。每 100 步完整验证。

| 方案 | 第 6000 步验证 loss | PPL | token accuracy |
|---|---:|---:|---:|
| 11 原计划复现 | 4.168433 | 64.6141 | 29.280% |
| 12 提高学习率后衰减 | 4.170248 | 64.7315 | 29.206% |

11 与原训练第 6000 步的验证 loss 完全一致，复现误差为 0。两组 AMP 重试均为 0，
各耗时约 291 秒。12 初期 loss 上升至 4.189436，最终恢复至 4.170248，
比原计划高约 0.001815。本次短期调整没有带来优势，差距很小，结论限于这个起点、
随机状态、数据顺序和 500 步预算。继续采用主训练最后一步的 4.160423 权重。

后期斜率本身不足以定位瓶颈。固定训练 probe 和验证同步改善，四领域验证均有收益，
现有观察支持继续学习但收益递减。更长的学习率计划、语料质量和模型容量对收益的
贡献还需要独立对照。此前“可以立即分析”的判断应与“是否提前停训”的决策分别论证。

11/12 的 `diagnostic.pt` 保存权重、优化器和随机状态，后续可通过
`--init-checkpoint ... --keep-optimizer-state` 开始新计划。其短期自定义调度已经完成。
结果同时保存到主实验的 `plateau-schedule-comparison.json` 和各自 TensorBoard 中。
