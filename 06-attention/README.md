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
