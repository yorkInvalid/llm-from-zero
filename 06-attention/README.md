# 多层 Encoder 学习实验

当前实现多层 Post-LN Encoder，每层为：多头注意力 → Add & Norm → FFN → Add & Norm。暂未加入 dropout、padding mask 和 batch 输入。

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

`history` 包含初始状态和每次更新后的状态，共 301 条；每条是已 detach 的 `(layer, head, key, query)` 张量，默认 `(3, 2, 3, 3)`。全部记录堆叠后为 `(301, 3, 2, 3, 3)`。即使只配置一层，记录也保留 layer 维度。`plot_training()` 仍接受不含 layer 维度的单层记录，可使用 `[step[0] for step in history]` 取出第一层。

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

- `TokenEncoder` 包含随机初始化的 `nn.Embedding`、位置编码和多层 Encoder。输入是非空的一维整数 token ID；输出仍为 `(d_model, n)`，同时返回逐层注意力。
- 每次 forward 都重新查表：`X = self.embedding(token_ids).T`。Embedding 在构造时初始化一次，不能每步重建或将查表结果 detach 后缓存。
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

`tests/test_token_embedding.py` 验证查表数值与重复 ID、三种位置编码的注入位置、整数 ID 和范围、参数注册及梯度、backward 与 step 的区别、未使用行在首次无 weight decay 的 Adam 更新中不变、零步不更新，以及三种模式能拟合固定标签。

这个任务只用于展示 Embedding 与后续网络一起训练。标签可由 token 身份直接确定，且只使用一个固定序列，没有独立验证集；准确率 100% 不代表理解语言、使用了上下文或具有泛化能力。后续再接入真实 Tokenizer、batch/padding mask 和依赖上下文的任务。
