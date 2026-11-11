# 多头注意力学习实验

当前实现 Post-LN Encoder Block 的主体结构：多头注意力 → Add & Norm → FFN → Add & Norm，并支持位置编码、训练与可视化。暂未加入 dropout、padding mask、batch 输入和多层堆叠。
所有输入均按列排列 token：输入和输出为 `(10, 3)`，注意力权重为 `(2, 3, 3)`，后两维分别是 key、query。

## 阅读顺序

1. `attention.py`：`MultiHeadAttention.forward()` 中保留逐头计算 Q、K、V、A、A′、B 的流程。`ParameterList` 注册各头参数，优化器通过 `model.parameters()` 获取全部参数。
2. `position_encoding.py`：固定 sin/cos 编码与 RoPE 旋转。每头 5 维时，RoPE 旋转前 4 维，最后 1 维保持不变。
3. `attention_sublayer.py`：封装 `Z = LayerNorm(X + Attention(X))`，采用 Post-LN；对每列 token 的特征归一化。
4. `feed_forward.py`：两层 FC 和 ReLU，默认 `10 → 40 → 10`，各 token 独立计算并共享参数。
5. `encoder_layer.py`：组合 AttentionSublayer、FFN、第二次残差和独立的 LayerNorm。
6. `train.py`：生成固定输入、设置平均值目标、选择编码并执行训练。sin/cos 在进入模型前加一次，RoPE 在注意力内部旋转 Q、K。训练调用 EncoderLayer，loss 使用第二次归一化后的 output。
7. `visualization.py`：`plot_training(loss_history, attention_history, output_dir, name)` 生成曲线和热力图，并返回图片路径。

导入这些文件不会自动训练或画图。训练入口已从 `attention.py` 移到 `train.py`。

## 运行

在项目根目录执行：

```bash
.venv/bin/python 06-attention/train.py --position-encoding sinusoidal
.venv/bin/python 06-attention/train.py --position-encoding rope
.venv/bin/python 06-attention/train.py --position-encoding none
```

默认使用 `sinusoidal`。图片仍保存在本目录的 `artifacts/`，文件名带编码模式，便于对照。

训练与绘图分开调用：

```python
from train import train
from visualization import plot_training

model, losses, weights = train("rope")
plot_training(losses, weights, "artifacts", name="rope")
```

上面的导入示例在本目录下使用。`train()` 默认更新 300 次，记录初始状态和每次更新后的状态，共 301 条。

求平均目标不依赖位置；本实验仅在一组固定输入上训练，不用于比较位置编码优劣或证明对新输入的泛化能力。

## 模型结构与测试

`MultiHeadAttention` 仍只返回注意力分支 B；`train()` 返回的 model 现在是 `EncoderLayer`，包含注意力、FFN、两次残差和两个独立的 LayerNorm。需要最终输出时直接调用：

```python
output, attention_weights = model(X)
```

注意力子层内部计算为：

```python
B, weights = self.attention(X)
Y = X + B
Z = self.norm(Y.T).T
```

X、Y、Z 均为 `(10, 3)`。转置是为了让 `nn.LayerNorm(10)` 沿每个 token 的 10 个特征归一化，而不是沿 token 维度。LayerNorm 在构造函数中创建，10 个 gamma 和 10 个 beta 在 token 间共享，注意力子层参数量从 400 增至 420。仿射变换后的输出不保证均值为 0、方差为 1。

EncoderLayer 再将 Z 交给 FFN，然后执行第二次 Add & Norm：

```python
Z, weights = self.attention_sublayer(X)
F = self.ffn(Z)
Y = Z + F
output = self.norm2(Y.T).T
```

FFN 的形状变化是 `(10, 3) → 转置 (3, 10) → FC (3, 40) → ReLU → FC (3, 10) → 转置 (10, 3)`。两层 FC 都有 bias，共新增 850 个参数，第二个 LayerNorm 再增加 20 个参数，模型总计 1290 个参数；这些参数通过子模块注册，由 `model.parameters()` 一起交给 Adam。

第二次残差加回的是 FFN 的输入 Z，不是原始 X。两个 LayerNorm 各自拥有独立的 gamma/beta，最终输出形状仍为 `(10, 3)`。

sin/cos 模式下加回的是包含位置编码的 X，目标仍是原始输入 I 的平均值。`forward_with_residual(attention, X)` 已移至 `attention_sublayer.py`（train.py 保留导入兼容），仅用于原始注意力模块，不应把完整子层传给它重复加残差。

在项目根目录运行（使用标准库 unittest，无需安装测试框架）：

```bash
.venv/bin/python -m unittest discover -s 06-attention/tests -v
```

`tests/test_residual.py` 覆盖：

- 手算结果：固定均匀注意力、恒等 V 投影和两倍 W_o，检查残差加在输出投影之后，且不修改输入或注意力权重。
- 恒等路径：将 W_o 置零，检查 Y = X，以及 Y.sum() 对输入的梯度为全 1。
- 梯度传播：检查输入梯度等于恒等路径与注意力路径梯度之和，所有注意力参数获得有限、非零梯度。
- 训练集成：对三种编码独立计算一次 Adam 更新，对照实际 train() 的 loss、参数及注意力记录，验证 loss 使用完整 Encoder 输出，并通过手算归一化、矩阵乘法和 ReLU 独立对照。
- 收敛回归：三种编码训练 300 步，检查最终 MSE 比初始值降低至少 99%、301 条记录、注意力形状与归一化。不要求每一步 loss 单调下降，也不证明泛化。

`tests/test_layernorm.py` 另外检查逐 token 归一化公式、非默认 gamma/beta、不同序列长度、常量 token 的数值稳定性，以及注意力和 LayerNorm 参数都能获得梯度并更新。

`tests/test_feed_forward.py` 检查固定权重与偏置下的手算结果、ReLU、不同输入形状、token 独立性与参数共享、Encoder 的组合顺序与注意力权重，以及注册的参数能获得梯度并更新。

`tests/test_encoder_layer.py` 检查第二个 LayerNorm 参数独立、FFN 归零时第二次残差保留 Z、该路径的梯度，以及不同形状和常量输入的数值稳定性。

artifacts 中带 none/sinusoidal/rope 后缀的六张图片已按当前 Encoder 重新生成；不带后缀的两张图片是早期实验记录。
