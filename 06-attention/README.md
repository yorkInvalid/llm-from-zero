# 多头注意力学习实验

当前实现多头注意力、位置编码、注意力后的残差连接与 LayerNorm、训练与可视化，尚未包含完整 Transformer Block（缺少 FFN 及其后的残差与归一化）。
所有输入均按列排列 token：输入和输出为 `(10, 3)`，注意力权重为 `(2, 3, 3)`，后两维分别是 key、query。

## 阅读顺序

1. `attention.py`：`MultiHeadAttention.forward()` 中保留逐头计算 Q、K、V、A、A′、B 的流程。`ParameterList` 注册各头参数，优化器通过 `model.parameters()` 获取全部参数。
2. `position_encoding.py`：固定 sin/cos 编码与 RoPE 旋转。每头 5 维时，RoPE 旋转前 4 维，最后 1 维保持不变。
3. `attention_sublayer.py`：封装 `Z = LayerNorm(X + Attention(X))`，采用 Post-LN；对每列 token 的特征归一化。
4. `train.py`：生成固定输入、设置平均值目标、选择编码并执行训练。sin/cos 在进入模型前加一次，RoPE 在注意力内部旋转 Q、K。训练调用 AttentionSublayer，loss 使用归一化后的 Z。
5. `visualization.py`：`plot_training(loss_history, attention_history, output_dir, name)` 生成曲线和热力图，并返回图片路径。

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

## 残差连接与测试

`MultiHeadAttention` 仍只返回注意力分支 B；`train()` 返回的 model 现在是 `AttentionSublayer`，包含注意力、残差和 LayerNorm。需要最终输出时直接调用：

```python
Z, attention_weights = model(X)
```

内部计算为：

```python
B, weights = self.attention(X)
Y = X + B
Z = self.norm(Y.T).T
```

X、Y、Z 均为 `(10, 3)`。转置是为了让 `nn.LayerNorm(10)` 沿每个 token 的 10 个特征归一化，而不是沿 token 维度。LayerNorm 在构造函数中创建，10 个 gamma 和 10 个 beta 在 token 间共享，默认总参数量从 400 增至 420。仿射变换后的输出不保证均值为 0、方差为 1。

sin/cos 模式下加回的是包含位置编码的 X，目标仍是原始输入 I 的平均值。`forward_with_residual(attention, X)` 已移至 `attention_sublayer.py`（train.py 保留导入兼容），仅用于原始注意力模块，不应把完整子层传给它重复加残差。

在项目根目录运行（使用标准库 unittest，无需安装测试框架）：

```bash
.venv/bin/python -m unittest discover -s 06-attention/tests -v
```

`tests/test_residual.py` 覆盖：

- 手算结果：固定均匀注意力、恒等 V 投影和两倍 W_o，检查残差加在输出投影之后，且不修改输入或注意力权重。
- 恒等路径：将 W_o 置零，检查 Y = X，以及 Y.sum() 对输入的梯度为全 1。
- 梯度传播：检查输入梯度等于恒等路径与注意力路径梯度之和，所有注意力参数获得有限、非零梯度。
- 训练集成：对三种编码独立计算一次 Adam 更新，对照实际 train() 的 loss、参数及注意力记录，验证 loss 使用残差后的 LayerNorm 输出，并通过手算均值、方差独立对照。
- 收敛回归：三种编码训练 300 步，检查最终 MSE 比初始值降低至少 99%、301 条记录、注意力形状与归一化。不要求每一步 loss 单调下降，也不证明泛化。

`tests/test_layernorm.py` 另外检查逐 token 归一化公式、非默认 gamma/beta、不同序列长度、常量 token 的数值稳定性，以及注意力和 LayerNorm 参数都能获得梯度并更新。

加入 LayerNorm 后，固定输入训练 300 步的三种模式 MSE 约为 0.0014。归一化改变了优化过程，旧的残差版 1e-3 阈值不再沿用；训练回归检查相对下降，计算正确性由公式和梯度测试保证。

已有 artifacts 图片可能来自加入 LayerNorm 前的实验；重新运行上述训练命令即可更新对应模式的图片。
