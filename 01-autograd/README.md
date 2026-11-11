# 阶段 1 · 神经网络基础：自动求导

这一阶段搞懂模型是怎么「学习」的：用反向传播算出每个参数的梯度，再用梯度下降一点点调参数。你会从零写出一个自动求导引擎（仿照 Karpathy 的 micrograd），用它训练一个小神经网络，并和 PyTorch 对答案。

## 文件

- [lesson1.ipynb](lesson1.ipynb) — 第 1 课：导数、链式法则、反向传播，写出 `Value` 引擎
- [lesson2.ipynb](lesson2.ipynb) — 第 2 课：神经元、MLP、训练循环，让 loss 降下来
- [homework.ipynb](homework.ipynb) — 作业：1 道简单、2 道中等、1 道困难思考

按顺序做：第 1 课会生成 `micrograd.py`，第 2 课会生成 `nn.py`，作业要用到这两个文件。

## 提示怎么用

- **API 和写法**：直接写在 TODO 上方，附最小示例。
- **概念**：灰色的「▶ 提示 1 / 提示 2 / 答案」点开才显示，卡住了一级一级点开。
