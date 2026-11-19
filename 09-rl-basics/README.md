# 09 · 奖励怎样改变回答概率

打开 [lesson.ipynb](lesson.ipynb)，选择项目 .venv 的 Python 内核，从第 1 格逐步运行。本课只使用现有 PyTorch 和 Matplotlib，在 CPU 上运行，不使用云 GPU、模型下载或 API。

三个参数构成一个只有一道固定题的教学策略。依次观察 softmax 概率、随机采样、0/1 奖励、优势、策略梯度 loss、backward 和 SGD 更新；再与 SFT 梯度对比，画出实际奖励与答案概率曲线，最后用常量奖励检查没有区分信号时的梯度。Notebook 已保存真实执行结果。

固定种子下，一次更新让 P(4) 从 0.4 变为约 0.4337；另一次从初值开始的 60 步训练达到约 0.9536。它不处理新题，不代表获得了通用算术能力，也不是完整的大模型 RL、PPO 或 GRPO 实现。

前置：已经理解 SFT 的 loss、backward 和 step。数学依据见 [策略梯度与基准](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。

下一课：[逐 token 的 RL](../10-rl-tokens/lesson.ipynb)，将一个答案扩展成包含结束符的完整序列。
