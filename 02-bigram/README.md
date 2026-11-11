# 阶段 2 · 语言模型入门

这一阶段做最简单的语言模型：给前面的字母，猜下一个字母。先用数数做一个 bigram 模型，再把它换成神经网络，用阶段 1 学的梯度下降来训练。

## 文件

- [lesson1.ipynb](lesson1.ipynb) — 第 1 课：数一数 bigram，计数表 → 概率 → 编名字 → 用 loss 打分
- [names.txt](names.txt) — 32,033 个英文名字，来自 Karpathy 的 [makemore](https://github.com/karpathy/makemore)（MIT 许可）
- [lesson2.ipynb](lesson2.ipynb) — 第 2 课：把计数表换成一层神经网络，梯度下降学出同一张表，最后调温度（API 的 temperature）
- 两课已完成（2026-10-30 用户确认）。阶段 2 作业留作可选复习，尚未出题。
- 下一课：[阶段 3 · 从一句需求到工单草稿](../03-llm-api/lesson.ipynb)，进入 Agent 与业务流程自动化主线；最新路线以根目录 README 为准。

## 提示怎么用

- **API 和写法**：直接写在 TODO 上方，附最小示例。
- **概念**：灰色的「▶ 提示 1 / 提示 2 / 答案」点开才显示，卡住了一级一级点开。
