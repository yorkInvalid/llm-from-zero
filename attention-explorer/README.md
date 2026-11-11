# 跟着 x₃，理解 Attention

与 `06-attention` 平级的本地学习网页。六个教学步骤在同一张可拖动、缩放的画布上逐步展开，使用暖白墨蓝配色和 Noto Sans SC。

## 启动

在项目根目录运行：

```bash
npm --prefix attention-explorer run dev
```

打开 http://127.0.0.1:4173 。网页必须通过 HTTP 服务打开，才能读取 `data.json`。

## 使用

- 用“下一步”跟随 x₃，从生成 query/key 到得到第一个头的 b₃。
- 点击节点展开数值。点击投影结果或 b₃ 的分量，可追踪对应的乘加过程。
- softmax 节点展示 √5 的来源、缩放结果、指数分子和归一化分母。
- 拖动空白处平移，滚轮或加减按钮缩放。“查看全图”显示全部已展开节点，“回到当前步骤”恢复可读视图。
- 键盘 Tab 可选择按钮，Enter 展开节点。聚焦画布后使用方向键平移。
- URL 保留步骤、展开节点和选中分量，可以复制或刷新恢复。

## 数据与验证

当前示例为种子 42 生成的三个 10 维输入，采用课程模型第一个头的初始化参数。Q、K、V 每个向量都是 5 维。网页沿用列向量约定，输出为该头的 b₃。

`dist/math.js` 使用完整精度计算，界面显示四位小数。基准来自 PyTorch float32，计算对照测试允许 2e-6 的绝对误差。

```bash
# 在项目根目录重新导出模型数据
.venv/bin/python attention-explorer/scripts/export_data.py

# 检查计算结果
npm --prefix attention-explorer test
```

## 文件

- `dist/index.html`、`styles.css`、`app.js` 构成网页。
- `dist/math.js` 负责前向计算，`data.json` 保存参数和 PyTorch 参考值。
- `scripts/export_data.py` 导出课程模型数据。
- `tests/math.test.mjs` 对照投影、分数、权重与输出。

字体通过 Google Fonts 加载，网络不可用时会回退本机字体。Noto Sans SC、STIX Two Text 和 IBM Plex Mono 均为免费开源字体。

当前教学采用因果注意力，位置 3 可关注三个已输入位置。后续 Transformer 计算和词表预测作为可展开的流程说明展示。

## 配合讲稿录屏

点击顶部“按稿演示”，使用“下一镜”按顺序推进 31 个镜头。每镜自动展开对应节点、选中分量并定位计算区域。镜头下拉框可直接定位补录位置。“查看本镜配音”展开逐字稿。

完整时间、鼠标指向和逐字配音见 [录屏讲解稿.md](录屏讲解稿.md)。目标时长 12 分 56 秒，每镜包含 3 秒静默操作空档。时间是试读目标，网页按手动点击推进。

修改分镜时编辑 `scripts/make_recording.py`，在项目根目录执行 `.venv/bin/python attention-explorer/scripts/make_recording.py`，同时更新网页使用的 `dist/recording.json` 和录屏稿。
