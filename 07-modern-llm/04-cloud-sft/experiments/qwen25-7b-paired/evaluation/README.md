# 同题双问：本地复核评分

GPU 已经对 60 道新题各生成两种回答。这里用普通 CPU 和 Python 3.11+ 标准库，把云端原始输出和当时冻结的题目、裁定对上，不需要模型权重或重新训练。

在本目录运行，参数为下载的原始云端测量包：

```bash
python3 replay.py /path/to/pair-measurements.tar.gz
```

默认在临时目录工作。如要保留逐题文本、评分与开发集选轮，增加 `--out ./replay-output`（必须是尚不存在的目录）；结果写在其 `evaluation/paired-scores.jsonl`。这条命令先核验测量包、评分代码和题目指纹，仅读取包内 17 份生成记录，再检查逐题结果是否和归档的 SHA-256 一致。

**结果。** 纯底座 + 两组各 8 个检查点，共 17 × 120 = 2,040 个回答。去除同一问法下重复的文字后有 1,152 份，130 份适用自动判分，其余 1,022 份沿用当时的匿名裁定（评阅者为 Codex）。最终 1,020 条同题配对评分的 SHA-256 为 `d2d74bd643cfef2390dbe743ed7958057bc930aa97b06de39b6002f7d7f72cca`。第 2 轮独立确认集上，两版都通过的题数是对照组 8/30、双问组 7/30。

| 文件 | 用途 |
|---|---|
| [core.tsv](core.tsv)、[freeze.py](freeze.py) → [items.jsonl](items.jsonl)、[prompts.jsonl](prompts.jsonl) | 原题、参考答案与评分依据；确定开发/确认分组；生成实际发送的 120 条提示。P28 是木块题。 |
| [SCORING.md](SCORING.md)、[freeze.json](freeze.json)、[grading-code-freeze.json](grading-code-freeze.json) | 评分口径、题目/规范哈希与评分脚本哈希。 |
| [grade.py](grade.py)、[review/decisions](review/decisions) | 原始评分脚本。先 `prepare` 产生不带模型名的去重输出，再 `score` 合并自动结果和历史盲评。换一批生成文本时，新增的待判文本需重新盲评。 |
| [select_dev.py](select_dev.py)、[selection.json](selection.json) | 仅用 30 道开发题选检查点，平手选更早；归档的选择为对照组 1.75 轮、双问组 0.5 轮。 |
| [replay.py](replay.py) | 此次整理新增的复核入口；原始评分脚本与历史裁定保持逐字节不变。 |
