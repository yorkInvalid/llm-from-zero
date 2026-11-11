# 04 · RAG 检查与有依据的改进

**打开 [LESSON.md](LESSON.md)，先从 A 看八个问题。** 第三课已经跑通真实回答；这课判断失败在哪一层，以及改动是否有收益。

继承 Datawhale 第五章的评估思路。使用第三课的真实中文 BGE/Chroma、三份公开资料和锁定 uv 环境。默认只评估检索；生成回答必须显式加 `--live`，不自动批量调用。

```bash
./run.sh prepare --name study
./run.sh evaluate --index study --k 3 --name k3
open artifacts/runs/k3.html
```

终端仅打印摘要。报告先显示结论表，再折叠标注依据、真实片段和消息。没有额外 homework。

| 文件 | 用途 |
|---|---|
| cases.json | 八个问题、人工标注依据和核对要点 |
| core.py | 建库、检索与依据覆盖检查；讲义只要求读两个短函数 |
| lab.py | 命令入口；单题预览与真实回答 |
| report.py | 静态 HTML 呈现，不是学习前置知识 |
| run.sh | 复用第三课 uv 环境，缩短运行命令 |
| NOTES.md | 你自己的结论，未预填 |
| SOURCES.md / VALIDATION.md | 来源、验证范围与限制 |

教师会验证真实本机检索和人工响应下的程序行为；你的真实 API 调用、判断和面试解释另行记录。
