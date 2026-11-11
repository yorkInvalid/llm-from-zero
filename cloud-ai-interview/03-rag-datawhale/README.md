# 03 · 基于 Datawhale 的资料存储与 RAG

**先打开 [LESSON.md](LESSON.md)，从 A 开始。** 在 VS Code 中可用 Command + Shift + V 预览；不需要 Jupyter，也不用把所有 Python 文件一起读完。

本课顺序：原文 → 入库 → 只检索 → 消息预览 → 真实回答 → 改切分与资料 → HTTP 接口。每一步的中间结果都保存在本课目录。沿用第二课密钥配置，但不修改旧课。

| 文件 | 什么时候看 |
|---|---|
| prepare.py | 下载公开资料时 |
| build.py | 观察入库时 |
| search.py / knowledge.py 的 search | 观察命中片段时 |
| answer.py | 对照消息与回答时 |
| server.py | 最后接回 POST /ask 时 |
| SOURCES.md | 原课链接、固定版本与改编说明 |
| NOTES.md | 自己填写观察，不预填实验结论 |
| VALIDATION.md | 教师验证范围，不作为学习成绩 |

环境：Python 3.12，uv；真实中文 embedding 在本机 CPU 运行，模型回答通过 DeepSeek/TokenHub。默认仅预览，`answer.py --live` 和 HTTP `/ask` 才调用回答 API。首次向量模型下载需要网络；没有假数据回退。

全程预算仍为 1000 元，本课不购买云资源。课堂先用两次回答观察有依据和无依据的问题，HTTP 接入再调用一次；不用循环压测。
