# 参考来源与改编范围

核对日期：2026-11-03。Datawhale 仓库固定提交：`77beb748047e8a0d8ff708716606a8e0b132dc45`。

- [Datawhale 第三章：搭建向量知识库](https://github.com/datawhalechina/llm-universe/blob/77beb748047e8a0d8ff708716606a8e0b132dc45/docs/C3/C3.md)：参考资料处理、切分、embedding、Chroma 的学习顺序。
- [Datawhale 第四章：构建 RAG 应用](https://github.com/datawhalechina/llm-universe/blob/77beb748047e8a0d8ff708716606a8e0b132dc45/docs/C4/C4.md)：参考检索结果接入生成的流程。
- [本课使用的知识库资料目录](https://github.com/datawhalechina/llm-universe/tree/77beb748047e8a0d8ff708716606a8e0b132dc45/data_base/knowledge_db/prompt_engineering)：提示原则、迭代优化、文本概括三份中文资料。原文中的翻译和课程来源说明继续保留。

本课不是 Datawhale 官方课程或其授权版本；讲义与程序为针对本学习项目新写。没有复制上游数据库或使用其预生成向量。`prepare.py` 从固定版本下载原文供本机学习，下载地址与哈希在 `data/downloads.jsonl`。没有在本课源文件中重新分发全文；下载资料、向量库和缓存均忽略提交。

所查固定提交未见仓库顶层 LICENSE，不推定为 MIT 或任意商用许可。源材料的权利归原作者，公开发布或商用前应核对各资料的授权；本课保留来源，不把材料改称自有创作。

## 哪些地方与原课不同

| 原课做法 | 本课选择与理由 |
|---|---|
| 多家远程 embedding API 示例 | 本机真实 BGE 模型，先免去新增账号；后续可替换 API，但需重建索引 |
| LangChain 检索问答组合 | 保留 LangChain 切分器；检索、消息、回答分开显示，便于追踪 |
| Chroma 保存和查询 | 仍用真实持久化 Chroma，明确配置余弦距离 |
| Notebook / Streamlit 演示 | 普通 Python 命令 + 可查看 JSON + 最后接 FastAPI |
| 通用知识库问答 | 固定一个中文问题追完整链路，再换条件验证 |

## 实现核对来源

- [LangChain 递归字符切分器](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)：字符大小、目标重叠、中文分隔符。
- [Chroma 集合配置](https://docs.trychroma.com/docs/collections/configure)：余弦距离、近似向量索引。
- [FastEmbed 支持的模型](https://qdrant.github.io/fastembed/examples/Supported_Models/)：本地推理和中文模型条目。
- [BGE 中文模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)：查询前缀、512 维、输入长度限制。它是成熟的小模型选择，不宣称是当前最佳检索模型。

依赖实际版本固定在 uv.lock。模型由 FastEmbed 对应的 Qdrant ONNX 仓库下载；模型名称固定但下载缓存 revision 不由本课程锁文件固定，严格复现还需固定权重 revision/校验值。本课不声称跨版本逐位相同。

- [OWASP 提示词注入防范说明](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)：用于核对原资料“分隔符防止注入”的过强表述。结构化区分数据与指令只是多层防护中的一部分，不能当作确定安全边界。
