# 08 · 谁查了库存？Tools、Function Calling 和 MCP

**从 [逐格实验 lesson.ipynb](lesson.ipynb) 开始：先只跑第一个代码格，看库存数量从哪来。**

本机准备：在项目根目录执行 `uv sync --extra tools`，在 Jupyter 或 VS Code 中选择项目 `.venv` 的 Python 内核。Notebook 使用 [sync_inventory.py](sync_inventory.py) 的同步教学客户端，通过 MCP 与库存服务通信；原版异步示例仍保留在 demo.py。同步客户端仅覆盖本课本地服务的发现与调用，不是通用 MCP 客户端。

模型已缓存时可直接运行；未缓存时，在本目录用 `python download.py` 从魔搭下载，再把 Notebook 的 `path = snapshot_download(...)` 换成 `path = str(lesson / "weights")`。

上节课读过 `model.generate`；这节仍然用它，但第一次生成可能是一张“请帮我查”的单子，宿主程序读完以后才决定要不要执行。采用指令模型 **Qwen3-0.6B**，只推理，不需要 API key。库存是本课设定的两条演示数据。

| 角色 | 代码 | 在这一题做什么 |
|---|---|---|
| Tool（工具） | [inventory.py](inventory.py) 的 `get_stock` | 普通 Python 函数，读表，区分零库存和无此记录。 |
| Function calling（模型请求调用） | [demo.py](demo.py) 的两次 `generate` | 第一次只输出工具名和实参；读到工具结果后，第二次生成用户看到的回答。 |
| MCP（连接协议） | 同文件的 `list_tools` / `call_tool` | 宿主启动独立的库存服务进程，通过标准输入输出发现工具并执行。模型本身不用懂 MCP。 |

**云端运行：** 在魔搭 Notebook 的终端切到本目录，执行 `bash run.sh`。使用镜像自带的 PyTorch，在子目录创建 Python 环境，从魔搭下载并校验约 1.5 GB 模型权重。不开公网端口；进程退出后显存释放，Notebook 实例仍须在工作台停止。脚本会打印实际设备：`cuda` 是云 GPU，`cpu` 则说明当前在用 CPU。重复实验可以直接运行 `.venv/bin/python demo.py '商品 A17 还有货吗？'`；不重复安装或下载。

**看一次真实输出。** 以下来自本地 Mac MPS；同一程序在不同硬件上措辞可能不同。两次真正送入模型的输入分别为 219、262 token，生成分别为 22、11 token。关闭思考模式，每次最多生成 96 token。

```text
用户：请查一下商品 P28 现在有多少件库存？
模型 → 宿主：<tool_call>{"name":"get_stock","arguments":{"sku":"P28"}}</tool_call>
宿主 → MCP 库存服务：call_tool("get_stock", {"sku": "P28"})
工具 → 宿主：P28：37 件
宿主把回执追加为 role="tool"，再次调用模型。
模型 → 用户：P28：37 件库存。
```

**动手顺序：**

1. `bash run.sh --probe`：绕过模型，观察 MCP 能否列出工具并读到 37 件。这一步不下载权重。
2. `bash run.sh`：看“模型首次输出”和“工具实际回执”。**37 最先出现在哪一行？**
3. 把 [inventory.py](inventory.py) 中的 37 改成 12，重新运行 `.venv/bin/python demo.py`。预测哪一段输出会变，是否需要改模型权重。
4. 分别问“商品 A17 有多少？”“商品 Z99 有多少？”“你好！”。前两题应区分 0 和不存在；最后一题通常无需查表。小模型可能选错，这时看首次输出，别把模型没调用当作库存为零。

[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B) · [MCP Python SDK（本课固定 v1）](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)
