# 阶段 0 · 环境与工具

## 文件

- [lesson.ipynb](lesson.ipynb) — 课堂笔记本：讲义加实验，`TODO` 处由你填
- [homework.ipynb](homework.ipynb) — 作业：1 道简单、2 道中等、1 道困难思考

## 在 VS Code 里打开

1. 确认装了 VS Code 的 **Jupyter** 扩展（微软出的，装 Python 扩展时一般会一起装上）。
2. 打开 `lesson.ipynb`，点右上角「选择内核 / Select Kernel」→「Python 环境」→ 选 `.venv`（Python 3.12）。
3. `Shift+Enter` 运行当前格并跳到下一格。

想用浏览器打开也行：先 `uv add jupyterlab`，再 `uv run jupyter lab`。

## 环境是怎么搭的

我已经跑过了；你想自己从头来一遍也可以（先删掉 `pyproject.toml`、`.python-version`、`uv.lock`、`.venv/`）：

```bash
uv init --bare --python 3.12               # 生成 pyproject.toml：项目的依赖清单
uv python pin 3.12                         # 生成 .python-version：固定 Python 版本
uv add torch numpy matplotlib ipykernel    # 装库，记进 pyproject.toml 和 uv.lock
```

- `.venv/`：装好的 Python 和库都在这里。删了也没事，`uv sync` 一条命令就能恢复。
- `uv.lock`：记下每个库的精确版本，换台电脑也能装出一模一样的环境。
- `ipykernel`：让 Jupyter 能用 `.venv` 里的 Python 当内核。
