"""在项目 .venv 内核中执行课件并保存真实输出。
仓库根目录：uv run --extra inference --with nbclient --with nbformat python 07-modern-llm/02-real-inference/execute_lesson.py
nbclient/nbformat 只供教师执行，不增加学生推理依赖。
"""
import json
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
path = HERE / 'lesson.ipynb'
nb = nbformat.read(path, as_version=4)
with tempfile.TemporaryDirectory() as directory:
    kernel = Path(directory) / 'lesson-validation'
    kernel.mkdir()
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [str(ROOT/'.venv/bin/python'), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'llm-from-zero', 'language': 'python',
    }))
    manager = KernelManager(kernel_name='lesson-validation', kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
    client = NotebookClient(nb, km=manager, timeout=600, resources={'metadata': {'path': str(HERE)}})
    try:
        client.execute()
    finally:
        # 外部提供的 manager 由这里负责关闭。
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        nbformat.write(nb, path)
print('通过：所有单元格在项目 .venv 内核执行完成。')
