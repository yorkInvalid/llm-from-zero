"""只下载公开学习资料；不调用模型、不读取密钥。"""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REVISION = "77beb748047e8a0d8ff708716606a8e0b132dc45"
FILES = ["2. 提示原则 Guidelines.md", "3. 迭代优化 Iterative.md", "4. 文本概括 Summarizing.md"]


def main():
    folder = ROOT / "data" / "source"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in FILES:
        path = "data_base/knowledge_db/prompt_engineering/" + name
        url = f"https://raw.githubusercontent.com/datawhalechina/llm-universe/{REVISION}/{quote(path)}"
        target = folder / name
        # 用户可能正在做修改资料实验，不覆盖已有文件。
        if target.exists():
            print("保留已有文件：", name)
            continue
        with urlopen(url, timeout=60) as response:
            content = response.read()
        target.write_bytes(content)
        manifest.append({"file": name, "url": url, "revision": REVISION,
                         "sha256": hashlib.sha256(content).hexdigest()})
        print(f"已下载 {name}，{len(content)} 字节")
    record = ROOT / "data" / "downloads.jsonl"
    with record.open("a", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("下一步：打开 data/source 的原文，再运行 build.py。来源与使用说明见 SOURCES.md。")


if __name__ == "__main__":
    main()
