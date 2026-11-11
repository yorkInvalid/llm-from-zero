"""课程 5 的本地分词器。导入、加载均不联网；--download 才下载公开文件。"""

import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen


ASSET_DIR = Path(__file__).parent / "assets"
REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_SIZE = 11_422_654
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"


def load_tokenizer(on_event=None):
    """加载固定本地文件；可选回调报告真实完成步骤，失败报告后仍抛出异常。"""
    def emit(stage, status, **details):
        if on_event is not None:
            on_event({"stage": stage, "status": status, **details})

    stage = "locate"
    path = ASSET_DIR / "tokenizer.json"
    try:
        emit(stage, "running", path=str(path.resolve()))
        if not path.is_file():
            raise FileNotFoundError(
                "分词器文件不存在。请在项目根目录运行："
                "uv run python 05-token-embedding/tokenizer_tools.py --download"
            )
        emit(stage, "done", path=str(path.resolve()), size_bytes=path.stat().st_size)
        stage = "read"
        emit(stage, "running")
        raw = path.read_bytes()
        emit(stage, "done", size_bytes=len(raw))
        stage = "verify"
        emit(stage, "running")
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != TOKENIZER_SIZE or digest != TOKENIZER_SHA256:
            raise ValueError("本地分词器校验失败，请用 --download 重新下载固定版本。")
        emit(stage, "done", sha256=digest, size_bytes=len(raw))
        stage = "parse"
        emit(stage, "running")
        text = raw.decode("utf-8")
        structure = {}
        if on_event is not None:
            document = json.loads(text)
            model = document.get("model", {})
            structure = {
                "model_type": model.get("type"),
                "base_vocab": len(model.get("vocab", {})),
                "merge_rules": len(model.get("merges", [])),
                "added_tokens": len(document.get("added_tokens", [])),
                "pre_tokenizer": (document.get("pre_tokenizer") or {}).get("type"),
                "decoder": (document.get("decoder") or {}).get("type"),
                "keys": list(document),
            }
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError("当前内核缺少 tokenizers。运行 uv sync，选择 llm-from-zero 环境并重启内核。") from None
        # 使用刚刚完成校验的字节，避免验证后再次读取另一份内容。
        tokenizer = Tokenizer.from_str(text)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        emit(stage, "done", **structure)
        stage = "probe"
        emit(stage, "running")
        sample = "你好，世界！"
        ids = tokenizer.encode(sample, add_special_tokens=False).ids
        restored = tokenizer.decode(ids, skip_special_tokens=False)
        if restored != sample:
            raise ValueError("试编码的完整还原与原文不同，停止加载。")
        emit(stage, "done", vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
             sample=sample, ids=ids, restored=restored)
        return tokenizer
    except Exception as exc:
        emit(stage, "error", error=f"{type(exc).__name__}: {exc}")
        raise


def download_assets():
    """下载约 11.4 MB 分词器和许可证；不需要账号，不下载模型权重。"""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local, expected in [
        ("tokenizer.json", "tokenizer.json", TOKENIZER_SHA256),
        ("LICENSE", "LICENSE-Qwen.txt", None),
    ]:
        target = ASSET_DIR / local
        if target.exists() and expected and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        with urlopen(f"{BASE_URL}/{remote}", timeout=60) as response:
            data = response.read()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("下载的分词器校验失败，没有覆盖原文件。")
        temporary = target.with_suffix(target.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(target)
    provenance = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "file": "tokenizer.json",
        "size_bytes": TOKENIZER_SIZE,
        "sha256": TOKENIZER_SHA256,
        "source_url": f"{BASE_URL}/tokenizer.json",
        "license": "Apache-2.0",
        "verified_date": "2026-10-31",
        "purpose": "Only tokenizer data; no model weights or remote Python code.",
    }
    (ASSET_DIR / "source.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("公开分词器和许可证已准备好；后续课堂可离线运行。")


if __name__ == "__main__":
    if sys.argv[1:] == ["--download"]:
        download_assets()
    else:
        print("本文件用于本地加载。重建资料：python tokenizer_tools.py --download")
