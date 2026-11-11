"""资料→片段→向量→检索。按讲义只读当前用到的一个函数。"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingInputTooLong(ValueError):
    pass


def documents(folder):
    paths = sorted(Path(folder).glob("*.md"))
    if not paths:
        raise ValueError("没有 Markdown 资料，请先运行 prepare.py。")
    return [{"source": p.name, "text": p.read_text(encoding="utf-8")} for p in paths]


def fingerprint(docs):
    return hashlib.sha256(json.dumps(docs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def split_documents(docs, size=300, overlap=40):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if not 50 <= size <= 400 or not 0 <= overlap < size:
        raise ValueError("本课 size 为 50–400 字符，overlap 需 >=0 且小于 size。")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = []
    for doc in docs:
        parts = splitter.create_documents([doc["text"]], [{"source": doc["source"]}])
        for part in parts:
            start = part.metadata["start_index"]
            chunks.append({"id": f"chunk-{len(chunks)+1:04d}", "text": part.page_content,
                           "source": doc["source"], "start": start,
                           "end": start + len(part.page_content)})
    return chunks


@lru_cache(maxsize=1)
def encoder():
    from fastembed import TextEmbedding
    print("加载真实中文向量模型（首次需要下载；不调用回答 API）……", flush=True)
    return TextEmbedding(model_name=MODEL, cache_dir=str(ROOT / ".model-cache"), threads=2)


def embed(texts, is_query=False):
    inputs = [QUERY_PREFIX + text if is_query else text for text in texts]
    model = encoder()
    # 固定依赖版本下复制 tokenizer 检查长度，避免库默默截断长问题。
    from tokenizers import Tokenizer
    checker = Tokenizer.from_str(model.model.tokenizer.to_str())
    checker.no_truncation()
    checker.no_padding()
    if any(len(item.ids) > 512 for item in checker.encode_batch(inputs)):
        raise EmbeddingInputTooLong("文字超过本课向量模型的 512 token 上限；缩短问题或减小片段，避免静默截断。")
    return [v.tolist() for v in model.embed(inputs)]


def collection(index):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(Path(index).resolve()), settings=Settings(anonymized_telemetry=False))
    return client.get_collection("learning_materials", embedding_function=None)


def search(question, index, top_k=3):
    if not question.strip() or len(question) > 2000 or not 1 <= top_k <= 10:
        raise ValueError("问题需为 1–2000 字符，top_k 为 1–10。")
    index = Path(index)
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != MODEL or manifest["query_prefix"] != QUERY_PREFIX:
        raise ValueError("索引与查询向量模型配置不同，请建立新索引。")
    current = documents(ROOT / "data" / "source")
    if fingerprint(current) != manifest["documents_sha256"]:
        raise ValueError("原始资料已经变化，旧索引尚未更新。请用新的 --name 重新运行 build.py。")
    db = collection(index)
    result = db.query(query_embeddings=embed([question], is_query=True),
                      n_results=min(top_k, db.count()), include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": text, **meta, "cosine_distance": float(distance)}
            for i, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                              result["metadatas"][0], result["distances"][0])]


def context_text(hits):
    return "\n\n".join(f"[{h['id']}] 来源：{h['source']}，字符 {h['start']}:{h['end']}\n{h['text']}" for h in hits)
