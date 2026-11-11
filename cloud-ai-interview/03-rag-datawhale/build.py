"""一次入库；换参数时生成新目录，保留上一次实验。"""
import argparse
import json
import re
import shutil
from pathlib import Path
from knowledge import ROOT, MODEL, QUERY_PREFIX, documents, fingerprint, split_documents, embed


def build(name, size, overlap):
    import chromadb
    from chromadb.config import Settings
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("name 只使用字母、数字、下划线或短横线。")
    target = ROOT / "artifacts" / name
    if target.exists():
        raise ValueError("目录已存在，请换一个 --name，保留之前的实验。")
    docs = documents(ROOT / "data" / "source")
    chunks = split_documents(docs, size, overlap)
    print(f"读取 {len(docs)} 份原文 → 切出 {len(chunks)} 个片段", flush=True)
    vectors = embed([c["text"] for c in chunks])
    if not vectors or len(vectors) != len(chunks):
        raise ValueError("向量数量与片段不一致。")
    target.mkdir(parents=True)
    db = chromadb.PersistentClient(path=str(target), settings=Settings(anonymized_telemetry=False))
    coll = db.create_collection("learning_materials", embedding_function=None, metadata={"hnsw:space": "cosine"})
    coll.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks], embeddings=vectors,
             metadatas=[{"source": c["source"], "start": c["start"], "end": c["end"]} for c in chunks])
    (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "vector-preview.json").write_text(json.dumps({"model": MODEL, "shape": [len(vectors), len(vectors[0])],
        "first_chunk": chunks[0], "first_vector": vectors[0]}, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copytree(ROOT / "data" / "source", target / "source-snapshot")
    manifest = {"model": MODEL, "query_prefix": QUERY_PREFIX, "size": size, "overlap": overlap,
                "documents_sha256": fingerprint(docs), "documents": len(docs), "chunks": len(chunks),
                "dimension": len(vectors[0]), "embedding_source": "real_local_model"}
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存：{target}\n向量形状：{len(vectors)} × {len(vectors[0])}。先打开 chunks.json 和 vector-preview.json。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="first")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--overlap", type=int, default=40)
    args = p.parse_args()
    build(args.name, args.size, args.overlap)
